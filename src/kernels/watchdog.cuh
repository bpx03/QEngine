#pragma once
// Collective watchdog: every device-side wait on a peer GPU (or on another kernel / stream) is
// bounded. A wait counts its polls; every kWdMask + 1 polls it reads %globaltimer (the first check
// starts the clock). Past the budget (QE_COLL_TIMEOUT_MS, default 20 s; 0 = off) the waiting thread
// writes a record (site, rank, peer, expected / seen value) to host-mapped memory, raises the shared
// abort word and traps. Every other wait checks the abort word at its next check and traps too, so
// one stuck GPU stops the whole process within milliseconds. The host's CUDA_CHECK (common.h,
// qe_fatal_cuda) turns the trap into "collective timeout: GPUx ... GPUy not responding" and exits.
// The fast path is an increment and a masked compare per poll (the polls wait on memory anyway);
// the budget lives in device memory, so it works the same inside captured CUDA graphs.
#include "../common.h"

enum QeWdSite : unsigned {
    kWdNone = 0,
    kWdArExchange,      // allreduce_kernel (<= 32 rows, LL off): exchange() flags
    kWdArRowsExchange,  // allreduce_rows_kernel / allreduce_rs_rows_kernel (prefill SM path)
    kWdAllgather,       // allgather_kernel (top-k / spec pairs): exchange() flags
    kWdLL,              // allreduce_ll_kernel (8-byte LL words)
    kWdLL16,            // allreduce_ll16_kernel: peer partials
    kWdLL16Sum,         // allreduce_ll16_kernel MODE 2: the owner's sum
    kWdLLC,             // allreduce_llc_kernel (lossless packed)
    kWdLLQ,             // allreduce_llq_kernel (QE_AR_Q)
    kWdArFused,         // GEMV-fused LL16 all-reduce (arf::ll16_chunk)
    kWdCeFlags,         // copy-engine AR: ce_wait_kernel
    kWdEarlyGemv,       // gemv_w4h_early: the all-reduce's ready tag (same GPU, other stream)
    kWdKvDecode,        // kv position split: decode softmax-stat exchange flag
    kWdKvPrefill,       // kv position split: prefill softmax-stat exchange flag
    kWdPkp,             // attn_pkp_kernel work-queue dependency (same GPU)
    kWdSites
};

struct QeWdRec {             // one per rank, written by the first thread of that GPU that trips
    unsigned kind;           // 0 none, 1 timeout, 2 stopped by another GPU's abort
    unsigned site;           // QeWdSite
    int peer;                // GPU rank waited for (-1: unknown / same GPU)
    unsigned want, seen;     // expected and last seen flag / tag
    unsigned waited;         // ticks (2^20 ns) since the wait's first check
    unsigned block, thread;  // blockIdx.x, threadIdx.x
};
struct QeWdHost {            // host-mapped, shared by all GPUs
    volatile unsigned abort; // 0, else 1 + rank of the GPU that timed out (0xFF: host error)
    unsigned pad[7];
    QeWdRec rec[8];
};
struct QeWdDev {             // per device and translation unit (__device__ qe_wd_dev)
    QeWdHost* wd;            // device pointer of the host record (null: not initialized)
    unsigned budget;         // ticks; 0 = default, 0xFFFFFFFF = off
    int rank;
    unsigned claim;          // first tripping thread of this GPU writes the record
};
constexpr unsigned kWdMask = 0x3FFF;                 // check every 16K polls (>= 5 ms of polling)
constexpr unsigned kWdTickMask = (1u << 18) - 1u;  // 18-bit ticks: budgets up to 2^17 ticks (137 s)
constexpr unsigned kWdDefaultTicks = (unsigned)(20000000000ull >> 20);  // 20 s (kernels run without qe_wd_init)

// host (watchdog.cu)
void qe_wd_register(void (*set)(const QeWdDev&));  // each TU with device waits registers its symbol
void qe_wd_init(int rank, int device);              // current device: install the record, budget, rank
int qe_wd_budget_ms();                              // 0 = off
bool qe_fault_skip(int rank, const char* kind);      // fault injection (QE_FAULT_SKIP_AR_RANK), off by default
[[noreturn]] void qe_wd_host_timeout(const char* what);  // a host-side wait ran past the budget

#ifdef __CUDACC__
namespace {
__device__ QeWdDev qe_wd_dev;
inline void qe_wd_set_symbol(const QeWdDev& d) { CUDA_CHECK(cudaMemcpyToSymbol(qe_wd_dev, &d, sizeof d)); }
struct QeWdRegister {
    QeWdRegister() { qe_wd_register(qe_wd_set_symbol); }
};
QeWdRegister qe_wd_register_;

// noinline: inlined at every poll site its loops and stores bloated the LL kernels 1.7x and cost
// 1.5-2.5 us per collective at m >= 3 (ar_bench); the call sits only on the timeout branch
__device__ __noinline__ void qe_wd_trip(bool aborted, unsigned site, int peer, unsigned want, unsigned seen,
                                        unsigned waited) {
    QeWdHost* h = qe_wd_dev.wd;
    if (!h) return;
    volatile QeWdRec* r = &h->rec[qe_wd_dev.rank & 7];
    if (atomicCAS(&qe_wd_dev.claim, 0u, 1u) != 0u) {
        // another thread of this GPU writes the record: trapping now would kill its kernel first
        for (int i = 0; i < 100000 && r->kind == 0u; i++) __nanosleep(100);
        return;
    }
    r->site = site;
    r->peer = peer;
    r->want = want;
    r->seen = seen;
    r->waited = waited;
    r->block = blockIdx.x;
    r->thread = threadIdx.x;
    __threadfence_system();
    r->kind = aborted ? 2u : 1u;
    if (!aborted) h->abort = 1u + (unsigned)qe_wd_dev.rank;
    __threadfence_system();
    // the trap discards stores still in flight: read back until they reached host memory (a read
    // over PCIe does not pass earlier posted writes)
    for (int i = 0; i < 100000 && (r->kind == 0u || h->abort == 0u); i++) __nanosleep(100);
}

// One register of state per wait: bits 0..13 count polls, bits 14..31 hold the tick (2^20 ns) of the
// wait's first check (0: not started). Called when the poll count wraps (the carry is undone here).
// inline (a noinline check put an ABI call in every poll loop: 40-register cap + stack, +2 us per collective)
__device__ __forceinline__ unsigned qe_wd_check(unsigned ws, unsigned site, int peer, unsigned want, unsigned seen) {
    ws -= kWdMask + 1u;
    unsigned long long ns;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(ns));
    const unsigned now = (unsigned)(ns >> 20) & kWdTickMask;
    const unsigned t0 = ws >> 14;
    const unsigned budget = *reinterpret_cast<volatile unsigned*>(&qe_wd_dev.budget);
    if (t0 == 0u) return ws | ((now ? now : 1u) << 14);  // first check: the clock starts
    const unsigned waited = (now - t0) & kWdTickMask;
    QeWdHost* h = *reinterpret_cast<QeWdHost* volatile*>(&qe_wd_dev.wd);
    const bool ab = h && h->abort != 0u;
    if (ab || (budget != 0xFFFFFFFFu && waited > (budget ? budget : kWdDefaultTicks))) {
        qe_wd_trip(ab, site, peer, want, seen, waited);
        __trap();
    }
    return ws;
}
}  // namespace

// In a wait loop body: `unsigned ws = 0;` before the loop, QE_WD_POLL(ws, ...) per poll.
// `seen` is evaluated only at a check.
#define QE_WD_POLL(ws, site, peer, want, seen)                                              \
    do {                                                                                    \
        if (((++(ws)) & kWdMask) == 0u) (ws) = qe_wd_check((ws), (site), (peer), (want), (seen)); \
    } while (0)
#endif
