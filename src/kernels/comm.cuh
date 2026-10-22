#pragma once
#include "../common.h"

// Cross-GPU reduction without P2P: every GPU writes its partial into pinned, host-mapped
// memory and raises a per-GPU flag; peers poll the flag over PCIe and read the data.
// Everything happens inside kernels (no host involvement), so it can live in CUDA graphs.
//
// The sequence counter lives in device memory and is advanced by the last block of every
// collective, so replays of a captured graph stay in lock-step across GPUs.
constexpr int kCommSlots = 4;
constexpr int kCommBlocks = 8;  // more blocks polling host memory saturates PCIe reads

struct Comm {
    float* data = nullptr;      // host-mapped [ngpu][slots][max_elems]
    unsigned* flags = nullptr;  // host-mapped [ngpu][slots], one flag per GPU per slot
    int ngpu = 1;
    int max_elems = 0;
    // LL ("low latency") buffers for decode-sized all-reduces: every 8-byte word carries three
    // bf16 values plus a 16-bit tag of the collective, so a reader knows the data is there
    // without separate flags, fences or atomics. [ngpu][4 slots][kMaxBatchRows][hidden / 3]
    uint2* ll = nullptr;
    int ll_hidden = 0;
    // P2P (GPU peer access, e.g. a P2P-enabled driver): the LL16 words go straight into every peer's
    // receive buffer in its own memory, [4 slots][ngpu senders][kMaxBatchRows][hidden / 7] uint4,
    // and each GPU polls its local buffer; nullptr / p2p = false: the host-memory path above
    bool p2p = false;
    // P2P LL16 receive words are cleared after they are read (QE_LL_NOCLEAR=1: off, for the alias test)
    bool clear = true;
    // LL16 value mapping of the rounded-sum protocols: value e of word w is w + hw e (coalesced loads/stores)
    // instead of 7 w + e; changes the order of the norm's sum of squares (same for every m)
    bool coal = true;
    uint4* pll[4] = {nullptr, nullptr, nullptr, nullptr};
    // prefill reduce-scatter over P2P: rows sent to their owner [ngpu senders][rs_own][h] bf16 and
    // the owners' sums [rs_rows][h] bf16, both in the receiving GPU's memory
    uint2* prs1[4] = {nullptr, nullptr, nullptr, nullptr};
    uint2* prs2[4] = {nullptr, nullptr, nullptr, nullptr};
    int rs_rows = 0, rs_own = 0;
    // barrier flags for collectives whose data go over P2P: [ngpu senders][kCommSlots] in each
    // receiving GPU's memory, written over the same P2P path as the data (ordered with it)
    unsigned* pflags[4] = {nullptr, nullptr, nullptr, nullptr};
    // owner-finished prefill all-reduce (OwnFin): every GPU's receive rows of the normed outputs,
    // [rs_rows][h] bf16, [rs_rows][h] int8 (W4A8 input layout) and [rs_rows][rec] scale records
    uint2* pxn[4] = {nullptr, nullptr, nullptr, nullptr};
    int8_t* pxq[4] = {nullptr, nullptr, nullptr, nullptr};
    float* pxsc[4] = {nullptr, nullptr, nullptr, nullptr};
    int pxsc_row = 0;  // floats per row of pxsc (the G = 32 record size, the largest)
    // copy-engine prefill all-reduce (CeAr): flags written by the peers' copy streams,
    // [2 kinds: RS, AG][ngpu senders][kCeSlots] in each receiving GPU's memory, and the local bf16
    // staging rows [rs_rows][h] of the packed partials
    unsigned* cfl[4] = {nullptr, nullptr, nullptr, nullptr};
    int dev[4] = {0, 0, 0, 0};  // CUDA device of each rank
    uint2* pstage[4] = {nullptr, nullptr, nullptr, nullptr};
};
constexpr int kCeSlots = 16;

// Owner-finished deferred prefill all-reduce: the owner of a row (reduce-scatter) also adds the
// sum to ITS residual rows, applies the norm and writes the normed row - int8 for the W4A8 GEMM
// (group G > 0) and / or bf16 (wxn) - into the receive rows of every GPU (P2P), so no GPU runs a
// finishing kernel over all m rows. The residual is then valid only on the owner's rows: every
// collective of a chunk's micro-batch must use the same row count (owner ranges). G = 0: bf16 only.
struct OwnFin {
    int G = 0;
    bool wxn = false;
    bool v2 = false;  // gemm_i8v_kernel record layout (QE_PREFILL_INT8_KERNEL >= 2), else x_to_i8_kernel's
};

struct CommCounters {  // per GPU, device memory
    unsigned* seq = nullptr;
    unsigned* done = nullptr;    // [2] block arrival counters
    unsigned* go = nullptr;      // local broadcast: peers' data is ready for collective `go`
    unsigned* bar = nullptr;     // [2] grid barrier (arrive counter, release flag)
    float* ss = nullptr;         // [kCommBlocks][kMaxBatchRows] partial sums of squares
    unsigned* epoch = nullptr;   // LL: bumped once per launch unit (graph / eager forward)
    // early consumer (QE_EARLY_GEMV): when set, the LL16 all-reduce publishes
    // ready[idx] = epoch * 1024 + idx + 1 after all its rows are written ([1024..2047]: arrival counters)
    unsigned* ready = nullptr;
};

Comm comm_create(int ngpu, int max_elems, int ll_hidden);
// Enable peer access between all GPUs and allocate the P2P receive buffers (QE_P2P=0 or no peer
// access: stays on the host path). Returns c.p2p.
bool comm_enable_p2p(Comm& c, const int* devices);
void comm_free_p2p(Comm& c, const int* devices);
// Starts a launch unit for the LL protocol: LL collectives inside it are numbered 0, 1, ...
void comm_begin_unit(const CommCounters& ctr, cudaStream_t st);

// resid[m][h] += sum_g partial_g  (fixed summation order -> bitwise identical on all GPUs).
// If norm_w is given, also writes xn = bf16(rmsnorm(resid) * (1 + norm_w)) (fused next norm).
// ll_idx >= 0 (and m <= kMaxBatchRows) selects the LL protocol: ll_idx is the collective's
// index within the current launch unit (identical on every GPU).
// overlapped: large-m reduction running next to GEMMs (few blocks so it does not crowd them out)
// xh (optional, m <= kMaxBatchRows): the normed rows also in the XH layout of gemv_h16.cuh (stride h);
// with norm_w given, xn may be null when only the XH copy is consumed (decode kernels only)
// defer (prefill over P2P only, see allreduce_deferrable): the kernel stops after publishing the
// bf16 sums; allreduce_finish (same arithmetic) adds them to resid and applies the norm later on
// another stream with all SMs. rs_off: row offset of these m rows in the shared sums buffer.
void allreduce_add(float* resid, const float* partial, int m, int h, const Comm& c, int rank,
                   const CommCounters& ctr, cudaStream_t st, const bf16* norm_w = nullptr,
                   float eps = 0.f, bf16* xn = nullptr, int ll_idx = -1, bool overlapped = false,
                   __half* xh = nullptr, bool defer = false, int rs_off = 0, bool part_bf16 = false,
                   const OwnFin* of = nullptr);
bool allreduce_ownfin_ok(const Comm& c, int h);

// Copy-engine deferred prefill all-reduce: the rows move by DMA (cudaMemcpyAsync on the peers'
// copy streams, issued by a copy thread - comm.cu has no host code for it), only local work runs
// on SMs. SM stores over PCIe / P2P stall the memory system for every concurrent kernel (GEMM
// next to 4 blocks writing to host memory: +300..680 %); the copy
// engine does not (+0..1 %). Per collective with sequence number seq (identical on every GPU):
//   ce_pack (partials fp32 -> bf16 staging rows, local) | [copy thread: owner rows -> owners' prs1,
//   then flag RS] | ce_sum (owner: wait RS flags, sum in the fixed GPU order, bf16 sums -> own prs2
//   rows) | [copy thread: own sum rows -> peers' prs2, flag AG] | ce_wait (AG flags) -> rows of prs2
//   complete, allreduce_finish as for the deferred SM all-reduce (bit-identical sums).
bool ce_ar_ok(const Comm& c, int m, int h, int rs_off);
void ce_pack(const float* partial, int m, int h, const Comm& c, int rank, int rs_off, cudaStream_t st, int blocks);  // 0: m blocks
void ce_sum(const float* partial, bool part_bf16, int m, int h, const Comm& c, int rank, int rs_off,
            unsigned seq, cudaStream_t st);
void ce_wait(const Comm& c, int rank, unsigned seq, cudaStream_t st, int kind);  // kind 0: RS flags, 1: AG
void ce_flag(unsigned* p, unsigned v, cudaStream_t st);
// copy descriptors of one collective for the copy thread (bytes 0: no copy for that peer)
struct CeCopy {
    void* dst = nullptr;
    const void* src = nullptr;
    size_t bytes = 0;
    unsigned* flag = nullptr;  // in the peer's memory
    int dev = 0;               // the peer's CUDA device
};
// RS: this GPU's rows of owner o into o's prs1 (src: staging rows or the bf16 partial)
CeCopy ce_rs_copy(const Comm& c, int rank, int peer, int m, int h, int rs_off, const void* src_rows, unsigned seq);
// owner-finished CE all-reduce: finishing kernel (after the RS copies) and the AG copies of the owner's
// normed rows (int8 + records for G > 0, bf16 for G == 0 or wxn) into the peer's receive buffers; returns n <= 3
void ce_finish(float* resid, const float* partial, bool pb, int m, int h, const Comm& c, int rank, const bf16* norm_w,
               float eps, int rs_off, int G, bool wxn, cudaStream_t st);
int ce_of_copies(const Comm& c, int rank, int peer, int m, int h, int rs_off, int G, bool wxn, CeCopy* out);
// AG: this GPU's (owner's) sum rows into the peer's prs2
CeCopy ce_ag_copy(const Comm& c, int rank, int peer, int m, int h, int rs_off, unsigned seq);
void zero_int(int* p, cudaStream_t st);
// part_bf16 (deferred prefill rows only): partial holds bf16 rows [m][h] (GEMM output mode bf16)
bool allreduce_deferrable(const Comm& c, int m, int h, int rs_off);
// xq / xsc (optional): also quantize the normed rows for a W4A8 GEMM (group G, layouts of
// gemm.cu's x_to_i8_kernel); needs h % 2048 == 0
void allreduce_finish(float* resid, int m, int h, const Comm& c, int rank, const bf16* norm_w, float eps,
                      bf16* xn, int rs_off, cudaStream_t st, int8_t* xq = nullptr, float* xsc = nullptr,
                      int G = 0, int* ctr = nullptr, bool wxn = true);  // ctr: zeroed by the kernel (stream-K counter)
// (wxn = false with xq: the bf16 xn rows are not written, only the int8 copy for the next GEMM)

// out[g][0..n) = src of GPU g, n <= 4096 floats
void allgather_small(const float* src, int n, float* out, const Comm& c, int rank,
                     const CommCounters& ctr, cudaStream_t st);
