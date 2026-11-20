#pragma once
// DSV4 CPU expert lane: the cold routed experts of a decode step (rows R <= 8) on the host cores.
// [dsv4-rel3-isa] two builds of the kernels (dsv4_cpu_isa.cpp, compiled twice): AVX-512 (the Zen 4 set: VNNI, VBMI,
// BITALG, BF16) and AVX2 (x86-64-v3: AVX2, FMA, F16C), picked at start from cpuid (cpu_isa()); DSV4_CPU_ISA=avx2
// forces the AVX2 kernels. The AVX2 kernels keep the AVX-512 kernels' per-lane integer sums, float operations and
// reduction order, so both give the same bits. The rest of this unit (dsv4_cpu.cpp) is plain x86-64.
// Formats per expert: native FP4 e2m1 + e8m0 g32 (kernel of gate (a'), checkpoint bytes), cb3g128 / cb2g128
// (codebook through vpermb, fp16 scale per 128). Activations: FP8-rounded fp32 from the GPU, int8 per row here.
#include <atomic>
#include <functional>
#include <memory>
#include <cstdint>
#include <thread>
#include <string>
#include <vector>

#include "dsv4.h"

namespace dsv4 {

// [dsv4-rel3-isa] 1 = AVX-512 kernels (the CPU has every extension of -march=znver4 and the OS saves the zmm state),
// 0 = AVX2 kernels (x86-64-v3), -1 = neither (the lane cannot run). DSV4_CPU_ISA=avx2 forces 0 (=avx512 asks for 1
// and falls back to 0 when the CPU lacks it). Read once.
int cpu_isa();
const char* cpu_isa_name(int isa);

struct CpuJob {
    const uint8_t* base;   // expert bytes (dsv4.h expert_views layout)
    uint8_t fmt;           // XFmt
    int row;               // step row
    float w;               // routing weight
    // [dsv4-rel-ram]: an expert with no host copy: its three GGUF matrices in place (the file's page cache, ggml block
    // layout); base = nat[0] then (unique per expert: the grouping and prefetch key)
    const uint8_t* nat[3] = {nullptr, nullptr, nullptr};
};

class CpuLane {
public:
    // nthreads workers pinned to cpus first_cpu .. (the caller's thread takes part as worker 0)
    void start(int nthreads, int first_cpu);
    void stop();
    // codebooks (dsv4_quant.py values) for cb3 / cb2: snapped to the int8 grid here
    void set_codebooks(const float* cb3, const float* cb2);
    // y [R][5120] = sum over jobs (job order) of w * expert(xq[row]); xq FP8-rounded
    void run(int R, const float* xq, int nj, const CpuJob* jobs, float* y);
    // R20 [dsv4-r20-xh16]: the next run's xq is bf16 (the FP8-rounded values exactly), [R][5120] uint16
    void set_x16(bool on) { x16_ = on; }
    // prefetch (decode, while the GPU computes the layer's routing): every thread loads the slices of these experts
    // that it computes in run() (same per-expert row split), so they sit in its core's L2 / its CCD's L3; stops
    // early once *stop == stopval (checked every 8 KB). The calling thread (worker 0) does its own share before
    // returning; the next run() waits for nothing (stale workers only read).
    void prefetch(int n, const CpuJob* xs, const volatile int* stop, int stopval);
    void clear_pf() { pf_bases_.clear(); }   // decode thread: a layer without a run (review r11 L5)
    long pf_lines = 0;        // statistics: 64 B lines touched by worker 0 (last prefetch)
    int threads() const { return nt_; }
    double last_ms[5] = {};   // phase times of the last run (worker 0)
    // R24 [dsv4-r24-laneprof]: DSV4_LANEPROF=1: per worker, its own compute ms per phase (before each barrier) and
    // its barrier wait, summed over runs; report = one line (and reset). Timing only, no effect on results
    std::string prof_report(bool reset);
    void set_steal(bool on) { steal_ = on; }   // R24 (tests): between runs only; the default comes from DSV4_LANESTEAL at start()
    std::atomic<long> stolen1{0}, stolen3{0};   // R24 [dsv4-r24-steal]: units taken from other threads' ranges
    // [dsv4-rel3-isa] kernels of this lane (cpu_isa() at start); set_isa between runs only (tests)
    int isa() const { return isa_; }
    void set_isa(int v) { isa_ = v == 1 && cpu_isa() == 1 ? 1 : 0; }   // (review rel3 L1)

private:
    void worker(int id);
    void work(int id, uint64_t g) { isa_ == 1 ? work_avx512(id, g) : work_avx2(id, g); }
    void work_avx512(int id, uint64_t g);   // dsv4_cpu_isa.cpp, one per build
    void work_avx2(int id, uint64_t g);
    int isa_ = 0;
public:
    struct alignas(64) Steal { std::atomic<uint64_t> fb{0}; };   // R24 [dsv4-r24-steal]: (front << 32) | back
private:
    bool steal_ = false;
    std::unique_ptr<Steal[]> st1_, st3_;
    struct alignas(64) Prof { double ph[5] = {}; double wait = 0; long n = 0; };
    std::vector<Prof> prof_;
    int nt_ = 0;
    std::vector<std::thread> th_;
    std::atomic<uint64_t> gen_{0}, bar_{0};   // 64-bit: no wrap (review L9)
    // [glm-idle] idle workers block on gen_ (futex) after their spin, counted here; every gen_ change goes through
    // bump_gen(), which wakes them. seq_cst on both sides (store-load), so no wake-up is lost
    std::atomic<int> sleepers_{0};
    void bump_gen(uint64_t g) {
        gen_.store(g, std::memory_order_seq_cst);
        if (sleepers_.load(std::memory_order_seq_cst) > 0) gen_.notify_all();
    }
    std::atomic<bool> quit_{false};
    // per generation (ring): 0 = run, 1 = prefetch; barrier base of a run
    uint8_t mode_[16] = {};
    uint64_t barbase_[16] = {};
    uint64_t runs_ = 0;
    // prefetch parameters (2-slot ring by generation parity)
    struct Pf { int n = 0; CpuJob xs[2 * MAXR * TOPK]; const volatile int* stop = nullptr; int stopval = 0; } pf_[2];
    void prefetch_work(int id, uint64_t g) { isa_ == 1 ? prefetch_work_avx512(id, g) : prefetch_work_avx2(id, g); }
    void prefetch_work_avx512(int id, uint64_t g);
    void prefetch_work_avx2(int id, uint64_t g);
    // job state
    int R_ = 0, nj_ = 0;
    const float* xq_ = nullptr;
    bool x16_ = false;
    const CpuJob* jobs_ = nullptr;
    // jobs grouped by expert (review r4 M7): rows of a pass that share an expert read its 16-row block from DRAM
    // once (the later jobs find it in L1 / L2); each job still runs the one-row kernel, so results are unchanged
    std::vector<int> gj_, gs_;
    // R11: experts prefetched since the last run (prefetch order): run() computes their groups first, so the lines
    // still sit in L3 when used (streaming the other experts from DRAM first evicted them); results unchanged
    // (per-job outputs, combined in job order)
    std::vector<const uint8_t*> pf_bases_;
    std::vector<int> gj2_, gs2_;
    float* y_ = nullptr;
    // buffers
    std::vector<int8_t> aq4_, aq_, hq_;
    std::vector<float> as32_, as128_, ascl_, gu_, hs_, hscl_, yj_;
    alignas(64) uint8_t lut3_[64] = {}, lut2_[64] = {};
    float eff3_ = 1.f, eff2_ = 1.f;
};

// [dsv4-pf2-cpu] prefill routed experts of few rows (<= 8) on the CPU from their host copies (gguf_pfcpu.h): the
// GPU's exact bf16 weights, FP8-rounded bf16 x, fp32 dots (order differs from the GPU), GLU + q_fp8 as the GPU.
// Persistent pinned workers; a run takes jobs in order (one worker per job), done(j) turns true when job j's rows
// are written.
class PfCpu {
public:
    struct Job {
        const uint8_t* base;   // expert bytes (engine layout)
        uint8_t fmt;           // XFmt (GGUF i-quant expert)
        int m;                 // rows (<= 8)
        const int* rows;       // token rows of xq
        const float* wt;       // routing weights
        float* y;              // [m][D] out
    };
    static bool fmt_ok(int f);
    void start(const std::vector<int>& cpus);
    void stop();
    bool started() const { return !th_.empty(); }
    int threads() const { return (int)th_.size(); }
    // xq [n][D] bf16 (host); prep() runs once on the first worker that starts (waits for xq / the output buffer),
    // the others wait for it
    // [dsv4-pf8-xqh] xe != nullptr: xq holds the packed rows instead (E4M3 codes [n][D] bytes, then xe [n][D / 32] the
    // power-of-2 block exponents), rebuilt to the same bf16 values in the gather
    void run(const uint16_t* xq, int nj, const Job* jobs, std::function<void()> prep, const int8_t* xe = nullptr);
    bool done(int j) const { return dn_[j].load(std::memory_order_acquire) != 0; }
    void wait_all();   // the run's jobs finished (worker loop exit)
    double busy_ms = 0;   // statistics: summed job ms over the workers (last runs)
    int isa() const { return isa_; }
    void set_isa(int v);   // [dsv4-rel3-isa] between runs only (tests)
private:
    int isa_ = 0;
    void worker(int id);
    std::vector<std::thread> th_;
    std::atomic<uint64_t> gen_{0};
    std::atomic<bool> quit_{false};
    std::atomic<int> next_{0}, fin_{0}, prep_st_{0};
    std::unique_ptr<std::atomic<uint8_t>[]> dn_;
    int cap_ = 0, nj_ = 0;
    const Job* jobs_ = nullptr;
    const uint16_t* xq_ = nullptr;
    const int8_t* xe_ = nullptr;   // [dsv4-pf8-xqh]
    std::function<void()> prep_;
    std::atomic<long> busy_ns_{0};
};

// [dsv4-rel3-isa] one prefill job (row gather + the expert) in each kernel build (dsv4_cpu_isa.cpp); scratch xg [8][D],
// h13 [8][2 INTER] floats, hq [8][INTER]
namespace cpu_avx512 {
void pf_job(const PfCpu::Job& J, const uint16_t* xq, const int8_t* xe, uint16_t* xg, float* h13, uint16_t* hq);
void pf_init();
}
namespace cpu_avx2 {
void pf_job(const PfCpu::Job& J, const uint16_t* xq, const int8_t* xe, uint16_t* xg, float* h13, uint16_t* hq);
void pf_init();
}

}  // namespace dsv4
