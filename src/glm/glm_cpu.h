#pragma once
// CPU lane: cold routed experts on the host cores (glm_cpu.cpp)
#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <new>
#include <thread>
#include <vector>

#include "glm.h"

namespace glm {

// review r20b-L6: 64-byte aligned buffers for the prefill lane's per-row arrays (outputs of neighbouring work items,
// written by different threads, never share a cache line through the 16-byte malloc offset)
template <class T>
struct Al64 {
    using value_type = T;
    Al64() = default;
    template <class U> Al64(const Al64<U>&) {}
    T* allocate(size_t n) {
        void* p = nullptr;
        if (posix_memalign(&p, 64, std::max<size_t>(64, n * sizeof(T)))) throw std::bad_alloc();
        return static_cast<T*>(p);
    }
    void deallocate(T* p, size_t) { free(p); }
    template <class U> bool operator==(const Al64<U>&) const { return true; }
    template <class U> bool operator!=(const Al64<U>&) const { return false; }
};

struct CpuLane {
    static constexpr int MAXE = 64, MR = 4;
    int nth = 0, cpu0 = -1;
    std::vector<int> cpus;  // round 11: GLM_LANE_CPUS ("1-14,16-18"): explicit CPU per thread (overrides cpu0 + id)
    std::vector<std::thread> th;
    // each counter on its own 128-byte pair of lines (22 threads on 4 CCDs RMW and spin on them)
    alignas(128) std::atomic<unsigned> gen{0};
    IdleGate gate;        // [glm-idle] idle workers block on gen after GLM_LANE_SPIN_US; every gen bump goes through it
    long spin_us = 20000, pf_spin_us = 1000000;  // GLM_LANE_SPIN_US / GLM_LANE_PF_SPIN_US (after a prefill-lane job)
    alignas(128) std::atomic<unsigned> done{0};
    alignas(128) std::atomic<unsigned> bar0{0};
    alignas(128) std::atomic<unsigned> bar{0};
    alignas(128) std::atomic<unsigned> bar2{0};
    alignas(128) std::atomic<unsigned> bar3{0};
    bool bar_touch = true;  // round 11: touch own upcoming tiles while waiting at the barriers (GLM_LANE_BTOUCH=0 off)
    alignas(128) size_t pf_chunk = 4096;  // prefetch abort poll interval in bytes (GLM_PF_CHUNK, power of two >= 64)
    const float* xin = nullptr;
    alignas(128) std::atomic<int> tA{0};
    alignas(128) std::atomic<int> tC{0};
    alignas(128) std::atomic<unsigned> bar4{0};  // round 20: prefill lane, before the per-row sums
    volatile bool stop = false;
    const HxLayout* hl = nullptr;
    // current job
    int ne = 0, nm = 0;
    const char* ex[MAXE] = {};
    int enr[MAXE] = {}, erow[MAXE][MR] = {};
    float ew[MAXE][MR] = {};
    float* y = nullptr;
    alignas(64) int32_t xe[MR][H / 8], xo[MR][H / 8], sx[MR][H / 128], re[MR][H / 8], ro[MR][H / 8], rsx[MR][H / 128];
    alignas(64) float xs[MR][H / 128], rs[MR][H / 128];
    std::vector<float> h = std::vector<float>((size_t)MAXE * MR * INTER);  // [expert][row slot][INTER] glu * weight
    // h (down input) quantized per 64-group (slices start at multiples of 64): 32 groups per row
    alignas(64) int32_t he[MAXE * MR * INTER / 8], ho[MAXE * MR * INTER / 8], hsx[MAXE * MR * 32];
    alignas(64) float hs[MAXE * MR * 32];
    alignas(64) int32_t hre[MAXE * MR * INTER / 8], hro[MAXE * MR * INTER / 8], hrsx[MAXE * MR * 32];
    alignas(64) float hrs[MAXE * MR * 32];
    // opt-in LOSSY down-row skipping (GLM_DSP = fraction of an expert's 8-row down blocks skipped, chosen by the
    // smallest max-over-rows |h| sum): skip[e][k] = 1 -> block k of expert e is neither read nor computed
    alignas(64) uint8_t dskip[MAXE][INTER / 8];

    // prefetch mode: touch the predicted experts' tiles in the computing threads' caches until *abort_word != abort_old
    int mode = 0, pn = 0;
    const char* pex[MAXE] = {};
    const volatile unsigned* pabort = nullptr;
    unsigned pabort0 = 0;
    std::atomic<unsigned> pacc{0};
    std::atomic<unsigned long long> pbytes{0};  // bytes touched by prefetches
    // round 9: work stealing in the gate/up and down phases (GLM_LANE_STEAL, default on): per-thread tile lists of the
    // static map, each consumed through its own counter by the owner first, then by the other threads
    bool steal = true;
    struct alignas(128) Ctr { std::atomic<int> n{0}; };
    Ctr sA[64], sC[64];
    std::vector<int> lA;
    int lA_n[64] = {}, lA_cap = 0;
    // round 9 timeline (GLM_PFSTAT): per-thread TSC stamps of the last job, accumulated per call (TSC ticks)
    bool tl_on = false;
    uint64_t ts[64][8] = {};
    uint64_t pf_end[64] = {};
    bool pf_ab[64] = {};
    double tl_acc[12] = {};  // calls, wake, quant x + bar, A span, A idle, quant h + bars, C span, C idle, tail, wall, -, -
    void prefetch(int n, const char* const* experts, const volatile unsigned* abort_word, unsigned abort_old);
    void wait();  // until the current job (prefetch) has stopped
    unsigned long long ekey(const char* p) const { return (unsigned long long)(uintptr_t)p / hl->stride; }
    void start(int nthreads, int first_cpu, const HxLayout* layout);
    // out[4096] = sum_i w[i] * expert_i(x) (experts: host copies in the HxLayout); blocks until done
    void run(const float* x, int n, const char* const* experts, const float* w, float* out);
    // m <= MR rows: out[m][4096]; expert i serves rows rows[i * MR + j] (j < nr[i]) with weights w[i * MR + j]
    // mid (round 15): called on the caller's thread after the lane threads were started, before waiting for them
    void run_rows(const float* x, int m, int n, const char* const* experts, const int* nr, const int* rows,
                  const float* w, float* out, void (*mid)(void*) = nullptr, void* mid_arg = nullptr);
    // round 20 ([glm-arch2-pflane]) prefill job: ne cold experts with few rows each; pair j in [off[e], off[e + 1])
    // serves row rows[j] of x [nx][4096] with weight w[j]; yp[j][4096] = w[j] * expert_e(x[rows[j]]) (same kernels and
    // activation quantization as a decode lane row; the caller sums the pairs per row). Blocks until done.
    // y [nx][4096] (optional): per row, the sum of its pairs in pair order (rows without pairs: 0)
    bool pf_reserve(int nx, int np);  // review r21 M1: the prefill job buffers at load (false: out of memory)
    void run_pf(const float* x, int nx, int ne, const char* const* experts, const int* off, const int* rows,
                const float* w, float* yp, float* y = nullptr);
    ~CpuLane();
    // [glm-rel-isa] kernel build: 1 = AVX-512 (the Zen 4 set), 0 = AVX2 (x86-64-v3); picked in start() from the CPU
    // (GLM_CPU_ISA=avx2 forces AVX2); set_isa (tests): 1 only when the CPU has the AVX-512 set
    int isa = 1;
    void set_isa(int v);

private:
    struct Pf {
        const float* x = nullptr; int nx = 0, ne = 0, np = 0;
        const char* const* ex = nullptr; const int *off = nullptr, *rows = nullptr; const float* w = nullptr;
        float* yp = nullptr;
        float* y = nullptr;
        std::vector<int> roff, rp, fill;  // per row: its pairs (CSR, increasing pair index)
        std::vector<int32_t, Al64<int32_t>> xe, xo, sx, re, ro, rsx;  // [nx][..]
        std::vector<float, Al64<float>> xs, rs;
        std::vector<float, Al64<float>> h;                           // [np][INTER]
        std::vector<int32_t, Al64<int32_t>> he, ho, hsx, hre, hro, hrsx;
        std::vector<float, Al64<float>> hs, hrs;
    } pf;
    void pf_work(int id);
    void worker(int id);
    void work(int id);
    void work_avx512(int id), work_avx2(int id), pf_work_avx512(int id), pf_work_avx2(int id);  // glm_cpu_isa.cpp
    void prefetch_work(int id);
    void barrier(std::atomic<unsigned>& b);
};

// [glm-rel-isa] CPU kernel build available here: 1 = AVX-512 set, 0 = AVX2 (x86-64-v3) only or GLM_CPU_ISA=avx2,
// -1 = neither (forced_unavail: GLM_CPU_ISA=avx512 asked on a CPU without the set)
int lane_cpu_isa(bool* forced_unavail = nullptr);

}  // namespace glm
