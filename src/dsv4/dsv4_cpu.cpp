// DSV4 CPU expert lane (see dsv4_cpu.h): threads, run / prefetch set-up, the prefill CPU workers and the kernel
// dispatch. [dsv4-rel3-isa] plain x86-64 code; the kernels are in dsv4_cpu_isa.cpp (one AVX-512 and one AVX2 build).
#include "dsv4_cpu.h"

#include <immintrin.h>
#include <pthread.h>
#include <sched.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "dsv4.h"

namespace dsv4 {

constexpr int HS = INTER / 32 + INTER / 128;   // per job: activation sums (<= per 32) then scales per 128 (= dsv4_cpu_isa.cpp)

// [dsv4-rel3-isa] the ISA extensions of the AVX-512 build's target pragma (dsv4_cpu_isa.cpp: the Zen 4 set; Sapphire
// Rapids and later have it too), checked through libgcc's cpuid table, which also requires the OS to save the AVX / zmm
// state (XCR0)
static bool has_znver4_set() {
    __builtin_cpu_init();
    return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("avx") && __builtin_cpu_supports("fma") &&
           __builtin_cpu_supports("f16c") && __builtin_cpu_supports("bmi") && __builtin_cpu_supports("bmi2") &&
           __builtin_cpu_supports("lzcnt") && __builtin_cpu_supports("movbe") && __builtin_cpu_supports("popcnt") &&
           __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512dq") &&
           __builtin_cpu_supports("avx512cd") && __builtin_cpu_supports("avx512bw") &&
           __builtin_cpu_supports("avx512vl") && __builtin_cpu_supports("avx512ifma") &&
           __builtin_cpu_supports("avx512vbmi") && __builtin_cpu_supports("avx512vbmi2") &&
           __builtin_cpu_supports("avx512vnni") && __builtin_cpu_supports("avx512bitalg") &&
           __builtin_cpu_supports("avx512vpopcntdq") && __builtin_cpu_supports("avx512bf16") &&
           __builtin_cpu_supports("gfni") && __builtin_cpu_supports("vaes") && __builtin_cpu_supports("vpclmulqdq");
}
static bool has_x86_64_v3() {
    __builtin_cpu_init();
    return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("avx") && __builtin_cpu_supports("fma") &&
           __builtin_cpu_supports("f16c") && __builtin_cpu_supports("bmi") && __builtin_cpu_supports("bmi2") &&
           __builtin_cpu_supports("lzcnt") && __builtin_cpu_supports("movbe") && __builtin_cpu_supports("popcnt") &&
           __builtin_cpu_supports("sse4.2") && __builtin_cpu_supports("ssse3");
}
int cpu_isa() {
    static const int v = [] {
        const char* e = getenv("DSV4_CPU_ISA");
        const bool v3 = has_x86_64_v3(), v4 = v3 && has_znver4_set();
        if (e && !strcmp(e, "avx2")) return v3 ? 0 : -1;
        if (e && *e && strcmp(e, "avx512") && strcmp(e, "auto"))
            fprintf(stderr, "DSV4_CPU_ISA=%s not known (avx2, avx512, auto): auto\n", e);
        return v4 ? 1 : v3 ? 0 : -1;
    }();
    return v;
}
const char* cpu_isa_name(int isa) { return isa == 1 ? "AVX-512" : isa == 0 ? "AVX2" : "none"; }

// ---------------------------------------------------------------- lane
void CpuLane::set_codebooks(const float* cb3, const float* cb2) {
    auto snap = [](const float* c, int n, uint8_t* lut, float* eff) {
        float m = 0.f;
        for (int i = 0; i < n; i++) m = std::max(m, fabsf(c[i]));
        for (int i = 0; i < 64; i++) lut[i] = 128;
        for (int i = 0; i < n; i++) lut[i] = (uint8_t)(128 + (int)lrintf(c[i] / m * 127.f));
        *eff = m / 127.f;
    };
    snap(cb3, 8, lut3_, &eff3_);
    snap(cb2, 4, lut2_, &eff2_);
}

void CpuLane::start(int nthreads, int first_cpu) {
    nt_ = nthreads;
    isa_ = cpu_isa();   // [dsv4-rel3-isa]
    if (isa_ < 0) {   // (review rel3 L1: never the AVX2 kernels on a CPU without AVX2)
        fprintf(stderr, "the DSV4 CPU lane needs an x86-64-v3 CPU (AVX2, FMA, F16C, BMI2)\n");
        abort();
    }
    const int MJ = MAXR * TOPK;
    aq4_.resize((size_t)MAXR * D);
    aq_.resize((size_t)MAXR * D);
    as32_.resize((size_t)MAXR * D / 32);
    as128_.resize((size_t)MAXR * D / 128);
    ascl_.resize((size_t)MAXR * D / 128);
    gu_.resize((size_t)MJ * 2 * INTER);
    hs_.resize((size_t)MJ * INTER);
    hq_.resize((size_t)MJ * INTER);
    hscl_.resize((size_t)MJ * HS);
    yj_.resize((size_t)MJ * D);
    prof_.assign(nt_, Prof{});
    steal_ = !getenv("DSV4_LANESTEAL") || atoi(getenv("DSV4_LANESTEAL")) != 0;   // R24 [dsv4-r24-steal]: default, =0 off
    st1_ = std::unique_ptr<Steal[]>(new Steal[nt_]);
    st3_ = std::unique_ptr<Steal[]>(new Steal[nt_]);
    for (int i = 1; i < nt_; i++) th_.emplace_back([this, i, first_cpu] {
        cpu_set_t cs;
        CPU_ZERO(&cs);
        CPU_SET(first_cpu + i, &cs);
        pthread_setaffinity_np(pthread_self(), sizeof(cs), &cs);
        worker(i);
    });
    cpu_set_t cs;
    CPU_ZERO(&cs);
    CPU_SET(first_cpu, &cs);
    pthread_setaffinity_np(pthread_self(), sizeof(cs), &cs);
}
void CpuLane::stop() {
    quit_ = true;
    gen_.fetch_add(1, std::memory_order_seq_cst);
    gen_.notify_all();
    for (auto& t : th_) t.join();
    th_.clear();
}
void CpuLane::worker(int id) {
    uint64_t seen = 0;
    while (true) {
        uint64_t g;
        int spins = 0;
        while ((g = gen_.load(std::memory_order_acquire)) == seen) {
            if (++spins < 200000) _mm_pause();
            else if (spins < 400000) std::this_thread::yield();
            else {   // idle (prefill, between requests): [glm-idle] block until the next job (was a 50 us sleep poll)
                sleepers_.fetch_add(1, std::memory_order_seq_cst);
                gen_.wait(seen, std::memory_order_seq_cst);
                sleepers_.fetch_sub(1, std::memory_order_seq_cst);
            }
        }
        seen = g;
        if (quit_) return;
        if (mode_[g & 15] == 1) prefetch_work(id, g);
        else work(id, g);
    }
}
void CpuLane::run(int R, const float* xq, int nj, const CpuJob* jobs, float* y) {
    R_ = R;
    nj_ = nj;
    xq_ = xq;
    jobs_ = jobs;
    y_ = y;
    gj_.clear();
    gs_.clear();
    for (int j = 0; j < nj; j++) {
        bool seen = false;
        for (int i = 0; i < j && !seen; i++) seen = jobs[i].base == jobs[j].base;
        if (seen) continue;
        gs_.push_back((int)gj_.size());
        for (int i = j; i < nj; i++)
            if (jobs[i].base == jobs[j].base) gj_.push_back(i);
    }
    gs_.push_back((int)gj_.size());
    static const bool noord = getenv("DSV4_NOPFORD") != nullptr;
    if (!pf_bases_.empty() && !noord && gs_.size() > 2) {   // R11: prefetched groups first, in prefetch order
        const int ng = (int)gs_.size() - 1;
        std::vector<std::pair<int, int>> key(ng);   // (prefetch rank or large, group)
        for (int q = 0; q < ng; q++) {
            int rk = 1 << 20;
            const uint8_t* b = jobs[gj_[gs_[q]]].base;
            for (int i = 0; i < (int)pf_bases_.size(); i++)
                if (pf_bases_[i] == b) { rk = i; break; }
            key[q] = {rk, q};
        }
        std::stable_sort(key.begin(), key.end(), [](auto& a, auto& b) { return a.first < b.first; });
        gj2_.clear();
        gs2_.clear();
        for (auto& kq : key) {
            gs2_.push_back((int)gj2_.size());
            for (int i = gs_[kq.second]; i < gs_[kq.second + 1]; i++) gj2_.push_back(gj_[i]);
        }
        gs2_.push_back((int)gj2_.size());
        gj_.swap(gj2_);
        gs_.swap(gs2_);
    }
    pf_bases_.clear();
    uint64_t g = gen_.load(std::memory_order_relaxed) + 1;
    mode_[g & 15] = 0;
    barbase_[g & 15] = runs_ * 5;
    runs_++;
    bump_gen(g);
    work(0, g);
}
void CpuLane::prefetch(int n, const CpuJob* xs, const volatile int* stop, int stopval) {
    uint64_t g = gen_.load(std::memory_order_relaxed) + 1;
    Pf& P = pf_[g & 1];
    P.n = std::min<int>(n, 2 * MAXR * TOPK);
    for (int i = 0; i < P.n; i++) P.xs[i] = xs[i];
    for (int i = 0; i < P.n; i++) {   // (R11) a re-prefetched expert keeps its first rank
        bool seen = false;
        for (const uint8_t* b : pf_bases_) seen |= b == xs[i].base;
        if (!seen) pf_bases_.push_back(xs[i].base);
    }
    P.stop = stop;
    P.stopval = stopval;
    mode_[g & 15] = 1;
    bump_gen(g);
    prefetch_work(0, g);
}

std::string CpuLane::prof_report(bool reset) {
    std::string o;
    if (prof_.empty() || prof_[0].n == 0) return o;
    char b[160];
    const double n = (double)prof_[0].n;
    snprintf(b, sizeof b, "%ld runs; per worker ms/run ph1 gate-up | ph3 down | wait:", prof_[0].n);
    o += b;
    for (int i = 0; i < nt_; i++) {
        snprintf(b, sizeof b, " %d:%.3f|%.3f|%.3f", i, prof_[i].ph[1] / n, prof_[i].ph[3] / n, prof_[i].wait / n);
        o += b;
    }
    snprintf(b, sizeof b, "; stolen units per run ph1 %.2f ph3 %.2f", stolen1 / n, stolen3 / n);
    o += b;
    if (reset) {
        for (auto& p : prof_) p = Prof{};
        stolen1 = stolen3 = 0;
    }
    return o;
}

// ---------------------------------------------------------------- [dsv4-pf2-cpu] prefill experts on the CPU
bool PfCpu::fmt_ok(int f) {   // every matrix an i-quant (gguf_pfcpu.h mat_fmt_ok)
    const XFmt x = (XFmt)f;
    return is_gg(x) && is_iq(mat_fmt(x, 0)) && is_iq(mat_fmt(x, 1)) && is_iq(mat_fmt(x, 2));
}
void PfCpu::start(const std::vector<int>& cpus) {
    isa_ = cpu_isa();   // [dsv4-rel3-isa]
    if (isa_ < 0) {   // (review rel3 L1)
        fprintf(stderr, "DSV4 CPU prefill workers need an x86-64-v3 CPU (AVX2, FMA, F16C, BMI2)\n");
        abort();
    }
    isa_ == 1 ? cpu_avx512::pf_init() : cpu_avx2::pf_init();
    for (size_t i = 0; i < cpus.size(); i++) th_.emplace_back([this, i, c = cpus[i]] {
        cpu_set_t cs;
        CPU_ZERO(&cs);
        CPU_SET(c, &cs);
        pthread_setaffinity_np(pthread_self(), sizeof(cs), &cs);
        worker((int)i);
    });
}
void PfCpu::set_isa(int v) {
    isa_ = v == 1 && cpu_isa() == 1 ? 1 : 0;   // (review rel3 L1: never above what the CPU has)
    isa_ == 1 ? cpu_avx512::pf_init() : cpu_avx2::pf_init();   // (each build has its own tables)
}
void PfCpu::stop() {
    quit_ = true;
    gen_.fetch_add(1, std::memory_order_release);
    gen_.notify_all();
    for (auto& t : th_) t.join();
    th_.clear();
}
void PfCpu::run(const uint16_t* xq, int nj, const Job* jobs, std::function<void()> prep, const int8_t* xe) {
    if (nj > cap_) {
        cap_ = std::max(nj, 2 * cap_);
        dn_.reset(new std::atomic<uint8_t>[cap_]);
    }
    for (int j = 0; j < nj; j++) dn_[j].store(0, std::memory_order_relaxed);
    xq_ = xq;
    xe_ = xe;
    jobs_ = jobs;
    nj_ = nj;
    prep_ = std::move(prep);
    next_.store(0, std::memory_order_relaxed);
    fin_.store(0, std::memory_order_relaxed);
    prep_st_.store(0, std::memory_order_relaxed);
    gen_.fetch_add(1, std::memory_order_release);
    gen_.notify_all();
}
void PfCpu::wait_all() {
    for (int sp = 0; fin_.load(std::memory_order_acquire) < (int)th_.size(); sp++)
        if (sp < 4096) _mm_pause(); else std::this_thread::yield();
    busy_ms = busy_ns_.exchange(0) / 1e6;
}
void PfCpu::worker(int id) {
    (void)id;
    std::vector<uint16_t> xg((size_t)8 * D), hq((size_t)8 * INTER);
    std::vector<float> h13((size_t)8 * 2 * INTER);
    uint64_t seen = 0;
    while (true) {
        uint64_t g;
        int spins = 0;
        while ((g = gen_.load(std::memory_order_acquire)) == seen) {
            if (++spins < 20000) _mm_pause();   // (a layer's next run comes within ~30 ms)
            else gen_.wait(seen, std::memory_order_acquire);   // review cpu H1: blocked (futex), never woken by a timer
        }
        seen = g;
        if (quit_) return;
        int ps = 0;
        if (prep_st_.compare_exchange_strong(ps, 1, std::memory_order_acq_rel)) {
            if (prep_) prep_();
            prep_st_.store(2, std::memory_order_release);
        }
        while (prep_st_.load(std::memory_order_acquire) != 2) _mm_pause();
        long ns = 0;
        for (int j; (j = next_.fetch_add(1, std::memory_order_relaxed)) < nj_;) {
            const auto t0 = std::chrono::steady_clock::now();
            const Job& J = jobs_[j];
            if (isa_ == 1) cpu_avx512::pf_job(J, xq_, xe_, xg.data(), h13.data(), hq.data());   // [dsv4-rel3-isa]
            else cpu_avx2::pf_job(J, xq_, xe_, xg.data(), h13.data(), hq.data());
            dn_[j].store(1, std::memory_order_release);
            ns += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count();
        }
        busy_ns_.fetch_add(ns, std::memory_order_relaxed);
        fin_.fetch_add(1, std::memory_order_acq_rel);
    }
}

}  // namespace dsv4
