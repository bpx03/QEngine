#include <algorithm>
// CPU lane of the GLM engine: cold routed experts computed from the pinned host copy (AutoRound INT4 g128 sym layout,
// qweight [K/8][N] int32 with nibble j = row 8k8 + j), int8 activations with a scale per 128-group, vpdpbusd
// (u8 weight nibbles x s8 activations), -8 * sum(x) zero-point correction per group. Our own kernel (ideas from
// bench/cpu_moe gate (a)). A pinned thread pool splits 32-column tiles; deterministic summation order.
// [glm-rel-isa] this unit is plain x86-64 (threads, pinning, packing, job plumbing, ISA dispatch); the kernels and the
// per-thread work / pf_work bodies are in glm_cpu_isa.cpp, built once with AVX-512 and once with AVX2 (x86-64-v3).
#include "glm_cpu.h"

#include <immintrin.h>
#include <pthread.h>
#include <sched.h>

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>

#include "../common.h"

namespace glm {

unsigned IdleGate::wait(const std::atomic<unsigned>& v, unsigned seen, long spin_us) {
    unsigned g = v.load(std::memory_order_acquire);
    if (g != seen) return g;
    const auto t0 = std::chrono::steady_clock::now();
    for (unsigned i = 1;; i++) {
        if ((g = v.load(std::memory_order_acquire)) != seen) return g;
        _mm_pause();
        if (spin_us >= 0 && (i & 255) == 0 &&
            std::chrono::steady_clock::now() - t0 >= std::chrono::microseconds(spin_us)) break;
    }
    for (;;) {
        sleepers.fetch_add(1, std::memory_order_seq_cst);
        v.wait(seen, std::memory_order_seq_cst);
        sleepers.fetch_sub(1, std::memory_order_seq_cst);
        if ((g = v.load(std::memory_order_acquire)) != seen) return g;
    }
}
void IdleGate::bump(std::atomic<unsigned>& v) {
    v.fetch_add(1, std::memory_order_seq_cst);
    if (sleepers.load(std::memory_order_seq_cst) > 0) v.notify_all();
}
void IdleGate::set(std::atomic<unsigned>& v, unsigned x) {
    v.store(x, std::memory_order_seq_cst);
    if (sleepers.load(std::memory_order_seq_cst) > 0) v.notify_all();
}
long idle_spin_us(const char* env, long def) {
    const char* e = getenv(env);
    if (!e || !*e) return def;
    char* end = nullptr;
    const long v = strtol(e, &end, 10);
    if (end == e || *end || v < -1) {  // review L3: a typo would give 0 (every job pays a wake)
        fprintf(stderr, "[glm] %s=%s not a number of us (-1 = spin forever): %ld\n", env, e, def);
        return def;
    }
    fprintf(stderr, "[glm] %s=%ld (default %ld)\n", env, v, def);
    return v;
}

// [glm-rel-isa] the ISA extensions of the AVX-512 build's target pragma (glm_cpu_isa.cpp: the Zen 4 set, what
// -march=znver4 gave before; Sapphire Rapids and later have it too), checked through libgcc's cpuid table, which also
// requires the OS to save the AVX / zmm state (XCR0)
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
static bool has_x86_64_v3() {  // the AVX2 build's pragma list
    __builtin_cpu_init();
    return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("avx") && __builtin_cpu_supports("fma") &&
           __builtin_cpu_supports("f16c") && __builtin_cpu_supports("bmi") && __builtin_cpu_supports("bmi2") &&
           __builtin_cpu_supports("lzcnt") && __builtin_cpu_supports("movbe") && __builtin_cpu_supports("popcnt");
}
int lane_cpu_isa(bool* forced_unavail) {
    static bool fu = false;
    static const int v = [] {
        const char* e = getenv("GLM_CPU_ISA");
        const bool v3 = has_x86_64_v3(), v4 = v3 && has_znver4_set();
        if (e && !strcmp(e, "avx2")) return v3 ? 0 : -1;
        if (e && !strcmp(e, "avx512") && !v4) fu = true;
        if (e && *e && strcmp(e, "avx2") && strcmp(e, "avx512") && strcmp(e, "auto"))
            fprintf(stderr, "[glm] GLM_CPU_ISA=%s not known (avx2, avx512, auto): auto\n", e);
        return v4 ? 1 : v3 ? 0 : -1;
    }();
    if (forced_unavail) *forced_unavail = fu;
    return v;
}
void CpuLane::set_isa(int v) { isa = v == 1 && lane_cpu_isa() == 1 ? 1 : 0; }  // never above what the CPU has
void CpuLane::work(int id) { isa == 1 ? work_avx512(id) : work_avx2(id); }
void CpuLane::pf_work(int id) { isa == 1 ? pf_work_avx512(id) : pf_work_avx2(id); }

void CpuLane::worker(int id) {
    if (cpu0 >= 0) {
        cpu_set_t cs;
        CPU_ZERO(&cs);
        CPU_SET(id < (int)cpus.size() ? cpus[id] : cpu0 + id, &cs);
        pthread_setaffinity_np(pthread_self(), sizeof cs, &cs);
    }
    unsigned seen = 0;
    int last = 0;
    for (;;) {
        // [glm-idle] spin, then block until the next job; after a prefill-lane job the longer pf window (review M1: a
        // prefill layer's GPU part can exceed the decode window, so the next layer's job would pay a wake per layer)
        seen = gate.wait(gen, seen, last == 2 ? pf_spin_us : spin_us);
        last = mode;
        if (stop) return;
        if (mode == 1) prefetch_work(id);
        else if (mode == 2) pf_work(id);
        else work(id);
        done.fetch_add(1, std::memory_order_acq_rel);
    }
}

// host slices are TILE-MAJOR (32-column tiles, each one contiguous block: sequential DRAM streams for the CPU lane):
// gq / uq [ni/32][512][32] i32, gs / us [ni/32][32][32] f16, dq [128][ni/8][32] i32 (tiles of 32 output columns),
// ds [128][ni/64][32] f16 (each 128-group's scale twice). The GPU slots keep the row-major SliceL layout (k_untile).
bool hx_fits_int3(const void* gq, const void* uq) {
    for (int m = 0; m < 2; m++) {
        const uint32_t* q = (const uint32_t*)(m ? uq : gq);
        for (size_t i = 0; i < (size_t)512 * INTER; i++) {
            const uint32_t w = q[i], lo = w & 0x0F0F0F0Fu, hi = (w >> 4) & 0x0F0F0F0Fu;
            // every nibble in 4..11 <=> (nibble + 4) in 8..15 for all, i.e. bit 3 set after adding 4 per nibble
            const uint32_t a = lo + 0x04040404u, b = hi + 0x04040404u;
            if (((a & b) & 0x08080808u) != 0x08080808u || ((a | b) & 0x10101010u)) return false;
        }
    }
    return true;
}

void hx_pack_expert(const HxLayout& L, char* dst, const void* gq, const void* gs, const void* uq, const void* us,
                    const void* dq, const void* ds, bool g3) {
    if (L.mixed) dst[L.flag_off] = g3 ? 1 : 0;
    for (int r = 0; r < L.ns; r++) {
        const SliceL& S = L.mixed && g3 ? L.sl3[r] : L.sl[r];
        const int ni = S.ni, e0 = L.e0[r];
        char* d = dst + L.off[r];
        for (int tc = 0; tc < ni / 32; tc++) {
            if (S.gu3) {  // INT3 planes (SliceL::gu3): source nibbles n = code + 8, code in -4..3 -> v = n - 4 in 0..7
                for (int m = 0; m < 2; m++) {
                    const uint32_t* q = (const uint32_t*)(m ? uq : gq);
                    uint32_t* o = (uint32_t*)(d + (m ? S.uq : S.gq)) + (size_t)tc * 128 * 3 * 32;
                    for (int kb = 0; kb < 128; kb++)
                        for (int cc = 0; cc < 32; cc++) {
                            uint32_t w0 = 0, w1 = 0, w2 = 0;
                            for (int t = 0; t < 4; t++) {
                                const uint32_t wd = q[(size_t)(4 * kb + t) * INTER + e0 + tc * 32 + cc];
                                for (int b = 0; b < 4; b++)
                                    for (int p = 0; p < 2; p++) {
                                        const int v = (int)((wd >> (4 * (2 * b + p))) & 15u) - 4;
                                        CHECK(v >= 0 && v <= 7, "gu3: gate/up code out of the INT3 range (%d)", v - 4);
                                        const int jl = 2 * (t & 1) + p, jh = 2 * t + p;
                                        (t < 2 ? w0 : w1) |= (uint32_t)(v & 3) << (8 * b + 2 * jl);
                                        w2 |= (uint32_t)(v >> 2) << (8 * b + jh);
                                    }
                            }
                            uint32_t* ob = o + (size_t)kb * 3 * 32 + cc;
                            ob[0] = w0; ob[32] = w1; ob[64] = w2;
                        }
                }
            } else
            for (int k8 = 0; k8 < 512; k8++) {
                memcpy(d + S.gq + (((size_t)tc * 512 + k8) * 32) * 4, (const char*)gq + ((size_t)k8 * INTER + e0 + tc * 32) * 4, 128);
                memcpy(d + S.uq + (((size_t)tc * 512 + k8) * 32) * 4, (const char*)uq + ((size_t)k8 * INTER + e0 + tc * 32) * 4, 128);
            }
            for (int g = 0; g < 32; g++) {
                memcpy(d + S.gs + (((size_t)tc * 32 + g) * 32) * 2, (const char*)gs + ((size_t)g * INTER + e0 + tc * 32) * 2, 64);
                memcpy(d + S.us + (((size_t)tc * 32 + g) * 32) * 2, (const char*)us + ((size_t)g * INTER + e0 + tc * 32) * 2, 64);
            }
        }
        for (int t = 0; t < H / 32; t++) {
            for (int k8 = 0; k8 < ni / 8; k8++)
                memcpy(d + S.dq + (((size_t)t * (ni / 8) + k8) * 32) * 4, (const char*)dq + ((size_t)(e0 / 8 + k8) * H + t * 32) * 4, 128);
            for (int g = 0; g < ni / 64; g++)
                memcpy(d + S.ds + (((size_t)t * (ni / 64) + g) * 32) * 2, (const char*)ds + ((size_t)((e0 + 64 * g) / 128) * H + t * 32) * 2, 64);
        }
    }
}

void CpuLane::start(int nthreads, int first_cpu, const HxLayout* layout) {
    {   // [glm-rel-isa] kernel build for this CPU (R4: no SIGILL on CPUs without AVX-512); the engine has no lane-off
        // path (cold experts always run here), so a CPU without x86-64-v3 is refused at load with a clear message
        bool fu = false;
        const int v = lane_cpu_isa(&fu);
        CHECK(v >= 0, "[glm] CPU lane: this CPU lacks AVX2 / FMA / F16C / BMI2 (x86-64-v3); the GLM engine computes cold "
                      "experts on the host cores and needs at least x86-64-v3");
        isa = v;
        static std::atomic<bool> logged{false};
        if (!logged.exchange(true))
            fprintf(stderr, "[glm] CPU lane: %s kernels%s\n", v == 1 ? "AVX-512" : "AVX2",
                    fu ? " (GLM_CPU_ISA=avx512: this CPU lacks part of the AVX-512 set)" : "");
    }
    hl = layout;
    nth = nthreads;
    cpu0 = first_cpu;
    if (const char* lc = getenv("GLM_LANE_CPUS")) {  // round 11: list of ranges; the thread count follows the list
        for (const char* q = lc; *q;) {
            const int a = atoi(q);
            while (*q && *q != ',' && *q != '-') q++;
            int b = a;
            if (*q == '-') { b = atoi(++q); while (*q && *q != ',') q++; }
            for (int c = a; c <= b; c++) cpus.push_back(c);
            if (*q) q++;
        }
        if (!cpus.empty()) nth = (int)cpus.size();
    }
    CHECK(nth >= 1 && nth <= 64, "CPU lane: %d threads (1..64; per-thread arrays are fixed)", nth);  // r12 (review C6)
    if (const char* pc = getenv("GLM_PF_CHUNK")) pf_chunk = std::max<size_t>(64, (size_t)atol(pc));
    if (const char* sv = getenv("GLM_LANE_STEAL")) steal = atoi(sv) != 0;
    if (const char* bt = getenv("GLM_LANE_BTOUCH")) bar_touch = atoi(bt) != 0;
    spin_us = idle_spin_us("GLM_LANE_SPIN_US");
    pf_spin_us = idle_spin_us("GLM_LANE_PF_SPIN_US", 1000000);

    done = nth;  // idle: no job running
    for (int i = 0; i < nth; i++) th.emplace_back([this, i] { worker(i); });
}
CpuLane::~CpuLane() {
    stop = true;
    gate.bump(gen);
    for (auto& t : th) t.join();
}

// phase 0: gate/up tiles (expert, 32 cols) -> h ; phase 1: quantize h ; phase 2: down tiles of 32 output cols
void CpuLane::barrier(std::atomic<unsigned>& b) {
    b.fetch_add(1, std::memory_order_acq_rel);
    while (b.load(std::memory_order_acquire) < (unsigned)nth) _mm_pause();
}


// touch the tiles this thread will compute for the experts (gate/up tiles by the static map, down tiles of its output
// columns), until the abort word changes
void CpuLane::prefetch_work(int id) {
    struct End { CpuLane* L; int id; ~End() { if (L->tl_on) { L->pf_end[id] = __rdtsc(); L->pf_ab[id] = *L->pabort != L->pabort0; } } } end_{this, id};
    unsigned acc = 0;
    // the abort word is polled every pf_chunk bytes (GLM_PF_CHUNK, default 4 KB): the coordinator's lane->wait() sits
    // on the critical path between the request and run_rows
    bool stop = false;
    auto touch = [&](const char* p, size_t n) {
        size_t i = 0;
        for (; i < n && !stop; i += 64) {
            acc += *(const volatile unsigned*)(p + i);
            if ((i & (pf_chunk - 1)) == pf_chunk - 64 && *pabort != pabort0) stop = true;
        }
        pbytes.fetch_add(i, std::memory_order_relaxed);
    };
    for (int e = 0; e < pn; e++) {
        const char* base = pex[e];
        for (int ci = 0; ci < INTER / 32; ci++) {
            if ((int)((ekey(base) * 7 + ci) % nth) != id) continue;
            if (*pabort != pabort0) { pacc += acc; return; }
            const int c = ci * 32;
            int r = 0;
            while (r + 1 < hl->ns && c >= hl->e0[r + 1]) r++;
            const SliceL& S = hl->lay(base, r);
            const char* b = base + hl->off[r];
            const size_t tc = (size_t)(c - hl->e0[r]) / 32;
            const size_t tb = S.gu3 ? (size_t)128 * 3 * 32 * 4 : (size_t)512 * 32 * 4;
            touch(b + S.gq + tc * tb, tb);
            touch(b + S.uq + tc * tb, tb);
            if (stop) { pacc += acc; return; }
        }
        for (int t = id; t < H / 32; t += nth) {
            if (*pabort != pabort0) { pacc += acc; return; }
            for (int r = 0; r < hl->ns; r++) {
                const SliceL& S = hl->lay(base, r);
                touch(base + hl->off[r] + S.dq + (size_t)t * (S.ni / 8) * 32 * 4, (size_t)(S.ni / 8) * 32 * 4);
                if (stop) { pacc += acc; return; }
            }
        }
    }
    pacc += acc;
}

void CpuLane::prefetch(int n, const char* const* experts, const volatile unsigned* abort_word, unsigned abort_old) {
    pn = std::min(n, MAXE);
    for (int i = 0; i < pn; i++) pex[i] = experts[i];
    pabort = abort_word;
    pabort0 = abort_old;
    mode = 1;
    done = 0;
    gate.bump(gen);
}
void CpuLane::wait() {
    while (done.load(std::memory_order_acquire) < (unsigned)nth) _mm_pause();
}

void CpuLane::run(const float* x, int n, const char* const* experts, const float* w, float* out) {
    int nr[MAXE], rows[MAXE * MR];
    float ws[MAXE * MR];
    for (int i = 0; i < n; i++) { nr[i] = 1; rows[i * MR] = 0; ws[i * MR] = w[i]; }
    run_rows(x, 1, n, experts, nr, rows, ws, out);
}

void CpuLane::run_rows(const float* x, int m, int n, const char* const* experts, const int* nr, const int* rows,
                       const float* w, float* out, void (*mid)(void*), void* mid_arg) {
    CHECK(n <= MAXE && m <= MR, "cpu lane: %d experts, %d rows", n, m);
    if (n == 0) { memset(out, 0, (size_t)m * H * 4); if (mid) mid(mid_arg); return; }
    xin = x;
    nm = m;
    ne = n;
    for (int i = 0; i < n; i++) {
        ex[i] = experts[i];
        enr[i] = nr[i];
        for (int j = 0; j < nr[i]; j++) { erow[i][j] = rows[i * MR + j]; ew[i][j] = w[i * MR + j]; }
    }
    y = out;
    if (steal) {
        const int nt = n * (INTER / 32);
        lA_cap = nt;
        if ((int)lA.size() < nth * nt) lA.resize((size_t)nth * nt);
        for (int i = 0; i < nth; i++) { lA_n[i] = 0; sA[i].n.store(0, std::memory_order_relaxed); sC[i].n.store(0, std::memory_order_relaxed); }
        for (int t = 0; t < nt; t++) {
            const int o = (int)((ekey(ex[t / (INTER / 32)]) * 7 + t % (INTER / 32)) % nth);
            lA[o * nt + lA_n[o]++] = t;
        }
    }
    tA = 0; tC = 0; bar0 = 0; bar = 0; bar2 = 0; bar3 = 0; done = 0;
    mode = 0;
    const uint64_t c0 = tl_on ? __rdtsc() : 0;
    gate.bump(gen);
    if (mid) mid(mid_arg);
    while (done.load(std::memory_order_acquire) < (unsigned)nth) _mm_pause();
    if (tl_on) {
        const uint64_t c1 = __rdtsc();
        uint64_t mn0 = ~0ull, mx1 = 0, mx2 = 0, mx3 = 0, mx4 = 0, mx5 = 0, mx6 = 0;
        double sa = 0, sc = 0;
        for (int i = 0; i < nth; i++) {
            mn0 = std::min(mn0, ts[i][0]); mx1 = std::max(mx1, ts[i][1]); mx2 = std::max(mx2, ts[i][2]);
            mx3 = std::max(mx3, ts[i][3]); mx4 = std::max(mx4, ts[i][4]); mx5 = std::max(mx5, ts[i][5]);
            mx6 = std::max(mx6, ts[i][6]);
            sa += (double)(ts[i][3] - ts[i][2]); sc += (double)(ts[i][6] - ts[i][5]);
        }
        tl_acc[0] += 1;
        tl_acc[1] += (double)(mn0 - c0);           // wake-up
        tl_acc[2] += (double)(mx2 - mn0);          // quantize x + barrier
        tl_acc[3] += (double)(mx3 - mx2);          // gate/up span (to the last thread)
        tl_acc[4] += (double)(mx3 - mx2) - sa / nth;  // mean thread idle in it (tail imbalance)
        tl_acc[5] += (double)(mx5 - mx3);          // barrier + quantize h + barriers
        tl_acc[6] += (double)(mx6 - mx5);          // down span
        tl_acc[7] += (double)(mx6 - mx5) - sc / nth;
        tl_acc[8] += (double)(c1 - mx6);           // last thread -> caller sees done
        tl_acc[9] += (double)(c1 - c0);
    }
}

bool CpuLane::pf_reserve(int nx, int np_) {
    Pf& P = pf;
    constexpr int GX = H / 128;
    const size_t x8 = (size_t)nx * (H / 8), xg = (size_t)nx * GX, np = (size_t)np_;
    try {
        for (auto* v : {&P.xe, &P.xo, &P.re, &P.ro}) v->resize(x8);
        for (auto* v : {&P.sx, &P.rsx}) v->resize(xg);
        for (auto* v : {&P.xs, &P.rs}) v->resize(xg);
        P.h.resize(np * INTER);
        for (auto* v : {&P.he, &P.ho, &P.hre, &P.hro}) v->resize(np * (INTER / 8));
        for (auto* v : {&P.hsx, &P.hrsx}) v->resize(np * 32);
        for (auto* v : {&P.hs, &P.hrs}) v->resize(np * 32);
        P.roff.reserve(nx + 1); P.rp.reserve(np); P.fill.reserve(nx + 1);
    } catch (const std::bad_alloc&) { return false; }
    return true;
}
void CpuLane::run_pf(const float* x, int nx, int ne_, const char* const* experts, const int* off, const int* rows,
                     const float* w, float* yp, float* y) {
    if (ne_ <= 0) return;
    if (done.load(std::memory_order_acquire) != (unsigned)nth) {  // a decode job / prefetch still running
        fprintf(stderr, "[glm] prefill lane: waiting for a running lane job (%u of %d threads done)\n", done.load(), nth);
        wait();
    }
    Pf& P = pf;
    P.x = x; P.nx = nx; P.ne = ne_; P.np = off[ne_]; P.ex = experts; P.off = off; P.rows = rows; P.w = w; P.yp = yp;
    auto fit = [](auto& v, size_t n) { if (v.size() < n) v.resize(n); };
    constexpr int GX = H / 128;
    fit(P.xe, (size_t)nx * (H / 8)); fit(P.xo, (size_t)nx * (H / 8)); fit(P.re, (size_t)nx * (H / 8)); fit(P.ro, (size_t)nx * (H / 8));
    fit(P.sx, (size_t)nx * GX); fit(P.rsx, (size_t)nx * GX); fit(P.xs, (size_t)nx * GX); fit(P.rs, (size_t)nx * GX);
    const size_t np = P.np;
    fit(P.h, np * INTER);
    fit(P.he, np * (INTER / 8)); fit(P.ho, np * (INTER / 8)); fit(P.hre, np * (INTER / 8)); fit(P.hro, np * (INTER / 8));
    fit(P.hs, np * 32); fit(P.hsx, np * 32); fit(P.hrs, np * 32); fit(P.hrsx, np * 32);
    P.y = y;
    if (y) {
        P.roff.assign(nx + 1, 0);
        for (int j = 0; j < P.np; j++) P.roff[rows[j] + 1]++;
        for (int x = 0; x < nx; x++) P.roff[x + 1] += P.roff[x];
        P.rp.resize(P.np);
        P.fill.assign(P.roff.begin(), P.roff.end() - 1);
        for (int j = 0; j < P.np; j++) P.rp[P.fill[rows[j]]++] = j;
    }
    tA = 0; tC = 0; bar0 = 0; bar = 0; bar2 = 0; bar4 = 0; done = 0;
    mode = 2;
    gate.bump(gen);
    while (done.load(std::memory_order_acquire) < (unsigned)nth) _mm_pause();
    mode = 0;
}
}  // namespace glm
