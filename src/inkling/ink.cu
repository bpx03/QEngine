// Inkling 2 Mini (thinkingmachines/Inkling-2-Mini) on ONE RTX 3090: text-only engine, official checkpoint precision.
// Experts read bit-exact from the shipped NVFP4 (layers 0-19) / MXFP8 (20-23) bytes; everything else BF16 weights with
// FP32 math, FP32 residual stream, BF16 K/V cache. Step log:.
//
// Model (from the HF reference modeling_inkling2_loop.py):
//   24 layers, loop layers 6-17 run 3 passes (48 layer applications per token); block attention residuals over
//   blocks of 6 layers; passes 2-3 restart from the pre-loop state and mix in 4 messages per loop layer from the
//   previous pass (message gates); local layers sliding 1024, full layers 5/11/17/23; K/V short convs (dynamic taps on
//   full layers); post-attention / post-MoE short convs; relative position bias from 16-dim per-head states; per-token
//   sink logit; sigmoid output gate; MoE 128 experts top-8 sigmoid router, softcapped SwiGLU.
#include "ink.h"

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <array>
#include <functional>
#include <future>
#include <system_error>
#include <numeric>
#include <tuple>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "../json.h"
#include "ink_prefill.cuh"   // prefill kernels ([ink-pf-*])
#include "ink_tp.h"         // [ink-tp-plan] TP-N planner + SPMD runtime

namespace ink {
static std::chrono::steady_clock::time_point g_hts[8];   // [ink-n4-hosthop] INK_BSDBG host timestamps (rank 0)
static inline void hts(int i) { g_hts[i] = std::chrono::steady_clock::now(); }

[[noreturn]] void fail(const char* file, int line, const char* fmt, ...) {
    char b[1024];
    int n = snprintf(b, sizeof b, "inkling: ");
    va_list ap;
    va_start(ap, fmt);
    n += vsnprintf(b + n, sizeof b - n, fmt, ap);
    va_end(ap);
    if (n < (int)sizeof b) snprintf(b + n, sizeof b - n, " (%s:%d)", file, line);
    throw std::runtime_error(b);
}
#define ICHECK(c, ...)                                   \
    do {                                                 \
        if (!(c)) fail(__FILE__, __LINE__, __VA_ARGS__); \
    } while (0)
#define CU(x)                                                                                        \
    do {                                                                                             \
        cudaError_t e__ = (x);                                                                       \
        if (e__ != cudaSuccess) fail(__FILE__, __LINE__, "CUDA error: %s", cudaGetErrorString(e__)); \
    } while (0)
#define CB(x)                                                                                    \
    do {                                                                                         \
        cublasStatus_t s__ = (x);                                                                \
        if (s__ != CUBLAS_STATUS_SUCCESS) fail(__FILE__, __LINE__, "cuBLAS error %d", (int)s__); \
    } while (0)

typedef __nv_bfloat16 bf16;

// ---------------------------------------------------------------- safetensors (one file, mmapped)
struct Tens {
    const uint8_t* p = nullptr;
    size_t n = 0;
    std::string dtype;
    std::vector<long long> shape;
};
struct File {
    std::map<std::string, Tens> t;
    void* map = nullptr;
    size_t size = 0;
    void open(const std::string& path) {
        int fd = ::open(path.c_str(), O_RDONLY);
        ICHECK(fd >= 0, "cannot open %s", path.c_str());
        struct stat sb;
        ICHECK(fstat(fd, &sb) == 0, "fstat %s", path.c_str());
        size = sb.st_size;
        map = mmap(nullptr, size, PROT_READ, MAP_SHARED, fd, 0);
        ::close(fd);
        ICHECK(map != MAP_FAILED, "mmap %s", path.c_str());
        const uint8_t* b = (const uint8_t*)map;
        uint64_t hn;
        memcpy(&hn, b, 8);
        ICHECK(hn + 8 <= size, "bad header");
        Json h = Json::parse(std::string((const char*)b + 8, hn));
        const uint8_t* data = b + 8 + hn;
        for (auto& kv : h.obj) {
            if (kv.first == "__metadata__") continue;
            Tens x;
            x.dtype = kv.second["dtype"].str;
            for (auto& s : kv.second["shape"].arr) x.shape.push_back(s.i());
            long long o0 = kv.second["data_offsets"][0].i(), o1 = kv.second["data_offsets"][1].i();
            ICHECK(o0 >= 0 && o0 <= o1 && (uint64_t)o1 <= size - 8 - hn, "bad offsets %s", kv.first.c_str());
            x.p = data + o0;
            x.n = o1 - o0;
            t[kv.first] = x;
        }
    }
    const Tens& get(const std::string& k, const char* dt = nullptr, std::vector<long long> shape = {}) const {
        auto it = t.find(k);
        ICHECK(it != t.end(), "missing tensor %s", k.c_str());
        if (dt) ICHECK(it->second.dtype == dt, "%s: dtype %s, want %s", k.c_str(), it->second.dtype.c_str(), dt);
        if (!shape.empty()) ICHECK(it->second.shape == shape, "%s: unexpected shape", k.c_str());
        return it->second;
    }
};

static bool env_on(const char* k) {
    const char* v = getenv(k);
    return v && *v && *v != '0';
}
static float __uint_as_float_h(unsigned u) {
    float f;
    memcpy(&f, &u, 4);
    return f;
}
static int __float_as_int_h(float f) {
    int i;
    memcpy(&i, &f, 4);
    return i;
}
static float bf2f(uint16_t v) {
    uint32_t u = (uint32_t)v << 16;
    float f;
    memcpy(&f, &u, 4);
    return f;
}
static float e4m3_to_f(uint8_t v) {
    int s = v >> 7, e = (v >> 3) & 15, m = v & 7;
    float r;
    if (e == 15 && m == 7) r = NAN;
    else if (e == 0) r = ldexpf((float)m / 8.f, -6);
    else r = ldexpf(1.f + (float)m / 8.f, e - 7);
    return s ? -r : r;
}

// ---------------------------------------------------------------- G15 encoder (host; exact inverse = k_dec12)
// Port of tools/inkling_exp12/exp12_fmt.py encode() (branch inkling-exp12): a group = 2 rows (one warp of
// k_gemv12); per element a sign | mantissa byte and a 4-bit exponent index (exponent = row base + idx, idx 15 =
// escape; the best 15-wide window per row, first on ties); 16 B meta per group (bases, escape count, 6 escape
// entries lane | row << 5 | exp << 6); a group with > 6 escaping lane-rows or two exponents in one lane-row is a
// fallback group (its raw rows in the side store, slot in meta bytes 4..7).
struct Enc12 {
    std::vector<uint8_t> sm, meta;
    std::vector<uint32_t> idx;
    std::vector<uint16_t> fb;
    int G = 0, nfb = 0;
};
static void encode12(const uint16_t* A, int rows, int K, Enc12* o) {
    ICHECK(rows % 2 == 0 && K % 256 == 0, "encode12 shape");
    const int NC = K / 256, WD = 2 * NC, NF = WD / 4, TL = WD % 4, G = rows / 2;
    o->G = G;
    o->sm.assign((size_t)G * NC * 32 * 16, 0);
    o->idx.assign((size_t)G * 32 * WD, 0);
    o->meta.assign((size_t)G * 16, 0);
    std::vector<uint8_t> isfb(G, 0);
    auto work = [&](int g0, int g1) {
        for (int g = g0; g < g1; g++) {
            int base[2];
            for (int r = 0; r < 2; r++) {
                const uint16_t* row = A + (size_t)(2 * g + r) * K;
                int hist[256] = {0};
                for (int c = 0; c < K; c++) hist[(row[c] >> 7) & 0xff]++;
                int best = -1, bb = 0, win = 0;
                for (int e = 0; e < 15; e++) win += hist[e];
                for (int b = 0; b + 14 <= 255; b++) {
                    if (b > 0) win += hist[b + 14] - hist[b - 1];
                    if (win > best) {
                        best = win;
                        bb = b;
                    }
                }
                base[r] = bb;
            }
            int emin[64], emax[64];   // escapes per lane-row, key = r * 32 + lane
            for (int k = 0; k < 64; k++) emin[k] = 999, emax[k] = -1;
            uint8_t* sm = o->sm.data() + (size_t)g * NC * 32 * 16;
            uint32_t* ix = o->idx.data() + (size_t)g * 32 * WD;
            for (int r = 0; r < 2; r++) {
                const uint16_t* row = A + (size_t)(2 * g + r) * K;
                for (int c = 0; c < NC; c++)
                    for (int lane = 0; lane < 32; lane++) {
                        uint32_t word = 0;
                        for (int j = 0; j < 8; j++) {
                            const uint16_t v = row[(c * 32 + lane) * 8 + j];
                            const int e = (v >> 7) & 0xff, d = e - base[r];
                            const uint32_t id = (d < 0 || d > 14) ? 15u : (uint32_t)d;
                            if (id == 15) {
                                const int k = r * 32 + lane;
                                emin[k] = std::min(emin[k], e);
                                emax[k] = std::max(emax[k], e);
                            }
                            word |= id << ((j / 2) * 4 + (j % 2) * 16);
                            const int b = (r * NC + c) * 8 + j;   // the lane's byte stream, 16 B units unit-major
                            sm[((size_t)(b / 16) * 32 + lane) * 16 + b % 16] = (uint8_t)(((v >> 8) & 0x80) | (v & 0x7f));
                        }
                        const int wq = r * NC + c;
                        if (wq < NF * 4) ix[((wq / 4) * 32 + lane) * 4 + wq % 4] = word;
                        else ix[NF * 128 + lane * TL + (wq - NF * 4)] = word;
                    }
            }
            uint8_t* m = o->meta.data() + (size_t)g * 16;
            m[0] = (uint8_t)base[0];
            m[1] = (uint8_t)base[1];
            int n = 0;
            bool conflict = false;
            for (int k = 0; k < 64; k++)
                if (emax[k] >= 0) {
                    n++;
                    conflict |= emin[k] != emax[k];
                }
            if (conflict || n > 6) {
                isfb[g] = 1;
                m[2] = 255;
            } else {
                m[2] = (uint8_t)n;
                int s = 0;
                for (int k = 0; k < 64; k++)
                    if (emax[k] >= 0) {
                        const uint16_t ent = (uint16_t)((k & 31) | ((k >> 5) << 5) | (emin[k] << 6));
                        memcpy(m + 4 + 2 * s, &ent, 2);
                        s++;
                    }
            }
        }
    };
    const int nth = (int)std::max(1u, std::min(24u, std::thread::hardware_concurrency()));
    std::vector<std::thread> th;
    for (int i = 0; i < nth; i++) th.emplace_back(work, (int)((long long)G * i / nth), (int)((long long)G * (i + 1) / nth));
    for (auto& x : th) x.join();
    o->nfb = 0;
    o->fb.clear();
    for (int g = 0; g < G; g++)
        if (isfb[g]) {
            const uint32_t slot = o->nfb++;
            memcpy(o->meta.data() + (size_t)g * 16 + 4, &slot, 4);
            o->fb.insert(o->fb.end(), A + (size_t)2 * g * K, A + (size_t)(2 * g + 2) * K);
        }
}

// [ink-head2] host encoder of the head screen plane (branch inkling-exp12, exp12_host.h)
inline void exp12_encode_screen(const uint16_t* A, size_t rows, int K, uint8_t* plane, uint8_t* eb) {
    for (size_t r = 0; r < rows; r++) {
        const uint16_t* a = A + r * K;
        int emax = 0;
        for (int i = 0; i < K; i++) emax = std::max<int>(emax, (a[i] >> 7) & 0xff);
        if (emax < 15) throw std::runtime_error("screen: row top exponent < 15");
        const int b = emax - 14;
        eb[r] = (uint8_t)b;
        for (int i = 0; i < K; i++) {
            const int v = ((a[i] & 0x7fff) + 8) >> 4;
            int q = (v >> 3) - b, m = v & 7;
            if (q < 0) q = m = 0;
            plane[r * K + i] = (uint8_t)(((a[i] >> 8) & 0x80) | (q << 3) | m);
        }
    }
}

// ---------------------------------------------------------------- device kernels
__constant__ float c_e2m1[16];
__constant__ float c_e4m3[256];

// barrier over the threads of one phase body: the whole block in a kernel of its own; in the persistent step a
// 128-thread body runs on the first 128 threads of a 256-thread block (named barrier 1)
__device__ __forceinline__ void bsync_n(int) {   // every body runs as its own kernel with its own block size
    __syncthreads();
}
// one H-float shared buffer for every body that needs one (a __shared__ in one function = one allocation per kernel:
// the persistent step's bodies run one at a time and share it)
__device__ __forceinline__ float* xbuf_h() {
    __shared__ __align__(16) float buf[H];
    return buf;
}
__device__ __forceinline__ float* xbuf2_h() {   // a second one (the pair gate / up kernels)
    __shared__ __align__(16) float buf[H];
    return buf;
}
#define BSYNC(nt) bsync_n(nt)
__device__ __forceinline__ float warp_sum(float v) {
    for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
__device__ __forceinline__ float warp_max(float v) {
    for (int o = 16; o; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}
// block-wide sum of N values per thread (blockDim multiple of 32, <= 1024; N <= 32); red: >= 32*N + N floats.
// Warp sums, then lane i of warp 0 adds value i over the warps; everyone reads the N totals.
template <int N>
__device__ void block_sum(float (&v)[N], float* red) {
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5, nw = blockDim.x >> 5;
#pragma unroll
    for (int i = 0; i < N; i++) v[i] = warp_sum(v[i]);
    if (lane == 0)
#pragma unroll
        for (int i = 0; i < N; i++) red[w * N + i] = v[i];
    __syncthreads();
    if (w == 0 && lane < N) {
        float s = 0.f;
        for (int j = 0; j < nw; j++) s += red[j * N + lane];
        red[32 * N + lane] = s;
    }
    __syncthreads();
#pragma unroll
    for (int i = 0; i < N; i++) v[i] = red[32 * N + i];
    __syncthreads();
}

// [ink-bres-gram] transposed shuffle level: values [0, 2 Hh) -> [0, Hh), exchange with lane ^ O
template <int Hh, int O>
__device__ __forceinline__ void sumT_level(float* u, int lane) {
    const bool up = lane & O;
#pragma unroll
    for (int i = 0; i < Hh; i++) {
        const float snd = up ? u[i] : u[i + Hh], kp = up ? u[i + Hh] : u[i];
        u[i] = kp + __shfl_xor_sync(0xffffffffu, snd, O);
    }
}
// transposed block sum of N <= 32 values (P - 1 shuffles for P = pow2 >= N instead of 5 N per warp), totals only in
// warp 0 (lane i < N: value i); NW = warps (compile time)
template <int N, int NW>
__device__ __forceinline__ float block_sumW(const float (&v)[N], float* red) {
    constexpr int P = N <= 1 ? 1 : N <= 2 ? 2 : N <= 4 ? 4 : N <= 8 ? 8 : N <= 16 ? 16 : 32;
    constexpr int L = P == 1 ? 0 : P == 2 ? 1 : P == 4 ? 2 : P == 8 ? 3 : P == 16 ? 4 : 5;
    static_assert(N <= 32, "block_sumW: N <= 32");
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    float u[P];
#pragma unroll
    for (int i = 0; i < P; i++) u[i] = i < N ? v[i] : 0.f;
    if constexpr (L >= 1) sumT_level<P / 2, 16>(u, lane);
    if constexpr (L >= 2) sumT_level<P / 4, 8>(u, lane);
    if constexpr (L >= 3) sumT_level<P / 8, 4>(u, lane);
    if constexpr (L >= 4) sumT_level<P / 16, 2>(u, lane);
    if constexpr (L >= 5) sumT_level<P / 32, 1>(u, lane);
#pragma unroll
    for (int o = 16 >> L; o; o >>= 1) u[0] += __shfl_xor_sync(0xffffffffu, u[0], o);
    if ((lane & ((32 >> L) - 1)) == 0) red[w * P + (lane >> (5 - L))] = u[0];
    __syncthreads();
    float t = 0.f;
    if (w == 0 && lane < N)
#pragma unroll
        for (int j = 0; j < NW; j++) t += red[j * P + lane];
    return t;
}

// embedding rows (bf16) -> RMSNorm(embed_norm) -> fp32
struct A_embnorm {
    const bf16* rows;
    const bf16* w;
    float* out;
};
__device__ __forceinline__ void b_embnorm(const A_embnorm& A_, const dim3 bid_, const dim3 gdim_) {
    const auto& rows = A_.rows;
    const auto& w = A_.w;
    const auto& out = A_.out;

    __shared__ float red[33];
    const int t = bid_.x;
    float x[9], ss[1] = {0.f};
#pragma unroll
    for (int i = 0; i < 9; i++) {
        x[i] = __bfloat162float(rows[(size_t)t * H + threadIdx.x + 256 * i]);
        ss[0] += x[i] * x[i];
    }
    block_sum<1>(ss, red);
    const float inv = rsqrtf(ss[0] / H + 1e-6f);
#pragma unroll
    for (int i = 0; i < 9; i++) {
        const int c = threadIdx.x + 256 * i;
        out[(size_t)t * H + c] = __bfloat162float(w[c]) * (x[i] * inv);
    }
}
__global__ void __launch_bounds__(256) k_embnorm(const __grid_constant__ A_embnorm A_) { b_embnorm(A_, blockIdx, gridDim); }

#include "ink_tpf.cuh"   // [ink-tpf-ppush] TP producer push + the receive-only collective


// Block attention residual (and the loop messages): per token, stacked = unweighted-RMSNorm(states[j]) for j < ns,
// out[q] = sum_j softmax_j(query[q] . stacked_j / sqrt(H)) stacked_j. ns <= 5, nq <= 4. 256 threads, 9 elements each.
// Folded extras (all optional):
//   ident: out[0] = s[0] (no attention; used to put the norm / router fold behind a message gate)
//   nw[q]: RMSNorm(weight nw[q]) of out[q] -> outb[q] (bf16); normf: out[q] holds the normed fp32 instead of raw
//   Wr / sc: router logits of the normed out[0]: grid.y = NE / RPB blocks, block y -> experts RPB y .., sigmoid -> sc
//   qls / ols: grid.y = layers; block y uses queries q[k] + y * qls and writes out[k] + y * ols (all loop messages)
__device__ __forceinline__ void block_topk(const float* sc, const float* bias, int t, int slot, int* ids, float* wts,
                                           bool publish, int* e_sh);
__device__ __forceinline__ void block_topk_rank(const float* sc, const float* bias, int* ids, float* wts, int* e_sh);
struct BresArgs {
    const float* s[5];
    const float* q[4];
    float* out[4];
    const bf16* nw[4];
    bf16* outb[4];

    int ns, nq, ident, normf;
    int alg;   // [ink-bres-gram] 1: one transposed reduction of dots + Gram, output norm from the Gram
    const bf16* Wr;
    float* sc;
    int qls;
    size_t ols;
    // router top-8 in the last router block (T = 1): ids / weights for the expert kernels; counter reset in-graph
    int* rids;
    float* rwts;
    const float* rbias;
    int* rcnt;
    // zg > 0 (T = 1, 256-thread form): the message-gate z rows folded in; grid.y = 8 zg, block y -> gate y / 8,
    // z rows 8 (y % 8) + warp; each warp's down-row weights are loaded before the residual math (in flight under it)
    int zg;
    unsigned* chk;       // alg precision check: [0] max rel. error (float bits), [1] min |o|^2 / sum c^2 |x|^2, [2..65] -log2 bins
    float* mno;          // alg, loop messages: squared norms [layer][T][4]
    int rpw;             // [ink-bres-rpw] router experts per warp (0 / 1: one; 2: grid.y = NE / (2 RPB))
    const float* mni;    // alg, zg: squared norms of this layer's messages [T][4] (gate g -> mni[t * 4 + mgi[g]])
    int mgi[3];
    const float* mm[3];
    const bf16* gdn[3];
    float* z;
    float* zx;           // [ink-b-bres2] phase A of row steps (grid T x 1, zg > 0): the normed [h, m] inputs of every gate
                         // -> zx[g][T][2 H] (the values the z dot reads), no z dot here (k_bresZ does it)
    int qone;            // [ink-audit-nq3split] NQ 3, alg: block y forms only output q = y (zg: q = y / 8, written by
                         // block 8 q); the dots / coefficients / fallback choice stay those of all queries (same bits)
};
#ifdef BRES_PROF
__device__ unsigned long long bp_acc[8][8], bp_n[8][8];
#endif
constexpr int BT = 768, EP = H / BT, RPB = 8;   // RPB <= 8: the persistent step runs k_bres with 8 warps   // k_bres: threads, elements per thread, router experts per block
struct A_bres {
    BresArgs a;
};
// ZL: [ink-vp-brzl] z form of row steps with the z row loaded at the dot (not before the residual math) and no router
// path: fewer live registers, so the (T, 24) grid runs 2 blocks / SM (one wave at 4-6 rows); the same loads and products
template <int NQ, int BTT = BT, bool PF = false, bool ZL = false>   // queries (a.nq == NQ; ident uses NQ 1), threads; PF: prefill form (no z / router paths: [ink-pf-bresp])
__device__ __forceinline__ void b_bres(const A_bres& A_, const dim3 bid_, const dim3 gdim_) {
    const auto& a = A_.a;
#ifdef BRES_PROF
    const long long t0_ = clock64();
    const int kind_ = (a.zg ? 4 : 0) + (a.Wr ? 2 : 0) + (a.qls ? 1 : 0);
#define BP(ph) if (threadIdx.x == 0 && gdim_.x == 1 && (bid_.y == 0 || (ph) == 4)) { atomicAdd(&bp_acc[kind_][ph], (unsigned long long)(clock64() - t0_)); if ((ph) == 0 || (ph) == 4) atomicAdd(&bp_n[kind_][ph], 1ull); }
#else
#define BP(ph)
#endif

    __shared__ float red[33 * 25];
    float* xs = xbuf_h();
    const int t = bid_.x, by = bid_.y;
    const size_t qo = a.qls ? (size_t)by * a.qls : 0, oo = a.qls ? by * a.ols : 0;
    const bool writer = PF || (a.Wr == nullptr && a.zg == 0) || (a.qone ? (by & 7) == 0 : by == 0);
    const int qs = a.qone ? (a.zg ? by >> 3 : by) : -1;   // [ink-audit-nq3split] the one output of this block (-1: all)
    uint4 wz[2 * H / 256];   // zg: this warp's z row (2 H bf16), 18 x 16 B per lane
    if constexpr (BTT == 256 && !PF && !ZL) if (a.zg && !a.zx) {   // only the 256-thread form carries the z path (review M1: spills)
        const int lane = threadIdx.x & 31, g = by >> 3, r = (by & 7) * 8 + (threadIdx.x >> 5);
        const bf16* row = a.gdn[g] + (size_t)r * 2 * H;
#pragma unroll
        for (int k = 0; k < 2 * H / 256; k++)
            asm volatile("ld.global.nc.v4.u32 {%0, %1, %2, %3}, [%4];"
                         : "=r"(wz[k].x), "=r"(wz[k].y), "=r"(wz[k].z), "=r"(wz[k].w)
                         : "l"(row + (k * 32 + lane) * 8));
    }
    // router rows loaded first (weights-first: in flight under the residual math); the router dot below is unchanged
    uint4 wr[H / 256];
    const bool wrf = !PF && !ZL && NQ != 4 && (BTT == BT || a.alg) && a.Wr && (threadIdx.x >> 5) < RPB;
    if (wrf) {
        const bf16* row = a.Wr + (size_t)(by * RPB * (a.rpw == 2 ? 2 : 1) + (threadIdx.x >> 5)) * H + (threadIdx.x & 31) * 8;
#pragma unroll
        for (int k = 0; k < H / 256; k++)
            asm volatile("ld.global.nc.v4.u32 {%0, %1, %2, %3}, [%4];" : "=r"(wr[k].x), "=r"(wr[k].y), "=r"(wr[k].z), "=r"(wr[k].w) : "l"(row + k * 256));
    }
    float o[NQ][(H / BTT)];
    float gss[NQ];   // alg: squared norms of the outputs from the Gram matrix
    float nwv[NQ][H / BTT];   // alg: norm weights loaded with the states
    bool gram = false;
    if (a.alg && !a.ident) {
        // dots q . x_j and the Gram matrix x_j . x_k (j <= k) in one transposed reduction; |o_q|^2 = c_q^T G c_q
        constexpr int NG = NQ == 4 ? 6 : 15, ND = 5 * NQ + NG;
        constexpr int NSX = NQ == 4 ? 2 : 5;   // review M3: NQ 4 (loop messages) has 2 states, pair (0, 1) only (host check in bres_go)
        float x[5][(H / BTT)];
        float d[ND];
#pragma unroll
        for (int k = 0; k < ND; k++) d[k] = 0.f;
#pragma unroll
        for (int j = 0; j < 5; j++)
            if (j < NSX && j < a.ns)
#pragma unroll
                for (int i = 0; i < (H / BTT); i++) x[j][i] = a.s[j][(size_t)t * H + threadIdx.x + BTT * i];
        if (NG == 15 && a.nw[0])
#pragma unroll
            for (int q = 0; q < NQ; q++)
                if (qs < 0 || q == qs)
#pragma unroll
                for (int i = 0; i < (H / BTT); i++) nwv[q][i] = __bfloat162float(a.nw[q][threadIdx.x + BTT * i]);
#pragma unroll
        for (int q = 0; q < NQ; q++)
#pragma unroll
            for (int i = 0; i < (H / BTT); i++) {
                const float qv = a.q[q][qo + threadIdx.x + BTT * i];
#pragma unroll
                for (int j = 0; j < 5; j++)
                    if (j < NSX && j < a.ns) d[q * 5 + j] += qv * x[j][i];
            }
        // Gram entries: diagonal first (k = 5 NQ + j), then the pairs j < k
#pragma unroll
        for (int j = 0; j < 5; j++)
            if (j < NSX && j < a.ns)
#pragma unroll
                for (int i = 0; i < (H / BTT); i++) d[5 * NQ + j] += x[j][i] * x[j][i];
        {
            int e = 5 * NQ + 5;
#pragma unroll
            for (int j = 0; j < 5; j++)
#pragma unroll
                for (int k = j + 1; k < 5; k++, e++)
                    if (e < ND && k < NSX && k < a.ns)
#pragma unroll
                        for (int i = 0; i < (H / BTT); i++) d[e] += x[j][i] * x[k][i];
        }
        BP(5)
        // totals in warp 0; its lanes q < NQ compute the mixing coefficients c_qj = softmax_j * inv_j and |o_q|^2 =
        // c_q^T G c_q once (not in every thread); one barrier hands them to the block
        __shared__ float cf[NQ * 5 + NQ];
        __shared__ float x2tot[5];
        __shared__ int gfb;
        const float tot = block_sumW<ND, BTT / 32>(d, red);
        BP(1)
        if (threadIdx.x < 32) {   // lane = 8 q + j: one (query, state) pair per lane, group reductions by shuffles
            const int lane = threadIdx.x, q = lane >> 3, j = lane & 7;
            const bool on = q < NQ && j < a.ns;
            const int qq = q < NQ ? q : 0, jj = j < 5 ? j : 0;
            const float dq = __shfl_sync(0xffffffffu, tot, qq * 5 + jj);
            const float gjj = __shfl_sync(0xffffffffu, tot, 5 * NQ + jj);
            const float inv = on ? rsqrtf(gjj / H + 1e-6f) : 0.f;
            const float lg = on ? dq * inv * (1.f / sqrtf((float)H)) : -INFINITY;
            float m = lg;
#pragma unroll
            for (int o = 1; o < 8; o <<= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
            float p = on ? expf(lg - m) : 0.f, l = p;
#pragma unroll
            for (int o = 1; o < 8; o <<= 1) l += __shfl_xor_sync(0xffffffffu, l, o);
            const float c = on ? p / l * inv : 0.f;
            float g = 0.f;
            {   // (G c)_j over the group, then c . (G c)
                float acc = c * gjj;
#pragma unroll
                for (int k = 0; k < 5; k++) {
                    const int lo = jj < k ? jj : k, hi2 = jj < k ? k : jj;
                    const int e = lo * 5 - lo * (lo + 1) / 2 + (hi2 - lo - 1);   // pair index of (lo, hi), lo < hi
                    const float gjk = __shfl_sync(0xffffffffu, tot, 5 * NQ + 5 + (k == jj || e >= NG - 5 ? 0 : e));
                    const float ck = __shfl_sync(0xffffffffu, c, (lane & ~7) + k);
                    if (k != jj && k < a.ns && e < NG - 5) acc += gjk * ck;
                }
                g = c * acc;
                float sp = c * c * gjj;   // sum_j c_j^2 |x_j|^2: the scale of the rounding error of g
#pragma unroll
                for (int o = 1; o < 8; o <<= 1) {
                    g += __shfl_xor_sync(0xffffffffu, g, o);
                    sp += __shfl_xor_sync(0xffffffffu, sp, o);
                }
                // review M1: strong cancellation (or a negative / NaN g) -> the direct sum below, block-uniform
                const bool bad = q < NQ && j == 0 && !(g >= 1e-3f * sp);
                const unsigned anyb = __any_sync(0xffffffffu, bad);
                if (lane == 0) gfb = anyb != 0;
            }
            if (q < NQ && j < 5) cf[q * 5 + j] = c;
            if (a.chk && q == 0 && j < 5) x2tot[j] = gjj;
            if (q < NQ && j == 0) {
                cf[NQ * 5 + q] = g;
                if (a.mno) a.mno[((size_t)by * gdim_.x + t) * 4 + q] = g;   // loop messages: |m|^2 for the gates
            }
        }
#ifndef BRES_PROF2
        BP(6)
#endif
        __syncthreads();
#ifndef BRES_PROF2
        BP(7)
#endif
#pragma unroll
        for (int q = 0; q < NQ; q++) {
            gss[q] = cf[NQ * 5 + q];
            if (qs >= 0 && q != qs) continue;   // [ink-audit-nq3split]
            float p[5];
#pragma unroll
            for (int j = 0; j < 5; j++) p[j] = cf[q * 5 + j];
#pragma unroll
            for (int i = 0; i < (H / BTT); i++) {
                float v = 0.f;
#pragma unroll
                for (int j = 0; j < 5; j++)
                    if (j < NSX && j < a.ns) v += p[j] * x[j][i];
                o[q][i] = v;
            }
        }
        if (gfb) {   // review M1 fallback: |o|^2 by the direct fp32 sum (rare; the loop-message norms too)
            float dss[NQ];
#pragma unroll
            for (int q = 0; q < NQ; q++) {
                dss[q] = 0.f;
                if (qs >= 0 && q != qs) continue;   // [ink-audit-nq3split] (a value's block sum does not depend on the others)
#pragma unroll
                for (int i = 0; i < (H / BTT); i++) dss[q] += o[q][i] * o[q][i];
            }
            block_sum<NQ>(dss, red);
#pragma unroll
            for (int q = 0; q < NQ; q++) gss[q] = dss[q];
            if (a.mno && threadIdx.x == 0)
                for (int q = 0; q < NQ; q++) a.mno[((size_t)by * gdim_.x + t) * 4 + q] = dss[q];
            if (a.chk && threadIdx.x == 0 && by == 0) atomicAdd(a.chk + 66, 1u);
        }
        if (a.chk) {   // [ink-bres-gram] precision check: Gram |o|^2 vs the direct fp32 sum (INK_BRES_CHK=1)
            float dss[NQ];
#pragma unroll
            for (int q = 0; q < NQ; q++) {
                dss[q] = 0.f;
#pragma unroll
                for (int i = 0; i < (H / BTT); i++) dss[q] += o[q][i] * o[q][i];
            }
            block_sum<NQ>(dss, red);
            if (threadIdx.x == 0 && by == 0)
                for (int q = 0; q < NQ; q++) {
                    float sp = 0.f;   // sum of c_j^2 |x_j|^2: cancellation ratio = |o|^2 / sp
                    for (int j = 0; j < a.ns; j++) sp += cf[q * 5 + j] * cf[q * 5 + j] * x2tot[j];
                    const float e = fabsf(gss[q] - dss[q]) / fmaxf(dss[q], 1e-30f), r = dss[q] / fmaxf(sp, 1e-30f);
                    atomicMax(a.chk, __float_as_uint(e));
                    atomicMin(a.chk + 1, __float_as_uint(r));
                    const int b = e > 0.f ? min(63, max(0, (int)floorf(-log2f(e)))) : 63;
                    atomicAdd(a.chk + 2 + b, 1u);
                }
        }
        gram = true;
    } else if (a.ident) {
#pragma unroll
        for (int i = 0; i < (H / BTT); i++) o[0][i] = a.s[0][(size_t)t * H + threadIdx.x + BTT * i];
    } else {
        // one reduction for the squared norms and the raw dots: q . (x_j inv_j) = (q . x_j) inv_j
        constexpr int ND = 5 * NQ + 5;
        float x[5][(H / BTT)];
        float d[ND];
#pragma unroll
        for (int k = 0; k < ND; k++) d[k] = 0.f;
#pragma unroll
        for (int j = 0; j < 5; j++)
            if (j < a.ns)
#pragma unroll
                for (int i = 0; i < (H / BTT); i++) {
                    x[j][i] = a.s[j][(size_t)t * H + threadIdx.x + BTT * i];
                    d[5 * NQ + j] += x[j][i] * x[j][i];
                }
#pragma unroll
        for (int q = 0; q < NQ; q++)
#pragma unroll
            for (int i = 0; i < (H / BTT); i++) {
                const float qv = a.q[q][qo + threadIdx.x + BTT * i];
#pragma unroll
                for (int j = 0; j < 5; j++)
                    if (j < a.ns) d[q * 5 + j] += qv * x[j][i];
            }
        BP(5)
        block_sum<ND>(d, red);
        BP(1)
        float inv[5];
#pragma unroll
        for (int j = 0; j < 5; j++) inv[j] = j < a.ns ? rsqrtf(d[5 * NQ + j] / H + 1e-6f) : 0.f;
        const float sc = 1.f / sqrtf((float)H);
#pragma unroll
        for (int q = 0; q < NQ; q++) {
            float m = -INFINITY;
#pragma unroll
            for (int j = 0; j < 5; j++)
                if (j < a.ns) m = fmaxf(m, d[q * 5 + j] * inv[j] * sc);
            float p[5], l = 0.f;
#pragma unroll
            for (int j = 0; j < 5; j++) {
                p[j] = j < a.ns ? expf(d[q * 5 + j] * inv[j] * sc - m) : 0.f;
                l += p[j];
            }
#pragma unroll
            for (int j = 0; j < 5; j++) p[j] = p[j] / l * inv[j];
#pragma unroll
            for (int i = 0; i < (H / BTT); i++) {
                float v = 0.f;
#pragma unroll
                for (int j = 0; j < 5; j++)
                    if (j < a.ns) v += p[j] * x[j][i];
                o[q][i] = v;
            }
        }
    }
    if (a.nw[0]) {
        float ss[NQ];
#pragma unroll
        for (int q = 0; q < NQ; q++) ss[q] = 0.f;
        if (gram) {
#pragma unroll
            for (int q = 0; q < NQ; q++) ss[q] = gss[q];
        } else {
#pragma unroll
            for (int q = 0; q < NQ; q++)
#pragma unroll
                for (int i = 0; i < (H / BTT); i++) ss[q] += o[q][i] * o[q][i];
            block_sum<NQ>(ss, red);
        }
        BP(2)
#pragma unroll
        for (int q = 0; q < NQ; q++) {
            if (qs >= 0 && q != qs) continue;   // [ink-audit-nq3split]
            const float inv = rsqrtf(ss[q] / H + 1e-6f);
#pragma unroll
            for (int i = 0; i < (H / BTT); i++) {
                const int c = threadIdx.x + BTT * i;
                const float v = (gram ? nwv[q][i] : __bfloat162float(a.nw[q][c])) * (o[q][i] * inv);
                if (writer) {
                    if (a.outb[q]) a.outb[q][(size_t)t * H + c] = __float2bfloat16(v);
                    if (a.out[q]) a.out[q][(size_t)t * H + c] = a.normf ? v : o[q][i];
                }
                if (q == 0 && !PF && a.Wr) xs[c] = v;
            }
        }
    } else if (writer) {
#pragma unroll
        for (int q = 0; q < NQ; q++)
            if (qs < 0 || q == qs)
#pragma unroll
                for (int i = 0; i < (H / BTT); i++) a.out[q][oo + (size_t)t * H + threadIdx.x + BTT * i] = o[q][i];
    }
    if constexpr (BTT == 256 && !PF) if (a.zg && a.zx) {   // [ink-b-bres2] phase A: the z inputs of every gate, same values
        const bool gz = gram && a.mni;
        for (int g = 0; g < a.zg; g++) {
            float hv[H / BTT], ss[2] = {0.f, 0.f}, mv[H / BTT];
#pragma unroll
            for (int i = 0; i < H / BTT; i++) {
                hv[i] = o[0][i];
#pragma unroll
                for (int q = 1; q < NQ; q++)
                    if (q == g) hv[i] = o[q][i];
                mv[i] = a.mm[g][(size_t)t * H + threadIdx.x + BTT * i];
                ss[0] += hv[i] * hv[i];
                ss[1] += mv[i] * mv[i];
            }
            if (gz) {
                float gq = gss[0];
#pragma unroll
                for (int q = 1; q < NQ; q++)
                    if (q == g) gq = gss[q];
                ss[0] = gq;
                ss[1] = a.mni[(size_t)t * 4 + a.mgi[g]];
            } else
                block_sum<2>(ss, red);
            const float ih = rsqrtf(ss[0] / H + 1e-6f), im = rsqrtf(ss[1] / H + 1e-6f);
            float* zr = a.zx + ((size_t)g * gdim_.x + t) * 2 * H;
#pragma unroll
            for (int i = 0; i < H / BTT; i++) {
                zr[threadIdx.x + BTT * i] = hv[i] * ih;
                zr[H + threadIdx.x + BTT * i] = mv[i] * im;
            }
        }
        BP(0)
        return;
    }
    if constexpr (BTT == 256 && !PF) if (a.zg) {
        const int g = by >> 3, lane = threadIdx.x & 31, r = (by & 7) * 8 + (threadIdx.x >> 5);
        __shared__ __align__(16) float xm2[H];
        float hv[H / BTT], ss[2] = {0.f, 0.f};
        const bool gz = gram && a.mni;   // both norms known: no reduction here
#pragma unroll
        for (int i = 0; i < H / BTT; i++) {
            hv[i] = o[0][i];
#pragma unroll
            for (int q = 1; q < NQ; q++)
                if (q == g) hv[i] = o[q][i];
            const float mv = a.mm[g][(size_t)t * H + threadIdx.x + BTT * i];
            xm2[threadIdx.x + BTT * i] = mv;
            ss[0] += hv[i] * hv[i];
            ss[1] += mv * mv;
        }
        if (gz) {
            float gq = gss[0];
#pragma unroll
            for (int q = 1; q < NQ; q++)
                if (q == g) gq = gss[q];
            ss[0] = gq;
            ss[1] = a.mni[(size_t)t * 4 + a.mgi[g]];
        } else
            block_sum<2>(ss, red);
        const float ih = rsqrtf(ss[0] / H + 1e-6f), im = rsqrtf(ss[1] / H + 1e-6f);
#pragma unroll
        for (int i = 0; i < H / BTT; i++) {
            xs[threadIdx.x + BTT * i] = hv[i] * ih;
            xm2[threadIdx.x + BTT * i] *= im;
        }
        if constexpr (ZL) {   // [ink-vp-brzl] this warp's z row now (the early-load form's addresses and values)
            const bf16* row = a.gdn[g] + (size_t)r * 2 * H;
#pragma unroll
            for (int k = 0; k < 2 * H / 256; k++)
                asm volatile("ld.global.nc.v4.u32 {%0, %1, %2, %3}, [%4];"
                             : "=r"(wz[k].x), "=r"(wz[k].y), "=r"(wz[k].z), "=r"(wz[k].w)
                             : "l"(row + (k * 32 + lane) * 8));
        }
        BSYNC(BTT);
        float s = 0.f;
#pragma unroll
        for (int k = 0; k < 2 * H / 256; k++) {
            const int c = (k * 32 + lane) * 8;
            const float* xv = c < H ? xs + c : xm2 + (c - H);
            const uint32_t ww[4] = {wz[k].x, wz[k].y, wz[k].z, wz[k].w};
#pragma unroll
            for (int j = 0; j < 4; j++)
                s += __uint_as_float(ww[j] << 16) * xv[2 * j] + __uint_as_float(ww[j] & 0xffff0000u) * xv[2 * j + 1];
        }
        s = warp_sum(s);
        if (lane == 0) a.z[((size_t)g * gdim_.x + t) * GR + r] = s;
        BP(3)
        BP(0)
        return;
    }
    if (!PF && !ZL && a.Wr) {
        BSYNC(BTT);
#ifdef BRES_PROF2
        BP(6)
#endif
        const int lane = threadIdx.x & 31, w = threadIdx.x >> 5, rpw = a.rpw == 2 ? 2 : 1;
        if (w < RPB)
            for (int k = 0; k < rpw; k++) {   // [ink-bres-rpw] the second expert of a warp: the same dot (row loaded here)
                const int e = by * RPB * rpw + k * RPB + w;
                const bf16* row = a.Wr + (size_t)e * H;
                float s = 0.f;
#pragma unroll
                for (int c = lane * 8; c < H; c += 256) {
                    const uint4 wv = (wrf && k == 0) ? wr[c / 256] : *(const uint4*)(row + c);
                    const float* xv = xs + c;
                    s += __uint_as_float(wv.x << 16) * xv[0] + __uint_as_float(wv.x & 0xffff0000u) * xv[1] +
                         __uint_as_float(wv.y << 16) * xv[2] + __uint_as_float(wv.y & 0xffff0000u) * xv[3] +
                         __uint_as_float(wv.z << 16) * xv[4] + __uint_as_float(wv.z & 0xffff0000u) * xv[5] +
                         __uint_as_float(wv.w << 16) * xv[6] + __uint_as_float(wv.w & 0xffff0000u) * xv[7];
                }
                s = warp_sum(s);
                if (lane == 0) a.sc[(size_t)t * NE + e] = 1.f / (1.f + expf(-s));
            }
        if (a.rids) {   // the last router block selects the experts
            __shared__ int lastr;
            __shared__ float scs[NE];
            __shared__ int esr[2];
#ifdef BRES_PROF2
            BP(7)
#endif
            __threadfence();
            BSYNC(BTT);
            if (threadIdx.x == 0) lastr = atomicAdd(a.rcnt + t, 1) == (int)gdim_.y - 1;
            BSYNC(BTT);
            BP(3)
            if (!lastr) { BP(0) return; }
            __threadfence();
            if (threadIdx.x < NE) scs[threadIdx.x] = __ldcg(a.sc + (size_t)t * NE + threadIdx.x);
            BSYNC(BTT);
            block_topk_rank(scs, a.rbias, a.rids + t * TOPK, a.rwts + t * TOPK, esr);
            if (threadIdx.x == 0) a.rcnt[t] = 0;
            BP(4)
        }
    }
    BP(0)
#undef BP
}
template <int NQ>
__global__ void __launch_bounds__(BT) k_bres(const __grid_constant__ A_bres A_) { b_bres<NQ>(A_, blockIdx, gridDim); }
template <int NQ>
#ifndef BRES_MB
#define BRES_MB 1
#endif
__global__ void __launch_bounds__(256, BRES_MB) k_bres256(const __grid_constant__ A_bres A_) { b_bres<NQ, 256>(A_, blockIdx, gridDim); }
template <int NQ>   // [ink-vp-brzl]
__global__ void __launch_bounds__(256, 2) k_bres256z(const __grid_constant__ A_bres A_) { b_bres<NQ, 256, false, true>(A_, blockIdx, gridDim); }

// [ink-b-bres2] phase B of row steps: the weight-row dots of k_bres256 over a tile of tokens, each warp's row held in
// registers for the whole tile (k_bres256 re-reads it and redoes the token's residual mixing in every block). The
// per-token arithmetic is the k_bres256 code: the same loads, the same products in the same order, warp_sum.
// k_bresZ: grid (T tiles, 8 zg), block y -> gate y / 8, z rows 8 (y % 8) + warp; inputs zx[g][T][2 H] from phase A.
#ifndef BRZ_MB
#define BRZ_MB 1
#endif
__global__ void __launch_bounds__(256, BRZ_MB) k_bresZ(const __grid_constant__ A_bres A_, int T, int TT) {
    const auto& a = A_.a;
    const int lane = threadIdx.x & 31, g = blockIdx.y >> 3, r = (blockIdx.y & 7) * 8 + (threadIdx.x >> 5);
    uint4 wz[2 * H / 256];
    const bf16* row = a.gdn[g] + (size_t)r * 2 * H;
#pragma unroll
    for (int k = 0; k < 2 * H / 256; k++)
        asm volatile("ld.global.nc.v4.u32 {%0, %1, %2, %3}, [%4];"
                     : "=r"(wz[k].x), "=r"(wz[k].y), "=r"(wz[k].z), "=r"(wz[k].w)
                     : "l"(row + (k * 32 + lane) * 8));
    const int t0 = blockIdx.x * TT, t1 = min(T, t0 + TT);
    __shared__ __align__(16) float xz[2][2 * H];   // the token's inputs, staged once per block (double buffer)
    auto stage = [&](int t, float* d) {
        const float4* src = (const float4*)(a.zx + ((size_t)g * T + t) * 2 * H);
        for (int i = threadIdx.x; i < 2 * H / 4; i += 256) ((float4*)d)[i] = __ldcg(src + i);
    };
    if (t0 < t1) stage(t0, xz[0]);
    for (int t = t0; t < t1; t++) {
        __syncthreads();
        const float* zr = xz[(t - t0) & 1];
        if (t + 1 < t1) stage(t + 1, xz[(t + 1 - t0) & 1]);
        float s = 0.f;
#pragma unroll
        for (int k = 0; k < 2 * H / 256; k++) {
            const int c = (k * 32 + lane) * 8;
            const float* xv = zr + c;
            const uint32_t ww[4] = {wz[k].x, wz[k].y, wz[k].z, wz[k].w};
#pragma unroll
            for (int j = 0; j < 4; j++)
                s += __uint_as_float(ww[j] << 16) * xv[2 * j] + __uint_as_float(ww[j] & 0xffff0000u) * xv[2 * j + 1];
        }
        s = warp_sum(s);
        if (lane == 0) a.z[((size_t)g * T + t) * GR + r] = s;
    }
}
// k_bresR: router of row steps from the normed fp32 input xr[T][H] (phase A = k_bres256 without Wr, out = XMF):
// grid (T tiles, NE / 8), warp -> expert 8 y + warp; the last block of a token tile selects the experts of its tokens.
__global__ void __launch_bounds__(256) k_bresR(const __grid_constant__ A_bres A_, const float* xr, int T, int TT) {
    const auto& a = A_.a;
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5, e = blockIdx.y * 8 + w;
    uint4 wr[H / 256];
    {
        const bf16* row = a.Wr + (size_t)e * H + lane * 8;
#pragma unroll
        for (int k = 0; k < H / 256; k++)
            asm volatile("ld.global.nc.v4.u32 {%0, %1, %2, %3}, [%4];" : "=r"(wr[k].x), "=r"(wr[k].y), "=r"(wr[k].z), "=r"(wr[k].w) : "l"(row + k * 256));
    }
    const int t0 = blockIdx.x * TT, t1 = min(T, t0 + TT);
    __shared__ __align__(16) float xq[2][H];   // the token's normed input, staged once per block (double buffer)
    auto stage = [&](int t, float* d) {
        const float4* src = (const float4*)(xr + (size_t)t * H);
        for (int i = threadIdx.x; i < H / 4; i += 256) ((float4*)d)[i] = __ldcg(src + i);
    };
    if (t0 < t1) stage(t0, xq[0]);
    for (int t = t0; t < t1; t++) {
        __syncthreads();
        const float* xt = xq[(t - t0) & 1];
        if (t + 1 < t1) stage(t + 1, xq[(t + 1 - t0) & 1]);
        float s = 0.f;
#pragma unroll
        for (int c = lane * 8; c < H; c += 256) {
            const uint4 wv = wr[c / 256];
            const float* xv = xt + c;
            s += __uint_as_float(wv.x << 16) * xv[0] + __uint_as_float(wv.x & 0xffff0000u) * xv[1] +
                 __uint_as_float(wv.y << 16) * xv[2] + __uint_as_float(wv.y & 0xffff0000u) * xv[3] +
                 __uint_as_float(wv.z << 16) * xv[4] + __uint_as_float(wv.z & 0xffff0000u) * xv[5] +
                 __uint_as_float(wv.w << 16) * xv[6] + __uint_as_float(wv.w & 0xffff0000u) * xv[7];
        }
        s = warp_sum(s);
        if (lane == 0) a.sc[(size_t)t * NE + e] = 1.f / (1.f + expf(-s));
    }
    if (!a.rids) return;
    __shared__ int lastr;
    __shared__ float scs[NE];
    __shared__ int esr[2];
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) lastr = atomicAdd(a.rcnt + blockIdx.x, 1) == (int)gridDim.y - 1;
    __syncthreads();
    if (!lastr) return;
    __threadfence();
    for (int t = t0; t < t1; t++) {
        if (threadIdx.x < NE) scs[threadIdx.x] = __ldcg(a.sc + (size_t)t * NE + threadIdx.x);
        __syncthreads();
        block_topk_rank(scs, a.rbias, a.rids + t * TOPK, a.rwts + t * TOPK, esr);
    }
    if (threadIdx.x == 0) a.rcnt[blockIdx.x] = 0;
}
// k_bresZ4 / k_bresR4: the same dots with the TT tokens of a tile staged together (dynamic smem TT x 2 H / H floats)
// and their per-token sums interleaved (independent chains: ILP), one barrier per tile; per-token order unchanged
template <int TT>
__global__ void __launch_bounds__(256, 1) k_bresZ4(const __grid_constant__ A_bres A_, int T) {
    const auto& a = A_.a;
    extern __shared__ __align__(16) float xz4[];   // [TT][2 H]
    const int lane = threadIdx.x & 31, g = blockIdx.y >> 3, r = (blockIdx.y & 7) * 8 + (threadIdx.x >> 5);
    uint4 wz[2 * H / 256];
    const bf16* row = a.gdn[g] + (size_t)r * 2 * H;
#pragma unroll
    for (int k = 0; k < 2 * H / 256; k++)
        asm volatile("ld.global.nc.v4.u32 {%0, %1, %2, %3}, [%4];"
                     : "=r"(wz[k].x), "=r"(wz[k].y), "=r"(wz[k].z), "=r"(wz[k].w)
                     : "l"(row + (k * 32 + lane) * 8));
    const int t0 = blockIdx.x * TT, nt = min(T - t0, TT);
    {
        const float4* src = (const float4*)(a.zx + ((size_t)g * T + t0) * 2 * H);
        for (int i = threadIdx.x; i < nt * 2 * H / 4; i += 256) {   // halves of each 8-float chunk in two planes
            const int u = i / (2 * H / 4), j = i % (2 * H / 4);
            ((float4*)xz4)[u * (2 * H / 4) + (j & 1) * (H / 4) + (j >> 1)] = __ldcg(src + i);
        }
    }
    __syncthreads();
    float s[TT];
#pragma unroll
    for (int u = 0; u < TT; u++) s[u] = 0.f;
#pragma unroll
    for (int k = 0; k < 2 * H / 256; k++) {
        const int c = (k * 32 + lane) * 8;
        const uint32_t ww[4] = {wz[k].x, wz[k].y, wz[k].z, wz[k].w};
#pragma unroll
        for (int u = 0; u < TT; u++) {
            const float4 xa = *(const float4*)(xz4 + u * 2 * H + c / 2), xb = *(const float4*)(xz4 + u * 2 * H + H + c / 2);
            const float xv[8] = {xa.x, xa.y, xa.z, xa.w, xb.x, xb.y, xb.z, xb.w};
#pragma unroll
            for (int j = 0; j < 4; j++)
                s[u] += __uint_as_float(ww[j] << 16) * xv[2 * j] + __uint_as_float(ww[j] & 0xffff0000u) * xv[2 * j + 1];
        }
    }
#pragma unroll
    for (int u = 0; u < TT; u++) {
        const float v = warp_sum(s[u]);
        if (lane == 0 && u < nt) a.z[((size_t)g * T + t0 + u) * GR + r] = v;
    }
}
template <int TT>
__global__ void __launch_bounds__(256) k_bresR4(const __grid_constant__ A_bres A_, const float* xr, int T) {
    const auto& a = A_.a;
    extern __shared__ __align__(16) float xq4[];   // [TT][H]
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5, e = blockIdx.y * 8 + w;
    uint4 wr[H / 256];
    {
        const bf16* row = a.Wr + (size_t)e * H + lane * 8;
#pragma unroll
        for (int k = 0; k < H / 256; k++)
            asm volatile("ld.global.nc.v4.u32 {%0, %1, %2, %3}, [%4];" : "=r"(wr[k].x), "=r"(wr[k].y), "=r"(wr[k].z), "=r"(wr[k].w) : "l"(row + k * 256));
    }
    const int t0 = blockIdx.x * TT, nt = min(T - t0, TT);
    {
        const float4* src = (const float4*)(xr + (size_t)t0 * H);
        for (int i = threadIdx.x; i < nt * H / 4; i += 256) {   // halves of each 8-float chunk in two planes
            const int u = i / (H / 4), j = i % (H / 4);
            ((float4*)xq4)[u * (H / 4) + (j & 1) * (H / 8) + (j >> 1)] = __ldcg(src + i);
        }
    }
    __syncthreads();
    float s[TT];
#pragma unroll
    for (int u = 0; u < TT; u++) s[u] = 0.f;
#pragma unroll
    for (int c = lane * 8; c < H; c += 256) {
        const uint4 wv = wr[c / 256];
#pragma unroll
        for (int u = 0; u < TT; u++) {
            const float4 xa = *(const float4*)(xq4 + u * H + c / 2), xb = *(const float4*)(xq4 + u * H + H / 2 + c / 2);
            const float xv[8] = {xa.x, xa.y, xa.z, xa.w, xb.x, xb.y, xb.z, xb.w};
            s[u] += __uint_as_float(wv.x << 16) * xv[0] + __uint_as_float(wv.x & 0xffff0000u) * xv[1] +
                    __uint_as_float(wv.y << 16) * xv[2] + __uint_as_float(wv.y & 0xffff0000u) * xv[3] +
                    __uint_as_float(wv.z << 16) * xv[4] + __uint_as_float(wv.z & 0xffff0000u) * xv[5] +
                    __uint_as_float(wv.w << 16) * xv[6] + __uint_as_float(wv.w & 0xffff0000u) * xv[7];
        }
    }
#pragma unroll
    for (int u = 0; u < TT; u++) {
        const float v = warp_sum(s[u]);
        if (lane == 0 && u < nt) a.sc[(size_t)(t0 + u) * NE + e] = 1.f / (1.f + expf(-v));
    }
}
// top-8 of row-step tokens as its own launch (one block per token, the k_bresR tail's code)
__global__ void __launch_bounds__(256) k_topkR(const float* sc, const float* rbias, int* rids, float* rwts) {
    __shared__ float scs[NE];
    __shared__ int esr[2];
    const int t = blockIdx.x;
    if (threadIdx.x < NE) scs[threadIdx.x] = __ldcg(sc + (size_t)t * NE + threadIdx.x);
    __syncthreads();
    block_topk_rank(scs, rbias, rids + t * TOPK, rwts + t * TOPK, esr);
}
template <int NQ>
__global__ void __launch_bounds__(256) k_bres256p(const __grid_constant__ A_bres A_) { b_bres<NQ, 256, true>(A_, blockIdx, gridDim); }


// message gates, up to 3 at once (grid.z): z = down([rms(h), rms(m)]) (rank 64), then
// out = (1 - tanh(F z)) h + tanh(G z) m. Two kernels so that one token spreads over many SMs.
struct MgArgs {
    const float* h[3];
    const float* m[3];
    const bf16* down[3];
    const bf16* f[3];
    const bf16* g[3];
    float* out[3];
};
// grid (GR/8, T, ng), 256 threads: warp -> one z row; z: [ng][T][GR]
struct A_mgz {
    MgArgs a;
    float* z;
};
__device__ __forceinline__ void b_mgz(const A_mgz& A_, const dim3 bid_, const dim3 gdim_) {
    const auto& a = A_.a;
    const auto& z = A_.z;

    __shared__ float red[66];
    const int t = bid_.y, gi = bid_.z, T = gdim_.y;
    const float* hs = a.h[gi] + (size_t)t * H;
    const float* ms = a.m[gi] + (size_t)t * H;
    float ss[2] = {0.f, 0.f};
    for (int c = threadIdx.x; c < H; c += 256) {
        ss[0] += hs[c] * hs[c];
        ss[1] += ms[c] * ms[c];
    }
    block_sum<2>(ss, red);
    const float ih = rsqrtf(ss[0] / H + 1e-6f), im = rsqrtf(ss[1] / H + 1e-6f);
    const int lane = threadIdx.x & 31, r = bid_.x * 8 + (threadIdx.x >> 5);
    const bf16* row = a.down[gi] + (size_t)r * 2 * H;
    float s0 = 0.f, s1 = 0.f;
    for (int c = lane * 8; c < H; c += 256) {
        const uint4 wa = *(const uint4*)(row + c), wb = *(const uint4*)(row + H + c);
        const float4 h0 = *(const float4*)(hs + c), h1 = *(const float4*)(hs + c + 4);
        const float4 m0 = *(const float4*)(ms + c), m1 = *(const float4*)(ms + c + 4);
        const float hv[8] = {h0.x, h0.y, h0.z, h0.w, h1.x, h1.y, h1.z, h1.w};
        const float mv[8] = {m0.x, m0.y, m0.z, m0.w, m1.x, m1.y, m1.z, m1.w};
        const uint32_t wv[4] = {wa.x, wa.y, wa.z, wa.w}, uv[4] = {wb.x, wb.y, wb.z, wb.w};
#pragma unroll
        for (int k = 0; k < 4; k++) {
            s0 += __uint_as_float(wv[k] << 16) * hv[2 * k] + __uint_as_float(wv[k] & 0xffff0000u) * hv[2 * k + 1];
            s1 += __uint_as_float(uv[k] << 16) * mv[2 * k] + __uint_as_float(uv[k] & 0xffff0000u) * mv[2 * k + 1];
        }
    }
    s0 = warp_sum(s0);
    s1 = warp_sum(s1);
    if (lane == 0) z[((size_t)gi * T + t) * GR + r] = s0 * ih + s1 * im;
}
__global__ void __launch_bounds__(256) k_mgz(const __grid_constant__ A_mgz A_) { b_mgz(A_, blockIdx, gridDim); }

// [ink-audit-mgor] passes 2-3, MLP side: norm + router + top-8 of MIG without the ident k_bres256 launch. Block b (256
// channels) of token t: partial router dots P[t][b][e] = sum_c Wr[e][c] (nw[c] o[c]) and ss[t][b] = sum_c o[c]^2; the
// token's last block (ticket): inv from the 9 partial ss in order, XMF = nw (o inv) (the ident form's formula), logits
// = inv x sum_b P[t][b][e] in order, sigmoid, top-8. Same math as the ident form up to fp32 order (set-2 gate). One
// device function for the plain step (folded into k_mgo) and row steps (k_rpart after k_mgoT): the same bits.
constexpr int RF_NB = H / 256;
struct RouteF {
    const bf16* Wr = nullptr;
    const bf16* nw;
    const float* bias;
    float* P;      // [MROW][RF_NB][NE]
    float* ss;     // [MROW][RF_NB]
    int* cnt;      // [MROW] tickets (reset by the last block)
    const float* mig;
    float* xmf;
    float* sc;
    int* ids;
    float* wts;
};
__device__ __forceinline__ void rf_block(const RouteF& r, int t, int b, float o) {
    __shared__ float xw[256];
    __shared__ float red[33];
    __shared__ int last;
    __shared__ float scs[NE];
    __shared__ int esr[2];
    const int c = b * 256 + threadIdx.x, lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    uint4 wv[NE / 8];   // this warp's 16 experts x this lane's 8 channels, loads first
#pragma unroll
    for (int i = 0; i < NE / 8; i++) wv[i] = __ldg((const uint4*)(r.Wr + (size_t)(w * (NE / 8) + i) * H + b * 256 + lane * 8));
    xw[threadIdx.x] = __bfloat162float(r.nw[c]) * o;
    float s1[1] = {o * o};
    block_sum<1>(s1, red);   // its barriers also publish xw
    float xv[8];
#pragma unroll
    for (int k = 0; k < 8; k++) xv[k] = xw[lane * 8 + k];
    float* P = r.P + ((size_t)t * RF_NB + b) * NE;
#pragma unroll
    for (int i = 0; i < NE / 8; i++) {
        const uint32_t ww[4] = {wv[i].x, wv[i].y, wv[i].z, wv[i].w};
        float d = 0.f;
#pragma unroll
        for (int k = 0; k < 4; k++) d += __uint_as_float(ww[k] << 16) * xv[2 * k] + __uint_as_float(ww[k] & 0xffff0000u) * xv[2 * k + 1];
        d = warp_sum(d);
        if (lane == 0) P[w * (NE / 8) + i] = d;
    }
    if (threadIdx.x == 0) r.ss[t * RF_NB + b] = s1[0];
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) last = atomicAdd(r.cnt + t, 1) == RF_NB - 1;
    __syncthreads();
    if (!last) return;
    __threadfence();
    float tot = 0.f;
#pragma unroll
    for (int bb = 0; bb < RF_NB; bb++) tot += __ldcg(r.ss + t * RF_NB + bb);
    const float inv = rsqrtf(tot / H + 1e-6f);
#pragma unroll
    for (int i = 0; i < RF_NB; i++) {
        const int cc = threadIdx.x + 256 * i;
        r.xmf[(size_t)t * H + cc] = __bfloat162float(r.nw[cc]) * (__ldcg(r.mig + (size_t)t * H + cc) * inv);
    }
    if (threadIdx.x < NE) {
        const float* Pt = r.P + (size_t)t * RF_NB * NE + threadIdx.x;
        float sm = 0.f;
#pragma unroll
        for (int bb = 0; bb < RF_NB; bb++) sm += __ldcg(Pt + bb * NE);
        const float v = 1.f / (1.f + expf(-(inv * sm)));
        scs[threadIdx.x] = v;
        r.sc[(size_t)t * NE + threadIdx.x] = v;
    }
    __syncthreads();
    block_topk_rank(scs, r.bias, r.ids + t * TOPK, r.wts + t * TOPK, esr);
    if (threadIdx.x == 0) r.cnt[t] = 0;
}
// row steps: grid (RF_NB, T) after k_mgoT
__global__ void __launch_bounds__(256) k_rpart(const __grid_constant__ RouteF r) {
    const int t = blockIdx.y, b = blockIdx.x;
    rf_block(r, t, b, r.mig[(size_t)t * H + b * 256 + threadIdx.x]);
}
// grid (H/256, T, ng), 256 threads: one output channel per thread
struct A_mgo {
    MgArgs a;
    const float* z;
    RouteF rf{};   // [ink-audit-mgor] Wr set (ng 1): the router of the output folded in
};
__device__ __forceinline__ void b_mgo(const A_mgo& A_, const dim3 bid_, const dim3 gdim_) {
    const auto& a = A_.a;
    const auto& z = A_.z;

    __shared__ float zs[GR];
    const int t = bid_.y, gi = bid_.z, T = gdim_.y;
    if (threadIdx.x < GR) zs[threadIdx.x] = z[((size_t)gi * T + t) * GR + threadIdx.x];
    BSYNC(256);
    const int c = bid_.x * 256 + threadIdx.x;
    const uint4* fr = (const uint4*)(a.f[gi] + (size_t)c * GR);
    const uint4* gr = (const uint4*)(a.g[gi] + (size_t)c * GR);
    float f = 0.f, g = 0.f;
#pragma unroll
    for (int j = 0; j < GR / 8; j++) {
        const uint4 fv = fr[j], gv = gr[j];
        const uint32_t fw[4] = {fv.x, fv.y, fv.z, fv.w}, gw[4] = {gv.x, gv.y, gv.z, gv.w};
#pragma unroll
        for (int k = 0; k < 4; k++) {
            f += __uint_as_float(fw[k] << 16) * zs[8 * j + 2 * k] + __uint_as_float(fw[k] & 0xffff0000u) * zs[8 * j + 2 * k + 1];
            g += __uint_as_float(gw[k] << 16) * zs[8 * j + 2 * k] + __uint_as_float(gw[k] & 0xffff0000u) * zs[8 * j + 2 * k + 1];
        }
    }
    const size_t i = (size_t)t * H + c;
    const float ov = (1.f - tanhf(f)) * a.h[gi][i] + tanhf(g) * a.m[gi][i];
    a.out[gi][i] = ov;
    if (A_.rf.Wr) rf_block(A_.rf, t, bid_.x, ov);   // [ink-audit-mgor]
}
__global__ void __launch_bounds__(256) k_mgo(const __grid_constant__ A_mgo A_) { b_mgo(A_, blockIdx, gridDim); }
// [ink-b-small] row steps: a block takes TB tokens of its 256 channels, the f / g rows read once (registers) instead of
// once per token; per (token, channel) the same arithmetic in the same order as b_mgo. grid (H/256, ceil(T/TB), ng)
template <int TB>
__global__ void __launch_bounds__(256) k_mgoT(const __grid_constant__ A_mgo A_, int T) {
    const auto& a = A_.a;
    __shared__ float zs[TB][GR];
    const int t0 = blockIdx.y * TB, gi = blockIdx.z, nt = min(TB, T - t0);
    for (int x = threadIdx.x; x < TB * GR; x += 256)
        if (x / GR < nt) zs[x / GR][x % GR] = A_.z[((size_t)gi * T + t0 + x / GR) * GR + x % GR];
    const int c = blockIdx.x * 256 + threadIdx.x;
    const uint4* fr = (const uint4*)(a.f[gi] + (size_t)c * GR);
    const uint4* gr = (const uint4*)(a.g[gi] + (size_t)c * GR);
    uint4 fv[GR / 8], gv[GR / 8];
#pragma unroll
    for (int j = 0; j < GR / 8; j++) {
        fv[j] = fr[j];
        gv[j] = gr[j];
    }
    BSYNC(256);
    for (int tt = 0; tt < nt; tt++) {
        const float* zt = zs[tt];
        float f = 0.f, g = 0.f;
#pragma unroll
        for (int j = 0; j < GR / 8; j++) {
            const uint32_t fw[4] = {fv[j].x, fv[j].y, fv[j].z, fv[j].w}, gw[4] = {gv[j].x, gv[j].y, gv[j].z, gv[j].w};
#pragma unroll
            for (int k = 0; k < 4; k++) {
                f += __uint_as_float(fw[k] << 16) * zt[8 * j + 2 * k] + __uint_as_float(fw[k] & 0xffff0000u) * zt[8 * j + 2 * k + 1];
                g += __uint_as_float(gw[k] << 16) * zt[8 * j + 2 * k] + __uint_as_float(gw[k] & 0xffff0000u) * zt[8 * j + 2 * k + 1];
            }
        }
        const size_t i = (size_t)(t0 + tt) * H + c;
        a.out[gi][i] = (1.f - tanhf(f)) * a.h[gi][i] + tanhf(g) * a.m[gi][i];
    }
}


// RMSNorm with weight: fp32 in -> bf16 out (+ optional fp32 out)
struct A_rmsnorm {
    const float* x;
    const bf16* w;
    bf16* ob;
    float* of;
};
__device__ __forceinline__ void b_rmsnorm(const A_rmsnorm& A_, const dim3 bid_, const dim3 gdim_) {
    const auto& x = A_.x;
    const auto& w = A_.w;
    const auto& ob = A_.ob;
    const auto& of = A_.of;

    __shared__ float red[33];
    const int t = bid_.x;
    float v[9], ss[1] = {0.f};
#pragma unroll
    for (int i = 0; i < 9; i++) {
        v[i] = x[(size_t)t * H + threadIdx.x + 256 * i];
        ss[0] += v[i] * v[i];
    }
    block_sum<1>(ss, red);
    const float inv = rsqrtf(ss[0] / H + 1e-6f);
#pragma unroll
    for (int i = 0; i < 9; i++) {
        const int c = threadIdx.x + 256 * i;
        const float o = __bfloat162float(w[c]) * (v[i] * inv);
        if (ob) ob[(size_t)t * H + c] = __float2bfloat16(o);
        if (of) of[(size_t)t * H + c] = o;
    }
}
__global__ void __launch_bounds__(256) k_rmsnorm(const __grid_constant__ A_rmsnorm A_) { b_rmsnorm(A_, blockIdx, gridDim); }


__device__ __forceinline__ float log_tau(int pos) {
    return 1.f + 0.1f * logf(fmaxf((float)(pos + 1) / 8192.f, 1.f));
}

// q / rel post: grid (T, NQ), 128 threads. q = clip_L2(scale * q, 22.63) [* tau]; r = clip_L2(rscale * r, 8) [* tau].
__global__ void __launch_bounds__(128) k_qpost(const float* P, int ldp, const float* qs, const float* rs, int full,
                                               const int* posp, float* Q, float* R, float* S, int h0 = 0, int dq = 0,
                                               int dr = 0, int ds = 0) {   // [ink-pf3-qkv] dX: compact-layout shifts
    __shared__ float red[8];
    const int t = blockIdx.x, h = h0 + blockIdx.y, i = threadIdx.x;   // [ink-tp-pf] h0: a TP rank's first q head
    const float* p = P + (size_t)t * ldp;
    const float tau = full ? log_tau(*posp + t) : 1.f;
    float q = p[OFF_Q - dq + h * HD + i] * qs[h * HD + i];
    float s = warp_sum(q * q);
    if ((i & 31) == 0) red[i >> 5] = s;
    __syncthreads();
    const float nrm = sqrtf(red[0] + red[1] + red[2] + red[3]);
    q = q * fminf(QK_CLIP / (nrm + 1e-6f), 1.f);
    Q[((size_t)t * NQ + h) * HD + i] = q * tau;
    if (i < 32) {
        float r = i < DREL ? p[OFF_R - dr + h * DREL + i] * rs[h * DREL + i] : 0.f;
        float rn = sqrtf(warp_sum(r * r));
        r = r * fminf(REL_CLIP / (rn + 1e-6f), 1.f);
        if (i < DREL) R[((size_t)t * NQ + h) * DREL + i] = r * tau;
        if (i == 0) S[(size_t)t * NQ + h] = p[OFF_S - ds + h];
    }
}

// K or V: short conv over the linear output (static 4 taps on local layers, static + per-token dynamic taps per KV head
// on full layers), residual, then K: L2 clip 22.63 / V: unweighted RMSNorm, per head -> bf16 cache. grid (T, NKV), 128.
// Y = P + offY (row stride ldp); dyn = P + offDyn ([NKV][4] per token) or -1; st [3][KVD] (oldest first);
// w: local [KVD][4] (conv1d, w[3] = current token), full [4][KVD] (static_w, lag-major, lag 0 = current token).
__global__ void __launch_bounds__(128) k_kv(const float* P, int ldp, int offY, int offDyn, const float* st,
                                            const float* w, int isK, bf16* cache, int cap, int ring, const int* posp, int h0 = 0) {
    __shared__ float red[4];
    const int t = blockIdx.x, h = h0 + blockIdx.y, c = h * HD + threadIdx.x;   // [ink-tp-pf] h0: a TP rank's first KV head
    float x[4];   // x[j] = input j tokens back
#pragma unroll
    for (int j = 0; j < 4; j++) x[j] = (t - j >= 0) ? P[(size_t)(t - j) * ldp + offY + c] : st[(3 + t - j) * KVD + c];
    float o;
    if (offDyn < 0) {
        o = w[c * 4 + 3] * x[0] + w[c * 4 + 2] * x[1] + w[c * 4 + 1] * x[2] + w[c * 4 + 0] * x[3];
        o = (o + x[0]) * (1.f / sqrtf(5.f));
    } else {
        const float* dy = P + (size_t)t * ldp + offDyn + h * 4;
        o = 0.f;
#pragma unroll
        for (int j = 0; j < 4; j++) o += (w[j * KVD + c] + dy[j]) * x[j];
        o = (o + x[0]) * (1.f / 3.f);
    }
    float s = warp_sum(o * o);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = s;
    __syncthreads();
    const float ssum = red[0] + red[1] + red[2] + red[3];
    if (isK) o = o * fminf(QK_CLIP / (sqrtf(ssum) + 1e-6f), 1.f);
    else o = o * rsqrtf(ssum / HD + 1e-6f);
    const int pos = *posp + t;
    const int slot = ring ? pos % cap : pos;
    cache[((size_t)h * cap + slot) * HD + threadIdx.x] = __float2bfloat16(o);
}

// conv state update: new state = last 3 of [state, X rows]. one thread per channel.
__global__ void k_cstate(const float* X, int ldx, int T, int C, float* st, int lds = 0) {   // lds: state row stride (0: C)
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    const int L = lds ? lds : C;
    float old[3] = {st[c], st[L + c], st[2 * L + c]};
    for (int j = 0; j < 3; j++) st[j * L + c] = (T + j < 3) ? old[T + j] : X[(size_t)(T + j - 3) * ldx + c];
}

// static depthwise conv over hidden (attn_sconv / mlp_sconv): out = add + (conv(x) + x) / sqrt(5). (T*C threads)
__global__ void k_conv(const float* X, int T, int C, const float* st, const float* w, const float* add, float* out) {
    const size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (size_t)T * C) return;
    const int t = idx / C, c = idx % C;
    float x[4];
#pragma unroll
    for (int j = 0; j < 4; j++) x[j] = (t - j >= 0) ? X[(size_t)(t - j) * C + c] : st[(3 + t - j) * C + c];
    float o = w[c * 4 + 3] * x[0] + w[c * 4 + 2] * x[1] + w[c * 4 + 1] * x[2] + w[c * 4 + 0] * x[3];
    o = (o + x[0]) * (1.f / sqrtf(5.f));
    out[idx] = (add ? add[idx] : 0.f) + o;
}

__global__ void k_add(const float* a, const float* b, float* o, size_t n) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) o[i] = a[i] + b[i];
}

// attention, split over the keys: grid (T, NQ, NS), 4 warps per block, a warp per key; online softmax per split.
// rel bias = R[t][h] . proj[:, d] for d = q - k < 1024 (0 beyond). Partials: pm / pl [T][NQ][NS], pa [..][HD].
struct A_attn_part {
    const float* Q;
    const float* R;
    const float* proj;
    const bf16* Kc;
    const bf16* Vc;
    int cap;
    int ring;
    int local;
    const int* posp;
    float* pm;
    float* pl;
    float* pa;
    // optional (T = 1): the last split block of a head merges (lane per split) and writes AO; counter per head
    int* cnt;
    const float* S;
    const float* P;
    int ldp;
    bf16* AO;
    int two;   // two keys per warp iteration
    float* AOf;   // optional fp32 copy of the gated output (precision checks)
};
__device__ __forceinline__ void b_attn_part(const A_attn_part& A_, const dim3 bid_, const dim3 gdim_) {
    const bool two_keys_ = A_.two;
    const auto& Q = A_.Q;
    const auto& R = A_.R;
    const auto& proj = A_.proj;
    const auto& Kc = A_.Kc;
    const auto& Vc = A_.Vc;
    const auto& cap = A_.cap;
    const auto& ring = A_.ring;
    const auto& local = A_.local;
    const auto& posp = A_.posp;
    const auto& pm = A_.pm;
    const auto& pl = A_.pl;
    const auto& pa = A_.pa;

    __shared__ float sm[4], sl[4], sacc[4][HD];
    const int t = bid_.x, h = bid_.y, z = bid_.z, NS = gdim_.z, kvh = h / (NQ / NKV);
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    const int qp = *posp + t;
    const int lo = local ? max(0, qp - (WIN - 1)) : 0;
    const int n = qp - lo + 1, ks = (n + NS - 1) / NS;
    const int k0 = lo + z * ks, k1 = min(qp, k0 + ks - 1);
    const float* qv = Q + ((size_t)t * NQ + h) * HD;
    const float4 q = *(const float4*)(qv + lane * 4);
    float r[DREL];
#pragma unroll
    for (int i = 0; i < DREL; i++) r[i] = R[((size_t)t * NQ + h) * DREL + i];
    const float scale = 1.f / sqrtf((float)HD);
    float m = -INFINITY, l = 0.f, acc[4] = {0.f, 0.f, 0.f, 0.f};
    if (two_keys_) {   // two keys per warp iteration: both dots and V loads in flight, one softmax update
        for (int k = k0 + w; k <= k1; k += 8) {
            const int kb = k + 4;
            const bool hb = kb <= k1;
            const int sa = ring ? k % cap : k, sb = ring ? (hb ? kb : k) % cap : (hb ? kb : k);
            const uint2 ka = *(const uint2*)(Kc + ((size_t)kvh * cap + sa) * HD + lane * 4);
            const uint2 va = *(const uint2*)(Vc + ((size_t)kvh * cap + sa) * HD + lane * 4);
            const uint2 kb2 = *(const uint2*)(Kc + ((size_t)kvh * cap + sb) * HD + lane * 4);
            const uint2 vb = *(const uint2*)(Vc + ((size_t)kvh * cap + sb) * HD + lane * 4);
            float da = q.x * __uint_as_float(ka.x << 16) + q.y * __uint_as_float(ka.x & 0xffff0000u) +
                       q.z * __uint_as_float(ka.y << 16) + q.w * __uint_as_float(ka.y & 0xffff0000u);
            float db = q.x * __uint_as_float(kb2.x << 16) + q.y * __uint_as_float(kb2.x & 0xffff0000u) +
                       q.z * __uint_as_float(kb2.y << 16) + q.w * __uint_as_float(kb2.y & 0xffff0000u);
            for (int o = 16; o; o >>= 1) {
                da += __shfl_xor_sync(0xffffffffu, da, o);
                db += __shfl_xor_sync(0xffffffffu, db, o);
            }
            da *= scale;
            db *= scale;
            const int ta = qp - k, tb = qp - kb;
            if (ta < REL) {
                float b = 0.f;
#pragma unroll
                for (int i = 0; i < DREL; i++) b += r[i] * __ldg(proj + i * REL + ta);
                da += b;
            }
            if (hb && tb < REL) {
                float b = 0.f;
#pragma unroll
                for (int i = 0; i < DREL; i++) b += r[i] * __ldg(proj + i * REL + tb);
                db += b;
            }
            if (!hb) db = -INFINITY;
            const float mn = fmaxf(m, fmaxf(da, db));
            const float cf = expf(m - mn), pa_ = expf(da - mn), pb = hb ? expf(db - mn) : 0.f;
            l = l * cf + pa_ + pb;
            acc[0] = acc[0] * cf + pa_ * __uint_as_float(va.x << 16) + pb * __uint_as_float(vb.x << 16);
            acc[1] = acc[1] * cf + pa_ * __uint_as_float(va.x & 0xffff0000u) + pb * __uint_as_float(vb.x & 0xffff0000u);
            acc[2] = acc[2] * cf + pa_ * __uint_as_float(va.y << 16) + pb * __uint_as_float(vb.y << 16);
            acc[3] = acc[3] * cf + pa_ * __uint_as_float(va.y & 0xffff0000u) + pb * __uint_as_float(vb.y & 0xffff0000u);
            m = mn;
        }
    } else
    for (int k = k0 + w; k <= k1; k += 4) {
        const int slot = ring ? k % cap : k;
        const uint2 kr = *(const uint2*)(Kc + ((size_t)kvh * cap + slot) * HD + lane * 4);
        const uint2 vr = *(const uint2*)(Vc + ((size_t)kvh * cap + slot) * HD + lane * 4);
        float d = q.x * __uint_as_float(kr.x << 16) + q.y * __uint_as_float(kr.x & 0xffff0000u) +
                  q.z * __uint_as_float(kr.y << 16) + q.w * __uint_as_float(kr.y & 0xffff0000u);
        d = warp_sum(d) * scale;
        const int dist = qp - k;
        if (dist < REL) {
            float b = 0.f;
#pragma unroll
            for (int i = 0; i < DREL; i++) b += r[i] * __ldg(proj + i * REL + dist);
            d += b;
        }
        const float mn = fmaxf(m, d);
        const float cf = expf(m - mn), p = expf(d - mn);
        l = l * cf + p;
        acc[0] = acc[0] * cf + p * __uint_as_float(vr.x << 16);
        acc[1] = acc[1] * cf + p * __uint_as_float(vr.x & 0xffff0000u);
        acc[2] = acc[2] * cf + p * __uint_as_float(vr.y << 16);
        acc[3] = acc[3] * cf + p * __uint_as_float(vr.y & 0xffff0000u);
        m = mn;
    }
    if (lane == 0) {
        sm[w] = m;
        sl[w] = l;
    }
#pragma unroll
    for (int i = 0; i < 4; i++) sacc[w][lane * 4 + i] = acc[i];
    BSYNC(128);
    float M = -INFINITY;
    for (int j = 0; j < 4; j++) M = fmaxf(M, sm[j]);
    const size_t pi = ((size_t)t * NQ + h) * NS + z;
    const int dd = threadIdx.x;
    float L = 0.f, o = 0.f;
    if (M != -INFINITY)
        for (int j = 0; j < 4; j++)
            if (sm[j] != -INFINITY) {
                const float e = expf(sm[j] - M);
                L += sl[j] * e;
                o += sacc[j][dd] * e;
            }
    pa[pi * HD + dd] = o;
    if (dd == 0) {
        pm[pi] = M;
        pl[pi] = L;
    }
    if (!A_.cnt) return;
    __shared__ int last;
    __shared__ float we[128];
    __threadfence();
    BSYNC(128);
    if (threadIdx.x == 0) last = atomicAdd(A_.cnt + h, 1) == NS - 1;
    BSYNC(128);
    if (!last) return;
    __threadfence();
    const size_t b0 = (size_t)h * NS;
    if (threadIdx.x < 32) {   // merge weights: a lane per split (NS <= 128)
        const float sink = A_.S[h];
        float mz[4], lz[4], M = sink;
#pragma unroll
        for (int j = 0; j < 4; j++) {
            const int zz = threadIdx.x + 32 * j;
            mz[j] = zz < NS ? __ldcg(pm + b0 + zz) : -INFINITY;
            lz[j] = zz < NS ? __ldcg(pl + b0 + zz) : 0.f;
            M = fmaxf(M, mz[j]);
        }
        M = warp_max(M);
        float L = 0.f, e[4];
#pragma unroll
        for (int j = 0; j < 4; j++) {
            e[j] = mz[j] == -INFINITY ? 0.f : expf(mz[j] - M);
            L += lz[j] * e[j];
        }
        L = warp_sum(L) + expf(sink - M);
#pragma unroll
        for (int j = 0; j < 4; j++)
            if (threadIdx.x + 32 * j < NS) we[threadIdx.x + 32 * j] = e[j] / L;
    }
    BSYNC(128);
    float oo = 0.f;
#pragma unroll 8
    for (int zz = 0; zz < NS; zz++) oo += we[zz] * __ldcg(pa + (b0 + zz) * HD + threadIdx.x);
    const float g = A_.P[OFF_G + h * HD + threadIdx.x];
    A_.AO[h * HD + threadIdx.x] = __float2bfloat16(oo * (1.f / (1.f + expf(-g))));
    if (A_.AOf) A_.AOf[h * HD + threadIdx.x] = oo * (1.f / (1.f + expf(-g)));
    if (threadIdx.x == 0) A_.cnt[h] = 0;
}
__global__ void __launch_bounds__(128) k_attn_part(const __grid_constant__ A_attn_part A_) { b_attn_part(A_, blockIdx, gridDim); }

constexpr int GQ = NQ / NKV;
// ---- tensor-core decode attention (T = 1): grid (NKV, NS), 4 warps. A block serves the 3 q heads of one KV head
// over one key split; a warp takes 16-key chunks (K / V rows staged in its own shared memory): S = Q K^T with
// mma.sync m16n8k16 (q heads = rows 0-2 of the 16-row A tile, bf16), rel bias + online softmax on the C fragments,
// P (bf16) reused as the A fragment of O += P V (V via ldmatrix.trans). Partials as k_attn_part (per q head), the
// last split block of the KV head merges (lane per split) and writes AO.
__device__ __forceinline__ uint32_t pack_bf2(float lo, float hi) {
    const __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
    return *(const uint32_t*)&v;
}
__device__ __forceinline__ void mma_bf16(float (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,
                                         uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ uint32_t pack_h2(float lo, float hi) {
    const __half2 v = __floats2half2_rn(lo, hi);
    return *(const uint32_t*)&v;
}
__device__ __forceinline__ void mma_f16(float (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,
                                        uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
struct A_attn_tc {
    const float* Q;
    const float* R;
    const float* proj;
    const bf16* Kc;
    const bf16* Vc;
    int cap, ring, local;
    const int* posp;
    float *pm, *pl, *pa;
    int* cnt;   // per KV head
    const float* S;
    const float* P;
    bf16* AO;
    float* AOf;   // optional fp32 copy of the gated output (precision checks)
    int merge3;   // A/B: merge the 3 q heads in one pass
    int ldp;          // verify rows: P row stride
    int nsv[MROW];      // verify rows: splits of row blockIdx.z (0: gridDim.y for every row)
    const float* projT = nullptr;   // [REL][DREL] (k_attn_tc3)
    const int* sko = nullptr;   // [ink-srv-pool] row steps: K / V offset of a slot in HD-rows (null: slot x NKV x cap)
    int kv0 = 0;   // [ink-tp-dec1] KV head of blockIdx.x 0
};
__global__ void k_projT(const float* p, float* t) {   // same values, transposed: t[d][i] = p[i][d]
    const int x = blockIdx.x * 256 + threadIdx.x;
    if (x < REL * DREL) t[x] = p[(x % DREL) * REL + x / DREL];
}
constexpr int TC_LD = HD + 8;   // padded smem row (bf16): conflict-free ldmatrix / 32-bit loads
// HL: hi / lo split in the unused mma rows 8-15 (Q and P as bf16 hi + bf16 lo: ~16-bit operands, V exact bf16)
template <bool PH, bool HL = false>
__global__ void __launch_bounds__(128) k_attn_tc(const __grid_constant__ A_attn_tc A) {
    const bool ab_merge3 = A.merge3;
    __shared__ __align__(16) bf16 ks[4][16][TC_LD], vs[4][16][TC_LD];
    __shared__ float sm[4][GQ], sl[4][GQ], sacc[4][GQ][HD];
    __shared__ float rsh[GQ][DREL];
    __shared__ int last;
    __shared__ float we[GQ][128];
    const int kvh = blockIdx.x, z = blockIdx.y, jr = blockIdx.z, NSX = gridDim.y;
    const int NS = A.nsv[0] ? A.nsv[jr] : NSX;
    if (z >= NS) return;
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5, g = lane >> 2, tq = lane & 3;
    const int qp = A.posp[jr];   // row table (RT_*)
    const size_t sqo = A.sko ? (size_t)A.sko[A.posp[RT_SLOT + jr]] * HD : (size_t)A.posp[RT_SLOT + jr] * NKV * A.cap * HD;
    const int lo = A.local ? max(0, qp - (WIN - 1)) : 0;
    const int n = qp - lo + 1, kspl = (n + NS - 1) / NS;
    const int k0 = lo + z * kspl, k1 = min(qp, k0 + kspl - 1);
    if (threadIdx.x < GQ * DREL) rsh[threadIdx.x / DREL][threadIdx.x % DREL] = A.R[(size_t)jr * NQ * DREL + (kvh * GQ) * DREL + threadIdx.x];
    // Q A-fragments (rows g < 3 real), 8 k-steps over the 128 dims
    uint32_t qa0[8], qa2[8], qa1[8], qa3[8];
    {
        const float* qr = A.Q + (size_t)jr * QD + (size_t)(kvh * GQ + (g < GQ ? g : 0)) * HD;
#pragma unroll
        for (int s = 0; s < 8; s++) {
            const int c = s * 16 + 2 * tq;
            qa0[s] = g < GQ ? pack_bf2(qr[c], qr[c + 1]) : 0u;
            qa2[s] = g < GQ ? pack_bf2(qr[c + 8], qr[c + 9]) : 0u;
            qa1[s] = qa3[s] = 0u;
            if (HL && g < GQ) {   // the residuals q - bf16(q) as rows g + 8
                auto lo = [](float x) { return x - __bfloat162float(__float2bfloat16(x)); };
                qa1[s] = pack_bf2(lo(qr[c]), lo(qr[c + 1]));
                qa3[s] = pack_bf2(lo(qr[c + 8]), lo(qr[c + 9]));
            }
        }
    }
    __syncthreads();
    float rr[DREL];
#pragma unroll
    for (int i = 0; i < DREL; i++) rr[i] = g < GQ ? rsh[g][i] : 0.f;
    const float scale = 1.f / sqrtf((float)HD);
    float m = -INFINITY, l = 0.f;
    float o[16][4];
#pragma unroll
    for (int j = 0; j < 16; j++) o[j][0] = o[j][1] = o[j][2] = o[j][3] = 0.f;
    for (int base = k0 + w * 16; base <= k1; base += 64) {
        // stage 16 K / V rows of this warp (lane: row lane / 2, half lane % 2)
        {
            const int rI = lane >> 1, hh = lane & 1, kk = min(base + rI, k1);
            const int slot = A.ring ? kk % A.cap : kk;
            const uint4* kp = (const uint4*)(A.Kc + sqo + ((size_t)kvh * A.cap + slot) * HD + hh * 64);
            const uint4* vp = (const uint4*)(A.Vc + sqo + ((size_t)kvh * A.cap + slot) * HD + hh * 64);
            uint4* kd = (uint4*)&ks[w][rI][hh * 64];
            uint4* vd = (uint4*)&vs[w][rI][hh * 64];
#pragma unroll
            for (int i = 0; i < 8; i++) {
                kd[i] = kp[i];
                uint4 v = vp[i];
                if (PH && !HL) {
                    uint32_t* u = (uint32_t*)&v;
#pragma unroll
                    for (int q = 0; q < 4; q++) u[q] = pack_h2(__uint_as_float(u[q] << 16), __uint_as_float(u[q] & 0xffff0000u));
                }
                vd[i] = v;
            }
        }
        __syncwarp();
        // S = Q K^T: 2 n-tiles of 8 keys
        float s[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
#pragma unroll
        for (int nt = 0; nt < 2; nt++)
#pragma unroll
            for (int st = 0; st < 8; st++) {
                const bf16* kr = &ks[w][nt * 8 + g][st * 16 + 2 * tq];
                mma_bf16(s[nt], qa0[st], qa1[st], qa2[st], qa3[st], *(const uint32_t*)kr, *(const uint32_t*)(kr + 8));
            }
        if (HL)
#pragma unroll
            for (int nt = 0; nt < 2; nt++) {   // rows g + 8 hold the lo part of the same dot products
                s[nt][0] += s[nt][2];
                s[nt][1] += s[nt][3];
            }
        // scale, rel bias, mask (rows g < 3 in c0 / c1)
        float mx = -INFINITY;
#pragma unroll
        for (int nt = 0; nt < 2; nt++)
#pragma unroll
            for (int e = 0; e < 2; e++) {
                const int key = base + nt * 8 + 2 * tq + e;
                float v = s[nt][e] * scale;
                const int dist = qp - key;
                if (g < GQ && dist < REL && key <= k1) {
                    float b = 0.f;
#pragma unroll
                    for (int i = 0; i < DREL; i++) b += rr[i] * __ldg(A.proj + i * REL + dist);
                    v += b;
                }
                if (key > k1 || g >= GQ) v = -INFINITY;
                s[nt][e] = v;
                mx = fmaxf(mx, v);
            }
        mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 1));
        mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 2));
        const float mn = fmaxf(m, mx);
        const float cf = (m == -INFINITY) ? 0.f : expf(m - mn);
        float ps = 0.f, p[2][2];
#pragma unroll
        for (int nt = 0; nt < 2; nt++)
#pragma unroll
            for (int e = 0; e < 2; e++) {
                p[nt][e] = (s[nt][e] == -INFINITY) ? 0.f : expf(s[nt][e] - mn);
                ps += p[nt][e];
            }
        ps += __shfl_xor_sync(0xffffffffu, ps, 1);
        ps += __shfl_xor_sync(0xffffffffu, ps, 2);
        if (mn != -INFINITY) {
            l = l * cf + ps;
            m = mn;
#pragma unroll
            for (int j = 0; j < 16; j++) {
                o[j][0] *= cf;
                o[j][1] *= cf;
                if (HL) {
                    o[j][2] *= cf;
                    o[j][3] *= cf;
                }
            }
        }
        // O += P V: P rows g (keys 0-7 | 8-15) as the A fragment, V by ldmatrix.trans (2 n-tiles per load)
        const bool ph = PH && !HL;
        const uint32_t pa0 = ph ? pack_h2(p[0][0], p[0][1]) : pack_bf2(p[0][0], p[0][1]);
        const uint32_t pa2 = ph ? pack_h2(p[1][0], p[1][1]) : pack_bf2(p[1][0], p[1][1]);
        uint32_t pa1 = 0u, pa3 = 0u;
        if (HL) {
            auto lo = [](float x) { return x - __bfloat162float(__float2bfloat16(x)); };
            pa1 = pack_bf2(lo(p[0][0]), lo(p[0][1]));
            pa3 = pack_bf2(lo(p[1][0]), lo(p[1][1]));
        }
#pragma unroll
        for (int j = 0; j < 16; j += 2) {
            uint32_t b0, b1, b2, b3;
            const int mrow = (lane & 7) + ((lane >> 3) & 1) * 8, mcol = j * 8 + (lane >> 4) * 8;
            const uint32_t addr = (uint32_t)__cvta_generic_to_shared(&vs[w][mrow][mcol]);
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
                         : "=r"(b0), "=r"(b1), "=r"(b2), "=r"(b3)
                         : "r"(addr));
            if (ph) {
                mma_f16(o[j], pa0, 0u, pa2, 0u, b0, b1);
                mma_f16(o[j + 1], pa0, 0u, pa2, 0u, b2, b3);
            } else {
                mma_bf16(o[j], pa0, pa1, pa2, pa3, b0, b1);
                mma_bf16(o[j + 1], pa0, pa1, pa2, pa3, b2, b3);
            }
        }
        __syncwarp();
    }
    // per-warp partials (rows g < 3: lanes 0-11) -> smem, merge the 4 warps per q head
    if (g < GQ) {
        if (tq == 0) {
            sm[w][g] = m;
            sl[w][g] = l;
        }
#pragma unroll
        for (int j = 0; j < 16; j++) {   // HL: rows g + 8 carry the lo-part products
            sacc[w][g][j * 8 + 2 * tq] = o[j][0] + (HL ? o[j][2] : 0.f);
            sacc[w][g][j * 8 + 2 * tq + 1] = o[j][1] + (HL ? o[j][3] : 0.f);
        }
    }
    __syncthreads();
    const int dd = threadIdx.x;
#pragma unroll
    for (int q = 0; q < GQ; q++) {
        float M = -INFINITY;
        for (int j = 0; j < 4; j++) M = fmaxf(M, sm[j][q]);
        float L = 0.f, oo = 0.f;
        if (M != -INFINITY)
            for (int j = 0; j < 4; j++)
                if (sm[j][q] != -INFINITY) {
                    const float e = expf(sm[j][q] - M);
                    L += sl[j][q] * e;
                    oo += sacc[j][q][dd] * e;
                }
        const size_t pi = ((size_t)jr * NQ + kvh * GQ + q) * NSX + z;
        A.pa[pi * HD + dd] = oo;
        if (dd == 0) {
            A.pm[pi] = M;
            A.pl[pi] = L;
        }
    }
    // the last split block of this KV head merges its 3 q heads
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) last = atomicAdd(A.cnt + jr * NKV + kvh, 1) == NS - 1;
    __syncthreads();
    if (!last) return;
    __threadfence();
    if (w < GQ) {   // warp q: merge weights of q head kvh * 3 + q, a lane per split (NS <= 128)
        const int h = kvh * GQ + w;
        const size_t b0 = ((size_t)jr * NQ + h) * NSX;
        const float sink = A.S[(size_t)jr * NQ + h];
        float mz[4], lz[4], M = sink;
#pragma unroll
        for (int j = 0; j < 4; j++) {
            const int zz = lane + 32 * j;
            mz[j] = zz < NS ? __ldcg(A.pm + b0 + zz) : -INFINITY;
            lz[j] = zz < NS ? __ldcg(A.pl + b0 + zz) : 0.f;
            M = fmaxf(M, mz[j]);
        }
        M = warp_max(M);
        float L = 0.f, e[4];
#pragma unroll
        for (int j = 0; j < 4; j++) {
            e[j] = mz[j] == -INFINITY ? 0.f : expf(mz[j] - M);
            L += lz[j] * e[j];
        }
        L = warp_sum(L) + expf(sink - M);
#pragma unroll
        for (int j = 0; j < 4; j++)
            if (lane + 32 * j < NS) we[w][lane + 32 * j] = e[j] / L;
    }
    __syncthreads();
    float o3[GQ] = {0.f, 0.f, 0.f};
    if (ab_merge3) {   // the 3 q heads in one pass (3 independent load chains)
#pragma unroll 4
        for (int zz = 0; zz < NS; zz++)
#pragma unroll
            for (int q = 0; q < GQ; q++) o3[q] += we[q][zz] * __ldcg(A.pa + (((size_t)jr * NQ + kvh * GQ + q) * NSX + zz) * HD + dd);
    } else
#pragma unroll
        for (int q = 0; q < GQ; q++) {
            const size_t b0 = ((size_t)jr * NQ + kvh * GQ + q) * NSX;
#pragma unroll 8
            for (int zz = 0; zz < NS; zz++) o3[q] += we[q][zz] * __ldcg(A.pa + (b0 + zz) * HD + dd);
        }
#pragma unroll
    for (int q = 0; q < GQ; q++) {
        const int h = kvh * GQ + q;
        const float oo = o3[q];
        const float gt = A.P[(size_t)jr * A.ldp + OFF_G + h * HD + dd];
        A.AO[(size_t)jr * QD + h * HD + dd] = __float2bfloat16(oo * (1.f / (1.f + expf(-gt))));
        if (A.AOf) A.AOf[h * HD + dd] = oo * (1.f / (1.f + expf(-gt)));
    }
    if (threadIdx.x == 0) A.cnt[jr * NKV + kvh] = 0;
}

// merge the splits with the sink logit, times sigmoid(gate) -> bf16 AO. grid (T, NQ), 128 threads.
struct A_attn_comb {
    const float* pm;
    const float* pl;
    const float* pa;
    int NS;
    const float* S;
    const float* P;
    int ldp;
    bf16* AO;
};
__device__ __forceinline__ void b_attn_comb(const A_attn_comb& A_, const dim3 bid_, const dim3 gdim_) {
    const auto& pm = A_.pm;
    const auto& pl = A_.pl;
    const auto& pa = A_.pa;
    const auto& NS = A_.NS;
    const auto& S = A_.S;
    const auto& P = A_.P;
    const auto& ldp = A_.ldp;
    const auto& AO = A_.AO;

    const int t = bid_.x, h = bid_.y, dd = threadIdx.x;
    const size_t b = ((size_t)t * NQ + h) * NS;
    const float sink = S[(size_t)t * NQ + h];
    float M = sink;
    for (int z = 0; z < NS; z++) M = fmaxf(M, pm[b + z]);
    float L = expf(sink - M), o = 0.f;
    for (int z = 0; z < NS; z++)
        if (pm[b + z] != -INFINITY) {
            const float e = expf(pm[b + z] - M);
            L += pl[b + z] * e;
            o += pa[(b + z) * HD + dd] * e;
        }
    o /= L;
    const float g = P[(size_t)t * ldp + OFF_G + h * HD + dd];
    AO[(size_t)t * QD + h * HD + dd] = __float2bfloat16(o * (1.f / (1.f + expf(-g))));
}
__global__ void __launch_bounds__(128) k_attn_comb(const __grid_constant__ A_attn_comb A_) { b_attn_comb(A_, blockIdx, gridDim); }


// router logits: grid (NE/8, T), warp -> expert; sigmoid scores -> sc [T][NE]
__global__ void __launch_bounds__(256) k_rlog(const float* x, const bf16* W, float* sc) {
    const int t = blockIdx.y, lane = threadIdx.x & 31, e = blockIdx.x * 8 + (threadIdx.x >> 5);
    const bf16* row = W + (size_t)e * H;
    const float* xs = x + (size_t)t * H;
    float s = 0.f;
    for (int c = lane * 8; c < H; c += 256) {
        const uint4 wv = *(const uint4*)(row + c);
        const float4 a = *(const float4*)(xs + c), b = *(const float4*)(xs + c + 4);
        s += __uint_as_float(wv.x << 16) * a.x + __uint_as_float(wv.x & 0xffff0000u) * a.y +
             __uint_as_float(wv.y << 16) * a.z + __uint_as_float(wv.y & 0xffff0000u) * a.w +
             __uint_as_float(wv.z << 16) * b.x + __uint_as_float(wv.z & 0xffff0000u) * b.y +
             __uint_as_float(wv.w << 16) * b.z + __uint_as_float(wv.w & 0xffff0000u) * b.w;
    }
    s = warp_sum(s);
    if (lane == 0) sc[(size_t)t * NE + e] = 1.f / (1.f + expf(-s));
}
// top-8 of score + bias (ties: lower index), weights = score / sum * 8. one warp per token.
__global__ void __launch_bounds__(32) k_topk(const float* sc, const float* bias, int* ids, float* wts) {
    const int t = blockIdx.x, lane = threadIdx.x;
    float s[4], v[4];
#pragma unroll
    for (int j = 0; j < 4; j++) {
        s[j] = sc[(size_t)t * NE + lane + 32 * j];
        v[j] = s[j] + bias[lane + 32 * j];
    }
    float sum = 0.f, mine = 0.f;
    int myid = 0;
    for (int k = 0; k < TOPK; k++) {
        float bv = -INFINITY;
        int bi = 1 << 30;
#pragma unroll
        for (int j = 0; j < 4; j++)
            if (v[j] > bv) {
                bv = v[j];
                bi = lane + 32 * j;
            }
        for (int o = 16; o; o >>= 1) {
            const float v2 = __shfl_xor_sync(0xffffffffu, bv, o);
            const int i2 = __shfl_xor_sync(0xffffffffu, bi, o);
            if (v2 > bv || (v2 == bv && i2 < bi)) {
                bv = v2;
                bi = i2;
            }
        }
        const int bj = (bi >> 5) & 3;
        const float sl = bj == 0 ? s[0] : bj == 1 ? s[1] : bj == 2 ? s[2] : s[3];
        const float sb = __shfl_sync(0xffffffffu, sl, bi & 31);
        sum += sb;
        if (lane == k) {
            mine = sb;
            myid = bi;
        }
#pragma unroll
        for (int j = 0; j < 4; j++)
            if (bi == lane + 32 * j) v[j] = -INFINITY;
    }
    if (lane < TOPK) {
        ids[t * TOPK + lane] = myid;
        wts[t * TOPK + lane] = mine / (sum + 1e-20f) * ROUTE_SCALE;
    }
}


// ---- T = 1 folds (decode)
// q post (24 blocks), K and V conv + clip / norm + cache write + conv-state shift (8 + 8 blocks): one launch.
struct A_qkv1 {
    const float* P;
    int ldp;
    int offK;
    int offV;
    int full;
    const float* qs;
    const float* rs;
    const float* kw;
    const float* vw;
    float* kst;
    float* vst;
    bf16* Kc;
    bf16* Vc;
    int cap;
    int ring;
    const int* posp;
    float* Q;
    float* R;
    float* S;
    int q0 = 0, nq = NQ, kv0 = 0, nkv = NKV;   // [ink-tp-dec1] the rank's q heads [q0, q0 + nq), K / V heads [kv0, kv0 + nkv)
};
__device__ __forceinline__ void b_qkv1(const A_qkv1& A_, const dim3 bid_, const dim3 gdim_) {
    const auto& P = A_.P;
    const auto& ldp = A_.ldp;
    const auto& offK = A_.offK;
    const auto& offV = A_.offV;
    const auto& full = A_.full;
    const auto& qs = A_.qs;
    const auto& rs = A_.rs;
    const auto& kw = A_.kw;
    const auto& vw = A_.vw;
    const auto& kst = A_.kst;
    const auto& vst = A_.vst;
    const auto& Kc = A_.Kc;
    const auto& Vc = A_.Vc;
    const auto& cap = A_.cap;
    const auto& ring = A_.ring;
    const auto& posp = A_.posp;
    const auto& Q = A_.Q;
    const auto& R = A_.R;
    const auto& S = A_.S;

    __shared__ float red[8];
    const int b = bid_.x, i = threadIdx.x;
    const int pos = *posp;
    if (b < A_.nq) {
        const int h = A_.q0 + b;
        const float tau = full ? log_tau(pos) : 1.f;
        float q = P[OFF_Q + h * HD + i] * qs[h * HD + i];
        float s = warp_sum(q * q);
        if ((i & 31) == 0) red[i >> 5] = s;
        BSYNC(128);
        const float nrm = sqrtf(red[0] + red[1] + red[2] + red[3]);
        q = q * fminf(QK_CLIP / (nrm + 1e-6f), 1.f);
        Q[h * HD + i] = q * tau;
        if (i < 32) {
            float r = i < DREL ? P[OFF_R + h * DREL + i] * rs[h * DREL + i] : 0.f;
            const float rn = sqrtf(warp_sum(r * r));
            r = r * fminf(REL_CLIP / (rn + 1e-6f), 1.f);
            if (i < DREL) R[h * DREL + i] = r * tau;
            if (i == 0) S[h] = P[OFF_S + h];
        }
        return;
    }
    const int isK = b < A_.nq + A_.nkv, h = A_.kv0 + (isK ? b - A_.nq : b - A_.nq - A_.nkv), c = h * HD + i;
    const int off = isK ? offK : offV;
    float* st = isK ? kst : vst;
    const float* w = isK ? kw : vw;
    float x[4] = {P[off + c], st[2 * KVD + c], st[KVD + c], st[c]};
    float o;
    if (!full) {
        o = w[c * 4 + 3] * x[0] + w[c * 4 + 2] * x[1] + w[c * 4 + 1] * x[2] + w[c * 4 + 0] * x[3];
        o = (o + x[0]) * (1.f / sqrtf(5.f));
    } else {
        const float* dy = P + off + KVD + h * 4;
        o = 0.f;
#pragma unroll
        for (int j = 0; j < 4; j++) o += (w[j * KVD + c] + dy[j]) * x[j];
        o = (o + x[0]) * (1.f / 3.f);
    }
    st[c] = x[2];
    st[KVD + c] = x[1];
    st[2 * KVD + c] = x[0];
    float s = warp_sum(o * o);
    if ((i & 31) == 0) red[i >> 5] = s;
    BSYNC(128);
    const float ssum = red[0] + red[1] + red[2] + red[3];
    if (isK) o = o * fminf(QK_CLIP / (sqrtf(ssum) + 1e-6f), 1.f);
    else o = o * rsqrtf(ssum / HD + 1e-6f);
    const int slot = ring ? pos % cap : pos;
    (isK ? Kc : Vc)[((size_t)h * cap + slot) * HD + i] = __float2bfloat16(o);
}
__global__ void __launch_bounds__(128) k_qkv1(const __grid_constant__ A_qkv1 A_) { b_qkv1(A_, blockIdx, gridDim); }

// static conv over hidden for one token, state shift and residual: out = add + (conv(x) + x) / sqrt(5)
__device__ __forceinline__ float conv1(float x0, int c, float* st, const float* w, const float* add) {
    // [ink-1c8] review L3: the FMA chain spelled out (the contraction nvcc gave the plain expression), so k_q8w's MODE 1
    // epilogue (same chain) is bitwise this one by construction, not by the compiler's choice
    const float x1 = st[2 * H + c], x2 = st[H + c], x3 = st[c];
    const float o = __fmaf_rn(w[c * 4 + 0], x3, __fmaf_rn(w[c * 4 + 1], x2, __fmaf_rn(w[c * 4 + 3], x0, __fmul_rn(w[c * 4 + 2], x1))));
    st[c] = x2;
    st[H + c] = x1;
    st[2 * H + c] = x0;
    return __fmaf_rn(__fadd_rn(o, x0), 1.f / sqrtf(5.f), add ? add[c] : 0.f);
}
__global__ void k_conv1(const float* X, float* st, const float* w, const float* add, float* out) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c < H) out[c] = conv1(X[c], c, st, w, add);
}
// three RMSNorms at once (grid (T, n)): the message-gated q / k / v inputs
struct Norm3 {
    const float* x[3];
    const bf16* w[3];
    bf16* o[3];
};
struct A_rmsnorm3 {
    Norm3 a;
};
__device__ __forceinline__ void b_rmsnorm3(const A_rmsnorm3& A_, const dim3 bid_, const dim3 gdim_) {
    const auto& a = A_.a;

    __shared__ float red[33];
    const int t = bid_.x, g = bid_.y;
    float v[9], ss[1] = {0.f};
#pragma unroll
    for (int i = 0; i < 9; i++) {
        v[i] = a.x[g][(size_t)t * H + threadIdx.x + 256 * i];
        ss[0] += v[i] * v[i];
    }
    block_sum<1>(ss, red);
    const float inv = rsqrtf(ss[0] / H + 1e-6f);
#pragma unroll
    for (int i = 0; i < 9; i++) {
        const int c = threadIdx.x + 256 * i;
        a.o[g][(size_t)t * H + c] = __float2bfloat16(__bfloat162float(a.w[g][c]) * (v[i] * inv));
    }
}
__global__ void __launch_bounds__(256) k_rmsnorm3(const __grid_constant__ A_rmsnorm3 A_) { b_rmsnorm3(A_, blockIdx, gridDim); }

// top-8 inside an expert kernel: warp 0 of every block selects (the same result in every block); block (0, 0)
// publishes ids / weights for the down kernel. Leaves the expert of this block's slot in *e_out.
__device__ __forceinline__ void block_topk(const float* sc, const float* bias, int t, int slot, int* ids, float* wts,
                                           bool publish, int* e_sh) {
    if (threadIdx.x < 32) {
        const int lane = threadIdx.x;
        float s[4], v[4];
#pragma unroll
        for (int j = 0; j < 4; j++) {
            s[j] = sc[(size_t)t * NE + lane + 32 * j];
            v[j] = s[j] + bias[lane + 32 * j];
        }
        float sum = 0.f, mine = 0.f;
        int myid = 0;
        for (int k = 0; k < TOPK; k++) {
            float bv = -INFINITY;
            int bi = 1 << 30;
#pragma unroll
            for (int j = 0; j < 4; j++)
                if (v[j] > bv) {
                    bv = v[j];
                    bi = lane + 32 * j;
                }
            for (int o = 16; o; o >>= 1) {
                const float v2 = __shfl_xor_sync(0xffffffffu, bv, o);
                const int i2 = __shfl_xor_sync(0xffffffffu, bi, o);
                if (v2 > bv || (v2 == bv && i2 < bi)) {
                    bv = v2;
                    bi = i2;
                }
            }
            if (bi >= NE) bi = k;   // NaN scores (review L1): a valid, distinct expert instead of an out-of-range id
            const int bj = (bi >> 5) & 3;
        const float sl = bj == 0 ? s[0] : bj == 1 ? s[1] : bj == 2 ? s[2] : s[3];
        const float sb = __shfl_sync(0xffffffffu, sl, bi & 31);
            sum += sb;
            if (lane == k) {
                mine = sb;
                myid = bi;
            }
    #pragma unroll
        for (int j = 0; j < 4; j++)
            if (bi == lane + 32 * j) v[j] = -INFINITY;
        }
        const float w = mine / (sum + 1e-20f) * ROUTE_SCALE;
        if (lane == slot) {
            e_sh[0] = myid;
            e_sh[1] = __float_as_int(w);
        }
        if (publish && lane < TOPK) {
            ids[t * TOPK + lane] = myid;
            wts[t * TOPK + lane] = w;
        }
    }
    __syncthreads();
}

// top-8 by rank (same result and the same weight arithmetic as block_topk): thread i < NE counts the values above its
// own (ties: lower index first); ranks 0..7 are the selection in block_topk's order; the sum of the selected scores in
// that order. Non-finite scores: block_topk (its NaN guard).
__device__ __forceinline__ void block_topk_rank(const float* sc, const float* bias, int* ids, float* wts, int* e_sh) {
    __shared__ float vv[NE];
    __shared__ int sid[TOPK];
    __shared__ float ssc[TOPK];
    const int i = threadIdx.x;
    float v = 0.f;
    if (i < NE) {
        v = sc[i] + bias[i];
        vv[i] = v;
    }
    const int bad = __syncthreads_or(i < NE && !isfinite(v));
    if (bad) {
        block_topk(sc, bias, 0, 0, ids, wts, true, e_sh);
        return;
    }
    if (i < NE) {
        int r = 0;
#pragma unroll 8
        for (int j = 0; j < NE; j++) {
            const float u = vv[j];
            r += (u > v) || (u == v && j < i);
        }
        if (r < TOPK) {
            sid[r] = i;
            ssc[r] = sc[i];
        }
    }
    __syncthreads();
    if (i < TOPK) {
        float sum = 0.f;
#pragma unroll
        for (int k = 0; k < TOPK; k++) sum += ssc[k];
        ids[i] = sid[i];
        wts[i] = ssc[i] / (sum + 1e-20f) * ROUTE_SCALE;
    }
    __syncthreads();
}

// ---- BF16 GEMV for decode (T = 1): y[r] = W[r] . x, a warp owns R rows; each lane keeps its K/256 16-byte x chunks
// in registers and issues all R * K/256 weight loads at once. Rows [0, r1) read x0, [r1, r2) x1, [r2, N) x2 (one
// launch for every projection of an application). Epilogues: conv (post-attention conv + state + residual, out =
// add + conv(dot)); bmax (per-block max and its row, for the head argmax).
struct GemvArgs {
    const bf16* W;
    const bf16 *x0, *x1, *x2;
    int N, r1, r2;
    float* y;
    float* cst;          // conv state (epilogue conv) or null
    const float* cw;
    const float* add;
    float* bmax;         // [gridDim.x][2] (value, row as float bits) or null
};
template <int K, int R>
__global__ void __launch_bounds__(256) k_gemv(const __grid_constant__ GemvArgs a) {
    constexpr int NC = K / 256;   // 16 B chunks per lane
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    const int r0 = (blockIdx.x * 8 + w) * R;
    const bf16* xp = r0 < a.r1 ? a.x0 : r0 < a.r2 ? a.x1 : a.x2;   // R divides the segment sizes
    uint4 wv[R][NC];
#pragma unroll
    for (int r = 0; r < R; r++)
#pragma unroll
        for (int c = 0; c < NC; c++)
            if (r0 + r < a.N) wv[r][c] = *(const uint4*)(a.W + (size_t)(r0 + r) * K + (c * 32 + lane) * 8);
    uint4 xv[NC];
#pragma unroll
    for (int c = 0; c < NC; c++) xv[c] = *(const uint4*)(xp + (c * 32 + lane) * 8);
    float acc[R];
#pragma unroll
    for (int r = 0; r < R; r++) {
        float s = 0.f;
#pragma unroll
        for (int c = 0; c < NC; c++) {
            const uint32_t ww[4] = {wv[r][c].x, wv[r][c].y, wv[r][c].z, wv[r][c].w};
            const uint32_t xx[4] = {xv[c].x, xv[c].y, xv[c].z, xv[c].w};
#pragma unroll
            for (int k = 0; k < 4; k++)
                s += __uint_as_float(ww[k] << 16) * __uint_as_float(xx[k] << 16) +
                     __uint_as_float(ww[k] & 0xffff0000u) * __uint_as_float(xx[k] & 0xffff0000u);
        }
        acc[r] = warp_sum(s);
    }
    __shared__ float bm[8];
    __shared__ int bi[8];
    if (lane == 0) {
        float m = -INFINITY;
        int mi = 0;
#pragma unroll
        for (int r = 0; r < R; r++) {
            const int row = r0 + r;
            if (row >= a.N) continue;
            float v = acc[r];
            if (a.cst) v = conv1(v, row, a.cst, a.cw, a.add);
            a.y[row] = v;
            if (v > m) {
                m = v;
                mi = row;
            }
        }
        bm[w] = m;
        bi[w] = mi;
    }
    if (a.bmax) {
        __syncthreads();
        if (threadIdx.x == 0) {
            float m = bm[0];
            int mi = bi[0];
            for (int j = 1; j < 8; j++)
                if (bm[j] > m) {
                    m = bm[j];
                    mi = bi[j];
                }
            a.bmax[2 * blockIdx.x] = m;
            a.bmax[2 * blockIdx.x + 1] = __int_as_float(mi);
        }
    }
}
// ---- lossless 12-bit BF16 ("G15", [ink-exp12], user decision 2026-11-22): kernels from the exp12 branch
// (tools/inkling_exp12/gemv12.cu), bit-identical outputs to k_gemv; encoder: encode12() on the host at load.
// shared epilogue: per-warp rows -> y (+conv), block max
template <int R>
__device__ __forceinline__ void epilogue(const GemvArgs& a, const float* acc, int r0, int lane, int w, int bx) {
    __shared__ float bm[8];
    __shared__ int bi[8];
    if (lane == 0) {
        float m = -INFINITY;
        int mi = 0;
#pragma unroll
        for (int r = 0; r < R; r++) {
            const int row = r0 + r;
            if (row >= a.N) continue;
            float v = acc[r];
            if (a.cst) v = conv1(v, row, a.cst, a.cw, a.add);
            a.y[row] = v;
            if (v > m) {
                m = v;
                mi = row;
            }
        }
        bm[w] = m;
        bi[w] = mi;
    }
    if (a.bmax) {
        BSYNC(256);
        if (threadIdx.x == 0) {
            float m = bm[0];
            int mi = bi[0];
            for (int j = 1; j < 8; j++)
                if (bm[j] > m) {
                    m = bm[j];
                    mi = bi[j];
                }
            a.bmax[2 * bx] = m;
            a.bmax[2 * bx + 1] = __int_as_float(mi);
        }
    }
}
// ---- 12-bit GEMV. Group g = 2 rows = one warp. Planes: sm [G][NC][32] uint4, idx [G] 32 * 2NC words (NF full
// uint4 units unit-major + a tail of TL words per lane), meta [G] 16 B, fallback rows raw.
struct G12Args {
    GemvArgs a;
    const uint4* sm;
    const uint32_t* idx;
    const uint4* meta;
    const bf16* fb;
    int i8 = 0;   // [ink-n4-int8] INT8 g128 planes (sm = codes, idx = scales; ink_tcr.cuh)
    int ks = 0;   // [ink-vp-qs] K parts of the fragment order when not tcr_S<K> (the qkv planes: tcr_SQ)
};
// one BF16 pair word (elements 2k, 2k+1 of a chunk) from the sign|mantissa word, the index word, the row's base and
// escape exponents (both pre-shifted to bits 7 / 23)
template <int k>
__device__ __forceinline__ uint32_t dec_pair(uint32_t smw, uint32_t I, uint32_t B, uint32_t ESC) {
    const uint32_t x2 = __byte_perm(smw, 0u, (k & 1) ? 0x4342 : 0x4140);
    const uint32_t f = (x2 | (x2 << 8)) & 0x807f807fu;
    const uint32_t E = (k == 0 ? (I << 7) : k == 1 ? (I << 3) : k == 2 ? (I >> 1) : (I >> 5)) & 0x07800780u;
    const uint32_t X = E + B;
    const uint32_t M = (((E + 0x00800080u) >> 4) & 0x00800080u) * 0xffu;   // 0x7f80 in each escaped half
    return f | (X & ~M) | (ESC & M);
}
struct A_gemv12 {
    G12Args ga;
};
template <int K, bool ESCOPS>
#ifndef EXP12_MINB
#define EXP12_MINB 1
#endif
__device__ __forceinline__ void b_gemv12(const A_gemv12& A_, const dim3 bid_, const dim3 gdim_) {
    const auto& ga = A_.ga;

    constexpr int NC = K / 256, WD = 2 * NC, NF = WD / 4, TL = WD % 4;
    const GemvArgs& a = ga.a;
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    const int g = bid_.x * 8 + w;
    const int r0 = g * 2;
    const int G = (a.N + 1) / 2;
    const bf16* xp = r0 < a.r1 ? a.x0 : r0 < a.r2 ? a.x1 : a.x2;
    uint4 sv[NC], qv[NF + (TL ? 1 : 0)], m = make_uint4(0, 0, 0, 0);
    if (g < G) {
#pragma unroll
        for (int i = 0; i < NC; i++) sv[i] = ga.sm[((size_t)g * NC + i) * 32 + lane];
        const uint32_t* ip = ga.idx + (size_t)g * 32 * WD;
#pragma unroll
        for (int i = 0; i < NF; i++) qv[i] = *(const uint4*)(ip + (i * 32 + lane) * 4);
        if (TL) {
            const uint2 t = *(const uint2*)(ip + NF * 128 + lane * 2);
            qv[NF] = make_uint4(t.x, t.y, 0, 0);
        }
        m = ga.meta[g];
    }
    uint4 xv[NC];
#pragma unroll
    for (int c = 0; c < NC; c++) xv[c] = *(const uint4*)(xp + (c * 32 + lane) * 8);
    float acc[2];
    const uint32_t n = (m.x >> 16) & 0xff;
    if (n == 255) {   // fallback group (warp-uniform): raw BF16 rows, the port's code
        const bf16* W = ga.fb + (size_t)m.y * 2 * K;
#pragma unroll 1
        for (int r = 0; r < 2; r++) {   // one row at a time (rare path, keeps the register budget of the hot path)
            uint4 wv[NC];
#pragma unroll
            for (int c = 0; c < NC; c++) wv[c] = *(const uint4*)(W + (size_t)r * K + (c * 32 + lane) * 8);
            float s = 0.f;
#pragma unroll
            for (int c = 0; c < NC; c++) {
                const uint32_t ww[4] = {wv[c].x, wv[c].y, wv[c].z, wv[c].w};
                const uint32_t xx[4] = {xv[c].x, xv[c].y, xv[c].z, xv[c].w};
#pragma unroll
                for (int k = 0; k < 4; k++)
                    s += __uint_as_float(ww[k] << 16) * __uint_as_float(xx[k] << 16) +
                         __uint_as_float(ww[k] & 0xffff0000u) * __uint_as_float(xx[k] & 0xffff0000u);
            }
            const float t = warp_sum(s);
            if (r == 0) acc[0] = t; else acc[1] = t;
        }
    } else {
        const uint32_t b0 = m.x & 0xff, b1 = (m.x >> 8) & 0xff;
        const uint32_t Bv[2] = {(b0 << 7) | (b0 << 23), (b1 << 7) | (b1 << 23)};
        uint32_t ESC[2] = {0u, 0u};
        if (ESCOPS) {
            const uint32_t ent[6] = {m.y & 0xffff, m.y >> 16, m.z & 0xffff, m.z >> 16, m.w & 0xffff, m.w >> 16};
#pragma unroll
            for (int s = 0; s < 6; s++) {
                const uint32_t e = ent[s], ex = e >> 6;
                const bool mine = s < (int)n && (e & 31) == (uint32_t)lane;
                const uint32_t v = (ex << 7) | (ex << 23);
                if ((e >> 5) & 1) ESC[1] = mine ? v : ESC[1];
                else ESC[0] = mine ? v : ESC[0];
            }
        }
        const uint32_t* sw = (const uint32_t*)sv;
        const uint32_t* iw = (const uint32_t*)qv;
#pragma unroll
        for (int r = 0; r < 2; r++) {
            float s = 0.f;
#pragma unroll
            for (int c = 0; c < NC; c++) {
                const int t = r * NC + c;
                const uint32_t s0 = sw[2 * t], s1 = sw[2 * t + 1], I = iw[t];
                const uint32_t ww[4] = {dec_pair<0>(s0, I, Bv[r], ESC[r]), dec_pair<1>(s0, I, Bv[r], ESC[r]),
                                        dec_pair<2>(s1, I, Bv[r], ESC[r]), dec_pair<3>(s1, I, Bv[r], ESC[r])};
                const uint32_t xx[4] = {xv[c].x, xv[c].y, xv[c].z, xv[c].w};
#pragma unroll
                for (int k = 0; k < 4; k++)
                    s += __uint_as_float(ww[k] << 16) * __uint_as_float(xx[k] << 16) +
                         __uint_as_float(ww[k] & 0xffff0000u) * __uint_as_float(xx[k] & 0xffff0000u);
            }
            acc[r] = warp_sum(s);
        }
    }
    epilogue<2>(a, acc, r0, lane, w, bid_.x);
}
template <int K, bool ESCOPS>
#ifndef EXP12_MINB
#define EXP12_MINB 1
#endif
__global__ void __launch_bounds__(256, EXP12_MINB) k_gemv12(const __grid_constant__ A_gemv12 A_) { b_gemv12<K, ESCOPS>(A_, blockIdx, gridDim); }

// one row per warp (K % 1024 == 0, e.g. wo K = 3072): the warp pair (2g, 2g+1) shares group g; a warp loads only its
// row's units, so the register budget is half and the grid has twice the warps
struct A_gemv12r1 {
    G12Args ga;
};
template <int K, bool ESCOPS>
__device__ __forceinline__ void b_gemv12r1(const A_gemv12r1& A_, const dim3 bid_, const dim3 gdim_) {
    const auto& ga = A_.ga;

    constexpr int NC = K / 256, WD = 2 * NC, US = NC / 2, UI = NC / 4;
    static_assert(NC % 4 == 0, "row-aligned units");
    const GemvArgs& a = ga.a;
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    const int row = bid_.x * 8 + w, g = row >> 1, r = row & 1;
    const bf16* xp = row < a.r1 ? a.x0 : row < a.r2 ? a.x1 : a.x2;
    uint4 sv[US], qv[UI], m = make_uint4(0, 0, 0, 0);
    if (row < a.N) {
#pragma unroll
        for (int i = 0; i < US; i++) sv[i] = ga.sm[((size_t)g * NC + r * US + i) * 32 + lane];
        const uint32_t* ip = ga.idx + (size_t)g * 32 * WD;
#pragma unroll
        for (int i = 0; i < UI; i++) qv[i] = *(const uint4*)(ip + ((r * UI + i) * 32 + lane) * 4);
        m = ga.meta[g];
    }
    uint4 xv[NC];
#pragma unroll
    for (int c = 0; c < NC; c++) xv[c] = *(const uint4*)(xp + (c * 32 + lane) * 8);
    float acc[1];
    const uint32_t n = (m.x >> 16) & 0xff;
    float s = 0.f;
    if (n == 255) {
        const bf16* W = ga.fb + (size_t)m.y * 2 * K + (size_t)r * K;
        uint4 wv[NC];
#pragma unroll
        for (int c = 0; c < NC; c++) wv[c] = *(const uint4*)(W + (c * 32 + lane) * 8);
#pragma unroll
        for (int c = 0; c < NC; c++) {
            const uint32_t ww[4] = {wv[c].x, wv[c].y, wv[c].z, wv[c].w};
            const uint32_t xx[4] = {xv[c].x, xv[c].y, xv[c].z, xv[c].w};
#pragma unroll
            for (int k = 0; k < 4; k++)
                s += __uint_as_float(ww[k] << 16) * __uint_as_float(xx[k] << 16) +
                     __uint_as_float(ww[k] & 0xffff0000u) * __uint_as_float(xx[k] & 0xffff0000u);
        }
    } else {
        const uint32_t b = r ? (m.x >> 8) & 0xff : m.x & 0xff;
        const uint32_t Bv = (b << 7) | (b << 23);
        uint32_t ESC = 0u;
        if (ESCOPS) {
            const uint32_t ent[6] = {m.y & 0xffff, m.y >> 16, m.z & 0xffff, m.z >> 16, m.w & 0xffff, m.w >> 16};
#pragma unroll
            for (int j = 0; j < 6; j++) {
                const uint32_t e = ent[j], ex = e >> 6;
                const bool mine = j < (int)n && (e & 31) == (uint32_t)lane && ((e >> 5) & 1) == (uint32_t)r;
                ESC = mine ? ((ex << 7) | (ex << 23)) : ESC;
            }
        }
        const uint32_t* sw = (const uint32_t*)sv;
        const uint32_t* iw = (const uint32_t*)qv;
#pragma unroll
        for (int c = 0; c < NC; c++) {
            const uint32_t s0 = sw[2 * c], s1 = sw[2 * c + 1], I = iw[c];
            const uint32_t ww[4] = {dec_pair<0>(s0, I, Bv, ESC), dec_pair<1>(s0, I, Bv, ESC), dec_pair<2>(s1, I, Bv, ESC),
                                    dec_pair<3>(s1, I, Bv, ESC)};
            const uint32_t xx[4] = {xv[c].x, xv[c].y, xv[c].z, xv[c].w};
#pragma unroll
            for (int k = 0; k < 4; k++)
                s += __uint_as_float(ww[k] << 16) * __uint_as_float(xx[k] << 16) +
                     __uint_as_float(ww[k] & 0xffff0000u) * __uint_as_float(xx[k] & 0xffff0000u);
        }
    }
    acc[0] = warp_sum(s);
    epilogue<1>(a, acc, row, lane, w, bid_.x);
}
template <int K, bool ESCOPS>
__global__ void __launch_bounds__(256) k_gemv12r1(const __grid_constant__ A_gemv12r1 A_) { b_gemv12r1<K, ESCOPS>(A_, blockIdx, gridDim); }

// decode-only check: writes the BF16 words of every non-fallback group (fallback groups: copies the raw rows)
template <int K>
__global__ void k_dec12(G12Args ga, uint32_t* out) {
    constexpr int NC = K / 256, WD = 2 * NC, NF = WD / 4, TL = WD % 4;
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    const int g = blockIdx.x * 8 + w;
    if (g >= (ga.a.N + 1) / 2) return;
    uint4 sv[NC], qv[NF + 1];
    for (int i = 0; i < NC; i++) sv[i] = ga.sm[((size_t)g * NC + i) * 32 + lane];
    const uint32_t* ip = ga.idx + (size_t)g * 32 * WD;
    for (int i = 0; i < NF; i++) qv[i] = *(const uint4*)(ip + (i * 32 + lane) * 4);
    if (TL) qv[NF] = make_uint4(ip[NF * 128 + lane * 2], ip[NF * 128 + lane * 2 + 1], 0, 0);
    const uint4 m = ga.meta[g];
    const uint32_t n = (m.x >> 16) & 0xff;
    const uint32_t b0 = m.x & 0xff, b1 = (m.x >> 8) & 0xff;
    const uint32_t Bv[2] = {(b0 << 7) | (b0 << 23), (b1 << 7) | (b1 << 23)};
    uint32_t ESC[2] = {0u, 0u};
    const uint32_t ent[6] = {m.y & 0xffff, m.y >> 16, m.z & 0xffff, m.z >> 16, m.w & 0xffff, m.w >> 16};
    for (int s = 0; s < 6; s++)
        if (s < (int)n && n != 255 && (ent[s] & 31) == (uint32_t)lane) ESC[(ent[s] >> 5) & 1] = ((ent[s] >> 6) << 7) | ((ent[s] >> 6) << 23);
    const uint32_t* sw = (const uint32_t*)sv;
    const uint32_t* iw = (const uint32_t*)qv;
    for (int r = 0; r < 2; r++)
        for (int c = 0; c < NC; c++) {
            const int t = r * NC + c;
            uint32_t* o = out + ((size_t)(2 * g + r) * K + (c * 32 + lane) * 8) / 2;
            if (n == 255) {
                const uint32_t* f = (const uint32_t*)(ga.fb + (size_t)m.y * 2 * K + (size_t)r * K + (c * 32 + lane) * 8);
                for (int k = 0; k < 4; k++) o[k] = f[k];
            } else   // one 16 B store per lane ([ink-pf-rowfuse]: 4 x 4 B stores before)
                *(uint4*)o = make_uint4(dec_pair<0>(sw[2 * t], iw[t], Bv[r], ESC[r]), dec_pair<1>(sw[2 * t], iw[t], Bv[r], ESC[r]),
                                        dec_pair<2>(sw[2 * t + 1], iw[t], Bv[r], ESC[r]), dec_pair<3>(sw[2 * t + 1], iw[t], Bv[r], ESC[r]));
        }
}

// load-time check of a G15 matrix: decoded words vs the original BF16 words (review L: round trip at load)
__global__ void k_cmp_u32(const uint32_t* a, const uint32_t* b, size_t n, unsigned long long* bad) {
    unsigned long long c = 0;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) c += a[i] != b[i];
    if (c) atomicAdd(bad, c);
}
// argmax over the head GEMV's block maxima (ties: lower row)
struct A_argmax2 {
    const float* bmax;
    int n;
    int* out;
};
template <int NT = 1024>
__device__ __forceinline__ void b_argmax2(const A_argmax2& A_, const dim3 bid_, const dim3 gdim_) {
    const auto& bmax = A_.bmax;
    const auto& n = A_.n;
    const auto& out = A_.out;

    __shared__ float bv[32];
    __shared__ int bi[32];
    float v = -INFINITY;
    int id = 1 << 30;
    for (int i = threadIdx.x; i < n; i += NT) {
        const float x = bmax[2 * i];
        const int j = __float_as_int(bmax[2 * i + 1]);
        if (x > v || (x == v && j < id)) {
            v = x;
            id = j;
        }
    }
    for (int o = 16; o; o >>= 1) {
        const float v2 = __shfl_xor_sync(0xffffffffu, v, o);
        const int i2 = __shfl_xor_sync(0xffffffffu, id, o);
        if (v2 > v || (v2 == v && i2 < id)) {
            v = v2;
            id = i2;
        }
    }
    if ((threadIdx.x & 31) == 0) {
        bv[threadIdx.x >> 5] = v;
        bi[threadIdx.x >> 5] = id;
    }
    BSYNC(NT);
    if (threadIdx.x == 0) {
        for (int j = 1; j < NT / 32; j++)
            if (bv[j] > v || (bv[j] == v && bi[j] < id)) {
                v = bv[j];
                id = bi[j];
            }
        *out = id;
    }
}
__global__ void __launch_bounds__(1024) k_argmax2(const __grid_constant__ A_argmax2 A_) { b_argmax2<1024>(A_, blockIdx, gridDim); }


// ---- expert GEMV (decode form; prefill runs it per token). Codes decoded by bit placement into fp16:
// e2m1 code bits -> fp16 exponent / mantissa top bit: fp16 value = e2m1 value * 2^-14 (exact, 0.5 included);
// e4m3 byte -> fp16 (s, e, m placed at bits 15, 10.., 7..): fp16 value = e4m3 value * 2^-8 (subnormals exact).
__device__ __forceinline__ float2 fp4pair(uint32_t t) {   // elements of nibbles 0 and 4 of t
    const uint32_t h = ((t & 0x00070007u) << 9) | ((t & 0x00080008u) << 12);
    return __half22float2(*(const __half2*)&h);
}
__device__ __forceinline__ float2 fp8pair(uint32_t t) {   // bytes 0 and 2 of t
    const uint32_t h = ((t & 0x007f007fu) << 7) | ((t & 0x00800080u) << 8);
    return __half22float2(*(const __half2*)&h);
}
__device__ __forceinline__ float e4m3f(uint32_t b) {   // one e4m3 byte, exact
    return fp8pair(b).x * 256.f;
}
// 16 e2m1 values (8 bytes = 2 words) . x[0..15] (fp32), unscaled (* 2^-14)
__device__ __forceinline__ float fp4dot16(uint32_t w0, uint32_t w1, const float* x) {
    const float4 a = *(const float4*)x, b = *(const float4*)(x + 4), c = *(const float4*)(x + 8), d = *(const float4*)(x + 12);
    float2 p0 = fp4pair(w0), p1 = fp4pair(w0 >> 4), p2 = fp4pair(w0 >> 8), p3 = fp4pair(w0 >> 12);
    float s = p0.x * a.x + p1.x * a.y + p2.x * a.z + p3.x * a.w + p0.y * b.x + p1.y * b.y + p2.y * b.z + p3.y * b.w;
    p0 = fp4pair(w1), p1 = fp4pair(w1 >> 4), p2 = fp4pair(w1 >> 8), p3 = fp4pair(w1 >> 12);
    s += p0.x * c.x + p1.x * c.y + p2.x * c.z + p3.x * c.w + p0.y * d.x + p1.y * d.y + p2.y * d.z + p3.y * d.w;
    return s;
}
// 32 e2m1 values of a 16 B unit decoded (fp4pair order of fp4dot16: words x, y, z, w, pairs of nibbles k, k + 4)
__device__ __forceinline__ void fp4dec32(const uint4 c, float2* pd) {
    const uint32_t w[4] = {c.x, c.y, c.z, c.w};
#pragma unroll
    for (int j = 0; j < 4; j++)
#pragma unroll
        for (int k = 0; k < 4; k++) pd[4 * j + k] = fp4pair(w[j] >> (4 * k));
}
// fp4dot16 on decoded values (the same expression)
__device__ __forceinline__ float fp4dot16d(const float2* p, const float* x) {
    const float4 a = *(const float4*)x, b = *(const float4*)(x + 4), c = *(const float4*)(x + 8), d = *(const float4*)(x + 12);
    float s = p[0].x * a.x + p[1].x * a.y + p[2].x * a.z + p[3].x * a.w + p[0].y * b.x + p[1].y * b.y + p[2].y * b.z + p[3].y * b.w;
    s += p[4].x * c.x + p[5].x * c.y + p[6].x * c.z + p[7].x * c.w + p[4].y * d.x + p[5].y * d.y + p[6].y * d.z + p[7].y * d.w;
    return s;
}
// 16 e4m3 values (one uint4) . x[0..15], unscaled (* 2^-8)
__device__ __forceinline__ float fp8dot16(uint4 u, const float* x) {
    const uint32_t w[4] = {u.x, u.y, u.z, u.w};
    float s = 0.f;
#pragma unroll
    for (int j = 0; j < 4; j++) {
        const float4 a = *(const float4*)(x + 4 * j);
        const float2 p = fp8pair(w[j]), q = fp8pair(w[j] >> 8);
        s += p.x * a.x + q.x * a.y + p.y * a.z + q.y * a.w;
    }
    return s;
}
// float4 slot swizzle for the expert kernels' shared x / activations (review M2): lanes read slots 8 apart; the XOR
// spreads them over the 8 float4 bank groups of a 128 B row
__device__ __forceinline__ int swz4(int s) { return s ^ ((s >> 3) & 7); }
__device__ __forceinline__ float act_fn(float g, float u) {
    return 4.f * tanhf(g / 4.f) * (1.f / (1.f + expf(-g))) * 25.f * tanhf(u / 25.f);
}
// gate/up: grid (I/8, TOPK, T), warp -> intermediate index i (rows 2i gate, 2i+1 up). act = wts * f2[e] * act_fn
// (the routing weight and the down factor folded in, so the down kernel only sums).
// NVFP4 gate/up: a warp owns GP gate/up pairs (2 GP rows) and keeps its 32-element x blocks in registers across
// them (one x load per 2 GP code loads); grid (I / (8 GP), TOPK, T)
constexpr int GP = 2;   // A/B (before [ink-gu4-bal], which needs GP 2): 2 pairs per warp -4.4 % (512) / -3.9 % (4K) vs 4; 1: +0.5 %; 8: +0.1 %
struct A_gu4 {
    const uint8_t* cod;
    const uint8_t* scl;
    const float* f13;
    const float* f2;
    const float* x;
    const float* sc;
    const float* bias;
    int* ids;
    float* wts;
    float* act;
    int tfast;   // verify rows: grid (T, TOPK, chunks), rows fastest (rows sharing an expert read its chunk from L2)
    int ia = 0, ldi = I;   // [ink-tp-dec1] the rank's intermediate columns [ia, ia + ldi) (codes / scales hold only these)
};
template <int GP>
__device__ __forceinline__ void b_gu4(const A_gu4& A_, const dim3 bid_, const dim3 gdim_) {
    const auto& cod = A_.cod;
    const auto& scl = A_.scl;
    const auto& f13 = A_.f13;
    const auto& f2 = A_.f2;
    const auto& x = A_.x;
    const auto& sc = A_.sc;
    const auto& bias = A_.bias;
    const auto& ids = A_.ids;
    const auto& wts = A_.wts;
    const auto& act = A_.act;

    const int t = bid_.z, s = bid_.y, lane = threadIdx.x & 31;
    const int i0 = A_.ia + (bid_.x * 8 + (threadIdx.x >> 5)) * GP;
    __shared__ int esh[2];
    float* xs = xbuf_h();   // loaded while warp 0 selects the experts
    for (int c = threadIdx.x; c < H / 4; c += 256) ((float4*)xs)[swz4(c)] = ((const float4*)(x + (size_t)t * H))[c];
    int e;
    float wsel;
    if (sc) {
        block_topk(sc, bias, t, s, ids, wts, bid_.x == 0 && s == 0, esh);
        e = esh[0];
        wsel = __int_as_float(esh[1]);
    } else {   // selected by the router's last block
        e = ids[t * TOPK + s];
        wsel = wts[t * TOPK + s];
        BSYNC(256);   // x staging
    }
    const size_t r0 = (size_t)e * 2 * A_.ldi + 2 * (i0 - A_.ia);
    float acc[2 * GP];
#pragma unroll
    for (int r = 0; r < 2 * GP; r++) acc[r] = 0.f;
    // [ink-gu4-bal] H / 32 = 72 units per row: units 0..63 by all lanes for every row, the 8 x 2 GP tail units spread
    // over the lanes (lane -> row lane / 8, unit 64 + lane % 8) instead of 8 lanes doing all rows
    static_assert(H / 32 > 64 && 2 * GP * (H / 32 - 64) == 32, "tail units = lanes");
    constexpr int NIT = 2;
#pragma unroll
    for (int it = 0; it < NIT; it++) {
        const int b = it * 32 + lane;   // 16 B = 32 elements = 2 scale groups
        if (b < H / 32) {
            float xv[32];
#pragma unroll
            for (int k = 0; k < 8; k++) *(float4*)(xv + 4 * k) = ((const float4*)xs)[swz4(b * 8 + k)];
            uint4 c[2 * GP];
            uint16_t sc[2 * GP];
#pragma unroll
            for (int r = 0; r < 2 * GP; r++) {
                c[r] = *(const uint4*)(cod + (r0 + r) * (H / 2) + b * 16);
                sc[r] = *(const uint16_t*)(scl + (r0 + r) * (H / 16) + 2 * b);
            }
#pragma unroll
            for (int r = 0; r < 2 * GP; r++)
                acc[r] += fp4dot16(c[r].x, c[r].y, xv) * e4m3f(sc[r] & 255) + fp4dot16(c[r].z, c[r].w, xv + 16) * e4m3f(sc[r] >> 8);
        }
    }
    {
        const int rr = lane >> 3, b = 64 + (lane & 7);
        float xv[32];
#pragma unroll
        for (int k = 0; k < 8; k++) *(float4*)(xv + 4 * k) = ((const float4*)xs)[swz4(b * 8 + k)];
        const uint4 c = *(const uint4*)(cod + (r0 + rr) * (H / 2) + b * 16);
        const uint16_t scv = *(const uint16_t*)(scl + (r0 + rr) * (H / 16) + 2 * b);
        const float v = fp4dot16(c.x, c.y, xv) * e4m3f(scv & 255) + fp4dot16(c.z, c.w, xv + 16) * e4m3f(scv >> 8);
#pragma unroll
        for (int r = 0; r < 2 * GP; r++)
            if (r == rr) acc[r] += v;
    }
    const float fe = 16384.f * f13[e], wf = wsel * f2[e];
#pragma unroll
    for (int r = 0; r < 2 * GP; r++) acc[r] = warp_sum(acc[r]);
    if (lane < GP) {
        float g = acc[0], u = acc[1];
#pragma unroll
        for (int q = 1; q < GP; q++)
            if (lane == q) {
                g = acc[2 * q];
                u = acc[2 * q + 1];
            }
        act[((size_t)t * TOPK + s) * I + i0 + lane] = wf * act_fn(g * fe, u * fe);
    }
}
template <int GPT = GP>
__global__ void __launch_bounds__(256) k_gu4(const __grid_constant__ A_gu4 A_) {
    b_gu4<GPT>(A_, A_.tfast ? dim3(blockIdx.z, blockIdx.y, blockIdx.x) : dim3(blockIdx), gridDim);
}
// [ink-gu4-pair] verify gate / up: the routed (row, slot) entries of one expert taken two at a time (occurrence
// ranks 0+1, 2+3, .. in row order): the block of the even occurrence loads the expert chunk's codes once and runs the
// b_gu4 arithmetic for both rows (decoded values shared, every sum the same expression in the same order), the odd
// occurrence's block exits. Grid and per-output arithmetic as k_gu4mb (bitwise).
__device__ __forceinline__ void fp4dot16x2(uint32_t w0, uint32_t w1, const float* x, const float* y, float& sx, float& sy) {
    const float4 a = *(const float4*)x, b = *(const float4*)(x + 4), c = *(const float4*)(x + 8), d = *(const float4*)(x + 12);
    const float4 a2 = *(const float4*)y, b2 = *(const float4*)(y + 4), c2 = *(const float4*)(y + 8), d2 = *(const float4*)(y + 12);
    float2 p0 = fp4pair(w0), p1 = fp4pair(w0 >> 4), p2 = fp4pair(w0 >> 8), p3 = fp4pair(w0 >> 12);
    float s = p0.x * a.x + p1.x * a.y + p2.x * a.z + p3.x * a.w + p0.y * b.x + p1.y * b.y + p2.y * b.z + p3.y * b.w;
    float t = p0.x * a2.x + p1.x * a2.y + p2.x * a2.z + p3.x * a2.w + p0.y * b2.x + p1.y * b2.y + p2.y * b2.z + p3.y * b2.w;
    p0 = fp4pair(w1), p1 = fp4pair(w1 >> 4), p2 = fp4pair(w1 >> 8), p3 = fp4pair(w1 >> 12);
    s += p0.x * c.x + p1.x * c.y + p2.x * c.z + p3.x * c.w + p0.y * d.x + p1.y * d.y + p2.y * d.z + p3.y * d.w;
    t += p0.x * c2.x + p1.x * c2.y + p2.x * c2.z + p3.x * c2.w + p0.y * d2.x + p1.y * d2.y + p2.y * d2.z + p3.y * d2.w;
    sx = s;
    sy = t;
}
// the pair body: entries me (leader) and nx (its partner, same expert), chunk cz of I / (8 GP)
__device__ __forceinline__ void b_gu4pr(const A_gu4& A_, const int me, const int nx, const int cz) {
    constexpr int GP_ = GP;
    const auto& cod = A_.cod;
    const auto& scl = A_.scl;
    const int t = me / TOPK, s = me % TOPK, lane = threadIdx.x & 31;
    const int i0 = A_.ia + (cz * 8 + (threadIdx.x >> 5)) * GP_;
    const int e = A_.ids[me];
    const int t2 = nx / TOPK, s2 = nx % TOPK;
    float* xs = xbuf_h();
    float* xs2 = xbuf2_h();
    for (int c = threadIdx.x; c < H / 4; c += 256) {
        ((float4*)xs)[swz4(c)] = ((const float4*)(A_.x + (size_t)t * H))[c];
        ((float4*)xs2)[swz4(c)] = ((const float4*)(A_.x + (size_t)t2 * H))[c];
    }
    BSYNC(256);
    const size_t r0 = (size_t)e * 2 * A_.ldi + 2 * (i0 - A_.ia);
    float acc[2 * GP_], acc2[2 * GP_];
#pragma unroll
    for (int r = 0; r < 2 * GP_; r++) acc[r] = acc2[r] = 0.f;
#pragma unroll
    for (int it = 0; it < 2; it++) {
        const int b = it * 32 + lane;
        if (b < H / 32) {
            float xv[32], yv[32];
#pragma unroll
            for (int k = 0; k < 8; k++) {
                *(float4*)(xv + 4 * k) = ((const float4*)xs)[swz4(b * 8 + k)];
                *(float4*)(yv + 4 * k) = ((const float4*)xs2)[swz4(b * 8 + k)];
            }
            uint4 c[2 * GP_];
            uint16_t sc[2 * GP_];
#pragma unroll
            for (int r = 0; r < 2 * GP_; r++) {
                c[r] = *(const uint4*)(cod + (r0 + r) * (H / 2) + b * 16);
                sc[r] = *(const uint16_t*)(scl + (r0 + r) * (H / 16) + 2 * b);
            }
#pragma unroll
            for (int r = 0; r < 2 * GP_; r++) {
                float d0, d1, f0, f1;
                fp4dot16x2(c[r].x, c[r].y, xv, yv, d0, f0);
                fp4dot16x2(c[r].z, c[r].w, xv + 16, yv + 16, d1, f1);
                acc[r] += d0 * e4m3f(sc[r] & 255) + d1 * e4m3f(sc[r] >> 8);
                acc2[r] += f0 * e4m3f(sc[r] & 255) + f1 * e4m3f(sc[r] >> 8);
            }
        }
    }
    {
        const int rr = lane >> 3, b = 64 + (lane & 7);
        float xv[32], yv[32];
#pragma unroll
        for (int k = 0; k < 8; k++) {
            *(float4*)(xv + 4 * k) = ((const float4*)xs)[swz4(b * 8 + k)];
            *(float4*)(yv + 4 * k) = ((const float4*)xs2)[swz4(b * 8 + k)];
        }
        const uint4 c = *(const uint4*)(cod + (r0 + rr) * (H / 2) + b * 16);
        const uint16_t scv = *(const uint16_t*)(scl + (r0 + rr) * (H / 16) + 2 * b);
        float d0, d1, f0, f1;
        fp4dot16x2(c.x, c.y, xv, yv, d0, f0);
        fp4dot16x2(c.z, c.w, xv + 16, yv + 16, d1, f1);
        const float v = d0 * e4m3f(scv & 255) + d1 * e4m3f(scv >> 8);
        const float v2 = f0 * e4m3f(scv & 255) + f1 * e4m3f(scv >> 8);
#pragma unroll
        for (int r = 0; r < 2 * GP_; r++)
            if (r == rr) {
                acc[r] += v;
                acc2[r] += v2;
            }
    }
    const float fe = 16384.f * A_.f13[e];
#pragma unroll
    for (int r = 0; r < 2 * GP_; r++) {
        acc[r] = warp_sum(acc[r]);
        acc2[r] = warp_sum(acc2[r]);
    }
    if (lane < GP_) {
        float g = acc[0], u = acc[1], g2 = acc2[0], u2 = acc2[1];
#pragma unroll
        for (int q = 1; q < GP_; q++)
            if (lane == q) {
                g = acc[2 * q];
                u = acc[2 * q + 1];
                g2 = acc2[2 * q];
                u2 = acc2[2 * q + 1];
            }
        A_.act[((size_t)t * TOPK + s) * I + i0 + lane] = A_.wts[me] * A_.f2[e] * act_fn(g * fe, u * fe);
        A_.act[((size_t)t2 * TOPK + s2) * I + i0 + lane] = A_.wts[nx] * A_.f2[e] * act_fn(g2 * fe, u2 * fe);
    }
}
__global__ void __launch_bounds__(256, 2) k_gu4p(const __grid_constant__ A_gu4 A_) {
    const int t = blockIdx.x, s = blockIdx.y, lane = threadIdx.x & 31, nr = gridDim.x;
    const int me = t * TOPK + s, e = A_.ids[me];
    // occurrence rank of (t, s) among the entries of expert e (row order), and the next occurrence
    int rk, nx;
    {   // each warp: lane c holds entries c and c + 32 (T * TOPK <= 64), ballots give the rank and the next one
        const int n = nr * TOPK;
        const unsigned m0 = __ballot_sync(0xffffffffu, lane < n && A_.ids[lane] == e);
        const unsigned m1 = __ballot_sync(0xffffffffu, lane + 32 < n && A_.ids[lane + 32] == e);
        const unsigned long long mm = ((unsigned long long)m1 << 32) | m0;
        rk = __popcll(mm & ((1ull << me) - 1ull));
        const unsigned long long above = me < 63 ? mm & ~((2ull << me) - 1ull) : 0ull;
        nx = above ? __ffsll((long long)above) - 1 : -1;
    }
    if (rk & 1) return;   // served by the block of the previous occurrence
    if (nx < 0) {   // a single entry: the per-row b_gu4 (same arithmetic)
        b_gu4<GP>(A_, dim3(blockIdx.z, s, t), gridDim);
        return;
    }
    b_gu4pr(A_, me, nx, blockIdx.z);
}
// [ink-b-xord] per-application order of the routed entries (one 256-thread block, n = T x TOPK <= 256): entries
// counting-sorted by expert id (entry order inside an expert) -> srt; pair jobs for gate / up (2 consecutive users of
// an expert: leader | (partner + 1) << 16), MU-user jobs for down (sorted position | users << 16), in expert-id order.
// The job counts go to cnt[0] / cnt[1]; the kernels' grids are sized for n jobs (graph-safe), extra blocks exit.
struct OrdBuf {
    int cnt[4];
    int srt[MROW * TOPK];
    int pj[MROW * TOPK];
    int dj[MROW * TOPK];
};
__global__ void __launch_bounds__(256) k_order(const int* __restrict__ ids, const int n, const int mu, OrdBuf* __restrict__ ob) {
    __shared__ int sid[MROW * TOPK], cnt[NE], ps[NE], pp[NE], pd[NE], ssrt[MROW * TOPK];
    const int i = threadIdx.x;
    for (int k = i; k < NE; k += 256) cnt[k] = 0;
    __syncthreads();
    int e = -1;
    if (i < n) {
        e = ids[i];
        sid[i] = e;
        atomicAdd(&cnt[e], 1);
    }
    __syncthreads();
    if (i < 32) {   // exclusive scans over the 128 experts (4 per lane): sorted positions, pair jobs, down jobs
        int a[4], b[4], c[4], sa = 0, sb = 0, sc = 0;
#pragma unroll
        for (int q = 0; q < 4; q++) {
            const int m = cnt[4 * i + q];
            a[q] = sa; b[q] = sb; c[q] = sc;
            sa += m; sb += (m + 1) >> 1; sc += (m + mu - 1) / mu;
        }
        int xa = sa, xb = sb, xc = sc;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            const int ya = __shfl_up_sync(0xffffffffu, xa, o), yb = __shfl_up_sync(0xffffffffu, xb, o),
                      yc = __shfl_up_sync(0xffffffffu, xc, o);
            if (i >= o) { xa += ya; xb += yb; xc += yc; }
        }
        xa -= sa; xb -= sb; xc -= sc;   // exclusive
#pragma unroll
        for (int q = 0; q < 4; q++) {
            ps[4 * i + q] = xa + a[q];
            pp[4 * i + q] = xb + b[q];
            pd[4 * i + q] = xc + c[q];
        }
        if (i == 31) {
            ob->cnt[0] = xb + sb;
            ob->cnt[1] = xc + sc;
        }
    }
    __syncthreads();
    int rk = 0, pos = 0;
    if (i < n) {
        for (int j = 0; j < i; j++) rk += sid[j] == e;
        pos = ps[e] + rk;
        ssrt[pos] = i;
        ob->srt[pos] = i;
    }
    __syncthreads();
    if (i < n) {
        const int m = cnt[e];
        if (!(rk & 1)) ob->pj[pp[e] + (rk >> 1)] = i | ((rk + 1 < m ? ssrt[pos + 1] + 1 : 0) << 16);
        if (rk % mu == 0) ob->dj[pd[e] + rk / mu] = pos | (min(mu, m - rk) << 16);
    }
}
// [ink-b-xord] gate / up over the ordered pair jobs: grid (n jobs, I / (8 GP)), jobs fastest (the users of an expert
// are neighbours in launch order: L2). Per output the k_gu4p arithmetic (pair body / b_gu4 single), bitwise.
__global__ void __launch_bounds__(256, 2) k_gu4o(const __grid_constant__ A_gu4 A_, const OrdBuf* __restrict__ ob) {
    if ((int)blockIdx.x >= ob->cnt[0]) return;
    const int jb = ob->pj[blockIdx.x], me = jb & 0xffff, nx = (jb >> 16) - 1;
    if (nx < 0) b_gu4<GP>(A_, dim3(blockIdx.y, me % TOPK, me / TOPK), gridDim);
    else b_gu4pr(A_, me, nx, blockIdx.y);
}
template <int MB>
__global__ void __launch_bounds__(256, MB) k_gu4mb(const __grid_constant__ A_gu4 A_) {
    b_gu4<GP>(A_, A_.tfast ? dim3(blockIdx.z, blockIdx.y, blockIdx.x) : dim3(blockIdx), gridDim);
}

struct A_gu8 {
    const uint8_t* cod;
    const uint8_t* scl;
    const float* x;
    const float* sc;
    const float* bias;
    int* ids;
    float* wts;
    float* act;
    int tfast;
    int ia = 0, ldi = I;   // [ink-tp-dec1] as A_gu4
};
__device__ __forceinline__ void b_gu8(const A_gu8& A_, const dim3 bid_, const dim3 gdim_) {
    const auto& cod = A_.cod;
    const auto& scl = A_.scl;
    const auto& x = A_.x;
    const auto& sc = A_.sc;
    const auto& bias = A_.bias;
    const auto& ids = A_.ids;
    const auto& wts = A_.wts;
    const auto& act = A_.act;

    const int t = bid_.z, s = bid_.y, lane = threadIdx.x & 31, i = A_.ia + bid_.x * 8 + (threadIdx.x >> 5);
    __shared__ int esh[2];
    int e;
    float wsel;
    if (sc) {
        block_topk(sc, bias, t, s, ids, wts, bid_.x == 0 && s == 0, esh);
        e = esh[0];
        wsel = __int_as_float(esh[1]);
    } else {   // selected by the router's last block
        e = ids[t * TOPK + s];
        wsel = wts[t * TOPK + s];
        BSYNC(256);   // x staging
    }
    const float* xs = x + (size_t)t * H;
    const size_t r0 = (size_t)e * 2 * A_.ldi + 2 * (i - A_.ia);
    float d[2];
#pragma unroll
    for (int g = 0; g < 2; g++) {
        const uint8_t* cr = cod + (r0 + g) * H;
        const uint8_t* sr = scl + (r0 + g) * (H / 32);
        float acc = 0.f;
#pragma unroll
        for (int it = 0; it < 5; it++) {
            const int b = it * 32 + lane;   // 16 B = 16 elements, scale per 32 = b / 2
            if (b < H / 16) acc += fp8dot16(*(const uint4*)(cr + b * 16), xs + b * 16) * __uint_as_float((uint32_t)sr[b >> 1] << 23);
        }
        d[g] = warp_sum(acc) * 256.f;
    }
    if (lane == 0) act[((size_t)t * TOPK + s) * I + i] = wsel * act_fn(d[0], d[1]);
}
__global__ void __launch_bounds__(256) k_gu8(const __grid_constant__ A_gu8 A_) {
    b_gu8(A_, A_.tfast ? dim3(blockIdx.z, blockIdx.y, blockIdx.x) : dim3(blockIdx), gridDim);
}
// [ink-tp-x8sh] MXFP8 gate / up of row steps (> 8 rows) over user jobs (expert, <= 4 users; from k_order in
// expert-id order; plain and <= 8 rows keep b_gu8: the CW form loses there, +0.5 % plain). b_gu8 is L1-bound (ncu: L1/TEX 98 %, DRAM 32 %): every warp reads 4 B of fp32 x per
// weight byte and uses it for 2 rows. Here a warp owns CW columns (2 CW rows) and each user's x unit is read once
// for all of them; the expert rows' 16 B units are read once for all users. Per output the b_gu8 expression in the
// same order (sum over the 5 lane steps, warp_sum, x 256): bitwise. grid (jobs, ldi / (8 CW)); extra blocks exit.
constexpr int X8RW = 4;   // [ink-tp-x8sh] hidden rows per warp of k_dn8r
constexpr int X8CW = 4;   // columns per warp (TP_UNIT 32 / (8 X8CW) must divide)
template <int CW>
__global__ void __launch_bounds__(256, 2) k_gu8s(const __grid_constant__ A_gu8 A_, const OrdBuf* __restrict__ ob) {
    if ((int)blockIdx.x >= ob->cnt[1]) return;
    const int jb = ob->dj[blockIdx.x], pos = jb & 0xffff, nu = jb >> 16;
    const int lane = threadIdx.x & 31, i0 = A_.ia + (blockIdx.y * 8 + (threadIdx.x >> 5)) * CW;
    int me[4];
#pragma unroll
    for (int u = 0; u < 4; u++) me[u] = ob->srt[pos + min(u, nu - 1)];
    const int e = A_.ids[me[0]];
    const size_t r0 = (size_t)e * 2 * A_.ldi + 2 * (i0 - A_.ia);
    float acc[2 * CW][4];
#pragma unroll
    for (int r = 0; r < 2 * CW; r++)
#pragma unroll
        for (int u = 0; u < 4; u++) acc[r][u] = 0.f;
#pragma unroll
    for (int it = 0; it < 5; it++) {
        const int b = it * 32 + lane;
        if (b < H / 16) {
            uint4 w[2 * CW];
            float sv[2 * CW];
#pragma unroll
            for (int r = 0; r < 2 * CW; r++) {
                w[r] = *(const uint4*)(A_.cod + (r0 + r) * H + b * 16);
                sv[r] = __uint_as_float((uint32_t)A_.scl[(r0 + r) * (H / 32) + (b >> 1)] << 23);
            }
#pragma unroll
            for (int u = 0; u < 4; u++)
                if (u < nu) {
                    float xr[16];
                    const float4* xp = (const float4*)(A_.x + (size_t)(me[u] / TOPK) * H + b * 16);
#pragma unroll
                    for (int q = 0; q < 4; q++) *(float4*)(xr + 4 * q) = xp[q];
#pragma unroll
                    for (int r = 0; r < 2 * CW; r++) acc[r][u] += fp8dot16(w[r], xr) * sv[r];
                }
        }
    }
#pragma unroll
    for (int u = 0; u < 4; u++)
        if (u < nu) {
            float g[CW], v[CW];
#pragma unroll
            for (int c = 0; c < CW; c++) {
                g[c] = warp_sum(acc[2 * c][u]) * 256.f;
                v[c] = warp_sum(acc[2 * c + 1][u]) * 256.f;
            }
            if (lane < CW) {
                float gg = g[0], vv = v[0];
#pragma unroll
                for (int c = 1; c < CW; c++)
                    if (lane == c) {
                        gg = g[c];
                        vv = v[c];
                    }
                A_.act[(size_t)me[u] * I + i0 + lane] = A_.wts[me[u]] * act_fn(gg, vv);
            }
        }
}

// down: grid (H/8, T), warp -> hidden row r; the lanes walk the 8 experts' rows as one flat list of 16 B blocks.
// NVFP4 down: a warp owns DR hidden rows; the lanes walk the 8 experts' rows as one flat list of 16 B blocks and
// reuse each 32-element act block (registers) across the DR rows. grid (H / (8 DR), T)
constexpr int DR = 4;
struct A_dn4 {
    const uint8_t* cod;
    const uint8_t* scl;
    const float* act;
    const int* ids;
    float* out;
    float* cst;
    const float* cw;
    const float* add;
    int tfast;   // verify rows: grid (T, H / 32), rows fastest
    int ua = 0, nu = I / 32;   // [ink-tp-dec1] the rank's 32-column units [ua, ua + nu) of I (codes / scales hold only these)
    TpPush pp{};               // [ink-tpf-ppush] k_dn4p<.., true>: the partial row's values also go to the peers' receive words
};
template <int DR, bool SMA = false>
__device__ __forceinline__ void b_dn4(const A_dn4& A_, const dim3 bid_, const dim3 gdim_) {
    const auto& cod = A_.cod;
    const auto& scl = A_.scl;
    const auto& act = A_.act;
    const auto& ids = A_.ids;
    const auto& out = A_.out;
    const auto& cst = A_.cst;
    const auto& cw = A_.cw;
    const auto& add = A_.add;

    const int t = bid_.y, lane = threadIdx.x & 31, r0 = (bid_.x * 8 + (threadIdx.x >> 5)) * DR;
    constexpr int NB = I / 32;   // 36 blocks per row
    __shared__ int sid[TOPK];
    if (threadIdx.x < TOPK) sid[threadIdx.x] = ids[t * TOPK + threadIdx.x];
    const float* actb = act + (size_t)t * TOPK * I;
    if (SMA) {   // the token's 8 x 1,152 activations staged once per block
        __shared__ __align__(16) float as[TOPK * I];
        for (int k = threadIdx.x; k < TOPK * I / 4; k += 256) ((float4*)as)[swz4(k)] = ((const float4*)actb)[k];
        actb = as;
    }
    BSYNC(256);
    float acc[DR];
#pragma unroll
    for (int r = 0; r < DR; r++) acc[r] = 0.f;
#pragma unroll 3
    for (int j = lane; j < TOPK * NB; j += 32) {
        const int s = j / NB, b = j % NB;
        const size_t row = (size_t)sid[s] * H + r0;
        uint4 c[DR];
        uint16_t sc[DR];
#pragma unroll
        for (int r = 0; r < DR; r++) {
            c[r] = *(const uint4*)(cod + (row + r) * (I / 2) + b * 16);
            sc[r] = *(const uint16_t*)(scl + (row + r) * (I / 16) + 2 * b);
        }
        float av[32];
        const float* ap = actb + (size_t)s * I + b * 32;
#pragma unroll
        if (SMA) {
            const int s0 = (s * I + b * 32) / 4;
#pragma unroll
            for (int k = 0; k < 8; k++) *(float4*)(av + 4 * k) = ((const float4*)actb)[swz4(s0 + k)];
        } else
#pragma unroll
            for (int k = 0; k < 8; k++) *(float4*)(av + 4 * k) = *(const float4*)(ap + 4 * k);
#pragma unroll
        for (int r = 0; r < DR; r++)
            acc[r] += fp4dot16(c[r].x, c[r].y, av) * e4m3f(sc[r] & 255) + fp4dot16(c[r].z, c[r].w, av + 16) * e4m3f(sc[r] >> 8);
    }
#pragma unroll
    for (int r = 0; r < DR; r++) acc[r] = warp_sum(acc[r]);
    if (lane < DR) {
        float v = acc[0];
#pragma unroll
        for (int q = 1; q < DR; q++)
            if (lane == q) v = acc[q];
        if (cst) out[r0 + lane] = conv1(v * 16384.f, r0 + lane, cst, cw, add);   // T = 1: post-MoE conv folded
        else out[(size_t)t * H + r0 + lane] = v * 16384.f;
    }
}
template <int DRT = DR, bool SMA = false>
__global__ void __launch_bounds__(256) k_dn4(const __grid_constant__ A_dn4 A_) {
    b_dn4<DRT, SMA>(A_, A_.tfast ? dim3(blockIdx.y, blockIdx.x, 0) : dim3(blockIdx), gridDim);
}

// [ink-dn4-ks] NVFP4 down, RG row groups of DR rows per block, the flat (expert, block) list split over KS warps per
// group, parts added in warp order. KS = 1: per row exactly the b_dn4 arithmetic (bitwise). Block 32 RG KS threads.
template <int DR, int RG, int KS>
__global__ void __launch_bounds__(32 * RG * KS) k_dn4s(const __grid_constant__ A_dn4 A_) {
    const dim3 bid_ = A_.tfast ? dim3(blockIdx.y, blockIdx.x, 0) : dim3(blockIdx);
    constexpr int NT = 32 * RG * KS, NB = I / 32, NJ = TOPK * NB / KS;
    static_assert(TOPK * NB % KS == 0, "parts");
    const int t = bid_.y, lane = threadIdx.x & 31, w = threadIdx.x >> 5, rg = w / KS, kp = w % KS;
    const int r0 = (bid_.x * RG + rg) * DR;
    __shared__ int sid[TOPK];
    if (threadIdx.x < TOPK) sid[threadIdx.x] = A_.ids[t * TOPK + threadIdx.x];
    __shared__ __align__(16) float as[TOPK * I];
    const float* actb = A_.act + (size_t)t * TOPK * I;
    for (int k = threadIdx.x; k < TOPK * I / 4; k += NT) ((float4*)as)[swz4(k)] = ((const float4*)actb)[k];
    __syncthreads();
    float acc[DR];
#pragma unroll
    for (int r = 0; r < DR; r++) acc[r] = 0.f;
#pragma unroll 3
    for (int j = kp * NJ + lane; j < (kp + 1) * NJ; j += 32) {
        const int s = j / NB, b = j % NB;
        const size_t row = (size_t)sid[s] * H + r0;
        uint4 c[DR];
        uint16_t sc[DR];
#pragma unroll
        for (int r = 0; r < DR; r++) {
            c[r] = *(const uint4*)(A_.cod + (row + r) * (I / 2) + b * 16);
            sc[r] = *(const uint16_t*)(A_.scl + (row + r) * (I / 16) + 2 * b);
        }
        float av[32];
        const int s0 = (s * I + b * 32) / 4;
#pragma unroll
        for (int k = 0; k < 8; k++) *(float4*)(av + 4 * k) = ((const float4*)as)[swz4(s0 + k)];
#pragma unroll
        for (int r = 0; r < DR; r++)
            acc[r] += fp4dot16(c[r].x, c[r].y, av) * e4m3f(sc[r] & 255) + fp4dot16(c[r].z, c[r].w, av + 16) * e4m3f(sc[r] >> 8);
    }
    __shared__ float part[RG][KS][DR];
#pragma unroll
    for (int r = 0; r < DR; r++) {
        const float v = warp_sum(acc[r]);
        if (lane == 0) part[rg][kp][r] = v;
    }
    __syncthreads();
    if (threadIdx.x < RG * DR) {
        const int g = threadIdx.x / DR, r = threadIdx.x % DR, row = (bid_.x * RG + g) * DR + r;
        float v = part[g][0][r];
#pragma unroll
        for (int k = 1; k < KS; k++) v += part[g][k][r];
        if (A_.cst) A_.out[row] = conv1(v * 16384.f, row, A_.cst, A_.cw, A_.add);
        else A_.out[(size_t)t * H + row] = v * 16384.f;
    }
}

// [ink-b-dnsh] NVFP4 down with the decode shared by up to MU users of one expert: grid (n jobs, H / 64), jobs (sorted
// position, users) from k_order in expert-id order. Warp = 8 hidden rows, lane = row lane / 4, units (lane % 4) + 4 q
// (q < 9); per unit and user the b_dn4 unit expression; the 4 lanes of a row are added (xor 1, 2) into the entry's
// partial part[entry][row] (unscaled). k_dsum adds a row's 8 partials in slot order (x 16384).
// one user-count instance: the lane's RL rows (RS apart) x M users; per unit and half: RL rows decoded once, each
// user's 16 act values loaded once for the RL rows
template <int M, int RL, int RS>
__device__ __forceinline__ void b_dn4o(const uint8_t* cr, const uint8_t* sr, const float* as, const int bl, const int row,
                                       const int* sen, float* part, const int pst = H, const int ua = 0, const int nu = I / 32) {
    float acc[RL][M];
#pragma unroll
    for (int r = 0; r < RL; r++)
#pragma unroll
        for (int u = 0; u < M; u++) acc[r][u] = 0.f;
#pragma unroll 3
    for (int q = 0; bl + 4 * q < nu; q++) {   // [ink-tp-dec1] local unit b, global unit ua + b (nu = 36: the 9 units per lane)
        const int b = bl + 4 * q;
        uint4 c[RL];
        uint16_t sc[RL];
#pragma unroll
        for (int r = 0; r < RL; r++) {
            c[r] = *(const uint4*)(cr + (size_t)r * RS * (nu * 16) + b * 16);
            sc[r] = *(const uint16_t*)(sr + (size_t)r * RS * (nu * 2) + 2 * b);
        }
        float d0[RL][M];
#pragma unroll
        for (int h = 0; h < 2; h++) {
            float2 pd[RL][8];
#pragma unroll
            for (int r = 0; r < RL; r++) {
                const uint32_t w0 = h ? c[r].z : c[r].x, w1 = h ? c[r].w : c[r].y;
#pragma unroll
                for (int k = 0; k < 4; k++) {
                    pd[r][k] = fp4pair(w0 >> (4 * k));
                    pd[r][4 + k] = fp4pair(w1 >> (4 * k));
                }
            }
#pragma unroll
            for (int u = 0; u < M; u++) {
                float av[16];
                const float4* ap = (const float4*)as + u * (I / 4) + (ua + b) + 4 * h * (I / 32);
#pragma unroll
                for (int k = 0; k < 4; k++) *(float4*)(av + 4 * k) = ap[k * (I / 32)];
#pragma unroll
                for (int r = 0; r < RL; r++) {
                    const float d = fp4dot16d(pd[r], av);
                    if (h == 0) d0[r][u] = d;
                    else acc[r][u] += d0[r][u] * e4m3f(sc[r] & 255) + d * e4m3f(sc[r] >> 8);
                }
            }
        }
    }
#pragma unroll
    for (int r = 0; r < RL; r++)
#pragma unroll
        for (int u = 0; u < M; u++) {
            float v = acc[r][u];
            v += __shfl_xor_sync(0xffffffffu, v, 1);
            v += __shfl_xor_sync(0xffffffffu, v, 2);
            if (bl == 0) part[(size_t)sen[u] * pst + row + r * RS] = v;
        }
}
template <int MU, int RL = 2>
__global__ void __launch_bounds__(256) k_dn4o(const __grid_constant__ A_dn4 A_, const OrdBuf* __restrict__ ob, float* __restrict__ part) {
    if ((int)blockIdx.x >= ob->cnt[1]) return;
    const int jb = ob->dj[blockIdx.x], p0 = jb & 0xffff, m = jb >> 16;
    __shared__ int sen[MU];
    __shared__ __align__(16) float as[MU * I];
    if (threadIdx.x < MU) sen[threadIdx.x] = threadIdx.x < m ? ob->srt[p0 + threadIdx.x] : 0;
    __syncthreads();
    // act of user u as [u][k][unit] float4 (k = float4 of the 32-element unit): the 4 units a warp reads at once are
    // neighbours (no bank conflict, constant offsets per k)
    // [ink-tp-x8sh] only the rank's units [ua, ua + nu) are staged (TP2: half of I; the same layout positions)
    for (int f = threadIdx.x; f < m * A_.nu * 8; f += 256) {
        const int u = f / (A_.nu * 8), g = f - u * (A_.nu * 8) + 8 * A_.ua;
        ((float4*)as)[u * (I / 4) + (g & 7) * (I / 32) + (g >> 3)] = ((const float4*)(A_.act + (size_t)sen[u] * I))[g];
    }
    const int e = A_.ids[sen[0]];
    const int lane = threadIdx.x & 31, rr = lane >> 2, bl = lane & 3;
    const int row = blockIdx.y * (64 * RL) + (threadIdx.x >> 5) * (8 * RL) + rr;   // rows row, row + 8, ..
    const uint8_t* cr = A_.cod + ((size_t)e * H + row) * (A_.nu * 16);   // [ink-tp-dec1] the rank's units
    const uint8_t* sr = A_.scl + ((size_t)e * H + row) * (A_.nu * 2);
    __syncthreads();
    switch (m) {   // the user count as a compile-time bound (no predicated bodies for absent users)
#define DN4O_CASE(M_) case M_: if constexpr (M_ <= MU) b_dn4o<M_, RL, 8>(cr, sr, as, bl, row, sen, part, H, A_.ua, A_.nu); break;
    DN4O_CASE(1) DN4O_CASE(2) DN4O_CASE(3) DN4O_CASE(4) DN4O_CASE(5) DN4O_CASE(6) DN4O_CASE(7) DN4O_CASE(8)
#undef DN4O_CASE
    }
}
// [ink-b-dnsh] plain step (T = 1) in the k_dn4o order: block = 8 RL hidden rows, warp w = slot w (the b_dn4o<1, 2, 8>
// lane arithmetic), partials in smem, the 8 slots added in slot order (x 16384), post-MoE conv folded when cst.
template <int RL, int G = 1, bool PP = false>   // PP: [ink-tpf-ppush] TP producer push (the same arithmetic)
__global__ void __launch_bounds__(256 * G) k_dn4p(const __grid_constant__ A_dn4 A_) {
    constexpr int NR = 8 * RL * G, NT = 256 * G;   // hidden rows per block; warp w: slot w % 8, row group w / 8
    unsigned ptag = 0u;
    TpConvPre cpre;
    if constexpr (PP) {
        ptag = tp_push_tag(A_.pp);
        if (A_.pp.np && threadIdx.x < NR) tp_conv_pre(A_.pp, blockIdx.x * NR + threadIdx.x, cpre);   // [ink-tpf-pred]
    }
    __shared__ __align__(16) float as[TOPK * I];
    __shared__ float ps[TOPK * NR];
    __shared__ int sid[TOPK];
    if (threadIdx.x < TOPK) sid[threadIdx.x] = A_.ids[threadIdx.x];
    for (int f = threadIdx.x; f < TOPK * I / 4; f += NT) {
        const int u = f / (I / 4), g = f - u * (I / 4);
        ((float4*)as)[u * (I / 4) + (g & 7) * (I / 32) + (g >> 3)] = ((const float4*)A_.act)[f];
    }
    __syncthreads();
    const int w = (threadIdx.x >> 5) & 7, rg = threadIdx.x >> 8, lane = threadIdx.x & 31, rr = lane >> 2, bl = lane & 3;
    const int lrow = rg * 8 * RL + rr, row = blockIdx.x * NR + lrow;
    const uint8_t* cr = A_.cod + ((size_t)sid[w] * H + row) * (A_.nu * 16);
    const uint8_t* sr = A_.scl + ((size_t)sid[w] * H + row) * (A_.nu * 2);
    const int sw = w;
    b_dn4o<1, RL, 8>(cr, sr, as + w * I, bl, lrow, &sw, ps, NR, A_.ua, A_.nu);
    __syncthreads();
    if (threadIdx.x < NR) {
        const int r = threadIdx.x, orow = blockIdx.x * NR + r;
        float v = ps[r];
#pragma unroll
        for (int q = 1; q < TOPK; q++) v += ps[q * NR + r];
        if (A_.cst) A_.out[orow] = conv1(v * 16384.f, orow, A_.cst, A_.cw, A_.add);
        else {
            if constexpr (PP) {
                tp_push1(A_.pp, ptag, orow, v * 16384.f);
                if (A_.pp.np) {   // [ink-tpf-pred] sum + conv here
                    float sm;
                    if (tp_reduce1(A_.pp, ptag, orow, v * 16384.f, sm)) tp_conv_fin(A_.pp, orow, sm, cpre);
                } else
                    A_.out[orow] = v * 16384.f;
            } else
                A_.out[orow] = v * 16384.f;
        }
    }
}
// row t's down output = its 8 slot partials in slot order (x 16384); grid (H / 256, T)
__global__ void __launch_bounds__(256) k_dsum(const float* __restrict__ part, float* __restrict__ out) {
    const int t = blockIdx.y, r = blockIdx.x * 256 + threadIdx.x;
    const float* p = part + (size_t)t * TOPK * H + r;
    float v = p[0];
#pragma unroll
    for (int s = 1; s < TOPK; s++) v += p[(size_t)s * H];
    out[(size_t)t * H + r] = v * 16384.f;
}

#include "ink_tcx.cuh"   // [ink-tp-tcx]

struct A_dn8 {
    const uint8_t* cod;
    const uint8_t* scl;
    const float* act;
    const int* ids;
    float* out;
    float* cst;
    const float* cw;
    const float* add;
    int tfast;
    int ua = 0, nu = I / 32;   // [ink-tp-dec1] as A_dn4
    TpPush pp{};               // [ink-tpf-ppush] k_dn8pp
};
template <bool PP = false>
__device__ __forceinline__ void b_dn8(const A_dn8& A_, const dim3 bid_, const dim3 gdim_) {
    const auto& cod = A_.cod;
    const auto& scl = A_.scl;
    const auto& act = A_.act;
    const auto& ids = A_.ids;
    const auto& out = A_.out;
    const auto& cst = A_.cst;
    const auto& cw = A_.cw;
    const auto& add = A_.add;

    const int t = bid_.y, lane = threadIdx.x & 31, r = bid_.x * 8 + (threadIdx.x >> 5);
    const int NB = 2 * A_.nu;   // 72 blocks of 16 per row (the rank's: 2 nu, from column 32 ua)
    unsigned ptag = 0u;
    TpConvPre cpre;
    if constexpr (PP) {
        ptag = tp_push_tag(A_.pp);
        if (A_.pp.np && lane == 0) tp_conv_pre(A_.pp, r, cpre);   // [ink-tpf-pred]
    }
    __shared__ int sid[TOPK];
    if (threadIdx.x < TOPK) sid[threadIdx.x] = ids[t * TOPK + threadIdx.x];
    BSYNC(256);
    float acc = 0.f;
    int s = lane / NB, b = lane % NB;   // (slot, block) of j = lane + 32 k, stepped without a division
#pragma unroll 6
    for (int j = lane; j < TOPK * NB; j += 32) {
        const size_t row = (size_t)sid[s] * H + r;
        acc += fp8dot16(*(const uint4*)(cod + row * (NB * 16) + b * 16), act + ((size_t)t * TOPK + s) * I + 32 * A_.ua + b * 16) *
               __uint_as_float((uint32_t)scl[row * A_.nu + (b >> 1)] << 23);
        b += 32;
        while (b >= NB) {
            b -= NB;
            s++;
        }
    }
    acc = warp_sum(acc);
    if (lane == 0) {
        if (cst) out[r] = conv1(acc * 256.f, r, cst, cw, add);
        else {
            if constexpr (PP) {   // T == 1 (host rule)
                tp_push1(A_.pp, ptag, r, acc * 256.f);
                if (A_.pp.np) {   // [ink-tpf-pred] sum + conv here
                    float sm;
                    if (tp_reduce1(A_.pp, ptag, r, acc * 256.f, sm)) tp_conv_fin(A_.pp, r, sm, cpre);
                } else
                    out[(size_t)t * H + r] = acc * 256.f;
            } else
                out[(size_t)t * H + r] = acc * 256.f;
        }
    }
}
__global__ void __launch_bounds__(256) k_dn8(const __grid_constant__ A_dn8 A_) {
    b_dn8<>(A_, A_.tfast ? dim3(blockIdx.y, blockIdx.x, 0) : dim3(blockIdx), gridDim);
}
// [ink-tp-x8sh] MXFP8 down of row steps: a warp owns RW hidden rows that share each act unit load (b_dn8 is L1-bound
// like b_gu8: 4 B of fp32 act per weight byte, used for one row). Per row the b_dn8 lane list and order: bitwise.
// grid (T, H / (8 RW)).
template <int RW>
__global__ void __launch_bounds__(256) k_dn8r(const __grid_constant__ A_dn8 A_) {
    const int t = blockIdx.x, lane = threadIdx.x & 31, r0 = (blockIdx.y * 8 + (threadIdx.x >> 5)) * RW;
    const int NB = 2 * A_.nu;
    __shared__ int sid[TOPK];
    if (threadIdx.x < TOPK) sid[threadIdx.x] = A_.ids[t * TOPK + threadIdx.x];
    BSYNC(256);
    float acc[RW];
#pragma unroll
    for (int q = 0; q < RW; q++) acc[q] = 0.f;
    int s = lane / NB, b = lane % NB;
#pragma unroll 3
    for (int j = lane; j < TOPK * NB; j += 32) {
        float xr[16];
        const float4* ap = (const float4*)(A_.act + ((size_t)t * TOPK + s) * I + 32 * A_.ua + b * 16);
#pragma unroll
        for (int k = 0; k < 4; k++) *(float4*)(xr + 4 * k) = ap[k];
        const size_t row = (size_t)sid[s] * H + r0;
        uint4 w[RW];
        uint8_t sc[RW];
#pragma unroll
        for (int q = 0; q < RW; q++) {
            w[q] = *(const uint4*)(A_.cod + (row + q) * (NB * 16) + b * 16);
            sc[q] = A_.scl[(row + q) * A_.nu + (b >> 1)];
        }
#pragma unroll
        for (int q = 0; q < RW; q++) acc[q] += fp8dot16(w[q], xr) * __uint_as_float((uint32_t)sc[q] << 23);
        b += 32;
        while (b >= NB) {
            b -= NB;
            s++;
        }
    }
#pragma unroll
    for (int q = 0; q < RW; q++) acc[q] = warp_sum(acc[q]);
    if (lane < RW) {
        float v = acc[0];
#pragma unroll
        for (int q = 1; q < RW; q++)
            if (lane == q) v = acc[q];
        A_.out[(size_t)t * H + r0 + lane] = v * 256.f;
    }
}
__global__ void __launch_bounds__(256) k_dn8pp(const __grid_constant__ A_dn8 A_) {   // [ink-tpf-ppush] plain step, TP producer push
    b_dn8<true>(A_, dim3(blockIdx), gridDim);
}


__global__ void k_f2bf(const float* x, bf16* y, size_t n) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = __float2bfloat16(x[i]);
}

__global__ void __launch_bounds__(1024) k_argmax(const float* x, int n, int* out) {
    __shared__ float bv[32];
    __shared__ int bi[32];
    float v = -INFINITY;
    int id = 0;
    for (int i = threadIdx.x; i < n; i += 1024)
        if (x[i] > v) {
            v = x[i];
            id = i;
        }
    for (int o = 16; o; o >>= 1) {
        const float v2 = __shfl_xor_sync(0xffffffffu, v, o);
        const int i2 = __shfl_xor_sync(0xffffffffu, id, o);
        if (v2 > v || (v2 == v && i2 < id)) {
            v = v2;
            id = i2;
        }
    }
    if ((threadIdx.x & 31) == 0) {
        bv[threadIdx.x >> 5] = v;
        bi[threadIdx.x >> 5] = id;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        for (int j = 1; j < 32; j++)
            if (bv[j] > v || (bv[j] == v && bi[j] < id)) {
                v = bv[j];
                id = bi[j];
            }
        *out = id;
    }
}

// ---- exact two-stage head ([ink-head2], user decision 2026-11-22; kernels from branch inkling-exp12,
// tools/inkling_exp12/exp12_kernels.cuh): an 8-bit screen plane bounds every logit, the rows that can reach the
// maximum are rescored with the exact G15 rows (bit-identical logit), so the token equals the full head's argmax.
constexpr int HEAD_K = 2304;
struct HeadArgs {
    const uint8_t* scr;    // [N][2304] screen plane
    const uint8_t* seb;    // [N] eb per row
    const bf16* x;         // head input (the port's FINB row)
    int N;
    float* hi;             // [N] upper ends (score scale)
    unsigned* thr;         // ordered largest lower end (0 = none)
    // exact rows: G15 planes of the head
    const uint4* sm;
    const uint32_t* idx;
    const uint4* meta;
    const bf16* fb;
    unsigned long long* best;   // (ordered score << 32) | ~row
    unsigned* done;
    unsigned* ncand;            // candidates of the current step (reset by the last block)
    int* arg;                   // token out
    int* ncand_out;             // candidate count out (statistics)
    // sampling
    float invT;
    uint32_t seed;
    const int* step;
    // [ink-srv-samp] per-row sampling (row steps, the plain step: row 0): rT[j] = 1 / temperature (0: greedy), rS[j] the
    // row's seed, rP[j] its position (the Gumbel noise of a row = g(seed, position, vocab row): batch == alone and
    // verify == plain for the same seed); null: every row greedy
    const float* rT = nullptr;
    const unsigned* rS = nullptr;
    const int* rP = nullptr;
    int r0 = 0;                          // [ink-tp-dec1] vocab row of local row 0 (a rank's rows [r0, r0 + N))
    unsigned long long* bkey = nullptr;  // [ink-tp-dec1] the rank's best key (global row) for the argmax over ranks
    // [ink-samp-dev] filtered rows (rK[j] & 0xff = list size K > 0, with rT[j] > 0): top_k / top_p / min_p of row j
    // (rK[j] & FK_CERT: no user top_k, the min_p set must fit in the K-list). Screens give raw bounds and write the
    // lower ends (lo + j N); fc + 16 j = the row's counters; flist + j FCAP the rescored candidates (key = ordered
    // exact logit << 32 | ~global row)
    const int* rK = nullptr;
    const float* rTP = nullptr;
    const float* rMP = nullptr;
    float* lo = nullptr;
    unsigned* fc = nullptr;
    unsigned* fhist = nullptr;
    unsigned long long* flist = nullptr;
    double* ftail = nullptr;   // [ink-samp-dev] FK_ZT rows: sum over the non-candidates of exp((cap(hi) - cap(thr)) / T)
    int fmode = 0;   // [ink-samp-dev] rescore of filtered rows: 0 = rows with hi >= fc[1] (final list), 1 = round A: rows
                     // with lo >= fc[1] (>= K rows; their K-th exact logit is the tighter threshold of round B)
};
// [ink-samp-dev] filtered sampling limits: list size, candidate list cap per row, histogram bins of the threshold passes
constexpr int FK_MAX = 64, FK_CERT = 0x100, FK_ZT = 0x200, FCAP = 65536, FHB = 2048;
constexpr int FLOC_W = FK_MAX + 2;   // local list words per row: header, K keys, tail bound (double bits)
// fc counters per row: 0 candidates, 1 candidate threshold (ordered lower edge), 2 blocks done, 3 pass-1 bin,
// 4 rows above that bin, 5 list size after the select (statistics), 6 fallbacks (statistics)
__device__ __forceinline__ int row_filt(const HeadArgs& h, int j) {
    if (!h.rK || !h.rT || !(h.rT[j] > 0.f)) return 0;
    return h.rK[j] & 0xff;
}
__device__ __forceinline__ unsigned f2ord(float f) {
    const unsigned u = __float_as_uint(f);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}
__device__ __forceinline__ float ord2f(unsigned o) { return __uint_as_float((o & 0x80000000u) ? (o & 0x7fffffffu) : ~o); }
__device__ __forceinline__ float head_cap(float z) { return 30.f * tanhf(z * (1.f / 30.f)); }
__device__ __forceinline__ float gumbel(uint32_t seed, int step, int row);
// [ink-srv-samp] row j of a step samples: its 1 / T, seed and position
__device__ __forceinline__ bool row_samp(const HeadArgs& h, int j, float& iT, uint32_t& sd, int& st) {
    if (!h.rT) return false;
    if (h.rK && (h.rK[j] & 0xff)) return false;   // [ink-samp-dev] a filtered row: raw bounds (its own path)
    iT = h.rT[j];
    if (!(iT > 0.f)) return false;
    sd = h.rS[j];
    st = h.rP[j];
    return true;
}
__device__ __forceinline__ float gumbel(uint32_t seed, int step, int row) {
    uint32_t h = seed ^ (uint32_t)step * 0x9E3779B9u ^ (uint32_t)row * 0x85EBCA6Bu;
    h ^= h >> 16; h *= 0x7feb352du; h ^= h >> 15; h *= 0x846ca68bu; h ^= h >> 16;
    const float u = ((h >> 9) + 0.5f) * (1.f / 8388608.f);   // 23 bits: (0, 1) exactly, never 1.0
    return -logf(-logf(u));
}
// one screen pair (bytes 2k, 2k+1 of a word) -> bf16 pair word
template <int k>
__device__ __forceinline__ uint32_t scr_pair(uint32_t w, uint32_t EB) {
    const uint32_t x2 = __byte_perm(w, 0u, k ? 0x4342 : 0x4140);
    return ((x2 << 4) & 0x07f007f0u) + ((x2 << 8) & 0x80008000u) + EB;
}
template <bool SAMP>
__global__ void __launch_bounds__(256) k_head_screen(const __grid_constant__ HeadArgs h) {
    constexpr int NCH = 2 * HEAD_K / 512;   // 9 chunks of 16 B per lane per 2 rows
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    const int g = blockIdx.x * 8 + w;
    const bool live = 2 * g < h.N;
    uint4 wv[NCH], xa[NCH], xb[NCH];
    uint32_t ebw = 0;
    if (live) {
        const uint8_t* base = h.scr + (size_t)g * 2 * HEAD_K;
#pragma unroll
        for (int c = 0; c < NCH; c++) wv[c] = *(const uint4*)(base + (c * 32 + lane) * 16);
        ebw = *(const uint16_t*)(h.seb + 2 * g);
    }
#pragma unroll
    for (int c = 0; c < NCH; c++) {
        const int j = c * 32 + lane;
        const int xi = (j < 144 ? j : j - 144) * 16;
        xa[c] = *(const uint4*)(h.x + xi);
        xb[c] = *(const uint4*)(h.x + xi + 8);
    }
    const uint32_t e0 = ebw & 0xff, e1 = ebw >> 8;
    const uint32_t EB0 = (e0 << 7) | (e0 << 23), EB1 = (e1 << 7) | (e1 << 23);
    float s0 = 0.f, a0 = 0.f, s1 = 0.f, a1 = 0.f, x1 = 0.f;
#pragma unroll
    for (int c = 0; c < NCH; c++) {
        const bool r1 = c * 32 + lane >= 144;   // c < 4: row 0, c > 4: row 1, c == 4: lanes 16-31 row 1
        const uint32_t EB = r1 ? EB1 : EB0;
        const uint32_t ww[4] = {wv[c].x, wv[c].y, wv[c].z, wv[c].w};
        const uint32_t xx[8] = {xa[c].x, xa[c].y, xa[c].z, xa[c].w, xb[c].x, xb[c].y, xb[c].z, xb[c].w};
        float s = 0.f, a = 0.f, xs = 0.f;
#pragma unroll
        for (int q = 0; q < 4; q++) {
#pragma unroll
            for (int k = 0; k < 2; k++) {
                const uint32_t p = k ? scr_pair<1>(ww[q], EB) : scr_pair<0>(ww[q], EB);
                const uint32_t xw = xx[2 * q + k];
                const float wl = __uint_as_float(p << 16), wh = __uint_as_float(p & 0xffff0000u);
                const float xl = __uint_as_float(xw << 16), xh = __uint_as_float(xw & 0xffff0000u);
                s = fmaf(wl, xl, s);
                s = fmaf(wh, xh, s);
                a = fmaf(fabsf(wl), fabsf(xl), a);
                a = fmaf(fabsf(wh), fabsf(xh), a);
                xs += fabsf(xl) + fabsf(xh);
            }
        }
        if (r1) { s1 += s; a1 += a; }
        else { s0 += s; a0 += a; x1 += xs; }
    }
    s0 = warp_sum(s0); a0 = warp_sum(a0); s1 = warp_sum(s1); a1 = warp_sum(a1); x1 = warp_sum(x1);
    constexpr float EPS = 1.f / 16 + 1.f / 1024, XM = 1.f + 1.f / 1024;
    const float B0 = a0 * EPS + __uint_as_float(e0 << 23) * x1 * XM;
    const float B1 = a1 * EPS + __uint_as_float(e1 << 23) * x1 * XM;
    float lo0 = s0 - B0, hi0 = s0 + B0, lo1 = s1 - B1, hi1 = s1 + B1;
    float iT_ = 0.f;
    uint32_t sd_ = 0;
    int st_ = 0;
    if (SAMP) {
        iT_ = h.invT;
        sd_ = h.seed;
        st_ = *h.step;
    }
    if (SAMP || row_samp(h, 0, iT_, sd_, st_)) {
        const float g0 = gumbel(sd_, st_, h.r0 + 2 * g), g1 = gumbel(sd_, st_, h.r0 + 2 * g + 1);
        auto tr = [&](float z, float gg, float sgn) {
            const float v = head_cap(z) * iT_ + gg;
            return v + sgn * (1e-5f * (30.f * iT_ + fabsf(gg)) + 1e-6f);
        };
        lo0 = tr(lo0, g0, -1.f); hi0 = tr(hi0, g0, 1.f); lo1 = tr(lo1, g1, -1.f); hi1 = tr(hi1, g1, 1.f);
    }
    __shared__ unsigned bl[8];
    if (lane == 0) {
        unsigned m = 0;
        if (live) {
            const bool fl = !SAMP && h.lo && row_filt(h, 0);   // [ink-samp-dev] the raw lower ends of a filtered row
            h.hi[2 * g] = hi0;
            if (fl) h.lo[2 * g] = lo0;
            m = f2ord(lo0);
            if (2 * g + 1 < h.N) {
                h.hi[2 * g + 1] = hi1;
                if (fl) h.lo[2 * g + 1] = lo1;
                m = max(m, f2ord(lo1));
            }
        }
        bl[w] = m;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        unsigned m = bl[0];
        for (int j = 1; j < 8; j++) m = max(m, bl[j]);
        atomicMax(h.thr, m);
    }
}
// exact logit of one head row from the G15 planes, the port's k_gemv order (warp-uniform r)
__device__ __forceinline__ float head_rowdot12(const HeadArgs& h, int r, const uint4* xv, int lane) {
    constexpr int NC = HEAD_K / 256, WD = 2 * NC, NF = WD / 4;
    const int g = r >> 1, rr = r & 1;
    const uint4 m = h.meta[g];
    const uint32_t* ip = h.idx + (size_t)g * 32 * WD;
    uint2 smw[NC];
    uint32_t I[NC];
#pragma unroll
    for (int c = 0; c < NC; c++) {
        const int t = rr * NC + c;
        smw[c] = ((const uint2*)(h.sm + ((size_t)g * NC + (t >> 1)) * 32 + lane))[t & 1];
        I[c] = t < NF * 4 ? ip[((t >> 2) * 32 + lane) * 4 + (t & 3)] : ip[NF * 128 + lane * 2 + (t - NF * 4)];
    }
    const uint32_t n = (m.x >> 16) & 0xff;
    float s = 0.f;
    if (n == 255) {
        const bf16* W = h.fb + (size_t)m.y * 2 * HEAD_K + (size_t)rr * HEAD_K;
#pragma unroll
        for (int c = 0; c < NC; c++) {
            const uint4 wq = *(const uint4*)(W + (c * 32 + lane) * 8);
            const uint32_t ww[4] = {wq.x, wq.y, wq.z, wq.w};
            const uint32_t xx[4] = {xv[c].x, xv[c].y, xv[c].z, xv[c].w};
#pragma unroll
            for (int k = 0; k < 4; k++)
                s += __uint_as_float(ww[k] << 16) * __uint_as_float(xx[k] << 16) +
                     __uint_as_float(ww[k] & 0xffff0000u) * __uint_as_float(xx[k] & 0xffff0000u);
        }
    } else {
        const uint32_t b = rr ? (m.x >> 8) & 0xff : m.x & 0xff;
        const uint32_t Bv = (b << 7) | (b << 23);
        uint32_t ESC = 0u;
        const uint32_t ent[6] = {m.y & 0xffff, m.y >> 16, m.z & 0xffff, m.z >> 16, m.w & 0xffff, m.w >> 16};
#pragma unroll
        for (int j = 0; j < 6; j++) {
            const uint32_t e = ent[j], ex = e >> 6;
            const bool mine = j < (int)n && (e & 31) == (uint32_t)lane && ((e >> 5) & 1) == (uint32_t)rr;
            ESC = mine ? ((ex << 7) | (ex << 23)) : ESC;
        }
#pragma unroll
        for (int c = 0; c < NC; c++) {
            const uint32_t ww[4] = {dec_pair<0>(smw[c].x, I[c], Bv, ESC), dec_pair<1>(smw[c].x, I[c], Bv, ESC),
                                    dec_pair<2>(smw[c].y, I[c], Bv, ESC), dec_pair<3>(smw[c].y, I[c], Bv, ESC)};
            const uint32_t xx[4] = {xv[c].x, xv[c].y, xv[c].z, xv[c].w};
#pragma unroll
            for (int k = 0; k < 4; k++)
                s += __uint_as_float(ww[k] << 16) * __uint_as_float(xx[k] << 16) +
                     __uint_as_float(ww[k] & 0xffff0000u) * __uint_as_float(xx[k] & 0xffff0000u);
        }
    }
    return warp_sum(s);
}
template <bool SAMP>
__device__ __forceinline__ void b_head_rescore(const HeadArgs& h) {
    constexpr int NC = HEAD_K / 256;
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    const int nw = gridDim.x * 8, wid = blockIdx.x * 8 + w;
    uint4 xv[NC];
#pragma unroll
    for (int c = 0; c < NC; c++) xv[c] = *(const uint4*)(h.x + (c * 32 + lane) * 8);
    // [ink-samp-dev] a filtered row: candidates = rows that can reach its top K (upper end >= the threshold of k_fthr),
    // every candidate's exact logit goes into the row's list (k_fsel picks the top K)
    const bool filt = !SAMP && h.fc && row_filt(h, 0);
    if (h.fmode && !filt) return;   // round A: filtered rows only (the others' counters stay untouched)
    const float thr = filt ? ord2f(((volatile unsigned*)h.fc)[1]) : ord2f(*(volatile unsigned*)h.thr);
    const float* hv = h.fmode ? h.lo : h.hi;
    // [ink-samp-dev] FK_ZT (top_p alone), final pass: exp((cap(hi) - cap(thr)) / T) <= 1 summed over the non-candidates
    const bool ztail = filt && !h.fmode && h.ftail && (h.rK[0] & FK_ZT);
    const float iTz = ztail ? h.rT[0] : 0.f, cthr = ztail ? head_cap(thr) : 0.f;
    float zt = 0.f;
    float best = -INFINITY;
    int bi = 0x7fffffff;
    unsigned cnt = 0;
    float iT = SAMP ? h.invT : 0.f;
    uint32_t sd = SAMP ? h.seed : 0u;
    int st = SAMP ? *h.step : 0;
    const bool samp = SAMP || row_samp(h, 0, iT, sd, st);   // [ink-srv-samp] row 0 of these pointers
    for (int b = wid * 32; b < h.N; b += nw * 32) {   // 32-row blocks dealt round-robin: clustered candidates spread
        const int r = b + lane;
        const bool c = r < h.N && hv[r] >= thr;
        unsigned msk = __ballot_sync(0xffffffffu, c);
        cnt += __popc(msk);
        if (filt) {
            if (ztail) {   // [ink-samp-dev] top_p alone: the tail bound of the rows that stay out of the list
                float e = 0.f;
                if (r < h.N && !c) e = expf((head_cap(hv[r]) - cthr) * iTz);
                zt += e;
            }
            unsigned base = 0;
            if (lane == 0 && msk) base = atomicAdd(h.fc, (unsigned)__popc(msk));
            base = __shfl_sync(0xffffffffu, base, 0);
            for (unsigned q = 0; msk; q++) {
                const int rr = b + __ffs(msk) - 1;
                msk &= msk - 1;
                const float v = head_rowdot12(h, rr, xv, lane);
                if (lane == 0 && base + q < (unsigned)FCAP)
                    h.flist[base + q] = ((unsigned long long)f2ord(v) << 32) | (0xffffffffu - (unsigned)(h.r0 + rr));
            }
            continue;
        }
        while (msk) {
            const int rr = b + __ffs(msk) - 1;
            msk &= msk - 1;
            float v = head_rowdot12(h, rr, xv, lane);
            if (samp) v = head_cap(v) * iT + gumbel(sd, st, h.r0 + rr);
            if (v > best) {   // rows ascend: ties keep the lower row
                best = v;
                bi = rr;
            }
        }
    }
    if (ztail) {
        zt = warp_sum(zt);
        if (lane == 0 && !(zt <= 0.f)) atomicAdd(h.ftail, (double)zt * (1.0 + 1e-4 + 4e-6 * iTz));   // (slack over expf / sum rounding, review2 L2; a NaN forces the host pick, L3)
    }
    __shared__ unsigned long long kb[8];
    __shared__ unsigned cb[8];
    __shared__ bool last;
    if (lane == 0) {
        kb[w] = bi == 0x7fffffff ? 0ull : ((unsigned long long)f2ord(best) << 32) | (0xffffffffu - (unsigned)(h.r0 + bi));
        cb[w] = cnt;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        unsigned long long k = kb[0];
        unsigned cn = cb[0];
        for (int j = 1; j < 8; j++) {
            k = max(k, kb[j]);
            cn += cb[j];
        }
        if (k) atomicMax(h.best, k);
        atomicAdd(h.ncand, cn);
        __threadfence();
        last = atomicAdd(h.done, 1u) == gridDim.x - 1;
    }
    __syncthreads();
    if (last && threadIdx.x == 0) {
        __threadfence();
        const unsigned long long k = *(volatile unsigned long long*)h.best;
        *h.arg = (int)(0xffffffffu - (unsigned)(k & 0xffffffffu));
        if (h.bkey) *h.bkey = k;
        if (h.ncand_out) *h.ncand_out = (int)*(volatile unsigned*)h.ncand;
        *h.best = 0ull;
        *h.thr = 0u;
        *h.ncand = 0u;
        *h.done = 0u;
    }
}
template <bool SAMP>
__global__ void __launch_bounds__(256) k_head_rescore(const __grid_constant__ HeadArgs h) { b_head_rescore<SAMP>(h); }
// [ink-b-small] the rescore of all rows of a row step in one launch: blockIdx.y = row j (the per-row launch's pointers)
__global__ void __launch_bounds__(256) k_head_rescoreR(const __grid_constant__ HeadArgs h0, unsigned* hc, int* arg) {
    HeadArgs r = h0;
    const int j = blockIdx.y;
    r.x = h0.x + (size_t)j * H;
    r.hi = h0.hi + (size_t)j * h0.N;   // the screens' row stride (V; a TP rank's vocab rows)
    r.thr = hc + j * 16;
    r.done = r.thr + 1;
    r.ncand = r.thr + 2;
    r.best = (unsigned long long*)(r.thr + 4);
    r.arg = arg + j;
    if (h0.bkey) r.bkey = h0.bkey + j;   // [ink-tp-dec1]
    if (h0.rT) {   // [ink-srv-samp] row j's sampling as row 0
        r.rT = h0.rT + j;
        r.rS = h0.rS + j;
        r.rP = h0.rP + j;
    }
    if (h0.rK) {   // [ink-samp-dev] row j's filter as row 0
        r.rK = h0.rK + j;
        r.rTP = h0.rTP + j;
        r.rMP = h0.rMP + j;
        r.fc = h0.fc + j * 16;
        r.flist = h0.flist + (size_t)j * FCAP;
        if (h0.ftail) r.ftail = h0.ftail + j;
        if (h0.lo) r.lo = h0.lo + (size_t)j * h0.N;
    }
    b_head_rescore<false>(r);
}

#include "ink_rows.cuh"
#include "ink_draft.cuh"
#include "ink_tcr.cuh"
#include "ink_tp.cuh"   // [ink-tp-dec1] TP-N all-reduce
#include "ink_samp.cuh"   // [ink-samp-dev] filtered sampling in the head
#include "ink_attn.cuh"
#include "ink_attn3.cuh"
#include "ink_attn_tb3.cuh"

// ---------------------------------------------------------------- engine
static void* dmalloc(size_t n, size_t* tot) {
    void* p = nullptr;
    CU(cudaMalloc(&p, n));
    *tot += n;
    return p;
}


struct Layer {
    bool full = false, loop = false;
    int NP = 0, offK = 0, offV = 0;   // fused projection rows: [q r gate sink | k (+kdyn) | v (+vdyn)]
    bf16* Wp = nullptr;               // [NP][H] BF16 (only with INK_BF16KEEP=1: A/B against the 12-bit path)
    bf16* Wo = nullptr;               // [H][QD]
    G12Args qp12{}, wo12{};           // the same matrices in G15 (lossless 12-bit), planes only
    G12Args qp8{}, wo8{};             // [ink-n4-int8] TP rank: the decode GEMVs' INT8 g128 planes (INK_DENSE8=1)
    float *qs = nullptr, *rs = nullptr, *proj = nullptr, *projT = nullptr;   // projT: [REL][DREL] copy of proj ([ink-attn-pT])
    float *kw = nullptr, *vw = nullptr;   // local [KVD][4], full [4][KVD]
    float *aw = nullptr, *mw = nullptr;   // [H][4]
    bf16 *nq = nullptr, *nk = nullptr, *nv = nullptr, *nm = nullptr;   // local: nq = attn_norm
    float* aq = nullptr;   // attention_input query (1 or 3 x H), null for layer 0
    float* mq = nullptr;   // mlp_input query
    bf16* Wr = nullptr;
    float* rb = nullptr;
    int fmt = 0;
    uint8_t *c13 = nullptr, *s13 = nullptr, *c2 = nullptr, *s2 = nullptr;
    float *f13 = nullptr, *f2 = nullptr;
    uint8_t *m13 = nullptr, *m2 = nullptr;   // [ink-tp-tcx] MXFP8: max scale byte per (rank) expert row (only with INK_TCX_MIN)
    bf16 *gd[4] = {}, *gf[4] = {}, *gg[4] = {};   // controllers attn(q), k, v, mlp
    // [ink-tp-dec1] the rank's rows of the fused projection: qp12 holds NPr gathered rows (q side [0, qr1), k [qr1, qr2),
    // v [qr2, NPr)), qmap[r] = the row in the full layout of P (null: all rows, in order); woR the rank's column chunks
    int NPr = 0, qr1 = 0, qr2 = 0;
    int* qmap = nullptr;
    G12Args woR{};
    // [ink-tp-pf] batched TP prefill: contiguous runs of the gathered rows (rank row, full-layout row, length, input
    // 0 q / 1 k / 2 v) for one cuBLAS GEMM each into the full layout of P; the rank's wo columns as BF16 [H][256 nc]
    std::vector<std::array<int, 4>> qruns;
    bf16* woB = nullptr;
    // [ink-pf3-qkv] the rank's compact P layout (ldp NPr, columns = the gathered rows): shifts dX = full column of
    // segment X's first rank element - its compact column; K / V (+ taps) blocks at cK / cKt / cV / cVt
    int dq = 0, dr = 0, dg = 0, ds = 0, cK = 0, cKt = -1, cV = 0, cVt = -1;
    bool pcok = false;   // pf3_review M1 / M2: whole GQA groups on the rank and 16 B aligned segment starts
};

struct AppState {   // one layer application (48): K/V cache + conv states
    bf16 *K = nullptr, *V = nullptr;
    int cap = 0, ring = 0;
    float *kst = nullptr, *vst = nullptr, *ast = nullptr, *mst = nullptr;
    float *kh = nullptr, *vh = nullptr, *ah = nullptr, *mh = nullptr;   // verify: raw conv inputs of the rows [VR][C]
    bf16 *K0 = nullptr, *V0 = nullptr;   // [ink-tp-dec1] the allocations (K / V are based so that KV head kv0 is in place)
    size_t kvb = 0;                      // their bytes
};
struct DLayer {   // drafter layer (BF16 as stored)
    bf16 *dn, *iln, *pln, *qn, *kn, *knc, *Wqkv, *Wkvc, *Wo, *Wgu, *Wd;
    G12Args Wqkv12{}, Wgu12{};   // [ink-dr-g15] lossless 12-bit copies (tensor-core GEMV); BF16 kept only for A/B 98
    G12Args Wgu12r{};            // [ink-tp-drv2b] TP: the rank's gate | up rows [f0, f0 + nf) (G15)
    bf16* Wdr = nullptr;         // [ink-tp-drv2b] the rank's down columns [H][nf]
    G12Args Wqkv12r{};           // [ink-tp-dra] TP: the rank's q | k | v head rows (G15)
    bf16* Wor = nullptr;         // [ink-tp-dra] the rank's o_proj columns [H][nq HD]
    int f0 = 0, nf = 0;
};

// [ink-tp-dec1] state shared by the ranks of one TP engine (one per Engine; ranks = Engine::Impl, one host thread each)
struct TpCtx {
    int N = 1;
    std::vector<Engine::Impl*> R;
    std::vector<int> devs;
    std::vector<TpShard> plan;
    TpBarrier bar;
    TpPool pool;
    bool push = false;   // ranks on distinct devices with peer access: P2P push collectives; else all on one device (direct)
    File file;           // one mapping for all ranks (replicated uploads are shared by device: key = mapped bytes)
    std::mutex rep_mu;
    std::map<std::tuple<int, const void*, size_t, int>, void*> rep;   // (device, host bytes, size, kind) -> device copy
    std::map<int, int> per_dev;   // ranks per device
    // review tp M2: abort word (pinned, mapped, portable): set by a failing rank, polled by every collective's wait loop
    unsigned* abort_h = nullptr;
    const unsigned* abort_d = nullptr;
    bool pf_ok = true;   // [ink-tp-pf] every rank's expert units fit the prefill expert GEMM tiles (else: decode-row prefill)
    // review tp3 M2: a pending TP prompt-end snapshot: every rank's async blob, assembled into *pend_out by ring_finalize
    std::vector<std::vector<unsigned char>> pend;
    std::vector<unsigned char>* pend_out = nullptr;
};
struct Engine::Impl {
    Opts o;
    File f;
    // [ink-tp-dec1] TP-N rank (tpc null: the one-card engine, unchanged)
    TpCtx* tpc = nullptr;
    TpShard sh;
    bool tps = false;          // this rank holds a part (not the whole model): the TP forms + collectives
    int dev = 0;
    float *WOP = nullptr, *MOP = nullptr;   // partial rows of the two collectives of an application [MROW][H]
    unsigned* tstep = nullptr;              // push: step counter (tags)
    uint4* trw = nullptr;                   // push: receive words [TP_SLOTS][N][MROW][TPW]
    // [ink-tp-r2-fmt] QE_TP_CHK=n: xor hashes of every collective's sum [256] (xor-accumulated over calls), compared
    // across the ranks every n calls; any mismatch is a fatal error (stress gate of the 16 B word format)
    unsigned* tchk = nullptr;
    long long tchk_every = getenv("QE_TP_CHK") ? atoll(getenv("QE_TP_CHK")) : 0, tchk_calls = 0, tchk_cmp = 0;
    unsigned tchk_h[256];
    cudaEvent_t tev[TP_SLOTS] = {};         // direct: partial ready, per slot
    int nkvA() const { return tps ? sh.nkv() : NKV; }
    bool pcP(int T, const Layer& y) const { return T <= pcq && tps && pfx && !vR && y.pcok; }   // [ink-pf3-qkv] the TP prefill rows' P is in the compact layout
    int kv0A() const { return tps ? sh.kv0 : 0; }
    // replicated uploads: the first rank on a device uploads, the others share the copy
    template <class F>
    void* rep_up(const void* hp, size_t n, int kind, F up) {
        if (!tpc) return up();
        std::lock_guard<std::mutex> l(tpc->rep_mu);
        auto k = std::make_tuple(dev, hp, n, kind);
        auto it = tpc->rep.find(k);
        if (it != tpc->rep.end()) return it->second;
        void* d = up();
        tpc->rep[k] = d;
        return d;
    }
    // collective idx (0 .. 2 NAPP - 1 per step): sum of every rank's partial rows `part` (nr rows), then the epilogue
    void tp_ar(int idx, float* part, int nr, int epi, float* out, float* cst = nullptr, const float* cw = nullptr, const float* add = nullptr) {
        A_tpar a{};
        a.t.n = sh.N;
        a.t.rank = sh.rank;
        a.t.push = tpc->push;
        a.t.step = tstep;
        a.t.rd = trw;
        a.t.ab = tpc->abort_d;
        a.t.chk = tchk;
        a.part = part;
        a.idx = idx;
        a.nr = nr;
        a.epi = epi;
        a.out = out;
        a.cst = cst;
        a.cw = cw;
        a.add = add;
        if (tpc->push) {
            for (int g = 0; g < sh.N; g++) a.t.src[g] = tpc->R[g]->trw;
        } else {
            const int sl = idx % TP_SLOTS;
            CU(cudaEventRecord(tev[sl], st));
            tpc->bar.wait();   // every rank recorded this collective's event before anyone waits on it
            for (int g = 0; g < sh.N; g++) {
                if (g != sh.rank) CU(cudaStreamWaitEvent(st, tpc->R[g]->tev[sl], 0));
                a.t.src[g] = part == WOP ? (const void*)tpc->R[g]->WOP : part == GATH ? (const void*)tpc->R[g]->GATH
                           : (DPT && part >= DPT && part < DPT + (size_t)2 * NDS * DB * H) ? (const void*)(tpc->R[g]->DPT + (part - DPT))   // [ink-tp-drv2b]
                           : (const void*)tpc->R[g]->MOP;
            }
        }
        if (tpcc_y) {   // [ink-tp-tpcc] + the conv chain
            a.out2 = tpcc_y;
            k_tpartC<<<nr, TPA_W, 0, st>>>(A_tpc{a, tpcc_h, DRT});
        } else if (tpc->push && epi == 0 && nr >= tpart_min) k_tpart<<<nr, TPA_W, 0, st>>>(a);   // [ink-tp-art]
        else k_tpar<<<nr, TPT, 0, st>>>(a);
    }
    // [ink-tp-tpcc] row all-reduce + verify conv chain in one launch (push transport, trailer form); else tp_ar + the chain
    float* tpcc_y = nullptr;
    float* tpcc_h = nullptr;
    bool moC = false;
    int tpcc = getenv("INK_TPCC") ? atoi(getenv("INK_TPCC")) : 1;
    bool tp_arC(int idx, float* part, int nr, float* sum, float* cst, const float* cw, const float* add, float* y, float* hist) {
        if (!tpcc || !tpc->push || nr < tpart_min) return false;
        tpcc_y = y;
        tpcc_h = hist;
        tp_ar(idx, part, nr, 0, sum, cst, cw, add);
        tpcc_y = tpcc_h = nullptr;
        if (tpcc_chk && tpcc_buf) {   // INK_TPCC_CHK=1: the old chain on the same sums, every output bit compared (buffers from init: review7 M1)
            k_tpchain<<<dim3((H + 255) / 256, convseg ? nseg_h : 1), 256, 0, st>>>(sum, nr, cst, cw, add, tpcc_buf, tpcc_buf + (size_t)MROW * H, DRT);
            k_cmpbits<<<(nr * H + 255) / 256, 256, 0, st>>>(y, tpcc_buf, nr * H, tpcc_bad, nullptr);
            k_cmpbits<<<(nr * H + 255) / 256, 256, 0, st>>>(hist, tpcc_buf + (size_t)MROW * H, nr * H, tpcc_bad + 1, tpcc_bad + 2);
        }
        return true;
    }
    bool tpcc_chk = getenv("INK_TPCC_CHK") != nullptr;
    float* tpcc_buf = nullptr;
    unsigned long long* tpcc_bad = nullptr;
    int tpart_min = getenv("INK_TPART") ? atoi(getenv("INK_TPART")) : 2;   // [ink-tp-art] trailer form from this many rows (epilogue 0, push transport; INK_TPART=99: off)
    // [ink-tp-pf] prefill all-reduce of T rows (copy engines, any N, any transport): every rank copies its partial rows
    // into each peer's receive rows (slot by call parity), records "sent", meets the others at the host barrier, waits for
    // the peers' "sent", sums in rank order (k_tpsumP), records "free". A copy into a peer's slot first waits for that
    // peer's "free" of the slot's previous use (recorded before an earlier barrier).
    float* tpP = nullptr;    // [C][H] this rank's partial rows
    float* tpRX = nullptr;   // [2][N - 1][C][H] receive rows (peer g at g - (g > rank))
    // review tp2 L6: events, slot parities and receive rows per prefill sub-chunk (ps() = the current view's sub-chunk;
    // rows at pf_r0, the view's first row), so one sub-chunk's slot reuse never waits for another sub-chunk's progress
    static constexpr int PS_MAX = 6;
    cudaEvent_t tpES[PS_MAX][2] = {}, tpEF[PS_MAX][2] = {};
    int tpsl_[PS_MAX] = {};
    int pf_r0 = 0;
    int pf_sub = 0;
    int ps() const { return pf_sub; }
    float* tp_rx(int slot, int g, int r0 = 0) { return tpRX + ((size_t)slot * (sh.N - 1) + (g - (g > sh.rank))) * (size_t)pcap * H + (size_t)r0 * H; }
    void tp_arP(int T, float* out) {
        const int sl = tpsl_[ps()];
        tpsl_[ps()] ^= 1;
        const size_t nb = (size_t)T * H * 4;
        for (int g = 0; g < sh.N; g++) {
            if (g == sh.rank) continue;
            Impl& q = *tpc->R[g];
            CU(cudaStreamWaitEvent(st, q.tpEF[ps()][sl], 0));
            CU(cudaMemcpyAsync(q.tp_rx(sl, sh.rank, pf_r0), tpP, nb, cudaMemcpyDefault, st));
        }
        CU(cudaEventRecord(tpES[ps()][sl], st));
        tpc->bar.wait();
        A_tpsumP a{};
        a.n = sh.N;
        a.n4 = (size_t)T * H / 4;
        a.out = (float4*)out;
        for (int g = 0; g < sh.N; g++) {
            if (g == sh.rank) {
                a.src[g] = (const float4*)tpP;
                continue;
            }
            CU(cudaStreamWaitEvent(st, tpc->R[g]->tpES[ps()][sl], 0));
            a.src[g] = (const float4*)tp_rx(sl, g, pf_r0);
        }
        k_tpsumP<<<(int)std::min<size_t>((a.n4 + 255) / 256, 82 * 8), 256, 0, st>>>(a);
        CU(cudaEventRecord(tpEF[ps()][sl], st));
    }
    // [ink-tp-sp] sequence-parallel rows of the TP prefill: rank r runs the row-local work (residual mixes, message
    // gates, norms, convs, router) on rows [a_r, a_r+1) only; the head / expert-split GEMM stages see all rows through an
    // all-gather of their bf16 / fp16 inputs, and their fp32 partials come back as a reduce-scatter of the rank's rows
    // + a 3-row conv halo + the chunk's last 3 rows (every rank keeps the conv states)
    bool sp_on = !(getenv("INK_TPSP") && atoi(getenv("INK_TPSP")) == 0);
    // [ink-pf3-qkv] TP prefill qkv: one GEMM per input (q side / K / V, or one for a shared input) into the rank's
    // compact P layout (ldp NPr; consumers take the shifts) for sub-chunks of <= INK_PF3QKV rows (at 2K +1.7 %, at
    // 4K-row sub-chunks (8K chunk) -4 %). Default OFF (full-layout runs everywhere): run6 c = 1 306.8 / 307.6 vs
    // 317.5 / 319.2 tok/s off (-3.4 %, 2026-11-25); INK_PF3QKV=1 = opt-in at 2048 rows, N > 1 = N rows
    int pcq = !getenv("INK_PF3QKV") ? 0 : atoi(getenv("INK_PF3QKV")) == 1 ? 2048 : atoi(getenv("INK_PF3QKV"));
    int spm = 0xff;   // stage mask of layer_ (one call = all stages)
    bool sp = false;  // inside layer_sp: the TP branches leave the collectives to it
    bool sp_h0 = false;   // the rank's rows start at row 0 (conv history = the stored state)
    enum { SP_IN = 1, SP_ATT = 2, SP_C1 = 4, SP_MIN = 8, SP_MOE = 16, SP_C2 = 32 };
    enum { SPB_XB = 0, SPB_X16 = 3, SPB_IDS, SPB_WTS, SPB_FIN, SPB_FINB, SPB_TAP, SPB_N = SPB_TAP + 3 };
    void* spb[SPB_N] = {};   // base pointers of the gathered buffers (immutable after alloc; peers read them)
    cudaEvent_t agF[PS_MAX][2] = {}, agS[PS_MAX][2] = {};
    int agsl_[PS_MAX] = {};
    int sp_a(int T, int r) const { return (int)((long long)T * r / sh.N); }
    bool sp_ok(int T) const { return sp_on && tps && pfx && !vR && sh.N > 1 && T >= 64 * sh.N && !dumping; }
    // all-gather in place: rows [a_r, a_r+1) of each listed buffer from this rank into every peer's same buffer
    struct AgItem {
        int id;
        void* x;
        size_t rb;
    };
    void tp_agP(int T, std::initializer_list<AgItem> it) {
        tp_ag_send(T, it);
        tp_ag_wait();
    }
    int ag_last = 0;
    // review tp2 L8: the outgoing copies run on a copy stream (cps; per sub-chunk), so the compute stream goes on with
    // the own rows while they fly; the compute stream waits for this rank's own sends before it rewrites those rows
    cudaStream_t cps = nullptr;
    cudaEvent_t rsR[PS_MAX][2] = {};
    void tp_ag_wait() {   // the peers' rows are in, this rank's rows are out (stream order)
        for (int g = 0; g < sh.N; g++) CU(cudaStreamWaitEvent(st, tpc->R[g]->agS[ps()][ag_last], 0));
    }
    void tp_ag_send(int T, std::initializer_list<AgItem> it) {
        const int sl = agsl_[ps()];
        ag_last = sl;
        agsl_[ps()] ^= 1;
        CU(cudaEventRecord(agF[ps()][sl], st));   // this rank's earlier reads of the buffers are before this point
        tpc->bar.wait();
        const int a = sp_a(T, sh.rank), b = sp_a(T, sh.rank + 1);
        CU(cudaStreamWaitEvent(cps, agF[ps()][sl], 0));   // this rank's rows are written (stream order before agF)
        for (int g = 0; g < sh.N; g++) {
            if (g == sh.rank) continue;
            const Impl& q = *tpc->R[g];
            CU(cudaStreamWaitEvent(cps, q.agF[ps()][sl], 0));
            for (const AgItem& x : it) {
                const size_t off = (size_t)((char*)x.x - (char*)spb[x.id]) + (size_t)a * x.rb;
                CU(cudaMemcpyAsync((char*)q.spb[x.id] + off, (char*)x.x + (size_t)a * x.rb, (size_t)(b - a) * x.rb, cudaMemcpyDefault, cps));
            }
        }
        CU(cudaEventRecord(agS[ps()][sl], cps));
        tpc->bar.wait();
    }
    // reduce-scatter of tpP's T rows: this rank gets the rank-order sums of rows [a - 3, b) and [T - 3, T) in out
    void sp_rows(int T, int r, int* lo, int* hi, int* tl) const {
        *lo = std::max(0, sp_a(T, r) - 3);
        *hi = sp_a(T, r + 1);
        *tl = std::max(*hi, T - 3);
    }
    void tp_rsP(int T, float* out) {
        tp_rs_send(T);
        tp_rs_recv(T, out);
    }
    // this rank's rows nobody else needs (computed after the send): [a, min(b, T) - 3)
    void sp_priv(int T, int* p0, int* p1) const {
        *p0 = sp_a(T, sh.rank);
        *p1 = std::max(*p0, sp_a(T, sh.rank + 1) - 3);
    }
    int rs_last = 0;
    void tp_rs_send(int T) {   // rows of tpP the peers need: all but this rank's private rows must be computed
        const int sl = tpsl_[ps()];
        tpsl_[ps()] ^= 1;
        rs_last = sl;
        CU(cudaEventRecord(rsR[ps()][sl], st));   // the rows the peers need are computed
        CU(cudaStreamWaitEvent(cps, rsR[ps()][sl], 0));
        for (int g = 0; g < sh.N; g++) {
            if (g == sh.rank) continue;
            Impl& q = *tpc->R[g];
            int lo, hi, tl;
            sp_rows(T, g, &lo, &hi, &tl);
            CU(cudaStreamWaitEvent(cps, q.tpEF[ps()][sl], 0));
            float* d = q.tp_rx(sl, sh.rank, pf_r0);
            CU(cudaMemcpyAsync(d + (size_t)lo * H, tpP + (size_t)lo * H, (size_t)(hi - lo) * H * 4, cudaMemcpyDefault, cps));
            if (T > tl) CU(cudaMemcpyAsync(d + (size_t)tl * H, tpP + (size_t)tl * H, (size_t)(T - tl) * H * 4, cudaMemcpyDefault, cps));
        }
        CU(cudaEventRecord(tpES[ps()][sl], cps));
        tpc->bar.wait();
    }
    void tp_rs_recv(int T, float* out) {   // after this rank's private rows: wait for the peers, sum the rank's rows
        const int sl = rs_last;
        for (int g = 0; g < sh.N; g++) CU(cudaStreamWaitEvent(st, tpc->R[g]->tpES[ps()][sl], 0));   // (own: tpP free again)
        int lo, hi, tl;
        sp_rows(T, sh.rank, &lo, &hi, &tl);
        for (int k = 0; k < 2; k++) {
            const int r0 = k ? tl : lo, r1 = k ? T : hi;
            if (r1 <= r0) continue;
            A_tpsumP a{};
            a.n = sh.N;
            a.n4 = (size_t)(r1 - r0) * H / 4;
            a.out = (float4*)(out + (size_t)r0 * H);
            for (int g = 0; g < sh.N; g++) a.src[g] = (const float4*)((g == sh.rank ? tpP : tp_rx(sl, g, pf_r0)) + (size_t)r0 * H);
            k_tpsumP<<<(int)std::min<size_t>((a.n4 + 255) / 256, 82 * 8), 256, 0, st>>>(a);
        }
        CU(cudaEventRecord(tpEF[ps()][sl], st));
    }
    // row buffers of the row-local stages advanced by r rows (r < 0 undoes)
    void sp_shift(int r) {
        auto adv = [&](auto*& ptr, size_t per_row) { ptr += (ptrdiff_t)r * (ptrdiff_t)per_row; };
        for (float** q : {&XA[0], &XA[1], &XA[2], &XG[0], &XG[1], &XG[2], &Rr, &ATT, &MI, &MIG, &XMF, &MO, &FIN}) adv(*q, H);
        for (bf16** q : {&XB[0], &XB[1], &XB[2], &FINB}) adv(*q, H);
        adv(SC, NE);
        adv(IDS, TOPK);
        adv(WTS, TOPK);
        adv(pX16, H);
    }
    // one application with sequence-parallel rows (stages of layer_ by mask, the collectives in between)
    void layer_sp(int l, int p, int T, const float* hin, const std::vector<const float*>& bs, float* const* msg, float* hout) {
        const Layer& y = L[l];
        const AppState& s = A[app(l, p)];
        const int a = sp_a(T, sh.rank), n = sp_a(T, sh.rank + 1) - a;
        const size_t o = (size_t)a * H;
        std::vector<const float*> bss;
        for (auto b : bs) bss.push_back(b + o);
        float* ms[4] = {};
        if (msg)
            for (int k = 0; k < 4; k++) ms[k] = msg[k] + o;
        struct SpGuard {   // review tp2 H1: a throwing stage leaves no shift / mask behind
            Impl* p;
            int sh = 0;
            ~SpGuard() {
                if (sh) p->sp_shift(-sh);
                p->sp = false;
                p->spm = 0xff;
                p->sp_h0 = false;
            }
        } g{this};
        auto rows = [&](int m) {   // a row-local stage on this rank's rows
            sp_shift(a);
            g.sh = a;
            spm = m;
            sp_h0 = a == 0;
            layer_(l, p, n, hin + o, bss, msg ? ms : nullptr, hout + o);
            sp_shift(-a);
            g.sh = 0;
        };
        auto all = [&](int m) {
            spm = m;
            layer_(l, p, T, hin, bs, msg, hout);
        };
        sp = true;
        rows(SP_IN);
        const bool shared_in = !y.full && !msg;
        // the XB rows go out now; the qkv GEMM runs this rank's rows first and waits for the peers' rows after them,
        // the wo GEMM the rows the peers need first (reduce-scatter sent), then the private rows (tp_rs_recv after)
        if (shared_in) tp_ag_send(T, {{SPB_XB, XB[0], (size_t)H * 2}});
        else tp_ag_send(T, {{SPB_XB, XB[0], (size_t)H * 2}, {SPB_XB + 1, XB[1], (size_t)H * 2}, {SPB_XB + 2, XB[2], (size_t)H * 2}});
        all(SP_ATT);
        tp_rs_recv(T, ATT);
        rows(SP_C1 | SP_MIN);
        k_cstate<<<(H + 255) / 256, 256, 0, st>>>(ATT, H, T, H, s.ast);
        tp_agP(T, {{SPB_X16, pX16, (size_t)H * 2}, {SPB_IDS, IDS, (size_t)TOPK * 4}, {SPB_WTS, WTS, (size_t)TOPK * 4}});
        all(SP_MOE);
        tp_rs_recv(T, MO);
        rows(SP_C2);
        k_cstate<<<(H + 255) / 256, 256, 0, st>>>(MO, H, T, H, s.mst);
        sp = false;
        spm = 0xff;
        (void)y;
    }
    // ---- [ink-tpf-ppush] / [ink-tpf-pred] plain-step collectives (one row + conv): the producer's output threads
    // push their values to the peers, receive the peers' values, sum in rank order and apply the conv (ink_tpf.cuh);
    // no collective kernel. QE_TP_PP=0: the k_tpar collectives.
    bool tp_pp = !getenv("QE_TP_PP") || atoi(getenv("QE_TP_PP")) != 0;
    uint2* trw8 = nullptr;    // push: receive words of the producer pushes [TP8_SLOTS][N][H]
    // every rank runs the producer of a collective kind (0: wo, 1: down)? Else a rank would push nothing: k_tpar.
    bool tp_all_prod(int kind) const {
        for (int g = 0; g < sh.N; g++) {
            const TpShard& x = tpc->R[g]->sh;
            if (kind == 0 ? x.ce <= x.cb : x.nu() <= 0) return false;
        }
        return true;
    }
    bool tp_ppush(int kind) const { return tps && tp_pp && tpc->push && tp_all_prod(kind); }
    // producer arguments of collective idx (sum, conv1 with state cst, + add -> out); np 0: the producer writes its
    // partial row and tp_ar1 launches k_tpar
    TpPush tp_push_args(int idx, int kind, float* out, float* cst, const float* cw, const float* add) const {
        TpPush p{};
        if (!tp_ppush(kind)) return p;
        p.step = tstep;
        p.idx = idx;
        p.rd8 = trw8 + (size_t)(idx % TP8_SLOTS) * sh.N * H;
        p.ab = tpc->abort_d;
        p.chk = tchk;
        p.out = out;
        p.cst = cst;
        p.cw = cw;
        p.add = add;
        p.n = sh.N;
        p.rank = sh.rank;
        for (int g = 0; g < sh.N; g++)
            if (g != sh.rank) p.dst[p.np++] = tpc->R[g]->trw8 + ((size_t)(idx % TP8_SLOTS) * sh.N + sh.rank) * H;
        return p;
    }
    // [ink-vp-cpush] row steps of nr <= INK_TPCW (<= TPC_MAXR) rows: the column-major producer push (np 0: not used)
    uint4* trwC = nullptr;   // push: receive words [TPC_SLOTS][N][TPC_RW][H]
    int tpcw = getenv("INK_TPCW") ? atoi(getenv("INK_TPCW")) : TPC_MAXR;
    TpCol tp_col_args(int idx, int kind, int nr, float* sum, float* cst, const float* cw, const float* add, float* y, float* hist) const {
        TpCol p{};
        if (!tps || !tpc->push || !trwC || nr < 1 || nr > std::min(tpcw, TPC_MAXR) || !tp_all_prod(kind)) return p;
        p.step = tstep;
        p.idx = idx;
        p.rd = trwC + (size_t)(idx % TPC_SLOTS) * sh.N * TPC_RW * H;
        p.ab = tpc->abort_d;
        p.chk = tchk;
        p.sum = sum;
        p.y = y;
        p.hist = hist;
        p.cst = cst;
        p.cw = cw;
        p.add = add;
        p.rt = DRT;
        p.n = sh.N;
        p.rank = sh.rank;
        p.nr = nr;
        for (int g = 0; g < sh.N; g++)
            if (g != sh.rank) p.dst[p.np++] = tpc->R[g]->trwC + ((size_t)(idx % TPC_SLOTS) * sh.N + sh.rank) * TPC_RW * H;
        return p;
    }
    void tpc_check(int nr, const float* sum, const float* cst, const float* cw, const float* add, const float* y, const float* hist) {
        if (!tpcc_chk || !tpcc_buf) return;   // INK_TPCC_CHK=1: the old chain on the same sums, every output bit compared
        k_tpchain<<<dim3((H + 255) / 256, convseg ? nseg_h : 1), 256, 0, st>>>(sum, nr, cst, cw, add, tpcc_buf, tpcc_buf + (size_t)MROW * H, DRT);
        k_cmpbits<<<(nr * H + 255) / 256, 256, 0, st>>>(y, tpcc_buf, nr * H, tpcc_bad, nullptr);
        k_cmpbits<<<(nr * H + 255) / 256, 256, 0, st>>>(hist, tpcc_buf + (size_t)MROW * H, nr * H, tpcc_bad + 1, tpcc_bad + 2);
    }
    void tp_ar1(int idx, int kind, float* part, float* out, float* cst, const float* cw, const float* add) {
        if (tp_ppush(kind)) return;   // the producer did the sum + conv
        tp_ar(idx, part, 1, 1, out, cst, cw, add);
    }
    unsigned long long* BKEY = nullptr;   // [ink-tp-dec1] the rank's best head key per row
    // the drafter's base logits read the 8-bit screen plane of the WHOLE vocabulary: h2's plane on one card; a TP
    // rank (vocab part in h2) holds a full copy for its replicated drafter (shared by the ranks of a device), or with
    // the split drafter ([ink-tp-drv2a], review5 L3) only the rows of its 64-token chunks (pointers offset by row)
    const uint8_t *dscr = nullptr, *dseb = nullptr;
    void screen_encode(const uint16_t* src, int rows, std::vector<uint8_t>& scr, std::vector<uint8_t>& eb) {
        scr.resize((size_t)rows * H);
        eb.resize(rows);
        const int nth = (int)std::max(1u, std::min(24u, std::thread::hardware_concurrency()));
        std::vector<std::thread> th;
        for (int i = 0; i < nth; i++) {
            const size_t r0 = (size_t)rows * i / nth, r1 = (size_t)rows * (i + 1) / nth;
            th.emplace_back([&, r0, r1] { exp12_encode_screen(src + r0 * H, r1 - r0, H, scr.data() + r0 * H, eb.data() + r0); });
        }
        for (auto& x : th) x.join();
    }
    void tp_dr_screen() {
        const Tens& ue = f.get("model.llm.unembed.weight", "BF16", {VPAD, H});
        if (dr_split()) {   // review5 L3: the split drafter reads only this rank's 64-token chunks [drc0, drc1) of the plane
            const int r0 = drc0() * MKC_CH, r1 = std::min(drc1() * MKC_CH, V);
            std::vector<uint8_t> scr, eb;
            screen_encode((const uint16_t*)ue.p + (size_t)r0 * H, r1 - r0, scr, eb);
            dscr = upraw2<uint8_t>(scr.data(), scr.size()) - (size_t)r0 * HEAD_K;   // indexed by vocabulary row
            dseb = upraw2<uint8_t>(eb.data(), eb.size()) - r0;
            drs_plane = true;
            return;
        }
        dscr = (const uint8_t*)rep_up(ue.p, (size_t)V * H, 8, [&] {
            std::vector<uint8_t> scr, eb;
            screen_encode((const uint16_t*)ue.p, V, scr, eb);
            void* e = upraw2<uint8_t>(eb.data(), eb.size());
            std::lock_guard<std::mutex> l(dseb_mu());
            dseb_map()[std::make_pair(dev, (const void*)ue.p)] = e;
            return (void*)upraw2<uint8_t>(scr.data(), scr.size());
        });
        std::lock_guard<std::mutex> l(dseb_mu());
        dseb = (const uint8_t*)dseb_map()[std::make_pair(dev, (const void*)ue.p)];
    }
    static std::mutex& dseb_mu() {
        static std::mutex m;
        return m;
    }
    static std::map<std::pair<int, const void*>, void*>& dseb_map() {
        static std::map<std::pair<int, const void*>, void*> m;
        return m;
    }
    // [ink-tp-dec1] the drafts of rank 0 on every rank (the replicated drafter is not bitwise the same on every card:
    // cuBLAS / launch-order effects; a different draft on one rank changed the verify rows and hung the collectives):
    // host drafts + probabilities, the drafter time the policy uses, and (dtok) the device drafts k_accept compares
    double tp_dms = 0;
    // [ink-n4-hosthop] the split drafter ends every layer with a rank-order all-reduce and its drafts / probabilities in
    // k_mkmerge on the gathered row: the same bits on every rank by construction, so no host hand-off (no barriers) and
    // the policy's state is copied by the outer engine before the call (tp_pol_copy). INK_HHOP=0: the old hand-off;
    // INK_HHCHK=1: the old hand-off + every rank's drafts / probabilities compared with rank 0's (counts at exit)
    bool hh = !getenv("INK_HHOP") || atoi(getenv("INK_HHOP"));
    bool hh_chk = getenv("INK_HHCHK") && atoi(getenv("INK_HHCHK"));
    unsigned long long hh_n = 0, hh_bad = 0;
    bool tp_hh() const { return tps && hh && dr_split(); }
    void tp_bcast_draft(double* dms, bool dtok, int nb = 1) {
        if (tp_hh() && !hh_chk) return;
        tp_dms = dms ? *dms : 0;
        tpc->bar.wait();
        if (sh.rank) {
            const Impl& r = *tpc->R[0];
            if (hh_chk && tp_hh()) {
                hh_n++;
                bool diff = false;   // review night4 L1: only the written entries (drafts 1..7, probabilities 1..7 per slot)
                for (int z = 0; z < nb; z++)
                    for (int j = 1; j < DB; j++) diff = diff || hdr[z * DB + j] != r.hdr[z * DB + j] || memcmp(&hdp[z * 16 + j], &r.hdp[z * 16 + j], 4);
                if (diff) {
                    if (!hh_bad) fprintf(stderr, "[ink] INK_HHCHK rank %d: drafts differ from rank 0 (block %llu)\n", sh.rank, hh_n);
                    hh_bad++;
                }
            }
            memcpy(hdr, r.hdr, 1024);
            memcpy(hdp, r.hdp, 1024);
            if (dms) *dms = r.tp_dms;
        }
        tpc->bar.wait();
        if (sh.rank && dtok) CU(cudaMemcpyAsync(DTOK, hdr, DB * 4, cudaMemcpyHostToDevice, st));
    }
    // [ink-tp-dec1] tag wrap (tags = step x 256 + collective + 1, 32 bits: alias after 2^24 steps): every rank counts its
    // step launches (the same call sequence on every rank) and, past the limit, all ranks zero their receive words and
    // step counters between two calls (QE_TP_CLEAR = limit, tests)
    long long tp_launch = 0;
    // review6 L2: at most 2^22 launches between clears (tags below ~1e-19 as fp32: k_tpart leaves untagged payload words in
    // the receive rows that a later k_tpar polls)
    long long tp_clear_at = std::min(getenv("QE_TP_CLEAR") ? atoll(getenv("QE_TP_CLEAR")) : (1ll << 22), 1ll << 22);
    long long tp_clears = 0;
    void tp_reset_words() {   // review tp M2: after a drain (no collective in flight on any rank)
        if (!tps || !tpc->push) return;
        CU(cudaMemset(trw, 0, (size_t)TP_SLOTS_ALL * sh.N * MROW * TPW * 16));
        CU(cudaMemset(trw8, 0, (size_t)TP8_SLOTS * sh.N * H * 8));   // [ink-tpf-ppush]
        if (trwC) CU(cudaMemset(trwC, 0, (size_t)TPC_SLOTS * sh.N * TPC_RW * H * 16));   // [ink-vp-cpush]
        CU(cudaMemset(tstep, 0, 4));
        CU(cudaDeviceSynchronize());
        tp_launch = 0;
    }
    void tp_chk() {   // [ink-tp-r2-fmt] every rank's hashes == rank 0's (all calls since the last compare finished)
        CU(cudaStreamSynchronize(st));
        CU(cudaMemcpy(tchk_h, tchk, sizeof tchk_h, cudaMemcpyDeviceToHost));
        if (sh.rank == sh.N - 1 && getenv("QE_TP_CHK_FAULT") && tchk_calls >= atoll(getenv("QE_TP_CHK_FAULT"))) tchk_h[7] ^= 1u;   // gate self-test
        tpc->bar.wait();
        const Impl& r = *tpc->R[0];
        for (int i = 0; i < 256; i++)
            if (tchk_h[i] != r.tchk_h[i]) {
                fprintf(stderr, "[ink] QE_TP_CHK MISMATCH rank %d collective %d after %lld calls: %08x != rank 0 %08x\n", sh.rank, i,
                        tchk_calls, tchk_h[i], r.tchk_h[i]);
                fflush(stderr);
                _exit(86);
            }
        tchk_cmp++;
        if (sh.rank == 0 && tchk_cmp % 1000 == 0) fprintf(stderr, "[ink] QE_TP_CHK %lld compares, %lld calls: equal\n", tchk_cmp, tchk_calls);
        tpc->bar.wait();
    }
    void tp_tick() {
        if (tchk && ++tchk_calls % tchk_every == 0) tp_chk();
        if (!tps || !tpc->push) return;
        tp_launch += 2;   // at most two step-counter increments per call (body + an extra head)
        if (tp_launch < tp_clear_at) return;
        CU(cudaStreamSynchronize(st));
        tpc->bar.wait();   // nobody is inside a collective
        CU(cudaMemset(trw, 0, (size_t)TP_SLOTS_ALL * sh.N * MROW * TPW * 16));
        CU(cudaMemset(trw8, 0, (size_t)TP8_SLOTS * sh.N * H * 8));   // [ink-tpf-ppush]
        if (trwC) CU(cudaMemset(trwC, 0, (size_t)TPC_SLOTS * sh.N * TPC_RW * H * 16));   // [ink-vp-cpush]
        CU(cudaMemset(tstep, 0, 4));
        CU(cudaDeviceSynchronize());
        tpc->bar.wait();
        tp_launch = 0;
        tp_clears++;
    }
    // [ink-tp-dec1] the spec policy's host state (timings) of rank 0 on every rank before a step: identical decisions
    void tp_pol_sync() {
        if (tp_hh()) return;   // [ink-n4-hosthop] copied by the outer engine before the call (tp_pol_copy)
        tpc->bar.wait();
        if (sh.rank) tp_pol_from(*tpc->R[0]);
        tpc->bar.wait();
    }
    void tp_pol_from(const Impl& r) {
        {
            e_tp = r.e_tp;
            fb_ema = r.fb_ema;   // [ink-samp-dev] review M2
            std::copy(r.e_tv, r.e_tv + VR + 1, e_tv);
            e_rn = r.e_rn;
            e_rt = r.e_rt;
            plain_left = r.plain_left;
            since_tp = r.since_tp;
            n_ds = r.n_ds;
            plain_run = r.plain_run;
            memcpy(e_rows_b, r.e_rows_b, sizeof e_rows_b);
            memcpy(e_dr, r.e_dr, sizeof e_dr);
        }
    }
    void tp_max(int idx, unsigned long long* key, int* arg, int nr = 1) {
        A_tpmax a{};
        a.t.n = sh.N;
        a.t.rank = sh.rank;
        a.t.push = tpc->push;
        a.t.step = tstep;
        a.t.rd = trw;
        a.t.ab = tpc->abort_d;
        a.t.chk = tchk;
        a.key = key;
        a.idx = idx;
        a.arg = arg;
        a.nr = nr;
        if (tpc->push) {
            for (int g = 0; g < sh.N; g++) a.t.src[g] = tpc->R[g]->trw;
        } else {
            const int sl = idx % TP_SLOTS;
            CU(cudaEventRecord(tev[sl], st));
            tpc->bar.wait();
            for (int g = 0; g < sh.N; g++) {
                if (g != sh.rank) CU(cudaStreamWaitEvent(st, tpc->R[g]->tev[sl], 0));
                a.t.src[g] = tpc->R[g]->BKEY;
            }
        }
        k_tpmax<<<1, 32, 0, st>>>(a);
    }
    Layer L[NL];
    AppState A[NAPP];    // the selected slot's view (prefill / plain step); row steps use A0 + the row table
    AppState A0[NAPP];   // slot 0 = base of every per-slot array ([S][...])
    int S = 1, cur = 0;  // slots, selected slot
    // [ink-srv-pool] full-attention rows: one pool [NKV][POOL][HD] per application; slot sl holds [sbase, sbase + slim)
    // (its position sl's spos); the drafter's global layer the same interval of its own pool. Device tables of slot
    // offsets in HD-rows for the row-step kernels: rings (sl x NKV x RING), pool (sbase), drafter rings (sl x NKV x dcap)
    int POOL = 0;
    std::vector<int> sbase, slim;
    int *SKO_R = nullptr, *SKO_G = nullptr, *SKO_DR = nullptr;
    // [ink-srv-samp] per-slot sampling (1 / temperature, 0 = greedy; seed) and the per-row copy of a step: pinned HSAMP
    // = [MROW] 1 / T, [MROW] seeds, [4] a position; DSAMP the device copy (in the step's graph, like the row table)
    std::vector<float> sT;
    std::vector<unsigned> sS;
    // [ink-samp-dev] per-slot filter (sK = list size | FK_CERT, 0 = none; top_p; min_p) and its per-row copy after the
    // position words: [MROW] K, [MROW] top_p, [MROW] min_p
    std::vector<int> sK;
    std::vector<float> sTP, sMP;
    static constexpr int SAMP_N = 5 * MROW + 4;   // floats of HSAMP / DSAMP
    float *HSAMP = nullptr, *DSAMP = nullptr;
    float* sampT(float* b) const { return b; }
    unsigned* sampS(float* b) const { return (unsigned*)(b + MROW); }
    int* sampP(float* b) const { return (int*)(b + 2 * MROW); }
    int* sampK(float* b) const { return (int*)(b + 2 * MROW + 4); }
    float* sampTP(float* b) const { return b + 3 * MROW + 4; }
    float* sampMP(float* b) const { return b + 4 * MROW + 4; }
    void samp_row(int r, int sl) {
        HSAMP[r] = sT[sl];
        sampS(HSAMP)[r] = sS[sl];
        sampK(HSAMP)[r] = sT[sl] > 0.f ? sK[sl] : 0;
        sampTP(HSAMP)[r] = sTP[sl];
        sampMP(HSAMP)[r] = sMP[sl];
    }
    void samp_up() { CU(cudaMemcpyAsync(DSAMP, HSAMP, SAMP_N * 4, cudaMemcpyHostToDevice, st)); }
    // [ink-samp-dev] rows [0, k) of the staged step: any filtered row (the head launches its filter kernels)
    bool filt_rows(int k) const {
        if (!loR) return false;
        for (int r = 0; r < k; r++)
            if (HSAMP[r] > 0.f && (sampK(HSAMP)[r] & 0xff)) return true;
        return false;
    }
    float* loR = nullptr;
    unsigned *fcR = nullptr, *fhist = nullptr;
    unsigned long long *flist = nullptr, *FLOC = nullptr;
    double* ftail = nullptr;
    cudaEvent_t fev = nullptr;   // direct TP: this rank's list is ready
    // tests (review M3): INK_FFORCE=N: every filtered row at a position divisible by N goes to the host pick
    int filt_force = getenv("INK_FFORCE") ? atoi(getenv("INK_FFORCE")) : 0;
    void samp_args(HeadArgs& h, const int* rp) const {
        h.rT = sampT(DSAMP);
        h.rS = sampS(DSAMP);
        h.rP = rp;
        if (loR) {
            h.rK = sampK(DSAMP);
            h.rTP = sampTP(DSAMP);
            h.rMP = sampMP(DSAMP);
            h.lo = loR;
            h.fc = fcR;
            h.fhist = fhist;
            h.flist = flist;
            h.ftail = ftail;
        }
    }
    // [ink-samp-dev] the filter kernels of a head over nr rows: threshold passes (after the screen), then after the
    // rescore (and the TP argmax) the rank's top K, the exchange, the global pick into ARG
    void filt_thr(const HeadArgs& h, int nr) {
        k_fthr<0><<<dim3(48, nr), 256, 0, st>>>(h);
        k_fthr<1><<<dim3(48, nr), 256, 0, st>>>(h);
    }
    void filt_tight(const HeadArgs& h, int nr) {   // round A's list -> fc[1] = its K-th largest exact logit
        A_fsel a{};
        a.h = h;
        a.loc = FLOC;
        a.mode = 1;
        k_fsel<<<nr, 1024, 0, st>>>(a);
    }
    void filt_pick(const HeadArgs& h, int nr) {
        A_fsel a{};
        a.h = h;
        a.loc = FLOC;
        a.arg = ARG;
        a.idx = 2 * NAPP + 1;
        a.tp = tps ? 1 : 0;
        a.force = filt_force;
        if (tps) {
            a.t.n = sh.N;
            a.t.rank = sh.rank;
            a.t.push = tpc->push;
            a.t.step = tstep;
            a.t.rd = trw;
            a.t.ab = tpc->abort_d;
            if (tpc->push)
                for (int g = 0; g < sh.N; g++) a.t.src[g] = tpc->R[g]->trw;
        }
        k_fsel<<<nr, 1024, 0, st>>>(a);
        if (tps && !tpc->push) {
            CU(cudaEventRecord(fev, st));
            tpc->bar.wait();
            for (int g = 0; g < sh.N; g++) {
                if (g != sh.rank) CU(cudaStreamWaitEvent(st, tpc->R[g]->fev, 0));
                a.t.src[g] = tpc->R[g]->FLOC;
            }
        }
        k_fmerge<<<nr, 1024, 0, st>>>(a);
    }
    void sko_upload() {
        std::vector<int> r(S), g(S), d(S);
        for (int i = 0; i < S; i++) {
            r[i] = i * nkvA() * RING;
            g[i] = sbase[i];
            d[i] = i * NKV * dcap[0];
        }
        CU(cudaMemcpyAsync(SKO_R, r.data(), S * 4, cudaMemcpyHostToDevice, st));
        CU(cudaMemcpyAsync(SKO_G, g.data(), S * 4, cudaMemcpyHostToDevice, st));
        CU(cudaMemcpyAsync(SKO_DR, d.data(), S * 4, cudaMemcpyHostToDevice, st));
        CU(cudaStreamSynchronize(st));
    }
    const int* dsko(int i) const { return (i == 1 && !getenv("INK_DR_TEST")) ? SKO_G : SKO_DR; }
    std::vector<int> spos;                 // per-slot positions (spos[cur] is stale while selected: pos is live)
    std::vector<std::vector<int>> shist;   // per-slot token histories (hist is live for cur)
    int *DRT = nullptr, *HRT = nullptr;    // row table (device / pinned), RT_* layout
    int rpos_h[MROW] = {}, nseg_h = 1;
    bool rows_batch = false;   // the row step's tokens come from HTOK (step_rows) instead of the drafter's DTOK
    int* HTOK = nullptr;       // host copy: row positions, segments of the current row step
    AppState view(int a, int sl) const {
        AppState v = A0[a];
        const size_t ko = v.ring ? (size_t)sl * nkvA() * v.cap * HD : (size_t)sbase[sl] * HD;   // [ink-srv-pool] global rows: the slot's interval
        v.K += ko;
        v.V += ko;
        v.kst += (size_t)sl * 3 * KVD;
        v.vst += (size_t)sl * 3 * KVD;
        v.ast += (size_t)sl * 3 * H;
        v.mst += (size_t)sl * 3 * H;
        return v;
    }
    bf16* emb_host = nullptr;   // mmapped embedding (rows gathered on the host)
    bf16 *embn = nullptr, *fnorm = nullptr, *unemb = nullptr;
    G12Args head12{};
    HeadArgs h2{};   // two-stage head (INK_HEAD2=0: the full 12-bit head)
    bool head2 = true;
    bf16* WS = nullptr;   // prefill: one layer's qkv / wo planes decoded to BF16 for cuBLAS (only dec16 writes it; else ws_invalidate())
    const void* ws_tag = nullptr;   // planes WS currently holds (dec16)
    float *outq = nullptr, *msgq = nullptr, *msgkv = nullptr;
    cublasHandle_t blas;
    cudaStream_t st;
    size_t dev_bytes = 0;
    int pos = 0, last_T = 1;
    // buffers (CH rows)
    bf16 *emb_rows_h = nullptr, *emb_rows_d = nullptr;
    float *E, *X6, *X12[2], *H17[2], *X18, *HA, *HB, *XA[3], *XG[3], *M[4], *Rr, *ATT, *MI, *MIG, *XMF, *MO, *P, *Qf,
        *Rf, *Sf, *ACT, *WTS, *FIN, *LOG;
    bf16 *XB[3], *AO, *XMB, *FINB;
    int *IDS, *ARG, *dpos, *ACNT, *RCNT;
    OrdBuf* ORD = nullptr;   // [ink-b-xord] per-application entry order (row steps)
    float* DPART = nullptr;  // [ink-b-dnsh] per-entry down partials [MROW][TOPK][H]
    unsigned* BCHK = nullptr;  // [ink-bres-gram] precision check counters (INK_BRES_CHK=1)
    float* MN = nullptr;       // [ink-bres-gram] loop-message squared norms [12][VR][4]
    float *ZB, *SC, *PM, *PL, *PA, *M48, *BMAX, *RFP, *RFS;
    int* hpos = nullptr;   // pinned: position of the first token of the next forward
    cudaGraphExec_t gstep = nullptr;
    bool cap_filt = false;   // [ink-samp-dev] the step graph holds the filter kernels
    int cap_C[2] = {0, 0};   // [ink-attn-tb] chunk keys the step graph was captured with
    cudaEvent_t ev_stage;
    // [ink-tp-pf2] TP prefill in two half-chunks on two streams: half A runs body() on st, half B on st2 with every
    // application gated by A's event of the same application (K / V rows, rings, conv states of A first); the two halves
    // use disjoint rows of every row buffer (B = the row offset TA) and their own per-call scratch, so A may run ahead and
    // its collective copies (copy engines) overlap B's kernels
    // generalised to K sub-chunks (INK_TPPFK, default 2): sub-chunk j waits for j - 1's application; slot 0 of the arrays
    // below is unused (sub-chunk 0 runs on the members themselves)
    static constexpr int PFK_MAX = 6;
    bool pf2 = false;
    int pfk = 2;
    int pf_half = -1;   // -1: one body; j: sub-chunk j
    cudaStream_t st2[PFK_MAX] = {}, cps2[PFK_MAX] = {};
    cublasHandle_t blas2[PFK_MAX] = {};
    cudaEvent_t pfEv[PFK_MAX][NAPP] = {}, pfJ[PFK_MAX] = {};
    int *dpos2[PFK_MAX] = {}, *hpos2[PFK_MAX] = {};
    bf16* WS2[PFK_MAX] = {};
    const void* ws_tag2[PFK_MAX] = {};
    int *pOFF2[PFK_MAX] = {}, *pUNITS2[PFK_MAX] = {}, *pNU2[PFK_MAX] = {};
    float *pZp2[PFK_MAX] = {}, *pZs2[PFK_MAX] = {};
    int pf_npmax = 0;
    bool stage_pending = false, capturing = false;
    int ab = 0;
    bool cmp_on = getenv("INK_ATTN_CMP") != nullptr;
    bool ustat = getenv("INK_USTAT") != nullptr;
    int bres2_min = getenv("INK_BR2MIN") ? atoi(getenv("INK_BR2MIN")) : 8;   // [ink-b-bres2] two-phase bres of row steps from T rows (A/B 132: off; k2-k6 verify lose 0.1-0.3 ms with it)
    int bres2_tt = getenv("INK_BR2TT") ? atoi(getenv("INK_BR2TT")) : 4;     // tokens per phase-B block (z)
    int bres2_ttr = getenv("INK_BR2TR") ? atoi(getenv("INK_BR2TR")) : 4;    // tokens per phase-B block (router)
    float* XZ = nullptr;
    int rpw_min = getenv("INK_RPW") ? atoi(getenv("INK_RPW")) : 5;   // [ink-bres-rpw] verify router bres with 2 experts per warp from this k
    bool convseg = !getenv("INK_CONVSEG") || atoi(getenv("INK_CONVSEG"));   // [ink-tp-convseg] row-step conv chains one block row per segment
    int x8sh_min = getenv("INK_X8SH") ? atoi(getenv("INK_X8SH")) : 64;   // [ink-tp-x8sh] shared-decode MXFP8 gate / up above this many entries (INK_X8SH=-1: the old k_gu8)
    int tcx_min = getenv("INK_TCX_MIN") ? atoi(getenv("INK_TCX_MIN")) : 0;   // [ink-tp-tcx] tensor-core experts from this many rows (0: off; not bitwise)
    int tcx_parts = getenv("INK_TCX_PARTS") ? atoi(getenv("INK_TCX_PARTS")) : 15;   // bits: 1 NVFP4 gate/up, 2 NVFP4 down, 4 MXFP8 gate/up, 8 MXFP8 down (down needs its gate/up)
    int tcx_dv = getenv("INK_TCX_DV") ? atoi(getenv("INK_TCX_DV")) : 0;   // NVFP4 down form
    int tcx_v = getenv("INK_TCX_V") ? atoi(getenv("INK_TCX_V")) : 2;   // NVFP4 gate/up stage form (2: NS3 SUB2, 3 blocks / SM: best of 0-5 at 16 / 32 rows)
    __half *TXH = nullptr, *TXL = nullptr, *TAH = nullptr, *TAL = nullptr;
    int* TPX = nullptr;
    OrdBuf* ORD8 = nullptr;
    bool tcx_chk = getenv("INK_TCX_CHK") != nullptr;
    double tcx_cs[2][4] = {};
    int gu4p_min = getenv("INK_GU4P") ? atoi(getenv("INK_GU4P")) : 3;   // [ink-gu4-pair] from this many verify rows (vbench 2K with the single-entry path: k2 +0.05, k3 -0.15, k4 -0.22 ms)
    long long cmp3_n = 0, cmp3_bad = 0, cmptb3_n = 0, cmptb3_bad = 0;
    double us_n[2][MROW + 1] = {}, us_u[2][MROW + 1] = {}, us_p[2][MROW + 1] = {};   // + pair jobs (k_gu4o blocks)
    double cmp_err[3] = {0, 0, 0}, cmp_ref = 0;
    std::vector<double> cmp_g[4], cmp_gm[4];   // [v + 2 local]: global / local layers, per call: rel. L2 error, max |d| / max |ref| of TC FP16-P (short form) and k_attn_tb
    int kps = 128;          // keys per split target (A/B: 128 vs 32 -2.4 % at 512 / -2.6 % at 4K; 256 +2.0 %)
    int cap_ns[2] = {0, 0};   // split counts the step graph was captured with   // A/B variant switch for same-process speed comparisons (bench --ab)
    // attention splits: decode (T = 1) a fixed count (graph-safe; the key range per split is computed on the device),
    // prefill bounded by the partial buffer (T * NS <= PSPLIT)
    static constexpr int PSPLIT = 16384;
    int attn_splits(int T, bool local) const {
        if (T == 1) {
            {   // depth buckets: ~kps keys per split (graph recaptured when a bucket changes); A/B vs fixed 32 / 128: -0.9 % at 512, = at 4K / 32K
                const int n = local ? std::min(pos + 1, WIN) : pos + 1;
                int ns = 1;
                const int kp = kps;
                while (ns < (local ? 32 : 128) && ns * kp < n) ns *= 2;
                return std::max(ns, 4);
            }
        }
        const int range = local ? std::min(pos + T, WIN) : pos + T;
        return std::max(1, std::min(std::min(PSPLIT / T, 64), (range + 255) / 256));
    }
    // ---- speculative decoding ([ink-dspark-*])
    int vR = 0;                       // > 0 while a verify step of vR rows runs through body()
    float tau = 0.8f;   // confidence stop (re-tune 11-22, full answers: 0.8 > 0.7 > 0.6 > 0.5 on all 9): rows while the running product of the drafter's p >= tau; [ink-dr-drop] the trained drafter: TAU_QE 0.5
    int kmin = 2, lk_min = 0;   // rows at least; prompt lookup on a match of >= lk_min tokens (0 = off)
    // [ink-dpol] depth-aware fallback: measured costs (host ms, EMAs) of a plain step, verify per k, drafter block, and
    // the drafted steps' tokens / time; plain steps while spec decoding is slower than plain (re-probed), k capped
    // where the expected tokens per ms of the policy's k lose to plain (output unchanged: plain == verify)
    bool pol = !getenv("INK_POL") || atoi(getenv("INK_POL"));
    int pol_min = getenv("INK_POLMIN") ? atoi(getenv("INK_POLMIN")) : 0;   // [ink-tp-polmin] active from this depth (11-24: 0 = every depth, any N; TP2 chat 256 capital 152.6 -> 180.1, tcp / code -2.5 %, full answers level, one card level; INK_POLMIN=32768 the old rule). Old note (short depth: spec > plain on all 9 answers; the switch only lost there to noise)
    double e_tp = 0, e_tv[VR + 1] = {}, e_rn = 0, e_rt = 0;
    int plain_left = 0, since_tp = 0, n_ds = 0, plain_run = 32;
    long long st_pl = 0, st_kcap = 0;
    // samples capped at 2x the mean: a graph (re)capture at a depth bucket costs ms and must not flip the policy
    // the first sample is dropped (e = -1: it usually carries a graph capture), later ones capped at 2x the mean
    // (times only: token counts are never capped, cap = false)
    static void ema(double& e, double x, double a, bool cap = true) {
        if (e == 0) e = -1;
        else if (e < 0) e = x;
        else e = (1 - a) * e + a * (cap ? std::min(x, 2 * e) : x);
    }
    static constexpr double POL_M = 0.90;   // act only on a clear loss (0.97: short-depth py -2.3 % from noise-driven switches)
    float tau1 = 0.6f;   // [ink-tau1] one row when the first draft's p < tau1 (full answers 11-23: 0.6 >= 0.4 > off; INK_TAU1=0 off)
    int wrow_max = 3;                 // verify GEMVs: warp per row up to this many rows, else the row-loop kernels (A/B at 512: W better k <= 3, row loop k >= 4)
    CommitEnt* commit_ents = nullptr;
    int *ACC = nullptr, *VTOK = nullptr, *IOTA = nullptr;
    float* hiR = nullptr;
    unsigned* hcR = nullptr;
    // [ink-attn-tb] chunk keys for the row at position p: ~splits1 chunks, power of 2 >= 16
    // [ink-attn-tb-r4] the shared-row form only where attention is bytes-bound: global layers at n = p + 1 >= tb_min
    // (a rule of the position, so a verify row and the plain step at the same position use the same form)
    // [ink-b-attn-tb] default: the shared-row form (k_attn_tb3) at every position, local and global (one form for plain,
    // verify and batch rows; 8 x 4 rows at ~3K: attention 10.88 -> 6.13 ms per step, plain +1.1 %); INK_TBMIN=24576
    // INK_TBLMIN=0 = the round-5 rule
    int tb_min = getenv("INK_TBMIN") ? atoi(getenv("INK_TBMIN")) : 1;   // vbench 16K: plain +0.07 ms, k2 +0.18, k4 -1.15, k8 -2.9; 32K: plain level, k4 -3.2, k8 -6.7
    int tbt = getenv("INK_TBT") ? atoi(getenv("INK_TBT")) : 5;          // [ink-b-attn-tb] k_attn_tb3 rows per tile (<= 5)
    int tb_minb = getenv("INK_TBMINB") ? atoi(getenv("INK_TBMINB")) : 256;   // grids below this many blocks (at 5-row tiles): smaller tiles
    int tb_s1 = getenv("INK_TBS1") ? atoi(getenv("INK_TBS1")) : 0;   // [ink-tp-tb1] latency-bound grids: 1-row tiles up to this many rows (0: TP 6, one card 3; review7 M2: one card 4 / 6 rows at depth 1,000 +0.21 ms with 6)
    int tb_n1 = getenv("INK_TBN1") ? atoi(getenv("INK_TBN1")) : 640;   // [ink-tp-tb1] ... or at any rows while a row has <= this many keys
    int tb_nkv = getenv("INK_TBNKV") ? atoi(getenv("INK_TBNKV")) : 1;   // [ink-tp-tb1] 0: the latency test counts NKV (old)
    int tbl_min = getenv("INK_TBLMIN") ? atoi(getenv("INK_TBLMIN")) : 1;   // local layers: shared-row form from this n (0: never)
    bool tbOn(int p, bool local) const { return ab != 80 && (ab == 124 || ab == 128 || (local ? tbl_min > 0 && p + 1 >= tbl_min : p + 1 >= tb_min)); }   // A/B 124: tb at every position
    int tbK(int p, bool local) const { return tbOn(p, local) ? tbC(p, local) : 0; }   // graph keys
    int tbC(int p, bool local) const {
        const int n = local ? std::min(p + 1, WIN) : p + 1, ns = splits1(p, local);
        const int kp = (n + ns - 1) / ns;   // the relative split length; chunk = its power-of-2 floor (<= 2 ns + 1 chunks)
        int c = 16;
        while (2 * c <= kp) c *= 2;
        if (ab == 127 || ab == 124 + 4) c *= 2;   // A/B 127 / 128 (with tb everywhere): chunks of 2x the keys
        return c;
    }
    // [ink-b-attn-tb] chunk rows of a tb launch: <= 2 ns + 1 chunks per row; while ns is below its cap a row has
    // n <= ns x kps keys, so ceil(ns x kps / C) + 1 chunks (+1: an unaligned local window); the extra blocks exit
    int tbY(int nsx, int cmin, bool local) const {
        const int y0 = 2 * nsx + 3;
        if (ab == 126 || cmin <= 0 || nsx >= (local ? 32 : 128)) return y0;   // A/B 126: the old bound
        const int n = local ? std::min(nsx * kps, WIN) : nsx * kps;
        return std::min(y0, (n + cmin - 1) / cmin + 2);
    }
    int splits1(int p, bool local) const {   // attn_splits(1, local) at position p
        const int n = local ? std::min(p + 1, WIN) : p + 1;
        int ns = 1;
        while (ns < (local ? 32 : 128) && ns * kps < n) ns *= 2;
        return std::max(ns, 4);
    }
    bool dr_on = false;
    // [ink-dr-nsg] drafter global-layer attention splits by depth (vbench --fake: 128 vs 32 at 100K 2.87 vs 3.01 ms,
    // at 512 1.57 vs 1.52): 32 below 64K, 128 from 64K (the drafter graph is recaptured on a change)
    int dns_cap = 0;
    int dns_g(int p) const { return p + DB >= 65536 ? DNS_GX : DNS_G; }
    File df;
    DLayer DL[3];
    bf16 *d_mnorm = nullptr, *d_mask = nullptr, *d_norm = nullptr, *d_Pse = nullptr, *d_Ph = nullptr, *d_Pf = nullptr, *d_Pb = nullptr,
         *d_W1 = nullptr;
    __half* d_UO = nullptr;
    int8_t* d_UO8 = nullptr;
    float* d_us = nullptr;
    bool dr_exact = false;
    float* d_uOb = nullptr;
    float d_fw[16];
    uint8_t *DK[3] = {}, *DV[3] = {};   // [ink-dr-fp8] E4M3 [NKV][cap][HD]
    float *DKS[3] = {}, *DVS[3] = {};   // scales [NKV][cap]
    int dcap[3] = {0, 0, 0};
    size_t dkstr[3] = {0, 0, 0}, dsstr[3] = {0, 0, 0};   // [ink-b-draft] per sequence slot: K / V bytes, scales
    int NDS = 1;                                          // sequences per drafter block (row steps)
    bf16 *FMS = nullptr, *FMB = nullptr;                  // f_3 per slot; gathered rows of a block
    int *dsl = nullptr, *hbd = nullptr;                   // [segment] slots (device); pinned [anchor pos | slot | token]
    float *TAP[3] = {}, *DKVb = nullptr, *Dh = nullptr, *DQKVb = nullptr, *DQ = nullptr, *DPM = nullptr, *DPL = nullptr, *DPA = nullptr,
          *DGU = nullptr, *BASE = nullptr, *Zp = nullptr, *Fz = nullptr, *MS = nullptr, *MRO = nullptr, *PMX = nullptr, *PSX = nullptr,
          *DPROB = nullptr;
    bf16 *DF[4] = {}, *DX = nullptr, *DAO = nullptr, *DA = nullptr, *HIN = nullptr, *FM = nullptr, *demb_d = nullptr, *demb_h = nullptr;
    int *PIX = nullptr, *DTOK = nullptr, *dan = nullptr, *han = nullptr;
    unsigned* MCNT = nullptr;
    int nbl = 0;
    int* MKCAND = nullptr;   // [ink-dr-topc] candidates [7][MKC_NC], tail (max, sum) [7][MKC_NCH]
    float *MKTM = nullptr, *MKTS = nullptr, *MKPAX = nullptr;
    std::vector<float> dump;   // per-application outputs (dump mode)
    std::vector<int> route;    // per-application top-8 ids (dump mode)
    bool dumping = false;
#include "ink_pf_impl.inc"   // prefill path ([ink-pf-*])

    float* upf(const Tens& t) {   // bf16 / f32 tensor -> fp32 device ([ink-tp-dec1] shared by the ranks of a device)
        return (float*)rep_up(t.p, t.n, 1, [&] { return (void*)upf_(t); });
    }
    float* upf_(const Tens& t) {
        std::vector<float> h(t.n / (t.dtype == "F32" ? 4 : 2));
        if (t.dtype == "F32") memcpy(h.data(), t.p, t.n);
        else {
            ICHECK(t.dtype == "BF16", "dtype %s", t.dtype.c_str());
            for (size_t i = 0; i < h.size(); i++) h[i] = bf2f(((const uint16_t*)t.p)[i]);
        }
        float* d = (float*)dmalloc(h.size() * 4, &dev_bytes);
        CU(cudaMemcpy(d, h.data(), h.size() * 4, cudaMemcpyHostToDevice));
        return d;
    }
    // tc: [ink-tcr] fragment-ordered planes for the tensor-core GEMV (idx == nullptr marks them)
    // Sx > 0 ([ink-tp-dec1]): fragment order with Sx K parts (k_g12tcR: Sx = K / 256); no GPU round trip (the decode
    // kernels are compiled for the full widths; the encoder is the same lossless one)
    G12Args up12(const uint16_t* A, int rows, int K, bool tc = false, int Sx = 0, bool qkv = false) {
        // [ink-vp-qs] INK_QSQ=3: the qkv planes in 3 K parts (opt-in: c = 4 / 8 -2 %); review L1: only with tcr_S<H> == 1
        static const bool qsq3 = getenv("INK_QSQ") && atoi(getenv("INK_QSQ")) == 3;
        if (qkv && tc && K == H && tcr_SQ == 3 && tcr_S<H> == 1 && qsq3) return up12q(A, rows);
        Enc12 e;
        encode12(A, rows, K, &e);
        G12Args g{};
        auto up = [&](const void* p, size_t n) {
            void* d = dmalloc(std::max<size_t>(n, 16), &dev_bytes);
            a12 += std::max<size_t>(n, 16);   // [ink-pf8] review L6
            if (n) CU(cudaMemcpy(d, p, n, cudaMemcpyHostToDevice));
            return d;
        };
        if (tc) {
            const std::vector<uint32_t> P = repack12T(e, K, Sx > 0 ? Sx : K == QD ? tcr_S<QD> : tcr_S<H>);
            g.sm = (const uint4*)up(P.data(), P.size() * 4);
            g.idx = nullptr;
        } else {
            g.sm = (const uint4*)up(e.sm.data(), e.sm.size());
            g.idx = (const uint32_t*)up(e.idx.data(), e.idx.size() * 4);
        }
        g.meta = (const uint4*)up(e.meta.data(), e.meta.size());
        g.fb = (const bf16*)up(e.fb.data(), e.fb.size() * 2);
        g.a.N = rows;
        b12 += e.sm.size() + e.idx.size() * 4 + e.meta.size() + e.fb.size() * 2;
        if (!getenv("INK_NO_VERIFY12") && Sx == 0) {   // decode on the GPU and compare with the original bits
            const size_t nb = (size_t)rows * K * 2;
            uint32_t *orig = nullptr, *dec = nullptr;
            unsigned long long* bad = nullptr;
            CU(cudaMalloc(&orig, nb));
            CU(cudaMalloc(&dec, nb));
            CU(cudaMalloc(&bad, 8));
            CU(cudaMemcpy(orig, A, nb, cudaMemcpyHostToDevice));
            CU(cudaMemset(bad, 0, 8));
            if (tc && K == H) k_dec12T2<H><<<(rows / 2 + 7) / 8, (256 * tcr_S<H>)>>>(g, dec);
            else if (tc) k_dec12T2<QD><<<(rows / 2 + 7) / 8, (256 * tcr_S<QD>)>>>(g, dec);
            else if (K == H) k_dec12<H><<<(rows / 2 + 7) / 8, 256>>>(g, dec);
            else k_dec12<QD><<<(rows / 2 + 7) / 8, 256>>>(g, dec);
            k_cmp_u32<<<1024, 256>>>(orig, dec, nb / 4, bad);
            unsigned long long hb = 0;
            CU(cudaMemcpy(&hb, bad, 8, cudaMemcpyDeviceToHost));
            CU(cudaFree(orig));
            CU(cudaFree(dec));
            CU(cudaFree(bad));
            ICHECK(hb == 0, "G15 round trip: %llu words differ (%d x %d)", hb, rows, K);
            verified12++;
        }
        return g;
    }
    // [ink-vp-qs] the qkv planes in the tcr_SQ-part fragment order (decode-verified with the matching decoder)
    G12Args up12q(const uint16_t* A, int rows) {
        Enc12 e;
        encode12(A, rows, H, &e);
        G12Args g{};
        auto up = [&](const void* p, size_t n) {
            void* d = dmalloc(std::max<size_t>(n, 16), &dev_bytes);
            a12 += std::max<size_t>(n, 16);   // [ink-pf8] review L6
            if (n) CU(cudaMemcpy(d, p, n, cudaMemcpyHostToDevice));
            return d;
        };
        const std::vector<uint32_t> P = repack12T(e, H, tcr_SQ);
        g.sm = (const uint4*)up(P.data(), P.size() * 4);
        g.idx = nullptr;
        g.meta = (const uint4*)up(e.meta.data(), e.meta.size());
        g.fb = (const bf16*)up(e.fb.data(), e.fb.size() * 2);
        g.a.N = rows;
        g.ks = tcr_SQ;
        b12 += e.sm.size() + e.idx.size() * 4 + e.meta.size() + e.fb.size() * 2;
        if (!getenv("INK_NO_VERIFY12")) {
            const size_t nb = (size_t)rows * H * 2;
            uint32_t *orig = nullptr, *dec = nullptr;
            unsigned long long* bad = nullptr;
            CU(cudaMalloc(&orig, nb));
            CU(cudaMalloc(&dec, nb));
            CU(cudaMalloc(&bad, 8));
            CU(cudaMemcpy(orig, A, nb, cudaMemcpyHostToDevice));
            CU(cudaMemset(bad, 0, 8));
            k_dec12T2<H, tcr_SQ><<<(rows / 2 + 7) / 8, 256 * tcr_SQ>>>(g, dec);
            k_cmp_u32<<<1024, 256>>>(orig, dec, nb / 4, bad);
            unsigned long long hb = 0;
            CU(cudaMemcpy(&hb, bad, 8, cudaMemcpyDeviceToHost));
            CU(cudaFree(orig));
            CU(cudaFree(dec));
            CU(cudaFree(bad));
            ICHECK(hb == 0, "G15 qkv round trip: %llu words differ (%d x %d)", hb, rows, H);
            verified12++;
        }
        return g;
    }
    // [ink-vp-qs] qkv GEMV (K = H) of nr rows on planes of either fragment order
    void qkv_gemv(const A_g12tc& gt, int Npairs_blocks, cudaStream_t s_) {
        static const int ntg = getenv("INK_NTG") ? atoi(getenv("INK_NTG")) : 4;   // [ink-tcr-dec1] (1: per-tile decode)
        if (gt.ga.i8) {   // [ink-n4-int8]
            static const bool q8tc = !getenv("INK_Q8TC") || atoi(getenv("INK_Q8TC"));   // [ink-f52-q8occ] INK_Q8TC=0: k_g12tc
            if (gt.nr <= 8 && q8tc) {
                k_q8tc<H><<<Npairs_blocks, 256, 0, s_>>>(gt);
                return;
            }
            if (gt.nr > 8 && ntg == 4) k_g12tc<H, 0, false, 1, 4, true><<<Npairs_blocks, 256, 0, s_>>>(gt);
            else if (gt.nr > 8 && ntg == 2) k_g12tc<H, 0, false, 1, 2, true><<<Npairs_blocks, 256, 0, s_>>>(gt);
            else k_g12tc<H, 0, false, 1, 1, true><<<Npairs_blocks, 256, 0, s_>>>(gt);
            return;
        }
        if (gt.ga.ks == 3) {
            if (gt.nr > 8 && ntg == 4) k_g12tcV<H, 3, 4><<<Npairs_blocks, 256, 0, s_>>>(gt);
            else if (gt.nr > 8 && ntg == 2) k_g12tcV<H, 3, 2><<<Npairs_blocks, 256, 0, s_>>>(gt);
            else if (gt.nr > 8) k_g12tcV<H, 3, 1><<<Npairs_blocks, 256, 0, s_>>>(gt);
            else k_g12tc<H, 0, false, 3><<<Npairs_blocks, 768, 0, s_>>>(gt);
            return;
        }
        if (gt.nr > 8 && ntg == 4) k_g12tc<H, 0, false, tcr_S<H>, 4><<<Npairs_blocks, 256 * tcr_S<H>, 0, s_>>>(gt);
        else if (gt.nr > 8 && ntg == 2) k_g12tc<H, 0, false, tcr_S<H>, 2><<<Npairs_blocks, 256 * tcr_S<H>, 0, s_>>>(gt);
        else k_g12tc<H, 0><<<Npairs_blocks, 256 * tcr_S<H>, 0, s_>>>(gt);
    }
    // [ink-1c8] one card: wo GEMV (K = QD) with the conv epilogue, G15 or INT8 planes: nr 0 = the plain step (MODE 1: conv1 +
    // state shift), nr >= 1 = a row step (MODE 2: conv chain from the slot states, raw rows to hist)
    void wo_gemv1(const G12Args& ga, int nr, float* hist, const int* rt, cudaStream_t s_) {
        static const int ntg2 = getenv("INK_NTGW") ? atoi(getenv("INK_NTGW")) : 4;   // [ink-tcr-dec1] review4 M1: NTG 4 spills (512 threads; see k_g12tcR)
        constexpr int S = tcr_S<QD>;
        static const bool q8w = !getenv("INK_Q8W") || atoi(getenv("INK_Q8W"));   // [ink-1c8] INK_Q8W=0: the k_g12tc INT8 forms
        if (ga.i8 && q8w) {
            if (nr == 0) k_q8w<1><<<H / 16, 256, 0, s_>>>(A_g12tc{ga, 1, QD, H, nullptr});
            else if (nr > 8 && ntg2 == 4) k_q8w<2, 4><<<H / 16, 256, 0, s_>>>(A_g12tc{ga, nr, QD, H, hist, rt});
            else if (nr > 8 && ntg2 == 2) k_q8w<2, 2><<<H / 16, 256, 0, s_>>>(A_g12tc{ga, nr, QD, H, hist, rt});
            else k_q8w<2><<<H / 16, 256, 0, s_>>>(A_g12tc{ga, nr, QD, H, hist, rt});
            return;
        }
        if (nr == 0) {
            if (ga.i8) k_g12tc<QD, 1, false, S, 1, true><<<H / 16, 256 * S, 0, s_>>>(A_g12tc{ga, 1, QD, H, nullptr});
            else k_g12tc<QD, 1><<<H / 16, 256 * S, 0, s_>>>(A_g12tc{ga, 1, QD, H, nullptr});
            return;
        }
        auto kw = nr > 8 && ntg2 == 4 ? k_g12tc<QD, 2, false, S, 4> : nr > 8 && ntg2 == 2 ? k_g12tc<QD, 2, false, S, 2> : k_g12tc<QD, 2>;
        auto kw8 = nr > 8 && ntg2 == 4 ? k_g12tc<QD, 2, false, S, 4, true> : nr > 8 && ntg2 == 2 ? k_g12tc<QD, 2, false, S, 2, true> : k_g12tc<QD, 2, false, S, 1, true>;
        (ga.i8 ? kw8 : kw)<<<H / 16, 256 * S, 0, s_>>>(A_g12tc{ga, nr, QD, H, hist, rt});
    }
    // [ink-tcr] load-time self-test (review M4): the tensor-core GEMVs of layer 0 (qkv, full-layer qkv, wo) for 8 random
    // token rows must give, for every row count 1..8, bitwise the one-row result of each token
    // [ink-n4-int8] load-time check: the INT8 decode GEMVs == the G15 GEMVs of the dequantised planes, bit by bit (qkv of a
    // local + a full layer at 1..8 / 16 / 32 rows; wo at 1 / 3 / 8 / 16 rows, plain write form)
    static constexpr int D8ST[2] = {0, 5};   // [ink-pf8] review L1: d8_selftest's layers (their G15 copies are built for it)
    // review L2: INK_QSQ=3 (qkv planes in 3 K parts) skips the qkv part of d8_selftest, so no qkv G15 copy is built for it
    bool qsq3_on() const { return getenv("INK_QSQ") && atoi(getenv("INK_QSQ")) == 3 && tcr_SQ == 3 && tcr_S<H> == 1; }
    void d8_selftest() {
        bf16* x;
        float *ya, *yb;
        CU(cudaMalloc(&x, (size_t)3 * MROW * QD * 2));
        int ldmax = std::max(QD * 3, 3 * 4096);
        for (int l : D8ST) ldmax = std::max(ldmax, L[l].NP);
        const size_t ybytes = (size_t)MROW * ldmax * 4;
        CU(cudaMalloc(&ya, ybytes));
        CU(cudaMalloc(&yb, ybytes));
        std::vector<uint16_t> hx((size_t)3 * MROW * QD);
        uint32_t r = 777u;
        for (auto& v : hx) {
            r = r * 1664525u + 1013904223u;
            v = (uint16_t)(0x3c00u + ((r >> 9) & 0x3ffu) - 0x200u) | ((r >> 31) << 15);
        }
        CU(cudaMemcpy(x, hx.data(), hx.size() * 2, cudaMemcpyHostToDevice));
        size_t bad = 0, tot = 0;
        auto cmp = [&](size_t n) {
            std::vector<float> a(n), b(n);
            CU(cudaMemcpy(a.data(), ya, n * 4, cudaMemcpyDeviceToHost));
            CU(cudaMemcpy(b.data(), yb, n * 4, cudaMemcpyDeviceToHost));
            for (size_t i = 0; i < n; i++) bad += memcmp(&a[i], &b[i], 4) != 0;
            tot += n;
        };
        for (int l : D8ST) {
            const Layer& y = L[l];
            ICHECK(!y.NPr || !y.qp8.sm || qsq3_on() || y.qp12.sm, "d8_selftest: layer %d has no G15 qkv copy", l);   // review L1
            ICHECK(!y.wo8.sm || (tps ? y.woR : y.wo12).sm, "d8_selftest: layer %d has no G15 wo copy", l);
            if (y.NPr && y.qp8.sm && y.qp12.sm && y.qp12.ks != 3)
                for (int nr : {1, 2, 3, 4, 5, 6, 7, 8, 9, 16, 24, 32}) {   // [ink-1c8] review L6: 9 / 24 = part-filled NTG groups
                    for (int v = 0; v < 2; v++) {
                        G12Args g = v ? y.qp8 : y.qp12;
                        g.a.N = y.NPr;
                        g.a.r1 = y.qr1;
                        g.a.r2 = y.qr2;
                        g.a.x0 = x;
                        g.a.x1 = x + (size_t)MROW * H;
                        g.a.x2 = x + (size_t)2 * MROW * H;
                        g.a.y = v ? yb : ya;
                        g.a.cst = nullptr;
                        qkv_gemv(A_g12tc{g, nr, H, y.NPr, nullptr}, (y.NPr / 2 + 7) / 8, 0);
                    }
                    CU(cudaDeviceSynchronize());
                    cmp((size_t)nr * y.NPr);
                }
            // [f52_review L1] the decode form: the rank's rows scattered through the row map into the full layout of P
            if (y.NPr && y.qp8.sm && y.qp12.sm && y.qp12.ks != 3 && y.qmap)
                for (int nr : {1, 2, 3, 5, 8, 16}) {
                    CU(cudaMemset(ya, 0, ybytes));
                    CU(cudaMemset(yb, 0, ybytes));
                    for (int v = 0; v < 2; v++) {
                        G12Args g = v ? y.qp8 : y.qp12;
                        g.a.N = y.NPr;
                        g.a.r1 = y.qr1;
                        g.a.r2 = y.qr2;
                        g.a.x0 = x;
                        g.a.x1 = x + (size_t)MROW * H;
                        g.a.x2 = x + (size_t)2 * MROW * H;
                        g.a.y = v ? yb : ya;
                        g.a.cst = nullptr;
                        A_g12tc gt{g, nr, H, y.NP, nullptr};
                        gt.rmap = y.qmap;
                        qkv_gemv(gt, (y.NPr / 2 + 7) / 8, 0);
                    }
                    CU(cudaDeviceSynchronize());
                    cmp((size_t)nr * y.NP);
                }
            // [ink-1c8] one card: the conv-epilogue forms of the full wo (plain MODE 1; verify MODE 2 at 1..32 rows: outputs
            // and the raw history rows)
            if (y.wo8.sm && !tps) {
                // [ink-1c8] review L6: two slots' conv states (6 H) for the two-segment row table of pass 2
                std::vector<float> hs(6 * H + 4 * H + (size_t)MROW * H);
                for (auto& v : hs) {
                    r = r * 1664525u + 1013904223u;
                    v = ((r >> 8) * (1.f / 16777216.f) - 0.5f) * 0.25f;
                }
                float *cs[2], *cw, *hi[2], *ad;
                CU(cudaMalloc(&ad, (size_t)MROW * H * 4));
                CU(cudaMemcpy(ad, hs.data() + 10 * H, (size_t)MROW * H * 4, cudaMemcpyHostToDevice));
                int* rt0;
                for (int v = 0; v < 2; v++) {
                    CU(cudaMalloc(&cs[v], 6 * H * 4));
                    CU(cudaMalloc(&hi[v], (size_t)MROW * H * 4));
                }
                CU(cudaMalloc(&cw, 4 * H * 4));
                CU(cudaMalloc(&rt0, RT_N * 4));
                CU(cudaMemset(rt0, 0, RT_N * 4));
                CU(cudaMemcpy(cw, hs.data() + 6 * H, 4 * H * 4, cudaMemcpyHostToDevice));
                std::vector<int> hrt(RT_N, 0);
                for (int pass = 0; pass < 3; pass++)   // without / with the residual input; 2: + rows of two slots
                for (int nr : {0, 1, 2, 3, 5, 8, 9, 16, 24, 32}) {   // 0: the plain form
                    if (pass == 2 && nr < 2) continue;
                    if (pass == 2) {   // rows [0, nr / 2) slot 0, [nr / 2, nr) slot 1 (segment starts 0 / nr / 2)
                        for (int j = 0; j < nr; j++) {
                            hrt[RT_SLOT + j] = j < nr / 2 ? 0 : 1;
                            hrt[RT_SEG + j] = j < nr / 2 ? 0 : nr / 2;
                        }
                        CU(cudaMemcpy(rt0, hrt.data(), RT_N * 4, cudaMemcpyHostToDevice));
                    }
                    for (int v = 0; v < 2; v++) {
                        CU(cudaMemcpy(cs[v], hs.data(), 6 * H * 4, cudaMemcpyHostToDevice));
                        G12Args g = v ? y.wo8 : y.wo12;
                        g.a.x0 = g.a.x1 = g.a.x2 = x;
                        g.a.N = H;
                        g.a.r1 = g.a.r2 = H;
                        g.a.y = v ? yb : ya;
                        g.a.cst = cs[v];
                        g.a.cw = cw;
                        g.a.add = pass ? ad : nullptr;
                        wo_gemv1(g, nr, hi[v], rt0, 0);
                    }
                    CU(cudaDeviceSynchronize());
                    const size_t n = (size_t)std::max(nr, 1) * H;
                    cmp(n);
                    if (nr) {
                        std::vector<float> a(n), b(n);
                        CU(cudaMemcpy(a.data(), hi[0], n * 4, cudaMemcpyDeviceToHost));
                        CU(cudaMemcpy(b.data(), hi[1], n * 4, cudaMemcpyDeviceToHost));
                        for (size_t i = 0; i < n; i++) bad += memcmp(&a[i], &b[i], 4) != 0;
                        tot += n;
                    } else {   // the shifted conv state
                        std::vector<float> a(3 * H), b(3 * H);
                        CU(cudaMemcpy(a.data(), cs[0], 3 * H * 4, cudaMemcpyDeviceToHost));
                        CU(cudaMemcpy(b.data(), cs[1], 3 * H * 4, cudaMemcpyDeviceToHost));
                        for (size_t i = 0; i < a.size(); i++) bad += memcmp(&a[i], &b[i], 4) != 0;
                        tot += a.size();
                    }
                }
                for (int v = 0; v < 2; v++) {
                    CU(cudaFree(cs[v]));
                    CU(cudaFree(hi[v]));
                }
                CU(cudaFree(cw));
                CU(cudaFree(ad));
                CU(cudaFree(rt0));
            }
            if (y.wo8.sm && tps)
                for (int nr : {1, 3, 8, 16}) {
                    for (int v = 0; v < 2; v++) {
                        G12Args g = v ? y.wo8 : y.woR;
                        g.a.x0 = g.a.x1 = g.a.x2 = x;
                        g.a.N = H;
                        g.a.y = v ? yb : ya;
                        const A_g12tcR a{g, nr, QD, H, sh.cb, sh.ce - sh.cb, sh.q0 * HD, sh.q1 * HD};
                        if (nr > 8) (v ? k_g12tcR<4, true> : k_g12tcR<4>)<<<H / 16, 256>>>(a);
                        else (v ? k_g12tcR<1, true> : k_g12tcR<1>)<<<H / 16, 256>>>(a);
                    }
                    CU(cudaDeviceSynchronize());
                    cmp((size_t)nr * H);
                }
        }
        CU(cudaFree(x));
        CU(cudaFree(ya));
        CU(cudaFree(yb));
        fprintf(stderr, "[ink] INK_DENSE8 rank %d: INT8 g128 dense planes %.3f GB; INT8 GEMVs vs G15 of the dequantised planes (own rows + row-map decode form): %zu of %zu values differ\n",
                sh.rank, b8 / 1e9, bad, tot);
        ICHECK(bad == 0, "INK_DENSE8 self-test: %zu values differ", bad);
    }
    void tcr_selftest() {
        const int full = 5;
        bf16* x;
        float *y8, *y1;
        const int NPm = std::max(L[0].NP, L[full].NP);
        CU(cudaMalloc(&x, (size_t)3 * MROW * QD * 2));
        CU(cudaMalloc(&y8, (size_t)MROW * NPm * 4));
        CU(cudaMalloc(&y1, (size_t)NPm * 4));
        std::vector<uint16_t> hx((size_t)3 * MROW * QD);
        uint32_t r = 12345u;
        for (auto& v : hx) {
            r = r * 1664525u + 1013904223u;
            v = (uint16_t)(0x3c00u + ((r >> 9) & 0x3ffu) - 0x200u) | ((r >> 31) << 15);   // |x| ~ 2^-7 .. 2^-6
        }
        CU(cudaMemcpy(x, hx.data(), hx.size() * 2, cudaMemcpyHostToDevice));
        size_t bad = 0, tot = 0;
        std::vector<float> a8((size_t)MROW * NPm), a1(NPm);
        auto test = [&](const G12Args& m0, int K, int N, int nr) {
            G12Args g = m0;
            g.a.N = N;
            g.a.y = y8;
            g.a.cst = nullptr;
            const size_t seg = (size_t)MROW * K;
            g.a.x0 = x;
            g.a.x1 = g.a.r1 < N ? x + seg : x;
            g.a.x2 = g.a.r2 < N ? x + 2 * seg : x;
            if (K == H) {   // [ink-tcr-dec1] rows > 8: the 4-tile decode form (bitwise == one row each, checked below)
                qkv_gemv(A_g12tc{g, nr, K, N, nullptr}, (N / 2 + 7) / 8, 0);
            } else {
                auto kf = nr > 8 ? k_g12tc<QD, 0, false, tcr_S<QD>, 4> : k_g12tc<QD, 0>;
                kf<<<(N / 2 + 7) / 8, (256 * tcr_S<QD>)>>>(A_g12tc{g, nr, K, N, nullptr});
            }
            CU(cudaMemcpy(a8.data(), y8, (size_t)nr * N * 4, cudaMemcpyDeviceToHost));
            for (int t = 0; t < nr; t++) {
                G12Args g1 = g;
                g1.a.y = y1;
                g1.a.x0 += (size_t)t * K;
                g1.a.x1 += (size_t)t * K;
                g1.a.x2 += (size_t)t * K;
                if (K == H) qkv_gemv(A_g12tc{g1, 1, K, N, nullptr}, (N / 2 + 7) / 8, 0);
                else k_g12tc<QD, 0><<<(N / 2 + 7) / 8, (256 * tcr_S<QD>)>>>(A_g12tc{g1, 1, K, N, nullptr});
                CU(cudaMemcpy(a1.data(), y1, (size_t)N * 4, cudaMemcpyDeviceToHost));
                for (int i = 0; i < N; i++) bad += memcmp(&a1[i], &a8[(size_t)t * N + i], 4) != 0;
                tot += N;
            }
        };
        for (int nr = 1; nr <= MROW; nr++) {   // [ink-b-rows] row steps up to MROW rows (column tiles of 8)
            test(L[0].qp12, H, L[0].NP, nr);
            G12Args gf = L[full].qp12;
            gf.a.r1 = L[full].offK;
            gf.a.r2 = L[full].offV;
            test(gf, H, L[full].NP, nr);
            G12Args gw = L[0].wo12;
            gw.a.r1 = gw.a.r2 = H;
            test(gw, QD, H, nr);
        }
        // review r3 M2: wo epilogues, plain (conv1, state shift) vs verify (conv chain from the state) on one row
        size_t bad2 = 0, bad3 = 0;
        {
            std::vector<float> hs(3 * H + 4 * H + H);
            for (auto& v : hs) {
                r = r * 1664525u + 1013904223u;
                v = ((r >> 8) * (1.f / 16777216.f) - 0.5f) * 0.25f;
            }
            float *st1, *st2, *cw, *add, *ya, *yb, *hist;
            CU(cudaMalloc(&st1, 3 * H * 4));
            CU(cudaMalloc(&st2, 3 * H * 4));
            CU(cudaMalloc(&cw, 4 * H * 4));
            CU(cudaMalloc(&add, H * 4));
            CU(cudaMalloc(&ya, H * 4));
            CU(cudaMalloc(&yb, H * 4));
            CU(cudaMalloc(&hist, H * 4));
            CU(cudaMemcpy(st1, hs.data(), 3 * H * 4, cudaMemcpyHostToDevice));
            CU(cudaMemcpy(st2, hs.data(), 3 * H * 4, cudaMemcpyHostToDevice));
            CU(cudaMemcpy(cw, hs.data() + 3 * H, 4 * H * 4, cudaMemcpyHostToDevice));
            CU(cudaMemcpy(add, hs.data() + 7 * H, H * 4, cudaMemcpyHostToDevice));
            G12Args g = L[0].wo12;
            g.a.x0 = g.a.x1 = g.a.x2 = x;
            g.a.N = H;
            g.a.r1 = g.a.r2 = H;
            g.a.cw = cw;
            g.a.add = add;
            g.a.cst = st1;
            g.a.y = ya;
            k_g12tc<QD, 1><<<H / 16, (256 * tcr_S<QD>)>>>(A_g12tc{g, 1, QD, H, nullptr});
            g.a.cst = st2;
            g.a.y = yb;
            static int* rt0 = nullptr;   // row table of one row, slot 0
            if (!rt0) {
                CU(cudaMalloc(&rt0, RT_N * 4));
                CU(cudaMemset(rt0, 0, RT_N * 4));
            }
            k_g12tc<QD, 2><<<H / 16, (256 * tcr_S<QD>)>>>(A_g12tc{g, 1, QD, H, hist, rt0});
            std::vector<float> a(H), b(H);
            CU(cudaMemcpy(a.data(), ya, H * 4, cudaMemcpyDeviceToHost));
            CU(cudaMemcpy(b.data(), yb, H * 4, cudaMemcpyDeviceToHost));
            for (int i = 0; i < H; i++) bad2 += memcmp(&a[i], &b[i], 4) != 0;
            // review r3 M1 / [ink-b-dnsh]: NVFP4 down, plain form (k_dn4p) vs the row-step form (k_order + k_dn4o +
            // k_dsum) with the token as rows 0 and 4 of a 6-row step whose other rows share its experts (jobs of 1..4
            // users, an expert split over two jobs)
            const Layer& y = L[0];
            if (y.fmt == 0) {
                constexpr int TR = 6;
                float *act, *part, *yo;
                int* ids;
                OrdBuf* ob;
                CU(cudaMalloc(&act, (size_t)TR * TOPK * I * 4));
                CU(cudaMalloc(&part, (size_t)TR * TOPK * H * 4));
                CU(cudaMalloc(&yo, (size_t)TR * H * 4));
                CU(cudaMalloc(&ids, TR * TOPK * 4));
                CU(cudaMalloc(&ob, sizeof(OrdBuf)));
                std::vector<float> ha((size_t)TR * TOPK * I);
                for (auto& v : ha) {
                    r = r * 1664525u + 1013904223u;
                    v = ((r >> 8) * (1.f / 16777216.f) - 0.5f) * 4.f;
                }
                std::copy(ha.begin(), ha.begin() + TOPK * I, ha.begin() + 4 * TOPK * I);   // row 4 = row 0
                const int hid[TR * TOPK] = {3, 17, 29, 41, 64, 77, 101, 126,  17, 3, 5, 41, 90, 77, 8, 9,
                                            3, 18, 29, 40, 64, 70, 101, 127,  3, 11, 12, 13, 14, 15, 16, 17,
                                            3, 17, 29, 41, 64, 77, 101, 126,  126, 101, 3, 2, 1, 0, 41, 64};
                CU(cudaMemcpy(act, ha.data(), ha.size() * 4, cudaMemcpyHostToDevice));
                CU(cudaMemcpy(ids, hid, sizeof hid, cudaMemcpyHostToDevice));
                const A_dn4 d1{y.c2, y.s2, act, ids, ya, nullptr, nullptr, nullptr, 0};
                const A_dn4 d2{y.c2, y.s2, act, ids, yo, nullptr, nullptr, nullptr, 1};
                k_dn4p<2, 2><<<H / 32, 512>>>(d1);
                k_order<<<1, 256>>>(ids, TR * TOPK, 4, ob);
                k_dn4o<4, 2><<<dim3(TR * TOPK, H / 128), 256>>>(d2, ob, part);
                k_dsum<<<dim3(H / 256, TR), 256>>>(part, yo);
                CU(cudaMemcpy(a.data(), ya, H * 4, cudaMemcpyDeviceToHost));
                for (int row : {0, 4}) {
                    CU(cudaMemcpy(b.data(), yo + (size_t)row * H, H * 4, cudaMemcpyDeviceToHost));
                    for (int i = 0; i < H; i++) bad3 += memcmp(&a[i], &b[i], 4) != 0;
                }
                for (void* f : {(void*)act, (void*)part, (void*)yo, (void*)ids, (void*)ob}) CU(cudaFree(f));
            }
            for (float* f : {st1, st2, cw, add, ya, yb, hist}) CU(cudaFree(f));
        }
        CU(cudaFree(x));
        CU(cudaFree(y8));
        CU(cudaFree(y1));
        ICHECK(bad == 0, "[ink-tcr] self-test: %zu of %zu GEMV outputs differ between row counts", bad, tot);
        ICHECK(bad2 == 0, "[ink-tcr] self-test: %zu wo outputs differ between the plain and verify conv epilogues", bad2);
        ICHECK(bad3 == 0, "self-test: %zu NVFP4 down outputs differ between the plain and verify kernels", bad3);
        fprintf(stderr, "[ink] self-test: tensor-core GEMV %zu outputs, row counts 1..%d bitwise equal; wo plain / verify epilogues and NVFP4 down plain / verify kernels bitwise equal\n", tot, MROW);
    }
    size_t b12 = 0;
    // [ink-n4-int8] INK_DENSE8=1 (lossy, opt-in): the dense attention weights (wq / wr / gate_attn / wk / wv / wo of every
    // layer) RTN INT8 with an fp16 scale per 128 inputs, the arithmetic of tools/inkling/fakequant_dense.py int8g128
    // (s = fp16(amax / 127), 0 -> 1; q = clamp(rint(w / s), -127, 127)); every path computes with d = bf16_rn(q s) (the G15
    // planes and the prefill operands hold d), the TP decode GEMVs read the INT8 planes (same bits as d's G15 planes)
    // [ink-dense8-default] default on when <model>/dense8.safetensors (or INK_DENSE8_FILE) is present ([ink-1c8] one card
    // too: its decode GEMVs and prefill read INT8 planes now); INK_DENSE8=0 = off (G15); INK_DENSE8=1 = on (without the
    // file RTN at load)
    bool dense8 = false;
    bool dense8_st = false;   // [ink-pf8] stored codes (dense8.safetensors), else RTN at load
    size_t b8 = 0;
    struct Q8 {
        std::vector<int8_t> q;
        std::vector<uint16_t> s;    // [rows][K / 128] fp16 bits
        std::vector<uint8_t> isq;   // row quantised (else raw BF16: sink / sconv rows)
        int K = 0;
    };
    static uint16_t f2bf_rn(float f) {
        uint32_t u;
        memcpy(&u, &f, 4);
        u += 0x7fffu + ((u >> 16) & 1u);
        return (uint16_t)(u >> 16);
    }
    static float bf2f_(uint16_t b) {
        const uint32_t u = (uint32_t)b << 16;
        float f;
        memcpy(&f, &u, 4);
        return f;
    }
    // rows [r0, r1) of the N x K matrix A: quantise, and A's rows := d
    static void rtn8_rows(uint16_t* A, int K, int r0, int r1, Q8& o) {
        for (int r = r0; r < r1; r++) {
            uint16_t* w = A + (size_t)r * K;
            o.isq[r] = 1;
            for (int g0 = 0; g0 < K; g0 += 128) {
                float am = 0.f;
                for (int j = 0; j < 128; j++) am = std::max(am, fabsf(bf2f_(w[g0 + j])));
                __half hs = __float2half_rn(am / 127.0f);
                float sf = __half2float(hs);
                if (sf == 0.f) {
                    hs = __float2half_rn(1.f);
                    sf = 1.f;
                }
                o.s[(size_t)r * (K / 128) + g0 / 128] = __half_as_ushort(hs);
                for (int j = 0; j < 128; j++) {
                    const float v = std::min(127.f, std::max(-127.f, rintf(bf2f_(w[g0 + j]) / sf)));
                    o.q[(size_t)r * K + g0 + j] = (int8_t)v;
                    w[g0 + j] = f2bf_rn((float)(int)v * sf);
                }
            }
        }
    }
    // [ink-int8-margin] rows [r0, r1) of A from stored codes (int8 [r1 - r0, K]) and scales (fp16 [r1 - r0, K / 128]) of
    // dense8.safetensors (tools/inkling/gptq_dense.py: error-feedback rounding, same grid); A's rows := d = bf16_rn(q s).
    // Returns how many of A's values differed from d numerically (0 when A is the matching dequantised checkpoint).
    static size_t st8_rows(uint16_t* A, int K, int r0, int r1, Q8& o, const Tens& tq, const Tens& ts) {
        ICHECK(tq.dtype == "I8" && ts.dtype == "F16" && tq.shape.size() == 2 && ts.shape.size() == 2 &&
                   tq.shape[0] == r1 - r0 && tq.shape[1] == K && ts.shape[0] == r1 - r0 && ts.shape[1] == K / 128,
               "dense8.safetensors: unexpected code / scale tensors");
        const int8_t* q = (const int8_t*)tq.p;
        const uint16_t* sc = (const uint16_t*)ts.p;
        size_t diff = 0;
        for (int r = r0; r < r1; r++) {
            uint16_t* w = A + (size_t)r * K;
            const size_t rr = (size_t)(r - r0);
            o.isq[r] = 1;
            for (int g = 0; g < K / 128; g++) {
                const uint16_t hb = sc[rr * (K / 128) + g];
                const float sf = __half2float(__ushort_as_half(hb));
                ICHECK(sf > 0.f && std::isfinite(sf), "dense8.safetensors: bad scale");
                o.s[(size_t)r * (K / 128) + g] = hb;
                for (int j = 0; j < 128; j++) {
                    const int v = q[rr * K + g * 128 + j];
                    ICHECK(v >= -127 && v <= 127, "dense8.safetensors: code -128");
                    o.q[(size_t)r * K + g * 128 + j] = (int8_t)v;
                    const uint16_t d = f2bf_rn((float)v * sf);
                    diff += bf2f_(w[g * 128 + j]) != bf2f_(d);   // numeric (the tool writes -0 for negative zero codes)
                    w[g * 128 + j] = d;
                }
            }
        }
        return diff;
    }
    static void q8_init(Q8& o, int N, int K) {
        o.K = K;
        o.q.assign((size_t)N * K, 0);
        o.s.assign((size_t)N * (K / 128), 0);
        o.isq.assign(N, 0);
    }
    // planes of N rows (even) x K: codes q (+128), scales s, D = the BF16 rows (d; raw rows of the fallback groups)
    G12Args up8(const int8_t* q, const uint16_t* sc, const uint8_t* isq, const uint16_t* D, int N, int K) {
        ICHECK(N % 2 == 0 && K % 256 == 0, "up8 %d x %d", N, K);
        const int G = N / 2, NT = (G + 7) / 8, NC = K / 256, NG = K / 128;
        std::vector<uint32_t> P((size_t)NT * NC * 8 * 32 * 4, 0x80808080u), SC((size_t)NT * NC * 2 * 8, 0u);
        std::vector<uint32_t> meta((size_t)G * 4, 0u);
        std::vector<uint16_t> fb;
        for (int g = 0; g < G; g++) {
            const int b = g / 8, n = g % 8;
            if (!isq[2 * g] || !isq[2 * g + 1]) {
                meta[(size_t)g * 4] = 255u << 16;
                meta[(size_t)g * 4 + 1] = (uint32_t)(fb.size() / (2 * (size_t)K));
                fb.insert(fb.end(), D + (size_t)2 * g * K, D + (size_t)(2 * g + 2) * K);
                continue;
            }
            for (int c = 0; c < NC; c++)
                for (int w = 0; w < 8; w++)
                    for (int qq = 0; qq < 4; qq++) {
                        const int l = 4 * w + qq, L = 4 * n + qq, e0 = c * 256 + l * 8;
                        uint32_t* u = P.data() + ((((size_t)b * NC + c) * 8 + w) * 32 + L) * 4;
                        for (int r = 0; r < 2; r++)
                            for (int h = 0; h < 2; h++) {
                                uint32_t v = 0;
                                for (int k = 0; k < 4; k++) v |= (uint32_t)(uint8_t)(q[(size_t)(2 * g + r) * K + e0 + 4 * h + k] + 128) << (8 * k);
                                u[2 * r + h] = v;
                            }
                        if (qq == 0 && (w & 3) == 0)
                            SC[(((size_t)b * NC + c) * 2 + (w >> 2)) * 8 + n] =
                                (uint32_t)sc[(size_t)2 * g * NG + 2 * c + (w >> 2)] | ((uint32_t)sc[(size_t)(2 * g + 1) * NG + 2 * c + (w >> 2)] << 16);
                    }
        }
        auto up = [&](const void* p, size_t n) {
            void* d = dmalloc(std::max<size_t>(n, 16), &dev_bytes);
            if (n) CU(cudaMemcpy(d, p, n, cudaMemcpyHostToDevice));
            return d;
        };
        G12Args g{};
        g.sm = (const uint4*)up(P.data(), P.size() * 4);
        g.idx = (const uint32_t*)up(SC.data(), SC.size() * 4);
        g.meta = (const uint4*)up(meta.data(), meta.size() * 4);
        g.fb = (const bf16*)up(fb.data(), fb.size() * 2);
        g.a.N = N;
        g.i8 = 1;
        b8 += P.size() * 4 + SC.size() * 4 + meta.size() * 4 + fb.size() * 2;
        if (!getenv("INK_NO_VERIFY12")) {   // [ink-pf8] prefill decodes these planes (k_dec8): == D bitwise, every matrix
            const size_t nb = (size_t)N * K * 2;
            uint32_t *orig = nullptr, *dec = nullptr;
            unsigned long long* bad = nullptr;
            CU(cudaMalloc(&orig, nb));
            CU(cudaMalloc(&dec, nb));
            CU(cudaMalloc(&bad, 8));
            CU(cudaMemcpy(orig, D, nb, cudaMemcpyHostToDevice));
            CU(cudaMemset(bad, 0, 8));
            k_dec8<<<NT, 256>>>(g, dec, K);
            k_cmp_u32<<<1024, 256>>>(orig, dec, nb / 4, bad);
            unsigned long long hb = 0;
            CU(cudaMemcpy(&hb, bad, 8, cudaMemcpyDeviceToHost));
            CU(cudaFree(orig));
            CU(cudaFree(dec));
            CU(cudaFree(bad));
            ICHECK(hb == 0, "INT8 planes round trip: %llu words differ (%d x %d)", hb, N, K);
            verified8++;
        }
        return g;
    }
    size_t a12 = 0;   // [ink-pf8] bytes allocated for G15 planes (up12 / up12q)
    int verified12 = 0, verified8 = 0;
    bool keep16 = false;
    // [ink-pf8] dense8 on TP: prefill reads the INT8 planes (k_dec8 -> BF16 for cuBLAS), no G15 copy of qkv / wo except
    // layers 0 and 5 until d8_selftest (freed after it). INK_D8KEEP12=1: keep the G15 copy and prefill from it (A/B)
    // [ink-1c8] the same on one card (one-card INT8 planes: INK_1C8)
    bool keep12 = env_on("INK_D8KEEP12");
    // [ink-pf8-wob] dense8 on TP (default, user 11-25): no BF16 wo copy (woB); prefill decodes wo8 into WS (k_dec8, same
    // bits: up8 round trip): TP2 context 553,952 -> 565,216 for prefill 2K / 8K / 120K -0.3 / -0.6 / -0.8 %. INK_PF8WOB=0: keep woB
    bool wob8 = !keep12 && (!getenv("INK_PF8WOB") || !*getenv("INK_PF8WOB") || atoi(getenv("INK_PF8WOB")));   // (empty = unset)
    size_t b12st = 0, d12st = 0;   // the self-test layers' G15 planes: encoded / allocated bytes (freed after d8_selftest)
    static void free12(G12Args& g) {
        for (const void* p : {(const void*)g.sm, (const void*)g.idx, (const void*)g.meta, (const void*)g.fb})
            if (p) CU(cudaFree((void*)p));
        g = G12Args{};
    }
    // [ink-1c8] the matrix prefill decodes to BF16 (dec16): the INT8 planes when the layer has them
    const G12Args& pf_src(const G12Args& g8, const G12Args& g12) const { return g8.sm && !keep12 ? g8 : g12; }
    template <class T>
    T* upraw2(const void* p, size_t n) {
        void* d = dmalloc(n, &dev_bytes);
        CU(cudaMemcpy(d, p, n, cudaMemcpyHostToDevice));
        return (T*)d;
    }
    template <class T>
    T* upraw(const Tens& t, size_t n = 0) {
        if (!n) n = t.n;
        return (T*)rep_up(t.p, n, 0, [&] {   // [ink-tp-dec1] shared by the ranks of a device
            void* d = dmalloc(n, &dev_bytes);
            CU(cudaMemcpy(d, t.p, n, cudaMemcpyHostToDevice));
            return d;
        });
    }
    // ---- [ink-tp-dec1] the rank's part of a layer
    // fused projection rows: q side of the rank's q heads (q, r, gate, sink), K rows (+ dynamic taps) and V rows of its
    // K / V heads, gathered in this order; qmap = their rows in the full layout (decode keeps P in the full layout; TP prefill sub-chunks may use the compact gather order, [ink-pf3-qkv]).
    // An odd count repeats the last row (same value written twice). wo: columns [256 cb, 256 ce), one chunk per part.
    void tp_load_attn(Layer& y, const std::vector<uint16_t>& wp, const uint16_t* wo, const Q8* qp = nullptr, const Q8* qo = nullptr, bool g12q = true, bool g12o = true) {
        std::vector<int> rows;
        for (int h = sh.q0; h < sh.q1; h++)
            for (int i = 0; i < HD; i++) rows.push_back(OFF_Q + h * HD + i);
        for (int h = sh.q0; h < sh.q1; h++)
            for (int i = 0; i < DREL; i++) rows.push_back(OFF_R + h * DREL + i);
        for (int h = sh.q0; h < sh.q1; h++)
            for (int i = 0; i < HD; i++) rows.push_back(OFF_G + h * HD + i);
        for (int h = sh.q0; h < sh.q1; h++) rows.push_back(OFF_S + h);
        y.qr1 = (int)rows.size();
        for (int k = 0; k < 2; k++) {
            const int off = k ? y.offV : y.offK;
            for (int h = sh.kv0; h < sh.kv1; h++)
                for (int i = 0; i < HD; i++) rows.push_back(off + h * HD + i);
            if (y.full)
                for (int h = sh.kv0; h < sh.kv1; h++)
                    for (int j = 0; j < 4; j++) rows.push_back(off + KVD + h * 4 + j);
            if (k == 0) y.qr2 = (int)rows.size();
        }
        y.qruns.clear();
        for (int r = 0; r < (int)rows.size(); r++) {   // [ink-tp-pf] (before the odd-count pad row)
            const int seg = r < y.qr1 ? 0 : r < y.qr2 ? 1 : 2;
            auto& q = y.qruns;
            if (!q.empty() && q.back()[3] == seg && q.back()[1] + q.back()[2] == rows[r] && q.back()[0] + q.back()[2] == r) q.back()[2]++;
            else q.push_back({r, rows[r], 1, seg});
        }
        {   // [ink-pf3-qkv] compact layout (the gather order above)
            const int nq = sh.nq(), nkv = sh.nkv();
            y.dq = OFF_Q + sh.q0 * HD;
            y.dr = OFF_R + sh.q0 * DREL - nq * HD;
            y.dg = OFF_G + sh.q0 * HD - nq * (HD + DREL);
            y.ds = OFF_S + sh.q0 - nq * (2 * HD + DREL);
            y.cK = y.qr1;
            y.cKt = y.full ? y.qr1 + nkv * HD : -1;
            y.cV = y.qr2;
            y.cVt = y.full ? y.qr2 + nkv * HD : -1;
        }
        if (rows.size() % 2) rows.push_back(rows.back());
        y.NPr = (int)rows.size();
        // pf3_review M1: k_pattn reads the q heads of its KV heads' groups (partial groups: heads outside the rank's
        // compact columns); M2: segment starts and the row stride 16 B aligned for cuBLAS (TP2: 3,276 / 3,788 / 4,300)
        y.pcok = sh.q0 == GQ * sh.kv0 && sh.q1 == GQ * sh.kv1 && y.qr1 % 4 == 0 && y.qr2 % 4 == 0 && y.NPr % 4 == 0;
        if (y.NPr) {
            std::vector<uint16_t> g((size_t)y.NPr * H);
            for (int r = 0; r < y.NPr; r++) memcpy(g.data() + (size_t)r * H, wp.data() + (size_t)rows[r] * H, H * 2);
            if (!qp || g12q) y.qp12 = up12(g.data(), y.NPr, H, true, 0, true);   // [ink-pf8] dense8: only for the self-test
            if (qp) {   // [ink-n4-int8] the same rows
                std::vector<int8_t> q8((size_t)y.NPr * H);
                std::vector<uint16_t> s8((size_t)y.NPr * (H / 128));
                std::vector<uint8_t> iq(y.NPr);
                for (int r = 0; r < y.NPr; r++) {
                    memcpy(q8.data() + (size_t)r * H, qp->q.data() + (size_t)rows[r] * H, H);
                    memcpy(s8.data() + (size_t)r * (H / 128), qp->s.data() + (size_t)rows[r] * (H / 128), H / 128 * 2);
                    iq[r] = qp->isq[rows[r]];
                }
                y.qp8 = up8(q8.data(), s8.data(), iq.data(), g.data(), y.NPr, H);
            }
            y.qmap = (int*)dmalloc(rows.size() * 4, &dev_bytes);
            CU(cudaMemcpy(y.qmap, rows.data(), rows.size() * 4, cudaMemcpyHostToDevice));
        }
        const int nc = sh.ce - sh.cb;
        if (nc > 0) {
            std::vector<uint16_t> g((size_t)H * 256 * nc);
            for (int r = 0; r < H; r++) memcpy(g.data() + (size_t)r * 256 * nc, wo + (size_t)r * QD + 256 * sh.cb, (size_t)256 * nc * 2);
            if (!qo || g12o) y.woR = up12(g.data(), H, 256 * nc, true, nc);   // [ink-pf8] dense8: only for the self-test
            if (qo) {   // [ink-n4-int8] the rank's column chunks (128-groups inside a chunk)
                const int K = 256 * nc, NGf = QD / 128;
                std::vector<int8_t> q8((size_t)H * K);
                std::vector<uint16_t> s8((size_t)H * (K / 128));
                std::vector<uint8_t> iq(H, 1);
                for (int r = 0; r < H; r++) {
                    memcpy(q8.data() + (size_t)r * K, qo->q.data() + (size_t)r * QD + 256 * sh.cb, K);
                    memcpy(s8.data() + (size_t)r * (K / 128), qo->s.data() + (size_t)r * NGf + 2 * sh.cb, K / 128 * 2);
                    ICHECK(qo->isq[r], "wo row %d not quantised", r);
                }
                y.wo8 = up8(q8.data(), s8.data(), iq.data(), g.data(), H, K);
            }
            // [ink-tp-pf] prefill GEMM operand; [ink-pf8-wob] with the INT8 planes prefill decodes wo8 into WS instead
            if (pfx && !(qo && wob8)) y.woB = upraw2<bf16>((const bf16*)g.data(), g.size() * 2);
        }
    }
    // experts: intermediate columns [32 ua, 32 ub) of every expert (gate / up row pairs, down columns), per-expert factors
    // [ink-tp-tcx] MXFP8: the max E8M0 byte of rows [r0, r0 + nr) of each expert block (rpe rows x rb bytes), bytes
    // [c0, c0 + cn) of every row; blocks below max - 14 are inexact in the fp16 weight form (counted)
    uint8_t* tcx_rmax(const Tens& t, int rpe, size_t rb, int r0, int nr, size_t c0, size_t cn, long long& bad, long long& nb) {
        std::vector<uint8_t> h((size_t)NE * nr);
        for (int e = 0; e < NE; e++)
            for (int r = 0; r < nr; r++) {
                const uint8_t* q = t.p + ((size_t)e * rpe + r0 + r) * rb + c0;
                int m = 0;
                for (size_t j = 0; j < cn; j++) m = std::max(m, (int)q[j]);
                for (size_t j = 0; j < cn; j++) bad += (int)q[j] < m - 14;
                nb += cn;
                h[(size_t)e * nr + r] = (uint8_t)m;
            }
        return upraw2<uint8_t>(h.data(), h.size());
    }
    void tcx_rmax_layer(Layer& y, const Tens& s13, const Tens& s2, int i0, int ni) {
        long long bad = 0, nb = 0;
        y.m13 = tcx_rmax(s13, 2 * I, H / 32, 2 * i0, 2 * ni, 0, H / 32, bad, nb);
        y.m2 = tcx_rmax(s2, H, I / 32, 0, H, i0 / 32, ni / 32, bad, nb);
        tcx_bad += bad;
        tcx_nb += nb;
    }
    long long tcx_bad = 0, tcx_nb = 0;
    void tp_load_experts(Layer& y, const std::string& p, const Tens& w13) {
        const std::string pf = "model.llm.";
        const int nu = sh.nu(), ni = nu * TP_UNIT, i0 = sh.ua * TP_UNIT;
        // rows [r0, r0 + nr) of each of NE blocks of `rpe` rows x rb bytes, bytes [c0, c0 + cn) of every row
        auto cut = [&](const Tens& t, int rpe, size_t rb, int r0, int nr, size_t c0, size_t cn) -> uint8_t* {
            ICHECK(t.n == (size_t)NE * rpe * rb, "expert tensor bytes");
            std::vector<uint8_t> h((size_t)NE * nr * cn);
            for (int e = 0; e < NE; e++)
                for (int r = 0; r < nr; r++)
                    memcpy(h.data() + ((size_t)e * nr + r) * cn, t.p + ((size_t)e * rpe + r0 + r) * rb + c0, cn);
            return upraw2<uint8_t>(h.data(), h.size());
        };
        if (y.fmt == 0) {
            ICHECK(w13.shape == std::vector<long long>({NE, 2 * I, H / 2}), "w13 shape");
            const Tens& s13 = f.get(pf + p + "mlp.experts.w13_weight_scale", "F8_E4M3", {NE, 2 * I, H / 16});
            const Tens& c2 = f.get(pf + p + "mlp.experts.w2_weight", "U8", {NE, H, I / 2});
            const Tens& s2 = f.get(pf + p + "mlp.experts.w2_weight_scale", "F8_E4M3", {NE, H, I / 16});
            if (nu) {
                y.c13 = cut(w13, 2 * I, H / 2, 2 * i0, 2 * ni, 0, H / 2);
                y.s13 = cut(s13, 2 * I, H / 16, 2 * i0, 2 * ni, 0, H / 16);
                y.c2 = cut(c2, H, I / 2, 0, H, i0 / 2, ni / 2);
                y.s2 = cut(s2, H, I / 16, 0, H, i0 / 16, ni / 16);
            }
            for (int k = 0; k < 2; k++) {
                const std::string nm = k ? "w2" : "w13";
                const Tens& am = f.get(pf + p + "mlp.experts." + nm + "_weight_amax", "F32", {NE});
                const Tens& dn = f.get(pf + p + "mlp.experts." + nm + "_weight_outer_scale_denominator", "F32", {1});
                float d;
                memcpy(&d, dn.p, 4);
                std::vector<float> fac(NE);
                for (int e = 0; e < NE; e++) {
                    float a;
                    memcpy(&a, am.p + 4 * e, 4);
                    fac[e] = a / d;
                }
                (k ? y.f2 : y.f13) = upraw2<float>(fac.data(), NE * 4);
            }
        } else {
            ICHECK(w13.dtype == "F8_E4M3" && w13.shape == std::vector<long long>({NE, 2 * I, H}), "w13 fp8 shape");
            const Tens& s13 = f.get(pf + p + "mlp.experts.w13_weight_scale", "U8", {NE, 2 * I, H / 32});
            const Tens& c2 = f.get(pf + p + "mlp.experts.w2_weight", "F8_E4M3", {NE, H, I});
            const Tens& s2 = f.get(pf + p + "mlp.experts.w2_weight_scale", "U8", {NE, H, I / 32});
            if (nu) {
                y.c13 = cut(w13, 2 * I, H, 2 * i0, 2 * ni, 0, H);
                y.s13 = cut(s13, 2 * I, H / 32, 2 * i0, 2 * ni, 0, H / 32);
                y.c2 = cut(c2, H, I, 0, H, i0, ni);
                y.s2 = cut(s2, H, I / 32, 0, H, i0 / 32, ni / 32);
                if (tcx_min > 0) tcx_rmax_layer(y, s13, s2, i0, ni);
            }
            std::vector<float> one(NE, 1.f);
            y.f13 = upraw2<float>(one.data(), NE * 4);
            y.f2 = upraw2<float>(one.data(), NE * 4);
        }
    }
    void load() {
        if (tpc) f = tpc->file;   // [ink-tp-dec1] one mapping for every rank
        else f.open(o.model_dir + "/model-00001-of-00001.safetensors");
        const std::string pf = "model.llm.";
        auto G = [&](const std::string& k) -> const Tens& { return f.get(pf + k); };
        float e2[16] = {0.f, .5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f, -0.f, -.5f, -1.f, -1.5f, -2.f, -3.f, -4.f, -6.f};
        CU(cudaMemcpyToSymbol(c_e2m1, e2, sizeof e2));
        float e4[256];
        for (int i = 0; i < 256; i++) e4[i] = e4m3_to_f((uint8_t)i);
        CU(cudaMemcpyToSymbol(c_e4m3, e4, sizeof e4));
        const std::vector<int> fulls = {5, 11, 17, 23};
        // [ink-int8-margin] INK_DENSE8 with <model>/dense8.safetensors (or INK_DENSE8_FILE): stored codes + scales
        // (error-feedback rounding) instead of RTN at load; INK_DENSE8_RTN=1 forces RTN
        File f8;
        bool st8 = false;
        size_t st8_diff = 0;
        {
            const char* e8 = getenv("INK_DENSE8");
            const bool off8 = e8 && *e8 && !atoi(e8), on8 = e8 && atoi(e8), rtn8 = env_on("INK_DENSE8_RTN");
            const std::string p8 = getenv("INK_DENSE8_FILE") ? std::string(getenv("INK_DENSE8_FILE")) : o.model_dir + "/dense8.safetensors";
            struct stat sb8;
            const bool have8 = ::stat(p8.c_str(), &sb8) == 0;
            if (off8)
                dense8 = false;
            else if (rtn8 || on8)
                dense8 = true;   // explicit: stored codes when present, else RTN at load
            else
                dense8 = have8;   // default ([ink-1c8] one card too: INT8 planes 7.05 vs G15 7.97 ms/step)
            if (dense8 && !rtn8) {
                if (have8) {
                    f8.open(p8);
                    st8 = dense8_st = true;
                } else
                    ICHECK(!getenv("INK_DENSE8_FILE"), "INK_DENSE8_FILE %s not found", p8.c_str());
            }
            if (!off8 && !dense8 && sh.rank == 0)
                fprintf(stderr, "[ink] dense attention weights: %s not found, loading G15 (lossless BF16 path); INT8 dense "
                                "(the faster default) needs that file next to the checkpoint\n", p8.c_str());
        }
        for (int l = 0; l < NL; l++) {
            Layer& y = L[l];
            const std::string p = "layers." + std::to_string(l) + ".";
            y.full = std::find(fulls.begin(), fulls.end(), l) != fulls.end();
            y.loop = l >= LOOP0 && l < LOOP1;
            const int kd = y.full ? KVD + NKV * 4 : KVD;
            y.offK = OFF_END;
            y.offV = y.offK + kd;
            y.NP = y.offV + kd;
            // fused projection
            std::vector<std::pair<std::string, int>> parts = {{"attn.wq_du.weight", QD}, {"attn.wr_du.weight", NQ * DREL},
                                                              {"attn.gate_attn.weight", QD}, {"attn.attn_sink_proj.weight", NQ},
                                                              {"attn.wk_dv.weight", KVD}};
            if (y.full) parts.push_back({"attn.k_sconv.weight_proj", NKV * 4});
            parts.push_back({"attn.wv_dv.weight", KVD});
            if (y.full) parts.push_back({"attn.v_sconv.weight_proj", NKV * 4});
            std::vector<uint16_t> wp((size_t)y.NP * H, 0);
            size_t row = 0;
            for (auto& pr : parts) {
                const Tens& t = G(p + pr.first);
                ICHECK(t.dtype == "BF16" && t.shape.size() == 2 && t.shape[0] == pr.second && t.shape[1] == H, "%s shape",
                       pr.first.c_str());
                memcpy(wp.data() + row * H, t.p, t.n);
                row += pr.second;
            }
            ICHECK((int)row == y.NP, "fused rows");
            const Tens& wo = f.get(pf + p + "attn.wo_ud.weight", "BF16", {H, QD});
            // [ink-n4-int8] quantise the dense rows (not the sink / sconv rows) and wo; wp / wop hold d = bf16(q s)
            const uint16_t* wop = (const uint16_t*)wo.p;
            std::vector<uint16_t> wod;
            Q8 q8p, q8o;
            if (dense8) {
                q8_init(q8p, y.NP, H);
                size_t r0 = 0;
                auto stk = [&](const std::string& c, const char* sfx) {
                    return pf + p + c.substr(0, c.size() - 7) + sfx;   // "attn.wq_du.weight" -> "attn.wq_du.q8"
                };
                for (auto& pr : parts) {
                    static const char* cls[] = {"attn.wq_du.weight", "attn.wr_du.weight", "attn.gate_attn.weight", "attn.wk_dv.weight", "attn.wv_dv.weight"};
                    bool qd = false;
                    for (const char* c : cls) qd = qd || pr.first == c;
                    if (qd && st8)
                        st8_diff += st8_rows(wp.data(), H, (int)r0, (int)(r0 + pr.second), q8p, f8.get(stk(pr.first, ".q8")),
                                             f8.get(stk(pr.first, ".s8")));
                    else if (qd)
                        rtn8_rows(wp.data(), H, (int)r0, (int)(r0 + pr.second), q8p);
                    r0 += pr.second;
                }
                wod.assign(wop, wop + (size_t)H * QD);
                q8_init(q8o, H, QD);
                if (st8)
                    st8_diff += st8_rows(wod.data(), QD, 0, H, q8o, f8.get(stk("attn.wo_ud.weight", ".q8")),
                                         f8.get(stk("attn.wo_ud.weight", ".s8")));
                else
                    rtn8_rows(wod.data(), QD, 0, H, q8o);
                wop = wod.data();
                if (l == NL - 1)
                    fprintf(stderr, "[ink] INK_DENSE8: %s; checkpoint BF16 values != bf16(q s): %zu\n",
                            st8 ? "stored codes + scales (dense8.safetensors)" : "RTN at load", st8_diff);
            }
            if (!tps) {
                y.NPr = y.NP;
                y.qr1 = y.offK;
                y.qr2 = y.offV;
                // [ink-1c8] one card: the decode GEMVs and prefill (k_dec8) read INT8 planes of the full matrices (same
                // bits as the G15 planes of d); the G15 planes of d only for the self-test layers (freed after the
                // self-tests) or with INK_D8KEEP12=1. INK_1C8=0: G15 planes of d for everything (the old INK_DENSE8=1)
                static const bool c8 = !getenv("INK_1C8") || atoi(getenv("INK_1C8"));
                const bool c8on = dense8 && c8, stl = std::find(std::begin(D8ST), std::end(D8ST), l) != std::end(D8ST);   // tcr_selftest's / d8_selftest's layers
                // review L1: with the INT8 planes the qkv decode reads qp8 (one K part), so INK_QSQ=3 does not apply and
                // the G15 qkv copy (self-test reference / INK_D8KEEP12 prefill) is built in the plain order: the
                // qkv self-test always runs
                if (c8on && l == 0 && qsq3_on()) fprintf(stderr, "[ink] INK_QSQ=3 has no effect with INT8 dense weights (qkv reads the INT8 planes)\n");
                if (!c8on || keep12 || stl) {
                    const size_t b0 = b12, a0 = a12;
                    y.qp12 = up12(wp.data(), y.NP, H, true, 0, !c8on);
                    y.wo12 = up12(wop, H, QD, true);
                    if (c8on && !keep12) b12st += b12 - b0, d12st += a12 - a0;   // [ink-pf8] review L6: allocated bytes
                }
                if (c8on) {
                    // review L4: k_q8w's fallback-group path is not covered by the self-test: every wo row quantised (as TP)
                    for (int r = 0; r < H; r++) ICHECK(q8o.isq[r], "wo row %d not quantised", r);
                    y.qp8 = up8(q8p.q.data(), q8p.s.data(), q8p.isq.data(), wp.data(), y.NP, H);
                    y.wo8 = up8(q8o.q.data(), q8o.s.data(), q8o.isq.data(), wop, H, QD);
                }
            } else
            {
                const size_t b0 = b12, a0 = a12;
                const bool st = std::find(std::begin(D8ST), std::end(D8ST), l) != std::end(D8ST);
                const bool all12 = !dense8 || keep12;
                tp_load_attn(y, wp, wop, dense8 ? &q8p : nullptr, dense8 ? &q8o : nullptr, all12 || (st && !qsq3_on()), all12 || st);
                if (!all12 && st) {   // review L6: the allocated bytes (dev_bytes), not the encoded size (b12)
                    b12st += b12 - b0;
                    d12st += a12 - a0;
                }
            }
            if (keep16) {
                y.Wp = (bf16*)dmalloc(wp.size() * 2, &dev_bytes);
                CU(cudaMemcpy(y.Wp, wp.data(), wp.size() * 2, cudaMemcpyHostToDevice));
                y.Wo = dense8 ? upraw2<bf16>((const bf16*)wop, wod.size() * 2) : upraw<bf16>(wo);
            }
            y.qs = upf(G(p + "attn.q_l2_clip.scale"));
            y.rs = upf(G(p + "attn.rel_l2_clip.scale"));
            y.proj = upf(f.get(pf + p + "attn.rel_logits_proj.proj", "BF16", {DREL, REL}));
            y.projT = (float*)dmalloc((size_t)REL * DREL * 4, &dev_bytes);   // review r5 L5: counted
            k_projT<<<(REL * DREL + 255) / 256, 256>>>(y.proj, y.projT);
            CU(cudaGetLastError());   // review r5 L5
            CU(cudaDeviceSynchronize());
            if (y.full) {
                y.kw = upf(f.get(pf + p + "attn.k_sconv.static_w", "BF16", {4, KVD}));
                y.vw = upf(f.get(pf + p + "attn.v_sconv.static_w", "BF16", {4, KVD}));
                y.nq = upraw<bf16>(G(p + "q_input_norm.weight"));
                y.nk = upraw<bf16>(G(p + "k_input_norm.weight"));
                y.nv = upraw<bf16>(G(p + "v_input_norm.weight"));
            } else {
                y.kw = upf(f.get(pf + p + "attn.k_sconv.weight", "BF16", {KVD, 1, 4}));
                y.vw = upf(f.get(pf + p + "attn.v_sconv.weight", "BF16", {KVD, 1, 4}));
                y.nq = upraw<bf16>(G(p + "attn_norm.weight"));
            }
            y.aw = upf(f.get(pf + p + "attn_sconv.weight", "BF16", {H, 1, 4}));
            y.mw = upf(f.get(pf + p + "mlp_sconv.weight", "BF16", {H, 1, 4}));
            y.nm = upraw<bf16>(G(p + "mlp_norm.weight"));
            if (l > 0) {
                const Tens& t = G("attn_res.attn_queries." + std::to_string(l - 1));
                ICHECK((int)t.n == (y.full ? 3 : 1) * H * 4, "attn query %d", l);
                y.aq = upf(t);
            }
            y.mq = upf(f.get(pf + "attn_res.mlp_queries." + std::to_string(l), "F32", {H}));
            y.Wr = upraw<bf16>(f.get(pf + p + "mlp.gate.weight", "BF16", {NE, H}));
            y.rb = upf(f.get(pf + p + "mlp.gate.bias", "F32", {NE}));
            const Tens& w13 = G(p + "mlp.experts.w13_weight");
            y.fmt = w13.dtype == "U8" ? 0 : 1;
            if (tps) tp_load_experts(y, p, w13);
            else if (y.fmt == 0) {
                ICHECK(w13.shape == std::vector<long long>({NE, 2 * I, H / 2}), "w13 shape");
                y.c13 = upraw<uint8_t>(w13);
                y.s13 = upraw<uint8_t>(f.get(pf + p + "mlp.experts.w13_weight_scale", "F8_E4M3", {NE, 2 * I, H / 16}));
                y.c2 = upraw<uint8_t>(f.get(pf + p + "mlp.experts.w2_weight", "U8", {NE, H, I / 2}));
                y.s2 = upraw<uint8_t>(f.get(pf + p + "mlp.experts.w2_weight_scale", "F8_E4M3", {NE, H, I / 16}));
                for (int k = 0; k < 2; k++) {
                    const std::string nm = k ? "w2" : "w13";
                    const Tens& am = f.get(pf + p + "mlp.experts." + nm + "_weight_amax", "F32", {NE});
                    const Tens& dn = f.get(pf + p + "mlp.experts." + nm + "_weight_outer_scale_denominator", "F32", {1});
                    float d;
                    memcpy(&d, dn.p, 4);
                    std::vector<float> fac(NE);
                    for (int e = 0; e < NE; e++) {
                        float a;
                        memcpy(&a, am.p + 4 * e, 4);
                        fac[e] = a / d;
                    }
                    float* dp = (float*)dmalloc(NE * 4, &dev_bytes);
                    CU(cudaMemcpy(dp, fac.data(), NE * 4, cudaMemcpyHostToDevice));
                    (k ? y.f2 : y.f13) = dp;
                }
            } else {
                ICHECK(w13.dtype == "F8_E4M3" && w13.shape == std::vector<long long>({NE, 2 * I, H}), "w13 fp8 shape");
                y.c13 = upraw<uint8_t>(w13);
                y.s13 = upraw<uint8_t>(f.get(pf + p + "mlp.experts.w13_weight_scale", "U8", {NE, 2 * I, H / 32}));
                y.c2 = upraw<uint8_t>(f.get(pf + p + "mlp.experts.w2_weight", "F8_E4M3", {NE, H, I}));
                y.s2 = upraw<uint8_t>(f.get(pf + p + "mlp.experts.w2_weight_scale", "U8", {NE, H, I / 32}));
                if (tcx_min > 0)
                    tcx_rmax_layer(y, f.get(pf + p + "mlp.experts.w13_weight_scale", "U8", {NE, 2 * I, H / 32}),
                                   f.get(pf + p + "mlp.experts.w2_weight_scale", "U8", {NE, H, I / 32}), 0, I);
                std::vector<float> one(NE, 1.f);
                for (int k = 0; k < 2; k++) {
                    float* dp = (float*)dmalloc(NE * 4, &dev_bytes);
                    CU(cudaMemcpy(dp, one.data(), NE * 4, cudaMemcpyHostToDevice));
                    (k ? y.f2 : y.f13) = dp;
                }
            }
            if (y.loop) {
                const char* nm[4] = {"attn", "k", "v", "mlp"};
                for (int g = 0; g < 4; g++) {
                    const std::string c = p + "controllers." + nm[g] + ".";
                    y.gd[g] = upraw<bf16>(f.get(pf + c + "down_weight", "BF16", {GR, 2 * H}));
                    y.gf[g] = upraw<bf16>(f.get(pf + c + "forget_weight", "BF16", {H, GR}));
                    y.gg[g] = upraw<bf16>(f.get(pf + c + "gain_weight", "BF16", {H, GR}));
                }
            }
        }
        if (tcx_min > 0 && tcx_nb) fprintf(stderr, "[ink] tcx MXFP8 fp16 weight form: %lld of %lld scale blocks below row max - 14 (inexact)\n", tcx_bad, tcx_nb);
        const Tens& emb = f.get(pf + "embed.weight", "BF16", {VPAD, H});
        emb_host = (bf16*)emb.p;
        embn = upraw<bf16>(f.get(pf + "embed_norm.weight", "BF16", {H}));
        fnorm = upraw<bf16>(f.get(pf + "norm.weight", "BF16", {H}));
        outq = upf(f.get(pf + "attn_res.output_query", "F32", {H}));
        msgq = upf(f.get(pf + "middle_cycle_cross_pass_queries", "F32", {2 * (LOOP1 - LOOP0), H}));
        msgkv = upf(f.get(pf + "middle_cycle_cross_pass_kv_queries", "F32", {2 * (LOOP1 - LOOP0), H}));
        const Tens& ue = f.get(pf + "unembed.weight", "BF16", {VPAD, H});
        head2 = !(getenv("INK_HEAD2") && atoi(getenv("INK_HEAD2")) == 0);
        // [ink-tp-dec1] the head is replicated for now: one copy per device (planes + screen), work buffers per rank
        std::unique_lock<std::mutex> hl;
        Impl* howner = nullptr;
        if (tpc && !tps) {   // (ranks with a vocab part build their own)
            hl = std::unique_lock<std::mutex>(tpc->rep_mu);
            auto k = std::make_tuple(dev, (const void*)ue.p, (size_t)0, 7);
            if (tpc->rep.count(k)) howner = (Impl*)tpc->rep[k];
            else tpc->rep[k] = this;
        }
        if (howner) {
            head12 = howner->head12;
            h2 = howner->h2;
            h2.hi = (float*)dmalloc((size_t)V * 4, &dev_bytes);
            unsigned* w = (unsigned*)dmalloc(64, &dev_bytes);
            CU(cudaMemset(w, 0, 64));
            h2.thr = w;
            h2.done = w + 1;
            h2.ncand = w + 2;
            h2.best = (unsigned long long*)(w + 4);
        } else if (tps) {   // [ink-tp-dec1] the rank's vocab rows [v0, v1)
            ICHECK(head2, "TP: the two-stage head (INK_HEAD2=0 is not in the TP build)");
            head12 = up12((const uint16_t*)ue.p + (size_t)sh.v0 * H, sh.v1 - sh.v0, H);
        } else
            head12 = up12((const uint16_t*)ue.p, V, H);
        if (head2 && !howner) {
            const int V0 = tps ? sh.v0 : 0, VN = tps ? sh.v1 - sh.v0 : V;   // [ink-tp-dec1] the rank's rows
            std::vector<uint8_t> scr((size_t)VN * H), eb(VN);
            {   // threads over row ranges (review L: the encoder was single-threaded)
                const int nth = (int)std::max(1u, std::min(24u, std::thread::hardware_concurrency()));
                std::vector<std::thread> th;
                for (int i = 0; i < nth; i++) {
                    const size_t r0 = (size_t)VN * i / nth, r1 = (size_t)VN * (i + 1) / nth;
                    th.emplace_back([&, r0, r1] {
                        exp12_encode_screen((const uint16_t*)ue.p + (V0 + r0) * H, r1 - r0, H, scr.data() + r0 * H, eb.data() + r0);
                    });
                }
                for (auto& x : th) x.join();
            }
            h2.scr = upraw2<uint8_t>(scr.data(), scr.size());
            h2.seb = upraw2<uint8_t>(eb.data(), eb.size());
            h2.N = VN;
            h2.r0 = V0;
            h2.hi = (float*)dmalloc((size_t)V * 4, &dev_bytes);
            unsigned* w = (unsigned*)dmalloc(64, &dev_bytes);
            CU(cudaMemset(w, 0, 64));
            h2.thr = w;
            h2.done = w + 1;
            h2.ncand = w + 2;
            h2.best = (unsigned long long*)(w + 4);
            h2.sm = head12.sm;
            h2.idx = head12.idx;
            h2.meta = head12.meta;
            h2.fb = head12.fb;
        }
        if (hl.owns_lock()) hl.unlock();
        if (!tps) {
            dscr = h2.scr;
            dseb = h2.seb;
        }
        if (keep16) unemb = upraw<bf16>(ue, (size_t)V * H * 2);
        fprintf(stderr, "[ink] G15 planes (qkv, wo, unembed) %.2f GB, %d matrices round-trip checked\n", b12 / 1e9, verified12);
        if (!tps) tcr_selftest();
        const bool d8st = dense8 && (tps || L[0].qp8.sm);   // [ink-1c8] one card too (its INT8 planes: INK_1C8)
        if (d8st) d8_selftest();
        if (d8st && !keep12) {   // [ink-pf8] the G15 copies of the self-test layers
            for (int l : D8ST) {
                free12(L[l].qp12);
                free12(tps ? L[l].woR : L[l].wo12);
            }
            dev_bytes -= d12st;
            b12 -= b12st;
            if (sh.rank == 0)
                fprintf(stderr, "[ink] dense8: %s read the INT8 planes (%d matrices round-trip checked), no G15 copy of qkv / wo; "
                                "G15 planes now %.2f GB\n", tps ? "prefill and decode" : "decode and prefill", verified8, b12 / 1e9);
        }
        if (tpc) tp_fit();
        else fit_memory();   // may turn the drafter off or shrink max_ctx (never fail on memory)
        if (dr_on) dr_load();
        fprintf(stderr, "[ink] weights on GPU %.2f GB\n", dev_bytes / 1e9);
    }
    // Memory fit (rule: the engine never fails on memory; it shrinks what it must, warns and keeps serving). Called
    // after the target weights are on the card, before the drafter weights and the state. Estimates of what is still to
    // come: per-token K / V of the full-attention applications (BF16) and of the drafter's global layer (E4M3 + fp32
    // scale), plus fixed state (measured 11-23: rings, chunk buffers, verify scratch ~0.97 GB at any max_ctx; drafter
    // weights 0.61 GB + fixed drafter state 0.08 GB) and a margin for graphs / allocator rounding. Policy: keep the
    // requested context if possible (drafter off when only that fits), else the largest context that fits.
    size_t ctx_tok_bytes(bool dr) const {
        int nfull = 0;
        for (int a = 0; a < NAPP; a++) nfull += L[a < NL ? a : LOOP0 + (a - NL) % (LOOP1 - LOOP0)].full ? 1 : 0;
        size_t b = (size_t)nfull * 2 * nkvA() * HD * 2;
        if (dr) b += 2 * (size_t)NKV * (HD + 4) + (getenv("INK_DR_TEST") ? (size_t)H * 2 + 4 * (size_t)NKV * (HD + 4) : 0);
        return b;
    }
    size_t pf2_bytes() const {   // review tp2 M3: sub-chunks 1..K-1 (decode buffer, cuBLAS workspace, z partials, routing)
        return pf2 ? (size_t)(pfk - 1) * ((size_t)(KVD + NKV * 4) * 2 * H * 2 + (size_t)OFF_END * H * 2 + ((size_t)32 << 20) + ((size_t)16 << 20)) : 0;
    }
    size_t fixed_bytes(bool dr) const {
        size_t b = (size_t)(0.97e9) + (size_t)(0.06e9) + pf2_bytes();   // state + margin (180,224 without the drafter ran with ~70 MB left, 11-22)
        // drafter: weights + fixed state + runtime room for the verify / drafter graphs (149,504 at a 0.06 GB margin: OOM in a spec run)
        if (dr) b += (size_t)(0.61e9 + 0.08e9 + 0.30e9) + (getenv("INK_DR_EXACT") && atoi(getenv("INK_DR_EXACT")) ? (size_t)V * 256 * 2 : 0);
        if (dr && tps) {   // review tp M3: a TP rank's 8-bit screen copy for the drafter (tp_dr_screen); review6 L7: its chunks when split
            const bool sp = sh.N > 1 && drs_on && !(getenv("INK_DR_EXACT") && atoi(getenv("INK_DR_EXACT"))) && ab != 52 && ab != 70;
            b += sp ? (size_t)(std::min(drc1() * MKC_CH, V) - drc0() * MKC_CH) * (H + 1) : (size_t)V * H + V;
        }
        return b;
    }
    // [ink-tp-dec1] memory fit of a rank, then the smallest pool / context of all ranks for every rank; ranks sharing a
    // device (virtual TP) keep the requested context (the caller sizes it)
    int tp_pool = 0, tp_ctx = 0, tp_S = 1;
    int tp_ch[2] = {0, 0};   // [ink-tp-pf] this rank's chunk ladder choice (capacity, split)
    bool tp_dr = false;
    void tp_fit() {
        if (tpc->per_dev[dev] > 1) {
            POOL = o.pool > 0 ? o.pool : S * o.max_ctx;
            fprintf(stderr, "[ink] TP rank %d shares GPU %d with %d ranks: no memory fit (pool %d rows)\n", sh.rank, dev, tpc->per_dev[dev], POOL);
        } else
            fit_memory();
        tpc->bar.wait();
        if (sh.rank == 0) {
            int pm = POOL, cm = o.max_ctx, sm = S;
            bool dr = dr_on;
            for (auto* r : tpc->R) {
                pm = std::min(pm, r->POOL);
                cm = std::min(cm, r->o.max_ctx);
                sm = std::min(sm, r->S);   // review tp M3: one slot count (fewer slots than a rank's fit: less memory)
                dr = dr && r->dr_on;
            }
            tp_pool = pm;
            tp_ctx = cm;
            tp_S = sm;
            tp_dr = dr;
        }
        tpc->bar.wait();
        POOL = tpc->R[0]->tp_pool;
        o.max_ctx = tpc->R[0]->tp_ctx;
        S = tpc->R[0]->tp_S;
        dr_on = tpc->R[0]->tp_dr;   // every rank runs the same drafter or none
        if (dr_on && tps) tp_dr_screen();
    }
    void fit_memory() {
        size_t fr = 0, tot = 0;
        CU(cudaMemGetInfo(&fr, &tot));
        // slots: every slot has its rings + conv states (fixed) and max_ctx global K / V rows
        const size_t ring_b = (size_t)(NAPP - 8) * 2 * NKV * RING * HD * 2 + (size_t)NAPP * 6 * (KVD + H) * 4;
        // [ink-srv-merge] the batch build's MROW-row buffers, row-step graph room (S > 1), per-slot drafter rings and the
        // block buffers for min(S, 8) sequences, and the prefill reserve: a chunk of >= PF_MIN_ROWS rows. The 0.97 GB
        // of fixed_bytes already leaves ~0.72 GB for the chunk (M4 11-23: 141,312 + drafter ran with a 2,688-row chunk
        // at 277 KB/row and 555 MiB free at peak); only a larger minimum chunk adds to it.
        auto extra = [&](bool dr) {
            size_t b = batch_row_bytes() + srv_graph_room();
            if (dr) b += (size_t)(S - 1) * 2 * RING * NKV * (2 * HD + 8) + dr_blk_bytes(std::min(S, 8)) - dr_blk_bytes(1);
            const size_t pfmin = (size_t)PF_MIN_ROWS * pf_row_bytes(dr), covered = (size_t)(0.72e9);
            if (pfx && pfat && pfmin > covered) b += pfmin - covered;
            return b;
        };
        // [ink-srv-capi] server mode (Opts::pool_auto > 0, the C API): the pool is every row that fits with the drafter
        // kept (at most slots x max_ctx); while that pool is below pool_auto rows the slots are halved (each slot costs
        // its rings: ~0.34 GB). No fit with the drafter at all (< 1,024 rows): the policy below.
        if (o.pool_auto > 0 && o.pool <= 0) {
            // [ink-srv-hog] no fit with the drafter even at 1 slot: the same without the drafter (warned)
            const int S0 = S;
            bool floor_hit = false;
            for (int pass = dr_on ? 0 : 1; pass < 2; pass++) {
                const bool dr = pass == 0;
                int c = 0;
                for (S = S0;;) {
                    const int want = (int)std::min<long long>((long long)S * o.max_ctx, 1 << 30);
                    const size_t fx = fixed_bytes(dr) + extra(dr) + (size_t)(S - 1) * ring_b;
                    const size_t m = fr > fx ? (fr - fx) / ctx_tok_bytes(dr) : 0;
                    c = m >= (size_t)want ? want : (int)(m / 1024 * 1024);
                    if (c >= std::min(o.pool_auto, want) || S == 1) break;
                    S = std::max(1, S / 2);
                }
                // [ink-srv-review-fix] M1: the drafter stays only while its pool reaches a floor (INK_DRAFT_MIN_CTX,
                // default 16,384 rows, at most what was asked): its ~1 GB fixed cost buys ~30K rows at 1 slot, so a
                // drafter kept at a 1-4K context refused every longer prompt the card could serve without it
                static const int dmin = [] {
                    const char* v = getenv("INK_DRAFT_MIN_CTX");
                    return std::max(1024, v && *v ? atoi(v) : 16384);
                }();
                const int floor_dr = std::min(dmin, (int)std::min<long long>((long long)S0 * o.max_ctx, 1 << 30));
                if (dr && c >= 1024 && c < floor_dr) {
                    fprintf(stderr, "[ink] WARNING: %.2f GB free after the weights: with the drafter the pool would be %d rows (< %d): drafter OFF "
                                    "(plain decoding) to keep the context\n", fr / 1e9, c, floor_dr);
                    floor_hit = true;
                    continue;
                }
                if (c >= 1024) {
                    if (dr_on && !dr && !floor_hit)
                        fprintf(stderr, "[ink] WARNING: %.2f GB free after the weights: no room for the drafter, drafter OFF (plain decoding)\n", fr / 1e9);
                    dr_on = dr;
                    POOL = c;
                    o.max_ctx = std::min(o.max_ctx, c);
                    fprintf(stderr, "[ink] server pool: %d slots, %d rows (%.2f GB free after the weights; drafter %s)\n", S, POOL, fr / 1e9, dr ? "on" : "off");
                    return;
                }
            }
            S = 1;   // nothing fits: the policy below says why
        }
        // [ink-srv-pool] c = rows of the pool (all slots); one slot's context max_ctx <= the pool
        const int reqp = POOL = o.pool > 0 ? o.pool : S * o.max_ctx;
        auto need = [&](int c, bool dr) { return fixed_bytes(dr) + extra(dr) + (size_t)(S - 1) * ring_b + (size_t)c * ctx_tok_bytes(dr); };
        auto fit = [&](bool dr) -> int {
            const size_t fx = fixed_bytes(dr) + extra(dr) + (size_t)(S - 1) * ring_b;
            if (fr <= fx) return 0;
            const size_t c = (fr - fx) / ctx_tok_bytes(dr);
            return c >= (size_t)reqp ? reqp : (int)(c / 1024 * 1024);   // review r4 L: cap before rounding
        };
        const int req = reqp;
        if (need(req, dr_on) <= fr) return;
        const int cd = dr_on ? fit(true) : 0, cp = fit(false);
        if (dr_on && cp >= req) {
            fprintf(stderr, "[ink] WARNING: max_ctx %d with the drafter needs ~%.2f GB, %.2f GB free: drafter OFF (plain decoding) to keep "
                            "the context\n", req, need(req, true) / 1e9, fr / 1e9);
            if (cd >= 1024) fprintf(stderr, "[ink] with the drafter the largest context is %d (--max-ctx %d)\n", cd, cd);
            dr_on = false;
            return;
        }
        const bool dr = dr_on && cd >= cp;   // the drafter stays only if it costs no context
        const int c = dr ? cd : cp;
        ICHECK(c >= 1024, "not enough device memory for any context (%.2f GB free after the weights)", fr / 1e9);
        fprintf(stderr, "[ink] WARNING: max_ctx %d needs ~%.2f GB%s, %.2f GB free: max_ctx -> %d%s\n", req,
                need(req, dr_on) / 1e9, dr_on ? " with the drafter" : "", fr / 1e9, c,
                dr_on && !dr ? ", drafter OFF (plain decoding; with the drafter it would be smaller)" : "");
        if (dr_on && !dr && cd >= 1024) fprintf(stderr, "[ink] with the drafter the largest context is %d (--max-ctx %d)\n", cd, cd);
        dr_on = dr;
        POOL = c;
        o.max_ctx = std::min(o.max_ctx, c);
    }
    // application index of layer l in pass p (1-based): the HF cache layer index
    static int app(int l, int p) { return (l >= LOOP0 && l < LOOP1 && p > 1) ? NL + (p - 2) * (LOOP1 - LOOP0) + (l - LOOP0) : l; }
    void alloc_state() {
        // review2 L6: k_pattn<false> forms the global K / V element index (kv head x cap + key) x HD in int
        ICHECK((size_t)NKV * POOL * HD < (size_t)INT_MAX, "pool %d too large for the int K / V index", POOL);
        sbase.resize(S);
        slim.assign(S, o.max_ctx);
        for (int i = 0; i < S; i++) {   // default intervals (bind moves them): equal shares of the pool
            slim[i] = std::min(o.max_ctx, POOL / S);
            sbase[i] = i * slim[i];
        }
        SKO_R = (int*)dmalloc(S * 4, &dev_bytes);
        SKO_G = (int*)dmalloc(S * 4, &dev_bytes);
        SKO_DR = (int*)dmalloc(S * 4, &dev_bytes);
        sT.assign(S, 0.f);
        sS.assign(S, 0u);
        sK.assign(S, 0);
        fb_ema.assign(S, 0.f);
        sTP.assign(S, 1.f);
        sMP.assign(S, 0.f);
        DSAMP = (float*)dmalloc(SAMP_N * 4, &dev_bytes);
        CU(cudaMemset(DSAMP, 0, SAMP_N * 4));
        CU(cudaHostAlloc((void**)&HSAMP, SAMP_N * 4, 0));
        memset(HSAMP, 0, SAMP_N * 4);
        if (head2 && !(getenv("INK_SAMP_DEV") && atoi(getenv("INK_SAMP_DEV")) == 0)) {   // [ink-samp-dev] (INK_SAMP_DEV=0: off)
            loR = (float*)dmalloc((size_t)MROW * (tps ? sh.v1 - sh.v0 : V) * 4, &dev_bytes);   // review L4: the rank's rows
            fcR = (unsigned*)dmalloc(MROW * 16 * 4, &dev_bytes);
            CU(cudaMemset(fcR, 0, MROW * 16 * 4));
            fhist = (unsigned*)dmalloc((size_t)MROW * FHB * 4, &dev_bytes);
            CU(cudaMemset(fhist, 0, (size_t)MROW * FHB * 4));
            flist = (unsigned long long*)dmalloc((size_t)MROW * FCAP * 8, &dev_bytes);
            FLOC = (unsigned long long*)dmalloc((size_t)MROW * FLOC_W * 8, &dev_bytes);
            CU(cudaMemset(FLOC, 0, (size_t)MROW * FLOC_W * 8));
            ftail = (double*)dmalloc(MROW * 8, &dev_bytes);
            CU(cudaMemset(ftail, 0, MROW * 8));
            CU(cudaEventCreateWithFlags(&fev, cudaEventDisableTiming));
        }
        for (int a = 0; a < NAPP; a++) {
            const int l = a < NL ? a : LOOP0 + (a - NL) % (LOOP1 - LOOP0);
            AppState& s = A[a];
            s.ring = !L[l].full;
            s.cap = s.ring ? RING : POOL;
            // rings [slot][KV head][RING][HD]; full layers one pool [KV head][POOL][HD] ([ink-srv-pool])
            const size_t kvb = std::max<size_t>((s.ring ? (size_t)S : 1) * nkvA() * s.cap * HD * 2, 256);
            s.K0 = (bf16*)dmalloc(kvb, &dev_bytes);
            s.V0 = (bf16*)dmalloc(kvb, &dev_bytes);
            s.kvb = kvb;
            CU(cudaMemset(s.K0, 0, kvb));   // defined contents (bench fake depth reads them)
            CU(cudaMemset(s.V0, 0, kvb));
            // [ink-tp-dec1] a rank holds KV heads [kv0, kv0 + nkv): the kernels index head h at (h x cap + key) x HD
            s.K = s.K0 - (ptrdiff_t)kv0A() * s.cap * HD;
            s.V = s.V0 - (ptrdiff_t)kv0A() * s.cap * HD;
            s.kst = (float*)dmalloc((size_t)S * 3 * KVD * 4, &dev_bytes);
            s.vst = (float*)dmalloc((size_t)S * 3 * KVD * 4, &dev_bytes);
            s.ast = (float*)dmalloc((size_t)S * 3 * H * 4, &dev_bytes);
            s.mst = (float*)dmalloc((size_t)S * 3 * H * 4, &dev_bytes);
            s.kh = (float*)dmalloc((size_t)MROW * KVD * 4, &dev_bytes);
            s.vh = (float*)dmalloc((size_t)MROW * KVD * 4, &dev_bytes);
            s.ah = (float*)dmalloc((size_t)MROW * H * 4, &dev_bytes);
            s.mh = (float*)dmalloc((size_t)MROW * H * 4, &dev_bytes);
            A0[a] = s;
            for (int sl = 0; sl < S; sl++) {   // every slot starts empty
                const AppState v = view(a, sl);
                CU(cudaMemset(v.kst, 0, 3 * KVD * 4));
                CU(cudaMemset(v.vst, 0, 3 * KVD * 4));
                CU(cudaMemset(v.ast, 0, 3 * H * 4));
                CU(cudaMemset(v.mst, 0, 3 * H * 4));
            }
        }
        // prefill chunk ladder ([ink-pf-chunk]); [ink-tp-pf] TP ranks: the same ladder (collective buffers in pf_row_bytes);
        // ranks sharing a device (virtual TP, tests) 512 rows; decode-row fallback 64
        ch = tps && !pfx ? 64 : tps && tpc->per_dev[dev] > 1 ? (chs = std::min(512, (int)CH)) : pf_choose_chunk();
        if (tps && pfx) {   // [ink-tp-pf] one chunk plan for every rank (the collectives pair chunk by chunk): the smallest
            tp_ch[0] = ch;
            tp_ch[1] = chs;
            tpc->bar.wait();
            for (auto* r : tpc->R) {
                ch = std::min(ch, r->tp_ch[0]);
                chs = std::min(chs, r->tp_ch[1]);
            }
            tpc->bar.wait();
        }
        spos.assign(S, 0);
        shist.assign(S, {});
        DRT = (int*)dmalloc(RT_N * 4, &dev_bytes);
        CU(cudaMemset(DRT, 0, RT_N * 4));
        CU(cudaHostAlloc((void**)&HRT, RT_N * 4, 0));
        memset(HRT, 0, RT_N * 4);
        {   // conv-state commit table (verify)
            std::vector<CommitEnt> ce;
            for (int a = 0; a < NAPP; a++) {
                ce.push_back({A0[a].kst, A0[a].kh, KVD});
                ce.push_back({A0[a].vst, A0[a].vh, KVD});
                ce.push_back({A0[a].ast, A0[a].ah, H});
                ce.push_back({A0[a].mst, A0[a].mh, H});
            }
            commit_ents = (CommitEnt*)dmalloc(ce.size() * sizeof(CommitEnt), &dev_bytes);
            CU(cudaMemcpy(commit_ents, ce.data(), ce.size() * sizeof(CommitEnt), cudaMemcpyHostToDevice));
            ACC = (int*)dmalloc(MROW * 4, &dev_bytes);   // per segment
            hiR = (float*)dmalloc((size_t)MROW * V * 4, &dev_bytes);
            hcR = (unsigned*)dmalloc(MROW * 64, &dev_bytes);
            CU(cudaMemset(hcR, 0, MROW * 64));
            VTOK = (int*)dmalloc(MROW * 4, &dev_bytes);   // row-step tokens (k_accept of batched rows)
            CU(cudaHostAlloc((void**)&HTOK, MROW * 4, 0));
            std::vector<int> io(ch);
            for (int i = 0; i < ch; i++) io[i] = i;
            IOTA = (int*)dmalloc(ch * 4, &dev_bytes);
            CU(cudaMemcpy(IOTA, io.data(), ch * 4, cudaMemcpyHostToDevice));
        }
        if (dr_on) dr_alloc();
        const size_t C = ch;
        auto F = [&](size_t n) { return (float*)dmalloc(n * 4, &dev_bytes); };
        E = F(C * H); X6 = F(C * H); X12[0] = F(C * H); X12[1] = F(C * H); H17[0] = F(C * H); H17[1] = F(C * H);
        X18 = F(C * H); HA = F(C * H); HB = F(C * H);
        int npmax = 0;
        for (int l = 0; l < NL; l++) npmax = std::max(npmax, L[l].NP);
        pf_npmax = npmax;
        if (pfx && pf2) {   // [ink-tp-pf2] two halves run different phases at once: no phase aliases inside pBig (P, messages,
                            // attention inputs on their own; pBig holds the MoE slots only)
            pf_big(C, npmax);
            for (int i = 0; i < 3; i++) { XA[i] = F(C * H); XG[i] = F(C * H); XB[i] = (bf16*)dmalloc(C * H * 2, &dev_bytes); }
            for (int i = 0; i < 4; i++) M[i] = F(C * H);
        } else if (pfx) {   // [ink-pf-chunk] the message / attention-input rows live inside the MoE slot region (pf_big: 10 row
                     // blocks; dead while the expert GEMMs run); P (qkv out) at its front, dead after attention
            float* R = pf_big(C, npmax);
            for (int i = 0; i < 3; i++) { XA[i] = R + (size_t)i * C * H; XG[i] = R + (size_t)(3 + i) * C * H; }
            for (int i = 0; i < 4; i++) M[i] = R + (size_t)(6 + i) * C * H;
            for (int i = 0; i < 3; i++) XB[i] = (bf16*)dmalloc(C * H * 2, &dev_bytes);
        } else {
            for (int i = 0; i < 3; i++) { XA[i] = F(C * H); XG[i] = F(C * H); XB[i] = (bf16*)dmalloc(C * H * 2, &dev_bytes); }
            for (int i = 0; i < 4; i++) M[i] = F(C * H);
        }
        Rr = F(C * H);
        if (pfx) {   // phase-disjoint aliases ([ink-pf-chunk]): wo out / MLP msg input+output / MoE input / final hidden
            ATT = XG[0]; MI = XA[1]; MIG = XA[2]; XMF = XG[1]; MO = F(C * H); FIN = XA[0];
        } else {
            ATT = F(C * H); MI = F(C * H); MIG = F(C * H); XMF = F(C * H); MO = F(C * H); FIN = F(C * H);
        }
        P = pfx && !pf2 ? pBig : F(C * npmax); Qf = F(C * QD); Rf = F(C * NQ * DREL); Sf = F(C * NQ);
        if (!tps || pfx) WS = (bf16*)dmalloc((size_t)(KVD + NKV * 4) * 2 * H * 2 + (size_t)OFF_END * H * 2 + 4096, &dev_bytes);   // [ink-tp-dec1] prefill only
        if (pf2)   // [ink-tp-pf2] sub-chunks 1..K-1: stream, cuBLAS handle + workspace, position, decode buffer, events
            for (int j = 0; j < pfk; j++) {
                for (auto& e : pfEv[j]) CU(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
                CU(cudaEventCreateWithFlags(&pfJ[j], cudaEventDisableTiming));
                if (!j) continue;
                WS2[j] = (bf16*)dmalloc((size_t)(KVD + NKV * 4) * 2 * H * 2 + (size_t)OFF_END * H * 2 + 4096, &dev_bytes);
                CU(cudaStreamCreateWithFlags(&st2[j], cudaStreamNonBlocking));
                CU(cudaStreamCreateWithFlags(&cps2[j], cudaStreamNonBlocking));
                CB(cublasCreate(&blas2[j]));
                CB(cublasSetStream(blas2[j], st2[j]));
                CB(cublasSetMathMode(blas2[j], CUBLAS_DEFAULT_MATH));
                CB(cublasSetWorkspace(blas2[j], dmalloc(32 << 20, &dev_bytes), 32 << 20));
                dpos2[j] = (int*)dmalloc(RT_N * 4, &dev_bytes);
                CU(cudaMemset(dpos2[j], 0, RT_N * 4));
                CU(cudaHostAlloc((void**)&hpos2[j], 64, 0));
            }
        ACT = F((pfx ? MROW : C) * TOPK * I); WTS = F(C * TOPK); LOG = F(V);   // ACT: row steps hold MROW rows
        AO = (bf16*)dmalloc(C * QD * 2, &dev_bytes); XMB = (bf16*)dmalloc(C * H * 2, &dev_bytes);
        FINB = (bf16*)dmalloc(C * H * 2, &dev_bytes);
        IDS = (int*)dmalloc(C * TOPK * 4, &dev_bytes); ARG = (int*)dmalloc(MROW * 4, &dev_bytes);   // per row
        ORD = (OrdBuf*)dmalloc(sizeof(OrdBuf), &dev_bytes);
        DPART = F((size_t)MROW * TOPK * H);
        if (tcx_min > 0) {   // [ink-tp-tcx]
            TXH = (__half*)dmalloc((size_t)MROW * H * 2, &dev_bytes);
            TXL = (__half*)dmalloc((size_t)MROW * H * 2, &dev_bytes);
            TAH = (__half*)dmalloc((size_t)MROW * TOPK * I * 2, &dev_bytes);
            TAL = (__half*)dmalloc((size_t)MROW * TOPK * I * 2, &dev_bytes);
            TPX = (int*)dmalloc(MROW * 4, &dev_bytes);
            ORD8 = (OrdBuf*)dmalloc(sizeof(OrdBuf), &dev_bytes);
            fprintf(stderr, "[ink] tensor-core experts from %d rows (parts %d, NVFP4 form %d): NOT bitwise == alone above it\n", tcx_min, tcx_parts, tcx_v);
        }
        dpos = (int*)dmalloc(RT_N * 4, &dev_bytes);   // plain / prefill: position at [0]; the slot entries stay 0 (a row table of slot 0)
        CU(cudaMemset(dpos, 0, RT_N * 4));
        ACNT = (int*)dmalloc((NQ + MROW * NKV) * 4, &dev_bytes);
        RCNT = (int*)dmalloc(MROW * 4, &dev_bytes);   // router last-block counters per row
        CU(cudaMemset(RCNT, 0, MROW * 4));
        MN = (float*)dmalloc(12 * MROW * 4 * 4, &dev_bytes);
        if (getenv("INK_BRES_CHK")) {
            BCHK = (unsigned*)dmalloc(67 * 4, &dev_bytes);
            CU(cudaMemset(BCHK, 0, 67 * 4));
            const unsigned big = 0x7f7fffffu;
            CU(cudaMemcpy(BCHK + 1, &big, 4, cudaMemcpyHostToDevice));
        }
        CU(cudaMemset(ACNT, 0, (NQ + MROW * NKV) * 4));
        CU(cudaHostAlloc((void**)&hpos, 64, 0));
        XZ = F((size_t)3 * MROW * 2 * H);
        CU(cudaFuncSetAttribute(k_bresZ4<4>, cudaFuncAttributeMaxDynamicSharedMemorySize, 4 * 2 * H * 4));
        ZB = F(3 * C * GR); SC = F(C * NE); RFP = F((size_t)MROW * RF_NB * NE); RFS = F(MROW * RF_NB); M48 = F((size_t)MROW * 4 * (LOOP1 - LOOP0) * H); BMAX = F(2 * (V / 16 + 16));
        {   // split partials: row steps need MROW x TB_NSX slots per head; the CUDA-core prefill path PSPLIT
            const size_t ps = pf_psplit() && !tps ? (size_t)PSPLIT : (size_t)MROW * TB_NSX;   // [ink-tp-dec1] TP ranks: decode rows only
              // row steps: the tb partial slots of MROW rows (>= 128 splits per row)
            PM = F(ps * NQ); PL = F(ps * NQ); PA = F(ps * NQ * HD);
        }
        if (pfx) pf_alloc(ch);
        if (tps && pfx) {   // [ink-tp-sp] gathered buffers' bases + events
            for (int i = 0; i < 3; i++) spb[SPB_XB + i] = XB[i];
            spb[SPB_X16] = pX16;
            spb[SPB_IDS] = IDS;
            spb[SPB_WTS] = WTS;
            spb[SPB_FIN] = FIN;
            spb[SPB_FINB] = FINB;
            if (dr_on)
                for (int i = 0; i < 3; i++) spb[SPB_TAP + i] = TAP[i];
            for (int i = 0; i < 2; i++) {
                for (int j = 0; j < PS_MAX; j++) {
                    CU(cudaEventCreateWithFlags(&agF[j][i], cudaEventDisableTiming));
                    CU(cudaEventCreateWithFlags(&agS[j][i], cudaEventDisableTiming));
                    CU(cudaEventCreateWithFlags(&rsR[j][i], cudaEventDisableTiming));
                }
            }
            CU(cudaStreamCreateWithFlags(&cps, cudaStreamNonBlocking));
        }
        if (dr_on && pfx) {   // [ink-pf-dralias] drafter features / K-V rows of a chunk live only inside dr_ctx (after the
                              // last layer, before the next forward): the MoE slot region holds them (row blocks 1-3 of 10)
            bf16* b = (bf16*)(pBig + (size_t)ch * H);   // behind FIN (= XA[0], row block 0: read by dr_ctx and after it)
            for (int i = 0; i < 4; i++) DF[i] = b + (size_t)i * ch * H;
            DKVb = (float*)(b + (size_t)4 * ch * H);
        }
        CU(cudaHostAlloc((void**)&emb_rows_h, C * H * 2, 0));
        emb_rows_d = (bf16*)dmalloc(C * H * 2, &dev_bytes);
        if (tpc) {   // [ink-tp-dec1] collective buffers
            WOP = F((size_t)MROW * H);
            MOP = F((size_t)MROW * H);
            CU(cudaMemset(WOP, 0, (size_t)MROW * H * 4));   // a rank without heads / experts adds zeros
            CU(cudaMemset(MOP, 0, (size_t)MROW * H * 4));
            BKEY = (unsigned long long*)dmalloc(MROW * 8, &dev_bytes);
            CU(cudaMemset(BKEY, 0, MROW * 8));
            tstep = (unsigned*)dmalloc(64, &dev_bytes);
            CU(cudaMemset(tstep, 0, 64));
            if (tchk_every > 0) {
                tchk = (unsigned*)dmalloc(1024, &dev_bytes);
                CU(cudaMemset(tchk, 0, 1024));
            }
            for (auto& e : tev) CU(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
            GATH = (float*)dmalloc(H * 4, &dev_bytes);   // [ink-tp-drv2] drafter gather rows (allocated outside any capture)
            GATO = (float*)dmalloc(H * 4, &dev_bytes);
            if (tpc->push) {
                const size_t nb = (size_t)TP_SLOTS_ALL * sh.N * MROW * TPW * 16;
                trw = (uint4*)dmalloc(nb, &dev_bytes);
                CU(cudaMemset(trw, 0, nb));
                trw8 = (uint2*)dmalloc((size_t)TP8_SLOTS * sh.N * H * 8, &dev_bytes);   // [ink-tpf-ppush]
                CU(cudaMemset(trw8, 0, (size_t)TP8_SLOTS * sh.N * H * 8));
                if (tpcw > 0) {   // [ink-vp-cpush]
                    trwC = (uint4*)dmalloc((size_t)TPC_SLOTS * sh.N * TPC_RW * H * 16, &dev_bytes);
                    CU(cudaMemset(trwC, 0, (size_t)TPC_SLOTS * sh.N * TPC_RW * H * 16));
                }
            }
        }
        sko_upload();
        fprintf(stderr, "[ink] device total %.2f GB (max_ctx %d, pool %d rows, %d slots)\n", dev_bytes / 1e9, o.max_ctx, POOL, S);
    }
    // ---------------------------------------------------------------- DSpark drafter ([ink-dspark-fwd])
    void gemm_b(const bf16* W, int N, int K, const bf16* X, int T, float* Y, int ldy, float beta) {
        const float al = 1.f;
        CB(cublasGemmEx(blas, CUBLAS_OP_T, CUBLAS_OP_N, N, T, K, &al, W, CUDA_R_16BF, K, X, CUDA_R_16BF, K, &beta, Y,
                        CUDA_R_32F, ldy, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
    }
    void dr_load() {
        const std::string dfile = drafter_file(o.model_dir);
        fprintf(stderr, "[ink] drafter file %s (%s stop policy)\n", dfile.c_str(), drafter_qe(dfile) ? "trained" : "official");
        df.open(dfile);
        k_rope_init<<<1, 64>>>();   // [ink-dr-rope]
        const size_t b0 = dev_bytes;
        auto R = [&](const std::string& k, std::vector<long long> sh) -> const Tens& { return df.get(k, "BF16", sh); };
        auto cat = [&](std::vector<const Tens*> ts) {
            size_t n = 0;
            for (auto t : ts) n += t->n;
            std::vector<uint8_t> h(n);
            size_t o2 = 0;
            for (auto t : ts) {
                memcpy(h.data() + o2, t->p, t->n);
                o2 += t->n;
            }
            return upraw2<bf16>(h.data(), n);
        };
        for (int i = 0; i < 3; i++) {
            const std::string p = "layers." + std::to_string(i) + ".";
            DLayer& d = DL[i];
            d.dn = upraw<bf16>(R(p + "dflare_norm.weight", {H}));
            d.iln = upraw<bf16>(R(p + "input_layernorm.weight", {H}));
            d.pln = upraw<bf16>(R(p + "post_attention_layernorm.weight", {H}));
            d.qn = upraw<bf16>(R(p + "self_attn.q_norm.weight", {HD}));
            d.kn = upraw<bf16>(R(p + "self_attn.k_norm.weight", {HD}));
            d.knc = upraw<bf16>(R(p + "self_attn.k_norm_ctx.weight", {HD}));
            {   // [ink-dr-g15] q | k | v and gate | up in G15 (same bits, fragment order) for the tensor-core GEMV
                auto host = [&](std::vector<const Tens*> ts) {
                    size_t n = 0;
                    for (auto t : ts) n += t->n;
                    std::vector<uint16_t> h(n / 2);
                    size_t o2 = 0;
                    for (auto t : ts) {
                        memcpy((uint8_t*)h.data() + o2, t->p, t->n);
                        o2 += t->n;
                    }
                    return h;
                };
                const auto hq = host({&R(p + "self_attn.q_proj.weight", {QD, H}), &R(p + "self_attn.k_proj.weight", {KVD, H}),
                                      &R(p + "self_attn.v_proj.weight", {KVD, H})});
                const bool asl = tps && sh.N > 1 && das_on && !getenv("INK_DR_BF16");   // [ink-tp-dra] the rank's heads only
                if (!asl) d.Wqkv12 = up12(hq.data(), DQKV, H, true);
                else {   // q rows of heads [q0, q1), then k rows and v rows of KV heads [kv0, kv1)
                    const int nq = sh.nq(), nk = sh.nkv();
                    std::vector<uint16_t> hr((size_t)(nq + 2 * nk) * HD * H);
                    memcpy(hr.data(), hq.data() + (size_t)sh.q0 * HD * H, (size_t)nq * HD * H * 2);
                    memcpy(hr.data() + (size_t)nq * HD * H, hq.data() + ((size_t)QD + sh.kv0 * HD) * H, (size_t)nk * HD * H * 2);
                    memcpy(hr.data() + (size_t)(nq + nk) * HD * H, hq.data() + ((size_t)QD + KVD + sh.kv0 * HD) * H, (size_t)nk * HD * H * 2);
                    d.Wqkv12r = up12(hr.data(), (nq + 2 * nk) * HD, H, true);
                    const Tens& wo = R(p + "self_attn.o_proj.weight", {H, QD});
                    std::vector<uint16_t> ho((size_t)H * nq * HD);
                    for (int r = 0; r < H; r++) memcpy(ho.data() + (size_t)r * nq * HD, (const uint16_t*)wo.p + (size_t)r * QD + sh.q0 * HD, (size_t)nq * HD * 2);
                    d.Wor = upraw2<bf16>(ho.data(), ho.size() * 2);
                }
                const auto hg = host({&R(p + "mlp.gate_proj.weight", {DFF, H}), &R(p + "mlp.up_proj.weight", {DFF, H})});
                const bool msl = tps && sh.N > 1 && dms_on && !getenv("INK_DR_BF16");   // [ink-tp-drv2b] review5 M3: the slices only
                if (!msl) d.Wgu12 = up12(hg.data(), 2 * DFF, H, true);
                if (msl) {   // [ink-tp-drv2b] the rank's MLP columns (16-aligned ranges)
                    d.f0 = (int)((long long)(DFF / 16) * sh.rank / sh.N) * 16;
                    d.nf = (int)((long long)(DFF / 16) * (sh.rank + 1) / sh.N) * 16 - d.f0;
                    std::vector<uint16_t> hr((size_t)2 * d.nf * H);
                    memcpy(hr.data(), hg.data() + (size_t)d.f0 * H, (size_t)d.nf * H * 2);
                    memcpy(hr.data() + (size_t)d.nf * H, hg.data() + (size_t)(DFF + d.f0) * H, (size_t)d.nf * H * 2);
                    d.Wgu12r = up12(hr.data(), 2 * d.nf, H, true);
                    const Tens& wd = R(p + "mlp.down_proj.weight", {H, DFF});
                    std::vector<uint16_t> hd((size_t)H * d.nf);
                    for (int r = 0; r < H; r++) memcpy(hd.data() + (size_t)r * d.nf, (const uint16_t*)wd.p + (size_t)r * DFF + d.f0, (size_t)d.nf * 2);
                    d.Wdr = upraw2<bf16>(hd.data(), hd.size() * 2);
                }
            }
            const bool keepbf = getenv("INK_DR_BF16") != nullptr;
            d.Wqkv = keepbf ? cat({&R(p + "self_attn.q_proj.weight", {QD, H}), &R(p + "self_attn.k_proj.weight", {KVD, H}),
                                   &R(p + "self_attn.v_proj.weight", {KVD, H})})
                            : nullptr;
            d.Wkvc = cat({&R(p + "self_attn.k_proj_ctx.weight", {KVD, H}), &R(p + "self_attn.v_proj_ctx.weight", {KVD, H})});
            if (!d.Wor) d.Wo = upraw<bf16>(R(p + "self_attn.o_proj.weight", {H, QD}));   // [ink-tp-dra] split ranks: their columns only
            d.Wgu = keepbf ? cat({&R(p + "mlp.gate_proj.weight", {DFF, H}), &R(p + "mlp.up_proj.weight", {DFF, H})}) : nullptr;
            if (!d.Wdr) d.Wd = upraw<bf16>(R(p + "mlp.down_proj.weight", {H, DFF}));   // review5 M3: split ranks hold their columns only
        }
        d_mnorm = upraw<bf16>(R("markov_head.dflare_norm.weight", {H}));
        d_mask = upraw<bf16>(R("mask_embed", {H}));
        d_norm = upraw<bf16>(R("norm.weight", {H}));
        d_W1 = upraw<bf16>(R("markov_head.markov_w1.weight", {VPAD, MR}));
        d_Pb = upraw<bf16>(R("markov_head.rnn_proj.bias", {3 * MR}));
        {   // rnn_proj [768][5120] columns = [s 256 | e 256 | h 2304 | f 2304] -> three matrices
            const Tens& t = R("markov_head.rnn_proj.weight", {3 * MR, 2 * MR + 2 * H});
            const uint16_t* w = (const uint16_t*)t.p;
            const int K = 2 * MR + 2 * H;
            std::vector<uint16_t> a((size_t)3 * MR * 2 * MR), b((size_t)3 * MR * H), c((size_t)3 * MR * H);
            for (int r = 0; r < 3 * MR; r++) {
                memcpy(a.data() + (size_t)r * 2 * MR, w + (size_t)r * K, 2 * MR * 2);
                memcpy(b.data() + (size_t)r * H, w + (size_t)r * K + 2 * MR, H * 2);
                memcpy(c.data() + (size_t)r * H, w + (size_t)r * K + 2 * MR + H, H * 2);
            }
            d_Pse = upraw2<bf16>(a.data(), a.size() * 2);
            d_Ph = upraw2<bf16>(b.data(), b.size() * 2);
            d_Pf = upraw2<bf16>(c.data(), c.size() * 2);
        }
        {   // fuse weights: softmax over the taps per row
            const Tens& t = R("fuse_weight", {4, 4});
            for (int i = 0; i < 4; i++) {
                float v[4], m = -1e30f, sum = 0.f;
                for (int j = 0; j < 4; j++) m = std::max(m, v[j] = bf2f(((const uint16_t*)t.p)[i * 4 + j]));
                for (int j = 0; j < 4; j++) sum += (v[j] = expf(v[j] - m));
                for (int j = 0; j < 4; j++) d_fw[i * 4 + j] = v[j] / sum;
            }
        }
        {   // UO = U W_out [V][MR] (fp16), uOb = U b_out [V]
            const Tens& Ow = R("markov_head.hidden_out.weight", {H, MR});
            const Tens& Ob = R("markov_head.hidden_out.bias", {H});
            bf16* ow = upraw2<bf16>(Ow.p, Ow.n);
            bf16* ob = upraw2<bf16>(Ob.p, Ob.n);
            dr_exact = getenv("INK_DR_EXACT") && atoi(getenv("INK_DR_EXACT"));   // exact U h + fp16 UO (acceptance reference)
            ICHECK(!dr_exact || !tps, "TP: INK_DR_EXACT (the exact drafter head needs the whole vocabulary)");
            if (dr_exact) d_UO = (__half*)dmalloc((size_t)V * MR * 2, &dev_bytes);
            d_UO8 = (int8_t*)dmalloc((size_t)V * MR, &dev_bytes);
            d_us = (float*)dmalloc((size_t)V * 4, &dev_bytes);
            d_uOb = (float*)dmalloc((size_t)V * 4, &dev_bytes);
            const Tens& ue = f.get("model.llm.unembed.weight", "BF16", {VPAD, H});
            const int CK = 16384;
            bf16* uc = nullptr;
            float* yc = nullptr;
            CU(cudaMalloc(&uc, (size_t)CK * H * 2));
            CU(cudaMalloc(&yc, (size_t)CK * MR * 4));
            float mx = 0.f;
            for (int r0 = 0; r0 < V; r0 += CK) {
                const int n = std::min(CK, V - r0);
                CU(cudaMemcpy(uc, ue.p + (size_t)r0 * H * 2, (size_t)n * H * 2, cudaMemcpyHostToDevice));
                const float al = 1.f, be = 0.f;
                CB(cublasGemmEx(blas, CUBLAS_OP_N, CUBLAS_OP_N, MR, n, H, &al, ow, CUDA_R_16BF, MR, uc, CUDA_R_16BF, H, &be, yc,
                                CUDA_R_32F, MR, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
                CB(cublasGemmEx(blas, CUBLAS_OP_N, CUBLAS_OP_N, 1, n, H, &al, ob, CUDA_R_16BF, 1, uc, CUDA_R_16BF, H, &be,
                                d_uOb + r0, CUDA_R_32F, 1, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
                std::vector<float> h((size_t)n * MR);
                CU(cudaMemcpy(h.data(), yc, h.size() * 4, cudaMemcpyDeviceToHost));
                std::vector<__half> hh(h.size());
                std::vector<int8_t> q8(h.size());
                std::vector<float> qs(n);
                for (int r = 0; r < n; r++) {
                    float rm = 0.f;
                    for (int k = 0; k < MR; k++) rm = std::max(rm, fabsf(h[(size_t)r * MR + k]));
                    const float sc = rm > 0.f ? rm / 127.f : 1.f;
                    qs[r] = sc;
                    for (int k = 0; k < MR; k++) {
                        const float x = h[(size_t)r * MR + k];
                        mx = std::max(mx, fabsf(x));
                        hh[(size_t)r * MR + k] = __float2half(x);
                        q8[(size_t)r * MR + k] = (int8_t)lrintf(x / sc);
                    }
                }
                if (dr_exact) CU(cudaMemcpy(d_UO + (size_t)r0 * MR, hh.data(), hh.size() * 2, cudaMemcpyHostToDevice));
                CU(cudaMemcpy(d_UO8 + (size_t)r0 * MR, q8.data(), q8.size(), cudaMemcpyHostToDevice));
                CU(cudaMemcpy(d_us + r0, qs.data(), n * 4, cudaMemcpyHostToDevice));
            }
            ICHECK(mx < 60000.f, "UO out of fp16 range (%g)", mx);
            CU(cudaFree(uc));
            CU(cudaFree(yc));
            fprintf(stderr, "[ink] drafter: UO max |x| %.2f\n", mx);
        }
        fprintf(stderr, "[ink] drafter weights on GPU %.3f GB\n", (dev_bytes - b0) / 1e9);
    }
    // drafter K / V slots of layer i (tests: no ring wrap, anchors out of order); shared with the chunk reserve (review2 L4)
    int dr_cap(int i) const { return getenv("INK_DR_TEST") ? o.max_ctx + DB : i == 1 ? POOL + DB : RING; }   // [ink-srv-pool] global: the pool (+ a block)
    // [ink-srv-merge] device bytes of the drafter block buffers for n sequences (dr_alloc, without the chunk rows and
    // the K / V slots): the chunk reserve and fit_memory count n = min(S, 8) blocks (one block before the batch build)
    static size_t dr_blk_bytes(int n) {
        const size_t nbl = (V + MKL_ROWS - 1) / MKL_ROWS;
        size_t b = (size_t)DB * H * 4 + DB * H * 2 + (size_t)DB * DQKV * 4 + (size_t)DB * QD * 4 + 2 * (size_t)DB * NQ * DNS_GX * 4 +
                   (size_t)DB * NQ * DNS_GX * HD * 4 + (size_t)DB * QD * 2 + (size_t)DB * 2 * DFF * 4 + (size_t)DB * DFF * 2 + (size_t)DB * H * 2 +
                   H * 2 + (size_t)(DB - 1) * V * 4 + (size_t)(DB - 1) * 3 * MR * 4 + 3 * MR * 4 + 2 * (size_t)DB * MR * 4 + 4 * nbl * 4 +
                   (size_t)(DB - 1) * MKC_NC * 4 + 2 * (size_t)(DB - 1) * MKC_NCH * 4 + DB * 4 + 16 * 4 + H * 2 +
                   3 * (size_t)DB * H * 4;   // review5 M3: [ink-tp-drv2b] DPT (2 parities) + DSUM (TP ranks; an upper bound on one card)
        return (size_t)n * b;
    }
    // [ink-srv-merge] row-step buffers the batch build sizes for MROW rows where the prefill branch sized VR rows
    // (allocated after the chunk choice: attention partials of MROW x TB_NSX, head rows, loop messages, phase-A z
    // inputs, down partials, expert activations)
    static constexpr int PF_MIN_ROWS = 2048;   // prefill chunk the memory fit keeps room for
    // graph exec memory of the row-step / batched-drafter graphs (measured 11-23: bspec c = 8, 8 slots, ~0.35 GB above
    // the engine's allocations + context); one slot keeps the measured single-sequence headroom
    size_t srv_graph_room() const { return S > 1 ? (size_t)(0.35e9) : 0; }
    size_t batch_row_bytes() const {
        size_t b = (size_t)(MROW - VR) * V * 4 + (size_t)(MROW - VR) * 64 + (size_t)(MROW - VR) * 4 * (LOOP1 - LOOP0) * H * 4 +
                   (size_t)3 * MROW * 2 * H * 4 + (size_t)MROW * TOPK * H * 4 + ((size_t)1 << 20);
        if (!pf_psplit()) b += (size_t)(MROW * TB_NSX - VR * 128) * NQ * (2 + HD) * 4;
        if (pfx) b += (size_t)(MROW - VR) * TOPK * I * 4;
        return b;
    }
    void dr_alloc() {
        const size_t b0 = dev_bytes;
        NDS = std::min(S, 8);
        for (int i = 0; i < 3; i++) {
            dcap[i] = dr_cap(i);
            dkstr[i] = (size_t)NKV * dcap[i] * HD;
            dsstr[i] = (size_t)NKV * dcap[i];
            const size_t ns = dsko(i) == SKO_G ? 1 : S;   // [ink-srv-pool] the global layer: one pool
            DK[i] = (uint8_t*)dmalloc(ns * dkstr[i], &dev_bytes);
            DV[i] = (uint8_t*)dmalloc(ns * dkstr[i], &dev_bytes);
            DKS[i] = (float*)dmalloc(ns * dsstr[i] * 4, &dev_bytes);
            DVS[i] = (float*)dmalloc(ns * dsstr[i] * 4, &dev_bytes);
        }
        auto F = [&](size_t n) { return (float*)dmalloc(n * 4, &dev_bytes); };
        auto B = [&](size_t n) { return (bf16*)dmalloc(n * 2, &dev_bytes); };
        for (int i = 0; i < 3; i++) TAP[i] = F((size_t)ch * H);
        if (!pfx) {   // [ink-pf-dralias] with the prefill path: aliased into pBig (dead when dr_ctx runs), see alloc_state
            for (int i = 0; i < 4; i++) DF[i] = B((size_t)ch * H);
            DKVb = F((size_t)ch * 2 * KVD);
        }
        const int N = NDS;   // [ink-b-draft] block buffers for N sequences (segment z at z x the one-block size)
        Dh = F(N * DB * H); DX = B(N * DB * H); DQKVb = F((size_t)N * DB * DQKV); DQ = F((size_t)N * DB * QD);
        DPM = F(N * DB * NQ * DNS_GX); DPL = F(N * DB * NQ * DNS_GX); DPA = F((size_t)N * DB * NQ * DNS_GX * HD);
        if (tps) { DPT = F((size_t)2 * N * DB * H); DSUM = F((size_t)N * DB * H); }   // [ink-tp-drv2b] (DPT: 2 parities)
        DAO = B((size_t)N * DB * QD); DGU = F((size_t)N * DB * 2 * DFF); DA = B((size_t)N * DB * DFF); HIN = B(N * DB * H);
        FMS = B((size_t)S * H); FM = FMS; FMB = B(N * H);
        BASE = F((size_t)N * (DB - 1) * V); Zp = F(N * (DB - 1) * 3 * MR); Fz = F(N * 3 * MR); MS = F(N * DB * MR); MRO = F(N * DB * MR);
        nbl = (V + MKL_ROWS - 1) / MKL_ROWS;
        PMX = F(N * nbl); PSX = F(N * nbl); PIX = (int*)dmalloc(N * nbl * 4, &dev_bytes);
        MKCAND = (int*)dmalloc((size_t)N * (DB - 1) * MKC_NC * 4, &dev_bytes);
        MKTM = F(N * (DB - 1) * MKC_NCH); MKTS = F(N * (DB - 1) * MKC_NCH); MKPAX = F(N * nbl);
        MCNT = (unsigned*)dmalloc(64, &dev_bytes);
        CU(cudaMemset(MCNT, 0, 64));
        DTOK = (int*)dmalloc(N * DB * 4 + 64, &dev_bytes); DPROB = F(N * 16); dan = (int*)dmalloc(64, &dev_bytes);
        dsl = (int*)dmalloc(64, &dev_bytes);
        CU(cudaHostAlloc((void**)&hbd, 256, 0));
        DSTOP = F(4);
        set_stop(0.f, 0);
        demb_d = B(N * H);
        CU(cudaHostAlloc((void**)&demb_h, N * H * 2, 0));
        CU(cudaHostAlloc((void**)&han, 64, 0));
        if (getenv("INK_DR_TEST")) FMH = B((size_t)o.max_ctx * H);
        fprintf(stderr, "[ink] drafter state %.3f GB\n", (dev_bytes - b0) / 1e9);
    }
    bf16* FMH = nullptr;   // tests: f_3 of every position
    float* DSTOP = nullptr;   // [tau, kmin] of the Markov early stop (tau 0: all 7 drafts)
    float stop_h[2] = {-1.f, -1.f};
    void set_stop(float t, int km) {
        if (stop_h[0] == t && stop_h[1] == (float)km) return;
        stop_h[0] = t;
        stop_h[1] = (float)km;
        CU(cudaMemcpy(DSTOP, stop_h, 8, cudaMemcpyHostToDevice));
    }
    // context K / V of the T rows of the last forward (positions dpos ..); f_3 row *frow -> FM
    void dr_ctx(int T, const int* frow) {
        A_dfuse fa{};
        fa.tap[0] = TAP[0]; fa.tap[1] = TAP[1]; fa.tap[2] = TAP[2]; fa.tap[3] = FIN;
        memcpy(fa.fw, d_fw, sizeof d_fw);
        for (int i = 0; i < 3; i++) fa.nw[i] = DL[i].dn;
        fa.nw[3] = d_mnorm;
        for (int i = 0; i < 4; i++) fa.F[i] = DF[i];
        k_dfuse<<<dim3(T, 4), 256, 0, st>>>(fa);
        for (int i = 0; i < 3; i++) {
            gemm_b(DL[i].Wkvc, 2 * KVD, H, DF[i], T, DKVb, 2 * KVD, 0.f);
            if (i != 1 && T > dcap[i]) {   // [ink-pf-chunk] review H2: chunk > ring: only its last cap rows (no slot written twice)
                const int n = dcap[i];
                pf::k_addpos<<<1, 1, 0, st>>>(dpos, T - n, pDrPos);
                k_dctx<<<dim3(n, NKV), 128, 0, st>>>(A_dctx{DKVb + (size_t)(T - n) * 2 * KVD, DL[i].knc, dK(i), dV(i), dKS(i), dVS(i), dcap[i], 1, pDrPos});
            } else
                k_dctx<<<dim3(T, NKV), 128, 0, st>>>(A_dctx{DKVb, DL[i].knc, dK(i), dV(i), dKS(i), dVS(i), dcap[i], i != 1, dpos});
        }
        k_rowcopy<<<1, 256, 0, st>>>(DF[3], frow, 0, FM);
        if (FMH && !capturing) CU(cudaMemcpyAsync(FMH + (size_t)pos * H, DF[3], (size_t)T * H * 2, cudaMemcpyDeviceToDevice, st));
    }
    // the drafter block at anchor position *dan (host han[0]), anchor token DTOK[0] (embedding row staged in demb_h):
    // drafts -> DTOK[1..7], probabilities -> DPROB[1..7]
    // [ink-tp-drv2] TP: the drafter's base logits (8-bit screen), the Markov candidates and the per-position softmax over
    // this rank's 64-token chunks [drc0, drc1); per drafted position the ranks' parts are gathered (one row collective)
    // and merged identically on every rank: the same drafts on every rank and in real / virtual TP; equal to the one-card
    // drafter only with INK_DRMLP=0 and up to the stop decision (probabilities up to the rank-order sum; review5 L1).
    // INK_DRSPLIT=0: the replicated screen / Markov term.
    bool drs_on = !getenv("INK_DRSPLIT") || atoi(getenv("INK_DRSPLIT"));
    // [ink-tp-drv2b] the drafter MLP by rank: gate | up rows and down columns of [f0, f0 + nf); the down partial rows
    // summed in rank order (<= MROW rows per collective), then added to the residual. INK_DRMLP=0: replicated MLP.
    bool dms_on = !getenv("INK_DRMLP") || atoi(getenv("INK_DRMLP"));
    float *DPT = nullptr, *DSUM = nullptr;   // [NDS DB][H] partial / summed down rows
    int drs_idx = 1;                         // the drafter block's collective counter (from 1 after its k_tpbegin: review5 L2)
    int dpar = 0;                            // review5 M2: DPT buffer parity
    // review6 L3: drafter collectives skip slot 0 (the next sequence's collective 0 uses it; reuse distance 3 is safe:
    // a rank is at most one collective ahead of its peers)
    int drs_next() {
        if (drs_idx % TP_SLOTS == 0) drs_idx++;
        return drs_idx++;
    }
    bool drm_split() const { return DL[0].Wdr != nullptr; }   // review5 M3: the rank loaded its MLP slices only (fixed at load)
    // [ink-tp-dra] the drafter attention by heads: the rank's q heads [q0, q1) and KV heads [kv0, kv1) (the target's
    // planner shard: N > 8 computes a shared KV head on every rank that holds one of its q heads), q | k | v rows and
    // o_proj columns of those heads; the o_proj partial rows summed in rank order, then added to the residual. The
    // block's K / V are written for the rank's KV heads only (the caches keep the full layout; a rank reads only its
    // heads). INK_DRATT=0 (at start-up): the replicated attention.
    bool das_on = !getenv("INK_DRATT") || atoi(getenv("INK_DRATT"));
    bool dra_split() const { return DL[0].Wor != nullptr; }
    bool dr_coll() const { return dr_split() || drm_split() || dra_split(); }   // the drafter block has collectives
    void gemm_bx(const bf16* W, int N, int K, const bf16* X, int ldx, int T, float* Y, int ldy, float beta) {
        const float al = 1.f;
        CB(cublasGemmEx(blas, CUBLAS_OP_T, CUBLAS_OP_N, N, T, K, &al, W, CUDA_R_16BF, K, X, CUDA_R_16BF, ldx, &beta, Y,
                        CUDA_R_32F, ldy, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
    }
    // the rank's part of a drafter layer's attention (R block rows: DX -> residual Dh)
    void dra_attn(int i, const DLayer& d, int R, int ns, const std::function<void(const G12Args&, int, float*)>& g15,
                  const A_dqk& q0, const A_dattn& a0) {
        const int nq = sh.nq(), nk = sh.nkv();
        g15(d.Wqkv12r, (nq + 2 * nk) * HD, DQKVb);
        A_dqk q = q0;
        q.q0 = sh.q0;
        q.nq = nq;
        q.kv0 = sh.kv0;
        q.nkv = nk;
        k_dqk<<<dim3(R, nq + nk), 128, 0, st>>>(q);
        A_dattn da = a0;
        da.kv0 = sh.kv0;
        k_dattnL<<<dim3(nk, ns, R / 4), 128, 0, st>>>(da);
        k_dcomb<<<dim3(nq, R), 128, 0, st>>>(DPM, DPL, DPA, ns, DAO, sh.q0);
        float* dp = DPT + (size_t)(dpar++ & 1) * NDS * DB * H;   // review5 M2 parities (shared with drs_mlp)
        gemm_bx(d.Wor, H, nq * HD, DAO + (size_t)sh.q0 * HD, QD, R, dp, H, 0.f);
        for (int c0 = 0; c0 < R; c0 += MROW) tp_ar(drs_next(), dp + (size_t)c0 * H, std::min(MROW, R - c0), 0, DSUM + (size_t)c0 * H);
        k_addrows<<<(R * H / 4 + 255) / 256, 256, 0, st>>>((float4*)Dh, (const float4*)DSUM, R * H / 4);
        (void)i;
    }
    void drs_mlp(const DLayer& d, int R, const std::function<void(const G12Args&, int, float*)>& g15) {
        g15(d.Wgu12r, 2 * d.nf, DGU);
        k_dswiglu<<<(R * d.nf + 255) / 256, 256, 0, st>>>(DGU, DA, R, d.nf);
        // review5 M2: two partial buffers by layer parity (direct transport: a peer may still read layer L's rows while
        // this rank writes layer L + 1's; L + 2 comes after this rank waited on the peer's layer L + 1 events)
        float* dp = DPT + (size_t)(dpar++ & 1) * NDS * DB * H;
        gemm_b(d.Wdr, H, d.nf, DA, R, dp, H, 0.f);
        for (int c0 = 0; c0 < R; c0 += MROW) tp_ar(drs_next(), dp + (size_t)c0 * H, std::min(MROW, R - c0), 0, DSUM + (size_t)c0 * H);
        k_addrows<<<(R * H / 4 + 255) / 256, 256, 0, st>>>((float4*)Dh, (const float4*)DSUM, R * H / 4);
    }
    bool drs_plane = false;   // review6 M1: the rank loaded only its chunks of the screen plane (the split is then fixed)
    bool dr_split() const { return drs_plane || (tps && sh.N > 1 && drs_on && !dr_exact && ab != 52 && ab != 70); }
    int drc0() const { return (int)((long long)MKC_NCH * sh.rank / sh.N); }
    int drc1() const { return (int)((long long)MKC_NCH * (sh.rank + 1) / sh.N); }
    float *GATH = nullptr, *GATO = nullptr;   // [ink-tp-drv2] gather row (this rank's parts, zero elsewhere) / the sum
    void drs_begin() {   // at the head of a split drafter block: own collective step, gather row zeroed
        if (tpc->push) k_tpbegin<<<1, 1, 0, st>>>(tstep);
        CU(cudaMemsetAsync(GATH, 0, H * 4, st));
        drs_idx = 1;   // review5 L2: the block never ends on slot 0, where the next sequence starts
        dpar = 0;
    }
    void drs_screen(const bf16* x, int nr, float* y) {   // the rank's vocabulary rows of the base logits
        const int r0 = drc0() * MKC_CH, r1 = std::min(drc1() * MKC_CH, V), N = r1 - r0, nt = (nr + 7) / 8;
        const uint8_t *sc = dscr + (size_t)r0 * HEAD_K, *se = dseb + r0;
        float* yo = y + r0;
        const dim3 g((N + 127) / 128);
        if (nt == 1) k_screenTC<false, 1><<<g, 256, 0, st>>>(h2, sc, se, x, nr, yo, V, N);
        else if (nt == 2) k_screenTC<false, 2><<<g, 256, 0, st>>>(h2, sc, se, x, nr, yo, V, N);
        else if (nt == 3) k_screenTC<false, 3><<<g, 256, 0, st>>>(h2, sc, se, x, nr, yo, V, N);
        else if (nt == 4) k_screenTC<false, 4><<<g, 256, 0, st>>>(h2, sc, se, x, nr, yo, V, N);
        else if (nt == 5) k_screenTC<false, 5><<<g, 256, 0, st>>>(h2, sc, se, x, nr, yo, V, N);
        else if (nt == 6) k_screenTC<false, 6><<<g, 256, 0, st>>>(h2, sc, se, x, nr, yo, V, N);
        else if (nt == 7) k_screenTC<false, 7><<<g, 256, 0, st>>>(h2, sc, se, x, nr, yo, V, N);
        else k_screenTC<false, 8><<<g, 256, 0, st>>>(h2, sc, se, x, nr, yo, V, N);
    }
    A_mkl8C drs_mkl(const A_mkl8C& m, int j) const {
        A_mkl8C a = m;
        a.p0 = MKC_K * drc0();
        a.p1 = MKC_K * drc1();
        a.c0 = drc0();
        a.c1 = drc1();
        a.gat = GATH + (size_t)(j & 1) * drs_half();   // review5 M1
        a.gN = sh.N;
        a.grank = sh.rank;
        return a;
    }
    int drs_nblk() const { return (MKC_K * (drc1() - drc0()) + MKC_BL - 1) / MKC_BL; }
    // review5 M1: position j's parts in half (j & 1) of the gather row (direct transport: a peer may still read this
    // rank's position-j slots while its position j + 1 writes; j + 2 comes after this rank waited on the peer's j + 1)
    size_t drs_half() const { return H / 2; }
    void drs_merge(int j, int n) {
        const size_t o = (size_t)(j & 1) * drs_half();
        tp_ar(drs_next(), GATH, 1, 0, GATO);
        k_mkmerge<<<n, 32, 0, st>>>(GATO + o, sh.N, DTOK, DPROB, j, DSTOP);
        if (j == DB - 2) ICHECK((drs_idx - 1) % TP_SLOTS != 0, "drafter block ends on collective slot 0");   // review5 L2
    }
    void dr_block() {
        const bool spl = dr_split(), msp = drm_split(), asp = dra_split();
        if (spl || msp || asp) drs_begin();
        CU(cudaMemcpyAsync(dan, han, 4, cudaMemcpyHostToDevice, st));
        CU(cudaMemcpyAsync(DTOK, han + 1, 4, cudaMemcpyHostToDevice, st));
        CU(cudaMemcpyAsync(demb_d, demb_h, H * 2, cudaMemcpyHostToDevice, st));
        k_embnorm<<<1, 256, 0, st>>>(A_embnorm{demb_d, embn, Dh});
        k_dmask<<<((DB - 1) * H + 255) / 256, 256, 0, st>>>(d_mask, Dh);
        for (int i = 0; i < 3; i++) {
            const DLayer& d = DL[i];
            k_rmsnorm<<<DB, 256, 0, st>>>(A_rmsnorm{Dh, d.iln, DX, nullptr});
            auto g15 = [&](const G12Args& m, int N, float* y) {   // [ink-dr-g15] y[t][N] = W x_t, the 8 block rows
                G12Args g = m;
                g.a.x0 = g.a.x1 = g.a.x2 = DX;
                g.a.N = N;
                g.a.r1 = g.a.r2 = N;
                g.a.y = y;
                g.a.cst = nullptr;
                g.a.bmax = nullptr;
                k_g12tc<H, 0><<<(N / 2 + 7) / 8, 256 * tcr_S<H>, 0, st>>>(A_g12tc{g, DB, H, N, nullptr});
            };
            if (asp) {   // [ink-tp-dra]
                const int ns = i == 1 ? dns_g(pos) : DNS_LL;
                dra_attn(i, d, DB, ns, g15, A_dqk{DQKVb, d.qn, d.kn, DQ, dK(i), dV(i), dKS(i), dVS(i), dcap[i], i != 1, dan},
                         A_dattn{DQ, dK(i), dV(i), dKS(i), dVS(i), dcap[i], i != 1, i != 1, dan, DPM, DPL, DPA});
            } else {
            if (ab == 98 && d.Wqkv) gemm_b(d.Wqkv, DQKV, H, DX, DB, DQKVb, DQKV, 0.f);
            else g15(d.Wqkv12, DQKV, DQKVb);
            k_dqk<<<dim3(DB, NQ + NKV), 128, 0, st>>>(A_dqk{DQKVb, d.qn, d.kn, DQ, dK(i), dV(i), dKS(i), dVS(i), dcap[i], i != 1, dan});
            const int ns = i == 1 ? dns_g(pos) : ab == 63 ? DNS_L : DNS_LL;   // [ink-dr-attn-l]: 32 local splits
            const A_dattn da{DQ, dK(i), dV(i), dKS(i), dVS(i), dcap[i], i != 1, i != 1, dan, DPM, DPL, DPA};
            if (ab == 63) k_dattn<<<dim3(NQ, DB, ns), 128, 0, st>>>(da);   // A/B 63: one block per (head, row, split)
            else k_dattnL<<<dim3(NKV, ns, DB / 4), 128, 0, st>>>(da);   // [ink-dr-attn-l]
            k_dcomb<<<dim3(NQ, DB), 128, 0, st>>>(DPM, DPL, DPA, ns, DAO);
            gemm_b(d.Wo, H, QD, DAO, DB, Dh, H, 1.f);
            }
            k_rmsnorm<<<DB, 256, 0, st>>>(A_rmsnorm{Dh, d.pln, DX, nullptr});
            if (msp) drs_mlp(d, DB, g15);   // [ink-tp-drv2b]
            else {
            if (ab == 98 && d.Wgu) gemm_b(d.Wgu, 2 * DFF, H, DX, DB, DGU, 2 * DFF, 0.f);
            else g15(d.Wgu12, 2 * DFF, DGU);
            k_dswiglu<<<(DB * DFF + 255) / 256, 256, 0, st>>>(DGU, DA);
            gemm_b(d.Wd, H, DFF, DA, DB, Dh, H, 1.f);
            }
        }
        k_rmsnorm<<<DB - 1, 256, 0, st>>>(A_rmsnorm{Dh + H, d_norm, HIN, nullptr});
        if (!dr_exact)   // base logits U h of rows 1..7 from the 8-bit screen plane (drafts need no exact logits)
            // [ink-scrtc] tensor-core screen (drafter: 0.85 -> 0.58 ms per block); A/B 70: the CUDA-core k_screenR
            if (spl) drs_screen(HIN, DB - 1, BASE);
            else if (ab != 70) k_screenTC<false><<<(V + 127) / 128, 256, 0, st>>>(h2, dscr, dseb, HIN, DB - 1, BASE, V, V);
            else if (ab != 35) k_screenR<VR><<<(V / 2 + 7) / 8, 256, 0, st>>>(dscr, dseb, HIN, DB - 1, BASE, V);
            else k_screenS<DB - 1><<<82 * 4, 256, 0, st>>>(dscr, dseb, HIN, DB - 1, BASE, V);
        else {   // exact G15 rows
            G12Args ga = head12;
            ga.a.x0 = ga.a.x1 = ga.a.x2 = HIN;
            ga.a.N = V;
            ga.a.r1 = ga.a.r2 = V;
            ga.a.y = BASE;
            ga.a.cst = nullptr;
            ga.a.bmax = nullptr;
            k_gemv12R<H, VR, true><<<(V / 2 + 7) / 8, 256, 0, st>>>(A_gemv12R{ga, DB - 1, V});
        }
        gemm_b(d_Ph, 3 * MR, H, HIN, DB - 1, Zp, 3 * MR, 0.f);
        gemm_b(d_Pf, 3 * MR, H, FM, 1, Fz, 3 * MR, 0.f);
        k_zpre<<<((DB - 1) * 3 * MR + 255) / 256, 256, 0, st>>>(Zp, Fz, d_Pb);
        CU(cudaMemsetAsync(MS, 0, MR * 4, st));
        const bool topc = !dr_exact && ab != 52;   // [ink-dr-topc]; A/B 52: the full-vocabulary Markov term
        if (spl) k_mksel<<<dim3((drc1() - drc0() + 7) / 8, DB - 1), 256, 0, st>>>(BASE, d_uOb, MKCAND, MKTM, MKTS, drc0(), drc1());
        else if (topc) k_mksel<<<dim3((MKC_NCH + 7) / 8, DB - 1), 256, 0, st>>>(BASE, d_uOb, MKCAND, MKTM, MKTS);
        for (int j = 0; j < DB - 1; j++) {
            k_mkz<<<MR / 8, 256, 0, st>>>(A_mkz{d_Pse, Zp, d_W1, DTOK, MS, MRO, j, DSTOP, DPROB});
            const A_mkl ma{BASE, d_UO, d_uOb, MRO, PMX, PSX, PIX, MCNT, DTOK, DPROB, j, DSTOP};
            if (spl) {   // [ink-tp-drv2]
                k_mkl8C<<<drs_nblk(), 256, 0, st>>>(drs_mkl(A_mkl8C{ma, d_UO8, d_us, MKCAND, MKTM, MKTS, MKPAX}, j));
                drs_merge(j, 1);
            } else if (dr_exact) k_mkl<<<nbl, 256, 0, st>>>(ma);
            else if (topc) k_mkl8C<<<(MKC_NC + MKC_BL - 1) / MKC_BL, 256, 0, st>>>(A_mkl8C{ma, d_UO8, d_us, MKCAND, MKTM, MKTS, MKPAX});
            else k_mkl8<<<nbl, 256, 0, st>>>(A_mkl8{ma, d_UO8, d_us});
        }
        CU(cudaGetLastError());
    }
    size_t dko(int i) const { return dsko(i) == SKO_G ? (size_t)sbase[cur] : (size_t)cur * NKV * dcap[i]; }   // HD-rows
    uint8_t* dK(int i) const { return DK[i] + dko(i) * HD; }   // the selected slot's drafter caches
    uint8_t* dV(int i) const { return DV[i] + dko(i) * HD; }
    float* dKS(int i) const { return DKS[i] + dko(i); }
    float* dVS(int i) const { return DVS[i] + dko(i); }
    // [ink-b-draft] drafter context of a row step: every row's K / V into its slot's caches (positions / slots from
    // the row table), f_3 of each segment's last accepted row -> that slot's FM
    void dr_ctx_rows(int T) {
        A_dfuse fa{};
        fa.tap[0] = TAP[0]; fa.tap[1] = TAP[1]; fa.tap[2] = TAP[2]; fa.tap[3] = FIN;
        memcpy(fa.fw, d_fw, sizeof d_fw);
        for (int i = 0; i < 3; i++) fa.nw[i] = DL[i].dn;
        fa.nw[3] = d_mnorm;
        for (int i = 0; i < 4; i++) fa.F[i] = DF[i];
        k_dfuse<<<dim3(T, 4), 256, 0, st>>>(fa);
        for (int i = 0; i < 3; i++) {
            gemm_b(DL[i].Wkvc, 2 * KVD, H, DF[i], T, DKVb, 2 * KVD, 0.f);
            A_dctx c{DKVb, DL[i].knc, DK[i], DV[i], DKS[i], DVS[i], dcap[i], i != 1, dpos};
            c.rt = DRT;
            c.kstr = dkstr[i];
            c.sstr = dsstr[i];
            c.sko = dsko(i);
            k_dctx<<<dim3(T, NKV), 128, 0, st>>>(c);
        }
        k_fmcommit<<<nseg_h, 256, 0, st>>>(DF[3], DRT, ACC, FMS);
    }
    // one drafter block for n sequences (segment z = rows 8 z .. 8 z + 7; anchors / slots / tokens staged in hbd,
    // embedding rows in demb_h): drafts -> DTOK[8 z + 1 ..], probabilities -> DPROB[16 z + 1 ..]
    int dnr_cap = 0;
    void dr_block_rows(int n, int nsg) {
        const int R = n * DB;
        const bool spl = dr_split(), msp = drm_split(), asp = dra_split();   // [ink-tp-drv2]
        ICHECK(!spl || n * sh.N * 4 <= H / 2, "drafter gather row: %d segments x %d ranks", n, sh.N);
        if (spl || msp || asp) drs_begin();
        CU(cudaMemcpyAsync(dan, hbd, n * 4, cudaMemcpyHostToDevice, st));
        CU(cudaMemcpyAsync(dsl, hbd + 16, n * 4, cudaMemcpyHostToDevice, st));
        for (int z = 0; z < n; z++) CU(cudaMemcpyAsync(DTOK + z * DB, hbd + 32 + z, 4, cudaMemcpyHostToDevice, st));
        CU(cudaMemcpyAsync(demb_d, demb_h, (size_t)n * H * 2, cudaMemcpyHostToDevice, st));
        for (int z = 0; z < n; z++) k_embnorm<<<1, 256, 0, st>>>(A_embnorm{demb_d + (size_t)z * H, embn, Dh + (size_t)z * DB * H});
        k_dmask<<<dim3(((DB - 1) * H + 255) / 256, n), 256, 0, st>>>(d_mask, Dh);
        for (int i = 0; i < 3; i++) {
            const DLayer& d = DL[i];
            k_rmsnorm<<<R, 256, 0, st>>>(A_rmsnorm{Dh, d.iln, DX, nullptr});
            auto g15 = [&](const G12Args& m, int N, float* y) {
                G12Args g = m;
                g.a.x0 = g.a.x1 = g.a.x2 = DX;
                g.a.N = N;
                g.a.r1 = g.a.r2 = N;
                g.a.y = y;
                g.a.cst = nullptr;
                g.a.bmax = nullptr;
                static const int ntg = getenv("INK_NTG") ? atoi(getenv("INK_NTG")) : 4;   // [ink-tcr-dec1]
                if (R > 8 && ntg == 4) k_g12tc<H, 0, false, tcr_S<H>, 4><<<(N / 2 + 7) / 8, 256 * tcr_S<H>, 0, st>>>(A_g12tc{g, R, H, N, nullptr});
                else if (R > 8 && ntg == 2) k_g12tc<H, 0, false, tcr_S<H>, 2><<<(N / 2 + 7) / 8, 256 * tcr_S<H>, 0, st>>>(A_g12tc{g, R, H, N, nullptr});
                else k_g12tc<H, 0><<<(N / 2 + 7) / 8, 256 * tcr_S<H>, 0, st>>>(A_g12tc{g, R, H, N, nullptr});
            };
            A_dqk q{DQKVb, d.qn, d.kn, DQ, DK[i], DV[i], DKS[i], DVS[i], dcap[i], i != 1, dan};
            q.dsl = dsl;
            q.kstr = dkstr[i];
            q.sstr = dsstr[i];
            q.sko = dsko(i);
            const int ns = i == 1 ? nsg : DNS_LL;
            A_dattn da{DQ, DK[i], DV[i], DKS[i], DVS[i], dcap[i], i != 1, i != 1, dan, DPM, DPL, DPA};
            da.dsl = dsl;
            da.kstr = dkstr[i];
            da.sstr = dsstr[i];
            da.sko = dsko(i);
            if (asp) dra_attn(i, d, R, ns, g15, q, da);   // [ink-tp-dra]
            else {
                g15(d.Wqkv12, DQKV, DQKVb);
                k_dqk<<<dim3(R, NQ + NKV), 128, 0, st>>>(q);
                k_dattnL<<<dim3(NKV, ns, R / 4), 128, 0, st>>>(da);
                k_dcomb<<<dim3(NQ, R), 128, 0, st>>>(DPM, DPL, DPA, ns, DAO);
                gemm_b(d.Wo, H, QD, DAO, R, Dh, H, 1.f);
            }
            k_rmsnorm<<<R, 256, 0, st>>>(A_rmsnorm{Dh, d.pln, DX, nullptr});
            if (msp) drs_mlp(d, R, g15);   // [ink-tp-drv2b]
            else {
                g15(d.Wgu12, 2 * DFF, DGU);
                k_dswiglu<<<(R * DFF + 255) / 256, 256, 0, st>>>(DGU, DA, R);
                gemm_b(d.Wd, H, DFF, DA, R, Dh, H, 1.f);
            }
        }
        for (int z = 0; z < n; z++)
            k_rmsnorm<<<DB - 1, 256, 0, st>>>(A_rmsnorm{Dh + (size_t)(z * DB + 1) * H, d_norm, HIN + (size_t)z * (DB - 1) * H, nullptr});
        const int NR = n * (DB - 1);
        if (spl) {   // [ink-tp-drv2] the rank's vocabulary rows, <= 64 rows per launch
            static const int scrm = getenv("INK_SCR8") && !atoi(getenv("INK_SCR8")) ? 32 : 64;   // review5 L4
            for (int c0 = 0; c0 < NR; c0 += scrm) drs_screen(HIN + (size_t)c0 * H, std::min(scrm, NR - c0), BASE + (size_t)c0 * V);
        } else {
        // screen in <= 32-row chunks (NT <= 4 column tiles); [ink-dr-scr8] <= 64 rows (NT <= 8) in one launch: the 8-bit
        // plane read once for c = 5..8 (a column's arithmetic is the same in any tile count). INK_SCR8=0: 32-row chunks
        static const int scrmax = getenv("INK_SCR8") && !atoi(getenv("INK_SCR8")) ? 32 : 64;
        for (int c0 = 0; c0 < NR; c0 += scrmax) {
            const int nr = std::min(scrmax, NR - c0), nt = (nr + 7) / 8;
            const bf16* x = HIN + (size_t)c0 * H;
            float* y = BASE + (size_t)c0 * V;
            if (nt == 1) k_screenTC<false, 1><<<(V + 127) / 128, 256, 0, st>>>(h2, dscr, dseb, x, nr, y, V, V);
            else if (nt == 2) k_screenTC<false, 2><<<(V + 127) / 128, 256, 0, st>>>(h2, dscr, dseb, x, nr, y, V, V);
            else if (nt == 3) k_screenTC<false, 3><<<(V + 127) / 128, 256, 0, st>>>(h2, dscr, dseb, x, nr, y, V, V);
            else if (nt == 4) k_screenTC<false, 4><<<(V + 127) / 128, 256, 0, st>>>(h2, dscr, dseb, x, nr, y, V, V);
            else if (nt == 5) k_screenTC<false, 5><<<(V + 127) / 128, 256, 0, st>>>(h2, dscr, dseb, x, nr, y, V, V);
            else if (nt == 6) k_screenTC<false, 6><<<(V + 127) / 128, 256, 0, st>>>(h2, dscr, dseb, x, nr, y, V, V);
            else if (nt == 7) k_screenTC<false, 7><<<(V + 127) / 128, 256, 0, st>>>(h2, dscr, dseb, x, nr, y, V, V);
            else k_screenTC<false, 8><<<(V + 127) / 128, 256, 0, st>>>(h2, dscr, dseb, x, nr, y, V, V);
        }
        }
        gemm_b(d_Ph, 3 * MR, H, HIN, NR, Zp, 3 * MR, 0.f);
        k_fmgather<<<n, 256, 0, st>>>(FMS, dsl, FMB);
        gemm_b(d_Pf, 3 * MR, H, FMB, n, Fz, 3 * MR, 0.f);
        k_zpre<<<(NR * 3 * MR + 255) / 256, 256, 0, st>>>(Zp, Fz, d_Pb, n);
        CU(cudaMemsetAsync(MS, 0, (size_t)n * DB * MR * 4, st));
        if (spl) k_mksel<<<dim3((drc1() - drc0() + 7) / 8, NR), 256, 0, st>>>(BASE, d_uOb, MKCAND, MKTM, MKTS, drc0(), drc1());
        else k_mksel<<<dim3((MKC_NCH + 7) / 8, NR), 256, 0, st>>>(BASE, d_uOb, MKCAND, MKTM, MKTS);
        for (int j = 0; j < DB - 1; j++) {
            k_mkzZ<<<dim3(MR / 8, 1, n), 256, 0, st>>>(A_mkz{d_Pse, Zp, d_W1, DTOK, MS, MRO, j, DSTOP, DPROB});
            const A_mkl ma{BASE, d_UO, d_uOb, MRO, PMX, PSX, PIX, MCNT, DTOK, DPROB, j, DSTOP};
            if (spl) {   // [ink-tp-drv2]
                k_mkl8CZ<<<dim3(drs_nblk(), 1, n), 256, 0, st>>>(drs_mkl(A_mkl8C{ma, d_UO8, d_us, MKCAND, MKTM, MKTS, MKPAX}, j), nbl);
                drs_merge(j, n);
            } else
                k_mkl8CZ<<<dim3((MKC_NC + MKC_BL - 1) / MKC_BL, 1, n), 256, 0, st>>>(A_mkl8C{ma, d_UO8, d_us, MKCAND, MKTM, MKTS, MKPAX}, nbl);
        }
        CU(cudaGetLastError());
    }
    std::map<std::pair<int, int>, cudaGraphExec_t> dgraphR;   // batched drafter blocks by (n, global splits)
    void draft_rows_run(int n, int maxpos) {
        if (bs_dbg && (!tps || sh.rank == 0)) hts(6);
        const int nsg = dns_g(maxpos);
        if (dr_coll()) tp_tick();   // [ink-tp-drv2] a step-counter increment (tag wrap accounting)
        auto launch = [&] {
            dr_block_rows(n, nsg);
            CU(cudaMemcpyAsync(hdr, DTOK, (size_t)n * DB * 4, cudaMemcpyDeviceToHost, st));
            CU(cudaMemcpyAsync(hdp, DPROB, (size_t)n * 16 * 4, cudaMemcpyDeviceToHost, st));
        };
        if (getenv("INK_NOGRAPH") || (dr_coll() && !tpc->push)) launch();   // direct collectives: eager
        else {
            auto key = std::make_pair(n, nsg);
            auto it = dgraphR.find(key);
            if (it == dgraphR.end()) {
                cudaGraph_t g;
                capturing = true;
                CU(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal));
                launch();
                CU(cudaStreamEndCapture(st, &g));
                capturing = false;
                cudaGraphExec_t ge;
                if (ginst(&ge, g)) it = dgraphR.emplace(key, ge).first;
            }
            if (bs_dbg && (!tps || sh.rank == 0)) hts(5);
            if (it != dgraphR.end()) CU(cudaGraphLaunch(it->second, st));
            else launch();
        }
        CU(cudaStreamSynchronize(st));
        CU(cudaGetLastError());
    }
    void dr_stage(int ap, int anchor) {
        han[0] = ap;
        han[1] = anchor;
        memcpy(demb_h, emb_host + (size_t)anchor * H, H * 2);
    }
    void reset() {
        ring_finalize();
        pos = 0;
        plain_left = n_ds = since_tp = 0;   // [ink-dpol] a new context: costs re-measured from the next step on
        plain_run = 32;
        e_tp = e_rn = e_rt = 0;
        for (double& t : e_tv) t = 0;
        hist.clear();
        lk_map.clear();
        lk_done = 0;
        for (int a = 0; a < NAPP; a++) {
            CU(cudaMemsetAsync(A[a].kst, 0, 3 * KVD * 4, st));
            CU(cudaMemsetAsync(A[a].vst, 0, 3 * KVD * 4, st));
            CU(cudaMemsetAsync(A[a].ast, 0, 3 * H * 4, st));
            CU(cudaMemsetAsync(A[a].mst, 0, 3 * H * 4, st));
        }
    }
    void gemm(const bf16* W, int N, int K, const bf16* X, int T, float* Y, int ldy) {
        const float al = 1.f, be = 0.f;
        CB(cublasGemmEx(blas, CUBLAS_OP_T, CUBLAS_OP_N, N, T, K, &al, W, CUDA_R_16BF, K, X, CUDA_R_16BF, K, &be, Y,
                        CUDA_R_32F, ldy, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
    }
    void gemm2(const bf16* W, int N, int K, int ldw, const bf16* X, int ldx, int T, float* Y, int ldy) {   // [ink-tp-pf]
        const float al = 1.f, be = 0.f;
        CB(cublasGemmEx(blas, CUBLAS_OP_T, CUBLAS_OP_N, N, T, K, &al, W, CUDA_R_16BF, ldw, X, CUDA_R_16BF, ldx, &be, Y,
                        CUDA_R_32F, ldy, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
    }
    void bres(int T, std::initializer_list<const float*> s, std::initializer_list<const float*> q,
              std::initializer_list<float*> out) {
        BresArgs a{};
        a.ns = 0;
        for (auto p : s) a.s[a.ns++] = p;
        a.nq = 0;
        for (auto p : q) a.q[a.nq++] = p;
        int k = 0;
        for (auto p : out) a.out[k++] = p;
        bres_go(T, a);
    }
    // BF16 rows of a G15 matrix for the prefill GEMMs (the kept BF16 copy when INK_BF16KEEP=1)
    const bf16* dec16(const G12Args& m, int K, const bf16* keep) {
        if (keep) return keep;
        if (ws_tag == (const void*)m.sm) return WS;   // review2 L5: WS still holds this matrix (only dec16 writes WS; one stream)
        G12Args g = m;
        if (m.i8) k_dec8<<<(m.a.N / 2 + 7) / 8, 256, 0, st>>>(g, (uint32_t*)WS, K);   // [ink-pf8]
        else if (!m.idx && K == H && m.ks == 3) k_dec12T2<H, 3><<<(m.a.N / 2 + 7) / 8, 768, 0, st>>>(g, (uint32_t*)WS);   // [ink-vp-qs]
        else if (!m.idx && K == H) k_dec12T2<H><<<(m.a.N / 2 + 7) / 8, 256 * tcr_S<H>, 0, st>>>(g, (uint32_t*)WS);
        else if (!m.idx) k_dec12T2<QD><<<(m.a.N / 2 + 7) / 8, 256 * tcr_S<QD>, 0, st>>>(g, (uint32_t*)WS);
        else if (K == H) k_dec12<H><<<(m.a.N / 2 + 7) / 8, 256, 0, st>>>(g, (uint32_t*)WS);
        else k_dec12<QD><<<(m.a.N / 2 + 7) / 8, 256, 0, st>>>(g, (uint32_t*)WS);
        ws_tag = (const void*)m.sm;   // review3 L1: tag set after the decode launch
        return WS;
    }
    void ws_invalidate() { ws_tag = nullptr; }   // review3 L1: any other writer of WS must call this first
    template <class A>
    void go(void (*k)(A), int, dim3 g, int nt, const A& a) {
        k<<<g, nt, 0, st>>>(a);
    }
    void copy_go(float* dst, const float* src, size_t n) {
        CU(cudaMemcpyAsync(dst, src, n * 4, cudaMemcpyDeviceToDevice, st));
    }
    void bresZ_go(const BresArgs& a, int T, int gy) {   // A/B 134: the per-token loop forms
        if (ab == 134) k_bresZ<<<dim3((T + bres2_tt - 1) / bres2_tt, gy), 256, 0, st>>>(A_bres{a}, T, bres2_tt);
        else if (bres2_tt == 2) k_bresZ4<2><<<dim3((T + 1) / 2, gy), 256, 2 * 2 * H * 4, st>>>(A_bres{a}, T);
        else k_bresZ4<4><<<dim3((T + 3) / 4, gy), 256, 4 * 2 * H * 4, st>>>(A_bres{a}, T);
    }
    int brzl_min = getenv("INK_BRZL") ? atoi(getenv("INK_BRZL")) : 82;   // [ink-vp-brzl] z-form blocks above which the 2-block / SM form runs (9999: off)
    bool bres2(int T) const { return vR && T > 1 && T >= bres2_min && ab != 132; }
    void bres_go(dim3 g, const BresArgs& a0) {
        BresArgs a = a0;
        a.alg = 1;   // [ink-bres-gram] (A/B 47 below: the 768-thread form)
        a.chk = BCHK;
        const int nq = a.ident ? 1 : a.nq;
        ICHECK(a.ident || a.nq != 4 || (a.ns <= 2 && !a.nw[0]), "bres NQ 4 form: ns %d", a.ns);   // review L2
        static const bool nobresp = getenv("INK_NOBRESP") != nullptr;   // review3 L4
        if (ab != 47 && nq <= 3 && g.x > VR && g.y == 1 && !a.zg && !a.Wr && !nobresp) {   // [ink-pf-bresp] prefill rows (decode / verify keep k_bres256)
            if (nq == 1) go(k_bres256p<1>, 0, g, 256, A_bres{a});
            else go(k_bres256p<3>, 0, g, 256, A_bres{a});
            return;
        }
        if (ab != 47 && nq <= 3) {
            if (nq == 1) go(k_bres256<1>, 0, g, 256, A_bres{a});
            else go(k_bres256<3>, 0, g, 256, A_bres{a});
            return;
        }
        if (a.zg) {
            if (nq == 1) go(k_bres256<1>, 0, g, 256, A_bres{a});
            else go(k_bres256<3>, 0, g, 256, A_bres{a});
            return;
        }
        if (nq == 1) k_bres<1><<<g, BT, 0, st>>>({a});
        else if (nq == 3) k_bres<3><<<g, BT, 0, st>>>({a});
        else k_bres<4><<<g, BT, 0, st>>>({a});
    }
    // one layer application. hin: input (partial block sum); bs: block states before this layer (boundary layers
    // append hin for the MLP input); msg: messages (q, k, v, mlp) or null; hout: output.
    // K / V rows of a prefill application from P (conv taps, cache / staging writes, conv states)
    void kv_rows(const Layer& y, const AppState& s, int T) {
            const bool sg = pf_stage(s);   // sliding layers: the chunk's K / V go to the staging rows ([ink-pf-chunk])
            if (!nkvA()) return;   // [ink-tp-pf] a TP rank without K / V heads; else the rank's heads (kv0 offset)
            if (pcP(T, y)) {   // [ink-pf3-qkv] compact layout: the rank's K / V blocks (column = block + (h - kv0) x HD)
                const int k0 = kv0A();
                k_kv<<<dim3(T, nkvA()), 128, 0, st>>>(P, y.NPr, y.cK - k0 * HD, y.full ? y.cKt - k0 * 4 : -1, s.kst, y.kw, 1, sg ? pKs : s.K,
                                                   sg ? ch : s.cap, sg ? 0 : s.ring, sg ? pZero : dpos, k0);
                k_kv<<<dim3(T, nkvA()), 128, 0, st>>>(P, y.NPr, y.cV - k0 * HD, y.full ? y.cVt - k0 * 4 : -1, s.vst, y.vw, 0, sg ? pVs : s.V,
                                                   sg ? ch : s.cap, sg ? 0 : s.ring, sg ? pZero : dpos, k0);
                const int c0 = k0 * HD, nc = nkvA() * HD;
                k_cstate<<<(nc + 255) / 256, 256, 0, st>>>(P + y.cK, y.NPr, T, nc, s.kst + c0, KVD);
                k_cstate<<<(nc + 255) / 256, 256, 0, st>>>(P + y.cV, y.NPr, T, nc, s.vst + c0, KVD);
                return;
            }
            k_kv<<<dim3(T, nkvA()), 128, 0, st>>>(P, y.NP, y.offK, y.full ? y.offK + KVD : -1, s.kst, y.kw, 1, sg ? pKs : s.K, sg ? ch : s.cap,
                                               sg ? 0 : s.ring, sg ? pZero : dpos, kv0A());
            k_kv<<<dim3(T, nkvA()), 128, 0, st>>>(P, y.NP, y.offV, y.full ? y.offV + KVD : -1, s.vst, y.vw, 0, sg ? pVs : s.V, sg ? ch : s.cap,
                                               sg ? 0 : s.ring, sg ? pZero : dpos, kv0A());
            if (tps) {   // review tp2 L3: the rank's K / V head columns only (the others are not computed here)
                const int c0 = kv0A() * HD, nc = nkvA() * HD;
                k_cstate<<<(nc + 255) / 256, 256, 0, st>>>(P + y.offK + c0, y.NP, T, nc, s.kst + c0, KVD);
                k_cstate<<<(nc + 255) / 256, 256, 0, st>>>(P + y.offV + c0, y.NP, T, nc, s.vst + c0, KVD);
            } else {
                k_cstate<<<(KVD + 255) / 256, 256, 0, st>>>(P + y.offK, y.NP, T, KVD, s.kst);
                k_cstate<<<(KVD + 255) / 256, 256, 0, st>>>(P + y.offV, y.NP, T, KVD, s.vst);
            }
    }
    // [ink-pf-cut] last application when only the last row's output and the end conv states are needed (no drafter,
    // no hidden rows requested): K / V of every row by the full path's kernels; the query side (attention, output
    // projection, convs, MoE) for the last CUT_S rows only (cut 2) or not at all (cut 1: a chunk followed by >= 16
    // rows of the same prefill). The first 3 tail rows see a stale conv history (their outputs are unused); the last
    // row and the 3-row conv states come out right from CUT_S >= 7.
    static constexpr int CUT_S = 8;
    int cut = 0, cut_s = CUT_S;   // tail rows of cut 2 (CUT_S, or W with the drafter: [ink-pf-cutdr])
    bool kv_skip = false;
    int* pCutPos = nullptr;
    void layer_cut(int l, int T, const float* hin, const std::vector<const float*>& bs, float* hout) {
        Layer& y = L[l];
        AppState& s = A[app(l, 1)];
        ICHECK(y.full && l % BLOCK != 0 && l > 0, "layer_cut: layer %d", l);
        {   // attention inputs, as layer() (no messages after the loop)
            BresArgs a{};
            for (auto b : bs) a.s[a.ns++] = b;
            a.s[a.ns++] = hin;
            a.nq = 3;
            const bf16* nw[3] = {y.nq, y.nk, y.nv};
            for (int q = 0; q < 3; q++) {
                a.q[q] = y.aq + q * H;
                a.nw[q] = nw[q];
                a.outb[q] = XB[q];
            }
            bres_go(T, a);
        }
        const bf16* W = dec16(pf_src(y.qp8, y.qp12), H, y.Wp);   // [ink-1c8]
        gemm(W + (size_t)y.offK * H, y.offV - y.offK, H, XB[1], T, P + y.offK, y.NP);
        gemm(W + (size_t)y.offV * H, y.NP - y.offV, H, XB[2], T, P + y.offV, y.NP);
        kv_rows(y, s, T);
        // [ink-pf-cutdr] with the drafter, rows without the query side give dr_ctx the layer-22 residual stream as
        // their layer-23 output (drafter K / V only: acceptance, never the verified tokens)
        const int o = cut == 2 ? T - cut_s : T;
        if (dr_on && o > 0) CU(cudaMemcpyAsync(hout, hin, (size_t)o * H * 4, cudaMemcpyDeviceToDevice, st));
        if (cut != 2) return;
        std::vector<const float*> bo;
        for (auto b : bs) bo.push_back(b + (size_t)o * H);
        if (!pCutPos) pCutPos = (int*)dmalloc(64, &dev_bytes);
        int* dsave = dpos;
        pf::k_addpos<<<1, 1, 0, st>>>(dsave, o, pCutPos);
        dpos = pCutPos;
        pos += o;
        kv_skip = true;
        struct Restore {   // review2 L1
            Impl* p;
            int o;
            int* d;
            ~Restore() { p->kv_skip = false; p->pos -= o; p->dpos = d; }
        } rs{this, o, dsave};
        layer(l, 1, cut_s, hin + (size_t)o * H, bo, nullptr, hout + (size_t)o * H);
    }
    void layer(int l, int p, int T, const float* hin, std::vector<const float*> bs, float* const* msg, float* hout) {
        if (pf_half > 0) CU(cudaStreamWaitEvent(st, pfEv[pf_half - 1][app(l, p)], 0));   // [ink-tp-pf2] after sub-chunk j - 1's application
        if (sp_ok(T)) layer_sp(l, p, T, hin, bs, msg, hout);   // [ink-tp-sp]
        else layer_(l, p, T, hin, std::move(bs), msg, hout);
        if (pf_half >= 0 && pf_half < pfk - 1) CU(cudaEventRecord(pfEv[pf_half][app(l, p)], st));
    }
    void layer_(int l, int p, int T, const float* hin, std::vector<const float*> bs, float* const* msg, float* hout) {
        Layer& y = L[l];
        const bool dec = T == 1 || vR;   // decode form (one step, or the verify rows with the one-step arithmetic)
        const AppState& s = vR ? A0[app(l, p)] : A[app(l, p)];   // row steps: slot 0 base + the row table DRT
        const bool bnd = l % BLOCK == 0;
        const bool shared_in = !y.full && !msg;   // q, k, v from one normed input
        const bf16* nw[3] = {y.nq, y.full ? y.nk : y.nq, y.full ? y.nv : y.nq};
        // attention input(s) -> normed bf16 XB[0..2]
        if (spm & SP_IN) {   // [ink-tp-sp] stage mask (one-card / decode: all stages)
        if (l == 0) go(k_rmsnorm, 0, dim3(T), 256, A_rmsnorm{hin, y.nq, XB[0], nullptr});
        else {
            BresArgs a{};
            for (auto b : bs) a.s[a.ns++] = b;
            a.s[a.ns++] = hin;
            a.nq = y.full ? 3 : 1;
            for (int q = 0; q < a.nq; q++) {
                a.q[q] = y.aq + q * H;
                if (msg) a.out[q] = XA[q];
                else {
                    a.nw[q] = nw[q];
                    a.outb[q] = XB[q];
                }
            }
            const bool zf = msg && dec && ab != 13;   // message-gate z rows inside the residual kernel
            // [ink-audit-nq3split] full layers, decode / short row steps (the k_bres256 launches): one output per block
            static const bool nonq3 = getenv("INK_NONQ3") != nullptr;
            const bool q1 = a.nq == 3 && dec && T <= VR && ab != 47 && !nonq3 && !BCHK;   // review4 L3: the check sums every output
            if (zf) {
                a.zg = 3;
                for (int g = 0; g < 3; g++) {
                    a.mm[g] = msg[g];
                    a.gdn[g] = y.gd[g];
                    a.mgi[g] = g;
                }
                a.mni = MN + (size_t)(l - LOOP0) * T * 4;
                a.z = ZB;
                if (bres2(T)) {
                    a.zx = XZ;
                    bres_go(dim3(T, 1), a);
                    bresZ_go(a, T, 24);
                } else {
                    a.qone = q1;
                    if (T * 24 > brzl_min) {   // [ink-vp-brzl] more blocks than one wave of the early-load form
                        a.alg = 1;
                        a.chk = BCHK;
                        if (a.nq == 1) go(k_bres256z<1>, 0, dim3(T, 24), 256, A_bres{a});
                        else go(k_bres256z<3>, 0, dim3(T, 24), 256, A_bres{a});
                    } else bres_go(dim3(T, 24), a);
                }
            } else if (q1) {
                a.qone = 1;
                bres_go(dim3(T, 3), a);
            } else
                bres_go(T, a);
        }
        if (msg) {
            const bool zf = dec && ab != 13;
            MgArgs a{};
            Norm3 n{};
            for (int g = 0; g < 3; g++) {
                a.h[g] = y.full ? XA[g] : XA[0];
                a.m[g] = msg[g];
                a.down[g] = y.gd[g];
                a.f[g] = y.gf[g];
                a.g[g] = y.gg[g];
                a.out[g] = XG[g];
                n.x[g] = XG[g];
                n.w[g] = nw[g];
                n.o[g] = XB[g];
            }
            if (T > 1 && !vR && pfx && pfmg) pf_mg(a, 3, T);   // prefill: tensor-core message gates ([ink-pf-mg])
            else {
                if (!zf) go(k_mgz, 0, dim3(dim3(GR / 8, T, 3)), 256, A_mgz{a, ZB});
                if (dec && T > 1 && ab != 130) k_mgoT<4><<<dim3(H / 256, (T + 3) / 4, 3), 256, 0, st>>>(A_mgo{a, ZB}, T);   // A/B 130: k_mgo
                else go(k_mgo, 0, dim3(dim3(H / 256, T, 3)), 256, A_mgo{a, ZB});
            }
            go(k_rmsnorm3, 0, dim3(dim3(T, 3)), 256, A_rmsnorm3{n});
        }
        }   // SP_IN
        if (spm & SP_ATT) {
        if (dec) {   // one GEMV launch for all projections of the application
            G12Args ga = y.qp8.sm ? y.qp8 : y.qp12;   // [ink-n4-int8]
            GemvArgs& g = ga.a;
            g.W = y.Wp;
            g.x0 = XB[0];
            g.x1 = shared_in ? XB[0] : XB[1];
            g.x2 = shared_in ? XB[0] : XB[2];
            g.N = y.NPr;   // [ink-tp-dec1] the rank's rows (all: NP, offK, offV)
            g.r1 = y.qr1;
            g.r2 = y.qr2;
            g.y = P;
            // [ink-tcr] one tensor-core kernel for every row count (verify rows == plain bitwise)
            A_g12tc gt{ga, vR ? vR : 1, H, y.NP, nullptr};
            gt.rmap = y.qmap;
            if (y.NPr) qkv_gemv(gt, (y.NPr / 2 + 7) / 8, st);   // [ink-vp-qs] the planes' fragment order
        } else if (tps) {   // [ink-tp-pf] the rank's gathered rows: one GEMM per contiguous run into the full layout of P
            const bf16* W = y.NPr ? dec16(y.qp8.sm && !keep12 ? y.qp8 : y.qp12, H, nullptr) : nullptr;   // [ink-pf8]
            auto qrows = [&](int r0, int r1) {   // rows [r0, r1) of every run
                if (r1 > r0 && W && pcP(T, y)) {   // [ink-pf3-qkv] compact: [0, qr1) XB[0], [qr1, qr2) XB[1], [qr2, NPr) XB[2]
                    float* pc = P + (size_t)r0 * y.NPr;
                    if (shared_in) gemm(W, y.NPr, H, XB[0] + (size_t)r0 * H, r1 - r0, pc, y.NPr);
                    else {
                        const int b[4] = {0, y.qr1, y.qr2, y.NPr};
                        for (int g = 0; g < 3; g++)
                            if (b[g + 1] > b[g]) gemm(W + (size_t)b[g] * H, b[g + 1] - b[g], H, XB[g] + (size_t)r0 * H, r1 - r0, pc + b[g], y.NPr);
                    }
                } else if (r1 > r0 && W)
                    for (const auto& r : y.qruns)
                        gemm(W + (size_t)r[0] * H, r[2], H, XB[shared_in ? 0 : r[3]] + (size_t)r0 * H, r1 - r0, P + (size_t)r0 * y.NP + r[1], y.NP);
            };
            if (sp) {   // [ink-tp-sp] own rows while the peers' rows are in flight
                const int a = sp_a(T, sh.rank), b = sp_a(T, sh.rank + 1);
                qrows(a, b);
                tp_ag_wait();
                qrows(0, a);
                qrows(b, T);
            } else
                qrows(0, T);
        } else {
            const bf16* W = dec16(pf_src(y.qp8, y.qp12), H, y.Wp);   // prefill: BF16 for cuBLAS ([ink-1c8] from the INT8 planes)
            if (shared_in) gemm(W, y.NP, H, XB[0], T, P, y.NP);
            else {
                gemm(W, y.offK, H, XB[0], T, P, y.NP);
                if (!kv_skip) {   // [ink-pf-cut] tail rows: K / V already written by layer_kv
                    gemm(W + (size_t)y.offK * H, y.offV - y.offK, H, XB[1], T, P + y.offK, y.NP);
                    gemm(W + (size_t)y.offV * H, y.NP - y.offV, H, XB[2], T, P + y.offV, y.NP);
                }
            }
        }
        if (vR) {
            A_qkv1R qr{A_qkv1{P, y.NP, y.offK, y.offV, y.full, y.qs, y.rs, y.kw, y.vw, s.kst, s.vst, s.K, s.V, s.cap, s.ring, DRT, Qf, Rf, Sf},
                       s.kh, s.vh, s.ring ? SKO_R : SKO_G};
            if (tps) {   // [ink-tp-dec1] the rank's heads
                qr.a.q0 = sh.q0;
                qr.a.nq = sh.nq();
                qr.a.kv0 = sh.kv0;
                qr.a.nkv = sh.nkv();
            }
            if (qr.a.nq + qr.a.nkv) k_qkv1R<<<dim3(qr.a.nq + 2 * qr.a.nkv, vR), 128, 0, st>>>(qr);
        }
        else if (T == 1) {
            A_qkv1 q1{P, y.NP, y.offK, y.offV, y.full, y.qs, y.rs, y.kw, y.vw, s.kst, s.vst, s.K, s.V, s.cap, s.ring, dpos, Qf, Rf, Sf};
            if (tps) {   // [ink-tp-dec1] the rank's q heads and K / V heads
                q1.q0 = sh.q0;
                q1.nq = sh.nq();
                q1.kv0 = sh.kv0;
                q1.nkv = sh.nkv();
            }
            if (q1.nq + q1.nkv) go(k_qkv1, 0, dim3(q1.nq + 2 * q1.nkv), 128, q1);
        }
        else {
            if (!pqfuse(y, T) && (tps ? sh.nq() : NQ))   // [ink-pf-qfuse]; [ink-tp-pf] the rank's q heads
                k_qpost<<<dim3(T, tps ? sh.nq() : NQ), 128, 0, st>>>(P, pcP(T, y) ? y.NPr : y.NP, y.qs, y.rs, y.full, dpos, Qf, Rf, Sf, tps ? sh.q0 : 0,
                                                                   pcP(T, y) ? y.dq : 0, pcP(T, y) ? y.dr : 0, pcP(T, y) ? y.ds : 0);
            if (!kv_skip) kv_rows(y, s, T);
        }
        if (vR && tps && !sh.nkv()) {   // [ink-tp-dec1] a rank without attention heads
        } else if (vR) {   // per-row split counts = the one-step counts at each row's position; rows of one step may use both
            // forms (one launch each, the other form's rows exit at once): a row's form depends on its position only
            bool anyTb = false, allTb = true;
            for (int j = 0; j < vR; j++) {
                const bool t = tbOn(rpos_h[j], !y.full);
                anyTb |= t;
                allTb &= t;
            }
            if (!allTb) {   // the per-row k_attn_tc form (relative chunks)
                A_attn_tc at{Qf, Rf, y.proj, s.K, s.V, s.cap, s.ring, !y.full, DRT, PM, PL, PA, ACNT + NQ, Sf, P, AO, nullptr, ab != 22};
                at.sko = s.ring ? SKO_R : SKO_G;
                at.ldp = y.NP;
                int nsx = 1;
                for (int j = 0; j < vR; j++) {
                    at.nsv[j] = tbOn(rpos_h[j], !y.full) ? -1 : splits1(rpos_h[j], !y.full);
                    nsx = std::max(nsx, at.nsv[j]);
                }
                at.projT = y.projT;
                at.kv0 = kv0A();   // [ink-tp-dec1]
                ICHECK(!tps || ab != 81, "TP: attention A/B 81");
                go(ab == 81 ? k_attn_tc<true> : k_attn_tc3, 0, dim3(nkvA(), nsx, vR), 128, at);   // [ink-attn-occ3] (A/B 81: old form)
            }
            if (anyTb) {   // [ink-attn-tb] the rows' 3 q heads in shared m16 tiles (one sequence, one chunk grid), else a block row per token
                A_attn_tb at{Qf, Rf, y.proj, s.K, s.V, s.cap, s.ring, !y.full, DRT, PM, PL, PA, ACNT + NQ, Sf, P, AO, y.NP, vR, 5, 0, {}};
                at.sko = s.ring ? SKO_R : SKO_G;
                at.kv0 = kv0A();   // [ink-tp-dec1]
                ICHECK(!tps || ab != 125, "TP: attention A/B 125");
                bool same = true;
                int nsx = 1;
                for (int j = 0; j < vR; j++) {
                    const bool t = tbOn(rpos_h[j], !y.full);
                    at.Cv[j] = t ? tbC(rpos_h[j], !y.full) : 0;
                    same &= at.Cv[j] == at.Cv[0];
                    if (t) nsx = std::max(nsx, splits1(rpos_h[j], !y.full));
                }
                at.C = at.Cv[0];
                if (ab == 125) {   // A/B 125: the old k_attn_tb (shared tile only for one segment)
                    // shared tile from 2 rows of ONE sequence ([ink-attn-tb-r4] runs only at depth: k2 per-token blocks read K / V twice)
                    if (!same || vR == 1 || nseg_h > 1 || !allTb) at.tpt = 1;
                    go(at.tpt == 1 ? k_attn_tb<true> : k_attn_tb<false>, 0, dim3(NKV, 2 * nsx + 3, (vR + at.tpt - 1) / at.tpt), 128, at);
                } else {   // [ink-b-attn-tb] k_attn_tb3: tiles = runs of <= 5 rows of one segment with the same chunk keys
                    // [ink-audit-rowgraph] chunk keys + tiles of this step are data in the row table (tb_prep, copied with
                    // DRT); the launch shape (grid, form) is the graph key's
                    const TbShape& S = tbs[y.full ? 1 : 0];
                    at.projT = y.projT;
                    at.tbo = RT_TB + 4 * MROW * (y.full ? 1 : 0);
                    const int ny = S.ny;
                    if (S.one) go(ny <= 128 && ab != 129 ? k_attn_tb3<true, 4> : k_attn_tb3<true>, 0, dim3(nkvA(), ny, vR), 128, at);   // A/B 129: 9-lane merge
                    else go(ny <= 128 && ab != 129 ? k_attn_tb3<false, 4> : k_attn_tb3<false>, 0, dim3(nkvA(), ny, S.nz), 128, at);
                }
            }
        } else {
            const int ns = attn_splits(T, !y.full);
            ICHECK(!tps || ((T == 1 || pfx) && (ab == 0 || ab == 80)), "TP: attention form (T %d, A/B %d)", T, ab);
            if (tps && !sh.nkv()) {   // [ink-tp-dec1] a rank without attention heads
            } else if (T == 1 && ab != 15 && ab != 1 && tbOn(pos, !y.full)) {   // [ink-attn-tb] (A/B 80 / short: k_attn_tc below)
                A_attn_tb at{Qf, Rf, y.proj, s.K, s.V, s.cap, s.ring, !y.full, dpos, PM, PL, PA, ACNT + NQ, Sf, P, AO, y.NP, 1, 1, tbC(pos, !y.full), {}};
                at.Cv[0] = at.C;
                at.projT = y.projT;
                at.kv0 = kv0A();   // [ink-tp-dec1]
                if (ab == 125) go(k_attn_tb<true>, 0, dim3(NKV, 2 * splits1(pos, !y.full) + 3, 1), 128, at);   // A/B 125: old kernel
                else {
                    const int ny = tbY(splits1(pos, !y.full), at.C, !y.full);
                    go(ny <= 128 && ab != 129 ? k_attn_tb3<true, 4> : k_attn_tb3<true>, 0, dim3(nkvA(), ny, 1), 128, at);
                }
            } else if (T == 1 && ab != 15 && ab != 1) {   // tensor-core decode attention (A/B 15: the CUDA-core kernel)
                A_attn_tc at{Qf, Rf, y.proj, s.K, s.V, s.cap, s.ring, !y.full, dpos, PM, PL, PA, ACNT + NQ, Sf, P, AO, nullptr, ab != 22, 0, {}, y.projT};
                at.kv0 = kv0A();   // [ink-tp-dec1]
                go(ab == 18 ? k_attn_tc<false> : ab == 21 ? k_attn_tc<false, true> : ab == 81 ? k_attn_tc<true> : k_attn_tc3, 0,   // FP16 P (set-2 decode KL 0.00461 vs BF16 0.00497); 18 = BF16 P
                   dim3(nkvA(), ns), 128, at);
            }
            else if (T == 1 && ab != 1)   // merge in the last split block
                go(k_attn_part, 0, dim3(dim3(T, NQ, ns)), 128,
                   A_attn_part{Qf, Rf, y.proj, s.K, s.V, s.cap, s.ring, !y.full, dpos, PM, PL, PA, ACNT, Sf, P, y.NP, AO, ab != 14});
            else if (T > 1 && pfx && pfat) pf_attn(y, s, l, T);   // prefill: tensor-core flash attention ([ink-pf-attn])
            else {
                go(k_attn_part, 0, dim3(dim3(T, NQ, ns)), 128, A_attn_part{Qf, Rf, y.proj, s.K, s.V, s.cap, s.ring, !y.full, dpos, PM, PL, PA});
                go(k_attn_comb, 0, dim3(dim3(T, NQ)), 128, A_attn_comb{PM, PL, PA, ns, Sf, P, y.NP, AO});
            }
            if (T == 1 && cmp_on && !capturing) {   // precision check: each tensor-core form vs the CUDA-core kernel (fp32 Q, P)
                static float* buf = nullptr;
                static bf16* aob = nullptr;
                if (!buf) {
                    CU(cudaMalloc(&buf, 7 * QD * 4));
                    CU(cudaMalloc(&aob, QD * 2));
                }
                go(k_attn_part, 0, dim3(dim3(T, NQ, ns)), 128,
                   A_attn_part{Qf, Rf, y.proj, s.K, s.V, s.cap, s.ring, !y.full, dpos, PM, PL, PA, ACNT, Sf, P, y.NP, aob, 1, buf});
                const A_attn_tc at{Qf, Rf, y.proj, s.K, s.V, s.cap, s.ring, !y.full, dpos, PM, PL, PA, ACNT + NQ, Sf, P, aob, nullptr};
                A_attn_tc a1 = at, a2 = at, a3 = at;
                a1.AOf = buf + QD;
                a2.AOf = buf + 2 * QD;
                a3.AOf = buf + 3 * QD;
                go(k_attn_tc<false>, 0, dim3(NKV, ns), 128, a1);
                go(k_attn_tc<true>, 0, dim3(NKV, ns), 128, a2);
                go(k_attn_tc<false, true>, 0, dim3(NKV, ns), 128, a3);
                {   // [ink-attn-occ3] bitwise: k_attn_tc3 == k_attn_tc<true>
                    A_attn_tc a4 = at;
                    a4.projT = y.projT;
                    a4.AOf = buf + 5 * QD;
                    go(k_attn_tc3, 0, dim3(NKV, ns), 128, a4);
                    std::vector<float> h2(2 * QD);
                    CU(cudaMemcpyAsync(h2.data(), buf + 2 * QD, QD * 4, cudaMemcpyDeviceToHost, st));
                    CU(cudaMemcpyAsync(h2.data() + QD, buf + 5 * QD, QD * 4, cudaMemcpyDeviceToHost, st));
                    CU(cudaStreamSynchronize(st));
                    cmp3_n += QD;
                    for (int i = 0; i < QD; i++) cmp3_bad += memcmp(&h2[i], &h2[QD + i], 4) != 0;
                }
                {   // [ink-attn-tb] the shared-row form at this position (any depth), plain step; local layers too ([ink-b-attn-tb])
                    const bool lo_ = !y.full;
                    A_attn_tb tb{Qf, Rf, y.proj, s.K, s.V, s.cap, s.ring, lo_, dpos, PM, PL, PA, ACNT + NQ, Sf, P, aob, y.NP, 1, 1, tbC(pos, lo_), {}};
                    tb.Cv[0] = tb.C;
                    tb.AOf = buf + 4 * QD;
                    go(k_attn_tb<true>, 0, dim3(NKV, 2 * splits1(pos, lo_) + 3, 1), 128, tb);
                    A_attn_tb t3 = tb;   // [ink-b-attn-tb] bitwise: k_attn_tb3 == k_attn_tb
                    t3.projT = y.projT;
                    t3.AOf = buf + 6 * QD;
                    go(k_attn_tb3<true>, 0, dim3(NKV, 2 * splits1(pos, lo_) + 3, 1), 128, t3);
                    std::vector<float> h3(2 * QD);
                    CU(cudaMemcpyAsync(h3.data(), buf + 4 * QD, QD * 4, cudaMemcpyDeviceToHost, st));
                    CU(cudaMemcpyAsync(h3.data() + QD, buf + 6 * QD, QD * 4, cudaMemcpyDeviceToHost, st));
                    CU(cudaStreamSynchronize(st));
                    cmptb3_n += QD;
                    for (int i = 0; i < QD; i++) cmptb3_bad += memcmp(&h3[i], &h3[QD + i], 4) != 0;
                }
                std::vector<float> hb(5 * QD);
                CU(cudaMemcpyAsync(hb.data(), buf, hb.size() * 4, cudaMemcpyDeviceToHost, st));
                CU(cudaStreamSynchronize(st));
                for (int v = 0; v < 3; v++)
                    for (int i = 0; i < QD; i++) {
                        const double d = hb[(v + 1) * QD + i] - hb[i];
                        cmp_err[v] += d * d;
                        if (v == 0) cmp_ref += (double)hb[i] * hb[i];
                    }
                for (int v = 0; v < 2; v++) {
                        const float* x = hb.data() + (v == 0 ? 2 : 4) * QD;
                        double e2 = 0, r2 = 0, md = 0, mr = 0;
                        for (int i = 0; i < QD; i++) {
                            const double d = (double)x[i] - hb[i];
                            e2 += d * d;
                            r2 += (double)hb[i] * hb[i];
                            md = std::max(md, fabs(d));
                            mr = std::max(mr, (double)fabsf(hb[i]));
                        }
                        cmp_g[v + 2 * !y.full].push_back(sqrt(e2 / std::max(r2, 1e-30)));
                        cmp_gm[v + 2 * !y.full].push_back(md / std::max(mr, 1e-30));
                    }
            }
        }
        }   // SP_ATT (the wo partial of a TP prefill below)
        const int nb = (T * H + 255) / 256;
        // output projection, post-attention conv, residual
        if (vR && tps) {   // [ink-tp-dec1] row partials -> sum -> the verify conv chain (k_g12tc MODE 2 epilogue) after it
            const TpCol pc = tp_col_args(2 * app(l, p), 0, vR, ATT, s.ast, y.aw, bnd ? nullptr : hin, Rr, s.ah);   // [ink-vp-cpush]
            if (sh.ce > sh.cb && pc.np) {
                G12Args ga = y.wo8.sm ? y.wo8 : y.woR;   // [ink-n4-int8]
                ga.a.x0 = ga.a.x1 = ga.a.x2 = AO;
                ga.a.N = H;
                ga.a.y = WOP;
                (ga.i8 ? k_g12tcR<1, true> : k_g12tcR<1>)<<<H / 16, 256, 0, st>>>(A_g12tcR{ga, vR, QD, H, sh.cb, sh.ce - sh.cb, sh.q0 * HD, sh.q1 * HD, {}, pc});
                tpc_check(vR, ATT, s.ast, y.aw, bnd ? nullptr : hin, Rr, s.ah);
            } else if (sh.ce > sh.cb) {
                G12Args ga = y.wo8.sm ? y.wo8 : y.woR;   // [ink-n4-int8]
                ga.a.x0 = ga.a.x1 = ga.a.x2 = AO;
                ga.a.N = H;
                ga.a.y = WOP;
                // [ink-tcr-dec1] review4 M2: NTG 4 spills here (128 regs, stack 24) but measures faster than NTG 2 (TP2 24 rows
                // 25.59 vs 25.80 ms, 3 pairs); INK_NTGW = tiles per decode of the wo forms (1 / 2 / 4)
                static const int ntgR = getenv("INK_NTGW") ? atoi(getenv("INK_NTGW")) : 4;
                if (vR > 8 && ntgR == 2) (ga.i8 ? k_g12tcR<2, true> : k_g12tcR<2>)<<<H / 16, 256, 0, st>>>(A_g12tcR{ga, vR, QD, H, sh.cb, sh.ce - sh.cb, sh.q0 * HD, sh.q1 * HD});
                else if (vR > 8 && ntgR == 4) (ga.i8 ? k_g12tcR<4, true> : k_g12tcR<4>)<<<H / 16, 256, 0, st>>>(A_g12tcR{ga, vR, QD, H, sh.cb, sh.ce - sh.cb, sh.q0 * HD, sh.q1 * HD});
                else (ga.i8 ? k_g12tcR<1, true> : k_g12tcR<1>)<<<H / 16, 256, 0, st>>>(A_g12tcR{ga, vR, QD, H, sh.cb, sh.ce - sh.cb, sh.q0 * HD, sh.q1 * HD});
            }
            if (!pc.np && !tp_arC(2 * app(l, p), WOP, vR, ATT, s.ast, y.aw, bnd ? nullptr : hin, Rr, s.ah)) {
                tp_ar(2 * app(l, p), WOP, vR, 0, ATT);
                k_tpchain<<<dim3((H + 255) / 256, convseg ? nseg_h : 1), 256, 0, st>>>(ATT, vR, s.ast, y.aw, bnd ? nullptr : hin, Rr, s.ah, DRT);
            }
        } else if (vR) {
            G12Args ga = y.wo8.sm ? y.wo8 : y.wo12;   // [ink-1c8]
            GemvArgs& g = ga.a;
            g.x0 = g.x1 = g.x2 = AO;
            g.N = H;
            g.r1 = g.r2 = H;
            g.y = Rr;
            g.cst = s.ast;
            g.cw = y.aw;
            g.add = bnd ? nullptr : hin;
            wo_gemv1(ga, vR, s.ah, DRT, st);
        } else if (T == 1 && tps) {   // [ink-tp-dec1] the rank's columns -> partial rows; the sum, conv and residual after it
            if (sh.ce > sh.cb) {
                G12Args ga = y.wo8.sm ? y.wo8 : y.woR;   // [ink-n4-int8]
                ga.a.x0 = ga.a.x1 = ga.a.x2 = AO;
                ga.a.N = H;
                ga.a.y = WOP;
                (ga.i8 ? k_g12tcR<1, true> : k_g12tcR<1>)<<<H / 16, 256, 0, st>>>(A_g12tcR{ga, 1, QD, H, sh.cb, sh.ce - sh.cb, sh.q0 * HD, sh.q1 * HD, tp_push_args(2 * app(l, p), 0, Rr, s.ast, y.aw, bnd ? nullptr : hin)});
            }
            tp_ar1(2 * app(l, p), 0, WOP, Rr, s.ast, y.aw, bnd ? nullptr : hin);
        } else if (T == 1) {
            G12Args ga = y.wo8.sm ? y.wo8 : y.wo12;   // [ink-1c8]
            GemvArgs& g = ga.a;
            g.W = y.Wo;
            g.x0 = g.x1 = g.x2 = AO;
            g.N = H;
            g.r1 = g.r2 = H;
            g.y = Rr;
            g.cst = s.ast;
            g.cw = y.aw;
            g.add = bnd ? nullptr : hin;
            wo_gemv1(ga, 0, nullptr, nullptr, st);
        } else if (tps) {   // [ink-tp-pf] the rank's wo columns -> partial rows -> sum over the ranks -> conv + residual
            const int c0 = sh.cb * 256, c1 = sh.ce * 256;
            if (spm & SP_ATT) {
                if (c1 > c0 && (c0 < sh.q0 * HD || c1 > sh.q1 * HD)) k_tpzcols<<<T, 256, 0, st>>>(AO, QD, T, c0, c1, sh.q0 * HD, sh.q1 * HD);
                const bf16* WoB = c1 > c0 ? (y.woB ? y.woB : dec16(y.wo8, c1 - c0, nullptr)) : nullptr;   // [ink-pf8-wob]
                auto worows = [&](int r0, int r1) {
                    if (r1 <= r0) return;
                    if (c1 > c0) gemm2(WoB, H, c1 - c0, c1 - c0, AO + (size_t)r0 * QD + c0, QD, r1 - r0, tpP + (size_t)r0 * H, H);
                    else CU(cudaMemsetAsync(tpP + (size_t)r0 * H, 0, (size_t)(r1 - r0) * H * 4, st));
                };
                if (sp) {   // [ink-tp-sp] the rows the peers need, send, then the private rows
                    int p0, p1;
                    sp_priv(T, &p0, &p1);
                    worows(0, p0);
                    worows(p1, T);
                    tp_rs_send(T);
                    worows(p0, p1);
                } else
                    worows(0, T);
            }
            if (!sp) {
                tp_arP(T, ATT);
                k_conv<<<nb, 256, 0, st>>>(ATT, T, H, s.ast, y.aw, bnd ? nullptr : hin, Rr);
                k_cstate<<<(H + 255) / 256, 256, 0, st>>>(ATT, H, T, H, s.ast);
            } else if (spm & SP_C1)   // [ink-tp-sp] the rank's rows; history = the halo rows above them (or the state)
                k_conv<<<nb, 256, 0, st>>>(ATT, T, H, sp_h0 ? s.ast : ATT - 3 * H, y.aw, bnd ? nullptr : hin, Rr);
        } else {
            gemm(dec16(pf_src(y.wo8, y.wo12), QD, y.Wo), H, QD, AO, T, ATT, H);
            k_conv<<<nb, 256, 0, st>>>(ATT, T, H, s.ast, y.aw, bnd ? nullptr : hin, Rr);
            k_cstate<<<(H + 255) / 256, 256, 0, st>>>(ATT, H, T, H, s.ast);
        }
        // MLP input -> (message gate) -> norm -> fp32 XMF + router scores SC (folded into the residual kernel)
        if (spm & SP_MIN) {
            bool mgor = false;   // [ink-audit-mgor] the router ran in the gate-out kernels
            BresArgs a{};
            for (auto b : bs) a.s[a.ns++] = b;
            if (bnd) a.s[a.ns++] = hin;
            a.s[a.ns++] = Rr;
            a.nq = 1;
            a.q[0] = y.mq;
            if (msg) {
                a.out[0] = MI;
                const bool zf = dec && ab != 13;
                if (zf) {
                    a.zg = 1;
                    a.mni = MN + (size_t)(l - LOOP0) * T * 4;
                    a.mgi[0] = 3;
                    a.mm[0] = msg[3];
                    a.gdn[0] = y.gd[3];
                    a.z = ZB;
                    if (bres2(T)) {
                        a.zx = XZ;
                        bres_go(dim3(T, 1), a);
                        bresZ_go(a, T, 8);
                    } else
                        bres_go(dim3(T, 8), a);
                } else
                    bres_go(T, a);
                MgArgs g{};
                g.h[0] = MI;
                g.m[0] = msg[3];
                g.down[0] = y.gd[3];
                g.f[0] = y.gf[3];
                g.g[0] = y.gg[3];
                g.out[0] = MIG;
                // [ink-audit-mgor] decode / row steps: norm + router + top-8 from the gate-out kernel's blocks
                static const bool nomgor = getenv("INK_NOMGOR") != nullptr;
                RouteF rf{};
                if (zf && dec && ab != 10 && !nomgor) {
                    rf = RouteF{y.Wr, y.nm, y.rb, RFP, RFS, RCNT, MIG, XMF, SC, IDS, WTS};
                    mgor = true;
                }
                if (T > 1 && !vR && pfx && pfmg) pf_mg(g, 1, T);   // prefill: tensor-core message gates ([ink-pf-mg])
                else {
                    if (!zf) go(k_mgz, 0, dim3(dim3(GR / 8, T, 1)), 256, A_mgz{g, ZB});
                    if (dec && T > 1 && ab != 130) {
                        k_mgoT<4><<<dim3(H / 256, (T + 3) / 4, 1), 256, 0, st>>>(A_mgo{g, ZB}, T);
                        if (rf.Wr) k_rpart<<<dim3(RF_NB, T), 256, 0, st>>>(rf);
                    } else go(k_mgo, 0, dim3(dim3(H / 256, T, 1)), 256, A_mgo{g, ZB, rf});
                }
                a = BresArgs{};
                a.s[0] = MIG;
                a.ns = 1;
                a.ident = 1;
            }
            if (!mgor) {
            a.nw[0] = y.nm;
            a.normf = 1;
            a.out[0] = XMF;
            a.Wr = y.Wr;
            a.sc = SC;
            if (dec && ab != 10) {
                a.rids = IDS;
                a.rwts = WTS;
                a.rbias = y.rb;
                a.rcnt = RCNT;
            }
            if (vR && vR >= rpw_min) a.rpw = 2;   // review r5 L6: verify rows only
            if (T > 1 && !vR && pfx) {   // prefill: residual + norm once per row, router scores on tensor cores ([ink-pf-router])
                a.Wr = nullptr;
                bres_go(T, a);
                pf::k_prouter<pf::PR_NEB><<<dim3((T + 16 * pf::PR_W - 1) / (16 * pf::PR_W), NE / pf::PR_NEB), 32 * pf::PR_W, 0, st>>>(XMF, y.Wr, T, SC, (pfuse && !getenv("INK_NOF2H")) ? pX16 : nullptr);
            } else if (bres2(T)) {   // [ink-b-bres2] phase A (norm -> XMF), phase B router dots + top-8 per token tile
                BresArgs b = a;
                b.Wr = nullptr;
                b.rids = nullptr;
                bres_go(dim3(T, 1), b);
                if (ab == 133 || !a.rids)   // A/B 133: top-8 in the last block of a token tile
                    k_bresR<<<dim3((T + bres2_ttr - 1) / bres2_ttr, NE / 8), 256, 0, st>>>(A_bres{a}, XMF, T, bres2_ttr);
                else {
                    BresArgs c = a;
                    c.rids = nullptr;
                    if (ab == 134) k_bresR<<<dim3((T + bres2_ttr - 1) / bres2_ttr, NE / 8), 256, 0, st>>>(A_bres{c}, XMF, T, bres2_ttr);
                    else if (bres2_ttr == 2) k_bresR4<2><<<dim3((T + 1) / 2, NE / 8), 256, 2 * H * 4, st>>>(A_bres{c}, XMF, T);
                    else k_bresR4<4><<<dim3((T + 3) / 4, NE / 8), 256, 4 * H * 4, st>>>(A_bres{c}, XMF, T);
                    k_topkR<<<T, 256, 0, st>>>(a.sc, a.rbias, a.rids, a.rwts);
                }
            } else
                bres_go(dim3(T, NE / RPB / (a.rpw == 2 ? 2 : 1)), a);
            }   // !mgor
        }
        float* cst = (T == 1 && !vR) ? s.mst : nullptr;
        float* mo = (T == 1 && !vR) ? hout : MO;
        // [ink-tp-tcx] row steps from tcx_min rows: 0 = today's kernels, 1 = TC gate / up (fp32 act for today's down),
        // 2 = TC gate / up + TC down (act planes; per-entry partials + k_dsum)
        int tcx = (tcx_min > 0 && vR >= tcx_min && dec && ab != 10 && (tcx_parts & (y.fmt ? 4 : 1))) ? ((tcx_parts & (y.fmt ? 8 : 2)) ? 2 : 1) : 0;
        auto tcx_gu = [&](int ia, int ldi) {
            k_order<<<1, 256, 0, st>>>(IDS, T * TOPK, 8, ORD8);
            k_tcx_xs<<<T, 256, 0, st>>>(XMF, TXH, TXL, TPX);
            const A_tcx a{y.c13, y.s13, y.m13, y.f13, y.f2, IDS, ORD8, WTS, TXH, TXL, TPX, tcx == 2 ? TAH : nullptr,
                          tcx == 2 ? TAL : nullptr, tcx == 2 ? nullptr : ACT, ia, ldi};
            const dim3 gr(T * TOPK, 2 * ldi / 64);
            if (y.fmt) k_tcx<1, 0, 4, 4, 1><<<gr, 128, 0, st>>>(a);
            else if (tcx_v == 1) k_tcx<0, 0, 4, 3, 3><<<gr, 128, 0, st>>>(a);
            else if (tcx_v == 2) k_tcx<0, 0, 4, 3, 2, 3><<<gr, 128, 0, st>>>(a);
            else if (tcx_v == 3) k_tcx<0, 0, 4, 4, 1><<<gr, 128, 0, st>>>(a);
            else if (tcx_v == 4) k_tcx<0, 0, 4, 6, 1><<<gr, 128, 0, st>>>(a);
            else if (tcx_v == 5) k_tcx<0, 0, 8, 4, 1><<<dim3(T * TOPK, 2 * ldi / 128), 256, 0, st>>>(a);
            else k_tcx<0, 0, 4, 4, 2><<<gr, 128, 0, st>>>(a);
        };
        auto tcx_dn = [&](int ia, int ldi, float* out) {
            const A_tcx a{y.c2, y.s2, y.m2, y.f13, y.f2, IDS, ORD8, WTS, TAH, TAL, nullptr, nullptr, nullptr, DPART, ia, ldi};
            const dim3 gr(T * TOPK, H / 64);
            if (y.fmt) k_tcx<1, 1, 4, 4, 1><<<gr, 128, 0, st>>>(a);
            else if (tcx_dv == 1) k_tcx<0, 1, 4, 3, 1><<<gr, 128, 0, st>>>(a);
            else if (tcx_dv == 2) k_tcx<0, 1, 2, 4, 1><<<dim3(T * TOPK, H / 32), 64, 0, st>>>(a);
            else if (tcx_dv == 3) k_tcx<0, 1, 8, 4, 1><<<dim3(T * TOPK, H / 128), 256, 0, st>>>(a);
            else k_tcx<0, 1, 4, 4, 1><<<gr, 128, 0, st>>>(a);
            k_dsum<<<dim3(H / 256, T), 256, 0, st>>>(DPART, out);
        };
        if (tps && T > 1 && !vR && pfx) {   // [ink-tp-pf] grouped expert GEMM on the rank's columns, slots combined, sum over ranks
            if (sp && (spm & SP_MIN)) pf_moe(y, T, 1);   // [ink-tp-sp] top-8 of the rank's rows (gathered after this stage)
            if (spm & SP_MOE) {
                if (sh.nu()) pf_moe(y, T, sp ? 2 : 3);
                auto mrows = [&](int r0, int r1) {
                    if (r1 <= r0) return;
                    if (sh.nu()) pf::k_pcomb<<<((r1 - r0) * (H / 4) + 255) / 256, 256, 0, st>>>(pY + (size_t)r0 * TOPK * H, r1 - r0, tpP + (size_t)r0 * H);
                    else CU(cudaMemsetAsync(tpP + (size_t)r0 * H, 0, (size_t)(r1 - r0) * H * 4, st));
                };
                if (sp) {   // [ink-tp-sp] as the wo partial
                    int p0, p1;
                    sp_priv(T, &p0, &p1);
                    mrows(0, p0);
                    mrows(p1, T);
                    tp_rs_send(T);
                    mrows(p0, p1);
                } else
                    mrows(0, T);
            }
            if (!sp) tp_arP(T, MO);
        } else if (tps) {   // [ink-tp-dec1] the rank's intermediate units -> partial rows (unscaled sums x 16384 / 256: exact)
            ICHECK((T == 1 || vR) && dec && ab != 10 && ab != 34 && ab != 82 && ab != 121 && ab != 91 && ab != 120, "TP: MoE form");
            const int nu = sh.nu(), nch = nu * TP_UNIT / (8 * GP);
            // [ink-tp-tcx] INK_TCX_CHK=1 (with INK_NOGRAPH=1): today's kernels first (reference partial rows), then the TC form
            const bool tchk = tcx && tcx_chk && !capturing && vR;
            bool dnC = false;   // [ink-vp-cpush]
            const int tcx_real = tcx;
            std::vector<float> tref;
            for (int pass = tchk ? 0 : 1; pass < 2; pass++) {
            tcx = pass ? tcx_real : 0;
            if (nu && y.fmt == 0) {
                A_gu4 g{y.c13, y.s13, y.f13, y.f2, XMF, nullptr, y.rb, IDS, WTS, ACT, vR ? 1 : 0};
                g.ia = sh.ua * TP_UNIT;
                g.ldi = nu * TP_UNIT;
                A_dn4 dn{y.c2, y.s2, ACT, IDS, MOP, nullptr, y.mw, Rr, vR ? 1 : 0};
                dn.ua = sh.ua;
                dn.nu = nu;
                if (!vR) {
                    go(k_gu4mb<2>, 0, dim3(nch, TOPK, T), 256, g);
                    dn.pp = tp_push_args(2 * app(l, p) + 1, 1, hout, s.mst, y.mw, Rr);   // [ink-tpf-ppush]
                    if (dn.pp.np) k_dn4p<2, 2, true><<<H / 32, 512, 0, st>>>(dn);
                    else k_dn4p<2, 2><<<H / 32, 512, 0, st>>>(dn);
                } else {   // the one-card row forms (same choices by row count), on the rank's units
                    bool ord_done = false;
                    if (tcx) tcx_gu(g.ia, g.ldi);   // [ink-tp-tcx]
                    else if (vR >= gu4p_min && vR * TOPK <= 64) go(k_gu4p, 0, dim3(T, TOPK, nch), 256, g);
                    else if (vR * TOPK > 64) {
                        k_order<<<1, 256, 0, st>>>(IDS, T * TOPK, 4, ORD);
                        ord_done = true;
                        k_gu4o<<<dim3(T * TOPK, nch), 256, 0, st>>>(g, ORD);
                    } else go(k_gu4mb<2>, 0, dim3(T, TOPK, nch), 256, g);
                    if (tcx == 2) tcx_dn(g.ia, g.ldi, MOP);
                    else {
                        if (!ord_done) k_order<<<1, 256, 0, st>>>(IDS, T * TOPK, 4, ORD);
                        k_dn4o<4, 2><<<dim3(T * TOPK, H / 128), 256, 0, st>>>(dn, ORD, DPART);
                        const TpCol pc = pass && !tchk ? tp_col_args(2 * app(l, p) + 1, 1, vR, MO, s.mst, y.mw, Rr, hout, s.mh) : TpCol{};
                        if (pc.np) {   // [ink-vp-cpush] slot sum + all-reduce + conv chain
                            k_dsumC<<<H / 64, 64, 0, st>>>(DPART, pc);
                            tpc_check(vR, MO, s.mst, y.mw, Rr, hout, s.mh);
                            dnC = true;
                        } else k_dsum<<<dim3(H / 256, T), 256, 0, st>>>(DPART, MOP);
                    }
                }
            } else if (nu) {
                A_gu8 g{y.c13, y.s13, XMF, nullptr, y.rb, IDS, WTS, ACT, vR ? 1 : 0};
                g.ia = sh.ua * TP_UNIT;
                g.ldi = nu * TP_UNIT;
                if (tcx) tcx_gu(g.ia, g.ldi);   // [ink-tp-tcx]
                else if (vR && x8sh_min >= 0 && T * TOPK > x8sh_min) {   // [ink-tp-x8sh] CW columns per warp, decode shared by an expert's users (bitwise)
                    k_order<<<1, 256, 0, st>>>(IDS, T * TOPK, 4, ORD);
                    k_gu8s<X8CW><<<dim3(T * TOPK, nu * TP_UNIT / (8 * X8CW)), 256, 0, st>>>(g, ORD);
                } else go(k_gu8, 0, vR ? dim3(T, TOPK, nu * TP_UNIT / 8) : dim3(nu * TP_UNIT / 8, TOPK, T), 256, g);
                A_dn8 dn{y.c2, y.s2, ACT, IDS, MOP, nullptr, y.mw, Rr, vR ? 1 : 0};
                dn.ua = sh.ua;
                dn.nu = nu;
                if (!vR) dn.pp = tp_push_args(2 * app(l, p) + 1, 1, hout, s.mst, y.mw, Rr);   // [ink-tpf-ppush]
                if (dn.pp.np) go(k_dn8pp, 0, dim3(H / 8, T), 256, dn);
                else if (tcx == 2) tcx_dn(g.ia, g.ldi, MOP);   // [ink-tp-tcx]
                else if (vR && x8sh_min >= 0) k_dn8r<X8RW><<<dim3(T, H / (8 * X8RW)), 256, 0, st>>>(dn);   // [ink-tp-x8sh]
                else go(k_dn8, 0, vR ? dim3(T, H / 8) : dim3(H / 8, T), 256, dn);
            }
            if (tchk && nu) {
                std::vector<float> v((size_t)T * H);
                CU(cudaMemcpyAsync(v.data(), MOP, v.size() * 4, cudaMemcpyDeviceToHost, st));
                CU(cudaStreamSynchronize(st));
                if (!pass) tref = v;
                else {
                    double e2 = 0, r2 = 0, mx = 0, mr = 0;
                    for (size_t i = 0; i < v.size(); i++) {
                        const double d = (double)v[i] - tref[i];
                        e2 += d * d;
                        r2 += (double)tref[i] * tref[i];
                        mx = std::max(mx, fabs(d));
                        mr = std::max(mr, (double)fabs(tref[i]));
                    }
                    auto& c = tcx_cs[y.fmt != 0];
                    c[0] += e2; c[1] += r2; c[2] = std::max(c[2], mx / std::max(mr, 1e-30)); c[3] += 1;
                    if ((long long)c[3] % 100 == 1)
                        fprintf(stderr, "[tcx-chk] %s rank %d rows %d apps %.0f: rel L2 %.3e (this %.3e), max |d| / max |ref| %.3e\n", y.fmt ? "MXFP8" : "NVFP4",
                                sh.rank, T, c[3], sqrt(c[0] / c[1]), sqrt(e2 / std::max(r2, 1e-300)), c[2]);
                }
            }
            }   // pass
            if (dnC) moC = true;   // [ink-vp-cpush] the producer did the sum + chain
            else if (vR) {
                if (tp_arC(2 * app(l, p) + 1, MOP, vR, MO, s.mst, y.mw, Rr, hout, s.mh)) moC = true;   // [ink-tp-tpcc]
                else tp_ar(2 * app(l, p) + 1, MOP, vR, 0, MO);   // then k_convR below (MO -> hout)
            }
            else tp_ar1(2 * app(l, p) + 1, 1, MOP, hout, s.mst, y.mw, Rr);
        } else if (T > 1 && !vR && pfx) pf_moe(y, T);   // prefill: grouped expert GEMM ([ink-pf-xg])
        else if (y.fmt == 0) {
            bool ord_done = false;
            {
                const A_gu4 g{y.c13, y.s13, y.f13, y.f2, XMF, (dec && ab != 10) ? nullptr : SC, y.rb, IDS, WTS, ACT, vR ? 1 : 0};
                if (tcx) tcx_gu(0, I);   // [ink-tp-tcx]
                else if (vR && ab == 34) k_gu4u<GP><<<dim3(T * TOPK, I / (8 * GP)), 256, 0, st>>>(A_gu4u{g, T});   // union (A/B 31: per row); old sum order since [ink-gu4-bal]: not bitwise == plain
                // [ink-mb] __launch_bounds__(256, 2): a different ptxas schedule (A/B -0.83 %; 106 regs since [ink-gu4-bal]); A/B 91: the old bound
                else if (vR && vR >= gu4p_min && vR * TOPK <= 64 && !g.sc && ab != 82) go(k_gu4p, 0, dim3(T, TOPK, I / (8 * GP)), 256, g);   // [ink-gu4-pair] (A/B 82: per row)
                else if (vR && vR * TOPK > 64 && !g.sc && ab != 121) {   // [ink-b-xord] pair jobs in expert-id order (A/B 121: k_gu4mb)
                    k_order<<<1, 256, 0, st>>>(IDS, T * TOPK, 4, ORD);
                    ord_done = true;
                    k_gu4o<<<dim3(T * TOPK, I / (8 * GP)), 256, 0, st>>>(g, ORD);
                } else if (ab != 91) go(k_gu4mb<2>, 0, vR ? dim3(T, TOPK, I / (8 * GP)) : dim3(dim3(I / (8 * GP), TOPK, T)), 256, g);
                else go(k_gu4<>, 0, vR ? dim3(T, TOPK, I / (8 * GP)) : dim3(dim3(I / (8 * GP), TOPK, T)), 256, g);
            }
            const A_dn4 dn{y.c2, y.s2, ACT, IDS, mo, cst, y.mw, Rr, vR ? 1 : 0};
            // [ink-b-dnsh] decode rows (plain and verify): the shared-decode order (per-entry partials, slots added in
            // slot order); T = 1 one fused kernel, T > 1 jobs of <= 4 users of an expert in expert-id order + the slot
            // sum. A/B 120: the old forms (k_dn4s / k_dn4: not bitwise == the new order)
            if (tcx == 2) tcx_dn(0, I, mo);   // [ink-tp-tcx]
            else if (dec && ab != 120) {
                if (T == 1) k_dn4p<2, 2><<<H / 32, 512, 0, st>>>(dn);
                else {
                    if (!ord_done) k_order<<<1, 256, 0, st>>>(IDS, T * TOPK, 4, ORD);
                    k_dn4o<4, 2><<<dim3(T * TOPK, H / 128), 256, 0, st>>>(dn, ORD, DPART);
                    k_dsum<<<dim3(H / 256, T), 256, 0, st>>>(DPART, mo);
                }
            }
            // [ink-dn4-ks] plain step: 16 warps x 2 rows per block (the per-row arithmetic of k_dn4, bitwise); verify
            // rows keep k_dn4 (grid T x 72 already fills the card; the 16-warp form is slower there)
            else if (T == 1 && !vR && ab != 61) go(k_dn4s<2, 16, 1>, 0, dim3(H / 32, T), 512, dn);
            else go(k_dn4<4, true>, 0, vR ? dim3(T, H / 32) : dim3(H / 32, T), 256, dn);   // A/B: smem act -2.5 %
        } else {
            const A_gu8 g8{y.c13, y.s13, XMF, (dec && ab != 10) ? nullptr : SC, y.rb, IDS, WTS, ACT, vR ? 1 : 0};
            if (tcx && !g8.sc) tcx_gu(0, I);   // [ink-tp-tcx]
            else if (vR && x8sh_min >= 0 && T * TOPK > x8sh_min && !g8.sc) {   // [ink-tp-x8sh] CW columns per warp, decode shared by an expert's users (bitwise)
                k_order<<<1, 256, 0, st>>>(IDS, T * TOPK, 4, ORD);
                k_gu8s<X8CW><<<dim3(T * TOPK, I / (8 * X8CW)), 256, 0, st>>>(g8, ORD);
            } else go(k_gu8, 0, vR ? dim3(T, TOPK, I / 8) : dim3(dim3(I / 8, TOPK, T)), 256, g8);
            if (tcx == 2 && !g8.sc) tcx_dn(0, I, mo);   // [ink-tp-tcx]
            else if (vR && x8sh_min >= 0) k_dn8r<X8RW><<<dim3(T, H / (8 * X8RW)), 256, 0, st>>>(A_dn8{y.c2, y.s2, ACT, IDS, mo, cst, y.mw, Rr, 1});   // [ink-tp-x8sh]
            else go(k_dn8, 0, vR ? dim3(T, H / 8) : dim3(dim3(H / 8, T)), 256, A_dn8{y.c2, y.s2, ACT, IDS, mo, cst, y.mw, Rr, vR ? 1 : 0});
        }
        if (vR && ustat && !capturing) {   // INK_USTAT=1 (with INK_NOGRAPH=1): distinct experts per application by k
            std::vector<int> r((size_t)T * TOPK);
            CU(cudaMemcpyAsync(r.data(), IDS, r.size() * 4, cudaMemcpyDeviceToHost, st));
            CU(cudaStreamSynchronize(st));
            std::sort(r.begin(), r.end());
            if (T <= MROW) {
                int pj = 0;
                for (size_t i = 0, j; i < r.size(); i = j) {
                    for (j = i; j < r.size() && r[j] == r[i]; j++) {}
                    pj += (int)(j - i + 1) / 2;
                }
                us_n[y.fmt != 0][T] += 1;
                us_p[y.fmt != 0][T] += pj;
                us_u[y.fmt != 0][T] += std::unique(r.begin(), r.end()) - r.begin();
            }
        }
        if (dumping) {   // routing of every application (debug / gate evidence)
            std::vector<int> r((size_t)T * TOPK);
            CU(cudaMemcpyAsync(r.data(), IDS, r.size() * 4, cudaMemcpyDeviceToHost, st));
            CU(cudaStreamSynchronize(st));
            route.insert(route.end(), r.begin(), r.end());
        }
        if (moC) moC = false;   // [ink-tp-tpcc] the chain ran inside the all-reduce
        else if (vR) k_convR<<<dim3((H + 255) / 256, convseg ? nseg_h : 1), 256, 0, st>>>(MO, vR, s.mst, y.mw, Rr, hout, s.mh, DRT);
        else if (T > 1 && pfx && pfuse && !tps) pf_conv_out(y, s, T, hout);   // prefill: combine + conv fused ([ink-pf-rowfuse]); TP: MO summed above
        else if (sp) {   // [ink-tp-sp] the rank's rows (the state after the reduce-scatter, in layer_sp)
            if (spm & SP_C2) k_conv<<<nb, 256, 0, st>>>(MO, T, H, sp_h0 ? s.mst : MO - 3 * H, y.mw, Rr, hout);
        } else if (T > 1) {
            k_conv<<<nb, 256, 0, st>>>(MO, T, H, s.mst, y.mw, Rr, hout);
            k_cstate<<<(H + 255) / 256, 256, 0, st>>>(MO, H, T, H, s.mst);
        }
        CU(cudaGetLastError());
        if (dumping) {
            std::vector<float> h((size_t)T * H);
            CU(cudaMemcpyAsync(h.data(), hout, h.size() * 4, cudaMemcpyDeviceToHost, st));
            CU(cudaStreamSynchronize(st));
            dump.insert(dump.end(), h.begin(), h.end());
        }
    }
    // forward of T <= CH tokens at positions pos..pos+T-1; leaves the final normed hidden in FIN (fp32) / FINB (bf16)
    // token history (prompt + emitted) for the prompt-lookup drafts: 3-gram -> latest end position
    std::vector<int> hist;
    std::unordered_map<uint64_t, int> lk_map;
    size_t lk_done = 0;
    static uint64_t key3(int a, int b, int c) { return ((uint64_t)(uint32_t)a * 1000003ull + (uint32_t)b) * 1000003ull + (uint32_t)c; }
    void lk_update() {
        for (; lk_done < hist.size(); lk_done++)
            if (lk_done >= 2) lk_map[key3(hist[lk_done - 2], hist[lk_done - 1], hist[lk_done])] = (int)lk_done;
    }
    // drafts copied from the latest earlier occurrence of the context's last 3 tokens (+ anchor); returns the match length
    int lookup(int anchor, int maxd, std::vector<int>& d) {
        d.clear();
        const int n = (int)hist.size();
        if (n < 3) return 0;
        lk_update();
        auto it = lk_map.find(key3(hist[n - 2], hist[n - 1], anchor));
        if (it == lk_map.end()) return 0;
        const int j = it->second;   // hist[j] == anchor, j <= n - 1
        auto at = [&](int i) { return i < n ? hist[i] : anchor; };   // the sequence hist + [anchor]
        int ml = 0;
        while (ml < 64 && j - ml >= 0 && hist[j - ml] == at(n - ml)) ml++;
        for (int i = j + 1; i <= n && (int)d.size() < maxd; i++) d.push_back(at(i));
        return ml;
    }
    long long st_steps = 0, st_lk = 0, st_rows = 0, st_tok = 0, st_kh[VR + 1] = {}, st_ah[VR + 1] = {};
    double st_dms = 0, st_vms = 0;
    long long st_dn = 0;
    // [ink-tp-pf2] half B's view: every row buffer advanced by r rows (P / ZB by r rows of their widest layout), the
    // per-call scratch, stream, cuBLAS handle, position and decode buffer of half B; called again with -r to undo
    void pf_view(int j, int r) {   // sub-chunk j's view (r = its first row), pf_view(j, -r) undoes
        const bool b = r > 0;
        auto adv = [&](auto*& ptr, size_t per_row) { ptr += (ptrdiff_t)r * (ptrdiff_t)per_row; };
        for (float** q : {&E, &X6, &X12[0], &X12[1], &H17[0], &H17[1], &X18, &HA, &HB, &XA[0], &XA[1], &XA[2], &XG[0], &XG[1],
                          &XG[2], &M[0], &M[1], &M[2], &M[3], &Rr, &ATT, &MI, &MIG, &XMF, &MO, &FIN, &tpP})
            adv(*q, H);
        if (dr_on)
            for (int i = 0; i < 3; i++) adv(TAP[i], H);
        for (bf16** q : {&XB[0], &XB[1], &XB[2], &XMB, &FINB, &emb_rows_d, &emb_rows_h}) adv(*q, H);
        adv(AO, QD);
        adv(Qf, QD);
        adv(Rf, (size_t)NQ * DREL);
        adv(Sf, NQ);
        adv(SC, NE);
        adv(IDS, TOPK);
        adv(WTS, TOPK);
        adv(pPENT, TOPK);
        adv(pPW, TOPK);
        adv(pX16, H);
        adv(pY, (size_t)TOPK * H);
        adv(pH16, (size_t)TOPK * I);
        adv(P, pf_npmax);
        adv(ZB, 3 * GR);
        adv(pKs, HD);   // staging [NKV][ch][HD]: rows TA.. of every head
        adv(pVs, HD);
        pf_sub = b ? j : 0;   // review tp2 L6: this view's collective events / receive rows
        pf_r0 = b ? r : 0;
        if (j == 0) {   // sub-chunk 0: the members themselves (its first row is 0)
            pos += r;
            pf_half = -1;
            return;
        }
        std::swap(st, st2[j]);
        std::swap(cps, cps2[j]);
        std::swap(blas, blas2[j]);
        std::swap(dpos, dpos2[j]);
        std::swap(hpos, hpos2[j]);
        std::swap(WS, WS2[j]);
        std::swap(ws_tag, ws_tag2[j]);
        std::swap(pOFF, pOFF2[j]);
        std::swap(pUNITS, pUNITS2[j]);
        std::swap(pNU, pNU2[j]);
        std::swap(pZp, pZp2[j]);
        std::swap(pZs, pZs2[j]);
        pos += r;
        pf_half = b ? j : -1;
    }
    void forward(const int* ids, int T) {
        tp_tick();   // [ink-tp-pf] body() advances the step counter (tag wrap accounting)
        hist.insert(hist.end(), ids, ids + T);
        prep(ids, T);
        // review tp3 M1: prep() proved this rank's GPU finished the previous chunk; the barrier makes it every rank's
        // (the next chunk's sub-chunk row ranges differ, and its first copy must not land in rows a peer still reads)
        if (tps && pfx && T > 1) tpc->bar.wait();
        if (pf2 && T >= 128 * pfk) {   // [ink-tp-pf2] K sub-chunks (rows multiples of 64), j one application behind j - 1
            int r0[PFK_MAX + 1];
            for (int j = 0; j < pfk; j++) r0[j] = (int)((long long)T * j / pfk) & ~63;
            r0[pfk] = T;
            std::vector<std::function<void()>> q[PFK_MAX];
            for (int j = 0; j < pfk; j++) {   // steps bound to sub-chunk j's rows
                pf_view(j, r0[j]);
                struct BGuard {   // review tp3 L2: the view undone if the build throws
                    Impl* p;
                    int j, r;
                    ~BGuard() { p->pf_view(j, -r); }
                } bg{this, j, r0[j]};
                body(r0[j + 1] - r0[j], &q[j]);
            }
            for (int j = 0; j < pfk; j++) ICHECK(q[j].size() == q[0].size(), "pf2 steps");
            CU(cudaEventRecord(pfJ[0], st));
            for (int j = 1; j < pfk; j++) {
                CU(cudaStreamWaitEvent(st2[j], pfJ[0], 0));
                *hpos2[j] = pos + r0[j];
            }
            const int ns = (int)q[0].size();
            try {
                for (int i = 0; i < ns + pfk - 1; i++)   // host order: a wavefront (step i of 0, i - 1 of 1, ...)
                    for (int j = 0; j < pfk; j++) {
                        const int k = i - j;
                        if (k < 0 || k >= ns) continue;
                        pf_view(j, r0[j]);
                        struct VGuard {   // review tp2 H1: the view undone also when the step throws
                            Impl* p;
                            int j, r;
                            ~VGuard() { p->pf_view(j, -r); }
                        } vg{this, j, r0[j]};
                        if (!j) pf_half = 0;
                        q[j][k]();
                    }
            } catch (...) {   // join the sub-chunk streams (their copies may target the peers), then the caller recovers
                pf_half = -1;
                for (int j = 1; j < pfk; j++) cudaStreamSynchronize(st2[j]);
                for (int j = 1; j < pfk; j++) cudaStreamSynchronize(cps2[j]);
                if (cps) cudaStreamSynchronize(cps);
                cudaGetLastError();
                throw;
            }
            for (int j = 1; j < pfk; j++) {
                CU(cudaEventRecord(pfJ[j], st2[j]));
                CU(cudaStreamWaitEvent(st, pfJ[j], 0));
            }
            CU(cudaEventRecord(ev_stage, st));   // every sub-chunk's staging copies are before this point
            stage_pending = true;
        } else
            body(T);
        if (dr_on) dr_ctx(T, IOTA + T - 1);
        pos += T;
    }
    // verify step: rows toks[0..k) at positions pos.. (toks[0] = anchor, then drafts; device copy in DTOK) through the
    // decode-form kernels; per-row exact head; accepted prefix -> conv commit, drafter context; returns acc (0..k-1),
    // the emitted tokens are hres[0..acc]
    int* hres = nullptr;
    std::map<std::vector<int>, cudaGraphExec_t> vgraph;   // verify graphs by (k, per-row split counts)
    std::map<std::vector<int>, int> vseen;                // [ink-tp-vseen] sightings of a key (capture from the second)
    // [ink-vp-vasync] one-segment row steps (c = 1): sighting INK_VASYNC of a key records the graph, runs eagerly and
    // instantiates the graph on a helper thread; the next sighting replays it. INK_VASYNC=0: the vseen rule (capture +
    // instantiate on the critical path) for every step.
    std::map<std::vector<int>, std::future<cudaGraphExec_t>> vpend;
    int vasync = getenv("INK_VASYNC") ? atoi(getenv("INK_VASYNC")) : 2;   // (11-25 TP2 bspec c = 1, 4 rotations: 0 / 2 / 4 = 258.7 / 274.3 / 268.1 tok/s)
    void vpend_drain() {
        for (auto& kv : vpend) {
            cudaGraphExec_t ge = kv.second.get();
            if (ge) cudaGraphExecDestroy(ge);
        }
        vpend.clear();
    }
    bool rr_cold = false;                                 // the last row step captured its graph
    int vseen_min = getenv("INK_VSEEN") ? std::max(1, atoi(getenv("INK_VSEEN"))) : 4;   // (A/B 11-24: 4 >= 2 at c = 1 / 2 / 4)
    // [ink-srv-graph] instantiate a captured graph; on failure (device memory) drop every cached graph and retry once;
    // false: the caller runs the same launches eagerly (degrade, never fail on memory). g is destroyed here.
    long long graph_fallbacks = 0;
    void graphs_flush() {
        vseen.clear();   // review tp3 L8
        vpend_drain();   // [ink-vp-vasync]
        for (auto& kv : segpool)   // review6 L5
            for (auto x : kv.second) cudaGraphExecDestroy(x);
        segpool.clear();
        for (auto& kv : vgraph) cudaGraphExecDestroy(kv.second);
        vgraph.clear();
        for (auto& kv : dgraphR) cudaGraphExecDestroy(kv.second);
        dgraphR.clear();
        if (dgraph) cudaGraphExecDestroy(dgraph);
        dgraph = nullptr;
        if (gstep) cudaGraphExecDestroy(gstep);
        gstep = nullptr;
    }
    bool ginst(cudaGraphExec_t* ge, cudaGraph_t g) {
        cudaError_t e = cudaGraphInstantiate(ge, g, 0);
        if (e != cudaSuccess) {
            cudaGetLastError();
            graphs_flush();
            e = cudaGraphInstantiate(ge, g, 0);
            if (e != cudaSuccess) {
                cudaGetLastError();
                *ge = nullptr;
                if (graph_fallbacks++ % 1000 == 0)
                    fprintf(stderr, "[ink] WARNING: graph instantiate failed (%s) after dropping the graph cache: eager launches\n", cudaGetErrorString(e));
            }
        }
        CU(cudaGraphDestroy(g));
        return *ge != nullptr;
    }
    cudaGraphExec_t dgraph = nullptr;                      // drafter block
    int stop_tok = 200006;                                 // review M5: spec output ends at this token (EOS)
    // [ink-audit-rowgraph] k_attn_tb3 tiles of a row step per layer kind (0 local, 1 global), written into the row
    // table (HRT + RT_TB) before the step; same rules as the per-launch tiling before (a row's arithmetic depends on
    // its chunk keys only, never on its tile). The launch shape (ny, form, tile grid) goes into the graph key.
    struct TbShape {
        int any = 0, all = 1, ny = 1, one = 1, nz = 1;
    };
    TbShape tbs[2];
    void tb_prep(int k) {
        for (int g = 0; g < 2; g++) {
            const bool local = !g;
            int* Tb = HRT + RT_TB + 4 * MROW * g;
            TbShape S;
            int nsx = 1, cm = 1 << 30;
            for (int j = 0; j < MROW; j++) Tb[j] = 0;
            for (int j = 0; j < k; j++) {
                const bool t = tbOn(rpos_h[j], local);
                S.any |= t;
                S.all &= t;
                Tb[j] = t ? tbC(rpos_h[j], local) : 0;
                if (t) {
                    nsx = std::max(nsx, splits1(rpos_h[j], local));
                    cm = std::min(cm, Tb[j]);
                }
            }
            // tile rows cap (launch shape only, a row's arithmetic is the same in any tile): fewer rows per tile while
            // the grid is small (latency-bound: one sequence at short depth), 5 when the grid is large
            // (vbench one sequence, 512 / 4K: k2-k3 best as 1-row tiles, k4-k8 as even tiles of <= 3 rows; 8 x 4 batch:
            // 4-row tiles 6.12 vs 2 + 2 7.68 ms per step)
            int tmax = tbt, smax = 1, t5 = 0;
            for (int j = 0, a = 0; j <= k; j++)   // tiles at 5 rows, largest segment
                if (j == k || (j && HRT[RT_SEG + j] != HRT[RT_SEG + j - 1])) {
                    t5 += (j - a + 4) / 5;
                    smax = std::max(smax, j - a);
                    a = j;
                }
            // [ink-tp-tb1] the ranks' KV heads (ceil(NKV / N): the same on every rank, so every rank picks the same graph key, review7 L2); latency-bound grids take 1-row tiles up to
            // 6 rows of a segment, or at any row count while a row has <= 640 keys (TP2 fixed rows, ms 1-row vs <= 3-row
            // tiles: depth 300 4 / 8 rows 8.60 / 10.99 vs 8.89 / 11.39; depth 1,000 6 / 8 rows 10.25 / 11.69 vs 10.30 / 11.58;
            // 2,000 10.38 / 11.99 vs 10.55 / 11.77). INK_TBS1=3 INK_TBN1=0 INK_TBNKV=0: the old rule
            int nmax = 0;
            for (int j = 0; j < k; j++) nmax = std::max(nmax, local ? std::min(rpos_h[j] + 1, WIN) : rpos_h[j] + 1);
            if ((tb_nkv ? (tps ? std::max(1, (NKV + sh.N - 1) / sh.N) : NKV) : NKV) * tbY(nsx, cm, local) * t5 < tb_minb) tmax = smax <= (tb_s1 ? tb_s1 : tps ? 6 : 3) || nmax <= tb_n1 ? 1 : std::min(tmax, 3);   // latency-bound grid
            int nt = 0, mx = 1;
            for (int j = 0; j < k;) {   // runs of rows of one segment with the same chunk keys, split into even tiles
                if (!Tb[j]) {
                    j++;
                    continue;
                }
                int e = j + 1;
                while (e < k && Tb[e] == Tb[j] && HRT[RT_SEG + e] == HRT[RT_SEG + j]) e++;
                const int n = e - j, kt = (n + tmax - 1) / tmax;
                for (int i = 0, a = j; i < kt; i++) {
                    const int sz = n / kt + (i < n % kt);
                    Tb[MROW + nt] = a;
                    Tb[2 * MROW + nt++] = sz;
                    mx = std::max(mx, sz);
                    a += sz;
                }
                j = e;
            }
            Tb[3 * MROW] = nt;
            S.ny = tbY(nsx, cm, local);
            S.one = mx == 1;
            S.nz = std::max(nt, 1);
            tbs[g] = S;
        }
    }
    // row table of one segment: rows pos..pos + k - 1 of the selected slot
    void rows_one(int k) {
        for (int j = 0; j < k; j++) {
            HRT[j] = rpos_h[j] = pos + j;
            HRT[RT_SLOT + j] = cur;
            samp_row(j, cur);
            HRT[RT_SEG + j] = 0;
        }
        HRT[RT_FIRST] = 0;
        nseg_h = 1;
    }
    // run the row step (rows / table already in HRT, rpos_h, nseg_h, staged tokens): graph keyed by the row count,
    // segments and per-row split / form keys (never by slot: the table is data)
    bool rg_old = getenv("INK_RGOLD") && atoi(getenv("INK_RGOLD"));   // [ink-audit-rowgraph] A/B: the per-row graph key
    long long rg_n[3] = {};
    double rg_ms[2] = {};   // [ink-vp-vasync] host ms of captures (record) and instantiates (vseen path)
    bool launch_dbg = getenv("INK_LNDBG") != nullptr;   // row steps: eager / captured / replayed (INK_RGSTAT=1 prints them at exit)
    void rows_run(int k) {
        tp_tick();
        tb_prep(k);
        if (getenv("INK_NOGRAPH") || dumping || (tpc && !tpc->push)) verify_body(k);   // [ink-tp-dec1] direct collectives: eager; review6 L6: dumps eager
        else {
            // [ink-audit-rowgraph] key = row count, segments, launch shapes; the per-row chunk keys / tiles are data. Steps
            // with rows of the k_attn_tc form (or A/B 125) keep the per-row key.
            std::vector<int> key = {k, nseg_h, rows_batch, filt_rows(k)};   // [ink-samp-dev] filter kernels in the head
            for (int g = 0; g < 2; g++) {
                key.push_back(tbs[g].ny);
                key.push_back(tbs[g].one);
                key.push_back(tbs[g].nz);
            }
            if (!tbs[0].all || !tbs[1].all || ab == 125 || rg_old)
            for (int j = 0; j < k; j++) {
                key.push_back(HRT[RT_SEG + j]);
                key.push_back(splits1(rpos_h[j], true));
                key.push_back(splits1(rpos_h[j], false));
                key.push_back(tbK(rpos_h[j], true));   // [ink-attn-tb] chunk keys (0: k_attn_tc)
                key.push_back(tbK(rpos_h[j], false));
            }
            auto it = vgraph.find(key);
            // [ink-tp-vseen] a key seen for the first time runs eagerly (same launches); its graph is captured on the
            // second sighting: batched row steps rarely repeat a key (rows per slot, segments, split keys), and a capture +
            // instantiate cost ~10-23 ms per new key (TP2 bspec c = 4: 41 of 44 steps). INK_VSEEN=1: capture at once.
            rr_cold = false;   // set below when this step captures (a capture is not a steady-state time: policy EMAs skip it)
            if (it == vgraph.end() && vasync && nseg_h == 1) {   // [ink-vp-vasync]
                auto pi = vpend.find(key);
                if (pi != vpend.end()) {   // instantiated on the helper thread (waits if it is still running)
                    cudaGraphExec_t ge = pi->second.get();
                    vpend.erase(pi);
                    if (ge) it = vgraph.emplace(key, ge).first;
                    else vseen[key] = std::max(vseen[key], vseen_min - 1);   // failed: the vseen rule captures it now (ginst: flush + retry)
                } else if (++vseen[key] < vasync) {   // eager sightings before the capture (the vseen rule's count)
                    if (vseen.size() > 4096) vseen.clear();
                    rg_n[0]++;
                    verify_body(k);
                    CU(cudaStreamSynchronize(st));
                    CU(cudaGetLastError());
                    return;
                } else {
                    if (vgraph.size() + vpend.size() > 64) {   // depth moved on: drop the old buckets
                        for (auto& kv : vgraph) CU(cudaGraphExecDestroy(kv.second));
                        vgraph.clear();
                        vpend_drain();
                    }
                    rr_cold = true;
                    rg_n[1]++;
                    cudaGraph_t g;
                    const auto ta0 = std::chrono::steady_clock::now();
                    capturing = true;
                    CU(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal));
                    verify_body(k);
                    CU(cudaStreamEndCapture(st, &g));
                    capturing = false;
                    int dv = 0;
                    CU(cudaGetDevice(&dv));
                    rg_ms[0] += 1e3 * std::chrono::duration<double>(std::chrono::steady_clock::now() - ta0).count();   // review L8
                    try {
                        vpend.emplace(key, std::async(std::launch::async, [g, dv]() {
                            cudaSetDevice(dv);
                            cudaGraphExec_t ge = nullptr;
                            if (cudaGraphInstantiate(&ge, g, 0) != cudaSuccess) {
                                cudaGetLastError();
                                ge = nullptr;
                            }
                            cudaGraphDestroy(g);
                            return ge;
                        }));
                    } catch (const std::system_error&) {   // review L6: no thread: instantiate here (the vseen path's cost)
                        cudaGraphExec_t ge;
                        if (ginst(&ge, g)) {
                            vgraph.emplace(key, ge);
                            it = vgraph.find(key);
                        }
                    }
                    verify_body(k);   // this sighting: the same launches, eagerly
                    CU(cudaStreamSynchronize(st));
                    CU(cudaGetLastError());
                    return;
                }
            }
            if (it == vgraph.end() && ++vseen[key] < vseen_min) {
                if (vseen.size() > 4096) vseen.clear();
                rg_n[0]++;
                if (segcap && nseg_h > (segc1 ? 0 : 1)) verify_seg(k);   // [ink-tp-segcap] batched steps (c = 1 keeps eager first sightings: its policy picked fewer rows)
                else verify_body(k);
                CU(cudaStreamSynchronize(st));
                CU(cudaGetLastError());
                return;
            }
            if (it == vgraph.end()) {
                rr_cold = true;
                rg_n[1]++;
                if (vgraph.size() > 64) {   // depth moved on: drop the old buckets
                    for (auto& kv : vgraph) CU(cudaGraphExecDestroy(kv.second));
                    vgraph.clear();
                    vpend_drain();   // [ink-vp-vasync]
                }
                cudaGraph_t g;
                const auto tc0 = std::chrono::steady_clock::now();
                capturing = true;
                CU(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal));
                verify_body(k);
                CU(cudaStreamEndCapture(st, &g));
                capturing = false;
                const auto tc1 = std::chrono::steady_clock::now();
                cudaGraphExec_t ge;
                if (ginst(&ge, g)) it = vgraph.emplace(key, ge).first;
                rg_ms[0] += 1e3 * std::chrono::duration<double>(tc1 - tc0).count();
                rg_ms[1] += 1e3 * std::chrono::duration<double>(std::chrono::steady_clock::now() - tc1).count();
            }
            if (it != vgraph.end()) {
                if (!rr_cold) rg_n[2]++;
                const auto tl0 = std::chrono::steady_clock::now();
                CU(cudaGraphLaunch(it->second, st));
                if (launch_dbg && (!tps || sh.rank == 0)) {
                    const auto tl1 = std::chrono::steady_clock::now();
                    CU(cudaStreamSynchronize(st));
                    const auto tl2 = std::chrono::steady_clock::now();
                    fprintf(stderr, "[lndbg] rows %d launch %.0f us, then sync %.0f us\n", k, 1e6 * std::chrono::duration<double>(tl1 - tl0).count(),
                            1e6 * std::chrono::duration<double>(tl2 - tl1).count());
                }
            } else verify_body(k);
        }
        CU(cudaStreamSynchronize(st));
        if (bs_dbg && (!tps || sh.rank == 0)) hts(0);
        CU(cudaGetLastError());
    }
    int verify(const int* toks, int k) {
        ICHECK(k >= 1 && k <= VR, "verify rows %d", k);
        prep(toks, k);
        rows_one(k);
        rows_run(k);
        const int acc = hres[MROW];
        pos += acc + 1;
        hist.insert(hist.end(), toks, toks + acc + 1);
        return acc;
    }
    // [ink-scrtc] bound check (INK_SCR_CHK=1, uncaptured: INK_NOGRAPH=1): exact G15 logits of every vocab row vs the
    // screen's upper ends, and the threshold (largest lower end) vs the exact maximum, per verify row
    bool scr_chk = getenv("INK_SCR_CHK") != nullptr;
    unsigned long long scr_rows = 0, scr_bad = 0, scr_thr_bad = 0;
    double scr_minmargin = 1e30;
    void scr_check(const HeadArgs& h, int k) {
        static float* ex = nullptr;
        if (!ex) CU(cudaMalloc(&ex, (size_t)V * 4));
        std::vector<float> e(V), hv(V);
        std::vector<unsigned> th((size_t)k * 16);
        CU(cudaMemcpyAsync(th.data(), h.thr, th.size() * 4, cudaMemcpyDeviceToHost, st));
        for (int j = 0; j < k; j++) {
            G12Args ga = head12;
            ga.a.x0 = ga.a.x1 = ga.a.x2 = h.x + (size_t)j * H;
            ga.a.N = V;
            ga.a.r1 = ga.a.r2 = V;
            ga.a.y = ex;
            ga.a.cst = nullptr;
            ga.a.bmax = nullptr;
            k_gemv12<H, true><<<(V / 2 + 7) / 8, 256, 0, st>>>(A_gemv12{ga});
            CU(cudaMemcpyAsync(e.data(), ex, (size_t)V * 4, cudaMemcpyDeviceToHost, st));
            CU(cudaMemcpyAsync(hv.data(), h.hi + (size_t)j * V, (size_t)V * 4, cudaMemcpyDeviceToHost, st));
            CU(cudaStreamSynchronize(st));
            float mx = -INFINITY;
            for (int r = 0; r < V; r++) {
                mx = std::max(mx, e[r]);
                scr_bad += e[r] > hv[r];
                scr_minmargin = std::min(scr_minmargin, (double)hv[r] - e[r]);
            }
            if (h.lo && HSAMP[j] > 0.f && (sampK(HSAMP)[j] & 0xff)) {   // [ink-samp-dev] review L5: lower ends of a filtered row
                std::vector<float> lv(V);
                CU(cudaMemcpyAsync(lv.data(), h.lo + (size_t)j * h.N, (size_t)h.N * 4, cudaMemcpyDeviceToHost, st));
                CU(cudaStreamSynchronize(st));
                for (int r = 0; r < h.N; r++) scr_bad += lv[r] > e[r];
            }
            const unsigned o = th[(size_t)j * 16];
            const float thr = (o & 0x80000000u) ? __uint_as_float_h(o & 0x7fffffffu) : __uint_as_float_h(~o);
            scr_thr_bad += o && thr > mx;
            scr_rows += V;
        }
    }
    void verify_body(int k) {
        vR = k;
        body(k);
        verify_tail(k);
    }
    // [ink-tp-segcap] a row step whose key has no graph yet: the body's layer closures each captured as a small graph,
    // the graph of the same topology from a pool updated (cudaGraphExecUpdate) and launched; the host keeps running
    // ahead of the GPU as in an eager step, the GPU runs each layer without per-kernel launch gaps (same launches:
    // bitwise the eager step)
    std::map<size_t, std::vector<cudaGraphExec_t>> segpool;
    long long seg_n[3] = {};   // segments updated / instantiated / failed
    bool segcap = !getenv("INK_SEGCAP") || atoi(getenv("INK_SEGCAP"));   // INK_SEGCAP=0: eager first sightings
    bool segc1 = getenv("INK_SEGC1") && atoi(getenv("INK_SEGC1"));   // [ink-tp-segc1] one-segment steps (c = 1) too
    void seg_run(const std::function<void()>& f) {
        cudaGraph_t g;
        capturing = true;
        CU(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal));
        f();
        CU(cudaStreamEndCapture(st, &g));
        capturing = false;
        size_t nn = 0;
        CU(cudaGraphGetNodes(g, nullptr, &nn));
        auto& pool = segpool[nn];
        cudaGraphExec_t ge = nullptr;
        for (auto x : pool) {
            cudaGraphExecUpdateResultInfo inf;
            if (cudaGraphExecUpdate(x, g, &inf) == cudaSuccess) {
                ge = x;
                break;
            }
            cudaGetLastError();
        }
        if (ge) seg_n[0]++;
        else if (cudaGraphInstantiate(&ge, g, 0) == cudaSuccess) {
            seg_n[1]++;
            pool.push_back(ge);
        } else {
            cudaGetLastError();
            ge = nullptr;
            seg_n[2]++;
        }
        CU(cudaGraphDestroy(g));
        if (ge) CU(cudaGraphLaunch(ge, st));
        else f();
    }
    void verify_seg(int k) {
        std::vector<std::function<void()>> dq;
        vR = k;
        body(k, &dq);
        dq.emplace_back([=, this] { vR = k; verify_tail(k); });   // review6 L4: vR again if the closure is rerun eagerly
        vR = k;
        for (auto& f : dq) seg_run(f);
        vR = 0;
    }
    void verify_tail(int k) {
        {   // screen once for the k rows, rescore per row (own counters and upper ends per row)
            HeadArgs h = h2;
            h.x = FINB;
            h.hi = hiR;
            h.thr = hcR;
            samp_args(h, DRT);
            const bool fl = filt_rows(k);   // [ink-samp-dev] (host-known: part of the row graph's key)
            // [ink-scrtc] tensor-core head screen for k >= 3 rows (a valid bound is all the exact head needs; k4 0.80 -> 0.60
            // ms); k = 2 keeps the CUDA-core form (level there). A/B 71: always the CUDA-core form
            if (k > 24) k_screenTC<true, 4><<<(h.N + 127) / 128, 256, 0, st>>>(h, h.scr, h.seb, h.x, k, nullptr, 0, h.N);   // [ink-b-rows] column tiles
            else if (k > 16) k_screenTC<true, 3><<<(h.N + 127) / 128, 256, 0, st>>>(h, h.scr, h.seb, h.x, k, nullptr, 0, h.N);
            else if (k > 8) k_screenTC<true, 2><<<(h.N + 127) / 128, 256, 0, st>>>(h, h.scr, h.seb, h.x, k, nullptr, 0, h.N);
            else if (k >= 3 && ab != 71) k_screenTC<true><<<(h.N + 127) / 128, 256, 0, st>>>(h, h.scr, h.seb, h.x, k, nullptr, 0, h.N);
            else k_head_screenR<VR><<<(h.N / 2 + 7) / 8, 256, 0, st>>>(h, k);
            if (!capturing && scr_chk) scr_check(h, k);
            if (fl) {
                ICHECK(ab != 131, "A/B 131 with filtered rows");
                filt_thr(h, k);
            }
            if (ab != 131) {   // A/B 131: one launch per row
                HeadArgs r = h2;
                r.x = FINB;
                r.hi = hiR;
                r.step = dpos;
                samp_args(r, DRT);
                if (tps) r.bkey = BKEY;   // [ink-tp-dec1] per row; the tokens = argmax over ranks
                if (fl) {   // [ink-samp-dev] round A (filtered rows only), the tighter threshold
                    HeadArgs a = r;
                    a.fmode = 1;
                    k_head_rescoreR<<<dim3(164, k), 256, 0, st>>>(a, hcR, ARG);
                    filt_tight(r, k);
                }
                k_head_rescoreR<<<dim3(164, k), 256, 0, st>>>(r, hcR, ARG);
                if (tps) tp_max(2 * NAPP, BKEY, ARG, k);
                if (fl) filt_pick(r, k);
            } else
                for (int j = 0; j < k; j++) {
                    ICHECK(!tps, "TP: A/B 131");
                    HeadArgs r = h2;
                    r.x = FINB + (size_t)j * H;
                    r.hi = hiR + (size_t)j * V;
                    r.thr = hcR + j * 16;
                    r.done = r.thr + 1;
                    r.ncand = r.thr + 2;
                    r.best = (unsigned long long*)(r.thr + 4);
                    r.arg = ARG + j;
                    r.step = dpos;
                    k_head_rescore<false><<<164, 256, 0, st>>>(r);
                }
        }
        k_accept<<<1, 32, 0, st>>>(rows_batch ? VTOK : DTOK, ARG, k, ACC, stop_tok, DRT, nseg_h);
        k_commit<<<dim3((H + 255) / 256, NAPP * 4, nseg_h), 256, 0, st>>>(commit_ents, ACC, DRT);
        // [ink-b-draft] per-slot drafter context of a row step; [ink-srv-pool] the single-sequence verify too (its graph
        // is shared by every slot: the slot comes from the row table, never from pointers captured for one slot)
        if (dr_on && (rows_batch || (nseg_h == 1 && !FMH))) dr_ctx_rows(k);
        else if (dr_on && nseg_h == 1) dr_ctx(k, ACC);
        vR = 0;
        CU(cudaMemcpyAsync(hres, ARG, MROW * 4, cudaMemcpyDeviceToHost, st));
        CU(cudaMemcpyAsync(hres + MROW, ACC, MROW * 4, cudaMemcpyDeviceToHost, st));
    }
    // drafter block (graph) -> hdr[0..7] tokens (anchor, 7 drafts), hdp[0..7] probabilities
    int* hdr = nullptr;
    float* hdp = nullptr;
    void draft_run() {
        if (dr_coll()) tp_tick();   // [ink-tp-drv2]
        auto launch = [&] {
            dr_block();
            CU(cudaMemcpyAsync(hdr, DTOK, DB * 4, cudaMemcpyDeviceToHost, st));
            CU(cudaMemcpyAsync(hdp, DPROB, DB * 4, cudaMemcpyDeviceToHost, st));
        };
        if (getenv("INK_NOGRAPH") || (dr_coll() && !tpc->push)) launch();   // [ink-tp-drv2] direct collectives: eager
        else {
            if (dgraph && dns_cap != dns_g(pos)) {   // [ink-dr-nsg] the global-layer split count changed with depth
                CU(cudaGraphExecDestroy(dgraph));
                dgraph = nullptr;
            }
            if (!dgraph) {
                cudaGraph_t g;
                capturing = true;
                dns_cap = dns_g(pos);
                CU(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal));
                launch();
                CU(cudaStreamEndCapture(st, &g));
                capturing = false;
                ginst(&dgraph, g);
            }
            if (dgraph) CU(cudaGraphLaunch(dgraph, st));
            else launch();
        }
        CU(cudaStreamSynchronize(st));
        CU(cudaGetLastError());
    }
    void prep(const int* ids, int T) {
        ICHECK(T >= 1 && T <= ch, "chunk %d", T);
        // review H1: the previous chunk's async copies from the pinned staging buffers must be done before they
        // are overwritten (eager prefill without a blocking hidden-state copy)
        if (stage_pending) {
            CU(cudaEventSynchronize(ev_stage));
            stage_pending = false;
        }
        ICHECK(pos + T <= slim[cur], "context full (%d + %d > %d)", pos, T, slim[cur]);
        for (int t = 0; t < T; t++) {
            ICHECK(ids[t] >= 0 && ids[t] < VPAD, "token id %d", ids[t]);
            memcpy(emb_rows_h + (size_t)t * H, emb_host + (size_t)ids[t] * H, H * 2);
        }
        *hpos = pos;
    }
    // everything on the device for T tokens (graph-capturable for T = 1: no host reads, position from dpos)
    // [ink-tp-pf2] dq non-null: the work is not launched but appended as one step per application (+ a first and a last
    // step), with the buffer pointers of the current view bound at build time; the caller interleaves two halves' steps
    void body(int T, std::vector<std::function<void()>>* dq = nullptr) {
        auto R = [&](auto&& f) {
            if (dq) dq->emplace_back(f);
            else f();
        };
        R([=, this] {
            // [ink-tp-dec1] collective tags of this step; [ink-tp-pf2] once per chunk (sub-chunk 0 only: the sub-chunks'
            // streams run at once, and a racing increment on one rank desynchronises the next head's tags)
            if (tps && tpc->push && pf_half <= 0) k_tpbegin<<<1, 1, 0, st>>>(tstep);
            CU(cudaMemcpyAsync(dpos, hpos, 4, cudaMemcpyHostToDevice, st));
            if (vR) CU(cudaMemcpyAsync(DRT, HRT, RT_N * 4, cudaMemcpyHostToDevice, st));
            if (pf_half <= 0) samp_up();   // [ink-srv-samp] the rows' sampling (filled with the row table / by step()); review tp2 L4
            if (vR && rows_batch) CU(cudaMemcpyAsync(VTOK, HTOK, MROW * 4, cudaMemcpyHostToDevice, st));
            CU(cudaMemcpyAsync(emb_rows_d, emb_rows_h, (size_t)T * H * 2, cudaMemcpyHostToDevice, st));
            if (!capturing) {
                CU(cudaEventRecord(ev_stage, st));
                stage_pending = true;
            }
            go(k_embnorm, 0, dim3(T), 256, A_embnorm{emb_rows_d, embn, E});
        });
        // pre-loop layers 0-5: layer 0 is a block boundary (input E)
        {
            float* e = E;
            float* ha = HA;
            R([=, this] { layer(0, 1, T, e, {}, nullptr, ha); });
        }
        float* h = HA;
        float* other = HB;
        for (int l = 1; l < LOOP0; l++) {   // layer 5 writes the block state X6 directly
            float* out = l == LOOP0 - 1 ? X6 : other;
            float* e = E;
            float* tap = dr_on && l == 4 ? TAP[0] : nullptr;
            R([=, this] {
                layer(l, 1, T, h, {e}, nullptr, out);
                if (tap) {   // [ink-tp-sp] sequence-parallel rows: this rank's rows (gathered at the end)
                    const size_t so = sp_ok(T) ? (size_t)sp_a(T, sh.rank) * H : 0, sn = sp_ok(T) ? (size_t)sp_a(T, sh.rank + 1) * H - so : (size_t)T * H;
                    copy_go(tap + so, other + so, sn);
                }
            });
            if (l < LOOP0 - 1) std::swap(h, other);
        }
        int cur = 0;
        for (int p = 1; p <= NPASS; p++) {
            if (p > 1) cur ^= 1;
            const int pr = cur ^ 1;   // previous pass (messages), p > 1
            const float* hin = X6;
            const bool dec = T == 1 || vR;
            if (p > 1 && dec) {   // all 12 layers' messages in one launch
                BresArgs a{};
                a.s[0] = X12[pr];
                a.s[1] = H17[pr];
                a.ns = 2;
                a.nq = 4;
                a.q[0] = msgq;
                a.q[1] = msgkv;
                a.q[2] = msgkv + H;
                a.q[3] = msgq + H;
                for (int k = 0; k < 4; k++) a.out[k] = M48 + (size_t)k * T * H;
                a.qls = 2 * H;
                a.ols = (size_t)4 * T * H;
                a.mno = MN;
                R([=, this] { bres_go(dim3(T, LOOP1 - LOOP0), a); });
            }
            for (int l = LOOP0; l < LOOP1; l++) {
                float* msg[4] = {M[0], M[1], M[2], M[3]};
                if (p > 1 && dec)
                    for (int k = 0; k < 4; k++) msg[k] = M48 + ((size_t)(l - LOOP0) * 4 + k) * T * H;
                const bool mb = p > 1 && !dec;
                const float *x12 = X12[pr], *h17 = H17[pr];
                // layer 11 writes the block state X12 directly, layer 17 H17 (the last pass: X18)
                float* out = (l == LOOP1 - 1) ? (p == NPASS ? X18 : H17[cur]) : l == 11 ? X12[cur] : (hin == HA ? HB : HA);
                std::vector<const float*> bs = {E};
                if (l > LOOP0) bs.push_back(X6);
                if (l > 12) bs.push_back(X12[cur]);
                float* tap = dr_on && ((p == 2 && l == 10) || (p == 3 && l == 16)) ? TAP[p - 1] : nullptr;
                R([=, this] {
                    const bool spr = sp_ok(T);   // [ink-tp-sp] row-local work on this rank's rows
                    const int sa = spr ? sp_a(T, sh.rank) : 0, sn = spr ? sp_a(T, sh.rank + 1) - sa : T;
                    const size_t so = (size_t)sa * H;
                    if (mb) {
                        const int i = l - LOOP0;
                        bres(sn, {x12 + so, h17 + so}, {msgq + (2 * i) * H, msgkv + (2 * i) * H, msgkv + (2 * i + 1) * H, msgq + (2 * i + 1) * H},
                             {msg[0] + so, msg[1] + so, msg[2] + so, msg[3] + so});
                    }
                    float* const mp[4] = {msg[0], msg[1], msg[2], msg[3]};
                    layer(l, p, T, hin, bs, p > 1 ? mp : nullptr, out);
                    if (tap) copy_go(tap + so, out + so, (size_t)sn * H);
                });
                hin = out;
            }
        }
        // post-loop 18-23: 18 is a boundary
        const float* hin = X18;
        for (int l = LOOP1; l < NL; l++) {
            float* out = (hin == HA) ? HB : HA;
            std::vector<const float*> bs = {E, X6, X12[cur]};
            if (l > LOOP1) bs.push_back(X18);
            const bool lc = l == NL - 1 && T > 1 && !vR && (cut == 1 || (cut == 2 && T > cut_s + CUT_S));
            R([=, this] {
                if (lc) layer_cut(l, T, hin, bs, out);   // [ink-pf-cut]
                else layer(l, 1, T, hin, bs, nullptr, out);
            });
            hin = out;
        }
        {
            const float *e = E, *x6 = X6, *x12 = X12[cur], *x18 = X18;
            float* fin = FIN;
            bf16* finb = FINB;
            R([=, this] {
                const bool spr = sp_ok(T);   // [ink-tp-sp] this rank's rows, then every row on every rank (head, dr_ctx, hidden)
                const int sa = spr ? sp_a(T, sh.rank) : 0, sn = spr ? sp_a(T, sh.rank + 1) - sa : T;
                const size_t so = (size_t)sa * H;
                bres(sn, {e + so, x6 + so, x12 + so, x18 + so, hin + so}, {outq}, {fin + so});
                go(k_rmsnorm, 0, dim3(sn), 256, A_rmsnorm{fin + so, fnorm, finb + so, fin + so});
                if (spr) {
                    if (dr_on)
                        tp_agP(T, {{SPB_FIN, fin, (size_t)H * 4}, {SPB_FINB, finb, (size_t)H * 2}, {SPB_TAP, TAP[0], (size_t)H * 4},
                                   {SPB_TAP + 1, TAP[1], (size_t)H * 4}, {SPB_TAP + 2, TAP[2], (size_t)H * 4}});
                    else tp_agP(T, {{SPB_FIN, fin, (size_t)H * 4}, {SPB_FINB, finb, (size_t)H * 2}});
                }
                CU(cudaGetLastError());
            });
        }
    }
    // greedy head; slot for the exact two-stage head ([ink-exp12] step 2): replace the body, keep ARG / hpos[1]
    void head(int T, const int* rp = nullptr) {
        ICHECK(!tps || (head2 && ab != 2), "TP: the two-stage head only");
        if (head2 && ab != 2) {   // A/B variant 2: the full head
            HeadArgs h = h2;
            h.x = FINB + (size_t)(T - 1) * H;
            h.arg = ARG;
            h.step = dpos;
            samp_args(h, rp ? rp : dpos);   // [ink-srv-samp] row 0 = the plain step's row (position dpos[0])
            if (tps) h.bkey = BKEY;   // [ink-tp-dec1] the rank's best key; the token = argmax over ranks
            const bool fl = filt_rows(1);   // [ink-samp-dev] (host-known: part of the plain step graph's key)
            k_head_screen<false><<<(h.N / 2 + 7) / 8, 256, 0, st>>>(h);
            if (fl) {
                filt_thr(h, 1);
                HeadArgs a = h;
                a.fmode = 1;
                k_head_rescore<false><<<164, 256, 0, st>>>(a);
                filt_tight(h, 1);
            }
            k_head_rescore<false><<<164, 256, 0, st>>>(h);
            if (tps) tp_max(2 * NAPP, BKEY, ARG);
            if (fl) filt_pick(h, 1);
            CU(cudaMemcpyAsync(hpos + 1, ARG, 4, cudaMemcpyDeviceToHost, st));
            return;
        }
        G12Args ga = head12;
        GemvArgs& g = ga.a;
        g.W = unemb;
        g.x0 = g.x1 = g.x2 = FINB + (size_t)(T - 1) * H;
        g.N = V;
        g.r1 = g.r2 = V;
        g.y = LOG;
        g.bmax = BMAX;
        const int nb = (V + 15) / 16;
        if (ab && unemb) k_gemv<H, 2><<<nb, 256, 0, st>>>(g);
        else go(k_gemv12<H, true>, 0, dim3(nb), 256, A_gemv12{ga});
        go(k_argmax2, 0, dim3(1), 1024, A_argmax2{BMAX, nb, ARG});
        CU(cudaMemcpyAsync(hpos + 1, ARG, 4, cudaMemcpyDeviceToHost, st));
    }
    // review tp H2: every head outside body() takes its own step value (tag + parity slot of k_tpmax)
    void tp_head_begin() {
        tp_tick();
        if (tps && tpc->push) k_tpbegin<<<1, 1, 0, st>>>(tstep);
    }
    int argmax_last(int T) {
        tp_head_begin();   // [ink-tp-dec1] a head outside a step: its own tag
        head(T);
        CU(cudaStreamSynchronize(st));
        return hpos[1];
    }
    // one decode step through a CUDA graph (captured once: every buffer is fixed, the position is read on the device)
    int step(int id) {
        tp_tick();
        samp_row(0, cur);   // [ink-srv-samp]
        prep(&id, 1);
        hist.push_back(id);
        if (dumping || getenv("INK_NOGRAPH") || (tpc && !tpc->push)) {   // [ink-tp-dec1] direct collectives: host-ordered (eager)
            body(1);
            if (dr_on) dr_ctx(1, IOTA);
            pos += 1;
            return argmax_last(1);
        }
        if (gstep && (cap_ns[0] != attn_splits(1, true) || cap_ns[1] != attn_splits(1, false) || cap_C[0] != tbK(pos, true) ||
                      cap_C[1] != tbK(pos, false) || cap_filt != filt_rows(1))) {
            CU(cudaGraphExecDestroy(gstep));   // a depth bucket changed: recapture
            gstep = nullptr;
        }
        if (!gstep) {
            cap_ns[0] = attn_splits(1, true);
            cap_ns[1] = attn_splits(1, false);
            cap_C[0] = tbK(pos, true);
            cap_C[1] = tbK(pos, false);
            cap_filt = filt_rows(1);
            cudaGraph_t g;
            capturing = true;
            CU(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal));
            body(1);
            head(1);
            if (dr_on) dr_ctx(1, IOTA);
            CU(cudaStreamEndCapture(st, &g));
            capturing = false;
            ginst(&gstep, g);
        }
        if (gstep) CU(cudaGraphLaunch(gstep, st));
        else {
            body(1);
            head(1);
            if (dr_on) dr_ctx(1, IOTA);
        }
        CU(cudaStreamSynchronize(st));
        if (getenv("INK_DBG")) {
            float f[4], l[4], bm[4];
            int ar;
            CU(cudaMemcpy(f, FIN, 16, cudaMemcpyDeviceToHost));
            CU(cudaMemcpy(l, LOG, 16, cudaMemcpyDeviceToHost));
            CU(cudaMemcpy(bm, BMAX, 16, cudaMemcpyDeviceToHost));
            CU(cudaMemcpy(&ar, ARG, 4, cudaMemcpyDeviceToHost));
            fprintf(stderr, "[dbg] pos %d FIN %g %g LOG %g %g ARG %d hpos1 %d\n", pos, f[0], f[1], l[0], l[1], ar, hpos[1]);
        }
        pos += 1;
        return hpos[1];
    }
    // ---- slots
    void select(int sl) {
        ICHECK(sl >= 0 && sl < S, "slot %d of %d", sl, S);
        if (sl == cur) return;
        spos[cur] = pos;
        shist[cur].swap(hist);
        cur = sl;
        pos = spos[cur];
        hist.swap(shist[cur]);
        shist[cur].clear();
        lk_map.clear();   // prompt lookup index: rebuilt from hist
        lk_done = 0;
        views_changed();
    }
    void views_changed() {   // the selected slot's views (select, bind of the selected slot)
        for (int a = 0; a < NAPP; a++) A[a] = view(a, cur);
        if (FMS) FM = FMS + (size_t)cur * H;
        if (dgraph) {   // the single-sequence drafter graph holds the old slot's cache pointers
            CU(cudaGraphExecDestroy(dgraph));
            dgraph = nullptr;
        }
        if (gstep) {   // the plain step graph holds the old slot's pointers
            CU(cudaGraphExecDestroy(gstep));
            gstep = nullptr;
        }
    }
    // ---------------------------------------------------------------- [ink-srv-pool] server hooks
    void bind(int sl, int base, int p, int lim) {
        ICHECK(sl >= 0 && sl < S && base >= 0 && lim >= 0 && (long long)base + lim <= POOL && p >= 0 && p <= lim,
               "bind slot %d base %d pos %d lim %d (pool %d)", sl, base, p, lim, POOL);
        const bool moved = sbase[sl] != base;
        sbase[sl] = base;
        slim[sl] = lim;
        if (sl == cur) pos = p;
        else spos[sl] = p;
        if (moved) {
            sko_upload();
            if (sl == cur) views_changed();
        }
    }
    // the full layers' K / V (8 applications) + the drafter's global layer of pool rows: copy / save / load
    template <class F>
    void pool_parts(F&& f) {   // f(device base, row bytes, head pitch bytes, heads)
        // [ink-tp-srv] a TP rank: its K / V heads [kv0, kv0 + nkv) (the drafter is replicated: all heads)
        const size_t ho = (size_t)kv0A() * POOL * HD * 2;
        for (int a = 0; a < NAPP && nkvA(); a++)
            if (!A0[a].ring) {
                f((uint8_t*)A0[a].K + ho, (size_t)HD * 2, (size_t)POOL * HD * 2, nkvA());
                f((uint8_t*)A0[a].V + ho, (size_t)HD * 2, (size_t)POOL * HD * 2, nkvA());
            }
        if (dr_on && dsko(1) == SKO_G) {
            f(DK[1], (size_t)HD, (size_t)dcap[1] * HD, NKV);
            f(DV[1], (size_t)HD, (size_t)dcap[1] * HD, NKV);
            f((uint8_t*)DKS[1], 4, (size_t)dcap[1] * 4, NKV);
            f((uint8_t*)DVS[1], 4, (size_t)dcap[1] * 4, NKV);
        }
    }
    size_t pool_row_bytes() {
        size_t b = 0;
        pool_parts([&](uint8_t*, size_t rb, size_t, int nh) { b += rb * nh; });
        return b;
    }
    void copy_rows(int src, int dst, int n) {
        ring_finalize();
        ICHECK(n >= 0 && src >= 0 && dst >= 0 && src + n <= POOL && dst + n <= POOL && (src + n <= dst || dst + n <= src || n == 0),
               "copy_rows %d -> %d x %d (pool %d)", src, dst, n, POOL);
        if (n == 0 || src == dst) return;
        pool_parts([&](uint8_t* b, size_t rb, size_t pitch, int nh) {
            CU(cudaMemcpy2DAsync(b + dst * rb, pitch, b + src * rb, pitch, n * rb, nh, cudaMemcpyDeviceToDevice, st));
        });
        CU(cudaStreamSynchronize(st));
    }
    // [ink-srv-gap] pinned staging for the host copies of the server hooks (pageable copies ran at ~1.5 GB/s: the
    // prompt-end snapshot of a 2K prompt, 176 MB, took 121 ms of its TTFT). Lazily grown up to STG_MAX; a larger copy
    // or a failed pin uses the pageable path (same bytes)
    const size_t STG_MAX = (size_t)(getenv("INK_STG_MB") ? std::max(1, atoi(getenv("INK_STG_MB"))) : 512) << 20;   // (INK_STG_MB: tests)
    uint8_t* stg = nullptr;
    size_t stg_cap = 0;
    uint8_t* staging(size_t n) {
        ring_finalize();   // (a pending snapshot holds the buffer)
        if (n == 0 || n > STG_MAX || getenv("INK_NOSTAGE")) return nullptr;
        if (stg_cap >= n) return stg;
        if (stg) cudaFreeHost(stg);
        stg = nullptr;
        stg_cap = 0;
        const size_t want = std::min(STG_MAX, std::max(n, (size_t)64 << 20));
        if (cudaHostAlloc((void**)&stg, want, 0) != cudaSuccess) {
            cudaGetLastError();   // (host memory pressure: the pageable path)
            stg = nullptr;
            return nullptr;
        }
        stg_cap = want;
        return stg;
    }
    // rows [src, src + n) of every pool part, the blob laid out part by part ([part][head][row] as before); through the
    // staging buffer in row pieces of <= STG_MAX bytes (an archive of 62K rows, 2.1 GB, ran pageable inside a TTFT)
    void rows_save(int src, int n, std::vector<unsigned char>& out) {
        out.resize(pool_row_bytes() * n);
        rows_save_raw(src, n, out.data());
    }
    void rows_save_raw(int src, int n, unsigned char* outp) {   // [ink-tp-srv] into pool_row_bytes() x n bytes at outp
        ICHECK(n >= 0 && src >= 0 && src + n <= POOL, "rows_save %d x %d", src, n);
        const size_t per = pool_row_bytes(), tot = per * n;
        (void)tot;
        const int step = per ? (int)std::max<size_t>(1, STG_MAX / per) : n;
        uint8_t* sg = n > 0 ? staging(std::min(tot, per * (size_t)step)) : nullptr;
        if (!sg) {
            size_t o = 0;
            pool_parts([&](uint8_t* b, size_t rb, size_t pitch, int nh) {
                CU(cudaMemcpy2DAsync(outp + o, n * rb, b + src * rb, pitch, n * rb, nh, cudaMemcpyDeviceToHost, st));
                o += n * rb * nh;
            });
            CU(cudaStreamSynchronize(st));
            return;
        }
        for (int r0 = 0; r0 < n; r0 += step) {
            const int k = std::min(step, n - r0);
            size_t o = 0, oo = 0;   // o: offset in the piece, oo: offset of this part in the blob
            std::vector<std::tuple<size_t, size_t, size_t, int>> cp;   // (piece offset, blob offset, row bytes, heads)
            pool_parts([&](uint8_t* b, size_t rb, size_t pitch, int nh) {
                CU(cudaMemcpy2DAsync(sg + o, k * rb, b + (src + r0) * rb, pitch, k * rb, nh, cudaMemcpyDeviceToHost, st));
                cp.emplace_back(o, oo, rb, nh);
                o += k * rb * nh;
                oo += (size_t)n * rb * nh;
            });
            CU(cudaStreamSynchronize(st));
            for (auto& [po, bo, rb, nh] : cp)
                for (int hh = 0; hh < nh; hh++)
                    memcpy(outp + bo + (size_t)hh * n * rb + (size_t)r0 * rb, sg + po + (size_t)hh * k * rb, k * rb);
        }
    }
    void rows_load(const std::vector<unsigned char>& in, int dst, int n) {   // in: a rows_save blob of >= n rows (its first n)
        const size_t per = pool_row_bytes();
        ICHECK(per == 0 || in.size() % per == 0, "rows_load: blob size");
        rows_load_raw(in.data(), per ? in.size() / per : 0, dst, n);
    }
    void rows_load_raw(const unsigned char* inp, size_t m, int dst, int n) {   // [ink-tp-srv] a blob of m rows at inp
        ICHECK(n >= 0 && dst >= 0 && dst + n <= POOL, "rows_load %d x %d", dst, n);
        const size_t per = pool_row_bytes();
        ICHECK((size_t)n <= m, "rows_load: %d rows from a blob of %zu", n, m);
        const int step = per ? (int)std::max<size_t>(1, STG_MAX / per) : n;
        uint8_t* sg = n > 0 ? staging(std::min(per * (size_t)n, per * (size_t)step)) : nullptr;
        if (!sg) {
            size_t o = 0;
            pool_parts([&](uint8_t* b, size_t rb, size_t pitch, int nh) {
                CU(cudaMemcpy2DAsync(b + dst * rb, pitch, inp + o, m * rb, n * rb, nh, cudaMemcpyHostToDevice, st));
                o += m * rb * nh;
            });
            CU(cudaStreamSynchronize(st));
            return;
        }
        for (int r0 = 0; r0 < n; r0 += step) {
            const int k = std::min(step, n - r0);
            size_t o = 0, oo = 0;
            pool_parts([&](uint8_t*, size_t rb, size_t, int nh) {
                for (int hh = 0; hh < nh; hh++) memcpy(sg + o + (size_t)hh * k * rb, inp + oo + (size_t)hh * m * rb + (size_t)r0 * rb, k * rb);
                o += k * rb * nh;
                oo += m * rb * nh;
            });
            o = 0;
            pool_parts([&](uint8_t* b, size_t rb, size_t pitch, int nh) {
                CU(cudaMemcpy2DAsync(b + (dst + r0) * rb, pitch, sg + o, k * rb, k * rb, nh, cudaMemcpyHostToDevice, st));
                o += k * rb * nh;
            });
            CU(cudaStreamSynchronize(st));
        }
    }
    // the selected slot's state at position p besides the pool rows: the sliding rings' window, the 4 conv states of
    // every application, the drafter's rings and its last feature row (f(device ptr, bytes))
    template <class F>
    void ring_ranges(int p, F&& f) {
        for (int a = 0; a < NAPP; a++) {
            const AppState& s = A[a];
            if (s.ring)
                for (int kv = 0; kv < 2; kv++)
                    for (int h = kv0A(); h < kv0A() + nkvA(); h++) {   // [ink-tp-srv] the rank's heads
                        bf16* hb = (kv ? s.V : s.K) + (size_t)h * s.cap * HD;
                        const int lo = std::max(0, p - WIN), n = p - lo, s0 = lo % s.cap, n1 = std::min(n, s.cap - s0);
                        f((uint8_t*)(hb + (size_t)s0 * HD), (size_t)n1 * HD * 2, true);
                        if (n > n1) f((uint8_t*)hb, (size_t)(n - n1) * HD * 2, true);
                    }
            f((uint8_t*)s.kst, 3 * KVD * 4, false);
            f((uint8_t*)s.vst, 3 * KVD * 4, false);
            f((uint8_t*)s.ast, 3 * H * 4, false);
            f((uint8_t*)s.mst, 3 * H * 4, false);
        }
        if (dr_on) {
            for (int i = 0; i < 3; i += 2)
                for (int kv = 0; kv < 2; kv++)
                    for (int h = 0; h < NKV; h++) {
                        uint8_t* hb = (kv ? dV(i) : dK(i)) + (size_t)h * dcap[i] * HD;
                        float* sb = (kv ? dVS(i) : dKS(i)) + (size_t)h * dcap[i];
                        const int lo = std::max(0, p - WIN), n = p - lo, s0 = lo % dcap[i], n1 = std::min(n, dcap[i] - s0);
                        f(hb + (size_t)s0 * HD, (size_t)n1 * HD, true);
                        f((uint8_t*)(sb + s0), (size_t)n1 * 4, true);
                        if (n > n1) {
                            f(hb, (size_t)(n - n1) * HD, true);
                            f((uint8_t*)sb, (size_t)(n - n1) * 4, true);
                        }
                    }
            f((uint8_t*)FM, H * 2, false);
        }
    }
    void ring_save(std::vector<unsigned char>& out) {
        size_t n = 8;
        ring_ranges(pos, [&](uint8_t*, size_t b, bool) { n += b; });
        uint8_t* sg = staging(n);
        if (sg) out.clear();
        else out.resize(n);
        uint8_t* dst = sg ? sg : out.data();
        memcpy(dst, &pos, 4);
        const int dr = dr_on;
        memcpy(dst + 4, &dr, 4);
        size_t o = 8;
        ring_ranges(pos, [&](uint8_t* d, size_t b, bool) {
            CU(cudaMemcpyAsync(dst + o, d, b, cudaMemcpyDeviceToHost, st));
            o += b;
        });
        CU(cudaStreamSynchronize(st));
        if (sg) out.assign(sg, sg + n);
    }
    // [ink-srv-gap] the same blob, the window rows copied on a side stream while the next step runs (the conv states and
    // the feature now: the step overwrites them); `out` is filled by ring_finalize (at the end of the next row step, or
    // before any call that could touch those rows or the staging buffer). A prompt-end snapshot left the TTFT window.
    cudaStream_t cst = nullptr;
    cudaEvent_t ev_r0 = nullptr, ev_r1 = nullptr;
    std::vector<unsigned char>* rpend = nullptr;
    size_t rpend_n = 0;
    std::thread rthr;   // the host copy (pinned -> out) off the step path; joined by ring_finalize
    // [ink-srv-review-fix] M3: an exception in the copy thread would be std::terminate: the thread catches it and leaves
    // its vector here; ring_finalize drops that snapshot (only an optimisation: the next turn re-prefills)
    std::vector<unsigned char>* rfail = nullptr;
    void ring_finalize() {
        if (rthr.joinable()) rthr.join();
        if (rfail) {
            rfail->clear();
            rfail->shrink_to_fit();
            rfail = nullptr;
            fprintf(stderr, "[ink] prompt-end snapshot dropped (host copy failed)\n");
        }
        if (!rpend) return;
        std::vector<unsigned char>* o = rpend;
        rpend = nullptr;
        CU(cudaEventSynchronize(ev_r1));
        try {
            o->assign(stg, stg + rpend_n);
        } catch (const std::bad_alloc&) {
            o->clear();
            o->shrink_to_fit();
            fprintf(stderr, "[ink] prompt-end snapshot dropped (no host memory for its copy)\n");
        }
    }
    void ring_finalize_bg() {   // after the next step: the copy finishes on a host thread (its vector is not read before
        if (!rpend || rthr.joinable()) return;   // ring_finalize, which every reader of the vector calls first)
        std::vector<unsigned char>* o = rpend;
        rpend = nullptr;
        const size_t n = rpend_n;
        uint8_t* src = stg;
        cudaEvent_t ev = ev_r1;
        rthr = std::thread([this, o, n, src, ev] {
            try {
                if (cudaEventSynchronize(ev) != cudaSuccess) throw std::runtime_error("snapshot copy event");
                o->assign(src, src + n);
            } catch (...) {
                rfail = o;   // (read by ring_finalize after the join)
            }
        });
    }
    void ring_save_async(std::vector<unsigned char>& out) {
        ring_finalize();
        size_t n = 8;
        ring_ranges(pos, [&](uint8_t*, size_t b, bool) { n += b; });
        uint8_t* sg = staging(n);
        if (!sg || getenv("INK_SNAP_SYNC")) {
            ring_save(out);
            return;
        }
        if (!cst) {
            CU(cudaStreamCreateWithFlags(&cst, cudaStreamNonBlocking));
            CU(cudaEventCreateWithFlags(&ev_r0, cudaEventDisableTiming));
            CU(cudaEventCreateWithFlags(&ev_r1, cudaEventDisableTiming));
        }
        out.clear();
        memcpy(sg, &pos, 4);
        const int dr = dr_on;
        memcpy(sg + 4, &dr, 4);
        CU(cudaEventRecord(ev_r0, st));
        CU(cudaStreamWaitEvent(cst, ev_r0, 0));
        size_t o = 8;
        ring_ranges(pos, [&](uint8_t* d, size_t b, bool win) {
            CU(cudaMemcpyAsync(sg + o, d, b, cudaMemcpyDeviceToHost, win ? cst : st));
            o += b;
        });
        CU(cudaEventRecord(ev_r1, cst));
        CU(cudaStreamSynchronize(st));
        rpend = &out;
        rpend_n = n;
    }
    void ring_load(const std::vector<unsigned char>& in, int n) {   // the selected slot = the first n rows (pool rows in place)
        int p = -1, dr = -1;
        ICHECK(in.size() >= 8, "ring_load: empty blob");
        memcpy(&p, in.data(), 4);
        memcpy(&dr, in.data() + 4, 4);
        ICHECK(p == n && dr == (int)dr_on && n <= slim[cur], "ring_load: blob at %d (drafter %d), asked %d (drafter %d, lim %d)", p, dr, n,
               (int)dr_on, slim[cur]);
        reset();
        uint8_t* sg = staging(in.size());
        if (sg) memcpy(sg, in.data(), in.size());
        const uint8_t* srcp = sg ? sg : in.data();
        size_t o = 8;
        ring_ranges(n, [&](uint8_t* d, size_t b, bool) {
            CU(cudaMemcpyAsync(d, srcp + o, b, cudaMemcpyHostToDevice, st));
            o += b;
        });
        ICHECK(o == in.size(), "ring_load: blob size");
        pos = n;
        last_T = 1;
        CU(cudaStreamSynchronize(st));
    }
    // nr[i] rows for slot sl[i]: toks (flattened) = anchor + nr[i] - 1 draft tokens per slot; arg (flattened) = the
    // greedy token after each row; acc[i] = accepted drafts (emitted: arg[f_i .. f_i + acc[i]])
    void step_rows_k(const int* sl, const int* nr, const int* toks, int n, int* arg, int* acc) {
        ICHECK(n >= 1 && n <= MROW, "step_rows: %d slots", n);
        spos[cur] = pos;
        int R = 0;
        for (int j = 0; j < n; j++) {
            ICHECK(sl[j] >= 0 && sl[j] < S, "slot %d", sl[j]);
            for (int i = 0; i < j; i++) ICHECK(sl[i] != sl[j], "slot %d twice", sl[j]);
            ICHECK(nr[j] >= 1 && spos[sl[j]] + nr[j] <= slim[sl[j]], "slot %d: %d rows, context full", sl[j], nr[j]);
            R += nr[j];
        }
        ICHECK(R <= MROW, "step_rows: %d rows (max %d)", R, MROW);
        if (stage_pending) {
            CU(cudaEventSynchronize(ev_stage));
            stage_pending = false;
        }
        int r = 0;
        for (int j = 0; j < n; j++) {
            HRT[RT_FIRST + j] = r;
            for (int i = 0; i < nr[j]; i++, r++) {
                ICHECK(toks[r] >= 0 && toks[r] < VPAD, "token id %d", toks[r]);
                memcpy(emb_rows_h + (size_t)r * H, emb_host + (size_t)toks[r] * H, H * 2);
                HRT[r] = rpos_h[r] = spos[sl[j]] + i;
                HRT[RT_SLOT + r] = sl[j];
                samp_row(r, sl[j]);
                HRT[RT_SEG + r] = HRT[RT_FIRST + j];
                HTOK[r] = toks[r];
            }
        }
        nseg_h = n;
        *hpos = rpos_h[0];
        rows_batch = true;
        rows_run(R);
        rows_batch = false;
        r = 0;
        for (int j = 0; j < n; j++) {
            const int a = hres[MROW + j];
            acc[j] = a;
            auto& hs = sl[j] == cur ? hist : shist[sl[j]];
            hs.insert(hs.end(), toks + r, toks + r + a + 1);
            spos[sl[j]] += a + 1;
            r += nr[j];
        }
        for (int i = 0; i < R; i++) arg[i] = hres[i];
        pos = spos[cur];
        last_T = 1;   // [ink-srv-capi] FINB rows are this step's now (row_logits(0) must not take a prefill's last row)
        ring_finalize_bg();   // [ink-srv-gap] a snapshot copied beside this step: its host copy goes on a thread
        if (bs_dbg && (!tps || sh.rank == 0)) hts(1);
    }
    void step_rows(const int* sl, const int* toks, int n, int* out) {
        std::vector<int> one(n, 1), acc(n);
        step_rows_k(sl, one.data(), toks, n, out, acc.data());
    }
    // ---- [ink-b-draft] batched speculative step: one drafter block for the n slots, rows per slot by the policy
    // (INK_BPOL 0 off, 1 per-slot confidence stop, 2 row budget), one row step; emitted tokens == greedy alone
    int bpol = getenv("INK_BPOL") ? atoi(getenv("INK_BPOL")) : 2;
    // [ink-srv-gap] step ms by rows, per depth bucket (the deepest slot of the step: < 4K, 4-8K, 8-16K, .. >= 256K): one
    // table for all depths kept a 120K request's costs (extra rows cheap) for the next short reply, which then drafted
    // 2.2x the rows (prose 111 vs 133 tok/s on a fresh server)
    double e_rows_b[8][MROW + 1] = {}, e_dr[9] = {};
    double* e_rows = e_rows_b[0];
    static int depth_bucket(int p) {
        int b = 0;
        for (long long x = 4096; b < 7 && p >= x; x *= 2) b++;
        return b;
    }
    long long bs_steps = 0, bs_rows = 0, bs_tok = 0, bs_dr = 0;
    bool bs_dbg = getenv("INK_BSDBG") != nullptr;
    std::chrono::steady_clock::time_point bs_te, bs_tv;   // INK_BSDBG: entry / verify start of the last call (rank skew)   // per-call host times of spec_rows (rank 0)
    double bs_dms = 0, bs_vms = 0;
    // drafter confidence -> conditional hit rate by 20 p bins (official drafter: b6_replay.py calibration, 9 full
    // answers); [ink-dr-drop] INK_DCAL = 20 comma-separated values replaces the table
    float dcal_c[20] = {0.03f, 0.06f, 0.10f, 0.10f, 0.11f, 0.14f, 0.19f, 0.24f, 0.24f, 0.26f,
                        0.34f, 0.35f, 0.39f, 0.39f, 0.44f, 0.48f, 0.51f, 0.57f, 0.63f, 0.95f};
    // [ink-dr-drop] our fine-tuned DSpark (same 3-layer shapes, drop-in): less over-confident than the official one,
    // so its own calibration (TP2, all 7 drafts verified: 9 gate items x 1,024 + 100 held-out prompts, 17.5K blocks)
    // and confidence stop (offline re-fit on the same blocks: tau 0.5 best on both sets, tau1 0.6, kmin 2)
    static constexpr float DCAL_QE[20] = {0.00f, 0.05f, 0.09f, 0.11f, 0.15f, 0.18f, 0.20f, 0.25f, 0.29f, 0.31f,
                                          0.37f, 0.37f, 0.43f, 0.49f, 0.53f, 0.54f, 0.60f, 0.66f, 0.74f, 0.95f};
    static constexpr float TAU_QE = 0.5f;
    // drafter weights: INK_DRAFTER_FILE = a path (with '/') or a file name in the model dir; else the trained
    // mtp_qe.safetensors when the model dir has it, else the official mtp.safetensors (INK_DRAFTER_FILE=mtp.safetensors
    // is the off-switch to the official drafter)
    static std::string drafter_file(const std::string& md) {
        if (const char* e = getenv("INK_DRAFTER_FILE"); e && *e) return strchr(e, '/') ? std::string(e) : md + "/" + e;
        const std::string q = md + "/mtp_qe.safetensors";
        if (FILE* f = fopen(q.c_str(), "rb")) {
            fclose(f);
            return q;
        }
        return md + "/mtp.safetensors";
    }
    static bool drafter_qe(const std::string& f) {   // any drafter file but the official one: the trained policy
        const size_t s = f.find_last_of('/');
        return f.substr(s == std::string::npos ? 0 : s + 1) != "mtp.safetensors";
    }
    float dcal(float p) const {
        const int b = std::min(19, std::max(0, (int)(p * 20.f)));
        return dcal_c[b];
    }
    static double t_tab(int r) {   // the one-card engine's ~3K-depth step ms by rows
        static const int R[8] = {1, 2, 3, 4, 6, 8, 16, 32};
        static const double Tm[8] = {8.1, 10.4, 12.1, 13.5, 16.3, 21.3, 32.4, 53.9};
        for (int i = 0; i + 1 < 8; i++)
            if (r <= R[i + 1]) return Tm[i] + (Tm[i + 1] - Tm[i]) * (r - R[i]) / (R[i + 1] - R[i]);
        return Tm[7];
    }
    // step ms by rows: the measured EMA; [ink-pol-fit] an unmeasured row count from the nearest measured counts of
    // this depth bucket (linear between two, the table's shape scaled to one), the table only when none is measured
    // (the one-card table priced unmeasured counts 1.6x too high on TP2: the budget kept to the first counts it measured)
    bool pol_fit = !getenv("INK_POLFIT") || atoi(getenv("INK_POLFIT"));
    double t_rows(int r) {
        if (e_rows[r] > 0) return e_rows[r];
        if (!pol_fit) return t_tab(r);
        int lo = 0, hi = 0;
        for (int q = r - 1; q >= 1 && !lo; q--)
            if (e_rows[q] > 0) lo = q;
        for (int q = r + 1; q <= MROW && !hi; q++)
            if (e_rows[q] > 0) hi = q;
        if (lo && hi) return e_rows[lo] + (e_rows[hi] - e_rows[lo]) * (r - lo) / (hi - lo);
        if (lo || hi) {
            const int m = lo ? lo : hi;
            return t_tab(r) * e_rows[m] / t_tab(m);
        }
        return t_tab(r);
    }
    long long filt_fb = 0, filt_tok = 0;   // [ink-samp-dev] filtered rows picked on the host (Engine::filt_host), filtered tokens
    // review M2: per-slot rate of host picks (EMA over the slot's filtered tokens); above FB_NODRAFT the slot steps one row
    // without drafts (a host pick ends the slot's drafted rows anyway), so a fallback costs no more than master's host path
    std::vector<float> fb_ema;
    static constexpr float FB_NODRAFT = 0.15f;
    std::vector<int> last_k;   // [ink-srv-capi] drafts per slot of the last spec_rows call (rows = 1 + k)
    void spec_rows(const int* sl, const int* anch, int n, int* out, int* nout, const unsigned char* nodraft = nullptr) {
        const auto tq0 = std::chrono::steady_clock::now();
        bs_te = tq0;
        if (bs_dbg && (!tps || sh.rank == 0)) {
            auto us = [](auto a, auto b) { return 1e6 * std::chrono::duration<double>(b - a).count(); };
            fprintf(stderr, "[htsdbg] sync->rows_end %.0f ->spec_end %.0f ->outer_ret %.0f ->outer_entry %.0f ->rank_entry %.0f ->dlaunch(prev) %.0f\n", us(g_hts[0], g_hts[1]),
                    us(g_hts[1], g_hts[2]), us(g_hts[2], g_hts[3]), us(g_hts[3], g_hts[4]), us(g_hts[4], tq0), us(g_hts[6], g_hts[5]));
        }
        if (tps) tp_pol_sync();   // [ink-tp-dec1] the row-budget policy of rank 0
        const auto tq1 = std::chrono::steady_clock::now();
        ICHECK(n >= 1 && n <= std::min(NDS, MROW), "spec_rows: %d slots", n);
        last_k.assign(n, 0);
        if (!dr_on || bpol == 0) {
            std::vector<int> o1(n);
            step_rows(sl, anch, n, o1.data());
            for (int z = 0; z < n; z++) {
                out[z * VR] = o1[z];
                nout[z] = 1;
            }
            return;
        }
        spos[cur] = pos;
        int maxpos = 0;
        for (int z = 0; z < n; z++) {
            const int ap = spos[sl[z]];
            hbd[z] = ap;
            hbd[16 + z] = sl[z];
            hbd[32 + z] = anch[z];
            memcpy(demb_h + (size_t)z * H, emb_host + (size_t)anch[z] * H, H * 2);
            maxpos = std::max(maxpos, ap);
        }
        set_stop(0.f, 0);
        const auto td = std::chrono::steady_clock::now();
        e_rows = e_rows_b[depth_bucket(maxpos)];
        draft_rows_run(n, maxpos);
        double dms = 1e3 * std::chrono::duration<double>(std::chrono::steady_clock::now() - td).count();
        if (tps) tp_bcast_draft(&dms, false, n);   // the rows' tokens come from hdr (VTOK)
        const auto tq2 = std::chrono::steady_clock::now();
        // [ink-n4-hosthop] no hand-off of rank 0's drafter time: the budget prices the drafter by its EMA before this
        // block (the same on every rank: copied state); 1 ms before the first measurement
        const double dpol = tp_hh() ? (e_dr[n] > 0 ? e_dr[n] : 1.0) : dms;
        Impl::ema(e_dr[n], dms, 0.2);
        bs_dms += dms;
        bs_dr++;
        std::vector<int> k(n, 0);   // drafts per slot
        // nodraft[z]: one row, no drafts (a slot whose token the caller picks from row_logits)
        // (review M2: a filtered slot whose rows often go to the host pick steps one row, no drafts)
        auto room = [&](int z) {
            if ((nodraft && nodraft[z]) || (sl[z] < (int)fb_ema.size() && fb_ema[sl[z]] > FB_NODRAFT)) return 0;
            return std::max(0, std::min(VR - 1, slim[sl[z]] - spos[sl[z]] - 1));
        };
        if (bpol == 1) {
            for (int z = 0; z < n; z++) {
                int m = 0;
                float pr = 1.f;
                while (m < VR - 1) {
                    pr *= hdp[z * 16 + 1 + m];
                    if (pr < tau) break;
                    m++;
                }
                k[z] = std::min(std::max(std::min(kmin, VR) - 1, m), room(z));
            }
            for (int R = n + std::accumulate(k.begin(), k.end(), 0); R > MROW; R--) (*std::max_element(k.begin(), k.end()))--;
        } else {
            std::vector<std::tuple<float, int, int>> cand;
            for (int z = 0; z < n; z++) {
                float c = 1.f;
                for (int j = 0; j < room(z); j++) {
                    c *= dcal(hdp[z * 16 + 1 + j]);
                    cand.emplace_back(c, z, j);
                }
            }
            std::sort(cand.begin(), cand.end(), [](auto& a, auto& b) { return std::get<0>(a) > std::get<0>(b); });
            double e = n, best = n / (t_rows(n) + dpol);
            int bk = 0;
            for (int i = 0; i < (int)cand.size() && n + i + 1 <= MROW; i++) {
                e += std::get<0>(cand[i]);
                const double r = e / (t_rows(n + i + 1) + dpol);
                if (r > best) {
                    best = r;
                    bk = i + 1;
                }
            }
            for (int i = 0; i < bk; i++) k[std::get<1>(cand[i])] = std::max(k[std::get<1>(cand[i])], std::get<2>(cand[i]) + 1);
        }
        std::vector<int> nr(n), toks, arg(MROW), acc(n);
        last_k = k;
        for (int z = 0; z < n; z++) {
            nr[z] = 1 + k[z];
            toks.push_back(anch[z]);
            for (int j = 0; j < k[z]; j++) toks.push_back(hdr[z * DB + 1 + j]);
        }
        const int R = (int)toks.size();
        const auto tv = std::chrono::steady_clock::now();
        bs_tv = tv;
        step_rows_k(sl, nr.data(), toks.data(), n, arg.data(), acc.data());
        const double vms = 1e3 * std::chrono::duration<double>(std::chrono::steady_clock::now() - tv).count();
        if (!rr_cold) Impl::ema(e_rows[R], vms, 0.2);   // [ink-tp-vseen] steady-state times only
        if (bs_dbg && (!tps || sh.rank == 0)) {
            auto us = [](auto a, auto b) { return 1e6 * std::chrono::duration<double>(b - a).count(); };
            fprintf(stderr, "[bsdbg] R %d pol %.0f draft %.0f bcast+pol %.0f verify %.0f us cold %d\n", R, us(tq0, tq1), dms * 1e3,
                    us(tq2, tv) + us(td, tq2) - dms * 1e3, vms * 1e3, (int)rr_cold);
            for (int z = 0; z < n; z++)
                fprintf(stderr, "[bsdbgp] k %d acc %d p %.4f %.4f %.4f %.4f %.4f %.4f %.4f\n", k[z], acc[z], hdp[z * 16 + 1], hdp[z * 16 + 2],
                        hdp[z * 16 + 3], hdp[z * 16 + 4], hdp[z * 16 + 5], hdp[z * 16 + 6], hdp[z * 16 + 7]);
        }
        bs_vms += vms;
        bs_steps++;
        bs_rows += R;
        for (int z = 0, f = 0; z < n; f += nr[z], z++) {
            nout[z] = acc[z] + 1;
            bs_tok += acc[z] + 1;
            for (int q = 0; q <= acc[z]; q++) out[z * VR + q] = arg[f + q];
        }
        if (bs_dbg && (!tps || sh.rank == 0)) hts(2);
    }
};

// [ink-tp-dec1] TP: the outer engine (owns the ranks) runs every call on all ranks, one worker thread per rank; the
// workers call the same Engine methods on non-owning views of their rank
static TpCtx* tp_of(Engine::Impl* p);
static std::vector<Engine*>& tp_views(TpCtx* t);
Engine::Engine(Impl* p) : p_(p), own_(false) {}
static std::vector<int> csv_ints(const char* s) {
    std::vector<int> v;
    for (const char* c = s; c && *c;) {
        v.push_back(atoi(c));
        c = strchr(c, ',');
        if (c) c++;
    }
    return v;
}
static void impl_init(Engine::Impl* p_, const Opts& o, int dev);
Engine::Engine(const Opts& o) : p_(nullptr) {
    std::vector<int> devs = o.devices;
    if (devs.empty() && getenv("QE_DEVICES")) devs = csv_ints(getenv("QE_DEVICES"));
    if (devs.size() > 1) {
        tp_init(o, devs);
        return;
    }
    // N = 1: the planner's shard is the whole model, i.e. the one-card engine itself (no ranks, no collectives)
    if (!devs.empty()) fprintf(stderr, "[ink] TP 1 rank on device %d: the one-card engine\n%s", devs[0], tp_plan_str(tp_plan(1)).c_str());
    p_ = new Impl;
    impl_init(p_, o, devs.empty() ? 0 : devs[0]);
}
static void impl_init(Engine::Impl* p_, const Opts& o, int dev) {
    p_->o = o;
    p_->dev = dev;
    CU(cudaSetDevice(dev));
    // [ink-tp-pf] else batched TP prefill (decode rows only as the fallback); review tp2 M2: the CUDA-core prefill
    // attention (INK_PFATT=0) and its precision check (INK_PATTN_CMP) are one-card forms: decode rows in TP
    if (p_->tps && (!p_->tpc->pf_ok || !p_->pfat || p_->pcmp)) p_->pfx = false;
    // [ink-tp-pf2] TP prefill as two half-chunks on two streams (INK_TPPF2=0: one stream)
    p_->pf2 = p_->tps && p_->pfx && !(getenv("INK_TPPF2") && atoi(getenv("INK_TPPF2")) == 0);
    if (const char* e = getenv("INK_TPPFK")) p_->pfk = std::max(2, std::min((int)Engine::Impl::PFK_MAX, atoi(e)));
    CU(cudaStreamCreateWithFlags(&p_->st, cudaStreamNonBlocking));
    if (p_->tpcc_chk) {   // review7 M1: check buffers before any capture
        CU(cudaMalloc(&p_->tpcc_buf, (size_t)2 * MROW * H * 4));
        CU(cudaMallocManaged(&p_->tpcc_bad, 3 * 8));
        p_->tpcc_bad[0] = p_->tpcc_bad[1] = p_->tpcc_bad[2] = 0;
    }
    CU(cudaEventCreateWithFlags(&p_->ev_stage, cudaEventDisableTiming));
    CB(cublasCreate(&p_->blas));
    CB(cublasSetStream(p_->blas, p_->st));
    CB(cublasSetMathMode(p_->blas, CUBLAS_DEFAULT_MATH));
    if (!p_->tps || o.drafter || p_->pfx) {   // fixed workspace: cuBLAS calls inside the step graph must not allocate ([ink-tp-dec1] TP ranks: the drafter only)
        void* ws = dmalloc(32 << 20, &p_->dev_bytes);
        CB(cublasSetWorkspace(p_->blas, ws, 32 << 20));
    }
    p_->keep16 = getenv("INK_BF16KEEP") != nullptr;
    p_->dr_on = o.drafter;
    if (Engine::Impl::drafter_qe(Engine::Impl::drafter_file(o.model_dir))) {   // [ink-dr-drop] the trained drafter's policy (env overrides below)
        p_->tau = Engine::Impl::TAU_QE;
        memcpy(p_->dcal_c, Engine::Impl::DCAL_QE, sizeof p_->dcal_c);
    }
    if (const char* e = getenv("INK_TAU")) p_->tau = (float)atof(e);
    if (const char* e = getenv("INK_KMIN")) p_->kmin = atoi(e);
    if (const char* e = getenv("INK_TAU1")) p_->tau1 = (float)atof(e);
    if (const char* e = getenv("INK_DCAL")) {   // [ink-dr-drop] 20 values
        const char* c = e;
        for (int i = 0; i < 20 && c && *c; i++) {
            p_->dcal_c[i] = (float)atof(c);
            c = strchr(c, ',');
            if (c) c++;
        }
    }
    if (const char* e = getenv("INK_LK")) p_->lk_min = atoi(e);
    CU(cudaHostAlloc((void**)&p_->hres, 2 * MROW * 4, 0));
    CU(cudaHostAlloc((void**)&p_->hdr, 1024, 0));
    CU(cudaHostAlloc((void**)&p_->hdp, 1024, 0));
    if (const char* e = getenv("INK_KPS")) p_->kps = atoi(e);
    if (const char* e = getenv("INK_AB")) p_->ab = atoi(e);   // development: run a variant everywhere
    p_->S = std::max(1, o.slots);
    p_->load();
    p_->alloc_state();
    for (int a = 0; a < NAPP; a++) p_->A[a] = p_->view(a, 0);
    p_->reset();
    CU(cudaStreamSynchronize(p_->st));
    if (!p_->tps) p_->pf_warm();   // [ink-pf-warm] (TP ranks: the first prefill warms; pf_warm has no collectives)
}
struct TpOwner {   // what the outer TP engine owns besides the ranks
    std::vector<Engine*> views;
};
static std::map<TpCtx*, TpOwner> g_tp_owner;
static std::mutex g_tp_mu;
static TpCtx* tp_of(Engine::Impl* p) { return p ? p->tpc : nullptr; }
static std::vector<Engine*>& tp_views(TpCtx* t) {
    std::lock_guard<std::mutex> l(g_tp_mu);
    return g_tp_owner[t].views;
}
void Engine::tp_init(const Opts& o, const std::vector<int>& devs) {
    const int n = (int)devs.size();
    ICHECK(n >= 1 && n <= TP_MAX, "TP: %d ranks (1..%d)", n, TP_MAX);
    TpCtx* t = new TpCtx;
    t->N = n;
    t->devs = devs;
    std::vector<double> w = o.tp_w;
    if (w.empty() && getenv("QE_TP_W"))
        for (const char* c = getenv("QE_TP_W"); c && *c;) {
            w.push_back(atof(c));
            c = strchr(c, ',');
            if (c) c++;
        }
    ICHECK(w.empty() || (int)w.size() == n, "QE_TP_W: %zu weights for %d ranks", w.size(), n);
    for (double x : w) ICHECK(std::isfinite(x) && x > 0, "QE_TP_W: weight %g (every card needs a finite weight > 0)", x);   // review tp L1
    t->plan = tp_plan(n, w);
    for (int d : devs) t->per_dev[d]++;
    const bool same = t->per_dev.size() == 1, distinct = (int)t->per_dev.size() == n;
    ICHECK(same || distinct, "TP: ranks must be all on one device (virtual) or on distinct devices");
    if (distinct && n > 1) {
        for (int a : devs)
            for (int b : devs) {
                if (a == b) continue;
                int ok = 0;
                CU(cudaDeviceCanAccessPeer(&ok, a, b));
                ICHECK(ok, "TP: no peer access %d -> %d (the host-memory transport is not built yet)", a, b);
            }
        t->push = true;
    } else
        t->push = n == 1;
    fprintf(stderr, "[ink] TP %d ranks on devices", n);
    for (int d : devs) fprintf(stderr, " %d", d);
    fprintf(stderr, " (%s)\n%s", n == 1 ? "one rank" : t->push ? "P2P push collectives" : "virtual: one device, direct collectives",
            tp_plan_str(t->plan).c_str());
    t->file.open(o.model_dir + "/model-00001-of-00001.safetensors");
    for (int k = 0; k < n; k++) {
        Impl* r = new Impl;
        r->tpc = t;
        r->sh = t->plan[k];
        r->sh.dev = devs[k];
        r->tps = !r->sh.full();
        t->R.push_back(r);
        // [ink-tp-pf] the grouped expert GEMM tiles: 2 x columns % 128 (gate / up rows), columns % 64 (down K)
        const int ni = r->sh.nu() * TP_UNIT;
        if (ni && ((2 * ni) % pf::XBM || ni % pf::XKS)) t->pf_ok = false;
    }
    if (!t->pf_ok) fprintf(stderr, "[ink] WARNING: TP %d: an expert share does not fit the prefill GEMM tiles: prefill runs as decode rows (slow)\n", n);
    t->bar.n = n;
    t->pool.bar = &t->bar;
    CU(cudaHostAlloc((void**)&t->abort_h, 64, cudaHostAllocMapped | cudaHostAllocPortable));   // review tp M2
    *(volatile unsigned*)t->abort_h = 0;
    t->abort_d = t->abort_h;   // UVA: the host pointer is valid on every device
    t->pool.abort_word = t->abort_h;
    t->pool.start(n, [t](int k) {
        CU(cudaSetDevice(t->devs[k]));
        if (t->push && t->N > 1)
            for (int g = 0; g < t->N; g++)
                if (g != k) {
                    const cudaError_t e = cudaDeviceEnablePeerAccess(t->devs[g], 0);
                    if (e == cudaErrorPeerAccessAlreadyEnabled) cudaGetLastError();
                    else CU(e);
                }
    });
    t->pool.run([&](int k) { impl_init(t->R[k], o, devs[k]); });
    auto& v = tp_views(t);
    for (int k = 0; k < n; k++) v.push_back(new Engine(t->R[k]));
    p_ = t->R[0];
}
// review tp M2: a failed call (one rank threw; the abort word made the peers' device waits leave): every rank drains
// its stream, the collective words and step counters are cleared on every rank (the wrap-clear state), the abort word
// is reset, then the error goes to the caller. A sticky CUDA error stays an error (the drain throws).
static void tp_recover(TpCtx* t) {
    t->pool.run([&](int k) {
        Engine::Impl& q = *t->R[k];
        try {
            CU(cudaStreamSynchronize(q.st));
            for (int j = 1; j < q.pfk; j++)   // review tp2 H1: the prefill sub-chunk streams too
                if (q.st2[j]) CU(cudaStreamSynchronize(q.st2[j]));
            for (int j = 1; j < q.pfk; j++)
                if (q.cps2[j]) CU(cudaStreamSynchronize(q.cps2[j]));
            if (q.cps) CU(cudaStreamSynchronize(q.cps));
        } catch (const std::exception& e) {
            fprintf(stderr, "[ink] TP rank %d drain: %s\n", k, e.what());
        }
        // review tp2 H1 / L2: the collective slot parities and the stress-gate counters equal on every rank again
        for (int j = 0; j < Engine::Impl::PS_MAX; j++) q.tpsl_[j] = q.agsl_[j] = 0;
        q.rs_last = q.ag_last = 0;
        q.pf_sub = q.pf_r0 = 0;
        q.tchk_calls = 0;
        if (q.tchk) cudaMemset(q.tchk, 0, 1024);
        q.pf_half = -1;
        q.sp = false;
        q.spm = 0xff;
    });
    t->pool.run([&](int k) { t->R[k]->tp_reset_words(); });
    *(volatile unsigned*)t->abort_h = 0;
}
template <class F>
static void tp_run(TpCtx* t, F f) {
    try {
        t->pool.run(f);
    } catch (...) {
        if (*(volatile unsigned*)t->abort_h) tp_recover(t);
        throw;
    }
}
// run f(view of rank k) on every rank's worker; returns rank 0's value, checks that every rank agrees
template <class F>
static int tp_all_i(TpCtx* t, F f, const char* what) {
    std::vector<int> r(t->N);
    auto& v = tp_views(t);
    tp_run(t, [&](int k) { r[k] = f(*v[k], k); });
    for (int k = 1; k < t->N; k++) ICHECK(r[k] == r[0], "TP: rank %d %s %d != rank 0's %d", k, what, r[k], r[0]);
    return r[0];
}
// [ink-n4-hosthop] the spec policy's host state of rank 0 on every rank, between two pool runs (no rank is running):
// replaces tp_pol_sync's two barriers inside the call
static void tp_pol_copy(TpCtx* t) {
    if (t->R[0]->bs_dbg && t->N > 1 && t->R[0]->bs_steps) {
        auto us = [](auto a, auto b) { return 1e6 * std::chrono::duration<double>(b - a).count(); };
        fprintf(stderr, "[bsdbg] rank 1 - rank 0: entry %+.0f us, verify start %+.0f us\n", us(t->R[0]->bs_te, t->R[1]->bs_te),
                us(t->R[0]->bs_tv, t->R[1]->bs_tv));
    }
    if (!t->R[0]->tp_hh()) return;
    for (int k = 1; k < t->N; k++) t->R[k]->tp_pol_from(*t->R[0]);
}
template <class F>
static void tp_all_v(TpCtx* t, F f) {
    auto& v = tp_views(t);
    tp_run(t, [&](int k) { f(*v[k], k); });
}
#define TP_OUTER (own_ && p_ && p_->tpc)
#define TP_NOT(what) ICHECK(!p_->tpc, "TP: %s is not in the TP build yet", what)
Engine::~Engine() {
    if (getenv("INK_RGSTAT") && (p_->rg_n[0] + p_->rg_n[1] + p_->rg_n[2]))
        fprintf(stderr, "[ink] row steps: eager %lld captured %lld replayed %lld (graphs %zu); segments updated %lld instantiated %lld failed %lld; record %.1f ms, instantiate %.1f ms\n", p_->rg_n[0], p_->rg_n[1], p_->rg_n[2], p_->vgraph.size(),
                p_->seg_n[0], p_->seg_n[1], p_->seg_n[2], p_->rg_ms[0], p_->rg_ms[1]);
    if (p_->hh_chk && p_->hh_n) fprintf(stderr, "[ink] INK_HHCHK rank %d: %llu drafter blocks compared, %llu differ\n", p_->sh.rank, p_->hh_n, p_->hh_bad);
    if (p_->tpcc_bad) {
        CU(cudaDeviceSynchronize());
        fprintf(stderr, "[ink] tpcc check: %llu fused collectives compared, %llu mismatching values\n", p_->tpcc_bad[2], p_->tpcc_bad[0] + p_->tpcc_bad[1]);
    }
    if (p_->ustat)
        for (int f = 0; f < 2; f++) {
            fprintf(stderr, "[ink] verify expert union per application (%s):", f ? "MXFP8" : "NVFP4");
            for (int k = 1; k <= MROW; k++)
                if (p_->us_n[f][k] > 0) fprintf(stderr, " k%d %.2f (pairs %.2f, n %.0f)", k, p_->us_u[f][k] / p_->us_n[f][k], p_->us_p[f][k] / p_->us_n[f][k], p_->us_n[f][k]);
            fprintf(stderr, "\n");
        }
    if (!own_) return;   // a rank view
    if (TP_OUTER) {      // ranks: device memory goes with the process (the Impl has no destructor of its own)
        TpCtx* t = p_->tpc;
        tp_all_v(t, [](Engine& e, int) {
            e.sync();
            if (e.p_->rthr.joinable()) e.p_->rthr.join();
            e.p_->vpend_drain();   // [ink-vp-vasync] review L6: no helper thread past the engine
        });
        for (auto* e : tp_views(t)) delete e;
        delete t;
        return;
    }
    if (p_->rthr.joinable()) p_->rthr.join();   // [ink-srv-review-fix] M3 (a joinable std::thread in delete p_ is std::terminate)
    p_->vpend_drain();   // [ink-vp-vasync] review L6
    for (auto& kv : p_->vgraph) cudaGraphExecDestroy(kv.second);
    p_->vgraph.clear();
    p_->pf_free();
    if (p_->scr_chk)
        fprintf(stderr, "[scr-chk] %llu vocab rows checked: %llu exact logits above the upper end, %llu thresholds above the exact max, min margin hi - exact %.4g\n",
                p_->scr_rows, p_->scr_bad, p_->scr_thr_bad, p_->scr_minmargin);
    if (p_->BCHK) {
        unsigned h[67];
        CU(cudaMemcpy(h, p_->BCHK, sizeof h, cudaMemcpyDeviceToHost));
        unsigned long long n = 0, c = 0;
        for (int b = 0; b < 64; b++) n += h[2 + b];
        int med = -1;
        for (int b = 63; b >= 0 && med < 0; b--) {   // bins from the smallest errors up
            c += h[2 + b];
            if (c * 2 >= n) med = b;
        }
        float mx, mr;
        memcpy(&mx, h, 4);
        memcpy(&mr, h + 1, 4);
        fprintf(stderr, "[bres-chk] %llu norms (direct-sum fallbacks %u): max rel. error %.3g, median bin 2^-%d, min cancellation ratio %.3g; bins (2^-b: count)", n, h[66], mx, med, mr);
        for (int b = 0; b < 64; b++)
            if (h[2 + b]) fprintf(stderr, " %d:%u", b, h[2 + b]);
        fprintf(stderr, "\n");
    }
#ifdef BRES_PROF
    {
        unsigned long long acc[8][8], n[8][8];
        CU(cudaMemcpyFromSymbol(acc, bp_acc, sizeof acc));
        CU(cudaMemcpyFromSymbol(n, bp_n, sizeof n));
        const char* nm[8] = {"plain", "msgs", "router", "?", "zg", "?", "?", "?"};
        for (int k = 0; k < 8; k++)
            if (n[k][0]) {
                fprintf(stderr, "[bres] %-6s launches-blocks %llu: cycles at loads+dots %.0f, residual reduce %.0f, norm %.0f, z/router %.0f, end %.0f, last-block topk %.0f (n %llu)\n", nm[k], n[k][0],
                        (double)acc[k][5] / n[k][0], (double)acc[k][1] / n[k][0], (double)acc[k][2] / n[k][0], (double)acc[k][3] / n[k][0], (double)acc[k][0] / n[k][0],
                        n[k][4] ? (double)acc[k][4] / n[k][4] : 0., n[k][4]);
                fprintf(stderr, "[bres]        scalar done %.0f, after barrier %.0f\n", (double)acc[k][6] / n[k][0], (double)acc[k][7] / n[k][0]);
            }
    }
#endif
    if (p_->cmp_on && p_->cmptb3_n) fprintf(stderr, "[ink] k_attn_tb3 vs k_attn_tb: %lld / %lld outputs differ (bitwise)\n", p_->cmptb3_bad, p_->cmptb3_n);
    if (p_->cmp_on && p_->cmp3_n) fprintf(stderr, "[ink] k_attn_tc3 vs k_attn_tc<FP16-P>: %lld / %lld outputs differ (bitwise)\n", p_->cmp3_bad, p_->cmp3_n);
    if (p_->cmp_on)
        for (int v = 0; v < 4; v++) {
            if (p_->cmp_g[v].empty()) continue;
            auto a = p_->cmp_g[v], b = p_->cmp_gm[v];
            std::sort(a.begin(), a.end());
            std::sort(b.begin(), b.end());
            fprintf(stderr, "[ink] %s layers vs CUDA-core, %s (%zu calls): rel. L2 median %.3e p99 %.3e max %.3e | max|d|/max|ref| median %.3e max %.3e\n",
                    v < 2 ? "global" : "local", v % 2 == 0 ? "TC FP16-P (k_attn_tc, short form)" : "k_attn_tb (shared-row form)", a.size(), a[a.size() / 2], a[a.size() * 99 / 100], a.back(),
                    b[b.size() / 2], b.back());
        }
    p_->pf_report();
    if (p_->cmp_on && p_->cmp_ref > 0)
        fprintf(stderr, "[ink] attention vs CUDA-core (fp32 Q, P), relative L2 error: TC BF16-P %.3e, TC FP16-P %.3e, TC hi/lo %.3e\n",
                sqrt(p_->cmp_err[0] / p_->cmp_ref), sqrt(p_->cmp_err[1] / p_->cmp_ref), sqrt(p_->cmp_err[2] / p_->cmp_ref));
    delete p_;
}
void Engine::reset() {
    if (TP_OUTER) return tp_all_v(p_->tpc, [](Engine& e, int) { e.reset(); });
    p_->reset();
    CU(cudaStreamSynchronize(p_->st));
}
int Engine::pos() const { return p_->pos; }
int Engine::slots() const { return p_->S; }
void Engine::select(int slot) {
    if (TP_OUTER) return tp_all_v(p_->tpc, [=](Engine& e, int) { e.select(slot); });
    p_->select(slot);
}
// [ink-tp-dec1] run a row call on every rank with its own output arrays, check that they agree, return rank 0's
template <class F>
static void tp_rows(TpCtx* t, int nout, int* out, F f, const char* what) {
    std::vector<std::vector<int>> o(t->N, std::vector<int>(std::max(nout, 1)));
    tp_all_v(t, [&](Engine& e, int r) { f(e, r, o[r].data()); });
    for (int r = 1; r < t->N; r++) ICHECK(o[r] == o[0], "TP: rank %d %s differ", r, what);
    std::copy(o[0].begin(), o[0].begin() + nout, out);
}
int Engine::slot_pos(int slot) const { return slot == p_->cur ? p_->pos : p_->spos.at(slot); }
void Engine::step_rows(const int* slots, const int* toks, int n, int* out) {
    if (TP_OUTER) tp_rows(p_->tpc, n, out, [=](Engine& e, int, int* o) { e.step_rows(slots, toks, n, o); }, "step_rows tokens");
    else p_->step_rows(slots, toks, n, out);
    if (own_)   // [ink-samp-dev] a -2 (filtered row the device could not settle): the host pick of that row
        for (int i = 0; i < n; i++) {
            const bool fb = out[i] == -2;
            if (fb) out[i] = filt_host(slots[i], i);
            fb_note(slots[i], 1, fb);
        }
}
void Engine::spec_rows(const int* slots, const int* anchors, int n, int* out, int* nout, const unsigned char* nodraft) {
    if (TP_OUTER) {
        std::vector<int> no(n);
        if (p_->tpc->R[0]->bs_dbg) hts(4);
        tp_pol_copy(p_->tpc);   // [ink-n4-hosthop]
        tp_rows(p_->tpc, n * VR, out, [&](Engine& e, int r, int* o) {
            std::vector<int> nn(n);
            e.spec_rows(slots, anchors, n, o, nn.data(), nodraft);
            if (r == 0) no = nn;
            for (int i = 0; i < n; i++)   // emitted tokens only (the rest of each VR block is not written)
                for (int j = nn[i]; j < VR; j++) o[i * VR + j] = 0;
        }, "spec_rows tokens");
        std::copy(no.begin(), no.end(), nout);
        if (p_->tpc->R[0]->bs_dbg) hts(3);
    } else
        p_->spec_rows(slots, anchors, n, out, nout, nodraft);
    if (!own_) return;
    // [ink-samp-dev] a -2 is always a slot's last emitted token (a draft never equals it): the host pick of that row
    // (rows of the step: slot z holds 1 + last_k[z] rows from f_z)
    const Impl* q0 = p_->tpc ? p_->tpc->R[0] : p_;
    for (int z = 0, f = 0; z < n; z++) {
        const int kz = z < (int)q0->last_k.size() ? q0->last_k[z] : 0;
        const bool fb = nout[z] >= 1 && out[z * VR + nout[z] - 1] == -2;
        if (fb) out[z * VR + nout[z] - 1] = filt_host(slots[z], f + nout[z] - 1);
        fb_note(slots[z], nout[z], fb);
        f += 1 + kz;
    }
}
void Engine::last_spec_drafts(int* k) const { std::copy(p_->last_k.begin(), p_->last_k.end(), k); }
std::string Engine::spec_rows_stats(bool reset) {
    Impl& q = *p_;
    char b[256];
    snprintf(b, sizeof b, "steps %lld rows/step %.2f tok/step %.2f step %.2f ms drafter %.2f ms", q.bs_steps,
             q.bs_steps ? (double)q.bs_rows / q.bs_steps : 0., q.bs_steps ? (double)q.bs_tok / q.bs_steps : 0.,
             q.bs_steps ? q.bs_vms / q.bs_steps : 0., q.bs_dr ? q.bs_dms / q.bs_dr : 0.);
    if (reset) {
        q.bs_steps = q.bs_rows = q.bs_tok = q.bs_dr = 0;
        q.bs_dms = q.bs_vms = 0;
    }
    return b;
}
void Engine::set_bpol(int v) {
    if (TP_OUTER) return tp_all_v(p_->tpc, [=](Engine& e, int) { e.set_bpol(v); });
    p_->bpol = v;
}
void Engine::step_rows_k(const int* slots, const int* nrows, const int* toks, int n, int* arg, int* acc) {
    if (TP_OUTER) {
        int tot = 0;
        for (int i = 0; i < n; i++) tot += nrows[i];
        std::vector<int> ac(n);
        tp_rows(p_->tpc, tot, arg, [&](Engine& e, int r, int* o) {
            std::vector<int> a(n);
            e.step_rows_k(slots, nrows, toks, n, o, a.data());
            if (r == 0) ac = a;
        }, "step_rows_k tokens");
        std::copy(ac.begin(), ac.end(), acc);
    } else
        p_->step_rows_k(slots, nrows, toks, n, arg, acc);
    if (!own_) return;
    for (int i = 0, f = 0; i < n; f += nrows[i], i++) {   // [ink-samp-dev] the emitted last row of each slot
        const bool fb = arg[f + acc[i]] == -2;
        if (fb) arg[f + acc[i]] = filt_host(slots[i], f + acc[i]);
        fb_note(slots[i], acc[i] + 1, fb);
    }
}
int Engine::save_prefix() { TP_NOT("save_prefix"); return p_->pf_save(); }
bool Engine::restore_prefix(int id) { TP_NOT("restore_prefix"); return p_->pf_restore(id); }   // review tp H1
void Engine::prepare_host_copies(size_t bytes) {
    if (TP_OUTER) return tp_all_v(p_->tpc, [bytes](Engine& e, int) { e.prepare_host_copies(bytes); });   // [ink-tp-srv]
    p_->staging(bytes);
}
size_t Engine::archive_bytes() const { return p_->parch_bytes; }
void Engine::fake_depth(int n) {   // bench only: decode as if n tokens were in the cache (zero K / V, zero states)
    if (TP_OUTER) return tp_all_v(p_->tpc, [n](Engine& e, int) { e.fake_depth(n); });
    ICHECK(n >= 0 && n < p_->o.max_ctx, "fake depth %d >= max_ctx %d", n, p_->o.max_ctx);
    p_->reset();
    Impl& q = *p_;
    for (int a = 0; a < NAPP; a++) {   // zero K / V: reset() keeps the caches (a real context rewrites every slot it reads)
        if (q.tpc) {   // [ink-tp-dec1] the rank's K / V allocation
            CU(cudaMemsetAsync(q.A0[a].K0, 0, q.A0[a].kvb, q.st));
            CU(cudaMemsetAsync(q.A0[a].V0, 0, q.A0[a].kvb, q.st));
            continue;
        }
        CU(cudaMemsetAsync(q.A[a].K, 0, (size_t)NKV * q.A[a].cap * HD * 2, q.st));
        CU(cudaMemsetAsync(q.A[a].V, 0, (size_t)NKV * q.A[a].cap * HD * 2, q.st));
    }
    if (q.dr_on)
        for (int i = 0; i < 3; i++) {
            CU(cudaMemsetAsync(q.DK[i], 0, (size_t)NKV * q.dcap[i] * HD, q.st));
            CU(cudaMemsetAsync(q.DV[i], 0, (size_t)NKV * q.dcap[i] * HD, q.st));
            CU(cudaMemsetAsync(q.DKS[i], 0, (size_t)NKV * q.dcap[i] * 4, q.st));
            CU(cudaMemsetAsync(q.DVS[i], 0, (size_t)NKV * q.dcap[i] * 4, q.st));
        }
    p_->pos = n;
    CU(cudaStreamSynchronize(p_->st));
}
// prefill in chunks; hid (optional): final normed hidden fp32 for every token [n][H]
void Engine::prefill(const int* ids, int n, float* hid) {
    if (TP_OUTER) return tp_all_v(p_->tpc, [=](Engine& e, int k) { e.prefill(ids, n, k == 0 ? hid : nullptr); });
    if (p_->tps && !p_->pfx) {   // [ink-tp-dec1] fallback (expert shares outside the prefill GEMM tiles): decode steps
        for (int i = 0; i < n; i++) {
            p_->step(ids[i]);
            if (hid) {
                CU(cudaMemcpyAsync(hid + (size_t)i * H, p_->FIN, H * 4, cudaMemcpyDeviceToHost, p_->st));
                CU(cudaStreamSynchronize(p_->st));
            }
        }
        p_->last_T = 1;
        return;
    }
    // [ink-pf-cut] last application cut when only the last row matters (INK_NOCUT=1 off; INK_CUTTEST=1 also with
    // hidden rows requested: only the last row of the returned rows is valid then)
    static const bool nocut = getenv("INK_NOCUT") != nullptr, cuttest = getenv("INK_CUTTEST") != nullptr;
    // [ink-pf-cutdr] with the drafter: tail of W rows (INK_CUTW, default 1024; INK_NOCUTDR=1 off)
    static const bool nocutdr = getenv("INK_NOCUTDR") != nullptr;
    static const int cutw = getenv("INK_CUTW") ? std::max(16, atoi(getenv("INK_CUTW"))) : 1024;
    if (n <= 0) return;   // review2 M3
    p_->ring_finalize();   // [ink-srv-gap] (a pending snapshot's rows must be copied before a prefill writes rows)
    // [ink-tp-pf] TP ranks: no cut (layer_cut is the one-card form)
    const bool cutok = (!hid || cuttest) && !nocut && !(p_->dr_on && nocutdr) && !p_->dumping && p_->pfx && !p_->FMH && !p_->tps;   // M2: drafter tests read FMH
    p_->cut_s = p_->dr_on ? cutw : Engine::Impl::CUT_S;
    // [ink-pf-wave] up to 2 x capacity: one chunk or capacity + rest; longer: chunks of chs rows (wave-filling split
    // size), a remainder that fits the capacity runs as one chunk
    std::vector<int> lens;
    if (n > p_->ch && n <= 2 * p_->ch)
        lens = {p_->ch, n - p_->ch};   // capacity first: a prefix of capacity rows chunks like the cold run (restore == cold)
    else
        for (int i = 0; i < n;) {
            const int T = n - i <= p_->ch ? n - i : p_->chs;
            lens.push_back(T);
            i += T;
        }
    // the tail chunk (cut 2) needs >= 2 x CUT_S rows so its last row and end conv states see a computed history;
    // chunks before it: K / V only (cut 1); a short last chunk after it runs in full
    // review2 M1: with the drafter the tail chunk must hold W exact rows (+ CUT_S of history), else the tail goes on
    // the chunk before and the short last chunk runs in full (exact rows = W + last chunk)
    const int nch = (int)lens.size(), need = p_->dr_on ? p_->cut_s + Engine::Impl::CUT_S : 2 * Engine::Impl::CUT_S;
    const int ct = lens.back() >= need ? nch - 1 : nch - 2;
    for (int i = 0, c = 0; c < nch; i += lens[c], c++) {
        const int T = lens[c];
        p_->cut = !cutok || ct < 0 ? 0 : c < ct ? 1 : c == ct ? 2 : 0;
        struct CutReset {   // review2 L1: never leave a cut level set after an exception
            Engine::Impl* p;
            ~CutReset() { p->cut = 0; }
        } cr{p_};
        p_->forward(ids + i, T);
        if (hid) CU(cudaMemcpyAsync(hid + (size_t)i * H, p_->FIN, (size_t)T * H * 4, cudaMemcpyDeviceToHost, p_->st));
        p_->last_T = T;
    }
    p_->pf_last_row();   // review2 L3: FIN / FINB row 0 = the last row (after a cut prefill the other rows are not valid)
    CU(cudaStreamSynchronize(p_->st));
}
// a token outside [0, V) means non-finite logits (the head returns -1 on NaN): an error, never a token id downstream
static int checked_token(int id) {
    ICHECK(id >= 0 && id < V, "non-finite logits: head returned token %d (step aborted)", id);
    return id;
}
int Engine::argmax() {   // greedy (the sampled first token: sample_last)
    if (TP_OUTER) return tp_all_i(p_->tpc, [](Engine& e, int) { return e.argmax(); }, "argmax");
    p_->HSAMP[0] = 0.f;
    p_->samp_up();
    return checked_token(p_->argmax_last(p_->last_T));
}
void Engine::set_sampling(int slot, float temperature, unsigned long long seed) {
    if (TP_OUTER) return tp_all_v(p_->tpc, [=](Engine& e, int) { e.set_sampling(slot, temperature, seed); });
    Impl& q = *p_;
    q.sT.at(slot) = temperature > 0.f ? 1.f / temperature : 0.f;
    q.sS.at(slot) = (unsigned)(seed ^ (seed >> 32));
}
void Engine::set_filter(int slot, int top_k, float top_p, float min_p) {
    if (TP_OUTER) return tp_all_v(p_->tpc, [=](Engine& e, int) { e.set_filter(slot, top_k, top_p, min_p); });
    Impl& q = *p_;
    int K = 0;
    // top_k 2..64: K; no top_k: the 64-list, certified by min_p (its set inside the list) or, top_p alone, by the tail bound
    if (filter_on_device(top_k, top_p, min_p)) K = top_k >= 2 ? top_k : min_p > 0.f ? (FK_MAX | FK_CERT) : (FK_MAX | FK_ZT);
    if (top_k == 1 || (!K && (top_k > 0 || (top_p > 0.f && top_p < 1.f) || (min_p > 0.f && min_p < 1.f)))) {   // review L8
        static std::atomic<bool> said{false};   // (review2 L6: TP rank threads)
        if (!said.exchange(true)) fprintf(stderr, "[ink] set_filter: top_k %d top_p %.3f min_p %.3f is not a device filter: this slot samples without it (the C API picks such tokens on the host)\n", top_k, top_p, min_p);
    }
    q.sK.at(slot) = q.loR ? K : 0;
    q.fb_ema.at(slot) = 0.f;
    q.sTP.at(slot) = top_p > 0.f && top_p < 1.f ? top_p : 1.f;
    q.sMP.at(slot) = min_p > 0.f && min_p < 1.f ? min_p : 0.f;
}
// review M2: the per-slot host-pick rate of filtered slots (owner side, rank 0's state; copied to the ranks with the
// policy state): an EMA over the slot's emitted tokens, a pick counts 1
void Engine::fb_note(int slot, int ntok, bool fb) {
    Impl* q = p_;
    if (slot < 0 || slot >= (int)q->fb_ema.size() || !(q->sK[slot] & 0xff) || !(q->sT[slot] > 0.f) || ntok <= 0) return;
    q->filt_tok += ntok;
    float& e = q->fb_ema[slot];
    for (int i = 0; i < ntok; i++) e = 0.9f * e + (fb && i == ntok - 1 ? 0.1f : 0.f);
}
bool Engine::filter_ready() const {
    const Impl* q = p_->tpc ? p_->tpc->R[0] : p_;
    return q->loR != nullptr;
}
long long Engine::filter_fallbacks() const { return p_->filt_fb; }
std::string Engine::filter_stats() const {   // candidates per filtered row (rank 0): < 64 / 256 / 1K / 4K / FCAP / overflow
    const Impl* q = p_->tpc ? p_->tpc->R[0] : p_;
    if (!q->fcR) return "off";
    std::vector<unsigned> f(MROW * 16);
    int d0 = 0;
    CU(cudaGetDevice(&d0));
    CU(cudaSetDevice(q->dev));
    CU(cudaStreamSynchronize(q->st));   // review L6
    CU(cudaMemcpy(f.data(), q->fcR, f.size() * 4, cudaMemcpyDeviceToHost));
    CU(cudaSetDevice(d0));
    unsigned long long b[6] = {};   // (32-bit counters per row: statistics only, they wrap after 4G rows)
    for (int j = 0; j < MROW; j++)
        for (int i = 0; i < 6; i++) b[i] += f[j * 16 + 8 + i];
    char o[256];
    snprintf(o, sizeof o, "candidates <64 %llu, <256 %llu, <1K %llu, <4K %llu, <=%d %llu, overflow %llu", b[0], b[1], b[2], b[3], FCAP, b[4], b[5]);
    return o;
}
// [ink-samp-dev] host pick of a filtered row the device could not settle (token -2: candidate list overflow, uncertified
// min_p set): the full head of the row, then the device rule (top_k, min_p, top_p; Gumbel-max with the same noise)
int Engine::filt_host(int slot, int row) {
    const Impl* q = p_->tpc ? p_->tpc->R[0] : p_;
    const float iT = q->sT.at(slot);
    const unsigned sd = q->sS.at(slot);
    const int Kf = q->sK.at(slot), K = Kf & 0xff;
    const float tp = q->sTP.at(slot), mp = q->sMP.at(slot);
    const int ps = slot_pos(slot) - 1;   // the row's position (the step advanced the slot past it)
    std::vector<float> x(V);
    row_logits(row, x.data());
    p_->filt_fb++;
    if (p_->filt_fb % 100 == 0)
        fprintf(stderr, "[ink] filtered sampling: %lld rows picked on the host of %lld filtered tokens\n", p_->filt_fb, p_->filt_tok);
    // review M1: non-finite logits fault as on the greedy / host paths (-1); NaN rows sort last (a strict weak order)
    for (float& v : x)
        if (!std::isfinite(v)) v = -INFINITY;
    std::vector<int> idx(V);
    std::iota(idx.begin(), idx.end(), 0);
    auto higher = [&](int a, int b) { return x[a] > x[b] || (x[a] == x[b] && a < b); };
    const bool all = (Kf & (FK_CERT | FK_ZT)) != 0;
    const int mx = (int)(std::max_element(x.begin(), x.end()) - x.begin());
    if (!std::isfinite(x[mx])) return -1;
    // top_p alone: the normaliser of the whole vocabulary (its kept set is a prefix of the sorted rows)
    double zall = 0;
    if (Kf & FK_ZT)
        for (int v = 0; v < V; v++) zall += std::exp((double)(x[v] - x[mx]) * (double)iT);
    std::vector<double> qv;
    size_t m = 0;
    // the largest t rows in order (t = K, or 256, 1,024, .. without top_k: nth_element + sort of the head only)
    for (size_t t = all ? 256 : (size_t)std::min(K, V);; t *= 4) {
        const size_t tt = std::min(t, (size_t)V);
        std::iota(idx.begin(), idx.end(), 0);
        if (tt < (size_t)V) std::nth_element(idx.begin(), idx.begin() + tt, idx.end(), higher);
        std::sort(idx.begin(), idx.begin() + tt, higher);
        qv.resize(tt);
        for (size_t i = 0; i < tt; i++) qv[i] = std::exp((double)(x[idx[i]] - x[idx[0]]) * (double)iT);
        m = tt;
        bool done = !all || tt == (size_t)V;
        if (mp > 0.f) {
            m = 1;
            while (m < tt && qv[m] >= (double)mp) m++;
            done = done || m < tt;
        }
        if (tp < 1.f && done) {
            double sum = 0, acc = 0;
            if (Kf & FK_ZT) sum = zall;
            else
                for (size_t i = 0; i < m; i++) sum += qv[i];
            size_t u = 0;
            while (u < m && acc < (double)tp * sum) acc += qv[u++];
            m = std::max<size_t>(u, 1);
        } else if (tp < 1.f && (Kf & FK_ZT) && mp <= 0.f) {   // top_p alone: done once the prefix reaches top_p of Z
            double acc = 0;
            size_t u = 0;
            while (u < tt && acc < (double)tp * zall) acc += qv[u++];
            if (acc >= (double)tp * zall) {
                m = std::max<size_t>(u, 1);
                done = true;
            }
        }
        if (done) break;
    }
    auto gum = [](uint32_t seed, int step, int r) {   // == the device gumbel()
        uint32_t h = seed ^ (uint32_t)step * 0x9E3779B9u ^ (uint32_t)r * 0x85EBCA6Bu;
        h ^= h >> 16; h *= 0x7feb352du; h ^= h >> 15; h *= 0x846ca68bu; h ^= h >> 16;
        const float u = ((h >> 9) + 0.5f) * (1.f / 8388608.f);
        return -logf(-logf(u));
    };
    float bv = -INFINITY;
    int br = idx[0];
    for (size_t i = 0; i < m; i++) {
        const float v = x[idx[i]] * iT + gum(sd, ps, idx[i]);
        if (v > bv || (v == bv && idx[i] < br)) {
            bv = v;
            br = idx[i];
        }
    }
    return br;
}
int Engine::sample_last() {   // after a prefill: the next token with the selected slot's sampling (noise at position pos - 1)
    if (own_) {   // [ink-samp-dev] the owner (one card / TP outer) settles a -2 on the host
        int t;
        if (TP_OUTER) t = tp_all_i(p_->tpc, [](Engine& e, int) { return e.sample_last_raw(); }, "sampled token");
        else t = sample_last_raw();
        return t == -2 ? checked_token(filt_host((p_->tpc ? p_->tpc->R[0] : p_)->cur, 0)) : t;
    }
    return sample_last_raw();
}
int Engine::sample_last_raw() {
    Impl& q = *p_;
    if (!(q.sT[q.cur] > 0.f)) return argmax();
    q.pf_last_row();
    q.samp_row(0, q.cur);
    q.sampP(q.HSAMP)[0] = q.pos - 1;
    q.samp_up();
    q.tp_head_begin();   // review tp H2
    q.head(1, q.sampP(q.DSAMP));
    CU(cudaStreamSynchronize(q.st));
    return q.hpos[1] == -2 ? -2 : checked_token(q.hpos[1]);
}
void Engine::row_logits(int row, float* out) {   // full head of FINB row `row` (capped logits), e.g. host-side filtered sampling
    if (TP_OUTER) return tp_all_v(p_->tpc, [=](Engine& e, int) { e.row_logits(row, out); });   // [ink-tp-srv] each rank its vocab rows
    Impl& q = *p_;
    const int V0 = q.tps ? q.sh.v0 : 0, NV = q.tps ? q.sh.v1 - q.sh.v0 : V;   // head12 holds rows [V0, V0 + NV)
    ICHECK(row >= 0 && row < MROW, "row_logits row %d", row);
    if (q.last_T > 1 && row == 0) q.pf_last_row();
    G12Args ga = q.head12;
    GemvArgs& g = ga.a;
    g.W = q.unemb;
    g.x0 = g.x1 = g.x2 = q.FINB + (size_t)row * H;
    g.N = NV;
    g.r1 = g.r2 = NV;
    g.y = q.LOG;
    g.bmax = q.BMAX;
    const int nb = (NV + 15) / 16;
    q.go(k_gemv12<H, true>, 0, dim3(nb), 256, A_gemv12{ga});
    CU(cudaMemcpyAsync(out + V0, q.LOG, (size_t)NV * 4, cudaMemcpyDeviceToHost, q.st));
    CU(cudaStreamSynchronize(q.st));
    for (int v = V0; v < V0 + NV; v++) out[v] = 30.f * tanhf(out[v] * (1.f / 30.f));   // the device head_cap expression (review2 L1)
}
int Engine::step(int id) {
    int t;
    if (TP_OUTER) t = tp_all_i(p_->tpc, [id](Engine& e, int) { return e.step(id); }, "step token");
    else {
        p_->last_T = 1;
        t = p_->step(id);
    }
    if (t == -2) return own_ ? checked_token(filt_host((p_->tpc ? p_->tpc->R[0] : p_)->cur, 0)) : -2;   // [ink-samp-dev]
    return checked_token(t);
}
// development: counts of V cache values outside fp16's normal range (|v| < 2^-14, nonzero; |v| > 65504)
void Engine::v_stats(long long* tot, long long* sub, long long* zero, long long* big) {
    TP_NOT("v_stats");   // review tp H1 (walks NKV heads)
    for (int ap = 0; ap < NAPP; ap++) {
        const AppState& s = p_->A[ap];
        const int n = s.ring ? std::min(p_->pos, s.cap) : p_->pos;
        std::vector<uint16_t> v((size_t)NKV * s.cap * HD);
        CU(cudaMemcpy(v.data(), s.V, v.size() * 2, cudaMemcpyDeviceToHost));
        for (int h = 0; h < NKV; h++)
            for (int k = 0; k < n; k++)
                for (int d = 0; d < HD; d++) {
                    const float x = fabsf(bf2f(v[((size_t)h * s.cap + k) * HD + d]));
                    (*tot)++;
                    if (x == 0.f) (*zero)++;
                    else if (x < 6.103515625e-05f) (*sub)++;
                    else if (x > 65504.f) (*big)++;
                }
    }
}
void Engine::last_hidden(float* out) {   // final normed hidden (fp32) of the last decode step
    if (TP_OUTER) return tp_all_v(p_->tpc, [out](Engine& e, int k) { if (k == 0) e.last_hidden(out); });
    CU(cudaMemcpyAsync(out, p_->FIN, H * 4, cudaMemcpyDeviceToHost, p_->st));
    CU(cudaStreamSynchronize(p_->st));
}
void Engine::row_hidden(int row, float* out) {   // [ink-tp-tcx] tests: final normed hidden (fp32) of a row of the last row step
    if (TP_OUTER) return tp_all_v(p_->tpc, [row, out](Engine& e, int k) { if (k == 0) e.row_hidden(row, out); });
    ICHECK(row >= 0 && row < MROW, "row_hidden row %d", row);
    CU(cudaMemcpyAsync(out, p_->FIN + (size_t)row * H, H * 4, cudaMemcpyDeviceToHost, p_->st));
    CU(cudaStreamSynchronize(p_->st));
}
void Engine::set_dump(bool on) {
    if (TP_OUTER) return tp_all_v(p_->tpc, [on](Engine& e, int) { e.set_dump(on); });   // review tp H1: every rank (same step path)
    p_->dumping = on;
    p_->dump.clear();
    p_->route.clear();
}
const std::vector<int>& Engine::routed() const { return p_->route; }
const std::vector<float>& Engine::dumped() const { return p_->dump; }
void Engine::sync() {
    if (TP_OUTER) return tp_all_v(p_->tpc, [](Engine& e, int) { e.sync(); });
    CU(cudaStreamSynchronize(p_->st));
}
void Engine::set_ab(int v) {
    if (TP_OUTER) return tp_all_v(p_->tpc, [v](Engine& e, int) { e.set_ab(v); });   // review tp H1
    p_->ab = v;
    if (p_->gstep) {
        CU(cudaGraphExecDestroy(p_->gstep));
        p_->gstep = nullptr;
    }
    for (auto& kv : p_->vgraph) CU(cudaGraphExecDestroy(kv.second));   // review L3: verify / drafter graphs too
    p_->vgraph.clear();
    p_->vpend_drain();   // [ink-vp-vasync]
    if (p_->dgraph) {
        CU(cudaGraphExecDestroy(p_->dgraph));
        p_->dgraph = nullptr;
    }
}

}  // namespace ink

namespace ink {
int Engine::max_ctx() const { return p_->o.max_ctx; }
int Engine::pool() const { return p_->POOL; }
// review tp H1: the pool / ring methods walk all NKV heads and change one rank's slot state: not in TP yet (a server
// step will dispatch them per rank over the rank's heads [kv0, kv0 + nkv))
// [ink-tp-srv] TP: every rank the same call on its own heads (pool rows / rings of its K / V heads; conv states and the
// drafter replicated). Blobs: rows_save = the ranks' blobs back to back (rank r: pool_row_bytes(r) x n bytes; a blob of
// m rows splits by the ranks' row bytes); ring_save = a header ("TPRB", N, N sizes) + the ranks' blobs.
static constexpr uint32_t TP_RING_MAGIC = 0x42525054u;
void Engine::bind(int slot, int base, int pos, int lim) {
    if (TP_OUTER) return tp_all_v(p_->tpc, [=](Engine& e, int) { e.bind(slot, base, pos, lim); });
    p_->bind(slot, base, pos, lim);
}
void Engine::copy_rows(int src, int dst, int n) {
    if (TP_OUTER) return tp_all_v(p_->tpc, [=](Engine& e, int) { e.copy_rows(src, dst, n); });
    p_->copy_rows(src, dst, n);
}
void Engine::rows_save(int src, int n, std::vector<unsigned char>& out) const {
    if (TP_OUTER) {
        TpCtx* t = p_->tpc;
        std::vector<size_t> off(t->N + 1, 0);
        for (int k = 0; k < t->N; k++) off[k + 1] = off[k] + t->R[k]->pool_row_bytes() * (size_t)std::max(n, 0);
        out.resize(off[t->N]);
        unsigned char* o = out.data();
        return tp_all_v(t, [&](Engine& e, int k) { e.p_->rows_save_raw(src, n, o + off[k]); });
    }
    p_->rows_save(src, n, out);
}
void Engine::rows_load(const std::vector<unsigned char>& in, int dst, int n) {
    if (TP_OUTER) {
        TpCtx* t = p_->tpc;
        size_t per = 0;
        std::vector<size_t> pr(t->N);
        for (int k = 0; k < t->N; k++) per += pr[k] = t->R[k]->pool_row_bytes();
        ICHECK(per == 0 || in.size() % per == 0, "rows_load: TP blob size");
        const size_t m = per ? in.size() / per : 0;
        std::vector<size_t> off(t->N + 1, 0);
        for (int k = 0; k < t->N; k++) off[k + 1] = off[k] + pr[k] * m;
        const unsigned char* b = in.data();
        return tp_all_v(t, [&](Engine& e, int k) { e.p_->rows_load_raw(b + off[k], m, dst, n); });
    }
    p_->rows_load(in, dst, n);
}
static void tp_ring_blob(int N, const std::vector<std::vector<unsigned char>>& v, std::vector<unsigned char>& out);
void Engine::ring_save(std::vector<unsigned char>& out) const {
    if (TP_OUTER) {
        TpCtx* t = p_->tpc;
        const_cast<Engine*>(this)->ring_finalize();   // (a pending snapshot first)
        std::vector<std::vector<unsigned char>> v(t->N);
        tp_all_v(t, [&](Engine& e, int k) { e.p_->ring_save(v[k]); });
        return tp_ring_blob(t->N, v, out);
    }
    p_->ring_save(out);
}
static void tp_ring_blob(int N, const std::vector<std::vector<unsigned char>>& v, std::vector<unsigned char>& out) {
    size_t n = 8 + 8 * (size_t)N;
    for (auto& x : v) n += x.size();
    out.resize(n);
    const uint32_t hd[2] = {TP_RING_MAGIC, (uint32_t)N};
    memcpy(out.data(), hd, 8);
    size_t o = 8 + 8 * (size_t)N;
    for (int k = 0; k < N; k++) {
        const uint64_t z = v[k].size();
        memcpy(out.data() + 8 + 8 * k, &z, 8);
        memcpy(out.data() + o, v[k].data(), v[k].size());
        o += v[k].size();
    }
}
void Engine::ring_load(const std::vector<unsigned char>& in, int n) {
    if (TP_OUTER) {
        TpCtx* t = p_->tpc;
        uint32_t hd[2] = {0, 0};
        ICHECK(in.size() >= 8, "ring_load: empty TP blob");
        memcpy(hd, in.data(), 8);
        ICHECK(hd[0] == TP_RING_MAGIC && (int)hd[1] == t->N, "ring_load: not a TP%d ring blob", t->N);
        std::vector<std::vector<unsigned char>> v(t->N);
        size_t o = 8 + 8 * (size_t)t->N;
        for (int k = 0; k < t->N; k++) {
            uint64_t z = 0;
            memcpy(&z, in.data() + 8 + 8 * k, 8);
            ICHECK(o + z <= in.size(), "ring_load: TP blob size");
            v[k].assign(in.data() + o, in.data() + o + z);
            o += z;
        }
        ICHECK(o == in.size(), "ring_load: TP blob has %zu trailing bytes", in.size() - o);   // review tp3 L4
        return tp_all_v(t, [&](Engine& e, int k) { e.ring_load(v[k], n); });
    }
    p_->ring_load(in, n);
}
void Engine::ring_save_async(std::vector<unsigned char>& out) {
    if (TP_OUTER) {   // review tp3 M2: every rank's async snapshot; the TP blob is assembled by ring_finalize
        TpCtx* t = p_->tpc;
        ring_finalize();
        t->pend.assign(t->N, {});
        out.clear();
        t->pend_out = &out;
        return tp_all_v(t, [t](Engine& e, int k) { e.ring_save_async(t->pend[k]); });
    }
    p_->ring_save_async(out);
}
void Engine::ring_finalize() {
    if (TP_OUTER) {
        TpCtx* t = p_->tpc;
        tp_all_v(t, [](Engine& e, int) { e.ring_finalize(); });
        if (t->pend_out) {
            std::vector<unsigned char>* o = t->pend_out;
            t->pend_out = nullptr;
            bool ok = true;
            for (auto& x : t->pend) ok = ok && !x.empty();   // a rank that dropped its copy: the whole snapshot is dropped
            if (ok) tp_ring_blob(t->N, t->pend, *o);
            else o->clear();
            t->pend.clear();
        }
        return;
    }
    p_->ring_finalize();
}
int Engine::slot_base(int slot) const { return p_->sbase.at(slot); }
int Engine::slot_lim(int slot) const { return p_->slim.at(slot); }
bool Engine::drafter_on() const { return p_->dr_on; }
// [ink-pf8] the dense attention weights' form: 8 = INT8 g128 from dense8.safetensors, 9 = INT8 RTN at load, 15 = G15
int Engine::dense_form() const { return p_->dense8 ? (p_->dense8_st ? 8 : 9) : 15; }
int Engine::cards() const { return p_->tps ? p_->sh.N : 1; }
// [ink-samp-dev] the owner settles a -2 (a filtered row left to the host; always the last emitted token, row n - 1)
int Engine::spec_step(int anchor, int k, int* out, float* probs) {
    const int n = spec_step_raw(anchor, k, out, probs);
    if (own_ && n >= 1 && out[n - 1] == -2) out[n - 1] = checked_token(filt_host(p_->cur, n - 1));
    return n;
}
static int tok_or_host(int id) { return id == -2 ? -2 : checked_token(id); }
int Engine::spec_step_raw(int anchor, int k, int* out, float* probs) {
    if (TP_OUTER) {   // [ink-tp-dec1] every rank the same step (policy state from rank 0, drafts replicated)
        TpCtx* t = p_->tpc;
        std::vector<std::vector<int>> o(t->N, std::vector<int>(DB));
        tp_pol_copy(t);   // [ink-n4-hosthop]
        const int n = tp_all_i(t, [&](Engine& e, int r) { return e.spec_step_raw(anchor, k, o[r].data(), r == 0 ? probs : nullptr); }, "spec tokens");
        for (int r = 1; r < t->N; r++) ICHECK(std::equal(o[r].begin(), o[r].begin() + n, o[0].begin()), "TP: rank %d spec tokens differ", r);
        std::copy(o[0].begin(), o[0].begin() + n, out);
        return n;
    }
    Impl& q = *p_;
    if (q.tps) q.tp_pol_sync();
    ICHECK(anchor >= 0 && anchor < V, "anchor %d", anchor);
    q.last_T = 1;
    if (k == 1 || k == 0 || !q.dr_on) {   // drafter off (also when the memory fit turned it off): a plain step
        out[0] = tok_or_host(q.step(anchor));
        return 1;
    }
    {   // review L5: never more rows than the context has left (plain step() still works there)
        const int room = q.slim[q.cur] - q.pos;
        if (k < 0) k = -std::min(-k, std::max(room, 1));
        else k = std::min(k, std::max(room, 1));
        if (k == -1) k = 1;   // review L6: one row = a plain step, no drafter block
    }
    {   // [ink-attn-tb-r4] the rows of one verify step on one side of the attention-form threshold
        for (int t0 : {q.tb_min - 1, q.tbl_min - 1}) {
            const int ak = k < 0 ? -k : k;
            if (t0 > 0 && q.pos < t0 && q.pos + ak > t0) k = k < 0 ? -std::max(t0 - q.pos, 1) : std::max(t0 - q.pos, 1);
            if (k == -1) k = 1;
        }
    }
    if (k == 1 || k == 0) {
        out[0] = tok_or_host(q.step(anchor));
        return 1;
    }
    int toks[DB];
    toks[0] = anchor;
    if (k < 0 && q.pol && q.pos >= q.pol_min && (q.e_tp <= 0 || q.plain_left > 0 || q.since_tp >= 256)) {   // [ink-dpol] plain step (also measures it)
        if (q.plain_left > 0) q.plain_left--;
        q.since_tp = 0;
        q.st_pl++;
        const auto tp = std::chrono::steady_clock::now();
        out[0] = tok_or_host(q.step(anchor));
        Impl::ema(q.e_tp, 1e3 * std::chrono::duration<double>(std::chrono::steady_clock::now() - tp).count(), 0.2);
        return 1;
    }
    double dms = 0;
    if (k < 0) {   // adaptive: prompt lookup on a long match, else the drafter with the confidence stop
        const int kmax = std::min(-k, VR);
        std::vector<int> d;
        const int ml = q.lk_min > 0 ? q.lookup(anchor, kmax - 1, d) : 0;
        if (ml >= q.lk_min && q.lk_min > 0 && !d.empty()) {
            k = 1 + (int)d.size();
            for (int j = 1; j < k; j++) toks[j] = d[j - 1];
            CU(cudaMemcpyAsync(q.DTOK, toks, k * 4, cudaMemcpyHostToDevice, q.st));
            q.st_lk++;
        } else {
            q.set_stop(q.tau, q.kmin);
            q.dr_stage(q.pos, anchor);
            const auto td = std::chrono::steady_clock::now();
            q.draft_run();
            dms = 1e3 * std::chrono::duration<double>(std::chrono::steady_clock::now() - td).count();
            if (q.tps) q.tp_bcast_draft(&dms, true);
            q.st_dms += dms;
            q.st_dn++;
            for (int j = 1; j < DB; j++) toks[j] = q.hdr[j];
            int m = 0;
            float pr = 1.f;
            while (m < kmax - 1) {
                pr *= q.hdp[m + 1];
                if (pr < q.tau) break;
                m++;
            }
            k = std::max(std::min(q.kmin, kmax), 1 + m);
            if (q.tau1 > 0.f && q.hdp[1] < q.tau1) k = 1;   // [ink-tau1] first draft unlikely: a one-row verify
            if (q.pol && q.pos >= q.pol_min && q.e_tp > 0 && q.e_tv[k] > 0) {   // [ink-dpol] k cap: expected tokens / verify ms vs plain
                double E[VR + 1], c = 1.0;
                E[1] = 1.0;
                for (int j = 2; j <= k; j++) E[j] = E[j - 1] + (c *= q.hdp[j - 1]);
                if (E[k] / q.e_tv[k] < Impl::POL_M / q.e_tp) {
                    int kb = 1;
                    double best = Impl::POL_M / q.e_tp;   // k = 1 -> a one-row verify
                    for (int j = 2; j < k; j++)
                        if (q.e_tv[j] > 0 && E[j] / q.e_tv[j] > best) best = E[j] / q.e_tv[kb = j];
                    if (kb != k) q.st_kcap++;
                    k = kb;
                    if (k == 1) {   // review r4b M1: the cap compared with the plain rate, so take a plain step
                        out[0] = tok_or_host(q.step(anchor));
                        return 1;
                    }
                }
            }
            if (probs)
                for (int j = 1; j < DB; j++) probs[j - 1] = q.hdp[j];
        }
    } else {
        q.set_stop(0.f, 0);
        q.dr_stage(q.pos, anchor);
        q.draft_run();
        if (q.tps) q.tp_bcast_draft(nullptr, true);
        for (int j = 1; j < DB; j++) toks[j] = q.hdr[j];
        if (probs)
            for (int j = 1; j < DB; j++) probs[j - 1] = q.hdp[j];
    }
    q.st_steps++;
    q.st_rows += k;
    q.st_kh[k]++;
    const auto tv = std::chrono::steady_clock::now();
    const int acc = q.verify(toks, k);
    const double vms = 1e3 * std::chrono::duration<double>(std::chrono::steady_clock::now() - tv).count();
    q.st_vms += vms;
    if (q.pol && q.pos >= q.pol_min && dms > 0) {   // [ink-dpol] drafted steps: costs and achieved rate; plain steps while spec loses
        if (!q.rr_cold) Impl::ema(q.e_tv[k], vms, 0.2);   // [ink-tp-vseen] steady-state times only
        Impl::ema(q.e_rn, acc + 1, 0.1, false);
        if (!q.rr_cold) Impl::ema(q.e_rt, vms + dms, 0.1);
        q.since_tp++;
        if (++q.n_ds >= 16 && q.e_tp > 0 && q.e_rt > 0 && q.e_rn > 0) {   // a decision on >= 16 fresh drafted steps
            if (q.e_rn / q.e_rt < Impl::POL_M / q.e_tp) {   // spec loses: plain blocks of growing length, then re-measure
                q.plain_left = q.plain_run;
                q.plain_run = std::min(2 * q.plain_run, 1024);
                q.n_ds = 0;
                q.e_rn = q.e_rt = 0;
            } else
                q.plain_run = 32;
        }
    }
    q.st_ah[acc + 1]++;
    q.st_tok += acc + 1;
    for (int j = 0; j <= acc; j++) out[j] = tok_or_host(q.hres[j]);
    return acc + 1;
}
void Engine::draft_probe(int anchor, float* probs) {   // tests: one drafter block at pos() (its rows' K / V are rewritten later)
    TP_NOT("draft_probe");   // review tp H1
    Impl& q = *p_;
    ICHECK(q.dr_on, "draft_probe needs the drafter");
    q.set_stop(0.f, 0);
    q.dr_stage(q.pos, anchor);
    q.dr_block();
    CU(cudaMemcpyAsync(probs, q.DPROB, DB * 4, cudaMemcpyDeviceToHost, q.st));
    CU(cudaStreamSynchronize(q.st));
}
void Engine::draft_at(int ap, int anchor, int* drafts, float* probs) {
    TP_NOT("draft_at");   // review tp H1
    Impl& q = *p_;
    ICHECK(q.dr_on && q.FMH, "draft_at needs the drafter and INK_DR_TEST=1");
    ICHECK(ap >= 1 && ap <= q.pos, "draft_at position %d (pos %d)", ap, q.pos);
    q.set_stop(0.f, 0);
    CU(cudaMemcpyAsync(q.FM, q.FMH + (size_t)(ap - 1) * H, H * 2, cudaMemcpyDeviceToDevice, q.st));
    q.dr_stage(ap, anchor);
    q.dr_block();
    float pr[DB];
    CU(cudaMemcpyAsync(q.hres, q.DTOK, DB * 4, cudaMemcpyDeviceToHost, q.st));
    CU(cudaMemcpyAsync(pr, q.DPROB, DB * 4, cudaMemcpyDeviceToHost, q.st));
    CU(cudaStreamSynchronize(q.st));
    for (int j = 1; j < DB; j++) {
        drafts[j - 1] = q.hres[j];
        if (probs) probs[j - 1] = pr[j];
    }
}
}  // namespace ink

namespace ink {
void Engine::bench_spec(int k, int n, double* verify_ms, double* draft_ms) {
    if (TP_OUTER) return tp_all_v(p_->tpc, [=](Engine& e, int r) {
        double v = 0, d = 0;
        e.bench_spec(k, n, &v, &d);
        if (r == 0) {
            *verify_ms = v;
            *draft_ms = d;
        }
    });
    Impl& q = *p_;
    if (!q.dr_on) {   // review r4 M1: the memory fit turned the drafter off (no drafter buffers)
        *verify_ms = *draft_ms = 0.0;
        return;
    }
    int toks[DB];
    for (int j = 0; j < DB; j++) toks[j] = 1000 + 37 * j;
    CU(cudaMemcpy(q.DTOK, toks, DB * 4, cudaMemcpyHostToDevice));
    for (int w = 0; w < 3; w++) q.verify(toks, k);
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < n; i++) q.verify(toks, k);
    *verify_ms = 1e3 * std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() / n;
    q.set_stop(0.f, 0);
    q.dr_stage(q.pos, toks[0]);
    for (int w = 0; w < 3; w++) q.draft_run();
    t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < n; i++) q.draft_run();
    *draft_ms = 1e3 * std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() / n;
}
}  // namespace ink

namespace ink {
std::string Engine::spec_stats(bool reset) {
    Impl& q = *p_;
    char b[512];
    int n = snprintf(b, sizeof b, "steps %lld (lookup %lld) rows/step %.2f tok/step %.2f verify %.2f ms drafter %.2f ms | k hist", q.st_steps,
                     q.st_lk, q.st_steps ? (double)q.st_rows / q.st_steps : 0., q.st_steps ? (double)q.st_tok / q.st_steps : 0.,
                     q.st_steps ? q.st_vms / q.st_steps : 0., q.st_dn ? q.st_dms / q.st_dn : 0.);
    for (int k = 1; k <= VR; k++) n += snprintf(b + n, sizeof b - n, " %lld", q.st_kh[k]);
    n += snprintf(b + n, sizeof b - n, " | emitted hist");
    for (int k = 1; k <= VR; k++) n += snprintf(b + n, sizeof b - n, " %lld", q.st_ah[k]);
    n += snprintf(b + n, sizeof b - n, " | policy plain %lld k-cap %lld", q.st_pl, q.st_kcap);
    if (reset) {
        q.st_pl = q.st_kcap = 0;
        q.st_steps = q.st_lk = q.st_rows = q.st_tok = q.st_dn = 0;
        q.st_dms = q.st_vms = 0;
        for (int k = 0; k <= VR; k++) q.st_kh[k] = q.st_ah[k] = 0;
    }
    return b;
}
}  // namespace ink
