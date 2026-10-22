// Tensor-core GEMM for prefill: C[t][n] = sum_k X[t][k] * W[n][k] for many tokens t.
// Weights use the same fragment-packed layouts as the GEMV path (gemv.cuh), so the A operand
// (weights) streams straight from global memory into registers, while token activations are
// staged through shared memory with cp.async and read as B fragments with ldmatrix.
#include <cerrno>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <tuple>
#include <dirent.h>
#include <sys/stat.h>
#include <vector>
#include <algorithm>
#include <unistd.h>

#include "gemv.cuh"

namespace {

constexpr int BT = 128;   // tokens per block
constexpr int BK = 64;    // k per pipeline stage
constexpr int STAGES = 3;
constexpr int kThreads = 256;
// 8 warps as WR (rows) x WT (tokens); a warp covers 32 weight rows x (128 / WT) tokens.
// WT = 1 (block = 256 rows) dequantizes each weight fragment once per 128 tokens.
template <int WT> constexpr int block_rows() { return 32 * (8 / WT); }

__device__ __forceinline__ void cp_async16(void* smem, const void* gmem, bool pred) {
    const unsigned s = __cvta_generic_to_shared(smem);
    const int n = pred ? 16 : 0;
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(s), "l"(gmem), "r"(n));
}
__device__ __forceinline__ void cp_async8(void* smem, const void* gmem, bool pred) {
    const unsigned s = __cvta_generic_to_shared(smem);
    const int n = pred ? 8 : 0;
    asm volatile("cp.async.ca.shared.global [%0], [%1], 8, %2;" ::"r"(s), "l"(gmem), "r"(n));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;"); }
template <int N>
__device__ __forceinline__ void cp_async_wait() {
    asm volatile("cp.async.wait_group %0;" ::"n"(N));
}

__device__ __forceinline__ void ldmatrix_x4(uint32_t r[4], const void* smem) {
    const unsigned s = __cvta_generic_to_shared(smem);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(s));
}

__device__ __forceinline__ void mma_bf16(float c[4], const uint32_t a[4], uint32_t b0, uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, "
        "{%8,%9}, {%0,%1,%2,%3};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// X tile in shared memory: [BT tokens][BK k] bf16, 128-byte rows, 16-byte chunks XOR-swizzled
// by (row & 7) so ldmatrix row reads are bank-conflict free.
__device__ __forceinline__ int swz(int row, int chunk) { return row * 8 + (chunk ^ (row & 7)); }

// Dequantize one packed u32 (8 nibbles) into 4 bf16x2 A-fragment registers, scaled by the
// row scales (rows g and g+8) and rounded to bf16 like the reference decompression.
__device__ __forceinline__ void dequant_scaled(uint32_t q, bf162 sg, bf162 sg8, uint32_t a[4]) {
    const bf162 bias = __halves2bfloat162(__float2bfloat16(136.f), __float2bfloat16(136.f));
#pragma unroll
    for (int j = 0; j < 4; j++) {
        uint32_t v = ((q >> (4 * j)) & 0x000F000Fu) | 0x43004300u;
        bf162 h = __hsub2(*reinterpret_cast<bf162*>(&v), bias);
        h = __hmul2(h, (j & 1) ? sg8 : sg);
        a[j] = *reinterpret_cast<uint32_t*>(&h);
    }
}

// TB: tokens per block (128 for prefill chunks; 64 / 32 for short appends, so little compute is
// spent on padding rows)
template <bool W4T, int WT, int TB = BT>
__global__ void __launch_bounds__(kThreads) gemm_kernel(const uint4* __restrict__ wq,
                                                       const bf16* __restrict__ ws, int N, int K,
                                                       const bf16* __restrict__ x, int ldx, int T,
                                                       float* __restrict__ c, int ldc,
                                                       size_t cstride, bool glu) {
    __shared__ __align__(128) uint4 xs[STAGES][TB * BK / 8];
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    constexpr int WR = 8 / WT, NT8 = TB / 8 / WT;  // n-tiles (8 tokens) per warp
    static_assert(NT8 >= 2 && NT8 % 2 == 0, "token tile");
    const int wr = warp % WR, wt = warp / WR;  // warp row group, warp token group
    const int g = lane >> 2, t = lane & 3;
    const int row0 = blockIdx.x * block_rows<WT>() + wr * 32;  // two 16-row tiles
    const int tok0 = blockIdx.y * TB;
    // split-K over gridDim.z (tiny N): slice z covers chunks [kb, ke) and writes c + z*cstride
    const int nkt = K / BK;
    const int kb = blockIdx.z * nkt / gridDim.z, ke = (blockIdx.z + 1) * nkt / gridDim.z;
    const int nk = ke - kb;
    c += blockIdx.z * cstride;

    // --- producer: X tile loads (each thread 4 x 16B per stage) ---
    auto load_x = [&](int stage, int kc) {
#pragma unroll
        for (int i = 0; i < (TB * BK / 8 + kThreads - 1) / kThreads; i++) {
            const int idx = tid + i * kThreads;
            if (idx >= TB * BK / 8) break;
            const int r = idx >> 3, ch = idx & 7;
            const int tok = tok0 + r;
            const bf16* src = x + (size_t)min(tok, T - 1) * ldx + (kb + kc) * BK + ch * 8;
            cp_async16(&xs[stage][swz(r, ch)], src, tok < T);
        }
    };

    float acc[2][NT8][4];
#pragma unroll
    for (int i = 0; i < 2; i++)
#pragma unroll
        for (int j = 0; j < NT8; j++)
#pragma unroll
            for (int e = 0; e < 4; e++) acc[i][j][e] = 0.f;

    // weight fragment pointers for the warp's two 16-row tiles (N may not fill the block)
    const int rt0 = row0 / 16;
    const bool v0 = rt0 < N / 16, v1 = rt0 + 1 < N / 16;
    const uint4* wp0;
    const uint4* wp1;
    if (W4T) {
        wp0 = wq + (size_t)rt0 * (K / 64) * 32 + lane;
        wp1 = wq + (size_t)(rt0 + 1) * (K / 64) * 32 + lane;
    } else {
        wp0 = wq + (size_t)rt0 * (K / 16) * 32 + lane;
        wp1 = wq + (size_t)(rt0 + 1) * (K / 16) * 32 + lane;
    }

#pragma unroll
    for (int s = 0; s < STAGES - 1; s++) {
        if (s < nk) load_x(s, s);
        cp_async_commit();
    }

    for (int kc = 0; kc < nk; kc++) {
        // weights for this k chunk (A fragments), loaded before waiting on the X tile
        uint32_t aw[2][4][4];  // [mtile][kstep][reg]
        if (W4T) {
            const uint4 z = make_uint4(0x88888888u, 0x88888888u, 0x88888888u, 0x88888888u);
            const uint4 q0 = v0 ? wp0[(size_t)(kb + kc) * 32] : z;
            const uint4 q1 = v1 ? wp1[(size_t)(kb + kc) * 32] : z;
            const int gi = ((kb + kc) * BK) >> 7;
            const int groups = K >> 7;
            const bf16* s0 = ws + ((size_t)(v0 ? rt0 : 0) * groups + gi) * 16;
            const bf16* s1 = ws + ((size_t)(v1 ? rt0 + 1 : 0) * groups + gi) * 16;
            const bf162 s0g = __bfloat162bfloat162(s0[g]), s0g8 = __bfloat162bfloat162(s0[g + 8]);
            const bf162 s1g = __bfloat162bfloat162(s1[g]), s1g8 = __bfloat162bfloat162(s1[g + 8]);
            const uint32_t w0[4] = {q0.x, q0.y, q0.z, q0.w}, w1[4] = {q1.x, q1.y, q1.z, q1.w};
#pragma unroll
            for (int s = 0; s < 4; s++) {
                dequant_scaled(w0[s], s0g, s0g8, aw[0][s]);
                dequant_scaled(w1[s], s1g, s1g8, aw[1][s]);
            }
        } else {
            const uint4 z = make_uint4(0, 0, 0, 0);
#pragma unroll
            for (int s = 0; s < 4; s++) {
                const uint4 a0 = v0 ? wp0[(size_t)((kb + kc) * 4 + s) * 32] : z;
                const uint4 a1 = v1 ? wp1[(size_t)((kb + kc) * 4 + s) * 32] : z;
                aw[0][s][0] = a0.x; aw[0][s][1] = a0.y; aw[0][s][2] = a0.z; aw[0][s][3] = a0.w;
                aw[1][s][0] = a1.x; aw[1][s][1] = a1.y; aw[1][s][2] = a1.z; aw[1][s][3] = a1.w;
            }
        }

        cp_async_wait<STAGES - 2>();
        __syncthreads();
        // prefetch stage kc + STAGES - 1 (its buffer was consumed in iteration kc - 1)
        if (kc + STAGES - 1 < nk) load_x((kc + STAGES - 1) % STAGES, kc + STAGES - 1);
        cp_async_commit();

        const uint4* xb = xs[kc % STAGES];
#pragma unroll
        for (int s = 0; s < 4; s++) {
#pragma unroll
            for (int jp = 0; jp < NT8 / 2; jp++) {  // pairs of n-tiles (8 tokens each)
                const int mi = lane >> 3;
                const int r = wt * (TB / WT) + (jp * 2 + (mi >> 1)) * 8 + (lane & 7);
                const int ch = s * 2 + (mi & 1);
                uint32_t b[4];
                ldmatrix_x4(b, &xb[swz(r, ch)]);
#pragma unroll
                for (int i = 0; i < 2; i++) {
                    mma_bf16(acc[i][jp * 2], aw[i][s], b[0], b[1]);
                    mma_bf16(acc[i][jp * 2 + 1], aw[i][s], b[2], b[3]);
                }
            }
        }
    }
    cp_async_wait<0>();

    // epilogue: c0 (row g, tok 2t) c1 (row g, tok 2t+1) c2 (row g+8, tok 2t) c3 (row g+8, tok 2t+1)
    if (glu) {  // rows g / g + 8 of each 16-row tile: gate / up of activation column tile*8+g
        bf16* act = reinterpret_cast<bf16*>(c);
#pragma unroll
        for (int i = 0; i < 2; i++) {
            if (!(i == 0 ? v0 : v1)) continue;
            const int col = (row0 + i * 16) / 2 + g;
#pragma unroll
            for (int j = 0; j < NT8; j++) {
                const int tk = tok0 + wt * (TB / WT) + j * 8 + 2 * t;
                if (tk < T)
                    act[(size_t)tk * ldc + col] =
                        __float2bfloat16(acc[i][j][0] / (1.f + __expf(-acc[i][j][0])) * acc[i][j][2]);
                if (tk + 1 < T)
                    act[(size_t)(tk + 1) * ldc + col] =
                        __float2bfloat16(acc[i][j][1] / (1.f + __expf(-acc[i][j][1])) * acc[i][j][3]);
            }
        }
        return;
    }
#pragma unroll
    for (int i = 0; i < 2; i++) {
        if (!(i == 0 ? v0 : v1)) continue;
        const int r = row0 + i * 16 + g;
#pragma unroll
        for (int j = 0; j < NT8; j++) {
            const int tk = tok0 + wt * (TB / WT) + j * 8 + 2 * t;
            if (tk < T) {
                c[(size_t)tk * ldc + r] = acc[i][j][0];
                c[(size_t)tk * ldc + r + 8] = acc[i][j][2];
            }
            if (tk + 1 < T) {
                c[(size_t)(tk + 1) * ldc + r] = acc[i][j][1];
                c[(size_t)(tk + 1) * ldc + r + 8] = acc[i][j][3];
            }
        }
    }
}

// ---------------------------------------------------------------- fp16-accumulate W4 GEMM
// GA102 tensor cores run fp16-accumulate MMA at twice the fp32-accumulate rate. Weights enter
// as exact integers (q - 8) in fp16, activations as fp16 scaled per token so that |x| <= 16:
// a chunk of 64 products then stays far below the fp16 range (64 * 8 * 16 = 8192). Each chunk's
// fp16 partial is scaled by the weight group scale and added to fp32 accumulators; the token
// scale is applied in the epilogue.
__device__ __forceinline__ void mma_h16(uint32_t c[2], const uint32_t a[4], uint32_t b0, uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, "
        "{%0,%1};"
        : "+r"(c[0]), "+r"(c[1])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void dequant_int_h(uint32_t q, uint32_t a[4]) {
    const half2 bias = __halves2half2(__float2half(1032.f), __float2half(1032.f));
#pragma unroll
    for (int j = 0; j < 4; j++) {
        uint32_t v = ((q >> (4 * j)) & 0x000F000Fu) | 0x64006400u;
        half2 h = __hsub2(*reinterpret_cast<half2*>(&v), bias);
        a[j] = *reinterpret_cast<uint32_t*>(&h);
    }
}

constexpr float kXTarget = 16.f;

// one block per token: xs[t] = amax / 16, xh[t][k] = x[t][k] / xs[t] in fp16
__global__ void x_to_h_kernel(const bf16* __restrict__ x, int ldx, int K, __half* __restrict__ xh,
                              float* __restrict__ xs) {
    const int t = blockIdx.x;
    const bf16* r = x + (size_t)t * ldx;
    float mx = 0.f;
    for (int k = threadIdx.x * 2; k < K; k += blockDim.x * 2) {
        const float2 f = __bfloat1622float2(*reinterpret_cast<const bf162*>(r + k));
        mx = fmaxf(mx, fmaxf(fabsf(f.x), fabsf(f.y)));
    }
    __shared__ float red[32];
    for (int o = 16; o; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = mx;
    __syncthreads();
    if (threadIdx.x < 32) {
        mx = threadIdx.x < (blockDim.x >> 5) ? red[threadIdx.x] : 0.f;
        for (int o = 16; o; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
        if (threadIdx.x == 0) red[0] = mx;
    }
    __syncthreads();
    const float sc = red[0] > 0.f ? red[0] / kXTarget : 1.f, inv = 1.f / sc;
    if (threadIdx.x == 0) xs[t] = sc;
    __half* o = xh + (size_t)t * K;
    for (int k = threadIdx.x * 2; k < K; k += blockDim.x * 2) {
        const float2 f = __bfloat1622float2(*reinterpret_cast<const bf162*>(r + k));
        *reinterpret_cast<half2*>(o + k) = __floats2half2_rn(f.x * inv, f.y * inv);
    }
}

template <int WT, int TB>
__global__ void __launch_bounds__(kThreads) gemm_h_kernel(const uint4* __restrict__ wq,
                                                         const bf16* __restrict__ ws, int N, int K,
                                                         const __half* __restrict__ x,
                                                         const float* __restrict__ xsc, int T,
                                                         float* __restrict__ c, int ldc,
                                                         size_t cstride, bool glu) {
    __shared__ __align__(128) uint4 xs[STAGES][TB * BK / 8];
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    constexpr int WR = 8 / WT, NT8 = TB / 8 / WT;
    static_assert(NT8 >= 2 && NT8 % 2 == 0, "token tile");
    const int wr = warp % WR, wt = warp / WR;
    const int g = lane >> 2, t = lane & 3;
    const int row0 = blockIdx.x * block_rows<WT>() + wr * 32;
    const int tok0 = blockIdx.y * TB;
    const int nkt = K / BK;
    const int kb = blockIdx.z * nkt / gridDim.z, ke = (blockIdx.z + 1) * nkt / gridDim.z;
    const int nk = ke - kb;
    c += blockIdx.z * cstride;

    auto load_x = [&](int stage, int kc) {
#pragma unroll
        for (int i = 0; i < (TB * BK / 8 + kThreads - 1) / kThreads; i++) {
            const int idx = tid + i * kThreads;
            if (idx >= TB * BK / 8) break;
            const int r = idx >> 3, ch = idx & 7;
            const int tok = tok0 + r;
            const __half* src = x + (size_t)min(tok, T - 1) * K + (kb + kc) * BK + ch * 8;
            cp_async16(&xs[stage][swz(r, ch)], src, tok < T);
        }
    };

    float acc[2][NT8][4];
#pragma unroll
    for (int i = 0; i < 2; i++)
#pragma unroll
        for (int j = 0; j < NT8; j++)
#pragma unroll
            for (int e = 0; e < 4; e++) acc[i][j][e] = 0.f;

    const int rt0 = row0 / 16;
    const bool v0 = rt0 < N / 16, v1 = rt0 + 1 < N / 16;
    const uint4* wp0 = wq + (size_t)rt0 * (K / 64) * 32 + lane;
    const uint4* wp1 = wq + (size_t)(rt0 + 1) * (K / 64) * 32 + lane;

#pragma unroll
    for (int s = 0; s < STAGES - 1; s++) {
        if (s < nk) load_x(s, s);
        cp_async_commit();
    }

    for (int kc = 0; kc < nk; kc++) {
        uint32_t aw[2][4][4];
        const uint4 z = make_uint4(0x88888888u, 0x88888888u, 0x88888888u, 0x88888888u);
        const uint4 q0 = v0 ? wp0[(size_t)(kb + kc) * 32] : z;
        const uint4 q1 = v1 ? wp1[(size_t)(kb + kc) * 32] : z;
        const int gi = ((kb + kc) * BK) >> 7;
        const int groups = K >> 7;
        const bf16* s0 = ws + ((size_t)(v0 ? rt0 : 0) * groups + gi) * 16;
        const bf16* s1 = ws + ((size_t)(v1 ? rt0 + 1 : 0) * groups + gi) * 16;
        const float sc[2][2] = {{__bfloat162float(s0[g]), __bfloat162float(s0[g + 8])},
                                {__bfloat162float(s1[g]), __bfloat162float(s1[g + 8])}};
        const uint32_t w0[4] = {q0.x, q0.y, q0.z, q0.w}, w1[4] = {q1.x, q1.y, q1.z, q1.w};
#pragma unroll
        for (int s = 0; s < 4; s++) {
            dequant_int_h(w0[s], aw[0][s]);
            dequant_int_h(w1[s], aw[1][s]);
        }

        cp_async_wait<STAGES - 2>();
        __syncthreads();
        if (kc + STAGES - 1 < nk) load_x((kc + STAGES - 1) % STAGES, kc + STAGES - 1);
        cp_async_commit();

        const uint4* xb = xs[kc % STAGES];
#pragma unroll
        for (int jp = 0; jp < NT8 / 2; jp++) {
            uint32_t ch[2][2][2] = {};
#pragma unroll
            for (int s = 0; s < 4; s++) {
                const int mi = lane >> 3;
                const int r = wt * (TB / WT) + (jp * 2 + (mi >> 1)) * 8 + (lane & 7);
                const int chn = s * 2 + (mi & 1);
                uint32_t b[4];
                ldmatrix_x4(b, &xb[swz(r, chn)]);
#pragma unroll
                for (int i = 0; i < 2; i++) {
                    mma_h16(ch[i][0], aw[i][s], b[0], b[1]);
                    mma_h16(ch[i][1], aw[i][s], b[2], b[3]);
                }
            }
#pragma unroll
            for (int i = 0; i < 2; i++)
#pragma unroll
                for (int h = 0; h < 2; h++) {
                    const float2 lo = __half22float2(*reinterpret_cast<half2*>(&ch[i][h][0]));
                    const float2 hi = __half22float2(*reinterpret_cast<half2*>(&ch[i][h][1]));
                    float* a = acc[i][jp * 2 + h];
                    a[0] += lo.x * sc[i][0];
                    a[1] += lo.y * sc[i][0];
                    a[2] += hi.x * sc[i][1];
                    a[3] += hi.y * sc[i][1];
                }
        }
    }
    cp_async_wait<0>();

    // token scales
#pragma unroll
    for (int j = 0; j < NT8; j++) {
        const int tk = tok0 + wt * (TB / WT) + j * 8 + 2 * t;
        const float sa = tk < T ? xsc[tk] : 0.f, sb = tk + 1 < T ? xsc[tk + 1] : 0.f;
#pragma unroll
        for (int i = 0; i < 2; i++) {
            acc[i][j][0] *= sa; acc[i][j][2] *= sa;
            acc[i][j][1] *= sb; acc[i][j][3] *= sb;
        }
    }
    if (glu) {
        bf16* act = reinterpret_cast<bf16*>(c);
#pragma unroll
        for (int i = 0; i < 2; i++) {
            if (!(i == 0 ? v0 : v1)) continue;
            const int col = (row0 + i * 16) / 2 + g;
#pragma unroll
            for (int j = 0; j < NT8; j++) {
                const int tk = tok0 + wt * (TB / WT) + j * 8 + 2 * t;
                if (tk < T)
                    act[(size_t)tk * ldc + col] =
                        __float2bfloat16(acc[i][j][0] / (1.f + __expf(-acc[i][j][0])) * acc[i][j][2]);
                if (tk + 1 < T)
                    act[(size_t)(tk + 1) * ldc + col] =
                        __float2bfloat16(acc[i][j][1] / (1.f + __expf(-acc[i][j][1])) * acc[i][j][3]);
            }
        }
        return;
    }
#pragma unroll
    for (int i = 0; i < 2; i++) {
        if (!(i == 0 ? v0 : v1)) continue;
        const int r = row0 + i * 16 + g;
#pragma unroll
        for (int j = 0; j < NT8; j++) {
            const int tk = tok0 + wt * (TB / WT) + j * 8 + 2 * t;
            if (tk < T) {
                c[(size_t)tk * ldc + r] = acc[i][j][0];
                c[(size_t)tk * ldc + r + 8] = acc[i][j][2];
            }
            if (tk + 1 < T) {
                c[(size_t)(tk + 1) * ldc + r] = acc[i][j][1];
                c[(size_t)(tk + 1) * ldc + r + 8] = acc[i][j][3];
            }
        }
    }
}

// ---------------------------------------------------------------- INT8-activation W4 GEMM (W4A8)
// GA102 runs s8 x s8 -> s32 MMA at twice the fp16-accumulate rate. Activations are quantized
// per (token, group of G k) to int8 by x_to_i8_kernel; weights enter as exact signed int8
// (u - 8) unpacked from the nibbles. A k32 MMA step takes the two 16-k W4 words of a chunk, so
// lane t holds k {2t, 2t+8, 2t+1, 2t+9} (+16) of every 32-k block: the activations are packed
// in the same permutation by the pre-pass, which makes the dot product order-independent.
// The int32 accumulator starts at the bit pattern of M = 1.5 * 2^23, so after a group the
// register already holds the float M + sum (|sum| < 2^22) and the int -> float conversion is a
// subtraction (I2F runs at a quarter rate on sm_86). Per group: tmp += (c - M) * sx[t][g];
// per 128-k weight group: acc += tmp * sw[n][g128].
constexpr int BKI = 128;                   // k per pipeline stage (one weight group)
constexpr float kMagic = 12582912.f;       // 1.5 * 2^23
constexpr uint32_t kMagicBits = 0x4B400000u;

__device__ __forceinline__ void mma_s8(uint32_t c[4], const uint32_t a[4], uint32_t b0, uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, "
        "{%8,%9}, {%0,%1,%2,%3};"
        : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
// first step of a group: C operand = the magic constant (no accumulator zeroing)
__device__ __forceinline__ void mma_s8_init(uint32_t c[4], const uint32_t a[4], uint32_t b0,
                                            uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, "
        "{%8,%9}, {%10,%10,%10,%10};"
        : "=r"(c[0]), "=r"(c[1]), "=r"(c[2]), "=r"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(kMagicBits));
}

// 4 nibbles (u = w + 8, even or odd positions of q) -> 4 signed int8 bytes (w)
__device__ __forceinline__ uint32_t nib4_to_s8x4(uint32_t q) {
    const uint32_t u = (q & 0x0F0F0F0Fu) | 0x80808080u;
    return (u - 0x08080808u) ^ 0x80808080u;
}
// two 16-k W4 words -> A fragment of one k32 MMA step
__device__ __forceinline__ void dequant_s8(uint32_t q0, uint32_t q1, uint32_t a[4]) {
    a[0] = nib4_to_s8x4(q0);
    a[1] = nib4_to_s8x4(q0 >> 4);
    a[2] = nib4_to_s8x4(q1);
    a[3] = nib4_to_s8x4(q1 >> 4);
}

// ---------------------------------------------------------------- W4A8 lean variant (QE_PREFILL_INT8_KERNEL=2, default)
// gemm_i8t_kernel with a shorter instruction stream per MMA (the SASS of
// gemm_i8t_kernel<1,64,64> had 11.2 instructions per IMMA in the K loop, 120 of 715 per stage
// register moves building the init MMA's C quad {C_g, C_g, C_g8, C_g8}):
// - scale records in MMA-operand order per 16-token tile and 128-k group: for each sub-group and
//   token pair (g, g + 8) a float4 {C_g, C_g, C_g8, C_g8} that is the init MMA's C operand as
//   loaded (no register shuffles) and a float4 {sx_g, -M sx_g, sx_g8, -M sx_g8} (no FMUL); one
//   LDS.128 each, a warp reads 128 contiguous bytes (8 pairs) per load;
// - HI: the odd (high) nibbles enter the MMA in place as 16 u (<= 240, still u8: one AND, no
//   shift); their accumulators start at C_hi = M_bits - 128 sum x and use the weight scale sw / 16
//   (exact): fma(f, sx, -M sx) is then 16 x the low-nibble formula's value, rounded once, so
//   fma(that, sw / 16, acc) is bit-identical to gemm_i8t_kernel's flush;
// - the weight scales of the previous stage stay in registers for the deferred flush.
// Same tile decomposition, pipeline, stream-K and epilogue as gemm_i8t_kernel (bit-identical outputs).
template <bool HI> constexpr int v_kinds() { return HI ? 3 : 2; }
template <int G, bool HI> constexpr int v_blk() { return (128 / G) * v_kinds<HI>() * 8; }  // float4 per (16-token tile, 128-k group)
constexpr size_t i8v_rec_floats(int k, int G, bool HI) {  // per token; buffers hold ceil16(t) tokens
    return (size_t)(k / 128) * (128 / G) * (HI ? 3 : 2) * 2;
}
// record of token r (GEMM-local), 128-k group grp, sub-group sub: scale sc, int8 sum of the sub-group
template <int G, bool HI>
__device__ __forceinline__ void i8v_rec_store(float* xsc, int r, int grp, int groups, int sub, float sc, int sum) {
    constexpr int SUB = 128 / G, KN = v_kinds<HI>();
    float2* p = reinterpret_cast<float2*>(reinterpret_cast<float4*>(xsc) + ((size_t)(r >> 4) * groups + grp) * (SUB * KN * 8) +
                                          sub * KN * 8 + (r & 7)) + ((r >> 3) & 1);
    const float c = __int_as_float((int)kMagicBits - 8 * sum);
    p[0] = make_float2(c, c);
    if constexpr (HI) {
        const float ch = __int_as_float((int)kMagicBits - 128 * sum);
        p[16] = make_float2(ch, ch);
    }
    p[(KN - 1) * 16] = make_float2(sc, -kMagic * sc);
}
// One block per token, thread j quantizes k [16j, 16j + 16): xq[t][k] int8 in the MMA
// permutation (byte 4u + i of each 16-k half <- k 2u + {0, 8, 1, 9}[i]); scale records per
// (token, 128-k group): ceil(SUB / 2) float4 {sx(0), C(0), sx(1), C(1)} for the SUB = 128 / G
// sub-groups, where C = M_bits - 8 * sum of the sub-group's int8 values is the MMA accumulator
// start of the u8 x s8 kernel (weights as nibbles u = w + 8: M + sum u x + C = M + sum w x
// exactly); -M sx is recomputed by the kernels (exact: sx has 22 significant bits).
// FMT: 1 = gemm_i8t_kernel records, 2 / 3 = gemm_i8v_kernel records (3: with C_hi)
template <int G, int FMT = 1>
__global__ void x_to_i8_kernel(const bf16* __restrict__ x, int ldx, int K, int8_t* __restrict__ xq,
                               float* __restrict__ xsc, int* __restrict__ ctr, int nctr) {
    const int t = blockIdx.x;
    if (t == 0)  // stream-K counters of the GEMM that follows on this stream
        for (int i = threadIdx.x; i < nctr; i += blockDim.x) ctr[i] = 0;
    const bf16* r = x + (size_t)t * ldx;
    constexpr int L = G / 16;  // lanes per group
    for (int j = threadIdx.x; j < K / 16; j += blockDim.x) {
        const uint4 p0 = *reinterpret_cast<const uint4*>(r + 16 * j);
        const uint4 p1 = *reinterpret_cast<const uint4*>(r + 16 * j + 8);
        const uint32_t w[8] = {p0.x, p0.y, p0.z, p0.w, p1.x, p1.y, p1.z, p1.w};
        float v[16];
        float mx = 0.f;
#pragma unroll
        for (int i = 0; i < 8; i++) {
            const float2 f = __bfloat1622float2(*reinterpret_cast<const bf162*>(&w[i]));
            v[2 * i] = f.x;
            v[2 * i + 1] = f.y;
            mx = fmaxf(mx, fmaxf(fabsf(f.x), fabsf(f.y)));
        }
#pragma unroll
        for (int o = 1; o < L; o <<= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
        // scale with 22 significant bits: M * sc is then exact in fp32, so the kernel's
        // fma(f, sc, -M * sc) equals (f - M) * sc rounded once
        const float sc = mx > 0.f ? __uint_as_float(__float_as_uint(mx / 127.f) & ~3u) : 1.f;
        const float inv = mx > 0.f ? __frcp_rn(sc) : 0.f;
        uint32_t o[4];
        int sum = 0;
#pragma unroll
        for (int u = 0; u < 4; u++) {
            const int q0 = __float2int_rn(v[2 * u] * inv), q1 = __float2int_rn(v[2 * u + 8] * inv);
            const int q2 = __float2int_rn(v[2 * u + 1] * inv), q3 = __float2int_rn(v[2 * u + 9] * inv);
            sum += q0 + q1 + q2 + q3;
            o[u] = (uint32_t)(q0 & 0xFF) | ((uint32_t)(q1 & 0xFF) << 8) | ((uint32_t)(q2 & 0xFF) << 16) |
                   ((uint32_t)(q3 & 0xFF) << 24);
        }
#pragma unroll
        for (int o2 = 1; o2 < L; o2 <<= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o2);
        constexpr int NREC = (128 / G + 1) / 2;
        if (FMT >= 2 && (j % L) == 0) i8v_rec_store<G, FMT == 3>(xsc, t, (16 * j) / 128, K / 128, ((16 * j) % 128) / G, sc, sum);
        else if ((j % L) == 0) {
            const int sub = ((16 * j) % 128) / G;
            *reinterpret_cast<float2*>(xsc + (((size_t)t * (K / 128) + (16 * j) / 128) * NREC + sub / 2) * 4 +
                                       (sub & 1) * 2) = make_float2(sc, __int_as_float((int)kMagicBits - 8 * sum));
        }
        *reinterpret_cast<uint4*>(xq + (size_t)t * K + 16 * j) = make_uint4(o[0], o[1], o[2], o[3]);
    }
}
constexpr size_t i8_rec_floats(int k, int G) { return (size_t)(k / 128) * ((128 / G + 1) / 2) * 4; }  // per token
// (the old format has the same size: (k / 128) * max(2, 128 / G) * 2 floats)
// (QE_PREFILL_INT8_KERNEL=0) master's pre-pass for gemm_i8_kernel: one block per token, thread j quantizes k [16j, 16j + 16): xq[t][k] int8 in the MMA
// permutation (byte 4u + i of each 16-k half <- k 2u + {0, 8, 1, 9}[i]), scales
// xsc[t][k / 128][SP][2] = {sx, -M sx} per G-group (SP = max(2, 128 / G) pairs: 16-byte rows).
template <int G>
__global__ void x_to_i8_old_kernel(const bf16* __restrict__ x, int ldx, int K, int8_t* __restrict__ xq,
                               float* __restrict__ xsc, int* __restrict__ ctr, int nctr) {
    const int t = blockIdx.x;
    if (t == 0)  // stream-K counters of the GEMM that follows on this stream
        for (int i = threadIdx.x; i < nctr; i += blockDim.x) ctr[i] = 0;
    const bf16* r = x + (size_t)t * ldx;
    constexpr int L = G / 16;  // lanes per group
    for (int j = threadIdx.x; j < K / 16; j += blockDim.x) {
        const uint4 p0 = *reinterpret_cast<const uint4*>(r + 16 * j);
        const uint4 p1 = *reinterpret_cast<const uint4*>(r + 16 * j + 8);
        const uint32_t w[8] = {p0.x, p0.y, p0.z, p0.w, p1.x, p1.y, p1.z, p1.w};
        float v[16];
        float mx = 0.f;
#pragma unroll
        for (int i = 0; i < 8; i++) {
            const float2 f = __bfloat1622float2(*reinterpret_cast<const bf162*>(&w[i]));
            v[2 * i] = f.x;
            v[2 * i + 1] = f.y;
            mx = fmaxf(mx, fmaxf(fabsf(f.x), fabsf(f.y)));
        }
#pragma unroll
        for (int o = 1; o < L; o <<= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
        // scale with 22 significant bits: M * sc is then exact in fp32, so the kernel's
        // fma(f, sc, -M * sc) equals (f - M) * sc rounded once
        const float sc = mx > 0.f ? __uint_as_float(__float_as_uint(mx / 127.f) & ~3u) : 1.f;
        const float inv = mx > 0.f ? __frcp_rn(sc) : 0.f;
        constexpr int SP = G == 128 ? 2 : 128 / G;  // (sx, -M sx) pairs per 128-k group (16-byte rows)
        if ((j % L) == 0)
            *reinterpret_cast<float2*>(xsc + (((size_t)t * (K / 128) + (16 * j) / 128) * SP +
                                              ((16 * j) % 128) / G) * 2) = make_float2(sc, -kMagic * sc);
        uint32_t o[4];
#pragma unroll
        for (int u = 0; u < 4; u++) {
            const int q0 = __float2int_rn(v[2 * u] * inv), q1 = __float2int_rn(v[2 * u + 8] * inv);
            const int q2 = __float2int_rn(v[2 * u + 1] * inv), q3 = __float2int_rn(v[2 * u + 9] * inv);
            o[u] = (uint32_t)(q0 & 0xFF) | ((uint32_t)(q1 & 0xFF) << 8) | ((uint32_t)(q2 & 0xFF) << 16) |
                   ((uint32_t)(q3 & 0xFF) << 24);
        }
        *reinterpret_cast<uint4*>(xq + (size_t)t * K + 16 * j) = make_uint4(o[0], o[1], o[2], o[3]);
    }
}



// Persistent ("stream-K") scheduling: the grid is at most the number of SMs; blocks grab work
// units from a counter. Units 0 .. full-1 are whole output tiles (256 x TB, tile = unit, row
// block fastest as in the plain grid); the remaining rem tiles (the last partial wave) are
// split into `parts` K-ranges each: every part stores its fp32 partial to the workspace
// [tile][part][ROWS][TB] and sk_reduce_kernel sums the parts in a fixed order (deterministic)
// and runs the epilogue. The work counter is zeroed by x_to_i8_kernel.
template <int WT, int TB, int G, bool DIRECT>
__global__ void __launch_bounds__(kThreads) gemm_i8_kernel(const uint4* __restrict__ wq,
                                                          const bf16* __restrict__ ws, int N, int K,
                                                          const int8_t* __restrict__ x,
                                                          const float* __restrict__ xsc, int T,
                                                          float* __restrict__ c, int ldc, bool glu,
                                                          int* __restrict__ ctr, float* __restrict__ wsp,
                                                          int full, int parts) {
    extern __shared__ __align__(128) uint4 dyn_smem[];
    __shared__ int s_unit;
    uint4* xs = dyn_smem;                                                     // [STAGES][TB * 8]
    constexpr int SUB = BKI / G;  // activation groups per stage
    constexpr int SP = SUB < 2 ? 2 : SUB;  // stored (sx, -M sx) pairs per token and stage
    float* ss = reinterpret_cast<float*>(dyn_smem + STAGES * TB * 8);  // [STAGES][TB][SP][sx, -M sx]
    // weights of the next stage, staged by cp.async (double buffer): [2][4 chunks][thread]
    uint4* wsm = dyn_smem + STAGES * TB * 8 + STAGES * TB * SP / 2;
    uint4* ssm = wsm + 2 * 4 * kThreads;  // group scales of the stage: [buf][warp][4 x 16 B]
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    constexpr int WR = 8 / WT, NT8 = TB / 8 / WT, ROWS = block_rows<WT>();
    static_assert(NT8 >= 2 && NT8 % 2 == 0, "token tile");
    static_assert(G == 32 || G == 64 || G == 128, "group");
    const int wr = warp % WR, wt = warp / WR;
    const int g = lane >> 2, t = lane & 3;
    const int nkt = K / BKI;
    const int groups = K >> 7;
    const int rb = (N + ROWS - 1) / ROWS;
    const int tiles = rb * ((T + TB - 1) / TB);
    const int total = full + (tiles - full) * parts;

    for (;;) {
        if (tid == 0) s_unit = atomicAdd(ctr, 1);
        __syncthreads();
        const int u = s_unit;
        if (u >= total) break;
        int tile, kb, ke, part;
        if (u < full) {
            tile = u; kb = 0; ke = nkt; part = -1;
        } else {
            const int r = (u - full) / parts;
            part = (u - full) % parts;
            tile = full + r;
            kb = part * nkt / parts;
            ke = (part + 1) * nkt / parts;
        }
        const int nk = ke - kb;
        const int row0 = (tile % rb) * ROWS + wr * 32;
        const int tok0 = (tile / rb) * TB;

        auto load_x = [&](int stage, int kc) {
            uint4* xb = xs + stage * TB * 8;
#pragma unroll
            for (int i = 0; i < (TB * 8 + kThreads - 1) / kThreads; i++) {
                const int idx = tid + i * kThreads;
                if (idx >= TB * 8) break;
                const int r = idx >> 3, ch = idx & 7;
                const int tok = tok0 + r;
                const int8_t* src = x + (size_t)min(tok, T - 1) * K + (kb + kc) * BKI + ch * 16;
                cp_async16(&xb[swz(r, ch)], src, tok < T);
            }
            {  // scales: SP / 2 16-byte pieces per token
                constexpr int P = SP / 2;
                if (tid < TB * P) {
                    const int tok = tok0 + tid / P, pc = tid % P;
                    const float* src = xsc + (((size_t)min(tok, T - 1) * groups + kb + kc) * SP + pc * 2) * 2;
                    cp_async16(ss + ((stage * TB + tid / P) * SP + pc * 2) * 2, src, tok < T);
                }
            }
        };

        float acc[2][NT8][4];
#pragma unroll
        for (int i = 0; i < 2; i++)
#pragma unroll
            for (int j = 0; j < NT8; j++)
#pragma unroll
                for (int e = 0; e < 4; e++) acc[i][j][e] = 0.f;

        const int rt0 = row0 / 16;
        const bool v0 = rt0 < N / 16, v1 = rt0 + 1 < N / 16;
        const uint4* wp0 = wq + (size_t)(v0 ? rt0 : 0) * (K / 64) * 32 + lane;
        const uint4* wp1 = wq + (size_t)(v1 ? rt0 + 1 : 0) * (K / 64) * 32 + lane;
        // (rows past N: zero-filled -> w = -8; their results are never stored)
        const bf16* sp0 = ws + (size_t)(v0 ? rt0 : 0) * groups * 16;
        const bf16* sp1 = ws + (size_t)(v1 ? rt0 + 1 : 0) * groups * 16;
        auto load_w = [&](int buf, int kc) {
            const size_t ci = (size_t)(kb + kc) * 2;  // first 64-k chunk of the stage
            uint4* d = wsm + (size_t)buf * 4 * kThreads + tid;  // [buf][chunk][thread]: conflict-free
            cp_async16(d + 0 * kThreads, wp0 + ci * 32, v0);
            cp_async16(d + 1 * kThreads, wp0 + (ci + 1) * 32, v0);
            cp_async16(d + 2 * kThreads, wp1 + ci * 32, v1);
            cp_async16(d + 3 * kThreads, wp1 + (ci + 1) * 32, v1);
            if (lane < 4)  // 2 m-tiles x 16 bf16 scales (always a valid address; row 0 for rows past N)
                cp_async16(ssm + (buf * 8 + warp) * 4 + lane,
                           (lane < 2 ? sp0 : sp1) + ((size_t)(kb + kc) * 16) + (lane & 1) * 8, true);
        };

        // groups: [x(0) w(0)] [x(1)] ... each iteration kc then commits [w(kc+1)] and [x(kc+2)];
        // the wait<1> at the top of iteration kc leaves only x(kc+1) pending: x(kc), w(kc) are in
        load_w(0, 0);
#pragma unroll
        for (int s = 0; s < STAGES - 1; s++) {
            if (s < nk) load_x(s, s);
            cp_async_commit();
        }

        for (int kc = 0; kc < nk; kc++) {
            cp_async_wait<STAGES - 2>();
            __syncthreads();
            if (kc + 1 < nk) load_w((kc + 1) & 1, kc + 1);
            cp_async_commit();
            if (kc + STAGES - 1 < nk) load_x((kc + STAGES - 1) % STAGES, kc + STAGES - 1);
            cp_async_commit();

            uint32_t aw[2][4][4];  // [mtile][k32 step][reg]
            {
                const uint4* d = wsm + (size_t)(kc & 1) * 4 * kThreads + tid;
                const uint4 q00 = d[0], q01 = d[kThreads], q10 = d[2 * kThreads], q11 = d[3 * kThreads];
                dequant_s8(q00.x, q00.y, aw[0][0]);
                dequant_s8(q00.z, q00.w, aw[0][1]);
                dequant_s8(q01.x, q01.y, aw[0][2]);
                dequant_s8(q01.z, q01.w, aw[0][3]);
                dequant_s8(q10.x, q10.y, aw[1][0]);
                dequant_s8(q10.z, q10.w, aw[1][1]);
                dequant_s8(q11.x, q11.y, aw[1][2]);
                dequant_s8(q11.z, q11.w, aw[1][3]);
            }
            const bf16* s0 = reinterpret_cast<const bf16*>(ssm + ((kc & 1) * 8 + warp) * 4);
            const bf16* s1 = s0 + 16;
            const float sw[2][2] = {{__bfloat162float(s0[g]), __bfloat162float(s0[g + 8])},
                                    {__bfloat162float(s1[g]), __bfloat162float(s1[g + 8])}};

            const int stage = kc % STAGES;
            const uint4* xb = xs + stage * TB * 8;
            const float* sb = ss + stage * TB * SP * 2;
#pragma unroll
            for (int jp = 0; jp < NT8 / 2; jp++) {
                float tmp[2][2][4];
                uint32_t cc[2][2][4];
#pragma unroll
                for (int s = 0; s < 4; s++) {
                    const int mi = lane >> 3;
                    const int r = wt * (TB / WT) + (jp * 2 + (mi >> 1)) * 8 + (lane & 7);
                    const int chn = s * 2 + (mi & 1);
                    uint32_t b[4];
                    ldmatrix_x4(b, &xb[swz(r, chn)]);
                    if ((s % (4 / SUB)) == 0) {
#pragma unroll
                        for (int i = 0; i < 2; i++) {
                            mma_s8_init(cc[i][0], aw[i][s], b[0], b[1]);
                            mma_s8_init(cc[i][1], aw[i][s], b[2], b[3]);
                        }
                    } else {
#pragma unroll
                        for (int i = 0; i < 2; i++) {
                            mma_s8(cc[i][0], aw[i][s], b[0], b[1]);
                            mma_s8(cc[i][1], aw[i][s], b[2], b[3]);
                        }
                    }
                    if (((s + 1) % (4 / SUB)) == 0) {  // end of an activation group: flush
                        const int sub = s / (4 / SUB);
#pragma unroll
                        for (int h = 0; h < 2; h++) {
                            const int tk = wt * (TB / WT) + (jp * 2 + h) * 8 + 2 * t;
                            const float2 pa = *reinterpret_cast<const float2*>(sb + (tk * SP + sub) * 2);
                            const float2 pb = *reinterpret_cast<const float2*>(sb + ((tk + 1) * SP + sub) * 2);
#pragma unroll
                            for (int i = 0; i < 2; i++) {
#pragma unroll
                                for (int e = 0; e < 4; e++) {
                                    const float f = __int_as_float(cc[i][h][e]);
                                    const float sx = (e & 1) ? pb.x : pa.x, nm = (e & 1) ? pb.y : pa.y;
                                    // fma(f, sx, -M sx) == (f - M) * sx rounded once (M sx exact)
                                    if (DIRECT) {
                                        acc[i][jp * 2 + h][e] = fmaf(fmaf(f, sx, nm), sw[i][e >> 1], acc[i][jp * 2 + h][e]);
                                    } else if (sub == 0) {
                                        tmp[i][h][e] = fmaf(f, sx, nm);
                                    } else {
                                        tmp[i][h][e] += fmaf(f, sx, nm);
                                    }
                                }
                            }
                        }
                    }
                }
                if (!DIRECT) {
#pragma unroll
                    for (int i = 0; i < 2; i++)
#pragma unroll
                        for (int h = 0; h < 2; h++) {
                            float* a = acc[i][jp * 2 + h];
                            a[0] = fmaf(tmp[i][h][0], sw[i][0], a[0]);
                            a[1] = fmaf(tmp[i][h][1], sw[i][0], a[1]);
                            a[2] = fmaf(tmp[i][h][2], sw[i][1], a[2]);
                            a[3] = fmaf(tmp[i][h][3], sw[i][1], a[3]);
                        }
                }
            }
        }
        cp_async_wait<0>();

        if (part >= 0) {  // split tile: publish the fp32 partial, sk_reduce_kernel sums the parts
            float* pw = wsp + ((size_t)(tile - full) * parts + part) * ROWS * TB;
#pragma unroll
            for (int i = 0; i < 2; i++)
#pragma unroll
                for (int j = 0; j < NT8; j++) {
                    const int lr = wr * 32 + i * 16 + g, lt = wt * (TB / WT) + j * 8 + 2 * t;
                    *reinterpret_cast<float2*>(pw + (size_t)lr * TB + lt) = make_float2(acc[i][j][0], acc[i][j][1]);
                    *reinterpret_cast<float2*>(pw + (size_t)(lr + 8) * TB + lt) = make_float2(acc[i][j][2], acc[i][j][3]);
                }
        }
        const bool emit = part < 0;
        if (emit) {
            if (glu) {
                bf16* act = reinterpret_cast<bf16*>(c);
#pragma unroll
                for (int i = 0; i < 2; i++) {
                    if (!(i == 0 ? v0 : v1)) continue;
                    const int col = (row0 + i * 16) / 2 + g;
#pragma unroll
                    for (int j = 0; j < NT8; j++) {
                        const int tk = tok0 + wt * (TB / WT) + j * 8 + 2 * t;
                        if (tk < T)
                            act[(size_t)tk * ldc + col] =
                                __float2bfloat16(acc[i][j][0] / (1.f + __expf(-acc[i][j][0])) * acc[i][j][2]);
                        if (tk + 1 < T)
                            act[(size_t)(tk + 1) * ldc + col] =
                                __float2bfloat16(acc[i][j][1] / (1.f + __expf(-acc[i][j][1])) * acc[i][j][3]);
                    }
                }
            } else {
#pragma unroll
                for (int i = 0; i < 2; i++) {
                    if (!(i == 0 ? v0 : v1)) continue;
                    const int r = row0 + i * 16 + g;
#pragma unroll
                    for (int j = 0; j < NT8; j++) {
                        const int tk = tok0 + wt * (TB / WT) + j * 8 + 2 * t;
                        if (tk < T) {
                            c[(size_t)tk * ldc + r] = acc[i][j][0];
                            c[(size_t)tk * ldc + r + 8] = acc[i][j][2];
                        }
                        if (tk + 1 < T) {
                            c[(size_t)(tk + 1) * ldc + r] = acc[i][j][1];
                            c[(size_t)(tk + 1) * ldc + r + 8] = acc[i][j][3];
                        }
                    }
                }
            }
        }
        __syncthreads();  // smem stages are reused by the next unit's prologue
    }
}

// ---------------------------------------------------------------- W4A8, activations as the A operand
// Same inputs as gemm_i8_kernel, different inner structure:
// - MMA roles swapped: A = 16 tokens x 32 k (ldmatrix from the x tile), B = 8 weight rows (the
//   packed W4 word of lane (g, t) holds row g / g+8 at k positions 4t..4t+3 and 16+4t.., i.e.
//   exactly the two B fragments of the 8-row halves of a 16-row tile), C = tokens x rows;
// - weights enter the MMA as unsigned nibbles u = w + 8 (s8 x u8 -> s32: unpack = one AND per
//   register, a shift for the odd nibbles); the zero point is folded into the accumulator start
//   C = M_bits - 8 sum_g x_t taken from the pre-pass record, so the integer result M + sum w x is
//   exact and the flush fma(f, sx, -M sx) is unchanged;
// - per activation sub-group (G k) a warp runs all its MMAs (MT token tiles x 4 row tiles,
//   chain depth G / 32) on separate accumulators before flushing them into acc with
//   acc = fma(fma(f, sx, -M sx), sw, acc): the flush of one sub-group overlaps the MMAs of the
//   next instead of serialising on a per-tile accumulator (the old kernel's 57 % tensor pipe).
// Warp tile: (TB / WT) tokens x 32 rows; block: 8 warps = (8 / WT) row groups x WT token groups.
__device__ __forceinline__ void mma_s8u8(uint32_t c[4], const uint32_t a[4], uint32_t b0, uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.u8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, "
        "{%8,%9}, {%0,%1,%2,%3};"
        : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
// first step of a sub-group: C = the record constants of tokens g (c0, c1) and g + 8 (c2, c3)
__device__ __forceinline__ void mma_s8u8_init(uint32_t c[4], const uint32_t a[4], uint32_t b0,
                                              uint32_t b1, uint32_t cg, uint32_t cg8) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.u8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, "
        "{%8,%9}, {%10,%10,%11,%11};"
        : "=r"(c[0]), "=r"(c[1]), "=r"(c[2]), "=r"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(cg), "r"(cg8));
}

// NWB: weight stage buffers (issued NWB - 1 stages ahead); the scale rows have NWB + 1 buffers
template <int WT, int TB, int G, int NST = STAGES, int NWB = 3, bool WREG = true>
constexpr size_t i8t_smem() {
    return (size_t)NST * TB * 128 + (size_t)NST * ((128 / G + 1) / 2) * TB * 16 + (WREG ? 0 : NWB * 4 * (kThreads / WT) * 16) + (NWB + 1) * 8 * 64;
}

template <int WT, int TB, int G, int CCB, int ABL = 0, int MINB = 1, int NST = STAGES, int NWB = 3, bool WREG = true>
__global__ void __launch_bounds__(kThreads, MINB) gemm_i8t_kernel(const uint4* __restrict__ wq,
                                                           const bf16* __restrict__ ws, int N, int K,
                                                           const int8_t* __restrict__ x,
                                                           const float4* __restrict__ xsc, int T,
                                                           float* __restrict__ c, int ldc, bool glu,
                                                           int* __restrict__ ctr, float* __restrict__ wsp,
                                                           int full, int parts, int grp_rows) {
    extern __shared__ __align__(128) uint4 dyn_smem[];
    __shared__ int s_unit;
    constexpr int SUB = BKI / G, KS = 4 / SUB;  // sub-groups per stage, k32 steps per sub-group
    constexpr int NREC = (SUB + 1) / 2;         // float4 records {sx0, C0, sx1, C1} per token and stage
    constexpr int WR = 8 / WT, MT = TB / WT / 16, ROWS = block_rows<WT>();
    static_assert(MT >= 1 && TB % (16 * WT) == 0, "token tile");
    (void)CCB;
    static_assert(G == 32 || G == 64 || G == 128, "group");
    uint4* xs = dyn_smem;                                                       // [NST][TB * 8]
    float4* srec = reinterpret_cast<float4*>(dyn_smem + NST * TB * 8);         // [NST][TB][NREC]
    constexpr int WTHR = kThreads / WT;  // threads of one token group (wt == 0) load the weights
    uint4* wsm = dyn_smem + NST * TB * 8 + NST * NREC * TB;                    // [NWB][4 chunks][WTHR]
    uint4* ssm = wsm + (WREG ? 0 : NWB * 4 * WTHR);                             // [NWB + 1][row warp][4 x 16 B]
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int wr = warp % WR, wt = warp / WR;
    const int g = lane >> 2, t = lane & 3;
    const int nkt = K / BKI;
    const int groups = K >> 7;
    const int rb = (N + ROWS - 1) / ROWS;
    const int tiles = rb * ((T + TB - 1) / TB);
    const int total = full + (tiles - full) * parts;
    const int tokw = wt * (TB / WT);  // first token of the warp within the block tile

    // unit state (set by decode; the loads below read it, so the next unit's prologue runs on
    // the next unit's values while the current unit's epilogue uses its saved copies)
    int tile, kb, nk, part, row0, tok0, rt0;
    bool v0, v1;
    const uint4 *wp0, *wp1;
    const bf16 *sp0, *sp1;
    auto decode = [&](int u) {
        int ke;
        if (u < full) {
            tile = u; kb = 0; ke = nkt; part = -1;
        } else {
            const int r = (u - full) / parts;
            part = (u - full) % parts;
            tile = full + r;
            kb = part * nkt / parts;
            ke = (part + 1) * nkt / parts;
        }
        nk = ke - kb;
        int rbi, tbi;
        {
            const int GRP = min(grp_rows, rb);
            const int tbn = (T + TB - 1) / TB, grp = tile / (GRP * tbn);
            const int gsz = min(GRP, rb - grp * GRP), idx = tile - grp * GRP * tbn;
            tbi = idx / gsz;
            rbi = grp * GRP + idx % gsz;
        }
        row0 = rbi * ROWS + wr * 32;
        tok0 = tbi * TB;
        rt0 = row0 / 16;
        v0 = rt0 < N / 16;
        v1 = rt0 + 1 < N / 16;
        wp0 = wq + (size_t)(v0 ? rt0 : 0) * (K / 64) * 32 + lane;
        wp1 = wq + (size_t)(v1 ? rt0 + 1 : 0) * (K / 64) * 32 + lane;
        // (rows past N read row 0: valid addresses, their results are never stored)
        sp0 = ws + (size_t)(v0 ? rt0 : 0) * groups * 16;
        sp1 = ws + (size_t)(v1 ? rt0 + 1 : 0) * groups * 16;
    };
    uint4 qcur[4], qnxt[4];  // WREG: packed weight words of the stage / the next stage in flight
    bool first = true;
    for (;;) {

        auto load_x = [&](int stage, int kc) {
            uint4* xb = xs + stage * TB * 8;
#pragma unroll
            for (int i = 0; i < (TB * 8 + kThreads - 1) / kThreads; i++) {
                const int idx = tid + i * kThreads;
                if (idx >= TB * 8) break;
                const int r = idx >> 3, ch = idx & 7;
                const int tok = tok0 + r;
                const int8_t* src = x + (size_t)min(tok, T - 1) * K + ((ABL & 1) ? 0 : (kb + kc) * BKI) + ch * 16;
                cp_async16(&xb[swz(r, ch)], src, tok < T);
            }
            if (!(ABL & 2))
#pragma unroll
            for (int i = 0; i < (TB * NREC + kThreads - 1) / kThreads; i++) {  // records: [token][NREC]
                const int idx = tid + i * kThreads;
                if (idx >= TB * NREC) break;
                const int tl = idx / NREC, r = idx % NREC, tok = tok0 + tl;
                const float4* src = xsc + ((size_t)min(tok, T - 1) * groups + ((ABL & 1) ? 0 : kb + kc)) * NREC + r;
                cp_async16(srec + (stage * TB + tl) * NREC + r, src, tok < T);
            }
        };

        auto load_w = [&](int buf, int kc) {
            if ((ABL & (8 | 64)) || wt != 0) return;
            const size_t ci = (size_t)((ABL & 1) ? 0 : kb + kc) * 2;
            if constexpr (!WREG) {
                uint4* d = wsm + (size_t)buf * 4 * WTHR + wr * 32 + lane;
                cp_async16(d + 0 * WTHR, wp0 + ci * 32, v0);
                cp_async16(d + 1 * WTHR, wp0 + (ci + 1) * 32, v0);
                cp_async16(d + 2 * WTHR, wp1 + ci * 32, v1);
                cp_async16(d + 3 * WTHR, wp1 + (ci + 1) * 32, v1);
            }
            if (lane < 4)
                cp_async16(ssm + ((kc % (NWB + 1)) * 8 + wr) * 4 + lane,
                           (lane < 2 ? sp0 : sp1) + ((size_t)((ABL & 1) ? 0 : kb + kc) * 16) + (lane & 1) * 8, true);
        };

        auto load_wreg = [&](int kcs, uint4 qd[4]) {
            if (ABL & 8) return;
            const size_t ci = (size_t)((ABL & 1) ? 0 : kb + kcs) * 2;
            qd[0] = __ldg(wp0 + ci * 32);
            qd[1] = __ldg(wp0 + (ci + 1) * 32);
            qd[2] = __ldg(wp1 + ci * 32);
            qd[3] = __ldg(wp1 + (ci + 1) * 32);
        };
        // prologue groups: [w(0) x(0)] [w(1) x(1)] ... (NST - 1 groups); each iteration kc then commits
        // [w(kc + NWB - 1)] at the top and [x(kc + NST - 1)] after the first sub-group, so the
        // wait<NST - 2> + barrier at the top of iteration kc + 1 guarantees w(kc + 1) and x(kc + 1)
        static_assert(NWB - 1 <= NST - 1, "weight prefetch distance");
        auto prologue = [&]() {
#pragma unroll
            for (int s = 0; s < NST - 1; s++) {
                if (s < nk && s < NWB - 1) load_w(s % NWB, s);
                if (s < nk) load_x(s, s);
                cp_async_commit();
            }
            if constexpr (WREG) load_wreg(0, qnxt);
        };
        if (first) {
            first = false;
            if (tid == 0) s_unit = atomicAdd(ctr, 1);
            __syncthreads();
            if (s_unit >= total) break;
            decode(s_unit);
            prologue();
        }
        float acc[MT][4][4];
#pragma unroll
        for (int m = 0; m < MT; m++)
#pragma unroll
            for (int n = 0; n < 4; n++)
#pragma unroll
                for (int e = 0; e < 4; e++) acc[m][n][e] = 0.f;
        uint32_t cc[2][4][4];  // ring of integer accumulator sets (tile parity)
        float3 rec_p[2];       // record of the pending (last) tile of the previous stage
        static_assert((SUB * MT) % 2 == 0, "tile ring parity");
        auto load_sw = [&](int kcs, float swf[4][2]) {  // weight scales of stage kcs: tile n/2, rows (n & 1) * 8 + 2t, +1
            if (ABL & 8) {
#pragma unroll
                for (int n = 0; n < 4; n++) { swf[n][0] = 0.01f * (n + 1); swf[n][1] = 0.02f; }
            } else {
                const bf16* s0 = reinterpret_cast<const bf16*>(ssm + ((kcs % (NWB + 1)) * 8 + wr) * 4);
#pragma unroll
                for (int n = 0; n < 4; n++) {
                    const float2 f = __bfloat1622float2(*reinterpret_cast<const bf162*>(s0 + n * 8 + 2 * t));
                    swf[n][0] = f.x;
                    swf[n][1] = f.y;
                }
            }
        };
        auto flush_last = [&](int m, const uint32_t cm[4][4], const float3 r[2], const float swf[4][2]) {
#pragma unroll
            for (int n = 0; n < 4; n++)
#pragma unroll
                for (int e = 0; e < 4; e++) {
                    const float f = __int_as_float(cm[n][e]);
                    const float3 rr = r[e >> 1];
                    acc[m][n][e] = fmaf(fmaf(f, rr.x, rr.y), swf[n][e & 1], acc[m][n][e]);
                }
        };

        for (int kc = 0; kc < nk; kc++) {
            if (!(ABL & 16)) {
                cp_async_wait<NST - 2>();
                __syncthreads();
            }
            if (kc + NWB - 1 < nk) load_w((kc + NWB - 1) % NWB, kc + NWB - 1);
            cp_async_commit();
            if constexpr (WREG) {
#pragma unroll
                for (int i = 0; i < 4; i++) qcur[i] = qnxt[i];
                if (kc + 1 < nk) load_wreg(kc + 1, qnxt);
            }
            if (ABL & 128) {
                if (!(ABL & 32) && kc + NST - 1 < nk) load_x((kc + NST - 1) % NST, kc + NST - 1);
                cp_async_commit();
            }

            float sw[4][2];
            load_sw(kc, sw);

            const int stage = kc % NST;
            const uint4* xb = xs + stage * TB * 8;
            const float2* rb0 = reinterpret_cast<const float2*>(srec + (stage * TB + tokw + g) * NREC);
            const uint4* wd = wsm + (size_t)(kc % NWB) * 4 * WTHR + wr * 32 + lane;
            auto load_rec = [&](int sub, int m, float3 r[2]) {  // (sx, -M sx, C) of tokens g, g + 8 of tile m
                if (ABL & 2) {
                    const float sx = 1e-3f * (m + 1 + sub);
                    r[0] = make_float3(sx, -kMagic * sx, __int_as_float((int)kMagicBits - 8 * (m + kc)));
                    r[1] = make_float3(sx * 0.5f, -kMagic * sx * 0.5f, __int_as_float((int)kMagicBits - 8 * (sub + 1)));
                } else {
                    const float2 ra = rb0[(m * 16) * NREC * 2 + sub], rc = rb0[(m * 16 + 8) * NREC * 2 + sub];
                    r[0] = make_float3(ra.x, -kMagic * ra.x, ra.y);
                    r[1] = make_float3(rc.x, -kMagic * rc.x, rc.y);
                }
            };
            auto load_a = [&](int s, int m, int sub, uint32_t a[4]) {
                if (ABL & 4) {
                    a[0] = 0x01010101u * (m + 1) + s; a[1] = 0x02020202u + kc; a[2] = 0x03030303u + sub; a[3] = 0x04040404u;
                } else {
                    const int r = tokw + m * 16 + (lane & 7) + ((lane >> 3) & 1) * 8;
                    ldmatrix_x4(a, &xb[swz(r, 2 * s + (lane >> 4))]);
                }
            };
            // The stage's SUB * MT token tiles j = sub * MT + m stream through a ring of 2 integer
            // accumulator sets; the flush of tile j - 1 (acc += ((f - M) sx) sw) is issued after
            // the MMAs of tile j, also across the stage boundary (the last tile of a stage is flushed
            // after the first MMAs of the next stage with the previous stage's weight scales), so a
            // warp reaches the barrier with MMAs queued and never waits on its own flush.
            auto flush = [&](int m, const uint32_t cm[4][4], const float3 r[2], const float swf[4][2]) {
#pragma unroll
                for (int n = 0; n < 4; n++)
#pragma unroll
                    for (int e = 0; e < 4; e++) {
                        const float f = __int_as_float(cm[n][e]);
                        const float3 rr = r[e >> 1];
                        acc[m][n][e] = fmaf(fmaf(f, rr.x, rr.y), swf[n][e & 1], acc[m][n][e]);
                    }
            };
            float3 rec[2][2];  // ring: tiles j - 1, j (slot j & 1)
#pragma unroll
            for (int sub = 0; sub < SUB; sub++) {
                // B fragments of this sub-group: bw[n][ks] for row tile n (0/1: rows 0-7 / 8-15 of
                // 16-row tile 0, 2/3: tile 1), k32 step ks; unsigned nibbles (even / odd of a word)
                uint32_t bw[4][KS][2];
                if (ABL & 8) {
#pragma unroll
                    for (int n = 0; n < 4; n++)
#pragma unroll
                        for (int s = 0; s < KS; s++) { bw[n][s][0] = 0x01020304u * (n + 1) + kc; bw[n][s][1] = 0x05060708u + s + sub; }
                } else {
                    constexpr uint32_t M4 = 0x0F0F0F0Fu;
#pragma unroll
                    for (int i = 0; i < 2; i++)  // 16-row tile i
#pragma unroll
                        for (int ks = 0; ks < KS; ks++) {
                            const int s = sub * KS + ks;  // k32 step of the stage: chunk s / 2, words (s & 1) * 2, +1
                            const uint4 q = WREG ? qcur[2 * i + (s >> 1)] : wd[(2 * i + (s >> 1)) * WTHR];
                            const uint32_t w0 = (s & 1) ? q.z : q.x, w1 = (s & 1) ? q.w : q.y;
                            bw[2 * i][ks][0] = w0 & M4;            bw[2 * i][ks][1] = w1 & M4;
                            bw[2 * i + 1][ks][0] = (w0 >> 4) & M4; bw[2 * i + 1][ks][1] = (w1 >> 4) & M4;
                        }
                }
#pragma unroll
                for (int m = 0; m < MT; m++) {
                    const int j = sub * MT + m;  // tile index within the stage
                    load_rec(sub, m, rec[j & 1]);
                    const float3* rj = rec[j & 1];
                    const uint32_t cg = __float_as_uint(rj[0].z), cg8 = __float_as_uint(rj[1].z);
                    uint32_t (*cj)[4] = cc[j & 1];
#pragma unroll
                    for (int ks = 0; ks < KS; ks++) {
                        uint32_t a[4];
                        load_a(sub * KS + ks, m, sub, a);
                        if (ks == 0) {
#pragma unroll
                            for (int n = 0; n < 4; n++) mma_s8u8_init(cj[n], a, bw[n][ks][0], bw[n][ks][1], cg, cg8);
                        } else {
#pragma unroll
                            for (int n = 0; n < 4; n++) mma_s8u8(cj[n], a, bw[n][ks][0], bw[n][ks][1]);
                        }
                    }
                    if (j == 0) {
                        if (kc > 0) {  // last tile of the previous stage, with that stage's weight scales
                            float swp[4][2];
                            load_sw(kc - 1, swp);
                            flush(MT - 1, cc[1], rec_p, swp);
                        }
                    } else {
                        flush((j - 1) % MT, cc[(j - 1) & 1], rec[(j - 1) & 1], sw);
                    }
                    if (j == MT - 1 && !(ABL & 128)) {  // next x stage, issued away from the barrier
                        if (!(ABL & 32) && kc + NST - 1 < nk) load_x((kc + NST - 1) % NST, kc + NST - 1);
                        cp_async_commit();
                    }
                }
            }
            // carry the last tile (slot 1: even tile count per stage) to the next stage
#pragma unroll
            for (int h = 0; h < 2; h++) rec_p[h] = rec[1][h];
        }
        if (nk > 0) {
            float swp[4][2];
            load_sw(nk - 1, swp);
            flush_last(MT - 1, cc[1], rec_p, swp);
        }
        cp_async_wait<0>();
        __syncthreads();  // every warp is done with the stage buffers: the next unit's prologue may refill them
        const int e_part = part, e_tile = tile, e_row0 = row0, e_tok0 = tok0;
        const bool e_v0 = v0, e_v1 = v1;
        if (tid == 0) s_unit = atomicAdd(ctr, 1);
        __syncthreads();
        const bool more = s_unit < total;
        if (more) {  // the next unit's first stages land while this unit stores its outputs
            decode(s_unit);
            prologue();
        }

        if (e_part >= 0) {  // split tile: fp32 partial [TB tokens][ROWS] for sk_reduce_kernel (tokmajor)
            float* pw = wsp + ((size_t)(e_tile - full) * parts + e_part) * ROWS * TB;
#pragma unroll
            for (int m = 0; m < MT; m++)
#pragma unroll
                for (int n = 0; n < 4; n++) {
                    const int lt = tokw + m * 16 + g, lr = wr * 32 + n * 8 + 2 * t;
                    *reinterpret_cast<float2*>(pw + (size_t)lt * ROWS + lr) = make_float2(acc[m][n][0], acc[m][n][1]);
                    *reinterpret_cast<float2*>(pw + (size_t)(lt + 8) * ROWS + lr) = make_float2(acc[m][n][2], acc[m][n][3]);
                }
        } else if (glu) {  // tile i: rows (n = 2i) gate, (n = 2i + 1) up of activation columns tile*8 + 2t, +1
            bf16* act = reinterpret_cast<bf16*>(c);
#pragma unroll
            for (int i = 0; i < 2; i++) {
                if (!(i == 0 ? e_v0 : e_v1)) continue;
                const int col = (e_row0 + i * 16) / 2 + 2 * t;
#pragma unroll
                for (int m = 0; m < MT; m++)
#pragma unroll
                    for (int h = 0; h < 2; h++) {
                        const int tk = e_tok0 + tokw + m * 16 + g + h * 8;
                        if (tk < T) {
                            const float g0 = acc[m][2 * i][2 * h], g1 = acc[m][2 * i][2 * h + 1];
                            const float u0 = acc[m][2 * i + 1][2 * h], u1 = acc[m][2 * i + 1][2 * h + 1];
                            *reinterpret_cast<bf162*>(act + (size_t)tk * ldc + col) = __floats2bfloat162_rn(
                                g0 / (1.f + __expf(-g0)) * u0, g1 / (1.f + __expf(-g1)) * u1);
                        }
                    }
            }
        } else {
#pragma unroll
            for (int n = 0; n < 4; n++) {
                if (!((n < 2) ? e_v0 : e_v1)) continue;
                const int r = e_row0 + n * 8 + 2 * t;
#pragma unroll
                for (int m = 0; m < MT; m++)
#pragma unroll
                    for (int h = 0; h < 2; h++) {
                        const int tk = e_tok0 + tokw + m * 16 + g + h * 8;
                        if (tk < T)
                            *reinterpret_cast<float2*>(c + (size_t)tk * ldc + r) =
                                make_float2(acc[m][n][2 * h], acc[m][n][2 * h + 1]);
                    }
            }
        }
        if (!more) break;
    }
}

// init step with the C quad of the record (4 distinct PTX registers: ptxas uses the LDS.128 quad)
__device__ __forceinline__ void mma_s8u8_initq(uint32_t c[4], const uint32_t a[4], uint32_t b0, uint32_t b1, const float4 q) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.u8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, "
        "{%8,%9}, {%10,%11,%12,%13};"
        : "=r"(c[0]), "=r"(c[1]), "=r"(c[2]), "=r"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(__float_as_uint(q.x)),
          "r"(__float_as_uint(q.y)), "r"(__float_as_uint(q.z)), "r"(__float_as_uint(q.w)));
}

template <uint32_t M> __device__ __forceinline__ uint32_t and_v(uint32_t a) {
    uint32_t r;
    asm volatile("and.b32 %0, %1, %2;" : "=r"(r) : "r"(a), "n"(M));
    return r;
}
__device__ __forceinline__ uint4 ldg_v(const uint4* p) {
    uint4 r;
    asm volatile("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
    return r;
}
__device__ __forceinline__ uint4 ldg_v_na(const uint4* p) {  // streamed once per block: no L1 allocation
    uint4 r;
    asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
    return r;
}

template <uint32_t M, uint32_t X> __device__ __forceinline__ uint32_t andxor_v(uint32_t a) {
    uint32_t r;  // (a & M) ^ X: one LOP3
    asm volatile("lop3.b32 %0, %1, %2, %3, 0x6A;" : "=r"(r) : "r"(a), "n"(M), "n"(X));
    return r;
}
__device__ __forceinline__ void mma_s8s8_initq(uint32_t c[4], const uint32_t a[4], uint32_t b0, uint32_t b1, const uint4 q) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, "
        "{%8,%9}, {%10,%11,%12,%13};"
        : "=r"(c[0]), "=r"(c[1]), "=r"(c[2]), "=r"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(q.x), "r"(q.y), "r"(q.z), "r"(q.w));
}
// x tile rows of 144 B (128 B + 16 B pad): ldmatrix rows of a matrix land in 8 distinct 16-B bank
// groups without an XOR swizzle, so every per-step offset is an immediate
constexpr int kXRow = 144;
template <int WT, int TB, int G, int HM, int NST = STAGES, int OPT = 0>
constexpr size_t i8v_smem() {
    return (size_t)NST * TB * kXRow + (size_t)NST * (TB / 16) * v_blk<G, HM == 1>() * 16 + 4 * 8 * 64 + 16 + 16 * NST + 16 + 4 * 32 +
           ((OPT & 524288) ? (size_t)NST * 8 * 4 * 32 * 16 : 0);
}

// gemm_i8c_kernel: thread 0 decodes stream-K unit u (fields u, tile, kb, nk, part, rbi, tbi) into ring slot
// sl, releases tag n + 1 at tg and fetches the next unit id (returned); out of line so the integer
// divisions stay out of the K loop's register allocation
__device__ __noinline__ int i8c_publish(int u, int n, unsigned sl, unsigned tg, int total, int full, int parts, int nkt,
                                        int grp_rows, int rb, int tbn, int* ctr) {
    int f[7] = {u, 0, 0, 0, 0, 0, 0};
    if (u < total) {
        int tile, kb, ke, part;
        if (u < full) {
            tile = u; kb = 0; ke = nkt; part = -1;
        } else {
            const int r = (u - full) / parts;
            part = (u - full) % parts;
            tile = full + r;
            kb = part * nkt / parts;
            ke = (part + 1) * nkt / parts;
        }
        const int GRP = min(grp_rows, rb);
        const int grp = tile / (GRP * tbn);
        const int gsz = min(GRP, rb - grp * GRP), idx = tile - grp * GRP * tbn;
        f[1] = tile; f[2] = kb; f[3] = ke - kb; f[4] = part; f[5] = grp * GRP + idx % gsz; f[6] = idx / gsz;
    }
    asm volatile("st.shared.v4.u32 [%0], {%1,%2,%3,%4};" ::"r"(sl), "r"(f[0]), "r"(f[1]), "r"(f[2]), "r"(f[3]) : "memory");
    asm volatile("st.shared.v4.u32 [%0], {%1,%2,%3,%4};" ::"r"(sl + 16), "r"(f[4]), "r"(f[5]), "r"(f[6]), "r"(0) : "memory");
    asm volatile("st.release.cta.shared.u32 [%0], %1;" ::"r"(tg), "r"(n + 1) : "memory");
    return atomicAdd(ctr, 1);
}

template <int WT, int TB, int G, int HM, int MINB = 1, int NST = STAGES, int OPT = 0>
__global__ void __launch_bounds__(kThreads, MINB) gemm_i8v_kernel(const uint4* __restrict__ wq,
                                                           const bf16* __restrict__ ws, int N, int K,
                                                           const int8_t* __restrict__ x,
                                                           const float4* __restrict__ xsc, int T,
                                                           float* __restrict__ c, int ldc, bool glu,
                                                           int* __restrict__ ctr, float* __restrict__ wsp,
                                                           int full, int parts, int grp_rows,
                                                           int8_t* __restrict__ gq, float* __restrict__ gqs, bool o16) {
    extern __shared__ __align__(128) uint4 dyn_smem[];
    __shared__ int s_unit, s_skf;
    __shared__ float s_gred[2][TB][8];  // GLU quantization: per-warp amax / int sums per token
    constexpr int NWB = 3;
    constexpr int SUB = BKI / G, KS = 4 / SUB;  // sub-groups per stage, k32 steps per sub-group
    constexpr bool MB = OPT & 1, PF = OPT & 2;
    constexpr bool CSL = OPT & 8;  // continuous stream across units (requires MB, nk >= 3)
    static_assert(!CSL || MB, "CSL needs the mbarrier pipeline");
    constexpr bool C1 = OPT & 4096;  // single accumulator set, immediate flush
    constexpr int NSL = (OPT & 8192) ? 32 : (OPT & 16384) ? 256 : (OPT & 2097152) ? 64 : (OPT & 4194304) ? 128
                      : (OPT & 8388608) ? 16 : 0;  // mbarrier spin back-off (ns)
    constexpr bool WSM = OPT & 524288;          // weight words through shared memory (x stream)
    constexpr bool NAL = OPT & 16777216;        // weight LDGs without L1 allocation
    constexpr bool CS2 = OPT & 1048576;         // one load stream across unit boundaries (needs WSM, MB)
    static_assert(!WSM || !CSL, "WSM replaces the register weight stream CSL switches");
    constexpr bool SKF = OPT & 131072;          // in-kernel stream-K fixup (tickets in ctr[1 + split tile])
    constexpr bool SEP = OPT & 65536;           // ... without the block barrier (spin on a published unit id)
    constexpr bool EP = (OPT & 32768) || SEP;  // next unit's prologue in the last stage (needs MB)
    static_assert(!EP || (MB && !CSL), "EP needs the mbarrier pipeline");
    constexpr int AB = OPT & ~15 & ~4096 & ~8192 & ~16384 & ~32768 & ~65536 & ~131072 & ~524288 & ~1048576 & ~2097152 &
                       ~4194304 & ~8388608 & ~16777216;  // ablation bits (bench only)
    constexpr bool HI = HM == 1, HS = HM == 2;  // HI: C_hi records; HS: high nibbles as signed 16 w
    constexpr int KN = v_kinds<HI>(), BLK = v_blk<G, HI>();
    constexpr int WR = 8 / WT, MT = TB / WT / 16, ROWS = block_rows<WT>();
    constexpr int XST = TB * kXRow, RST = (TB / 16) * BLK * 16;  // bytes per stage: x tile, records
    constexpr int NXI = TB * 8 / kThreads, NRI = ((TB / 16) * BLK + kThreads - 1) / kThreads;
    static_assert(MT >= 1 && TB % (16 * WT) == 0, "token tile");
    static_assert(G == 32 || G == 64 || G == 128, "group");
    static_assert(TB * 8 % kThreads == 0, "x chunks per thread");
    static_assert(NWB - 1 <= NST - 1, "weight prefetch distance");
    char* const smb = reinterpret_cast<char*>(dyn_smem);
    const unsigned sm0 = __cvta_generic_to_shared(smb);                 // x stages [NST][TB][144 B]
    const unsigned srec0 = sm0 + NST * XST;                             // records [NST][TB / 16][BLK]
    const unsigned ssm0 = srec0 + NST * RST;                            // weight scales [NWB + 1][8][64 B]
    const unsigned mb0 = ssm0 + 4 * 8 * 64 + 16;                        // mbarriers full[NST], empty[NST]
    const unsigned wsm0 = mb0 + 16 * NST + 16 + 4 * 32;                // WSM: weight words [NST][8 warps][4][32 lanes]
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int wr = warp % WR, wt = warp / WR;
    const int g = lane >> 2, t = lane & 3;
    const int nkt = K / BKI;
    const int groups = K >> 7;
    const int rb = (N + ROWS - 1) / ROWS;
    const int tiles = rb * ((T + TB - 1) / TB);
    const int total = full + (tiles - full) * parts;
    const int tokw = wt * (TB / WT);
    // lane parts of the smem reads: ldmatrix row (tokens r, r + 8 of a 16-token tile) and the chunk
    // of the lane's 8 x 8 matrix, record pair g of the warp's first 16-token tile, scale words
    const unsigned a_lane = sm0 + (tokw + (lane & 7) + ((lane >> 3) & 1) * 8) * kXRow + (lane >> 4) * 16;
    const unsigned r_lane = srec0 + (tokw / 16) * BLK * 16 + g * 16;
    const unsigned s_lane = ssm0 + wr * 64 + t * 4;

    int tile, kb, nk, part, row0, tok0;
    bool v0, v1;
    // per-unit running sources of this thread (decode sets them, each stage advances them)
    const int8_t* xg[NXI];
    bool xok[NXI];
    const float4* rg[NRI];
    bool rok[NRI];
    const uint4 *wg0, *wg1;  // packed weight words, row tiles rt0 / rt0 + 1 (stage = 2 chunks = 64 uint4)
    const bf16* sg;          // weight-scale row of this lane (lane < 4 of the wt == 0 warps)
    int kx = 0, kw = 0;      // global counts of issued stage loads: x / records, weight scales
    auto ptr_q = [&](int rbi_, int kb_) {  // weight words of row block rbi_ from stage kb_
        const int rt = (rbi_ * ROWS + wr * 32) / 16;
        wg0 = wq + ((size_t)(rt < N / 16 ? rt : 0) * (K / 64) + kb_ * 2) * 32 + lane;
        wg1 = wq + ((size_t)(rt + 1 < N / 16 ? rt + 1 : 0) * (K / 64) + kb_ * 2) * 32 + lane;
    };
    auto ptr_s = [&](int rbi_, int kb_) {  // weight-scale row of row block rbi_ from stage kb_
        const int rt = (rbi_ * ROWS + wr * 32) / 16;
        const bool a0 = rt < N / 16, a1 = rt + 1 < N / 16;
        sg = ws + (size_t)((lane < 2 ? a0 : a1) ? rt + (lane >> 1 & 1) : 0) * groups * 16 + kb_ * 16 + (lane & 1) * 8;
    };
    auto ptr_x = [&](int tok0_, int kb_) {
#pragma unroll
        for (int i = 0; i < NXI; i++) {
            const int idx = tid + i * kThreads, tok = tok0_ + (idx >> 3);
            xg[i] = x + (size_t)min(tok, T - 1) * K + kb_ * BKI + (idx & 7) * 16;
            xok[i] = tok < T;
        }
#pragma unroll
        for (int i = 0; i < NRI; i++) {
            const int idx = tid + i * kThreads, tt = (tok0_ >> 4) + idx / BLK;
            rok[i] = tt * 16 < T;
            rg[i] = xsc + ((size_t)(rok[i] ? tt : 0) * groups + kb_) * BLK + idx % BLK;
        }
    };
    auto set_unit = [&](int tile_, int kb_, int nk_, int part_, int rbi_, int tbi_) {
        tile = tile_; kb = kb_; nk = nk_; part = part_;
        row0 = rbi_ * ROWS + wr * 32;
        tok0 = tbi_ * TB;
        const int rt0 = row0 / 16;
        v0 = rt0 < N / 16;
        v1 = rt0 + 1 < N / 16;
    };
    auto decode = [&](int u, bool with_ptr = true) {
        (void)with_ptr;
        int ke;
        if (u < full) {
            tile = u; kb = 0; ke = nkt; part = -1;
        } else {
            const int r = (u - full) / parts;
            part = (u - full) % parts;
            tile = full + r;
            kb = part * nkt / parts;
            ke = (part + 1) * nkt / parts;
        }
        nk = ke - kb;
        int rbi, tbi;
        {
            const int GRP = min(grp_rows, rb);
            const int tbn = (T + TB - 1) / TB, grp = tile / (GRP * tbn);
            const int gsz = min(GRP, rb - grp * GRP), idx = tile - grp * GRP * tbn;
            tbi = idx / gsz;
            rbi = grp * GRP + idx % gsz;
        }
        row0 = rbi * ROWS + wr * 32;
        tok0 = tbi * TB;
        const int rt0 = row0 / 16;
        v0 = rt0 < N / 16;
        v1 = rt0 + 1 < N / 16;
        if (EP && !with_ptr) return;
        wg0 = wq + ((size_t)(v0 ? rt0 : 0) * (K / 64) + kb * 2) * 32 + lane;
        wg1 = wq + ((size_t)(v1 ? rt0 + 1 : 0) * (K / 64) + kb * 2) * 32 + lane;
        sg = ws + (size_t)((lane < 2 ? v0 : v1) ? rt0 + (lane >> 1 & 1) : 0) * groups * 16 + kb * 16 + (lane & 1) * 8;
#pragma unroll
        for (int i = 0; i < NXI; i++) {
            const int idx = tid + i * kThreads, tok = tok0 + (idx >> 3);
            xg[i] = x + (size_t)min(tok, T - 1) * K + kb * BKI + (idx & 7) * 16;
            xok[i] = tok < T;
        }
#pragma unroll
        for (int i = 0; i < NRI; i++) {
            const int idx = tid + i * kThreads, tt = (tok0 >> 4) + idx / BLK;
            rok[i] = tt * 16 < T;
            rg[i] = xsc + ((size_t)(rok[i] ? tt : 0) * groups + kb) * BLK + idx % BLK;
        }
    };
    // CS2: the load stream's unit (sequence xs_n, stages left xs_left) and the on-demand unit ring
    int xs_n = -1, xs_left = 0, pub_n = 0, nxt2 = 0;
    bool xs_done = false;
    const unsigned ur2 = mb0 + 16 * NST;  // ring: tags[4], slots[4][8 ints] (same place as the CSL ring)
    auto cs_get = [&](int n, int f[7]) {
        const unsigned tg = ur2 + (n & 3) * 4, sl = ur2 + 16 + (n & 3) * 32;
        if (tid == 0 && n > pub_n) {
            nxt2 = i8c_publish(nxt2, n, sl, tg, total, full, parts, nkt, grp_rows, rb, (T + TB - 1) / TB, ctr);
            pub_n = n;
        } else if (tid != 0) {
            uint32_t v;
            do {
                asm volatile("ld.acquire.cta.shared.u32 %0, [%1];" : "=r"(v) : "r"(tg) : "memory");
            } while (v != (uint32_t)(n + 1));
        }
        int dummy;
        asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(f[0]), "=r"(f[1]), "=r"(f[2]), "=r"(f[3]) : "r"(sl) : "memory");
        asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(f[4]), "=r"(f[5]), "=r"(f[6]), "=r"(dummy) : "r"(sl + 16) : "memory");
        (void)dummy;
    };
    auto load_x = [&]() {  // x tile + records of the next stage into buffer kx % NST, advance
        if constexpr (CS2) {  // the stream moves to the next unit by itself
            while (xs_left == 0 && !xs_done) {
                int f[7];
                cs_get(++xs_n, f);
                if (f[0] >= total) {
                    xs_done = true;
                } else {
                    ptr_x(f[6] * TB, f[2]);
                    ptr_q(f[5], f[2]);
                    ptr_s(f[5], f[2]);
                    xs_left = f[3];
                }
            }
            if (xs_done) return;
        }
        const int stage = kx % NST;
#pragma unroll
        for (int i = 0; i < NXI; i++) {
            const int idx = tid + i * kThreads;
            if (MB && i == 0 && kx >= NST) {  // the buffer's previous stage must be consumed by all warps
                const unsigned eb = mb0 + NST * 8 + stage * 8;
                const unsigned par = ((kx / NST) + 1) & 1;
                uint32_t ok;
                do {
                    asm volatile("{ .reg .pred p; mbarrier.test_wait.parity.shared.b64 p, [%1], %2; selp.u32 %0, 1, 0, p; }"
                                 : "=r"(ok) : "r"(eb), "r"(par) : "memory");
                    if (NSL && !ok) __nanosleep(NSL);
                } while (!ok);
            }
            const unsigned d = sm0 + stage * XST + (idx >> 3) * kXRow + (idx & 7) * 16;
            if (!(AB & 128)) asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(d), "l"(xg[i]), "r"(xok[i] ? 16 : 0));
            xg[i] += BKI;
        }
#pragma unroll
        for (int i = 0; i < NRI; i++) {
            const int idx = tid + i * kThreads;
            if ((TB / 16) * BLK % kThreads == 0 || idx < (TB / 16) * BLK) {
                const unsigned d = srec0 + stage * RST + idx * 16;
                if (!(AB & 128)) asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(d), "l"(rg[i]), "r"(rok[i] ? 16 : 0));
            }
            rg[i] += BLK;
        }
        if constexpr (WSM) {  // this lane's 4 packed weight words of the stage (row tiles rt0, rt0 + 1, 2 chunks)
            const unsigned d = wsm0 + ((stage * 8 + warp) * 4 * 32 + lane) * 16;
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16, 16;" ::"r"(d), "l"(wg0));
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16, 16;" ::"r"(d + 32 * 16), "l"(wg0 + 32));
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16, 16;" ::"r"(d + 2 * 32 * 16), "l"(wg1));
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16, 16;" ::"r"(d + 3 * 32 * 16), "l"(wg1 + 32));
            wg0 += 64;
            wg1 += 64;
        }
        if constexpr (CS2) {  // weight-scale rows of the same stage (ring by global stage)
            if (wt == 0 && lane < 4) {
                const unsigned d = ssm0 + ((kx & 3) * 8 + wr) * 64 + lane * 16;
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16, 16;" ::"r"(d), "l"(sg));
            }
            sg += 16;
            xs_left--;
        }
        if constexpr (MB) {
            const unsigned fb = mb0 + stage * 8;  // full[stage]: completes when every thread's copies landed
            asm volatile("cp.async.mbarrier.arrive.noinc.shared.b64 [%0];" ::"r"(fb));
        }
        kx++;
    };
    auto load_w = [&]() {  // weight-scale rows of stage kw (16 B x 4 per row warp), advance
        if constexpr (CS2) return;  // (part of the load stream)
        if (wt == 0 && lane < 4) {
            const unsigned d = ssm0 + ((kw & 3) * 8 + wr) * 64 + lane * 16;  // ring by global stage
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16, 16;" ::"r"(d), "l"(sg));
        }
        sg += 16;
        kw++;
    };
    uint4 qw[4];  // packed weight words of the next stage (in flight; unpacked at the stage top)
    auto load_wreg = [&]() {  // volatile: stays behind the (volatile) unpack of the previous words
        if constexpr (WSM) return;
        if (AB & 64) {
            qw[0] = make_uint4(tid, 1, 2, 3); qw[1] = make_uint4(4, tid, 6, 7); qw[2] = make_uint4(8, 9, tid, 11); qw[3] = make_uint4(12, 13, 14, tid);
        } else {
        if constexpr (NAL) {
            qw[0] = ldg_v_na(wg0);
            qw[1] = ldg_v_na(wg0 + 32);
            qw[2] = ldg_v_na(wg1);
            qw[3] = ldg_v_na(wg1 + 32);
        } else {
            qw[0] = ldg_v(wg0);
            qw[1] = ldg_v(wg0 + 32);
            qw[2] = ldg_v(wg1);
            qw[3] = ldg_v(wg1 + 32);
        }
        }
        wg0 += 64;
        wg1 += 64;
    };
    auto prologue = [&]() {
        if constexpr (CS2) {
#pragma unroll
            for (int s = 0; s < NST - 1; s++) load_x();
            return;
        }
#pragma unroll
        for (int s = 0; s < NST - 1; s++) {
            if (s < nk && s < NWB - 1) load_w();
            if (s < nk) load_x();
            cp_async_commit();
        }
        load_wreg();
    };
    // magic accumulator start of the signed high-nibble tiles, read from shared memory once per stage:
    // a constant would be rematerialized by ptxas into every init MMA's C quad (4 MOVs each)
    uint4* const mq_s = reinterpret_cast<uint4*>(smb + NST * XST + NST * RST + 4 * 8 * 64);
    if (HS && tid == 0) *mq_s = make_uint4(kMagicBits, kMagicBits, kMagicBits, kMagicBits);
    const unsigned ur0 = mb0 + 16 * NST;  // CSL unit ring: tags[4], then 4 slots of 8 ints
    int nun = 0, nxt = 0;                 // CSL: sequence number of the current unit; (thread 0) next unit id
    int ep_next = 0, ep_u = 0;            // EP: (thread 0) fetched next unit id; next unit id of all threads
    int ep_nk = 0;                        // EP: stages of the next unit
    auto sep_pub = [&](int n, int u) {    // SEP (thread 0): unit id of unit sequence n into ring slot n & 3
        asm volatile("st.shared.u32 [%0], %1;" ::"r"(ur0 + 16 + (n & 3) * 4), "r"(u) : "memory");
        asm volatile("st.release.cta.shared.u32 [%0], %1;" ::"r"(ur0 + (n & 3) * 4), "r"(n + 1) : "memory");
    };
    auto sep_get = [&](int n) -> int {
        uint32_t v;
        do {
            asm volatile("ld.acquire.cta.shared.u32 %0, [%1];" : "=r"(v) : "r"(ur0 + (n & 3) * 4) : "memory");
        } while (v != (uint32_t)(n + 1));
        uint32_t u;
        asm volatile("ld.shared.u32 %0, [%1];" : "=r"(u) : "r"(ur0 + 16 + (n & 3) * 4) : "memory");
        return (int)u;
    };
    auto ring_read = [&](int n, int f[7], bool spin) {
        const unsigned tg = ur0 + (n & 3) * 4, sl = ur0 + 16 + (n & 3) * 32;
        if (spin) {
            uint32_t v;
            do {
                asm volatile("ld.acquire.cta.shared.u32 %0, [%1];" : "=r"(v) : "r"(tg) : "memory");
            } while (v != (uint32_t)(n + 1));
        }
        asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(f[0]), "=r"(f[1]), "=r"(f[2]), "=r"(f[3]) : "r"(sl) : "memory");
        int dummy;
        asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(f[4]), "=r"(f[5]), "=r"(f[6]), "=r"(dummy) : "r"(sl + 16) : "memory");
        (void)dummy;
    };
    if (MB && tid == 0)
        for (int b = 0; b < NST; b++) {
            asm volatile("mbarrier.init.shared.b64 [%0], %1;" ::"r"(mb0 + b * 8), "r"(kThreads));
            asm volatile("mbarrier.init.shared.b64 [%0], %1;" ::"r"(mb0 + NST * 8 + b * 8), "r"(8));
        }
    int gsc = 0;  // global count of consumed stages (buffer gsc % NST, phase gsc / NST)
    bool first = true;
    for (;;) {
        if (first) {
            first = false;
            if constexpr (CS2) {
                if (tid == 0) {
                    for (int i = 0; i < 4; i++) asm volatile("st.shared.u32 [%0], %1;" ::"r"(ur2 + i * 4), "r"(0) : "memory");
                    nxt2 = i8c_publish(atomicAdd(ctr, 1), 0, ur2 + 16, ur2, total, full, parts, nkt, grp_rows, rb,
                                       (T + TB - 1) / TB, ctr);
                }
                __syncthreads();
                int f[7];
                cs_get(0, f);
                if (f[0] >= total) break;
                set_unit(f[1], f[2], f[3], f[4], f[5], f[6]);  // (the load stream enters unit 0 in prologue())
            } else if constexpr (CSL) {
                if (tid == 0) {
                    for (int i = 0; i < 4; i++) asm volatile("st.shared.u32 [%0], %1;" ::"r"(ur0 + i * 4), "r"(0) : "memory");
                    nxt = i8c_publish(atomicAdd(ctr, 1), 0, ur0 + 16, ur0, total, full, parts, nkt, grp_rows, rb,
                                      (T + TB - 1) / TB, ctr);
                }
                __syncthreads();
                int f[7];
                ring_read(0, f, false);
                if (f[0] >= total) break;
                set_unit(f[1], f[2], f[3], f[4], f[5], f[6]);
                ptr_q(f[5], f[2]);
                ptr_s(f[5], f[2]);
                ptr_x(f[6] * TB, f[2]);
            } else {
                if (tid == 0) {
                    s_unit = atomicAdd(ctr, 1);
                    if (SEP)
                        for (int i = 0; i < 4; i++) asm volatile("st.shared.u32 [%0], %1;" ::"r"(ur0 + i * 4), "r"(0) : "memory");
                }
                __syncthreads();
                if (s_unit >= total) break;
                decode(s_unit);
            }
            prologue();
        }
        float acc[MT][4][4];
#pragma unroll
        for (int m = 0; m < MT; m++)
#pragma unroll
            for (int n = 0; n < 4; n++)
#pragma unroll
                for (int e = 0; e < 4; e++) acc[m][n][e] = 0.f;
        uint32_t cc[C1 ? 1 : 2][4][4];  // ring of integer accumulator sets (tile parity)
        static_assert((SUB * MT) % 2 == 0, "tile ring parity");
        auto load_sw = [&](int kcs, float swf[4][2]) {  // tile n/2, rows (n & 1) * 8 + 2t, +1; HI: odd n / 16
            const unsigned a = s_lane + (kcs & 3) * 8 * 64;
#pragma unroll
            for (int n = 0; n < 4; n++) {
                const uint32_t v = *reinterpret_cast<const uint32_t*>(smb + (a - sm0) + n * 16);
                const float2 f = __bfloat1622float2(*reinterpret_cast<const bf162*>(&v));
                swf[n][0] = ((HI || HS) && (n & 1)) ? f.x * 0.0625f : f.x;
                swf[n][1] = ((HI || HS) && (n & 1)) ? f.y * 0.0625f : f.y;
            }
        };
        auto flush = [&](int m, const uint32_t cm[4][4], const float4 r, const float swf[4][2]) {
#pragma unroll
            for (int n = 0; n < 4; n++)
#pragma unroll
                for (int e = 0; e < 4; e++) {
                    const float f = __int_as_float(cm[n][e]);
                    const float sx = (e >> 1) ? r.z : r.x, nm = (e >> 1) ? r.w : r.y;
                    if (AB & 256) acc[m][n][e] = fmaf(f, sx, acc[m][n][e]);
                    else acc[m][n][e] = fmaf(fmaf(f, sx, nm), swf[n][e & 1], acc[m][n][e]);
                }
        };
        float4 rec[2];  // ring: scales of tiles j - 1, j (slot j & 1); slot 1 carries a stage's last tile
        float sw[4][2];  // weight scales; at the top of a stage still the previous stage's (deferred flush)
        for (int kc = 0; kc < nk; kc++) {
            const int stage = gsc % NST;
            if constexpr (MB) {
                const unsigned fb = mb0 + stage * 8, par = (gsc / NST) & 1;
                uint32_t ok;
                do {
                    asm volatile("{ .reg .pred p; mbarrier.test_wait.parity.shared.b64 p, [%1], %2; selp.u32 %0, 1, 0, p; }"
                                 : "=r"(ok) : "r"(fb), "r"(par) : "memory");
                    if (NSL && !ok) __nanosleep(NSL);
                } while (!ok);
            } else {
                cp_async_wait<NST - 2>();
                __syncthreads();
            }
            bool nv = false;  // CSL: a next unit exists (read at kc = nk - 2)
            if constexpr (EP) {
                if (tid == 0 && kc == max(0, nk - (SEP ? 4 : 3))) ep_next = atomicAdd(ctr, 1);
                if (SEP && tid == 0 && kc == max(0, nk - 2)) sep_pub(nun + 1, ep_next);
                if (kc == nk - 1) {
                    if constexpr (SEP) {
                        ep_u = sep_get(nun + 1);
                    } else {
                        if (tid == 0) s_unit = ep_next;
                        __syncthreads();  // every warp is past stage nk - 2: its buffers may be refilled
                        ep_u = s_unit;
                    }
                    if (ep_u < total) {  // sources of the next unit (this unit's are dead now)
                        const int c_tile = tile, c_kb = kb, c_nk = nk, c_part = part, c_row0 = row0, c_tok0 = tok0;
                        const bool c_v0 = v0, c_v1 = v1;
                        decode(ep_u);
                        ep_nk = nk;
                        tile = c_tile; kb = c_kb; nk = c_nk; part = c_part; row0 = c_row0; tok0 = c_tok0; v0 = c_v0; v1 = c_v1;
                        if constexpr (SEP) {  // stage 0 now (its buffer held stage nk - 3), stage 1 mid-stage
                            if (ep_nk > 0) { load_w(); load_x(); }
                        } else {
#pragma unroll
                            for (int s2 = 0; s2 < NST - 1; s2++)
                                if (s2 < ep_nk) { if (s2 < NWB - 1) load_w(); load_x(); }
                        }
                    }
                }
            }
            if constexpr (CSL) {
                if (tid == 0 && kc == 1)
                    nxt = i8c_publish(nxt, nun + 1, ur0 + 16 + ((nun + 1) & 3) * 32, ur0 + ((nun + 1) & 3) * 4, total, full,
                                      parts, nkt, grp_rows, rb, (T + TB - 1) / TB, ctr);
                if (kc >= nk - 2) {
                    int f[7];
                    ring_read(nun + 1, f, kc == nk - 2);
                    nv = f[0] < total;
                    if (nv && kc == nk - 2) ptr_s(f[5], f[2]);  // weight-scale stream moves to the next unit
                }
            }
            if (kc + NWB - 1 < nk || (CSL && nv)) load_w();
            cp_async_commit();
            // B fragments of the whole stage: bw[sub][n][ks] for row tile n (0/1: rows 0-7 / 8-15 of
            // 16-row tile 0, 2/3: tile 1); afterwards qw is free for the next stage's weights
            uint32_t bw[WSM ? 1 : SUB][4][KS][2];
            if constexpr (!WSM) {
                constexpr uint32_t M4 = 0x0F0F0F0Fu, M4H = 0xF0F0F0F0u;
#pragma unroll
                for (int sub = 0; sub < SUB; sub++)
#pragma unroll
                for (int i = 0; i < 2; i++)
#pragma unroll
                    for (int ks = 0; ks < KS; ks++) {
                        const int s = sub * KS + ks;
                        const uint4 q = qw[2 * i + (s >> 1)];
                        const uint32_t w0 = (s & 1) ? q.z : q.x, w1 = (s & 1) ? q.w : q.y;
                        bw[sub][2 * i][ks][0] = and_v<M4>(w0);
                        bw[sub][2 * i][ks][1] = and_v<M4>(w1);
                        if constexpr (HS) {
                            bw[sub][2 * i + 1][ks][0] = andxor_v<M4H, 0x80808080u>(w0);
                            bw[sub][2 * i + 1][ks][1] = andxor_v<M4H, 0x80808080u>(w1);
                        } else if constexpr (HI) {
                            bw[sub][2 * i + 1][ks][0] = and_v<M4H>(w0);
                            bw[sub][2 * i + 1][ks][1] = and_v<M4H>(w1);
                        } else {
                            bw[sub][2 * i + 1][ks][0] = (w0 >> 4) & M4;
                            bw[sub][2 * i + 1][ks][1] = (w1 >> 4) & M4;
                        }
                    }
            }
            if (kc + 1 < nk) {
                load_wreg();
            } else if (EP && ep_u < total) {  // weight registers: stage 0 of the next unit
                load_wreg();
            } else if (CSL && nv) {  // weight registers: first stage of the next unit
                int f[7];
                ring_read(nun + 1, f, false);
                ptr_q(f[5], f[2]);
                load_wreg();
            }
            const unsigned xa = a_lane + stage * XST, ra = r_lane + stage * RST;
            uint4 mq;
            if constexpr (HS) mq = *mq_s;
            uint32_t af[2][KS][4];  // PF: A fragments of tiles j (slot j & 1) and j + 1
            auto load_a = [&](int jj, uint32_t (&ad)[KS][4]) {
#pragma unroll
                for (int ks = 0; ks < KS; ks++)
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                                 : "=r"(ad[ks][0]), "=r"(ad[ks][1]), "=r"(ad[ks][2]), "=r"(ad[ks][3])
                                 : "r"(xa + (jj % MT) * 16 * kXRow + ((jj / MT) * KS + ks) * 32));
            };
            if constexpr (PF) load_a(0, af[0]);
#pragma unroll
            for (int sub = 0; sub < SUB; sub++) {
                uint32_t bws[4][KS][2];  // WSM: this sub-group's B fragments, unpacked from shared memory
                if constexpr (WSM) {
                    constexpr uint32_t M4 = 0x0F0F0F0Fu, M4H = 0xF0F0F0F0u;
                    const char* wb = smb + (wsm0 - sm0) + ((stage * 8 + warp) * 4 * 32 + lane) * 16;
#pragma unroll
                    for (int i = 0; i < 2; i++)
#pragma unroll
                        for (int ks = 0; ks < KS; ks++) {
                            const int s = sub * KS + ks;
                            const uint4 q = *reinterpret_cast<const uint4*>(wb + (2 * i + (s >> 1)) * 32 * 16);
                            const uint32_t w0 = (s & 1) ? q.z : q.x, w1 = (s & 1) ? q.w : q.y;
                            bws[2 * i][ks][0] = and_v<M4>(w0);
                            bws[2 * i][ks][1] = and_v<M4>(w1);
                            if constexpr (HS) {
                                bws[2 * i + 1][ks][0] = andxor_v<M4H, 0x80808080u>(w0);
                                bws[2 * i + 1][ks][1] = andxor_v<M4H, 0x80808080u>(w1);
                            } else {
                                bws[2 * i + 1][ks][0] = (w0 >> 4) & M4;
                                bws[2 * i + 1][ks][1] = (w1 >> 4) & M4;
                            }
                        }
                }
                auto bwv = [&](int n, int ks, int h) -> uint32_t { return WSM ? bws[n][ks][h] : bw[WSM ? 0 : sub][n][ks][h]; };
#pragma unroll
                for (int m = 0; m < MT; m++) {
                    const int j = sub * MT + m;
                    const unsigned rm = ra + (m * BLK + sub * KN * 8) * 16;
                    const float4* rp = reinterpret_cast<const float4*>(smb + (rm - sm0));
                    float4 cq, cqh;
                    if (AB & 32) {
                        cq = make_float4(__int_as_float(kMagicBits), __int_as_float(kMagicBits), __int_as_float(kMagicBits + j), __int_as_float(kMagicBits));
                        cqh = cq;
                        rec[j & 1] = make_float4(1e-3f * (j + 1), -kMagic * 1e-3f, 2e-3f, -kMagic * 2e-3f);
                    } else {
                        cq = rp[0];
                        cqh = HI ? rp[8] : cq;
                        rec[j & 1] = rp[(KN - 1) * 8];
                    }
                    uint32_t (*cj)[4] = cc[C1 ? 0 : (j & 1)];
                    if (PF && j + 1 < SUB * MT) load_a(j + 1, af[(j + 1) & 1]);
#pragma unroll
                    for (int ks = 0; ks < KS; ks++) {
                        uint32_t a[4];
                        const int s = sub * KS + ks;
                        if constexpr (PF) {
#pragma unroll
                            for (int q = 0; q < 4; q++) a[q] = af[j & 1][ks][q];
                        } else if (AB & 16) {
                            a[0] = 0x01010101u * (m + 1) + s; a[1] = 0x02020202u + kc; a[2] = 0x03030303u + sub; a[3] = 0x04040404u + j;
                        } else {
                            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                                         : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3]) : "r"(xa + m * 16 * kXRow + s * 32));
                        }
#pragma unroll
                        for (int n = 0; n < 4; n++) {
                            if (HS && (n & 1)) {
                                if (ks == 0) mma_s8s8_initq(cj[n], a, bwv(n, ks, 0), bwv(n, ks, 1), mq);
                                else mma_s8(cj[n], a, bwv(n, ks, 0), bwv(n, ks, 1));
                            } else {
                                if (ks == 0) mma_s8u8_initq(cj[n], a, bwv(n, ks, 0), bwv(n, ks, 1), (n & 1) ? cqh : cq);
                                else mma_s8u8(cj[n], a, bwv(n, ks, 0), bwv(n, ks, 1));
                            }
                        }
                    }
                    if constexpr (C1) {
                        if (j == 0) load_sw(gsc, sw);
                        flush(m, cc[0], rec[j & 1], sw);
                    } else if (j == 0) {
                        if (kc > 0) flush(MT - 1, cc[1], rec[1], sw);
                        load_sw(gsc, sw);  // after the previous stage's last flush: no second register set
                    } else {
                        flush((j - 1) % MT, cc[(j - 1) & 1], rec[(j - 1) & 1], sw);
                    }
                    if (j == MT - 1) {  // next x stage, issued away from the barrier
                        if (CS2 || kc + NST - 1 < nk) {
                            load_x();
                        } else if (SEP && kc == nk - 1 && ep_u < total && ep_nk > 1) {  // next unit's stage 1
                            load_w();
                            load_x();
                        } else if (CSL && nv) {
                            if (kc == nk - 2) {
                                int f[7];
                                ring_read(nun + 1, f, false);
                                ptr_x(f[6] * TB, f[2]);
                            }
                            load_x();
                        }
                        cp_async_commit();
                    }
                }
            }
            if constexpr (MB) {  // this warp is done with the stage's buffer
                __syncwarp();
                if (lane == 0) {
                    uint64_t st;
                    asm volatile("mbarrier.arrive.shared.b64 %0, [%1];" : "=l"(st) : "r"(mb0 + NST * 8 + stage * 8) : "memory");
                }
            }
            gsc++;
        }
        if (!C1 && nk > 0) flush(MT - 1, cc[C1 ? 0 : 1], rec[1], sw);
        const int e_part = part, e_tile = tile, e_row0 = row0, e_tok0 = tok0;
        const bool e_v0 = v0, e_v1 = v1;
        bool more;
        if constexpr (EP) {  // the next unit's first stages are in flight since the last stage's top
            more = ep_u < total;
            if (nk == 0) {  // (no last stage ran: fetch now)
                if constexpr (SEP) {
                    if (tid == 0) sep_pub(nun + 1, atomicAdd(ctr, 1));
                    ep_u = sep_get(nun + 1);
                } else {
                    if (tid == 0) s_unit = atomicAdd(ctr, 1);
                    __syncthreads();
                    ep_u = s_unit;
                }
                more = ep_u < total;
                if (more) { decode(ep_u); prologue(); }
                ep_u = total;  // unit variables already set
            }
            if constexpr (SEP) nun++;
        } else if constexpr (CS2) {  // the next unit's first stages are already in flight
            int f[7];
            cs_get(nun + 1, f);
            more = f[0] < total;
            if (more) set_unit(f[1], f[2], f[3], f[4], f[5], f[6]);
            nun++;
        } else if constexpr (CSL) {  // the next unit's first stages are already in flight
            int f[7];
            ring_read(nun + 1, f, true);
            more = f[0] < total;
            if (more) set_unit(f[1], f[2], f[3], f[4], f[5], f[6]);
            nun++;
        } else {
            cp_async_wait<0>();
            __syncthreads();
            if (tid == 0) s_unit = atomicAdd(ctr, 1);
            __syncthreads();
            more = s_unit < total;
            if (more) {
                decode(s_unit);
                prologue();
            }
        }

        if (AB & 512) {
            float sacc = 0.f;
#pragma unroll
            for (int m = 0; m < MT; m++)
#pragma unroll
                for (int n = 0; n < 4; n++)
#pragma unroll
                    for (int e = 0; e < 4; e++) sacc += acc[m][n][e];
            if (sacc == 12345.678f) c[tid] = sacc;
        } else if (e_part >= 0) {
            float* pw = wsp + ((size_t)(e_tile - full) * parts + e_part) * ROWS * TB;
#pragma unroll
            for (int m = 0; m < MT; m++)
#pragma unroll
                for (int n = 0; n < 4; n++) {
                    const int lt = tokw + m * 16 + g, lr = wr * 32 + n * 8 + 2 * t;
                    *reinterpret_cast<float2*>(pw + (size_t)lt * ROWS + lr) = make_float2(acc[m][n][0], acc[m][n][1]);
                    *reinterpret_cast<float2*>(pw + (size_t)(lt + 8) * ROWS + lr) = make_float2(acc[m][n][2], acc[m][n][3]);
                }
            if constexpr (SKF) {  // the block finishing the tile's last part sums all parts (order p = 0 .. parts - 1)
                __threadfence();
                __syncthreads();
                if (tid == 0) s_skf = atomicAdd(ctr + 1 + (e_tile - full), 1) == parts - 1;
                __syncthreads();
                if (s_skf) {
                    __threadfence();
                    const int rbase = e_row0 - wr * 32, tbase = e_tok0;
                    const float* pt = wsp + (size_t)(e_tile - full) * parts * ROWS * TB;
                    const size_t ps = (size_t)ROWS * TB;
                    if (glu) {
                        bf16* act = reinterpret_cast<bf16*>(c);
                        for (int i = tid; i < TB * (ROWS / 2); i += kThreads) {
                            const int lt = i / (ROWS / 2), o = i % (ROWS / 2), lr = (o / 8) * 16 + o % 8;
                            const int tok = tbase + lt;
                            if (tok >= T || rbase + lr >= N) continue;
                            float gv = 0.f, uv = 0.f;
                            for (int p = 0; p < parts; p++) {
                                gv += __ldcg(pt + p * ps + (size_t)lt * ROWS + lr);
                                uv += __ldcg(pt + p * ps + (size_t)lt * ROWS + lr + 8);
                            }
                            act[(size_t)tok * ldc + rbase / 2 + o] = __float2bfloat16(gv / (1.f + __expf(-gv)) * uv);
                        }
                    } else {
                        for (int i = tid; i < TB * ROWS; i += kThreads) {
                            const int lt = i / ROWS, o = i % ROWS, tok = tbase + lt;
                            if (tok >= T || rbase + o >= N) continue;
                            float v = 0.f;
                            for (int p = 0; p < parts; p++) v += __ldcg(pt + p * ps + (size_t)lt * ROWS + o);
                            if (o16) reinterpret_cast<bf16*>(c)[(size_t)tok * ldc + rbase + o] = __float2bfloat16(v);
                            else c[(size_t)tok * ldc + rbase + o] = v;
                        }
                    }
                }
            }
        } else if (glu && gq) {
            // GLU + int8 quantization of the activations for the next (down) GEMM, G = 64: the block's 128
            // activation columns are one 128-k group, warps 4j..4j+3 hold sub-group j; lane (g, t) holds
            // columns 2t, 2t + 1 of the 8-column halves i = 0, 1 of its warp's 16 columns, which are
            // exactly the 4 bytes of word t of the MMA k permutation (k 2t, 2t + 8, 2t + 1, 2t + 9)
            bf16* act = reinterpret_cast<bf16*>(c);
            const int K2 = N / 2, sj = wr >> 2;
            bf162 av[MT][2][2];
            float am[MT][2];
#pragma unroll
            for (int m = 0; m < MT; m++)
#pragma unroll
                for (int h = 0; h < 2; h++) {
                    float mx = 0.f;
#pragma unroll
                    for (int i = 0; i < 2; i++) {
                        const float g0 = acc[m][2 * i][2 * h], g1 = acc[m][2 * i][2 * h + 1];
                        const float u0 = acc[m][2 * i + 1][2 * h], u1 = acc[m][2 * i + 1][2 * h + 1];
                        av[m][h][i] = __floats2bfloat162_rn(g0 / (1.f + __expf(-g0)) * u0, g1 / (1.f + __expf(-g1)) * u1);
                        const float2 f = __bfloat1622float2(av[m][h][i]);
                        mx = fmaxf(mx, fmaxf(fabsf(f.x), fabsf(f.y)));
                    }
                    mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 1));
                    mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 2));
                    am[m][h] = mx;
                    if (t == 0) s_gred[0][m * 16 + g + h * 8][wr] = mx;
                }
            asm volatile("bar.sync %0, 128;" ::"r"(1 + sj) : "memory");
            int sm[MT][2];
#pragma unroll
            for (int m = 0; m < MT; m++)
#pragma unroll
                for (int h = 0; h < 2; h++) {
                    const int lt = m * 16 + g + h * 8, tk = e_tok0 + lt;
                    const float* r4 = s_gred[0][lt] + 4 * sj;
                    const float mx = fmaxf(fmaxf(r4[0], r4[1]), fmaxf(r4[2], r4[3]));
                    const float sc = mx > 0.f ? __uint_as_float(__float_as_uint(mx / 127.f) & ~3u) : 1.f;
                    const float inv = mx > 0.f ? __frcp_rn(sc) : 0.f;
                    const float2 f0 = __bfloat1622float2(av[m][h][0]), f1 = __bfloat1622float2(av[m][h][1]);
                    const int q0 = __float2int_rn(f0.x * inv), q1 = __float2int_rn(f1.x * inv);
                    const int q2 = __float2int_rn(f0.y * inv), q3 = __float2int_rn(f1.y * inv);
                    int sum = q0 + q1 + q2 + q3;
                    sum += __shfl_xor_sync(0xffffffffu, sum, 1);
                    sum += __shfl_xor_sync(0xffffffffu, sum, 2);
                    sm[m][h] = sum;
                    am[m][h] = sc;
                    if (tk < T) {
                        const int col = (e_row0 + 0 * 16) / 2;  // warp's first activation column
                        *reinterpret_cast<bf162*>(act + (size_t)tk * ldc + col + 2 * t) = av[m][h][0];
                        *reinterpret_cast<bf162*>(act + (size_t)tk * ldc + col + 8 + 2 * t) = av[m][h][1];
                        *reinterpret_cast<uint32_t*>(gq + (size_t)tk * K2 + col + 4 * t) =
                            (uint32_t)(q0 & 0xFF) | ((uint32_t)(q1 & 0xFF) << 8) | ((uint32_t)(q2 & 0xFF) << 16) |
                            ((uint32_t)(q3 & 0xFF) << 24);
                    }
                    if (t == 0) s_gred[1][lt][wr] = __int_as_float(sum);
                }
            asm volatile("bar.sync %0, 128;" ::"r"(1 + sj) : "memory");
            if ((wr & 3) == 0 && t == 0) {
#pragma unroll
                for (int m = 0; m < MT; m++)
#pragma unroll
                    for (int h = 0; h < 2; h++) {
                        const int lt = m * 16 + g + h * 8, tk = e_tok0 + lt;
                        const float* r4 = s_gred[1][lt] + 4 * sj;
                        const int sum = __float_as_int(r4[0]) + __float_as_int(r4[1]) + __float_as_int(r4[2]) + __float_as_int(r4[3]);
                        if (tk < T) i8v_rec_store<64, false>(gqs, tk, (e_row0 / 2) / 128, K2 / 128, sj, am[m][h], sum);
                    }
            }
            (void)sm;
        } else if (glu) {
            bf16* act = reinterpret_cast<bf16*>(c);
#pragma unroll
            for (int i = 0; i < 2; i++) {
                if (!(i == 0 ? e_v0 : e_v1)) continue;
                const int col = (e_row0 + i * 16) / 2 + 2 * t;
#pragma unroll
                for (int m = 0; m < MT; m++)
#pragma unroll
                    for (int h = 0; h < 2; h++) {
                        const int tk = e_tok0 + tokw + m * 16 + g + h * 8;
                        if (tk < T) {
                            const float g0 = acc[m][2 * i][2 * h], g1 = acc[m][2 * i][2 * h + 1];
                            const float u0 = acc[m][2 * i + 1][2 * h], u1 = acc[m][2 * i + 1][2 * h + 1];
                            *reinterpret_cast<bf162*>(act + (size_t)tk * ldc + col) = __floats2bfloat162_rn(
                                g0 / (1.f + __expf(-g0)) * u0, g1 / (1.f + __expf(-g1)) * u1);
                        }
                    }
            }
        } else {
#pragma unroll
            for (int n = 0; n < 4; n++) {
                if (!((n < 2) ? e_v0 : e_v1)) continue;
                const int r = e_row0 + n * 8 + 2 * t;
#pragma unroll
                for (int m = 0; m < MT; m++)
#pragma unroll
                    for (int h = 0; h < 2; h++) {
                        const int tk = e_tok0 + tokw + m * 16 + g + h * 8;
                        if (tk < T) {
                            if (o16)
                                *reinterpret_cast<bf162*>(reinterpret_cast<bf16*>(c) + (size_t)tk * ldc + r) =
                                    __floats2bfloat162_rn(acc[m][n][2 * h], acc[m][n][2 * h + 1]);
                            else
                                *reinterpret_cast<float2*>(c + (size_t)tk * ldc + r) =
                                    make_float2(acc[m][n][2 * h], acc[m][n][2 * h + 1]);
                        }
                    }
            }
        }
        if (!more) break;
        if (EP && ep_u < total) decode(ep_u, false);  // unit variables of the next unit (sources already advanced)
    }
}

// ---------------------------------------------------------------- W4A8, continuous stage stream across units
// gemm_i8v_kernel<HM = 2, OPT = 1> (signed high nibbles, mbarrier pipeline) without the per-unit
// block barriers: the x / record copies, the weight-scale copies and the weight-register loads are
// three streams over one global sequence of stages (all units of this block back to back), so
// the next unit's first stages are already in flight while the current unit's last stages and
// epilogue run, and warps never wait for each other at a unit boundary. Unit ids come from the
// stream-K counter as before; thread 0 publishes unit n in a ring of shared slots (release / acquire
// with a sequence tag) when any of its streams first needs it, and fetches unit n + 1 right after.
template <int WT, int TB, int G, int NST = STAGES>
constexpr size_t i8c_smem() {
    return (size_t)NST * TB * kXRow + (size_t)NST * (TB / 16) * v_blk<G, false>() * 16 + 4 * 8 * 64 + 16 + 16 * NST + 32 + 8 * 32;
}

template <int WT, int TB, int G, int MINB = 1, int NST = STAGES>
__global__ void __launch_bounds__(kThreads, MINB) gemm_i8c_kernel(const uint4* __restrict__ wq,
                                                           const bf16* __restrict__ ws, int N, int K,
                                                           const int8_t* __restrict__ x,
                                                           const float4* __restrict__ xsc, int T,
                                                           float* __restrict__ c, int ldc, bool glu,
                                                           int* __restrict__ ctr, float* __restrict__ wsp,
                                                           int full, int parts, int grp_rows) {
    extern __shared__ __align__(128) uint4 dyn_smem[];
    constexpr int NWB = 3;
    constexpr int SUB = BKI / G, KS = 4 / SUB;
    constexpr int KN = 2, BLK = v_blk<G, false>();
    constexpr int WR = 8 / WT, MT = TB / WT / 16, ROWS = block_rows<WT>();
    constexpr int XST = TB * kXRow, RST = (TB / 16) * BLK * 16;
    constexpr int NXI = TB * 8 / kThreads, NRI = ((TB / 16) * BLK + kThreads - 1) / kThreads;
    static_assert(MT >= 1 && TB % (16 * WT) == 0, "token tile");
    static_assert(G == 32 || G == 64 || G == 128, "group");
    static_assert(TB * 8 % kThreads == 0, "x chunks per thread");
    static_assert(NWB - 1 <= NST - 1, "weight prefetch distance");
    char* const smb = reinterpret_cast<char*>(dyn_smem);
    const unsigned sm0 = __cvta_generic_to_shared(smb);
    const unsigned srec0 = sm0 + NST * XST;
    const unsigned ssm0 = srec0 + NST * RST;
    const unsigned mq0 = ssm0 + 4 * 8 * 64;
    const unsigned mb0 = mq0 + 16;                 // mbarriers full[NST], empty[NST]
    const unsigned us0 = mb0 + 16 * NST;           // unit ring: tag[8], then 8 slots of 8 ints (u, Unit fields)
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int wr = warp % WR, wt = warp / WR;
    const int g = lane >> 2, t = lane & 3;
    const int nkt = K / BKI;
    const int groups = K >> 7;
    const int rb = (N + ROWS - 1) / ROWS;
    const int tbn = (T + TB - 1) / TB;
    const int tiles = rb * tbn;
    const int total = full + (tiles - full) * parts;
    const int tokw = wt * (TB / WT);
    const unsigned a_lane = sm0 + (tokw + (lane & 7) + ((lane >> 3) & 1) * 8) * kXRow + (lane >> 4) * 16;
    const unsigned r_lane = srec0 + (tokw / 16) * BLK * 16 + g * 16;
    const unsigned s_lane = ssm0 + wr * 64 + t * 4;
    const bool wthr = wt == 0 && lane < 4;  // issues the weight-scale copies of its row warp

    struct Unit { int u, tile, kb, nk, part, rbi, tbi, pad; };
    auto decode = [&](int u) {
        Unit d;
        d.u = u;
        d.pad = 0;
        int ke;
        if (u < full) {
            d.tile = u; d.kb = 0; ke = nkt; d.part = -1;
        } else {
            const int r = (u - full) / parts;
            d.part = (u - full) % parts;
            d.tile = full + r;
            d.kb = d.part * nkt / parts;
            ke = (d.part + 1) * nkt / parts;
        }
        d.nk = ke - d.kb;
        const int GRP = min(grp_rows, rb);
        const int grp = d.tile / (GRP * tbn);
        const int gsz = min(GRP, rb - grp * GRP), idx = d.tile - grp * GRP * tbn;
        d.tbi = idx / gsz;
        d.rbi = grp * GRP + idx % gsz;
        return d;
    };
    // unit ring: thread 0 decodes and publishes unit n (fields in slot n & 7, tag n + 1) when one of its
    // streams first needs it; every other reader spins on the tag (acquire); u >= total ends the work
    int pub_n = 0, nxt = 0;  // (thread 0) last published unit, id fetched for the next one
    auto get_unit = [&](int n) -> Unit {
        const unsigned sl = us0 + 32 + (n & 7) * 32, tg = us0 + (n & 7) * 4;
        if (tid == 0) {
            if (n > pub_n) {
                nxt = i8c_publish(nxt, n, sl, tg, total, full, parts, nkt, grp_rows, rb, tbn, ctr);
                pub_n = n;
            }
        } else {
            uint32_t v;
            do {
                asm volatile("ld.acquire.cta.shared.u32 %0, [%1];" : "=r"(v) : "r"(tg) : "memory");
            } while (v != (uint32_t)(n + 1));
        }
        Unit d;
        asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(d.u), "=r"(d.tile), "=r"(d.kb), "=r"(d.nk) : "r"(sl) : "memory");
        asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(d.part), "=r"(d.rbi), "=r"(d.tbi), "=r"(d.pad) : "r"(sl + 16) : "memory");
        return d;
    };

    // ---- X stream (x tile + records), all threads
    const int8_t* xg[NXI];
    bool xok[NXI];
    const float4* rg[NRI];
    bool rok[NRI];
    int gx = 0, endx = 0, nx = -1;
    bool xdone = false;
    auto x_switch = [&]() {
        while (gx == endx && !xdone) {
            const Unit d = get_unit(++nx);
            if (d.u >= total) { xdone = true; break; }
            const int tok0 = d.tbi * TB;
#pragma unroll
            for (int i = 0; i < NXI; i++) {
                const int idx = tid + i * kThreads, tok = tok0 + (idx >> 3);
                xg[i] = x + (size_t)min(tok, T - 1) * K + d.kb * BKI + (idx & 7) * 16;
                xok[i] = tok < T;
            }
#pragma unroll
            for (int i = 0; i < NRI; i++) {
                const int idx = tid + i * kThreads, tt = (tok0 >> 4) + idx / BLK;
                rok[i] = tt * 16 < T;
                rg[i] = xsc + ((size_t)(rok[i] ? tt : 0) * groups + d.kb) * BLK + idx % BLK;
            }
            endx += d.nk;
        }
    };
    auto load_x = [&]() {
        x_switch();
        if (xdone) return;
        const int stage = gx % NST;
        if (gx >= NST) {
            const unsigned eb = mb0 + NST * 8 + stage * 8, par = ((gx / NST) + 1) & 1;
            uint32_t ok;
            do {
                asm volatile("{ .reg .pred p; mbarrier.test_wait.parity.shared.b64 p, [%1], %2; selp.u32 %0, 1, 0, p; }"
                             : "=r"(ok) : "r"(eb), "r"(par) : "memory");
            } while (!ok);
        }
#pragma unroll
        for (int i = 0; i < NXI; i++) {
            const int idx = tid + i * kThreads;
            const unsigned d = sm0 + stage * XST + (idx >> 3) * kXRow + (idx & 7) * 16;
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(d), "l"(xg[i]), "r"(xok[i] ? 16 : 0));
            xg[i] += BKI;
        }
#pragma unroll
        for (int i = 0; i < NRI; i++) {
            const int idx = tid + i * kThreads;
            if ((TB / 16) * BLK % kThreads == 0 || idx < (TB / 16) * BLK) {
                const unsigned d = srec0 + stage * RST + idx * 16;
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(d), "l"(rg[i]), "r"(rok[i] ? 16 : 0));
            }
            rg[i] += BLK;
        }
        asm volatile("cp.async.mbarrier.arrive.noinc.shared.b64 [%0];" ::"r"(mb0 + stage * 8));
        gx++;
    };
    // ---- W stream (weight-scale rows), threads wthr; ring of 4 by global stage
    const bf16* sg = ws;
    int gw = 0, endw = 0, nw = -1;
    bool wdone = false;
    auto load_w = [&]() {
        if (!wthr) return;
        while (gw == endw && !wdone) {
            const Unit d = get_unit(++nw);
            if (d.u >= total) { wdone = true; break; }
            const int rt0 = (d.rbi * ROWS + wr * 32) / 16 + (lane >> 1);
            sg = ws + (size_t)(rt0 < N / 16 ? rt0 : 0) * groups * 16 + d.kb * 16 + (lane & 1) * 8;
            endw += d.nk;
        }
        if (wdone) return;
        const unsigned d = ssm0 + ((gw & 3) * 8 + wr) * 64 + lane * 16;
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16, 16;" ::"r"(d), "l"(sg));
        sg += 16;
        gw++;
    };
    // ---- Q stream (packed weight words in registers), all threads, one stage ahead
    const uint4 *wg0 = wq, *wg1 = wq;
    int gq = 0, endq = 0, nq = -1;
    bool qdone = false;
    uint4 qw[4];
    auto load_wreg = [&]() {
        while (gq == endq && !qdone) {
            const Unit d = get_unit(++nq);
            if (d.u >= total) { qdone = true; break; }
            const int rt0 = (d.rbi * ROWS + wr * 32) / 16;
            wg0 = wq + ((size_t)(rt0 < N / 16 ? rt0 : 0) * (K / 64) + d.kb * 2) * 32 + lane;
            wg1 = wq + ((size_t)(rt0 + 1 < N / 16 ? rt0 + 1 : 0) * (K / 64) + d.kb * 2) * 32 + lane;
            endq += d.nk;
        }
        if (qdone) return;
        qw[0] = ldg_v(wg0);
        qw[1] = ldg_v(wg0 + 32);
        qw[2] = ldg_v(wg1);
        qw[3] = ldg_v(wg1 + 32);
        wg0 += 64;
        wg1 += 64;
        gq++;
    };

    uint4* const mq_s = reinterpret_cast<uint4*>(smb + (mq0 - sm0));
    if (tid == 0) {
        *mq_s = make_uint4(kMagicBits, kMagicBits, kMagicBits, kMagicBits);
        for (int b = 0; b < NST; b++) {
            asm volatile("mbarrier.init.shared.b64 [%0], %1;" ::"r"(mb0 + b * 8), "r"(kThreads));
            asm volatile("mbarrier.init.shared.b64 [%0], %1;" ::"r"(mb0 + NST * 8 + b * 8), "r"(8));
        }
        int* ring = reinterpret_cast<int*>(smb + (us0 - sm0));
        for (int i = 0; i < 8; i++) ring[i] = 0;
        // unit 0 now, unit 1 fetched (published on first need)
        nxt = i8c_publish(atomicAdd(ctr, 1), 0, us0 + 32, us0, total, full, parts, nkt, grp_rows, rb, tbn, ctr);
    }
    __syncthreads();
    // prologue of the streams
#pragma unroll
    for (int s = 0; s < NWB - 1; s++) load_w();
#pragma unroll
    for (int s = 0; s < NST - 1; s++) load_x();
    load_wreg();

    auto load_sw = [&](int gs, float swf[4][2]) {
        const unsigned a = s_lane + (gs & 3) * 8 * 64;
#pragma unroll
        for (int n = 0; n < 4; n++) {
            const uint32_t v = *reinterpret_cast<const uint32_t*>(smb + (a - sm0) + n * 16);
            const float2 f = __bfloat1622float2(*reinterpret_cast<const bf162*>(&v));
            swf[n][0] = (n & 1) ? f.x * 0.0625f : f.x;
            swf[n][1] = (n & 1) ? f.y * 0.0625f : f.y;
        }
    };
    float acc[MT][4][4];
    uint32_t cc[2][4][4];
    static_assert((SUB * MT) % 2 == 0, "tile ring parity");
    auto flush = [&](int m, const uint32_t cm[4][4], const float4 r, const float swf[4][2]) {
#pragma unroll
        for (int n = 0; n < 4; n++)
#pragma unroll
            for (int e = 0; e < 4; e++) {
                const float f = __int_as_float(cm[n][e]);
                const float sx = (e >> 1) ? r.z : r.x, nm = (e >> 1) ? r.w : r.y;
                acc[m][n][e] = fmaf(fmaf(f, sx, nm), swf[n][e & 1], acc[m][n][e]);
            }
    };
    float4 rec[2];
    float sw[4][2];
    int gc = 0;  // consumer: global stage
    for (int nc = 0;; nc++) {
        const Unit d = get_unit(nc);
        if (d.u >= total) break;
#pragma unroll
        for (int m = 0; m < MT; m++)
#pragma unroll
            for (int n = 0; n < 4; n++)
#pragma unroll
                for (int e = 0; e < 4; e++) acc[m][n][e] = 0.f;
        for (int kc = 0; kc < d.nk; kc++, gc++) {
            const int stage = gc % NST;
            {
                const unsigned fb = mb0 + stage * 8, par = (gc / NST) & 1;
                uint32_t ok;
                do {
                    asm volatile("{ .reg .pred p; mbarrier.test_wait.parity.shared.b64 p, [%1], %2; selp.u32 %0, 1, 0, p; }"
                                 : "=r"(ok) : "r"(fb), "r"(par) : "memory");
                } while (!ok);
            }
            load_w();  // stage gc + NWB - 1 (any unit)
            uint32_t bw[SUB][4][KS][2];
            {
                constexpr uint32_t M4 = 0x0F0F0F0Fu, M4H = 0xF0F0F0F0u;
#pragma unroll
                for (int sub = 0; sub < SUB; sub++)
#pragma unroll
                    for (int i = 0; i < 2; i++)
#pragma unroll
                        for (int ks = 0; ks < KS; ks++) {
                            const int s = sub * KS + ks;
                            const uint4 q = qw[2 * i + (s >> 1)];
                            const uint32_t w0 = (s & 1) ? q.z : q.x, w1 = (s & 1) ? q.w : q.y;
                            bw[sub][2 * i][ks][0] = and_v<M4>(w0);
                            bw[sub][2 * i][ks][1] = and_v<M4>(w1);
                            bw[sub][2 * i + 1][ks][0] = andxor_v<M4H, 0x80808080u>(w0);
                            bw[sub][2 * i + 1][ks][1] = andxor_v<M4H, 0x80808080u>(w1);
                        }
            }
            load_wreg();  // stage gc + 1
            const unsigned xa = a_lane + stage * XST, ra = r_lane + stage * RST;
            const uint4 mq = *mq_s;
#pragma unroll
            for (int sub = 0; sub < SUB; sub++) {
#pragma unroll
                for (int m = 0; m < MT; m++) {
                    const int j = sub * MT + m;
                    const float4* rp = reinterpret_cast<const float4*>(smb + (ra - sm0) + (m * BLK + sub * KN * 8) * 16);
                    const float4 cq = rp[0];
                    rec[j & 1] = rp[8];
                    uint32_t (*cj)[4] = cc[j & 1];
#pragma unroll
                    for (int ks = 0; ks < KS; ks++) {
                        uint32_t a[4];
                        const int s = sub * KS + ks;
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                                     : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3]) : "r"(xa + m * 16 * kXRow + s * 32));
#pragma unroll
                        for (int n = 0; n < 4; n++) {
                            if (n & 1) {
                                if (ks == 0) mma_s8s8_initq(cj[n], a, bw[sub][n][ks][0], bw[sub][n][ks][1], mq);
                                else mma_s8(cj[n], a, bw[sub][n][ks][0], bw[sub][n][ks][1]);
                            } else {
                                if (ks == 0) mma_s8u8_initq(cj[n], a, bw[sub][n][ks][0], bw[sub][n][ks][1], cq);
                                else mma_s8u8(cj[n], a, bw[sub][n][ks][0], bw[sub][n][ks][1]);
                            }
                        }
                    }
                    if (j == 0) {
                        if (kc > 0) flush(MT - 1, cc[1], rec[1], sw);
                        load_sw(gc, sw);
                    } else {
                        flush((j - 1) % MT, cc[(j - 1) & 1], rec[(j - 1) & 1], sw);
                    }
                    if (j == MT - 1) load_x();  // stage gc + NST - 1 (any unit)
                }
            }
            __syncwarp();
            if (lane == 0) {
                uint64_t st;
                asm volatile("mbarrier.arrive.shared.b64 %0, [%1];" : "=l"(st) : "r"(mb0 + NST * 8 + stage * 8) : "memory");
            }
        }
        if (d.nk > 0) flush(MT - 1, cc[1], rec[1], sw);
        // epilogue of this unit (the next unit's stages are already streaming in)
        const int row0 = d.rbi * ROWS + wr * 32, tok0 = d.tbi * TB, rt0 = row0 / 16;
        const bool v0 = rt0 < N / 16, v1 = rt0 + 1 < N / 16;
        if (d.part >= 0) {
            float* pw = wsp + ((size_t)(d.tile - full) * parts + d.part) * ROWS * TB;
#pragma unroll
            for (int m = 0; m < MT; m++)
#pragma unroll
                for (int n = 0; n < 4; n++) {
                    const int lt = tokw + m * 16 + g, lr = wr * 32 + n * 8 + 2 * t;
                    *reinterpret_cast<float2*>(pw + (size_t)lt * ROWS + lr) = make_float2(acc[m][n][0], acc[m][n][1]);
                    *reinterpret_cast<float2*>(pw + (size_t)(lt + 8) * ROWS + lr) = make_float2(acc[m][n][2], acc[m][n][3]);
                }
        } else if (glu) {
            bf16* act = reinterpret_cast<bf16*>(c);
#pragma unroll
            for (int i = 0; i < 2; i++) {
                if (!(i == 0 ? v0 : v1)) continue;
                const int col = (row0 + i * 16) / 2 + 2 * t;
#pragma unroll
                for (int m = 0; m < MT; m++)
#pragma unroll
                    for (int h = 0; h < 2; h++) {
                        const int tk = tok0 + tokw + m * 16 + g + h * 8;
                        if (tk < T) {
                            const float g0 = acc[m][2 * i][2 * h], g1 = acc[m][2 * i][2 * h + 1];
                            const float u0 = acc[m][2 * i + 1][2 * h], u1 = acc[m][2 * i + 1][2 * h + 1];
                            *reinterpret_cast<bf162*>(act + (size_t)tk * ldc + col) = __floats2bfloat162_rn(
                                g0 / (1.f + __expf(-g0)) * u0, g1 / (1.f + __expf(-g1)) * u1);
                        }
                    }
            }
        } else {
#pragma unroll
            for (int n = 0; n < 4; n++) {
                if (!((n < 2) ? v0 : v1)) continue;
                const int r = row0 + n * 8 + 2 * t;
#pragma unroll
                for (int m = 0; m < MT; m++)
#pragma unroll
                    for (int h = 0; h < 2; h++) {
                        const int tk = tok0 + tokw + m * 16 + g + h * 8;
                        if (tk < T)
                            *reinterpret_cast<float2*>(c + (size_t)tk * ldc + r) =
                                make_float2(acc[m][n][2 * h], acc[m][n][2 * h + 1]);
                    }
            }
        }
    }
    asm volatile("cp.async.wait_all;" ::: "memory");  // (no copies are issued past the last unit)
}

// ---------------------------------------------------------------- W4A8, 16 warps of 16 rows (4 warps per scheduler)
// gemm_i8v_kernel's inner loop with half the warp tile (16 weight rows x 64 tokens, 128 registers):
// the block is still 256 rows x 64 tokens (same L2 traffic, stream-K decomposition, records and
// epilogue layouts), but 512 threads give each scheduler 4 warps to hide the ldmatrix / record /
// MMA latencies that 2 warps of the 32-row tile leave exposed. Per MMA: unpack 0.5, ldmatrix 0.5,
// records 0.5 LDS.128 (vs 0.5 / 0.25 / 0.375).
constexpr int kWThreads = 512;
template <int TB, int G, int HM, int NST = STAGES>
constexpr size_t i8w_smem() {
    return (size_t)NST * TB * kXRow + (size_t)NST * (TB / 16) * v_blk<G, false>() * 16 + 4 * 16 * 32 + 16 + 16 * NST;
}

template <int TB, int G, int HM, int OPT = 0, int NST = STAGES>
__global__ void __launch_bounds__(kWThreads, 1) gemm_i8w_kernel(const uint4* __restrict__ wq,
                                                           const bf16* __restrict__ ws, int N, int K,
                                                           const int8_t* __restrict__ x,
                                                           const float4* __restrict__ xsc, int T,
                                                           float* __restrict__ c, int ldc, bool glu,
                                                           int* __restrict__ ctr, float* __restrict__ wsp,
                                                           int full, int parts, int grp_rows) {
    extern __shared__ __align__(128) uint4 dyn_smem[];
    __shared__ int s_unit;
    constexpr bool MB = OPT & 1, PF = OPT & 2, PSU = OPT & 4;
    constexpr bool HS = HM == 2;
    static_assert(HM == 0 || HM == 2, "high-nibble mode");
    constexpr int NWB = 3;
    constexpr int SUB = BKI / G, KS = 4 / SUB;
    constexpr int KN = 2, BLK = v_blk<G, false>();
    constexpr int MT = TB / 16, ROWS = 256;
    constexpr int XST = TB * kXRow, RST = (TB / 16) * BLK * 16;
    constexpr int NXI = (TB * 8 + kWThreads - 1) / kWThreads, NRI = ((TB / 16) * BLK + kWThreads - 1) / kWThreads;
    static_assert(G == 32 || G == 64 || G == 128, "group");
    static_assert(NWB - 1 <= NST - 1, "weight prefetch distance");
    char* const smb = reinterpret_cast<char*>(dyn_smem);
    const unsigned sm0 = __cvta_generic_to_shared(smb);
    const unsigned srec0 = sm0 + NST * XST;
    const unsigned ssm0 = srec0 + NST * RST;  // weight scales [4][16 warps][32 B]
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int g = lane >> 2, t = lane & 3;
    const int nkt = K / BKI;
    const int groups = K >> 7;
    const int rb = (N + ROWS - 1) / ROWS;
    const int tiles = rb * ((T + TB - 1) / TB);
    const int total = full + (tiles - full) * parts;
    const unsigned a_lane = sm0 + ((lane & 7) + ((lane >> 3) & 1) * 8) * kXRow + (lane >> 4) * 16;
    const unsigned r_lane = srec0 + g * 16;
    const unsigned s_lane = ssm0 + warp * 32 + t * 4;
    const unsigned mb0 = ssm0 + 4 * 16 * 32 + 16;  // mbarriers full[NST], empty[NST]

    int tile, kb, nk, part, row0, tok0;
    bool v0;
    const int8_t* xg[NXI];
    bool xok[NXI];
    const float4* rg[NRI];
    bool rok[NRI];
    const uint4* wg;   // packed weight words of the warp's 16-row tile (stage = 2 chunks = 64 uint4)
    const bf16* sg;    // weight-scale row half of this lane (lanes 0, 1)
    int kx = 0, kw;
    auto decode = [&](int u) {
        int ke;
        if (u < full) {
            tile = u; kb = 0; ke = nkt; part = -1;
        } else {
            const int r = (u - full) / parts;
            part = (u - full) % parts;
            tile = full + r;
            kb = part * nkt / parts;
            ke = (part + 1) * nkt / parts;
        }
        nk = ke - kb;
        int rbi, tbi;
        {
            const int GRP = min(grp_rows, rb);
            const int tbn = (T + TB - 1) / TB, grp = tile / (GRP * tbn);
            const int gsz = min(GRP, rb - grp * GRP), idx = tile - grp * GRP * tbn;
            tbi = idx / gsz;
            rbi = grp * GRP + idx % gsz;
        }
        row0 = rbi * ROWS + warp * 16;
        tok0 = tbi * TB;
        const int rt0 = row0 / 16;
        v0 = rt0 < N / 16;
        wg = wq + ((size_t)(v0 ? rt0 : 0) * (K / 64) + kb * 2) * 32 + lane;
        sg = ws + (size_t)(v0 ? rt0 : 0) * groups * 16 + kb * 16 + (lane & 1) * 8;
#pragma unroll
        for (int i = 0; i < NXI; i++) {
            const int idx = tid + i * kWThreads, tok = tok0 + (idx >> 3);
            xg[i] = x + (size_t)min(tok, T - 1) * K + kb * BKI + (idx & 7) * 16;
            xok[i] = tok < T;
        }
#pragma unroll
        for (int i = 0; i < NRI; i++) {
            const int idx = tid + i * kWThreads, tt = (tok0 >> 4) + idx / BLK;
            rok[i] = tt * 16 < T;
            rg[i] = xsc + ((size_t)(rok[i] ? tt : 0) * groups + kb) * BLK + idx % BLK;
        }
        kw = 0;
    };
    auto load_x = [&]() {
        const int stage = kx % NST;
        if (MB && kx >= NST) {  // the buffer's previous stage must be consumed by all warps
            const unsigned eb = mb0 + NST * 8 + stage * 8, par = ((kx / NST) + 1) & 1;
            uint32_t ok;
            do {
                asm volatile("{ .reg .pred p; mbarrier.test_wait.parity.shared.b64 p, [%1], %2; selp.u32 %0, 1, 0, p; }"
                             : "=r"(ok) : "r"(eb), "r"(par) : "memory");
            } while (!ok);
        }
#pragma unroll
        for (int i = 0; i < NXI; i++) {
            const int idx = tid + i * kWThreads;
            if ((TB * 8) % kWThreads == 0 || idx < TB * 8) {
                const unsigned d = sm0 + stage * XST + (idx >> 3) * kXRow + (idx & 7) * 16;
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(d), "l"(xg[i]), "r"(xok[i] ? 16 : 0));
            }
            xg[i] += BKI;
        }
#pragma unroll
        for (int i = 0; i < NRI; i++) {
            const int idx = tid + i * kWThreads;
            if ((TB / 16) * BLK % kWThreads == 0 || idx < (TB / 16) * BLK) {
                const unsigned d = srec0 + stage * RST + idx * 16;
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(d), "l"(rg[i]), "r"(rok[i] ? 16 : 0));
            }
            rg[i] += BLK;
        }
        if constexpr (MB) asm volatile("cp.async.mbarrier.arrive.noinc.shared.b64 [%0];" ::"r"(mb0 + stage * 8));
        kx++;
    };
    auto load_w = [&]() {  // weight-scale row (2 x 16 B per warp) of stage kw
        if (lane < 2) {
            const unsigned d = ssm0 + ((kw & 3) * 16 + warp) * 32 + lane * 16;
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16, 16;" ::"r"(d), "l"(sg));
        }
        sg += 16;
        kw++;
    };
    uint4 qw[2];
    auto load_wreg = [&]() {
        qw[0] = ldg_v(wg);
        qw[1] = ldg_v(wg + 32);
        wg += 64;
    };
    auto prologue = [&]() {
#pragma unroll
        for (int s = 0; s < NST - 1; s++) {
            if (s < nk && s < NWB - 1) load_w();
            if (s < nk) load_x();
            cp_async_commit();
        }
        load_wreg();
    };
    uint4* const mq_s = reinterpret_cast<uint4*>(smb + NST * XST + NST * RST + 4 * 16 * 32);
    if (HS && tid == 0) *mq_s = make_uint4(kMagicBits, kMagicBits, kMagicBits, kMagicBits);
    if (MB && tid == 0)
        for (int b = 0; b < NST; b++) {
            asm volatile("mbarrier.init.shared.b64 [%0], %1;" ::"r"(mb0 + b * 8), "r"(kWThreads));
            asm volatile("mbarrier.init.shared.b64 [%0], %1;" ::"r"(mb0 + NST * 8 + b * 8), "r"(16));
        }
    int gsc = 0;
    bool first = true;
    for (;;) {
        if (first) {
            first = false;
            if (tid == 0) s_unit = atomicAdd(ctr, 1);
            __syncthreads();
            if (s_unit >= total) break;
            decode(s_unit);
            prologue();
        }
        float acc[MT][2][4];
#pragma unroll
        for (int m = 0; m < MT; m++)
#pragma unroll
            for (int n = 0; n < 2; n++)
#pragma unroll
                for (int e = 0; e < 4; e++) acc[m][n][e] = 0.f;
        uint32_t cc[2][2][4];
        static_assert((SUB * MT) % 2 == 0, "tile ring parity");
        auto load_sw = [&](int kcs, float swf[2][2]) {
            const unsigned a = s_lane + (kcs & 3) * 16 * 32;
#pragma unroll
            for (int n = 0; n < 2; n++) {
                const uint32_t v = *reinterpret_cast<const uint32_t*>(smb + (a - sm0) + n * 16);
                const float2 f = __bfloat1622float2(*reinterpret_cast<const bf162*>(&v));
                swf[n][0] = (HS && n) ? f.x * 0.0625f : f.x;
                swf[n][1] = (HS && n) ? f.y * 0.0625f : f.y;
            }
        };
        auto flush = [&](int m, const uint32_t cm[2][4], const float4 r, const float swf[2][2]) {
#pragma unroll
            for (int n = 0; n < 2; n++)
#pragma unroll
                for (int e = 0; e < 4; e++) {
                    const float f = __int_as_float(cm[n][e]);
                    const float sx = (e >> 1) ? r.z : r.x, nm = (e >> 1) ? r.w : r.y;
                    acc[m][n][e] = fmaf(fmaf(f, sx, nm), swf[n][e & 1], acc[m][n][e]);
                }
        };
        float4 rec[2];
        float sw[2][2];
        for (int kc = 0; kc < nk; kc++) {
            const int stage = gsc % NST;
            if constexpr (MB) {
                const unsigned fb = mb0 + stage * 8, par = (gsc / NST) & 1;
                uint32_t ok;
                do {
                    asm volatile("{ .reg .pred p; mbarrier.test_wait.parity.shared.b64 p, [%1], %2; selp.u32 %0, 1, 0, p; }"
                                 : "=r"(ok) : "r"(fb), "r"(par) : "memory");
                } while (!ok);
            } else {
                cp_async_wait<NST - 2>();
                __syncthreads();
            }
            if (kc + NWB - 1 < nk) load_w();
            cp_async_commit();
            uint32_t bw[SUB][2][KS][2];
            auto unpack = [&](int sub) {
                constexpr uint32_t M4 = 0x0F0F0F0Fu, M4H = 0xF0F0F0F0u;
#pragma unroll
                    for (int ks = 0; ks < KS; ks++) {
                        const int s = sub * KS + ks;
                        const uint4 q = qw[s >> 1];
                        const uint32_t w0 = (s & 1) ? q.z : q.x, w1 = (s & 1) ? q.w : q.y;
                        bw[sub][0][ks][0] = and_v<M4>(w0);
                        bw[sub][0][ks][1] = and_v<M4>(w1);
                        if constexpr (HS) {
                            bw[sub][1][ks][0] = andxor_v<M4H, 0x80808080u>(w0);
                            bw[sub][1][ks][1] = andxor_v<M4H, 0x80808080u>(w1);
                        } else {
                            bw[sub][1][ks][0] = (w0 >> 4) & M4;
                            bw[sub][1][ks][1] = (w1 >> 4) & M4;
                        }
                    }
            };
            if constexpr (!PSU) {
#pragma unroll
                for (int sub = 0; sub < SUB; sub++) unpack(sub);
                if (kc + 1 < nk) load_wreg();
            }
            const unsigned xa = a_lane + stage * XST, ra = r_lane + stage * RST;
            uint4 mq;
            if constexpr (HS) mq = *mq_s;
            uint32_t af[2][KS][4];
            auto load_a = [&](int jj, uint32_t (&ad)[KS][4]) {
#pragma unroll
                for (int ks = 0; ks < KS; ks++)
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                                 : "=r"(ad[ks][0]), "=r"(ad[ks][1]), "=r"(ad[ks][2]), "=r"(ad[ks][3])
                                 : "r"(xa + (jj % MT) * 16 * kXRow + ((jj / MT) * KS + ks) * 32));
            };
            if constexpr (PF) load_a(0, af[0]);
#pragma unroll
            for (int sub = 0; sub < SUB; sub++) {
                if constexpr (PSU) {
                    unpack(sub);
                    if (kc + 1 < nk) {
#pragma unroll
                        for (int cw = 0; cw < 2; cw++)
                            if ((2 * cw + 1) / KS == sub) qw[cw] = ldg_v(wg + cw * 32);
                    }
                }
#pragma unroll
                for (int m = 0; m < MT; m++) {
                    const int j = sub * MT + m;
                    const float4* rp = reinterpret_cast<const float4*>(smb + (ra - sm0) + (m * BLK + sub * KN * 8) * 16);
                    const float4 cq = rp[0];
                    rec[j & 1] = rp[8];
                    uint32_t (*cj)[4] = cc[j & 1];
                    if (PF && j + 1 < SUB * MT) load_a(j + 1, af[(j + 1) & 1]);
                    if (!PF) load_a(j, af[j & 1]);
#pragma unroll
                    for (int ks = 0; ks < KS; ks++) {
                        const uint32_t* a = af[j & 1][ks];
                        if (ks == 0) mma_s8u8_initq(cj[0], a, bw[sub][0][ks][0], bw[sub][0][ks][1], cq);
                        else mma_s8u8(cj[0], a, bw[sub][0][ks][0], bw[sub][0][ks][1]);
                        if constexpr (HS) {
                            if (ks == 0) mma_s8s8_initq(cj[1], a, bw[sub][1][ks][0], bw[sub][1][ks][1], mq);
                            else mma_s8(cj[1], a, bw[sub][1][ks][0], bw[sub][1][ks][1]);
                        } else {
                            if (ks == 0) mma_s8u8_initq(cj[1], a, bw[sub][1][ks][0], bw[sub][1][ks][1], cq);
                            else mma_s8u8(cj[1], a, bw[sub][1][ks][0], bw[sub][1][ks][1]);
                        }
                    }
                    if (j == 0) {
                        if (kc > 0) flush(MT - 1, cc[1], rec[1], sw);
                        load_sw(kc, sw);
                    } else {
                        flush((j - 1) % MT, cc[(j - 1) & 1], rec[(j - 1) & 1], sw);
                    }
                    if (j == MT - 1) {
                        if (kc + NST - 1 < nk) load_x();
                        cp_async_commit();
                    }
                }
            }
            if constexpr (PSU) wg += 64;
            if constexpr (MB) {
                __syncwarp();
                if (lane == 0) {
                    uint64_t st;
                    asm volatile("mbarrier.arrive.shared.b64 %0, [%1];" : "=l"(st) : "r"(mb0 + NST * 8 + stage * 8) : "memory");
                }
            }
            gsc++;
        }
        if (nk > 0) flush(MT - 1, cc[1], rec[1], sw);
        cp_async_wait<0>();
        __syncthreads();
        const int e_part = part, e_tile = tile, e_row0 = row0, e_tok0 = tok0;
        const bool e_v0 = v0;
        if (tid == 0) s_unit = atomicAdd(ctr, 1);
        __syncthreads();
        const bool more = s_unit < total;
        if (more) {
            decode(s_unit);
            prologue();
        }

        if (e_part >= 0) {
            float* pw = wsp + ((size_t)(e_tile - full) * parts + e_part) * ROWS * TB;
#pragma unroll
            for (int m = 0; m < MT; m++)
#pragma unroll
                for (int n = 0; n < 2; n++) {
                    const int lt = m * 16 + g, lr = warp * 16 + n * 8 + 2 * t;
                    *reinterpret_cast<float2*>(pw + (size_t)lt * ROWS + lr) = make_float2(acc[m][n][0], acc[m][n][1]);
                    *reinterpret_cast<float2*>(pw + (size_t)(lt + 8) * ROWS + lr) = make_float2(acc[m][n][2], acc[m][n][3]);
                }
        } else if (glu) {
            bf16* act = reinterpret_cast<bf16*>(c);
            if (e_v0) {
                const int col = e_row0 / 2 + 2 * t;
#pragma unroll
                for (int m = 0; m < MT; m++)
#pragma unroll
                    for (int h = 0; h < 2; h++) {
                        const int tk = e_tok0 + m * 16 + g + h * 8;
                        if (tk < T) {
                            const float g0 = acc[m][0][2 * h], g1 = acc[m][0][2 * h + 1];
                            const float u0 = acc[m][1][2 * h], u1 = acc[m][1][2 * h + 1];
                            *reinterpret_cast<bf162*>(act + (size_t)tk * ldc + col) = __floats2bfloat162_rn(
                                g0 / (1.f + __expf(-g0)) * u0, g1 / (1.f + __expf(-g1)) * u1);
                        }
                    }
            }
        } else if (e_v0) {
#pragma unroll
            for (int n = 0; n < 2; n++) {
                const int r = e_row0 + n * 8 + 2 * t;
#pragma unroll
                for (int m = 0; m < MT; m++)
#pragma unroll
                    for (int h = 0; h < 2; h++) {
                        const int tk = e_tok0 + m * 16 + g + h * 8;
                        if (tk < T)
                            *reinterpret_cast<float2*>(c + (size_t)tk * ldc + r) =
                                make_float2(acc[m][n][2 * h], acc[m][n][2 * h + 1]);
                    }
            }
        }
        if (!more) break;
    }
}

// ---------------------------------------------------------------- warp-specialized variant
// gemm_i8t_kernel with a ninth (producer) warp that issues every cp.async (x tile, scale
// records, weight scale rows) into the NST-deep stage ring; the eight MMA warps only ldmatrix /
// LDS / ld.global.nc (packed weights) and execute neither cp.async.wait nor __syncthreads in
// the K loop. Per stage buffer s: named barrier FULL(s) (producer arrives after its
// cp.async.wait, consumers sync) and EMPTY(s) (consumers arrive after the stage, producer syncs
// before refilling). Ids: 0 = __syncthreads, 1..NST FULL, NST + 1..2 NST EMPTY.
constexpr int kWsThreads = kThreads + 32;
__device__ __forceinline__ void bar_sync(int id, int count) {
    asm volatile("bar.sync %0, %1;" ::"r"(id), "r"(count) : "memory");
}
__device__ __forceinline__ void bar_arrive(int id, int count) {
    asm volatile("bar.arrive %0, %1;" ::"r"(id), "r"(count) : "memory");
}
template <int WT, int TB, int G, int NST = STAGES>
constexpr size_t i8ws_smem() {
    return (size_t)NST * TB * 128 + (size_t)NST * ((128 / G + 1) / 2) * TB * 16 + (NST + 1) * 8 * 64;
}

template <int WT, int TB, int G, int NST = STAGES>
__global__ void __launch_bounds__(kWsThreads, 1) gemm_i8ws_kernel(const uint4* __restrict__ wq,
                                                                 const bf16* __restrict__ ws, int N, int K,
                                                                 const int8_t* __restrict__ x,
                                                                 const float4* __restrict__ xsc, int T,
                                                                 float* __restrict__ c, int ldc, bool glu,
                                                                 int* __restrict__ ctr, float* __restrict__ wsp,
                                                                 int full, int parts, int grp_rows) {
    extern __shared__ __align__(128) uint4 dyn_smem[];
    __shared__ int s_unit;
    constexpr int SUB = BKI / G, KS = 4 / SUB;  // sub-groups per stage, k32 steps per sub-group
    constexpr int NREC = (SUB + 1) / 2;         // float4 records {sx0, C0, sx1, C1} per token and stage
    constexpr int WR = 8 / WT, MT = TB / WT / 16, ROWS = block_rows<WT>();
    static_assert(MT >= 1 && TB % (16 * WT) == 0, "token tile");
    static_assert(G == 32 || G == 64 || G == 128, "group");
    static_assert((SUB * MT) % 2 == 0, "tile ring parity");
    static_assert(2 * NST < 16, "named barrier ids");
    uint4* xs = dyn_smem;                                                       // [NST][TB * 8]
    float4* srec = reinterpret_cast<float4*>(dyn_smem + NST * TB * 8);         // [NST][TB][NREC]
    uint4* ssm = dyn_smem + NST * TB * 8 + NST * NREC * TB;                    // [NST + 1][row warp][4 x 16 B]
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const bool producer = warp == 8;
    const int wr = producer ? 0 : warp % WR, wt = producer ? 0 : warp / WR;
    const int g = lane >> 2, t = lane & 3;
    const int nkt = K / BKI;
    const int groups = K >> 7;
    const int rb = (N + ROWS - 1) / ROWS;
    const int tiles = rb * ((T + TB - 1) / TB);
    const int total = full + (tiles - full) * parts;
    const int tokw = wt * (TB / WT);  // first token of the warp within the block tile

    for (;;) {
        if (tid == 0) s_unit = atomicAdd(ctr, 1);
        __syncthreads();
        const int u = s_unit;
        if (u >= total) break;
        int tile, kb, ke, part;
        if (u < full) {
            tile = u; kb = 0; ke = nkt; part = -1;
        } else {
            const int r = (u - full) / parts;
            part = (u - full) % parts;
            tile = full + r;
            kb = part * nkt / parts;
            ke = (part + 1) * nkt / parts;
        }
        const int nk = ke - kb;
        int rbi, tbi;
        {
            const int GRP = min(grp_rows, rb);
            const int tbn = (T + TB - 1) / TB, grp = tile / (GRP * tbn);
            const int gsz = min(GRP, rb - grp * GRP), idx = tile - grp * GRP * tbn;
            tbi = idx / gsz;
            rbi = grp * GRP + idx % gsz;
        }
        const int row0 = rbi * ROWS + wr * 32;
        const int tok0 = tbi * TB;

        if (producer) {
            auto load_stage = [&](int stage, int kc) {
                uint4* xb = xs + stage * TB * 8;
                for (int idx = lane; idx < TB * 8; idx += 32) {
                    const int r = idx >> 3, ch = idx & 7;
                    const int tok = tok0 + r;
                    const int8_t* src = x + (size_t)min(tok, T - 1) * K + (kb + kc) * BKI + ch * 16;
                    cp_async16(&xb[swz(r, ch)], src, tok < T);
                }
                for (int idx = lane; idx < TB * NREC; idx += 32) {  // records: [token][NREC]
                    const int tl = idx / NREC, r = idx % NREC, tok = tok0 + tl;
                    const float4* src = xsc + ((size_t)min(tok, T - 1) * groups + kb + kc) * NREC + r;
                    cp_async16(srec + (stage * TB + tl) * NREC + r, src, tok < T);
                }
                for (int idx = lane; idx < WR * 4; idx += 32) {  // weight scale rows: 2 x 16 scales per row warp
                    const int wrr = idx >> 2, piece = idx & 3;
                    const int rt = (rbi * ROWS + wrr * 32) / 16 + (piece >> 1);
                    const bf16* sp = ws + (size_t)(rt < N / 16 ? rt : 0) * groups * 16 + (size_t)(kb + kc) * 16 + (piece & 1) * 8;
                    cp_async16(ssm + ((kc % (NST + 1)) * 8 + wrr) * 4 + piece, sp, true);
                }
            };
            for (int kc = 0; kc < nk; kc++) {
                const int s = kc % NST;
                if (kc >= NST) bar_sync(1 + NST + s, kWsThreads);  // EMPTY(s): stage kc - NST consumed
                load_stage(s, kc);
                cp_async_commit();
                if (kc >= 1) {
                    cp_async_wait<1>();  // stage kc - 1 landed
                    bar_arrive(1 + (kc - 1) % NST, kWsThreads);  // FULL(kc - 1)
                }
            }
            if (nk > 0) {
                cp_async_wait<0>();
                bar_arrive(1 + (nk - 1) % NST, kWsThreads);
            }
            // consume the EMPTY arrivals of the last min(nk, NST) stages (every generation must
            // complete, or the next unit's barriers start with a stale count: hang)
            for (int j = max(0, nk - NST); j < nk; j++) bar_sync(1 + NST + j % NST, kWsThreads);
        } else {
            float acc[MT][4][4];
#pragma unroll
            for (int m = 0; m < MT; m++)
#pragma unroll
                for (int n = 0; n < 4; n++)
#pragma unroll
                    for (int e = 0; e < 4; e++) acc[m][n][e] = 0.f;

            const int rt0 = row0 / 16;
            const bool v0 = rt0 < N / 16, v1 = rt0 + 1 < N / 16;
            const uint4* wp0 = wq + (size_t)(v0 ? rt0 : 0) * (K / 64) * 32 + lane;
            const uint4* wp1 = wq + (size_t)(v1 ? rt0 + 1 : 0) * (K / 64) * 32 + lane;
            // packed weight words of the stage in registers, the next stage's loads in flight
            uint4 qcur[4], qnxt[4];
            auto load_wreg = [&](int kcs, uint4 qd[4]) {
                const size_t ci = (size_t)(kb + kcs) * 2;
                qd[0] = __ldg(wp0 + ci * 32);
                qd[1] = __ldg(wp0 + (ci + 1) * 32);
                qd[2] = __ldg(wp1 + ci * 32);
                qd[3] = __ldg(wp1 + (ci + 1) * 32);
            };
            if (nk > 0) load_wreg(0, qnxt);
            uint32_t cc[2][4][4];  // ring of integer accumulator sets (tile parity)
            float3 rec_p[2];       // record of the pending (last) tile of the previous stage
            auto load_sw = [&](int kcs, float swf[4][2]) {  // weight scales: tile n/2, rows (n & 1) * 8 + 2t, +1
                const bf16* s0 = reinterpret_cast<const bf16*>(ssm + ((kcs % (NST + 1)) * 8 + wr) * 4);
#pragma unroll
                for (int n = 0; n < 4; n++) {
                    const float2 f = __bfloat1622float2(*reinterpret_cast<const bf162*>(s0 + n * 8 + 2 * t));
                    swf[n][0] = f.x;
                    swf[n][1] = f.y;
                }
            };
            auto flush = [&](int m, const uint32_t cm[4][4], const float3 r[2], const float swf[4][2]) {
#pragma unroll
                for (int n = 0; n < 4; n++)
#pragma unroll
                    for (int e = 0; e < 4; e++) {
                        const float f = __int_as_float(cm[n][e]);
                        const float3 rr = r[e >> 1];
                        acc[m][n][e] = fmaf(fmaf(f, rr.x, rr.y), swf[n][e & 1], acc[m][n][e]);
                    }
            };

            for (int kc = 0; kc < nk; kc++) {
                bar_sync(1 + kc % NST, kWsThreads);  // FULL(kc)
#pragma unroll
                for (int i = 0; i < 4; i++) qcur[i] = qnxt[i];
                if (kc + 1 < nk) load_wreg(kc + 1, qnxt);
                float sw[4][2];
                load_sw(kc, sw);

                const int stage = kc % NST;
                const uint4* xb = xs + stage * TB * 8;
                const float2* rb0 = reinterpret_cast<const float2*>(srec + (stage * TB + tokw + g) * NREC);
                auto load_rec = [&](int sub, int m, float3 r[2]) {  // (sx, -M sx, C) of tokens g, g + 8 of tile m
                    const float2 ra = rb0[(m * 16) * NREC * 2 + sub], rc = rb0[(m * 16 + 8) * NREC * 2 + sub];
                    r[0] = make_float3(ra.x, -kMagic * ra.x, ra.y);
                    r[1] = make_float3(rc.x, -kMagic * rc.x, rc.y);
                };
                float3 rec[2][2];  // ring: tiles j - 1, j (slot j & 1)
#pragma unroll
                for (int sub = 0; sub < SUB; sub++) {
                    uint32_t bw[4][KS][2];
                    constexpr uint32_t M4 = 0x0F0F0F0Fu;
#pragma unroll
                    for (int i = 0; i < 2; i++)  // 16-row tile i
#pragma unroll
                        for (int ks = 0; ks < KS; ks++) {
                            const int s = sub * KS + ks;  // k32 step of the stage: chunk s / 2, words (s & 1) * 2, +1
                            const uint4 q = qcur[2 * i + (s >> 1)];
                            const uint32_t w0 = (s & 1) ? q.z : q.x, w1 = (s & 1) ? q.w : q.y;
                            bw[2 * i][ks][0] = w0 & M4;            bw[2 * i][ks][1] = w1 & M4;
                            bw[2 * i + 1][ks][0] = (w0 >> 4) & M4; bw[2 * i + 1][ks][1] = (w1 >> 4) & M4;
                        }
#pragma unroll
                    for (int m = 0; m < MT; m++) {
                        const int j = sub * MT + m;  // tile index within the stage
                        load_rec(sub, m, rec[j & 1]);
                        const float3* rj = rec[j & 1];
                        const uint32_t cg = __float_as_uint(rj[0].z), cg8 = __float_as_uint(rj[1].z);
                        uint32_t (*cj)[4] = cc[j & 1];
#pragma unroll
                        for (int ks = 0; ks < KS; ks++) {
                            uint32_t a[4];
                            const int r = tokw + m * 16 + (lane & 7) + ((lane >> 3) & 1) * 8;
                            ldmatrix_x4(a, &xb[swz(r, 2 * (sub * KS + ks) + (lane >> 4))]);
                            if (ks == 0) {
#pragma unroll
                                for (int n = 0; n < 4; n++) mma_s8u8_init(cj[n], a, bw[n][ks][0], bw[n][ks][1], cg, cg8);
                            } else {
#pragma unroll
                                for (int n = 0; n < 4; n++) mma_s8u8(cj[n], a, bw[n][ks][0], bw[n][ks][1]);
                            }
                        }
                        if (j == 0) {
                            if (kc > 0) {  // last tile of the previous stage, with that stage's weight scales
                                float swp[4][2];
                                load_sw(kc - 1, swp);
                                flush(MT - 1, cc[1], rec_p, swp);
                            }
                        } else {
                            flush((j - 1) % MT, cc[(j - 1) & 1], rec[(j - 1) & 1], sw);
                        }
                    }
                }
#pragma unroll
                for (int h = 0; h < 2; h++) rec_p[h] = rec[1][h];
                bar_arrive(1 + NST + kc % NST, kWsThreads);  // EMPTY(kc)
            }
            if (nk > 0) {
                float swp[4][2];
                load_sw(nk - 1, swp);
                flush(MT - 1, cc[1], rec_p, swp);
            }

            if (part >= 0) {  // split tile: fp32 partial [TB tokens][ROWS] for sk_reduce_kernel (tokmajor)
                float* pw = wsp + ((size_t)(tile - full) * parts + part) * ROWS * TB;
#pragma unroll
                for (int m = 0; m < MT; m++)
#pragma unroll
                    for (int n = 0; n < 4; n++) {
                        const int lt = tokw + m * 16 + g, lr = wr * 32 + n * 8 + 2 * t;
                        *reinterpret_cast<float2*>(pw + (size_t)lt * ROWS + lr) = make_float2(acc[m][n][0], acc[m][n][1]);
                        *reinterpret_cast<float2*>(pw + (size_t)(lt + 8) * ROWS + lr) = make_float2(acc[m][n][2], acc[m][n][3]);
                    }
            } else if (glu) {
                bf16* act = reinterpret_cast<bf16*>(c);
#pragma unroll
                for (int i = 0; i < 2; i++) {
                    if (!(i == 0 ? v0 : v1)) continue;
                    const int col = (row0 + i * 16) / 2 + 2 * t;
#pragma unroll
                    for (int m = 0; m < MT; m++)
#pragma unroll
                        for (int h = 0; h < 2; h++) {
                            const int tk = tok0 + tokw + m * 16 + g + h * 8;
                            if (tk < T) {
                                const float g0 = acc[m][2 * i][2 * h], g1 = acc[m][2 * i][2 * h + 1];
                                const float u0 = acc[m][2 * i + 1][2 * h], u1 = acc[m][2 * i + 1][2 * h + 1];
                                *reinterpret_cast<bf162*>(act + (size_t)tk * ldc + col) = __floats2bfloat162_rn(
                                    g0 / (1.f + __expf(-g0)) * u0, g1 / (1.f + __expf(-g1)) * u1);
                            }
                        }
                }
            } else {
#pragma unroll
                for (int n = 0; n < 4; n++) {
                    if (!((n < 2) ? v0 : v1)) continue;
                    const int r = row0 + n * 8 + 2 * t;
#pragma unroll
                    for (int m = 0; m < MT; m++)
#pragma unroll
                        for (int h = 0; h < 2; h++) {
                            const int tk = tok0 + tokw + m * 16 + g + h * 8;
                            if (tk < T)
                                *reinterpret_cast<float2*>(c + (size_t)tk * ldc + r) =
                                    make_float2(acc[m][n][2 * h], acc[m][n][2 * h + 1]);
                        }
                }
            }
        }
        __syncthreads();  // smem stages are reused by the next unit's prologue
    }
}

// c[t][n] = sum_z part[z][t][n] (fixed order: deterministic split-K)
__global__ void splitk_reduce_kernel(const float* __restrict__ part, int ks, int t, int n,
                                     size_t zstride, float* __restrict__ c, int ldc, bool glu) {
    const size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (glu) {  // one thread per (token, activation column)
        if (i >= (size_t)t * (n / 2)) return;
        const int r = i / (n / 2), a = i % (n / 2);
        const int gr = (a / 8) * 16 + a % 8;
        float gv = 0.f, uv = 0.f;
        for (int z = 0; z < ks; z++) {
            gv += part[z * zstride + (size_t)r * n + gr];
            uv += part[z * zstride + (size_t)r * n + gr + 8];
        }
        reinterpret_cast<bf16*>(c)[(size_t)r * ldc + a] = __float2bfloat16(gv / (1.f + __expf(-gv)) * uv);
        return;
    }
    if (i >= (size_t)t * n) return;
    const int r = i / n, col = i % n;
    float v = 0.f;
    for (int z = 0; z < ks; z++) v += part[z * zstride + (size_t)r * n + col];
    c[(size_t)r * ldc + col] = v;
}

// per-thread device scratch, freed when the (GPU pool) thread exits
template <typename T>
struct TlBuf {
    T* p = nullptr;
    size_t n = 0;
    ~TlBuf() { if (p) cudaFree(p); }
};
thread_local TlBuf<float> tl_splitk_buf;
#define tl_splitk tl_splitk_buf.p
#define tl_splitk_n tl_splitk_buf.n

// One configuration of the W4 GEMM: token tile TB, warp layout WT, split-K count ks.
thread_local TlBuf<__half> tl_xh_buf;
thread_local TlBuf<float> tl_xs_buf;
#define tl_xh tl_xh_buf.p
#define tl_xs tl_xs_buf.p
#define tl_xh_n tl_xh_buf.n
bool g_gemm_h16 = [] {
    const char* e = getenv("QE_GEMM_H16");
    return e ? atoi(e) != 0 : true;
}();

// ---- W4A8 selection (QE_PREFILL_INT8: 1/all (default) | 0 | comma list of classes
// gu,in,out,down,fc; QE_PREFILL_INT8_G: activation group 32 / 64 (default) / 128, per class as
// "gu=64,in=64,out=32,down=32")
thread_local TlBuf<int8_t> tl_xq_buf;
thread_local TlBuf<float> tl_xqs_buf;
enum { I8_GU = 1, I8_IN = 2, I8_OUT = 4, I8_DOWN = 8, I8_FC = 16 };
int g_i8_mask = [] {
    const char* e = getenv("QE_PREFILL_INT8");
    if (!e || !*e || !strcmp(e, "1") || !strcmp(e, "all")) return 31;
    if (!strcmp(e, "0")) return 0;
    int m = 0;
    for (const char* p = e; *p;) {
        const char* q = strchr(p, ',');
        const size_t n = q ? (size_t)(q - p) : strlen(p);
        if (!strncmp(p, "gu", n)) m |= I8_GU;
        else if (!strncmp(p, "in", n)) m |= I8_IN;
        else if (!strncmp(p, "out", n)) m |= I8_OUT;
        else if (!strncmp(p, "down", n)) m |= I8_DOWN;
        else if (!strncmp(p, "fc", n)) m |= I8_FC;
        p = q ? q + 1 : p + n;
    }
    return m;
}();
// activation group per class: QE_PREFILL_INT8_G = "64" (all) or "gu=64,in=64,out=32,down=32,fc=64"
int g_i8_group[32] = {};
int i8_group_init = [] {
    const char* e = getenv("QE_PREFILL_INT8_G");
    auto valid = [](int g) { return (g == 64 || g == 128) ? g : 32; };
    int all = valid(e && !strchr(e, '=') ? atoi(e) : 64);
    for (int c = 0; c < 32; c++) g_i8_group[c] = all;
    if (e && strchr(e, '=')) {
        for (const char* p = e; *p;) {
            const char* q = strchr(p, ',');
            const size_t n = q ? (size_t)(q - p) : strlen(p);
            const char* eq = (const char*)memchr(p, '=', n);
            if (eq) {
                const size_t k = eq - p;
                const int g = valid(atoi(eq + 1));
                if (!strncmp(p, "gu", k)) g_i8_group[I8_GU] = g;
                else if (!strncmp(p, "in", k)) g_i8_group[I8_IN] = g;
                else if (!strncmp(p, "out", k)) g_i8_group[I8_OUT] = g;
                else if (!strncmp(p, "down", k)) g_i8_group[I8_DOWN] = g;
                else if (!strncmp(p, "fc", k)) g_i8_group[I8_FC] = g;
            }
            p = q ? q + 1 : p + n;
        }
    }
    return 0;
}();
// class of a prefill matrix by shape (Qwen3.8-27B: hidden 5120, mixer K 2048, MTP fc K 10240)
int i8_class(const W4& w) {
    if (w.glu) return I8_GU;
    if (w.k == 2048) return I8_OUT;
    if (w.k == 5120) return I8_IN;
    if (w.k == 10240) return I8_FC;
    return I8_DOWN;
}
bool i8_use(const W4& w) { return (g_i8_mask & i8_class(w)) != 0 && w.k % BKI == 0; }

// Epilogue of the split tiles of a persistent launch: c (or the GLU activation) of tile
// full + r = sum over parts (fixed order) of the fp32 partials wsp[(r * parts + p)][ROWS][TB].
// One thread per output element (row fastest: coalesced stores; the partials come from L2).
// tokmajor: partials stored [TB][ROWS] (gemm_i8t_kernel) instead of [ROWS][TB]
__global__ void sk_reduce_kernel(const float* __restrict__ wsp, int parts, int rem, int full, int rb,
                                 int ROWS, int TB, int N, int T, float* __restrict__ c, int ldc,
                                 bool glu, bool tokmajor, int grp_rows, bool o16 = false) {
    const size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    const int orows = glu ? ROWS / 2 : ROWS;  // outputs per token of a tile
    if (i >= (size_t)rem * orows * TB) return;
    const int r = i / ((size_t)orows * TB), lt = (i / orows) % TB, o = i % orows;
    const int tile = full + r;
    int rbi, tbi;
    if (tokmajor) {  // gemm_i8t_kernel's tile order (groups of grp_rows row blocks)
        const int GRP = min(grp_rows, rb);
        const int tbn = (T + TB - 1) / TB, grp = tile / (GRP * tbn);
        const int gsz = min(GRP, rb - grp * GRP), idx = tile - grp * GRP * tbn;
        tbi = idx / gsz;
        rbi = grp * GRP + idx % gsz;
    } else {
        rbi = tile % rb;
        tbi = tile / rb;
    }
    const int row0 = rbi * ROWS, tok = tbi * TB + lt;
    if (tok >= T) return;
    const size_t rs = tokmajor ? 1 : TB;  // element stride along rows
    const float* p0 = wsp + (size_t)r * parts * ROWS * TB + (tokmajor ? (size_t)lt * ROWS : (size_t)lt);
    const size_t ps = (size_t)ROWS * TB;
    if (glu) {  // activation column o of the tile <- gate row (o / 8) * 16 + o % 8, up row + 8
        const int lr = (o / 8) * 16 + o % 8;
        if (row0 + lr >= N) return;
        float gv = 0.f, uv = 0.f;
        for (int p = 0; p < parts; p++) {
            gv += p0[p * ps + (size_t)lr * rs];
            uv += p0[p * ps + (size_t)(lr + 8) * rs];
        }
        reinterpret_cast<bf16*>(c)[(size_t)tok * ldc + row0 / 2 + o] = __float2bfloat16(gv / (1.f + __expf(-gv)) * uv);
        return;
    }
    if (row0 + o >= N) return;
    float v = 0.f;
    for (int p = 0; p < parts; p++) v += p0[p * ps + (size_t)o * rs];
    if (o16) reinterpret_cast<bf16*>(c)[(size_t)tok * ldc + row0 + o] = __float2bfloat16(v);
    else c[(size_t)tok * ldc + row0 + o] = v;
}

int g_i8_direct = [] { const char* e = getenv("QE_PREFILL_INT8_DIRECT"); return e ? atoi(e) : 0; }();
// gemm_i8t_kernel tile order: groups of QE_I8_GRP row blocks (token blocks inside a group) so the
// concurrently running blocks share weights and activations in L2; 0 / large = row block fastest
int g_i8_grp = [] { const char* e = getenv("QE_I8_GRP"); const int v = e ? atoi(e) : 8; return v > 0 ? v : 1 << 20; }();
// persistent grid (QE_GEMM_SK_BLOCKS, default 82 = all SMs; measured better than leaving 4 SMs
// to the overlapped all-reduce blocks, 8K prefill 1384 vs 1406 ms) and the number of workers the
// decomposition assumes (QE_GEMM_SK_WORKERS, default = blocks)
int g_sk_blocks = [] { const char* e = getenv("QE_GEMM_SK_BLOCKS"); return e ? atoi(e) : 82; }();
int g_sk_workers = [] { const char* e = getenv("QE_GEMM_SK_WORKERS"); return e ? atoi(e) : 0; }();
thread_local TlBuf<int> tl_sk_ctr_buf;
thread_local TlBuf<float> tl_sk_ws_buf;
constexpr int kSkCtrs = 256;  // [0]: unit counter, [1 + r]: stream-K tickets of split tile r (SKF)

// Work decomposition of a persistent launch: full tiles, then the remainder tiles split into
// `parts` K-ranges so that the last wave fills the grid (cost model: waves of the split units
// times the unit length plus a fixed per-unit overhead of ~3 stages).
int g_sk_maxparts_rt = 0;  // (bench) run-time cap of the stream-K split count, 0 = QE_SK_MAXPARTS / default
struct SkPlan { int grid, full, rem, parts; };
SkPlan sk_plan(int tiles, int nkt, int per_sm = 1, int min_nk = 1) {
    const int skb = env_int("QE_GEMM_SK_BLOCKS", g_sk_blocks);  // per call: in-process A/B
    const int B = std::max(1, skb) * per_sm, P = (g_sk_workers > 0 ? g_sk_workers : B / per_sm) * per_sm;
    SkPlan pl{std::min(B, tiles), (tiles / P) * P, tiles % P, 1};
    if (pl.rem) {
        double best = 1e30;
        static const int maxp_env = [] { const char* e = getenv("QE_SK_MAXPARTS"); return e ? std::max(1, atoi(e)) : 64; }();
        const int maxp = g_sk_maxparts_rt > 0 ? g_sk_maxparts_rt : maxp_env;
        for (int parts = 1; parts <= std::min(std::min(nkt / min_nk, 64), maxp); parts++) {
            const int waves = (pl.rem * parts + P - 1) / P;
            // split units: pipeline prologue ~3 stages each; the reduction kernel reads all
            // partials of rem tiles (~1 stage per 8 tile-parts, measured ~40 us per 56 x 4)
            const double cost = waves * ((double)nkt / parts + 3.0) +
                                (parts > 1 ? 4.0 + pl.rem * parts / 8.0 : 0.0);
            if (cost < best - 1e-9) { best = cost; pl.parts = parts; }
        }
        if (pl.parts == 1) { pl.full = tiles; pl.rem = 0; }
        pl.grid = std::min(B, pl.full + pl.rem * pl.parts);
    }
    return pl;
}

// Input quantized by its producer (the deferred all-reduce tail writes xn and, when the next
// GEMM runs W4A8, also the int8 rows + scales): one pending entry per device thread, consumed
// by the first gemm_i8_cfg call with the same x pointer, rows, k and group.
struct PreQuant { const bf16* x = nullptr; int t = 0, k = 0, G = 0; int8_t* xq = nullptr; float* xsc = nullptr; bool zeroed = false, must = false; };
thread_local PreQuant tl_prequant;
thread_local TlBuf<int8_t> tl_pq_xq_buf;
thread_local TlBuf<float> tl_pq_xs_buf;
// bf16 output of non-GLU prefill GEMMs (gemm_set_out_bf16): written by gemm_i8v_kernel / sk_reduce directly,
// converted from an fp32 scratch result on the other paths
thread_local bool tl_out16 = false;
thread_local TlBuf<float> tl_o16_buf;
__global__ void f32_to_bf16_kernel(const float* __restrict__ src, int n, int t, bf16* __restrict__ dst, int ldd) {
    const size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= (size_t)n * t) return;
    const int r = i / n, col = i % n;
    dst[(size_t)r * ldd + col] = __float2bfloat16(src[i]);
}
bool g_i8_fuse = [] { const char* e = getenv("QE_PREFILL_INT8_FUSE"); return e ? atoi(e) != 0 : true; }();
// GLU epilogue quantizes the activations for the down GEMM (gemm_i8v_kernel; QE_I8_GLUQ=0 disables)
bool g_i8_gluq = [] { const char* e = getenv("QE_I8_GLUQ"); return e ? atoi(e) != 0 : true; }();
thread_local TlBuf<int8_t> tl_gq_xq_buf;
thread_local bool tl_gluq_var = false;  // set by gemm_i8_dispatch: only the engine variant (100) quantizes in the GLU epilogue
thread_local TlBuf<float> tl_gq_xs_buf;

// VF: gemm_i8v_kernel with HM = VF - 2 (2: shifted, 3: in-place u8 + C_hi records, 4: signed in place); MBV: mbarrier pipeline
template <int TB, int WT, int G, bool DIRECT, bool TR, int ABL = 0, int CCB = 64, int MINB = 1, int NST = STAGES, int NWB = 3, bool WREG = true, bool WS = false, int VF = 0, int MBV = 0>
void gemm_i8_cfg(const W4& w, const bf16* x, int ldx, float* c, int ldc, int t, int ks,
                 cudaStream_t st) {
    (void)ks;  // split-K is handled by the persistent decomposition
    const int rb = (w.n + block_rows<WT>() - 1) / block_rows<WT>(), tb = (t + TB - 1) / TB;
    const int nkt = w.k / BKI;
    const SkPlan pl = sk_plan(rb * tb, nkt, MINB, (VF >= 2 && VF <= 4 && (MBV & 8)) ? 3 : 1);
    const size_t need = (size_t)t * w.k;
    if (need > tl_xq_buf.n) {
        if (tl_xq_buf.p) CUDA_CHECK(cudaFree(tl_xq_buf.p));
        CUDA_CHECK(cudaMalloc(&tl_xq_buf.p, need));
        tl_xq_buf.n = need;
    }
    const size_t needs = VF ? (size_t)((t + 15) / 16 * 16) * i8v_rec_floats(w.k, G, VF == 3) : (size_t)t * i8_rec_floats(w.k, G);
    if (needs > tl_xqs_buf.n) {
        if (tl_xqs_buf.p) CUDA_CHECK(cudaFree(tl_xqs_buf.p));
        CUDA_CHECK(cudaMalloc(&tl_xqs_buf.p, needs * 4));
        tl_xqs_buf.n = needs;
    }
    if (!tl_sk_ctr_buf.p) {
        CUDA_CHECK(cudaMalloc(&tl_sk_ctr_buf.p, kSkCtrs * sizeof(int)));
        tl_sk_ctr_buf.n = kSkCtrs;
    }
    const int8_t* xq = tl_xq_buf.p;
    const float* xsc = tl_xqs_buf.p;
    bool pre = false, zeroed = false;
    if (tl_prequant.x == x && tl_prequant.t == t && tl_prequant.k == w.k && tl_prequant.G == G) {
        xq = tl_prequant.xq;
        xsc = tl_prequant.xsc;
        pre = true;
        zeroed = tl_prequant.zeroed;
    }
    // an external int8 input (owner-finished AR: the bf16 rows were never written) must be consumed
    CHECK(pre || !tl_prequant.must || tl_prequant.x == nullptr, "prequantized input not consumed (x %p)", (const void*)tl_prequant.x);
    tl_prequant.x = nullptr;  // consumed (or stale)
    const size_t needw = (size_t)pl.rem * pl.parts * block_rows<WT>() * TB;
    if (needw > tl_sk_ws_buf.n) {
        if (tl_sk_ws_buf.p) CUDA_CHECK(cudaFree(tl_sk_ws_buf.p));
        CUDA_CHECK(cudaMalloc(&tl_sk_ws_buf.p, needw * sizeof(float)));
        tl_sk_ws_buf.n = needw;
    }
    // GLU GEMM (gemm_i8v_kernel, 256-row blocks, no split tiles): quantize the activations for the down GEMM in
    // the epilogue and hand them over like the AR-tail quantizer does (QE_I8_GLUQ=0: down runs its pre-pass)
    int8_t* gq = nullptr;
    float* gsc = nullptr;
    if constexpr (VF >= 2 && VF <= 4) {
        if (w.glu && g_i8_gluq && tl_gluq_var && WT == 1 && w.n % 256 == 0 && pl.rem == 0 && (g_i8_mask & I8_DOWN) &&
            g_i8_group[I8_DOWN] == 64) {
            const int K2 = w.n / 2;
            const size_t nq = (size_t)t * K2, ns = (size_t)((t + 15) / 16 * 16) * i8v_rec_floats(K2, 64, false);
            if (nq > tl_gq_xq_buf.n) {
                if (tl_gq_xq_buf.p) CUDA_CHECK(cudaFree(tl_gq_xq_buf.p));
                CUDA_CHECK(cudaMalloc(&tl_gq_xq_buf.p, nq));
                tl_gq_xq_buf.n = nq;
            }
            if (ns > tl_gq_xs_buf.n) {
                if (tl_gq_xs_buf.p) CUDA_CHECK(cudaFree(tl_gq_xs_buf.p));
                CUDA_CHECK(cudaMalloc(&tl_gq_xs_buf.p, ns * 4));
                tl_gq_xs_buf.n = ns;
            }
            gq = tl_gq_xq_buf.p;
            gsc = tl_gq_xs_buf.p;
        }
    }
    // (merge prefill-night-combo: ar_finish zeroes only the unit counter ctr[0] (prefill-rest, QE_I8_CTR_FUSE); the
    // stream-K tickets of the bench-only SKF variant need the full memset)
    constexpr bool SKFV = VF >= 2 && VF <= 4 && (MBV & 131072);
    const int nctr = SKFV && pl.rem ? 1 + pl.rem : 1;
    CHECK(nctr <= kSkCtrs, "stream-K tickets %d", nctr);
    if (pre) { if (!zeroed || nctr > 1) CUDA_CHECK(cudaMemsetAsync(tl_sk_ctr_buf.p, 0, nctr * sizeof(int), st)); }
    else if constexpr (VF != 0) x_to_i8_kernel<G, VF == 3 ? 3 : 2><<<t, 256, 0, st>>>(x, ldx, w.k, tl_xq_buf.p, tl_xqs_buf.p, tl_sk_ctr_buf.p, nctr);
    else if constexpr (TR) x_to_i8_kernel<G><<<t, 256, 0, st>>>(x, ldx, w.k, tl_xq_buf.p, tl_xqs_buf.p, tl_sk_ctr_buf.p, 1);
    else x_to_i8_old_kernel<G><<<t, 256, 0, st>>>(x, ldx, w.k, tl_xq_buf.p, tl_xqs_buf.p, tl_sk_ctr_buf.p, 1);
    constexpr size_t smem = VF == 20 ? i8c_smem<WT, TB, G, NST>() : VF >= 10 ? i8w_smem<TB, G, VF - 10, NST>() : VF ? i8v_smem<WT, TB, G, VF - 2, NST, MBV>() : WS ? i8ws_smem<WT, TB, G, NST>() : TR ? i8t_smem<WT, TB, G, NST, NWB, WREG>()
                                : (size_t)STAGES * TB * (128 + 8 * (G == 128 ? 2 : 128 / G)) + 2 * kThreads * 64 + 2 * 8 * 64;
    static thread_local bool attr = false;  // per device (thread == device)
    if (!attr) {
        if constexpr (VF == 20) CUDA_CHECK(cudaFuncSetAttribute(gemm_i8c_kernel<WT, TB, G, MINB, NST>,
                                                cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
        else if constexpr (VF >= 10) CUDA_CHECK(cudaFuncSetAttribute(gemm_i8w_kernel<TB, G, VF - 10, MBV, NST>,
                                                cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
        else if constexpr (VF != 0) CUDA_CHECK(cudaFuncSetAttribute(gemm_i8v_kernel<WT, TB, G, VF - 2, MINB, NST, MBV>,
                                                cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
        else if constexpr (WS) CUDA_CHECK(cudaFuncSetAttribute(gemm_i8ws_kernel<WT, TB, G, NST>,
                                                cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
        else if constexpr (TR) CUDA_CHECK(cudaFuncSetAttribute(gemm_i8t_kernel<WT, TB, G, CCB, ABL, MINB, NST, NWB, WREG>,
                                                cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
        else CUDA_CHECK(cudaFuncSetAttribute(gemm_i8_kernel<WT, TB, G, DIRECT>,
                                             cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
        attr = true;
    }
    if (getenv("QE_GEMM_DEBUG"))
        fprintf(stderr, "gemm i8%s n=%d k=%d t=%d tb=%d wt=%d G=%d: tiles %d grid %d full %d rem %d parts %d\n",
                TR ? "t" : "", w.n, w.k, t, TB, WT, G, rb * tb, pl.grid, pl.full, pl.rem, pl.parts);
    if constexpr (VF == 20)
        gemm_i8c_kernel<WT, TB, G, MINB, NST><<<pl.grid, kThreads, smem, st>>>(
            w.q, w.s, w.n, w.k, xq, reinterpret_cast<const float4*>(xsc), t, c, ldc, w.glu, tl_sk_ctr_buf.p,
            tl_sk_ws_buf.p, pl.full, pl.parts, g_i8_grp);
    else if constexpr (VF >= 10)
        gemm_i8w_kernel<TB, G, VF - 10, MBV, NST><<<pl.grid, kWThreads, smem, st>>>(
            w.q, w.s, w.n, w.k, xq, reinterpret_cast<const float4*>(xsc), t, c, ldc, w.glu, tl_sk_ctr_buf.p,
            tl_sk_ws_buf.p, pl.full, pl.parts, g_i8_grp);
    else if constexpr (VF != 0)
        gemm_i8v_kernel<WT, TB, G, VF - 2, MINB, NST, MBV><<<pl.grid, kThreads, smem, st>>>(
            w.q, w.s, w.n, w.k, xq, reinterpret_cast<const float4*>(xsc), t, c, ldc, w.glu, tl_sk_ctr_buf.p,
            tl_sk_ws_buf.p, pl.full, pl.parts, g_i8_grp, gq, gsc, tl_out16 && !w.glu);
    else if constexpr (WS)
        gemm_i8ws_kernel<WT, TB, G, NST><<<pl.grid, kWsThreads, smem, st>>>(
            w.q, w.s, w.n, w.k, xq, reinterpret_cast<const float4*>(xsc), t, c, ldc, w.glu, tl_sk_ctr_buf.p,
            tl_sk_ws_buf.p, pl.full, pl.parts, g_i8_grp);
    else if constexpr (TR)
        gemm_i8t_kernel<WT, TB, G, CCB, ABL, MINB, NST, NWB, WREG><<<pl.grid, kThreads, smem, st>>>(
            w.q, w.s, w.n, w.k, xq, reinterpret_cast<const float4*>(xsc), t, c, ldc, w.glu, tl_sk_ctr_buf.p,
            tl_sk_ws_buf.p, pl.full, pl.parts, g_i8_grp);
    else
        gemm_i8_kernel<WT, TB, G, DIRECT><<<pl.grid, kThreads, smem, st>>>(
            w.q, w.s, w.n, w.k, xq, xsc, t, c, ldc, w.glu, tl_sk_ctr_buf.p,
            tl_sk_ws_buf.p, pl.full, pl.parts);
    if (gq) tl_prequant = PreQuant{reinterpret_cast<const bf16*>(c), t, w.n / 2, 64, gq, gsc};
    if (pl.rem && !SKFV) {
        const size_t n = (size_t)pl.rem * (w.glu ? block_rows<WT>() / 2 : block_rows<WT>()) * TB;
        sk_reduce_kernel<<<(n + 255) / 256, 256, 0, st>>>(tl_sk_ws_buf.p, pl.parts, pl.rem, pl.full, rb,
                                                          block_rows<WT>(), TB, w.n, t, c, ldc, w.glu, TR, g_i8_grp,
                                                          VF >= 2 && VF <= 4 && tl_out16 && !w.glu);
    }
}
// kernel variant: QE_PREFILL_INT8_KERNEL = 2 (gemm_i8v_kernel, default) | 1 (gemm_i8t_kernel) | 0 (gemm_i8_kernel)
int g_i8_kernel = [] { const char* e = getenv("QE_PREFILL_INT8_KERNEL"); return e ? atoi(e) : 2; }();
// bench variant (gemm_i8_dispatch) that QE_PREFILL_INT8_KERNEL=2 runs: 100 = gemm_i8v_kernel with the
// engine configuration below (high nibbles signed in place, mbarrier pipeline), any group size
constexpr int kI8vVar = 100;
// candidates of the W4A8 kernels by index ci (as gemm_run): the old kernel shares the fp16
// path's tiles; the new one keeps the warp tile at <= 64 tokens x 32 rows (ci 2 = 128 x 128
// block of 2 x 4 warps, ci 3 = 128 rows x 64 tokens) and needs TB <= 64 for G = 32 (smem).
template <int TB, int WT, int G, int CCB = 64>
void gemm_i8t_cfg(const W4& w, const bf16* x, int ldx, float* c, int ldc, int t, cudaStream_t st) {
    if constexpr (i8t_smem<WT, TB, G>() <= 99 * 1024) gemm_i8_cfg<TB, WT, G, false, true, 0, CCB>(w, x, ldx, c, ldc, t, 1, st);
    else gemm_i8_cfg<64, 1, G, false, true, 0, CCB>(w, x, ldx, c, ldc, t, 1, st);
}
// ci 0: 256 rows x 32 tokens; 1: 256 x 64 (the usual winner); 2: 128 x 64 (2 x 4 warps of
// 32 x 32); 3: 128 x 64 with 2 blocks per SM (128 registers). The 256 x 128 block (warp 32 x 128)
// needs 128 fp32 + 32 int accumulators and spills at 255 registers (122 TOPS): not offered.
template <int G, int VF, int MBV = 0>
void gemm_i8v_run(int ci, const W4& w, const bf16* x, int ldx, float* c, int ldc, int t, cudaStream_t st) {
    switch (ci) {
        case 0: gemm_i8_cfg<32, 1, G, false, true, 0, 64, 1, STAGES, 3, true, false, VF, MBV>(w, x, ldx, c, ldc, t, 1, st); break;
        case 1: gemm_i8_cfg<64, 1, G, false, true, 0, 64, 1, STAGES, 3, true, false, VF, MBV>(w, x, ldx, c, ldc, t, 1, st); break;
        case 2: gemm_i8_cfg<64, 2, G, false, true, 0, 64, 1, STAGES, 3, true, false, VF, MBV>(w, x, ldx, c, ldc, t, 1, st); break;
        default: gemm_i8_cfg<64, 2, G, false, true, 0, 64, 2, STAGES, 3, true, false, VF, MBV>(w, x, ldx, c, ldc, t, 1, st); break;
    }
}
template <int G>
void gemm_i8t_run(int ci, const W4& w, const bf16* x, int ldx, float* c, int ldc, int t, cudaStream_t st) {
    switch (ci) {
        case 0: gemm_i8t_cfg<32, 1, G>(w, x, ldx, c, ldc, t, st); break;
        case 1: gemm_i8t_cfg<64, 1, G>(w, x, ldx, c, ldc, t, st); break;
        case 2: gemm_i8t_cfg<64, 2, G>(w, x, ldx, c, ldc, t, st); break;
        default: gemm_i8_cfg<64, 2, G, false, true, 0, 64, 2>(w, x, ldx, c, ldc, t, 1, st); break;
    }
}
template <int TB, int WT>
void gemm_i8_run(int G, const W4& w, const bf16* x, int ldx, float* c, int ldc, int t, int ks,
                 cudaStream_t st) {
    if (g_i8_direct) {
        if (G == 128) gemm_i8_cfg<TB, WT, 128, true, false>(w, x, ldx, c, ldc, t, ks, st);
        else if (G == 64) gemm_i8_cfg<TB, WT, 64, true, false>(w, x, ldx, c, ldc, t, ks, st);
        else gemm_i8_cfg<TB, WT, 32, true, false>(w, x, ldx, c, ldc, t, ks, st);
        return;
    }
    if (G == 128) gemm_i8_cfg<TB, WT, 128, false, false>(w, x, ldx, c, ldc, t, ks, st);
    else if (G == 64) gemm_i8_cfg<TB, WT, 64, false, false>(w, x, ldx, c, ldc, t, ks, st);
    else gemm_i8_cfg<TB, WT, 32, false, false>(w, x, ldx, c, ldc, t, ks, st);
}
// W4A8 dispatch by candidate index and kernel variant
void gemm_i8_dispatch(int var, int ci, int G, const W4& w, const bf16* x, int ldx, float* c, int ldc, int t,
                      cudaStream_t st) {
    tl_gluq_var = var == 100 || var == 101;
    if (var == 2) {  // warp-specialized variant (gate 1), 256 x 64 tiles
        if (G == 128) gemm_i8_cfg<64, 1, 128, false, true, 0, 64, 1, STAGES, 3, true, true>(w, x, ldx, c, ldc, t, 1, st);
        else if (G == 64) gemm_i8_cfg<64, 1, 64, false, true, 0, 64, 1, STAGES, 3, true, true>(w, x, ldx, c, ldc, t, 1, st);
        else gemm_i8_cfg<64, 1, 32, false, true, 0, 64, 1, STAGES, 3, true, true>(w, x, ldx, c, ldc, t, 1, st);
        return;
    }
    if (var == 22 || var == 23) {  // gemm_i8c_kernel (continuous stream), 3 / 4 stages
        if (var == 23) {
            if (G == 64) gemm_i8_cfg<64, 1, 64, false, true, 0, 64, 1, 4, 3, true, false, 20, 0>(w, x, ldx, c, ldc, t, 1, st);
            else gemm_i8v_run<128, 4, 1>(ci, w, x, ldx, c, ldc, t, st);
            return;
        }
        if (G == 128) gemm_i8v_run<128, 20, 0>(ci, w, x, ldx, c, ldc, t, st);
        else if (G == 64) gemm_i8v_run<64, 20, 0>(ci, w, x, ldx, c, ldc, t, st);
        else gemm_i8v_run<32, 20, 0>(ci, w, x, ldx, c, ldc, t, st);
        return;
    }
    if (var >= 1000) {  // ablations of the engine configuration: var = 1000 + OPT ablation bits (G = 64, ci as given)
        switch (var - 1000) {
            case 16: gemm_i8v_run<64, 4, 1 | 16>(ci, w, x, ldx, c, ldc, t, st); break;
            case 32: gemm_i8v_run<64, 4, 1 | 32>(ci, w, x, ldx, c, ldc, t, st); break;
            case 64: gemm_i8v_run<64, 4, 1 | 64>(ci, w, x, ldx, c, ldc, t, st); break;
            case 128: gemm_i8v_run<64, 4, 1 | 128>(ci, w, x, ldx, c, ldc, t, st); break;
            case 256: gemm_i8v_run<64, 4, 1 | 256>(ci, w, x, ldx, c, ldc, t, st); break;
            case 512: gemm_i8v_run<64, 4, 1 | 512>(ci, w, x, ldx, c, ldc, t, st); break;
            case 16 | 32 | 64 | 128: gemm_i8v_run<64, 4, 1 | 16 | 32 | 64 | 128>(ci, w, x, ldx, c, ldc, t, st); break;
            case 16 | 32 | 64 | 128 | 512: gemm_i8v_run<64, 4, 1 | 16 | 32 | 64 | 128 | 512>(ci, w, x, ldx, c, ldc, t, st); break;
            case 16 | 32 | 64 | 128 | 256 | 512: gemm_i8v_run<64, 4, 1 | 16 | 32 | 64 | 128 | 256 | 512>(ci, w, x, ldx, c, ldc, t, st); break;
            default: CHECK(false, "ablation %d", var);
        }
        return;
    }
    if (var == 27 || var == 28) {  // C1 + mbarrier with spin back-off 32 / 256 ns
        if (var == 27) {
            if (G == 128) gemm_i8v_run<128, 4, 1 | 4096 | 8192>(ci, w, x, ldx, c, ldc, t, st);
            else if (G == 64) gemm_i8v_run<64, 4, 1 | 4096 | 8192>(ci, w, x, ldx, c, ldc, t, st);
            else gemm_i8v_run<32, 4, 1 | 4096 | 8192>(ci, w, x, ldx, c, ldc, t, st);
        } else {
            if (G == 128) gemm_i8v_run<128, 4, 1 | 4096 | 16384>(ci, w, x, ldx, c, ldc, t, st);
            else if (G == 64) gemm_i8v_run<64, 4, 1 | 4096 | 16384>(ci, w, x, ldx, c, ldc, t, st);
            else gemm_i8v_run<32, 4, 1 | 4096 | 16384>(ci, w, x, ldx, c, ldc, t, st);
        }
        return;
    }
    if (var == 26) {  // C1 + continuous stream (CSL)
        if (w.k / BKI < 3) { gemm_i8v_run<64, 4, 1>(ci, w, x, ldx, c, ldc, t, st); return; }
        if (G == 128) gemm_i8v_run<128, 4, 1 | 8 | 4096>(ci, w, x, ldx, c, ldc, t, st);
        else if (G == 64) gemm_i8v_run<64, 4, 1 | 8 | 4096>(ci, w, x, ldx, c, ldc, t, st);
        else gemm_i8v_run<32, 4, 1 | 8 | 4096>(ci, w, x, ldx, c, ldc, t, st);
        return;
    }
    if (var == 25) {  // engine configuration with one accumulator set (C1)
        if (G == 128) gemm_i8v_run<128, 4, 1 | 4096>(ci, w, x, ldx, c, ldc, t, st);
        else if (G == 64) gemm_i8v_run<64, 4, 1 | 4096>(ci, w, x, ldx, c, ldc, t, st);
        else gemm_i8v_run<32, 4, 1 | 4096>(ci, w, x, ldx, c, ldc, t, st);
        return;
    }
    if (var == 24) {  // engine configuration + continuous stream across units (needs K >= 384)
        if (w.k / BKI < 3) { gemm_i8v_run<64, 4, 1>(ci, w, x, ldx, c, ldc, t, st); return; }
        if (G == 128) gemm_i8v_run<128, 4, 1 | 8>(ci, w, x, ldx, c, ldc, t, st);
        else if (G == 64) gemm_i8v_run<64, 4, 1 | 8>(ci, w, x, ldx, c, ldc, t, st);
        else gemm_i8v_run<32, 4, 1 | 8>(ci, w, x, ldx, c, ldc, t, st);
        return;
    }
    if (var == 100) {  // engine default (QE_PREFILL_INT8_KERNEL=2): HS + mbarrier pipeline (32 ns spin back-off) +
                       // one accumulator set (C1)
        if (G == 128) gemm_i8v_run<128, 4, 1 | 4096 | 8192>(ci, w, x, ldx, c, ldc, t, st);
        else if (G == 64) gemm_i8v_run<64, 4, 1 | 4096 | 8192>(ci, w, x, ldx, c, ldc, t, st);
        else gemm_i8v_run<32, 4, 1 | 4096 | 8192>(ci, w, x, ldx, c, ldc, t, st);
        return;
    }
    if (var == 29) {  // engine default + EP
        if (G == 128) gemm_i8v_run<128, 4, 1 | 4096 | 8192 | 32768>(ci, w, x, ldx, c, ldc, t, st);
        else if (G == 64) gemm_i8v_run<64, 4, 1 | 4096 | 8192 | 32768>(ci, w, x, ldx, c, ldc, t, st);
        else gemm_i8v_run<32, 4, 1 | 4096 | 8192 | 32768>(ci, w, x, ldx, c, ldc, t, st);
        return;
    }
    if (var == 30) {  // engine default + SEP (barrier-free unit boundaries)
        if (G == 128) gemm_i8v_run<128, 4, 1 | 4096 | 8192 | 65536>(ci, w, x, ldx, c, ldc, t, st);
        else if (G == 64) gemm_i8v_run<64, 4, 1 | 4096 | 8192 | 65536>(ci, w, x, ldx, c, ldc, t, st);
        else gemm_i8v_run<32, 4, 1 | 4096 | 8192 | 65536>(ci, w, x, ldx, c, ldc, t, st);
        return;
    }
    if (var == 31) {  // engine default + in-kernel stream-K fixup (SKF)
        if (G == 128) gemm_i8v_run<128, 4, 1 | 4096 | 8192 | 131072>(ci, w, x, ldx, c, ldc, t, st);
        else if (G == 64) gemm_i8v_run<64, 4, 1 | 4096 | 8192 | 131072>(ci, w, x, ldx, c, ldc, t, st);
        else gemm_i8v_run<32, 4, 1 | 4096 | 8192 | 131072>(ci, w, x, ldx, c, ldc, t, st);
        return;
    }
    if (var == 45) {  // engine default + weight LDGs without L1 allocation
        if (G != 64) { gemm_i8v_run<64, 4, 1 | 4096 | 8192>(ci, w, x, ldx, c, ldc, t, st); return; }
        gemm_i8v_run<64, 4, 1 | 4096 | 8192 | 16777216>(ci, w, x, ldx, c, ldc, t, st);
        return;
    }
    if (var >= 41 && var <= 44) {  // engine default with other spin back-offs: 16 / 64 / 128 ns, none
        constexpr int B = 1 | 4096;
        if (G != 64) { gemm_i8v_run<64, 4, 1 | 4096 | 8192>(ci, w, x, ldx, c, ldc, t, st); return; }
        if (var == 41) gemm_i8v_run<64, 4, B | 8388608>(ci, w, x, ldx, c, ldc, t, st);
        else if (var == 42) gemm_i8v_run<64, 4, B | 2097152>(ci, w, x, ldx, c, ldc, t, st);
        else if (var == 43) gemm_i8v_run<64, 4, B | 4194304>(ci, w, x, ldx, c, ldc, t, st);
        else gemm_i8v_run<64, 4, B>(ci, w, x, ldx, c, ldc, t, st);
        tl_gluq_var = false;
        return;
    }
    if (var == 40) {  // WSM + continuous load stream (CS2)
        if (G != 64) { gemm_i8v_run<64, 4, 1 | 4096 | 8192>(ci, w, x, ldx, c, ldc, t, st); return; }
        gemm_i8v_run<64, 4, 1 | 4096 | 8192 | 524288 | 1048576>(ci, w, x, ldx, c, ldc, t, st);
        return;
    }
    if (var == 39) {  // WSM with the 2-set accumulator ring (no C1)
        if (G != 64) { gemm_i8v_run<64, 4, 1 | 4096 | 8192>(ci, w, x, ldx, c, ldc, t, st); return; }
        gemm_i8v_run<64, 4, 1 | 8192 | 524288>(ci, w, x, ldx, c, ldc, t, st);
        return;
    }
    if (var >= 35 && var <= 38) {  // WSM: engine default (35), + SEP (36), + A prefetch (37), + EP (38)
        constexpr int B = 1 | 4096 | 8192 | 524288;
        if (G != 64) { gemm_i8v_run<64, 4, 1 | 4096 | 8192>(ci, w, x, ldx, c, ldc, t, st); return; }
        if (var == 35) gemm_i8v_run<64, 4, B>(ci, w, x, ldx, c, ldc, t, st);
        else if (var == 36) gemm_i8v_run<64, 4, B | 65536>(ci, w, x, ldx, c, ldc, t, st);
        else if (var == 37) gemm_i8v_run<64, 4, B | 2>(ci, w, x, ldx, c, ldc, t, st);
        else gemm_i8v_run<64, 4, B | 32768>(ci, w, x, ldx, c, ldc, t, st);
        return;
    }
    if (var == 101) {  // previous engine configuration (HS + mbarrier, ring of 2 accumulator sets)
        if (G == 128) gemm_i8v_run<128, 4, 1>(ci, w, x, ldx, c, ldc, t, st);
        else if (G == 64) gemm_i8v_run<64, 4, 1>(ci, w, x, ldx, c, ldc, t, st);
        else gemm_i8v_run<32, 4, 1>(ci, w, x, ldx, c, ldc, t, st);
        return;
    }
    if (var >= 16 && var <= 21) {  // mbarrier variants: 16 HM0, 17 HS 4 stages, 18 i8w HS, 19 i8w HM0, 20 i8w HS 4 st, 21 HM0 4 st
        if (G != 64) { gemm_i8v_run<64, 4>(ci, w, x, ldx, c, ldc, t, st); return; }
        switch (var) {
            case 16: gemm_i8v_run<64, 2, 1>(ci, w, x, ldx, c, ldc, t, st); break;
            case 17: gemm_i8_cfg<64, 1, 64, false, true, 0, 64, 1, 4, 3, true, false, 4, 1>(w, x, ldx, c, ldc, t, 1, st); break;
            case 18: gemm_i8_cfg<64, 1, 64, false, true, 0, 64, 1, STAGES, 3, true, false, 12, 1>(w, x, ldx, c, ldc, t, 1, st); break;
            case 19: gemm_i8_cfg<64, 1, 64, false, true, 0, 64, 1, STAGES, 3, true, false, 10, 1>(w, x, ldx, c, ldc, t, 1, st); break;
            case 20: gemm_i8_cfg<64, 1, 64, false, true, 0, 64, 1, 4, 3, true, false, 12, 1>(w, x, ldx, c, ldc, t, 1, st); break;
            default: gemm_i8_cfg<64, 1, 64, false, true, 0, 64, 1, 4, 3, true, false, 2, 1>(w, x, ldx, c, ldc, t, 1, st); break;
        }
        return;
    }
    if (var >= 10 && var <= 15) {  // gemm_i8w_kernel 256 x 64 (16 warps): 10 HM0, 11 HS, 12 HM0 + PF, 13 HS + PF, 14 HS + PSU, 15 HS + PF + PSU
        if (G != 64) { gemm_i8v_run<64, 4>(ci, w, x, ldx, c, ldc, t, st); return; }
        switch (var) {
            case 10: gemm_i8_cfg<64, 1, 64, false, true, 0, 64, 1, STAGES, 3, true, false, 10, 0>(w, x, ldx, c, ldc, t, 1, st); break;
            case 11: gemm_i8_cfg<64, 1, 64, false, true, 0, 64, 1, STAGES, 3, true, false, 12, 0>(w, x, ldx, c, ldc, t, 1, st); break;
            case 12: gemm_i8_cfg<64, 1, 64, false, true, 0, 64, 1, STAGES, 3, true, false, 10, 2>(w, x, ldx, c, ldc, t, 1, st); break;
            case 14: gemm_i8_cfg<64, 1, 64, false, true, 0, 64, 1, STAGES, 3, true, false, 12, 4>(w, x, ldx, c, ldc, t, 1, st); break;
            case 15: gemm_i8_cfg<64, 1, 64, false, true, 0, 64, 1, STAGES, 3, true, false, 12, 6>(w, x, ldx, c, ldc, t, 1, st); break;
            default: gemm_i8_cfg<64, 1, 64, false, true, 0, 64, 1, STAGES, 3, true, false, 12, 2>(w, x, ldx, c, ldc, t, 1, st); break;
        }
        return;
    }
    if (var >= 7 && var <= 9) {  // var 6 + OPT var - 6 (1: mbarrier pipeline, 2: A prefetch, 3: both)
        switch (var) {
            case 7: gemm_i8v_run<64, 4, 1>(ci, w, x, ldx, c, ldc, t, st); break;
            case 8: gemm_i8v_run<64, 4, 2>(ci, w, x, ldx, c, ldc, t, st); break;
            default: gemm_i8v_run<64, 4, 3>(ci, w, x, ldx, c, ldc, t, st); break;
        }
        return;
    }
    if (var == 6) {  // gemm_i8v_kernel, high nibbles as signed 16 w (s8 x s8 from the magic constant)
        if (G == 128) gemm_i8v_run<128, 4>(ci, w, x, ldx, c, ldc, t, st);
        else if (G == 64) gemm_i8v_run<64, 4>(ci, w, x, ldx, c, ldc, t, st);
        else gemm_i8v_run<32, 4>(ci, w, x, ldx, c, ldc, t, st);
        return;
    }
    if (var == 4 || var == 5) {  // gemm_i8v_kernel (4: in-place high nibbles, 5: shifted)
        if (var == 4) {
            if (G == 128) gemm_i8v_run<128, 3>(ci, w, x, ldx, c, ldc, t, st);
            else if (G == 64) gemm_i8v_run<64, 3>(ci, w, x, ldx, c, ldc, t, st);
            else gemm_i8v_run<32, 3>(ci, w, x, ldx, c, ldc, t, st);
        } else {
            if (G == 128) gemm_i8v_run<128, 2>(ci, w, x, ldx, c, ldc, t, st);
            else if (G == 64) gemm_i8v_run<64, 2>(ci, w, x, ldx, c, ldc, t, st);
            else gemm_i8v_run<32, 2>(ci, w, x, ldx, c, ldc, t, st);
        }
        return;
    }
    if (var == 3) {  // warp-specialized, 4 stages
        gemm_i8_cfg<64, 1, 64, false, true, 0, 64, 1, 4, 3, true, true>(w, x, ldx, c, ldc, t, 1, st);
        return;
    }
    if (var >= 16) {  // ablation experiments: <64, 1, 64> only
        switch (var >> 4) {
            case 1: gemm_i8_cfg<64, 1, 64, false, true, 1>(w, x, ldx, c, ldc, t, 1, st); break;
            case 2: gemm_i8_cfg<64, 1, 64, false, true, 2>(w, x, ldx, c, ldc, t, 1, st); break;
            case 3: gemm_i8_cfg<64, 1, 64, false, true, 3>(w, x, ldx, c, ldc, t, 1, st); break;
            case 4: gemm_i8_cfg<64, 1, 64, false, true, 4>(w, x, ldx, c, ldc, t, 1, st); break;
            case 7: gemm_i8_cfg<64, 1, 64, false, true, 7>(w, x, ldx, c, ldc, t, 1, st); break;
            case 8: gemm_i8_cfg<64, 1, 64, false, true, 8>(w, x, ldx, c, ldc, t, 1, st); break;
            case 9: gemm_i8_cfg<64, 1, 64, false, true, 9>(w, x, ldx, c, ldc, t, 1, st); break;
            case 11: gemm_i8_cfg<64, 1, 64, false, true, 11>(w, x, ldx, c, ldc, t, 1, st); break;
            case 15: gemm_i8_cfg<64, 1, 64, false, true, 15>(w, x, ldx, c, ldc, t, 1, st); break;
            case 16: gemm_i8_cfg<64, 1, 64, false, true, 16>(w, x, ldx, c, ldc, t, 1, st); break;
            case 17: gemm_i8_cfg<64, 1, 64, false, true, 17>(w, x, ldx, c, ldc, t, 1, st); break;
            case 31: gemm_i8_cfg<64, 1, 64, false, true, 31>(w, x, ldx, c, ldc, t, 1, st); break;
            case 47: gemm_i8_cfg<64, 1, 64, false, true, 47>(w, x, ldx, c, ldc, t, 1, st); break;
            case 63: gemm_i8_cfg<64, 1, 64, false, true, 63>(w, x, ldx, c, ldc, t, 1, st); break;
            // 128-token warp tile, ring of 2 (ci 2): 64 + ABL; ring of 3: 192 + ABL
            case 64 + 0: gemm_i8_cfg<128, 1, 64, false, true, 0, 64, 1, STAGES, 2>(w, x, ldx, c, ldc, t, 1, st); break;
            case 64 + 8: gemm_i8_cfg<128, 1, 64, false, true, 8, 64, 1, STAGES, 2>(w, x, ldx, c, ldc, t, 1, st); break;
            case 64 + 15: gemm_i8_cfg<128, 1, 64, false, true, 15, 64, 1, STAGES, 2>(w, x, ldx, c, ldc, t, 1, st); break;
            case 64 + 16: gemm_i8_cfg<128, 1, 64, false, true, 16, 64, 1, STAGES, 2>(w, x, ldx, c, ldc, t, 1, st); break;
            case 64 + 63: gemm_i8_cfg<128, 1, 64, false, true, 63, 64, 1, STAGES, 2>(w, x, ldx, c, ldc, t, 1, st); break;
            case 192 + 0: gemm_i8_cfg<128, 1, 64, false, true, 0, 64, 1, 2, 2>(w, x, ldx, c, ldc, t, 1, st); break;  // x 2 stages, w 2
            case 192 + 15: gemm_i8_cfg<128, 1, 64, false, true, 15, 64, 1, 2, 2>(w, x, ldx, c, ldc, t, 1, st); break;
            case 256 + 0: gemm_i8_cfg<64, 2, 64, false, true, 0, 64, 2>(w, x, ldx, c, ldc, t, 1, st); break;
            case 256 + 15: gemm_i8_cfg<64, 2, 64, false, true, 15, 64, 2>(w, x, ldx, c, ldc, t, 1, st); break;
            case 256 + 63: gemm_i8_cfg<64, 2, 64, false, true, 63, 64, 2>(w, x, ldx, c, ldc, t, 1, st); break;
            case 256 + 64: gemm_i8_cfg<64, 2, 64, false, true, 0, 64, 1>(w, x, ldx, c, ldc, t, 1, st); break;
            case 100: gemm_i8_cfg<64, 1, 64, false, true, 0, 64, 1, 4>(w, x, ldx, c, ldc, t, 1, st); break;   // 4 stages
            case 101: gemm_i8_cfg<64, 1, 64, false, true, 64, 64, 1, 4>(w, x, ldx, c, ldc, t, 1, st); break;
            case 102: gemm_i8_cfg<64, 1, 64, false, true, 0, 64, 1, 5>(w, x, ldx, c, ldc, t, 1, st); break;   // 5 stages
            case 103: gemm_i8_cfg<64, 1, 64, false, true, 64>(w, x, ldx, c, ldc, t, 1, st); break;           // no weight cp.async
            case 104: gemm_i8_cfg<64, 1, 64, false, true, 65>(w, x, ldx, c, ldc, t, 1, st); break;
            case 105: gemm_i8_cfg<64, 1, 64, false, true, 64 | 32>(w, x, ldx, c, ldc, t, 1, st); break;
            case 106: gemm_i8_cfg<64, 1, 64, false, true, 128>(w, x, ldx, c, ldc, t, 1, st); break;   // x loads at the top
            case 107: gemm_i8_cfg<64, 1, 64, false, true, 0, 64, 1, STAGES, 3, false>(w, x, ldx, c, ldc, t, 1, st); break;  // weights via smem
            case 108: gemm_i8_cfg<64, 1, 64, false, true, 0, 64, 1, 4, 3, true>(w, x, ldx, c, ldc, t, 1, st); break;  // 4 x stages
            case 109: gemm_i8_cfg<64, 1, 64, false, true, 0, 64, 1, 5, 3, true>(w, x, ldx, c, ldc, t, 1, st); break;  // 5 x stages      // no weight / x cp.async
            // 64-token warp tile with the ring of 2 (vs. all-at-once): 128 + ABL
            case 128 + 0: gemm_i8_cfg<64, 1, 64, false, true, 0, 2>(w, x, ldx, c, ldc, t, 1, st); break;
            case 128 + 15: gemm_i8_cfg<64, 1, 64, false, true, 15, 2>(w, x, ldx, c, ldc, t, 1, st); break;
            default: CHECK(false, "ablation %d", var);
        }
        return;
    }
    if (var) {
        if (G == 128) gemm_i8t_run<128>(ci, w, x, ldx, c, ldc, t, st);
        else if (G == 64) gemm_i8t_run<64>(ci, w, x, ldx, c, ldc, t, st);
        else gemm_i8t_run<32>(ci, w, x, ldx, c, ldc, t, st);
        return;
    }
    switch (ci) {
        case 0: gemm_i8_run<32, 1>(G, w, x, ldx, c, ldc, t, 1, st); break;
        case 1: gemm_i8_run<64, 1>(G, w, x, ldx, c, ldc, t, 1, st); break;
        case 2: gemm_i8_run<128, 1>(G, w, x, ldx, c, ldc, t, 1, st); break;
        default: gemm_i8_run<128, 2>(G, w, x, ldx, c, ldc, t, 1, st); break;
    }
}

template <int TB, int WT>
void gemm_cfg(const W4& w, const bf16* x, int ldx, float* c, int ldc, int t, int ks,
              cudaStream_t st) {
    const int rb = (w.n + block_rows<WT>() - 1) / block_rows<WT>(), tb = (t + TB - 1) / TB;
    if (g_gemm_h16) {
        // grow during the first prefills (never inside graph capture); the fp16 activations
        // need t * k elements, the token scales t: both checked (a call with more tokens but a
        // smaller k must still grow the scales)
        const size_t need = (size_t)t * w.k;
        if (need > tl_xh_n) {
            if (tl_xh) CUDA_CHECK(cudaFree(tl_xh));
            CUDA_CHECK(cudaMalloc(&tl_xh, need * 2));
            tl_xh_n = need;
        }
        if ((size_t)t > tl_xs_buf.n) {
            if (tl_xs) CUDA_CHECK(cudaFree(tl_xs));
            CUDA_CHECK(cudaMalloc(&tl_xs, (size_t)t * 4));
            tl_xs_buf.n = t;
        }
        x_to_h_kernel<<<t, 256, 0, st>>>(x, ldx, w.k, tl_xh, tl_xs);
        if (ks == 1) {
            gemm_h_kernel<WT, TB><<<dim3(rb, tb), kThreads, 0, st>>>(w.q, w.s, w.n, w.k, tl_xh, tl_xs,
                                                                    t, c, ldc, 0, w.glu);
            return;
        }
        const size_t zs = (size_t)t * w.n, needp = zs * ks;
        if (needp > tl_splitk_n) {
            if (tl_splitk) CUDA_CHECK(cudaFree(tl_splitk));
            CUDA_CHECK(cudaMalloc(&tl_splitk, needp * sizeof(float)));
            tl_splitk_n = needp;
        }
        gemm_h_kernel<WT, TB><<<dim3(rb, tb, ks), kThreads, 0, st>>>(w.q, w.s, w.n, w.k, tl_xh, tl_xs,
                                                                    t, tl_splitk, w.n, zs, false);
        splitk_reduce_kernel<<<(zs + 255) / 256, 256, 0, st>>>(tl_splitk, ks, t, w.n, zs, c, ldc,
                                                               w.glu);
        return;
    }
    if (ks == 1) {
        gemm_kernel<true, WT, TB><<<dim3(rb, tb), kThreads, 0, st>>>(w.q, w.s, w.n, w.k, x, ldx,
                                                                     t, c, ldc, 0, w.glu);
        return;
    }
    const size_t zs = (size_t)t * w.n, need = zs * ks;
    if (need > tl_splitk_n) {
        if (tl_splitk) CUDA_CHECK(cudaFree(tl_splitk));
        CUDA_CHECK(cudaMalloc(&tl_splitk, need * sizeof(float)));
        tl_splitk_n = need;
    }
    gemm_kernel<true, WT, TB><<<dim3(rb, tb, ks), kThreads, 0, st>>>(w.q, w.s, w.n, w.k, x, ldx,
                                                                     t, tl_splitk, w.n, zs, false);
    splitk_reduce_kernel<<<(zs + 255) / 256, 256, 0, st>>>(tl_splitk, ks, t, w.n, zs, c, ldc,
                                                           w.glu);
}

// Estimated time of a configuration: waves of equal blocks (one block per SM on 82 SMs) times
// the block's work, plus the split-K partial traffic. eff: relative efficiency of the layout.
double gemm_cost(int n, int k, int t, int tb_tokens, int rows, int ks, double eff) {
    const long blocks = (long)((n + rows - 1) / rows) * ((t + tb_tokens - 1) / tb_tokens) * ks;
    const double t_block = 2.0 * rows * tb_tokens * k / ks / (0.75e12 * eff);
    return (double)((blocks + 81) / 82) * t_block + (ks > 1 ? (double)ks * t * n * 8 / 700e9 : 0.0);
}

void gemm_run(int ci, const W4& w, const bf16* x, int ldx, float* c, int ldc, int t, int ks,
              cudaStream_t st) {
    if (i8_use(w)) {
        gemm_i8_dispatch(g_i8_kernel >= 2 ? kI8vVar : g_i8_kernel, ci, g_i8_group[i8_class(w)], w, x, ldx, c, ldc, t, st);
        return;
    }
    switch (ci) {
        case 0: gemm_cfg<32, 1>(w, x, ldx, c, ldc, t, ks, st); break;
        case 1: gemm_cfg<64, 1>(w, x, ldx, c, ldc, t, ks, st); break;
        case 2: gemm_cfg<128, 1>(w, x, ldx, c, ldc, t, ks, st); break;
        default: gemm_cfg<128, 2>(w, x, ldx, c, ldc, t, ks, st); break;
    }
}

// Measured GEMM choices per device: (n, k, power-of-two token bucket, glu) -> (candidate, split-K).
using TuneKey = std::tuple<int, int, int, bool>;
using TuneMap = std::map<TuneKey, std::pair<int, int>>;
thread_local std::map<int, TuneMap> tl_tuned;  // device ordinal -> choices (this thread's copy)
thread_local bool tl_tune_defer = false;       // gemm_tune_defer: save once at the end of a batch
bool gemm_tune_on() {
    static const bool on = !getenv("QE_GEMM_TUNE") || atoi(getenv("QE_GEMM_TUNE"));
    return on;
}

// Persisted tuning table: the timed choice differs between processes
// (noise), so does the rounding of everything downstream. The first process that tunes a key writes
// it to a small text file; later processes use the stored choice without timing (bit-identical
// results across restarts) and only tune keys the file lacks. The header fingerprints everything
// the choice depends on (build of this file, candidate list, CUDA runtime / driver, model, the
// environment variables that change the GEMM kernels); a device section is matched by its PCI bus
// id, name and SM count. Any mismatch or a malformed file: ignored and rewritten. Written via a
// temporary file + rename (a reader sees the old or the new file, never a half-written one).
// Only the engine enables it (gemm_tune_set_model); QE_TUNE_FILE=path overrides the location
// (default ~/.cache/qengine/gemm_tune-<model dir name>-<hash of the setup>.txt, the newest
// kTuneKeep of a model kept), QE_TUNE_FILE=off disables it.
struct TuneStore {
    std::mutex mu;
    bool on = false, loaded = false, dirty = false, warned = false, pruned = false, retune = false;
    std::string path, header, dir, prefix;  // dir, prefix: default location (tune_prune)
    std::map<std::string, TuneMap> devs;  // device line -> choices
};
TuneStore& tune_store() {
    static TuneStore s;
    return s;
}
const char* const kTuneEnv[] = {"QE_GEMM_H16", "QE_GEMM_SK_BLOCKS", "QE_GEMM_SK_WORKERS", "QE_I8_GLUQ",
                                "QE_I8_GRP", "QE_PREFILL_INT8", "QE_PREFILL_INT8_DIRECT", "QE_PREFILL_INT8_FUSE",
                                "QE_PREFILL_INT8_G", "QE_PREFILL_INT8_KERNEL", "QE_SK_MAXPARTS"};
std::string tune_dev_line(int dev) {
    char bus[32] = "?";
    cudaDeviceGetPCIBusId(bus, sizeof(bus), dev);
    cudaDeviceProp p;
    if (cudaGetDeviceProperties(&p, dev) != cudaSuccess) { cudaGetLastError(); return std::string("device ") + bus; }
    char b[256];
    snprintf(b, sizeof(b), "device %s %s sm%d cc%d.%d", bus, p.name, p.multiProcessorCount, p.major, p.minor);
    return b;
}
// Parses path; false (message on stderr) unless the header matches and the file is complete.
bool tune_read(const TuneStore& s, std::map<std::string, TuneMap>& out, bool verbose) {
    FILE* f = fopen(s.path.c_str(), "r");
    if (!f) return false;
    std::string hdr, line, dev;
    bool ok = false, in_hdr = true;
    char buf[1024];
    std::map<std::string, TuneMap> m;
    while (fgets(buf, sizeof(buf), f)) {
        line = buf;
        if (!line.empty() && line.back() == '\n') line.pop_back();
        if (line == "end") { ok = true; break; }
        if (line.rfind("device ", 0) == 0) {
            if (in_hdr && hdr != s.header) {
                if (verbose) fprintf(stderr, "gemm tune: %s is for another build / model / setup, retuning\n", s.path.c_str());
                fclose(f);
                return false;
            }
            in_hdr = false;
            dev = line;
            m[dev];
            continue;
        }
        if (in_hdr) { hdr += line + "\n"; continue; }
        int n, k, b, g, ci, ks;
        if (sscanf(line.c_str(), "%d %d %d %d %d %d", &n, &k, &b, &g, &ci, &ks) != 6 || ci < 0 || ci > 3 ||
            ks < 1 || ks > 8)
            break;
        m[dev][std::make_tuple(n, k, b, g != 0)] = std::make_pair(ci, ks);
    }
    fclose(f);
    if (ok && in_hdr && hdr != s.header) ok = false;  // header only, another setup
    if (!ok) {
        if (verbose) fprintf(stderr, "gemm tune: %s is malformed or truncated, ignored\n", s.path.c_str());
        return false;
    }
    out = std::move(m);
    return true;
}
void tune_load(TuneStore& s) {  // under s.mu
    if (s.loaded || !s.on) return;
    s.loaded = true;
    if (getenv("QE_TUNE_RETUNE") && atoi(getenv("QE_TUNE_RETUNE"))) {  // tune again, overwrite the file
        fprintf(stderr, "gemm tune: QE_TUNE_RETUNE, ignoring %s\n", s.path.c_str());
        s.retune = true;
        return;
    }
    if (tune_read(s, s.devs, true)) {
        size_t n = 0;
        for (auto& d : s.devs) n += d.second.size();
        fprintf(stderr, "gemm tune: %zu stored choices (%zu devices) from %s\n", n, s.devs.size(), s.path.c_str());
    }
}
constexpr int kTuneKeep = 8;
// Default location: keep the newest kTuneKeep tables of this model (by modification time), so
// setups that are no longer used (old worktrees, env experiments) do not pile up.
void tune_prune(TuneStore& s) {
    if (s.pruned || s.prefix.empty()) return;
    s.pruned = true;
    std::vector<std::pair<time_t, std::string>> files;
    if (DIR* d = opendir(s.dir.c_str())) {
        while (dirent* e = readdir(d)) {
            const std::string nm = e->d_name, p = s.dir + "/" + nm;
            if (nm.rfind(s.prefix, 0) != 0 || p == s.path) continue;
            struct stat st;
            if (stat(p.c_str(), &st) == 0 && S_ISREG(st.st_mode)) files.emplace_back(st.st_mtime, p);
        }
        closedir(d);
    }
    std::sort(files.begin(), files.end(), std::greater<>());
    for (size_t i = kTuneKeep - 1; i < files.size(); i++) unlink(files[i].second.c_str());
}
void tune_save(TuneStore& s) {  // under s.mu
    if (!s.on || !s.dirty) return;
    s.dirty = false;
    std::map<std::string, TuneMap> disk;  // another process may have added keys / devices meanwhile
    if (!s.retune && tune_read(s, disk, false))
        for (auto& d : disk)
            for (auto& e : d.second) s.devs[d.first].emplace(e.first, e.second);  // ours win
    const size_t sl = s.path.rfind('/');
    if (sl != std::string::npos && sl > 0) {  // mkdir -p of the parent
        const std::string dir = s.path.substr(0, sl);
        for (size_t p = 1; p != std::string::npos; p = dir.find('/', p + 1)) mkdir(dir.substr(0, p).c_str(), 0755);
        mkdir(dir.c_str(), 0755);
    }
    tune_prune(s);
    char tmp[64];
    snprintf(tmp, sizeof(tmp), ".tmp.%d", (int)getpid());
    const std::string tp = s.path + tmp;
    FILE* f = fopen(tp.c_str(), "w");
    if (!f) {
        if (!s.warned) fprintf(stderr, "gemm tune: cannot write %s (%s), not persisted\n", tp.c_str(), strerror(errno));
        s.warned = true;
        return;
    }
    fputs(s.header.c_str(), f);
    for (auto& d : s.devs) {
        fprintf(f, "%s\n", d.first.c_str());
        for (auto& e : d.second)
            fprintf(f, "%d %d %d %d %d %d\n", std::get<0>(e.first), std::get<1>(e.first), std::get<2>(e.first),
                    (int)std::get<3>(e.first), e.second.first, e.second.second);
    }
    fputs("end\n", f);
    const bool ok = fflush(f) == 0 && fsync(fileno(f)) == 0;
    fclose(f);
    if (!ok || rename(tp.c_str(), s.path.c_str()) != 0) {
        if (!s.warned) fprintf(stderr, "gemm tune: cannot write %s (%s), not persisted\n", s.path.c_str(), strerror(errno));
        s.warned = true;
        unlink(tp.c_str());
    }
}
// This thread's choices for the current device (filled from the file on first use).
TuneMap& tune_map() {
    int dev = 0;
    cudaGetDevice(&dev);
    auto it = tl_tuned.find(dev);
    if (it != tl_tuned.end()) return it->second;
    TuneMap& m = tl_tuned[dev];
    TuneStore& s = tune_store();
    std::lock_guard<std::mutex> l(s.mu);
    if (s.on) {
        tune_load(s);
        auto d = s.devs.find(tune_dev_line(dev));
        if (d != s.devs.end()) m = d->second;
    }
    return m;
}
void tune_record(const TuneKey& key, std::pair<int, int> v) {
    TuneStore& s = tune_store();
    std::lock_guard<std::mutex> l(s.mu);
    if (!s.on) return;
    int dev = 0;
    cudaGetDevice(&dev);
    s.devs[tune_dev_line(dev)][key] = v;
    s.dirty = true;
    if (!tl_tune_defer) tune_save(s);
}

}  // namespace

void gemm_set_out_bf16(bool on) { tl_out16 = on; }

void gemm_tune_set_model(const std::string& model) {
    TuneStore& s = tune_store();
    std::lock_guard<std::mutex> l(s.mu);
    const char* f = getenv("QE_TUNE_FILE");
    if (f && (!strcmp(f, "off") || !strcmp(f, "0") || !*f)) return;
    int rt = 0, drv = 0;
    cudaRuntimeGetVersion(&rt);
    cudaDriverGetVersion(&drv);
    // setup: everything but the build time (it names the file, so tests / servers with another env,
    // MTP head or worktree keep their own tables instead of overwriting one another)
    std::string setup = "build " __FILE__ "\n";
    setup += "cands 32x1:0.80 64x1:0.93 128x1:1.0 128x2:0.96 ks1,2,4 bucket64 tuner median-rr warmup1 ab3% reps" +
             std::to_string(std::max(1, env_int("QE_GEMM_TUNE_REPS", 7))) + "\n";
    setup += "cuda runtime " + std::to_string(rt) + " driver " + std::to_string(drv) + "\n";
    setup += "model " + model + "\n";
    setup += "env";
    for (const char* e : kTuneEnv) setup += std::string(" ") + e + "=" + (getenv(e) ? getenv(e) : "-");
    setup += "\n";
    // a rebuilt gemm.cu (kernels may have changed): same file, retuned
    s.header = "qengine-gemm-tune 1\nbuilt " __DATE__ " " __TIME__ "\n" + setup;
    if (f) s.path = f;
    else {
        const char* xdg = getenv("XDG_CACHE_HOME");
        const char* home = getenv("HOME");
        if (xdg && *xdg) s.dir = std::string(xdg) + "/qengine";
        else if (home && *home) s.dir = std::string(home) + "/.cache/qengine";
        else return;
        std::string m = model.substr(0, model.find('|'));
        while (m.size() > 1 && m.back() == '/') m.pop_back();
        std::string base = m.substr(m.rfind('/') + 1);
        for (char& c : base)
            if (!isalnum((unsigned char)c) && c != '-' && c != '_' && c != '.') c = '_';
        unsigned long long h = 1469598103934665603ull;  // FNV-1a
        for (unsigned char c : setup) h = (h ^ c) * 1099511628211ull;
        char hx[16];
        snprintf(hx, sizeof(hx), "%08llx", (h ^ (h >> 32)) & 0xffffffffull);
        s.prefix = "gemm_tune-" + (base.empty() ? std::string("model") : base) + "-";
        s.path = s.dir + "/" + s.prefix + hx + ".txt";
    }
    s.on = true;
    s.loaded = false;
}

void gemm_tune_defer(bool on) {
    tl_tune_defer = on;
    if (on) return;
    TuneStore& s = tune_store();
    std::lock_guard<std::mutex> l(s.mu);
    tune_save(s);
}
void gemm_set_sk_maxparts(int n) { g_sk_maxparts_rt = n; }

void gemm_w4(const W4& w, const bf16* x, int ldx, float* c, int ldc, int t, cudaStream_t st) {
    CHECK(w.n % 16 == 0 && w.k % 128 == 0, "gemm_w4 %d %d", w.n, w.k);
    if (tl_out16 && !w.glu && !(i8_use(w) && g_i8_kernel >= 2)) {  // path without a bf16 epilogue
        const size_t need = (size_t)t * w.n;
        if (need > tl_o16_buf.n) {
            if (tl_o16_buf.p) CUDA_CHECK(cudaFree(tl_o16_buf.p));
            CUDA_CHECK(cudaMalloc(&tl_o16_buf.p, need * sizeof(float)));
            tl_o16_buf.n = need;
        }
        tl_out16 = false;
        gemm_w4(w, x, ldx, tl_o16_buf.p, w.n, t, st);
        tl_out16 = true;
        f32_to_bf16_kernel<<<(need + 255) / 256, 256, 0, st>>>(tl_o16_buf.p, w.n, t, reinterpret_cast<bf16*>(c), ldc);
        return;
    }
    // choose token tile, warp layout and split-K count by the cost model
    struct Cand { int tb, wt; double eff; };
    static const Cand cands[] = {{32, 1, 0.80}, {64, 1, 0.93}, {128, 1, 1.0}, {128, 2, 0.96}};
    const int nk = w.k / BK;
    int bi = 3, bks = 1;
    double best = 1e30;
    for (int ci = 0; ci < 4; ci++) {
        const Cand& cd = cands[ci];
        if (cd.tb < 128 && t > 4 * cd.tb) continue;
        const int rows = block_rows<1>() / cd.wt;
        for (int ks = 1; ks <= std::min(8, nk / 4); ks++) {
            if (t >= 1024 && ks > 1) break;  // big chunks: plenty of blocks already
            const double cst = gemm_cost(w.n, w.k, t, cd.tb, rows, ks, cd.eff);
            if (cst < best) { best = cst; bi = ci; bks = ks; }
        }
    }
    if (const char* f = getenv("QE_GEMM_FORCE")) {  // "ci,ks" (experiments)
        int fc = 0, fk = 1;
        if (sscanf(f, "%d,%d", &fc, &fk) >= 1) { bi = fc; bks = std::max(1, fk); }
    } else if (gemm_tune_on()) {
        // measured choice per (shape, power-of-two token bucket): the cost model above is only
        // the fallback (it misjudges small token tiles with the fp16-accumulate kernel by up to
        // ~25%); candidates are timed once, outside graph capture
        int bucket = 64;
        while (bucket < t) bucket *= 2;
        const auto key = std::make_tuple(w.n, w.k, bucket, w.glu);
        TuneMap& tuned = tune_map();
        auto it = tuned.find(key);
        cudaStreamCaptureStatus cs = cudaStreamCaptureStatusNone;
        cudaStreamIsCapturing(st, &cs);
        if (it == tuned.end() && cs == cudaStreamCaptureStatusNone) {
            cudaEvent_t e0, e1;
            CUDA_CHECK(cudaEventCreate(&e0));
            CUDA_CHECK(cudaEventCreate(&e1));
            float bt = 1e30f;
            int tb_i = bi, tb_k = bks;
            // every timed run consumes the pending prequantized input (tl_prequant): hand it to
            // each run again, else the later runs - the final one included - quantize x itself,
            // whose bf16 rows the producer may not have written (int8-only all-reduce tail):
            // the first forward of each new shape then ran on stale activations. The stream-K
            // counters it zeroed are used up after the first run (gemm_i8_cfg memsets them).
            const PreQuant pq0 = tl_prequant;
            bool ran = false;
            auto restore_pq = [&] {
                tl_prequant = pq0;
                if (ran) tl_prequant.zeroed = false;
                ran = true;
            };
            // candidates: tile ci x split-K ks. Timing: one untimed run of each (clocks / caches),
            // then rounds of one timed run per candidate in rotating order (drifting clocks hit all
            // alike), median per candidate (QE_GEMM_TUNE_REPS rounds, default 7; 2-run minimum
            // before the tuning-table fix); a runner-up within 3% gets a second A/B of the two
            std::vector<std::pair<int, int>> cl;
            for (int ci = 0; ci < 4; ci++) {
                if (cands[ci].tb < 128 && t > 16 * cands[ci].tb) continue;  // (wider than the model)
                for (int ks : {1, 2, 4}) {
                    if (ks > 1 && (ks > nk / 4 || t >= 1024 || i8_use(w))) continue;
                    cl.emplace_back(ci, ks);
                }
            }
            static const int reps = std::max(1, env_int("QE_GEMM_TUNE_REPS", 7));
            auto time1 = [&](const std::pair<int, int>& cd) {
                CUDA_CHECK(cudaEventRecord(e0, st));
                restore_pq();
                gemm_run(cd.first, w, x, ldx, c, ldc, t, cd.second, st);
                CUDA_CHECK(cudaEventRecord(e1, st));
                CUDA_CHECK(cudaEventSynchronize(e1));
                float ms;
                CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
                return ms;
            };
            auto median = [](std::vector<float> v) {
                std::sort(v.begin(), v.end());
                return v[v.size() / 2];
            };
            auto measure = [&](const std::vector<int>& idx, int rounds, std::vector<float>& med) {
                std::vector<std::vector<float>> ts(idx.size());
                for (int r = 0; r < rounds; r++)
                    for (size_t j = 0; j < idx.size(); j++) {
                        const size_t q = (j + r) % idx.size();
                        ts[q].push_back(time1(cl[idx[q]]));
                    }
                med.resize(idx.size());
                for (size_t j = 0; j < idx.size(); j++) med[j] = median(ts[j]);
            };
            for (auto& cd : cl) time1(cd);  // warm-up
            std::vector<int> all(cl.size());
            for (size_t j = 0; j < cl.size(); j++) all[j] = j;
            std::vector<float> med;
            measure(all, reps, med);
            int b1 = 0, b2 = -1;
            for (size_t j = 1; j < cl.size(); j++)
                if (med[j] < med[b1]) b1 = j;
            for (size_t j = 0; j < cl.size(); j++)
                if ((int)j != b1 && (b2 < 0 || med[j] < med[b2])) b2 = j;
            const bool dbg2 = getenv("QE_GEMM_DEBUG") && atoi(getenv("QE_GEMM_DEBUG")) >= 2;
            if (dbg2) {
                int dev = 0;
                cudaGetDevice(&dev);
                fprintf(stderr, "gemm tune dev %d n=%d k=%d t=%d:", dev, w.n, w.k, t);
                for (size_t j = 0; j < cl.size(); j++) fprintf(stderr, " %d/%d %.1f", cl[j].first, cl[j].second, med[j] * 1e3);
                fprintf(stderr, "\n");
            }
            if (b2 >= 0 && med[b2] < med[b1] * 1.03f) {  // close: A/B of the two
                std::vector<float> m2;
                measure({b1, b2}, 2 * reps, m2);
                if (dbg2) fprintf(stderr, "gemm tune   A/B %d/%d %.1f vs %d/%d %.1f\n", cl[b1].first, cl[b1].second,
                                  m2[0] * 1e3, cl[b2].first, cl[b2].second, m2[1] * 1e3);
                if (m2[1] < m2[0]) std::swap(b1, b2), med[b1] = m2[1];
                else med[b1] = m2[0];
            }
            tb_i = cl[b1].first;
            tb_k = cl[b1].second;
            bt = med[b1];
            CUDA_CHECK(cudaEventDestroy(e0));
            CUDA_CHECK(cudaEventDestroy(e1));
            restore_pq();  // for the run below
            it = tuned.emplace(key, std::make_pair(tb_i, tb_k)).first;
            tune_record(key, it->second);
            if (getenv("QE_GEMM_DEBUG"))
                fprintf(stderr, "gemm tune n=%d k=%d t<=%d -> cand %d ks %d (%.1f us)\n", w.n, w.k,
                        bucket, tb_i, tb_k, bt * 1e3);
        }
        if (it != tuned.end()) { bi = it->second.first; bks = it->second.second; }
    }
    if (getenv("QE_GEMM_DEBUG"))
        fprintf(stderr, "gemm n=%d k=%d t=%d -> tb=%d wt=%d ks=%d (%.0f us est)\n", w.n, w.k, t,
                cands[bi].tb, cands[bi].wt, bks, best * 1e6);
    gemm_run(bi, w, x, ldx, c, ldc, t, bks, st);
}

// experiments: explicit W4A8 configuration (group G, candidate ci as in gemm_run, kernel
// variant var: 0 = gemm_i8_kernel, 1 = gemm_i8t_kernel)
void gemm_w4_i8(const W4& w, const bf16* x, int ldx, float* c, int ldc, int t, cudaStream_t st,
                int G, int ci, int var) {
    CHECK(w.n % 16 == 0 && w.k % 128 == 0, "gemm_w4_i8 %d %d", w.n, w.k);
    gemm_i8_dispatch(var, ci, G, w, x, ldx, c, ldc, t, st);
}

int gemm_i8_variant() { return g_i8_kernel; }

// The int8 group the W4A8 GEMM uses for an input with k columns that feeds the MLP gate_up
// (glu_next) or a mixer projection, 0 when that class does not run W4A8 / fusion is off.
int gemm_i8_prequant_group(int k, bool glu_next) {
    if (!g_i8_fuse || k % 128 != 0 || (k / 4) % 32 != 0) return 0;
    const int cls = glu_next ? I8_GU : I8_IN;
    if (!(g_i8_mask & cls)) return 0;
    return g_i8_group[cls];
}
// registers rows quantized elsewhere (layouts of x_to_i8_kernel) as the input of the next W4A8
// GEMM with this x, t, k, G; its stream-K counter is zeroed by the caller (gemm_i8_sk_ctr)
void gemm_i8_prequant_ext(const bf16* x, int t, int k, int G, int8_t* xq, float* xsc) {
    tl_prequant = PreQuant{x, t, k, G, xq, xsc, true, true};
}
// the producer of an out_proj input (class OUT, k = 2048: dn_out) quantizes it: own buffers,
// allocated once at the largest size (never re-allocated in the middle of a prefill)
thread_local TlBuf<int8_t> tl_po_xq_buf;
thread_local TlBuf<float> tl_po_xs_buf;
bool gemm_i8_prequant_out(const bf16* x, int t, int k, int row0, int8_t** xq, float** xsc, int* G) {
    if (!g_i8_fuse || g_i8_kernel < 1 || k != 2048 || !(g_i8_mask & I8_OUT) || !env_int("QE_DNO_I8", 0)) return false;
    const bool v2 = g_i8_kernel >= 2;
    const int g = g_i8_group[I8_OUT];
    if (g != 32 && g != 64 && g != 128) return false;
    const size_t need = (size_t)kChunk * k, needs = (size_t)(kChunk + 32) * i8v_rec_floats(k, 32, false);
    if (!tl_po_xq_buf.p) {
        CUDA_CHECK(cudaMalloc(&tl_po_xq_buf.p, need));
        tl_po_xq_buf.n = need;
        CUDA_CHECK(cudaMalloc(&tl_po_xs_buf.p, needs * 4));
        tl_po_xs_buf.n = needs;
    }
    CHECK(row0 >= 0 && row0 + t <= kChunk, "prequant out rows %d + %d", row0, t);
    int8_t* q = tl_po_xq_buf.p + (size_t)row0 * k;
    float* sc = tl_po_xs_buf.p + (v2 ? (size_t)(row0 + 15) / 16 * 16 * i8v_rec_floats(k, g, false)
                                     : (size_t)row0 * i8_rec_floats(k, g));
    tl_prequant = PreQuant{x, t, k, g, q, sc, false, true};
    *xq = q;
    *xsc = sc;
    *G = g;
    return true;
}
int* gemm_i8_sk_ctr() {
    if (!tl_sk_ctr_buf.p) {
        CUDA_CHECK(cudaMalloc(&tl_sk_ctr_buf.p, kSkCtrs * sizeof(int)));
        tl_sk_ctr_buf.n = kSkCtrs;
    }
    return tl_sk_ctr_buf.p;
}
// QE_I8_CTR_FUSE=0: the stream-K counter of a prequantized GEMM is zeroed by a memset again
bool gemm_i8_prequant(const bf16* x, int t, int k, int row0, bool glu_next, int8_t** xq, float** xsc, int* G,
                      int** ctr) {
    if (!g_i8_fuse || k % 128 != 0 || (k / 4) % 32 != 0) return false;
    const int cls = glu_next ? I8_GU : I8_IN;
    if (!(g_i8_mask & cls)) return false;
    const int g = g_i8_group[cls];
    // gemm_i8v_kernel records come in 16-token tiles: each micro-batch starts on a tile boundary
    const bool vf = g_i8_kernel >= 2;
    const size_t need = (size_t)kChunk * k,
                 needs = vf ? (size_t)(kChunk + 32) * i8v_rec_floats(k, g, false) : (size_t)kChunk * i8_rec_floats(k, g);
    if (need > tl_pq_xq_buf.n) {
        if (tl_pq_xq_buf.p) CUDA_CHECK(cudaFree(tl_pq_xq_buf.p));
        CUDA_CHECK(cudaMalloc(&tl_pq_xq_buf.p, need));
        tl_pq_xq_buf.n = need;
    }
    if (needs > tl_pq_xs_buf.n) {
        if (tl_pq_xs_buf.p) CUDA_CHECK(cudaFree(tl_pq_xs_buf.p));
        CUDA_CHECK(cudaMalloc(&tl_pq_xs_buf.p, needs * 4));
        tl_pq_xs_buf.n = needs;
    }
    CHECK(row0 >= 0 && row0 + t <= kChunk, "prequant rows %d + %d", row0, t);
    // the two micro-batches of a chunk keep their own rows of the buffers (row0 = chunk row)
    int8_t* q = tl_pq_xq_buf.p + (size_t)row0 * k;
    float* sc = tl_pq_xs_buf.p + (vf ? (size_t)(row0 + 15) / 16 * 16 * i8v_rec_floats(k, g, false) : (size_t)row0 * i8_rec_floats(k, g));
    const bool zero = ctr && env_int("QE_I8_CTR_FUSE", 1) != 0;
    if (zero) {
        if (!tl_sk_ctr_buf.p) {
            CUDA_CHECK(cudaMalloc(&tl_sk_ctr_buf.p, kSkCtrs * sizeof(int)));
            tl_sk_ctr_buf.n = kSkCtrs;
        }
        *ctr = tl_sk_ctr_buf.p;
    } else if (ctr) {
        *ctr = nullptr;
    }
    tl_prequant = PreQuant{x, t, k, g, q, sc, zero};
    *xq = q;
    *xsc = sc;
    *G = g;
    return true;
}

void gemm_w16(const W16& w, const bf16* x, int ldx, float* c, int ldc, int t, cudaStream_t st,
              int ksplit, size_t cstride) {
    CHECK(w.n % 16 == 0 && w.k % BK == 0 && w.k / BK >= ksplit, "gemm_w16 %d %d", w.n, w.k);
    dim3 grid((w.n + block_rows<2>() - 1) / block_rows<2>(), (t + BT - 1) / BT, ksplit);
    gemm_kernel<false, 2><<<grid, kThreads, 0, st>>>(w.w, nullptr, w.n, w.k, x, ldx, t, c, ldc,
                                                     cstride, false);
}
