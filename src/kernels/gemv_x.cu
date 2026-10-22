// Wide fp16-accumulate INT4 GEMV for batched decode steps with up to 64 token rows (gemv_w4hx).
//
// gemv_w4hw keeps one 16-row weight tile per block and every warp reads its token rows of x from L2 per
// chunk; at 32+ rows those re-reads (tiles x m x K x 2 bytes) and the per-chunk weight latency bound it,
// and at > 32 rows its registers do not fit. Here a block is TR warps = TR consecutive tiles on ONE K
// slice: the slice's x chunk (m rows x 128 bytes) is staged once per block in shared memory (cp.async,
// S stages) and shared by the TR warps, weights stream to registers with a 2-chunk prefetch ring. The nw
// slices of a tile group are adjacent blocks; each writes its fp32 partials to a workspace and the last
// one to arrive (atomic counter per group) adds them in slice order 0..nw-1 and stores the result.
//
// Numerics are those of every H16 kernel (gemv_h16.cuh): same slices (pick_warps, shape only), same
// per-chunk fp16 mma chain and xh_flush, slices added in the same order as reduce_store_nt -> bit-identical
// to gemv_w4h / gemv_w4hw / skinny_w4h for every m (batch == alone).
#include "gemv.cuh"
#include "gemv_h16.cuh"

namespace {

__device__ __forceinline__ uint4 ldg_stream(const uint4* p) {
    uint4 v;
    asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                 : "l"(p));
    return v;
}
__device__ __forceinline__ void cp_async16(unsigned smem, const void* gmem) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(smem), "l"(gmem));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;"); }
template <int N>
__device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;" ::"n"(N)); }
__device__ __forceinline__ float silu_f(float x) { return x / (1.f + __expf(-x)); }

constexpr int kRowB = 144;  // 128 bytes of a chunk row + 16 padding (conflict-free 32-byte lane reads)
constexpr int kMaxTR = 16;

template <int NT, int S, int ABL = 0>  // ABL (microbench ablation): 1 no mma, 2 no smem x reads, 4 no weight loads
__global__ void __launch_bounds__(512) gemv_w4hx_kernel(
    const uint4* __restrict__ q, const bf16* __restrict__ s, int K, const __half* __restrict__ xh, int ldx,
    float* __restrict__ y, int ldy, int m, bool glu, __half* __restrict__ yh, int ldyh, int tiles, int nw,
    float* __restrict__ red, unsigned* __restrict__ ctr) {
    extern __shared__ __align__(16) unsigned char smem[];
    __shared__ int s_last;
    const unsigned smem_u = (unsigned)__cvta_generic_to_shared(smem);
    const int TR = blockDim.x >> 5;
    const int tid = threadIdx.x, lane = tid & 31, wid = tid >> 5;
    const int g = lane >> 2, t = lane & 3;
    const int grp = blockIdx.x / nw, w = blockIdx.x - grp * nw;
    const int tile = grp * TR + wid;
    const bool valid = tile < tiles;
    const int groups = K >> 7, chunks = K >> 6;
    const int gb = w * groups / nw, ge = (w + 1) * groups / nw;
    const int L = 2 * (ge - gb);  // chunks of this slice
    constexpr int rows = 8 * NT;
    constexpr int stage_bytes = rows * kRowB;

    // rows >= m stay zero (their columns of the mma are discarded)
    for (int i = m * kRowB + tid * 16; i < stage_bytes; i += blockDim.x * 16)
#pragma unroll
        for (int st = 0; st < S; st++) *reinterpret_cast<uint4*>(smem + st * stage_bytes + i) = make_uint4(0, 0, 0, 0);

    const int pieces = m * 8;
    auto issue = [&](int i, int st) {
        const __half* src = xh + (size_t)(2 * gb + i) * 64;
        for (int p = tid; p < pieces; p += blockDim.x) {
            const int r = p >> 3, c = p & 7;
            cp_async16(smem_u + st * stage_bytes + r * kRowB + c * 16, src + (size_t)r * ldx + c * 8);
        }
    };
#pragma unroll
    for (int i = 0; i < S - 1; i++) {
        if (i < L) issue(i, i);
        cp_async_commit();
    }

    const uint4* qp = q + ((size_t)(valid ? tile : 0) * chunks + 2 * gb) * 32 + lane;
    const bf16* sp = s + ((size_t)(valid ? tile : 0) * groups + gb) * 16 + g;
    uint4 wb0 = make_uint4(0, 0, 0, 0), wb1 = wb0;
    if (!(ABL & 4) && valid && L > 0) wb0 = ldg_stream(qp);
    if (!(ABL & 4) && valid && L > 1) wb1 = ldg_stream(qp + 32);
    float sg = 0.f, sg8 = 0.f;
    float acc[NT][4];
#pragma unroll
    for (int j = 0; j < NT; j++) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.f;
    const unsigned xoff = g * kRowB + t * 32;

    for (int i = 0; i < L; i++) {
        cp_async_wait<S - 2>();
        __syncthreads();
        if (i + S - 1 < L) issue(i + S - 1, (i + S - 1) % S);
        cp_async_commit();
        if (!valid) continue;
        const uint4 wv = (i & 1) ? wb1 : wb0;
        if (!(ABL & 4) && i + 2 < L) {
            if (i & 1) wb1 = ldg_stream(qp + (size_t)(i + 2) * 32);
            else wb0 = ldg_stream(qp + (size_t)(i + 2) * 32);
        }
        if ((i & 1) == 0) {
            sg = xh_scale(sp[(i >> 1) * 16]);
            sg8 = xh_scale(sp[(i >> 1) * 16 + 8]);
        }
        const unsigned xs = smem_u + (i % S) * stage_bytes + xoff;
        const uint32_t words[4] = {wv.x, wv.y, wv.z, wv.w};
        uint32_t h[NT][2];
#pragma unroll
        for (int nt = 0; nt < NT; nt++) h[nt][0] = h[nt][1] = 0u;
#pragma unroll
        for (int hp = 0; hp < 2; hp++) {
            uint4 b[NT];
#pragma unroll
            for (int nt = 0; nt < NT; nt++) {
                if (ABL & 2) b[nt] = make_uint4(xs + nt, hp, i, 0);
                else asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];"
                             : "=r"(b[nt].x), "=r"(b[nt].y), "=r"(b[nt].z), "=r"(b[nt].w)
                             : "r"(xs + nt * 8 * kRowB + hp * 16));
            }
#pragma unroll
            for (int jj = 0; jj < 2; jj++) {
                uint32_t a[4];
                xh_dequant(words[2 * hp + jj], a);
#pragma unroll
                for (int nt = 0; nt < NT; nt++) {
                    if (ABL & 1) { h[nt][0] ^= a[0] ^ (jj ? b[nt].z : b[nt].x); h[nt][1] ^= a[1] ^ (jj ? b[nt].w : b[nt].y); }
                    else xh_mma(h[nt], a, jj ? b[nt].z : b[nt].x, jj ? b[nt].w : b[nt].y);
                }
            }
        }
#pragma unroll
        for (int nt = 0; nt < NT; nt++) xh_flush(acc[nt], h[nt], sg, sg8);
    }
    cp_async_wait<0>();

    // partials [nw][tiles][NT][4][32]; the last slice block of the group adds them in order 0..nw-1
    if (nw > 1 && !(ABL & 8)) {
        if (valid)
#pragma unroll
            for (int j = 0; j < NT; j++)
#pragma unroll
                for (int c = 0; c < 4; c++)
                    __stcg(&red[((((size_t)w * tiles + tile) * NT + j) * 4 + c) * 32 + lane], acc[j][c]);
        __threadfence();
        __syncthreads();
        if (tid == 0) s_last = atomicAdd(&ctr[grp], 1u) == (unsigned)(nw - 1);
        __syncthreads();
        if (!s_last) return;
        __threadfence();
        if (tid == 0) ctr[grp] = 0u;
        if (!valid) return;
#pragma unroll
        for (int j = 0; j < NT; j++)
#pragma unroll
            for (int c = 0; c < 4; c++) acc[j][c] = __ldcg(&red[((((size_t)0 * tiles + tile) * NT + j) * 4 + c) * 32 + lane]);
        for (int ww = 1; ww < nw; ww++)
#pragma unroll
            for (int j = 0; j < NT; j++)
#pragma unroll
                for (int c = 0; c < 4; c++)
                    acc[j][c] += __ldcg(&red[((((size_t)ww * tiles + tile) * NT + j) * 4 + c) * 32 + lane]);
    } else if (!valid) {
        return;
    }
    if ((ABL & 8) && w != 0) return;
    const int row0 = tile * 16;
#pragma unroll
    for (int j = 0; j < NT; j++) {
        const int n0 = 8 * j + 2 * t, n1 = n0 + 1;
        if (glu) {
            bf16* act = reinterpret_cast<bf16*>(y);
            const int col = row0 / 2 + g;
            const float v0 = silu_f(acc[j][0]) * acc[j][2], v1 = silu_f(acc[j][1]) * acc[j][3];
            if (n0 < m) act[(size_t)n0 * ldy + col] = __float2bfloat16(v0);
            if (n1 < m) act[(size_t)n1 * ldy + col] = __float2bfloat16(v1);
            if (yh) {
                if (n0 < m) store_xh(yh + (size_t)n0 * ldyh, col, v0);
                if (n1 < m) store_xh(yh + (size_t)n1 * ldyh, col, v1);
            }
            continue;
        }
        if (n0 < m) {
            y[(size_t)n0 * ldy + row0 + g] = acc[j][0];
            y[(size_t)n0 * ldy + row0 + g + 8] = acc[j][2];
        }
        if (n1 < m) {
            y[(size_t)n1 * ldy + row0 + g] = acc[j][1];
            y[(size_t)n1 * ldy + row0 + g + 8] = acc[j][3];
        }
    }
}


// Variant 2 (QE_W4HX_MODE=1): no cross-block reduction. A block is TR warps = TR tiles over the WHOLE K; the
// x chunks are staged once per block for all TR warps; each warp runs its tile's slices one after another
// and adds them in slice order (acc_tot = s0, += s1, ...), exactly reduce_store_nt's order.
template <int NT, int S, int R>
__global__ void __launch_bounds__(256) gemv_w4hs_kernel(
    const uint4* __restrict__ q, const bf16* __restrict__ s, int K, const __half* __restrict__ xh, int ldx,
    float* __restrict__ y, int ldy, int m, bool glu, __half* __restrict__ yh, int ldyh, int tiles, int nw) {
    extern __shared__ __align__(16) unsigned char smem[];
    const unsigned smem_u = (unsigned)__cvta_generic_to_shared(smem);
    const int tid = threadIdx.x, lane = tid & 31, wid = tid >> 5;
    const int g = lane >> 2, t = lane & 3;
    const int TR = blockDim.x >> 5;
    const int tile = blockIdx.x * TR + wid;
    const bool valid = tile < tiles;
    const int groups = K >> 7, chunks = K >> 6;
    constexpr int stage_bytes = 8 * NT * kRowB;
    for (int i = m * kRowB + tid * 16; i < stage_bytes; i += blockDim.x * 16)
#pragma unroll
        for (int st = 0; st < S; st++) *reinterpret_cast<uint4*>(smem + st * stage_bytes + i) = make_uint4(0, 0, 0, 0);
    const int pieces = m * 8;
    auto issue = [&](int i, int st) {
        const __half* src = xh + (size_t)i * 64;
        for (int p = tid; p < pieces; p += blockDim.x) {
            const int r = p >> 3, c = p & 7;
            cp_async16(smem_u + st * stage_bytes + r * kRowB + c * 16, src + (size_t)r * ldx + c * 8);
        }
    };
#pragma unroll
    for (int i = 0; i < S - 1; i++) {
        if (i < chunks) issue(i, i);
        cp_async_commit();
    }
    const uint4* qp = q + (size_t)(valid ? tile : 0) * chunks * 32 + lane;
    const bf16* sp = s + (size_t)(valid ? tile : 0) * groups * 16 + g;
    uint4 wb[R];
#pragma unroll
    for (int d = 0; d < R; d++) wb[d] = valid && d < chunks ? ldg_stream(qp + d * 32) : make_uint4(0, 0, 0, 0);
    float acc[NT][4], tot[NT][4];
#pragma unroll
    for (int j = 0; j < NT; j++)
#pragma unroll
        for (int c = 0; c < 4; c++) acc[j][c] = tot[j][c] = 0.f;
    const unsigned xoff = g * kRowB + t * 32;
    int w = 0, wend = 2 * (groups / nw);  // chunk where slice w ends (exclusive)
    float sg = 0.f, sg8 = 0.f;
    for (int i0 = 0; i0 < chunks; i0 += R) {
#pragma unroll
        for (int d = 0; d < R; d++) {
            const int i = i0 + d;
            if (i >= chunks) break;
            cp_async_wait<S - 2>();
            __syncthreads();
            if (i + S - 1 < chunks) issue(i + S - 1, (i + S - 1) % S);
            cp_async_commit();
            if (!valid) continue;
            const uint4 wv = wb[d];
            if (i + R < chunks) wb[d] = ldg_stream(qp + (size_t)(i + R) * 32);
            if ((i & 1) == 0) {
                sg = xh_scale(sp[(i >> 1) * 16]);
                sg8 = xh_scale(sp[(i >> 1) * 16 + 8]);
            }
            const unsigned xs = smem_u + (i % S) * stage_bytes + xoff;
            const uint32_t words[4] = {wv.x, wv.y, wv.z, wv.w};
            uint32_t h[NT][2];
#pragma unroll
            for (int nt = 0; nt < NT; nt++) h[nt][0] = h[nt][1] = 0u;
#pragma unroll
            for (int hp = 0; hp < 2; hp++) {
                uint4 b[NT];
#pragma unroll
                for (int nt = 0; nt < NT; nt++)
                    asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];"
                                 : "=r"(b[nt].x), "=r"(b[nt].y), "=r"(b[nt].z), "=r"(b[nt].w)
                                 : "r"(xs + nt * 8 * kRowB + hp * 16));
#pragma unroll
                for (int jj = 0; jj < 2; jj++) {
                    uint32_t a[4];
                    xh_dequant(words[2 * hp + jj], a);
#pragma unroll
                    for (int nt = 0; nt < NT; nt++)
                        xh_mma(h[nt], a, jj ? b[nt].z : b[nt].x, jj ? b[nt].w : b[nt].y);
                }
            }
#pragma unroll
            for (int nt = 0; nt < NT; nt++) xh_flush(acc[nt], h[nt], sg, sg8);
            if (i + 1 == wend) {  // slice w done: add it in order
#pragma unroll
                for (int j = 0; j < NT; j++)
#pragma unroll
                    for (int c = 0; c < 4; c++) {
                        tot[j][c] = w == 0 ? acc[j][c] : tot[j][c] + acc[j][c];
                        acc[j][c] = 0.f;
                    }
                w++;
                wend = 2 * ((w + 1) * groups / nw);
            }
        }
    }
    cp_async_wait<0>();
    if (!valid) return;
    const int row0 = tile * 16;
#pragma unroll
    for (int j = 0; j < NT; j++) {
        const int n0 = 8 * j + 2 * t, n1 = n0 + 1;
        if (glu) {
            bf16* act = reinterpret_cast<bf16*>(y);
            const int col = row0 / 2 + g;
            const float v0 = silu_f(tot[j][0]) * tot[j][2], v1 = silu_f(tot[j][1]) * tot[j][3];
            if (n0 < m) act[(size_t)n0 * ldy + col] = __float2bfloat16(v0);
            if (n1 < m) act[(size_t)n1 * ldy + col] = __float2bfloat16(v1);
            if (yh) {
                if (n0 < m) store_xh(yh + (size_t)n0 * ldyh, col, v0);
                if (n1 < m) store_xh(yh + (size_t)n1 * ldyh, col, v1);
            }
            continue;
        }
        if (n0 < m) {
            y[(size_t)n0 * ldy + row0 + g] = tot[j][0];
            y[(size_t)n0 * ldy + row0 + g + 8] = tot[j][2];
        }
        if (n1 < m) {
            y[(size_t)n1 * ldy + row0 + g] = tot[j][1];
            y[(size_t)n1 * ldy + row0 + g + 8] = tot[j][3];
        }
    }
}

// Variant 3 (QE_W4HX_MODE=2): gemv_w4hw's structure (one block per 16-row tile, its nw warps = the K slices,
// x read straight from L2/L1 - co-resident blocks share it through L1) with up to 8 token tiles per warp:
// B fragments loaded per (k-step pair, token tile) right before their two mma, so the registers stay ~90.
template <int NT>
__global__ void __launch_bounds__(160, 4) gemv_w4hz_kernel(const uint4* __restrict__ q, const bf16* __restrict__ s,
                                                           int K, const __half* __restrict__ xh, int ldx,
                                                           float* __restrict__ y, int ldy, int m, bool glu,
                                                           __half* __restrict__ yh, int ldyh) {
    const int r = blockIdx.x, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int nwarps = blockDim.x >> 5;
    const int g = lane >> 2, t = lane & 3;
    const int groups = K >> 7, chunks = K >> 6;
    const int gb = warp * groups / nwarps, ge = (warp + 1) * groups / nwarps;
    const uint4* qp = q + (size_t)r * chunks * 32 + lane;
    const bf16* sp = s + (size_t)r * groups * 16;
    const __half* xb = xh + t * 16;
    float acc[NT][4];
#pragma unroll
    for (int j = 0; j < NT; j++) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.f;
    uint4 wn = gb < ge ? ldg_stream(qp + (size_t)(2 * gb) * 32) : make_uint4(0, 0, 0, 0);
    for (int gi = gb; gi < ge; gi++) {
        const float sg = xh_scale(sp[gi * 16 + g]), sg8 = xh_scale(sp[gi * 16 + g + 8]);
#pragma unroll
        for (int c = 0; c < 2; c++) {
            const int ch = 2 * gi + c;
            const uint4 w = wn;
            if (ch + 1 < 2 * ge) wn = ldg_stream(qp + (size_t)(ch + 1) * 32);
            const uint32_t words[4] = {w.x, w.y, w.z, w.w};
            uint32_t h[NT][2];
#pragma unroll
            for (int nt = 0; nt < NT; nt++) h[nt][0] = h[nt][1] = 0u;
#pragma unroll
            for (int hp = 0; hp < 2; hp++) {
                uint32_t a0[4], a1[4];
                xh_dequant(words[2 * hp], a0);
                xh_dequant(words[2 * hp + 1], a1);
#pragma unroll
                for (int nt = 0; nt < NT; nt++) {
                    const int tok = min(8 * nt + g, m - 1);
                    const uint4 bb = *reinterpret_cast<const uint4*>(xb + (size_t)tok * ldx + (size_t)ch * 64 + hp * 8);
                    xh_mma(h[nt], a0, bb.x, bb.y);
                    xh_mma(h[nt], a1, bb.z, bb.w);
                }
            }
#pragma unroll
            for (int nt = 0; nt < NT; nt++) xh_flush(acc[nt], h[nt], sg, sg8);
        }
    }
    // slice reduction in order (reduce_store_nt): warp 0 adds warps 1..nw-1
    __shared__ float red[5][4][32];
    const int row0 = r * 16;
#pragma unroll
    for (int j = 0; j < NT; j++) {
        if (nwarps > 1) {
            __syncthreads();
            if (warp != 0)
#pragma unroll
                for (int c = 0; c < 4; c++) red[warp][c][lane] = acc[j][c];
            __syncthreads();
            if (warp == 0)
                for (int ww = 1; ww < nwarps; ww++)
#pragma unroll
                    for (int c = 0; c < 4; c++) acc[j][c] += red[ww][c][lane];
        }
        if (warp != 0) continue;
        const int n0 = 8 * j + 2 * t, n1 = n0 + 1;
        if (glu) {
            bf16* act = reinterpret_cast<bf16*>(y);
            const int col = row0 / 2 + g;
            const float v0 = silu_f(acc[j][0]) * acc[j][2], v1 = silu_f(acc[j][1]) * acc[j][3];
            if (n0 < m) act[(size_t)n0 * ldy + col] = __float2bfloat16(v0);
            if (n1 < m) act[(size_t)n1 * ldy + col] = __float2bfloat16(v1);
            if (yh) {
                if (n0 < m) store_xh(yh + (size_t)n0 * ldyh, col, v0);
                if (n1 < m) store_xh(yh + (size_t)n1 * ldyh, col, v1);
            }
            continue;
        }
        if (n0 < m) {
            y[(size_t)n0 * ldy + row0 + g] = acc[j][0];
            y[(size_t)n0 * ldy + row0 + g + 8] = acc[j][2];
        }
        if (n1 < m) {
            y[(size_t)n1 * ldy + row0 + g] = acc[j][1];
            y[(size_t)n1 * ldy + row0 + g + 8] = acc[j][3];
        }
    }
}

// Variant 4 (QE_W4HX_MODE=3): like variant 3, but a warp computes TW = 2 adjacent tiles for one 32-row half of
// the tokens (NT <= 4 token tiles): every B fragment load feeds 2 tiles x 2 k-steps = 4 mma. Grid: tile pairs x
// row halves (adjacent blocks: the second half re-reads the pair's weights while they are in L2).
template <int NT>
__global__ void __launch_bounds__(160, NT > 4 ? 2 : NT >= 4 ? 3 : 4) gemv_w4hp_kernel(const uint4* __restrict__ q, const bf16* __restrict__ s,
                                                           int K, const __half* __restrict__ xh, int ldx,
                                                           float* __restrict__ y, int ldy, int m, bool glu,
                                                           __half* __restrict__ yh, int ldyh, int tiles, int halves) {
    const int pair = blockIdx.x / halves, half = blockIdx.x - pair * halves;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int nwarps = blockDim.x >> 5;
    const int g = lane >> 2, t = lane & 3;
    const int groups = K >> 7, chunks = K >> 6;
    const int gb = warp * groups / nwarps, ge = (warp + 1) * groups / nwarps;
    const int r0t = pair * 2;
    const bool two = r0t + 1 < tiles;
    const uint4* qp0 = q + (size_t)r0t * chunks * 32 + lane;
    const uint4* qp1 = q + (size_t)(two ? r0t + 1 : r0t) * chunks * 32 + lane;
    const bf16* sp0 = s + (size_t)r0t * groups * 16;
    const bf16* sp1 = s + (size_t)(two ? r0t + 1 : r0t) * groups * 16;
    const int rb = half * 8 * NT;  // first token row of this half
    const __half* xb = xh + t * 16;
    float acc[2][NT][4];
#pragma unroll
    for (int u = 0; u < 2; u++)
#pragma unroll
        for (int j = 0; j < NT; j++) acc[u][j][0] = acc[u][j][1] = acc[u][j][2] = acc[u][j][3] = 0.f;
    uint4 wn0 = gb < ge ? ldg_stream(qp0 + (size_t)(2 * gb) * 32) : make_uint4(0, 0, 0, 0);
    uint4 wn1 = gb < ge ? ldg_stream(qp1 + (size_t)(2 * gb) * 32) : make_uint4(0, 0, 0, 0);
    for (int gi = gb; gi < ge; gi++) {
        const float sa = xh_scale(sp0[gi * 16 + g]), sa8 = xh_scale(sp0[gi * 16 + g + 8]);
        const float sb = xh_scale(sp1[gi * 16 + g]), sb8 = xh_scale(sp1[gi * 16 + g + 8]);
#pragma unroll
        for (int c = 0; c < 2; c++) {
            const int ch = 2 * gi + c;
            const uint4 w0 = wn0, w1 = wn1;
            if (ch + 1 < 2 * ge) {
                wn0 = ldg_stream(qp0 + (size_t)(ch + 1) * 32);
                wn1 = ldg_stream(qp1 + (size_t)(ch + 1) * 32);
            }
            const uint32_t wa[4] = {w0.x, w0.y, w0.z, w0.w}, wb[4] = {w1.x, w1.y, w1.z, w1.w};
            uint32_t h[2][NT][2];
#pragma unroll
            for (int u = 0; u < 2; u++)
#pragma unroll
                for (int nt = 0; nt < NT; nt++) h[u][nt][0] = h[u][nt][1] = 0u;
#pragma unroll
            for (int hp = 0; hp < 2; hp++) {
                uint32_t a0[4], a1[4], b0[4], b1[4];
                xh_dequant(wa[2 * hp], a0);
                xh_dequant(wa[2 * hp + 1], a1);
                xh_dequant(wb[2 * hp], b0);
                xh_dequant(wb[2 * hp + 1], b1);
#pragma unroll
                for (int nt = 0; nt < NT; nt++) {
                    const int tok = min(rb + 8 * nt + g, m - 1);
                    const uint4 bb = *reinterpret_cast<const uint4*>(xb + (size_t)tok * ldx + (size_t)ch * 64 + hp * 8);
                    xh_mma(h[0][nt], a0, bb.x, bb.y);
                    xh_mma(h[0][nt], a1, bb.z, bb.w);
                    xh_mma(h[1][nt], b0, bb.x, bb.y);
                    xh_mma(h[1][nt], b1, bb.z, bb.w);
                }
            }
#pragma unroll
            for (int nt = 0; nt < NT; nt++) {
                xh_flush(acc[0][nt], h[0][nt], sa, sa8);
                xh_flush(acc[1][nt], h[1][nt], sb, sb8);
            }
        }
    }
    __shared__ float red[5][4][32];
#pragma unroll
    for (int u = 0; u < 2; u++) {
        const int row0 = (r0t + u) * 16;
        const bool ok = u == 0 || two;
#pragma unroll
        for (int j = 0; j < NT; j++) {
            if (nwarps > 1) {
                __syncthreads();
                if (warp != 0)
#pragma unroll
                    for (int c = 0; c < 4; c++) red[warp][c][lane] = acc[u][j][c];
                __syncthreads();
                if (warp == 0)
                    for (int ww = 1; ww < nwarps; ww++)
#pragma unroll
                        for (int c = 0; c < 4; c++) acc[u][j][c] += red[ww][c][lane];
            }
            if (warp != 0 || !ok) continue;
            const int n0 = rb + 8 * j + 2 * t, n1 = n0 + 1;
            if (glu) {
                bf16* act = reinterpret_cast<bf16*>(y);
                const int col = row0 / 2 + g;
                const float v0 = silu_f(acc[u][j][0]) * acc[u][j][2], v1 = silu_f(acc[u][j][1]) * acc[u][j][3];
                if (n0 < m) act[(size_t)n0 * ldy + col] = __float2bfloat16(v0);
                if (n1 < m) act[(size_t)n1 * ldy + col] = __float2bfloat16(v1);
                if (yh) {
                    if (n0 < m) store_xh(yh + (size_t)n0 * ldyh, col, v0);
                    if (n1 < m) store_xh(yh + (size_t)n1 * ldyh, col, v1);
                }
                continue;
            }
            if (n0 < m) {
                y[(size_t)n0 * ldy + row0 + g] = acc[u][j][0];
                y[(size_t)n0 * ldy + row0 + g + 8] = acc[u][j][2];
            }
            if (n1 < m) {
                y[(size_t)n1 * ldy + row0 + g] = acc[u][j][1];
                y[(size_t)n1 * ldy + row0 + g + 8] = acc[u][j][3];
            }
        }
    }
}

// Variant 5 (QE_W4HX_MODE=4): a block owns TPB tiles x all nw K slices (warp = (tile, slice)) and stages the x
// chunk of every slice in shared memory (one x stream per slice, shared by the TPB tiles); slices are added in
// order in shared memory at the end - no cross-block partials. For 320-tile matrices (TPB = 4: 80 blocks of 20
// warps = one wave).
template <int NT, int S>
__global__ void __launch_bounds__(640, 1) gemv_w4hk_kernel(const uint4* __restrict__ q, const bf16* __restrict__ s,
                                                           int K, const __half* __restrict__ xh, int ldx,
                                                           float* __restrict__ y, int ldy, int m, bool glu,
                                                           __half* __restrict__ yh, int ldyh, int tiles, int TPB,
                                                           int nw) {
    extern __shared__ __align__(16) unsigned char smem[];
    const unsigned smem_u = (unsigned)__cvta_generic_to_shared(smem);
    const int tid = threadIdx.x, lane = tid & 31, wid = tid >> 5;
    const int g = lane >> 2, t = lane & 3;
    const int groups = K >> 7, chunks = K >> 6;
    const int w = wid % nw, tl = wid / nw;
    const int tile = blockIdx.x * TPB + tl;
    const bool valid = tile < tiles;
    const int gb = w * groups / nw, ge = (w + 1) * groups / nw;
    const int len = 2 * (ge - gb);
    const int Lmax = 2 * ((groups + nw - 1) / nw);
    constexpr int slice_rows = 8 * NT;
    const int slice_bytes = slice_rows * kRowB, stage_bytes = nw * slice_bytes;
    for (int i = tid * 16; i < S * stage_bytes; i += blockDim.x * 16) *reinterpret_cast<uint4*>(smem + i) = make_uint4(0, 0, 0, 0);
    __syncthreads();
    const int pieces = nw * m * 8;
    auto issue = [&](int i, int st) {
        for (int p = tid; p < pieces; p += blockDim.x) {
            const int sw = p / (m * 8), rr = p - sw * m * 8, r = rr >> 3, c = rr & 7;
            const int sgb = sw * groups / nw, slen = 2 * ((sw + 1) * groups / nw - sgb);
            if (i < slen)
                cp_async16(smem_u + st * stage_bytes + sw * slice_bytes + r * kRowB + c * 16,
                           xh + (size_t)r * ldx + (size_t)(2 * sgb + i) * 64 + c * 8);
        }
    };
#pragma unroll
    for (int i = 0; i < S - 1; i++) {
        if (i < Lmax) issue(i, i);
        cp_async_commit();
    }
    const uint4* qp = q + ((size_t)(valid ? tile : 0) * chunks + 2 * gb) * 32 + lane;
    const bf16* sp = s + ((size_t)(valid ? tile : 0) * groups + gb) * 16 + g;
    uint4 wb0 = make_uint4(0, 0, 0, 0), wb1 = wb0;
    if (valid && len > 0) wb0 = ldg_stream(qp);
    if (valid && len > 1) wb1 = ldg_stream(qp + 32);
    float sg = 0.f, sg8 = 0.f;
    float acc[NT][4];
#pragma unroll
    for (int j = 0; j < NT; j++) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.f;
    const unsigned xoff = w * slice_bytes + g * kRowB + t * 32;
    for (int i = 0; i < Lmax; i++) {
        cp_async_wait<S - 2>();
        __syncthreads();
        if (i + S - 1 < Lmax) issue(i + S - 1, (i + S - 1) % S);
        cp_async_commit();
        if (!valid || i >= len) continue;
        const uint4 wv = (i & 1) ? wb1 : wb0;
        if (i + 2 < len) {
            if (i & 1) wb1 = ldg_stream(qp + (size_t)(i + 2) * 32);
            else wb0 = ldg_stream(qp + (size_t)(i + 2) * 32);
        }
        if ((i & 1) == 0) {
            sg = xh_scale(sp[(i >> 1) * 16]);
            sg8 = xh_scale(sp[(i >> 1) * 16 + 8]);
        }
        const unsigned xs = smem_u + (i % S) * stage_bytes + xoff;
        const uint32_t words[4] = {wv.x, wv.y, wv.z, wv.w};
        uint32_t h[NT][2];
#pragma unroll
        for (int nt = 0; nt < NT; nt++) h[nt][0] = h[nt][1] = 0u;
#pragma unroll
        for (int hp = 0; hp < 2; hp++) {
            uint32_t a0[4], a1[4];
            xh_dequant(words[2 * hp], a0);
            xh_dequant(words[2 * hp + 1], a1);
#pragma unroll
            for (int nt = 0; nt < NT; nt++) {
                uint4 b;
                asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];"
                             : "=r"(b.x), "=r"(b.y), "=r"(b.z), "=r"(b.w) : "r"(xs + nt * 8 * kRowB + hp * 16));
                xh_mma(h[nt], a0, b.x, b.y);
                xh_mma(h[nt], a1, b.z, b.w);
            }
        }
#pragma unroll
        for (int nt = 0; nt < NT; nt++) xh_flush(acc[nt], h[nt], sg, sg8);
    }
    cp_async_wait<0>();
    __syncthreads();
    float* red = reinterpret_cast<float*>(smem);  // [nw][TPB][NT][4][32]
    if (w != 0 && valid)
#pragma unroll
        for (int j = 0; j < NT; j++)
#pragma unroll
            for (int c = 0; c < 4; c++) red[((((size_t)w * TPB + tl) * NT + j) * 4 + c) * 32 + lane] = acc[j][c];
    __syncthreads();
    if (w != 0 || !valid) return;
    for (int ww = 1; ww < nw; ww++)
#pragma unroll
        for (int j = 0; j < NT; j++)
#pragma unroll
            for (int c = 0; c < 4; c++) acc[j][c] += red[((((size_t)ww * TPB + tl) * NT + j) * 4 + c) * 32 + lane];
    const int row0 = tile * 16;
#pragma unroll
    for (int j = 0; j < NT; j++) {
        const int n0 = 8 * j + 2 * t, n1 = n0 + 1;
        if (glu) {
            bf16* act = reinterpret_cast<bf16*>(y);
            const int col = row0 / 2 + g;
            const float v0 = silu_f(acc[j][0]) * acc[j][2], v1 = silu_f(acc[j][1]) * acc[j][3];
            if (n0 < m) act[(size_t)n0 * ldy + col] = __float2bfloat16(v0);
            if (n1 < m) act[(size_t)n1 * ldy + col] = __float2bfloat16(v1);
            if (yh) {
                if (n0 < m) store_xh(yh + (size_t)n0 * ldyh, col, v0);
                if (n1 < m) store_xh(yh + (size_t)n1 * ldyh, col, v1);
            }
            continue;
        }
        if (n0 < m) {
            y[(size_t)n0 * ldy + row0 + g] = acc[j][0];
            y[(size_t)n0 * ldy + row0 + g + 8] = acc[j][2];
        }
        if (n1 < m) {
            y[(size_t)n1 * ldy + row0 + g] = acc[j][1];
            y[(size_t)n1 * ldy + row0 + g + 8] = acc[j][3];
        }
    }
}

// Variant 6 (QE_W4HX_MODE=5): one CTA per SM. A CTA owns TT = ceil(tiles / SMs) tiles x all token rows x the WHOLE
// K: x (m rows x 128 k) and the TT tiles' weights (2 chunks each) of one 128-k group are staged per pipeline stage
// with cp.async (barrier every 128 k); warp = 2 tiles x 32 tokens (B fragment -> 4 mma, A from smem shared by the
// two token halves). Each warp runs the K slices (pick_warps) one after another and adds them in slice order
// (tot = s0, += s1, ...) - no cross-block partials, one wave. ABL (QE_W4HM_ABL, skeleton gates): 1 no mma,
// 2 no x staging, 4 no weight staging.
constexpr int kXRowB = 272;  // 256 bytes (128 k) + 16 padding per staged x row
template <int S, int ABL>
__global__ void __launch_bounds__(512, 1) gemv_w4hm_kernel(const uint4* __restrict__ q, const bf16* __restrict__ s,
                                                           int K, const __half* __restrict__ xh, int ldx,
                                                           float* __restrict__ y, int ldy, int m, bool glu,
                                                           __half* __restrict__ yh, int ldyh, int tiles, int TT,
                                                           int nw) {
    constexpr int NT = 4;
    extern __shared__ __align__(16) unsigned char smem[];
    const unsigned smem_u = (unsigned)__cvta_generic_to_shared(smem);
    const int tid = threadIdx.x, lane = tid & 31, wid = tid >> 5;
    const int g = lane >> 2, t = lane & 3;
    const int groups = K >> 7, chunks = K >> 6;
    const int H = (m + 31) / 32;
    const int pair = wid / H, half = wid - pair * H;
    const int tb = blockIdx.x * TT;                 // first tile of the CTA
    const int ntl = min(TT, tiles - tb);            // tiles of this CTA
    const int lt0 = 2 * pair, lt1 = 2 * pair + 1;   // local tiles of this warp
    const bool v0 = lt0 < ntl, v1 = lt1 < ntl;
    const int xbytes = 64 * kXRowB, wbytes = TT * 2 * 512;
    const int stage_bytes = xbytes + wbytes;
    for (int i = tid * 16; i < S * stage_bytes; i += blockDim.x * 16) *reinterpret_cast<uint4*>(smem + i) = make_uint4(0, 0, 0, 0);
    __syncthreads();
    const int xpieces = (ABL & 2) ? 0 : m * 16, wpieces = (ABL & 4) ? 0 : ntl * 64;
    auto issue = [&](int gi, int st) {
        const unsigned base = smem_u + st * stage_bytes;
        for (int p = tid; p < xpieces + wpieces; p += blockDim.x) {
            if (p < xpieces) {
                const int r = p >> 4, c = p & 15;
                cp_async16(base + r * kXRowB + c * 16, xh + (size_t)r * ldx + (size_t)gi * 128 + c * 8);
            } else {
                const int pp = p - xpieces, lt = pp >> 6, rest = pp & 63;  // rest = chunk-in-group * 32 + lane
                cp_async16(base + xbytes + (lt * 64 + rest) * 16, q + ((size_t)(tb + lt) * chunks + 2 * gi) * 32 + rest);
            }
        }
    };
#pragma unroll
    for (int i = 0; i < S - 1; i++) {
        if (i < groups) issue(i, i);
        cp_async_commit();
    }
    const bf16* sp0 = s + (size_t)(tb + (v0 ? lt0 : 0)) * groups * 16 + g;
    const bf16* sp1 = s + (size_t)(tb + (v1 ? lt1 : (v0 ? lt0 : 0))) * groups * 16 + g;
    float acc[2][NT][4], tot[2][NT][4];
#pragma unroll
    for (int u = 0; u < 2; u++)
#pragma unroll
        for (int j = 0; j < NT; j++)
#pragma unroll
            for (int c = 0; c < 4; c++) acc[u][j][c] = tot[u][j][c] = 0.f;
    int w = 0, wend = groups / nw;  // group where slice w ends (exclusive)
    const unsigned xo = (half * 32 + g) * kXRowB + t * 32;
    for (int gi = 0; gi < groups; gi++) {
        cp_async_wait<S - 2>();
        __syncthreads();
        if (gi + S - 1 < groups) issue(gi + S - 1, (gi + S - 1) % S);
        cp_async_commit();
        if (v0) {
            const float sa = xh_scale(sp0[gi * 16]), sa8 = xh_scale(sp0[gi * 16 + 8]);
            const float sb = xh_scale(sp1[gi * 16]), sb8 = xh_scale(sp1[gi * 16 + 8]);
            const unsigned base = smem_u + (gi % S) * stage_bytes;
#pragma unroll
            for (int c = 0; c < 2; c++) {
                uint4 w0, w1;
                asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(w0.x), "=r"(w0.y), "=r"(w0.z), "=r"(w0.w)
                             : "r"(base + xbytes + (lt0 * 64 + c * 32 + lane) * 16));
                asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(w1.x), "=r"(w1.y), "=r"(w1.z), "=r"(w1.w)
                             : "r"(base + xbytes + ((v1 ? lt1 : lt0) * 64 + c * 32 + lane) * 16));
                const uint32_t wa[4] = {w0.x, w0.y, w0.z, w0.w}, wb[4] = {w1.x, w1.y, w1.z, w1.w};
                uint32_t h[2][NT][2];
#pragma unroll
                for (int u = 0; u < 2; u++)
#pragma unroll
                    for (int nt = 0; nt < NT; nt++) h[u][nt][0] = h[u][nt][1] = 0u;
#pragma unroll
                for (int hp = 0; hp < 2; hp++) {
                    uint32_t a0[4], a1[4], b0[4], b1[4];
                    xh_dequant(wa[2 * hp], a0);
                    xh_dequant(wa[2 * hp + 1], a1);
                    xh_dequant(wb[2 * hp], b0);
                    xh_dequant(wb[2 * hp + 1], b1);
#pragma unroll
                    for (int nt = 0; nt < NT; nt++) {
                        uint4 bb;
                        asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(bb.x), "=r"(bb.y), "=r"(bb.z), "=r"(bb.w)
                                     : "r"(base + xo + nt * 8 * kXRowB + c * 128 + hp * 16));
                        if (ABL & 1) {
                            h[0][nt][0] ^= a0[0] ^ a1[1] ^ bb.x ^ bb.z; h[0][nt][1] ^= a0[2] ^ a1[3] ^ bb.y ^ bb.w;
                            h[1][nt][0] ^= b0[0] ^ b1[1] ^ bb.x; h[1][nt][1] ^= b0[2] ^ b1[3] ^ bb.y;
                        } else {
                            xh_mma(h[0][nt], a0, bb.x, bb.y);
                            xh_mma(h[0][nt], a1, bb.z, bb.w);
                            xh_mma(h[1][nt], b0, bb.x, bb.y);
                            xh_mma(h[1][nt], b1, bb.z, bb.w);
                        }
                    }
                }
#pragma unroll
                for (int nt = 0; nt < NT; nt++) {
                    xh_flush(acc[0][nt], h[0][nt], sa, sa8);
                    xh_flush(acc[1][nt], h[1][nt], sb, sb8);
                }
            }
        }
        if (gi + 1 == wend) {  // slice w done: add it in slice order
#pragma unroll
            for (int u = 0; u < 2; u++)
#pragma unroll
                for (int j = 0; j < NT; j++)
#pragma unroll
                    for (int c = 0; c < 4; c++) {
                        tot[u][j][c] = w == 0 ? acc[u][j][c] : tot[u][j][c] + acc[u][j][c];
                        acc[u][j][c] = 0.f;
                    }
            w++;
            wend = (w + 1) * groups / nw;
        }
    }
    cp_async_wait<0>();
    if (!v0) return;
#pragma unroll
    for (int u = 0; u < 2; u++) {
        if (u == 1 && !v1) break;
        const int row0 = (tb + (u ? lt1 : lt0)) * 16;
#pragma unroll
        for (int j = 0; j < NT; j++) {
            const int n0 = half * 32 + 8 * j + 2 * t, n1 = n0 + 1;
            if (glu) {
                bf16* act = reinterpret_cast<bf16*>(y);
                const int col = row0 / 2 + g;
                const float q0 = silu_f(tot[u][j][0]) * tot[u][j][2], q1 = silu_f(tot[u][j][1]) * tot[u][j][3];
                if (n0 < m) act[(size_t)n0 * ldy + col] = __float2bfloat16(q0);
                if (n1 < m) act[(size_t)n1 * ldy + col] = __float2bfloat16(q1);
                if (yh) {
                    if (n0 < m) store_xh(yh + (size_t)n0 * ldyh, col, q0);
                    if (n1 < m) store_xh(yh + (size_t)n1 * ldyh, col, q1);
                }
                continue;
            }
            if (n0 < m) {
                y[(size_t)n0 * ldy + row0 + g] = tot[u][j][0];
                y[(size_t)n0 * ldy + row0 + g + 8] = tot[u][j][2];
            }
            if (n1 < m) {
                y[(size_t)n1 * ldy + row0 + g] = tot[u][j][1];
                y[(size_t)n1 * ldy + row0 + g + 8] = tot[u][j][3];
            }
        }
    }
}

constexpr size_t kRedFloats = (size_t)4 << 20;  // 16 MB: nw x tiles x 8 NT x 128 (dn_in / qkv 7168: 9 MB)
constexpr int kMaxGroups = 4096;
float* g_red[16] = {};
unsigned* g_ctr[16] = {};

int sm_count_x() {
    static int n[16] = {};
    int dev;
    CUDA_CHECK(cudaGetDevice(&dev));
    if (!n[dev]) CUDA_CHECK(cudaDeviceGetAttribute(&n[dev], cudaDevAttrMultiProcessorCount, dev));
    return n[dev];
}

template <int NT, int S>
void launch(const W4& w, const __half* xh, int ldx, float* y, int ldy, int m, cudaStream_t st, __half* yh,
            int ldyh, int TR, int nw, float* red, unsigned* ctr) {
    const int tiles = w.n / 16;
    const int groupsN = (tiles + TR - 1) / TR;
    const size_t smem = (size_t)S * 8 * NT * kRowB;
    static const int abl = env_int("QE_W4HX_ABL", 0);
#define QE_XA(A_) gemv_w4hx_kernel<NT, S, A_><<<groupsN * nw, TR * 32, smem, st>>>(w.q, w.s, w.k, xh, ldx, y, ldy, m, \
                                                                        w.glu, yh, ldyh, tiles, nw, red, ctr)
    if constexpr (NT == 8 && S == 4) {
        switch (abl) {
            case 1: QE_XA(1); return;
            case 2: QE_XA(2); return;
            case 3: QE_XA(3); return;
            case 4: QE_XA(4); return;
            case 5: QE_XA(5); return;
            case 6: QE_XA(6); return;
            case 7: QE_XA(7); return;
            case 8: QE_XA(8); return;
            case 15: QE_XA(15); return;
            default: break;
        }
    }
    QE_XA(0);
#undef QE_XA
}

template <int NT, int S>
void launch_s(const W4& w, const __half* xh, int ldx, float* y, int ldy, int m, cudaStream_t st, __half* yh,
              int ldyh, int TR, int nw) {
    const int tiles = w.n / 16;
    const size_t smem = (size_t)S * 8 * NT * kRowB;
    gemv_w4hs_kernel<NT, S, 4><<<(tiles + TR - 1) / TR, TR * 32, smem, st>>>(w.q, w.s, w.k, xh, ldx, y, ldy, m, w.glu,
                                                                        yh, ldyh, tiles, nw);
}

template <int NT>
void set_attr() {
    if constexpr (NT == 2) {
        CUDA_CHECK(cudaFuncSetAttribute(gemv_w4hm_kernel<2, 0>, cudaFuncAttributeMaxDynamicSharedMemorySize, 99 * 1024));
        CUDA_CHECK(cudaFuncSetAttribute(gemv_w4hm_kernel<2, 1>, cudaFuncAttributeMaxDynamicSharedMemorySize, 99 * 1024));
        CUDA_CHECK(cudaFuncSetAttribute(gemv_w4hm_kernel<2, 3>, cudaFuncAttributeMaxDynamicSharedMemorySize, 99 * 1024));
        CUDA_CHECK(cudaFuncSetAttribute(gemv_w4hm_kernel<2, 5>, cudaFuncAttributeMaxDynamicSharedMemorySize, 99 * 1024));
        CUDA_CHECK(cudaFuncSetAttribute(gemv_w4hm_kernel<2, 7>, cudaFuncAttributeMaxDynamicSharedMemorySize, 99 * 1024));
        CUDA_CHECK(cudaFuncSetAttribute(gemv_w4hm_kernel<3, 0>, cudaFuncAttributeMaxDynamicSharedMemorySize, 99 * 1024));
        CUDA_CHECK(cudaFuncSetAttribute(gemv_w4hm_kernel<3, 1>, cudaFuncAttributeMaxDynamicSharedMemorySize, 99 * 1024));
        CUDA_CHECK(cudaFuncSetAttribute(gemv_w4hm_kernel<3, 3>, cudaFuncAttributeMaxDynamicSharedMemorySize, 99 * 1024));
        CUDA_CHECK(cudaFuncSetAttribute(gemv_w4hm_kernel<3, 5>, cudaFuncAttributeMaxDynamicSharedMemorySize, 99 * 1024));
        CUDA_CHECK(cudaFuncSetAttribute(gemv_w4hm_kernel<3, 7>, cudaFuncAttributeMaxDynamicSharedMemorySize, 99 * 1024));
    }
    CUDA_CHECK(cudaFuncSetAttribute(gemv_w4hk_kernel<NT, 2>, cudaFuncAttributeMaxDynamicSharedMemorySize, 99 * 1024));
    CUDA_CHECK(cudaFuncSetAttribute(gemv_w4hk_kernel<NT, 3>, cudaFuncAttributeMaxDynamicSharedMemorySize, 99 * 1024));
    CUDA_CHECK(cudaFuncSetAttribute(gemv_w4hs_kernel<NT, 3, 4>, cudaFuncAttributeMaxDynamicSharedMemorySize, 64 * 1024));
    CUDA_CHECK(cudaFuncSetAttribute(gemv_w4hs_kernel<NT, 4, 4>, cudaFuncAttributeMaxDynamicSharedMemorySize, 64 * 1024));
    CUDA_CHECK(cudaFuncSetAttribute(gemv_w4hx_kernel<NT, 3>, cudaFuncAttributeMaxDynamicSharedMemorySize, 64 * 1024));
    CUDA_CHECK(cudaFuncSetAttribute(gemv_w4hx_kernel<NT, 4>, cudaFuncAttributeMaxDynamicSharedMemorySize, 64 * 1024));
    if constexpr (NT == 8) {
        CUDA_CHECK(cudaFuncSetAttribute(gemv_w4hx_kernel<8, 4, 1>, cudaFuncAttributeMaxDynamicSharedMemorySize, 64 * 1024));
        CUDA_CHECK(cudaFuncSetAttribute(gemv_w4hx_kernel<8, 4, 2>, cudaFuncAttributeMaxDynamicSharedMemorySize, 64 * 1024));
        CUDA_CHECK(cudaFuncSetAttribute(gemv_w4hx_kernel<8, 4, 3>, cudaFuncAttributeMaxDynamicSharedMemorySize, 64 * 1024));
        CUDA_CHECK(cudaFuncSetAttribute(gemv_w4hx_kernel<8, 4, 4>, cudaFuncAttributeMaxDynamicSharedMemorySize, 64 * 1024));
        CUDA_CHECK(cudaFuncSetAttribute(gemv_w4hx_kernel<8, 4, 5>, cudaFuncAttributeMaxDynamicSharedMemorySize, 64 * 1024));
        CUDA_CHECK(cudaFuncSetAttribute(gemv_w4hx_kernel<8, 4, 6>, cudaFuncAttributeMaxDynamicSharedMemorySize, 64 * 1024));
        CUDA_CHECK(cudaFuncSetAttribute(gemv_w4hx_kernel<8, 4, 7>, cudaFuncAttributeMaxDynamicSharedMemorySize, 64 * 1024));
        CUDA_CHECK(cudaFuncSetAttribute(gemv_w4hx_kernel<8, 4, 8>, cudaFuncAttributeMaxDynamicSharedMemorySize, 64 * 1024));
        CUDA_CHECK(cudaFuncSetAttribute(gemv_w4hx_kernel<8, 4, 15>, cudaFuncAttributeMaxDynamicSharedMemorySize, 64 * 1024));
    }
}

}  // namespace

void gemv_x_init() {
    int dev;
    CUDA_CHECK(cudaGetDevice(&dev));
    if (g_red[dev]) return;
    CUDA_CHECK(cudaMalloc(&g_red[dev], kRedFloats * 4));
    CUDA_CHECK(cudaMalloc(&g_ctr[dev], kMaxGroups * 4));
    CUDA_CHECK(cudaMemset(g_ctr[dev], 0, kMaxGroups * 4));
    set_attr<2>(); set_attr<3>(); set_attr<4>(); set_attr<5>(); set_attr<6>(); set_attr<7>(); set_attr<8>();
}

// TR (warps = tiles per block, QE_W4HX_TR) and S (stages, QE_W4HX_S): experiments; defaults measured
bool gemv_w4hx(const W4& w, const __half* xh, int ldx, float* y, int ldy, int m, cudaStream_t st, __half* yh,
               int ldyh) {
    int dev;
    CUDA_CHECK(cudaGetDevice(&dev));
    if (!g_red[dev] || m < 9 || m > 64 || w.n % 16 || w.k % 128 || ldx % 64) return false;
    const int tiles = w.n / 16;
    const int nw = pick_warps(tiles, w.k / 128, true);
    static const int tr_env = env_int("QE_W4HX_TR", 8);

    const int TR = std::max(1, std::min(kMaxTR, tr_env));
    if ((size_t)nw * tiles * 8 * 128 > kRedFloats || (tiles + TR - 1) / TR > kMaxGroups) return false;
    float* red = g_red[dev];
    unsigned* ctr = g_ctr[dev];
    const int NT = (m + 7) / 8;
    // QE_W4HX_MODE: -1 (default) per shape - 33..64 rows: matrices of <= 320 tiles (N = 5120: down, o, DeltaNet
    // out) the per-tile kernel (2), wider ones the staged one (0, S = 3); 25..32 rows (called from gemv_w4h
    // for >= 352 tiles): tile pairs (3).
    static const int mode_env = env_int("QE_W4HX_MODE", -1);
    const int mode = mode_env >= 0 ? mode_env : m <= 32 ? 3 : tiles <= 320 ? 2 : 0;
    const int s_env = env_int("QE_W4HX_S", mode_env < 0 ? 3 : 4);
    if (mode == 5) {  // one CTA per SM, whole K, TT tiles
        const int TT = (tiles + sm_count_x() - 1) / sm_count_x();
        const int H = (m + 31) / 32, warps = (TT + 1) / 2 * H;
        const int S = env_int("QE_W4HM_S", 3), abl = env_int("QE_W4HM_ABL", 0);
        const size_t smem = (size_t)S * (64 * kXRowB + TT * 2 * 512);
        if (warps * 32 > 512 || smem > 99 * 1024) return false;
        const int blocks = (tiles + TT - 1) / TT;
#define QE_M(S_, A_) gemv_w4hm_kernel<S_, A_><<<blocks, warps * 32, smem, st>>>(w.q, w.s, w.k, xh, ldx, y, ldy, m, w.glu, yh, ldyh, tiles, TT, nw)
#define QE_MA(S_) do { if (abl == 1) QE_M(S_, 1); else if (abl == 3) QE_M(S_, 3); else if (abl == 5) QE_M(S_, 5); \
                       else if (abl == 7) QE_M(S_, 7); else QE_M(S_, 0); } while (0)
        if (S == 2) QE_MA(2); else QE_MA(3);
#undef QE_MA
#undef QE_M
        return true;
    }
    if (mode == 4) {  // all slices of TPB tiles per block
        static const int tpb_env = env_int("QE_W4HX_TPB", 4);
        const int TPB = tpb_env, NTk = (m + 7) / 8;
        const int S = env_int("QE_W4HX_KS", 2);
        const size_t smem = std::max((size_t)S * nw * 8 * NTk * kRowB, (size_t)nw * TPB * NTk * 128 * 4);
        if (TPB * nw > 20 || smem > 99 * 1024) return false;
        const int blocks = (tiles + TPB - 1) / TPB;
#define QE_K(NT_) do { if (S == 3) gemv_w4hk_kernel<NT_, 3><<<blocks, TPB * nw * 32, smem, st>>>(w.q, w.s, w.k, xh, ldx, y, ldy, m, w.glu, yh, ldyh, tiles, TPB, nw); \
                       else gemv_w4hk_kernel<NT_, 2><<<blocks, TPB * nw * 32, smem, st>>>(w.q, w.s, w.k, xh, ldx, y, ldy, m, w.glu, yh, ldyh, tiles, TPB, nw); } while (0)
        switch (NTk) { case 2: QE_K(2); break; case 3: QE_K(3); break; case 4: QE_K(4); break; case 5: QE_K(5); break;
                       case 6: QE_K(6); break; case 7: QE_K(7); break; default: QE_K(8); break; }
#undef QE_K
        return true;
    }
    if (mode == 3) {  // tile pairs x 32-row halves
        CHECK(nw <= 5, "w4hp slices");
        static const int hrows = env_int("QE_W4HX_HALF", 32);  // token rows per block (32 or 64)
        const int halves = (m + hrows - 1) / hrows, mh = std::min(m, hrows), ntt = (mh + 7) / 8;
        const int blocks = (tiles + 1) / 2 * halves;
#define QE_P(NT_) gemv_w4hp_kernel<NT_><<<blocks, nw * 32, 0, st>>>(w.q, w.s, w.k, xh, ldx, y, ldy, m, w.glu, yh, ldyh, tiles, halves)
        if (ntt == 1) QE_P(1); else if (ntt == 2) QE_P(2); else if (ntt == 3) QE_P(3); else if (ntt == 4) QE_P(4);
        else if (ntt == 5) QE_P(5); else if (ntt == 6) QE_P(6); else if (ntt == 7) QE_P(7); else QE_P(8);
#undef QE_P
        return true;
    }
#define QE_X(NT_) do { if (mode == 2) { CHECK(nw <= 5, "w4hz slices"); \
                           gemv_w4hz_kernel<NT_><<<tiles, nw * 32, 0, st>>>(w.q, w.s, w.k, xh, ldx, y, ldy, m, w.glu, yh, ldyh); } \
                       else if (mode == 1) { if (s_env == 3) launch_s<NT_, 3>(w, xh, ldx, y, ldy, m, st, yh, ldyh, TR, nw); \
                                       else launch_s<NT_, 4>(w, xh, ldx, y, ldy, m, st, yh, ldyh, TR, nw); } \
                       else if (s_env == 3) launch<NT_, 3>(w, xh, ldx, y, ldy, m, st, yh, ldyh, TR, nw, red, ctr); \
                       else launch<NT_, 4>(w, xh, ldx, y, ldy, m, st, yh, ldyh, TR, nw, red, ctr); } while (0)
    switch (NT) {
        case 2: QE_X(2); break;
        case 3: QE_X(3); break;
        case 4: QE_X(4); break;
        case 5: QE_X(5); break;
        case 6: QE_X(6); break;
        case 7: QE_X(7); break;
        default: QE_X(8); break;
    }
#undef QE_X
    return true;
}
