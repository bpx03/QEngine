// "Skinny" INT4 GEMM for batched decode steps with 9..32 token rows.
//
// The wide GEMV (gemv_w4p_kernel) keeps one 16-row weight tile per block and re-reads the packed
// activations for every tile (m x K x 2 bytes per tile: 235 MB per gate_up call at m = 32), so it
// is bound by L2 bandwidth (~2.2 TB/s on GA102) from ~12 rows on, and it has no software prefetch
// of the weights, so every group starts with a DRAM latency stall. Here the grid is one block per
// SM (82): a block owns ceil(tiles / 82) weight tiles (TW = 2 per warp) and the activation chunk
// of each K slice is staged once per block in shared memory (cp.async, S stages of 64 k) and
// shared by all its warps through ldmatrix; weights go straight to registers with a prefetch ring.
//
// Summation order is exactly the GEMV's (bit-identical results, required for batch == alone):
// warp (tiles, w) accumulates K slice w = groups [w*G/nw, (w+1)*G/nw) of each of its tiles
// sequentially in the mma accumulator, nw = pick_warps(tiles, G) as in gemv_w4, and the slices
// are added in order w = 0..nw-1 at the end. Weights are dequantized exactly as in the GEMV.
#include "../env.h"
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

__device__ __forceinline__ void mma_bf16(float c[4], const uint32_t a[4], uint32_t b0, uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, "
        "{%8,%9}, {%0,%1,%2,%3};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void dequant_w4(uint32_t q, uint32_t a[4]) {
    const bf162 bias = __halves2bfloat162(__float2bfloat16(136.f), __float2bfloat16(136.f));
#pragma unroll
    for (int j = 0; j < 4; j++) {
        uint32_t v = ((q >> (4 * j)) & 0x000F000Fu) | 0x43004300u;
        bf162 h = *reinterpret_cast<bf162*>(&v);
        h = __hsub2(h, bias);
        a[j] = *reinterpret_cast<uint32_t*>(&h);
    }
}

__device__ __forceinline__ float silu_f(float x) { return x / (1.f + __expf(-x)); }

__device__ __forceinline__ void cp_async16(unsigned smem, const void* gmem) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(smem), "l"(gmem));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;"); }
template <int N>
__device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;" ::"n"(N)); }

__device__ __forceinline__ void ldmatrix_x4(uint32_t r[4], unsigned s) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(s));
}

constexpr int kChunkK = 64;                 // k per pipeline stage (one uint4 of weights per lane)
constexpr int kRowBytes = kChunkK * 2 + 16; // 144: padded so ldmatrix is bank-conflict-free
constexpr int kSMs = 82;
constexpr int kMaxThreads = 512;            // <= 16 warps per block -> 128 registers
constexpr int kMaxSlices = 8;
constexpr int kMaxSmem = 98 * 1024;        // opt-in limit 99 KB minus the static arrays

// Warp work table built by the host: warp k handles `cnt` (1 or 2) consecutive tiles [tile0,
// tile0 + cnt) of the block on K slice `w`. Pairs never cross a slice (the two tiles share the
// B fragments), singles pad the warp count to a multiple of 4 (one warp per SM partition).
struct SkinnyWarps {
    unsigned char w[16], tile0[16], cnt[16];
};

// Dynamic smem: [S][nw][8*NT rows][kRowBytes] activation stages; reused for the slice reduction.
template <int NT, int S>
__global__ void __launch_bounds__(kMaxThreads) skinny_w4_kernel(
    const uint4* __restrict__ q, const bf16* __restrict__ s, int K, const bf16* __restrict__ x,
    int ldx, float* __restrict__ y, int ldy, int m, bool glu, int tiles, int TPB, int nw,
    SkinnyWarps wt, int L) {
    constexpr int TW = 2;                       // tiles per warp (same slice)
    constexpr int kRing = 2;                    // weight chunks in flight per tile
    extern __shared__ __align__(16) unsigned char smem[];
    __shared__ int slice_c0[kMaxSlices], slice_len[kMaxSlices];
    const unsigned smem_u = (unsigned)__cvta_generic_to_shared(smem);
    const int tid = threadIdx.x, lane = tid & 31, wid = tid >> 5;
    const int nthreads = blockDim.x;
    const int g = lane >> 2, t = lane & 3;
    const int groups = K >> 7, chunks = K >> 6;
    const int rows = 8 * NT;
    const int slice_bytes = rows * kRowBytes;
    const int stage_bytes = nw * slice_bytes;

    if (tid < nw) {
        slice_c0[tid] = 2 * (tid * groups / nw);
        slice_len[tid] = 2 * ((tid + 1) * groups / nw - tid * groups / nw);
    }
    // zero the stages once: rows >= m are never written and stay zero
    for (int i = tid * 16; i < S * stage_bytes; i += nthreads * 16)
        *reinterpret_cast<uint4*>(smem + i) = make_uint4(0, 0, 0, 0);
    __syncthreads();

    // stage loader: thread -> one (row, 16-byte piece) of the chunk, strided over the slices
    // (nsets = nthreads / (8 m) slices per round; the host guarantees nsets >= 1)
    const int per_slice = m * 8;
    const int nsets = nthreads / per_slice;
    const int myset = tid / per_slice, r = tid - myset * per_slice;
    const bool loader = myset < nsets;
    const bf16* xsrc = x + (size_t)(r >> 3) * ldx + (r & 7) * 8;
    const unsigned xdst = smem_u + (r >> 3) * kRowBytes + (r & 7) * 16;
    auto issue_stage = [&](int i, unsigned stage_off) {
        if (!loader) return;
        for (int sw = myset; sw < nw; sw += nsets)
            if (i < slice_len[sw])
                cp_async16(xdst + stage_off + sw * slice_bytes, xsrc + (slice_c0[sw] + i) * kChunkK);
    };
    unsigned issue_off = 0;
    for (int i = 0; i < S - 1; i++) {
        if (i < L) issue_stage(i, issue_off);
        cp_async_commit();
        issue_off = issue_off + stage_bytes == S * stage_bytes ? 0u : issue_off + stage_bytes;
    }

    // this warp: tiles [tl0, tl0 + cnt) of the block on slice w
    const int w = wt.w[wid], tl0 = wt.tile0[wid], cnt = wt.cnt[wid];
    const int gb = w * groups / nw, ge = (w + 1) * groups / nw;
    const int len = cnt > 0 ? 2 * (ge - gb) : 0;    // chunks of this slice
    const int nload = len - kRing;
    const int ngroup = ge - gb - 1;
    // ldmatrix row address of this lane within the slice stage: token 8*nt + (lane & 7)
    const unsigned xoff = w * slice_bytes + (lane & 7) * kRowBytes + (lane >> 3) * 16;
    bool valid[TW];
    const uint4* qp[TW];
    const bf16* sp[TW];
    uint4 wbuf[TW][kRing];
    bf162 sg[TW], sg8[TW], ns[TW], ns8[TW];
    float acc[TW][NT][4];
    const bf16 z = __float2bfloat16(0.f);
#pragma unroll
    for (int tw = 0; tw < TW; tw++) {
        const int tile = blockIdx.x * TPB + tl0 + tw;
        valid[tw] = tw < cnt && tile < tiles;
        const int tt = valid[tw] ? tile : 0;
        qp[tw] = q + ((size_t)tt * chunks + 2 * gb) * 32 + lane;
        sp[tw] = s + ((size_t)tt * groups + gb) * 16 + g;
#pragma unroll
        for (int d = 0; d < kRing; d++) {
            wbuf[tw][d] = make_uint4(0, 0, 0, 0);
            if (valid[tw] && d < len) wbuf[tw][d] = ldg_stream(qp[tw] + d * 32);
        }
        qp[tw] += kRing * 32;
        sg[tw] = sg8[tw] = ns[tw] = ns8[tw] = __bfloat162bfloat162(z);
        if (valid[tw] && len > 0) {
            ns[tw] = __bfloat162bfloat162(sp[tw][0]);
            ns8[tw] = __bfloat162bfloat162(sp[tw][8]);
        }
        sp[tw] += 16;
#pragma unroll
        for (int j = 0; j < NT; j++) acc[tw][j][0] = acc[tw][j][1] = acc[tw][j][2] = acc[tw][j][3] = 0.f;
    }
    const bool work = valid[0];
    unsigned use_off = 0;

    for (int i0 = 0; i0 < L; i0 += kRing) {
#pragma unroll
        for (int d = 0; d < kRing; d++) {
            const int i = i0 + d;
            if (i >= L) break;
            cp_async_wait<S - 2>();
            __syncthreads();
            if (i + S - 1 < L) issue_stage(i + S - 1, issue_off);
            cp_async_commit();
            issue_off = issue_off + stage_bytes == S * stage_bytes ? 0u : issue_off + stage_bytes;
            const unsigned xs = smem_u + use_off + xoff;
            use_off = use_off + stage_bytes == S * stage_bytes ? 0u : use_off + stage_bytes;
            if (!work || i >= len) continue;
            uint4 wv[TW];
#pragma unroll
            for (int tw = 0; tw < TW; tw++) {
                wv[tw] = wbuf[tw][d];
                if (valid[tw] && i < nload) wbuf[tw][d] = ldg_stream(qp[tw]);
                qp[tw] += 32;
            }
            if ((i & 1) == 0) {
#pragma unroll
                for (int tw = 0; tw < TW; tw++) {
                    sg[tw] = ns[tw];
                    sg8[tw] = ns8[tw];
                    if (valid[tw] && (i >> 1) < ngroup) {
                        ns[tw] = __bfloat162bfloat162(sp[tw][0]);
                        ns8[tw] = __bfloat162bfloat162(sp[tw][8]);
                    }
                    sp[tw] += 16;
                }
            }
#pragma unroll
            for (int hp = 0; hp < 2; hp++) {  // two k-step pairs per chunk
                uint32_t bq[NT][4];
#pragma unroll
                for (int nt = 0; nt < NT; nt++) ldmatrix_x4(bq[nt], xs + nt * 8 * kRowBytes + hp * 64);
#pragma unroll
                for (int h = 0; h < 2; h++) {
#pragma unroll
                    for (int tw = 0; tw < TW; tw++) {
                        if (!valid[tw]) continue;
                        const int wi = hp * 2 + h;
                        const uint32_t word = wi == 0 ? wv[tw].x : wi == 1 ? wv[tw].y : wi == 2 ? wv[tw].z : wv[tw].w;
                        uint32_t a[4];
                        dequant_w4(word, a);
#pragma unroll
                        for (int qq = 0; qq < 4; qq++) {
                            bf162 hh = __hmul2(*reinterpret_cast<bf162*>(&a[qq]), (qq & 1) ? sg8[tw] : sg[tw]);
                            a[qq] = *reinterpret_cast<uint32_t*>(&hh);
                        }
#pragma unroll
                        for (int nt = 0; nt < NT; nt++)
                            mma_bf16(acc[tw][nt], a, bq[nt][2 * h], bq[nt][2 * h + 1]);
                    }
                }
            }
        }
    }
    cp_async_wait<0>();
    __syncthreads();

    // slice reduction in order w = 0..nw-1 (as reduce_store_nt: slice 0's acc += slices 1..nw-1)
    float* red = reinterpret_cast<float*>(smem);   // [nw][TPB][NT][4][32]
    if (w != 0) {
#pragma unroll
        for (int tw = 0; tw < TW; tw++)
            if (valid[tw])
#pragma unroll
                for (int j = 0; j < NT; j++)
#pragma unroll
                    for (int c = 0; c < 4; c++)
                        red[((((size_t)w * TPB + tl0 + tw) * NT + j) * 4 + c) * 32 + lane] = acc[tw][j][c];
    }
    __syncthreads();
    if (w != 0) return;
#pragma unroll
    for (int tw = 0; tw < TW; tw++) {
        if (!valid[tw]) continue;
        for (int ww = 1; ww < nw; ww++)
#pragma unroll
            for (int j = 0; j < NT; j++)
#pragma unroll
                for (int c = 0; c < 4; c++)
                    acc[tw][j][c] += red[((((size_t)ww * TPB + tl0 + tw) * NT + j) * 4 + c) * 32 + lane];
        const int row0 = (blockIdx.x * TPB + tl0 + tw) * 16;
#pragma unroll
        for (int j = 0; j < NT; j++) {
            const int n0 = 8 * j + 2 * t, n1 = n0 + 1;
            if (glu) {
                bf16* act = reinterpret_cast<bf16*>(y);
                const int col = row0 / 2 + g;
                if (n0 < m) act[(size_t)n0 * ldy + col] = __float2bfloat16(silu_f(acc[tw][j][0]) * acc[tw][j][2]);
                if (n1 < m) act[(size_t)n1 * ldy + col] = __float2bfloat16(silu_f(acc[tw][j][1]) * acc[tw][j][3]);
                continue;
            }
            if (n0 < m) {
                y[(size_t)n0 * ldy + row0 + g] = acc[tw][j][0];
                y[(size_t)n0 * ldy + row0 + g + 8] = acc[tw][j][2];
            }
            if (n1 < m) {
                y[(size_t)n1 * ldy + row0 + g] = acc[tw][j][1];
                y[(size_t)n1 * ldy + row0 + g + 8] = acc[tw][j][3];
            }
        }
    }
}


// ---------------------------------------------------------------- fp16-accumulate variant (XH input)
// Same block / warp / K-slice structure and staging as skinny_w4_kernel; the staged rows are XH rows
// (gemv_h16.cuh), so a lane reads its B fragments of a chunk straight from shared memory (two 16-byte
// loads at row 8 nt + g, offset 32 t: conflict-free with the 144-byte row pitch) and the math is the
// shared xh_dequant / xh_mma / xh_flush of the other H16 kernels (bit-identical results for every m).
template <int NT, int S>
__global__ void __launch_bounds__(kMaxThreads) skinny_w4h_kernel(
    const uint4* __restrict__ q, const bf16* __restrict__ s, int K, const __half* __restrict__ xh,
    int ldx, float* __restrict__ y, int ldy, int m, bool glu, int tiles, int TPB, int nw,
    SkinnyWarps wt, int L, __half* __restrict__ yh, int ldyh) {
    constexpr int TW = 2;
    constexpr int kRing = 2;
    extern __shared__ __align__(16) unsigned char smem[];
    __shared__ int slice_c0[kMaxSlices], slice_len[kMaxSlices];
    const unsigned smem_u = (unsigned)__cvta_generic_to_shared(smem);
    const int tid = threadIdx.x, lane = tid & 31, wid = tid >> 5;
    const int nthreads = blockDim.x;
    const int g = lane >> 2, t = lane & 3;
    const int groups = K >> 7, chunks = K >> 6;
    const int rows = 8 * NT;
    const int slice_bytes = rows * kRowBytes;
    const int stage_bytes = nw * slice_bytes;

    if (tid < nw) {
        slice_c0[tid] = 2 * (tid * groups / nw);
        slice_len[tid] = 2 * ((tid + 1) * groups / nw - tid * groups / nw);
    }
    for (int i = tid * 16; i < S * stage_bytes; i += nthreads * 16)
        *reinterpret_cast<uint4*>(smem + i) = make_uint4(0, 0, 0, 0);
    __syncthreads();

    const int per_slice = m * 8;
    const int nsets = nthreads / per_slice;
    const int myset = tid / per_slice, r = tid - myset * per_slice;
    const bool loader = myset < nsets;
    const __half* xsrc = xh + (size_t)(r >> 3) * ldx + (r & 7) * 8;
    const unsigned xdst = smem_u + (r >> 3) * kRowBytes + (r & 7) * 16;
    auto issue_stage = [&](int i, unsigned stage_off) {
        if (!loader) return;
        for (int sw = myset; sw < nw; sw += nsets)
            if (i < slice_len[sw])
                cp_async16(xdst + stage_off + sw * slice_bytes, xsrc + (slice_c0[sw] + i) * kChunkK);
    };
    unsigned issue_off = 0;
    for (int i = 0; i < S - 1; i++) {
        if (i < L) issue_stage(i, issue_off);
        cp_async_commit();
        issue_off = issue_off + stage_bytes == S * stage_bytes ? 0u : issue_off + stage_bytes;
    }

    const int w = wt.w[wid], tl0 = wt.tile0[wid], cnt = wt.cnt[wid];
    const int gb = w * groups / nw, ge = (w + 1) * groups / nw;
    const int len = cnt > 0 ? 2 * (ge - gb) : 0;
    const int nload = len - kRing;
    const int ngroup = ge - gb - 1;
    // this lane's B fragments: row 8 nt + g of the slice stage, bytes [32 t, 32 t + 32)
    const unsigned xoff = w * slice_bytes + g * kRowBytes + t * 32;
    bool valid[TW];
    const uint4* qp[TW];
    const bf16* sp[TW];
    uint4 wbuf[TW][kRing];
    float sg[TW], sg8[TW], nsg[TW], nsg8[TW];
    float acc[TW][NT][4];
#pragma unroll
    for (int tw = 0; tw < TW; tw++) {
        const int tile = blockIdx.x * TPB + tl0 + tw;
        valid[tw] = tw < cnt && tile < tiles;
        const int tt = valid[tw] ? tile : 0;
        qp[tw] = q + ((size_t)tt * chunks + 2 * gb) * 32 + lane;
        sp[tw] = s + ((size_t)tt * groups + gb) * 16 + g;
#pragma unroll
        for (int d = 0; d < kRing; d++) {
            wbuf[tw][d] = make_uint4(0, 0, 0, 0);
            if (valid[tw] && d < len) wbuf[tw][d] = ldg_stream(qp[tw] + d * 32);
        }
        qp[tw] += kRing * 32;
        sg[tw] = sg8[tw] = nsg[tw] = nsg8[tw] = 0.f;
        if (valid[tw] && len > 0) {
            nsg[tw] = xh_scale(sp[tw][0]);
            nsg8[tw] = xh_scale(sp[tw][8]);
        }
        sp[tw] += 16;
#pragma unroll
        for (int j = 0; j < NT; j++) acc[tw][j][0] = acc[tw][j][1] = acc[tw][j][2] = acc[tw][j][3] = 0.f;
    }
    const bool work = valid[0];
    unsigned use_off = 0;

    for (int i0 = 0; i0 < L; i0 += kRing) {
#pragma unroll
        for (int d = 0; d < kRing; d++) {
            const int i = i0 + d;
            if (i >= L) break;
            cp_async_wait<S - 2>();
            __syncthreads();
            if (i + S - 1 < L) issue_stage(i + S - 1, issue_off);
            cp_async_commit();
            issue_off = issue_off + stage_bytes == S * stage_bytes ? 0u : issue_off + stage_bytes;
            const unsigned xs = smem_u + use_off + xoff;
            use_off = use_off + stage_bytes == S * stage_bytes ? 0u : use_off + stage_bytes;
            if (!work || i >= len) continue;
            uint4 wv[TW];
#pragma unroll
            for (int tw = 0; tw < TW; tw++) {
                wv[tw] = wbuf[tw][d];
                if (valid[tw] && i < nload) wbuf[tw][d] = ldg_stream(qp[tw]);
                qp[tw] += 32;
            }
            if ((i & 1) == 0) {
#pragma unroll
                for (int tw = 0; tw < TW; tw++) {
                    sg[tw] = nsg[tw];
                    sg8[tw] = nsg8[tw];
                    if (valid[tw] && (i >> 1) < ngroup) {
                        nsg[tw] = xh_scale(sp[tw][0]);
                        nsg8[tw] = xh_scale(sp[tw][8]);
                    }
                    sp[tw] += 16;
                }
            }
            uint4 bq[NT][2];
#pragma unroll
            for (int nt = 0; nt < NT; nt++) {
                const unsigned a = xs + nt * 8 * kRowBytes;
                asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];"
                             : "=r"(bq[nt][0].x), "=r"(bq[nt][0].y), "=r"(bq[nt][0].z), "=r"(bq[nt][0].w) : "r"(a));
                asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];"
                             : "=r"(bq[nt][1].x), "=r"(bq[nt][1].y), "=r"(bq[nt][1].z), "=r"(bq[nt][1].w) : "r"(a + 16));
            }
            uint32_t h[TW][NT][2];
#pragma unroll
            for (int tw = 0; tw < TW; tw++)
#pragma unroll
                for (int nt = 0; nt < NT; nt++) h[tw][nt][0] = h[tw][nt][1] = 0u;
#pragma unroll
            for (int j = 0; j < 4; j++) {
                const uint32_t words[4] = {wv[0].x, wv[0].y, wv[0].z, wv[0].w};
                const uint32_t words1[4] = {wv[1].x, wv[1].y, wv[1].z, wv[1].w};
#pragma unroll
                for (int tw = 0; tw < TW; tw++) {
                    if (!valid[tw]) continue;
                    uint32_t a[4];
                    xh_dequant(tw == 0 ? words[j] : words1[j], a);
#pragma unroll
                    for (int nt = 0; nt < NT; nt++) {
                        const uint4& bb = bq[nt][j >> 1];
                        xh_mma(h[tw][nt], a, (j & 1) ? bb.z : bb.x, (j & 1) ? bb.w : bb.y);
                    }
                }
            }
#pragma unroll
            for (int tw = 0; tw < TW; tw++) {
                if (!valid[tw]) continue;
#pragma unroll
                for (int nt = 0; nt < NT; nt++) xh_flush(acc[tw][nt], h[tw][nt], sg[tw], sg8[tw]);
            }
        }
    }
    cp_async_wait<0>();
    __syncthreads();

    float* red = reinterpret_cast<float*>(smem);
    if (w != 0) {
#pragma unroll
        for (int tw = 0; tw < TW; tw++)
            if (valid[tw])
#pragma unroll
                for (int j = 0; j < NT; j++)
#pragma unroll
                    for (int c = 0; c < 4; c++)
                        red[((((size_t)w * TPB + tl0 + tw) * NT + j) * 4 + c) * 32 + lane] = acc[tw][j][c];
    }
    __syncthreads();
    if (w != 0) return;
#pragma unroll
    for (int tw = 0; tw < TW; tw++) {
        if (!valid[tw]) continue;
        for (int ww = 1; ww < nw; ww++)
#pragma unroll
            for (int j = 0; j < NT; j++)
#pragma unroll
                for (int c = 0; c < 4; c++)
                    acc[tw][j][c] += red[((((size_t)ww * TPB + tl0 + tw) * NT + j) * 4 + c) * 32 + lane];
        const int row0 = (blockIdx.x * TPB + tl0 + tw) * 16;
#pragma unroll
        for (int j = 0; j < NT; j++) {
            const int n0 = 8 * j + 2 * t, n1 = n0 + 1;
            if (glu) {
                bf16* act = reinterpret_cast<bf16*>(y);
                const int col = row0 / 2 + g;
                const float v0 = silu_f(acc[tw][j][0]) * acc[tw][j][2], v1 = silu_f(acc[tw][j][1]) * acc[tw][j][3];
                if (n0 < m) act[(size_t)n0 * ldy + col] = __float2bfloat16(v0);
                if (n1 < m) act[(size_t)n1 * ldy + col] = __float2bfloat16(v1);
                if (yh) {
                    if (n0 < m) store_xh(yh + (size_t)n0 * ldyh, col, v0);
                    if (n1 < m) store_xh(yh + (size_t)n1 * ldyh, col, v1);
                }
                continue;
            }
            if (n0 < m) {
                y[(size_t)n0 * ldy + row0 + g] = acc[tw][j][0];
                y[(size_t)n0 * ldy + row0 + g + 8] = acc[tw][j][2];
            }
            if (n1 < m) {
                y[(size_t)n1 * ldy + row0 + g] = acc[tw][j][1];
                y[(size_t)n1 * ldy + row0 + g + 8] = acc[tw][j][3];
            }
        }
    }
}

template <int NT, int S>
void set_attr_h() {
    CUDA_CHECK(cudaFuncSetAttribute(skinny_w4h_kernel<NT, S>,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize, kMaxSmem));
}
template <int NT, int S>
void launch_skinny_h(const W4& w, const __half* xh, int ldx, float* y, int ldy, int m, int tiles,
                     int TPB, int nw, int warps, const SkinnyWarps& wt, int L, size_t smem,
                     cudaStream_t st, __half* yh, int ldyh) {
    const int blocks = (tiles + TPB - 1) / TPB;
    skinny_w4h_kernel<NT, S><<<blocks, warps * 32, smem, st>>>(w.q, w.s, w.k, xh, ldx, y, ldy, m, w.glu,
                                                               tiles, TPB, nw, wt, L, yh, ldyh);
}

int skinny_env(const char* name, int def) {
    return env_int(name, def);
}


template <int NT, int S>
void set_attr() {
    CUDA_CHECK(cudaFuncSetAttribute(skinny_w4_kernel<NT, S>,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize, kMaxSmem));
}

template <int NT, int S>
void launch_skinny(const W4& w, const bf16* x, int ldx, float* y, int ldy, int m, int tiles, int TPB,
                   int nw, int warps, const SkinnyWarps& wt, int L, size_t smem, cudaStream_t st) {
    const int blocks = (tiles + TPB - 1) / TPB;
    skinny_w4_kernel<NT, S><<<blocks, warps * 32, smem, st>>>(w.q, w.s, w.k, x, ldx, y, ldy, m,
                                                              w.glu, tiles, TPB, nw, wt, L);
}

// Warp table: pairs of tiles within a slice, singles to pad the warp count to a multiple of 4
// (at most 16 warps). Returns the warp count or 0 if the work does not fit.
int build_warps(int TPB, int nw, SkinnyWarps& wt) {
    const int ntasks = TPB * nw;
    const int warps = ((ntasks + 1) / 2 + 3) / 4 * 4;
    if (warps > 16) return 0;
    int npairs = ntasks - warps;          // pairs needed so that pairs + singles == warps
    if (npairs > nw * (TPB / 2)) return 0;
    int k = 0;
    for (int w = 0; w < nw; w++) {
        int tl = 0;
        while (tl < TPB) {
            const int cnt = (tl + 1 < TPB && npairs > 0) ? 2 : 1;
            if (cnt == 2) npairs--;
            wt.w[k] = (unsigned char)w;
            wt.tile0[k] = (unsigned char)tl;
            wt.cnt[k] = (unsigned char)cnt;
            k++;
            tl += cnt;
        }
    }
    CHECK(k == warps, "skinny warp table %d != %d", k, warps);
    return warps;
}

}  // namespace

// Per-device one-time setup (opt-in shared memory); must run outside stream capture.
void skinny_init() {
    static bool done[16] = {};
    int dev;
    CUDA_CHECK(cudaGetDevice(&dev));
    if (done[dev]) return;
    done[dev] = true;
    set_attr<1, 3>(); set_attr<1, 4>(); set_attr<1, 6>();
    set_attr<2, 3>(); set_attr<2, 4>(); set_attr<2, 6>();
    set_attr<3, 3>(); set_attr<3, 4>(); set_attr<3, 6>();
    set_attr<4, 3>(); set_attr<4, 4>(); set_attr<4, 6>();
    set_attr_h<1, 3>(); set_attr_h<1, 4>(); set_attr_h<1, 6>();
    set_attr_h<2, 3>(); set_attr_h<2, 4>(); set_attr_h<2, 6>();
    set_attr_h<3, 3>(); set_attr_h<3, 4>(); set_attr_h<3, 6>();
    set_attr_h<4, 3>(); set_attr_h<4, 4>(); set_attr_h<4, 6>();
}

// Engine policy: on by default (QE_SKINNY=0 disables) for m >= QE_SKINNY_MIN_M (17) rows on matrices
// with >= QE_SKINNY_MIN_TILES (512) 16-row tiles - the regime where it beats gemv_w4 (measured).
bool skinny_use(int n, int m) {
    static const int on = skinny_env("QE_SKINNY", 1);
    static const int min_m = skinny_env("QE_SKINNY_MIN_M", 17);
    static const int min_tiles = skinny_env("QE_SKINNY_MIN_TILES", 512);
    return on && m >= min_m && n / 16 >= min_tiles;
}

bool skinny_w4(const W4& w, const bf16* x, int ldx, float* y, int ldy, int m, cudaStream_t st) {
    if (m < 1 || m > 32 || w.n % 16 || w.k % 128 || ldx % 8) return false;
    const int tiles = w.n / 16, groups = w.k / 128;
    const int nw = pick_warps(tiles, groups, true);
    const int NT = (m + 7) / 8;
    static const int blocks_env = skinny_env("QE_SKINNY_BLOCKS", kSMs);
    static const int s_env = skinny_env("QE_SKINNY_S", 0);
    const int TPB = (tiles + blocks_env - 1) / blocks_env;
    if (TPB < 1 || TPB > 255) return false;
    SkinnyWarps wt = {};
    const int warps = build_warps(TPB, nw, wt);
    if (warps == 0 || warps * 32 > kMaxThreads || warps * 32 < 8 * m) return false;
    int L = 0;
    for (int ww = 0; ww < nw; ww++) L = std::max(L, 2 * ((ww + 1) * groups / nw - ww * groups / nw));
    const size_t stage = (size_t)nw * 8 * NT * kRowBytes;
    const size_t red = (size_t)nw * TPB * NT * 4 * 32 * sizeof(float);
    int S = s_env ? std::min(6, std::max(3, s_env)) : 4;
    while (S > 3 && S * stage > kMaxSmem) S = S == 6 ? 4 : 3;
    const size_t smem = std::max(S * stage, red);
    if (smem > kMaxSmem) return false;
#define QE_SK(NT_, S_) launch_skinny<NT_, S_>(w, x, ldx, y, ldy, m, tiles, TPB, nw, warps, wt, L, smem, st)
#define QE_SKS(NT_) if (S == 6) QE_SK(NT_, 6); else if (S == 4) QE_SK(NT_, 4); else QE_SK(NT_, 3)
    switch (NT) {
        case 1: QE_SKS(1); break;
        case 2: QE_SKS(2); break;
        case 3: QE_SKS(3); break;
        case 4: QE_SKS(4); break;
        default: return false;
    }
#undef QE_SKS
#undef QE_SK
    return true;
}

bool skinny_w4h(const W4& w, const __half* xh, int ldx, float* y, int ldy, int m, cudaStream_t st,
                __half* yh, int ldyh) {
    if (m < 1 || m > 32 || w.n % 16 || w.k % 128 || ldx % 64) return false;
    const int tiles = w.n / 16, groups = w.k / 128;
    const int nw = pick_warps(tiles, groups, true);
    const int NT = (m + 7) / 8;
    static const int blocks_env = skinny_env("QE_SKINNY_BLOCKS", kSMs);
    static const int s_env = skinny_env("QE_SKINNY_S", 0);
    const int TPB = (tiles + blocks_env - 1) / blocks_env;
    if (TPB < 1 || TPB > 255) return false;
    SkinnyWarps wt = {};
    const int warps = build_warps(TPB, nw, wt);
    if (warps == 0 || warps * 32 > kMaxThreads || warps * 32 < 8 * m) return false;
    int L = 0;
    for (int ww = 0; ww < nw; ww++) L = std::max(L, 2 * ((ww + 1) * groups / nw - ww * groups / nw));
    const size_t stage = (size_t)nw * 8 * NT * kRowBytes;
    const size_t red = (size_t)nw * TPB * NT * 4 * 32 * sizeof(float);
    int S = s_env ? std::min(6, std::max(3, s_env)) : 4;
    while (S > 3 && S * stage > kMaxSmem) S = S == 6 ? 4 : 3;
    const size_t smem = std::max(S * stage, red);
    if (smem > kMaxSmem) return false;
#define QE_SKH(NT_, S_) launch_skinny_h<NT_, S_>(w, xh, ldx, y, ldy, m, tiles, TPB, nw, warps, wt, L, smem, st, yh, ldyh)
#define QE_SKHS(NT_) if (S == 6) QE_SKH(NT_, 6); else if (S == 4) QE_SKH(NT_, 4); else QE_SKH(NT_, 3)
    switch (NT) {
        case 1: QE_SKHS(1); break;
        case 2: QE_SKHS(2); break;
        case 3: QE_SKHS(3); break;
        case 4: QE_SKHS(4); break;
        default: return false;
    }
#undef QE_SKHS
#undef QE_SKH
    return true;
}
