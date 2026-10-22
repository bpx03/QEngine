#include "gemv.cuh"
#include "gemv_h16.cuh"
#include "ar_fused.cuh"

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

// Unsigned nibbles (w+8) -> bf16x2 (w) without int->float conversion: OR the nibble into
// the mantissa of 128.0 (0x4300) giving 128+u exactly, then subtract 136.
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

__device__ __forceinline__ uint32_t load_x2(const bf16* p, bool ok) {
    return ok ? *reinterpret_cast<const uint32_t*>(p) : 0u;
}

constexpr int kMaxWarps = 8;

__device__ __forceinline__ float silu_f(float x) { return x / (1.f + __expf(-x)); }

__device__ __forceinline__ void reduce_store(float acc[4], int nwarps, float* y, int ldy, int m,
                                             int row0, bool glu = false, __half* yh = nullptr,
                                             int ldyh = 0) {
    __shared__ float red[kMaxWarps][32][4];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    if (nwarps > 1) {
#pragma unroll
        for (int i = 0; i < 4; i++) red[warp][lane][i] = acc[i];
        __syncthreads();
        if (warp != 0) return;
        for (int w = 1; w < nwarps; w++)
#pragma unroll
            for (int i = 0; i < 4; i++) acc[i] += red[w][lane][i];
    }
    const int g = lane >> 2, t = lane & 3;
    const int n0 = 2 * t, n1 = 2 * t + 1;
    if (glu) {  // rows g / g + 8 are gate / up of activation column row0 / 2 + g
        bf16* act = reinterpret_cast<bf16*>(y);
        const int col = row0 / 2 + g;
        if (n0 < m) act[(size_t)n0 * ldy + col] = __float2bfloat16(silu_f(acc[0]) * acc[2]);
        if (n1 < m) act[(size_t)n1 * ldy + col] = __float2bfloat16(silu_f(acc[1]) * acc[3]);
        if (yh) {
            if (n0 < m) store_xh(yh + (size_t)n0 * ldyh, col, silu_f(acc[0]) * acc[2]);
            if (n1 < m) store_xh(yh + (size_t)n1 * ldyh, col, silu_f(acc[1]) * acc[3]);
        }
        return;
    }
    if (n0 < m) {
        y[(size_t)n0 * ldy + row0 + g] = acc[0];
        y[(size_t)n0 * ldy + row0 + g + 8] = acc[2];
    }
    if (n1 < m) {
        y[(size_t)n1 * ldy + row0 + g] = acc[1];
        y[(size_t)n1 * ldy + row0 + g + 8] = acc[3];
    }
}

// Reduction + store of NT n8 tiles (tokens 8*nt + n) for wide GEMV.
template <int NT, int MAXW = kMaxWarps>
__device__ __forceinline__ void reduce_store_nt(float (&acc)[NT][4], int nwarps, float* y, int ldy,
                                                int m, int row0, bool glu, __half* yh = nullptr,
                                                int ldyh = 0) {
    if constexpr (NT == 1) {
        reduce_store(acc[0], nwarps, y, ldy, m, row0, glu, yh, ldyh);
        return;
    }
    __shared__ float red[MAXW][NT][4][32];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    if (nwarps > 1) {
#pragma unroll
        for (int j = 0; j < NT; j++)
#pragma unroll
            for (int i = 0; i < 4; i++) red[warp][j][i][lane] = acc[j][i];
        __syncthreads();
        if (warp != 0) return;
        for (int w = 1; w < nwarps; w++)
#pragma unroll
            for (int j = 0; j < NT; j++)
#pragma unroll
                for (int i = 0; i < 4; i++) acc[j][i] += red[w][j][i][lane];
    }
    const int g = lane >> 2, t = lane & 3;
#pragma unroll
    for (int j = 0; j < NT; j++) {
        const int n0 = 8 * j + 2 * t, n1 = n0 + 1;
        if (glu) {
            bf16* act = reinterpret_cast<bf16*>(y);
            const int col = row0 / 2 + g;
            if (n0 < m) act[(size_t)n0 * ldy + col] = __float2bfloat16(silu_f(acc[j][0]) * acc[j][2]);
            if (n1 < m) act[(size_t)n1 * ldy + col] = __float2bfloat16(silu_f(acc[j][1]) * acc[j][3]);
            if (yh) {
                if (n0 < m) store_xh(yh + (size_t)n0 * ldyh, col, silu_f(acc[j][0]) * acc[j][2]);
                if (n1 < m) store_xh(yh + (size_t)n1 * ldyh, col, silu_f(acc[j][1]) * acc[j][3]);
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

// One block per 16-row tile; its warps split K. B-fragment column n = token index (lane/4);
// NT n8 tiles cover tokens [0, 8*NT), each dequantized weight fragment feeds NT mma.
template <int NT>
__global__ void gemv_w4_kernel(const uint4* __restrict__ q, const bf16* __restrict__ s, int K,
                               const bf16* __restrict__ x, int ldx, float* __restrict__ y, int ldy,
                               int m, bool glu) {
    const int r = blockIdx.x, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int nwarps = blockDim.x >> 5;
    const int g = lane >> 2, t = lane & 3;
    const int groups = K >> 7;
    const int gb = warp * groups / nwarps, ge = (warp + 1) * groups / nwarps;

    const uint4* qp = q + (size_t)r * (K >> 6) * 32 + lane;
    const bf16* sp = s + (size_t)r * groups * 16;
    bool act[NT];
    const bf16* xr[NT];
#pragma unroll
    for (int j = 0; j < NT; j++) {
        act[j] = 8 * j + g < m;
        xr[j] = x + (size_t)(act[j] ? 8 * j + g : 0) * ldx + 2 * t;
    }

    // Weights are dequantized to bf16 with the scale applied and rounded (like the reference
    // decompression and the prefill GEMM), so decode and prefill see identical weights.
    float acc[NT][4];
#pragma unroll
    for (int j = 0; j < NT; j++) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.f;
    // software pipeline: the next group's weights and scales are in flight while the current
    // group is dequantized, so each warp keeps two groups of loads outstanding
    uint4 w0, w1;
    bf16 s0 = __float2bfloat16(0.f), s8 = s0;
    if (gb < ge) {
        w0 = ldg_stream(qp + (size_t)(2 * gb) * 32);
        w1 = ldg_stream(qp + (size_t)(2 * gb + 1) * 32);
        s0 = sp[gb * 16 + g];
        s8 = sp[gb * 16 + g + 8];
    }
    for (int gi = gb; gi < ge; gi++) {
        uint4 n0 = w0, n1 = w1;
        bf16 ns0 = s0, ns8 = s8;
        if (gi + 1 < ge) {
            n0 = ldg_stream(qp + (size_t)(2 * gi + 2) * 32);
            n1 = ldg_stream(qp + (size_t)(2 * gi + 3) * 32);
            ns0 = sp[(gi + 1) * 16 + g];
            ns8 = sp[(gi + 1) * 16 + g + 8];
        }
        const bf162 sg = __bfloat162bfloat162(s0);
        const bf162 sg8 = __bfloat162bfloat162(s8);
        const uint32_t words[8] = {w0.x, w0.y, w0.z, w0.w, w1.x, w1.y, w1.z, w1.w};
#pragma unroll
        for (int j = 0; j < 8; j++) {
            const int k0 = gi * 128 + j * 16;
            uint32_t a[4];
            dequant_w4(words[j], a);
#pragma unroll
            for (int q = 0; q < 4; q++) {
                bf162 h = __hmul2(*reinterpret_cast<bf162*>(&a[q]), (q & 1) ? sg8 : sg);
                a[q] = *reinterpret_cast<uint32_t*>(&h);
            }
#pragma unroll
            for (int nt = 0; nt < NT; nt++)
                mma_bf16(acc[nt], a, load_x2(xr[nt] + k0, act[nt]), load_x2(xr[nt] + k0 + 8, act[nt]));
        }
        w0 = n0; w1 = n1; s0 = ns0; s8 = ns8;
    }
    reduce_store_nt<NT>(acc, nwarps, y, ldy, m, r * 16, glu);
}

// Activations repacked into B-fragment order for the wide GEMV: uint4 at
// ((nt * (K / 32) + kp) * 32 + lane) holds, for token 8 * nt + lane / 4 and t = lane % 4, the
// bf16 pairs at k = 32 kp + {2t, 8 + 2t, 16 + 2t, 24 + 2t} (b0/b1 of two k16 steps).
constexpr int kXpackMaxK = 32768;
__device__ __forceinline__ uint32_t bf2_to_h2(uint32_t v) {
    const float2 f = __bfloat1622float2(*reinterpret_cast<bf162*>(&v));
    __half2 h = __floats2half2_rn(f.x, f.y);
    return *reinterpret_cast<uint32_t*>(&h);
}

__global__ void xpack_kernel(const bf16* __restrict__ x, int ldx, int K, int m, int nt_total,
                             uint4* __restrict__ xp, bool f16) {
    const int kps = K >> 5;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;  // over nt_total * kps * 32
    if (i >= nt_total * kps * 32) return;
    const int lane = i & 31, kp = (i >> 5) % kps, nt = (i >> 5) / kps;
    const int tok = 8 * nt + (lane >> 2), t = lane & 3;
    uint4 v = make_uint4(0, 0, 0, 0);
    if (tok < m) {
        const uint32_t* r = reinterpret_cast<const uint32_t*>(x + (size_t)tok * ldx + kp * 32 + 2 * t);
        v.x = r[0]; v.y = r[4]; v.z = r[8]; v.w = r[12];
        if (f16) { v.x = bf2_to_h2(v.x); v.y = bf2_to_h2(v.y); v.z = bf2_to_h2(v.z); v.w = bf2_to_h2(v.w); }
    }
    xp[i] = v;
}

// RT 16-row tiles per block (each warp computes all of them for its K range, so every packed
// activation fragment loaded is reused RT times).
template <int NT, int RT>
__global__ void gemv_w4p_kernel(const uint4* __restrict__ q, const bf16* __restrict__ s, int K,
                                const uint4* __restrict__ xp, float* __restrict__ y, int ldy, int m,
                                bool glu) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int nwarps = blockDim.x >> 5;
    const int g = lane >> 2;
    const int groups = K >> 7, kps = K >> 5;
    const int gb = warp * groups / nwarps, ge = (warp + 1) * groups / nwarps;
    const uint4* xl = xp + lane;
    float acc[RT][NT][4];
#pragma unroll
    for (int rt = 0; rt < RT; rt++)
#pragma unroll
        for (int j = 0; j < NT; j++) acc[rt][j][0] = acc[rt][j][1] = acc[rt][j][2] = acc[rt][j][3] = 0.f;
    for (int gi = gb; gi < ge; gi++) {
        uint4 w0[RT], w1[RT];
        bf162 sg[RT], sg8[RT];
#pragma unroll
        for (int rt = 0; rt < RT; rt++) {
            const int r = blockIdx.x * RT + rt;
            const uint4* qp = q + (size_t)r * (K >> 6) * 32 + lane;
            const bf16* sp = s + (size_t)r * groups * 16;
            w0[rt] = ldg_stream(qp + (size_t)(2 * gi) * 32);
            w1[rt] = ldg_stream(qp + (size_t)(2 * gi + 1) * 32);
            sg[rt] = __bfloat162bfloat162(sp[gi * 16 + g]);
            sg8[rt] = __bfloat162bfloat162(sp[gi * 16 + g + 8]);
        }
#pragma unroll
        for (int jp = 0; jp < 4; jp++) {
            uint4 b[NT];
#pragma unroll
            for (int nt = 0; nt < NT; nt++) b[nt] = xl[((size_t)nt * kps + gi * 4 + jp) * 32];
#pragma unroll
            for (int rt = 0; rt < RT; rt++) {
                const uint32_t words[8] = {w0[rt].x, w0[rt].y, w0[rt].z, w0[rt].w,
                                           w1[rt].x, w1[rt].y, w1[rt].z, w1[rt].w};
#pragma unroll
                for (int h = 0; h < 2; h++) {
                    uint32_t a[4];
                    dequant_w4(words[2 * jp + h], a);
#pragma unroll
                    for (int qq = 0; qq < 4; qq++) {
                        bf162 hh = __hmul2(*reinterpret_cast<bf162*>(&a[qq]), (qq & 1) ? sg8[rt] : sg[rt]);
                        a[qq] = *reinterpret_cast<uint32_t*>(&hh);
                    }
#pragma unroll
                    for (int nt = 0; nt < NT; nt++)
                        mma_bf16(acc[rt][nt], a, h ? b[nt].z : b[nt].x, h ? b[nt].w : b[nt].y);
                }
            }
        }
    }
#pragma unroll
    for (int rt = 0; rt < RT; rt++) {
        if (rt) __syncthreads();
        reduce_store_nt<NT>(acc[rt], nwarps, y, ldy, m, (blockIdx.x * RT + rt) * 16, glu);
    }
}


// ---------------------------------------------------------------- fp16-accumulate INT4 GEMV (XH input)
// One block per 16-row tile, nw warps split K exactly like gemv_w4_kernel; NT n8 tiles cover tokens
// [0, 8 NT). x comes in the XH layout (gemv_h16.cuh): per chunk a lane loads its 32 bytes of the token
// row directly (two 16-byte loads). Used for m <= 8 (NT = 1): <= 64 registers like the bf16 kernel, so the
// blocks per SM and the wave structure stay the same.
template <int NT>
__global__ void __launch_bounds__(256, 4) gemv_w4h_kernel(const uint4* __restrict__ q, const bf16* __restrict__ s, int K,
                const __half* __restrict__ xh, int ldx, float* __restrict__ y, int ldy, int m, bool glu,
                __half* __restrict__ yh, int ldyh) {
    const int r = blockIdx.x, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int nwarps = blockDim.x >> 5;
    const int g = lane >> 2, t = lane & 3;
    const int groups = K >> 7, chunks = K >> 6;
    const int gb = warp * groups / nwarps, ge = (warp + 1) * groups / nwarps;
    const uint4* qp = q + (size_t)r * chunks * 32 + lane;
    const bf16* sp = s + (size_t)r * groups * 16;
    const uint4* xr[NT];
#pragma unroll
    for (int j = 0; j < NT; j++) {
        const int tok = 8 * j + g < m ? 8 * j + g : 0;   // padding rows read row 0 (results discarded)
        xr[j] = reinterpret_cast<const uint4*>(xh + (size_t)tok * ldx + t * 16);
    }
    float acc[NT][4];
#pragma unroll
    for (int j = 0; j < NT; j++) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.f;
    // NT == 1 (DRAM-bound): the next group's weights and scales are in flight while the current group
    // is computed, as in gemv_w4_kernel. NT >= 2: loads per chunk (fewer registers -> more warps per SM,
    // which the activation re-reads need more than the prefetch)
    constexpr bool kPrefetch = true;
    uint4 w0, w1;
    bf16 s0 = __float2bfloat16(0.f), s8 = s0;
    if (kPrefetch && gb < ge) {
        w0 = ldg_stream(qp + (size_t)(2 * gb) * 32);
        w1 = ldg_stream(qp + (size_t)(2 * gb + 1) * 32);
        s0 = sp[gb * 16 + g];
        s8 = sp[gb * 16 + g + 8];
    }
    for (int gi = gb; gi < ge; gi++) {
        uint4 n0, n1;
        bf16 ns0, ns8;
        if (kPrefetch) {
            n0 = w0; n1 = w1; ns0 = s0; ns8 = s8;
            if (gi + 1 < ge) {
                n0 = ldg_stream(qp + (size_t)(2 * gi + 2) * 32);
                n1 = ldg_stream(qp + (size_t)(2 * gi + 3) * 32);
                ns0 = sp[(gi + 1) * 16 + g];
                ns8 = sp[(gi + 1) * 16 + g + 8];
            }
        } else {
            s0 = sp[gi * 16 + g];
            s8 = sp[gi * 16 + g + 8];
        }
        const float sg = xh_scale(s0), sg8 = xh_scale(s8);
        uint32_t h[NT][2];
#pragma unroll
        for (int nt = 0; nt < NT; nt++) h[nt][0] = h[nt][1] = 0u;
#pragma unroll
        for (int c = 0; c < 2; c++) {
            const int ch = 2 * gi + c;
            const uint4 wc = kPrefetch ? (c ? w1 : w0) : ldg_stream(qp + (size_t)ch * 32);
            const uint32_t words[4] = {wc.x, wc.y, wc.z, wc.w};
            uint4 b[NT][2];
#pragma unroll
            for (int nt = 0; nt < NT; nt++) {
                b[nt][0] = xr[nt][(size_t)ch * 8];
                b[nt][1] = xr[nt][(size_t)ch * 8 + 1];
            }
#pragma unroll
            for (int j = 0; j < 4; j++) {
                uint32_t a[4];
                xh_dequant(words[j], a);
#pragma unroll
                for (int nt = 0; nt < NT; nt++) {
                    const uint4& bb = b[nt][j >> 1];
                    xh_mma(h[nt], a, (j & 1) ? bb.z : bb.x, (j & 1) ? bb.w : bb.y);
                }
            }
#pragma unroll
            for (int nt = 0; nt < NT; nt++) xh_flush(acc[nt], h[nt], sg, sg8);
        }
        if (kPrefetch) { w0 = n0; w1 = n1; s0 = ns0; s8 = ns8; }
    }
    reduce_store_nt<NT>(acc, nwarps, y, ldy, m, r * 16, glu, yh, ldyh);
}


// 9..32 rows: same math, no weight prefetch (per-chunk loads) - the activation re-reads (NT x 8 token
// rows per chunk) want warps more than loads in flight (measured: the prefetch structure above costs
// 85-136 registers at NT >= 2 and is 20-30% slower here).
// H16 caps the K slices at 5 for <= 512 tiles and wider matrices get <= 4 (pick_warps), so blocks have
// <= 160 threads and the reduction buffer is sized for 5 warps (10 KB at NT = 4 instead of 16 KB).
// Register budgets per NT: NT = 2 free (58), NT = 3 capped at 5 blocks/SM
// (72; dn_in's 352 tiles then run in one wave: 24 rows 38 -> 26 us), NT = 4 at 4 blocks/SM (96 instead of
// 80: down/qkv/dn_in at 27-32 rows -20%); NT = 3 uses the uncapped kernel (82) unless the cap saves a
// wave (see gemv_w4h).
constexpr int kW4hwMaxWarps = 5;
template <int NT>
__device__ __forceinline__ void w4hw_body(const uint4* __restrict__ q, const bf16* __restrict__ s, int K,
                                 const __half* __restrict__ xh, int ldx, float* __restrict__ y, int ldy,
                                 int m, bool glu, __half* __restrict__ yh, int ldyh) {
    const int r = blockIdx.x, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int nwarps = blockDim.x >> 5;
    const int g = lane >> 2, t = lane & 3;
    const int groups = K >> 7, chunks = K >> 6;
    const int gb = warp * groups / nwarps, ge = (warp + 1) * groups / nwarps;
    const uint4* qp = q + (size_t)r * chunks * 32 + lane;
    const bf16* sp = s + (size_t)r * groups * 16;
    const uint4* xr[NT];
#pragma unroll
    for (int j = 0; j < NT; j++) {
        const int tok = 8 * j + g < m ? 8 * j + g : 0;
        xr[j] = reinterpret_cast<const uint4*>(xh + (size_t)tok * ldx + t * 16);
    }
    float acc[NT][4];
#pragma unroll
    for (int j = 0; j < NT; j++) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.f;
    for (int gi = gb; gi < ge; gi++) {
        const float sg = xh_scale(sp[gi * 16 + g]), sg8 = xh_scale(sp[gi * 16 + g + 8]);
#pragma unroll
        for (int c = 0; c < 2; c++) {
            const int ch = 2 * gi + c;
            const uint4 w = ldg_stream(qp + (size_t)ch * 32);
            uint4 b[NT][2];
#pragma unroll
            for (int nt = 0; nt < NT; nt++) {
                b[nt][0] = xr[nt][(size_t)ch * 8];
                b[nt][1] = xr[nt][(size_t)ch * 8 + 1];
            }
            const uint32_t words[4] = {w.x, w.y, w.z, w.w};
            uint32_t h[NT][2];
#pragma unroll
            for (int nt = 0; nt < NT; nt++) h[nt][0] = h[nt][1] = 0u;
#pragma unroll
            for (int j = 0; j < 4; j++) {
                uint32_t a[4];
                xh_dequant(words[j], a);
#pragma unroll
                for (int nt = 0; nt < NT; nt++) {
                    const uint4& bb = b[nt][j >> 1];
                    xh_mma(h[nt], a, (j & 1) ? bb.z : bb.x, (j & 1) ? bb.w : bb.y);
                }
            }
#pragma unroll
            for (int nt = 0; nt < NT; nt++) xh_flush(acc[nt], h[nt], sg, sg8);
        }
    }
    reduce_store_nt<NT, kW4hwMaxWarps>(acc, nwarps, y, ldy, m, r * 16, glu, yh, ldyh);
}

// register budget per token-tile count (QE_W4HW_ATTR2/3/4 override, experiments only)
#if __has_include("w4hw_tune.h")
#include "w4hw_tune.h"
#endif
#ifndef QE_W4HW_ATTR2
#define QE_W4HW_ATTR2
#endif
#ifndef QE_W4HW_ATTR3
#define QE_W4HW_ATTR3 __launch_bounds__(kW4hwMaxWarps * 32, 5)
#endif
#ifndef QE_W4HW_ATTR4
#define QE_W4HW_ATTR4 __launch_bounds__(kW4hwMaxWarps * 32, 4)
#endif
#define QE_W4HW_KERNEL(NAME_, NT_, ATTR_)                                                                        \
    __global__ void ATTR_ gemv_w4hw##NAME_(const uint4* __restrict__ q, const bf16* __restrict__ s, int K,   \
                                         const __half* __restrict__ xh, int ldx, float* __restrict__ y,     \
                                         int ldy, int m, bool glu, __half* __restrict__ yh, int ldyh) {      \
        w4hw_body<NT_>(q, s, K, xh, ldx, y, ldy, m, glu, yh, ldyh);                                          \
    }
QE_W4HW_KERNEL(2, 2, QE_W4HW_ATTR2)
QE_W4HW_KERNEL(3, 3, QE_W4HW_ATTR3)
QE_W4HW_KERNEL(4, 4, QE_W4HW_ATTR4)
QE_W4HW_KERNEL(3w, 3, )  // NT = 3 without the cap, for matrices with more tiles than 5 blocks/SM hold
#undef QE_W4HW_KERNEL

// XH packing of a bf16 activation matrix (fallback for inputs whose producer does not write XH)
__global__ void xh_pack_kernel(const bf16* __restrict__ x, int ldx, int K, int m, __half* __restrict__ xh,
                               int ldxh) {
    const size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= (size_t)m * K) return;
    const int tok = i / K, k = i % K;
    store_xh(xh + (size_t)tok * ldxh, k, __bfloat162float(x[(size_t)tok * ldx + k]));
}

uint4* g_xpack[16] = {};
template <int NT>
__global__ void gemv_w16_kernel(const uint4* __restrict__ w, int K, const bf16* __restrict__ x,
                                int ldx, float* __restrict__ y, int ldy, int m, size_t ystride) {
    const int r = blockIdx.x, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int nwarps = blockDim.x >> 5;
    const int g = lane >> 2, t = lane & 3;
    const int steps = K >> 4;
    const int part = warp + blockIdx.y * nwarps, nparts = nwarps * gridDim.y;
    const int sb = part * steps / nparts, se = (part + 1) * steps / nparts;
    y += blockIdx.y * ystride;

    const uint4* wp = w + (size_t)r * steps * 32 + lane;
    bool act[NT];
    const bf16* xr[NT];
#pragma unroll
    for (int j = 0; j < NT; j++) {
        act[j] = 8 * j + g < m;
        xr[j] = x + (size_t)(act[j] ? 8 * j + g : 0) * ldx + 2 * t;
    }

    float acc[NT][4];
#pragma unroll
    for (int j = 0; j < NT; j++) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.f;
#pragma unroll 4
    for (int st = sb; st < se; st++) {
        const uint4 v = ldg_stream(wp + (size_t)st * 32);
        const uint32_t a[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
        for (int j = 0; j < NT; j++)
            mma_bf16(acc[j], a, load_x2(xr[j] + st * 16, act[j]), load_x2(xr[j] + st * 16 + 8, act[j]));
    }
    if (NT == 1) reduce_store(acc[0], nwarps, y, ldy, m, r * 16);
    else reduce_store_nt<NT>(acc, nwarps, y, ldy, m, r * 16, false);
}


__device__ __forceinline__ uint32_t src_nib(const int32_t* q, int K, int row, int k) {
    return ((uint32_t)q[(size_t)row * (K >> 3) + (k >> 3)] >> (4 * (k & 7))) & 15u;
}

__global__ void repack_w4_kernel(const int32_t* __restrict__ src, int N, int K,
                                 uint32_t* __restrict__ dst) {
    const size_t idx = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    const size_t total = (size_t)N * K / 8;
    if (idx >= total) return;
    const int j = idx & 3, lane = (idx >> 2) & 31;
    const size_t chunk_id = idx >> 7;  // r * (K/64) + c
    const int chunks = K >> 6;
    const int r = chunk_id / chunks, c = chunk_id % chunks;
    const int g = lane >> 2, t = lane & 3;
    const int k0 = c * 64 + j * 16 + 2 * t;
    const int row = r * 16 + g;
    // nibble slot -> (row offset, k offset); see dequant_w4: slot j low half, slot j+4 high half
    const uint32_t a0 = src_nib(src, K, row, k0), a1 = src_nib(src, K, row, k0 + 1);
    const uint32_t a2 = src_nib(src, K, row + 8, k0), a3 = src_nib(src, K, row + 8, k0 + 1);
    const uint32_t a4 = src_nib(src, K, row, k0 + 8), a5 = src_nib(src, K, row, k0 + 9);
    const uint32_t a6 = src_nib(src, K, row + 8, k0 + 8), a7 = src_nib(src, K, row + 8, k0 + 9);
    dst[idx] = a0 | (a2 << 4) | (a4 << 8) | (a6 << 12) | (a1 << 16) | (a3 << 20) | (a5 << 24) |
               (a7 << 28);
}

__global__ void repack_scales_kernel(const bf16* __restrict__ src, int N, int K,
                                     bf16* __restrict__ dst) {
    const size_t idx = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    const int groups = K >> 7;
    if (idx >= (size_t)N * groups) return;
    const int i = idx & 15;
    const size_t rg = idx >> 4;
    const int r = rg / groups, gi = rg % groups;
    dst[idx] = src[(size_t)(r * 16 + i) * groups + gi];
}

__global__ void repack_w16_kernel(const bf16* __restrict__ src, int N, int K,
                                  uint32_t* __restrict__ dst) {
    const size_t idx = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    const size_t total = (size_t)N * K / 2;
    if (idx >= total) return;
    const int j = idx & 3, lane = (idx >> 2) & 31;
    const size_t step_id = idx >> 7;
    const int steps = K >> 4;
    const int r = step_id / steps, s = step_id % steps;
    const int g = lane >> 2, t = lane & 3;
    const int row = r * 16 + g + ((j & 1) ? 8 : 0);
    const int k = s * 16 + 2 * t + ((j & 2) ? 8 : 0);
    dst[idx] = *reinterpret_cast<const uint32_t*>(src + (size_t)row * K + k);
}

__device__ __forceinline__ void mma_f16(float c[4], const uint32_t a[4], uint32_t b0, uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, "
        "{%8,%9}, {%0,%1,%2,%3};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}


// bf16 W16 GEMV reading the XH activations (QE_GEMV_H16): the weights are an fp16 copy of the bf16
// fragments (exact), x comes as fp16 x/16 (exact), products accumulate in fp32 like gemv_w16_kernel - the
// results differ from the bf16 path only through the clamp and sub-normal fp16 values of |x| < 2e-4.
// Same warp / split structure as gemv_w16_kernel.
template <int NT>
__global__ void gemv_w16h_kernel(const uint4* __restrict__ w, int K, const __half* __restrict__ xh,
                                 int ldx, float* __restrict__ y, int ldy, int m, size_t ystride) {
    const int r = blockIdx.x, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int nwarps = blockDim.x >> 5;
    const int g = lane >> 2, t = lane & 3;
    const int steps = K >> 4;
    const int part = warp + blockIdx.y * nwarps, nparts = nwarps * gridDim.y;
    const int sb = part * steps / nparts, se = (part + 1) * steps / nparts;
    y += blockIdx.y * ystride;

    const uint4* wp = w + (size_t)r * steps * 32 + lane;
    const __half* xr[NT];
#pragma unroll
    for (int j = 0; j < NT; j++) {
        const int tok = 8 * j + g < m ? 8 * j + g : 0;
        xr[j] = xh + (size_t)tok * ldx + 16 * t;
    }
    float acc[NT][4];
#pragma unroll
    for (int j = 0; j < NT; j++) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.f;
#pragma unroll 4
    for (int st = sb; st < se; st++) {
        const uint4 v = ldg_stream(wp + (size_t)st * 32);
        const uint32_t a[4] = {v.x, v.y, v.z, v.w};
        const int off = (st >> 2) * 64 + (st & 3) * 4;   // XH: chunk st/4, k-step st%4 -> 8 contiguous bytes
#pragma unroll
        for (int j = 0; j < NT; j++) {
            const uint2 b = *reinterpret_cast<const uint2*>(xr[j] + off);
            mma_f16(acc[j], a, b.x, b.y);
        }
    }
#pragma unroll
    for (int j = 0; j < NT; j++)
#pragma unroll
        for (int i = 0; i < 4; i++) acc[j][i] *= kXhScale;
    if (NT == 1) reduce_store(acc[0], nwarps, y, ldy, m, r * 16);
    else reduce_store_nt<NT>(acc, nwarps, y, ldy, m, r * 16, false);
}

__global__ void w16_to_h_kernel(const uint32_t* __restrict__ src, uint32_t* __restrict__ dst, size_t n) {
    const size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= n) return;
    const uint32_t v = src[i];
    const float2 f = __bfloat1622float2(*reinterpret_cast<const bf162*>(&v));
    const __half2 h = __floats2half2_rn(f.x, f.y);
    dst[i] = *reinterpret_cast<const uint32_t*>(&h);
}

// Two signed bytes (low bytes of the 16-bit lanes after byte_perm) -> exact fp16 integers:
// (b ^ 0x80) is b + 128 in 0..255; OR into 1024.0 and subtract 1152.
__device__ __forceinline__ uint32_t s8x2_to_f16x2(uint32_t t) {
    uint32_t v = ((t ^ 0x00800080u) & 0x00FF00FFu) | 0x64006400u;
    __half2 h = __hsub2(*reinterpret_cast<__half2*>(&v), __float2half2_rn(1152.f));
    return *reinterpret_cast<uint32_t*>(&h);
}

__device__ __forceinline__ uint32_t load_x2_f16(const bf16* p, bool ok) {
    if (!ok) return 0u;
    const float2 f = __bfloat1622float2(*reinterpret_cast<const bf162*>(p));
    __half2 h = __floats2half2_rn(f.x, f.y);
    return *reinterpret_cast<uint32_t*>(&h);
}

// INT8 GEMV (lm_head): fp16 tensor-core math on exact integer weights, per-group fp32 scaling.
__global__ void gemv_w8_kernel(const uint4* __restrict__ q, const bf16* __restrict__ s, int K,
                               const bf16* __restrict__ x, int ldx, float* __restrict__ y, int ldy,
                               int m) {
    const int r = blockIdx.x, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int nwarps = blockDim.x >> 5;
    const int g = lane >> 2, t = lane & 3;
    const int groups = K >> 7;
    const int gb = warp * groups / nwarps, ge = (warp + 1) * groups / nwarps;
    const uint4* qp = q + (size_t)r * (K >> 5) * 32 + lane;
    const bf16* sp = s + (size_t)r * groups * 16;
    const bool act = g < m;
    const bf16* xr = x + (size_t)(act ? g : 0) * ldx + 2 * t;
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
    for (int gi = gb; gi < ge; gi++) {
        uint4 w[4];
#pragma unroll
        for (int i = 0; i < 4; i++) w[i] = ldg_stream(qp + (size_t)(gi * 4 + i) * 32);
        const float sg = __bfloat162float(sp[gi * 16 + g]);
        const float sg8 = __bfloat162float(sp[gi * 16 + g + 8]);
        float c[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (int i = 0; i < 4; i++) {
            const uint32_t ws[4] = {w[i].x, w[i].y, w[i].z, w[i].w};
#pragma unroll
            for (int h = 0; h < 2; h++) {  // two k-steps per 16 bytes
                const int k0 = gi * 128 + (i * 2 + h) * 16;
                const uint32_t lo = ws[h * 2], hi = ws[h * 2 + 1];
                uint32_t a[4];
                a[0] = s8x2_to_f16x2(__byte_perm(lo, 0, 0x4140));
                a[1] = s8x2_to_f16x2(__byte_perm(lo, 0, 0x4342));
                a[2] = s8x2_to_f16x2(__byte_perm(hi, 0, 0x4140));
                a[3] = s8x2_to_f16x2(__byte_perm(hi, 0, 0x4342));
                mma_f16(c, a, load_x2_f16(xr + k0, act), load_x2_f16(xr + k0 + 8, act));
            }
        }
        acc[0] += c[0] * sg;
        acc[1] += c[1] * sg;
        acc[2] += c[2] * sg8;
        acc[3] += c[3] * sg8;
    }
    reduce_store(acc, nwarps, y, ldy, m, r * 16);
}

// INT8 GEMV for 9..8*NT tokens with fp16-packed activations (xpack layout).
template <int NT>
__global__ void gemv_w8p_kernel(const uint4* __restrict__ q, const bf16* __restrict__ s, int K,
                                const uint4* __restrict__ xp, float* __restrict__ y, int ldy, int m) {
    const int r = blockIdx.x, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int nwarps = blockDim.x >> 5;
    const int g = lane >> 2;
    const int groups = K >> 7, kps = K >> 5;
    const int gb = warp * groups / nwarps, ge = (warp + 1) * groups / nwarps;
    const uint4* qp = q + (size_t)r * (K >> 5) * 32 + lane;
    const bf16* sp = s + (size_t)r * groups * 16;
    const uint4* xl = xp + lane;
    float acc[NT][4];
#pragma unroll
    for (int j = 0; j < NT; j++) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.f;
    for (int gi = gb; gi < ge; gi++) {
        uint4 w[4];
#pragma unroll
        for (int i = 0; i < 4; i++) w[i] = ldg_stream(qp + (size_t)(gi * 4 + i) * 32);
        const float sg = __bfloat162float(sp[gi * 16 + g]);
        const float sg8 = __bfloat162float(sp[gi * 16 + g + 8]);
        float c[NT][4];
#pragma unroll
        for (int j = 0; j < NT; j++) c[j][0] = c[j][1] = c[j][2] = c[j][3] = 0.f;
#pragma unroll
        for (int i = 0; i < 4; i++) {  // 16 bytes of weights = two k-steps = one packed x pair
            uint4 b[NT];
#pragma unroll
            for (int nt = 0; nt < NT; nt++) b[nt] = xl[((size_t)nt * kps + gi * 4 + i) * 32];
            const uint32_t ws[4] = {w[i].x, w[i].y, w[i].z, w[i].w};
#pragma unroll
            for (int h = 0; h < 2; h++) {
                const uint32_t lo = ws[h * 2], hi = ws[h * 2 + 1];
                uint32_t a[4];
                a[0] = s8x2_to_f16x2(__byte_perm(lo, 0, 0x4140));
                a[1] = s8x2_to_f16x2(__byte_perm(lo, 0, 0x4342));
                a[2] = s8x2_to_f16x2(__byte_perm(hi, 0, 0x4140));
                a[3] = s8x2_to_f16x2(__byte_perm(hi, 0, 0x4342));
#pragma unroll
                for (int nt = 0; nt < NT; nt++)
                    mma_f16(c[nt], a, h ? b[nt].z : b[nt].x, h ? b[nt].w : b[nt].y);
            }
        }
#pragma unroll
        for (int nt = 0; nt < NT; nt++) {
            acc[nt][0] += c[nt][0] * sg; acc[nt][1] += c[nt][1] * sg;
            acc[nt][2] += c[nt][2] * sg8; acc[nt][3] += c[nt][3] * sg8;
        }
    }
    reduce_store_nt<NT>(acc, nwarps, y, ldy, m, r * 16, false);
}

// INT8 lm_head for 17..32 tokens (one K slice per tile, i.e. pick_warps == 1 as for the 5174-tile vocab
// shard): the per-tile kernel above re-reads the packed activations for every tile (NT x 2 KB per group:
// 1.7 GB per call at m = 32, L2-bound at ~50 % of DRAM). Here a block of WB warps owns WB x TW consecutive
// tiles and stages the packed activations of each 128-k group once in shared memory (cp.async, S stages);
// each warp reads its fragments from there (lane-contiguous 16-byte loads, conflict-free) and prefetches
// the weights of the next group. Per-tile arithmetic and order are exactly gemv_w8p_kernel's (bit-identical).
__device__ __forceinline__ void cp_async16_g(unsigned smem, const void* gmem) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(smem), "l"(gmem));
}
template <int NT, int WB, int TW, int S>
__global__ void __launch_bounds__(WB * 32) gemv_w8s_kernel(const uint4* __restrict__ q, const bf16* __restrict__ s,
                                                           int K, const uint4* __restrict__ xp,
                                                           float* __restrict__ y, int ldy, int m, int tiles) {
    constexpr int kStage = NT * 128;                 // uint4 per stage (NT x 4 k-pairs x 32 lanes)
    __shared__ uint4 xs[S][kStage];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int g = lane >> 2, t = lane & 3;
    const int groups = K >> 7, kps = K >> 5;
    auto issue = [&](int gi) {
        if (gi < groups) {
            const int st = gi % S;
            for (int i = threadIdx.x; i < kStage; i += WB * 32) {
                const int nt = i >> 7, r = i & 127;
                cp_async16_g((unsigned)__cvta_generic_to_shared(&xs[st][i]), xp + ((size_t)nt * kps + gi * 4) * 32 + r);
            }
        }
        asm volatile("cp.async.commit_group;");
    };
#pragma unroll
    for (int i = 0; i < S - 1; i++) issue(i);

    const int tile0 = (blockIdx.x * WB + warp) * TW;
    bool ok[TW];
    const uint4* qp[TW];
    const bf16* sp[TW];
#pragma unroll
    for (int tw = 0; tw < TW; tw++) {
        ok[tw] = tile0 + tw < tiles;
        const int r = ok[tw] ? tile0 + tw : 0;
        qp[tw] = q + (size_t)r * (K >> 5) * 32 + lane;
        sp[tw] = s + (size_t)r * groups * 16;
    }
    float acc[TW][NT][4];
#pragma unroll
    for (int tw = 0; tw < TW; tw++)
#pragma unroll
        for (int j = 0; j < NT; j++) acc[tw][j][0] = acc[tw][j][1] = acc[tw][j][2] = acc[tw][j][3] = 0.f;
    uint4 wn[TW][4];
#pragma unroll
    for (int tw = 0; tw < TW; tw++)
#pragma unroll
        for (int i = 0; i < 4; i++) wn[tw][i] = ok[tw] ? ldg_stream(qp[tw] + (size_t)i * 32) : make_uint4(0, 0, 0, 0);
    for (int gi = 0; gi < groups; gi++) {
        asm volatile("cp.async.wait_group %0;" ::"n"(S - 2));
        __syncthreads();
        issue(gi + S - 1);
        uint4 w[TW][4];
        float sg[TW], sg8[TW];
#pragma unroll
        for (int tw = 0; tw < TW; tw++) {
#pragma unroll
            for (int i = 0; i < 4; i++) {
                w[tw][i] = wn[tw][i];
                if (ok[tw] && gi + 1 < groups) wn[tw][i] = ldg_stream(qp[tw] + (size_t)((gi + 1) * 4 + i) * 32);
            }
            sg[tw] = __bfloat162float(sp[tw][gi * 16 + g]);
            sg8[tw] = __bfloat162float(sp[tw][gi * 16 + g + 8]);
        }
        const uint4* xl = &xs[gi % S][lane];
        float c[TW][NT][4];
#pragma unroll
        for (int tw = 0; tw < TW; tw++)
#pragma unroll
            for (int j = 0; j < NT; j++) c[tw][j][0] = c[tw][j][1] = c[tw][j][2] = c[tw][j][3] = 0.f;
#pragma unroll
        for (int i = 0; i < 4; i++) {
            uint4 b[NT];
#pragma unroll
            for (int nt = 0; nt < NT; nt++) b[nt] = xl[nt * 128 + i * 32];
#pragma unroll
            for (int tw = 0; tw < TW; tw++) {
                const uint32_t ws[4] = {w[tw][i].x, w[tw][i].y, w[tw][i].z, w[tw][i].w};
#pragma unroll
                for (int h = 0; h < 2; h++) {
                    const uint32_t lo = ws[h * 2], hi = ws[h * 2 + 1];
                    uint32_t a[4];
                    a[0] = s8x2_to_f16x2(__byte_perm(lo, 0, 0x4140));
                    a[1] = s8x2_to_f16x2(__byte_perm(lo, 0, 0x4342));
                    a[2] = s8x2_to_f16x2(__byte_perm(hi, 0, 0x4140));
                    a[3] = s8x2_to_f16x2(__byte_perm(hi, 0, 0x4342));
#pragma unroll
                    for (int nt = 0; nt < NT; nt++)
                        mma_f16(c[tw][nt], a, h ? b[nt].z : b[nt].x, h ? b[nt].w : b[nt].y);
                }
            }
        }
#pragma unroll
        for (int tw = 0; tw < TW; tw++)
#pragma unroll
            for (int nt = 0; nt < NT; nt++) {
                acc[tw][nt][0] += c[tw][nt][0] * sg[tw]; acc[tw][nt][1] += c[tw][nt][1] * sg[tw];
                acc[tw][nt][2] += c[tw][nt][2] * sg8[tw]; acc[tw][nt][3] += c[tw][nt][3] * sg8[tw];
            }
    }
    asm volatile("cp.async.wait_group 0;");
#pragma unroll
    for (int tw = 0; tw < TW; tw++) {
        if (!ok[tw]) continue;
        const int row0 = (tile0 + tw) * 16;
#pragma unroll
        for (int j = 0; j < NT; j++) {
            const int n0 = 8 * j + 2 * t, n1 = n0 + 1;
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

// one warp per (row, group of 128): RTN int8 with scale absmax / 127, written in W8 layout.
// Lane L handles 4 consecutive k values of the group.
__global__ void quantize_w8_kernel(const bf16* __restrict__ src, int N, int K,
                                   uint8_t* __restrict__ q, bf16* __restrict__ s) {
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
    const int groups = K >> 7;
    if (warp >= N * groups) return;
    const int row = warp / groups, gi = warp % groups;
    float v[4], mx = 0.f;
#pragma unroll
    for (int i = 0; i < 4; i++) {
        v[i] = __bfloat162float(src[(size_t)row * K + gi * 128 + lane * 4 + i]);
        mx = fmaxf(mx, fabsf(v[i]));
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
    const bf16 sb = __float2bfloat16(mx / 127.f);
    const float sc = __bfloat162float(sb);
    const float inv = sc > 0.f ? 1.f / sc : 0.f;
    const int rt = row >> 4, rr = row & 15;
    for (int i = 0; i < 4; i++) {
        const int k = gi * 128 + lane * 4 + i;
        int qi = max(-127, min(127, __float2int_rn(v[i] * inv)));
        // fragment position: k-step ks = k/16, within it col = k%16; row rr: g = rr%8, hi = rr/8
        const int ks = k >> 4, col = k & 15;
        const int tq = (col & 7) >> 1, half = col >> 3, odd = col & 1;
        const int ln = (rr & 7) * 4 + tq;
        // a-index: a0a1 (row g, col 2t..), a2a3 (row g+8), a4a5 (row g, col+8), a6a7 (row g+8, col+8)
        const int ai = (half * 2 + (rr >> 3)) * 2 + odd;
        const size_t base = (((size_t)rt * (K >> 5) + (ks >> 1)) * 32 + ln) * 16 + (ks & 1) * 8;
        q[base + ai] = (uint8_t)(int8_t)qi;
    }
    if (lane == 0) s[((size_t)rt * groups + gi) * 16 + rr] = sb;
}

// one warp per (row, group of 128): lane holds 4 consecutive values
__global__ void quantize_rtn_kernel(const bf16* __restrict__ src, int N, int K,
                                    int32_t* __restrict__ q, bf16* __restrict__ s) {
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
    const int groups = K >> 7;
    if (warp >= N * groups) return;
    const int row = warp / groups, gi = warp % groups;
    const bf16* p = src + (size_t)row * K + gi * 128 + lane * 4;
    float v[4], mx = 0.f;
#pragma unroll
    for (int i = 0; i < 4; i++) {
        v[i] = __bfloat162float(p[i]);
        mx = fmaxf(mx, fabsf(v[i]));
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
    const bf16 sb = __float2bfloat16(mx / 7.f);
    const float sc = __bfloat162float(sb);
    const float inv = sc > 0.f ? 1.f / sc : 0.f;
    uint32_t nib = 0;
#pragma unroll
    for (int i = 0; i < 4; i++) {
        int qi = __float2int_rn(v[i] * inv);
        qi = max(-8, min(7, qi));
        nib |= (uint32_t)(qi + 8) << (4 * i);
    }
    // two lanes form one int32 (8 nibbles, LSB first)
    const uint32_t other = __shfl_down_sync(0xffffffffu, nib, 1);
    if ((lane & 1) == 0)
        q[(size_t)row * (K >> 3) + gi * 16 + (lane >> 1)] = (int32_t)(nib | (other << 16));
    if (lane == 0) s[(size_t)row * groups + gi] = sb;
}

// one block per destination row
__global__ void glu_rows_kernel(const int32_t* __restrict__ sq, const bf16* __restrict__ ss, int N,
                                int K, int32_t* __restrict__ dq, bf16* __restrict__ ds) {
    const int dr = blockIdx.x, I = N / 2;
    const int tile = dr / 16, j = dr % 16;
    const int sr = j < 8 ? tile * 8 + j : I + tile * 8 + (j - 8);
    for (int i = threadIdx.x; i < K / 8; i += blockDim.x) dq[(size_t)dr * (K / 8) + i] = sq[(size_t)sr * (K / 8) + i];
    for (int i = threadIdx.x; i < K / 128; i += blockDim.x) ds[(size_t)dr * (K / 128) + i] = ss[(size_t)sr * (K / 128) + i];
}

__global__ void prefetch_l2_kernel(const char* __restrict__ p, size_t bytes) {
    for (size_t off = ((size_t)blockIdx.x * blockDim.x + threadIdx.x) * 128; off < bytes;
         off += (size_t)gridDim.x * blockDim.x * 128)
        asm volatile("prefetch.global.L2::evict_last [%0];" ::"l"(p + off));
}

// ---- experiment: GEMV launched before its input is ready. Each warp stages the weights
// and scales of its first D groups in registers, then the block waits for `flag` (a stand-in for the
// all-reduce that produces x), then runs the m <= 8 H16 loop with a D-deep register ring. Same
// arithmetic order as gemv_w4h_kernel<1> (bit-identical outputs).
#ifndef QE_EARLY_POLL_NS
#define QE_EARLY_POLL_NS 32
#endif
constexpr unsigned kEarlyPollNs = QE_EARLY_POLL_NS;  // consumer poll of the ready tag (hundreds of pollers)
__device__ __forceinline__ uint4 ld_coherent(const uint4* p) {  // ld.global (L1-cached, not .nc)
    uint4 v;
    asm volatile("ld.global.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p) : "memory");
    return v;
}

template <int D, int S, bool XCG>
__global__ void __launch_bounds__(256, 4) gemv_w4h_early_kernel(const uint4* __restrict__ q, const bf16* __restrict__ s, int K,
                const __half* xh, int ldx, float* __restrict__ y, int ldy, int m,
                const unsigned* flag, const unsigned* epoch, int idx, bool glu, __half* __restrict__ yh, int ldyh,
                const FusedAR far, bool fused) {
    extern __shared__ uint4 stage[];  // [warp][S][2][32]: groups gb + D .. gb + D + S - 1
    const int r = blockIdx.x, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int nwarps = blockDim.x >> 5;
    const int g = lane >> 2, t = lane & 3;
    const int groups = K >> 7, chunks = K >> 6;
    const int gb = warp * groups / nwarps, ge = (warp + 1) * groups / nwarps;
    const uint4* qp = q + (size_t)r * chunks * 32 + lane;
    const bf16* sp = s + (size_t)r * groups * 16;
    const int tok = g < m ? g : 0;
    const uint4* xr = reinterpret_cast<const uint4*>(xh + (size_t)tok * ldx + t * 16);
    uint4 w[D][2];
    bf16 s0[D], s8[D];
#pragma unroll
    for (int d = 0; d < D; d++)
        if (gb + d < ge) {
            w[d][0] = ldg_stream(qp + (size_t)(2 * (gb + d)) * 32);
            w[d][1] = ldg_stream(qp + (size_t)(2 * (gb + d) + 1) * 32);
            s0[d] = sp[(gb + d) * 16 + g];
            s8[d] = sp[(gb + d) * 16 + g + 8];
        }
    uint4* stw = stage + (size_t)warp * S * 64;
#pragma unroll
    for (int j = 0; j < S; j++)
        if (gb + D + j < ge) {
            const uint4* src = qp + (size_t)(2 * (gb + D + j)) * 32;
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"((unsigned)__cvta_generic_to_shared(stw + j * 64 + lane)), "l"(src));
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"((unsigned)__cvta_generic_to_shared(stw + j * 64 + 32 + lane)), "l"(src + 32));
        }
    asm volatile("cp.async.commit_group;");
    if (fused && r < arf::items(far)) arf::ll16_chunk(far);  // this block also reduces 32 words of a row
    if (flag) {
        if (threadIdx.x == 0) {
            unsigned long long t0, t1;
            asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
            // engine: the all-reduce's tag ready[idx] == epoch * 1024 + idx + 1; bench: flag != 0
            const unsigned* f = epoch ? flag + idx : flag;
            const unsigned want = epoch ? *epoch * 1024u + (unsigned)idx + 1u : 0u;
            unsigned ws = 0;
            for (;;) {
                unsigned v;
                asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(f) : "memory");
                if (epoch ? v == want : v != 0u) break;
                __nanosleep(kEarlyPollNs);
                if (epoch) {  // engine: the collective watchdog (peers may legitimately lag by tens of ms: lazy graph capture on one GPU)
                    QE_WD_POLL(ws, kWdEarlyGemv, -1, want, v);
                } else {      // bench: give up after 10 s
                    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t1));
                    if (t1 - t0 > 10000000000ull) { atomicAdd((unsigned*)flag + 2048, 1u); break; }
                }
            }
            __threadfence();
        }
        __syncthreads();
    }
    asm volatile("cp.async.wait_all;" ::: "memory");
    __syncwarp();
    float acc[1][4] = {{0.f, 0.f, 0.f, 0.f}};
    for (int g0 = gb; g0 < ge; g0 += D) {
#pragma unroll
        for (int d = 0; d < D; d++) {
            const int gi = g0 + d;
            if (gi >= ge) break;
            const float sg = xh_scale(s0[d]), sg8 = xh_scale(s8[d]);
            uint32_t h[2] = {0u, 0u};
#pragma unroll
            for (int c = 0; c < 2; c++) {
                const int ch = 2 * gi + c;
                const uint4 wc = w[d][c];
                const uint32_t words[4] = {wc.x, wc.y, wc.z, wc.w};
                uint4 b[2];
                b[0] = XCG ? __ldcg(xr + (size_t)ch * 8) : ld_coherent(xr + (size_t)ch * 8);
                b[1] = XCG ? __ldcg(xr + (size_t)ch * 8 + 1) : ld_coherent(xr + (size_t)ch * 8 + 1);
#pragma unroll
                for (int j = 0; j < 4; j++) {
                    uint32_t a[4];
                    xh_dequant(words[j], a);
                    const uint4& bb = b[j >> 1];
                    xh_mma(h, a, (j & 1) ? bb.z : bb.x, (j & 1) ? bb.w : bb.y);
                }
                xh_flush(acc[0], h, sg, sg8);
            }
            if (gi + D < ge) {
                const int sj = gi - gb;  // staged index of group gi + D
                if (S > 0 && sj < S) {
                    w[d][0] = stw[sj * 64 + lane];
                    w[d][1] = stw[sj * 64 + 32 + lane];
                } else {
                    w[d][0] = ldg_stream(qp + (size_t)(2 * (gi + D)) * 32);
                    w[d][1] = ldg_stream(qp + (size_t)(2 * (gi + D) + 1) * 32);
                }
                s0[d] = sp[(gi + D) * 16 + g];
                s8[d] = sp[(gi + D) * 16 + g + 8];
            }
        }
    }
    reduce_store_nt<1>(acc, nwarps, y, ldy, m, r * 16, glu, yh, ldyh);
}

// AR stand-in for the co-residency test: `blocks` blocks of `threads` threads (LL16: m x 768) spin for
// ns, then the last block to finish sets the flag
__global__ void flag_kernel(unsigned* flag, unsigned v, unsigned long long ns, const uint4* src, uint4* dst, int n16) {
    __shared__ int last;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n16; i += gridDim.x * blockDim.x) dst[i] = src[i];
    __syncthreads();
    if (threadIdx.x == 0) {
        unsigned long long t0, t1;
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
        do {
            asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t1));
        } while (t1 - t0 < ns);
        __threadfence();
        last = atomicAdd(flag + 1024, 1u) == gridDim.x - 1;
    }
    __syncthreads();
    if (threadIdx.x == 0 && last) {
        flag[1024] = 0;
        __threadfence();
        *(volatile unsigned*)flag = v;
    }
}

}  // namespace

void gemv_w4h_early(const W4& w, const __half* xh, int ldx, float* y, int ldy, int m, int depth,
                    const unsigned* flag, int nw_override, cudaStream_t st, const unsigned* epoch, int idx,
                    __half* yh, int ldyh, const FusedAR* far) {
    const FusedAR fa = far ? *far : FusedAR{};
    CHECK(w.n % 16 == 0 && w.k % 128 == 0 && m >= 1 && m <= 8 && ldx % 64 == 0, "gemv_w4h_early %d %d %d", w.n, w.k, m);
    const int tiles = w.n / 16;
    const int nw = nw_override > 0 ? nw_override : pick_warps(tiles, w.k / 128, true);
    const bool xcg = depth >= 100;  // + 100: x through L2 only
    depth %= 100;
    const int S = depth / 10, D = depth % 10;  // depth = 10 * smem-staged groups + register groups
#define QE_E(D_, S_) do { if (xcg) QE_E3(D_, S_, true); else QE_E3(D_, S_, false); } while (0)
#define QE_E3(D_, S_, X_) do { const size_t sm = (size_t)nw * (S_) * 64 * 16; \
        static bool once = false; if (!once) { once = true; CUDA_CHECK(cudaFuncSetAttribute(gemv_w4h_early_kernel<D_, S_, X_>, cudaFuncAttributeMaxDynamicSharedMemorySize, 64 * 1024)); } \
        gemv_w4h_early_kernel<D_, S_, X_><<<tiles, nw * 32, sm, st>>>(w.q, w.s, w.k, xh, ldx, y, ldy, m, flag, epoch, idx, w.glu, yh, ldyh, fa, far != nullptr); } while (0)
    if (S == 0) {
        if (D <= 1) QE_E(1, 0);
        else if (D == 2) QE_E(2, 0);
        else if (D <= 4) QE_E(4, 0);
        else if (D <= 6) QE_E(6, 0);
        else QE_E(8, 0);
    } else if (S <= 2) QE_E(2, 2);
    else if (S <= 4) QE_E(2, 4);
    else QE_E(2, 6);
#undef QE_E
#undef QE_E3
}
void flag_set_after(unsigned* flag, unsigned v, double us, cudaStream_t st, int blocks, int threads,
                    const void* src, void* dst, size_t bytes) {
    flag_kernel<<<blocks, threads, 0, st>>>(flag, v, (unsigned long long)(us * 1000), (const uint4*)src, (uint4*)dst,
                                            (int)(bytes / 16));
}

int pick_warps(int tiles, int units, bool w4) {
    // Aim for ~2k resident warps on 82 SMs so enough loads are in flight. Matrices with <= 512
    // tiles (N <= 8192: down, qkv, o, DeltaNet in) can use fewer, longer K slices
    // (QE_GEMV_NW_SMALL, e.g. 4): the m <= 8 GEMV is as fast and the skinny kernel benefits.
    // fp16-accumulate path (QE_GEMV_H16): 5 slices measured best for its kernels over m = 1..32 (7: m = 24/32
    // 20-25% slower on these matrices; 4: m = 12/16 7% slower); the bf16 path keeps its default
    static const int nw_small = getenv("QE_GEMV_NW_SMALL") ? atoi(getenv("QE_GEMV_NW_SMALL"))
                              : gemv_h16_on() ? 5 : 0;
    int wpt = (2048 + tiles - 1) / tiles;
    if (w4 && nw_small > 0 && tiles <= 512 && wpt > nw_small) wpt = nw_small;
    if (wpt > kMaxWarps) wpt = kMaxWarps;
    if (wpt > units) wpt = units;
    if (wpt < 1) wpt = 1;
    return wpt;
}

void glu_interleave_rows(const int32_t* src_q, const bf16* src_s, int n, int k, int32_t* dst_q,
                         bf16* dst_s, cudaStream_t st) {
    CHECK(n % 16 == 0, "glu rows %d", n);
    glu_rows_kernel<<<n, 256, 0, st>>>(src_q, src_s, n, k, dst_q, dst_s);
}

void prefetch_l2(const void* p, size_t bytes, cudaStream_t st) {
    prefetch_l2_kernel<<<16, 256, 0, st>>>((const char*)p, bytes);
}

void gemv_w8(const W8& w, const bf16* x, int ldx, float* y, int ldy, int m, cudaStream_t st) {
    CHECK(w.n % 16 == 0 && w.k % 128 == 0 && m >= 1 && m <= kMaxBatchRows, "gemv_w8");
    static const bool s_on = !getenv("QE_W8S") || atoi(getenv("QE_W8S"));
    if (m > 32 && !(s_on && pick_warps(w.n / 16, w.k / 128, true) == 1)) {  // two calls (rows independent)
        gemv_w8(w, x, ldx, y, ldy, 32, st);
        gemv_w8(w, x + (size_t)32 * ldx, ldx, y + (size_t)32 * ldy, ldy, m - 32, st);
        return;
    }
    const int tiles = w.n / 16;
    const int nw = pick_warps(tiles, w.k / 128, true);
    if (m <= 8) {
        gemv_w8_kernel<<<tiles, nw * 32, 0, st>>>(w.q, w.s, w.k, x, ldx, y, ldy, m);
        return;
    }
    int dev;
    CUDA_CHECK(cudaGetDevice(&dev));
    uint4* xp = g_xpack[dev];
    CHECK(xp && w.k <= kXpackMaxK, "gemv workspace (k=%d)", w.k);
    const int ntiles = m <= 16 ? 2 : (m + 7) / 8;
    const int items = ntiles * (w.k / 32) * 32;
    xpack_kernel<<<(items + 255) / 256, 256, 0, st>>>(x, ldx, w.k, m, ntiles, xp, true);
    // staged-activation kernel for single-slice matrices (the lm_head vocab shard): 9..32 rows
    // (QE_W8S=0 restores gemv_w8p). 8 warps x 1 tile, 3 stages measured best (4/16 warps,
    // 2 tiles per warp and 4 stages within 1-2 %, 16 warps x 2 tiles spills at NT = 4)
    if (s_on && nw == 1) {
        constexpr int WB = 8;
        const int blocks = (tiles + WB - 1) / WB;
        // 33..64 rows (batched steps with one draft per sequence beyond 16 sequences): same kernel, NT 5..8
        static const int wb_hi = getenv("QE_W8S_WB") ? atoi(getenv("QE_W8S_WB")) : 8;  // warps per block, 33..64 rows
        if (ntiles > 4 && wb_hi == 4) {
            const int b4 = (tiles + 3) / 4;
            if (ntiles == 5) gemv_w8s_kernel<5, 4, 1, 3><<<b4, 128, 0, st>>>(w.q, w.s, w.k, xp, y, ldy, m, tiles);
            else if (ntiles == 6) gemv_w8s_kernel<6, 4, 1, 2><<<b4, 128, 0, st>>>(w.q, w.s, w.k, xp, y, ldy, m, tiles);
            else if (ntiles == 7) gemv_w8s_kernel<7, 4, 1, 2><<<b4, 128, 0, st>>>(w.q, w.s, w.k, xp, y, ldy, m, tiles);
            else gemv_w8s_kernel<8, 4, 1, 2><<<b4, 128, 0, st>>>(w.q, w.s, w.k, xp, y, ldy, m, tiles);
            return;
        }
        if (ntiles > 4) {
            if (ntiles == 5) gemv_w8s_kernel<5, WB, 1, 3><<<blocks, WB * 32, 0, st>>>(w.q, w.s, w.k, xp, y, ldy, m, tiles);
            else if (ntiles == 6) gemv_w8s_kernel<6, WB, 1, 2><<<blocks, WB * 32, 0, st>>>(w.q, w.s, w.k, xp, y, ldy, m, tiles);
            else if (ntiles == 7) gemv_w8s_kernel<7, WB, 1, 2><<<blocks, WB * 32, 0, st>>>(w.q, w.s, w.k, xp, y, ldy, m, tiles);
            else gemv_w8s_kernel<8, WB, 1, 2><<<blocks, WB * 32, 0, st>>>(w.q, w.s, w.k, xp, y, ldy, m, tiles);
            return;
        }
        if (ntiles == 2) gemv_w8s_kernel<2, WB, 1, 3><<<blocks, WB * 32, 0, st>>>(w.q, w.s, w.k, xp, y, ldy, m, tiles);
        else if (ntiles == 3) gemv_w8s_kernel<3, WB, 1, 3><<<blocks, WB * 32, 0, st>>>(w.q, w.s, w.k, xp, y, ldy, m, tiles);
        else gemv_w8s_kernel<4, WB, 1, 3><<<blocks, WB * 32, 0, st>>>(w.q, w.s, w.k, xp, y, ldy, m, tiles);
        return;
    }
    if (ntiles == 2) gemv_w8p_kernel<2><<<tiles, nw * 32, 0, st>>>(w.q, w.s, w.k, xp, y, ldy, m);
    else if (ntiles == 3) gemv_w8p_kernel<3><<<tiles, nw * 32, 0, st>>>(w.q, w.s, w.k, xp, y, ldy, m);
    else gemv_w8p_kernel<4><<<tiles, nw * 32, 0, st>>>(w.q, w.s, w.k, xp, y, ldy, m);
}

void quantize_w8(const bf16* src, int n, int k, uint4* dst_q, bf16* dst_s, cudaStream_t st) {
    CHECK(n % 16 == 0 && k % 128 == 0, "quantize_w8 %d %d", n, k);
    const size_t warps = (size_t)n * (k / 128);
    quantize_w8_kernel<<<(warps * 32 + 255) / 256, 256, 0, st>>>(src, n, k, (uint8_t*)dst_q, dst_s);
}

void quantize_w4_rtn(const bf16* src, int n, int k, int32_t* dst_q, bf16* dst_s, cudaStream_t st) {
    CHECK(k % 128 == 0, "rtn k %d", k);
    const size_t warps = (size_t)n * (k / 128);
    quantize_rtn_kernel<<<(warps * 32 + 255) / 256, 256, 0, st>>>(src, n, k, dst_q, dst_s);
}

void gemv_workspace_init() {
    int dev;
    CUDA_CHECK(cudaGetDevice(&dev));
    if (!g_xpack[dev]) {
        CUDA_CHECK(cudaMalloc(&g_xpack[dev], (size_t)kMaxBatchRows * kXpackMaxK * 2));
        skinny_init();
        gemv_x_init();
        // The DeltaNet ab projection (gemv_w16h) runs on a second stream beside dn_in (gemv_w4hw). Left to
        // the driver the two get different smem carveouts (w16h: 13 KB/block x 5 -> presumably the 100 KB
        // one, w4hw: 5 x 8.5 KB -> 64 KB); an SM cannot run blocks of both configs at once, so dn_in loses the
        // SMs holding ab blocks until they drain (engine pattern, 24 rows 25 -> 42 us/call, 32 rows 45 ->
        // 59). One carveout for both (50 % -> 64 KB on w16h; 64 %+ on w4hw costs it L1) removes that, bit
        // for bit the same results. QE_W16H_CARVE=-1: driver default
        static const int carve = getenv("QE_W16H_CARVE") ? atoi(getenv("QE_W16H_CARVE")) : 50;
        if (carve >= 0) gemv_set_carveout(carve, 1);
    }
}


int g_w4hw3_mode = -1;  // microbench override: -1 rule, 0 uncapped NT=3, 1 capped (5 blocks/SM)

static int sm_count() {
    static int n[16] = {};
    int dev;
    CUDA_CHECK(cudaGetDevice(&dev));
    if (!n[dev]) CUDA_CHECK(cudaDeviceGetAttribute(&n[dev], cudaDevAttrMultiProcessorCount, dev));
    return n[dev];
}

bool gemv_h16_on() {
    static const bool on = !getenv("QE_GEMV_H16") || atoi(getenv("QE_GEMV_H16"));  // default on, 0 disables
    return on;
}

void xh_pack(const bf16* x, int ldx, int K, int m, __half* xh, int ldxh, cudaStream_t st) {
    CHECK(K % 64 == 0 && ldxh % 64 == 0, "xh_pack %d %d", K, ldxh);
    const size_t n = (size_t)m * K;
    xh_pack_kernel<<<(unsigned)((n + 255) / 256), 256, 0, st>>>(x, ldx, K, m, xh, ldxh);
}

void gemv_w4h(const W4& w, const __half* xh, int ldx, float* y, int ldy, int m, cudaStream_t st,
              __half* yh, int ldyh) {
    if (m > kMaxGemvW4Tokens) {  // 33..64 rows: the wide kernel (same numerics)
        CHECK(gemv_w4hx(w, xh, ldx, y, ldy, m, st, yh, ldyh), "gemv_w4hx %d %d %d", w.n, w.k, m);
        return;
    }
    CHECK(w.n % 16 == 0 && w.k % 128 == 0 && m >= 1 && m <= kMaxGemvW4Tokens && ldx % 64 == 0 &&
          (reinterpret_cast<uintptr_t>(xh) & 127) == 0 && (!yh || ldyh % 64 == 0),
          "gemv_w4h %d %d %d %d", w.n, w.k, m, ldx);
    const int tiles = w.n / 16;
    const int nw = pick_warps(tiles, w.k / 128, true);
    const int ntt = (m + 7) / 8;
    // 25..32 rows on matrices of >= 352 tiles (gate_up, DeltaNet in, qkv): tile pairs per warp (gemv_x.cu), the
    // B fragments feed twice the mma: 32 rows gate_up 66 -> 56 us, dn_in 47 -> 37, qkv 45 -> 36 (QE_W4HP=0 off)
    static const bool hp_on = !getenv("QE_W4HP") || atoi(getenv("QE_W4HP"));
    if (hp_on && m >= 25 && tiles >= 352 && gemv_w4hx(w, xh, ldx, y, ldy, m, st, yh, ldyh)) return;
    if (ntt > 1 && skinny_use(w.n, m) && skinny_w4h(w, xh, ldx, y, ldy, m, st, yh, ldyh)) return;
    CHECK(ntt == 1 || nw <= kW4hwMaxWarps, "gemv_w4hw slices %d", nw);
#define QE_W4H(NT_) gemv_w4hw##NT_<<<tiles, nw * 32, 0, st>>>(w.q, w.s, w.k, xh, ldx, y, ldy, m, w.glu, yh, ldyh)
    if (ntt == 1) gemv_w4h_kernel<1><<<tiles, nw * 32, 0, st>>>(w.q, w.s, w.k, xh, ldx, y, ldy, m, w.glu, yh, ldyh);
    else if (ntt == 2) QE_W4H(2);
    // NT = 3: the 5-blocks/SM kernel only where it saves a wave (4 x SMs < tiles <= 5 x SMs: dn_in);
    // elsewhere the uncapped one is 2-6 % faster (down/qkv/o at 24 rows)
    else if (ntt == 3 && (g_w4hw3_mode == 1 || (g_w4hw3_mode < 0 && tiles > 4 * sm_count() && tiles <= 5 * sm_count())))
        QE_W4H(3);
    else if (ntt == 3) QE_W4H(3w);
    else QE_W4H(4);
#undef QE_W4H
}

void gemv_w4(const W4& w, const bf16* x, int ldx, float* y, int ldy, int m, cudaStream_t st) {
    if (m > kMaxGemvW4Tokens && m <= kMaxBatchRows) {  // 33..64 rows (draft head of a wide step): two calls
        gemv_w4(w, x, ldx, y, ldy, 32, st);
        gemv_w4(w, x + (size_t)32 * ldx, ldx, y + (size_t)32 * ldy, ldy, m - 32, st);
        return;
    }
    CHECK(w.n % 16 == 0 && w.k % 128 == 0 && m >= 1 && m <= kMaxGemvW4Tokens, "gemv_w4 %d %d %d",
          w.n, w.k, m);
    const int tiles = w.n / 16;
    const int nw = pick_warps(tiles, w.k / 128, true);
    if (m <= 8) {
        gemv_w4_kernel<1><<<tiles, nw * 32, 0, st>>>(w.q, w.s, w.k, x, ldx, y, ldy, m, w.glu);
        return;
    }
    if (skinny_use(w.n, m) && skinny_w4(w, x, ldx, y, ldy, m, st)) return;
    int dev;
    CUDA_CHECK(cudaGetDevice(&dev));
    uint4* xp = g_xpack[dev];
    CHECK(xp && w.k <= kXpackMaxK, "gemv workspace (k=%d)", w.k);
    // 8-token mma tiles: 2 (m <= 16), 3 (m <= 24) or 4 - past 16 rows the kernel is no longer bound
    // by DRAM, so a padding tile costs real time
    const int ntt = (m + 7) / 8, ntiles = ntt <= 2 ? 2 : ntt;
    const int items = ntiles * (w.k / 32) * 32;
    xpack_kernel<<<(items + 255) / 256, 256, 0, st>>>(x, ldx, w.k, m, ntiles, xp, false);
    static const int rt_env = getenv("QE_GEMV_RT") ? atoi(getenv("QE_GEMV_RT")) : 1;
    const int rt = tiles % rt_env == 0 ? rt_env : 1;
    const int nwr = pick_warps(tiles / rt, w.k / 128, true);
#define QE_W4P(NT_, RT_) gemv_w4p_kernel<NT_, RT_><<<tiles / RT_, nwr * 32, 0, st>>>(w.q, w.s, w.k, xp, y, ldy, m, w.glu)
    if (ntiles == 2) {
        if (rt == 4) QE_W4P(2, 4); else if (rt == 2) QE_W4P(2, 2); else QE_W4P(2, 1);
    } else if (ntiles == 3) {
        if (rt == 4) QE_W4P(3, 4); else if (rt == 2) QE_W4P(3, 2); else QE_W4P(3, 1);
    } else {
        if (rt == 4) QE_W4P(4, 4); else if (rt == 2) QE_W4P(4, 2); else QE_W4P(4, 1);
    }
#undef QE_W4P
}

void gemv_w16(const W16& w, const bf16* x, int ldx, float* y, int ldy, int m, cudaStream_t st,
              int ksplit, size_t ystride) {
    CHECK(w.n % 16 == 0 && w.k % 16 == 0 && m >= 1 && m <= kMaxBatchRows, "gemv_w16 %d %d %d",
          w.n, w.k, m);
    if (m > 32) {  // 33..64 rows: two calls (rows are independent)
        gemv_w16(w, x, ldx, y, ldy, 32, st, ksplit, ystride);
        gemv_w16(w, x + (size_t)32 * ldx, ldx, y + (size_t)32 * ldy, ldy, m - 32, st, ksplit, ystride);
        return;
    }
    const int tiles = w.n / 16;
    const int nw = pick_warps(tiles * ksplit, w.k / 16 / ksplit);
    if (m <= 8)
        gemv_w16_kernel<1><<<dim3(tiles, ksplit), nw * 32, 0, st>>>(w.w, w.k, x, ldx, y, ldy, m, ystride);
    else if (m <= 16)
        gemv_w16_kernel<2><<<dim3(tiles, ksplit), nw * 32, 0, st>>>(w.w, w.k, x, ldx, y, ldy, m, ystride);
    else if (m <= 24)
        gemv_w16_kernel<3><<<dim3(tiles, ksplit), nw * 32, 0, st>>>(w.w, w.k, x, ldx, y, ldy, m, ystride);
    else
        gemv_w16_kernel<4><<<dim3(tiles, ksplit), nw * 32, 0, st>>>(w.w, w.k, x, ldx, y, ldy, m, ystride);
}


void gemv_w16h(const W16& w, const __half* xh, int ldx, float* y, int ldy, int m, cudaStream_t st,
               int ksplit, size_t ystride) {
    CHECK(w.n % 16 == 0 && w.k % 64 == 0 && m >= 1 && m <= kMaxBatchRows && ldx % 64 == 0,
          "gemv_w16h %d %d %d", w.n, w.k, m);
    if (m > 32) {  // 33..64 rows: two calls (rows are independent)
        gemv_w16h(w, xh, ldx, y, ldy, 32, st, ksplit, ystride);
        gemv_w16h(w, xh + (size_t)32 * ldx, ldx, y + (size_t)32 * ldy, ldy, m - 32, st, ksplit, ystride);
        return;
    }
    const int tiles = w.n / 16;
    const int nw = pick_warps(tiles * ksplit, w.k / 16 / ksplit);
    if (m <= 8)
        gemv_w16h_kernel<1><<<dim3(tiles, ksplit), nw * 32, 0, st>>>(w.w, w.k, xh, ldx, y, ldy, m, ystride);
    else if (m <= 16)
        gemv_w16h_kernel<2><<<dim3(tiles, ksplit), nw * 32, 0, st>>>(w.w, w.k, xh, ldx, y, ldy, m, ystride);
    else if (m <= 24)
        gemv_w16h_kernel<3><<<dim3(tiles, ksplit), nw * 32, 0, st>>>(w.w, w.k, xh, ldx, y, ldy, m, ystride);
    else
        gemv_w16h_kernel<4><<<dim3(tiles, ksplit), nw * 32, 0, st>>>(w.w, w.k, xh, ldx, y, ldy, m, ystride);
}

void w16_to_h(const W16& src, uint4* dst, cudaStream_t st) {
    const size_t n = (size_t)src.n * src.k / 2;
    w16_to_h_kernel<<<(unsigned)((n + 255) / 256), 256, 0, st>>>(reinterpret_cast<const uint32_t*>(src.w),
                                                                reinterpret_cast<uint32_t*>(dst), n);
}

void repack_w4(const int32_t* src_q, const bf16* src_s, int n, int k, uint4* dst_q, bf16* dst_s,
               cudaStream_t st) {
    CHECK(n % 16 == 0 && k % 128 == 0, "repack_w4 %d %d", n, k);
    size_t words = (size_t)n * k / 8;
    repack_w4_kernel<<<(words + 255) / 256, 256, 0, st>>>(src_q, n, k,
                                                         reinterpret_cast<uint32_t*>(dst_q));
    size_t ns = (size_t)n * (k / 128);
    repack_scales_kernel<<<(ns + 255) / 256, 256, 0, st>>>(src_s, n, k, dst_s);
}

void repack_w16(const bf16* src, int n, int k, uint4* dst, cudaStream_t st) {
    CHECK(n % 16 == 0 && k % 16 == 0, "repack_w16 %d %d", n, k);
    size_t words = (size_t)n * k / 2;
    repack_w16_kernel<<<(words + 255) / 256, 256, 0, st>>>(src, n, k,
                                                          reinterpret_cast<uint32_t*>(dst));
}

// shared-memory carveout preference (percent, -1 = driver default) of the decode GEMV kernels on the current
// device; which: 0 all, 1 gemv_w16h only (the engine default, gemv_workspace_init), 2 the INT4 ones only
void gemv_set_carveout(int pct, int which) {
    const int v = pct < 0 ? (int)cudaSharedmemCarveoutDefault : pct;
    const auto a = cudaFuncAttributePreferredSharedMemoryCarveout;
    if (which != 1) {
        CUDA_CHECK(cudaFuncSetAttribute(gemv_w4hw2, a, v));
        CUDA_CHECK(cudaFuncSetAttribute(gemv_w4hw3, a, v));
        CUDA_CHECK(cudaFuncSetAttribute(gemv_w4hw3w, a, v));
        CUDA_CHECK(cudaFuncSetAttribute(gemv_w4hw4, a, v));
        CUDA_CHECK(cudaFuncSetAttribute(gemv_w4h_kernel<1>, a, v));
    }
    if (which == 2) return;
    CUDA_CHECK(cudaFuncSetAttribute(gemv_w16h_kernel<1>, a, v));
    CUDA_CHECK(cudaFuncSetAttribute(gemv_w16h_kernel<2>, a, v));
    CUDA_CHECK(cudaFuncSetAttribute(gemv_w16h_kernel<3>, a, v));
    CUDA_CHECK(cudaFuncSetAttribute(gemv_w16h_kernel<4>, a, v));
}

// debug: resident blocks per SM of the wide kernels at a given block size

void gemv_occupancy_report(int threads) {
    int b2, b3, b4, h1, p4;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&b2, gemv_w4hw2, threads, 0));
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&b3, gemv_w4hw3, threads, 0));
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&b4, gemv_w4hw4, threads, 0));
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&h1, gemv_w4h_kernel<1>, threads, 0));
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&p4, gemv_w8p_kernel<4>, threads, 0));
    printf("threads %d: w4h<1> %d, w4hw<2> %d, w4hw<3> %d, w4hw<4> %d, w8p<4> %d blocks/SM\n", threads, h1,
           b2, b3, b4, p4);
}
