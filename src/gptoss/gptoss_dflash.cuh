// [gptoss-r15-dflash] Kernels of the DFlash block drafter (z-lab/gpt-oss-120b-DFlash: 8 Qwen3-style layers, hidden 2880,
// 64 q / 8 kv heads x 64 with q / k norm, MLP 7680, context injected as K/V made from 5 residual taps of the target).
// Own forward, written from the config and the tensor shapes. The drafter only proposes tokens: the target's verify step
// decides, so nothing here has to be bit-stable (fp16 tensor-core attention, bf16 rows).
// The dense products run through k_gemv_m, the collectives through k_ar_t, the draft rows' argmax through the target's
// certified head screen (gptoss_kernels.cuh).
#pragma once
#include <cuda_fp16.h>
#include <map>
#include <utility>
#include <vector>

namespace gptoss {

constexpr int DF_L = 8, DF_MAXB = 8, DF_NTAP = 5, DF_INTER = 7680, DF_FC = DF_NTAP * 2880;
constexpr int DF_TK = 64;          // keys per attention tile
constexpr int DF_PW = 66;          // attention partial of a (query, tile): m, l, o[64]
constexpr int DF_RING = 256;       // rows of the tap ring (rows of the last decode steps)
constexpr int DF_PAD = 64;         // store rows behind the context rows: the block's own K/V (and whole-tile reads)
constexpr int DF_MASK = 200000;    // mask token of the block's draft rows
constexpr int DF_TAP_LAYER[DF_NTAP] = {1, 9, 17, 25, 33};   // residual after these target layers (0-based)

struct DfPar {   // one drafter pass (uploaded per pass)
    int na;        // context rows made in this pass
    int B;         // block rows (the token and B - 1 mask rows); < 2: no block
    int bpos;      // position of block row 0
    int nv;        // valid context rows of the store: slots [0, nv)
    int cap;       // slot of block row 0 (behind the context rows)
    int ring[16];  // new context row -> row of the tap ring
    int cpos[16];  // new context row -> position
    int cslot[16]; // new context row -> store slot
};

__device__ __forceinline__ unsigned short df_h(float v) {   // fp16 bits, finite
    return __half_as_ushort(__float2half_rn(fminf(fmaxf(v, -65000.f), 65000.f)));
}
__device__ __forceinline__ void mma_f16(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// taps of a decode step: the residual rows after target layer DF_TAP_LAYER[tap] -> bf16 ring rows (posp[3 MAXR] + row)
__global__ void __launch_bounds__(960) k_df_tap(const float* h, bf16* ring, const int* posp, int tap) {
    const int rr = blockIdx.x, w = threadIdx.x;
    const float* s = h + (size_t)rr * 2880;
    bf16* d = ring + ((size_t)((posp[3 * MAXR] + rr) % DF_RING) * DF_NTAP + tap) * 2880;
#pragma unroll
    for (int j = 0; j < 3; j++) d[3 * w + j] = __float2bfloat16_rn(s[3 * w + j]);
}
// taps of prefill rows (the owner's residual rows) -> bf16 [survivor][5 x 2880] (dst = the tap's column block of row 0):
// own row i of the lane goes to survivor b + i (i < nh) or b + nh + i - s0 (i >= s0), rows in between are not kept
// (they leave the store before the chunk ends), survivors at or past pcap are dropped (a hole: lower acceptance only)
__global__ void __launch_bounds__(960) k_df_ptap(const float* h, bf16* dst, int b, int nh, int s0, int pcap) {
    const int i = blockIdx.x;
    if (i >= nh && i < s0) return;
    const int k = b + (i < nh ? i : nh + i - s0);
    if (k >= pcap) return;
    const float* s = h + (size_t)i * 2880;
    bf16* d = dst + (size_t)k * DF_FC;
#pragma unroll
    for (int j = 0; j < 3; j++) d[3 * threadIdx.x + j] = __float2bfloat16_rn(s[3 * threadIdx.x + j]);
}
// ring rows of the pass' new context rows -> contiguous [na][5 x 2880]
__global__ void k_df_gather(const bf16* ring, const DfPar* par, bf16* out) {
    const int a = blockIdx.x;
    const uint4* s = (const uint4*)(ring + (size_t)par->ring[a] * DF_FC);
    uint4* d = (uint4*)(out + (size_t)a * DF_FC);
    for (int i = threadIdx.x; i < DF_FC / 8; i += blockDim.x) d[i] = s[i];
}

// RMS norm over the 64 dims of a head + RoPE of a pair (d, d + 32); lane d holds a = x[d], b = x[d + 32]
__device__ __forceinline__ void df_norm_rope(float& a, float& b, const bf16* nw, int d, int pos, const float* invf, float att, bool rope) {
    float ss = a * a + b * b;
    for (int o = 16; o; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
    const float rs = rsqrtf(ss * (1.f / 64.f) + 1e-5f);
    a = a * rs * bf2f(nw[d]);
    b = b * rs * bf2f(nw[d + 32]);
    if (rope) {
        const float fr = (float)pos * invf[d];
        const float c = cosf(fr) * att, s = sinf(fr) * att;
        const float x1 = a * c - b * s, x2 = b * c + a * s;
        a = x1;
        b = x2;
    }
}
// new context rows: Y [na][8 layers][k rows of the card | v rows] (bias added) -> k_norm + RoPE at the row's position ->
// the layer stores (fp16 [slot][kvn][64]). Block = (row, layer, k or v head), 32 threads.
__global__ void __launch_bounds__(32) k_df_kvstore(const float* Y, int kvn, const bf16* knw, const DfPar* par, const float* invf,
                                                   float att, __half* Ks, __half* Vs, size_t lrows) {
    const int a = blockIdx.x, hh = blockIdx.y, l = hh / (2 * kvn), j = hh % (2 * kvn), d = threadIdx.x;
    const float* y = Y + ((size_t)a * DF_L * 2 * kvn + hh) * 64;
    float x0 = y[d], x1 = y[d + 32];
    const bool isk = j < kvn;
    if (isk) df_norm_rope(x0, x1, knw + l * 64, d, par->cpos[a], invf, att, true);
    unsigned short* dst = (unsigned short*)(isk ? Ks : Vs) + (((size_t)l * lrows + par->cslot[a]) * kvn + (isk ? j : j - kvn)) * 64;
    dst[d] = df_h(x0);
    dst[d + 32] = df_h(x1);
}
// prefill survivors: the same from Y [n][8 layers][k | v] (bias added), row a at position pos[a] (distinct slots)
__global__ void __launch_bounds__(32) k_df_kvstore_p(const float* Y, int kvn, const bf16* knw, const int* pos, int cap, int head, const float* invf,
                                                     float att, __half* Ks, __half* Vs, size_t lrows) {
    const int a = blockIdx.x, hh = blockIdx.y, l = hh / (2 * kvn), j = hh % (2 * kvn), d = threadIdx.x;
    const int p = pos[a];
    const int slot = p < cap ? p : head + (p - head) % (cap - head);
    const float* y = Y + ((size_t)a * DF_L * 2 * kvn + hh) * 64;
    float x0 = y[d], x1 = y[d + 32];
    const bool isk = j < kvn;
    if (isk) df_norm_rope(x0, x1, knw + l * 64, d, p, invf, att, true);
    unsigned short* dst = (unsigned short*)(isk ? Ks : Vs) + (((size_t)l * lrows + slot) * kvn + (isk ? j : j - kvn)) * 64;
    dst[d] = df_h(x0);
    dst[d + 32] = df_h(x1);
}
// block rows: Y [B][q rows | k rows | v rows] (bias added) -> q: q_norm + RoPE, x 1/8 -> Qh [row][nq][64]; k / v -> the
// layer store's rows cap + row. Block = (row, head), 32 threads.
__global__ void __launch_bounds__(32) k_df_qkv(const float* Y, int nq, int kvn, const bf16* qnw, const bf16* knw, const DfPar* par,
                                               const float* invf, float att, __half* Qh, __half* Kl, __half* Vl) {
    const int b = blockIdx.x, h = blockIdx.y, d = threadIdx.x;
    const float* y = Y + ((size_t)b * (nq + 2 * kvn) + h) * 64;
    float x0 = y[d], x1 = y[d + 32];
    const int pos = par->bpos + b;
    unsigned short* dst;
    if (h < nq) {
        df_norm_rope(x0, x1, qnw, d, pos, invf, att, true);
        x0 *= 0.125f;
        x1 *= 0.125f;
        dst = (unsigned short*)Qh + ((size_t)b * nq + h) * 64;
    } else if (h < nq + kvn) {
        df_norm_rope(x0, x1, knw, d, pos, invf, att, true);
        dst = (unsigned short*)Kl + ((size_t)(par->cap + b) * kvn + (h - nq)) * 64;
    } else
        dst = (unsigned short*)Vl + ((size_t)(par->cap + b) * kvn + (h - nq - kvn)) * 64;
    dst[d] = df_h(x0);
    dst[d + 32] = df_h(x1);
}

// Attention of the block's queries over one tile of 64 keys of one KV head, on fp16 tensor cores. Block = (tile, KV head);
// tile < nct: context slots [64 tile, ..) (the valid ones), tile == nct: the block's own rows (non-causal). The 8 B queries
// of the KV head (row b, q head j: index 8 b + j) in groups of 16. Partial per (query, tile): m, l, o[64].
// The k index of a 16-wide product chunk c: lane t supplies elements 16 c + 4 t .. + 3 (both operands the same way).
__global__ void __launch_bounds__(256) k_df_attn(const __half* Qh, const __half* Kl, const __half* Vl, const DfPar* par, int nq, int kvn,
                                                 float* apart, int NT) {
    __shared__ float S[64 * 64];
    __shared__ __align__(8) unsigned short Ph[64 * 64];
    const int tile = blockIdx.x, g = blockIdx.y;
    const int nv = par->nv, B = par->B, nct = (nv + DF_TK - 1) / DF_TK;
    if (tile > nct) return;
    const int key0 = tile < nct ? tile * DF_TK : par->cap;
    const int nk = tile < nct ? min(DF_TK, nv - tile * DF_TK) : B;
    const int nqr = 8 * B, nqg = (nqr + 15) / 16;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g8 = lane >> 2, t = lane & 3;
    {   // scores: warp = keys 8 warp .. + 7 of the tile
        const unsigned short* kr = (const unsigned short*)Kl + ((size_t)(key0 + 8 * warp + g8) * kvn + g) * 64;
        uint2 kb[4];
#pragma unroll
        for (int c = 0; c < 4; c++) kb[c] = *(const uint2*)(kr + 16 * c + 4 * t);
        for (int qg = 0; qg < nqg; qg++) {
            const int qa = 16 * qg + g8;   // and qa + 8 = the same q head of the next row
            const unsigned short* q0 = (const unsigned short*)Qh + ((size_t)(qa >> 3) * nq + 8 * g + (qa & 7)) * 64;
            const unsigned short* q1 = q0 + (size_t)nq * 64;
            float c4[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
            for (int c = 0; c < 4; c++) {
                const uint2 a = *(const uint2*)(q0 + 16 * c + 4 * t), b = *(const uint2*)(q1 + 16 * c + 4 * t);
                mma_f16(c4, a.x, b.x, a.y, b.y, kb[c].x, kb[c].y);
            }
            const int k = 8 * warp + 2 * t;
            S[qa * 64 + k] = c4[0];
            S[qa * 64 + k + 1] = c4[1];
            S[(qa + 8) * 64 + k] = c4[2];
            S[(qa + 8) * 64 + k + 1] = c4[3];
        }
    }
    __syncthreads();
    {   // softmax over the tile's valid keys: 4 lanes per query, 16 keys each
        const int qi = threadIdx.x >> 2, sub = threadIdx.x & 3;
        float m = NEG_INF;
        float s[16];
#pragma unroll
        for (int i = 0; i < 16; i++) {
            const int k = sub * 16 + i;
            s[i] = k < nk ? S[qi * 64 + k] : NEG_INF;
            m = fmaxf(m, s[i]);
        }
        m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 1));
        m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 2));
        float l = 0.f;
#pragma unroll
        for (int i = 0; i < 16; i++) {
            const int k = sub * 16 + i;
            const float p = k < nk ? __expf(s[i] - m) : 0.f;
            l += p;
            Ph[qi * 64 + k] = __half_as_ushort(__float2half_rn(p));
        }
        l += __shfl_xor_sync(0xffffffffu, l, 1);
        l += __shfl_xor_sync(0xffffffffu, l, 2);
        if (sub == 0 && qi < nqr) {
            float* o = apart + (((size_t)g * NT + tile) * 64 + qi) * DF_PW;
            o[0] = m;
            o[1] = l;
        }
    }
    __syncthreads();
    {   // P . V: warp = dims 8 warp .. + 7
        const int dim = 8 * warp + g8;
        const unsigned short* vr = (const unsigned short*)Vl + ((size_t)key0 * kvn + g) * 64 + dim;
        const size_t ks = (size_t)kvn * 64;   // elements between two keys
        uint2 vb[4];
#pragma unroll
        for (int c = 0; c < 4; c++) {
            const unsigned short* v = vr + (size_t)(16 * c + 4 * t) * ks;
            vb[c].x = (uint32_t)v[0] | ((uint32_t)v[ks] << 16);
            vb[c].y = (uint32_t)v[2 * ks] | ((uint32_t)v[3 * ks] << 16);
        }
        for (int qg = 0; qg < nqg; qg++) {
            const int qa = 16 * qg + g8;
            float c4[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
            for (int c = 0; c < 4; c++) {
                const uint2 a = *(const uint2*)(Ph + qa * 64 + 16 * c + 4 * t), b = *(const uint2*)(Ph + (qa + 8) * 64 + 16 * c + 4 * t);
                mma_f16(c4, a.x, b.x, a.y, b.y, vb[c].x, vb[c].y);
            }
            const int d = 8 * warp + 2 * t;
            float* o = apart + (((size_t)g * NT + tile) * 64 + qa) * DF_PW + 2;
            if (qa < nqr) {
                o[d] = c4[0];
                o[d + 1] = c4[1];
            }
            if (qa + 8 < nqr) {
                o[8 * DF_PW + d] = c4[2];
                o[8 * DF_PW + d + 1] = c4[3];
            }
        }
    }
}
// tiles of a query -> AO bf16 [row][nq x 64]. Block = (q head, row), 64 threads (thread = dim).
__global__ void __launch_bounds__(64) k_df_comb(const float* apart, const DfPar* par, int nq, int NT, bf16* AO) {
    const int h = blockIdx.x, b = blockIdx.y, d = threadIdx.x;
    const int g = h >> 3, qi = 8 * b + (h & 7);
    const int nt = (par->nv + DF_TK - 1) / DF_TK + 1;
    const float* p = apart + ((size_t)g * NT * 64 + qi) * DF_PW;
    const size_t ts = (size_t)64 * DF_PW;
    float M = NEG_INF;
    for (int t = 0; t < nt; t++) M = fmaxf(M, p[t * ts]);
    float L = 0.f, o = 0.f;
    for (int t = 0; t < nt; t++) {
        const float e = __expf(p[t * ts] - M);
        L += p[t * ts + 1] * e;
        o += p[t * ts + 2 + d] * e;
    }
    AO[((size_t)b * nq + h) * 64 + d] = __float2bfloat16_rn(o / L);
}
// [phase 2] 4-bit dense product out[col][row] = W[row] . x[col] (+ bias) for R <= 8 NC columns. W = 4-bit codes
// [N][ldc bytes] (byte i = inputs 2 i (low nibble) and 2 i + 1) with one scale per 32 inputs [N][lds]: INT4 codes
// (nibble - 8, -7..7) with an fp16 scale (the P2 gate's i4 form). Block = rows 16 b .. 16 b + 15; a lane holds whole
// groups of 32 inputs (group 4 q + t of rows g and g + 8: one 16 B load each), so the k order inside a tensor-core
// product is the lane's own (x is read in the same order). K % 32 == 0, N % 16 == 0, x rows and ldc 16 B aligned.
// acc: out += instead of out =.
__device__ __forceinline__ uint32_t q4_frag(uint32_t w, int sh, float sc) {
    const uint32_t b = (w >> sh) & 255u;
    const uint32_t h = 0x64006400u | (b & 15u) | ((b & 0xF0u) << 12);   // 1024 + nibble as fp16 pair
    const float2 v = __half22float2(__hsub2(*(const __half2*)&h, __float2half2_rn(1032.f)));
    const bf162 r = __floats2bfloat162_rn(v.x * sc, v.y * sc);
    return *(const uint32_t*)&r;
}
template <int NC>
__global__ void __launch_bounds__(32 * DM_MAXW) k_q4_m(const uint8_t* W, const __half* S, int N, int K, size_t ldc, size_t lds,
                                                       const bf16* x, int ldx, int R, const bf16* bias, float* out, int ldo, int acc) {
    extern __shared__ float red[];   // dm_smem(NC, warps) bytes
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3, nw = blockDim.x >> 5;
    const int G = K >> 5, Q = (G + 3) >> 2, q0 = warp * Q / nw, q1 = (warp + 1) * Q / nw;
    const int ra = blockIdx.x * 16 + g, rb = ra + 8;
    const float ba = bias ? bf2f(bias[ra]) : 0.f, bb = bias ? bf2f(bias[rb]) : 0.f;
    const uint4* wa = (const uint4*)(W + (size_t)ra * ldc);
    const uint4* wb = (const uint4*)(W + (size_t)rb * ldc);
    const __half *sa = S + (size_t)ra * lds, *sb = S + (size_t)rb * lds;
    const uint4* px[NC];
    bool xon[NC];
#pragma unroll
    for (int n = 0; n < NC; n++) {
        xon[n] = 8 * n + g < R;
        px[n] = (const uint4*)(x + (size_t)(xon[n] ? 8 * n + g : 0) * ldx);
    }
    float c[4 * NC];
#pragma unroll
    for (int i = 0; i < 4 * NC; i++) c[i] = 0.f;
    for (int q = q0; q < q1; q++) {
        const int gi = 4 * q + t;
        const bool on = gi < G;   // the last quad of groups may be partial: those lanes add zeros (every lane takes part in the products)
        const uint4 za = make_uint4(0, 0, 0, 0);
        const uint4 a = on ? wa[gi] : za, b = on ? wb[gi] : za;
        const float fa = on ? __half2float(sa[gi]) : 0.f, fb = on ? __half2float(sb[gi]) : 0.f;
        const uint32_t aw[4] = {a.x, a.y, a.z, a.w}, bw[4] = {b.x, b.y, b.z, b.w};
#pragma unroll
        for (int k = 0; k < 4; k++) {   // 8 inputs per k: two products of 16 (4 per lane)
            uint4 xv[NC];
#pragma unroll
            for (int n = 0; n < NC; n++) xv[n] = (on && xon[n]) ? px[n][4 * gi + k] : za;
#pragma unroll
            for (int h = 0; h < 2; h++) {
                const int sh = 16 * h;
                const uint32_t a0 = q4_frag(aw[k], sh, fa), a2 = q4_frag(aw[k], sh + 8, fa);
                const uint32_t a1 = q4_frag(bw[k], sh, fb), a3 = q4_frag(bw[k], sh + 8, fb);
#pragma unroll
                for (int n = 0; n < NC; n++) mma_bf16(c + 4 * n, a0, a1, a2, a3, h ? xv[n].z : xv[n].x, h ? xv[n].w : xv[n].y);
            }
        }
    }
    if (nw > 1) {
#pragma unroll
        for (int i = 0; i < 4 * NC; i++) red[(warp * 4 * NC + i) * 32 + lane] = c[i];
        __syncthreads();
        if (warp) return;
        for (int w = 1; w < nw; w++)
#pragma unroll
            for (int i = 0; i < 4 * NC; i++) c[i] += red[(w * 4 * NC + i) * 32 + lane];
    }
#pragma unroll
    for (int n = 0; n < NC; n++)
#pragma unroll
        for (int h = 0; h < 2; h++) {
            const int col = 8 * n + 2 * t + h;
            if (col < R) {
                float* o = out + (size_t)col * ldo;
                o[ra] = c[4 * n + h] + ba + (acc ? o[ra] : 0.f);
                o[rb] = c[4 * n + 2 + h] + bb + (acc ? o[rb] : 0.f);
            }
        }
}
__host__ __device__ constexpr int q4_warps(int K) { return (((K >> 5) + 3) / 4 + 1) / 2 > DM_MAXW ? DM_MAXW : (((K >> 5) + 3) / 4 + 1) / 2; }

// act = silu(gate) * up; GU [B][gate rows jn | up rows jn]
__global__ void k_df_act(const float* GU, int jn, int B, bf16* act) {
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < (size_t)B * jn; i += (size_t)gridDim.x * blockDim.x) {
        const size_t b = i / jn, j = i % jn;
        const float gt = GU[b * 2 * jn + j], up = GU[b * 2 * jn + jn + j];
        act[i] = __float2bfloat16_rn(gt / (1.f + __expf(-gt)) * up);
    }
}


// ---- host side (gptoss_dflash.inc)
struct DfQ4 {   // a 4-bit matrix (k_q4_m): codes [N][ldc] bytes, fp16 scales [N][lds]
    uint8_t* c = nullptr;
    __half* s = nullptr;
    int N = 0, K = 0;
    size_t ldc = 0, lds = 0;
    DfQ4 rows(int r0, int n) const { DfQ4 q = *this; q.c += (size_t)r0 * ldc; q.s += (size_t)r0 * lds; q.N = n; return q; }
    DfQ4 cols(int k0, int k) const { DfQ4 q = *this; q.c += k0 / 2; q.s += k0 / 32; q.K = k; return q; }
};
struct DfLayerW {
    DfQ4 wq, wo, wgu, wdn;   // q rows of the card's heads; o_proj columns; gate | up rows; down columns (MLP slice)
    bf16 *bq = nullptr, *bkv = nullptr, *bo = nullptr;
    bf16 *n1 = nullptr, *n2 = nullptr, *qn = nullptr, *kn = nullptr;
};
struct DfRank {   // one card
    int jn = 0, j0 = 0, nkv = 0, NT = 0;
    size_t lrows = 0, srows = 0;   // store rows per layer of a slot; per slot
    size_t bytes = 0;
    char* abase = nullptr;         // the drafter's one device allocation (df_put)
    size_t acap = 0, aoff = 0;
    DfLayerW L[DF_L];
    DfQ4 wfc;                      // fc, whole [2880][14400] on every card (no collective for context rows)
    DfQ4 wkv;                      // k | v rows of the card's heads, all 8 layers [8][2 kvn 64][2880]: block rows and context rows
    bf16* ptap = nullptr;          // prefill: taps of the chunk's own rows that stay in the store [pcap][5 x 2880]
    int pcap = 0;
    int pbase[8] = {}, pnh[8] = {}, ps0[8] = {};   // per lane: first survivor index, survivors at the start, first tail row, own rows
    int poff[4] = {}, pcnt[4] = {}, pm = 0;   // prefill: compact survivor rows of each card, all cards
    int* ppos = nullptr;           // prefill: positions of the compact survivor rows of all cards (device)
    int* ppos_host = nullptr;
    bf16 *hn = nullptr, *nf = nullptr, *knall = nullptr, *bkvall = nullptr;
    bf16 *ring = nullptr, *Tg = nullptr, *C = nullptr, *XN = nullptr, *AO = nullptr, *act = nullptr;
    float *Hc = nullptr, *Hs = nullptr, *part = nullptr, *Y = nullptr, *Ykv = nullptr, *GU = nullptr, *apart = nullptr;
    __half *Qh = nullptr, *Ks = nullptr, *Vs = nullptr;
    DfPar *par = nullptr, *par_host = nullptr;
    float *h_host = nullptr, *res_host = nullptr;
    bf16* tg_host = nullptr;
    std::map<unsigned long long, cudaGraphExec_t> gx;   // pass graphs (Impl::df_pass)
};
struct DfSlot {
    int cnt = 0;                                   // context rows = the next position
    std::vector<std::pair<long long, int>> pend;   // queued rows: (tap ring sequence number, position)
};
struct DfHost {
    int cap = 0, head = 0;
    std::vector<DfSlot> slot;
    long dropped = 0;
    int slot_of(int pos) const { return pos < cap ? pos : head + (pos - head) % (cap - head); }
};

}  // namespace gptoss
