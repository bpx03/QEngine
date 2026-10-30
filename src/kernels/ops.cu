#include "../env.h"
#include <algorithm>
#include <cfloat>

#include <cuda_fp8.h>

#include "ops.cuh"
#include "watchdog.cuh"
#include "gemv_h16.cuh"

namespace {

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

// Sum across the block; all threads get the result. blockDim.x must be a multiple of 32.
__device__ float block_sum(float v) {
    __shared__ float sh[32];
    __shared__ float total;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    v = warp_sum(v);
    __syncthreads();
    if (lane == 0) sh[warp] = v;
    __syncthreads();
    if (warp == 0) {
        float x = lane < (blockDim.x >> 5) ? sh[lane] : 0.f;
        x = warp_sum(x);
        if (lane == 0) total = x;
    }
    __syncthreads();
    return total;
}

__device__ __forceinline__ float silu(float x) { return x / (1.f + __expf(-x)); }
__device__ __forceinline__ float sigmoid(float x) { return 1.f / (1.f + __expf(-x)); }

// ---------------------------------------------------------------------------------------

__device__ const bf16* g_img_table = nullptr;

__global__ void embed_kernel(const bf16* __restrict__ table, const int* __restrict__ tokens,
                             int hidden, float* __restrict__ resid) {
    const int mi = blockIdx.x, tok = tokens[mi];
    const bf16* src = tok >= kImgToken ? g_img_table + (size_t)(tok - kImgToken) * hidden
                                       : table + (size_t)tok * hidden;
    const uint4* row = reinterpret_cast<const uint4*>(src);
    float4* o = reinterpret_cast<float4*>(resid + (size_t)mi * hidden);
    for (int i = threadIdx.x; i < hidden / 8; i += blockDim.x) {  // 16-byte reads (host memory)
        const uint4 v = row[i];
        const bf16* e = reinterpret_cast<const bf16*>(&v);
        o[2 * i] = make_float4(__bfloat162float(e[0]), __bfloat162float(e[1]),
                               __bfloat162float(e[2]), __bfloat162float(e[3]));
        o[2 * i + 1] = make_float4(__bfloat162float(e[4]), __bfloat162float(e[5]),
                                   __bfloat162float(e[6]), __bfloat162float(e[7]));
    }
}

__global__ void rmsnorm_kernel(const float* __restrict__ x, const bf16* __restrict__ w, int n,
                               float eps, bf16* __restrict__ out, __half* __restrict__ xh) {
    const float* xr = x + (size_t)blockIdx.x * n;
    float ss = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) ss += xr[i] * xr[i];
    const float r = rsqrtf(block_sum(ss) / n + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float v = xr[i] * r * (1.f + __bfloat162float(w[i]));
        out[(size_t)blockIdx.x * n + i] = __float2bfloat16(v);
        if (xh) store_xh(xh + (size_t)blockIdx.x * n, i, v);
    }
}

// grid (m, nq + 2*nkv), block 256 (one thread per head dim)
// RoPE angle of cache index p for rotary frequency fi: p + rdelta (text after images) unless p
// falls in the multimodal rows of the prompt chunk (rpos3), interleaved M-RoPE sections 11/11/10:
// frequency fi uses the h position if fi % 3 == 1 (fi < 33), w if fi % 3 == 2 (fi < 30), else t
__device__ __forceinline__ float rope_angle(const SeqView& v, int p, int fi, const float* __restrict__ inv_freq) {
    int rp = p + (v.rdelta ? *v.rdelta : 0);
    if (v.rpos3) {
        const int r = p - v.rpos3[1];
        if (r >= 0 && r < v.rpos3[0]) {
            const int ax = fi % 3 == 1 ? 1 : (fi % 3 == 2 && fi < 30 ? 2 : 0);
            rp = v.rpos3[2 + 3 * r + ax];
        }
    }
    return (float)rp * inv_freq[fi];  // fp32 like HF, trig in double
}

__device__ __forceinline__ void rope_table_dev(const SeqView& v, const int* __restrict__ pos, int row0,
                                               const float* __restrict__ inv_freq, double2* __restrict__ tab) {
    const int mi = blockIdx.x, fi = threadIdx.x;
    const float ang = rope_angle(v, pos[0] + mi, fi, inv_freq);
    tab[(size_t)(row0 + mi) * 32 + fi] = make_double2(cos((double)ang), sin((double)ang));
}

__global__ void rope_table_kernel(const AttnShape s, const int* __restrict__ pos, const float* __restrict__ inv_freq,
                                  double2* __restrict__ tab) {
    rope_table_dev(seq_view(s), pos, s.xrow0, inv_freq, tab);
}

__global__ void rope_table_seg_kernel(const AttnSegs g, const float* __restrict__ inv_freq,
                                      double2* __restrict__ tab) {
    const AttnSeg& c = g.c[blockIdx.y];
    if ((int)blockIdx.x >= c.m) return;
    rope_table_dev(seq_view(c), c.pos, seg_row0(g, blockIdx.y), inv_freq, tab);
}

__device__ __forceinline__ void attn_prep_dev(const float* __restrict__ proj, int ld, const AttnShape& s,
                                              const SeqView& sv,
                                              const bf16* __restrict__ qn_w, const bf16* __restrict__ kn_w,
                                              float eps, const float* __restrict__ inv_freq,
                                              const int* __restrict__ pos, bf16* __restrict__ q_out,
                                              void* __restrict__ kc, void* __restrict__ vc, int row0) {
    __shared__ float xs[256];
    const int mi = blockIdx.x, h = blockIdx.y, d = threadIdx.x;
    const int p = pos[0] + mi;
    const float* row = proj + (size_t)mi * ld;
    const bool is_q = h < s.nq, is_k = !is_q && h < s.nq + s.nkv;
    if (!is_q && !is_k) {  // v: straight copy into the cache (kv-pos-split: owned positions only)
        const int j = h - s.nq - s.nkv;
        const PosMap pm = s.kv_pm[j];
        if (!pm_owned(pm, p)) return;
        const float v = row[s.nq * 512 + s.nkv * 256 + j * 256 + d];
        const size_t idx = (head_rows(s, j, sv.max_t) + (pm_rank(pm, p) - pm_rank(pm, sv.split))) * 256 + d;
        if (s.fp8)
            ((__nv_fp8_storage_t*)vc)[idx] = __nv_cvt_float_to_fp8(v / s.vs, __NV_SATFINITE, __NV_E4M3);
        else
            ((bf16*)vc)[idx] = __float2bfloat16(v);
        return;
    }
    const int j = is_q ? h : h - s.nq;
    if (!is_q && !pm_owned(s.kv_pm[j], p)) return;  // this key lives on the peer
    const float x = is_q ? row[j * 512 + d] : row[s.nq * 512 + j * 256 + d];
    const float r = rsqrtf(block_sum(x * x) / 256.f + eps);
    const float xn = x * r * (1.f + __bfloat162float(is_q ? qn_w[d] : kn_w[d]));
    xs[d] = xn;
    __syncthreads();
    float y = xn;
    if (d < 64) {
        // rotate_half over the 64 rotary dims; angle in fp32 like HF, trig in double (from the
        // step's table when given: the same values, computed once per row instead of per head)
        const int fi = d & 31;
        double c, sn;
        if (s.rtab) {
            const double2 t = s.rtab[(size_t)(row0 + mi) * 32 + fi];
            c = t.x;
            sn = t.y;
        } else {
            const float ang = rope_angle(sv, p, fi, inv_freq);
            c = cos((double)ang);
            sn = sin((double)ang);
        }
        y = d < 32 ? (float)(xn * c - xs[d + 32] * sn) : (float)(xn * c + xs[d - 32] * sn);
    }
    if (is_q) {
        q_out[((size_t)mi * s.nq + j) * 256 + d] = __float2bfloat16(y);
        return;
    }
    const PosMap pm = s.kv_pm[j];
    const size_t idx = (head_rows(s, j, sv.max_t) + (pm_rank(pm, p) - pm_rank(pm, sv.split))) * 256 + d;
    if (s.fp8)
        ((__nv_fp8_storage_t*)kc)[idx] = __nv_cvt_float_to_fp8(y / s.ks, __NV_SATFINITE, __NV_E4M3);
    else
        ((bf16*)kc)[idx] = __float2bfloat16(y);
}

__global__ void attn_prep_kernel(const float* __restrict__ proj, int ld, const AttnShape s,
                                 const bf16* __restrict__ qn_w, const bf16* __restrict__ kn_w,
                                 float eps, const float* __restrict__ inv_freq,
                                 const int* __restrict__ pos, bf16* __restrict__ q_out,
                                 void* __restrict__ kc, void* __restrict__ vc) {
    attn_prep_dev(proj, ld, s, seq_view(s), qn_w, kn_w, eps, inv_freq, pos, q_out, kc, vc, s.xrow0);
}

__global__ void attn_prep_seg_kernel(const AttnSegs g, int ld, const AttnShape s, const bf16* __restrict__ qn_w,
                                     const bf16* __restrict__ kn_w, float eps,
                                     const float* __restrict__ inv_freq) {
    const AttnSeg& c = g.c[blockIdx.z];
    if ((int)blockIdx.x >= c.m) return;
    attn_prep_dev(c.proj, ld, s, seq_view(c), qn_w, kn_w, eps, inv_freq, c.pos, c.q, c.kc, c.vc, seg_row0(g, blockIdx.z));
}

// Flash-decoding split. grid (nkv, kAttnSplits), block 256.
// Group g = local_head_in_kv_group * m + mi.
constexpr int kTile = 64;
constexpr int kKPad = 264;  // padded K row (bf16) in shared memory

// MAXG: compile-time bound on G so the per-thread accumulators and unrolled loops only cover
// the query rows that exist (G = 6 for single-token decode with 6 q heads per kv head).
template <int MAXG>
__global__ void __launch_bounds__(256) attn_split_kernel(
    const bf16* __restrict__ q, const bf16* __restrict__ kc, const bf16* __restrict__ vc,
    AttnShape s, const int* __restrict__ pos, int m, float* __restrict__ work) {
    extern __shared__ __align__(16) unsigned char smem[];
    bf16* qs = reinterpret_cast<bf16*>(smem);                  // [G][256]
    bf16* ks = qs + kMaxGroup * 256;                           // [kTile][kKPad]
    float* sc = reinterpret_cast<float*>(ks + kTile * kKPad);  // [G][kTile]
    float* mrow = sc + kMaxGroup * kTile;                      // running max [G]
    float* lrow = mrow + kMaxGroup;                            // running sum [G]
    float* alpha = lrow + kMaxGroup;                           // rescale [G]

    const int kv = blockIdx.x, split = blockIdx.y, tid = threadIdx.x;
    const int qb = s.kv_qb[kv], nh = s.kv_qe[kv] - qb;
    const int G = nh * m;
    const int p0 = pos[0];
    const int L = p0 + m;
    int chunk = (p0 + kMaxGemvTokens + kAttnSplits - 1) / kAttnSplits;  // independent of m
    chunk = (chunk + kTile - 1) / kTile * kTile;
    const int s0 = split * chunk, s1 = min(L, s0 + chunk);

    float* wacc = work + ((size_t)(kv * kAttnSplits + split) * kMaxGroup) * 256;
    float* wml = work + (size_t)s.nkv * kAttnSplits * kMaxGroup * 256 +
                 ((size_t)(kv * kAttnSplits + split) * kMaxGroup) * 2;
    if (s0 >= s1) {
        if (tid < G) {
            wml[tid * 2] = -FLT_MAX;
            wml[tid * 2 + 1] = 0.f;
        }
        return;
    }

    for (int i = tid; i < G * 32; i += 256) {  // 16-byte chunks of q rows
        const int g = i / 32, c = i % 32;
        const int hl = g / m, mi = g % m;
        reinterpret_cast<uint4*>(qs + g * 256)[c] =
            reinterpret_cast<const uint4*>(q + ((size_t)mi * s.nq + qb + hl) * 256)[c];
    }
    if (tid < G) {
        mrow[tid] = -FLT_MAX;
        lrow[tid] = 0.f;
    }
    float acc[MAXG];
#pragma unroll
    for (int g = 0; g < MAXG; g++) acc[g] = 0.f;

    const bf16* kbase = kc + (size_t)kv * s.max_t * 256;
    const bf16* vbase = vc + (size_t)kv * s.max_t * 256;
    const float scale = 0.0625f;  // 1/sqrt(256)

    for (int t0 = s0; t0 < s1; t0 += kTile) {
        const int nt = min(kTile, s1 - t0);
        __syncthreads();
        for (int i = tid; i < nt * 32; i += 256) {
            const int j = i / 32, c = i % 32;
            *reinterpret_cast<uint4*>(ks + j * kKPad + c * 8) =
                reinterpret_cast<const uint4*>(kbase + (size_t)(t0 + j) * 256)[c];
        }
        __syncthreads();
        for (int i = tid; i < G * kTile; i += 256) {
            const int g = i / kTile, j = i % kTile;
            float dot = -FLT_MAX;
            const int key = t0 + j, mi = g % m;
            if (j < nt && key <= p0 + mi) {
                const bf162* qr = reinterpret_cast<const bf162*>(qs + g * 256);
                const bf162* kr = reinterpret_cast<const bf162*>(ks + j * kKPad);
                float a = 0.f;
#pragma unroll 16
                for (int d = 0; d < 128; d++) {
                    const float2 qq = __bfloat1622float2(qr[d]);
                    const float2 kk = __bfloat1622float2(kr[d]);
                    a += qq.x * kk.x + qq.y * kk.y;
                }
                dot = a * scale;
            }
            sc[g * kTile + j] = dot;
        }
        __syncthreads();
        if (tid < G) {
            float mx = mrow[tid];
            for (int j = 0; j < nt; j++) mx = fmaxf(mx, sc[tid * kTile + j]);
            const float a = __expf(mrow[tid] - mx);
            float l = lrow[tid] * a;
            for (int j = 0; j < nt; j++) {
                const float v = sc[tid * kTile + j];
                const float pr = v == -FLT_MAX ? 0.f : __expf(v - mx);
                sc[tid * kTile + j] = pr;
                l += pr;
            }
            mrow[tid] = mx;
            lrow[tid] = l;
            alpha[tid] = a;
        }
        __syncthreads();
#pragma unroll
        for (int g = 0; g < MAXG; g++)
            if (g < G) acc[g] *= alpha[g];
        // issue 16 independent V loads at a time: one block per SM can't hide latency otherwise
        for (int j0 = 0; j0 < nt; j0 += 16) {
            float v[16];
#pragma unroll
            for (int u = 0; u < 16; u++)
                v[u] = j0 + u < nt ? __bfloat162float(vbase[(size_t)(t0 + j0 + u) * 256 + tid]) : 0.f;
#pragma unroll
            for (int u = 0; u < 16; u++) {
                if (j0 + u >= nt) break;
#pragma unroll
                for (int g = 0; g < MAXG; g++)
                    if (g < G) acc[g] += sc[g * kTile + j0 + u] * v[u];
            }
        }
    }
#pragma unroll
    for (int g = 0; g < MAXG; g++)
        if (g < G) wacc[(size_t)g * 256 + tid] = acc[g];
    if (tid < G) {
        wml[tid * 2] = mrow[tid];
        wml[tid * 2 + 1] = lrow[tid];
    }
}

// splits of the fp8 decode kernel that hold keys of kv head kv (local key space, kv-pos-split)
__device__ __forceinline__ int dec8_nsplits(const AttnShape& s, int kv, int p0, int m) {
    const PosMap pm = s.kv_pm[kv];
    const int k0wa = dec8_first(p0, s.win), k0w = pm_rank(pm, k0wa);
    const int p0l = pm_rank(pm, p0), L = pm_rank(pm, p0 + m);
    const int c = dec8_chunk_pm(s, kv, p0l - k0w, p0 - k0wa);
    return min(kAttnSplits, (L - k0w + c - 1) / c);
}

// Merge of the local splits of (row g, kv head kv) in the fixed split order: weights w_sp =
// exp(m_sp - mx) into wsp[], the running max and sum(w * l); returns the merged numerator for dim d.
// (shared by the combine and the exchange sender so both sides scale partials the same way)
__device__ __forceinline__ float merge_splits(const float* __restrict__ work, int nkv, int kv, int ns, int g,
                                              int d, float* wsp, float& mx_out, float& den_out) {
    const float* ml = work + (size_t)nkv * kAttnSplits * kMaxGroup * 256;
    __shared__ float s_mx, s_den;
    if (d < 32) {
        float mx = -FLT_MAX;
        for (int sp = d; sp < ns; sp += 32)
            mx = fmaxf(mx, ml[((size_t)(kv * kAttnSplits + sp) * kMaxGroup + g) * 2]);
        for (int o = 16; o; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
        float den = 0.f;
        for (int sp = d; sp < ns; sp += 32) {
            const size_t base = (size_t)(kv * kAttnSplits + sp) * kMaxGroup + g;
            const float l = ml[base * 2 + 1];
            const float w = l == 0.f ? 0.f : __expf(ml[base * 2] - mx);
            wsp[sp] = w;
            den += w * l;
        }
        for (int o = 16; o; o >>= 1) den += __shfl_xor_sync(0xffffffffu, den, o);
        if (d == 0) { s_mx = mx; s_den = den; }
    }
    __syncthreads();
    float num = 0.f;
#pragma unroll 8
    for (int sp = 0; sp < ns; sp++) {
        const float w = wsp[sp];
        if (w != 0.f) num += w * work[((size_t)(kv * kAttnSplits + sp) * kMaxGroup + g) * 256 + d];
    }
    mx_out = s_mx;
    den_out = s_den;
    return num;
}

// grid (m, nq), block 256: merge splits, apply sigmoid(gate), write bf16. Only the o_proj heads
// [ob, ob + no); a head whose keys are split with a peer (kv_x == 2) merges the peer's partial
// (received after its own splits, in a fixed order) when the peer holds keys in the window.
// The blocks of the q heads of a kv head whose o_proj lives on the peer (kv_x == 1) send instead:
// the local splits merged into one unnormalized partial (o * vs, m, l) per (row, q head), written
// over P2P into the peer's receive rows; the last of these blocks (arrival counter over all send
// blocks of the launch, also those beyond a sequence's rows) raises the peer's flag with the
// exchange sequence after a system fence.
// Stats exchange, grid (ceil(max rows * nx / 8), 1, sequences), 8 warps per block: warp w of block b
// handles record b * 8 + w = (row, exchange q head) of a position-split kv head (kv_x == 3) with keys
// of this card in the window: the lanes merge (max, sum) over the local splits (shuffles, the same
// numbers as merge_splits in the combine) and lane 0 sends the float2 to the peer's receive rows
// over P2P. The last block of the launch (arrival counter over all blocks, also those without
// records) raises the peer's flag with the exchange sequence after a system fence - on every
// exchange, also when no sequence of the launch has keys here (then no data: the peer waits for
// the flag but does not read the record, xchg_needed on the same inputs).
__device__ __forceinline__ void attn_stats_dev(const float* __restrict__ work, const AttnShape& s,
                                               const int* __restrict__ pos, int m, int row0, bool dec8) {
    __shared__ int xh[32];
    __shared__ int s_nx;
    if (threadIdx.x == 0) s_nx = xchg_heads(s, xh);
    __syncthreads();
    const int nx = s_nx, lane = threadIdx.x & 31;
    const int r = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (r < m * nx) {
        const int mi = r / nx, h = xh[r % nx], kv = s.q2kv[h];
        if (xchg_has(s, kv, pos[0], m, s.win)) {
            const int g = (h - s.kv_qb[kv]) * m + mi;
            const int ns = dec8 ? dec8_nsplits(s, kv, pos[0], m) : kAttnSplits;
            const float* ml = work + (size_t)s.nkv * kAttnSplits * kMaxGroup * 256;
            float mx = -FLT_MAX;
            for (int sp = lane; sp < ns; sp += 32)
                mx = fmaxf(mx, ml[((size_t)(kv * kAttnSplits + sp) * kMaxGroup + g) * 2]);
            for (int o = 16; o; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
            float den = 0.f;
            for (int sp = lane; sp < ns; sp += 32) {
                const size_t base = (size_t)(kv * kAttnSplits + sp) * kMaxGroup + g;
                const float l = ml[base * 2 + 1];
                den += (l == 0.f ? 0.f : __expf(ml[base * 2] - mx)) * l;
            }
            for (int o = 16; o; o >>= 1) den += __shfl_xor_sync(0xffffffffu, den, o);
            if (lane == 0) {
                float* dst = s.xpo[kv] + ((size_t)(row0 + mi) * (s.kv_qe[kv] - s.kv_qb[kv]) + (h - s.kv_qb[kv])) * kXchgFloats;
                *reinterpret_cast<float2*>(dst) = make_float2(mx, den);
            }
        }
    }
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0 && atomicAdd(s.xdone, 1u) == gridDim.x * gridDim.y * gridDim.z - 1) {
        *s.xdone = 0;
        __threadfence();
        for (int j = 0; j < s.nkv; j++)
            if (s.kv_x[j] == 3) qe_st_release_sys(s.xpflag[j], xchg_seq(s));
    }
}

// blocks of 8 warps covering rows * (exchange q heads) records
static int stats_blocks(const AttnShape& s, int rows) {
    int nx = 0;
    for (int j = 0; j < s.nkv; j++)
        if (s.kv_x[j] == 3) nx += s.kv_qe[j] - s.kv_qb[j];
    return (rows * nx + 7) / 8;
}

__global__ void attn_stats_kernel(const float* __restrict__ work, const AttnShape s, const int* __restrict__ pos,
                                  int m, bool dec8) {
    attn_stats_dev(work, s, pos, m, s.xrow0, dec8);
}

__global__ void attn_stats_seg_kernel(const AttnSegs g, const AttnShape s, bool dec8) {
    const AttnSeg& c = g.c[blockIdx.z];
    attn_stats_dev(c.work, s, c.pos, c.m, seg_row0(g, blockIdx.z), dec8);
}

// Combine, grid (m, nq), block 256: merges the local splits of (row, q head) and writes the gated
// bf16 output. A position-split head waits for the peer's stats (when the peer holds keys in the
// window) and scales its numerator by e^(m - M) / D with the global max M and sum D, so the two
// cards' outputs add up to the attention through the linear o_proj and the all-reduce.
__device__ __forceinline__ void attn_combine_dev(const float* __restrict__ work, const AttnShape& s,
                                                 const float* __restrict__ proj, int ld, int m,
                                                 bf16* __restrict__ out, const int* __restrict__ pos,
                                                 bool dec8, __half* __restrict__ xh, int row0) {
    const int mi = blockIdx.x, h = blockIdx.y, d = threadIdx.x;
    const int no = s.no ? s.no : s.nq;
    const int kv = s.q2kv[h];
    __shared__ float wsp[kAttnSplits];
    if (mi >= m || h < s.ob || h >= s.ob + no) return;
    const int g = (h - s.kv_qb[kv]) * m + mi;
    // fp8 decode kernel: only the splits that hold keys
    const int ns = dec8 ? dec8_nsplits(s, kv, pos[0], m) : kAttnSplits;
    // a position-split head waits for the peer's flag on every exchange; the peer's record is
    // merged only when the peer holds keys of the head in the window (else it wrote nothing)
    const bool xch = s.kv_x[kv] == 3;
    const bool recv = xch && xchg_needed(s, kv, pos[0], m, s.win);
    __shared__ float s_rm, s_rl;
    if (xch) {  // wait for the peer's flag (raised after all rows)
        if (d == 0) {
            unsigned ws = 0;
            while ((int)(qe_ld_acquire_sys(s.xflag[kv]) - xchg_seq(s)) < 0) {
                __nanosleep(64);
                // a stuck wait (peer gone / flag-sequence mismatch) is reported by the watchdog, not silent
                QE_WD_POLL(ws, kWdKvDecode, -1, xchg_seq(s), qe_ld_acquire_sys(s.xflag[kv]));
            }
        }
        __syncthreads();
    }
    if (recv) {  // the peer's stats of this (row, head)
        const int gh = s.kv_qe[kv] - s.kv_qb[kv], hl = h - s.kv_qb[kv];
        if (d == 0) {
            const float* rem = s.xo[kv] + ((size_t)(row0 + mi) * gh + hl) * kXchgFloats;
            const float2 r = __ldcg(reinterpret_cast<const float2*>(rem));
            s_rm = r.x;
            s_rl = r.y;
        }
        __syncthreads();
    }
    // split weights once per block (they do not depend on d): w = exp(m_sp - max) / sum(w * l)
    float mx, den;
    float num = merge_splits(work, s.nkv, kv, ns, g, d, wsp, mx, den);
    float v = 0.f;
    if (recv) {  // global max / sum with the peer: this card's share of the attention
        const float rm = s_rm, rl = s_rl;
        const float M = fmaxf(mx, rm);
        const float wa = mx == -FLT_MAX ? 0.f : __expf(mx - M), wb = rm == -FLT_MAX ? 0.f : __expf(rm - M);
        const float D = den * wa + rl * wb;
        v = den == 0.f ? 0.f : num * wa / D;
    } else {
        v = num * (1.f / den);  // (num * inv_den, as before the split)
    }
    const float gate = proj[(size_t)mi * ld + h * 512 + 256 + d];
    v *= sigmoid(gate);
    out[((size_t)mi * no + (h - s.ob)) * 256 + d] = __float2bfloat16(v);
    if (xh) store_xh(xh + (size_t)mi * no * 256, (h - s.ob) * 256 + d, v);
}

__global__ void attn_combine_kernel(const float* __restrict__ work, const AttnShape s,
                                    const float* __restrict__ proj, int ld, int m,
                                    bf16* __restrict__ out, const int* __restrict__ pos, bool dec8,
                                    __half* __restrict__ xh) {
    attn_combine_dev(work, s, proj, ld, m, out, pos, dec8, xh, s.xrow0);
}

__global__ void attn_combine_seg_kernel(const AttnSegs g, const AttnShape s, int ld, bool dec8) {
    const AttnSeg& c = g.c[blockIdx.z];
    if ((int)blockIdx.x >= c.m) return;
    attn_combine_dev(c.work, s, c.proj, ld, c.m, c.out, c.pos, dec8, c.xh, seg_row0(g, blockIdx.z));
}

static bool xchg_on(const AttnShape& s) {
    for (int j = 0; j < s.nkv; j++)
        if (s.kv_x[j] == 3) return true;
    return false;
}


__device__ __forceinline__ void dn_conv_dev(const float* __restrict__ proj, int ld, int ch,
                                            const bf16* __restrict__ w, bf16* __restrict__ hist,
                                            const int* __restrict__ pos, int m,
                                            float* __restrict__ out) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= ch) return;
    const float w0 = __bfloat162float(w[c * 4 + 0]), w1 = __bfloat162float(w[c * 4 + 1]);
    const float w2 = __bfloat162float(w[c * 4 + 2]), w3 = __bfloat162float(w[c * 4 + 3]);
    const int p0 = pos[0];
    for (int mi = 0; mi < m; mi++) {
        const int p = p0 + mi;
        const bf16 x = __float2bfloat16(proj[(size_t)mi * ld + c]);
        hist[(size_t)(p % kConvRing) * ch + c] = x;
        auto h = [&](int q) {
            return q < 0 ? 0.f : __bfloat162float(hist[(size_t)(q % kConvRing) * ch + c]);
        };
        const float y = w0 * h(p - 3) + w1 * h(p - 2) + w2 * h(p - 1) + w3 * __bfloat162float(x);
        out[(size_t)mi * ch + c] = silu(y);
    }
}

__global__ void dn_conv_kernel(const float* __restrict__ proj, int ld, int ch,
                               const bf16* __restrict__ w, bf16* __restrict__ hist,
                               const int* __restrict__ pos, int m, float* __restrict__ out) {
    dn_conv_dev(proj, ld, ch, w, hist, pos, m, out);
}

// Decode (m <= 13 rows per sequence): conv + q/k normalization + decay/beta in one kernel,
// one row per blockIdx.y. Blocks x = 0..nk-1: q and k channels of k head x (threads 0-127 /
// 128-255), normalized; then the v channels (256 per block); the last block: decay and beta.
// Inputs before the step come from the conv history ring, the step's own from proj (the ring
// slots written by this step do not overlap the 3 read before it while m <= 13).
__device__ __forceinline__ void dn_convqk_dev(const float* __restrict__ proj, int ld, int ch,
                                              const bf16* __restrict__ w, bf16* __restrict__ hist,
                                              const int* __restrict__ pos, int mi, int r0,
                                              float* __restrict__ cv, const DeltaShape& s,
                                              const float* __restrict__ ab,
                                              const float* __restrict__ A,
                                              const float* __restrict__ dt_bias,
                                              float* __restrict__ qk, float* __restrict__ gb) {
    __shared__ float red[8];
    const int b = blockIdx.x, t = threadIdx.x;
    const int nvb = (s.nv * 128 + 255) / 256;
    const int row = r0 + mi;
    if (b == s.nk + nvb) {  // decay / beta
        const int ldab = 2 * s.ab_pad;
        for (int h = t; h < s.nv; h += blockDim.x) {
            float bb = 0.f, aa = dt_bias[h];
            for (int sp = 0; sp < s.ab_splits; sp++) {
                const float* abs = ab + (size_t)sp * s.ab_rows * ldab + (size_t)row * ldab;
                bb += abs[h];
                aa += abs[s.ab_pad + h];
            }
            const float sp_a = aa > 20.f ? aa : log1pf(__expf(aa));
            gb[(size_t)row * 2 * s.nv + h] = __expf(A[h] * sp_a);
            gb[(size_t)row * 2 * s.nv + s.nv + h] = sigmoid(bb);
        }
        return;
    }
    const bool qkb = b < s.nk;
    int c;
    if (qkb) c = (t < 128 ? 0 : s.nk * 128) + b * 128 + (t & 127);
    else {
        c = 2 * s.nk * 128 + (b - s.nk) * 256 + t;
        if (c >= ch) return;
    }
    const float w0 = __bfloat162float(w[c * 4 + 0]), w1 = __bfloat162float(w[c * 4 + 1]);
    const float w2 = __bfloat162float(w[c * 4 + 2]), w3 = __bfloat162float(w[c * 4 + 3]);
    const int p0 = pos[0], p = p0 + mi;
    auto h = [&](int q) {
        if (q < 0) return 0.f;
        if (q >= p0) return __bfloat162float(__float2bfloat16(proj[(size_t)(r0 + q - p0) * ld + c]));
        return __bfloat162float(hist[(size_t)(q % kConvRing) * ch + c]);
    };
    const float y = silu(w0 * h(p - 3) + w1 * h(p - 2) + w2 * h(p - 1) + w3 * h(p));
    cv[(size_t)row * ch + c] = y;
    if (qkb) {  // L2-normalize q and k of this head (fixed reduction order)
        float x2 = y * y;
#pragma unroll
        for (int o = 16; o; o >>= 1) x2 += __shfl_xor_sync(0xffffffffu, x2, o);
        if ((t & 31) == 0) red[t >> 5] = x2;
        __syncthreads();
        const int hb = (t >> 7) * 4;
        const float n2 = red[hb] + red[hb + 1] + red[hb + 2] + red[hb + 3];
        float* o = qk + (size_t)row * 2 * s.nk * 128;
        if (t < 128) o[b * 128 + t] = y * rsqrtf(n2 + 1e-6f) * 0.08838834764831845f;
        else o[s.nk * 128 + b * 128 + (t - 128)] = y * rsqrtf(n2 + 1e-6f);
        __syncthreads();  // all reads of the ring (other blocks: none of these slots) done
    }
    hist[(size_t)(p % kConvRing) * ch + c] = __float2bfloat16(proj[(size_t)row * ld + c]);
}

__global__ void __launch_bounds__(256) dn_convqk_kernel(
    const float* __restrict__ proj, int ld, int ch, const bf16* __restrict__ w,
    bf16* __restrict__ hist, const int* __restrict__ pos, float* __restrict__ cv,
    DeltaShape s, const float* __restrict__ ab, const float* __restrict__ A,
    const float* __restrict__ dt_bias, float* __restrict__ qk, float* __restrict__ gb) {
    dn_convqk_dev(proj, ld, ch, w, hist, pos, blockIdx.y, 0, cv, s, ab, A, dt_bias, qk, gb);
}

__global__ void __launch_bounds__(256) dn_convqk_seg_kernel(
    DnSegs g, const float* __restrict__ proj, int ld, int ch, const bf16* __restrict__ w,
    float* __restrict__ cv, DeltaShape s, const float* __restrict__ ab,
    const float* __restrict__ A, const float* __restrict__ dt_bias, float* __restrict__ qk,
    float* __restrict__ gb) {
    const int z = blockIdx.z;
    if ((int)blockIdx.y >= g.m[z]) return;
    dn_convqk_dev(proj, ld, ch, w, g.hist[z], g.pos[z], blockIdx.y, g.r0[z], cv, s, ab, A, dt_bias, qk, gb);
}

// several sequences (batched decode): segment blockIdx.y
__global__ void dn_conv_seg_kernel(DnSegs g, const float* __restrict__ proj, int ld, int ch,
                                   const bf16* __restrict__ w, float* __restrict__ out) {
    const int z = blockIdx.y;
    dn_conv_dev(proj + (size_t)g.r0[z] * ld, ld, ch, w, g.hist[z], g.pos[z], g.m[z],
                out + (size_t)g.r0[z] * ch);
}

// Prefill version: one thread per (token, channel); inputs before the chunk come from the
// history ring, which dn_conv_hist_kernel then updates (same arithmetic as dn_conv_kernel).
__global__ void dn_conv_par_kernel(const float* __restrict__ proj, int ld, int ch,
                                   const bf16* __restrict__ w, const bf16* __restrict__ hist,
                                   const int* __restrict__ pos, int m, float* __restrict__ out) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x, mi = blockIdx.y;
    if (c >= ch) return;
    const int p0 = pos[0], p = p0 + mi;
    auto h = [&](int q) {
        if (q < 0) return 0.f;
        if (q >= p0) return __bfloat162float(__float2bfloat16(proj[(size_t)(q - p0) * ld + c]));
        return __bfloat162float(hist[(size_t)(q % kConvRing) * ch + c]);
    };
    const float w0 = __bfloat162float(w[c * 4 + 0]), w1 = __bfloat162float(w[c * 4 + 1]);
    const float w2 = __bfloat162float(w[c * 4 + 2]), w3 = __bfloat162float(w[c * 4 + 3]);
    const float y = w0 * h(p - 3) + w1 * h(p - 2) + w2 * h(p - 1) + w3 * h(p);
    out[(size_t)mi * ch + c] = silu(y);
}

// Prefill front end fused (chunked path): conv + SiLU + q/k normalization + decay/beta in one
// launch, grid (m, 2 nk + nv + 1), block 128: y < nk q head, < 2 nk k head (normalized straight
// into qk, the conv output of q / k is not stored), < 2 nk + nv v head (conv output -> cv, read
// by the chunk prep), last: decay / beta. Same arithmetic and reduction order as dn_conv_par_kernel
// followed by dn_qk_kernel (bit-identical qk, gb and v part of cv).
__global__ void __launch_bounds__(128) dn_convqk_par_kernel(
    const float* __restrict__ proj, int ld, int ch, const bf16* __restrict__ w,
    const bf16* __restrict__ hist, const int* __restrict__ pos, float* __restrict__ cv, DeltaShape s,
    const float* __restrict__ ab, const float* __restrict__ A, const float* __restrict__ dt_bias,
    float* __restrict__ qk, float* __restrict__ gb) {
    const int mi = blockIdx.x, j = blockIdx.y, t = threadIdx.x;
    if (j == 2 * s.nk + s.nv) {
        const int ldab = 2 * s.ab_pad;
        for (int h = t; h < s.nv; h += blockDim.x) {
            float bb = 0.f, aa = dt_bias[h];
            for (int sp = 0; sp < s.ab_splits; sp++) {
                const float* abs = ab + (size_t)sp * s.ab_rows * ldab + (size_t)mi * ldab;
                bb += abs[h];
                aa += abs[s.ab_pad + h];
            }
            const float sp_a = aa > 20.f ? aa : log1pf(__expf(aa));
            gb[(size_t)mi * 2 * s.nv + h] = __expf(A[h] * sp_a);
            gb[(size_t)mi * 2 * s.nv + s.nv + h] = sigmoid(bb);
        }
        return;
    }
    const int c = j * 128 + t;  // q heads, then k heads, then v heads: the channel order of cv
    const int p0 = pos[0], p = p0 + mi;
    auto h = [&](int q) {
        if (q < 0) return 0.f;
        if (q >= p0) return __bfloat162float(__float2bfloat16(proj[(size_t)(q - p0) * ld + c]));
        return __bfloat162float(hist[(size_t)(q % kConvRing) * ch + c]);
    };
    const float w0 = __bfloat162float(w[c * 4 + 0]), w1 = __bfloat162float(w[c * 4 + 1]);
    const float w2 = __bfloat162float(w[c * 4 + 2]), w3 = __bfloat162float(w[c * 4 + 3]);
    const float y = silu(w0 * h(p - 3) + w1 * h(p - 2) + w2 * h(p - 1) + w3 * h(p));
    if (j >= 2 * s.nk) {
        cv[(size_t)mi * ch + c] = y;
        return;
    }
    const float n2 = block_sum(y * y);
    float* o = qk + (size_t)mi * 2 * s.nk * 128;
    if (j < s.nk) o[j * 128 + t] = y * rsqrtf(n2 + 1e-6f) * 0.08838834764831845f;  // / sqrt(128)
    else o[j * 128 + t] = y * rsqrtf(n2 + 1e-6f);
}

// Same outputs as dn_convqk_par_kernel, but a block walks TW consecutive tokens of one head with
// the 4-tap window in registers: every proj value is read once (dn_convqk_par_kernel read each
// row 4 times, for the token and its 3 successors). grid (ceil(m / TW), 2 nk + nv + 1), block 128.
constexpr int kConvTW = 32;
template <bool H16 = false>  // fp16 q / k / v outputs (dn_chunk_prep_kernel<true>)
__global__ void __launch_bounds__(128) dn_convqk_win_kernel(
    const float* __restrict__ proj, int ld, int ch, const bf16* __restrict__ w,
    const bf16* __restrict__ hist, const int* __restrict__ pos, int m, float* __restrict__ cv, DeltaShape s,
    const float* __restrict__ ab, const float* __restrict__ A, const float* __restrict__ dt_bias,
    float* __restrict__ qk, float* __restrict__ gb) {
    const int j = blockIdx.y, t = threadIdx.x;
    const int i0 = blockIdx.x * kConvTW, i1 = min(m, i0 + kConvTW);
    if (j == 2 * s.nk + s.nv) {
        const int ldab = 2 * s.ab_pad;
        for (int e = t; e < (i1 - i0) * s.nv; e += blockDim.x) {
            const int mi = i0 + e / s.nv, h = e % s.nv;
            float bb = 0.f, aa = dt_bias[h];
            for (int sp = 0; sp < s.ab_splits; sp++) {
                const float* abs = ab + (size_t)sp * s.ab_rows * ldab + (size_t)mi * ldab;
                bb += abs[h];
                aa += abs[s.ab_pad + h];
            }
            const float sp_a = aa > 20.f ? aa : log1pf(__expf(aa));
            gb[(size_t)mi * 2 * s.nv + h] = __expf(A[h] * sp_a);
            gb[(size_t)mi * 2 * s.nv + s.nv + h] = sigmoid(bb);
        }
        return;
    }
    const int c = j * 128 + t;
    const int p0 = pos[0];
    auto h = [&](int q) {
        if (q < 0) return 0.f;
        if (q >= p0) return __bfloat162float(__float2bfloat16(proj[(size_t)(q - p0) * ld + c]));
        return __bfloat162float(hist[(size_t)(q % kConvRing) * ch + c]);
    };
    const float w0 = __bfloat162float(w[c * 4 + 0]), w1 = __bfloat162float(w[c * 4 + 1]);
    const float w2 = __bfloat162float(w[c * 4 + 2]), w3 = __bfloat162float(w[c * 4 + 3]);
    float x3 = h(p0 + i0 - 3), x2 = h(p0 + i0 - 2), x1 = h(p0 + i0 - 1);
    float* o = qk + (j < s.nk ? 0 : 0);
    for (int mi = i0; mi < i1; mi++) {
        const float x0 = h(p0 + mi);
        const float y = silu(w0 * x3 + w1 * x2 + w2 * x1 + w3 * x0);
        x3 = x2; x2 = x1; x1 = x0;
        if (j >= 2 * s.nk) {
            if constexpr (H16) reinterpret_cast<__half*>(cv)[(size_t)mi * ch + c] = __float2half_rn(y);
            else cv[(size_t)mi * ch + c] = y;
            continue;
        }
        const float n2 = block_sum(y * y);
        const float yn = j < s.nk ? y * rsqrtf(n2 + 1e-6f) * 0.08838834764831845f : y * rsqrtf(n2 + 1e-6f);
        if constexpr (H16) {
            reinterpret_cast<__half*>(qk)[(size_t)mi * 2 * s.nk * 128 + j * 128 + t] = __float2half_rn(yn);
        } else {
            o = qk + (size_t)mi * 2 * s.nk * 128;
            o[j * 128 + t] = yn;
        }
    }
}

__global__ void dn_conv_hist_kernel(const float* __restrict__ proj, int ld, int ch,
                                    bf16* __restrict__ hist, const int* __restrict__ pos, int m) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= ch) return;
    const int p0 = pos[0];
    for (int mi = max(0, m - kConvRing); mi < m; mi++)
        hist[(size_t)((p0 + mi) % kConvRing) * ch + c] = __float2bfloat16(proj[(size_t)mi * ld + c]);
}

// Per token: L2-normalized q (pre-scaled by 1/sqrt(128)) and k for each local key head, and
// per value head decay = exp(A * softplus(a + dt_bias)), beta = sigmoid(b). Fully parallel,
// so the sequential recurrence below has no reductions besides its own dot products.
// grid (m, nk + 1), block 128. qk: [m][2][nk][128], gb: [m][2][nv]
__global__ void dn_qk_kernel(const float* __restrict__ cv, int ch, DeltaShape s,
                             const float* __restrict__ ab, const float* __restrict__ A,
                             const float* __restrict__ dt_bias, float* __restrict__ qk,
                             float* __restrict__ gb) {
    const int mi = blockIdx.x, kh = blockIdx.y, t = threadIdx.x;
    if (kh == s.nk) {
        const int ldab = 2 * s.ab_pad;
        for (int h = t; h < s.nv; h += blockDim.x) {
            float bb = 0.f, aa = dt_bias[h];
            for (int sp = 0; sp < s.ab_splits; sp++) {
                const float* abs = ab + (size_t)sp * s.ab_rows * ldab + (size_t)mi * ldab;
                bb += abs[h];
                aa += abs[s.ab_pad + h];
            }
            const float sp_a = aa > 20.f ? aa : log1pf(__expf(aa));
            gb[(size_t)mi * 2 * s.nv + h] = __expf(A[h] * sp_a);
            gb[(size_t)mi * 2 * s.nv + s.nv + h] = sigmoid(bb);
        }
        return;
    }
    const float* row = cv + (size_t)mi * ch;
    const float qv = row[kh * 128 + t];
    const float kv = row[s.nk * 128 + kh * 128 + t];
    const float qn = block_sum(qv * qv), kn = block_sum(kv * kv);
    float* o = qk + (size_t)mi * 2 * s.nk * 128;
    o[kh * 128 + t] = qv * rsqrtf(qn + 1e-6f) * 0.08838834764831845f;  // / sqrt(128)
    o[s.nk * 128 + kh * 128 + t] = kv * rsqrtf(kn + 1e-6f);
}

// Gated delta rule recurrence. grid (nv, 8): each block owns 16 value columns of one head.
// Thread (vi = t/8, kq = t%8) keeps S[v][kq*16 .. +16) in registers for all m tokens; the 8
// threads of one column are adjacent lanes, so the k-dot-products reduce with shuffles and
// the token loop has no block barriers. Next token's inputs are prefetched.
__device__ __forceinline__ void dn_core_dev(const float* __restrict__ cv, int ch, const DeltaShape& s,
                                            const float* __restrict__ qk,
                                            const float* __restrict__ gb, float* __restrict__ S,
                                            size_t slot_floats, const int* __restrict__ curp, int m,
                                            bool write_all, float* __restrict__ o, int slot_off,
                                            const int* __restrict__ pendp = nullptr,
                                            const float* __restrict__ pend = nullptr) {
    const int h = blockIdx.x, vb = blockIdx.y, t = threadIdx.x;
    const int vi = t >> 3, kq = t & 7;
    const int vcol = vb * (blockDim.x >> 3) + vi;  // blockDim / 8 value columns per block
    const int kh = s.v2k[h];
    const int cur = (*curp + slot_off) % kStateSlots;
    const size_t off = ((size_t)h * 128 + vcol) * 128 + kq * 16;
    const float* sp = S + (size_t)cur * slot_floats + off;
    float st[16];
#pragma unroll
    for (int i = 0; i < 16; i += 4) {
        const float4 v = *reinterpret_cast<const float4*>(sp + i);
        st[i] = v.x; st[i + 1] = v.y; st[i + 2] = v.z; st[i + 3] = v.w;
    }
    if (pendp) {  // deferred commit: the previous step's accepted rows, then the state is written
        const int np = *pendp;
        if (np > 0) {
            const int prf = dn_pend_row_floats(s);
            for (int mi = 0; mi < np; mi++) {  // same arithmetic as dn_replay_kernel
                const float* pr = pend + (size_t)mi * prf;
                const float* kr = pr + kh * 128 + kq * 16;
                float k[16];
#pragma unroll
                for (int i = 0; i < 16; i += 4) {
                    const float4 b = *reinterpret_cast<const float4*>(kr + i);
                    k[i] = b.x; k[i + 1] = b.y; k[i + 2] = b.z; k[i + 3] = b.w;
                }
                const float* vg = pr + s.nk * 128;
                const float v = vg[h * 128 + vcol];
                const float dec = vg[s.nv * 128 + h];
                const float beta = vg[s.nv * 128 + s.nv + h];
                float part = 0.f;
#pragma unroll
                for (int i = 0; i < 16; i++) {
                    st[i] *= dec;
                    part += st[i] * k[i];
                }
                part += __shfl_xor_sync(0xffffffffu, part, 1);
                part += __shfl_xor_sync(0xffffffffu, part, 2);
                part += __shfl_xor_sync(0xffffffffu, part, 4);
                const float delta = (v - part) * beta;
#pragma unroll
                for (int i = 0; i < 16; i++) st[i] += k[i] * delta;
            }
            if (write_all && np >= dn_lazy_of(s, m)) {  // (else written later / after its own rows)
                float* dp = S + (size_t)(cur ^ 1) * slot_floats + off;
#pragma unroll
                for (int i = 0; i < 16; i += 4)
                    *reinterpret_cast<float4*>(dp + i) = make_float4(st[i], st[i + 1], st[i + 2], st[i + 3]);
            }
        }
    }
    const size_t qk_row = (size_t)2 * s.nk * 128;
    auto load = [&](int mi, float* q, float* k, float& v, float& dec, float& beta) {
        const float* qr = qk + mi * qk_row + kh * 128 + kq * 16;
        const float* kr = qr + s.nk * 128;
#pragma unroll
        for (int i = 0; i < 16; i += 4) {
            const float4 a = *reinterpret_cast<const float4*>(qr + i);
            const float4 b = *reinterpret_cast<const float4*>(kr + i);
            q[i] = a.x; q[i + 1] = a.y; q[i + 2] = a.z; q[i + 3] = a.w;
            k[i] = b.x; k[i + 1] = b.y; k[i + 2] = b.z; k[i + 3] = b.w;
        }
        v = cv[(size_t)mi * ch + 2 * s.nk * 128 + h * 128 + vcol];
        dec = gb[(size_t)mi * 2 * s.nv + h];
        beta = gb[(size_t)mi * 2 * s.nv + s.nv + h];
    };
    float q[16], k[16], v, dec, beta;
    float nq[16], nk[16], nv, ndec, nbeta;
    load(0, q, k, v, dec, beta);
    for (int mi = 0; mi < m; mi++) {
        if (mi + 1 < m) load(mi + 1, nq, nk, nv, ndec, nbeta);
        float part = 0.f;
#pragma unroll
        for (int i = 0; i < 16; i++) {
            st[i] *= dec;
            part += st[i] * k[i];
        }
        part += __shfl_xor_sync(0xffffffffu, part, 1);
        part += __shfl_xor_sync(0xffffffffu, part, 2);
        part += __shfl_xor_sync(0xffffffffu, part, 4);
        const float delta = (v - part) * beta;
        float po = 0.f;
#pragma unroll
        for (int i = 0; i < 16; i++) {
            st[i] += k[i] * delta;
            po += st[i] * q[i];
        }
        po += __shfl_xor_sync(0xffffffffu, po, 1);
        po += __shfl_xor_sync(0xffffffffu, po, 2);
        po += __shfl_xor_sync(0xffffffffu, po, 4);
        if (kq == 0) o[((size_t)mi * s.nv + h) * 128 + vcol] = po;
        if (!write_all && mi == m - 1) {
            float* dp = S + (size_t)((cur + 1 + mi) % kStateSlots) * slot_floats + off;
#pragma unroll
            for (int i = 0; i < 16; i += 4)
                *reinterpret_cast<float4*>(dp + i) =
                    make_float4(st[i], st[i + 1], st[i + 2], st[i + 3]);
        }
#pragma unroll
        for (int i = 0; i < 16; i++) { q[i] = nq[i]; k[i] = nk[i]; }
        v = nv; dec = ndec; beta = nbeta;
    }
}

__global__ void __launch_bounds__(128) dn_core_kernel(const float* __restrict__ cv, int ch,
                                                      DeltaShape s, const float* __restrict__ qk,
                                                      const float* __restrict__ gb,
                                                      float* __restrict__ S, size_t slot_floats,
                                                      const int* __restrict__ curp, int m,
                                                      bool write_all, float* __restrict__ o,
                                                      int slot_off, const int* __restrict__ pendp,
                                                      const float* __restrict__ pend) {
    dn_core_dev(cv, ch, s, qk, gb, S, slot_floats, curp, m, write_all, o, slot_off, pendp, pend);
}

// several sequences (batched speculative verification, write_all): segment blockIdx.z
__global__ void __launch_bounds__(128) dn_core_seg_kernel(DnSegs g, const float* __restrict__ cv,
                                                          int ch, DeltaShape s,
                                                          const float* __restrict__ qk,
                                                          const float* __restrict__ gb,
                                                          size_t slot_floats, float* __restrict__ o) {
    const int z = blockIdx.z, r0 = g.r0[z];
    dn_core_dev(cv + (size_t)r0 * ch, ch, s, qk + (size_t)r0 * 2 * s.nk * 128,
                gb + (size_t)r0 * 2 * s.nv, g.S[z], slot_floats, g.cur[z], g.m[z], !g.fin[z],
                o + (size_t)r0 * s.nv * 128, 0, g.pn[z], g.pend[z]);
}

// Deferred commit, end of a verification step: the accepted rows (k | v | decay | beta) of every
// layer go to the sequence's stash, cur flips when the step committed the previous rows.
// grid (layers, nseg), block 256
__global__ void dn_stash_seg_kernel(DnStashSegs r, int ch, DeltaShape s) {
    const int layer = blockIdx.x, z = blockIdx.y;
    if (r.fin[z]) {  // the pass wrote the state after its only row (pending ones included)
        if (layer == 0 && threadIdx.x == 0) {
            int* pn = r.pn[z];
            pn[1] = pn[0];  // undo: back to S[cur] + the same pending rows (stash untouched)
            r.save[z][0] = *r.cur[z];
            *r.cur[z] ^= 1;
            pn[0] = 0;
        }
        return;
    }
    const int n = *r.acc[z] + 1, r0 = r.r0[z];
    const int np = *r.pn[z], wrote = np >= dn_lazy_of(s, r.m[z]), base = wrote ? 0 : np;
    const int prf = dn_pend_row_floats(s), nkk = s.nk * 128, nvv = s.nv * 128;
    float* dst = r.pend[z] + ((size_t)layer * dn_pcap(s) + base) * prf;
    for (int mi = 0; mi < n; mi++) {
        float* d = dst + (size_t)mi * prf;
        const float* kq = r.qk[layer] + (size_t)(r0 + mi) * 2 * nkk + nkk;
        const float* cv = r.cv[layer] + (size_t)(r0 + mi) * ch + 2 * nkk;
        const float* gb = r.gb[layer] + (size_t)(r0 + mi) * 2 * s.nv;
        for (int i = threadIdx.x; i < prf; i += blockDim.x)
            d[i] = i < nkk ? kq[i] : i < nkk + nvv ? cv[i - nkk] : gb[i - nkk - nvv];
    }
    if (layer == 0 && threadIdx.x == 0) {
        int* pn = r.pn[z];
        if (wrote) *r.cur[z] ^= 1;
        r.save[z][0] = *r.cur[z];  // undo: same slot, the pending rows before this step
        pn[1] = base;
        *pn = base + n;
    }
}

// Deferred commit of one sequence before anything else reads its state: S[cur] + pending rows ->
// S[cur ^ 1] (dn_pend_commit then flips cur). grid (layers, nv, 8), block 128
__global__ void __launch_bounds__(128) dn_flush_kernel(DnFlush r, DeltaShape s, size_t slot_floats) {
    const int np = *r.pn;
    if (np <= 0) return;
    const int layer = blockIdx.x, h = blockIdx.y, vb = blockIdx.z, t = threadIdx.x;
    const int vi = t >> 3, kq = t & 7;
    const int vcol = vb * 16 + vi;
    const int kh = s.v2k[h];
    const int cur = *r.cur;
    const size_t off = ((size_t)h * 128 + vcol) * 128 + kq * 16;
    const float* sp = r.S[layer] + (size_t)cur * slot_floats + off;
    float st[16];
#pragma unroll
    for (int i = 0; i < 16; i += 4) {
        const float4 v = *reinterpret_cast<const float4*>(sp + i);
        st[i] = v.x; st[i + 1] = v.y; st[i + 2] = v.z; st[i + 3] = v.w;
    }
    const int prf = dn_pend_row_floats(s);
    const float* pend = r.pend + (size_t)layer * dn_pcap(s) * prf;
    for (int mi = 0; mi < np; mi++) {
        const float* pr = pend + (size_t)mi * prf;
        const float* kr = pr + kh * 128 + kq * 16;
        float k[16];
#pragma unroll
        for (int i = 0; i < 16; i += 4) {
            const float4 b = *reinterpret_cast<const float4*>(kr + i);
            k[i] = b.x; k[i + 1] = b.y; k[i + 2] = b.z; k[i + 3] = b.w;
        }
        const float* vg = pr + s.nk * 128;
        const float v = vg[h * 128 + vcol];
        const float dec = vg[s.nv * 128 + h];
        const float beta = vg[s.nv * 128 + s.nv + h];
        float part = 0.f;
#pragma unroll
        for (int i = 0; i < 16; i++) {
            st[i] *= dec;
            part += st[i] * k[i];
        }
        part += __shfl_xor_sync(0xffffffffu, part, 1);
        part += __shfl_xor_sync(0xffffffffu, part, 2);
        part += __shfl_xor_sync(0xffffffffu, part, 4);
        const float delta = (v - part) * beta;
#pragma unroll
        for (int i = 0; i < 16; i++) st[i] += k[i] * delta;
    }
    float* dp = r.S[layer] + (size_t)(cur ^ 1) * slot_floats + off;
#pragma unroll
    for (int i = 0; i < 16; i += 4)
        *reinterpret_cast<float4*>(dp + i) = make_float4(st[i], st[i + 1], st[i + 2], st[i + 3]);
}

__global__ void dn_pend_commit_kernel(int* cur, int* pn) {
    if (*pn > 0) *cur ^= 1;
    *pn = 0;
}

// grid (layers, nv, 8), block 128: same thread mapping and arithmetic as dn_core_kernel
__global__ void __launch_bounds__(128) dn_replay_kernel(DnReplay r, int ch, DeltaShape s,
                                                        size_t slot_floats,
                                                        const int* __restrict__ curp,
                                                        const int* __restrict__ accp) {
    const int layer = blockIdx.x, h = blockIdx.y, vb = blockIdx.z, t = threadIdx.x;
    const int vi = t >> 3, kq = t & 7;
    const int vcol = vb * 16 + vi;
    const int kh = s.v2k[h];
    const int cur = *curp, n = *accp + 1;
    const size_t off = ((size_t)h * 128 + vcol) * 128 + kq * 16;
    const float* sp = r.S[layer] + (size_t)cur * slot_floats + off;
    float st[16];
#pragma unroll
    for (int i = 0; i < 16; i += 4) {
        const float4 v = *reinterpret_cast<const float4*>(sp + i);
        st[i] = v.x; st[i + 1] = v.y; st[i + 2] = v.z; st[i + 3] = v.w;
    }
    const size_t qk_row = (size_t)2 * s.nk * 128;
    for (int mi = 0; mi < n; mi++) {
        const float* kr = r.qk[layer] + mi * qk_row + s.nk * 128 + kh * 128 + kq * 16;
        float k[16];
#pragma unroll
        for (int i = 0; i < 16; i += 4) {
            const float4 b = *reinterpret_cast<const float4*>(kr + i);
            k[i] = b.x; k[i + 1] = b.y; k[i + 2] = b.z; k[i + 3] = b.w;
        }
        const float v = r.cv[layer][(size_t)mi * ch + 2 * s.nk * 128 + h * 128 + vcol];
        const float dec = r.gb[layer][(size_t)mi * 2 * s.nv + h];
        const float beta = r.gb[layer][(size_t)mi * 2 * s.nv + s.nv + h];
        float part = 0.f;
#pragma unroll
        for (int i = 0; i < 16; i++) {
            st[i] *= dec;
            part += st[i] * k[i];
        }
        part += __shfl_xor_sync(0xffffffffu, part, 1);
        part += __shfl_xor_sync(0xffffffffu, part, 2);
        part += __shfl_xor_sync(0xffffffffu, part, 4);
        const float delta = (v - part) * beta;
#pragma unroll
        for (int i = 0; i < 16; i++) st[i] += k[i] * delta;
    }
    float* dp = r.S[layer] + (size_t)(cur ^ 1) * slot_floats + off;
#pragma unroll
    for (int i = 0; i < 16; i += 4)
        *reinterpret_cast<float4*>(dp + i) = make_float4(st[i], st[i + 1], st[i + 2], st[i + 3]);
}

// replay for several sequences: grid (layers, nv, 8 * nseg)
__global__ void __launch_bounds__(128) dn_replay_seg_kernel(DnReplaySegs r, int ch, DeltaShape s,
                                                            size_t slot_floats) {
    const int layer = blockIdx.x, h = blockIdx.y, z = blockIdx.z / 8, vb = blockIdx.z % 8;
    const int t = threadIdx.x, vi = t >> 3, kq = t & 7;
    const int vcol = vb * 16 + vi;
    const int kh = s.v2k[h];
    const int cur = *r.cur[z], n = *r.acc[z] + 1, r0 = r.r0[z];
    const size_t off = ((size_t)h * 128 + vcol) * 128 + kq * 16;
    float* Sb = r.S[z][layer];
    const float* sp = Sb + (size_t)cur * slot_floats + off;
    float st[16];
#pragma unroll
    for (int i = 0; i < 16; i += 4) {
        const float4 v = *reinterpret_cast<const float4*>(sp + i);
        st[i] = v.x; st[i + 1] = v.y; st[i + 2] = v.z; st[i + 3] = v.w;
    }
    const size_t qk_row = (size_t)2 * s.nk * 128;
    for (int mi = 0; mi < n; mi++) {
        const float* kr = r.qk[layer] + (r0 + mi) * qk_row + s.nk * 128 + kh * 128 + kq * 16;
        float k[16];
#pragma unroll
        for (int i = 0; i < 16; i += 4) {
            const float4 b = *reinterpret_cast<const float4*>(kr + i);
            k[i] = b.x; k[i + 1] = b.y; k[i + 2] = b.z; k[i + 3] = b.w;
        }
        const float v = r.cv[layer][(size_t)(r0 + mi) * ch + 2 * s.nk * 128 + h * 128 + vcol];
        const float dec = r.gb[layer][(size_t)(r0 + mi) * 2 * s.nv + h];
        const float beta = r.gb[layer][(size_t)(r0 + mi) * 2 * s.nv + s.nv + h];
        float part = 0.f;
#pragma unroll
        for (int i = 0; i < 16; i++) {
            st[i] *= dec;
            part += st[i] * k[i];
        }
        part += __shfl_xor_sync(0xffffffffu, part, 1);
        part += __shfl_xor_sync(0xffffffffu, part, 2);
        part += __shfl_xor_sync(0xffffffffu, part, 4);
        const float delta = (v - part) * beta;
#pragma unroll
        for (int i = 0; i < 16; i++) st[i] += k[i] * delta;
    }
    float* dp = Sb + (size_t)(cur ^ 1) * slot_floats + off;
#pragma unroll
    for (int i = 0; i < 16; i += 4)
        *reinterpret_cast<float4*>(dp + i) = make_float4(st[i], st[i + 1], st[i + 2], st[i + 3]);
}

__global__ void dn_commit_seg_kernel(DnReplaySegs r) {  // cur = (cur + 1) % 2 per sequence
    const int z = threadIdx.x;
    if (z < r.n) *(int*)r.cur[z] = (*r.cur[z] + 1) % kStateSlots;
}

__global__ void dn_commit_kernel(int* cur, const int* adv, int adv_const) {
    *cur = (*cur + (adv ? *adv + 1 : adv_const)) % kStateSlots;
}

// grid (m, nv), block 128
__global__ void dn_out_kernel(const float* __restrict__ o, const float* __restrict__ z, int ldz,
                              const bf16* __restrict__ w, int nv, float eps,
                              bf16* __restrict__ out, __half* __restrict__ xh) {
    const int mi = blockIdx.x, h = blockIdx.y, t = threadIdx.x;
    const size_t idx = ((size_t)mi * nv + h) * 128 + t;
    const float x = o[idx];
    const float r = rsqrtf(block_sum(x * x) / 128.f + eps);
    const float zz = z[(size_t)mi * ldz + h * 128 + t];
    const float v = __bfloat162float(w[t]) * (x * r) * silu(zz);
    out[idx] = __float2bfloat16(v);
    if (xh) store_xh(xh + (size_t)mi * nv * 128, h * 128 + t, v);
}

__global__ void silu_mul_kernel(const float* __restrict__ gu, int ld, int inter, int m,
                                bf16* __restrict__ out, __half* __restrict__ xh) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= inter * m) return;
    const int mi = i / inter, j = i % inter;
    const float g = gu[(size_t)mi * ld + j], u = gu[(size_t)mi * ld + inter + j];
    const float v = silu(g) * u;
    out[i] = __float2bfloat16(v);
    if (xh) store_xh(xh + (size_t)mi * inter, j, v);
}

__global__ void argmax_local_kernel(const float* __restrict__ logits, int v, int off,
                                    float* __restrict__ out, const int* __restrict__ map) {
    __shared__ float bv[32];
    __shared__ int bi[32];
    const int mi = blockIdx.x;
    const float* row = logits + (size_t)mi * v;
    float best = -FLT_MAX;
    int idx = 0;
    for (int i = threadIdx.x; i < v; i += blockDim.x)
        if (row[i] > best) { best = row[i]; idx = i; }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        const float ob = __shfl_xor_sync(0xffffffffu, best, o);
        const int oi = __shfl_xor_sync(0xffffffffu, idx, o);
        if (ob > best || (ob == best && oi < idx)) { best = ob; idx = oi; }
    }
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if (lane == 0) { bv[warp] = best; bi[warp] = idx; }
    __syncthreads();
    if (threadIdx.x == 0) {
        for (int w = 1; w < (int)(blockDim.x >> 5); w++)
            if (bv[w] > best || (bv[w] == best && bi[w] < idx)) { best = bv[w]; idx = bi[w]; }
        out[mi * 2] = best;
        out[mi * 2 + 1] = __int_as_float(map ? map[idx] : idx + off);
    }
}

__global__ void argmax_final_kernel(const float* __restrict__ pairs, int ngpu, int m,
                                    int* __restrict__ tokens) {
    const int mi = threadIdx.x;
    if (mi >= m) return;
    float best = -FLT_MAX;
    int idx = 0;
    for (int g = 0; g < ngpu; g++) {
        const float v = pairs[(g * m + mi) * 2];
        const int i = __float_as_int(pairs[(g * m + mi) * 2 + 1]);
        if (v > best || (v == best && i < idx)) { best = v; idx = i; }
    }
    tokens[mi] = idx;
}

}  // namespace

void set_image_table(const bf16* table) {
    CUDA_CHECK(cudaMemcpyToSymbol(g_img_table, &table, sizeof(table)));
}

void embed_gather(const bf16* table, const int* tokens, int m, int hidden, float* resid,
                  cudaStream_t st) {
    CHECK(hidden % 8 == 0, "embed hidden");
    embed_kernel<<<m, 640, 0, st>>>(table, tokens, hidden, resid);
}

void rmsnorm(const float* x, const bf16* w, int m, int n, float eps, bf16* out, cudaStream_t st,
             __half* xh) {
    rmsnorm_kernel<<<m, 512, 0, st>>>(x, w, n, eps, out, xh);
}

void attn_prep(const float* proj, int ld, const AttnShape& s, const bf16* qn_w, const bf16* kn_w,
               float eps, const float* inv_freq, const int* pos, int m, bf16* q_out, void* kc,
               void* vc, cudaStream_t st) {
    attn_prep_kernel<<<dim3(m, s.nq + 2 * s.nkv), 256, 0, st>>>(proj, ld, s, qn_w, kn_w, eps,
                                                               inv_freq, pos, q_out, kc, vc);
}

size_t attn_work_floats(const AttnShape& s) {
    return (size_t)s.nkv * kAttnSplits * kMaxGroup * (256 + 2);
}

constexpr size_t kAttnSmem =
    kMaxGroup * 256 * 2 + kTile * kKPad * 2 + kMaxGroup * kTile * 4 + 3 * kMaxGroup * 4;

void attention_init() {
    attention_decode_tc_init();
    CUDA_CHECK(cudaFuncSetAttribute(attn_split_kernel<8>,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize, (int)kAttnSmem));
    CUDA_CHECK(cudaFuncSetAttribute(attn_split_kernel<16>,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize, (int)kAttnSmem));
    CUDA_CHECK(cudaFuncSetAttribute(attn_split_kernel<24>,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize, (int)kAttnSmem));
    CUDA_CHECK(cudaFuncSetAttribute(attn_split_kernel<48>,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize, (int)kAttnSmem));
}

void attention(const bf16* q, const void* kc_, const void* vc_, const AttnShape& s,
               const float* proj, int ld, const int* pos, int m, float* work, bf16* out,
               cudaStream_t st, __half* xh) {
    const bf16* kc = (const bf16*)kc_;  // bf16 view for the CUDA-core fallback
    const bf16* vc = (const bf16*)vc_;
    int G = 0;
    for (int k = 0; k < s.nkv; k++) G = std::max(G, (s.kv_qe[k] - s.kv_qb[k]) * m);
    CHECK(G <= kMaxGroup, "attention group too large");
    static const bool use_cuda_cores = env_int("QE_ATTN_CC", 0);
    CHECK(!use_cuda_cores || s.fp8 || s.split == 0, "the CUDA-core attention has no shared-prefix path");
    const dim3 grid(s.nkv, kAttnSplits);
    for (int j = 0; j < s.nkv; j++)
        CHECK(!use_cuda_cores || s.fp8 || pm_full(s.kv_pm[j]), "the CUDA-core attention has no position-split path");
    if (!use_cuda_cores || s.fp8)
        attention_decode_tc(q, kc_, vc_, s, pos, m, work, G, st);
    else if (G <= 8)
        attn_split_kernel<8><<<grid, 256, kAttnSmem, st>>>(q, kc, vc, s, pos, m, work);
    else if (G <= 16)
        attn_split_kernel<16><<<grid, 256, kAttnSmem, st>>>(q, kc, vc, s, pos, m, work);
    else if (G <= 24)
        attn_split_kernel<24><<<grid, 256, kAttnSmem, st>>>(q, kc, vc, s, pos, m, work);
    else
        attn_split_kernel<48><<<grid, 256, kAttnSmem, st>>>(q, kc, vc, s, pos, m, work);
    const bool dec8 = s.fp8 && attention_dec8_enabled();
    if (xchg_on(s)) attn_stats_kernel<<<stats_blocks(s, m), 256, 0, st>>>(work, s, pos, m, dec8);
    attn_combine_kernel<<<dim3(m, s.nq), 256, 0, st>>>(work, s, proj, ld, m, out, pos, dec8, xh);
}

void rope_table(const AttnShape& s, const int* pos, int m, const float* inv_freq, double2* tab, cudaStream_t st) {
    rope_table_kernel<<<m, 32, 0, st>>>(s, pos, inv_freq, tab);
}

void rope_table_seg(const AttnSegs& g, const AttnShape& s, const float* inv_freq, double2* tab, cudaStream_t st) {
    (void)s;
    rope_table_seg_kernel<<<dim3(g.max_m, g.n), 32, 0, st>>>(g, inv_freq, tab);
}

void dn_conv(const float* proj, int ld, int ch, const bf16* conv_w, bf16* hist, const int* pos,
             int m, float* out, cudaStream_t st) {
    if (m <= kMaxGemvW4Tokens) {  // prefill / single path only (batched steps use dn_conv_seg)
        dn_conv_kernel<<<(ch + 255) / 256, 256, 0, st>>>(proj, ld, ch, conv_w, hist, pos, m, out);
        return;
    }
    dn_conv_par_kernel<<<dim3((ch + 255) / 256, m), 256, 0, st>>>(proj, ld, ch, conv_w, hist, pos, m, out);
    dn_conv_hist_kernel<<<(ch + 255) / 256, 256, 0, st>>>(proj, ld, ch, hist, pos, m);
}

bool dn_prefill_fused(int m) {
    const bool on = env_int("QE_DN_CQK", 1) != 0;
    return on && dn_chunk_min() > 0 && m >= dn_chunk_min() && m > kMaxGemvW4Tokens;
}

void dn_prefill(const float* proj, int ld, int ch, const bf16* conv_w, bf16* hist, const int* pos,
                const DeltaShape& s, const float* ab, const float* A, const float* dt_bias, float* S,
                size_t slot_floats, const int* cur, int m, float* cv, float* qk, float* gb, float* o,
                cudaStream_t st, int slot_off) {
    if (env_int("QE_DN_PREPF", 0)) {  // the chunk prep computes the front end itself (no qk / cv / gb)
        const DnConvIn ci{proj, ld, conv_w, hist, pos, ab, A, dt_bias};
        dn_core_chunked(cv, ch, s, qk, gb, S, slot_floats, cur, m, o, st, slot_off, false, &ci);
        dn_conv_hist_kernel<<<(ch + 255) / 256, 256, 0, st>>>(proj, ld, ch, hist, pos, m);
        return;
    }
    const bool h16 = env_int("QE_DN_H16", 0) != 0 && env_int("QE_DN_CQKW", 0) != 0;
    if (h16)  // windowed, fp16 q / k / v for the chunk prep
        dn_convqk_win_kernel<true><<<dim3((m + kConvTW - 1) / kConvTW, 2 * s.nk + s.nv + 1), 128, 0, st>>>(
            proj, ld, ch, conv_w, hist, pos, m, cv, s, ab, A, dt_bias, qk, gb);
    else if (env_int("QE_DN_CQKW", 0))  // windowed: proj read once per value
        dn_convqk_win_kernel<false><<<dim3((m + kConvTW - 1) / kConvTW, 2 * s.nk + s.nv + 1), 128, 0, st>>>(
            proj, ld, ch, conv_w, hist, pos, m, cv, s, ab, A, dt_bias, qk, gb);
    else
        dn_convqk_par_kernel<<<dim3(m, 2 * s.nk + s.nv + 1), 128, 0, st>>>(proj, ld, ch, conv_w, hist, pos, cv, s,
                                                                       ab, A, dt_bias, qk, gb);
    dn_conv_hist_kernel<<<(ch + 255) / 256, 256, 0, st>>>(proj, ld, ch, hist, pos, m);
    dn_core_chunked(cv, ch, s, qk, gb, S, slot_floats, cur, m, o, st, slot_off, h16);
}

// ---- dn_core2: decode recurrence (m <= kPendRows rows, optional deferred-commit rows) with the
// row inputs staged in shared memory and L lanes per value column (128 / L k entries per lane in
// 128 / L / 16 chains of 16). Every chain is one old 16-entry lane of dn_core_dev; chains combine
// in the old butterfly order (distance 1, 2, 4 over the old lanes: in-lane first, then shuffles),
// so the arithmetic is bit identical to dn_core_dev / dn_replay_kernel. Block 128 threads =
// 128 / L value columns; grid (nv, L[, nseg]).
template <int L>
struct DnCore2 {
    static constexpr int KPL = 128 / L, NC = KPL / 16, COLS = 128 / L;
    // shared layout: own rows [m][k 128 | q 128 | v COLS | dec | beta], pending [kPendRows][k | v | dec | beta]
    static constexpr int OWN = 256 + COLS + 4, PND = 128 + COLS + 4;  // (16-B aligned rows)
    static size_t smem(int m, int pcap) { return (size_t)(m * OWN + pcap * PND) * 4; }
};

template <int NC>
__device__ __forceinline__ float dn_tree(float (&p)[NC]) {
#pragma unroll
    for (int w = 1; w < NC; w *= 2)
#pragma unroll
        for (int c = 0; c < NC; c += 2 * w) p[c] = p[c] + p[c + w];
    return p[0];
}

template <int L>
__device__ __forceinline__ float dn_lanes(float x) {
#pragma unroll
    for (int d = 1; d < L; d *= 2) x += __shfl_xor_sync(0xffffffffu, x, d);
    return x;
}

template <int L>
__device__ __forceinline__ void dn_core2_dev(const float* __restrict__ cv, int ch, const DeltaShape& s,
                                             const float* __restrict__ qk, const float* __restrict__ gb,
                                             float* __restrict__ S, size_t slot_floats,
                                             const int* __restrict__ curp, int m, bool write_all,
                                             float* __restrict__ o, int slot_off,
                                             const int* __restrict__ pendp, const float* __restrict__ pend) {
    using C = DnCore2<L>;
    constexpr int KPL = C::KPL, NC = C::NC, COLS = C::COLS;
    extern __shared__ float dsm[];
    float* own = dsm;
    float* pnd = dsm + m * C::OWN;
    const int h = blockIdx.x, vb = blockIdx.y, t = threadIdx.x;
    const int col = t / L, j = t % L;
    const int vcol = vb * COLS + col;
    const int kh = s.v2k[h];
    const int cur = (*curp + slot_off) % kStateSlots;
    const size_t off = ((size_t)h * 128 + vcol) * 128 + j * KPL;
    const float* sp = S + (size_t)cur * slot_floats + off;
    float st[KPL];
#pragma unroll
    for (int i = 0; i < KPL; i += 4) {
        const float4 v = *reinterpret_cast<const float4*>(sp + i);
        st[i] = v.x; st[i + 1] = v.y; st[i + 2] = v.z; st[i + 3] = v.w;
    }
    const int np = pendp ? *pendp : 0;
    const size_t qk_row = (size_t)2 * s.nk * 128;
    // stage the rows: every load of a batch of 8 rows is issued before the shared stores
    // (128 threads: one k and one q entry each; threads < COLS + 2 one of v / decay / beta)
    const int xo = t < COLS ? 2 * s.nk * 128 + h * 128 + vb * COLS + t : t == COLS ? h : s.nv + h;
    const bool xs = t < COLS + 2;
    const float* xb = t < COLS ? cv : gb;
    const size_t xr = t < COLS ? (size_t)ch : (size_t)2 * s.nv;
    {
        float a[kPendRows], b[kPendRows], c[kPendRows];
#pragma unroll
        for (int mi = 0; mi < kPendRows; mi++)
            if (mi < m) {
                const float* qr = qk + mi * qk_row + kh * 128;
                a[mi] = qr[s.nk * 128 + t];
                b[mi] = qr[t];
                c[mi] = xs ? xb[mi * xr + xo] : 0.f;
            }
#pragma unroll
        for (int mi = 0; mi < kPendRows; mi++)
            if (mi < m) {
                float* r = own + mi * C::OWN;
                r[t] = a[mi];
                r[128 + t] = b[mi];
                if (xs) r[256 + (t < COLS ? t : t == COLS ? COLS : COLS + 1)] = c[mi];
            }
    }
    const int prf = dn_pend_row_floats(s);
    const int po_ = t < COLS ? s.nk * 128 + h * 128 + vb * COLS + t
                  : t == COLS ? s.nk * 128 + s.nv * 128 + h : s.nk * 128 + s.nv * 128 + s.nv + h;
    for (int m0 = 0; m0 < np; m0 += kPendRows) {
        float a[kPendRows], c[kPendRows];
#pragma unroll
        for (int i = 0; i < kPendRows; i++)
            if (m0 + i < np) {
                const float* pr = pend + (size_t)(m0 + i) * prf;
                a[i] = pr[kh * 128 + t];
                c[i] = xs ? pr[po_] : 0.f;
            }
#pragma unroll
        for (int i = 0; i < kPendRows; i++)
            if (m0 + i < np) {
                float* r = pnd + (m0 + i) * C::PND;
                r[t] = a[i];
                if (xs) r[128 + (t < COLS ? t : t == COLS ? COLS : COLS + 1)] = c[i];
            }
    }
    __syncthreads();
    // one token: st = dec * st; delta = (v - st.k) * beta; st += k delta; returns st.q (if q)
    auto step = [&](const float* kr, float v, float dec, float beta, const float* qr) {
        float p[NC];
#pragma unroll
        for (int c = 0; c < NC; c++) {
            p[c] = 0.f;
#pragma unroll
            for (int i = 0; i < 16; i += 4) {
                const float4 k4 = *reinterpret_cast<const float4*>(kr + j * KPL + c * 16 + i);
                const float kk[4] = {k4.x, k4.y, k4.z, k4.w};
#pragma unroll
                for (int e = 0; e < 4; e++) {
                    st[c * 16 + i + e] *= dec;
                    p[c] += st[c * 16 + i + e] * kk[e];
                }
            }
        }
        const float part = dn_lanes<L>(dn_tree<NC>(p));
        const float delta = (v - part) * beta;
        float po[NC];
#pragma unroll
        for (int c = 0; c < NC; c++) {
            po[c] = 0.f;
#pragma unroll
            for (int i = 0; i < 16; i += 4) {
                const float4 k4 = *reinterpret_cast<const float4*>(kr + j * KPL + c * 16 + i);
                const float kk[4] = {k4.x, k4.y, k4.z, k4.w};
                float qq[4] = {0.f, 0.f, 0.f, 0.f};
                if (qr) {
                    const float4 q4 = *reinterpret_cast<const float4*>(qr + j * KPL + c * 16 + i);
                    qq[0] = q4.x; qq[1] = q4.y; qq[2] = q4.z; qq[3] = q4.w;
                }
#pragma unroll
                for (int e = 0; e < 4; e++) {
                    st[c * 16 + i + e] += kk[e] * delta;
                    if (qr) po[c] += st[c * 16 + i + e] * qq[e];
                }
            }
        }
        if (!qr) return 0.f;
        return dn_lanes<L>(dn_tree<NC>(po));
    };
    if (np > 0) {  // deferred commit: the previous step's accepted rows, then the state is written
        for (int mi = 0; mi < np; mi++) {
            const float* r = pnd + mi * C::PND;
            step(r, r[128 + col], r[128 + COLS], r[129 + COLS], nullptr);
        }
        if (write_all && np >= dn_lazy_of(s, m)) {
            float* dp = S + (size_t)(cur ^ 1) * slot_floats + off;
#pragma unroll
            for (int i = 0; i < KPL; i += 4)
                *reinterpret_cast<float4*>(dp + i) = make_float4(st[i], st[i + 1], st[i + 2], st[i + 3]);
        }
    }
    for (int mi = 0; mi < m; mi++) {
        const float* r = own + mi * C::OWN;
        const float po = step(r, r[256 + col], r[256 + COLS], r[257 + COLS], r + 128);
        if (j == 0) o[((size_t)mi * s.nv + h) * 128 + vcol] = po;
    }
    if (!write_all) {
        float* dp = S + (size_t)((cur + m) % kStateSlots) * slot_floats + off;
#pragma unroll
        for (int i = 0; i < KPL; i += 4)
            *reinterpret_cast<float4*>(dp + i) = make_float4(st[i], st[i + 1], st[i + 2], st[i + 3]);
    }
}

template <int L>
__global__ void __launch_bounds__(128) dn_core2_kernel(const float* __restrict__ cv, int ch, DeltaShape s,
                                                       const float* __restrict__ qk, const float* __restrict__ gb,
                                                       float* __restrict__ S, size_t slot_floats,
                                                       const int* __restrict__ curp, int m, bool write_all,
                                                       float* __restrict__ o, int slot_off,
                                                       const int* __restrict__ pendp, const float* __restrict__ pend) {
    dn_core2_dev<L>(cv, ch, s, qk, gb, S, slot_floats, curp, m, write_all, o, slot_off, pendp, pend);
}

template <int L>
__global__ void __launch_bounds__(128) dn_core2_seg_kernel(DnSegs g, const float* __restrict__ cv, int ch,
                                                           DeltaShape s, const float* __restrict__ qk,
                                                           const float* __restrict__ gb, size_t slot_floats,
                                                           float* __restrict__ o) {
    const int z = blockIdx.z, r0 = g.r0[z];
    dn_core2_dev<L>(cv + (size_t)r0 * ch, ch, s, qk + (size_t)r0 * 2 * s.nk * 128, gb + (size_t)r0 * 2 * s.nv,
                    g.S[z], slot_floats, g.cur[z], g.m[z], !g.fin[z], o + (size_t)r0 * s.nv * 128, 0,
                    g.pn[z], g.pend[z]);
}
// QE_DN_CORE2 = lanes per value column (8, 4, 2; 1 = auto: 2 lanes for one sequence, 8 + the
// pipelined dn_core3 for a batch; 0 = dn_core_dev, the default: neither won).
// All bit identical.
static int g_dn_core2 = -1;
void dn_core2_set(int lanes) { g_dn_core2 = lanes; }
static int dn_core2_lanes(int nseg) {
    if (g_dn_core2 < 0) g_dn_core2 = env_int("QE_DN_CORE2", 0);
    return g_dn_core2 == 1 ? (nseg == 1 ? 2 : 8) : g_dn_core2;
}

template <int L>
static void dn_core2_launch(const float* cv, int ch, const DeltaShape& s, const float* qk, const float* gb,
                            float* S, size_t slot_floats, const int* cur, int m, bool write_all, float* o,
                            int slot_off, const int* pn, const float* pend, cudaStream_t st) {
    dn_core2_kernel<L><<<dim3(s.nv, L), 128, DnCore2<L>::smem(m, dn_pcap(s)), st>>>(cv, ch, s, qk, gb, S, slot_floats, cur, m,
                                                                        write_all, o, slot_off, pn, pend);
}

// true when the dn_core2 kernel took the call
static bool dn_core2(const float* cv, int ch, const DeltaShape& s, const float* qk, const float* gb, float* S,
                     size_t slot_floats, const int* cur, int m, bool write_all, float* o, int slot_off,
                     const int* pn, const float* pend, cudaStream_t st) {
    if (m > kPendRows) return false;
    // single-sequence passes: off unless forced (QE_DN_CORE2_SINGLE=1). Bit identical in every
    // kernel test (plain / spec / pending), yet with it on
    // test_shared_prefix's alone-from-scratch agent 1 diverges after 20 tokens - unexplained,
    // (1 = spec and plain, 2 = spec only, 3 = plain only)
    static const int single = env_int("QE_DN_CORE2_SINGLE", 0);
    if (!single || (single == 2 && !write_all) || (single == 3 && write_all)) return false;
    switch (dn_core2_lanes(1)) {
    case 8: dn_core2_launch<8>(cv, ch, s, qk, gb, S, slot_floats, cur, m, write_all, o, slot_off, pn, pend, st); return true;
    case 4: dn_core2_launch<4>(cv, ch, s, qk, gb, S, slot_floats, cur, m, write_all, o, slot_off, pn, pend, st); return true;
    case 2: dn_core2_launch<2>(cv, ch, s, qk, gb, S, slot_floats, cur, m, write_all, o, slot_off, pn, pend, st); return true;
    default: return false;
    }
}

static bool dn_core2_seg(const DnSegs& g, const float* cv, int ch, const DeltaShape& s, const float* qk,
                         const float* gb, size_t slot_floats, float* o, cudaStream_t st) {
    int mx = 1;
    for (int i = 0; i < g.n; i++) mx = std::max(mx, g.m[i]);
    if (mx > kPendRows) return false;
    switch (dn_core2_lanes(g.n)) {
    case 8: dn_core2_seg_kernel<8><<<dim3(s.nv, 8, g.n), 128, DnCore2<8>::smem(mx, dn_pcap(s)), st>>>(g, cv, ch, s, qk, gb, slot_floats, o); return true;
    case 4: dn_core2_seg_kernel<4><<<dim3(s.nv, 4, g.n), 128, DnCore2<4>::smem(mx, dn_pcap(s)), st>>>(g, cv, ch, s, qk, gb, slot_floats, o); return true;
    case 2: dn_core2_seg_kernel<2><<<dim3(s.nv, 2, g.n), 128, DnCore2<2>::smem(mx, dn_pcap(s)), st>>>(g, cv, ch, s, qk, gb, slot_floats, o); return true;
    default: return false;
    }
}

// ---- dn_core3: batched decode recurrence as a persistent, software-pipelined kernel. A tile is
// (sequence, v head, 16 value columns) with the thread mapping of dn_core_dev (8 lanes of 16 k
// entries per column, same arithmetic: bit identical). Each block walks tiles b, b + G, ...; while
// it computes one tile, the next tile's state (registers) and row inputs (cp.async into the other
// shared buffer) are already in flight, so DRAM keeps streaming during the token loop instead of
// idling between the load and compute phases of a wave.
__device__ __forceinline__ void cpa4(float* sdst, const float* gsrc) {
    const unsigned sa = (unsigned)__cvta_generic_to_shared(sdst);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" ::"r"(sa), "l"(gsrc));
}
__device__ __forceinline__ void cpa_commit() { asm volatile("cp.async.commit_group;\n" ::); }
__device__ __forceinline__ void cpa_wait1() { asm volatile("cp.async.wait_group 1;\n" ::); }

struct DnCore3 {
    static constexpr int OWN = 256 + 16 + 4, PND = 128 + 16 + 4;  // as DnCore2<8>
    __host__ __device__ static int buf_floats(int mx, int pcap) { return mx * OWN + pcap * PND; }
};

__global__ void __launch_bounds__(128) dn_core3_kernel(DnSegs g, const float* __restrict__ cv, int ch,
                                                       DeltaShape s, const float* __restrict__ qk,
                                                       const float* __restrict__ gb, size_t slot_floats,
                                                       float* __restrict__ o, int mx, int ntiles) {
    extern __shared__ float dsm[];
    const int bf = DnCore3::buf_floats(mx, dn_pcap(s));
    const int t = threadIdx.x, col = t >> 3, j = t & 7;
    const int G = gridDim.x;
    const size_t qk_row = (size_t)2 * s.nk * 128;
    const int prf = dn_pend_row_floats(s);
    // per-tile scalars (seq, head, column block, rows, pending, slot)
    auto tz = [&](int tau) { return tau / (s.nv * 8); };
    auto prefetch = [&](int tau, int np, float* B) {
        const int z = tz(tau), h = (tau >> 3) % s.nv, vb = tau & 7, kh = s.v2k[h];
        const int m = g.m[z], r0 = g.r0[z];
        float* own = B;
        float* pnd = B + mx * DnCore3::OWN;
        for (int mi = 0; mi < m; mi++) {
            float* r = own + mi * DnCore3::OWN;
            const float* qr = qk + (size_t)(r0 + mi) * qk_row + kh * 128;
            cpa4(r + t, qr + s.nk * 128 + t);
            cpa4(r + 128 + t, qr + t);
            if (t < 16) cpa4(r + 256 + t, cv + (size_t)(r0 + mi) * ch + 2 * s.nk * 128 + h * 128 + vb * 16 + t);
            else if (t < 18) cpa4(r + 256 + t, gb + (size_t)(r0 + mi) * 2 * s.nv + (t - 16) * s.nv + h);
        }
        for (int mi = 0; mi < np; mi++) {
            float* r = pnd + mi * DnCore3::PND;
            const float* pr = g.pend[z] + (size_t)mi * prf;
            cpa4(r + t, pr + kh * 128 + t);
            if (t < 16) cpa4(r + 128 + t, pr + s.nk * 128 + h * 128 + vb * 16 + t);
            else if (t < 18) cpa4(r + 128 + t, pr + s.nk * 128 + s.nv * 128 + (t - 16) * s.nv + h);
        }
    };
    auto off_of = [&](int tau) {
        const int h = (tau >> 3) % s.nv, vb = tau & 7;
        return ((size_t)h * 128 + vb * 16 + col) * 128 + j * 16;
    };
    auto load_st = [&](int tau, int cur, float* st) {
        const float* sp = g.S[tz(tau)] + (size_t)cur * slot_floats + off_of(tau);
#pragma unroll
        for (int i = 0; i < 16; i += 4) {
            const float4 v = *reinterpret_cast<const float4*>(sp + i);
            st[i] = v.x; st[i + 1] = v.y; st[i + 2] = v.z; st[i + 3] = v.w;
        }
    };
    auto scal = [&](int tau, int& np, int& cur) {
        if (tau < ntiles) {
            const int z = tz(tau);
            np = g.pn[z] ? *g.pn[z] : 0;
            cur = *g.cur[z];
        }
    };
    int tau = blockIdx.x;
    if (tau >= ntiles) return;
    int np_c = 0, cur_c = 0, np_n = 0, cur_n = 0, np_nn = 0, cur_nn = 0;
    scal(tau, np_c, cur_c);
    float st[16], nst[16];
    load_st(tau, cur_c, st);
    prefetch(tau, np_c, dsm);
    cpa_commit();
    scal(tau + G, np_n, cur_n);
    int cb = 0;
    for (; tau < ntiles; tau += G) {
        const int tn = tau + G;
        if (tn < ntiles) {
            load_st(tn, cur_n, nst);
            prefetch(tn, np_n, dsm + (cb ^ 1) * bf);
        }
        cpa_commit();
        scal(tn + G, np_nn, cur_nn);
        cpa_wait1();
        __syncthreads();
        {  // tile tau
            const int z = tz(tau), h = (tau >> 3) % s.nv, vb = tau & 7;
            const int vcol = vb * 16 + col, m = g.m[z], r0 = g.r0[z];
            const bool write_all = !g.fin[z];
            const float* B = dsm + cb * bf;
            const float* own = B;
            const float* pnd = B + mx * DnCore3::OWN;
            float* S = g.S[z];
            const size_t off = off_of(tau);
            auto step = [&](const float* kr, float v, float dec, float beta, const float* qr) {
                float part = 0.f;
#pragma unroll
                for (int i = 0; i < 16; i += 4) {
                    const float4 k4 = *reinterpret_cast<const float4*>(kr + j * 16 + i);
                    const float kk[4] = {k4.x, k4.y, k4.z, k4.w};
#pragma unroll
                    for (int e = 0; e < 4; e++) {
                        st[i + e] *= dec;
                        part += st[i + e] * kk[e];
                    }
                }
                part += __shfl_xor_sync(0xffffffffu, part, 1);
                part += __shfl_xor_sync(0xffffffffu, part, 2);
                part += __shfl_xor_sync(0xffffffffu, part, 4);
                const float delta = (v - part) * beta;
                float po = 0.f;
#pragma unroll
                for (int i = 0; i < 16; i += 4) {
                    const float4 k4 = *reinterpret_cast<const float4*>(kr + j * 16 + i);
                    const float kk[4] = {k4.x, k4.y, k4.z, k4.w};
                    if (qr) {
                        const float4 q4 = *reinterpret_cast<const float4*>(qr + j * 16 + i);
                        const float qq[4] = {q4.x, q4.y, q4.z, q4.w};
#pragma unroll
                        for (int e = 0; e < 4; e++) {
                            st[i + e] += kk[e] * delta;
                            po += st[i + e] * qq[e];
                        }
                    } else {
#pragma unroll
                        for (int e = 0; e < 4; e++) st[i + e] += kk[e] * delta;
                    }
                }
                if (!qr) return 0.f;
                po += __shfl_xor_sync(0xffffffffu, po, 1);
                po += __shfl_xor_sync(0xffffffffu, po, 2);
                po += __shfl_xor_sync(0xffffffffu, po, 4);
                return po;
            };
            if (np_c > 0) {
                for (int mi = 0; mi < np_c; mi++) {
                    const float* r = pnd + mi * DnCore3::PND;
                    step(r, r[128 + col], r[144], r[145], nullptr);
                }
                if (write_all && np_c >= dn_lazy_of(s, m)) {
                    float* dp = S + (size_t)(cur_c ^ 1) * slot_floats + off;
#pragma unroll
                    for (int i = 0; i < 16; i += 4)
                        *reinterpret_cast<float4*>(dp + i) = make_float4(st[i], st[i + 1], st[i + 2], st[i + 3]);
                }
            }
            for (int mi = 0; mi < m; mi++) {
                const float* r = own + mi * DnCore3::OWN;
                const float po = step(r, r[256 + col], r[272], r[273], r + 128);
                if (j == 0) o[((size_t)(r0 + mi) * s.nv + h) * 128 + vcol] = po;
            }
            if (!write_all) {
                float* dp = S + (size_t)((cur_c + m) % kStateSlots) * slot_floats + off;
#pragma unroll
                for (int i = 0; i < 16; i += 4)
                    *reinterpret_cast<float4*>(dp + i) = make_float4(st[i], st[i + 1], st[i + 2], st[i + 3]);
            }
        }
        __syncthreads();
#pragma unroll
        for (int i = 0; i < 16; i++) st[i] = nst[i];
        np_c = np_n; cur_c = cur_n; np_n = np_nn; cur_n = cur_nn;
        cb ^= 1;
    }
}


// ---- dn_core4: batched decode recurrence with CPT value columns per thread group (8 lanes of 16
// k entries, as dn_core_dev). Every k / q fragment loaded from shared memory feeds CPT columns
// (CPT x fewer loads per FMA) and the CPT columns' chains and shuffles interleave (ILP). Each
// column's arithmetic is exactly dn_core_dev's: bit identical. Block 128 threads = 16 groups =
// 16 * CPT columns; grid (nv, 8 / CPT, nseg).
template <int CPT>
struct DnCore4 {
    static constexpr int COLS = 16 * CPT;
    static constexpr int OWN = 256 + COLS + 4, PND = 128 + COLS + 4;
    static size_t smem(int m, int pcap) { return (size_t)(m * OWN + pcap * PND) * 4; }
};

template <int CPT>
__global__ void __launch_bounds__(128) dn_core4_seg_kernel(DnSegs g, const float* __restrict__ cv0, int ch,
                                                           DeltaShape s, const float* __restrict__ qk0,
                                                           const float* __restrict__ gb0, size_t slot_floats,
                                                           float* __restrict__ o0) {
    using C = DnCore4<CPT>;
    constexpr int COLS = C::COLS;
    extern __shared__ float dsm[];
    const int z = blockIdx.z, r0 = g.r0[z], m = g.m[z];
    const float* cv = cv0 + (size_t)r0 * ch;
    const float* qk = qk0 + (size_t)r0 * 2 * s.nk * 128;
    const float* gb = gb0 + (size_t)r0 * 2 * s.nv;
    float* o = o0 + (size_t)r0 * s.nv * 128;
    float* S = g.S[z];
    const bool write_all = !g.fin[z];
    float* own = dsm;
    float* pnd = dsm + m * C::OWN;
    const int h = blockIdx.x, vb = blockIdx.y, t = threadIdx.x;
    const int grp = t >> 3, j = t & 7;  // columns vb * COLS + grp + 16 * c
    const int kh = s.v2k[h];
    const int cur = *g.cur[z];
    float st[CPT][16];
#pragma unroll
    for (int c = 0; c < CPT; c++) {
        const float* sp = S + (size_t)cur * slot_floats + ((size_t)h * 128 + vb * COLS + grp + 16 * c) * 128 + j * 16;
#pragma unroll
        for (int i = 0; i < 16; i += 4) {
            const float4 v = *reinterpret_cast<const float4*>(sp + i);
            st[c][i] = v.x; st[c][i + 1] = v.y; st[c][i + 2] = v.z; st[c][i + 3] = v.w;
        }
    }
    const int np = g.pn[z] ? *g.pn[z] : 0;
    const size_t qk_row = (size_t)2 * s.nk * 128;
    const int xo = t < COLS ? 2 * s.nk * 128 + h * 128 + vb * COLS + t : t == COLS ? h : s.nv + h;
    const bool xs = t < COLS + 2;
    const float* xb = t < COLS ? cv : gb;
    const size_t xr = t < COLS ? (size_t)ch : (size_t)2 * s.nv;
    {
        float a[kPendRows], b[kPendRows], c[kPendRows];
#pragma unroll
        for (int mi = 0; mi < kPendRows; mi++)
            if (mi < m) {
                const float* qr = qk + mi * qk_row + kh * 128;
                a[mi] = qr[s.nk * 128 + t];
                b[mi] = qr[t];
                c[mi] = xs ? xb[mi * xr + xo] : 0.f;
            }
#pragma unroll
        for (int mi = 0; mi < kPendRows; mi++)
            if (mi < m) {
                float* r = own + mi * C::OWN;
                r[t] = a[mi];
                r[128 + t] = b[mi];
                if (xs) r[256 + t] = c[mi];  // v[COLS] | dec | beta
            }
    }
    const int prf = dn_pend_row_floats(s);
    const int po_ = t < COLS ? s.nk * 128 + h * 128 + vb * COLS + t
                  : t == COLS ? s.nk * 128 + s.nv * 128 + h : s.nk * 128 + s.nv * 128 + s.nv + h;
    for (int m0 = 0; m0 < np; m0 += kPendRows) {
        float a[kPendRows], c[kPendRows];
#pragma unroll
        for (int i = 0; i < kPendRows; i++)
            if (m0 + i < np) {
                const float* pr = g.pend[z] + (size_t)(m0 + i) * prf;
                a[i] = pr[kh * 128 + t];
                c[i] = xs ? pr[po_] : 0.f;
            }
#pragma unroll
        for (int i = 0; i < kPendRows; i++)
            if (m0 + i < np) {
                float* r = pnd + (m0 + i) * C::PND;
                r[t] = a[i];
                if (xs) r[128 + t] = c[i];
            }
    }
    __syncthreads();
    // one token for the CPT columns: kr k[128], vr v[COLS] then dec, beta; qr q (null: no output)
    auto step = [&](const float* kr, const float* vr, const float* qr, float* po) {
        const float dec = vr[COLS], beta = vr[COLS + 1];
        float p[CPT];
#pragma unroll
        for (int c = 0; c < CPT; c++) p[c] = 0.f;
#pragma unroll
        for (int i = 0; i < 16; i += 4) {
            const float4 k4 = *reinterpret_cast<const float4*>(kr + j * 16 + i);
            const float kk[4] = {k4.x, k4.y, k4.z, k4.w};
#pragma unroll
            for (int e = 0; e < 4; e++)
#pragma unroll
                for (int c = 0; c < CPT; c++) {
                    st[c][i + e] *= dec;
                    p[c] += st[c][i + e] * kk[e];
                }
        }
#pragma unroll
        for (int d = 1; d < 8; d *= 2)
#pragma unroll
            for (int c = 0; c < CPT; c++) p[c] += __shfl_xor_sync(0xffffffffu, p[c], d);
        float delta[CPT];
#pragma unroll
        for (int c = 0; c < CPT; c++) {
            delta[c] = (vr[grp + 16 * c] - p[c]) * beta;
            po[c] = 0.f;
        }
#pragma unroll
        for (int i = 0; i < 16; i += 4) {
            const float4 k4 = *reinterpret_cast<const float4*>(kr + j * 16 + i);
            const float kk[4] = {k4.x, k4.y, k4.z, k4.w};
            if (qr) {
                const float4 q4 = *reinterpret_cast<const float4*>(qr + j * 16 + i);
                const float qq[4] = {q4.x, q4.y, q4.z, q4.w};
#pragma unroll
                for (int e = 0; e < 4; e++)
#pragma unroll
                    for (int c = 0; c < CPT; c++) {
                        st[c][i + e] += kk[e] * delta[c];
                        po[c] += st[c][i + e] * qq[e];
                    }
            } else {
#pragma unroll
                for (int e = 0; e < 4; e++)
#pragma unroll
                    for (int c = 0; c < CPT; c++) st[c][i + e] += kk[e] * delta[c];
            }
        }
        if (qr) {
#pragma unroll
            for (int d = 1; d < 8; d *= 2)
#pragma unroll
                for (int c = 0; c < CPT; c++) po[c] += __shfl_xor_sync(0xffffffffu, po[c], d);
        }
    };
    auto store = [&](int slot) {
#pragma unroll
        for (int c = 0; c < CPT; c++) {
            float* dp = S + (size_t)slot * slot_floats + ((size_t)h * 128 + vb * COLS + grp + 16 * c) * 128 + j * 16;
#pragma unroll
            for (int i = 0; i < 16; i += 4)
                *reinterpret_cast<float4*>(dp + i) = make_float4(st[c][i], st[c][i + 1], st[c][i + 2], st[c][i + 3]);
        }
    };
    float po[CPT];
    if (np > 0) {
        for (int mi = 0; mi < np; mi++) {
            const float* r = pnd + mi * C::PND;
            step(r, r + 128, nullptr, po);
        }
        if (write_all && np >= dn_lazy_of(s, m)) store(cur ^ 1);
    }
    for (int mi = 0; mi < m; mi++) {
        const float* r = own + mi * C::OWN;
        step(r, r + 256, r + 128, po);
        if (j == 0)
#pragma unroll
            for (int c = 0; c < CPT; c++) o[((size_t)mi * s.nv + h) * 128 + vb * COLS + grp + 16 * c] = po[c];
    }
    if (!write_all) store((cur + m) % kStateSlots);
}

static bool dn_core4_seg(const DnSegs& g, const float* cv, int ch, const DeltaShape& s, const float* qk,
                         const float* gb, size_t slot_floats, float* o, cudaStream_t st) {
    static const int cpt = env_int("QE_DN_CPT", 0);
    if (cpt != 2 && cpt != 4) return false;
    int mx = 1;
    for (int i = 0; i < g.n; i++) mx = std::max(mx, g.m[i]);
    if (mx > kPendRows) return false;
    if (cpt == 2)
        dn_core4_seg_kernel<2><<<dim3(s.nv, 4, g.n), 128, DnCore4<2>::smem(mx, dn_pcap(s)), st>>>(g, cv, ch, s, qk, gb, slot_floats, o);
    else
        dn_core4_seg_kernel<4><<<dim3(s.nv, 2, g.n), 128, DnCore4<4>::smem(mx, dn_pcap(s)), st>>>(g, cv, ch, s, qk, gb, slot_floats, o);
    return true;
}

static bool dn_core3_seg(const DnSegs& g, const float* cv, int ch, const DeltaShape& s, const float* qk,
                         const float* gb, size_t slot_floats, float* o, cudaStream_t st) {
    static const int on = env_int("QE_DN_CORE3", 1);
    static int g_bps = 0;
    if (!on || g.n < 2) return false;
    int mx = 1;
    for (int i = 0; i < g.n; i++) mx = std::max(mx, g.m[i]);
    if (mx > kPendRows) return false;
    const size_t smem = (size_t)2 * DnCore3::buf_floats(mx, dn_pcap(s)) * 4;
    static int nsm = 0;
    if (!nsm) {
        int dev;
        CUDA_CHECK(cudaGetDevice(&dev));
        CUDA_CHECK(cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev));
        CUDA_CHECK(cudaFuncSetAttribute(dn_core3_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 64 * 1024));
    }
    int bps = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&bps, dn_core3_kernel, 128, smem));
    if (env_set("QE_DN_CORE3_BPS")) bps = std::min(bps, env_int("QE_DN_CORE3_BPS", bps));
    g_bps = bps;
    const int ntiles = g.n * s.nv * 8;
    const int grid = std::min(ntiles, std::max(1, bps) * nsm);
    dn_core3_kernel<<<grid, 128, smem, st>>>(g, cv, ch, s, qk, gb, slot_floats, o, mx, ntiles);
    return true;
}

void dn_core(const float* conv_out, int ch, const DeltaShape& s, const float* ab, const float* A,
             const float* dt_bias, float* S, size_t slot_floats, const int* cur, int m,
             bool write_all, float* qk, float* gb, float* o, cudaStream_t st, int slot_off,
             const int* pn, const float* pend) {
    dn_qk_kernel<<<dim3(m, s.nk + 1), 128, 0, st>>>(conv_out, ch, s, ab, A, dt_bias, qk, gb);
    const int cmin = dn_chunk_min();
    if (!write_all && cmin > 0 && m >= cmin) {  // prefill: chunked form on tensor cores
        dn_core_chunked(conv_out, ch, s, qk, gb, S, slot_floats, cur, m, o, st, slot_off);
        return;
    }
    if (dn_core2(conv_out, ch, s, qk, gb, S, slot_floats, cur, m, write_all, o, slot_off, pn, pend, st)) return;
    dn_core_kernel<<<dim3(s.nv, 8), 128, 0, st>>>(conv_out, ch, s, qk, gb, S, slot_floats, cur, m,
                                                  write_all, o, slot_off, pn, pend);
}

__global__ void add_int_kernel(int* dst, const int* src, int off) { *dst = *src + off; }
void add_int(int* dst, const int* src, int off, cudaStream_t st) {
    add_int_kernel<<<1, 1, 0, st>>>(dst, src, off);
}

__global__ void copy_int_kernel(int* dst, const int* src) { *dst = *src; }
void copy_int(int* dst, const int* src, cudaStream_t st) { copy_int_kernel<<<1, 1, 0, st>>>(dst, src); }

void dn_replay(const DnReplay& r, int ch, const DeltaShape& s, size_t slot_floats, const int* cur,
               const int* acc, cudaStream_t st) {
    CHECK(kStateSlots == 2, "replay assumes ping-pong slots");
    dn_replay_kernel<<<dim3(r.n, s.nv, 8), 128, 0, st>>>(r, ch, s, slot_floats, cur, acc);
}

void attention_seg(const AttnSegs& g, const AttnShape& s, int ld, const bf16* qn_w,
                   const bf16* kn_w, float eps, const float* inv_freq, cudaStream_t st) {
    if (!(s.fp8 && attention_dec8_enabled())) {  // other kernels: one call per sequence
        // (the stats exchange has one flag value and receive slot per launch: several sequences
        // through attention() would share them, so split heads need the batched fp8 path)
        CHECK(!xchg_on(s), "position-split heads need the fp8 batched attention (QE_KV_FP8=1, QE_MTP_FP8=1)");
        for (int z = 0; z < g.n; z++) {
            const AttnSeg& c = g.c[z];
            AttnShape a = s;
            a.max_t = c.max_t;
            a.split = c.split;
            a.kc2 = c.kc2;
            a.vc2 = c.vc2;
            a.xrow0 = seg_row0(g, z);
            attn_prep_kernel<<<dim3(c.m, s.nq + 2 * s.nkv), 256, 0, st>>>(c.proj, ld, a, qn_w, kn_w, eps,
                                                                       inv_freq, c.pos, c.q, c.kc, c.vc);
            attention(c.q, c.kc, c.vc, a, c.proj, ld, c.pos, c.m, c.work, c.out, st);
        }
        return;
    }
    attn_prep_seg_kernel<<<dim3(g.max_m, s.nq + 2 * s.nkv, g.n), 256, 0, st>>>(g, ld, s, qn_w, kn_w, eps,
                                                                           inv_freq);
    CUDA_CHECK(cudaGetLastError());
    int G = 0;
    for (int k = 0; k < s.nkv; k++) G = std::max(G, (s.kv_qe[k] - s.kv_qb[k]) * g.max_m);
    CHECK(G <= kMaxGroup, "attention group too large");
    attention_dec8_seg(g, s, G, st);
    CUDA_CHECK(cudaGetLastError());
    if (xchg_on(s)) attn_stats_seg_kernel<<<dim3(stats_blocks(s, g.max_m), 1, g.n), 256, 0, st>>>(g, s, true);
    attn_combine_seg_kernel<<<dim3(g.max_m, s.nq, g.n), 256, 0, st>>>(g, s, ld, true);
    CUDA_CHECK(cudaGetLastError());
}

void dn_conv_seg(const DnSegs& g, const float* proj, int ld, int ch, const bf16* conv_w, float* out,
                 cudaStream_t st) {
    dn_conv_seg_kernel<<<dim3((ch + 255) / 256, g.n), 256, 0, st>>>(g, proj, ld, ch, conv_w, out);
}

void dn_core_seq(const float* conv_out, int ch, const DeltaShape& s, const float* ab, const float* A,
                 const float* dt_bias, float* S, size_t slot_floats, const int* cur, int m,
                 float* qk, float* gb, float* o, cudaStream_t st) {
    dn_qk_kernel<<<dim3(m, s.nk + 1), 128, 0, st>>>(conv_out, ch, s, ab, A, dt_bias, qk, gb);
    dn_core_kernel<<<dim3(s.nv, 8), 128, 0, st>>>(conv_out, ch, s, qk, gb, S, slot_floats, cur, m,
                                                  false, o, 0, nullptr, nullptr);
}
bool dn_fuse_on() {
    static const bool on = env_int("QE_DN_FUSE", 1);
    return on;
}
void dn_convqk(const float* proj, int ld, int ch, const bf16* conv_w, bf16* hist, const int* pos,
               int m, float* cv, const DeltaShape& s, const float* ab, const float* A,
               const float* dt_bias, float* qk, float* gb, cudaStream_t st) {
    const int nb = s.nk + (s.nv * 128 + 255) / 256 + 1;
    CHECK(m <= 13, "dn_convqk m=%d", m);
    dn_convqk_kernel<<<dim3(nb, m), 256, 0, st>>>(proj, ld, ch, conv_w, hist, pos, cv, s, ab, A, dt_bias, qk, gb);
}
void dn_convqk_seg(const DnSegs& g, const float* proj, int ld, int ch, const bf16* conv_w, float* cv,
                   const DeltaShape& s, const float* ab, const float* A, const float* dt_bias,
                   float* qk, float* gb, cudaStream_t st) {
    const int nb = s.nk + (s.nv * 128 + 255) / 256 + 1;
    int mx = 1;
    for (int i = 0; i < g.n; i++) mx = std::max(mx, g.m[i]);
    CHECK(mx <= 13, "dn_convqk_seg m=%d", mx);
    dn_convqk_seg_kernel<<<dim3(nb, mx, g.n), 256, 0, st>>>(g, proj, ld, ch, conv_w, cv, s, ab, A, dt_bias, qk, gb);
}
// recurrence only (q / k / decay / beta already computed)
void dn_core_only(const float* conv_out, int ch, const DeltaShape& s, float* S, size_t slot_floats,
                  const int* cur, int m, bool write_all, const float* qk, const float* gb, float* o,
                  cudaStream_t st, const int* pn, const float* pend) {
    if (dn_core2(conv_out, ch, s, qk, gb, S, slot_floats, cur, m, write_all, o, 0, pn, pend, st)) return;
    dn_core_kernel<<<dim3(s.nv, 8), 128, 0, st>>>(conv_out, ch, s, qk, gb, S, slot_floats, cur, m,
                                                  write_all, o, 0, pn, pend);
}

__global__ void dn_single_begin_kernel(int* cur, int* save, int* pn) {
    *save = *cur;
    pn[1] = 0;
}
void dn_single_begin(int* cur, int* save, int* pn, cudaStream_t st) {
    dn_single_begin_kernel<<<1, 1, 0, st>>>(cur, save, pn);
}
__global__ void dn_trim_kernel(int* cur, const int* save, int* pn, int r) {
    if (r == 0) {
        *cur = *save;
        pn[0] = pn[1];
    } else {
        pn[0] = pn[1] + r;
    }
}
void dn_trim(int* cur, const int* save, int* pn, int r, cudaStream_t st) {
    dn_trim_kernel<<<1, 1, 0, st>>>(cur, save, pn, r);
}
bool dn_defer1_on() {
    static const bool on = env_int("QE_DN_DEFER1", 0);
    return on;
}
bool dn_direct_on() {
    static const bool on = env_int("QE_DN_DIRECT", 1);
    return on;
}
bool dn_defer_on() {
    // default OFF until the verifier case C2 (heisenbug with dn_core2 plain + defer) is explained
    static const bool on = env_set("QE_DN_DEFER") ? env_int("QE_DN_DEFER", true) != 0 : true;  // default on (night-merge gates: shared_prefix, spec, batch, archive PASS; n=1 4K/19.5K unchanged)
    return on;
}
void dn_stash_seg(const DnStashSegs& r, int nlayers, int ch, const DeltaShape& s, cudaStream_t st) {
    dn_stash_seg_kernel<<<dim3(nlayers, r.n), 256, 0, st>>>(r, ch, s);
}
void dn_flush(const DnFlush& r, int nlayers, const DeltaShape& s, size_t slot_floats, cudaStream_t st) {
    CHECK(kStateSlots == 2, "deferred commit assumes ping-pong slots");
    dn_flush_kernel<<<dim3(nlayers, s.nv, 8), 128, 0, st>>>(r, s, slot_floats);
    dn_pend_commit_kernel<<<1, 1, 0, st>>>(r.cur, r.pn);
}

void dn_qk_rows(const float* conv_out, int ch, const DeltaShape& s, const float* ab, const float* A,
                const float* dt_bias, int m, float* qk, float* gb, cudaStream_t st) {
    dn_qk_kernel<<<dim3(m, s.nk + 1), 128, 0, st>>>(conv_out, ch, s, ab, A, dt_bias, qk, gb);
}

void dn_core_seg(const DnSegs& g, const float* conv_out, int ch, const DeltaShape& s,
                 const float* qk, const float* gb, size_t slot_floats, float* o, cudaStream_t st) {
    if (dn_core2_lanes(g.n) && dn_core4_seg(g, conv_out, ch, s, qk, gb, slot_floats, o, st)) return;
    if (dn_core2_lanes(g.n) && dn_core3_seg(g, conv_out, ch, s, qk, gb, slot_floats, o, st)) return;
    if (dn_core2_seg(g, conv_out, ch, s, qk, gb, slot_floats, o, st)) return;
    dn_core_seg_kernel<<<dim3(s.nv, 8, g.n), 128, 0, st>>>(g, conv_out, ch, s, qk, gb, slot_floats, o);
}

void dn_replay_seg(const DnReplaySegs& r, int nlayers, int ch, const DeltaShape& s,
                   size_t slot_floats, cudaStream_t st) {
    dn_replay_seg_kernel<<<dim3(nlayers, s.nv, 8 * r.n), 128, 0, st>>>(r, ch, s, slot_floats);
    dn_commit_seg_kernel<<<1, 32, 0, st>>>(r);
}

void dn_commit(int* cur, const int* adv, int adv_const, cudaStream_t st) {
    dn_commit_kernel<<<1, 1, 0, st>>>(cur, adv, adv_const);
}

int gemm_i8_variant();  // gemm.cu
// Prefill: dn_out + the int8 input of the W4A8 out_proj (x_to_i8_kernel's layouts, group G, from the
// bf16-rounded outputs: identical to quantizing `out` afterwards); the bf16 rows are not written.
template <int G>
__global__ void __launch_bounds__(128) dn_out_i8_kernel(const float* __restrict__ o, const float* __restrict__ z,
                                                        int ldz, const bf16* __restrict__ w, int nv, float eps,
                                                        int8_t* __restrict__ xq, float* __restrict__ xsc, bool v2) {
    __shared__ float sh[128];
    const int mi = blockIdx.x, h = blockIdx.y, t = threadIdx.x;
    const size_t idx = ((size_t)mi * nv + h) * 128 + t;
    const float x = o[idx];
    const float r = rsqrtf(block_sum(x * x) / 128.f + eps);
    const float zz = z[(size_t)mi * ldz + h * 128 + t];
    const float v = __bfloat162float(w[t]) * (x * r) * silu(zz);
    sh[t] = __bfloat162float(__float2bfloat16(v));
    __syncthreads();
    if (t >= 32) return;
    // warp 0: lane = float4 index within the head (ar_finish_kernel's quantization of 4 k per lane)
    constexpr int L = G / 4;
    const int K = nv * 128, col = h * 128 + 4 * t, i = col / 4;
    const float2 f0 = make_float2(sh[4 * t], sh[4 * t + 1]), f1 = make_float2(sh[4 * t + 2], sh[4 * t + 3]);
    float mx = fmaxf(fmaxf(fabsf(f0.x), fabsf(f0.y)), fmaxf(fabsf(f1.x), fabsf(f1.y)));
#pragma unroll
    for (int d = 1; d < L; d <<= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, d));
    const float sc = mx > 0.f ? __uint_as_float(__float_as_uint(mx / 127.f) & ~3u) : 1.f;
    const float inv = mx > 0.f ? __frcp_rn(sc) : 0.f;
    const int q0 = __float2int_rn(f0.x * inv), q1 = __float2int_rn(f0.y * inv);
    const int q2 = __float2int_rn(f1.x * inv), q3 = __float2int_rn(f1.y * inv);
    int sum = q0 + q1 + q2 + q3;
#pragma unroll
    for (int d = 1; d < L; d <<= 1) sum += __shfl_xor_sync(0xffffffffu, sum, d);
    constexpr int NREC = (128 / G + 1) / 2;
    if ((i % L) == 0 && v2) {  // gemm_i8v_kernel records (gemm.cu i8v_rec_store<G, false>), row mi of the call
        constexpr int SUB = 128 / G;
        float2* p = reinterpret_cast<float2*>(reinterpret_cast<float4*>(xsc) +
                                              ((size_t)(mi >> 4) * (K / 128) + col / 128) * (SUB * 2 * 8) +
                                              ((col % 128) / G) * 2 * 8 + (mi & 7)) + ((mi >> 3) & 1);
        const float cz = __int_as_float((int)0x4B400000 - 8 * sum);
        p[0] = make_float2(cz, cz);
        p[16] = make_float2(sc, -12582912.f * sc);
    } else if ((i % L) == 0) {
        const int sub = (col % 128) / G;
        *reinterpret_cast<float2*>(xsc + (((size_t)mi * (K / 128) + col / 128) * NREC + sub / 2) * 4 + (sub & 1) * 2) =
            make_float2(sc, __int_as_float((int)0x4B400000 - 8 * sum));
    }
    const uint32_t a = (uint32_t)(q0 & 0xFF) | ((uint32_t)(q1 & 0xFF) << 8) | ((uint32_t)(q2 & 0xFF) << 16) |
                       ((uint32_t)(q3 & 0xFF) << 24);
    const uint32_t b = __shfl_xor_sync(0xffffffffu, a, 2);
    const int ip = i & 3;
    uint32_t out;
    if ((ip & 2) == 0) out = (a & 0xFF) | ((b & 0xFF) << 8) | (((a >> 8) & 0xFF) << 16) | (((b >> 8) & 0xFF) << 24);
    else out = ((b >> 16) & 0xFF) | (((a >> 16) & 0xFF) << 8) | (((b >> 24) & 0xFF) << 16) | ((a >> 24) << 24);
    const int j = 2 * (ip & 1) + (ip >> 1);
    reinterpret_cast<uint32_t*>(xq + (size_t)mi * K + (col & ~15))[j] = out;
}

void dn_out(const float* o, const float* z, int ldz, const bf16* w, int nv, float eps, int m,
            bf16* out, cudaStream_t st, __half* xh, int8_t* xq, float* xsc, int G) {
    if (xq && G == 64) dn_out_i8_kernel<64><<<dim3(m, nv), 128, 0, st>>>(o, z, ldz, w, nv, eps, xq, xsc, gemm_i8_variant() >= 2);
    else if (xq && G == 32) dn_out_i8_kernel<32><<<dim3(m, nv), 128, 0, st>>>(o, z, ldz, w, nv, eps, xq, xsc, gemm_i8_variant() >= 2);
    else if (xq && G == 128) dn_out_i8_kernel<128><<<dim3(m, nv), 128, 0, st>>>(o, z, ldz, w, nv, eps, xq, xsc, gemm_i8_variant() >= 2);
    else dn_out_kernel<<<dim3(m, nv), 128, 0, st>>>(o, z, ldz, w, nv, eps, out, xh);
}

void silu_mul(const float* gu, int ld, int inter, int m, bf16* out, cudaStream_t st, __half* xh) {
    silu_mul_kernel<<<(inter * m + 255) / 256, 256, 0, st>>>(gu, ld, inter, m, out, xh);
}

void argmax_local(const float* logits, int v, int m, int vocab_off, float* out_pairs,
                  cudaStream_t st, const int* map) {
    argmax_local_kernel<<<m, 1024, 0, st>>>(logits, v, vocab_off, out_pairs, map);
}

void argmax_final(const float* pairs, int ngpu, int m, int* tokens, cudaStream_t st) {
    argmax_final_kernel<<<1, m > 32 ? 64 : 32, 0, st>>>(pairs, ngpu, m, tokens);  // thread per row (<= 64)
}
