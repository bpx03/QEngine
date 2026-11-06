// Latent Sparse Attention for the 27B full-attention layers. Own implementation of the D-LSA flat design
// of arXiv 2610.42147 (Apache-2.0 reference: github.com/Prefrontal-AI/latent-sparse-attention, ideas only).
// Step 2b: simple, correct kernels for the end-to-end gate; the fast prefill / decode kernels come in step 3.
#include <cfloat>
#include <cuda_fp8.h>
#include "lsa.cuh"
#include "watchdog.cuh"

__constant__ LsaCfg c_lsa = {kLsaBpb, kLsaK, kLsaP};
LsaCfg g_lsa = {kLsaBpb, kLsaK, kLsaP};
void lsa_config(const LsaCfg& c) {  // per device (constant memory is per context)
    g_lsa = c;
    CUDA_CHECK(cudaMemcpyToSymbol(c_lsa, &c, sizeof c));
}

namespace {

__device__ __forceinline__ float bfr(float x) { return __bfloat162float(__float2bfloat16(x)); }

__device__ __forceinline__ float fp8f(unsigned char b) {
    const __half_raw h = __nv_cvt_fp8_to_halfraw((__nv_fp8_storage_t)b, __NV_E4M3);
    return __half2float(__half(h));
}

// out[o] = sum_x s[x] * w[x][o] for the 768 x 64 map, 256 threads: 4 parts of 192 rows, fixed order
__device__ __forceinline__ void map_product(const float* s, const bf16* __restrict__ w, float* red, float* out) {
    const int t = threadIdx.x, o = t & 63, part = t >> 6;
    float acc = 0.f;
    for (int x = part * 192; x < part * 192 + 192; x++) acc += s[x] * __bfloat162float(w[x * kLsaDim + o]);
    red[part * 64 + o] = acc;
    __syncthreads();
    if (t < 64) out[t] = (red[t] + red[64 + t]) + (red[128 + t] + red[192 + t]);
}

// grid m, block 256: the 6 q heads of each kv group summed over dims 64..255 (fp32), bf16, times the map
__global__ void __launch_bounds__(256) lsa_iq_partial_kernel(const bf16* __restrict__ q, int nq, const LsaRoles r,
                                                              const bf16* __restrict__ wq, float* __restrict__ out) {
    __shared__ float s[kLsaIn];
    __shared__ float red[256];
    const int i = blockIdx.x, t = threadIdx.x;
    for (int x = t; x < kLsaIn; x += 256) s[x] = 0.f;
    __syncthreads();
    if (t < 192)
        for (int j = 0; j < nq; j++)
            if (r.qmask >> j & 1) s[r.qg[j] * 192 + t] += __bfloat162float(q[((size_t)i * nq + j) * 256 + 64 + t]);
    __syncthreads();
    for (int x = t; x < kLsaIn; x += 256) s[x] = bfr(s[x]);
    __syncthreads();
    map_product(s, wq, red, out + (size_t)i * kLsaDim);
}

// grid k1 - k0, block 256: the key's RoPE-free dims of every kv head this card counts at its position, dequantised
// (fp8 x scale), bf16, times the map
__global__ void __launch_bounds__(256) lsa_ik_partial_kernel(const unsigned char* __restrict__ kc, const AttnShape s,
                                                              float ks, int k0, const LsaRoles r,
                                                              const bf16* __restrict__ wk, float* __restrict__ out) {
    __shared__ float sm[kLsaIn];
    __shared__ float red[256];
    const int key = k0 + blockIdx.x, t = threadIdx.x;
    for (int x = t; x < kLsaIn; x += 256) sm[x] = 0.f;
    __syncthreads();
    if (t < 192)
        for (int j = 0; j < s.nkv; j++) {
            const PosMap& pm = s.kv_pm[j];
            if (!r.kcount[j] || !pm_owned(pm, key)) continue;
            const size_t row = head_rows(s, j, s.max_t) + (pm_rank(pm, key) - pm_rank(pm, s.split));
            sm[r.kg[j] * 192 + t] = bfr(fp8f(kc[row * 256 + 64 + t]) * ks);
        }
    __syncthreads();
    map_product(sm, wk, red, out + (size_t)blockIdx.x * kLsaDim);
}

__global__ void lsa_seq_kernel(unsigned* seq) { *seq += 1; }

// grid ceil(n / 16), block 256 (16 rows x 16 float4): this card's partial rows into every card's receive slot (own
// memory or P2P), then the flags
__global__ void lsa_send_kernel(const LsaXchg x, const float* __restrict__ part, int n, unsigned* done) {
    const unsigned seq = *x.seq;
    const int slot = seq & 1, row = blockIdx.x * 16 + (threadIdx.x >> 4), c4 = threadIdx.x & 15;
    if (row < n) {
        const float4 v = reinterpret_cast<const float4*>(part + (size_t)row * kLsaDim)[c4];
        for (int g = 0; g < x.ngpu; g++) {
            float* dst = g == x.rank ? x.rx : x.prx[g];
            reinterpret_cast<float4*>(dst + ((size_t)(slot * x.ngpu + x.rank) * kLsaXRows + row) * kLsaDim)[c4] = v;
        }
    }
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0 && atomicAdd(done, 1u) == gridDim.x - 1) {
        *done = 0;
        __threadfence_system();
        for (int g = 0; g < x.ngpu; g++)
            if (g != x.rank) qe_st_release_sys(x.prf[g] + slot * x.ngpu + x.rank, seq);
    }
}

// one block: wait for the peers' flags of this exchange
__global__ void lsa_wait_kernel(const LsaXchg x) {
    const unsigned seq = *x.seq;
    const int slot = seq & 1, g = threadIdx.x;
    if (g >= x.ngpu || g == x.rank) return;
    unsigned ws = 0;
    while ((int)(qe_ld_acquire_sys(x.rf + slot * x.ngpu + g) - seq) < 0) {
        __nanosleep(64);
        QE_WD_POLL(ws, kWdKvPrefill, g, seq, qe_ld_acquire_sys(x.rf + slot * x.ngpu + g));
    }
}

// grid n, block 64: sum in rank order, L2 normalise, bf16
__global__ void lsa_sum_kernel(const LsaXchg x, bf16* __restrict__ out) {
    __shared__ float red[2];
    const unsigned seq = *x.seq;
    const int slot = seq & 1, row = blockIdx.x, t = threadIdx.x;
    float v = 0.f;
    for (int g = 0; g < x.ngpu; g++) v += __ldcg(x.rx + ((size_t)(slot * x.ngpu + g) * kLsaXRows + row) * kLsaDim + t);
    float sq = v * v;
    for (int o = 16; o; o >>= 1) sq += __shfl_xor_sync(0xffffffffu, sq, o);
    if ((t & 31) == 0) red[t >> 5] = sq;
    __syncthreads();
    const float nrm = sqrtf(red[0] + red[1]);
    out[(size_t)row * kLsaDim + t] = __float2bfloat16(v / fmaxf(nrm, 1e-30f));
}

// Ward clustering of one 512-key block, grid blocks, block 512: 128 pools (sums of 4 unit index keys) and their Gram
// matrix in smem (computed once, then updated per merge); each round every live bucket finds its cheapest partner
// allowed by the cap (|a| + |b| - |a + b|, ties to the lower index), mutual pairs merge into the lower pool, the last
// round only the cheapest pairs, until 64 buckets.
constexpr int kWardSmem = (kLsaPools * kLsaDim + kLsaPools * kLsaPools) * 4;
__global__ void __launch_bounds__(512) lsa_ward_kernel(const bf16* __restrict__ ik, int b0, LsaLayer L) {
    constexpr int P = kLsaPools;
    extern __shared__ float wsm[];
    float* S = wsm;              // [P][64]
    float* G = wsm + P * kLsaDim;  // [P][P]
    __shared__ unsigned char sz[P], alive[P], parent[P], best[P], keep[P], num[P];
    __shared__ float bcost[P];
    __shared__ int s_alive, s_go;
    const int t = threadIdx.x, b = b0 + blockIdx.x, lane = t & 31;
    const bf16* kb = ik + (size_t)blockIdx.x * kLsaBlk * kLsaDim;
    for (int e = t; e < P * kLsaDim; e += 512) {
        const int pi = e / kLsaDim, d = e % kLsaDim;
        float a = 0.f;
        for (int r = 0; r < kLsaPool; r++) a += __bfloat162float(kb[(size_t)(pi * kLsaPool + r) * kLsaDim + d]);
        S[pi * kLsaDim + d] = a;
    }
    if (t < P) sz[t] = kLsaPool, alive[t] = 1, parent[t] = t;
    __syncthreads();
    const bool merge = LSA_BPB < P;  // block mode: every pool its own bucket, no Gram matrix, no rounds
    for (int e = t; merge && e < P * P; e += 512) {  // G = S S^T (each entry one fixed-order dot: symmetric bit for bit)
        const int i = e / P, j = e % P;
        const float* x = S + (i < j ? i : j) * kLsaDim;
        const float* y = S + (i < j ? j : i) * kLsaDim;
        float g = 0.f;
        for (int d = 0; d < kLsaDim; d++) g += x[d] * y[d];
        G[e] = g;
    }
    __syncthreads();
    for (int round = 0; merge && round < 256; round++) {
        if (t == 0) {
            int c = 0;
            for (int i = 0; i < P; i++) c += alive[i];
            s_alive = c;
            s_go = 0;
        }
        __syncthreads();
        const int left = s_alive - LSA_BPB;
        if (left <= 0) break;
        {   // thread (i, quarter): best partner among 32 candidates, then over the 4 quarters (adjacent lanes)
            const int i = t >> 2, qq = t & 3;
            float bc = FLT_MAX;
            int bj = 255;
            if (alive[i]) {
                const float gi = G[i * P + i], ai = sqrtf(fmaxf(gi, 0.f));
                for (int j = qq * 32; j < qq * 32 + 32; j++) {
                    if (j == i || !alive[j] || sz[i] + sz[j] > kLsaCap) continue;
                    const float gj = G[j * P + j];
                    const float c = ai + sqrtf(fmaxf(gj, 0.f)) - sqrtf(fmaxf(gi + gj + 2.f * G[i * P + j], 0.f));
                    if (c < bc) bc = c, bj = j;
                }
            }
            for (int o = 1; o <= 2; o <<= 1) {
                const float oc = __shfl_xor_sync(0xffffffffu, bc, o);
                const int oj = __shfl_xor_sync(0xffffffffu, bj, o);
                if (oc < bc || (oc == bc && oj < bj)) bc = oc, bj = oj;
            }
            if (qq == 0) best[i] = (unsigned char)bj, bcost[i] = bc;
            (void)lane;
        }
        __syncthreads();
        if (t < P) {  // mutual pairs, the cheapest `left` of them (ties: lower index)
            const int bt = best[t];
            int k = 0;
            if (alive[t] && bt != 255 && bt > t && best[bt] == t) {
                int rank = 0;
                for (int j = 0; j < P; j++) {
                    const int bj = best[j];
                    if (alive[j] && bj != 255 && bj > j && best[bj] == j &&
                        (bcost[j] < bcost[t] || (bcost[j] == bcost[t] && j < t)))
                        rank++;
                }
                k = rank < left;
            }
            keep[t] = (unsigned char)k;
        }
        __syncthreads();
        // merge: S and the Gram rows of the kept pools (from the old G), then their columns
        for (int e = t; e < P * kLsaDim; e += 512) {
            const int i = e / kLsaDim;
            if (keep[i]) S[e] += S[best[i] * kLsaDim + e % kLsaDim];
        }
        for (int e = t; e < P * P; e += 512) {
            const int i = e / P, x = e % P;
            if (keep[i]) G[e] += G[best[i] * P + x];
        }
        __syncthreads();
        for (int e = t; e < P * P; e += 512) {
            const int x = e / P, k = e % P;
            if (keep[k]) G[e] += G[x * P + best[k]];
        }
        if (t < P && keep[t]) {
            const int j = best[t];
            sz[t] += sz[j];
            alive[j] = 0;
            parent[j] = (unsigned char)t;
            s_go = 1;
        }
        __syncthreads();
        const int go = s_go;  // read by every thread before thread 0 resets it next round
        __syncthreads();
        if (!go) break;
    }
    // labels: surviving root of every pool, bucket number = live pools before it
    if (t < P) {
        int r = t;
        while (parent[r] != r) r = parent[r];
        best[t] = (unsigned char)r;
    }
    __syncthreads();
    if (t < P) {
        int c = 0;
        for (int j = 0; j < t; j++) c += alive[j];
        num[t] = (unsigned char)c;
    }
    __syncthreads();
    if (t < LSA_BPB) {  // every bucket of the block starts empty (a block that could not merge to 64: non-finite keys)
        L.mem[(size_t)b * LSA_BPB + t] = 0xFFFFFFFFu;
        for (int d = 0; d < kLsaDim; d++) L.cent[((size_t)b * LSA_BPB + t) * kLsaDim + d] = __float2bfloat16(0.f);
    }
    __syncthreads();
    if (t < P && alive[t] && num[t] < LSA_BPB) {
        const int n = num[t];
        unsigned m = 0xFFFFFFFFu;
        int k = 0;
        for (int j = 0; j < P; j++)
            if (best[j] == t && k < 4) {
                m = (m & ~(0xFFu << (8 * k))) | ((unsigned)j << (8 * k));
                k++;
            }
        L.mem[(size_t)b * LSA_BPB + n] = m;
        float a = 0.f;
        for (int d = 0; d < kLsaDim; d++) a += S[t * kLsaDim + d] * S[t * kLsaDim + d];
        const float inv = 1.f / fmaxf(sqrtf(a), 1e-30f);
        for (int d = 0; d < kLsaDim; d++)
            L.cent[((size_t)b * LSA_BPB + n) * kLsaDim + d] = __float2bfloat16(S[t * kLsaDim + d] * inv);
    }
}

// visible buckets / window start of the row at position p with `closed` clustered blocks: the blocks before its own
// that are clustered; every key not in one of them is in its window
__device__ __forceinline__ int lsa_vis(int p, int closed) { return min(p / kLsaBlk, closed) * LSA_BPB; }
__device__ __forceinline__ int lsa_ws(int p, int closed) { return min(max(0, p + 1 - kLsaBlk), closed * kLsaBlk); }

// scores of rows [r0, r0 + rows) against the visible centroids: S[row][b] fp32, pitch V (an upper bound); rows at
// pos[0] + r0 + i. grid (ceil(V / 64), ceil(rows / 32)), block 256
__global__ void __launch_bounds__(256) lsa_score_kernel(const bf16* __restrict__ iq, int r0, int rows, const int* pos,
                                                         const int* closed, const bf16* __restrict__ cent, int V,
                                                         float* __restrict__ S) {
    __shared__ float qs[32][kLsaDim + 1];
    __shared__ float cs[64][kLsaDim + 1];
    const int t = threadIdx.x, b0 = blockIdx.x * 64, i0 = blockIdx.y * 32;
    const int p0 = pos[0] + r0, cl = *closed;
    const int vmax = lsa_vis(p0 + min(rows, i0 + 32) - 1, cl);
    if (b0 >= vmax) return;
    for (int e = t; e < 32 * kLsaDim; e += 256) {
        const int i = e / kLsaDim, d = e % kLsaDim;
        qs[i][d] = i0 + i < rows ? __bfloat162float(iq[(size_t)(r0 + i0 + i) * kLsaDim + d]) : 0.f;
    }
    for (int e = t; e < 64 * kLsaDim; e += 256) {
        const int c = e / kLsaDim, d = e % kLsaDim;
        cs[c][d] = b0 + c < vmax ? __bfloat162float(cent[(size_t)(b0 + c) * kLsaDim + d]) : 0.f;
    }
    __syncthreads();
    const int c = t & 63, ig = t >> 6;  // 4 groups of 8 rows
    for (int ii = 0; ii < 8; ii++) {
        const int i = ig * 8 + ii;
        if (i0 + i >= rows) continue;
        const int vis = lsa_vis(p0 + i0 + i, cl);
        if (b0 + c >= vis) continue;
        float a = 0.f;
        for (int d = 0; d < kLsaDim; d++) a += qs[i][d] * cs[c][d];
        S[(size_t)(i0 + i) * V + b0 + c] = a;
    }
}

__device__ __forceinline__ unsigned ord_key(float f) {
    const unsigned u = __float_as_uint(f);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

// Radix-select step on 256 bins by one warp: the highest bin b (scanning 255 -> 0) where the elements above it plus
// the bin reach climit (counts hc) or wlimit (weights hw, optional). found = false: no bin reaches them; then bin = 0
// and the "before" sums cover bins 255..1 (the caller's serial loops stopped at bin 0 the same way).
__device__ __forceinline__ void warp_bin_find(const unsigned* hc, const unsigned* hw, unsigned c0, unsigned w0,
                                              unsigned climit, unsigned wlimit, int& bin, unsigned& cb, unsigned& wb,
                                              bool& found) {
    const int lane = threadIdx.x & 31, top = 255 - 8 * lane;
    unsigned sc = 0, sw = 0;
    for (int k = 0; k < 8; k++) sc += hc[top - k], sw += hw ? hw[top - k] : 0u;
    unsigned pc = sc, pw = sw;  // inclusive scan over the lanes (lane 0 = the highest bins)
    for (int o = 1; o < 32; o <<= 1) {
        const unsigned a = __shfl_up_sync(0xffffffffu, pc, o), b = __shfl_up_sync(0xffffffffu, pw, o);
        if (lane >= o) pc += a, pw += b;
    }
    unsigned c = c0 + pc - sc, w = w0 + pw - sw;
    int fb = -1;
    unsigned fc = 0, fw = 0;
    for (int k = 0; k < 8; k++) {
        const int bb = top - k;
        const unsigned hcb = hc[bb], hwb = hw ? hw[bb] : 0u;
        if (c + hcb >= climit || w + hwb >= wlimit) {
            fb = bb, fc = c, fw = w;
            break;
        }
        c += hcb;
        w += hwb;
    }
    const unsigned m = __ballot_sync(0xffffffffu, fb >= 0);
    if (m) {
        const int f = __ffs(m) - 1;
        bin = __shfl_sync(0xffffffffu, fb, f);
        cb = __shfl_sync(0xffffffffu, fc, f);
        wb = __shfl_sync(0xffffffffu, fw, f);
        found = true;
    } else {
        const unsigned tc = __shfl_sync(0xffffffffu, c0 + pc, 31), tw = __shfl_sync(0xffffffffu, w0 + pw, 31);
        bin = 0;
        cb = tc - hc[0];
        wb = tw - (hw ? hw[0] : 0u);
        found = false;
    }
}

// Exact top-p per row (largest score first, ties to the lower bucket), grid rows, block 1024: radix select of the
// p-th largest 32-bit order key (4 passes of 8 bits), then the candidates >= it sorted by (key desc, id asc).
constexpr int kSelCand = 2048;
__device__ void lsa_topp_row(const float* __restrict__ row, int V, int p, int closed, unsigned* __restrict__ selrow,
                             int* __restrict__ nselrow);
__global__ void __launch_bounds__(1024) lsa_topp_kernel(const float* __restrict__ S, int V, int r0, const int* pos,
                                                         const int* closed, unsigned* __restrict__ sel,
                                                         int* __restrict__ nsel) {
    const int i = blockIdx.x;
    lsa_topp_row(S + (size_t)i * V, V, pos[0] + r0 + i, *closed, sel + (size_t)i * kLsaP, nsel + i);
}
__device__ void lsa_topp_row(const float* __restrict__ row, int V, int p, int closed, unsigned* __restrict__ selrow,
                             int* __restrict__ nselrow) {
    __shared__ unsigned hist[256];
    __shared__ unsigned long long cand[kSelCand];
    __shared__ unsigned s_prefix, s_need, s_cnt;
    const int t = threadIdx.x;
    const int vis = min(V, lsa_vis(p, closed));
    unsigned prefix = 0, need = LSA_P;
    if (vis > LSA_P) {
        for (int pass = 0; pass < 4; pass++) {
            const int sh = 24 - 8 * pass;
            if (t < 256) hist[t] = 0;
            __syncthreads();
            const unsigned hm = pass ? (0xFFFFFFFFu << (32 - 8 * pass)) : 0u;
            for (int b = t; b < vis; b += 1024) {
                const unsigned k = ord_key(row[b]);
                if ((k & hm) == (prefix & hm)) atomicAdd(&hist[(k >> sh) & 255], 1u);
            }
            __syncthreads();
            if (t < 32) {
                int bin;
                unsigned c, wdummy;
                bool f;
                warp_bin_find(hist, nullptr, 0u, 0u, need, 0xFFFFFFFFu, bin, c, wdummy, f);
                if (t == 0) {
                    s_prefix = prefix | ((unsigned)bin << sh);
                    s_need = need - c;
                }
            }
            __syncthreads();
            prefix = s_prefix;
            need = s_need;
            __syncthreads();
        }
    } else {
        prefix = 0;  // take all visible
    }
    if (t == 0) s_cnt = 0;
    __syncthreads();
    for (int b = t; b < vis; b += 1024) {
        const unsigned k = ord_key(row[b]);
        if (k >= prefix) {
            const unsigned c = atomicAdd(&s_cnt, 1u);
            if (c < kSelCand) cand[c] = ((unsigned long long)k << 32) | (0xFFFFFFFFu - (unsigned)b);
        }
    }
    __syncthreads();
    const int nc = min((int)s_cnt, kSelCand);
    for (int e = nc + t; e < kSelCand; e += 1024) cand[e] = 0ull;
    __syncthreads();
    for (int k = 2; k <= kSelCand; k <<= 1)  // bitonic, descending
        for (int j = k >> 1; j > 0; j >>= 1) {
            for (int e = t; e < kSelCand; e += 1024) {
                const int x = e ^ j;
                if (x > e) {
                    const bool desc = (e & k) == 0;
                    const unsigned long long a = cand[e], c = cand[x];
                    if (desc ? a < c : a > c) cand[e] = c, cand[x] = a;
                }
            }
            __syncthreads();
        }
    const int np = min(nc, LSA_P);
    for (int e = t; e < np; e += 1024) selrow[e] = 0xFFFFFFFFu - (unsigned)(cand[e] & 0xFFFFFFFFull);
    if (t == 0) *nselrow = np;
}



// ---------------------------------------------------------------- fast selection (step 3)
// A: per row, a threshold from a strided sample (every 16th visible bucket): the 96th best sample score (about
//    1.5 p of the row above it); rows with <= 4,096 visible buckets are selected exactly right here.
// B: one tensor-core scan of all visible centroids (64 rows x 64 centroids per tile, spans of tiles per block) that
//    lists the buckets scoring >= threshold - margin (packed order key, never the score matrix).
// C: per row, the list sorted (bitonic, <= 4,096), the top p best first; a row whose list holds < p or overflowed
//    goes to the failure list.
// D: failing rows (compacted, all of them, kLsaFail at a time) take the exact path (scores into scratch + radix select).
constexpr int kLsaSampleStride = 17, kLsaSampleRank = 96, kLsaListCap = 4096, kLsaExact = 4096, kLsaFail = 64;
constexpr int kLsaFailCap = kChunk;  // the failure list holds every row: the fallback runs over it in groups of kLsaFail
constexpr float kLsaMargin = 1e-4f;
constexpr int kLsaSampRows = 16;  // up to this many rows the sample runs over 16 blocks per row

__device__ __forceinline__ float dot64(const bf16* __restrict__ a, const bf16* __restrict__ b) {
    float r = 0.f;
    const uint4* x = reinterpret_cast<const uint4*>(a);
    const uint4* y = reinterpret_cast<const uint4*>(b);
    for (int c = 0; c < 8; c++) {
        const uint4 u = x[c], v = y[c];
        const bf162* p = reinterpret_cast<const bf162*>(&u);
        const bf162* q = reinterpret_cast<const bf162*>(&v);
        for (int e = 0; e < 4; e++) {
            const float2 f = __bfloat1622float2(p[e]), g = __bfloat1622float2(q[e]);
            r += f.x * g.x + f.y * g.y;
        }
    }
    return r;
}

// A: grid m, block 512. Exact rows (vis <= 4,096) write sel / nsel and threshold +inf (no list); others write theta.
// sample order keys of row i (rows with > 4,096 visible buckets), grid (m, 16), block 256: samp[i][e] for the
// same strided sample as lsa_thresh_kernel (decode: 16 blocks per row instead of one)
__global__ void __launch_bounds__(256) lsa_sample_kernel(const bf16* __restrict__ iq, const int* pos, const int* closed,
                                                          const bf16* __restrict__ cent, unsigned* __restrict__ samp) {
    const int i = blockIdx.x, p = pos[0] + i, vis = lsa_vis(p, *closed);
    if (vis <= kLsaExact) return;
    const int stride = max(kLsaSampleStride, (vis + 2 * kLsaExact - 1) / (2 * kLsaExact)) | 1;  // odd: coprime with the buckets per block
    const int ns = (vis + stride - 1) / stride;
    __shared__ __align__(16) bf16 qs[kLsaDim];
    if (threadIdx.x < kLsaDim) qs[threadIdx.x] = iq[(size_t)i * kLsaDim + threadIdx.x];
    __syncthreads();
    for (int e = blockIdx.y * 256 + threadIdx.x; e < ns; e += gridDim.y * 256)
        samp[(size_t)i * 2 * kLsaExact + e] = ord_key(dot64(qs, cent + (size_t)e * stride * kLsaDim));
}
__global__ void __launch_bounds__(512) lsa_thresh_kernel(const bf16* __restrict__ iq, const int* pos, const int* closed,
                                                          const bf16* __restrict__ cent, float* __restrict__ theta,
                                                          unsigned long long* __restrict__ lists,
                                                          int* __restrict__ lcount, const unsigned* __restrict__ samp) {
    __shared__ unsigned long long a[kLsaExact];
    __shared__ __align__(16) bf16 qs[kLsaDim];
    const int i = blockIdx.x, t = threadIdx.x;
    const int p = pos[0] + i, vis = lsa_vis(p, *closed);
    if (t < kLsaDim) qs[t] = iq[(size_t)i * kLsaDim + t];
    if (t == 0 && vis > kLsaExact) lcount[i] = 0;
    __syncthreads();
    if (vis <= kLsaExact) {  // every visible bucket is a candidate (exact)
        for (int b = t; b < vis; b += 512)
            lists[(size_t)i * kLsaListCap + b] = ((unsigned long long)ord_key(dot64(qs, cent + (size_t)b * kLsaDim)) << 32) |
                                                 (0xFFFFFFFFu - (unsigned)b);
        if (t == 0) {
            lcount[i] = vis;
            theta[i] = FLT_MAX;  // no scan
        }
        return;
    }
    // the kSampleRank-th best sample score by radix select (4 passes of 8 bits over the order keys in smem)
    // stride 17 (odd) up to 8,192 samples (1M), wider beyond with a rank that keeps ~1.5 p expected above the threshold
    const int stride = max(kLsaSampleStride, (vis + 2 * kLsaExact - 1) / (2 * kLsaExact)) | 1;  // odd: coprime with the buckets per block
    const int ns = (vis + stride - 1) / stride;
    const int rank = max(8, kLsaSampleRank * kLsaSampleStride / stride * LSA_P / kLsaP);
    unsigned* sk = reinterpret_cast<unsigned*>(a);  // [<= 8,192] order keys of the sample
    if (samp)
        for (int e = t; e < ns; e += 512) sk[e] = samp[(size_t)i * 2 * kLsaExact + e];
    else
        for (int e = t; e < ns; e += 512) sk[e] = ord_key(dot64(qs, cent + (size_t)e * stride * kLsaDim));
    __shared__ unsigned hist[256];
    __shared__ unsigned s_prefix, s_need;
    unsigned prefix = 0, need = min(rank, ns);
    __syncthreads();
    for (int pass = 0; pass < 4; pass++) {
        const int sh = 24 - 8 * pass;
        if (t < 256) hist[t] = 0;
        __syncthreads();
        const unsigned hm = pass ? (0xFFFFFFFFu << (32 - 8 * pass)) : 0u;
        for (int e = t; e < ns; e += 512) {
            const unsigned k = sk[e];
            if ((k & hm) == (prefix & hm)) atomicAdd(&hist[(k >> sh) & 255], 1u);
        }
        __syncthreads();
        if (t < 32) {
            int bin;
            unsigned c, wdummy;
            bool f;
            warp_bin_find(hist, nullptr, 0u, 0u, need, 0xFFFFFFFFu, bin, c, wdummy, f);
            if (t == 0) {
                s_prefix = prefix | ((unsigned)bin << sh);
                s_need = need - c;
            }
        }
        __syncthreads();
        prefix = s_prefix;
        need = s_need;
    }
    if (t == 0) {
        const unsigned u = (prefix & 0x80000000u) ? (prefix & 0x7FFFFFFFu) : ~prefix;
        theta[i] = __uint_as_float(u) - kLsaMargin;
    }
}

// B: grid (ceil(m / 64), spans), block 128 (4 warps x 16 rows): bf16 MMA scores of 64 rows x 64-centroid tiles of the
// block's span, buckets >= theta (visible) appended to the row lists
__global__ void __launch_bounds__(128) lsa_scan_kernel(const bf16* __restrict__ iq, int m, const int* pos,
                                                        const int* closed, const bf16* __restrict__ cent, int span_tiles,
                                                        const float* __restrict__ theta,
                                                        unsigned long long* __restrict__ lists, int* __restrict__ lcount) {
    __shared__ __align__(16) bf16 cs[64][kLsaDim + 8];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t4 = lane & 3;
    const int r0 = blockIdx.x * 64 + warp * 16;
    const int p0 = pos[0], cl = *closed;
    const int rlast = min(m, blockIdx.x * 64 + 64) - 1;
    const int vmax = lsa_vis(p0 + rlast, cl);
    const int b_beg = blockIdx.y * span_tiles * 64;
    if (b_beg >= vmax) return;
    const int b_end = min(vmax, b_beg + span_tiles * 64);
    // A fragments: 16 rows x 64 dims (4 k16 slices)
    uint32_t qa[4][4];
    int rr[2], vis[2];
    float th[2];
    for (int x = 0; x < 2; x++) {
        rr[x] = r0 + g + 8 * x;
        vis[x] = rr[x] < m ? lsa_vis(p0 + rr[x], cl) : 0;
        th[x] = rr[x] < m ? theta[rr[x]] : FLT_MAX;
    }
    for (int sl = 0; sl < 4; sl++)
        for (int x = 0; x < 2; x++) {
            uint32_t lo = 0, hi = 0;
            if (rr[x] < m) {
                const bf16* q = iq + (size_t)rr[x] * kLsaDim + 16 * sl + 2 * t4;
                lo = *reinterpret_cast<const uint32_t*>(q);
                hi = *reinterpret_cast<const uint32_t*>(q + 8);
            }
            qa[sl][x] = lo;
            qa[sl][2 + x] = hi;
        }
    for (int b0 = b_beg; b0 < b_end; b0 += 64) {
        __syncthreads();
        for (int c = threadIdx.x; c < 64 * 8; c += 128) {
            const int row = c >> 3, ch = c & 7;
            uint4 v = make_uint4(0, 0, 0, 0);
            if (b0 + row < b_end) v = reinterpret_cast<const uint4*>(cent + (size_t)(b0 + row) * kLsaDim)[ch];
            *reinterpret_cast<uint4*>(&cs[row][ch * 8]) = v;
        }
        __syncthreads();
        for (int nb = 0; nb < 8; nb++) {
            float c[4] = {0.f, 0.f, 0.f, 0.f};
            for (int sl = 0; sl < 4; sl++) {
                const uint32_t b0r = *reinterpret_cast<const uint32_t*>(&cs[nb * 8 + g][16 * sl + 2 * t4]);
                const uint32_t b1r = *reinterpret_cast<const uint32_t*>(&cs[nb * 8 + g][16 * sl + 2 * t4 + 8]);
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                             : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                             : "r"(qa[sl][0]), "r"(qa[sl][1]), "r"(qa[sl][2]), "r"(qa[sl][3]), "r"(b0r), "r"(b1r));
            }
            for (int e = 0; e < 4; e++) {
                const int x = e >> 1, b = b0 + nb * 8 + 2 * t4 + (e & 1);
                if (b < vis[x] && c[e] >= th[x]) {
                    const int o = atomicAdd(&lcount[rr[x]], 1);
                    if (o < kLsaListCap)
                        lists[(size_t)rr[x] * kLsaListCap + o] = ((unsigned long long)ord_key(c[e]) << 32) |
                                                                 (0xFFFFFFFFu - (unsigned)b);
                }
            }
        }
    }
}

// C: the cut, grid rows (or one failing row of the fallback kernels), block 1024: candidates (order key desc = score desc, lower id first)
// taken best first until p buckets or k eligible keys (>= sinks, < window start) -- found by an 8-pass weighted radix
// select over the 64-bit keys (count and key sum per bin), no sort; the cut bucket gives its first keys in position
// order. Writes the pools with their key masks (any order). A sampled list that ends before the cut while the row
// has more visible buckets, or overflowed, goes to the failure list.
__device__ __forceinline__ int lsa_bucket_masks(unsigned mem, int blk, int ws, int mk[4]) {
    int keys = 0;
    for (int k = 0; k < 4; k++) {
        const unsigned pid = (mem >> (8 * k)) & 0xFF;
        mk[k] = 0;
        if (pid == 0xFF) continue;
        const int key0 = (blk * kLsaPools + (int)pid) * kLsaPool;
        for (int r = 0; r < 4; r++)
            if (key0 + r >= kLsaSink && key0 + r < ws) mk[k] |= 1 << r;
        keys += __popc(mk[k]);
    }
    return keys;
}
__device__ void lsa_cut_row(int i, const unsigned long long* __restrict__ lists, const int* __restrict__ lcount,
                            const int* pos, const int* closed, const LsaLayer& L, int* __restrict__ fail,
                            unsigned* __restrict__ list, int* __restrict__ nlist, unsigned* __restrict__ stats,
                            int force) {
    __shared__ unsigned long long a[kLsaListCap];
    __shared__ unsigned char w8[kLsaListCap];
    __shared__ unsigned hc[256], hw[256];
    __shared__ unsigned long long s_pre;
    __shared__ unsigned s_cb, s_wb, s_out;
    __shared__ int s_hit;
    const int t = threadIdx.x;
    const int p = pos[0] + i, cl = *closed, vis = lsa_vis(p, cl), ws = lsa_ws(p, cl);
    const int c = lcount[i];  // (the fallback's exact lists hold at most kLsaListCap)
    if (c > kLsaListCap) {  // the sampled list overflowed: exact path (block-uniform; every row fits the failure list,
        if (t == 0) {       // kLsaFailCap = kChunk >= m)
            fail[1 + atomicAdd(fail, 1)] = i;
            if (stats) atomicAdd(stats + (gridDim.x <= kLsaSampRows), 1u);  // [0] prompt chunks, [1] decode-sized
        }
        return;
    }
    for (int e = t; e < c; e += 1024) {
        const unsigned long long k = lists[(size_t)i * kLsaListCap + e];
        a[e] = k;
        const int b = (int)(0xFFFFFFFFu - (unsigned)(k & 0xFFFFFFFFull));
        int mk[4];
        w8[e] = (unsigned char)lsa_bucket_masks(L.mem[b], b / LSA_BPB, ws, mk);
    }
    if (t == 0) s_pre = 0, s_cb = 0, s_wb = 0, s_hit = 0, s_out = 0;
    __syncthreads();
    unsigned long long pre = 0;
    unsigned cb = 0, wb = 0;
    bool hit = false;
    for (int pass = 0; pass < 8; pass++) {
        const int sh = 56 - 8 * pass;
        if (t < 256) hc[t] = 0, hw[t] = 0;
        __syncthreads();
        const unsigned long long hm = pass ? (~0ull << (64 - 8 * pass)) : 0ull;
        for (int e = t; e < c; e += 1024) {
            const unsigned long long k = a[e];
            if ((k & hm) == (pre & hm)) {
                const int bin = (int)((k >> sh) & 255);
                atomicAdd(&hc[bin], 1u);
                atomicAdd(&hw[bin], (unsigned)w8[e]);
            }
        }
        __syncthreads();
        if (t < 32) {
            int bin;
            unsigned c0, w0;
            bool found;
            warp_bin_find(hc, hw, cb, wb, (unsigned)LSA_P, (unsigned)LSA_K, bin, c0, w0, found);
            if (!found) {  // nothing reaches the cut: every element of the prefix range is above it
                c0 += hc[0];
                w0 += hw[0];
            }
            if (t == 0) {
                if (found) s_pre = pre | ((unsigned long long)bin << sh);
                s_cb = c0;
                s_wb = w0;
                s_hit = found;
            }
        }
        __syncthreads();
        pre = s_pre;
        cb = s_cb;
        wb = s_wb;
        hit = s_hit;
        __syncthreads();
        if (!hit) break;  // the whole list is above the cut
    }
    // a sampled list that ended before the cut while the row has more visible buckets: exact path (hit, c and vis
    // are block-uniform; force: QE_LSA_FORCE_FALLBACK, bench only). The fallback calls this with fail = nullptr.
    if ((!hit || force) && c < vis && fail) {
        if (t == 0) {
            fail[1 + atomicAdd(fail, 1)] = i;
            if (stats) atomicAdd(stats + (gridDim.x <= kLsaSampRows), 1u);  // [0] prompt chunks, [1] decode-sized
        }
        return;
    }
    // take: keys > pre fully; the cut element (== pre) its first min(w, k - wb) keys; nothing below
    for (int e = t; e < c; e += 1024) {
        const unsigned long long k = a[e];
        int room;
        if (!hit || k > pre) room = 99;
        else if (k == pre) room = min((int)w8[e], LSA_K - (int)wb);
        else continue;
        if (room <= 0) continue;
        const int b = (int)(0xFFFFFFFFu - (unsigned)(k & 0xFFFFFFFFull));
        int mk[4];
        const unsigned mem = L.mem[b];
        lsa_bucket_masks(mem, b / LSA_BPB, ws, mk);
        for (int q = 0; q < 4; q++) {
            int kept = 0;
            for (int r = 0; r < 4; r++)
                if ((mk[q] >> r & 1) && room > 0) kept |= 1 << r, room--;
            if (!kept) continue;
            const unsigned o = atomicAdd(&s_out, 1u);
            const unsigned pid = (mem >> (8 * q)) & 0xFF;
            if (o < (unsigned)kLsaListMax)
                list[(size_t)i * kLsaListMax + o] = ((unsigned)((b / LSA_BPB) * kLsaPools + (int)pid) << 4) | (unsigned)kept;
        }
    }
    __syncthreads();
    if (t == 0) nlist[i] = min((int)s_out, kLsaListMax);
    __syncthreads();  // (smem reused by the next row of a persistent caller)
}
__global__ void __launch_bounds__(1024) lsa_cut_kernel(const unsigned long long* __restrict__ lists,
                                                        const int* __restrict__ lcount, const int* pos, const int* closed,
                                                        LsaLayer L, int* __restrict__ fail, unsigned* __restrict__ list,
                                                        int* __restrict__ nlist, unsigned* __restrict__ stats, int force) {
    lsa_cut_row(blockIdx.x, lists, lcount, pos, closed, L, fail, list, nlist, stats, force);
}

// D: the exact top-p list of one failing row from its score row (defined below, used by the fallback kernels)
__device__ void lsa_fail_list_row(const float* row, int V, int i, const int* pos, const int* closed,
                                  unsigned long long* __restrict__ lists, int* __restrict__ lcount);

// Expansion, grid rows, block 1024 (thread = picked bucket, best first): eligible keys (>= sinks, < window start)
// per bucket, scan, cut at k keys (the last bucket in position order); pools with their key masks -> list
__global__ void __launch_bounds__(1024) lsa_expand_kernel(const unsigned* __restrict__ sel, const int* __restrict__ nsel,
                                                           const int* pos, const int* closed, LsaLayer L,
                                                           unsigned* __restrict__ list, int* __restrict__ nlist) {
    __shared__ int sk[1024], sp[1024];
    const int i = blockIdx.x, t = threadIdx.x;
    const int p = pos[0] + i, ws = lsa_ws(p, *closed);
    const int np = nsel[i];
    unsigned mem = 0xFFFFFFFFu;
    int bucket = -1, msk[4] = {0, 0, 0, 0}, keys = 0, pools = 0;
    if (t < np) {
        bucket = sel[(size_t)i * kLsaP + t];
        mem = L.mem[bucket];
        const int blk = bucket / LSA_BPB;
        for (int k = 0; k < 4; k++) {
            const unsigned pid = (mem >> (8 * k)) & 0xFF;
            if (pid == 0xFF) continue;
            const int key0 = (blk * kLsaPools + pid) * kLsaPool;
            int mk = 0;
            for (int r = 0; r < 4; r++)
                if (key0 + r >= kLsaSink && key0 + r < ws) mk |= 1 << r;
            msk[k] = mk;
            keys += __popc(mk);
        }
    }
    sk[t] = keys;
    __syncthreads();
    for (int o = 1; o < 1024; o <<= 1) {  // inclusive scan of keys
        const int v = t >= o ? sk[t - o] : 0;
        __syncthreads();
        sk[t] += v;
        __syncthreads();
    }
    const int before = sk[t] - keys;
    int room = LSA_K - before;  // keys this bucket may still take
    if (room < 0) room = 0;
    if (t < np) {  // cut in position order (pool order inside the bucket, keys in order)
        for (int k = 0; k < 4; k++) {
            int mk = msk[k], kept = 0;
            for (int r = 0; r < 4; r++)
                if (mk >> r & 1) {
                    if (room > 0) kept |= 1 << r, room--;
                }
            msk[k] = kept;
            pools += kept != 0;
        }
    }
    sp[t] = pools;
    __syncthreads();
    for (int o = 1; o < 1024; o <<= 1) {
        const int v = t >= o ? sp[t - o] : 0;
        __syncthreads();
        sp[t] += v;
        __syncthreads();
    }
    int off = sp[t] - pools;
    if (t < np) {
        const int blk = bucket / LSA_BPB;
        for (int k = 0; k < 4; k++)
            if (msk[k] && off < kLsaListMax) {
                const unsigned pid = (mem >> (8 * k)) & 0xFF;
                list[(size_t)i * kLsaListMax + off++] = ((unsigned)(blk * kLsaPools + pid) << 4) | (unsigned)msk[k];
            }
    }
    if (t == 1023) nlist[i] = min(sp[1023], kLsaListMax);
}

// Sparse attention, grid (m, local kv head), block 128 (4 warps): each warp a strided share of the row's keys
// (long-range list, window, sinks; only the keys this card owns), online softmax for the head's q heads, merged
// over the warps; whole heads: gated bf16 output, position-split heads: split-K partial (one split) into work.
constexpr int kLsaMaxH = 6;
__global__ void __launch_bounds__(128) lsa_attn_kernel(const bf16* __restrict__ q, const unsigned char* __restrict__ kc,
                                                        const unsigned char* __restrict__ vc, const AttnShape s,
                                                        const float* __restrict__ proj, int ld, int m, const int* pos,
                                                        const int* closed, const unsigned* __restrict__ list,
                                                        const int* __restrict__ nlist, bf16* __restrict__ out,
                                                        float* __restrict__ work) {
    __shared__ float wm[4][kLsaMaxH], wl[4][kLsaMaxH];
    __shared__ float wacc[4][kLsaMaxH][256];
    const int qi = blockIdx.x, j = blockIdx.y, w = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int qb = s.kv_qb[j], nh = s.kv_qe[j] - s.kv_qb[j];
    const PosMap pm = s.kv_pm[j];
    const int p = pos[0] + qi, ws = lsa_ws(p, *closed);
    const int nl = nlist[qi], nwin = p + 1 - ws, nsink = min(kLsaSink, ws);
    const int nv = nl * 4 + nwin + nsink;
    const size_t hbase = head_rows(s, j, s.max_t), r0 = pm_rank(pm, s.split);
    float qr[kLsaMaxH][8];
    for (int h = 0; h < kLsaMaxH; h++)
        for (int e = 0; e < 8; e++)
            qr[h][e] = h < nh ? __bfloat162float(q[((size_t)qi * s.nq + qb + h) * 256 + lane * 8 + e]) : 0.f;
    const float sc = s.ks * 0.0625f;
    float mx[kLsaMaxH], l[kLsaMaxH], acc[kLsaMaxH][8];
    for (int h = 0; h < kLsaMaxH; h++) {
        mx[h] = -FLT_MAX;
        l[h] = 0.f;
        for (int e = 0; e < 8; e++) acc[h][e] = 0.f;
    }
    for (int v = w; v < nv; v += 4) {
        int key;
        if (v < nl * 4) {
            const unsigned ent = list[(size_t)qi * kLsaListMax + (v >> 2)];
            if (!((ent & 15u) >> (v & 3) & 1u)) continue;
            key = (int)(ent >> 4) * 4 + (v & 3);
        } else if (v < nl * 4 + nwin) {
            key = ws + (v - nl * 4);
        } else {
            key = v - nl * 4 - nwin;
        }
        if (!pm_owned(pm, key)) continue;
        const size_t row = hbase + (pm_rank(pm, key) - r0);
        const uint2 kr = *reinterpret_cast<const uint2*>(kc + row * 256 + lane * 8);
        const uint2 vr = *reinterpret_cast<const uint2*>(vc + row * 256 + lane * 8);
        float kf[8], vf[8];
        for (int e = 0; e < 4; e++) {
            kf[e] = fp8f((kr.x >> (8 * e)) & 255);
            kf[4 + e] = fp8f((kr.y >> (8 * e)) & 255);
            vf[e] = fp8f((vr.x >> (8 * e)) & 255) * s.vs;
            vf[4 + e] = fp8f((vr.y >> (8 * e)) & 255) * s.vs;
        }
        for (int h = 0; h < kLsaMaxH; h++) {
            if (h >= nh) break;
            float d = 0.f;
            for (int e = 0; e < 8; e++) d += qr[h][e] * kf[e];
            for (int o = 16; o; o >>= 1) d += __shfl_xor_sync(0xffffffffu, d, o);
            const float x = d * sc;
            const float nm = fmaxf(mx[h], x);
            const float al = __expf(mx[h] - nm), pe = __expf(x - nm);
            l[h] = l[h] * al + pe;
            for (int e = 0; e < 8; e++) acc[h][e] = acc[h][e] * al + pe * vf[e];
            mx[h] = nm;
        }
    }
    for (int h = 0; h < nh; h++) {
        if (lane == 0) wm[w][h] = mx[h], wl[w][h] = l[h];
        for (int e = 0; e < 8; e++) wacc[w][h][lane * 8 + e] = acc[h][e];
    }
    __syncthreads();
    const bool xch = s.kv_x[j] == 3;
    const int no = s.no ? s.no : s.nq;
    for (int e = threadIdx.x; e < nh * 256; e += 128) {
        const int h = e / 256, d = e % 256;
        float M = -FLT_MAX;
        for (int x = 0; x < 4; x++) M = fmaxf(M, wm[x][h]);
        float num = 0.f, den = 0.f;
        for (int x = 0; x < 4; x++) {
            if (wl[x][h] == 0.f) continue;
            const float f = __expf(wm[x][h] - M);
            num += f * wacc[x][h][d];
            den += f * wl[x][h];
        }
        const int hq = qb + h;
        if (xch) {
            work[((size_t)qi * s.nq + hq) * 256 + d] = num;
            if (d == 0) {
                float* ml = work + (size_t)m * s.nq * 256 + ((size_t)qi * s.nq + hq) * 2;
                ml[0] = den == 0.f ? -FLT_MAX : M;
                ml[1] = den;
            }
        } else if (hq >= s.ob && hq < s.ob + no) {
            const float gate = proj[(size_t)qi * ld + hq * 512 + 256 + d];
            out[((size_t)qi * no + (hq - s.ob)) * 256 + d] = __float2bfloat16(num / den / (1.f + __expf(-gate)));
        }
    }
}


// ---------------------------------------------------------------- fast prefill / decode attention (step 3)
// Union of the lists of 8 consecutive rows per (tile, local kv head), in 4-key pools of this card's keys: a bitmap over
// the pools [0, p_last / 4], set by the rows' long-range pools, the window range and the sink pool; then the union
// in position order with a mask per pool (row i: bits 4 i .. 4 i + 3 = its keys of the pool).
template <int NT>
__global__ void __launch_bounds__(NT) lsa_union_kernel(const unsigned* __restrict__ list, const int* __restrict__ nlist,
                                                        int m, const int* pos, const int* closed, const AttnShape s,
                                                        unsigned* __restrict__ upool, unsigned* __restrict__ umask,
                                                        int* __restrict__ ucount) {
    extern __shared__ unsigned usm[];
    __shared__ int s_off[NT + 1];
    const int tile = blockIdx.x, j = blockIdx.y, t = threadIdx.x;
    const PosMap pm = s.kv_pm[j];
    const int i0 = tile * kLsaTileRows, nr = min(kLsaTileRows, m - i0);
    const int p0 = pos[0] + i0, cl = *closed;
    const int plast = p0 + nr - 1, W = (plast / 4 + 32) / 32;
    unsigned* bm = usm;          // [W] bitmap
    unsigned* mk = usm + W;      // [kLsaUnionMax] masks
    for (int w = t; w < W; w += NT) bm[w] = 0;
    __syncthreads();
    const int ws0 = lsa_ws(p0, cl);
    for (int r = 0; r < nr; r++) {
        const unsigned* L = list + (size_t)(i0 + r) * kLsaListMax;
        const int n = nlist[i0 + r];
        for (int e = t; e < n; e += NT) {
            const int P = (int)(L[e] >> 4);
            if (pm_owned(pm, 4 * P)) atomicOr(&bm[P >> 5], 1u << (P & 31));
        }
    }
    for (int P = ws0 / 4 + t; P <= plast / 4; P += NT)
        if (pm_owned(pm, 4 * P)) atomicOr(&bm[P >> 5], 1u << (P & 31));
    if (t == 0 && ws0 > 0 && pm_owned(pm, 0)) atomicOr(&bm[0], 1u);  // sinks (pool 0)
    __syncthreads();
    // chunk of words per thread, exclusive scan of the chunk counts
    const int per = (W + NT - 1) / NT, w0 = min(W, t * per), w1 = min(W, w0 + per);
    int c = 0;
    for (int w = w0; w < w1; w++) c += __popc(bm[w]);
    {   // block exclusive scan of the chunk counts (warp shuffles, then the warp totals)
        __shared__ int s_wt[NT / 32];
        const int lane = t & 31, wid = t >> 5;
        int v = c;
        for (int o = 1; o < 32; o <<= 1) {
            const int u = __shfl_up_sync(0xffffffffu, v, o);
            if (lane >= o) v += u;
        }
        if (lane == 31) s_wt[wid] = v;
        __syncthreads();
        if (wid == 0) {
            int w = lane < NT / 32 ? s_wt[lane] : 0;
            for (int o = 1; o < 32; o <<= 1) {
                const int u = __shfl_up_sync(0xffffffffu, w, o);
                if (lane >= o) w += u;
            }
            if (lane < NT / 32) s_wt[lane] = w;  // inclusive warp totals
        }
        __syncthreads();
        s_off[t + 1] = v + (wid ? s_wt[wid - 1] : 0);
        if (t == 0) s_off[0] = 0;
    }
    __syncthreads();
    const int U = min(s_off[NT], kLsaUnionMax);
    unsigned* up = upool + ((size_t)tile * s.nkv + j) * kLsaUnionMax;
    {   // the local cache row of each pool's first key (the 4 keys of a pool are owned together: shares are multiples of 32)
        const size_t hb = head_rows(s, j, s.max_t);
        const int r0l = pm_rank(pm, s.split);
        int o = s_off[t];
        for (int w = w0; w < w1; w++)
            for (unsigned b = bm[w]; b; b &= b - 1) {
                const int P = w * 32 + __ffs(b) - 1;
                if (o < kLsaUnionMax) up[o] = (unsigned)(hb + (pm_rank(pm, 4 * P) - r0l));
                o++;
            }
    }
    for (int x = t; x < U; x += NT) mk[x] = 0;
    __syncthreads();
    auto rank = [&](int P) {  // union index of a set pool
        const int w = P >> 5, ch = min(NT - 1, w / per);
        int o = s_off[ch];
        for (int x = ch * per; x < w; x++) o += __popc(bm[x]);
        return o + __popc(bm[w] & ((1u << (P & 31)) - 1u));
    };
    for (int r = 0; r < nr; r++) {
        const unsigned* L = list + (size_t)(i0 + r) * kLsaListMax;
        const int n = nlist[i0 + r];
        for (int e = t; e < n; e += NT) {
            const int P = (int)(L[e] >> 4);
            if (!pm_owned(pm, 4 * P)) continue;
            const int x = rank(P);
            if (x < U) atomicOr(&mk[x], (L[e] & 15u) << (4 * r));
        }
        const int p = p0 + r, ws = lsa_ws(p, cl);
        for (int P = ws0 / 4 + t; P <= plast / 4; P += NT) {
            if (!pm_owned(pm, 4 * P)) continue;
            unsigned bits = 0;
            for (int k = 0; k < 4; k++)
                if (4 * P + k >= ws && 4 * P + k <= p) bits |= 1u << k;
            if (bits) {
                const int x = rank(P);
                if (x < U) atomicOr(&mk[x], bits << (4 * r));
            }
        }
        if (t == 0 && ws > 0 && pm_owned(pm, 0)) {
            const unsigned bits = (1u << min(kLsaSink, ws)) - 1u;
            atomicOr(&mk[0], bits << (4 * r));
        }
    }
    __syncthreads();
    unsigned* um = umask + ((size_t)tile * s.nkv + j) * kLsaUnionMax;
    for (int x = t; x < U; x += NT) um[x] = mk[x];
    if (t == 0) ucount[tile * s.nkv + j] = U;
}

__device__ __forceinline__ void mma16816(float c[4], const uint32_t a[4], uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
// four e4m3 bytes b0..b3 -> e = (b0, b2), o = (b1, b3) as fp16, value * 2^-8 (subnormals included)
__device__ __forceinline__ void f8x4_h(uint32_t q, uint32_t& e, uint32_t& o) {
    o = (q & 0x80008000u) | ((q & 0x7F007F00u) >> 1);
    const uint32_t sh = q << 8;
    e = (sh & 0x80008000u) | ((sh & 0x7F007F00u) >> 1);
}
__device__ __forceinline__ void cp16(void* smem, const void* g, bool pred) {
    const unsigned sa = (unsigned)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(sa), "l"(g), "r"(pred ? 16 : 0) : "memory");
}
__device__ __forceinline__ void ldsm4t(uint32_t r[4], unsigned a) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}
__device__ __forceinline__ uint32_t pack_h2(float a, float b) {
    const __half2 h = __floats2half2_rn(a, b);
    return *reinterpret_cast<const uint32_t*>(&h);
}

// Attention over a tile's union, grid (tiles x S, local kv head), block 32 x ceil(8 nh / 16) (one warp per 16 MMA
// rows = (query, q head) pairs). 32-key tiles (8 pools) gathered by cp.async (double buffer): K fp8 rows padded to 272
// B (conflict-free u32 reads straight into permuted-dim fp16 B fragments), V fp8 widened to fp16 for ldmatrix.trans.
// fp32 accumulate, exp2 softmax with the pool masks; writes split-K partials (numerator, max (natural log), sum) for
// every head: attention_prefill_combine merges, gates and runs the position-split exchange.
constexpr int kTK = 32;
// K fp8 rows and V fp16 rows: 16-byte chunks XOR-swizzled by (key & 7) (conflict-free fragment / ldmatrix reads)
constexpr int kTcSmem = 2 * kTK * 256 + 2 * kTK * 256 + kTK * 512 + kTK / 4 * 4;
__device__ __forceinline__ void mma16816h(uint32_t c[2], const uint32_t a[4], uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};"
                 : "+r"(c[0]), "+r"(c[1])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
template <bool QK16>
__global__ void __launch_bounds__(96) lsa_attn_tc_kernel(const bf16* __restrict__ q, const unsigned char* __restrict__ kc,
                                                          const unsigned char* __restrict__ vc, const AttnShape s,
                                                          const unsigned* __restrict__ upool,
                                                          const unsigned* __restrict__ umask,
                                                          const int* __restrict__ ucount, int m, int S,
                                                          float* __restrict__ work) {
    extern __shared__ __align__(16) unsigned char tsm[];
    unsigned char* ks8 = tsm;                               // [2][kTK][256] swizzled
    unsigned char* vs8 = ks8 + 2 * kTK * 256;               // [2][kTK][256]
    unsigned char* vh = vs8 + 2 * kTK * 256;                // [kTK][512 B] fp16, swizzled
    unsigned* tmask = reinterpret_cast<unsigned*>(vh + kTK * 512);  // [kTK / 4]
    const int tile = blockIdx.x / S, sp = blockIdx.x % S, j = blockIdx.y;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t4 = lane & 3;
    const int nt = blockDim.x;
    const int qb = s.kv_qb[j], nh = s.kv_qe[j] - s.kv_qb[j];
    const int i0 = tile * kLsaTileRows, nr = min(kLsaTileRows, m - i0);
    const int U = ucount[tile * s.nkv + j];
    const unsigned* up = upool + ((size_t)tile * s.nkv + j) * kLsaUnionMax;
    const unsigned* um = umask + ((size_t)tile * s.nkv + j) * kLsaUnionMax;
    // pools of this split
    const int per = ((U + S - 1) / S + (kTK / 4) - 1) / (kTK / 4) * (kTK / 4);
    const int u0 = min(U, sp * per), u1 = min(U, u0 + per);
    // this thread's two MMA rows (g, g + 8 of the warp's 16): (query, head)
    int rq[2], rh[2];
    bool rv[2];
    for (int x = 0; x < 2; x++) {
        const int R = warp * 16 + g + 8 * x;
        rq[x] = R / max(nh, 1);
        rh[x] = R % max(nh, 1);
        rv[x] = R < nr * nh && rq[x] < nr;
    }
    // Q fragments, permuted dims (slice sl: k = 2 t4, 2 t4 + 1 -> dims 16 sl + 4 t4, + 2; k + 8 -> + 1, + 3),
    // pre-scaled by 16 * k scale * log2(e) (K comes as value * 2^-8)
    const float qs = 16.f * s.ks * 1.4426950408889634f;
    uint32_t qa[16][4];
    for (int sl = 0; sl < 16; sl++)
        for (int x = 0; x < 2; x++) {
            float v[4] = {0.f, 0.f, 0.f, 0.f};
            if (rv[x]) {
                const bf16* qr = q + ((size_t)(i0 + rq[x]) * s.nq + qb + rh[x]) * 256 + 16 * sl + 4 * t4;
                for (int e = 0; e < 4; e++) v[e] = __bfloat162float(qr[e]) * qs;
            }
            qa[sl][x] = pack_h2(v[0], v[2]);      // a0 / a1: k 2t, 2t+1 -> dims +0, +2
            qa[sl][2 + x] = pack_h2(v[1], v[3]);  // a2 / a3: k 2t+8, 2t+9 -> dims +1, +3
        }
    float o[32][4];
    for (int n = 0; n < 32; n++) o[n][0] = o[n][1] = o[n][2] = o[n][3] = 0.f;
    float mx[2] = {-1e30f, -1e30f}, ls[2] = {0.f, 0.f};
    const int ntile = (u1 - u0 + kTK / 4 - 1) / (kTK / 4);
    auto load = [&](int ti, int st) {
        const int pb = u0 + ti * (kTK / 4);
        for (int c = threadIdx.x; c < kTK * 32; c += nt) {  // 32 keys x (16 K chunks + 16 V chunks)
            const int key = c >> 5, part = c & 31, pi = pb + key / 4;
            const bool ok = pi < u1;
            const size_t row = (size_t)(ok ? up[pi] : 0u) + (key & 3);
            if (part < 16)
                cp16(ks8 + ((size_t)st * kTK + key) * 256 + ((part ^ (key & 7)) * 16), kc + row * 256 + part * 16, ok);
            else
                cp16(vs8 + ((size_t)st * kTK + key) * 256 + (part - 16) * 16, vc + row * 256 + (part - 16) * 16, ok);
        }
        asm volatile("cp.async.commit_group;" ::: "memory");
    };
    if (ntile > 0) load(0, 0);
    for (int ti = 0; ti < ntile; ti++) {
        const int st = ti & 1;
        if (ti + 1 < ntile) {
            load(ti + 1, st ^ 1);
            asm volatile("cp.async.wait_group 1;" ::: "memory");
        } else {
            asm volatile("cp.async.wait_group 0;" ::: "memory");
        }
        __syncthreads();
        {   // widen V to fp16 [key][dim] (values * 2^-8), masks of the tile's pools
            const uint32_t* v32 = reinterpret_cast<const uint32_t*>(vs8 + (size_t)st * kTK * 256);
            for (int c = threadIdx.x; c < kTK * 64; c += nt) {
                const int key = c >> 6, d4 = c & 63;
                uint32_t e, od;
                f8x4_h(v32[key * 64 + d4], e, od);
                uint32_t* dst = reinterpret_cast<uint32_t*>(vh + key * 512 + (((d4 >> 1) ^ (key & 7)) * 16) + (d4 & 1) * 8);
                dst[0] = __byte_perm(e, od, 0x5410);
                dst[1] = __byte_perm(e, od, 0x7632);
            }
            const int pb = u0 + ti * (kTK / 4);
            for (int c = threadIdx.x; c < kTK / 4; c += nt) tmask[c] = pb + c < u1 ? um[pb + c] : 0u;
        }
        __syncthreads();
        // S = Q K^T: 4 n8 blocks of keys
        float sc[4][4];
        const unsigned char* kt = ks8 + (size_t)st * kTK * 256;
        for (int nb = 0; nb < 4; nb++) {
            const unsigned char* kr = kt + (nb * 8 + g) * 256 + 4 * t4;
            if (QK16) {  // fp16 accumulate (as the dense prefill kernel), two halves of the dims summed in fp32
                uint32_t h0[2] = {0u, 0u}, h1[2] = {0u, 0u};
                for (int sl = 0; sl < 16; sl++) {
                    uint32_t e, od;
                    f8x4_h(*reinterpret_cast<const uint32_t*>(kr + ((sl ^ g) * 16)), e, od);
                    mma16816h(sl < 8 ? h0 : h1, qa[sl], e, od);
                }
                const float2 a0 = __half22float2(*reinterpret_cast<__half2*>(&h0[0]));
                const float2 a1 = __half22float2(*reinterpret_cast<__half2*>(&h0[1]));
                const float2 b0 = __half22float2(*reinterpret_cast<__half2*>(&h1[0]));
                const float2 b1 = __half22float2(*reinterpret_cast<__half2*>(&h1[1]));
                sc[nb][0] = a0.x + b0.x;
                sc[nb][1] = a0.y + b0.y;
                sc[nb][2] = a1.x + b1.x;
                sc[nb][3] = a1.y + b1.y;
            } else {
                sc[nb][0] = sc[nb][1] = sc[nb][2] = sc[nb][3] = 0.f;
                for (int sl = 0; sl < 16; sl++) {
                    uint32_t e, od;
                    f8x4_h(*reinterpret_cast<const uint32_t*>(kr + ((sl ^ g) * 16)), e, od);
                    mma16816(sc[nb], qa[sl], e, od);
                }
            }
        }
        // masks, online softmax (log2 domain)
        float tm[2] = {-1e30f, -1e30f};
        for (int nb = 0; nb < 4; nb++)
            for (int e = 0; e < 4; e++) {
                const int x = e >> 1, key = nb * 8 + 2 * t4 + (e & 1);
                const unsigned mk = tmask[key >> 2];
                const bool ok = rv[x] && ((mk >> (4 * rq[x] + (key & 3))) & 1u);
                if (!ok) sc[nb][e] = -1e30f;
                tm[x] = fmaxf(tm[x], sc[nb][e]);
            }
        float al[2];
        for (int x = 0; x < 2; x++) {
            tm[x] = fmaxf(tm[x], __shfl_xor_sync(0xffffffffu, tm[x], 1));
            tm[x] = fmaxf(tm[x], __shfl_xor_sync(0xffffffffu, tm[x], 2));
            const float nm = fmaxf(mx[x], tm[x]);
            al[x] = exp2f(mx[x] - nm);
            mx[x] = nm;
            ls[x] *= al[x];
        }
        uint32_t pa[2][4];
        for (int nb = 0; nb < 4; nb++) {
            float pv[4];
            for (int e = 0; e < 4; e++) {
                const int x = e >> 1;
                pv[e] = sc[nb][e] <= -1e29f ? 0.f : exp2f(sc[nb][e] - mx[x]);
                ls[x] += pv[e];
            }
            // A fragments of P for the k16 slices (keys 0-15, 16-31): n8 block 2u -> a0 / a1, 2u + 1 -> a2 / a3
            const int u = nb >> 1, hi = nb & 1;
            pa[u][hi * 2 + 0] = pack_h2(pv[0], pv[1]);
            pa[u][hi * 2 + 1] = pack_h2(pv[2], pv[3]);
        }
        if (al[0] != 1.f || al[1] != 1.f)
            for (int n = 0; n < 32; n++) {
                o[n][0] *= al[0];
                o[n][1] *= al[0];
                o[n][2] *= al[1];
                o[n][3] *= al[1];
            }
        const unsigned vb = (unsigned)__cvta_generic_to_shared(vh);
        for (int u = 0; u < 2; u++) {
            const uint32_t a[4] = {pa[u][0], pa[u][1], pa[u][2], pa[u][3]};
            const int mrow = lane & 7, mat = lane >> 3;
            const int key = u * 16 + (mat & 1) * 8 + mrow;
            for (int d16 = 0; d16 < 16; d16++) {
                uint32_t b[4];
                ldsm4t(b, vb + (unsigned)(key * 512 + (((d16 * 2 + (mat >> 1)) ^ (key & 7)) * 16)));
                mma16816(o[2 * d16], a, b[0], b[1]);
                mma16816(o[2 * d16 + 1], a, b[2], b[3]);
            }
        }
        __syncthreads();
    }
    // row sums over the 4 threads of a row; partials of this split
    for (int x = 0; x < 2; x++) {
        ls[x] += __shfl_xor_sync(0xffffffffu, ls[x], 1);
        ls[x] += __shfl_xor_sync(0xffffffffu, ls[x], 2);
    }
    const float vsc = 256.f * s.vs;
    for (int x = 0; x < 2; x++) {
        if (!rv[x]) continue;
        const int qi = i0 + rq[x], h = qb + rh[x];
        float* acc = work + (((size_t)sp * m + qi) * s.nq + h) * 256;
        for (int n = 0; n < 32; n++) {
            const int d = n * 8 + 2 * t4;
            acc[d] = o[n][2 * x] * vsc;
            acc[d + 1] = o[n][2 * x + 1] * vsc;
        }
        if (t4 == 0) {
            float* ml = work + (size_t)S * m * s.nq * 256 + (((size_t)sp * m + qi) * s.nq + h) * 2;
            ml[0] = ls[x] > 0.f ? mx[x] * 0.6931471805599453f : -FLT_MAX;
            ml[1] = ls[x];
        }
    }
}
// (row without __restrict__: in the split decode fallback other blocks of the same kernel wrote it, so no
// non-coherent read-only loads)
__device__ void lsa_fail_list_row(const float* row, int V, int i, const int* pos, const int* closed,
                                  unsigned long long* __restrict__ lists, int* __restrict__ lcount) {
    __shared__ unsigned hist[256];
    __shared__ unsigned s_prefix, s_need, s_cnt, s_eq;
    __shared__ int s_fit;
    const int t = threadIdx.x;
    const int vis = min(V, lsa_vis(pos[0] + i, *closed));
    unsigned prefix = 0, need = LSA_P;
    if (vis > LSA_P)
        for (int pass = 0; pass < 4; pass++) {
            const int sh = 24 - 8 * pass;
            if (t < 256) hist[t] = 0;
            __syncthreads();
            const unsigned hm = pass ? (0xFFFFFFFFu << (32 - 8 * pass)) : 0u;
            for (int b = t; b < vis; b += 1024) {
                const unsigned k = ord_key(row[b]);
                if ((k & hm) == (prefix & hm)) atomicAdd(&hist[(k >> sh) & 255], 1u);
            }
            __syncthreads();
            if (t < 32) {
                int bin;
                unsigned c, wdummy;
                bool f;
                warp_bin_find(hist, nullptr, 0u, 0u, need, 0xFFFFFFFFu, bin, c, wdummy, f);
                if (t == 0) {
                    s_prefix = prefix | ((unsigned)bin << sh);
                    s_need = need - c;
                }
            }
            __syncthreads();
            prefix = s_prefix;
            need = s_need;
            __syncthreads();
        }
    if (t == 0) s_cnt = 0, s_eq = 0;
    __syncthreads();
    const bool all = vis <= LSA_P;
    for (int b = t; b < vis; b += 1024) {  // strictly above the threshold key (or every bucket): fewer than p
        const unsigned k = ord_key(row[b]);
        if (all || k > prefix) {
            const unsigned o = atomicAdd(&s_cnt, 1u);
            if (o < (unsigned)kLsaListCap)
                lists[(size_t)i * kLsaListCap + o] = ((unsigned long long)k << 32) | (0xFFFFFFFFu - (unsigned)b);
        } else if (k == prefix) {
            atomicAdd(&s_eq, 1u);
        }
    }
    __syncthreads();
    if (t == 0) s_fit = s_cnt + s_eq <= (unsigned)kLsaListCap;  // (read once here: the tie appends below move s_cnt)
    __syncthreads();
    const bool fit = s_fit;
    if (!all && fit) {  // the threshold key's ties all fit: any order (the cut sorts by the unique (key, id))
        for (int b = t; b < vis; b += 1024)
            if (ord_key(row[b]) == prefix) {
                const unsigned o = atomicAdd(&s_cnt, 1u);
                lists[(size_t)i * kLsaListCap + o] = ((unsigned long long)prefix << 32) | (0xFFFFFFFFu - (unsigned)b);
            }
    }
    __syncthreads();
    if (!all && !fit && t < 32) {  // more ties than the cap: the lowest bucket ids, in order (the same on every card)
        unsigned o = s_cnt;
        for (int b0 = 0; b0 < vis && o < (unsigned)kLsaListCap; b0 += 32) {
            const int b = b0 + t;
            const bool eq = b < vis && ord_key(row[b]) == prefix;
            const unsigned mk = __ballot_sync(0xffffffffu, eq);
            const unsigned at = o + __popc(mk & ((1u << t) - 1u));
            if (eq && at < (unsigned)kLsaListCap)
                lists[(size_t)i * kLsaListCap + at] = ((unsigned long long)prefix << 32) | (0xFFFFFFFFu - (unsigned)b);
            o += __popc(mk);
        }
        if (t == 0) s_cnt = o;
    }
    __syncthreads();
    if (t == 0) lcount[i] = min((int)s_cnt, kLsaListCap);  // top p (+ ties in bucket order) of the whole row: exact
    __syncthreads();
}

// D (exact fallback), one launch for every failing row of the call: kLsaFail persistent blocks, each over the rows
// blockIdx.x, + kLsaFail, ... of the failure list with its own score row: scores of all visible buckets, the exact
// top-p (+ ties) list, the cut. No host sync (graph-safe, and none inside a prefill forward), deterministic.
__global__ void __launch_bounds__(1024) lsa_fallback_kernel(const bf16* __restrict__ iq, const int* pos,
                                                             const int* closed, LsaLayer L, int V,
                                                             const int* __restrict__ fail, float* __restrict__ S,
                                                             unsigned long long* __restrict__ lists,
                                                             int* __restrict__ lcount, unsigned* __restrict__ list,
                                                             int* __restrict__ nlist) {
    const int nf = min(fail[0], kLsaFailCap);
    float* row = S + (size_t)blockIdx.x * V;
    for (int fi = blockIdx.x; fi < nf; fi += gridDim.x) {
        const int i = fail[1 + fi], vis = min(V, lsa_vis(pos[0] + i, *closed));
        for (int b = threadIdx.x; b < vis; b += blockDim.x)
            row[b] = dot64(iq + (size_t)i * kLsaDim, L.cent + (size_t)b * kLsaDim);
        __syncthreads();
        lsa_fail_list_row(row, V, i, pos, closed, lists, lcount);
        lsa_cut_row(i, lists, lcount, pos, closed, L, nullptr, list, nlist, nullptr, 0);
    }
}

// D, decode shape (few rows, m <= kLsaSampRows): kLsaFbSplit blocks score slices of one failing row each (the fused
// kernel's one SM per row reads the whole centroid array through one SM: ~0.9 ms per row at 1M); the last block of a
// row to finish (a ticket counter, no spin, no grid-wide wait) builds the exact list and cuts it. Grid
// kLsaFbSplit x m; blocks of rows past the failure count exit at once. done[]: reset with fail[0] every call.
constexpr int kLsaFbSplit = 8;
__global__ void __launch_bounds__(1024) lsa_fallback_split_kernel(const bf16* __restrict__ iq, const int* pos,
                                                                   const int* closed, LsaLayer L, int V,
                                                                   const int* __restrict__ fail, int* __restrict__ done,
                                                                   float* __restrict__ S,
                                                                   unsigned long long* __restrict__ lists,
                                                                   int* __restrict__ lcount, unsigned* __restrict__ list,
                                                                   int* __restrict__ nlist) {
    __shared__ int s_last;
    const int fi = blockIdx.x / kLsaFbSplit, sl = blockIdx.x % kLsaFbSplit;
    if (fi >= min(fail[0], kLsaFailCap)) return;  // (block-uniform)
    const int i = fail[1 + fi], vis = min(V, lsa_vis(pos[0] + i, *closed));
    float* row = S + (size_t)fi * V;
    const int per = (vis + kLsaFbSplit - 1) / kLsaFbSplit, b0 = sl * per, b1 = min(vis, b0 + per);
    for (int b = b0 + threadIdx.x; b < b1; b += blockDim.x)
        row[b] = dot64(iq + (size_t)i * kLsaDim, L.cent + (size_t)b * kLsaDim);
    __syncthreads();
    if (threadIdx.x == 0) {
        __threadfence();  // release: this block's scores visible before its ticket
        const bool last = atomicAdd(done + fi, 1) == kLsaFbSplit - 1;
        __threadfence();  // acquire, in the thread that read the ticket: the other slices' scores (then the barrier)
        s_last = last;
    }
    __syncthreads();
    if (!s_last) return;
    lsa_fail_list_row(row, V, i, pos, closed, lists, lcount);
    lsa_cut_row(i, lists, lcount, pos, closed, L, nullptr, list, nlist, nullptr, 0);
}
}  // namespace

void lsa_iq_partial(const bf16* q, int nq, int m, const LsaRoles& r, const bf16* wq, float* out, cudaStream_t st) {
    if (m > 0) lsa_iq_partial_kernel<<<m, 256, 0, st>>>(q, nq, r, wq, out);
}

void lsa_ik_partial(const void* kc, const AttnShape& s, float ks, int k0, int k1, const LsaRoles& r, const bf16* wk,
                    float* out, cudaStream_t st) {
    if (k1 > k0) lsa_ik_partial_kernel<<<k1 - k0, 256, 0, st>>>((const unsigned char*)kc, s, ks, k0, r, wk, out);
}

void lsa_exchange(const LsaXchg& x, const float* part, int n, bf16* out, cudaStream_t st) {
    CHECK(n > 0 && n <= kLsaXRows, "lsa exchange rows %d", n);
    lsa_seq_kernel<<<1, 1, 0, st>>>(x.seq);
    lsa_send_kernel<<<(n + 15) / 16, 256, 0, st>>>(x, part, n, x.done);
    lsa_wait_kernel<<<1, 32, 0, st>>>(x);
    lsa_sum_kernel<<<n, kLsaDim, 0, st>>>(x, out);
}

void lsa_cluster(const bf16* ik, int b0, int b1, const LsaLayer& L, cudaStream_t st) {
    if (b1 <= b0) return;
    static bool attr[16] = {};
    int dev = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    if (!attr[dev]) {
        CUDA_CHECK(cudaFuncSetAttribute(lsa_ward_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kWardSmem));
        attr[dev] = true;
    }
    lsa_ward_kernel<<<b1 - b0, 512, kWardSmem, st>>>(ik, b0, L);
}

// scratch: sel [kChunk][p] u32, nsel [kChunk], theta [kChunk], lcount [kChunk], fail [1 + kLsaFailCap],
// lists [kChunk][3 p] u64, then S [rows][V] (v1 path / failures: rows >= kLsaFail)
size_t lsa_select_scratch_bytes(int max_pos, int rows, int bpb) {
    const size_t V = (size_t)max_pos / kLsaBlk * bpb;  // >= every vmax of the mode (c_.max_pos / kLsaBlk * LSA_BPB)
    return (size_t)kChunk * (kLsaP * 4 + 12) + ((1 + kLsaFailCap) * 4 + 63) / 64 * 64 + 64 + (size_t)kChunk * kLsaListCap * 8 +
           (size_t)std::max(rows, kLsaFail) * V * 4 + (size_t)kLsaSampRows * 2 * kLsaExact * 4;
}

void lsa_select(const bf16* iq, int m, int vmax, const int* pos, const int* closed, const LsaLayer& L, void* scr,
                size_t scr_bytes, unsigned* list, int* n, cudaStream_t st, unsigned* stats) {
    CHECK(m <= kChunk, "lsa select rows %d", m);
    char* c = (char*)scr;
    unsigned* sel = (unsigned*)c;
    c += (size_t)kChunk * kLsaP * 4;
    int* nsel = (int*)c;
    c += (size_t)kChunk * 4;
    float* theta = (float*)c;
    c += (size_t)kChunk * 4;
    int* lcount = (int*)c;
    c += (size_t)kChunk * 4;
    int* fail = (int*)c;
    c += ((1 + kLsaFailCap) * 4 + 63) / 64 * 64 + 64;
    unsigned long long* lists = (unsigned long long*)c;
    c += (size_t)kChunk * kLsaListCap * 8;
    float* S = (float*)c;
    const int V = std::max(LSA_BPB, vmax);
    static const int fast = env_int("QE_LSA_FASTSEL", 1);
    if (fast) {
        int* done = (int*)((char*)fail + ((1 + kLsaFailCap) * 4 + 63) / 64 * 64);  // [kLsaSampRows] split tickets
        static_assert(kLsaSampRows * 4 <= 64, "lsa select: split tickets in the fail slack");
        CUDA_CHECK(cudaMemsetAsync(fail, 0, 4, st));
        if (m <= kLsaSampRows) CUDA_CHECK(cudaMemsetAsync(done, 0, kLsaSampRows * 4, st));  // (split fallback only)
        unsigned* samp = nullptr;
        if (m <= kLsaSampRows) {  // few rows (decode): the sample over 16 blocks per row
            samp = (unsigned*)((char*)scr + scr_bytes - (size_t)kLsaSampRows * 2 * kLsaExact * 4);
            lsa_sample_kernel<<<dim3(m, 16), 256, 0, st>>>(iq, pos, closed, L.cent, samp);
        }
        lsa_thresh_kernel<<<m, 512, 0, st>>>(iq, pos, closed, L.cent, theta, lists, lcount, samp);
        const int rb = (m + 63) / 64, tiles = (V + 63) / 64;
        const int spans = std::max(1, std::min(tiles, ((m <= kLsaSampRows ? 8 : 4) * 82 + rb - 1) / rb));
        const int span_tiles = (tiles + spans - 1) / spans;
        lsa_scan_kernel<<<dim3(rb, spans), 128, 0, st>>>(iq, m, pos, closed, L.cent, span_tiles, theta, lists, lcount);
        static const int force = env_int("QE_LSA_FORCE_FALLBACK", 0);  // bench: every sampled row takes the fallback
        lsa_cut_kernel<<<m, 1024, 0, st>>>(lists, lcount, pos, closed, L, fail, list, n, stats, force);
        // the exact fallback: decode (few rows) splits each failing row's scores over kLsaFbSplit SMs; prefill runs
        // the persistent kernel (many rows in parallel). QE_LSA_FB_SPLIT=0: the persistent kernel for decode too
        static const int split = env_int("QE_LSA_FB_SPLIT", 1);
        if (split && m <= kLsaSampRows)
            lsa_fallback_split_kernel<<<kLsaFbSplit * m, 1024, 0, st>>>(iq, pos, closed, L, V, fail, done, S, lists,
                                                                        lcount, list, n);
        else
            lsa_fallback_kernel<<<kLsaFail, 1024, 0, st>>>(iq, pos, closed, L, V, fail, S, lists, lcount, list, n);
        return;
    } else {
        const int sub = std::max(1, std::min(m, (int)(((char*)scr + scr_bytes - (char*)S) / ((size_t)V * 4))));
        for (int r0 = 0; r0 < m; r0 += sub) {
            const int rows = std::min(sub, m - r0);
            lsa_score_kernel<<<dim3((V + 63) / 64, (rows + 31) / 32), 256, 0, st>>>(iq, r0, rows, pos, closed, L.cent, V, S);
            lsa_topp_kernel<<<rows, 1024, 0, st>>>(S, V, r0, pos, closed, sel + (size_t)r0 * kLsaP, nsel + r0);
        }
    }
    lsa_expand_kernel<<<m, 1024, 0, st>>>(sel, nsel, pos, closed, L, list, n);
}

size_t lsa_union_bytes(int max_rows, int nkv) {
    const size_t tiles = (size_t)(max_rows + kLsaTileRows - 1) / kLsaTileRows;
    return tiles * nkv * ((size_t)kLsaUnionMax * 8 + 4);
}

void lsa_attend(const bf16* q, const void* kc, const void* vc, const AttnShape& s, const float* proj, int ld,
                const int* pos, const int* closed, int m, int max_pos, const unsigned* list, const int* n, void* uscr,
                bf16* out, float* work, size_t work_floats, cudaStream_t st) {
    CHECK(s.fp8 && !s.split, "lsa attention: fp8 cache, own region only");
    for (int j = 0; j < s.nkv; j++) CHECK(s.kv_qe[j] - s.kv_qb[j] <= kLsaMaxH, "lsa attention: %d q heads per kv head", s.kv_qe[j] - s.kv_qb[j]);
    static const int tc = env_int("QE_LSA_TC", 1);
    if (!tc) {
        lsa_attn_kernel<<<dim3(m, s.nkv), 128, 0, st>>>(q, (const unsigned char*)kc, (const unsigned char*)vc, s, proj, ld,
                                                        m, pos, closed, list, n, out, work);
        attention_prefill_combine(s, proj, ld, pos, m, out, st, work, 1);
        return;
    }
    const int tiles = (m + kLsaTileRows - 1) / kLsaTileRows;
    unsigned* upool = (unsigned*)uscr;
    unsigned* umask = upool + (size_t)tiles * s.nkv * kLsaUnionMax;
    int* ucount = (int*)(umask + (size_t)tiles * s.nkv * kLsaUnionMax);
    const int W = lsa_union_words(max_pos);
    const size_t usm = ((size_t)W + kLsaUnionMax) * 4;
    static bool attr[16] = {};  // function attributes are per device
    int dev = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    constexpr int kUnionSmemMax = kLsaUnionSmem + 4096;  // + the static scan offsets <= the 99 KB per block
    if (!attr[dev]) {
        CUDA_CHECK(cudaFuncSetAttribute(lsa_union_kernel<256>, cudaFuncAttributeMaxDynamicSharedMemorySize, kUnionSmemMax));
        CUDA_CHECK(cudaFuncSetAttribute(lsa_union_kernel<1024>, cudaFuncAttributeMaxDynamicSharedMemorySize, kUnionSmemMax - 4096));
        CUDA_CHECK(cudaFuncSetAttribute(lsa_attn_tc_kernel<false>, cudaFuncAttributeMaxDynamicSharedMemorySize, kTcSmem));
        CUDA_CHECK(cudaFuncSetAttribute(lsa_attn_tc_kernel<true>, cudaFuncAttributeMaxDynamicSharedMemorySize, kTcSmem));
        attr[dev] = true;
    }
    CHECK(usm <= kUnionSmemMax - 4096, "lsa union: context %d too long for the smem bitmap", max_pos);
    if (tiles * s.nkv <= 16)  // decode: few tiles, a wider block each
        lsa_union_kernel<1024><<<dim3(tiles, s.nkv), 1024, usm, st>>>(list, n, m, pos, closed, s, upool, umask, ucount);
    else
        lsa_union_kernel<256><<<dim3(tiles, s.nkv), 256, usm, st>>>(list, n, m, pos, closed, s, upool, umask, ucount);
    int nh = 0;
    for (int j = 0; j < s.nkv; j++) nh = std::max(nh, s.kv_qe[j] - s.kv_qb[j]);
    const int warps = (kLsaTileRows * nh + 15) / 16;
    // splits: enough blocks for ~4 per SM, within the work buffer (splits x rows <= its rows)
    int S = std::max(2, (4 * 82) / std::max(1, tiles * s.nkv));
    S = std::min(S, 64);
    while (S > 2 && attention_prefill_work_floats(S * m, s.nq) > work_floats) S--;
    CHECK(attention_prefill_work_floats(S * m, s.nq) <= work_floats, "lsa attention: work buffer for %d rows", m);
    static const int qk16 = env_int("QE_LSA_QK16", 0);  // fp16-acc Q K^T: no faster, 2x the error
    if (qk16)
        lsa_attn_tc_kernel<true><<<dim3(tiles * S, s.nkv), 32 * warps, kTcSmem, st>>>(q, (const unsigned char*)kc,
                                                                                    (const unsigned char*)vc, s, upool,
                                                                                    umask, ucount, m, S, work);
    else
        lsa_attn_tc_kernel<false><<<dim3(tiles * S, s.nkv), 32 * warps, kTcSmem, st>>>(q, (const unsigned char*)kc,
                                                                                     (const unsigned char*)vc, s, upool,
                                                                                     umask, ucount, m, S, work);
    attention_prefill_combine(s, proj, ld, pos, m, out, st, work, S);  // S >= 2: every head merged, gated
}

__global__ void lsa_set_kernel(int* p, int v) { *p = v; }
void lsa_set_int(int* p, int v, cudaStream_t st) { lsa_set_kernel<<<1, 1, 0, st>>>(p, v); }
