// [q27-head4r] verify head: INT4 logits screen the vocabulary, the top-R candidates of each row are rescored exactly
// (bf16 rows in mapped pinned host memory, or the INT8 head's rows on the device), the argmax of the exact values is the
// row's (value, id) pair for argmax_final.
//
// Candidate selection, per row, deterministic (independent of block order):
//   1. histogram of the top byte of an order-preserving 32-bit key over all logits (global atomics: counts only)
//   2. the bin b1 holding the R-th largest key; histogram of the second byte among the keys of bin b1
//   3. the bin b2 holding it there -> threshold t16 = b1:b2 (16 bits of the key); every key with key >> 16 >= t16 is
//      a candidate: at least R of them, a superset of the top-R
//   3b. more than kH4Cap of them (a flat row; review M4): two more levels give the exact key T of the R-th largest;
//      the candidates are every key > T (fewer than R) plus the keys == T in id order up to R (one ordered scan)
//   4. the candidates are compacted (atomics: order varies, the set does not), sorted in one block by (INT4 logit
//      desc, id asc) and the first R are rescored: always the exact top-R by (INT4 logit desc, id asc). Masked
//      logits (-inf, grammar) never become candidates (review H1: the masks are applied before the selection)
// The rescore sums are done in a fixed order per candidate; the final pick is (exact value desc, id asc).
#include <cfloat>
#include <algorithm>
#include <cstdint>

#include "../common.h"
#include "head4r.cuh"
#include "../env.h"

namespace {
constexpr int kH4Chunk = 4096;  // logits per block in the histogram / collect passes (256 threads x 16)

__device__ __forceinline__ unsigned okey(float f) {
    const unsigned u = f == 0.f ? 0u : __float_as_uint(f);  // -0 = +0, as the sort compares them (re-review N2)
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

// level L: hist[row][L][256] of byte 3 - L of the key among the keys whose higher bytes equal the prefix found so far
// (levels 2 / 3 only on rows whose 16-bit threshold took more than the cap)
template <int LVL>
__global__ void __launch_bounds__(256) h4_hist_kernel(const float* __restrict__ logits, int v, H4Ws ws) {
    const int r = blockIdx.x, c = blockIdx.y;
    if (LVL >= 2 && !ws.ovf[r]) return;
    __shared__ unsigned h[256];
    h[threadIdx.x] = 0;
    __syncthreads();
    const float* row = logits + (size_t)r * v;
    const unsigned pfx = LVL ? ws.pfx[r] : 0;
    constexpr int sh = 24 - 8 * LVL;
    const int i0 = c * kH4Chunk;
    for (int i = i0 + threadIdx.x; i < min(v, i0 + kH4Chunk); i += 256) {
        const unsigned k = okey(row[i]);
        if (LVL == 0) atomicAdd(&h[k >> 24], 1u);
        else if ((k >> (sh + 8)) == pfx) atomicAdd(&h[(k >> sh) & 255], 1u);
    }
    __syncthreads();
    if (h[threadIdx.x]) atomicAdd(&ws.hist[(r * 4 + LVL) * 256 + threadIdx.x], h[threadIdx.x]);
}

// one warp per row: the bin holding the R-th largest key (counting from the top), the count above it. Lane l owns
// bins 255 - 8 l .. 248 - 8 l; a suffix scan over the lanes finds the lane, then that lane walks its 8 bins
template <int LVL>
__global__ void h4_find_kernel(H4Ws ws, int R, int cap) {
    const int r = blockIdx.x, lane = threadIdx.x;
    if (LVL >= 2 && !ws.ovf[r]) return;
    const unsigned* h = ws.hist + (r * 4 + LVL) * 256;
    const int need = LVL ? R - ws.above[r] : R;
    int v[8], sum = 0;
#pragma unroll
    for (int j = 0; j < 8; j++) v[j] = (int)h[255 - 8 * lane - j], sum += v[j];
    int incl = sum;  // bins from the top down to this lane's last one
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        const int x = __shfl_up_sync(0xffffffffu, incl, o);
        if (lane >= o) incl += x;
    }
    const int excl = incl - sum;
    const unsigned hit = __ballot_sync(0xffffffffu, incl >= need);
    const int L = hit ? __ffs(hit) - 1 : 31;  // the lane holding the bin (lane 31 ends at bin 0)
    if (lane == L) {
        int acc = excl, b = 255 - 8 * lane, j = 0;
        for (; j < 7; j++, b--) {
            if (acc + v[j] >= need) break;
            acc += v[j];
        }
        if (LVL == 0) {
            ws.pfx[r] = b;
            ws.above[r] = acc;
        } else if (LVL == 1) {
            const unsigned t16 = (ws.pfx[r] << 8) | (unsigned)b;
            const int total = ws.above[r] + acc + v[j];
            ws.t16[r] = t16;
            ws.cnt[r] = 0;
            ws.ovf[r] = total > cap;  // more than the cap share the top 16 bits: refine (levels 2, 3)
            ws.pfx[r] = t16;
            ws.above[r] += acc;
        } else {
            ws.pfx[r] = (ws.pfx[r] << 8) | (unsigned)b;  // level 3: the full key T of the R-th largest
            ws.above[r] += acc;                          // keys > T (fewer than R)
        }
    }
}

__global__ void __launch_bounds__(256) h4_collect_kernel(const float* __restrict__ logits, int v, H4Ws ws, int cap) {
    const int r = blockIdx.x, c = blockIdx.y;
    const float* row = logits + (size_t)r * v;
    const bool ovf = ws.ovf[r] != 0;
    const unsigned t16 = ws.t16[r], T = ws.pfx[r];
    const int i0 = c * kH4Chunk;
    for (int i = i0 + threadIdx.x; i < min(v, i0 + kH4Chunk); i += 256) {
        const float x = row[i];
        const unsigned k = okey(x);
        if (ovf ? k > T : (k >> 16) >= t16) {
            const int p = atomicAdd(&ws.cnt[r], 1);
            if (p < cap) ws.cv[r * cap + p] = x, ws.ci[r * cap + p] = i;
        }
    }
}

// overflow rows only: the keys == T in id order (block-ordered scan) until the row has R candidates
__global__ void __launch_bounds__(256) h4_equal_kernel(const float* __restrict__ logits, int v, H4Ws ws, int R, int cap) {
    const int r = blockIdx.x, t = threadIdx.x, warp = t >> 5, lane = t & 31;
    if (!ws.ovf[r]) return;
    const float* row = logits + (size_t)r * v;
    const unsigned T = ws.pfx[r];
    const int base = ws.above[r];  // = cnt[r]: the keys > T
    const int need = min(R, cap) - base;
    __shared__ int wc[8];
    __shared__ int taken;
    if (t == 0) taken = 0;
    __syncthreads();
    for (int i0 = 0; i0 < v; i0 += 256) {
        const int i = i0 + t;
        const bool eq = i < v && okey(row[i]) == T;
        const unsigned b = __ballot_sync(0xffffffffu, eq);
        if (lane == 0) wc[warp] = __popc(b);
        __syncthreads();
        int pre = taken;
        for (int w = 0; w < warp; w++) pre += wc[w];
        pre += __popc(b & ((1u << lane) - 1));
        if (eq && pre < need) ws.cv[r * cap + base + pre] = row[i], ws.ci[r * cap + base + pre] = i;
        __syncthreads();
        if (t == 0) {
            int s = 0;
            for (int w = 0; w < 8; w++) s += wc[w];
            taken += s;
        }
        __syncthreads();
        if (taken >= need) break;
    }
    if (t == 0) ws.cnt[r] = base + min(taken, need);
}

// one block per row: bitonic sort of the candidates by (value desc, id asc) in shared memory, the first R ids out
__global__ void __launch_bounds__(kH4Cap) h4_sort_kernel(H4Ws ws, int cap, int R, int* __restrict__ cand) {
    const int r = blockIdx.x, t = threadIdx.x;
    __shared__ float sv[kH4Cap];
    __shared__ int si[kH4Cap];
    const int n = min(ws.cnt[r], cap);
    sv[t] = t < n ? ws.cv[r * cap + t] : -INFINITY;
    si[t] = t < n ? ws.ci[r * cap + t] : INT_MAX;
    __syncthreads();
    for (int k = 2; k <= kH4Cap; k <<= 1)
        for (int j = k >> 1; j > 0; j >>= 1) {
            const int o = t ^ j;
            if (o > t) {
                const bool desc = (t & k) == 0;  // descending runs first: the final order is descending
                const bool before = sv[o] > sv[t] || (sv[o] == sv[t] && si[o] < si[t]);  // o should precede t
                if (desc == before) {
                    const float a = sv[t]; sv[t] = sv[o]; sv[o] = a;
                    const int b = si[t]; si[t] = si[o]; si[o] = b;
                }
            }
            __syncthreads();
        }
    if (t < R) {  // -inf: a grammar-masked logit is never a candidate (review H1)
        const bool ok = t < n && sv[t] != -INFINITY;
        cand[r * R + t] = ok ? si[t] : -1;
        ws.cq[r * R + t] = sv[t];
    }
}

// exact rescore, one warp per (row, candidate); bf16 rows (host mapped or device), x bf16 [m][H]
__global__ void __launch_bounds__(256) h4_rescore_bf16_kernel(const bf16* __restrict__ w, const bf16* __restrict__ x, int H,
                                                              const int* __restrict__ cand, int m, int R, float* __restrict__ ex,
                                                              unsigned long long full, int rmin) {
    const int gw = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
    if (gw >= m * R) return;
    const int r = gw / R, j = gw - r * R;
    const int id = j < ((full >> r & 1ull) ? R : rmin) ? cand[gw] : -1;  // rows outside `full`: only the first rmin
    float s = 0.f;
    if (id >= 0) {
        const uint4* wr = reinterpret_cast<const uint4*>(w + (size_t)id * H);
        const uint4* xr = reinterpret_cast<const uint4*>(x + (size_t)r * H);
        for (int i = lane; i < H / 8; i += 32) {
            const uint4 a = wr[i], b = xr[i];
            const __nv_bfloat162* pa = reinterpret_cast<const __nv_bfloat162*>(&a);
            const __nv_bfloat162* pb = reinterpret_cast<const __nv_bfloat162*>(&b);
#pragma unroll
            for (int e = 0; e < 4; e++) {
                const float2 fa = __bfloat1622float2(pa[e]), fb = __bfloat1622float2(pb[e]);
                s = fmaf(fa.x, fb.x, s);
                s = fmaf(fa.y, fb.y, s);
            }
        }
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
    }
    if (lane == 0) ex[gw] = id >= 0 ? s : -INFINITY;
}

// deduplicated rescore (review M6): the rows of a call share most candidates (the verify rows of one sequence, and its
// greedy / sampled neighbours), and every candidate costs a 10 KB read over PCIe. One block sorts the (id, row, slot)
// entries by id; then one warp per distinct id reads its host row once (into registers) and dots it with every row that
// holds the id, in the per-lane order of h4_rescore_bf16_kernel (bit-identical values)
__global__ void __launch_bounds__(1024) h4_dedup_sort_kernel(const int* __restrict__ cand, int m, int R, unsigned long long full,
                                                             int rmin, int P, unsigned long long* __restrict__ keys,
                                                             float* __restrict__ ex) {
    extern __shared__ unsigned long long sk[];
    const int t = threadIdx.x;
    for (int k = t; k < P; k += blockDim.x) {
        unsigned long long key = ~0ull;
        if (k < m * R) {
            const int r = k / R, j = k - r * R;
            const int id = j < ((full >> r & 1ull) ? R : rmin) ? cand[k] : -1;
            if (id >= 0) key = ((unsigned long long)(unsigned)id << 32) | (unsigned)k;
            else ex[k] = -INFINITY;
        }
        sk[k] = key;
    }
    __syncthreads();
    for (int size = 2; size <= P; size <<= 1)
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
            for (int i = t; i < P / 2; i += blockDim.x) {
                const int lo = 2 * i - (i & (stride - 1)), hi = lo + stride;
                const bool asc = (lo & size) == 0;
                const unsigned long long a = sk[lo], b = sk[hi];
                if ((a > b) == asc) sk[lo] = b, sk[hi] = a;
            }
            __syncthreads();
        }
    for (int k = t; k < P; k += blockDim.x) keys[k] = sk[k];
}

template <int NC>
__global__ void __launch_bounds__(256) h4_dedup_rescore_kernel(const bf16* __restrict__ w, const bf16* __restrict__ x, int H,
                                                               const unsigned long long* __restrict__ keys, int P, int R,
                                                               float* __restrict__ ex) {
    const int gw = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
    if (gw >= P) return;
    const unsigned long long key = keys[gw];
    if (key == ~0ull) return;
    const unsigned id = (unsigned)(key >> 32);
    if (gw > 0 && (unsigned)(keys[gw - 1] >> 32) == id) return;  // not the first entry of its id
    uint4 wv[NC];
    const uint4* wr = reinterpret_cast<const uint4*>(w + (size_t)id * H);
#pragma unroll
    for (int c = 0; c < NC; c++) wv[c] = wr[lane + 32 * c];
    for (int e = gw; e < P; e++) {
        const unsigned long long k2 = keys[e];
        if (k2 == ~0ull || (unsigned)(k2 >> 32) != id) break;
        const int idx = (int)(unsigned)k2, r = idx / R;
        const uint4* xr = reinterpret_cast<const uint4*>(x + (size_t)r * H);
        float s = 0.f;
#pragma unroll
        for (int c = 0; c < NC; c++) {
            const uint4 a = wv[c], b = xr[lane + 32 * c];
            const __nv_bfloat162* pa = reinterpret_cast<const __nv_bfloat162*>(&a);
            const __nv_bfloat162* pb = reinterpret_cast<const __nv_bfloat162*>(&b);
#pragma unroll
            for (int q = 0; q < 4; q++) {
                const float2 fa = __bfloat1622float2(pa[q]), fb = __bfloat1622float2(pb[q]);
                s = fmaf(fa.x, fb.x, s);
                s = fmaf(fa.y, fb.y, s);
            }
        }
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
        if (lane == 0) ex[idx] = s;
    }
}

// the row's pair: best exact value among the first `use` candidates, ties -> lowest id; vocab offset of this card
// added. No finite exact value (NaN rows): the best INT4 candidate at -FLT_MAX; no candidate at all (a shard the mask
// empties): -inf, so any other card's pair wins (review M4 / L8: never a silent id 0 with a real value)
__global__ void h4_pick_kernel(const float* __restrict__ ex, const int* __restrict__ cand, int m, int R, int use, int off,
                               float* __restrict__ pairs) {
    const int r = threadIdx.x;
    if (r >= m) return;
    float best = -FLT_MAX;
    int idx = INT_MAX;
    for (int j = 0; j < use; j++) {
        const int id = cand[r * R + j];
        if (id < 0) continue;
        const float v = ex[r * R + j];
        if (v > best || (v == best && id < idx)) best = v, idx = id;
    }
    if (idx == INT_MAX) {
        const int c0 = cand[r * R];
        best = c0 >= 0 ? -FLT_MAX : -INFINITY;
        idx = c0 >= 0 ? c0 : 0;
    }
    pairs[r * 2] = best;
    pairs[r * 2 + 1] = __int_as_float(idx + off);
}
// sampling candidates straight from the exact values ([q27-head4r], review L2): per row its first `rr` candidates
// sorted by (exact value desc, id asc) -> kTopK (value, id + off) pairs, -FLT_MAX past them (topk_local's format)
__global__ void __launch_bounds__(64) h4_topk_kernel(const float* __restrict__ ex, const int* __restrict__ cand, int R,
                                                     unsigned long long full, int rmin, int off, float* __restrict__ pairs) {
    const int r = blockIdx.x, t = threadIdx.x;
    __shared__ float sv[64];
    __shared__ int si[64];
    const int rr = (full >> r & 1ull) ? R : rmin;
    const int id = t < rr ? cand[r * R + t] : -1;
    const float x = id >= 0 ? ex[r * R + t] : -FLT_MAX;
    const bool ok = id >= 0 && x == x && x > -FLT_MAX;
    sv[t] = ok ? x : -FLT_MAX;
    si[t] = ok ? id : INT_MAX;
    __syncthreads();
    for (int k = 2; k <= 64; k <<= 1)
        for (int j = k >> 1; j > 0; j >>= 1) {
            const int o = t ^ j;
            if (o > t) {
                const bool desc = (t & k) == 0;
                const bool before = sv[o] > sv[t] || (sv[o] == sv[t] && si[o] < si[t]);
                if (desc == before) {
                    const float a = sv[t]; sv[t] = sv[o]; sv[o] = a;
                    const int b = si[t]; si[t] = si[o]; si[o] = b;
                }
            }
            __syncthreads();
        }
    pairs[((size_t)r * 64 + t) * 2] = sv[t];
    pairs[((size_t)r * 64 + t) * 2 + 1] = __int_as_float((si[t] == INT_MAX ? 0 : si[t]) + off);
}
// exact values (back = false) or the INT4 ones (back = true) written over the logits at the candidates
__global__ void h4_scatter_kernel(const float* __restrict__ ex, const float* __restrict__ cq, const int* __restrict__ cand, int R,
                                  unsigned long long full, int rmin, bool back, float* __restrict__ logits, int v) {
    const int r = blockIdx.x;
    const int rr = (full >> r & 1ull) ? R : rmin;
    for (int j = threadIdx.x; j < rr; j += blockDim.x) {
        const int id = cand[r * R + j];
        if (id >= 0) logits[(size_t)r * v + id] = back ? cq[r * R + j] : ex[r * R + j];
    }
}
}  // namespace

size_t h4_ws_bytes(int m) {
    return (size_t)m * (4 * 256 * 4 + 5 * 4 + (size_t)kH4Cap * 8 + kH4MaxR * 20) + 256;
}

H4Ws h4_ws(void* base, int m) {
    H4Ws w;
    char* p = (char*)base;
    w.hist = (unsigned*)p; p += (size_t)m * 4 * 256 * 4;
    w.pfx = (unsigned*)p; p += (size_t)m * 4;
    w.t16 = (unsigned*)p; p += (size_t)m * 4;
    w.above = (int*)p; p += (size_t)m * 4;
    w.cnt = (int*)p; p += (size_t)m * 4;
    w.ovf = (int*)p; p += (size_t)m * 4;
    w.cv = (float*)p; p += (size_t)m * kH4Cap * 4;
    w.ci = (int*)p; p += (size_t)m * kH4Cap * 4;
    w.cand = (int*)p; p += (size_t)m * kH4MaxR * 4;
    w.ex = (float*)p; p += (size_t)m * kH4MaxR * 4;
    w.cq = (float*)p; p += (size_t)m * kH4MaxR * 4;
    w.key = (unsigned long long*)(((uintptr_t)p + 7) & ~(uintptr_t)7);
    return w;
}

void h4_select(const float* logits, int v, int m, int R, const H4Ws& ws, cudaStream_t st) {
    CHECK(R >= 1 && R <= kH4MaxR && m <= kH4MaxRows, "head4r R %d m %d", R, m);
    CUDA_CHECK(cudaMemsetAsync(ws.hist, 0, (size_t)m * 4 * 256 * 4, st));
    const dim3 g(m, (v + kH4Chunk - 1) / kH4Chunk);
    h4_hist_kernel<0><<<g, 256, 0, st>>>(logits, v, ws);
    h4_find_kernel<0><<<m, 32, 0, st>>>(ws, R, kH4Cap);
    h4_hist_kernel<1><<<g, 256, 0, st>>>(logits, v, ws);
    h4_find_kernel<1><<<m, 32, 0, st>>>(ws, R, kH4Cap);
    h4_hist_kernel<2><<<g, 256, 0, st>>>(logits, v, ws);  // (levels 2, 3 and the ordered scan: overflow rows only)
    h4_find_kernel<2><<<m, 32, 0, st>>>(ws, R, kH4Cap);
    h4_hist_kernel<3><<<g, 256, 0, st>>>(logits, v, ws);
    h4_find_kernel<3><<<m, 32, 0, st>>>(ws, R, kH4Cap);
    h4_collect_kernel<<<g, 256, 0, st>>>(logits, v, ws, kH4Cap);
    h4_equal_kernel<<<m, 256, 0, st>>>(logits, v, ws, R, kH4Cap);
    h4_sort_kernel<<<m, kH4Cap, 0, st>>>(ws, kH4Cap, R, ws.cand);
}

void h4_rescore_bf16(const bf16* w, const bf16* x, int H, int m, int R, const H4Ws& ws, cudaStream_t st,
                     unsigned long long full, int rmin) {
    static const bool dedup = env_int("QE_HEAD4R_DEDUP", 1) != 0;  // [q27-head4r-dedup] 0: one read per (row, candidate)
    if (dedup && m > 1 && H == 20 * 256 && m * R <= kH4MaxRows * kH4MaxR) {
        int P = 1;
        while (P < m * R) P <<= 1;
        h4_dedup_sort_kernel<<<1, 1024, (size_t)P * 8, st>>>(ws.cand, m, R, full, rmin, P, ws.key, ws.ex);
        h4_dedup_rescore_kernel<20><<<(P * 32 + 255) / 256, 256, 0, st>>>(w, x, H, ws.key, P, R, ws.ex);
        return;
    }
    const int warps = m * R;
    h4_rescore_bf16_kernel<<<(warps * 32 + 255) / 256, 256, 0, st>>>(w, x, H, ws.cand, m, R, ws.ex, full, rmin);
}

void h4_pick(const H4Ws& ws, int m, int R, int use, int off, float* pairs, cudaStream_t st) {
    h4_pick_kernel<<<1, 64, 0, st>>>(ws.ex, ws.cand, m, R, std::min(use, R), off, pairs);
}

void h4_topk(const H4Ws& ws, int m, int R, unsigned long long full, int rmin, int off, float* pairs, cudaStream_t st) {
    CHECK(R == 64, "h4_topk R %d", R);
    h4_topk_kernel<<<m, 64, 0, st>>>(ws.ex, ws.cand, R, full, rmin, off, pairs);
}

void h4_scatter(const H4Ws& ws, int m, int R, float* logits, int v, cudaStream_t st, unsigned long long full, int rmin,
                bool back) {
    h4_scatter_kernel<<<m, 64, 0, st>>>(ws.ex, ws.cq, ws.cand, R, full, rmin, back, logits, v);
}
