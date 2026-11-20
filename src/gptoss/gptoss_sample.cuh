// [srv-gptoss-capi] Sampled decode head of one card, inside the decode graph (Engine::set_sampling). Over the card's
// logit rows [v0, v0 + vn):
//   - the Gumbel-max draw: argmax of l / T + G(seed, pos, id) = an exact sample of softmax(l / T) over the card's
//     tokens; the merge over the cards (merge_res, the larger value wins) is then an exact sample over the vocabulary.
//     It goes to res = {value, id} like the greedy head, so a plain temperature reply needs no host work at all.
//   - for top_k / top_p / min_p on the host (capi_srv.cpp): the card's max M, S = sum exp((l - M) / T) and its 64
//     largest logits with their ids (out = {M, S, n, 0, val[64], id[64] as int bits}).
// Rows >= NTOK (embedding padding) and non-finite logits never count. Two kernels: k_samp_a (blocks of 2048 rows:
// bitonic sort in shared memory, keep 64) and k_samp_b (one block: merge of the blocks' 64s).
// Several rows of a batched step (Engine::step_rows_sampled): row r = blockIdx.y of k_samp_a, blockIdx.x of k_samp_b,
// its own params sp[r], position pos[r], logits lg + r * vn, results blk / out / res of row r. A row with draw = 0
// takes the plain argmax of the full-precision logits (a greedy slot batched with sampled ones).
#pragma once
#include <cuda_runtime.h>
#include <cstdint>

namespace gptoss {

constexpr int SAMP_K = 64, SAMP_ROWS = 2048, SAMP_NTOK = 200019;
constexpr int SAMP_OUT = 4 + 2 * SAMP_K;   // floats of the per-card result
struct SampParam {
    float inv_t;
    unsigned seed_lo, seed_hi;
    int cands;   // 0: the Gumbel draw only (plain temperature); 1: also M, S, the 64 largest and the logits on the host
    int draw;    // 1: Gumbel-max draw; 0: argmax (greedy row)
    // [gptoss-rel2-penalty] presence / frequency penalty of the row's sequence: pen = 1 + its slot (0: none). The slot's
    // penalty row pen[slot][vn] of every card holds, per token, presence * (c > 0) + frequency * c of the reply so far
    // (made on the host, Engine::pen_reset); the token this step feeds (the reply's newest) comes with the step:
    // pen_id / pen_val, stored into the row by the thread that owns the token. A penalised logit is l - pen in fp32,
    // taken before the temperature (OpenAI's definition); the bounds of the two-stage head get the same subtraction
    // (rounding is monotone, so lo - p <= l - p <= hi - p holds bit for bit).
    int pen, pen_id;
    float pen_val;
};
// the penalty of token i (global id gid) of the row with params sp (pen row base: the card's [slots][vn])
__device__ __forceinline__ float samp_pen(const SampParam* sp, float* pen, int vn, int i, int gid, bool store) {
    float* q = pen + (size_t)(sp->pen - 1) * vn + i;
    if (gid == sp->pen_id) {
        if (store) *q = sp->pen_val;
        return sp->pen_val;
    }
    return *q;
}

__device__ __forceinline__ unsigned long long samp_mix(unsigned long long x) {
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}
// Gumbel noise of token `id` at sequence position `pos` (the reply's seed): the same seed and position give the same draw
__device__ __forceinline__ float samp_gumbel(unsigned long long seed, int pos, int id) {
    const unsigned long long x = samp_mix(seed ^ samp_mix(((unsigned long long)(unsigned)pos << 32) | (unsigned)id));
    // 23 bits: u in [2^-24, 1 - 2^-24], exact in float (24 bits rounded 1 - 2^-25 up to 1.0: log(1) = 0 made the noise
    // +inf, and that token won the draw: ~1 % of the sampled steps picked a uniformly random token)
    const float u = ((float)(x >> 41) + 0.5f) * (1.0f / 8388608.0f);
    return -logf(-logf(u));
}

// score of a token in the Gumbel-max draw: one fused multiply-add, the same in the full head (k_samp_a) and in the
// screen's bounds (k_lm_gnoise), so a bound and the exact score are rounded by the same monotone function
// (-DGPTOSS_SCORE_PLAIN: the expression k_samp_a had before round 15, for the check that seeded outputs did not move)
#ifdef GPTOSS_SCORE_PLAIN
__device__ __forceinline__ float samp_score(float x, float it, float g) { return x * it + g; }
#else
__device__ __forceinline__ float samp_score(float x, float it, float g) { return __fmaf_rn(x, it, g); }
#endif

// bitonic sort of n (power of 2) pairs, descending by value, ties by the smaller id first
__device__ __forceinline__ void samp_sort(float* v, int* id, int n) {
    for (int k = 2; k <= n; k <<= 1)
        for (int j = k >> 1; j > 0; j >>= 1) {
            for (int t = threadIdx.x; t < n / 2; t += blockDim.x) {
                const int i = 2 * j * (t / j) + (t % j), l = i + j;
                const bool desc = (i & k) == 0;
                const float a = v[i], b = v[l];
                const bool a_first = a > b || (a == b && id[i] < id[l]);
                if (a_first != desc) {
                    v[i] = b;
                    v[l] = a;
                    const int x = id[i];
                    id[i] = id[l];
                    id[l] = x;
                }
            }
            __syncthreads();
        }
}

// blk: per block {M, S, gbest value, gbest id bits, val[64], id[64]}
__global__ void __launch_bounds__(1024) k_samp_a(const float* lg, int vn, int v0, const SampParam* sp, const int* pos,
                                                 float* blk, float* pen) {
    __shared__ float v[SAMP_ROWS];
    __shared__ int id[SAMP_ROWS];
    __shared__ float rv[32];
    __shared__ int ri[32];
    __shared__ float bmax;
    const int row = blockIdx.y;
    lg += (size_t)row * vn;
    sp += row;
    blk += (size_t)row * gridDim.x * SAMP_OUT;
    const float it = sp->inv_t;
    const bool draw = sp->draw != 0;
    const unsigned long long seed = ((unsigned long long)sp->seed_hi << 32) | sp->seed_lo;
    const int p = pos[row], base = blockIdx.x * SAMP_ROWS;
    float gv = -INFINITY, mx = -INFINITY;
    int gi = 0x7fffffff;
    for (int t = threadIdx.x; t < SAMP_ROWS; t += blockDim.x) {
        const int i = base + t, gid = v0 + i;
        float x = i < vn && gid < SAMP_NTOK ? lg[i] : -INFINITY;
        if (!isfinite(x)) x = -INFINITY;
        if (sp->pen && i < vn && gid < SAMP_NTOK) x -= samp_pen(sp, pen, vn, i, gid, true);   // [gptoss-rel2-penalty] (-inf stays)
        v[t] = x;
        id[t] = gid;
        if (x > -INFINITY) {
            mx = fmaxf(mx, x);
            const float g = draw ? samp_score(x, it, samp_gumbel(seed, p, gid)) : x;
            if (g > gv || (g == gv && gid < gi)) {
                gv = g;
                gi = gid;
            }
        }
    }
    // block max and the block's Gumbel best
    for (int o = 16; o; o >>= 1) {
        mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
        const float ov = __shfl_xor_sync(0xffffffffu, gv, o);
        const int oi = __shfl_xor_sync(0xffffffffu, gi, o);
        if (ov > gv || (ov == gv && oi < gi)) {
            gv = ov;
            gi = oi;
        }
    }
    const int w = threadIdx.x >> 5, nw = blockDim.x >> 5;
    if ((threadIdx.x & 31) == 0) {
        rv[w] = gv;
        ri[w] = gi;
    }
    __shared__ float wm[32];
    if ((threadIdx.x & 31) == 0) wm[w] = mx;
    __syncthreads();
    float* out = blk + (size_t)blockIdx.x * SAMP_OUT;
    if (threadIdx.x == 0) {
        float m = -INFINITY, bv = -INFINITY;
        int bi = 0x7fffffff;
        for (int k = 0; k < nw; k++) {
            m = fmaxf(m, wm[k]);
            if (rv[k] > bv || (rv[k] == bv && ri[k] < bi)) {
                bv = rv[k];
                bi = ri[k];
            }
        }
        bmax = m;
        out[0] = m;
        out[2] = bv;
        out[3] = __int_as_float(bi);
    }
    __syncthreads();
    if (!sp->cands) return;   // (uniform over the block)
    // S = sum exp((l - M) T^-1)
    float s = 0.f;
    if (bmax > -INFINITY)
        for (int t = threadIdx.x; t < SAMP_ROWS; t += blockDim.x)
            if (v[t] > -INFINITY) s += expf((v[t] - bmax) * it);
    for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
    __syncthreads();
    if ((threadIdx.x & 31) == 0) wm[w] = s;
    __syncthreads();
    if (threadIdx.x == 0) {
        float a = 0.f;
        for (int k = 0; k < nw; k++) a += wm[k];
        out[1] = a;
    }
    samp_sort(v, id, SAMP_ROWS);
    for (int t = threadIdx.x; t < SAMP_K; t += blockDim.x) {
        out[4 + t] = v[t];
        out[4 + SAMP_K + t] = __int_as_float(id[t]);
    }
}

// merge of nb blocks (nb * 64 <= 4096): out = {M, S, count of finite candidates, 0, val[64], id[64]}; res = Gumbel best
__global__ void __launch_bounds__(1024) k_samp_b(const float* blk, int nb, const SampParam* sp, float* out, float* res) {
    __shared__ float v[4096];
    __shared__ int id[4096];
    const int row = blockIdx.x;
    blk += (size_t)row * nb * SAMP_OUT;
    sp += row;
    out += (size_t)row * SAMP_OUT;
    res += 2 * row;
    if (!sp->cands) {   // the Gumbel best of the blocks only
        if (threadIdx.x == 0) {
            float bv = -INFINITY;
            int bi = 0x7fffffff;
            for (int k = 0; k < nb; k++) {
                const float* b = blk + (size_t)k * SAMP_OUT;
                const int i = __float_as_int(b[3]);
                if (b[2] > bv || (b[2] == bv && i < bi)) {
                    bv = b[2];
                    bi = i;
                }
            }
            res[0] = bv;
            res[1] = __int_as_float(bi);
        }
        return;
    }
    const int n = nb * SAMP_K;
    int N = 64;
    while (N < n) N <<= 1;
    for (int t = threadIdx.x; t < N; t += blockDim.x) {
        if (t < n) {
            const float* b = blk + (size_t)(t / SAMP_K) * SAMP_OUT;
            v[t] = b[4 + t % SAMP_K];
            id[t] = __float_as_int(b[4 + SAMP_K + t % SAMP_K]);
        } else {
            v[t] = -INFINITY;
            id[t] = 0x7fffffff;
        }
    }
    __syncthreads();
    samp_sort(v, id, N);
    if (threadIdx.x == 0) {
        const float it = sp->inv_t;
        float m = -INFINITY, bv = -INFINITY;
        int bi = 0x7fffffff;
        for (int k = 0; k < nb; k++) {
            const float* b = blk + (size_t)k * SAMP_OUT;
            m = fmaxf(m, b[0]);
            const int i = __float_as_int(b[3]);
            if (b[2] > bv || (b[2] == bv && i < bi)) {
                bv = b[2];
                bi = i;
            }
        }
        float s = 0.f;
        if (m > -INFINITY)
            for (int k = 0; k < nb; k++) {
                const float* b = blk + (size_t)k * SAMP_OUT;
                if (b[0] > -INFINITY) s += b[1] * expf((b[0] - m) * it);
            }
        int c = 0;
        while (c < SAMP_K && v[c] > -INFINITY) c++;
        out[0] = m;
        out[1] = s;
        out[2] = (float)c;
        out[3] = 0.f;
        res[0] = bv;
        res[1] = __int_as_float(bi);
    }
    for (int t = threadIdx.x; t < SAMP_K; t += blockDim.x) {
        out[4 + t] = v[t];
        out[4 + SAMP_K + t] = __int_as_float(id[t]);
    }
}

// ---- [gptoss-r15-samp2] certified two-stage head for sampled rows (plain temperature, no candidates)
// The draw is argmax over the tokens of samp_score(l, 1 / T, g(seed, pos, id)). The screen (k_lm_screen*) gives
// lo <= l <= hi per row; samp_score is monotone in l, so with the SAME noise
//     samp_score(lo, it, g) <= samp_score(l, it, g) <= samp_score(hi, it, g)
// holds bit for bit. k_lm_gnoise turns the bounds into score bounds in place (hi) and takes the block maxima of the
// lower score bounds (k_lm_thr then exchanges the largest of all cards); k_lm_rescore_w writes the exact logit (the
// full head's row_dot: the same bits) of every row whose upper score bound reaches that threshold into the card's
// logit row and -inf elsewhere; k_samp_a / k_samp_b then draw over that row exactly as over the full head's. The
// drawn token is the full head's: it is a candidate (its upper bound >= its score >= every lower bound), and every
// other candidate carries its exact score. A greedy row of the step (draw = 0) has it = 1, g = 0. Rows >= NTOK never
// count (hi = -inf), as in k_samp_a. A non-finite head input: NaN bounds qualify nothing, +inf bounds are rescored to
// non-finite logits that k_samp_a drops; either way the cards return the {-inf, 0x7fffffff} sentinel as the full head
// does for a row without finite logits (merge_res then throws on both paths). k_lm_thr's floor of -3e38 lies below
// merge_res's -1e37 bar, so it cannot hide a legal winner. Needs gptoss_kernels.cuh (bf16, row_dot) before this file.
// A -DGPTOSS_NOWAIT build (memcheck) reads the peers' threshold words without waiting, as the greedy two-stage head
// does: a stale threshold may leave no candidate (GPTOSS_SAMP=full avoids the exchange).
__global__ void __launch_bounds__(1024) k_lm_gnoise(const float* lo, float* hi, int N, int v0, const SampParam* sp,
                                                    const int* pos, float* lomax, float* pen) {
    __shared__ float bm[32];
    const int row = blockIdx.y;
    lo += (size_t)row * N;
    hi += (size_t)row * N;
    sp += row;
    const float it = sp->inv_t;
    const bool draw = sp->draw != 0;
    const unsigned long long seed = ((unsigned long long)sp->seed_hi << 32) | sp->seed_lo;
    const int p = pos[row];
    float m = -INFINITY;
    const int i = blockIdx.x * 1024 + threadIdx.x, gid = v0 + i;
    if (i < N) {
        if (gid < SAMP_NTOK) {
            const float g = draw ? samp_gumbel(seed, p, gid) : 0.f;
            float l = lo[i], h = hi[i];
            if (sp->pen) {   // [gptoss-rel2-penalty] the bounds of the penalised logit
                const float pv = samp_pen(sp, pen, N, i, gid, true);
                l -= pv;
                h -= pv;
            }
            m = draw ? samp_score(l, it, g) : l;
            hi[i] = draw ? samp_score(h, it, g) : h;
        } else
            hi[i] = -INFINITY;
    }
    for (int o = 16; o; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
    if ((threadIdx.x & 31) == 0) bm[threadIdx.x >> 5] = m;
    __syncthreads();
    if (threadIdx.x == 0) {
        for (int k = 1; k < 32; k++) m = fmaxf(m, bm[k]);
        lomax[(size_t)row * gridDim.x + blockIdx.x] = m;
    }
}
// exact logits of the rows with hi >= thr (32 rows per warp) into the card's logit row, -inf for the others
__global__ void __launch_bounds__(256) k_lm_rescore_w(const bf16* W, const bf16* x, const float* hi, const float* thr, int N,
                                                      float* out) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int rr = blockIdx.y;   // row of the step
    hi += (size_t)rr * N;
    x += (size_t)rr * 2880;
    out += (size_t)rr * N;
    const float T = thr[rr];
    const int r0 = (blockIdx.x * 8 + warp) * 32;
    const bool in = r0 + lane < N;
    const float mine = in ? hi[r0 + lane] : -INFINITY;
    float val = -INFINITY;   // every element is written once, by its own lane (review r15 #1)
    unsigned mask = __ballot_sync(0xffffffffu, in && mine >= T);
    while (mask) {
        const int row = r0 + __ffs(mask) - 1;
        mask &= mask - 1;
        const float v = row_dot(W + (size_t)row * 2880, x, 2880, lane);   // the same bits in every lane (warp_sum)
        if (lane == row - r0) val = v;
    }
    if (in) out[r0 + lane] = val;
}

// the card's logits to mapped host memory, only when the host may need them (top_k / top_p / min_p fallback)
// (a penalised row: the penalised logits, as k_samp_a saw them)
__global__ void k_samp_copy(const float* lg, int vn, const SampParam* sp, float* host, int v0, float* pen) {
    if (!sp->cands) return;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < vn; i += gridDim.x * blockDim.x)
        host[i] = sp->pen ? lg[i] - samp_pen(sp, pen, vn, i, v0 + i, false) : lg[i];
}

}  // namespace gptoss
