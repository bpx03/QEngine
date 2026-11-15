#include <algorithm>
#include <cfloat>

#include "../common.h"
#include "sample.cuh"
#include "sample_rng.h"

static_assert(kMaxBatchRows <= kSampleMaxRows, "sample_rng.h: a row must fit below the step counter's bits");

namespace {

constexpr int kPerThread = 4;     // local candidates kept per thread before the block merge
constexpr int kSelThreads = 1024;
constexpr int kCand = kPerThread * kSelThreads;

// grid m, block 1024: per-thread top-4 over a strided slice, then bitonic sort of the 4096
// candidates in shared memory and the best kTopK written out.
__global__ void __launch_bounds__(kSelThreads) topk_kernel(const float* __restrict__ logits, int v,
                                                           int off, float* __restrict__ pairs) {
    __shared__ float sv[kCand];
    __shared__ int si[kCand];
    const int mi = blockIdx.x, tid = threadIdx.x;
    const float* row = logits + (size_t)mi * v;
    float bv[kPerThread];
    int bi[kPerThread];
#pragma unroll
    for (int j = 0; j < kPerThread; j++) { bv[j] = -FLT_MAX; bi[j] = 0; }
    for (int i = tid; i < v; i += kSelThreads) {
        float x = row[i];
        int xi = i;
        if (x <= bv[kPerThread - 1]) continue;
#pragma unroll
        for (int j = 0; j < kPerThread; j++) {  // insertion into the sorted local list
            if (x > bv[j]) {
                const float tv = bv[j]; const int ti = bi[j];
                bv[j] = x; bi[j] = xi;
                x = tv; xi = ti;
            }
        }
    }
#pragma unroll
    for (int j = 0; j < kPerThread; j++) {
        sv[tid * kPerThread + j] = bv[j];
        si[tid * kPerThread + j] = bi[j];
    }
    __syncthreads();
    // bitonic sort, descending
    for (int size = 2; size <= kCand; size <<= 1) {
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
            for (int i = tid; i < kCand / 2; i += kSelThreads) {
                const int lo = 2 * i - (i & (stride - 1));
                const int hi = lo + stride;
                const bool desc = (lo & size) == 0;
                const float a = sv[lo], b = sv[hi];
                if ((a < b) == desc) {
                    sv[lo] = b; sv[hi] = a;
                    const int t = si[lo]; si[lo] = si[hi]; si[hi] = t;
                }
            }
            __syncthreads();
        }
    }
    if (tid < kTopK) {
        pairs[((size_t)mi * kTopK + tid) * 2] = sv[tid];
        pairs[((size_t)mi * kTopK + tid) * 2 + 1] = __int_as_float(si[tid] + off);
    }
}

__device__ __forceinline__ float uniform(const SampleParams& p, int row, int salt) {
    return sample_uniform(p.seed, p.counter, row, salt);
}

// Merge the per-GPU sorted lists of one row into the global top-K and turn them into
// probabilities (temperature, top-k, min-p, top-p). Returns the candidate count.
// Penalty (pen_active): all kTopK merged candidates get logit -= frequency * c + presence * (c > 0), c = the slot's
// count + occurrences in ext[0..next) (earlier verify rows), re-sorted, then the top-k of them as without it. A token
// outside the merged top-kTopK cannot enter (the penalty only lowers logits; 64 >> top_k of the profiles).
// rs != nullptr (rows of several sequences): only the ext[j] with rs[j] == slot count, the first kTopK of them (read in
// place: no local copy of the slot's rows, which put sample_rows_slots_kernel's frame above the 1024 B default stack)
__device__ int row_probs(const float* gathered, int ngpu, int m, int mi, const SampleParams& p,
                         float* prob, int* tok, const int* ext = nullptr, int next = 0, const int* rs = nullptr,
                         int slot = -1) {
    int head[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    const int K = max(1, min(p.top_k, kTopK));
    const bool pen = pen_active(p);
    const int want = pen ? kTopK : K;
    float lg[kTopK];
    int n = 0;
    for (; n < want; n++) {
        int best = -1;
        float bv = -FLT_MAX;
        for (int g = 0; g < ngpu; g++) {
            if (head[g] >= kTopK) continue;
            const float v = gathered[(((size_t)g * m + mi) * kTopK + head[g]) * 2];
            if (best < 0 || v > bv) { best = g; bv = v; }
        }
        if (best < 0 || bv == -FLT_MAX) break;
        lg[n] = bv;
        tok[n] = __float_as_int(gathered[(((size_t)best * m + mi) * kTopK + head[best]) * 2 + 1]);
        head[best]++;
    }
    if (n == 0) {  // every candidate is -FLT_MAX (e.g. a fully masked row): shard 0's first entry, as row_probs_warp (q27_samp_par review L1)
        tok[0] = __float_as_int(gathered[((size_t)mi * kTopK) * 2 + 1]);
        prob[0] = 1.f;
        return 1;
    }
    if (pen) {
        const unsigned char* cnt = p.pen->cnt;
        for (int i = 0; i < n; i++) {
            int c = cnt[tok[i]];
            if (rs) {
                for (int j = 0, q = 0; j < next && q < kTopK; j++)
                    if (rs[j] == slot) { q++; c += ext[j] == tok[i]; }
            } else
                for (int j = 0; j < next; j++) c += ext[j] == tok[i];
            if (c > 0) lg[i] -= p.frequency * (float)c + p.presence;
        }
        for (int i = 1; i < n; i++) {  // stable insertion sort, descending (ties keep the merge order)
            const float v = lg[i];
            const int t = tok[i];
            int j = i - 1;
            for (; j >= 0 && lg[j] < v; j--) { lg[j + 1] = lg[j]; tok[j + 1] = tok[j]; }
            lg[j + 1] = v;
            tok[j + 1] = t;
        }
        n = min(n, K);
    }
    const float inv_t = 1.f / fmaxf(p.temperature, 1e-5f);
    float sum = 0.f;
    for (int i = 0; i < n; i++) {
        prob[i] = __expf((lg[i] - lg[0]) * inv_t);
        sum += prob[i];
    }
    int keep = n;
    if (p.min_p > 0.f)
        for (int i = 1; i < keep; i++)
            if (prob[i] < p.min_p * prob[0]) { keep = i; break; }
    if (p.top_p < 1.f) {
        float tot = 0.f;
        for (int i = 0; i < keep; i++) tot += prob[i];
        float c = 0.f;
        for (int i = 0; i < keep; i++) {
            c += prob[i];
            if (c >= p.top_p * tot) { keep = i + 1; break; }
        }
    }
    sum = 0.f;
    for (int i = 0; i < keep; i++) sum += prob[i];
    for (int i = 0; i < keep; i++) prob[i] /= sum;
    return keep;
}

// Sample from prob[0..n) excluding token `skip` (whose mass is pskip), u in [0, 1).
__device__ int draw(const float* prob, const int* tok, int n, float u, int skip, float pskip) {
    float target = u * (1.f - pskip), c = 0.f;
    int last = -1;
    for (int i = 0; i < n; i++) {
        if (tok[i] == skip) continue;
        c += prob[i];
        last = i;
        if (c > target) return tok[i];
    }
    return last >= 0 ? tok[last] : tok[0];
}

// [pen-par] one block per row: row mi of a chain of verify rows sees the earlier rows' tokens as the drafts they verify
// (ext[j] = the row j + 1 input): where a row's token is used, every earlier row was accepted, so ext[j] equals the token
// row j sampled (the serial kernels' input). The slot counters advance after the rows (sample_counters_kernel).
__global__ void sample_rows_par_kernel(const float* gathered, int ngpu, int m, const SampleParams* pp, const int* ext,
                                       int* tokens) {
    if (threadIdx.x != 0) return;
    const int mi = blockIdx.x;
    const SampleParams p = *pp;
    float prob[kTopK];
    int tok[kTopK];
    const int n = row_probs(gathered, ngpu, m, mi, p, prob, tok, ext, mi);
    tokens[mi] = draw(prob, tok, n, uniform(p, mi, 0), -1, 0.f);
}
__global__ void sample_counter_kernel(SampleParams* pp) {
    if (threadIdx.x == 0) pp->counter += 1;
}
__global__ void sample_rows_slots_par_kernel(const float* gathered, int ngpu, int m, const SampleParams* base,
                                             const int* rowslot, const int* on, const int* ext, int* tokens) {
    if (threadIdx.x != 0) return;
    const int mi = blockIdx.x, s = rowslot[mi];
    const SampleParams p = base[s];
    float prob[kTopK];
    int tok[kTopK];
    if (!on[s]) {
        SampleParams g = p;
        g.top_k = 1;
        g.pen = nullptr;
        row_probs(gathered, ngpu, m, mi, g, prob, tok);
        tokens[mi] = tok[0];
        return;
    }
    const int n = row_probs(gathered, ngpu, m, mi, p, prob, tok, ext, mi, rowslot, s);
    tokens[mi] = draw(prob, tok, n, uniform(p, mi, 0), -1, 0.f);
}
// slot ids < 64 (a 64-bit mask, as sample_rows_slots_kernel): the only caller, Flash-Next, has at most 16 slots
// (flash_seq.cu clamps FLASH_MAX_SEQS to 1..16; pen_par review L2)
__global__ void sample_slots_counter_kernel(int m, SampleParams* base, const int* rowslot, const int* on) {
    if (threadIdx.x != 0) return;
    unsigned long long touched = 0ull;
    for (int mi = 0; mi < m; mi++)
        if (on[rowslot[mi]]) touched |= 1ull << (rowslot[mi] & 63);
    for (int s = 0; s < 64; s++)
        if ((touched >> s) & 1ull) base[s].counter += 1;
}

__global__ void sample_rows_kernel(const float* gathered, int ngpu, int m, SampleParams* pp,
                                   int* tokens, int chain) {
    if (threadIdx.x != 0) return;
    const SampleParams p = *pp;
    for (int mi = 0; mi < m; mi++) {
        float prob[kTopK];
        int tok[kTopK];
        const int n = row_probs(gathered, ngpu, m, mi, p, prob, tok, tokens, chain ? mi : 0);
        tokens[mi] = draw(prob, tok, n, uniform(p, mi, 0), -1, 0.f);
    }
    pp->counter = p.counter + 1;
}

// rows of several sequences, row mi with base[rowslot[mi]] (on[slot] == 0: greedy, the top candidate); every
// sampling slot's counter advances once
__global__ void sample_rows_slots_kernel(const float* gathered, int ngpu, int m, SampleParams* base,
                                         const int* rowslot, const int* on, int* tokens) {
    if (threadIdx.x != 0) return;
    unsigned long long touched = 0ull;
    for (int mi = 0; mi < m; mi++) {
        const int s = rowslot[mi];
        const SampleParams p = base[s];
        float prob[kTopK];
        int tok[kTopK];
        if (!on[s]) {
            SampleParams g = p;
            g.top_k = 1;
            g.pen = nullptr;
            row_probs(gathered, ngpu, m, mi, g, prob, tok);
            tokens[mi] = tok[0];
            continue;
        }
        // penalty: the slot's earlier (verify) rows of this step (tokens[j] of the rows j < mi with rowslot[j] == s)
        const int n = row_probs(gathered, ngpu, m, mi, p, prob, tok, tokens, mi, rowslot, s);
        tokens[mi] = draw(prob, tok, n, uniform(p, mi, 0), -1, 0.f);
        touched |= 1ull << (s & 63);
    }
    for (int s = 0; s < 64; s++)
        if ((touched >> s) & 1ull) base[s].counter += 1;
}

// penalty counts += t[0..n) at positions base.. (a lower base than the last commit: that tail is taken back first)
__device__ void pen_commit_dev(PenState* q, const int* t, int n, int base) {
    const int c = q->cpos;
    if (c >= 0 && base < c)
        for (int x = c - 1; x >= max(base, c - kPenRing); x--) {
            unsigned char& v = q->cnt[q->ring[x % kPenRing]];
            if (v != 0 && v != 255) v--;
        }
    for (int i = 0; i < n; i++) {
        const int tk = t[i];
        if (tk < 0) continue;
        if (q->cnt[tk] != 255) q->cnt[tk]++;
        q->ring[(base + i) % kPenRing] = tk;
    }
    q->cpos = base + n;
}

__device__ void accept_sampled_dev(const float* gathered, int ngpu, int k, SampleParams* pp,
                                   const SpecState& s, int m, int r0) {
    const SampleParams p = *pp;
    int a = k;
    for (int j = 0; j <= k; j++) {
        float prob[kTopK];
        int tok[kTopK];
        // penalty: row j continues the accepted drafts out[0..j) (== tokens[1..j])
        const int n = row_probs(gathered, ngpu, m, r0 + j, p, prob, tok, s.out, j);
        if (j == k) {  // every draft accepted: bonus token from the last row
            s.out[j] = draw(prob, tok, n, uniform(p, j, 1), -1, 0.f);
            break;
        }
        const int d = s.tokens[j + 1];
        float pd = 0.f;
        for (int i = 0; i < n; i++)
            if (tok[i] == d) pd = prob[i];
        if (uniform(p, j, 0) < pd) {
            s.out[j] = d;
            if (d == s.eos0 || d == s.eos1) {  // the turn ends here: later drafts must not enter the state
                a = j;
                break;
            }
            continue;
        }
        s.out[j] = draw(prob, tok, n, uniform(p, j, 1), d, pd);
        a = j;
        break;
    }
    s.emit[0] = a + 1;
    for (int j = 0; j <= a; j++) s.emit[1 + j] = s.out[j];
    for (int j = 0; j <= k; j++) s.mtp_tok[j] = s.out[min(j, a)];
    *s.acc = a;
    *s.mtp_pos = *s.pos;
    *s.hrow = 0;
    pp->counter = p.counter + 1;
    if (pen_active(p)) pen_commit_dev(p.pen, s.out, a + 1, *s.pos + 1);  // out[j] sits at pos + 1 + j
}

__global__ void accept_sampled_kernel(const float* gathered, int ngpu, int k, SampleParams* pp,
                                      SpecState s, int m, int r0) {
    if (threadIdx.x != 0) return;
    accept_sampled_dev(gathered, ngpu, k, pp, s, m, r0);
}

// [q27-samp-par] the rows' candidate lists in parallel, then the serial acceptance on them. Row j's penalty context is
// the drafts tokens[1..j]: row j is used only when the drafts before it were accepted, and then out[0..j) (the serial
// kernel's context) equals them. One warp per row (row_probs_warp; more than kAccGpus shards: its lane 0 runs row_probs),
// so every row's probabilities and the draws are the serial kernel's bit for bit. Precondition (topk_kernel): each
// shard's list is sorted descending (the merge works on ranks). kAccRows >= kMaxLookupK + 1 (the widest single-sequence step).
constexpr int kAccRows = 16;
constexpr int kAccThreads = 256;
struct AccRows {
    float prob[kAccRows][kTopK];
    int tok[kAccRows][kTopK];
    int n[kAccRows];
};
constexpr int kAccGpus = 4;  // warp form for up to 4 vocabulary shards (more: lane 0 runs row_probs)
struct AccWarp {
    float v[kAccGpus * kTopK];  // the row's gathered candidate logits of every shard
    float lg[kTopK];
    int tk[kTopK];
};
// row_probs (rs == nullptr) by one warp, bit for bit: the merge as ranks in the order (logit desc, shard asc, index
// asc) that the serial merge produces from the sorted shard lists, cut at the first -FLT_MAX; the penalty per
// candidate; the stable insertion sort as ranks (ties keep the merge order); exp per candidate; the min-p / top-p
// cuts and the normalization serially on lane 0 in the serial order. prob / tok: the row's slots (shared memory).
__device__ int row_probs_warp(const float* gathered, int ngpu, int m, int mi, const SampleParams& p, float* prob,
                              int* tok, const int* ext, int next, AccWarp& w) {
    const int lane = threadIdx.x & 31;
    const int K = max(1, min(p.top_k, kTopK));
    const bool pen = pen_active(p);
    const int want = pen ? kTopK : K;
    const int E = ngpu * kTopK;
    for (int e = lane; e < E; e += 32) {
        const int g = e / kTopK, h = e % kTopK;
        w.v[e] = gathered[(((size_t)g * m + mi) * kTopK + h) * 2];
    }
    __syncwarp();
    int cut = E;  // merged rank of the first -FLT_MAX entry (the serial merge stops there)
    for (int e = lane; e < E; e += 32) {
        const int g = e / kTopK, h = e % kTopK;
        const float v = w.v[e];
        int r = h;
        for (int g2 = 0; g2 < ngpu; g2++) {
            if (g2 == g) continue;
            const float* o = w.v + g2 * kTopK;
            int c = 0;
            if (g2 < g)
                for (int i = 0; i < kTopK; i++) c += o[i] >= v;
            else
                for (int i = 0; i < kTopK; i++) c += o[i] > v;
            r += c;
        }
        if (v == -FLT_MAX) cut = min(cut, r);
        if (r < want) {
            w.lg[r] = v;
            w.tk[r] = __float_as_int(gathered[(((size_t)g * m + mi) * kTopK + h) * 2 + 1]);
        }
    }
    for (int o = 16; o; o >>= 1) cut = min(cut, __shfl_xor_sync(0xffffffffu, cut, o));
    int n = min(want, cut);
    if (n == 0) {  // (warp-uniform) every candidate is -FLT_MAX: the same defined token as row_probs (review L1)
        if (lane == 0) tok[0] = __float_as_int(gathered[((size_t)mi * kTopK) * 2 + 1]), prob[0] = 1.f;
        __syncwarp();
        return 1;
    }
    __syncwarp();
    if (pen) {
        const unsigned char* cnt = p.pen->cnt;
        for (int i = lane; i < n; i += 32) {
            int c = cnt[w.tk[i]];
            for (int j = 0; j < next; j++) c += ext[j] == w.tk[i];
            if (c > 0) w.lg[i] -= p.frequency * (float)c + p.presence;
        }
        __syncwarp();
        float lv[2];
        int lt[2], lr[2];
        for (int q = 0; q < 2; q++) {
            const int i = lane + 32 * q;
            lr[q] = -1;
            if (i >= n) continue;
            const float v = w.lg[i];
            int r = 0;
            for (int j = 0; j < n; j++) {
                const float u = w.lg[j];
                r += u > v || (j < i && u == v);
            }
            lv[q] = v, lt[q] = w.tk[i], lr[q] = r;
        }
        n = min(n, K);
        for (int q = 0; q < 2; q++)
            if (lr[q] >= 0 && lr[q] < n) prob[lr[q]] = lv[q], tok[lr[q]] = lt[q];  // (prob holds the sorted logits)
    } else {
        for (int i = lane; i < n; i += 32) prob[i] = w.lg[i], tok[i] = w.tk[i];
    }
    __syncwarp();
    const float inv_t = 1.f / fmaxf(p.temperature, 1e-5f);
    const float l0 = prob[0];
    __syncwarp();
    for (int i = lane; i < n; i += 32) prob[i] = __expf((prob[i] - l0) * inv_t);
    __syncwarp();
    int keep = n;
    if (lane == 0) {
        if (p.min_p > 0.f)
            for (int i = 1; i < keep; i++)
                if (prob[i] < p.min_p * prob[0]) { keep = i; break; }
        if (p.top_p < 1.f) {
            float tot = 0.f;
            for (int i = 0; i < keep; i++) tot += prob[i];
            float c = 0.f;
            for (int i = 0; i < keep; i++) {
                c += prob[i];
                if (c >= p.top_p * tot) { keep = i + 1; break; }
            }
        }
        float sum = 0.f;
        for (int i = 0; i < keep; i++) sum += prob[i];
        for (int i = 0; i < keep; i++) prob[i] /= sum;
    }
    keep = __shfl_sync(0xffffffffu, keep, 0);
    __syncwarp();
    return keep;
}

__device__ void accept_rows_par(const float* gathered, int ngpu, int k, const SampleParams& p, const SpecState& s,
                                int m, int r0, AccRows& sh, AccWarp* aw) {
    const int w = threadIdx.x >> 5, lane = threadIdx.x & 31, nw = blockDim.x >> 5;
    for (int j = w; j <= k; j += nw) {
        if (ngpu <= kAccGpus) {
            const int n = row_probs_warp(gathered, ngpu, m, r0 + j, p, sh.prob[j], sh.tok[j], s.tokens + 1, j, aw[w]);
            if (lane == 0) sh.n[j] = n;
        } else if (lane == 0) {
            float prob[kTopK];
            int tok[kTopK];
            const int n = row_probs(gathered, ngpu, m, r0 + j, p, prob, tok, s.tokens + 1, j);
            sh.n[j] = n;
            for (int i = 0; i < n; i++) sh.prob[j][i] = prob[i], sh.tok[j][i] = tok[i];
        }
    }
}
// thread 0, after accept_rows_par: accept_sampled_dev's loop on the stored rows
__device__ void accept_sampled_rows(int k, SampleParams* pp, const SampleParams& p, const SpecState& s,
                                    const AccRows& sh) {
    int a = k;
    for (int j = 0; j <= k; j++) {
        const float* prob = sh.prob[j];
        const int* tok = sh.tok[j];
        const int n = sh.n[j];
        if (j == k) {
            s.out[j] = draw(prob, tok, n, uniform(p, j, 1), -1, 0.f);
            break;
        }
        const int d = s.tokens[j + 1];
        float pd = 0.f;
        for (int i = 0; i < n; i++)
            if (tok[i] == d) pd = prob[i];
        if (uniform(p, j, 0) < pd) {
            s.out[j] = d;
            if (d == s.eos0 || d == s.eos1) {
                a = j;
                break;
            }
            continue;
        }
        s.out[j] = draw(prob, tok, n, uniform(p, j, 1), d, pd);
        a = j;
        break;
    }
    s.emit[0] = a + 1;
    for (int j = 0; j <= a; j++) s.emit[1 + j] = s.out[j];
    for (int j = 0; j <= k; j++) s.mtp_tok[j] = s.out[min(j, a)];
    *s.acc = a;
    *s.mtp_pos = *s.pos;
    *s.hrow = 0;
    pp->counter = p.counter + 1;
    if (pen_active(p)) pen_commit_dev(p.pen, s.out, a + 1, *s.pos + 1);
}
__global__ void __launch_bounds__(kAccThreads) accept_sampled_par_kernel(const float* gathered, int ngpu, int k,
                                                                         SampleParams* pp, SpecState s, int m, int r0) {
    __shared__ AccRows sh;
    __shared__ AccWarp aw[kAccThreads / 32];
    if (k + 1 > kAccRows) return;  // (the host wrapper takes the serial kernel then; a direct launch must not overrun sh: review L6)
    const SampleParams p = *pp;  // (read by every thread before thread 0 advances the counter)
    accept_rows_par(gathered, ngpu, k, p, s, m, r0, sh, aw);
    __syncthreads();
    if (threadIdx.x == 0) accept_sampled_rows(k, pp, p, s, sh);
}

// greedy acceptance (as spec.cu accept_kernel)
__device__ void accept_greedy_dev(const SpecState& s, int k) {
    const int n = *s.pos;
    int a = 0;
    while (a < k && s.tokens[a + 1] == s.out[a] && s.out[a] != s.eos0 && s.out[a] != s.eos1) a++;
    s.emit[0] = a + 1;
    for (int j = 0; j <= a; j++) s.emit[1 + j] = s.out[j];
    for (int j = 0; j <= k; j++) s.mtp_tok[j] = s.out[j];
    *s.acc = a;
    *s.mtp_pos = n;
    *s.hrow = 0;
}

// every sequence of a batched step in one launch: block z = sequence z (each one sequential, as alone)
__global__ void accept_batch_kernel(const float* gathered, int ngpu, AcceptBatch b) {
    if (threadIdx.x != 0) return;
    const int z = blockIdx.x;
    if (b.p[z]) accept_sampled_dev(gathered, ngpu, b.k[z], b.p[z], b.s[z], b.m, b.r0[z]);
    else accept_greedy_dev(b.s[z], b.k[z]);
}

__global__ void __launch_bounds__(kAccThreads) accept_batch_par_kernel(const float* gathered, int ngpu, AcceptBatch b) {
    __shared__ AccRows sh;
    __shared__ AccWarp aw[kAccThreads / 32];
    const int z = blockIdx.x;
    if (b.k[z] + 1 > kAccRows) return;  // (host-checked: spec_accept_batch takes the serial kernel then; review L6)
    if (!b.p[z]) {
        if (threadIdx.x == 0) accept_greedy_dev(b.s[z], b.k[z]);
        return;
    }
    const SampleParams p = *b.p[z];
    accept_rows_par(gathered, ngpu, b.k[z], p, b.s[z], b.m, b.r0[z], sh, aw);
    __syncthreads();
    if (threadIdx.x == 0) accept_sampled_rows(b.k[z], b.p[z], p, b.s[z], sh);
}

// grammar-constrained decoding: logits of rows [r0, r0 + rows) whose global token id (off + i)
// has no bit in the row's mask become -inf (bits: one uint32 per 32 token ids of the whole
// vocabulary; row j uses bits + j * stride, stride 0: one mask for every row)
__global__ void mask_logits_kernel(float* __restrict__ logits, int v, int off,
                                   const unsigned* __restrict__ bits, int r0, size_t stride) {
    float* row = logits + (size_t)(r0 + blockIdx.y) * v;
    bits += blockIdx.y * stride;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < v; i += gridDim.x * blockDim.x) {
        const int t = off + i;
        if (!((bits[t >> 5] >> (t & 31)) & 1u)) row[i] = -INFINITY;
    }
}

__global__ void pen_commit_kernel(const SampleParams* p, const int* toks, const int* end_pos) {
    if (threadIdx.x != 0 || !pen_active(*p)) return;
    const int n = toks[0];
    pen_commit_dev(p->pen, toks + 1, n, *end_pos - n + 1);
}
__global__ void pen_commit_b_kernel(const SampleParams* base, const int* segs, int nseg, const int* toks, int stride,
                                    const int* svp) {
    const int s = threadIdx.x;
    if (s >= nseg) return;
    const int slot = segs[s];
    if (!pen_active(base[slot])) return;
    const int* e = toks + (size_t)s * stride;
    pen_commit_dev(base[slot].pen, e + 1, e[0], svp[slot * 2] - e[0] + 1);
}
struct HostToks {
    int t[16];
};
__global__ void pen_commit_host_kernel(const SampleParams* p, HostToks t, int n, int pos) {
    if (threadIdx.x != 0 || !pen_active(*p)) return;
    pen_commit_dev(p->pen, t.t, n, pos);
}
__global__ void pen_seed_kernel(PenState* q, const int* seed, int n, int vocab) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
        if (seed[i] >= 0 && seed[i] < vocab) q->cnt[seed[i]] = 1;
}

}  // namespace

void pen_commit(const SampleParams* p, const int* toks, const int* end_pos, cudaStream_t st) {
    pen_commit_kernel<<<1, 32, 0, st>>>(p, toks, end_pos);
}
void pen_commit_b(const SampleParams* base, const int* segs, int nseg, const int* toks, int stride, const int* svp,
                  cudaStream_t st) {
    if (nseg > 0) pen_commit_b_kernel<<<1, 64, 0, st>>>(base, segs, nseg, toks, stride, svp);
}
void pen_commit_host(const SampleParams* p, const int* toks, int n, int pos, cudaStream_t st) {
    HostToks t{};
    n = n < 0 ? 0 : n > 16 ? 16 : n;
    for (int i = 0; i < n; i++) t.t[i] = toks[i];
    if (n > 0) pen_commit_host_kernel<<<1, 32, 0, st>>>(p, t, n, pos);
}
void pen_reset(PenState* pen, int vocab, const int* seed, int n, cudaStream_t st) {
    CUDA_CHECK(cudaMemsetAsync(pen, 0xff, sizeof(int), st));  // cpos = -1
    CUDA_CHECK(cudaMemsetAsync(pen->cnt, 0, (size_t)vocab, st));
    if (n > 0) pen_seed_kernel<<<(n + 255) / 256, 256, 0, st>>>(pen, seed, n, vocab);
}

void mask_logits(float* logits, int v, int vocab_off, const unsigned* bits, int r0, int rows,
                 size_t stride, cudaStream_t st) {
    mask_logits_kernel<<<dim3(32, rows), 256, 0, st>>>(logits, v, vocab_off, bits, r0, stride);
}

void topk_local(const float* logits, int v, int m, int vocab_off, float* pairs, cudaStream_t st) {
    topk_kernel<<<m, kSelThreads, 0, st>>>(logits, v, vocab_off, pairs);
}

void sample_rows_slots(const float* gathered, int ngpu, int m, SampleParams* base, const int* rowslot,
                       const int* on, int* tokens, cudaStream_t st, const int* ext) {
    static const bool par = env_int("QE_SAMPLE_PAR", 1);
    if (ext && par && m > 1) {
        sample_rows_slots_par_kernel<<<m, 32, 0, st>>>(gathered, ngpu, m, base, rowslot, on, ext, tokens);
        sample_slots_counter_kernel<<<1, 32, 0, st>>>(m, base, rowslot, on);
        return;
    }
    sample_rows_slots_kernel<<<1, 32, 0, st>>>(gathered, ngpu, m, base, rowslot, on, tokens);
}
void sample_rows(const float* gathered, int ngpu, int m, SampleParams* p, int* tokens,
                 cudaStream_t st, bool chain, const int* ext) {
    static const bool par = env_int("QE_SAMPLE_PAR", 1);
    if (chain && ext && par && m > 1) {
        sample_rows_par_kernel<<<m, 32, 0, st>>>(gathered, ngpu, m, p, ext, tokens);
        sample_counter_kernel<<<1, 32, 0, st>>>(p);
        return;
    }
    sample_rows_kernel<<<1, 32, 0, st>>>(gathered, ngpu, m, p, tokens, chain ? 1 : 0);
}

size_t sample_stack_need() {
    size_t mx = 0;
    const void* k[] = {(const void*)sample_rows_kernel, (const void*)sample_rows_slots_kernel, (const void*)sample_rows_par_kernel,
                       (const void*)sample_rows_slots_par_kernel,
                       (const void*)accept_sampled_kernel, (const void*)accept_batch_kernel,
                       (const void*)accept_sampled_par_kernel, (const void*)accept_batch_par_kernel};
    for (const void* f : k) {
        cudaFuncAttributes a;
        if (cudaFuncGetAttributes(&a, f) == cudaSuccess) mx = std::max(mx, a.localSizeBytes);
    }
    return mx;
}

// [q27-samp-par] QE_SAMPLE_PAR=0: the serial kernels (one thread walks every row)
static bool accept_par_on() {
    static const bool on = env_int("QE_SAMPLE_PAR", 1) != 0;
    return on;
}

void spec_accept_batch(const float* gathered, int ngpu, const AcceptBatch& b, cudaStream_t st) {
    if (b.n <= 0) return;
    bool par = accept_par_on(), any = false;  // (only greedy sequences: the serial kernel, one thread each; review L3)
    for (int z = 0; z < b.n && par; z++) par = b.k[z] + 1 <= kAccRows, any |= b.p[z] != nullptr;
    par = par && any;
    if (par) accept_batch_par_kernel<<<b.n, kAccThreads, 0, st>>>(gathered, ngpu, b);
    else accept_batch_kernel<<<b.n, 32, 0, st>>>(gathered, ngpu, b);
}

void spec_accept_sampled(const float* gathered, int ngpu, int k, SampleParams* p,
                         const SpecState& s, cudaStream_t st, int mtot, int r0) {
    if (accept_par_on() && k + 1 <= kAccRows)
        accept_sampled_par_kernel<<<1, kAccThreads, 0, st>>>(gathered, ngpu, k, p, s, mtot < 0 ? k + 1 : mtot, r0);
    else
        accept_sampled_kernel<<<1, 32, 0, st>>>(gathered, ngpu, k, p, s, mtot < 0 ? k + 1 : mtot, r0);
}
