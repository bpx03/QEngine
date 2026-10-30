#include <algorithm>
#include <cfloat>

#include "sample.cuh"

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

__device__ __forceinline__ unsigned long long splitmix(unsigned long long x) {
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}
__device__ __forceinline__ float uniform(const SampleParams& p, int row, int salt) {
    const unsigned long long h =
        splitmix(p.seed ^ splitmix(p.counter * 64 + row * 4 + salt + 1));
    return (h >> 40) * (1.0f / 16777216.0f);  // 24-bit uniform in [0, 1)
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
                       const int* on, int* tokens, cudaStream_t st) {
    sample_rows_slots_kernel<<<1, 32, 0, st>>>(gathered, ngpu, m, base, rowslot, on, tokens);
}
void sample_rows(const float* gathered, int ngpu, int m, SampleParams* p, int* tokens,
                 cudaStream_t st, bool chain) {
    sample_rows_kernel<<<1, 32, 0, st>>>(gathered, ngpu, m, p, tokens, chain ? 1 : 0);
}

size_t sample_stack_need() {
    size_t mx = 0;
    const void* k[] = {(const void*)sample_rows_kernel, (const void*)sample_rows_slots_kernel,
                       (const void*)accept_sampled_kernel, (const void*)accept_batch_kernel};
    for (const void* f : k) {
        cudaFuncAttributes a;
        if (cudaFuncGetAttributes(&a, f) == cudaSuccess) mx = std::max(mx, a.localSizeBytes);
    }
    return mx;
}

void spec_accept_batch(const float* gathered, int ngpu, const AcceptBatch& b, cudaStream_t st) {
    if (b.n > 0) accept_batch_kernel<<<b.n, 32, 0, st>>>(gathered, ngpu, b);
}

void spec_accept_sampled(const float* gathered, int ngpu, int k, SampleParams* p,
                         const SpecState& s, cudaStream_t st, int mtot, int r0) {
    accept_sampled_kernel<<<1, 32, 0, st>>>(gathered, ngpu, k, p, s, mtot < 0 ? k + 1 : mtot, r0);
}
