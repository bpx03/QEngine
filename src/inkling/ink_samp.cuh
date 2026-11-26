// [ink-samp-dev] filtered sampling (top_k <= FK_MAX with top_p / min_p, temperature) inside the exact two-stage head
// (included from ink.cu after ink_tp.cuh). Per filtered row j of a step:
//   1. the screens give raw bounds [lo, hi] of every vocab row (no Gumbel transform) and write lo (HeadArgs::lo);
//   2. k_fthr<0 / 1>: a value <= the K-th largest lower end of the rank's rows (2-pass radix histogram over the
//      ordered lower ends, 11 + 11 bits; the threshold = the lower edge of the bin that holds the K-th): every row of
//      the rank's exact top K has hi >= it;
//   3. the rescore (b_head_rescore, filtered branch) computes the exact logit of every row with hi >= threshold into
//      the row's list (key = ordered logit << 32 | ~global row: larger logit first, ties the lower row);
//   4. k_fsel: the rank's exact top K of the list (radix select over the unique keys, sorted); TP: pushed to the peers;
//   5. k_fmerge: the global top K (all ranks' lists, the same on every rank), then the host rule of the C API in order
//      (top_k, min_p relative to the most likely token, top_p = the smallest prefix whose mass reaches top_p of the
//      kept mass), the token = argmax over the kept set of cap(z) / T + g(seed, position, row): the Gumbel-max form
//      draws exactly from softmax(cap(z) / T) restricted to the kept set and renormalised; the noise is a function of
//      (seed, position, row) only, so a seed gives the same text alone, in a batch, with or without drafts.
//   A list overflow (> FCAP candidates) or an uncertified min_p-only set gives token -2: the outer Engine call picks
//   that row's token on the host from the full head with the same rule and noise (Engine::filt_host).
struct A_fsel {
    HeadArgs h;                 // base pointers of the step's rows (rT / rS / rP / rK / rTP / rMP, fc, flist)
    unsigned long long* loc;    // [MROW][FLOC_W] this rank's list: [0] = count | overflow << 63, the keys, [K + 1] tail
    TpAr t;                     // TP exchange (tp = 0: one card); direct: t.src[g] = rank g's loc
    int idx, tp;
    int force;                  // tests: rows at positions divisible by force > 0 return -2 (the host pick)
    int mode;                   // k_fsel: 0 = the rank's top K (+ push), 1 = round A: fc[1] = the K-th largest key's logit
    int* arg;                   // [rows] tokens out (k_fmerge)
};

// warp 0: the digit d (255 .. 0 from the top) where the running count from the top reaches need; above = count above d
__device__ __forceinline__ void fs_digit(const unsigned* hs, unsigned need, unsigned& d, unsigned& above) {
    const int lane = threadIdx.x & 31;
    unsigned v[8], s = 0;
#pragma unroll
    for (int i = 0; i < 8; i++) {
        v[i] = hs[255 - 8 * lane - i];
        s += v[i];
    }
    unsigned inc = s;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        const unsigned y = __shfl_up_sync(0xffffffffu, inc, o);
        if (lane >= o) inc += y;
    }
    const unsigned bal = __ballot_sync(0xffffffffu, inc >= need);
    const int L = bal ? __ffs(bal) - 1 : 31;
    unsigned dd = 0, aa = 0;
    if (lane == L) {
        unsigned a = inc - s;
        int i = 0;
        for (; i < 7; i++) {
            if (a + v[i] >= need) break;
            a += v[i];
        }
        dd = 255 - 8 * lane - i;
        aa = a;
    }
    d = __shfl_sync(0xffffffffu, dd, L);
    above = __shfl_sync(0xffffffffu, aa, L);
}

// the same over FHB = 2048 bins with 256 threads (8 bins per thread, block scan from the top)
__device__ __forceinline__ void fs_bin2048(const unsigned* hs, unsigned need, unsigned& d, unsigned& above, unsigned* scr) {
    const int t = threadIdx.x, lane = t & 31, w = t >> 5;
    unsigned v[8], s = 0;
#pragma unroll
    for (int i = 0; i < 8; i++) {
        v[i] = hs[FHB - 1 - 8 * t - i];
        s += v[i];
    }
    unsigned inc = s;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        const unsigned y = __shfl_up_sync(0xffffffffu, inc, o);
        if (lane >= o) inc += y;
    }
    if (lane == 31) scr[w] = inc;
    __syncthreads();
    unsigned off = 0;
    for (int q = 0; q < w; q++) off += scr[q];
    inc += off;
    const unsigned exc = inc - s;
    // the first thread (from the top) whose inclusive count reaches need (unique: exc < need <= inc)
    if ((exc < need && inc >= need) || (t == 255 && inc < need)) {
        unsigned a = exc;
        int i = 0;
        for (; i < 7; i++) {
            if (a + v[i] >= need) break;
            a += v[i];
        }
        scr[8] = FHB - 1 - 8 * t - i;
        scr[9] = a;
    }
    __syncthreads();
    d = scr[8];
    above = scr[9];
}

template <int PASS>
__global__ void __launch_bounds__(256) k_fthr(const __grid_constant__ HeadArgs h) {
    const int j = blockIdx.y;
    const int K = row_filt(h, j);
    if (!K) return;
    unsigned* fc = h.fc + j * 16;
    unsigned* hist = h.fhist + (size_t)j * FHB;
    const float* lo = h.lo + (size_t)j * h.N;
    __shared__ unsigned sh[FHB];
    __shared__ unsigned scr[16];
    __shared__ bool last;
    for (int i = threadIdx.x; i < FHB; i += 256) sh[i] = 0;
    __syncthreads();
    const unsigned b1 = PASS ? ((volatile unsigned*)fc)[3] : 0u;
    for (int r = blockIdx.x * 256 + threadIdx.x; r < h.N; r += gridDim.x * 256) {
        const unsigned o = f2ord(lo[r]);
        if (PASS == 0) atomicAdd(&sh[o >> 21], 1u);
        else if ((o >> 21) == b1) atomicAdd(&sh[(o >> 10) & (FHB - 1)], 1u);
    }
    __syncthreads();
    for (int i = threadIdx.x; i < FHB; i += 256)
        if (sh[i]) atomicAdd(hist + i, sh[i]);
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) last = atomicAdd(fc + 2, 1u) == gridDim.x - 1;
    __syncthreads();
    if (!last) return;
    __threadfence();
    for (int i = threadIdx.x; i < FHB; i += 256) sh[i] = ((volatile unsigned*)hist)[i];
    __syncthreads();
    const unsigned need = PASS ? (unsigned)K - ((volatile unsigned*)fc)[4] : (unsigned)K;
    unsigned d, above;
    fs_bin2048(sh, need, d, above, scr);
    if (threadIdx.x == 0) {
        if (PASS == 0) {
            fc[3] = d;
            fc[4] = above;
        } else
            fc[1] = (b1 << 21) | (d << 10);   // the lower edge of the bin: <= the K-th largest lower end
        fc[2] = 0;
    }
    for (int i = threadIdx.x; i < FHB; i += 256) hist[i] = 0;
}

__global__ void __launch_bounds__(1024) k_fsel(const __grid_constant__ A_fsel A) {
    const HeadArgs& h = A.h;
    const int j = blockIdx.x, tid = threadIdx.x;
    const int K = row_filt(h, j);
    if (!K) return;
    unsigned* fc = h.fc + j * 16;
    const unsigned long long* L = h.flist + (size_t)j * FCAP;
    const unsigned cnt = *(volatile unsigned*)fc;
    const bool ovf = cnt > (unsigned)FCAP;
    const int C = (int)min(cnt, (unsigned)FCAP);
    const int n = min(K, C);
    __shared__ unsigned hs[256];
    __shared__ unsigned long long top[FK_MAX];
    __shared__ unsigned sd_, sa_, nt;
    unsigned long long prefix = 0, mask = 0;
    unsigned need = (unsigned)n;
    if (C > n || A.mode == 1) {   // radix select of the n-th largest key (keys are unique: one per row)
        for (int sh = 56; sh >= 0; sh -= 8) {
            if (tid < 256) hs[tid] = 0;
            __syncthreads();
            for (int i = tid; i < C; i += 1024) {
                const unsigned long long k = L[i];
                if ((k & mask) == prefix) atomicAdd(&hs[(unsigned)(k >> sh) & 255u], 1u);
            }
            __syncthreads();
            if (tid < 32) {
                unsigned d, a;
                fs_digit(hs, need, d, a);
                if (tid == 0) {
                    sd_ = d;
                    sa_ = a;
                }
            }
            __syncthreads();
            prefix |= (unsigned long long)sd_ << sh;
            mask |= 255ull << sh;
            need -= sa_;
            __syncthreads();
        }
    }
    if (A.mode == 1) {   // round A: the K-th largest exact logit of >= K rows is <= the rank's K-th largest
        if (tid == 0) {
            if (C >= K && !ovf) fc[1] = (unsigned)(prefix >> 32);   // (else fc[1] keeps the lower-end threshold)
            fc[0] = 0;
        }
        return;
    }
    if (tid == 0) nt = 0;
    __syncthreads();
    for (int i = tid; i < C; i += 1024) {
        const unsigned long long k = L[i];
        if (k >= prefix) {
            const unsigned q = atomicAdd(&nt, 1u);
            if (q < (unsigned)FK_MAX) top[q] = k;
        }
    }
    __syncthreads();
    unsigned long long* out = A.loc + (size_t)j * FLOC_W;
    if (tid < n) {
        const unsigned long long k = top[tid];
        int r = 0;
        for (int i = 0; i < n; i++) r += top[i] > k;
        out[1 + r] = k;
    }
    if (tid >= n && tid < K) out[1 + tid] = 0ull;
    // [ink-samp-dev] FK_ZT (top_p alone): an upper bound on the mass of the rank's rows outside its top n, relative to
    // its n-th logit (every term <= 1): the non-candidates' screen bound (rescore) + the list rows below the top n (exact)
    __shared__ double tl;
    const bool zt = (h.rK[j] & FK_ZT) != 0;
    if (tid == 0) tl = 0.0;
    __syncthreads();
    if (zt && n >= 1 && !ovf) {
        const float iT = h.rT[j];
        const float cxk = head_cap(ord2f((unsigned)(out[n] >> 32)));
        double a = 0.0;
        for (int i = tid; i < C; i += 1024) {
            const unsigned long long k = L[i];
            if (k < out[n]) a += exp((double)(head_cap(ord2f((unsigned)(k >> 32))) - cxk) * (double)iT);
        }
        if (a > 0.0) atomicAdd(&tl, a);
        __syncthreads();
        if (tid == 0) {
            const float cthr = head_cap(ord2f(((volatile unsigned*)fc)[1]));
            tl += h.ftail[j] * exp((double)(cthr - cxk) * (double)iT) * (1.0 + 1e-6);
        }
    }
    __syncthreads();
    if (tid == 0) {
        double tv = tl;
        unsigned long long tb;
        memcpy(&tb, &tv, 8);
        out[K + 1] = tb;
        if (zt) h.ftail[j] = 0.0;
        out[0] = (unsigned long long)n | (ovf ? 1ull << 63 : 0ull);
        fc[0] = 0;
        fc[5] = n;
        if (ovf) fc[6]++;
        const int b = cnt < 64 ? 0 : cnt < 256 ? 1 : cnt < 1024 ? 2 : cnt < 4096 ? 3 : cnt <= (unsigned)FCAP ? 4 : 5;
        fc[8 + b]++;   // statistics: candidates per filtered row (Engine::filter_stats)
    }
    if (!(A.tp && A.t.push)) return;
    __syncthreads();
    __threadfence();
    const TpAr& t = A.t;
    if (tid <= K + 1) {   // words 1 .. K + 2 of the row (word 0: k_tpmax), all K + 2 written (the readers wait for each)
        const unsigned long long v = out[tid];
        const unsigned stp = *t.step, tag = stp * 256u + (unsigned)A.idx + 1u;
        const int slot = TP_SLOTS + (int)(stp & 1u);
        const size_t me = (((size_t)slot * t.n + t.rank) * MROW + j) * TPW + 1 + tid;
        for (int g = 0; g < t.n; g++)
            if (g != t.rank) tp_stD((uint4*)t.src[g] + me, (unsigned)v, (unsigned)(v >> 32), tag);
    }
}

__device__ __forceinline__ float gumbel(uint32_t seed, int step, int row);
__global__ void __launch_bounds__(1024) k_fmerge(const __grid_constant__ A_fsel A) {
    const HeadArgs& h = A.h;
    const int j = blockIdx.x, tid = threadIdx.x;
    const int K = row_filt(h, j);
    if (!K) return;
    const TpAr& t = A.t;
    const int NR = A.tp ? t.n : 1, me = A.tp ? t.rank : 0;
    __shared__ unsigned long long all[TP_MAX * FK_MAX];
    __shared__ unsigned long long top[FK_MAX];
    __shared__ unsigned cg[TP_MAX];
    __shared__ double tg[TP_MAX];   // FK_ZT: each rank's tail bound (relative to its n-th logit)
    __shared__ int bad, tot;
    if (tid == 0) bad = 0;
    __syncthreads();
    const unsigned long long t0 = tp_timer();
    unsigned tag = 0;
    int slot = 0;
    if (A.tp && t.push) {
        const unsigned stp = *t.step;
        tag = stp * 256u + (unsigned)A.idx + 1u;
        slot = TP_SLOTS + (int)(stp & 1u);
    }
    for (int i = tid; i < NR * (K + 2); i += 1024) {
        const int g = i / (K + 2), w = i % (K + 2);
        unsigned long long v;
        if (g == me) v = ((volatile const unsigned long long*)A.loc)[(size_t)j * FLOC_W + w];
        else if (t.push) {
            unsigned lo, hi;
            if (!tp_ldD(t.rd + (((size_t)slot * t.n + g) * MROW + j) * TPW + 1 + w, tag, t.ab, t0, lo, hi)) {
                atomicOr(&bad, 2);   // aborted
                continue;
            }
            v = ((unsigned long long)hi << 32) | lo;
        } else
            v = ((const unsigned long long*)t.src[g])[(size_t)j * FLOC_W + w];
        if (w == 0) {
            cg[g] = (unsigned)(v & 0xffffu);
            if (v >> 63) atomicOr(&bad, 1);
        } else if (w == K + 1) {
            double d;
            memcpy(&d, &v, 8);
            tg[g] = d;
        } else
            all[g * FK_MAX + w - 1] = v;
    }
    __syncthreads();
    if (bad & 2) return;   // a failed rank: the host recovers the context
    if (tid == 0) {
        int s = 0;
        for (int g = 0; g < NR; g++) s += (int)cg[g];
        tot = s;
    }
    for (int i = tid; i < NR * K; i += 1024) {   // global rank of each entry (keys unique over all ranks)
        const int g = i / K, q = i % K;
        if (q >= (int)cg[g]) continue;
        const unsigned long long k = all[g * FK_MAX + q];
        int r = 0;
        for (int g2 = 0; g2 < NR; g2++)
            for (int q2 = 0; q2 < (int)cg[g2]; q2++) r += all[g2 * FK_MAX + q2] > k;
        if (r < K) top[r] = k;
    }
    __syncthreads();
    if (tid) return;
    const int nk = min(K, tot);
    if (bad || nk <= 0 || (A.force > 0 && h.rP[j] % A.force == 0)) {
        A.arg[j] = -2;
        return;
    }
    const float iT = h.rT[j], tp = h.rTP[j], mp = h.rMP[j];
    const bool cert = (h.rK[j] & FK_CERT) != 0;
    double q[FK_MAX];
    float cv[FK_MAX];
    for (int i = 0; i < nk; i++) {
        cv[i] = head_cap(ord2f((unsigned)(top[i] >> 32)));
        q[i] = exp((double)(cv[i] - cv[0]) * (double)iT);
    }
    int m = nk;
    if (mp > 0.f) {   // min_p: relative to the most likely token (q = 1 there)
        m = 1;
        while (m < nk && q[m] >= (double)mp) m++;
        if (cert && m == nk && nk == K) {   // no user top_k: the set may go on past the K-list
            A.arg[j] = -2;
            return;
        }
    }
    if (tp < 1.f && (h.rK[j] & FK_ZT)) {
        // top_p alone: the mass Z of the whole vocabulary lies in [lo, lo + tail]; lo = the global top K, tail = the
        // ranks' bounds of their rows outside their lists' top (scaled to the global maximum) + the merged entries
        // below the global top K (exact). The prefix is exact when both ends give the same one inside the list.
        double lo = 0, tl = 0;
        for (int i = 0; i < nk; i++) lo += q[i];
        for (int g = 0; g < NR; g++) {
            if (!cg[g]) continue;
            const float cx = head_cap(ord2f((unsigned)(all[g * FK_MAX + cg[g] - 1] >> 32)));
            tl += tg[g] * exp((double)(cx - cv[0]) * (double)iT);
            for (int q2 = 0; q2 < (int)cg[g]; q2++) {
                const unsigned long long k = all[g * FK_MAX + q2];
                if (k < top[nk - 1]) tl += exp((double)(head_cap(ord2f((unsigned)(k >> 32))) - cv[0]) * (double)iT);
            }
        }
        auto pre = [&](double Z) {   // the prefix length for normaliser Z (nk + 1: not reached inside the list)
            const double want = (double)tp * Z;
            double acc = 0;
            int u = 0;
            while (u < nk && acc < want) acc += q[u++];
            return acc >= want ? max(u, 1) : nk + 1;
        };
        const int u0 = pre(lo), u1 = pre(lo + tl);
        if (u0 != u1 || u1 > nk) {   // not certified: the host picks this row
            A.arg[j] = -2;
            return;
        }
        m = u1;
    } else if (tp < 1.f) {   // top_p: the smallest prefix whose mass reaches top_p of the kept mass
        double sum = 0;
        for (int i = 0; i < m; i++) sum += q[i];
        const double want = (double)tp * sum;
        double acc = 0;
        int u = 0;
        while (u < m && acc < want) acc += q[u++];
        m = max(u, 1);
    }
    const uint32_t sd = h.rS[j];
    const int st = h.rP[j];
    float bv = -INFINITY;
    unsigned br = 0xffffffffu;
    for (int i = 0; i < m; i++) {
        const unsigned row = 0xffffffffu - (unsigned)(top[i] & 0xffffffffu);
        const float v = cv[i] * iT + gumbel(sd, st, (int)row);
        if (v > bv || (v == bv && row < br)) {
            bv = v;
            br = row;
        }
    }
    A.arg[j] = (int)br;
}
