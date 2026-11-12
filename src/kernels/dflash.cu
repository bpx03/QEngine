// DFlash2 block drafter kernels (dflash.cuh)
#include <cfloat>
#include <climits>
#include "dflash.cuh"

namespace {

__device__ __forceinline__ float bfr(float x) { return __bfloat162float(__float2bfloat16(x)); }  // round to bf16

__device__ __forceinline__ float block_sum256(float v, float* sh) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    __syncthreads();
    if ((threadIdx.x & 31) == 0) sh[threadIdx.x >> 5] = v;
    __syncthreads();
    float t = 0.f;
    for (int i = 0; i < (int)(blockDim.x >> 5); i++) t += sh[i];
    return t;
}

__global__ void tap_kernel(const float* __restrict__ resid, int j, bf16* __restrict__ tap, int row0) {
    const int r = blockIdx.x;
    const float* x = resid + (size_t)r * kDflHid;
    bf16* o = tap + ((size_t)(row0 + r) * kDflTaps + j) * kDflHid;
    for (int c = threadIdx.x * 4; c < kDflHid; c += blockDim.x * 4) {
        const float4 v = *reinterpret_cast<const float4*>(x + c);
        bf162 a = __floats2bfloat162_rn(v.x, v.y), b = __floats2bfloat162_rn(v.z, v.w);
        *reinterpret_cast<bf162*>(o + c) = a;
        *reinterpret_cast<bf162*>(o + c + 2) = b;
    }
}

constexpr int kRmsThreads = kDflHid / 16;  // 320: 4 float4 per thread
__global__ void __launch_bounds__(kRmsThreads) rms_kernel(const float* __restrict__ x, int ldx, const bf16* __restrict__ w,
                                                         float eps, bf16* __restrict__ out) {
    __shared__ float sh[32];
    const float4* xr = reinterpret_cast<const float4*>(x + (size_t)blockIdx.x * ldx);
    float4 v[4];
#pragma unroll
    for (int i = 0; i < 4; i++) v[i] = xr[threadIdx.x + i * kRmsThreads];
    float ss = 0.f;
#pragma unroll
    for (int i = 0; i < 4; i++) ss += v[i].x * v[i].x + v[i].y * v[i].y + v[i].z * v[i].z + v[i].w * v[i].w;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
    if ((threadIdx.x & 31) == 0) sh[threadIdx.x >> 5] = ss;
    __syncthreads();
    float t = 0.f;
#pragma unroll
    for (int i = 0; i < kRmsThreads / 32; i++) t += sh[i];
    const float r = rsqrtf(t / kDflHid + eps);
    bf16* o = out + (size_t)blockIdx.x * kDflHid;
#pragma unroll
    for (int i = 0; i < 4; i++) {
        const int c = (threadIdx.x + i * kRmsThreads) * 4;
        const float* vv = reinterpret_cast<const float*>(&v[i]);
        bf16 res[4];
#pragma unroll
        for (int j = 0; j < 4; j++) res[j] = __float2bfloat16(bfr(vv[j] * r) * __bfloat162float(w[c + j]));
        *reinterpret_cast<uint2*>(o + c) = *reinterpret_cast<const uint2*>(res);
    }
}

__global__ void conv_kernel(const float* __restrict__ xf, const bf16* __restrict__ xb, const float* __restrict__ dyn, int c,
                            const bf16* __restrict__ base, bf16* __restrict__ out, float* __restrict__ add) {
    const int t = blockIdx.x;
    const float* d = dyn + (size_t)t * kDflKp + c * 2 * kDflG;
    {
        const int ch = blockIdx.y * blockDim.x + threadIdx.x;
        const int g = ch / kDflGs;
        float acc = 0.f;
#pragma unroll
        for (int o = 0; o < 2; o++) {
            if (t - o < 0) continue;
            const size_t ix = (size_t)(t - o) * kDflHid + ch;
            const float v = xf ? bfr(xf[ix]) : __bfloat162float(xb[ix]);
            acc += __bfloat162float(base[o * kDflHid + ch]) * v;
            acc += bfr(d[o * kDflG + g]) * v;
        }
        if (add) add[(size_t)t * kDflHid + ch] += bfr(acc);
        else out[(size_t)t * kDflHid + ch] = __float2bfloat16(acc);
    }
}

// one head (128 dims, one thread each): Qwen3 head norm + neox RoPE, bf16 rounding as the reference
__device__ __forceinline__ float head_norm_rope(float x, const bf16* w, const float* inv, int p, float eps, float* sh,
                                                bool rope) {
    const int d = threadIdx.x;
    float ss = x * x;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
    if ((d & 31) == 0) sh[d >> 5] = ss;
    __syncthreads();
    const float r = rsqrtf((sh[0] + sh[1] + sh[2] + sh[3]) / kDflHd + eps);
    const float y = bfr(bfr(x * r) * __bfloat162float(w[d]));
    if (!rope) return y;
    __syncthreads();
    sh[4 + d] = y;
    __syncthreads();
    const int i = d & 63;
    const float fr = (float)p * inv[i];
    double s, cs;
    sincos((double)fr, &s, &cs);
    const float cb = bfr((float)cs), sb = bfr((float)s);
    const float rot = d < 64 ? -sh[4 + d + 64] : sh[4 + d - 64];
    return bfr(bfr(y * cb) + bfr(rot * sb));
}

__global__ void __launch_bounds__(128) qkv_block_kernel(const float* __restrict__ proj, const bf16* __restrict__ qn,
                                                        const bf16* __restrict__ kn, const float* __restrict__ inv,
                                                        const int* __restrict__ pos, float eps, bf16* __restrict__ q,
                                                        bf16* __restrict__ k, bf16* __restrict__ v) {
    __shared__ float sh[4 + kDflHd];
    const int t = blockIdx.x, hh = blockIdx.y, d = threadIdx.x;
    const float* pr = proj + (size_t)t * (kDflNq + 2 * kDflNkv) * kDflHd;
    const int p = *pos + t;
    if (hh < kDflNq) {
        const float y = head_norm_rope(pr[hh * kDflHd + d], qn, inv, p, eps, sh, true);
        q[((size_t)t * kDflNq + hh) * kDflHd + d] = __float2bfloat16(y);
    } else if (hh < kDflNq + kDflNkv) {
        const int h = hh - kDflNq;
        const float y = head_norm_rope(pr[(kDflNq + h) * kDflHd + d], kn, inv, p, eps, sh, true);
        k[((size_t)t * kDflNkv + h) * kDflHd + d] = __float2bfloat16(y);
    } else {
        const int h = hh - kDflNq - kDflNkv;
        v[((size_t)t * kDflNkv + h) * kDflHd + d] = __float2bfloat16(pr[(kDflNq + kDflNkv + h) * kDflHd + d]);
    }
}

__global__ void __launch_bounds__(128) kv_ctx_kernel(const float* __restrict__ kv, const bf16* __restrict__ kn,
                                                     const float* __restrict__ inv, const int* __restrict__ pos, int row0,
                                                     float eps, const DflRing* __restrict__ ring, int l, int tags) {
    __shared__ float sh[4 + kDflHd];
    const int t = blockIdx.x, hh = blockIdx.y, d = threadIdx.x;
    const int p = *pos + row0 + t, row = p % kDflRing;
    const float* pr = kv + (size_t)t * 2 * kDflNkv * kDflHd;
    if (hh < kDflNkv) {
        const float y = head_norm_rope(pr[hh * kDflHd + d], kn, inv, p, eps, sh, true);
        ring->k[l][((size_t)row * kDflNkv + hh) * kDflHd + d] = __float2bfloat16(y);
    } else {
        const int h = hh - kDflNkv;
        ring->v[l][((size_t)row * kDflNkv + h) * kDflHd + d] = __float2bfloat16(pr[(kDflNkv + h) * kDflHd + d]);
    }
    if (tags && hh == 0 && d == 0) ring->tag[row] = p;
}

constexpr int kAttnPart = kDflHd + 2;
constexpr int kRingPer = (kDflRing + kDflSplits - 1) / kDflSplits;
static_assert(kRingPer <= 256, "attn_kernel lists a split's rows in one pass of 256 threads");

// grid (8 kv heads, kDflSplits + 1): split s < kDflSplits = ring rows [s * kRingPer, ..), the last = the block's own keys.
// The split's visible rows (any query row) are listed and their K / V staged in shared memory once; warp t = query row t
// for the 4 q heads of the kv head.
constexpr int kQpk = kDflNq / kDflNkv;
__global__ void __launch_bounds__(256) attn_kernel(const bf16* __restrict__ q, const bf16* __restrict__ kb,
                                                   const bf16* __restrict__ vb, const DflRing* __restrict__ ring, int l,
                                                   const int* __restrict__ pos, float* __restrict__ work) {
    __shared__ __align__(16) bf16 Ks[kRingPer][kDflHd];
    __shared__ __align__(16) bf16 Vs[kRingPer][kDflHd];
    __shared__ int Ps[kRingPer];
    __shared__ int nrow;
    const int kvh = blockIdx.x, s = blockIdx.y, t = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int apos = *pos, qpos = apos + t;
    int n = 0;
    if (s < kDflSplits) {
        const int r0 = s * kRingPer, r1 = min(r0 + kRingPer, kDflRing);
        // ordered compaction (deterministic drafts): rows r0 + threadIdx.x, kRingPer <= blockDim.x
        __shared__ int wcnt[8];
        const int* tag = ring->tag;
        const int r = r0 + threadIdx.x;
        const int p = r < r1 ? tag[r] : -1;
        const bool vis = p >= 0 && p < apos && apos - p < kDflWin;  // visible to row 0 at least
        const unsigned bal = __ballot_sync(0xffffffffu, vis);
        if (lane == 0) wcnt[t] = __popc(bal);
        __syncthreads();
        int off = 0;
        for (int w = 0; w < t; w++) off += wcnt[w];
        if (vis) Ps[off + __popc(bal & ((1u << lane) - 1))] = r;
        if (threadIdx.x == 0) {
            int tot = 0;
            for (int w = 0; w < 8; w++) tot += wcnt[w];
            nrow = tot;
        }
        __syncthreads();
        n = nrow;
        const bf16* K = ring->k[l];
        const bf16* V = ring->v[l];
        for (int i = threadIdx.x; i < n * (kDflHd / 8); i += blockDim.x) {
            const int j = i / (kDflHd / 8), c = (i % (kDflHd / 8)) * 8;
            const size_t o = ((size_t)Ps[j] * kDflNkv + kvh) * kDflHd + c;
            *reinterpret_cast<uint4*>(&Ks[j][c]) = *reinterpret_cast<const uint4*>(K + o);
            *reinterpret_cast<uint4*>(&Vs[j][c]) = *reinterpret_cast<const uint4*>(V + o);
        }
        __syncthreads();
        // positions of the listed rows (for the per-row window bound)
        for (int j = threadIdx.x; j < n; j += blockDim.x) Ps[j] = tag[Ps[j]];
        __syncthreads();
    } else {
        n = kDflBlock;
        for (int i = threadIdx.x; i < n * (kDflHd / 8); i += blockDim.x) {
            const int j = i / (kDflHd / 8), c = (i % (kDflHd / 8)) * 8;
            const size_t o = ((size_t)j * kDflNkv + kvh) * kDflHd + c;
            *reinterpret_cast<uint4*>(&Ks[j][c]) = *reinterpret_cast<const uint4*>(kb + o);
            *reinterpret_cast<uint4*>(&Vs[j][c]) = *reinterpret_cast<const uint4*>(vb + o);
        }
        if (threadIdx.x < kDflBlock) Ps[threadIdx.x] = -1;  // (always visible)
        __syncthreads();
    }
    float qv[kQpk][4];
#pragma unroll
    for (int hq = 0; hq < kQpk; hq++) {
        const bf16* qq = q + ((size_t)t * kDflNq + kvh * kQpk + hq) * kDflHd + lane * 4;
        const bf162 a = *reinterpret_cast<const bf162*>(qq), b = *reinterpret_cast<const bf162*>(qq + 2);
        qv[hq][0] = __low2float(a), qv[hq][1] = __high2float(a), qv[hq][2] = __low2float(b), qv[hq][3] = __high2float(b);
    }
    const float scale = rsqrtf((float)kDflHd);
    float m[kQpk], ls[kQpk], acc[kQpk][4];
#pragma unroll
    for (int hq = 0; hq < kQpk; hq++) {
        m[hq] = -INFINITY, ls[hq] = 0.f;
#pragma unroll
        for (int i = 0; i < 4; i++) acc[hq][i] = 0.f;
    }
    for (int j = 0; j < n; j++) {
        const int p = Ps[j];
        if (p >= 0 && qpos - p >= kDflWin) continue;
        const bf162 ka = *reinterpret_cast<const bf162*>(&Ks[j][lane * 4]), kb2 = *reinterpret_cast<const bf162*>(&Ks[j][lane * 4 + 2]);
        const float k0 = __low2float(ka), k1 = __high2float(ka), k2 = __low2float(kb2), k3 = __high2float(kb2);
        float dt[kQpk];
#pragma unroll
        for (int hq = 0; hq < kQpk; hq++) dt[hq] = qv[hq][0] * k0 + qv[hq][1] * k1 + qv[hq][2] * k2 + qv[hq][3] * k3;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1)
#pragma unroll
            for (int hq = 0; hq < kQpk; hq++) dt[hq] += __shfl_xor_sync(0xffffffffu, dt[hq], o);
        const bf162 va = *reinterpret_cast<const bf162*>(&Vs[j][lane * 4]), vb2 = *reinterpret_cast<const bf162*>(&Vs[j][lane * 4 + 2]);
        const float v0 = __low2float(va), v1 = __high2float(va), v2 = __low2float(vb2), v3 = __high2float(vb2);
#pragma unroll
        for (int hq = 0; hq < kQpk; hq++) {
            const float sc = dt[hq] * scale, mn = fmaxf(m[hq], sc), c0 = __expf(m[hq] - mn), c1 = __expf(sc - mn);
            acc[hq][0] = acc[hq][0] * c0 + c1 * v0;
            acc[hq][1] = acc[hq][1] * c0 + c1 * v1;
            acc[hq][2] = acc[hq][2] * c0 + c1 * v2;
            acc[hq][3] = acc[hq][3] * c0 + c1 * v3;
            ls[hq] = ls[hq] * c0 + c1;
            m[hq] = mn;
        }
    }
#pragma unroll
    for (int hq = 0; hq < kQpk; hq++) {
        float* w = work + (((size_t)s * kDflNq + kvh * kQpk + hq) * kDflBlock + t) * kAttnPart;
#pragma unroll
        for (int i = 0; i < 4; i++) w[lane * 4 + i] = acc[hq][i];
        if (lane == 0) w[kDflHd] = m[hq], w[kDflHd + 1] = ls[hq];
    }
}

__global__ void __launch_bounds__(128) attn_combine_kernel(const float* __restrict__ work, bf16* __restrict__ out) {
    const int h = blockIdx.x, t = blockIdx.y, d = threadIdx.x;
    float M = -INFINITY;
    for (int s = 0; s <= kDflSplits; s++) {
        const float* w = work + (((size_t)s * kDflNq + h) * kDflBlock + t) * kAttnPart;
        if (w[kDflHd + 1] > 0.f) M = fmaxf(M, w[kDflHd]);
    }
    float a = 0.f, l = 0.f;
    for (int s = 0; s <= kDflSplits; s++) {
        const float* w = work + (((size_t)s * kDflNq + h) * kDflBlock + t) * kAttnPart;
        if (!(w[kDflHd + 1] > 0.f)) continue;
        const float f = __expf(w[kDflHd] - M);
        a += w[d] * f;
        l += w[kDflHd + 1] * f;
    }
    out[(size_t)t * kDflNq * kDflHd + h * kDflHd + d] = __float2bfloat16(a / l);
}

__global__ void ring_inval_kernel(const DflRing* __restrict__ ring, const int* __restrict__ from, int from_host) {
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= kDflRing) return;
    const int f = from_host >= 0 ? from_host : *from;
    if (ring->tag[r] >= f) ring->tag[r] = -1;
}

// top-16 of the values a block holds (E per thread; ids unique among finite values): out in descending order
template <int E>
__device__ void block_top16(float (&v)[E], int (&id)[E], float* outv, int* outid) {
    __shared__ float sv[32];
    __shared__ int si[32];
    __shared__ int win;
    const int lane = threadIdx.x & 31, wp = threadIdx.x >> 5, nw = blockDim.x >> 5;
    for (int rd = 0; rd < kDflTopK; rd++) {
        float bv = -INFINITY;
        int bi = INT_MAX;
#pragma unroll
        for (int e = 0; e < E; e++)
            if (v[e] > bv || (v[e] == bv && id[e] < bi)) bv = v[e], bi = id[e];
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
            const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
            if (ov > bv || (ov == bv && oi < bi)) bv = ov, bi = oi;
        }
        if (lane == 0) sv[wp] = bv, si[wp] = bi;
        __syncthreads();
        if (threadIdx.x == 0) {
            for (int i = 1; i < nw; i++)
                if (sv[i] > bv || (sv[i] == bv && si[i] < bi)) bv = sv[i], bi = si[i];
            outv[rd] = bv;
            outid[rd] = bi;
            win = bi;
        }
        __syncthreads();
        const int wn = win;
#pragma unroll
        for (int e = 0; e < E; e++)
            if (id[e] == wn) v[e] = -INFINITY, id[e] = INT_MAX;
        __syncthreads();
    }
}

constexpr int kTkChunk = 2048, kTkE = kTkChunk / 256;

// stage 1: grid (rows, chunks) -> chunk [row][chunk][16][2]
__global__ void __launch_bounds__(256) topk_chunk_kernel(const float* __restrict__ logits, int v, const int* __restrict__ map,
                                                         int off, float* __restrict__ chunk) {
    const int r = blockIdx.x, c = blockIdx.y;
    float val[kTkE];
    int id[kTkE];
#pragma unroll
    for (int e = 0; e < kTkE; e++) {
        const int i = c * kTkChunk + e * 256 + threadIdx.x;
        if (i < v) val[e] = logits[(size_t)r * v + i], id[e] = map ? map[i] : off + i;
        else val[e] = -INFINITY, id[e] = INT_MAX;
    }
    __shared__ float ov[kDflTopK];
    __shared__ int oi[kDflTopK];
    block_top16<kTkE>(val, id, ov, oi);
    if (threadIdx.x < kDflTopK) {
        float* o = chunk + (((size_t)r * gridDim.y + c) * kDflTopK + threadIdx.x) * 2;
        o[0] = ov[threadIdx.x];
        o[1] = __int_as_float(oi[threadIdx.x]);
    }
}

// stage 2 / merge: candidates of row r at src[((r * rstride + g * gstride) * 16 + k) * 2] for g < groups, k < 16
// -> pairs [row][16][2] or (unary, cand)
template <int E>
__global__ void __launch_bounds__(256) topk_merge_kernel(const float* __restrict__ src, int groups, int rstride, int gstride,
                                                         float* __restrict__ pairs, float* __restrict__ unary,
                                                         int* __restrict__ cand) {
    const int r = blockIdx.x;
    float val[E];
    int id[E];
#pragma unroll
    for (int e = 0; e < E; e++) {
        const int i = e * 256 + threadIdx.x, g = i / kDflTopK, k = i % kDflTopK;
        if (g < groups) {
            const float* p = src + (((size_t)r * rstride + (size_t)g * gstride) * kDflTopK + k) * 2;
            val[e] = p[0], id[e] = __float_as_int(p[1]);
        } else {
            val[e] = -INFINITY, id[e] = INT_MAX;
        }
    }
    __shared__ float ov[kDflTopK];
    __shared__ int oi[kDflTopK];
    block_top16<E>(val, id, ov, oi);
    if (threadIdx.x < kDflTopK) {
        if (pairs) {
            float* o = pairs + ((size_t)r * kDflTopK + threadIdx.x) * 2;
            o[0] = ov[threadIdx.x];
            o[1] = __int_as_float(oi[threadIdx.x]);
        } else {
            unary[r * kDflTopK + threadIdx.x] = ov[threadIdx.x];
            cand[r * kDflTopK + threadIdx.x] = oi[threadIdx.x];
        }
    }
}

__global__ void cb_fetch_kernel(const bf16* __restrict__ pred_cb, const bf16* __restrict__ succ_cb, const int* __restrict__ anchor,
                                const int* __restrict__ cand, int nc, bf16* __restrict__ pc, bf16* __restrict__ sc, int V) {
    const int b = blockIdx.x;
    auto cl = [V](int t) { return t >= 0 && t < V ? t : 0; };  // (a padded / NaN row's INT_MAX id: review L3)
    const uint4* src;
    uint4* dst;
    if (b <= nc) {
        const int tk = cl(b == 0 ? *anchor : cand[b - 1]);
        src = reinterpret_cast<const uint4*>(pred_cb + (size_t)tk * kDflRank);
        dst = reinterpret_cast<uint4*>(pc + (size_t)b * kDflRank);
    } else {
        const int i = b - nc - 1;
        src = reinterpret_cast<const uint4*>(succ_cb + (size_t)cl(cand[i]) * kDflRank);
        dst = reinterpret_cast<uint4*>(sc + (size_t)i * kDflRank);
    }
    dst[threadIdx.x] = src[threadIdx.x];  // 32 x 16 B = one row
}

__global__ void __launch_bounds__(256) select_kernel(const float* __restrict__ unary, const int* __restrict__ cand,
                                                     const float* __restrict__ hp, const bf16* __restrict__ pc,
                                                     const bf16* __restrict__ sc, int m, int* __restrict__ tokens, int V) {
    __shared__ float part[8][kDflTopK];
    __shared__ int best;
    const int r = threadIdx.x, lane = r & 31, wp = r >> 5;
    int prow = 0;
    for (int j = 0; j < m; j++) {
        const float a = __bfloat162float(pc[(size_t)prow * kDflRank + r]) * bfr(hp[j * kDflRank + r]);
        float s[kDflTopK];
#pragma unroll
        for (int k = 0; k < kDflTopK; k++) s[k] = a * __bfloat162float(sc[((size_t)j * kDflTopK + k) * kDflRank + r]);
#pragma unroll
        for (int k = 0; k < kDflTopK; k++)
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) s[k] += __shfl_xor_sync(0xffffffffu, s[k], o);
        if (lane == 0)
#pragma unroll
            for (int k = 0; k < kDflTopK; k++) part[wp][k] = s[k];
        __syncthreads();
        if (r == 0) {
            int bk = 0;
            float bs = -INFINITY;
            for (int k = 0; k < kDflTopK; k++) {
                float x = unary[j * kDflTopK + k];
                for (int w = 0; w < 8; w++) x += part[w][k];
                if (x > bs) bs = x, bk = k;
            }
            best = bk;
            const int c = cand[j * kDflTopK + bk];
            tokens[1 + j] = c >= 0 && c < V ? c : tokens[0];  // (never an id past the vocabulary: review L3)
        }
        __syncthreads();
        prow = 1 + j * kDflTopK + best;
        __syncthreads();
    }
}

__global__ void set_tok_kernel(int* tok, const int* src) {
    const int i = threadIdx.x;
    if (i < kDflBlock) tok[i] = i == 0 ? *src : kDflMask;
}

}  // namespace

void dfl_tap(const float* resid, int m, int j, bf16* tap, int row0, cudaStream_t st) {
    if (m > 0) tap_kernel<<<m, 256, 0, st>>>(resid, j, tap, row0);
}
void dfl_rms(const float* x, int ldx, const bf16* w, int m, float eps, bf16* out, cudaStream_t st) {
    if (m > 0) rms_kernel<<<m, kRmsThreads, 0, st>>>(x, ldx, w, eps, out);
}
void dfl_conv(const float* xf, const bf16* xb, const float* dyn, int c, const bf16* base, int m, bf16* out, float* add,
              cudaStream_t st) {
    conv_kernel<<<dim3(m, kDflHid / 256), 256, 0, st>>>(xf, xb, dyn, c, base, out, add);
}
void dfl_qkv_block(const float* proj, const bf16* qn, const bf16* kn, const float* inv, const int* pos, int m, float eps,
                   bf16* q, bf16* k, bf16* v, cudaStream_t st) {
    qkv_block_kernel<<<dim3(m, kDflNq + 2 * kDflNkv), 128, 0, st>>>(proj, qn, kn, inv, pos, eps, q, k, v);
}
void dfl_kv_ctx(const float* kv, const bf16* kn, const float* inv, const int* pos, int row0, int m, float eps,
                const DflRing* ring, int l, bool tags, cudaStream_t st) {
    if (m > 0) kv_ctx_kernel<<<dim3(m, 2 * kDflNkv), 128, 0, st>>>(kv, kn, inv, pos, row0, eps, ring, l, tags ? 1 : 0);
}
size_t dfl_attn_work_floats() { return (size_t)(kDflSplits + 1) * kDflNq * kDflBlock * kAttnPart; }
void dfl_attn(const bf16* q, const bf16* kb, const bf16* vb, const DflRing* ring, int l, const int* pos, float* work,
              bf16* out, cudaStream_t st) {
    attn_kernel<<<dim3(kDflNkv, kDflSplits + 1), 256, 0, st>>>(q, kb, vb, ring, l, pos, work);
    attn_combine_kernel<<<dim3(kDflNq, kDflBlock), 128, 0, st>>>(work, out);
}
void dfl_ring_inval(const DflRing* ring, const int* from, int from_host, cudaStream_t st) {
    ring_inval_kernel<<<(kDflRing + 255) / 256, 256, 0, st>>>(ring, from, from_host);
}
size_t dfl_topk_chunk_floats(int v, int m) { return (size_t)m * ((v + kTkChunk - 1) / kTkChunk) * kDflTopK * 2 + (size_t)m * kDflTopK * 2; }
void dfl_topk_local(const float* logits, int v, int m, const int* map, int off, float* chunk, float* pairs, cudaStream_t st) {
    const int nc = (v + kTkChunk - 1) / kTkChunk;
    CHECK(nc * kDflTopK <= 2048, "dfl_topk_local: vocabulary share %d too large", v);
    topk_chunk_kernel<<<dim3(m, nc), 256, 0, st>>>(logits, v, map, off, chunk);
    // stage 2: the row's nc x 16 candidates (chunk layout [row][chunk][16])
    if (nc * kDflTopK <= 256) topk_merge_kernel<1><<<m, 256, 0, st>>>(chunk, nc, nc, 1, pairs, nullptr, nullptr);
    else if (nc * kDflTopK <= 1024) topk_merge_kernel<4><<<m, 256, 0, st>>>(chunk, nc, nc, 1, pairs, nullptr, nullptr);
    else topk_merge_kernel<8><<<m, 256, 0, st>>>(chunk, nc, nc, 1, pairs, nullptr, nullptr);
}
void dfl_topk_merge(const float* gathered, int ngpu, int m, float* unary, int* cand, cudaStream_t st) {
    topk_merge_kernel<1><<<m, 256, 0, st>>>(gathered, ngpu, 1, m, nullptr, unary, cand);
}
void dfl_cb_fetch(const bf16* pred_cb, const bf16* succ_cb, const int* anchor, const int* cand, int m, bf16* pc, bf16* sc,
                  int V, cudaStream_t st) {
    const int nc = m * kDflTopK;
    cb_fetch_kernel<<<1 + 2 * nc, 32, 0, st>>>(pred_cb, succ_cb, anchor, cand, nc, pc, sc, V);
}
void dfl_select(const float* unary, const int* cand, const float* hp, const bf16* pc, const bf16* sc, int m, int* tokens,
                int V, cudaStream_t st) {
    select_kernel<<<1, 256, 0, st>>>(unary, cand, hp, pc, sc, m, tokens, V);
}
// rank 0's drafts on every card (gathered [ngpu][8] int bits): tokens[1..7] = row 0 (review L1)
__global__ void take_row0_kernel(const float* __restrict__ g, int* __restrict__ tokens) {
    const int i = threadIdx.x;
    if (i >= 1 && i < kDflBlock) tokens[i] = __float_as_int(g[i]);
}
void dfl_take_row0(const float* gathered, int* tokens, cudaStream_t st) { take_row0_kernel<<<1, 32, 0, st>>>(gathered, tokens); }
void dfl_set_block_tokens(int* tok, const int* src, cudaStream_t st) { set_tok_kernel<<<1, 32, 0, st>>>(tok, src); }
