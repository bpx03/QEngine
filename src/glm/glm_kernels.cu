// GLM-5.3-Flash kernels (our own code; math of tools/glm/glm_model.py)
#include "glm_kernels.cuh"

#include <cfloat>

namespace glm {

// round 26 ([glm-p3-mlapos]): row strides (elements) of the MLA latent cache kc and the pooled index keys pk. Default:
// one array per layer (KVLR / ID). Position-major (GLM_MLA_POS): the rows of every MLA layer at one position side by
// side ([pos][layer][512], [pool][layer][128]); a layer's pointer is base + layer * 512 (128). Set per card at load.
__constant__ int c_kst = KVLR;
__constant__ int c_pst = ID;
void mla_set_strides(int kst, int pst) {
    CUDA_CHECK(cudaMemcpyToSymbol(c_kst, &kst, sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_pst, &pst, sizeof(int)));
}

__device__ __forceinline__ float warp_sum(float v) {
    for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
__device__ __forceinline__ float warp_max(float v) {
    for (int o = 16; o; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}
// block sum (blockDim multiple of 32, <= 1024); sh >= 32 floats
__device__ float block_sum(float v, float* sh) {
    const int w = threadIdx.x >> 5, l = threadIdx.x & 31, nw = blockDim.x >> 5;
    v = warp_sum(v);
    __syncthreads();
    if (l == 0) sh[w] = v;
    __syncthreads();
    float r = l < nw ? sh[l] : 0.f;
    r = warp_sum(r);
    return r;
}
__device__ __forceinline__ float sigm(float x) { return 1.f / (1.f + __expf(-x)); }
__device__ __forceinline__ float silu(float x) { return x / (1.f + __expf(-x)); }

// ---------------------------------------------------------------- embedding / mHC

__global__ void k_embed(const int* tok, const bf16* emb, float* R) {
    const int t = blockIdx.x;
    const bf16* e = emb + (size_t)tok[t] * H;
    float* r = R + (size_t)t * RW;
    for (int i = threadIdx.x; i < H; i += blockDim.x) {
        const float v = __bfloat162float(e[i]);
        r[i] = v; r[H + i] = v; r[2 * H + i] = v; r[3 * H + i] = v;
    }
}
void embed(const int* tok, const bf16* emb, float* R, int T, cudaStream_t st) {
    k_embed<<<T, 512, 0, st>>>(tok, emb, R);
}

// decode rows: mixraw[t][j] = fn[j] . R[t] (24 blocks per row), plain dot (the rms scale is applied in k_mhc_pre)
__global__ void k_mhc_mix(const float* R, const float* fn, float* mixraw) {
    __shared__ float sh[32];
    const int j = blockIdx.x, t = blockIdx.y;
    const float4* r = reinterpret_cast<const float4*>(R + (size_t)t * RW);
    const float4* w = reinterpret_cast<const float4*>(fn + (size_t)j * RW);
    float s = 0.f, q = 0.f;
    for (int i = threadIdx.x; i < RW / 4; i += blockDim.x) {
        const float4 a = r[i], b = w[i];
        s += a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
        if (j == 0) q += a.x * a.x + a.y * a.y + a.z * a.z + a.w * a.w;
    }
    s = block_sum(s, sh);
    if (threadIdx.x == 0) mixraw[t * MIXS + j] = s;
    if (j == 0) {
        __syncthreads();
        q = block_sum(q, sh);
        if (threadIdx.x == 0) mixraw[t * MIXS + NMIX] = q;  // sum of squares of the streams
    }
}
void mhc_mix(const float* R, const float* fn, float* mixraw, int T, cudaStream_t st) {
    k_mhc_mix<<<dim3(NMIX, T), 512, 0, st>>>(R, fn, mixraw);
}

// per token: mix = mixraw / rms(R) ; pre / post / comb (Sinkhorn) ; col = sum pre_i R_i ; x = rms(col) * ln
__global__ void k_mhc_pre(const float* R, const float* mixraw, const float* base, const float* scale, const float* ln,
                          float* post, float* comb, float* xf, bf16* xb, bool has_ss, const float* ssp, const float* mixp,
                          int nbp) {
    __shared__ float sh[32];
    __shared__ float m[NMIX], pre[4], cb[16], mp[NMIX + 1];
    const int t = blockIdx.x;
    const float* r = R + (size_t)t * RW;
    if (mixp) {  // mixes and the sum of squares from the merge's per-block partials (fixed order)
        if (threadIdx.x <= NMIX) {
            float a = 0.f;
            for (int q = 0; q < nbp; q++) a += mixp[((size_t)t * nbp + q) * (NMIX + 1) + threadIdx.x];
            mp[threadIdx.x] = a;
        }
        __syncthreads();
        has_ss = true;
    }
    const float* mr = mixp ? mp : mixraw + (size_t)t * MIXS;
    float ss;
    if (has_ss) ss = mr[NMIX];
    else if (ssp) {  // partial sums of squares from the merge (16 blocks of 256 columns, fixed order)
        ss = 0.f;
        for (int j = 0; j < H / 256; j++) ss += ssp[t * (H / 256) + j];
    } else {
        ss = 0.f;
        for (int i = threadIdx.x; i < RW; i += blockDim.x) ss += r[i] * r[i];
        ss = block_sum(ss, sh);
    }
    const float inv = rsqrtf(ss / RW + EPS);
    if (threadIdx.x < NMIX) m[threadIdx.x] = mr[threadIdx.x] * inv;
    __syncthreads();
    if (threadIdx.x < 32) {
        // lane l < 16 holds comb[i = l / 4][j = l % 4]: softmax over j, then Sinkhorn with shuffles
        const int ln = threadIdx.x, i = (ln >> 2) & 3;
        if (ln < 4) {
            pre[ln] = sigm(m[ln] * scale[0] + base[ln]) + HC_EPS;
            post[t * 4 + ln] = 2.f * sigm(m[4 + ln] * scale[1] + base[4 + ln]);
        }
        float c = m[8 + (ln & 15)] * scale[2] + base[8 + (ln & 15)];
        float mx = fmaxf(c, __shfl_xor_sync(0xffffffffu, c, 1));
        mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 2));
        c = expf(c - mx);
        float s = c + __shfl_xor_sync(0xffffffffu, c, 1);
        s += __shfl_xor_sync(0xffffffffu, s, 2);
        c = c / s + HC_EPS;
        (void)i;
        for (int it = 0; it < SINK; it++) {
            if (it > 0) {  // row pass (sum over j: lanes xor 1, 2)
                float r = c + __shfl_xor_sync(0xffffffffu, c, 1);
                r += __shfl_xor_sync(0xffffffffu, r, 2);
                c /= r + HC_EPS;
            }
            // column pass (sum over i: lanes xor 4, 8)
            float q = c + __shfl_xor_sync(0xffffffffu, c, 4);
            q += __shfl_xor_sync(0xffffffffu, q, 8);
            c /= q + HC_EPS;
        }
        if (ln < 16) { cb[ln] = c; comb[t * 16 + ln] = c; }
    }
    __syncthreads();
    float cs = 0.f;
    float* xo = xf + (size_t)t * H;
    for (int i = threadIdx.x; i < H; i += blockDim.x) {
        const float v = pre[0] * r[i] + pre[1] * r[H + i] + pre[2] * r[2 * H + i] + pre[3] * r[3 * H + i];
        xo[i] = v;
        cs += v * v;
    }
    cs = block_sum(cs, sh);
    const float iv = rsqrtf(cs / H + EPS);
    for (int i = threadIdx.x; i < H; i += blockDim.x) {
        const float v = xo[i] * iv * ln[i];
        xo[i] = v;
        xb[(size_t)t * H + i] = __float2bfloat16(v);
    }
}
// prefill variant (round 10): 256 threads per row, the row's 4 streams loaded into registers before the mixes /
// Sinkhorn (latency overlapped), x kept in registers (no write-back and re-read of xf); mixes from the merge's
// partials mixp[t][nbp][25] (index 24: sum of squares). Same arithmetic as k_mhc_pre except the order of the rms sum.
__global__ void __launch_bounds__(256) k_mhc_pre_pf(const float* R, const float* base, const float* scale,
                                                    const float* ln, float* post, float* comb, float* xf, bf16* xb,
                                                    const float* mixp, int nbp) {
    __shared__ float sh[32];
    __shared__ float m[NMIX], pre[4], mp[NMIX + 1];
    const int t = blockIdx.x, tid = threadIdx.x;
    const float4* r4 = reinterpret_cast<const float4*>(R + (size_t)t * RW);
    float4 x[4][4];  // [stream][k]: columns 4 tid + 1024 k
#pragma unroll
    for (int i = 0; i < 4; i++)
#pragma unroll
        for (int k = 0; k < 4; k++) x[i][k] = r4[i * (H / 4) + tid + 256 * k];
    if (tid <= NMIX) {
        float a = 0.f;
        for (int q = 0; q < nbp; q++) a += mixp[((size_t)t * nbp + q) * (NMIX + 1) + tid];
        mp[tid] = a;
    }
    __syncthreads();
    const float inv = rsqrtf(mp[NMIX] / RW + EPS);
    if (tid < NMIX) m[tid] = mp[tid] * inv;
    __syncthreads();
    if (tid < 32) {
        const int lnn = tid;
        if (lnn < 4) {
            pre[lnn] = sigm(m[lnn] * scale[0] + base[lnn]) + HC_EPS;
            post[t * 4 + lnn] = 2.f * sigm(m[4 + lnn] * scale[1] + base[4 + lnn]);
        }
        float c = m[8 + (lnn & 15)] * scale[2] + base[8 + (lnn & 15)];
        float mx = fmaxf(c, __shfl_xor_sync(0xffffffffu, c, 1));
        mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 2));
        c = expf(c - mx);
        float s = c + __shfl_xor_sync(0xffffffffu, c, 1);
        s += __shfl_xor_sync(0xffffffffu, s, 2);
        c = c / s + HC_EPS;
        for (int it = 0; it < SINK; it++) {
            if (it > 0) {
                float rr = c + __shfl_xor_sync(0xffffffffu, c, 1);
                rr += __shfl_xor_sync(0xffffffffu, rr, 2);
                c /= rr + HC_EPS;
            }
            float q = c + __shfl_xor_sync(0xffffffffu, c, 4);
            q += __shfl_xor_sync(0xffffffffu, q, 8);
            c /= q + HC_EPS;
        }
        if (lnn < 16) comb[t * 16 + lnn] = c;
    }
    __syncthreads();
    const float p0 = pre[0], p1 = pre[1], p2 = pre[2], p3 = pre[3];
    float4 v[4];
    float cs = 0.f;
#pragma unroll
    for (int k = 0; k < 4; k++) {
        v[k].x = p0 * x[0][k].x + p1 * x[1][k].x + p2 * x[2][k].x + p3 * x[3][k].x;
        v[k].y = p0 * x[0][k].y + p1 * x[1][k].y + p2 * x[2][k].y + p3 * x[3][k].y;
        v[k].z = p0 * x[0][k].z + p1 * x[1][k].z + p2 * x[2][k].z + p3 * x[3][k].z;
        v[k].w = p0 * x[0][k].w + p1 * x[1][k].w + p2 * x[2][k].w + p3 * x[3][k].w;
        cs += v[k].x * v[k].x + v[k].y * v[k].y + v[k].z * v[k].z + v[k].w * v[k].w;
    }
    cs = block_sum(cs, sh);
    const float iv = rsqrtf(cs / H + EPS);
    const float4* l4 = reinterpret_cast<const float4*>(ln);
    float4* xo = reinterpret_cast<float4*>(xf + (size_t)t * H);
    uint2* xbo = reinterpret_cast<uint2*>(xb + (size_t)t * H);
#pragma unroll
    for (int k = 0; k < 4; k++) {
        const int c4 = tid + 256 * k;
        const float4 w = l4[c4];
        const float4 o = make_float4(v[k].x * iv * w.x, v[k].y * iv * w.y, v[k].z * iv * w.z, v[k].w * iv * w.w);
        xo[c4] = o;
        const bf162 a = __floats2bfloat162_rn(o.x, o.y), b = __floats2bfloat162_rn(o.z, o.w);
        xbo[c4] = make_uint2(*reinterpret_cast<const unsigned*>(&a), *reinterpret_cast<const unsigned*>(&b));
    }
}
void mhc_pre_pf(const float* R, const float* base, const float* scale, const float* ln, float* post, float* comb,
                float* xf, bf16* xb, int T, const float* mixp, int nbp, cudaStream_t st) {
    k_mhc_pre_pf<<<T, 256, 0, st>>>(R, base, scale, ln, post, comb, xf, xb, mixp, nbp);
}
void mhc_pre(const float* R, const float* mixraw, const float* base, const float* scale, const float* ln, float* post,
             float* comb, float* xf, bf16* xb, int T, cudaStream_t st, bool has_ss, const float* ssp, const float* mixp,
             int nbp) {
    k_mhc_pre<<<T, 1024, 0, st>>>(R, mixraw, base, scale, ln, post, comb, xf, xb, has_ss, ssp, mixp, nbp);
}

// ---------------------------------------------------------------- fused decode block boundary
// grid barrier on flags (all blocks are resident: BNB blocks of 256 threads, 2 per SM): the last arriver resets the
// count and bumps the generation
__device__ __forceinline__ void gbar(unsigned* cnt, unsigned* gen, unsigned nb) {
    __syncthreads();
    if (threadIdx.x == 0) {
        const unsigned g = *(volatile unsigned*)gen;
        __threadfence();
        if (atomicAdd(cnt, 1u) == nb - 1) {
            *(volatile unsigned*)cnt = 0;
            __threadfence();
            atomicAdd(gen, 1u);
        } else {
            while (*(volatile unsigned*)gen == g) {}
        }
        __threadfence();
    }
    __syncthreads();
}
constexpr int BNB = 128, BCOL = H / BNB;  // blocks, columns per block (x 4 streams)
constexpr int BNP = NMIX + 1 + 10;          // partials per row and block: 24 mixes, sum of squares, Gram of the streams
// one decode block boundary for T <= 8 rows: R <- merge(R, post, comb, a) of the finished block, then the next block's
// mHC (mixes, Sinkhorn, pre / post / comb, x = sum pre_i R_i, rms * ln) -> post, comb, xf, xb. Replaces k_merge +
// k_mhc_mix + k_mhc_pre (3 launches, single-block mhc_pre per row) by one kernel with one grid barrier: the rms of x
// comes from the 4 x 4 Gram matrix of the new streams (|x|^2 = pre^T G pre), so no second reduction is needed.
__global__ void __launch_bounds__(256, 2) k_boundary(float* R, float* post, float* comb, const float* a, const float* fn,
                                                     const float* base, const float* scale, const float* ln, float* xf,
                                                     bf16* xb, int T, float* mixp, float* ssx, unsigned* bar, int dbg) {
    __shared__ float ys[8][4][BCOL];
    __shared__ float fs[NMIX][4][BCOL];
    __shared__ float pr[8][4], ivs[8];
    (void)ssx;
    const int b = blockIdx.x, c0 = b * BCOL, tid = threadIdx.x, w = tid >> 5, ln_ = tid & 31;
    for (int i = tid; i < NMIX * 4 * BCOL; i += 256) {
        const int j = i / (4 * BCOL), s = (i / BCOL) % 4, c = i % BCOL;
        fs[j][s][c] = fn[(size_t)j * RW + s * H + c0 + c];
    }
    for (int i = tid; i < T * 4 * BCOL; i += 256) {
        const int t = i / (4 * BCOL), k = (i / BCOL) % 4, c = i % BCOL, col = c0 + c;
        const float* r = R + (size_t)t * RW;
        const float* cb = comb + t * 16;
        ys[t][k][c] = post[t * 4 + k] * a[(size_t)t * H + col] + cb[k] * r[col] + cb[4 + k] * r[H + col] +
                      cb[8 + k] * r[2 * H + col] + cb[12 + k] * r[3 * H + col];
    }
    __syncthreads();
    for (int i = tid; i < T * 4 * BCOL; i += 256) {
        const int t = i / (4 * BCOL), k = (i / BCOL) % 4, c = i % BCOL;
        R[(size_t)t * RW + k * H + c0 + c] = ys[t][k][c];
    }
    // partials: warp w takes items w, w + 8, ... of (row, partial); lane = column
    for (int it = w; it < T * BNP; it += 8) {
        const int t = it / BNP, j = it % BNP;
        float v;
        if (j < NMIX)
            v = fs[j][0][ln_] * ys[t][0][ln_] + fs[j][1][ln_] * ys[t][1][ln_] + fs[j][2][ln_] * ys[t][2][ln_] +
                fs[j][3][ln_] * ys[t][3][ln_];
        else if (j == NMIX)
            v = ys[t][0][ln_] * ys[t][0][ln_] + ys[t][1][ln_] * ys[t][1][ln_] + ys[t][2][ln_] * ys[t][2][ln_] +
                ys[t][3][ln_] * ys[t][3][ln_];
        else {  // Gram entry (p, q), p <= q
            const int g = j - NMIX - 1;
            const int p = g < 4 ? 0 : g < 7 ? 1 : g < 9 ? 2 : 3;
            const int q = g < 4 ? g : g < 7 ? g - 3 : g < 9 ? g - 5 : 3;
            v = ys[t][p][ln_] * ys[t][q][ln_];
        }
        for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
        if (ln_ == 0) mixp[((size_t)t * BNP + j) * BNB + b] = v;
    }
    if (dbg == 1) return;
    gbar(bar, bar + 1, gridDim.x);
    if (dbg == 2) return;
    __shared__ float tot[8][BNP];
    // each item (row, partial) is summed by one block over the 128 block partials (fixed order), then shared
    float* totg = mixp + (size_t)8 * BNP * BNB;
    for (int it = b; it < T * BNP; it += gridDim.x)
        if (w == 0) {
            const float* src = mixp + (size_t)it * BNB;
            float v = (src[ln_] + src[32 + ln_]) + (src[64 + ln_] + src[96 + ln_]);
            for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
            if (ln_ == 0) totg[it] = v;
        }
    gbar(bar, bar + 1, gridDim.x);
    for (int i = tid; i < T * BNP; i += 256) tot[i / BNP][i % BNP] = totg[i];
    __syncthreads();
    if (dbg == 3) return;
    if (w < T) {
        const int t = w;
        const float inv = rsqrtf(tot[t][NMIX] / RW + EPS);
        const float m_pre = tot[t][ln_ & 3] * inv, m_post = tot[t][4 + (ln_ & 3)] * inv;
        const float m_c = tot[t][8 + (ln_ & 15)] * inv;
        if (ln_ < 4) {
            pr[t][ln_] = sigm(m_pre * scale[0] + base[ln_]) + HC_EPS;
            if (b == 0) post[t * 4 + ln_] = 2.f * sigm(m_post * scale[1] + base[4 + ln_]);
        }
        float c = m_c * scale[2] + base[8 + (ln_ & 15)];
        float mx = fmaxf(c, __shfl_xor_sync(0xffffffffu, c, 1));
        mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 2));
        c = expf(c - mx);
        float sm = c + __shfl_xor_sync(0xffffffffu, c, 1);
        sm += __shfl_xor_sync(0xffffffffu, sm, 2);
        c = c / sm + HC_EPS;
        for (int itr = 0; itr < SINK; itr++) {
            if (itr > 0) {
                float rr = c + __shfl_xor_sync(0xffffffffu, c, 1);
                rr += __shfl_xor_sync(0xffffffffu, rr, 2);
                c /= rr + HC_EPS;
            }
            float qq = c + __shfl_xor_sync(0xffffffffu, c, 4);
            qq += __shfl_xor_sync(0xffffffffu, qq, 8);
            c /= qq + HC_EPS;
        }
        if (b == 0 && ln_ < 16) comb[t * 16 + ln_] = c;
        __syncwarp();
        if (ln_ == 0) {  // |x|^2 = pre^T G pre from the Gram entries (upper triangle)
            float xx = 0.f;
            for (int g = 0; g < 10; g++) {
                const int p = g < 4 ? 0 : g < 7 ? 1 : g < 9 ? 2 : 3;
                const int q = g < 4 ? g : g < 7 ? g - 3 : g < 9 ? g - 5 : 3;
                xx += (p == q ? 1.f : 2.f) * pr[t][p] * pr[t][q] * tot[t][NMIX + 1 + g];
            }
            ivs[t] = rsqrtf(xx / H + EPS);
        }
    }
    __syncthreads();
    if (dbg == 4) return;
    for (int i = tid; i < T * BCOL; i += 256) {
        const int t = i / BCOL, c = i % BCOL;
        const float x = pr[t][0] * ys[t][0][c] + pr[t][1] * ys[t][1][c] + pr[t][2] * ys[t][2][c] + pr[t][3] * ys[t][3][c];
        const float v = x * ivs[t] * ln[c0 + c];
        xf[(size_t)t * H + c0 + c] = v;
        xb[(size_t)t * H + c0 + c] = __float2bfloat16(v);
    }
}
void boundary(float* R, float* post, float* comb, const float* a, const float* fn, const float* base, const float* scale,
              const float* ln, float* xf, bf16* xb, int T, float* scratch, unsigned* bar, cudaStream_t st) {
    float* mixp = scratch;
    float* ssx = nullptr;
    static const int dbg = [] { const char* e = getenv("GLM_BND_DBG"); return e ? atoi(e) : 0; }();
    k_boundary<<<BNB, 256, 0, st>>>(R, post, comb, a, fn, base, scale, ln, xf, xb, T, mixp, ssx, bar, dbg);
}

// R_i <- post_i a + sum_j comb[j][i] R_j
__global__ void k_merge(float* R, const float* post, const float* comb, const float* a, float* ssp) {
    __shared__ float sh[32];
    const int t = blockIdx.y;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    float* r = R + (size_t)t * RW;
    const float* c = comb + t * 16;
    const float* p = post + t * 4;
    const float x0 = r[i], x1 = r[H + i], x2 = r[2 * H + i], x3 = r[3 * H + i];
    const float av = a[(size_t)t * H + i];
    float q = 0.f;
#pragma unroll
    for (int k = 0; k < 4; k++) {
        const float y = p[k] * av + c[k] * x0 + c[4 + k] * x1 + c[8 + k] * x2 + c[12 + k] * x3;
        r[k * H + i] = y;
        q += y * y;
    }
    if (ssp) {  // this block's part of the sum of squares of the new streams (the next mhc_pre skips its own pass)
        q = block_sum(q, sh);
        if (threadIdx.x == 0) ssp[t * (H / 256) + blockIdx.x] = q;
    }
}
void merge(float* R, const float* post, const float* comb, const float* a, int T, cudaStream_t st, float* ssp) {
    k_merge<<<dim3(H / 256, T), 256, 0, st>>>(R, post, comb, a, ssp);
}
// decode: merge + the next block's mHC mixes (24 dots over the 4 x 4096 new streams) as per-block partials
// mixp[t][block][25] (index 24: sum of squares); k_mhc_pre sums them, so k_mhc_mix and its pass over R go away
constexpr int MMZ = 4, MMJ = NMIX / MMZ;  // mixes split over gridDim.z (more blocks, shorter reductions)
__global__ void __launch_bounds__(256) k_merge_mix(float* R, const float* post, const float* comb, const float* a,
                                                   const float* fn, float* mixp) {
    __shared__ float red[8][MMJ + 1];
    const int t = blockIdx.y, z = blockIdx.z, i = blockIdx.x * blockDim.x + threadIdx.x, w = threadIdx.x >> 5,
              ln = threadIdx.x & 31;
    float* r = R + (size_t)t * RW;
    const float* c = comb + t * 16;
    const float* p = post + t * 4;
    const float x0 = r[i], x1 = r[H + i], x2 = r[2 * H + i], x3 = r[3 * H + i];
    const float av = a[(size_t)t * H + i];
    float y[4], q = 0.f;
#pragma unroll
    for (int k = 0; k < 4; k++) {
        y[k] = p[k] * av + c[k] * x0 + c[4 + k] * x1 + c[8 + k] * x2 + c[12 + k] * x3;
        q += y[k] * y[k];
    }
    float d[MMJ + 1];
#pragma unroll
    for (int jj = 0; jj < MMJ; jj++) {
        const float* f = fn + (size_t)(z * MMJ + jj) * RW + i;
        d[jj] = f[0] * y[0] + f[H] * y[1] + f[2 * H] * y[2] + f[3 * H] * y[3];
    }
    d[MMJ] = q;
#pragma unroll
    for (int jj = 0; jj <= MMJ; jj++) {
        float v = d[jj];
        for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
        if (ln == 0) red[w][jj] = v;
    }
    __syncthreads();
    float* mo = mixp + ((size_t)t * (H / 256) + blockIdx.x) * (NMIX + 1);
    if (threadIdx.x < MMJ || (z == 0 && threadIdx.x == MMJ)) {
        float v = 0.f;
        for (int k = 0; k < 8; k++) v += red[k][threadIdx.x];
        mo[threadIdx.x < MMJ ? z * MMJ + threadIdx.x : NMIX] = v;
    }
}
void merge_mix(float* R, const float* post, const float* comb, const float* a, const float* fn, float* mixp, int T,
               cudaStream_t st) {
    // pass 1: the mixes from the merged values (R untouched), pass 2: the merge itself
    k_merge_mix<<<dim3(H / 256, T, MMZ), 256, 0, st>>>(R, post, comb, a, fn, mixp);
    k_merge<<<dim3(H / 256, T), 256, 0, st>>>(R, post, comb, a, nullptr);
}

// final head input: rms(mean of streams) * w -> bf16
__global__ void k_final(const float* R, const float* w, bf16* xb, float* hf) {
    __shared__ float sh[32];
    const int t = blockIdx.x;
    const float* r = R + (size_t)t * RW;
    float ss = 0.f;
    for (int i = threadIdx.x; i < H; i += blockDim.x) {
        const float v = 0.25f * (r[i] + r[H + i] + r[2 * H + i] + r[3 * H + i]);
        ss += v * v;
    }
    ss = block_sum(ss, sh);
    const float iv = rsqrtf(ss / H + EPS);
    for (int i = threadIdx.x; i < H; i += blockDim.x) {
        const float v = 0.25f * (r[i] + r[H + i] + r[2 * H + i] + r[3 * H + i]) * iv * w[i];
        xb[(size_t)t * H + i] = __float2bfloat16(v);
        if (hf) hf[(size_t)t * H + i] = v;
    }
}
void final_norm(const float* R, const float* w, bf16* xb, float* hf, int T, cudaStream_t st) {
    k_final<<<T, 512, 0, st>>>(R, w, xb, hf);
}

// ---------------------------------------------------------------- KDA

// grid (T, nh), 128 threads (channel d). proj rows [q nh*128][k nh*128][v nh*128][fa 128][ga 128][b nh], ld = ldp
__global__ void k_kda_prep(const float* proj, int ldp, int nh, const float* convw, const float* hist, const float* gf,
                           const float* A, const float* dt, float* q, float* k, float* v, float* g, float* beta, int r0,
                           const float* prev, float* rows_out) {
    __shared__ float sh[32];
    const int t = blockIdx.x, h = blockIdx.y, d = threadIdx.x;
    const int C = 3 * nh * KD;
    float o[3];
#pragma unroll
    for (int s = 0; s < 3; s++) {
        const int c = s * nh * KD + h * KD + d;
        float acc = 0.f;
#pragma unroll
        for (int j = 0; j < CONV; j++) {
            const int tt = t + j - (CONV - 1);  // input index: xx[t + j] = hist (tt < 0) or proj row tt
            const int ta = r0 + tt;             // verify group r0..: earlier rows of the step from prev (conv_rows)
            const float x = tt >= 0 ? proj[(size_t)tt * ldp + c]
                                    : (ta < 0 ? hist[(size_t)(CONV - 1 + ta) * C + c] : prev[(size_t)ta * C + c]);
            acc += convw[c * CONV + j] * x;
        }
        o[s] = silu(acc);
        if (rows_out) rows_out[(size_t)t * C + c] = proj[(size_t)t * ldp + c];  // verify rows: conv inputs for the commit
    }
    const float qq = block_sum(o[0] * o[0], sh);
    const float kk = block_sum(o[1] * o[1], sh);
    const size_t i = ((size_t)t * nh + h) * KD + d;
    q[i] = o[0] / sqrtf(qq + 1e-6f) * rsqrtf((float)KD);
    k[i] = o[1] / sqrtf(kk + 1e-6f);
    v[i] = o[2];
    const float x = __expf(A[h]) * (gf[(size_t)t * nh * KD + h * KD + d] + dt[h * KD + d]);
    g[i] = __expf(GLB * sigm(x));  // the decay factor itself (k_kda_rec multiplies by it)
    if (d == 0) beta[t * nh + h] = sigm(proj[(size_t)t * ldp + 3 * nh * KD + 2 * KD + h]);
}
// decode verify rows: kda_prep with the f gate GEMV inline (gf = f_b f_a, f_b transposed [128][nh 128]) and the conv
// inputs of the rows written for the commit (replaces cuBLAS sgemm + k_conv_rows)
__global__ void k_kda_prep_f(const float* proj, int ldp, int nh, const float* convw, const float* hist, const float* fbT,
                             const float* A, const float* dt, float* q, float* k, float* v, float* g, float* beta, int r0,
                             const float* prev, float* rows_out) {
    __shared__ float sh[32], fa[KD];
    const int t = blockIdx.x, h = blockIdx.y, d = threadIdx.x;
    const int C = 3 * nh * KD, HK = nh * KD;
    fa[d] = proj[(size_t)t * ldp + 3 * HK + d];
    float o[3];
#pragma unroll
    for (int s = 0; s < 3; s++) {
        const int c = s * nh * KD + h * KD + d;
        float acc = 0.f;
#pragma unroll
        for (int j = 0; j < CONV; j++) {
            const int tt = t + j - (CONV - 1);
            const int ta = r0 + tt;
            const float x = tt >= 0 ? proj[(size_t)tt * ldp + c]
                                    : (ta < 0 ? hist[(size_t)(CONV - 1 + ta) * C + c] : prev[(size_t)ta * C + c]);
            acc += convw[c * CONV + j] * x;
        }
        o[s] = silu(acc);
        rows_out[(size_t)t * C + c] = proj[(size_t)t * ldp + c];
    }
    __syncthreads();
    float gfv = 0.f;
#pragma unroll 8
    for (int j = 0; j < KD; j++) gfv += fbT[(size_t)j * HK + h * KD + d] * fa[j];
    const float qq = block_sum(o[0] * o[0], sh);
    const float kk = block_sum(o[1] * o[1], sh);
    const size_t i = ((size_t)t * nh + h) * KD + d;
    q[i] = o[0] / sqrtf(qq + 1e-6f) * rsqrtf((float)KD);
    k[i] = o[1] / sqrtf(kk + 1e-6f);
    v[i] = o[2];
    const float x = __expf(A[h]) * (gfv + dt[h * KD + d]);
    g[i] = __expf(GLB * sigm(x));  // the decay factor itself (k_kda_rec multiplies by it)
    if (d == 0) beta[t * nh + h] = sigm(proj[(size_t)t * ldp + 3 * nh * KD + 2 * KD + h]);
}
void kda_prep_f(const float* proj, int ldp, int nh, const float* convw, const float* hist, const float* fbT, const float* A,
                const float* dt, float* q, float* k, float* v, float* g, float* beta, int T, cudaStream_t st, float* rows_out,
                int r0, const float* prev) {
    k_kda_prep_f<<<dim3(T, nh), KD, 0, st>>>(proj, ldp, nh, convw, hist, fbT, A, dt, q, k, v, g, beta, r0, prev, rows_out);
}
// decode rows (T <= 4): both KDA gate GEMVs (gf = f_b f_a, gate = g_b g_a; weights transposed [128][HK]) in one kernel:
// grid (HK / 32, 2), 8 k-slices of 16 x 32 columns, slices summed in a fixed order
__global__ void __launch_bounds__(256) k_kgates(const float* fbT, const float* gbT, const float* proj, int ldp, int HK,
                                                int T, float* gf, float* gate) {
    __shared__ float xa[4][KD], red[8][4][32];
    const int z = blockIdx.y, col = blockIdx.x * 32 + (threadIdx.x & 31), ks = threadIdx.x >> 5;
    const float* W = z ? gbT : fbT;
    for (int i = threadIdx.x; i < T * KD; i += 256) xa[i / KD][i % KD] = proj[(size_t)(i / KD) * ldp + 3 * HK + z * KD + i % KD];
    __syncthreads();
    float w[16];
#pragma unroll
    for (int j = 0; j < 16; j++) w[j] = W[(size_t)(ks * 16 + j) * HK + col];
    for (int t = 0; t < T; t++) {
        float a = 0.f;
#pragma unroll
        for (int j = 0; j < 16; j++) a += w[j] * xa[t][ks * 16 + j];
        red[ks][t][threadIdx.x & 31] = a;
    }
    __syncthreads();
    if (threadIdx.x < 32 * T) {
        const int t = threadIdx.x >> 5, c = threadIdx.x & 31;
        float a = 0.f;
        for (int k = 0; k < 8; k++) a += red[k][t][c];
        (z ? gate : gf)[(size_t)t * HK + blockIdx.x * 32 + c] = a;
    }
}
void kda_gates(const float* fbT, const float* gbT, const float* proj, int ldp, int HK, int T, float* gf, float* gate,
               cudaStream_t st) {
    k_kgates<<<dim3(HK / 32, 2), 256, 0, st>>>(fbT, gbT, proj, ldp, HK, T, gf, gate);
}
// output gate inline: gate = g_b g_a (g_b transposed)
__global__ void k_kda_out_f(const float* o, const float* proj, int ldp, const float* gbT, const float* w, bf16* out, int nh) {
    __shared__ float sh[32], ga[KD];
    const int t = blockIdx.x, h = blockIdx.y, d = threadIdx.x, HK = nh * KD;
    ga[d] = proj[(size_t)t * ldp + 3 * HK + KD + d];
    __syncthreads();
    float gv = 0.f;
#pragma unroll 8
    for (int j = 0; j < KD; j++) gv += gbT[(size_t)j * HK + h * KD + d] * ga[j];
    const size_t i = ((size_t)t * nh + h) * KD + d;
    const float x = o[i];
    const float ss = block_sum(x * x, sh);
    out[i] = __float2bfloat16(x * rsqrtf(ss / KD + EPS) * w[d] * sigm(gv));
}
void kda_out_f(const float* o, const float* proj, int ldp, const float* gbT, const float* w, bf16* out, int T, int nh,
               cudaStream_t st) {
    k_kda_out_f<<<dim3(T, nh), KD, 0, st>>>(o, proj, ldp, gbT, w, out, nh);
}
__global__ void k_transpose(const float* a, float* b, int n, int m) {  // b[m][n] = a[n][m]
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < (size_t)n * m) b[(i % m) * n + i / m] = a[i];
}
void transpose(const float* a, float* b, int n, int m, cudaStream_t st) {
    k_transpose<<<(unsigned)(((size_t)n * m + 255) / 256), 256, 0, st>>>(a, b, n, m);
}

// conv history <- last 3 inputs of [hist; proj rows]
__global__ void k_kda_hist(const float* proj, int ldp, int C, int T, float* hist) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    float x[CONV - 1];
#pragma unroll
    for (int j = 0; j < CONV - 1; j++) {
        const int tt = T - (CONV - 1) + j;
        x[j] = tt < 0 ? hist[(size_t)(CONV - 1 + tt) * C + c] : proj[(size_t)tt * ldp + c];
    }
#pragma unroll
    for (int j = 0; j < CONV - 1; j++) hist[(size_t)j * C + c] = x[j];
}
// rows of the conv inputs -> dense [T][C] (decode verify: committed later)
__global__ void k_conv_rows(const float* proj, int ldp, int C, float* rows) {
    const int t = blockIdx.y;
    for (int c = blockIdx.x * blockDim.x + threadIdx.x; c < C; c += gridDim.x * blockDim.x)
        rows[(size_t)t * C + c] = proj[(size_t)t * ldp + c];
}
void kda_prep(const float* proj, int ldp, int nh, const float* convw, float* hist, const float* gf, const float* A,
              const float* dt, float* q, float* k, float* v, float* g, float* beta, int T, cudaStream_t st,
              float* conv_rows, int r0, const float* prev_rows) {
    k_kda_prep<<<dim3(T, nh), KD, 0, st>>>(proj, ldp, nh, convw, hist, gf, A, dt, q, k, v, g, beta, r0, prev_rows,
                                           conv_rows);
    const int C = 3 * nh * KD;
    if (conv_rows) return;
    k_kda_hist<<<(C + 255) / 256, 256, 0, st>>>(proj, ldp, C, T, hist);
}

// delta rule, blocks per (head, 32 value columns), TPC threads per value column (each holds KD / TPC of the 128 k rows
// of the state S[h][v][k] in registers); dots combined with shuffles. The value columns are independent, so splitting a
// head over KCS blocks keeps every column's arithmetic. More threads per column = shorter dependency chains and more
// warps per SM (the recurrence is latency-bound: 22 heads x 128 columns per card)
constexpr int KTILE = 16;
template <int TPC, int KCS>
__global__ void __launch_bounds__(TPC * KD / KCS) k_kda_rec(const float* q, const float* k, const float* v, const float* g,
                                                            const float* beta, float* S, float* o, int T, int nh, float* snap,
                                                            const int* pend) {
    constexpr int KPT = KD / TPC;
    __shared__ float sq[KTILE][KD], sk[KTILE][KD], se[KTILE][KD], sv[KTILE][KD], sb[KTILE];
    const int h = blockIdx.x, c = blockIdx.y * (KD / KCS) + threadIdx.x / TPC, hf = threadIdx.x % TPC, kb = hf * KPT;
    float s[KPT];
    float* Sp = S + ((size_t)h * KD + c) * KD + kb;
    // round 11: commit by pointer selection: the state after the last accepted row of the previous verify is still in
    // its snapshot (pend = accepted rows); each thread reads its own slice before it overwrites that slice's snapshots
    const int pa = pend ? *pend : 0;
    const float* Sr = pa > 0 ? snap + (size_t)(pa - 1) * nh * KD * KD + ((size_t)h * KD + c) * KD + kb : Sp;
#pragma unroll
    for (int i = 0; i < KPT; i += 4) {
        const float4 x = *reinterpret_cast<const float4*>(Sr + i);
        s[i] = x.x; s[i + 1] = x.y; s[i + 2] = x.z; s[i + 3] = x.w;
    }
    for (int t0 = 0; t0 < T; t0 += KTILE) {
        const int nt = min(KTILE, T - t0);
        __syncthreads();
        for (int i = threadIdx.x; i < nt * KD / 4; i += blockDim.x) {  // float4 loads; g holds exp(g) (kda_prep)
            const int j = i / (KD / 4), d = (i % (KD / 4)) * 4;
            const size_t b = ((size_t)(t0 + j) * nh + h) * KD + d;
            *reinterpret_cast<float4*>(&sq[j][d]) = *reinterpret_cast<const float4*>(q + b);
            *reinterpret_cast<float4*>(&sk[j][d]) = *reinterpret_cast<const float4*>(k + b);
            *reinterpret_cast<float4*>(&se[j][d]) = *reinterpret_cast<const float4*>(g + b);
            *reinterpret_cast<float4*>(&sv[j][d]) = *reinterpret_cast<const float4*>(v + b);
        }
        if (threadIdx.x < nt) sb[threadIdx.x] = beta[(t0 + threadIdx.x) * nh + h];
        __syncthreads();
        for (int j = 0; j < nt; j++) {
            float kv0 = 0.f, kv1 = 0.f;
#pragma unroll
            for (int i = 0; i < KPT; i += 2) {
                s[i] *= se[j][kb + i];
                s[i + 1] *= se[j][kb + i + 1];
                kv0 += s[i] * sk[j][kb + i];
                kv1 += s[i + 1] * sk[j][kb + i + 1];
            }
            float kv = kv0 + kv1;
#pragma unroll
            for (int m = 1; m < TPC; m <<= 1) kv += __shfl_xor_sync(0xffffffffu, kv, m);
            const float u = (sv[j][c] - kv) * sb[j];
            float o0 = 0.f, o1 = 0.f;
#pragma unroll
            for (int i = 0; i < KPT; i += 2) {
                s[i] += sk[j][kb + i] * u;
                s[i + 1] += sk[j][kb + i + 1] * u;
                o0 += s[i] * sq[j][kb + i];
                o1 += s[i + 1] * sq[j][kb + i + 1];
            }
            float oo = o0 + o1;
#pragma unroll
            for (int m = 1; m < TPC; m <<= 1) oo += __shfl_xor_sync(0xffffffffu, oo, m);
            if (hf == 0) o[((size_t)(t0 + j) * nh + h) * KD + c] = oo;
            if (snap) {  // verify rows: state after every row, S itself untouched
                float* sp = snap + (((size_t)(t0 + j) * nh + h) * KD + c) * KD + kb;
#pragma unroll
                for (int i = 0; i < KPT; i += 4) *reinterpret_cast<float4*>(sp + i) = make_float4(s[i], s[i + 1], s[i + 2], s[i + 3]);
            }
        }
    }
    if (snap) return;
#pragma unroll
    for (int i = 0; i < KPT; i += 4) *reinterpret_cast<float4*>(Sp + i) = make_float4(s[i], s[i + 1], s[i + 2], s[i + 3]);
}
// round 24 ([glm-r24-kdapk]): the prefill form (2 threads per column, 32 columns per unit) with G units per block, so the
// 88 units of a card take ~88 / G SMs instead of ~one each: a dense GEMM CTA (256 x 222 registers, 72 KB) does not fit
// next to a unit, so the GEMMs that overlap the recurrence (krst stream) had almost no SMs. Per column the same
// arithmetic in the same order as k_kda_rec<2, 4> (bit-identical); prefill only (no snapshots / pend).
template <int G>
__global__ void __launch_bounds__(64 * G) k_kda_rec_pk(const float* q, const float* k, const float* v, const float* g,
                                                       const float* beta, float* S, float* o, int T, int nh) {
    constexpr int TPC = 2, KCS = 4, KPT = KD / TPC, UT = TPC * KD / KCS;  // 64 threads per unit
    extern __shared__ float pk_sm[];
    const int u = threadIdx.x / UT, lt = threadIdx.x % UT, unit = blockIdx.x * G + u;
    const bool live = unit < nh * KCS;
    const int h = live ? unit / KCS : 0, c = (unit % KCS) * (KD / KCS) + lt / TPC, hf = lt % TPC, kb = hf * KPT;
    float (*sq)[KD] = reinterpret_cast<float (*)[KD]>(pk_sm + (size_t)u * (4 * KTILE * KD + KTILE));
    float (*sk)[KD] = sq + KTILE;
    float (*se)[KD] = sk + KTILE;
    float (*sv)[KD] = se + KTILE;
    float* sb = &sv[KTILE][0];
    float s[KPT];
    float* Sp = S + ((size_t)h * KD + c) * KD + kb;
#pragma unroll
    for (int i = 0; i < KPT; i += 4) {
        float4 x = make_float4(0.f, 0.f, 0.f, 0.f);
        if (live) x = *reinterpret_cast<const float4*>(Sp + i);
        s[i] = x.x; s[i + 1] = x.y; s[i + 2] = x.z; s[i + 3] = x.w;
    }
    for (int t0 = 0; t0 < T; t0 += KTILE) {
        const int nt = min(KTILE, T - t0);
        __syncthreads();
        if (live) {
            for (int i = lt; i < nt * KD / 4; i += UT) {
                const int j = i / (KD / 4), d = (i % (KD / 4)) * 4;
                const size_t b = ((size_t)(t0 + j) * nh + h) * KD + d;
                *reinterpret_cast<float4*>(&sq[j][d]) = *reinterpret_cast<const float4*>(q + b);
                *reinterpret_cast<float4*>(&sk[j][d]) = *reinterpret_cast<const float4*>(k + b);
                *reinterpret_cast<float4*>(&se[j][d]) = *reinterpret_cast<const float4*>(g + b);
                *reinterpret_cast<float4*>(&sv[j][d]) = *reinterpret_cast<const float4*>(v + b);
            }
            if (lt < nt) sb[lt] = beta[(t0 + lt) * nh + h];
        }
        __syncthreads();
        if (!live) continue;
        for (int j = 0; j < nt; j++) {
            float kv0 = 0.f, kv1 = 0.f;
#pragma unroll
            for (int i = 0; i < KPT; i += 2) {
                s[i] *= se[j][kb + i];
                s[i + 1] *= se[j][kb + i + 1];
                kv0 += s[i] * sk[j][kb + i];
                kv1 += s[i + 1] * sk[j][kb + i + 1];
            }
            float kv = kv0 + kv1;
#pragma unroll
            for (int m = 1; m < TPC; m <<= 1) kv += __shfl_xor_sync(0xffffffffu, kv, m);
            const float uu = (sv[j][c] - kv) * sb[j];
            float o0 = 0.f, o1 = 0.f;
#pragma unroll
            for (int i = 0; i < KPT; i += 2) {
                s[i] += sk[j][kb + i] * uu;
                s[i + 1] += sk[j][kb + i + 1] * uu;
                o0 += s[i] * sq[j][kb + i];
                o1 += s[i + 1] * sq[j][kb + i + 1];
            }
            float oo = o0 + o1;
#pragma unroll
            for (int m = 1; m < TPC; m <<= 1) oo += __shfl_xor_sync(0xffffffffu, oo, m);
            if (hf == 0) o[((size_t)(t0 + j) * nh + h) * KD + c] = oo;
        }
    }
    if (!live) return;
#pragma unroll
    for (int i = 0; i < KPT; i += 4) *reinterpret_cast<float4*>(Sp + i) = make_float4(s[i], s[i + 1], s[i + 2], s[i + 3]);
}
template <int G>
static void kda_rec_pk(const float* q, const float* k, const float* v, const float* g, const float* beta, float* S, float* o,
                       int T, int nh, cudaStream_t st) {
    const size_t sm = (size_t)G * (4 * KTILE * KD + KTILE) * 4;
    k_kda_rec_pk<G><<<(nh * 4 + G - 1) / G, 64 * G, sm, st>>>(q, k, v, g, beta, S, o, T, nh);
}
// the dynamic shared-memory attributes of both packed forms on the current device (at load, every card)
void kda_rec_pk_init() {
    CUDA_CHECK(cudaFuncSetAttribute(k_kda_rec_pk<2>, cudaFuncAttributeMaxDynamicSharedMemorySize, 2 * (4 * KTILE * KD + KTILE) * 4));
    CUDA_CHECK(cudaFuncSetAttribute(k_kda_rec_pk<3>, cudaFuncAttributeMaxDynamicSharedMemorySize, 3 * (4 * KTILE * KD + KTILE) * 4));
}
void kda_rec(const float* q, const float* k, const float* v, const float* g, const float* beta, float* S, float* o,
             int T, int nh, cudaStream_t st, float* snap, const int* pend) {
    static const int cfg = [] { const char* e = getenv("GLM_KDA_CFG"); return e ? atoi(e) : 0; }();
    switch (cfg) {
#define KR(t, c) k_kda_rec<t, c><<<dim3(nh, c), t * KD / c, 0, st>>>(q, k, v, g, beta, S, o, T, nh, snap, pend); break
        case 1: KR(2, 1);
        case 2: KR(2, 4);
        case 3: KR(4, 4);
        case 4: KR(8, 4);
        case 5: KR(4, 2);
        case 6: KR(8, 8);
        case 7: KR(16, 8);
        case 8: KR(4, 8);
        case 9: KR(2, 8);
        default:  // prefill: 2 threads per column, 4 blocks per head (0.64 us per token, kbench); decode rows: 16 x 8 (11 us)
            if (T > MR && !snap && !pend) {
                static const int pk = [] {
                    const char* e = getenv("GLM_KDA_PK");
                    const int v = e ? atoi(e) : 2;
                    if (v < 1 || v > 3) fprintf(stderr, "[glm] GLM_KDA_PK=%s is not 1, 2 or 3: unpacked recurrence\n", e);  // review r24 L2
                    return v;
                }();  // r24: default G = 2 (GLM_KDA_PK=1: k_kda_rec<2,4>)
                if (pk == 2) { kda_rec_pk<2>(q, k, v, g, beta, S, o, T, nh, st); break; }
                if (pk == 3) { kda_rec_pk<3>(q, k, v, g, beta, S, o, T, nh, st); break; }
            }
            if (T > MR) { KR(2, 4); }
            else { KR(16, 8); }
#undef KR
    }
}

// verify commit of every KDA layer: S <- snap[a - 1]; conv history <- last 3 of [hist; rows[0..a)]
// grid (layers, 64), 256 threads; tables of per-layer pointers
__global__ void k_kda_commit(float* const* S, float* const* snap, float* const* hist, float* const* rows, int nst,
                             int C, const int* acc, bool ptr) {
    const int l = blockIdx.x, a = *acc;
    if (ptr) {  // round 11: S stays stale; the next verify reads snap[a - 1] (acc[1] = pending rows)
        if (l == 0 && blockIdx.y == 0 && threadIdx.x == 0) const_cast<int*>(acc)[1] = a;
    } else {
        const float4* src = reinterpret_cast<const float4*>(snap[l] + (size_t)(a - 1) * nst);
        float4* dst = reinterpret_cast<float4*>(S[l]);
        for (int i = blockIdx.y * blockDim.x + threadIdx.x; i < nst / 4; i += gridDim.y * blockDim.x) dst[i] = src[i];
    }
    // conv history: new[j] (j = 0..2) = combined[a + j], combined = [hist 0..2; rows 0..a-1]
    for (int c = blockIdx.y * blockDim.x + threadIdx.x; c < C; c += gridDim.y * blockDim.x) {
        // unrolled with compile-time indices (a runtime-indexed x[] was a stack frame)
        float x[CONV - 1 + MR];
#pragma unroll
        for (int j = 0; j < CONV - 1; j++) x[j] = hist[l][(size_t)j * C + c];
#pragma unroll
        for (int j = 0; j < MR; j++) x[CONV - 1 + j] = j < a ? rows[l][(size_t)j * C + c] : 0.f;
#pragma unroll
        for (int j = 0; j < CONV - 1; j++) {
            float v = x[j];
#pragma unroll
            for (int q = 1; q <= MR; q++)
                if (q == a) v = x[q + j];
            hist[l][(size_t)j * C + c] = v;
        }
    }
}
void kda_commit(float* const* S, float* const* snap, float* const* hist, float* const* rows, int nl, int nst, int C,
                const int* acc, cudaStream_t st, bool ptr) {
    k_kda_commit<<<dim3(nl, 64), 256, 0, st>>>(S, snap, hist, rows, nst, C, acc, ptr);
}
// round 11: materialize S from the pending snapshot (before a prefill / sequence switch); acc[1] = 0 afterwards
__global__ void k_kda_mat(float* const* S, float* const* snap, int nst, int* acc) {
    const int l = blockIdx.x, a = acc[1];
    if (a > 0) {
        const float4* src = reinterpret_cast<const float4*>(snap[l] + (size_t)(a - 1) * nst);
        float4* dst = reinterpret_cast<float4*>(S[l]);
        for (int i = blockIdx.y * blockDim.x + threadIdx.x; i < nst / 4; i += gridDim.y * blockDim.x) dst[i] = src[i];
    }
}
__global__ void k_kda_mat_fin(int* acc) { acc[1] = 0; }
void kda_materialize(float* const* S, float* const* snap, int nl, int nst, int* acc, cudaStream_t st) {
    k_kda_mat<<<dim3(nl, 64), 256, 0, st>>>(S, snap, nst, acc);
    k_kda_mat_fin<<<1, 1, 0, st>>>(acc);
}

// o -> rms(o) * w * sigmoid(gate) -> bf16 ; grid (T, nh), 128 threads
__global__ void k_kda_out(const float* o, const float* gate, const float* w, bf16* out, int nh) {
    __shared__ float sh[32];
    const int t = blockIdx.x, h = blockIdx.y, d = threadIdx.x;
    const size_t i = ((size_t)t * nh + h) * KD + d;
    const float x = o[i];
    const float ss = block_sum(x * x, sh);
    out[i] = __float2bfloat16(x * rsqrtf(ss / KD + EPS) * w[d] * sigm(gate[i]));
}
void kda_out(const float* o, const float* gate, const float* w, bf16* out, int T, int nh, cudaStream_t st) {
    k_kda_out<<<dim3(T, nh), KD, 0, st>>>(o, gate, w, out, nh);
}

// ---------------------------------------------------------------- MLA prefill on tensor cores (query blocks)
// union of the selected tokens of a query block: bitmap over the context, then a sorted key list; per-query masks

__global__ void k_umark(const int* sel, const int* nsel, int t0, int nb, int n0, unsigned* bits) {
    const int q = blockIdx.x, t = t0 + q, pos = n0 + t;
    const int ns = nsel[t];
    if (ns < 0) {
        for (int w = threadIdx.x; w <= pos / 32; w += blockDim.x) {
            const int lo = w * 32, hi = min(pos, lo + 31);
            const unsigned m = hi - lo == 31 ? 0xffffffffu : ((1u << (hi - lo + 1)) - 1u);
            atomicOr(&bits[w], m);
        }
    } else {
        const int* s = sel + (size_t)t * MAXSEL;
        for (int j = threadIdx.x; j < ns; j += blockDim.x) atomicOr(&bits[s[j] >> 5], 1u << (s[j] & 31));
    }
}
// one block: exclusive prefix popcount of the words (pre[w]) and the key list; *U = total
__global__ void k_uscan(const unsigned* bits, int nw, int* pre, int* keys, int* U) {
    __shared__ int part[1024];
    __shared__ int carry;
    if (threadIdx.x == 0) carry = 0;
    __syncthreads();
    for (int b0 = 0; b0 < nw; b0 += 1024) {
        const int w = b0 + threadIdx.x;
        const unsigned v = w < nw ? bits[w] : 0u;
        part[threadIdx.x] = __popc(v);
        __syncthreads();
        for (int o = 1; o < 1024; o <<= 1) {  // inclusive scan
            const int x = threadIdx.x >= o ? part[threadIdx.x - o] : 0;
            __syncthreads();
            part[threadIdx.x] += x;
            __syncthreads();
        }
        const int ex = carry + part[threadIdx.x] - __popc(v);
        if (w < nw) {
            pre[w] = ex;
            unsigned m = v;
            int j = ex;
            while (m) { const int b = __ffs(m) - 1; keys[j++] = w * 32 + b; m &= m - 1; }
        }
        __syncthreads();
        if (threadIdx.x == 1023) carry += part[1023];
        __syncthreads();
    }
    if (threadIdx.x == 0) *U = carry;
}
// r12 (prefill review F1 fix A): the unions of all query blocks of one MLA call at once: block b's bitmap / prefix
// counts at b * uw words, its size in U[b]; the key list of a block is written later (k_ukeys), only for the blocks
// that take the union path. Same bitmaps, prefix counts and key order as k_umark + k_uscan per block
__global__ void k_umark_b(const int* sel, const int* nsel, int T, int qb, int n0, unsigned* bits, int uw) {
    const int b = blockIdx.y, t = b * qb + blockIdx.x;
    if (t >= T) return;
    unsigned* bb = bits + (size_t)b * uw;
    const int pos = n0 + t, ns = nsel[t];
    if (ns < 0) {
        for (int w = threadIdx.x; w <= pos / 32; w += blockDim.x) {
            const int lo = w * 32, hi = min(pos, lo + 31);
            const unsigned m = hi - lo == 31 ? 0xffffffffu : ((1u << (hi - lo + 1)) - 1u);
            atomicOr(&bb[w], m);
        }
    } else {
        const int* s = sel + (size_t)t * MAXSEL;
        for (int j = threadIdx.x; j < ns; j += blockDim.x) atomicOr(&bb[s[j] >> 5], 1u << (s[j] & 31));
    }
}
__global__ void k_uscan_b(const unsigned* bits, int nw, int uw, int* pre, int* U) {
    __shared__ int part[1024];
    __shared__ int carry;
    const unsigned* bb = bits + (size_t)blockIdx.x * uw;
    int* pb = pre + (size_t)blockIdx.x * uw;
    if (threadIdx.x == 0) carry = 0;
    __syncthreads();
    for (int b0 = 0; b0 < nw; b0 += 1024) {
        const int w = b0 + threadIdx.x;
        const unsigned v = w < nw ? bb[w] : 0u;
        part[threadIdx.x] = __popc(v);
        __syncthreads();
        for (int o = 1; o < 1024; o <<= 1) {
            const int x = threadIdx.x >= o ? part[threadIdx.x - o] : 0;
            __syncthreads();
            part[threadIdx.x] += x;
            __syncthreads();
        }
        if (w < nw) pb[w] = carry + part[threadIdx.x] - __popc(v);
        __syncthreads();
        if (threadIdx.x == 1023) carry += part[1023];
        __syncthreads();
    }
    if (threadIdx.x == 0) U[blockIdx.x] = carry;
}
__global__ void k_ukeys(const unsigned* bits, const int* pre, int nw, int* keys) {
    const int w = blockIdx.x * blockDim.x + threadIdx.x;
    if (w >= nw) return;
    unsigned m = bits[w];
    int j = pre[w];
    while (m) { const int b = __ffs(m) - 1; keys[j++] = w * 32 + b; m &= m - 1; }
}
void mla_union_all(const int* sel, const int* nsel, int T, int qb, int n0, int nw, int uw, unsigned* bits, int* pre,
                   int* dU, cudaStream_t st) {
    const int nblk = (T + qb - 1) / qb;
    CUDA_CHECK(cudaMemsetAsync(bits, 0, (size_t)nblk * uw * 4, st));
    k_umark_b<<<dim3(qb, nblk), 256, 0, st>>>(sel, nsel, T, qb, n0, bits, uw);
    k_uscan_b<<<nblk, 1024, 0, st>>>(bits, nw, uw, pre, dU);
}
void mla_ukeys(const unsigned* bits, const int* pre, int nw, int* keys, cudaStream_t st) {
    k_ukeys<<<(nw + 255) / 256, 256, 0, st>>>(bits, pre, nw, keys);
}
__global__ void k_ugather(const bf16* kc, const int* keys, int U, bf16* kb) {
    const int j = blockIdx.x;
    if (j >= U) return;
    const uint4* s = reinterpret_cast<const uint4*>(kc + (size_t)keys[j] * c_kst);
    uint4* d = reinterpret_cast<uint4*>(kb + (size_t)j * KVLR);
    for (int i = threadIdx.x; i < KVLR / 8; i += blockDim.x) d[i] = s[i];
}
// mask[q][j] = key j visible to query q
__global__ void k_umask(const int* sel, const int* nsel, const unsigned* bits, const int* pre, const int* keys, int U,
                        int ldm, int t0, int n0, unsigned char* mask) {
    const int q = blockIdx.x, t = t0 + q, pos = n0 + t, ns = nsel[t];
    unsigned char* m = mask + (size_t)q * ldm;
    if (ns < 0) {
        for (int j = threadIdx.x; j < U; j += blockDim.x) m[j] = keys[j] <= pos;
        return;
    }
    for (int j = threadIdx.x; j < U; j += blockDim.x) m[j] = 0;
    __syncthreads();
    const int* s = sel + (size_t)t * MAXSEL;
    for (int j = threadIdx.x; j < ns; j += blockDim.x) {
        const int tok = s[j], w = tok >> 5;
        m[pre[w] + __popc(bits[w] & ((1u << (tok & 31)) - 1u))] = 1;
    }
}
// rows (q, h): masked softmax of S[row][0..U) -> P bf16
__global__ void k_usoftmax(const float* S, const unsigned char* mask, int U, int nh, bf16* P) {
    __shared__ float sh[32];
    const int row = blockIdx.x, q = row / nh;
    const float* s = S + (size_t)row * U;
    const unsigned char* m = mask + (size_t)q * U;
    float mx = -FLT_MAX;
    for (int j = threadIdx.x; j < U; j += blockDim.x) if (m[j]) mx = fmaxf(mx, s[j]);
    for (int o = 16; o; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
    if ((threadIdx.x & 31) == 0) sh[threadIdx.x >> 5] = mx;
    __syncthreads();
    mx = -FLT_MAX;
    for (int w = 0; w < (int)(blockDim.x >> 5); w++) mx = fmaxf(mx, sh[w]);
    __syncthreads();
    float sum = 0.f;
    for (int j = threadIdx.x; j < U; j += blockDim.x) sum += m[j] ? __expf(s[j] - mx) : 0.f;
    sum = block_sum(sum, sh);
    const float inv = 1.f / sum;
    for (int j = threadIdx.x; j < U; j += blockDim.x)
        P[(size_t)row * U + j] = __float2bfloat16(m[j] ? __expf(s[j] - mx) * inv : 0.f);
}
__global__ void k_qscale(const float* q, bf16* qb, size_t n, float scale) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
        qb[i] = __float2bfloat16(q[i] * scale);
}
void mla_union(const int* sel, const int* nsel, int t0, int nb, int n0, int ctxmax, unsigned* bits, int* pre, int* keys,
               int* dU, cudaStream_t st) {
    const int nw = (n0 + t0 + nb + 31) / 32 + 1;
    CUDA_CHECK(cudaMemsetAsync(bits, 0, (size_t)nw * 4, st));
    k_umark<<<nb, 256, 0, st>>>(sel, nsel, t0, nb, n0, bits);
    k_uscan<<<1, 1024, 0, st>>>(bits, nw, pre, keys, dU);
    (void)ctxmax;
}
void mla_union_rest(const bf16* kc, const int* sel, const int* nsel, const unsigned* bits, const int* pre, const int* keys,
                    int U, int t0, int nb, int n0, bf16* kb, unsigned char* mask, cudaStream_t st) {
    k_ugather<<<U, 64, 0, st>>>(kc, keys, U, kb);
    k_umask<<<nb, 256, 0, st>>>(sel, nsel, bits, pre, keys, U, (U + 7) & ~7, t0, n0, mask);
}
void usoftmax(const float* S, const unsigned char* mask, int U, int nh, int rows, bf16* P, cudaStream_t st) {
    k_usoftmax<<<rows, 256, 0, st>>>(S, mask, U, nh, P);
}
void qscale(const float* q, bf16* qb, size_t n, float scale, cudaStream_t st) { k_qscale<<<256, 256, 0, st>>>(q, qb, n, scale); }

// ---------------------------------------------------------------- MLA + DSA indexer

// per token: proj [qa 1536][kva 512][wk 128][wp 32][gate 128] -> qr bf16, cache rows (c, ki, gs), wi
__global__ void k_mla_prep(const float* proj, const float* qaln, const float* kvln, const float* ikw, const float* ikb,
                           bf16* qr, bf16* kc, float* ki, float* gsc, float* wi, int n0, const int* dp, int ring) {
    __shared__ float sh[32];
    if (dp) n0 += *dp;
    const int t = blockIdx.x;
    const float* p = proj + (size_t)t * MLA_IN;
    float s = 0.f;
    for (int i = threadIdx.x; i < QLR; i += blockDim.x) s += p[i] * p[i];
    s = block_sum(s, sh);
    float iv = rsqrtf(s / QLR + EPS);
    for (int i = threadIdx.x; i < QLR; i += blockDim.x) qr[(size_t)t * QLR + i] = __float2bfloat16(p[i] * iv * qaln[i]);
    s = 0.f;
    for (int i = threadIdx.x; i < KVLR; i += blockDim.x) s += p[QLR + i] * p[QLR + i];
    s = block_sum(s, sh);
    iv = rsqrtf(s / KVLR + EPS);
    const size_t row = (size_t)(n0 + t);
    for (int i = threadIdx.x; i < KVLR; i += blockDim.x) kc[row * c_kst + i] = __float2bfloat16(p[QLR + i] * iv * kvln[i]);
    const size_t rr = row % ring;  // index keys / gate scores: ring (they only feed the pooled keys)
    // LayerNorm of wk x (eps 1e-6, weight + bias)
    const float* kx = p + QLR + KVLR;
    float mu = 0.f;
    for (int i = threadIdx.x; i < ID; i += blockDim.x) mu += kx[i];
    mu = block_sum(mu, sh) / ID;
    float var = 0.f;
    for (int i = threadIdx.x; i < ID; i += blockDim.x) var += (kx[i] - mu) * (kx[i] - mu);
    var = block_sum(var, sh) / ID;
    const float ivs = rsqrtf(var + LN_EPS);
    for (int i = threadIdx.x; i < ID; i += blockDim.x) {
        ki[rr * ID + i] = (kx[i] - mu) * ivs * ikw[i] + ikb[i];
        gsc[rr * ID + i] = p[QLR + KVLR + ID + IH + i];
    }
    for (int i = threadIdx.x; i < IH; i += blockDim.x) wi[t * IH + i] = p[QLR + KVLR + ID + i] * rsqrtf((float)IH);
}
void mla_prep(const float* proj, const float* qaln, const float* kvln, const float* ikw, const float* ikb, bf16* qr,
              bf16* kc, float* ki, float* gsc, float* wi, int n0, int T, cudaStream_t st, const int* dp, int ring) {
    k_mla_prep<<<T, 256, 0, st>>>(proj, qaln, kvln, ikw, ikb, qr, kc, ki, gsc, wi, n0, dp, ring);
}

// pooled keys of pools [p0, p1): per channel softmax over the 4 positions of (gs + ape)
__global__ void k_pool(const float* ki, const float* gsc, const float* ape, float* pk, int p0, int ring) {
    const int p = p0 + blockIdx.x, c = threadIdx.x;
    float gv[POOL], mx = -FLT_MAX;
#pragma unroll
    for (int j = 0; j < POOL; j++) { gv[j] = gsc[(size_t)((p * POOL + j) % ring) * ID + c] + ape[j * ID + c]; mx = fmaxf(mx, gv[j]); }
    float s = 0.f, acc = 0.f;
#pragma unroll
    for (int j = 0; j < POOL; j++) { const float e = expf(gv[j] - mx); s += e; acc += e * ki[(size_t)((p * POOL + j) % ring) * ID + c]; }
    pk[(size_t)p * c_pst + c] = acc / s;
}
void pool_keys(const float* ki, const float* gsc, const float* ape, float* pk, int p0, int p1, cudaStream_t st, int ring) {
    if (p1 > p0) k_pool<<<p1 - p0, ID, 0, st>>>(ki, gsc, ape, pk, p0, ring);
}
// decode: the pool completed by the token at position *dp (if any)
__global__ void k_pool_dec(const float* ki, const float* gsc, const float* ape, float* pk, const int* dp, int ring) {
    const int pos = *dp + blockIdx.x;  // row blockIdx.x of the step
    if ((pos + 1) % POOL) return;
    const int p = (pos + 1) / POOL - 1, c = threadIdx.x;
    float gv[POOL], mx = -FLT_MAX;
#pragma unroll
    for (int j = 0; j < POOL; j++) { gv[j] = gsc[(size_t)((p * POOL + j) % ring) * ID + c] + ape[j * ID + c]; mx = fmaxf(mx, gv[j]); }
    float s = 0.f, acc = 0.f;
#pragma unroll
    for (int j = 0; j < POOL; j++) { const float e = expf(gv[j] - mx); s += e; acc += e * ki[(size_t)((p * POOL + j) % ring) * ID + c]; }
    pk[(size_t)p * c_pst + c] = acc / s;
}
void pool_keys_dec(const float* ki, const float* gsc, const float* ape, float* pk, const int* dp, int T, cudaStream_t st,
                   int ring) {
    k_pool_dec<<<T, ID, 0, st>>>(ki, gsc, ape, pk, dp, ring);
}

// indexer scores of query t over its visible pools: s[p] = sum_h w_h relu(qi_h . pk_p / sqrt(128)); grid (tiles, T)
// decode / prefill indexer scores: 4 consecutive threads per pool, each 8 of the 32 heads (the per-pool head loop was
// 4,096 serial FMAs per thread with ~vis threads in flight); the 4 partial sums are added in lane order
__global__ void k_idx_score(const float* qi, int ldq, const float* wi, const float* pk, float* sc, int ld, int n0,
                            const int* dp, int vlim = 0) {
    __shared__ float sq[IH][ID + 1];  // padded: the 4 head groups of a pool read different banks
    __shared__ float sw[IH];
    if (dp) n0 += *dp;
    const int t = blockIdx.y;
    const int vis = (n0 + t + 1) / POOL;
    if (dp && vis <= TOPB) return;  // decode: dense attention, no scores needed
    if (dp && vlim > 0 && (n0 + (int)gridDim.y) / POOL > vlim) return;  // round 26: deeper steps take k_idx_dec
    // round 26 (D4 a, GLM_IDX_GRID): a decode grid smaller than the configured context's tiles strides over the visible
    // pools only (same per-pool arithmetic; the select reads pools < vis only, so nothing is written past them)
    const bool strided = (int)gridDim.x * (blockDim.x / 4) < ld;
    bool loaded = false;
    for (int p0 = blockIdx.x * (blockDim.x / 4); p0 < ld; p0 += gridDim.x * (blockDim.x / 4)) {
        if (p0 >= vis) {
            if (strided) return;
            for (int i = threadIdx.x; i < blockDim.x / 4; i += blockDim.x)
                if (p0 + i < ld) sc[(size_t)t * ld + p0 + i] = -FLT_MAX;
            return;
        }
        if (!loaded) {
            for (int i = threadIdx.x; i < IH * ID; i += blockDim.x) sq[i / ID][i % ID] = qi[(size_t)t * ldq + i];
            if (threadIdx.x < IH) sw[threadIdx.x] = wi[t * IH + threadIdx.x];
            __syncthreads();
            loaded = true;
        }
        const int p = p0 + threadIdx.x / 4, g = threadIdx.x & 3;
        float s = 0.f;
        if (p < vis) {
            float kv[ID];
            const float4* k4 = reinterpret_cast<const float4*>(pk + (size_t)p * c_pst);
#pragma unroll
            for (int i = 0; i < ID / 4; i++) { const float4 x = k4[i]; kv[4 * i] = x.x; kv[4 * i + 1] = x.y; kv[4 * i + 2] = x.z; kv[4 * i + 3] = x.w; }
            for (int hh = g * (IH / 4); hh < (g + 1) * (IH / 4); hh++) {
                float d = 0.f;
#pragma unroll
                for (int i = 0; i < ID; i++) d += sq[hh][i] * kv[i];
                s += sw[hh] * fmaxf(d * rsqrtf((float)ID), 0.f);
            }
        }
        const float s1 = __shfl_down_sync(0xffffffffu, s, 1), s2 = __shfl_down_sync(0xffffffffu, s, 2),
                    s3 = __shfl_down_sync(0xffffffffu, s, 3);
        if (g == 0 && p < ld) sc[(size_t)t * ld + p] = p < vis ? ((s + s1) + s2) + s3 : -FLT_MAX;
    }
}
// round 26 (D4 depth form, exact): decode indexer scores of all RT rows of a step in one block, PP pools per thread.
// k_idx_score re-reads every pool's key per row and issues one shared load per FMA (load-bound at depth: 157 us per
// pass at 100K). Here a thread holds one head group (8 heads, as there) of PP pools for all RT rows: the key is read
// once for the rows, every shared float4 of the queries feeds PP FMAs. Per (row, pool, head) the FMA chain over the
// 128 dims runs in the same order, the 8 heads of a group are summed in the same order and the 4 groups as
// ((s0 + s1) + s2) + s3: the same bits as k_idx_score.
int g_idx_dec = -1;  // GLM_IDX_DEC (-1: not read yet; the kernel bench sets it)
constexpr int IXG = 8 * ID + 4;  // floats per (row, group) in shared memory (+4: the 4 groups start in other banks)
template <int RT, int PP>
__global__ void __launch_bounds__(256) k_idx_dec(const float* qi, int ldq, const float* wi, const float* pk, float* sc,
                                                 int ld, const int* dp, int vlim) {
    extern __shared__ __align__(16) float ism[];
    float* sq = ism;                   // [RT][4][IXG]
    float* sw = ism + RT * 4 * IXG;    // [RT][IH]
    const int n0 = *dp;
    int vis[RT], vmax = 0;
#pragma unroll
    for (int t = 0; t < RT; t++) { vis[t] = (n0 + t + 1) / POOL; vmax = max(vmax, vis[t]); }
    if (vmax <= TOPB || vmax <= vlim) return;  // dense attention for every row / shallow: k_idx_score has the step
    if (blockIdx.x * 64 * PP >= vmax) return;  // review r26b-L5: no pools for this block: no query load either
    for (int i = threadIdx.x; i < RT * IH * ID; i += blockDim.x) {
        const int t = i / (IH * ID), r = i % (IH * ID), hh = r / ID, d = r % ID;
        sq[(t * 4 + hh / 8) * IXG + (hh % 8) * ID + d] = qi[(size_t)t * ldq + r];
    }
    for (int i = threadIdx.x; i < RT * IH; i += blockDim.x) sw[i] = wi[i];
    __syncthreads();
    const int g = threadIdx.x & 3, pl = threadIdx.x >> 2;  // head group, pool lane (64 per block)
    for (int pb = blockIdx.x * 64 * PP; pb < vmax; pb += gridDim.x * 64 * PP) {
        float d[PP][RT][8];
#pragma unroll
        for (int a = 0; a < PP; a++)
#pragma unroll
            for (int t = 0; t < RT; t++)
#pragma unroll
                for (int h = 0; h < 8; h++) d[a][t][h] = 0.f;
        const float4* k4[PP];
#pragma unroll
        for (int a = 0; a < PP; a++) k4[a] = reinterpret_cast<const float4*>(pk + (size_t)min(pb + a * 64 + pl, vmax - 1) * c_pst);
#pragma unroll 2
        for (int i4 = 0; i4 < ID / 4; i4++) {
            float4 kv[PP];
#pragma unroll
            for (int a = 0; a < PP; a++) kv[a] = k4[a][i4];
#pragma unroll
            for (int t = 0; t < RT; t++)
#pragma unroll
                for (int h = 0; h < 8; h++) {
                    const float4 q = *reinterpret_cast<const float4*>(sq + (t * 4 + g) * IXG + h * ID + 4 * i4);
#pragma unroll
                    for (int a = 0; a < PP; a++) {
                        float x = d[a][t][h];
                        x += q.x * kv[a].x;
                        x += q.y * kv[a].y;
                        x += q.z * kv[a].z;
                        x += q.w * kv[a].w;
                        d[a][t][h] = x;
                    }
                }
        }
#pragma unroll
        for (int a = 0; a < PP; a++) {
            const int p = pb + a * 64 + pl;
#pragma unroll
            for (int t = 0; t < RT; t++) {
                float s = 0.f;
#pragma unroll
                for (int h = 0; h < 8; h++) s += sw[t * IH + g * 8 + h] * fmaxf(d[a][t][h] * rsqrtf((float)ID), 0.f);
                const float s1 = __shfl_down_sync(0xffffffffu, s, 1), s2 = __shfl_down_sync(0xffffffffu, s, 2),
                            s3 = __shfl_down_sync(0xffffffffu, s, 3);
                if (g == 0 && p < vis[t] && p < ld && vis[t] > TOPB) sc[(size_t)t * ld + p] = ((s + s1) + s2) + s3;
            }
        }
    }
}
template <int RT, int PP> static void idx_dec_launch(const float* qi, int ldq, const float* wi, const float* pk, float* sc,
                                                     int ld, const int* dp, int gx, int vlim, cudaStream_t st) {
    const size_t sm = ((size_t)RT * 4 * IXG + RT * IH) * 4;  // (attribute set per card by idx_dec_init at load)
    k_idx_dec<RT, PP><<<gx, 256, sm, st>>>(qi, ldq, wi, pk, sc, ld, dp, vlim);
}
void sel2_init();
void idx_dec_init() {  // the shared-memory attribute on the current card (called per card at load)
    if (g_idx_dec < 0) { const char* e = getenv("GLM_IDX_DEC"); g_idx_dec = e ? atoi(e) : 1; }  // review r26b L4: at load
    for (int r = 1; r <= 4; r++) {
        const int sm = (r * 4 * IXG + r * IH) * 4;
        if (r == 1) CUDA_CHECK(cudaFuncSetAttribute(k_idx_dec<1, 2>, cudaFuncAttributeMaxDynamicSharedMemorySize, sm));
        if (r == 2) CUDA_CHECK(cudaFuncSetAttribute(k_idx_dec<2, 2>, cudaFuncAttributeMaxDynamicSharedMemorySize, sm));
        if (r == 3) CUDA_CHECK(cudaFuncSetAttribute(k_idx_dec<3, 2>, cudaFuncAttributeMaxDynamicSharedMemorySize, sm));
        if (r == 4) CUDA_CHECK(cudaFuncSetAttribute(k_idx_dec<4, 2>, cudaFuncAttributeMaxDynamicSharedMemorySize, sm));
    }
    sel2_init();
}
void idx_score(const float* qi, int ldq, const float* wi, const float* pk, float* sc, int ld, int n0, int T,
               cudaStream_t st, const int* dp) {
    // blocks per row; round 26 default 64 (ctx 163,840: -41 / -26 us per verify at 25 / 8K positions, exact), 0 = off
    static const int cap = [] { const char* e = getenv("GLM_IDX_GRID"); return e ? atoi(e) : 64; }();
    // round 26: decode steps take the row-shared form (GLM_IDX_DEC=0: k_idx_score per row); n0 is 0 there (dp holds it)
    if (g_idx_dec < 0) { const char* e = getenv("GLM_IDX_DEC"); g_idx_dec = e ? atoi(e) : 1; }
    // both kernels sit in the captured step and pick the step by its depth on the device: k_idx_score up to vlim
    // visible pools (fewer, lighter blocks win there), k_idx_dec beyond (GLM_IDX_VLIM pools, default 2,560 = 10K positions; kbench: 8K 27.6 vs 29.2 us, 12.5K 40.0 vs 37.1, 100K 275 vs 128)
    static const int vlim = [] { const char* e = getenv("GLM_IDX_VLIM"); return e ? atoi(e) : 2560; }();
    int vl = 0;
    if (dp && g_idx_dec && T >= 1 && T <= 4 && n0 == 0) {
        vl = std::max(0, vlim);
        const int gx = std::max(1, std::min(cap > 0 ? cap : 1 << 20, (ld + 127) / 128));
        if (T == 1) idx_dec_launch<1, 2>(qi, ldq, wi, pk, sc, ld, dp, gx, vl, st);
        else if (T == 2) idx_dec_launch<2, 2>(qi, ldq, wi, pk, sc, ld, dp, gx, vl, st);
        else if (T == 3) idx_dec_launch<3, 2>(qi, ldq, wi, pk, sc, ld, dp, gx, vl, st);
        else idx_dec_launch<4, 2>(qi, ldq, wi, pk, sc, ld, dp, gx, vl, st);
        if (vl == 0) return;
    }
    int gx = (ld + 63) / 64;
    if (dp && cap > 0) gx = std::min(gx, cap);
    k_idx_score<<<dim3(gx, T), 256, 0, st>>>(qi, ldq, wi, pk, sc, ld, n0, dp, vl);
}

// float -> order-preserving uint
__device__ __forceinline__ unsigned fkey(float f) {
    const unsigned u = __float_as_uint(f);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}
// top-TOPB pools of each query (radix select on the score bits) + its tail -> token list; dense (vis <= TOPB): nsel = -1
__global__ void k_select(const float* sc, int ld, int n0, int* sel, int* nsel, const int* dp, int* psel) {
    __shared__ unsigned hist[256];
    __shared__ unsigned pref, need, cnt_gt, cnt_eq;
    if (dp) n0 += *dp;
    const int t = blockIdx.x;
    const int pos = n0 + t, vis = (pos + 1) / POOL;
    if (vis <= TOPB) { if (threadIdx.x == 0) nsel[t] = -1; return; }
    const float* s = sc + (size_t)t * ld;
    if (threadIdx.x == 0) { pref = 0; need = TOPB; }
    // 4 passes of 8 bits from the top: find the key K of the TOPB-th largest
    for (int sh = 24; sh >= 0; sh -= 8) {
        for (int i = threadIdx.x; i < 256; i += blockDim.x) hist[i] = 0;
        __syncthreads();
        const unsigned mask = sh == 24 ? 0u : (0xffffffffu << (sh + 8));
        for (int p = threadIdx.x; p < vis; p += blockDim.x) {
            const unsigned kk = fkey(s[p]);
            if ((kk & mask) == pref) atomicAdd(&hist[(kk >> sh) & 255], 1u);
        }
        __syncthreads();
        {   // boundary bin: the highest b with (count of bins >= b) >= need; parallel suffix scan (bin 255 - tid)
            __shared__ unsigned wsum[8], nb_, nacc_, inc1_;
            const int tid = threadIdx.x;
            unsigned v = 0, incl = 0;
            if (tid < 256) {
                v = hist[255 - tid];
                incl = v;
#pragma unroll
                for (int o = 1; o < 32; o <<= 1) {
                    const unsigned y = __shfl_up_sync(0xffffffffu, incl, o);
                    if ((tid & 31) >= o) incl += y;
                }
                if ((tid & 31) == 31) wsum[tid >> 5] = incl;
            }
            if (tid == 0) { nb_ = 0; nacc_ = 0xffffffffu; }
            __syncthreads();
            if (tid < 256) {
                for (int i = 0; i < (tid >> 5); i++) incl += wsum[i];
                const unsigned excl = incl - v;
                const int b = 255 - tid;
                if (b > 0 && excl < need && incl >= need) { nb_ = b; nacc_ = excl; }
                if (b == 1) inc1_ = incl;  // used only if no bin > 0 qualified
            }
            __syncthreads();
            if (tid == 0) {
                const unsigned acc = nacc_ != 0xffffffffu ? nacc_ : inc1_;
                need -= acc;
                pref |= (unsigned)nb_ << sh;
            }
            __syncthreads();
        }
    }
    // pref = key of the boundary; take all > pref in pool order, then the first `need` of == pref (lowest pool index
    // first): block-wide ordered compaction (ballot + warp prefix), deterministic, O(vis / blockDim) steps
    __shared__ unsigned wc[32], run;
    int* out = sel + (size_t)t * MAXSEL;
    const int w = threadIdx.x >> 5, ln = threadIdx.x & 31, nw = blockDim.x >> 5;
    for (int pass = 0; pass < 2; pass++) {
        if (threadIdx.x == 0 && pass == 0) run = 0;
        __syncthreads();
        const unsigned lim = pass == 0 ? TOPB : run + need;  // pass 1: ties up to TOPB in total
        for (int p0 = 0; p0 < vis; p0 += blockDim.x) {
            if (run >= lim) break;  // uniform: run is read after a barrier
            const int p = p0 + threadIdx.x;
            bool f = false;
            if (p < vis) {
                const unsigned kk = fkey(s[p]);
                f = pass == 0 ? kk > pref : kk == pref;
            }
            const unsigned b = __ballot_sync(0xffffffffu, f);
            if (ln == 0) wc[w] = __popc(b);
            __syncthreads();
            unsigned pre = 0, tot = 0;
            for (int i = 0; i < nw; i++) { if (i < w) pre += wc[i]; tot += wc[i]; }
            const unsigned j = run + pre + __popc(b & ((1u << ln) - 1u));
            if (f && j < lim) {
                if (psel) psel[(size_t)t * TOPB + j] = p;
                else for (int q = 0; q < POOL; q++) out[j * POOL + q] = p * POOL + q;
            }
            __syncthreads();
            if (threadIdx.x == 0) run = min(lim, run + tot);
            __syncthreads();
        }
    }
    if (threadIdx.x == 0) {
        if (psel) { nsel[t] = TOPB; return; }  // pools only (expanded after the cards exchanged their rows)
        int n = TOPB * POOL;
        for (int q = vis * POOL; q <= pos; q++) out[n++] = q;  // incomplete tail
        nsel[t] = n;
    }
}
// round 27 (exact, decode): the same selection as k_select (top TOPB keys, ties lowest pool first, written in pool order)
// in 3 latency-friendly walks over the scores instead of 6 walks with a barrier per 512 pools: (1) a 12-bit histogram,
// (2) the boundary bin's pools gathered into shared memory (the last 20 bits are resolved there in two 10-bit passes)
// plus per-warp counts of the pools above it, (3) one ordered write: every warp owns a contiguous run of pools and
// knows its output offsets from the counts, so it writes with ballot prefixes and no block barrier. If the boundary
// bin holds more than SXCAP pools, passes 2-3 read the scores again (same result).
int g_sel2 = -1;  // GLM_SEL2 (1 default, 0: k_select)
constexpr int SXT = 1024, SXW = SXT / 32, SXCAP = 8192, SXU = 4;
constexpr size_t SXSM = (4096 + 2 * (size_t)SXCAP) * 4;
// highest bin b of h[0..nb) with (count of bins >= b) >= need; writes b and the count of bins > b (all threads call it)
__device__ __forceinline__ void sx_bound(const unsigned* h, int nb, unsigned need, unsigned* wsum, unsigned* ob,
                                         unsigned* oab) {
    const int tid = threadIdx.x, R = nb / SXT, top = nb - 1 - tid * R;  // thread tid: bins top .. top - R + 1
    unsigned v = 0;
    for (int r = 0; r < R; r++) v += h[top - r];
    unsigned incl = v;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        const unsigned y = __shfl_up_sync(0xffffffffu, incl, o);
        if ((tid & 31) >= o) incl += y;
    }
    if ((tid & 31) == 31) wsum[tid >> 5] = incl;
    __syncthreads();
    for (int i = 0; i < (tid >> 5); i++) incl += wsum[i];
    unsigned acc = incl - v;
    if (acc < need && incl >= need)
        for (int r = 0; r < R; r++) {
            const unsigned c = h[top - r];
            if (acc + c >= need) { *ob = top - r; *oab = acc; break; }
            acc += c;
        }
    __syncthreads();
}
__global__ void __launch_bounds__(SXT) k_select2(const float* sc, int ld, int* sel, int* nsel, const int* dp) {
    extern __shared__ __align__(16) unsigned sxm[];
    unsigned* hist = sxm;             // [4096]
    unsigned* ck = sxm + 4096;        // [SXCAP] keys of the boundary bin
    int* cp = (int*)(ck + SXCAP);     // [SXCAP] their pools
    __shared__ unsigned wsum[SXW], wgt[SXW], weq[SXW], ogt[SXW], oeq[SXW], b_, ab_, nc_;
    const int t = blockIdx.x, pos = *dp + t, vis = (pos + 1) / POOL;
    if (vis <= TOPB) { if (threadIdx.x == 0) nsel[t] = -1; return; }
    const float* s = sc + (size_t)t * ld;
    const int tid = threadIdx.x, w = tid >> 5, ln = tid & 31;
    const int S = (((vis + SXW - 1) / SXW) + 31) & ~31, e0 = w * S, e1 = min(vis, e0 + S);  // warp w: pools [e0, e1)
    for (int i = tid; i < 4096; i += SXT) hist[i] = 0;
    if (tid < SXW) { wgt[tid] = 0; weq[tid] = 0; }
    if (tid == 0) nc_ = 0;
    __syncthreads();
    // walk 1: histogram of the top 12 key bits (a warp whose pools share one bin adds once)
    for (int b = e0; b < e1; b += 32 * SXU) {
        unsigned k[SXU];
#pragma unroll
        for (int u = 0; u < SXU; u++) { const int p = b + u * 32 + ln; k[u] = p < e1 ? fkey(s[p]) : 0u; }
#pragma unroll
        for (int u = 0; u < SXU; u++) {
            const int p = b + u * 32 + ln;
            const bool v = p < e1;
            const unsigned d = k[u] >> 20, d0 = __shfl_sync(0xffffffffu, d, 0);
            if (__all_sync(0xffffffffu, !v || d == d0)) {
                const unsigned n = __popc(__ballot_sync(0xffffffffu, v));
                if (ln == 0 && n) atomicAdd(&hist[d0], n);
            } else if (v) atomicAdd(&hist[d], 1u);
        }
    }
    __syncthreads();
    sx_bound(hist, 4096, TOPB, wsum, &b_, &ab_);
    const unsigned b1 = b_, C1 = hist[b1];
    unsigned need = TOPB - ab_;
    const bool fit = C1 <= (unsigned)SXCAP;
    // walk 2: pools above the boundary bin per warp; the boundary bin's keys and pools into shared memory (review
    // r27-L5: skipped when the bin overflows: the count below then replaces walk 2's counts and nothing is collected)
    if (fit) {
        unsigned above = 0;
        for (int b = e0; b < e1; b += 32 * SXU) {
            unsigned k[SXU];
#pragma unroll
            for (int u = 0; u < SXU; u++) { const int p = b + u * 32 + ln; k[u] = p < e1 ? fkey(s[p]) : 0u; }
#pragma unroll
            for (int u = 0; u < SXU; u++) {
                const int p = b + u * 32 + ln;
                const bool v = p < e1;
                above += v && (k[u] >> 20) > b1;
                if (fit) {
                    const bool c = v && (k[u] >> 20) == b1;
                    const unsigned m = __ballot_sync(0xffffffffu, c);
                    if (m) {
                        unsigned base = 0;
                        if (ln == 0) base = atomicAdd(&nc_, __popc(m));
                        base = __shfl_sync(0xffffffffu, base, 0) + __popc(m & ((1u << ln) - 1u));
                        if (c) { ck[base] = k[u]; cp[base] = p; }
                    }
                }
            }
        }
#pragma unroll
        for (int o = 16; o; o >>= 1) above += __shfl_xor_sync(0xffffffffu, above, o);
        if (ln == 0) wgt[w] = above;
    }
    __syncthreads();
    // passes 2 and 3: bits 19..10 and 9..0 among the boundary bin (shared memory, or the scores again if it overflowed)
    unsigned pref = b1;  // key bits resolved so far, right-aligned
    for (int sh = 10; sh >= 0; sh -= 10) {
        for (int i = tid; i < 1024; i += SXT) hist[i] = 0;
        __syncthreads();
        if (fit) {
            for (int i = tid; i < (int)C1; i += SXT) {
                const unsigned kk = ck[i];
                if ((kk >> (sh + 10)) == pref) atomicAdd(&hist[(kk >> sh) & 1023], 1u);
            }
        } else {
            for (int p = tid; p < vis; p += SXT) {
                const unsigned kk = fkey(s[p]);
                if ((kk >> (sh + 10)) == pref) atomicAdd(&hist[(kk >> sh) & 1023], 1u);
            }
        }
        __syncthreads();
        sx_bound(hist, 1024, need, wsum, &b_, &ab_);
        need -= ab_;
        pref = (pref << 10) | b_;
    }
    const unsigned K = pref;  // the key of the TOPB-th largest score; take every key > K and the first `need` == K
    // per-warp counts of > K and == K
    if (fit) {
        for (int i = tid; i < (int)C1; i += SXT) {
            const unsigned kk = ck[i];
            if (kk > K) atomicAdd(&wgt[cp[i] / S], 1u);
            else if (kk == K) atomicAdd(&weq[cp[i] / S], 1u);
        }
    } else {
        unsigned g = 0, q = 0;
        for (int b = e0; b < e1; b += 32) {
            const int p = b + ln;
            const unsigned kk = p < e1 ? fkey(s[p]) : 0u;
            g += p < e1 && kk > K;
            q += p < e1 && kk == K;
        }
#pragma unroll
        for (int o = 16; o; o >>= 1) { g += __shfl_xor_sync(0xffffffffu, g, o); q += __shfl_xor_sync(0xffffffffu, q, o); }
        if (ln == 0) { wgt[w] = g; weq[w] = q; }  // (walk 2's counts are replaced: they cover the bins above b1 only)
    }
    __syncthreads();
    if (w == 0) {  // exclusive offsets in warp (= pool) order
        const unsigned g = wgt[ln], q = weq[ln];
        unsigned ig = g, iq = q;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            const unsigned yg = __shfl_up_sync(0xffffffffu, ig, o), yq = __shfl_up_sync(0xffffffffu, iq, o);
            if (ln >= o) { ig += yg; iq += yq; }
        }
        ogt[ln] = ig - g;
        oeq[ln] = iq - q;
        if (ln == 31) ab_ = ig;  // pools > K in total (= TOPB - need)
    }
    __syncthreads();
    // walk 3: ordered write, pools > K at their rank, then the first `need` pools == K after them
    int* out = sel + (size_t)t * MAXSEL;
    const unsigned G = ab_;
    if (wgt[w] || (weq[w] && oeq[w] < need)) {
        unsigned jg = ogt[w], je = oeq[w];
        const unsigned gend = ogt[w] + wgt[w], eend = min(need, oeq[w] + weq[w]);
        // review r27-L5: jg / je are warp-uniform; the warp stops once its pools > K and its taken pools == K are written
        for (int b = e0; b < e1 && (jg < gend || je < eend); b += 32 * SXU) {
            unsigned k[SXU];
#pragma unroll
            for (int u = 0; u < SXU; u++) { const int p = b + u * 32 + ln; k[u] = p < e1 ? fkey(s[p]) : 0u; }
#pragma unroll
            for (int u = 0; u < SXU; u++) {
                const int p = b + u * 32 + ln;
                const bool v = p < e1, fg = v && k[u] > K, fe = v && k[u] == K;
                const unsigned mg = __ballot_sync(0xffffffffu, fg), me = __ballot_sync(0xffffffffu, fe);
                const unsigned lt = (1u << ln) - 1u;
                int j = -1;
                if (fg) j = jg + __popc(mg & lt);
                else if (fe && je + __popc(me & lt) < need) j = G + je + __popc(me & lt);
                if (j >= 0)
#pragma unroll
                    for (int q = 0; q < POOL; q++) out[j * POOL + q] = p * POOL + q;
                jg += __popc(mg);
                je += __popc(me);
            }
        }
    }
    if (tid == 0) {
        int n = TOPB * POOL;
        for (int q = vis * POOL; q <= pos; q++) out[n++] = q;  // incomplete tail
        nsel[t] = n;
    }
}
void kernel_env_init() {  // review r27 L4: the kernels' switches read once on the host before the per-card inits
    if (g_sel2 < 0) { const char* e = getenv("GLM_SEL2"); g_sel2 = e ? atoi(e) : 1; }
    if (g_idx_dec < 0) { const char* e = getenv("GLM_IDX_DEC"); g_idx_dec = e ? atoi(e) : 1; }
}
void sel2_init() {  // round 27: GLM_SEL2 and the shared-memory attribute of k_select2 (per card at load)
    if (g_sel2 < 0) { const char* e = getenv("GLM_SEL2"); g_sel2 = e ? atoi(e) : 1; }
    CUDA_CHECK(cudaFuncSetAttribute(k_select2, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)SXSM));
}
void select_tokens(const float* sc, int ld, int n0, int* sel, int* nsel, int T, cudaStream_t st, const int* dp,
                   int* psel) {
    if (g_sel2 < 0) { const char* e = getenv("GLM_SEL2"); g_sel2 = e ? atoi(e) : 1; }
    if (g_sel2 && dp && !psel && n0 == 0) { k_select2<<<T, SXT, SXSM, st>>>(sc, ld, sel, nsel, dp); return; }
    k_select<<<T, 512, 0, st>>>(sc, ld, n0, sel, nsel, dp, psel);
}
// pools of each query (psel, pnsel = TOPB or -1 dense) -> token lists as k_select writes them (pools in list order,
// then the incomplete tail)
__global__ void k_sel_expand(const int* psel, const int* pnsel, int n0, int* sel, int* nsel) {
    const int t = blockIdx.x, pos = n0 + t, vis = (pos + 1) / POOL;
    if (pnsel[t] < 0) { if (threadIdx.x == 0) nsel[t] = -1; return; }
    int* out = sel + (size_t)t * MAXSEL;
    for (int i = threadIdx.x; i < TOPB * POOL; i += blockDim.x) out[i] = psel[(size_t)t * TOPB + i / POOL] * POOL + i % POOL;
    if (threadIdx.x == 0) {
        int n = TOPB * POOL;
        for (int q = vis * POOL; q <= pos; q++) out[n++] = q;
        nsel[t] = n;
    }
}
void sel_expand(const int* psel, const int* pnsel, int n0, int* sel, int* nsel, int T, cudaStream_t st) {
    k_sel_expand<<<T, 256, 0, st>>>(psel, pnsel, n0, sel, nsel);
}

// absorbed MLA attention: ctx[t][h] = softmax(q_lat . c / 16) c over the token list (dense: 0..pos)
// grid (T, ceil(nh / HG)), 256 threads (8 warps), lane holds 16 of the 512 latent dims
constexpr int HG = 4;
__global__ void __launch_bounds__(256) k_mla_attn(const float* qlat, const bf16* kc, const int* sel, const int* nsel,
                                                  float* ctx, int nh, int n0, float scale, const int* dp) {
    if (dp) n0 += *dp;
    __shared__ float m_s[8][HG], l_s[8][HG];
    __shared__ float acc_s[HG][KVLR];
    const int t = blockIdx.x, hb = blockIdx.y * HG, w = threadIdx.x >> 5, ln = threadIdx.x & 31;
    const int nhg = min(HG, nh - hb);
    float q[HG][16];
    for (int h = 0; h < HG; h++)
        for (int i = 0; i < 16; i++)
            q[h][i] = h < nhg ? qlat[((size_t)t * nh + hb + h) * KVLR + ln * 16 + i] * scale : 0.f;
    float m[HG], l[HG], a[HG][16];
    for (int h = 0; h < HG; h++) { m[h] = -FLT_MAX; l[h] = 0.f; for (int i = 0; i < 16; i++) a[h][i] = 0.f; }
    const int ns = nsel[t];
    const int n = ns < 0 ? n0 + t + 1 : ns;
    const int* sl = sel + (size_t)t * MAXSEL;
    for (int j = w; j < n; j += 8) {
        const int tok = ns < 0 ? j : sl[j];
        const uint4* c4 = reinterpret_cast<const uint4*>(kc + (size_t)tok * c_kst + ln * 16);
        float c[16];
#pragma unroll
        for (int i = 0; i < 2; i++) {
            const uint4 u = c4[i];
            const bf16* b = reinterpret_cast<const bf16*>(&u);
#pragma unroll
            for (int j = 0; j < 8; j++) c[8 * i + j] = __bfloat162float(b[j]);
        }
#pragma unroll
        for (int h = 0; h < HG; h++) {
            float d = 0.f;
#pragma unroll
            for (int i = 0; i < 16; i++) d += q[h][i] * c[i];
            d = warp_sum(d);
            const float mn = fmaxf(m[h], d);
            const float f = __expf(m[h] - mn), e = __expf(d - mn);
            l[h] = l[h] * f + e;
#pragma unroll
            for (int i = 0; i < 16; i++) a[h][i] = a[h][i] * f + e * c[i];
            m[h] = mn;
        }
    }
    if (ln == 0)
        for (int h = 0; h < HG; h++) { m_s[w][h] = m[h]; l_s[w][h] = l[h]; }
    __syncthreads();
    // global max / sum per head, then scaled sums of the warps' accumulators
    float gm[HG], gl[HG];
    for (int h = 0; h < HG; h++) {
        gm[h] = -FLT_MAX;
        for (int ww = 0; ww < 8; ww++) gm[h] = fmaxf(gm[h], m_s[ww][h]);
        gl[h] = 0.f;
        for (int ww = 0; ww < 8; ww++) gl[h] += m_s[ww][h] == -FLT_MAX ? 0.f : l_s[ww][h] * __expf(m_s[ww][h] - gm[h]);
    }
    for (int ww = 0; ww < 8; ww++) {
        if (w == ww)
            for (int h = 0; h < HG; h++) {
                const float f = m[h] == -FLT_MAX ? 0.f : __expf(m[h] - gm[h]);
                for (int i = 0; i < 16; i++) {
                    float* d = &acc_s[h][ln * 16 + i];
                    *d = (ww == 0 ? 0.f : *d) + a[h][i] * f;
                }
            }
        __syncthreads();
    }
    for (int i = threadIdx.x; i < nhg * KVLR; i += 256) {
        const int h = i / KVLR, c = i % KVLR;
        ctx[((size_t)t * nh + hb + h) * KVLR + c] = acc_s[h][c] / gl[h];
    }
}
// split-K form for decode rows (GLM_MLA_SPLIT): grid (T, ceil(nh / HG), MSPL); each block runs the same warp loop over
// its contiguous share of the token list and writes (max, sum, unnormalised acc) per head; k_mla_comb merges the splits
// in split order (deterministic)
constexpr int MSPL = 8;
__global__ void __launch_bounds__(256) k_mla_attn_sk(const float* qlat, const bf16* kc, const int* sel, const int* nsel,
                                                     float* pm, float* pl, float* pacc, int nh, int n0, float scale,
                                                     const int* dp) {
    if (dp) n0 += *dp;
    __shared__ float m_s[8][HG], l_s[8][HG];
    __shared__ float acc_s[HG][KVLR];
    const int t = blockIdx.x, hb = blockIdx.y * HG, sp = blockIdx.z, w = threadIdx.x >> 5, ln = threadIdx.x & 31;
    const int nhg = min(HG, nh - hb);
    float q[HG][16];
    for (int h = 0; h < HG; h++)
        for (int i = 0; i < 16; i++)
            q[h][i] = h < nhg ? qlat[((size_t)t * nh + hb + h) * KVLR + ln * 16 + i] * scale : 0.f;
    float m[HG], l[HG], a[HG][16];
    for (int h = 0; h < HG; h++) { m[h] = -FLT_MAX; l[h] = 0.f; for (int i = 0; i < 16; i++) a[h][i] = 0.f; }
    const int ns = nsel[t];
    const int n = ns < 0 ? n0 + t + 1 : ns;
    const int ch = (n + MSPL - 1) / MSPL, j0 = sp * ch, j1 = min(n, j0 + ch);
    const int* sl = sel + (size_t)t * MAXSEL;
    // software-pipelined: the next token's latents are loaded before this token's math (latency-bound otherwise)
    auto ldc = [&](int j, uint4* u) {
        const int tok = ns < 0 ? j : sl[j];
        const uint4* c4 = reinterpret_cast<const uint4*>(kc + (size_t)tok * c_kst + ln * 16);
        u[0] = c4[0];
        u[1] = c4[1];
    };
    uint4 un[2] = {};
    if (j0 + w < j1) ldc(j0 + w, un);
    for (int j = j0 + w; j < j1; j += 8) {
        const uint4 u0 = un[0], u1 = un[1];
        if (j + 8 < j1) ldc(j + 8, un);
        float c[16];
        {
            const bf16* b0 = reinterpret_cast<const bf16*>(&u0);
            const bf16* b1 = reinterpret_cast<const bf16*>(&u1);
#pragma unroll
            for (int jj = 0; jj < 8; jj++) { c[jj] = __bfloat162float(b0[jj]); c[8 + jj] = __bfloat162float(b1[jj]); }
        }
#pragma unroll
        for (int h = 0; h < HG; h++) {
            float d = 0.f;
#pragma unroll
            for (int i = 0; i < 16; i++) d += q[h][i] * c[i];
            d = warp_sum(d);
            const float mn = fmaxf(m[h], d);
            const float f = __expf(m[h] - mn), e = __expf(d - mn);
            l[h] = l[h] * f + e;
#pragma unroll
            for (int i = 0; i < 16; i++) a[h][i] = a[h][i] * f + e * c[i];
            m[h] = mn;
        }
    }
    if (ln == 0)
        for (int h = 0; h < HG; h++) { m_s[w][h] = m[h]; l_s[w][h] = l[h]; }
    __syncthreads();
    float gm[HG], gl[HG];
    for (int h = 0; h < HG; h++) {
        gm[h] = -FLT_MAX;
        for (int ww = 0; ww < 8; ww++) gm[h] = fmaxf(gm[h], m_s[ww][h]);
        gl[h] = 0.f;
        for (int ww = 0; ww < 8; ww++) gl[h] += m_s[ww][h] == -FLT_MAX ? 0.f : l_s[ww][h] * __expf(m_s[ww][h] - gm[h]);
    }
    for (int ww = 0; ww < 8; ww++) {
        if (w == ww)
            for (int h = 0; h < HG; h++) {
                const float f = m[h] == -FLT_MAX ? 0.f : __expf(m[h] - gm[h]);
                for (int i = 0; i < 16; i++) {
                    float* d = &acc_s[h][ln * 16 + i];
                    *d = (ww == 0 ? 0.f : *d) + a[h][i] * f;
                }
            }
        __syncthreads();
    }
    for (int i = threadIdx.x; i < nhg * KVLR; i += 256) {
        const int h = i / KVLR, c = i % KVLR;
        const size_t o = ((size_t)t * nh + hb + h) * MSPL + sp;
        pacc[o * KVLR + c] = acc_s[h][c];
        if (c == 0) { pm[o] = gm[h]; pl[o] = gl[h]; }
    }
}
__global__ void k_mla_comb(const float* pm, const float* pl, const float* pacc, float* ctx) {
    const size_t th = blockIdx.x;  // t * nh + h
    float M = -FLT_MAX;
    for (int s = 0; s < MSPL; s++) M = fmaxf(M, pm[th * MSPL + s]);
    float Lt = 0.f, f[MSPL];
    for (int s = 0; s < MSPL; s++) {
        f[s] = pm[th * MSPL + s] == -FLT_MAX ? 0.f : __expf(pm[th * MSPL + s] - M);
        Lt += pl[th * MSPL + s] * f[s];
    }
    for (int c = threadIdx.x; c < KVLR; c += blockDim.x) {
        float a = 0.f;
        for (int s = 0; s < MSPL; s++) a += pacc[(th * MSPL + s) * KVLR + c] * f[s];
        ctx[th * KVLR + c] = a / Lt;
    }
}
size_t mla_split_scratch(int nh, int T) { return (size_t)T * nh * 32 * (KVLR + 2); }  // 32 = max(MSPL, TDSP)
void mla_attn_split(const float* qlat, const bf16* kc, const int* sel, const int* nsel, float* ctx, int nh, int n0, int T,
                    float scale, cudaStream_t st, const int* dp, float* scr) {
    float* pm = scr;
    float* pl = pm + (size_t)T * nh * MSPL;
    float* pacc = pl + (size_t)T * nh * MSPL;
    k_mla_attn_sk<<<dim3(T, (nh + HG - 1) / HG, MSPL), 256, 0, st>>>(qlat, kc, sel, nsel, pm, pl, pacc, nh, n0, scale, dp);
    k_mla_comb<<<T * nh, 128, 0, st>>>(pm, pl, pacc, ctx);
}
void mla_attn(const float* qlat, const bf16* kc, const int* sel, const int* nsel, float* ctx, int nh, int n0, int T,
              float scale, cudaStream_t st, const int* dp) {
    k_mla_attn<<<dim3(T, (nh + HG - 1) / HG), 256, 0, st>>>(qlat, kc, sel, nsel, ctx, nh, n0, scale, dp);
}


// sparse MLA attention on tensor cores, one block (4 warps) per query: S = Q K^T over the query's own token list in
// tiles of 32 latents (mma m16n8k16 bf16, heads padded to 32), online softmax, O += P V (V = the same latents).
// Q (scaled) and P in bf16, fp32 accumulators (as the union GEMM path). Used where a query block's key union is too big.
constexpr int TCK = 32, TCLD = KVLR + 8, TCPL = TCK + 8;
__device__ __forceinline__ void mma16816(float* c, const unsigned* a, const unsigned* b) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
__device__ __forceinline__ unsigned pack2(bf16 lo, bf16 hi) {
    return (unsigned)__bfloat16_as_ushort(lo) | ((unsigned)__bfloat16_as_ushort(hi) << 16);
}
__global__ void __launch_bounds__(128) k_mla_tc(const float* qlat, const bf16* kc, const int* sel, const int* nsel,
                                               float* ctx, int nh, int n0, int t0, float scale) {
    extern __shared__ __align__(16) unsigned char smem[];
    bf16* Qs = reinterpret_cast<bf16*>(smem);                 // [24][TCLD] (heads <= 22; rows 24..31 of the mma are 0)
    bf16* Kb = Qs + 24 * TCLD;                                // [2][TCK][TCLD]: double buffer filled by cp.async
    bf16* Ps = Kb + 2 * TCK * TCLD;                           // [32][TCPL]
    float* red = reinterpret_cast<float*>(Ps + 32 * TCPL);    // [4][32] partial row max / sum
    float* mrow = red + 4 * 32;                               // [32] running max
    float* lrow = mrow + 32;                                  // [32] running sum
    float* arow = lrow + 32;                                  // [32] rescale of this tile
    const int t = t0 + blockIdx.x, w = threadIdx.x >> 5, ln = threadIdx.x & 31, g = ln >> 2, q4 = ln & 3;
    for (int i = threadIdx.x; i < 24 * KVLR; i += 128) {
        const int h = i / KVLR, d = i % KVLR;
        Qs[h * TCLD + d] = __float2bfloat16(h < nh ? qlat[((size_t)t * nh + h) * KVLR + d] * scale : 0.f);
    }
    if (threadIdx.x < 32) { mrow[threadIdx.x] = -FLT_MAX; lrow[threadIdx.x] = 0.f; }
    const int ns = nsel[t];
    const int n = ns < 0 ? n0 + t + 1 : ns;
    const int* sl = sel + (size_t)t * MAXSEL;
    float o[2][16][4];
#pragma unroll
    for (int mt = 0; mt < 2; mt++)
#pragma unroll
        for (int j = 0; j < 16; j++)
#pragma unroll
            for (int c = 0; c < 4; c++) o[mt][j][c] = 0.f;
    auto fetch = [&](int k0, bf16* dst) {   // async gather of the latents of keys [k0, k0 + TCK) (zero-fill past n)
        for (int i = threadIdx.x; i < TCK * (KVLR / 8); i += 128) {
            const int r = i / (KVLR / 8), c = (i % (KVLR / 8)) * 8;
            const int j = k0 + r;
            const int tk = j < n ? (ns < 0 ? j : sl[j]) : 0;
            const unsigned sa = (unsigned)__cvta_generic_to_shared(dst + r * TCLD + c);
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(sa),
                         "l"(kc + (size_t)tk * c_kst + c), "r"(j < n ? 16 : 0));
        }
        asm volatile("cp.async.commit_group;\n" ::);
    };
    fetch(0, Kb);
    int buf = 0;
    for (int k0 = 0; k0 < n; k0 += TCK, buf ^= 1) {
        const int nk = min(TCK, n - k0);
        bf16* Ks = Kb + buf * TCK * TCLD;
        __syncthreads();   // the other buffer's readers (previous tile) are done
        if (k0 + TCK < n) {
            fetch(k0 + TCK, Kb + (buf ^ 1) * TCK * TCLD);
            asm volatile("cp.async.wait_group 1;\n" ::);
        } else {
            asm volatile("cp.async.wait_group 0;\n" ::);
        }
        __syncthreads();
        // S: warp w -> keys [8w, 8w + 8)
        float s[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
#pragma unroll 4
        for (int kk = 0; kk < KVLR; kk += 16) {
            unsigned b[2];
            const bf16* kr = Ks + (8 * w + g) * TCLD + kk + 2 * q4;
            b[0] = *reinterpret_cast<const unsigned*>(kr);
            b[1] = *reinterpret_cast<const unsigned*>(kr + 8);
#pragma unroll
            for (int mt = 0; mt < 2; mt++) {
                unsigned a[4];
                const bf16* qr = Qs + (16 * mt + g) * TCLD + kk + 2 * q4;
                a[0] = *reinterpret_cast<const unsigned*>(qr);
                a[2] = *reinterpret_cast<const unsigned*>(qr + 8);
                if (mt == 0) {
                    a[1] = *reinterpret_cast<const unsigned*>(qr + 8 * TCLD);
                    a[3] = *reinterpret_cast<const unsigned*>(qr + 8 * TCLD + 8);
                } else {
                    a[1] = 0u; a[3] = 0u;   // heads 24..31
                }
                mma16816(s[mt], a, b);
            }
        }
        // mask padded keys; partial row max over this warp's 8 keys
        const int key0 = 8 * w + 2 * q4;
#pragma unroll
        for (int mt = 0; mt < 2; mt++) {
            if (key0 >= nk) { s[mt][0] = -FLT_MAX; s[mt][2] = -FLT_MAX; }
            if (key0 + 1 >= nk) { s[mt][1] = -FLT_MAX; s[mt][3] = -FLT_MAX; }
#pragma unroll
            for (int hh = 0; hh < 2; hh++) {
                float m = fmaxf(s[mt][2 * hh], s[mt][2 * hh + 1]);
                m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 1));
                m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 2));
                if (q4 == 0) red[w * 32 + 16 * mt + 8 * hh + g] = m;
            }
        }
        __syncthreads();
        if (threadIdx.x < 32) {
            const int r = threadIdx.x;
            const float mo = mrow[r];
            const float mn = fmaxf(mo, fmaxf(fmaxf(red[r], red[32 + r]), fmaxf(red[64 + r], red[96 + r])));
            arow[r] = __expf(mo - mn);
            mrow[r] = mn;
        }
        __syncthreads();
#pragma unroll
        for (int mt = 0; mt < 2; mt++)
#pragma unroll
            for (int hh = 0; hh < 2; hh++) {
                const int r = 16 * mt + 8 * hh + g;
                const float mn = mrow[r];
                const float p0 = s[mt][2 * hh] == -FLT_MAX ? 0.f : __expf(s[mt][2 * hh] - mn);
                const float p1 = s[mt][2 * hh + 1] == -FLT_MAX ? 0.f : __expf(s[mt][2 * hh + 1] - mn);
                const bf16 b0 = __float2bfloat16(p0), b1 = __float2bfloat16(p1);
                *reinterpret_cast<unsigned*>(Ps + r * TCPL + key0) = pack2(b0, b1);
                float sm = __bfloat162float(b0) + __bfloat162float(b1);
                sm += __shfl_xor_sync(0xffffffffu, sm, 1);
                sm += __shfl_xor_sync(0xffffffffu, sm, 2);
                if (q4 == 0) red[w * 32 + r] = sm;
            }
        __syncthreads();
        if (threadIdx.x < 32) {
            const int r = threadIdx.x;
            lrow[r] = lrow[r] * arow[r] + red[r] + red[32 + r] + red[64 + r] + red[96 + r];
        }
        // O = O * alpha + P V ; warp w -> latent dims [128 w, 128 w + 128) = 16 n-tiles
        float al[2][2];
#pragma unroll
        for (int mt = 0; mt < 2; mt++) { al[mt][0] = arow[16 * mt + g]; al[mt][1] = arow[16 * mt + 8 + g]; }
#pragma unroll
        for (int mt = 0; mt < 2; mt++)
#pragma unroll
            for (int j = 0; j < 16; j++) {
                o[mt][j][0] *= al[mt][0]; o[mt][j][1] *= al[mt][0];
                o[mt][j][2] *= al[mt][1]; o[mt][j][3] *= al[mt][1];
            }
#pragma unroll
        for (int kk = 0; kk < TCK; kk += 16) {
            unsigned a[2][4];
#pragma unroll
            for (int mt = 0; mt < 2; mt++) {
                const bf16* pr = Ps + (16 * mt + g) * TCPL + kk + 2 * q4;
                a[mt][0] = *reinterpret_cast<const unsigned*>(pr);
                a[mt][1] = *reinterpret_cast<const unsigned*>(pr + 8 * TCPL);
                a[mt][2] = *reinterpret_cast<const unsigned*>(pr + 8);
                a[mt][3] = *reinterpret_cast<const unsigned*>(pr + 8 * TCPL + 8);
            }
#pragma unroll
            for (int j = 0; j < 16; j++) {
                const int d = 128 * w + 8 * j + g;
                const bf16* vr = Ks + (kk + 2 * q4) * TCLD + d;
                unsigned b[2];
                b[0] = pack2(vr[0], vr[TCLD]);
                b[1] = pack2(vr[8 * TCLD], vr[9 * TCLD]);
                mma16816(o[0][j], a[0], b);
                mma16816(o[1][j], a[1], b);
            }
        }
    }
    __syncthreads();
#pragma unroll
    for (int mt = 0; mt < 2; mt++)
#pragma unroll
        for (int hh = 0; hh < 2; hh++) {
            const int h = 16 * mt + 8 * hh + g;
            if (h >= nh) continue;
            const float inv = 1.f / lrow[h];
#pragma unroll
            for (int j = 0; j < 16; j++) {
                const int d = 128 * w + 8 * j + 2 * q4;
                float2 v = make_float2(o[mt][j][2 * hh] * inv, o[mt][j][2 * hh + 1] * inv);
                *reinterpret_cast<float2*>(ctx + ((size_t)t * nh + h) * KVLR + d) = v;
            }
        }
}
void mla_attn_tc(const float* qlat, const bf16* kc, const int* sel, const int* nsel, float* ctx, int nh, int n0, int t0,
                 int T, float scale, cudaStream_t st) {
    const size_t sm = (size_t)(24 * TCLD + 2 * TCK * TCLD + 32 * TCPL) * 2 + (4 * 32 + 3 * 32) * 4;
    static bool init[16] = {};   // the attribute is per device
    int dev = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    if (!init[dev]) { CUDA_CHECK(cudaFuncSetAttribute(k_mla_tc, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm)); init[dev] = true; }
    k_mla_tc<<<T, 128, sm, st>>>(qlat, kc, sel, nsel, ctx, nh, n0, t0, scale);
    CUDA_CHECK(cudaGetLastError());
}


// round 8: decode form of k_mla_tc (GLM_MLA_DTC): grid (T, nsp), each block one contiguous share of the token list.
// Q and P are carried as bf16 hi + lo pairs (two mma each), so scores and weights keep ~fp32 precision; the row sums
// use the fp32 p. Writes (max, sum, unnormalised acc) per head and split; k_mla_comb_n merges in split order.
constexpr int TDSP = 32;   // splits at most (scratch)
template <bool QLO, bool PLO>
__global__ void __launch_bounds__(128) k_mla_tc_dec(const float* qlat, const bf16* kc, const int* sel, const int* nsel,
                                                   float* pm, float* pl, float* pacc, int nh, int n0, float scale,
                                                   const int* dp, int nsp) {
    if (dp) n0 += *dp;
    extern __shared__ __align__(16) unsigned char smem[];
    bf16* Qh = reinterpret_cast<bf16*>(smem);                 // [24][TCLD]
    bf16* Ql = Qh + 24 * TCLD;                                // [24][TCLD]
    bf16* Ks = Ql + 24 * TCLD;                                // [TCK][TCLD]
    bf16* Ph = Ks + TCK * TCLD;                               // [32][TCPL]
    bf16* Pl = Ph + 32 * TCPL;                                // [32][TCPL]
    float* red = reinterpret_cast<float*>(Pl + 32 * TCPL);    // [4][32]
    float* mrow = red + 4 * 32;
    float* lrow = mrow + 32;
    float* arow = lrow + 32;
    const int t = blockIdx.x, sp = blockIdx.y, w = threadIdx.x >> 5, ln = threadIdx.x & 31, g = ln >> 2, q4 = ln & 3;
#pragma unroll 8
    for (int i = threadIdx.x; i < 24 * KVLR / 4; i += 128) {   // float4 loads, all in flight (latency-bound otherwise)
        const int h = i / (KVLR / 4), d = (i % (KVLR / 4)) * 4;
        const float4 v4 = h < nh ? *reinterpret_cast<const float4*>(qlat + ((size_t)t * nh + h) * KVLR + d)
                                 : make_float4(0.f, 0.f, 0.f, 0.f);
        const float v[4] = {v4.x * scale, v4.y * scale, v4.z * scale, v4.w * scale};
        bf16 hi[4], lo[4];
#pragma unroll
        for (int k = 0; k < 4; k++) { hi[k] = __float2bfloat16(v[k]); lo[k] = __float2bfloat16(v[k] - __bfloat162float(hi[k])); }
        *reinterpret_cast<uint2*>(Qh + h * TCLD + d) = make_uint2(pack2(hi[0], hi[1]), pack2(hi[2], hi[3]));
        *reinterpret_cast<uint2*>(Ql + h * TCLD + d) = make_uint2(pack2(lo[0], lo[1]), pack2(lo[2], lo[3]));
    }
    if (threadIdx.x < 32) { mrow[threadIdx.x] = -FLT_MAX; lrow[threadIdx.x] = 0.f; }
    const int ns = nsel[t];
    const int n = ns < 0 ? n0 + t + 1 : ns;
    const int ch = (n + nsp - 1) / nsp, j0 = min(n, sp * ch), j1 = min(n, j0 + ch);
    const int* sl = sel + (size_t)t * MAXSEL;
    float o[2][16][4];
#pragma unroll
    for (int mt = 0; mt < 2; mt++)
#pragma unroll
        for (int j = 0; j < 16; j++)
#pragma unroll
            for (int c = 0; c < 4; c++) o[mt][j][c] = 0.f;
    for (int k0 = j0; k0 < j1; k0 += TCK) {
        const int nk = min(TCK, j1 - k0);
        __syncthreads();   // previous tile's readers are done
        {   // thread -> one key row (one index load), 16 chunks of 16 bytes
            const int r = threadIdx.x >> 2, c0 = (threadIdx.x & 3) * 128;
            const int j = k0 + r;
            const int tk = r < nk ? (ns < 0 ? j : sl[j]) : 0;
            const bf16* src = kc + (size_t)tk * c_kst + c0;
            bf16* dst = Ks + r * TCLD + c0;
#pragma unroll
            for (int c = 0; c < 128; c += 8) {
                const unsigned sa = (unsigned)__cvta_generic_to_shared(dst + c);
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(sa), "l"(src + c), "r"(r < nk ? 16 : 0));
            }
        }
        asm volatile("cp.async.commit_group;\n" ::);
        asm volatile("cp.async.wait_group 0;\n" ::);
        __syncthreads();
        float s[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
#pragma unroll 4
        for (int kk = 0; kk < KVLR; kk += 16) {
            unsigned b[2];
            const bf16* kr = Ks + (8 * w + g) * TCLD + kk + 2 * q4;
            b[0] = *reinterpret_cast<const unsigned*>(kr);
            b[1] = *reinterpret_cast<const unsigned*>(kr + 8);
#pragma unroll
            for (int pq = 0; pq < (QLO ? 2 : 1); pq++) {
                const bf16* Qs = pq ? Ql : Qh;
#pragma unroll
                for (int mt = 0; mt < 2; mt++) {
                    unsigned a[4];
                    const bf16* qr = Qs + (16 * mt + g) * TCLD + kk + 2 * q4;
                    a[0] = *reinterpret_cast<const unsigned*>(qr);
                    a[2] = *reinterpret_cast<const unsigned*>(qr + 8);
                    if (mt == 0) {
                        a[1] = *reinterpret_cast<const unsigned*>(qr + 8 * TCLD);
                        a[3] = *reinterpret_cast<const unsigned*>(qr + 8 * TCLD + 8);
                    } else {
                        a[1] = 0u; a[3] = 0u;
                    }
                    mma16816(s[mt], a, b);
                }
            }
        }
        const int key0 = 8 * w + 2 * q4;
#pragma unroll
        for (int mt = 0; mt < 2; mt++) {
            if (key0 >= nk) { s[mt][0] = -FLT_MAX; s[mt][2] = -FLT_MAX; }
            if (key0 + 1 >= nk) { s[mt][1] = -FLT_MAX; s[mt][3] = -FLT_MAX; }
#pragma unroll
            for (int hh = 0; hh < 2; hh++) {
                float m = fmaxf(s[mt][2 * hh], s[mt][2 * hh + 1]);
                m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 1));
                m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 2));
                if (q4 == 0) red[w * 32 + 16 * mt + 8 * hh + g] = m;
            }
        }
        __syncthreads();
        if (threadIdx.x < 32) {
            const int r = threadIdx.x;
            const float mo = mrow[r];
            const float mn = fmaxf(mo, fmaxf(fmaxf(red[r], red[32 + r]), fmaxf(red[64 + r], red[96 + r])));
            arow[r] = mo == -FLT_MAX ? 0.f : __expf(mo - mn);
            mrow[r] = mn;
        }
        __syncthreads();
#pragma unroll
        for (int mt = 0; mt < 2; mt++)
#pragma unroll
            for (int hh = 0; hh < 2; hh++) {
                const int r = 16 * mt + 8 * hh + g;
                const float mn = mrow[r];
                const float p0 = s[mt][2 * hh] == -FLT_MAX ? 0.f : __expf(s[mt][2 * hh] - mn);
                const float p1 = s[mt][2 * hh + 1] == -FLT_MAX ? 0.f : __expf(s[mt][2 * hh + 1] - mn);
                const bf16 h0 = __float2bfloat16(p0), h1 = __float2bfloat16(p1);
                *reinterpret_cast<unsigned*>(Ph + r * TCPL + key0) = pack2(h0, h1);
                *reinterpret_cast<unsigned*>(Pl + r * TCPL + key0) =
                    pack2(__float2bfloat16(p0 - __bfloat162float(h0)), __float2bfloat16(p1 - __bfloat162float(h1)));
                float sm = p0 + p1;
                sm += __shfl_xor_sync(0xffffffffu, sm, 1);
                sm += __shfl_xor_sync(0xffffffffu, sm, 2);
                if (q4 == 0) red[w * 32 + r] = sm;
            }
        __syncthreads();
        if (threadIdx.x < 32) {
            const int r = threadIdx.x;
            lrow[r] = lrow[r] * arow[r] + red[r] + red[32 + r] + red[64 + r] + red[96 + r];
        }
        float al[2][2];
#pragma unroll
        for (int mt = 0; mt < 2; mt++) { al[mt][0] = arow[16 * mt + g]; al[mt][1] = arow[16 * mt + 8 + g]; }
#pragma unroll
        for (int mt = 0; mt < 2; mt++)
#pragma unroll
            for (int j = 0; j < 16; j++) {
                o[mt][j][0] *= al[mt][0]; o[mt][j][1] *= al[mt][0];
                o[mt][j][2] *= al[mt][1]; o[mt][j][3] *= al[mt][1];
            }
#pragma unroll
        for (int kk = 0; kk < TCK; kk += 16) {
            unsigned a[2][2][4];
#pragma unroll
            for (int pq = 0; pq < 2; pq++)
#pragma unroll
                for (int mt = 0; mt < 2; mt++) {
                    const bf16* pr = (pq ? Pl : Ph) + (16 * mt + g) * TCPL + kk + 2 * q4;
                    a[pq][mt][0] = *reinterpret_cast<const unsigned*>(pr);
                    a[pq][mt][1] = *reinterpret_cast<const unsigned*>(pr + 8 * TCPL);
                    a[pq][mt][2] = *reinterpret_cast<const unsigned*>(pr + 8);
                    a[pq][mt][3] = *reinterpret_cast<const unsigned*>(pr + 8 * TCPL + 8);
                }
#pragma unroll
            for (int j = 0; j < 16; j++) {
                const int d = 128 * w + 8 * j + g;
                const bf16* vr = Ks + (kk + 2 * q4) * TCLD + d;
                unsigned b[2];
                b[0] = pack2(vr[0], vr[TCLD]);
                b[1] = pack2(vr[8 * TCLD], vr[9 * TCLD]);
                mma16816(o[0][j], a[0][0], b);
                mma16816(o[1][j], a[0][1], b);
                if (PLO) {
                    mma16816(o[0][j], a[1][0], b);
                    mma16816(o[1][j], a[1][1], b);
                }
            }
        }
    }
    __syncthreads();
#pragma unroll
    for (int mt = 0; mt < 2; mt++)
#pragma unroll
        for (int hh = 0; hh < 2; hh++) {
            const int h = 16 * mt + 8 * hh + g;
            if (h >= nh) continue;
            const size_t ob = ((size_t)t * nh + h) * nsp + sp;
#pragma unroll
            for (int j = 0; j < 16; j++) {
                const int d = 128 * w + 8 * j + 2 * q4;
                *reinterpret_cast<float2*>(pacc + ob * KVLR + d) = make_float2(o[mt][j][2 * hh], o[mt][j][2 * hh + 1]);
            }
            if (w == 0 && q4 == 0) { pm[ob] = mrow[h]; pl[ob] = lrow[h]; }
        }
}
__global__ void k_mla_comb_n(const float* pm, const float* pl, const float* pacc, float* ctx, int nsp) {
    const size_t th = blockIdx.x;  // t * nh + h
    // split factors in shared memory (a runtime-indexed f[TDSP] was a 128-byte stack frame per thread); same values
    // and the same summation order as before
    __shared__ float f[TDSP];
    float M = -FLT_MAX;
    for (int s = 0; s < nsp; s++) M = fmaxf(M, pm[th * nsp + s]);
    if (threadIdx.x < nsp) {
        const float v = pm[th * nsp + threadIdx.x];
        f[threadIdx.x] = v == -FLT_MAX ? 0.f : __expf(v - M);
    }
    __syncthreads();
    float Lt = 0.f;
    for (int s = 0; s < nsp; s++) Lt += pl[th * nsp + s] * f[s];
    for (int c = threadIdx.x; c < KVLR; c += blockDim.x) {
        float a = 0.f;
        for (int s = 0; s < nsp; s++) a += pacc[(th * nsp + s) * KVLR + c] * f[s];
        ctx[th * KVLR + c] = a / Lt;
    }
}
void mla_attn_tc_dec(const float* qlat, const bf16* kc, const int* sel, const int* nsel, float* ctx, int nh, int n0,
                     int T, float scale, cudaStream_t st, const int* dp, float* scr, int nsp) {
    nsp = std::max(1, std::min(nsp, TDSP));
    float* pm = scr;
    float* pl = pm + (size_t)T * nh * nsp;
    float* pacc = pl + (size_t)T * nh * nsp;
    const size_t sm = (size_t)(48 * TCLD + TCK * TCLD + 64 * TCPL) * 2 + (4 * 32 + 3 * 32) * 4;
    static bool init[16] = {};
    int dev = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    static const int prec = getenv("GLM_MLA_TC_PREC") ? atoi(getenv("GLM_MLA_TC_PREC")) : 3;  // bit 0: Q lo, bit 1: P lo
    auto kf = prec == 3 ? k_mla_tc_dec<true, true> : prec == 1 ? k_mla_tc_dec<true, false>
            : prec == 2 ? k_mla_tc_dec<false, true> : k_mla_tc_dec<false, false>;
    if (!init[dev]) {
        for (auto f : {k_mla_tc_dec<true, true>, k_mla_tc_dec<true, false>, k_mla_tc_dec<false, true>, k_mla_tc_dec<false, false>})
            CUDA_CHECK(cudaFuncSetAttribute(f, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm));
        init[dev] = true;
    }
    kf<<<dim3(T, nsp), 128, sm, st>>>(qlat, kc, sel, nsel, pm, pl, pacc, nh, n0, scale, dp, nsp);
    k_mla_comb_n<<<T * nh, 128, 0, st>>>(pm, pl, pacc, ctx, nsp);
}

// round 8: absorbed-MLA up projections from bf16 weights (the checkpoint's own precision; fp32 accumulation) for decode
// rows: qlat[t][h][c] = sum_d wuk[h][d][c] q[t][h * DN + d]; ov[t][h][v] = sum_c wuv[h][v][c] ctx[t][h][c]
// k_uk: grid (nh, KVLR / 128), 256 threads = 16 column groups of 8 x 16 d-groups of 16; fixed-order smem reduction
__global__ void __launch_bounds__(256) k_mla_uk(const bf16* wuk, const float* qq, int ldq, float* qlat, int nh, int T) {
    __shared__ float red[16][MR][128];
    const int h = blockIdx.x, c0 = blockIdx.y * 128, cg = threadIdx.x & 15, dg = threadIdx.x >> 4;
    float acc[MR][8];
#pragma unroll
    for (int r = 0; r < MR; r++)
#pragma unroll
        for (int i = 0; i < 8; i++) acc[r][i] = 0.f;
    const bf16* wb = wuk + (size_t)h * DN * KVLR + c0 + cg * 8;
#pragma unroll 4
    for (int dd = 0; dd < 16; dd++) {
        const int d = dg * 16 + dd;
        const uint4 u = *reinterpret_cast<const uint4*>(wb + (size_t)d * KVLR);
        const bf16* b = reinterpret_cast<const bf16*>(&u);
        float wv[8];
#pragma unroll
        for (int i = 0; i < 8; i++) wv[i] = __bfloat162float(b[i]);
#pragma unroll
        for (int r = 0; r < MR; r++) {
            if (r >= T) break;
            const float x = qq[(size_t)r * ldq + h * DN + d];
#pragma unroll
            for (int i = 0; i < 8; i++) acc[r][i] += wv[i] * x;
        }
    }
#pragma unroll
    for (int r = 0; r < MR; r++) {
        if (r >= T) break;
#pragma unroll
        for (int i = 0; i < 8; i++) red[dg][r][cg * 8 + i] = acc[r][i];
    }
    __syncthreads();
    for (int i = threadIdx.x; i < T * 128; i += 256) {
        const int r = i / 128, c = i % 128;
        float a = 0.f;
        for (int g2 = 0; g2 < 16; g2++) a += red[g2][r][c];
        qlat[((size_t)r * nh + h) * KVLR + c0 + c] = a;
    }
}
// k_uv: a warp per (h, v), 8 warps per block
__global__ void __launch_bounds__(256) k_mla_uv(const bf16* wuv, const float* ctx, float* ov, int nh, int T) {
    const int wid = blockIdx.x * 8 + (threadIdx.x >> 5), ln = threadIdx.x & 31;
    if (wid >= nh * DV) return;
    const int h = wid / DV, v = wid % DV;
    const bf16* wr = wuv + ((size_t)h * DV + v) * KVLR;
    float acc[MR] = {};
#pragma unroll
    for (int it = 0; it < 2; it++) {
        const int c = it * 256 + ln * 8;
        const uint4 u = *reinterpret_cast<const uint4*>(wr + c);
        const bf16* b = reinterpret_cast<const bf16*>(&u);
#pragma unroll
        for (int r = 0; r < MR; r++) {
            if (r >= T) break;
            const float* x = ctx + ((size_t)r * nh + h) * KVLR + c;
            const float4 x0 = *reinterpret_cast<const float4*>(x), x1 = *reinterpret_cast<const float4*>(x + 4);
            acc[r] += __bfloat162float(b[0]) * x0.x + __bfloat162float(b[1]) * x0.y + __bfloat162float(b[2]) * x0.z +
                      __bfloat162float(b[3]) * x0.w + __bfloat162float(b[4]) * x1.x + __bfloat162float(b[5]) * x1.y +
                      __bfloat162float(b[6]) * x1.z + __bfloat162float(b[7]) * x1.w;
        }
    }
#pragma unroll
    for (int r = 0; r < MR; r++) {
        if (r >= T) break;
        const float a = warp_sum(acc[r]);
        if (ln == 0) ov[((size_t)r * nh + h) * DV + v] = a;
    }
}
void mla_uk_bf16(const bf16* wuk, const float* qq, int ldq, float* qlat, int nh, int T, cudaStream_t st) {
    k_mla_uk<<<dim3(nh, KVLR / 128), 256, 0, st>>>(wuk, qq, ldq, qlat, nh, T);
}
void mla_uv_bf16(const bf16* wuv, const float* ctx, float* ov, int nh, int T, cudaStream_t st) {
    k_mla_uv<<<(nh * DV + 7) / 8, 256, 0, st>>>(wuv, ctx, ov, nh, T);
}

// indexer scores on tensor cores (prefill): 8 queries x 32 heads (= 256 rows, a warp per query) against tiles of 64
// pooled keys, bf16 mma with fp32 accumulators, epilogue s[p] = sum_h w_h relu(acc / sqrt(128)) reduced in the warp.
// grid (pool tiles, ceil(T / 8)), 256 threads. Values for p >= vis of a query are not written (never read).
constexpr int IXQ = 8, IXP = 64, IXLD = ID + 8;
__global__ void __launch_bounds__(256) k_idx_tc(const float* qi, int ldq, const float* wi, const float* pk, float* sc,
                                               int ld, int n0, int T) {
    extern __shared__ __align__(16) unsigned char smem[];
    bf16* Qs = reinterpret_cast<bf16*>(smem);          // [IXQ * 32][IXLD]
    bf16* Ks = Qs + IXQ * IH * IXLD;                    // [IXP][IXLD]
    const int w = threadIdx.x >> 5, ln = threadIdx.x & 31, g = ln >> 2, q4 = ln & 3;
    const int tq0 = blockIdx.y * IXQ, p0 = blockIdx.x * IXP;
    const int tl = min(T - 1, tq0 + IXQ - 1);
    if (p0 >= (n0 + tl + 1) / POOL) return;
    for (int i = threadIdx.x; i < IXQ * IH * ID; i += 256) {
        const int r = i / ID, d = i % ID, t = tq0 + r / IH, h = r % IH;
        Qs[r * IXLD + d] = __float2bfloat16(t < T ? qi[(size_t)t * ldq + h * ID + d] : 0.f);
    }
    for (int i = threadIdx.x; i < IXP * ID; i += 256) {
        const int r = i / ID, d = i % ID, p = p0 + r;
        Ks[r * IXLD + d] = __float2bfloat16(p < ld ? pk[(size_t)p * c_pst + d] : 0.f);
    }
    __syncthreads();
    const int t = tq0 + w;
    if (t >= T) return;
    const int vis = (n0 + t + 1) / POOL;
    if (p0 >= vis) return;
    float acc[2][8][4];
#pragma unroll
    for (int mt = 0; mt < 2; mt++)
#pragma unroll
        for (int j = 0; j < 8; j++)
#pragma unroll
            for (int c = 0; c < 4; c++) acc[mt][j][c] = 0.f;
#pragma unroll
    for (int kk = 0; kk < ID; kk += 16) {
        unsigned a[2][4];
#pragma unroll
        for (int mt = 0; mt < 2; mt++) {
            const bf16* qr = Qs + (w * IH + 16 * mt + g) * IXLD + kk + 2 * q4;
            a[mt][0] = *reinterpret_cast<const unsigned*>(qr);
            a[mt][1] = *reinterpret_cast<const unsigned*>(qr + 8 * IXLD);
            a[mt][2] = *reinterpret_cast<const unsigned*>(qr + 8);
            a[mt][3] = *reinterpret_cast<const unsigned*>(qr + 8 * IXLD + 8);
        }
#pragma unroll
        for (int j = 0; j < 8; j++) {
            const bf16* kr = Ks + (8 * j + g) * IXLD + kk + 2 * q4;
            unsigned b[2];
            b[0] = *reinterpret_cast<const unsigned*>(kr);
            b[1] = *reinterpret_cast<const unsigned*>(kr + 8);
            mma16816(acc[0][j], a[0], b);
            mma16816(acc[1][j], a[1], b);
        }
    }
    const float rs = rsqrtf((float)ID);
    const float* ww = wi + (size_t)t * IH;
    const float w0 = ww[g], w1 = ww[g + 8], w2 = ww[16 + g], w3 = ww[24 + g];
#pragma unroll
    for (int j = 0; j < 8; j++) {
        float s0 = w0 * fmaxf(acc[0][j][0] * rs, 0.f) + w1 * fmaxf(acc[0][j][2] * rs, 0.f) +
                   w2 * fmaxf(acc[1][j][0] * rs, 0.f) + w3 * fmaxf(acc[1][j][2] * rs, 0.f);
        float s1 = w0 * fmaxf(acc[0][j][1] * rs, 0.f) + w1 * fmaxf(acc[0][j][3] * rs, 0.f) +
                   w2 * fmaxf(acc[1][j][1] * rs, 0.f) + w3 * fmaxf(acc[1][j][3] * rs, 0.f);
#pragma unroll
        for (int o = 4; o < 32; o <<= 1) {
            s0 += __shfl_xor_sync(0xffffffffu, s0, o);
            s1 += __shfl_xor_sync(0xffffffffu, s1, o);
        }
        const int p = p0 + 8 * j + 2 * q4;
        if (g == 0) {
            if (p < vis) sc[(size_t)t * ld + p] = s0;
            if (p + 1 < vis) sc[(size_t)t * ld + p + 1] = s1;
        }
    }
}
void idx_score_tc(const float* qi, int ldq, const float* wi, const float* pk, float* sc, int ld, int n0, int T,
                  cudaStream_t st) {
    const size_t sm = (size_t)(IXQ * IH + IXP) * IXLD * 2;
    static bool init[16] = {};   // per device
    int dev = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    if (!init[dev]) { CUDA_CHECK(cudaFuncSetAttribute(k_idx_tc, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm)); init[dev] = true; }
    const int vmax = (n0 + T) / POOL;
    k_idx_tc<<<dim3((vmax + IXP - 1) / IXP, (T + IXQ - 1) / IXQ), 256, sm, st>>>(qi, ldq, wi, pk, sc, ld, n0, T);
    CUDA_CHECK(cudaGetLastError());
}

// ---------------------------------------------------------------- MLP / MoE

// sigmoid router: top-8 by score + bias, weights = normalised scores * 2.5 ; one warp per token
__device__ __forceinline__ void route_row(const float* lrow, const float* bias, int* ids, float* wts, int t, int ln);
// round 17 gate (GLM_TAU_ALL=1, off): a residency-independent skip, every pick with weight < tau * RSF gets weight 0
__device__ float g_tau_all = 0.f;
void set_tau_all(float v) { CUDA_CHECK(cudaMemcpyToSymbol(g_tau_all, &v, sizeof v)); }
__global__ void k_route(const float* logits, const float* bias, int* ids, float* wts, int T) {
    const int t = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5), ln = threadIdx.x & 31;
    if (t >= T) return;
    route_row(logits + (size_t)t * NE, bias, ids, wts, t, ln);
}
__device__ __forceinline__ void route_row(const float* lrow, const float* bias, int* ids, float* wts, int t, int ln) {
    float s[9], c[9];
#pragma unroll
    for (int j = 0; j < 9; j++) {
        const int e = ln + 32 * j;
        s[j] = sigm(lrow[e]);
        c[j] = s[j] + bias[e];
    }
    float sw = 0.f, myw = 0.f;  // lane k keeps the k-th pick (no runtime-indexed local arrays: no stack frame)
    int myi = 0;
#pragma unroll
    for (int k = 0; k < TOPK; k++) {
        float bv = -FLT_MAX;
        int bi = 1 << 30, bj = -1;
#pragma unroll
        for (int j = 0; j < 9; j++)
            if (c[j] > bv) { bv = c[j]; bi = ln + 32 * j; bj = j; }
        float wv = bj >= 0 ? s[bj] : 0.f;
        for (int o = 16; o; o >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
            const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
            const float ow = __shfl_xor_sync(0xffffffffu, wv, o);
            if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; wv = ow; }
        }
        if (ln == k) { myi = bi; myw = wv; }
        sw += wv;
#pragma unroll
        for (int j = 0; j < 9; j++)
            if (ln + 32 * j == bi) c[j] = -FLT_MAX;
    }
    if (ln < TOPK) {
        ids[t * TOPK + ln] = myi;
        const float w = myw / (sw + 1e-20f) * RSF;
        wts[t * TOPK + ln] = w < g_tau_all ? 0.f : w;
    }
}
void route(const float* logits, const float* bias, int* ids, float* wts, int T, cudaStream_t st) {
    k_route<<<(T + 7) / 8, 256, 0, st>>>(logits, bias, ids, wts, T);
}
// decode router (T <= 4 rows): fp32 GEMV, warp per (expert, K chunk of 512), all loads of a lane in one round;
// partials [RKS][T][288], summed in a fixed order by k_route_ks
constexpr int RKS = 8;
__global__ void __launch_bounds__(256) k_rgemv(const float* W, const float* x, int T, float* part) {
    const int e = blockIdx.x * 8 + (threadIdx.x >> 5), ln = threadIdx.x & 31, ks = blockIdx.y;
    const float4* w = reinterpret_cast<const float4*>(W + (size_t)e * H + ks * (H / RKS));
    float4 v[4];
#pragma unroll
    for (int u = 0; u < 4; u++) v[u] = __ldg(w + ln + 32 * u);
    for (int t = 0; t < T; t++) {
        const float4* xv = reinterpret_cast<const float4*>(x + (size_t)t * H + ks * (H / RKS));
        float a = 0.f;
#pragma unroll
        for (int u = 0; u < 4; u++) {
            const float4 b = xv[ln + 32 * u];
            a += v[u].x * b.x + v[u].y * b.y + v[u].z * b.z + v[u].w * b.w;
        }
        for (int o = 16; o; o >>= 1) a += __shfl_xor_sync(0xffffffffu, a, o);
        if (ln == 0) part[((size_t)ks * T + t) * NE + e] = a;
    }
}
__global__ void k_rsum(const float* part, int T, float* lg) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= T * NE) return;
    float s = 0.f;
#pragma unroll
    for (int q = 0; q < RKS; q++) s += part[(size_t)q * T * NE + i];
    lg[i] = s;
}
void router_dec(const float* W, const float* x, int T, float* part, float* lg, cudaStream_t st) {
    k_rgemv<<<dim3(NE / 8, RKS), 256, 0, st>>>(W, x, T, part);
    k_rsum<<<(T * NE + 255) / 256, 256, 0, st>>>(part, T, lg);
}

// gu [T][2 n] (gate | up) -> bf16 silu(min(g, 10)) * clamp(u, -10, 10) [T][n]
__global__ void k_glu(const float* gu, bf16* out, int n, int ldgu) {
    const int t = blockIdx.y, i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float g = fminf(gu[(size_t)t * ldgu + i], CLAMP);
    const float u = fminf(fmaxf(gu[(size_t)t * ldgu + n + i], -CLAMP), CLAMP);
    out[(size_t)t * n + i] = __float2bfloat16(silu(g) * u);
}
void glu(const float* gu, bf16* out, int n, int T, cudaStream_t st, int ldgu) {
    k_glu<<<dim3((n + 255) / 256, T), 256, 0, st>>>(gu, out, n, ldgu < 0 ? 2 * n : ldgu);
}

// dequantize an expert slice to bf16 for the GEMM path. gate/up: qweight [512][ld] i32 (cols c0..c0+n), scales
// [32][ld] f16 -> W [2n][4096] (gate rows then up rows). down: qweight rows [r0/8 ..) of [*][4096], scales rows r0/128..
// -> W [4096][n]
template <class OT>
__global__ void k_deq_gu(const int* gq, const __half* gs, const int* uq, const __half* us, int ld, int c0, int n,
                         OT* W) {
    // block: 32 output rows (cols of q) x 256 k ; thread handles one (row, k-octet)
    const int col = blockIdx.x * 32 + (threadIdx.x & 31);
    const int k8 = blockIdx.y * 8 + (threadIdx.x >> 5);  // octet index 0..511
    const int which = blockIdx.z;
    if (col >= n) return;
    const int* q = which ? uq : gq;
    const __half* s = which ? us : gs;
    const unsigned w = (unsigned)q[(size_t)k8 * ld + c0 + col];
    const float sc = __half2float(s[(size_t)(k8 / 16) * ld + c0 + col]);
    OT* o = W + ((size_t)which * n + col) * H + k8 * 8;
    __align__(16) OT v[8];
#pragma unroll
    for (int j = 0; j < 8; j++) v[j] = (OT)((float)((int)((w >> (4 * j)) & 15) - 8) * sc);
    *reinterpret_cast<uint4*>(o) = *reinterpret_cast<uint4*>(v);
}
template <class OT>
__global__ void k_deq_d(const int* dq, const __half* ds, int r0, int n, OT* W) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;  // output row (hidden) 0..4095
    const int k8 = blockIdx.y;                                // octet of the slice's K (n / 8)
    const unsigned w = (unsigned)dq[(size_t)(r0 / 8 + k8) * H + col];
    const float sc = __half2float(ds[(size_t)((r0 + k8 * 8) / 64) * H + col]);
    __align__(16) OT v[8];
#pragma unroll
    for (int j = 0; j < 8; j++) v[j] = (OT)((float)((int)((w >> (4 * j)) & 15) - 8) * sc);
    *reinterpret_cast<uint4*>(W + (size_t)col * n + k8 * 8) = *reinterpret_cast<uint4*>(v);
}
void deq_expert(const XPtr& x, int n, bf16* Wgu, bf16* Wd, cudaStream_t st) {
    k_deq_gu<bf16><<<dim3((n + 31) / 32, 512 / 8, 2), 256, 0, st>>>(x.gq, x.gs, x.uq, x.us, x.ld, x.c0, n, Wgu);
    k_deq_d<bf16><<<dim3(H / 256, n / 8), 256, 0, st>>>(x.dq, x.ds, x.r0, n, Wd);
}
void deq_expert16(const XPtr& x, int n, __half* Wgu, __half* Wd, cudaStream_t st) {
    k_deq_gu<__half><<<dim3((n + 31) / 32, 512 / 8, 2), 256, 0, st>>>(x.gq, x.gs, x.uq, x.us, x.ld, x.c0, n, Wgu);
    k_deq_d<__half><<<dim3(H / 256, n / 8), 256, 0, st>>>(x.dq, x.ds, x.r0, n, Wd);
}
// FP16 path of the prefill expert GEMMs: gather bf16 rows -> fp16, glu fp16 -> fp16, scatter from fp16
__global__ void k_gather16(const bf16* x, const int* rows, __half* out) {
    const int i = blockIdx.x;
    const bf16* s = x + (size_t)rows[i] * H;
    __half* d = out + (size_t)i * H;
    for (int j = threadIdx.x; j < H; j += blockDim.x) d[j] = __float2half(__bfloat162float(s[j]));
}
void gather_rows16(const bf16* x, const int* rows, __half* out, int n, cudaStream_t st) {
    if (n) k_gather16<<<n, 256, 0, st>>>(x, rows, out);
}
__global__ void k_glu16(const __half* gu, __half* out, int n) {
    const int t = blockIdx.y, i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float g = fminf(__half2float(gu[(size_t)t * 2 * n + i]), CLAMP);
    const float u = fminf(fmaxf(__half2float(gu[(size_t)t * 2 * n + n + i]), -CLAMP), CLAMP);
    out[(size_t)t * n + i] = __float2half(silu(g) * u);
}
void glu16(const __half* gu, __half* out, int n, int T, cudaStream_t st) {
    k_glu16<<<dim3((n + 255) / 256, T), 256, 0, st>>>(gu, out, n);
}
__global__ void k_scatter16(float* out, const __half* y, const int* rows, const float* w) {
    const int i = blockIdx.x;
    float* d = out + (size_t)rows[i] * H;
    const __half* s = y + (size_t)i * H;
    const float ww = w[i];
    for (int j = threadIdx.x; j < H; j += blockDim.x) d[j] += ww * __half2float(s[j]);
}
void scatter_add16(float* out, const __half* y, const int* rows, const float* w, int n, cudaStream_t st) {
    if (n) k_scatter16<<<n, 256, 0, st>>>(out, y, rows, w);
}

// rows gather / weighted scatter-add (sequential per expert -> deterministic)
__global__ void k_gather(const bf16* x, const int* rows, bf16* out) {
    const int i = blockIdx.x;
    const uint4* s = reinterpret_cast<const uint4*>(x + (size_t)rows[i] * H);
    uint4* d = reinterpret_cast<uint4*>(out + (size_t)i * H);
    for (int j = threadIdx.x; j < H / 8; j += blockDim.x) d[j] = s[j];
}
void gather_rows(const bf16* x, const int* rows, bf16* out, int n, cudaStream_t st) {
    if (n) k_gather<<<n, 256, 0, st>>>(x, rows, out);
}
__global__ void k_scatter(float* out, const float* y, const int* rows, const float* w) {
    const int i = blockIdx.x;
    float* d = out + (size_t)rows[i] * H;
    const float* s = y + (size_t)i * H;
    const float ww = w[i];
    for (int j = threadIdx.x; j < H; j += blockDim.x) d[j] += ww * s[j];
}
void scatter_add(float* out, const float* y, const int* rows, const float* w, int n, cudaStream_t st) {
    if (n) k_scatter<<<n, 256, 0, st>>>(out, y, rows, w);
}

// ---------------------------------------------------------------- expert GEMV (decode / verify rows)
// 1) gate/up: grid (n / 64, jobs, KSPL), 256 threads = 16 column-quads (uint4 = 4 columns) x 16 k-lanes
//    -> gu[job][split][r][2n]
// 2) glu: h[job][r][n] = silu(min(g, 10)) * clamp(u) * w (fixed split order)
// 3) down: grid (H / 128, jobs, DKS), 256 threads = 32 column-quads x 8 k-lanes -> yp[job][ks][r][H]
// 4) sum: out[row][c] += sum over jobs / splits in a fixed order
constexpr int DKS = 2;
// 8 nibbles of w (nibble j = bits 4j..4j+3) -> floats (q - 8), exact: each byte is placed under the exponent of 2^23
// (prmt) and 2^23 + 8 is subtracted (full-rate int / fp32 pipes instead of the quarter-rate I2F)
__device__ __forceinline__ void nib8(unsigned w, float* f) {
    const unsigned lo = w & 0x0F0F0F0Fu, hi = (w >> 4) & 0x0F0F0F0Fu;
#pragma unroll
    for (int b = 0; b < 4; b++) {
        f[2 * b] = __int_as_float(__byte_perm(lo, 0x4B000000u, 0x7440u | b)) - 8388616.f;
        f[2 * b + 1] = __int_as_float(__byte_perm(hi, 0x4B000000u, 0x7440u | b)) - 8388616.f;
    }
}
__global__ void __launch_bounds__(256) k_xgu(const XJob* jobs, const int* njp, const float* x, float* gu, int n) {
    __shared__ float red[16][2][64];
    if (njp && (int)blockIdx.y >= *njp) return;
    const XJob& J = jobs[blockIdx.y];
    const int cq = threadIdx.x & 15, kl = threadIdx.x >> 4;
    const int col = blockIdx.x * 64 + cq * 4;
    const int nr = J.nr;
    float ag[JR][4], au[JR][4];
#pragma unroll
    for (int r = 0; r < JR; r++)
#pragma unroll
        for (int c = 0; c < 4; c++) { ag[r][c] = 0.f; au[r][c] = 0.f; }
    auto step = [&](int k8, const unsigned* wgs, const unsigned* wus) {
        const uint2 sgr = *reinterpret_cast<const uint2*>(J.x.gs + (size_t)(k8 >> 4) * J.x.ld + J.x.c0 + col);
        const uint2 sur = *reinterpret_cast<const uint2*>(J.x.us + (size_t)(k8 >> 4) * J.x.ld + J.x.c0 + col);
        const __half* sgh = reinterpret_cast<const __half*>(&sgr);
        const __half* suh = reinterpret_cast<const __half*>(&sur);
        float xx[JR][8];
#pragma unroll
        for (int r = 0; r < JR; r++) {
            if (r >= nr) break;
            const float4* xv = reinterpret_cast<const float4*>(x + (size_t)J.r[r] * H + k8 * 8);
            const float4 a = xv[0], b = xv[1];
            xx[r][0] = a.x; xx[r][1] = a.y; xx[r][2] = a.z; xx[r][3] = a.w;
            xx[r][4] = b.x; xx[r][5] = b.y; xx[r][6] = b.z; xx[r][7] = b.w;
        }
#pragma unroll
        for (int c = 0; c < 4; c++) {  // decode once, use for every row (same per-row arithmetic order)
            float fg[8], fu[8];
            nib8(wgs[c], fg);
            nib8(wus[c], fu);
            const float sg = __half2float(sgh[c]), su = __half2float(suh[c]);
#pragma unroll
            for (int r = 0; r < JR; r++) {
                if (r >= nr) break;
                float dg = 0.f, du = 0.f;
#pragma unroll
                for (int j = 0; j < 8; j++) { dg += fg[j] * xx[r][j]; du += fu[j] * xx[r][j]; }
                ag[r][c] += dg * sg;
                au[r][c] += du * su;
            }
        }
    };
    if (J.x.g3) {  // INT3 slot (round 6, D-3 warm tier): lane kl owns the 32-row block kb = split * 16 + kl
        const int kb = blockIdx.z * (128 / KSPL) + kl;
        const int* gb = J.x.gq + (size_t)kb * 3 * J.x.ld + J.x.c0 + col;
        const int* ub = J.x.uq + (size_t)kb * 3 * J.x.ld + J.x.c0 + col;
        const uint4 gl = __ldg(reinterpret_cast<const uint4*>(gb)), gh = __ldg(reinterpret_cast<const uint4*>(gb + J.x.ld)),
                    gt = __ldg(reinterpret_cast<const uint4*>(gb + 2 * J.x.ld));
        const uint4 ul = __ldg(reinterpret_cast<const uint4*>(ub)), uh = __ldg(reinterpret_cast<const uint4*>(ub + J.x.ld)),
                    ut = __ldg(reinterpret_cast<const uint4*>(ub + 2 * J.x.ld));
#pragma unroll
        for (int t = 0; t < 4; t++) {
            const uint4 g2 = t < 2 ? gl : gh, u2 = t < 2 ? ul : uh;
            const unsigned wgs[4] = {gu3_word(g2.x, gt.x, t), gu3_word(g2.y, gt.y, t), gu3_word(g2.z, gt.z, t), gu3_word(g2.w, gt.w, t)};
            const unsigned wus[4] = {gu3_word(u2.x, ut.x, t), gu3_word(u2.y, ut.y, t), gu3_word(u2.z, ut.z, t), gu3_word(u2.w, ut.w, t)};
            step(4 * kb + t, wgs, wus);
        }
    } else {
    const int k8n = 512 / KSPL, kb = blockIdx.z * k8n;
#pragma unroll
    for (int it = 0; it < 512 / KSPL / 16; it++) {  // fixed trip count: the loads of all iterations in flight
        const int k8 = kb + kl + 16 * it;
        const uint4 wg = __ldg(reinterpret_cast<const uint4*>(J.x.gq + (size_t)k8 * J.x.ld + J.x.c0 + col));
        const uint4 wu = __ldg(reinterpret_cast<const uint4*>(J.x.uq + (size_t)k8 * J.x.ld + J.x.c0 + col));
        const unsigned wgs[4] = {wg.x, wg.y, wg.z, wg.w}, wus[4] = {wu.x, wu.y, wu.z, wu.w};
        step(k8, wgs, wus);
    }
    }
#pragma unroll
    for (int r = 0; r < JR; r++) {
        if (r >= nr) break;
        __syncthreads();
#pragma unroll
        for (int c = 0; c < 4; c++) { red[kl][0][cq * 4 + c] = ag[r][c]; red[kl][1][cq * 4 + c] = au[r][c]; }
        __syncthreads();
        if (threadIdx.x < 128) {
            const int w = threadIdx.x >> 6, c = threadIdx.x & 63;
            float s = 0.f;
            for (int q = 0; q < 16; q++) s += red[q][w][c];
            gu[(((size_t)blockIdx.y * KSPL + blockIdx.z) * JR + r) * 2 * n + w * n + blockIdx.x * 64 + c] = s;
        }
    }
}
__global__ void k_xglu(const XJob* jobs, const int* njp, const float* gu, float* hb, int n) {
    const int jb = blockIdx.y;
    if (jb >= *njp) return;
    const XJob& J = jobs[jb];
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < J.nr * n; i += gridDim.x * blockDim.x) {
        const int r = i / n, c = i % n;
        float g = 0.f, u = 0.f;
        for (int s = 0; s < KSPL; s++) {
            const float* o = gu + (((size_t)jb * KSPL + s) * JR + r) * 2 * n;
            g += o[c];
            u += o[n + c];
        }
        g = fminf(g, CLAMP);
        u = fminf(fmaxf(u, -CLAMP), CLAMP);
        hb[((size_t)jb * JR + r) * n + c] = silu(g) * u * J.w[r];
    }
}
__global__ void __launch_bounds__(256) k_xdown(const XJob* jobs, const int* njp, const float* hb, float* yp, int n) {
    __shared__ float red[8][128];
    if ((int)blockIdx.y >= *njp) return;
    const XJob& J = jobs[blockIdx.y];
    const int nr = J.nr;
    const int cq = threadIdx.x & 31, kl = threadIdx.x >> 5;
    const int col = blockIdx.x * 128 + cq * 4;
    const int k8n = n / 8 / DKS, kb = blockIdx.z * k8n;
    const float* h = hb + (size_t)blockIdx.y * JR * n;
    float a[JR][4];
#pragma unroll
    for (int r = 0; r < JR; r++)
#pragma unroll
        for (int c = 0; c < 4; c++) a[r][c] = 0.f;
    for (int k8 = kb + kl; k8 < kb + k8n; k8 += 8) {
        const uint4 w = __ldg(reinterpret_cast<const uint4*>(J.x.dq + (size_t)(J.x.r0 / 8 + k8) * H + col));
        const uint2 sr = *reinterpret_cast<const uint2*>(J.x.ds + (size_t)((J.x.r0 + k8 * 8) >> 6) * H + col);
        const __half* sh = reinterpret_cast<const __half*>(&sr);
        const unsigned ws[4] = {w.x, w.y, w.z, w.w};
        float hh[JR][8];
#pragma unroll
        for (int r = 0; r < JR; r++) {
            if (r >= nr) break;
            const float4* hv = reinterpret_cast<const float4*>(h + (size_t)r * n + k8 * 8);
            const float4 p = hv[0], q = hv[1];
            hh[r][0] = p.x; hh[r][1] = p.y; hh[r][2] = p.z; hh[r][3] = p.w;
            hh[r][4] = q.x; hh[r][5] = q.y; hh[r][6] = q.z; hh[r][7] = q.w;
        }
#pragma unroll
        for (int c = 0; c < 4; c++) {
            float f[8];
            nib8(ws[c], f);
            const float sc = __half2float(sh[c]);
#pragma unroll
            for (int r = 0; r < JR; r++) {
                if (r >= nr) break;
                float d = 0.f;
#pragma unroll
                for (int j = 0; j < 8; j++) d += f[j] * hh[r][j];
                a[r][c] += d * sc;
            }
        }
    }
#pragma unroll
    for (int r = 0; r < JR; r++) {
        if (r >= nr) break;
        __syncthreads();
#pragma unroll
        for (int c = 0; c < 4; c++) red[kl][cq * 4 + c] = a[r][c];
        __syncthreads();
        if (threadIdx.x < 128) {
            float s = 0.f;
            for (int q = 0; q < 8; q++) s += red[q][threadIdx.x];
            yp[(((size_t)blockIdx.y * DKS + blockIdx.z) * JR + r) * H + blockIdx.x * 128 + threadIdx.x] = s;
        }
    }
}
// grid (H / 128, m): each thread one (row, column); the (job, slot) pairs of the row in job order, sums in the same
// order as before (job ascending, split inner), independent loads
__global__ void k_xsum(const XJob* jobs, const int* njp, const float* yp, float* out, int m) {
    __shared__ int pj[MXE * 2];
    __shared__ int np;
    const int r = blockIdx.y, c = blockIdx.x * 128 + threadIdx.x;
    if (threadIdx.x < 32) {  // lane j = job j (at most one slot of row r per job), ordered compaction by ballot
        const int nj = *njp, j = threadIdx.x;
        int slot = -1;
        if (j < nj) {
            const XJob& J = jobs[j];
            for (int i = 0; i < J.nr; i++)
                if (J.r[i] == r) slot = i;
        }
        const unsigned b = __ballot_sync(0xffffffffu, slot >= 0);
        if (slot >= 0) pj[__popc(b & ((1u << j) - 1u))] = (j * DKS) * JR + slot;
        if (j == 0) np = __popc(b);
    }
    __syncthreads();
    const int n = np;
    float acc = 0.f;
    int q = 0;
    for (; q + 4 <= n; q += 4) {
        float v[4][DKS];
#pragma unroll
        for (int u = 0; u < 4; u++)
#pragma unroll
            for (int k = 0; k < DKS; k++) v[u][k] = yp[(size_t)(pj[q + u] + k * JR) * H + c];
#pragma unroll
        for (int u = 0; u < 4; u++) {
            float s = 0.f;
#pragma unroll
            for (int k = 0; k < DKS; k++) s += v[u][k];
            acc += s;
        }
    }
    for (; q < n; q++) {
        float s = 0.f;
#pragma unroll
        for (int k = 0; k < DKS; k++) s += yp[(size_t)(pj[q] + k * JR) * H + c];
        acc += s;
    }
    out[(size_t)r * H + c] += acc;
}
// round 22 ([glm-r22-xgemv], bit-identical to k_xglu / k_xsum, ~-2 us per call at 20 jobs): glu with one element per
// thread (all split loads in flight, 9-11 blocks per job instead of 4); job sum with 8 jobs' loads in flight
__global__ void k_xglu2(const XJob* jobs, const int* njp, const float* gu, float* hb, int n) {
    const int jb = blockIdx.y;
    if (jb >= *njp) return;
    const XJob& J = jobs[jb];
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= J.nr * n) return;
    const int r = i / n, c = i % n;
    float gv[KSPL], uv[KSPL];
#pragma unroll
    for (int s = 0; s < KSPL; s++) {
        const float* o = gu + (((size_t)jb * KSPL + s) * JR + r) * 2 * n;
        gv[s] = o[c];
        uv[s] = o[n + c];
    }
    float g = 0.f, u = 0.f;
#pragma unroll
    for (int s = 0; s < KSPL; s++) { g += gv[s]; u += uv[s]; }
    g = fminf(g, CLAMP);
    u = fminf(fmaxf(u, -CLAMP), CLAMP);
    hb[((size_t)jb * JR + r) * n + c] = silu(g) * u * J.w[r];
}
template <int U, int NSP = DKS>
__global__ void k_xsum2(const XJob* jobs, const int* njp, const float* yp, float* out, int m) {
    __shared__ int pj[MXE * 2];
    __shared__ int np;
    const int r = blockIdx.y, c = blockIdx.x * 128 + threadIdx.x;
    if (threadIdx.x < 32) {
        const int nj = *njp, j = threadIdx.x;
        int slot = -1;
        if (j < nj) {
            const XJob& J = jobs[j];
            for (int i = 0; i < J.nr; i++)
                if (J.r[i] == r) slot = i;
        }
        const unsigned b = __ballot_sync(0xffffffffu, slot >= 0);
        if (slot >= 0) pj[__popc(b & ((1u << j) - 1u))] = (j * NSP) * JR + slot;
        if (j == 0) np = __popc(b);
    }
    __syncthreads();
    const int n = np;
    float acc = 0.f;
    int q = 0;
    for (; q + U <= n; q += U) {
        float v[U][NSP];
#pragma unroll
        for (int u = 0; u < U; u++)
#pragma unroll
            for (int k = 0; k < NSP; k++) v[u][k] = yp[(size_t)(pj[q + u] + k * JR) * H + c];
#pragma unroll
        for (int u = 0; u < U; u++) {
            float s = 0.f;
#pragma unroll
            for (int k = 0; k < NSP; k++) s += v[u][k];
            acc += s;
        }
    }
    for (; q < n; q++) {
        float s = 0.f;
#pragma unroll
        for (int k = 0; k < NSP; k++) s += yp[(size_t)(pj[q] + k * JR) * H + c];
        acc += s;
    }
    out[(size_t)r * H + c] += acc;
}
__global__ void k_setn(int* p, int n) { *p = n; }
void expert_gemv(const XJob* jobs, int njobs, const float* x, float* gu, float* out, int n, int m, cudaStream_t st,
                 const int* njp, float* hb, float* yp, int* njtmp) {
    if (!njobs) return;
    if (!njp) {
        k_setn<<<1, 1, 0, st>>>(njtmp, njobs);
        njp = njtmp;
    }
    k_xgu<<<dim3(n / 64, njobs, KSPL), 256, 0, st>>>(jobs, njp, x, gu, n);
    static const bool old = getenv("GLM_XG_OLD") && atoi(getenv("GLM_XG_OLD"));  // off-switch: round-21 glu / sum
    if (old) k_xglu<<<dim3(4, njobs), 256, 0, st>>>(jobs, njp, gu, hb, n);
    else k_xglu2<<<dim3((JR * n + 255) / 256, njobs), 256, 0, st>>>(jobs, njp, gu, hb, n);
    k_xdown<<<dim3(H / 128, njobs, DKS), 256, 0, st>>>(jobs, njp, hb, yp, n);
    if (old) k_xsum<<<dim3(H / 128, m), 128, 0, st>>>(jobs, njp, yp, out, m);
    else k_xsum2<8><<<dim3(H / 128, m), 128, 0, st>>>(jobs, njp, yp, out, m);
}

// ---------------------------------------------------------------- INT8 weights

// symmetric INT8 with a scale per 128-group of a row: s[n][K / 128] (one block per row, warp per group)
__global__ void k_q8(const bf16* w, int K, int8_t* q, float* s) {
    const int n = blockIdx.x, ln = threadIdx.x & 31;
    const bf16* r = w + (size_t)n * K;
    for (int g = threadIdx.x >> 5; g < K / 128; g += blockDim.x >> 5) {
        float v[4], m = 0.f;
        for (int j = 0; j < 4; j++) { v[j] = __bfloat162float(r[g * 128 + ln * 4 + j]); m = fmaxf(m, fabsf(v[j])); }
        for (int o = 16; o; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
        const float sc = m > 0.f ? m / 127.f : 1.f, inv = 1.f / sc;
        for (int j = 0; j < 4; j++) q[(size_t)n * K + g * 128 + ln * 4 + j] = (int8_t)__float2int_rn(v[j] * inv);
        if (ln == 0) s[(size_t)n * (K / 128) + g] = sc;
    }
}
void quant8(const bf16* w, int N, int K, int8_t* q, float* s, cudaStream_t st) { k_q8<<<N, 256, 0, st>>>(w, K, q, s); }
// INT4 values (|q| <= 7) in the INT8 container, scale per 128-group chosen from 8 clip ratios by squared error
// (fake quantization: the INT8 kernels run INT4 numerics; quality gate before an INT4 storage format)
__global__ void k_q4f(const bf16* w, int K, int8_t* q, float* s) {
    const int n = blockIdx.x, ln = threadIdx.x & 31;
    const bf16* r = w + (size_t)n * K;
    for (int g = threadIdx.x >> 5; g < K / 128; g += blockDim.x >> 5) {
        float v[4], m = 0.f;
        for (int j = 0; j < 4; j++) { v[j] = __bfloat162float(r[g * 128 + ln * 4 + j]); m = fmaxf(m, fabsf(v[j])); }
        for (int o = 16; o; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
        float best = 1e30f, bs = m > 0.f ? m / 7.f : 1.f;
        for (int c = 0; c < 8; c++) {
            const float sc = m > 0.f ? m * (1.f - 0.04f * c) / 7.f : 1.f;
            float e = 0.f;
            for (int j = 0; j < 4; j++) {
                const float qq = fminf(fmaxf(rintf(v[j] / sc), -7.f), 7.f);
                e += (v[j] - qq * sc) * (v[j] - qq * sc);
            }
            for (int o = 16; o; o >>= 1) e += __shfl_xor_sync(0xffffffffu, e, o);
            if (e < best) { best = e; bs = sc; }
        }
        for (int j = 0; j < 4; j++)
            q[(size_t)n * K + g * 128 + ln * 4 + j] = (int8_t)fminf(fmaxf(rintf(v[j] / bs), -7.f), 7.f);
        if (ln == 0) s[(size_t)n * (K / 128) + g] = bs;
    }
}
// INT4 g32 (scale search per 32) dequantized, then held as INT8 g128 (adds the INT8 rounding: a pessimistic gate)
__global__ void k_q4f32(const bf16* w, int K, int8_t* q, float* s) {
    const int n = blockIdx.x, ln = threadIdx.x & 31;
    const bf16* r = w + (size_t)n * K;
    for (int g = threadIdx.x >> 5; g < K / 128; g += blockDim.x >> 5) {
        float v[4], m = 0.f;
        for (int j = 0; j < 4; j++) { v[j] = __bfloat162float(r[g * 128 + ln * 4 + j]); m = fmaxf(m, fabsf(v[j])); }
        for (int o = 4; o; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));  // 8 lanes = 32 values
        float best = 1e30f, bs = m > 0.f ? m / 7.f : 1.f;
        for (int c = 0; c < 8; c++) {
            const float sc = m > 0.f ? m * (1.f - 0.04f * c) / 7.f : 1.f;
            float e = 0.f;
            for (int j = 0; j < 4; j++) {
                const float qq = fminf(fmaxf(rintf(v[j] / sc), -7.f), 7.f);
                e += (v[j] - qq * sc) * (v[j] - qq * sc);
            }
            for (int o = 4; o; o >>= 1) e += __shfl_xor_sync(0xffffffffu, e, o);
            if (e < best) { best = e; bs = sc; }
        }
        float d[4], m8 = 0.f;
        for (int j = 0; j < 4; j++) { d[j] = fminf(fmaxf(rintf(v[j] / bs), -7.f), 7.f) * bs; m8 = fmaxf(m8, fabsf(d[j])); }
        for (int o = 16; o; o >>= 1) m8 = fmaxf(m8, __shfl_xor_sync(0xffffffffu, m8, o));
        const float s8 = m8 > 0.f ? m8 / 127.f : 1.f;
        for (int j = 0; j < 4; j++) q[(size_t)n * K + g * 128 + ln * 4 + j] = (int8_t)__float2int_rn(d[j] / s8);
        if (ln == 0) s[(size_t)n * (K / 128) + g] = s8;
    }
}
// round 11: INT8 g128 weights -> RTN INT4 nibbles (two's complement, low = even k) + fp32 scale per 128 (draft-only copies)
__global__ void k_8to4(const int8_t* q8, const float* s8, int K, uint8_t* q4, float* s4) {
    const int n = blockIdx.x, G = K / 128;
    for (int g = threadIdx.x >> 5; g < G; g += blockDim.x >> 5) {
        const int ln = threadIdx.x & 31;
        const float sc = s8[(size_t)n * G + g];
        float v[4], m = 0.f;
        for (int j = 0; j < 4; j++) { v[j] = q8[(size_t)n * K + g * 128 + ln * 4 + j] * sc; m = fmaxf(m, fabsf(v[j])); }
        for (int o = 16; o; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
        const float s = m > 0.f ? m / 7.f : 1.f;
        int c[4];
        for (int j = 0; j < 4; j++) c[j] = max(-8, min(7, __float2int_rn(v[j] / s)));
        q4[((size_t)n * K + g * 128 + ln * 4) / 2] = (uint8_t)((c[0] & 15) | ((c[1] & 15) << 4));
        q4[((size_t)n * K + g * 128 + ln * 4) / 2 + 1] = (uint8_t)((c[2] & 15) | ((c[3] & 15) << 4));
        if (ln == 0) s4[(size_t)n * G + g] = s;
    }
}
void int8_to4(const int8_t* q8, const float* s8, int N, int K, uint8_t* q4, float* s4, cudaStream_t st) {
    k_8to4<<<N, 256, 0, st>>>(q8, s8, K, q4, s4);
}
void quant4f(const bf16* w, int N, int K, int8_t* q, float* s, cudaStream_t st) {
    static const int g = [] { const char* e = getenv("GLM_KDA_W4F"); return e ? atoi(e) : 1; }();
    if (g == 32) k_q4f32<<<N, 256, 0, st>>>(w, K, q, s);
    else k_q4f<<<N, 256, 0, st>>>(w, K, q, s);
}

__global__ void k_dq8(const int8_t* q, const float* s, int K, bf16* w) {
    const int n = blockIdx.x;
    for (int i = threadIdx.x; i < K; i += blockDim.x)
        w[(size_t)n * K + i] = __float2bfloat16((float)q[(size_t)n * K + i] * s[(size_t)n * (K / 128) + i / 128]);
}
void dequant8(const int8_t* q, const float* s, int N, int K, bf16* w, cudaStream_t st) { k_dq8<<<N, 256, 0, st>>>(q, s, K, w); }
__global__ void k_dq8h(const int8_t* q, const float* s, int K, __half* w) {
    const int n = blockIdx.x;
    for (int i = threadIdx.x; i < K; i += blockDim.x)
        w[(size_t)n * K + i] = __float2half((float)q[(size_t)n * K + i] * s[(size_t)n * (K / 128) + i / 128]);
}
void dequant8h(const int8_t* q, const float* s, int N, int K, __half* w, cudaStream_t st) { k_dq8h<<<N, 256, 0, st>>>(q, s, K, w); }
__global__ void k_b2h_rows(const bf16* x, int ldx, int K, __half* y) {
    const int t = blockIdx.x;
    for (int i = threadIdx.x; i < K; i += blockDim.x) y[(size_t)t * K + i] = __float2half(__bfloat162float(x[(size_t)t * ldx + i]));
}
void b2h_rows(const bf16* x, int ldx, int K, int T, __half* y, cudaStream_t st) { k_b2h_rows<<<T, 256, 0, st>>>(x, ldx, K, y); }
// Y[t][i] (+)= y[t][i] (fp16 tile [T][n] -> fp32 rows of stride ldy)
__global__ void k_acc16(float* Y, int ldy, const __half* y, int n, bool acc) {
    const int t = blockIdx.y;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float v = __half2float(y[(size_t)t * n + i]);
    float* o = Y + (size_t)t * ldy + i;
    *o = acc ? *o + v : v;
}
void acc16(float* Y, int ldy, const __half* y, int n, int T, bool acc, cudaStream_t st) {
    k_acc16<<<dim3((n + 255) / 256, T), 256, 0, st>>>(Y, ldy, y, n, acc);
}

__device__ __forceinline__ void ld8x(const float* p, float* v) {
    const float4 a = *reinterpret_cast<const float4*>(p), b = *reinterpret_cast<const float4*>(p + 4);
    v[0] = a.x; v[1] = a.y; v[2] = a.z; v[3] = a.w; v[4] = b.x; v[5] = b.y; v[6] = b.z; v[7] = b.w;
}
__device__ __forceinline__ void ld8x(const bf16* p, float* v) {
    const uint4 u = *reinterpret_cast<const uint4*>(p);
    const bf16* b = reinterpret_cast<const bf16*>(&u);
#pragma unroll
    for (int j = 0; j < 8; j++) v[j] = __bfloat162float(b[j]);
}
// y[t][n] (+)= sum_k q[n][k] s[n][k / 128] x[t][k] ; warp per row, lane reads 16 int8 per step; K % 512 == 0
template <class XT>
__global__ void __launch_bounds__(256) k_gemv8(const int8_t* q, const float* s, int N, int K, const XT* x, int ldx,
                                               float* y, int ldy, int T, bool acc) {
    const int n = blockIdx.x * 8 + (threadIdx.x >> 5), ln = threadIdx.x & 31;
    if (n >= N) return;
    const int8_t* r = q + (size_t)n * K;
    float a[MAXR];
#pragma unroll
    for (int t = 0; t < MAXR; t++) a[t] = 0.f;
    constexpr int U = 4;  // 4 independent 16-byte loads per lane in flight
    for (int kb = ln * 16; kb < K; kb += 512 * U) {
        int4 w4[U];
        float gs[U];
#pragma unroll
        for (int u = 0; u < U; u++) {
            const int k = kb + u * 512;
            if (k < K) {
                w4[u] = __ldg(reinterpret_cast<const int4*>(r + k));
                gs[u] = __ldg(s + (size_t)n * (K / 128) + k / 128);
            }
        }
#pragma unroll
        for (int u = 0; u < U; u++) {
            const int k = kb + u * 512;
            if (k >= K) break;
            const int8_t* wb = reinterpret_cast<const int8_t*>(&w4[u]);
            float w[16];
#pragma unroll
            for (int j = 0; j < 16; j++) w[j] = (float)wb[j] * gs[u];
#pragma unroll
            for (int t = 0; t < MAXR; t++) {
                if (t >= T) break;
                float xv[16];
                ld8x(x + (size_t)t * ldx + k, xv);
                ld8x(x + (size_t)t * ldx + k + 8, xv + 8);
                float d = 0.f;
#pragma unroll
                for (int j = 0; j < 16; j++) d += w[j] * xv[j];
                a[t] += d;
            }
        }
    }
#pragma unroll
    for (int t = 0; t < MAXR; t++) {
        if (t >= T) break;
        float v = a[t];
        for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
        if (ln == 0) {
            float* yo = y + (size_t)t * ldy + n;
            *yo = acc ? *yo + v : v;
        }
    }
}
void gemv8(const int8_t* q, const float* s, int N, int K, const float* xf, const bf16* xb, int ldx, float* y, int ldy,
           int T, bool acc, cudaStream_t st) {
    if (xf) k_gemv8<float><<<(N + 7) / 8, 256, 0, st>>>(q, s, N, K, xf, ldx, y, ldy, T, acc);
    else k_gemv8<bf16><<<(N + 7) / 8, 256, 0, st>>>(q, s, N, K, xb, ldx, y, ldy, T, acc);
}

// ---------------------------------------------------------------- W8A8 (two-pass int8 activations) GEMV for decode rows

// x [T][ldx] (fp32 or bf16) -> per 128-group: q1 = round(x / s), q2 = round((x - q1 s) / (s / 254)); xq [2][T][K], xs [2][T][K/128]
template <class XT>
__global__ void k_qx(const XT* x, int ldx, int K, int T, int8_t* xq, float* xs) {
    const int g = blockIdx.x, t = blockIdx.y, ln = threadIdx.x;  // one warp per (group, row), 4 values per lane
    float v[4], m = 0.f;
    for (int j = 0; j < 4; j++) {
        const XT a = x[(size_t)t * ldx + g * 128 + ln * 4 + j];
        v[j] = (float)a;
        m = fmaxf(m, fabsf(v[j]));
    }
    for (int o = 16; o; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
    const float s = m > 0.f ? m / 127.f : 1.f, s2 = s / 254.f;
    for (int j = 0; j < 4; j++) {
        const int q1 = max(-127, min(127, __float2int_rn(v[j] / s)));
        const int q2 = max(-127, min(127, __float2int_rn((v[j] - q1 * s) / s2)));
        xq[(size_t)t * K + g * 128 + ln * 4 + j] = (int8_t)q1;
        xq[(size_t)(T + t) * K + g * 128 + ln * 4 + j] = (int8_t)q2;
    }
    if (ln == 0) { xs[(size_t)t * (K / 128) + g] = s; xs[(size_t)(T + t) * (K / 128) + g] = s2; }
}
template <class XT>
void quant_x(const XT* x, int ldx, int K, int T, int8_t* xq, float* xs, cudaStream_t st) {
    k_qx<XT><<<dim3(K / 128, T), 32, 0, st>>>(x, ldx, K, T, xq, xs);
}
template void quant_x<float>(const float*, int, int, int, int8_t*, float*, cudaStream_t);
template void quant_x<bf16>(const bf16*, int, int, int, int8_t*, float*, cudaStream_t);
__device__ __forceinline__ float bf2f_(bf16 v) { return __bfloat162float(v); }

// y[t][n] (+)= sum_k W[n][k] x[t][k]; W int8 * s[n][k/128]; warp per row, lane = 16 k per step, dp4a
__global__ void __launch_bounds__(256) k_gemv8q(const int8_t* q, const float* s, int N, int K, const int8_t* xq,
                                                const float* xs, float* y, int ldy, int T, bool acc) {
    const int n = blockIdx.x * 8 + (threadIdx.x >> 5), ln = threadIdx.x & 31;
    if (n >= N) return;
    const int8_t* r = q + (size_t)n * K;
    const int G = K / 128;
    float a[MAXR];
#pragma unroll
    for (int t = 0; t < MAXR; t++) a[t] = 0.f;
    constexpr int U = 4;
    for (int kb = ln * 16; kb < K; kb += 512 * U) {
        int4 w4[U];
        float ws[U];
#pragma unroll
        for (int u = 0; u < U; u++) {
            const int k = kb + u * 512;
            if (k < K) { w4[u] = __ldg(reinterpret_cast<const int4*>(r + k)); ws[u] = __ldg(s + (size_t)n * G + k / 128); }
        }
#pragma unroll
        for (int u = 0; u < U; u++) {
            const int k = kb + u * 512;
            if (k >= K) break;
            const int g = k / 128;
#pragma unroll
            for (int t = 0; t < MAXR; t++) {
                if (t >= T) break;
                const int4 x1 = *reinterpret_cast<const int4*>(xq + (size_t)t * K + k);
                const int4 x2 = *reinterpret_cast<const int4*>(xq + (size_t)(T + t) * K + k);
                int d1 = __dp4a(w4[u].x, x1.x, 0);
                d1 = __dp4a(w4[u].y, x1.y, d1); d1 = __dp4a(w4[u].z, x1.z, d1); d1 = __dp4a(w4[u].w, x1.w, d1);
                int d2 = __dp4a(w4[u].x, x2.x, 0);
                d2 = __dp4a(w4[u].y, x2.y, d2); d2 = __dp4a(w4[u].z, x2.z, d2); d2 = __dp4a(w4[u].w, x2.w, d2);
                a[t] += ((float)d1 * xs[(size_t)t * G + g] + (float)d2 * xs[(size_t)(T + t) * G + g]) * ws[u];
            }
        }
    }
#pragma unroll
    for (int t = 0; t < MAXR; t++) {
        if (t >= T) break;
        float v = a[t];
        for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
        if (ln == 0) {
            float* yo = y + (size_t)t * ldy + n;
            *yo = acc ? *yo + v : v;
        }
    }
}
// decode row counts (T <= 4) with T known at compile time: same arithmetic and order as k_gemv8q, fewer registers and
// no runtime row loop (the generic kernel's 16-row accumulators cost occupancy)
template <int T, int U = 8>  // round 13: U = k steps loaded ahead (U = 4 tested: slower at T = 3, INT8 rows need one round)
__global__ void __launch_bounds__(256) k_gemv8qt(const int8_t* q, const float* s, int N, int K, const int8_t* xq,
                                                 const float* xs, float* y, int ldy, bool acc, const int* dp = nullptr,
                                                 int nskip = 0) {
    const int n = blockIdx.x * 8 + (threadIdx.x >> 5), ln = threadIdx.x & 31;
    if (n >= N) return;
    // round 27 (D4 b): rows from nskip on (the MLA indexer's wq_b) are not needed while every row of the step attends
    // densely (position + 1 <= TOPB pools); whole blocks exit (nskip is a multiple of 8)
    if (dp && n >= nskip && (*dp + T) / POOL <= TOPB) return;
    const int8_t* r = q + (size_t)n * K;
    const int G = K / 128;
    float a[T];
#pragma unroll
    for (int t = 0; t < T; t++) a[t] = 0.f;
    for (int kb = ln * 16; kb < K; kb += 512 * U) {
        int4 w4[U];
        float ws[U];
#pragma unroll
        for (int u = 0; u < U; u++) {
            const int k = kb + u * 512;
            if (k < K) { w4[u] = __ldg(reinterpret_cast<const int4*>(r + k)); ws[u] = __ldg(s + (size_t)n * G + k / 128); }
        }
#pragma unroll
        for (int u = 0; u < U; u++) {
            const int k = kb + u * 512;
            if (k >= K) break;
            const int g = k / 128;
#pragma unroll
            for (int t = 0; t < T; t++) {
                const int4 x1 = *reinterpret_cast<const int4*>(xq + (size_t)t * K + k);
                const int4 x2 = *reinterpret_cast<const int4*>(xq + (size_t)(T + t) * K + k);
                int d1 = __dp4a(w4[u].x, x1.x, 0);
                d1 = __dp4a(w4[u].y, x1.y, d1); d1 = __dp4a(w4[u].z, x1.z, d1); d1 = __dp4a(w4[u].w, x1.w, d1);
                int d2 = __dp4a(w4[u].x, x2.x, 0);
                d2 = __dp4a(w4[u].y, x2.y, d2); d2 = __dp4a(w4[u].z, x2.z, d2); d2 = __dp4a(w4[u].w, x2.w, d2);
                a[t] += ((float)d1 * xs[(size_t)t * G + g] + (float)d2 * xs[(size_t)(T + t) * G + g]) * ws[u];
            }
        }
    }
#pragma unroll
    for (int t = 0; t < T; t++) {
        float v = a[t];
        for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
        if (ln == 0) {
            float* yo = y + (size_t)t * ldy + n;
            *yo = acc ? *yo + v : v;
        }
    }
}
// round 9: INT4 weights (two's-complement nibbles, low nibble = even k, fp32 scale per 128 k) x two-pass int8 rows
__device__ __forceinline__ void unpack4(unsigned w, int& a, int& b) {
    const unsigned lo = w & 0x0F0F0F0Fu, hi = (w >> 4) & 0x0F0F0F0Fu;
    a = (int)__vsub4(__byte_perm(lo, hi, 0x5140) ^ 0x08080808u, 0x08080808u);
    b = (int)__vsub4(__byte_perm(lo, hi, 0x7362) ^ 0x08080808u, 0x08080808u);
}
// round 13: U = k steps loaded ahead per lane. It changes only how far the loads run ahead, not the k order, so any U is
// bit-identical. The body is shared by k_gemv4q (plain __launch_bounds__(256): the pre-round-13 register allocation)
// and k_gemv4q_mb (min blocks MB; review r13 M1 / L1: an explicit min-blocks value changes ptxas' register choice).
template <int TM, bool EX, int U>
__device__ __forceinline__ void gemv4q_body(const uint8_t* q, const float* s, int N, int K, const int8_t* xq,
                                            const float* xs, float* y, int ldy, int T, bool acc) {
    const int n = blockIdx.x * 8 + (threadIdx.x >> 5), ln = threadIdx.x & 31;
    if (n >= N) return;
    const uint8_t* r = q + (size_t)n * (K / 2);
    const int G = K / 128;
    if (EX) T = TM;
    float a[TM];
#pragma unroll
    for (int t = 0; t < TM; t++) a[t] = 0.f;
    for (int kb = ln * 16; kb < K; kb += 512 * U) {
        uint2 w2[U];
        float ws[U];
#pragma unroll
        for (int u = 0; u < U; u++) {
            const int k = kb + u * 512;
            if (k < K) { w2[u] = __ldg(reinterpret_cast<const uint2*>(r + k / 2)); ws[u] = __ldg(s + (size_t)n * G + k / 128); }
        }
#pragma unroll
        for (int u = 0; u < U; u++) {
            const int k = kb + u * 512;
            if (k >= K) break;
            const int g = k / 128;
            int w0, w1, w2_, w3;
            unpack4(w2[u].x, w0, w1);
            unpack4(w2[u].y, w2_, w3);
#pragma unroll
            for (int t = 0; t < TM; t++) {
                if (t >= T) break;
                const int4 x1 = *reinterpret_cast<const int4*>(xq + (size_t)t * K + k);
                const int4 x2 = *reinterpret_cast<const int4*>(xq + (size_t)(T + t) * K + k);
                int d1 = __dp4a(w0, x1.x, 0);
                d1 = __dp4a(w1, x1.y, d1); d1 = __dp4a(w2_, x1.z, d1); d1 = __dp4a(w3, x1.w, d1);
                int d2 = __dp4a(w0, x2.x, 0);
                d2 = __dp4a(w1, x2.y, d2); d2 = __dp4a(w2_, x2.z, d2); d2 = __dp4a(w3, x2.w, d2);
                a[t] += ((float)d1 * xs[(size_t)t * G + g] + (float)d2 * xs[(size_t)(T + t) * G + g]) * ws[u];
            }
        }
    }
#pragma unroll
    for (int t = 0; t < TM; t++) {
        if (t >= T) break;
        float v = a[t];
        for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
        if (ln == 0) {
            float* yo = y + (size_t)t * ldy + n;
            *yo = acc ? *yo + v : v;
        }
    }
}
template <int TM, bool EX, int U = 8>
__global__ void __launch_bounds__(256) k_gemv4q(const uint8_t* q, const float* s, int N, int K, const int8_t* xq,
                                                const float* xs, float* y, int ldy, int T, bool acc) {
    gemv4q_body<TM, EX, U>(q, s, N, K, xq, xs, y, ldy, T, acc);
}
template <int TM, int U, int MB>
__global__ void __launch_bounds__(256, MB) k_gemv4q_mb(const uint8_t* q, const float* s, int N, int K, const int8_t* xq,
                                                       const float* xs, float* y, int ldy, int T, bool acc) {
    gemv4q_body<TM, true, U>(q, s, N, K, xq, xs, y, ldy, T, acc);
}
void gemv4q(const uint8_t* q, const float* s, int N, int K, const int8_t* xq, const float* xs, float* y, int ldy, int T,
            bool acc, cudaStream_t st) {
    const int nb = (N + 7) / 8;
    // round 13: U = 4 by default, capped at 128 registers (MB = 2). Bit-identical to U = 8. Verify T=3 (glm_kbench g4, vs
    // the old kernel): KDA in-proj 34.8 -> 31.9 us, o-proj 16.7 -> 14.9 us; at T = 2..4 both fit 2 blocks / SM, so the
    // gain is the shorter load-ahead chain, not occupancy (review r13 M1). GLM_G4_U8=1 = the old kernel (plain bounds).
    // U = 4 capped at 85 registers (MB = 3, 3 blocks / SM) was tried: slower (in-proj 32.6, o-proj 15.7 us; T=4 spills).
    static const bool u8 = getenv("GLM_G4_U8") && atoi(getenv("GLM_G4_U8")) != 0;
    if (!u8) {
        switch (T) {
            case 1: k_gemv4q_mb<1, 4, 2><<<nb, 256, 0, st>>>(q, s, N, K, xq, xs, y, ldy, T, acc); return;
            case 2: k_gemv4q_mb<2, 4, 2><<<nb, 256, 0, st>>>(q, s, N, K, xq, xs, y, ldy, T, acc); return;
            case 3: k_gemv4q_mb<3, 4, 2><<<nb, 256, 0, st>>>(q, s, N, K, xq, xs, y, ldy, T, acc); return;
            case 4: k_gemv4q_mb<4, 4, 2><<<nb, 256, 0, st>>>(q, s, N, K, xq, xs, y, ldy, T, acc); return;
            default: break;
        }
    }
    switch (T) {
        case 1: k_gemv4q<1, true><<<nb, 256, 0, st>>>(q, s, N, K, xq, xs, y, ldy, T, acc); break;
        case 2: k_gemv4q<2, true><<<nb, 256, 0, st>>>(q, s, N, K, xq, xs, y, ldy, T, acc); break;
        case 3: k_gemv4q<3, true><<<nb, 256, 0, st>>>(q, s, N, K, xq, xs, y, ldy, T, acc); break;
        case 4: k_gemv4q<4, true><<<nb, 256, 0, st>>>(q, s, N, K, xq, xs, y, ldy, T, acc); break;
        default: k_gemv4q<MAXR, false><<<nb, 256, 0, st>>>(q, s, N, K, xq, xs, y, ldy, T, acc);
    }
}
__global__ void k_dq4(const uint8_t* q, const float* s, int K, bf16* w) {
    const int n = blockIdx.x;
    for (int i = threadIdx.x; i < K / 2; i += blockDim.x) {
        const unsigned b = q[(size_t)n * (K / 2) + i];
        const float sc = s[(size_t)n * (K / 128) + (2 * i) / 128];
        w[(size_t)n * K + 2 * i] = __float2bfloat16((float)((int)((b & 15u) ^ 8u) - 8) * sc);
        w[(size_t)n * K + 2 * i + 1] = __float2bfloat16((float)((int)((b >> 4) ^ 8u) - 8) * sc);
    }
}
void dequant4(const uint8_t* q, const float* s, int N, int K, bf16* w, cudaStream_t st) { k_dq4<<<N, 256, 0, st>>>(q, s, K, w); }

void gemv8q(const int8_t* q, const float* s, int N, int K, const int8_t* xq, const float* xs, float* y, int ldy, int T,
            bool acc, cudaStream_t st, const int* dp, int nskip) {
    const int nb = (N + 7) / 8;
    if (nskip <= 0 || nskip % 8) dp = nullptr;  // (the skip is per block of 8 rows; review r27 L2)
    switch (T) {
        case 1: k_gemv8qt<1><<<nb, 256, 0, st>>>(q, s, N, K, xq, xs, y, ldy, acc, dp, nskip); break;
        case 2: k_gemv8qt<2><<<nb, 256, 0, st>>>(q, s, N, K, xq, xs, y, ldy, acc, dp, nskip); break;
        case 3: k_gemv8qt<3><<<nb, 256, 0, st>>>(q, s, N, K, xq, xs, y, ldy, acc, dp, nskip); break;
        case 4: k_gemv8qt<4><<<nb, 256, 0, st>>>(q, s, N, K, xq, xs, y, ldy, acc, dp, nskip); break;
        default: k_gemv8q<<<nb, 256, 0, st>>>(q, s, N, K, xq, xs, y, ldy, T, acc);
    }
}

// tile-major host slice (glm_cpu.cpp hx_pack_expert, source layout ss) -> row-major SliceL slot (sl); 16-byte units;
// src may be host-mapped
__global__ void k_untile(const char* src, char* dst, SliceL sl, SliceL ss) {
    const int ni = sl.ni;
    const size_t ng = (size_t)512 * (ni / 4), ns = (size_t)32 * (ni / 8), nd = (size_t)(ni / 8) * (H / 4),
                 nds = (size_t)(ni / 64) * (H / 8);
    const size_t tot = 2 * ng + 2 * ns + nd + nds;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < tot; i += (size_t)gridDim.x * blockDim.x) {
        size_t j = i;
        size_t so, dof;
        if (j < 2 * ng) {  // gq / uq: dst [512][ni] i32, 16 B = 4 cols
            const bool up = j >= ng;
            j %= ng;
            const size_t r = j / (ni / 4), c4 = j % (ni / 4), c = c4 * 4;
            dof = (up ? sl.uq : sl.gq) + (r * ni + c) * 4;
            if (ss.gu3) {
                *reinterpret_cast<uint4*>(dst + dof) = gu3_expand(src + (up ? ss.uq : ss.gq), r, c);
                continue;
            }
            so = (up ? ss.uq : ss.gq) + (((c / 32) * 512 + r) * 32 + c % 32) * 4;
        } else if ((j -= 2 * ng) < 2 * ns) {  // gs / us: dst [32][ni] f16, 16 B = 8 cols
            const bool up = j >= ns;
            j %= ns;
            const size_t g = j / (ni / 8), c = (j % (ni / 8)) * 8;
            dof = (up ? sl.us : sl.gs) + (g * ni + c) * 2;
            so = (up ? ss.us : ss.gs) + (((c / 32) * 32 + g) * 32 + c % 32) * 2;
        } else if ((j -= 2 * ns) < nd) {  // dq: dst [ni/8][4096] i32
            const size_t r = j / (H / 4), c = (j % (H / 4)) * 4;
            dof = sl.dq + (r * H + c) * 4;
            so = ss.dq + (((c / 32) * (ni / 8) + r) * 32 + c % 32) * 4;
        } else {  // ds: dst [ni/64][4096] f16
            j -= nd;
            const size_t g = j / (H / 8), c = (j % (H / 8)) * 8;
            dof = sl.ds + (g * H + c) * 2;
            so = ss.ds + (((c / 32) * (ni / 64) + g) * 32 + c % 32) * 2;
        }
        *reinterpret_cast<uint4*>(dst + dof) = *reinterpret_cast<const uint4*>(src + so);
    }
}

// round 27 (P4 b): copy n words from pinned host memory with a kernel (zero-copy reads) instead of a copy-engine
// transfer, so a small upload does not queue behind the ring DMA on the H2D engine
__global__ void k_h2d_words(int* __restrict__ d0, const int* __restrict__ s0, int* __restrict__ d1,
                            const int* __restrict__ s1, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        d0[i] = s0[i];
        d1[i] = s1[i];
    }
}
void h2d_words2(int* d0, const int* hs0, int* d1, const int* hs1, int n, cudaStream_t st) {
    if (n <= 0) return;
    k_h2d_words<<<std::min(64, (n + 255) / 256), 256, 0, st>>>(d0, hs0, d1, hs1, n);
}
void untile(const char* src, char* dst, const SliceL& sl, int blocks, cudaStream_t st, const SliceL* ssl) {
    k_untile<<<blocks, 256, 0, st>>>(src, dst, sl, ssl ? *ssl : sl);
}

// ---------------------------------------------------------------- MTP (layer 45, no mHC)

// xcat[t] = bf16([enorm(emb(tok[t])), hnorm(h[t])]) ; one block per row
__global__ void k_mtp_in(const int* tok, const bf16* emb, const float* h, const float* enw, const float* hnw, bf16* xcat) {
    __shared__ float sh[32];
    const int t = blockIdx.x;
    const bf16* e = emb + (size_t)tok[t] * H;
    const float* hr = h + (size_t)t * H;
    float se = 0.f, shh = 0.f;
    for (int i = threadIdx.x; i < H; i += blockDim.x) {
        const float a = __bfloat162float(e[i]);
        se += a * a;
        shh += hr[i] * hr[i];
    }
    se = block_sum(se, sh);
    shh = block_sum(shh, sh);
    const float ie = rsqrtf(se / H + EPS), ih = rsqrtf(shh / H + EPS);
    for (int i = threadIdx.x; i < H; i += blockDim.x) {
        xcat[(size_t)t * 2 * H + i] = __float2bfloat16(__bfloat162float(e[i]) * ie * enw[i]);
        xcat[(size_t)t * 2 * H + H + i] = __float2bfloat16(hr[i] * ih * hnw[i]);
    }
}
void mtp_in(const int* tok, const bf16* emb, const float* h, const float* enw, const float* hnw, bf16* xcat, int T,
            cudaStream_t st) {
    k_mtp_in<<<T, 512, 0, st>>>(tok, emb, h, enw, hnw, xcat);
}
// optional x += a first; then xf / xb = rms(x) * w
__global__ void k_add_rms(float* x, const float* a, const float* w, float* xf, bf16* xb) {
    __shared__ float sh[32];
    const int t = blockIdx.x;
    float* r = x + (size_t)t * H;
    float s = 0.f;
    for (int i = threadIdx.x; i < H; i += blockDim.x) {
        float v = r[i];
        if (a) { v += a[(size_t)t * H + i]; r[i] = v; }
        s += v * v;
    }
    s = block_sum(s, sh);
    if (!w) return;
    const float iv = rsqrtf(s / H + EPS);
    for (int i = threadIdx.x; i < H; i += blockDim.x) {
        const float v = r[i] * iv * w[i];
        if (xf) xf[(size_t)t * H + i] = v;
        if (xb) xb[(size_t)t * H + i] = __float2bfloat16(v);
    }
}
void add_rms(float* x, const float* a, const float* w, float* xf, bf16* xb, int T, cudaStream_t st) {
    k_add_rms<<<T, 512, 0, st>>>(x, a, w, xf, xb);
}

// ---------------------------------------------------------------- misc

// per row: max and its index -> out[2 r] (float bits), out[2 r + 1]
__global__ void k_argmax(const float* x, int n, int* out) {
    __shared__ float sv[32];
    __shared__ int si[32];
    const int r = blockIdx.x;
    const float* p = x + (size_t)r * n;
    float bv = -FLT_MAX;
    int bi = 0;
    for (int i = threadIdx.x; i < n; i += blockDim.x)
        if (p[i] > bv) { bv = p[i]; bi = i; }
    for (int o = 16; o; o >>= 1) {
        const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
        const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
        if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
    }
    if ((threadIdx.x & 31) == 0) { sv[threadIdx.x >> 5] = bv; si[threadIdx.x >> 5] = bi; }
    __syncthreads();
    if (threadIdx.x == 0) {
        for (int w = 1; w < (int)(blockDim.x >> 5); w++)
            if (sv[w] > bv || (sv[w] == bv && si[w] < bi)) { bv = sv[w]; bi = si[w]; }
        out[2 * r] = __float_as_int(bv);
        out[2 * r + 1] = bi;
    }
}
void argmax_rows(const float* x, int n, int rows, int* out, cudaStream_t st) { k_argmax<<<rows, 1024, 0, st>>>(x, n, out); }
// per row: sum exp(x - max) with the max from k_argmax (out[2 r]) -> se[r] (softmax normaliser of the slice)
__global__ void k_rowse(const float* x, int n, const int* am, float* se) {
    __shared__ float sh[32];
    const int r = blockIdx.x;
    const float mx = __int_as_float(am[2 * r]);
    const float* p = x + (size_t)r * n;
    float s = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) s += __expf(p[i] - mx);
    for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
    if ((threadIdx.x & 31) == 0) sh[threadIdx.x >> 5] = s;
    __syncthreads();
    if (threadIdx.x == 0) {
        float t = 0.f;
        for (int w = 0; w < (int)(blockDim.x >> 5); w++) t += sh[w];
        se[r] = t;
    }
}
void row_se(const float* x, int n, int rows, const int* am, float* se, cudaStream_t st) {
    k_rowse<<<rows, 1024, 0, st>>>(x, n, am, se);
}

// sampling candidates of a row slice: top-TKC (value desc, index asc) in TKC rounds (each round: the best element
// after the previous pick), plus max and sum exp((x - max) * invT) of the whole slice (global normaliser on the host)
__global__ void k_topk(const float* x, int n, const float* invT, TopK* out) {
    __shared__ float sv[32];
    __shared__ int si[32];
    __shared__ float bcv;
    __shared__ int bci;
    const int r = blockIdx.x;
    const float* p = x + (size_t)r * n;
    TopK& o = out[r];
    float pv = FLT_MAX;
    int pi = -1;
    for (int k = 0; k < TKC; k++) {
        float bv = -FLT_MAX;
        int bi = 0x7fffffff;
        for (int i = threadIdx.x; i < n; i += blockDim.x) {
            const float v = p[i];
            const bool after = v < pv || (v == pv && i > pi);
            if (after && (v > bv || (v == bv && i < bi))) { bv = v; bi = i; }
        }
        for (int s = 16; s; s >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, bv, s);
            const int oi = __shfl_xor_sync(0xffffffffu, bi, s);
            if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
        }
        if ((threadIdx.x & 31) == 0) { sv[threadIdx.x >> 5] = bv; si[threadIdx.x >> 5] = bi; }
        __syncthreads();
        if (threadIdx.x == 0) {
            for (int w = 1; w < (int)(blockDim.x >> 5); w++)
                if (sv[w] > bv || (sv[w] == bv && si[w] < bi)) { bv = sv[w]; bi = si[w]; }
            bcv = bv; bci = bi;
            o.v[k] = bv; o.i[k] = bi;
        }
        __syncthreads();
        pv = bcv; pi = bci;
        __syncthreads();
    }
    const float mx = o.v[0], it = *invT;
    float s = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) s += __expf((p[i] - mx) * it);
    for (int k = 16; k; k >>= 1) s += __shfl_xor_sync(0xffffffffu, s, k);
    if ((threadIdx.x & 31) == 0) sv[threadIdx.x >> 5] = s;
    __syncthreads();
    if (threadIdx.x == 0) {
        float t = 0.f;
        for (int w = 0; w < (int)(blockDim.x >> 5); w++) t += sv[w];
        o.mx = mx; o.se = t;
    }
}
void topk_rows(const float* x, int n, int rows, const float* invT, TopK* out, cudaStream_t st) {
    k_topk<<<rows, 1024, 0, st>>>(x, n, invT, out);
}

void kernels_init() {}

__global__ void k_f2b(const float* x, bf16* y, size_t n) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
        y[i] = __float2bfloat16(x[i]);
}
void f2b(const float* x, bf16* y, size_t n, cudaStream_t st) { k_f2b<<<512, 256, 0, st>>>(x, y, n); }
// y[t][0..n) = bf16(x[t * ldx + 0..n)), n % 4 == 0, ldx % 4 == 0
__global__ void k_f2b_rows(const float* x, int ldx, bf16* y, int n, int T) {
    const int n4 = n / 4;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < (size_t)T * n4; i += (size_t)gridDim.x * blockDim.x) {
        const size_t t = i / n4, c = i % n4;
        const float4 v = *reinterpret_cast<const float4*>(x + t * ldx + 4 * c);
        bf162* o = reinterpret_cast<bf162*>(y + t * n + 4 * c);
        o[0] = __floats2bfloat162_rn(v.x, v.y);
        o[1] = __floats2bfloat162_rn(v.z, v.w);
    }
}
void f2b_rows(const float* x, int ldx, bf16* y, int n, int T, cudaStream_t st) {
    k_f2b_rows<<<1024, 256, 0, st>>>(x, ldx, y, n, T);
}
__global__ void k_b2f(const bf16* x, float* y, size_t n) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
        y[i] = __bfloat162float(x[i]);
}
void b2f(const bf16* x, float* y, size_t n, cudaStream_t st) { k_b2f<<<512, 256, 0, st>>>(x, y, n); }

// strided copy of the q (own heads) part: [T][ld] cols [0, n) -> dense [T][n]
__global__ void k_cols(const float* x, int ld, float* y, int n) {
    const int t = blockIdx.y;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) y[(size_t)t * n + i] = x[(size_t)t * ld + i];
}
void copy_cols(const float* x, int ld, float* y, int n, int T, cudaStream_t st) {
    k_cols<<<dim3(16, T), 256, 0, st>>>(x, ld, y, n);
}

// ---------------------------------------------------------------- prefill: merge + the next block's mixes in one pass
// R_i <- post_i a + sum_j comb[j][i] R_j, then the parts of mix[t][j] = fn[j] . R_new[t] (24 dots over the 4 x 4096 new
// streams, fp32) and of the sum of squares of a block of MMC columns, as partials mixp[t][H / MMC][25] (k_mhc_pre sums
// them in a fixed order): the separate fp32 SGEMM pass over R goes away. a == nullptr: no merge (after the embedding).
// Grid (H / MMC column blocks, row blocks); the block's slice of fn stays in shared memory (48 KB); a warp takes two
// rows at a time, a lane 4 columns of each stream (512-byte row segments per warp load); the 2 x 25 per-lane dot
// partials are reduced over the warp with a butterfly reduce-scatter (62 shuffles).
constexpr int MMC = 128, MMNB = H / MMC, MMRPC = 256;
template <int HALF, int M>
__device__ __forceinline__ void bfly_step(float (&v)[64], int ln) {
    const bool up = ln & M;
#pragma unroll
    for (int i = 0; i < HALF; i++) {
        const float send = up ? v[i] : v[HALF + i];
        const float keep = up ? v[HALF + i] : v[i];
        v[i] = keep + __shfl_xor_sync(0xffffffffu, send, M);
    }
}
__global__ void __launch_bounds__(256, 1) k_merge_mix_pf(float* R, const float* __restrict__ post,
                                                         const float* __restrict__ comb, const float* a,
                                                         const float* __restrict__ fn, float* mixp, int T) {
    extern __shared__ __align__(16) float fsm[];  // [j * 4 + s][MMC]
    const int cb = blockIdx.x, c0 = cb * MMC, tid = threadIdx.x, w = tid >> 5, ln = tid & 31;
    for (int i = tid; i < NMIX * 4 * MMC / 4; i += 256) {
        const int js = i / (MMC / 4), c4 = i % (MMC / 4), j = js >> 2, s_ = js & 3;
        reinterpret_cast<float4*>(fsm)[i] = *reinterpret_cast<const float4*>(fn + (size_t)j * RW + s_ * H + c0 + 4 * c4);
    }
    __syncthreads();
    const int rbeg = blockIdx.y * MMRPC, rend = min(T, rbeg + MMRPC);
    for (int r = rbeg + 2 * w; r < rend; r += 16) {
        float v[64];
        float4 y[2][4];
#pragma unroll
        for (int h = 0; h < 2; h++) {
            const int t = r + h;
            float q = 0.f;
            if (t < rend) {
                float* rr = R + (size_t)t * RW + c0 + 4 * ln;
                float4 x[4];
#pragma unroll
                for (int k = 0; k < 4; k++) x[k] = *reinterpret_cast<const float4*>(rr + k * H);
                if (a) {
                    const float4 av = *reinterpret_cast<const float4*>(a + (size_t)t * H + c0 + 4 * ln);
                    const float* p = post + t * 4;
                    const float* c = comb + t * 16;
#pragma unroll
                    for (int k = 0; k < 4; k++) {
                        const float pk = __ldg(p + k), ck0 = __ldg(c + k), ck1 = __ldg(c + 4 + k), ck2 = __ldg(c + 8 + k),
                                    ck3 = __ldg(c + 12 + k);
                        y[h][k].x = pk * av.x + ck0 * x[0].x + ck1 * x[1].x + ck2 * x[2].x + ck3 * x[3].x;
                        y[h][k].y = pk * av.y + ck0 * x[0].y + ck1 * x[1].y + ck2 * x[2].y + ck3 * x[3].y;
                        y[h][k].z = pk * av.z + ck0 * x[0].z + ck1 * x[1].z + ck2 * x[2].z + ck3 * x[3].z;
                        y[h][k].w = pk * av.w + ck0 * x[0].w + ck1 * x[1].w + ck2 * x[2].w + ck3 * x[3].w;
                    }
#pragma unroll
                    for (int k = 0; k < 4; k++) *reinterpret_cast<float4*>(rr + k * H) = y[h][k];
                } else {
#pragma unroll
                    for (int k = 0; k < 4; k++) y[h][k] = x[k];
                }
#pragma unroll
                for (int k = 0; k < 4; k++) q += y[h][k].x * y[h][k].x + y[h][k].y * y[h][k].y + y[h][k].z * y[h][k].z + y[h][k].w * y[h][k].w;
            } else {
#pragma unroll
                for (int k = 0; k < 4; k++) y[h][k] = make_float4(0.f, 0.f, 0.f, 0.f);
            }
            v[h * 32 + NMIX] = q;
#pragma unroll
            for (int j = NMIX + 1; j < 32; j++) v[h * 32 + j] = 0.f;
        }
#pragma unroll
        for (int j = 0; j < NMIX; j++) {
            float d0 = 0.f, d1 = 0.f;
#pragma unroll
            for (int k = 0; k < 4; k++) {
                const float4 f = reinterpret_cast<const float4*>(fsm + (j * 4 + k) * MMC)[ln];
                d0 += f.x * y[0][k].x + f.y * y[0][k].y + f.z * y[0][k].z + f.w * y[0][k].w;
                d1 += f.x * y[1][k].x + f.y * y[1][k].y + f.z * y[1][k].z + f.w * y[1][k].w;
            }
            v[j] = d0;
            v[32 + j] = d1;
        }
        // butterfly reduce-scatter of v[64] over the 32 lanes: after the steps lane ln holds the warp sums of
        // indices idx and idx + 1, idx = 32 b4 + 16 b3 + 8 b2 + 4 b1 + 2 b0 (b = bits of ln, b4 = bit 4)
        bfly_step<32, 16>(v, ln);
        bfly_step<16, 8>(v, ln);
        bfly_step<8, 4>(v, ln);
        bfly_step<4, 2>(v, ln);
        bfly_step<2, 1>(v, ln);
        const int idx = ((ln >> 4) & 1) * 32 + ((ln >> 3) & 1) * 16 + ((ln >> 2) & 1) * 8 + ((ln >> 1) & 1) * 4 + (ln & 1) * 2;
#pragma unroll
        for (int e = 0; e < 2; e++) {
            const int id = idx + e, h = id >> 5, j = id & 31, t = r + h;
            if (j <= NMIX && t < rend) mixp[((size_t)t * MMNB + cb) * (NMIX + 1) + j] = v[e];
        }
    }
}
void merge_mix_pf(float* R, const float* post, const float* comb, const float* a, const float* fn, float* mixp, int T,
                  cudaStream_t st) {
    static int done[8] = {};
    int dev = 0;
    cudaGetDevice(&dev);
    if (!done[dev & 7]) {
        cudaFuncSetAttribute(k_merge_mix_pf, cudaFuncAttributeMaxDynamicSharedMemorySize, NMIX * 4 * MMC * 4);
        done[dev & 7] = 1;
    }
    k_merge_mix_pf<<<dim3(MMNB, (T + MMRPC - 1) / MMRPC), 256, NMIX * 4 * MMC * 4, st>>>(R, post, comb, a, fn, mixp, T);
}
int merge_mix_pf_parts() { return MMNB; }

// ---------------------------------------------------------------- prefill KDA, chunked delta rule (round 10, P-5)
// Chunks of KC = 16 tokens per head (tools/glm/glm_model.py kda_chunk, WY form, fp32). With G_t the cumulative log decay
// inside the chunk:  (I + A) [W1 | W2] = [beta V | beta (K exp G)],  A_ts = beta_t sum_k k_tk k_sk exp(G_t - G_s)_k (s < t),
// M_ts = sum_k q_tk k_sk exp(G_t - G_s)_k (s <= t). Then, sequentially over chunks: U = W1 - W2 S, O = (Q exp G) S + M U,
// S = exp(G_C) S + (K exp(G_C - G))^T U.  Phase 1 (k_kda_c1) is chunk-parallel; phase 2 (k_kda_c2) runs the chunks of a
// head in order, the state split by value columns over KVB blocks (the columns are independent).
constexpr int KC = 16, KCF = KC * KD * 4 + KC * KC + KD;  // floats per (chunk, head): W1, W2, QG, KG, M, GC
constexpr int KVB = 4, KVC = KD / KVB;                    // phase 2: column blocks per head, columns per block
size_t kda_chunk_floats(int T, int nh) { return (size_t)((T + KC - 1) / KC) * nh * KCF; }
__global__ void __launch_bounds__(128) k_kda_c1(const float* q, const float* k, const float* v, const float* g,
                                                const float* beta, int T, int nh, float* out) {
    __shared__ float Gs[KC][KD], ks[KC][KD], qs[KC][KD], As[KC][KC + 1], Ms[KC][KC + 1], bs[KC];
    const int c = blockIdx.x, h = blockIdx.y, d = threadIdx.x, w = d >> 5, ln = d & 31;
    const int t0 = c * KC;
    float acc = 0.f;
    float vv[KC];
#pragma unroll
    for (int t = 0; t < KC; t++) {
        const int tt = t0 + t;
        const bool ok = tt < T;
        const size_t i = ((size_t)tt * nh + h) * KD + d;
        acc += ok ? __logf(g[i]) : 0.f;
        Gs[t][d] = acc;
        ks[t][d] = ok ? k[i] : 0.f;
        qs[t][d] = ok ? q[i] : 0.f;
        vv[t] = ok ? v[i] : 0.f;
        if (d == 0) bs[t] = ok ? beta[tt * nh + h] : 0.f;
    }
    __syncthreads();
    // A, M: pairs s <= t (136), a warp per pair, lanes over the channels
    for (int pr = w; pr < KC * (KC + 1) / 2; pr += 4) {
        const int t = (int)((sqrtf(8.f * pr + 1.f) - 1.f) * 0.5f + 1e-4f);
        const int s = pr - t * (t + 1) / 2;
        float a = 0.f, m = 0.f;
#pragma unroll
        for (int j = 0; j < 4; j++) {
            const int dd = ln + 32 * j;
            const float e = __expf(Gs[t][dd] - Gs[s][dd]);
            const float ke = ks[s][dd] * e;
            a += ks[t][dd] * ke;
            m += qs[t][dd] * ke;
        }
#pragma unroll
        for (int o = 16; o; o >>= 1) {
            a += __shfl_xor_sync(0xffffffffu, a, o);
            m += __shfl_xor_sync(0xffffffffu, m, o);
        }
        if (ln == 0) {
            As[t][s] = s < t ? bs[t] * a : 0.f;
            Ms[t][s] = m;
        }
    }
    __syncthreads();
    float* o = out + ((size_t)c * nh + h) * KCF;
    float* W1 = o;                    // [KC][KD]
    float* W2 = W1 + KC * KD;         // [KC][KD]
    float* QG = W2 + KC * KD;         // [KC][KD]
    float* KG = QG + KC * KD;         // [KC][KD]
    float* M = KG + KC * KD;          // [KC][KC]
    float* GC = M + KC * KC;          // [KD]
    // forward substitution, thread d: column d of both right-hand sides
    float w1[KC], w2[KC];
#pragma unroll
    for (int i = 0; i < KC; i++) {
        float x1 = bs[i] * vv[i], x2 = bs[i] * ks[i][d] * __expf(Gs[i][d]);
#pragma unroll
        for (int j = 0; j < i; j++) { x1 -= As[i][j] * w1[j]; x2 -= As[i][j] * w2[j]; }
        w1[i] = x1; w2[i] = x2;
        W1[i * KD + d] = x1;
        W2[i * KD + d] = x2;
    }
    const float gc = Gs[KC - 1][d];
#pragma unroll
    for (int i = 0; i < KC; i++) {
        QG[i * KD + d] = qs[i][d] * __expf(Gs[i][d]);
        KG[i * KD + d] = ks[i][d] * __expf(gc - Gs[i][d]);
    }
    GC[d] = __expf(gc);
    for (int i = d; i < KC * KC; i += 128) M[i] = (i % KC) <= (i / KC) ? Ms[i / KC][i % KC] : 0.f;
}
// phase 2: grid (nh, KVB), 128 threads; S block [KD][KVC] in smem (rows padded), thread r keeps row r of the new state.
// The chunk data (W1 columns, W2, QG, KG, M, GC) are read through L1 (every CTA of the grid must be resident at once:
// the chunks run in sequence, a second wave would double the time)
__global__ void __launch_bounds__(256) k_kda_c2(const float* __restrict__ cf, float* S, float* o, int T, int nh) {
    __shared__ __align__(16) float Ss[KD][KVC + 4];
    __shared__ __align__(16) float Us[KC][KVC + 4];
    __shared__ __align__(16) float Ps[2][KC][KVC + 4];  // k-half partials of U (P in Ps[1] after the combine)
    const int h = blockIdx.x, cb = blockIdx.y, c0 = cb * KVC, tid = threadIdx.x, kh = tid >> 7, lt = tid & 127;
    const int nc = (T + KC - 1) / KC;
    for (int i = tid; i < KD * KVC; i += 256) {
        const int vv = i / KD, kk = i % KD;
        Ss[kk][vv] = S[((size_t)h * KD + c0 + vv) * KD + kk];
    }
    const int i_ = lt >> 3, jq = lt & 7;  // U / P partials: row i_, columns 4 jq.., k in [64 kh, 64 kh + 64)
    static_assert(KVC == 32, "phase 2 mapping: 32 columns per block");
    __syncthreads();
    for (int c = 0; c < nc; c++) {
        const float* W1 = cf + ((size_t)c * nh + h) * KCF;
        const float* W2 = W1 + KC * KD;
        const float* QG = W2 + KC * KD;
        const float* KG = QG + KC * KD;
        const float* M = KG + KC * KD;
        const float* GC = M + KC * KC;
        const int kk = lt;
        float kg[KC];
#pragma unroll
        for (int s_ = 0; s_ < KC; s_++) kg[s_] = __ldg(KG + s_ * KD + kk);
        const float gk = __ldg(GC + kk);
        float4 u = make_float4(0.f, 0.f, 0.f, 0.f), pq = make_float4(0.f, 0.f, 0.f, 0.f);
        const float4* w2 = reinterpret_cast<const float4*>(W2 + i_ * KD + 64 * kh);
        const float4* qg = reinterpret_cast<const float4*>(QG + i_ * KD + 64 * kh);
        float4 a2v[16], aqv[16];
#pragma unroll
        for (int k4 = 0; k4 < 16; k4++) { a2v[k4] = __ldg(w2 + k4); aqv[k4] = __ldg(qg + k4); }
#pragma unroll
        for (int k4 = 0; k4 < 16; k4++) {
            const float av[4] = {a2v[k4].x, a2v[k4].y, a2v[k4].z, a2v[k4].w};
            const float qv[4] = {aqv[k4].x, aqv[k4].y, aqv[k4].z, aqv[k4].w};
#pragma unroll
            for (int e = 0; e < 4; e++) {
                const float4 sv = *reinterpret_cast<const float4*>(&Ss[64 * kh + 4 * k4 + e][4 * jq]);
                u.x += av[e] * sv.x; u.y += av[e] * sv.y; u.z += av[e] * sv.z; u.w += av[e] * sv.w;
                pq.x += qv[e] * sv.x; pq.y += qv[e] * sv.y; pq.z += qv[e] * sv.z; pq.w += qv[e] * sv.w;
            }
        }
        if (kh) {
            *reinterpret_cast<float4*>(&Ps[0][i_][4 * jq]) = u;
            *reinterpret_cast<float4*>(&Ps[1][i_][4 * jq]) = pq;
        }
        __syncthreads();
        if (!kh) {
            const float4 w1 = __ldg(reinterpret_cast<const float4*>(W1 + i_ * KD + c0 + 4 * jq));
            const float4 u1 = *reinterpret_cast<const float4*>(&Ps[0][i_][4 * jq]);
            const float4 p1 = *reinterpret_cast<const float4*>(&Ps[1][i_][4 * jq]);
            u = make_float4(w1.x - (u.x + u1.x), w1.y - (u.y + u1.y), w1.z - (u.z + u1.z), w1.w - (u.w + u1.w));
            pq.x += p1.x; pq.y += p1.y; pq.z += p1.z; pq.w += p1.w;
            *reinterpret_cast<float4*>(&Us[i_][4 * jq]) = u;
        }
        __syncthreads();
        if (!kh) {  // O = P + M U
#pragma unroll
            for (int s_ = 0; s_ < KC; s_++) {
                const float mm = __ldg(M + i_ * KC + s_);
                const float4 us = *reinterpret_cast<const float4*>(&Us[s_][4 * jq]);
                pq.x += mm * us.x; pq.y += mm * us.y; pq.z += mm * us.z; pq.w += mm * us.w;
            }
            const int t = c * KC + i_;
            if (t < T) *reinterpret_cast<float4*>(o + ((size_t)t * nh + h) * KD + c0 + 4 * jq) = pq;
        }
        // S = GC S + KG^T U: thread (kh, kk): row kk, columns 16 kh .. 16 kh + 15
#pragma unroll
        for (int j4 = 4 * kh; j4 < 4 * kh + 4; j4++) {
            float4 x = *reinterpret_cast<float4*>(&Ss[kk][4 * j4]);
            x.x *= gk; x.y *= gk; x.z *= gk; x.w *= gk;
#pragma unroll
            for (int s_ = 0; s_ < KC; s_++) {
                const float4 us = *reinterpret_cast<const float4*>(&Us[s_][4 * j4]);
                x.x += kg[s_] * us.x; x.y += kg[s_] * us.y; x.z += kg[s_] * us.z; x.w += kg[s_] * us.w;
            }
            *reinterpret_cast<float4*>(&Ss[kk][4 * j4]) = x;
        }
        __syncthreads();
    }
    for (int i = tid; i < KD * KVC; i += 256) {
        const int vv = i / KD, kk = i % KD;
        S[((size_t)h * KD + c0 + vv) * KD + kk] = Ss[kk][vv];
    }
}
int kda_chunk_dbg = 3;  // kbench: bit 1 phase 1, bit 2 phase 2
void kda_chunk_rec(const float* q, const float* k, const float* v, const float* g, const float* beta, float* S, float* o,
                   int T, int nh, float* scratch, cudaStream_t st) {
    const int nc = (T + KC - 1) / KC;
    if (kda_chunk_dbg & 1) k_kda_c1<<<dim3(nc, nh), 128, 0, st>>>(q, k, v, g, beta, T, nh, scratch);
    if (!(kda_chunk_dbg & 2)) return;
    k_kda_c2<<<dim3(nh, KVB), 256, 0, st>>>(scratch, S, o, T, nh);
}

}  // namespace glm
