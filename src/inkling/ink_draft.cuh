// DSpark block drafter (mtp.safetensors) kernels, [ink-dspark-fwd]. Included from ink.cu (namespace ink).
// Wiring = the drafter gate's CPU forward (tools/inkling_drafter_gate/draft.py, config cfg_ds): taps = outputs of
// applications 4 / 22 / 40 + the final normed hidden; per drafter layer i a fused context f_i =
// RMSNorm_i(sum_j softmax(fuse[i])_j tap_j) gives context keys k_norm_ctx(k_proj_ctx f) (+RoPE) and values v_proj_ctx f
// at every committed position; the block (row 0 = normed embedding of the anchor, rows 1..7 = mask_embed) runs 3
// pre-norm layers (q / k norms per head, RoPE theta 1e6 half rotation, GQA 24 / 8, layers 0 / 2 window 1024) that attend
// to the context < anchor and to all 8 block rows; dense SwiGLU. Head: rows 1..7, h = norm(x), Markov state
// s' = g s + (1 - g) tanh(c), [g c o] = P [s, W1[prev], h, f_3(anchor - 1)] + b, logits = U (h + W_out tanh(o) + b_out).
// Draft logits need not be exact (the verify keeps the output exact): U h is computed once per block for the 7 rows,
// the per-position correction as UO = U W_out ([V][256], fp16) and uOb = U b_out.

constexpr int DFF = 9216, DQKV = QD + 2 * KVD, DB = 8;   // drafter MLP width, fused q|k|v rows, block
constexpr int MR = 256;                                  // Markov rank
constexpr int DNS_L = 8, DNS_LL = 32, DNS_G = 32, DNS_GX = 128;   // drafter attention splits: local (k_dattn A/B 63) / local (k_dattnL) / global
static_assert(DNS_GX >= DNS_LL && DNS_GX >= DNS_G && DNS_GX >= DNS_L, "partial buffers sized DNS_GX");
static_assert(DB % 4 == 0, "k_dattnL: 4 rows per block");   // review r4b L

// fused context rows: grid (n, 4), 256 threads. F[i][t] = RMSNorm(nw[i]) (sum_j fw[i][j] tap_j[t]) (bf16)
struct A_dfuse {
    const float* tap[4];
    float fw[16];
    const bf16* nw[4];
    bf16* F[4];
};
__global__ void __launch_bounds__(256) k_dfuse(const __grid_constant__ A_dfuse A) {
    __shared__ float red[33];
    const int t = blockIdx.x, i = blockIdx.y;
    float v[9], ss[1] = {0.f};
#pragma unroll
    for (int k = 0; k < 9; k++) {
        const size_t c = (size_t)t * H + threadIdx.x + 256 * k;
        v[k] = A.fw[i * 4 + 0] * A.tap[0][c] + A.fw[i * 4 + 1] * A.tap[1][c] + A.fw[i * 4 + 2] * A.tap[2][c] + A.fw[i * 4 + 3] * A.tap[3][c];
        ss[0] += v[k] * v[k];
    }
    block_sum<1>(ss, red);
    const float inv = rsqrtf(ss[0] / H + 1e-6f);
#pragma unroll
    for (int k = 0; k < 9; k++) {
        const int c = threadIdx.x + 256 * k;
        A.F[i][(size_t)t * H + c] = __float2bfloat16(__bfloat162float(A.nw[i][c]) * (v[k] * inv));
    }
}

// RoPE (theta 1e6, half rotation) of one 128-dim head held by 128 threads (x in smem xs), position pos.
// [ink-dr-rope] the 64 inverse frequencies come from a table filled once by the same device pow (k_rope_init), and
// each angle's cos / sin is computed once by thread i < 64 for both halves (same FP64 ops: bit-identical)
__device__ double g_rope_inv[64];
__global__ void k_rope_init() {
    const int i = threadIdx.x;
    if (i < 64) g_rope_inv[i] = pow(1e6, -(double)(2 * i) / HD);
}
// per-head RMSNorm (weight) of 128 values, then optional RoPE; 128 threads
__device__ __forceinline__ float head_norm_rope(float v, const bf16* w, int pos, bool rope, float* xs, float* red) {
    __shared__ float rc[64], rs[64];
    const int d = threadIdx.x;
    if (rope && d < 64) {
        const double ang = (double)pos * g_rope_inv[d];
        rc[d] = (float)cos(ang);
        rs[d] = (float)sin(ang);
    }
    float s = warp_sum(v * v);
    if ((d & 31) == 0) red[d >> 5] = s;
    __syncthreads();
    const float inv = rsqrtf((red[0] + red[1] + red[2] + red[3]) / HD + 1e-6f);
    const float y = __bfloat162float(w[d]) * (v * inv);
    if (!rope) return y;
    xs[d] = y;
    __syncthreads();
    const int i = d & 63;
    const float c = rc[i], sn = rs[i];
    return d < 64 ? xs[d] * c - xs[d + 64] * sn : xs[d] * c + xs[d - 64] * sn;
}

// [ink-dr-fp8] drafter K / V cache in FP8 E4M3 with one fp32 scale per (head, slot) = amax / 448 (the verify decides
// every token, so drafter-only precision changes only acceptance). 128 threads hold the 128 values of a head row.
__device__ __forceinline__ void fp8_store_row(float v, uint8_t* row, float* scl, float* red2) {
    const int d = threadIdx.x;
    float a = warp_max(fabsf(v));
    __syncthreads();   // red2 reuse
    if ((d & 31) == 0) red2[d >> 5] = a;
    __syncthreads();
    const float am = fmaxf(fmaxf(red2[0], red2[1]), fmaxf(red2[2], red2[3]));
    const float sc = am > 0.f ? am * (1.f / 448.f) : 1.f;
    // e4m3 = the fp16 bits of x / 256 (|x| <= 448) rounded to 3 mantissa bits (sm_86: no FP8 convert instruction)
    const uint32_t hb = __half_as_ushort(__float2half_rn((v / sc) * (1.f / 256.f)));
    const uint32_t m = (hb & 0x7fffu) + 0x3fu + ((hb >> 7) & 1u);
    row[d] = (uint8_t)(((m >> 7) & 0x7fu) | ((hb >> 8) & 0x80u));
    if (d == 0) *scl = sc;
}
__device__ __forceinline__ void fp8x4_load(const uint8_t* p, float sc, float (&o)[4]) {
    const uint32_t w = *(const uint32_t*)p;   // sm_86 has no FP8 convert: the e4m3 bits as fp16 bits (x 2^-8), exact
    const float2 fa = fp8pair(w), fb = fp8pair(w >> 8);   // bytes 0, 2 | 1, 3
    const float s2 = sc * 256.f;
    o[0] = fa.x * s2;
    o[1] = fb.x * s2;
    o[2] = fa.y * s2;
    o[3] = fb.y * s2;
}

// context K / V of n committed rows: grid (n, NKV), 128 threads. KV [n][2 KVD] fp32 (k | v) -> caches at pos0 + t
struct A_dctx {
    const float* KV;
    const bf16* kn;
    uint8_t* K;
    uint8_t* V;
    float *KS, *VS;   // [NKV][cap] scales
    int cap, ring;
    const int* pos0;
    const int* rt = nullptr;   // [ink-b-draft] row steps: row table (position RT[t], sequence slot RT[RT_SLOT + t])
    size_t kstr = 0, sstr = 0; // per sequence slot: K / V bytes, scale floats
    const int* sko = nullptr;   // [ink-srv-pool] slot offsets in HD-rows (K bytes / HD, scale floats); null: so x kstr / sstr
};
__global__ void __launch_bounds__(128) k_dctx(const __grid_constant__ A_dctx A) {
    __shared__ float xs[HD], red[4], red2[4];
    const int t = blockIdx.x, h = blockIdx.y, d = threadIdx.x;
    const int pos = A.rt ? A.rt[t] : *A.pos0 + t;
    const size_t so = A.rt ? (size_t)A.rt[RT_SLOT + t] : 0;
    const float* kv = A.KV + (size_t)t * 2 * KVD;
    const float k = head_norm_rope(kv[h * HD + d], A.kn, pos, true, xs, red);
    const int slot = A.ring ? pos % A.cap : pos;
    const size_t ri = (size_t)h * A.cap + slot;
    const size_t ko = A.sko ? (size_t)A.sko[so] * HD : so * A.kstr, kso = A.sko ? (size_t)A.sko[so] : so * A.sstr;
    fp8_store_row(k, A.K + ko + ri * HD, A.KS + kso + ri, red2);
    fp8_store_row(kv[KVD + h * HD + d], A.V + ko + ri * HD, A.VS + kso + ri, red2);
}

// block q / k / v post: grid (DB, NQ + NKV), 128 threads. QKV [DB][DQKV] fp32; q -> Qd [DB][QD]; k, v -> caches at
// anchor + row
struct A_dqk {
    const float* QKV;
    const bf16* qn;
    const bf16* kn;
    float* Q;
    uint8_t* K;
    uint8_t* V;
    float *KS, *VS;
    int cap, ring;
    const int* anchor;         // [segment] anchor positions (rows 8 z .. 8 z + 7 = segment z)
    const int* dsl = nullptr;  // [ink-b-draft] [segment] sequence slot (null: slot 0 of the given pointers)
    size_t kstr = 0, sstr = 0;
    const int* sko = nullptr;   // [ink-srv-pool]
    int q0 = 0, nq = NQ, kv0 = 0, nkv = NKV;   // [ink-tp-dra] the rank's heads; QKV rows [nq q | nkv k | nkv v] heads
};
__global__ void __launch_bounds__(128) k_dqk(const __grid_constant__ A_dqk A) {   // grid (rows, nq + nkv)
    __shared__ float xs[HD], red[4], red2[4];
    const int r = blockIdx.x, h = blockIdx.y, d = threadIdx.x;
    const int pos = A.anchor[r / DB] + r % DB;
    const size_t so = A.dsl ? (size_t)A.dsl[r / DB] : 0;
    const int ld = (A.nq + 2 * A.nkv) * HD;
    const float* row = A.QKV + (size_t)r * ld;
    if (h < A.nq) {
        A.Q[(size_t)r * QD + (A.q0 + h) * HD + d] = head_norm_rope(row[h * HD + d], A.qn, pos, true, xs, red);
        return;
    }
    const int hl = h - A.nq, kh = A.kv0 + hl;
    const float k = head_norm_rope(row[A.nq * HD + hl * HD + d], A.kn, pos, true, xs, red);
    const int slot = A.ring ? pos % A.cap : pos;
    const size_t ri = (size_t)kh * A.cap + slot;
    const size_t ko = A.sko ? (size_t)A.sko[so] * HD : so * A.kstr, kso = A.sko ? (size_t)A.sko[so] : so * A.sstr;
    fp8_store_row(k, A.K + ko + ri * HD, A.KS + kso + ri, red2);
    fp8_store_row(row[(A.nq + A.nkv) * HD + hl * HD + d], A.V + ko + ri * HD, A.VS + kso + ri, red2);
}

// block attention: grid (NQ, DB, NS), 128 threads (a warp per key); keys [lo, anchor + DB) (context < anchor, the
// block rows at anchor..), lo = anchor - 1024 on the local layers. Partials pm / pl [DB][NQ][NS], pa [..][HD].
struct A_dattn {
    const float* Q;
    const uint8_t* K;
    const uint8_t* V;
    const float *KS, *VS;
    int cap, ring, local;
    const int* anchor;
    float *pm, *pl, *pa;
    const int* dsl = nullptr;   // [ink-b-draft] (k_dattnL) [segment] sequence slot
    size_t kstr = 0, sstr = 0;
    const int* sko = nullptr;   // [ink-srv-pool]
    int kv0 = 0;                // [ink-tp-dra] k_dattnL: KV head of block x = kv0 + x
};
__global__ void __launch_bounds__(128) k_dattn(const __grid_constant__ A_dattn A) {
    __shared__ float sm[4], sl[4], sacc[4][HD];
    const int h = blockIdx.x, r = blockIdx.y, z = blockIdx.z, NS = gridDim.z, kvh = h / (NQ / NKV);
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    const int an = *A.anchor;
    const int lo = A.local ? max(0, an - WIN) : 0, hi = an + DB;   // keys [lo, hi)
    const int n = hi - lo, ks = (n + NS - 1) / NS;
    const int k0 = lo + z * ks, k1 = min(hi, k0 + ks);
    const float4 q = *(const float4*)(A.Q + (size_t)r * QD + h * HD + lane * 4);
    const float scale = 1.f / sqrtf((float)HD);
    float m = -INFINITY, l = 0.f, acc[4] = {0.f, 0.f, 0.f, 0.f};
    for (int k = k0 + w; k < k1; k += 4) {
        const int slot = A.ring ? k % A.cap : k;
        const size_t ri = (size_t)kvh * A.cap + slot;
        float kf[4], vf[4];
        fp8x4_load(A.K + ri * HD + lane * 4, A.KS[ri], kf);
        fp8x4_load(A.V + ri * HD + lane * 4, A.VS[ri], vf);
        float d = q.x * kf[0] + q.y * kf[1] + q.z * kf[2] + q.w * kf[3];
        d = warp_sum(d) * scale;
        const float mn = fmaxf(m, d);
        const float cf = expf(m - mn), p = expf(d - mn);
        l = l * cf + p;
        acc[0] = acc[0] * cf + p * vf[0];
        acc[1] = acc[1] * cf + p * vf[1];
        acc[2] = acc[2] * cf + p * vf[2];
        acc[3] = acc[3] * cf + p * vf[3];
        m = mn;
    }
    if (lane == 0) {
        sm[w] = m;
        sl[w] = l;
    }
#pragma unroll
    for (int i = 0; i < 4; i++) sacc[w][lane * 4 + i] = acc[i];
    __syncthreads();
    float M = -INFINITY;
    for (int j = 0; j < 4; j++) M = fmaxf(M, sm[j]);
    const size_t pi = ((size_t)r * NQ + h) * NS + z;
    const int dd = threadIdx.x;
    float L = 0.f, o = 0.f;
    if (M != -INFINITY)
        for (int j = 0; j < 4; j++)
            if (sm[j] != -INFINITY) {
                const float e = expf(sm[j] - M);
                L += sl[j] * e;
                o += sacc[j][dd] * e;
            }
    A.pa[pi * HD + dd] = o;
    if (dd == 0) {
        A.pm[pi] = M;
        A.pl[pi] = L;
    }
}
// [ink-dr-attn-l] lane-per-key block attention (drafter): grid (NKV, NS, DB / 4), 128 threads; warp i = row 4 z + i,
// the 3 GQA heads at once. Scores: lane = key of a 32-key tile, full 128-dim dots against q in smem (no per-key
// shuffles), one warp max / sum per tile; P V: lane = 4 dims, p of key j broadcast by a shuffle. Partials pm / pl /
// pa per (row, head, split) as k_dattn (k_dcomb unchanged). Drafts only (verify keeps the output exact).
__global__ void __launch_bounds__(128) k_dattnL(const __grid_constant__ A_dattn A) {
    constexpr int GQ = NQ / NKV;
    __shared__ __align__(16) float qs[4][GQ][HD];
    const int kvh = A.kv0 + blockIdx.x, z = blockIdx.y, NS = gridDim.y;
    const int lane = threadIdx.x & 31, wi = threadIdx.x >> 5, r = blockIdx.z * 4 + wi;
    const float scale = 1.f / sqrtf((float)HD);
#pragma unroll
    for (int g = 0; g < GQ; g++)
        *(float4*)&qs[wi][g][lane * 4] = *(const float4*)(A.Q + (size_t)r * QD + (kvh * GQ + g) * HD + lane * 4);
    __syncwarp();
    const int an = A.anchor[r / DB];
    const size_t so = A.dsl ? (size_t)A.dsl[r / DB] : 0;
    const size_t ko = A.sko ? (size_t)A.sko[so] * HD : so * A.kstr, kso = A.sko ? (size_t)A.sko[so] : so * A.sstr;
    const uint8_t* const AK = A.K + ko;
    const uint8_t* const AV = A.V + ko;
    const float* const AKS = A.KS + kso;
    const float* const AVS = A.VS + kso;
    const int lo = A.local ? max(0, an - WIN) : 0, hi = an + DB;
    const int n = hi - lo, ks = (n + NS - 1) / NS;
    const int k0 = lo + z * ks, k1 = min(hi, k0 + ks);
    float m[GQ], l[GQ], acc[GQ][4];
#pragma unroll
    for (int g = 0; g < GQ; g++) {
        m[g] = -INFINITY;
        l[g] = 0.f;
        acc[g][0] = acc[g][1] = acc[g][2] = acc[g][3] = 0.f;
    }
    for (int t0 = k0; t0 < k1; t0 += 32) {
        const int k = t0 + lane;
        const bool ok = k < k1;
        float sc[GQ];
#pragma unroll
        for (int g = 0; g < GQ; g++) sc[g] = 0.f;
        if (ok) {
            const size_t ix = (size_t)kvh * A.cap + (A.ring ? k % A.cap : k);
            const uint4* kr = (const uint4*)(AK + ix * HD);
            const float s2 = AKS[ix] * 256.f;
#pragma unroll 2
            for (int c = 0; c < HD / 16; c++) {
                const uint4 u = kr[c];
                const uint32_t uw[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
                for (int e = 0; e < 4; e++) {   // bytes 4e..4e+3 of the 16: elements 16 c + 4 e + {0, 1, 2, 3}
                    const float2 fa = fp8pair(uw[e]), fb = fp8pair(uw[e] >> 8);
                    const float kf[4] = {fa.x * s2, fb.x * s2, fa.y * s2, fb.y * s2};
#pragma unroll
                    for (int g = 0; g < GQ; g++) {
                        const float4 qv = *(const float4*)&qs[wi][g][16 * c + 4 * e];
                        sc[g] += qv.x * kf[0] + qv.y * kf[1] + qv.z * kf[2] + qv.w * kf[3];
                    }
                }
            }
        }
        float p[GQ];
#pragma unroll
        for (int g = 0; g < GQ; g++) {
            const float d = ok ? sc[g] * scale : -INFINITY;
            const float mn = fmaxf(m[g], warp_max(d));
            const float cf = expf(m[g] - mn);
            p[g] = ok ? expf(d - mn) : 0.f;
            l[g] = l[g] * cf + warp_sum(p[g]);
#pragma unroll
            for (int i = 0; i < 4; i++) acc[g][i] *= cf;
            m[g] = mn;
        }
        const int nk = min(32, k1 - t0);
        // V of the tile's keys loaded up front (lane: 4 dims of every key; its own key's scale), then folded in key order
        const float vsl = ok ? AVS[(size_t)kvh * A.cap + (A.ring ? k % A.cap : k)] : 0.f;
        uint32_t vw[32];
#pragma unroll
        for (int j = 0; j < 32; j++) {
            const int kj = t0 + j;
            vw[j] = j < nk ? *(const uint32_t*)(AV + ((size_t)kvh * A.cap + (A.ring ? kj % A.cap : kj)) * HD + lane * 4) : 0u;
        }
#pragma unroll
        for (int j = 0; j < 32; j++) {
            const float s2 = __shfl_sync(0xffffffffu, vsl, j) * 256.f;
            const float2 fa = fp8pair(vw[j]), fb = fp8pair(vw[j] >> 8);
            const float vf[4] = {fa.x * s2, fb.x * s2, fa.y * s2, fb.y * s2};
#pragma unroll
            for (int g = 0; g < GQ; g++) {
                const float pj = __shfl_sync(0xffffffffu, p[g], j);   // 0 for keys past the split
#pragma unroll
                for (int i = 0; i < 4; i++) acc[g][i] += pj * vf[i];
            }
        }
    }
#pragma unroll
    for (int g = 0; g < GQ; g++) {
        const size_t pi = ((size_t)r * NQ + kvh * GQ + g) * NS + z;
        *(float4*)(A.pa + pi * HD + lane * 4) = make_float4(acc[g][0], acc[g][1], acc[g][2], acc[g][3]);
        if (lane == 0) {
            A.pm[pi] = m[g];
            A.pl[pi] = l[g];
        }
    }
}
// merge: grid (NQ, DB), 128 threads -> AO bf16 [DB][QD]
__global__ void __launch_bounds__(128) k_dcomb(const float* pm, const float* pl, const float* pa, int NS, bf16* AO, int h0 = 0) {
    const int h = h0 + blockIdx.x, r = blockIdx.y, d = threadIdx.x;   // [ink-tp-dra] h0: the rank's first q head
    const size_t b0 = ((size_t)r * NQ + h) * NS;
    float M = -INFINITY;
    for (int z = 0; z < NS; z++) M = fmaxf(M, pm[b0 + z]);
    float L = 0.f, o = 0.f;
    for (int z = 0; z < NS; z++)
        if (pm[b0 + z] != -INFINITY) {
            const float e = expf(pm[b0 + z] - M);
            L += pl[b0 + z] * e;
            o += pa[(b0 + z) * HD + d] * e;
        }
    AO[(size_t)r * QD + h * HD + d] = __float2bfloat16(o / L);
}

// SwiGLU: A[r][i] = silu(GU[r][i]) * GU[r][DFF + i] (bf16), DB * DFF threads
__global__ void k_dswiglu(const float* GU, bf16* A, int nr = DB, int nf = DFF) {   // [ink-tp-drv2b] nf: the rank's columns
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nr * nf) return;
    const int r = idx / nf, i = idx % nf;
    const float g = GU[(size_t)r * 2 * nf + i], u = GU[(size_t)r * 2 * nf + nf + i];
    A[idx] = __float2bfloat16(g / (1.f + expf(-g)) * u);
}

// [ink-tp-drv2b] residual += the summed down rows (float4)
__global__ void k_addrows(float4* y, const float4* x, int n4) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n4) return;
    float4 a = y[i];
    const float4 b = x[i];
    a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w;
    y[i] = a;
}
// block init: rows 1..7 = mask_embed (row 0 is written by k_embnorm)
__global__ void k_dmask(const bf16* me, float* h) {   // grid.y = segment (rows 8 y + 1 .. 8 y + 7)
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < (DB - 1) * H) h[(size_t)blockIdx.y * DB * H + H + idx] = __bfloat162float(me[idx % H]);
}
// [ink-b-draft] f_3 of each segment's last accepted row -> its slot's FM; gather of the slots' FM rows for a block
__global__ void k_fmcommit(const bf16* F3, const int* rt, const int* acc, bf16* FMS) {
    const int z = blockIdx.x, r = rt[RT_FIRST + z] + acc[z], sl = rt[RT_SLOT + r];
    for (int c = threadIdx.x; c < H; c += blockDim.x) FMS[(size_t)sl * H + c] = F3[(size_t)r * H + c];
}
__global__ void k_fmgather(const bf16* FMS, const int* dsl, bf16* out) {
    const int z = blockIdx.x;
    for (int c = threadIdx.x; c < H; c += blockDim.x) out[(size_t)z * H + c] = FMS[(size_t)dsl[z] * H + c];
}

// Markov step j: grid MR / 8 blocks, 256 threads; warp k -> z rows k (g), MR + k (c), 2 MR + k (o) of
// Pse [3 MR][2 MR] . [s_j, W1[prev]] + Zpre[j]; s_{j+1}[k] = g s_j + (1 - g) tanh(c), ro_j[k] = tanh(o)
struct A_mkz {
    const bf16* Pse;
    const float* Zpre;   // [7][3 MR]
    const bf16* W1;      // [VPAD][MR]
    const int* toks;     // [8]: anchor, d1 .. ; prev = toks[j]
    float* S;            // [8][MR], S[0] = 0
    float* RO;           // [7][MR]
    int j;
    const float* stop;   // [tau, kmin]: skip this step once the block length is decided (prob[j + 1] = 0)
    float* prob;
};
// the confidence stop decided the block before draft j + 1: running product of prob[1..j] < tau and j + 1 > kmin - 1
__device__ __forceinline__ bool mk_skip(const float* stop, const float* prob, int j) {
    if (!stop || j + 1 <= (int)stop[1] - 1) return false;
    float p = 1.f;
    for (int i = 1; i <= j; i++) p *= prob[i];
    return p < stop[0];
}
__device__ __forceinline__ void b_mkz(const A_mkz& A);
__global__ void __launch_bounds__(256) k_mkz(const __grid_constant__ A_mkz A) { b_mkz(A); }
// [ink-b-draft] Markov step of segment blockIdx.z (per-segment state at fixed strides)
__global__ void __launch_bounds__(256) k_mkzZ(const __grid_constant__ A_mkz A) {
    const int z = blockIdx.z;
    A_mkz a = A;
    a.Zpre += (size_t)z * (DB - 1) * 3 * MR;
    a.toks += z * DB;
    a.S += (size_t)z * DB * MR;
    a.RO += (size_t)z * (DB - 1) * MR;
    a.prob += z * 16;
    b_mkz(a);
}
__device__ __forceinline__ void b_mkz(const A_mkz& A) {
    __shared__ float inp[2 * MR];
    const int j = A.j;
    if (mk_skip(A.stop, A.prob, j)) {
        if (blockIdx.x == 0 && threadIdx.x == 0) A.prob[j + 1] = 0.f;
        return;
    }
    const int prev = A.toks[j];
    for (int c = threadIdx.x; c < 2 * MR; c += 256)
        inp[c] = c < MR ? A.S[j * MR + c] : __bfloat162float(A.W1[(size_t)prev * MR + c - MR]);
    __syncthreads();
    const int lane = threadIdx.x & 31, k = blockIdx.x * 8 + (threadIdx.x >> 5);
    float z[3];
#pragma unroll
    for (int g = 0; g < 3; g++) {
        const bf16* row = A.Pse + (size_t)(g * MR + k) * 2 * MR;
        float s = 0.f;
#pragma unroll
        for (int c = lane * 8; c < 2 * MR; c += 256) {
            const uint4 wv = *(const uint4*)(row + c);
            const uint32_t ww[4] = {wv.x, wv.y, wv.z, wv.w};
#pragma unroll
            for (int q = 0; q < 4; q++)
                s += __uint_as_float(ww[q] << 16) * inp[c + 2 * q] + __uint_as_float(ww[q] & 0xffff0000u) * inp[c + 2 * q + 1];
        }
        z[g] = warp_sum(s) + A.Zpre[j * 3 * MR + g * MR + k];
    }
    if (lane == 0) {
        const float gt = 1.f / (1.f + expf(-z[0]));
        A.S[(j + 1) * MR + k] = gt * inp[k] + (1.f - gt) * tanhf(z[1]);
        A.RO[j * MR + k] = tanhf(z[2]);
    }
}

// Markov logits of step j: l[v] = base[j][v] + UO[v] . ro_j + uOb[v]; argmax + sum exp -> toks[j + 1], prob[j + 1].
// grid NBL blocks of 64 rows, 256 threads (warp: 8 rows); block partials, the last block merges.
constexpr int MKL_ROWS = 64;
struct A_mkl {
    const float* base;   // [7][V] (row stride V)
    const __half* UO;    // [V][MR]
    const float* uOb;
    const float* RO;
    float* pmx;          // [NBL] block max
    float* psx;          // [NBL] sum exp (block max)
    int* pix;            // [NBL] argmax
    unsigned* cnt;
    int* toks;
    float* prob;
    int j;
    const float* stop;
};
__global__ void __launch_bounds__(256) k_mkl(const __grid_constant__ A_mkl A) {
    if (mk_skip(A.stop, A.prob, A.j)) return;
    __shared__ float ro[MR];
    __shared__ float wm[8], ws[8];
    __shared__ int wi[8];
    __shared__ bool last;
    const int j = A.j;
    ro[threadIdx.x] = A.RO[j * MR + threadIdx.x];   // 256 threads == MR
    __syncthreads();
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    float rv[8];
#pragma unroll
    for (int q = 0; q < 8; q++) rv[q] = ro[lane * 8 + q];
    float m = -INFINITY, s = 0.f;
    int mi = 0x7fffffff;
    const float* bj = A.base + (size_t)j * V;
    for (int i = 0; i < MKL_ROWS / 8; i++) {
        const int v = blockIdx.x * MKL_ROWS + w * (MKL_ROWS / 8) + i;
        if (v >= V) break;
        const uint4 u = *(const uint4*)(A.UO + (size_t)v * MR + lane * 8);
        const __half2* hp = (const __half2*)&u;
        float d = 0.f;
#pragma unroll
        for (int q = 0; q < 4; q++) {
            const float2 f = __half22float2(hp[q]);
            d += f.x * rv[2 * q] + f.y * rv[2 * q + 1];
        }
        d = warp_sum(d);
        const float l = bj[v] + d + A.uOb[v];
        if (l > m) {
            s = s * expf(m - l) + 1.f;
            m = l;
            mi = v;
        } else
            s += expf(l - m);
    }
    if (lane == 0) {
        wm[w] = m;
        ws[w] = s;
        wi[w] = mi;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float M = -INFINITY, S = 0.f;
        int I = 0x7fffffff;
        for (int q = 0; q < 8; q++)
            if (wm[q] > M || (wm[q] == M && wi[q] < I)) {
                M = wm[q];
                I = wi[q];
            }
        for (int q = 0; q < 8; q++)
            if (wm[q] != -INFINITY) S += ws[q] * expf(wm[q] - M);
        A.pmx[blockIdx.x] = M;
        A.psx[blockIdx.x] = S;
        A.pix[blockIdx.x] = I;
        __threadfence();
        last = atomicAdd(A.cnt, 1u) == gridDim.x - 1;
    }
    __syncthreads();
    if (!last) return;
    __threadfence();
    float M = -INFINITY;
    int I = 0x7fffffff;
    for (int b = threadIdx.x; b < (int)gridDim.x; b += 256) {
        const float x = __ldcg(A.pmx + b);
        const int ii = __ldcg(A.pix + b);
        if (x > M || (x == M && ii < I)) {
            M = x;
            I = ii;
        }
    }
    __shared__ float bm[256];
    __shared__ int bi[256];
    bm[threadIdx.x] = M;
    bi[threadIdx.x] = I;
    __syncthreads();
    for (int o = 128; o; o >>= 1) {
        if (threadIdx.x < o) {
            const float x = bm[threadIdx.x + o];
            const int ii = bi[threadIdx.x + o];
            if (x > bm[threadIdx.x] || (x == bm[threadIdx.x] && ii < bi[threadIdx.x])) {
                bm[threadIdx.x] = x;
                bi[threadIdx.x] = ii;
            }
        }
        __syncthreads();
    }
    const float GM = bm[0];
    float S = 0.f;
    for (int b = threadIdx.x; b < (int)gridDim.x; b += 256) {
        const float x = __ldcg(A.pmx + b);
        if (x != -INFINITY) S += __ldcg(A.psx + b) * expf(x - GM);
    }
    __shared__ float red[33];
    float sv[1] = {S};
    block_sum<1>(sv, red);
    if (threadIdx.x == 0) {
        const bool okt = bi[0] >= 0 && bi[0] < V;   // review L1: all-NaN logits -> token 0, prob 0
        A.toks[j + 1] = okt ? bi[0] : 0;
        A.prob[j + 1] = okt ? 1.f / sv[0] : 0.f;
        *A.cnt = 0u;
    }
}

// Zpre[j] = PH h_j + PF f + b (PH h and PF f from cuBLAS): Z [7][3 MR] += F1 [3 MR] + b
__global__ void k_zpre(float* Z, const float* Fz, const bf16* b, int nseg = 1) {   // Fz [segment][3 MR]
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < nseg * (DB - 1) * 3 * MR) Z[idx] += Fz[(idx / ((DB - 1) * 3 * MR)) * 3 * MR + idx % (3 * MR)] + __bfloat162float(b[idx % (3 * MR)]);
}
// copy one bf16 row (index from the device) -> dst
__global__ void k_rowcopy(const bf16* src, const int* rowp, int off, bf16* dst) {
    const int r = *rowp + off;
    for (int c = threadIdx.x; c < H; c += blockDim.x) dst[c] = src[(size_t)r * H + c];
}

// approximate base logits from the head's 8-bit screen plane for nr <= R rows of x (bf16, stride H): the
// k_head_screen layout (group of 2 vocab rows per warp), score only. y row j at y + j * ldy.
template <int R>
__global__ void __launch_bounds__(256) k_screenR(const uint8_t* scr, const uint8_t* seb, const bf16* x, int nr, float* y, int ldy) {
    constexpr int NCH = 2 * HEAD_K / 512;
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    const int g = blockIdx.x * 8 + w;
    if (2 * g >= V) return;
    uint4 wv[NCH];
    const uint8_t* base = scr + (size_t)g * 2 * HEAD_K;
#pragma unroll
    for (int c = 0; c < NCH; c++) wv[c] = *(const uint4*)(base + (c * 32 + lane) * 16);
    const uint32_t ebw = *(const uint16_t*)(seb + 2 * g);
    const uint32_t e0 = ebw & 0xff, e1 = ebw >> 8;
    const uint32_t EB0 = (e0 << 7) | (e0 << 23), EB1 = (e1 << 7) | (e1 << 23);
    float s0[R], s1[R];
#pragma unroll
    for (int j = 0; j < R; j++) s0[j] = s1[j] = 0.f;
#pragma unroll
    for (int c = 0; c < NCH; c++) {
        const int jj = c * 32 + lane;
        const bool r1 = jj >= 144;
        const int xi = (r1 ? jj - 144 : jj) * 16;
        const uint32_t EB = r1 ? EB1 : EB0;
        const uint32_t ww[4] = {wv[c].x, wv[c].y, wv[c].z, wv[c].w};
        float wf[16];
#pragma unroll
        for (int q = 0; q < 4; q++) {
            const uint32_t p0 = scr_pair<0>(ww[q], EB), p1 = scr_pair<1>(ww[q], EB);
            wf[4 * q + 0] = __uint_as_float(p0 << 16);
            wf[4 * q + 1] = __uint_as_float(p0 & 0xffff0000u);
            wf[4 * q + 2] = __uint_as_float(p1 << 16);
            wf[4 * q + 3] = __uint_as_float(p1 & 0xffff0000u);
        }
#pragma unroll
        for (int j = 0; j < R; j++) {
            if (j >= nr) break;
            const uint4 xa = *(const uint4*)(x + (size_t)j * H + xi), xb = *(const uint4*)(x + (size_t)j * H + xi + 8);
            const uint32_t xx[8] = {xa.x, xa.y, xa.z, xa.w, xb.x, xb.y, xb.z, xb.w};
            float s = 0.f;
#pragma unroll
            for (int q = 0; q < 8; q++) s += wf[2 * q] * __uint_as_float(xx[q] << 16) + wf[2 * q + 1] * __uint_as_float(xx[q] & 0xffff0000u);
            if (r1) s1[j] += s;
            else s0[j] += s;
        }
    }
#pragma unroll
    for (int j = 0; j < R; j++) {
        if (j >= nr) break;
        const float a = warp_sum(s0[j]), b = warp_sum(s1[j]);
        if (lane == 0) {
            y[(size_t)j * ldy + 2 * g] = a;
            if (2 * g + 1 < V) y[(size_t)j * ldy + 2 * g + 1] = b;
        }
    }
}

// Markov logits with UO in int8 (per-row scale): as k_mkl
struct A_mkl8 {
    A_mkl a;
    const int8_t* UO8;   // [V][MR]
    const float* us;     // [V]
};
__global__ void __launch_bounds__(256) k_mkl8(const __grid_constant__ A_mkl8 A8) {
    const A_mkl& A = A8.a;
    if (mk_skip(A.stop, A.prob, A.j)) return;   // k_mkz of this step wrote prob[j + 1] = 0 (stream order)
    __shared__ float ro[MR];
    __shared__ float wm[8], ws[8];
    __shared__ int wi[8];
    __shared__ bool last;
    const int j = A.j;
    ro[threadIdx.x] = A.RO[j * MR + threadIdx.x];
    __syncthreads();
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    float rv[8];
#pragma unroll
    for (int q = 0; q < 8; q++) rv[q] = ro[lane * 8 + q];
    float m = -INFINITY, s = 0.f;
    int mi = 0x7fffffff;
    const float* bj = A.base + (size_t)j * V;
    for (int i = 0; i < MKL_ROWS / 8; i++) {
        const int v = blockIdx.x * MKL_ROWS + w * (MKL_ROWS / 8) + i;
        if (v >= V) break;
        const uint2 u = *(const uint2*)(A8.UO8 + (size_t)v * MR + lane * 8);
        const uint32_t uw[2] = {u.x, u.y};
        float d = 0.f;
#pragma unroll
        for (int q = 0; q < 8; q++) d += (float)(int8_t)(uw[q >> 2] >> (8 * (q & 3))) * rv[q];
        d = warp_sum(d);
        const float l = bj[v] + d * A8.us[v] + A.uOb[v];
        if (l > m) {
            s = s * expf(m - l) + 1.f;
            m = l;
            mi = v;
        } else
            s += expf(l - m);
    }
    if (lane == 0) {
        wm[w] = m;
        ws[w] = s;
        wi[w] = mi;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float M = -INFINITY, S = 0.f;
        int I = 0x7fffffff;
        for (int q = 0; q < 8; q++)
            if (wm[q] > M || (wm[q] == M && wi[q] < I)) {
                M = wm[q];
                I = wi[q];
            }
        for (int q = 0; q < 8; q++)
            if (wm[q] != -INFINITY) S += ws[q] * expf(wm[q] - M);
        A.pmx[blockIdx.x] = M;
        A.psx[blockIdx.x] = S;
        A.pix[blockIdx.x] = I;
        __threadfence();
        last = atomicAdd(A.cnt, 1u) == gridDim.x - 1;
    }
    __syncthreads();
    if (!last) return;
    __threadfence();
    float M = -INFINITY;
    int I = 0x7fffffff;
    for (int b = threadIdx.x; b < (int)gridDim.x; b += 256) {
        const float x = __ldcg(A.pmx + b);
        const int ii = __ldcg(A.pix + b);
        if (x > M || (x == M && ii < I)) {
            M = x;
            I = ii;
        }
    }
    __shared__ float bm[256];
    __shared__ int bi[256];
    bm[threadIdx.x] = M;
    bi[threadIdx.x] = I;
    __syncthreads();
    for (int o = 128; o; o >>= 1) {
        if (threadIdx.x < o) {
            const float x = bm[threadIdx.x + o];
            const int ii = bi[threadIdx.x + o];
            if (x > bm[threadIdx.x] || (x == bm[threadIdx.x] && ii < bi[threadIdx.x])) {
                bm[threadIdx.x] = x;
                bi[threadIdx.x] = ii;
            }
        }
        __syncthreads();
    }
    const float GM = bm[0];
    float S = 0.f;
    for (int b = threadIdx.x; b < (int)gridDim.x; b += 256) {
        const float x = __ldcg(A.pmx + b);
        if (x != -INFINITY) S += __ldcg(A.psx + b) * expf(x - GM);
    }
    __shared__ float red[33];
    float sv[1] = {S};
    block_sum<1>(sv, red);
    if (threadIdx.x == 0) {
        const bool okt = bi[0] >= 0 && bi[0] < V;   // review L1: all-NaN logits -> token 0, prob 0
        A.toks[j + 1] = okt ? bi[0] : 0;
        A.prob[j + 1] = okt ? 1.f / sv[0] : 0.f;
        *A.cnt = 0u;
    }
}

// exact two-stage head for nr <= R verify rows: the screen of k_head_screen with the plane read once for all rows
// (per-row arithmetic unchanged); row j: x + j * H, upper ends hi + j * V, threshold thr[j * 16] (one 64 B counter
// set per row: thr, done, ncand, best as in HeadArgs)
template <int R>
__global__ void __launch_bounds__(256) k_head_screenR(const __grid_constant__ HeadArgs h, int nr) {
    constexpr int NCH = 2 * HEAD_K / 512;
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    const int g = blockIdx.x * 8 + w;
    const bool live = 2 * g < h.N;
    uint4 wv[NCH];
    uint32_t ebw = 0;
    if (live) {
        const uint8_t* base = h.scr + (size_t)g * 2 * HEAD_K;
#pragma unroll
        for (int c = 0; c < NCH; c++) wv[c] = *(const uint4*)(base + (c * 32 + lane) * 16);
        ebw = *(const uint16_t*)(h.seb + 2 * g);
    } else
#pragma unroll
        for (int c = 0; c < NCH; c++) wv[c] = make_uint4(0, 0, 0, 0);
    const uint32_t e0 = ebw & 0xff, e1 = ebw >> 8;
    const uint32_t EB0 = (e0 << 7) | (e0 << 23), EB1 = (e1 << 7) | (e1 << 23);
    float s0[R], a0[R], s1[R], a1[R], x1[R];
#pragma unroll
    for (int j = 0; j < R; j++) s0[j] = a0[j] = s1[j] = a1[j] = x1[j] = 0.f;
#pragma unroll
    for (int c = 0; c < NCH; c++) {
        const int jj = c * 32 + lane;
        const bool r1 = jj >= 144;
        const int xi = (jj < 144 ? jj : jj - 144) * 16;
        const uint32_t EB = r1 ? EB1 : EB0;
        const uint32_t ww[4] = {wv[c].x, wv[c].y, wv[c].z, wv[c].w};
        uint32_t pw[8];
#pragma unroll
        for (int q = 0; q < 4; q++) {
            pw[2 * q] = scr_pair<0>(ww[q], EB);
            pw[2 * q + 1] = scr_pair<1>(ww[q], EB);
        }
#pragma unroll
        for (int j = 0; j < R; j++) {
            if (j >= nr) break;
            const uint4 xa = *(const uint4*)(h.x + (size_t)j * H + xi), xb = *(const uint4*)(h.x + (size_t)j * H + xi + 8);
            const uint32_t xx[8] = {xa.x, xa.y, xa.z, xa.w, xb.x, xb.y, xb.z, xb.w};
            float s = 0.f, a = 0.f, xs = 0.f;
#pragma unroll
            for (int q = 0; q < 4; q++) {
#pragma unroll
                for (int k = 0; k < 2; k++) {
                    const uint32_t p = pw[2 * q + k];
                    const uint32_t xw = xx[2 * q + k];
                    const float wl = __uint_as_float(p << 16), wh = __uint_as_float(p & 0xffff0000u);
                    const float xl = __uint_as_float(xw << 16), xh = __uint_as_float(xw & 0xffff0000u);
                    s = fmaf(wl, xl, s);
                    s = fmaf(wh, xh, s);
                    a = fmaf(fabsf(wl), fabsf(xl), a);
                    a = fmaf(fabsf(wh), fabsf(xh), a);
                    xs += fabsf(xl) + fabsf(xh);
                }
            }
            if (r1) { s1[j] += s; a1[j] += a; }
            else { s0[j] += s; a0[j] += a; x1[j] += xs; }
        }
    }
    __shared__ unsigned bl[R][8];
    constexpr float EPS = 1.f / 16 + 1.f / 1024, XM = 1.f + 1.f / 1024;
#pragma unroll
    for (int j = 0; j < R; j++) {
        if (j >= nr) break;
        const float S0 = warp_sum(s0[j]), A0 = warp_sum(a0[j]), S1 = warp_sum(s1[j]), A1 = warp_sum(a1[j]), X1 = warp_sum(x1[j]);
        const float B0 = A0 * EPS + __uint_as_float(e0 << 23) * X1 * XM;
        const float B1 = A1 * EPS + __uint_as_float(e1 << 23) * X1 * XM;
        float lo0 = S0 - B0, hi0 = S0 + B0, lo1 = S1 - B1, hi1 = S1 + B1;
        float iT;
        uint32_t sd;
        int st;
        if (row_samp(h, j, iT, sd, st)) {   // [ink-srv-samp] the certified Gumbel bounds of a sampled row
            const float g0 = gumbel(sd, st, h.r0 + 2 * g), g1 = gumbel(sd, st, h.r0 + 2 * g + 1);
            auto tr = [&](float z, float gg, float sgn) {
                const float v = head_cap(z) * iT + gg;
                return v + sgn * (1e-5f * (30.f * iT + fabsf(gg)) + 1e-6f);
            };
            lo0 = tr(lo0, g0, -1.f); hi0 = tr(hi0, g0, 1.f); lo1 = tr(lo1, g1, -1.f); hi1 = tr(hi1, g1, 1.f);
        }
        if (lane == 0) {
            unsigned m = 0;
            if (live) {
                const bool fl = h.lo && row_filt(h, j);   // [ink-samp-dev] the raw lower ends of a filtered row
                h.hi[(size_t)j * h.N + 2 * g] = hi0;
                if (fl) h.lo[(size_t)j * h.N + 2 * g] = lo0;
                m = f2ord(lo0);
                if (2 * g + 1 < h.N) {
                    h.hi[(size_t)j * h.N + 2 * g + 1] = hi1;
                    if (fl) h.lo[(size_t)j * h.N + 2 * g + 1] = lo1;
                    m = max(m, f2ord(lo1));
                }
            }
            bl[j][w] = m;
        }
    }
    __syncthreads();
    if (threadIdx.x < nr) {
        const int j = threadIdx.x;
        unsigned m = bl[j][0];
        for (int q = 1; q < 8; q++) m = max(m, bl[j][q]);
        atomicMax(h.thr + j * 16, m);
    }
}

// k_screenR with the rows' x staged once per block in shared memory and a grid-stride loop over the groups
template <int R>
__global__ void __launch_bounds__(256) k_screenS(const uint8_t* scr, const uint8_t* seb, const bf16* x, int nr, float* y, int ldy) {
    constexpr int NCH = 2 * HEAD_K / 512;
    __shared__ __align__(16) bf16 xs[R * H];
    for (int i = threadIdx.x; i < nr * H / 8; i += 256) ((uint4*)xs)[i] = ((const uint4*)x)[i];
    __syncthreads();
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    for (int g = blockIdx.x * 8 + w; 2 * g < V; g += gridDim.x * 8) {
        uint4 wv[NCH];
        const uint8_t* base = scr + (size_t)g * 2 * HEAD_K;
#pragma unroll
        for (int c = 0; c < NCH; c++) wv[c] = *(const uint4*)(base + (c * 32 + lane) * 16);
        const uint32_t ebw = *(const uint16_t*)(seb + 2 * g);
        const uint32_t e0 = ebw & 0xff, e1 = ebw >> 8;
        const uint32_t EB0 = (e0 << 7) | (e0 << 23), EB1 = (e1 << 7) | (e1 << 23);
        float s0[R], s1[R];
#pragma unroll
        for (int j = 0; j < R; j++) s0[j] = s1[j] = 0.f;
#pragma unroll
        for (int c = 0; c < NCH; c++) {
            const int jj = c * 32 + lane;
            const bool r1 = jj >= 144;
            const int xi = (r1 ? jj - 144 : jj) * 16;
            const uint32_t EB = r1 ? EB1 : EB0;
            const uint32_t ww[4] = {wv[c].x, wv[c].y, wv[c].z, wv[c].w};
            float wf[16];
#pragma unroll
            for (int q = 0; q < 4; q++) {
                const uint32_t p0 = scr_pair<0>(ww[q], EB), p1 = scr_pair<1>(ww[q], EB);
                wf[4 * q + 0] = __uint_as_float(p0 << 16);
                wf[4 * q + 1] = __uint_as_float(p0 & 0xffff0000u);
                wf[4 * q + 2] = __uint_as_float(p1 << 16);
                wf[4 * q + 3] = __uint_as_float(p1 & 0xffff0000u);
            }
#pragma unroll
            for (int j = 0; j < R; j++) {
                if (j >= nr) break;
                const uint4 xa = *(const uint4*)(xs + j * H + xi), xb = *(const uint4*)(xs + j * H + xi + 8);
                const uint32_t xx[8] = {xa.x, xa.y, xa.z, xa.w, xb.x, xb.y, xb.z, xb.w};
                float s = 0.f;
#pragma unroll
                for (int q = 0; q < 8; q++) s += wf[2 * q] * __uint_as_float(xx[q] << 16) + wf[2 * q + 1] * __uint_as_float(xx[q] & 0xffff0000u);
                if (r1) s1[j] += s;
                else s0[j] += s;
            }
        }
#pragma unroll
        for (int j = 0; j < R; j++) {
            if (j >= nr) break;
            const float a = warp_sum(s0[j]), b = warp_sum(s1[j]);
            if (lane == 0) {
                y[(size_t)j * ldy + 2 * g] = a;
                if (2 * g + 1 < V) y[(size_t)j * ldy + 2 * g + 1] = b;
            }
        }
    }
}

// ---- [ink-dr-topc] Markov term over candidates only (drafts need no exact logits; verify decides): per 64-token
// chunk of base + uOb, the top 2 tokens are candidates, the rest folds into a tail (max, sum exp) with the Markov term
// taken as 0. k_mksel once per block for the 7 rows, then per position k_mkl8C over the candidates.
#ifndef MKC_K
#define MKC_K 4
#endif
constexpr int MKC_CH = 64, MKC_NCH = (V + MKC_CH - 1) / MKC_CH, MKC_NC = MKC_K * MKC_NCH;
// [ink-tp-drv2] c0 / c1: the rank's 64-token chunks (the chunks' arithmetic does not depend on the range)
__global__ void __launch_bounds__(256) k_mksel(const float* base, const float* uOb, int* cand, float* tm, float* ts, int c0 = 0, int c1 = MKC_NCH) {
    const int lane = threadIdx.x & 31, ch = c0 + blockIdx.x * 8 + (threadIdx.x >> 5), j = blockIdx.y;
    if (ch >= c1) return;
    const float* bj = base + (size_t)j * V;
    const int v0 = ch * MKC_CH + lane, v1 = v0 + 32;
    float x0 = v0 < V ? bj[v0] + uOb[v0] : -INFINITY, x1 = v1 < V ? bj[v1] + uOb[v1] : -INFINITY;
    int sel[MKC_K];
#pragma unroll
    for (int r = 0; r < MKC_K; r++) {   // warp argmax (ties: lower token), MKC_K times
        float m = x0 >= x1 ? x0 : x1;
        int mi = x0 >= x1 ? v0 : v1;
        for (int o = 16; o; o >>= 1) {
            const float om = __shfl_xor_sync(0xffffffffu, m, o);
            const int oi = __shfl_xor_sync(0xffffffffu, mi, o);
            if (om > m || (om == m && oi < mi)) { m = om; mi = oi; }
        }
        sel[r] = m == -INFINITY ? -1 : mi;
        if (mi == v0) x0 = -INFINITY;
        if (mi == v1) x1 = -INFINITY;
    }
    float m = fmaxf(x0, x1);
    for (int o = 16; o; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
    float s = (m == -INFINITY) ? 0.f : (x0 == -INFINITY ? 0.f : expf(x0 - m)) + (x1 == -INFINITY ? 0.f : expf(x1 - m));
    s = warp_sum(s);
    if (lane == 0) {
#pragma unroll
        for (int r = 0; r < MKC_K; r++) cand[(size_t)j * MKC_NC + MKC_K * ch + r] = sel[r];
        tm[(size_t)j * MKC_NCH + ch] = m;
        ts[(size_t)j * MKC_NCH + ch] = s;
    }
}
struct A_mkl8C {
    A_mkl a;
    const int8_t* UO8;
    const float* us;
    const int* cand;
    const float* tm;
    const float* ts;
    float* pax;   // [nbl] argmax value per block (pmx / psx: the softmax normalizer)
    // [ink-tp-drv2] TP vocabulary split: this rank's candidates [p0, p1) and tail chunks [c0, c1); gat != null: the last
    // block writes (argmax value, token, max, sum) to gat[grank * 4 ..] (zero elsewhere) instead of the token / prob,
    // the ranks' rows are summed (= gathered) and k_mkmerge picks the token on every rank
    int p0 = 0, p1 = MKC_NC, c0 = 0, c1 = MKC_NCH;
    float* gat = nullptr;
    int gN = 1, grank = 0;
};
constexpr int MKC_BL = 64;   // candidates per block
__device__ __forceinline__ void b_mkl8C(const A_mkl8C& A8);
__global__ void __launch_bounds__(256) k_mkl8C(const __grid_constant__ A_mkl8C A8) { b_mkl8C(A8); }
// [ink-b-draft] segment blockIdx.z: per-segment state at fixed strides (nblk = merge blocks per segment)
__global__ void __launch_bounds__(256) k_mkl8CZ(const __grid_constant__ A_mkl8C A8, int nblk) {
    const int z = blockIdx.z;
    A_mkl8C b = A8;
    b.a.base += (size_t)z * (DB - 1) * V;
    b.a.RO += (size_t)z * (DB - 1) * MR;
    b.a.pmx += (size_t)z * nblk;
    b.a.psx += (size_t)z * nblk;
    b.a.pix += (size_t)z * nblk;
    b.a.cnt += z;
    b.a.toks += z * DB;
    b.a.prob += z * 16;
    b.cand += (size_t)z * (DB - 1) * MKC_NC;
    b.tm += (size_t)z * (DB - 1) * MKC_NCH;
    b.ts += (size_t)z * (DB - 1) * MKC_NCH;
    b.pax += (size_t)z * nblk;
    if (b.gat) b.gat += (size_t)z * b.gN * 4;
    b_mkl8C(b);
}
__device__ __forceinline__ void b_mkl8C(const A_mkl8C& A8) {
    const A_mkl& A = A8.a;
    if (mk_skip(A.stop, A.prob, A.j)) return;
    __shared__ float ro[MR];
    __shared__ float wm[8], ws[8], wa[8];
    __shared__ int wi[8];
    __shared__ bool last;
    const int j = A.j;
    ro[threadIdx.x] = A.RO[j * MR + threadIdx.x];
    __syncthreads();
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    float rv[8];
#pragma unroll
    for (int q = 0; q < 8; q++) rv[q] = ro[lane * 8 + q];
    float m = -INFINITY, s = 0.f, am = -INFINITY;
    int mi = 0x7fffffff;
    const float* bj = A.base + (size_t)j * V;
    const int* cj = A8.cand + (size_t)j * MKC_NC;
    for (int i = 0; i < MKC_BL / 8; i++) {
        const int p = A8.p0 + blockIdx.x * MKC_BL + w * (MKC_BL / 8) + i;
        if (p >= A8.p1) break;
        const int v = cj[p];
        if (v < 0) continue;
        const uint2 u = *(const uint2*)(A8.UO8 + (size_t)v * MR + lane * 8);
        const uint32_t uw[2] = {u.x, u.y};
        float d = 0.f;
#pragma unroll
        for (int q = 0; q < 8; q++) d += (float)(int8_t)(uw[q >> 2] >> (8 * (q & 3))) * rv[q];
        d = warp_sum(d);
        const float l = bj[v] + d * A8.us[v] + A.uOb[v];
        if (l > am || (l == am && v < mi)) { am = l; mi = v; }
        if (l > m) { s = s * expf(m - l) + 1.f; m = l; }
        else s += expf(l - m);
    }
    if (lane == 0) { wm[w] = m; ws[w] = s; wa[w] = am; wi[w] = mi; }
    __syncthreads();
    if (threadIdx.x == 0) {
        float M = -INFINITY, S = 0.f, AM = -INFINITY;
        int I = 0x7fffffff;
        for (int q = 0; q < 8; q++) {
            if (wa[q] > AM || (wa[q] == AM && wi[q] < I)) { AM = wa[q]; I = wi[q]; }
            M = fmaxf(M, wm[q]);
        }
        for (int q = 0; q < 8; q++)
            if (wm[q] != -INFINITY) S += ws[q] * expf(wm[q] - M);
        A.pmx[blockIdx.x] = M;
        A.psx[blockIdx.x] = S;
        A8.pax[blockIdx.x] = AM;
        A.pix[blockIdx.x] = I;
        __threadfence();
        last = atomicAdd(A.cnt, 1u) == gridDim.x - 1;
    }
    __syncthreads();
    if (!last) return;
    __threadfence();
    __shared__ float bm[256];
    __shared__ int bi[256];
    __shared__ float bx[256];
    float AM = -INFINITY, M = -INFINITY;
    int I = 0x7fffffff;
    for (int b = threadIdx.x; b < (int)gridDim.x; b += 256) {
        const float x = __ldcg(A8.pax + b);
        const int ii = __ldcg(A.pix + b);
        if (x > AM || (x == AM && ii < I)) { AM = x; I = ii; }
        M = fmaxf(M, __ldcg(A.pmx + b));
    }
    for (int c = A8.c0 + threadIdx.x; c < A8.c1; c += 256) M = fmaxf(M, A8.tm[(size_t)j * MKC_NCH + c]);
    bm[threadIdx.x] = AM;
    bi[threadIdx.x] = I;
    bx[threadIdx.x] = M;
    __syncthreads();
    for (int o = 128; o; o >>= 1) {
        if (threadIdx.x < o) {
            const float x = bm[threadIdx.x + o];
            const int ii = bi[threadIdx.x + o];
            if (x > bm[threadIdx.x] || (x == bm[threadIdx.x] && ii < bi[threadIdx.x])) {
                bm[threadIdx.x] = x;
                bi[threadIdx.x] = ii;
            }
            bx[threadIdx.x] = fmaxf(bx[threadIdx.x], bx[threadIdx.x + o]);
        }
        __syncthreads();
    }
    const float GM = bx[0];
    float S = 0.f;
    for (int b = threadIdx.x; b < (int)gridDim.x; b += 256) {
        const float x = __ldcg(A.pmx + b);
        if (x != -INFINITY) S += __ldcg(A.psx + b) * expf(x - GM);
    }
    for (int c = A8.c0 + threadIdx.x; c < A8.c1; c += 256) {
        const float x = A8.tm[(size_t)j * MKC_NCH + c];
        if (x != -INFINITY) S += A8.ts[(size_t)j * MKC_NCH + c] * expf(x - GM);
    }
    __shared__ float red[33];
    float sv[1] = {S};
    block_sum<1>(sv, red);
    if (threadIdx.x == 0) {
        if (A8.gat) {   // [ink-tp-drv2] this rank's part (token as an exact float, -1: none)
            float* g = A8.gat + A8.grank * 4;
            g[0] = bm[0];
            g[1] = bi[0] >= 0 && bi[0] < V ? (float)bi[0] : -1.f;
            g[2] = GM;
            g[3] = sv[0];
        } else {
            const bool okt = bi[0] >= 0 && bi[0] < V;   // review L1: all-NaN logits -> token 0, prob 0 (the stop ends the draft)
            A.toks[j + 1] = okt ? bi[0] : 0;
            A.prob[j + 1] = okt ? expf(bm[0] - GM) / sv[0] : 0.f;
        }
        *A.cnt = 0u;
    }
}
// [ink-tp-drv2] the ranks' parts of drafted position j (gathered rows: segment z at g + z * N * 4): argmax over ranks
// (ties: lower token, the one-card rule), max, sum rescaled in rank order; every rank the same token / prob. grid n.
static_assert(VPAD < (1 << 24), "token ids travel as exact floats");   // review5 L5
__global__ void k_mkmerge(const float* g, int N, int* toks, float* prob, int j, const float* stop) {
    const int z = blockIdx.x;
    toks += z * DB;
    prob += z * 16;
    if (threadIdx.x != 0 || mk_skip(stop, prob, j)) return;
    g += (size_t)z * N * 4;
    float AM = -INFINITY, GM = -INFINITY;
    int I = 0x7fffffff;
    for (int r = 0; r < N; r++) {
        const float a = g[r * 4], ti = g[r * 4 + 1];
        const int t = ti < 0.f ? 0x7fffffff : (int)ti;
        if (a > AM || (a == AM && t < I)) { AM = a; I = t; }
        GM = fmaxf(GM, g[r * 4 + 2]);
    }
    float S = 0.f;
    for (int r = 0; r < N; r++)
        if (g[r * 4 + 2] != -INFINITY) S += g[r * 4 + 3] * expf(g[r * 4 + 2] - GM);
    const bool okt = I >= 0 && I < V;
    toks[j + 1] = okt ? I : 0;
    prob[j + 1] = okt ? expf(AM - GM) / S : 0.f;
}

// ---- [ink-scrtc] tensor-core screen over the 8-bit head plane for up to 8 token rows (drafter U h, verify head
// screen). Warp = 16 vocab rows (A = rows n / n + 8 of lane 4 n + q), B = the tokens (n8). Lane q reads the 16 B
// units q, q + 4, .. of its two rows (64 B per row per instruction); unit u, mma m: k slots {2q, 2q+1} = elements
// 4m, 4m+1, {2q+8, 2q+9} = 4m+2, 4m+3 of the unit (A and B alike). The screen only needs a valid bound (head) or a
// good score (drafter), so its accumulation order is free; the head bound keeps the k_head_screen margins (EPS has
// 1/1024 slack, far above the tensor-core accumulation error ~144 ulp of a). HEAD: also sum |w||x| and the bound.
// NT column tiles of 8 tokens ([ink-b-rows]: up to MR rows; the weights are loaded once for all tiles, each tile's
// columns get the same arithmetic as in a one-tile launch)
template <bool HEAD, int NT = 1>
__global__ void __launch_bounds__(256) k_screenTC(const __grid_constant__ HeadArgs h, const uint8_t* scr, const uint8_t* seb,
                                                    const bf16* x, int nr, float* y, int ldy, int N) {
    constexpr int NU = HEAD_K / 16, NB = 9;   // 144 units per row, 9 per lane per batch (4 batches, double-buffered)
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5, n = lane >> 2, q = lane & 3;
    const int r0 = (blockIdx.x * 8 + w) * 16 + n, r1 = r0 + 8;
    const bool v0 = r0 < N, v1 = r1 < N;
    const uint8_t* p0 = scr + (size_t)(v0 ? r0 : 0) * HEAD_K + q * 16;
    const uint8_t* p1 = scr + (size_t)(v1 ? r1 : 0) * HEAD_K + q * 16;
    const uint32_t e0 = v0 ? seb[r0] : 0u, e1 = v1 ? seb[r1] : 0u;
    const uint32_t EB0 = (e0 << 7) | (e0 << 23), EB1 = (e1 << 7) | (e1 << 23);
    const bf16* xt[NT];   // B column n of tile ct = token 8 ct + n
#pragma unroll
    for (int ct = 0; ct < NT; ct++) xt[ct] = x + (size_t)(8 * ct + n < nr ? 8 * ct + n : 0) * H + q * 16;
    float s[NT][4], a[NT][4];
#pragma unroll
    for (int ct = 0; ct < NT; ct++)
#pragma unroll
        for (int e = 0; e < 4; e++) s[ct][e] = a[ct][e] = 0.f;
    // rows past N read row 0 and tokens past nr read token 0: their mma rows / columns are never stored
    uint4 wa[2][NB], wb[2][NB];
    auto load = [&](int bt, uint4 (&w0)[NB], uint4 (&w1)[NB]) {
#pragma unroll
        for (int i = 0; i < NB; i++) {
            const int u = (bt * NB + i) * 64;   // byte offset of unit 4 (bt NB + i) + q
            w0[i] = *(const uint4*)(p0 + u);
            w1[i] = *(const uint4*)(p1 + u);
        }
    };
    auto comp = [&](int bt, const uint4 (&w0)[NB], const uint4 (&w1)[NB]) {
#pragma unroll
        for (int i = 0; i < NB; i++) {
            const int u = (bt * NB + i) * 64;
            const uint32_t A0[4] = {w0[i].x, w0[i].y, w0[i].z, w0[i].w}, A1[4] = {w1[i].x, w1[i].y, w1[i].z, w1[i].w};
#pragma unroll
            for (int ct = 0; ct < NT; ct++) {
                const uint4 xa = *(const uint4*)(xt[ct] + u), xb = *(const uint4*)(xt[ct] + u + 8);
                const uint32_t xx[8] = {xa.x, xa.y, xa.z, xa.w, xb.x, xb.y, xb.z, xb.w};
#pragma unroll
                for (int m = 0; m < 4; m++) {
                    const uint32_t q00 = scr_pair<0>(A0[m], EB0), q01 = scr_pair<1>(A0[m], EB0);
                    const uint32_t q10 = scr_pair<0>(A1[m], EB1), q11 = scr_pair<1>(A1[m], EB1);
                    mma_bf16(s[ct], q00, q10, q01, q11, xx[2 * m], xx[2 * m + 1]);
                    if (HEAD)
                        mma_bf16(a[ct], q00 & 0x7fff7fffu, q10 & 0x7fff7fffu, q01 & 0x7fff7fffu, q11 & 0x7fff7fffu, xx[2 * m] & 0x7fff7fffu,
                                 xx[2 * m + 1] & 0x7fff7fffu);
                }
            }
        }
    };
    constexpr int NBT = NU / 4 / NB;
    static_assert(NBT % 2 == 0, "double buffer");
    load(0, wa[0], wb[0]);
#pragma unroll 1
    for (int bt = 0; bt < NBT; bt += 2) {   // the next batch's loads in flight under this batch's math
        load(bt + 1, wa[1], wb[1]);
        comp(bt, wa[0], wb[0]);
        if (bt + 2 < NBT) load(bt + 2, wa[0], wb[0]);
        comp(bt + 1, wa[1], wb[1]);
    }
    // c0 / c1: row r0, tokens 2q / 2q+1; c2 / c3: row r1
    if (!HEAD) {
#pragma unroll
        for (int ct = 0; ct < NT; ct++) {
            const int t0 = 8 * ct + 2 * q, t1 = t0 + 1;
            if (v0 && t0 < nr) y[(size_t)t0 * ldy + r0] = s[ct][0];
            if (v0 && t1 < nr) y[(size_t)t1 * ldy + r0] = s[ct][1];
            if (v1 && t0 < nr) y[(size_t)t0 * ldy + r1] = s[ct][2];
            if (v1 && t1 < nr) y[(size_t)t1 * ldy + r1] = s[ct][3];
        }
        return;
    }
    // per-token sum |x| (the bound's second term), every block the same fixed order
    __shared__ float xs1[8 * NT];
    for (int t = w; t < 8 * NT; t += 8) {   // warp w sums tokens w, w + 8, ..
        float v = 0.f;
        if (t < nr)
            for (int c = lane; c < H / 2; c += 32) {
                const uint32_t p = *(const uint32_t*)(x + (size_t)t * H + 2 * c);
                v += fabsf(__uint_as_float(p << 16)) + fabsf(__uint_as_float(p & 0xffff0000u));
            }
        v = warp_sum(v);
        if (lane == 0) xs1[t] = v;
    }
    __syncthreads();
    constexpr float EPS = 1.f / 16 + 1.f / 1024, XM = 1.f + 1.f / 1024;
    __shared__ unsigned bl[8 * NT][8];
#pragma unroll
    for (int ct = 0; ct < NT; ct++) {
        const int t0 = 8 * ct + 2 * q, t1 = t0 + 1;
        unsigned mx[2] = {0u, 0u};   // tokens t0, t1
#pragma unroll
        for (int e = 0; e < 4; e++) {
            const int t = (e & 1) ? t1 : t0, row = e < 2 ? r0 : r1;
            const bool ok = t < nr && (e < 2 ? v0 : v1);
            const float X1 = t < nr ? xs1[t] : 0.f;
            const float B = a[ct][e] * EPS + __uint_as_float((e < 2 ? e0 : e1) << 23) * X1 * XM;
            if (ok) {
                float hi = s[ct][e] + B, lo = s[ct][e] - B, iT;
                uint32_t sd;
                int st;
                if (row_samp(h, t, iT, sd, st)) {   // [ink-srv-samp] the certified Gumbel bounds of a sampled row
                    const float gg = gumbel(sd, st, h.r0 + row), sl = 1e-5f * (30.f * iT + fabsf(gg)) + 1e-6f;
                    hi = head_cap(hi) * iT + gg + sl;
                    lo = head_cap(lo) * iT + gg - sl;
                }
                h.hi[(size_t)t * h.N + row] = hi;
                if (h.lo && row_filt(h, t)) h.lo[(size_t)t * h.N + row] = lo;   // [ink-samp-dev] raw lower end of a filtered row
                mx[e & 1] = max(mx[e & 1], f2ord(lo));
            }
        }
        // max lower end per token over the block: lanes with the same q hold the same tokens
#pragma unroll
        for (int k = 0; k < 2; k++) {
            unsigned m = mx[k];
            m = max(m, __shfl_xor_sync(0xffffffffu, m, 4));
            m = max(m, __shfl_xor_sync(0xffffffffu, m, 8));
            m = max(m, __shfl_xor_sync(0xffffffffu, m, 16));
            if (n == 0) bl[8 * ct + 2 * q + k][w] = m;
        }
    }
    __syncthreads();
    if (threadIdx.x < nr) {
        const int j = threadIdx.x;
        unsigned m = bl[j][0];
        for (int k = 1; k < 8; k++) m = max(m, bl[j][k]);
        atomicMax(h.thr + j * 16, m);
    }
}
