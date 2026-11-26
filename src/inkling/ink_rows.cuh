// Multi-row decode-form kernels for the speculative verify step ([ink-dspark-verify]): R <= 8 consecutive tokens
// through the T = 1 kernels' arithmetic. Every row's result must be bit-identical to the one-token step at that
// position, so each kernel keeps the T = 1 kernel's per-row operation order exactly (same chunk loop, same expression,
// same warp reduction); only the weight loads / decodes are shared by the rows. Conv states are not shifted here: the
// raw inputs of every row go to a history buffer and k_commit() rebuilds the states for the accepted prefix.
// Included from ink.cu (namespace ink).


// the post-attention / post-MoE / K / V short conv of one token, same expression as conv1 / k_conv / k_kv
__device__ __forceinline__ float conv_core(float x0, float x1, float x2, float x3, const float* w, int c) {
    float o = w[c * 4 + 3] * x0 + w[c * 4 + 2] * x1 + w[c * 4 + 1] * x2 + w[c * 4 + 0] * x3;
    return (o + x0) * (1.f / sqrtf(5.f));
}

// ---- G15 GEMV over nr <= R rows of x (row stride K): the k_gemv12 group layout, the decoded pair words shared by the
// rows. y row j at y + j * ldy.
struct A_gemv12R {
    G12Args ga;
    int nr, ldy;
};
// one group (2 rows) of the multi-row G15 GEMV from its loaded planes (k_gemv12R)
template <int K, int R, bool ESCOPS, int NQV>
__device__ __forceinline__ void gemv12R_group(const A_gemv12R& A_, int g, const uint4 (&sv)[K / 256], const uint4 (&qv)[NQV],
                                              const uint4 m, int lane) {
    const G12Args& ga = A_.ga;
    constexpr int NC = K / 256;
    const GemvArgs& a = ga.a;
    const int nr = A_.nr;
    const int r0 = g * 2;
    const bf16* xp = r0 < a.r1 ? a.x0 : r0 < a.r2 ? a.x1 : a.x2;
    const uint32_t n = (m.x >> 16) & 0xff;
    auto put = [&](int r, int j, float v) {   // lane 0 holds the warp sums
        if (lane == 0 && r0 + r < a.N) a.y[(size_t)j * A_.ldy + r0 + r] = v;
    };
    if (n == 255) {
        const bf16* W = ga.fb + (size_t)m.y * 2 * K;
#pragma unroll 1
        for (int r = 0; r < 2; r++) {
            uint4 wv[NC];
#pragma unroll
            for (int c = 0; c < NC; c++) wv[c] = *(const uint4*)(W + (size_t)r * K + (c * 32 + lane) * 8);
#pragma unroll
            for (int j = 0; j < R; j++) {
                if (j >= nr) break;
                const bf16* xj = xp + (size_t)j * K;
                float s = 0.f;
#pragma unroll
                for (int c = 0; c < NC; c++) {
                    const uint4 xq = *(const uint4*)(xj + (c * 32 + lane) * 8);
                    const uint32_t ww[4] = {wv[c].x, wv[c].y, wv[c].z, wv[c].w};
                    const uint32_t xx[4] = {xq.x, xq.y, xq.z, xq.w};
#pragma unroll
                    for (int k = 0; k < 4; k++)
                        s += __uint_as_float(ww[k] << 16) * __uint_as_float(xx[k] << 16) +
                             __uint_as_float(ww[k] & 0xffff0000u) * __uint_as_float(xx[k] & 0xffff0000u);
                }
                put(r, j, warp_sum(s));
            }
        }
    } else {
        const uint32_t b0 = m.x & 0xff, b1 = (m.x >> 8) & 0xff;
        const uint32_t Bv[2] = {(b0 << 7) | (b0 << 23), (b1 << 7) | (b1 << 23)};
        uint32_t ESC[2] = {0u, 0u};
        if (ESCOPS) {
            const uint32_t ent[6] = {m.y & 0xffff, m.y >> 16, m.z & 0xffff, m.z >> 16, m.w & 0xffff, m.w >> 16};
#pragma unroll
            for (int s = 0; s < 6; s++) {
                const uint32_t e = ent[s], ex = e >> 6;
                const bool mine = s < (int)n && (e & 31) == (uint32_t)lane;
                const uint32_t v = (ex << 7) | (ex << 23);
                if ((e >> 5) & 1) ESC[1] = mine ? v : ESC[1];
                else ESC[0] = mine ? v : ESC[0];
            }
        }
        const uint32_t* sw = (const uint32_t*)sv;
        const uint32_t* iw = (const uint32_t*)qv;
        // [ink-vrow-xshare] chunk-outer: one x chunk load + unpack per (chunk, token) serves both rows of the group
        // (each output keeps its chunk order: bit-identical)
        float s[2][R];
#pragma unroll
        for (int j = 0; j < R; j++) s[0][j] = s[1][j] = 0.f;
#pragma unroll
        for (int c = 0; c < NC; c++) {
            uint32_t ww[2][4];
#pragma unroll
            for (int r = 0; r < 2; r++) {
                const int t = r * NC + c;
                const uint32_t s0 = sw[2 * t], s1 = sw[2 * t + 1], I = iw[t];
                ww[r][0] = dec_pair<0>(s0, I, Bv[r], ESC[r]);
                ww[r][1] = dec_pair<1>(s0, I, Bv[r], ESC[r]);
                ww[r][2] = dec_pair<2>(s1, I, Bv[r], ESC[r]);
                ww[r][3] = dec_pair<3>(s1, I, Bv[r], ESC[r]);
            }
#pragma unroll
            for (int j = 0; j < R; j++) {
                if (j >= nr) break;
                const uint4 xq = *(const uint4*)(xp + (size_t)j * K + (c * 32 + lane) * 8);
                const uint32_t xx[4] = {xq.x, xq.y, xq.z, xq.w};
                float xl[4], xh[4];
#pragma unroll
                for (int k = 0; k < 4; k++) {
                    xl[k] = __uint_as_float(xx[k] << 16);
                    xh[k] = __uint_as_float(xx[k] & 0xffff0000u);
                }
#pragma unroll
                for (int r = 0; r < 2; r++)
#pragma unroll
                    for (int k = 0; k < 4; k++)
                        s[r][j] += __uint_as_float(ww[r][k] << 16) * xl[k] + __uint_as_float(ww[r][k] & 0xffff0000u) * xh[k];
            }
        }
#pragma unroll
        for (int r = 0; r < 2; r++)
#pragma unroll
            for (int j = 0; j < R; j++)
                if (j < nr) put(r, j, warp_sum(s[r][j]));
    }
}


template <int K, int R, bool ESCOPS>
__global__ void __launch_bounds__(256) k_gemv12R(const __grid_constant__ A_gemv12R A_) {
    const G12Args& ga = A_.ga;
    constexpr int NC = K / 256, WD = 2 * NC, NF = WD / 4, TL = WD % 4;
    const GemvArgs& a = ga.a;
    const int nr = A_.nr;
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    const int g = blockIdx.x * 8 + w;
    const int r0 = g * 2;
    const int G = (a.N + 1) / 2;
    const bf16* xp = r0 < a.r1 ? a.x0 : r0 < a.r2 ? a.x1 : a.x2;
    uint4 sv[NC], qv[NF + (TL ? 1 : 0)], m = make_uint4(0, 0, 0, 0);
    if (g < G) {
#pragma unroll
        for (int i = 0; i < NC; i++) sv[i] = ga.sm[((size_t)g * NC + i) * 32 + lane];
        const uint32_t* ip = ga.idx + (size_t)g * 32 * WD;
#pragma unroll
        for (int i = 0; i < NF; i++) qv[i] = *(const uint4*)(ip + (i * 32 + lane) * 4);
        if (TL) {
            const uint2 t = *(const uint2*)(ip + NF * 128 + lane * 2);
            qv[NF] = make_uint4(t.x, t.y, 0, 0);
        }
        m = ga.meta[g];
    }
    gemv12R_group<K, R, ESCOPS, NF + (TL ? 1 : 0)>(A_, g, sv, qv, m, lane);
}

// ---- G15 GEMV, one row per warp (wo, K = QD) over nr rows, with the post-attention conv over the rows:
// out_j = add_j + conv(dot_j, dot_{j-1}, ..) with the state before the step; raw dots -> hist [j][H].
struct A_gemv12r1R {
    G12Args ga;
    int nr;
    float* hist;
};
template <int K, int R, bool ESCOPS>
__global__ void __launch_bounds__(256) k_gemv12r1R(const __grid_constant__ A_gemv12r1R A_) {
    const G12Args& ga = A_.ga;
    constexpr int NC = K / 256, WD = 2 * NC, US = NC / 2, UI = NC / 4;
    static_assert(NC % 4 == 0, "row-aligned units");
    const GemvArgs& a = ga.a;
    const int nr = A_.nr;
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    const int row = blockIdx.x * 8 + w, g = row >> 1, r = row & 1;
    const bf16* xp = row < a.r1 ? a.x0 : row < a.r2 ? a.x1 : a.x2;
    uint4 sv[US], qv[UI], m = make_uint4(0, 0, 0, 0);
    if (row < a.N) {
#pragma unroll
        for (int i = 0; i < US; i++) sv[i] = ga.sm[((size_t)g * NC + r * US + i) * 32 + lane];
        const uint32_t* ip = ga.idx + (size_t)g * 32 * WD;
#pragma unroll
        for (int i = 0; i < UI; i++) qv[i] = *(const uint4*)(ip + ((r * UI + i) * 32 + lane) * 4);
        m = ga.meta[g];
    }
    const uint32_t n = (m.x >> 16) & 0xff;
    float s[R];
#pragma unroll
    for (int j = 0; j < R; j++) s[j] = 0.f;
    if (n == 255) {
        const bf16* W = ga.fb + (size_t)m.y * 2 * K + (size_t)r * K;
#pragma unroll
        for (int c = 0; c < NC; c++) {
            const uint4 wq = *(const uint4*)(W + (c * 32 + lane) * 8);
            const uint32_t ww[4] = {wq.x, wq.y, wq.z, wq.w};
#pragma unroll
            for (int j = 0; j < R; j++) {
                if (j >= nr) break;
                const uint4 xq = *(const uint4*)(xp + (size_t)j * K + (c * 32 + lane) * 8);
                const uint32_t xx[4] = {xq.x, xq.y, xq.z, xq.w};
#pragma unroll
                for (int k = 0; k < 4; k++)
                    s[j] += __uint_as_float(ww[k] << 16) * __uint_as_float(xx[k] << 16) +
                            __uint_as_float(ww[k] & 0xffff0000u) * __uint_as_float(xx[k] & 0xffff0000u);
            }
        }
    } else {
        const uint32_t b = r ? (m.x >> 8) & 0xff : m.x & 0xff;
        const uint32_t Bv = (b << 7) | (b << 23);
        uint32_t ESC = 0u;
        if (ESCOPS) {
            const uint32_t ent[6] = {m.y & 0xffff, m.y >> 16, m.z & 0xffff, m.z >> 16, m.w & 0xffff, m.w >> 16};
#pragma unroll
            for (int q = 0; q < 6; q++) {
                const uint32_t e = ent[q], ex = e >> 6;
                const bool mine = q < (int)n && (e & 31) == (uint32_t)lane && ((e >> 5) & 1) == (uint32_t)r;
                ESC = mine ? ((ex << 7) | (ex << 23)) : ESC;
            }
        }
        const uint32_t* sw = (const uint32_t*)sv;
        const uint32_t* iw = (const uint32_t*)qv;
#pragma unroll
        for (int c = 0; c < NC; c++) {
            const uint32_t s0 = sw[2 * c], s1 = sw[2 * c + 1], I = iw[c];
            const uint32_t ww[4] = {dec_pair<0>(s0, I, Bv, ESC), dec_pair<1>(s0, I, Bv, ESC), dec_pair<2>(s1, I, Bv, ESC),
                                    dec_pair<3>(s1, I, Bv, ESC)};
#pragma unroll
            for (int j = 0; j < R; j++) {
                if (j >= nr) break;
                const uint4 xq = *(const uint4*)(xp + (size_t)j * K + (c * 32 + lane) * 8);
                const uint32_t xx[4] = {xq.x, xq.y, xq.z, xq.w};
#pragma unroll
                for (int k = 0; k < 4; k++)
                    s[j] += __uint_as_float(ww[k] << 16) * __uint_as_float(xx[k] << 16) +
                            __uint_as_float(ww[k] & 0xffff0000u) * __uint_as_float(xx[k] & 0xffff0000u);
            }
        }
    }
#pragma unroll
    for (int j = 0; j < R; j++)
        if (j < nr) s[j] = warp_sum(s[j]);
    if (lane == 0 && row < a.N) {
        if (a.cst) {
            const int c = row;
            float p3 = a.cst[c], p2 = a.cst[H + c], p1 = a.cst[2 * H + c];   // x3, x2, x1 of row 0
#pragma unroll
            for (int j = 0; j < R; j++) {
                if (j >= nr) break;
                const float x0 = s[j];
                const float o = conv_core(x0, p1, p2, p3, a.cw, c);
                a.y[(size_t)j * H + c] = (a.add ? a.add[(size_t)j * H + c] : 0.f) + o;
                A_.hist[(size_t)j * H + c] = x0;
                p3 = p2;
                p2 = p1;
                p1 = x0;
            }
        } else
#pragma unroll
            for (int j = 0; j < R; j++)
                if (j < nr) a.y[(size_t)j * H + row] = s[j];
    }
}

// static conv over hidden for R rows with the pre-step state (MLP conv after the expert down kernel):
// out_j = add_j + conv(X_j, X_{j-1}, ..); X rows -> hist. One thread per channel, rows in order.
// [ink-tp-convseg] grid (H / 256, segments): block y runs segment y's rows only (segments are independent chains:
// each restarts from its slot's state), the same per-element arithmetic as the one-block-row form (bitwise)
__device__ __forceinline__ void conv_seg(const float* X, int nr, const float* st, const float* w, const float* add, float* out,
                                         float* hist, const int* rt, int c, int z) {
    int j0 = -1;
    for (int j = 0, n = 0; j < nr; j++)
        if (rt[RT_SEG + j] == j) {
            if (n == z) {
                j0 = j;
                break;
            }
            n++;
        }
    if (j0 < 0) return;
    const float* s = st + (size_t)rt[RT_SLOT + j0] * 3 * H;
    float p3 = s[c], p2 = s[H + c], p1 = s[2 * H + c];
    for (int j = j0; j < nr && rt[RT_SEG + j] == j0; j++) {
        const float x0 = X[(size_t)j * H + c];
        const float o = conv_core(x0, p1, p2, p3, w, c);
        out[(size_t)j * H + c] = (add ? add[(size_t)j * H + c] : 0.f) + o;
        hist[(size_t)j * H + c] = x0;
        p3 = p2;
        p2 = p1;
        p1 = x0;
    }
}
__global__ void k_convR(const float* X, int nr, const float* st, const float* w, const float* add, float* out, float* hist,
                        const int* rt) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= H) return;
    if (gridDim.y > 1) {   // [ink-tp-convseg]
        conv_seg(X, nr, st, w, add, out, hist, rt, c, blockIdx.y);
        return;
    }
    float p3 = 0.f, p2 = 0.f, p1 = 0.f;
    for (int j = 0; j < nr; j++) {
        if (rt[RT_SEG + j] == j) {   // a segment starts: its slot's pre-step state
            const float* s = st + (size_t)rt[RT_SLOT + j] * 3 * H;
            p3 = s[c], p2 = s[H + c], p1 = s[2 * H + c];
        }
        const float x0 = X[(size_t)j * H + c];
        const float o = conv_core(x0, p1, p2, p3, w, c);
        out[(size_t)j * H + c] = (add ? add[(size_t)j * H + c] : 0.f) + o;
        hist[(size_t)j * H + c] = x0;
        p3 = p2;
        p2 = p1;
        p1 = x0;
    }
}

// q post + K / V conv + clip / norm + cache write for R rows (grid (NQ + 2 NKV, nr)); k_qkv1 per row, the conv
// inputs of earlier rows from P, the state not shifted (raw K / V inputs -> kh / vh [j][KVD]).
struct A_qkv1R {
    A_qkv1 a;
    float* kh;
    float* vh;
    const int* sko = nullptr;   // [ink-srv-pool] K / V offset of a slot in HD-rows (null: slot x NKV x cap)
};
__global__ void __launch_bounds__(128) k_qkv1R(const __grid_constant__ A_qkv1R A_) {
    const A_qkv1& a = A_.a;
    __shared__ float red[8];
    const int b = blockIdx.x, i = threadIdx.x, j = blockIdx.y;
    // row table (RT_*): position, slot, segment; the conv chain runs over the earlier rows of the row's own segment
    const int pos = a.posp[j], sq = a.posp[RT_SLOT + j], jj = j - a.posp[RT_SEG + j];
    const float* P = a.P + (size_t)j * a.ldp;
    if (b < a.nq) {   // [ink-tp-dec1] the rank's heads (a.q0 / nq / kv0 / nkv; whole model by default)
        const int h = a.q0 + b;
        const float tau = a.full ? log_tau(pos) : 1.f;
        float q = P[OFF_Q + h * HD + i] * a.qs[h * HD + i];
        float s = warp_sum(q * q);
        if ((i & 31) == 0) red[i >> 5] = s;
        __syncthreads();
        const float nrm = sqrtf(red[0] + red[1] + red[2] + red[3]);
        q = q * fminf(QK_CLIP / (nrm + 1e-6f), 1.f);
        a.Q[(size_t)j * QD + h * HD + i] = q * tau;
        if (i < 32) {
            float r = i < DREL ? P[OFF_R + h * DREL + i] * a.rs[h * DREL + i] : 0.f;
            const float rn = sqrtf(warp_sum(r * r));
            r = r * fminf(REL_CLIP / (rn + 1e-6f), 1.f);
            if (i < DREL) a.R[(size_t)j * NQ * DREL + h * DREL + i] = r * tau;
            if (i == 0) a.S[(size_t)j * NQ + h] = P[OFF_S + h];
        }
        return;
    }
    const int isK = b < a.nq + a.nkv, h = a.kv0 + (isK ? b - a.nq : b - a.nq - a.nkv), c = h * HD + i;
    const int off = isK ? a.offK : a.offV;
    const float* st = (isK ? a.kst : a.vst) + (size_t)sq * 3 * KVD;
    const float* w = isK ? a.kw : a.vw;
    float x[4];
#pragma unroll
    for (int q = 0; q < 4; q++) x[q] = (jj - q >= 0) ? a.P[(size_t)(j - q) * a.ldp + off + c] : st[(3 + jj - q) * KVD + c];
    float o;
    if (!a.full) {
        o = w[c * 4 + 3] * x[0] + w[c * 4 + 2] * x[1] + w[c * 4 + 1] * x[2] + w[c * 4 + 0] * x[3];
        o = (o + x[0]) * (1.f / sqrtf(5.f));
    } else {
        const float* dy = P + off + KVD + h * 4;
        o = 0.f;
#pragma unroll
        for (int q = 0; q < 4; q++) o += (w[q * KVD + c] + dy[q]) * x[q];
        o = (o + x[0]) * (1.f / 3.f);
    }
    (isK ? A_.kh : A_.vh)[(size_t)j * KVD + c] = x[0];
    float s = warp_sum(o * o);
    if ((i & 31) == 0) red[i >> 5] = s;
    __syncthreads();
    const float ssum = red[0] + red[1] + red[2] + red[3];
    if (isK) o = o * fminf(QK_CLIP / (sqrtf(ssum) + 1e-6f), 1.f);
    else o = o * rsqrtf(ssum / HD + 1e-6f);
    const int slot = a.ring ? pos % a.cap : pos;
    (isK ? a.Kc : a.Vc)[((A_.sko ? (size_t)A_.sko[sq] : (size_t)sq * NKV * a.cap) + (size_t)h * a.cap + slot) * HD + i] = __float2bfloat16(o);
}

// conv-state commit after a verify step: state = last 3 of [old state, hist rows 0..acc] (in place, ascending)
struct CommitEnt {
    float* st;          // slot 0's state; slot s at st + s * 3 C
    const float* hist;  // [row][C]
    int C;
};
// grid (C / 256, entries, segments): segment z (rows from rt[RT_FIRST + z], slot of its first row) keeps acc[z] + 1 rows
__global__ void k_commit(const CommitEnt* ents, const int* accp, const int* rt) {
    const CommitEnt e = ents[blockIdx.y];
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= e.C) return;
    const int z = blockIdx.z, f = rt[RT_FIRST + z], a = accp[z];
    float* st = e.st + (size_t)rt[RT_SLOT + f] * 3 * e.C;
    const float* hist = e.hist + (size_t)f * e.C;
#pragma unroll
    for (int i = 0; i < 3; i++) {
        const int idx = a + 1 + i;
        st[i * e.C + c] = idx < 3 ? st[idx * e.C + c] : hist[(size_t)(idx - 3) * e.C + c];
    }
}

// accepted drafts: rows 0..nr-1 carry [anchor, d1 ..]; arg[j] = target greedy after row j.
// acc = largest m with d_1..d_m == arg[0..m-1] (0..nr-1); the emitted tokens are arg[0..acc]. Review M5: the
// output ends at the first stop token (acc = j when arg[j] == stop), so no row past it is committed.
// per segment z (thread z of nseg): rows [f_z, f_z+1)
__global__ void k_accept(const int* toks, const int* arg, int nr, int* accp, int stop, const int* rt, int nseg) {
    const int z = threadIdx.x;
    if (z >= nseg) return;
    const int f = rt[RT_FIRST + z], end = z + 1 < nseg ? rt[RT_FIRST + z + 1] : nr;
    int m = 0;
    while (f + m + 1 < end && toks[f + m + 1] == arg[f + m] && arg[f + m] != stop) m++;
    accp[z] = m;
}

// ---- verify rows, warp per row (exact by construction: the one-step kernels' code with the row's x / y)
// G15 GEMV, one group (2 weight rows) per block, warp j = verify row j (blockDim 32 nr); the rows share the planes via L1
template <int K, bool ESCOPS>
__global__ void __launch_bounds__(256) k_gemv12W(const __grid_constant__ A_gemv12R A_) {
    const auto& ga = A_.ga;

    constexpr int NC = K / 256, WD = 2 * NC, NF = WD / 4, TL = WD % 4;
    const GemvArgs& a = ga.a;
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    const int g = blockIdx.x, jrow = w;   // one group per block, a warp per verify row
    const int r0 = g * 2;
    const int G = (a.N + 1) / 2;
    const bf16* xp = (r0 < a.r1 ? a.x0 : r0 < a.r2 ? a.x1 : a.x2) + (size_t)jrow * K;
    uint4 sv[NC], qv[NF + (TL ? 1 : 0)], m = make_uint4(0, 0, 0, 0);
    if (g < G) {
#pragma unroll
        for (int i = 0; i < NC; i++) sv[i] = ga.sm[((size_t)g * NC + i) * 32 + lane];
        const uint32_t* ip = ga.idx + (size_t)g * 32 * WD;
#pragma unroll
        for (int i = 0; i < NF; i++) qv[i] = *(const uint4*)(ip + (i * 32 + lane) * 4);
        if (TL) {
            const uint2 t = *(const uint2*)(ip + NF * 128 + lane * 2);
            qv[NF] = make_uint4(t.x, t.y, 0, 0);
        }
        m = ga.meta[g];
    }
    uint4 xv[NC];
#pragma unroll
    for (int c = 0; c < NC; c++) xv[c] = *(const uint4*)(xp + (c * 32 + lane) * 8);
    float acc[2];
    const uint32_t n = (m.x >> 16) & 0xff;
    if (n == 255) {   // fallback group (warp-uniform): raw BF16 rows, the port's code
        const bf16* W = ga.fb + (size_t)m.y * 2 * K;
#pragma unroll 1
        for (int r = 0; r < 2; r++) {   // one row at a time (rare path, keeps the register budget of the hot path)
            uint4 wv[NC];
#pragma unroll
            for (int c = 0; c < NC; c++) wv[c] = *(const uint4*)(W + (size_t)r * K + (c * 32 + lane) * 8);
            float s = 0.f;
#pragma unroll
            for (int c = 0; c < NC; c++) {
                const uint32_t ww[4] = {wv[c].x, wv[c].y, wv[c].z, wv[c].w};
                const uint32_t xx[4] = {xv[c].x, xv[c].y, xv[c].z, xv[c].w};
#pragma unroll
                for (int k = 0; k < 4; k++)
                    s += __uint_as_float(ww[k] << 16) * __uint_as_float(xx[k] << 16) +
                         __uint_as_float(ww[k] & 0xffff0000u) * __uint_as_float(xx[k] & 0xffff0000u);
            }
            const float t = warp_sum(s);
            if (r == 0) acc[0] = t; else acc[1] = t;
        }
    } else {
        const uint32_t b0 = m.x & 0xff, b1 = (m.x >> 8) & 0xff;
        const uint32_t Bv[2] = {(b0 << 7) | (b0 << 23), (b1 << 7) | (b1 << 23)};
        uint32_t ESC[2] = {0u, 0u};
        if (ESCOPS) {
            const uint32_t ent[6] = {m.y & 0xffff, m.y >> 16, m.z & 0xffff, m.z >> 16, m.w & 0xffff, m.w >> 16};
#pragma unroll
            for (int s = 0; s < 6; s++) {
                const uint32_t e = ent[s], ex = e >> 6;
                const bool mine = s < (int)n && (e & 31) == (uint32_t)lane;
                const uint32_t v = (ex << 7) | (ex << 23);
                if ((e >> 5) & 1) ESC[1] = mine ? v : ESC[1];
                else ESC[0] = mine ? v : ESC[0];
            }
        }
        const uint32_t* sw = (const uint32_t*)sv;
        const uint32_t* iw = (const uint32_t*)qv;
#pragma unroll
        for (int r = 0; r < 2; r++) {
            float s = 0.f;
#pragma unroll
            for (int c = 0; c < NC; c++) {
                const int t = r * NC + c;
                const uint32_t s0 = sw[2 * t], s1 = sw[2 * t + 1], I = iw[t];
                const uint32_t ww[4] = {dec_pair<0>(s0, I, Bv[r], ESC[r]), dec_pair<1>(s0, I, Bv[r], ESC[r]),
                                        dec_pair<2>(s1, I, Bv[r], ESC[r]), dec_pair<3>(s1, I, Bv[r], ESC[r])};
                const uint32_t xx[4] = {xv[c].x, xv[c].y, xv[c].z, xv[c].w};
#pragma unroll
                for (int k = 0; k < 4; k++)
                    s += __uint_as_float(ww[k] << 16) * __uint_as_float(xx[k] << 16) +
                         __uint_as_float(ww[k] & 0xffff0000u) * __uint_as_float(xx[k] & 0xffff0000u);
            }
            acc[r] = warp_sum(s);
        }
    }
    if (lane == 0)
#pragma unroll
        for (int r = 0; r < 2; r++)
            if (r0 + r < a.N) a.y[(size_t)jrow * A_.ldy + r0 + r] = acc[r];
}

// G15 GEMV one row per block (wo), warp j = verify row j, conv over the rows in the epilogue
template <int K, bool ESCOPS>
__global__ void __launch_bounds__(256) k_gemv12r1W(const __grid_constant__ A_gemv12r1R A_) {
    const auto& ga = A_.ga;

    constexpr int NC = K / 256, WD = 2 * NC, US = NC / 2, UI = NC / 4;
    static_assert(NC % 4 == 0, "row-aligned units");
    const GemvArgs& a = ga.a;
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    const int row = blockIdx.x, jrow = w, g = row >> 1, r = row & 1;
    const bf16* xp = (row < a.r1 ? a.x0 : row < a.r2 ? a.x1 : a.x2) + (size_t)jrow * K;
    uint4 sv[US], qv[UI], m = make_uint4(0, 0, 0, 0);
    if (row < a.N) {
#pragma unroll
        for (int i = 0; i < US; i++) sv[i] = ga.sm[((size_t)g * NC + r * US + i) * 32 + lane];
        const uint32_t* ip = ga.idx + (size_t)g * 32 * WD;
#pragma unroll
        for (int i = 0; i < UI; i++) qv[i] = *(const uint4*)(ip + ((r * UI + i) * 32 + lane) * 4);
        m = ga.meta[g];
    }
    uint4 xv[NC];
#pragma unroll
    for (int c = 0; c < NC; c++) xv[c] = *(const uint4*)(xp + (c * 32 + lane) * 8);
    float acc[1];
    const uint32_t n = (m.x >> 16) & 0xff;
    float s = 0.f;
    if (n == 255) {
        const bf16* W = ga.fb + (size_t)m.y * 2 * K + (size_t)r * K;
        uint4 wv[NC];
#pragma unroll
        for (int c = 0; c < NC; c++) wv[c] = *(const uint4*)(W + (c * 32 + lane) * 8);
#pragma unroll
        for (int c = 0; c < NC; c++) {
            const uint32_t ww[4] = {wv[c].x, wv[c].y, wv[c].z, wv[c].w};
            const uint32_t xx[4] = {xv[c].x, xv[c].y, xv[c].z, xv[c].w};
#pragma unroll
            for (int k = 0; k < 4; k++)
                s += __uint_as_float(ww[k] << 16) * __uint_as_float(xx[k] << 16) +
                     __uint_as_float(ww[k] & 0xffff0000u) * __uint_as_float(xx[k] & 0xffff0000u);
        }
    } else {
        const uint32_t b = r ? (m.x >> 8) & 0xff : m.x & 0xff;
        const uint32_t Bv = (b << 7) | (b << 23);
        uint32_t ESC = 0u;
        if (ESCOPS) {
            const uint32_t ent[6] = {m.y & 0xffff, m.y >> 16, m.z & 0xffff, m.z >> 16, m.w & 0xffff, m.w >> 16};
#pragma unroll
            for (int j = 0; j < 6; j++) {
                const uint32_t e = ent[j], ex = e >> 6;
                const bool mine = j < (int)n && (e & 31) == (uint32_t)lane && ((e >> 5) & 1) == (uint32_t)r;
                ESC = mine ? ((ex << 7) | (ex << 23)) : ESC;
            }
        }
        const uint32_t* sw = (const uint32_t*)sv;
        const uint32_t* iw = (const uint32_t*)qv;
#pragma unroll
        for (int c = 0; c < NC; c++) {
            const uint32_t s0 = sw[2 * c], s1 = sw[2 * c + 1], I = iw[c];
            const uint32_t ww[4] = {dec_pair<0>(s0, I, Bv, ESC), dec_pair<1>(s0, I, Bv, ESC), dec_pair<2>(s1, I, Bv, ESC),
                                    dec_pair<3>(s1, I, Bv, ESC)};
            const uint32_t xx[4] = {xv[c].x, xv[c].y, xv[c].z, xv[c].w};
#pragma unroll
            for (int k = 0; k < 4; k++)
                s += __uint_as_float(ww[k] << 16) * __uint_as_float(xx[k] << 16) +
                     __uint_as_float(ww[k] & 0xffff0000u) * __uint_as_float(xx[k] & 0xffff0000u);
        }
    }
    acc[0] = warp_sum(s);
    __shared__ float dots[VR];
    if (lane == 0) dots[jrow] = acc[0];
    __syncthreads();
    if (threadIdx.x == 0 && row < a.N) {
        const int nr = blockDim.x >> 5;
        if (a.cst) {   // post-attention conv over the rows, the state before the step; raw dots -> hist
            const int c = row;
            float p3 = a.cst[c], p2 = a.cst[H + c], p1 = a.cst[2 * H + c];
            for (int j = 0; j < nr; j++) {
                const float x0 = dots[j];
                const float o = conv_core(x0, p1, p2, p3, a.cw, c);
                a.y[(size_t)j * H + c] = (a.add ? a.add[(size_t)j * H + c] : 0.f) + o;
                A_.hist[(size_t)j * H + c] = x0;
                p3 = p2;
                p2 = p1;
                p1 = x0;
            }
        } else
            for (int j = 0; j < nr; j++) a.y[(size_t)j * H + row] = dots[j];
    }
}

// ---- NVFP4 gate / up over the verify rows' expert union: block (candidate t * TOPK + s, chunk); only the first
// occurrence of an expert works and serves every row that routes to it, the 2 GP code rows decoded once per block and
// reused by the rows. Per row the arithmetic of b_gu4 (same element order, same expressions).
__device__ __forceinline__ void fp4dec16(uint32_t w0, uint32_t w1, float* o) {   // the fp4dot16 element order
    float2 p0 = fp4pair(w0), p1 = fp4pair(w0 >> 4), p2 = fp4pair(w0 >> 8), p3 = fp4pair(w0 >> 12);
    o[0] = p0.x; o[1] = p1.x; o[2] = p2.x; o[3] = p3.x; o[4] = p0.y; o[5] = p1.y; o[6] = p2.y; o[7] = p3.y;
    p0 = fp4pair(w1), p1 = fp4pair(w1 >> 4), p2 = fp4pair(w1 >> 8), p3 = fp4pair(w1 >> 12);
    o[8] = p0.x; o[9] = p1.x; o[10] = p2.x; o[11] = p3.x; o[12] = p0.y; o[13] = p1.y; o[14] = p2.y; o[15] = p3.y;
}
__device__ __forceinline__ float dot16f(const float* w, const float* x) {   // == fp4dot16 on decoded values
    const float4 a = *(const float4*)x, b = *(const float4*)(x + 4), c = *(const float4*)(x + 8), d = *(const float4*)(x + 12);
    float s = w[0] * a.x + w[1] * a.y + w[2] * a.z + w[3] * a.w + w[4] * b.x + w[5] * b.y + w[6] * b.z + w[7] * b.w;
    s += w[8] * c.x + w[9] * c.y + w[10] * c.z + w[11] * c.w + w[12] * d.x + w[13] * d.y + w[14] * d.z + w[15] * d.w;
    return s;
}
struct A_gu4u {
    A_gu4 a;
    int T;
};
template <int GP>
__global__ void __launch_bounds__(256) k_gu4u(const __grid_constant__ A_gu4u U) {
    const A_gu4& A = U.a;
    const int T = U.T, cand = blockIdx.x, t = cand / TOPK, lane = threadIdx.x & 31;
    const int i0 = (blockIdx.y * 8 + (threadIdx.x >> 5)) * GP;
    const int e = A.ids[cand];
    for (int c = 0; c < t * TOPK; c++)
        if (A.ids[c] == e) return;   // an earlier row owns this expert
    int ru[VR], su[VR], nu = 0;
    for (int u = t; u < T; u++)
        for (int s = 0; s < TOPK; s++)
            if (A.ids[u * TOPK + s] == e) {
                ru[nu] = u;
                su[nu++] = s;
            }
    const size_t r0 = (size_t)e * 2 * I + 2 * i0;
    float acc[VR][2 * GP];
#pragma unroll
    for (int j = 0; j < VR; j++)
#pragma unroll
        for (int r = 0; r < 2 * GP; r++) acc[j][r] = 0.f;
#pragma unroll
    for (int it = 0; it < 3; it++) {
        const int b = it * 32 + lane;
        if (b < H / 32) {
            uint4 c[2 * GP];
            uint16_t sc[2 * GP];
#pragma unroll
            for (int r = 0; r < 2 * GP; r++) {
                c[r] = *(const uint4*)(A.cod + (r0 + r) * (H / 2) + b * 16);
                sc[r] = *(const uint16_t*)(A.scl + (r0 + r) * (H / 16) + 2 * b);
            }
#pragma unroll
            for (int r = 0; r < 2 * GP; r++) {
                float w[32];
                fp4dec16(c[r].x, c[r].y, w);
                fp4dec16(c[r].z, c[r].w, w + 16);
                const float f0 = e4m3f(sc[r] & 255), f1 = e4m3f(sc[r] >> 8);
#pragma unroll
                for (int j = 0; j < VR; j++) {
                    if (j >= nu) break;
                    float xv[32];
                    const float4* xp = (const float4*)(A.x + (size_t)ru[j] * H + b * 32);
#pragma unroll
                    for (int k = 0; k < 8; k++) *(float4*)(xv + 4 * k) = __ldg(xp + k);
                    acc[j][r] += dot16f(w, xv) * f0 + dot16f(w + 16, xv + 16) * f1;
                }
            }
        }
    }
    const float fe = 16384.f * A.f13[e];
#pragma unroll
    for (int j = 0; j < VR; j++) {
        if (j >= nu) break;
        const float wf = A.wts[ru[j] * TOPK + su[j]] * A.f2[e];
#pragma unroll
        for (int r = 0; r < 2 * GP; r++) acc[j][r] = warp_sum(acc[j][r]);
        if (lane < GP) {
            float g = acc[j][0], u = acc[j][1];
#pragma unroll
            for (int q = 1; q < GP; q++)
                if (lane == q) {
                    g = acc[j][2 * q];
                    u = acc[j][2 * q + 1];
                }
            A.act[((size_t)ru[j] * TOPK + su[j]) * I + i0 + lane] = wf * act_fn(g * fe, u * fe);
        }
    }
}

