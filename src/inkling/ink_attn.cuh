// [ink-attn-tb] decode attention with chunks aligned to absolute key positions, one form for the plain step and for
// the verify rows. Chunk z = keys [z C, z C + C) (C per depth bucket, the same for every row of a step); a block
// (KV head, chunk, m-tile) runs up to 5 tokens x 3 q heads as the 15 rows of one m16 tile, so K / V are read once
// for all rows. Each row masks the keys outside [lo_j, qp_j]; its partial for chunk z goes to slot z - lo_j / C and
// its merge (the last of its chunks' blocks) is per token as in k_attn_tc. A row's arithmetic depends only on its own
// Q, keys and chunk grid (mma rows are independent; fully masked 16-key steps are exact no-ops), so a verify row is
// bitwise the plain step at its position. FP16 P / V (the k_attn_tc<true> numerics). Included from ink.cu.

struct A_attn_tb {
    const float* Q;
    const float* R;
    const float* proj;
    const bf16* Kc;
    const bf16* Vc;
    int cap, ring, local;
    const int* posp;
    float *pm, *pl, *pa;
    int* cnt;   // [token][KV head]
    const float* S;
    const float* P;
    bf16* AO;
    int ldp;
    int nr;     // tokens (rows of the step)
    int tpt;    // tokens per m-tile (5: batched; 1: one block row per token)
    int C;      // chunk keys (power of 2, >= 16)
    int Cv[MROW];  // tpt == 1: chunk keys of block row (token) blockIdx.z
    float* AOf = nullptr;   // optional fp32 copy of the gated output [token][QD] (precision checks, INK_ATTN_CMP)
    const float* projT = nullptr;   // [REL][DREL] (k_attn_tb3)
    int tj0[MROW] = {}, tnt[MROW] = {};   // k_attn_tb3<false>: tile blockIdx.z = rows tj0 .. tj0 + tnt - 1 (one segment, one C)
    const int* sko = nullptr;   // [ink-srv-pool] row steps: K / V offset of a slot in HD-rows (null: slot x NKV x cap)
    int kv0 = 0;   // [ink-tp-dec1] KV head of blockIdx.x 0 (a rank's heads; Kc / Vc based so that head kv0 + i is in place)
    int tbo = 0;   // [ink-audit-rowgraph] k_attn_tb3: > 0 = Cv / tj0 / tnt / tile count from posp + tbo (RT_TB layout), not from here
};
constexpr int TB_NSX = 288;   // partial slots per (token, head): <= 2 x 128 + 1 chunks
__device__ __forceinline__ void tb_range_(const int* posp, int local, int j, int& lo, int& qp) {
    qp = posp[j];   // row table (RT_*)
    lo = local ? max(0, qp - (WIN - 1)) : 0;
}
#define tb_range(A, j, lo, qp) tb_range_((A).posp, (A).local, j, lo, qp)
// ONE: one token per block (plain step, verify k <= 2): rows g + 8 compile out; row g's arithmetic is the same code
template <bool ONE>
__global__ void __launch_bounds__(128) k_attn_tb(const __grid_constant__ A_attn_tb A) {
    __shared__ __align__(16) unsigned char kvbuf[2 * 4 * 16 * TC_LD * 2];   // ks, vs; then the per-warp partials
    bf16 (*ks)[16][TC_LD] = (bf16 (*)[16][TC_LD])kvbuf;
    bf16 (*vs)[16][TC_LD] = (bf16 (*)[16][TC_LD])(kvbuf + 4 * 16 * TC_LD * 2);
    __shared__ float sm[4][16], sl[4][16];
    __shared__ float rsh[16][DREL];
    __shared__ int lastv[5];
    __shared__ float we[GQ][TB_NSX];
    const int kvh = blockIdx.x, mt = blockIdx.z;
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5, g = lane >> 2, tq = lane & 3;
    const int j0 = ONE ? mt : mt * A.tpt, nt_ = ONE ? 1 : min(A.tpt, A.nr - j0);   // tokens of this block
    const int C = (ONE || A.tpt == 1) ? A.Cv[mt] : A.C;
    if (C == 0) return;   // a row of the other attention form (mixed steps)
    // the block's tokens are rows of one segment (host rule): one slot
    const size_t sqo = A.sko ? (size_t)A.sko[A.posp[RT_SLOT + j0]] * HD : (size_t)A.posp[RT_SLOT + j0] * NKV * A.cap * HD;
    // block chunk: z = (smallest lo of the block's tokens) / C + blockIdx.y
    int lo0, qp0, loN, qpN;
    tb_range(A, j0, lo0, qp0);
    tb_range(A, j0 + nt_ - 1, loN, qpN);
    const int z = lo0 / C + blockIdx.y;
    const int kb0 = z * C, kb1 = kb0 + C - 1;
    if (kb0 > qpN) return;   // past every row (grid bound)
    const int klo = max(kb0, lo0), khi = min(kb1, qpN);
    // rows of lane g: g and g + 8 (row r = 3 t + q, t < nt_)
    const int rA = g, rB = g + 8;
    const bool vA = rA < 3 * nt_, vB = !ONE && rB < 3 * nt_;
    const int tA = j0 + (vA ? rA / 3 : 0), qA = rA % 3, tB = j0 + (vB ? rB / 3 : 0), qB = rB % 3;
    int loA, qpA, loB, qpB;
    tb_range(A, tA, loA, qpA);
    tb_range(A, tB, loB, qpB);
    for (int x = threadIdx.x; x < 16 * DREL; x += 128) {   // 16 rows x 16 > 128 threads
        const int r = x / DREL, i = x % DREL;
        const bool v = r < 3 * nt_;
        rsh[r][i] = v ? A.R[(size_t)(j0 + r / 3) * NQ * DREL + (kvh * GQ + r % 3) * DREL + i] : 0.f;
    }
    uint32_t qa0[8], qa2[8], qa1[8], qa3[8];
    {
        const float* qr = A.Q + (size_t)tA * QD + (size_t)(kvh * GQ + qA) * HD;
        const float* qs = A.Q + (size_t)tB * QD + (size_t)(kvh * GQ + qB) * HD;
#pragma unroll
        for (int s = 0; s < 8; s++) {
            const int c = s * 16 + 2 * tq;
            qa0[s] = vA ? pack_bf2(qr[c], qr[c + 1]) : 0u;
            qa2[s] = vA ? pack_bf2(qr[c + 8], qr[c + 9]) : 0u;
            qa1[s] = vB ? pack_bf2(qs[c], qs[c + 1]) : 0u;
            qa3[s] = vB ? pack_bf2(qs[c + 8], qs[c + 9]) : 0u;
        }
    }
    __syncthreads();
    float rrA[DREL], rrB[DREL];
#pragma unroll
    for (int i = 0; i < DREL; i++) {
        rrA[i] = rsh[rA][i];
        rrB[i] = rsh[rB][i];
    }
    const float scale = 1.f / sqrtf((float)HD);
    float mA = -INFINITY, lA = 0.f, mB = -INFINITY, lB = 0.f;
    float o[16][4];
#pragma unroll
    for (int j = 0; j < 16; j++) o[j][0] = o[j][1] = o[j][2] = o[j][3] = 0.f;
    // 16-key steps kb0 + 16 s, warp w takes s = w mod 4; steps wholly below klo are skipped (masked for every row)
    int s0 = (klo - kb0) >> 4;
    s0 += (w - s0 % 4 + 4) % 4;
    for (int base = kb0 + 16 * s0; base <= khi; base += 64) {
        {
            const int rI = lane >> 1, hh = lane & 1, kk = min(base + rI, khi);
            const int slot = A.ring ? kk % A.cap : kk;
            const uint4* kp = (const uint4*)(A.Kc + sqo + ((size_t)kvh * A.cap + slot) * HD + hh * 64);
            const uint4* vp = (const uint4*)(A.Vc + sqo + ((size_t)kvh * A.cap + slot) * HD + hh * 64);
            uint4* kd = (uint4*)&ks[w][rI][hh * 64];
            uint4* vd = (uint4*)&vs[w][rI][hh * 64];
#pragma unroll
            for (int i = 0; i < 8; i++) {
                kd[i] = kp[i];
                uint4 v = vp[i];
                uint32_t* u = (uint32_t*)&v;
#pragma unroll
                for (int q = 0; q < 4; q++) u[q] = pack_h2(__uint_as_float(u[q] << 16), __uint_as_float(u[q] & 0xffff0000u));
                vd[i] = v;
            }
        }
        __syncwarp();
        float s[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
#pragma unroll
        for (int nt = 0; nt < 2; nt++)
#pragma unroll
            for (int st = 0; st < 8; st++) {
                const bf16* kr = &ks[w][nt * 8 + g][st * 16 + 2 * tq];
                mma_bf16(s[nt], qa0[st], qa1[st], qa2[st], qa3[st], *(const uint32_t*)kr, *(const uint32_t*)(kr + 8));
            }
        float mxA = -INFINITY, mxB = -INFINITY;
#pragma unroll
        for (int nt = 0; nt < 2; nt++)
#pragma unroll
            for (int e = 0; e < 4; e++) {
                const bool rb = e >= 2;
                const int key = base + nt * 8 + 2 * tq + (e & 1);
                const int qp = rb ? qpB : qpA, lo = rb ? loB : loA;
                float v = s[nt][e] * scale;
                const int dist = qp - key;
                const bool ok = (rb ? vB : vA) && key <= qp && key >= lo && key <= kb1;
                if (ok && dist < REL) {
                    float b = 0.f;
#pragma unroll
                    for (int i = 0; i < DREL; i++) b += (rb ? rrB[i] : rrA[i]) * __ldg(A.proj + i * REL + dist);
                    v += b;
                }
                if (!ok) v = -INFINITY;
                s[nt][e] = v;
                if (rb) mxB = fmaxf(mxB, v);
                else mxA = fmaxf(mxA, v);
            }
        mxA = fmaxf(mxA, __shfl_xor_sync(0xffffffffu, mxA, 1));
        mxA = fmaxf(mxA, __shfl_xor_sync(0xffffffffu, mxA, 2));
        mxB = fmaxf(mxB, __shfl_xor_sync(0xffffffffu, mxB, 1));
        mxB = fmaxf(mxB, __shfl_xor_sync(0xffffffffu, mxB, 2));
        const float mnA = fmaxf(mA, mxA), mnB = fmaxf(mB, mxB);
        const float cfA = (mA == -INFINITY) ? 0.f : expf(mA - mnA), cfB = (mB == -INFINITY) ? 0.f : expf(mB - mnB);
        float psA = 0.f, psB = 0.f, p[2][4];
#pragma unroll
        for (int nt = 0; nt < 2; nt++)
#pragma unroll
            for (int e = 0; e < 4; e++) {
                const bool rb = e >= 2;
                p[nt][e] = (s[nt][e] == -INFINITY) ? 0.f : expf(s[nt][e] - (rb ? mnB : mnA));
                if (rb) psB += p[nt][e];
                else psA += p[nt][e];
            }
        psA += __shfl_xor_sync(0xffffffffu, psA, 1);
        psA += __shfl_xor_sync(0xffffffffu, psA, 2);
        psB += __shfl_xor_sync(0xffffffffu, psB, 1);
        psB += __shfl_xor_sync(0xffffffffu, psB, 2);
        if (mnA != -INFINITY) {
            lA = lA * cfA + psA;
            mA = mnA;
#pragma unroll
            for (int j = 0; j < 16; j++) {
                o[j][0] *= cfA;
                o[j][1] *= cfA;
            }
        }
        if (mnB != -INFINITY) {
            lB = lB * cfB + psB;
            mB = mnB;
#pragma unroll
            for (int j = 0; j < 16; j++) {
                o[j][2] *= cfB;
                o[j][3] *= cfB;
            }
        }
        const uint32_t pa0 = pack_h2(p[0][0], p[0][1]), pa2 = pack_h2(p[1][0], p[1][1]);
        const uint32_t pa1 = pack_h2(p[0][2], p[0][3]), pa3 = pack_h2(p[1][2], p[1][3]);
#pragma unroll
        for (int j = 0; j < 16; j += 2) {
            uint32_t b0, b1, b2, b3;
            const int mrow = (lane & 7) + ((lane >> 3) & 1) * 8, mcol = j * 8 + (lane >> 4) * 8;
            const uint32_t addr = (uint32_t)__cvta_generic_to_shared(&vs[w][mrow][mcol]);
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
                         : "=r"(b0), "=r"(b1), "=r"(b2), "=r"(b3)
                         : "r"(addr));
            mma_f16(o[j], pa0, pa1, pa2, pa3, b0, b1);
            mma_f16(o[j + 1], pa0, pa1, pa2, pa3, b2, b3);
        }
        __syncwarp();
    }
    __syncthreads();   // ks / vs dead: their space holds the per-warp partials
    float (*sacc)[16][HD] = (float (*)[16][HD])kvbuf;
    static_assert(4 * 16 * HD * 4 <= sizeof(kvbuf), "partials fit");
    if (tq == 0) {
        sm[w][rA] = mA;
        sl[w][rA] = lA;
        sm[w][rB] = mB;
        sl[w][rB] = lB;
    }
#pragma unroll
    for (int j = 0; j < 16; j++) {
        sacc[w][rA][j * 8 + 2 * tq] = o[j][0];
        sacc[w][rA][j * 8 + 2 * tq + 1] = o[j][1];
        sacc[w][rB][j * 8 + 2 * tq] = o[j][2];
        sacc[w][rB][j * 8 + 2 * tq + 1] = o[j][3];
    }
    __syncthreads();
    const int dd = threadIdx.x;
    for (int r = 0; r < 3 * nt_; r++) {
        const int t = j0 + r / 3, q = r % 3;
        int lo, qp;
        tb_range(A, t, lo, qp);
        if (z > qp / C || z < lo / C) continue;   // chunk outside this row's range
        float M = -INFINITY;
        for (int j = 0; j < 4; j++) M = fmaxf(M, sm[j][r]);
        float L = 0.f, oo = 0.f;
        if (M != -INFINITY)
            for (int j = 0; j < 4; j++)
                if (sm[j][r] != -INFINITY) {
                    const float e = expf(sm[j][r] - M);
                    L += sl[j][r] * e;
                    oo += sacc[j][r][dd] * e;
                }
        const size_t pi = ((size_t)t * NQ + kvh * GQ + q) * TB_NSX + (z - lo / C);
        A.pa[pi * HD + dd] = oo;
        if (dd == 0) {
            A.pm[pi] = M;
            A.pl[pi] = L;
        }
    }
    __threadfence();
    __syncthreads();
    if (threadIdx.x < nt_) {
        const int t = j0 + threadIdx.x;
        int lo, qp;
        tb_range(A, t, lo, qp);
        const int NS = qp / C - lo / C + 1;
        lastv[threadIdx.x] = (z <= qp / C && z >= lo / C) ? atomicAdd(A.cnt + t * NKV + kvh, 1) == NS - 1 : 0;
    }
    __syncthreads();
    for (int ti = 0; ti < nt_; ti++) {
        if (!lastv[ti]) continue;   // block-uniform
        __threadfence();
        const int jr = j0 + ti;
        int lo, qp;
        tb_range(A, jr, lo, qp);
        const int NS = qp / C - lo / C + 1;
        if (w < GQ) {   // warp q: merge weights of q head kvh * 3 + q, a lane per split (NS <= 160)
            const int h = kvh * GQ + w;
            const size_t b0 = ((size_t)jr * NQ + h) * TB_NSX;
            const float sink = A.S[(size_t)jr * NQ + h];
            float mz[9], lz[9], M = sink;
#pragma unroll
            for (int j = 0; j < 9; j++) {
                const int zz = lane + 32 * j;
                mz[j] = zz < NS ? __ldcg(A.pm + b0 + zz) : -INFINITY;
                lz[j] = zz < NS ? __ldcg(A.pl + b0 + zz) : 0.f;
                M = fmaxf(M, mz[j]);
            }
            M = warp_max(M);
            float L = 0.f, e[9];
#pragma unroll
            for (int j = 0; j < 9; j++) {
                e[j] = mz[j] == -INFINITY ? 0.f : expf(mz[j] - M);
                L += lz[j] * e[j];
            }
            L = warp_sum(L) + expf(sink - M);
#pragma unroll
            for (int j = 0; j < 9; j++)
                if (lane + 32 * j < NS) we[w][lane + 32 * j] = e[j] / L;
        }
        __syncthreads();
        float o3[GQ] = {0.f, 0.f, 0.f};
#pragma unroll
        for (int q = 0; q < GQ; q++) {
            const size_t b0 = ((size_t)jr * NQ + kvh * GQ + q) * TB_NSX;
#pragma unroll 8
            for (int zz = 0; zz < NS; zz++) o3[q] += we[q][zz] * __ldcg(A.pa + (b0 + zz) * HD + dd);
        }
#pragma unroll
        for (int q = 0; q < GQ; q++) {
            const int h = kvh * GQ + q;
            const float gt = A.P[(size_t)jr * A.ldp + OFF_G + h * HD + dd];
            A.AO[(size_t)jr * QD + h * HD + dd] = __float2bfloat16(o3[q] * (1.f / (1.f + expf(-gt))));
            if (A.AOf) A.AOf[(size_t)jr * QD + h * HD + dd] = o3[q] * (1.f / (1.f + expf(-gt)));
        }
        if (threadIdx.x == 0) A.cnt[jr * NKV + kvh] = 0;
        __syncthreads();   // we[] reuse by the next token
    }
}
#undef tb_range
