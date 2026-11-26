// [ink-b-attn-tb] k_attn_tb (absolute chunks, up to 5 tokens x 3 q heads per m16 tile) with the k_attn_tc3 data
// movement: K / V rows global -> shared by cp.async in two groups per 16-key step (S runs once K is in, the next K is
// issued after S, the next V after P V), unpadded rows with the 16 B XOR swizzle, V bf16 -> fp16 after ldmatrix.trans,
// the rel-bias query straight into registers, proj from the transposed copy (same sum order), the per-warp partials
// and the merge weights alias the K / V tiles. Every float operation of a row is the k_attn_tb one in the same order,
// so tb3 == k_attn_tb bitwise (checked by INK_ATTN_CMP); a row's result is independent of the tile it sits in.
// Tiles: blockIdx.z = tile; tile = rows tj0 .. tj0 + tnt - 1 of ONE segment with the same chunk keys (host rule).
// Included from ink.cu after ink_attn3.cuh.

template <bool ONE, int NJ = 9>   // NJ: merge lanes x 32 >= the row's chunks (padding terms add exact zeros)
__global__ void __launch_bounds__(128, 3) k_attn_tb3(const __grid_constant__ A_attn_tb A) {
    __shared__ __align__(128) unsigned char kvb[2][4][16 * HD * 2];   // [K | V][warp][16 rows x 256 B, swizzled]
    __shared__ int lastv[5];
    // after the key loop: per-warp partials [4][15][HD] (rows 3 t + q < 15) then sm / sl [4][16] in the tile space
    float (*sm)[16] = (float (*)[16])&kvb[0][0][0] + 4 * 15 * HD / 16;
    float (*sl)[16] = sm + 4;
    const int kvh = blockIdx.x + A.kv0, mt = blockIdx.z;
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5, g = lane >> 2, tq = lane & 3;
    const int* const TB = A.tbo ? A.posp + A.tbo : nullptr;   // [ink-audit-rowgraph] the step's tiles as data
    if (!ONE && TB && mt >= TB[3 * MROW]) return;                // grid sized for the row count: tiles past the step's
    const int j0 = ONE ? mt : TB ? TB[MROW + mt] : A.tj0[mt], nt_ = ONE ? 1 : TB ? TB[2 * MROW + mt] : A.tnt[mt];
    const int C = TB ? TB[j0] : A.Cv[j0];
    if (C == 0) return;   // a row of the other attention form (mixed steps)
    const size_t sqo = A.sko ? (size_t)A.sko[A.posp[RT_SLOT + j0]] * HD : (size_t)A.posp[RT_SLOT + j0] * NKV * A.cap * HD;
    const bf16* const Kc_ = A.Kc + sqo;
    const bf16* const Vc_ = A.Vc + sqo;
    int lo0, qp0, loN, qpN;
    tb_range_(A.posp, A.local, j0, lo0, qp0);
    tb_range_(A.posp, A.local, j0 + nt_ - 1, loN, qpN);
    const int z = lo0 / C + blockIdx.y;
    const int kb0 = z * C, kb1 = kb0 + C - 1;
    if (kb0 > qpN) return;   // past every row (grid bound)
    const int klo = max(kb0, lo0), khi = min(kb1, qpN);
    const uint32_t ksb = (uint32_t)__cvta_generic_to_shared(&kvb[0][w][0]), vsb = (uint32_t)__cvta_generic_to_shared(&kvb[1][w][0]);
    auto issue = [&](int base, uint32_t sb, const bf16* src) {   // 16 rows of this warp (lane: row lane / 2, half lane % 2)
        const int rI = lane >> 1, hh = lane & 1, kk = min(base + rI, khi);
        const int slot = A.ring ? kk % A.cap : kk;
        const char* sp = (const char*)(src + ((size_t)kvh * A.cap + slot) * HD) + hh * 128;
#pragma unroll
        for (int i = 0; i < 8; i++) asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(sb + sw16(rI, hh * 8 + i)), "l"(sp + 16 * i));
        asm volatile("cp.async.commit_group;" ::: "memory");
    };
    // 16-key steps kb0 + 16 s, warp w takes s = w mod 4 (a rule of the chunk only); steps wholly below klo are skipped
    int s0 = (klo - kb0) >> 4;
    s0 += (w - s0 % 4 + 4) % 4;
    const int bfirst = kb0 + 16 * s0;
    if (bfirst <= khi) {   // the first K / V tiles in flight before the Q / rel-bias loads
        issue(bfirst, ksb, Kc_);
        issue(bfirst, vsb, Vc_);
    }
    const int rA = g, rB = g + 8;
    const bool vA = rA < 3 * nt_, vB = !ONE && rB < 3 * nt_;
    const int tA = j0 + (vA ? rA / 3 : 0), qA = rA % 3, tB = j0 + (vB ? rB / 3 : 0), qB = rB % 3;
    uint32_t qa0[8], qa2[8], qa1[8], qa3[8];
    float rrA[DREL], rrB[DREL];
    {   // the same values as scalar loads: float2 / float4 loads
        const float* qr = A.Q + (size_t)tA * QD + (size_t)(kvh * GQ + qA) * HD;
        const float* qs = A.Q + (size_t)tB * QD + (size_t)(kvh * GQ + qB) * HD;
#pragma unroll
        for (int s = 0; s < 8; s++) {
            const int c = s * 16 + 2 * tq;
            const float2 a0 = vA ? *(const float2*)(qr + c) : make_float2(0.f, 0.f), a2 = vA ? *(const float2*)(qr + c + 8) : make_float2(0.f, 0.f);
            qa0[s] = vA ? pack_bf2(a0.x, a0.y) : 0u;
            qa2[s] = vA ? pack_bf2(a2.x, a2.y) : 0u;
            if (!ONE) {
                const float2 b0 = vB ? *(const float2*)(qs + c) : make_float2(0.f, 0.f), b2 = vB ? *(const float2*)(qs + c + 8) : make_float2(0.f, 0.f);
                qa1[s] = vB ? pack_bf2(b0.x, b0.y) : 0u;
                qa3[s] = vB ? pack_bf2(b2.x, b2.y) : 0u;
            } else
                qa1[s] = qa3[s] = 0u;
        }
        const float4* ra = (const float4*)(A.R + (size_t)tA * NQ * DREL + (kvh * GQ + qA) * DREL);
        const float4* rb = (const float4*)(A.R + (size_t)tB * NQ * DREL + (kvh * GQ + qB) * DREL);
#pragma unroll
        for (int i = 0; i < DREL / 4; i++) {
            const float4 x = vA ? ra[i] : make_float4(0.f, 0.f, 0.f, 0.f);
            rrA[4 * i] = x.x, rrA[4 * i + 1] = x.y, rrA[4 * i + 2] = x.z, rrA[4 * i + 3] = x.w;
            const float4 y = (!ONE && vB) ? rb[i] : make_float4(0.f, 0.f, 0.f, 0.f);
            rrB[4 * i] = y.x, rrB[4 * i + 1] = y.y, rrB[4 * i + 2] = y.z, rrB[4 * i + 3] = y.w;
        }
    }
    int loA, qpA, loB, qpB;
    tb_range_(A.posp, A.local, tA, loA, qpA);
    tb_range_(A.posp, A.local, tB, loB, qpB);
    const float scale = 1.f / sqrtf((float)HD);
    float mA = -INFINITY, lA = 0.f, mB = -INFINITY, lB = 0.f;
    float o[16][4];
#pragma unroll
    for (int j = 0; j < 16; j++) o[j][0] = o[j][1] = o[j][2] = o[j][3] = 0.f;
    for (int base = bfirst; base <= khi; base += 64) {
        const bool nxt = base + 64 <= khi;
        asm volatile("cp.async.wait_group 1;" ::: "memory");
        __syncwarp();
        float s[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
#pragma unroll
        for (int nt = 0; nt < 2; nt++)
#pragma unroll
            for (int st = 0; st < 8; st++) {
                const int r = nt * 8 + g;
                uint32_t b0, b1;
                asm volatile("ld.shared.u32 %0, [%1];" : "=r"(b0) : "r"(ksb + sw16(r, 2 * st) + 4 * tq));
                asm volatile("ld.shared.u32 %0, [%1];" : "=r"(b1) : "r"(ksb + sw16(r, 2 * st + 1) + 4 * tq));
                if (ONE) mma_bf16(s[nt], qa0[st], 0u, qa2[st], 0u, b0, b1);
                else mma_bf16(s[nt], qa0[st], qa1[st], qa2[st], qa3[st], b0, b1);
            }
        __syncwarp();
        if (nxt) issue(base + 64, ksb, Kc_);
        float mxA = -INFINITY, mxB = -INFINITY;
#pragma unroll
        for (int nt = 0; nt < 2; nt++)
#pragma unroll
            for (int e = 0; e < 4; e++) {
                const bool rb = e >= 2;
                if (ONE && rb) continue;
                const int key = base + nt * 8 + 2 * tq + (e & 1);
                const int qp = rb ? qpB : qpA, lo = rb ? loB : loA;
                float v = s[nt][e] * scale;
                const int dist = qp - key;
                const bool ok = (rb ? vB : vA) && key <= qp && key >= lo && key <= kb1;
                if (ok && dist < REL) {
                    float b = 0.f;
                    const float4* pt = (const float4*)(A.projT + (size_t)dist * DREL);
#pragma unroll
                    for (int i4 = 0; i4 < DREL / 4; i4++) {
                        const float4 pv = __ldg(pt + i4);
                        b += (rb ? rrB[4 * i4] : rrA[4 * i4]) * pv.x;
                        b += (rb ? rrB[4 * i4 + 1] : rrA[4 * i4 + 1]) * pv.y;
                        b += (rb ? rrB[4 * i4 + 2] : rrA[4 * i4 + 2]) * pv.z;
                        b += (rb ? rrB[4 * i4 + 3] : rrA[4 * i4 + 3]) * pv.w;
                    }
                    v += b;
                }
                if (!ok) v = -INFINITY;
                s[nt][e] = v;
                if (rb) mxB = fmaxf(mxB, v);
                else mxA = fmaxf(mxA, v);
            }
        mxA = fmaxf(mxA, __shfl_xor_sync(0xffffffffu, mxA, 1));
        mxA = fmaxf(mxA, __shfl_xor_sync(0xffffffffu, mxA, 2));
        if (!ONE) {
            mxB = fmaxf(mxB, __shfl_xor_sync(0xffffffffu, mxB, 1));
            mxB = fmaxf(mxB, __shfl_xor_sync(0xffffffffu, mxB, 2));
        }
        const float mnA = fmaxf(mA, mxA), mnB = fmaxf(mB, mxB);
        const float cfA = (mA == -INFINITY) ? 0.f : expf(mA - mnA), cfB = (ONE || mB == -INFINITY) ? 0.f : expf(mB - mnB);
        float psA = 0.f, psB = 0.f, p[2][4];
#pragma unroll
        for (int nt = 0; nt < 2; nt++)
#pragma unroll
            for (int e = 0; e < 4; e++) {
                const bool rb = e >= 2;
                if (ONE && rb) {
                    p[nt][e] = 0.f;
                    continue;
                }
                p[nt][e] = (s[nt][e] == -INFINITY) ? 0.f : expf(s[nt][e] - (rb ? mnB : mnA));
                if (rb) psB += p[nt][e];
                else psA += p[nt][e];
            }
        psA += __shfl_xor_sync(0xffffffffu, psA, 1);
        psA += __shfl_xor_sync(0xffffffffu, psA, 2);
        if (!ONE) {
            psB += __shfl_xor_sync(0xffffffffu, psB, 1);
            psB += __shfl_xor_sync(0xffffffffu, psB, 2);
        }
        if (mnA != -INFINITY) {
            lA = lA * cfA + psA;
            mA = mnA;
#pragma unroll
            for (int j = 0; j < 16; j++) {
                o[j][0] *= cfA;
                o[j][1] *= cfA;
            }
        }
        if (!ONE && mnB != -INFINITY) {
            lB = lB * cfB + psB;
            mB = mnB;
#pragma unroll
            for (int j = 0; j < 16; j++) {
                o[j][2] *= cfB;
                o[j][3] *= cfB;
            }
        }
        if (nxt) asm volatile("cp.async.wait_group 1;" ::: "memory");
        else asm volatile("cp.async.wait_group 0;" ::: "memory");
        __syncwarp();
        const uint32_t pa0 = pack_h2(p[0][0], p[0][1]), pa2 = pack_h2(p[1][0], p[1][1]);
        const uint32_t pa1 = ONE ? 0u : pack_h2(p[0][2], p[0][3]), pa3 = ONE ? 0u : pack_h2(p[1][2], p[1][3]);
#pragma unroll
        for (int j = 0; j < 16; j += 2) {
            uint32_t b0, b1, b2, b3;
            const int mrow = (lane & 7) + ((lane >> 3) & 1) * 8, mu = j + (lane >> 4);
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
                         : "=r"(b0), "=r"(b1), "=r"(b2), "=r"(b3)
                         : "r"(vsb + sw16(mrow, mu)));
            mma_f16(o[j], pa0, pa1, pa2, pa3, bf2h2(b0), bf2h2(b1));
            mma_f16(o[j + 1], pa0, pa1, pa2, pa3, bf2h2(b2), bf2h2(b3));
        }
        __syncwarp();
        if (nxt) issue(base + 64, vsb, Vc_);
    }
    asm volatile("cp.async.wait_group 0;" ::: "memory");
    __syncthreads();   // K / V tiles dead: their space holds the per-warp partials [4][16][HD] fp32 (32 KB)
    float (*sacc)[15][HD] = (float (*)[15][HD])&kvb[0][0][0];
    static_assert(4 * 15 * HD * 4 + 2 * 4 * 16 * 4 <= sizeof(kvb), "partials fit");
    if (tq == 0) {
        sm[w][rA] = mA;
        sl[w][rA] = lA;
        if (!ONE) {
            sm[w][rB] = mB;
            sl[w][rB] = lB;
        }
    }
    if (vA)
#pragma unroll
        for (int j = 0; j < 16; j++) {
            sacc[w][rA][j * 8 + 2 * tq] = o[j][0];
            sacc[w][rA][j * 8 + 2 * tq + 1] = o[j][1];
        }
    if (vB)
#pragma unroll
        for (int j = 0; j < 16; j++) {
            sacc[w][rB][j * 8 + 2 * tq] = o[j][2];
            sacc[w][rB][j * 8 + 2 * tq + 1] = o[j][3];
        }
    // per-row merge info once (threads < 16): the same M / L / e values every thread computed in k_attn_tb
    float (*ew)[16] = sl + 4;            // [4][16] warp weights
    float* Ms = (float*)(ew + 4);        // [16]
    float* Ls = Ms + 16;                 // [16]
    int* Pz = (int*)(Ls + 16);           // [16] partial slot z - lo / C, -1: chunk outside the row's range
    __syncthreads();
    if (threadIdx.x < 3 * nt_) {
        const int r = threadIdx.x, t = j0 + r / 3;
        int lo, qp;
        tb_range_(A.posp, A.local, t, lo, qp);
        Pz[r] = (z > qp / C || z < lo / C) ? -1 : z - lo / C;
        float M = -INFINITY;
        for (int j = 0; j < 4; j++) M = fmaxf(M, sm[j][r]);
        float L = 0.f;
        for (int j = 0; j < 4; j++) {
            float e = 0.f;
            if (M != -INFINITY && sm[j][r] != -INFINITY) {
                e = expf(sm[j][r] - M);
                L += sl[j][r] * e;
            }
            ew[j][r] = e;
        }
        Ms[r] = M;
        Ls[r] = L;
    }
    __syncthreads();
    const int dd = threadIdx.x;
    for (int r = 0; r < 3 * nt_; r++) {
        const int pz = Pz[r];
        if (pz < 0) continue;
        const int t = j0 + r / 3, q = r % 3;
        float oo = 0.f;
        if (Ms[r] != -INFINITY)
#pragma unroll
            for (int j = 0; j < 4; j++)
                if (sm[j][r] != -INFINITY) oo += sacc[j][r][dd] * ew[j][r];
        const size_t pi = ((size_t)t * NQ + kvh * GQ + q) * TB_NSX + pz;
        A.pa[pi * HD + dd] = oo;
        if (dd == 0) {
            A.pm[pi] = Ms[r];
            A.pl[pi] = Ls[r];
        }
    }
    __threadfence();
    __syncthreads();   // partials read: the tiles now hold the merge weights
    if (threadIdx.x < nt_) {
        const int t = j0 + threadIdx.x;
        int lo, qp;
        tb_range_(A.posp, A.local, t, lo, qp);
        const int NS = qp / C - lo / C + 1;
        lastv[threadIdx.x] = (z <= qp / C && z >= lo / C) ? atomicAdd(A.cnt + t * NKV + kvh, 1) == NS - 1 : 0;
    }
    __syncthreads();
    float (*we)[TB_NSX] = (float (*)[TB_NSX])&kvb[0][0][0];
    static_assert(GQ * TB_NSX * 4 <= sizeof(kvb), "merge weights fit");
    for (int ti = 0; ti < nt_; ti++) {
        if (!lastv[ti]) continue;   // block-uniform
        __threadfence();
        const int jr = j0 + ti;
        int lo, qp;
        tb_range_(A.posp, A.local, jr, lo, qp);
        const int NS = qp / C - lo / C + 1;
        if (w < GQ) {
            const int h = kvh * GQ + w;
            const size_t b0 = ((size_t)jr * NQ + h) * TB_NSX;
            const float sink = A.S[(size_t)jr * NQ + h];
            float mz[NJ], lz[NJ], M = sink;
#pragma unroll
            for (int j = 0; j < NJ; j++) {
                const int zz = lane + 32 * j;
                mz[j] = zz < NS ? __ldcg(A.pm + b0 + zz) : -INFINITY;
                lz[j] = zz < NS ? __ldcg(A.pl + b0 + zz) : 0.f;
                M = fmaxf(M, mz[j]);
            }
            M = warp_max(M);
            float L = 0.f, e[NJ];
#pragma unroll
            for (int j = 0; j < NJ; j++) {
                e[j] = mz[j] == -INFINITY ? 0.f : expf(mz[j] - M);
                L += lz[j] * e[j];
            }
            L = warp_sum(L) + expf(sink - M);
#pragma unroll
            for (int j = 0; j < NJ; j++)
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
