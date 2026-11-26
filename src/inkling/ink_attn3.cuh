// [ink-attn-occ3] k_attn_tc<true> (FP16 P / V) with 3 blocks per SM: the same per-split arithmetic (every float
// operation in the same order: S mma, scale / bias / mask, online softmax, P V mma, warp merge, split merge), only
// the data movement differs: K / V rows go global -> shared by cp.async (no staging registers), unpadded rows with
// a 16 B XOR swizzle (conflict-free fragment / ldmatrix reads), V converted bf16 -> fp16 after ldmatrix.trans (the
// same elementwise conversion as at staging), the warp partials alias the warp's own K tile and the merge weights the
// V tiles, the rel-bias query read straight into registers. Smem 32.9 KB, <= 168 registers. Included from ink.cu.

__device__ __forceinline__ uint32_t sw16(int r, int u) { return (uint32_t)(r * 16 + (u ^ (r & 7))) * 16u; }   // byte offset of 16 B unit u of row r
__device__ __forceinline__ uint32_t bf2h2(uint32_t u) { return pack_h2(__uint_as_float(u << 16), __uint_as_float(u & 0xffff0000u)); }

__global__ void __launch_bounds__(128, 3) k_attn_tc3(const __grid_constant__ A_attn_tc A) {
    const bool ab_merge3 = A.merge3;
    __shared__ __align__(128) unsigned char kvb[2][4][16 * HD * 2];   // [K | V][warp][16 rows x 256 B, swizzled]
    __shared__ float sm[4][GQ], sl[4][GQ];
    __shared__ int last;
    const int kvh = blockIdx.x + A.kv0, z = blockIdx.y, jr = blockIdx.z, NSX = gridDim.y;
    const int NS = A.nsv[0] ? A.nsv[jr] : NSX;
    if (z >= NS) return;
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5, g = lane >> 2, tq = lane & 3;
    const int qp = A.posp[jr];   // row table (RT_*): plain steps read a table with slot 0
    const size_t sqo = A.sko ? (size_t)A.sko[A.posp[RT_SLOT + jr]] * HD : (size_t)A.posp[RT_SLOT + jr] * NKV * A.cap * HD;
    const bf16* const Kc_ = A.Kc + sqo;
    const bf16* const Vc_ = A.Vc + sqo;
    const int lo = A.local ? max(0, qp - (WIN - 1)) : 0;
    const int n = qp - lo + 1, kspl = (n + NS - 1) / NS;
    const int k0 = lo + z * kspl, k1 = min(qp, k0 + kspl - 1);
    const uint32_t ksb = (uint32_t)__cvta_generic_to_shared(&kvb[0][w][0]), vsb = (uint32_t)__cvta_generic_to_shared(&kvb[1][w][0]);
    uint32_t qa0[8], qa2[8];
    {
        const float* qr = A.Q + (size_t)jr * QD + (size_t)(kvh * GQ + (g < GQ ? g : 0)) * HD;
#pragma unroll
        for (int s = 0; s < 8; s++) {
            const int c = s * 16 + 2 * tq;
            qa0[s] = g < GQ ? pack_bf2(qr[c], qr[c + 1]) : 0u;
            qa2[s] = g < GQ ? pack_bf2(qr[c + 8], qr[c + 9]) : 0u;
        }
    }
    float rr[DREL];
    {
        const float* rp = A.R + (size_t)jr * NQ * DREL + (kvh * GQ + (g < GQ ? g : 0)) * DREL;
#pragma unroll
        for (int i = 0; i < DREL; i++) rr[i] = g < GQ ? rp[i] : 0.f;
    }
    const float scale = 1.f / sqrtf((float)HD);
    float m = -INFINITY, l = 0.f;
    float o[16][4];
#pragma unroll
    for (int j = 0; j < 16; j++) o[j][0] = o[j][1] = o[j][2] = o[j][3] = 0.f;
    // [ink-attn-pipe] K and V of a chunk in separate cp.async groups: S runs once K is in (V still in flight), the next
    // chunk's K is issued right after S (its tile is free), the next V right after P V; data movement only
    auto issue = [&](int base, uint32_t sb, const bf16* src) {   // 16 rows of this warp (lane: row lane / 2, half lane % 2)
        const int rI = lane >> 1, hh = lane & 1, kk = min(base + rI, k1);
        const int slot = A.ring ? kk % A.cap : kk;
        const char* sp = (const char*)(src + ((size_t)kvh * A.cap + slot) * HD) + hh * 128;
#pragma unroll
        for (int i = 0; i < 8; i++) asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(sb + sw16(rI, hh * 8 + i)), "l"(sp + 16 * i));
        asm volatile("cp.async.commit_group;" ::: "memory");
    };
    if (k0 + w * 16 <= k1) {
        issue(k0 + w * 16, ksb, Kc_);
        issue(k0 + w * 16, vsb, Vc_);
    }
    for (int base = k0 + w * 16; base <= k1; base += 64) {
        const bool nxt = base + 64 <= k1;
        asm volatile("cp.async.wait_group 1;" ::: "memory");   // K of this chunk (V may still be in flight)
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
                mma_bf16(s[nt], qa0[st], 0u, qa2[st], 0u, b0, b1);
            }
        __syncwarp();   // every lane's K reads done
        if (nxt) issue(base + 64, ksb, Kc_);
        float mx = -INFINITY;
#pragma unroll
        for (int nt = 0; nt < 2; nt++)
#pragma unroll
            for (int e = 0; e < 2; e++) {
                const int key = base + nt * 8 + 2 * tq + e;
                float v = s[nt][e] * scale;
                const int dist = qp - key;
                if (g < GQ && dist < REL && key <= k1) {
                    float b = 0.f;   // [ink-attn-pT] proj[i][dist] from the transposed copy: 4 x 16 B loads, same sum order
                    const float4* pt = (const float4*)(A.projT + (size_t)dist * DREL);
#pragma unroll
                    for (int i4 = 0; i4 < DREL / 4; i4++) {
                        const float4 pv = __ldg(pt + i4);
                        b += rr[4 * i4] * pv.x;
                        b += rr[4 * i4 + 1] * pv.y;
                        b += rr[4 * i4 + 2] * pv.z;
                        b += rr[4 * i4 + 3] * pv.w;
                    }
                    v += b;
                }
                if (key > k1 || g >= GQ) v = -INFINITY;
                s[nt][e] = v;
                mx = fmaxf(mx, v);
            }
        mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 1));
        mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 2));
        const float mn = fmaxf(m, mx);
        const float cf = (m == -INFINITY) ? 0.f : expf(m - mn);
        float ps = 0.f, p[2][2];
#pragma unroll
        for (int nt = 0; nt < 2; nt++)
#pragma unroll
            for (int e = 0; e < 2; e++) {
                p[nt][e] = (s[nt][e] == -INFINITY) ? 0.f : expf(s[nt][e] - mn);
                ps += p[nt][e];
            }
        ps += __shfl_xor_sync(0xffffffffu, ps, 1);
        ps += __shfl_xor_sync(0xffffffffu, ps, 2);
        if (mn != -INFINITY) {
            l = l * cf + ps;
            m = mn;
#pragma unroll
            for (int j = 0; j < 16; j++) {
                o[j][0] *= cf;
                o[j][1] *= cf;
            }
        }
        if (nxt) asm volatile("cp.async.wait_group 1;" ::: "memory");   // V of this chunk (the next K may be in flight)
        else asm volatile("cp.async.wait_group 0;" ::: "memory");
        __syncwarp();
        const uint32_t pa0 = pack_h2(p[0][0], p[0][1]);
        const uint32_t pa2 = pack_h2(p[1][0], p[1][1]);
#pragma unroll
        for (int j = 0; j < 16; j += 2) {
            uint32_t b0, b1, b2, b3;
            const int mrow = (lane & 7) + ((lane >> 3) & 1) * 8, mu = j + (lane >> 4);
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
                         : "=r"(b0), "=r"(b1), "=r"(b2), "=r"(b3)
                         : "r"(vsb + sw16(mrow, mu)));
            mma_f16(o[j], pa0, 0u, pa2, 0u, bf2h2(b0), bf2h2(b1));
            mma_f16(o[j + 1], pa0, 0u, pa2, 0u, bf2h2(b2), bf2h2(b3));
        }
        __syncwarp();   // every lane's V reads done
        if (nxt) issue(base + 64, vsb, Vc_);
    }
    // per-warp partials into the warp's own K tile (rows g < 3: lanes 0-11), then the 4-warp merge per q head
    float* saw = (float*)&kvb[0][w][0];   // warp w's partials [GQ][HD] inside its own K tile (1.5 KB of 4 KB)
    if (g < GQ) {
        if (tq == 0) {
            sm[w][g] = m;
            sl[w][g] = l;
        }
#pragma unroll
        for (int j = 0; j < 16; j++) {
            saw[g * HD + j * 8 + 2 * tq] = o[j][0];
            saw[g * HD + j * 8 + 2 * tq + 1] = o[j][1];
        }
    }
    __syncthreads();
    const int dd = threadIdx.x;
    constexpr int WS = 16 * HD * 2 / 4;   // floats per warp tile
#pragma unroll
    for (int q = 0; q < GQ; q++) {
        float M = -INFINITY;
        for (int j = 0; j < 4; j++) M = fmaxf(M, sm[j][q]);
        float L = 0.f, oo = 0.f;
        if (M != -INFINITY)
            for (int j = 0; j < 4; j++)
                if (sm[j][q] != -INFINITY) {
                    const float e = expf(sm[j][q] - M);
                    L += sl[j][q] * e;
                    oo += ((const float*)&kvb[0][0][0])[j * WS + q * HD + dd] * e;
                }
        const size_t pi = ((size_t)jr * NQ + kvh * GQ + q) * NSX + z;
        A.pa[pi * HD + dd] = oo;
        if (dd == 0) {
            A.pm[pi] = M;
            A.pl[pi] = L;
        }
    }
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) last = atomicAdd(A.cnt + jr * NKV + kvh, 1) == NS - 1;
    __syncthreads();
    if (!last) return;
    __threadfence();
    float (*we)[128] = (float (*)[128])&kvb[1][0][0];   // merge weights in the V tiles
    if (w < GQ) {
        const int h = kvh * GQ + w;
        const size_t b0 = ((size_t)jr * NQ + h) * NSX;
        const float sink = A.S[(size_t)jr * NQ + h];
        float mz[4], lz[4], M = sink;
#pragma unroll
        for (int j = 0; j < 4; j++) {
            const int zz = lane + 32 * j;
            mz[j] = zz < NS ? __ldcg(A.pm + b0 + zz) : -INFINITY;
            lz[j] = zz < NS ? __ldcg(A.pl + b0 + zz) : 0.f;
            M = fmaxf(M, mz[j]);
        }
        M = warp_max(M);
        float L = 0.f, e[4];
#pragma unroll
        for (int j = 0; j < 4; j++) {
            e[j] = mz[j] == -INFINITY ? 0.f : expf(mz[j] - M);
            L += lz[j] * e[j];
        }
        L = warp_sum(L) + expf(sink - M);
#pragma unroll
        for (int j = 0; j < 4; j++)
            if (lane + 32 * j < NS) we[w][lane + 32 * j] = e[j] / L;
    }
    __syncthreads();
    float o3[GQ] = {0.f, 0.f, 0.f};
    if (ab_merge3) {
#pragma unroll 4
        for (int zz = 0; zz < NS; zz++)
#pragma unroll
            for (int q = 0; q < GQ; q++) o3[q] += we[q][zz] * __ldcg(A.pa + (((size_t)jr * NQ + kvh * GQ + q) * NSX + zz) * HD + dd);
    } else
#pragma unroll
        for (int q = 0; q < GQ; q++) {
            const size_t b0 = ((size_t)jr * NQ + kvh * GQ + q) * NSX;
#pragma unroll 8
            for (int zz = 0; zz < NS; zz++) o3[q] += we[q][zz] * __ldcg(A.pa + (b0 + zz) * HD + dd);
        }
#pragma unroll
    for (int q = 0; q < GQ; q++) {
        const int h = kvh * GQ + q;
        const float oo = o3[q];
        const float gt = A.P[(size_t)jr * A.ldp + OFF_G + h * HD + dd];
        A.AO[(size_t)jr * QD + h * HD + dd] = __float2bfloat16(oo * (1.f / (1.f + expf(-gt))));
        if (A.AOf) A.AOf[(size_t)jr * QD + h * HD + dd] = oo * (1.f / (1.f + expf(-gt)));   // review r5 L4
    }
    if (threadIdx.x == 0) A.cnt[jr * NKV + kvh] = 0;
}
