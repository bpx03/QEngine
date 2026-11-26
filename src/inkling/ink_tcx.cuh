// [ink-tp-tcx] hybrid tensor-core experts of row steps (INK_TCX_MIN = T rows, default off). NOT bitwise == alone:
// a deliberate exception under evaluation. Rows < T keep the CUDA-core fp32 kernels (bitwise).
// Form of [ink-b-xgate] (wt/inkling-batch tools/inkling_batch/xgate.cu k_xtc2), generic over the rank's columns:
//  - weights exact in fp16: NVFP4 e2m1 x e4m3 block scale (x 2^-8); MXFP8 e4m3 (x 2^-8) x 2^(s - m + 7) with m the
//    row's max E8M0 scale (exact for blocks s >= m - 14; the loader counts the others);
//  - activations as two fp16 parts (hi + lo): gate / up x rows scaled by 2^px (max |x| in [2^14, 2^15)), down the
//    gate / up act_fn x 2^9 (|act_fn| <= 100);
//  - m16n8k16 fp16 MMA, a fresh fp32 accumulator per k-step (hi then lo), the k-step sums added on the CUDA cores;
//  - one job = (expert, <= 8 users) from k_order (mu 8, expert-id order); the block's 16 W weight rows staged by
//    cp.async in NS stages of 128 SUB columns, every weight tile read once for all users of the job.
// Lane (g, tq): rows g / g + 8 of the warp's 16-row tile, user column g of B; per 128-column sub-chunk the lane owns the
// 32 elements [32 tq, 32 tq + 32) (k-step s takes its elements 4 s .. 4 s + 3; A and B permuted alike). K tails
// (down K = the rank's columns, e.g. 576) run as a partial last sub-chunk with A = 0 and B = 0 for absent units.
constexpr int TCX_ASC = 9;
__device__ __forceinline__ void tcx_e2m1(uint32_t w16, uint32_t& h0, uint32_t& h1) {   // 4 nibbles -> 2 half2 (exact)
    const uint32_t mag = __byte_perm(0x3E3C3800u, 0x46444240u, w16 & 0x7777u);
    const uint32_t sg = __byte_perm(0x00008000u, 0u, (w16 >> 3) & 0x1111u);
    const uint32_t hb = mag | sg;
    h0 = __byte_perm(hb, 0u, 0x1404);
    h1 = __byte_perm(hb, 0u, 0x3424);
}
__device__ __forceinline__ uint32_t tcx_hmul2(uint32_t a, uint32_t b) {
    const __half2 r = __hmul2(*(const __half2*)&a, *(const __half2*)&b);
    return *(const uint32_t*)&r;
}
__device__ __forceinline__ uint32_t tcx_e4m3h2(uint32_t b) {   // one e4m3 byte -> fp16 (value x 2^-8) in both halves
    const uint32_t h = ((b & 0x7fu) << 7) | ((b & 0x80u) << 8);
    return h | (h << 16);
}
__device__ __forceinline__ uint32_t tcx_e4m3p(uint32_t t) {   // bytes 0 and 2 of t -> half2 (x 2^-8, exact)
    return ((t & 0x007f007fu) << 7) | ((t & 0x00800080u) << 8);
}
__device__ __forceinline__ uint32_t tcx_p2h2(int e) {   // 2^e in fp16, both halves (0 below 2^-24)
    const uint32_t h = e >= -14 ? (uint32_t)(e + 15) << 10 : e >= -24 ? 1u << (e + 24) : 0u;
    return h | (h << 16);
}
struct A_tcx {
    const uint8_t* cod;
    const uint8_t* scl;
    const uint8_t* rmx;   // MXFP8: the row's max scale byte (rank rows [e][NR]); NVFP4: unused
    const float* f13;
    const float* f2;
    const int* ids;
    const OrdBuf* ob;     // dj (jobs of <= 8 users) / srt from k_order(.., 8, ..)
    const float* wts;
    const __half* bh;     // B: gate / up x planes [T][H]; down act planes [entry][ldi]
    const __half* bl;
    const int* px;        // gate / up: x row t scaled by 2^px[t]
    __half* oh;           // gate / up: act planes [entry][ldi] (act_fn x 2^9) or null
    __half* ol;
    float* of;            // gate / up: fp32 act [entry][I] at column ia (wts f2 act_fn) or null; down: partials [entry][H] / 16384
    int ia, ldi;          // the rank's intermediate columns [ia, ia + ldi)
};
template <int FMT, int SUB>
__device__ __forceinline__ int tcx_swz(int r, int u) {   // 16 B unit position in a staged row (rows g, g + 1 apart in banks)
    if constexpr (FMT == 1) return u ^ (r & 1);
    else return (SUB % 2 == 0) ? (u ^ ((r & 1) << 2)) : u;
}
template <int FMT, int MODE, int W, int NS, int SUB, int MINB = 1>
__global__ void __launch_bounds__(32 * W, MINB) k_tcx(const __grid_constant__ A_tcx A) {
    constexpr int BR = 16 * W, NT = 32 * W, KC = 128 * SUB;
    constexpr int SBC = (FMT == 0 ? 64 : 128) * SUB, UC = SBC / 16;   // code bytes / 16 B units per row and stage
    constexpr int SSC = (FMT == 0 ? 8 : 4) * SUB;                      // scale bytes per row and stage
    constexpr int SEL = FMT == 0 ? 2 : 1;                              // scale element bytes (NVFP4: 2 = one 32-element unit)
    constexpr int SPR = SSC / SEL;                                     // scale elements per row and stage
    constexpr int NSE = (BR * SPR + NT - 1) / NT;                      // scale elements per thread and stage
    static_assert(NS >= 3, "scale pipeline needs NS >= 3");
    __shared__ __align__(16) uint8_t scod[NS][BR][SBC];
    __shared__ __align__(16) uint8_t sscl[NS][BR][SSC];
    const OrdBuf* ob = A.ob;
    if ((int)blockIdx.x >= ob->cnt[1]) return;
    const int jb = ob->dj[blockIdx.x], p0 = jb & 0xffff, nu = jb >> 16;
    const int K = MODE == 0 ? H : A.ldi, NR = MODE == 0 ? 2 * A.ldi : H;
    const int RBC = FMT == 0 ? K / 2 : K, RBS = FMT == 0 ? K / 16 : K / 32;   // code / scale bytes per row
    const int NCH = (K + KC - 1) / KC, NUK = K / 32;                          // stages, 32-element units of K
    const int w = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, tq = lane & 3;
    const int e = A.ids[ob->srt[p0]];
    const size_t row0 = (size_t)e * NR + (size_t)blockIdx.y * BR;
    const uint8_t* cg = A.cod + row0 * RBC;
    const uint8_t* sg = A.scl + row0 * RBS;
    auto ldc = [&](int c, int st) {
#pragma unroll
        for (int idx = threadIdx.x; idx < BR * UC; idx += NT) {
            const int r = idx / UC, u = idx % UC, off = c * SBC + u * 16;
            if (off < RBC) {
                const uint32_t d = (uint32_t)__cvta_generic_to_shared(&scod[st][r][tcx_swz<FMT, SUB>(r, u) * 16]);
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(d), "l"(cg + (size_t)r * RBC + off));
            }
        }
    };
    uint32_t sv[NSE];
    auto lds = [&](int c) {   // scale elements of stage c -> registers
#pragma unroll
        for (int q = 0; q < NSE; q++) {
            const int idx = threadIdx.x + q * NT, r = idx / SPR, j = idx % SPR, off = c * SSC + j * SEL;
            sv[q] = 0u;
            if (idx < BR * SPR && off < RBS) {
                if constexpr (SEL == 2) sv[q] = __ldg((const unsigned short*)(sg + (size_t)r * RBS + off));
                else sv[q] = __ldg(sg + (size_t)r * RBS + off);
            }
        }
    };
    auto sts = [&](int st) {
#pragma unroll
        for (int q = 0; q < NSE; q++) {
            const int idx = threadIdx.x + q * NT, r = idx / SPR, j = idx % SPR;
            if (idx < BR * SPR) {
                if constexpr (SEL == 2) *(uint16_t*)&sscl[st][r][j * 2] = (uint16_t)sv[q];
                else sscl[st][r][j] = (uint8_t)sv[q];
            }
        }
    };
    const bool hasb = g < nu;
    int brow = 0, ent_g = 0;
    if (hasb) {
        ent_g = ob->srt[p0 + g];
        brow = MODE == 0 ? ent_g / TOPK : ent_g;
    }
    const int ldb = MODE == 0 ? H : A.ldi;
    const uint4* bhp = (const uint4*)(A.bh + (size_t)brow * ldb + tq * 32);
    const uint4* blp = (const uint4*)(A.bl + (size_t)brow * ldb + tq * 32);
    const int ra = w * 16 + g, rb = ra + 8;
    int mra = 0, mrb = 0;
    if constexpr (FMT == 1) {
        mra = A.rmx[row0 + ra];
        mrb = A.rmx[row0 + rb];
    }
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int s = 0; s < NS - 1; s++) {
        if (s < NCH) {
            ldc(s, s);
            lds(s);
            sts(s);
        }
        asm volatile("cp.async.commit_group;" ::: "memory");
    }
    if (NS - 1 < NCH) lds(NS - 1);
#pragma unroll 1
    for (int c = 0; c < NCH; c++) {
        asm volatile("cp.async.wait_group %0;" ::"n"(NS - 2) : "memory");
        __syncthreads();
        // stage c + NS - 1: codes by cp.async now; its scales (loaded in the previous iteration) stored now, the next
        // stage's scales loaded into registers (consumed >= 2 barriers later)
        if (c + NS - 1 < NCH) {
            ldc(c + NS - 1, (c + NS - 1) % NS);
            sts((c + NS - 1) % NS);
        }
        asm volatile("cp.async.commit_group;" ::: "memory");
        if (c + NS < NCH) lds(c + NS);
        const int st = c % NS;
#pragma unroll
        for (int j = 0; j < SUB; j++) {
            const int gu = (c * SUB + j) * 4 + tq;   // the lane's 32-element unit of K
            const bool kv = gu < NUK;
            uint4 bh4[4], bl4[4];
            if (hasb && kv) {
#pragma unroll
                for (int q = 0; q < 4; q++) {
                    bh4[q] = __ldg(bhp + (c * SUB + j) * 16 + q);
                    bl4[q] = __ldg(blp + (c * SUB + j) * 16 + q);
                }
            } else {
#pragma unroll
                for (int q = 0; q < 4; q++) bh4[q] = bl4[q] = make_uint4(0, 0, 0, 0);
            }
            uint32_t wd[2][8];   // the lane's 32 elements of rows ra / rb as 8 code words (NVFP4: words 0..3)
            uint32_t s2[2][2];   // scale half2 per row: NVFP4 the two 16-element block scales, MXFP8 one
#pragma unroll
            for (int h = 0; h < 2; h++) {
                const int r = h ? rb : ra;
                if constexpr (FMT == 0) {
                    const uint4 cw = *(const uint4*)&scod[st][r][tcx_swz<FMT, SUB>(r, 4 * j + tq) * 16];
                    wd[h][0] = cw.x; wd[h][1] = cw.y; wd[h][2] = cw.z; wd[h][3] = cw.w;
                    const uint32_t sc = *(const uint16_t*)&sscl[st][r][(4 * j + tq) * 2];
                    s2[h][0] = tcx_e4m3h2(sc & 255u);
                    s2[h][1] = tcx_e4m3h2(sc >> 8);
                } else {
                    const uint4 c0 = *(const uint4*)&scod[st][r][tcx_swz<FMT, SUB>(r, 8 * j + 2 * tq) * 16];
                    const uint4 c1 = *(const uint4*)&scod[st][r][tcx_swz<FMT, SUB>(r, 8 * j + 2 * tq + 1) * 16];
                    wd[h][0] = c0.x; wd[h][1] = c0.y; wd[h][2] = c0.z; wd[h][3] = c0.w;
                    wd[h][4] = c1.x; wd[h][5] = c1.y; wd[h][6] = c1.z; wd[h][7] = c1.w;
                    const int sc = sscl[st][r][4 * j + tq];
                    s2[h][0] = s2[h][1] = tcx_p2h2(sc - (h ? mrb : mra) + 7);
                }
                if (!kv) s2[h][0] = s2[h][1] = 0u;   // absent K tail: A = 0 (B = 0 too)
            }
#pragma unroll
            for (int s = 0; s < 8; s++) {
                uint32_t a[2][2];
#pragma unroll
                for (int h = 0; h < 2; h++) {
                    uint32_t h0, h1;
                    if constexpr (FMT == 0) {
                        const uint32_t ww = wd[h][s >> 1];
                        tcx_e2m1((s & 1) ? ww >> 16 : ww & 0xffffu, h0, h1);
                    } else {
                        const uint32_t ww = wd[h][s];
                        h0 = tcx_e4m3p(__byte_perm(ww, 0u, 0x4140));
                        h1 = tcx_e4m3p(__byte_perm(ww, 0u, 0x4342));
                    }
                    a[h][0] = tcx_hmul2(h0, s2[h][s >> 2]);
                    a[h][1] = tcx_hmul2(h1, s2[h][s >> 2]);
                }
                const uint4 bq = bh4[s >> 1], lq = bl4[s >> 1];
                float d[4] = {0.f, 0.f, 0.f, 0.f};
                mma_f16(d, a[0][0], a[1][0], a[0][1], a[1][1], (s & 1) ? bq.z : bq.x, (s & 1) ? bq.w : bq.y);
                mma_f16(d, a[0][0], a[1][0], a[0][1], a[1][1], (s & 1) ? lq.z : lq.x, (s & 1) ? lq.w : lq.y);
#pragma unroll
                for (int q = 0; q < 4; q++) acc[q] += d[q];
            }
        }
    }
    // C: acc[0..1] = row ra, user columns 2 tq, 2 tq + 1; acc[2..3] = row rb
    float rfa, rfb;   // the rows' weight factor (A16 -> true weight)
    if constexpr (FMT == 0) rfa = rfb = 256.f * (MODE == 0 ? A.f13[e] : 1.f);
    else {
        rfa = __int_as_float((mra - 126 + 127) << 23);
        rfb = __int_as_float((mrb - 126 + 127) << 23);
    }
    acc[0] *= rfa;
    acc[1] *= rfa;
    acc[2] *= rfb;
    acc[3] *= rfb;
    const int rbase = blockIdx.y * BR + w * 16;
    if constexpr (MODE == 0) {
        float up[4];
#pragma unroll
        for (int q = 0; q < 4; q++) up[q] = __shfl_down_sync(0xffffffffu, acc[q], 4);   // lane g + 1: the up row
        if (!(g & 1)) {
#pragma unroll
            for (int q = 0; q < 4; q++) {
                const int col = 2 * tq + (q & 1);
                if (col < nu) {
                    const int ent = ob->srt[p0 + col], i = (rbase + g + (q >> 1) * 8) >> 1;
                    const float fx = __int_as_float((127 - A.px[ent / TOPK]) << 23);
                    const float af = act_fn(acc[q] * fx, up[q] * fx);
                    if (A.oh) {
                        const float as = af * (float)(1 << TCX_ASC);
                        const __half hh = __float2half_rn(as);
                        A.oh[(size_t)ent * A.ldi + i] = hh;
                        A.ol[(size_t)ent * A.ldi + i] = __float2half_rn(as - __half2float(hh));
                    }
                    if (A.of) A.of[(size_t)ent * I + A.ia + i] = A.wts[ent] * A.f2[e] * af;
                }
            }
        }
    } else {
#pragma unroll
        for (int q = 0; q < 4; q++) {
            const int col = 2 * tq + (q & 1);
            if (col < nu) {
                const int ent = ob->srt[p0 + col], r = rbase + g + (q >> 1) * 8;
                A.of[(size_t)ent * H + r] = acc[q] * (1.f / (float)(1 << TCX_ASC) / 16384.f) * A.wts[ent] * A.f2[e];
            }
        }
    }
}
// x rows -> hi / lo fp16 planes with a power-of-two scale: max |x| 2^px in [2^14, 2^15); grid T
__global__ void __launch_bounds__(256) k_tcx_xs(const float* __restrict__ x, __half* __restrict__ xh, __half* __restrict__ xl,
                                                int* __restrict__ px) {
    const int t = blockIdx.x;
    float m = 0.f;
    for (int k = threadIdx.x; k < H; k += 256) m = fmaxf(m, fabsf(x[(size_t)t * H + k]));
    __shared__ float red[8];
    m = warp_max(m);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = m;
    __syncthreads();
    m = red[0];
#pragma unroll
    for (int j = 1; j < 8; j++) m = fmaxf(m, red[j]);
    const int ex = m > 0.f ? ((__float_as_int(m) >> 23) & 255) - 127 : 0;
    const int p = 14 - ex;
    if (threadIdx.x == 0) px[t] = p;
    const float sc = __int_as_float((127 + p) << 23);
    for (int k = threadIdx.x; k < H; k += 256) {
        const float v = x[(size_t)t * H + k] * sc;
        const __half hh = __float2half_rn(v);
        xh[(size_t)t * H + k] = hh;
        xl[(size_t)t * H + k] = __float2half_rn(v - __half2float(hh));
    }
}
