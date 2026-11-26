// [ink-tcr] tensor-core G15 GEMV over fragment-ordered planes (same bits as G15, units reordered at load).
// Tile b = groups 8b..8b+7 (16 rows); warp w of the 8-warp block takes the original G15 lanes 4w..4w+3; lane L = 4n + q
// holds group 8b + n, original lane 4w + q, BOTH rows: per chunk c six words [r0 s0, r0 s1, r1 s0, r1 s1, r0 I, r1 I].
// A lane's 6 NC words are uint4 units, unit-major over the warp's 32 lanes (512 B per load instruction), plus a uint2
// tail when 6 NC % 4 == 2. The decoded pair words are the mma A fragment as they are: rows n / n + 8 of the m16 tile =
// rows 0 / 1 of group n, k slots {2q, 2q+1, 2q+8, 2q+9} = elements {0,1,2,3} (first mma) / {4..7} (second) of the
// lane's 8-element block; B = x of token n at the same elements (one 16 B load per chunk feeds both mmas). Every row
// count (1..8 tokens = the n8 columns) runs this one kernel and a token's column never sees the other columns, so a
// verify row is bitwise the plain step's result. The 8 warp partials add in warp order through shared memory.
// Included from ink.cu (namespace ink).

// host: G15 planes (encode12 layout) -> fragment-ordered planes; tiles padded to 8 groups (zero words)
// S: K parts per tile (warp w of 8 S: lanes 4 (w % 8) .., chunks [w / 8 * NC / S, ..))
static std::vector<uint32_t> repack12T(const Enc12& e, int K, int S) {
    const int NC = K / 256, WD = 2 * NC, NF = WD / 4, TL = WD % 4, NCs = NC / S, NW = 6 * NCs, NU = NW / 4;
    const int G = e.G, NT = (G + 7) / 8;
    std::vector<uint32_t> o((size_t)NT * 8 * S * NW * 32, 0u);
    const uint32_t* smw = (const uint32_t*)e.sm.data();
    for (int g = 0; g < G; g++) {
        const int b = g / 8, n = g % 8;
        for (int l = 0; l < 32; l++) {
            const int w = l / 4, q = l % 4, L = 4 * n + q;
            uint32_t* blk = nullptr;
            auto put = [&](int wi, uint32_t v) {
                if (wi < NU * 4) blk[((wi / 4) * 32 + L) * 4 + wi % 4] = v;
                else blk[NU * 128 + L * 2 + (wi - NU * 4)] = v;
            };
            const uint32_t* ix = e.idx.data() + (size_t)g * 32 * WD;
            for (int c = 0; c < NC; c++) {
                const int cc = c % NCs;
                blk = o.data() + (size_t)(b * 8 * S + (c / NCs) * 8 + w) * NW * 32;
                for (int r = 0; r < 2; r++) {
                    const int t = r * NC + c, by = t * 8;   // the lane's sm byte stream
                    const size_t u = ((size_t)g * NC + by / 16) * 32 + l;
                    put(6 * cc + 2 * r, smw[u * 4 + (by % 16) / 4]);
                    put(6 * cc + 2 * r + 1, smw[u * 4 + (by % 16) / 4 + 1]);
                    const uint32_t I = t < NF * 4 ? ix[((t / 4) * 32 + l) * 4 + t % 4] : ix[NF * 128 + l * TL + (t - NF * 4)];
                    put(6 * cc + 4 + r, I);
                }
            }
        }
    }
    return o;
}

// [ink-n4-int8] INT8 g128 planes (INK_DENSE8, TP ranks): per lane and 256-element chunk one uint4 {row 0 bytes 0-3,
// 4-7, row 1 bytes 0-3, 4-7} (q + 128) at ((b NC + c) 8 + w) 32 + lane, the fp16 scales of the lane's 128-group as one
// word {row 0, row 1} at ((b NC + c) 2 + w / 4) 8 + n (ga.idx), ga.meta / ga.fb as G15 (fallback groups: raw BF16
// rows). A pair word = bf16_rn(q s) of 2 elements: the same BF16 weights as the dequantised matrix (q s exact in fp32),
// so every mma below sees the bits of the G15 planes of that matrix.
template <int J>
__device__ __forceinline__ uint32_t d8pair(uint32_t a, float s) {
    const float f0 = __int_as_float(__byte_perm(a, 0x4B000000u, 0x7540 + 2 * J)) - 8388736.f;   // 2^23 + 128
    const float f1 = __int_as_float(__byte_perm(a, 0x4B000000u, 0x7541 + 2 * J)) - 8388736.f;
    const __nv_bfloat162 h = __floats2bfloat162_rn(f0 * s, f1 * s);
    return *(const uint32_t*)&h;
}
__device__ __forceinline__ void d8dec(const uint4 u, uint32_t sw, uint32_t* p0, uint32_t* p1) {
    const float s0 = __half2float(__ushort_as_half((unsigned short)(sw & 0xffffu))), s1 = __half2float(__ushort_as_half((unsigned short)(sw >> 16)));
    p0[0] = d8pair<0>(u.x, s0);
    p0[1] = d8pair<1>(u.x, s0);
    p0[2] = d8pair<0>(u.y, s0);
    p0[3] = d8pair<1>(u.y, s0);
    p1[0] = d8pair<0>(u.z, s1);
    p1[1] = d8pair<1>(u.z, s1);
    p1[2] = d8pair<0>(u.w, s1);
    p1[3] = d8pair<1>(u.w, s1);
}
// [ink-pf8] prefill from the INT8 planes: BF16 rows [N][K] (row-major, the k_dec12T2 output layout) of an INT8 g128
// matrix (up8 layout), each element bf16_rn(q s) through d8dec (the bits every INT8 GEMV feeds its mma) and the raw BF16
// rows of the fallback groups: the G15 planes of the same dequantised matrix decode to the same words (up8 round trip)
__global__ void __launch_bounds__(256) k_dec8(G12Args ga, uint32_t* out, int K) {
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    const int n = lane >> 2, b = blockIdx.x;
    const int g = b * 8 + n, l = 4 * w + (lane & 3), NC = K / 256;
    if (g >= (ga.a.N + 1) / 2) return;
    const uint2 m = *(const uint2*)(ga.meta + g);
    if (((m.x >> 16) & 0xff) == 255) {
        for (int r = 0; r < 2; r++)
            for (int c = 0; c < NC; c++) {
                const size_t e = (size_t)r * K + c * 256 + l * 8;
                *(uint4*)(out + ((size_t)2 * g * K + e) / 2) = __ldg((const uint4*)(ga.fb + (size_t)m.y * 2 * K + e));
            }
        return;
    }
    for (int c = 0; c < NC; c++) {
        const uint4 u = __ldg(ga.sm + (((size_t)b * NC + c) * 8 + w) * 32 + lane);
        const uint32_t sw = __ldg(ga.idx + (((size_t)b * NC + c) * 2 + (w >> 2)) * 8 + n);
        uint32_t p0[4], p1[4];
        d8dec(u, sw, p0, p1);
        *(uint4*)(out + ((size_t)2 * g * K + c * 256 + l * 8) / 2) = make_uint4(p0[0], p0[1], p0[2], p0[3]);
        *(uint4*)(out + ((size_t)(2 * g + 1) * K + c * 256 + l * 8) / 2) = make_uint4(p1[0], p1[1], p1[2], p1[3]);
    }
}
struct A_g12tc {
    G12Args ga;     // ga.sm = fragment-ordered planes (repack12T), ga.idx unused
    int nr, ldx, ldy;
    float* hist;    // MODE 2: raw rows (verify conv history)
    const int* rt = nullptr;   // MODE 2: row table (RT_*): the chain restarts from the slot's state at each segment
    const int* rmap = nullptr; // [ink-tp-dec1] MODE 0: output row of matrix row r = rmap[r] (a rank's gathered rows), null: r
};
// MODE 0: y[j * ldy + row] = dot; MODE 1: one token, conv1 epilogue (state shift); MODE 2: nr tokens, conv chain from
// the pre-step state, hist rows (the k_gemv12r1R epilogue). LO: loads only (gate build).
#ifndef INK_QS
#define INK_QS 1   // [ink-tp-qs] 3: K = H GEMVs in 3 K parts (🟡: plain -0.1 ms, c = 1 -0.1, but 24 rows +1.6 / NTG 1 +0.26 ms, drafter c = 8 +0.37)
#endif
constexpr int tcr_SH = INK_QS;
template <int K> constexpr int tcr_S = K == QD ? 2 : K == H ? tcr_SH : 1;   // K parts per tile (wo: 16 warps of 6 chunks; [ink-tp-qs] K = H: 3 parts of 3 chunks)
#ifndef INK_QSQ
#define INK_QSQ 3   // [ink-vp-qs] K parts of the qkv planes only (the drafter's K = H GEMVs keep tcr_S<H>); 1: today's form
#endif
constexpr int tcr_SQ = INK_QSQ;
static_assert(tcr_SQ == 1 || tcr_SQ == 3, "qkv K parts");
// NTG ([ink-tcr-dec1]): column tiles per weight decode (rows > 8: the chunk's pairs decoded once for NTG tiles; each
// tile's mma sequence over the chunks is the NTG = 1 one, so a column's bits do not depend on NTG)
template <int K, int MODE, bool LO = false, int S = tcr_S<K>, int NTG = 1, bool I8 = false>
__global__ void __launch_bounds__(256 * S) k_g12tc(const __grid_constant__ A_g12tc A_) {
    constexpr int NC = K / 256 / S, NW = I8 ? 4 : 6 * NC, NU = NW / 4, TW = NW % 4;
    static_assert(!I8 || !LO, "INT8 planes: no loads-only form");   // [ink-1c8] S > 1: warp w = lanes of w % 8, chunks of part w / 8
    static_assert(TW == 0 || TW == 2, "tail");
    static_assert((K / 256) % S == 0, "K parts");   // review r3 L1
    const G12Args& ga = A_.ga;
    const GemvArgs& a = ga.a;
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    const int n = lane >> 2, q = lane & 3, b = blockIdx.x;
    const int g = b * 8 + n, l = 4 * (w & 7) + q, c0 = (w >> 3) * NC;
    const int G = (a.N + 1) / 2;
    const uint4* base = ga.sm + (size_t)(b * 8 * S + w) * NW * 8;
    uint32_t W[NW];
    uint4 W8[I8 ? NC : 1];
    uint32_t SW[I8 ? NC : 1];
    if constexpr (I8) {
        const uint4* b8 = ga.sm + (((size_t)b * NC * S + c0) * 8 + (w & 7)) * 32 + lane;
#pragma unroll
        for (int c = 0; c < NC; c++) W8[c] = b8[(size_t)c * 256];
#pragma unroll
        for (int c = 0; c < NC; c++) SW[c] = ga.idx[(((size_t)b * NC * S + c0 + c) * 2 + ((w & 7) >> 2)) * 8 + n];
    } else {
        uint4 u[NU];
#pragma unroll
        for (int i = 0; i < NU; i++) u[i] = base[i * 32 + lane];
#pragma unroll
        for (int i = 0; i < NU; i++) {
            W[4 * i] = u[i].x;
            W[4 * i + 1] = u[i].y;
            W[4 * i + 2] = u[i].z;
            W[4 * i + 3] = u[i].w;
        }
        if (TW) {
            const uint2 t = ((const uint2*)(base + NU * 32))[lane];
            W[NU * 4] = t.x;
            W[NU * 4 + 1] = t.y;
        }
    }
    const uint4 m = g < G ? ga.meta[g] : make_uint4(0, 0, 0, 0);
    const int nr = A_.nr;
    __shared__ float red[8 * S][32][4];
    __shared__ float cv[16][MROW];
    const int row0 = 16 * b, rowL = min(row0 + 15, a.N - 1);
    auto xg = [&](int s) { return s == 0 ? a.x0 : s == 1 ? a.x1 : a.x2; };
    auto bnd = [&](int s) { return s == 0 ? 0 : s == 1 ? a.r1 : s == 2 ? a.r2 : a.N; };
    auto seg = [&](int r) { return r < a.r1 ? 0 : r < a.r2 ? 1 : 2; };
    const uint32_t nf = (m.x >> 16) & 0xff;
    const uint32_t b0 = m.x & 0xff, b1 = (m.x >> 8) & 0xff;
    const uint32_t Bv0 = (b0 << 7) | (b0 << 23), Bv1 = (b1 << 7) | (b1 << 23);
    uint32_t E0 = 0u, E1 = 0u;
    {
        const uint32_t ent[6] = {m.y & 0xffff, m.y >> 16, m.z & 0xffff, m.z >> 16, m.w & 0xffff, m.w >> 16};
#pragma unroll
        for (int s = 0; s < 6; s++) {
            const uint32_t e = ent[s], ex = e >> 6;
            const bool mine = s < (int)nf && nf != 255 && (e & 31) == (uint32_t)l;
            const uint32_t v = (ex << 7) | (ex << 23);
            if ((e >> 5) & 1) E1 = mine ? v : E1;
            else E0 = mine ? v : E0;
        }
    }
    const bf16* fbr = ga.fb + (size_t)m.y * 2 * K + (size_t)c0 * 256;
    const int s_hi = seg(rowL);
#pragma unroll 1
    for (int s = seg(row0); s <= s_hi;) {   // a tile across distinct inputs runs once per input (rare)
        int e = s;
        while (e < s_hi && xg(e + 1) == xg(s)) e++;
        const int lo = bnd(s), hi = bnd(e + 1);
        // [ink-b-rows] column tiles of 8 tokens (rows > 8): the same per-column arithmetic in every tile
        // [ink-tcr-dec1] groups of NTG tiles share one decode of each chunk
#pragma unroll 1
        for (int cg = 0; cg * 8 * NTG < nr; cg++) {
        int tnk[NTG];
        const bf16* xtk[NTG];
        float d0k[NTG][4], d1k[NTG][4];
#pragma unroll
        for (int k = 0; k < NTG; k++) {
            tnk[k] = (cg * NTG + k) * 8 + n;
            xtk[k] = xg(s) + (size_t)(tnk[k] < nr ? tnk[k] : 0) * A_.ldx + (size_t)c0 * 256;
#pragma unroll
            for (int i = 0; i < 4; i++) d0k[k][i] = d1k[k][i] = 0.f;
        }
#pragma unroll
        for (int c = 0; c < NC; c++) {
            uint4 xqk[NTG];
#pragma unroll
            for (int k = 0; k < NTG; k++) xqk[k] = tnk[k] < nr ? *(const uint4*)(xtk[k] + (c * 32 + l) * 8) : make_uint4(0, 0, 0, 0);
            if constexpr (LO) {
                d0k[0][0] += __uint_as_float((W[6 * c] ^ W[6 * c + 1] ^ W[6 * c + 2] ^ W[6 * c + 3] ^ W[6 * c + 4] ^ W[6 * c + 5] ^ xqk[0].x) & 0x3f00ffffu);
                continue;
            }
            uint32_t p0[4], p1[4];
            if (nf == 255) {   // fallback group: raw BF16 rows
                const uint4 r0 = *(const uint4*)(fbr + (c * 32 + l) * 8), r1 = *(const uint4*)(fbr + K + (c * 32 + l) * 8);
                p0[0] = r0.x; p0[1] = r0.y; p0[2] = r0.z; p0[3] = r0.w;
                p1[0] = r1.x; p1[1] = r1.y; p1[2] = r1.z; p1[3] = r1.w;
            } else if constexpr (I8) {
                d8dec(W8[c], SW[c], p0, p1);
            } else {
                const uint32_t a0 = W[6 * c], a1 = W[6 * c + 1], c0 = W[6 * c + 2], c1 = W[6 * c + 3], I0 = W[6 * c + 4], I1 = W[6 * c + 5];
                p0[0] = dec_pair<0>(a0, I0, Bv0, E0);
                p0[1] = dec_pair<1>(a0, I0, Bv0, E0);
                p0[2] = dec_pair<2>(a1, I0, Bv0, E0);
                p0[3] = dec_pair<3>(a1, I0, Bv0, E0);
                p1[0] = dec_pair<0>(c0, I1, Bv1, E1);
                p1[1] = dec_pair<1>(c0, I1, Bv1, E1);
                p1[2] = dec_pair<2>(c1, I1, Bv1, E1);
                p1[3] = dec_pair<3>(c1, I1, Bv1, E1);
            }
#pragma unroll
            for (int k = 0; k < NTG; k++)
                if (k == 0 || (cg * NTG + k) * 8 < nr) {
                    mma_bf16(d0k[k], p0[0], p1[0], p0[1], p1[1], xqk[k].x, xqk[k].y);
                    mma_bf16(d1k[k], p0[2], p1[2], p0[3], p1[3], xqk[k].z, xqk[k].w);
                }
        }
#pragma unroll
        for (int k = 0; k < NTG; k++) {
        const int ct = cg * NTG + k;
        if (ct * 8 >= nr) break;
#pragma unroll
        for (int i = 0; i < 4; i++) red[w][lane][i] = d0k[k][i] + d1k[k][i];
        __syncthreads();
        if (threadIdx.x < 128) {   // output (row 16 b + 2 n' + r, token t): lane 4 n' + t / 2 of each warp, element 2 r + t % 2
            const int t = ct * 8 + (threadIdx.x & 7), r = (threadIdx.x >> 3) & 1, nn = threadIdx.x >> 4;
            const int src = 4 * nn + ((t & 7) >> 1), el = 2 * r + (t & 1);
            float v = red[0][src][el];
#pragma unroll
            for (int ww = 1; ww < 8 * S; ww++) v += red[ww][src][el];
            const int row = row0 + 2 * nn + r;
            const bool ok = t < nr && row >= lo && row < hi && row < a.N;
            if (MODE == 0) {
                if (ok) a.y[(size_t)t * A_.ldy + (A_.rmap ? A_.rmap[row] : row)] = v;
            } else if (MODE == 1) {
                if (ok) a.y[row] = conv1(v, row, a.cst, a.cw, a.add);
            } else if (t < nr)
                cv[2 * nn + r][t] = v;
        }
        __syncthreads();   // red is reused by the next tile
        }   // k
        }   // cg
        if (MODE == 2) {
            __syncthreads();
            if (threadIdx.x < 16) {
                const int c = row0 + threadIdx.x;
                if (c >= lo && c < hi && c < a.N) {
                    float p3 = 0.f, p2 = 0.f, p1 = 0.f;
                    for (int j = 0; j < nr; j++) {
                        if (A_.rt[RT_SEG + j] == j) {
                            const float* cs = a.cst + (size_t)A_.rt[RT_SLOT + j] * 3 * H;
                            p3 = cs[c], p2 = cs[H + c], p1 = cs[2 * H + c];
                        }
                        const float x0 = cv[threadIdx.x][j];
                        const float o = conv_core(x0, p1, p2, p3, a.cw, c);
                        a.y[(size_t)j * H + c] = (a.add ? a.add[(size_t)j * H + c] : 0.f) + o;
                        A_.hist[(size_t)j * H + c] = x0;
                        p3 = p2;
                        p2 = p1;
                        p1 = x0;
                    }
                }
            }
        }
        __syncthreads();
        s = e + 1;
    }
}

// prefill: BF16 rows of a fragment-ordered G15 matrix (k_dec12 for the repacked planes); same block / lane map
// [ink-f52-q8occ] INT8 qkv GEMV for <= 8 rows (MODE 0) at 4 blocks per SM: k_g12tc<K, 0, false, 1, 1, true> compiled
// to 166 registers = ONE 256-thread block per SM (a TP2 rank's 269 tiles in 3.3 waves; a register cap on it spills).
// Here the chunk loop is rolled and a lane's data comes through a ring of INK_Q8PF chunks in flight, so the kernel
// fits INK_Q8MB blocks per SM (all tiles resident in one wave).
// [ink-f52-q8fb] ring entry = two uint4: {codes, scale word} or, for a lane of a fallback group, the raw BF16 chunks of
// its two rows. The raw rows were loaded inside the chunk loop (one DRAM round trip per chunk under a saturated bus:
// the tile with the BF16 sink rows ended the kernel ~1.5 us late); now they are in flight like the codes. The output
// row map is read at the start (was a cold load at the end).
// Per chunk the same d8dec and the same two mma_bf16 in the same order, the same warp-order sum: bitwise
// k_g12tc<.., I8> (load self-test d8_selftest).
#ifndef INK_Q8MB
#define INK_Q8MB 4
#endif
#ifndef INK_Q8PF
#define INK_Q8PF 2
#endif
template <int K>
__global__ void __launch_bounds__(256, INK_Q8MB) k_q8tc(const __grid_constant__ A_g12tc A_) {
    constexpr int NC = K / 256, PF = INK_Q8PF;
    static_assert(PF >= 1 && PF <= NC, "ring depth");
    const G12Args& ga = A_.ga;
    const GemvArgs& a = ga.a;
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    const int n = lane >> 2, q = lane & 3, b = blockIdx.x;
    const int g = b * 8 + n, l = 4 * w + q;
    const uint4* const b8 = ga.sm + ((size_t)b * NC * 8 + w) * 32 + lane;           // + 256 c
    const uint32_t* const si = ga.idx + ((size_t)b * NC * 2 + (w >> 2)) * 8 + n;    // + 16 c
    uint4 wa[PF], wb[PF];
#pragma unroll
    for (int i = 0; i < PF; i++) wa[i] = b8[(size_t)i * 256];
#pragma unroll
    for (int i = 0; i < PF; i++) wb[i] = make_uint4(si[i * 16], 0u, 0u, 0u);
    uint32_t nf = 0, fy = 0;
    if (g < (a.N + 1) / 2) {
        const uint2 m = *(const uint2*)(ga.meta + g);
        nf = (m.x >> 16) & 0xff;
        fy = m.y;
    }
    const int row0 = 16 * b, rowL = min(row0 + 15, a.N - 1);
    // output thread (row 16 b + 2 n' + r, token t): its row and the row's place in y
    const int ot = threadIdx.x & 7, orow = row0 + 2 * (threadIdx.x >> 4) + ((threadIdx.x >> 3) & 1);
    int oy = orow;
    if (threadIdx.x < 128 && orow < a.N && A_.rmap) oy = A_.rmap[orow];
    const bool fbk = nf == 255;
    const uint4* const f0 = (const uint4*)(ga.fb + (size_t)fy * 2 * K + (size_t)l * 8);   // + 32 c; row 1: + K / 8 uint4
    const int nr = A_.nr;
    __shared__ float red[8][32][4];
    auto xg = [&](int s) { return s == 0 ? a.x0 : s == 1 ? a.x1 : a.x2; };
    auto bnd = [&](int s) { return s == 0 ? 0 : s == 1 ? a.r1 : s == 2 ? a.r2 : a.N; };
    auto seg = [&](int r) { return r < a.r1 ? 0 : r < a.r2 ? 1 : 2; };
    const int s_hi = seg(rowL);
    bool first = true;
#pragma unroll 1
    for (int s = seg(row0); s <= s_hi;) {   // a tile across distinct inputs runs once per input (rare)
        int e = s;
        while (e < s_hi && xg(e + 1) == xg(s)) e++;
        const int lo = bnd(s), hi = bnd(e + 1);
        const bool xon = n < nr;
        const uint4* const xt = (const uint4*)(xg(s) + (size_t)(xon ? n : 0) * A_.ldx + (size_t)l * 8);   // + 32 c (uint4 units)
        if (fbk) {   // the lane's raw rows instead of the (filler) codes
#pragma unroll
            for (int i = 0; i < PF; i++) {
                wa[i] = f0[i * 32];
                wb[i] = f0[K / 8 + i * 32];
            }
        } else if (!first) {   // the rare second run reloads the ring
#pragma unroll
            for (int i = 0; i < PF; i++) {
                wa[i] = b8[(size_t)i * 256];
                wb[i] = make_uint4(si[i * 16], 0u, 0u, 0u);
            }
        }
        first = false;
        float d0[4] = {0.f, 0.f, 0.f, 0.f}, d1[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll 1
        for (int c = 0; c < NC; c++) {
            const uint4 u = wa[0], v = wb[0];
#pragma unroll
            for (int i = 0; i + 1 < PF; i++) {
                wa[i] = wa[i + 1];
                wb[i] = wb[i + 1];
            }
            if (c + PF < NC) {
                if (fbk) {
                    wa[PF - 1] = f0[(c + PF) * 32];
                    wb[PF - 1] = f0[K / 8 + (c + PF) * 32];
                } else {
                    wa[PF - 1] = b8[(size_t)(c + PF) * 256];
                    wb[PF - 1] = make_uint4(si[(c + PF) * 16], 0u, 0u, 0u);
                }
            }
            const uint4 xq = xon ? xt[c * 32] : make_uint4(0, 0, 0, 0);
            uint32_t p0[4], p1[4];
            if (fbk) {   // fallback group: raw BF16 rows
                p0[0] = u.x; p0[1] = u.y; p0[2] = u.z; p0[3] = u.w;
                p1[0] = v.x; p1[1] = v.y; p1[2] = v.z; p1[3] = v.w;
            } else
                d8dec(u, v.x, p0, p1);
            mma_bf16(d0, p0[0], p1[0], p0[1], p1[1], xq.x, xq.y);
            mma_bf16(d1, p0[2], p1[2], p0[3], p1[3], xq.z, xq.w);
        }
#pragma unroll
        for (int i = 0; i < 4; i++) red[w][lane][i] = d0[i] + d1[i];
        __syncthreads();
        if (threadIdx.x < 128) {   // output (row 16 b + 2 n' + r, token t): lane 4 n' + t / 2 of each warp, element 2 r + t % 2
            const int src = 4 * (threadIdx.x >> 4) + (ot >> 1), el = 2 * ((threadIdx.x >> 3) & 1) + (ot & 1);
            float v = red[0][src][el];
#pragma unroll
            for (int ww = 1; ww < 8; ww++) v += red[ww][src][el];
            if (ot < nr && orow >= lo && orow < hi && orow < a.N) a.y[(size_t)ot * A_.ldy + oy] = v;
        }
        __syncthreads();   // red is reused by the next run
        s = e + 1;
    }
}

// [ink-1c8] one-card INT8 wo GEMV (K = QD, the full matrix) with the conv epilogues, in the k_q8tc form: 256 threads,
// rolled chunk loop, a ring of INK_Q8WPF chunks in flight (codes + scale word, or a fallback group's raw rows), 2 blocks
// per SM = all 144 tiles resident in one wave (k_g12tc<QD, MODE, .., S, NTG, I8> is a 512-thread block at 128 registers
// = one block per SM, 1.8 waves). A warp walks the S K parts one after the other and leaves one partial per part
// (red[part 8 + w]), so the sum over the 8 S partials is k_g12tc's (warp order of the S-part block): bitwise that kernel.
// MODE 1 (plain step, nr = 1): conv1 + state shift; the conv state / weights / residual of a row are read by its output
// thread at kernel start (in flight under the GEMV, not cold loads at its end). MODE 2 (row step): raw rows to hist
// and the conv chain from the slot states (the k_g12tc epilogue, verbatim). NTG: column tiles per decode (rows > 8).
// One input (a.x0; the wo call has no row segments).
#ifndef INK_Q8WPF
#define INK_Q8WPF 2
#endif
template <int MODE, int NTG = 1, int PF = INK_Q8WPF>
__global__ void __launch_bounds__(256, 2) k_q8w(const __grid_constant__ A_g12tc A_) {
    constexpr int K = QD, S = tcr_S<QD>, NC = K / 256, NCs = NC / S;
    static_assert(MODE == 1 || MODE == 2, "conv epilogue forms");
    static_assert(PF >= 1 && PF <= NC && NC % S == 0, "ring depth / K parts");
    const G12Args& ga = A_.ga;
    const GemvArgs& a = ga.a;
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    const int n = lane >> 2, q = lane & 3, b = blockIdx.x;
    const int g = b * 8 + n, l = 4 * w + q;
    const uint4* const b8 = ga.sm + ((size_t)b * NC * 8 + w) * 32 + lane;           // + 256 c
    const uint32_t* const si = ga.idx + ((size_t)b * NC * 2 + (w >> 2)) * 8 + n;    // + 16 c
    uint4 wa[PF], wb[PF];
#pragma unroll
    for (int i = 0; i < PF; i++) wa[i] = b8[(size_t)i * 256];
#pragma unroll
    for (int i = 0; i < PF; i++) wb[i] = make_uint4(si[i * 16], 0u, 0u, 0u);
    uint32_t nf = 0, fy = 0;
    if (g < (a.N + 1) / 2) {
        const uint2 m = *(const uint2*)(ga.meta + g);
        nf = (m.x >> 16) & 0xff;
        fy = m.y;
    }
    const int row0 = 16 * b, nr = A_.nr;
    // MODE 1: the output thread of row orow (token 0) and its conv operands
    const int orow = row0 + 2 * (threadIdx.x >> 4) + ((threadIdx.x >> 3) & 1);
    const bool othr = MODE == 1 && threadIdx.x < 128 && (threadIdx.x & 7) == 0 && orow < a.N;
    float cx1 = 0.f, cx2 = 0.f, cx3 = 0.f, cw0 = 0.f, cw1 = 0.f, cw2 = 0.f, cw3 = 0.f, cad = 0.f;
    if (othr) {
        cx1 = a.cst[2 * H + orow], cx2 = a.cst[H + orow], cx3 = a.cst[orow];
        cw0 = a.cw[orow * 4], cw1 = a.cw[orow * 4 + 1], cw2 = a.cw[orow * 4 + 2], cw3 = a.cw[orow * 4 + 3];
        cad = a.add ? a.add[orow] : 0.f;
    }
    const bool fbk = nf == 255;
    const uint4* const f0 = (const uint4*)(ga.fb + (size_t)fy * 2 * K + (size_t)l * 8);   // + 32 c; row 1: + K / 8 uint4
    __shared__ float red[NTG][8 * S][32][4];
    __shared__ float cv[MODE == 2 ? 16 : 1][MROW];
#pragma unroll 1
    for (int cg = 0; cg * 8 * NTG < nr; cg++) {   // column tiles of 8 tokens, NTG tiles per decode
        const uint4* xtk[NTG];
        bool xon[NTG];
        float d0k[NTG][4], d1k[NTG][4];
#pragma unroll
        for (int k = 0; k < NTG; k++) {
            const int tn = (cg * NTG + k) * 8 + n;
            xon[k] = tn < nr;
            xtk[k] = (const uint4*)(a.x0 + (size_t)(xon[k] ? tn : 0) * A_.ldx + (size_t)l * 8);   // + 32 c
#pragma unroll
            for (int i = 0; i < 4; i++) d0k[k][i] = d1k[k][i] = 0.f;
        }
        if (fbk) {   // the lane's raw rows instead of the (filler) codes
#pragma unroll
            for (int i = 0; i < PF; i++) {
                wa[i] = f0[i * 32];
                wb[i] = f0[K / 8 + i * 32];
            }
        } else if (cg) {   // a further tile group reloads the ring
#pragma unroll
            for (int i = 0; i < PF; i++) {
                wa[i] = b8[(size_t)i * 256];
                wb[i] = make_uint4(si[i * 16], 0u, 0u, 0u);
            }
        }
#pragma unroll 1
        for (int c = 0; c < NC; c++) {
            const uint4 u = wa[0], v = wb[0];
#pragma unroll
            for (int i = 0; i + 1 < PF; i++) {
                wa[i] = wa[i + 1];
                wb[i] = wb[i + 1];
            }
            if (c + PF < NC) {
                if (fbk) {
                    wa[PF - 1] = f0[(c + PF) * 32];
                    wb[PF - 1] = f0[K / 8 + (c + PF) * 32];
                } else {
                    wa[PF - 1] = b8[(size_t)(c + PF) * 256];
                    wb[PF - 1] = make_uint4(si[(c + PF) * 16], 0u, 0u, 0u);
                }
            }
            uint4 xqk[NTG];
#pragma unroll
            for (int k = 0; k < NTG; k++) xqk[k] = xon[k] ? xtk[k][c * 32] : make_uint4(0, 0, 0, 0);
            uint32_t p0[4], p1[4];
            if (fbk) {   // fallback group: raw BF16 rows
                p0[0] = u.x; p0[1] = u.y; p0[2] = u.z; p0[3] = u.w;
                p1[0] = v.x; p1[1] = v.y; p1[2] = v.z; p1[3] = v.w;
            } else
                d8dec(u, v.x, p0, p1);
#pragma unroll
            for (int k = 0; k < NTG; k++)
                if (k == 0 || (cg * NTG + k) * 8 < nr) {
                    mma_bf16(d0k[k], p0[0], p1[0], p0[1], p1[1], xqk[k].x, xqk[k].y);
                    mma_bf16(d1k[k], p0[2], p1[2], p0[3], p1[3], xqk[k].z, xqk[k].w);
                }
            if ((c + 1) % NCs == 0) {   // end of a K part: the warp's partial of that part (k_g12tc: warp part 8 + w)
                const int pw = (c / NCs) * 8 + w;
#pragma unroll
                for (int k = 0; k < NTG; k++)
#pragma unroll
                    for (int i = 0; i < 4; i++) {
                        red[k][pw][lane][i] = d0k[k][i] + d1k[k][i];
                        d0k[k][i] = d1k[k][i] = 0.f;
                    }
            }
        }
        __syncthreads();
        if (threadIdx.x < 128) {   // output (row 16 b + 2 n' + r, token t): lane 4 n' + t / 2 of each warp, element 2 r + t % 2
#pragma unroll
            for (int k = 0; k < NTG; k++) {
                const int ct = cg * NTG + k;
                if (ct * 8 >= nr) break;
                const int t = ct * 8 + (threadIdx.x & 7), r = (threadIdx.x >> 3) & 1, nn = threadIdx.x >> 4;
                const int src = 4 * nn + ((t & 7) >> 1), el = 2 * r + (t & 1);
                float v = red[k][0][src][el];
#pragma unroll
                for (int ww = 1; ww < 8 * S; ww++) v += red[k][ww][src][el];
                if constexpr (MODE == 1) {
                    if (othr) {   // conv1 (ink.cu) + state shift on the operands read at the start
                        // the FMA chain of conv1 (ink.cu spells it out the same way: bitwise by construction, review
                        // L3); d8_selftest still compares the two kernels bit by bit, with and without the residual
                        const float o = __fmaf_rn(cw0, cx3, __fmaf_rn(cw1, cx2, __fmaf_rn(cw3, v, __fmul_rn(cw2, cx1))));
                        const float res = __fmaf_rn(__fadd_rn(o, v), 1.f / sqrtf(5.f), cad);
                        a.cst[orow] = cx2;
                        a.cst[H + orow] = cx1;
                        a.cst[2 * H + orow] = v;
                        a.y[orow] = res;
                    }
                } else if (t < nr)
                    cv[2 * nn + r][t] = v;
            }
        }
        __syncthreads();   // red is reused by the next tile group
    }
    if constexpr (MODE == 2) {
        if (threadIdx.x < 16) {
            const int c = row0 + threadIdx.x;
            if (c < a.N) {
                float p3 = 0.f, p2 = 0.f, p1 = 0.f;
                for (int j = 0; j < nr; j++) {
                    if (A_.rt[RT_SEG + j] == j) {
                        const float* cs = a.cst + (size_t)A_.rt[RT_SLOT + j] * 3 * H;
                        p3 = cs[c], p2 = cs[H + c], p1 = cs[2 * H + c];
                    }
                    const float x0 = cv[threadIdx.x][j];
                    const float o = conv_core(x0, p1, p2, p3, a.cw, c);
                    a.y[(size_t)j * H + c] = (a.add ? a.add[(size_t)j * H + c] : 0.f) + o;
                    A_.hist[(size_t)j * H + c] = x0;
                    p3 = p2;
                    p2 = p1;
                    p1 = x0;
                }
            }
        }
    }
}
// [ink-vp-qs] rows > 8 on planes in the SV-part fragment order (k_g12tc<K, 0, false, SV> layout) with 256 threads: warp
// w does the SV parts' work of the SV-part warps 8 p + w one part after another (each part's mma sequence from 0, as
// that warp's), and the output thread continues one running sum over the 8 SV warp partials in warp order 0 .. 8 SV - 1
// (the k_g12tc sum): bitwise the SV-part kernel, at the register budget of a 256-thread block. MODE 0 only.
template <int K, int SV, int NTG>
__global__ void __launch_bounds__(256) k_g12tcV(const __grid_constant__ A_g12tc A_) {
    constexpr int NC = K / 256 / SV, NW = 6 * NC, NU = NW / 4, TW = NW % 4;
    static_assert(TW == 0 || TW == 2, "tail");
    static_assert((K / 256) % SV == 0, "K parts");
    const G12Args& ga = A_.ga;
    const GemvArgs& a = ga.a;
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    const int n = lane >> 2, q = lane & 3, b = blockIdx.x;
    const int g = b * 8 + n, l = 4 * w + q;
    const int G = (a.N + 1) / 2;
    uint32_t W[SV][NW];
#pragma unroll
    for (int p = 0; p < SV; p++) {
        const uint4* base = ga.sm + (size_t)(b * 8 * SV + 8 * p + w) * NW * 8;
        uint4 u[NU];
#pragma unroll
        for (int i = 0; i < NU; i++) u[i] = base[i * 32 + lane];
#pragma unroll
        for (int i = 0; i < NU; i++) {
            W[p][4 * i] = u[i].x;
            W[p][4 * i + 1] = u[i].y;
            W[p][4 * i + 2] = u[i].z;
            W[p][4 * i + 3] = u[i].w;
        }
        if (TW) {
            const uint2 t = ((const uint2*)(base + NU * 32))[lane];
            W[p][NU * 4] = t.x;
            W[p][NU * 4 + 1] = t.y;
        }
    }
    const uint4 m = g < G ? ga.meta[g] : make_uint4(0, 0, 0, 0);
    const int nr = A_.nr;
    __shared__ float red[NTG][8][32][4];
    const int row0 = 16 * b, rowL = min(row0 + 15, a.N - 1);
    auto xg = [&](int s) { return s == 0 ? a.x0 : s == 1 ? a.x1 : a.x2; };
    auto bnd = [&](int s) { return s == 0 ? 0 : s == 1 ? a.r1 : s == 2 ? a.r2 : a.N; };
    auto seg = [&](int r) { return r < a.r1 ? 0 : r < a.r2 ? 1 : 2; };
    const uint32_t nf = (m.x >> 16) & 0xff;
    const uint32_t b0 = m.x & 0xff, b1 = (m.x >> 8) & 0xff;
    const uint32_t Bv0 = (b0 << 7) | (b0 << 23), Bv1 = (b1 << 7) | (b1 << 23);
    uint32_t E0 = 0u, E1 = 0u;
    {
        const uint32_t ent[6] = {m.y & 0xffff, m.y >> 16, m.z & 0xffff, m.z >> 16, m.w & 0xffff, m.w >> 16};
#pragma unroll
        for (int s = 0; s < 6; s++) {
            const uint32_t e = ent[s], ex = e >> 6;
            const bool mine = s < (int)nf && nf != 255 && (e & 31) == (uint32_t)l;
            const uint32_t v = (ex << 7) | (ex << 23);
            if ((e >> 5) & 1) E1 = mine ? v : E1;
            else E0 = mine ? v : E0;
        }
    }
    const bf16* fbr = ga.fb + (size_t)m.y * 2 * K;
    const int s_hi = seg(rowL);
#pragma unroll 1
    for (int s = seg(row0); s <= s_hi;) {
        int e = s;
        while (e < s_hi && xg(e + 1) == xg(s)) e++;
        const int lo = bnd(s), hi = bnd(e + 1);
#pragma unroll 1
        for (int cg = 0; cg * 8 * NTG < nr; cg++) {
            int tnk[NTG];
            const bf16* xtk[NTG];
#pragma unroll
            for (int k = 0; k < NTG; k++) {
                tnk[k] = (cg * NTG + k) * 8 + n;
                xtk[k] = xg(s) + (size_t)(tnk[k] < nr ? tnk[k] : 0) * A_.ldx;
            }
            float acc[NTG];   // output threads: the running sum of the warp partials (k_g12tc's order)
#pragma unroll
            for (int k = 0; k < NTG; k++) acc[k] = 0.f;
#pragma unroll
            for (int p = 0; p < SV; p++) {
                float d0k[NTG][4], d1k[NTG][4];
#pragma unroll
                for (int k = 0; k < NTG; k++)
#pragma unroll
                    for (int i = 0; i < 4; i++) d0k[k][i] = d1k[k][i] = 0.f;
#pragma unroll
                for (int c = 0; c < NC; c++) {
                    const int cc = p * NC + c;
                    uint4 xqk[NTG];
#pragma unroll
                    for (int k = 0; k < NTG; k++) xqk[k] = tnk[k] < nr ? *(const uint4*)(xtk[k] + (cc * 32 + l) * 8) : make_uint4(0, 0, 0, 0);
                    uint32_t p0[4], p1[4];
                    if (nf == 255) {
                        const uint4 r0 = *(const uint4*)(fbr + (cc * 32 + l) * 8), r1 = *(const uint4*)(fbr + K + (cc * 32 + l) * 8);
                        p0[0] = r0.x; p0[1] = r0.y; p0[2] = r0.z; p0[3] = r0.w;
                        p1[0] = r1.x; p1[1] = r1.y; p1[2] = r1.z; p1[3] = r1.w;
                    } else {
                        const uint32_t a0 = W[p][6 * c], a1 = W[p][6 * c + 1], c0 = W[p][6 * c + 2], c1 = W[p][6 * c + 3], I0 = W[p][6 * c + 4], I1 = W[p][6 * c + 5];
                        p0[0] = dec_pair<0>(a0, I0, Bv0, E0);
                        p0[1] = dec_pair<1>(a0, I0, Bv0, E0);
                        p0[2] = dec_pair<2>(a1, I0, Bv0, E0);
                        p0[3] = dec_pair<3>(a1, I0, Bv0, E0);
                        p1[0] = dec_pair<0>(c0, I1, Bv1, E1);
                        p1[1] = dec_pair<1>(c0, I1, Bv1, E1);
                        p1[2] = dec_pair<2>(c1, I1, Bv1, E1);
                        p1[3] = dec_pair<3>(c1, I1, Bv1, E1);
                    }
#pragma unroll
                    for (int k = 0; k < NTG; k++)
                        if (k == 0 || (cg * NTG + k) * 8 < nr) {
                            mma_bf16(d0k[k], p0[0], p1[0], p0[1], p1[1], xqk[k].x, xqk[k].y);
                            mma_bf16(d1k[k], p0[2], p1[2], p0[3], p1[3], xqk[k].z, xqk[k].w);
                        }
                }
#pragma unroll
                for (int k = 0; k < NTG; k++)
                    if ((cg * NTG + k) * 8 < nr)
#pragma unroll
                        for (int i = 0; i < 4; i++) red[k][w][lane][i] = d0k[k][i] + d1k[k][i];
                __syncthreads();
                if (threadIdx.x < 128) {
#pragma unroll
                    for (int k = 0; k < NTG; k++) {
                        if ((cg * NTG + k) * 8 >= nr) break;
                        const int t = (cg * NTG + k) * 8 + (threadIdx.x & 7), nn = threadIdx.x >> 4, r = (threadIdx.x >> 3) & 1;
                        const int src = 4 * nn + ((t & 7) >> 1), el = 2 * r + (t & 1);
                        float v = p == 0 ? red[k][0][src][el] : acc[k] + red[k][0][src][el];
#pragma unroll
                        for (int ww = 1; ww < 8; ww++) v += red[k][ww][src][el];
                        acc[k] = v;
                    }
                }
                __syncthreads();   // red is reused by the next part
            }
            if (threadIdx.x < 128) {
#pragma unroll
                for (int k = 0; k < NTG; k++) {
                    const int t = (cg * NTG + k) * 8 + (threadIdx.x & 7), r = (threadIdx.x >> 3) & 1, nn = threadIdx.x >> 4;
                    const int row = row0 + 2 * nn + r;
                    if (t < nr && row >= lo && row < hi && row < a.N) a.y[(size_t)t * A_.ldy + (A_.rmap ? A_.rmap[row] : row)] = acc[k];
                }
            }
        }   // cg
        __syncthreads();
        s = e + 1;
    }
}

template <int K, int S = tcr_S<K>>
__global__ void __launch_bounds__(256 * S) k_dec12T(G12Args ga, uint32_t* out) {
    constexpr int NC = K / 256 / S, NW = 6 * NC, NU = NW / 4, TW = NW % 4;
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    const int n = lane >> 2, q = lane & 3, b = blockIdx.x;
    const int g = b * 8 + n, l = 4 * (w & 7) + q, c0 = (w >> 3) * NC;
    if (g >= (ga.a.N + 1) / 2) return;
    const uint32_t* base = (const uint32_t*)(ga.sm + (size_t)(b * 8 * S + w) * NW * 8);
    auto wd = [&](int i) { return i < NU * 4 ? base[((i / 4) * 32 + lane) * 4 + i % 4] : base[NU * 128 + lane * 2 + (i - NU * 4)]; };
    const uint4 m = ga.meta[g];
    const uint32_t nf = (m.x >> 16) & 0xff;
    const uint32_t b0 = m.x & 0xff, b1 = (m.x >> 8) & 0xff;
    const uint32_t Bv[2] = {(b0 << 7) | (b0 << 23), (b1 << 7) | (b1 << 23)};
    uint32_t ESC[2] = {0u, 0u};
    const uint32_t ent[6] = {m.y & 0xffff, m.y >> 16, m.z & 0xffff, m.z >> 16, m.w & 0xffff, m.w >> 16};
    for (int s = 0; s < 6; s++)
        if (s < (int)nf && nf != 255 && (ent[s] & 31) == (uint32_t)l) ESC[(ent[s] >> 5) & 1] = ((ent[s] >> 6) << 7) | ((ent[s] >> 6) << 23);
    (void)TW;
    for (int r = 0; r < 2; r++)
        for (int c = 0; c < NC; c++) {
            uint32_t* o = out + ((size_t)(2 * g + r) * K + ((c0 + c) * 32 + l) * 8) / 2;
            if (nf == 255) {
                const uint32_t* f = (const uint32_t*)(ga.fb + (size_t)m.y * 2 * K + (size_t)r * K + ((c0 + c) * 32 + l) * 8);
                for (int k = 0; k < 4; k++) o[k] = f[k];
            } else {
                const uint32_t s0 = wd(6 * c + 2 * r), s1 = wd(6 * c + 2 * r + 1), I = wd(6 * c + 4 + r);
                o[0] = dec_pair<0>(s0, I, Bv[r], ESC[r]);
                o[1] = dec_pair<1>(s0, I, Bv[r], ESC[r]);
                o[2] = dec_pair<2>(s1, I, Bv[r], ESC[r]);
                o[3] = dec_pair<3>(s1, I, Bv[r], ESC[r]);
            }
        }
}

// [ink-pf-dec16v] the same decode with vector memory ops: the lane's NW words as NU uint4 units (+ the uint2 tail), one
// 16 B store per (row, chunk) instead of four 4 B stores; output identical to k_dec12T
template <int K, int S = tcr_S<K>>
__global__ void __launch_bounds__(256 * S) k_dec12T2(G12Args ga, uint32_t* out) {
    constexpr int NC = K / 256 / S, NW = 6 * NC, NU = NW / 4, TW = NW % 4;
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    const int n = lane >> 2, q = lane & 3, b = blockIdx.x;
    const int g = b * 8 + n, l = 4 * (w & 7) + q, c0 = (w >> 3) * NC;
    if (g >= (ga.a.N + 1) / 2) return;
    const uint4* base = ga.sm + (size_t)(b * 8 * S + w) * NW * 8;
    uint32_t W[NW];
#pragma unroll
    for (int i = 0; i < NU; i++) {
        const uint4 u = __ldg(base + i * 32 + lane);
        W[4 * i] = u.x;
        W[4 * i + 1] = u.y;
        W[4 * i + 2] = u.z;
        W[4 * i + 3] = u.w;
    }
    if (TW) {
        const uint2 t = __ldg((const uint2*)(base + NU * 32) + lane);
        W[NU * 4] = t.x;
        W[NU * 4 + 1] = t.y;
    }
    const uint4 m = ga.meta[g];
    const uint32_t nf = (m.x >> 16) & 0xff;
    const uint32_t b0 = m.x & 0xff, b1 = (m.x >> 8) & 0xff;
    const uint32_t Bv[2] = {(b0 << 7) | (b0 << 23), (b1 << 7) | (b1 << 23)};
    uint32_t ESC[2] = {0u, 0u};
    const uint32_t ent[6] = {m.y & 0xffff, m.y >> 16, m.z & 0xffff, m.z >> 16, m.w & 0xffff, m.w >> 16};
#pragma unroll
    for (int s = 0; s < 6; s++) {
        const uint32_t e = ent[s], v = ((e >> 6) << 7) | ((e >> 6) << 23);
        const bool mine = s < (int)nf && nf != 255 && (e & 31) == (uint32_t)l;
        if ((e >> 5) & 1) ESC[1] = mine ? v : ESC[1];
        else ESC[0] = mine ? v : ESC[0];
    }
#pragma unroll
    for (int r = 0; r < 2; r++)
#pragma unroll
        for (int c = 0; c < NC; c++) {
            uint4* o = (uint4*)(out + ((size_t)(2 * g + r) * K + ((c0 + c) * 32 + l) * 8) / 2);
            if (nf == 255) *o = __ldg((const uint4*)(ga.fb + (size_t)m.y * 2 * K + (size_t)r * K + ((c0 + c) * 32 + l) * 8));
            else {
                const uint32_t s0 = W[6 * c + 2 * r], s1 = W[6 * c + 2 * r + 1], I = W[6 * c + 4 + r];
                *o = make_uint4(dec_pair<0>(s0, I, Bv[r], ESC[r]), dec_pair<1>(s0, I, Bv[r], ESC[r]), dec_pair<2>(s1, I, Bv[r], ESC[r]),
                                dec_pair<3>(s1, I, Bv[r], ESC[r]));
            }
        }
}

// [ink-tp-dec1] a rank's column block of a G15 matrix (TP-N output projection): columns [256 cb, 256 (cb + nc)) of the
// full matrix, planes from repack12T(e, 256 nc, nc) (one chunk per part), any chunk count at run time; x elements
// outside [xlo, xhi) count as zero (the columns of heads another rank owns). Raw partial y[t * ldy + row], nr tokens in
// column tiles of 8. The same decode and mma sequence per chunk as k_g12tc (8 warps, partials added in warp order).
// [ink-vp-cpush] row steps of <= 3 TPC_RW rows: the row all-reduce inside the producer, column-major. The thread that
// holds output column c of all nr rows stores one 16 B word {3 row values, tag} per 3 rows into each peer's receive
// words [sender][TPC_RW][H] (2,304 stores per collective for nr <= 3, as the plain producer push), waits for the peers'
// words of the same column, adds the N partials of each row in rank order from 0.f (the k_tpart / k_tpartC sum, bitwise)
// and runs the verify conv chain of that column (k_tpchain's per-column expression: y = add + conv, hist = x, the
// slot's state at each segment start). No collective kernel. Push transport, every rank a producer.
constexpr int TPC_RW = 2;            // words per column: rows <= 3 TPC_RW
constexpr int TPC_MAXR = 3 * TPC_RW;
constexpr int TPC_SLOTS = 4;         // receive slots (collective index mod 4)
struct TpCol {
    uint4* dst[TP_MAX - 1];   // each peer's receive words of this rank for the slot [TPC_RW][H]
    const uint4* rd;          // this rank's receive words of the slot [N][TPC_RW][H]
    const unsigned* step;     // this rank's step counter (device)
    const unsigned* ab;       // the context's abort word
    unsigned* chk;            // QE_TP_CHK hashes or null
    float *sum, *y, *hist;    // [nr][H]: the summed rows, the chain's output rows, its raw inputs
    const float* cst;         // conv states [slot][3][H]
    const float* cw;
    const float* add;         // [nr][H] or null
    const int* rt;            // device row table (segments, slots)
    int np, n, rank, idx, nr;   // np 0: off
};
__device__ __forceinline__ unsigned tpc_tag(const TpCol& p) { return p.np ? *p.step * 256u + (unsigned)p.idx + 1u : 0u; }
// column c: v[j] = this rank's partial of row j (j < nr)
__device__ __forceinline__ void tpc_col(const TpCol& p, const unsigned tag, const int c, const float (&v)[TPC_MAXR]) {
    const int nw = (p.nr + 2) / 3;
#pragma unroll
    for (int i = 0; i < TPC_RW; i++)
        if (i < nw)
            for (int g = 0; g < p.np; g++)
                asm volatile("st.volatile.global.v4.u32 [%0], {%1, %2, %3, %4};" ::"l"(p.dst[g] + (size_t)i * H + c), "r"(__float_as_uint(v[3 * i])),
                             "r"(__float_as_uint(v[3 * i + 1])), "r"(__float_as_uint(v[3 * i + 2])), "r"(tag) : "memory");
    // the chain's inputs in flight under the transfer
    const int j00 = p.rt[RT_SEG];
    const float* cs0 = p.cst + (size_t)p.rt[RT_SLOT + j00] * 3 * H;
    const float q3 = cs0[c], q2 = cs0[H + c], q1 = cs0[2 * H + c];
    float s[TPC_MAXR];
#pragma unroll
    for (int j = 0; j < TPC_MAXR; j++) s[j] = 0.f;
    unsigned long long t0 = 0;
    for (int g = 0; g < p.n; g++) {
        if (g == p.rank) {
#pragma unroll
            for (int j = 0; j < TPC_MAXR; j++) s[j] += v[j];
            continue;
        }
#pragma unroll
        for (int i = 0; i < TPC_RW; i++) {
            if (i >= nw) break;
            const uint4* rp = p.rd + ((size_t)g * TPC_RW + i) * H + c;
            uint4 x;
            unsigned it = 0;
            for (;;) {
                asm volatile("ld.volatile.global.v4.u32 {%0, %1, %2, %3}, [%4];" : "=r"(x.x), "=r"(x.y), "=r"(x.z), "=r"(x.w) : "l"(rp) : "memory");
                if (x.w == tag) break;
                if ((++it & 4095u) == 0) {
                    if (*(volatile const unsigned*)p.ab) return;   // aborted: no epilogue
                    if (!t0) t0 = tpf_timer();
                    else if (tpf_timer() - t0 > 20000000000ull) __trap();   // last resort: a peer whose host thread is gone (20 s)
                }
            }
            s[3 * i] += __uint_as_float(x.x);
            s[3 * i + 1] += __uint_as_float(x.y);
            s[3 * i + 2] += __uint_as_float(x.z);
        }
    }
    float p3 = 0.f, p2 = 0.f, p1 = 0.f;
#pragma unroll
    for (int j = 0; j < TPC_MAXR; j++) {
        if (j >= p.nr) break;
        if (p.chk) atomicXor(p.chk + p.idx, (__float_as_uint(s[j]) * (2u * (unsigned)(j * H + c) + 1u)) ^ (unsigned)c);
        if (j == 0) p3 = q3, p2 = q2, p1 = q1;
        else if (p.rt[RT_SEG + j] == j) {
            const float* cs = p.cst + (size_t)p.rt[RT_SLOT + j] * 3 * H;
            p3 = cs[c], p2 = cs[H + c], p1 = cs[2 * H + c];
        }
        const float x0 = s[j];
        const float o = conv_core(x0, p1, p2, p3, p.cw, c);
        p.y[(size_t)j * H + c] = (p.add ? p.add[(size_t)j * H + c] : 0.f) + o;
        p.hist[(size_t)j * H + c] = x0;
        p.sum[(size_t)j * H + c] = x0;
        p3 = p2;
        p2 = p1;
        p1 = x0;
    }
}
struct A_g12tcR {
    G12Args ga;     // ga.sm fragment-ordered planes (S = nc), ga.a.x0 = x (full width), ga.a.y, ga.a.N = rows
    int nr, ldx, ldy, cb, nc, xlo, xhi;
    TpPush pp{};    // [ink-tpf-ppush] plain step (nr 1): every output value also goes to the peers' receive words
    TpCol pc{};     // [ink-vp-cpush] row step (nr <= TPC_MAXR, NTG 1): column-major push + sum + verify conv chain
};
template <int NTG = 1, bool I8 = false>   // [ink-tcr-dec1] column tiles per chunk load + decode (a column's mma sequence does not change)
__global__ void __launch_bounds__(256) k_g12tcR(const __grid_constant__ A_g12tcR A_) {
    const G12Args& ga = A_.ga;
    const GemvArgs& a = ga.a;
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    const int n = lane >> 2, q = lane & 3, b = blockIdx.x;
    const int g = b * 8 + n, l = 4 * w + q;
    const int G = (a.N + 1) / 2, nc = A_.nc, K = 256 * nc;
    const uint4 m = g < G ? ga.meta[g] : make_uint4(0, 0, 0, 0);
    __shared__ float red[8][32][4];
    const int row0 = 16 * b;
    const uint32_t nf = (m.x >> 16) & 0xff;
    const uint32_t b0 = m.x & 0xff, b1 = (m.x >> 8) & 0xff;
    const uint32_t Bv0 = (b0 << 7) | (b0 << 23), Bv1 = (b1 << 7) | (b1 << 23);
    uint32_t E0 = 0u, E1 = 0u;
    {
        const uint32_t ent[6] = {m.y & 0xffff, m.y >> 16, m.z & 0xffff, m.z >> 16, m.w & 0xffff, m.w >> 16};
#pragma unroll
        for (int s = 0; s < 6; s++) {
            const uint32_t e = ent[s], ex = e >> 6;
            const bool mine = s < (int)nf && nf != 255 && (e & 31) == (uint32_t)l;
            const uint32_t v = (ex << 7) | (ex << 23);
            if ((e >> 5) & 1) E1 = mine ? v : E1;
            else E0 = mine ? v : E0;
        }
    }
    const bf16* fbr = ga.fb + (size_t)m.y * 2 * K;
    const unsigned ptag = tp_push_tag(A_.pp);   // [ink-tpf-ppush] loaded up front (in flight under the GEMV)
    const unsigned ctag = NTG == 1 ? tpc_tag(A_.pc) : 0u;   // [ink-vp-cpush]
    TpConvPre cpre;
    if (A_.pp.np && threadIdx.x < 128 && (threadIdx.x & 7) == 0)   // [ink-tpf-pred] the output thread of row row0 + threadIdx.x / 8
        tp_conv_pre(A_.pp, row0 + 2 * (threadIdx.x >> 4) + ((threadIdx.x >> 3) & 1), cpre);
#pragma unroll 1
    for (int cg = 0; cg * 8 * NTG < A_.nr; cg++) {
        int tnk[NTG];
        const bf16* xtk[NTG];
        float d0k[NTG][4], d1k[NTG][4];
#pragma unroll
        for (int k = 0; k < NTG; k++) {
            tnk[k] = (cg * NTG + k) * 8 + n;
            xtk[k] = a.x0 + (size_t)(tnk[k] < A_.nr ? tnk[k] : 0) * A_.ldx + (size_t)A_.cb * 256;
#pragma unroll
            for (int i = 0; i < 4; i++) d0k[k][i] = d1k[k][i] = 0.f;
        }
#pragma unroll 2
        for (int c = 0; c < nc; c++) {
            uint4 u;
            uint2 tl;
            uint32_t sw = 0u;
            if constexpr (I8) {   // [ink-n4-int8]
                u = ga.sm[(((size_t)b * nc + c) * 8 + w) * 32 + lane];
                sw = ga.idx[(((size_t)b * nc + c) * 2 + (w >> 2)) * 8 + n];
                tl = make_uint2(0u, 0u);
            } else {
                const uint4* base = ga.sm + (size_t)(b * 8 * nc + c * 8 + w) * 6 * 8;
                u = base[lane];
                tl = ((const uint2*)(base + 32))[lane];
            }
            const int el = (A_.cb + c) * 256 + l * 8;
            uint4 xqk[NTG];
#pragma unroll
            for (int k = 0; k < NTG; k++)
                xqk[k] = (tnk[k] < A_.nr && el >= A_.xlo && el < A_.xhi) ? *(const uint4*)(xtk[k] + (c * 32 + l) * 8) : make_uint4(0, 0, 0, 0);
            uint32_t p0[4], p1[4];
            if (nf == 255) {
                const uint4 r0 = *(const uint4*)(fbr + c * 256 + l * 8), r1 = *(const uint4*)(fbr + K + c * 256 + l * 8);
                p0[0] = r0.x; p0[1] = r0.y; p0[2] = r0.z; p0[3] = r0.w;
                p1[0] = r1.x; p1[1] = r1.y; p1[2] = r1.z; p1[3] = r1.w;
            } else if constexpr (I8) {
                d8dec(u, sw, p0, p1);
            } else {
                const uint32_t a0 = u.x, a1 = u.y, c0 = u.z, c1 = u.w, I0 = tl.x, I1 = tl.y;
                p0[0] = dec_pair<0>(a0, I0, Bv0, E0);
                p0[1] = dec_pair<1>(a0, I0, Bv0, E0);
                p0[2] = dec_pair<2>(a1, I0, Bv0, E0);
                p0[3] = dec_pair<3>(a1, I0, Bv0, E0);
                p1[0] = dec_pair<0>(c0, I1, Bv1, E1);
                p1[1] = dec_pair<1>(c0, I1, Bv1, E1);
                p1[2] = dec_pair<2>(c1, I1, Bv1, E1);
                p1[3] = dec_pair<3>(c1, I1, Bv1, E1);
            }
#pragma unroll
            for (int k = 0; k < NTG; k++)
                if (k == 0 || (cg * NTG + k) * 8 < A_.nr) {
                    mma_bf16(d0k[k], p0[0], p1[0], p0[1], p1[1], xqk[k].x, xqk[k].y);
                    mma_bf16(d1k[k], p0[2], p1[2], p0[3], p1[3], xqk[k].z, xqk[k].w);
                }
        }
#pragma unroll
        for (int k = 0; k < NTG; k++) {
        const int ct = cg * NTG + k;
        if (ct * 8 >= A_.nr) break;
#pragma unroll
        for (int i = 0; i < 4; i++) red[w][lane][i] = d0k[k][i] + d1k[k][i];
        __syncthreads();
        if (threadIdx.x < 128) {
            const int t = ct * 8 + (threadIdx.x & 7), r = (threadIdx.x >> 3) & 1, nn = threadIdx.x >> 4;
            const int src = 4 * nn + ((t & 7) >> 1), el = 2 * r + (t & 1);
            float v = red[0][src][el];
#pragma unroll
            for (int ww = 1; ww < 8; ww++) v += red[ww][src][el];
            const int row = row0 + 2 * nn + r;
            if (NTG == 1 && A_.pc.np) {   // [ink-vp-cpush] the column's rows from the 8 lanes of its group (warps 0-3 whole here)
                float vv[TPC_MAXR];
                const int gb = threadIdx.x & 24;
#pragma unroll
                for (int j = 0; j < TPC_MAXR; j++) vv[j] = __shfl_sync(0xffffffffu, v, gb + j);
                if ((threadIdx.x & 7) == 0 && row < a.N) tpc_col(A_.pc, ctag, row, vv);
            } else if (t < A_.nr && row < a.N) {
                if (A_.pp.np) tp_push1(A_.pp, ptag, row, v);   // nr == 1 (host rule)
                if (A_.pp.np) {   // [ink-tpf-pred] sum + conv here
                    float sm;
                    if (tp_reduce1(A_.pp, ptag, row, v, sm)) tp_conv_fin(A_.pp, row, sm, cpre);
                } else
                    a.y[(size_t)t * A_.ldy + row] = v;
            }
        }
        __syncthreads();
        }   // k
    }
}
