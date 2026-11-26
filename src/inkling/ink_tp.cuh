// [ink-tp-dec1] TP-N all-reduce of rows of H fp32 with the epilogue that follows the sum (included from ink.cu after
// conv1). Every rank adds the N partials in rank order 0..N-1 (from 0.f), so the sum is bitwise the same on every rank
// and the same in both transports:
//   direct (ranks on one device, "virtual TP"): src[g] = rank g's partial rows; the host orders the launches by events;
//   push (P2P): every rank stores its partial as 16 B words (3 fp32 + tag) into every peer's receive words, then waits
//     for the tags of all peers (gpt-oss k_ar form, any N). tag = step x 256 + collective + 1.
// Word format (review tp M4 / [ink-tp-r2-fmt] / review tp2 M1): the default (INK_TP_W3 = 1) is one 16 B word of 3 fp32 +
// the tag in .w (768 threads, thread w owns elements 3w .. 3w + 2). PTX guarantees single-copy atomicity per 8 B element,
// not per 16 B vector, so this rests on the observed 16 B atomicity of sm_86 stores over PCIe P2P (stress gate QE_TP_CHK:
// 11.8 M collectives, no mismatch; DELIBERATE TRADE-OFF: keep QE_TP_CHK in the release gates and rerun it after driver /
// topology changes). -DINK_TP_W3=0 builds the M4 form (2 fp32, the tag in both 8 B halves; -0.21 ms per step slower).
// The head argmax (k_tpmax, one word per row) always uses the both-halves tag (no cost).
// review tp M2: the spin loops also poll the TP context's abort word (mapped host memory): a failed rank's host thread
// sets it, the peers' kernels leave their waits (no epilogue) and the host recovers the context (tp_recover).
#ifndef INK_TP_W3
#define INK_TP_W3 1   // [ink-tp-r2-fmt] 1 = 16 B word (3 fp32 + tag in .w, 768 threads; default, backed by the
                      // QE_TP_CHK stress gate); 0 = review M4 form (2 fp32, tag in both 8 B halves; -0.21 ms/step slower)
#endif
#ifndef INK_TP_AB
#define INK_TP_AB 1   // r2 A/B: 0 = no abort-word poll in the waits
#endif
constexpr int TPE = INK_TP_W3 ? 3 : 4;   // elements per thread
constexpr int TPW = INK_TP_W3 ? H / 3 : H / 2;      // words per row
constexpr int TPT = H / TPE;      // threads of k_tpar
constexpr int TP_SLOTS = 4;     // receive-word slots of the all-reduces (collective index mod 4)
// + 2 slots for the head's argmax, by step parity: the head is followed by the next step's collective 0 (slot 0), and
// an extra head outside a step follows the step's own head (review of the first TP2 spec run: slot 0 shared by the
// head and the next collective 0 let a fast rank overwrite words a slow rank was still waiting for)
constexpr int TP_SLOTS_ALL = TP_SLOTS + 2;
static_assert(INK_TP_W3 ? (TPW == 768 && TPT == 768) : (TPW == 1152 && TPT == 576), "H = 2304");

struct TpAr {
    const void* src[TP_MAX];   // direct: rank g's partial rows; push: rank g's receive words (uint4)
    const uint4* rd;           // push: this rank's receive words
    int n, rank, push;
    const unsigned* step;      // push: this rank's step counter (device)
    const unsigned* ab;        // push: the context's abort word (mapped host memory), nonzero = leave every wait
    unsigned* chk;             // [ink-tp-r2-fmt] QE_TP_CHK: per-collective xor hash of the sums (nullptr = off)
};
// [ink-tp-r2-fmt] stress gate: every thread hashes its summed values (bits x odd position key), warp xor, one atomic
__device__ __forceinline__ void tp_hash(unsigned* chk, int idx, int row, int c, const float* s, int n) {
    unsigned h = 0;
    for (int i = 0; i < n; i++) h ^= (__float_as_uint(s[i]) * (2u * (unsigned)(row * H + c + i) + 1u)) ^ (unsigned)(c + i);
    const unsigned m = __activemask();   // review tp2 L2: lanes that left on an abort are not in the shuffle (the hash is void then)
    for (int o = 16; o; o >>= 1) h ^= __shfl_xor_sync(m, h, o);
    if ((threadIdx.x & 31) == 0) atomicXor(chk + idx, h);
}
struct A_tpar {
    TpAr t;
    const float* part;   // this rank's partial rows [nr][H]
    int idx, nr;
    int epi;             // 0: out[row][c] = sum; 1: one row, out[c] = conv1(sum, c, cst, cw, add)
    float* out;
    float* cst;
    const float* cw;
    const float* add;
    float* out2;         // [ink-tp-tpcc] the conv chain's output rows
};
__device__ __forceinline__ unsigned long long tp_timer() {
    unsigned long long t;
    asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t));
    return t;
}
__device__ __forceinline__ void tp_st3(uint4* dst, unsigned a, unsigned b, unsigned c, unsigned tag) {
    asm volatile("st.volatile.global.v4.u32 [%0], {%1, %2, %3, %4};" ::"l"(dst), "r"(a), "r"(b), "r"(c), "r"(tag) : "memory");
}
__device__ __forceinline__ bool tp_ab(const unsigned* ab) { return INK_TP_AB && *(volatile const unsigned*)ab; }
// review tp2 M1: the tag in both 8 B halves regardless of INK_TP_W3 (k_tpmax)
__device__ __forceinline__ void tp_stD(uint4* dst, unsigned a, unsigned b, unsigned tag) {
    asm volatile("st.volatile.global.v4.u32 [%0], {%1, %2, %3, %4};" ::"l"(dst), "r"(a), "r"(tag), "r"(b), "r"(tag) : "memory");
}
__device__ __forceinline__ bool tp_ldD(const uint4* rp, unsigned tag, const unsigned* ab, unsigned long long t0, unsigned& a, unsigned& b) {
    uint4 x;
    unsigned it = 0;
    for (;;) {
        asm volatile("ld.volatile.global.v4.u32 {%0, %1, %2, %3}, [%4];" : "=r"(x.x), "=r"(x.y), "=r"(x.z), "=r"(x.w) : "l"(rp) : "memory");
        if (x.y == tag && x.w == tag) break;
        if ((++it & 4095u) == 0) {
            if (tp_ab(ab)) return false;
            if (tp_timer() - t0 > 20000000000ull) __trap();
        }
    }
    a = x.x;
    b = x.z;
    return true;
}
__device__ __forceinline__ void tp_st(uint4* dst, unsigned a, unsigned b, unsigned tag) {
    if (INK_TP_W3) { tp_st3(dst, a, b, 0u, tag); return; }
    asm volatile("st.volatile.global.v4.u32 [%0], {%1, %2, %3, %4};" ::"l"(dst), "r"(a), "r"(tag), "r"(b), "r"(tag) : "memory");
}
// waits for the word's tag in both halves; false: the context aborted (the values are not this collective's)
__device__ __forceinline__ bool tp_ld(const uint4* rp, unsigned tag, const unsigned* ab, unsigned long long t0, unsigned& a, unsigned& b) {
    uint4 x;
    unsigned it = 0;
    for (;;) {
        asm volatile("ld.volatile.global.v4.u32 {%0, %1, %2, %3}, [%4];" : "=r"(x.x), "=r"(x.y), "=r"(x.z), "=r"(x.w) : "l"(rp) : "memory");
        if ((INK_TP_W3 || x.y == tag) && x.w == tag) break;
        if ((++it & 4095u) == 0) {
            if (tp_ab(ab)) return false;
            if (tp_timer() - t0 > 20000000000ull) __trap();   // last resort: a peer whose host thread is gone (20 s)
        }
    }
    a = x.x;
    b = INK_TP_W3 ? x.y : x.z;
    return true;
}
// both words of a thread in flight together (one round trip per poll)
__device__ __forceinline__ bool tp_ld2(const uint4* rp, unsigned tag, const unsigned* ab, unsigned long long t0, uint4& x, uint4& y) {
    unsigned it = 0;
    for (;;) {
        asm volatile("ld.volatile.global.v4.u32 {%0, %1, %2, %3}, [%4];" : "=r"(x.x), "=r"(x.y), "=r"(x.z), "=r"(x.w) : "l"(rp) : "memory");
        asm volatile("ld.volatile.global.v4.u32 {%0, %1, %2, %3}, [%4];" : "=r"(y.x), "=r"(y.y), "=r"(y.z), "=r"(y.w) : "l"(rp + 1) : "memory");
        if (x.y == tag && x.w == tag && y.y == tag && y.w == tag) return true;
        if ((++it & 4095u) == 0) {
            if (tp_ab(ab)) return false;
            if (tp_timer() - t0 > 20000000000ull) __trap();
        }
    }
}
// [ink-tp-art] row all-reduce with the tags off the payload (gpt-oss [gptoss-r9-ar-trailer] idea): per (sender, row)
// 576 words of 4 fp32 + one trailer word per warp of 32 words (tag in .x / .w, 64-bit checksum of the warp's words in
// .y / .z) at words 576 .. 593 of the row's receive region; the receiving warp polls its trailer, then reads the
// words and re-reads them while their checksum differs (a trailer may arrive before its words). Same rank-order sum
// as k_tpar (bitwise); 25 % fewer link bytes than 3 fp32 + tag per word. Push transport, epilogue 0 only.
constexpr int TPA_W = H / 4, TPA_T = TPA_W / 32;
static_assert(TPA_W + TPA_T <= TPW, "trailer words inside the row region");
// review6 L1: per lane a 64-bit finalizer (murmur fmix64) of both 64-bit halves and the word index; the warp xor is
// then folded with the tag (an old checksum cannot validate an old payload under a new tag)
__device__ __forceinline__ unsigned long long tpa_fmix(unsigned long long k) {
    k ^= k >> 33;
    k *= 0xff51afd7ed558ccdull;
    k ^= k >> 33;
    k *= 0xc4ceb9fe1a85ec53ull;
    k ^= k >> 33;
    return k;
}
__device__ __forceinline__ unsigned long long tpa_mix(uint4 v, int i) {
    const unsigned long long a = (unsigned long long)v.x | ((unsigned long long)v.y << 32), b = (unsigned long long)v.z | ((unsigned long long)v.w << 32);
    return tpa_fmix(a ^ tpa_fmix(b + 0x9E3779B97F4A7C15ull * (unsigned long long)(i + 1)));
}
__device__ __forceinline__ unsigned long long tpa_wxor(unsigned long long h) {
    unsigned lo = (unsigned)h, hi = (unsigned)(h >> 32);
#pragma unroll
    for (int o = 16; o; o >>= 1) {
        lo ^= __shfl_xor_sync(0xffffffffu, lo, o);
        hi ^= __shfl_xor_sync(0xffffffffu, hi, o);
    }
    return (unsigned long long)lo | ((unsigned long long)hi << 32);
}
// [ink-tp-art] receive + rank-order sum of one row (warp-uniform; false: the context aborted)
__device__ __forceinline__ bool tpa_sum(const TpAr& t, const int slot, const int row, const int w, const unsigned tag, const float4 p,
                                        const unsigned long long t0, float4& o) {
    const int wp = w >> 5;
    float s0 = 0.f, s1 = 0.f, s2 = 0.f, s3 = 0.f;
    for (int g = 0; g < t.n; g++) {
        if (g == t.rank) {
            s0 += p.x;
            s1 += p.y;
            s2 += p.z;
            s3 += p.w;
            continue;
        }
        const uint4* rp = t.rd + (((size_t)slot * t.n + g) * MROW + row) * TPW;
        uint4 x, tr;
        unsigned it = 0;
        for (;;) {
            asm volatile("ld.volatile.global.v4.u32 {%0, %1, %2, %3}, [%4];" : "=r"(tr.x), "=r"(tr.y), "=r"(tr.z), "=r"(tr.w) : "l"(rp + TPA_W + wp) : "memory");
            asm volatile("ld.volatile.global.v4.u32 {%0, %1, %2, %3}, [%4];" : "=r"(x.x), "=r"(x.y), "=r"(x.z), "=r"(x.w) : "l"(rp + w) : "memory");
            if (__all_sync(0xffffffffu, tr.x == tag && tr.w == tag)) {
                const unsigned long long hh = tpa_wxor(tpa_mix(x, w)) ^ tpa_fmix(tag);
                if (hh == ((unsigned long long)tr.y | ((unsigned long long)tr.z << 32))) break;
            }
            if ((++it & 4095u) == 0) {
                if (tp_ab(t.ab)) return false;
                if (tp_timer() - t0 > 20000000000ull) __trap();
            }
        }
        s0 += __uint_as_float(x.x);
        s1 += __uint_as_float(x.y);
        s2 += __uint_as_float(x.z);
        s3 += __uint_as_float(x.w);
    }
    o = make_float4(s0, s1, s2, s3);
    return true;
}
// push of this rank's partial row (words + the warp's trailer) into every peer
__device__ __forceinline__ void tpa_push(const TpAr& t, const int slot, const int row, const int w, const unsigned tag, const float4 p) {
    const int lane = w & 31, wp = w >> 5;
    const uint4 pv = make_uint4(__float_as_uint(p.x), __float_as_uint(p.y), __float_as_uint(p.z), __float_as_uint(p.w));
    const unsigned long long h = tpa_wxor(tpa_mix(pv, w)) ^ tpa_fmix(tag);
    const size_t mine = (((size_t)slot * t.n + t.rank) * MROW + row) * TPW;
    for (int g = 0; g < t.n; g++) {
        if (g == t.rank) continue;
        uint4* d = (uint4*)t.src[g] + mine;
        asm volatile("st.volatile.global.v4.u32 [%0], {%1, %2, %3, %4};" ::"l"(d + w), "r"(pv.x), "r"(pv.y), "r"(pv.z), "r"(pv.w) : "memory");
        if (lane == 0)
            asm volatile("st.volatile.global.v4.u32 [%0], {%1, %2, %3, %4};" ::"l"(d + TPA_W + wp), "r"(tag), "r"((unsigned)h), "r"((unsigned)(h >> 32)), "r"(tag) : "memory");
    }
}
__global__ void __launch_bounds__(TPA_W) k_tpart(const __grid_constant__ A_tpar A) {
    const int w = threadIdx.x, row = blockIdx.x;
    const TpAr& t = A.t;
    const float4 p = *(const float4*)(A.part + (size_t)row * H + 4 * w);
    const unsigned tag = *t.step * 256u + (unsigned)A.idx + 1u;
    const int slot = A.idx % TP_SLOTS;
    tpa_push(t, slot, row, w, tag, p);
    float4 sv;
    if (!tpa_sum(t, slot, row, w, tag, p, tp_timer(), sv)) return;
    const int c = 4 * w;
    if (t.chk) {
        const float sa[4] = {sv.x, sv.y, sv.z, sv.w};
        tp_hash(t.chk, A.idx, row, c, sa, 4);
    }
    *(float4*)(A.out + (size_t)row * H + c) = sv;
}
__global__ void k_cmpbits(const float* a, const float* b, int n, unsigned long long* bad, unsigned long long* cnt) {
    const int i = blockIdx.x * 256 + threadIdx.x;
    if (i < n && __float_as_uint(a[i]) != __float_as_uint(b[i])) atomicAdd(bad, 1ull);
    if (i == 0 && cnt) atomicAdd(cnt, 1ull);   // compared collectives (graph replays included)
}
// [ink-tp-tpcc] the row all-reduce with the verify conv chain after it (k_tpart + k_tpchain / k_convR in one launch): the
// block of row j also sums the (already pushed) rows j - 1 .. j - 3 of its segment (the same receive + rank-order sum,
// so the same bits) and forms y = add + conv(x_j, x_j-1, x_j-2, x_j-3 | the slot's state), hist = x_j: the chain kernel's
// expression per row (conv states are not shifted in row steps; k_commit rebuilds them)
struct A_tpc {
    A_tpar a;
    float* hist;
    const int* rt;
};
__global__ void __launch_bounds__(TPA_W) k_tpartC(const __grid_constant__ A_tpc C) {
    const A_tpar& A = C.a;
    const int w = threadIdx.x, row = blockIdx.x, wp = w >> 5, c = 4 * w;
    const TpAr& t = A.t;
    const float4 p = *(const float4*)(A.part + (size_t)row * H + c);
    const unsigned tag = *t.step * 256u + (unsigned)A.idx + 1u;
    const int slot = A.idx % TP_SLOTS;
    tpa_push(t, slot, row, w, tag, p);
    // everything the epilogue reads, in flight under the wait: earlier rows' own partials, conv taps, residual, state
    const int j0 = C.rt[RT_SEG + row], nb = min(3, row - j0);
    float4 q[4];
    q[0] = p;
#pragma unroll
    for (int i = 1; i < 4; i++) q[i] = i <= nb ? *(const float4*)(A.part + (size_t)(row - i) * H + c) : make_float4(0.f, 0.f, 0.f, 0.f);
    const float4* cw4 = (const float4*)(A.cw + (size_t)c * 4);
    const float4 w0 = cw4[0], w1 = cw4[1], w2 = cw4[2], w3 = cw4[3];
    const float4 ad = A.add ? *(const float4*)(A.add + (size_t)row * H + c) : make_float4(0.f, 0.f, 0.f, 0.f);
    const float* s = A.cst + (size_t)C.rt[RT_SLOT + j0] * 3 * H + c;
    const float4 st0 = *(const float4*)s, st1 = *(const float4*)(s + H), st2 = *(const float4*)(s + 2 * H);   // p3, p2, p1 of the state
    // rank-order sums of rows row .. row - nb: every peer's words of those rows polled together
    float4 x[4];
#pragma unroll
    for (int i = 0; i < 4; i++) x[i] = make_float4(0.f, 0.f, 0.f, 0.f);
    const unsigned long long t0 = tp_timer();
    for (int g = 0; g < t.n; g++) {
        if (g == t.rank) {
#pragma unroll
            for (int i = 0; i < 4; i++) {
                x[i].x += q[i].x;
                x[i].y += q[i].y;
                x[i].z += q[i].z;
                x[i].w += q[i].w;
            }
            continue;
        }
        uint4 v[4];
        unsigned it = 0;
        for (;;) {
            bool ok = true;
#pragma unroll
            for (int i = 0; i < 4; i++) {
                if (i > nb) break;
                const uint4* rp = t.rd + (((size_t)slot * t.n + g) * MROW + row - i) * TPW;
                uint4 tr;
                asm volatile("ld.volatile.global.v4.u32 {%0, %1, %2, %3}, [%4];" : "=r"(tr.x), "=r"(tr.y), "=r"(tr.z), "=r"(tr.w) : "l"(rp + TPA_W + wp) : "memory");
                asm volatile("ld.volatile.global.v4.u32 {%0, %1, %2, %3}, [%4];" : "=r"(v[i].x), "=r"(v[i].y), "=r"(v[i].z), "=r"(v[i].w) : "l"(rp + w) : "memory");
                if (__all_sync(0xffffffffu, tr.x == tag && tr.w == tag)) {
                    const unsigned long long hh = tpa_wxor(tpa_mix(v[i], w)) ^ tpa_fmix(tag);
                    ok &= hh == ((unsigned long long)tr.y | ((unsigned long long)tr.z << 32));
                } else ok = false;
            }
            if (ok) break;
            if ((++it & 4095u) == 0) {
                if (tp_ab(t.ab)) return;
                if (tp_timer() - t0 > 20000000000ull) __trap();
            }
        }
#pragma unroll
        for (int i = 0; i < 4; i++) {
            if (i > nb) break;
            x[i].x += __uint_as_float(v[i].x);
            x[i].y += __uint_as_float(v[i].y);
            x[i].z += __uint_as_float(v[i].z);
            x[i].w += __uint_as_float(v[i].w);
        }
    }
    if (t.chk) {
        const float sa[4] = {x[0].x, x[0].y, x[0].z, x[0].w};
        tp_hash(t.chk, A.idx, row, c, sa, 4);
    }
    *(float4*)(A.out + (size_t)row * H + c) = x[0];
    *(float4*)(C.hist + (size_t)row * H + c) = x[0];
    // before the segment: the slot's state (p1 = state row 2, p2 = row 1, p3 = row 0)
    if (nb < 3) x[3] = nb == 2 ? st2 : nb == 1 ? st1 : st0;
    if (nb < 2) x[2] = nb == 1 ? st2 : st1;
    if (nb < 1) x[1] = st2;
    const float xs[4][4] = {{x[0].x, x[0].y, x[0].z, x[0].w}, {x[1].x, x[1].y, x[1].z, x[1].w}, {x[2].x, x[2].y, x[2].z, x[2].w},
                            {x[3].x, x[3].y, x[3].z, x[3].w}};
    const float wv[4][4] = {{w0.x, w0.y, w0.z, w0.w}, {w1.x, w1.y, w1.z, w1.w}, {w2.x, w2.y, w2.z, w2.w}, {w3.x, w3.y, w3.z, w3.w}};
    const float av[4] = {ad.x, ad.y, ad.z, ad.w};
    float yo[4];
#pragma unroll
    for (int e = 0; e < 4; e++) {   // conv_core's expression on the taps in registers
        const float o = conv_core(xs[0][e], xs[1][e], xs[2][e], xs[3][e], A.cw, c + e);   // review7 L1: the chain kernels' helper
        yo[e] = (A.add ? av[e] : 0.f) + o;
    }
    *(float4*)(A.out2 + (size_t)row * H + c) = make_float4(yo[0], yo[1], yo[2], yo[3]);
}
#if INK_TP_W3
__global__ void __launch_bounds__(TPT) k_tpar(const __grid_constant__ A_tpar A) {
    const int w = threadIdx.x, row = blockIdx.x;
    const TpAr& t = A.t;
    const float* pp = A.part + (size_t)row * H + 3 * w;
    float s0 = 0.f, s1 = 0.f, s2 = 0.f;
    if (t.push) {
        const unsigned tag = *t.step * 256u + (unsigned)A.idx + 1u;
        const int slot = A.idx % TP_SLOTS;
        const float p0 = pp[0], p1 = pp[1], p2 = pp[2];
        const size_t mine = (((size_t)slot * t.n + t.rank) * MROW + row) * TPW + w;
        for (int g = 0; g < t.n; g++) {
            if (g == t.rank) continue;
            tp_st3((uint4*)t.src[g] + mine, __float_as_uint(p0), __float_as_uint(p1), __float_as_uint(p2), tag);
        }
        const unsigned long long t0 = tp_timer();
        for (int g = 0; g < t.n; g++) {
            if (g == t.rank) {
                s0 += p0;
                s1 += p1;
                s2 += p2;
                continue;
            }
            const uint4* rp = t.rd + (((size_t)slot * t.n + g) * MROW + row) * TPW + w;
            uint4 x;
            unsigned it = 0;
            for (;;) {
                asm volatile("ld.volatile.global.v4.u32 {%0, %1, %2, %3}, [%4];" : "=r"(x.x), "=r"(x.y), "=r"(x.z), "=r"(x.w) : "l"(rp) : "memory");
                if (x.w == tag) break;
                if ((++it & 4095u) == 0) {
                    if (tp_ab(t.ab)) return;
                    if (tp_timer() - t0 > 20000000000ull) __trap();
                }
            }
            s0 += __uint_as_float(x.x);
            s1 += __uint_as_float(x.y);
            s2 += __uint_as_float(x.z);
        }
    } else {
        for (int g = 0; g < t.n; g++) {
            const float* q = (const float*)t.src[g] + (size_t)row * H + 3 * w;
            s0 += q[0];
            s1 += q[1];
            s2 += q[2];
        }
    }
    const int c = 3 * w;
    if (t.chk) {
        const float sv[3] = {s0, s1, s2};
        tp_hash(t.chk, A.idx, row, c, sv, 3);
    }
    if (A.epi == 1) {
        A.out[c] = conv1(s0, c, A.cst, A.cw, A.add);
        A.out[c + 1] = conv1(s1, c + 1, A.cst, A.cw, A.add);
        A.out[c + 2] = conv1(s2, c + 2, A.cst, A.cw, A.add);
    } else {
        float* o = A.out + (size_t)row * H + c;
        o[0] = s0;
        o[1] = s1;
        o[2] = s2;
    }
}
#else
__global__ void __launch_bounds__(TPT) k_tpar(const __grid_constant__ A_tpar A) {
    const int w = threadIdx.x, row = blockIdx.x;
    const TpAr& t = A.t;
    const float* pp = A.part + (size_t)row * H + 4 * w;
    float s[4] = {0.f, 0.f, 0.f, 0.f};
    if (t.push) {
        const unsigned tag = *t.step * 256u + (unsigned)A.idx + 1u;
        const int slot = A.idx % TP_SLOTS;
        const float4 p = *(const float4*)pp;
        const size_t mine = (((size_t)slot * t.n + t.rank) * MROW + row) * TPW + 2 * w;
        for (int g = 0; g < t.n; g++) {
            if (g == t.rank) continue;
            uint4* dst = (uint4*)t.src[g] + mine;
            tp_st(dst, __float_as_uint(p.x), __float_as_uint(p.y), tag);
            tp_st(dst + 1, __float_as_uint(p.z), __float_as_uint(p.w), tag);
        }
        const unsigned long long t0 = tp_timer();
        for (int g = 0; g < t.n; g++) {
            if (g == t.rank) {
                s[0] += p.x;
                s[1] += p.y;
                s[2] += p.z;
                s[3] += p.w;
                continue;
            }
            const uint4* rp = t.rd + (((size_t)slot * t.n + g) * MROW + row) * TPW + 2 * w;
            uint4 x, y;
            if (!tp_ld2(rp, tag, t.ab, t0, x, y)) return;   // aborted
            s[0] += __uint_as_float(x.x);
            s[1] += __uint_as_float(x.z);
            s[2] += __uint_as_float(y.x);
            s[3] += __uint_as_float(y.z);
        }
    } else {
        for (int g = 0; g < t.n; g++) {
            const float* q = (const float*)t.src[g] + (size_t)row * H + 4 * w;
#pragma unroll
            for (int i = 0; i < 4; i++) s[i] += q[i];
        }
    }
    const int c = 4 * w;
    if (t.chk) tp_hash(t.chk, A.idx, row, c, s, 4);
    if (A.epi == 1) {
#pragma unroll
        for (int i = 0; i < 4; i++) A.out[c + i] = conv1(s[i], c + i, A.cst, A.cw, A.add);
    } else {
        float* o = A.out + (size_t)row * H + c;
#pragma unroll
        for (int i = 0; i < 4; i++) o[i] = s[i];
    }
}
#endif
__global__ void k_tpbegin(unsigned* step) { *step += 1u; }
// [ink-vp-cpush] NVFP4 down of a row step (nr <= TPC_MAXR): k_dsum (row j = its 8 slot partials in slot order x 16384)
// and the column-major row all-reduce + verify conv chain (tpc_col) in one launch; thread = column
__global__ void __launch_bounds__(64) k_dsumC(const float* __restrict__ part, const __grid_constant__ TpCol p) {
    const int c = blockIdx.x * 64 + threadIdx.x;
    const unsigned tag = tpc_tag(p);
    float v[TPC_MAXR];
#pragma unroll
    for (int j = 0; j < TPC_MAXR; j++) {
        v[j] = 0.f;
        if (j < p.nr) {
            const float* q = part + (size_t)j * TOPK * H + c;
            float a = q[0];
#pragma unroll
            for (int s2 = 1; s2 < TOPK; s2++) a += q[(size_t)s2 * H];
            v[j] = a * 16384.f;
        }
    }
    tpc_col(p, tag, c, v);
}

// [ink-tp-dec1] argmax over the ranks' vocab parts: the largest key ((ordered score << 32) | ~global row) of all ranks
// (ties: the lower row, as the one-card head) -> the token. One thread; push: one 16 B word per peer (key, tag).
struct A_tpmax {
    TpAr t;
    const unsigned long long* key;   // this rank's keys [nr] (direct: t.src[g] = rank g's keys)
    int idx;
    int* arg;                        // [nr]
    int nr;
};
__global__ void k_tpmax(const __grid_constant__ A_tpmax A) {   // thread j = row j
    const TpAr& t = A.t;
    const int row = threadIdx.x;
    if (row >= A.nr) return;
    const unsigned long long mine = ((volatile const unsigned long long*)A.key)[row];
    unsigned long long best = 0;
    if (t.push) {
        const unsigned stp = *t.step, tag = stp * 256u + (unsigned)A.idx + 1u;
        const int slot = TP_SLOTS + (int)(stp & 1u);
        const size_t me = (((size_t)slot * t.n + t.rank) * MROW + row) * TPW;
        for (int g = 0; g < t.n; g++) {
            if (g == t.rank) continue;
            tp_stD((uint4*)t.src[g] + me, (unsigned)mine, (unsigned)(mine >> 32), tag);
        }
        const unsigned long long t0 = tp_timer();
        for (int g = 0; g < t.n; g++) {
            unsigned long long k = mine;
            if (g != t.rank) {
                unsigned lo, hi;
                if (!tp_ldD(t.rd + (((size_t)slot * t.n + g) * MROW + row) * TPW, tag, t.ab, t0, lo, hi)) return;   // aborted
                k = ((unsigned long long)hi << 32) | lo;
            }
            best = max(best, k);
        }
    } else
        for (int g = 0; g < t.n; g++) best = max(best, ((const unsigned long long*)t.src[g])[row]);
    A.arg[row] = (int)(0xffffffffu - (unsigned)(best & 0xffffffffu));
    if (t.chk) atomicXor(t.chk + 255, (unsigned)(best >> 32) * (2u * row + 1u) ^ (unsigned)best);
}

// [ink-tp-dec1] the verify conv chain after the wo sum (k_g12tc MODE 2 epilogue on the summed rows): per column c the
// rows in order, restarting from the slot's state at each segment; y = add + conv, hist = the raw rows
__global__ void __launch_bounds__(256) k_tpchain(const float* sum, int nr, const float* cst, const float* cw, const float* add, float* y,
                                                 float* hist, const int* rt) {
    const int c = blockIdx.x * 256 + threadIdx.x;
    if (c >= H) return;
    if (gridDim.y > 1) {   // [ink-tp-convseg] one block row per segment
        conv_seg(sum, nr, cst, cw, add, y, hist, rt, c, blockIdx.y);
        return;
    }
    float p3 = 0.f, p2 = 0.f, p1 = 0.f;
    for (int j = 0; j < nr; j++) {
        if (rt[RT_SEG + j] == j) {
            const float* cs = cst + (size_t)rt[RT_SLOT + j] * 3 * H;
            p3 = cs[c], p2 = cs[H + c], p1 = cs[2 * H + c];
        }
        const float x0 = sum[(size_t)j * H + c];
        const float o = conv_core(x0, p1, p2, p3, cw, c);
        y[(size_t)j * H + c] = (add ? add[(size_t)j * H + c] : 0.f) + o;
        hist[(size_t)j * H + c] = x0;
        p3 = p2;
        p2 = p1;
        p1 = x0;
    }
}

// [ink-tp-pf] prefill all-reduce of T rows: out = sum of the ranks' partial rows in rank order 0..N-1 (from 0.f, the
// decode collectives' order); src[g] = rank g's rows (own partial or the receive rows its copy landed in)
struct A_tpsumP {
    const float4* src[TP_MAX];
    int n;
    size_t n4;   // float4 per buffer
    float4* out;
};
__global__ void __launch_bounds__(256) k_tpsumP(const __grid_constant__ A_tpsumP A) {
    for (size_t i = (size_t)blockIdx.x * 256 + threadIdx.x; i < A.n4; i += (size_t)gridDim.x * 256) {
        float4 s = make_float4(0.f, 0.f, 0.f, 0.f);
        for (int g = 0; g < A.n; g++) {
            const float4 v = A.src[g][i];
            s.x += v.x;
            s.y += v.y;
            s.z += v.z;
            s.w += v.w;
        }
        A.out[i] = s;
    }
}
// [ink-tp-pf] zero the AO columns [c0, c1) outside the rank's own q-head columns [o0, o1) (wo reads whole 256-column
// chunks; a chunk half of another rank's head, or a shared KV group's other q heads, must add exact zeros)
__global__ void k_tpzcols(bf16* AO, int ld, int T, int c0, int c1, int o0, int o1) {
    const int t = blockIdx.x;
    for (int c = c0 + threadIdx.x; c < c1; c += blockDim.x)
        if (c < o0 || c >= o1) AO[(size_t)t * ld + c] = __float2bfloat16(0.f);
}
