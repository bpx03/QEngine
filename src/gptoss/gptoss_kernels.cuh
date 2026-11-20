// gpt-oss-120b kernels, first engine (M2). Included by gptoss.cu only.
// Arithmetic: every weight product takes bf16 operands and accumulates in fp32; the residual stream is fp32.
#pragma once
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace gptoss {

typedef __nv_bfloat16 bf16;
typedef __nv_bfloat162 bf162;

constexpr int NSPLIT = 128;      // key splits of a full-attention decode step (per KV head)
constexpr int SPLIT_MIN = 32;    // keys per split before a second split is used (a warp walks its keys one by one:
                                 // 32 keys = 4 dependent steps per warp; 128 keys in one block cost 12.7 us)
constexpr int ASC = 3;           // split classes of a decode step's full layers: grid of AS_GRID[class] key splits
constexpr int AS_GRID[ASC] = {8, 32, NSPLIT};
static_assert(NSPLIT % 16 == 0, "k_attn_comb reads the o words 16 splits at a time");
constexpr int SSPLIT = 4;        // splits of a sliding layer: 4 x 32 window rows
constexpr int APW = 66;          // attention partial: m, l, o[64]
constexpr int MAXR = 16;         // rows of one decode step (a token and its drafts, and / or rows of several sequences)
// Row table of a decode step (device ints, `posp` of the decode kernels): row r has
//   position rt[r], first full-layer K/V row of its sequence rt[MAXR + r], ring set rt[2 MAXR + r].
// A full-layer key at position k of that sequence is row base + k; a sliding-layer key is row
// ring * RING_ROWS + (k & 255) relative to the layer's ring set 0. Rows of one sequence with consecutive positions
// see each other's K/V inside the step (drafts); rows of different sequences are independent.
constexpr int RING_ROWS = 18 * 256;   // rows of one ring set over the 18 sliding layers (the stride between sets)
constexpr int AR_SLOTS = 4;
constexpr int ARW = 960;         // all-reduce words: 3 fp32 + tag each (H / 3)
constexpr float NEG_INF = -3.0e38f;

// Words of sender g for slot s sit at base + (s * ngpu + g) * words.
//   P2P:  nw = ngpu, wr[g] = rank g's receive words (16 B words in device memory, pushed by the senders), rd = own
//   host: nw = 1, sh = 1, wr[0] = rd = one pinned host buffer mapped on every card, written once by the sender and
//         read by the other ranks ([gptoss-rel2-hostar]: k_ar_t's 16 B words + trailers; GPTOSS_AR_TRAIL=0: k_ar_host's
//         8 B words, value + tag)
struct ArBuf {
    void* wr[4];
    void* rd;
    int nw;
    int sh;   // 1: wr[0] is shared by every rank (a sender stores its words once, also "to itself")
};
// Push target of a kernel that produces the partial of a collective (round 9): d[i] = this rank's words of row 0 in
// peer i's receive buffer for the collective's slot (row rr: + rr * 4 ARW floats; the payload of k_ar_t is the row
// itself, 4 fp32 per word). The producer stores its sums there as it finishes them, so the bytes cross the link while
// the kernel still runs; k_ar_t (pushed = 1) then sends only the trailers and checks every word against them.
struct ArPush {
    float* d[3];
    int n;
};

__device__ __forceinline__ float bf2f(bf16 v) { return __bfloat162float(v); }
// 16 B from global to shared memory without holding the warp (cp.async); groups are committed and waited for per warp
__device__ __forceinline__ void cp16(void* dst_shared, const void* src) {
    const uint32_t a = (uint32_t)__cvta_generic_to_shared(dst_shared);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(a), "l"(src) : "memory");
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;" ::: "memory"); }
__device__ __forceinline__ void cp_wait0() { asm volatile("cp.async.wait_group 0;" ::: "memory"); }   // every group
__device__ __forceinline__ void cp_wait1() { asm volatile("cp.async.wait_group 1;" ::: "memory"); }   // all but the newest
__device__ __forceinline__ float warp_sum(float v) {
    for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
// 8 bf16 weights times 8 bf16 inputs (two bf16 per 32-bit word: the low one is the even element)
__device__ __forceinline__ float dot8(uint4 w, uint4 x) {
    const uint32_t ww[4] = {w.x, w.y, w.z, w.w}, xx[4] = {x.x, x.y, x.z, x.w};
    float d = 0.f;
#pragma unroll
    for (int j = 0; j < 4; j++)
        d += __uint_as_float(ww[j] << 16) * __uint_as_float(xx[j] << 16) +
             __uint_as_float(ww[j] & 0xffff0000u) * __uint_as_float(xx[j] & 0xffff0000u);
    return d;
}
// one warp walks a dense bf16 row of K weights (K % 8 == 0): each lane takes 16 B per 512 B step
__device__ __forceinline__ float row_dot(const bf16* wr, const bf16* xs, int K, int lane) {
    float acc = 0.f;
    int e0 = lane * 8;
    for (; e0 + 768 < K; e0 += 1024) {
        const uint4 a = *(const uint4*)(wr + e0), b = *(const uint4*)(wr + e0 + 256);
        const uint4 c = *(const uint4*)(wr + e0 + 512), d = *(const uint4*)(wr + e0 + 768);
        acc += dot8(a, *(const uint4*)(xs + e0)) + dot8(b, *(const uint4*)(xs + e0 + 256));
        acc += dot8(c, *(const uint4*)(xs + e0 + 512)) + dot8(d, *(const uint4*)(xs + e0 + 768));
    }
    for (; e0 < K; e0 += 256) acc += dot8(*(const uint4*)(wr + e0), *(const uint4*)(xs + e0));
    return warp_sum(acc);
}
// two rows at once (their loads are issued together)
__device__ __forceinline__ void row_dot2(const bf16* wa, const bf16* wb, const bf16* xs, int K, int lane, float& ra, float& rb) {
    float a = 0.f, b = 0.f;
    int e0 = lane * 8;
    for (; e0 + 256 < K; e0 += 512) {
        const uint4 a0 = *(const uint4*)(wa + e0), a1 = *(const uint4*)(wa + e0 + 256);
        const uint4 b0 = *(const uint4*)(wb + e0), b1 = *(const uint4*)(wb + e0 + 256);
        const uint4 x0 = *(const uint4*)(xs + e0), x1 = *(const uint4*)(xs + e0 + 256);
        a += dot8(a0, x0) + dot8(a1, x1);
        b += dot8(b0, x0) + dot8(b1, x1);
    }
    for (; e0 < K; e0 += 256) {
        const uint4 x0 = *(const uint4*)(xs + e0);
        a += dot8(*(const uint4*)(wa + e0), x0);
        b += dot8(*(const uint4*)(wb + e0), x0);
    }
    ra = warp_sum(a);
    rb = warp_sum(b);
}
__device__ __forceinline__ void stage_x(bf16* xs, const bf16* x, int K) {
    for (int i = threadIdx.x; i < K / 8; i += blockDim.x) ((uint4*)xs)[i] = ((const uint4*)x)[i];
}

// ---------------------------------------------------------------- MXFP4
// Two e2m1 codes (bits 0-3 and 16-19 of t) as fp16 bit patterns: code bits 0-2 land in the fp16 exponent / top
// mantissa bit, the sign in bit 15. The fp16 value is the e2m1 value / 2^14 (exact, the subnormal 0.5 included).
__device__ __forceinline__ float2 fp4pair(uint32_t t) {
    const uint32_t h = ((t & 0x00070007u) << 9) | ((t & 0x00080008u) << 12);
    return __half22float2(*(const __half2*)&h);
}
// 2^(s - 127) * 2^14 (the fp4pair factor); scale bytes in the files are 112..137 (s = 0 would give 0, not 2^-127)
__device__ __forceinline__ float mx_scale(uint8_t s) { return __int_as_float((int)s << 23) * 16384.f; }
// Expert row record (plan D2): one aligned stream per weight row instead of a code plane and a scale plane:
//   [KB * 16 B codes][ceil(KB / 2) B: 4-bit scale offsets, low nibble = even block][1 B: the row's smallest scale]
// padded to 16 B (gate_up, 90 blocks: 1,488 B; down, 30 blocks: 496 B). Worst scale range in the files: 11 <= 15.
__host__ __device__ __forceinline__ int mx_rs(int KB) { return ((KB * 16 + ((KB + 1) >> 1) + 1 + 15) >> 4) << 4; }
__device__ __forceinline__ uint32_t mx_sbyte(const uint8_t* rec, int KB, int b) {
    const uint8_t* t = rec + KB * 16;
    return (uint32_t)t[(KB + 1) >> 1] + ((t[b >> 1] >> ((b & 1) * 4)) & 15u);
}
#ifdef GPTOSS_EXP_NOSCL   // timing experiment only (wrong output): the decode expert kernels without any scale load
#define MX_SCALE_AT(rec, KB, b) mx_scale(120)
#else
#define MX_SCALE_AT(rec, KB, b) mx_scale((uint8_t)mx_sbyte(rec, KB, b))
#endif
// one MXFP4 block (16 B = 32 codes, low nibble = even element) times 32 bf16 inputs (16 B aligned)
__device__ __forceinline__ float fp4dot32(uint4 c, const bf16* xb) {
    const uint32_t cw[4] = {c.x, c.y, c.z, c.w};
    const uint4* xp = (const uint4*)xb;
    float d = 0.f;
#pragma unroll
    for (int j = 0; j < 4; j++) {   // word j: elements 8j .. 8j+7, nibble i = element i
        const uint32_t w = cw[j];
        const float2 e04 = fp4pair(w), e15 = fp4pair(w >> 4), e26 = fp4pair(w >> 8), e37 = fp4pair(w >> 12);
        const uint4 xv = xp[j];
        d += e04.x * __uint_as_float(xv.x << 16) + e15.x * __uint_as_float(xv.x & 0xffff0000u) +
             e26.x * __uint_as_float(xv.y << 16) + e37.x * __uint_as_float(xv.y & 0xffff0000u);
        d += e04.y * __uint_as_float(xv.z << 16) + e15.y * __uint_as_float(xv.z & 0xffff0000u) +
             e26.y * __uint_as_float(xv.w << 16) + e37.y * __uint_as_float(xv.w & 0xffff0000u);
    }
    return d;
}
// fp4dot32 with the 32 inputs already in registers (the same expression, so the same sum)
__device__ __forceinline__ float fp4dot32_r(uint4 c, const uint4* xr) {
    const uint32_t cw[4] = {c.x, c.y, c.z, c.w};
    float d = 0.f;
#pragma unroll
    for (int j = 0; j < 4; j++) {
        const uint32_t w = cw[j];
        const float2 e04 = fp4pair(w), e15 = fp4pair(w >> 4), e26 = fp4pair(w >> 8), e37 = fp4pair(w >> 12);
        const uint4 xv = xr[j];
        d += e04.x * __uint_as_float(xv.x << 16) + e15.x * __uint_as_float(xv.x & 0xffff0000u) +
             e26.x * __uint_as_float(xv.y << 16) + e37.x * __uint_as_float(xv.y & 0xffff0000u);
        d += e04.y * __uint_as_float(xv.z << 16) + e15.y * __uint_as_float(xv.z & 0xffff0000u) +
             e26.y * __uint_as_float(xv.w << 16) + e37.y * __uint_as_float(xv.w & 0xffff0000u);
    }
    return d;
}
__device__ __forceinline__ float swiglu(float gate, float up) {
    gate = fminf(gate, 7.f);
    up = fminf(fmaxf(up, -7.f), 7.f);
    return (up + 1.f) * (gate * (1.f / (1.f + expf(-1.702f * gate))));
}
// ns clock for the bounded waits of the all-reduce: a peer that never arrives ends the step with an error after
// AR_WAIT_NS instead of spinning for an hour (review L4)
__device__ __forceinline__ unsigned long long gtimer() {
    unsigned long long t;
    asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t));
    return t;
}
constexpr unsigned long long AR_WAIT_NS = 20000000000ull;
// -DGPTOSS_NOWAIT (memcheck builds only, EXTRA of build.sh): no tag wait. compute-sanitizer runs one kernel at a time,
// so a kernel that waits for a peer card's words would wait until the trap; the sums are then meaningless, the
// memory accesses are the same.
#ifdef GPTOSS_NOWAIT
#define AR_PENDING(c) false
#else
#define AR_PENDING(c) (c)
#endif
__device__ __forceinline__ float round_bf16(float v) { return __bfloat162float(__float2bfloat16_rn(v)); }

// Tail prefetch (idea of [qfn-tailpf], bit-identical): a latency-bound kernel asks for the first bytes of the next
// byte-bound kernel's weights to be brought into L2, so that kernel's first loads do not start a DRAM round trip
// after the kernel boundary. tid / nt: this thread's index and the thread count of the whole launch.
__device__ __forceinline__ void tail_pf(const void* pf, size_t pfb, unsigned tid, unsigned nt) {
    if (!pf) return;
    for (size_t o = (size_t)tid * 128; o < pfb; o += (size_t)nt * 128) asm volatile("prefetch.global.L2 [%0];" ::"l"((const char*)pf + o));
}

// ---------------------------------------------------------------- small helpers
__global__ void k_begin(unsigned* step) { *step += 1u; }

// h[row] += p0 + p1 + p2 + p3 (rank order; null = skipped), xn = bf16(nw * h * rsqrt(mean(h^2) + eps)). One block per row.
__global__ void __launch_bounds__(960) k_resnorm(float* h, const float* p0, const float* p1, const float* p2,
                                                 const float* p3, const bf16* nw, bf16* xn, int HH) {
    __shared__ float red[32];
    const size_t base = (size_t)blockIdx.x * HH;
    const int w = threadIdx.x;
    float v[3];
    float ss = 0.f;
#pragma unroll
    for (int j = 0; j < 3; j++) {
        const size_t i = base + 3 * w + j;
        float s = 0.f;
        if (p0) s += p0[i];
        if (p1) s += p1[i];
        if (p2) s += p2[i];
        if (p3) s += p3[i];
        v[j] = h[i] + s;
        h[i] = v[j];
        ss += v[j] * v[j];
    }
    ss = warp_sum(ss);
    if ((w & 31) == 0) red[w >> 5] = ss;
    __syncthreads();
    if (w < 32) {
        float t = w < 30 ? red[w] : 0.f;
        t = warp_sum(t);
        if (w == 0) red[31] = t;
    }
    __syncthreads();
    const float rs = rsqrtf(red[31] / (float)HH + 1e-5f);
#pragma unroll
    for (int j = 0; j < 3; j++) xn[base + 3 * w + j] = __float2bfloat16_rn(bf2f(nw[3 * w + j]) * (v[j] * rs));
}

// residual add + RMSNorm of one row inside a single 960-thread block (thread w owns elements 3w .. 3w+2)
__device__ __forceinline__ void ar_tail(float* h, const bf16* nw, bf16* xn, int w, float s0, float s1, float s2) {
    __shared__ float red[32];
    const float v0 = h[3 * w] + s0, v1 = h[3 * w + 1] + s1, v2 = h[3 * w + 2] + s2;
    h[3 * w] = v0;
    h[3 * w + 1] = v1;
    h[3 * w + 2] = v2;
    float ss = warp_sum(v0 * v0 + v1 * v1 + v2 * v2);
    if ((w & 31) == 0) red[w >> 5] = ss;
    __syncthreads();
    if (w < 32) {
        float t = w < 30 ? red[w] : 0.f;
        t = warp_sum(t);
        if (w == 0) red[31] = t;
    }
    __syncthreads();
    const float rs = rsqrtf(red[31] / (float)(3 * ARW) + 1e-5f);
    xn[3 * w] = __float2bfloat16_rn(bf2f(nw[3 * w]) * (v0 * rs));
    xn[3 * w + 1] = __float2bfloat16_rn(bf2f(nw[3 * w + 1]) * (v1 * rs));
    xn[3 * w + 2] = __float2bfloat16_rn(bf2f(nw[3 * w + 2]) * (v2 * rs));
}

// Decode all-reduce of one row with the residual add and the next norm fused. Every rank stores its partial as 16-byte
// words (3 fp32 + tag) into every rank's receive words (peer memory over P2P), then each thread waits for the tag of
// all senders and sums in rank order: the sum is bitwise identical on every card. tag = step * 256 + idx + 1, so a
// replayed graph never matches a stale word.
__global__ void __launch_bounds__(960) k_ar(const float* part, float* h, const bf16* nw, bf16* xn,
                                            const __grid_constant__ ArBuf a, int rank, int ngpu, const unsigned* step,
                                            int idx, const void* pf, size_t pfb) {
    const unsigned tag = *step * 256u + (unsigned)idx + 1u;
    const int slot = idx % AR_SLOTS;
    const int w = threadIdx.x;
    const int rr = blockIdx.x;                 // row of the step: its own partial, residual, normed row and words
    part += (size_t)rr * 2880;
    h += (size_t)rr * 2880;
    xn += (size_t)rr * 2880;
    const float p0 = part[3 * w], p1 = part[3 * w + 1], p2 = part[3 * w + 2];
    for (int g = 0; g < a.nw; g++) {
#ifdef GPTOSS_AR_NOSTORE   // timing probe only (wrong output, with -DGPTOSS_NOWAIT): no stores into the peers' words
        if (g != rank) continue;
#endif
        uint4* dst = (uint4*)a.wr[g] + (((size_t)slot * ngpu + rank) * MAXR + rr) * ARW + w;
        asm volatile("st.volatile.global.v4.u32 [%0], {%1, %2, %3, %4};" ::"l"(dst), "r"(__float_as_uint(p0)),
                     "r"(__float_as_uint(p1)), "r"(__float_as_uint(p2)), "r"(tag) : "memory");
    }
    tail_pf(pf, pfb, w, 960);   // while the peers' words travel
    const uint4* rb = (const uint4*)a.rd + ((size_t)slot * ngpu * MAXR + rr) * ARW + w;
    float s0 = 0.f, s1 = 0.f, s2 = 0.f;
    const unsigned long long t0 = gtimer();
    for (int g = 0; g < ngpu; g++) {
        uint4 x;
        unsigned it = 0;
        do {
            asm volatile("ld.volatile.global.v4.u32 {%0, %1, %2, %3}, [%4];"
                         : "=r"(x.x), "=r"(x.y), "=r"(x.z), "=r"(x.w) : "l"(rb + (size_t)g * MAXR * ARW) : "memory");
            if ((++it & 4095u) == 0 && gtimer() - t0 > AR_WAIT_NS) __trap();
        } while (AR_PENDING(x.w != tag));
        s0 += __uint_as_float(x.x);
        s1 += __uint_as_float(x.y);
        s2 += __uint_as_float(x.z);
    }
    ar_tail(h, nw, xn, w, s0, s1, s2);
}

// The same collective with the tags off the payload (round 9; default at every row count: the link's bytes bound
// the collective from 2 rows on, one row is at parity). With GPTOSS_AR_TRAIL > 1 k_ar and k_ar_t share the receive
// words: k_ar's tag field then holds an old fp32 until its own store lands, which never equals a tag in practice. Every rank stores its partial as ARP = 720 words of 4 fp32 into every other rank's receive words.
// A warp (32 words; the last one 16) then stores one trailer word per peer: {checksum lo, checksum hi, 0, tag},
// checksum = xor over the warp's words of a 64-bit value of (word bits, word index). The stores of different threads
// arrive in no known order (measured: a warp's trailer is often there before its words), so a trailer alone proves
// nothing: a receiving warp reads the peers' words and trailers, reduces the words' checksums and reads again while
// a tag is not this collective's or a checksum differs from its trailer's (a mix of stale and new words passes only
// on a 64-bit collision; the senders' checksums are compared in one reduction, each rotated by 21 bits per rank). Nothing in the wait crosses a warp. The own partial is taken from local memory. Sum in
// rank order from 0, then the same residual add and norm (thread w owns elements 3w .. 3w + 2 for the sum of
// squares; the sums pass through shared memory): bitwise k_ar's result, and the same on every card. Link bytes per
// (sender, row, peer): 11.5 KB + 23 trailers (368 B) instead of 15.4 KB. Every card must run the same form.
constexpr int ARP = 720;
constexpr int ARPW = (ARP + 31) / 32;   // warps with payload words; warp k's trailer is word ARP + k
static_assert(ARP * 4 == 3 * ARW && ARP + ARPW <= 800 && ARW >= 808, "payload + trailers fit the 960-word stride (800..807: the timing probe)");
__device__ __forceinline__ unsigned long long warp_xor64(unsigned long long v) {
    unsigned lo = (unsigned)v, hi = (unsigned)(v >> 32);
#pragma unroll
    for (int o = 16; o; o >>= 1) {
        lo ^= __shfl_xor_sync(0xffffffffu, lo, o);
        hi ^= __shfl_xor_sync(0xffffffffu, hi, o);
    }
    return ((unsigned long long)hi << 32) | lo;
}
// a word's checksum term: sum of its four 32-bit parts times four odd multipliers of the word index (mod 2^64), folded
struct ArKey {
    unsigned c0, c1, c2, c3;
};
__device__ __forceinline__ ArKey ar_key(unsigned w) {
    const unsigned b = (w + 1u) * 0x9E3779B1u;
    return {(b ^ 0x85EBCA6Bu) | 1u, (b ^ 0xC2B2AE35u) | 1u, (b ^ 0x27D4EB2Fu) | 1u, (b ^ 0x165667B1u) | 1u};
}
__device__ __forceinline__ unsigned long long ar_mix(const ArKey& k, unsigned x0, unsigned x1, unsigned x2, unsigned x3) {
    const unsigned long long v = ((unsigned long long)x0 * k.c0 + (unsigned long long)x1 * k.c1) +
                                 ((unsigned long long)x2 * k.c2 + (unsigned long long)x3 * k.c3);
    return v ^ (v >> 29);
}
__global__ void __launch_bounds__(960) k_ar_t(const float* part, float* h, const bf16* nw, bf16* xn,
                                              const __grid_constant__ ArBuf a, int rank, int ngpu, const unsigned* step,
                                              int idx, const void* pf, size_t pfb, int pushed) {
    __shared__ __align__(16) float sv[4 * ARP];
    __shared__ float red[32];
#ifdef GPTOSS_AR_TIME   // timing probe: thread 0 of row 0 adds its marks (ns since entry) to spare words of its own row
    unsigned long long tm[6];
    tm[0] = gtimer();
#define AR_MARK(k) tm[k] = gtimer()
#else
#define AR_MARK(k)
#endif
    const unsigned tag = *step * 256u + (unsigned)idx + 1u;
    const int slot = idx % AR_SLOTS;
    const int w = threadIdx.x, warp = w >> 5, lane = w & 31;
    const int rr = blockIdx.x;                 // row of the step: its own partial, residual, normed row and words
    const bool pay = w < ARP;
    part += (size_t)rr * 2880;
    h += (size_t)rr * 2880;
    xn += (size_t)rr * 2880;
    const size_t so = ((size_t)slot * ngpu + rank) * MAXR + rr;   // this rank's words of the row in a receive buffer
    unsigned it = 0;
    if (warp < ARPW) {
        const ArKey key = ar_key((unsigned)w);
        unsigned p0 = 0, p1 = 0, p2 = 0, p3 = 0;
        if (pay) {
            const uint4 pv = *(const uint4*)(part + 4 * w);
            p0 = pv.x, p1 = pv.y, p2 = pv.z, p3 = pv.w;
#ifndef GPTOSS_AR_NOSTORE   // timing probe only (wrong output, with -DGPTOSS_NOWAIT): no stores into the peers' words
            for (int g = 0; g < a.nw; g++)
                if ((g != rank || a.sh) && !pushed)   // pushed: the kernel that made the partial stored the words already
                    asm volatile("st.volatile.global.v4.u32 [%0], {%1, %2, %3, %4};" ::"l"((uint4*)a.wr[g] + so * ARW + w), "r"(p0),
                                 "r"(p1), "r"(p2), "r"(p3) : "memory");
#endif
        }
        AR_MARK(1);
        {   // the warp's trailer: checksum of its own words, lane g stores it to peer g
            // the tag is part of the stored checksum (review r9 L1): an old checksum under a new tag never validates
            const unsigned long long c = warp_xor64(pay ? ar_mix(key, p0, p1, p2, p3) : 0ull) ^ (tag * 0x9E3779B97F4A7C15ull);
#ifndef GPTOSS_AR_NOSTORE
            if (lane < a.nw && (lane != rank || a.sh))
                asm volatile("st.volatile.global.v4.u32 [%0], {%1, %2, %3, %4};" ::"l"((uint4*)a.wr[lane] + so * ARW + ARP + warp),
                             "r"((unsigned)c), "r"((unsigned)(c >> 32)), "r"(0u), "r"(tag) : "memory");
#endif
        }
        tail_pf(pf, pfb, w, 32 * ARPW);   // while the peers' words travel
        float4 hv = make_float4(0.f, 0.f, 0.f, 0.f);
        if (pay) hv = *(const float4*)(h + 4 * w);   // the residual: read before the wait
        AR_MARK(2);
        const uint4* rb = (const uint4*)a.rd + ((size_t)slot * ngpu * MAXR + rr) * ARW;   // sender g: + g * MAXR * ARW
        const unsigned long long t0 = gtimer();
        const bool poll = lane < ngpu && lane != rank;   // lane g reads sender g's trailer of this warp
        float s0, s1, s2, s3;
        for (;;) {
            unsigned x[4][4], tl = 0, th = 0, tg = tag;
            // all loads of the pass first (one round trip), then the arithmetic
#pragma unroll
            for (int g = 0; g < 4; g++) {
                x[g][0] = p0, x[g][1] = p1, x[g][2] = p2, x[g][3] = p3;
                if (g < ngpu && g != rank && pay)
                    asm volatile("ld.volatile.global.v4.u32 {%0, %1, %2, %3}, [%4];"
                                 : "=r"(x[g][0]), "=r"(x[g][1]), "=r"(x[g][2]), "=r"(x[g][3]) : "l"(rb + (size_t)g * MAXR * ARW + w) : "memory");
            }
            if (poll) {
                unsigned z;
                asm volatile("ld.volatile.global.v4.u32 {%0, %1, %2, %3}, [%4];"
                             : "=r"(tl), "=r"(th), "=r"(z), "=r"(tg) : "l"(rb + (size_t)lane * MAXR * ARW + ARP + warp) : "memory");
                (void)z;   // (the word's third field is not used here: review r9 L7)
            }
            // a pass without all tags does no arithmetic; with them, one reduction for all senders: xor over the
            // lanes of (sender g's term rotated by 21 g) against the same combination of the trailers' checksums
            if (!AR_PENDING(!__all_sync(0xffffffffu, tg == tag))) {
                unsigned long long c = 0ull, want = (((unsigned long long)th << 32) | tl) ^ (tag * 0x9E3779B97F4A7C15ull);   // lane g: sender g's checksum
                want = poll ? (want << (21 * lane)) | (want >> ((64 - 21 * lane) & 63)) : 0ull;
                s0 = s1 = s2 = s3 = 0.f;
#pragma unroll
                for (int g = 0; g < 4; g++) {
                    if (g >= ngpu) break;
                    if (g != rank && pay) {
                        const unsigned long long m = ar_mix(key, x[g][0], x[g][1], x[g][2], x[g][3]);
                        c ^= g ? (m << (21 * g)) | (m >> (64 - 21 * g)) : m;
                    }
                    s0 += __uint_as_float(x[g][0]);
                    s1 += __uint_as_float(x[g][1]);
                    s2 += __uint_as_float(x[g][2]);
                    s3 += __uint_as_float(x[g][3]);
                }
                if (!AR_PENDING(warp_xor64(c ^ want) != 0ull)) break;   // the same for every lane of the warp
            }
            if ((++it & 1023u) == 0 && gtimer() - t0 > AR_WAIT_NS) __trap();
        }
        AR_MARK(4);
        if (pay) {
            const float4 v = make_float4(hv.x + s0, hv.y + s1, hv.z + s2, hv.w + s3);
            *(float4*)(h + 4 * w) = v;
            *(float4*)(sv + 4 * w) = v;
        }
    }
    const float n0 = bf2f(nw[3 * w]), n1 = bf2f(nw[3 * w + 1]), n2 = bf2f(nw[3 * w + 2]);
    __syncthreads();
    const float v0 = sv[3 * w], v1 = sv[3 * w + 1], v2 = sv[3 * w + 2];
    float ss = warp_sum(v0 * v0 + v1 * v1 + v2 * v2);
    if (lane == 0) red[warp] = ss;
    __syncthreads();
    if (w < 32) {
        float t = w < 30 ? red[w] : 0.f;
        t = warp_sum(t);
        if (w == 0) red[31] = t;
    }
    __syncthreads();
    const float rs = rsqrtf(red[31] / (float)(3 * ARW) + 1e-5f);
    xn[3 * w] = __float2bfloat16_rn(n0 * (v0 * rs));
    xn[3 * w + 1] = __float2bfloat16_rn(n1 * (v1 * rs));
    xn[3 * w + 2] = __float2bfloat16_rn(n2 * (v2 * rs));
#ifdef GPTOSS_AR_TIME
    if (w == 0 && rr == 0) {
        unsigned long long* acc = (unsigned long long*)((uint4*)a.rd + so * ARW + 800);
        tm[5] = gtimer();
        acc[1] += tm[1] - tm[0], acc[2] += tm[2] - tm[0], acc[4] += tm[4] - tm[0], acc[5] += tm[5] - tm[0];
        acc[0] += 1;
        acc[6] += it;
    }
#endif
}

// The same collective without peer access: 8-byte words (fp32 value, tag) in pinned host memory, one store per
// value by the sender, polled over PCIe by every rank. Same rank-order sum and the same tail: bitwise equal to k_ar.
__global__ void __launch_bounds__(960) k_ar_host(const float* part, float* h, const bf16* nw, bf16* xn,
                                                 const __grid_constant__ ArBuf a, int rank, int ngpu,
                                                 const unsigned* step, int idx) {
    const unsigned tag = *step * 256u + (unsigned)idx + 1u;
    const int slot = idx % AR_SLOTS;
    const int w = threadIdx.x;
    const int rr = blockIdx.x;
    part += (size_t)rr * 2880;
    h += (size_t)rr * 2880;
    xn += (size_t)rr * 2880;
    uint2* dst = (uint2*)a.wr[0] + (((size_t)slot * ngpu + rank) * MAXR + rr) * (3 * ARW) + 3 * w;
#pragma unroll
    for (int j = 0; j < 3; j++)
        asm volatile("st.volatile.global.v2.u32 [%0], {%1, %2};" ::"l"(dst + j), "r"(__float_as_uint(part[3 * w + j])), "r"(tag) : "memory");
    const uint2* rb = (const uint2*)a.rd + ((size_t)slot * ngpu * MAXR + rr) * (3 * ARW) + 3 * w;
    float s[3] = {0.f, 0.f, 0.f};
    const unsigned long long t0 = gtimer();
    for (int g = 0; g < ngpu; g++) {
#pragma unroll
        for (int j = 0; j < 3; j++) {
            uint2 x;
            unsigned it = 0;
            do {
                asm volatile("ld.volatile.global.v2.u32 {%0, %1}, [%2];" : "=r"(x.x), "=r"(x.y) : "l"(rb + (size_t)g * MAXR * (3 * ARW) + j) : "memory");
                if ((++it & 255u) == 0 && gtimer() - t0 > AR_WAIT_NS) __trap();
            } while (AR_PENDING(x.y != tag));
            s[j] += __uint_as_float(x.x);
        }
    }
    ar_tail(h, nw, xn, w, s[0], s[1], s[2]);
}

// ---------------------------------------------------------------- dense GEMV (decode)
// out[row] = W[row] . x + bias[row]; W bf16 [N][K] row-major, x bf16 [K]; 8 rows per block, one warp per row
// RW rows per warp one after another: fewer blocks, so the input is staged into shared memory RW times less often
template <int RW>
// R input rows x[r] (stride ldx) -> out[r] (stride ldo): the weight row is walked once per input row (from L1 after
// the first), with the same arithmetic for every row.
__global__ void __launch_bounds__(256) k_gemv(const bf16* W, int N, int K, const bf16* x, const bf16* bias, float* out,
                                              int R, int ldx, int ldo) {
    const int row0 = (blockIdx.x * 8 + (threadIdx.x >> 5)) * RW, lane = threadIdx.x & 31;
#pragma unroll
    for (int i = 0; i < RW; i++) {
        const int row = row0 + i;
        if (row >= N) return;
        for (int rr = 0; rr < R; rr++) {
            const float acc = row_dot(W + (size_t)row * K, x + (size_t)rr * ldx, K, lane);
            if (lane == 0) out[(size_t)rr * ldo + row] = acc + (bias ? bf2f(bias[row]) : 0.f);
        }
    }
}

// Router logits: block = one of the 128 rows, each warp takes 360 of its 2880 weights (2 loads deep instead of 12)
__global__ void __launch_bounds__(256) k_router(const bf16* W, const bf16* x, const bf16* bias, float* out) {
    __shared__ float red[8];
    x += (size_t)blockIdx.y * 2880;   // row of the step
    out += (size_t)blockIdx.y * 128;
    const int row = blockIdx.x, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const bf16* wr = W + (size_t)row * 2880 + warp * 360;
    const bf16* xr = x + warp * 360;
    float acc = 0.f;
    for (int u = lane; u < 45; u += 32) acc += dot8(*(const uint4*)(wr + u * 8), *(const uint4*)(xr + u * 8));
    acc = warp_sum(acc);
    if (lane == 0) red[warp] = acc;
    __syncthreads();
    if (threadIdx.x == 0) {
        float s = 0.f;
        for (int k = 0; k < 8; k++) s += red[k];
        out[row] = s + bf2f(bias[row]);
    }
}

// top-4 of the 128 router logits inside one warp (lane l holds logits 4l .. 4l+3), same rule as k_top4: ties go to
// the lowest index, softmax over the 4. Every lane returns the same result. Used by the selector block of k_ar1 (computing it
// in every gate_up block was slower, registry [gptoss-m3-top4fuse]).
__device__ __forceinline__ void top4_warp(const float* lg, int lane, int* ids, float* w) {
    const float4 v4 = *(const float4*)(lg + 4 * lane);
    float v[4] = {v4.x, v4.y, v4.z, v4.w};
    float tv[4];
#pragma unroll
    for (int r = 0; r < 4; r++) {
        float val = v[0];
        int idx = 4 * lane;
#pragma unroll
        for (int j = 1; j < 4; j++)
            if (v[j] > val) {
                val = v[j];
                idx = 4 * lane + j;
            }
#pragma unroll
        for (int o = 16; o; o >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, val, o);
            const int oi = __shfl_xor_sync(0xffffffffu, idx, o);
            if (ov > val || (ov == val && oi < idx)) {
                val = ov;
                idx = oi;
            }
        }
        tv[r] = val;
        ids[r] = idx;
        if ((idx >> 2) == lane) v[idx & 3] = NEG_INF;
    }
    float e[4], sum = 0.f;
#pragma unroll
    for (int k = 0; k < 4; k++) {
        e[k] = expf(tv[k] - tv[0]);
        sum += e[k];
    }
#pragma unroll
    for (int k = 0; k < 4; k++) w[k] = e[k] / sum;
}

// ---------------------------------------------------------------- split all-reduce (plan D1)
// The single 960-thread block is bound by one SM. Here 8 blocks of 120 words each send and receive; the norm's sum
// of squares goes through 8 tagged device words (every block reads all 8 and adds them in block order, so every
// block and every card gets the same bits). k_ar1 also carries the router: 128 blocks (one logit row each) wait for
// the 8 "normed row written" words, publish their logit as a tagged word, and one selector block takes the top 4.
struct ArAux {
    uint2 *ss, *ready, *lg;   // device words (value, tag): [8] partial sums of squares, [8] ready, [128] logits
};
__device__ __forceinline__ void st_tag(uint2* p, unsigned v, unsigned tag) {
    asm volatile("st.volatile.global.v2.u32 [%0], {%1, %2};" ::"l"(p), "r"(v), "r"(tag) : "memory");
}
__device__ __forceinline__ unsigned wait_tag(const uint2* p, unsigned tag, unsigned long long t0) {
    uint2 x;
    unsigned it = 0;
    do {
        asm volatile("ld.volatile.global.v2.u32 {%0, %1}, [%2];" : "=r"(x.x), "=r"(x.y) : "l"(p) : "memory");
        if ((++it & 4095u) == 0 && gtimer() - t0 > AR_WAIT_NS) __trap();
    } while (AR_PENDING(x.y != tag));
    return x.x;
}
// one of the 8 all-reduce blocks (b), 128 threads launched, 120 active
__device__ __forceinline__ void ar_block(const float* part, float* h, const bf16* nw, bf16* xn, const ArBuf& a, int rank,
                                         int ngpu, unsigned tag, int slot, const ArAux& aux, int b, bool publish) {
    __shared__ float red[4];
    __shared__ float tot[8];
    const int t = threadIdx.x;
    const bool on = t < 120;
    const int w = b * 120 + t;
    const unsigned long long t0 = gtimer();
    float v0 = 0.f, v1 = 0.f, v2 = 0.f;
    if (on) {
        const float p0 = part[3 * w], p1 = part[3 * w + 1], p2 = part[3 * w + 2];
        for (int g = 0; g < a.nw; g++) {
            uint4* dst = (uint4*)a.wr[g] + ((size_t)slot * ngpu + rank) * ARW + w;
            asm volatile("st.volatile.global.v4.u32 [%0], {%1, %2, %3, %4};" ::"l"(dst), "r"(__float_as_uint(p0)),
                         "r"(__float_as_uint(p1)), "r"(__float_as_uint(p2)), "r"(tag) : "memory");
        }
        const uint4* rb = (const uint4*)a.rd + (size_t)slot * ngpu * ARW + w;
        float s0 = 0.f, s1 = 0.f, s2 = 0.f;
        for (int g = 0; g < ngpu; g++) {
            uint4 x;
            unsigned it = 0;
            do {
                asm volatile("ld.volatile.global.v4.u32 {%0, %1, %2, %3}, [%4];"
                             : "=r"(x.x), "=r"(x.y), "=r"(x.z), "=r"(x.w) : "l"(rb + (size_t)g * ARW) : "memory");
                if ((++it & 4095u) == 0 && gtimer() - t0 > AR_WAIT_NS) __trap();
            } while (AR_PENDING(x.w != tag));
            s0 += __uint_as_float(x.x);
            s1 += __uint_as_float(x.y);
            s2 += __uint_as_float(x.z);
        }
        v0 = h[3 * w] + s0;
        v1 = h[3 * w + 1] + s1;
        v2 = h[3 * w + 2] + s2;
        h[3 * w] = v0;
        h[3 * w + 1] = v1;
        h[3 * w + 2] = v2;
    }
    const float ss = warp_sum(v0 * v0 + v1 * v1 + v2 * v2);
    if ((t & 31) == 0) red[t >> 5] = ss;
    __syncthreads();
    if (t == 0) st_tag(aux.ss + b, __float_as_uint(((red[0] + red[1]) + red[2]) + red[3]), tag);
    if (t < 8) tot[t] = __uint_as_float(wait_tag(aux.ss + t, tag, t0));
    __syncthreads();
    float sum = tot[0];
#pragma unroll
    for (int k = 1; k < 8; k++) sum += tot[k];
    const float rs = rsqrtf(sum / (float)(3 * ARW) + 1e-5f);
    if (on) {
        xn[3 * w] = __float2bfloat16_rn(bf2f(nw[3 * w]) * (v0 * rs));
        xn[3 * w + 1] = __float2bfloat16_rn(bf2f(nw[3 * w + 1]) * (v1 * rs));
        xn[3 * w + 2] = __float2bfloat16_rn(bf2f(nw[3 * w + 2]) * (v2 * rs));
    }
    if (publish) {
        __syncthreads();
        __threadfence();
        if (t == 0) st_tag(aux.ready + b, 1u, tag);
    }
}
__global__ void __launch_bounds__(128) k_ar2(const float* part, float* h, const bf16* nw, bf16* xn,
                                             const __grid_constant__ ArBuf a, int rank, int ngpu, const unsigned* step,
                                             int idx, const __grid_constant__ ArAux aux) {
    ar_block(part, h, nw, xn, a, rank, ngpu, *step * 256u + (unsigned)idx + 1u, idx % AR_SLOTS, aux, blockIdx.x, false);
}
// blocks 0..7: all-reduce + norm; 8..135: router row (block - 8); 136: top-4 + softmax -> ids, rw
__global__ void __launch_bounds__(128) k_ar1(const float* part, float* h, const bf16* nw, bf16* xn,
                                             const __grid_constant__ ArBuf a, int rank, int ngpu, const unsigned* step,
                                             int idx, const __grid_constant__ ArAux aux, const bf16* W, const bf16* bias,
                                             int* ids, float* rw) {
    const unsigned tag = *step * 256u + (unsigned)idx + 1u;
    const int b = blockIdx.x, t = threadIdx.x;
    if (b < 8) {
        ar_block(part, h, nw, xn, a, rank, ngpu, tag, idx % AR_SLOTS, aux, b, true);
        return;
    }
    const unsigned long long t0 = gtimer();
    if (b < 136) {
        __shared__ float red[4];
        if (t < 8) wait_tag(aux.ready + t, tag, t0);
        __syncthreads();
        const int row = b - 8, warp = t >> 5, lane = t & 31;
        const bf16* wr = W + (size_t)row * 2880 + warp * 720;
        const bf16* xr = xn + warp * 720;
        float acc = 0.f;
#pragma unroll
        for (int u = lane; u < 90; u += 32) acc += dot8(*(const uint4*)(wr + u * 8), *(const uint4*)(xr + u * 8));
        acc = warp_sum(acc);
        if (lane == 0) red[warp] = acc;
        __syncthreads();
        if (t == 0) st_tag(aux.lg + row, __float_as_uint((((red[0] + red[1]) + red[2]) + red[3]) + bf2f(bias[row])), tag);
        return;
    }
    __shared__ __align__(16) float lgs[128];
    lgs[t] = __uint_as_float(wait_tag(aux.lg + t, tag, t0));
    __syncthreads();
    if (t < 32) {
        int id4[4];
        float w4[4];
        top4_warp(lgs, t, id4, w4);
        if (t < 4) {
            ids[t] = id4[t];
            rw[t] = w4[t];
        }
    }
}

// q | k | v of one decode row. The card's matrix holds its query heads, then its K heads, then its V heads, 64 rows
// each. A warp computes rows d and d + 32 of one head, so it holds a RoPE pair: q and k are rotated (cos / sin carry
// the YaRN attention factor), k and v are stored as bf16 at the row of *posp (ring: row = pos & 255).
// ring: 0 = full-layer rows (sequence base + position), 1 = the sequence's ring set, 2 = row rr of a staging buffer
// (8-bit K/V: k_kv_pack takes the step's rows from there).
__global__ void __launch_bounds__(256) k_qkv(const bf16* W, const bf16* bias, const bf16* x, int nq, int kvn,
                                             const int* posp, const float* invf, float att, float* q, bf16* Kd,
                                             bf16* Vd, int ring, int R) {
    const int pair = blockIdx.x * 8 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    const int head = pair >> 5, d = pair & 31;
    if (head >= nq + 2 * kvn) return;
    const int r0 = head * 64 + d, r1 = r0 + 32;
    // Rows of one step: row rr is input x[rr] at position *posp + rr. The warp walks its two weight rows once per
    // input row (after the first pass they come from L1), with the same arithmetic for every row: a row's result
    // does not depend on R (row-count invariance).
    for (int rr = 0; rr < R; rr++) {
    const bf16* xs = x + (size_t)rr * 2880;
    float a, b;
    row_dot2(W + (size_t)r0 * 2880, W + (size_t)r1 * 2880, xs, 2880, lane, a, b);
    if (lane) continue;
    a += bf2f(bias[r0]);
    b += bf2f(bias[r1]);
    const int pos = posp[rr];
    if (head < nq + kvn) {
        const float fr = (float)pos * invf[d];
        const float c = cosf(fr) * att, s = sinf(fr) * att;
        const float x1 = a * c - b * s, x2 = b * c + a * s;
        a = x1;
        b = x2;
    }
    if (head < nq) {
        q[(size_t)rr * nq * 64 + r0] = a;
        q[(size_t)rr * nq * 64 + r1] = b;
        continue;
    }
    const size_t row = ring == 2 ? (size_t)rr : ring ? (size_t)posp[2 * MAXR + rr] * RING_ROWS + (size_t)(pos & 255) : (size_t)posp[MAXR + rr] + (size_t)pos;
    const bool isk = head < nq + kvn;
    bf16* dst = (isk ? Kd : Vd) + (row * kvn + (head - nq - (isk ? 0 : kvn))) * 64;
    dst[d] = __float2bfloat16_rn(a);
    dst[d + 32] = __float2bfloat16_rn(b);
    }
}

// ---------------------------------------------------------------- dense products with the rows of a step as tensor-core columns (round 7)
// out[c][row] = W[row] . x[c] for 16 weight rows and up to 8 NC input rows; bf16 operands, fp32 accumulation in the
// MMA. A block owns the 16 weight rows wa + g and wb + g (g = 0..7); its warps split K (K % 32 == 0) into runs of
// 32-weight groups. In a group, lane (g, t) takes 16 B = 8 weights of row wa + g and of row wb + g at element 8 t,
// and the same 8 elements of input row g (and g + 8): two MMAs per group, the k index of the MMA being a fixed
// permutation of the 16 elements a quad of lanes holds (the same permutation on both operands). So every weight
// load is a coalesced 16 B load and the weights go from memory into the fragments as they are.
// A column's sums see only that column's input: the value of (row, input row) is the same for every R and every
// column the input sits in (gptoss_kbench d7check), which is what batch == alone and drafts == plain need; the
// single-row step runs the same kernels. The warp partials are added in warp order.
// After the call warp 0's lane (g, t) holds c[4 n + 0..3] = rows wa + g / wa + g / wb + g / wb + g of input rows
// 8 n + 2 t / 2 t + 1 / 2 t / 2 t + 1; the other warps return false.
constexpr int DM_MAXW = 16;
__host__ __device__ constexpr int dm_smem(int NC, int warps) { return warps * 4 * NC * 32 * 4; }
// warps per block: ~9 groups (288 weights of each of the 16 rows) per warp: 10 for K = 2880, 6 for 1536, 4 for 1024
// (gptoss_kbench, card A: q/k/v 15.2 us at 5-15 warps; o_proj 12.1 / 12.4 / 12.4 us at 6 / 8 / 12)
__host__ __device__ constexpr int dm_warps(int K) { return ((K >> 5) + 8) / 9 > DM_MAXW ? DM_MAXW : ((K >> 5) + 8) / 9; }
__device__ __forceinline__ void mma_bf16(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
template <int NC, int UG>
__device__ __forceinline__ bool dense_mma(const bf16* wa, const bf16* wb, int K, const bf16* x, int ldx, int R, float* red, float* c) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3, nw = blockDim.x >> 5;
    const int G = K >> 5, j0 = warp * G / nw, j1 = (warp + 1) * G / nw;
    const uint4* pa = (const uint4*)(wa + (size_t)g * K) + t;       // 4 uint4 per group
    const uint4* pb = (const uint4*)(wb + (size_t)g * K) + t;
    const uint4* px[NC];
    bool xon[NC];
#pragma unroll
    for (int n = 0; n < NC; n++) {
        xon[n] = 8 * n + g < R;
        px[n] = (const uint4*)(x + (size_t)(xon[n] ? 8 * n + g : 0) * ldx) + t;
    }
#pragma unroll
    for (int i = 0; i < 4 * NC; i++) c[i] = 0.f;
    int j = j0;
    for (; j + UG <= j1; j += UG) {   // UG groups per round: 2 UG weight loads of a lane in flight
        uint4 a[UG], b[UG];
#pragma unroll
        for (int u = 0; u < UG; u++) {
            a[u] = pa[4 * (j + u)];
            b[u] = pb[4 * (j + u)];
        }
#pragma unroll
        for (int n = 0; n < NC; n++)
#pragma unroll
            for (int u = 0; u < UG; u++) {
                uint4 xv = make_uint4(0, 0, 0, 0);
                if (xon[n]) xv = px[n][4 * (j + u)];
                mma_bf16(c + 4 * n, a[u].x, b[u].x, a[u].y, b[u].y, xv.x, xv.y);
                mma_bf16(c + 4 * n, a[u].z, b[u].z, a[u].w, b[u].w, xv.z, xv.w);
            }
    }
    for (; j < j1; j++) {
        const uint4 a = pa[4 * j], b = pb[4 * j];
#pragma unroll
        for (int n = 0; n < NC; n++) {
            uint4 xv = make_uint4(0, 0, 0, 0);
            if (xon[n]) xv = px[n][4 * j];
            mma_bf16(c + 4 * n, a.x, b.x, a.y, b.y, xv.x, xv.y);
            mma_bf16(c + 4 * n, a.z, b.z, a.w, b.w, xv.z, xv.w);
        }
    }
    if (nw == 1) return true;
#pragma unroll
    for (int i = 0; i < 4 * NC; i++) red[(warp * 4 * NC + i) * 32 + lane] = c[i];
    __syncthreads();
    if (warp) return false;
    for (int w = 1; w < nw; w++)
#pragma unroll
        for (int i = 0; i < 4 * NC; i++) c[i] += red[(w * 4 * NC + i) * 32 + lane];
    return true;
}
// o_proj (and any dense [N][K] product, N % 16 == 0): block = weight rows 16 b .. 16 b + 15
template <int NC, int UG = 2, int PUSH = 0>
__global__ void __launch_bounds__(32 * DM_MAXW) k_gemv_m(const bf16* W, int N, int K, const bf16* x, int ldx, int R,
                                                         const bf16* bias, float* out, int ldo, const __grid_constant__ ArPush ps = ArPush{}) {
    extern __shared__ float red[];   // dm_smem(NC, warps) bytes
    const int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const int ra = blockIdx.x * 16 + g, rb = ra + 8;
    float c[4 * NC];
    // loaded before the products: the epilogue then waits for nothing but the sums
    const float ba = bias ? bf2f(bias[ra]) : 0.f, bb = bias ? bf2f(bias[rb]) : 0.f;
    if (!dense_mma<NC, UG>(W + (size_t)blockIdx.x * 16 * K, W + (size_t)(blockIdx.x * 16 + 8) * K, K, x, ldx, R, red, c)) return;
#pragma unroll
    for (int n = 0; n < NC; n++)
#pragma unroll
        for (int h = 0; h < 2; h++) {
            const int col = 8 * n + 2 * t + h;
            const float va = c[4 * n + h] + ba, vb = c[4 * n + 2 + h] + bb;
            if (col < R) {
                out[(size_t)col * ldo + ra] = va;
                out[(size_t)col * ldo + rb] = vb;
            }
            if (PUSH) {   // (out row = H floats = the payload; the peers' row stride is 4 ARW floats) lanes g = 0 and 4 gather rows ra .. ra + 3 (lanes + 4, + 8, + 12) and store one word per peer
                const float a1 = __shfl_down_sync(0xffffffffu, va, 4), a2 = __shfl_down_sync(0xffffffffu, va, 8), a3 = __shfl_down_sync(0xffffffffu, va, 12);
                const float b1 = __shfl_down_sync(0xffffffffu, vb, 4), b2 = __shfl_down_sync(0xffffffffu, vb, 8), b3 = __shfl_down_sync(0xffffffffu, vb, 12);
#ifndef GPTOSS_AR_NOSTORE
                if (col < R && (g & 3) == 0)
                    for (int i = 0; i < ps.n; i++) {
                        float* d = ps.d[i] + (size_t)col * (4 * ARW);
                        asm volatile("st.volatile.global.v4.f32 [%0], {%1, %2, %3, %4};" ::"l"(d + ra), "f"(va), "f"(a1), "f"(a2), "f"(a3) : "memory");
                        asm volatile("st.volatile.global.v4.f32 [%0], {%1, %2, %3, %4};" ::"l"(d + rb), "f"(vb), "f"(b1), "f"(b2), "f"(b3) : "memory");
                    }
#endif
            }
        }
}
// q | k | v: block = 8 RoPE pairs (rows d and d + 32) of one head; the epilogue of k_qkv per (pair, input row)
template <int NC, int UG = 2>
__global__ void __launch_bounds__(32 * DM_MAXW) k_qkv_m(const bf16* W, const bf16* bias, const bf16* x, int nq, int kvn,
                                                        const int* posp, const float* invf, float att, float* q, bf16* Kd,
                                                        bf16* Vd, int ring, int R) {
    extern __shared__ float red[];   // dm_smem(NC, warps) bytes
    const int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const int head = blockIdx.x >> 2, d = (blockIdx.x & 3) * 8 + g;
    const int r0 = head * 64 + d, r1 = r0 + 32;
    float c[4 * NC];
    // everything the epilogue needs is loaded and computed before the products: it then waits for the sums only
    const float b0 = bf2f(bias[r0]), b1 = bf2f(bias[r1]);
    const bool rot = head < nq + kvn, isk = head < nq + kvn;
    float cs[2 * NC], sn[2 * NC];
    size_t krow[2 * NC];
#pragma unroll
    for (int i = 0; i < 2 * NC; i++) {
        const int rr = 8 * (i >> 1) + 2 * t + (i & 1);
        cs[i] = 1.f;
        sn[i] = 0.f;
        krow[i] = 0;
        if (rr >= R || threadIdx.x >= 32) continue;
        const int pos = posp[rr];
        if (rot) {
            const float fr = (float)pos * invf[d];
            cs[i] = cosf(fr) * att;
            sn[i] = sinf(fr) * att;
        }
        krow[i] = ring == 2 ? (size_t)rr : ring ? (size_t)posp[2 * MAXR + rr] * RING_ROWS + (size_t)(pos & 255) : (size_t)posp[MAXR + rr] + (size_t)pos;
    }
    if (!dense_mma<NC, UG>(W + (size_t)(r0 - g) * 2880, W + (size_t)(r1 - g) * 2880, 2880, x, 2880, R, red, c)) return;
#pragma unroll
    for (int i = 0; i < 2 * NC; i++) {
        const int n = i >> 1, h = i & 1, rr = 8 * n + 2 * t + h;
        if (rr >= R) continue;
        float a = c[4 * n + h] + b0, b = c[4 * n + 2 + h] + b1;
        if (rot) {
            // explicit operations: the NC = 1 and NC = 2 instances must round alike (review r7 L3)
            const float x1 = __fmaf_rn(a, cs[i], -__fmul_rn(b, sn[i])), x2 = __fmaf_rn(b, cs[i], __fmul_rn(a, sn[i]));
            a = x1;
            b = x2;
        }
        if (head < nq) {
            q[(size_t)rr * nq * 64 + r0] = a;
            q[(size_t)rr * nq * 64 + r1] = b;
            continue;
        }
        bf16* dst = (isk ? Kd : Vd) + (krow[i] * kvn + (head - nq - (isk ? 0 : kvn))) * 64;
        dst[d] = __float2bfloat16_rn(a);
        dst[d + 32] = __float2bfloat16_rn(b);
    }
}

// ---------------------------------------------------------------- attention (decode, one row)
// Block = (KV head, key split). A lane holds 16 dims of one of the 8 query heads of the group (lane = head * 4 +
// chunk); a warp takes every 8th key of the split: 32 B of K and 32 B of V per lane and key, online softmax in fp32.
// Output: unnormalised partials (m, l, o[64]) per (query head, split); the sink is added in k_attn_comb.
// Split rule (context length only): ns = min(NSPLIT, ceil(keys / SPLIT_MIN)), equal ranges.
// EP >= 1 (round 10; same values and sums as EP = 0, measured at R = 8 with 4K keys: the block's fixed part was 2 / 3
// of the kernel):
//  - the group's 8 x 64 q values are staged once per block (128 loads of 16 B) and read by the lanes as 4 conflict-free
//    16 B shared loads; EP = 0: 16 scalar loads per lane = 32 cache lines per warp instruction, 8 warps;
//  - the warps' outputs go to shared memory in a rotated order (word j * 32 + ((lane + 2 j) & 31)): the 32 lanes of a
//    store and the 32 consecutive elements of a merge load each hit 32 banks (EP = 0: 2 banks per store);
//  - each warp's merge factor exp(m_w - M) is made once (64 expf per block instead of 4,096), the merge has no
//    branch, and a key costs one expf instead of two;
//  - a warp's keys come in tiles of 4 (1 KB of K and V), copied to shared memory by cp.async one tile ahead (two
//    buffers per warp in the warp's `so` row, which is free until the loop ends); the first tile is asked for
//    before q is staged. EP = 0 has one key's loads in flight per warp (4 dependent DRAM round trips at 4K keys);
//    the kernel was bound by the loads in flight per SM (4 blocks by registers), not by its bytes or arithmetic.
template <int EP, int MB = 4>
__global__ void __launch_bounds__(256, EP ? MB : 0) k_attn(const float* q, const bf16* Kb, const bf16* Vb, int kvn, const int* posp,
                                              int sliding, float* apart) {
    __shared__ float sm[8][8], sl[8][8];
    __shared__ __align__(16) float so[8][512];
    __shared__ float sf[8][8], sM[8];
    __shared__ __align__(16) float4 sq[128];
    const int kh = blockIdx.x, split = blockIdx.y;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, hh = lane >> 2, c = lane & 3;
    const int pos = posp[blockIdx.z];                        // row blockIdx.z of the step: its own position, query and partials
    const size_t kbase = sliding ? (size_t)posp[2 * MAXR + blockIdx.z] * RING_ROWS : (size_t)posp[MAXR + blockIdx.z];
    q += (size_t)blockIdx.z * kvn * 512;
    apart += (size_t)blockIdx.z * kvn * 8 * NSPLIT * APW;
    int k0 = 0, k1 = 0;
    if (sliding) {
        k0 = max(0, pos - 127) + split * 32;
        k1 = min(pos + 1, k0 + 32);
    } else {
        const int nk = pos + 1;
        const int ns = min(NSPLIT, (nk + SPLIT_MIN - 1) / SPLIT_MIN);
        if (split < ns) {
            const int per = (nk + ns - 1) / ns;
            k0 = split * per;
            k1 = min(nk, k0 + per);
        }
    }
    float* ap = apart + ((size_t)(kh * 8) * NSPLIT + split) * APW;   // + head * NSPLIT * APW
    if (k0 >= k1) {
        if (threadIdx.x < 8) {
            float* p = ap + (size_t)threadIdx.x * NSPLIT * APW;
            p[0] = NEG_INF;
            p[1] = 0.f;
        }
        return;
    }
    float qv[16], o[16];
    float m = NEG_INF, l = 0.f;
    // one key: 64-dim dot per query head (lanes of a head add up by shuffles), online softmax step
    auto key_step = [&](const uint4 ka, const uint4 kb, const uint4 va, const uint4 vb) {
        const uint32_t kw[8] = {ka.x, ka.y, ka.z, ka.w, kb.x, kb.y, kb.z, kb.w};
        const uint32_t vw[8] = {va.x, va.y, va.z, va.w, vb.x, vb.y, vb.z, vb.w};
        float d = 0.f;
#pragma unroll
        for (int j = 0; j < 8; j++)
            d += qv[2 * j] * __uint_as_float(kw[j] << 16) + qv[2 * j + 1] * __uint_as_float(kw[j] & 0xffff0000u);
        d += __shfl_xor_sync(0xffffffffu, d, 1);
        d += __shfl_xor_sync(0xffffffffu, d, 2);
        const float s = d * 0.125f;
        const float mn = fmaxf(m, s);
        float fa, fp;
        if (EP) {
            // one of the two arguments is 0 and expf(0) is 1: one expf per key, the same two factors. The other
            // factor is written (x - mn) + 1: 1 for a finite score, NaN for +inf - +inf as expf gives; a NaN score
            // stays in the smaller argument (review r10 L2).
            const bool up = s > m;
            const float ex = expf((up ? m : s) - mn);
            const float one = ((up ? s : m) - mn) + 1.f;
            fa = up ? ex : one;
            fp = up ? one : ex;
        } else {
            fa = expf(m - mn);
            fp = expf(s - mn);
        }
        l = l * fa + fp;
#pragma unroll
        for (int j = 0; j < 8; j++) {
            o[2 * j] = o[2 * j] * fa + fp * __uint_as_float(vw[j] << 16);
            o[2 * j + 1] = o[2 * j + 1] * fa + fp * __uint_as_float(vw[j] & 0xffff0000u);
        }
        m = mn;
    };
#pragma unroll
    for (int j = 0; j < 16; j++) o[j] = 0.f;
    if (EP) {
        // tile t of the warp = keys k0 + warp + 8 (4 t + i), i = 0..3; buffer t & 1 = 64 slots of 16 B:
        // K of key i at slots 8 i .. 8 i + 7, V at 32 + 8 i ..; a lane copies one K and one V slot
        uint4* stg = (uint4*)so[warp];
        auto stage = [&](int t) {
            const int key = k0 + warp + 8 * (4 * t + (lane >> 3));
            if (key < k1) {
                const size_t row = kbase + (sliding ? (size_t)(key & 255) : (size_t)key);
                const size_t off = (row * kvn + kh) * 64 + (lane & 7) * 8;
                cp16(stg + (t & 1) * 64 + lane, Kb + off);
                cp16(stg + (t & 1) * 64 + 32 + lane, Vb + off);
            }
            cp_commit();
        };
        stage(0);
        if (threadIdx.x < 128) {
            const int t = threadIdx.x;   // float4 t of the group = (head t >> 4, chunk (t >> 2) & 3, quarter t & 3)
            sq[(t & 3) * 32 + (t >> 4) * 4 + ((t >> 2) & 3)] = ((const float4*)(q + kh * 8 * 64))[t];
        }
        __syncthreads();
#pragma unroll
        for (int i = 0; i < 4; i++) {
            const float4 v = sq[i * 32 + lane];
            qv[4 * i] = v.x;
            qv[4 * i + 1] = v.y;
            qv[4 * i + 2] = v.z;
            qv[4 * i + 3] = v.w;
        }
        for (int t = 0; k0 + warp + 32 * t < k1; t++) {
            if (k0 + warp + 32 * (t + 1) < k1) {
                stage(t + 1);
                cp_wait1();
            } else {
                cp_wait0();
            }
            // a lane reads slots that other lanes copied: the wait covers a thread's own copies, the barrier the
            // warp's (review r10 M1)
            __syncwarp();
            const uint4* tb = stg + (t & 1) * 64 + c * 2;
#pragma unroll
            for (int i = 0; i < 4; i++)
                if (k0 + warp + 8 * (4 * t + i) < k1) key_step(tb[8 * i], tb[8 * i + 1], tb[32 + 8 * i], tb[32 + 8 * i + 1]);
            __syncwarp();   // every lane is done with this buffer before the tile after next, or the merge stores, overwrite it
        }
    } else {
        const float* qp = q + (kh * 8 + hh) * 64 + c * 16;
#pragma unroll
        for (int j = 0; j < 16; j++) qv[j] = qp[j];
        for (int key = k0 + warp; key < k1; key += 8) {
            const size_t row = kbase + (sliding ? (size_t)(key & 255) : (size_t)key);
            const uint4* kp = (const uint4*)(Kb + (row * kvn + kh) * 64 + c * 16);
            const uint4* vp = (const uint4*)(Vb + (row * kvn + kh) * 64 + c * 16);
            key_step(kp[0], kp[1], vp[0], vp[1]);
        }
    }
    if (c == 0) {
        sm[warp][hh] = m;
        sl[warp][hh] = l;
    }
#pragma unroll
    for (int j = 0; j < 16; j++) so[warp][EP ? j * 32 + ((lane + 2 * j) & 31) : hh * 64 + c * 16 + j] = o[j];
    __syncthreads();
    if (EP) {
        // a warp without keys (m = NEG_INF) gets the factor -0 and its l and o are taken as +0: its terms are -0 and
        // x + (-0) is x for every x, both zeros included, so the merge needs no branch (EP = 0 skips such warps; a
        // live factor is never -0; review r10 L1)
        if (threadIdx.x < 64) {
            const int w = threadIdx.x >> 3, h2 = threadIdx.x & 7;
            float M = NEG_INF;
#pragma unroll
            for (int w2 = 0; w2 < 8; w2++) M = fmaxf(M, sm[w2][h2]);
            sf[w][h2] = sm[w][h2] > NEG_INF ? expf(sm[w][h2] - M) : -0.f;
            if (w == 0) sM[h2] = M;
        }
        __syncthreads();
        if (threadIdx.x < 8) {
            const int h2 = threadIdx.x;
            float L = 0.f;
#pragma unroll
            for (int w = 0; w < 8; w++) L += (__float_as_int(sf[w][h2]) < 0 ? 0.f : sl[w][h2]) * sf[w][h2];
            float* p = ap + (size_t)h2 * NSPLIT * APW;
            p[0] = sM[h2];
            p[1] = L;
        }
#pragma unroll
        for (int i = 0; i < 2; i++) {
            const int e = threadIdx.x + 256 * i, h2 = e >> 6;
            // element e = (head h2, chunk (e >> 4) & 3, j = e & 15) sits at the rotated word of its (lane, j)
            const int ei = (e & 15) * 32 + ((h2 * 4 + ((e >> 4) & 3) + 2 * (e & 15)) & 31);
            float O = 0.f;
#pragma unroll
            for (int w = 0; w < 8; w++) O += (__float_as_int(sf[w][h2]) < 0 ? 0.f : so[w][ei]) * sf[w][h2];
            ap[(size_t)h2 * NSPLIT * APW + 2 + (e & 63)] = O;
        }
        return;
    }
    for (int e = threadIdx.x; e < 512; e += 256) {
        const int h2 = e >> 6;
        float M = NEG_INF;
        for (int w = 0; w < 8; w++) M = fmaxf(M, sm[w][h2]);
        float L = 0.f, O = 0.f;
        for (int w = 0; w < 8; w++)
            if (sm[w][h2] > NEG_INF) {
                const float f = expf(sm[w][h2] - M);
                L += sl[w][h2] * f;
                O += so[w][e] * f;
            }
        float* p = ap + (size_t)h2 * NSPLIT * APW;
        p[2 + (e & 63)] = O;
        if ((e & 63) == 0) {
            p[0] = M;
            p[1] = L;
        }
    }
}
// One-kernel attention for the sliding layers (plan D3): block = KV head, its window (<= 128 rows of K and V, 32 KB)
// staged in shared memory, warp = one of the 8 query heads. Pass 1: a lane scores keys lane, lane + 32, ... (64-dim
// dots from shared memory); the warp takes the max and writes exp(s - max) per key. Pass 2: a lane owns 2 output
// dims and walks the keys in order. The sink goes into the denominator; the result is written as bf16: no combine.
__global__ void __launch_bounds__(256) k_attn_win(const float* q, const bf16* Kb, const bf16* Vb, int kvn, const int* posp,
                                                  const float* sinks, bf16* ao) {
    __shared__ __align__(16) bf16 ks[128 * 64];
    __shared__ __align__(16) bf16 vs[128 * 64];
    __shared__ float ps[8][128];
    const int kh = blockIdx.x, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int pos = *posp;
    const int k0 = max(0, pos - 127), nk = pos + 1 - k0;
    for (int u = threadIdx.x; u < nk * 8; u += 256) {   // 16 B units: key = u / 8
        const size_t row = (size_t)((k0 + (u >> 3)) & 255);
        const size_t off = (row * kvn + kh) * 64 + (u & 7) * 8;
        ((uint4*)ks)[u] = *(const uint4*)(Kb + off);
        ((uint4*)vs)[u] = *(const uint4*)(Vb + off);
    }
    __syncthreads();
    const int qh = kh * 8 + warp;
    const float4* qp = (const float4*)(q + qh * 64);
    float sc[4];
    float m = NEG_INF;
#pragma unroll
    for (int i = 0; i < 4; i++) {
        const int key = lane + 32 * i;
        sc[i] = NEG_INF;
        if (key < nk) {
            const uint4* kp = (const uint4*)(ks + key * 64);
            float d = 0.f;
#pragma unroll
            for (int j = 0; j < 8; j++) {
                const uint4 kv = kp[j];
                const float4 qa = qp[2 * j], qb = qp[2 * j + 1];
                d += qa.x * __uint_as_float(kv.x << 16) + qa.y * __uint_as_float(kv.x & 0xffff0000u) +
                     qa.z * __uint_as_float(kv.y << 16) + qa.w * __uint_as_float(kv.y & 0xffff0000u);
                d += qb.x * __uint_as_float(kv.z << 16) + qb.y * __uint_as_float(kv.z & 0xffff0000u) +
                     qb.z * __uint_as_float(kv.w << 16) + qb.w * __uint_as_float(kv.w & 0xffff0000u);
            }
            sc[i] = d * 0.125f;
            m = fmaxf(m, sc[i]);
        }
    }
    for (int o = 16; o; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
    float l = 0.f;
#pragma unroll
    for (int i = 0; i < 4; i++) {
        const int key = lane + 32 * i;
        const float pv = key < nk ? expf(sc[i] - m) : 0.f;
        ps[warp][key] = pv;
        l += pv;
    }
    l = warp_sum(l);
    __syncwarp();
    float o0 = 0.f, o1 = 0.f;
    const float* pw = ps[warp];
    for (int key = 0; key < nk; key++) {
        const uint32_t v = *(const uint32_t*)(vs + key * 64 + 2 * lane);
        const float pv = pw[key];
        o0 += pv * __uint_as_float(v << 16);
        o1 += pv * __uint_as_float(v & 0xffff0000u);
    }
    const float inv = 1.f / (l + expf(sinks[qh] - m));
    *(bf162*)(ao + qh * 64 + 2 * lane) = __floats2bfloat162_rn(o0 * inv, o1 * inv);
}
// merges the splits of one query head and adds the sink to the denominator: out = O / (L + exp(sink - M)), as bf16
// EP = 1 (round 10): the splits' (m, l) are loaded by all threads at once, the factors exp(m_s - M) are made once per
// split, and a thread issues its o loads 16 splits at a time and adds them without a branch (EP = 0: an expf per
// thread and split and a branch per term; 17 us per full layer at R = 8, 10 us at R = 1, where the two warps of a
// block run alone on their SM and every dependent step shows). Same values, same sums in the same order.
template <int EP>
__global__ void __launch_bounds__(64) k_attn_comb(const float* apart, const float* sinks, const int* posp, int sliding, bf16* ao,
                                                  const void* pf, size_t pfb) {
    tail_pf(pf, pfb, blockIdx.x * 64 + threadIdx.x, gridDim.x * 64);
    const int qh = blockIdx.x, d = threadIdx.x;
    apart += (size_t)blockIdx.y * gridDim.x * NSPLIT * APW;   // row blockIdx.y of the step
    ao += (size_t)blockIdx.y * gridDim.x * 64;
    const int nk = posp[blockIdx.y] + 1;   // the split rule of k_attn
    const int nsp = sliding ? min(SSPLIT, (min(nk, 128) + 31) / 32) : min(NSPLIT, (nk + SPLIT_MIN - 1) / SPLIT_MIN);
    const float* ap = apart + (size_t)qh * NSPLIT * APW;
    if (EP && nsp > 8) {   // up to 8 splits (every sliding layer, full layers up to 256 keys) the plain loop below is as fast or faster
        // an empty split has m = NEG_INF and l = +0 (its o words are not written): factor -0, o taken as +0, so its
        // terms are -0 and x + (-0) is x for every x, both zeros included: the sums need no branch (EP = 0 skips such
        // splits). A live split's factor is never -0 (it may be +0 by underflow: then v * 0 is added as in EP = 0), so
        // the factor's sign bit tells the two apart (review r10 L1).
        __shared__ float cl[NSPLIT], cf[NSPLIT], cM[2];
        float ms[2] = {NEG_INF, NEG_INF};
#pragma unroll
        for (int i = 0; i < 2; i++) {
            const int s = d + 64 * i;
            if (s < nsp) {
                ms[i] = ap[s * APW];
                cl[s] = ap[s * APW + 1];
            }
        }
        float mx = fmaxf(ms[0], ms[1]);
        for (int o = 16; o; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
        if ((d & 31) == 0) cM[d >> 5] = mx;
        __syncthreads();
        const float M = fmaxf(cM[0], cM[1]);
#pragma unroll
        for (int i = 0; i < 2; i++) {
            const int s = d + 64 * i;
            if (s < nsp) cf[s] = ms[i] > NEG_INF ? expf(ms[i] - M) : -0.f;
        }
        __syncthreads();
        float L = 0.f, O = 0.f;
        for (int s0 = 0; s0 < nsp; s0 += 16) {
            float v[16], f[16], lv[16];
#pragma unroll
            for (int j = 0; j < 16; j++) {
                const bool on = s0 + j < nsp;
                f[j] = on ? cf[s0 + j] : -0.f;
                lv[j] = on ? cl[s0 + j] : 0.f;
                v[j] = ap[(s0 + j) * APW + 2 + d];   // inside the head's NSPLIT partials for every j
            }
#pragma unroll
            for (int j = 0; j < 16; j++) {
                L += lv[j] * f[j];
                O += (__float_as_int(f[j]) < 0 ? 0.f : v[j]) * f[j];
            }
        }
        ao[qh * 64 + d] = __float2bfloat16_rn(O / (L + expf(sinks[qh] - M)));
        return;
    }
    float M = NEG_INF;
    for (int s = 0; s < nsp; s++) M = fmaxf(M, ap[s * APW]);
    float L = 0.f, O = 0.f;
    for (int s = 0; s < nsp; s++)
        if (ap[s * APW] > NEG_INF) {
            const float f = expf(ap[s * APW] - M);
            L += ap[s * APW + 1] * f;
            O += ap[s * APW + 2 + d] * f;
        }
    ao[qh * 64 + d] = __float2bfloat16_rn(O / (L + expf(sinks[qh] - M)));
}

// ---------------------------------------------------------------- router
// top-4 of 128 logits per row (ties: the lowest index wins), softmax over the 4; block per row
__global__ void __launch_bounds__(128) k_top4(const float* lg, int* ids, float* w) {
    __shared__ float v[128];
    __shared__ float bv[4];
    __shared__ int bi[4];
    __shared__ int pick;
    const int t = threadIdx.x;
    const size_t row = blockIdx.x;
    v[t] = lg[row * 128 + t];
    __syncthreads();
    float tv[4];
    for (int r = 0; r < 4; r++) {
        float val = v[t];
        int idx = t;
        for (int o = 16; o; o >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, val, o);
            const int oi = __shfl_xor_sync(0xffffffffu, idx, o);
            if (ov > val || (ov == val && oi < idx)) {
                val = ov;
                idx = oi;
            }
        }
        if ((t & 31) == 0) {
            bv[t >> 5] = val;
            bi[t >> 5] = idx;
        }
        __syncthreads();
        if (t == 0) {
            float b = bv[0];
            int i = bi[0];
            for (int k = 1; k < 4; k++)
                if (bv[k] > b || (bv[k] == b && bi[k] < i)) {
                    b = bv[k];
                    i = bi[k];
                }
            pick = i;
            tv[r] = b;
            ids[row * 4 + r] = i;
        }
        __syncthreads();
        if (t == pick) v[t] = NEG_INF;
        __syncthreads();
    }
    if (t == 0) {
        float e[4], s = 0.f;
        for (int k = 0; k < 4; k++) {
            e[k] = expf(tv[k] - tv[0]);
            s += e[k];
        }
        for (int k = 0; k < 4; k++) w[row * 4 + k] = e[k] / s;
    }
}

// Router logits of all rows of a decode step (round 9): block = one of the 128 router rows, its weights held in
// registers (a lane's one or two 16 B loads) across the step's rows. A row's sums are k_router's, in the same order
// (lane: the group at u = lane, then u = lane + 32; warp tree; the 8 warps in order; + bias), so the logits are
// bitwise the same; the weights are read once instead of once per row and the launch is 128 blocks at every R.
template <int C>   // C rows per round: their input loads are issued together, then the products
__global__ void __launch_bounds__(256) k_router_r(const bf16* W, const bf16* x, const bf16* bias, float* out, int R) {
    __shared__ float red[MAXR][8];
    const int row = blockIdx.x, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const bf16* wr = W + (size_t)row * 2880 + warp * 360;
    const bool two = lane + 32 < 45;
    const uint4 w0 = *(const uint4*)(wr + lane * 8);
    uint4 w1 = make_uint4(0, 0, 0, 0);
    if (two) w1 = *(const uint4*)(wr + (lane + 32) * 8);
    const float b = bf2f(bias[row]);
    for (int r0 = 0; r0 < R; r0 += C) {
        uint4 x0[C], x1[C];
#pragma unroll
        for (int c = 0; c < C; c++) {
            const bf16* xr = x + (size_t)(r0 + c < R ? r0 + c : 0) * 2880 + warp * 360;
            x0[c] = *(const uint4*)(xr + lane * 8);
            x1[c] = make_uint4(0, 0, 0, 0);
            if (two) x1[c] = *(const uint4*)(xr + (lane + 32) * 8);
        }
#pragma unroll
        for (int c = 0; c < C; c++) {
            float acc = 0.f;
            acc += dot8(w0, x0[c]);
            if (two) acc += dot8(w1, x1[c]);
            acc = warp_sum(acc);
            if (lane == 0 && r0 + c < R) red[r0 + c][warp] = acc;
        }
    }
    __syncthreads();
    if (threadIdx.x < R) {
        const int rr = threadIdx.x;
        float s = 0.f;
        for (int k = 0; k < 8; k++) s += red[rr][k];
        out[(size_t)rr * 128 + row] = s + b;
    }
}
// top-4 + softmax of every row of a decode step and the slot order in one block (round 9): warp rr takes row rr
// (top4_warp: the picks, ties and weights of k_top4), then thread s < 4 R ranks slot s by (expert id, slot) as
// k_order does (do_order = 0: no order array). Launch: 32 R threads.
__global__ void __launch_bounds__(32 * MAXR) k_top4_ord(const float* lg, int* ids, float* w, int R, int do_order, int* order) {
    __shared__ int sid[4 * MAXR];
    const int rr = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int id4[4];
    float w4[4];
    top4_warp(lg + (size_t)rr * 128, lane, id4, w4);
    if (lane < 4) {
        ids[rr * 4 + lane] = id4[lane];
        w[rr * 4 + lane] = w4[lane];
        sid[rr * 4 + lane] = id4[lane];
    }
    if (!do_order) return;
    __syncthreads();
    const int s = threadIdx.x, n = 4 * R;
    if (s >= n) return;
    const int my = sid[s];
    int place = 0;
    for (int u = 0; u < n; u++) {
        const int v = sid[u];
        place += (v < my) | ((v == my) & (u < s));
    }
    order[place] = s;
}

// ---------------------------------------------------------------- experts (decode, one row)
// gate_up of the 4 routed experts on this card's intermediate slice: rows [2 j0, 2 j1) of each expert, NR = 2 (j1 -
// j0) rows of 90 MXFP4 blocks. Block = 8 rows of one expert (4 gate / up pairs), a lane takes one 16 B block per
// step. Epilogue: clamped SwiGLU times the routing weight, rounded to bf16 (the operand of the down product).
// PF = 1 keeps the down-record prefetch at the block's end (GPTOSS_PF_DN_KB, a lost try); PF = 0, the default, has
// no trace of it: the same sums, 18.6 -> 17.3 us in a graph (round 6).
// The slots of a multi-row step in the order of their expert ids (round 8): blockIdx.y is a slot's place in that
// order (`order`, from k_order; one row: the identity), so the slots that picked one expert are neighbours in the
// launch order, run in the same wave, and all but the first read the expert's records from L2 (bytes-only probe at
// R = 8, 32 slots on 24 experts: 104.6 -> 82.2 us, the time of the distinct bytes). A slot's sums are the same
// wherever it runs. The array is passed unconditionally: `order ? order[y] : y` costs 62 registers instead of 48
// (4 blocks per SM instead of 5) and with them the gain.
// order[place] = slot, places by (expert id, slot); n = 4 R <= 64 slots, one block
static_assert(4 * MAXR <= 64, "k_order ranks the slots of a step with one thread per slot in a 64-thread block");
__global__ void __launch_bounds__(64) k_order(const int* ids, int n, int* order) {
    const int s = threadIdx.x;
    if (s >= n) return;
    const int my = ids[s];
    int place = 0;
    for (int u = 0; u < n; u++) {
        const int v = ids[u];
        place += (v < my) | ((v == my) & (u < s));
    }
    order[place] = s;
}
template <int PF>
__global__ void __launch_bounds__(256) k_gu(const uint8_t* cod, const bf16* bias, int NR,
                                            const bf16* x, const int* ids, const float* rw, bf16* act,
                                            const uint8_t* dnc, size_t dn_expert_bytes, size_t pfb, const int* order) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int slot = order[blockIdx.y];   // slot = row of the step * 4 + routed expert
    const int k = slot & 3, rr = slot >> 2;
    const bf16* xs = x + (size_t)rr * 2880;
    ids += rr * 4;
    rw += rr * 4;
    act += (size_t)rr * 4 * (NR / 2);
    const int e = ids[k];
    const int row0 = (blockIdx.x * 8 + warp) * 4;   // a warp owns 2 gate / up pairs
    float g[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int i = 0; i < 4; i += 2) {   // gate and up row of one pair together
        const int row = row0 + i;
        if (row >= NR) continue;
        const size_t er = (size_t)e * NR + row;
        const uint8_t* cr = cod + er * 1488;   // mx_rs(90); the up row is the next record
        float ag = 0.f, au = 0.f;
#pragma unroll
        for (int it = 0; it < 3; it++) {
            const int b = it * 32 + lane;
            if (b < 90) {
                const uint4 cg = *(const uint4*)(cr + b * 16), cu = *(const uint4*)(cr + 1488 + b * 16);
                const float sg = MX_SCALE_AT(cr, 90, b), su = MX_SCALE_AT(cr + 1488, 90, b);
                ag += fp4dot32(cg, xs + b * 32) * sg;
                au += fp4dot32(cu, xs + b * 32) * su;
            }
        }
        g[i] = warp_sum(ag) + bf2f(bias[er]);
        g[i + 1] = warp_sum(au) + bf2f(bias[er + 1]);
    }
    if (lane < 2 && row0 + 2 * lane < NR)
        act[k * (NR / 2) + row0 / 2 + lane] = __float2bfloat16_rn(swiglu(g[2 * lane], g[2 * lane + 1]) * rw[k]);
    // at the block's end: this block's share of the routed expert's down records into L2 (they are the next kernel's
    // bytes; all gate_up blocks end together, so the requests travel during the kernel boundary)
    if (PF && pfb) {
        const size_t per = min(pfb, dn_expert_bytes) / gridDim.x;
        const char* p0 = (const char*)dnc + (size_t)e * dn_expert_bytes + (dn_expert_bytes / gridDim.x) * blockIdx.x;
        for (size_t o = (size_t)threadIdx.x * 128; o < per; o += 256 * 128) asm volatile("prefetch.global.L2 [%0];" ::"l"(p0 + o));
    }
}
// down of the 4 routed experts summed: out row = sum_k act_k . down_k[row][this card's KB blocks] (+ on rank 0 the
// weighted biases). Block = 8 output rows, a lane takes one 16 B block of each expert.
__global__ void __launch_bounds__(256) k_dn(const uint8_t* cod, int KB, const bf16* act,
                                            const int* ids, const float* rw, const bf16* bias, float* part,
                                            const void* pf, size_t pfb) {
    const int rr = blockIdx.y;                 // row of the step
    const bf16* as = act + (size_t)rr * 4 * KB * 32;
    ids += rr * 4;
    rw += rr * 4;
    part += (size_t)rr * 2880;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int row = blockIdx.x * 8 + warp;
    tail_pf(pf, pfb, blockIdx.x * 256 + threadIdx.x, gridDim.x * 256);   // issued first: the loads below overlap it
    if (row >= 2880) return;
    const size_t RS = mx_rs(KB);
    float acc = 0.f;
#pragma unroll
    for (int k = 0; k < 4; k++) {
        const size_t er = (size_t)ids[k] * 2880 + row;
        for (int b = lane; b < KB; b += 32)
            acc += fp4dot32(*(const uint4*)(cod + er * RS + b * 16), as + (k * KB + b) * 32) * MX_SCALE_AT(cod + er * RS, KB, b);
    }
    acc = warp_sum(acc);
    if (lane == 0) {
        if (bias)
            for (int k = 0; k < 4; k++) acc += rw[k] * bf2f(bias[(size_t)ids[k] * 2880 + row]);
        part[row] = acc;
    }
}

// down with the block steps unrolled (NIT = code loads per lane and expert: 1 for KB <= 32, 2 for KB <= 64) and no
// tail prefetch: the sums of k_dn, bit for bit (gptoss_kbench d6check). Round 6 measured the floor of this access
// pattern (every weight load issued, no products): 14.4 us for gate_up and ~8.9 us for down on card A, of which an
// empty launch is ~1 us in a graph; the forms that tried to get closer (cp.async rings, all loads up front, volatile
// or prefetch statements, fp32 or shared-memory inputs, a byte table) all lost:.
template <int NIT>
__global__ void __launch_bounds__(256) k_dn_n(const uint8_t* cod, int KB, const bf16* act,
                                              const int* ids, const float* rw, const bf16* bias, float* part) {
    // grid (step rows, 360): the blocks of all step rows for the same 8 output rows are neighbours in the launch
    // order, so an expert picked by several rows of the step is read from DRAM once and from L2 after that (round 8)
    const int rr = blockIdx.x;                 // row of the step
    const bf16* as = act + (size_t)rr * 4 * KB * 32;
    ids += rr * 4;
    rw += rr * 4;
    part += (size_t)rr * 2880;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int row = blockIdx.y * 8 + warp;
    if (row >= 2880) return;
    const size_t RS = mx_rs(KB);
    const int i0 = ids[0], i1 = ids[1], i2 = ids[2], i3 = ids[3];
    const int idk[4] = {i0, i1, i2, i3};
    float acc = 0.f;
#pragma unroll
    for (int k = 0; k < 4; k++) {
        const uint8_t* rec = cod + ((size_t)idk[k] * 2880 + row) * RS;
#pragma unroll
        for (int it = 0; it < NIT; it++) {
            const int b = lane + 32 * it;
            if (b < KB) acc += fp4dot32(*(const uint4*)(rec + b * 16), as + (k * KB + b) * 32) * MX_SCALE_AT(rec, KB, b);
        }
    }
    acc = warp_sum(acc);
    if (lane == 0) {
        if (bias)
            for (int k = 0; k < 4; k++) acc += rw[k] * bf2f(bias[(size_t)idk[k] * 2880 + row]);
        part[row] = acc;
    }
}

// ---------------------------------------------------------------- expert kernels of a multi-row step, inputs held in registers (round 8)
// The per-slot kernels load a lane's 64 B of inputs again for every weight row (4 of 5 loads in down, 2 of 3 in
// gate_up); with the bytes of a shared expert coming from L2 that load traffic is the bound (fp32 inputs, twice the
// input bytes: twice the time). Here a warp owns N weight rows (gate_up: N pairs) and walks the inputs in the outer
// loop: a lane loads its 64 B once and uses them for the N rows. Every accumulator sees its terms in the order of
// k_gu / k_dn_n (block steps, experts ascending), so the sums are the same bit for bit (kbench d8check).
// The held inputs cost registers (64-69 instead of 40-48), so it pays only in part of the range (card A, us,
// per-row kernel -> this one): gate_up R = 2 / 4 / 8 / 16 34.2 -> 28.9 / 60.9 -> 55.7 / 90.6 -> 92.3 / 159.7 ->
// 168.1; down (128 threads) 18.2 -> 17.7 / 29.6 -> 30.4 / 51.7 -> 46.8 / 98.5 -> 82.0. The engine takes it for
// gate_up at 2-4 rows and for down from 8 rows (GPTOSS_XREG=0: never).
template <int N>
__global__ void __launch_bounds__(256) k_gu_x(const uint8_t* cod, const bf16* bias, int NR,
                                              const bf16* x, const int* ids, const float* rw, bf16* act, const int* order) {
    static_assert(N <= 32, "the lane == n selection needs N <= 32 (review r8 p2 L1)");
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int slot = order[blockIdx.y];   // slot = row of the step * 4 + routed expert
    const int k = slot & 3, rr = slot >> 2;
    const bf16* xs = x + (size_t)rr * 2880;
    const int e = ids[slot];
    const int row0 = (blockIdx.x * (blockDim.x >> 5) + warp) * 2 * N;   // a warp owns N gate / up pairs
    if (row0 >= NR) return;
    const size_t er0 = (size_t)e * NR + row0;
    const uint8_t* cr0 = cod + er0 * 1488;   // mx_rs(90); the up row is the next record
    float ag[N], au[N];
#pragma unroll
    for (int n = 0; n < N; n++) ag[n] = au[n] = 0.f;
#pragma unroll
    for (int it = 0; it < 3; it++) {
        const int b = it * 32 + lane;
        if (b < 90) {
            const uint4* xq = (const uint4*)(xs + b * 32);
            const uint4 xr[4] = {xq[0], xq[1], xq[2], xq[3]};
#pragma unroll
            for (int n = 0; n < N; n++) {
                if (row0 + 2 * n >= NR) continue;
                const uint8_t* cr = cr0 + (size_t)n * 2 * 1488;
                const uint4 cg = *(const uint4*)(cr + b * 16), cu = *(const uint4*)(cr + 1488 + b * 16);
                const float sg = MX_SCALE_AT(cr, 90, b), su = MX_SCALE_AT(cr + 1488, 90, b);
                ag[n] += fp4dot32_r(cg, xr) * sg;
                au[n] += fp4dot32_r(cu, xr) * su;
                asm volatile("" ::: "memory");   // one pair at a time: the next pair's loads stay behind this one's products (registers)
            }
        }
    }
    float gs = 0.f, us = 0.f;
#pragma unroll
    for (int n = 0; n < N; n++) {
        if (row0 + 2 * n >= NR) continue;
        const float g = warp_sum(ag[n]) + bf2f(bias[er0 + 2 * n]), u = warp_sum(au[n]) + bf2f(bias[er0 + 2 * n + 1]);
        if (lane == n) { gs = g; us = u; }
    }
    if (lane < N && row0 + 2 * lane < NR)
        act[((size_t)rr * 4 + k) * (NR / 2) + row0 / 2 + lane] = __float2bfloat16_rn(swiglu(gs, us) * rw[slot]);
}
// down, KB <= 32 (one code block per lane and expert); grid (step rows, 2880 / (N x warps per block))
template <int N, int PUSH = 0>
__global__ void __launch_bounds__(256) k_dn_x(const uint8_t* cod, int KB, const bf16* act,
                                              const int* ids, const float* rw, const bf16* bias, float* part, const __grid_constant__ ArPush ps = ArPush{}) {
    static_assert(!PUSH || N == 2, "the push takes blocks of 4 warps x 2 rows (128 threads)");
    static_assert(2880 % N == 0 && N <= 32, "N rows per warp: a divisor of 2880, at most one per lane (review r8 p2 L1)");
    const int rr = blockIdx.x;                 // row of the step
    const bf16* as = act + (size_t)rr * 4 * KB * 32;
    ids += rr * 4;
    rw += rr * 4;
    part += (size_t)rr * 2880;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int row0 = (blockIdx.y * (blockDim.x >> 5) + warp) * N;
    if (row0 >= 2880) return;
    const size_t RS = mx_rs(KB);
    const int i0 = ids[0], i1 = ids[1], i2 = ids[2], i3 = ids[3];
    const int idk[4] = {i0, i1, i2, i3};
    float acc[N];
#pragma unroll
    for (int n = 0; n < N; n++) acc[n] = 0.f;
#pragma unroll
    for (int k = 0; k < 4; k++) {
        if (lane < KB) {
            const uint4* xq = (const uint4*)(as + (k * KB + lane) * 32);
            const uint4 xr[4] = {xq[0], xq[1], xq[2], xq[3]};
            const uint8_t* rec0 = cod + ((size_t)idk[k] * 2880 + row0) * RS;
#pragma unroll
            for (int n = 0; n < N; n++) {
                if (row0 + n >= 2880) continue;   // (the launch must cover the rows once: grid.y x warps x N = 2880)
                const uint8_t* rec = rec0 + (size_t)n * RS;
                acc[n] += fp4dot32_r(*(const uint4*)(rec + lane * 16), xr) * MX_SCALE_AT(rec, KB, lane);
                asm volatile("" ::: "memory");   // one row at a time (registers)
            }
        }
    }
    float av[N];
#pragma unroll
    for (int n = 0; n < N; n++) {
        av[n] = 0.f;
        if (row0 + n >= 2880) continue;
        float a = warp_sum(acc[n]);
        if (lane == 0) {
            if (bias)
                for (int k = 0; k < 4; k++) a += rw[k] * bf2f(bias[(size_t)idk[k] * 2880 + row0 + n]);
            part[row0 + n] = a;
            av[n] = a;
        }
    }
    if (PUSH) {   // the block's 8 sums (4 warps x 2 rows; 2880 % 8 == 0) as two words to every peer's words of this row
        __shared__ __align__(16) float pw[8];
        if (lane == 0) pw[2 * warp] = av[0], pw[2 * warp + 1] = av[N - 1];
        __syncthreads();
#ifndef GPTOSS_AR_NOSTORE
        if (threadIdx.x < 2 && blockDim.x == 128) {   // (the launch is 4 warps; any other block size pushes nothing and k_ar_t traps on the missing words)
            const float4 v = *(const float4*)(pw + 4 * threadIdx.x);
            for (int i = 0; i < ps.n; i++)
                asm volatile("st.volatile.global.v4.f32 [%0], {%1, %2, %3, %4};" ::"l"(ps.d[i] + (size_t)rr * (4 * ARW) + blockIdx.y * 8 + 4 * threadIdx.x),
                             "f"(v.x), "f"(v.y), "f"(v.z), "f"(v.w) : "memory");
        }
#endif
    }
}

// ---------------------------------------------------------------- expert union over the rows of a step (R > 1)
// Rows of one step often route to the same expert (M1: 11.8 experts for 4 consecutive rows instead of 16). The jobs
// are the distinct experts of the step; a job's users are the (row, k) slots that picked it. The union kernels read
// an expert's bytes from DRAM once and serve every user from L1 with the arithmetic of the single-row kernels, so a
// row's result is the same as alone.
struct Jobs {
    int* nu;      // [1] distinct experts
    int* ue;      // [4 MAXR] expert of job j
    int* ucnt;    // [4 MAXR] users of job j
    int* uslot;   // [4 MAXR][MAXR] their slots (row * 4 + k), ascending
};
__global__ void __launch_bounds__(64) k_jobs(const int* ids, int R, const __grid_constant__ Jobs jb) {
    __shared__ int se[64], sfirst[64];
    const int s = threadIdx.x, n = 4 * R;
    int e = -1, f = s;
    if (s < n) {
        e = ids[s];
        for (int t = s - 1; t >= 0; t--)
            if (ids[t] == e) f = t;      // the first slot with this expert
    }
    se[s] = e;
    sfirst[s] = s < n ? f : -1;
    __syncthreads();
    if (s < n) {
        int j = 0;
        for (int t = 0; t < f; t++) j += sfirst[t] == t;   // rank of the first slot among the first slots
        int before = 0, total = 0;
        for (int t = 0; t < n; t++)
            if (se[t] == e) {
                before += t < s;
                total++;
            }
        jb.uslot[j * MAXR + before] = s;
        if (f == s) {
            jb.ue[j] = e;
            jb.ucnt[j] = total;
        }
    }
    __syncthreads();
    if (s == 0) {
        int nu = 0;
        for (int t = 0; t < n; t++) nu += sfirst[t] == t;
        *jb.nu = nu;
    }
}
// k_gu for a job: blockIdx.y = job; the two pairs of the warp are computed for every user of the job
__global__ void __launch_bounds__(256) k_gu_u(const uint8_t* cod, const bf16* bias, int NR, const bf16* x,
                                              const __grid_constant__ Jobs jb, const float* rw, bf16* act) {
    const int j = blockIdx.y;
    if (j >= *jb.nu) return;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int e = jb.ue[j], nus = jb.ucnt[j];
    const int row0 = (blockIdx.x * 8 + warp) * 4;
    for (int u = 0; u < nus; u++) {
        const int slot = jb.uslot[j * MAXR + u];
        const bf16* xs = x + (size_t)(slot >> 2) * 2880;
        float g[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (int i = 0; i < 4; i += 2) {
            const int row = row0 + i;
            if (row >= NR) continue;
            const size_t er = (size_t)e * NR + row;
            const uint8_t* cr = cod + er * 1488;
            float ag = 0.f, au = 0.f;
#pragma unroll
            for (int it = 0; it < 3; it++) {
                const int b = it * 32 + lane;
                if (b < 90) {
                    const uint4 cg = *(const uint4*)(cr + b * 16), cu = *(const uint4*)(cr + 1488 + b * 16);
                    const float sg = MX_SCALE_AT(cr, 90, b), su = MX_SCALE_AT(cr + 1488, 90, b);
                    ag += fp4dot32(cg, xs + b * 32) * sg;
                    au += fp4dot32(cu, xs + b * 32) * su;
                }
            }
            g[i] = warp_sum(ag) + bf2f(bias[er]);
            g[i + 1] = warp_sum(au) + bf2f(bias[er + 1]);
        }
        if (lane < 2 && row0 + 2 * lane < NR)
            act[(size_t)slot * (NR / 2) + row0 / 2 + lane] = __float2bfloat16_rn(swiglu(g[2 * lane], g[2 * lane + 1]) * rw[slot]);
    }
}
// k_dn over the jobs: an output row reads each distinct expert's blocks once; the per-slot lane partials are then
// added per row in k order, as the single-row kernel adds them (KB <= 32: one block per lane and expert).
template <int RM>
__global__ void __launch_bounds__(256) k_dn_u(const uint8_t* cod, int KB, const bf16* act, const int* ids, const float* rw,
                                              const bf16* bias, float* part, const __grid_constant__ Jobs jb, int R) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int row = blockIdx.x * 8 + warp;
    if (row >= 2880) return;
    const size_t RS = mx_rs(KB);
    float p[4 * RM];
#pragma unroll
    for (int i = 0; i < 4 * RM; i++) p[i] = 0.f;
    const int nu = *jb.nu;
    for (int j = 0; j < nu; j++) {
        const size_t er = (size_t)jb.ue[j] * 2880 + row;
        const int nus = jb.ucnt[j];
        for (int b = lane; b < KB; b += 32) {
            const uint4 c = *(const uint4*)(cod + er * RS + b * 16);
            const float sc = MX_SCALE_AT(cod + er * RS, KB, b);
            for (int u = 0; u < nus; u++) {
                const int slot = jb.uslot[j * MAXR + u];
                p[slot] += fp4dot32(c, act + ((size_t)slot * KB + b) * 32) * sc;
            }
        }
    }
    for (int rr = 0; rr < R; rr++) {
        float acc = 0.f;
#pragma unroll
        for (int k = 0; k < 4; k++) acc += p[rr * 4 + k];
        acc = warp_sum(acc);
        if (lane == 0) {
            if (bias)
                for (int k = 0; k < 4; k++) acc += rw[rr * 4 + k] * bf2f(bias[(size_t)ids[rr * 4 + k] * 2880 + row]);
            part[(size_t)rr * 2880 + row] = acc;
        }
    }
}

// ---------------------------------------------------------------- gate_up on tensor cores, rows as MMA columns (plan D7)
// A warp owns 16 weight rows (8 gate / up pairs) of one expert and walks their 90 MXFP4 blocks; the fragments come
// from a 256-entry byte -> bf16-pair table (unscaled codes), the block scale multiplies the block's partial sums, the
// input rows are the 8 MMA columns: every input row that uses this expert rides on the same weight read, and a
// column's result does not depend on the other columns. x[c] = input of column c (stride ldx), `cols` <= 8 columns;
// act[c] (stride lda) = bf16(swiglu(gate, up) * rw[c]) for this card's NR / 2 units.
// The next line of every weight row is prefetched one line ahead: a warp alone keeps only 16 short streams in flight.
__global__ void __launch_bounds__(256) k_gu_mma(const uint8_t* cod, const bf16* bias, int NR, int e, const bf16* x, int ldx,
                                                int cols, const float* rw, bf16* act, int lda) {
    __shared__ uint32_t tab[256];
    {
        const float2 v = fp4pair((threadIdx.x & 15u) | ((threadIdx.x >> 4) << 16));
        const bf162 r = __floats2bfloat162_rn(v.x * 16384.f, v.y * 16384.f);
        tab[threadIdx.x] = *(const uint32_t*)&r;
    }
    __syncthreads();
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const int row0 = (blockIdx.x * 8 + warp) * 16;
    if (row0 >= NR) return;
    const uint8_t* rA = cod + ((size_t)e * NR + row0 + g) * 1488;
    const uint8_t* rB = rA + 8 * 1488;
    const uint32_t baseA = rA[1440 + 45], baseB = rB[1440 + 45];
    const uint32_t* xp = (const uint32_t*)(x + (size_t)min(g, cols - 1) * ldx) + t;
    const bool xon = g < cols;
    float c0 = 0.f, c1 = 0.f, c2 = 0.f, c3 = 0.f;
#pragma unroll 2
    for (int b = 0; b < 90; b++) {
        if ((b & 7) == 0 && t == 0) {   // one line (8 blocks) ahead, once per row
            asm volatile("prefetch.global.L1 [%0];" ::"l"(rA + b * 16 + 256));
            asm volatile("prefetch.global.L1 [%0];" ::"l"(rB + b * 16 + 256));
        }
        const uint4 wa = *(const uint4*)(rA + b * 16), wb = *(const uint4*)(rB + b * 16);
        const float sa = __int_as_float((int)(baseA + ((rA[1440 + (b >> 1)] >> ((b & 1) * 4)) & 15u)) << 23);
        const float sb = __int_as_float((int)(baseB + ((rB[1440 + (b >> 1)] >> ((b & 1) * 4)) & 15u)) << 23);
        float p0 = 0.f, p1 = 0.f, p2 = 0.f, p3 = 0.f;
#pragma unroll
        for (int half = 0; half < 2; half++) {
            const uint32_t alo = half ? wa.z : wa.x, ahi = half ? wa.w : wa.y;
            const uint32_t blo = half ? wb.z : wb.x, bhi = half ? wb.w : wb.y;
            const uint32_t a0 = tab[(alo >> (8 * t)) & 255u], a1 = tab[(blo >> (8 * t)) & 255u];
            const uint32_t a2 = tab[(ahi >> (8 * t)) & 255u], a3 = tab[(bhi >> (8 * t)) & 255u];
            const int k0 = b * 16 + half * 8;   // word index: 2 inputs per word
            uint32_t b0 = xp[k0], b1 = xp[k0 + 4];
            if (!xon) b0 = b1 = 0u;
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                         : "+f"(p0), "+f"(p1), "+f"(p2), "+f"(p3)
                         : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
        }
        c0 += p0 * sa;
        c1 += p1 * sa;
        c2 += p2 * sb;
        c3 += p3 * sb;
    }
    // lane (g, t): weight rows row0 + g (c0, c1) and row0 + g + 8 (c2, c3), columns 2t and 2t + 1
    const float bA = bf2f(bias[(size_t)e * NR + row0 + g]), bB = bf2f(bias[(size_t)e * NR + row0 + g + 8]);
    c0 += bA;
    c1 += bA;
    c2 += bB;
    c3 += bB;
    // gate = even weight row (even g), up = the next row: the lane 4 further holds it
    const float u0 = __shfl_xor_sync(0xffffffffu, c0, 4), u1 = __shfl_xor_sync(0xffffffffu, c1, 4);
    const float u2 = __shfl_xor_sync(0xffffffffu, c2, 4), u3 = __shfl_xor_sync(0xffffffffu, c3, 4);
    if (g & 1) return;
    const int unitA = (row0 + g) / 2, unitB = unitA + 4;
    if (2 * t < cols) {
        act[(size_t)(2 * t) * lda + unitA] = __float2bfloat16_rn(swiglu(c0, u0) * rw[2 * t]);
        act[(size_t)(2 * t) * lda + unitB] = __float2bfloat16_rn(swiglu(c2, u2) * rw[2 * t]);
    }
    if (2 * t + 1 < cols) {
        act[(size_t)(2 * t + 1) * lda + unitA] = __float2bfloat16_rn(swiglu(c1, u1) * rw[2 * t + 1]);
        act[(size_t)(2 * t + 1) * lda + unitB] = __float2bfloat16_rn(swiglu(c3, u3) * rw[2 * t + 1]);
    }
}

// The same with the weights staged: a block owns 128 weight rows of one routed expert (blockIdx.y = k) and brings
// their codes into shared memory in three thirds of 30 blocks (61 KB; every thread issues its 15 loads back to back,
// so the whole third is in flight at once), then the 8 warps run the MMAs from shared memory. Needs 68.6 KB of
// dynamic shared memory (GU_MMA_SMEM): one block per SM, 60 blocks for the 4 experts of card A = one wave.
constexpr int GU_MMA_SMEM = 1024 + 128 * 48 + 128 * 480;
__global__ void __launch_bounds__(256) k_gu_mma2(const uint8_t* cod, const bf16* bias, int NR, const int* ids, const bf16* x,
                                                 int ldx, int cols, const float* rw, bf16* act, int lda) {
    extern __shared__ __align__(16) unsigned char sm[];
    uint32_t* tab = (uint32_t*)sm;
    uint8_t* tails = sm + 1024;              // [128 rows][48]: 45 nibble bytes, the base byte, pad
    uint4* codes = (uint4*)(sm + 1024 + 128 * 48);   // [128 rows][30 blocks]
    const int e = ids[blockIdx.y];
    const int rowb = blockIdx.x * 128;
    {
        const float2 v = fp4pair((threadIdx.x & 15u) | ((threadIdx.x >> 4) << 16));
        const bf162 r = __floats2bfloat162_rn(v.x * 16384.f, v.y * 16384.f);
        tab[threadIdx.x] = *(const uint32_t*)&r;
    }
    const uint8_t* rec0 = cod + ((size_t)e * NR + rowb) * 1488;
    for (int q = threadIdx.x; q < 128 * 3; q += 256) {
        const int r = q / 3, i = q % 3;
        ((uint4*)tails)[q] = rowb + r < NR ? *(const uint4*)(rec0 + (size_t)r * 1488 + 1440 + i * 16) : make_uint4(0, 0, 0, 0);
    }
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const int lr = warp * 16 + g;            // row within the block (and lr + 8)
    const bool live = rowb + warp * 16 < NR;
    const uint32_t* xp = (const uint32_t*)(x + (size_t)min(g, cols - 1) * ldx) + t;
    const bool xon = g < cols;
    float c0 = 0.f, c1 = 0.f, c2 = 0.f, c3 = 0.f;
    for (int ph = 0; ph < 3; ph++) {
        __syncthreads();                     // table / tails staged, or the previous third is used up
        for (int q = threadIdx.x; q < 128 * 30; q += 256) {
            const int r = q / 30, i = q % 30;
            codes[q] = rowb + r < NR ? *(const uint4*)(rec0 + (size_t)r * 1488 + (ph * 30 + i) * 16) : make_uint4(0, 0, 0, 0);
        }
        __syncthreads();
        if (!live) continue;
        const uint8_t *tA = tails + lr * 48, *tB = tails + (lr + 8) * 48;
        const uint32_t baseA = tA[45], baseB = tB[45];
#pragma unroll 2
        for (int i = 0; i < 30; i++) {
            const int b = ph * 30 + i;
            const uint4 wa = codes[lr * 30 + i], wb = codes[(lr + 8) * 30 + i];
            const float sa = __int_as_float((int)(baseA + ((tA[b >> 1] >> ((b & 1) * 4)) & 15u)) << 23);
            const float sb = __int_as_float((int)(baseB + ((tB[b >> 1] >> ((b & 1) * 4)) & 15u)) << 23);
            float p0 = 0.f, p1 = 0.f, p2 = 0.f, p3 = 0.f;
#pragma unroll
            for (int half = 0; half < 2; half++) {
                const uint32_t alo = half ? wa.z : wa.x, ahi = half ? wa.w : wa.y;
                const uint32_t blo = half ? wb.z : wb.x, bhi = half ? wb.w : wb.y;
                const uint32_t a0 = tab[(alo >> (8 * t)) & 255u], a1 = tab[(blo >> (8 * t)) & 255u];
                const uint32_t a2 = tab[(ahi >> (8 * t)) & 255u], a3 = tab[(bhi >> (8 * t)) & 255u];
                const int k0 = b * 16 + half * 8;
                uint32_t b0 = xp[k0], b1 = xp[k0 + 4];
                if (!xon) b0 = b1 = 0u;
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                             : "+f"(p0), "+f"(p1), "+f"(p2), "+f"(p3)
                             : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
            }
            c0 += p0 * sa;
            c1 += p1 * sa;
            c2 += p2 * sb;
            c3 += p3 * sb;
        }
    }
    if (!live) return;
    const int row0 = rowb + warp * 16;
    const float bA = bf2f(bias[(size_t)e * NR + row0 + g]), bB = bf2f(bias[(size_t)e * NR + row0 + g + 8]);
    c0 += bA;
    c1 += bA;
    c2 += bB;
    c3 += bB;
    const float u0 = __shfl_xor_sync(0xffffffffu, c0, 4), u1 = __shfl_xor_sync(0xffffffffu, c1, 4);
    const float u2 = __shfl_xor_sync(0xffffffffu, c2, 4), u3 = __shfl_xor_sync(0xffffffffu, c3, 4);
    if (g & 1) return;
    const int unitA = (row0 + g) / 2, unitB = unitA + 4;
    bf16* ak = act + (size_t)blockIdx.y * cols * lda;   // [k][column][units]
    const float* rk = rw + blockIdx.y * cols;
    if (2 * t < cols) {
        ak[(size_t)(2 * t) * lda + unitA] = __float2bfloat16_rn(swiglu(c0, u0) * rk[2 * t]);
        ak[(size_t)(2 * t) * lda + unitB] = __float2bfloat16_rn(swiglu(c2, u2) * rk[2 * t]);
    }
    if (2 * t + 1 < cols) {
        ak[(size_t)(2 * t + 1) * lda + unitA] = __float2bfloat16_rn(swiglu(c1, u1) * rk[2 * t + 1]);
        ak[(size_t)(2 * t + 1) * lda + unitB] = __float2bfloat16_rn(swiglu(c3, u3) * rk[2 * t + 1]);
    }
}

// argmax of n logits (ties: the lowest index; NaN never wins, all-NaN gives index 0); res = {value, index + v0 as int bits}
__global__ void __launch_bounds__(1024) k_argmax(const float* lg, int n, int v0, float* res) {
    __shared__ float bv[32];
    __shared__ int bi[32];
    lg += (size_t)blockIdx.x * n;   // row of the step
    res += 2 * blockIdx.x;
    float val = NEG_INF;
    int idx = threadIdx.x < n ? threadIdx.x : 0x7fffffff;
    for (int i = threadIdx.x; i < n; i += 1024) {
        const float x = lg[i];
        if (x > val) {
            val = x;
            idx = i;
        }
    }
    for (int o = 16; o; o >>= 1) {
        const float ov = __shfl_xor_sync(0xffffffffu, val, o);
        const int oi = __shfl_xor_sync(0xffffffffu, idx, o);
        if (ov > val || (ov == val && oi < idx)) {
            val = ov;
            idx = oi;
        }
    }
    if ((threadIdx.x & 31) == 0) {
        bv[threadIdx.x >> 5] = val;
        bi[threadIdx.x >> 5] = idx;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        for (int k = 1; k < 32; k++)
            if (bv[k] > val || (bv[k] == val && bi[k] < idx)) {
                val = bv[k];
                idx = bi[k];
            }
        res[0] = val;
        res[1] = __int_as_float(idx + v0);
    }
}

// ---------------------------------------------------------------- 12-bit dense (plan D6), lossless
// byte plane: sign | exponent offset below the row's largest exponent (15 = patched) | top 3 mantissa bits;
// nibble plane: low 4 mantissa bits, two weights per byte. A weight is rebuilt as fp32 bits with exponent 127 - off
// and the row sum is scaled by 2^(base - 127) (exact); patched weights (zero, or 15+ exponents below the row's
// largest: 0.6-1.0 per row) count as 0 in the planes and are added from a per-row list by lane 0.
struct D12 {
    const uint8_t *bp, *np;      // [N][K], [N][K / 2]
    const float* sc;             // [N] 2^(base - 127)
    const int* pptr;             // [N + 1]
    const uint16_t *pcol, *pval; // patches: column, bf16 bits
};
__device__ __forceinline__ float d12_16(uint4 c, uint2 n, uint4 xa, uint4 xb) {
    const uint32_t cw[4] = {c.x, c.y, c.z, c.w}, nw[2] = {n.x, n.y};
    const uint32_t xw[8] = {xa.x, xa.y, xa.z, xa.w, xb.x, xb.y, xb.z, xb.w};
    float d = 0.f;
#pragma unroll
    for (int j = 0; j < 16; j++) {
        const uint32_t b = (cw[j >> 2] >> (8 * (j & 3))) & 255u;
        const uint32_t eo = (b >> 3) & 15u;
        const uint32_t lo = (nw[j >> 3] >> (4 * (j & 7))) & 15u;
        const uint32_t bits = ((b & 128u) << 24) | ((127u - eo) << 23) | ((b & 7u) << 20) | (lo << 16);
        const float wv = eo == 15u ? 0.f : __uint_as_float(bits);
        d += wv * __uint_as_float((j & 1) ? (xw[j >> 1] & 0xffff0000u) : (xw[j >> 1] << 16));
    }
    return d;
}
__device__ __forceinline__ float d12_tail(const D12& w, int row, const bf16* xs, float acc) {   // lane 0
    acc *= w.sc[row];
    for (int p = w.pptr[row]; p < w.pptr[row + 1]; p++)
        acc += __uint_as_float((uint32_t)w.pval[p] << 16) * bf2f(xs[w.pcol[p]]);
    return acc;
}
// two rows at once (as row_dot2)
__device__ __forceinline__ void row_dot12_2(const D12& w, int ra, int rb, const bf16* xs, int K, int lane, float& oa, float& ob) {
    const uint8_t *ba = w.bp + (size_t)ra * K, *bb = w.bp + (size_t)rb * K;
    const uint8_t *na = w.np + (size_t)ra * (K / 2), *nb = w.np + (size_t)rb * (K / 2);
    float a = 0.f, b = 0.f;
    for (int e0 = lane * 16; e0 < K; e0 += 512) {
        const uint4 ca = *(const uint4*)(ba + e0), cb = *(const uint4*)(bb + e0);
        const uint2 qa = *(const uint2*)(na + e0 / 2), qb = *(const uint2*)(nb + e0 / 2);
        const uint4 xa = *(const uint4*)(xs + e0), xb = *(const uint4*)(xs + e0 + 8);
        a += d12_16(ca, qa, xa, xb);
        b += d12_16(cb, qb, xa, xb);
    }
    oa = warp_sum(a);
    ob = warp_sum(b);
}
// k_gemv on the 12-bit layout: 2 rows per warp at once, 16 rows per block
__global__ void __launch_bounds__(256) k_gemv12(const __grid_constant__ D12 w, int N, int K, const bf16* x, const bf16* bias, float* out) {
    const int r0 = (blockIdx.x * 8 + (threadIdx.x >> 5)) * 2, lane = threadIdx.x & 31;
    if (r0 >= N) return;
    const int r1 = min(r0 + 1, N - 1);
    float a, b;
    row_dot12_2(w, r0, r1, x, K, lane, a, b);
    if (lane) return;
    out[r0] = d12_tail(w, r0, x, a) + (bias ? bf2f(bias[r0]) : 0.f);
    if (r0 + 1 < N) out[r0 + 1] = d12_tail(w, r0 + 1, x, b) + (bias ? bf2f(bias[r0 + 1]) : 0.f);
}
// k_qkv on the 12-bit layout (same pairing, RoPE and stores)
__global__ void __launch_bounds__(256) k_qkv12(const __grid_constant__ D12 w, const bf16* bias, const bf16* x, int nq, int kvn,
                                               const int* posp, const float* invf, float att, float* q, bf16* Kd,
                                               bf16* Vd, int ring) {
    const int pair = blockIdx.x * 8 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    const int head = pair >> 5, d = pair & 31;
    if (head >= nq + 2 * kvn) return;
    const int r0 = head * 64 + d, r1 = r0 + 32;
    float a, b;
    row_dot12_2(w, r0, r1, x, 2880, lane, a, b);
    if (lane) return;
    a = d12_tail(w, r0, x, a) + bf2f(bias[r0]);
    b = d12_tail(w, r1, x, b) + bf2f(bias[r1]);
    const int pos = *posp;
    if (head < nq + kvn) {
        const float fr = (float)pos * invf[d];
        const float c = cosf(fr) * att, s = sinf(fr) * att;
        const float x1 = a * c - b * s, x2 = b * c + a * s;
        a = x1;
        b = x2;
    }
    if (head < nq) {
        q[r0] = a;
        q[r1] = b;
        return;
    }
    const size_t row = ring == 2 ? (size_t)0 : ring ? (size_t)(pos & 255) : (size_t)pos;
    const bool isk = head < nq + kvn;
    bf16* dst = (isk ? Kd : Vd) + (row * kvn + (head - nq - (isk ? 0 : kvn))) * 64;
    dst[d] = __float2bfloat16_rn(a);
    dst[d + 32] = __float2bfloat16_rn(b);
}

// ---------------------------------------------------------------- certified two-stage lm_head (plan D5)
// Screen plane: one byte per weight = sign | 15 - (exponent offset below the row's largest exponent) | top 3 mantissa
// bits (field 0 = offset 15: zero or below the range). With the dropped 4 mantissa bits at their midpoint the screen weight is
// within 2^(e - 4) of the bf16 weight (offset 15: within 2^(base - 14)), so for a head input x
//     s_r = sum w~ x,   B_r = sum err |x|,   exact logit in [s_r - B_r, s_r + B_r].
// Stage 2 computes the exact logit (the same row_dot as the full head: bitwise the same value) only for rows whose
// upper bound reaches the largest lower bound of all cards. The argmax is the full head's argmax.
struct ThrBuf {
    uint2* wr[4];   // every rank's threshold words [ngpu] (value, tag)
    uint2* rd;
    int nw;
};
// lo / hi of 4 rows per warp; lomax[block] = max lo of the block. base[2r] = 2^(largest exponent of row r),
// base[2r + 1] = (number of offset-15 weights of the row) * 2^(largest exponent - 14) = the most they can add per
// unit of |x|. The per-weight error 2^(e - 4) is at most |w~| / 17 (w~ >= 17/16 * 2^e), so the bound is accumulated
// as |w~ x| / 17: one fabs + add per weight instead of a second product.
__global__ void __launch_bounds__(256) k_lm_screen(const uint8_t* S8, const float* base, const bf16* x, const float* xmaxp,
                                                   int N, float* lo, float* hi, float* lomax, int R) {
    __shared__ float bm[8];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    float wmax[MAXR];
#pragma unroll
    for (int rr = 0; rr < MAXR; rr++) wmax[rr] = NEG_INF;
#pragma unroll
    for (int i = 0; i < 4; i++) {
        const int row = (blockIdx.x * 8 + warp) * 4 + i;
        if (row >= N) break;
        const uint8_t* sr = S8 + (size_t)row * 2880;
        for (int rr = 0; rr < R; rr++) {   // the head row is walked once per row of the step (from L1 after the first)
            const bf16* xr = x + (size_t)rr * 2880;
            float acc = 0.f, bnd = 0.f;
            for (int e0 = lane * 16; e0 < 2880; e0 += 512) {
                const uint4 c = *(const uint4*)(sr + e0);
                const uint4 xa = *(const uint4*)(xr + e0), xb = *(const uint4*)(xr + e0 + 8);
                const uint32_t cw[4] = {c.x, c.y, c.z, c.w};
                const uint32_t xw[8] = {xa.x, xa.y, xa.z, xa.w, xb.x, xb.y, xb.z, xb.w};
#pragma unroll
                for (int j = 0; j < 16; j++) {
                    const uint32_t b = (cw[j >> 2] >> (8 * (j & 3))) & 255u;
                    const uint32_t xbits = (j & 1) ? (xw[j >> 1] & 0xffff0000u) : (xw[j >> 1] << 16);
                    // byte = sign | (15 - offset) << 3 | m3: bits 6..0 shifted to the fp32 exponent / mantissa, + 112 << 23
                    const uint32_t mb = (((b & 127u) << 20) + 0x38080000u) | ((b & 128u) << 24);
                    const float pr = ((b & 0x78u) ? __uint_as_float(mb) : 0.f) * __uint_as_float(xbits);
                    acc += pr;
                    bnd += fabsf(pr);
                }
            }
            acc = warp_sum(acc);
            bnd = warp_sum(bnd);
            // 1 % on the bound covers the fp32 rounding of both sums (<= 2880 * 2^-24 * sum |w x|, far below B / 100)
            const float sv = acc * base[2 * row];
            const float B = (bnd * base[2 * row] * (1.f / 17.f) + base[2 * row + 1] * xmaxp[rr]) * 1.01f;
            if (lane == 0) {
                lo[(size_t)rr * N + row] = sv - B;
                hi[(size_t)rr * N + row] = sv + B;
            }
            wmax[rr] = fmaxf(wmax[rr], sv - B);
        }
    }
    for (int rr = 0; rr < R; rr++) {
        if (lane == 0) bm[warp] = wmax[rr];
        __syncthreads();
        if (threadIdx.x == 0) {
            float m = bm[0];
            for (int k = 1; k < 8; k++) m = fmaxf(m, bm[k]);
            lomax[(size_t)rr * gridDim.x + blockIdx.x] = m;
        }
        __syncthreads();
    }
}
// The single-row form (R = 1 steps): same bounds, tighter code (the row loop above costs 60 us at R = 1). The two
// forms may differ in the last bit of a bound; the step's result does not (it is the exact rescoring).
__global__ void __launch_bounds__(256) k_lm_screen1(const uint8_t* S8, const float* base, const bf16* x, const float* xmaxp,
                                                   int N, float* lo, float* hi, float* lomax) {
    __shared__ float bm[8];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const float xmax = *xmaxp;
    float wmax = NEG_INF;
#pragma unroll
    for (int i = 0; i < 4; i++) {
        const int row = (blockIdx.x * 8 + warp) * 4 + i;
        if (row >= N) break;
        const uint8_t* sr = S8 + (size_t)row * 2880;
        float acc = 0.f, bnd = 0.f;
        for (int e0 = lane * 16; e0 < 2880; e0 += 512) {
            const uint4 c = *(const uint4*)(sr + e0);
            const uint4 xa = *(const uint4*)(x + e0), xb = *(const uint4*)(x + e0 + 8);
            const uint32_t cw[4] = {c.x, c.y, c.z, c.w};
            const uint32_t xw[8] = {xa.x, xa.y, xa.z, xa.w, xb.x, xb.y, xb.z, xb.w};
#pragma unroll
            for (int j = 0; j < 16; j++) {
                const uint32_t b = (cw[j >> 2] >> (8 * (j & 3))) & 255u;
                const uint32_t xbits = (j & 1) ? (xw[j >> 1] & 0xffff0000u) : (xw[j >> 1] << 16);
                // byte = sign | (15 - offset) << 3 | m3: bits 6..0 shifted to the fp32 exponent / mantissa, + 112 << 23
                const uint32_t mb = (((b & 127u) << 20) + 0x38080000u) | ((b & 128u) << 24);
                const float pr = ((b & 0x78u) ? __uint_as_float(mb) : 0.f) * __uint_as_float(xbits);
                acc += pr;
                bnd += fabsf(pr);
            }
        }
        acc = warp_sum(acc);
        bnd = warp_sum(bnd);
        // 1 % on the bound covers the fp32 rounding of both sums (<= 2880 * 2^-24 * sum |w x|, far below B / 100)
        const float s = acc * base[2 * row];
        const float B = (bnd * base[2 * row] * (1.f / 17.f) + base[2 * row + 1] * xmax) * 1.01f;
        if (lane == 0) {
            lo[row] = s - B;
            hi[row] = s + B;
        }
        wmax = fmaxf(wmax, s - B);
    }
    if (lane == 0) bm[warp] = wmax;
    __syncthreads();
    if (threadIdx.x == 0) {
        float m = bm[0];
        for (int k = 1; k < 8; k++) m = fmaxf(m, bm[k]);
        lomax[blockIdx.x] = m;
    }
}
// The screen for all rows of a step in one pass (round 7): the R head inputs are the columns of a tensor-core
// product, so the screen plane is read once per step whatever R is. A warp owns 16 head rows; lane (g, t) takes 16
// screen bytes of rows g and g + 8 per 64-byte group and builds their bf16 fragments in registers (the byte's seven
// low bits shifted into exponent / mantissa, the midpoint bit and the exponent offset added, the sign moved to bit
// 15, field 0 -> zero); a second product on the absolute values gives the bound. k index = a fixed permutation
// inside the 16 elements a quad of lanes holds, as in dense_mma.
// The values are the screen sums of k_lm_screen with a different summation order. The bound's 1 % leaves room for
// a relative error of the two sums of ~5e-4 of sum |w~ x| (0.01 / 17 = 5.9e-4). Assumed tensor-core arithmetic:
// exact products, terms aligned to the largest exponent in a >= 24-bit window and truncated: <= ~20 units of 2^-23
// per instruction, 3.9-4.3e-4 over the 180 instructions of a row, so the worst case leaves 25-40 % of the margin
// (review r7 L6); a narrower window on other hardware would need a larger factor. gptoss_kbench d7screen measures
// the slack against sums in double (0.4 % of the margin used on sm_86). The step's result is the exact
// rescoring either way. lomax[rr * blocks + block], blocks = ceil(N / 128).
__device__ __forceinline__ uint32_t s8pair(uint32_t p) {   // two screen bytes (bits 0-7, 16-23) -> two bf16
    const uint32_t nz = ((p & 0x00780078u) + 0x00780078u) & 0x00800080u;
    return ((((p & 0x007f007fu) << 4) + 0x38083808u) | ((p & 0x00800080u) << 8)) & ((nz >> 7) * 0xffffu);
}
template <int NC>
__global__ void __launch_bounds__(256) k_lm_screen_m(const uint8_t* S8, const float* base, const bf16* x, const float* xmaxp,
                                                     int N, float* lo, float* hi, float* lomax, int R) {
    __shared__ float bm[8 * 8 * NC];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const int row0 = (blockIdx.x * 8 + warp) * 16;
    const int ra = row0 + g, rb = ra + 8;
    const bool ina = ra < N, inb = rb < N;
    const uint4* pa = (const uint4*)(S8 + (size_t)(ina ? ra : N - 1) * 2880) + t;
    const uint4* pb = (const uint4*)(S8 + (size_t)(inb ? rb : N - 1) * 2880) + t;
    const float ba0 = base[2 * (ina ? ra : 0)], ba1 = base[2 * (ina ? ra : 0) + 1];
    const float bb0 = base[2 * (inb ? rb : 0)], bb1 = base[2 * (inb ? rb : 0) + 1];
    const uint4* px[NC];
    bool xon[NC];
    float c[4 * NC], a[4 * NC];
#pragma unroll
    for (int n = 0; n < NC; n++) {
        xon[n] = 8 * n + g < R;
        px[n] = (const uint4*)(x + (size_t)(xon[n] ? 8 * n + g : 0) * 2880) + 2 * t;
    }
#pragma unroll
    for (int i = 0; i < 4 * NC; i++) c[i] = a[i] = 0.f;
    // One group per round: 2 / 3 groups per round, no bound product and no byte conversion all measure the same
    // 380-400 us (gptoss_kbench, card A: 51 % of the bytes' time; the scalar screen reads 4 full lines of one row per
    // round, this one 64 B of each of 16 rows).
    constexpr int SC_UG = 1;
    for (int j0 = 0; j0 < 45; j0 += SC_UG) {
        uint4 wa[SC_UG], wb[SC_UG];
#pragma unroll
        for (int u = 0; u < SC_UG; u++) {
            const int j = min(j0 + u, 44);
            wa[u] = pa[4 * j];
            wb[u] = pb[4 * j];
        }
#pragma unroll
        for (int u = 0; u < SC_UG; u++) {
            const int j = j0 + u;
            if (j >= 45) break;
            const uint32_t ca[4] = {wa[u].x, wa[u].y, wa[u].z, wa[u].w}, cb[4] = {wb[u].x, wb[u].y, wb[u].z, wb[u].w};
            uint32_t fa[8], fb[8];
#pragma unroll
            for (int s4 = 0; s4 < 4; s4++) {
                fa[2 * s4] = s8pair(__byte_perm(ca[s4], 0, 0x4140));
                fa[2 * s4 + 1] = s8pair(__byte_perm(ca[s4], 0, 0x4342));
                fb[2 * s4] = s8pair(__byte_perm(cb[s4], 0, 0x4140));
                fb[2 * s4 + 1] = s8pair(__byte_perm(cb[s4], 0, 0x4342));
            }
#pragma unroll
            for (int n = 0; n < NC; n++) {
                uint4 xa = make_uint4(0, 0, 0, 0), xb = make_uint4(0, 0, 0, 0);
                if (xon[n]) {
                    xa = px[n][8 * j];
                    xb = px[n][8 * j + 1];
                }
                const uint32_t xw[8] = {xa.x, xa.y, xa.z, xa.w, xb.x, xb.y, xb.z, xb.w};
#pragma unroll
                for (int s4 = 0; s4 < 4; s4++) {
                    mma_bf16(c + 4 * n, fa[2 * s4], fb[2 * s4], fa[2 * s4 + 1], fb[2 * s4 + 1], xw[2 * s4], xw[2 * s4 + 1]);
                    mma_bf16(a + 4 * n, fa[2 * s4] & 0x7fff7fffu, fb[2 * s4] & 0x7fff7fffu, fa[2 * s4 + 1] & 0x7fff7fffu,
                             fb[2 * s4 + 1] & 0x7fff7fffu, xw[2 * s4] & 0x7fff7fffu, xw[2 * s4 + 1] & 0x7fff7fffu);
                }
            }
        }
    }
    float wm[2 * NC];
#pragma unroll
    for (int i = 0; i < 2 * NC; i++) {
        const int n = i >> 1, h = i & 1, col = 8 * n + 2 * t + h;
        wm[i] = NEG_INF;
        if (col >= R) continue;
        const float xm = xmaxp[col];
        if (ina) {
            const float sv = c[4 * n + h] * ba0, B = (a[4 * n + h] * ba0 * (1.f / 17.f) + ba1 * xm) * 1.01f;
            lo[(size_t)col * N + ra] = sv - B;
            hi[(size_t)col * N + ra] = sv + B;
            wm[i] = sv - B;
        }
        if (inb) {
            const float sv = c[4 * n + 2 + h] * bb0, B = (a[4 * n + 2 + h] * bb0 * (1.f / 17.f) + bb1 * xm) * 1.01f;
            lo[(size_t)col * N + rb] = sv - B;
            hi[(size_t)col * N + rb] = sv + B;
            wm[i] = fmaxf(wm[i], sv - B);
        }
    }
#pragma unroll
    for (int i = 0; i < 2 * NC; i++) {
        for (int o = 16; o >= 4; o >>= 1) wm[i] = fmaxf(wm[i], __shfl_xor_sync(0xffffffffu, wm[i], o));   // over g
        if (g == 0) bm[(warp * 4 + t) * 2 * NC + i] = wm[i];
    }
    __syncthreads();
    if ((int)threadIdx.x < R) {   // thread = row of the step: column 8 n + 2 t + h
        const int col = threadIdx.x, n = col >> 3, tt = (col & 7) >> 1, h = col & 1;
        float m = NEG_INF;
        for (int w = 0; w < 8; w++) m = fmaxf(m, bm[(w * 4 + tt) * 2 * NC + 2 * n + h]);
        lomax[(size_t)col * gridDim.x + blockIdx.x] = m;
    }
}
// max |x| of the 2880-wide head input
__global__ void __launch_bounds__(960) k_xmax(const bf16* x, float* out) {
    __shared__ float bm[32];
    x += (size_t)blockIdx.x * 2880;   // row of the step
    out += blockIdx.x;
    float m = 0.f;
#pragma unroll
    for (int j = 0; j < 3; j++) m = fmaxf(m, fabsf(bf2f(x[3 * threadIdx.x + j])));
    for (int o = 16; o; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
    if ((threadIdx.x & 31) == 0) bm[threadIdx.x >> 5] = m;
    __syncthreads();
    if (threadIdx.x == 0) {
        for (int k = 1; k < 30; k++) m = fmaxf(m, bm[k]);
        *out = m;
    }
}
// local max of the lower bounds, exchanged with the other cards as tagged words: thr = max over all cards
__global__ void __launch_bounds__(1024) k_lm_thr(const float* lomax, int nb, const __grid_constant__ ThrBuf tb, int rank,
                                                 int ngpu, const unsigned* step, float* thr) {
    __shared__ float bm[32];
    const int rr = blockIdx.x;   // row of the step
    lomax += (size_t)rr * nb;
    thr += rr;
    float m = NEG_INF;
    for (int i = threadIdx.x; i < nb; i += 1024) m = fmaxf(m, lomax[i]);
    for (int o = 16; o; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
    if ((threadIdx.x & 31) == 0) bm[threadIdx.x >> 5] = m;
    __syncthreads();
    if (threadIdx.x >= 4) return;
    m = bm[0];
    for (int k = 1; k < 32; k++) m = fmaxf(m, bm[k]);
    // (review m2 L4: the words hold thresholds only, so the tag is the step itself, odd = never the zeroed start: a
    // stale word matches after 2^31 steps, not after 2^24 = ~14 h of c = 1 decode)
    const unsigned tag = *step * 2u + 1u;
    if (threadIdx.x == 0)
        for (int g = 0; g < tb.nw; g++) st_tag(tb.wr[g] + rank * MAXR + rr, __float_as_uint(m), tag);
    if (threadIdx.x == 0) {
        const unsigned long long t0 = gtimer();
        float t = NEG_INF;
        for (int g = 0; g < ngpu; g++) t = fmaxf(t, __uint_as_float(wait_tag(tb.rd + g * MAXR + rr, tag, t0)));
        *thr = t;
    }
}
// exact logits of the rows with hi >= thr (32 rows per warp), best per warp -> bv / bi (index within the card, -1: none)
__global__ void __launch_bounds__(256) k_lm_rescore(const bf16* W, const bf16* x, const float* hi, const float* thr, int N,
                                                    float* bv, int* bi) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int w = blockIdx.x * 8 + warp;
    const int rr = blockIdx.y;   // row of the step
    hi += (size_t)rr * N;
    x += (size_t)rr * 2880;
    bv += (size_t)rr * gridDim.x * 8;
    bi += (size_t)rr * gridDim.x * 8;
    const float T = thr[rr];
    float best = NEG_INF;
    int bidx = -1;
    const int r0 = w * 32;
    // rows past N never qualify, whatever the threshold (a NaN input makes it NEG_INF: review M1)
    const bool in = r0 + lane < N;
    const float mine = in ? hi[r0 + lane] : NEG_INF;
    unsigned mask = __ballot_sync(0xffffffffu, in && mine >= T);
    while (mask) {
        const int row = r0 + __ffs(mask) - 1;
        mask &= mask - 1;
        const float v = row_dot(W + (size_t)row * 2880, x, 2880, lane);
        if (v > best) {   // rows ascend: the first of equal values (lowest index) stays
            best = v;
            bidx = row;
        }
    }
    if (lane == 0) {
        bv[w] = best;
        bi[w] = bidx;
    }
}
// best of the per-warp candidates (ties: the lowest index); res = {value, index + v0 as int bits}
__global__ void __launch_bounds__(1024) k_argbest(const float* bv, const int* bi, int n, int v0, float* res) {
    __shared__ float sv[32];
    __shared__ int si[32];
    bv += (size_t)blockIdx.x * n;   // row of the step
    bi += (size_t)blockIdx.x * n;
    res += 2 * blockIdx.x;
    float val = NEG_INF;
    int idx = 0x7fffffff;
    for (int i = threadIdx.x; i < n; i += 1024) {
        const float x = bv[i];
        const int j = bi[i];
        if (j >= 0 && (x > val || (x == val && j < idx))) {
            val = x;
            idx = j;
        }
    }
    for (int o = 16; o; o >>= 1) {
        const float ov = __shfl_xor_sync(0xffffffffu, val, o);
        const int oi = __shfl_xor_sync(0xffffffffu, idx, o);
        if (ov > val || (ov == val && oi < idx)) {
            val = ov;
            idx = oi;
        }
    }
    if ((threadIdx.x & 31) == 0) {
        sv[threadIdx.x >> 5] = val;
        si[threadIdx.x >> 5] = idx;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        for (int k = 1; k < 32; k++)
            if (sv[k] > val || (sv[k] == val && si[k] < idx)) {
                val = sv[k];
                idx = si[k];
            }
        res[0] = val;   // a card without candidates reports NEG_INF and index 0
        res[1] = __int_as_float((idx == 0x7fffffff ? 0 : idx) + v0);
    }
}

// ================================================================ prefill (T rows), simple and correct
// rows of the GEMM output Y [T][nrows] (q | k | v): bias, RoPE at pos0 + t, q -> Qf fp32, k / v -> bf16 rows krow0 + t
__global__ void k_qkv_rows(const float* Y, const bf16* bias, int T, int nq, int kvn, int pos0, const float* invf,
                           float att, float* Qf, bf16* Kd, bf16* Vd, int krow0) {
    const int nh = nq + 2 * kvn;
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)T * nh * 32) return;
    const int t = (int)(i / (nh * 32)), pr = (int)(i % (nh * 32));
    const int head = pr >> 5, d = pr & 31;
    const int r0 = head * 64 + d, r1 = r0 + 32;
    const float* y = Y + (size_t)t * nh * 64;
    float a = y[r0] + bf2f(bias[r0]), b = y[r1] + bf2f(bias[r1]);
    if (head < nq + kvn) {
        const float fr = (float)(pos0 + t) * invf[d];
        const float c = cosf(fr) * att, s = sinf(fr) * att;
        const float x1 = a * c - b * s, x2 = b * c + a * s;
        a = x1;
        b = x2;
    }
    if (head < nq) {
        Qf[(size_t)t * nq * 64 + r0] = a;
        Qf[(size_t)t * nq * 64 + r1] = b;
        return;
    }
    const bool isk = head < nq + kvn;
    bf16* dst = (isk ? Kd : Vd) + ((size_t)(krow0 + t) * kvn + (head - nq - (isk ? 0 : kvn))) * 64;
    dst[d] = __float2bfloat16_rn(a);
    dst[d + 32] = __float2bfloat16_rn(b);
}
// copies n K/V rows [kvn][64] between a ring (row = pos & 255) and contiguous rows; dir 0: ring -> rows, 1: rows -> ring
__global__ void k_ring_copy(bf16* ring, bf16* rows, int kvn, int pos_first, int n, int dir) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;   // 16 B units
    const int per = kvn * 8;
    if (i >= (size_t)n * per) return;
    const int r = (int)(i / per), u = (int)(i % per);
    uint4* a = (uint4*)(ring + (size_t)((pos_first + r) & 255) * kvn * 64) + u;
    uint4* b = (uint4*)(rows + (size_t)r * kvn * 64) + u;
    if (dir)
        *a = *b;
    else
        *b = *a;
}
// ---------------------------------------------------------------- 8-bit K/V of the full-attention layers (round 11, GPTOSS_KV=fp8)
// A stored K or V vector (one key, one KV head, 64 dims) is 64 int8 codes in [-127, 127] and one scale: value =
// code * scale. The scale is max|x| / 127 rounded up to 4 significant bits, so code * scale has <= 11 significant bits:
// it is an fp16 value (the prefill flash attention's operands) as long as it lies in fp16's normal range (a vector
// with max|x| in about [6e-5, 65504]; outside it the fp16 operand is rounded / saturated, as the bf16 store's is).
// The decode attention uses the codes as integers and the scale as a factor of the sum. A vector of zeros (or one
// below 2^-126) gets the scale 0 and codes 0; a non-finite element saturates (the code is clamped).
__device__ __forceinline__ float kv8_scale(float amax) {
    if (!(amax > 0.f)) return 0.f;   // codes 0 (kv8_code), value 0; a scale of 1 would take part in k_attn8's tile normalisation of p * V scale
    amax = fminf(amax, 3.0e38f);
    const uint32_t b = (__float_as_uint(amax * (1.f / 127.f)) + 0x000fffffu) & 0xfff00000u;
    return __uint_as_float(b);
}
__device__ __forceinline__ int kv8_code(float x, float s) {
    const float c = s > 0.f ? rintf(x / s) : 0.f;
    return (int)fminf(fmaxf(c == c ? c : 0.f, -127.f), 127.f);
}
// 12-bit K (round 12, GPTOSS_KV=k12): code in [-2047, 2047] = 16 * hi + lo with hi = code >> 4 in the int8 plane
// (the 8-bit store's K rows) and lo = code & 15 in a nibble plane; the scale is a power of two >= max|x| / 2047, so
// code * scale has <= 11 significant bits and is an fp16 value as the 8-bit codes are (same range condition).
// Nibble plane of a (row, KV head): 8 words; word 4 * hf + c holds the nibbles of dims 32 hf + 4 c + e at bits 4 e
// and of dims 32 hf + 16 + 4 c + e at bits 16 + 4 e (e = 0..3): one word is the two B registers of a lane of the
// decode kernel's score MMA.
__device__ __forceinline__ float kv12_scale(float amax) {
    if (!(amax > 0.f)) return 0.f;
    amax = fminf(amax, 3.0e38f);
    const uint32_t b = (__float_as_uint(amax * (1.f / 2047.f)) + 0x007fffffu) & 0xff800000u;
    return __uint_as_float(b);
}
__device__ __forceinline__ int kv12_code(float x, float s) {
    const float c = s > 0.f ? rintf(x / s) : 0.f;
    return (int)fminf(fmaxf(c == c ? c : 0.f, -2047.f), 2047.f);
}
// 4 nibbles (16 bits) -> 4 bytes 0..15
__device__ __forceinline__ uint32_t kv12_nib(uint32_t x) {
    x = (x | (x << 8)) & 0x00ff00ffu;
    return (x | (x << 4)) & 0x0f0f0f0fu;
}
// Quality emulation (GPTOSS_KV=fp8sim): the rows just written to the bf16 store are replaced by their 8-bit values
// (rounded to bf16 again: a little worse than the real store). One warp per (row, KV head, K | V).
// posp: decode rows (row = posp[MAXR + i] + posp[i]); null: rows row0 + i.
// the vector as a warp holds it (lane: dims lane and lane + 32) times the 64-point Hadamard matrix / 8 (its own inverse)
__device__ __forceinline__ void kv8_had(float& a, float& b, int lane) {
    const float t = a + b;
    b = a - b;
    a = t;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        const float oa = __shfl_xor_sync(0xffffffffu, a, o), ob = __shfl_xor_sync(0xffffffffu, b, o);
        a = (lane & o) ? oa - a : a + oa;
        b = (lane & o) ? ob - b : b + ob;
    }
    a *= 0.125f;
    b *= 0.125f;
}
__device__ __forceinline__ float kv8_sign(int d) { return ((0x9E3779B9u * (unsigned)(d + 1)) >> 15 & 1u) ? -1.f : 1.f; }
__global__ void __launch_bounds__(256) k_kv_sim(bf16* Kd, bf16* Vd, int kvn, const int* posp, int row0, int n, int mode) {
    const int w = blockIdx.x * 8 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (w >= n * kvn * 2) return;
    const int i = w / (kvn * 2), hv = w % (kvn * 2), kh = hv >> 1;
    if (!(mode >> (hv & 1) & 1)) return;   // bit 0: K, bit 1: V, bit 2: quantise in the rotated basis
    const size_t row = posp ? (size_t)posp[MAXR + i] + (size_t)posp[i] : (size_t)row0 + i;
    bf16* v = ((hv & 1) ? Vd : Kd) + (row * kvn + kh) * 64;
    float a = bf2f(v[lane]), b = bf2f(v[lane + 32]);
    const float sa = kv8_sign(lane), sb = kv8_sign(lane + 32);
    if (mode & 4) {
        a *= sa;
        b *= sb;
        kv8_had(a, b, lane);
    }
    float am = fmaxf(fabsf(a), fabsf(b));
#pragma unroll
    for (int o = 16; o; o >>= 1) am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, o));
    const float s = kv8_scale(am);
    a = (float)kv8_code(a, s) * s;
    b = (float)kv8_code(b, s) * s;
    if (mode & 4) {
        kv8_had(a, b, lane);
        a *= sa;
        b *= sb;
    }
    v[lane] = __float2bfloat16_rn(a);
    v[lane + 32] = __float2bfloat16_rn(b);
}
// The 8-bit store (GPTOSS_KV=fp8). Vectors are stored in a rotated basis: x' = (x o sign) H / 8 with the 64-point
// Hadamard matrix H (kv8_had), which spreads a vector's large elements over all dims before the 8-bit rounding
// (measured: KL to the bf16 store 3x lower than without). q is rotated the same way, so q' . k' = q . k; the
// attention output of rotated V is the rotated output: the combine (decode) / the kernel's end (prefill) turn it back.
// Rows [row][kvn][64] of int8 for K and for V, S[row][kvn] = (bf16 bits of the K scale) | (bf16 bits of the V scale) << 16.
// k_kv_pack: one warp per (row, q head | KV head x K | V): q heads are rotated in place (fp32), K / V rows come from
// bf16 staging rows and go to the store. posp: decode step (staging row i, store row posp[MAXR + i] + posp[i]);
// null: staging rows srow0 + i, store rows row0 + i (prefill). nq = 0: K / V only; kvn = 0: q rows only.
__global__ void __launch_bounds__(256) k_kv_pack(const bf16* Ks, const bf16* Vs, float* Q, int nq, int kvn, const int* posp,
                                                 int srow0, int row0, int n, int8_t* K8, int8_t* V8, uint32_t* S, uint32_t* K4 = nullptr) {
    const size_t w = (size_t)blockIdx.x * 8 + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31, per = nq + 2 * kvn;
    if (w >= (size_t)n * per) return;
    const int i = (int)(w / per), j = (int)(w % per);
    const float sa = kv8_sign(lane), sb = kv8_sign(lane + 32);
    if (j < nq) {
        float* qp = Q + ((size_t)i * nq + j) * 64;
        float a = qp[lane] * sa, b = qp[lane + 32] * sb;
        kv8_had(a, b, lane);
        qp[lane] = a;
        qp[lane + 32] = b;
        return;
    }
    const int kh = (j - nq) >> 1, isv = (j - nq) & 1;
    const size_t srow = posp ? (size_t)i : (size_t)srow0 + i, row = posp ? (size_t)posp[MAXR + i] + (size_t)posp[i] : (size_t)row0 + i;
    const bf16* src = (isv ? Vs : Ks) + (srow * kvn + kh) * 64;
    float a = bf2f(src[lane]) * sa, b = bf2f(src[lane + 32]) * sb;
    kv8_had(a, b, lane);
    float am = fmaxf(fabsf(a), fabsf(b));
#pragma unroll
    for (int o = 16; o; o >>= 1) am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, o));
    int8_t* dst = (isv ? V8 : K8) + (row * kvn + kh) * 64;
    if (K4 && !isv) {   // 12-bit K: high 8 bits to the int8 plane, low nibbles to the nibble plane
        const float sc = kv12_scale(am);
        const int ca = kv12_code(a, sc), cb = kv12_code(b, sc);
        dst[lane] = (int8_t)(ca >> 4);
        dst[lane + 32] = (int8_t)(cb >> 4);
        const int shn = 16 * (lane >> 4) + 4 * (lane & 3);   // the lane's dims: lane (word (lane & 15) >> 2) and lane + 32 (word 4 + ...)
        uint32_t wa = (uint32_t)(ca & 15) << shn, wb = (uint32_t)(cb & 15) << shn;
        wa |= __shfl_xor_sync(0xffffffffu, wa, 1);
        wb |= __shfl_xor_sync(0xffffffffu, wb, 1);
        wa |= __shfl_xor_sync(0xffffffffu, wa, 2);
        wb |= __shfl_xor_sync(0xffffffffu, wb, 2);
        wa |= __shfl_xor_sync(0xffffffffu, wa, 16);
        wb |= __shfl_xor_sync(0xffffffffu, wb, 16);
        if (lane < 16 && (lane & 3) == 0) {
            uint32_t* nd = K4 + (row * kvn + kh) * 8;
            nd[lane >> 2] = wa;
            nd[4 + (lane >> 2)] = wb;
        }
        if (lane == 0) ((uint16_t*)(S + row * kvn + kh))[0] = (uint16_t)(__float_as_uint(sc) >> 16);
        return;
    }
    const float sc = kv8_scale(am);
    dst[lane] = (int8_t)kv8_code(a, sc);
    dst[lane + 32] = (int8_t)kv8_code(b, sc);
    if (lane == 0) ((uint16_t*)(S + row * kvn + kh))[isv] = (uint16_t)(__float_as_uint(sc) >> 16);
}
__device__ __forceinline__ void mma_s8(int* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
// Decode attention of a full layer over the 8-bit store, on int8 tensor cores. Block = (KV head, split, step row), as
// k_attn; output: the same partials (m, l, o[64]) per (query head, split), o in the rotated basis (k_attn_comb8).
// Keys go in tiles of 32 counted from the sequence's first key; a split is a run of whole tiles (rule by the row's
// own key count only: nt tiles, ns = min(NSPLIT, ceil(nt / 8)) splits of ceil(nt / ns) tiles), a warp takes every 8th
// tile of its split. A row's result does not depend on the other rows of the step or on where its sequence lies.
// Lane = (g = lane >> 2, c = lane & 3). Per tile:
//  - scores: q of query head g as two signed 8-bit limbs (value = (256 hi + lo) qmax / 32512: 16 bits of the head's
//    largest |q|; against the 8-bit K this adds about 1 % to a score's error) are MMA rows g and g + 8, the
//    K bytes are the B operand straight from memory (column j of group n = key 8 n + (j >> 1) + 4 (j & 1), so the
//    lane ends up with the scores of keys 8 n + c and 8 n + 4 + c); the integer sums are exact (|sum| <= 2.7e8;
//    their fp32 form rounds above 2^24, at 6e-8 of the value); times the key's K scale;
//  - softmax inside the tile (max mt, p = exp(s - mt)), p times the key's V scale, normalised by the tile's largest
//    and split into two limbs the same way (error <= 2^-16 of the tile's largest weight);
//  - P.V: MMA rows = (limb, head of a pair, dim slot d of 4), k = (key slot, d): a lane's A bytes are its head's
//    limbs at byte d, the B operand is a 4-dim word of V straight from memory; C = sums for dims 4 n + d. A lane
//    gets the limbs of the 4 heads 2 hp + (g >> 2) by shuffles and holds their dims 32 half + 8 c + 4 b + (g & 3);
//  - the tile joins the running (m, l, o) as in the online softmax of k_attn_pf.
// The last tile may reach up to 31 rows past the sequence's last key: those bytes are read (the store has 32 rows of
// slack) and never used (their scores are NEG_INF, their limbs 0).
// K12 = 1 (GPTOSS_KV=k12): K codes of 12 bits; the nibble plane K4 goes through a second score MMA with the same q
// limbs (bytes 0..15 are int8 values), score = (16 * sum over the int8 plane + sum over the nibbles) * q scale * K scale.
template <int K12>
__global__ void __launch_bounds__(256) k_attn8(const float* q, const int8_t* K8, const int8_t* V8, const uint32_t* S, int kvn,
                                               const int* posp, float* apart, const uint32_t* K4 = nullptr) {
    __shared__ float sm[8][8], sl[8][8];
    __shared__ float so[8][512];
    const int kh = blockIdx.x, split = blockIdx.y;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, c = lane & 3;
    const int pos = posp[blockIdx.z];
    const size_t kbase = (size_t)posp[MAXR + blockIdx.z];
    q += (size_t)blockIdx.z * kvn * 512;
    apart += (size_t)blockIdx.z * kvn * 8 * NSPLIT * APW;
    float* ap = apart + ((size_t)(kh * 8) * NSPLIT + split) * APW;   // + head * NSPLIT * APW
    const int nt = (pos + 32) >> 5, ns = min(NSPLIT, (nt + 7) >> 3), per = (nt + ns - 1) / ns;
    const int t0 = split * per, t1 = min(nt, t0 + per);
    if (t0 >= t1) {
        if (threadIdx.x < 8) {
            float* p = ap + (size_t)threadIdx.x * NSPLIT * APW;
            p[0] = NEG_INF;
            p[1] = 0.f;
        }
        return;
    }
    float m = NEG_INF, l = 0.f, o[4][2][2];
#pragma unroll
    for (int i = 0; i < 16; i++) o[i >> 2][(i >> 1) & 1][i & 1] = 0.f;
    if (t0 + warp < t1) {
        // q limbs of head g: dims 32 half + 16 j + 4 c + i
        uint32_t qa[2][4];
        float qs;
        {
            float qv[16], qm = 0.f;
            const float4* qp = (const float4*)(q + (kh * 8 + g) * 64);
#pragma unroll
            for (int i = 0; i < 4; i++) {
                const float4 v = qp[(i >> 1) * 8 + (i & 1) * 4 + c];
                qv[4 * i] = v.x;
                qv[4 * i + 1] = v.y;
                qv[4 * i + 2] = v.z;
                qv[4 * i + 3] = v.w;
                qm = fmaxf(fmaxf(qm, fmaxf(fabsf(v.x), fabsf(v.y))), fmaxf(fabsf(v.z), fabsf(v.w)));
            }
            qm = fmaxf(qm, __shfl_xor_sync(0xffffffffu, qm, 1));
            qm = fmaxf(qm, __shfl_xor_sync(0xffffffffu, qm, 2));
            qm = fminf(qm, 3.0e38f);
            const float qi = qm > 0.f ? 32512.f / qm : 0.f;
            qs = qm * (0.125f / 32512.f);
#pragma unroll
            for (int i = 0; i < 4; i++) {
                uint32_t hi = 0, lo = 0;
#pragma unroll
                for (int e = 0; e < 4; e++) {
                    const float r = rintf(qv[4 * i + e] * qi);
                    const int v = max(-32512, min(32512, (int)(r == r ? r : 0.f)));
                    const int vh = (v + 128) >> 8;   // v = 256 vh + vl, vh in [-127, 127], vl in [-128, 127]
                    hi |= (uint32_t)(vh & 0xff) << (8 * e);
                    lo |= (uint32_t)((v - 256 * vh) & 0xff) << (8 * e);
                }
                qa[i >> 1][(i & 1) * 2] = hi;
                qa[i >> 1][(i & 1) * 2 + 1] = lo;
            }
        }
        const size_t kst = (size_t)kvn * 64;
        const int sh = 8 * (g & 3), ks = (g >> 1) + 4 * (g & 1);
        for (int tt = t0 + warp; tt < t1; tt += 8) {
            const size_t r0 = kbase + (size_t)tt * 32;
            const int8_t* kp = K8 + (r0 * kvn + kh) * 64;
            const int8_t* vp = V8 + (r0 * kvn + kh) * 64;
            const uint32_t* sp = S + r0 * kvn + kh;
            float sc[4][2], vs[4][2];
            uint32_t vw[4][2][2];
#pragma unroll
            for (int n = 0; n < 4; n++) {
                int cc[4] = {0, 0, 0, 0};
                const uint32_t* kw = (const uint32_t*)(kp + (size_t)(8 * n + ks) * kst);
#pragma unroll
                for (int hf = 0; hf < 2; hf++) mma_s8(cc, qa[hf][0], qa[hf][1], qa[hf][2], qa[hf][3], kw[hf * 8 + c], kw[hf * 8 + 4 + c]);
                int cl[4] = {0, 0, 0, 0};
                if (K12) {
                    const uint32_t* nw = K4 + ((r0 + 8 * n + ks) * kvn + kh) * 8;
#pragma unroll
                    for (int hf = 0; hf < 2; hf++) {
                        const uint32_t w = nw[hf * 4 + c];
                        mma_s8(cl, qa[hf][0], qa[hf][1], qa[hf][2], qa[hf][3], kv12_nib(w & 0xffffu), kv12_nib(w >> 16));
                    }
                }
#pragma unroll
                for (int b = 0; b < 2; b++) {
                    const int slot = 8 * n + 4 * b + c;
                    const uint32_t sw = sp[(size_t)slot * kvn];
                    const bool valid = tt * 32 + slot <= pos;
                    sc[n][b] = valid ? (K12 ? (float)(cc[b] * 256 + cc[2 + b]) * 16.f + (float)(cl[b] * 256 + cl[2 + b]) : (float)(cc[b] * 256 + cc[2 + b])) * qs * __uint_as_float((sw & 0xffffu) << 16) : NEG_INF;
                    vs[n][b] = valid ? __uint_as_float(sw & 0xffff0000u) : 0.f;
                    const uint32_t* vq = (const uint32_t*)(vp + (size_t)slot * kst);
                    vw[n][b][0] = vq[g];
                    vw[n][b][1] = vq[8 + g];
                }
            }
            float mt = NEG_INF;
#pragma unroll
            for (int n = 0; n < 4; n++) mt = fmaxf(mt, fmaxf(sc[n][0], sc[n][1]));
            mt = fmaxf(mt, __shfl_xor_sync(0xffffffffu, mt, 1));
            mt = fmaxf(mt, __shfl_xor_sync(0xffffffffu, mt, 2));
            float lt = 0.f, xm = 0.f;
#pragma unroll
            for (int n = 0; n < 4; n++)
#pragma unroll
                for (int b = 0; b < 2; b++) {
                    const float p = sc[n][b] > NEG_INF ? expf(sc[n][b] - mt) : 0.f;
                    lt += p;
                    sc[n][b] = p * vs[n][b];   // from here: the weight times the V scale
                    xm = fmaxf(xm, sc[n][b]);
                }
            lt += __shfl_xor_sync(0xffffffffu, lt, 1);
            lt += __shfl_xor_sync(0xffffffffu, lt, 2);
            xm = fmaxf(xm, __shfl_xor_sync(0xffffffffu, xm, 1));
            xm = fmaxf(xm, __shfl_xor_sync(0xffffffffu, xm, 2));
            xm = fminf(xm, 3.0e38f);
            const float xi = xm > 0.f ? 32512.f / xm : 0.f;
            uint32_t pw[4];
#pragma unroll
            for (int n = 0; n < 4; n++) {
                const float ra = rintf(sc[n][0] * xi), rb = rintf(sc[n][1] * xi);
                const int v0 = max(0, min(32512, (int)(ra == ra ? ra : 0.f))), v1 = max(0, min(32512, (int)(rb == rb ? rb : 0.f)));
                const int h0 = (v0 + 128) >> 8, h1 = (v1 + 128) >> 8;   // v = 256 h + l, h in [0, 127], l in [-128, 127]
                pw[n] = (uint32_t)h0 | (uint32_t)((v0 - 256 * h0) & 0xff) << 8 | (uint32_t)h1 << 16 | (uint32_t)((v1 - 256 * h1) & 0xff) << 24;
            }
            const float mn = fmaxf(m, mt);
            const float fa = expf(m - mn), ft = expf(mt - mn);
            l = l * fa + lt * ft;
            m = mn;
            const float f2 = ft * xm * (1.f / 32512.f);
#pragma unroll
            for (int hp = 0; hp < 4; hp++) {
                const int src = (2 * hp + (g >> 2)) * 4 + c;
                const float fah = __shfl_sync(0xffffffffu, fa, src), f2h = __shfl_sync(0xffffffffu, f2, src);
                int cc[2][4] = {{0, 0, 0, 0}, {0, 0, 0, 0}};
#pragma unroll
                for (int n = 0; n < 4; n++) {
                    const uint32_t w = __shfl_sync(0xffffffffu, pw[n], src);
                    const uint32_t a0 = (w & 0xffu) << sh, a1 = ((w >> 8) & 0xffu) << sh, a2 = ((w >> 16) & 0xffu) << sh, a3 = (w >> 24) << sh;
                    mma_s8(cc[0], a0, a1, a2, a3, vw[n][0][0], vw[n][1][0]);
                    mma_s8(cc[1], a0, a1, a2, a3, vw[n][0][1], vw[n][1][1]);
                }
#pragma unroll
                for (int i = 0; i < 4; i++) o[hp][i >> 1][i & 1] = o[hp][i >> 1][i & 1] * fah + f2h * (float)(cc[i >> 1][i & 1] * 256 + cc[i >> 1][2 + (i & 1)]);
            }
        }
    }
    if (c == 0) {
        sm[warp][g] = m;
        sl[warp][g] = l;
    }
#pragma unroll
    for (int i = 0; i < 16; i++) so[warp][((2 * (i >> 2) + (g >> 2)) << 6) + 32 * ((i >> 1) & 1) + 8 * c + 4 * (i & 1) + (g & 3)] = o[i >> 2][(i >> 1) & 1][i & 1];
    __syncthreads();
    for (int e = threadIdx.x; e < 512; e += 256) {
        const int h2 = e >> 6;
        float M = NEG_INF;
        for (int w = 0; w < 8; w++) M = fmaxf(M, sm[w][h2]);
        float L = 0.f, O = 0.f;
        for (int w = 0; w < 8; w++)
            if (sm[w][h2] > NEG_INF) {
                const float f = expf(sm[w][h2] - M);
                L += sl[w][h2] * f;
                O += so[w][e] * f;
            }
        float* p = ap + (size_t)h2 * NSPLIT * APW;
        p[2 + (e & 63)] = O;
        if ((e & 63) == 0) {
            p[0] = M;
            p[1] = L;
        }
    }
}
// combine of k_attn8's partials: the splits of the row (k_attn8's rule), the sink, the turn back from the rotated
// basis (o = (o' H / 8) o sign), bf16
__global__ void __launch_bounds__(64) k_attn_comb8(const float* apart, const float* sinks, const int* posp, bf16* ao) {
    __shared__ float sO[64];
    const int qh = blockIdx.x, d = threadIdx.x;
    apart += (size_t)blockIdx.y * gridDim.x * NSPLIT * APW;
    ao += (size_t)blockIdx.y * gridDim.x * 64;
    const int nt = (posp[blockIdx.y] + 32) >> 5, nsp = min(NSPLIT, (nt + 7) >> 3);
    const float* ap = apart + (size_t)qh * NSPLIT * APW;
    float M = NEG_INF;
    for (int s = 0; s < nsp; s++) M = fmaxf(M, ap[s * APW]);
    float L = 0.f, O = 0.f;
    for (int s = 0; s < nsp; s++)
        if (ap[s * APW] > NEG_INF) {
            const float f = expf(ap[s * APW] - M);
            L += ap[s * APW + 1] * f;
            O += ap[s * APW + 2 + d] * f;
        }
    sO[d] = O;
    __syncthreads();
    float u = 0.f;
#pragma unroll 8
    for (int j = 0; j < 64; j++) u += (__popc(d & j) & 1) ? -sO[j] : sO[j];
    u *= 0.125f * kv8_sign(d);
    ao[qh * 64 + d] = __float2bfloat16_rn(u / (L + expf(sinks[qh] - M)));
}
// one KV head of n contiguous bf16 rows as fp32 [n][64]
__global__ void k_kv_f32(const bf16* Kb, int kvn, int kh, int n, float* out) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)n * 64) return;
    out[i] = bf2f(Kb[((i >> 6) * kvn + kh) * 64 + (i & 63)]);
}
// Qg[(t * 8 + hh)][64] = Qf[t0 + t][(kh * 8 + hh)][64]
__global__ void k_gather_q(const float* Qf, int nq, int kh, int t0, int nt, float* Qg) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)nt * 512) return;
    Qg[i] = Qf[(size_t)(t0 + i / 512) * nq * 64 + kh * 512 + (i % 512)];
}
// Scores S [nt * 8][Lk] (raw q.k) -> probabilities in place: scale 1/8, causal (and window) mask, the sink in the
// denominator only. Row r = t * 8 + hh is query position qpos0 + t; column j is key position kpos0 + j.
// The rows are left unnormalised: rinv[r] = 1 / (sum + exp(sink - max)) is applied to the 64 outputs of the row
// after the P.V product (k_scatter_ao), which saves a third pass over S.
__global__ void __launch_bounds__(256) k_softmax_rows(float* S, int Lk, int kpos0, int qpos0, int sliding,
                                                      const float* sinks, float* rinv) {
    __shared__ float red[8];
    const int r = blockIdx.x, t = r >> 3, hh = r & 7;
    float* s = S + (size_t)r * Lk;
    const int qp = qpos0 + t;
    const int j0 = sliding ? max(0, qp - 127 - kpos0) : 0, j1 = min(Lk, qp - kpos0 + 1);
    float m = NEG_INF;
    for (int j = j0 + threadIdx.x; j < j1; j += 256) m = fmaxf(m, s[j]);
    for (int o = 16; o; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = m;
    __syncthreads();
    m = red[0];
    for (int k = 1; k < 8; k++) m = fmaxf(m, red[k]);
    m *= 0.125f;
    __syncthreads();
    float sum = 0.f;
    for (int j = threadIdx.x; j < Lk; j += 256) {
        const float e = (j >= j0 && j < j1) ? expf(s[j] * 0.125f - m) : 0.f;
        s[j] = e;
        sum += e;
    }
    sum = warp_sum(sum);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = sum;
    __syncthreads();
    sum = 0.f;
    for (int k = 0; k < 8; k++) sum += red[k];
    if (threadIdx.x == 0) rinv[r] = 1.f / (sum + expf(sinks[hh] - m));
}
// AO[t0 + t][(kh * 8 + hh)][64] = bf16(Og[t * 8 + hh][64] * rinv[t * 8 + hh])
__global__ void k_scatter_ao(const float* Og, const float* rinv, int ncol, int kh, int t0, int nt, bf16* AO) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)nt * 512) return;
    AO[(size_t)(t0 + i / 512) * ncol + kh * 512 + (i % 512)] = __float2bfloat16_rn(Og[i] * rinv[i >> 6]);
}
// out[i] = p0 + p1 + p2 + p3 in rank order (null = skipped): the owner's sum of a reduce-scatter
__global__ void k_sum_parts(float* out, const float* p0, const float* p1, const float* p2, const float* p3, size_t n) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float s = 0.f;
    if (p0) s += p0[i];
    if (p1) s += p1[i];
    if (p2) s += p2[i];
    if (p3) s += p3[i];
    out[i] = s;
}
// X[t][i] += bias[i] for T rows of width N
__global__ void k_bias_rows(float* X, const bf16* bias, int T, int N) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)T * N) return;
    X[i] += bf2f(bias[i % N]);
}
// out[r] = X[rows[r]] (bf16 rows of 2880, 16 B units)
__global__ void k_gather_rows(const bf16* X, const int* rows, int n, bf16* out) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)n * 360) return;
    const size_t r = i / 360, u = i % 360;
    ((uint4*)out)[i] = ((const uint4*)(X + (size_t)rows[r] * 2880))[u];
}
// MXFP4 rows -> bf16 (exact): one thread per 16 B block (32 values)
__global__ void k_deq(const uint8_t* cod, int nrows, int KB, bf16* out) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)nrows * KB) return;
    const uint8_t* rec = cod + (i / KB) * mx_rs(KB);
    const int bq = (int)(i % KB);
    const uint4 c = *(const uint4*)(rec + bq * 16);
    const float sc = mx_scale((uint8_t)mx_sbyte(rec, KB, bq));
    const uint32_t cw[4] = {c.x, c.y, c.z, c.w};
    __align__(16) bf16 o[32];
#pragma unroll
    for (int j = 0; j < 4; j++) {
        const uint32_t w = cw[j];
        const float2 e04 = fp4pair(w), e15 = fp4pair(w >> 4), e26 = fp4pair(w >> 8), e37 = fp4pair(w >> 12);
        o[8 * j] = __float2bfloat16_rn(e04.x * sc);
        o[8 * j + 1] = __float2bfloat16_rn(e15.x * sc);
        o[8 * j + 2] = __float2bfloat16_rn(e26.x * sc);
        o[8 * j + 3] = __float2bfloat16_rn(e37.x * sc);
        o[8 * j + 4] = __float2bfloat16_rn(e04.y * sc);
        o[8 * j + 5] = __float2bfloat16_rn(e15.y * sc);
        o[8 * j + 6] = __float2bfloat16_rn(e26.y * sc);
        o[8 * j + 7] = __float2bfloat16_rn(e37.y * sc);
    }
    uint4* dst = (uint4*)(out + i * 32);
#pragma unroll
    for (int j = 0; j < 4; j++) dst[j] = ((const uint4*)o)[j];
}
// A[r][j] = bf16(swiglu(GU[r][2j] + b, GU[r][2j+1] + b) * w[r]) for n rows of one expert
__global__ void k_act_rows(const float* GU, const bf16* bias, const float* w, int n, int NR, bf16* A) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int half = NR / 2;
    if (i >= (size_t)n * half) return;
    const size_t r = i / half;
    const int j = (int)(i % half);
    const float* g = GU + r * NR;
    A[i] = __float2bfloat16_rn(swiglu(g[2 * j] + bf2f(bias[2 * j]), g[2 * j + 1] + bf2f(bias[2 * j + 1])) * w[r]);
}
// part[rows[r]] += O[r] (+ w[r] * bias on rank 0); the rows of one expert are distinct
__global__ void k_scatter_add(const float* O, const int* rows, const float* w, const bf16* bias, int n, float* part) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)n * 2880) return;
    const size_t r = i / 2880;
    const int c = (int)(i % 2880);
    float v = O[i];
    if (bias) v += w[r] * bf2f(bias[c]);
    part[(size_t)rows[r] * 2880 + c] += v;
}

// ---------------------------------------------------------------- fused MXFP4 GEMM (prefill, switch GPTOSS_PF_MOE=fused)
// out[o + i][row] = X[o + i] . W_e[row] for every unit (expert e, first sorted row o, cnt <= 32 rows): the weights
// are built as bf16 MMA fragments straight from the MXFP4 bytes (code x block scale is exact in bf16), fp32
// accumulation. Block = 128 weight rows (a warp owns 16) x one unit; a warp's 4 accumulator tiles are the unit's
// 4 x 8 token rows. First step towards the k_mx_gemm of the design (no shared-memory staging of X yet).
__device__ __forceinline__ uint32_t mx_frag(uint32_t byte, float sc) {   // two codes of one byte -> bf16 pair
    const float2 v = fp4pair((byte & 15u) | ((byte >> 4) << 16));
    const bf162 r = __floats2bfloat162_rn(v.x * sc, v.y * sc);
    return *(const uint32_t*)&r;
}
__global__ void __launch_bounds__(256) k_mx_gemm(const uint8_t* cod, int NRW, int KB, const bf16* X,
                                                 int ldx, const int* units, float* out, int ldo) {
    const int u = blockIdx.y;
    const int e = units[3 * u], o = units[3 * u + 1], cnt = units[3 * u + 2];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const int row0 = blockIdx.x * 128 + warp * 16;
    if (row0 >= NRW) return;
    const size_t rA = (size_t)e * NRW + row0 + g, rB = rA + 8;
    const uint8_t *cA = cod + rA * mx_rs(KB), *cB = cod + rB * mx_rs(KB);
    float c[4][4];
#pragma unroll
    for (int i = 0; i < 4; i++)
#pragma unroll
        for (int j = 0; j < 4; j++) c[i][j] = 0.f;
    const bf16* xr[4];
#pragma unroll
    for (int i = 0; i < 4; i++) xr[i] = X + (size_t)(o + min(i * 8 + g, cnt - 1)) * ldx + 2 * t;   // clamped: masked below
    for (int b = 0; b < KB; b++) {
        const uint4 wa = *(const uint4*)(cA + b * 16), wb = *(const uint4*)(cB + b * 16);
        const float sa = mx_scale((uint8_t)mx_sbyte(cA, KB, b)), sb = mx_scale((uint8_t)mx_sbyte(cB, KB, b));
#pragma unroll
        for (int half = 0; half < 2; half++) {
            const uint32_t alo = half ? wa.z : wa.x, ahi = half ? wa.w : wa.y;
            const uint32_t blo = half ? wb.z : wb.x, bhi = half ? wb.w : wb.y;
            const uint32_t a0 = mx_frag((alo >> (8 * t)) & 255u, sa), a1 = mx_frag((blo >> (8 * t)) & 255u, sb);
            const uint32_t a2 = mx_frag((ahi >> (8 * t)) & 255u, sa), a3 = mx_frag((bhi >> (8 * t)) & 255u, sb);
            const int k0 = b * 32 + half * 16;
#pragma unroll
            for (int i = 0; i < 4; i++) {
                uint32_t b0 = *(const uint32_t*)(xr[i] + k0), b1 = *(const uint32_t*)(xr[i] + k0 + 8);
                if (i * 8 + g >= cnt) b0 = b1 = 0u;
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                             : "+f"(c[i][0]), "+f"(c[i][1]), "+f"(c[i][2]), "+f"(c[i][3])
                             : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
            }
        }
    }
#pragma unroll
    for (int i = 0; i < 4; i++)
#pragma unroll
        for (int j = 0; j < 2; j++) {
            const int tok = i * 8 + 2 * t + j;
            if (tok < cnt) {
                float* po = out + (size_t)(o + tok) * ldo + row0 + g;
                po[0] = c[i][j];
                po[8] = c[i][2 + j];
            }
        }
}
// Second form (plan P3 pieces): the unit's activations are staged k-tiled in shared memory (32 rows x 480 columns,
// 30 KB, coalesced 16 B loads instead of 4 B global loads per MMA), the fragments come from a 256-entry byte ->
// bf16-pair table (unscaled codes), and the block scale multiplies the block's partial sums (exact: a power of two)
// instead of every weight.
__global__ void __launch_bounds__(256) k_mx_gemm2(const uint8_t* cod, int NRW, int KB, const bf16* X,
                                                  int ldx, const int* units, float* out, int ldo) {
    __shared__ uint32_t tab[256];
    __shared__ __align__(16) bf16 xs[32 * 480];
    {
        const float2 v = fp4pair((threadIdx.x & 15u) | ((threadIdx.x >> 4) << 16));
        const bf162 r = __floats2bfloat162_rn(v.x * 16384.f, v.y * 16384.f);
        tab[threadIdx.x] = *(const uint32_t*)&r;
    }
    const int u = blockIdx.y;
    const int e = units[3 * u], o = units[3 * u + 1], cnt = units[3 * u + 2];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const int row0 = blockIdx.x * 128 + warp * 16;
    const bool live = row0 < NRW;
    const size_t rA = (size_t)e * NRW + (live ? row0 : 0) + g, rB = rA + 8;
    const uint8_t *cA = cod + rA * mx_rs(KB), *cB = cod + rB * mx_rs(KB);
    float c[4][4];
#pragma unroll
    for (int i = 0; i < 4; i++)
#pragma unroll
        for (int j = 0; j < 4; j++) c[i][j] = 0.f;
    for (int ks = 0; ks * 15 < KB; ks++) {
        __syncthreads();   // the table (first pass) / every warp is done with the previous tile
        for (int q = threadIdx.x; q < 32 * 60; q += 256) {
            const int tok = q / 60, col = q % 60;
            // (columns past the row's K stay zero: the last tile is short when KB % 15 != 0, e.g. the 4-card plan)
            ((uint4*)xs)[q] = tok < cnt && ks * 480 + col * 8 < KB * 32 ? *(const uint4*)(X + (size_t)(o + tok) * ldx + ks * 480 + col * 8)
                                                                        : make_uint4(0, 0, 0, 0);
        }
        __syncthreads();
        if (!live) continue;
        const int nb = min(15, KB - ks * 15);
        for (int b = 0; b < nb; b++) {
            const int bb = ks * 15 + b;
            const uint4 wa = *(const uint4*)(cA + bb * 16), wb = *(const uint4*)(cB + bb * 16);
            const float sa = __int_as_float((int)mx_sbyte(cA, KB, bb) << 23), sb = __int_as_float((int)mx_sbyte(cB, KB, bb) << 23);
            float p[4][4];
#pragma unroll
            for (int i = 0; i < 4; i++)
#pragma unroll
                for (int j = 0; j < 4; j++) p[i][j] = 0.f;
#pragma unroll
            for (int half = 0; half < 2; half++) {
                const uint32_t alo = half ? wa.z : wa.x, ahi = half ? wa.w : wa.y;
                const uint32_t blo = half ? wb.z : wb.x, bhi = half ? wb.w : wb.y;
                const uint32_t a0 = tab[(alo >> (8 * t)) & 255u], a1 = tab[(blo >> (8 * t)) & 255u];
                const uint32_t a2 = tab[(ahi >> (8 * t)) & 255u], a3 = tab[(bhi >> (8 * t)) & 255u];
                const int k0 = b * 32 + half * 16 + 2 * t;
#pragma unroll
                for (int i = 0; i < 4; i++) {
                    const bf16* xr = xs + (i * 8 + g) * 480 + k0;
                    const uint32_t b0 = *(const uint32_t*)xr, b1 = *(const uint32_t*)(xr + 8);
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                                 : "+f"(p[i][0]), "+f"(p[i][1]), "+f"(p[i][2]), "+f"(p[i][3])
                                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
                }
            }
#pragma unroll
            for (int i = 0; i < 4; i++) {
                c[i][0] += p[i][0] * sa;
                c[i][1] += p[i][1] * sa;
                c[i][2] += p[i][2] * sb;
                c[i][3] += p[i][3] * sb;
            }
        }
    }
    if (!live) return;
#pragma unroll
    for (int i = 0; i < 4; i++)
#pragma unroll
        for (int j = 0; j < 2; j++) {
            const int tok = i * 8 + 2 * t + j;
            if (tok < cnt) {
                float* po = out + (size_t)(o + tok) * ldo + row0 + g;
                po[0] = c[i][j];
                po[8] = c[i][2 + j];
            }
        }
}
// k_act_rows over all sorted rows of a chunk: row r belongs to expert ex[r]
__global__ void k_act_all(const float* GU, const bf16* bias, const int* ex, const float* w, int n, int NR, bf16* A) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int half = NR / 2;
    if (i >= (size_t)n * half) return;
    const size_t r = i / half;
    const int j = (int)(i % half);
    const float* g = GU + r * NR;
    const bf16* b = bias + (size_t)ex[r] * NR;
    A[i] = __float2bfloat16_rn(swiglu(g[2 * j] + bf2f(b[2 * j]), g[2 * j + 1] + bf2f(b[2 * j + 1])) * w[r]);
}
// part[t] = sum_k O[inv[t][k]] (+ on rank 0 sum_k w[t][k] * bias[ids[t][k]]), k order as in the decode step
__global__ void k_sum4(const float* O, const int* inv, const int* ids, const float* rw, const bf16* bias, int T, float* part) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)T * 2880) return;
    const size_t t = i / 2880;
    const int c = (int)(i % 2880);
    float v = 0.f;
#pragma unroll
    for (int k = 0; k < 4; k++) v += O[(size_t)inv[t * 4 + k] * 2880 + c];
    if (bias)
#pragma unroll
        for (int k = 0; k < 4; k++) v += rw[t * 4 + k] * bf2f(bias[(size_t)ids[t * 4 + k] * 2880 + c]);
    part[i] = v;
}

// ---- prefill attention, flash form on bf16 tensor cores (speed plan P2)
// One block = one KV head x 8 query heads x 16 * QG query rows; a warp = (row group, query head). The keys come in
// tiles of 64 rows staged in shared memory (K row-major, V transposed; row stride 72 = no bank conflicts on the
// fragment loads), shared by every warp of the block. Scores never leave the registers: online softmax per row
// (running max m, sum l, outputs rescaled when m grows), the sink enters the denominator at the row's end.
// Key rows: index j of Kd / Vd rows [j][kvn][64] is key position kpos_row0 + j; rows [row_lo, row_hi) are valid.
// Query row t is position qpos0 + t. Sliding layers see keys qp - 127 .. qp. Numerics against the sgemm form: the MMA
// operands are fp16 (q and the probabilities rounded to 11 bits; K / V bf16 -> fp16 is exact above 6e-5), fp32
// accumulation, the sums run tile by tile. bf16 operands (8 bits) were measured first: per-layer median error
// 1.5e-3 (8.4e-4 with an exact q) against 4.9e-4 of the sgemm form, over the 5e-4 gate.
constexpr int APF_LD = 72;
// QL = 1: the query also sends its low limb (q - fp16(q)) through a second score MMA, so the scores see the fp32 q
// to ~2^-22 (the keys are bf16 in the store anyway).
// PL = 1: the same for the probabilities in the P.V product (p - fp16(p) through a second MMA).
// KV8 = 1: Kd / Vd are the 8-bit store (int8 rows, scales Sd; k_kv_pack), q is rotated: code * scale is exact in
// fp16, and the output is turned back from the rotated basis before it is rounded to bf16.
// KV8 = 2: the same with 12-bit K codes (int8 plane in Kd, nibble plane K4d; k12).
template <int QG, int QL, int PL, int KV8 = 0>
__global__ void __launch_bounds__(256 * QG) k_attn_pf(const float* Qf, int nq, const bf16* Kd, const bf16* Vd, int kvn,
                                                      int T, int qpos0, int kpos_row0, int row_lo, int row_hi, int sliding,
                                                      const float* sinks, bf16* AO, int ncol, const uint32_t* Sd = nullptr,
                                                      const uint32_t* K4d = nullptr) {
    __shared__ __align__(16) __half ks[64 * APF_LD];
    __shared__ __align__(16) __half vt[64 * APF_LD];   // [dim][key]
    const int kh = blockIdx.y, warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, c = lane & 3;
    const int hh = warp & 7, tb = blockIdx.x * 16 * QG, t0 = tb + (warp >> 3) * 16;
    const int tA = t0 + g, tB = tA + 8;
    const int qpA = qpos0 + tA, qpB = qpos0 + tB, win = sliding ? 127 : 0x3fffffff;
    uint32_t qa[4][4], ql[4][4];
#pragma unroll
    for (int kk = 0; kk < 4; kk++)
#pragma unroll
        for (int i = 0; i < 4; i++) {
            const int row = (i & 1) ? tB : tA, col = kk * 16 + 2 * c + (i >> 1) * 8;
            float2 v = make_float2(0.f, 0.f);
            if (row < T) v = *(const float2*)(Qf + ((size_t)row * nq + kh * 8 + hh) * 64 + col);
            v.x = fminf(fmaxf(v.x, -65504.f), 65504.f);   // fp16 range (review L5): saturate, never inf
            v.y = fminf(fmaxf(v.y, -65504.f), 65504.f);
            const __half2 b = __floats2half2_rn(v.x, v.y);
            qa[kk][i] = *(const uint32_t*)&b;
            const __half2 lo = __floats2half2_rn(v.x - __half2float(b.x), v.y - __half2float(b.y));
            ql[kk][i] = *(const uint32_t*)&lo;
        }
    float o[8][4];
#pragma unroll
    for (int n = 0; n < 8; n++)
#pragma unroll
        for (int j = 0; j < 4; j++) o[n][j] = 0.f;
    float mA = NEG_INF, mB = NEG_INF, lA = 0.f, lB = 0.f;
    const int tend = min(T, tb + 16 * QG);                         // one past the block's last query row
    const int jlo = sliding ? max(row_lo, qpos0 + tb - 127 - kpos_row0) : row_lo;
    const int jhi = min(row_hi, qpos0 + tend - kpos_row0);
    const int wq_lo = qpos0 + t0, wq_hi = qpos0 + min(T, t0 + 16) - 1;   // the warp's query positions
    for (int j0 = jlo; j0 < jhi; j0 += 64) {
        __syncthreads();
        for (int q = threadIdx.x; q < 512; q += 256 * QG) {
            const int jr = q >> 3, u = q & 7, row = j0 + jr;
            if (KV8) {
                uint2 kw = make_uint2(0, 0), vw = kw;
                uint32_t nb = 0;   // 12-bit K: the nibbles of dims 8 u .. 8 u + 7 (4 bits each)
                float sk = 0.f, sv = 0.f;
                if (row < jhi) {
                    if (KV8 == 2) {
                        const uint2 nw = *(const uint2*)(K4d + ((size_t)row * kvn + kh) * 8 + (u >> 2) * 4 + (u & 1) * 2);
                        const int sh = (u & 2) ? 16 : 0;
                        nb = ((nw.x >> sh) & 0xffffu) | ((nw.y >> sh) & 0xffffu) << 16;
                    }
                    kw = *(const uint2*)((const int8_t*)Kd + ((size_t)row * kvn + kh) * 64 + u * 8);
                    vw = *(const uint2*)((const int8_t*)Vd + ((size_t)row * kvn + kh) * 64 + u * 8);
                    const uint32_t sw = Sd[(size_t)row * kvn + kh];
                    sk = __uint_as_float((sw & 0xffffu) << 16);
                    sv = __uint_as_float(sw & 0xffff0000u);
                }
                const int8_t *pk = (const int8_t*)&kw, *pv = (const int8_t*)&vw;
#pragma unroll
                for (int i = 0; i < 8; i++) {
                    const float kc = KV8 == 2 ? (float)((int)pk[i] * 16 + (int)((nb >> (4 * i)) & 15u)) : (float)pk[i];
                    ks[jr * APF_LD + u * 8 + i] = __float2half_rn(fminf(fmaxf(kc * sk, -65504.f), 65504.f));
                    vt[(u * 8 + i) * APF_LD + jr] = __float2half_rn(fminf(fmaxf((float)pv[i] * sv, -65504.f), 65504.f));
                }
                continue;
            }
            uint4 kv = make_uint4(0, 0, 0, 0), vv = kv;
            if (row < jhi) {
                kv = *(const uint4*)(Kd + ((size_t)row * kvn + kh) * 64 + u * 8);
                vv = *(const uint4*)(Vd + ((size_t)row * kvn + kh) * 64 + u * 8);
            }
            const bf16 *pk = (const bf16*)&kv, *pv = (const bf16*)&vv;
#pragma unroll
            for (int i = 0; i < 8; i++) {
                ks[jr * APF_LD + u * 8 + i] = __float2half_rn(fminf(fmaxf(__bfloat162float(pk[i]), -65504.f), 65504.f));
                vt[(u * 8 + i) * APF_LD + jr] = __float2half_rn(fminf(fmaxf(__bfloat162float(pv[i]), -65504.f), 65504.f));
            }
        }
        __syncthreads();
        const int kb = kpos_row0 + j0;
        if (t0 >= T || kb > wq_hi || kb + 63 < wq_lo - win) continue;
        float s[8][4];
#pragma unroll
        for (int n = 0; n < 8; n++) {
#pragma unroll
            for (int j = 0; j < 4; j++) s[n][j] = 0.f;
#pragma unroll
            for (int kk = 0; kk < 4; kk++) {
                const __half* kr = ks + (n * 8 + g) * APF_LD + kk * 16 + 2 * c;
                const uint32_t b0 = *(const uint32_t*)kr, b1 = *(const uint32_t*)(kr + 8);
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                             : "+f"(s[n][0]), "+f"(s[n][1]), "+f"(s[n][2]), "+f"(s[n][3])
                             : "r"(qa[kk][0]), "r"(qa[kk][1]), "r"(qa[kk][2]), "r"(qa[kk][3]), "r"(b0), "r"(b1));
                if (QL)
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                                 : "+f"(s[n][0]), "+f"(s[n][1]), "+f"(s[n][2]), "+f"(s[n][3])
                                 : "r"(ql[kk][0]), "r"(ql[kk][1]), "r"(ql[kk][2]), "r"(ql[kk][3]), "r"(b0), "r"(b1));
            }
        }
        float xA = NEG_INF, xB = NEG_INF;
#pragma unroll
        for (int n = 0; n < 8; n++)
#pragma unroll
            for (int j = 0; j < 2; j++) {
                const int kp = kb + n * 8 + 2 * c + j;
                s[n][j] = (kp <= qpA && kp >= qpA - win && tA < T) ? s[n][j] * 0.125f : NEG_INF;
                s[n][2 + j] = (kp <= qpB && kp >= qpB - win && tB < T) ? s[n][2 + j] * 0.125f : NEG_INF;
                xA = fmaxf(xA, s[n][j]);
                xB = fmaxf(xB, s[n][2 + j]);
            }
        xA = fmaxf(xA, __shfl_xor_sync(0xffffffffu, xA, 1));
        xA = fmaxf(xA, __shfl_xor_sync(0xffffffffu, xA, 2));
        xB = fmaxf(xB, __shfl_xor_sync(0xffffffffu, xB, 1));
        xB = fmaxf(xB, __shfl_xor_sync(0xffffffffu, xB, 2));
        const float nA = fmaxf(mA, xA), nB = fmaxf(mB, xB);
        const float fA = expf(mA - nA), fB = expf(mB - nB);   // both NEG_INF: 1 (nothing accumulated yet)
        mA = nA;
        mB = nB;
        float sA = 0.f, sB = 0.f;
        uint32_t pa[8][2], pl[8][2];
#pragma unroll
        for (int n = 0; n < 8; n++) {
            const float a0 = s[n][0] > -1.0e38f ? expf(s[n][0] - nA) : 0.f, a1 = s[n][1] > -1.0e38f ? expf(s[n][1] - nA) : 0.f;
            const float b0 = s[n][2] > -1.0e38f ? expf(s[n][2] - nB) : 0.f, b1 = s[n][3] > -1.0e38f ? expf(s[n][3] - nB) : 0.f;
            const __half2 ra = __floats2half2_rn(a0, a1), rb = __floats2half2_rn(b0, b1);
            if (PL) {
                const __half2 la = __floats2half2_rn(a0 - __half2float(ra.x), a1 - __half2float(ra.y));
                const __half2 lb = __floats2half2_rn(b0 - __half2float(rb.x), b1 - __half2float(rb.y));
                pl[n][0] = *(const uint32_t*)&la;
                pl[n][1] = *(const uint32_t*)&lb;
                sA += a0 + a1;
                sB += b0 + b1;
            } else {
                sA += __half2float(ra.x) + __half2float(ra.y);   // the sum of what the P.V product uses
                sB += __half2float(rb.x) + __half2float(rb.y);
            }
            pa[n][0] = *(const uint32_t*)&ra;
            pa[n][1] = *(const uint32_t*)&rb;
            o[n][0] *= fA;
            o[n][1] *= fA;
            o[n][2] *= fB;
            o[n][3] *= fB;
        }
        sA += __shfl_xor_sync(0xffffffffu, sA, 1);
        sA += __shfl_xor_sync(0xffffffffu, sA, 2);
        sB += __shfl_xor_sync(0xffffffffu, sB, 1);
        sB += __shfl_xor_sync(0xffffffffu, sB, 2);
        lA = lA * fA + sA;
        lB = lB * fB + sB;
#pragma unroll
        for (int n = 0; n < 8; n++)
#pragma unroll
            for (int kk = 0; kk < 4; kk++) {
                const __half* vr = vt + (n * 8 + g) * APF_LD + kk * 16 + 2 * c;
                const uint32_t b0 = *(const uint32_t*)vr, b1 = *(const uint32_t*)(vr + 8);
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                             : "+f"(o[n][0]), "+f"(o[n][1]), "+f"(o[n][2]), "+f"(o[n][3])
                             : "r"(pa[2 * kk][0]), "r"(pa[2 * kk][1]), "r"(pa[2 * kk + 1][0]), "r"(pa[2 * kk + 1][1]), "r"(b0), "r"(b1));
                if (PL)
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                                 : "+f"(o[n][0]), "+f"(o[n][1]), "+f"(o[n][2]), "+f"(o[n][3])
                                 : "r"(pl[2 * kk][0]), "r"(pl[2 * kk][1]), "r"(pl[2 * kk + 1][0]), "r"(pl[2 * kk + 1][1]), "r"(b0), "r"(b1));
            }
    }
    const float sk = sinks[kh * 8 + hh];
    const float rA = 1.f / (lA + expf(sk - mA)), rB = 1.f / (lB + expf(sk - mB));
    if (KV8) {
        // o[n][j]: row A at j = 0, 1, row B at j = 2, 3, dim 8 n + 2 c + (j & 1): the 64 dims of a (row, head) lie in the 4
        // lanes of a group. Hadamard stages: dim bit 0 (j), bits 1-2 (c: shuffles), bits 3-5 (n).
#pragma unroll
        for (int n = 0; n < 8; n++)
#pragma unroll
            for (int j = 0; j < 4; j += 2) {
                const float t = o[n][j];
                o[n][j] = t + o[n][j + 1];
                o[n][j + 1] = t - o[n][j + 1];
            }
#pragma unroll
        for (int b = 1; b < 4; b <<= 1)
#pragma unroll
            for (int n = 0; n < 8; n++)
#pragma unroll
                for (int j = 0; j < 4; j++) {
                    const float ot = __shfl_xor_sync(0xffffffffu, o[n][j], b);
                    o[n][j] = (c & b) ? ot - o[n][j] : o[n][j] + ot;
                }
#pragma unroll
        for (int b = 1; b < 8; b <<= 1)
#pragma unroll
            for (int n = 0; n < 8; n++)
                if (!(n & b))
#pragma unroll
                    for (int j = 0; j < 4; j++) {
                        const float t = o[n][j];
                        o[n][j] = t + o[n | b][j];
                        o[n | b][j] = t - o[n | b][j];
                    }
#pragma unroll
        for (int n = 0; n < 8; n++)
#pragma unroll
            for (int j = 0; j < 4; j++) o[n][j] *= 0.125f * kv8_sign(n * 8 + 2 * c + (j & 1));
    }
#pragma unroll
    for (int n = 0; n < 8; n++) {
        const size_t col = (size_t)(kh * 8 + hh) * 64 + n * 8 + 2 * c;
        if (tA < T) *(bf162*)(AO + (size_t)tA * ncol + col) = __floats2bfloat162_rn(o[n][0] * rA, o[n][1] * rA);
        if (tB < T) *(bf162*)(AO + (size_t)tB * ncol + col) = __floats2bfloat162_rn(o[n][2] * rB, o[n][3] * rB);
    }
}

// ---- third form of the fused expert GEMM (speed plan P3). The second form is byte-bound: a 128-row x 32-token
// block streams 184 KB of weights and 184 KB of activations for 23.6 MF (64 flops per byte). Here a block is
// 240 weight rows (15 warps) x 8 * NTT tokens (NTT = 8: 64 tokens, 4: 32 tokens for the experts' remainders):
// per flop half the weight bytes and (NTT = 8) a quarter of the activation re-reads.
//  - activations come in the fragment layout (X_perm: per 32 columns, 16 B slot t = the bf16 pairs at columns
//    2t, 8 + 2t, 16 + 2t, 24 + 2t; written by k_gather_rows_p / k_act_all_p), staged 160 columns at a time by plain
//    16 B copies; one 16 B shared load = the B fragments of two MMAs. 5 block steps per tile (odd): the 32 lanes
//    spread evenly over the bank groups.
//  - the block's scale offset (0..15) is folded into the fragment table (16 x 256 bf16 pairs, exact powers of two),
//    the row's base scale multiplies the finished sums: no per-block partial accumulators.
constexpr int G3_KTB = 5;
__device__ __forceinline__ int g3_perm(int c) { return (c & ~31) | (((c >> 1) & 3) << 3) | (((c >> 3) & 3) << 1) | (c & 1); }
template <int NTT>
__global__ void __launch_bounds__(480) k_mx_gemm3(const uint8_t* cod, int NRW, int KB, const bf16* X, int ldx,
                                                  const int* units, float* out, int ldo, const int* ncnt = nullptr) {
    __shared__ uint32_t tab[16 * 256];
    __shared__ __align__(16) uint4 xs[NTT * 8 * G3_KTB * 4];
    if (ncnt && (int)blockIdx.y >= *ncnt) return;   // device routing: the grid is the upper bound of the unit count
    for (int i = threadIdx.x; i < 4096; i += 480) {
        const float2 v = fp4pair((i & 15u) | (((i >> 4) & 15u) << 16));
        const float f = 16384.f * (float)(1 << (i >> 8));
        const bf162 r = __floats2bfloat162_rn(v.x * f, v.y * f);
        tab[i] = *(const uint32_t*)&r;
    }
    const int u = blockIdx.y;
    const int e = units[3 * u], o = units[3 * u + 1], cnt = units[3 * u + 2];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const int row0 = blockIdx.x * 240 + warp * 16;
    const bool live = row0 < NRW;
    const size_t rA = (size_t)e * NRW + (live ? row0 : 0) + g, rB = rA + 8;
    const uint8_t *cA = cod + rA * mx_rs(KB), *cB = cod + rB * mx_rs(KB);
    const uint8_t *tA = cA + KB * 16, *tB = cB + KB * 16;
    float c[NTT][4];
#pragma unroll
    for (int i = 0; i < NTT; i++)
#pragma unroll
        for (int j = 0; j < 4; j++) c[i][j] = 0.f;
    for (int ks = 0; ks * G3_KTB < KB; ks++) {
        __syncthreads();   // the table (first pass) / every warp is done with the previous tile
        // the tile's weight blocks and scale nibbles go to registers before the staging, all loads issued together:
        // one DRAM round trip per tile instead of one per block step; the next tile's lines are asked for now
        const int nb = min(G3_KTB, KB - ks * G3_KTB), bb0 = ks * G3_KTB;
        uint4 wa[G3_KTB], wb[G3_KTB];
        uint32_t na = 0, nbv = 0;
        if (live) {
#pragma unroll
            for (int b = 0; b < G3_KTB; b++)
                if (b < nb) {
                    wa[b] = *(const uint4*)(cA + (bb0 + b) * 16);
                    wb[b] = *(const uint4*)(cB + (bb0 + b) * 16);
                }
#pragma unroll
            for (int j = 0; j < (G3_KTB + 2) / 2; j++)
                if ((bb0 >> 1) + j <= (KB - 1) >> 1) {
                    na |= (uint32_t)tA[(bb0 >> 1) + j] << (8 * j);
                    nbv |= (uint32_t)tB[(bb0 >> 1) + j] << (8 * j);
                }
            if (bb0 + G3_KTB < KB) {
                asm volatile("prefetch.global.L1 [%0];" ::"l"(cA + (bb0 + G3_KTB) * 16));
                asm volatile("prefetch.global.L1 [%0];" ::"l"(cA + (bb0 + 2 * G3_KTB) * 16 - 16));
                asm volatile("prefetch.global.L1 [%0];" ::"l"(cB + (bb0 + G3_KTB) * 16));
                asm volatile("prefetch.global.L1 [%0];" ::"l"(cB + (bb0 + 2 * G3_KTB) * 16 - 16));
            }
        }
        for (int q = threadIdx.x; q < NTT * 8 * G3_KTB * 4; q += 480) {
            const int tok = q / (G3_KTB * 4), u16 = ks * G3_KTB * 4 + q % (G3_KTB * 4);
            xs[q] = tok < cnt && u16 < KB * 4 ? *(const uint4*)(X + (size_t)(o + tok) * ldx + u16 * 8) : make_uint4(0, 0, 0, 0);
        }
        __syncthreads();
        if (!live) continue;
#pragma unroll
        for (int b = 0; b < G3_KTB; b++) {
            if (b >= nb) break;
            const int nsh = 4 * (bb0 + b) - 8 * (bb0 >> 1);   // nibble of block bb0 + b inside na / nbv
            const uint32_t* ta = tab + ((na >> nsh) & 15u) * 256;
            const uint32_t* tb = tab + ((nbv >> nsh) & 15u) * 256;
            const int sh = 8 * t;
            const uint32_t f00 = ta[(wa[b].x >> sh) & 255u], f01 = tb[(wb[b].x >> sh) & 255u], f02 = ta[(wa[b].y >> sh) & 255u], f03 = tb[(wb[b].y >> sh) & 255u];
            const uint32_t f10 = ta[(wa[b].z >> sh) & 255u], f11 = tb[(wb[b].z >> sh) & 255u], f12 = ta[(wa[b].w >> sh) & 255u], f13 = tb[(wb[b].w >> sh) & 255u];
#pragma unroll
            for (int i = 0; i < NTT; i++) {
                const uint4 xv = xs[((i * 8 + g) * G3_KTB + b) * 4 + t];
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                             : "+f"(c[i][0]), "+f"(c[i][1]), "+f"(c[i][2]), "+f"(c[i][3])
                             : "r"(f00), "r"(f01), "r"(f02), "r"(f03), "r"(xv.x), "r"(xv.y));
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                             : "+f"(c[i][0]), "+f"(c[i][1]), "+f"(c[i][2]), "+f"(c[i][3])
                             : "r"(f10), "r"(f11), "r"(f12), "r"(f13), "r"(xv.z), "r"(xv.w));
            }
        }
    }
    if (!live) return;
    const float sa = __int_as_float((int)tA[(KB + 1) >> 1] << 23), sb = __int_as_float((int)tB[(KB + 1) >> 1] << 23);
#pragma unroll
    for (int i = 0; i < NTT; i++)
#pragma unroll
        for (int j = 0; j < 2; j++) {
            const int tok = i * 8 + 2 * t + j;
            if (tok < cnt) {
                float* po = out + (size_t)(o + tok) * ldo + row0 + g;
                po[0] = c[i][j] * sa;
                po[8] = c[i][2 + j] * sb;
            }
        }
}
// ---- fourth form (round 5). ncu of the third form: one block per SM, no eligible warp 82-88 % of the time, DRAM at
// 40 % (64-token units) / 66 % (8-token units) of its rate: the warps sit in blocking weight loads (units whose
// weights are in L2 already run at 114 TF, cold ones at 38). Here nothing of a tile is loaded by a blocking load:
//  - the tile's weight codes (240 rows x KTB blocks of 16 B) and activations (8 * NTT tokens x KTB * 4 slots of 16 B)
//    are copied into shared memory by cp.async one tile ahead (two stages each), while the tile in front computes;
//    one barrier per tile (all copies of the tile have landed and every warp has left the stage that is refilled next);
//  - the rows' scale tails (offset nibbles + base scale) are staged once per block;
//  - the fragments and the order of the MMAs per accumulator are those of the third form (K blocks ascending, two
//    MMAs per block): the sums are bit-identical for every KTB. The table has the 256 byte values at scale offset 2
//    (1 KB instead of 16 KB); the block's offset k - 2 is added to the exponent fields of the non-zero halves (the
//    e2m1 values x 4 are 2 .. 24: exponent field 128 .. 131, so bit 14 of a half tells zero from non-zero, and
//    -2 .. +13 never leaves the field or borrows from the other half).
// Shared bytes (g4_smem): table 1 KB + tails + 2 weight stages + 2 activation stages <= 99 KB, one block per SM.
// The activation rows are padded to 4 mod 8 slots of 16 B when KTB is even (bank spread of the 32 lanes: lane groups
// g and g + 1 must land 16 words apart; odd KTB has it by itself).
__host__ __device__ constexpr int g4_xst(int KTB) { return KTB * 4 + ((KTB & 1) ? 0 : 4); }   // slots per token row: 4 mod 8 (review r5 L2)
__host__ __device__ constexpr int g4_ts(int KB) { return (((KB + 1) >> 1) + 1 + 15) >> 4; }
__host__ __device__ constexpr int g4_smem(int NTT, int KTB, int KB) { return 16 * (64 + 240 * g4_ts(KB) + 2 * 240 * KTB + 2 * NTT * 8 * g4_xst(KTB)); }
// Round 14: (a) the MMAs of the 8-token groups past the unit's count are skipped (their accumulators were never
// stored); (b) NTT = 16: units of up to 128 tokens (64 accumulators per thread, <16,3>: 88.6 KB shared), so an expert's
// rows need one weight stream per 128 rows instead of one per 64 + one for a remainder (the kernel is bound by the
// weight bytes: ~3.5 us per unit and slab-set, the tokens add at the tensor rate); (c) EP = 1 (gate_up): the
// epilogue adds the bias, applies SwiGLU and the routing weight and stores the bf16 activation in the fragment layout
// (what k_act_all_p made from the fp32 sums: same operations in the same order, the scale product is its own
// rounding step); `out` is then the bf16 A buffer with ldo = NRW / 2, `ebias` the layer's [expert][NRW] bias and
// `ew` the routing weight per sorted row.
template <int NTT, int KTB, int EP = 0>
__global__ void __launch_bounds__(480) k_mx_gemm4(const uint8_t* cod, int NRW, int KB, const bf16* X, int ldx,
                                                  const int* units, float* out, int ldo, const int* ncnt = nullptr,
                                                  const bf16* ebias = nullptr, const float* ew = nullptr) {
    extern __shared__ __align__(16) uint4 g4sm[];
    constexpr int XST = g4_xst(KTB), XN = NTT * 8 * XST, WN = 240 * KTB;
    if (ncnt && (int)blockIdx.y >= *ncnt) return;   // device routing: the grid is the upper bound of the unit count
    const int TS = g4_ts(KB);
    uint32_t* tab = (uint32_t*)g4sm;
    uint4* tl = g4sm + 64;
    uint4* ws = tl + 240 * TS;
    uint4* xs = ws + 2 * WN;
    const int u = blockIdx.y;
    const int e = units[3 * u], o = units[3 * u + 1], cnt = units[3 * u + 2];
    const int nt = min(NTT, (cnt + 7) >> 3);   // live 8-token groups
    const int rowb = blockIdx.x * 240;
    const uint8_t* cb = cod + (size_t)e * NRW * mx_rs(KB);
    const size_t rs = mx_rs(KB);
    if (threadIdx.x < 256) {
        const float2 v = fp4pair((threadIdx.x & 15u) | ((threadIdx.x >> 4) << 16));
        const bf162 r = __floats2bfloat162_rn(v.x * 65536.f, v.y * 65536.f);
        tab[threadIdx.x] = *(const uint32_t*)&r;
    }
    for (int q = threadIdx.x; q < 240 * TS; q += 480) {
        const int row = min(rowb + q / TS, NRW - 1);
        cp16(tl + q, cb + row * rs + KB * 16 + (q % TS) * 16);
    }
    // token rows past the unit's count stay zero in both stages (their accumulators are never stored)
    for (int q = threadIdx.x; q < 2 * XN; q += 480)
        if ((q % XN) / XST >= cnt || (q % XN) % XST >= KTB * 4) xs[q] = make_uint4(0, 0, 0, 0);
    const int NT = (KB + KTB - 1) / KTB;
    auto issue = [&](int ks) {
        const int st = ks & 1, bb0 = ks * KTB;
        uint4* wd = ws + st * WN;
        for (int q = threadIdx.x; q < WN; q += 480) {
            const int b = q % KTB;
            if (bb0 + b < KB) cp16(wd + q, cb + min(rowb + q / KTB, NRW - 1) * rs + (bb0 + b) * 16);
        }
        uint4* xd = xs + st * XN;
        for (int q = threadIdx.x; q < NTT * 8 * KTB * 4; q += 480) {
            const int tok = q / (KTB * 4), j = q % (KTB * 4);
            if (tok < cnt && bb0 * 4 + j < KB * 4) cp16(xd + tok * XST + j, X + (size_t)(o + tok) * ldx + (bb0 * 4 + j) * 8);
        }
        asm volatile("cp.async.commit_group;" ::: "memory");
    };
    issue(0);
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const int rl = warp * 16 + g;   // rows rl and rl + 8 of the block
    const bool live = rowb + warp * 16 < NRW;
    const uint32_t *tla = (const uint32_t*)(tl + rl * TS), *tlb = (const uint32_t*)(tl + (rl + 8) * TS);
    float c[NTT][4];
#pragma unroll
    for (int i = 0; i < NTT; i++)
#pragma unroll
        for (int j = 0; j < 4; j++) c[i][j] = 0.f;
    for (int ks = 0; ks < NT; ks++) {
        asm volatile("cp.async.wait_all;" ::: "memory");
        __syncthreads();   // tile ks is in its stage (and the table, the tails); every warp has left the other stage
        if (ks + 1 < NT) issue(ks + 1);
        if (!live) continue;
        const int bb0 = ks * KTB, nb = min(KTB, KB - bb0);
        const uint4 *wA = ws + (ks & 1) * WN + rl * KTB, *wB = wA + 8 * KTB;
        const uint4* xt = xs + (ks & 1) * XN;
#pragma unroll
        for (int b = 0; b < KTB; b++) {
            if (b >= nb) break;
            const int bb = bb0 + b;
            const uint32_t ka = (tla[bb >> 3] >> (4 * (bb & 7))) & 15u, kb = (tlb[bb >> 3] >> (4 * (bb & 7))) & 15u;
            const uint4 wa = wA[b], wb = wB[b];
            const int sh = 8 * t;
#define G4F(w, k) ({ const uint32_t v_ = tab[((w) >> sh) & 255u]; v_ + ((k) - 2u) * ((v_ >> 7) & 0x00800080u); })
            const uint32_t f00 = G4F(wa.x, ka), f01 = G4F(wb.x, kb), f02 = G4F(wa.y, ka), f03 = G4F(wb.y, kb);
            const uint32_t f10 = G4F(wa.z, ka), f11 = G4F(wb.z, kb), f12 = G4F(wa.w, ka), f13 = G4F(wb.w, kb);
#undef G4F
#define G4MMA(i) { \
                const uint4 xv = xt[((i) * 8 + g) * XST + b * 4 + t]; \
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};" \
                             : "+f"(c[i][0]), "+f"(c[i][1]), "+f"(c[i][2]), "+f"(c[i][3]) \
                             : "r"(f00), "r"(f01), "r"(f02), "r"(f03), "r"(xv.x), "r"(xv.y)); \
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};" \
                             : "+f"(c[i][0]), "+f"(c[i][1]), "+f"(c[i][2]), "+f"(c[i][3]) \
                             : "r"(f10), "r"(f11), "r"(f12), "r"(f13), "r"(xv.z), "r"(xv.w)); }
            if (nt == NTT) {   // a full unit: no test per 8-token group (one branch per block step, uniform in the block)
#pragma unroll
                for (int i = 0; i < NTT; i++) G4MMA(i)
            } else {
#pragma unroll
                for (int i = 0; i < NTT; i++) {
                    if (i >= nt) break;
                    G4MMA(i)
                }
            }
#undef G4MMA
        }
    }
    if (!live) return;
    const int sbi = (KB + 1) >> 1;   // the base scale byte behind the nibbles
    const float sa = __int_as_float((int)((tla[sbi >> 2] >> (8 * (sbi & 3))) & 255u) << 23);
    const float sb = __int_as_float((int)((tlb[sbi >> 2] >> (8 * (sbi & 3))) & 255u) << 23);
    if (EP) {   // rows 2 j (gate) and 2 j + 1 (up) sit in the lane groups g and g + 1 of a warp: the even group stores
        const bf16* bb = ebias + (size_t)e * NRW + rowb + rl;
        const float bA = bf2f(bb[0]), bB = bf2f(bb[8]);
        const int ja = g3_perm((rowb + rl) >> 1), jb = g3_perm(((rowb + rl) >> 1) + 4);
#pragma unroll
        for (int i = 0; i < NTT; i++) {
            if (i >= nt) break;
#pragma unroll
            for (int j = 0; j < 2; j++) {
                const int tok = i * 8 + 2 * t + j;
                const float va = __fadd_rn(__fmul_rn(c[i][j], sa), bA), vb = __fadd_rn(__fmul_rn(c[i][2 + j], sb), bB);
                const float ua = __shfl_xor_sync(0xffffffffu, va, 4), ub = __shfl_xor_sync(0xffffffffu, vb, 4);
                if (!(g & 1) && tok < cnt) {
                    const float wt = ew[o + tok];
                    bf16* pa = (bf16*)out + (size_t)(o + tok) * ldo;
                    pa[ja] = __float2bfloat16_rn(swiglu(va, ua) * wt);
                    pa[jb] = __float2bfloat16_rn(swiglu(vb, ub) * wt);
                }
            }
        }
        return;
    }
#pragma unroll
    for (int i = 0; i < NTT; i++)
#pragma unroll
        for (int j = 0; j < 2; j++) {
            const int tok = i * 8 + 2 * t + j;
            if (tok < cnt) {
                float* po = out + (size_t)(o + tok) * ldo + rowb + rl;
                po[0] = c[i][j] * sa;
                po[8] = c[i][2 + j] * sb;
            }
        }
}
// k_gather_rows into the fragment layout of k_mx_gemm3 (16 B units: 4 bf16 pairs from 4 places of the source row)
__global__ void k_gather_rows_p(const bf16* X, const int* rows, int n, bf16* out) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)n * 360) return;
    const size_t r = i / 360;
    const int u = (int)(i % 360), b = u >> 2, t = u & 3;
    const uint32_t* src = (const uint32_t*)(X + (size_t)rows[r] * 2880 + b * 32 + 2 * t);
    ((uint4*)out)[i] = make_uint4(src[0], src[4], src[8], src[12]);
}
// k_act_all into the fragment layout of k_mx_gemm3
__global__ void k_act_all_p(const float* GU, const bf16* bias, const int* ex, const float* w, int n, int NR, bf16* A) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int half = NR / 2;
    if (i >= (size_t)n * half) return;
    const size_t r = i / half;
    const int j = (int)(i % half);
    const float* g = GU + r * NR;
    const bf16* b = bias + (size_t)ex[r] * NR;
    A[r * half + g3_perm(j)] = __float2bfloat16_rn(swiglu(g[2 * j] + bf2f(b[2 * j]), g[2 * j + 1] + bf2f(b[2 * j + 1])) * w[r]);
}

// ---- prefill routing on the device (round 4): the sorted row lists and the units of k_mx_gemm3 built from the
// router's top-4 ids by two kernels on every card, identical to the host bucketing (stable by (row, k) inside an
// expert), so that no host thread waits inside the layer loop. Route record of a lane (ints):
//   [nc[4]: units per class][12 pad][prow 4T][pw 4T (float)][pexp 4T][pinv 4T][units: class c at 3 * ub[c]]
// k_route_plan: one block; histogram of the 4T ids, then thread e (< 128) owns expert e: offset, 64-token units
// (the last one when it has more than 32 rows), remainder unit of <= 32 rows in class 1 (17..32), 2 (9..16), 3 (1..8).
// Every card routes for itself (same normed rows, same router weights, same kernels); `sig` accumulates a signature
// of the ids and of the weights' bits over the pass, compared between the cards on the host at the end of the chunk.
__global__ void __launch_bounds__(1024) k_route_plan(const int* ids, const float* rw, int n4, int* nc, int* off, int* units, int ub1, int ub2, int ub3,
                                                     unsigned* sig, int plan = 4, int ub4 = 0) {
    // plan 4: 64-row units in class 0, a remainder of <= 32 rows in class 1 / 2 / 3 (a remainder above 32 is a class-0
    // unit). plan 5 (round 14): 128-row units in class 4, a remainder of <= 64 rows in class 0 (33..64) / 1 / 2 / 3 (a
    // remainder above 64 is a class-4 unit). cl = -1: no remainder unit.
    __shared__ int hist[128], u0[128], cl[128];
    __shared__ unsigned hs;
    const int t = threadIdx.x;
    const int US = plan == 5 ? 128 : 64, big = plan == 5 ? 4 : 0;
    if (t < 128) hist[t] = 0;
    if (t == 0) hs = 0;
    __syncthreads();
    unsigned h = 0;
    for (int i = t; i < n4; i += 1024) {
        atomicAdd(&hist[ids[i] & 127], 1);
        h += ((unsigned)ids[i] + 1u) * ((unsigned)i * 2654435761u + 40503u) + (unsigned)__float_as_int(rw[i]) * ((unsigned)i * 40503u + 2246822519u);
    }
    atomicAdd(&hs, h);
    __syncthreads();
    if (t == 0) *sig = (*sig) * 31u + hs;
    if (t < 128) {
        const int n = hist[t], rem = n & (US - 1);
        u0[t] = n / US + (rem > US / 2);
        cl[t] = rem == 0 || rem > US / 2 ? -1 : rem > 32 ? 0 : rem > 16 ? 1 : rem > 8 ? 2 : 3;
    }
    __syncthreads();
    if (t < 128) {
        int o = 0, p0 = 0, pc[4] = {0, 0, 0, 0};
        for (int j = 0; j < t; j++) {
            o += hist[j];
            p0 += u0[j];
            if (cl[j] >= 0) pc[cl[j]]++;
        }
        const int n = hist[t], c = cl[t];
        const int ubs[5] = {0, ub1, ub2, ub3, ub4};
        off[t] = o;
        for (int i = 0; i < u0[t]; i++) {
            int* u = units + 3 * (ubs[big] + p0 + i);
            u[0] = t;
            u[1] = o + US * i;
            u[2] = min(US, n - US * i);
        }
        if (c >= 0) {
            const int rem = n & (US - 1);
            int* u = units + 3 * (ubs[c] + pc[c]);
            u[0] = t;
            u[1] = o + n - rem;
            u[2] = rem;
        }
        if (t == 127) {
            off[128] = o + n;
            if (c >= 0) pc[c]++;
            for (int k = 0; k < 5; k++) nc[k] = k == big ? p0 + u0[t] : k < 4 ? pc[k] : 0;
        }
    }
}
// k_route_fill: block = expert. The 4T (row, k) entries are cut into 256 segments (one per thread); count the
// expert's entries per segment, prefix over the segments, then place them in entry order.
__global__ void __launch_bounds__(256) k_route_fill(const int* ids, const float* rw, int n4, const int* off, int* prow, float* pw,
                                                    int* pexp, int* pinv) {
    __shared__ int cs[256];
    const int e = blockIdx.x, t = threadIdx.x;
    const int o0 = off[e];
    if (off[e + 1] == o0) return;
    const int seg = (n4 + 255) / 256, a = t * seg, b = min(n4, a + seg);
    int c = 0;
    for (int i = a; i < b; i++) c += ids[i] == e;
    cs[t] = c;
    __syncthreads();
    if (!c) return;
    int o = o0;
    for (int j = 0; j < t; j++) o += cs[j];
    for (int i = a; i < b; i++)
        if (ids[i] == e) {
            prow[o] = i >> 2;
            pw[o] = rw[i];
            pexp[o] = e;
            pinv[i] = o;
            o++;
        }
}

}  // namespace gptoss
