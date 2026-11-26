// [ink-tpf-ppush] / [ink-tpf-pred] TP plain step: the all-reduce + post-collective conv inside the producer.
// Included from ink.cu before the residual kernels (namespace ink; needs H, TP_MAX).
//
// The kernel that computes a rank's partial row (wo: k_g12tcR, down: k_dn4p / k_dn8pp) has one thread per output
// column at its end. That thread
//   1. stores its value as one 8 B word {fp32 bits, tag} into each peer's receive words [TP8_SLOTS][sender][H]
//      (tag = step x 256 + collective + 1, as k_tpar; the value and its tag share one 8 B store);
//   2. waits for the peers' words of the same column and adds the N partials in rank order starting from 0.f
//      (bitwise the k_tpar sum);
//   3. applies the post-collective conv + residual (the conv1 expression; inputs loaded at kernel entry, state
//      shifted in place: one thread per column, so no ticket) and writes the output row.
// No collective kernel is launched. Every column is received once, by the thread that already holds its partial.
// Used only when every rank runs the producer (else a rank would push nothing) on the push transport; the direct
// transport (ranks on one device) and the row steps keep k_tpar. QE_TP_PP=0: k_tpar in plain steps (review vpkg L4: row
// steps of <= INK_TPCW rows use the column-major producer push of [ink-vp-cpush]; INK_TPCW=0 turns that off).
// Lost forms: receive in the next residual kernel ([ink-tpf-arfuse2] ❌, bench/patches/ink_tpf_arfuse2.patch);
// push + a receive-only collective kernel (5.76 ms vs 5.48 at TP2 512).
constexpr int TP8_SLOTS = 4;   // receive-word slots (collective index mod 4), as TP_SLOTS

struct TpPush {
    uint2* dst[TP_MAX - 1];   // each peer's receive words of this rank for the collective's slot [H]
    const unsigned* step;     // this rank's step counter (device)
    int np, idx;              // peers (0: no push), collective index
    const uint2* rd8;         // this rank's receive words of the slot [n][H]
    const unsigned* ab;       // the context's abort word
    unsigned* chk;            // QE_TP_CHK hashes or null
    float* out;
    float* cst;
    const float* cw;
    const float* add;
    int n, rank;
};
__device__ __forceinline__ unsigned tp_push_tag(const TpPush& p) { return p.np ? *p.step * 256u + (unsigned)p.idx + 1u : 0u; }
__device__ __forceinline__ void tp_push1(const TpPush& p, const unsigned tag, const int c, const float v) {
    for (int g = 0; g < p.np; g++)
        asm volatile("st.volatile.global.v2.u32 [%0], {%1, %2};" ::"l"(p.dst[g] + c), "r"(__float_as_uint(v)), "r"(tag) : "memory");
}

__device__ __forceinline__ unsigned long long tpf_timer() {
    unsigned long long t;
    asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t));
    return t;
}
// column c: this rank's partial v pushed already; s = the N partials in rank order from 0.f (bitwise
// the k_tpar sum). false: the context aborted.
__device__ __forceinline__ bool tp_reduce1(const TpPush& p, const unsigned tag, const int c, const float v, float& s) {
    s = 0.f;
    for (int g = 0; g < p.n; g++) {
        if (g == p.rank) {
            s += v;
            continue;
        }
        const uint2* rp = p.rd8 + (size_t)g * H + c;
        uint2 x;
        unsigned it = 0;
        unsigned long long t0 = 0;
        for (;;) {
            asm volatile("ld.volatile.global.v2.u32 {%0, %1}, [%2];" : "=r"(x.x), "=r"(x.y) : "l"(rp) : "memory");
            if (x.y == tag) break;
            if ((++it & 4095u) == 0) {
                if (*(volatile const unsigned*)p.ab) return false;
                if (!t0) t0 = tpf_timer();
                else if (tpf_timer() - t0 > 20000000000ull) __trap();   // last resort: a peer whose host thread is gone (20 s)
            }
        }
        s += __uint_as_float(x.x);
    }
    if (p.chk) atomicXor(p.chk + p.idx, (__float_as_uint(s) * (2u * (unsigned)c + 1u)) ^ (unsigned)c);
    return true;
}
// [ink-tpf-pred] the conv inputs of column c loaded at kernel entry by the thread that will reduce it (in flight
// under the producer's own work), and the conv1 epilogue on them: the conv1 expression, state shift and output
struct TpConvPre {
    float x1, x2, x3, w0, w1, w2, w3, ad;
};
__device__ __forceinline__ void tp_conv_pre(const TpPush& p, const int c, TpConvPre& q) {
    q.x1 = p.cst[2 * H + c];
    q.x2 = p.cst[H + c];
    q.x3 = p.cst[c];
    q.w0 = p.cw[c * 4 + 0];
    q.w1 = p.cw[c * 4 + 1];
    q.w2 = p.cw[c * 4 + 2];
    q.w3 = p.cw[c * 4 + 3];
    q.ad = p.add ? p.add[c] : 0.f;
}
// The operation sequence is pinned to the one the compiler emits for conv1 inside k_tpar (found by trying the fused
// multiply-add placements against the k_tpar path, every collective's output row bitwise): with the inputs in registers
// it would otherwise pick another contraction of the same source expression.
__device__ __forceinline__ void tp_conv_fin(const TpPush& p, const int c, const float x0, const TpConvPre& q) {
    const float o = __fmaf_rn(q.w0, q.x3, __fmaf_rn(q.w1, q.x2, __fmaf_rn(q.w3, x0, __fmul_rn(q.w2, q.x1))));
    p.cst[c] = q.x2;
    p.cst[H + c] = q.x1;
    p.cst[2 * H + c] = x0;
    p.out[c] = __fmaf_rn(__fadd_rn(o, x0), 1.f / sqrtf(5.f), q.ad);
}
