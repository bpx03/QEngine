#pragma once
#include "glm.h"

namespace glm {
void kernels_init();  // per device
void embed(const int* tok, const bf16* emb, float* R, int T, cudaStream_t st);
void mhc_pre(const float* R, const float* mixraw, const float* base, const float* scale, const float* ln, float* post,
             float* comb, float* xf, bf16* xb, int T, cudaStream_t st, bool has_ss = false, const float* ssp = nullptr,
             const float* mixp = nullptr, int nbp = H / 256);
void merge(float* R, const float* post, const float* comb, const float* a, int T, cudaStream_t st, float* ssp = nullptr);
void boundary(float* R, float* post, float* comb, const float* a, const float* fn, const float* base, const float* scale,
              const float* ln, float* xf, bf16* xb, int T, float* scratch, unsigned* bar, cudaStream_t st);
void merge_mix(float* R, const float* post, const float* comb, const float* a, const float* fn, float* mixp, int T,
               cudaStream_t st);
// prefill: merge (a != nullptr) + the next block's mixes and sum of squares as partials mixp[t][parts][25] (one pass over R)
int merge_mix_pf_parts();
void mhc_pre_pf(const float* R, const float* base, const float* scale, const float* ln, float* post, float* comb,
                float* xf, bf16* xb, int T, const float* mixp, int nbp, cudaStream_t st);
void merge_mix_pf(float* R, const float* post, const float* comb, const float* a, const float* fn, float* mixp, int T,
                  cudaStream_t st);
void final_norm(const float* R, const float* w, bf16* xb, float* hf, int T, cudaStream_t st);
void kda_prep(const float* proj, int ldp, int nh, const float* convw, float* hist, const float* gf, const float* A,
              const float* dt, float* q, float* k, float* v, float* g, float* beta, int T, cudaStream_t st,
              float* conv_rows = nullptr, int r0 = 0, const float* prev_rows = nullptr);
void kda_prep_f(const float* proj, int ldp, int nh, const float* convw, const float* hist, const float* fbT, const float* A,
                const float* dt, float* q, float* k, float* v, float* g, float* beta, int T, cudaStream_t st, float* rows_out,
                int r0, const float* prev);
void kda_out_f(const float* o, const float* proj, int ldp, const float* gbT, const float* w, bf16* out, int T, int nh,
               cudaStream_t st);
void transpose(const float* a, float* b, int n, int m, cudaStream_t st);
void kda_gates(const float* fbT, const float* gbT, const float* proj, int ldp, int HK, int T, float* gf, float* gate,
               cudaStream_t st);
// round 10 (P-5): chunked prefill delta rule (2 phases), scratch kda_chunk_floats(T, nh)
size_t kda_chunk_floats(int T, int nh);
extern int kda_chunk_dbg;
void kda_chunk_rec(const float* q, const float* k, const float* v, const float* g, const float* beta, float* S, float* o,
                   int T, int nh, float* scratch, cudaStream_t st);
void kda_rec_pk_init();
void kda_rec(const float* q, const float* k, const float* v, const float* g, const float* beta, float* S, float* o,
             int T, int nh, cudaStream_t st, float* snap = nullptr, const int* pend = nullptr);
void kda_commit(float* const* S, float* const* snap, float* const* hist, float* const* rows, int nl, int nst, int C,
                const int* acc, cudaStream_t st, bool ptr = false);
void kda_materialize(float* const* S, float* const* snap, int nl, int nst, int* acc, cudaStream_t st);
void kda_out(const float* o, const float* gate, const float* w, bf16* out, int T, int nh, cudaStream_t st);
void mla_set_strides(int kst, int pst);  // round 26: kc / pk row strides of the current card (elements)
void kernel_env_init();  // review r27 L4: GLM_SEL2 / GLM_IDX_DEC read once on the host at load
void idx_dec_init();  // round 26: shared-memory attribute of the row-shared decode indexer (per card at load)
extern int g_sel2;  // round 27: 1 = k_select2 for decode steps (GLM_SEL2), 0 = k_select
extern int g_idx_dec;  // round 26: 1 = row-shared decode indexer (GLM_IDX_DEC), 0 = k_idx_score per row
void mla_prep(const float* proj, const float* qaln, const float* kvln, const float* ikw, const float* ikb, bf16* qr,
              bf16* kc, float* ki, float* gsc, float* wi, int n0, int T, cudaStream_t st, const int* dp, int ring);
void pool_keys(const float* ki, const float* gsc, const float* ape, float* pk, int p0, int p1, cudaStream_t st, int ring);
void idx_score(const float* qi, int ldq, const float* wi, const float* pk, float* sc, int ld, int n0, int T,
               cudaStream_t st, const int* dp = nullptr);
void pool_keys_dec(const float* ki, const float* gsc, const float* ape, float* pk, const int* dp, int T, cudaStream_t st,
                   int ring);
void mhc_mix(const float* R, const float* fn, float* mixraw, int T, cudaStream_t st);
void select_tokens(const float* sc, int ld, int n0, int* sel, int* nsel, int T, cudaStream_t st, const int* dp = nullptr,
                   int* psel = nullptr);
void sel_expand(const int* psel, const int* pnsel, int n0, int* sel, int* nsel, int T, cudaStream_t st);
size_t mla_split_scratch(int nh, int T);
void mla_attn_tc_dec(const float* qlat, const bf16* kc, const int* sel, const int* nsel, float* ctx, int nh, int n0,
                     int T, float scale, cudaStream_t st, const int* dp, float* scr, int nsp);
void mla_uk_bf16(const bf16* wuk, const float* qq, int ldq, float* qlat, int nh, int T, cudaStream_t st);
void mla_uv_bf16(const bf16* wuv, const float* ctx, float* ov, int nh, int T, cudaStream_t st);
void mla_attn_split(const float* qlat, const bf16* kc, const int* sel, const int* nsel, float* ctx, int nh, int n0, int T,
                    float scale, cudaStream_t st, const int* dp, float* scr);
void mla_attn(const float* qlat, const bf16* kc, const int* sel, const int* nsel, float* ctx, int nh, int n0, int T,
              float scale, cudaStream_t st, const int* dp = nullptr);
void route(const float* logits, const float* bias, int* ids, float* wts, int T, cudaStream_t st);
void set_tau_all(float v);  // round 17 gate: residency-independent skip threshold (tau * RSF), 0 = off
// decode rows (T <= 4): router logits lg[T][288] = W x (fp32), part = scratch [8][T][288]
void router_dec(const float* W, const float* x, int T, float* part, float* lg, cudaStream_t st);
void glu(const float* gu, bf16* out, int n, int T, cudaStream_t st, int ldgu = -1);
void deq_expert(const XPtr& x, int n, bf16* Wgu, bf16* Wd, cudaStream_t st);
void deq_expert16(const XPtr& x, int n, __half* Wgu, __half* Wd, cudaStream_t st);
void gather_rows16(const bf16* x, const int* rows, __half* out, int n, cudaStream_t st);
void glu16(const __half* gu, __half* out, int n, int T, cudaStream_t st);
void scatter_add16(float* out, const __half* y, const int* rows, const float* w, int n, cudaStream_t st);
void gather_rows(const bf16* x, const int* rows, bf16* out, int n, cudaStream_t st);
void scatter_add(float* out, const float* y, const int* rows, const float* w, int n, cudaStream_t st);
// njp (device, optional): the job count at run time (njobs = the launch bound)
// scratch: gu [jobs][KSPL][MAXR][2n], hb [jobs][MAXR][n], yp [jobs][2][MAXR][H], njtmp (one int)
void expert_gemv(const XJob* jobs, int njobs, const float* x, float* gu, float* out, int n, int m, cudaStream_t st,
                 const int* njp, float* hb, float* yp, int* njtmp);
void argmax_rows(const float* x, int n, int rows, int* out, cudaStream_t st);
void row_se(const float* x, int n, int rows, const int* am, float* se, cudaStream_t st);
void idx_score_tc(const float* qi, int ldq, const float* wi, const float* pk, float* sc, int ld, int n0, int T,
                  cudaStream_t st);
void mla_attn_tc(const float* qlat, const bf16* kc, const int* sel, const int* nsel, float* ctx, int nh, int n0, int t0,
                 int T, float scale, cudaStream_t st);
void topk_rows(const float* x, int n, int rows, const float* invT, TopK* out, cudaStream_t st);
void quant8(const bf16* w, int N, int K, int8_t* q, float* s, cudaStream_t st);
void int8_to4(const int8_t* q8, const float* s8, int N, int K, uint8_t* q4, float* s4, cudaStream_t st);
void quant4f(const bf16* w, int N, int K, int8_t* q, float* s, cudaStream_t st);
// round 10: prefill dense projections on INT8 tensor cores (glm_gemm8.cu); np = activation passes (2: ~lossless)
void gemm8_quant(const bf16* x, int ldx, int K, int T, int np, int8_t* q, float* s, cudaStream_t st);
void gemm8(const int8_t* xq, const float* xs, int T, const W8& w, float* Y, int ldy, bool acc, int np, cudaStream_t st);
void dequant8(const int8_t* q, const float* s, int N, int K, bf16* w, cudaStream_t st);
void dequant4(const uint8_t* q, const float* s, int N, int K, bf16* w, cudaStream_t st);
void gemv4q(const uint8_t* q, const float* s, int N, int K, const int8_t* xq, const float* xs, float* y, int ldy, int T,
            bool acc, cudaStream_t st);
void dequant8h(const int8_t* q, const float* s, int N, int K, __half* w, cudaStream_t st);
void b2h_rows(const bf16* x, int ldx, int K, int T, __half* y, cudaStream_t st);
void acc16(float* Y, int ldy, const __half* y, int n, int T, bool acc, cudaStream_t st);
void gemv8(const int8_t* q, const float* s, int N, int K, const float* xf, const bf16* xb, int ldx, float* y, int ldy,
           int T, bool acc, cudaStream_t st);
void mtp_in(const int* tok, const bf16* emb, const float* h, const float* enw, const float* hnw, bf16* xcat, int T,
            cudaStream_t st);
void add_rms(float* x, const float* a, const float* w, float* xf, bf16* xb, int T, cudaStream_t st);
template <class XT>
void quant_x(const XT* x, int ldx, int K, int T, int8_t* xq, float* xs, cudaStream_t st);
// dp / nskip (round 27, D4 b): rows >= nskip are skipped while (*dp + T) / POOL <= TOPB (decode, T <= 4)
void gemv8q(const int8_t* q, const float* s, int N, int K, const int8_t* xq, const float* xs, float* y, int ldy, int T,
            bool acc, cudaStream_t st, const int* dp = nullptr, int nskip = 0);
void h2d_words2(int* d0, const int* hs0, int* d1, const int* hs1, int n, cudaStream_t st);  // round 27: pinned host -> device by a kernel
void untile(const char* src, char* dst, const SliceL& sl, int blocks, cudaStream_t st, const SliceL* ssl = nullptr);
// INT3 gate/up word (round 6): host tile-major planes (SliceL::gu3) -> one INT4 word (k8 row r, column c; nibble
// 2b + p = row 8r + 2b + p, value code + 4 so the slot keeps the INT4 zero point 8)
static __device__ __forceinline__ uint32_t gu3_word(uint32_t lo, uint32_t hi, int t) {
    uint32_t o = 0;
#pragma unroll
    for (int b = 0; b < 4; b++)
#pragma unroll
        for (int p = 0; p < 2; p++) {
            const int jl = 2 * (t & 1) + p, jh = 2 * t + p;
            const uint32_t v = ((lo >> (8 * b + 2 * jl)) & 3u) | (((hi >> (8 * b + jh)) & 1u) << 2);
            o |= (v + 4u) << (4 * (2 * b + p));
        }
    return o;
}
static __device__ __forceinline__ uint4 gu3_expand(const char* base, size_t r, size_t c) {
    const size_t kb = r >> 2;
    const int t = (int)(r & 3);
    const uint32_t* w = reinterpret_cast<const uint32_t*>(base) + (((c / 32) * 128 + kb) * 3) * 32 + c % 32;
    const uint4 lo = *reinterpret_cast<const uint4*>(w + (t < 2 ? 0 : 32));
    const uint4 hi = *reinterpret_cast<const uint4*>(w + 64);
    return make_uint4(gu3_word(lo.x, hi.x, t), gu3_word(lo.y, hi.y, t), gu3_word(lo.z, hi.z, t), gu3_word(lo.w, hi.w, t));
}


void mla_union(const int* sel, const int* nsel, int t0, int nb, int n0, int ctxmax, unsigned* bits, int* pre, int* keys,
               int* dU, cudaStream_t st);
void mla_union_all(const int* sel, const int* nsel, int T, int qb, int n0, int nw, int uw, unsigned* bits, int* pre,
                   int* dU, cudaStream_t st);
void mla_ukeys(const unsigned* bits, const int* pre, int nw, int* keys, cudaStream_t st);
void mla_union_rest(const bf16* kc, const int* sel, const int* nsel, const unsigned* bits, const int* pre, const int* keys,
                    int U, int t0, int nb, int n0, bf16* kb, unsigned char* mask, cudaStream_t st);
void usoftmax(const float* S, const unsigned char* mask, int U, int nh, int rows, bf16* P, cudaStream_t st);
void qscale(const float* q, bf16* qb, size_t n, float scale, cudaStream_t st);
void f2b(const float* x, bf16* y, size_t n, cudaStream_t st);
void f2b_rows(const float* x, int ldx, bf16* y, int n, int T, cudaStream_t st);
void b2f(const bf16* x, float* y, size_t n, cudaStream_t st);
void copy_cols(const float* x, int ld, float* y, int n, int T, cudaStream_t st);
void moe16_expert(const bf16* xb, const int* rows, const float* wts, int n, const XPtr& w, int ni, __half* hb, float* part,
                  cudaStream_t st);
void moe16_gu_batch(const bf16* xb, const int* rows, const GuJob* jobs, int njobs, int maxn, int ni, __half* hb,
                    cudaStream_t st);
void moe16_dn(const __half* hb, const int* rows, const float* wts, int n, const XPtr& w, int ni, float* part, cudaStream_t st);
void moe16t_expert(const __half* x16, const int* rows, const float* wts, int n, const XPtr& w, int ni, __half* hb, float* part,
                   bool gate_up, cudaStream_t st);
void moe16t_dn(const __half* hb, const int* rows, const float* wts, int n, const XPtr& w, int ni, float* part, cudaStream_t st);
void moe16t_gu_batch(const __half* x16, const int* rows, const GuJob* jobs, int njobs, int maxn, int ni, __half* hb,
                     cudaStream_t st);
void moe8_quant(const bf16* x, int T, int8_t* q, float* s, cudaStream_t st);
void moe8_gu_batch(const int8_t* xq, const float* xs, const int* rows, const GuJob* jobs, int njobs, int maxn, int ni,
                   int8_t* hq, float* hs, cudaStream_t st);
// round 10: the two halves of moe8_expert (gate/up of the next expert on a side stream next to this one's down)
void moe8_gu(const int8_t* xq, const float* xs, const int* rows, int n, const XPtr& w, int ni, int8_t* hq, float* hs,
             cudaStream_t st);
void moe8_dn(const int8_t* hq, const float* hs, const int* rows, const float* wts, int n, const XPtr& w, int ni,
             float* part, cudaStream_t st);
// round 23: W4A8 prefill expert kernels, larger tiles + cp.async pipeline + raw INT4 B fragments (permuted k order;
// bit-identical to moe8_gu_batch / moe8_dn)
int moe8n_mode();  // GLM_PF_MOEN (1 default: the round-23 kernels; 0: the round-6 form)
void moe8n_gu_batch(const int8_t* xq, const float* xs, const int* rows, const GuJob* jobs, int njobs, int maxn, int ni,
                    int8_t* hq, float* hs, cudaStream_t st);
void moe8n_dn(const int8_t* hq, const float* hs, const int* rows, const float* wts, int n, const XPtr& w, int ni,
              float* part, cudaStream_t st);
void moe8_expert(const int8_t* xq, const float* xs, const int* rows, const float* wts, int n, const XPtr& w, int ni,
                 int8_t* hq, float* hs, float* part, cudaStream_t st);
}  // namespace glm
