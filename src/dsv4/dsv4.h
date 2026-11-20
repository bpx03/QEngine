#pragma once
// DeepSeek-V4.1-Flash-1017 engine (M1) for 3 x RTX 3090.
//
// Layout: layer split (pipeline) over the cards, layers [l0, l1) per card, the head on the last card. Hot routed
// experts in native FP4 (checkpoint bytes) in per-card VRAM slots (slot table per layer: expert -> slot or -1);
// all other experts in pinned host RAM in a per-expert format (cb3g128 / cb2g128 / fp4), computed by the CPU lane
// at decode and streamed to the owner card at prefill. Engram rows in host RAM, packed (eg2/eg3/eg4 codebooks, an
// optional eg4 table for frequent rows). Reference math: tools/dsv4/dsv4_model.py (+ dsv4_quant.py for the lossy
// tiers). Report: section M1.
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <atomic>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#define DCK(x)                                                                                   \
    do {                                                                                         \
        cudaError_t e__ = (x);                                                                   \
        if (e__ != cudaSuccess) {                                                                \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e__), __FILE__, __LINE__); \
            abort();                                                                             \
        }                                                                                        \
    } while (0)
#define DCHECK(c, ...)                                                       \
    do {                                                                     \
        if (!(c)) {                                                          \
            fprintf(stderr, "check failed %s at %s:%d: ", #c, __FILE__, __LINE__); \
            fprintf(stderr, __VA_ARGS__);                                    \
            fprintf(stderr, "\n");                                           \
            abort();                                                         \
        }                                                                    \
    } while (0)

namespace dsv4 {

// [kb-dsv4-ctxgrow] the context K/V of a request could not grow (lazy reservation, its VRAM taken by another process):
// thrown by the forward paths before anything moves; the C API fails that request instead of aborting the engine
struct CtxNoRoom : std::runtime_error {
    using std::runtime_error::runtime_error;
};
// [srv-disconnect] the server's client left while its prompt was prefilled: prefill() stops at the next chunk start
// (set_prefill_abort's flag became non-zero) and throws this after giving back its chunk workspace; the sequence is
// partial, the caller resets it
struct PrefillAborted : std::runtime_error {
    using std::runtime_error::runtime_error;
};

constexpr int D = 5120, NL = 40, V = 129280, NE = 384, TOPK = 6, INTER = 2304;
constexpr int NH = 64, HD = 512, RD = 64, QLR = 1280, OLR = 1024, OG = 8, WIN = 128;
constexpr int HC = 4, NMIX = 24, HC_ITERS = 20;
constexpr int IH = 32, IDIM = 128, ITOPK = 512, CAND_K = 2048, CAND_BS = 8, CAND_SRC = 20;
constexpr int EG_ROWS = 24, EG_HD = 256, EG_DIM = EG_ROWS * EG_HD;   // 6144
constexpr float EPS = 1e-20f, HC_EPS = 1e-6f, RSF = 1.5f, CLAMP = 10.f;
constexpr int EOS = 1;
constexpr int MAXR = 8;     // rows of the GEMV paths (decode, later verify blocks)

inline int ratio_of(int l) {
    static const int r[NL] = {0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                              1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    return r[l];
}
inline bool is_kv_src(int l) { return l == 2 || l == 8 || l == 14 || l == 20; }
inline bool is_idx_src(int l) { return l >= 2 && (l == 2 || l == 8 || l == 14 || (l >= 20 && l % 4 == 0 && l <= 36)); }
inline int kv_src_of(int l) { return l >= 20 ? 20 : l >= 14 ? 14 : l >= 8 ? 8 : 2; }
inline int src_index(int s) { return s == 2 ? 0 : s == 8 ? 1 : s == 14 ? 2 : 3; }
inline bool is_engram(int l) { return l == 1 || l == 14; }

// expert formats (host tier); GPU slots are FP4, the dynamic pool cb3 / M23
// R16: XF_M23 = Maya's byte budget (opt-in pack): w1 / w3 cb2g128, w2 cb3g128 (10.875 MB per expert)
// R16 [dsv4-gguf-iq]: GGUF i-quant matrix formats 4..7 (gguf_iq.h; Maya-S's bytes, opt-in --gguf-experts); an expert
// format of the GGUF mode = XF_GG | gate (bits 0-1) | up (bits 2-3) | down (bits 4-5), each = matrix format - 4
enum XFmt : uint8_t { XF_FP4 = 0, XF_CB3 = 1, XF_CB2 = 2, XF_M23 = 3, XF_IQ2XXS = 4, XF_IQ3XXS = 5, XF_IQ2S = 6, XF_IQ3S = 7,
                      XF_GG = 0x40 };
__host__ __device__ inline bool is_gg(int f) { return f >= 0 && (f & XF_GG); }
__host__ __device__ inline bool is_iq(int mf) { return mf >= XF_IQ2XXS && mf <= XF_IQ3S; }
__host__ __device__ inline XFmt gg_expert_fmt(XFmt g, XFmt u, XFmt d) {
    return (XFmt)(XF_GG | (g - 4) | ((u - 4) << 2) | ((d - 4) << 4));
}
// code bytes of one 256-weight block without its fp16 d (gguf_iq.h layout)
__host__ __device__ inline int gg_cbytes(int mf) { return mf == XF_IQ2XXS ? 64 : mf == XF_IQ3XXS ? 96 : mf == XF_IQ2S ? 80 : 108; }
// format of matrix i (0 w1, 1 w3, 2 w2) of an expert in format f
__host__ __device__ inline XFmt mat_fmt(XFmt f, int i) {
    if (is_gg(f)) return (XFmt)(4 + ((f >> (2 * i)) & 3));
    return f == XF_M23 ? (i < 2 ? XF_CB2 : XF_CB3) : f;
}
// bits of a codebook matrix format (3 for cb3, 2 for cb2)
__host__ __device__ inline int cb_bits(XFmt mf) {
#ifndef __CUDA_ARCH__
    assert(mf == XF_CB3 || mf == XF_CB2);   // review r16 L8: a matrix format (mat_fmt), never FP4 / M23
#endif
    return mf == XF_CB3 ? 3 : 2;
}
// byte sizes of one matrix [N][K] in a single-matrix format: FP4 = K/2 per row + K/32 e8m0 scales; cbB = (B*16 B per
// 128) codes per row + K/128 fp16 scales
__host__ __device__ inline size_t mat_bytes(XFmt f, int N, int K) {
#ifndef __CUDA_ARCH__
    assert(f != XF_M23);   // review r16 L8: per-expert format; callers pass mat_fmt(f, i) (host check: kernels unchanged)
#endif
    if (f == XF_FP4) return (size_t)N * (K / 2 + K / 32);
    if (is_iq(f)) return (size_t)N * (K / 256) * (gg_cbytes(f) + 2);
    int gb = f == XF_CB3 ? 48 : 32;
    return (size_t)N * ((K / 128) * gb + (K / 128) * 2);
}
// one expert = w1 [2304][5120] | w3 [2304][5120] | w2 [5120][2304], each matrix = codes (all rows) then scales
__host__ __device__ inline size_t expert_bytes(XFmt f) {
    return mat_bytes(mat_fmt(f, 0), INTER, D) + mat_bytes(mat_fmt(f, 1), INTER, D) + mat_bytes(mat_fmt(f, 2), D, INTER);
}
struct MatView {          // pointers into one matrix of an expert
    const uint8_t* q;     // codes
    const void* s;        // scales (e8m0 bytes or fp16)
};
__host__ __device__ inline void expert_views(XFmt f, const uint8_t* base, MatView m[3]) {
    const int N[3] = {INTER, INTER, D}, K[3] = {D, D, INTER};
    size_t off = 0;
    for (int i = 0; i < 3; i++) {
        const XFmt mf = mat_fmt(f, i);
        size_t qb = mf == XF_FP4 ? (size_t)N[i] * K[i] / 2
                    : is_iq(mf)   ? (size_t)N[i] * (K[i] / 256) * gg_cbytes(mf)
                                  : (size_t)N[i] * (K[i] / 128) * (mf == XF_CB3 ? 48 : 32);
        m[i].q = base + off;
        m[i].s = base + off + qb;
        off += mat_bytes(mf, N[i], K[i]);
    }
}

// FP8 e4m3 weight [N][K] with an e8m0 scale per 32 x 32 block
struct F8 {
    const uint8_t* w = nullptr;
    const uint8_t* s = nullptr;
    int N = 0, K = 0;
    const float* si = nullptr;   // R22 [dsv4-r22-di8]: set = w holds int8 (a drafter tensor), scales per row and 128 columns
    const __half* sq = nullptr;  // release prep [dsv4-rel-q8]: set = w holds int8 from a GGUF Q8_0 tensor, fp16 scales per
                                 // row and 32 columns ([N][K / 32])
};

typedef __nv_bfloat16 bf16;
typedef __nv_bfloat162 bf162;

// engine API (dsv4.cu)
struct Options {
    std::string ckpt;            // safetensors checkpoint dir (dev tool; default: env DSV4_CKPT); the server uses gguf_model
    std::string packed;          // converted host tiers (tools/dsv4/dsv4_pack.py)
    std::string gguf_dir;        // R16 [dsv4-gguf-iq]: routed experts from this GGUF dir (Maya-S), opt-in; needs an M23 pack
    std::string gguf_model;      // release prep [dsv4-rel-gguf]: the whole model from this GGUF dir (Maya-S split set): no
                                 // checkpoint, no pack; derived data in cache_dir
    std::string dspark_dir;      // the DSpark drafter dir (dspark.bin + dspark.json); default <gguf_model>/../dspark
    std::string cache_dir;       // derived data (expert profile, drafter repack); default $XDG_CACHE_HOME or ~/.cache /qengine/dsv4
    std::string profile;         // [dsv4-rel-gguf]: starting expert profile (a dsv4-profile file or a pack.json) when the
                                 // cache has none yet
    int maxctx = 16384;          // window / compressed KV capacity
    int ctx_static = 0;          // [dsv4-rel8-ctxdef] > 0: maxctx is the server's default (not the user's): with the static
                                 // context reservation (no lazy one: no peer access between every pair, DSV4_NOCTXLAZY)
                                 // the context is this instead (--ctx-static; the dsv4 profile passes 262144, user 11-19)
    int chunk = 10240;           // prefill tokens per layer-major pass
    int kv_pool_mib = 0;         // [dsv4-rel4-p2] compressed K/V pool beyond one sequence's capacity (MiB per card): parked
                                 // sequences stay in VRAM there; 0 = the pool holds one maxctx sequence (parked ones live
                                 // in what the active one does not use yet)
    int cpu_threads = 24;
    double reserve_gb = 0.6;     // per card, left free after the slots
    int attn_tc = 1;             // prefill attention on tensor cores (fp16 q / P); 0 = fp32 SIMT kernel
    int dyn = 1;                 // dynamic expert tier (cb3 copies admitted into VRAM at decode); 0 = off
    int dyn_window = 16;         // admit an expert on its second miss within this many tokens (R13 default 16, was 64)
    float dyn_decay = 0.999f;    // LFU score decay per token
    float car_margin = 0.02f;    // R8 default 0.02 (KL noise, R8-7 / R8-12); cache-aware routing at decode (R7): swap a cold choice for a resident expert within
                                 // this choice-score margin; < 0 = off (exact routing)
    float drop_w = 0.f;          // adaptive top-k at decode (R8): the 6th / 5th choice with a weight (of 1.5) below this is left out
                                 // (weight 0, no host read, no tier update); 0 = off = exact routing. [dsv4-rel2-dropw] default 0
                                 // since 11-17 (was 0.1): same-session NLL band (3 draws) 0.1 1.6295 vs exact 1.6140, all 4
                                 // fixtures worse at 0.1; speed equal within noise (-2.4 / +4.9 / -0.6 %)
    float nv_cap = 4.f;          // NVMe-sourced (checkpoint FP4) admissions per token, engine-wide; 0 = no limit.
                                 // NVMe ~6.6 GB/s / 18.8 MB = ~350 per second: uncapped they cap decode (R7 plan)
    int small_prefill = 32;      // prompts up to this many tokens prefill in rows of MAXR on the decode path; 0 = off
    int prefetch = 3;            // decode: predict each layer's experts (router on the pre-attention stream) and load
                                 // up to this many predicted cold experts into the CPU caches while the GPU works; 0 off
    int pf_early = 2;            // R11 default (R11-5; 0 = off): one layer earlier: layer l's router on layer l-1's FFN input (at l-1's routing);
                                 // its top-N cold picks are prefetched from the end of lane l-1 until l's own prediction
    int prefetch_mr = 1;         // also predict / prefetch for passes of 2..6 rows (DSpark verify); 0 = one row only
    int pool_min = 0;            // FP4 dynamic tier: at least this many reassignable slots per card; must-resident
                                 // experts of the lowest ranks get FP4 host copies from the checkpoint to make them
    double host_budget_gib = 0;  // pinned host RAM for experts (0: MemTotal - 26 GiB); pool copies of resident
                                 // experts are dropped (slots become static) when the plan does not fit (review r4 M5)
    int dspark = 0;              // DSpark draft head (3 stages in VRAM on the last card); decode with decode_spec
    int dspark_n = 5;            // drafts verified per pass (1..5)
    int band_cb3 = 0;            // R8 mixed VRAM tier: the experts of PACK/band_cb3.json resident as cb3g128 next to the FP4
                                 // slots (static; lossy for them: KL in R8-11); 0 = off
    int dspark_adapt = 0;        // R8: DSpark only while it beats one-row steps (running ms/token of passes vs of plain
                                 // steps; plain for 32 tokens after a loss, then 8 passes to re-probe); 0 = always passes
                                 // [dsv4-rel-adapt0] default 0: the ms-timed judge made greedy text timing-dependent
                                 // (plain / pass order -> admission ages -> residency) and cost 2-14 % tok/s
    int dspark_cb3 = 1;          // R8: drafter experts as cb3g128 from PACK/draft_cb3.bin when present (5.3 instead of
                                 // 7.2 GB on the last card; only the acceptance can change, never the output); 0 = FP4
    float dspark_tau = 0.f;      // > 0: verify the leading drafts while the product of their confidences
                                 // (sigmoid of the confidence head) stays >= tau
    std::string placement_in;    // resident set to load (json of placement_out), for runs that must match
    std::string placement_out;   // write the resident expert set here (json), for the reference check
    std::string dump;            // directory: per-layer stream dumps (check mode)
    int asplit = 1;              // R8: decode attention half (wq_b, attention, wo_a, wo_b) split over the 3 cards (heads by
                                 // wo_a group 2/3/3, wo_b by rows), exchanges over P2P (LL12 words); bit-identical
    int hsplit = 1;              // R9: output head rows (vocab) split over the 3 cards (needs P2P), logits written into the
                                 // last card's buffer; bit-identical
    int borrow = 1;              // R9 default (R9-10): the prefill-only chunk buffers (~1.7 GB per card) live in the tail expert slots
                                 // (pool experts with host copies); a long prefill evicts them and copies them back after
    int pdist = 1;               // R11: prefill experts computed on every card (remote residents on their card, host experts
                                 // staged over each card's own PCIe link), results added on the owner in the same order
    int swap = 1;                // R13: every VRAM slot reassignable (FP4 dynamic tier): a resident expert without a host
                                 // copy is evicted by copying its slot bytes into a free host buffer; an admitted FP4-host
                                 // expert frees its host buffer once resident (same bytes; 0 = off)
    float swap_cap = 0.f;        // R13: swaps (victims without a host copy) per token, engine-wide, burst 4 tokens; 0 = no cap
    bool dyn_margin_set = false; // given on the command line (re-review N5: warn when the swap tier is off)
    float dyn_margin = 1.f;      // R13 default 1.0 (R13-5): admit only when the candidate's score beats the victim's by more than this
    void (*reload_without_dspark)() = nullptr;   // R13: host RAM short with DSpark: restart the load without it
    bool ram_reserve_off = false;  // issue #11 rule (no env writes): the re-exec'd load ignores DSV4_RAM_RESERVE_GIB
    int xsplit = 1;              // R9: every layer's resident FP4 experts spread over the 3 cards (traffic-balanced per
                                 // layer); peers get x / ids / weights and push their choice rows back (LL12); bit-identical
};
struct StepTimes {               // decode step split (ms), summed over layers
    double total = 0, gpu_attn = 0, wait_route = 0, cpu = 0, tail = 0, head = 0, engram = 0;
    int ncold = 0, nhot = 0, ndyn = 0, nadmit = 0;
    int npred = 0, npred_hit = 0, nprev_hit = 0;   // prefetched cold experts, of them used; cold experts the previous token also used
    double pf_wait = 0;                           // host wait for the prediction (ms)
    int npred_e = 0, npred_e_hit = 0;             // R11: early-prefetched cold experts, of them used
    double pf_mb = 0;                             // R11: MB touched by the L3 prefetches (worker 0 x threads)
    double host_bytes = 0;                        // host RAM bytes read for experts: lane + admissions (NVMe ones twice)
    double lane_bytes = 0;                        // advisor H17: of them the CPU lane's reads (distinct cold experts)
    int nnat = 0;                                 // [dsv4-rel-ram]: of them experts read from the GGUF mapping (no copy)
    double draft = 0;                             // DSpark: draft head ms (window update + block + head, synced)
    int npass = 0, nacc = 0;                      // DSpark: verify passes, accepted drafts
    int rows = 0, nucold = 0, plain = 0;          // R17: rows of the call's pass, unique cold experts (lane reads), plain step
    double enq = 0;                               // R20: enqueue-thread time of the call's forward (ms)
    double bnd = 0;                               // R21: the pass boundary's tier apply (ms; waits on unfinished copies)
    double commit = 0;                            // advisor H1: spec_commit (dyn_update of the pass) on the decode thread (ms)
    int cv_cold = 0, cv_e = 0, cv_p = 0, cv_ep = 0;  // [dsv4-dec1-pfmr] DSV4_PFCOV=1: distinct cold experts, of them in the early /
                                                    // same-layer / either prediction union
    double win_e = 0, win_p = 0;                  // [dsv4-dec1-pfmr] ms: early ids seen -> own ids seen; own ids seen -> routing seen
};
class Engine;
Engine* create(const Options& o);
// prefill the prompt (all tokens), returns the greedy next token; nll (optional): per-position NLL of the next
// prompt token (ppl check; costs a head pass per position).
// follow [dsv4-xcall]: prompt tokens later prefill calls still feed (a chunked prompt's part): the call's tail may skip
// layers 21-39 (the cone across chunks); its return value is then -1 and the held state is 'pending' until a prefill of
// >= ~2.6K rows past the last skipped chunk (any other use first re-prefills the held tokens). 0 = this call ends the prompt
int prefill(Engine* E, const std::vector<int>& ids, std::vector<float>* nll = nullptr, std::vector<int>* argmax = nullptr,
            int follow = 0);
// one decode step with token t (appended), returns the greedy next token
int decode(Engine* E, int t, StepTimes* tm = nullptr);
// review r14 M3: decode(E, t) (the default step-graph path when the graph is on) + the step's logits [V] -> out
int decode_logits(Engine* E, int t, float* out);
// [shared-srv] decode_logits as a DSpark-consistent plain step (cancels a speculative capture, keeps the drafter window)
int decode_logits_plain(Engine* E, int t, float* out);
// DSpark (Options::dspark): one verify pass of t + the drafts; appends the accepted tokens + the bonus token to out
// (the last one is the next input). Greedy: the tokens equal what decode() would produce one by one.
// cap > 0 [dsv4-dec3-cap]: at most cap tokens out of this pass (the block is cut to cap - 1 drafts: exact); the server
// passes what max_tokens still allows, so a max_tokens end leaves the held sequence = what the client got (a cancel
// between passes still leaves it ahead: the next prompt then re-prefills)
int decode_spec(Engine* E, int t, std::vector<int>& out, StepTimes* tm = nullptr, int cap = 0);
// check: append n <= MAXR known tokens as one pass (n = 1: the decode path), logits of every row -> out [n][V]
void forward_rows(Engine* E, const int* t, int n, float* out);
void set_fuse(Engine* E, int on);
void set_lookup(Engine* E, int n);   // R17: prompt lookup beside DSpark (min match; 0 = off)
void set_dspark(Engine* E, int n, float tau);   // n < 0 / tau < 0: unchanged
void reset(Engine* E);
// [srv-disconnect] flag (nullptr = none): read before every prefill chunk (another thread may set it); non-zero ->
// PrefillAborted. The pointer is the caller's per call (the server sets it around one start and clears it after)
void set_prefill_abort(Engine* E, const int* flag);
// [dsv4-srv-pesnap] prompt-end snapshot: snap_mark(P) before a prefill that covers position P (pos <= P <= pos + n)
// keeps the state at P (P >= 1; one snapshot, replaced by the next mark, dropped by reset); snap_pos = its position or
// -1; snap_restore moves the sequence back to it (returns P, -1 = none). The tokens below P are the caller's to check.
void snap_mark(Engine* E, int P);
int snap_pos(Engine* E);
int engine_pos(Engine* E);   // [dsv4-xcall] positions the engine holds (a CtxNoRoom inside a pending-part re-prefill leaves none)
const std::vector<int>& engine_tokens(Engine* E);   // [srv-disconnect] the tokens of those positions
int snap_restore(Engine* E);
// [dsv4-srv-stop] keep the first L positions after a reply that ended inside a pass; 1 = done (exact), 0 = not
// possible here (state unchanged: the caller drops its hold)
int trim(Engine* E, int L);
// [dsv4-rel1-arch] the whole sequence (pos > 0) as a host blob and back: seq_save returns its bytes (0 = nothing held or a
// prompt-end snapshot still pending); seq_load replaces the engine's sequence with the saved one (returns pos, -1 = not
// this engine's blob); a continuation of the loaded sequence reads the same bytes it would have read had it never left.
// seq_bytes = the size seq_save would write now
size_t seq_bytes(Engine* E, bool park = false);   // park: the bytes of seq_park's blob
size_t seq_save(Engine* E, std::vector<uint8_t>& out);
int seq_load(Engine* E, const std::vector<uint8_t>& in);   // -4: no VRAM for its rows now (the blob is fine; [kb-dsv4-ctxgrow])
// [dsv4-rel4-p2] park / unpark: as seq_save / seq_load, but the compressed K/V and index keys stay with the engine (in the
// VRAM pool, later host RAM when the active sequence needs the pages): seq_park returns an id (> 0; 0 = nothing parked,
// the blob is empty) and leaves the engine without a sequence (reset); seq_unpark(blob, id) makes it the engine's
// sequence (returns pos, -1 = not possible: the id was dropped or the blob is not this engine's; the id is gone either
// way). seq_res_free drops a parked id; seq_res_info: parked ids in VRAM, their VRAM bytes per card, host bytes, and
// the counters (activated in place, moved on the device, restored from host, evicted to host, dropped)
int seq_park(Engine* E, std::vector<uint8_t>& out);
int seq_unpark(Engine* E, const std::vector<uint8_t>& in, int id);
void seq_res_free(Engine* E, int id);
size_t seq_res_host(Engine* E, int id);   // host bytes of a parked id (0 while in VRAM)
bool seq_res_alive(Engine* E, int id);    // false: dropped (or unknown)
void seq_res_floor(Engine* E, double gib);   // evictions keep a host copy only while MemAvailable stays above this
void seq_res_info(Engine* E, int* nvram, double* vram_mb, double* host_mb, long cnt[5]);
// [dsv4-rel5-p3] host copies live in pinned buffers; freed ones stay in a cache (DSV4_KVH_CACHE_MIB, default 4096, at most
// half of MemAvailable above the floor) for the next eviction. seq_res_trim releases the cache (memory pressure)
void seq_res_hostinfo(Engine* E, double* pinned_mb, double* cache_mb, long cnt[3]);
void seq_res_trim(Engine* E);
// [dsv4-rel-tierreset]: the dynamic tier back to its state after the load (ms; -1 = not applicable)
double tier_reset(Engine* E);
// [dsv4-rel8-giveback] between requests: a stale lazy-context loan back to the pool, the left-out experts restored
int ctx_idle(Engine* E);
// [dsv4-rel8-giveback] review rel8 M1: --tier-reset on: a give-back refills the freed slots with their seed experts
void set_lz_seed(Engine* E, bool on);
// the held run lies above the base context while the loan is out (hold: and DSV4_CTX_HOLD_S passed): park it first
bool lz_held_above(Engine* E, bool hold);
// [dsv4-rel8-ctxcap] the served context (below --ctx when the load had to cap it)
int engine_max_ctx(Engine* E);
// R24 [dsv4-r24-laneprof]: the CPU lane's per-worker timing line (DSV4_LANEPROF=1), empty otherwise
std::string lane_prof(Engine* E, bool reset);
void set_dump_step(Engine* E, int s);
void set_prefetch(Engine* E, int n);
void set_prefetch_mr(Engine* E, int on);
void set_pf_early(Engine* E, int n);
void set_nv_cap(Engine* E, float cap);
void set_car(Engine* E, float margin);
void set_drop(Engine* E, float w);
void set_gg_fp32(Engine* E, int v);   // R21 [dsv4-r21-dp4a]: GGUF pool kernel, 1 = R16 fp32, 0 = dp4a (default)
void set_dyn_window(Engine* E, int w);   // R16 [dsv4-gguf-iq]: < 0 = no admissions from here on (frozen tier)
void set_small_prefill(Engine* E, int n);
void set_pprof(Engine* E, int lvl);   // [dsv4-pf-pprof] DSV4_PPROF level (0 off, 1 stages, 2 + sampled experts)
double instr_ms(Engine* E);   // [dsv4-pf6-net]: ms the last prefill spent in the DSV4_PFHASH hash and the pprof report
void set_f32k(Engine* E, int on);
void set_q1(Engine* E, int on);
void set_runahead(Engine* E, int on);   // >= 0: decode steps dump per layer (check mode)
void set_qrb(Engine* E, int on);
// [dsv4-rel-sample]: sampling for prefill's first token, decode and decode_spec (exact speculative acceptance for the
// drafts); temperature <= 0 = greedy (the default; bit-identical to before). top_k 0 = 64 (the sampler's candidates)
void set_sampling(Engine* E, float temperature, float top_p, int top_k, float min_p, unsigned long long seed);   // R15: bf16 qr for wq_b / X1 (in-process A/B)
void destroy(Engine* E);
void print_memory(Engine* E);

}  // namespace dsv4
