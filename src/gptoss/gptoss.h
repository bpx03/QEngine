// gpt-oss-120b engine for N cards (TP over all layers), first engine (M2): plain greedy decode of one sequence,
// chunked prefill, contiguous full-layer K/V and a 256-row ring per sliding layer.
#pragma once
#include <cstddef>
#include <string>
#include <vector>

namespace gptoss {

constexpr int NL = 36, H = 2880, NE = 128, TOPK = 4, NH = 64, NKV = 8, HD = 64, WIN = 128;
constexpr int VOCAB = 201088, NBLK = 90;   // NBLK: MXFP4 blocks of 32 per 2880-wide row

struct Config {
    std::string model_dir;
    std::vector<int> devices = {0, 1, 2};
    int max_ctx = 16384;      // tokens of the one sequence (full-layer K/V rows)
    bool ctx_all = false;     // max_ctx = every K/V row that fits in the free VRAM after the weights (the C API's pool; max_ctx then only says how many rows one conversation has: the sgemm prefill attention's scratch is sized by it)
    size_t reserve_extra = 0; // bytes per card left free on top of the engine's own reserve when the context is sized (the C API: room for the decode graphs a server builds over time)
    int chunk = 10240;        // prefill rows per chunk, at most (GPTOSS_CHUNK / --chunk; a call runs in the fewest chunks of equal size). Round 14: 4,096 -> 10,240: every lane of every chunk streams all experts' weights once per layer, so few large chunks are faster (10K: 3 chunks 0.89 s, 1 chunk 0.74 s); without lanes the chunk is 4,096 at most
    int pf_nlanes = 0;        // lanes per chunk of >= 512 rows (GPTOSS_PF_NLANES; 0 = as few as fit the lane scratch: 2 up to 6,828 rows, then 3)
    int pf_stagger = 1;       // three or more lanes run their steps staggered (GPTOSS_PF_STAGGER=0: layer by layer)
    bool graph = true;        // decode steps replay one CUDA graph per card
    // [gptoss-rel2-tagwrap] the collectives' tags are step * 256 + collective + 1 in 32 bits: a word left by a step 2^24
    // steps ago (a row count not used since) would carry the tag of today's step. Every ar_clear launches (decode
    // steps + drafter passes) all cards zero their receive words between two steps, so no word is older than that.
    // GPTOSS_AR_CLEAR=N (>= 1; default 2^22); GPTOSS_STEP0=N: the step counter's start (tests: a wrap inside a short run)
    long ar_clear = 1L << 22;
    unsigned step0 = 0;
    bool verbose = true;
    // switches for A/B (environment overrides in the constructor: GPTOSS_PF_MOE=blas|fused, GPTOSS_COMM=auto|p2p|host)
    bool ar_split = false;    // decode all-reduce over 8 blocks with the router in the same launch (GPTOSS_AR=split); off until it measures faster and passes the per-layer check
    bool lm2 = true;          // certified two-stage lm_head for greedy steps (exact; +193 MiB per card; GPTOSS_LM=full: off)
    bool samp2 = true;        // sampled rows without candidates (plain temperature) take the certified two-stage head with the row's Gumbel noise in the bounds: the same tokens as the full head for the same seed (needs lm2; GPTOSS_SAMP=full: the full head)
    bool dense12 = false;     // decode q/k/v and o_proj from the lossless 12-bit layout (GPTOSS_DENSE=12); opt-in: +0.5 % measured, and the bf16 copy still stays for prefill (+0.5 GiB per card)
    int pf_gemm = 5;          // fused expert GEMM form (GPTOSS_PF_GEMM): 1 = first, 2 = staged activations + table fragments, 3 = 240-row x 64-token blocks, fragment-layout activations, scale folded into the table, 4 = form 3 with cp.async stages for weights and activations (bit-identical to 3), 5 = the form-4 kernels with units of up to 128 tokens (one weight stream per 128 rows of an expert; bit-identical to 4)
    bool pf_gu_ep = true;     // form 5: bias + SwiGLU + routing weight in the gate_up epilogue (bit-identical to k_act_all_p over the fp32 sums); GPTOSS_PF_GU_EP=0
    bool lm_screen_mma = true;   // head screen of a multi-row step in one tensor-core pass (GPTOSS_LM_SCREEN=scalar: the row loop)
    int dense_mma = 1;        // decode q/k/v and o_proj with the rows of a step as tensor-core columns (round 7): 1 = every step
                              // (a row's value is the same for every row count: batch == alone bitwise), 2 = steps of >= 2 rows
                              // only (the single-row step keeps the scalar kernels: wide and narrow steps then agree to
                              // rounding), 0 = scalar kernels for every step (GPTOSS_DENSE_MMA)
    size_t tailpf = 0;        // bytes of the next dense matrix prefetched into L2 by the latency-bound kernel in front (GPTOSS_TAILPF_MB)
    size_t tailpf_dn = 0;     // the same, issued by the down kernel's blocks for the next layer's q/k/v (GPTOSS_TAILPF_DN_MB)
    bool dn_old = false;      // decode down through k_dn instead of the unrolled k_dn_n (GPTOSS_DN_OLD=1; the same sums)
    bool row_union = false;   // multi-row steps read each distinct expert once (GPTOSS_ROW_UNION=1); off: measured slower than per-row expert blocks
    bool xreg = true;         // multi-row expert kernels with the inputs held in registers across weight rows where measured faster (gate_up at 2-4 rows, down from 8 rows; GPTOSS_XREG=0: the per-row kernels; the same sums)
    int ar_trail = 1;         // decode all-reduce (P2P words or, without peer access, the shared pinned host words) of steps with >= this many rows: tags off the payload (k_ar_t: 4 fp32 per word + a checksum trailer per warp; bitwise the same sums, 25 % fewer link bytes; GPTOSS_AR_TRAIL=N, 0 = never)
    int ar_push = 3;          // with ar_trail: the kernels that make a collective's partial store it into the peers' words as they finish their sums, so the bytes travel while the kernel runs (bit 0: o_proj on tensor cores, bit 1: the 8-row down; GPTOSS_AR_PUSH=0: the collective sends it; the same sums)
    bool router_r = true;     // decode router: one block per router row for all rows of the step, then top-4 + slot order in one block (the same logits, picks and weights; GPTOSS_ROUTER_R=0: k_router / k_top4 / k_order)
    bool attn_ep = true;      // decode attention + combine in the round 10 form: q staged per block, K/V tiles by cp.async one ahead, conflict-free branch-free merges; the same sums bit for bit (GPTOSS_ATTN_EP=0: round 9 kernels)
    bool attn_grid = true;    // decode full-attention layers launch 8 / 32 / 128 key splits by the longest row of the step (<= 256 / <= 1,024 / more keys) instead of always 128; the same sums (GPTOSS_ATTN_GRID=0: always 128)
    bool slot_order = true;   // multi-row steps run their expert slots in expert-id order: a shared expert is read from DRAM once, then from L2 (GPTOSS_SLOT_ORDER=0: as routed; the same sums)
    int kv8 = 0;              // K/V of the full-attention layers (GPTOSS_KV): 0 bf16 (default), 3 k12 = 12-bit K + 8-bit V (the fp8 store + a nibble plane for K; round 12), 1 fp8 = int8 codes + a scale per (row, KV head) (opt-in, lossy: about twice the rows), 2 fp8sim = bf16 store holding the 8-bit values (quality emulation)
    int kv8_sim = 3;          // fp8sim only (GPTOSS_KV_SIM): bit 0 K, bit 1 V, bit 2 quantise in a rotated basis (sign flips + Hadamard)
    int slots = 1;            // ring sets of the sliding layers = sequences that step_rows can hold side by side (GPTOSS_SLOTS)
    size_t pf_dn = 0;         // bytes per routed expert of the down records that the gate_up blocks prefetch into L2 at their end (GPTOSS_PF_DN_KB)
    int pf_attn = 3;          // prefill attention (GPTOSS_PF_ATTN): flash form on fp16 tensor cores: 3 flashq (default since 11-19: fp16 operands + a low limb of q; per-layer check 5.46e-4 vs bar 6.5e-4, 90K pass 19.56 -> 17.46 s), 4 flashx (+ low limb of p too, the sgemm numerics to ~2^-22), 1 flash, 2 flash2 (32 query rows per block); 0 sgemm = fp32 sgemm scores + softmax passes (the off-switch; `blas` is the old name)
    bool pf_lanes = true;     // prefill collectives on a second stream ordered by events; chunks of >= 512 rows run as two lanes, one lane's collective beside the other lane's compute (GPTOSS_PF_LANES=0: one lane, host-synchronised collectives)
    bool pf_hostlanes = true; // [gptoss-rel2-hostpf] without peer access the lanes' collectives are staged through pinned host memory by the copy engines (D2H by the sender, H2D by the owner; the same sums). GPTOSS_PF_HOSTLANES=0: one lane, chunk <= 4,096, peer copies staged by the driver
    bool pf_route = true;     // prefill routing lists built on the device, by every card for itself (signature compared between the cards per chunk): no host thread waits inside the layer loop (needs lanes, the third GEMM form and cards of one model; GPTOSS_PF_ROUTE=host: download, bucket on the host, upload)
    bool pf_owner = true;     // prefill collective, row-owner form: the owner of a row sums the partials, adds the residual, norms, and the peers pull the normed bf16 row (exact, same bits; 3/4 of the link bytes). GPTOSS_PF_AR=rs: fp32 sums to every card (also used by passes with a dump); needs lanes
    bool pf_skip = true;      // prefill, last layer: only q / k / v + the K/V store for the chunk's rows; attention, o_proj, experts and the collectives for the call's last row alone (GPTOSS_PF_SKIP=0: the whole layer for every row; also with a layer dump, sgemm attention or host routing)
    bool pf_rs = true;        // prefill all-reduce: reduce-scatter + all-gather (GPTOSS_PF_AR=pull: full pulls, host-synchronised)
    bool attn_win = false;    // sliding layers: one-kernel attention with the sink inside (GPTOSS_ATTN=win)
    bool pf_fused = true;     // prefill experts: fused MXFP4 GEMM kernel (GPTOSS_PF_MOE=blas: per-expert dequant + cuBLAS)
    int comm = 0;             // decode all-reduce: 0 auto (P2P when every pair has peer access), 1 P2P, 2 host memory
    // [gptoss-r15-dflash] block drafter (z-lab/gpt-oss-120b-DFlash): directory with its model.safetensors (GPTOSS_DFLASH;
    // empty = none). Needs P2P and the trailer all-reduce. Its weights, context stores and scratch are loaded before the
    // K/V pool is sized.
    std::string dflash_dir;
    int df_ctx = 4096;        // drafter context rows per ring set (GPTOSS_DF_CTX): the first quarter stays (the prompt's start), the rest is a ring of the latest rows; 6,144 B per row on a 3-head card
};

// Per-card slices (planner): KV heads [kv0, kv0 + kvn), expert intermediate blocks [jb0, jb0 + jbn) of 90,
// lm_head rows [v0, v0 + vn)
struct Plan {
    int ngpu = 0;
    int dev[4] = {}, kv0[4] = {}, kvn[4] = {}, jb0[4] = {}, jbn[4] = {}, v0[4] = {}, vn[4] = {};
};
// free_mb: free VRAM per card before the load; the cards with the least free memory get the smaller KV-head share
Plan make_plan(const std::vector<int>& devices, const std::vector<long long>& free_mb);

struct Dump {                 // check hooks; null = off
    float* layers = nullptr;  // [NL][rows][H] residual after every layer (first `rows` rows of the call)
    bool taps = false;        // [gptoss-r15-dflash] prefill: layers is [5][rows][H], the residual after layers 1 / 9 / 17 / 25 / 33 only
    int rows = 0;
    float* hfinal = nullptr;  // prefill: [T][H] residual after the last layer, before the final norm
    float* logits = nullptr;  // [VOCAB] of the last row
};

// Errors (missing or bad files, no memory, CUDA errors, bad token ids, a card that stops answering) are
// std::runtime_error. After an error inside prefill / decode the engine refuses further calls.
// max_ctx is a request: the constructor cuts it to what fits in the free VRAM after the weights (max_ctx() tells);
// with Config.ctx_all it is every row that fits (the C API's pool of all slots).
// One row of a batched decode step: sequence slot (its ring set), first full-layer K/V row of the sequence, position
// inside the sequence, input token.
struct StepRow {
    int slot = 0, base = 0, pos = 0, token = 0;
};
// [srv-gptoss-capi] one card's result of a sampled decode step row that asked for candidates (Engine::set_sampling with
// cands; a plain-temperature step fills only the drawn token: its head is the certified two-stage one, round 15):
// out = {M, S, candidates, 0, val[64], id[64] as int bits} (gptoss_sample.cuh); logits = the card's rows [v0, v0 + vn)
// of that step (row 0 only, filled when the step asked for candidates)
struct SampleCard {
    const float* out;
    const float* logits;
    int v0, vn;
};
// [srv-gptoss-capi] a row's draw in step_rows_sampled: temperature > 0 = Gumbel-max sample of softmax(logits / T)
// seeded by (seed, position); <= 0 = the argmax of the full-precision logits
struct RowSamp {
    float temperature = 0.f;
    unsigned long long seed = 0;
    // [gptoss-rel2-penalty] the row's sequence has penalties (its slot's penalty row, Engine::pen_reset); pen_id /
    // pen_val: the entry of the token this row feeds (the reply's newest token; -1: none)
    bool pen = false;
    int pen_id = -1;
    float pen_val = 0.f;
};
struct Impl;
class Engine {
public:
    explicit Engine(const Config& cfg);
    ~Engine();
    int max_ctx() const;
    int chunk() const;                     // rows of one prefill pass (GPTOSS_CHUNK, clamped at load; a longer call splits into equal passes)
    int pos() const;                       // rows in the sequence
    void reset();                          // empty sequence
    void truncate(int n);                  // keep the first n rows; allowed when pos() - n <= 128 (ring) or n == 0
    int prefill(const int* ids, int n);    // appends n rows; returns the greedy next token
    int decode(int token);                 // appends 1 row; returns the greedy next token
    // appends R rows (1..16: a token and R - 1 drafts, positions pos() .. pos() + R - 1) in one step; out[i] = the greedy
    // token after row i. A row's result is the same as in a single-row step. The caller keeps the rows whose input
    // matched (truncate(pos() - rejected); the rejected rows' K/V are overwritten by the next step).
    void decode_rows(const int* tokens, int R, int* out);
    // Rows of several sequences in one step (n <= 16): row i reads ring set `slot` and full-layer rows [base, base + pos)
    // and writes row base + pos; out[i] = its greedy next token. Rows of one sequence with consecutive positions are
    // causal inside the step (drafts). A row's result does not depend on the other rows (batch == alone). The caller
    // owns the layout (no overlap of [base, base + length) between sequences) and the positions; pos() is not changed.
    void step_rows(const StepRow* rows, int n, int* out);
    // [gptoss-rel1-rowfault] on: in a step of >= 2 rows, a row whose logits are not finite gets out[i] = -1 (the other
    // rows their tokens; every row bad, or a failed card, still throws). Off (default): such a row throws.
    void set_soft_rows(bool on);
    // ---- [srv-gptoss-capi] server hooks
    // the bound sequence: decode / decode_rows / prefill / truncate / ring_save / ring_load act on ring set `slot` and
    // full-layer rows [base, base + lim), with pos rows held (Config.slots ring sets; default binding slot 0, base 0)
    void bind(int slot, int base, int pos, int lim);
    int num_slots() const;                 // Config.slots
    bool broken() const;                   // [gptoss-rel1-exit] an error left the cards in an unknown state (every later call fails)
    void copy_rows(int src, int dst, int n);   // full-layer rows [src, src + n) -> [dst, dst + n) (no overlap): relocation
    // the bound slot's ring set (18 sliding layers x 256 rows) as a host blob (state at pos()), and back:
    // ring_load(blob, n) makes the sequence the first n rows, valid when the blob was saved at pos n and the full layers'
    // rows < n were not written since (prompt-end snapshot: turn 2 drops the analysis, the reply diverges at the prompt end)
    void ring_save(std::vector<unsigned char>& out) const;
    void ring_load(const std::vector<unsigned char>& in, int n);
    // temperature > 0: one-row decode steps draw their token (Gumbel-max, seeded by (seed, position)) in a second
    // graph; cands: with the full-precision head, filling sample_card() with the 64 largest + row 0's logits (top_k /
    // top_p / min_p on the host); without cands sample_card() is NOT filled (certified two-stage head, the same token
    // as the full head; GPTOSS_SAMP=full: the full head); <= 0: greedy (default)
    // draw = false: the argmax of the row's (penalised) full-precision logits instead of the draw (greedy with penalties)
    void set_sampling(float temperature, unsigned long long seed, bool cands, bool draw = true);
    // [gptoss-rel2-penalty] presence / frequency penalties, per sequence (slot). pen_reset: the slot's penalty of every
    // token = 0, then vals[i] for ids[i] (the reply so far: presence * (c > 0) + frequency * c). A sampled step
    // subtracts the slot's penalties from its logits before the temperature (two-stage head: from the bounds too; the
    // same tokens as the full head). set_sampling_penalty (after set_sampling, which turns it off): the one-row steps
    // of the bound slot `slot` are penalised and the entry of token id becomes val first (slot < 0: off).
    void pen_reset(int slot, const int* ids, const float* vals, int n);
    void set_sampling_penalty(int slot, int id, float val);
    // step_rows with the sampled head: row i draws per rs[i] (no candidates); its own graphs per row count
    void step_rows_sampled(const StepRow* rows, const RowSamp* rs, int n, int* out);
    SampleCard sample_card(int card, int row = 0) const;
    // full-layer rows [src, src + n) as a host blob (every card and layer; row_bytes() per row), and the first n rows of a
    // blob back to [dst, dst + n): the host archive of conversations and of system-part snapshots
    size_t row_bytes() const;
    void rows_save(int src, int n, std::vector<unsigned char>& out) const;
    void rows_load(const std::vector<unsigned char>& in, int dst, int n);
    // ---- [gptoss-r15-dflash] block drafter (Config.dflash_dir). A drafter context per ring set (slot): row p of it is made
    // from the target's residual taps of the sequence's row p. Decode steps keep the taps of their rows on the cards (a ring
    // of the last 256 rows); a prefill with a drafter commits its rows by itself.
    bool has_drafter() const;
    int draft_count(int slot) const;           // context rows (= the next position) of the slot's drafter context
    // rows [row0, row0 + n) of the LAST decode step (step_rows / decode_rows) become the slot's next n context rows (queued;
    // computed with the next draft_block, or dropped when their taps left the ring)
    void draft_commit(int slot, int row0, int n);
    void draft_truncate(int slot, int n);      // keep the context rows of positions < n (0: empty)
    void draft_warm(int slot, int maxb);       // [phase 2] makes the pass graphs (new rows 1..5 x blocks 2..maxb, 16 rows alone) at the slot's context size; state unchanged
    void draft_drop(int slot, int keep);       // [phase 2] queued rows: keep the newest `keep`, the rest never joins the context (holes: lower acceptance only)
    // one drafter pass: block = [token at position pos == draft_count(slot), B - 1 mask rows] (2 <= B <= 8), out[0 .. B - 2] =
    // the drafts of positions pos + 1 ..; B < 2 only computes the queued context rows
    void draft_block(int slot, int pos, int token, int B, int* out);
    // context rows from host taps: taps [n][5][2880] fp32 (residual after target layers 1 / 9 / 17 / 25 / 33 of n consecutive rows)
    void draft_feed(int slot, const float* taps, int n);
    double draft_vram_mb() const;              // drafter bytes on the fullest card
    void set_dump(const Dump& d);
    double load_seconds() const;
    const Plan& plan() const;
    bool p2p() const;                      // decode all-reduce path in use
    std::string vram_table() const;        // per card: weights by kind, K/V, scratch, free after the load

private:
    Impl* p_;
};

}  // namespace gptoss
