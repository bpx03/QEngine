// Inkling 2 Mini one-card engine (see ink.cu).
#pragma once
#include <string>
#include <vector>

namespace ink {

constexpr int H = 2304, NQ = 24, NKV = 8, HD = 128, QD = NQ * HD, KVD = NKV * HD, DREL = 16, REL = 1024, WIN = 1024;
constexpr int NE = 128, TOPK = 8, I = 1152, NL = 24, LOOP0 = 6, LOOP1 = 18, NPASS = 3, GR = 64, BLOCK = 6;
constexpr int NAPP = NL + (NPASS - 1) * (LOOP1 - LOOP0);   // 48 layer applications per token
constexpr int V = 200058, VPAD = 201024;
constexpr float QK_CLIP = 22.627416997969522f, REL_CLIP = 8.f, ROUTE_SCALE = 8.f, LOGIT_CAP = 30.f;
// fused projection row offsets (rows fed by the query input)
constexpr int OFF_Q = 0, OFF_R = QD, OFF_G = OFF_R + NQ * DREL, OFF_S = OFF_G + QD, OFF_END = OFF_S + NQ;
constexpr int CH = 1024;     // prefill chunk (<= the ring slack)
constexpr int VR = 8;        // max rows of a speculative verify step
constexpr int MROW = 32;       // max rows of a row step (batched slots x their rows)
constexpr int RING = 2048;   // sliding-layer K/V ring (window 1024 + one chunk)
// row table of a multi-row step (verify rows and / or rows of several slots), device int[RT_N]: [j] position of row
// j, [RT_SLOT + j] its slot, [RT_SEG + j] the first row of its segment (the rows of one sequence), [RT_FIRST + z]
// the first row of segment z; [ink-audit-rowgraph] [RT_TB + 4 MROW x (global)] the k_attn_tb3 tiles of the step per
// layer kind (local 0 / global 1): [0, MROW) chunk keys per row (0: the other form), [MROW + t] first row of tile t,
// [2 MROW + t] its rows, [3 MROW] the tile count (data, so one row-step graph serves every per-row split / tile shape)
constexpr int RT_SLOT = 64, RT_SEG = 128, RT_FIRST = 192, RT_TB = 256, RT_N = RT_TB + 8 * MROW;

struct Opts {
    std::string model_dir = "models/Inkling-2-Mini";
    int max_ctx = 32768;
    bool drafter = false;   // load the DSpark block drafter (mtp.safetensors) for spec_step()
    int slots = 1;          // sequence slots (batched decoding); max_ctx is per slot
    int pool = 0;           // [ink-srv-pool] full-attention K / V rows of all slots together (0: slots x max_ctx); slot
                            // intervals [base, base + lim) by bind() (default slot i: [i max_ctx, (i + 1) max_ctx))
    int pool_auto = 0;      // [ink-srv-capi] > 0 (pool 0, drafter on): pool = every row that fits with the drafter (at
                            // most slots x max_ctx); slots halved while that is below pool_auto rows (slots() tells)
    // [ink-tp-dec1] tensor parallelism over these CUDA devices (empty: QE_DEVICES, else the one-card engine). One rank
    // per entry; a device listed k times holds k ranks ("virtual TP": the same arithmetic as k cards, for tests).
    // tp_w: relative card speeds for the planner (empty: QE_TP_W, else equal).
    std::vector<int> devices;
    std::vector<double> tp_w;
};

// [ink-samp-dev] filters the head settles on the device: top_k 2..64 (with any top_p / min_p); without top_k: min_p
// (certified when its set fits in 64) and / or top_p (top_p alone certified by a bound on the mass outside the 64-list)
inline bool filter_on_device(int top_k, float top_p, float min_p) {   // (+ top_p alone / with min_p, no top_k)
    if (top_k >= 2 && top_k <= 64) return true;
    return top_k <= 0 && ((min_p > 0.f && min_p < 1.f) || (top_p > 0.f && top_p < 1.f));
}

class Engine {
public:
    explicit Engine(const Opts& o);
    ~Engine();
    void reset();
    int pos() const;
    int max_ctx() const;   // after the load-time memory fit (may be below Opts::max_ctx)
    bool drafter_on() const;   // false when Opts::drafter was off or the memory fit turned it off
    void prefill(const int* ids, int n, float* hid = nullptr);   // hid: [n][H] final normed hidden (fp32)
    int argmax();                                                 // greedy token after the last prefill / step
    // [ink-srv-samp] sampling of a slot (temperature <= 0: greedy): every step of the slot draws the Gumbel-max token
    // of softmax(logits / T) with noise g(seed, row position, vocab row) through the exact two-stage head (ids == the
    // full head); drafts verify against that token, so spec == plain and batch == alone for the same seed.
    void set_sampling(int slot, float temperature, unsigned long long seed);
    int sample_last();   // after a prefill of the selected slot: its next token with the slot's sampling
    // [ink-samp-dev] filter of a sampled slot (after set_sampling; top_k 0 = none, top_p 1 = none, min_p 0 = none):
    // when filter_on_device() the head draws the Gumbel-max token over the row's kept set (top_k, then min_p relative
    // to the most likely token, then the smallest top_p prefix), drafts verify against it; else the filter is ignored
    // here (the caller samples on the host). filter_ready(): the device path is built (INK_SAMP_DEV != 0, two-stage head)
    void set_filter(int slot, int top_k, float top_p, float min_p);
    bool filter_ready() const;
    long long filter_fallbacks() const;
    std::string filter_stats() const;
    // rows the device left to filt_host (list overflow, an uncertified min_p / top_p set); candidate counts (statistics)
    // host pick of `row` of the last step with the slot's filter and noise; -1 when the row has no finite logit (the
    // row wrappers pass that -1 on: callers check tokens, as the C API does)
    int filt_host(int slot, int row);
    void fb_note(int slot, int ntok, bool fb);   // (owner) the slot's host-pick rate (review M2)
    int sample_last_raw();                // (rank level: -2 = left to the host)
    int spec_step_raw(int anchor, int k, int* out, float* probs = nullptr);   // (rank level: -2 = left to the host)
    void row_logits(int row, float* out);   // [V] capped logits of row `row` of the last step / prefill (row 0)
    int step(int id);                                             // one decode step, returns the greedy next token
    void set_dump(bool on);                                       // record every layer application's output
    const std::vector<float>& dumped() const;
    const std::vector<int>& routed() const;   // [app][T][8] in dump mode
    void sync();
    void v_stats(long long* tot, long long* sub, long long* zero, long long* big);   // development
    void last_hidden(float* out);
    void row_hidden(int row, float* out);   // [ink-tp-tcx] tests: [H] final normed hidden of row `row` of the last row step   // [H] final normed hidden of the last step
    void fake_depth(int n);   // bench only: set the position without a prefill (speed at depth)
    void set_ab(int v);   // A/B variant (development); recaptures the step graph
    // speculative decoding (Opts::drafter): one step = drafter block + verify of k rows (anchor + k-1 drafts).
    // anchor = the last emitted token (not yet in the cache); out gets the emitted tokens (accepted drafts + the
    // target's next token), returns their count (1..k). Greedy; output identical to step() one token at a time.
    int spec_step(int anchor, int k, int* out, float* probs = nullptr);
    // drafter only (tests): 7 drafts for an anchor at position ap <= pos() (context < ap must be in the caches)
    void draft_at(int ap, int anchor, int* drafts, float* probs = nullptr);
    void draft_probe(int anchor, float* probs);   // tests: probs[0..8) of one drafter block at pos() for this anchor
    // bench: mean ms of n verify steps of k rows (fixed junk drafts; ~1 token per step) and of n drafter blocks
    void bench_spec(int k, int n, double* verify_ms, double* draft_ms);
    std::string spec_stats(bool reset);   // steps, lookup steps, rows / step, tokens / step since the last reset
    // spec_step(k < 0): adaptive, up to -k rows (prompt lookup when INK_LK > 0 and matched, else drafter + confidence stop)
    // ---- slots (batched decoding, branch inkling-batch)
    int slots() const;
    void select(int slot);   // the slot that reset / prefill / step / spec_step / pos / argmax act on
    int slot_pos(int slot) const;
    // one decode row for each of n distinct slots (n <= VR): toks[i] = the last emitted token of slots[i] (not yet in
    // its cache); out[i] = its greedy next token. Row i's result == step() on that slot alone (bitwise).
    void step_rows(const int* slots, const int* toks, int n, int* out);
    // nrows[i] rows for slots[i] (anchor + nrows[i] - 1 drafts, flattened in toks; total <= MROW): arg[r] = the greedy
    // token after row r (flattened), acc[i] = accepted drafts of slot i (it emits arg[f_i .. f_i + acc[i]])
    void step_rows_k(const int* slots, const int* nrows, const int* toks, int n, int* arg, int* acc);
    // [ink-b-draft] batched speculative step (drafter on): anchors[i] = the last emitted token of slots[i]; out[i * VR ..
    // + nout[i]) = its emitted tokens (== greedy alone). Policy INK_BPOL / set_bpol: 0 off, 1 per-slot stop, 2 budget
    // [ink-srv-capi] nodraft (optional, n flags): those slots run one row without drafts (row_logits(first row of the
    // slot) gives its full head: host-side sampling); last_spec_drafts: drafts per slot of the last call (rows = 1 + k)
    void spec_rows(const int* slots, const int* anchors, int n, int* out, int* nout, const unsigned char* nodraft = nullptr);
    void last_spec_drafts(int* k) const;
    std::string spec_rows_stats(bool reset);
    void set_bpol(int v);
    // prefix checkpoints ([ink-pf-ckpt]): save the state at pos() into the pinned host archive (LRU, INK_ARCHIVE_GB,
    // default 64); returns an id (0: skipped under host-memory pressure). restore_prefix: state as after the saved
    // prefill (argmax() valid), then prefill() the suffix. false if the id was evicted.
    int save_prefix();
    // ---- server hooks ([ink-srv-pool]): the full layers' K / V rows of all slots are one pool of pool() rows; slot i
    // holds the interval [base, base + lim) (bind; pos = rows held). copy_rows / rows_save / rows_load act on pool rows
    // (the 8 full applications + the drafter's global layer). ring_save / ring_load: the selected slot's state besides
    // its pool rows at pos() (sliding windows, conv states, drafter rings + last feature): ring_load(blob, n) makes the
    // slot hold its first n rows when the blob was saved at n and the pool rows [base, base + n) hold that prefix.
    int pool() const;
    int dense_form() const;   // [ink-pf8] dense attention weights: 8 INT8 GPTQ (dense8.safetensors), 9 INT8 RTN, 15 G15 lossless
    int cards() const;        // ranks (1: one card)
    void bind(int slot, int base, int pos, int lim);
    int slot_base(int slot) const;
    int slot_lim(int slot) const;
    void copy_rows(int src, int dst, int n);
    void rows_save(int src, int n, std::vector<unsigned char>& out) const;
    void rows_load(const std::vector<unsigned char>& in, int dst, int n);
    void ring_save(std::vector<unsigned char>& out) const;
    void ring_load(const std::vector<unsigned char>& in, int n);
    // [ink-srv-gap] ring_save with the window rows copied beside the next step; `out` is filled at the end of the next
    // row step or before any engine call that touches rows / the staging buffer; ring_finalize() forces it now
    void ring_save_async(std::vector<unsigned char>& out);
    void ring_finalize();
    bool restore_prefix(int id);
    // [ink-srv-gap] pin the host staging buffer of the copies above now (bytes; the first copy otherwise pays the pin)
    void prepare_host_copies(size_t bytes);
    size_t archive_bytes() const;
    struct Impl;

private:
    Impl* p_;
    bool own_ = true;              // [ink-tp-dec1] false: a rank's view (TP), the outer engine owns the ranks
    explicit Engine(Impl* p);
    void tp_init(const Opts& o, const std::vector<int>& devs);
};

}  // namespace ink
