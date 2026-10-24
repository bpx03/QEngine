#pragma once
#include <atomic>
#include <condition_variable>
#include <deque>
#include <functional>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "model.h"

// One persistent host thread per GPU; run() executes a job on all of them concurrently.
class GpuPool {
public:
    explicit GpuPool(const std::vector<int>& devices);
    ~GpuPool();
    void run(const std::function<void(int rank)>& job);

private:
    void loop(int rank);
    std::vector<int> devices_;
    std::vector<std::thread> threads_;
    std::mutex mu_;
    std::condition_variable cv_, done_cv_;
    std::function<void(int)> job_;
    unsigned long gen_ = 0;
    int pending_ = 0;
    unsigned busy_ = 0;  // ranks still running the current job (diagnostics of a stuck run())
    bool stop_ = false;
};

class Engine {
public:
    Engine(const std::string& model_dir, const std::vector<int>& devices, int max_t,
           const std::vector<double>& split_weights);
    ~Engine();

    // Forget all cached prefixes (every slot's KV contents and all snapshots).
    void reset();
    // Build the speculative-step CUDA graphs of the first slot for every draft count / sampling
    // mode up front (each takes tens of ms to capture on first use otherwise). Returns seconds.
    // modes: bit 0 greedy, bit 1 sampling
    double warmup(int modes);
    // Feed prompt tokens; returns the greedy next token.
    int prefill(const int* ids, int n);
    // Feed one token (the previously generated one); returns the greedy next token.
    int decode(int tok);
    // Speculative decoding (MTP) with k drafts per step. start() prefills the prompt, fills the
    // MTP cache and returns the first generated token; each spec_step() returns 1..k+1 tokens.
    // affinity: caller's conversation key (e.g. hash of the system prompt); requests with the
    // same key prefer the slot that served it before (0 = none).
    // hint: extra snapshot position (e.g. end of the system prompt) so other conversations
    // sharing that prefix can resume from it; 0 = none.
    int start(const int* ids, int n, int k, unsigned long long affinity = 0, int hint = 0);
    // Multimodal prompt: ids >= kImgToken are image rows (content-derived ids, used only for
    // prefix matching); img holds the vision encoder output of all of them in prompt order
    // ([img_rows][hidden] bf16), rope3 the (t, h, w) RoPE position of every prompt row and
    // delta the shift of the positions that follow the prompt (Qwen M-RoPE).
    int start_mm(const int* ids, int n, int k, unsigned long long affinity, int hint, const bf16* img,
                 int img_rows, const int* rope3, int delta);
    // start_mm in steps (other sequences may decode between prefill_more calls): the slot or -1,
    // then prefill_more until it returns true, then start_finish returns the first token
    int start_begin(const int* ids, int n, int k, unsigned long long affinity, int hint,
                    const bf16* img, int img_rows, const int* rope3, int delta);
    bool prefill_more(int budget);  // of the newest prefill
    int start_finish();
    // Several prompts may be in flight (up to kMaxPrefills: the server runs a short prefill
    // between the chunks of a long one): chunks / first token of a given slot's prompt
    bool prefill_more_slot(int slot, int budget);
    int start_finish_slot(int slot);
    bool prefill_active() const { return !pfs_.empty(); }
    // rows of the slot's prompt prefilled so far, -1 if no prefill of it is in flight
    int prefill_pos(int slot) const { const PrefillState* p = pf_of(slot); return p ? p->i : -1; }
    int num_prefills() const { return pfs_.size(); }
    // false while another sequence's prefill is interleaved with decode steps: a step queued
    // ahead would be undone before the next prefill chunk (losing its MTP drafts)
    void set_lookahead(bool on) { lookahead_ = on; }
    int spec_step(int* out);
    // End generation: waits for the step queued ahead and undoes it.
    void finish();
    // Concurrent sequences. start() marks its slot active (other starts avoid it) until
    // release(). step_batch() advances the given active slots by one speculative step together
    // (shared weight reads); out[i*64 ..] receives cnt[i] tokens of slot slots[i]. Single-
    // sequence spec_step() still works on the slot of the last start() / use().
    void release(int slot);
    void step_batch(const int* slots, int n, int* out, int* cnt);
    // make slot the current single sequence again (for spec_step)
    void resume(int slot);
    bool active(int slot) const { return seqs_.at(slot).active; }
    // false (default): every start() releases all slots (one sequence at a time)
    void set_concurrent(bool on) {
        finish();
        for (auto& q : seqs_) q.active = false;
        concurrent_ = on;
    }
    bool has_mtp() const { return mtp_st_ != nullptr; }
    // Grammar-constrained decoding (structured output) with speculative steps.
    // mask_buffer(slot): pinned host bitmasks [kMaxSpecK + 1][(vocab + 31) / 32] uint32 (bit t =
    // token t allowed); spec_io(slot): pinned int[32]: [0] = drafts the model made for the next
    // step, [1..] those drafts (written by start_finish / every step of the slot); [16] = drafts
    // to verify next step, [17..] those tokens (written by the caller, e.g. the grammar-valid
    // prefix of the model's drafts or tokens the grammar forces). Before every step the caller
    // fills [16..] and masks 0..n (mask j: allowed tokens after the pending token and drafts
    // 1..j; mask 0 also serves the prompt's last chunk). constrain(slot, true) (after
    // start_begin, before the prompt's last chunk) turns this on. A constrained sequence is never
    // queued ahead; drafts and masks are copied to the GPUs right before each launch.
    // release() / the next start_begin end it.
    unsigned* mask_buffer(int slot);
    int* spec_io(int slot);
    void constrain(int slot, bool on);
    bool constrained(int slot) const { return seqs_.at(slot).constrained; }
    // temperature <= 0 selects greedy decoding
    void set_sampling(float temperature, float top_p, int top_k, float min_p,
                      unsigned long long seed);
    // Debug: run n <= 8 tokens from position 0 and dump residuals [layers][n][hidden].
    void debug_hidden(const int* ids, int n, float* out);
    // MTP training data: the next start() copies every prompt row's final normed hidden state
    // (bf16, [n][hidden]) to host memory `out` (nullptr: off)
    void set_dump_hidden(uint16_t* out) { for (auto& w : workers_) w->dump_hidden = out; }
    // Full logits of the last forward ([m][vocab]).
    void logits(float* out, int m);

    const Config& config() const { return cfg_; }
    int pos() const { return seqs_[cur_].pos; }

    // prompt-cache statistics of the last start(): tokens reused, slot used
    int last_reused() const { return last_reused_; }
    int last_copied() const { return last_copied_; }
    bool last_from_archive() const { return last_from_archive_; }
    int num_archives() const { return archives_.size(); }
    int num_disk_archives() const;
    // Archive every live slot (and wait for the disk writes): before shutdown / when idle.
    void persist();
    // Archive idle conversations to host memory piece by piece (about `budget` bytes per GPU
    // per call), so that evicting them later costs nothing on the request path: the server
    // calls it when nothing runs. true when nothing is left to archive.
    bool archive_more(size_t budget);
    int last_slot() const { return cur_; }
    int num_slots() const { return seqs_.size(); }
    int slot_capacity(int i) const { return seqs_.at(i).max_t; }
    int max_context() const { return max_ctx_; }
    size_t pool_tokens() const { return pool_tokens_; }
    // a new sequence of n tokens can start now: a free slot, and the pool holds it next to the
    // running ones (each counted with 16K of room to grow; idle ones can be evicted)
    bool can_admit(int n) const;
    // Eviction value of a slot's conversation (kept with its archive): 0 = finished (a subagent
    // whose last reply had no tool call), 1 = normal, 2 = the main thread: evicted from the GPU
    // last, never dropped from the RAM archive while anything else can go.
    void set_prio(int slot, int prio) { seqs_.at(slot).prio = prio; }
    // the reply may emit at most `tokens` more tokens (max_tokens): steps verify only the drafts
    // that fit, so the slot ends exactly at the returned reply and the next turn resumes from its
    // live end (cleared by release / the next prompt)
    void watch_stops(int slot, bool on) { seqs_.at(slot).want_drafts = on; }
    void set_no_ahead(int slot, bool on) { seqs_.at(slot).no_ahead = on; }
    int drafts(int slot, int* out) const {
        const auto& d = seqs_.at(slot).drafts;
        std::copy(d.begin(), d.end(), out);
        return (int)d.size();
    }
    // end the slot's tokens at position pos (a reply cut where its stop sequence starts): within
    // the last step if nothing ran since (else false, the slot is unchanged)
    bool trim(int slot, int pos);
    void set_limit(int slot, int tokens) {
        Seq& q = seqs_.at(slot);
        q.end = tokens >= 0 ? q.pos + tokens : 0;
    }
    int prio(int slot) const { return seqs_.at(slot).prio; }
    // Prompt tokens that a start() of this prompt would resume from (live slot, snapshot,
    // archive or pinned prefix), without side effects: the scheduler's prefill-cost estimate.
    int probe_cached(const int* ids, int n) const;
    int num_pinned() const;
    // monitoring (host-side state; call from the engine's thread): slot i -> {offset, capacity,
    // cached tokens, active, last use}; archives -> {in RAM, RAM bytes, on disk, disk bytes};
    // pool events -> {regions grown, regions moved, conversations evicted}
    void slot_info(int i, long long* out) const;
    void archive_info(double* out) const;
    void pool_counters(long long* out) const { out[0] = n_grow_; out[1] = n_move_; out[2] = n_evict_; }

private:
    void step(const int* ids, int m, bool want_logits);
    void reset_slot();  // the current slot starts from position 0
    void plan_slots(int main_cap);
    // KV pool (see plan_slots)
    static constexpr size_t kMaxPos = 262144;
    size_t pool_tokens_ = 0;
    // kv-pos-split: pool tokens every region takes beyond its rows (Worker::region_slack, max over
    // the cards; 0 without a split): a region of r rows occupies r + slack_ pool tokens
    size_t slack_ = 0;
    size_t units(size_t rows) const { return rows ? rows + slack_ : 0; }
    KvSplit kvs_;
    bool kvs_auto_ = false;  // shares chosen in plan_slots from the free memory after the weights
    long long n_grow_ = 0, n_move_ = 0, n_evict_ = 0;
    int max_ctx_ = 0;
    int round_cap(int need) const;
    size_t free_after(int s) const;
    void place(int s, size_t off, int cap);
    int held_rows(int s) const;
    void compact();
    void pack_up_above(int s);
    bool place_in_gap(int s, int want);
    bool try_grow(int s, int want);
    bool ensure_cap(int s, int need, const std::vector<int>& protect = {});
    void use_slot(int i);
    int pick_slot(const int* ids, int n, int k, unsigned long long affinity);
    struct KvArchive;  // defined below
    struct Snapshot {
        int slot = 0, pos = 0;
        std::vector<int> toks;      // tokens [0, pos) the state was computed from
        std::vector<void*> bufs;    // one pinned host buffer per GPU
        unsigned long long used = 0;
        int kind = 0;               // 0 checkpoint (every 8K / divergence), 1 prompt end (own quota: snap_policy.h)
    };
    // Host-side mirror of one sequence slot.
    struct Seq {
        int max_t = 0;              // KV capacity: region of the pool at token offset off
        size_t off = 0;
        std::vector<int> seq;       // tokens whose effect is in the caches: positions [0, pos)
        int pos = 0, pending = -1, last_tok = 0;
        unsigned long long used = 0, affinity = 0;
        int k_avail = 0;            // drafts in the slot's device tokens for its next step
        bool active = false;        // in use by a generating request
        bool stale = false;         // device token/position invalid (queued step undone)
        int prio = 1;               // eviction value (set_prio): 0 finished, 1 normal, 2 main thread
        bool constrained = false;   // grammar mask on (constrain)
        int end = 0;                // > 0: last position the reply may reach (set_limit)
        // stop sequences (watch_stops): the drafts of the next step for the caller, no step queued
        // ahead while a stop may start in the step in flight (set_no_ahead), and the undo of the
        // last step so the reply can end where its stop starts (trim)
        bool want_drafts = false, no_ahead = false;
        std::vector<int> drafts;
        int rb_pos = 0, rb_rows = 0;   // last step: position before it, rows it committed
        long long rb_epoch = -1;       // launch epoch after it (-1: a step of the slot queued after it)
        long long launches = 0, rb_launch = -1;  // the slot's own step launches (batched: stash per slot)
        bool rb_batched = false;
        bool full = false;          // no room to grow (step_batch): ends with its end token until released
        // shared prefix (stage 4): keys [0, split) are read from the prefix region pres_[pre];
        // the own region holds keys [split, max_t) (max_t stays the logical capacity)
        int split = 0, pre = -1;
    };
    // Shared prefix regions: the KV rows [0, n) of a pinned prefix once in the pool, read by every
    // slot attached to it (Seq::pre); the delta-net state comes from the pinned archive. Placed
    // in the topmost gap (compaction packs conversations from the bottom), never moved; an
    // unreferenced one is dropped under pressure and re-materialized from its host copy.
    struct GpuPrefix {
        std::vector<int> toks;
        int n = 0, refs = 0;
        size_t off = 0;
        unsigned long long used = 0;
    };
    std::vector<GpuPrefix> pres_;
    bool share_prefix_ = true;  // QE_SHARE_PREFIX (0: prefixes are copied into every slot as before)
    int rows_of(const Seq& q) const { return q.max_t > 0 ? q.max_t - q.split : 0; }
    size_t units_of(const Seq& q) const { return units(rows_of(q)); }
    void used_intervals(std::vector<std::pair<size_t, size_t>>& out) const;
    int find_prefix(const int* ids, int n) const;
    int materialize_prefix(const KvArchive& a, const std::vector<int>& protect);
    size_t prefix_room(int n, const std::vector<int>& protect);
    bool drop_lru_prefix();
    void attach(int slot, int pre);
    void detach(int slot);
    bool evict_one(int keep_slot, const std::vector<int>& protect);
    void copy_rows(int dst, int src, int lo, int hi);
    void resync(Seq& q, int slot);
    // batched steps are queued one ahead too while the set of sequences stays the same
    void launch_batch(int buf, bool ahead);  // ahead: queued behind the step in buffer buf ^ 1
    void finish_batch();
    void finish_single();
    void finish_for(int slot);  // undo only what a change to `slot` invalidates
    bool batch_live(int slot) const;
    std::vector<int> bslots_;
    std::vector<int> bdrop_;  // slots of the queued batched step released meanwhile (undone alone)
    int bkvs_[2][kMaxSegs] = {};
    std::vector<int> bstep_[2];          // slots of the batched step in each buffer (bkvs_ order)
    bool bclamp_[2][kMaxSegs] = {};
    // lookup drafts in batched steps (QE_LKB): per slot the history length the cards hold for sure (-1: unknown,
    // full upload), dirty (an undone step may have appended past it), sticky lookup mode, misses since the last find
    int blkr_[2][kMaxSegs] = {};         // lookup rows of each sequence of the step in each buffer (0: MTP)
    std::vector<int> lkb_len_, lkb_miss_;
    std::vector<char> lkb_dirty_, lkb_mode_;
    std::vector<float> lkb_ema_;           // accepted lookup drafts per lookup step (EMA): sizes the slot's rows
    std::vector<int> lkb_cool_;
    std::vector<int*> lkb_pin_;            // pinned upload buffer per slot (lazily allocated)            // steps left before a slot that left lookup mode (low acceptance) may return
    bool lkb_on_ = false;
    int lkb_rows_ = 32, lkb_hold_ = 2, lkb_min_ = 4;
  public:
    long long lkb_steps_ = 0, lkb_rows_sum_ = 0, lkb_tok_ = 0, lkb_bsteps_ = 0, lkb_slotsteps_ = 0;
  private:
    void lkb_inval(int slot) {
        if (slot < 0 || slot >= (int)lkb_len_.size()) return;
        lkb_len_[slot] = -1;
        lkb_mode_[slot] = 0, lkb_miss_[slot] = 0, lkb_cool_[slot] = 0, lkb_ema_[slot] = 6.f;
    }
    int bcap_ = 0, bbuf_ = 0;
    bool binflight_ = false;
    cudaEvent_t bev_[2] = {nullptr, nullptr};
    bool concurrent_ = false;
    int* batch_host_ = nullptr;     // [2][kMaxBatch][64] pinned
    struct PrefillState {  // a prompt being prefilled (start_begin .. start_finish)
        int slot = 0, n = 0, i = 0, kstart = 0, last_m = 1;
        int pin = 0;                      // position whose state is pinned (pin_prefix), 0 = none
        int end_snap = 0;                 // prompt-end state saved at this position (n - 1), 0 = at n
        int tail_from = 0, last_ck = 0;   // tail checkpoints from this position; the last one taken
        std::vector<int> rolling;         // positions of this prefill's periodic / tail checkpoints
        unsigned long long affinity = 0;
        int reused = 0, copied = 0;       // prompt-cache statistics (last_reused() after start_finish)
        bool from_archive = false;
        std::vector<int> ids, dev_ids, rope3, marks;
    };
    static constexpr int kMaxPrefills = 2;
    std::vector<PrefillState> pfs_;       // in flight, the newest last
    PrefillState* pf_of(int slot) {
        for (auto& p : pfs_) if (p.slot == slot) return &p;
        return nullptr;
    }
    const PrefillState* pf_of(int slot) const { return const_cast<Engine*>(this)->pf_of(slot); }
    bool lookahead_ = true;
    bf16* img_table_ = nullptr;     // image embedding rows of the prompt being prefilled (mapped)
    size_t img_cap_ = 0;            // rows
    void take_snapshot(const int* ids, int pos, int kind = 0, int replace = 0);
    std::vector<Snapshot> snaps_;
    // Pinned prefixes: QE_PREFIX_GB of host memory (default 8) outside the archive LRU
    size_t prefix_budget_ = 0, prefix_used_ = 0;
    void pin_prefix(int slot, int pos, unsigned long long affinity);
    // longest prefix of ids[0, n) that some archive / pinned prefix can restore (end state or
    // a checkpoint inside it)
    int archive_coverage(const int* ids, int n) const;
    const KvArchive* prefix_host(const KvArchive& a) const;
    // Context archive: a slot's KV of [0, n) + its delta-net state at n (plus the states of
    // checkpoints inside it, e.g. the end of the system prompt) saved before the slot is reused
    // for another conversation and restored instead of recomputing the prompt. Two tiers: host
    // memory (pageable, allocated on demand: pinning gigabytes stalls the driver; LRU within
    // QE_KV_ARCHIVE_GB) and, with QE_DISK_CACHE=<dir>, files written in the background (LRU
    // within QE_DISK_CACHE_GB) that survive restarts. Per-GPU buffer layout: archive_layout(n),
    // then one delta-net state per sub-checkpoint.
    struct DiskState {  // shared with the writer thread
        std::mutex mu;
        std::condition_variable cv;
        bool done = false, ok = false, cancel = false;
    };
    // Host memory of the archives: pinned slabs (cudaHostAlloc, pinned in the background at
    // startup up to the archive + prefix budget, QE_ARCHIVE_PINNED=0 disables) carved by a
    // first-fit allocator with coalescing; an archive that finds no pinned range (pool not
    // pinned yet, fragmentation, oversized) falls back to pageable memory. Pinned memory makes
    // the GPU copies ~5x faster (no driver staging, no page faults).
    class HostPool {
    public:
        ~HostPool();
        void start(size_t budget, size_t slab);  // pins `budget` bytes in `slab`-byte slabs
        char* alloc(size_t bytes, bool& pinned);
        void free(char* p, size_t bytes, bool pinned);
        size_t pinned_bytes() const { return pinned_; }
        size_t used_bytes() const { return used_; }
    private:
        struct Slab { char* base; size_t size; std::map<size_t, size_t> free; };  // offset -> length
        std::vector<Slab> slabs_;
        mutable std::mutex mu_;
        std::thread th_;
        std::atomic<bool> stop_{false};
        size_t pinned_ = 0, used_ = 0;
    };
    HostPool hpool_;
    struct Bufs {  // one host buffer per GPU
        struct Buf {
            char* p = nullptr;
            size_t n = 0;
            bool pinned = false;
            char* get() const { return p; }
        };
        HostPool* pool;
        std::vector<Buf> v;
        explicit Bufs(HostPool* pl) : pool(pl) {}
        ~Bufs() { for (auto& b : v) pool->free(b.p, b.n, b.pinned); }
        void add(size_t n) { Buf b; b.n = n; b.p = pool->alloc(n, b.pinned); v.push_back(b); }
        Buf& at(size_t i) { return v.at(i); }
        Buf& back() { return v.back(); }
        bool all_pinned() const { for (auto& b : v) if (!b.pinned) return false; return true; }
    };
    struct KvArchive {
        std::vector<int> toks;              // the archived positions [0, n)
        int n = 0;
        std::vector<int> subs;              // positions of the extra delta-net states (< n)
        std::vector<size_t> sizes;          // per GPU
        size_t total = 0;
        std::shared_ptr<Bufs> ram;          // null: on disk only
        std::string path;                   // non-empty: on disk (or being written)
        std::shared_ptr<DiskState> disk;
        size_t data_off = 0;                // file offset of GPU 0's buffer
        unsigned long long used = 0;
        int prio = 1;                       // of the conversation when archived (Seq::prio)
        int split = 0;                      // rows [0, split) live in the prefix named by toks[0, split)
        // pinned prefix (end of a system prompt / shared tools block): outside the LRU and the
        // archive budget, never superseded; new agents of that kind and evicted conversations
        // resume from it (pin_prefix)
        bool pinned = false;
        unsigned long long affinity = 0;    // conversation key it was taken from (pinned only)
        // > 0: a start() is restoring from it; making room (prefix materialization, ensure_cap
        // evictions archiving other slots) must not drop it before its rows / state are copied
        mutable int hold = 0;
        size_t bytes() const { return total; }
    };
    std::vector<int> devs_;  // CUDA devices of the workers (P2P buffers are per device)
    std::list<KvArchive> archives_;
    static constexpr size_t kArchiveChunk = ~(size_t)0 >> 1;  // one contiguous buffer per GPU
    static constexpr int kMaxSubs = 4;
    size_t archive_budget_ = 0, archive_used_ = 0;
    int archive_min_ = 2048;
    bool pool_log_ = false;  // QE_POOL_LOG=1: pool / archive events on stdout
    void archive_slot(int slot);
    // Incremental archiving of idle conversations (archive_more): the archive under
    // construction and each GPU's next piece; completed by archive_slot() if the slot is
    // needed before it is done.
    struct BgArchive {
        int slot = -1, n = 0;
        KvArchive a;
        std::vector<int> piece;
    } bg_;
    bool bg_prepare(int slot);
    void bg_finish(bool all);
    void drop_archive(std::list<KvArchive>::iterator it);
    bool ram_make_room(size_t need, int prio);
    void disk_make_room(size_t need);
    void disk_forget(KvArchive& a);
    bool archive_load(KvArchive& a);
    void disk_init();
    void disk_enqueue(KvArchive& a);
    void disk_wait_idle();
    void disk_writer();
    std::string model_dir_, fingerprint_, disk_dir_;
    size_t disk_budget_ = 0, disk_used_ = 0;
    struct DiskJob {
        std::shared_ptr<Bufs> ram;
        std::vector<size_t> sizes;
        std::vector<char> header;
        std::string path;
        std::shared_ptr<DiskState> st;
    };
    std::deque<DiskJob> disk_q_;
    std::mutex disk_mu_;
    std::condition_variable disk_cv_;
    bool disk_stop_ = false;
    int disk_busy_ = 0;
    std::thread disk_thread_;
    std::thread alloc_thread_;               // pre-pins snapshot buffers in the background
    std::mutex snap_mu_;
    std::vector<std::vector<void*>> free_bufs_;
    std::atomic<bool> stop_alloc_{false};
    std::vector<Seq> seqs_;
    int cur_ = 0;
    int last_reused_ = 0, max_snaps_ = 32;
    int snap_reserve0_ = 16;        // snapshot entries kept for checkpoints (QE_SNAP_CKPT_RESERVE)
    int ckpt_every_ = 8192;         // checkpoint spacing of a prefill (QE_CKPT)
    int end_cut_ = 4096;
    int tail_ckpt_ = 8192;
    int ckpt_keep_ = 4;             // periodic / tail checkpoints kept per prefill (QE_CKPT_KEEP)          // tail checkpoints in the last this many tokens (QE_TAIL_CKPT)            // prompt-end state at n - 1 from this many prefilled tokens (QE_END_CUT)
    int last_rows_ = 1;    // logit rows of the last forward (logits())
    int last_copied_ = 0;  // prefix tokens copied from another slot / an archive by the last start()
    bool last_from_archive_ = false;
    unsigned long long tick_ = 0;
    Config cfg_;
    std::vector<CardPlan> plan_;
    SafeTensors st_;
    std::unique_ptr<SafeTensors> mtp_st_;
    int k_ = 0;        // max drafts per step (0 = plain decoding)
    SampleParams samp_{1.f, 1.f, kTopK, 0.f, 0, 0};
    bool samp_on_ = false;
    // Adaptive draft count: per-depth acceptance statistics (exponentially decayed) predict the
    // expected tokens per step for each k; next_k() picks the best tokens-per-millisecond.
    int next_k();
    void observe(int kv, int accepted);
    double acc_num_[kMaxSpecK + 2] = {}, acc_den_[kMaxSpecK + 2] = {};
    int adapt_ = 1;        // 0 fixed k, 1 per-depth statistics (default), 2 grow/shrink heuristic
    int k_heur_ = 4;
    double cost0_ = 8.0, cost1_ = 0.65;   // step time model: cost0 + cost1 * k (ms)
    // measured step times (GPU events): decayed least squares of ms ~ c0 + c1 * drafts verified,
    // so the draft count follows the real cost (it grows with the context length)
    double rs_[5] = {};                    // sum w, w x, w x^2, w y, w x y
    cudaEvent_t tstart_[2] = {nullptr, nullptr};
    bool buf_captured_[2] = {false, false};  // the step's graph was built (host time in its timing)
    void observe_time(int buf);
    int buf_kv_[2] = {0, 0};               // drafts verified by the step in each buffer
    bool buf_clamped_[2] = {false, false};
    bool buf_lk_[2] = {false, false};      // the step in each buffer was a lookup step
    bool lk_on_ = false;                   // QE_LOOKUP: lookup drafts on the GPU (single-sequence steps)
    int lk_slot_ = -1;                     // slot whose history the cards hold (-1: none / stale)
    int lk_next_ = 0;                      // lookup drafts found by the last finished step
  public:
    long long lk_steps_ = 0, lk_rows_ = 0, lk_tok_ = 0, steps_all_ = 0;  // statistics (lookup steps, their rows / tokens)
  private:
    int buf_kd_[2] = {0, 0}, bkd_[2] = {0, 0};  // drafts made by the step in each buffer
    long long epoch_ = 0;                  // launches (steps, prefill chunks): trim's validity  // fewer drafts than available (set_limit): no statistics
    // drafts a step of slot q starting at position `base` may verify under its limit (set_limit)
    static int clamp_kv(const Seq& q, int kv, int base) {
        return q.end > 0 ? std::max(0, std::min(kv, q.end - base - 1)) : kv;
    }
    // Steps are queued one ahead (graph launches cost ~0.8 ms of host time each), so the GPU
    // never waits for the host; emit buffers / events alternate between the two steps.
    void launch_step(int buf, int base, int lk = 0);  // base: the slot's position when the step runs; lk: lookup drafts
    int* emit_host_ = nullptr;   // [2][64] pinned
    std::vector<unsigned*> mask_host_;  // per slot, pinned (mask_buffer)
    std::vector<int*> spec_io_;          // per slot, pinned (spec_io)
    int con_kv(int slot, int cap);       // drafts a constrained slot verifies next (spec_io[16], capped)
    void push_mask(int slot);    // host mask of a constrained slot -> every GPU (stream-ordered)
    cudaEvent_t emit_ev_[2] = {nullptr, nullptr};
    bool inflight_ = false, ahead_ = false;
    int buf_ = 0;
    std::vector<std::unique_ptr<Worker>> workers_;
    std::unique_ptr<GpuPool> pool_;
    Comm comm_;
    bf16* embed_ = nullptr;
};
