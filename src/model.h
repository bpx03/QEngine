#pragma once
#include "env.h"
#include <set>
#include <deque>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <map>
#include <tuple>
#include <memory>
#include <string>
#include <vector>

#include "config.h"
#include "kernels/comm.cuh"
#include "kernels/gemv.cuh"
#include "kernels/ops.cuh"
#include "kernels/sample.cuh"
#include "kernels/spec.cuh"
#include "safetensors.h"

struct FullAttnLayer {
    QW qkv, o;
    bf16 *q_norm = nullptr, *k_norm = nullptr;
    void *kc = nullptr, *vc = nullptr;  // [nkv][max_t][256] bf16 or e4m3
    const void *kc2 = nullptr, *vc2 = nullptr;  // shared prefix region of the selected slot
    float ks = 1.f, vs = 1.f;           // fp8 cache scales from the checkpoint
};

struct DeltaLayer {
    QW in, out;  // in rows: [q kl*128][k kl*128][v vl*128][z vl*128]
    W16 ab;      // rows: [b vl..pad][a vl..pad]
    W16 abh;     // fp16 copy of ab for the XH path (QE_GEMV_H16), else empty
    bf16* conv_w = nullptr;  // [ch][4]
    float *A = nullptr, *dt_bias = nullptr;
    bf16* norm_w = nullptr;
    float* S = nullptr;      // [vl][128][128]
    bf16* hist = nullptr;    // [kConvRing][ch]
};

struct MtpLayer {
    Lin fc, qkv, o, gate_up, down;
    bf16 *pre_e = nullptr, *pre_h = nullptr, *in_norm = nullptr, *post_norm = nullptr;
    bf16 *q_norm = nullptr, *k_norm = nullptr, *norm = nullptr;
    bf16 *kc = nullptr, *vc = nullptr;
    const void *kc2 = nullptr, *vc2 = nullptr;
};

struct Layer {
    bool full = false;
    bf16 *in_norm = nullptr, *post_norm = nullptr;
    FullAttnLayer fa;
    DeltaLayer dn;
    QW gate_up, down;
};

// Per-sequence state of one GPU for one sequence slot (KV caches, delta-net state ring,
// conv history, MTP KV). Slots let several conversations (main agent + subagents) keep
// their caches; Worker::select_slot() points the layers at a slot's buffers.
int dec38_max();  // QE_DEC38_MAX: keys up to which decode attention uses ~38 splits (+ the n=1 row split)

struct SeqSlot {
    int max_t = 0;                // rows of the own region
    size_t off = 0;               // its pool offset (region_arrays)
    std::vector<void*> kc, vc;    // per full-attention layer (bf16 or e4m3)
    // shared prefix region (set_prefix): keys [0, split) of every full-attention layer and of
    // the MTP layer; the own region holds key k at row k - split
    int split = 0;
    std::vector<const void*> kc2, vc2;
    const bf16 *mkc2 = nullptr, *mvc2 = nullptr;
    std::vector<float*> S;        // per delta-net layer: [kStateSlots][nv][128][128]
    std::vector<bf16*> hist;      // per delta-net layer: [kConvRing][ch]
    bf16 *mkc = nullptr, *mvc = nullptr;  // MTP layer cache (fp8 bytes when kv_fp8_)
    int* dn_cur = nullptr;
    int* dn_save = nullptr;       // dn_cur at the start of the last speculative step
    int* dn_pn = nullptr;         // deferred commit (dn_stash_seg): pending accepted rows
    float* dn_pend = nullptr;     // their inputs [n_lin][dn_pcap][dn_pend_row_floats]
    int* rdelta = nullptr;        // RoPE position shift of text after images (device)
    SpecState spec;               // this sequence's tokens / drafts / positions (device)
    SampleParams* sp = nullptr;   // sampling parameters (device)
    PenState* pen = nullptr;      // presence / frequency penalty counts (device; sp->pen when a penalty is on)
    bool sampling = false;
    // grammar-constrained decoding (response_format): allowed-token bitmasks over the whole
    // vocabulary, one per verification row ([kMaxSpecK + 1][words], device, fixed address so
    // captured graphs stay valid), applied to this slot's logits before argmax / top-k only in
    // the graphs of a constrained slot (row j: the grammar state after drafts 1..j)
    unsigned* mask = nullptr;
    bool constrained = false;
    LookupDev lk;                 // lookup drafts of batched steps: this sequence's history (QE_LKB, lk_init)
    int* pos8 = nullptr;          // device: spec.pos + 8 / mtp_pos + 8 (second attention segment of > 8 rows)
    int* mpos8 = nullptr;
};

// Owns one GPU's slice of the model and runs its part of every forward pass.
// CUDA graph cache with an LRU bound: every instantiated graph holds device memory (~4 MB for
// a ~1500-node decode step), so an unbounded cache would eventually exhaust a GPU's headroom.
// (An in-flight graph that gets evicted is freed when it completes.)
template <typename K>
struct GraphCache {
    struct E {
        cudaGraphExec_t g;
        unsigned long long used;
    };
    std::map<K, E> m;
    size_t cap = 64;
    unsigned long long tick = 0;
    cudaGraphExec_t get(const K& k) {
        auto it = m.find(k);
        if (it == m.end()) return nullptr;
        it->second.used = ++tick;
        return it->second.g;
    }
    void put(const K& k, cudaGraphExec_t g) {
        while (m.size() >= cap && !m.empty()) {
            auto lru = m.begin();
            for (auto it = m.begin(); it != m.end(); ++it)
                if (it->second.used < lru->second.used) lru = it;
            cudaGraphExecDestroy(lru->second.g);
            m.erase(lru);
        }
        m.emplace(k, E{g, ++tick});
    }
    void clear() {
        for (auto& x : m) cudaGraphExecDestroy(x.second.g);
        m.clear();
    }
    template <typename P>
    void erase_if(P pred) {
        for (auto it = m.begin(); it != m.end();)
            if (pred(it->first)) {
                cudaGraphExecDestroy(it->second.g);
                it = m.erase(it);
            } else {
                ++it;
            }
    }
};

class Worker {
public:
    void set_pf_p0(int p) { pf_p0_ = p; }  // first position of the next prefill call (int8 Q K^T threshold)
    Worker(const Config& cfg, const std::vector<CardPlan>& plan, int rank, int device,
           const Comm& comm);
    ~Worker();
    void load(const SafeTensors& st, const SafeTensors* mtp_st, const bf16* embed_mapped);
    // Sequence slots: bytes per cached token / fixed bytes per slot on this GPU. With a position
    // split (kv-pos-split) a region of cap keys takes cap + region_slack() pool tokens: its arrays
    // hold pm_cap(cap) rows per head (the most any window of cap keys can need), which exceeds
    // cap * share by up to A(P - A)/P rows per split head.
    size_t slot_token_bytes() const;
    size_t region_slack() const { return slack_tokens_; }
    size_t slot_fixed_bytes() const;
    size_t free_bytes() const;
    void add_slot();  // a sequence slot without KV (see set_region)
    // KV pool: one allocation per GPU; slot regions at token offsets (same on every GPU)
    void alloc_pool(size_t tokens);
    void set_region(int slot, size_t off, int cap);
    // rows [0, rows) (keys [split, split + rows)) of a region move from (off0, cap0) to (off1, cap1)
    void move_region(size_t off0, int cap0, size_t off1, int cap1, int split, int rows);
    void clear_graphs(int slot);  // after its region moved: captured graphs hold the addresses
    // Slot reads keys [0, n) from the region at pool offset off (n rows, shared with other
    // slots); n = 0 detaches. Its own region then holds key k at row k - n.
    void set_prefix(int slot, size_t off, int n);
    int prefix_of(int slot) const { return slots_.at(slot).split; }
    // keys [lo, hi) of the region (src_off, src_cap) whose first key is src_split -> the same keys
    // of the region (dst_off, dst_cap, dst_split), every array (layers, kv heads, k/v, MTP); each
    // card copies the rows of the keys it owns
    void copy_keys(size_t src_off, int src_cap, int src_split, size_t dst_off, int dst_cap, int dst_split, int lo,
                   int hi);
    // archive_xfer on an arbitrary region: a host layout of keys [lo, hi) (archive_bytes(lo, hi)),
    // keys [lo, copy_hi) of it <-> the region (off, cap) whose first key is `split`
    int region_xfer(size_t off, int cap, int split, int lo, int hi, int copy_hi, void* const* chunks, size_t chunk,
                    bool save, bool with_dn, int piece = 0, size_t budget = ~(size_t)0);
    // kv-pos-split: the stats exchange buffers (alloc_xchg on every card after load; each card of
    // a split pair gets the peer's receive rows / flags: set_xchg_peer)
    void alloc_xchg();
    float* xchg_buf() const { return xbuf_; }
    unsigned* xchg_flags() const { return xflags_; }
    void set_xchg_peer(int j, float* buf, unsigned* flags);
    // the position maps of the plan may be set after load (shares chosen from the free memory)
    void set_plan(const CardPlan& p);
    void select_slot(int i);
    int slot() const { return cur_slot_; }
    void reset();  // clear recurrent state of the current slot (new sequence)

    // Process tokens_[0..m) at positions pos_[0]..; if logits, writes argmax tokens to out_tok_.
    // spec: keep every intermediate delta-net state and leave the commit to the caller.
    void forward(int m, bool logits, bool spec = false);
    // Same as forward() but through a CUDA graph captured on first use.
    void run(int m, bool logits);
    void run_(int m, bool logits);
    void dn_dump(const char* tag, int m);
    // MTP head over m rows: pairs (hsrc[hrow + i], tok[i]) at positions pos..; drafts -> spec_.mtp_out
    void mtp_forward(int m, const int* pos, const int* tok, const bf16* hsrc, const int* hrow,
                     bool logits, int win = 0);
    // Prompt chunk [i0, i0+m) of an n-token prompt (eager): main forward + MTP KV fill; the
    // last chunk also produces x_n and k drafts in tokens_.
    void prefill_chunk(const int* ids, int i0, int m, int n, int k, const int* rope3 = nullptr);
    // RoPE position shift of the current slot (text after images; 0 for text-only sequences)
    void set_rope_delta(int delta);
    // One speculative step with k drafts (CUDA graph). Emitted tokens -> spec_.emit.
    // kv: drafts verified this step (made by the previous step); kd: drafts made for the next.
    // lookup = true: the kv drafts are the device lookup drafts (lk_dev_), the MTP draft chain (drafts 2..kd) runs
    // only when the step's own lookup finds none (conditional graph node)
    void run_spec(int kv, int kd, bool launch = true, bool lookup = false);  // launch = false: only build the graph
    bool has_spec_graph(int kv, int kd, bool lookup = false) {
        return spec_graphs_.m.count(std::make_tuple(kv, kd | (lookup ? 256 : 0), mode(), cur_slot_)) != 0;
    }
    // Lookup drafts on the GPU (QE_LOOKUP): the current sequence's token history on the device, appended and scanned
    // at the end of every single-sequence step graph; lk_dev_.lk = drafts for the next step.
    void lk_init(int cap, int mmin, int lmax);
    void lk_set_history(const int* ids, int n);  // + scan (synchronous)
    bool lk_on() const { return lk_dev_.hd != nullptr; }
    const int* lk_dev_ptr() const { return lk_dev_.lk; }
    LookupDev lk_dev_;
    cudaStream_t lk_cst_ = nullptr;  // capture stream of the conditional draft chain
    int* pos8_ = nullptr;   // device: position + 8 (second half of a long verification's attention)
    bool verify_ = false;   // attend() of a verification / MTP pass (decode attention up to kPendRows rows)
    // Undo the last speculative step's recurrent-state commit (KV past the position is unused).
    void undo_spec();
    // One speculative step for several sequences at once (CUDA graph): sequence i lives in
    // slot slots[i] and verifies kvs[i] drafts; all make kd drafts for their next step. The
    // weight-streaming matmuls, all-reduces and norms run once over all rows.
    // lkr[i] > 0: sequence i verifies lkr[i] = kvs[i] lookup drafts (its own history, see LkBatch); kav[i]: its MTP
    // drafts made by the last step (the fallback when the history has no match)
    void run_spec_batch(const std::vector<int>& slots, const std::vector<int>& kvs, int kd,
                        const std::vector<int>* lkr = nullptr, const std::vector<int>* kav = nullptr);
    // batched lookup history of slot i: tokens [from, n) uploaded (tail), length := n, scan
    void lkb_sync(int i, const int* tail, int from, int n);
    bool lkb_on() const { return lkb_cap_ > 0; }
    const LookupDev& slot_lk(int i) const { return slots_[i].lk; }
    int lkb_cap_ = 0;
    void lkb_alloc(SeqSlot& sl);
    const SpecState& slot_spec(int i) const { return slots_.at(i).spec; }
    // Point slot i's next speculative step at (token, position) with no drafts to verify
    // (after a queued-ahead step was undone, its device-side token/position are stale).
    void spec_resync(int i, int pos, int tok);
    // Undo slot i's last (queued-ahead) speculative step: recurrent state back to dn_save.
    void undo_slot(int i);
    // Keep only the first r rows (pending token + accepted drafts) of slot i's last speculative
    // step (nothing ran since): stash = the step stashed its rows (deferred commit), else its rows
    // are still in the step buffers (single-sequence replay) and are replayed from dn_save.
    void trim_slot(int i, int r, bool stash);
    // Time the prefill GEMM configurations of every weight shape for t = 64 .. kChunk up front
    // (otherwise each shape / size bucket is tuned on first use).
    void tune_gemm();
    // Host archive of a slot's context (prompt cache that survives the slot being reused):
    // [delta-net state][KV of keys [lo, hi) per full layer / kv head (K, V), MTP layer] (the rows
    // this card owns) packed into pinned chunks of `chunk` bytes (a piece never straddles two).
    int archive_chunks(int lo, int hi, size_t chunk) const;
    size_t archive_bytes(int lo, int hi) const;  // single contiguous buffer
    // save: slot -> host (copy_hi = hi); load: host -> slot, keys [lo, copy_hi) and, if with_dn,
    // the delta-net state. The slot must be the selected one (select_slot); lo = its split.
    // Pieces (the delta-net state, then one array per layer / kv head / k|v) from `piece` on,
    // until `budget` bytes were moved (at least one piece); returns the next piece index, -1
    // once all are done (a full transfer: piece 0, unlimited budget).
    int archive_xfer(int slot, int lo, int hi, int copy_hi, void* const* chunks, size_t chunk,
                     bool save, bool with_dn, int piece = 0, size_t budget = ~(size_t)0);
    bool has_mtp = false;
    SpecState spec_;
    // Pruned MTP draft vocabulary (this GPU's share of the most frequent token ids); empty =
    // full vocabulary. Set before load().
    std::vector<int> draft_ids;
    // Sampling of the current slot: params live on the device; `sampling` selects greedy vs
    // sampled graphs.
    void set_sampling(const SampleParams& p, bool on);  // penalty off (set_penalty after it)
    // presence / frequency penalty of the current slot: counts restart with the distinct seed tokens; pen_host:
    // host-known tokens of the current slot at positions pos.. (first token, steps without drafts)
    void set_penalty(float presence, float frequency, const int* seed, int n);
    void pen_host(const int* toks, int n, int pos);
    static constexpr int kPenSeedMax = 8192;
    int* pseed_d_ = nullptr;
    bool sampling = false;
    // Grammar-constrained decoding of slot i: its logits are masked with its device bitmask
    // (words: (vocab + 31) / 32 uint32) before the token choice. upload_mask copies a host mask
    // into it on this GPU's stream (ordered before the next launch).
    void set_constrained(int i, bool on);
    void upload_mask(int i, const unsigned* host, int rows = 1);
    // host drafts of a constrained slot -> its device tokens[1, 1 + n) (stream-ordered)
    void upload_drafts(int i, const int* host, int n);
    size_t mask_words() const { return ((size_t)c_.vocab + 31) / 32; }
    bool constrained = false;
    int mode() const { return (sampling ? 1 : 0) | (constrained ? 2 : 0) | (attn_rs_ ? 4 : 0); }
    // attn_dec8d row split for the current step (set by the engine from the host-side length: one sequence
    // below kD8RsMax keys); part of the graph mode, results bit-identical either way
    bool attn_rs_ = false;
    // Recurrent-state snapshots for prompt caching (delta-net states of the current slot +
    // conv histories). The KV cache is position-indexed and needs no copy.
    size_t snapshot_bytes() const;
    void snapshot_save(void* dst);
    void snapshot_restore(const void* src);

    // Activation steering (QE_STEER / QE_STEER_DIR, src/steer.h, kernels/steer.cuh): nv vectors, w =
    // [nv][2 modes][layers][hidden] with the sidecar strength folded in (add: alpha v; project: sqrt(alpha) unit v),
    // vector v present on layers flo[v]..fhi[v]; applied to the residual at the output of a slot's layers. Set once
    // after the slots exist, before any forward (graphs capture the buffer addresses).
    void set_steer(int nv, const int* flo, const int* fhi, const float* w);
    bool steer_loaded() const { return steer_w_ != nullptr; }
    // slot s: the vector (-1: off, the default), its mode (0 add, 1 project), layers lo..hi (inside the vector's) and
    // scale; device values at fixed addresses written in this GPU's stream order (no graph changes)
    void set_steer_slot(int s, int vec, int mode, int lo, int hi, float scale);
    // activation capture (qe_capture, eager passes only): per layer the sum over the prompt rows >= cap_r0 of the
    // layer output's residual; cap_begin zeroes, cap_end copies out [layers][hidden] and ends it
    void cap_begin(int r0);
    void cap_end(float* host);
    bool capturing() const { return cap_ != nullptr; }

    // optional per-layer residual dump (host pointer, [layers][m][hidden] floats), debug only
    float* debug_hidden = nullptr;
    uint16_t* dump_hidden = nullptr;  // host: final normed hidden (MTP input) of every prompt row
    // mixed precision (QE_W4_FROM / QE_W4_LAYERS): the layer matrices of the classes in w4_mask come from
    // w4_st (an INT4 checkpoint with the same tensor names) instead of the main checkpoint
    enum { W4C_QKV = 1, W4C_O = 2, W4C_IN = 4, W4C_OUT = 8, W4C_GU = 16, W4C_DOWN = 32 };
    const SafeTensors* w4_st = nullptr;
    int w4_mask = 0;

    int rank, device;
    cudaStream_t stream = nullptr;
    int *pos_ = nullptr, *tokens_ = nullptr, *out_tok_ = nullptr;  // device
    float* logits_ = nullptr;
    size_t weight_bytes = 0, cache_bytes = 0;

private:
    void* dalloc(size_t bytes, bool weight);
    struct RowSpec { std::string name; int r0, r1; };
    W4 load_w4(const SafeTensors& st, const std::vector<RowSpec>& rows, int c0, int c1,
               bool glu = false);
    W8 load_w8(const SafeTensors& st, const std::vector<RowSpec>& rows, int c0, int c1,
               bool glu = false);
    // a layer matrix at the checkpoint's bit width (weight_shape vs weight_packed: 4 or 8 bits); INT8
    // matrices set w8_any_ (gate_up keeps the GLU interleave unless QE_W8_GLU=0)
    QW load_qw(const SafeTensors& st, const std::vector<RowSpec>& rows, int c0, int c1, bool glu, int cls);
    bool w8_any_ = false;  // some layer matrix is INT8: H16 / early GEMV / W4A8 / bf16 partials off
    W16 load_w16(const SafeTensors& st, const std::vector<RowSpec>& rows, int pad_to, int c0 = -1,
                 int c1 = -1);
    // BF16 checkpoint tensor kept as BF16, or RTN-quantized to INT4 (draft-only weights)
    Lin load_lin(const SafeTensors& st, const std::vector<RowSpec>& rows, int pad_to, bool as4,
                 int c0 = -1, int c1 = -1);
    bf16* gather_bf16(const SafeTensors& st, const std::vector<RowSpec>& rows, int pad_to, int c0,
                      int c1, int* n_out, int* k_out);
    void load_mtp(const SafeTensors& st);
    // xh: XH copy of the attention output (decode path only; prefill attention leaves it unwritten)
    void attend(const AttnShape& a, const bf16* q, const void* kc, const void* vc, const int* pos,
                int m, const float* proj = nullptr, bf16* out = nullptr, __half* xh = nullptr);
    // Prompt chunk as two micro-batches: each half's all-reduces run on comm_stream_ while the
    // other half computes, hiding most of the PCIe traffic behind the GEMMs.
    void forward_prefill(int m, bool logits);
    void mtp_forward_prefill(int m, bool logits);  // MTP layer of a prefill chunk, two micro-batches
    // attention / delta-net part of layer i for rows [r0, r0 + mb), ending in partial_
    void mixer_rows(int i, int r0, int mb, const int* pos, int slot_off, bool pb = false);
    void matmul_part(const W4& w, const bf16* x, int ldx, int m, size_t row0, bool pb);
    void matmul_part(const QW& w, const bf16* x, int ldx, int m, size_t row0, bool pb);
    // Copy-engine prefill all-reduce (comm.cuh ce_*): a copy thread per GPU issues the DMA copies
    // and peer flags on two copy streams (one per peer). The worker thread only enqueues: it never
    // blocks on the copy thread (a host thread blocked in cudaMemcpyAsync while peers spin on its
    // kernels was the 10-18 hang), and every device-side wait is on an older operation.
    struct CeCmd {
        cudaEvent_t ev = nullptr;  // recorded on comm_stream_ by the worker; the copies wait on it
        CeCopy cp[2];
        int peer[2] = {-1, -1};    // rank of the destination of cp[j]
        int ncp[2] = {1, 1};       // > 1: the copies cpx[j][0 .. ncp) instead of cp[j]
        CeCopy cpx[2][3];
        int kind = 0;              // 0 = RS rows, 1 = AG sums
        unsigned seq = 0;
        bool stop = false;
    };
    void ce_push(const CeCmd& c);
    void ce_take_wait(unsigned seq, int kind, cudaStream_t st);
    std::deque<cudaEvent_t> ce_old_take_;
    void ce_loop();
    std::thread ce_thread_;
    std::mutex ce_mu_;
    std::condition_variable ce_cv_;
    std::deque<CeCmd> ce_q_;
    cudaStream_t ce_st_[2] = {nullptr, nullptr};
    unsigned ce_seq_ = 0;  // collectives issued (identical on every GPU)
    std::deque<cudaEvent_t> ce_old_;
    static constexpr unsigned kCeVals = 1u << 16;  // ring of flag values (a slot is reused 65536 collectives later)
    unsigned* ce_vals_ = nullptr;
    // owner-finished prefill AR: bf16 rows of micro-batch b in the comm receive buffer (else xn_)
    const bf16* xbf_[2] = {nullptr, nullptr};
    const bf16* xbf_cur_ = nullptr;
    cudaStream_t comm_stream_ = nullptr;
    cudaStream_t pf_stream_ = nullptr;   // L2 prefetch side stream (decode graphs)
    // all-reduce that overlaps an L2 prefetch of the next matmul's weights (decode, m <= 8); sa: the layer output's
    // steering (steer.cuh)
    void allreduce_pf(float* resid, int m, const bf16* norm, bf16* xn, const QW* next, const SteerArg* sa = nullptr);
    // steering (set_steer): vectors, per-slot {vector, mode, lo, hi} [slots][4] and scale [slots] on the device, the
    // host mirrors, the union of the vectors' file layers
    float *steer_w_ = nullptr, *steer_sc_d_ = nullptr;
    int* steer_sl_d_ = nullptr;
    int steer_nv_ = 0, steer_lo_ = 1 << 30, steer_hi_ = -1;
    std::vector<int> steer_sl_h_, steer_lo_v_, steer_hi_v_;
    std::vector<float> steer_sc_h_;
    // the steering of layer li's output for rows of slot `slot` (li = -1 in the result: nothing to do)
    SteerArg steer_arg(int li, int slot) const;
    float* cap_ = nullptr;      // capture accumulators [layers][hidden] (between cap_begin / cap_end)
    float* cap_buf_ = nullptr;
    int cap_r0_ = 0;
    void capture_rows(int li, int m);  // layer li's output rows of the current pass (eager forward)
    // QE_EARLY_GEMV: the all-reduce + its consumer GEMV `w` (m <= 8, XH input xh_xn_)
    // launched together; the GEMV runs on pf_stream_, stages weights while the all-reduce is in flight
    // and waits for its tag. join = false leaves the join to the caller (early_join()). False: not
    // applicable, nothing launched (caller uses allreduce_pf + mm4).
    bool allreduce_early(int m, const bf16* norm, const QW& w, float* y, int ldy, __half* yh, int ldyh,
                         bool join = true, int site = 7);
    void early_join();
    unsigned* ready_ = nullptr;   // [4096]: tags, arrival counters, guard timeouts, fused-AR scratch
    cudaEvent_t early_join_ev_ = nullptr;
    size_t pf_bytes_ = 0;
    std::vector<cudaEvent_t> evs_;
    int ev_next_ = 0;
    cudaEvent_t ev() { return evs_[ev_next_++ % evs_.size()]; }
    int* pos_mb_ = nullptr;  // position of the second micro-batch
    AttnShape layer_shape(const Layer* L, int win = 0) const;  // ash_ + kv format/scales (nullptr = MTP)
    bool kv_fp8_ = true;
    float mtp_ks_ = 0.f, mtp_vs_ = 0.f;  // fp8 scales of the MTP layer cache (max over layers)
    float mtp_sc_mul_ = env_double("QE_MTP_SCALE", 1.f);
    bool mtp_fp8() const {
        static const bool on = env_set("QE_MTP_FP8") ? env_int("QE_MTP_FP8", true) != 0 : true;
        return kv_fp8_ && on;
    }
    bool glu_ = true;  // fused silu(gate) * up in the gate_up matmul epilogue
    // lm_head + distributed argmax over m rows of x -> out[0..m)
    void head(const Lin& w, const bf16* x, int m, int* out, bool draft = false);
    // grammar masks applied by the next non-draft head / head_topk: (first row, rows, bits)
    struct HeadMask { int r0, rows; const unsigned* bits; size_t stride; };
    std::vector<HeadMask> head_masks_;
    void apply_head_masks();
    // the selected slot's masks on rows [0, rows) of the next head (per_row: row j gets mask j
    // (speculative verification), else mask 0 on every row (prompt chunk))
    void mask_current(int rows, bool per_row) {
        if (constrained) head_masks_.push_back({0, rows, slots_.at(cur_slot_).mask, per_row ? mask_words() : 0});
    }
    // lm_head + gathered top-k candidates per row -> topk_all_ (sampling path)
    void head_topk(const Lin& w, const bf16* x, int m);
    bf16* load_bf16(const SafeTensors& st, const std::string& name, int b, int e);
    float* load_f32(const SafeTensors& st, const std::string& name, int b, int e, bool neg_exp);

    const Config& c_;
    CardPlan p_;
    int ngpu_;
    std::vector<SeqSlot> slots_;
    char* pool_ = nullptr;
    // kv-pos-split: pool bytes per token (sum over the local heads of their shares), the slack
    // tokens every region takes beyond its capacity, and the per-head bytes of one key
    size_t tok_bytes_ = 0, slack_tokens_ = 0, head_tok_bytes_ = 0;
    void plan_bytes();
    void region_arrays(size_t off, int cap, std::vector<std::pair<char*, size_t>>& out) const;
    // attention output width (the o_proj heads) and its first local q head
    int ow_ = 0;
    // stats exchange (kv-pos-split): this card's receive rows [2 slots][2 peers][kChunk][6][2] and
    // flags [2 peers][2 slots], the peers' (set_xchg_peer), the pair's index in each card's buffers
    // (0 = the lower-ranked card of the pair writes there), the arrival counter, the exchange
    // counter of the launch unit, and the RoPE table of the step's rows.
    // (No cudaMemcpyPeerAsync on this path: a staged-copy variant deadlocked, the sender's host
    // thread blocked inside the peer copy while the receiver's GPU spun on the flag.)
    float* xbuf_ = nullptr;
    unsigned* xflags_ = nullptr;
    float* xpeer_buf_[kMaxLocalKv] = {};
    unsigned* xpeer_flags_[kMaxLocalKv] = {};
    int xridx_[kMaxLocalKv] = {}, xsidx_[kMaxLocalKv] = {};
    unsigned* xdone_ = nullptr;
    int xidx_ = 0;
    int pf_p0_ = -1;  // first position of the prefill chunk being run (attention_prefill picks int8 Q K^T from it)
    double2* rtab_ = nullptr;
    void xchg_fill(AttnShape& a);  // exchange pointers / sequence of the next layer
    void move_bytes(char* dst, const char* src, size_t len);
    int cur_slot_ = -1;
    Comm comm_;
    CommCounters ctr_;
    AttnShape ash_;
    DeltaShape dsh_;
    int dn_ch_ = 0;  // local conv channels

    std::vector<Layer> layers_;
    MtpLayer mtp_;
    int* dn_cur_ = nullptr;  // current delta-net state slot (device)
    int* dn_save_ = nullptr;
    int* dn_pn_ = nullptr;       // current slot's deferred-commit rows / stash (null: off)
    float* dn_pend_ = nullptr;
    size_t dn_pend_layer_floats() const { return (size_t)dn_pcap(dsh_) * dn_pend_row_floats(dsh_); }
    // the current slot's pending rows -> its state (stream-ordered; no-op when none / off)
    void dn_flush_cur(cudaStream_t st);
    size_t dn_slot_floats_ = 0;
    bf16 *mtp_in_ = nullptr, *mtp_h_ = nullptr;
    float* mtp_resid_ = nullptr;
    bf16* final_norm_ = nullptr;
    Lin lm_head_;
    Lin draft_head_;  // lm_head used for MTP drafts (INT4 by default)
    int* draft_map_ = nullptr;  // device: draft row -> token id (pruned vocabulary)
    int draft_v_ = 0;
    const bf16* embed_ = nullptr;
    float* inv_freq_ = nullptr;
    int* rope3_ = nullptr;  // [2 + 3 * kChunk]: multimodal chunk positions (AttnShape::rpos3)

    // activations
    float *resid_ = nullptr, *proj_ = nullptr, *partial_ = nullptr, *ab_ = nullptr;
    float *conv_out_ = nullptr, *dno_ = nullptr, *attn_work_ = nullptr;
    float *dn_qk_ = nullptr, *dn_gb_ = nullptr;
    float* pattn_work_ = nullptr;  // prefill attention split-K partials
    uint2* pattn_ksc_ = nullptr;   // prefill attention: per-key int8 K scales ([nkv][kPattnKscRows])
    float* pattn_kmu_ = nullptr;   // prefill attention: per-channel K sums ([nkv][256], K smoothing)
    PattnScratch pscr_;            // prefill attention: persistent windowed kernel scratch (in idle delta-net buffers)
    // per delta-net layer inputs kept by a verification pass for dn_replay
    std::vector<float*> spec_cv_, spec_qk_, spec_gb_;
    int* trim_acc_ = nullptr;  // accepted count of trim_slot's replay
    size_t pattn_work_floats_ = 0;
    float *pairs_ = nullptr, *pairs_all_ = nullptr;
    bf16 *xn_ = nullptr, *act_ = nullptr, *qbuf_ = nullptr;
    // fp16-accumulate GEMV (QE_GEMV_H16, gemv_h16.cuh): XH copies of the decode GEMV inputs, written by
    // the producers (xh_xn_: norm outputs, stride H; xh_act_: attention / DeltaNet / GLU outputs, stride
    // act_w; xh_tmp_: fallback packing of inputs without an XH producer)
    bool h16_ = false;
    __half *xh_xn_ = nullptr, *xh_act_ = nullptr, *xh_tmp_ = nullptr;
    int xh_act_ld_ = 0;
    // INT4 matmul of a decode input: the fp16-accumulate kernel on the XH copy (packed here when xh is
    // null) when enabled and m fits, else the bf16/fp32 path
    bf16* xn_bf16(bf16* xn, int m, const QW* next);
    static int dec_gemm_min() {
        static const int v = env_int("QE_DEC_GEMM", 33);
        return v;
    }
    bool dec_gemm_rows(int m) const { return dec_gemm_min() > 0 && m >= dec_gemm_min() && rows_cap_ > kMaxGemvW4Tokens; }
    bool dec_gemm_warm_ = false;
    void dec_gemm_warm(int M);
    void mm4(const W4& w, const bf16* x, int ldx, float* y, int ldy, int m, cudaStream_t st,
             const __half* xh = nullptr, int ldxh = 0, __half* yh = nullptr, int ldyh = 0) {
        // QE_DEC_GEMM = R (default 33, 0: off): batched decode passes of >= R rows run the prefill GEMM (W4A8
        // when QE_PREFILL_INT8 covers the class) instead of the H16 GEMV
        if (dec_gemm_rows(m)) {
            gemm_w4(w, x, ldx, y, ldy, m, st);
            return;
        }
        if (h16_ && m <= rows_cap_) {
            if (!xh) {
                xh_pack(x, ldx, w.k, m, xh_tmp_, w.k, st);
                xh = xh_tmp_;
                ldxh = w.k;
            }
            gemv_w4h(w, xh, ldxh, y, ldy, m, st, yh, ldyh);
        } else if (m <= rows_cap_) {  // decode rows without H16 (INT8 setups: the MTP head): the bf16 GEMV up
            gemv_w4(w, x, ldx, y, ldy, m, st);  // to kMaxBatchRows, never the GEMM (no buffers inside graph capture)
        } else {
            matmul(w, x, ldx, y, ldy, m, st);
        }
    }
    void mm4(const Lin& w, const bf16* x, int ldx, float* y, int ldy, int m, cudaStream_t st,
             const __half* xh = nullptr, int ldxh = 0, __half* yh = nullptr, int ldyh = 0) {
        if (w.is4) mm4(w.q, x, ldx, y, ldy, m, st, xh, ldxh, yh, ldyh);
        else matmul(w, x, ldx, y, ldy, m, st);
    }
    // a layer matrix: INT4 as above; INT8 reads the bf16 input (h16_ is off whenever w8_any_). Batched decode
    // passes of >= QE_DEC_GEMM rows take the W8A8 GEMM on matrices with K >= QE_DEC_GEMM8_MINK (4096: the
    // GEMM reads the weights once, 1.2-1.9x the two 32-row GEMV passes at 33..64 rows; the K = 3072 o / out
    // projections are ~10 % faster on the GEMV)
    static bool dec_gemm8(const W8& w) {
        static const int mink = env_int("QE_DEC_GEMM8_MINK", 4096);
        return w.k >= mink;
    }
    void mm4(const QW& w, const bf16* x, int ldx, float* y, int ldy, int m, cudaStream_t st,
             const __half* xh = nullptr, int ldxh = 0, __half* yh = nullptr, int ldyh = 0) {
        if (w.is8 && dec_gemm_rows(m) && dec_gemm8(w.q8)) gemm_w8(w.q8, x, ldx, y, ldy, m, st);
        else if (w.is8) matmul(w.q8, x, ldx, y, ldy, m, st);
        else mm4(w.q4, x, ldx, y, ldy, m, st, xh, ldxh, yh, ldyh);
    }
    int proj_ld_ = 0;
    int ll_idx_ = 0;      // LL collective counter within the current launch unit
    bool use_ll_ = true;
    int next_ll(int m) { return use_ll_ && m <= rows_cap_ ? ll_idx_++ : -1; }
    // decode-path row limit: 32 (kMaxGemvW4Tokens) for prompt chunks / single-sequence passes, as before the
    // wide batched steps; kMaxBatchRows inside forward_batch / mtp_batch (RowsCap guard)
    int rows_cap_ = kMaxGemvW4Tokens;
    struct RowsCap {
        int& r;
        int old;
        RowsCap(int& r_, int v) : r(r_), old(r_) { r = v; }
        ~RowsCap() { r = old; }
    };
    GraphCache<std::tuple<int, bool, int, int>> graphs_;      // (m, logits, mode, slot)
    std::set<std::tuple<int, bool, int, int>> eager_seen_;    // short prefill passes run without a graph once
    GraphCache<std::tuple<int, int, int, int>> spec_graphs_;  // (kv, kd, mode, slot)
    GraphCache<std::string> batch_graphs_;
    // a new batched-step graph of the same sequences / modes as a cached one (other draft or lookup rows) updates that
    // graph in place (cudaGraphExecUpdate) instead of a new instantiation when the topology matches (QE_GRAPH_UPDATE)
    cudaGraphExec_t blaunch_[2] = {nullptr, nullptr};  // the last two launched (maybe in flight): never updated
    bool batch_update(const std::string& key, cudaGraph_t g, cudaGraphExec_t* out);
    long long bupd_ok_ = 0, bupd_fail_ = 0;
    void instantiate(cudaGraphExec_t* ge, cudaGraph_t g);
    std::vector<void*> allocs_;  // everything from dalloc, freed by the destructor
    int *bout_ = nullptr, *bmout_ = nullptr;  // per-row argmax of a batched step / its MTP pass
    void forward_batch(const std::vector<int>& slots, const std::vector<int>& r0, const std::vector<int>& ms);
    void mtp_batch(const std::vector<int>& slots, const std::vector<int>& r0, const std::vector<int>& ms,
                   const std::vector<const bf16*>& hsrc);
    SampleParams* sp_ = nullptr;
    float *topk_ = nullptr, *topk_all_ = nullptr;
};
