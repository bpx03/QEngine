#pragma once
// Pipeline-parallel driver of the Flash-Next engine (FLASH_PAR=pp / pp_prefill).
// N >= 2 stage workers (Plan::pp, one per card, whole layers each) run as a pipeline:
//  - prefill: the prompt is cut into micro-batches; stage s runs batch j once stage s - 1 handed it over (the residual
//    rows [b][10240] fp32 through pinned host buffers, double-buffered per boundary), so all stages work at once after
//    the fill; each stage's KV / recurrent state stays on its card;
//  - decode (mode B): a step runs through the stages in order; several sequences (slots) circulate so every stage
//    works on a different one (interleave), the last stage verifies, drafts (MTP) and sends the next step to stage 0;
//  - mode A (pp_prefill): after a pipelined prefill the prompt's KV and recurrent state move to the TP workers'
//    layout (pp_handover) and decode runs TP as today.
// Every stage worker's CUDA work runs on its own host thread (StageThreads).
// PP x TP (FLASH_PAR=pptp): a stage may be a TP group of t ranks (Plan::ngpu = t, the stage's
// layers split TP-wise, all-reduces inside the group on its own comm). A stage task then runs its worker calls on all t
// ranks at once (RankPool: rank 0 on the stage thread, the others on helper threads; the collectives need every rank's
// kernels in flight), results come from rank 0. The residual rows go from rank i of a stage to the ranks j of the next
// with j % t_s == i (P2P: one copy per receiver into its receive buffer; host: one pinned buffer per sender).
#include <atomic>
#include <condition_variable>
#include <deque>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

#include "flash.h"
#include "pp_plan.h"

namespace flash {

// One worker thread per stage with a FIFO of tasks.
class StageThreads {
  public:
    explicit StageThreads(int n);
    ~StageThreads();
    void post(int s, std::function<void()> fn);
    void run_on(int s, const std::function<void()>& fn);  // fn on stage s's thread, after what is queued there; waits for fn only
    void wait(int s);   // until stage s ran everything posted so far (and anything posted meanwhile)
    void wait_all();
    int size() const { return (int)q_.size(); }

  private:
    struct Q {
        std::mutex mu;
        std::condition_variable cv, idle;
        std::deque<std::function<void()>> fns;
        std::atomic<int> nq{0};  // tasks queued (polled without the lock while spinning)
        bool busy = false, quit = false;
        std::thread th;
    };
    std::vector<Q*> q_;
};

// Runs fn(r) for r in [0, t) at once: r = 0 on the calling thread, the others on helper threads (one per rank), and
// waits for all of them. t = 1: a plain call.
class RankPool {
  public:
    explicit RankPool(int t);
    ~RankPool();
    void run(const std::function<void(int)>& fn);
    int size() const { return t_; }

  private:
    int t_;
    std::mutex mu_;
    std::condition_variable cv_, dn_;
    const std::function<void(int)>* fn_ = nullptr;
    long long gen_ = 0;
    std::atomic<int> left_{0};
    bool quit_ = false;
    std::atomic<long long> agen_{0};
    std::vector<std::thread> th_;
};

class PpXfer;
class PpEngine {
  public:
    // workers: every rank of every stage, stage by stage (rank order inside a stage); stage s holds layers
    // [plan.l0, plan.l1). Mode B: one worker per stage. max_rows: the largest batch any call hands over
    PpEngine(std::vector<Worker*> workers, int max_rows);
    ~PpEngine();
    int nstage() const { return (int)W_.size(); }
    Worker* stage(int s) const { return W_[s]; }         // rank 0 of stage s
    int ranks(int s) const { return (int)G_[s].size(); }
    Worker* rank(int s, int r) const { return G_[s][r]; }
    int flat(int s, int r) const { return base_[s] + r; }  // index in the workers given to the constructor
    int nworkers() const { return base_.back() + ranks(nstage() - 1); }
    StageThreads& threads() { return th_; }
    // run fn(worker index, worker) for every worker (each stage's ranks at once, on the stage's thread and its rank
    // helpers) and wait for these calls (not for work posted after them)
    void all(const std::function<void(int, Worker*)>& fn);

    // ---- prefill: the prompt [0, np) of the bound slot in micro-batches of b rows (b <= every stage's pf_batch()).
    // ngram(t0, m, out) writes the n-gram rows of positions [t0, t0 + m) (PLE input; called once per batch, on the
    // first stage's thread). mtp: the last stage also fills the MTP KV and returns the first draft.
    struct PfOut {
        int next = -1, draft = -1, batches = 0;
        double wall_ms = 0, fill_ms = 0;   // fill: until the last stage started its first batch (the pipeline bubble)
    };
    PfOut prefill(const int* prompt, int np, int b, const std::function<void(int, int, bf16*)>& ngram, bool mtp);
    // several prompts at once (cross-request pipelining), each on its slot / KV region; reset: the slot's state zeroed
    // before its first batch. One result per job (wall_ms: from the call to the job's last batch).
    struct PfJob {
        int slot = 0, base = 0;
        const int* prompt = nullptr;
        int np = 0;
        bool mtp = false, reset = false;
        std::function<void(int, int, bf16*)> ngram;
        int start = 0;     // the prompt's rows [start, np) (the state before start is already on the stages)
        int nknown = 0;    // tokens known past np (the MTP fill of the last row reads prompt[np] when np < nknown); 0: np
        class PpXfer* xf = nullptr;  // overlapped hand-over: each stage's KV rows of a batch to the TP ranks right after it
    };
    std::vector<PfOut> prefill_many(std::vector<PfJob>& jobs, int b);

    // ---- decode (host-driven): one forward of m <= MAXP rows at pos through every stage, the argmax of each row from
    // the last stage into out; commit: rows of the previous forward of the bound slot that become state (as
    // Worker::forward). seq: hand-off buffer set (one per sequence in flight, < max_seqs).
    // slot >= 0: every stage binds slot (KV region base) first, in the same task (the server's prefill flow re-binds them)
    void forward(const int* toks, int pos, int m, const bf16* ple, int* out, int commit, int seq = 0, int slot = -1, int base = 0);
    // MTP head on the last stage (Worker::forward_mtp)
    int mtp(const int* toks, int pos, int m, bool chain, int slot = -1, int base = 0);
    // the first k drafts after a prompt of slot (KV region base) that ended at position n - 1: d0 (the prefill's draft),
    // then the MTP chain from the slot's saved chain row (rmin_load), all in one task of the last stage
    std::vector<int> mtp_chain(int slot, int base, int d0, int n, int k);

    // ---- interleaved decode (mode B, several sequences): each sequence is bound to its own slot / KV region on every
    // stage; up to n + 1 of them are in flight (pp_plan.h decode_ngroups), stage s runs one while stage s + 1 runs the
    // one before it. Greedy, MTP drafts (k) or plain steps (k = 0); stops a sequence at gen tokens or EOS.
    struct Seq {
        int slot = 0, base = 0;
        std::vector<int> hist;      // prompt + emitted tokens (n-gram rows, positions)
        int nxt = -1;               // pending token (emitted, not yet forwarded)
        std::vector<int> drafts;    // MTP drafts for the next step
        int commit = 0;             // rows of the previous forward to commit
        int pos = 0;                // position of the pending token
        std::vector<int> out;       // emitted tokens
        int gen = 0;                // tokens to emit
        int steps = 0;
        bool done = false;
    };
    struct DecodeStats { double wall_s = 0; long long tokens = 0, steps = 0; int groups = 0; };
    // host-driven variant: one sequence per stage task, single-sequence forwards (kept for comparison, FLASH_PP_DECODE=host)
    DecodeStats decode_interleaved(std::vector<Seq>& seqs, int k, const std::function<void(const std::vector<int>&, int, bf16*)>& ngram_row);
    // batched variant (default): the sequences in min(c, N + 1) groups (pp_plan.h decode_groups); a stage task runs one
    // group as one batched step (Worker::batch_launch over the group's slots, its layers only); the last stage accepts,
    // drafts (MTP chain) and advances the per-slot speculative state in the same graph; its state rows go to the other
    // stages before the group's next step. Every sequence must be prepared on every stage (spec_init on its slot).
    // bases: KV region (period) of every slot of the workers.
    DecodeStats decode_batched(std::vector<Seq>& seqs, int k, const std::vector<int>& bases);

    void set_max_seqs(int n);  // hand-off buffer sets for decode (default 1)

    // ---- server mode B (flash_capi.cpp, FLASH_PAR=pp): prompts flow through the stage queues as micro-batch tasks next to
    // the decode steps, so a decode step waits at most for one micro-batch per stage; the server polls the jobs.
    struct SrvJob {
        int slot = 0, base = 0;
        std::vector<int> ids;          // the whole prompt, content ids (n-gram rows; a text prompt: what the devices embed)
        std::vector<int> dev, r3;      // image prompt: device ids (image rows -> rows of the image table), (t, h, w) per row
        int i0 = 0, n = 0;             // its rows [i0, n) go through the pipeline; the last row is sampled (next) and drafted
        bool mtp = false;
        std::vector<int> cuts;         // micro-batch ends, ascending, the last = n (checkpoint positions are cuts)
        std::vector<int> ck;           // checkpoint positions (cuts): ck_fn(k, stage, worker) on the stage's thread right
        std::function<void(int, int, Worker*)> ck_fn;  // after it ran the micro-batch that ends at ck[k] (the worker's index:
                                                       // every rank of the stage, PP x TP)
        std::function<void(int, int, bf16*)> ngram;    // n-gram rows of positions [t0, t0 + m)
        const unsigned* mask0 = nullptr;  // grammar-constrained slot: mask row 0 of its first token (the last row's head)
        // results (read after done)
        int next = -1, draft = -1;
        double t_submit = 0, t_done = 0;
        std::atomic<bool> done{false}, cancel{false};
        std::atomic<bool> nf{false};   // the prompt's last row had non-finite values (Worker::nf_check on the last stage)
    };
    // queue the job's micro-batches (in submission order after the jobs already queued); the job object must live until done
    void srv_submit(SrvJob* j);
    // wait until job j is done or ms passed (true: done)
    bool srv_wait(SrvJob* j, double ms);
    int srv_inflight() const;  // micro-batches queued and not finished on the last stage
    // micro-batch rows for a job: the planner's (pp_microbatch) capped by the batch buffers of every stage at the job's
    // context end, and by the receive buffers on peer-to-peer boundaries
    int srv_cap(int base, int n) const;
    // decode: the slots in min(n, N + 1) groups, each group one batched step (k drafts) per round through every stage, up to
    // `rounds` rounds; after a group's step on the last stage cont(i, tokens, cnt, &lim) gets slot i's tokens and returns
    // whether it steps again (lim: its next accept limit). The per-slot speculative state of the last stage is relayed to
    // the other stages before each step (relay cache per slot; relay_drop after a host-side spec_init on every stage).
    // con: a grammar-constrained slot stepping alone (slots = {con->slot}): its planned drafts go into its speculative
    // row on every stage before the step, its mask rows to the last stage (the head applies them)
    struct SrvCon { const int* drafts = nullptr; int nd = 0; const unsigned* mask = nullptr; int mrows = 1; };
    void srv_decode(const std::vector<int>& slots, const std::vector<int>& bases, std::vector<int> lims, int k, int rounds,
                    const std::function<bool(int, const int*, int, int*)>& cont, const SrvCon* con = nullptr,
                    std::vector<char>* bad = nullptr);  // bad[i]: slot i's rows had non-finite values
    void relay_drop(int slot);
    // the last stage's speculative row of slot after its last step (relay cache; false: none since a host-side init)
    bool relay_row(int slot, std::vector<int>& row) const;

    // ---- server mode B, persistent decode (no prompt in flight): the groups keep stepping between the server's calls,
    // so the pipeline never drains at a call boundary. Each slot's steps are buffered until the caller takes them; a
    // slot runs at most `ahead` steps past what was taken, stops at an end token, its limit, a non-finite row, or when
    // its next step would pass `room` (its region / the context), and parks while its buffer is full. Groups (N + 1 at
    // most) keep their members across calls: a slot joins the group with the fewest members, a slot the caller no
    // longer steps leaves after its step in flight. Every other use of the stages (all(), forward, srv_decode,
    // srv_submit, mtp_chain) stops the groups first through on_touch (the server's settle: buffered steps into the
    // slots' histories).
    struct BgReq { int slot = 0, lim = 0, rem = -1, ahead = 1, room = 0, pos = 0; };
    struct BgTake { std::vector<int> toks, steps, lims; bool bad = false, stopped = false; };
    // start / update the members (the caller's slots of this call), launch idle groups; k: drafts per step (a change of k
    // or of a member's base needs bg_stop first); false: refused (k or bases differ while members exist)
    bool bg_run(const std::vector<BgReq>& req, int k, const std::vector<int>& bases);
    // wait until slot has a step buffered (or is stopped / not a member) and take its buffer; hold: the slot's next step
    // waits for the next bg_run (a stop string may be near: the caller checks first), else it goes on at once
    BgTake bg_take(int slot, bool hold);
    int bg_pos(int slot) const;      // the slot's position on the devices (-1: not a member)
    bool bg_active() const;          // members exist
    // stop every group (steps in flight finish) and hand every member's buffer to fn; members cleared
    void bg_stop(const std::function<void(int, BgTake&)>& fn);
    std::function<void()> on_touch;  // set by the server: called before any other use of the stage threads
    struct BgStats { long long calls = 0, steps = 0, parks = 0, stops = 0, joins = 0, leaves = 0; };
    BgStats bg_stats() const;

  private:
    std::vector<Worker*> W_;                   // rank 0 of every stage
    std::vector<std::vector<Worker*>> G_;      // every rank of every stage
    std::vector<int> base_;                    // flat index of each stage's rank 0
    std::vector<RankPool*> rp_;                // per stage (t > 1)
    StageThreads th_;
    // fn(r, worker) on every rank of stage s at once (on the stage's thread: call it from a stage task)
    void each(int s, const std::function<void(int, Worker*)>& fn);
    // rank r of stage s: its hop buffers (pf: prefill buffer parity idx; else decode buffer set idx)
    void io(int s, int r, Worker* w, bool pf, int idx);
    // persistent decode state (bg_*), under bmu_
    struct BgSlot {
        bool member = false, park = false, leave = false, stop = false, inflight = false, bad = false, room_stop = false;
        int lim = 0, rem = -1, ahead = 1, room = 0, pos = 0, group = -1;
        std::vector<int> toks, steps, lims;
    };
    struct BgGroup { std::vector<int> mem, cur, curlim; bool running = false; std::vector<int> out, cnt; };
    mutable std::mutex bmu_;
    std::condition_variable bcv_;
    std::vector<BgSlot> bs_;
    std::vector<BgGroup> bg_;
    std::vector<int> bbases_;
    int bk_ = -1;
    bool bstop_ = false;
    BgStats bst_;
    // FLASH_PPB_PROF=1: per stage task count, task wall and GPU time, rows (printed at bg_stop)
    std::vector<double> pr_wall_, pr_gpu_, pr_rows_;
    std::vector<long long> pr_n_;
    double pr_t0_ = 0;
    void bg_launch(int gi);          // stage 0 entry of group gi (posts the stage-0 task)
    void bg_task(int s, int gi);
    void touch() { if (on_touch) on_touch(); }
    // server prefill flow: one global order of micro-batches over every job (double-buffered boundaries as prefill_many)
    struct Mb { SrvJob* job; int i, m; bool final; };
    mutable std::mutex fmu_;
    std::condition_variable fcv_;
    std::deque<Mb> fq_;
    long long fq0_ = 0;                  // global index of fq_.front()
    std::vector<long long> fdone_, fposted_;
    int sple_ = 0;
    int fhold_ = 0;                      // decode steps in progress: no new micro-batch starts (srv_decode)
    double fblock_until_ = 0;            // decoders' share: no micro-batch starts before this time (ms)
    double fwave_ms_ = 0;                // a stage-0 micro-batch's time (moving average)
  public:
    void srv_kick();                     // retry the posts (the decoders' share may have ended)
  private:
    void srv_try_post(int s);            // under fmu_
    void srv_run(int s, long long g);
    std::vector<int> rel_;               // relay cache [slot][SPEC_ROW]
    std::vector<char> rel_ok_;
    int max_rows_ = 0;
    // per boundary b (stage b -> b + 1, ranks ta -> tb): the rows go peer-to-peer into the receive buffers of every rank
    // of stage b + 1 (pp_rx, one copy each; receiver j from sender j % ta) when all of them have one (peer access,
    // FLASH_PP_P2P), else through pinned host buffers, one per sender (D2H + H2D; receiver j reads sender j % ta's)
    struct Bd { bool p2p = false; int ta = 1, tb = 1; int nbuf() const { return p2p ? tb : ta; } };
    std::vector<Bd> bd_;
    std::vector<std::vector<void*>> pf_buf_;   // [boundary * 2 + parity][nbuf] prefill rows (host: pinned, P2P: pp_rx)
    std::vector<std::vector<void*>> dc_buf_;   // [group * (N - 1) + boundary][nbuf] decode rows
    int dc_seqs_ = 0;
    const std::vector<void*>& pf(int boundary, int par) const { return pf_buf_[(size_t)boundary * 2 + par]; }
    const std::vector<void*>& dc(int seq, int boundary) const { return dc_buf_[(size_t)seq * (W_.size() - 1) + boundary]; }
  public:
    bool p2p(int boundary) const { return bd_[boundary].p2p; }
};

// Mode A hand-over: the bound slot's prompt [0, n) from the stage workers (stage layout: every position on the card
// holding the layer) to the TP workers (slot `tp_slot`, KV region at period `tp_base`, TP position ownership): QSA K / V /
// scales / pooled indexer keys of every QSA layer (+ the MTP block's), the indexer key rings, DeltaNet S + conv history
// (by value / key heads), the PLE history. TP workers: claim their KV rows first (Worker::kv_claim), state reset.
// Returns the bytes moved.
size_t pp_handover(const std::vector<Worker*>& stages, const std::vector<Worker*>& tp, int n, bool mtp);

// The general transfer between the stages' layout and the TP ranks' (both bound to their slots / regions): KV rows of a
// position range and the recurrent state, in either direction (dir 0: stages -> TP, the hand-over; 1: TP -> stages, a
// reused prefix of a TP slot before a pipelined prefill continues it). Copies run on one stream per (TP rank, stage);
// after_stage(s) orders the next copies of stage s after its stream's work so far (overlap: a micro-batch's KV leaves
// while the next batches compute). wait() drains them. The caller runs pp_xfer_prepare first.
class PpXfer {
  public:
    PpXfer(const std::vector<Worker*>& stages, const std::vector<Worker*>& tp);
    ~PpXfer();
    void kv(int t0, int t1, bool mtp, int dir, int only_stage = -1);
    void state(bool mtp, int dir);
    void after_stage(int s);
    void wait();
    size_t moved() const { return moved_; }

  private:
    std::vector<Worker*> S_, T_;
    std::vector<long long> sb_, tb_;  // the bound regions' first rows (snapshot at construction: the TP ranks may bind
                                      // other slots while overlapped copies run)
    std::mutex mu_;                   // kv() from several stage threads
    std::vector<cudaStream_t> ss_;  // [rank][stage]
    std::vector<cudaEvent_t> ev_;   // per stage
    std::atomic<size_t> moved_{0};
    const char* what_ = "";
    int what_li_ = -1;
    void cp(size_t r, int s, void* tpp, const void* stp, size_t bytes, int dir);
    void cp2d(size_t r, int s, char* tpp, size_t tpitch, const char* stp, size_t spitch, size_t width, size_t height, int dir);
};
void pp_xfer_prepare(const std::vector<Worker*>& stages, const std::vector<Worker*>& tp, int n);

// set-up (flash_run, the server): stage bounds (FLASH_PP_LAYERS or the planner; empty + *err on a bad value), the peer
// copy integrity of a card pair, the stage workers (one per card, loaded in parallel; views: mode A shared expert views
// of these TP workers, one per card in order), the prefill micro-batch for a prompt of np tokens
std::vector<int> pp_bounds(int n, long long np, bool decode, std::string* err);
// heterogeneous cards: per-card speeds (relative to the median card) and free VRAM from the boot calibration; pp_bounds
// then splits by them when the cards differ (equal cards keep the plain split)
void pp_set_cards(const std::vector<flash_pp::CardSpeed>& c);
const std::vector<flash_pp::CardSpeed>& pp_cards();
// calibrate the cards (card_probe.cu; FLASH_TEST_SLOW_CARD first), set them for pp_bounds, return the log line
std::string pp_calibrate(const std::vector<int>& devs);
bool pp_p2p_ok(int a, int b);
std::vector<Worker*> pp_make_stages(const std::vector<int>& devs, const std::vector<int>& bounds, int ctx, const SafeTensors& st,
                                    const HostTables& ht, const SafeTensors* mst, const std::vector<std::vector<int>>& cold,
                                    const std::vector<float>& prior, const std::vector<int>& draft_ids, const NgramDev& ngd,
                                    const std::vector<const Worker*>& views = {}, int max_seqs = 0);
// PP x TP: the workers of every rank of every stage (groups[s] = the devices of stage s, its ranks in order), stage by
// stage; each group of t > 1 gets its own comm (P2P where the cards allow, the TP exchange path probe) and its ranks
// the peers' exchange buffers. comms: the groups' comms (kept by the caller for the run)
std::vector<Worker*> pp_make_groups(const std::vector<std::vector<int>>& groups, const std::vector<int>& bounds, int ctx,
                                    const SafeTensors& st, const HostTables& ht, const SafeTensors* mst,
                                    const std::vector<std::vector<int>>& cold, const std::vector<float>& prior,
                                    const std::vector<int>& draft_ids, const NgramDev& ngd, int max_seqs = 0);
// the C API's start-up decisions for the pipeline planner, each read through the environment variable when the user
// set it (variables, not setenv: issue #11). The first probe of the process stays. The host link in GB/s:
void pp_set_h2d_probe(double gbps);
double pp_h2d_gbps(double def);  // FLASH_PP_H2D_GBPS, else the probe, else def
void pp_set_no_p2p();            // QE_P2P=0: the stages' hop through pinned host memory too
bool pp_hop_p2p();               // FLASH_PP_P2P, else not after pp_set_no_p2p
// the layout of every worker (stage, rank, layers, expert groups per layer, KV period / ownership, VRAM, host-backed
// experts) as log lines (remote diagnosis: 4+ card and PP x TP boots, FLASH_BOX_REPORT=1)
std::string layout_report(const std::vector<Worker*>& W);
bool box_report_wanted(const std::vector<int>& devs, bool pptp);
// PP x TP stage bounds: FLASH_PP_LAYERS, else the planner's split by each group's speed (pp_plan.h group_speeds)
std::vector<int> pptp_bounds(const std::vector<std::vector<int>>& groups, long long np, std::string* err);
// the groups for FLASH_PAR=pptp from FLASH_PP_MAP / FLASH_PP_TP (default 2) over devs (empty + *err when malformed)
std::vector<std::vector<int>> pptp_groups(const std::vector<int>& devs, std::string* err);
// wall: the model's prefill time; ctx0: the positions before the rows (a suffix after a cached prefix: its attention reads
// them, and the batch buffers are sized at the context end)
// ux: the compute term's scale (the server's mode B: the measured stage times, see ppb_submit)
// S: every worker (PP x TP: every rank; a stage's cost model is its rank 0's share x its TP speed-up)
int pp_microbatch(const std::vector<Worker*>& S, const std::vector<int>& b, long long np, double* wall_ms = nullptr, long long ctx0 = 0,
                  double ux = 1.0, int lo = 0);  // lo: the smallest rows considered (0: FLASH_PP_MB_MIN, 2048)

}  // namespace flash
