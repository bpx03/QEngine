// C API of the GLM-5.3-Flash engine (ctypes / server)
#include <vector>
#include <cuda_profiler_api.h>

#include "glm.h"
#include "glm_cpu.h"

using glm::Engine;
extern "C" {
void* glm_create(const char* dir, int ctx) {
    Engine* e = new Engine();
    e->load(dir, ctx);
    return e;
}
// T tokens at the current position. am (T ints, optional): argmax per row. nl rows (lrows) of full logits into lout
// [nl][VOCAB]; nl = -1: logits of the last row into lout.
int glm_forward(void* h, const int* tok, int T, int* am, int nl, const int* lrows, float* lout) {
    Engine* e = (Engine*)h;
    std::vector<int> a;
    std::vector<int> lr;
    if (nl > 0) lr.assign(lrows, lrows + nl);
    e->forward(tok, T, nl ? lout : nullptr, am ? &a : nullptr, nl > 0 ? &lr : nullptr);
    if (am) for (int i = 0; i < T; i++) am[i] = a[i];
    return e->pos;
}
void glm_reset(void* h) { ((Engine*)h)->reset(); }
int glm_pos(void* h) { return ((Engine*)h)->pos; }
void glm_set_dump(void* h, const char* dir) { ((Engine*)h)->dump_dir = dir ? dir : ""; }
// decode one token (greedy), returns the next token
int glm_decode(void* h, int tok) { return ((Engine*)h)->decode(tok); }
// round 15 (decode-path quality gate): decode one token and copy row 0's full logits (VOCAB floats) to out
int glm_decode_lg(void* h, int tok, float* out) {
    Engine* e = (Engine*)h;
    const int o = e->decode(tok);
    for (auto& w : e->W) {
        CUDA_CHECK(cudaSetDevice(w->dev));
        CUDA_CHECK(cudaMemcpy(out + w->p.v0, w->logits, (size_t)(w->p.v1 - w->p.v0) * 4, cudaMemcpyDeviceToHost));
    }
    CUDA_CHECK(cudaSetDevice(e->W[0]->dev));  // review r15 L6
    return o;
}
// cache stats: hits, misses, admitted, cpu-lane experts
void glm_stats(void* h, unsigned long long* o) {
    Engine* e = (Engine*)h;
    o[0] = e->st_hit; o[1] = e->st_miss; o[2] = e->st_admit; o[3] = e->st_cpu;
    o[4] = (unsigned long long)e->st_cpu_us; o[5] = (unsigned long long)e->st_wait_us;
    o[6] = e->lane ? e->lane->pbytes.load() : 0;
    o[7] = (unsigned long long)e->hxl.stride;
}
// extra counters: CPU-lane expert-rows, admitted expert-rows
void glm_stats2(void* h, unsigned long long* o) {
    Engine* e = (Engine*)h;
    o[0] = e->st_cpu_rows; o[1] = e->st_adm_rows;
}
void glm_profile(void* h, int on) {
    Engine* e = (Engine*)h;
    e->run_all([&](glm::Worker& w) { on ? cudaProfilerStart() : cudaProfilerStop(); });
}
// verify m rows (last token + drafts) -> argmax per row (out[m]); then commit the first a rows
void glm_verify(void* h, const int* toks, int m, int* out) { ((Engine*)h)->verify(toks, m, out); }
void glm_commit(void* h, int a) { ((Engine*)h)->commit(a); }
// greedy generation with k MTP chain drafts per step; returns the number of tokens written to out
int glm_generate(void* h, const int* prompt, int T, int n, int k, int* out) { return ((Engine*)h)->generate(prompt, T, n, k, out); }
// two sequences at once (GLM_SEQS=2 at load): got[2] = tokens of each; returns the total
int glm_generate2(void* h, const int* pa, int ta, const int* pb, int tb, int n, int k, int* outa, int* outb, int* got) {
    return ((Engine*)h)->generate2(pa, ta, pb, tb, n, k, outa, outb, got);
}
// spec stats: steps, rows, accepted tokens, accepted drafts at depth 1..3
void glm_spec_stats(void* h, unsigned long long* o) {
    Engine* e = (Engine*)h;
    o[0] = e->st_steps; o[1] = e->st_rows; o[2] = e->st_acc;
    for (int i = 0; i < 3; i++) o[3 + i] = e->st_dacc[i];
    o[6] = (unsigned long long)e->st_dec_us;
    o[7] = (unsigned long long)e->st_ver_us;
    o[8] = (unsigned long long)e->st_mtp_us;
}
int glm_vocab() { return glm::VOCAB; }
void glm_set_sampling(void* h, float T, float top_p, int top_k, float min_p, unsigned long long seed) {
    ((Engine*)h)->set_sampling(T, top_p, top_k, min_p, seed);
}
// teacher-forced forward of T tokens that also writes the final normed hidden of every row (fp32 [T][4096])
void glm_forward_hidden(void* h, const int* tok, int T, float* hid, int* am) {
    Engine* e = (Engine*)h;
    e->hid_dump = hid;
    std::vector<int> a;
    e->forward(tok, T, nullptr, am ? &a : nullptr);
    e->hid_dump = nullptr;
    if (am) for (int i = 0; i < T; i++) am[i] = a[i];
}
// MTP routing counts of the draft passes (GLM_MTP_COUNT=1), 288 values
void glm_mtp_counts(void* h, unsigned long long* o) {
    Engine* e = (Engine*)h;
    CUDA_CHECK(cudaSetDevice(e->W[0]->dev));
    CUDA_CHECK(cudaMemcpy(o, e->W[0]->mtp_cnt, glm::NE * 8, cudaMemcpyDeviceToHost));
}
// steer the MTP router to n experts (ids); n <= 0: all
void glm_mtp_mask(void* h, int n, const int* ids) {
    Engine* e = (Engine*)h;
    std::vector<int> r;
    if (n <= 0) for (int i = 0; i < glm::NE; i++) r.push_back(i);
    else r.assign(ids, ids + n);
    e->mtp_mask(r);
}
}

// ---------------------------------------------------------------- server C API (server/qe_capi.h tier 0)
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <exception>
#include <mutex>
#include <thread>
extern "C" {
static std::vector<int> int_list(const char* s) {
    std::vector<int> v;
    for (const char* q = s ? s : ""; *q;) {
        while (*q == ' ' || *q == ',') q++;
        if (!*q) break;
        v.push_back(atoi(q));
        while (*q && *q != ',') q++;
    }
    return v;
}
void* qe_create(const char* model_dir, const char* devices, int max_t, const char* split) {
    // [glm-rel-host] the server's context (default 262,144; GLM_MAX_CTX caps it), cards and split; the engine lowers the
    // context only when its MLA cache does not fit (logged)
    int cap = 262144;
    if (const char* c = getenv("GLM_MAX_CTX")) cap = atoi(c);
    const int ctx = std::max(1024, std::min(max_t, cap));
    if (ctx != max_t) fprintf(stderr, "[glm] context %d (asked %d; GLM_MAX_CTX / the 262,144 cap)\n", ctx, max_t);
    Engine* e = new Engine();
    e->want_devs = int_list(devices);
    e->want_ecols = int_list(split);
    e->load(model_dir, ctx);
    // round 25: one dummy prefill at load, so the first request does not pay the first-use loads of the cuBLAS / CUDA
    // modules of the prefill kernels (lazy loading: 32K first 8.65 s vs 8.29 s for a second one; CUDA_MODULE_LOADING=EAGER
    // costs ~0.8 GB per card = -4.5 % slots). GLM_WARM = rows (0 = off); the cache goes back to the seed after it
    const int warm = getenv("GLM_WARM") ? atoi(getenv("GLM_WARM")) : 4096;  // 4K covers 32K: 8.32 vs 8.30 s (r25 wm0)
    // re-review r25: a card set that only just loads skips it (the lend of the warm window must fit the slots)
    const bool warm_fits = warm > 0 && e->lend_slots(e->pf_window(warm)) < e->lend_top() / 2;
    if (warm > 0 && !warm_fits) fprintf(stderr, "[glm] warm-up prefill skipped (few expert slots)\n");
    if (warm_fits && warm < e->ctx) {
        std::vector<int> t((size_t)warm);
        for (int i = 0; i < warm; i++) t[i] = 1000 + (int)((i * 7919LL) % 100000);
        e->forward(t.data(), warm, nullptr, nullptr);
        e->reset();
        e->sp_hist.clear();
    }
    return e;
}
void qe_destroy(void* h) { (void)h; }  // the process exits; pinned host copy and threads go with it
int qe_max_context(void* h) { return ((Engine*)h)->ctx; }
// ---- slots (GLM_SEQS=2, opt-in): one sequence bank per slot; the pair step verifies the rows of both slots in one
// step (greedy); a sampled slot steps on its own. Prefill is whole-prompt (prefill_more completes it at once).
}
namespace {
struct SlotS {
    bool active = false, pending = false;
    std::vector<int> ids;
    int k = 0, first = -1, n = 0, hint = 0;
    float T = 0.f, top_p = 1.f, min_p = 0.f;
    int top_k = 0;
    int limit = -1;  // round 25: tokens the reply may still emit (qe_set_limit), -1 none
    bool ign_eos = false;  // review r25-M2: ignore_eos reply (qe_set_ignore_eos): its steps do not stop at an EOS
    std::mt19937_64 rng{1};
    unsigned long long use = 0;
    // release track [glm-rel-mask]: grammar group (masks [8][MASK_WORDS], row j = allowed after drafts 1..j; io: [0]
    // model drafts n, [1..] drafts, [16] planned drafts n, [17..] them), logprobs (raw logits of the last step's rows,
    // or of the prompt end after a start); [glm-rel-stops] stop strings watched (the step keeps its undo)
    bool constrained = false, capture = false, watch = false;
    std::vector<uint32_t> mask;
    int io[32] = {};
    std::vector<float> lg;  // [rows][VOCAB]
    int lg_rows = 0;
    std::vector<float> tmp;
    // [glm-rel-slice] virtual slots (nseq 1): a reply not bound to the engine is parked (its whole state saved);
    // pend_tok: the first token of a re-prefilled reply (its parked rows were lost), returned by its next step
    bool parked = false;
    glm::Engine::ConvPark pk;
    int pend_tok = -1;
    bool dead = false;
    double t_bound = 0, t_run = 0;
};
constexpr int MASK_WORDS = (glm::VOCAB + 31) / 32;
struct Slots {
    std::vector<SlotS> s;
    SlotS next;  // sampling of the next start (qe_set_sampling before qe_start3)
    int cur = 0, last = 0, newest = -1;
    unsigned long long clock = 0;
    bool virt = false;   // [glm-rel-slice] GLM_CONVS virtual slots time-sliced on one engine sequence
    int want = -1;       // the slot qe_resume asked for (qe_spec_step binds it if that was not possible then)
    // [glm-rel-slice-cost] the cost of one preemption = a park now + a resume later, each an exponential mean (weight 0.2) of
    // its own events (a new prompt's switch is a park alone, ~60 ms: one mean over all switches put the quantum near 2 s and
    // short replies were preempted near their end); the resume mean starts from 500 ms (measured 0.3-0.6 s)
    double park_ms = 0, res_ms = 500.0;
    unsigned long long npark = 0, nres = 0;
    double sw_ms = 500.0;  // park_ms + res_ms
};
// [glm-rel-slice] conversations served at once (default 8; 1 = one at a time, the old behaviour). Only with one engine
// sequence (GLM_SEQS=2 keeps its banks)
int conv_slots(Engine* e) {
    if (e->nseq != 1) return e->nseq;
    static const int v = std::max(1, std::min(64, getenv("GLM_CONVS") ? atoi(getenv("GLM_CONVS")) : 8));
    return v;
}
Slots& SL(Engine* e) {
    static Slots x;
    const int n = conv_slots(e);
    if ((int)x.s.size() != n) {
        x.s.assign(n, SlotS{});
        x.virt = e->nseq == 1 && n > 1;
    }
    return x;
}
double now_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
void take(SlotS& d, const SlotS& src) {
    d.T = src.T; d.top_p = src.top_p; d.top_k = src.top_k; d.min_p = src.min_p; d.rng = src.rng;
}
void apply(Engine* e, const SlotS& q) {
    e->set_sampling(q.T, q.top_p, q.top_k, q.min_p, 1);
    e->srng = q.rng;
    e->sp_watch = q.watch;
}
// [glm-rel-mask] the token of verify row j (or of the prompt end, j = 0) under mask row j: greedy = the allowed argmax
// (the engine's tie rule: lowest id), sampled = the slot's distribution renormalised over the allowed tokens. x is the
// row's full logits (masked in place); the raw row is kept first when logprobs are captured. -1: nothing allowed
int pick_masked(Engine* e, SlotS& q, int j, float* x) {
    if (q.capture) {
        if ((int)q.lg.size() < (j + 1) * glm::VOCAB) q.lg.resize((size_t)(j + 1) * glm::VOCAB);
        std::copy(x, x + glm::VOCAB, q.lg.begin() + (size_t)j * glm::VOCAB);
        q.lg_rows = j + 1;
    }
    if (q.mask.empty()) q.mask.assign((size_t)8 * MASK_WORDS, 0xFFFFFFFFu);
    const uint32_t* m = q.mask.data() + (size_t)std::min(j, 7) * MASK_WORDS;
    int best = -1, raw = 0;
    float rawv = -INFINITY;
    for (int i = 0; i < glm::VOCAB; i++) {
        if (x[i] > rawv) { rawv = x[i]; raw = i; }
        if (!(m[i >> 5] >> (i & 31) & 1u) || !std::isfinite(x[i])) {
            x[i] = -INFINITY;
            continue;
        }
        if (best < 0 || x[i] > x[best]) best = i;
    }
    if (best < 0) return raw;  // review rel L1: nothing allowed (the grammar is in its error state): the raw argmax
    if (e->samp_T <= 0) return best;
    std::vector<int> tk;
    std::vector<double> pp;
    e->full_dist(x, tk, pp);
    return e->sample_from(tk, pp, -1);
}
// the model's drafts for the server's grammar plan (io[0], io[1..])
void put_drafts(Engine* e, SlotS& q) {
    const int n = std::min(e->sp_kcur, 15);
    q.io[0] = n;
    for (int i = 0; i < n; i++) q.io[1 + i] = e->sp_d[i];
}
// spec_start under the slot's grammar / logprobs: the first token comes from the prompt end's logits
int start_slot(Engine* e, SlotS& q, const int* ids, int n, int k) {
    if (q.capture) q.lg_rows = 0;
    if (q.constrained)
        e->sp_first = [&](const float* lg) {
            q.tmp.assign(lg, lg + glm::VOCAB);
            return pick_masked(e, q, 0, q.tmp.data());
        };
    int t;
    try {
        t = e->spec_start(ids, n, k);
    } catch (...) {
        e->sp_first = nullptr;
        throw;
    }
    e->sp_first = nullptr;
    if (q.constrained) put_drafts(e, q);
    return t;
}
bool job_busy();
void refill(Engine* e, SlotS& q);
// make slot s current: sequence bank (GLM_SEQS=2) or, with virtual slots, park the bound reply and resume s's;
// + its sampling parameters and rng. false: not possible now (a prompt's prefill holds the engine, or the bound reply
// cannot be parked: no host memory for its state); nothing changed
bool act(Engine* e, int s) {
    Slots& X = SL(e);
    if (s == X.cur) return true;
    if (!X.virt) {
        X.s[X.cur].rng = e->srng;
        e->bind_seq(s);
        X.cur = s;
        apply(e, X.s[s]);
        return true;
    }
    if (job_busy()) return false;
    const double t0 = now_ms();
    SlotS& c = X.s[X.cur];
    c.rng = e->srng;
    if ((c.active || (c.pending && c.first >= 0)) && !c.parked) {
        if (!e->conv_park(c.pk)) {
            static int warned = 0;
            if (warned++ < 3) fprintf(stderr, "[glm] a reply could not be parked (snapshots off or no host memory): the "
                                              "next conversation waits for it\n");
            return false;
        }
        c.parked = true;
        const double ms = now_ms() - t0;
        X.park_ms = X.npark++ ? 0.8 * X.park_ms + 0.2 * ms : ms;
    }
    X.cur = s;
    SlotS& q = X.s[s];
    apply(e, q);
    if (q.parked) {
        const double t1 = now_ms();
        q.parked = false;
        if (e->conv_resume(q.pk) != 1) refill(e, q);  // (a lost state's re-prefill is no switch cost: not sampled)
        else {
            const double ms = now_ms() - t1;
            X.res_ms = X.nres++ ? 0.8 * X.res_ms + 0.2 * ms : ms;
        }
    }
    X.sw_ms = X.park_ms + X.res_ms;
    q.t_bound = now_ms();
    return true;
}
// free slot for a prompt: the one whose cached tokens prefix it (longest), else the least recently used
int pick(Engine* e, const int* ids, int n) {
    Slots& X = SL(e);
    int best = -1, bl = -1;
    if (X.virt) {  // [glm-rel-slice] prefix reuse comes from the snapshots, whatever the slot: the least recently used one
        for (int s = 0; s < (int)X.s.size(); s++)
            if (!X.s[s].active && !X.s[s].pending && !X.s[s].parked && (best < 0 || X.s[s].use < X.s[best].use)) best = s;
        return best;
    }
    for (int s = 0; s < e->nseq; s++) {
        if (X.s[s].active || X.s[s].pending) continue;
        const std::vector<int>& h = s == e->cur_seq ? e->sp_hist : e->eseq[s].sp_hist;
        const int pos = s == e->cur_seq ? e->pos : e->eseq[s].pos;
        int l = 0;
        if (pos > 0 && (int)h.size() == pos && n > pos && std::equal(h.begin(), h.end(), ids)) l = pos;
        if (l > bl || (l == bl && best >= 0 && X.s[s].use < X.s[best].use)) { bl = l; best = s; }
    }
    return best;
}
// ---- [glm-rel-pfgate] a prompt's prefill on a job thread that returns to the server between layers (cancel, progress,
// the hang monitor). GLM_PF_SLICE_MS (default 1000): host time per qe_prefill_more call; 0 = the whole prompt in one call
struct PfJob {
    std::thread th;
    bool open = false, done = false, cancelled = false, fallback = false, finishing = false;
    int slot = -1, result = -1;
    std::exception_ptr err;
};
PfJob& JOB() {
    static PfJob* j = new PfJob();  // review rel L12: never destroyed (a joinable thread at exit would terminate)
    return *j;
}
int slice_ms() {
    const char* v = getenv("GLM_PF_SLICE_MS");  // (read per call: the check tool switches it)
    return v && *v ? atoi(v) : 1000;
}
bool all_parked(Engine* e) {  // (pfg.mu held)
    if (e->pfg.waiting != e->ngpu) return false;
    for (long u : e->pfg.used) if (u != e->pfg.credit) return false;
    return true;
}
void job_start(Engine* e, SlotS& q, int slot) {
    PfJob& J = JOB();
    J.open = true; J.done = false; J.cancelled = false; J.fallback = false; J.finishing = false; J.slot = slot; J.result = -1; J.err = nullptr;
    {
        std::lock_guard<std::mutex> lk(e->pfg.mu);
        e->pfg.on = true; e->pfg.free = false; e->pfg.stop_win = false;
        e->pfg.credit = 0; e->pfg.used.assign(e->ngpu, 0); e->pfg.waiting = 0; e->pfg.prog = 0;
    }
    const int dev = e->W[0]->dev;
    J.th = std::thread([e, &q, dev] {
        PfJob& J = JOB();
        int r = -1;
        std::exception_ptr err;
        try {
            CUDA_CHECK(cudaSetDevice(dev));
            r = start_slot(e, q, q.ids.data(), q.n, q.k);
            bool cancelled;
            {
                std::lock_guard<std::mutex> lk(e->pfg.mu);
                cancelled = J.cancelled;
                J.finishing = true;  // (a release from here on waits for the join and releases itself)
            }
            if (cancelled) {  // its release came while the prompt ran: what qe_release does
                // [glm-rel3-cancel] a continuation put back live keeps its answer-end snapshot due (ck_prefill)
                if (!e->pf_live_kept) e->ck_ans_due = false;
                e->seq_end();
            }
        } catch (const std::exception& ex) {  // review rel L11: said here (a rethrow through ctypes would terminate)
            fprintf(stderr, "[glm] prefill job: %s\n", ex.what());
            err = std::current_exception();
        } catch (...) {
            fprintf(stderr, "[glm] prefill job: unknown exception\n");
            err = std::current_exception();
        }
        std::lock_guard<std::mutex> lk(e->pfg.mu);
        J.result = r;
        J.err = err;
        J.done = true;
        e->pfg.cv.notify_all();
    });
}
// run the job for about `ms` of host time (ms <= 0: to its end); true = done (not joined yet)
bool job_loop(Engine* e, int ms) {
    PfJob& J = JOB();
    auto& g = e->pfg;
    const auto t0 = std::chrono::steady_clock::now();
    auto el = [&] { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count(); };
    std::unique_lock<std::mutex> lk(g.mu);
    if (ms <= 0) { g.free = true; g.cv.notify_all(); }
    double stall = 0;
    while (!J.done) {
        if (!g.free && all_parked(e)) {
            if (ms > 0 && el() >= ms) return false;
            g.credit++;
            g.cv.notify_all();
            stall = el();
        }
        g.cv.wait_for(lk, std::chrono::milliseconds(50), [&] { return J.done || (!g.free && all_parked(e)); });
        // some cards parked and the others not for 10 s: a host wait across layers the gate did not expect; the rest of
        // the prompt runs ungated (the result is the same; only the returns between layers are lost)
        if (!J.done && !g.free && !all_parked(e) && g.waiting > 0 && el() - stall > 10000) {
            fprintf(stderr, "[glm] gated prefill: cards stuck at different layers; the rest of this prompt runs in one call\n");
            g.free = true;
            J.fallback = true;
            g.cv.notify_all();
        }
    }
    return true;
}
// run the job for about `ms` of host time (ms <= 0: to its end); true = done (joined)
bool job_run(Engine* e, int ms) {
    PfJob& J = JOB();
    auto& g = e->pfg;
    if (!job_loop(e, ms)) return false;
    J.th.join();
    J.open = false;
    {
        std::lock_guard<std::mutex> lk2(g.mu);
        g.on = false;
    }
    if (J.err) {  // review rel L11: the engine state is unknown after a failed prefill: a fatal, reported end
        fprintf(stderr, "[glm] fatal: the prefill job failed (see above)\n");
        fflush(stderr);
        abort();
    }
    return true;
}
// a cancel (qe_release of the slot whose prompt runs): stop at the gate every card waits at (only while every card
// waits at a gate: the same decision on each; [glm-rel3-cancel] before the next layer, GLM_PF_CANCEL_MID=0: the next
// window start), the job ends in the background. false: the prompt is complete already (the caller joins and
// releases as usual)
bool job_cancel(Engine* e) {
    PfJob& J = JOB();
    auto& g = e->pfg;
    std::lock_guard<std::mutex> lk(g.mu);
    if (J.finishing || J.done) return false;
    J.cancelled = true;
    if (!g.free && all_parked(e)) g.stop_win = true;
    g.free = true;
    g.cv.notify_all();
    return true;
}
// a finished job (done in the background after a cancel) is joined; false: it still runs
bool job_busy() { return JOB().open; }
// [glm-rel-slice] a step call while a prompt's prefill job runs: the job runs for one slice instead (the replies wait)
void job_advance(Engine* e, int ms) {
    PfJob& J = JOB();
    if (!J.open) return;
    // review R3-H1: a finished cancelled job is joined here (nothing else would until a new request arrives); a finished
    // normal job stays open for its slot's prefill call (its result)
    if (job_loop(e, ms > 0 ? ms : 1000) && J.cancelled) job_run(e, 0);
}
bool job_idle(Engine* e) {
    PfJob& J = JOB();
    if (!J.open) return true;
    {
        std::lock_guard<std::mutex> lk(e->pfg.mu);
        if (!J.done) return false;
    }
    job_run(e, 0);
    return true;
}
// [glm-rel-slice] a parked reply whose archived rows were given back (host memory short): it is prefilled again from its
// prompt + the tokens it has sent; the start's token is its next one (not bit-identical to the run without the switch)
void refill(Engine* e, SlotS& q) {
    std::vector<int> ids = q.pk.sp_hist;
    ids.push_back(q.pk.sp_t);
    const int k = q.pk.sp_k;
    e->conv_drop(q.pk);
    fprintf(stderr, "[glm] a parked reply lost its archived rows (host memory short): %zu tokens prefilled again (this "
                    "reply is no longer bit-identical to a run alone)\n", ids.size());
    if ((int)ids.size() >= e->ctx) {  // (review R3-L3: the context is full: the reply ends here, it never steps again)
        q.pend_tok = glm::EOS0;
        q.dead = true;
        return;
    }
    q.pend_tok = start_slot(e, q, ids.data(), (int)ids.size(), k);
}
// the slot a time-sliced step gives the engine to: the bound one until its slice ends, then the one waiting longest
double slice_ms_now(const Slots& X) {
    static const double base = getenv("GLM_SLICE_MS") ? atof(getenv("GLM_SLICE_MS")) : 2000.0;
    static const double mult = getenv("GLM_SLICE_X") ? atof(getenv("GLM_SLICE_X")) : 8.0;
    return std::max(base, mult * X.sw_ms);
}
// a new prompt parks the bound reply only after this much of its turn (GLM_PREEMPT_MS, default = the slice)
double quantum_ms(const Slots& X) {
    static const double q = getenv("GLM_PREEMPT_MS") ? atof(getenv("GLM_PREEMPT_MS")) : -1.0;
    return q >= 0 ? q : slice_ms_now(X);
}
// one line every 30 s while switches happen (count, cost, slots copied, seed restores before prompts)
void sw_log(Engine* e) {
    static double t_last = 0;
    static unsigned long long n_last = 0;
    const double t = now_ms();
    if (e->st_sw_res == n_last || t - t_last < 30000) return;
    t_last = t;
    n_last = e->st_sw_res;
    fprintf(stderr, "[glm] time slicing: %llu parks %.1f ms, %llu resumes %.0f ms each (%.0f slots from host, %.0f on the "
                    "card), %llu lost; seed restores %llu, %.0f ms each\n", e->st_sw_park,
            e->st_sw_park_ms / std::max(1ull, e->st_sw_park), e->st_sw_res, e->st_sw_res_ms / std::max(1ull, e->st_sw_res),
            (double)e->st_sw_slots / std::max(1ull, e->st_sw_res), (double)e->st_sw_d2d / std::max(1ull, e->st_sw_res),
            e->st_sw_lost, e->st_det_n, e->st_det_us / 1e3 / std::max(1ull, e->st_det_n));
}
}  // namespace
extern "C" {
// sampling: for a prompt begun and not prefilled yet (server order: right after qe_start_begin) that slot, else the
// next qe_start3 and the current slot
void qe_set_sampling(void* h, float temperature, float top_p, int top_k, float min_p, unsigned long long seed) {
    Engine* e = (Engine*)h;
    Slots& X = SL(e);
    SlotS q;
    q.T = temperature > 0 ? temperature : 0.f; q.top_p = top_p; q.top_k = top_k; q.min_p = min_p;
    q.rng.seed(seed ? seed : std::random_device{}());
    X.next = q;
    int s = X.cur;
    if (X.newest >= 0 && X.s[X.newest].pending && X.s[X.newest].first < 0) s = X.newest;
    take(X.s[s], q);
    if (s == X.cur) e->set_sampling(temperature, top_p, top_k, min_p, seed);
}
// check tool (release track): the next constrained step accepts at most r rows (the oracle of qe_trim)
void glm_test_accept(void* h, int r) { ((Engine*)h)->sp_test_a = r; }
// check tool ([glm-rel3-cancel]): 0 = a cancel stops at window starts only (the old rule), 1 = at the next layer gate
void glm_test_cancel_mid(void* h, int on) {
    std::lock_guard<std::mutex> lk(((Engine*)h)->pfg.mu);
    ((Engine*)h)->pf_cancel_mid = on != 0;
}
// check tool (round 18): extra prefix-snapshot cuts of the next start
void glm_set_cuts(void* h, const int* c, int n) { ((Engine*)h)->ck_dbg_cuts.assign(c, c + n); }
int qe_num_slots(void* h) { return (int)SL((Engine*)h).s.size(); }  // [glm-rel-slice] GLM_CONVS virtual slots
int qe_start3(void* h, const int* ids, int n, int k, unsigned long long affinity, int hint) {
    (void)affinity;
    Engine* e = (Engine*)h;
    if (n >= e->ctx) return -1;
    if (!job_idle(e)) job_run(e, 0);  // review rel M1: a cancelled prompt ends its window first (the engine waits)
    Slots& X = SL(e);
    const int s = pick(e, ids, n);
    if (s < 0) return -1;
    take(X.s[s], X.next);  // sampling set before start3 applies to the new sequence
    if (X.virt) {
        if (!act(e, s)) return -1;
    } else {
        X.s[X.cur].rng = e->srng;
        if (s != X.cur) { e->bind_seq(s); X.cur = s; }
    }
    apply(e, X.s[s]);
    X.s[s].active = true;
    X.s[s].limit = -1;
    X.s[s].ign_eos = false;
    X.s[s].dead = false;
    X.s[s].lg_rows = 0;
    X.s[s].use = ++X.clock;
    X.last = s;
    e->sp_hint = hint;  // round 18: end of the system part, a prefix-snapshot cut
    return start_slot(e, X.s[s], ids, n, k);
}
// round 25 ([glm-p3-turnend]): the step accepts at most the reply's remaining limit and stops at the first EOS, so the
// cache never holds tokens the client did not get (GLM_STEP_CAP=0: the old full-step accept)
static int slot_step(Engine* e, int* out) {
    static const bool on = getenv("GLM_STEP_CAP") ? atoi(getenv("GLM_STEP_CAP")) != 0 : true;
    SlotS& q = SL(e).s[SL(e).cur];
    q.t_run = now_ms();
    if (q.pend_tok >= 0 || q.dead) {  // [glm-rel-slice] a re-prefilled reply's start token (dead: context full, EOS)
        out[0] = q.dead ? glm::EOS0 : q.pend_tok;
        q.pend_tok = -1;
        if (q.limit >= 0) q.limit = std::max(0, q.limit - 1);
        return 1;
    }
    // review r25-L5: a limit of 0 still returns one token (a step always emits its first row): the 27B engine's semantics
    // (the reviewer asked for no code change)
    e->sp_cap = on && q.limit >= 0 ? std::max(1, q.limit) : 0;
    e->sp_eos_stop = on && !q.ign_eos;
    int n;
    if (q.constrained) {  // [glm-rel-mask] the server's planned drafts under its row masks
        q.lg_rows = 0;
        const int nd = std::max(0, std::min(q.io[16], 15));
        n = e->spec_step_pick(out, q.io + 17, nd, [&](int j, float* x) { return pick_masked(e, q, j, x); });
        put_drafts(e, q);
    } else n = e->spec_step(out);
    e->sp_cap = 0;
    e->sp_eos_stop = false;
    if (q.limit >= 0) q.limit = std::max(0, q.limit - n);
    return n;
}
int qe_spec_step(void* h, int* out) {
    Engine* e = (Engine*)h;
    Slots& X = SL(e);
    if (JOB().open) {  // review rel L3: no step while a prompt's prefill job runs
        if (X.virt) job_advance(e, slice_ms());  // [glm-rel-slice] the replies wait: the prompt goes on instead
        return 0;
    }
    if (X.virt) {
        if (X.want >= 0 && X.want != X.cur && !act(e, X.want)) return 0;  // (bound when possible)
        if (!X.s[X.cur].active) return 0;
    }
    return slot_step(e, out);
}
void qe_set_limit(void* h, int slot, int tokens) {
    Slots& X = SL((Engine*)h);
    if (slot >= 0 && slot < (int)X.s.size()) X.s[slot].limit = tokens;
}
void qe_set_ignore_eos(void* h, int slot, int on) {  // review r25-M2 (reset at start and release)
    Slots& X = SL((Engine*)h);
    if (slot >= 0 && slot < (int)X.s.size()) X.s[slot].ign_eos = on != 0;
}
int qe_last_reused(void* h) { return ((Engine*)h)->sp_reused; }
int qe_last_slot(void* h) { return SL((Engine*)h).last; }
void qe_release(void* h, int slot) {
    Slots& X = SL((Engine*)h);
    if (X.virt) sw_log((Engine*)h);
    PfJob& J = JOB();
    if (X.virt && slot >= 0 && slot < (int)X.s.size() && slot != X.cur && !(J.open && J.slot == slot)) {
        // [glm-rel-slice] a reply that is not bound (parked, or a prompt not prefilled yet): its state goes, the bound
        // reply's stays as it is
        SlotS& q = X.s[slot];
        if (q.parked) {  // review R3-M2: its state is the answer end of its turn so far: kept as an answer-end snapshot
            // (review R3-L1: a running prefill job writes the snapshot table: the job ends its prompt first)
            if (J.open) job_run((Engine*)h, 0);
            ((Engine*)h)->conv_keep(q.pk);
        }
        q.parked = false;
        q.pend_tok = -1;
        q.active = q.pending = false;
        q.limit = -1;
        q.ign_eos = false;
        q.constrained = q.capture = q.watch = false;
        q.lg_rows = 0;
        if (X.want == slot) X.want = -1;
        return;
    }
    if (J.open && J.slot == slot) {  // [glm-rel-pfgate] a cancel mid-prompt
        if (job_cancel((Engine*)h)) {
            X.s[slot].active = X.s[slot].pending = false;
            X.s[slot].constrained = X.s[slot].capture = X.s[slot].watch = false;
            job_idle((Engine*)h);
            return;  // (the job thread takes the seq_end)
        }
        job_run((Engine*)h, 0);  // complete already: joined, then the usual release
    }
    if (slot >= 0 && slot < (int)X.s.size()) {
        SlotS& q = X.s[slot];
        q.active = q.pending = false;
        q.limit = -1;
        q.ign_eos = false;
        q.constrained = q.capture = q.watch = false;
        q.lg_rows = 0;
        if (slot == X.cur) ((Engine*)h)->sp_watch = false;
    }
    if (slot == X.cur) ((Engine*)h)->ck_ans_due = true;  // round 25: answer-end snapshot when the next prompt needs it
    if (X.virt && slot >= 0 && slot < (int)X.s.size()) {
        X.s[slot].pend_tok = -1;
        if (X.want == slot) X.want = -1;
        // [glm-rel-slice] a parked reply resumes next: its own cache state is put back, a seed restore now is wasted
        for (auto& q : X.s) if (q.parked) return;
    }
    ((Engine*)h)->seq_end();  // round 17 (GLM_DET): the cache goes back to the seed while the server waits for the next request
}
// chunked group (whole-prompt prefill): begin picks the slot, more prefills it, finish returns the first token
int qe_start_begin(void* h, const int* ids, int n, int k, unsigned long long affinity, int hint, const void* img,
                   int img_rows, const int* rope3, int delta) {
    (void)affinity; (void)img; (void)img_rows; (void)rope3; (void)delta;
    Engine* e = (Engine*)h;
    if (n >= e->ctx) return -3;  // (engine.py StartRefused: too long)
    if (!job_idle(e)) job_run(e, 0);  // review rel M1: a cancelled prompt ends its window first (the engine waits)
    Slots& X = SL(e);
    const int s = pick(e, ids, n);
    if (s < 0) return -1;
    SlotS& q = X.s[s];
    q.pending = true; q.ids.assign(ids, ids + n); q.k = k; q.first = -1; q.n = n; q.hint = hint; q.limit = -1;
    q.ign_eos = false;
    q.dead = false;
    q.constrained = q.capture = q.watch = false;  // (qe_constrain / qe_watch_stops per request, after start_begin)
    q.lg_rows = 0;
    take(q, X.next);
    q.use = ++X.clock;
    X.newest = s;
    return s;
}
int qe_prefill_more_slot(void* h, int slot, int budget) {
    (void)budget;  // (the slice is host time: GLM_PF_SLICE_MS)
    Engine* e = (Engine*)h;
    Slots& X = SL(e);
    if (slot < 0 || slot >= (int)X.s.size() || !X.s[slot].pending) return 1;
    SlotS& q = X.s[slot];
    if (q.first >= 0) return 1;
    PfJob& J = JOB();
    if (!J.open) {
        if (!job_idle(e)) return 0;
        if (X.virt && X.cur != slot && X.s[X.cur].active && !X.s[X.cur].parked &&
            now_ms() - X.s[X.cur].t_bound < quantum_ms(X))
            return 0;  // [glm-rel-slice] the bound reply runs its quantum first (a short reply ends without a switch)
        if (!act(e, slot)) return 0;  // [glm-rel-slice] the bound reply could not be parked: this prompt waits
        apply(e, q);
        e->sp_hint = q.hint;
        // ungated: the whole prompt in this call (review rel L5: also short prompts, GLM_PF_GATE_MIN rows, default 8192,
        // which take well under a slice anyway and would pay the job thread for nothing)
        static const int gate_min = getenv("GLM_PF_GATE_MIN") ? atoi(getenv("GLM_PF_GATE_MIN")) : 8192;
        if (slice_ms() <= 0 || e->nseq != 1 || q.n < gate_min) {
            q.first = start_slot(e, q, q.ids.data(), q.n, q.k);
            return 1;
        }
        job_start(e, q, slot);
    }
    if (J.slot != slot) return 0;  // (one prompt at a time)
    if (!job_run(e, slice_ms())) return 0;
    q.first = J.result;
    return 1;
}
int qe_prefill_more(void* h, int budget) { return qe_prefill_more_slot(h, SL((Engine*)h).newest, budget); }
int qe_start_finish_slot(void* h, int slot) {
    Engine* e = (Engine*)h;
    Slots& X = SL(e);
    {   // the rest of the prompt in this call
        PfJob& J = JOB();
        if (!(J.open && J.slot == slot)) qe_prefill_more_slot(h, slot, 0);
        if (J.open && J.slot == slot) {
            job_run(e, 0);
            X.s[slot].first = J.result;
        }
    }
    SlotS& q = X.s[slot];
    // review R3-L4: a prompt that was not prefilled (its prefill call refused: the bound reply could not be parked)
    if (q.first < 0 || X.cur != slot) return -1;
    q.pending = false;
    q.active = true;
    X.last = slot;
    act(e, slot);
    q.t_bound = now_ms();  // [glm-rel-slice] its slice starts with its first token
    return q.first;
}
int qe_start_finish(void* h) { return qe_start_finish_slot(h, SL((Engine*)h).newest); }
int qe_num_prefills(void* h) {
    int n = 0;
    for (auto& q : SL((Engine*)h).s) n += q.pending;
    return n;
}
int qe_prefill_pos(void* h, int slot) {
    Engine* e = (Engine*)h;
    Slots& X = SL(e);
    if (slot < 0 || slot >= (int)X.s.size() || !X.s[slot].pending) return -1;
    if (X.s[slot].first >= 0) return X.s[slot].n;
    PfJob& J = JOB();
    if (J.open && J.slot == slot) {
        std::lock_guard<std::mutex> lk(e->pfg.mu);
        return std::min(std::max(0, e->pfg.prog), X.s[slot].n - 1);
    }
    return 0;
}
// batch group
void qe_set_concurrent(void* h, int on) { (void)h; (void)on; }
void qe_resume(void* h, int slot) {
    Slots& X = SL((Engine*)h);
    if (X.virt) X.want = slot;
    act((Engine*)h, slot);
}
void qe_step_batch(void* h, const int* slots, int n, int* out, int* cnt) {
    Engine* e = (Engine*)h;
    Slots& X = SL(e);
    if (JOB().open) {  // review rel L3
        for (int i = 0; i < n; i++) cnt[i] = 0;
        if (X.virt) job_advance(e, slice_ms());  // [glm-rel-slice] the replies wait: the prompt goes on instead
        return;
    }
    if (X.virt) {  // [glm-rel-slice] time slicing: one reply steps, the others wait parked (0 tokens, reported paused)
        X.want = -1;
        int ci = -1, pick_i = -1;
        for (int i = 0; i < n; i++) {
            cnt[i] = 0;
            const int s = slots[i];
            if (s < 0 || s >= (int)X.s.size() || !X.s[s].active) continue;
            if (s == X.cur) ci = i;
            else if (pick_i < 0 || X.s[s].t_run < X.s[slots[pick_i]].t_run) pick_i = i;
        }
        int run = ci;
        if (pick_i >= 0 && (ci < 0 || now_ms() - X.s[X.cur].t_bound >= slice_ms_now(X))) {
            if (act(e, slots[pick_i])) run = pick_i;
        }
        if (run < 0) return;
        cnt[run] = slot_step(e, out + run * 64);
        return;
    }
    // (review rel L2: constrained slots step alone, through slot_step)
    // (review r25 L1 / R3-L5: the pair step has no cap: only while neither limit can bind in it (more than MR tokens
    // left), and both limits count down after it; rows past an EOS stay in the cache, the next turn resumes from a snapshot)
    auto roomy = [&](int s) { return X.s[s].limit < 0 || X.s[s].limit > glm::MR; };
    if (n == 2 && e->nseq == 2 && slots[0] != slots[1] && X.s[slots[0]].T <= 0 && X.s[slots[1]].T <= 0 &&
        !X.s[slots[0]].constrained && !X.s[slots[1]].constrained && roomy(slots[0]) && roomy(slots[1])) {
        act(e, 0);
        int o[2][64], c[2];
        e->spec_step2(o[0], &c[0], o[1], &c[1]);  // leaves sequence 0 bound
        for (int i = 0; i < 2; i++) {
            cnt[i] = c[slots[i]];
            for (int j = 0; j < cnt[i]; j++) out[i * 64 + j] = o[slots[i]][j];
            SlotS& q = X.s[slots[i]];
            if (q.limit >= 0) q.limit = std::max(0, q.limit - cnt[i]);
        }
        return;
    }
    for (int i = 0; i < n; i++) {
        act(e, slots[i]);
        cnt[i] = slot_step(e, out + i * 64);
    }
}
int qe_vocab(void* h) { (void)h; return glm::VOCAB; }

// ---- release track [glm-rel-mask]: grammar group (engine.py "grammar") and logprobs
unsigned* qe_mask_buffer(void* h, int slot) {
    Slots& X = SL((Engine*)h);
    if (slot < 0 || slot >= (int)X.s.size()) return nullptr;
    SlotS& q = X.s[slot];
    if (q.mask.empty()) q.mask.assign((size_t)8 * MASK_WORDS, 0xFFFFFFFFu);
    return q.mask.data();
}
int* qe_spec_io(void* h, int slot) {
    Slots& X = SL((Engine*)h);
    return slot >= 0 && slot < (int)X.s.size() ? X.s[slot].io : nullptr;
}
void qe_constrain(void* h, int slot, int on) {
    Engine* e = (Engine*)h;
    Slots& X = SL(e);
    if (slot < 0 || slot >= (int)X.s.size()) return;
    SlotS& q = X.s[slot];
    if (on && q.mask.empty()) q.mask.assign((size_t)8 * MASK_WORDS, 0xFFFFFFFFu);
    q.constrained = on != 0;
    if (!on) q.capture = false;  // (the logprobs capture ends with the constraint)
    if (on) {
        q.io[16] = 0;
        if (slot == X.cur && !q.pending) put_drafts(e, q);  // a reply constrained mid-way (forced think close)
        else if (q.parked) {  // [glm-rel-slice] its drafts are in its parked state
            q.io[0] = std::min(q.pk.sp_kcur, 15);
            for (int i = 0; i < q.io[0]; i++) q.io[1 + i] = q.pk.sp_d[i];
        }
    }
}
int qe_constrained(void* h, int slot) {
    Slots& X = SL((Engine*)h);
    return slot >= 0 && slot < (int)X.s.size() && X.s[slot].constrained;
}
void qe_logits_capture(void* h, int slot, int on) {
    Slots& X = SL((Engine*)h);
    if (slot < 0 || slot >= (int)X.s.size()) return;
    X.s[slot].capture = on != 0 && X.s[slot].constrained;
}
int qe_logits_read(void* h, int slot, int row, float* out) {
    Slots& X = SL((Engine*)h);
    if (slot < 0 || slot >= (int)X.s.size()) return 0;
    const SlotS& q = X.s[slot];
    if (!q.capture || row < 0 || row >= q.lg_rows) return 0;
    std::copy(q.lg.begin() + (size_t)row * glm::VOCAB, q.lg.begin() + (size_t)(row + 1) * glm::VOCAB, out);
    return glm::VOCAB;
}

// ---- release track [glm-rel-stops]: stops group (engine.py "stops"); no step runs ahead of the server in GLM
void qe_watch_stops(void* h, int slot, int on) {
    Engine* e = (Engine*)h;
    Slots& X = SL(e);
    if (slot < 0 || slot >= (int)X.s.size()) return;
    X.s[slot].watch = on != 0;
    if (slot == X.cur) e->sp_watch = on != 0;
}
void qe_set_no_ahead(void* h, int slot, int on) { (void)h; (void)slot; (void)on; }
int qe_drafts(void* h, int slot, int* out) {
    Engine* e = (Engine*)h;
    Slots& X = SL(e);
    if (slot != X.cur || slot < 0 || slot >= (int)X.s.size() || !X.s[slot].active) return 0;
    const int n = std::min(e->sp_kcur, 64);
    for (int i = 0; i < n; i++) out[i] = e->sp_d[i];
    return n;
}
// keep the first pos tokens of prompt + reply (the last one is the next input): 1 = done, 0 = not possible (the cut
// is before the last step, or the step was not watched); the next turn then resumes from a snapshot as before
int qe_trim(void* h, int slot, int pos) {
    Engine* e = (Engine*)h;
    Slots& X = SL(e);
    if (JOB().open || slot != X.cur || slot < 0 || slot >= (int)X.s.size() || pos < 1) return 0;
    return e->spec_trim(pos - 1) ? 1 : 0;
}
// [glm-rel-slice] monitor: off, capacity, tokens, active, used, prio, paused (a reply waiting for its turn, or every
// reply while a prompt's prefill runs: the server's no-progress check skips paused replies)
void qe_slot_info(void* h, int slot, long long* b) {
    Engine* e = (Engine*)h;
    Slots& X = SL(e);
    for (int i = 0; i < 7; i++) b[i] = 0;
    if (slot < 0 || slot >= (int)X.s.size()) return;
    const SlotS& q = X.s[slot];
    b[1] = e->ctx;
    b[2] = slot == X.cur ? e->pos : q.parked ? q.pk.pos : 0;
    b[3] = q.active;
    b[4] = b[2];
    b[5] = 1;
    b[6] = q.active && (slot != X.cur || JOB().open) ? 1 : 0;
}
// [glm-rel-slice] switch statistics: parks, resumes, slots copied from host, lost (re-prefilled), park ms, resume ms,
// preemption cost ms (mean park + mean resume, 500 before any resume; the slice uses it), slots moved on the card
void glm_switch_stats(void* h, double* o) {
    Engine* e = (Engine*)h;
    o[0] = (double)e->st_sw_park; o[1] = (double)e->st_sw_res; o[2] = (double)e->st_sw_slots; o[3] = (double)e->st_sw_lost;
    o[4] = e->st_sw_park_ms; o[5] = e->st_sw_res_ms; o[6] = SL(e).sw_ms; o[7] = (double)e->st_sw_d2d;
}
// [glm-rel-pfgate] the server's client left while the slot's prompt is prefilled: stop it at the next layer gate ([glm-rel3-cancel])
void qe_prefill_cancel(void* h, int slot) {
    Slots& X = SL((Engine*)h);
    PfJob& J = JOB();
    if (slot < 0 || slot >= (int)X.s.size() || !J.open || J.slot != slot) return;
    job_cancel((Engine*)h);
}
}
