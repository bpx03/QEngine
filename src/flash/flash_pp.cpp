// Pipeline-parallel driver of the Flash-Next engine (see flash_pp.h).
#include "flash_pp.h"
#include "card_probe.h"
#include "../env.h"

#include <cmath>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>

namespace flash {

static double pp_now_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// ---------------------------------------------------------------- stage threads
// After a task the thread polls its queue for FLASH_PP_SPIN_US (default 200) before it sleeps: a decode step hands the
// group to the next stage every few ms, and a condition-variable wake-up costs tens of microseconds per hand-off
// (the host turn-around of a stage step)
StageThreads::StageThreads(int n) {
    static const int spin_us = env_int("FLASH_PP_SPIN_US", 200);
    for (int s = 0; s < n; s++) {
        Q* q = new Q;
        q->th = std::thread([q] {
            std::unique_lock<std::mutex> lk(q->mu);
            while (true) {
                if (q->fns.empty() && !q->quit && spin_us > 0) {
                    lk.unlock();
                    const auto t0 = std::chrono::steady_clock::now();
                    while (q->nq.load(std::memory_order_acquire) == 0 &&
                           std::chrono::steady_clock::now() - t0 < std::chrono::microseconds(spin_us))
                        std::this_thread::yield();
                    lk.lock();
                }
                q->cv.wait(lk, [q] { return q->quit || !q->fns.empty(); });
                if (q->fns.empty() && q->quit) return;
                auto fn = std::move(q->fns.front());
                q->fns.pop_front();
                q->nq.fetch_sub(1, std::memory_order_acq_rel);
                q->busy = true;
                lk.unlock();
                fn();
                lk.lock();
                q->busy = false;
                if (q->fns.empty()) q->idle.notify_all();
            }
        });
        q_.push_back(q);
    }
}
StageThreads::~StageThreads() {
    for (Q* q : q_) {
        { std::lock_guard<std::mutex> lk(q->mu); q->quit = true; }
        q->cv.notify_all();
        q->th.join();
        delete q;
    }
}
void StageThreads::post(int s, std::function<void()> fn) {
    Q* q = q_.at(s);
    {
        std::lock_guard<std::mutex> lk(q->mu);
        q->fns.push_back(std::move(fn));
        q->nq.fetch_add(1, std::memory_order_acq_rel);
    }
    q->cv.notify_one();
}
void StageThreads::run_on(int s, const std::function<void()>& fn) {
    std::mutex mu;
    std::condition_variable cv;
    bool fin = false;
    post(s, [&] {
        fn();
        std::lock_guard<std::mutex> lk(mu);
        fin = true;
        cv.notify_all();
    });
    std::unique_lock<std::mutex> lk(mu);
    cv.wait(lk, [&] { return fin; });
}
void StageThreads::wait(int s) {
    Q* q = q_.at(s);
    std::unique_lock<std::mutex> lk(q->mu);
    q->idle.wait(lk, [q] { return q->fns.empty() && !q->busy; });
}
void StageThreads::wait_all() {
    for (int s = 0; s < size(); s++) wait(s);
}

// ---------------------------------------------------------------- rank pool (PP x TP stages)
// The helpers spin FLASH_PP_RANK_SPIN_US (default 5000) after a call before they sleep, and the caller spins for its
// helpers before it waits: a stage task makes several calls in a row, and a decode step comes every few ms, so a
// condition-variable wake-up (tens of us) would sit in series with every stage step (review M3)
RankPool::RankPool(int t) : t_(std::max(1, t)) {
    static const int spin_us = env_int("FLASH_PP_RANK_SPIN_US", 5000);
    for (int r = 1; r < t_; r++)
        th_.emplace_back([this, r] {
            long long seen = 0;
            while (true) {
                // a short spin first (a stage task runs several fan-outs back to back), then the condition variable
                const auto t0 = std::chrono::steady_clock::now();
                while (agen_.load(std::memory_order_acquire) == seen &&
                       std::chrono::steady_clock::now() - t0 < std::chrono::microseconds(spin_us))
                    std::this_thread::yield();
                const std::function<void(int)>* fn;
                {
                    std::unique_lock<std::mutex> lk(mu_);
                    cv_.wait(lk, [&] { return quit_ || gen_ != seen; });
                    if (quit_) return;
                    seen = gen_;
                    fn = fn_;
                }
                (*fn)(r);
                if (left_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
                    std::lock_guard<std::mutex> lk(mu_);
                    dn_.notify_all();
                }
            }
        });
}
RankPool::~RankPool() {
    {
        std::lock_guard<std::mutex> lk(mu_);
        quit_ = true;
    }
    cv_.notify_all();
    for (auto& t : th_) t.join();
}
void RankPool::run(const std::function<void(int)>& fn) {
    if (t_ == 1) { fn(0); return; }
    {
        std::lock_guard<std::mutex> lk(mu_);
        fn_ = &fn;
        left_.store(t_ - 1, std::memory_order_release);
        gen_++;
        agen_.store(gen_, std::memory_order_release);
    }
    cv_.notify_all();
    fn(0);
    static const int spin_us = env_int("FLASH_PP_RANK_SPIN_US", 5000);
    const auto t0 = std::chrono::steady_clock::now();
    while (left_.load(std::memory_order_acquire) != 0 && std::chrono::steady_clock::now() - t0 < std::chrono::microseconds(spin_us))
        std::this_thread::yield();
    std::unique_lock<std::mutex> lk(mu_);
    dn_.wait(lk, [&] { return left_.load(std::memory_order_acquire) == 0; });
}

// ---------------------------------------------------------------- engine
static void* pinned_rows(size_t rows) {  // (fp32 rows: room for either wire format)
    void* p = nullptr;
    CUDA_CHECK(cudaHostAlloc(&p, rows * RW * 4, cudaHostAllocPortable));
    return p;
}

static int stages_of(const std::vector<Worker*>& w) {
    int n = 0;
    for (Worker* x : w) n = std::max(n, x->plan().stage + 1);
    return n;
}
PpEngine::PpEngine(std::vector<Worker*> workers, int max_rows) : th_(stages_of(workers)), max_rows_(max_rows) {
    const int n = stages_of(workers);
    CHECK(n >= 2, "flash: a pipeline needs at least 2 stages (%d)", n);
    G_.resize(n);
    for (Worker* w : workers) {
        const Plan& p = w->plan();
        CHECK(p.pp() && p.nstage == n && p.stage >= 0 && p.stage < n, "flash: pipeline worker of stage %d of %d", p.stage, p.nstage);
        G_[p.stage].push_back(w);
    }
    int at = 0;
    for (int s = 0; s < n; s++) {
        CHECK(!G_[s].empty(), "flash: pipeline stage %d has no worker", s);
        const int t = (int)G_[s].size();
        for (int r = 0; r < t; r++) {
            const Plan& p = G_[s][r]->plan();
            CHECK(G_[s][r] == workers[at + r], "flash: pipeline workers must come stage by stage, rank by rank");
            CHECK(p.rank == r && p.ngpu == t && p.l0 == G_[s][0]->plan().l0 && p.l1 == G_[s][0]->plan().l1,
                  "flash: stage %d rank %d: plan rank %d of %d, layers %d..%d", s, r, p.rank, p.ngpu, p.l0, p.l1);
        }
        const Plan& p = G_[s][0]->plan();
        CHECK(p.l0 == (s ? G_[s - 1][0]->plan().l1 : 0), "flash: pipeline stage %d: layers %d..%d do not follow the previous stage", s, p.l0, p.l1);
        W_.push_back(G_[s][0]);
        base_.push_back(at);
        rp_.push_back(t > 1 ? new RankPool(t) : nullptr);
        at += t;
    }
    CHECK(W_.back()->plan().last(), "flash: the last pipeline stage must hold the last layer");
    for (int b = 0; b + 1 < n; b++) {
        Bd d;
        d.ta = ranks(b);
        d.tb = ranks(b + 1);
        d.p2p = true;
        for (int j = 0; j < d.tb; j++) d.p2p = d.p2p && G_[b + 1][j]->pp_rx(0) != nullptr;
        bd_.push_back(d);
        for (int par = 0; par < 2; par++) {
            std::vector<void*> v;
            for (int k = 0; k < d.nbuf(); k++) v.push_back(d.p2p ? G_[b + 1][k]->pp_rx(par) : pinned_rows((size_t)max_rows_));
            pf_buf_.push_back(v);
        }
        fprintf(stderr, "[flash] pipeline boundary %d -> %d (%d -> %d ranks): %s, %s rows\n", b, b + 1, d.ta, d.tb,
                d.p2p ? "peer-to-peer" : "pinned host", Worker::pp_hop16() ? "bf16" : "fp32");
    }
    set_max_seqs(1);
}
PpEngine::~PpEngine() {
    th_.wait_all();
    for (size_t i = 0; i < pf_buf_.size(); i++)
        if (!bd_[i / 2].p2p)
            for (void* p : pf_buf_[i]) cudaFreeHost(p);
    for (size_t i = 0; i < dc_buf_.size(); i++)
        for (void* p : dc_buf_[i]) {
            if (bd_[i % bd_.size()].p2p) cudaFree(p);
            else cudaFreeHost(p);
        }
    for (RankPool* r : rp_) delete r;
}
void PpEngine::set_max_seqs(int ns) {
    if (ns <= dc_seqs_) return;
    int dev0 = 0;
    CUDA_CHECK(cudaGetDevice(&dev0));
    for (int q = dc_seqs_; q < ns; q++)
        for (int b = 0; b + 1 < nstage(); b++) {  // (a batched group: <= MAXM rows)
            std::vector<void*> v;
            for (int k = 0; k < bd_[b].nbuf(); k++) {
                void* p = nullptr;
                if (bd_[b].p2p) {  // device memory of the consumer's card: the producer's peer copy is the only leg
                    CUDA_CHECK(cudaSetDevice(G_[b + 1][k]->device()));
                    CUDA_CHECK(cudaMalloc(&p, (size_t)MAXM * RW * 4));
                } else {
                    p = pinned_rows(MAXM);
                }
                v.push_back(p);
            }
            dc_buf_.push_back(v);
        }
    dc_seqs_ = ns;
    CUDA_CHECK(cudaSetDevice(dev0));
}
void PpEngine::each(int s, const std::function<void(int, Worker*)>& fn) {
    if (!rp_[s]) { fn(0, W_[s]); return; }
    const std::vector<Worker*>& g = G_[s];
    rp_[s]->run([&](int r) { fn(r, g[r]); });
}
void PpEngine::io(int s, int r, Worker* w, bool pfb, int idx) {
    const int N = nstage();
    const void* in = nullptr;
    void* outs[4];
    int no = 0;
    if (s > 0) {
        const Bd& d = bd_[s - 1];
        const std::vector<void*>& v = pfb ? pf(s - 1, idx) : dc(idx, s - 1);
        in = v[d.p2p ? r : r % d.ta];
    }
    if (s + 1 < N) {
        const Bd& d = bd_[s];
        const std::vector<void*>& v = pfb ? pf(s, idx) : dc(idx, s);
        if (d.p2p) {
            for (int j = r; j < d.tb && no < 4; j += d.ta) outs[no++] = v[j];
        } else if (r < d.tb) {  // (a sender no receiver reads from, t_s > t_s+1: nothing to send)
            outs[no++] = v[r];
        }
    }
    w->pp_io(in, outs, no, pfb ? idx : 0);
}
void PpEngine::all(const std::function<void(int, Worker*)>& fn) {
    touch();
    // a latch for these calls only: the server's prefill flow may keep posting micro-batches to the queues meanwhile
    std::mutex mu;
    std::condition_variable cv;
    int left = nstage();
    for (int s = 0; s < nstage(); s++)
        th_.post(s, [&, s] {
            each(s, [&](int r, Worker* w) { fn(flat(s, r), w); });
            std::lock_guard<std::mutex> lk(mu);
            if (--left == 0) cv.notify_all();
        });
    std::unique_lock<std::mutex> lk(mu);
    cv.wait(lk, [&] { return left == 0; });
}

// ---- pipelined prefill
PpEngine::PfOut PpEngine::prefill(const int* prompt, int np, int b, const std::function<void(int, int, bf16*)>& ngram, bool mtp) {
    PfJob jb;
    jb.slot = W_[0]->cur_slot();
    jb.base = W_[0]->cur_base();
    jb.prompt = prompt;
    jb.np = np;
    jb.mtp = mtp;
    jb.ngram = ngram;
    std::vector<PfJob> v = {jb};
    return prefill_many(v, b)[0];
}

// Several prompts through the pipeline at once (cross-request pipelining): their micro-batches form one queue in job
// order; every stage walks the queue under the double-buffer protocol of the boundary buffers, binding each batch's slot
// first, so one prompt's first batches fill the stages while the previous one drains. A helper thread builds the n-gram
// rows (PLE input, only the stage holding layer 1 reads them) up to two batches ahead of that stage.
std::vector<PpEngine::PfOut> PpEngine::prefill_many(std::vector<PfJob>& jobs, int b) {
    const int N = nstage(), J = (int)jobs.size();
    std::vector<PfOut> res(J);
    struct B { int job, j, i, m; bool first, final; };
    std::vector<B> q;
    for (int k = 0; k < J; k++) {
        PfJob& jb = jobs[k];
        // cuts: b rows, a <= PFM tail joins the last batch when it fits every stage's batch buffers (the KV tails must
        // be free up to the job's absolute context end for a big batch)
        int cap = max_rows_;
        for (auto& g : G_)
            for (Worker* w : g) cap = std::min(cap, w->pf_batch((long long)jb.base * w->plan().period + jb.np + 64));
        for (int bd = 0; bd + 1 < N; bd++)  // peer-to-peer boundaries: within the receive buffers
            if (bd_[bd].p2p)
                for (Worker* w : G_[bd + 1]) cap = std::min(cap, w->pp_rx_rows());
        const int bb = std::max(PFM, std::min(b, cap - PFM));
        int nb = 0;
        if (jb.nknown < jb.np) jb.nknown = jb.np;
        for (int i = jb.start; i < jb.np;) {
            const int r = jb.np - i, m = r <= bb + PFM && r <= cap ? r : bb;
            q.push_back({k, nb++, i, m, i == 0, i + m >= jb.np});
            i += m;
        }
        res[k].batches = nb;
    }
    const int nq = (int)q.size();
    int sple = 0;  // the stage that reads the n-gram rows
    for (int s = 0; s < N; s++) if (W_[s]->layer_here(PLE_LAYER)) sple = s;
    std::vector<std::vector<bf16>> rows(nq);
    std::vector<int> done(N, 0);  // queue entries finished per stage
    int built = 0;                // n-gram rows ready for entries < built
    bool stop = false;
    std::mutex mu;
    std::condition_variable cv;
    const double t0 = pp_now_ms();
    std::vector<double> t_fill(J, -1.0);
    std::thread rowth([&] {
        for (int g = 0; g < nq; g++) {
            {
                std::unique_lock<std::mutex> lk(mu);
                cv.wait(lk, [&] { return stop || g <= done[sple] + 2; });
                if (stop) return;
            }
            const B& e = q[g];
            rows[g].resize((size_t)e.m * H);
            jobs[e.job].ngram(e.i, e.m, rows[g].data());
            {
                std::lock_guard<std::mutex> lk(mu);
                built = g + 1;
            }
            cv.notify_all();
        }
    });
    for (int s = 0; s < N; s++)
        th_.post(s, [&, s] {
            const bool last = s == N - 1;
            for (int g = 0; g < nq; g++) {
                const B& e = q[g];
                PfJob& jb = jobs[e.job];
                {
                    // input: stage s - 1 finished entry g; output buffer g & 1: stage s + 1 finished entry g - 2
                    std::unique_lock<std::mutex> lk(mu);
                    cv.wait(lk, [&] { return (s == 0 || done[s - 1] > g) && (last || done[s + 1] >= g - 1) && (s != sple || built > g); });
                }
                if (last && e.j == 0) t_fill[e.job] = pp_now_ms() - t0;
                std::vector<int> mt;
                if (last && jb.mtp)
                    for (int r = 0; r < e.m; r++) mt.push_back(e.i + r + 1 < jb.nknown ? jb.prompt[e.i + r + 1] : -1);
                int o = -1, d = -2;
                each(s, [&](int r, Worker* w) {
                    w->use_slot(jb.slot, jb.base);
                    if (e.first && jb.reset) w->reset();
                    io(s, r, w, true, g & 1);
                    int o1 = -1, d1 = -2;
                    std::vector<int> mt1 = mt;
                    // (last: the prompt's end, where the device puts the sampled token as the last row's MTP input; a job
                    // that stops before tokens it knows, mode A's [t0, n - 1), takes them from the prompt)
                    w->prefill_batch(jb.prompt + e.i, e.i, e.m, s == sple ? rows[g].data() : nullptr, &o1, last && jb.mtp ? mt1.data() : nullptr,
                                     e.final && jb.np >= jb.nknown, last && jb.mtp ? &d1 : nullptr);
                    if (last && jb.mtp && d1 == -2) {  // the MTP fill outside the batch pipeline (short batch)
                        if (e.final && jb.np >= jb.nknown) mt1[e.m - 1] = o1;
                        w->prefill_batch_mtp(mt1.data(), e.i, e.m, &d1);
                    }
                    if (last && e.final && jb.mtp) w->rmin_save(jb.slot);  // (the next job's batches overwrite the chain row)
                    if (r == 0) { o = o1; d = d1; }
                });
                if (last && e.final) {
                    res[e.job].next = o;
                    res[e.job].draft = jb.mtp ? d : -1;
                    res[e.job].wall_ms = pp_now_ms() - t0;
                }
                if (jb.xf) {  // overlapped hand-over: this stage's KV rows of the batch leave while the next batches run
                    CHECK(ranks(s) == 1, "flash: mode A hand-over from PP x TP stages is not supported");
                    jb.xf->after_stage(s);
                    jb.xf->kv(e.i, e.i + e.m, jb.mtp, 0, s);
                }
                if (s == sple) std::vector<bf16>().swap(rows[g]);  // (consumed)
                {
                    std::lock_guard<std::mutex> lk(mu);
                    done[s] = g + 1;
                }
                cv.notify_all();
            }
        });
    th_.wait_all();
    {
        std::lock_guard<std::mutex> lk(mu);
        stop = true;
    }
    cv.notify_all();
    rowth.join();
    for (int k = 0; k < J; k++) res[k].fill_ms = t_fill[k];
    return res;
}

// ---- host-driven decode
void PpEngine::forward(const int* toks, int pos, int m, const bf16* ple, int* out, int commit, int seq, int slot, int base) {
    touch();
    const int N = nstage();
    CHECK(seq >= 0 && seq < dc_seqs_, "flash: pipeline decode buffer set %d of %d", seq, dc_seqs_);
    for (int s = 0; s < N; s++)
        th_.run_on(s, [&, s] {
            each(s, [&](int r, Worker* w) {
                std::vector<int> o(MAXM);
                if (slot >= 0) { w->use_slot(slot, base); w->ws_ensure(); }
                io(s, r, w, false, seq);
                w->forward(toks, pos, m, ple, o.data(), nullptr, commit);
                if (s == N - 1 && r == 0) memcpy(out, o.data(), (size_t)m * 4);
            });
        });
}
int PpEngine::mtp(const int* toks, int pos, int m, bool chain, int slot, int base) {
    touch();
    const int s = nstage() - 1;
    int d = -1;
    th_.run_on(s, [&] {
        each(s, [&](int r, Worker* w) {
            std::vector<int> o(MAXM);
            if (slot >= 0) w->use_slot(slot, base);
            w->forward_mtp(toks, pos, m, chain, o.data());
            if (r == 0) d = o[m - 1];
        });
    });
    return d;
}

std::vector<int> PpEngine::mtp_chain(int slot, int base, int d0, int n, int k) {
    touch();
    std::vector<int> d = {d0};
    th_.run_on(nstage() - 1, [&] {
        each(nstage() - 1, [&](int r, Worker* w) {
            std::vector<int> dl = {d0};  // (every rank chains the same drafts: rank 0's are returned)
            w->use_slot(slot, base);
            w->rmin_load(slot);
            std::vector<int> o(MAXM);
            for (int j = 1; j < k; j++) {
                w->forward_mtp(&dl.back(), n + j - 1, 1, true, o.data());
                dl.push_back(o[0]);
            }
            if (r == 0) d = dl;
        });
    });
    return d;
}

// ---- interleaved decode: a sequence moves stage to stage as a task; the last stage verifies and posts the next step
PpEngine::DecodeStats PpEngine::decode_interleaved(std::vector<Seq>& seqs, int k,
                                                   const std::function<void(const std::vector<int>&, int, bf16*)>& ngram_row) {
    const int N = nstage(), c = (int)seqs.size();
    set_max_seqs(c);
    struct Step { std::vector<int> v, o; std::vector<bf16> ple; };
    std::vector<Step> st(c);
    std::mutex mu;
    std::condition_variable cv;
    int finished = 0;
    long long ntok = 0, nsteps = 0;
    // inputs of sequence i's next step (host): verify rows {pending, drafts}, their n-gram rows
    auto prep = [&](int i) {
        Seq& q = seqs[i];
        Step& x = st[i];
        x.v = {q.nxt};
        x.v.insert(x.v.end(), q.drafts.begin(), q.drafts.end());
        x.ple.resize((size_t)x.v.size() * H);
        std::vector<int> hh = q.hist;
        for (size_t j = 0; j < x.v.size(); j++) {
            hh.push_back(x.v[j]);
            ngram_row(hh, (int)hh.size() - 1, x.ple.data() + j * H);
        }
        x.o.assign(MAXM, 0);
    };
    std::function<void(int, int)> run;  // stage s runs sequence i's step
    run = [&](int s, int i) {
        th_.post(s, [&, s, i] {
            Seq& q = seqs[i];
            Step& x = st[i];
            each(s, [&](int r, Worker* w) {
                std::vector<int> o1(MAXM);
                w->use_slot(q.slot, q.base);
                io(s, r, w, false, i);
                w->forward(x.v.data(), q.pos, (int)x.v.size(), x.ple.data(), r ? o1.data() : x.o.data(), nullptr, q.commit);
            });
            if (s + 1 < N) { run(s + 1, i); return; }
            // last stage: accept the drafts that match, MTP drafts for the next step (same thread: Rsave_ of this forward)
            const int kd = (int)q.drafts.size();
            int a = 0;
            while (a < kd && x.v[a + 1] == x.o[a]) a++;
            for (int j = 0; j <= a; j++) q.hist.push_back(x.v[j]);
            int emitted = 0;
            for (int j = 0; j <= a && (int)q.out.size() < q.gen; j++) { q.out.push_back(x.o[j]); emitted++; }
            if (k > 0) {
                std::vector<int> dr;
                each(s, [&](int r, Worker* w) {
                    std::vector<int> mo(MAXM), dl;
                    w->forward_mtp(x.o.data(), q.pos, a + 1, false, mo.data());
                    dl = {mo[a]};
                    for (int j = 1; j < k; j++) {
                        w->forward_mtp(&dl.back(), q.pos + a + j, 1, true, mo.data());
                        dl.push_back(mo[0]);
                    }
                    if (r == 0) dr = dl;
                });
                q.drafts = dr;
            }
            q.nxt = x.o[a];
            q.pos += a + 1;
            q.commit = a + 1;
            q.steps++;
            {  // (no EOS stop: a fixed token count per sequence, as flash_run's TP decode)
                std::lock_guard<std::mutex> lk(mu);
                ntok += emitted;
                nsteps++;
                if ((int)q.out.size() >= q.gen) { q.done = true; finished++; }
            }
            if (q.done) { cv.notify_all(); return; }
            prep(i);
            run(0, i);
        });
    };
    const double t0 = pp_now_ms();
    std::vector<int> start;
    for (int i = 0; i < c; i++) {  // (counted before any task runs: the tasks update `finished` under the lock)
        if ((int)seqs[i].out.size() >= seqs[i].gen) seqs[i].done = true;
        if (seqs[i].done) finished++;
        else start.push_back(i);
    }
    for (int i : start) {
        prep(i);
        run(0, i);
    }
    {
        std::unique_lock<std::mutex> lk(mu);
        cv.wait(lk, [&] { return finished >= c; });
    }
    th_.wait_all();
    DecodeStats ds;
    ds.wall_s = (pp_now_ms() - t0) / 1e3;
    ds.tokens = ntok;
    ds.steps = nsteps;
    return ds;
}

// ---- batched decode: groups of sequences, one batched step per stage task
PpEngine::DecodeStats PpEngine::decode_batched(std::vector<Seq>& seqs, int k, const std::vector<int>& bases) {
    const int N = nstage(), c = (int)seqs.size();
    const std::vector<int> grp = flash_pp::decode_groups(c, N);
    const int G = flash_pp::decode_ngroups(c, N);
    CHECK(c <= MAXM / (k + 1) * G, "flash: %d sequences x %d rows do not fit %d groups of <= %d rows", c, k + 1, G, MAXM);
    set_max_seqs(G);
    struct Group {
        std::vector<int> seq;    // sequences still running
        std::vector<int> st;     // their speculative state rows (SPEC_ROW each), from the last stage
        std::vector<int> out, cnt;
    };
    std::vector<Group> gs(G);
    for (int i = 0; i < c; i++) {
        if ((int)seqs[i].out.size() >= seqs[i].gen) seqs[i].done = true;
        if (!seqs[i].done) gs[grp[i]].seq.push_back(i);
    }
    auto slots_of = [&](const Group& g) {
        std::vector<int> s;
        for (int i : g.seq) s.push_back(seqs[i].slot);
        return s;
    };
    const int SL = N - 1;
    std::mutex mu;
    std::condition_variable cv;
    int live = 0;
    long long ntok = 0, nsteps = 0;
    for (Group& g : gs)
        if (!g.seq.empty()) {
            live++;
            g.st.resize(g.seq.size() * Worker::SPEC_ROW);
            const std::vector<int> sl = slots_of(g);
            int* dst = g.st.data();
            th_.post(N - 1, [this, SL, sl, dst] { W_[SL]->spec_state_get(sl.data(), (int)sl.size(), dst); });
        }
    th_.wait(N - 1);
    std::function<void(int, int)> run;
    run = [&](int s, int gi) {
        th_.post(s, [&, s, gi] {
            Group& g = gs[gi];
            Worker* w = W_[s];
            const std::vector<int> sl = slots_of(g);
            const int n = (int)sl.size();
            if (s + 1 < N) each(s, [&](int, Worker* x) { x->spec_state_set(sl.data(), n, g.st.data()); });  // the state the last stage left
            static const bool trace = env_int("FLASH_PP_TRACE", 0);  // debug: the state row each stage starts from
            if (trace) {
                std::vector<int> mine(Worker::SPEC_ROW);
                w->spec_state_get(sl.data(), 1, mine.data());
                const int* r = mine.data();
                fprintf(stderr, "[pptrace] stage %d group %d slot %d: tok %d %d %d %d pos %d commit %d hist %d %d spos %d prevm %d\n", s, gi, sl[0],
                        r[0], r[1], r[2], r[3], r[MAXM + 1], r[MAXM + 2], r[MAXM + 3], r[MAXM + 4], r[MAXM + 5], r[MAXM + 6]);
            }
            g.out.assign((size_t)n * 64, 0);
            g.cnt.assign(n, 0);
            each(s, [&](int r, Worker* x) {
                std::vector<int> o1((size_t)n * 64), c1(n);
                io(s, r, x, false, gi);
                x->batch_launch(sl.data(), bases.data(), nullptr, n, k);
                x->batch_finish(r ? o1.data() : g.out.data(), r ? c1.data() : g.cnt.data());
            });
            if (s + 1 < N) { run(s + 1, gi); return; }
            // last stage: the group's emitted tokens; finished sequences leave the group
            std::vector<int> keep;
            long long em = 0;
            for (int i = 0; i < n; i++) {
                Seq& q = seqs[g.seq[i]];
                for (int j = 0; j < g.cnt[i] && (int)q.out.size() < q.gen; j++) { q.out.push_back(g.out[(size_t)i * 64 + j]); em++; }
                q.steps++;
                if ((int)q.out.size() >= q.gen) q.done = true;
                else keep.push_back(g.seq[i]);
            }
            g.seq = keep;
            bool gone = g.seq.empty();
            if (!gone) {
                const std::vector<int> sk = slots_of(g);
                g.st.resize(sk.size() * Worker::SPEC_ROW);
                w->spec_state_get(sk.data(), (int)sk.size(), g.st.data());
            }
            {
                std::lock_guard<std::mutex> lk(mu);
                ntok += em;
                nsteps++;
                if (gone) live--;
            }
            if (gone) { cv.notify_all(); return; }
            run(0, gi);
        });
    };
    const double t0 = pp_now_ms();
    for (int gi = 0; gi < G; gi++)
        if (!gs[gi].seq.empty()) run(0, gi);
    {
        std::unique_lock<std::mutex> lk(mu);
        cv.wait(lk, [&] { return live == 0; });
    }
    th_.wait_all();
    DecodeStats ds;
    ds.wall_s = (pp_now_ms() - t0) / 1e3;
    ds.tokens = ntok;
    ds.steps = nsteps;
    ds.groups = G;
    return ds;
}

// ---------------------------------------------------------------- server mode B (FLASH_PAR=pp in flash_capi.cpp)
// Prompts: every job's micro-batches join one global order; stage s runs entry g once stage s - 1 finished it and stage
// s + 1 finished entry g - 2 (the boundary buffer g & 1 is free again), the same protocol as prefill_many, but event
// driven: a task is posted to a stage queue only when it can run, so it never blocks the stage's thread and the decode
// steps queued next to it wait for at most the micro-batch in progress.
int PpEngine::srv_cap(int base, int n) const {
    int cap = max_rows_;
    for (auto& g : G_)
        for (Worker* w : g) cap = std::min(cap, w->pf_batch((long long)base * w->plan().period + n + 64));
    for (int bd = 0; bd + 1 < nstage(); bd++)
        if (bd_[bd].p2p)
            for (Worker* w : G_[bd + 1]) cap = std::min(cap, w->pp_rx_rows());
    return cap;
}
void PpEngine::srv_submit(SrvJob* j) {
    touch();
    const int N = nstage();
    std::lock_guard<std::mutex> lk(fmu_);
    if (fdone_.empty()) {
        fdone_.assign(N, 0);
        fposted_.assign(N, 0);
        for (int s = 0; s < N; s++) if (W_[s]->layer_here(PLE_LAYER)) sple_ = s;
    }
    j->t_submit = pp_now_ms();
    int i = j->i0;
    for (size_t c = 0; c < j->cuts.size(); c++) {
        const int e = j->cuts[c];
        CHECK(e > i && e <= j->n && e - i <= max_rows_, "flash pipeline: micro-batch [%d, %d) of a %d-row job", i, e, j->n);
        fq_.push_back({j, i, e - i, e >= j->n});
        i = e;
    }
    CHECK(i == j->n && !j->cuts.empty(), "flash pipeline: job cuts end at %d, not %d", i, j->n);
    srv_try_post(0);
}
void PpEngine::srv_try_post(int s) {
    const int N = nstage();
    if (s < 0 || s >= N || fposted_.empty()) return;
    if (fhold_ > 0 || (fblock_until_ > 0 && pp_now_ms() < fblock_until_)) return;  // a decode call / the decoders' share
    const long long g = fposted_[s];
    if (g != fdone_[s] || g >= fq0_ + (long long)fq_.size()) return;  // one micro-batch in flight per stage, in order
    if (s > 0 && fdone_[s - 1] <= g) return;                           // input not there yet
    if (s + 1 < N && fdone_[s + 1] < g - 1) return;                    // output buffer g & 1 still being read
    fposted_[s] = g + 1;
    th_.post(s, [this, s, g] { srv_run(s, g); });
}
void PpEngine::srv_run(int s, long long g) {
    const int N = nstage();
    Mb e;
    {
        std::lock_guard<std::mutex> lk(fmu_);
        e = fq_[(size_t)(g - fq0_)];
    }
    SrvJob& jb = *e.job;
    const bool last = s == N - 1;
    const double tw0 = pp_now_ms();
    if (!jb.cancel.load()) {
        std::vector<bf16> rows;
        if (s == sple_) {
            rows.resize((size_t)e.m * H);
            jb.ngram(e.i, e.m, rows.data());
        }
        std::vector<int> mt;
        if (last && jb.mtp)
            for (int r = 0; r < e.m; r++) mt.push_back(e.i + r + 1 < jb.n ? jb.ids[e.i + r + 1] : -1);
        int o = -1, d = -2;
        std::atomic<bool> nf{false};
        each(s, [&](int r, Worker* w) {
            w->use_slot(jb.slot, jb.base);
            w->ws_ensure();  // (a batched decode step in between left the bound slot's state in its storage)
            io(s, r, w, true, (int)(g & 1));
            // (use_slot made the head constrained; every head rank masks its vocabulary slice)
            if (last && e.final && jb.mask0) w->upload_mask(jb.mask0, 1);
            int o1 = -1, d1 = -2;
            std::vector<int> mt1 = mt;
            w->prefill_batch(jb.ids.data() + e.i, e.i, e.m, s == sple_ ? rows.data() : nullptr, &o1, last && jb.mtp ? mt1.data() : nullptr,
                             e.final, last && jb.mtp ? &d1 : nullptr);
            if (last && jb.mtp && d1 == -2) {
                if (e.final) mt1[e.m - 1] = o1;
                w->prefill_batch_mtp(mt1.data(), e.i, e.m, &d1);
            }
            if (last && e.final) {
                if (jb.mtp) w->rmin_save(jb.slot);
                // the guard of its last row, kept with the job (the decode steps' flags of this card stay with them)
                const unsigned keep = w->take_nf();
                if (w->nf_check("prefill (pipeline)", e.i + e.m)) nf.store(true);
                w->take_nf();
                w->put_nf(keep);
            }
            if (r == 0) { o = o1; d = d1; }
            for (size_t k = 0; k < jb.ck.size(); k++)
                if (jb.ck[k] == e.i + e.m && jb.ck_fn) jb.ck_fn((int)k, flat(s, r), w);
        });
        if (last && e.final) {
            if (nf.load()) jb.nf.store(true);
            jb.next = o;
            jb.draft = jb.mtp ? d : -1;
        }
    }
    // (cancelled: nothing runs; a producer's overlapped hop may still write the consumer's buffer (never read))
    std::lock_guard<std::mutex> lk(fmu_);
    if (s == 0) fwave_ms_ = fwave_ms_ > 0 ? 0.7 * fwave_ms_ + 0.3 * (pp_now_ms() - tw0) : pp_now_ms() - tw0;
    fdone_[s] = g + 1;
    if (last && e.final) {
        jb.t_done = pp_now_ms();
        jb.done.store(true);
        fcv_.notify_all();
    }
    // entries every stage finished leave the queue (their job may be gone)
    long long mn = fdone_[0];
    for (long long x : fdone_) mn = std::min(mn, x);
    while (fq0_ < mn && !fq_.empty()) { fq_.pop_front(); fq0_++; }
    srv_try_post(s);
    srv_try_post(s + 1);
    srv_try_post(s - 1);
}
void PpEngine::srv_kick() {
    std::lock_guard<std::mutex> lk(fmu_);
    for (int s = 0; s < nstage(); s++) srv_try_post(s);
}
bool PpEngine::srv_wait(SrvJob* j, double ms) {
    const double t_end = pp_now_ms() + std::max(0.0, ms);
    std::unique_lock<std::mutex> lk(fmu_);
    while (!j->done.load()) {
        for (int s = 0; s < nstage(); s++) srv_try_post(s);  // (the decoders' share window may have ended)
        const double left = t_end - pp_now_ms();
        if (left <= 0) break;
        fcv_.wait_for(lk, std::chrono::microseconds((long long)(std::min(left, 5.0) * 1000)), [&] { return j->done.load(); });
    }
    return j->done.load();
}
int PpEngine::srv_inflight() const {
    std::lock_guard<std::mutex> lk(fmu_);
    if (fdone_.empty()) return 0;
    return (int)(fq0_ + (long long)fq_.size() - fdone_.back());
}

void PpEngine::relay_drop(int slot) {
    if (slot >= 0 && slot < (int)rel_ok_.size()) rel_ok_[slot] = 0;
}
// decode rounds of the server: the slots in groups (pp_plan.h decode_groups), each group stage to stage as tasks; the
// last stage's state rows relayed to the other stages before the group's next step (as decode_batched)
bool PpEngine::relay_row(int slot, std::vector<int>& row) const {
    if (slot < 0 || slot >= (int)rel_ok_.size() || !rel_ok_[slot]) return false;
    row.assign(rel_.begin() + (size_t)slot * Worker::SPEC_ROW, rel_.begin() + (size_t)(slot + 1) * Worker::SPEC_ROW);
    return true;
}
void PpEngine::srv_decode(const std::vector<int>& slots, const std::vector<int>& bases, std::vector<int> lims, int k, int rounds,
                          const std::function<bool(int, const int*, int, int*)>& cont, const SrvCon* con, std::vector<char>* bad) {
    CHECK(!con || (slots.size() == 1 && rounds == 1), "flash pipeline: a constrained step is one slot, one round");
    touch();
    const int N = nstage(), c = (int)slots.size();
    if (c == 0) return;
    const int ns = W_[0]->num_slots();
    if ((int)rel_ok_.size() < ns) { rel_ok_.resize(ns, 0); rel_.resize((size_t)ns * Worker::SPEC_ROW, 0); }
    const std::vector<int> grp = flash_pp::decode_groups(c, N);
    const int G = flash_pp::decode_ngroups(c, N);
    set_max_seqs(G);
    // prompts in the queues: no micro-batch starts while this call's steps run (the running ones finish first), so a
    // decode step waits for at most one micro-batch wave instead of one micro-batch per stage it passes
    // (FLASH_PPB_HOLD=0: interleave freely)
    static const bool hold = env_int("FLASH_PPB_HOLD", 1) != 0;
    if (hold) {
        std::lock_guard<std::mutex> lk(fmu_);
        fhold_++;
    }
    struct Unhold {
        PpEngine* e;
        bool on;
        ~Unhold() {
            if (!on) return;
            std::lock_guard<std::mutex> lk(e->fmu_);
            // the decoders' share of the time while prompts are queued (FLASH_PPB_DSHARE, default 0.5 as the server's
            // --prefill-share): after a decode call no micro-batch starts for share / (1 - share) x a micro-batch's
            // time, so the next decode calls run on free stages
            static const double sh = std::clamp(env_double("FLASH_PPB_DSHARE", 0.5), 0.0, 0.95);
            if (--e->fhold_ == 0) {
                const bool queued = !e->fposted_.empty() && e->fdone_.back() < e->fq0_ + (long long)e->fq_.size();
                if (queued && sh > 0 && e->fwave_ms_ > 0) {
                    if (e->fblock_until_ < pp_now_ms() - e->fwave_ms_)  // (a new share window; not extended by every call)
                        e->fblock_until_ = pp_now_ms() + sh / (1 - sh) * e->fwave_ms_;
                } else {
                    e->fblock_until_ = 0;
                }
                for (int s = 0; s < e->nstage(); s++) e->srv_try_post(s);
            }
        }
    } unhold{this, hold};
    struct Group { std::vector<int> idx; int round = 0; std::vector<int> out, cnt; };
    std::vector<Group> gs(G);
    for (int i = 0; i < c; i++) gs[grp[i]].idx.push_back(i);
    std::mutex mu;
    std::condition_variable cv;
    int live = 0;
    for (auto& g : gs) live += !g.idx.empty();
    std::function<void(int, int)> run;
    run = [&](int s, int gi) {
        th_.post(s, [&, s, gi] {
            Group& g = gs[gi];
            Worker* w = W_[s];
            const int n = (int)g.idx.size();
            std::vector<int> sl(n), lm(n);
            for (int i = 0; i < n; i++) { sl[i] = slots[g.idx[i]]; lm[i] = lims[g.idx[i]]; }
            std::vector<int> ss, rows;
            if (s + 1 < N)  // the state rows the last stage left (none after a host-side spec_init: every stage has it)
                for (int x : sl)
                    if (rel_ok_[x]) {
                        ss.push_back(x);
                        rows.insert(rows.end(), rel_.begin() + (size_t)x * Worker::SPEC_ROW, rel_.begin() + (size_t)(x + 1) * Worker::SPEC_ROW);
                    }
            g.out.assign((size_t)n * 64, 0);
            g.cnt.assign(n, 0);
            std::atomic<unsigned> nfa{0};
            each(s, [&](int r, Worker* x) {
                if (!ss.empty()) x->spec_state_set(ss.data(), (int)ss.size(), rows.data());
                if (con) {  // the planned drafts on every stage (embedding, n-gram rows, the last stage's accept)
                    if (con->nd > 0) x->set_slot_drafts(sl[0], con->drafts, con->nd);
                    if (s + 1 == N) x->upload_mask(con->mask, con->mrows);  // (every head rank: its slice)
                }
                io(s, r, x, false, gi);
                std::vector<int> o1((size_t)n * 64), c1(n);
                x->batch_launch(sl.data(), bases.data(), lm.data(), n, k);
                x->batch_finish(r ? o1.data() : g.out.data(), r ? c1.data() : g.cnt.data());
                if (s + 1 == N) nfa.fetch_or(x->take_nf());
            });
            if (s + 1 < N) { run(s + 1, gi); return; }
            if (const unsigned nf = nfa.load(); nf && bad)  // rows with non-finite values -> their slots
                for (int i = 0; i < n && i * (k + 1) < 32; i++) {
                    const unsigned rm = lm[i] + 1 >= 32 ? ~0u : ((1u << (lm[i] + 1)) - 1);
                    if (nf & (rm << (i * (k + 1)))) (*bad)[g.idx[i]] = 1;
                }
            {
                std::vector<int> rows((size_t)n * Worker::SPEC_ROW);
                w->spec_state_get(sl.data(), n, rows.data());
                for (int i = 0; i < n; i++) {
                    std::copy(rows.begin() + (size_t)i * Worker::SPEC_ROW, rows.begin() + (size_t)(i + 1) * Worker::SPEC_ROW,
                              rel_.begin() + (size_t)sl[i] * Worker::SPEC_ROW);
                    rel_ok_[sl[i]] = 1;
                }
            }
            std::vector<int> keep;
            for (int i = 0; i < n; i++) {
                int lim = lims[g.idx[i]];
                if (cont(g.idx[i], g.out.data() + (size_t)i * 64, g.cnt[i], &lim)) {
                    lims[g.idx[i]] = lim;
                    keep.push_back(g.idx[i]);
                }
            }
            g.idx = keep;
            if (!keep.empty() && ++g.round < rounds) { run(0, gi); return; }
            std::lock_guard<std::mutex> lk(mu);
            if (--live == 0) cv.notify_all();
        });
    };
    for (int gi = 0; gi < G; gi++)
        if (!gs[gi].idx.empty()) run(0, gi);
    std::unique_lock<std::mutex> lk(mu);
    cv.wait(lk, [&] { return live == 0; });
}

// ---------------------------------------------------------------- server mode B, persistent decode
bool PpEngine::bg_run(const std::vector<BgReq>& req, int k, const std::vector<int>& bases) {
    const int N = nstage();
    std::unique_lock<std::mutex> lk(bmu_);
    bool any = false;
    for (auto& b : bs_) any |= b.member;
    if (any && (k != bk_ || bases != bbases_)) return false;
    if (bs_.empty()) {
        const int ns = W_[0]->num_slots();
        bs_.resize(ns);
        if ((int)rel_ok_.size() < ns) { rel_ok_.resize(ns, 0); rel_.resize((size_t)ns * Worker::SPEC_ROW, 0); }
        bg_.resize(N + 1);
        lk.unlock();
        set_max_seqs(N + 1);  // (no group runs yet)
        lk.lock();
    }
    bst_.calls++;
    bk_ = k;
    if (!any) bbases_ = bases;  // (with members the contents are equal (checked above) and running groups read it: review pp L5)
    std::vector<char> in(bs_.size(), 0);
    for (const BgReq& r : req) in.at(r.slot) = 1;
    for (int x = 0; x < (int)bs_.size(); x++) {  // members the caller no longer steps leave their group
        BgSlot& b = bs_[x];
        if (!b.member || in[x] || b.group < 0 || b.leave) continue;
        b.leave = true;
        bst_.leaves++;
        BgGroup& g = bg_[b.group];
        if (!b.inflight && !g.running) {
            g.mem.erase(std::remove(g.mem.begin(), g.mem.end(), x), g.mem.end());
            b.group = -1;
        }
    }
    for (const BgReq& r : req) {
        BgSlot& b = bs_[r.slot];
        if (!b.member) {
            b = BgSlot();
            b.member = true;
            bst_.joins++;
        }
        b.leave = false;
        if (b.group < 0) {
            // groups at most: N (one per stage; the persistent pipeline has no host turn-around at a call boundary to
            // cover, and fewer groups = more rows per batched step: 3 cards c=8 676 vs 590 tok/s, c=4 441 vs 393)
            static const int GM = std::max(1, std::min(N + 1, env_int("FLASH_PPB_GROUPS", N)));
            int best = 0;
            for (int gi = 1; gi < GM; gi++) if (bg_[gi].mem.size() < bg_[best].mem.size()) best = gi;
            bg_[best].mem.push_back(r.slot);
            b.group = best;
        }
        b.ahead = std::max(1, r.ahead);
        b.room = std::max(b.room, r.room);
        if (!b.inflight && b.toks.empty()) {  // nothing ahead of the caller: its view of the slot is current
            b.rem = r.rem;
            b.lim = r.lim;
            b.pos = r.pos;
            b.room = r.room;
            b.stop = b.bad = false;
        }
        if (b.stop && !b.bad && b.room_stop && b.pos + k + 3 <= b.room) b.stop = b.room_stop = false;  // more room now
        b.park = (int)b.steps.size() >= b.ahead || (int)b.toks.size() + k + 1 > 64;
    }
    for (int gi = 0; gi <= N; gi++) {
        BgGroup& g = bg_[gi];
        if (g.running) continue;
        bool go = false;
        for (int x : g.mem) go |= !bs_[x].park && !bs_[x].stop && !bs_[x].leave;
        if (go) {
            g.running = true;
            th_.post(0, [this, gi] { bg_task(0, gi); });
        }
    }
    return true;
}
PpEngine::BgTake PpEngine::bg_take(int slot, bool hold) {
    std::unique_lock<std::mutex> lk(bmu_);
    BgTake t;
    if (slot < 0 || slot >= (int)bs_.size()) return t;
    BgSlot& b = bs_[slot];
    bcv_.wait(lk, [&] { return !b.member || !b.steps.empty() || (!b.inflight && (b.stop || b.group < 0 || b.park)); });
    t.toks.swap(b.toks);
    t.steps.swap(b.steps);
    t.lims.swap(b.lims);
    t.bad = b.bad;
    t.stopped = b.stop;
    if (!b.member) return t;
    b.park = hold;  // held (a stop string may be near): the next step waits for the caller's next call
    if (!b.park && !b.stop && b.group >= 0 && !bg_[b.group].running) {
        bg_[b.group].running = true;
        const int gi = b.group;
        th_.post(0, [this, gi] { bg_task(0, gi); });
    }
    return t;
}
int PpEngine::bg_pos(int slot) const {
    std::lock_guard<std::mutex> lk(bmu_);
    return slot >= 0 && slot < (int)bs_.size() && bs_[slot].member ? bs_[slot].pos : -1;
}
bool PpEngine::bg_active() const {
    std::lock_guard<std::mutex> lk(bmu_);
    for (auto& b : bs_) if (b.member) return true;
    return false;
}
PpEngine::BgStats PpEngine::bg_stats() const {
    std::lock_guard<std::mutex> lk(bmu_);
    return bst_;
}
void PpEngine::bg_stop(const std::function<void(int, BgTake&)>& fn) {
    std::vector<std::pair<int, BgTake>> out;
    {
        std::unique_lock<std::mutex> lk(bmu_);
        bool any = false;
        for (auto& b : bs_) any |= b.member;
        if (!any) return;
        bstop_ = true;
        bcv_.wait(lk, [&] {
            for (auto& g : bg_) if (g.running) return false;
            return true;
        });
        bstop_ = false;
        if (!pr_n_.empty()) {
            const double el = pp_now_ms() - pr_t0_;
            std::string o;
            for (int s = 0; s < nstage(); s++) {
                char b[160];
                snprintf(b, sizeof b, " | stage %d: %lld tasks, %.2f ms wall / %.2f ms GPU each, %.1f rows, busy %.0f%%", s, pr_n_[s],
                         pr_wall_[s] / std::max(1LL, pr_n_[s]), pr_gpu_[s] / std::max(1LL, pr_n_[s]), pr_rows_[s] / std::max(1LL, pr_n_[s]),
                         100 * pr_wall_[s] / std::max(1e-9, el));
                o += b;
            }
            fprintf(stderr, "[qflash] PP decode profile over %.0f ms%s\n", el, o.c_str());
            pr_n_.clear();
        }
        for (int x = 0; x < (int)bs_.size(); x++) {
            BgSlot& b = bs_[x];
            if (!b.member) continue;
            BgTake t;
            t.toks.swap(b.toks);
            t.steps.swap(b.steps);
            t.lims.swap(b.lims);
            t.bad = b.bad;
            t.stopped = b.stop;
            out.emplace_back(x, std::move(t));
            b = BgSlot();
        }
        for (auto& g : bg_) g.mem.clear();
    }
    for (auto& o : out) fn(o.first, o.second);
}
// one group's batched step on stage s; stage 0 picks the runnable members, the last stage files the tokens per slot
void PpEngine::bg_task(int s, int gi) {
    const int N = nstage();
    BgGroup& g = bg_[gi];
    if (s == 0) {
        std::lock_guard<std::mutex> lk(bmu_);
        for (size_t i = 0; i < g.mem.size();) {  // members that left (nothing in flight)
            BgSlot& b = bs_[g.mem[i]];
            if (b.leave) { b.group = -1; g.mem.erase(g.mem.begin() + i); } else i++;
        }
        g.cur.clear();
        g.curlim.clear();
        if (!bstop_)
            for (int x : g.mem) {
                BgSlot& b = bs_[x];
                if (b.park || b.stop) continue;
                g.cur.push_back(x);
                g.curlim.push_back(b.lim);
                b.inflight = true;
            }
        if (g.cur.empty()) {
            g.running = false;
            bcv_.notify_all();
            return;
        }
    }
    static const bool prof = env_int("FLASH_PPB_PROF", 0) != 0;
    const double tw0 = prof ? pp_now_ms() : 0;
    Worker* w = W_[s];
    const std::vector<int>& sl = g.cur;  // (changes only at this group's stage-0 entry, after its last stage)
    const int n = (int)sl.size();
    std::vector<int> ss, srows;
    if (s + 1 < N)
        for (int x : sl)
            if (rel_ok_[x]) {
                ss.push_back(x);
                srows.insert(srows.end(), rel_.begin() + (size_t)x * Worker::SPEC_ROW, rel_.begin() + (size_t)(x + 1) * Worker::SPEC_ROW);
            }
    g.out.assign((size_t)n * 64, 0);
    g.cnt.assign(n, 0);
    std::atomic<unsigned> nfa{0};
    each(s, [&](int r, Worker* x) {
        if (!ss.empty()) x->spec_state_set(ss.data(), (int)ss.size(), srows.data());
        io(s, r, x, false, gi);
        std::vector<int> o1((size_t)n * 64), c1(n);
        x->batch_launch(sl.data(), bbases_.data(), g.curlim.data(), n, bk_);
        x->batch_finish(r ? o1.data() : g.out.data(), r ? c1.data() : g.cnt.data());
        if (s + 1 == N) nfa.fetch_or(x->take_nf());
    });
    if (prof) {
        std::lock_guard<std::mutex> lk(bmu_);
        if (pr_n_.empty()) { pr_n_.assign(N, 0); pr_wall_.assign(N, 0); pr_gpu_.assign(N, 0); pr_rows_.assign(N, 0); pr_t0_ = tw0; }
        pr_n_[s]++;
        pr_wall_[s] += pp_now_ms() - tw0;
        pr_gpu_[s] += w->last_batch_ms();
        pr_rows_[s] += n * (bk_ + 1);
    }
    if (s + 1 < N) {
        th_.post(s + 1, [this, s, gi] { bg_task(s + 1, gi); });
        return;
    }
    const unsigned nf = nfa.load();  // rows with non-finite values: only their slots stop (and fail)
    {
        std::vector<int> rows((size_t)n * Worker::SPEC_ROW);
        w->spec_state_get(sl.data(), n, rows.data());
        for (int i = 0; i < n; i++) {
            std::copy(rows.begin() + (size_t)i * Worker::SPEC_ROW, rows.begin() + (size_t)(i + 1) * Worker::SPEC_ROW,
                      rel_.begin() + (size_t)sl[i] * Worker::SPEC_ROW);
            rel_ok_[sl[i]] = 1;
        }
    }
    const int k = bk_, rps = k + 1;
    {
        std::lock_guard<std::mutex> lk(bmu_);
        bst_.steps++;
        for (int i = 0; i < n; i++) {
            BgSlot& b = bs_[sl[i]];
            const int c = g.cnt[i];
            const int* t = g.out.data() + (size_t)i * 64;
            b.toks.insert(b.toks.end(), t, t + c);
            b.steps.push_back(c);
            b.lims.push_back(g.curlim[i]);
            b.pos += c;
            b.inflight = false;
            const unsigned rmask = (g.curlim[i] + 1 >= 32 ? ~0u : ((1u << (g.curlim[i] + 1)) - 1));
            const bool bad = i * rps < 32 && (nf & (rmask << (i * rps)));
            bool eos = false;
            for (int j = 0; j < c; j++) eos |= t[j] == EOS || t[j] == EOS + 2;
            if (b.rem >= 0) b.rem = std::max(0, b.rem - c);
            b.lim = b.rem > 0 ? std::min(k, b.rem - 1) : k;
            if (bad || eos || b.rem == 0) {
                b.stop = true;
                b.bad |= bad;
                bst_.stops++;
            } else if (b.pos + rps + 2 > b.room) {
                b.stop = b.room_stop = true;
                bst_.stops++;
            }
            if ((int)b.steps.size() >= b.ahead || (int)b.toks.size() + rps > 64) {
                b.park = true;
                bst_.parks++;
            }
        }
        bcv_.notify_all();
    }
    th_.post(0, [this, gi] { bg_task(0, gi); });
}

// ---------------------------------------------------------------- mode A hand-over
// the hand-over moves pooled indexer keys of complete blocks only; the open block at the prompt end lives in the raw key
// ring (ik_tail, copied whole), which must hold at least one block of positions
static_assert(IDX_TAIL >= IDX_R, "the indexer key ring must cover the open block of the hand-over");

// The stage that holds layer li (li = LAYERS: the MTP block, on the last stage)
static Worker* pp_owner(const std::vector<Worker*>& stages, int li) {
    for (Worker* w : stages)
        if (li < LAYERS ? w->layer_here(li) : w->plan().last()) return w;
    return nullptr;
}

PpXfer::PpXfer(const std::vector<Worker*>& stages, const std::vector<Worker*>& tp) : S_(stages), T_(tp) {
    int dev0 = 0;
    CUDA_CHECK(cudaGetDevice(&dev0));
    ss_.resize(T_.size() * S_.size());
    for (size_t r = 0; r < T_.size(); r++) {
        CUDA_CHECK(cudaSetDevice(T_[r]->device()));
        for (size_t s = 0; s < S_.size(); s++) CUDA_CHECK(cudaStreamCreateWithFlags(&ss_[r * S_.size() + s], cudaStreamNonBlocking));
    }
    for (Worker* w : S_) sb_.push_back((long long)w->cur_base() * w->plan().own);
    for (Worker* w : T_) tb_.push_back((long long)w->cur_base() * w->plan().own);
    ev_.resize(S_.size());
    for (size_t s = 0; s < S_.size(); s++) {
        CUDA_CHECK(cudaSetDevice(S_[s]->device()));
        CUDA_CHECK(cudaEventCreateWithFlags(&ev_[s], cudaEventDisableTiming));
    }
    CUDA_CHECK(cudaSetDevice(dev0));
}
PpXfer::~PpXfer() {
    for (cudaStream_t x : ss_) cudaStreamDestroy(x);
    for (cudaEvent_t e : ev_) cudaEventDestroy(e);
}
void PpXfer::wait() {
    for (size_t r = 0; r < T_.size(); r++) {
        CUDA_CHECK(cudaSetDevice(T_[r]->device()));
        for (size_t s = 0; s < S_.size(); s++) CUDA_CHECK(cudaStreamSynchronize(ss_[r * S_.size() + s]));
    }
}
void PpXfer::after_stage(int s) {
    CUDA_CHECK(cudaSetDevice(S_[s]->device()));
    CUDA_CHECK(cudaEventRecord(ev_[s], S_[s]->stream()));
    for (size_t r = 0; r < T_.size(); r++) {
        CUDA_CHECK(cudaSetDevice(T_[r]->device()));
        CUDA_CHECK(cudaStreamWaitEvent(ss_[r * S_.size() + s], ev_[s], 0));
    }
}

// one copy between a TP rank's memory and a stage's (dir 0: stage -> TP, 1: TP -> stage) on the stream of (rank r,
// stage s); a copy across cards goes peer-to-peer (staged by the driver without peer access)
void PpXfer::cp(size_t r, int s, void* tpp, const void* stp_c, size_t bytes, int dir) {
    if (!bytes) return;
    void* stp = const_cast<void*>(stp_c);
    const int td = T_[r]->device(), sd = S_[s]->device();
    cudaStream_t st = ss_[r * S_.size() + s];
    CUDA_CHECK(cudaSetDevice(td));
    void* dst = dir ? stp : tpp;
    const void* src = dir ? tpp : stp;
    const int ddev = dir ? sd : td, sdev = dir ? td : sd;
    cudaError_t e = ddev == sdev ? cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToDevice, st)
                                 : cudaMemcpyPeerAsync(dst, ddev, src, sdev, bytes, st);
    CHECK(e == cudaSuccess, "flash: PP transfer copy %s (layer %d, TP rank %zu, stage %d, dir %d): %s; %zu bytes", what_, what_li_, r, s, dir,
          cudaGetErrorString(e), bytes);
    moved_ += bytes;
}
// strided: height rows of width bytes; pitches in bytes on each side
void PpXfer::cp2d(size_t r, int s, char* tpp, size_t tpitch, const char* stp, size_t spitch, size_t width, size_t height, int dir) {
    if (!width || !height) return;
    const int td = T_[r]->device(), sd = S_[s]->device();
    cudaStream_t st = ss_[r * S_.size() + s];
    CUDA_CHECK(cudaSetDevice(td));
    char* dst = dir ? const_cast<char*>(stp) : tpp;
    const char* src = dir ? tpp : stp;
    const size_t dp = dir ? spitch : tpitch, sp = dir ? tpitch : spitch;
    const int ddev = dir ? sd : td, sdev = dir ? td : sd;
    cudaError_t e;
    if (ddev == sdev) {
        e = cudaMemcpy2DAsync(dst, dp, src, sp, width, height, cudaMemcpyDeviceToDevice, st);
    } else {
        cudaMemcpy3DPeerParms pm = {};
        pm.srcPtr = make_cudaPitchedPtr((void*)src, sp, width, height);
        pm.srcDevice = sdev;
        pm.dstPtr = make_cudaPitchedPtr(dst, dp, width, height);
        pm.dstDevice = ddev;
        pm.extent = make_cudaExtent(width, height, 1);
        e = cudaMemcpy3DPeerAsync(&pm, st);
    }
    CHECK(e == cudaSuccess, "flash: PP transfer strided copy %s (layer %d, TP rank %zu, stage %d, dir %d): %s", what_, what_li_, r, s, dir,
          cudaGetErrorString(e));
    moved_ += width * height;
}

// KV rows of positions [t0, t1) of the QSA layers held by stage `only` (-1: every stage) and the MTP block (mtp).
// Stage rows: row = position (period = own); TP rank rows: its `own` positions of every `period`. Full runs of a rank go
// as one strided copy per array, the partial first / last runs as plain copies. Pooled indexer keys: the blocks
// [t0 / IDX_R, t1 / IDX_R) (a block starting before t0 was completed by the side that wrote t0's rows; an open block at
// t1 lives in the raw key ring, which state() copies).
void PpXfer::kv(int t0, int t1, bool mtp, int dir, int only) {
    if (t1 <= t0) return;
    std::lock_guard<std::mutex> lk(mu_);
    for (int li = 0; li <= LAYERS; li++) {
        if (li < LAYERS && !flash_pp::is_qsa(li)) continue;
        if (li == LAYERS && !mtp) continue;
        Worker* sw = pp_owner(S_, li);
        CHECK(sw, "flash: no pipeline stage holds layer %d", li);
        const int s = sw->plan().stage;
        if (only >= 0 && s != only) continue;
        const Worker::KvArrays a = sw->kv_arrays(li);
        const long long sb = sb_[s];
        for (size_t r = 0; r < T_.size(); r++) {
            const Worker::KvArrays b = T_[r]->kv_arrays(li);
            const Plan& q = T_[r]->plan();
            const long long tb = tb_[r];
            auto row = [&](const Plan& pl, long long p, long long base) { return base + p / pl.period * pl.own + p % pl.period - pl.own0; };
            // this rank's runs intersecting [t0, t1): period k covers [k * period + own0, + own)
            const long long k0 = std::max(0LL, (long long)(t0 - q.own0) / q.period), k1 = (long long)(t1 - 1 - q.own0) / q.period;  // (t1 - 1 >= own0 checked below)
            if (t1 - 1 < q.own0) continue;
            std::vector<std::pair<long long, long long>> part;  // partial runs [a, b)
            long long f0 = -1, f1 = -1;                          // full runs: periods [f0, f1)
            for (long long k = k0; k <= k1; k++) {
                const long long ps = k * q.period + q.own0, pe = ps + q.own;
                const long long x = std::max<long long>(ps, (long long)t0), y = std::min<long long>(pe, (long long)t1);
                if (y <= x) continue;
                if (x == ps && y == pe) {  // (full runs are consecutive: only the first / last run can be partial)
                    if (f0 < 0) f0 = k;
                    f1 = k + 1;
                } else {
                    part.push_back({x, y});
                }
            }
            auto arr = [&](char* tpa, const char* sta, size_t rowb, int div) {
                // div: positions per record (IDX_R for the pooled keys: whole blocks only)
                if (f0 >= 0) {
                    const long long p = f0 * q.period + q.own0;
                    const long long sr = row(sw->plan(), p, sb), dr = row(q, p, tb);
                    cp2d(r, s, tpa + dr / div * rowb, (size_t)q.own / div * rowb, sta + sr / div * rowb, (size_t)q.period / div * rowb,
                         (size_t)q.own / div * rowb, (size_t)(f1 - f0), dir);
                }
                for (auto& [x, y] : part) {
                    const long long bx = x / div, by = y / div;  // records [bx, by) of the run
                    if (by <= bx) continue;
                    const long long px = bx * div;               // (a block starts inside this rank's run: runs are block-aligned)
                    const long long sr = row(sw->plan(), px, sb), dr = row(q, px, tb);
                    cp(r, s, tpa + dr / div * rowb, sta + sr / div * rowb, (size_t)(by - bx) * rowb, dir);
                }
            };
            what_li_ = li;
            what_ = "kc"; arr((char*)b.kc, (const char*)a.kc, (size_t)NKV * HD, 1);
            what_ = "vc"; arr((char*)b.vc, (const char*)a.vc, (size_t)NKV * HD, 1);
            what_ = "ks"; arr((char*)b.ks, (const char*)a.ks, (size_t)NKV * 4, 1);
            what_ = "vs"; arr((char*)b.vs, (const char*)a.vs, (size_t)NKV * 4, 1);
            what_ = "ikb"; arr((char*)b.ikb, (const char*)a.ikb, (size_t)IDX_HD * 2, IDX_R);
        }
    }
}

// recurrent state of the bound slots at the current position: DeltaNet S by value heads + conv history by channel
// blocks, the indexer raw-key rings, the PLE history, the MTP chain row. dir 1 (TP -> stage): replicated parts come from
// TP rank 0; a stage's key-head channels shared by two ranks are written by both (equal values).
void PpXfer::state(bool mtp, int dir) {
    for (int li = 0; li < LAYERS; li++) {
        Worker* sw = pp_owner(S_, li);
        const int s = sw->plan().stage;
        if (flash_pp::is_qsa(li)) {
            const Worker::KvArrays a = sw->kv_arrays(li);
            what_ = "ik_tail", what_li_ = li;
            for (size_t r = 0; r < T_.size(); r++) {
                if (dir && r) break;
                cp(r, s, T_[r]->kv_arrays(li).ik_tail, a.ik_tail, (size_t)IDX_TAIL * IDX_HD * 4, dir);
            }
            continue;
        }
        const Worker::DnState a = sw->dn_state(li);
        const int chs = 2 * (a.k1 - a.k0) * DK + (a.v1 - a.v0) * DV;
        for (size_t r = 0; r < T_.size(); r++) {
            const Worker::DnState b = T_[r]->dn_state(li);
            const int kl = b.k1 - b.k0, vl = b.v1 - b.v0, chd = 2 * kl * DK + vl * DV;
            what_ = "dn S / conv", what_li_ = li;
            cp(r, s, b.S, a.S + (size_t)(b.v0 - a.v0) * DK * DV, (size_t)vl * DK * DV * 4, dir);
            const int sk = (a.k1 - a.k0) * DK;  // the stage's q (and k) channel block width
            for (int row = 0; row < 3; row++) {
                const float* sr = a.conv_hist + (size_t)row * chs;
                float* dr = b.conv_hist + (size_t)row * chd;
                cp(r, s, dr, sr + (size_t)(b.k0 - a.k0) * DK, (size_t)kl * DK * 4, dir);                                   // q
                cp(r, s, dr + (size_t)kl * DK, sr + sk + (size_t)(b.k0 - a.k0) * DK, (size_t)kl * DK * 4, dir);            // k
                cp(r, s, dr + (size_t)2 * kl * DK, sr + 2 * sk + (size_t)(b.v0 - a.v0) * DV, (size_t)vl * DV * 4, dir);    // v
            }
        }
    }
    if (mtp) {
        Worker* sw = S_.back();
        what_ = "mtp chain row", what_li_ = LAYERS;
        what_ = "mtp ik_tail";
        const Worker::KvArrays a = sw->kv_arrays(LAYERS);
        for (size_t r = 0; r < T_.size(); r++) {
            if (dir && r) break;
            if (sw->mtp_chain_row() && T_[r]->mtp_chain_row())
                cp(r, (int)S_.size() - 1, T_[r]->mtp_chain_row(), sw->mtp_chain_row(), (size_t)RW * 4, dir);
            cp(r, (int)S_.size() - 1, T_[r]->kv_arrays(LAYERS).ik_tail, a.ik_tail, (size_t)IDX_TAIL * IDX_HD * 4, dir);
        }
    }
    if (Worker* sw = pp_owner(S_, PLE_LAYER)) {
        what_ = "ple hist", what_li_ = PLE_LAYER;
        float* h = sw->ple_hist();
        for (size_t r = 0; r < T_.size(); r++) {
            if (dir && r) break;
            if (float* d = T_[r]->ple_hist(); d && h) cp(r, sw->plan().stage, d, h, (size_t)9 * RW * 4, dir);
        }
    }
}

// before copies into / out of the TP ranks' bound slot: its KV rows of [0, n) claimed back from the expert cache, the
// recurrent state in the workspace, every worker's own stream drained (the copies run on separate streams)
void pp_xfer_prepare(const std::vector<Worker*>& stages, const std::vector<Worker*>& tp, int n) {
    auto first_dn = []() { for (int li = 0; li < LAYERS; li++) if (!flash_pp::is_qsa(li)) return li; return -1; };
    for (Worker* t : tp) {
        CUDA_CHECK(cudaSetDevice(t->device()));
        t->kv_claim(n);
        t->dn_state(first_dn());
        t->sync();
    }
    for (Worker* w : stages) {
        for (int li = w->plan().l0; li < w->plan().l1; li++)
            if (!flash_pp::is_qsa(li)) { w->dn_state(li); break; }
        w->sync();
    }
}

size_t pp_handover(const std::vector<Worker*>& stages, const std::vector<Worker*>& tp, int n, bool mtp) {
    pp_xfer_prepare(stages, tp, n);
    PpXfer x(stages, tp);
    x.kv(0, n, mtp, 0);
    x.state(mtp, 0);
    x.wait();
    return x.moved();
}

// ---------------------------------------------------------------- set-up helpers (flash_run, the server)
// stage bounds: FLASH_PP_LAYERS=a,b,... or the planner's balance of the cost model at the prompt's mean context; mode B
// (decode through the stages too) blends in the decode cost (FLASH_PP_DECODE_W, default 0.5), which takes the head + MTP
// work of the last stage into account
static std::vector<flash_pp::CardSpeed> g_cards;
static std::vector<int> g_card_devs;  // the device of each g_cards entry (pp_calibrate's list order)
void pp_set_cards(const std::vector<flash_pp::CardSpeed>& c) { g_cards = c; }
const std::vector<flash_pp::CardSpeed>& pp_cards() { return g_cards; }
std::string pp_calibrate(const std::vector<int>& devs) {
    const int n_slow = test_slow_cards();
    std::vector<CardCal> cal;
    for (int d : devs) cal.push_back(calibrate_card(d));
    std::vector<double> tf, bw;
    for (auto& c : cal) if (c.ok) { tf.push_back(c.tflops); bw.push_back(c.gbs); }
    if (tf.size() != devs.size()) return "calibration failed";
    // test hook FLASH_TEST_CAL_NOISE=<ordinal>:<factor>: that card's first (short) measurement is scaled, as a one-off
    // misreading would be (a card still clocking up); the re-check below must not see it
    if (const char* nz = env_str("FLASH_TEST_CAL_NOISE")) {
        int d = -1;
        double f = 1;
        if (sscanf(nz, "%d:%lf", &d, &f) == 2)
            for (size_t i = 0; i < devs.size(); i++)
                if (devs[i] == d) { tf[i] *= f; bw[i] *= f; fprintf(stderr, "[test] calibration of GPU%d scaled by %.2f (FLASH_TEST_CAL_NOISE)\n", d, f); }
    }
    // cards of the same model (name, SM count, memory) that measured apart: re-measure with longer kernels; only a gap
    // that holds there counts (review pp M2: a ~0.3 ms kernel on a card still clocking up, or a display card busy for a
    // moment, must not flip the default pick away from TP on equal cards)
    bool same_hw = true;
    for (auto& c : cal)
        same_hw = same_hw && c.sms == cal[0].sms && std::abs(c.vram - cal[0].vram) < 1e9 && !strcmp(c.name, cal[0].name);  // (one
    // card's VBIOS / ReBAR set-up can report a few hundred MB less: 25.3 vs 25.4 GB on the same model)
    auto spread = [](const std::vector<double>& v) {
        const auto [a, b] = std::minmax_element(v.begin(), v.end());
        return *b / std::max(1e-9, *a);
    };
    std::string recheck;
    // (FLASH_TEST_SLOW_CARD stands for other hardware: its SM hog yields to long kernels, a re-check would hide it)
    if (n_slow == 0 && same_hw && cal.size() > 1 && (spread(tf) > 1.10 || spread(bw) > 1.10)) {
        const double s0 = std::max(spread(tf), spread(bw));
        std::vector<double> tf2, bw2;
        for (int d : devs) {
            CardCal c = calibrate_card(d, std::max(1, env_int("FLASH_CAL_RECHECK", 64)));
            if (!c.ok) break;
            tf2.push_back(c.tflops);
            bw2.push_back(c.gbs);
        }
        if (tf2.size() == devs.size()) {
            tf = tf2;
            bw = bw2;
            for (size_t i = 0; i < cal.size(); i++) { cal[i].tflops = tf[i]; cal[i].gbs = bw[i]; }
        }
        char rb[160];
        snprintf(rb, sizeof rb, "; equal cards measured %.2fx apart, re-check with longer kernels: %.2fx", s0,
                 std::max(spread(tf), spread(bw)));
        recheck = rb;
    }
    std::vector<double> t2 = tf, b2 = bw;
    std::sort(t2.begin(), t2.end());
    std::sort(b2.begin(), b2.end());
    const double tm = t2[t2.size() / 2], bm = b2[b2.size() / 2];
    std::vector<flash_pp::CardSpeed> cs;
    // prefill: compute-bound (the GEMMs); decode: weight reads (bandwidth). FLASH_PP_CARD_SPEED=a,b,c overrides both
    // (relative speeds, tests)
    std::vector<double> ov;
    if (const char* e = env_str("FLASH_PP_CARD_SPEED"))
        for (const char* q = e; *q;) { ov.push_back(atof(q)); while (*q && *q != ',') q++; if (*q) q++; }
    for (size_t i = 0; i < cal.size(); i++) {
        flash_pp::CardSpeed c;
        c.pf = ov.size() == cal.size() ? ov[i] : tf[i] / tm;
        c.dc = ov.size() == cal.size() ? ov[i] : bw[i] / bm;
        c.free = std::max(0.0, cal[i].free - 1.5e9);
        cs.push_back(c);
    }
    pp_set_cards(cs);
    g_card_devs = devs;
    std::string line = card_line(cal);
    char b[96];
    for (size_t i = 0; i < cs.size(); i++) {
        snprintf(b, sizeof b, "%s%.2f/%.2f", i ? ", " : "; relative prefill/decode speed ", cs[i].pf, cs[i].dc);
        line += b;
    }
    line += recheck;
    line += flash_pp::cards_uneven(cs) ? " (uneven: the pipeline splits by them)" : " (even)";
    return line;
}

std::vector<int> pp_bounds(int n, long long np, bool decode, std::string* err) {
    std::vector<int> b = flash_pp::parse_layers(env_str("FLASH_PP_LAYERS"), n, err);
    if (!err->empty()) return {};
    if (!b.empty()) return b;
    const flash_pp::CostModel cm;
    const double w = decode ? std::clamp(env_double("FLASH_PP_DECODE_W", 0.5), 0.0, 1.0) : 0.0;
    if ((int)g_cards.size() == n && flash_pp::cards_uneven(g_cards) && env_int("FLASH_PP_CARD_SPLIT", 1)) {
        const double h2d = env_double("FLASH_PP_H2D_GBPS", 22.0) * 1e9;
        return flash_pp::partition_cards(n, cm, np / 2.0, flash_pp::DecodeCost{}, w, g_cards, h2d);
    }
    return flash_pp::partition_mixed(n, cm, np / 2.0, flash_pp::DecodeCost{}, w);
}
// peer-copy integrity of a card pair (both directions, 8 MB pattern): the hop rows must arrive bit-exact; a broken peer
// path (seen with patched P2P drivers) keeps the boundary on host staging
bool pp_p2p_ok(int a, int b) {
    const size_t n = (size_t)8 << 20;
    std::vector<unsigned> h(n / 4), r(n / 4);
    for (size_t i = 0; i < h.size(); i++) h[i] = (unsigned)(i * 2654435761u) ^ 0x9e3779b9u;
    bool ok = true;
    for (int dir = 0; dir < 2 && ok; dir++) {
        const int s = dir ? b : a, d = dir ? a : b;
        void *ps = nullptr, *pd = nullptr;
        CUDA_CHECK(cudaSetDevice(s));
        CUDA_CHECK(cudaMalloc(&ps, n));
        CUDA_CHECK(cudaMemcpy(ps, h.data(), n, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaSetDevice(d));
        CUDA_CHECK(cudaMalloc(&pd, n));
        CUDA_CHECK(cudaMemcpyPeer(pd, d, ps, s, n));
        CUDA_CHECK(cudaMemcpy(r.data(), pd, n, cudaMemcpyDeviceToHost));
        ok = memcmp(h.data(), r.data(), n) == 0;
        CUDA_CHECK(cudaFree(pd));
        CUDA_CHECK(cudaSetDevice(s));
        CUDA_CHECK(cudaFree(ps));
    }
    if (!ok) fprintf(stderr, "[flash] pipeline: peer copies between GPU%d and GPU%d are not bit-exact: host staging there\n", a, b);
    return ok;
}

// one stage worker per card, loaded in parallel (the MTP block on the last stage only)
std::vector<Worker*> pp_make_stages(const std::vector<int>& devs, const std::vector<int>& bounds, int ctx, const SafeTensors& st,
                                           const HostTables& ht, const SafeTensors* mst, const std::vector<std::vector<int>>& cold,
                                           const std::vector<float>& prior, const std::vector<int>& draft_ids, const NgramDev& ngd,
                                           const std::vector<const Worker*>& views) {
    const int n = (int)devs.size();
    std::vector<Plan> plans = make_stage_plans(bounds);
    std::vector<Worker*> S(n);
    // the hop between neighbours goes peer-to-peer (one copy into the next stage's receive buffers) where the cards have
    // peer access, unless FLASH_PP_P2P=0; else through pinned host memory (D2H + H2D). Peer access is enabled here, before
    // the loads (the receive buffers are allocated ahead of the expert budget)
    std::vector<char> rx(n, 0);
    for (int s = 1; s < n && env_int("FLASH_PP_P2P", 1); s++) {
        int a = 0, b = 0;
        if (cudaDeviceCanAccessPeer(&a, devs[s - 1], devs[s]) != cudaSuccess || cudaDeviceCanAccessPeer(&b, devs[s], devs[s - 1]) != cudaSuccess || !a || !b) continue;
        CUDA_CHECK(cudaSetDevice(devs[s - 1]));
        cudaError_t e1 = cudaDeviceEnablePeerAccess(devs[s], 0);
        CUDA_CHECK(cudaSetDevice(devs[s]));
        cudaError_t e2 = cudaDeviceEnablePeerAccess(devs[s - 1], 0);
        (void)cudaGetLastError();  // (already enabled, e.g. by the TP ranks' comm in mode A: fine)
        rx[s] = (e1 == cudaSuccess || e1 == cudaErrorPeerAccessAlreadyEnabled) && (e2 == cudaSuccess || e2 == cudaErrorPeerAccessAlreadyEnabled) &&
                pp_p2p_ok(devs[s - 1], devs[s]);
    }
    // several stages on one card (a test of the N-way split on fewer cards, e.g. 0,1,2,0): that card's stages load one
    // after the other, each sizing its expert cache to its share of what is free then (1 / the stages still to load)
    std::vector<std::thread> th;
    std::vector<int> udev;
    for (int d : devs) if (std::find(udev.begin(), udev.end(), d) == udev.end()) udev.push_back(d);
    for (int d : udev)
        th.emplace_back([&, d] {
            std::vector<int> mine;
            for (int s = 0; s < n; s++) if (devs[s] == d) mine.push_back(s);
            for (size_t j = 0; j < mine.size(); j++) {
                const int s = mine[j];
                CUDA_CHECK(cudaSetDevice(devs[s]));
                Comm c1 = comm_create(1, ARU_MAX * H, H);  // a stage's collectives are local (no peers)
                c1.dev[0] = devs[s];
                S[s] = new Worker(plans[s], devs[s], c1, ctx);
                S[s]->set_cold(cold);
                S[s]->set_expert_prior(prior);
                S[s]->set_draft_vocab(draft_ids);
                S[s]->set_pp_rx(rx[s]);
                if (mine.size() > 1) S[s]->set_card_share(1.0 / (double)(mine.size() - j));
                if (s < (int)views.size()) S[s]->set_views(views, s);  // the TP rank on this card
                S[s]->load(st, ht, s == n - 1 ? mst : nullptr);
                S[s]->alloc_state();
            }
        });
    for (auto& t : th) t.join();
    for (Worker* w : S) w->set_ngram(ngd);
    return S;
}
std::string layout_report(const std::vector<Worker*>& W) {
    std::string o;
    char b[400];
    for (size_t i = 0; i < W.size(); i++) {
        const Plan& p = W[i]->plan();
        snprintf(b, sizeof b, "[layout] worker %zu GPU%d: stage %d of %d, rank %d of %d, layers %d-%d%s, %d expert groups per layer, KV period %d "
                 "(own %d from %d), %.2f GB on the card + %.2f GB host-backed experts, %.2f GiB free\n", i, W[i]->device(), p.stage, p.nstage,
                 p.rank, p.ngpu, p.l0, p.l1 - 1, p.last() ? " + head + MTP" : "", plan_groups(p), p.period, p.own, p.own0, W[i]->weight_gb(),
                 W[i]->host_gb(), W[i]->free_bytes() / 1073741824.0);
        o += b;
    }
    return o;
}
bool box_report_wanted(const std::vector<int>& devs, bool pptp) {
    std::vector<int> u;
    for (int d : devs) if (std::find(u.begin(), u.end(), d) == u.end()) u.push_back(d);
    return env_int("FLASH_BOX_REPORT", pptp || u.size() >= 4 || devs.size() >= 4 ? 1 : 0) != 0;
}

// PP x TP: the groups and the stage bounds
std::vector<std::vector<int>> pptp_groups(const std::vector<int>& devs, std::string* err) {
    return flash_pp::parse_stage_map(env_str("FLASH_PP_MAP"), devs, env_int("FLASH_PP_TP", 2), err);
}
std::vector<int> pptp_bounds(const std::vector<std::vector<int>>& groups, long long np, std::string* err) {
    const int n = (int)groups.size();
    std::vector<int> b = flash_pp::parse_layers(env_str("FLASH_PP_LAYERS"), n, err);
    if (!err->empty()) return {};
    if (!b.empty()) return b;
    // each group's slowest card (TP waits on it every layer) from the boot calibration, when it covers these devices
    std::vector<flash_pp::CardSpeed> gmin(n);
    // (looked up by device id in the calibration's own order: --devices need not be ascending; review pp L4)
    const std::vector<flash_pp::CardSpeed>& cs = g_cards;
    bool covered = cs.size() == g_card_devs.size() && !cs.empty();
    for (auto& g : groups)
        for (int d : g) covered = covered && std::find(g_card_devs.begin(), g_card_devs.end(), d) != g_card_devs.end();
    if (covered)
        for (int s = 0; s < n; s++) {
            flash_pp::CardSpeed m{1e30, 1e30, 1e30};
            for (int d : groups[s]) {
                const auto& c = cs[std::find(g_card_devs.begin(), g_card_devs.end(), d) - g_card_devs.begin()];
                m.pf = std::min(m.pf, c.pf);
                m.dc = std::min(m.dc, c.dc);
                m.free = std::min(m.free, c.free);
            }
            gmin[s] = m;
        }
    // several ranks on one card (test layouts): each takes its share of that card's memory
    for (int s = 0; s < n; s++) {
        int share = 1;
        for (int d : groups[s]) {
            int k = 0;
            for (auto& g : groups) for (int x : g) k += x == d;
            share = std::max(share, k);
        }
        gmin[s].free /= share;
    }
    const bool p2p = env_int("QE_P2P", 1) != 0;
    const std::vector<flash_pp::CardSpeed> gs = flash_pp::group_speeds(groups, gmin, p2p);
    const flash_pp::CostModel cm;
    const double w = std::clamp(env_double("FLASH_PP_DECODE_W", 0.5), 0.0, 1.0);
    const double h2d = env_double("FLASH_PP_H2D_GBPS", 22.0) * 1e9;
    return flash_pp::partition_cards(n, cm, np / 2.0, flash_pp::DecodeCost{}, w, gs, h2d);
}

std::vector<Worker*> pp_make_groups(const std::vector<std::vector<int>>& groups, const std::vector<int>& bounds, int ctx,
                                    const SafeTensors& st, const HostTables& ht, const SafeTensors* mst,
                                    const std::vector<std::vector<int>>& cold, const std::vector<float>& prior,
                                    const std::vector<int>& draft_ids, const NgramDev& ngd) {
    const int n = (int)groups.size();
    std::vector<int> tps;
    for (auto& g : groups) tps.push_back((int)g.size());
    const std::vector<Plan> plans = make_stage_plans(bounds, tps);
    std::vector<int> dev, stg, rk;  // per worker (flat)
    for (int s = 0; s < n; s++)
        for (size_t r = 0; r < groups[s].size(); r++) { dev.push_back(groups[s][r]); stg.push_back(s); rk.push_back((int)r); }
    const int nw = (int)dev.size();
    std::vector<int> base(n, 0);
    for (int s = 1; s < n; s++) base[s] = base[s - 1] + tps[s - 1];
    // the groups' comms (t > 1): P2P where every pair of the group has it, the TP exchange path probe; 1-rank stages a
    // local comm. (Comm is a struct of pointers: the workers keep copies; the buffers live for the process.)
    std::vector<Comm> comm(n);
    for (int s = 0; s < n; s++) {
        const int t = tps[s];
        CUDA_CHECK(cudaSetDevice(groups[s][0]));
        comm[s] = comm_create(t, ARU_MAX * H, H);
        if (t > 1) {
            const bool p2p = comm_enable_p2p(comm[s], groups[s].data());
            comm_select_path(comm[s], groups[s].data(), "FLASH_COMM", true);
            fprintf(stderr, "[flash] PP x TP stage %d: TP over GPU%s, peer access %s\n", s, flash_pp::stage_map_str({groups[s]}).c_str(),
                    p2p ? "yes" : "no (host-staged collectives)");
        }
        for (int r = 0; r < t; r++) comm[s].dev[r] = groups[s][r];
    }
    // receive buffers: rank j of stage s reads rank j % t_(s-1) of the previous stage; peer-to-peer where those two cards
    // have peer access (or are the same card), unless FLASH_PP_P2P=0. The engine uses P2P on a boundary only when every
    // receiver of it has a receive buffer.
    std::vector<char> rx(nw, 0);
    for (int s = 1; s < n && env_int("FLASH_PP_P2P", 1); s++)
        for (int j = 0; j < tps[s]; j++) {
            const int a = groups[s - 1][j % tps[s - 1]], b = groups[s][j];
            bool ok = a == b;
            if (!ok) {
                int x = 0, y = 0;
                if (cudaDeviceCanAccessPeer(&x, a, b) == cudaSuccess && cudaDeviceCanAccessPeer(&y, b, a) == cudaSuccess && x && y) {
                    CUDA_CHECK(cudaSetDevice(a));
                    cudaError_t e1 = cudaDeviceEnablePeerAccess(b, 0);
                    CUDA_CHECK(cudaSetDevice(b));
                    cudaError_t e2 = cudaDeviceEnablePeerAccess(a, 0);
                    (void)cudaGetLastError();
                    ok = (e1 == cudaSuccess || e1 == cudaErrorPeerAccessAlreadyEnabled) &&
                         (e2 == cudaSuccess || e2 == cudaErrorPeerAccessAlreadyEnabled) && pp_p2p_ok(a, b);
                }
                (void)cudaGetLastError();
            }
            rx[base[s] + j] = ok;
        }
    // the workers of a card load one after the other, each sizing its expert cache to its share of what is free then
    // (several ranks on one card: test layouts), the cards in parallel
    std::vector<Worker*> W(nw, nullptr);
    std::vector<int> udev;
    for (int d : dev) if (std::find(udev.begin(), udev.end(), d) == udev.end()) udev.push_back(d);
    std::vector<std::thread> th;
    for (int d : udev)
        th.emplace_back([&, d] {
            std::vector<int> mine;
            for (int i = 0; i < nw; i++) if (dev[i] == d) mine.push_back(i);
            for (size_t j = 0; j < mine.size(); j++) {
                const int i = mine[j], s = stg[i];
                CUDA_CHECK(cudaSetDevice(d));
                W[i] = new Worker(plans[i], d, comm[s], ctx);
                W[i]->set_cold(cold);
                W[i]->set_expert_prior(prior);
                W[i]->set_draft_vocab(draft_ids);
                W[i]->set_pp_rx(rx[i]);
                if (mine.size() > 1) W[i]->set_card_share(1.0 / (double)(mine.size() - j));
                W[i]->load(st, ht, s == n - 1 ? mst : nullptr);
                W[i]->alloc_state();
            }
        });
    for (auto& t : th) t.join();
    // the TP exchange peers inside every group (as the TP server does over all its ranks)
    for (int s = 0; s < n; s++)
        for (int r = 0; r < tps[s] && tps[s] > 1; r++) {
            Worker* w = W[base[s] + r];
            for (int g = 0; g < tps[s]; g++) {
                Worker* x = W[base[s] + g];
                w->set_peer(g, x->xrecv(), x->xflags());
                w->set_qpeer(g, x->xqrecv(), x->xqflags());
                w->set_ipeer(g, x->xirecv(), x->xiflags());
                w->set_agpeer(g, x);
            }
        }
    for (Worker* w : W) w->set_ngram(ngd);
    return W;
}

// micro-batch rows: FLASH_PP_MB, else the planner's best for this prompt from the stages' host-backed expert bytes
int pp_microbatch(const std::vector<Worker*>& Wall, const std::vector<int>& b, long long np, double* wall_ms, long long ctx0, double ux, int lo) {
    if (env_set("FLASH_PP_MB")) return std::max(PFM, env_int("FLASH_PP_MB", 4096) / PFM * PFM);
    const flash_pp::CostModel cm;
    std::vector<flash_pp::PfStage> st;
    // the stages' rank 0 (PP x TP: a stage's compute shrinks by its TP speed-up, its host-backed bytes are per card)
    std::vector<Worker*> S;
    std::vector<int> T;
    for (Worker* w : Wall) {
        if (w->plan().rank == 0) { S.push_back(w); T.push_back(w->plan().ngpu); }
    }
    // the host link: FLASH_PP_H2D_GBPS (the server sets it from its startup probe), else the slow-link emulation's rate
    // when it slows the staging too (FLASH_TEST_LINK_HOST), else 22 GB/s (PCIe 4.0 x16)
    const double emu = env_int("FLASH_TEST_LINK_HOST", 0) ? env_double("FLASH_TEST_LINK_GBPS", 0.0) : 0.0;
    const double h2d = env_double("FLASH_PP_H2D_GBPS", emu > 0 ? emu : 22.0) * 1e9;
    for (size_t s = 0; s < S.size(); s++) {
        flash_pp::PfStage p;
        for (int li = b[s]; li < b[s + 1]; li++) p.u += cm.layer(li, ctx0 + np / 2.0);
        if (s + 1 == S.size()) p.u += cm.head_last;
        p.u *= ux / flash_pp::group_speed_prefill(T[s]);
        p.host = (double)S[s]->cold_bytes();
        p.h2d = h2d;
        // (all device weights: an upper bound of the per-batch read) + priming the staging ring at every batch start
        // (about two layers' cold groups over the host link before the first MoE; x FLASH_PP_PRIME_X, 1.65 from the
        // 2-card fit ~107 ms + 0.128 ms per row; small on 3+ cards, where few groups are cold)
        const double prime = env_double("FLASH_PP_PRIME_X", 1.65) * 2.0 * p.host / std::max(1, b[s + 1] - b[s]) / h2d * 1e3;
        p.fixed = flash_pp::pf_fixed_ms(S[s]->weight_gb() * 1e9) + prime;
        st.push_back(p);
    }
    int hi = 1 << 30;
    for (Worker* w : Wall) hi = std::min(hi, w->pf_batch(ctx0 + np + 64) - PFM);
    for (Worker* w : Wall)  // (peer-to-peer boundaries: within the receive buffers)
        if (w->plan().stage > 0 && w->pp_rx(0)) hi = std::min(hi, w->pp_rx_rows() - PFM);
    flash_pp::HopModel hm;
    hm.link = env_double("FLASH_TEST_LINK_GBPS", env_double("FLASH_PP_LINK_GBPS", 22.0)) * 1e9;
    bool all_p2p = true;  // one leg when every boundary goes peer-to-peer, else the host's two
    for (Worker* w : Wall) if (w->plan().stage > 0) all_p2p = all_p2p && w->pp_rx(0) != nullptr;
    hm.legs = all_p2p ? 1 : 2;
    hm.overlap = env_int("FLASH_PP_HOP_OV", 1) != 0;   // chunked copies on the copy stream (multi-chunk batches)
    if (Worker::pp_hop16()) hm.link *= 2;               // bf16 rows: half the bytes
    hm.stage_min = env_int("FLASH_PF_STAGE_MIN", 2048);
    // (the per-row cost model comes from 4096-row batches; below FLASH_PP_MB_MIN, default 2048, it is not trusted)
    const int mb = flash_pp::pick_microbatch(st, np, std::min(lo > 0 ? lo : env_int("FLASH_PP_MB_MIN", 2048), hi), hi, hm);
    if (wall_ms) *wall_ms = flash_pp::pf_wall_ms(st, np, mb, hm);
    static const bool lg = env_int("FLASH_PP_MB_LOG", 0) != 0;
    if (lg) {
        std::string o;
        for (auto& p : st) {
            char b[128];
            snprintf(b, sizeof b, " [u %.1f ms/1K, host %.2f GB, fixed %.0f ms]", p.u, p.host / 1e9, p.fixed);
            o += b;
        }
        for (int b = 2048; b <= hi; b += 1024) {
            char x[64];
            snprintf(x, sizeof x, " b%d %.0f ms", b, flash_pp::pf_wall_ms(st, np, b, hm));
            o += x;
        }
        fprintf(stderr, "[qflash] PP micro-batch model: %lld rows at %lld, hi %d -> %d;%s\n", np, ctx0, hi, mb, o.c_str());
    }
    return mb;
}

}  // namespace flash
