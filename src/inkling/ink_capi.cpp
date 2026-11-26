// [ink-srv-capi] C API of the Inkling 2 Mini engine (server/qe_capi.h): tier 0, chunked (tier 2), batch (tier 3), the
// single functions (vocab, slots, pool, can_admit, set_limit, warmup, reset), sampling (engine Gumbel-max for plain
// temperature and [ink-samp-dev] top_k 2..64 / min_p / top_p filters, drafts on; top_k > 64 and presence /
// frequency penalties on the host from row_logits, no drafts), the
// stops group (trim only at an exact saved point), faults and the slot pool. See ink_capi.h; the prompt cache and the
// host-RAM archive are in ink_capi_cache.cpp.
#include "ink_capi.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <numeric>
#include <stdexcept>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

using namespace ink;

namespace ink {

static int env_int(const char* k, int d) {
    const char* v = getenv(k);
    return v && *v ? atoi(v) : d;
}
static bool cache_log() {
    static const bool on = env_int("INK_CACHE_LOG", 0) != 0;
    return on;
}

void srv_bind(Handle& h, int i, bool limited) {
    Slot& s = h.s[i];
    const int held = s.held();
    int lim = std::max(held, s.cap - SLACK);
    if (limited && s.limit > 0) lim = std::max(held, (int)std::min<long>(lim, (long)held + s.limit));
    h.e->select(i);
    h.e->bind(i, s.base, held, lim);
}
// bind without select (the row steps address slots by table)
static void bind_only(Handle& h, int i, bool limited) {
    Slot& s = h.s[i];
    const int held = s.held();
    int lim = std::max(held, s.cap - SLACK);
    if (limited && s.limit > 0) lim = std::max(held, (int)std::min<long>(lim, (long)held + s.limit));
    h.e->bind(i, s.base, held, lim);
}

long srv_busy_rows(const Handle& h) {
    long n = 0;
    for (const Slot& s : h.s)
        if (!s.idle()) n += std::max(s.cap, s.held());
    return n;
}

static bool archive_cheap(const Handle& h, int j) {
    const Slot& s = h.s[j];
    if (s.archived || s.cap == 0 || s.toks.empty()) return true;
    // INK_ARCHIVE_SYNC_ROWS: the most rows an eviction / a start copies to host RAM on the spot (default: no limit; a
    // conversation evicted without its copy is prefilled again when it returns)
    static const long lim = [] {
        const char* v = getenv("INK_ARCHIVE_SYNC_ROWS");
        return v && *v ? atol(v) : (1L << 30);
    }();
    // ([ink-archfix] L2 / review2 L3: a partial archive counts its rows below the common prefix, current or stale: the
    // limit holds for the rows still to copy)
    return (long)s.toks.size() - std::max(s.prov_n, srv_arch_kept(h, j)) <= lim;
}

static void evict(Handle& h, int j) {
    h.e->ring_finalize();   // [ink-srv-gap] (a snapshot's host copy may still run: it fills a slot's snap vector)
    Slot& s = h.s[j];
    try {
        if (archive_cheap(h, j)) srv_archive_slot(h, j, 0, true);   // (its snapshot is dropped right after)
    } catch (const std::exception& ex) {
        fprintf(stderr, "inkling: archive of slot %d: %s\n", j, ex.what());
    }
    s.toks.clear();
    s.next = -1;
    s.cap = 0;
    s.snap.clear();
    s.snap_toks.clear();
    s.prov_cut(0);
    s.archived = false;
    h.evictions++;
}

// rows [src, src + n) -> [dst, dst + n), overlapping ranges in steps that never overwrite a row before it is read
static void move_rows(Handle& h, int src, int dst, int n) {
    if (n <= 0 || src == dst) return;
    const int d = std::abs(src - dst), step = std::min(n, d);
    if (dst < src)
        for (int o = 0; o < n; o += step) h.e->copy_rows(src + o, dst + o, std::min(step, n - o));
    else
        for (int o = n; o > 0; o -= step) {
            const int k = std::min(step, o);
            h.e->copy_rows(src + o - k, dst + o - k, k);
        }
    h.moves++;
}

bool srv_room(Handle& h, int i, int need) {
    Slot& me = h.s[i];
    const int P = h.pool, grow = std::max(64, env_int("INK_SLOT_GROW", 2048));
    if (need > P || need - SLACK > h.seq_max) return false;
    if (me.cap >= need) return true;
    bool compacted = false;
    for (;;) {
        std::vector<std::pair<int, int>> iv;   // [base, end) of the others
        for (size_t j = 0; j < h.s.size(); j++)
            if ((int)j != i && h.s[j].cap > 0) iv.push_back({h.s[j].base, h.s[j].base + h.s[j].cap});
        std::sort(iv.begin(), iv.end());
        std::vector<std::pair<int, int>> gaps;
        int at = 0;
        for (auto& x : iv) {
            if (x.first > at) gaps.push_back({at, x.first});
            at = std::max(at, x.second);
        }
        if (at < P) gaps.push_back({at, P});
        const int h0 = me.cap > 0 ? me.held() : 0;
        // 1. in place: the gap holding my base
        if (me.cap > 0)
            for (auto& g : gaps)
                if (g.first <= me.base && me.base < g.second && g.second - me.base >= need) {
                    me.cap = std::min(g.second - me.base, need + grow);
                    return true;
                }
        // 2. a place in a gap (the largest first): a new sequence in its middle, a held one where its rows need no
        // overlapping copy, else at the gap's start (copied in steps)
        std::sort(gaps.begin(), gaps.end(), [](auto& x, auto& y) { return x.second - x.first > y.second - y.first; });
        for (auto& g : gaps) {
            if (g.second - g.first < need) continue;
            int nb = -1;
            if (h0 == 0) {
                nb = iv.empty() ? 0 : g.first + (g.second - g.first - need) / 2;
            } else {
                for (int c : {g.first, me.base + h0, g.second - need})
                    if (nb < 0 && c >= g.first && c + need <= g.second && (c >= me.base + h0 || c + h0 <= me.base)) nb = c;
                if (nb < 0) nb = g.first;
                move_rows(h, me.base, nb, h0);
            }
            me.base = nb;
            me.cap = std::min(g.second - nb, need + grow);
            return true;
        }
        // 3. compaction (once): every slot's rows to the pool's start in order; idle intervals (and mine) cut to their
        // rows, busy ones keep the room of their next step / prompt
        if (!compacted) {
            compacted = true;
            std::vector<int> ord;
            for (size_t j = 0; j < h.s.size(); j++)
                if (h.s[j].cap > 0) ord.push_back((int)j);
            std::sort(ord.begin(), ord.end(), [&](int x, int y) { return h.s[x].base < h.s[y].base; });
            auto keep = [&](int j) {
                const Slot& o = h.s[j];
                const int r = o.held();
                if (j == i || o.idle()) return r;
                const int want = std::max(r + VR + SLACK, o.pf_done >= 0 ? (int)o.pf.size() + SLACK : 0);
                return std::max(r, std::min(o.cap, want));
            };
            long total = 0;
            for (int j : ord) total += keep(j);
            if (total <= P) {
                int to = 0;
                for (int j : ord) {
                    Slot& o = h.s[j];
                    const int k = keep(j), r = o.held();
                    if (k == 0) {
                        o.cap = 0;
                        continue;
                    }
                    move_rows(h, o.base, to, r);
                    o.base = to;
                    o.cap = k;
                    to += k;
                }
                h.compactions++;
            }
            continue;
        }
        // 4. evict the least recently used idle slot with rows, then try again
        int lru = -1;
        for (size_t j = 0; j < h.s.size(); j++)
            if ((int)j != i && h.s[j].cap > 0 && h.s[j].idle() && (lru < 0 || h.s[j].use < h.s[lru].use)) lru = (int)j;
        if (lru < 0) return false;
        evict(h, lru);
        compacted = false;
    }
}

// a fatal engine state (a sticky CUDA error): every later call would fail while /health stays ok. The server's exit
// hook runs (bounded by QE_EXIT_HOOK_S), then the process ends with code 75 so it can be restarted (INK_BROKEN_EXIT=0: keep)
static std::atomic<void (*)(const char*)> g_exit_hook{nullptr};

[[noreturn]] static void exit_broken(const char* what) {
    static char reason[640];
    snprintf(reason, sizeof reason, "Inkling engine error, the card is in an unknown state: %s", what ? what : "engine error");
    fprintf(stderr, "inkling: %s\n", reason);
    fflush(stderr);
    if (void (*f)(const char*) = g_exit_hook.exchange(nullptr)) {
        static std::atomic<int> done{0};
        try {
            std::thread([f] { f(reason); done.store(1); }).detach();
            const double lim = getenv("QE_EXIT_HOOK_S") ? atof(getenv("QE_EXIT_HOOK_S")) : 5.0;
            for (int k = 0; k < (int)(lim * 100) && !done.load(); k++) usleep(10000);
        } catch (...) {
        }
    }
    fprintf(stderr, "inkling: exiting (code 75) so the server can be restarted\n");
    fflush(stderr);
    fflush(stdout);
    _exit(75);
}
static void check_broken(const char* what) {
    const cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess && env_int("INK_BROKEN_EXIT", 1)) {
        std::string m = std::string(what ? what : "") + " (" + cudaGetErrorString(e) + ")";
        exit_broken(m.c_str());
    }
}

void srv_full(Handle& h, int i, int need) {
    h.faults++;
    h.full_ends++;
    if (i < 0 || i >= (int)h.s.size()) return;
    Slot& s = h.s[i];
    s.fault = need - SLACK > h.seq_max ? "KV pool full: this conversation reached its context of " + std::to_string(h.seq_max) + " tokens"
                                       : "KV pool full: no room for this sequence to grow next to the running ones";
    s.gen = false;
    s.next = -1;
    fprintf(stderr, "inkling: slot %d: %s (%d rows held); the reply ends with finish_reason length\n", i, s.fault.c_str(), s.held());
}

void srv_caught(void* p, int slot, const char* where) {
    std::string m;
    try {
        throw;
    } catch (const std::exception& e) {
        m = e.what();
    } catch (...) {
        m = "unknown exception";
    }
    fprintf(stderr, "inkling: %s: %s\n", where, m.c_str());
    if (!p) return;
    try {
        srv_fault(*(Handle*)p, slot, m.c_str());
    } catch (...) {
    }
}

void srv_fault(Handle& h, int i, const char* what) {
    h.faults++;
    try {
        h.e->ring_finalize();
    } catch (...) {
    }
    check_broken(what);
    if (i < 0 || i >= (int)h.s.size()) return;
    Slot& s = h.s[i];
    s.fault = what ? what : "engine error";
    s.gen = false;
    if (s.pf_done >= 0 && h.newest == i) h.newest = -1;
    s.pf_done = -1;
    s.pf.clear();
    s.toks.clear();
    s.next = -1;
    s.snap.clear();
    s.snap_toks.clear();
    s.prov_cut(0);
    s.archived = false;
    s.paused = false;
    s.pk = Park{};
}

// x: [V] capped logits of the row; temperature, penalties, top-k, min-p, top-p, then a draw (host RNG of the reply)
int srv_pick_host(Handle& h, Slot& s, float* x) {
    (void)h;
    Samp& sp = s.sp;
    const int n = V;
    if (sp.pen())
        for (const auto& kv : sp.cnt)
            if (kv.first >= 0 && kv.first < n) x[kv.first] -= sp.pen_val(kv.first);
    int best = -1;
    for (int i = 0; i < n; i++)
        if (std::isfinite(x[i]) && (best < 0 || x[i] > x[best])) best = i;
    if (best < 0) return -1;
    if (sp.one() || !(sp.temp > 0.f)) return best;
    // candidates: top_k (0 = all) by logit (ties: the lower id)
    auto higher = [&](int a, int b) { return x[a] > x[b] || (x[a] == x[b] && a < b); };
    std::vector<int> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    const int k = sp.top_k > 0 && sp.top_k < n ? sp.top_k : n;
    if (k < n) {
        std::nth_element(idx.begin(), idx.begin() + k, idx.end(), higher);
        idx.resize(k);
    }
    const float mx = x[best];
    double sum = 0;   // the kept mass (p = 1 at the most likely token)
    {
        std::vector<int> keep;
        keep.reserve(idx.size());
        for (int i : idx) {
            const double q = std::isfinite(x[i]) ? std::exp((double)(x[i] - mx) / sp.temp) : 0.0;
            if (q > 0 && !(sp.min_p > 0.f && q < sp.min_p)) {   // min_p: relative to the most likely token
                keep.push_back(i);
                sum += q;
            }
        }
        idx.swap(keep);
    }
    if (idx.empty() || !(sum > 0)) return best;
    auto prob = [&](int i) { return std::exp((double)(x[i] - mx) / sp.temp); };
    size_t m = idx.size();   // idx[0, m) = the kept set
    if (sp.top_p < 1.f) {
        // the smallest set of the most likely tokens whose mass reaches top_p: the largest t candidates in order
        // (t = 64, 256, ..: nth_element + sort, exact; a draw needs only them)
        const double want = sp.top_p * sum;
        for (size_t t = 64;; t *= 4) {
            const size_t tt = std::min(t, idx.size());
            if (tt < idx.size()) std::nth_element(idx.begin(), idx.begin() + tt, idx.end(), higher);
            std::sort(idx.begin(), idx.begin() + tt, higher);
            double acc = 0;
            size_t j = 0;
            for (; j < tt && acc < want; j++) acc += prob(idx[j]);
            if (acc >= want || tt == idx.size()) {
                m = j;
                sum = acc;
                break;
            }
        }
    }
    std::uniform_real_distribution<double> u(0.0, sum);
    double r = u(sp.rng);
    for (size_t j = 0; j < m; j++) {
        r -= prob(idx[j]);
        if (r <= 0) return idx[j];
    }
    return idx[m - 1];
}

}  // namespace ink

namespace {

// [ink-srv-gap] INK_PROF=1: one stderr line per prefill call and per qe_step_batch (host ms of each phase)
bool prof_on() {
    static const bool on = env_int("INK_PROF", 0) != 0;
    return on;
}
double now_ms() { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

void pen_count(Slot& s, int t) {
    if (s.sp.pen() && t >= 0) s.sp.cnt[t]++;
}

// rows of ids[0, n) slot s can keep itself: its held sequence (the prompt extends it) or its prompt-end snapshot;
// live: the reuse is the held sequence (the slot's state is at its end)
int own_reuse(const Slot& s, const int* ids, int n, bool* live) {
    *live = false;
    if (s.cap == 0) return 0;
    const int held = s.held();
    if (held >= 1 && held <= n - 1 && std::equal(s.toks.begin(), s.toks.end(), ids)) {
        *live = true;
        return held;
    }
    const int P = (int)s.snap_toks.size();
    if (P >= 1 && P <= n - 1 && !s.snap.empty() && held >= P && std::equal(s.snap_toks.begin(), s.snap_toks.end(), ids) &&
        std::equal(s.toks.begin(), s.toks.begin() + P, ids))
        return P;
    return 0;
}

int pick_slot(Handle& h, const int* ids, int n, int* reuse, bool* live) {
    int best = -1, br = -1;
    bool bl = false;
    for (size_t i = 0; i < h.s.size(); i++) {
        const Slot& s = h.s[i];
        if (!s.idle()) continue;
        bool lv = false;
        const int r = own_reuse(s, ids, n, &lv);
        // more rows reused wins; at 0: a slot without rows, else the least recently used
        const bool better = best < 0 || r > br ||
                            (r == br && r == 0 && ((s.cap == 0) > (h.s[best].cap == 0) ||
                                                   ((s.cap == 0) == (h.s[best].cap == 0) && s.use < h.s[best].use)));
        if (better) {
            best = (int)i;
            br = r;
            bl = lv;
        }
    }
    *reuse = br;
    *live = bl;
    return best;
}

int snap_min() {
    static const int v = std::max(1, env_int("INK_SNAP_MIN", 16));
    return v;
}

// one prefill call of slot i: up to `budget` rows of its prompt (all: budget <= 0); true when the prompt is complete
bool prefill_some(Handle& h, int i, int budget) {
    Slot& s = h.s[i];
    const int n = (int)s.pf.size();
    if (s.pf_done >= n) return true;
    int k = n - s.pf_done;
    if (budget > 0 && budget < k) k = budget;
    // [ink-srv-gap] a remainder below 1,024 rows goes with this call (a 3-row last call of a 122,883-row prompt cost 100 ms)
    if (n - s.pf_done - k > 0 && n - s.pf_done - k < 1024) k = n - s.pf_done;
    if (s.pf_hint > s.pf_done) k = std::min(k, s.pf_hint - s.pf_done);   // stop once at the system part's end
    const double t0 = prof_on() ? now_ms() : 0;
    srv_bind(h, i);
    const bool last = s.pf_done + k == n;
    s.pf_started = true;
    h.e->prefill(s.pf.data() + s.pf_done, k);
    const double t1 = prof_on() ? now_ms() : 0;
    double t2 = t1, t3 = t1, t4 = t1;
    s.toks.insert(s.toks.end(), s.pf.begin() + s.pf_done, s.pf.begin() + s.pf_done + k);
    s.pf_done += k;
    if (s.pf_hint > 0 && s.pf_done == s.pf_hint) srv_hint_save(h, i);
    if (prof_on()) t2 = now_ms();
    if (last) {
        int t;
        if (s.sp.host()) {
            if ((int)h.logits.size() != V) h.logits.assign(V, 0.f);
            h.e->row_logits(0, h.logits.data());
            t = srv_pick_host(h, s, h.logits.data());
        } else {
            t = h.e->sample_last();   // the slot's sampling (greedy at T 0)
        }
        if (t < 0 || t >= V) throw std::runtime_error("the logits are not finite (prompt end)");
        s.pf_first = t;
        if (prof_on()) t3 = now_ms();
        // the prompt-end snapshot: the next turn of the conversation starts here when its history is re-rendered
        // (the reply's reasoning is usually not sent back, so the held sequence diverges right after the prompt)
        if (n >= snap_min() && env_int("INK_SNAP", 1)) {
            h.e->ring_save_async(s.snap);   // [ink-srv-gap] the window rows copy beside the first step (out of the TTFT)
            s.snap_toks = s.pf;
        } else {
            s.snap.clear();
            s.snap_toks.clear();
        }
        if (prof_on()) t4 = now_ms();
    }
    if (prof_on())
        fprintf(stderr, "[ink prof] prefill slot %d rows %d..%d of %d: bind %.2f + engine %.2f ms (%.0f tok/s)%s, hint %.2f, first token %.2f, snapshot %.2f ms (%.1f MB)\n",
                i, s.pf_done - k, s.pf_done, n, 0.0, t1 - t0, k / std::max(1e-6, (t1 - t0) / 1e3), last ? " last" : "", t2 - t1, t3 - t2, t4 - t3,
                s.snap.size() / 1e6);
    return last;
}

int num_prefills(const Handle& h) {
    int c = 0;
    for (const Slot& s : h.s) c += s.pf_done >= 0;
    return c;
}

// the tokens of one step of slot k (o[0, c)): fed = next + o[0, c - 1), next = o[c - 1]; false: a bad token (faulted)
bool take(Handle& h, int k, const int* o, int c) {
    Slot& s = h.s[k];
    for (int j = 0; j < c; j++)
        if (o[j] < 0 || o[j] >= V) {
            fprintf(stderr, "inkling: slot %d: the logits are not finite; this request fails, the others go on\n", k);
            srv_fault(h, k, "the logits are not finite (this sequence's step)");
            return false;
        }
    s.toks.push_back(s.next);
    for (int j = 0; j + 1 < c; j++) s.toks.push_back(o[j]);
    s.next = o[c - 1];
    for (int j = 0; j < c; j++) pen_count(s, o[j]);
    if (s.limit > 0) s.limit -= std::min<long>(s.limit, c);
    s.use = s.emit = ++h.tick;
    s.archived = false;
    s.st[0]++;
    s.st[2] += c - 1;
    return true;
}

// [ink-srv-review-fix] M2: host bytes a park_save of slot i adds (rows + its state at held; the snapshot is moved or
// stays in the slot, never copied): the state is ~176 MB at >= 1,024 rows (sliding windows), less below
size_t park_need(const Handle& h, int i) {
    const Slot& s = h.s[i];
    const size_t held = (size_t)s.held();
    const size_t ring = (size_t)(176e6 * (double)std::min<size_t>(held, 1024) / 1024.0) + (size_t)8e6;
    return h.row_bytes * held + ring + 8 * held;
}
// slot i's decoding state -> pk (host): its pool rows [base, base + held), ring_save at held, the reply. move_snap: the
// prompt-end snapshot moves into pk (a paused slot has no rows, so it is useless there); else it stays in the slot
// (a parked reply's slot keeps its rows as an idle conversation, the snapshot serves that conversation's next turn)
void park_save(Handle& h, int i, Park& pk, bool move_snap) {
    h.e->ring_finalize();
    Slot& s = h.s[i];
    pk = Park{};
    srv_bind(h, i);
    h.e->ring_save(pk.ring);
    pk.rows = std::make_shared<std::vector<unsigned char>>();
    h.e->rows_save(s.base, s.held(), *pk.rows);
    pk.toks = s.toks;
    pk.next = s.next;
    pk.pen_from = s.pen_from;
    pk.sp = s.sp;
    pk.limit = s.limit;
    pk.watch = s.watch;
    if (move_snap) {
        pk.snap_toks = std::move(s.snap_toks);
        pk.snap = std::move(s.snap);
        s.snap_toks.clear();
        s.snap.clear();
    }
    pk.orig = i;
    pk.admit = s.admit;
    std::copy(s.st, s.st + 5, pk.st);
}
// pk -> slot i (its interval already holds room for held + VR + SLACK rows): rows, state, the reply; generating
void park_load(Handle& h, int i, Park& pk) {
    h.e->ring_finalize();
    Slot& s = h.s[i];
    const int n = (int)pk.toks.size();
    h.e->rows_load(*pk.rows, s.base, n);
    s.toks = pk.toks;
    s.prov_cut(0);
    srv_bind(h, i);
    h.e->ring_load(pk.ring, n);
    s.next = pk.next;
    s.pen_from = pk.pen_from;
    s.sp = pk.sp;
    eng_samp(*h.e, i, s.sp);
    s.limit = pk.limit;
    s.watch = pk.watch;
    s.snap_toks = std::move(pk.snap_toks);   // (pk is dropped after the load)
    s.snap = std::move(pk.snap);
    std::copy(pk.st, pk.st + 5, s.st);
    s.admit = pk.admit;
    s.gen = true;
    s.archived = false;
    s.fault.clear();
    s.use = s.emit = ++h.tick;
}
// a parked / paused conversation that ends: into the store (its rows and state are host copies already)
void park_store(Handle& h, Park& pk) {
    if (pk.toks.empty() || !pk.rows) return;
    Entry e;   // (everything moved out of pk: the store fit must not count it twice, srv_host_extra)
    const int n = (int)pk.toks.size();
    e.toks = std::move(pk.toks);
    pk.toks.clear();
    e.rows.push_back(RowPart{0, n, std::move(pk.rows)});
    pk.rows.reset();
    e.ring = std::move(pk.ring);
    pk.ring.clear();
    if ((int)pk.snap_toks.size() <= n && std::equal(pk.snap_toks.begin(), pk.snap_toks.end(), e.toks.begin())) {
        e.snap_toks = std::move(pk.snap_toks);
        e.snap = std::move(pk.snap);
    }
    pk.snap.clear();
    pk.snap_toks.clear();
    srv_store_put(h, std::move(e));
}
// a paused reply that ends: its conversation goes to the store as it is, the slot holds nothing
void end_paused(Handle& h, Slot& s) {
    try {
        park_store(h, s.pk);
    } catch (const std::exception& ex) {
        fprintf(stderr, "inkling: store of a paused slot: %s\n", ex.what());
    }
    s.pk = Park{};
    s.paused = false;
    s.toks.clear();
    s.next = -1;
    s.cap = 0;
    s.snap.clear();
    s.snap_toks.clear();
    s.prov_cut(0);
}
// [ink-resume] one decoding slot paused (its rows + state to host memory, region freed); false: not pausable now
// (host memory for the copy is short, or the copy failed: then the slot faults)
bool pause_one(Handle& h, int j, const char* why) {
    Slot& s = h.s[j];
    if (!s.gen || s.paused || s.pf_done >= 0 || s.next < 0 || s.cap == 0 || s.toks.empty()) return false;
    if (!srv_store_fit(h, park_need(h, j))) return false;
    try {
        park_save(h, j, s.pk, true);
    } catch (const std::exception& ex) {
        fprintf(stderr, "inkling: pause of slot %d: %s\n", j, ex.what());
        srv_fault(h, j, ex.what());
        return false;
    }
    fprintf(stderr, "inkling: pool: slot %d paused at %d tokens %s (region [%d, %d) freed)\n", j, s.held(), why, s.base,
            s.base + s.cap);
    s.paused = true;
    s.pause_ord = ++h.tick;
    s.cap = 0;
    s.prov_cut(0);
    s.archived = false;
    h.pauses++;
    return true;
}
// [ink-resume] INK_RESUME_ROOM: reply (default: min(INK_ADMIT_REPLY_ROWS, pool / 16) rows per sequence, capped at its
// context left; the pause on growth keeps a growing sequence going), growth (its whole remaining growth: a paused reply
// waits while an unlimited one runs), bounded (min(remaining growth, pool / 8)); another value: reply, with a line
int resume_mode() {
    static const int mode = [] {
        const char* v = getenv("INK_RESUME_ROOM");
        const std::string m = v ? v : "";
        if (!m.empty() && m != "reply" && m != "growth" && m != "bounded")
            fprintf(stderr, "inkling: WARNING: INK_RESUME_ROOM=%s is not reply / growth / bounded: reply is used\n", m.c_str());
        return m == "growth" ? 0 : m == "bounded" ? 2 : 1;
    }();
    return mode;
}
long resume_room(const Handle& h) {
    static const long reply = std::max(0, env_int("INK_ADMIT_REPLY_ROWS", 2048));
    return resume_mode() == 1 ? std::min<long>(reply, (long)h.pool / 16) : 0;
}
// rows a sequence keeps through a compaction (srv_room step 3; resume_review2 L-d)
long keep_rows(const Slot& o) {
    const int r = o.held(), want = std::max(r + VR + SLACK, o.pf_done >= 0 ? (int)o.pf.size() + SLACK : 0);
    return std::max(r, std::min(o.cap, want));
}
// [ink-resume] resume_review2 L-b: the rows a running sequence (not idle, not paused) and a paused reply count for, one
// formula for the resume (resume_paused) and the admission (qe_can_admit) while a reply is paused
long run_rows(const Handle& h, const Slot& o) {
    const int mode = resume_mode(), held = std::max(o.held(), (int)o.pf.size());
    return std::max<long>(o.cap, (long)held + (mode == 1 ? 0 : VR) + SLACK +
                                     pool_reserve(mode, resume_room(h), h.pool, h.seq_max, held, o.limit));
}
long paused_rows(const Handle& h, const Slot& s) {
    const int held = (int)s.pk.toks.size();
    return (long)held + VR + SLACK + pool_reserve(resume_mode(), resume_room(h), h.pool, h.seq_max, held, s.pk.limit);
}
// [ink-resume] room for `need` rows on decoding slot k, pausing decoding replies when the pool has none, as the 27B
// engine's room_or_pause: the most recently admitted decoding reply first (src/pool_order.h, no priority), k itself
// included (then k waits, paused, while another sequence holds rows); never a prompt being prefilled (admitted with
// its room) nor a reply at its max_tokens. false: no room (the reply then ends with length), or k paused itself.
// INK_PREEMPT=0: off (the reply ends with length at once, as before)
bool room_or_pause(Handle& h, int k, int need) {
    static const bool on = env_int("INK_PREEMPT", 1) != 0;
    if (srv_room(h, k, need)) return true;
    // (resume_review M1: at its context or past the pool nothing makes room: it ends before anyone is paused)
    if (!on || !pool_fits_ever(need, h.pool, h.seq_max)) return false;
    std::vector<int> tried;   // (review L1: a reply whose pause failed is passed over, the next one tried)
    for (;;) {
        std::vector<int> cand{k};
        long fixed = 0;
        bool others = false;   // another sequence holds rows that will free (a prefill or a reply)
        for (size_t j = 0; j < h.s.size(); j++) {
            const Slot& o = h.s[j];
            if ((int)j == k || o.idle() || o.paused || o.cap == 0) continue;
            others = true;
            // (resume_review2 L-a: only a reply that can come back: one row past its tokens fits its context)
            const bool pausable = o.gen && o.pf_done < 0 && o.next >= 0 && o.limit != 0 &&
                                  pool_fits_ever((long)o.held() + 1 + SLACK, h.pool, h.seq_max) &&
                                  std::find(tried.begin(), tried.end(), (int)j) == tried.end();
            if (pausable) cand.push_back((int)j);
            else fixed += keep_rows(o);
        }
        const int v = pause_pick(cand, k, [&](int j) { return h.s[j].admit; }, [&](int j) { return keep_rows(h.s[j]); },
                                 fixed, need, h.pool);
        if (v == k) {
            // k waits itself (paused, resumed when rows free) while another sequence holds rows; alone it ends (length)
            if (!others) return false;
            pause_one(h, k, "until rows free (no room to grow)");
            return false;
        }
        char why[64];
        snprintf(why, sizeof why, "for slot %d to grow", k);
        if (!pause_one(h, v, why)) {
            tried.push_back(v);
            continue;
        }
        if (srv_room(h, k, need)) return true;
    }
}
// paused slots among slots[0, n) back, first paused first (head of line: one that has no room stops the later ones)
void resume_paused(Handle& h, const int* slots, int n) {
    std::vector<int> pz;
    for (int i = 0; i < n; i++)
        if (slots[i] >= 0 && slots[i] < (int)h.s.size() && h.s[slots[i]].paused &&
            std::find(pz.begin(), pz.end(), slots[i]) == pz.end()) pz.push_back(slots[i]);
    std::sort(pz.begin(), pz.end(), [&](int a, int b) { return h.s[a].pause_ord < h.s[b].pause_ord; });
    // [ink-resume] a paused reply resumes when the rows left keep every running sequence's reserve (resume_mode,
    // pool_reserve: capped at its context left) and its own; alone it resumes with one row past its tokens
    // (resume_want, resume_review H1). One that does not fit waits (head of line) until rows free; it is never dropped.
    // (with INK_PREEMPT, the default, a sequence that cannot grow pauses a reply instead of ending)
    const int mode = resume_mode();
    const bool old_rule = mode == 1;
    const long room = resume_room(h);
    auto growth = [&](int held, long limit) { return pool_reserve(mode, room, h.pool, h.seq_max, held, limit); };
    long busy = 0;
    for (const Slot& o : h.s)
        if (!o.idle() && !o.paused) busy += run_rows(h, o);
    for (int k : pz) {
        Slot& s = h.s[k];
        const int held = (int)s.pk.toks.size(), need = held + VR + SLACK;
        const long want = resume_want(busy, held, growth(held, s.pk.limit));
        if (busy + want > h.pool) return;
        // (resume_review2 L-c: a reply resumed alone counts with its reserve for the next one, as a running sequence)
        busy += std::max(want, (long)held + (old_rule ? 0 : VR) + SLACK + growth(held, s.pk.limit));
        try {
            s.toks.clear();   // (no rows of its own in the pool while paused)
            // (H1: one row past its tokens when its step's rows do not fit, as qe_unpark)
            if (!srv_room(h, k, need) && !srv_room(h, k, held + 1 + SLACK)) {
                s.toks = s.pk.toks;
                return;
            }
            Park pk = std::move(s.pk);
            s.pk = Park{};
            s.paused = false;
            park_load(h, k, pk);
            h.resumes++;
            fprintf(stderr, "inkling: pool: slot %d resumed at %d tokens\n", k, s.held());
        } catch (const std::exception& ex) {
            fprintf(stderr, "inkling: resume of slot %d: %s\n", k, ex.what());
            srv_fault(h, k, ex.what());
        }
    }
}

}  // namespace

extern "C" {

__attribute__((visibility("default"))) void qe_set_exit_hook(void (*f)(const char*)) { g_exit_hook.store(f); }

// NULL when the engine cannot start (missing / bad files, CUDA error); the engine itself fits its memory (fewer
// slots, a smaller pool, the drafter off) and never fails for lack of it
// [ink-pf8-1c2] default slot count (user 11-25): 8 on 2+ cards (TP), 2 on one card (one card's fixed per-slot state
// ~0.36 GB would otherwise take most of its K/V pool: 8 slots 53,216 tokens per conversation, 2 slots 116,704). devices:
// the server's list (empty: the engine's QE_DEVICES, else one card); distinct cards count (review2: virtual TP 0,0 = one).
extern "C" int qe_ink_default_slots(const char* devices) {
    const char* d = devices && *devices ? devices : getenv("QE_DEVICES");
    std::vector<long> ids;
    for (const char* c = d ? d : ""; *c;) {
        if (*c < '0' || *c > '9') {
            c++;
            continue;
        }
        long v = 0;
        while (*c >= '0' && *c <= '9') v = std::min(10 * v + (*c++ - '0'), 1L << 20);
        if (std::find(ids.begin(), ids.end(), v) == ids.end()) ids.push_back(v);
    }
    return ids.size() > 1 ? 8 : 2;
}
// the slot count qe_create asks the engine for: INK_SLOTS (non-empty; atoi, clamped to 1..16) wins over the default
extern "C" int qe_ink_slots(const char* devices, const char* ink_slots) {
    return ink_slots && *ink_slots ? std::max(1, std::min(16, atoi(ink_slots))) : qe_ink_default_slots(devices);
}
// review2 L1: the start-up line when an explicit INK_SLOTS is not what runs (clamped to 1..16, or halved by the memory
// fit); the raw value as given. Writes it into buf (n bytes) and returns its length, 0 = no line.
extern "C" int qe_ink_slot_note(const char* ink_slots, int served, char* buf, int n) {
    if (!ink_slots || !*ink_slots || n <= 0) return 0;
    const int rq = atoi(ink_slots);
    if (served >= rq && rq >= 1 && rq <= 16) return 0;
    const char* why = rq > 16 ? "at most 16" : rq < 1 ? "at least 1" : "memory: each slot's fixed state ~0.36 GB";
    const int k = snprintf(buf, n, "inkling: requested %s slots, serving %d (%s)", ink_slots, served, why);
    return k < 0 ? 0 : std::min(k, n - 1);
}
void* qe_create(const char* model_dir, const char* devices, int max_t, const char* split) {
    (void)split;
    Handle* h = nullptr;
    try {
        Opts o;
        if (model_dir && *model_dir) o.model_dir = model_dir;
        o.drafter = env_int("INK_DRAFTER", 1) != 0;
        // the context of one conversation: what the caller asks for (INK_MAX_CTX caps it; the model's own limit 1M)
        const int cap = std::min(1 << 20, env_int("INK_MAX_CTX", 1 << 20));
        const int seq = std::max(1024, max_t > 0 && max_t < cap ? max_t : cap);
        o.max_ctx = seq;
        // slots (INK_SLOTS, 1..16; default qe_ink_default_slots: 8 on 2+ cards, 2 on one card; halved by the engine while the pool would be below INK_POOL_MIN rows,
        // default 32768); the pool: every row that fits with the drafter (INK_POOL=N pins N rows)
        o.slots = qe_ink_slots(devices, getenv("INK_SLOTS"));   // [ink-pf8-1c2]
        o.pool = std::max(0, env_int("INK_POOL", 0));
        o.pool_auto = getenv("INK_SLOTS") && *getenv("INK_SLOTS") ? 1024 : std::max(1024, env_int("INK_POOL_MIN", 32768));
        // [ink-tp-srv] several cards: tensor parallelism (every engine call dispatched to the ranks). [ink-ctx-dev] the
        // cards are the server's `devices` (indices within CUDA_VISIBLE_DEVICES, as every other engine takes them;
        // server/run.sh maps QE_DEVICES to CUDA_VISIBLE_DEVICES + --devices); the engine's QE_DEVICES env (ink_run) only
        // when the caller passes none
        if (devices && *devices) {   // review L4: digits and commas only ("1,x", "0,,1", "-1": refused)
            o.devices.clear();
            for (const char* c = devices;;) {
                if (*c < '0' || *c > '9') throw std::runtime_error(std::string("devices '") + devices + "': a comma list of GPU indices (0,1)");
                long v = 0;
                while (*c >= '0' && *c <= '9') v = std::min(10 * v + (*c++ - '0'), 1L << 20);
                o.devices.push_back((int)v);
                if (!*c) break;
                if (*c++ != ',') throw std::runtime_error(std::string("devices '") + devices + "': a comma list of GPU indices (0,1)");
            }
        }
        h = new Handle;
        h->e = new Engine(o);
        h->pool = h->e->pool();
        h->e->prepare_host_copies((size_t)512 << 20);   // [ink-srv-gap] (the first prompt-end snapshot paid a 130 ms pin)
        h->seq_max = std::min(seq, std::min(h->e->max_ctx(), h->pool - SLACK));
        h->s.resize(h->e->slots());
        {   // [ink-pf8-1c2] an explicit count that does not run as asked (clamped or halved by the fit)
            char nb[200];
            if (qe_ink_slot_note(getenv("INK_SLOTS"), h->e->slots(), nb, sizeof nb)) fprintf(stderr, "%s\n", nb);
        }
        for (int i = 0; i < h->e->slots(); i++) h->s[i].base = 0;
        {
            std::vector<unsigned char> b;
            h->e->rows_save(0, 1, b);
            h->row_bytes = b.size();
        }
        if (h->seq_max < seq)
            fprintf(stderr, "inkling: context per conversation %d tokens (asked %d): what the pool of %d rows holds\n", h->seq_max, seq, h->pool);
        fprintf(stderr, "inkling: K/V pool %d rows for %d slots (%s), context per conversation %d tokens, drafter %s, %.1f KB per row\n",
                h->pool, (int)h->s.size(), o.pool ? "INK_POOL" : "every row that fits", h->seq_max, h->e->drafter_on() ? "on" : "off",
                h->row_bytes / 1024.0);
        {   // [ink-pf8] the dense attention weights' form, the served context and how to switch (any card count)
            const int df = h->e->dense_form(), nc = h->e->cards();
            const char* form = df == 8 ? "int8 (GPTQ INT8 g128 from dense8.safetensors: faster, small quality loss)"
                               : df == 9 ? "int8 (RTN INT8 at load: dense8.safetensors missing or INK_DENSE8_RTN=1; lower quality than GPTQ, set-2 KL 0.00589)"
                                         : "g15 (original weights, lossless)";
            // [ink-1c8] int8 is the faster form on any card count now (one card 7.05 vs 7.97 ms/step)
            const char* how = df != 15 ? "--dense-weights g15 (INK_DENSE8=0) = original weights, lossless"
                                       : "--dense-weights int8 (INK_DENSE8=1) = GPTQ INT8, faster, small quality loss";
            fprintf(stderr, "inkling: dense attention weights %s on %d card%s, context per conversation %d tokens; switch: %s\n", form,
                    nc, nc > 1 ? "s" : "", h->seq_max, how);
        }
        return h;
    } catch (const std::exception& e) {
        size_t fr = 0, tot = 0;
        cudaMemGetInfo(&fr, &tot);
        if (strstr(e.what(), "peer access") || strstr(e.what(), "devices '"))   // review H1: the cause, not a memory hint
            fprintf(stderr, "%s\ninkling: qe_create failed: %s\n", e.what(),
                    strstr(e.what(), "peer access") ? "tensor parallelism needs GPU peer access (P2P) between the cards; "
                                                      "without it start on one card (--devices 0)" : "bad device list");
        else
            fprintf(stderr, "%s\ninkling: qe_create failed (%.2f GB of %.2f GB free on the card now; the weights + drafter need ~20.2 GB, "
                            "the smallest context ~0.6 GB more)\n", e.what(), fr / 1e9, tot / 1e9);
        if (h) delete h->e;
        delete h;
        return nullptr;
    }
}
void qe_destroy(void* p) try {
    Handle* h = (Handle*)p;
    if (!h) return;
    delete h->e;
    delete h;
} catch (...) {
    ::ink::srv_caught(nullptr, -1, __func__);
}
int qe_max_context(void* p) { return ((Handle*)p)->seq_max; }
long long qe_pool_tokens(void* p) { return ((Handle*)p)->pool; }
int qe_num_slots(void* p) { return (int)((Handle*)p)->s.size(); }
int qe_vocab(void* p) { (void)p; return V; }

// temperature <= 0: greedy; plain temperature: the engine's exact Gumbel-max (drafts on); [ink-samp-dev] top_k 2..64
// (with top_p / min_p), or no top_k with min_p and / or top_p: the engine's Gumbel-max over the filtered set (drafts
// on; without top_k the set is certified inside the 64 largest logits: min_p's set must end there, top_p alone by an
// upper bound on the remaining vocabulary mass; else that row is picked on the host with the same rule and noise);
// top_k > 64 (and penalties): picked on the host from the row's full logits, no drafts. Applies to the next start and
// to the newest prompt in flight while none of its chunks ran.
void qe_set_sampling(void* p, float temperature, float top_p, int top_k, float min_p, unsigned long long seed) try {
    Handle* h = (Handle*)p;
    Samp& sp = h->pending;
    sp.temp = temperature > 0 && std::isfinite(temperature) ? temperature : 0.f;
    sp.top_p = top_p > 0 && top_p < 1 ? top_p : 1.f;
    sp.top_k = top_k > 0 ? top_k : 0;
    sp.min_p = min_p > 0 && min_p < 1 ? min_p : 0.f;
    sp.seed = seed;
    sp.rng.seed(seed);
    static const bool devf_on = env_int("INK_SAMP_DEV", 1) != 0;   // [ink-samp-dev] INK_SAMP_DEV=0: filters on the host
    sp.devf = devf_on && h->e->filter_ready() && filter_on_device(sp.top_k, sp.top_p, sp.min_p);
    sp.pres = sp.freq = 0.f;
    sp.cnt.clear();
    if (h->newest >= 0 && h->s[h->newest].pf_done >= 0 && !h->s[h->newest].pf_started) {
        h->s[h->newest].sp = sp;
        eng_samp(*h->e, h->newest, sp);
    }
} QE_CATCH(-1)
// presence / frequency penalty (right after qe_set_sampling): logit -= frequency * c + presence * (c > 0); sampled
// replies only; such a reply picks its tokens on the host (no drafts)
void qe_set_penalty(void* p, float presence, float frequency, const int* seed, int n) try {
    Handle* h = (Handle*)p;
    Samp& sp = h->pending;
    sp.pres = std::isfinite(presence) ? std::min(100.f, std::max(-100.f, presence)) : 0.f;
    sp.freq = std::isfinite(frequency) ? std::min(100.f, std::max(-100.f, frequency)) : 0.f;
    sp.cnt.clear();
    for (int i = 0; seed && i < n; i++)
        if (seed[i] >= 0 && seed[i] < V) sp.cnt.emplace(seed[i], 1);
    if (h->newest >= 0 && h->s[h->newest].pf_done >= 0 && !h->s[h->newest].pf_started) {
        Slot& s = h->s[h->newest];
        s.sp.pres = sp.pres, s.sp.freq = sp.freq, s.sp.cnt = sp.cnt;
        eng_samp(*h->e, h->newest, s.sp);
    }
} QE_CATCH(-1)

// a free slot and room for n rows (+ reply room when others run) next to the running sequences
int qe_can_admit(void* p, int n) {
    Handle* h = (Handle*)p;
    bool free_slot = false;
    for (const Slot& s : h->s) free_slot = free_slot || s.idle();
    static const int reply = std::max(0, env_int("INK_ADMIT_REPLY_ROWS", 2048));
    long busy = srv_busy_rows(*h);
    // [ink-resume] resume_review M3 / review2 L-b: while a reply is paused, every sequence counts as resume_paused counts
    // it (a paused one its resume rows), so new requests cannot keep taking the rows it waits for
    bool any_paused = false;
    for (const Slot& s : h->s) any_paused = any_paused || s.paused;
    if (any_paused) {
        busy = 0;
        for (const Slot& s : h->s)
            if (!s.idle()) busy += s.paused ? paused_rows(*h, s) : run_rows(*h, s);
    }
    const long room = std::min<long>({(long)reply, (long)h->pool / 16, (long)std::max(0, h->seq_max - n - 1)});
    const long want = (long)n + SLACK + (busy > 0 ? room : 0);
    return free_slot && n < h->seq_max && busy + want <= h->pool;
}

// chunked group. Codes: -1 no free slot / no room (retried), -2 bad input or an engine error, -3 too long, -4 two
// prompts in flight already
int qe_start_begin(void* p, const int* ids, int n, int k, unsigned long long affinity, int hint, const void* img,
                   int img_rows, const int* rope3, int delta) try {
    (void)k; (void)affinity; (void)rope3; (void)delta;
    Handle* h = (Handle*)p;
    if (!ids || n < 1 || img || img_rows) return -2;
    h->e->ring_finalize();   // [ink-srv-gap] every snapshot vector complete before the slot choice reads them
    if (n >= h->seq_max) return -3;
    for (int i = 0; i < n; i++)
        if (ids[i] < 0 || ids[i] >= V) return -2;
    // [ink-archprot] the entry this prompt restores from stays while its own archive / evictions and (archprot_review
    // M1) the ended paused replies' park copies make room
    struct Prot {
        Handle& h;
        ~Prot() { srv_protect_end(h); }
    } prot{*h};
    h->protect_next.clear();   // (archprot_review2 nit: qe_protect_ids is one-shot)
    srv_protect_begin(*h, ids, n);
    if (!h->concurrent)   // one reply at a time: a new start ends the others
        for (Slot& o : h->s) {
            // [ink-srv-review-fix] L1: a paused reply ends as a release ends it, never an idle slot that still carries
            // `paused` and its old reply
            if (o.paused) end_paused(*h, o);
            o.gen = false;
        }
    if (num_prefills(*h) >= 2) return -4;
    int reuse = 0;
    bool live = false;
    const int i = pick_slot(*h, ids, n, &reuse, &live);
    if (i < 0) return -1;
    Slot& s = h->s[i];
    // archprot_review2 L2 / M1: again with the slot known (an ended paused reply is in the store now; the fallback only
    // for a slot source this start may evict)
    srv_protect_begin(*h, ids, n, i, true);
    try {
        auto ms = [](auto a, auto b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
        const auto t0 = std::chrono::steady_clock::now();
        // the slot's old conversation goes to the store first when this prompt gives most of it up
        // (a start that keeps nothing of it gives its prompt-end snapshot up: moved into the entry)
        if (s.held() - reuse > 64 && !s.archived && archive_cheap(*h, i)) srv_archive_slot(*h, i, 0, reuse == 0);
        const auto t1 = std::chrono::steady_clock::now();
        // only the reused rows stay held (no truncate: the state is valid only at the held end or at the snapshot)
        if (s.held() > reuse) {
            s.toks.resize(reuse);
            s.archived = false;
        }
        if (reuse == 0) {
            s.snap.clear();
            s.snap_toks.clear();
        }
        s.prov_cut(reuse);
        // INK_TEST_ROOM_FAIL=K (tests): the room check of the K-th start (counted from 1) fails (the server's retry path)
        static const int room_fail = env_int("INK_TEST_ROOM_FAIL", 0);
        static int nstarts = 0;
        if (++nstarts == room_fail || !srv_room(*h, i, n + SLACK)) {
            // [ink-srv-review-fix] H1: the tokens are cut to the snapshot point, the engine state is still at the old held
            // end: load the snapshot so the slot's state matches its tokens again (srv_room may have moved the held rows,
            // so the old tokens are not restored)
            if (reuse > 0 && !live) {
                srv_bind(*h, i);
                h->e->ring_load(s.snap, reuse);
            }
            return -1;
        }
        const auto t2 = std::chrono::steady_clock::now();
        const Source src = srv_source(*h, i, live, ids, n);
        srv_apply(*h, i, src, ids);
        if (cache_log())
            fprintf(stderr, "[ink cache] slot %d prompt %d: own %d%s, archive %.1f ms, room %.1f ms, source c %d (state %s, rows %s) %.1f ms\n", i, n,
                    reuse, live ? " live" : "", ms(t0, t1), ms(t1, t2), src.c,
                    src.state_slot >= 0 ? (src.state_slot == i ? "own" : "slot") : src.blob ? "blob" : "-",
                    src.rows_slot == i ? "own" : src.rows_slot >= 0 ? "slot" : src.rows_entry >= 0 ? "store" : "-",
                    ms(t2, std::chrono::steady_clock::now()));
        if (src.blob && src.blob == &s.snap) h->snap_used++;
        reuse = src.c;
        s.archived = false;
        static const int hint_min = std::max(1, env_int("INK_HINT_MIN", 256));
        s.pf_hint = hint >= hint_min && hint > reuse && hint < n ? hint : 0;
        s.next = -1;
        s.pf.assign(ids, ids + n);
        s.pf_done = reuse;
        s.pf_reused = reuse;
        s.pf_started = false;
        s.pf_first = -1;
        s.sp = h->pending;
        eng_samp(*h->e, i, s.sp);
        s.pen_from = -1;
        s.limit = -1;
        s.fault.clear();
        s.use = ++h->tick;
        s.admit = (unsigned long long)s.use;   // [ink-resume] the pause order
        h->newest = i;
        return i;
    } catch (const std::exception& ex) {
        fprintf(stderr, "inkling: start: %s\n", ex.what());
        srv_fault(*h, i, ex.what());
        return -2;
    }
} QE_CATCH(-1, -2)
int qe_prefill_more_slot(void* p, int slot, int budget) try {
    Handle* h = (Handle*)p;
    if (slot < 0 || slot >= (int)h->s.size() || h->s[slot].pf_done < 0) return 1;
    try {
        return prefill_some(*h, slot, budget) ? 1 : 0;
    } catch (const std::exception& ex) {
        fprintf(stderr, "inkling: prefill: %s\n", ex.what());
        srv_fault(*h, slot, ex.what());
        return 1;   // (start_finish reports the fault)
    }
} QE_CATCH(slot, 1)
int qe_prefill_more(void* p, int budget) { return qe_prefill_more_slot(p, ((Handle*)p)->newest, budget); }
int qe_start_finish_slot(void* p, int slot) try {
    Handle* h = (Handle*)p;
    if (slot < 0 || slot >= (int)h->s.size()) return -2;
    Slot& s = h->s[slot];
    h->last_slot = slot;
    if (s.pf_done < 0) return -2;
    try {
        while (!prefill_some(*h, slot, 0)) {}
    } catch (const std::exception& ex) {
        fprintf(stderr, "inkling: prefill: %s\n", ex.what());
        srv_fault(*h, slot, ex.what());
        return -2;
    }
    h->reused = s.pf_reused;
    s.next = s.pf_first;
    s.pen_from = s.held();
    pen_count(s, s.next);
    s.pf.clear();
    s.pf_done = -1;
    s.gen = true;
    s.use = ++h->tick;
    if (h->newest == slot) h->newest = -1;
    h->cur = slot;
    return s.next;
} QE_CATCH(slot, -2)
int qe_start_finish(void* p) { return qe_start_finish_slot(p, ((Handle*)p)->newest); }
int qe_num_prefills(void* p) { return num_prefills(*(Handle*)p); }
int qe_prefill_pos(void* p, int slot) {
    Handle* h = (Handle*)p;
    return slot >= 0 && slot < (int)h->s.size() ? h->s[slot].pf_done : -1;
}
int qe_start3(void* p, const int* ids, int n, int k, unsigned long long affinity, int hint) {
    const int slot = qe_start_begin(p, ids, n, k, affinity, hint, nullptr, 0, nullptr, 0);
    if (slot < 0) return slot;
    return qe_start_finish_slot(p, slot);
}

// batch group: one step of each listed slot; out[i * 64 ..] gets cnt[i] tokens of slots[i] (0: not generating, limit
// reached, no room, engine error). Engine-sampled slots (greedy / plain temperature) step together in spec_rows calls
// of <= min(slots, 8) (drafter block + verify; step_rows without the drafter); host-sampled ones (top-p / top-k / min-p,
// penalties) ride in the same call as one row without drafts, their tokens picked from row_logits. INK_BATCH=0: each
// slot alone.
void qe_step_batch(void* p, const int* slots, int n, int* out, int* cnt) try {
    Handle* h = (Handle*)p;
    static const bool batch_on = env_int("INK_BATCH", 1) != 0;
    for (int i = 0; i < n; i++) cnt[i] = 0;
    const double tb0 = prof_on() ? now_ms() : 0;
    double eng_ms = 0;
    int ngroups = 0, nrows = 0, ntok = 0;
    struct ProfLine {   // (printed at every return)
        bool on; double t0; double* e; int *g, *r, *t, n;
        ~ProfLine() {
            if (on) fprintf(stderr, "[ink prof] step %d slots: total %.2f ms, engine %.2f ms, %d calls, %d rows, %d tokens\n", n, now_ms() - t0, *e, *g, *r, *t);
        }
    } pl{prof_on(), tb0, &eng_ms, &ngroups, &nrows, &ntok, n};
    resume_paused(*h, slots, n);
    std::vector<int> act;   // indices into slots[]
    for (int i = 0; i < n; i++) {
        const int k = slots[i];
        if (k < 0 || k >= (int)h->s.size()) continue;
        const Slot& s = h->s[k];
        if (!s.gen || s.paused || s.next < 0 || s.next >= V || s.limit == 0) continue;
        bool dup = false;
        for (int j : act) dup = dup || slots[j] == k;
        if (!dup) act.push_back(i);
    }
    if (act.empty()) return;
    std::vector<int> ok;
    try {   // room for the step's rows (drafts: VR); a slot without room for one row ends (length)
        for (int i : act) {
            const int k = slots[i];
            Slot& s = h->s[k];
            if (s.paused || !s.gen) continue;   // ([ink-resume] paused, or faulted by a pause, in this step)
            if (s.cap >= s.held() + VR + SLACK || srv_room(*h, k, s.held() + VR + SLACK) || room_or_pause(*h, k, s.held() + 1 + SLACK))
                ok.push_back(i);
            else if (!s.paused && s.gen)   // (paused by room_or_pause: it waits for rows; faulted: keeps its error)
                srv_full(*h, k, s.held() + 1 + SLACK);
        }
        std::vector<int> keep;   // a later slot's room may have moved or shrunk an earlier one (or paused it)
        for (int i : ok) {
            const Slot& s = h->s[slots[i]];
            if (s.paused || !s.gen) continue;
            if (s.cap >= s.held() + 1 + SLACK) keep.push_back(i);
            else srv_full(*h, slots[i], s.held() + 1 + SLACK);
        }
        ok.swap(keep);
    } catch (const std::exception& ex) {
        fprintf(stderr, "inkling: step room: %s\n", ex.what());
        for (int i : act) srv_fault(*h, slots[i], ex.what());
        return;
    }
    // one group list: engine-sampled and host-sampled slots together; with the drafter, groups of <= min(slots, 8) in one
    // spec_rows call each (host-sampled slots flagged nodraft: one row, the token from row_logits of their row), without
    // it one step_rows call of <= MROW rows
    const bool dr = h->e->drafter_on();
    const int per = batch_on ? (dr ? std::min((int)h->s.size(), 8) : MROW) : 1;
    if ((int)h->logits.size() != V) h->logits.assign(V, 0.f);
    for (size_t g = 0; g < ok.size(); g += per) {
        const int m = (int)std::min<size_t>(per, ok.size() - g);
        std::vector<int> sl(m), an(m), o((size_t)m * VR), no(m, 0), kd(m, 0);
        std::vector<unsigned char> nd(m, 0);
        bool any_host = false;
        for (int z = 0; z < m; z++) {
            sl[z] = slots[ok[g + z]];
            an[z] = h->s[sl[z]].next;
            nd[z] = h->s[sl[z]].sp.host() ? 1 : 0;
            any_host = any_host || nd[z];
        }
        try {
            for (int z = 0; z < m; z++) bind_only(*h, sl[z], true);
            const double te = prof_on() ? now_ms() : 0;
            ngroups++;
            if (dr) {
                h->e->spec_rows(sl.data(), an.data(), m, o.data(), no.data(), any_host ? nd.data() : nullptr);
                h->e->last_spec_drafts(kd.data());
            } else {
                std::vector<int> o1(m);
                h->e->step_rows(sl.data(), an.data(), m, o1.data());
                for (int z = 0; z < m; z++) {
                    o[(size_t)z * VR] = o1[z];
                    no[z] = 1;
                }
            }
            if (prof_on()) eng_ms += now_ms() - te;
            for (int z = 0, f = 0; z < m; f += 1 + kd[z], z++) {   // host-sampled rows: f = the slot's first row
                if (!nd[z] || o[(size_t)z * VR] < 0 || o[(size_t)z * VR] >= V) continue;   // (a bad row: faulted in take)
                h->e->row_logits(f, h->logits.data());
                o[(size_t)z * VR] = srv_pick_host(*h, h->s[sl[z]], h->logits.data());
                no[z] = 1;
            }
        } catch (const std::exception& ex) {
            fprintf(stderr, "inkling: step: %s\n", ex.what());
            for (int z = 0; z < m; z++) srv_fault(*h, sl[z], ex.what());
            continue;
        }
        h->batch_steps++;
        for (int z = 0; z < m; z++) {
            h->batch_rows += 1 + kd[z];
            nrows += 1 + kd[z];
            ntok += no[z];
            const int i = ok[g + z];
            // (the engine fed nout rows: all of them are taken, the bound interval limit keeps them inside max_tokens)
            const int c = std::max(1, std::min(no[z], VR));
            if (!take(*h, sl[z], &o[(size_t)z * VR], c)) continue;
            h->s[sl[z]].st[1] += kd[z];
            std::copy(&o[(size_t)z * VR], &o[(size_t)z * VR] + c, out + (size_t)i * 64);
            cnt[i] = c;
        }
    }
} QE_CATCH(-1)
int qe_spec_step(void* p, int* out) {
    Handle* h = (Handle*)p;
    if (h->cur < 0) return 0;
    int c = 0;
    qe_step_batch(p, &h->cur, 1, out, &c);
    return c;
}
void qe_resume(void* p, int slot) { ((Handle*)p)->cur = slot; }
void qe_set_concurrent(void* p, int on) { ((Handle*)p)->concurrent = on != 0; }
int qe_last_reused(void* p) { return ((Handle*)p)->reused; }
int qe_last_slot(void* p) { return ((Handle*)p)->last_slot; }
void qe_release(void* p, int slot) try {
    Handle* h = (Handle*)p;
    h->e->ring_finalize();
    if (slot < 0 || slot >= (int)h->s.size()) return;
    Slot& s = h->s[slot];
    if (s.paused) end_paused(*h, s);   // released while paused (finished / cancelled)
    s.gen = false;
    if (s.pf_done >= 0) {   // a prompt dropped mid-prefill: the rows done so far stay (the state is at their end)
        s.pf.clear();
        s.pf_done = -1;
        if (h->newest == slot) h->newest = -1;
    }
    s.use = ++h->tick;
} QE_CATCH(slot)
void qe_prefill_cancel(void* p, int slot) { qe_release(p, slot); }

// [ink-srv-park] room for a new n-token sequence by pausing decoding sequences: the most recently admitted first ([ink-resume]; no
// priority: `prio` is ignored), only as many as needed and only when that makes it fit; a paused slot keeps its slot
// (generating, no tokens) with its rows + state in host memory and resumes by itself (qe_step_batch, first paused
// first) once its rows fit again. 1: fits now.
// [ink-archprot] archprot_review M1 / review2 L4: the prompt of the request the next qe_pause_for (or qe_park) makes room
// for (the server calls it right before): its store source is kept while the paused / parked copies make room, when they
// fit without dropping it (one-shot: the next start / pause / park clears it)
void qe_protect_ids(void* p, const int* ids, int n) try {
    Handle* h = (Handle*)p;
    if (!ids || n <= 0)
        h->protect_next.clear();   // (n 0: drop a pending prompt)
    else
        h->protect_next.assign(ids, ids + n);
} QE_CATCH(-1)
int qe_pause_for(void* p, int n, int prio) try {
    (void)prio;
    Handle* h = (Handle*)p;
    struct Prot {
        Handle& h;
        ~Prot() {
            srv_protect_end(h);
            h.protect_next.clear();
        }
    } prot{*h};
    if (!h->protect_next.empty()) {   // (no fallback: a pause evicts no idle slot)
        h->e->ring_finalize();   // (archprot_review2 nit: snapshot vectors complete before srv_source reads them)
        srv_protect_begin(*h, h->protect_next.data(), (int)h->protect_next.size());
    }
    if (n >= h->seq_max || num_prefills(*h) >= 2) return 0;
    bool free_slot = false;
    for (const Slot& s : h->s) free_slot = free_slot || s.idle();
    if (!free_slot) return 0;   // pausing frees no slot
    static const int reply = std::max(0, env_int("INK_ADMIT_REPLY_ROWS", 2048));
    long busy = 0;
    std::vector<int> vs;
    for (size_t i = 0; i < h->s.size(); i++) {
        const Slot& s = h->s[i];
        if (s.idle() || s.paused) continue;
        busy += std::max(s.cap, s.held());
        // ([ink-resume] resume_review2 L-a: only a reply that can come back)
        if (s.gen && s.next >= 0 && s.cap > 0 && s.pf_done < 0 && pool_fits_ever((long)s.held() + 1 + SLACK, h->pool, h->seq_max))
            vs.push_back((int)i);
    }
    const long room = std::min<long>({(long)reply, (long)h->pool / 16, (long)std::max(0, h->seq_max - n - 1)});
    const long want = (long)n + SLACK + room;
    if (busy + want <= h->pool) return 1;
    // ([ink-resume] resume_review2 L-f: src/pool_order.h, the most recently admitted first, as the growth pause)
    vs = qe_pool::pause_order(vs, [&](int j) { return h->s[j].admit; });
    size_t k = 0;
    long left = busy;
    while (k < vs.size() && left + want > h->pool) {
        const Slot& s = h->s[vs[k++]];
        left -= std::max(s.cap, s.held());
    }
    if (left + want > h->pool) return 0;   // pausing all of them would not make it fit: pause none
    {   // [ink-srv-review-fix] M2: their host copies within the host-RAM budget (store entries dropped first, LRU), else none
        size_t need = 0;
        for (size_t i = 0; i < k; i++) need += park_need(*h, vs[i]);
        // archprot_review2 M2: the protection is a preference; admission wins (a pause that fits only by dropping the
        // protected entry goes ahead as before)
        if (!srv_store_fit(*h, need) && !(h->protect_on && (srv_protect_end(*h), srv_store_fit(*h, need)))) {
            fprintf(stderr, "inkling: pool: no pause for a %d-token start: host memory for %.2f GB of copies is short\n", n, need / 1e9);
            return 0;
        }
    }
    for (size_t i = 0; i < k; i++) {
        const int j = vs[i];
        Slot& s = h->s[j];
        try {
            park_save(*h, j, s.pk, true);
        } catch (const std::exception& ex) {
            fprintf(stderr, "inkling: pause of slot %d: %s\n", j, ex.what());
            srv_fault(*h, j, ex.what());
            return 0;
        }
        fprintf(stderr, "inkling: pool: slot %d paused at %d tokens for a %d-token start (region [%d, %d) freed)\n", j, s.held(), n,
                s.base, s.base + s.cap);
        s.paused = true;
        s.pause_ord = ++h->tick;
        s.cap = 0;
        s.prov_cut(0);
        s.archived = false;
        h->pauses++;
    }
    return 1;
} QE_CATCH(-1, 0)

// [ink-srv-park] park tier: a decoding reply's state to host memory under a handle, its slot released (the rows stay as
// an idle conversation); unpark into any free slot with the same rows and state (no prefill: bit-identical)
int qe_park(void* p, int slot) try {
    Handle* h = (Handle*)p;
    // [ink-archprot] archprot_review2 L4: the queue head's source (qe_protect_ids) stays if the copy fits without it
    // (review3 nit: built before the early returns, so a pending prompt never outlives this call)
    struct Prot {
        Handle& h;
        ~Prot() {
            srv_protect_end(h);
            h.protect_next.clear();
        }
    } prot{*h};
    if (slot < 0 || slot >= (int)h->s.size()) return -1;
    Slot& s = h->s[slot];
    if (!s.gen || s.paused || s.pf_done >= 0 || s.next < 0 || s.cap == 0 || s.toks.empty()) return -1;
    if (!h->protect_next.empty()) {
        h->e->ring_finalize();
        srv_protect_begin(*h, h->protect_next.data(), (int)h->protect_next.size());
    }
    const size_t need = park_need(*h, slot);
    if (!srv_store_fit(*h, need) && !(h->protect_on && (srv_protect_end(*h), srv_store_fit(*h, need))))
        return -1;   // [ink-srv-review-fix] M2 (host-RAM budget; store LRU first)
    Park pk;
    try {
        park_save(*h, slot, pk, false);
    } catch (const std::exception& ex) {
        fprintf(stderr, "inkling: park of slot %d: %s\n", slot, ex.what());
        srv_fault(*h, slot, ex.what());
        return -1;
    }
    const long id = ++h->park_next;
    h->parked.emplace(id, std::move(pk));
    s.gen = false;   // (an idle conversation now: reusable, evictable)
    s.use = ++h->tick;
    if (h->cur == slot) h->cur = -1;
    return (int)id;
} QE_CATCH(slot, -1)
int qe_unpark(void* p, int hd, int k, int* extra) try {
    (void)k;
    Handle* h = (Handle*)p;
    if (extra) *extra = -1;
    auto it = h->parked.find(hd);
    if (it == h->parked.end()) return -2;
    Park& pk = it->second;
    const int n = (int)pk.toks.size();
    // its own slot when it is still idle with the same sequence (state live at its end): nothing to load
    int i = -1;
    if (pk.orig >= 0 && pk.orig < (int)h->s.size()) {
        const Slot& o = h->s[pk.orig];
        if (o.idle() && o.cap > 0 && o.toks == pk.toks) i = pk.orig;
    }
    try {
        if (i >= 0) {
            Slot& s = h->s[i];
            if (!srv_room(*h, i, n + VR + SLACK) && !srv_room(*h, i, n + 1 + SLACK)) return -1;
            srv_bind(*h, i);
            s.next = pk.next, s.pen_from = pk.pen_from, s.sp = pk.sp, s.limit = pk.limit, s.watch = pk.watch;
            s.admit = pk.admit;
            eng_samp(*h->e, i, s.sp);
            std::copy(pk.st, pk.st + 5, s.st);
            s.gen = true;
            s.fault.clear();
            s.use = s.emit = ++h->tick;
        } else {
            for (size_t j = 0; j < h->s.size(); j++) {   // a free slot: one without rows, else the least recently used idle one
                const Slot& o = h->s[j];
                if (!o.idle()) continue;
                if (i < 0 || (o.cap == 0) > (h->s[i].cap == 0) || ((o.cap == 0) == (h->s[i].cap == 0) && o.use < h->s[i].use)) i = (int)j;
            }
            if (i < 0) return -1;
            Slot& s = h->s[i];
            if (s.cap > 0) evict(*h, i);   // (its conversation to the store first)
            if (!srv_room(*h, i, n + VR + SLACK)) return -1;
            park_load(*h, i, pk);
        }
    } catch (const std::exception& ex) {
        fprintf(stderr, "inkling: unpark: %s\n", ex.what());
        if (i >= 0) srv_fault(*h, i, ex.what());
        h->parked.erase(it);
        return -3;
    }
    h->parked.erase(it);
    h->cur = i;
    return i;
} QE_CATCH(-1, -3)
void qe_park_drop(void* p, int hd) try {
    Handle* h = (Handle*)p;
    auto it = h->parked.find(hd);
    if (it == h->parked.end()) return;
    try {
        park_store(*h, it->second);
    } catch (const std::exception& ex) {
        fprintf(stderr, "inkling: park_drop: %s\n", ex.what());
    }
    h->parked.erase(it);
} QE_CATCH(-1)
int qe_num_parked(void* p) { return (int)((Handle*)p)->parked.size(); }

// per-slot draft statistics since the slot's creation: {verify steps, drafted, accepted, 0, 0} (DSpark drafts count as the
// "MTP" pair; no lookup drafts)
void qe_draft_stats(void* p, int slot, long long* out) {
    Handle* h = (Handle*)p;
    for (int i = 0; i < 5; i++) out[i] = 0;
    if (slot < 0 || slot >= (int)h->s.size()) return;
    for (int i = 0; i < 3; i++) out[i] = h->s[slot].st[i];
}

// max_tokens: the reply emits at most `tokens` more; the engine's interval limit keeps the drafts inside it, so the
// held sequence is what the client got
void qe_set_limit(void* p, int slot, int tokens) {
    Handle* h = (Handle*)p;
    if (slot >= 0 && slot < (int)h->s.size()) h->s[slot].limit = tokens < 0 ? -1 : tokens;
}

// stops group: no step runs ahead (watch / no_ahead only record), the drafts are made inside the engine's step (none
// to report). trim: the slot's tokens end at `pos` (prompt + reply) only where that is exact without a truncate: pos
// - 1 == the prompt-end snapshot (the reply's first token becomes the next input again) or the held end itself;
// otherwise 0 and the held sequence stays as it is (it is consistent: the next start reuses it only where its tokens
// match, else the prompt-end snapshot)
void qe_watch_stops(void* p, int slot, int on) {
    Handle* h = (Handle*)p;
    if (slot >= 0 && slot < (int)h->s.size()) h->s[slot].watch = on != 0;
}
void qe_set_no_ahead(void* p, int slot, int on) { (void)p; (void)slot; (void)on; }
int qe_drafts(void* p, int slot, int* out) { (void)p; (void)slot; (void)out; return 0; }
int qe_trim(void* p, int slot, int pos) try {
    Handle* h = (Handle*)p;
    h->e->ring_finalize();
    if (slot < 0 || slot >= (int)h->s.size()) return 0;
    Slot& s = h->s[slot];
    std::vector<int> seq = s.toks;
    if (s.next >= 0) seq.push_back(s.next);
    if (pos >= (int)seq.size()) return 1;
    if (pos < 1 || s.cap == 0 || s.pf_done >= 0) return 0;
    const int rows = pos - 1;
    if (rows == s.held()) return 1;
    const int P = (int)s.snap_toks.size();
    if (rows != P || s.snap.empty() || s.held() < P || !std::equal(s.snap_toks.begin(), s.snap_toks.end(), s.toks.begin())) return 0;
    try {
        s.toks.resize(P);
        srv_bind(*h, slot);
        h->e->ring_load(s.snap, P);
    } catch (const std::exception& ex) {
        srv_fault(*h, slot, ex.what());
        return 0;
    }
    s.next = seq[P];
    s.prov_cut(P);
    s.archived = false;
    return 1;
} QE_CATCH(slot, 0)

long long qe_fault_count(void* p) { return ((Handle*)p)->faults; }
int qe_slot_fault(void* p, int slot, char* buf, int n) {
    Handle* h = (Handle*)p;
    if (slot < 0 || slot >= (int)h->s.size() || h->s[slot].fault.empty() || n < 1) return 0;
    snprintf(buf, n, "%s", h->s[slot].fault.c_str());
    h->s[slot].fault.clear();
    return 1;
}
void qe_reset(void* p) try {
    Handle* h = (Handle*)p;
    h->e->ring_finalize();
    for (Slot& s : h->s) s = Slot{};
    h->cur = h->newest = -1;
    h->parked.clear();
    h->store.clear();
    h->store_bytes = 0;
    h->arch = Entry{};
    h->arch_slot = -1;
    h->ring_est = 0;   // [ink-archfix]
} QE_CATCH(-1)

// start-up warm-up (nothing cached yet): a short dummy prompt on each slot, then row steps of 1, 2, 4, 8 slots
// (drafter blocks + verify) greedy and / or sampled, a host-sampled row; every slot is empty afterwards. Returns seconds.
double qe_warmup(void* p, int modes) try {
    Handle* h = (Handle*)p;
    h->e->ring_finalize();
    for (const Slot& s : h->s)
        if (s.cap > 0 || !s.idle()) return 0.0;
    const auto t0 = std::chrono::steady_clock::now();
    const int S = (int)h->s.size(), L = 48, iv = L + 2 * 4 * VR + 2 + 64 + SLACK;
    if ((long)S * iv > h->pool) return 0.0;
    try {
        std::vector<int> ids(L);
        for (int i = 0; i < L; i++) ids[i] = 1000 + (i * 7919) % 150000;
        std::vector<int> nx(S);
        for (int k = 0; k < S; k++) {
            h->e->select(k);
            h->e->bind(k, k * iv, 0, iv - SLACK);
            h->e->reset();
            h->e->set_sampling(k, 0.f, 0);
            h->e->prefill(ids.data(), L);
            nx[k] = h->e->sample_last();
        }
        std::vector<int> sl(S), o((size_t)S * VR), no(S);
        std::iota(sl.begin(), sl.end(), 0);
        for (int mode = 0; mode < 3; mode++) {   // greedy, sampled, [ink-samp-dev] filtered sampled (with the sampled bit)
            if (!(modes >> std::min(mode, 1) & 1) || (mode == 2 && !h->e->filter_ready())) continue;
            for (int k = 0; k < S; k++) {
                h->e->set_sampling(k, mode ? 1.f : 0.f, 7);
                h->e->set_filter(k, mode == 2 ? 20 : 0, mode == 2 ? 0.95f : 1.f, 0.f);
            }
            for (int m : {1, 2, 4, 8}) {
                if (m > S) break;
                if (h->e->drafter_on()) {
                    h->e->spec_rows(sl.data(), nx.data(), m, o.data(), no.data());
                    for (int z = 0; z < m; z++) nx[z] = o[(size_t)z * VR + no[z] - 1];
                } else {
                    h->e->step_rows(sl.data(), nx.data(), m, o.data());
                    for (int z = 0; z < m; z++) nx[z] = o[z];
                }
                for (int z = 0; z < m; z++)
                    if (nx[z] < 0 || nx[z] >= V) nx[z] = 1000;
            }
        }
        if (modes & 2) {   // a host-sampled row
            h->e->set_sampling(0, 0.f, 0);
            std::vector<int> o1(1);
            h->e->step_rows(sl.data(), nx.data(), 1, o1.data());
            if ((int)h->logits.size() != V) h->logits.assign(V, 0.f);
            h->e->row_logits(0, h->logits.data());
        }
        for (int k = 0; k < S; k++) {
            h->e->set_sampling(k, 0.f, 0);
            h->e->set_filter(k, 0, 1.f, 0.f);
            h->e->select(k);
            h->e->bind(k, 0, 0, 0);
            h->e->reset();
        }
    } catch (const std::exception& ex) {
        fprintf(stderr, "inkling: warm-up: %s\n", ex.what());
        check_broken(ex.what());
    }
    for (Slot& s : h->s) s = Slot{};
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
} QE_CATCH(-1, 0.0)

}  // extern "C"
