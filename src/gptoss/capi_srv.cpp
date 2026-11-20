// [srv-gptoss-capi] Server side of the gpt-oss C API: sampling, the slot pool, the max_tokens limit, the stops group,
// faults. See capi_srv.h. The groups follow server/qe_capi.h / server/engine.py (a group counts only when complete).
#include "capi_srv.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <cstdlib>
#include <exception>
#include <atomic>
#include <thread>
#include <unistd.h>

using namespace gptoss;

namespace gptoss {

// the vocabulary proper: rows from here on have no token (embedding padding)
static constexpr int NTOK = 200019;

static int env_int(const char* k, int d) {
    const char* v = getenv(k);
    return v && *v ? atoi(v) : d;
}

void srv_bind(Handle& h, int i) {
    Slot& s = h.s[i];
    h.e->bind(i, s.base, (int)s.toks.size(), std::max(s.cap, 1));
    h.bound = i;
}

void srv_arm(Handle& h, Samp& sp, bool prefill, bool masked) {
    // a sampled or constrained prefill reads its last row's logits on the host (once per reply); a sampled decode step
    // draws on the cards (Engine::set_sampling) and never dumps (the dump would turn the graph off); a constrained
    // decode step runs the sampled head with its logits copied to the host
    Dump d;
    if ((sp.temp > 0.f || masked) && prefill) {
        if ((int)h.logits.size() != VOCAB) h.logits.assign(VOCAB, 0.f);
        d.logits = h.logits.data();
    }
    h.e->set_dump(d);
    if (prefill)
        h.e->set_sampling(0.f, sp.seed, false);
    else if (masked)
        h.e->set_sampling(sp.temp > 0.f ? sp.temp : 1.f, sp.seed, true);
    else
        h.e->set_sampling(sp.temp, sp.seed, sp.filtered(), !sp.one());
}
void srv_pen_sync(Handle& h, int i) {
    Slot& s = h.s[i];
    if (s.sp.pen() && !s.pen_synced) {
        std::vector<int> ids;
        std::vector<float> vals;
        for (const auto& kv : s.sp.cnt) {
            if (kv.first < 0 || kv.first >= VOCAB) continue;
            ids.push_back(kv.first);
            vals.push_back(s.sp.pen_val(kv.first));
        }
        h.e->pen_reset(i, ids.data(), vals.data(), (int)ids.size());
        s.pen_synced = true;
    }
}
void srv_arm_pen(Handle& h, int i) {
    Slot& s = h.s[i];
    if (!s.sp.pen()) return;
    srv_pen_sync(h, i);
    h.e->set_sampling_penalty(i, s.next, s.sp.pen_val(s.next));
}
void srv_pen_count(Slot& s, int tok) {
    if (s.sp.pen() && tok >= 0) s.sp.cnt[tok]++;
}
void srv_pen_trim(Slot& s, const std::vector<int>& seq, int pos) {
    if (s.pen_from > pos) s.pen_from = pos;   // (a trim into the prompt: tokens generated after it are reply tokens again)
    if (!s.sp.pen()) return;
    s.sp.cnt.clear();
    for (int id : s.sp.seed_ids) s.sp.cnt[id] = 1;
    for (int q = std::max(s.pen_from, 0); s.pen_from >= 0 && q < pos && q < (int)seq.size(); q++)
        if (seq[q] >= 0) s.sp.cnt[seq[q]]++;
    s.pen_synced = false;
}
void srv_pen_logits(const Slot& s, float* x) {
    if (!s.sp.pen()) return;
    for (const auto& kv : s.sp.cnt)
        if (kv.first >= 0 && kv.first < NTOK) x[kv.first] -= s.sp.pen_val(kv.first);
}

// the step's full logits on the host (after a constrained decode step: the cards' mapped copies)
static void gather_logits(Handle& h) {
    if ((int)h.logits.size() != VOCAB) h.logits.assign(VOCAB, -INFINITY);
    for (int g = 0; g < h.e->plan().ngpu; g++) {
        const SampleCard sc = h.e->sample_card(g);
        memcpy(h.logits.data() + sc.v0, sc.logits, (size_t)sc.vn * 4);
    }
}
int srv_pick_step(Handle& h, Slot& s, int engine_tok) {
    if (!s.constrained) return srv_pick(h, s.sp, engine_tok, false);
    gather_logits(h);
    return srv_pick_masked(h, s, h.logits.data());
}

static int pick_host(Samp& h, const float* x, int greedy) {
    const int n = NTOK;
    if (h.one()) {   // the argmax (the lowest id of equal values), as the cards take it
        int best = -1;
        for (int i = 0; i < n; i++)
            if (std::isfinite(x[i]) && (best < 0 || x[i] > x[best])) best = i;
        return best < 0 ? greedy : best;
    }
    // candidates: top_k (0 = all), by logit
    std::vector<int> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    int k = h.top_k > 0 && h.top_k < n ? h.top_k : n;
    if (k < n) {
        std::nth_element(idx.begin(), idx.begin() + k, idx.end(), [&](int a, int b) { return x[a] > x[b]; });
        idx.resize(k);
    }
    float mx = -INFINITY;
    for (int i : idx) mx = std::max(mx, std::isfinite(x[i]) ? x[i] : -INFINITY);
    if (!std::isfinite(mx)) return greedy;
    std::vector<double> p(idx.size());
    double sum = 0;
    for (size_t j = 0; j < idx.size(); j++) {
        const float v = x[idx[j]];
        p[j] = std::isfinite(v) ? std::exp((double)(v - mx) / h.temp) : 0.0;
        sum += p[j];
    }
    // min_p: relative to the most likely token (p = 1 there before normalising)
    if (h.min_p > 0.f)
        for (double& q : p)
            if (q < h.min_p) { sum -= q; q = 0; }
    // top_p: the smallest set of the most likely tokens whose mass reaches top_p
    if (h.top_p < 1.f) {
        std::vector<int> o(idx.size());
        std::iota(o.begin(), o.end(), 0);
        std::sort(o.begin(), o.end(), [&](int a, int b) { return p[a] > p[b]; });
        double acc = 0, keep = h.top_p * sum;
        size_t j = 0;
        for (; j < o.size() && acc < keep; j++) acc += p[o[j]];
        for (; j < o.size(); j++) { sum -= p[o[j]]; p[o[j]] = 0; }
    }
    if (!(sum > 0)) return greedy;
    std::uniform_real_distribution<double> u(0.0, sum);
    double r = u(h.rng);
    for (size_t j = 0; j < idx.size(); j++) {
        r -= p[j];
        if (r <= 0 && p[j] > 0) return idx[j];
    }
    for (size_t j = idx.size(); j-- > 0;)
        if (p[j] > 0) return idx[j];
    return greedy;
}

// a decode step's draw from the cards' results (Engine::sample_card): plain temperature = the cards' Gumbel draw
// (`drawn`); top_k / top_p / min_p from the cards' 64 largest when they hold the whole kept set (exact), else from
// the step's full logits on the host (exact, slower)
static int pick_cards(Handle& hh, Samp& h, int drawn) {
    if (!h.filtered()) return drawn;
    const int ng = hh.e->plan().ngpu;
    const double it = 1.0 / h.temp;
    double M = -INFINITY;
    for (int g = 0; g < ng; g++) M = std::max(M, (double)hh.e->sample_card(g).out[0]);
    if (!std::isfinite(M)) return drawn;
    double Z = 0, cut = -INFINITY;   // cut: every token outside the candidates has a logit <= cut
    std::vector<std::pair<float, int>> c;
    for (int g = 0; g < ng; g++) {
        const SampleCard sc = hh.e->sample_card(g);
        if (std::isfinite(sc.out[0])) Z += sc.out[1] * std::exp((sc.out[0] - M) * it);
        const int nc = (int)sc.out[2];
        for (int j = 0; j < nc; j++) {
            int id;
            memcpy(&id, sc.out + 4 + 64 + j, 4);
            c.push_back({sc.out[4 + j], id});
        }
        if (nc == 64) cut = std::max(cut, (double)sc.out[4 + 63]);
    }
    std::sort(c.begin(), c.end(), [](auto& a, auto& b) { return a.first > b.first || (a.first == b.first && a.second < b.second); });
    // complete: the leading candidates that are surely the vocabulary's largest, in order
    size_t complete = 0;
    while (complete < c.size() && c[complete].first > cut) complete++;
    size_t n = c.size();          // the kept set = c[0, n) once `known`
    bool known = false;
    if (h.top_k > 0) {
        if (h.top_k > 64 && (size_t)h.top_k > complete) { hh.fallbacks++; return -1; }
        n = std::min(n, (size_t)h.top_k);
        known = true;
    }
    if (h.min_p > 0.f) {
        const double thr = M + h.temp * std::log((double)h.min_p);
        if (!known && !(thr > cut)) { hh.fallbacks++; return -1; }
        size_t j = 0;
        while (j < n && c[j].first >= thr) j++;
        n = j;
        known = true;
    }
    std::vector<double> p(n);
    double sum = 0;
    for (size_t j = 0; j < n; j++) sum += (p[j] = std::exp((c[j].first - M) * it));
    if (h.top_p < 1.f) {
        const double mass = known ? sum : Z;
        const double keep = h.top_p * mass;
        double acc = 0;
        size_t j = 0;
        const size_t lim = known ? n : complete;
        for (; j < lim && acc < keep; j++) acc += p[j];
        if (acc < keep && !known) { hh.fallbacks++; return -1; }
        n = j;
        sum = acc;
        known = true;
    }
    if (!known || n == 0 || !(sum > 0)) return drawn;
    std::uniform_real_distribution<double> u(0.0, sum);
    double r = u(h.rng);
    for (size_t j = 0; j < n; j++) {
        r -= p[j];
        if (r <= 0) return c[j].second;
    }
    return c[n - 1].second;
}

int srv_pick_masked(Handle& h, Slot& s, float* x) {
    (void)h;
    const uint32_t* m = s.mask.data();
    int best = -1;
    for (int i = 0; i < NTOK; i++) {
        if (!(m[i >> 5] >> (i & 31) & 1u) || !std::isfinite(x[i])) {
            x[i] = -INFINITY;
            continue;
        }
        if (best < 0 || x[i] > x[best]) best = i;
    }
    if (best < 0) return -1;   // nothing allowed (the server's grammar is in an error state)
    if (s.sp.temp <= 0.f) return best;
    return pick_host(s.sp, x, best);
}

int srv_pick(Handle& h, Samp& sp, int greedy, bool prefill) {
    if (sp.temp <= 0.f) return greedy;
    if (prefill) return (int)h.logits.size() == VOCAB ? pick_host(sp, h.logits.data(), greedy) : greedy;
    h.cards++;
    const int t = pick_cards(h, sp, greedy);
    if (t >= 0) return t;
    gather_logits(h);   // the full logits of this step from the cards' host copies
    return pick_host(sp, h.logits.data(), greedy);
}

static bool snap_on() {
    static const bool on = env_int("GPTOSS_SNAP", 1) != 0;
    return on;
}

void srv_snap_save(Handle& h, Slot& s, const int* ids, int n) {
    if (!snap_on()) return;
    h.e->ring_save(s.snap);
    s.snap_toks.assign(ids, ids + n);
    s.snap_hi = s.hi;
}

int srv_reuse(const Handle& h, const Slot& s, const int* ids, int n, bool* snap) {
    (void)h;
    *snap = false;
    if (s.cap == 0) return 0;
    const int have = (int)s.toks.size();
    int common = 0;
    while (common < have && common < n && s.toks[common] == ids[common]) common++;
    if (common == n) common = n - 1;   // the last row is recomputed for its logits
    if (common > 0 && s.hi - common <= 128) return common;
    // back to the snapshot: the full layers' rows < P are intact while the held sequence still extends its prompt
    const int P = (int)s.snap_toks.size();
    if (!snap_on() || P < 1 || P > n - 1 || s.snap.empty() || have < P || s.snap_hi - P > 128) return 0;
    for (int i = 0; i < P; i++)
        if (ids[i] != s.snap_toks[i] || s.toks[i] != s.snap_toks[i]) return 0;
    *snap = true;
    return P;
}

static bool idle(const Slot& s) { return !s.gen && s.pf_done < 0; }
static int held(const Slot& s) { return (int)s.toks.size(); }

long srv_busy_rows(const Handle& h) {
    long n = 0;
    for (const Slot& s : h.s)
        if (!idle(s)) n += std::max(s.cap, held(s));
    return n;
}

static void evict(Handle& h, int j) {
    Slot& s = h.s[j];
    try {   // [srv-gptoss-cache] its conversation into the host store first (nothing when the store has it already),
        // when that is a bounded copy [review M4: a whole 100K-row conversation stalled every running stream]
        if (srv_archive_cheap(h, j)) srv_archive_slot(h, j);
    } catch (const std::exception& ex) {
        fprintf(stderr, "archive of slot %d: %s\n", j, ex.what());
        if (h.e->broken()) srv_fault(h, -1, ex.what());   // (review rel1 L11: a broken engine ends the process now)
    }
    s.toks.clear();
    s.next = -1;
    s.cap = 0;
    s.hi = s.snap_hi = 0;
    s.snap.clear();
    s.snap_toks.clear();
    s.prov_cut(0);
    h.evictions++;
}

// rows [src, src + n) -> [dst, dst + n), the ranges may overlap: steps of at most |src - dst| rows, in the order that
// never overwrites a row before it is read
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
    const int P = h.e->max_ctx(), grow = std::max(64, env_int("GPTOSS_SLOT_GROW", 2048));
    // one conversation never holds more than the context served, however large the pool is (review r11 P2: the
    // decode paths ask only here; the Python server ends replies before, a bare C API caller would grow on)
    if (need > P || need > h.seq_max) return false;
    if (me.cap >= need) return true;
    bool compacted = false;
    for (;;) {
        // free gaps, my own interval counted as free
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
        const int h0 = me.cap > 0 ? held(me) : 0;
        // 1. in place: the gap holding my base
        if (me.cap > 0)
            for (auto& g : gaps)
                if (g.first <= me.base && me.base < g.second && g.second - me.base >= need) {
                    me.cap = std::min(g.second - me.base, need + grow);
                    return true;
                }
        // 2. a place in a gap (the largest first): a new sequence in the middle of it (room to grow for it and for the one
        // before it), a held one where its rows need no overlapping copy, else at the gap's start (copied in steps)
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
        // 3. compaction (once): every slot's rows to the pool's start in order; idle intervals (and mine: sized right
        // after) cut to their rows, busy ones keep the room of their next step / prompt [review H1: a generating slot
        // cut to its rows wrote its next row into the neighbour's]; a copy inside VRAM (~1 GB per card per 100K rows)
        // is cheaper than losing a cached conversation
        if (!compacted) {
            compacted = true;
            std::vector<int> ord;
            for (size_t j = 0; j < h.s.size(); j++)
                if (h.s[j].cap > 0) ord.push_back((int)j);
            std::sort(ord.begin(), ord.end(), [&](int x, int y) { return h.s[x].base < h.s[y].base; });
            auto keep = [&](int j) {
                const Slot& o = h.s[j];
                const int r = held(o);
                if (j == i || idle(o)) return r;
                const int want = std::max(r + 16, o.pf_done >= 0 ? (int)o.pf.size() + 1 : 0);   // (16: a step and its drafts)
                return std::max(r, std::min(o.cap, want));
            };
            long total = 0;
            for (int j : ord) total += keep(j);
            if (total <= P) {
                int to = 0;
                for (int j : ord) {
                    Slot& o = h.s[j];
                    const int k = keep(j), r = held(o);
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
            if ((int)j != i && h.s[j].cap > 0 && idle(h.s[j]) && (lru < 0 || h.s[j].use < h.s[lru].use)) lru = (int)j;
        if (lru < 0) return false;
        evict(h, lru);
        compacted = false;   // (the freed rows may sit between the others)
    }
}

// [gptoss-rel1-exit] the server's last word before the process ends (qe_set_exit_hook; server/app.py _on_engine_exit
// fails the open requests with an error event). Run on its own thread, bounded by QE_EXIT_HOOK_S (5) seconds, as the
// 27B engine's watchdog does; it must not call into the engine.
static std::atomic<void (*)(const char*)> g_exit_hook{nullptr};

[[noreturn]] static void srv_exit_broken(const char* what) {
    static char reason[640];
    snprintf(reason, sizeof reason, "gpt-oss engine error, the cards are in an unknown state: %s", what ? what : "engine error");
    fprintf(stderr, "gptoss: %s\n", reason);
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
    fprintf(stderr, "gptoss: exiting (code 75) so the server can be restarted\n");
    fflush(stderr);
    fflush(stdout);
    _exit(75);
}

// [gptoss-rel1-full] a running reply that cannot grow: no room in the pool next to the other running replies (every
// idle conversation already went to the host archive, srv_room step 4), or its own context is full. The reply ends
// (the server says finish_reason "length", not an error) and the rows it holds stay: the conversation is idle and
// reusable for its next turn like any finished one.
void srv_full(Handle& h, int i, int need) {
    h.faults++;
    h.full_ends++;
    if (i < 0 || i >= (int)h.s.size()) return;
    Slot& s = h.s[i];
    s.fault = need > h.seq_max ? "KV pool full: this conversation reached its context of " + std::to_string(h.seq_max) + " rows"
                               : "KV pool full: no room for this sequence to grow next to the running ones";
    s.gen = false;
    s.next = -1;
    fprintf(stderr, "gptoss: slot %d: %s (%d rows held); the reply ends with finish_reason length\n", i, s.fault.c_str(),
            (int)s.toks.size());
}

void srv_fault(Handle& h, int i, const char* what) {
    h.faults++;
    // [gptoss-rel1-exit] a broken engine fails every later call while /health stays ok: end the process (75) so
    // llama-swap / Docker restart it (GPTOSS_BROKEN_EXIT=0: keep the process, every request fails, for debugging)
    if (h.e && h.e->broken() && env_int("GPTOSS_BROKEN_EXIT", 1)) srv_exit_broken(what);
    if (i < 0 || i >= (int)h.s.size()) return;
    Slot& s = h.s[i];
    s.fault = what ? what : "engine error";
    s.gen = false;
    s.pf_done = -1;
    s.toks.clear();
    s.next = -1;
    s.hi = s.snap_hi = 0;
    s.snap.clear();
    s.snap_toks.clear();
    s.prov_cut(0);
}

}  // namespace gptoss

extern "C" {

// [gptoss-rel1-exit] the server's fatal-exit hook (see srv_exit_broken); nullptr removes it
__attribute__((visibility("default"))) void qe_set_exit_hook(void (*f)(const char*)) { g_exit_hook.store(f); }

// grammar group (engine.py "grammar"): masks [8][(vocab + 31) / 32] (row 0 = the next token), the spec io, the switch
unsigned* qe_mask_buffer(void* p, int slot) {
    Handle* h = (Handle*)p;
    if (slot < 0 || slot >= (int)h->s.size()) return nullptr;
    Slot& s = h->s[slot];
    if (s.mask.empty()) s.mask.assign((size_t)8 * MASK_WORDS, 0xFFFFFFFFu);
    return s.mask.data();
}
int* qe_spec_io(void* p, int slot) {
    Handle* h = (Handle*)p;
    return slot >= 0 && slot < (int)h->s.size() ? h->s[slot].io : nullptr;
}
void qe_constrain(void* p, int slot, int on) {
    Handle* h = (Handle*)p;
    if (slot < 0 || slot >= (int)h->s.size()) return;
    Slot& s = h->s[slot];
    if (on && s.mask.empty()) s.mask.assign((size_t)8 * MASK_WORDS, 0xFFFFFFFFu);
    s.constrained = on != 0;
}
int qe_constrained(void* p, int slot) {
    Handle* h = (Handle*)p;
    return slot >= 0 && slot < (int)h->s.size() && h->s[slot].constrained;
}

// max_tokens: the reply may emit at most `tokens` more (one token per step today, so the server's own count ends it
// first; with drafts this keeps the held sequence = what the client got)
void qe_set_limit(void* p, int slot, int tokens) {
    Handle* h = (Handle*)p;
    if (slot >= 0 && slot < (int)h->s.size()) h->s[slot].limit = tokens < 0 ? -1 : tokens;
}

// stops group (engine.py "stops"): no step runs ahead and no drafts exist, so watch / no_ahead only record; trim moves
// the sequence back inside its last 128 rows (the sliding layers' rings still hold that window)
void qe_watch_stops(void* p, int slot, int on) {
    Handle* h = (Handle*)p;
    if (slot >= 0 && slot < (int)h->s.size()) h->s[slot].watch = on != 0;
}
void qe_set_no_ahead(void* p, int slot, int on) { (void)p; (void)slot; (void)on; }
int qe_drafts(void* p, int slot, int* out) { (void)p; (void)slot; (void)out; return 0; }
// keep the first `pos` tokens of prompt + reply (the last one becomes the next input); 1 = done, 0 = not possible
// (more than 128 rows back): the slot's rows are then dropped and the next prompt prefills again
int qe_trim(void* p, int slot, int pos) {
    Handle* h = (Handle*)p;
    if (slot < 0 || slot >= (int)h->s.size()) return 0;
    Slot& s = h->s[slot];
    std::vector<int> seq = s.toks;
    if (s.next >= 0) seq.push_back(s.next);
    if (pos >= (int)seq.size()) return 1;
    if (pos < 1 || s.cap == 0) return 0;
    const int rows = pos - 1;
    if (s.hi - rows > 128) {
        s.prov_cut(0);
        s.toks.clear();
        s.next = -1;
        s.snap.clear();
        s.snap_toks.clear();
        srv_pen_trim(s, seq, 0);   // [gptoss-rel3-pentrim] nothing of the reply is held
        return 0;
    }
    try {
        srv_bind(*h, slot);
        h->e->truncate(rows);
    } catch (const std::exception& ex) {
        srv_fault(*h, slot, ex.what());
        return 0;
    }
    s.next = seq[rows];
    s.toks.resize(rows);
    s.prov_cut(rows);
    srv_pen_trim(s, seq, pos);   // [gptoss-rel3-pentrim] the trimmed tokens leave the penalty counts (review rel2 L4)
    s.lk_clear();
    s.archived = false;
    return 1;
}

// faults (engine.py: qe_fault_count / qe_slot_fault): an engine error inside a start or a step fails that request with
// its message instead of an empty reply
long long qe_fault_count(void* p) { return ((Handle*)p)->faults; }
int qe_slot_fault(void* p, int slot, char* buf, int n) {
    Handle* h = (Handle*)p;
    if (slot < 0 || slot >= (int)h->s.size() || h->s[slot].fault.empty() || n < 1) return 0;
    snprintf(buf, n, "%s", h->s[slot].fault.c_str());
    h->s[slot].fault.clear();
    return 1;
}
// forget every held sequence (cached prefixes)
void qe_reset(void* p) {
    Handle* h = (Handle*)p;
    for (size_t i = 0; i < h->s.size(); i++) {
        Slot& s = h->s[i];
        s = Slot{};
    }
    h->cur = -1;
    h->newest = -1;
    h->store.clear();   // [srv-gptoss-cache]
    h->store_bytes = 0;
    h->arch = Entry{};
    h->arch_slot = -1;
    h->e->reset();
}

}  // extern "C"
