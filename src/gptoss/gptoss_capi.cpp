// C API of the gpt-oss engine (server/qe_capi.h): tier 0, and ([srv-gptoss-capi]) the chunked and batch groups over
// several sequence slots (capi_srv.h: the slot pool; time-sliced decode until the engine has a batched step).
//   A start reuses the slot's held sequence when the new prompt extends it (or cuts at most 128 rows off its end: the
//   sliding layers' 256-row rings still hold the window of the cut point), else its prompt-end snapshot, else it
//   prefills from row 0. The slot chosen is the one that reuses most, else a free one, else the least recently used idle one.
#include "gptoss.h"
#include "capi_srv.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <sstream>
#include <string>
#include <vector>

using namespace gptoss;

extern "C" {   // (defined in the extern "C" part below)
static int lookup_rows();
static void lk_extend(Slot& s, int upto);
}

namespace {

bool cache_log() {
    static const bool on = [] {
        const char* v = getenv("GPTOSS_CACHE_LOG");
        return v && *v && atoi(v) != 0;
    }();
    return on;
}

int pick_slot(Handle& h, const int* ids, int n, int* reuse, bool* snap) {
    int best = -1, br = -1;
    bool bs = false;
    for (size_t i = 0; i < h.s.size(); i++) {
        const Slot& s = h.s[i];
        if (s.gen || s.pf_done >= 0) continue;
        bool sn = false;
        const int r = srv_reuse(h, s, ids, n, &sn);
        // more rows reused wins; at 0: a slot without rows, else the least recently used
        const bool better = best < 0 || r > br ||
                            (r == br && r == 0 && ((s.cap == 0) > (h.s[best].cap == 0) ||
                                                   ((s.cap == 0) == (h.s[best].cap == 0) && s.use < h.s[best].use)));
        if (better) {
            best = (int)i;
            br = r;
            bs = sn;
        }
    }
    *reuse = br;
    *snap = bs;
    return best;
}

// one prefill call of slot i: up to `budget` rows of its prompt (all: budget <= 0); true when the prompt is complete
bool prefill_some(Handle& h, int i, int budget) {
    Slot& s = h.s[i];
    const int n = (int)s.pf.size();
    if (s.pf_done >= n) return true;
    int k = n - s.pf_done;
    if (budget > 0 && budget < k) k = budget;
    if (s.pf_hint > s.pf_done) k = std::min(k, s.pf_hint - s.pf_done);   // [srv-gptoss-cache] stop once at the system part's end
    srv_bind(h, i);
    const bool last = s.pf_done + k == n;
    srv_arm(h, s.sp, last, s.constrained);
    s.pf_started = true;
    const int g = h.e->prefill(s.pf.data() + s.pf_done, k);
    s.toks.insert(s.toks.end(), s.pf.begin() + s.pf_done, s.pf.begin() + s.pf_done + k);
    s.pf_done += k;
    s.hi = std::max(s.hi, s.pf_done);   // [review H5] the rings' end
    // the lookup index of a greedy reply grows with its prompt, chunk by chunk (review srv L6: the first decode step
    // built it for the whole prompt, ~100K inserts inside a step that every joint slot waited for); the same entries
    // as lookup_propose would make (positions <= toks.size() read only the prompt's rows)
    if (!s.constrained && s.sp.temp <= 0.f && lookup_rows() > 1 && (int)s.toks.size() >= 6) lk_extend(s, (int)s.toks.size() + 1);
    if (s.pf_hint > 0 && s.pf_done == s.pf_hint) srv_hint_save(h, i);   // [srv-gptoss-cache]
    if (last) {
        if (s.sp.pen() && (int)h.logits.size() == VOCAB) srv_pen_logits(s, h.logits.data());   // [gptoss-rel2-penalty] the seed tokens
        s.pf_first = s.constrained ? srv_pick_masked(h, s, h.logits.data()) : srv_pick(h, s.sp, g, true);
        if (s.pf_first < 0) throw std::runtime_error("structured output: the mask allows no token");
        srv_snap_save(h, s, s.pf.data(), n);
    }
    return last;
}

int num_prefills(const Handle& h) {
    int c = 0;
    for (const Slot& s : h.s) c += s.pf_done >= 0;
    return c;
}

}  // namespace

extern "C" {

// NULL when the engine cannot start (missing / bad files, no memory for the weights plus 256 context rows, CUDA error)
void* qe_create(const char* model_dir, const char* devices, int max_t, const char* split) {
    (void)split;
    Handle* h = nullptr;
    try {
        Config cfg;
        cfg.model_dir = model_dir ? model_dir : "";
        if (devices && *devices) {
            cfg.devices.clear();
            std::stringstream ss(devices);
            std::string t;
            while (std::getline(ss, t, ',')) cfg.devices.push_back(atoi(t.c_str()));
        }
        // The context of one conversation is what the caller asks for (GPTOSS_MAX_CTX caps it; the model's own limit
        // is 131,072). The K/V pool that the slots share is every row that fits in the free VRAM (round 11; before:
        // the pool was the context asked for, so 131,072 rows at most and 32,768 with --max-ctx 32768);
        // GPTOSS_POOL_ROWS=N asks for N rows instead (never fewer than one conversation: a smaller or unreadable N
        // means "one conversation", the rule before round 11).
        const char* c = getenv("GPTOSS_MAX_CTX");
        const int cap = std::min(131072, c ? atoi(c) : 131072);   // never above the model's positions (review r11 P5)
        const int seq = std::max(256, max_t > 0 && max_t < cap ? max_t : cap);
        const char* pr = getenv("GPTOSS_POOL_ROWS");
        cfg.ctx_all = !(pr && *pr);
        // A server builds its decode graphs as requests come (row counts 1..16 x 3 split classes, greedy and sampled:
        // up to 96 per card), after the pool is sized. Measured (stress_bf.free: c = 8 / 16 greedy, c = 8 sampled,
        // short and 3K-token prompts): 292 MiB on the tightest card, which had 401 MiB free after the load. The pool
        // therefore leaves GPTOSS_POOL_RESERVE_MB (default 320) more than the engine's own 384 MiB (review r11 P1).
        const char* rs = getenv("GPTOSS_POOL_RESERVE_MB");
        if (cfg.ctx_all) cfg.reserve_extra = (size_t)std::max(0, rs && *rs ? atoi(rs) : 320) << 20;
        cfg.max_ctx = cfg.ctx_all ? seq : std::max(atoi(pr), seq);
        if (const char* ch = getenv("GPTOSS_CHUNK")) cfg.chunk = atoi(ch);
        // [srv-gptoss-capi] sequence slots (GPTOSS_SLOTS, 1..16): a ring set each (Config.slots), the rows shared.
        // Default since round 6: 8 (c = 8 in one joint step) when the pool asked for is >= 32,768 rows, else 4 as
        // before: admission reserves no reply room, so 8 sequences in a small pool would end in "KV pool full"
        // faults where 4 queued (review r6 M2).
        const char* ns = getenv("GPTOSS_SLOTS");
        // (the rule stays on the context asked for, as before the pool was separated from it: review r11 P3)
        cfg.slots = std::max(1, std::min(16, ns && *ns ? atoi(ns) : (seq >= 32768 ? 8 : 4)));
        h = new Handle;
        h->e = new Engine(cfg);
        h->seq_max = std::min(seq, h->e->max_ctx());
        if (h->seq_max < seq)
            fprintf(stderr, "gptoss: WARNING context per conversation reduced from %d to %d rows: that is what fits in the free VRAM\n", seq, h->seq_max);
        fprintf(stderr, "gptoss: K/V pool %d rows for %d slots (%s), context per conversation %d rows\n", h->e->max_ctx(), cfg.slots,
                cfg.ctx_all ? "every row that fits; GPTOSS_POOL_ROWS=N pins it" : "GPTOSS_POOL_ROWS", h->seq_max);
        h->s.resize(std::max(1, std::min(16, h->e->num_slots())));
        return h;
    } catch (const std::exception& e) {
        fprintf(stderr, "%s\ngptoss: qe_create failed\n", e.what());
        if (h) delete h->e;
        delete h;
        return nullptr;
    }
}
void qe_destroy(void* p) {
    Handle* h = (Handle*)p;
    if (!h) return;
    delete h->e;
    delete h;
}
// the rows one conversation may hold (the context served); qe_pool_tokens = the rows all slots share
int qe_max_context(void* p) { return ((Handle*)p)->seq_max; }
long long qe_pool_tokens(void* p) { return ((Handle*)p)->e->max_ctx(); }
int qe_num_slots(void* p) { return (int)((Handle*)p)->s.size(); }
// [gptoss-lows] rows of one engine prefill pass: a prefill call of more rows splits into the fewest equal passes, so the
// server's idle chunk stays a multiple of it (review pfidle L1: 10,240 over a 4,096-row engine chunk ran as 3 x 3,413)
int qe_prefill_chunk(void* p) { return ((Handle*)p)->e->chunk(); }
// [srv-gptoss-capi] temperature <= 0: greedy (the engine's argmax); else sampled (decode: on the cards; top_k 0 = no
// cut, top_p, min_p, seeded). Applies to the next start, and to the newest prompt in flight while none of its chunks ran
void qe_set_sampling(void* p, float temperature, float top_p, int top_k, float min_p, unsigned long long seed) {
    Handle* h = (Handle*)p;
    Samp& sp = h->pending;
    sp.temp = temperature > 0 ? temperature : 0.f;
    sp.top_p = top_p > 0 && top_p < 1 ? top_p : 1.f;
    sp.top_k = top_k > 0 ? top_k : 0;
    sp.min_p = min_p > 0 && min_p < 1 ? min_p : 0.f;
    sp.seed = seed;
    sp.rng.seed(seed);
    sp.pres = sp.freq = 0.f;   // [gptoss-rel2-penalty] off until qe_set_penalty
    sp.cnt.clear();
    sp.seed_ids.clear();
    if (h->newest >= 0 && h->s[h->newest].pf_done >= 0 && !h->s[h->newest].pf_started) {
        h->s[h->newest].sp = sp;
        h->s[h->newest].limit = -1;
        h->s[h->newest].pen_synced = false;
    }
}
// [gptoss-rel2-penalty] presence / frequency penalty (server/qe_capi.h): right after qe_set_sampling, for the same
// reply (the next start, and the newest prompt in flight while none of its chunks ran). Sampled replies only; the
// server sends "greedy with a penalty" as temperature 1, top_k 1 (the argmax of the penalised logits).
void qe_set_penalty(void* p, float presence, float frequency, const int* seed, int n) {
    Handle* h = (Handle*)p;
    Samp& sp = h->pending;
    // (OpenAI's range is -2 .. 2; clamped so that frequency * count stays finite whatever a caller sends: review rel2 L1)
    sp.pres = std::isfinite(presence) ? std::min(100.f, std::max(-100.f, presence)) : 0.f;
    sp.freq = std::isfinite(frequency) ? std::min(100.f, std::max(-100.f, frequency)) : 0.f;
    sp.cnt.clear();
    sp.seed_ids.clear();
    for (int i = 0; seed && i < n; i++)
        if (seed[i] >= 0 && seed[i] < VOCAB && sp.cnt.emplace(seed[i], 1).second) sp.seed_ids.push_back(seed[i]);   // each once
    if (h->newest >= 0 && h->s[h->newest].pf_done >= 0 && !h->s[h->newest].pf_started) {
        Slot& s = h->s[h->newest];
        s.sp.pres = sp.pres, s.sp.freq = sp.freq, s.sp.cnt = sp.cnt, s.sp.seed_ids = sp.seed_ids;
        s.pen_synced = false;
    }
}
// a free slot and room for n rows next to the running sequences (idle ones are evicted for it)
int qe_can_admit(void* p, int n) {
    Handle* h = (Handle*)p;
    bool free_slot = false;
    for (const Slot& s : h->s) free_slot = free_slot || (!s.gen && s.pf_done < 0);
    // [gptoss-rel1-full] room for the prompt plus reply room (GPTOSS_ADMIT_REPLY_ROWS, default 2,048; at most 1/16 of the
    // pool and never past the conversation's context) when other replies run: a new prompt that fills the pool leaves the running ones no room to
    // grow (before: busy + n + 1 <= pool, then "KV pool full" mid-reply). Alone it is admitted whenever it fits.
    static const int reply = [] {
        const char* v = getenv("GPTOSS_ADMIT_REPLY_ROWS");
        return std::max(0, v && *v ? atoi(v) : 2048);
    }();
    const long busy = srv_busy_rows(*h);
    const long room = std::min<long>({(long)reply, (long)h->e->max_ctx() / 16, (long)std::max(0, h->seq_max - n - 1)});
    const long want = (long)n + 1 + (busy > 0 ? room : 0);
    return free_slot && n < h->seq_max && busy + want <= h->e->max_ctx();
}

// chunked group. Codes: -1 no free slot / no room next to the running sequences (retried), -2 bad input or an engine
// error, -3 too long, -4 two prompts in flight already
int qe_start_begin(void* p, const int* ids, int n, int k, unsigned long long affinity, int hint, const void* img,
                   int img_rows, const int* rope3, int delta) {
    (void)k; (void)affinity; (void)rope3; (void)delta;
    Handle* h = (Handle*)p;
    if (!ids || n < 1 || img || img_rows) return -2;
    if (n >= h->seq_max) return -3;
    for (int i = 0; i < n; i++)
        if (ids[i] < 0 || ids[i] >= VOCAB) return -2;
    if (!h->concurrent)
        for (Slot& o : h->s) o.gen = false;
    if (num_prefills(*h) >= 2) return -4;
    int reuse = 0;
    bool snap = false;
    const int i = pick_slot(*h, ids, n, &reuse, &snap);
    if (i < 0) return -1;
    Slot& s = h->s[i];
    try {
        // [srv-gptoss-cache] the slot's old conversation goes to the store first when this prompt gives most of it up
        const auto t0 = std::chrono::steady_clock::now();
        auto ms = [](auto a, auto b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
        const bool arch = (int)s.toks.size() - reuse > 128 && !s.archived;
        if (arch && srv_archive_cheap(*h, i)) srv_archive_slot(*h, i);   // (a bounded copy only: review M4)
        const auto t1 = std::chrono::steady_clock::now();
        // room for the prompt (the held rows that stay count); the reply grows it step by step
        // (only the reused rows count as held: the rest of the slot's old sequence is given up for this prompt)
        if ((int)s.toks.size() > reuse) s.toks.resize(reuse);
        s.prov_cut(reuse);
        if (!srv_room(*h, i, n + 1)) return -1;
        // the longest prefix any source holds (own slot, another slot, the store), then the rows and rings from it
        const auto t2 = std::chrono::steady_clock::now();
        const Source src = srv_source(*h, i, !snap, ids, n);
        srv_apply(*h, i, src, ids);
        if (cache_log()) {
            const auto t3 = std::chrono::steady_clock::now();
            fprintf(stderr, "[gptoss cache] slot %d prompt %d: archive %s%.1f ms, room %.1f ms, source c %d (ring %s, rows %s) %.1f ms\n", i, n,
                    arch ? "" : "none ", ms(t0, t1), ms(t1, t2), src.c,
                    src.ring_slot >= 0 ? (src.ring_slot == i ? "own" : "slot") : src.ring_blob ? "blob" : "-",
                    src.rows_slot == i ? "own" : src.rows_slot >= 0 ? "slot" : src.rows_entry >= 0 ? "store" : "-", ms(t2, t3));
        }
        if (src.ring_blob == &s.snap) h->snap_used++;
        reuse = src.c;
        s.archived = false;
        s.pf_hint = hint >= 256 && hint > reuse && hint < n ? hint : 0;   // (smaller system parts: not worth a ring copy)
        s.next = -1;
        s.pf.assign(ids, ids + n);
        s.pf_done = reuse;
        s.pf_reused = reuse;
        s.pf_started = false;
        s.pf_first = -1;
        s.sp = h->pending;
        s.pen_synced = false;
        s.pen_from = -1;
        s.limit = -1;
        s.fault.clear();
        s.constrained = false;   // (qe_constrain per request, after start_begin)
        s.lk_clear();
        s.use = ++h->tick;
        h->newest = i;
        return i;
    } catch (const std::exception& ex) {
        fprintf(stderr, "%s\n", ex.what());
        srv_fault(*h, i, ex.what());
        return -2;
    }
}
int qe_prefill_more_slot(void* p, int slot, int budget) {
    Handle* h = (Handle*)p;
    if (slot < 0 || slot >= (int)h->s.size() || h->s[slot].pf_done < 0) return 1;
    try {
        return prefill_some(*h, slot, budget) ? 1 : 0;
    } catch (const std::exception& ex) {
        fprintf(stderr, "%s\n", ex.what());
        srv_fault(*h, slot, ex.what());
        return 1;   // (start_finish reports the fault)
    }
}
int qe_prefill_more(void* p, int budget) { return qe_prefill_more_slot(p, ((Handle*)p)->newest, budget); }
// the first token; -2: the prompt failed (qe_slot_fault)
int qe_start_finish_slot(void* p, int slot) {
    Handle* h = (Handle*)p;
    if (slot < 0 || slot >= (int)h->s.size()) return -2;
    Slot& s = h->s[slot];
    h->last_slot = slot;
    if (s.pf_done < 0) return -2;   // failed (fault kept) or never begun
    try {
        while (!prefill_some(*h, slot, 0)) {}
    } catch (const std::exception& ex) {
        fprintf(stderr, "%s\n", ex.what());
        srv_fault(*h, slot, ex.what());
        return -2;
    }
    h->reused = s.pf_reused;
    s.next = s.pf_first;
    s.pen_from = (int)s.toks.size();   // [gptoss-rel3-pentrim] (toks = the prompt's rows; next = the reply's first token)
    srv_pen_count(s, s.next);   // [gptoss-rel2-penalty] the reply's first token
    s.pf.clear();
    s.pf_done = -1;
    s.gen = true;
    s.use = ++h->tick;
    if (h->newest == slot) h->newest = -1;
    h->cur = slot;
    return s.next;
}
int qe_start_finish(void* p) {
    Handle* h = (Handle*)p;
    return qe_start_finish_slot(p, h->newest);
}
int qe_num_prefills(void* p) { return num_prefills(*(Handle*)p); }
int qe_prefill_pos(void* p, int slot) {
    Handle* h = (Handle*)p;
    return slot >= 0 && slot < (int)h->s.size() ? h->s[slot].pf_done : -1;
}

// tier 0: the whole prompt in one call
int qe_start3(void* p, const int* ids, int n, int k, unsigned long long affinity, int hint) {
    const int slot = qe_start_begin(p, ids, n, k, affinity, hint, nullptr, 0, nullptr, 0);
    if (slot < 0) return slot;
    return qe_start_finish_slot(p, slot);
}

// one decode step of slot i: 1 token into out, 0 = nothing (not generating, limit reached, no room, engine error)
// GPTOSS_LOOKUP: rows per step of the prompt-lookup drafts on greedy replies (default 9, the measured knee; 0 / 1: off)
static int lookup_rows() {
    static const int r = [] {
        const char* v = getenv("GPTOSS_LOOKUP");
        const int x = v && *v ? atoi(v) : 9;
        return x < 1 ? 1 : x > 16 ? 16 : x;
    }();
    return r;
}
// GPTOSS_LOOKUP_MIN: the shortest context match that drafts (default 8 tokens; the index key is 6)
static int lookup_min() {
    static const int r = [] {
        const char* v = getenv("GPTOSS_LOOKUP_MIN");
        const int x = v && *v ? atoi(v) : 8;
        return x < 6 ? 6 : x > 64 ? 64 : x;
    }();
    return r;
}
static bool is_eos(int t) { return t == 200002 || t == 199999 || t == 200012; }

// prompt-lookup drafts of a greedy slot (gptoss_main.cpp's --lookup, ids = plain greedy): history = toks + next; a
// 6-token context seen before proposes what followed it (8 tokens, 16 after a 24-token match), at most `maxd`; the
// gate: after 2 drafted steps in a row with nothing accepted, no drafts for 4, 8, .. 64 steps (text with nothing to
// copy pays no multi-row steps); one accepted draft resets it. Returns the number of drafts written to d.
// the 6 tokens before position p of toks + next, hashed (the index key)
static unsigned long long lk_key(const Slot& s, int p) {
    unsigned long long x = 1469598103934665603ull;
    for (int k = p - 6; k < p; k++) x = (x ^ (unsigned long long)(k < (int)s.toks.size() ? s.toks[k] : s.next)) * 1099511628211ull;
    return x;
}
// index the positions [lk_known, upto) (each: the latest position after its 6-token context)
static void lk_extend(Slot& s, int upto) {
    for (; s.lk_known < upto; s.lk_known++) s.lk[lk_key(s, s.lk_known)] = s.lk_known;
}
static int lookup_propose(Handle& h, Slot& s, int* d, int maxd) {
    const int n0 = (int)s.toks.size() + 1;   // hist[n0 - 1] = s.next
    auto hist = [&](int k) { return k < (int)s.toks.size() ? s.toks[k] : s.next; };
    auto key = [&](int p) { return lk_key(s, p); };
    int nd = 0;
    if (n0 >= 6) {
        if (s.lk_known > n0) s.lk_clear();
        lk_extend(s, n0);
    }
    if (s.lk_skip > 0) {
        s.lk_skip--;
    } else if (n0 >= 6 && maxd > 0) {
        auto it = s.lk.find(key(n0));
        if (it != s.lk.end()) {
            const int j = it->second;
            bool same = true;
            for (int k = 1; k <= 6 && same; k++) same = hist(j - k) == hist(n0 - k);
            if (same) {
                int m = 6;
                while (m < 64 && j - m - 1 >= 0 && hist(j - m - 1) == hist(n0 - m - 1)) m++;
                int want = m < lookup_min() ? 0 : 1;
                if (want) want = std::min({m >= 24 ? 16 : 8, lookup_rows() - 1, maxd});
                if (s.limit > 0) want = std::min<long>(want, s.limit - 1);
                want = std::min(want, h.seq_max - n0 - 1);
                for (; nd < want && j + nd < n0; nd++) d[nd] = hist(j + nd);
            }
        }
    }
    return nd;
}
// after a step of a greedy slot over rows = {next, drafts[0, nd)} with results res: the accepted drafts and the step's
// own token (an end token ends them) go to out; the slot's sequence advances; returns the count
static int lookup_accept(Handle& h, Slot& s, const int* rows, const int* res, int nd, int* out) {
    int acc = 0;
    while (acc < nd && rows[1 + acc] == res[acc] && !is_eos(res[acc])) acc++;
    if (s.limit > 0) s.limit -= std::min<long>(s.limit, acc + 1);
    for (int k = 0; k <= acc; k++) {
        s.toks.push_back(rows[k]);
        out[k] = res[k];
    }
    s.next = res[acc];
    s.use = ++h.tick;
    if (nd > 0) {
        if (acc == 0) {
            if (++s.lk_miss >= 2) s.lk_skip = std::min(64, 4 << std::min(4, s.lk_miss - 2));
        } else {
            s.lk_miss = 0;
        }
        h.lk_steps++;
        h.lk_rows += nd + 1;
        h.lk_tokens += acc + 1;
    }
    return acc + 1;
}
// one greedy step of the bound slot i with prompt-lookup drafts, verified in one multi-row step
static int step_lookup(Handle& h, int i, int* out) {
    Slot& s = h.s[i];
    int rows[16], res[16];
    rows[0] = s.next;
    int nd = lookup_propose(h, s, rows + 1, 15);
    if (nd > 0 && (int)s.toks.size() + nd + 1 > s.cap && !srv_room(h, i, (int)s.toks.size() + nd + 1)) nd = 0;
    srv_bind(h, i);   // (srv_room may have moved the rows)
    srv_arm(h, s.sp, false, false);
    const int before = h.e->pos();
    h.e->decode_rows(rows, nd + 1, res);
    s.hi = std::max(s.hi, before + nd + 1);   // (rejected drafts wrote rings past the accepted end: review H5)
    int acc = 0;
    while (acc < nd && rows[1 + acc] == res[acc] && !is_eos(res[acc])) acc++;
    if (acc < nd) h.e->truncate(before + acc + 1);
    return lookup_accept(h, s, rows, res, nd, out);
}

static int step_slot(Handle& h, int i, int* out) {
    if (i < 0 || i >= (int)h.s.size()) return 0;
    Slot& s = h.s[i];
    if (!s.gen || s.next < 0 || s.limit == 0) return 0;
    try {
        if ((int)s.toks.size() + 1 > s.cap && !srv_room(h, i, (int)s.toks.size() + 1)) {
            srv_full(h, i, (int)s.toks.size() + 1);
            return 0;
        }
        srv_bind(h, i);
        if (!s.constrained && s.sp.temp <= 0.f && lookup_rows() > 1) return step_lookup(h, i, out);
        srv_arm(h, s.sp, false, s.constrained);
        srv_arm_pen(h, i);
        const int t = srv_pick_step(h, s, h.e->decode(s.next));
        srv_pen_count(s, t);
        s.hi = std::max(s.hi, (int)s.toks.size() + 1);
        if (t < 0) throw std::runtime_error("structured output: the mask allows no token");
        if (s.limit > 0) s.limit--;
        s.toks.push_back(s.next);
        s.next = t;
        s.use = ++h.tick;
        out[0] = t;
        return 1;
    } catch (const std::exception& ex) {
        fprintf(stderr, "%s\n", ex.what());
        srv_fault(h, i, ex.what());
        return 0;
    }
}
int qe_spec_step(void* p, int* out) {
    Handle* h = (Handle*)p;
    return step_slot(*h, h->cur, out);
}
// batch group: one step of each slot; out[i * 64] gets cnt[i] tokens of slots[i]. Slots that are greedy or sampled
// with a plain temperature step together in one engine step (Engine::step_rows: per-row ring set and rows, a row's
// result = alone), greedy ones with their lookup drafts (16 rows per step at most: one per slot first, the drafts in
// the rows left); constrained and top_k / top_p / min_p slots step alone after it (GPTOSS_BATCH=0: all alone)
static bool batch_on() {
    static const bool on = [] {
        const char* v = getenv("GPTOSS_BATCH");
        return !(v && *v && atoi(v) == 0);
    }();
    return on;
}
// one engine step over the slots slots[joint[..]] (>= 2): rows, drafts, results; every listed index gets its count
// in cnt (0 = failed or no room)
static void step_joint(Handle& h, const int* slots, const std::vector<int>& joint, int* out, int* cnt) {
    std::vector<int> ok;   // the slots that step (indices into slots[])
    int dr[16][16], nd[16] = {}, nr = 0;
    bool sampled = false;
    try {   // (an engine error in a room copy faults the joint slots, never escapes into the server: review M3)
        for (int i : joint) {   // room for one row each (a slot without room fails, as in step_slot)
            Slot& s = h.s[slots[i]];
            if ((int)s.toks.size() + 1 > s.cap && !srv_room(h, slots[i], (int)s.toks.size() + 1))
                srv_full(h, slots[i], (int)s.toks.size() + 1);
            else
                ok.push_back(i);
        }
        nr = (int)ok.size();
        for (int q = 0; q < (int)ok.size(); q++) {   // drafts of the greedy slots into the rows left
            Slot& s = h.s[slots[ok[q]]];
            if (s.sp.temp > 0.f || lookup_rows() <= 1) continue;
            int d = lookup_propose(h, s, dr[q], std::min(16 - nr, lookup_rows() - 1));
            if (d > 0 && (int)s.toks.size() + d + 1 > s.cap && !srv_room(h, slots[ok[q]], (int)s.toks.size() + d + 1)) d = 0;
            nd[q] = d;
            nr += d;
        }
        // a later slot's room may have moved or resized an earlier one: every row inside its slot's interval, or its
        // drafts go, or it fails [review H1]
        std::vector<int> keep;
        int nd2[16] = {};
        nr = 0;
        for (int q = 0; q < (int)ok.size(); q++) {
            const int k = slots[ok[q]];
            const Slot& s = h.s[k];
            int d = nd[q];
            if ((int)s.toks.size() + d + 1 > s.cap) d = 0;
            if ((int)s.toks.size() + 1 > s.cap) {
                srv_full(h, k, (int)s.toks.size() + 1);
                continue;
            }
            if ((int)keep.size() != q) std::copy(dr[q], dr[q] + d, dr[keep.size()]);
            nd2[keep.size()] = d;
            keep.push_back(ok[q]);
            nr += 1 + d;
        }
        ok.swap(keep);
        std::copy(nd2, nd2 + 16, nd);
    } catch (const std::exception& ex) {
        fprintf(stderr, "%s\n", ex.what());
        for (int i : joint) srv_fault(h, slots[i], ex.what());
        return;
    }
    const int ns = (int)ok.size();
    if (ns == 0) return;
    StepRow rows[16];
    RowSamp rs[16];
    int tok[16], res[16], first[16], at = 0;
    for (int q = 0; q < ns; q++) {
        const int k = slots[ok[q]];
        const Slot& s = h.s[k];
        sampled = sampled || s.sp.temp > 0.f;
        const int p0 = (int)s.toks.size();
        first[q] = at;
        for (int d = 0; d <= nd[q]; d++, at++) {
            tok[at] = d == 0 ? s.next : dr[q][d - 1];
            rows[at] = StepRow{k, s.base, p0 + d, tok[at]};
            rs[at] = RowSamp{d == 0 && !s.sp.one() ? s.sp.temp : 0.f, s.sp.seed};
        }
    }
    try {   // [gptoss-rel2-penalty] the penalised slots' rows on the cards, and each one's newest token with its row
        for (int q = 0; q < ns; q++) {
            const int k = slots[ok[q]];
            Slot& s = h.s[k];
            if (!s.sp.pen()) continue;
            srv_pen_sync(h, k);
            rs[first[q]].pen = true;
            rs[first[q]].pen_id = s.next;
            rs[first[q]].pen_val = s.sp.pen_val(s.next);
        }
    } catch (const std::exception& ex) {
        fprintf(stderr, "%s\n", ex.what());
        for (int q = 0; q < ns; q++) srv_fault(h, slots[ok[q]], ex.what());
        return;
    }
    try {
        h.e->set_dump(Dump{});
        struct Soft {   // [gptoss-rel1-rowfault] a row with non-finite logits fails its own slot only (below)
            Engine* e;
            explicit Soft(Engine* x) : e(x) { e->set_soft_rows(true); }
            ~Soft() { e->set_soft_rows(false); }
        } soft{h.e};
        if (sampled)
            h.e->step_rows_sampled(rows, rs, nr, res);
        else
            h.e->step_rows(rows, nr, res);
    } catch (const std::exception& ex) {
        fprintf(stderr, "%s\n", ex.what());
        for (int q = 0; q < ns; q++) srv_fault(h, slots[ok[q]], ex.what());
        return;
    }
    {   // [gptoss-rel1-rowfault] the slots with a bad row are faulted and leave the step
        std::vector<int> ok2;
        int f2[16], n2[16], m2 = 0;
        for (int q = 0; q < ns; q++) {
            bool bad = false;
            for (int d = 0; d <= nd[q]; d++) bad = bad || res[first[q] + d] < 0;
            if (bad) {
                fprintf(stderr, "gptoss: slot %d: the logits are not finite; this request fails, the others go on\n", slots[ok[q]]);
                srv_fault(h, slots[ok[q]], "the logits are not finite (this sequence's step)");
                continue;
            }
            f2[m2] = first[q];
            n2[m2++] = nd[q];
            ok2.push_back(ok[q]);
        }
        if (m2 != ns) {
            ok.swap(ok2);
            std::copy(f2, f2 + m2, first);
            std::copy(n2, n2 + m2, nd);
        }
    }
    const int ns2 = (int)ok.size();
    h.bound = -1;   // (the engine's own position belongs to no slot now; srv_bind sets it again)
    h.batch_steps++;
    h.batch_rows += nr;
    for (int q = 0; q < ns2; q++) {
        const int i = ok[q];
        Slot& s = h.s[slots[i]];
        s.hi = std::max(s.hi, (int)s.toks.size() + 1 + nd[q]);   // (its draft rows wrote rings too: review H5)
        if (s.sp.temp > 0.f) {   // the cards' draw (plain temperature)
            if (s.limit > 0) s.limit--;
            s.toks.push_back(s.next);
            s.next = res[first[q]];
            srv_pen_count(s, s.next);
            s.use = ++h.tick;
            out[i * 64] = s.next;
            cnt[i] = 1;
        } else {
            cnt[i] = lookup_accept(h, s, tok + first[q], res + first[q], nd[q], out + i * 64);
        }
    }
}
void qe_step_batch(void* p, const int* slots, int n, int* out, int* cnt) {
    Handle* h = (Handle*)p;
    std::vector<int> joint;   // indices into slots[] that step together
    std::vector<bool> taken(n, false);
    if (batch_on())
        for (int i = 0; i < n && (int)joint.size() < 16; i++) {
            const int k = slots[i];
            if (k < 0 || k >= (int)h->s.size()) continue;
            const Slot& s = h->s[k];
            if (!s.gen || s.next < 0 || s.limit == 0 || s.constrained || (s.sp.temp > 0.f && s.sp.filtered())) continue;
            bool dup = false;
            for (int j : joint) dup = dup || slots[j] == k;
            if (!dup) joint.push_back(i);
        }
    for (int i = 0; i < n; i++) {   // a slot listed twice steps once (review L4)
        cnt[i] = 0;
        for (int j = 0; j < i && !taken[i]; j++) taken[i] = slots[j] == slots[i];
    }
    if (joint.size() >= 2) {
        step_joint(*h, slots, joint, out, cnt);
        for (int i : joint) taken[i] = true;
    }
    for (int i = 0; i < n; i++)
        if (!taken[i]) cnt[i] = step_slot(*h, slots[i], out + i * 64);
}
void qe_resume(void* p, int slot) { ((Handle*)p)->cur = slot; }
// 1: starts do not release the other slots (the server); 0 (default): a start ends the generating ones (tier 0 callers)
void qe_set_concurrent(void* p, int on) { ((Handle*)p)->concurrent = on != 0; }
int qe_last_reused(void* p) { return ((Handle*)p)->reused; }
int qe_last_slot(void* p) { return ((Handle*)p)->last_slot; }
void qe_release(void* p, int slot) {
    Handle* h = (Handle*)p;
    if (slot < 0 || slot >= (int)h->s.size()) return;
    Slot& s = h->s[slot];
    s.gen = false;
    s.constrained = false;
    if (s.pf_done >= 0) {   // a prompt dropped mid-prefill: the rows done so far stay (s.toks)
        s.pf.clear();
        s.pf_done = -1;
        if (h->newest == slot) h->newest = -1;
    }
    s.use = ++h->tick;
}
int qe_vocab(void* p) { (void)p; return VOCAB; }

}  // extern "C"
