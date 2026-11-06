// Flash-Next eviction order and archive-copy rules (src/flash/kv_evict.h), no GPU: a model of the archive whose rows carry
// a hash of the token prefix that produced them. Random start / step / sync / refused write / eviction / fork / rewind /
// cut sequences; every restore must rebuild the exact rows and state of the prompt's prefix.
//   g++ -std=c++20 -O1 -I src tests/kv_evict_unit.cpp -o /tmp/kv_evict_unit && /tmp/kv_evict_unit
#include <cstdint>
#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <list>
#include <memory>
#include <random>
#include <vector>
#include "flash/kv_evict.h"

static int fails = 0;
#define EXPECT(c) do { if (!(c)) { fails++; fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #c); } } while (0)

static uint64_t mix(uint64_t h, int t) { return (h ^ (uint64_t)t) * 0x100000001b3ULL + 7; }
// rows[i] = what position i holds when the tokens are v: the hash of v[0..i]
static std::vector<uint64_t> rows_of(const std::vector<int>& v) {
    std::vector<uint64_t> r(v.size());
    uint64_t h = 14695981039346656037ULL;
    for (size_t i = 0; i < v.size(); i++) r[i] = h = mix(h, v[i]);
    return r;
}

struct Seg { int t0 = 0, t1 = 0; std::vector<uint64_t> rows; };
struct State { int pos = 0; uint64_t h = 0; int kind = 3; };
struct Conv {
    std::vector<int> toks;
    std::vector<std::shared_ptr<Seg>> segs;
    int kv_n = 0;
    std::shared_ptr<State> end;
    std::vector<std::shared_ptr<State>> ck;
    int prio = 1;
    bool res = false;
    std::vector<int> hist;              // resident: the tokens in its slot
    unsigned long long used = 0;
};
static std::shared_ptr<State> state_at(const std::vector<int>& hist) {
    auto s = std::make_shared<State>();
    s->pos = hist.size();
    s->h = hist.empty() ? 0 : rows_of(hist).back();
    return s;
}
// as FE::sync: write [kv_n, min(hist, upto)); ok = false: the write is refused (the archive has no room)
static bool sync_conv(Conv& c, int upto, bool end, bool ok) {
    const int hn = c.hist.size();
    if (c.kv_n > hn) kve::cut(c, hn);
    const int t1 = std::min(hn, upto);
    if (t1 > c.kv_n) {
        if (!ok) return false;
        auto g = std::make_shared<Seg>();
        g->t0 = c.kv_n;
        g->t1 = t1;
        const auto r = rows_of(c.hist);
        g->rows.assign(r.begin() + g->t0, r.begin() + t1);
        c.segs.push_back(g);
        c.kv_n = t1;
    }
    if (t1 < hn || !end) return true;
    c.toks = c.hist;
    c.end = nullptr;
    if (!ok) return false;
    c.end = state_at(c.hist);
    return true;
}
// a restore of ids[0, L) from c's copy with state st: the exact rows and state, or a failure
static long long restores = 0, partial_seg = 0;
static void check_restore(const Conv& c, const std::vector<int>& ids, int L, const State& st, const char* how) {
    restores++;
    const std::vector<int> pre(ids.begin(), ids.begin() + L);
    const auto want = rows_of(pre);
    std::vector<uint64_t> got(L, 0);
    for (const auto& g : c.segs) {
        if (g->t0 >= L) continue;
        if (g->t1 > c.kv_n) partial_seg++;
        for (int p = g->t0; p < std::min(g->t1, L); p++) got[p] = g->rows[p - g->t0];
    }
    if (got != want || st.pos != L || st.h != (want.empty() ? 0 : want.back())) {
        fails++;
        fprintf(stderr, "FAIL: wrong restore (%s) at L=%d, kv_n=%d, %zu segments\n", how, L, c.kv_n, c.segs.size());
    }
}

static void fixed_cases() {
    // review 10 H1: a fork at 60 of a parent archived [0, 100): the shared segment reaches past the branch point
    Conv par;
    par.hist.resize(100);
    for (int i = 0; i < 100; i++) par.hist[i] = 1;
    sync_conv(par, 1 << 30, true, true);
    std::vector<int> ids(par.hist.begin(), par.hist.begin() + 60);
    Conv d;
    kve::fork_from(d, par, ids.data(), 60);
    EXPECT(d.kv_n == 60);
    EXPECT(d.segs.size() == 1 && d.segs[0]->t1 == 100);
    // a whole drop leaves nothing to claim rows, and the next sync writes all of it
    d.hist = ids;
    d.hist.insert(d.hist.end(), 30, 2);
    sync_conv(d, 1 << 30, true, true);
    EXPECT(d.segs.back()->t0 == 60 && d.kv_n == 90);
    kve::drop_copy(d);
    EXPECT(d.kv_n == 0 && d.segs.empty() && !d.end && d.ck.empty());
    EXPECT(!kve::end_usable(d, 90));
    // rows archived past a cut are not reused when the sequence regrows with other tokens
    Conv c;
    c.hist.assign(50, 3);
    sync_conv(c, 1 << 30, false, true);
    kve::cut(c, 40);
    EXPECT(c.kv_n == 40);
    c.hist.resize(40);
    c.hist.insert(c.hist.end(), 30, 4);
    sync_conv(c, 1 << 30, true, true);
    EXPECT(c.segs.back()->t0 == 40);
    State st = *c.end;
    check_restore(c, c.hist, 70, st, "cut");
    // the checkpoint cap: the end of the main's turn survives a fork of 8 turns in place, at the real cap of 6 and at 2
    for (int keep : {6, 2}) {
        Conv m;
        auto st = [](int pos, int kind) { auto x = std::make_shared<State>(); x->pos = pos; x->kind = kind; return x; };
        m.prio = 2;
        kve::add_ck(m, st(100, 2), keep);
        kve::add_ck(m, st(101, 4), keep);  // the live end before the next request extends it, its latest turn was main
        EXPECT(m.ck.back()->kind == 6);
        m.prio = 1;                        // the fork's turns
        for (int t = 0; t < 8; t++) {
            kve::add_ck(m, st(200 + 10 * t, 2), keep);
            kve::add_ck(m, st(201 + 10 * t, 4), keep);
        }
        bool has = false;
        for (auto& x : m.ck) has |= x->pos == 101 && x->kind == 6;
        EXPECT(has);
        EXPECT((int)m.ck.size() == keep);
        m.prio = 2;                        // a later main turn: the newer end takes over
        kve::add_ck(m, st(300, 4), keep);
        int n6 = 0;
        for (auto& x : m.ck) n6 += x->kind == 6;
        EXPECT(n6 == 1 && m.ck.back()->pos == 300);
    }
    // the order: least recently used; an active main after every other, also after a newer one; stale: an ordinary one
    using kve::goes_first;
    using kve::Rank;
    EXPECT(goes_first(Rank{false, 1}, Rank{false, 2}));
    EXPECT(goes_first(Rank{false, 9}, Rank{true, 1}));
    EXPECT(!goes_first(Rank{true, 1}, Rank{false, 9}));
    EXPECT(goes_first(Rank{true, 1}, Rank{true, 2}));
    EXPECT(kve::active_main(true, 0.0, 1000.0, 300.0));
    EXPECT(!kve::active_main(true, 0.0, 301.0 * 1000.0, 300.0));
    EXPECT(!kve::active_main(false, 0.0, 1000.0, 300.0));
}

// ---- room in the archive (kve::make_room, freeable, copy_victim), the rules FE::make_room runs on: a small model
namespace room {
struct B { size_t n; };
struct C {
    int id = 0;
    std::vector<std::shared_ptr<B>> segs;
    std::shared_ptr<B> end;
    std::vector<std::shared_ptr<B>> ck;
    kve::Rank r;
    bool stays = false;  // the writer, a start's source, a paused slot's copy
};
struct M {
    std::list<C> convs;
    std::vector<int> gone;
    C& add(int id, size_t bytes, unsigned long long used, bool main = false, bool stays = false) {
        convs.emplace_back();
        C& c = convs.back();
        c.id = id;
        c.r = kve::Rank{main, used};
        c.stays = stays;
        if (bytes) c.segs.push_back(std::make_shared<B>(B{bytes}));
        return c;
    }
    size_t used() const {
        std::vector<const B*> u;
        size_t n = 0;
        auto one = [&](const std::shared_ptr<B>& b) { if (std::find(u.begin(), u.end(), b.get()) == u.end()) { u.push_back(b.get()); n += b->n; } };
        for (auto& c : convs) {
            for (auto& g : c.segs) one(g);
            if (c.end) one(c.end);
            for (auto& s : c.ck) one(s);
        }
        return n;
    }
    bool room(size_t need, size_t limit, std::vector<const void*> pinned = {}, const kve::Rank* older_than = nullptr) {
        auto may_go = [&](const C& c) { return !c.stays && (!older_than || kve::goes_first(c.r, *older_than)); };
        return kve::make_room(need, limit, [&] { return used(); },
            [&] { return kve::freeable(convs, may_go, pinned, [](const B& b) { return b.n; }); },
            [&]() -> C* {
                C* v = kve::copy_victim(convs, [&](const C& c) { return c.stays; }, [](const C& c) { return c.r; });
                return v && may_go(*v) ? v : nullptr;
            },
            [&](C* v) {
                gone.push_back(v->id);
                for (auto it = convs.begin(); it != convs.end(); ++it) if (&*it == v) { convs.erase(it); break; }
            });
    }
};
static void test_main_moved() {
    // the main's turn diverges near the end: the old branch stops being a main, ranks by LRU among the others (review M6)
    EXPECT(kve::main_moved(59000, 60000, false));
    EXPECT(kve::main_moved(30000, 60000, false));
    EXPECT(!kve::main_moved(29999, 60000, false));  // a second session on the same system prompt keeps its own main
    EXPECT(!kve::main_moved(59000, 60000, true));   // still generating
    const kve::Rank dead{false, 5}, live_sub{false, 8}, old_main_kept{true, 5};
    EXPECT(kve::goes_first(dead, live_sub));
    EXPECT(!kve::goes_first(old_main_kept, live_sub));
}
static void test_reclaim_order() {
    using S = kve::IdleSlot;
    using V = std::vector<int>;
    auto m = [](int s, int cap, double idle, unsigned long long u) { return S{s, cap, idle, kve::Rank{true, u}}; };
    auto o = [](int s, int cap, double idle, unsigned long long u) { return S{s, cap, idle, kve::Rank{false, u}}; };
    const int np = 256;
    // review M7: 8 mains of 13 periods (42 % of the pool), each below 1/16 (16 periods): the oldest go until the rest hold less
    std::vector<S> eight;
    for (int i = 0; i < 8; i++) eight.push_back(m(i, 13, 60, 100 - i));
    EXPECT(kve::reclaim_order(eight, true, 2, 30, np) == V({7, 6, 5, 4, 3, 2, 1}));  // 13 < 16 stays
    EXPECT(kve::reclaim_order(eight, false, 30, 30, np).empty());                    // no pressure: mains stay
    // one small main stays, one large main goes: as before
    EXPECT(kve::reclaim_order({m(0, 15, 60, 1)}, true, 2, 30, np).empty());
    EXPECT(kve::reclaim_order({m(0, 16, 60, 1)}, true, 2, 30, np) == V({0}));
    // two large mains both go; a main idle too shortly neither goes nor counts
    EXPECT(kve::reclaim_order({m(0, 87, 60, 2), m(1, 87, 60, 1)}, true, 2, 30, np) == V({1, 0}));
    EXPECT(kve::reclaim_order({m(0, 87, 10, 1), m(1, 8, 60, 2)}, true, 2, 30, np).empty());
    EXPECT(kve::reclaim_order({m(0, 87, 60, 1)}, true, 2, 0, np).empty());  // FLASH_EC_RECLAIM_MAIN_S=0: never
    // ordinary ones first, oldest first, by their own time
    EXPECT(kve::reclaim_order({o(0, 5, 60, 9), m(1, 87, 60, 1), o(2, 5, 60, 3), o(3, 5, 1, 2)}, true, 2, 30, np) == V({2, 0, 1}));
    EXPECT(kve::reclaim_order({o(0, 5, 20, 9), o(1, 5, 31, 3)}, false, 30, 30, np) == V({1}));
}
// one pass of FE::reclaim_idle on the model: each one in the order writes under ROOM_OLDER (without pressure: nothing may go) and
// leaves; a refused one follows kve::RefusedWrite. out: who left with a copy, who left lost.
struct Pass { std::vector<int> archived, lost; };
static Pass reclaim_pass(M& m, const std::vector<kve::IdleSlot>& idle, const std::vector<size_t>& need, size_t limit, bool pressure,
                         std::vector<bool>& resident) {
    Pass out;
    kve::RefusedWrite refused;
    const std::vector<int> ord = kve::reclaim_order(idle, pressure, 2, 30, 256);
    for (int i : ord) {
        C* c = nullptr;
        for (auto& x : m.convs) if (x.id == i) c = &x;
        if (!c) continue;  // (gone in an earlier pass)
        c->stays = true;  // the writer
        const kve::Rank none{false, 0};
        const bool ok = m.room(need[i], limit, {}, pressure ? &c->r : &none);
        c->stays = false;
        if (ok) {
            c->segs.push_back(std::make_shared<B>(B{need[i]}));
            out.archived.push_back(i);
            resident[i] = false;
        } else if (refused.leaves(pressure, c->r.main)) {
            out.lost.push_back(i);
            resident[i] = false;
            for (auto it = m.convs.begin(); it != m.convs.end(); ++it) if (&*it == c) { m.convs.erase(it); break; }
        }
    }
    return out;
}
static void test_refused_write() {
    using S = kve::IdleSlot;
    using V = std::vector<int>;
    auto ord = [](int s, unsigned long long u) { return S{s, 20, 60, kve::Rank{false, u}}; };
    auto mn = [](int s, unsigned long long u) { return S{s, 20, 60, kve::Rank{true, u}}; };
    {  // an archive full of newer copies: the oldest goes, one per pass; newer copies, the active main and a generating one are untouched
        M m;
        m.add(1, 0, 1);            // idle, oldest
        m.add(2, 0, 2);            // idle
        m.add(3, 0, 3, true);      // idle active main, larger than the archive (a smaller one may take ordinary copies, and its write fits)
        m.add(4, 0, 0);            // generating, the oldest of all: not an idle slot, so not in the list
        m.add(8, 2, 8);
        m.add(9, 2, 9);
        const std::vector<S> idle = {ord(1, 1), ord(2, 2), mn(3, 3)};
        const std::vector<size_t> need = {0, 1, 1, 5};
        std::vector<bool> res(4, true);
        Pass p = reclaim_pass(m, idle, need, 4, true, res);
        EXPECT(p.lost == V({1}) && p.archived.empty() && m.gone.empty());
        EXPECT(res[2] && res[3] && m.convs.size() == 5);
        p = reclaim_pass(m, {ord(2, 2), mn(3, 3)}, need, 4, true, res);
        EXPECT(p.lost == V({2}) && m.gone.empty());
        p = reclaim_pass(m, {mn(3, 3)}, need, 4, true, res);  // only the active main left: it stays, every time
        EXPECT(p.lost.empty() && p.archived.empty() && res[3] && m.gone.empty() && m.convs.size() == 4);
    }
    {  // no pressure: everyone stays
        M m;
        m.add(1, 0, 1);
        m.add(9, 4, 9);
        std::vector<bool> res(2, true);
        const Pass p = reclaim_pass(m, {ord(1, 1)}, {0, 1}, 4, false, res);
        EXPECT(p.lost.empty() && p.archived.empty() && res[1] && m.gone.empty());
    }
    {  // bigger than the whole archive, which is empty: an ordinary one goes, an active main stays
        M m;
        m.add(1, 0, 1);
        m.add(2, 0, 2, true);
        std::vector<bool> res(3, true);
        const Pass p = reclaim_pass(m, {ord(1, 1), mn(2, 2)}, {0, 7, 7}, 6, true, res);
        EXPECT(p.lost == V({1}) && res[2]);
    }
    {  // an older archived copy too small to make room stays: it would not save the write
        M m;
        m.add(1, 0, 5);
        m.add(7, 1, 1);
        m.add(9, 3, 9);
        std::vector<bool> res(2, true);
        const Pass p = reclaim_pass(m, {ord(1, 5)}, {0, 3}, 4, true, res);
        EXPECT(p.lost == V({1}) && m.gone.empty() && m.convs.size() == 2);
    }
    {  // older copies that do make room: nobody is lost beyond them, the write goes through as before
        M m;
        m.add(1, 0, 5);
        m.add(7, 2, 1);
        m.add(9, 2, 9);
        std::vector<bool> res(2, true);
        const Pass p = reclaim_pass(m, {ord(1, 5)}, {0, 2}, 4, true, res);
        EXPECT(p.archived == V({1}) && p.lost.empty() && m.gone == V({7}));
    }
    {  // the first one is written, the next is refused: that one is now the first, it goes
        M m;
        m.add(1, 0, 1);
        m.add(2, 0, 2);
        std::vector<bool> res(3, true);
        const Pass p = reclaim_pass(m, {ord(1, 1), ord(2, 2)}, {0, 2, 5}, 4, true, res);
        EXPECT(p.archived == V({1}) && p.lost == V({2}) && m.gone.empty());
    }
}
struct RC { std::vector<int> toks; std::shared_ptr<B> end; std::vector<std::shared_ptr<B>> ck; };
static void test_reachable() {
    RC c;
    c.ck.push_back(std::make_shared<B>(B{1}));
    EXPECT(!kve::reachable(c));  // a refused first write: checkpoints, no tokens to compare a prompt with
    c.toks = {1, 2};
    EXPECT(kve::reachable(c));
    c.ck.clear();
    EXPECT(!kve::reachable(c));
    c.end = std::make_shared<B>(B{1});
    EXPECT(kve::reachable(c));
}
static void test_room() {
    {  // review M5: writer K holds 4 of the 6 GB limit, needs 2.5, the others hold 2: refused, and nobody is lost for it
        M m;
        m.add(1, 4, 9, false, true);
        m.add(2, 1, 1);
        m.add(3, 1, 2);
        EXPECT(!m.room(3, 6) && m.gone.empty());  // 4 + 3 > 6 with both others gone
        EXPECT(m.convs.size() == 3);
    }
    {  // it fits once the older ones are gone: the order holds, and only as many as needed go
        M m;
        m.add(1, 2, 9, false, true);
        m.add(2, 1, 5);
        m.add(3, 1, 1);
        m.add(4, 1, 3);
        EXPECT(m.room(3, 6));
        EXPECT(m.gone == std::vector<int>({3, 4}));
    }
    {  // an active main goes after all the others, and it goes when it must
        M m;
        m.add(1, 2, 1, true);
        m.add(2, 2, 5);
        EXPECT(m.room(3, 4));
        EXPECT(m.gone == std::vector<int>({2, 1}));
    }
    {  // blobs shared with what stays free nothing: a fork of the writer, a pinned prefix
        M m;
        C& k = m.add(1, 4, 9, false, true);
        C& f = m.add(2, 0, 1);
        f.segs.push_back(k.segs[0]);
        C& o = m.add(3, 1, 2);
        const std::vector<const void*> pinned = {o.segs[0].get()};
        EXPECT(!m.room(2, 5, pinned) && m.gone.empty());  // nothing of 2 and 3 can free a byte
        EXPECT(m.room(1, 5) && m.gone == std::vector<int>({2, 3}));  // no pin: it fits with them gone, oldest first
    }
    {  // a blob two evictable conversations share counts once
        M m;
        m.add(9, 2, 9, false, true);
        C& a = m.add(1, 3, 1);
        C& b = m.add(2, 0, 2);
        b.segs.push_back(a.segs[0]);
        EXPECT(!m.room(5, 6) && m.gone.empty());  // 5 used + 5 > 6 + 3 (counted twice it would seem to fit)
    }
    {  // the reclaim's mode: only what goes before the writer; newer ones stay and the write is refused without a loss
        M m;
        kve::Rank w{false, 5};
        m.add(1, 1, 9, false, true);
        m.add(2, 2, 7);
        m.add(3, 2, 8);
        EXPECT(!m.room(2, 5, {}, &w) && m.gone.empty());
        m.add(4, 2, 1);
        EXPECT(m.room(2, 7, {}, &w) && m.gone == std::vector<int>({4}));
        kve::Rank wm{true, 5};  // a main writer may take the ordinary ones
        EXPECT(m.room(2, 5, {}, &wm));
    }
    {  // too big for the whole limit: nothing goes
        M m;
        m.add(2, 1, 1);
        EXPECT(!m.room(7, 6) && m.gone.empty());
    }
}
}  // namespace room

int main() {
    fixed_cases();
    std::mt19937 rng(12345);
    auto R = [&](int n) { return (int)(rng() % (unsigned)n); };
    long long ops = 0, drops = 0, forks = 0, cuts = 0, refused = 0;
    for (int run = 0; run < 400; run++) {
        std::list<Conv> convs;
        unsigned long long tick = 0;
        std::vector<std::vector<int>> prompts(1, std::vector<int>{1, 2, 3});  // earlier prompts: a request repeats or extends one
        const int alpha = 2 + R(3);  // few tokens: prefixes match by chance, too
        auto random_prompt = [&]() {
            std::vector<int> p = prompts[R((int)prompts.size())];
            if (R(3) == 0 && !p.empty()) p.resize(R((int)p.size()) + 1);  // a rewind
            const int more = 1 + R(40);
            for (int i = 0; i < more; i++) p.push_back(R(alpha));
            prompts.push_back(p);
            return p;
        };
        for (int op = 0; op < 300; op++) {
            ops++;
            const int k = R(100);
            auto pick = [&]() -> Conv* {
                if (convs.empty()) return nullptr;
                auto it = convs.begin();
                std::advance(it, R((int)convs.size()));
                return &*it;
            };
            if (k < 30) {  // a request: as best() picks its source, restored like qe_start_begin
                const std::vector<int> ids = random_prompt();
                const int n = ids.size();
                Conv* src = nullptr;
                std::shared_ptr<State> st;
                bool live = false;
                int L = 0;
                for (auto& c : convs) {
                    const auto& t = c.res ? c.hist : c.toks;
                    int M = 0;
                    while (M < (int)t.size() && M < n && t[M] == ids[M]) M++;
                    if (c.res && M == (int)t.size() && M < n && M > L) { src = &c; st = nullptr; live = true; L = M; }
                    if (!c.res && kve::end_usable(c, M) && M < n && M > L) { src = &c; st = c.end; live = false; L = M; }
                    for (auto& s : c.ck)
                        if (kve::ck_usable(c, *s, M, c.res) && s->pos < n && s->pos > L) { src = &c; st = s; live = false; L = s->pos; }
                }
                Conv* target;
                if (src && live) {
                    target = src;  // in place
                } else if (src) {
                    if (!src->res) check_restore(*src, ids, L, *st, "archive");
                    else if (st->pos != L || st->h != rows_of(std::vector<int>(ids.begin(), ids.begin() + L)).back()) {
                        fails++;
                        fprintf(stderr, "FAIL: wrong state from a resident checkpoint at %d\n", L);
                    }
                    if (st == src->end && !src->res) {
                        target = src;  // the whole conversation continues
                        kve::cut(*src, L);  // (a refused write left states / rows of the old branch past L)
                        src->hist.assign(ids.begin(), ids.begin() + L);
                        src->res = true;
                    } else {
                        Conv d;
                        kve::fork_from(d, *src, ids.data(), L);
                        d.hist.assign(ids.begin(), ids.begin() + L);
                        d.res = true;
                        convs.push_back(std::move(d));
                        target = &convs.back();
                        forks++;
                    }
                } else {
                    convs.emplace_back();
                    target = &convs.back();
                    target->res = true;
                }
                // prefill + reply; a state checkpoint on the way
                const int cp = target->hist.size() + R(std::max(1, n - (int)target->hist.size()));
                target->hist.insert(target->hist.end(), ids.begin() + target->hist.size(), ids.begin() + cp);
                if (R(2)) target->ck.push_back(state_at(target->hist));
                target->hist.insert(target->hist.end(), ids.begin() + cp, ids.end());
                target->used = ++tick;
                if (R(4) == 0) {  // a pause writes the rows; then a stop string cuts the reply back, a next turn regrows it
                    const int more = 1 + R(6);
                    for (int i = 0; i < more; i++) target->hist.push_back(R(alpha));
                    sync_conv(*target, 1 << 30, false, R(8) != 0);
                    const int pos = (int)target->hist.size() - 1 - R(more);
                    target->hist.resize(pos);
                    kve::cut(*target, pos);
                    cuts++;
                }
            } else if (k < 50) {  // a full or partial archive write, or a refused one
                if (Conv* c = pick(); c && c->res) {
                    const bool ok = R(5) != 0;
                    refused += !ok;
                    sync_conv(*c, R(2) ? 1 << 30 : R((int)c->hist.size() + 1), R(2), ok);
                }
            } else if (k < 65) {  // the eviction loop: the whole copy goes (a resident one keeps its slot)
                if (Conv* c = pick()) {
                    drops++;
                    if (c->res) kve::drop_copy(*c);
                    else for (auto it = convs.begin(); it != convs.end(); ++it) if (&*it == c) { convs.erase(it); break; }
                }
            } else if (k < 90) {  // a slot leaves VRAM: as FE::evict_slot, the write may be refused
                if (Conv* c = pick(); c && c->res) {
                    const bool ok = R(4) != 0;
                    refused += !ok;
                    sync_conv(*c, 1 << 30, true, ok);
                    c->res = false;
                    if (!c->end && c->ck.empty())
                        for (auto it = convs.begin(); it != convs.end(); ++it) if (&*it == c) { convs.erase(it); break; }
                }
            } else {  // every archived conversation: what it claims is exact, at every usable position
                for (auto& c : convs) {
                    if (c.res) continue;
                    if (c.end && kve::end_usable(c, (int)c.toks.size())) check_restore(c, c.toks, c.toks.size(), *c.end, "end");
                    for (auto& s : c.ck)
                        if (s->pos <= (int)c.toks.size() && kve::ck_usable(c, *s, (int)c.toks.size(), false))
                            check_restore(c, c.toks, s->pos, *s, "checkpoint");
                }
            }
            if ((int)prompts.size() > 40) prompts.erase(prompts.begin(), prompts.begin() + 20);
            if (fails > 20) break;
        }
    }
    room::test_room();
    room::test_main_moved();
    room::test_reclaim_order();
    room::test_refused_write();
    room::test_reachable();
    fprintf(stderr, "%lld ops, %lld restores (%lld through a segment reaching past kv_n), %lld forks, %lld whole drops, %lld cuts, "
            "%lld refused writes\n", ops, restores, partial_seg, forks, drops, cuts, refused);
    EXPECT(restores > 1000 && forks > 100 && drops > 100 && cuts > 100 && partial_seg > 0);
    if (fails) { fprintf(stderr, "%d failures\n", fails); return 1; }
    fprintf(stderr, "all passed\n");
    return 0;
}
