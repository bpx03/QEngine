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
#include "flash/host_slabs.h"

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
    // the checkpoint cap: the latest turn end survives the turns' prompt-end states, at the real cap of 6 and at 2, for every
    // conversation alike (prio is not read: user decision 2026-11-12); no state is relabelled (review L2)
    auto mk = [](int pos, int kind) { auto x = std::make_shared<State>(); x->pos = pos; x->kind = kind; return x; };
    auto has_pos = [](const Conv& c, int pos) { for (auto& x : c.ck) if (x->pos == pos) return true; return false; };
    auto lte_pos = [](const Conv& c) { auto* k = (const State*)kve::last_turn_end(c); return k ? k->pos : -1; };
    for (int keep : {6, 2})
        for (int prio : {0, 1, 2}) {
            Conv m;
            m.prio = prio;
            kve::add_ck(m, mk(50, 1), keep);  // the system end
            for (int t = 0; t < 8; t++) {
                kve::add_ck(m, mk(100 + 10 * t, 2), keep);  // the prompt end
                kve::add_ck(m, mk(101 + 10 * t, 4), keep);  // the live end before the next request extends it
                EXPECT(lte_pos(m) == 101 + 10 * t && (int)m.ck.size() <= keep);
                for (auto& x : m.ck) EXPECT(x->kind != 6);
            }
            kve::add_ck(m, mk(300, 2), keep);  // the next prompt end does not push it out
            EXPECT(has_pos(m, 171) && lte_pos(m) == 171 && (int)m.ck.size() == keep);
        }
    {  // [qfn-ttft-pe] a prompt end (7) per turn: only the latest is kept, the system end / periodic / n - 1 states stay
        Conv m;
        kve::add_ck(m, mk(50, 1), 6);      // system end
        kve::add_ck(m, mk(8192, 3), 6);    // periodic
        kve::add_ck(m, mk(9999, 2), 6);    // prompt end - 1 of a long turn
        kve::add_ck(m, mk(10000, 7), 6);
        for (int t = 1; t <= 8; t++) kve::add_ck(m, mk(10000 + 300 * t, 7), 6);  // short agent turns
        int n7 = 0;
        for (auto& x : m.ck) n7 += x->kind == 7;
        EXPECT(n7 == 1 && has_pos(m, 12400) && has_pos(m, 50) && has_pos(m, 8192) && has_pos(m, 9999) && m.ck.size() == 4);
        kve::drop_kind(m, 7);
        EXPECT(m.ck.size() == 3 && !has_pos(m, 12400));
        // the default cap 3: a long turn's periodic state goes, its n - 1 (an identical retry) and n stay with the system end
        Conv m3;
        kve::add_ck(m3, mk(50, 1), 3);
        kve::add_ck(m3, mk(8192, 3), 3);
        kve::add_ck(m3, mk(9999, 2), 3);
        kve::add_ck(m3, mk(10000, 7), 3);
        EXPECT(m3.ck.size() == 3 && has_pos(m3, 50) && has_pos(m3, 9999) && has_pos(m3, 10000));
        kve::add_ck(m3, mk(10300, 7), 3);  // the next short turn
        EXPECT(m3.ck.size() == 3 && has_pos(m3, 9999) && has_pos(m3, 10300) && !has_pos(m3, 10000));
        // a fork keeps the parent's prompt end until its own turn replaces it in the fork only
        Conv p;
        kve::add_ck(p, mk(50, 1), 6);
        kve::add_ck(p, mk(500, 7), 6);
        const std::vector<int> ids(500, 7);
        p.toks = ids; p.kv_n = 500;
        Conv f;
        kve::fork_from(f, p, ids.data(), 500);
        kve::add_ck(f, mk(800, 7), 6);
        EXPECT(has_pos(p, 500) && !has_pos(f, 500) && has_pos(f, 800) && has_pos(f, 50));
    }
    {  // a state of an earlier build's mark (kind 6, read back from a disk-cache file) counts as a turn end
        Conv m;
        kve::add_ck(m, mk(10, 6), 2);
        kve::add_ck(m, mk(20, 2), 2);
        kve::add_ck(m, mk(30, 2), 2);
        EXPECT(has_pos(m, 10) && lte_pos(m) == 10);
    }
    {  // review L2: a fork shares the parent's states; the fork's turns neither relabel nor push out the parent's latest turn end
        for (int keep : {6, 2}) {
            Conv p;
            kve::add_ck(p, mk(50, 1), keep);
            kve::add_ck(p, mk(100, 2), keep);
            kve::add_ck(p, mk(101, 4), keep);  // the parent's latest turn end
            Conv f;
            const std::vector<int> ids(101, 7);
            p.toks = ids; p.kv_n = 101;
            kve::fork_from(f, p, ids.data(), 101);
            EXPECT(lte_pos(f) == 101);
            for (int t = 0; t < 4; t++) {  // the fork's own turns
                kve::add_ck(f, mk(200 + 10 * t, 2), keep);
                kve::add_ck(f, mk(201 + 10 * t, 4), keep);
            }
            EXPECT(lte_pos(f) == 231);
            for (auto& x : p.ck) EXPECT(x->pos != 101 || x->kind == 4);  // the shared state keeps its kind
            kve::add_ck(p, mk(300, 2), keep);  // the parent's next prompt end: its own latest turn end stays
            kve::add_ck(p, mk(310, 3), keep);
            EXPECT(has_pos(p, 101) && lte_pos(p) == 101 && (int)p.ck.size() <= keep);
        }
    }
    // the order: least recently used, nothing else (no main tier, user decision 2026-11-12)
    using kve::goes_first;
    using kve::Rank;
    EXPECT(goes_first(Rank{1}, Rank{2}));
    EXPECT(!goes_first(Rank{2}, Rank{1}));
    EXPECT(!goes_first(Rank{3}, Rank{3}));
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
    // (the archive order, as FE::copy_rank: least recently used, every conversation alike; no main tier)
    C& add(int id, size_t bytes, unsigned long long used, bool stays = false) {
        convs.emplace_back();
        C& c = convs.back();
        c.id = id;
        c.r = kve::Rank{used};
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
static void test_reclaim_order() {
    using S = kve::IdleSlot;
    using V = std::vector<int>;
    auto o = [](int s, int cap, double idle, unsigned long long u) { return S{s, cap, idle, kve::Rank{u}}; };
    const int np = 256;
    // least recently used first, each by its own idle time; no size rule, no extra wait (no main tier, user decision 2026-11-12:
    // before, a session's main waited FLASH_EC_RECLAIM_MAIN_S and went only while the idle mains held >= 1/16 of the pool)
    std::vector<S> eight;
    for (int i = 0; i < 8; i++) eight.push_back(o(i, 13, 60, 100 - i));
    EXPECT(kve::reclaim_order(eight, true, 2, np) == V({7, 6, 5, 4, 3, 2, 1, 0}));
    EXPECT(kve::reclaim_order(eight, false, 30, np) == V({7, 6, 5, 4, 3, 2, 1, 0}));  // idle 60 s >= 30 s: also without pressure
    EXPECT(kve::reclaim_order({o(0, 87, 60, 1)}, true, 2, np) == V({0}));             // a big one at the fast wait (#1928 shape)
    EXPECT(kve::reclaim_order({o(0, 5, 60, 9), o(1, 87, 60, 1), o(2, 5, 60, 3), o(3, 5, 1, 2)}, true, 2, np) == V({1, 2, 0}));
    EXPECT(kve::reclaim_order({o(0, 5, 20, 9), o(1, 5, 31, 3)}, false, 30, np) == V({1}));
    EXPECT(kve::reclaim_order({o(0, 5, 1.9, 1)}, true, 2, np).empty());               // inside its wait
    // qfn-kv-vram-place: an idle one above every generating region leaves before Tord under pressure (kve::at_once)
    auto a = [](int s, int cap, double idle, bool, bool above) { return S{s, cap, idle, kve::Rank{1}, above}; };
    EXPECT(kve::at_once(a(0, 9, 0.1, false, true), true, 0, false, np));
    EXPECT(!kve::at_once(a(0, 9, 0.1, false, true), false, 0, false, np));   // no pressure
    EXPECT(!kve::at_once(a(0, 9, 0.1, false, false), true, 0, false, np));   // not above the generating ones
    EXPECT(kve::at_once(a(0, 9, 0.1, true, true), true, 0, false, np));      // a session's main too (no tier)
    EXPECT(!kve::at_once(a(0, 9, 0.1, false, true), true, -1, false, np));   // off
    EXPECT(!kve::at_once(a(0, 9, 0.1, false, true), true, 0.5, false, np));  // inside its wait
    EXPECT(kve::at_once(a(0, 9, 0.6, false, true), true, 0.5, false, np));
    EXPECT(!kve::at_once(a(0, 15, 0.1, false, true), true, 0, true, np));    // min size: < 1/16 of the pool stays
    EXPECT(kve::at_once(a(0, 16, 0.1, false, true), true, 0, true, np));
    EXPECT(kve::reclaim_order({a(0, 9, 0.1, false, true), a(1, 9, 0.1, false, false)}, true, 2, np, 0) == V({0}));
    EXPECT(kve::reclaim_order({a(0, 9, 0.1, false, true)}, true, 2, np).empty());  // default: the rule is off
}
// one pass of FE::reclaim_idle on the model: each one in the order writes under ROOM_OLDER (without pressure: nothing may go) and
// leaves; a refused one follows kve::RefusedWrite. out: who left with a copy, who left lost.
struct Pass { std::vector<int> archived, lost; };
static Pass reclaim_pass(M& m, const std::vector<kve::IdleSlot>& idle, const std::vector<size_t>& need, size_t limit, bool pressure,
                         std::vector<bool>& resident) {
    Pass out;
    kve::RefusedWrite refused;
    const std::vector<int> ord = kve::reclaim_order(idle, pressure, 2, 256);
    for (int i : ord) {
        C* c = nullptr;
        for (auto& x : m.convs) if (x.id == i) c = &x;
        if (!c) continue;  // (gone in an earlier pass)
        c->stays = true;  // the writer
        const kve::Rank none{0};
        const bool ok = m.room(need[i], limit, {}, pressure ? &c->r : &none);
        c->stays = false;
        if (ok) {
            c->segs.push_back(std::make_shared<B>(B{need[i]}));
            out.archived.push_back(i);
            resident[i] = false;
        } else if (refused.leaves(pressure)) {
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
    auto ord = [](int s, unsigned long long u) { return S{s, 20, 60, kve::Rank{u}}; };
    auto mn = ord;  // a session's main: no tier, the same rank (user decision 2026-11-12)
    {  // an archive full of newer copies: the oldest goes, one per pass; newer copies and a generating one are untouched; a session's
       // main is no exception (user decision 2026-11-12): it goes in its turn
        M m;
        m.add(1, 0, 1);            // idle, oldest
        m.add(2, 0, 2);            // idle
        m.add(3, 0, 3);  // idle, a session main once: no tier, ordered as any other
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
        p = reclaim_pass(m, {mn(3, 3)}, need, 4, true, res);  // the main left: it is the first refused one, it leaves (#1928)
        EXPECT(p.lost == V({3}) && p.archived.empty() && !res[3] && m.gone.empty() && m.convs.size() == 3);
    }
    {  // no pressure: everyone stays
        M m;
        m.add(1, 0, 1);
        m.add(9, 4, 9);
        std::vector<bool> res(2, true);
        const Pass p = reclaim_pass(m, {ord(1, 1)}, {0, 1}, 4, false, res);
        EXPECT(p.lost.empty() && p.archived.empty() && res[1] && m.gone.empty());
    }
    {  // bigger than the whole archive, which is empty: the first refused one goes (one per pass), the next pass the other
        M m;
        m.add(1, 0, 1);
        m.add(2, 0, 2);
        std::vector<bool> res(3, true);
        Pass p = reclaim_pass(m, {ord(1, 1), mn(2, 2)}, {0, 7, 7}, 6, true, res);
        EXPECT(p.lost == V({1}) && res[2]);
        p = reclaim_pass(m, {mn(2, 2)}, {0, 7, 7}, 6, true, res);
        EXPECT(p.lost == V({2}) && !res[2]);
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
        m.add(1, 4, 9, true);
        m.add(2, 1, 1);
        m.add(3, 1, 2);
        EXPECT(!m.room(3, 6) && m.gone.empty());  // 4 + 3 > 6 with both others gone
        EXPECT(m.convs.size() == 3);
    }
    {  // it fits once the older ones are gone: the order holds, and only as many as needed go
        M m;
        m.add(1, 2, 9, true);
        m.add(2, 1, 5);
        m.add(3, 1, 1);
        m.add(4, 1, 3);
        EXPECT(m.room(3, 6));
        EXPECT(m.gone == std::vector<int>({3, 4}));
    }
    {  // every conversation alike: the least recently used goes first (no main tier in the archive, user decision 2026-11-12)
        M m;
        m.add(1, 2, 1);
        m.add(2, 2, 5);
        EXPECT(m.room(3, 4));
        EXPECT(m.gone == std::vector<int>({1, 2}));
    }
    {  // blobs shared with what stays free nothing: a fork of the writer, a pinned prefix
        M m;
        C& k = m.add(1, 4, 9, true);
        C& f = m.add(2, 0, 1);
        f.segs.push_back(k.segs[0]);
        C& o = m.add(3, 1, 2);
        const std::vector<const void*> pinned = {o.segs[0].get()};
        EXPECT(!m.room(2, 5, pinned) && m.gone.empty());  // nothing of 2 and 3 can free a byte
        EXPECT(m.room(1, 5) && m.gone == std::vector<int>({2, 3}));  // no pin: it fits with them gone, oldest first
    }
    {  // a blob two evictable conversations share counts once
        M m;
        m.add(9, 2, 9, true);
        C& a = m.add(1, 3, 1);
        C& b = m.add(2, 0, 2);
        b.segs.push_back(a.segs[0]);
        EXPECT(!m.room(5, 6) && m.gone.empty());  // 5 used + 5 > 6 + 3 (counted twice it would seem to fit)
    }
    {  // the reclaim's mode: only what goes before the writer; newer ones stay and the write is refused without a loss
        M m;
        kve::Rank w{5};
        m.add(1, 1, 9, true);
        m.add(2, 2, 7);
        m.add(3, 2, 8);
        EXPECT(!m.room(2, 5, {}, &w) && m.gone.empty());
        m.add(4, 2, 1);
        EXPECT(m.room(2, 7, {}, &w) && m.gone == std::vector<int>({4}));
    }
    {  // too big for the whole limit: nothing goes
        M m;
        m.add(2, 1, 1);
        EXPECT(!m.room(7, 6) && m.gone.empty());
    }
}
}  // namespace room

namespace guard {
// FE::mem_guard on the real slab bookkeeping (src/flash/host_slabs.h: SlabSet, kve::compact) and kve::give_back, over plain
// memory: conversations' blobs carved first fit as copies grow turn by turn; only whole free slabs go back to the host.
constexpr size_t PG = 4096, SLAB = 64 * PG;
struct Pool {
    kve::SlabSet set;
    std::vector<std::unique_ptr<char[]>> mem;
    size_t returned = 0, pend_page = 0, trimmed = 0;
    explicit Pool(int n) {
        for (int i = 0; i < n; i++) { mem.emplace_back(new char[SLAB]); set.add(mem.back().get(), SLAB); }
    }
    std::vector<const char*> slabs_by_use() const { return set.by_use(); }
    std::pair<size_t, size_t> slab_use(const char* b) const {
        const auto* s = set.find(b);
        return s ? std::make_pair(kve::SlabSet::used_of(*s), s->size) : std::make_pair((size_t)0, (size_t)0);
    }
    bool plan(const char* f, const std::vector<size_t>& z, const std::vector<const char*>& a, std::vector<char*>& d) const { return set.plan(f, z, a, d); }
    bool claim(char* p, size_t n) { return set.claim(p, kve::SlabSet::round(n)); }
    void free(char* p, size_t n, bool) { if (!set.put_back(p, kve::SlabSet::round(n))) { fails++; fprintf(stderr, "FAIL put_back %p %zu (slabs %zu)\n", (void*)p, n, set.slabs.size()); } }
    size_t release(size_t want) { const size_t g = set.drop_free(want, [](char*, size_t) {}); returned += g; return g; }
    const char* slab_of(const char* p) const {
        for (auto& s : set.slabs) if (p >= s.base && p < s.base + s.size) return s.base;
        return nullptr;
    }
    size_t pinned_free() const { size_t f = 0; for (auto& s : set.slabs) f += s.size - kve::SlabSet::used_of(s); return f; }
};
struct Blob {  // one piece, pinned (in a slab) or pageable; filled with a pattern of its id
    Pool* pool;
    char* p = nullptr;
    size_t n = 0;
    bool pinned = true;
    int id = 0, kind = 3;  // (a state's kind: 6 = the latest turn end, kve::add_ck)
    ~Blob() {
        if (!p) return;
        if (pinned) pool->free(p, n, true);
        else { delete[] p; pool->pend_page += n; }
    }
    void fill() { for (size_t i = 0; i < n; i += 512) p[i] = (char)(id * 31 + i / 512); }
    bool intact() const { for (size_t i = 0; i < n; i += 512) if (p[i] != (char)(id * 31 + i / 512)) return false; return true; }
};
using BP = std::shared_ptr<Blob>;
struct Cv {
    int id;
    unsigned long long used;
    bool res = false;
    std::vector<BP> segs, ck;
    BP end;  // its end state (kve: c.end, dropped by a level-1 eviction with the other spare states; qfn_archive_loss rr-L2)
};
struct G {
    Pool pool;
    std::list<Cv> convs;
    std::vector<BP> pins, held;  // pinned prefixes (outside the LRU), blobs a disk write holds
    bool pins_drop = false;      // FE::mem_guard: a pinned prefix goes when no copy is left (pins: oldest first)
    std::vector<char> events;    // 'c' a copy went, 'p' a pinned prefix
    size_t ext = 0;              // what another process took: the deficit before anything went back
    std::vector<int> lost;
    int next_blob = 1;
    size_t max_moved = 0, max_got = 0;  // most bytes one pass copied; most one compaction call returned
    explicit G(int nslab) : pool(nslab) {}
    BP blob(size_t pages, bool pinned = true) {
        auto b = std::make_shared<Blob>();
        b->pool = &pool;
        b->n = pages * PG;
        b->id = next_blob++;
        b->pinned = pinned;
        b->p = pinned ? pool.set.carve(b->n) : new char[b->n];
        if (!b->p) return nullptr;
        b->fill();
        return b;
    }
    size_t deficit() const { const size_t g = pool.returned + pool.trimmed; return ext > g ? ext - g : 0; }
    void trim() { pool.trimmed += pool.pend_page; pool.pend_page = 0; }
    // its last turn end (kve::last_turn_end: the turn-end state, kind 4, latest): kept by a level-1 drop
    static const Blob* lte(const Cv& c) { const Blob* k = nullptr; for (auto& b : c.ck) if (b->kind == 4 && (!k || b->id > k->id)) k = b.get(); return k; }
    static bool spare_states(const Cv& c) { if (c.end) return true; for (auto& b : c.ck) if (b.get() != lte(c)) return true; return false; }
    // the archive order (FE::copy_rank): every conversation alike, the ram-mode spare tiers first, then least recently used
    kve::Rank rank(const Cv& c) const { return kve::Rank{c.used, !c.res ? 0 : !c.segs.empty() ? 2 : spare_states(c) ? 1 : 0}; }
    static size_t bytes(const std::vector<BP>& v) { size_t n = 0; for (auto& b : v) n += b->n; return n; }
    bool evict(size_t& cost) {
        auto v = convs.end();
        for (auto it = convs.begin(); it != convs.end(); ++it)
            if ((!it->segs.empty() || !it->ck.empty() || it->end) && (v == convs.end() || kve::goes_first(rank(*it), rank(*v)))) v = it;
        if (v == convs.end()) {
            if (!pins_drop || pins.empty()) return false;
            cost = pins.front()->n;
            pins.erase(pins.begin());
            events.push_back('p');
            return true;
        }
        events.push_back('c');
        if (rank(*v).spare == 2) { cost = bytes(v->segs); v->segs.clear(); return true; }
        if (rank(*v).spare == 1) {  // (kve::drop_spare_states)
            cost = v->end ? v->end->n : 0;
            v->end = nullptr;
            const Blob* k = lte(*v);
            for (auto& b : v->ck) if (b.get() != k) cost += b->n;
            v->ck.erase(std::remove_if(v->ck.begin(), v->ck.end(), [&](const BP& b) { return b.get() != k; }), v->ck.end());
            return true;
        }
        cost = bytes(v->segs) + bytes(v->ck) + (v->end ? v->end->n : 0);
        if (v->res) { v->segs.clear(); v->ck.clear(); v->end = nullptr; return true; }
        lost.push_back(v->id);
        convs.erase(v);
        return true;
    }
    std::vector<kve::Piece> pieces() {
        std::vector<kve::Piece> out;
        auto add = [&](const BP& b) {
            if (!b->pinned) return;
            for (auto& h : held) if (h == b) return;
            for (auto& x : out) if (x.p == &b->p) return;
            out.push_back({&b->p, b->n});
        };
        for (auto& c : convs) { for (auto& b : c.segs) add(b); for (auto& b : c.ck) add(b); if (c.end) add(c.end); }
        for (auto& b : pins) add(b);
        return out;
    }
    bool all_intact() const {
        for (auto& c : convs) {
            for (auto& b : c.segs) if (!b->intact()) return false;
            for (auto& b : c.ck) if (!b->intact()) return false;
            if (c.end && !c.end->intact()) return false;
        }
        for (auto& b : pins) if (!b->intact()) return false;
        return true;
    }
    // one guard pass: fixed = FE::mem_guard (give_back + kve::compact, the episode count `ep`), else the 1.3.5.9 loop
    size_t pass(bool fixed, size_t& ep, size_t cap = 2 * SLAB, size_t urgent = 1000 * SLAB) {
        size_t gone = 0, pass_moved = 0;
        if (!deficit()) ep = 0;  // (FE::mem_guard: the episode ends when the deficit is gone)
        if (fixed) {
            gone = kve::give_back([&] { return deficit(); }, [&](size_t x) { pool.release(x); },
                [&](size_t x) {
                    if (pass_moved >= cap) return -1;
                    size_t mv;
                    int h = 0;
                    auto pc = pieces();
                    const size_t got = kve::compact(pool, pc, x, cap - pass_moved, mv, &h);
                    pass_moved += mv;
                    max_moved = std::max(max_moved, pass_moved);
                    max_got = std::max(max_got, got);
                    return got ? 1 : h ? -1 : 0;
                },
                [&] { return pool.pinned_free(); }, [&](size_t& f) { return evict(f); }, ep, SLAB, urgent);
        } else {
            pool.release(deficit());
            while (deficit()) {
                size_t f = 0;
                if (!evict(f)) break;
                gone += f;
                pool.release(deficit());
            }
        }
        trim();
        return gone;
    }
};
// n conversations, each turn appends a blob of 1-4 pages to one of them (every slab ends up holding many of them); every
// `ck_every`-th blob is a state (ck)
static void fill(G& g, int nconv, unsigned seed, int res_every = 0, int ck_every = 3) {
    std::mt19937 rng(seed);
    for (int i = 0; i < nconv; i++) g.convs.push_back({i, (unsigned long long)i, res_every && i % res_every == 0, {}, {}});
    std::vector<Cv*> cv;
    for (auto& c : g.convs) cv.push_back(&c);
    for (int t = 0;; t++) {
        BP b = g.blob(1 + rng() % 4);
        if (!b) break;
        Cv* c = cv[rng() % cv.size()];
        (t % ck_every ? c->segs : c->ck).push_back(b);
        c->used = 1000 + t;
    }
}
static size_t max_conv(const G& g) { size_t m = 0; for (auto& c : g.convs) m = std::max(m, G::bytes(c.segs) + G::bytes(c.ck) + (c.end ? c.end->n : 0)); return m; }

static void test_give_back() {
    size_t old_lost = 0, new_lost = 0;
    for (unsigned seed = 1; seed <= 40; seed++) {  // a full archive, a deficit of one slab: something must go
        G a(16), b(16);
        fill(a, 30, seed);
        fill(b, 30, seed);
        a.ext = b.ext = SLAB;
        size_t ea = 0, eb = 0;
        a.pass(false, ea);
        // the fixed guard may stop with the room in place (free ranges it could not compact yet): later passes go on
        for (int k = 0; k < 6 && (k == 0 || b.deficit()); k++) b.pass(true, eb);
        EXPECT(eb <= SLAB + SLAB + max_conv(b) + SLAB);
        old_lost += a.lost.size();
        new_lost += b.lost.size();
        EXPECT(b.deficit() == 0);
        EXPECT(b.lost.size() <= a.lost.size());
        EXPECT(b.all_intact());  // every moved blob holds its bytes
    }
    fprintf(stderr, "guard: 40 full archives, deficit 1 slab: 1.3.5.9 loop lost %zu conversations, give_back + compaction %zu\n",
            old_lost, new_lost);
    EXPECT(new_lost * 3 < old_lost);
    for (unsigned seed = 1; seed <= 20; seed++) {  // free room in the slabs (10 of 30 gone): compaction alone, nothing lost
        G g(16);
        fill(g, 30, seed);
        for (int i = 0; i < 10; i++) { size_t f; g.evict(f); }
        g.lost.clear();
        g.ext = 2 * SLAB;
        size_t ep = 0;
        g.pass(true, ep);
        EXPECT(g.lost.empty());
        EXPECT(g.deficit() == 0);
        EXPECT(g.all_intact());
    }
    {  // the per-pass cap: one pass moves at most cap (+ one slab), the next passes finish it, nothing lost
        G g(16);
        fill(g, 30, 5);
        for (int i = 0; i < 12; i++) { size_t f; g.evict(f); }
        g.lost.clear();
        g.ext = 6 * SLAB;
        size_t ep = 0;
        for (int k = 0; k < 40 && g.deficit(); k++) g.pass(true, ep, PG);
        EXPECT(g.max_moved <= SLAB);  // per pass (the first slab of a pass may pass the cap)
        EXPECT(g.deficit() == 0);
        EXPECT(g.lost.empty());
        EXPECT(g.all_intact());
    }
    {  // a disk write holds a blob in every slab (none can be emptied) and the host never gets memory back: many passes evict at
       // most the deficit + one slab + one conversation in all; the 1.3.5.9 loop evicts everything
        G a(16), b(16);
        fill(a, 30, 7);
        fill(b, 30, 7);
        for (G* g : {&a, &b})
            for (auto& s : g->pool.set.slabs) {  // the first carved blob of each slab becomes held
                for (auto& c : g->convs)
                    for (auto& x : c.segs)
                        if (x->p >= s.base && x->p < s.base + s.size && g->held.size() < g->pool.set.slabs.size()) {
                            bool have = false;
                            for (auto& h : g->held) have |= h->p >= s.base && h->p < s.base + s.size;
                            if (!have) g->held.push_back(x);
                        }
            }
        a.ext = b.ext = 2 * SLAB;
        size_t ea = 0, eb = 0, gone = 0;
        for (int k = 0; k < 20; k++) gone += b.pass(true, eb);
        EXPECT(gone <= 2 * SLAB + SLAB + max_conv(a));
        EXPECT(b.lost.size() < b.convs.size() + b.lost.size());
        EXPECT(b.all_intact());
        a.pass(false, ea);
        EXPECT(a.convs.empty());
        // the write finished: the next passes compact, the deficit goes, still nothing more lost
        const size_t before = b.lost.size();
        b.held.clear();
        for (int k = 0; k < 8 && b.deficit(); k++) b.pass(true, eb);
        EXPECT(b.deficit() == 0);
        EXPECT(b.lost.size() == before);
        EXPECT(b.all_intact());
    }
    {  // every conversation's blobs shared with a pinned prefix (or a fork): an eviction frees nothing, but it counts; the old loop
       // would evict all of them
        G g(16);
        fill(g, 30, 9);
        for (auto& c : g.convs) for (auto& x : c.segs) g.pins.push_back(x);
        for (auto& c : g.convs) for (auto& x : c.ck) g.pins.push_back(x);
        g.ext = 2 * SLAB;
        size_t ep = 0, gone = 0;
        for (int k = 0; k < 10; k++) gone += g.pass(true, ep);
        EXPECT(gone <= 2 * SLAB + SLAB + max_conv(g));
        EXPECT(g.lost.size() < 15);
    }
    {  // a small deficit, plenty of free ranges and slabs that can be emptied: compaction alone, nothing lost
        G g(4);
        BP big = g.blob(40), small1 = g.blob(30), small2 = g.blob(30), s3 = g.blob(30), s4 = g.blob(30);  // 40 | 30+30 | ...
        Cv c1{1, 1, false, {big}, {}}, c2{2, 2, false, {small1, small2}, {}}, c3{3, 3, false, {s3, s4}, {}};
        g.convs.push_back(c1); g.convs.push_back(c2); g.convs.push_back(c3);
        big.reset(); small1.reset(); small2.reset(); s3.reset(); s4.reset();
        g.ext = SLAB / 4;  // well below the free ranges (24 + 4 + 4 + 64 pages free)
        size_t ep = 0;
        g.pass(true, ep);
        EXPECT(g.lost.empty());
    }
    {  // a disk write holds a blob in the only slab compaction could empty, and the free ranges cover the deficit: the guard
       // waits (no eviction over many passes) and compacts once the write let go. Slabs: 0 = A 56 + 8 free, 1 = B 48 + 16 free,
       // 2 = C1 16 (held) + C2 8 + 40 free, 3 = empty
        G g(4);
        BP A = g.blob(56), F0 = g.blob(8), B = g.blob(48), F1 = g.blob(16), C1 = g.blob(16), C2 = g.blob(8);
        EXPECT(g.pool.slab_of(C1->p) == g.pool.set.slabs[2].base && g.pool.slab_of(C2->p) == g.pool.set.slabs[2].base);
        F0.reset();
        F1.reset();
        g.convs.push_back({1, 1, false, {A}, {}});
        g.convs.push_back({2, 2, false, {B}, {}});
        g.convs.push_back({3, 3, false, {C1, C2}, {}});
        A.reset(); B.reset(); C2.reset();
        g.held.push_back(C1);
        g.ext = SLAB + SLAB / 2;     // slab 3 goes at once; then 32 pages short, 64 pages free inside slabs 0-2
        size_t ep = 0;
        for (int k = 0; k < 10; k++) g.pass(true, ep);
        EXPECT(g.lost.empty());
        EXPECT(g.deficit() == SLAB / 2);
        g.held.clear();
        C1.reset();
        for (int k = 0; k < 3 && g.deficit(); k++) g.pass(true, ep);
        EXPECT(g.lost.empty());
        EXPECT(g.deficit() == 0);
        EXPECT(g.all_intact());
    }
    {  // residents' rows (spare) go first, then their states but the recorded last turn end; only then archived ones, in LRU order
        G g(16);
        fill(g, 30, 11, 4);
        std::vector<std::pair<int, int>> lte_of;  // resident id, its last turn end's blob id
        for (auto& c : g.convs)
            if (c.res) {
                if (c.ck.empty()) c.ck.push_back(g.blob(1));
                c.ck.front()->kind = 4;  // (an older turn end too: only the latest stays)
                c.ck.back()->kind = 4;
                lte_of.push_back({c.id, c.ck.back()->id});
            }
        g.ext = 10 * SLAB;
        size_t ep = 0;
        for (int k = 0; k < 12 && g.deficit(); k++) g.pass(true, ep);
        EXPECT(!g.lost.empty());  // (a deficit that costs conversations: the checks below are real)
        for (auto& c : g.convs) if (c.res) EXPECT(c.segs.empty());  // every resident's rows went before the first loss
        for (auto& [id, k] : lte_of)
            for (auto& c : g.convs)
                if (c.id == id && !c.ck.empty()) EXPECT(c.ck.size() == 1 && c.ck[0]->id == k);  // (what is left is the recorded one)
        EXPECT(g.all_intact());
    }
    {  // pageable copies: freed bytes come back only after the pass (malloc_trim): the bound stops the pass, the next one sees it
        G g(2);
        for (int i = 0; i < 20; i++) g.convs.push_back({i, (unsigned long long)i, false, {g.blob(8, false)}, {}});
        g.ext = 16 * PG;
        size_t ep = 0;
        g.pass(true, ep);
        EXPECT(g.lost.size() <= 4);
        EXPECT(g.deficit() == 0);
    }
    {  // urgent (the deficit above `urgent`), every victim's blobs shared with a pinned prefix (they free nothing now): each pass
       // and the whole episode stop at their limits; the 1.3.5.9 loop would evict everything
        G g(16);
        fill(g, 30, 17);
        for (auto& c : g.convs) { for (auto& x : c.segs) g.pins.push_back(x); for (auto& x : c.ck) g.pins.push_back(x); }
        g.ext = 4 * SLAB;
        size_t ep = 0, gone = 0;
        for (int k = 0; k < 20; k++) gone += g.pass(true, ep, 2 * SLAB, SLAB);
        EXPECT(gone <= 2 * 4 * SLAB + SLAB + max_conv(g));
        EXPECT(!g.convs.empty());
    }
    {  // the episode: its limit holds across passes, a new episode (the deficit was gone) may evict again
        G g(16);
        fill(g, 30, 19);
        for (auto& c : g.convs) { for (auto& x : c.segs) g.pins.push_back(x); for (auto& x : c.ck) g.pins.push_back(x); }
        g.ext = 2 * SLAB;
        size_t ep = 0, gone = 0;
        for (int k = 0; k < 10; k++) gone += g.pass(true, ep);
        const size_t lost1 = g.lost.size();
        EXPECT(lost1 > 0 && gone <= 2 * SLAB + SLAB + max_conv(g));
        EXPECT(g.pass(true, ep) == 0);           // (still the same episode: nothing more)
        g.pins.clear();                           // the prefixes went: memory comes back, the deficit is gone
        for (int k = 0; k < 4; k++) g.pass(true, ep);
        EXPECT(g.deficit() == 0);
        g.ext = g.pool.returned + g.pool.trimmed + 3 * SLAB;  // a new episode
        for (auto& c : g.convs) { for (auto& x : c.segs) g.pins.push_back(x); for (auto& x : c.ck) g.pins.push_back(x); }
        g.pass(true, ep);
        EXPECT(g.lost.size() > lost1);            // (it evicts again)
    }
    {  // every session returning (the user's workload: separate `claude -p` runs): residents with only states (no rows archived
       // yet: the archiver had no idle time) give those states, all but their last turn end, before an archived conversation goes
        G g(16);
        fill(g, 30, 13, 3);
        for (auto& c : g.convs)
            if (c.res) {
                c.ck.insert(c.ck.end(), c.segs.begin(), c.segs.end());
                c.segs.clear();
                c.ck.back()->kind = 4;
                if (c.ck.size() > 2) { c.end = c.ck.front(); c.ck.erase(c.ck.begin()); }  // (an end state too: rr-L2)
            }
        std::vector<int> keep;
        for (auto& c : g.convs) if (c.res) keep.push_back(G::lte(c)->id);
        g.ext = 8 * SLAB;
        size_t ep = 0;
        for (int k = 0; k < 12 && g.deficit(); k++) g.pass(true, ep);
        EXPECT(!g.lost.empty());  // (a deficit that costs conversations: the checks below are real)
        int res_with_states = 0;  // (states a level-1 eviction could still drop)
        for (auto& c : g.convs) if (c.res && G::spare_states(c)) res_with_states++;
        EXPECT(res_with_states == 0);  // before the first loss, every resident gave its states but the last turn end
        for (auto& c : g.convs) EXPECT(!c.res || !c.end);  // (its end state went with them)
        size_t kept = 0;
        for (auto& c : g.convs) for (auto& b : c.ck) kept += std::find(keep.begin(), keep.end(), b->id) != keep.end();
        EXPECT(kept > 0);              // (last turn ends go only in LRU order with the archived ones)
        EXPECT(ep <= 8 * SLAB + SLAB + max_conv(g));
        EXPECT(g.all_intact());
    }
    {  // a pinned prefix goes only when no copy is left (FE::mem_guard: a shared system prefix is worth more than any one copy)
        G g(16);
        fill(g, 30, 23);
        int n = 0;
        for (auto& c : g.convs) if (n++ % 3 == 0 && !c.segs.empty()) g.pins.push_back(c.segs[0]);  // (shared with a copy)
        for (int i = 0; i < 4; i++) if (BP b = g.blob(1)) g.pins.push_back(b);                    // (only pinned)
        g.pins_drop = true;
        g.ext = 1000 * SLAB;  // (more than everything: all goes, the order is what counts)
        size_t ep = 0;
        for (int k = 0; k < 40; k++) g.pass(true, ep, 2 * SLAB, 100000 * SLAB);
        const auto first_p = std::find(g.events.begin(), g.events.end(), 'p');
        EXPECT(first_p != g.events.end());
        EXPECT(std::find(first_p, g.events.end(), 'c') == g.events.end());  // no copy after the first prefix
        EXPECT(g.convs.empty() || [&] { for (auto& c : g.convs) if (!c.segs.empty() || !c.ck.empty()) return false; return true; }());
    }
    {  // LRU fairness: among archived conversations the least recently used goes first, whatever else (no main tier)
        G g(16);
        fill(g, 30, 21);
        std::vector<std::pair<unsigned long long, int>> by;
        for (auto& c : g.convs) by.push_back({c.used, c.id});
        std::sort(by.begin(), by.end());
        g.ext = 2 * SLAB;
        size_t ep = 0;
        for (int k = 0; k < 4 && g.deficit(); k++) g.pass(true, ep);
        for (size_t i = 0; i < g.lost.size(); i++) EXPECT(g.lost[i] == by[i].second);
    }
    // the archive order: rows of a resident, then a resident's states, then least recently used
    EXPECT(kve::goes_first(kve::Rank{9, 2}, kve::Rank{1, 1}));
    EXPECT(kve::goes_first(kve::Rank{9, 1}, kve::Rank{1, 0}));
    EXPECT(kve::goes_first(kve::Rank{1, 0}, kve::Rank{9, 0}));
}
// slab bookkeeping: carve / put_back coalescing, plan never changes anything, claim exactly, a failed plan copies nothing
static void test_slabset() {
    std::vector<std::unique_ptr<char[]>> mem;
    kve::SlabSet s;
    for (int i = 0; i < 3; i++) { mem.emplace_back(new char[SLAB]); s.add(mem.back().get(), SLAB); }
    std::vector<char*> v;
    for (int i = 0; i < 48; i++) v.push_back(s.carve(4 * PG));  // all three slabs full
    EXPECT(s.carve(PG) == nullptr);
    for (int i = 0; i < 48; i += 2) s.put_back(v[i], 4 * PG);  // every other range free: 24 x 4 pages, no 8-page range
    for (auto& sl : s.slabs) EXPECT(kve::SlabSet::used_of(sl) == SLAB / 2);
    EXPECT(!s.claim(v[0], 8 * PG));     // (a free range of 4 pages followed by a used one)
    EXPECT(!s.claim(v[1], 4 * PG));     // (used)
    std::vector<char*> dst;
    EXPECT(!s.plan(s.slabs[0].base, {8 * PG}, {}, dst) && dst.empty());
    EXPECT(s.plan(s.slabs[0].base, std::vector<size_t>(8, 4 * PG), {}, dst));  // slab 0's 8 pieces into slabs 1, 2
    for (char* q : dst) EXPECT(q >= s.slabs[1].base);
    for (auto& sl : s.slabs) EXPECT(kve::SlabSet::used_of(sl) == SLAB / 2);  // (plan changed nothing)
    for (char* q : dst) EXPECT(s.claim(q, 4 * PG));
    EXPECT(!s.claim(dst[0], 4 * PG));  // (taken)
    for (int i = 1; i < 16; i += 2) s.put_back(v[i], 4 * PG);  // slab 0 empty now
    EXPECT(s.drop_free(SLAB, [](char*, size_t) {}) == SLAB);
    EXPECT(s.slabs.size() == 2);
    EXPECT(kve::SlabSet::used_of(s.slabs[0]) + kve::SlabSet::used_of(s.slabs[1]) == SLAB + SLAB / 2);
    for (int i = 17; i < 48; i += 2) s.put_back(v[i], 4 * PG);
    for (char* q : dst) s.put_back(q, 4 * PG);
    for (auto& sl : s.slabs) EXPECT(sl.free.size() == 1 && sl.free.begin()->second == SLAB);  // coalesced back to one range
}
// KV in RAM: a resident's rows dropped (drop_rows) keep its states as restore points while resident; once it leaves the slot
// without a new write, cut() drops the states past its (now no) rows
static void test_drop_rows() {
    ::Conv c;
    c.toks.assign(100, 1);
    auto g = std::make_shared<::Seg>();
    g->t0 = 0;
    g->t1 = 100;
    c.segs.push_back(g);
    c.kv_n = 100;
    auto k = std::make_shared<::State>();
    k->pos = 40;
    c.ck.push_back(k);
    c.end = std::make_shared<::State>();
    c.end->pos = 100;
    kve::drop_rows(c);
    EXPECT(c.segs.empty() && c.kv_n == 0 && c.ck.size() == 1 && c.end);
    EXPECT(kve::ck_usable(c, *c.ck[0], 60, true));    // resident: the rows are in its region
    EXPECT(!kve::ck_usable(c, *c.ck[0], 60, false));  // archived only: no rows
    kve::cut(c, c.kv_n);
    EXPECT(c.ck.empty() && !c.end && !kve::reachable(c));
    // a resident's states: all but its last turn end (the latest turn-end state, kind 4 or 6) are spare
    ::Conv m;
    for (int kd : {3, 4, 2, 4}) { auto x = std::make_shared<::State>(); x->kind = kd; x->pos = 10 * (int)m.ck.size() + 10; m.ck.push_back(x); }
    m.end = std::make_shared<::State>();
    EXPECT(kve::spare_states(m));
    kve::drop_spare_states(m);
    EXPECT(m.ck.size() == 1 && m.ck[0]->kind == 4 && m.ck[0]->pos == 40 && !m.end && !kve::spare_states(m));
    {  // a kind 4 at 20 and a kind 6 at 10: the later one (the 4) is the last turn end, whatever its kind
        ::Conv x;
        auto a = std::make_shared<::State>(); a->kind = 4; a->pos = 20;
        auto b = std::make_shared<::State>(); b->kind = 6; b->pos = 10;
        x.ck = {a, b};  // (the later state first in the vector: the choice is by position, not by vector order)
        EXPECT(kve::last_turn_end(x) == a.get());
        kve::drop_spare_states(x);
        EXPECT(x.ck.size() == 1 && x.ck[0] == a);
        auto c = std::make_shared<::State>(); c->kind = 6; c->pos = 30;  // a later kind 6 is the last turn end
        x.ck = {a, c};
        EXPECT(kve::last_turn_end(x) == c.get());
    }
    {  // neither kind (periodic 3, prompt end 2): no last turn end, all of them spare
        ::Conv x;
        auto a = std::make_shared<::State>(); a->kind = 3; a->pos = 20;
        auto b = std::make_shared<::State>(); b->kind = 2; b->pos = 30;
        x.ck = {a, b};
        EXPECT(kve::last_turn_end(x) == nullptr && kve::spare_states(x));
        kve::drop_spare_states(x);
        EXPECT(x.ck.empty() && !kve::spare_states(x));
    }
}
}  // namespace guard

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
    room::test_reclaim_order();
    room::test_refused_write();
    room::test_reachable();
    guard::test_give_back();
    guard::test_slabset();
    guard::test_drop_rows();
    fprintf(stderr, "%lld ops, %lld restores (%lld through a segment reaching past kv_n), %lld forks, %lld whole drops, %lld cuts, "
            "%lld refused writes\n", ops, restores, partial_seg, forks, drops, cuts, refused);
    EXPECT(restores > 1000 && forks > 100 && drops > 100 && cuts > 100 && partial_seg > 0);
    if (fails) { fprintf(stderr, "%d failures\n", fails); return 1; }
    fprintf(stderr, "all passed\n");
    return 0;
}
