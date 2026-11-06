#pragma once
// Flash-Next: the one eviction order and the rules of an archive copy.
//
// Order, for every path that frees memory (archive room, host-RAM guard, VRAM slots, idle reclaim): least recently
// used conversation first; an active main (the tracker's main, used within FLASH_MAIN_STALE_S) after all the others.
// No guarantee: it goes when it must.
//
// A conversation's copy: segs in order, later ones overwrite earlier ones; kv_n, not a segment's t1, bounds the valid
// rows [0, kv_n) (a fork or a cut keeps a segment reaching past its own rows). A state serves a restore only at a
// position within kv_n whose tokens were compared with the prompt. Only a whole copy is ever dropped (drop_copy); cut
// is for the sequence itself getting shorter.
// Header only, so that tests/kv_evict_unit.cpp runs these on a model of the archive without a GPU.
#include <algorithm>
#include <cstddef>
#include <vector>

namespace kve {

struct Rank {
    bool main = false;            // active main: evicted last
    unsigned long long used = 0;  // last use (start or release), a tick
};
inline bool goes_first(const Rank& a, const Rank& b) { return a.main != b.main ? b.main : a.used < b.used; }
inline bool active_main(bool main_turn, double last_ms, double now_ms, double stale_s) {
    return main_turn && now_ms - last_ms < stale_s * 1000.0;
}

// A main turn that branched off `parent` at L moved the main there: the parent, the old branch, is no main any more (review M6:
// for FLASH_MAIN_STALE_S it outlived live subagents' conversations). Not when the parent is generating, and not for a branch at
// a small part of it: a second session sharing only the system prompt, each session keeps its own main.
inline bool main_moved(int L, int parent_len, bool parent_generating) { return !parent_generating && L * 2 >= parent_len; }

// The conversation whose archive copy goes next: the first in the order among those skip() lets go, none without a copy.
template <class L, class Skip, class RankOf>
auto* copy_victim(L& convs, Skip skip, RankOf rank) {
    decltype(&*convs.begin()) v = nullptr;
    for (auto& c : convs) {
        if (skip(c) || (c.segs.empty() && !c.end && c.ck.empty())) continue;
        if (!v || goes_first(rank(c), rank(*v))) v = &c;
    }
    return v;
}
// What evicting every conversation can_go() lets go would free. A blob is freed with its last holder: one that a conversation
// staying (or in kept_extra: a pinned prefix) also holds frees nothing.
template <class L, class CanGo, class Size>
size_t freeable(const L& convs, CanGo can_go, const std::vector<const void*>& kept_extra, Size size_of) {
    std::vector<const void*> kept = kept_extra, seen;
    auto each = [](const auto& c, auto fn) {
        for (const auto& g : c.segs) fn(g.get(), *g);
        if (c.end) fn(c.end.get(), *c.end);
        for (const auto& s : c.ck) fn(s.get(), *s);
    };
    for (const auto& c : convs)
        if (!can_go(c)) each(c, [&](const void* p, const auto&) { kept.push_back(p); });
    std::sort(kept.begin(), kept.end());
    size_t n = 0;
    for (const auto& c : convs)
        if (can_go(c))
            each(c, [&](const void* p, const auto& b) {
                if (std::binary_search(kept.begin(), kept.end(), p) || std::find(seen.begin(), seen.end(), p) != seen.end()) return;
                seen.push_back(p);
                n += size_of(b);
            });
    return n;
}
// Room for `need` more bytes under `limit`: evicts in the order until it fits; false when it cannot, and then nothing was
// evicted for it: a write that does not fit even with everything evictable gone costs no conversation (review M5).
// used(): the bytes held now; can_free(): what every possible eviction would free; next(): the next victim or null.
template <class Used, class CanFree, class Next, class Evict>
bool make_room(size_t need, size_t limit, Used used, CanFree can_free, Next next, Evict evict) {
    if (need > limit) return false;
    if (used() + need <= limit) return true;
    if (used() + need > limit + can_free()) return false;
    while (used() + need > limit) {
        auto* v = next();
        if (!v) return false;
        evict(v);
    }
    return true;
}

// The idle reclaim: which idle conversations leave VRAM for the expert cache, in the order. An ordinary one after `Tord` seconds
// idle. Active mains only under pressure, after max(Tord, Tmain), while together they hold at least 1/16 of the pool: then the
// oldest go until the rest hold less (review M7: several small mains together kept 42 % of the pool, each below the 1/16 of its
// own). A list on a snapshot of the ranks: the order cannot change during the sort.
struct IdleSlot {
    int slot = 0, cap = 0;  // cap: periods of its region
    double idle_s = 0;
    Rank rank;
};
inline std::vector<int> reclaim_order(std::vector<IdleSlot> v, bool pressure, double Tord, double Tmain, int np) {
    std::sort(v.begin(), v.end(), [](const IdleSlot& a, const IdleSlot& b) { return goes_first(a.rank, b.rank); });
    const auto main_ok = [&](const IdleSlot& q) { return pressure && Tmain > 0 && q.idle_s >= std::max(Tord, Tmain); };
    long long mains = 0;
    for (const auto& q : v)
        if (q.rank.main && main_ok(q)) mains += q.cap;
    std::vector<int> out;
    for (const auto& q : v) {
        if (!q.rank.main) {
            if (q.idle_s >= Tord) out.push_back(q.slot);
        } else if (main_ok(q) && mains * 16 >= np) {
            out.push_back(q.slot);
            mains -= q.cap;
        }
    }
    return out;
}

// A reclaim's archive write was refused: does this conversation leave VRAM anyway? Under pressure the first refused ordinary one
// of a pass does (user decision 2026-11-06), so a full archive of newer copies cannot keep the oldest one there; `ord` runs in the
// order, so that one is the first of what can leave VRAM. An older archived copy holds no VRAM and did not free enough, so it stays.
// At most one per pass: pressure is measured again before the next. Whatever the reason of the refusal, also bigger than the whole
// archive. Not an active main: larger than the archive it would re-prefill on every turn. Generating and paused ones are not in `ord`.
struct RefusedWrite {
    bool used = false;
    bool leaves(bool pressure, bool active_main) {
        if (!pressure || active_main || used) return false;
        return used = true;
    }
};

// the whole copy goes (a resident conversation keeps its state in the slot; its next sync writes all of it again)
template <class C>
void drop_copy(C& c) {
    c.segs.clear();
    c.kv_n = 0;
    c.end = nullptr;
    c.ck.clear();
}
// the sequence is n tokens long now: nothing of the copy past n stays valid (the next sync writes from there)
template <class C>
void cut(C& c, int n) {
    while (!c.segs.empty() && c.segs.back()->t0 >= n) c.segs.pop_back();
    c.kv_n = std::min(c.kv_n, n);
    c.ck.erase(std::remove_if(c.ck.begin(), c.ck.end(), [&](const auto& x) { return x->pos > n; }), c.ck.end());
    if (c.end && c.end->pos > n) c.end = nullptr;
}
// d: a new conversation branching off `from` at L (it shares the blobs; a segment may reach past L, kv_n bounds it)
template <class C>
void fork_from(C& d, const C& from, const int* ids, int L) {
    d.toks.assign(ids, ids + L);  // (reachable before its first complete sync)
    for (const auto& g : from.segs)
        if (g->t0 < L) d.segs.push_back(g);
    d.kv_n = std::min(from.kv_n, L);
    for (const auto& s : from.ck)
        if (s->pos <= L) d.ck.push_back(s);
    d.prio = from.prio;
}
// a new state of c, at most `keep` of them: drop the oldest periodic (3), divergence (5), turn-end (4), prompt-end (2), then
// system-end (1) state. The end of the main's last turn (6: a turn end while c's latest request was a main turn; one per
// conversation) goes last: a side task or fork extending c in place for many turns must not cost the main its resume point
template <class C, class P>
void add_ck(C& c, P s, int keep) {
    auto& v = c.ck;
    v.erase(std::remove_if(v.begin(), v.end(), [&](const auto& x) { return x->pos == s->pos; }), v.end());
    if (s->kind == 4 && c.prio >= 2) {
        for (auto& x : v)
            if (x->kind == 6) x->kind = 4;
        s->kind = 6;
    }
    v.push_back(s);
    while ((int)v.size() > keep) {
        int drop = -1;
        for (int want : {3, 5, 4, 2, 1, 6}) {
            for (int i = 0; i < (int)v.size() - 1; i++)
                if (v[i]->kind == want) { drop = i; break; }
            if (drop >= 0) break;
        }
        if (drop < 0) drop = 0;
        v.erase(v.begin() + drop);
    }
}
// can a later turn resume from it once it left VRAM? Not with no tokens to compare a prompt with: toks is empty until the first
// complete sync (review Low 1: a refused first write kept its bytes and logged a resume that could not happen)
template <class C>
bool reachable(const C& c) {
    return !c.toks.empty() && (c.end || !c.ck.empty());
}
// M: tokens of the conversation that match the prompt
template <class C>
bool end_usable(const C& c, int M) {
    return c.end && M == (int)c.toks.size() && c.end->pos == M && c.kv_n >= M;
}
// resident: the rows below a checkpoint are in the slot's region
template <class C, class S>
bool ck_usable(const C& c, const S& s, int M, bool resident) {
    return s.pos <= M && (resident || c.kv_n >= s.pos);
}

}  // namespace kve
