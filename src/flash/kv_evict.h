#pragma once
// Flash-Next: the one eviction order and the rules of an archive copy.
//
// Order: least recently used conversation first, every conversation alike (user decision 2026-11-12: no main priority; the
// scheduling value a request carries orders nothing here). The archive (archive room, host-RAM guard): only the ram-mode spare
// tiers go before (Rank::spare). VRAM slots and the idle reclaim: plain LRU.
//
// A conversation's copy: segs in order, later ones overwrite earlier ones; kv_n, not a segment's t1, bounds the valid
// rows [0, kv_n) (a fork or a cut keeps a segment reaching past its own rows). A state serves a restore only at a
// position within kv_n whose tokens were compared with the prompt. Only a whole copy is ever dropped (drop_copy; KV in RAM: a
// resident one's rows alone, drop_rows); cut
// is for the sequence itself getting shorter.
// Header only, so that tests/kv_evict_unit.cpp runs these on a model of the archive without a GPU.
#include <algorithm>
#include <cstddef>
#include <vector>

namespace kve {

struct Rank {
    unsigned long long used = 0;  // last use (start or release), a tick
    // the archive order (copies, FE::copy_rank). FLASH_KV=ram,
    // a conversation resident in the host KV pool: 2 its archived rows (they are in the pool too: dropping
    // them loses nothing, its next sync writes them again; its states stay), 1 its states but its last turn end (restore points
    // below the live end, a branch or a rewritten turn; drop_spare_states); spare copies go before any conversation, rows first.
    // 0: not spare (an archived conversation; a resident with only its last turn end left). An archived conversation is a whole
    // re-prefill when it returns, a resident's checkpoint only when it branches (repro: protecting the states lost 12 archived ones)
    int spare = 0;
};
inline bool goes_first(const Rank& a, const Rank& b) {
    if (a.spare != b.spare) return a.spare > b.spare;
    return a.used < b.used;
}

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

// The idle reclaim: which idle conversations leave VRAM for the expert cache, least recently used first: each after `Tord`
// seconds idle, every conversation alike (user decision 2026-11-12; before, an active main waited FLASH_EC_RECLAIM_MAIN_S, 30 s,
// under pressure, and only while the idle mains held >= 1/16 of the pool). A list on a snapshot of the ranks: the order cannot
// change during the sort.
struct IdleSlot {
    int slot = 0, cap = 0;  // cap: periods of its region
    double idle_s = 0;
    Rank rank;
    bool above = false;     // its region starts at or above the end of every generating one (KV in VRAM: it alone holds the top up)
};
// Under pressure an idle conversation above every generating region leaves before Tord: after Tabove (< 0: never) and,
// with min_size, only when it holds >= 1/16 of the pool (a finished subagent above the main)
inline bool at_once(const IdleSlot& q, bool pressure, double Tabove, bool min_size, int np) {
    return pressure && q.above && Tabove >= 0 && q.idle_s >= Tabove && (!min_size || (long long)q.cap * 16 >= np);
}
inline std::vector<int> reclaim_order(std::vector<IdleSlot> v, bool pressure, double Tord, int np, double Tabove = -1,
                                      bool min_size = false) {
    std::sort(v.begin(), v.end(), [](const IdleSlot& a, const IdleSlot& b) { return goes_first(a.rank, b.rank); });
    std::vector<int> out;
    for (const auto& q : v)
        if (q.idle_s >= Tord || at_once(q, pressure, Tabove, min_size, np)) out.push_back(q.slot);
    return out;
}

// A reclaim's archive write was refused: does this conversation leave VRAM anyway? Under pressure the first refused one (any
// conversation) of a pass does (user decision 2026-11-06), so a full archive of newer copies cannot keep the oldest one there; `ord` runs in the
// order, so that one is the first of what can leave VRAM. An older archived copy holds no VRAM and did not free enough, so it stays.
// At most one per pass: pressure is measured again before the next. Whatever the reason of the refusal, also bigger than the whole
// archive. Every conversation alike (user decision 2026-11-12, no main priority: a session's main leaves as any other, with the
// partial copy it has; before, it stayed in VRAM pass after pass, #1928). Generating and paused ones are not in `ord`.
struct RefusedWrite {
    bool used = false;
    bool leaves(bool pressure) {
        if (!pressure || used) return false;
        return used = true;
    }
};

// Host memory below the margin (FE::mem_guard): give back what the deficit asks for, losing as little as possible. Order: whole
// free slabs (release), slabs emptied by moving their blobs into the free ranges of the others (compact: nothing is lost), then
// copies in the one order. An eviction's bytes lower the deficit only once a slab is empty (they are free ranges inside slabs
// others still use) or the allocator gives pageable pages back; evicting on until MemAvailable moved lost 10-150x the deficit
// (1.3.5.9: a 0.1 GB deficit cost 10 conversations, 15.8 GB, 0 GB returned). So, bounded:
//  - no eviction while compaction waits for a holder (a disk write, a disk read, a prompt's checkpoints keep bytes in a slab) and
//    the pinned slabs' free ranges (room()) already cover the deficit: a later pass returns them once the holder let go;
//  - at most the deficit + one slab evicted per pressure episode (`ep`: kept by the caller across passes until memory went back
//    or the deficit is gone), counting each victim's own bytes (an eviction that frees nothing, its blobs shared with a fork, a
//    pinned prefix or a disk write, still counts);
//  - urgent (deficit above `urgent`, the host close to its limit): no wait, at most the deficit + one slab in this pass and twice
//    the deficit + one slab in the episode (victims whose blobs a disk write holds free nothing now; review N4).
// deficit(): bytes below the margin now; release(d): whole free slabs back; compact(d): 1 returned a slab, 0 cannot (the pieces do
// not fit: evicting more frees ranges), -1 waits for a holder; room(): free
// bytes inside the pinned slabs; evict(cost): one copy in the order goes, its own bytes in `cost`, false when none is left.
// bounded = false: no limit (FLASH_ARCHIVE_GUARD_BOUND=0). Returns the bytes evicted in this pass.
template <class Deficit, class Release, class Compact, class Room, class Evict>
size_t give_back(Deficit deficit, Release release, Compact compact, Room room, Evict evict, size_t& ep, size_t slab, size_t urgent,
                 bool bounded = true) {
    size_t now = 0, d;
    release(deficit());
    while ((d = deficit()) > 0) {
        const int c = compact(d);
        if (c > 0) continue;
        if (bounded) {
            if (d > urgent) {
                if (now >= d + slab || ep >= 2 * d + slab) break;
            } else if ((c < 0 && room() >= d) || ep >= d + slab) {
                break;
            }
        }
        size_t f = 0;
        if (!evict(f)) break;
        now += f;
        ep += f;
        release(deficit());
    }
    return now;
}

// KV in RAM, a resident conversation's states but its last turn end (the turn-end state, kind 4 or 6, at the highest position:
// where its latest turn ended, the resume point for a regenerated or edited reply): the states a spare eviction of level 1
// drops (spare_states: whether there are any). Every resident alike
template <class C>
const void* last_turn_end(const C& c) {
    const void* k = nullptr;
    int pos = -1;
    for (const auto& s : c.ck)
        if ((s->kind == 4 || s->kind == 6) && s->pos > pos) { pos = s->pos; k = s.get(); }
    return k;
}
template <class C>
bool spare_states(const C& c) {
    if (c.end) return true;
    const void* k = last_turn_end(c);
    for (const auto& s : c.ck)
        if (s.get() != k) return true;
    return false;
}
template <class C>
void drop_spare_states(C& c) {
    const void* k = last_turn_end(c);
    c.end = nullptr;
    c.ck.erase(std::remove_if(c.ck.begin(), c.ck.end(), [&](const auto& s) { return s.get() != k; }), c.ck.end());
}
// KV in RAM, a resident conversation: only its archived rows go (the pool holds them); its states stay, they are valid restore
// points while it is resident (ck_usable: the rows below them are in its region); the next sync writes all rows again
template <class C>
void drop_rows(C& c) {
    c.segs.clear();
    c.kv_n = 0;
}

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
// the states of one kind go (a fork's parent keeps its own references)
template <class C>
void drop_kind(C& c, int kind) {
    c.ck.erase(std::remove_if(c.ck.begin(), c.ck.end(), [&](const auto& x) { return x->kind == kind; }), c.ck.end());
}
// a new state of c, at most `keep` of them: drop the oldest periodic (3), divergence (5), turn-end (4), prompt-end - 1 (2),
// then system-end (1) state; a prompt end (7) only as the last resort. c's latest turn end (last_turn_end: the turn-end state
// at the highest position) goes last, for every conversation alike (user decision 2026-11-12): without that each turn's
// prompt-end state would push out the one turn end left (4 goes before 2), and the ram-mode spare drop keeps exactly that
// state. It is protected here, not marked on the state: a fork shares its parent's State objects, so a mark would let one
// conversation's turn demote the other's (review L2). A kind 6 (the mark of earlier builds, read back from a disk-cache file)
// counts as a turn end (4).
// A prompt-end state (kind 7, [qfn-ttft-pe]: the state after all rows of a prompt, the next turn's resume point when the
// client sends another text for the reply) replaces the conversation's earlier one: a state per turn would push out the
// system end, periodic and divergence states (keep) and grow every archived conversation (64 agent sessions on the 48 GB
// archive: ~1 GB each, live sessions dropped, 287 old rows recomputed per turn).
template <class C, class P>
void add_ck(C& c, P s, int keep) {
    auto& v = c.ck;
    v.erase(std::remove_if(v.begin(), v.end(), [&](const auto& x) { return x->pos == s->pos || (s->kind == 7 && x->kind == 7); }),
            v.end());
    v.push_back(s);
    while ((int)v.size() > keep) {
        const void* k = last_turn_end(c);
        int drop = -1;
        for (int want : {3, 5, 4, 2, 1}) {
            for (int i = 0; i < (int)v.size() - 1; i++)
                if (v[i].get() != k && (v[i]->kind == want || (want == 4 && v[i]->kind == 6))) { drop = i; break; }
            if (drop >= 0) break;
        }
        for (int i = 0; drop < 0 && i < (int)v.size() - 1; i++)
            if (v[i].get() != k) drop = i;
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
