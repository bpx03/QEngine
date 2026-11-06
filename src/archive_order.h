#pragma once
// 27B: the order the host-RAM archive frees memory in, the same rule as Flash-Next:
// least recently used conversation first; an active main (the tracker's main, prio 2, used within FLASH_MAIN_STALE_S)
// after all the others. No guarantee: it goes when nothing else can. Only a whole archive is ever dropped, and a restore
// compares its tokens with the prompt, so an eviction cannot make a restore read wrong rows.
// Header only, so that tests/archive_order_unit.cpp runs it on a model of the archive without a GPU.

namespace qe_arch {

struct Rank {
    bool main = false;            // active main: evicted last
    double last_s = 0;            // the conversation's last use (start or release); 0: found on disk, not used since
    unsigned long long used = 0;  // tie-break, the archive's tick (files found on disk: oldest modification first)
};
inline bool goes_first(const Rank& a, const Rank& b) {
    if (a.main != b.main) return b.main;
    return a.last_s != b.last_s ? a.last_s < b.last_s : a.used < b.used;
}
inline bool active_main(int prio, double last_s, double now_s, double stale_s) {
    return prio >= 2 && now_s - last_s < stale_s;
}
// Not evictable: no memory copy (on disk only, or the archive being loaded for a restore), a pinned prefix (its own
// budget), one a restore or a paused slot holds. An archive being written is not in the list yet.
template <class A>
bool evictable(const A& a) {
    return a.ram && !a.pinned && !a.hold;
}
// The next archive to go; `end`: none.
template <class L, class R>
auto victim(L& l, const R& rank) -> decltype(l.begin()) {
    auto v = l.end();
    for (auto it = l.begin(); it != l.end(); ++it)
        if (evictable(*it) && (v == l.end() || goes_first(rank(*it), rank(*v)))) v = it;
    return v;
}
// Bytes a write may free, checked first so that a write that cannot fit evicts nothing. older_than (an optional write): only
// what goes before the writer; victim() takes those first, so a write that fits on them touches nothing else.
template <class L, class R>
unsigned long long can_free(const L& l, const R& rank, const Rank* older_than = nullptr) {
    unsigned long long b = 0;
    for (const auto& a : l)
        if (evictable(a) && (!older_than || goes_first(rank(a), *older_than))) b += a.total;
    return b;
}

}  // namespace qe_arch
