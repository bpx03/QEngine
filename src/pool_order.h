#pragma once
// Both engines: the orders the KV pool picks sequences in, every conversation alike (user decision 2026-11-12: no priority;
// a request's scheduling value orders nothing here). Header only, so that tests/pool_order_unit.cpp checks them without a GPU.
//  - pause (room_or_pause, pause_for): the most recently admitted decoding sequence first;
//  - resume: in pause order, first paused first (strictly head-of-line at the full pool);
//  - 27B slot overwrite (pick_slot): the cached context cheapest to lose, fewest tokens, then least recently used;
//  - 27B background archive (archive_more): the longest idle conversation first, the next one when its write is refused.
#include <algorithm>
#include <vector>

namespace qe_pool {

// admit / pause ticks are unique per engine, so these orders are strict
inline bool pauses_first(unsigned long long admit_a, unsigned long long admit_b) { return admit_a > admit_b; }
inline bool resumes_first(unsigned long long paused_a, unsigned long long paused_b) { return paused_a < paused_b; }
inline bool overwrite_first(int pos_a, unsigned long long used_a, int pos_b, unsigned long long used_b) {
    return pos_a != pos_b ? pos_a < pos_b : used_a < used_b;
}
// slots in an order: key(i) gives the value the comparator reads; stable, so equal keys keep the input order
template <class Key, class Less>
std::vector<int> ordered(std::vector<int> slots, Key key, Less less) {
    std::stable_sort(slots.begin(), slots.end(), [&](int a, int b) { return less(key(a), key(b)); });
    return slots;
}
template <class Admit>
std::vector<int> pause_order(std::vector<int> slots, Admit admit) {
    return ordered(std::move(slots), admit, pauses_first);
}
template <class PauseTick>
std::vector<int> resume_order(std::vector<int> slots, PauseTick paused) {
    return ordered(std::move(slots), paused, resumes_first);
}
// the first slot in `slots` (longest first; ties: the lower slot) whose write try_slot accepts; -1: none accepted
template <class Pos, class Try>
int longest_accepted(std::vector<int> slots, Pos pos, Try try_slot) {
    std::stable_sort(slots.begin(), slots.end(), [&](int a, int b) { return pos(a) > pos(b); });
    for (int s : slots)
        if (try_slot(s)) return s;
    return -1;
}

}  // namespace qe_pool
