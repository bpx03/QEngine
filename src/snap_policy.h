// Prompt-cache snapshot eviction (DeltaNet state checkpoints in pinned host buffers), header-only
// so that it can be checked on the CPU in isolation.
//
// Two kinds share the store: checkpoints (kind 0: every 8192 prompt tokens, the divergence mark,
// the live end before a fork) and prompt-end states (kind 1). Each kind has its own quota
// (checkpoints: `reserve0` entries, prompt ends: the rest); a full store evicts the least
// recently used entry of the kind that is at or over its quota, so
//  - a long prompt's many checkpoints cycle among themselves and never flush the prompt ends of
//    other conversations (the goal of the kind order introduced in 5077cc8), and
//  - prompt ends can no longer starve the checkpoints: before, once the store held only prompt
//    ends, every new checkpoint was the first victim of the next snapshot (usually the same
//    prompt's own end state), so a prompt sharing a long prefix with an earlier one reused 0.
#pragma once
#include <cstddef>

// Within the kind, the lowest value goes first, then the least recently used: value(entry) is the
// conversation's priority (main agent 2, normal 1, finished subagent 0) plus 1 for a state past
// 32K tokens within 8K of its slot's current end (expensive to recompute: a near-max-context
// conversation's end state and latest checkpoints stay while short prompts come and go; its
// older turns' checkpoints do not).
// S: sequence of entries with `.kind` (0/1) and `.used` (LRU tick). Returns the index of the entry
// to overwrite with a new snapshot of kind `kind`, the store being full (size >= 1).
template <typename S, typename V>
size_t snap_victim(const S& snaps, int kind, int reserve0, V value) {
    const size_t total = snaps.size();
    size_t n0 = 0;
    for (const auto& s : snaps) n0 += s.kind == 0;
    const size_t n1 = total - n0;
    const size_t q0 = (size_t)(reserve0 < 0 ? 0 : reserve0) < total ? (size_t)reserve0 : total;
    // the kind to evict: the new snapshot's own kind when it is at or over its quota, else the
    // other kind (over its quota since the store is full); fall back to whatever exists
    int k;
    if (kind == 0) k = n0 >= q0 ? 0 : 1;
    else k = n1 >= total - q0 ? 1 : 0;
    if ((k == 0 && n0 == 0) || (k == 1 && n1 == 0)) k = 1 - k;
    size_t best = total;
    int bv = 0;
    for (size_t i = 0; i < total; i++) {
        if (snaps[i].kind != k) continue;
        const int v = value(snaps[i]);
        if (best == total || v < bv || (v == bv && snaps[i].used < snaps[best].used)) best = i, bv = v;
    }
    return best == total ? 0 : best;
}
