#pragma once
// NaN guard abort rule (FE::fault, flash_capi.cpp). Pure host logic, no CUDA: tests/guard_rule_unit.cpp checks it with a
// plain C++ compiler.
//
// A reply whose forward produced non-finite values fails alone. Many faults within the window (more than
// FLASH_GUARD_MAX, default 4, within 5 minutes) point at shared state (weights, expert cache) and end the process. A
// deterministic fault of one prompt repeats at the same position every time that prompt is sent again (issue #2: a
// client re-sent it and the server died), so a fault counts toward the limit only once per key: the token history up
// to the faulting position, the position and the kind of step. Distinct faults still count.
#include <cstddef>
#include <cstdint>
#include <vector>

// FNV-1a over the token ids [0, n), the position and the kind ("prefill", "decode step", ...)
inline uint64_t guard_key(const int* toks, size_t n, int pos, const char* what) {
    uint64_t h = 1469598103934665603ull;
    auto mix = [&](uint32_t v) {
        for (int b = 0; b < 4; b++) h = (h ^ ((v >> (8 * b)) & 0xFFu)) * 1099511628211ull;
    };
    for (size_t i = 0; i < n; i++) mix((uint32_t)toks[i]);
    mix((uint32_t)pos);
    mix((uint32_t)n);
    for (const char* p = what; p && *p; p++) h = (h ^ (unsigned char)*p) * 1099511628211ull;
    return h;
}

struct GuardEvent {
    double t_ms;
    uint64_t key;
    int hits;  // faults with this key in the window (the first counts, the repeats do not)
};

struct GuardVerdict {
    bool repeat;    // the key already faulted within the window: not counted again
    bool abort;     // more than fmax distinct faults within the window
    int distinct;   // distinct faults within the window (after this one)
    int hits;       // faults of this key within the window (after this one)
};

// Records a fault at t_ms with `key`; drops events older than window_ms. A repeat refreshes its key's time, so a
// prompt re-sent forever stays one entry.
inline GuardVerdict guard_note(std::vector<GuardEvent>& ev, double t_ms, uint64_t key, int fmax,
                               double window_ms = 300e3) {
    std::vector<GuardEvent> keep;
    keep.reserve(ev.size() + 1);
    for (const GuardEvent& e : ev)
        if (e.t_ms >= t_ms - window_ms) keep.push_back(e);
    ev.swap(keep);
    GuardVerdict v{false, false, 0, 1};
    for (GuardEvent& e : ev)
        if (e.key == key) {
            v.repeat = true;
            e.t_ms = t_ms;
            v.hits = ++e.hits;
        }
    if (!v.repeat) ev.push_back({t_ms, key, 1});
    v.distinct = (int)ev.size();
    v.abort = v.distinct > fmax;
    return v;
}
