// CPU-only unit test of the NaN guard abort rule (src/flash/guard_rule.h). Built and run by tests/test_env.py.
#include "flash/guard_rule.h"

#include <cstdio>
#include <vector>

static int fails = 0;
#define EXPECT(c)                                                     \
    do {                                                              \
        if (!(c)) { fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); fails++; } \
    } while (0)

int main() {
    const int fmax = 4;
    std::vector<int> a(83248), b(83248);
    for (size_t i = 0; i < a.size(); i++) a[i] = (int)(i * 7 % 151000), b[i] = a[i];
    b[100] ^= 1;
    const uint64_t ka = guard_key(a.data(), a.size(), 83248, "prefill");
    // the key: same tokens / position / kind -> same; any of them differs -> differs
    EXPECT(ka == guard_key(a.data(), a.size(), 83248, "prefill"));
    EXPECT(ka != guard_key(b.data(), b.size(), 83248, "prefill"));
    EXPECT(ka != guard_key(a.data(), a.size() - 1, 83247, "prefill"));
    EXPECT(ka != guard_key(a.data(), a.size(), 83248, "batch step"));

    {  // issue #2: the same prompt re-sent 20 times within 5 minutes never aborts, each repeat is reported
        std::vector<GuardEvent> ev;
        GuardVerdict v = guard_note(ev, 0.0, ka, fmax);
        EXPECT(!v.repeat && !v.abort && v.distinct == 1 && v.hits == 1);
        for (int i = 1; i < 20; i++) {
            v = guard_note(ev, i * 1000.0, ka, fmax);
            EXPECT(v.repeat && !v.abort && v.distinct == 1 && v.hits == i + 1);
        }
    }
    {  // distinct faults still count: the 5th distinct one within 5 minutes aborts (as before)
        std::vector<GuardEvent> ev;
        for (uint64_t k = 1; k <= 4; k++) EXPECT(!guard_note(ev, k * 1000.0, k, fmax).abort);
        EXPECT(guard_note(ev, 5000.0, ka, fmax).abort);
    }
    {  // repeats mixed with distinct faults: only the distinct ones count
        std::vector<GuardEvent> ev;
        for (int i = 0; i < 10; i++) EXPECT(!guard_note(ev, i * 100.0, ka, fmax).abort);
        for (uint64_t k = 1; k <= 3; k++) EXPECT(!guard_note(ev, 2000.0 + k, k, fmax).abort);
        EXPECT(guard_note(ev, 3000.0, 99, fmax).distinct == 5);
    }
    {  // the window: faults older than 5 minutes no longer count
        std::vector<GuardEvent> ev;
        for (uint64_t k = 1; k <= 4; k++) guard_note(ev, k * 1000.0, k, fmax);
        const GuardVerdict v = guard_note(ev, 3999.0 + 300e3, 5, fmax);  // 1..3 dropped, 4 kept
        EXPECT(!v.abort && v.distinct == 2);
        // a key seen again after it left the window counts once more (as a new fault)
        const GuardVerdict w = guard_note(ev, 700e3, 1, fmax);
        EXPECT(!w.repeat && w.hits == 1);
    }
    {  // a repeat refreshes the key's time: a prompt re-sent every 4 minutes stays one entry
        std::vector<GuardEvent> ev;
        for (int i = 0; i < 10; i++) {
            const GuardVerdict v = guard_note(ev, i * 240e3, ka, fmax);
            EXPECT(v.distinct == 1 && v.repeat == (i > 0));
        }
    }
    {  // FLASH_GUARD_MAX=0: the first fault aborts
        std::vector<GuardEvent> ev;
        EXPECT(guard_note(ev, 0.0, ka, 0).abort);
    }
    if (fails) return 1;
    printf("guard_rule_unit: ok\n");
    return 0;
}
