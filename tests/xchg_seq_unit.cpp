// CPU unit test of the 27B KV-split exchange flag protocol (src/kernels/xchg_seq.h, review rounds 4 S1 / 5 T2-T3): one
// directed pair, the set of each launch unit chosen by xchg_set (the function Worker::begin_unit and xchg_fill call),
// one epoch counter per set (Worker::xep_), the compare by xchg_arrived (the function the kernels call). For every
// exchange the receiver must wait (not arrived) until the sender writes, then see it arrived: with long stretches on
// one set, path switches (two sets -> one -> two), a set re-enabled after any idle time, epochs across 2^31 and the
// 2^32 wrap. Also shows the old shared-counter scheme failing (the reason for per-set counters).
// Built by CMake (ctest) or: g++ -std=c++17 -O2 -Isrc tests/xchg_seq_unit.cpp -o /tmp/xsu && /tmp/xsu
#include "kernels/xchg_seq.h"

#include <cstdio>
#include <initializer_list>

static int fails = 0;
static bool quiet = false;
#define EXPECT(c)                                                                                       \
    do {                                                                                                \
        if (!(c)) { if (fails < 10 && !quiet) fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); fails++; } \
    } while (0)

struct Sim {  // the worker's per-set scheme
    unsigned ep[2], flag[2];
    bool xh;  // both decode modes on host memory (Worker::xh_)
    explicit Sim(unsigned start) : ep{start, start}, flag{xchg_seq_of(start, 0) - 1u, xchg_seq_of(start, 0) - 1u}, xh(true) {}
    void unit(bool in_pf, int nx) {
        const int s = xchg_set(xh, in_pf);
        ep[s]++;  // begin_unit: the unit's set only
        for (int x = 0; x < nx; x++) {
            const unsigned want = xchg_seq_of(ep[s], x);
            EXPECT(!xchg_arrived(flag[s], want));  // before the sender's write: the receiver waits
            flag[s] = want;
            EXPECT(xchg_arrived(flag[s], want));
        }
    }
};

int main() {
    EXPECT(xchg_set(true, false) == 1 && xchg_set(true, true) == 0 && xchg_set(false, false) == 0 && xchg_set(false, true) == 0);
    for (unsigned start : {0u, 0x7FFFF000u, 0xFFFFF000u / 4096u, 0x000FFFF0u, 0xFFFFFFF0u}) {
        Sim m(start);
        for (int round = 0; round < 2; round++) {
            for (int u = 0; u < 8; u++) m.unit(true, 1 + u % 3);          // a prefill: the device set
            for (int u = 0; u < (3 << 20); u++) m.unit(false, 1 + u % 3);  // 3M decode units: the host set
        }
        m.unit(true, 2);                                                // the next prefill after the long decode
        m.xh = false;                                                   // two sets -> one (P2P pick)
        for (int u = 0; u < (3 << 20); u++) m.unit(u % 97 == 0, 1);     // 3M units on the device set
        m.xh = true;                                                    // -> two again: the host set idle for 3M units
        m.unit(false, 3);
        m.unit(true, 1);
    }
    // the old shared-counter scheme fails on this pattern: the idle set lags by >= 2^31
    {
        quiet = true;
        const int f0 = fails;
        unsigned shared = 0, flag = 0;
        auto unit = [&](bool use) {
            shared++;
            if (!use) return;
            const unsigned want = xchg_seq_of(shared, 0);
            EXPECT(!xchg_arrived(flag, want));
            flag = want;
        };
        unit(true);
        for (int u = 0; u < (1 << 19) + 16; u++) unit(false);  // 2^19 + 16 units on the other set
        unit(true);
        const bool broke = fails > f0;
        fails = f0;
        quiet = false;
        if (!broke) { fprintf(stderr, "FAIL: the shared-counter model did not show the wrap\n"); fails++; }
    }
    if (fails) return 1;
    printf("xchg_seq_unit: ok\n");
    return 0;
}
