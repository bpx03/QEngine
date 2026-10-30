// CPU-only unit test of the FLASH_COMM=auto tax rule (src/kernels/comm_rule.h). Built and run by tests/test_env.py.
#include "kernels/comm_rule.h"

#include <cstdio>
#include <initializer_list>
#include <string>

static int fails = 0;
#define EXPECT(c)                                                     \
    do {                                                              \
        if (!(c)) { fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); fails++; } \
    } while (0)

int main() {
    // QE_COMM_TAX, QE_COMM_TAX_1NODE, QE_COMM_TAX_SEVERE defaults; R(): 1 NUMA node, XNUMA rule on, bandwidth ok
    const float cap = 1.30f, cap1 = 1.90f, sev = 3.0f;
    auto R = [&](float t, float ce, bool narrow) { return comm_auto_rule(t, ce, cap, cap1, sev, false, true, narrow, false); };
    // narrow link: fewer than 8 lanes on some GPU, or P2P probe bandwidth under 10 GB/s
    EXPECT(!comm_narrow_link(16, 26.2f, 8, 10.f));  // our 3x3090 (x16 Gen4, 26 GB/s)
    EXPECT(comm_narrow_link(4, 6.5f, 8, 10.f));     // the reporter: one card in a gen4 x4 slot
    EXPECT(comm_narrow_link(4, 26.f, 8, 10.f));     // the lanes alone
    EXPECT(comm_narrow_link(0, 6.5f, 8, 10.f));     // sysfs unreadable (container): the bandwidth alone
    EXPECT(!comm_narrow_link(0, 26.f, 8, 10.f));
    EXPECT(!comm_narrow_link(8, 12.f, 8, 10.f));    // x8 Gen4 / x16 Gen3: not narrow
    EXPECT(comm_narrow_link(16, 26.f, 17, 10.f));   // QE_COMM_NARROW_X=17: every link counts (test hook)
    // our 3x3090 (1 NUMA node, x16): P2P tax medians 1.33-1.84 typical, 2.00 / 2.17 on 2 of 249 boots -> P2P stays
    for (float t : {1.33f, 1.36f, 1.39f, 1.54f, 1.62f, 1.84f, 2.00f, 2.17f}) EXPECT(!R(t, 1.05f, false).leave_p2p);
    {
        const CommRule r = R(2.17f, 1.03f, false);
        EXPECT(std::string(r.why).find("no narrow link") != std::string::npos);
    }
    // issue #2 reporter (3090 + 3090 Ti, gen4 x4, 1 NUMA node): P2P tax 2.17 / 2.11, CE 1.09 -> copy engine
    EXPECT(R(2.17f, 1.09f, true).leave_p2p);
    EXPECT(R(2.11f, 1.09f, true).leave_p2p);
    EXPECT(std::string(R(2.17f, 1.09f, true).why).find("narrow link") != std::string::npos);
    // narrow link but a normal tax: P2P stays
    EXPECT(!R(1.50f, 1.05f, true).leave_p2p);
    // the 1 NUMA node cap edge (narrow)
    EXPECT(R(1.90f, 1.0f, true).leave_p2p);
    EXPECT(!R(1.89f, 1.0f, true).leave_p2p);
    // severe on any link (2 cards on Gen2 x8: P2P tax ~100x, CE ~1x) -> leave P2P; 2.9 on x16 -> P2P
    EXPECT(R(100.f, 1.0f, false).leave_p2p);
    EXPECT(R(3.0f, 1.0f, false).leave_p2p);
    EXPECT(!R(2.9f, 1.0f, false).leave_p2p);
    // high but the copy engine as bad: P2P stays
    EXPECT(!R(100.f, 95.f, false).leave_p2p);
    EXPECT(!R(2.17f, 1.95f, true).leave_p2p);
    // cross-NUMA: the cap decides (as before)
    EXPECT(comm_auto_rule(1.37f, 1.05f, cap, cap1, sev, true, true, false, false).leave_p2p);
    EXPECT(!comm_auto_rule(1.20f, 1.05f, cap, cap1, sev, true, true, false, false).leave_p2p);
    // QE_COMM_XNUMA=0: the cap decides on one node
    EXPECT(comm_auto_rule(1.37f, 1.05f, cap, cap1, sev, false, false, false, false).leave_p2p);
    // slow P2P bandwidth always leaves
    EXPECT(comm_auto_rule(1.0f, 1.0f, cap, cap1, sev, false, true, false, true).leave_p2p);
    // co-run tax median (issue #2): our boots 1.33-1.54 with one 3.69 outlier -> the median stays under the caps
    {
        const CommStat m = comm_median({1.36f, 3.69f, 1.33f, 1.54f, 1.35f});
        EXPECT(m.med == 1.36f && m.lo == 1.33f && m.hi == 3.69f);
        EXPECT(!R(m.med, 1.05f, false).leave_p2p);
        EXPECT(comm_median({1.f, 2.f, 3.f, 10.f}).med == 2.5f);
        EXPECT(comm_median({}).med == 1.f);
        // the reporter's runs (2.11-2.17) on the x4 link -> copy engine
        EXPECT(R(comm_median({2.11f, 2.17f, 2.14f}).med, 1.09f, true).leave_p2p);
    }
    // P2P data check: per pair, a failing direction is dropped while the other one passed
    EXPECT(!comm_check_pull(true, true, false));   // loads corrupt -> push
    EXPECT(comm_check_pull(false, false, true));   // stores corrupt -> pull
    EXPECT(comm_check_pull(true, false, false) && !comm_check_pull(false, false, false));
    EXPECT(comm_check_pull(true, true, true));     // both corrupt: unchanged (comm_check_pick leaves P2P)
    // path: SM stores corrupt -> peer access off (host); CE corrupt while picked -> host-staged; else unchanged
    {
        CommCheckPick k = comm_check_pick(0, true, false);
        EXPECT(k.x == 2 && k.p2p_off && k.why);
        k = comm_check_pick(1, false, true);
        EXPECT(k.x == 2 && !k.p2p_off && k.why);
        k = comm_check_pick(0, false, true);  // CE corrupt but P2P picked: unchanged
        EXPECT(k.x == 0 && !k.p2p_off && !k.why);
        k = comm_check_pick(1, false, false);
        EXPECT(k.x == 1 && !k.why);
    }
    if (fails) return 1;
    printf("comm_rule_unit: ok\n");
    return 0;
}
