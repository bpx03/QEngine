#pragma once
// FLASH_COMM=auto: whether the prefill transfers leave P2P (comm_select_path, comm_probe.cuh). Pure host logic, no
// CUDA: tests/comm_rule_unit.cpp checks it with a plain C++ compiler.
//
// P2P (pull / push) keeps compute on the memory system it shares with the transfer, so a slow link stalls it. The
// co-run tax (local compute slowdown next to the transfer, worst pair) counts against P2P when it is at or above the
// cap (QE_COMM_TAX, 1.30) and markedly above the copy engine's (> 1.15x). Cross-NUMA (or QE_COMM_XNUMA=0) the cap
// alone decides. On one NUMA node the tax proxy over-reads a fast PCIe link (~1.4x vs the real GEMM's ~1.1x; P2P pull
// is fastest end to end on our 3x3090), so there it decides only
//  - when it is severe (QE_COMM_TAX_SEVERE, 3.0; 2 cards on Gen2 x8 measured ~100x, CE ~1x), or
//  - when it reaches the 1 NUMA node cap (QE_COMM_TAX_1NODE, 1.90) on a narrow link (issue #2: 3090 + 3090 Ti, one
//    card in a gen4 x4 slot: P2P tax 2.11-2.17, CE 1.09, NaN in prefill on P2P, clean on the copy engine). The tax
//    alone does not separate that box from ours: our 3x3090 at x16 (249 boots 10-29/10-30, median of 5 runs) reads
//    p50 1.36-1.37, p99 1.55 / 1.84 (2 / 3 GPUs), but 2.00 and 2.17 on two boots, where the copy engine would cost
//    3-11 % prefill. Narrow (comm_narrow_link): a GPU of the set negotiated fewer than QE_COMM_NARROW_X (8) lanes, or
//    the P2P probe bandwidth is under QE_COMM_NARROW_GBPS (10; our x16 Gen4 26 GB/s, a Gen4 x4 / Gen3 x8 link ~6.5).
// P2P bandwidth far below the alternatives (bw_bad) always leaves P2P.
#include <algorithm>
#include <vector>

struct CommRule {
    bool leave_p2p;
    const char* why;
};

// min_lanes: narrowest negotiated PCIe link width of the set (0 or less: unknown, sysfs not readable); p2p_gbps: the
// chosen P2P mode's probe bandwidth, slowest pair
inline bool comm_narrow_link(int min_lanes, float p2p_gbps, int narrow_x, float narrow_gbps) {
    return (min_lanes > 0 && min_lanes < narrow_x) || p2p_gbps < narrow_gbps;
}

inline CommRule comm_auto_rule(float tax_p2p, float tax_ce, float cap, float cap1, float severe, bool cross_numa,
                               bool xnuma, bool narrow, bool bw_bad) {
    const bool over = tax_p2p >= cap && tax_p2p > tax_ce * 1.15f;
    const bool numa_rule = !xnuma || cross_numa;
    if (over && numa_rule) return {true, "P2P tax over the cap"};
    if (over && tax_p2p >= severe) return {true, "P2P tax severe (any NUMA layout)"};
    if (over && tax_p2p >= cap1 && narrow) return {true, "P2P tax over the 1 NUMA node cap on a narrow link"};
    if (bw_bad) return {true, "P2P bandwidth below the alternatives"};
    if (over && tax_p2p >= cap1) return {false, "P2P kept: tax over the 1 NUMA node cap, but no narrow link"};
    if (over) return {false, "P2P kept: tax over the cap, under the 1 NUMA node cap"};
    if (tax_p2p >= cap) return {false, "P2P kept: CE tax as high"};
    return {false, "P2P kept: tax under the cap"};
}

// Co-run tax over several probe runs (issue #2: one run on our box read 3.69 instead of 1.33-1.54 and flipped auto to
// CE): the median of the runs decides, the spread (min / max) goes to the log. Even counts: mean of the middle two.
struct CommStat {
    float med, lo, hi;
};
inline CommStat comm_median(std::vector<float> v) {
    if (v.empty()) return {1.f, 1.f, 1.f};
    std::sort(v.begin(), v.end());
    const size_t n = v.size();
    const float med = n % 2 ? v[n / 2] : 0.5f * (v[n / 2 - 1] + v[n / 2]);
    return {med, v.front(), v.back()};
}

// P2P data check (QE_COMM_CHECK, issue #2): the path after the tax rule's pick x (0 P2P, 1 copy engine, 2 host-staged,
// the kXfer* values of comm.cuh). bad_st: SM stores into a peer corrupted data on some pair (every P2P collective,
// decode included, stores into the peers: peer access goes off, host path); bad_ce: the copy engine did. SM loads
// (pull) that fail are handled per pair (that pair pushes, comm_check_pull), so they never change the path here.
struct CommCheckPick {
    int x;
    bool p2p_off;     // turn peer access off (comm_free_p2p): every collective through host memory
    const char* why;  // nullptr: the check changed nothing
};
inline CommCheckPick comm_check_pick(int x, bool bad_st, bool bad_ce) {
    if (bad_st) return {2, true, "P2P data check: SM stores corrupt data, peer access off (host memory path)"};
    if (x == 1 && bad_ce) return {2, false, "P2P data check: copy engine corrupts data, host-staged instead"};
    return {x, false, nullptr};
}
// Per directed pair: pull (SM loads by the receiver) or push (SM stores by the sender) after the check. want_pull: the
// probe's choice. A failing direction is never picked while the other one passed; both failing keeps want_pull (the
// path is left by comm_check_pick then).
inline bool comm_check_pull(bool want_pull, bool bad_ld, bool bad_st) {
    if (bad_ld && !bad_st) return false;
    if (bad_st && !bad_ld) return true;
    return want_pull;
}
