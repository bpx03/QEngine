#pragma once
// CUDA-graph instantiation under VRAM pressure (club-3090#1846). The only device memory a Flash-Next worker allocates
// after start-up is cached graph executables (decode / MTP / speculative / constrained / batched / prefill graphs,
// several hundred shapes per card); the expert budget at start leaves FLASH_EC_MARGIN_MB (256) free, and the graphs
// of a server that sees many conversation shapes grow into it until an instantiate fails. The worker then drops every
// cached graph and retries; if that still fails the step runs without a graph (same kernels: graphs and eager runs mix
// across ranks, the collectives' epochs are device-side) and the next kCoolSteps graph captures are skipped.
// The driver keeps the memory of destroyed graph executables in its own pool: cudaMemGetInfo does not show it as
// free again, but later instantiates and cudaMalloc get it back (measured 2026-10-29),
// so free memory is no signal for dropping graphs early (a headroom watermark only thrashed the graph cache).
// Test hooks (off by default): FLASH_TEST_GRAPH_OOM=a,b,... makes the a-th, b-th, ... instantiate of each card (1-based)
// fail as out of memory; FLASH_TEST_GRAPH_OOM_HARD=1 makes the retry fail too (eager path); FLASH_TEST_GRAPH_OOM_RANK=r
// limits both to rank r (default: every rank).
// No CUDA dependency: tests/graph_oom_unit.cpp builds this header with a plain C++ compiler.
#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <vector>

namespace graph_oom {

constexpr int kCoolSteps = 64;  // graph captures skipped after a failed retry

// device memory of an instantiated graph executable, estimated from its nodes (child-graph bodies included; a
// conditional node's body is not reachable through the runtime API). Measured on RTX 3090 / CUDA 13.1 / driver
// 615.71 (54 fresh instantiates): 7.0-7.7 KB per node, taken in 2 MB steps
// (decode / batched / speculative graphs ~1,100-1,300 nodes = 8-12 MB, a big prefill batch up to 6,100 nodes = 46 MB).
// 8 KB per node + one 2 MB step: never below the measured cost, so the budget is not overrun
constexpr size_t kNodeBytes = 8192, kGraphBytes = (size_t)2 << 20;
inline size_t est_bytes(size_t nodes, size_t /*kernels*/) { return kGraphBytes + nodes * kNodeBytes; }

// the graph budget of a card in MB (FLASH_GRAPH_MB), added to the expert budget's margin at load. Agent traffic on
// 3 cards (1.3.4-rc2 gate shape) keeps ~17 decode-side graphs (~170 MB) plus the 4 big prefill batch graphs of the
// newest prompt lengths (up to 46 MB each); 256 MB holds the decode set and the newest prefill batches, older prefill
// batch graphs (one prompt length each, rarely replayed) are destroyed first
constexpr int kGraphMbDefault = 256;
// the part of the graph budget charged to FLASH_EC_MARGIN_MB (at most half of it), where every graph lived up to
// 1.3.4; the rest is added to the margin. At the peak of the hermes-like stress the full budget on top left 262 MB
// free on GPU0 (the user margin untouched), and 2 cards lost 1.6 % decode (c=1) to it against 0.5 % on 3 cards
constexpr int kGraphInMarginMb = 128;

// "3,7, 12" -> {3, 7, 12}; empty / null -> {}; junk entries are skipped
inline std::vector<int> parse_list(const char* s) {
    std::vector<int> v;
    if (!s) return v;
    for (const char* q = s; *q;) {
        char* end = nullptr;
        const long x = strtol(q, &end, 10);
        if (end != q) {
            v.push_back((int)x);
            q = end;
        }
        while (*q && *q != ',') q++;  // the rest of the entry (junk or a number's tail)
        if (*q) q++;
    }
    return v;
}

struct Inject {
    std::vector<int> at;  // instantiate calls (1-based, per card) that fail
    bool hard = false;    // the retry fails too
    int rank = -1;        // -1: every rank
    bool first(int call, int r) const {
        return (rank < 0 || rank == r) && std::find(at.begin(), at.end(), call) != at.end();
    }
    bool retry(int call, int r) const { return hard && first(call, r); }
};

// skip graph captures for a while after a failed retry (a card that cannot hold one more graph would otherwise
// capture + fail every step)
struct Cool {
    int left = 0;
    void trip() { left = kCoolSteps; }
    bool skip() {  // called when a graph is missing: true = run eagerly without capturing
        if (left <= 0) return false;
        left--;
        return true;
    }
};

}  // namespace graph_oom
