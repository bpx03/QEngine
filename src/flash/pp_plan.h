#pragma once
// Pipeline-parallel (PP) planning for the Flash-Next engine: pure host logic, no CUDA, so tests/qfn_pp_plan_unit.cpp
// checks it with a plain C++ compiler.
//
// FLASH_PAR selects how the cards split the model:
//   tp          tensor-parallel (today's default): every card holds a slice of every layer, all cards work on every
//               token (2 or 3 cards: make_plan).
//   pp_prefill  mode A: prompts run layer-pipelined over N >= 2 stages (each card holds whole layers), decode stays TP
//               on the first T in {2, 3} cards; the prompt's KV and recurrent state move to the TP layout at its end.
//   pp          mode B: prompts and decode steps both run layer-pipelined over N >= 2 stages; several sequences are
//               interleaved across the stages so every card works (c >= N).
//   pptp        PP x TP: mode B whose stages are tensor-parallel groups of t <= 3 cards (FLASH_PP_MAP=0,1|2,3 or
//               FLASH_PP_TP=t): each stage's layers are split TP-wise over its cards (all-reduces inside the group), the
//               residual rows hop from every rank of a stage to the ranks of the next.
// Nothing here assumes 2 or 3 cards: partitions, schedules and budgets take any N in [2, kLayers].
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <limits>
#include <string>
#include <vector>

namespace flash_pp {

enum class Par { TP = 0, PP_PREFILL = 1, PP = 2, AUTO = 3, PPTP = 4 };

inline const char* par_name(Par p) {
    return p == Par::TP ? "tp" : p == Par::PP_PREFILL ? "pp_prefill" : p == Par::PP ? "pp" : p == Par::PPTP ? "pptp" : "auto";
}

// FLASH_PAR value -> mode; null / empty: TP (flash_run; the server passes auto_default = true: null / empty = AUTO, the
// boot-time pick below). false (and *err) on an unknown value.
inline bool parse_par(const char* s, Par* out, std::string* err = nullptr, bool auto_default = false) {
    if (!s || !*s) { *out = auto_default ? Par::AUTO : Par::TP; return true; }
    if (!strcmp(s, "tp")) { *out = Par::TP; return true; }
    if (!strcmp(s, "auto")) { *out = Par::AUTO; return true; }
    if (!strcmp(s, "pp_prefill") || !strcmp(s, "a")) { *out = Par::PP_PREFILL; return true; }
    if (!strcmp(s, "pp") || !strcmp(s, "b")) { *out = Par::PP; return true; }
    if (!strcmp(s, "pptp")) { *out = Par::PPTP; return true; }
    if (err) *err = std::string("FLASH_PAR=") + s + ": expected auto, tp, pp_prefill, pp or pptp";
    return false;
}

// ---------------------------------------------------------------- model shape (mirrors flash.h; static_asserts there)
constexpr int kLayers = 48;
constexpr int kQsaEvery = 4;   // layers 3, 7, ..., 47 are QSA (full attention), the others DeltaNet
constexpr int kPleLayer = 1;   // the n-gram embedding (PLE) block sits in layer 1
constexpr int kRW = 4 * 2560;  // the hyper-connection residual row (4 streams x 2560) crossing a stage boundary
constexpr double kHopBytesPerRow = (double)kRW * 4;  // fp32: 40,960 B per token per boundary
inline bool is_qsa(int li) { return li % kQsaEvery == kQsaEvery - 1; }

// Bytes of the INT4 checkpoint as the engine lays it out (sizes from the loader: GRP_BYTES = 506,880 per 128-wide
// expert group, 513 experts x 5 groups per layer; dense parts from the per-card numbers of the PP=2 gate,
// scaled to the whole layer).
struct ModelBytes {
    double expert_layer = 2565.0 * 506880.0;  // 1.300 GB: all expert groups of one layer (routed + shared)
    double gdn_dense = 30.5e6;                // DeltaNet in / out projections, a / b, conv, norms (one whole layer)
    double qsa_dense = 38.3e6;                // QSA q / k / v / gate / o, indexer projection, norms
    double hc_layer = 15.2e6;                 // the two hyper-connection reads of a layer (INT8, replicated in TP)
    double router_layer = 1.4e6;              // INT8 router (528 x 2560)
    double ple = 16.5e6;                      // layer 1: PLE key / value projections + norms + conv
    double lm_head = 0.636e9;                 // INT8 lm_head (248,320 x 2560)
    double draft_head = 0.168e9;              // balanced draft head (65,536 rows)
    double mtp_dense = 0.07e9;                // MTP block: its QSA layer, fc_embedding / fc_hidden, HC, router
    double mtp_experts = 1.300e9;             // its expert groups
    double kv_tok_layer = 1040.0 + 64.0;      // one QSA layer, one token: fp8 K / V + scales (1,040 B) + pooled indexer key
    double dn_state_layer = 48.0 * 128 * 128 * 4 + 3.0 * (2 * 16 * 128 + 48 * 128) * 4;  // DeltaNet S + conv history, one sequence
    double row_buf = 72.0 * 1024;             // prefill batch buffers per row (R, HC sums, x / xh, part, y, router, n-gram rows, MoE slots)
    double ctx_overhead = 0.9e9;              // CUDA context, cuBLAS / graph workspaces, small tables per card
};

// Prefill cost per layer, ms per 1K prompt tokens of ONE card doing the whole layer (TP measurements x 2 cards:
// 2-GPU nsys 10K / 90K, per card: MoE 92.8 / 88.8 ms/1K over 48 + MTP
// layers, QSA attention + indexer + exchanges 47 / 70 over 12 layers, GDN GEMM + chunk kernels 42 / 37 over 36 layers,
// HC 19 over 48). The exchanges inside a TP layer vanish in PP; they stay in the numbers as a margin.
struct CostModel {
    double moe = 2 * 92.8 / 49;       // 3.79 ms/1K per layer (whole layer on one card)
    double gdn = 2 * 42.3 / 36;       // 2.35
    double qsa_10k = 2 * 47.0 / 12;   // 7.83 at a 10K context
    double qsa_90k = 2 * 70.4 / 12;   // 11.7 at 90K (block selection + attention grow with the context)
    double hc = 2 * 19.6 / 48;        // 0.82
    double head_last = 2.0;           // final mixer + lm_head of the batch's last row + MTP fill (approx. one QSA mixer)
    double mtp_layer = 0.0;           // MTP block over the prompt rows (set from moe + qsa when the MTP fill runs)
    double ple_first = 1.0;           // PLE block (n-gram rows) on the stage that holds layer 1
    // ms per 1K tokens of layer li at a mean context ctx (tokens)
    double layer(int li, double ctx) const {
        double mix = gdn;
        if (is_qsa(li)) {
            const double f = std::clamp((ctx - 10000.0) / 80000.0, 0.0, 1.5);
            mix = qsa_10k + f * (qsa_90k - qsa_10k);
        }
        return moe + mix + hc + (li == kPleLayer ? ple_first : 0.0);
    }
};

// Decode cost per layer in layer-equivalents of a verify step (m = 4 rows): every trunk layer reads its dense weights and
// the touched experts about alike (QSA ~1.1x a DeltaNet layer). The last stage also runs the final mixer + full lm_head
// (636 MB INT8), the draft head per MTP draft (168 MB) and the MTP block per draft: ~1.7-2.0 ms on a ~5-7 ms stage step
// of 24 layers, i.e. ~5 layer-equivalents at k = 3.
struct DecodeCost {
    double gdn = 1.0, qsa = 1.1;
    double head = 5.0;     // last stage, k = 3 MTP drafts
    double layer(int li) const { return is_qsa(li) ? qsa : gdn; }
};

// ---------------------------------------------------------------- stage partition
// Contiguous layer ranges for n stages: bounds[s] .. bounds[s + 1]. stage_time(s, l0, l1) is the time of stage s
// running layers [l0, l1) (it may include memory-bound terms, the head on the last stage, ...); the partition minimizes
// the slowest stage (the pipeline's throughput) by dynamic programming over all contiguous splits (n x L^2), ties to
// the earlier cut. Every stage gets at least one layer.
inline std::vector<int> partition(int n, int L, const std::function<double(int, int, int)>& stage_time) {
    if (n < 1 || n > L) return {};
    const double INF = std::numeric_limits<double>::infinity();
    // f[s][j]: best max stage time of stages 0..s-1 covering layers [0, j); arg[s][j]: start of stage s - 1
    std::vector<std::vector<double>> f(n + 1, std::vector<double>(L + 1, INF));
    std::vector<std::vector<int>> arg(n + 1, std::vector<int>(L + 1, -1));
    f[0][0] = 0.0;
    for (int s = 1; s <= n; s++)
        for (int j = s; j <= L - (n - s); j++)
            for (int i = s - 1; i < j; i++) {
                if (f[s - 1][i] == INF) continue;
                const double v = std::max(f[s - 1][i], stage_time(s - 1, i, j));
                if (v < f[s][j] - 1e-12) { f[s][j] = v; arg[s][j] = i; }
            }
    std::vector<int> b(n + 1);
    b[n] = L;
    for (int s = n; s >= 1; s--) b[s - 1] = arg[s][b[s]];
    return b;
}

// Partition for prefill and decode at once: stage cost = (1 - w) x its share of the prefill time + w x its share of the
// decode time (each normalized by its mean per stage), the max over stages minimized. w = 0: prefill only (mode A, whose
// decode is TP); mode B: w = FLASH_PP_DECODE_W (default 0.5).
inline std::vector<int> partition_mixed(int n, const CostModel& cm, double ctx, const DecodeCost& dc, double w) {
    double pf_tot = cm.head_last, dc_tot = dc.head;
    for (int li = 0; li < kLayers; li++) { pf_tot += cm.layer(li, ctx); dc_tot += dc.layer(li); }
    return partition(n, kLayers, [&](int s, int l0, int l1) {
        double pf = 0, de = 0;
        for (int li = l0; li < l1; li++) { pf += cm.layer(li, ctx); de += dc.layer(li); }
        if (s == n - 1) { pf += cm.head_last; de += dc.head; }
        return (1 - w) * pf / (pf_tot / n) + w * de / (dc_tot / n);
    });
}

// Heterogeneous cards: the same blend with each stage's time divided by its card's speed (relative to the reference
// card the cost model was fit on: pf = prefill compute, dc = decode memory bandwidth, from the boot calibration) plus
// the prefill staging of the experts that do not fit its card's free VRAM (per 4096-row micro-batch at h2d).
struct CardSpeed { double pf = 1.0, dc = 1.0, free = 22e9; };
inline bool cards_uneven(const std::vector<CardSpeed>& c, double tol = 0.10, double mem_tol = 2e9) {
    for (const CardSpeed& x : c)
        for (const CardSpeed& y : c)
            if (x.pf > y.pf * (1 + tol) || x.dc > y.dc * (1 + tol) || x.free > y.free + mem_tol) return true;
    return false;
}
struct ModelBytes;
inline double stage_host_bytes(int l0, int l1, bool last, double free_bytes);
inline std::vector<int> partition_cards(int n, const CostModel& cm, double ctx, const DecodeCost& dc, double w,
                                        const std::vector<CardSpeed>& cs, double h2d) {
    double pf_tot = cm.head_last, dc_tot = dc.head;
    for (int li = 0; li < kLayers; li++) { pf_tot += cm.layer(li, ctx); dc_tot += dc.layer(li); }
    return partition(n, kLayers, [&](int s, int l0, int l1) {
        double pf = 0, de = 0;
        for (int li = l0; li < l1; li++) { pf += cm.layer(li, ctx); de += dc.layer(li); }
        if (s == n - 1) { pf += cm.head_last; de += dc.head; }
        const CardSpeed& c = cs[s];
        pf /= std::max(0.05, c.pf);
        de /= std::max(0.05, c.dc);
        pf += stage_host_bytes(l0, l1, s == n - 1, c.free) / h2d * 1e3 / 4.096;  // ms per 1K prompt tokens
        return (1 - w) * pf / (pf_tot / n) + w * de / (dc_tot / n);
    });
}

// FLASH_PP_LAYERS=a,b,c,...: layers per stage (as many numbers as stages, each >= 1, summing to kLayers) -> bounds;
// empty: none given. *err on a malformed value.
inline std::vector<int> parse_layers(const char* s, int n, std::string* err) {
    std::vector<int> b;
    if (!s || !*s) return b;
    b.push_back(0);
    const char* p = s;
    while (*p) {
        char* e = nullptr;
        const long v = strtol(p, &e, 10);
        if (e == p || v < 1) { *err = std::string("FLASH_PP_LAYERS=") + s + ": positive layer counts expected"; return {}; }
        b.push_back(b.back() + (int)v);
        p = e;
        if (*p == ',') p++;
        else if (*p) { *err = std::string("FLASH_PP_LAYERS=") + s + ": comma-separated counts expected"; return {}; }
    }
    if ((int)b.size() != n + 1 || b.back() != kLayers) {
        char m[160];
        snprintf(m, sizeof m, "FLASH_PP_LAYERS=%s: %d stages need %d counts summing to %d", s, n, n, kLayers);
        *err = m;
        return {};
    }
    return b;
}

// ---------------------------------------------------------------- PP x TP stage groups (FLASH_PAR=pptp)
// A stage is a TP group of t cards (1 <= t <= kMaxGroupTp). The groups come from FLASH_PP_MAP ("0,1|2,3": device
// indices, '|' between stages, ',' inside one; a device may appear in two stages, a test of the split on fewer cards,
// but not twice in one group) or from FLASH_PP_TP=t over the device list (consecutive groups of t, the last one takes
// the rest). Returns the groups (empty + *err on a malformed map).
constexpr int kMaxGroupTp = 3;   // make_plan: 2 or 3 TP ranks (TP4 is a separate item)
constexpr int kMaxWorkers = 16;  // the server's per-rank tables (run() monitor)
inline std::vector<std::vector<int>> parse_stage_map(const char* map, const std::vector<int>& devs, int tp, std::string* err) {
    std::vector<std::vector<int>> g;
    auto fail = [&](const std::string& m) { *err = m; return std::vector<std::vector<int>>{}; };
    if (map && *map) {
        std::vector<int> cur;
        const char* p = map;
        while (true) {
            char* e = nullptr;
            const long v = strtol(p, &e, 10);
            if (e == p || v < 0) return fail(std::string("FLASH_PP_MAP=") + map + ": device indices expected (e.g. 0,1|2,3)");
            for (int x : cur)
                if (x == (int)v) return fail(std::string("FLASH_PP_MAP=") + map + ": a device twice in one stage");
            cur.push_back((int)v);
            p = e;
            if (*p == ',') { p++; continue; }
            g.push_back(cur);
            cur.clear();
            if (*p == '|') { p++; continue; }
            if (*p) return fail(std::string("FLASH_PP_MAP=") + map + ": ',' inside a stage, '|' between stages");
            break;
        }
    } else {
        if (tp < 1) return fail("FLASH_PP_TP: a group size >= 1 expected");
        for (size_t i = 0; i < devs.size(); i += tp) {
            std::vector<int> cur;
            for (size_t j = i; j < std::min(devs.size(), i + (size_t)tp); j++) cur.push_back(devs[j]);
            g.push_back(cur);
        }
        // a 1-card remainder after bigger groups stays its own stage (e.g. 5 cards, t = 2: 2 | 2 | 1)
    }
    if (g.size() < 2) return fail("FLASH_PAR=pptp: at least 2 stages expected (FLASH_PP_MAP / FLASH_PP_TP)");
    int nw = 0;
    for (auto& x : g) {
        if ((int)x.size() > kMaxGroupTp) {
            char m[160];
            snprintf(m, sizeof m, "FLASH_PAR=pptp: a stage of %zu cards (TP groups hold 1 to %d cards)", x.size(), kMaxGroupTp);
            return fail(m);
        }
        nw += (int)x.size();
    }
    if (nw > kMaxWorkers) return fail("FLASH_PAR=pptp: more than 16 ranks");
    // one KV period for every stage (make_stage_plans): TP groups of one size, plus single cards
    int T = 1;
    for (auto& x : g) T = std::max(T, (int)x.size());
    for (auto& x : g)
        if (x.size() != 1 && (int)x.size() != T) return fail("FLASH_PAR=pptp: TP groups of one size expected (single cards may mix in)");
    return g;
}
inline std::string stage_map_str(const std::vector<std::vector<int>>& g) {
    std::string o;
    for (size_t s = 0; s < g.size(); s++) {
        if (s) o += "|";
        for (size_t r = 0; r < g[s].size(); r++) o += (r ? "," : "") + std::to_string(g[s][r]);
    }
    return o;
}
// Relative speed of a TP group of t cards vs one card running the same layers (fits on 3 x 3090, P2P):
//   decode (c = 1 verify step of 4 rows): one card's step = replicated R 5.2 ms + split S 9.8 ms; a TP-t step =
//   R + S / t + all-reduces 0.38 / 0.45 / 0.54 ms (t = 2 / 3 / 4) -> 1.43 / 1.68 / 1.83;
//   prefill: TP prefill fit A + B / t (10K / 90K) against mode B's 0.290 card-ms per token -> 1.78 / 2.1 / 2.3.
// Without peer access the TP group's collectives are host-staged: decode +0.4 ms of the 15 ms (x 0.97), prefill
// +0.047 ms per token (x 0.78 at t = 2..3).
inline double group_speed_decode(int t, bool p2p = true) {
    static const double v[5] = {1.0, 1.0, 1.43, 1.68, 1.83};
    const double x = v[std::clamp(t, 1, 4)];
    return t > 1 && !p2p ? x * 0.97 : x;
}
inline double group_speed_prefill(int t, bool p2p = true) {
    static const double v[5] = {1.0, 1.0, 1.78, 2.1, 2.3};
    const double x = v[std::clamp(t, 1, 4)];
    return t > 1 && !p2p ? x * 0.78 : x;
}
// The cards of the groups as CardSpeed entries for partition_cards: a group's speed = the TP speed-up x its slowest
// card (TP waits on it every layer), its free memory = the sum of its cards' (the stage's expert groups spread over
// them). cards: per device of the group (empty: equal 3090s).
inline std::vector<CardSpeed> group_speeds(const std::vector<std::vector<int>>& g, const std::vector<CardSpeed>& per_group_min,
                                           bool p2p = true) {
    std::vector<CardSpeed> v;
    for (size_t s = 0; s < g.size(); s++) {
        const int t = (int)g[s].size();
        CardSpeed c = s < per_group_min.size() ? per_group_min[s] : CardSpeed{};
        c.pf *= group_speed_prefill(t, p2p);
        c.dc *= group_speed_decode(t, p2p);
        c.free *= t;
        v.push_back(c);
    }
    return v;
}

// ---------------------------------------------------------------- per-stage memory
struct Card {
    double vram = 24e9;      // bytes
    double h2d = 24e9;       // host -> device bytes/s (expert staging, KV in RAM)
};
struct StageMem {
    double fixed = 0;        // dense weights, HC, router, head / MTP dense, buffers, context
    double experts = 0;      // expert bytes the stage's layers need
    double resident = 0;     // of them in VRAM
    double host = 0;         // host-backed (staged per prefill batch, zero-copy / cached in decode)
    double kv_tok = 0;       // VRAM bytes per context token of this stage (its QSA layers, all positions)
    double kv_tokens = 0;    // tokens of KV that fit next to the minimum resident experts (the pool)
};
// Stage s of n holding layers [l0, l1): rows = prefill batch rows (its buffers), kv_reserve = VRAM kept for the KV
// pool, ring = staging ring bytes. The last stage holds lm_head, the draft head and (mtp) the MTP block.
inline StageMem stage_mem(const ModelBytes& mb, const Card& c, int l0, int l1, bool last, bool mtp, int rows,
                          double kv_reserve, double ring) {
    StageMem m;
    int nq = 0;
    for (int li = l0; li < l1; li++) {
        nq += is_qsa(li);
        m.fixed += (is_qsa(li) ? mb.qsa_dense : mb.gdn_dense) + mb.hc_layer + mb.router_layer + (li == kPleLayer ? mb.ple : 0.0);
        m.experts += mb.expert_layer;
    }
    if (last) {
        m.fixed += mb.lm_head + mb.draft_head + 2 * mb.hc_layer;  // + the final mixer
        if (mtp) { m.fixed += mb.mtp_dense; m.experts += mb.mtp_experts; nq++; }
    }
    m.fixed += mb.ctx_overhead + rows * mb.row_buf + ring;
    m.kv_tok = nq * mb.kv_tok_layer;
    const double room = std::max(0.0, c.vram - m.fixed - kv_reserve);
    m.resident = std::min(m.experts, room);
    m.host = m.experts - m.resident;
    m.kv_tokens = m.kv_tok > 0 ? (kv_reserve + std::max(0.0, room - m.experts)) / m.kv_tok : 0;
    return m;
}

// expert bytes of a stage that do not fit next to its fixed part in free_bytes (host-backed, staged per micro-batch)
inline double stage_host_bytes(int l0, int l1, bool last, double free_bytes) {
    const ModelBytes mb;
    return stage_mem(mb, Card{free_bytes, 24e9}, l0, l1, last, true, 8704, 0.0, 2.0e9).host;
}

// A TP card of t ranks (today's layout: 1/t of every layer's expert groups and dense heads, HC / router replicated,
// lm_head split, MTP split): the same fields.
inline StageMem tp_mem(const ModelBytes& mb, const Card& c, int t, bool mtp, int rows, double kv_reserve, double ring) {
    StageMem m;
    for (int li = 0; li < kLayers; li++) {
        m.fixed += (is_qsa(li) ? mb.qsa_dense : mb.gdn_dense) / t + mb.hc_layer + mb.router_layer + (li == kPleLayer ? mb.ple : 0.0);
        m.experts += mb.expert_layer / t;
    }
    m.fixed += (mb.lm_head) / t + mb.draft_head / t + 2 * mb.hc_layer;
    if (mtp) { m.fixed += mb.mtp_dense; m.experts += mb.mtp_experts / t; }
    m.fixed += mb.ctx_overhead + rows * mb.row_buf + ring;
    m.kv_tok = (kLayers / kQsaEvery + (mtp ? 1 : 0)) * mb.kv_tok_layer / t;
    const double room = std::max(0.0, c.vram - m.fixed - kv_reserve);
    m.resident = std::min(m.experts, room);
    m.host = m.experts - m.resident;
    m.kv_tokens = m.kv_tok > 0 ? (kv_reserve + std::max(0.0, room - m.experts)) / m.kv_tok : 0;
    return m;
}

// ---------------------------------------------------------------- prefill schedule
// The pipeline runs a prompt of P tokens as micro-batches (b rows each; a tail of <= 512 rows joins the last batch, as
// PpEngine::prefill cuts): stage s runs batch j after stage s - 1 handed it over and after its own batch j - 1. The wall
// is simulated exactly under the model (uneven stages included), not a closed formula: the fill is the first FULL batch
// through stages 0..N-2, and every batch boundary costs the hop.
//   batch time on a stage: compute b x u / 1000 ms, or the staging of its host-backed experts (host / h2d) when that is
//   longer (the ring copies overlap compute); below stage_min rows the engine reads them zero-copy instead (no staging,
//   FLASH_PF_STAGE_MIN): compute + host / h2d, not overlapped (pessimistic)
//   hop: legs x rows x 40,960 B / link (2 legs through pinned host: D2H then H2D; 1 leg peer-to-peer); with overlap
//   (copy stream, per-512-row chunk events) only one chunk's copy is exposed per leg
struct PfStage {
    double u = 0;          // compute ms per 1K rows
    double host = 0;       // host-backed expert bytes staged per batch
    double h2d = 24e9;
    double fixed = 0;      // ms per batch whatever its rows: every resident expert group read once (bytes / HBM), the
                           // launches of the batch graph (pf_fixed_ms); it is what keeps all-resident stages off tiny batches
};
inline double pf_fixed_ms(double resident_bytes, double hbm = 900e9, double launch_ms = 3.0) {
    return resident_bytes / hbm * 1e3 + launch_ms;
}
struct HopModel {
    double link = 22e9;    // bytes / s of one leg
    int legs = 2;          // 2: through pinned host, 1: peer-to-peer
    bool overlap = false;  // copy stream with chunk events
    int stage_min = 2048;  // rows from which the ring stages (FLASH_PF_STAGE_MIN)
};
inline double pf_batch_ms(const PfStage& s, int b, int stage_min = 2048) {
    const double c = b * s.u / 1000.0 + s.fixed, h = s.h2d > 0 ? s.host / s.h2d * 1e3 : 0.0;
    return b < stage_min ? c + h : std::max(c, h);
}
inline double pf_hop_ms(int b, const HopModel& hm) {
    const int rows = hm.overlap ? std::min(b, 512) : b;
    return hm.legs * rows * kHopBytesPerRow / hm.link * 1e3;
}
inline std::vector<int> pf_cuts(long long P, int b) {  // the driver's cuts (PpEngine::prefill): b rows, a <= 512-row tail joins
    std::vector<int> c;
    for (long long i = 0; i < P;) {
        const long long r = P - i;
        const int m = (int)(r <= b + 512 ? r : b);
        c.push_back(m);
        i += m;
    }
    return c;
}
inline double pf_wall_ms(const std::vector<PfStage>& st, long long P, int b, const HopModel& hm) {
    if (st.empty() || b <= 0 || P <= 0) return 0;
    const std::vector<int> cut = pf_cuts(P, b);
    const int n = (int)st.size(), nb = (int)cut.size();
    std::vector<double> fin(n, 0.0);  // finish time of each stage's latest batch
    for (int j = 0; j < nb; j++)
        for (int s = 0; s < n; s++) {
            const double ready = s ? fin[s - 1] + pf_hop_ms(cut[j], hm) : 0.0;  // fin[s - 1] is batch j there already
            fin[s] = std::max(ready, fin[s]) + pf_batch_ms(st[s], cut[j], hm.stage_min);
        }
    return fin[n - 1];
}
// The best micro-batch rows among multiples of 512 in [lo, hi] (the engine's batch buffers bound hi).
inline int pick_microbatch(const std::vector<PfStage>& st, long long P, int lo, int hi, const HopModel& hm) {
    int best = lo;
    double bt = std::numeric_limits<double>::infinity();
    for (int b = lo; b <= hi; b += 512) {
        const double t = pf_wall_ms(st, P, b, hm);
        if (t < bt - 1e-9) { bt = t; best = b; }
        if (b >= P) break;
    }
    return best;
}


// ---------------------------------------------------------------- per-request pick: TP prefill vs pipelined prefill
// TP prefill of P tokens in passes (batches) of up to B rows on t cards (all cards in parallel, so one card's time):
//   per pass of b rows: max(compute b x u + fixed, restaging the card's host-backed experts host / h2d) + the exposed part
//   of its exchanges (b x xb bytes per token per card over the link at xlink) - the exchanges overlap compute except
//   `expose` of them.
// The floors of a run (bench tables): staging = passes x host / h2d, exchange = P x xb / xlink.
struct TpModel {
    double u = 0.157;      // ms per row per pass of compute (2 x 3090 fit: 90K 15.06 s in 11 passes at a 24.8 GB/s host link)
    double fixed = 3.0;    // ms per pass (launches)
    double host = 14.4e9;  // host-backed expert bytes per card, restaged every pass
    double h2d = 24e9;     // host -> card bytes / s (probe)
    double xb = 555e3;     // exchange bytes per token per card (2 cards, QX_LOCAL + MOE_AGQ; pp2_gate step 2)
    double xlink = 26e9;   // per-card bytes / s of the exchange path (P2P / CE / host-staged, probe)
    double expose = 0.3;   // fraction of the exchange time not hidden behind compute
    int B = 8192;          // rows per pass (big batches)
};
inline int tp_passes(long long P, int B) { return (int)pf_cuts(P, B).size(); }
inline double tp_prefill_ms(const TpModel& m, long long P) {
    double t = 0;
    for (int b : pf_cuts(P, m.B)) {
        const double comp = b * m.u + m.fixed, stg = m.h2d > 0 ? m.host / m.h2d * 1e3 : 0.0;
        t += std::max(comp, stg) + m.expose * b * m.xb / m.xlink * 1e3;
    }
    return t;
}
inline double tp_staging_floor_ms(const TpModel& m, long long P) { return tp_passes(P, m.B) * (m.h2d > 0 ? m.host / m.h2d * 1e3 : 0.0); }
inline double tp_exchange_floor_ms(const TpModel& m, long long P) { return P * m.xb / m.xlink * 1e3; }
// Pipelined prefill time from the stages' model (pf_wall_ms) and the TP time: the pick is PP when it is faster by more
// than `margin` (the hand-over and a cold start are not in the model). Returns the smallest prompt (tokens, step 1024)
// from which PP wins and keeps winning up to max_p, or -1 when it never does.
inline long long pp_threshold(const std::function<double(long long)>& tp_ms, const std::function<double(long long)>& pp_ms,
                              long long max_p = 262144, double margin = 0.05) {
    long long th = -1;
    for (long long p = 1024; p <= max_p; p += 1024) {
        const bool win = pp_ms(p) * (1.0 + margin) < tp_ms(p);
        if (win && th < 0) th = p;
        if (!win) th = -1;
    }
    return th;
}

// ---------------------------------------------------------------- decode interleave (mode B)
// c running sequences on n stages: g = min(c, n + 1) groups in flight, sequences dealt round robin (sizes differ by at
// most one). At any moment stage s works on one group; n + 1 groups keep every stage busy across the host turn-around
// between a group's last stage and its next first stage (tokens back, drafts, the next step's inputs). Returns the
// group of every sequence.
inline int decode_ngroups(int c, int n) { return std::max(1, std::min(c, n + 1)); }
inline std::vector<int> decode_groups(int c, int n) {
    std::vector<int> g(std::max(c, 0));
    const int ng = decode_ngroups(c, n);
    for (int i = 0; i < c; i++) g[i] = i % ng;
    return g;
}
// Steady-state group steps per second of the interleaved pipeline: g groups circulate over n stages, a group's step
// takes n stage times + n hops + the host turn-around (round); the rate is g per round, at most one group step per
// slowest-stage time (+ hop) once the stages never idle.
inline double decode_group_steps_per_s(int g, int n, double stage_ms, double hop_ms, double host_ms) {
    const double round = n * (stage_ms + hop_ms) + host_ms;
    const double bound = (double)g / round;                               // groups per ms limited by the round trip
    const double pipe = 1.0 / (stage_ms + hop_ms);                        // one group per stage time at best
    return 1e3 * std::min(bound, pipe);
}

// ---------------------------------------------------------------- boot-time pick (server FLASH_PAR=auto)
// TP or mode B for the whole run, from the startup probes (peer access, the host link, the path the TP exchanges take)
// and a cost model of a typical agent request at concurrency c (default 4 and 8): its uncached prompt (P tokens) and its
// reply (D tokens). Per request the machine spends prefill(P) + D / aggregate(c) seconds; mode B is picked only when the
// mean over the mix's concurrencies (1, 4, 8; capped at the server's sequences) is lower by more than `margin`.
// Calibration (3 x 3090, PCIe 4.0 x16, 2026-11-01):
//   TP prefill with P2P: 2 cards 10K 1.65 s / 90K 15.06 s, 3 cards 1.36 / 13.08 s; without P2P (host-staged exchanges)
//   3 cards 3.75 / 12.55 (32K) / 34.95 s: +0.244 ms per prompt token; TP decode c=1 step 9.63 ms (P2P) vs 12.72 ms
//   (host-staged), +3.1 ms per step. Mode B prefill 2 cards 2.33 / 13.80 s, 3 cards 1.57 / 9.30 s (with or without
//   P2P: 1.51 / 9.62 s, the hop is hidden). Decode aggregates (tok/s, p2k prompt, k = 3) below.
// The TP exchange inputs come from the probed path: P2P none, copy engine small, host staging a per-token and a per-step
// cost scaled by the probed host rate (the TP no-P2P path is being reworked: FLASH_PAR_TP_XMS / FLASH_PAR_TP_XSTEP override
// both until the model is re-fit).
struct PickIn {
    int n = 3;                    // cards
    int tp_path = 0;              // the TP exchanges' path (comm.cuh kXferP2P 0, kXferCE 1, kXferHost 2)
    bool p2p = true;              // peer access between the cards (mode B's hop goes peer-to-peer)
    double h2d = 24.8e9;          // host -> card bytes / s (probe_host_link, the slowest card)
    double vram = 24e9;           // bytes per card (smallest)
    double tp_x_ms_tok = -1;      // exposed TP exchange cost per prompt token (ms); < 0: from the path and h2d
    double tp_x_ms_step = -1;     // the same per decode step (ms); < 0: from the path
    long long P = 8192;           // uncached prompt tokens of a typical request
    int D = 512;                  // reply tokens
    std::vector<int> cs = {1, 4, 8};  // concurrencies of the mix (the server caps them at its sequences)
    std::vector<CardSpeed> cards; // boot calibration, relative to the median card (empty: equal cards)
    double uneven_tol = 0.25;     // max / min speed above 1 + this: the cards count as uneven (TP waits on the slowest)
    double margin = 0.10;
    int rounds = 4;               // mode B decode rounds per server call (pipeline fill / drain once per call) when
    bool persist = true;          // the server's decode pipeline is not persistent (FLASH_PPB_PERSIST=0)
    bool tp_fits = true;          // 4 cards: tp holds the requested context in the free VRAM (tp_fixed_need)
    bool verified = true;         // 4 cards: the data check ran and left nothing corrupt in use (comm p2p_verified)
};
struct PickOut {
    Par par = Par::TP;
    double tp_pf_s = 0, pp_pf_s = 0;                  // prefill of P tokens
    std::vector<double> tp_agg, pp_agg, tp_req, pp_req;  // per c: decode tok/s aggregate, seconds per request
    double x_ms_tok = 0, x_ms_step = 0;               // TP exchange costs used
    bool uneven = false;                              // the cards differ (calibration)
    std::string why;
};
// decode aggregate tok/s by concurrency (measured points, linear in between, flat past the last); TP: same prompt as
// mode B for c = 1, the server agent bench for c = 8 (867-939 on 3 cards, 446 on 2); c = 2 / 4 interpolated
struct DecTab { std::vector<double> c, tok; };
inline double dec_at(const DecTab& t, double c) {
    if (c <= t.c.front()) return t.tok.front() * c / t.c.front();
    for (size_t i = 1; i < t.c.size(); i++)
        if (c <= t.c[i]) return t.tok[i - 1] + (t.tok[i] - t.tok[i - 1]) * (c - t.c[i - 1]) / (t.c[i] - t.c[i - 1]);
    return t.tok.back();
}
inline DecTab tp_dec_tab(int n) {
    if (n == 2) return {{1, 8}, {205, 446}};
    if (n >= 4) return {{1, 8}, {420, 1020}};  // TP4: model, not measured
    return {{1, 8}, {400, 900}};
}
// mode B in flash_run (continuous pipeline); the server runs `rounds` steps per call, so each call fills and drains the
// pipeline once: efficiency rounds g / (rounds g + n - 1) with g = min(c, n + 1) groups
inline DecTab pp_dec_tab(int n) {
    if (n == 2) return {{1, 2, 4, 8, 16}, {120, 243, 291, 330, 324}};
    if (n >= 4) return {{1, 2, 4, 8, 16}, {220, 450, 610, 975, 1180}};  // B4: model
    return {{1, 2, 4, 8, 16}, {218, 450, 513, 800, 954}};
}
// The fixed part a TP rank needs per card, bytes: n cards, S slots, C tokens of context, KV in VRAM (dense weights, the
// slots' recurrent state, the KV pool, buffers; the experts can all be host-backed). Fitted to 2- / 3-card loads,
// 2026-11-03: 1.58 GB + 5.46 GB / n + (S - 1) x 206 MB / n + 24.8 KB x C / n + 0.9 GB (context,
// workspaces, margin).
inline double tp_fixed_need(int n, int S, long long c) {
    return 1.58e9 + 5.46e9 / n + (S - 1) * 0.206e9 / n + 24.8e3 * (double)c / n + 0.9e9;
}
// The wiring around the pick (flash_capi.cpp), as pure functions for the unit test.
// The card-wide comm: up to 3 cards as ever; 4 for tp asked for, and for auto where the driver reports peer access on all
// pairs, so the data check can decide. Forced mode B on 4+ cards and PP x TP build none.
inline bool comm_wanted(int n, Par par, bool pptp, bool driver_all) {
    return !pptp && (n <= 3 || par == Par::TP || (n == 4 && par == Par::AUTO && driver_all));
}
// A pick mode B can replace: auto, exactly 4 cards, tp, KV in VRAM (FLASH_KV=ram has no mode B to go to).
inline bool restart_armed(int n, bool forced, bool kv_ram, Par picked) {
    return !forced && n == 4 && !kv_ram && picked == Par::TP;
}
// The same command in pp: the environment without FLASH_PAR, then FLASH_PAR=pp and the note for the log.
inline std::vector<std::string> restart_env(const std::vector<std::string>& env) {
    std::vector<std::string> out;
    for (const std::string& v : env)
        if (v.rfind("FLASH_PAR=", 0) != 0 && v.rfind("FLASH_PAR_FALLBACK=", 0) != 0) out.push_back(v);
    out.push_back("FLASH_PAR=pp");
    out.push_back("FLASH_PAR_FALLBACK=1");
    return out;
}
// /proc/self/cmdline: NUL-terminated arguments
inline std::vector<std::string> split_cmdline(const std::string& c) {
    std::vector<std::string> out;
    for (size_t i = 0; i < c.size(); i += out.back().size() + 1) out.push_back(c.c_str() + i);
    return out;
}
inline PickOut pick_par(const PickIn& in) {
    PickOut o;
    const int n = in.n;
    if (n > 4) { o.par = Par::PP; o.why = "TP runs on 2 to 4 cards only"; return o; }
    if (n < 2) { o.par = Par::TP; o.why = "one card"; return o; }
    // TP exchange costs of the probed path: host staging is latency-bound (3 cards: +0.244 ms per prompt token on the
    // real x16 link and about the same at 5.5 GB/s), the copy engine / P2P pay bytes beyond the fit's link speed
    const double xb = n == 2 ? 555e3 : n == 3 ? 370e3 : 320e3;  // exchange bytes per token per card (4: model)
    // host-staged refit after the no-P2P TP rework (master 153cd8f0): 3 cards 10K / 32K / 90K 1.85 / 6.06 / 17.03 s vs
    // P2P 1.35 / 4.54 / 13.06 s = +0.044-0.050 ms per prompt token; decode step 9.9 vs 9.5 ms (+0.4 ms)
    o.x_ms_tok = in.tp_x_ms_tok >= 0 ? in.tp_x_ms_tok : in.tp_path == 2 ? 0.05 * xb / 370e3 : in.tp_path == 1 ? 0.004 : 0.0;
    o.x_ms_step = in.tp_x_ms_step >= 0 ? in.tp_x_ms_step : in.tp_path == 2 ? 0.5 * xb / 370e3 : 0.0;
    // prefill = per-token compute + what a slower host link adds (both modes restage their host-backed experts every
    // TP pass / micro-batch). Effective host-backed bytes per card, fitted on the slow-link matrix (3 cards: TP 5.0 GB
    // per 8192-row pass, mode B 3.9 GB per 4096-row micro-batch; 2 cards 14.4 / 13.6 GB per 8192 rows), less the VRAM a
    // card has above 24 GB. 3 cards measured vs model: no P2P 5.5 GB/s TP 4.09 / 36.03 s (10K / 90K) vs 3.9 / 35,
    // B 3.02 / 14.05 vs 2.7 / 15.6; P2P + CE 3.3 GB/s TP 4.99 / 36.79 vs 5.0 / 36.4, B 7.81 / 31.9 vs 4.1 / 27.
    const double extra = std::max(0.0, in.vram - 24e9);
    // (4 cards: every expert resident in both modes, nothing restaged)
    const double tp_host = std::max(0.0, (n == 2 ? 14.4e9 : n == 3 ? 5.0e9 : 0.0) - extra);
    const double pp_host = std::max(0.0, (n == 2 ? 13.6e9 : n == 3 ? 3.9e9 : 0.0) - extra);
    const double slow = std::max(0.0, 1.0 / in.h2d - 1.0 / 24.8e9);   // s per byte beyond the fit's link
    const double P = (double)in.P;
    // 4 cards (model): TP4 ~0.133 ms per token, B4 ~0.070 + a ~590 ms fill
    const double tp_u = n == 2 ? 0.1648 : n == 3 ? 0.1440 : 0.1335, pp_u = n == 2 ? 0.1434 : n == 3 ? 0.0966 : 0.070;
    const double pp_fill = n == 2 ? 900.0 : n == 3 ? 600.0 : 590.0;
    const double stg = tp_passes((long long)P, 8192) * tp_host * slow * 1e3;  // ms
    if (in.tp_path == 2)
        o.tp_pf_s = (P * tp_u + std::max(stg, P * o.x_ms_tok)) / 1e3;
    else
        o.tp_pf_s = (P * (tp_u + o.x_ms_tok) + stg + P * xb * slow * 1e3) / 1e3;
    const double mbr = n == 2 ? 8192.0 : 4096.0, nmb = std::ceil(P / mbr);
    const double mb_ms = std::max(mbr * pp_u, pp_host / in.h2d * 1e3);       // a micro-batch: compute or its staging
    o.pp_pf_s = (pp_fill + std::max(P * pp_u, nmb * mb_ms)) / 1e3;
    // uneven cards: TP runs at the slowest card's speed (relative to the median, which the fit stands for), mode B with
    // the per-card split at about the mean speed
    double pf_min = 1, pf_mean = 1, dc_min = 1, dc_mean = 1;
    if (!in.cards.empty()) {
        pf_min = dc_min = 1e30;
        pf_mean = dc_mean = 0;
        for (const CardSpeed& c : in.cards) {
            pf_min = std::min(pf_min, c.pf);
            dc_min = std::min(dc_min, c.dc);
            pf_mean += c.pf / in.cards.size();
            dc_mean += c.dc / in.cards.size();
        }
        o.tp_pf_s /= std::max(0.05, pf_min);
        o.pp_pf_s /= std::max(0.05, pf_mean);
    }
    o.uneven = pf_mean > pf_min * (1 + in.uneven_tol) || dc_mean > dc_min * (1 + in.uneven_tol);
    double mt = 0, mp = 0;
    for (int c : in.cs) {
        const double tagg0 = dec_at(tp_dec_tab(n), c);
        // TP step without P2P: + x_ms_step per step; the step of the table point: c x 3.2 tokens / aggregate
        const double tstep = c * 3.2 / tagg0 * 1e3;
        const double tagg = tagg0 * tstep / (tstep + o.x_ms_step) * dc_min;
        const int g = std::max(1, std::min(c, n + 1));
        const double eff_r = in.persist ? 1.0 : (double)in.rounds * g / (in.rounds * g + n - 1);
        const double pagg = dec_at(pp_dec_tab(n), c) * (c > 1 ? eff_r : 1.0) * dc_mean;
        const double tr = o.tp_pf_s + in.D / tagg, pr = o.pp_pf_s + in.D / pagg;
        o.tp_agg.push_back(tagg);
        o.pp_agg.push_back(pagg);
        o.tp_req.push_back(tr);
        o.pp_req.push_back(pr);
        mt += tr / in.cs.size();
        mp += pr / in.cs.size();
    }
    // the mix: mode B only when the mean request time over the concurrencies is lower by the margin (a big c = 1 loss
    // is not outweighed by small wins at c = 4 / 8)
    const bool all = !in.cs.empty() && mp * (1.0 + in.margin) < mt;
    // coordinator's rule (2026-11-01): with peer access TP always (also on slow links: its exchanges keep P2P); without
    // it mode B where the model says it wins
    if (n == 4) {  // (#12, 4 x RTX 5080: tp 310 vs pp 145 tok/s at c = 1; the model's 4-card numbers stay unfitted, so
        // the rule does not use them.) Uneven cards keep mode B, whose split follows the card speeds: no 4-card run
        // says otherwise. Not decided here: a tp load that fails anyway (#9): the engine restarts in pp (flash_capi.cpp)
        o.par = in.p2p && in.verified && !o.uneven && in.tp_fits ? Par::TP : Par::PP;
        o.why = !in.p2p     ? "4 cards, no peer access between all of them: mode B"
                : !in.verified ? "4 cards, peer access not verified (QE_COMM_CHECK=0, or a forced FLASH_COMM on a transport that failed the check): mode B"
                : o.uneven  ? "4 cards, uneven: mode B"
                : !in.tp_fits ? "4 cards, TP does not hold the context in the free VRAM: mode B"
                              : "4 cards, verified peer access between all of them: TP";
    } else if (in.p2p && !o.uneven) {
        o.par = Par::TP;
        o.why = "peer access between the cards: TP";
    } else if (in.p2p) {
        o.par = all ? Par::PP : Par::TP;
        o.why = all ? "uneven cards (TP waits on the slowest every layer): mode B cheaper per request over the mix"
                    : "uneven cards, but TP not beaten by the margin over the mix";
    } else {
        o.par = all ? Par::PP : Par::TP;
        o.why = all ? "no peer access: mode B cheaper per request over the mix"
                    : "no peer access, but TP not beaten by the margin over the mix";
    }
    return o;
}

}  // namespace flash_pp
