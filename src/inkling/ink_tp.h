// [ink-tp-plan] Inkling generic tensor parallelism: the partition planner (host only, no CUDA). One function for every N:
// it takes the rank count and per-card weights and returns every split as data (q / KV heads, output-projection column
// chunks, expert intermediate units). Kernels get these ranges as arguments; nothing in the engine assumes a rank count.
// Design and the plan table for N = 1..16:.
#pragma once
#include <algorithm>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <functional>
#include <mutex>
#include <numeric>
#include <string>
#include <thread>
#include <vector>

namespace ink {

constexpr int TP_MAX = 16;         // ranks (receive-word and pointer tables; raise freely, nothing else depends on it)
constexpr int TP_UNIT = 32;        // expert intermediate unit: MXFP8 scale block 32, NVFP4 16, gate/up block 8 x GP = 16
constexpr int TP_NU = 1152 / TP_UNIT;   // 36 units (I / TP_UNIT)
constexpr int TP_V = 200058;            // vocab rows of the head (split in row pairs: the 12-bit groups / screen pairs)

struct TpShard {
    int N = 1, rank = 0, dev = 0;
    float w = 1.f;                 // relative card speed
    int q0 = 0, q1 = 24;           // q heads [q0, q1)
    int kv0 = 0, kv1 = 8;          // K / V heads computed [kv0, kv1) (q0 / 3 .. ceil(q1 / 3): shared heads when q0 % 3 != 0)
    int cb = 0, ce = 12;           // output-projection column chunks of 256 [cb, ce) (x masked to [128 q0, 128 q1))
    int ua = 0, ub = TP_NU;        // expert units [ua, ub): intermediate columns [32 ua, 32 ub) of every expert
    int v0 = 0, v1 = TP_V;         // head rows [v0, v1) (two-stage head per rank, argmax over ranks)
    double bytes = 0;              // planner's per-token read estimate (weights + K / V at the plan depth)
    int nq() const { return q1 - q0; }
    int nkv() const { return kv1 - kv0; }
    int nu() const { return ub - ua; }
    bool full() const { return q0 == 0 && q1 == 24 && kv0 == 0 && kv1 == 8 && ua == 0 && ub == TP_NU && v0 == 0 && v1 == TP_V; }
};

// n integer units over weights w (largest remainder; at least one each while units >= ranks), contiguous in rank order
static inline std::vector<int> tp_apportion(int units, const std::vector<double>& w, bool min1) {
    const int n = (int)w.size();
    std::vector<int> c(n, 0);
    const double tw = std::accumulate(w.begin(), w.end(), 0.0);
    std::vector<std::pair<double, int>> rem;
    int used = 0;
    for (int i = 0; i < n; i++) {
        const double x = tw > 0 ? units * w[i] / tw : 0.0;
        c[i] = (int)std::floor(x);
        used += c[i];
        rem.push_back({x - c[i], -i});
    }
    std::sort(rem.rbegin(), rem.rend());
    for (int k = 0; used < units; k++, used++) c[-rem[k % n].second]++;
    if (min1 && units >= n)
        for (int i = 0; i < n; i++)
            while (c[i] == 0) {   // take one from the largest
                const int j = (int)(std::max_element(c.begin(), c.end()) - c.begin());
                c[j]--;
                c[i]++;
            }
    return c;
}

// Byte model per token (decode, one row) in bytes; 48 applications, 12-bit dense, NVFP4 x 44 + MXFP8 x 4 experts.
struct TpCost {
    static constexpr double H = 2304, APPS = 48, B12 = 1.5;
    static double qhead() { return APPS * (128 + 16 + 128 + 1 + 128) * H * B12; }   // q, r, gate, sink rows + wo columns
    static double kvhead(int depth) {                                               // k + v rows (+ taps) + K / V reads
        const double w = (40 * 256 + 8 * 264) * H * B12;
        const double kv = 8.0 * 512 * depth + 40.0 * 512 * std::min(depth, 1024);
        return w + kv;
    }
    static double vrow() { return H * 1.0 + 1; }   // 8-bit screen row (+ its exponent byte; the rescore is ~40 rows)
    static double unit() {   // one 32-column unit of the 8 routed experts
        const double nv = 8 * (64 * H * 0.5625 + H * 32 * 0.5625), mx = 8 * (64 * H * (1 + 1.0 / 32) + H * 32 * (1 + 1.0 / 32));
        return 44 * nv + 4 * mx;
    }
};

// The planner. N ranks, w[r] relative speed (empty: equal), depth = the K / V depth the balance is computed for.
// Attention: N <= 8 splits whole KV heads (no replication); N > 8 splits q heads, a KV head whose q heads land on two
// ranks is computed on both (its K / V rows stored twice); N > 24 leaves ranks without attention. Experts: 36 units,
// assigned so that every rank's bytes / speed come out level (attention-heavy ranks get fewer units); N > 36 leaves
// ranks without experts. Every rank keeps the full hidden state (replicated small ops, router, head for now).
static inline std::vector<TpShard> tp_plan(int N, std::vector<double> w = {}, int depth = 4096) {
    if (w.empty()) w.assign(N, 1.0);
    std::vector<TpShard> r(N);
    const bool kvsplit = N <= 8;
    const std::vector<int> au = tp_apportion(kvsplit ? 8 : 24, w, true);
    int a = 0;
    double attn_tot = 0;
    std::vector<double> attn(N, 0);
    for (int i = 0; i < N; i++) {
        TpShard& s = r[i];
        s.N = N;
        s.rank = i;
        s.w = (float)w[i];
        s.q0 = kvsplit ? 3 * a : a;
        s.q1 = kvsplit ? 3 * (a + au[i]) : a + au[i];
        a += au[i];
        s.kv0 = s.q0 / 3;
        s.kv1 = (s.q1 + 2) / 3;
        if (s.q1 == s.q0) s.kv0 = s.kv1 = s.q0 / 3;
        s.cb = (s.q0 * 128) / 256;
        s.ce = (s.q1 * 128 + 255) / 256;
        if (s.q1 == s.q0) s.cb = s.ce = 0;
        attn[i] = s.nq() * TpCost::qhead() + s.nkv() * TpCost::kvhead(depth);
    }
    const std::vector<int> vu = tp_apportion(TP_V / 2, w, true);   // head row pairs by speed (review tp L1: >= 1 pair each)
    for (int i = 0, v = 0; i < N; i++) {
        r[i].v0 = 2 * v;
        r[i].v1 = 2 * (v + vu[i]);
        v += vu[i];
        attn[i] += (r[i].v1 - r[i].v0) * TpCost::vrow();
        attn_tot += attn[i];
    }
    const double U = TpCost::unit(), tot = attn_tot + TP_NU * U, tw = std::accumulate(w.begin(), w.end(), 0.0);
    std::vector<double> want(N);
    for (int i = 0; i < N; i++) want[i] = std::max(0.0, (tot * w[i] / tw - attn[i]) / U);
    const std::vector<int> eu = tp_apportion(TP_NU, want, false);
    int u = 0;
    for (int i = 0; i < N; i++) {
        r[i].ua = u;
        r[i].ub = u + eu[i];
        u += eu[i];
        r[i].bytes = attn[i] + eu[i] * U;
    }
    return r;
}

static inline std::string tp_plan_str(const std::vector<TpShard>& p) {
    std::string o;
    char b[256];
    double mx = 0, sum = 0;
    for (auto& s : p) {
        mx = std::max(mx, s.bytes / s.w);
        sum += s.bytes;
    }
    for (auto& s : p) {
        snprintf(b, sizeof b, "rank %2d w %.2f: q %2d-%2d kv %d-%d wo chunks %2d-%2d experts I %4d-%4d (%2d units) vocab %6d-%6d %.3f GB\n",
                 s.rank, s.w, s.q0, s.q1, s.kv0, s.kv1, s.cb, s.ce, TP_UNIT * s.ua, TP_UNIT * s.ub, s.nu(), s.v0, s.v1, s.bytes / 1e9);
        o += b;
    }
    snprintf(b, sizeof b, "sum %.3f GB, slowest rank %.3f GB-equivalent\n", sum / 1e9, mx / 1e9);
    return o + b;
}

// ---- SPMD runtime: one host thread per rank, a barrier for the host side of the collectives
struct TpBarrier {
    std::mutex m;
    std::condition_variable cv;
    int n = 1, cnt = 0;
    unsigned gen = 0;
    bool dead = false;
    void wait() {
        std::unique_lock<std::mutex> l(m);
        if (dead) throw std::runtime_error("inkling TP: a rank failed");
        const unsigned g = gen;
        if (++cnt == n) {
            cnt = 0;
            gen++;
            cv.notify_all();
            return;
        }
        cv.wait(l, [&] { return gen != g || dead; });
        if (dead && gen == g) throw std::runtime_error("inkling TP: a rank failed");
    }
    void kill() {
        std::lock_guard<std::mutex> l(m);
        dead = true;
        cv.notify_all();
    }
    void revive() {   // review tp M1: only while no rank is inside wait() (between two pool runs)
        std::lock_guard<std::mutex> l(m);
        dead = false;
        cnt = 0;
    }
};
struct TpPool {   // N persistent workers; run(f) calls f(rank) on every worker and returns when all are done
    int n = 0;
    std::vector<std::thread> th;
    std::mutex m;
    std::condition_variable cv, done;
    std::function<void(int)> job;
    unsigned gen = 0;
    int left = 0;
    bool quit = false;
    std::string err, init_err;
    TpBarrier* bar = nullptr;
    unsigned* abort_word = nullptr;   // review tp M2: set on a rank's failure (the peers' device waits leave)
    void start(int n_, std::function<void(int)> init) {
        n = n_;
        for (int i = 0; i < n; i++)
            th.emplace_back([this, i, init] {
                try {   // review tp L5: an init failure is an error of the run, not std::terminate
                    init(i);
                } catch (const std::exception& e) {
                    fprintf(stderr, "[ink] TP rank %d init failed: %s\n", i, e.what());
                    std::lock_guard<std::mutex> l(m);
                    if (init_err.empty()) init_err = e.what();
                }
                unsigned seen = 0;
                for (;;) {
                    std::function<void(int)> f;
                    {
                        std::unique_lock<std::mutex> l(m);
                        cv.wait(l, [&] { return quit || gen != seen; });
                        if (quit) return;
                        seen = gen;
                        f = job;
                    }
                    try {
                        f(i);
                    } catch (const std::exception& e) {
                        fprintf(stderr, "[ink] TP rank %d failed: %s\n", i, e.what());   // before the peers' waits time out
                        if (abort_word) *(volatile unsigned*)abort_word = 1;   // review tp M2: peers' device waits leave
                        std::lock_guard<std::mutex> l(m);
                        if (err.empty()) err = e.what();
                        if (bar) bar->kill();
                    }
                    std::lock_guard<std::mutex> l(m);
                    if (--left == 0) done.notify_all();
                }
            });
    }
    void run(std::function<void(int)> f) {
        {
            std::lock_guard<std::mutex> l(m);
            if (!init_err.empty()) throw std::runtime_error("inkling TP: rank init failed: " + init_err);
        }
        if (bar) bar->revive();   // review tp M1: a request-level error stays request-level (no rank is in wait() here)
        {
            std::lock_guard<std::mutex> l(m);
            job = std::move(f);
            left = n;
            err.clear();
            gen++;
        }
        cv.notify_all();
        std::unique_lock<std::mutex> l(m);
        done.wait(l, [&] { return left == 0; });
        if (!err.empty()) throw std::runtime_error(err);
    }
    ~TpPool() {
        {
            std::lock_guard<std::mutex> l(m);
            quit = true;
        }
        cv.notify_all();
        for (auto& t : th) t.join();
    }
};

}  // namespace ink
