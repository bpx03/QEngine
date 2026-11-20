// [gptoss-r15-dflash] Block size of the drafter per step, from measured costs and acceptance (host only, no CUDA).
// Phase 2 form (bar P3: never slower than plain, also on a short reply at a long context):
// - Costs (ms of the verify step of R rows, of the drafter pass of a block of B) live in a process-wide table by context
//   bucket (DfCosts: powers of two from 2K keys), so a new sequence starts with the costs the process has seen. Every
//   timed step updates it (a drafted step's verify is the same R-row step as any other).
// - A sequence starts on plain steps. Its acceptance is learnt from SHADOW drafts: a drafter pass alone (no verify rows,
//   ~1.2 ms) whose drafts are compared with the tokens the next plain steps produce. A block runs only when the shadow
//   acceptance says a block beats plain (>= 3 shadows, 3 % margin); never where no block could pay even with every
//   draft accepted. Shadows: 3 at the start (16 steps apart; 2 when far below plain), then one per gap that keeps
//   their cost <= 0.2 % of the time (>= 64 steps, doubling to 2,048 while they change nothing).
// - While a block runs: the best size by expected tokens per ms, a neighbour size probed now and then (gap 32,
//   doubling to 1,024 while nothing changes); it stops (back to shadows) with >= 12 tested drafts and a 3 % margin.
// The caller times its own calls and passes the position (context keys) of each step; it runs the shadow pass when
// want_shadow is set after choose() and reports it with shadow_seen() once the drafts are decided. It feeds the
// drafter's context only around passes (want_context()): a shadow or a block keeps only the newest queued rows.
#pragma once
#include <algorithm>
#include <cmath>

namespace gptoss {

struct DfCosts {   // process-wide: ms by context bucket
    static constexpr int MAXB = 8, NB = 10;
    double tv[NB][MAXB + 1] = {}, td[NB][MAXB + 1] = {};
    bool hv[NB][MAXB + 1] = {}, hd[NB][MAXB + 1] = {};
    static int bucket(int pos) {   // 0: < 2K keys, 1: 2-4K, .. 9: >= 512K
        int b = 0;
        for (int k = pos >> 11; k && b < NB - 1; k >>= 1) b++;
        return b;
    }
    static void upd(double& t, bool& h, double v) {
        if (!h) t = v, h = true;
        else t += (v - t) * (v > 1.6 * t ? 0.02 : v < t ? 0.5 : 0.125);   // a stall (another job, a graph made on first use) moves the mean little; a lower time is believed at once
    }
    static DfCosts& get() {
        static DfCosts c;
        return c;
    }
};

struct DfPolicy {
    static constexpr int MAXB = DfCosts::MAXB;
    int maxb = MAXB;            // largest block the caller allows (2..8)
    double acc[MAXB] = {};      // acc[k]: draft k + 1 accepted, given drafts 1..k accepted (this sequence)
    double accn[MAXB] = {};
    long step = 0, next_try = 16, try_gap = 0, next_probe = 0, gap = 32;
    int cur = 1, bk = 0, nshadow = 0;   // cur: block size running (1 = plain); bk: cost bucket of the current step
    bool probing = false, want_shadow = false, shadow_busy = false;
    long tries = 0, fails = 0;          // shadows, shadows that left the block off
    DfPolicy() { reset(); }
    void prior() {
        for (int k = 0; k < MAXB; k++) acc[k] = 0.75, accn[k] = 2.;
    }
    void reset() {   // a new sequence, or its context changed by more than a few rows
        prior();
        step = 0, next_try = 16, try_gap = 0, gap = 32, next_probe = 0, cur = 1, probing = want_shadow = shadow_busy = false, nshadow = 0;
    }
    // costs of sizes not measured in this bucket: on the line through the nearest measured sizes; nothing at all: a prior
    // (verify: +25 % of the one-row step per extra row; drafter pass: 0.45 of the one-row step)
    static double est(const double* t, const bool* h, int R, int lo) {
        if (h[R]) return t[R];
        int a = -1, b = -1;
        for (int i = R - 1; i >= lo; i--)
            if (h[i]) { a = i; break; }
        for (int i = R + 1; i <= MAXB; i++)
            if (h[i]) { b = i; break; }
        if (a >= 0 && b >= 0) return t[a] + (t[b] - t[a]) * (R - a) / (b - a);
        int c = -1;
        if (a >= 0) {
            for (int i = a - 1; i >= lo; i--)
                if (h[i]) { c = i; break; }
            if (c >= 0) return t[a] + (t[a] - t[c]) * (R - a) / (a - c);
            return t[a] * (1. + 0.25 * (R - a));
        }
        if (b >= 0) {
            for (int i = b + 1; i <= MAXB; i++)
                if (h[i]) { c = i; break; }
            if (c >= 0) return std::max(0.2 * t[b], t[b] - (t[c] - t[b]) * (b - R) / (c - b));
            return t[b] / (1. + 0.25 * (b - R));
        }
        return -1.;
    }
    double tv1() const {
        const DfCosts& C = DfCosts::get();
        if (C.hv[bk][1]) return C.tv[bk][1];
        for (int d = 1; d < DfCosts::NB; d++) {   // nearest bucket with a one-row time
            if (bk - d >= 0 && C.hv[bk - d][1]) return C.tv[bk - d][1];
            if (bk + d < DfCosts::NB && C.hv[bk + d][1]) return C.tv[bk + d][1];
        }
        return 3.;
    }
    double verify_ms(int R) const {
        const DfCosts& C = DfCosts::get();
        if (R == 1) return tv1();
        const double v = est(C.tv[bk], C.hv[bk], R, 1);
        return v > 0 ? v : tv1() * (1. + 0.25 * (R - 1));
    }
    // the drafter pass hardly depends on the block size or the context (1.55-1.71 ms for 2-8 rows, 2K-64K keys): the
    // nearest measured size, in this bucket, else in the nearest bucket; nothing measured: a prior
    static double near_b(const double* t, const bool* h, int B) {
        for (int d = 0; d <= MAXB; d++) {
            if (B - d >= 2 && h[B - d]) return t[B - d];
            if (B + d <= MAXB && h[B + d]) return t[B + d];
        }
        return -1.;
    }
    double draft_ms(int B) const {
        const DfCosts& C = DfCosts::get();
        double v = near_b(C.td[bk], C.hd[bk], B);
        for (int d = 1; d < DfCosts::NB && v <= 0; d++) {
            if (bk - d >= 0) v = near_b(C.td[bk - d], C.hd[bk - d], B);
            if (v <= 0 && bk + d < DfCosts::NB) v = near_b(C.td[bk + d], C.hd[bk + d], B);
        }
        return v > 0 ? v : 0.45 * tv1();
    }
    double tokens(int B, double a0 = -1.) const {   // expected tokens of a step with B - 1 drafts (a0 >= 0: flat acceptance a0)
        double e = 1., p = 1.;
        for (int k = 0; k + 1 < B; k++) {
            p *= a0 >= 0 ? a0 : acc[k];
            e += p;
        }
        return e;
    }
    double rate(int B, double a0 = -1.) const {   // tokens per ms
        if (B <= 1) return 1. / tv1();
        return tokens(B, a0) / (draft_ms(B) + verify_ms(B));
    }
    int best() const {
        int b = 1;
        double r = rate(1);
        for (int B = 2; B <= maxb; B++)
            if (rate(B) > r) r = rate(B), b = B;
        return b;
    }
    bool hopeless() const {   // no block pays even if every draft were accepted
        for (int B = 2; B <= maxb; B++)
            if (rate(B, 1.) > rate(1)) return false;
        return true;
    }
    bool want_context() const { return cur > 1; }   // the caller feeds the drafter's context only while a block runs
    void at(int pos) { bk = DfCosts::bucket(pos); }
    int choose(int pos) {   // block size of the next step at this position
        at(pos);
        probing = want_shadow = false;
        if (cur == 1) {
            want_shadow = !shadow_busy && step >= next_try && !hopeless();
            return 1;
        }
        const int b = best();   // (a switch to plain is made in seen())
        if (b >= 2 && b != cur && rate(b) > rate(cur) * 1.02) cur = b, gap = 32, next_probe = step + gap;
        if (step >= next_probe) {   // a neighbour size, one up / one down in turn
            probing = true;
            const int alt = (cur < maxb && (step / gap) % 2 == 0) ? cur + 1 : cur - 1;
            if (alt >= 2) return alt;
        }
        return cur;
    }
    void learn(int B, int ok) {   // one acceptance for every position (flat over the block): pooled
        const int tested = std::min(ok + 1, B - 1);
        accn[0] = std::min(accn[0] + tested, cur > 1 ? 48. : 24.);
        acc[0] += (ok - tested * acc[0]) / accn[0];
        acc[0] = std::min(0.99, std::max(0.01, acc[0]));
        for (int k = 1; k < MAXB; k++) acc[k] = acc[0];
    }
    void shadow_started() { shadow_busy = true; }
    // a shadow block of B rows (its pass took dms) had its first ok drafts right; called when that is decided
    void shadow_seen(int B, int ok, double dms) {
        shadow_busy = false;
        DfCosts& C = DfCosts::get();
        DfCosts::upd(C.td[bk][B], C.hd[bk][B], dms);
        learn(B, ok);
        nshadow++;
        tries++;
        if (cur > 1) return;
        const int b = best();
        if (b >= 2 && nshadow >= 3 && rate(b) > 1.03 * rate(1)) {   // >= 3 shadows (9 positions at most) and a 3 % margin
            cur = b, gap = 32, next_probe = step + gap, try_gap = 0;
            return;
        }
        fails++;
        double rb = 0.;
        for (int B2 = 2; B2 <= maxb; B2++) rb = std::max(rb, rate(B2));
        if (nshadow < 3 && !(nshadow >= 2 && rb < 0.85 * rate(1))) {   // the first 3 shadows 16 steps apart (2 when the block is far below plain; 16: the context rows of a shadow are one pass)
            next_try = step + 16;
            return;
        }
        const long g0 = (long)std::max(64., (draft_ms(4) + 0.2) / (0.002 * tv1()));   // a shadow costs <= 0.2 % of the time between shadows
        try_gap = try_gap ? std::min(2048L, std::max(g0, 2 * try_gap)) : g0;
        next_try = step + try_gap;
    }
    // after the step: rows = B (1 + drafts), draft_ms = 0 when no drafter pass ran (plain or lookup rows), ok = drafts accepted
    void seen(int B, double dms, double vms, int ok, bool drafted, int pos) {
        at(pos);
        DfCosts& C = DfCosts::get();
        if (B >= 1 && B <= MAXB) DfCosts::upd(C.tv[bk][B], C.hv[bk][B], vms);
        if (drafted && B >= 2 && B <= MAXB) {
            DfCosts::upd(C.td[bk][B], C.hd[bk][B], dms);
            learn(B, ok);
        }
        if (cur > 1) {
            double rb = 0.;
            for (int B2 = 2; B2 <= maxb; B2++) rb = std::max(rb, rate(B2));
            if (accn[0] >= 12. && rate(1) > 1.03 * rb) {   // the block stopped paying (clearly: a 3 % margin against the estimate's noise): shadows again
                cur = 1;
                nshadow = 3;
                try_gap = (long)std::max(64., (draft_ms(4) + 0.2) / (0.002 * tv1()));
                next_try = step + 1 + try_gap;
            } else if (probing) {
                const int b = best();
                if (b != cur && rate(b) > rate(cur) * 1.02) gap = 32;
                else gap = std::min(1024L, std::max(32L, gap) * 2);
                next_probe = step + gap;
            }
        }
        step++;
    }
};

}  // namespace gptoss
