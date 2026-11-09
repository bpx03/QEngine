// CPU-only unit test of the host-RAM budget at boot (src/host_budget.h). Built and run by tests/test_env.py.
// MemTotal 64 / 92 / 128 / 251 GiB, 2 and 3 cards, both models; on the 251 GiB host (our box: 263,356,892 kB) the plan
// must be today's defaults exactly. QE_HOST_BUDGET_TABLE=1 prints the table.
#include "host_budget.h"
#include "host_mem.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

static int fails = 0;
#define EXPECT(c)                                                                         \
    do {                                                                                  \
        if (!(c)) { fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); fails++; } \
    } while (0)

using namespace qe_hostbudget;

static const double GiB = 1073741824.0;
static const double kOurTotal = 263356892.0 * 1024;  // MemTotal of our host (251 GiB)
static const double kNgram = 51.8e9;                 // INT8 n-gram table (ngram_q8 shards: 128 x 2,500,012 x 162 B)

// a 3090 at boot: 24,576 MiB, ~0.45 GB taken by the CUDA context; the desktop card ~1.5 GB more
static std::vector<double> cards_free(int n, bool desktop) {
    std::vector<double> f(n, 24576.0 * 1048576 - 0.45e9);
    if (desktop && n > 1) f[1] -= 1.5e9;
    return f;
}

static In qfn(double total, double avail, int n) {
    In in;
    in.tag = "[qflash] ";
    in.qfn = true;
    in.cards = n;
    in.total = total;
    in.avail = avail;
    in.fixed = 248320.0 * 2560 * 2;  // pinned embeddings
    in.experts = qfn_expert_host_est(n, cards_free(n, n == 3));
    in.ngram = kNgram;
    in.arch_def = 48e9;
    in.pref_def = 8e9;
    in.arch_env = "FLASH_KV_ARCHIVE_GB";
    in.pref_env = "FLASH_PREFIX_GB";
    return in;
}

static In b27(double total, double avail, int n) {
    In in;
    in.cards = n;
    in.total = total;
    in.avail = avail;
    in.fixed = 248320.0 * 5120 * 2 + 32 * 0.03e9 * n;  // pinned embeddings + 32 DeltaNet snapshot buffers per card
    in.arch_def = 32 * GiB;
    in.pref_def = 8 * GiB;
    in.arch_env = "QE_KV_ARCHIVE_GB";
    in.pref_env = "QE_PREFIX_GB";
    in.prefix_needs_archive = false;
    return in;
}

static bool table = false;
static void row(const char* model, double tot_gib, int n, const In& in, const Out& o) {
    if (!table) return;
    printf("| %s | %.0f | %d | %.1f | %.1f | %.1f | %s | %.1f | %.1f | %.1f | %s |\n", model, tot_gib, n, in.avail / GB,
           o.reserve / GB, in.experts / GB, in.ngram > 0 ? (o.ngram_pinned ? "pinned 51.8" : "page cache") : "-",
           o.ngram_cache / GB, o.archive / GB, o.prefix / GB, o.reduced ? "yes" : "no");
}

int main() {
    table = getenv("QE_HOST_BUDGET_TABLE") && !strcmp(getenv("QE_HOST_BUDGET_TABLE"), "1");
    if (table) {
        printf("| model | MemTotal GiB | cards | MemAvailable GB | reserve GB | experts GB (est.) | n-gram | page cache GB | archive GB | prefixes GB | reduced |\n");
        printf("|---|---|---|---|---|---|---|---|---|---|---|\n");
    }
    // the estimate against the 1.3.4 boots: 2 cards 31.2 GB host-backed, 3 cards (GPU1 desktop) 11.9 GB
    EXPECT(std::fabs(qfn_expert_host_est(2, cards_free(2, false)) - 31.2e9) < 2.5e9);
    EXPECT(std::fabs(qfn_expert_host_est(3, cards_free(3, true)) - 11.9e9) < 2.5e9);

    for (int n : {2, 3}) {
        for (double tg : {64.0, 92.0, 128.0, 251.0}) {
            const double total = tg == 251.0 ? kOurTotal : tg * GiB;
            const double avail = total - 6e9;  // desktop + services at boot
            const In a = qfn(total, avail, n);
            const Out oa = plan(a);
            row("Flash-Next", tg, n, a, oa);
            const In b = b27(total, avail, n);
            const Out ob = plan(b);
            row("27B", tg, n, b, ob);
            EXPECT(ob.warn.empty());
            // never more than fits: everything planned stays inside MemTotal - reserve
            const double used = a.base + a.fixed + a.experts + (oa.ngram_pinned ? a.ngram : oa.ngram_cache) + oa.archive + oa.prefix;
            EXPECT(used <= total - oa.reserve + 1e6);
            EXPECT(b.base + b.fixed + ob.archive + ob.prefix <= total - ob.reserve + 1e6);
            if (tg == 251.0) {  // our host: today's defaults, exactly
                EXPECT(oa.ngram_pinned && oa.archive == 48e9 && oa.prefix == 8e9 && !oa.reduced && oa.why.empty() && oa.warn.empty());
                EXPECT(ob.archive == 32 * GiB && ob.prefix == 8 * GiB && !ob.reduced && ob.why.empty());
                // with the quality-gate server holding 124 GB next to it (MemAvailable 111 GB at boot): the machine's size
                // decides, the runtime guard holds the archive back (review M1)
                In c = qfn(total, 111e9, n);
                Out oc = plan(c);
                EXPECT(oc.ngram_pinned && !oc.reduced && oc.archive == 48e9 && oc.prefix == 8e9);
                In d = b27(total, 111e9, n);
                EXPECT(plan(d).archive == 32 * GiB && plan(d).prefix == 8 * GiB);  // (the 27B's 40 GiB fit)
                // the start-up line says nothing was reduced
                EXPECT(line(a, oa).find("reduced: nothing") != std::string::npos);
            }
            if (tg == 92.0 && n == 2) {  // club #1906: the table does not fit next to ~31 GB of experts
                EXPECT(!oa.ngram_pinned && oa.reduced && oa.archive > 0 && oa.archive < 48e9 && oa.ngram_cache > 0);
                EXPECT(line(a, oa).find("n-gram table not pinned") != std::string::npos);
            }
            if (tg == 92.0 && n == 3) EXPECT(oa.ngram_pinned && oa.archive > 8e9 && oa.archive < 48e9);
            if (tg == 64.0) EXPECT(!oa.ngram_pinned);
            if (tg == 128.0 && n == 3) EXPECT(oa.ngram_pinned);
            EXPECT(!ob.reduced);  // the 27B's defaults (40 GiB) fit from 64 GiB up
        }
    }
    // explicit values win, with a warning when they do not fit
    {
        In a = qfn(92 * GiB, 86e9, 2);
        a.arch_user = 48e9;
        Out o = plan(a);
        EXPECT(o.archive == 48e9 && !o.warn.empty());  // kept as set
        EXPECT(o.prefix < 8e9);                         // the default part (prefixes) gives way
        a.pref_user = 8e9;
        o = plan(a);
        EXPECT(o.archive == 48e9 && o.prefix == 8e9 && !o.warn.empty() && o.why.size() == 1);  // (only the n-gram line)
        In b = qfn(92 * GiB, 86e9, 2);
        b.ngram_user = 1;  // FLASH_NGRAM_PIN=1
        o = plan(b);
        EXPECT(o.ngram_pinned && !o.warn.empty());
        b.ngram_user = 0;  // FLASH_NGRAM_PIN=0 on a big host: mapped, the archive keeps its default
        b.total = kOurTotal;
        b.avail = 200e9;
        o = plan(b);
        EXPECT(!o.ngram_pinned && o.archive == 48e9 && o.prefix == 8e9 && o.warn.empty());
        In c = qfn(92 * GiB, 86e9, 2);
        c.ngram_can_map = false;  // BF16 table only: cannot be staged, pinned with a warning
        o = plan(c);
        EXPECT(o.ngram_pinned && !o.warn.empty());
        In d = qfn(92 * GiB, 86e9, 2);
        d.arch_user = 0;  // archive off: no prefixes, the table needs no archive floor
        o = plan(d);
        EXPECT(o.archive == 0 && o.prefix == 0);
        In e = b27(64 * GiB, 60e9, 2);
        e.pref_user = 0;  // 27B, prefixes off: the archive takes the room up to its default
        o = plan(e);
        EXPECT(o.prefix == 0 && o.archive > 0 && o.archive <= 32 * GiB);
        In f = b27(32 * GiB, 30e9, 2);  // 27B on a 32 GiB host: archive and prefixes trimmed
        o = plan(f);
        EXPECT(o.reduced && o.archive < 32 * GiB && o.prefix < 8 * GiB && o.archive > 0);
        EXPECT(f.base + f.fixed + o.archive + o.prefix <= f.total - o.reserve + 1e6);
        // a cgroup limit (systemd-run -p MemoryMax=92G on the 251 GiB host) plans like a 92 GiB host
        In g = qfn(kOurTotal, 200e9, 2);
        g.cg_limit = 92 * GiB;
        g.cg_anon = 2e9;
        o = plan(g);
        const Out o92 = plan(qfn(92 * GiB, 92 * GiB - 2e9, 2));
        EXPECT(!o.ngram_pinned && o.archive == o92.archive && o.prefix == o92.prefix);
        EXPECT(line(g, o).find("cgroup limit 98.8 GB") != std::string::npos);
        // review M1: a short MemAvailable at boot does not fix the budgets: the same plan as with a free host, the table pinned
        // with a warning (plan_wait waits a while first)
        In h = qfn(kOurTotal, 60e9, 2);
        o = plan(h);
        EXPECT(o.ngram_pinned && !o.reduced && o.archive == 48e9 && o.prefix == 8e9 && o.avail_short && !o.experts_short);
        // the review's cases: QFN, 3 cards, 128 GiB, a 50 GB process alive at boot; 27B, 64 GiB, MemAvailable 40 GB
        In h3 = qfn(128 * GiB, 70e9, 3);
        const Out oh3 = plan(h3), oh3f = plan(qfn(128 * GiB, 128 * GiB - 6e9, 3));
        EXPECT(oh3.ngram_pinned && oh3.archive == oh3f.archive && oh3.prefix == oh3f.prefix);
        const Out o27 = plan(b27(64 * GiB, 40e9, 3)), o27f = plan(b27(64 * GiB, 60e9, 3));
        EXPECT(o27.archive == o27f.archive && o27.prefix == o27f.prefix && !o27.avail_short);
        // below the pinned experts: a warning, never a refusal (review M2, user rule: if the model can load, it runs; the pins'
        // guard stops the load only when the memory is really not there)
        auto warned = [](const Out& x, const char* w) {
            for (auto& v : x.warn) if (v.find(w) != std::string::npos) return true;
            return false;
        };
        In fz = qfn(kOurTotal, 30e9, 2);
        o = plan(fz);
        EXPECT(o.experts_short && warned(o, "host-backed expert groups"));
        In f4 = qfn(kOurTotal, 40e9, 2);  // the review's case: the experts fit with 6 GB left, under the 8 GB margin
        o = plan(f4);
        EXPECT(o.experts_short && warned(o, "host-backed expert groups"));
        In f3 = qfn(kOurTotal, 30e9, 3);  // 3 cards keep more experts in VRAM: no warning
        o = plan(f3);
        EXPECT(!o.experts_short && !warned(o, "host-backed expert groups"));
        // plan_wait still waits for a previous server when the experts are short, also where the full plan is as small
        {
            In w = qfn(40e9, 30e9, 2);
            double waited = -1;
            plan_wait(w, [&] { return 30e9; }, 5.0, &waited, [] { return true; });
            EXPECT(waited >= 5.0);
        }
        // review H1: the pins' guard counts only the untouched part of the other pins in flight. The review's state: 110 GB
        // MemAvailable at the start of the 51.8 GB table, 25 GB of experts pinned, a 0.13 GB expert piece starts
        {
            using namespace qe_hostmem;
            const size_t G = 1000000000, tbl = (size_t)(51.8 * G), piece = (size_t)(0.13 * G);
            auto state = [&](size_t read, size_t experts) { return 110 * G - read - experts; };
            PinCharge table(tbl);
            EXPECT(table.others() == 0);
            table.touched(32 * G);
            {
                PinCharge p(piece);
                EXPECT(p.others() == tbl - 32 * G);
                EXPECT(pin_fits(state(32 * G, 25 * G), piece, p.others()));   // the whole table counted here: exit 3 on a healthy host
                EXPECT(!pin_fits(state(32 * G, 25 * G), piece, tbl));
                p.touched(piece / 2);
                EXPECT(pin_untouched().load() == tbl - 32 * G + piece / 2);
            }
            EXPECT(pin_untouched().load() == tbl - 32 * G);  // an ended pin leaves nothing, whatever it had touched
            table.touched(tbl - 32 * G);
            EXPECT(pin_untouched().load() == 0);
            // a host really short still stops: 25 GB free, the table's untouched 19.8 GB plus 6 GB of experts to come
            PinCharge t2(tbl);
            t2.touched(32 * G);
            PinCharge e2(6 * G);
            EXPECT(!pin_fits(25 * G, 6 * G, e2.others()));
        }
        // plan_wait: MemAvailable comes back (the previous server exits): the full plan
        {
            In w = qfn(kOurTotal, 60e9, 2);
            double waited = -1;
            int calls = 0;
            const Out ow = plan_wait(w, [&] { return ++calls >= 1 ? 200e9 : 60e9; }, 5.0, &waited);
            EXPECT(ow.ngram_pinned && waited >= 1 && waited <= 2);
            // a host that stays short: no long wait (stops at the first read that does not rise)
            In w2 = qfn(kOurTotal, 60e9, 2);
            double waited2 = -1;
            plan_wait(w2, [&] { return 60e9; }, 20.0, &waited2);
            EXPECT(waited2 <= 3.0);
            // review N-L2: a previous holder alive (busy) keeps the wait going past three flat seconds, up to wait_s
            In w3 = qfn(kOurTotal, 60e9, 2);
            double waited3 = -1;
            plan_wait(w3, [&] { return 60e9; }, 5.0, &waited3, [] { return true; });
            EXPECT(waited3 >= 5.0);
        }
        // a very small planned archive warns, names what changes it, and an explicit value does not warn
        {
            auto small_warn = [](const Out& o) {
                for (auto& w : o.warn) if (w.find("QE_HOST_RESERVE_GB") != std::string::npos) return true;
                return false;
            };
            In s = b27(kOurTotal, 200e9, 3);
            EXPECT(!small_warn(plan(s)));
            s.reserve = kOurTotal - s.base - s.fixed;  // nothing left for the archive
            const Out os = plan(s);
            EXPECT(os.archive < 2e9 && small_warn(os));
            s.arch_user = 0;
            EXPECT(!small_warn(plan(s)));
        }
        // review M4: the 27B's prefixes do not need the archive (QE_KV_ARCHIVE_GB=0 keeps 8 GiB of prefixes, as before)
        In k = b27(kOurTotal, 200e9, 3);
        k.prefix_needs_archive = false;
        k.arch_user = 0;
        o = plan(k);
        EXPECT(o.archive == 0 && o.prefix == 8 * GiB && o.warn.empty());
        // club-3090 #1966 (FLASH_KV=ram): 125 GiB host (MemTotal 134.5 GB), 4x RTX 5080 16 GB at TP2 / TP3 / TP4, a 262K window,
        // 3 seats, the default 1M-token pool. Before: the pool (13.5 GB) was not in the plan, so the archive took it too and the
        // plan summed 13.5 GB past MemTotal - reserve. Now everything planned fits MemTotal - reserve, and what is pinned at
        // start fits MemAvailable - margin
        {
            const double tot = 134.5e9, w = 262144, tb = 13520;
            auto ram = [&](In x, double want_tok, int seats) {
                x.kv_tok = tb;
                x.kv_want = std::max(w, want_tok) * tb;
                x.kv_seats = seats * w * tb;
                x.kv_one = w * tb;
                return x;
            };
            auto r5080 = [&](int n, double avail) {
                In x = qfn(tot, avail, n);
                x.experts = qfn_expert_host_est(n, std::vector<double>(n, 16303.0 * 1048576 - 0.45e9));
                return x;
            };
            for (int n : {2, 3, 4})
                for (double av : {127e9, 110e9, 95e9}) {
                    const In x = ram(r5080(n, av), 1e6, 3);
                    const Out ox = plan(x);
                    if (table)
                        printf("| #1966 TP%d | MemAvailable %.0f | experts %.1f | n-gram %s | KV %.1f GB (%.2fM) | archive %.1f + %.1f | "
                               "pinned %.1f of %.1f free | %s |\n", n, av / GB, x.experts / GB, ox.ngram_pinned ? "pinned" : "page cache",
                               ox.kv / GB, ox.kv / tb / 1e6, ox.archive / GB, ox.prefix / GB, ox.pinned / GB, ox.room_now / GB,
                               ox.why.empty() ? "-" : ox.why[0].c_str());
                    const double used = x.base + x.fixed + ox.pinned + (ox.ngram_pinned ? 0 : ox.ngram_cache) + ox.archive + ox.prefix;
                    EXPECT(used <= tot - ox.reserve + 1e6);   // the plan inside MemTotal - reserve
                    EXPECT(ox.pinned <= ox.room_now + 1e6 || !ox.warn.empty());  // pinned at start inside MemAvailable - margin
                    EXPECT(ox.kv >= w * tb && ox.kv <= 1e6 * tb + 1);
                    EXPECT(line(x, ox).find("KV pool in RAM") != std::string::npos);
                    EXPECT(json(x, ox).find("\"kv_ram_gb\"") != std::string::npos);
                    if (av == 127e9) EXPECT(ox.ngram_pinned && ox.warn.empty());  // a desktop's worth used: the table stays pinned
                    if (av == 127e9 && n >= 3) EXPECT(ox.kv == 1e6 * tb);         // and the whole pool
                }
            // the order: the archive first (TP4, 127 GB free: the pool whole, the archive smaller than without the pool)
            {
                const In x = ram(r5080(4, 127e9), 1e6, 3);
                const Out ox = plan(x), o0 = plan(r5080(4, 127e9));
                EXPECT(ox.kv == 1e6 * tb && ox.ngram_pinned && ox.archive + ox.prefix <= o0.archive + o0.prefix - 13.5e9 + 0.3e9);
            }
            // then the pool to seats x window, then one window, then the n-gram table; the experts never
            {
                In x = ram(r5080(2, 120.5e9), 1e6, 3);  // TP2, a desktop using a few GB more
                Out ox = plan(x);
                EXPECT(ox.ngram_pinned && ox.kv < 1e6 * tb && ox.kv >= 3 * w * tb - 1 && ox.archive < 5e9);
                x.avail = 116e9;
                ox = plan(x);
                EXPECT(ox.ngram_pinned && ox.kv < 3 * w * tb && ox.kv >= w * tb && ox.archive < 10e9);
                x.avail = 95e9;
                ox = plan(x);
                EXPECT(!ox.ngram_pinned && ox.kv >= w * tb && ox.ngram_cache > 0 && ox.warn.empty());
                EXPECT(line(x, ox).find("n-gram table not pinned") != std::string::npos);
                x.ngram_user = 1;  // FLASH_NGRAM_PIN=1: pinned as set, the pool at one window, a warning
                ox = plan(x);
                EXPECT(ox.ngram_pinned && ox.kv == w * tb && !ox.warn.empty());
                x.ngram_user = -1;
                x.ngram_can_map = false;  // BF16 table only: pinned anyway, warned
                ox = plan(x);
                EXPECT(ox.ngram_pinned && ox.kv == w * tb && !ox.warn.empty());
                // nothing fits: never refused, a warning; the pool never below one window
                In y = ram(r5080(2, 55e9), 1e6, 3);
                const Out oy = plan(y);
                EXPECT(!oy.ngram_pinned && oy.kv == w * tb && !oy.warn.empty() && oy.experts_short);
            }
            // FLASH_NGRAM_PIN=0 with the pool: the table mapped, the pool whole where it fits (review L7)
            {
                In x = ram(r5080(2, 116e9), 1e6, 3);
                x.ngram_user = 0;
                const Out ox = plan(x);
                EXPECT(!ox.ngram_pinned && ox.kv == 1e6 * tb && ox.warn.empty());
                // FLASH_KV_RAM_TOKENS below one window: the pool is one window (the engine takes at least the context)
                const Out os = plan(ram(r5080(3, 127e9), 1000, 3));
                EXPECT(os.kv == w * tb);
                // the re-plan after the load with the pool fixed: no reduction line for it, the archive from what is left
                In f = ram(r5080(3, 127e9), 1e6, 3);
                f.kv_want = f.kv_seats = f.kv_one = 1001472 * tb;
                f.ngram_user = 1;
                const Out of = plan(f);
                EXPECT(of.kv == f.kv_want && of.warn.empty() && of.archive > 0);
                for (auto& y : of.why) EXPECT(y.find("KV pool in RAM") == std::string::npos);
                // a host where the room holds the plan but MemAvailable does not: plan_wait waits; where neither does, it
                // does not wait for nothing (review L4)
                EXPECT(plan(ram(r5080(3, 80e9), 1e6, 3)).avail_short);
                In t = ram(qfn(64 * GiB, 68e9, 2), 1e6, 3);  // (room 33.4 GB < free above the margin 40 GB)
                t.experts = 20e9;
                EXPECT(!plan(t).avail_short);
            }
            // our host (251 GiB, 3 x 3090): FLASH_KV=ram keeps the old sizes exactly (pool 1M, archive 48 + 8, table pinned)
            for (int n : {2, 3}) {
                const In x = ram(qfn(kOurTotal, kOurTotal - 6e9, n), 1e6, 16);
                const Out ox = plan(x);
                EXPECT(ox.ngram_pinned && ox.kv == 1e6 * tb && ox.archive == 48e9 && ox.prefix == 8e9 && !ox.reduced && ox.warn.empty());
                // next to another server (MemAvailable 111 GB): still the old sizes
                const Out oz = plan(ram(qfn(kOurTotal, 111e9, n), 1e6, 16));
                EXPECT(oz.ngram_pinned && oz.kv == 1e6 * tb && oz.archive == 48e9 && !oz.reduced);
                // a 4M pool asked for fits as well
                const Out o4 = plan(ram(qfn(kOurTotal, kOurTotal - 6e9, n), 4e6, 16));
                EXPECT(o4.kv == 4e6 * tb && o4.ngram_pinned && o4.archive == 48e9);
            }
            // a cgroup of 134.5 GB on our host plans like the 125 GiB host
            {
                In g = ram(r5080(3, 200e9), 1e6, 3);
                g.total = kOurTotal;
                g.cg_limit = tot;
                g.cg_anon = 7.5e9;
                const Out og = plan(g), oh = plan(ram(r5080(3, tot - 7.5e9), 1e6, 3));
                EXPECT(og.kv == oh.kv && og.archive == oh.archive && og.ngram_pinned == oh.ngram_pinned);
            }
            // KV in VRAM: the plan as before (the pool fields stay 0)
            const Out ov = plan(r5080(3, 127e9));
            EXPECT(ov.kv == 0 && line(r5080(3, 127e9), ov).find("KV pool in RAM") == std::string::npos);
        }
        // review L3: a non-finite value keeps the JSON valid
        In n = qfn(kOurTotal, 200e9, 3);
        n.arch_user = HUGE_VAL;
        const std::string jn = json(n, plan(n));
        EXPECT(jn.find("inf") == std::string::npos && jn.find("nan") == std::string::npos);
        // JSON is well-formed enough for the server (quotes in reasons replaced)
        const std::string j = json(a, plan(a), 30e9);
        EXPECT(j.front() == '{' && j.back() == '}' && j.find("\"ngram_pinned\": false") != std::string::npos);
    }
    if (fails) {
        fprintf(stderr, "%d failure(s)\n", fails);
        return 1;
    }
    printf("host_budget_unit: ok\n");
    return 0;
}
