// CPU-only unit test of the runtime host-memory pressure response (src/host_pressure.h): PSI / vmstat parsing, the policy's
// stage escalation, recovery hysteresis and the flap back-off on fake readings, a simulated engine under a RAM hog, and the
// monitor thread on fake /proc files (QE_MEMP_PSI_FILE, QE_MEMP_VMSTAT_FILE, QE_MEMAVAIL_GB=@file). Built and run by
// tests/test_env.py and ctest.
#include "host_pressure.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <unistd.h>
#include <vector>

static int fails = 0;
#define EXPECT(c)                                                                         \
    do {                                                                                  \
        if (!(c)) { fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); fails++; } \
    } while (0)

using namespace qe_memp;

static const double GB = 1e9;

static Reading rd(double avail_gb, double some_now = 0, double some10 = 0, double full10 = 0, double swap = 0) {
    Reading r;
    r.n = 1;
    r.psi = true;
    r.avail = avail_gb * GB;
    r.margin = 8 * GB;
    r.some_now = some_now;
    r.some10 = some10;
    r.full10 = full10;
    r.swapin_mbs = swap;
    return r;
}
static Config cfg() {
    Config c;  // defaults: some 10 %, full 5 %, swap 8 MB/s, escalate 3 s (severe 1 s), recover 30 s, gap 4 GB, step 4 GB
    return c;
}

static void test_parse() {
    const char* psi =
        "some avg10=12.50 avg60=3.00 avg300=1.00 total=123456\n"
        "full avg10=4.25 avg60=1.00 avg300=0.50 total=6789\n";
    Psi p = parse_psi(psi);
    EXPECT(p.ok);
    EXPECT(p.some10 == 12.5 && p.full10 == 4.25);
    EXPECT(p.some_us == 123456ULL && p.full_us == 6789ULL);
    EXPECT(!parse_psi("").ok);
    EXPECT(!parse_psi(nullptr).ok);
    EXPECT(!parse_psi("garbage\n").ok);
    Psi q = parse_psi("some avg10=0.00 avg60=0.00 avg300=0.00 total=5\n");  // (old kernels / cgroups: no "full" line)
    EXPECT(q.ok && q.some_us == 5 && q.full_us == 0);
    unsigned long long v = 0;
    const char* vm = "nr_free_pages 100\npswpin 4242\npswpout 99\n";
    EXPECT(parse_vmstat(vm, "pswpin", &v) && v == 4242);
    EXPECT(parse_vmstat(vm, "pswpout", &v) && v == 99);
    EXPECT(!parse_vmstat(vm, "pswp", &v));  // (a prefix of a key is not the key)
    EXPECT(!parse_vmstat("", "pswpin", &v));
}

static void test_classify() {
    const Config c = cfg();
    EXPECT(!Policy::classify(c, rd(40)).hot);
    EXPECT(Policy::classify(c, rd(40)).calm);
    EXPECT(Policy::classify(c, rd(7)).hot);                 // below the margin
    EXPECT(Policy::classify(c, rd(10, 12)).hot);            // PSI some over the interval, free RAM near the margin
    EXPECT(!Policy::classify(c, rd(40, 30, 30)).hot);       // PSI on a roomy host (page-cache refaults elsewhere): not hot
    EXPECT(!Policy::classify(c, rd(100, 60, 60, 20)).severe);
    EXPECT(!Policy::classify(c, rd(10, 2, 15)).hot);        // a high avg10 that is decaying: not hot
    EXPECT(Policy::classify(c, rd(10, 6, 15)).hot);         // avg10 high and the interval still at half of it
    EXPECT(Policy::classify(c, rd(7)).deficit_only && !Policy::classify(c, rd(7, 20)).deficit_only);
    EXPECT(!Policy::classify(c, rd(40, 0, 0, 0, 50)).hot);  // swap-in on a roomy host: pages coming back
    EXPECT(Policy::classify(c, rd(10, 0, 0, 0, 10)).hot);   // swap-in near the margin (10 < 8 + 4)
    EXPECT(!Policy::classify(c, rd(10)).calm);              // not hot, but not roomy either
    EXPECT(Policy::classify(c, rd(10, 12, 0, 6)).severe);   // PSI full
    EXPECT(Policy::classify(c, rd(3)).severe);              // deficit 5 GB > margin / 2
    EXPECT(!Policy::classify(c, rd(7)).severe);
    EXPECT(Policy::classify(c, rd(40, 5, 5)).calm);         // PSI on a roomy host does not block recovery (F4)
    EXPECT(!Policy::classify(c, rd(40, 0, 0, 0, 2)).calm);  // swap-in still coming back: not yet
    Reading nopsi = rd(40);
    nopsi.psi = false;
    nopsi.some_now = 50;  // (ignored without PSI)
    EXPECT(!Policy::classify(c, nopsi).hot && Policy::classify(c, nopsi).calm);
}

static void test_escalation() {
    Policy p(cfg());
    double t = 0;
    auto ev = p.step(rd(40), t);
    EXPECT(p.stage() == 0 && !ev.act);
    Reading none = rd(40);
    none.n = 0;  // (no reading yet: nothing)
    EXPECT(p.step(none, t).to == 0);
    // pressure: stage 1 at once, due
    ev = p.step(rd(7), t += 1);
    EXPECT(ev.from == 0 && ev.to == 1 && ev.act);
    p.acted(t, true);
    EXPECT(p.stage() == 1 && !p.act_due());
    // still hot 1 s later: not yet (3 s escalate interval)
    EXPECT(!p.step(rd(7), t += 1).act);
    EXPECT(!p.step(rd(7), t += 1).act);
    EXPECT(p.step(rd(7), t += 1).act);  // 3 s after the action
    // stage 1 has nothing more: stage 2 due at once; stage 2 gives
    p.acted(t, false);
    EXPECT(p.stage() == 2 && p.act_due());
    p.acted(t, true);
    EXPECT(p.stage() == 2 && !p.act_due());
    // stage 2 empty next time -> 3, 3 gives twice, then empty -> 4 -> 5 -> exhausted
    p.step(rd(7), t += 3);
    p.acted(t, false);
    EXPECT(p.stage() == 3);
    p.acted(t, true);
    p.step(rd(7), t += 3);
    EXPECT(p.act_due());
    p.acted(t, true);
    EXPECT(p.stage() == 3);
    p.step(rd(7), t += 3);
    p.acted(t, false);
    EXPECT(p.stage() == 4 && p.act_due());
    p.acted(t, false);
    EXPECT(p.stage() == 5 && !p.act_due());  // (stage 5 never in the same pass: the next action time)
    EXPECT(!p.step(rd(7), t += 1).act);
    EXPECT(p.step(rd(7), t += 2).act);
    p.acted(t, false);
    EXPECT(p.stage() == 5 && p.exhausted() && !p.act_due());
    // never past 5
    p.step(rd(7), t += 3);
    p.acted(t, false);
    EXPECT(p.stage() == 5);
}

// WAIT (nothing back yet, something left) retries at the next action; kWaits in a row move on; at max_stage it keeps waiting
static void test_wait() {
    Policy p(cfg());
    double t = 0;
    p.step(rd(7), t);
    p.acted(t, Policy::EMPTY);  // 1 -> 2
    p.acted(t, Policy::EMPTY);  // 2 -> 3
    EXPECT(p.stage() == 3 && p.act_due());
    p.acted(t, Policy::WAIT);
    EXPECT(p.stage() == 3 && !p.act_due());
    p.step(rd(7), t += 3);
    p.acted(t, Policy::GAVE);  // (a GAVE resets the count)
    p.step(rd(7), t += 3);
    p.acted(t, Policy::WAIT);
    p.step(rd(7), t += 3);
    p.acted(t, Policy::WAIT);
    EXPECT(p.stage() == 3);
    p.step(rd(7), t += 3);
    p.acted(t, Policy::WAIT);  // third in a row
    EXPECT(p.stage() == 4 && p.act_due());
    p.acted(t, Policy::EMPTY);
    EXPECT(p.stage() == 5 && !p.act_due());
    for (int i = 0; i < 10; i++) {
        p.step(rd(7), t += 3);
        EXPECT(p.act_due());
        p.acted(t, Policy::WAIT);
        EXPECT(p.stage() == 5 && !p.exhausted());
    }
    // the H2 case: one pass with every lossy stage "not now" cannot reach stage 5
    Policy q(cfg());
    q.step(rd(7), 0);
    for (int g = 0; q.act_due() && g < 8; g++) q.acted(0, q.stage() <= RELEASE ? Policy::EMPTY : Policy::WAIT);
    EXPECT(q.stage() == SHRINK);
}

// a reading older than 3 sample periods is ignored (no escalation, no recovery clock)
static void test_stale() {
    Policy p(cfg());
    Reading r = rd(7);
    r.t = 100;
    EXPECT(p.step(r, 101).to == HOLD);
    p.acted(101, true);
    EXPECT(!p.step(r, 110).act);  // 10 s old: ignored
    r.t = 110;
    EXPECT(p.step(r, 110.5).act);
    Reading calm = rd(40);
    calm.t = 120;
    p.acted(110.5, true);
    for (int i = 0; i < 60; i++) p.step(calm, 121 + i);  // (stale after 4 s: the recovery clock never reaches 30 s)
    EXPECT(p.stage() == 1);
}

static void test_severe_is_faster() {
    Policy p(cfg());
    double t = 0;
    p.step(rd(3), t);
    p.acted(t, true);
    EXPECT(p.step(rd(3), t += 1).act);  // severe: 1 s
    Policy q(cfg());
    t = 0;
    q.step(rd(7), t);
    q.acted(t, true);
    EXPECT(!q.step(rd(7), t += 1).act);  // not severe: 3 s
}

static void test_max_stage_and_off() {
    Config c = cfg();
    c.max_stage = 3;
    Policy p(c);
    double t = 0;
    p.step(rd(7), t);
    p.acted(t, false);
    p.acted(t, false);
    EXPECT(p.stage() == 3);
    p.acted(t, false);
    EXPECT(p.stage() == 3 && p.exhausted());
    Config off = cfg();
    off.on = false;
    Policy q(off);
    for (int i = 0; i < 20; i++) EXPECT(q.step(rd(1, 90, 90, 90, 500), i).to == 0 && !q.act_due());
}

static void test_recovery_hysteresis() {
    Policy p(cfg());
    double t = 0;
    p.step(rd(7), t);
    p.acted(t, false);  // 2
    p.acted(t, false);  // 3
    p.acted(t, true);
    EXPECT(p.stage() == 3);
    // calm from t + 1 to t + 29 (28 s): still 3
    for (int i = 1; i <= 29; i++) p.step(rd(40), t + i);
    EXPECT(p.stage() == 3);
    // one in-between reading (MemAvailable 10 GB: not hot, not roomy) resets the calm clock
    p.step(rd(10), t + 30);
    EXPECT(p.stage() == 3);
    for (int i = 31; i <= 60; i++) p.step(rd(40), t + i);  // (calm since t + 31: 29 s)
    EXPECT(p.stage() == 3);
    p.step(rd(40), t + 61);  // 30 s of calm
    EXPECT(p.stage() == 2);
    // the next step down needs another full period
    for (int i = 62; i <= 90; i++) p.step(rd(40), t + i);
    EXPECT(p.stage() == 2);
    p.step(rd(40), t + 91);
    EXPECT(p.stage() == 1);
    for (int i = 92; i <= 121; i++) p.step(rd(40), t + i);
    EXPECT(p.stage() == 0);
    EXPECT(!p.act_due());
}

// readings at t0 + 1 .. t0 + n (calm); returns the last time
static double calm_for(Policy& p, double t0, int n) {
    for (int i = 1; i <= n; i++) p.step(rd(40), t0 + i);
    return t0 + n;
}

static void test_flap_backoff() {
    Policy p(cfg());
    double t = 0;
    p.step(rd(7), t);
    p.acted(t, true);
    t = calm_for(p, t, 30);  // (calm since t + 1: 29 s)
    EXPECT(p.stage() == 1);
    t = calm_for(p, t, 1);
    EXPECT(p.stage() == 0);
    EXPECT(p.recover_mult() == 1);
    // pressure back 9 s after the recovery (< 2 x 30 s): the next recovery waits 60 s
    t += 9;
    p.step(rd(7), t);
    EXPECT(p.stage() == 1);
    EXPECT(p.recover_mult() == 2);
    p.acted(t, true);
    t = calm_for(p, t, 60);
    EXPECT(p.stage() == 1);
    t = calm_for(p, t, 1);
    EXPECT(p.stage() == 0);
    // back again soon each time: 4x, 8x, capped at 8x
    for (int k = 0; k < 5; k++) {
        t += 5;
        p.step(rd(7), t);
        p.acted(t, true);
        t = calm_for(p, t, (int)(30 * p.recover_mult()) + 1);
        EXPECT(p.stage() == 0);
    }
    EXPECT(p.recover_mult() == 8);
    // pressure that comes back late (> 2x the recovery time) does not double
    Policy r(cfg());
    r.step(rd(7), 0);
    r.acted(0, true);
    double u = calm_for(r, 0, 31);
    EXPECT(r.stage() == 0);
    r.step(rd(7), u + 61);
    EXPECT(r.recover_mult() == 1);
    // 10 min at stage 0: back to 1x
    t = calm_for(p, t, 601);
    EXPECT(p.recover_mult() == 1);
    // several hot readings in one episode double once
    Policy q(cfg());
    q.step(rd(7), 0);
    q.acted(0, true);
    calm_for(q, 0, 31);
    q.step(rd(7), 35);
    q.step(rd(7), 36);
    q.step(rd(7), 37);
    EXPECT(q.recover_mult() == 2);
}

// a simulated engine: archive (free slabs, used copies, prefixes, n-gram table) next to a hog that grows, holds and leaves;
// MemAvailable follows what the engine gives back. No flapping: the stage changes a bounded number of times, never goes
// down while the hog holds, ends at 0, and nothing is shed after the hog left.
static void test_sim() {
    Policy p(cfg());
    double free_slabs = 6, used = 20, prefixes = 3, ngram = 52;  // GB
    const double host_other = 30;
    double hog = 0, shed_total = 0;
    int changes = 0, prev = 0, max_stage = 0;
    bool held = false, shed_after_leave = false, down_while_hog = false;
    for (int t = 0; t < 900; t++) {
        // the hog: grows 2 GB/s from t=10 to 60 GB, holds until t=200, then leaves
        if (t >= 10 && t < 200) hog = std::min(60.0, hog + 2);
        if (t >= 200) hog = 0;
        const double avail = 128 - host_other - (free_slabs + used + prefixes + ngram) - hog;
        p.step(rd(avail, avail < 8 ? 30 : 0, avail < 8 ? 20 : 0), t);
        for (int g = 0; p.act_due() && g < 8; g++) {
            bool gave = false;
            const double before = free_slabs + used + prefixes + ngram;
            switch (p.stage()) {
            case HOLD: gave = !held; held = true; break;
            case RELEASE: gave = free_slabs > 0; free_slabs = 0; break;
            case SHRINK: gave = used > 0; used = std::max(0.0, used - 4); break;
            case PREFIXES: gave = prefixes > 0; prefixes = 0; break;
            case NGRAM: gave = ngram > 0; ngram = 0; break;
            }
            const double d = before - (free_slabs + used + prefixes + ngram);
            shed_total += d;
            if (t > 200 && d > 0) shed_after_leave = true;
            p.acted(t, gave);
        }
        if (p.stage() == 0) held = false;
        if (p.stage() != prev) {
            changes++;
            if (t < 40 && p.stage() < prev) down_while_hog = true;  // (not while the hog still grows)
            prev = p.stage();
        }
        max_stage = std::max(max_stage, p.stage());
    }
    EXPECT(p.stage() == 0);
    EXPECT(!shed_after_leave);
    EXPECT(!down_while_hog);
    EXPECT(max_stage == 5);  // (the hog's 60 GB needs the n-gram table: 128 - 30 - 81 = 17 GB free before it)
    EXPECT(ngram == 0 && used == 0 && prefixes == 0);
    EXPECT(changes <= 12);  // 0 -> 5 up (5 changes), 5 -> 0 down (5 changes): no flapping
    fprintf(stderr, "sim: max stage %d, %d stage changes, %.0f GB given back\n", max_stage, changes, shed_total);
    // a hog that fits after the archive shrank: stops at stage 3, the table stays pinned
    Policy q(cfg());
    free_slabs = 6, used = 20, prefixes = 3, ngram = 52, hog = 0, held = false;
    int qmax = 0;
    for (int t = 0; t < 300; t++) {
        if (t >= 10 && t < 100) hog = std::min(20.0, hog + 2);
        if (t >= 100) hog = 0;
        const double avail = 128 - host_other - (free_slabs + used + prefixes + ngram) - hog;
        q.step(rd(avail, avail < 8 ? 30 : 0, avail < 8 ? 20 : 0), t);
        for (int g = 0; q.act_due() && g < 8; g++) {
            bool gave = false;
            switch (q.stage()) {
            case HOLD: gave = !held; held = true; break;
            case RELEASE: gave = free_slabs > 0; free_slabs = 0; break;
            case SHRINK: gave = used > 0; used = std::max(0.0, used - 4); break;
            case PREFIXES: gave = prefixes > 0; prefixes = 0; break;
            case NGRAM: gave = ngram > 0; ngram = 0; break;
            }
            q.acted(t, gave);
        }
        qmax = std::max(qmax, q.stage());
    }
    EXPECT(qmax == 3 && ngram == 52 && prefixes == 3);
    // shrink returns memory only every 3rd action (a holder keeps slabs): WAIT between, the archive still goes first and the
    // table is unpinned only once the archive and the prefixes are gone
    Policy w(cfg());
    free_slabs = 6, used = 20, prefixes = 3, ngram = 52, hog = 0, held = false;
    int nact = 0;
    bool ngram_before_archive = false;
    for (int t = 0; t < 400; t++) {
        if (t >= 10 && t < 250) hog = std::min(60.0, hog + 2);
        if (t >= 250) hog = 0;
        const double avail = 128 - host_other - (free_slabs + used + prefixes + ngram) - hog;
        w.step(rd(avail, avail < 8 ? 30 : 0, avail < 8 ? 20 : 0), t);
        for (int g = 0; w.act_due() && g < 8; g++) {
            Policy::Result res = Policy::EMPTY;
            switch (w.stage()) {
            case HOLD: res = held ? Policy::EMPTY : Policy::GAVE; held = true; break;
            case RELEASE: res = free_slabs > 0 ? Policy::GAVE : Policy::EMPTY; free_slabs = 0; break;
            case SHRINK:
                if (used <= 0) res = Policy::EMPTY;
                else if (++nact % 3) res = Policy::WAIT;
                else { used = std::max(0.0, used - 4); res = Policy::GAVE; }
                break;
            case PREFIXES: res = prefixes > 0 ? Policy::GAVE : Policy::EMPTY; prefixes = 0; break;
            case NGRAM:
                if (used > 0) ngram_before_archive = true;
                res = ngram > 0 ? Policy::GAVE : Policy::EMPTY;
                ngram = 0;
                break;
            }
            w.acted(t, res);
        }
    }
    EXPECT(!ngram_before_archive);
    EXPECT(used == 0 && ngram == 0 && w.stage() == 0);
}

// the monitor thread on fake files: rates from the total deltas
static void test_monitor() {
    char dir[] = "/tmp/qe_memp_XXXXXX";
    const char* tmp = getenv("TMPDIR");
    std::string base = tmp && *tmp ? std::string(tmp) + "/qe_memp_XXXXXX" : dir;
    std::vector<char> b(base.begin(), base.end());
    b.push_back(0);
    if (!mkdtemp(b.data())) { fprintf(stderr, "FAIL mkdtemp\n"); fails++; return; }
    const std::string d = b.data(), psi = d + "/psi", vm = d + "/vmstat", av = d + "/avail";
    auto put = [](const std::string& f, const std::string& s) {  // (write + rename: the sampler never reads half a file)
        const std::string tmpf = f + ".tmp";
        FILE* o = fopen(tmpf.c_str(), "w");
        if (o) { fputs(s.c_str(), o); fclose(o); rename(tmpf.c_str(), f.c_str()); }
    };
    put(psi, "some avg10=0.00 avg60=0.00 avg300=0.00 total=0\nfull avg10=0.00 avg60=0.00 avg300=0.00 total=0\n");
    put(vm, "pswpin 0\n");
    put(av, "40\n");
    setenv("QE_MEMP_PSI_FILE", psi.c_str(), 1);
    setenv("QE_MEMP_VMSTAT_FILE", vm.c_str(), 1);
    setenv("QE_MEMAVAIL_GB", ("@" + av).c_str(), 1);
    setenv("QE_MEM_MARGIN_GB", "8", 1);
    setenv("QE_MEMP_SAMPLE_MS", "200", 1);
    Monitor& m = Monitor::get();
    m.start();
    m.start();  // (idempotent)
    usleep(500000);
    Reading r = m.last();
    EXPECT(r.n >= 1);
    EXPECT(r.psi);
    EXPECT(r.some_now < 1 && r.swapin_mbs < 0.1);
    EXPECT(r.avail > 39 * GB && r.avail < 41 * GB);
    EXPECT(r.margin == 8 * GB);
    // 50 % stall and 25,000 pages/s swap-in from now on (totals grow per sample), MemAvailable 6 GB
    const long pg = sysconf(_SC_PAGESIZE) > 0 ? sysconf(_SC_PAGESIZE) : 4096;
    unsigned long long some = 0, sw = 0;
    put(av, "6\n");
    for (int i = 0; i < 12; i++) {
        some += 100000;  // 0.1 s of stall per 0.2 s
        sw += 5000;      // per 0.2 s
        put(psi, "some avg10=40.00 avg60=10.00 avg300=2.00 total=" + std::to_string(some) +
                     "\nfull avg10=6.00 avg60=1.00 avg300=0.50 total=" + std::to_string(some / 4) + "\n");
        put(vm, "pswpin " + std::to_string(sw) + "\n");
        usleep(200000);
    }
    r = m.last();
    // (the sampler and the writer are not in step: wide bounds)
    EXPECT(r.some_now > 20 && r.some_now <= 100);
    EXPECT(r.some10 == 40 && r.full10 == 6);
    const double want_mbs = 25000.0 * pg / 1e6;
    EXPECT(r.swapin_mbs > want_mbs * 0.4 && r.swapin_mbs < want_mbs * 2.5);
    EXPECT(r.avail < 6.5 * GB);
    Policy p(Config::from_env(8 * GB));
    const Policy::Event ev = p.step(r, 1);
    EXPECT(ev.f.hot && ev.f.severe && ev.to == HOLD);
    fprintf(stderr, "monitor: some %.1f %%, swap-in %.1f MB/s (want ~%.1f), MemAvailable %.1f GB\n", r.some_now, r.swapin_mbs, want_mbs,
            r.avail / 1e9);
    EXPECT(json(p, r).find("\"stage\": 1") != std::string::npos);
    EXPECT(describe(r).find("PSI some") != std::string::npos);
    unlink(psi.c_str());
    unlink(vm.c_str());
    unlink(av.c_str());
    rmdir(d.c_str());
}

static void test_env_config() {
    setenv("QE_MEMP_SOME_PCT", "20", 1);
    setenv("QE_MEMP_MAX_STAGE", "9", 1);
    setenv("QE_MEMP_RECOVER_S", "5", 1);
    Config c = Config::from_env(16e9);
    EXPECT(c.some_pct == 20 && c.max_stage == 5 && c.recover_s == 5);
    EXPECT(c.recover_gap == 8e9);  // max(4, margin / 2)
    setenv("QE_MEMP", "0", 1);
    EXPECT(!Config::from_env(8e9).on);
    unsetenv("QE_MEMP");
    unsetenv("QE_MEMP_SOME_PCT");
    unsetenv("QE_MEMP_MAX_STAGE");
    unsetenv("QE_MEMP_RECOVER_S");
}

int main() {
    test_parse();
    test_classify();
    test_escalation();
    test_wait();
    test_stale();
    test_severe_is_faster();
    test_max_stage_and_off();
    test_recovery_hysteresis();
    test_flap_backoff();
    test_sim();
    test_env_config();
    test_monitor();
    if (fails) {
        fprintf(stderr, "host_pressure_unit: %d FAILED\n", fails);
        return 1;
    }
    fprintf(stderr, "host_pressure_unit: all passed\n");
    return 0;
}
