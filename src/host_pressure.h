#pragma once
// Runtime host-memory pressure (both engines): a boot-time RAM plan cannot keep the host out of swap, because other
// programs grow while the server runs (issue #25: a 128 GB box at ~110 GB RSS swapped and every rank stalled 51.6 s in a
// launch). The engine has to see pressure while it runs and give memory back, in stages, and keep serving.
//
// Sensing (Monitor, its own thread, never on a step): every QE_MEMP_SAMPLE_MS (1000)
//   - /proc/pressure/memory: PSI "some" / "full" avg10 (%) and the stall time over the last interval (total deltas);
//   - /proc/vmstat pswpin: the swap-in rate (MB/s): pages coming back from swap;
//   - MemAvailable (host_mem.h, the cgroup limit included) against the margin (QE_MEM_MARGIN_GB).
// Policy (one object per engine, stepped by the engine's ~1/s memory check between steps):
//   hot     = MemAvailable below the margin, or PSI some over the last interval >= QE_MEMP_SOME_PCT (10), or avg10 at it
//             while the last interval is still at half of it, or swap-in >= QE_MEMP_SWAPIN_MBS (8) with MemAvailable
//             below margin + the recovery gap (swap-in on a host with room is pages coming back, not pressure);
//   severe  = PSI full avg10 >= QE_MEMP_FULL_PCT (5), swap-in >= 4x its threshold, or a deficit above half the margin:
//             the next stage comes after 1 s instead of QE_MEMP_ESCALATE_S (3);
//   calm    = not hot, swap-in < 1/8 of its threshold and MemAvailable >= margin + QE_MEMP_RECOVER_GB (default max(4 GB,
//             margin / 2)); PSI and swap-in count as hot only below that line (the PSI file is host-wide).
// Stages (each entered with one log line by the engine):
//   1 hold:     no growth (archive pinning held, archive capped at its size, no new pinned prefix, no optional copy);
//   2 release:  the archive's free pinned slabs go back to the host;
//   3 shrink:   the archive sheds QE_MEMP_STEP_GB (4) per action, oldest copies first;
//   4 prefixes: pinned prefixes are dropped (QE_MEMP_STEP_GB per action);
//   5 n-gram:   (Flash-Next) the pinned n-gram table is unpinned, rows read from the page cache (the boot fallback path).
// While hot, the current stage acts again every escalate interval; a stage that has nothing left to give moves to the
// next one at once. Recovery: one stage down per QE_MEMP_RECOVER_S (30) of continuous calm; growth resumes at stage 0.
// Pressure that comes back within 2x the recovery time of the last step down doubles the recovery time (up to 8x;
// back to 1x after 10 min at stage 0): no flapping between shedding and regrowing.
// QE_MEMP=0: off (stage stays 0: the MemAvailable deficit guard of host_mem.h alone, as in 1.3.6). QE_MEMP_MAX_STAGE
// (1..5, default 5) stops the escalation at that stage. Tests: QE_MEMP_PSI_FILE / QE_MEMP_VMSTAT_FILE read these files
// instead of /proc (QE_MEMAVAIL_GB=@file fakes MemAvailable, host_mem.h).
#include "env.h"
#include "host_mem.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <unistd.h>

namespace qe_memp {

enum Stage { NONE = 0, HOLD = 1, RELEASE = 2, SHRINK = 3, PREFIXES = 4, NGRAM = 5 };
inline const char* stage_name(int s) {
    static const char* n[] = {"none", "hold growth", "release free slabs", "shrink the archive", "drop pinned prefixes",
                              "unpin the n-gram table"};
    return s >= 0 && s <= 5 ? n[s] : "?";
}

// ---- parsing (pure: unit-tested on fixed text)
// "some avg10=1.23 avg60=... total=123\nfull avg10=... total=...\n"
struct Psi {
    bool ok = false;
    double some10 = 0, full10 = 0;
    unsigned long long some_us = 0, full_us = 0;
};
inline Psi parse_psi(const char* text) {
    Psi p;
    if (!text) return p;
    bool some = false;
    for (const char* line = text; line && *line;) {
        double a10 = 0;
        unsigned long long tot = 0;
        const char* av = strstr(line, "avg10=");
        const char* tt = strstr(line, "total=");
        const char* nl = strchr(line, '\n');
        if (av && tt && (!nl || (av < nl && tt < nl))) {
            a10 = atof(av + 6);
            tot = strtoull(tt + 6, nullptr, 10);
            if (!strncmp(line, "some", 4)) { p.some10 = a10; p.some_us = tot; some = true; }
            else if (!strncmp(line, "full", 4)) { p.full10 = a10; p.full_us = tot; }
        }
        line = nl ? nl + 1 : nullptr;
    }
    p.ok = some;
    return p;
}
// the value of `key` in /proc/vmstat text ("pswpin 123\n"); false when absent
inline bool parse_vmstat(const char* text, const char* key, unsigned long long* v) {
    if (!text) return false;
    const size_t kl = strlen(key);
    for (const char* line = text; line && *line;) {
        if (!strncmp(line, key, kl) && line[kl] == ' ') {
            *v = strtoull(line + kl + 1, nullptr, 10);
            return true;
        }
        const char* nl = strchr(line, '\n');
        line = nl ? nl + 1 : nullptr;
    }
    return false;
}
inline bool read_text(const char* path, std::string& out) {
    out.clear();
    FILE* f = fopen(path, "r");
    if (!f) return false;
    char b[4096];
    size_t n;
    while ((n = fread(b, 1, sizeof b, f)) > 0) out.append(b, n);
    fclose(f);
    return true;
}

// one reading of the monitor: the rates over the last interval
struct Reading {
    double t = 0;           // seconds (steady clock)
    bool psi = false;       // PSI available
    double some10 = 0, full10 = 0;  // avg10, %
    double some_now = 0, full_now = 0;  // stall % over the last interval (total deltas)
    double swapin_mbs = 0;  // swap-in rate over the last interval
    double avail = 0, margin = 0;  // bytes
    long long n = 0;        // readings so far (0: none yet)
};

struct Config {
    bool on = true;
    int max_stage = 5;
    double some_pct = 10, full_pct = 5, swap_mbs = 8;
    double esc_s = 3, severe_s = 1, recover_s = 30;
    double recover_gap = 4e9, step = 4e9;
    double sample_s = 1;
    static Config from_env(double margin) {
        Config c;
        c.on = env_int("QE_MEMP", 1) != 0;
        c.max_stage = std::max(1, std::min(5, env_int("QE_MEMP_MAX_STAGE", 5)));
        c.some_pct = std::max(0.1, env_double("QE_MEMP_SOME_PCT", 10.0));
        c.full_pct = std::max(0.1, env_double("QE_MEMP_FULL_PCT", 5.0));
        c.swap_mbs = std::max(0.01, env_double("QE_MEMP_SWAPIN_MBS", 8.0));
        c.esc_s = std::max(0.0, env_double("QE_MEMP_ESCALATE_S", 3.0));
        c.severe_s = std::min(c.esc_s, 1.0);
        c.recover_s = std::max(0.0, env_double("QE_MEMP_RECOVER_S", 30.0));
        c.recover_gap = std::max(0.0, env_double("QE_MEMP_RECOVER_GB", std::max(4.0, margin / 2e9))) * 1e9;
        c.step = std::max(0.1, env_double("QE_MEMP_STEP_GB", 4.0)) * 1e9;
        c.sample_s = std::max(50, env_int("QE_MEMP_SAMPLE_MS", 1000)) / 1000.0;
        return c;
    }
};

// ---- the policy (pure: no clock, no files; the unit test drives it with fake readings)
class Policy {
public:
    explicit Policy(const Config& c = Config()) : c_(c) {}
    struct Flags {
        bool hot = false, severe = false, calm = false;
        bool deficit_only = false;  // hot only because MemAvailable is below the margin (no stalls, no swap-in)
        double deficit = 0;
    };
    static Flags classify(const Config& c, const Reading& r) {
        Flags f;
        f.deficit = std::max(0.0, r.margin - r.avail);
        const bool roomy = r.avail >= r.margin + c.recover_gap;
        // (PSI and swap-in count only while free RAM is near the margin: /proc/pressure/memory is host-wide, and page-cache
        // refaults of another program streaming a large file, or of the n-gram table read from the page cache, stall tasks
        // on a host with plenty of MemAvailable; shedding the archive there frees nothing anyone needs; review H1)
        const bool psi_hot = r.psi && !roomy && (r.some_now >= c.some_pct || (r.some10 >= c.some_pct && r.some_now >= c.some_pct / 2));
        const bool swap_hot = r.swapin_mbs >= c.swap_mbs && !roomy;
        f.hot = f.deficit > 0 || psi_hot || swap_hot;
        f.deficit_only = f.deficit > 0 && !psi_hot && !swap_hot;
        f.severe = f.hot && ((r.psi && r.full10 >= c.full_pct) || r.swapin_mbs >= 4 * c.swap_mbs || f.deficit > r.margin / 2);
        // (PSI is not asked for calm: on a roomy host it can stay up for good from other programs' page-cache refaults or the
        // n-gram table read from the page cache after stage 5, and the stage would never step back; follow-up F4)
        f.calm = !f.hot && roomy && r.swapin_mbs < c.swap_mbs / 8;
        return f;
    }
    // what one step changed (the engine logs it)
    struct Event {
        int from = 0, to = 0;      // stage before / after (to < from: a recovery step)
        bool act = false;          // an action of stage() is due now
        Flags f;
    };
    // one reading at time now (seconds); the engine then runs act() while act_due() and reports with acted()
    Event step(const Reading& r, double now) {
        Event ev;
        ev.from = ev.to = s_;
        if (!c_.on || r.n == 0) return ev;
        // a reading older than 3 sample periods (the monitor thread stalled, e.g. its own pages reclaimed): neither escalation
        // nor the recovery clock act on it (review L1)
        if (r.t > 0 && now - r.t > 3 * c_.sample_s + 1) return ev;
        ev.f = classify(c_, r);
        if (ev.f.hot) {
            calm_since_ = -1;
            // (pressure back soon after a step down: the recovery was too early, the next one waits twice as long)
            if (!hot_prev_ && last_down_ >= 0 && now - last_down_ < 2 * c_.recover_s * mult_) mult_ = std::min(8.0, mult_ * 2);
            if (!hot_prev_) last_down_ = -1;
            if (s_ == NONE) {
                s_ = HOLD;
                due_ = true;
            } else if (now - t_act_ >= (ev.f.severe ? c_.severe_s : c_.esc_s)) {
                due_ = true;
            }
        } else if (ev.f.calm) {
            if (calm_since_ < 0) calm_since_ = now;
            if (s_ > NONE && now - calm_since_ >= c_.recover_s * mult_) {
                s_--;
                waits_ = 0;
                calm_since_ = now;  // (the next step down needs another full calm period)
                last_down_ = now;
                exhausted_ = false;
                if (s_ == NONE) zero_since_ = now;
            }
        } else {
            calm_since_ = -1;  // in between: hold the stage, no recovery clock
        }
        if (s_ == NONE && mult_ > 1 && zero_since_ >= 0 && now - zero_since_ >= 600) mult_ = 1;
        hot_prev_ = ev.f.hot;
        ev.to = s_;
        ev.act = due_;
        return ev;
    }
    bool act_due() const { return due_; }
    // what stage()'s action did: GAVE (it shed something; stage 1: it started holding), WAIT (nothing now, but something is
    // left: a holder keeps a slab, compaction spent its budget, a prompt is in flight; review H2), EMPTY (nothing left)
    enum Result { EMPTY = 0, GAVE = 1, WAIT = 2 };
    // EMPTY moves to the next stage, due at once (up to max_stage), except into stage 5 (one-way until the next start): that
    // comes at the next action time if the pressure is still there, never in the same pass. kWaits WAITs in a row at one
    // stage count as EMPTY (a holder that never lets go must not stop the response; at max_stage it just waits). At
    // max_stage with nothing to give: exhausted (logged once)
    static constexpr int kWaits = 3;
    void acted(double now, Result res) {
        due_ = false;
        t_act_ = now;
        if (res == GAVE) { waits_ = 0; return; }
        if (res == WAIT && (++waits_ < kWaits || s_ >= c_.max_stage)) return;
        waits_ = 0;
        if (s_ < c_.max_stage) {
            s_++;
            due_ = s_ != NGRAM;
        } else {
            exhausted_ = true;
        }
    }
    void acted(double now, bool gave) { acted(now, gave ? GAVE : EMPTY); }
    int stage() const { return s_; }
    bool exhausted() const { return exhausted_; }
    double recover_mult() const { return mult_; }
    const Config& config() const { return c_; }

private:
    Config c_;
    int s_ = NONE;
    bool due_ = false, exhausted_ = false, hot_prev_ = false;
    int waits_ = 0;
    double t_act_ = -1e18, calm_since_ = -1, last_down_ = -1, zero_since_ = -1, mult_ = 1;
};

// the stage the engine is at (published by the engine's check; read by the pinners: >= HOLD holds growth)
inline std::atomic<int>& g_stage() {
    static std::atomic<int> s{0};
    return s;
}
inline bool holding() { return g_stage().load(std::memory_order_relaxed) >= HOLD; }

// ---- the monitor: one background thread per process, sampling the sources
class Monitor {
public:
    static Monitor& get() {
        static Monitor* m = new Monitor;  // (never destroyed: engine threads may read it until the process ends)
        return *m;
    }
    // starts the thread once (idempotent); off (QE_MEMP=0): no thread
    void start() {
        std::lock_guard<std::mutex> l(mu_);
        if (started_ || env_int("QE_MEMP", 1) == 0) return;
        started_ = true;
        const char* pf = env_str("QE_MEMP_PSI_FILE");
        const char* vf = env_str("QE_MEMP_VMSTAT_FILE");
        psi_path_ = pf && *pf ? pf : "/proc/pressure/memory";
        vm_path_ = vf && *vf ? vf : "/proc/vmstat";
        sample_ms_ = std::max(50, env_int("QE_MEMP_SAMPLE_MS", 1000));
        std::string t;
        const bool psi = read_text(psi_path_.c_str(), t) && parse_psi(t.c_str()).ok;
        unsigned long long sw = 0;
        const bool vm = read_text(vm_path_.c_str(), t) && parse_vmstat(t.c_str(), "pswpin", &sw);
        fprintf(stderr, "[qe] host memory pressure monitor: PSI %s, swap-in %s, MemAvailable %.1f GB, margin %.1f GB (QE_MEMP=0: off)\n",
                psi ? "on" : "not available (kernel psi=0 or no /proc/pressure)", vm ? "on" : "not available",
                qe_hostmem::available(true) / 1e9, qe_hostmem::margin() / 1e9);
        std::thread([this] { loop(); }).detach();
    }
    Reading last() const {
        std::lock_guard<std::mutex> l(mu_);
        return r_;
    }

private:
    void loop() {
        const long pg = sysconf(_SC_PAGESIZE) > 0 ? sysconf(_SC_PAGESIZE) : 4096;
        bool have = false;
        Psi p0;
        unsigned long long sw0 = 0;
        double t0 = 0;
        std::string t;
        for (;;) {
            const double now = std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
            Psi p;
            if (read_text(psi_path_.c_str(), t)) p = parse_psi(t.c_str());
            unsigned long long sw = 0;
            const bool vm = read_text(vm_path_.c_str(), t) && parse_vmstat(t.c_str(), "pswpin", &sw);
            Reading r;
            r.t = now;
            r.psi = p.ok;
            r.some10 = p.some10;
            r.full10 = p.full10;
            r.avail = (double)qe_hostmem::available(true);
            r.margin = (double)qe_hostmem::margin();
            if (have && now > t0) {
                const double dt_us = (now - t0) * 1e6;
                if (p.ok && p0.ok && p.some_us >= p0.some_us) r.some_now = std::min(100.0, (p.some_us - p0.some_us) / dt_us * 100);
                if (p.ok && p0.ok && p.full_us >= p0.full_us) r.full_now = std::min(100.0, (p.full_us - p0.full_us) / dt_us * 100);
                if (vm && sw >= sw0) r.swapin_mbs = (double)(sw - sw0) * pg / 1e6 / (now - t0);
            }
            {
                std::lock_guard<std::mutex> l(mu_);
                r.n = r_.n + (have ? 1 : 0);
                r_ = r;
            }
            have = true;
            p0 = p;
            sw0 = vm ? sw : sw0;
            t0 = now;
            std::this_thread::sleep_for(std::chrono::milliseconds(sample_ms_));
        }
    }
    mutable std::mutex mu_;
    bool started_ = false;
    std::string psi_path_, vm_path_;
    int sample_ms_ = 1000;
    Reading r_;
};

inline double now_s() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

// "MemAvailable 7.1 GB (margin 8.0), PSI some 23.0 % (avg10 12.4), full 4.1 %, swap-in 31.0 MB/s" for the log lines
inline std::string describe(const Reading& r) {
    char b[256];
    if (r.psi)
        snprintf(b, sizeof b, "MemAvailable %.1f GB (margin %.1f), PSI some %.1f %% (avg10 %.1f), full avg10 %.1f %%, swap-in %.1f MB/s",
                 r.avail / 1e9, r.margin / 1e9, r.some_now, r.some10, r.full10, r.swapin_mbs);
    else
        snprintf(b, sizeof b, "MemAvailable %.1f GB (margin %.1f), swap-in %.1f MB/s (no PSI)", r.avail / 1e9, r.margin / 1e9,
                 r.swapin_mbs);
    return b;
}

// JSON of the state for /api/stats
inline std::string json(const Policy& p, const Reading& r) {
    char b[384];
    snprintf(b, sizeof b, "{\"on\": %s, \"stage\": %d, \"stage_name\": \"%s\", \"exhausted\": %s, \"mem_available_gb\": %.2f, "
             "\"margin_gb\": %.2f, \"psi\": %s, \"some_pct\": %.2f, \"some_avg10\": %.2f, \"full_avg10\": %.2f, \"swapin_mbs\": %.2f}",
             p.config().on ? "true" : "false", p.stage(), stage_name(p.stage()), p.exhausted() ? "true" : "false", r.avail / 1e9,
             r.margin / 1e9, r.psi ? "true" : "false", r.some_now, r.some10, r.full10, r.swapin_mbs);
    return b;
}

}  // namespace qe_memp
