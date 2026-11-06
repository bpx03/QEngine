#pragma once
// Host-RAM budget at boot (both engines; club-3090 discussion #1906): the defaults of the big host-memory consumers
// fit the machine instead of assuming a 251 GB host.
//
// On a 92 GB host with 2 cards the Flash-Next defaults of 1.3.3.1 pinned ~31 GB of host-backed expert groups + the
// 51.8 GB INT8 n-gram table and then grew a 48 GB KV archive (+ 8 GB pinned prefixes) on top: swap from boot, and the
// archive fought the page cache for the rest. The plan sizes them in this order:
//   1. host-backed experts (Flash-Next; forced by the cards' VRAM, never reduced): an estimate before the load
//      (qfn_expert_host_est), the real size logged after it;
//   2. the n-gram table (Flash-Next; a model weight of the PLE block, it cannot be dropped): pinned when it fits with
//      room for a minimal archive, else read from the file mapping through the page cache (the staged mode of
//      FLASH_NGRAM_PIN=0: a host thread copies a step's rows into a pinned buffer; no HMM needed). The page cache
//      keeps a share of what is left (ngram_cache) for its hot rows;
//   3. the KV archive + pinned prefixes get what is left, up to their defaults.
// room = MemTotal - reserve - base - fixed: reserve for the OS and the desktop (max(8 GB, 8 % of MemTotal),
// QE_HOST_RESERVE_GB), base for the server process (Python, torch, the vision encoder, CUDA; QE_HOST_BASE_GB, 6),
// fixed = the engine's other pinned buffers it knows (embeddings, snapshot buffers). The machine's size decides the plan: a
// short MemAvailable at boot only warns. The archive is a static budget (the runtime guard of host_mem.h keeps it below
// MemAvailable - margin as it fills and lets it grow when the memory comes back); the n-gram table goes to the page cache
// instead only when it really does not fit at the moment it is pinned (flash_capi.cpp).
// Explicit env values win (FLASH_NGRAM_PIN, FLASH_KV_ARCHIVE_GB / QE_KV_ARCHIVE_GB, FLASH_PREFIX_GB / QE_PREFIX_GB),
// with a warning when they do not fit. On a host where everything fits (our 251 GB host) the plan is the old
// defaults, exactly.
// No CUDA dependency: tests/host_budget_unit.cpp builds this header with a plain C++ compiler.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <chrono>
#include <thread>
#include <vector>

namespace qe_hostbudget {

constexpr double GB = 1e9;

struct In {
    const char* tag = "";       // log prefix
    bool qfn = false;            // Flash-Next (experts, n-gram table) or the 27B
    int cards = 1;
    double total = 0;            // MemTotal (bytes)
    double avail = 0;            // MemAvailable at boot (bytes)
    double cg_limit = 0, cg_anon = 0;  // cgroup v2 memory.max of the process and its anonymous bytes (0: no limit)
    double reserve = -1;         // OS + desktop reserve (< 0: max(8 GB, 8 % of MemTotal))
    double margin_now = -1;      // the runtime guard's margin for MemAvailable (< 0: max(8 GB, 5 %), host_mem.h)
    double base = 6 * GB;        // the server process (Python, torch, vision, CUDA)
    double fixed = 0;            // other host buffers the engine pins (embeddings, snapshots)
    double experts = 0;          // host-backed expert groups (estimate before the load)
    double ngram = 0;            // the n-gram table's bytes (0: none)
    int ngram_user = -1;         // FLASH_NGRAM_PIN: -1 unset (auto), 0 mapped, 1 pinned
    bool ngram_can_map = true;   // the staged (mapped) mode needs the INT8 table
    double arch_def = 0, pref_def = 0;    // default archive / pinned-prefix budgets (bytes)
    double arch_user = -1, pref_user = -1;  // explicit env values (bytes; < 0: unset)
    bool prefix_needs_archive = true;  // Flash-Next: the pinned prefixes live in the archive's pool (27B: separate)
    const char* arch_env = "";   // env names, for the messages
    const char* pref_env = "";
};

struct Out {
    double reserve = 0, room = 0, room_now = 0;
    bool ngram_pinned = false;
    double ngram_cache = 0;      // page-cache share kept for the mapped n-gram table
    double archive = 0, prefix = 0;
    bool reduced = false;        // a default was lowered to fit
    std::vector<std::string> why;   // what was reduced and why
    std::vector<std::string> warn;  // explicit values that do not fit
    bool experts_short = false;     // the host-backed experts do not fit what is free above the margin (warned; the pins' guard decides)
    bool avail_short = false;       // what is pinned at load does not fit what is free now: a previous server may still be exiting (plan_wait)
};

inline double default_reserve(double total) { return std::max(8 * GB, 0.08 * total); }

// round a reduced budget down to 0.1 GB (the defaults are never touched)
inline double floor01(double b) { return b <= 0 ? 0.0 : std::floor(b / (0.1 * GB)) * (0.1 * GB); }

inline std::string fmt(const char* f, double a = 0, double b = 0, double c = 0) {
    for (double* v : {&a, &b, &c}) if (!std::isfinite(*v)) *v = -1;  // (no "inf" / "nan" in the log and the JSON)
    char s[512];
    snprintf(s, sizeof s, f, a, b, c);
    return s;
}

// Flash-Next: host-backed expert bytes before the load. All routed groups (48 layers x 512 experts x 5 groups of
// 506,880 B = 62.3 GB) minus what the cards keep resident: per card its free VRAM at boot minus the non-expert part,
// 4.6 GB + 10.44 GB / cards (dense split over the TP ranks, KV pool, buffers, margins), fitted to the 1.3.4 boots:
// 2 cards 15.5 GB resident / card (31.2 GB host), 3 cards 17.3 GB (11.9 GB host).
inline double qfn_expert_host_est(int cards, const std::vector<double>& free_bytes) {
    const double routed = 48.0 * 512 * 5 * 506880.0;
    double res = 0;
    for (double f : free_bytes) res += std::max(0.0, f - 4.6 * GB - 10.44 * GB / std::max(1, cards));
    return std::max(0.0, routed - res);
}

// MemTotal / MemAvailable as the process sees them: a cgroup memory limit (systemd-run -p MemoryMax, docker --memory)
// caps both
inline double eff_total(const In& in) { return in.cg_limit > 0 ? std::min(in.total, in.cg_limit) : in.total; }
inline double eff_avail(const In& in) {
    return in.cg_limit > 0 ? std::min(in.avail, std::max(0.0, in.cg_limit - in.cg_anon)) : in.avail;
}

inline Out plan(const In& in) {
    Out o;
    const double total = eff_total(in), avail = eff_avail(in);
    o.reserve = in.reserve >= 0 ? in.reserve : default_reserve(total);
    o.room = total - o.reserve - in.base - in.fixed;
    // what the host has free now, kept above the runtime guard's margin (host_mem.h: the archive's own guard)
    // (MemAvailable already lacks what the process holds: the server base and the engine's buffers are not taken again)
    double mnow = in.margin_now >= 0 ? in.margin_now : std::max(8 * GB, 0.05 * in.total);
    if (in.cg_limit > 0) mnow = std::min(mnow, std::max(8 * GB, 0.05 * total));
    o.room_now = avail - mnow;
    // the machine's size decides; what is free right now only warns (review M1: a process alive at boot must not fix the
    // budgets low until a restart, the runtime guard holds the archive back while memory is short)
    const double r = o.room - in.experts, rn = o.room_now - in.experts;
    // pinned at load and never reduced. The size is an estimate: no refusal here, a warning. If the model can load it runs;
    // the guard of the pins (host_mem.h pin_fits, real sizes) stops the load only when the memory is really not there
    o.experts_short = in.experts > 0 && rn < 0;
    if (o.experts_short)
        o.warn.push_back(fmt("the host-backed expert groups need about %.1f GB of pinned host RAM, the host has %.1f GB free above "
                             "its margin (MemAvailable %.1f GB at start)", in.experts / GB, std::max(0.0, o.room_now) / GB, avail / GB) +
                         fmt(": the host may swap and the archive stays small; free host RAM, add RAM or cards (more cards keep "
                             "more experts in VRAM)"));
    else if (in.experts > 0 && o.room - in.experts < 0)
        o.warn.push_back(fmt("the host-backed experts (%.1f GB) alone take more than the room (%.1f GB): add host RAM or "
                             "cards; the host may swap", in.experts / GB, o.room / GB));
    const bool arch_on = in.arch_user != 0 && (in.arch_user > 0 || in.arch_def > 0);
    // 2. the n-gram table
    if (in.ngram > 0) {
        const double fit = std::min(r, rn);
        if (in.ngram_user == 1) {
            o.ngram_pinned = true;
            if (in.ngram > fit)
                o.warn.push_back(fmt("FLASH_NGRAM_PIN=1: the pinned n-gram table (%.1f GB) does not fit the room after "
                                     "the experts (%.1f GB): the host may swap", in.ngram / GB, fit / GB));
        } else if (in.ngram_user == 0) {
            o.ngram_pinned = false;
        } else {
            // room for a minimal archive next to the pinned table (8 GB), else the page cache serves the table
            const double floor_ = arch_on ? std::min(8 * GB, in.arch_def + in.pref_def) : 0.0;
            o.ngram_pinned = in.ngram + floor_ <= r;
            if (o.ngram_pinned && in.ngram > rn)
                o.warn.push_back(fmt("MemAvailable is short right now (%.1f GB free above the margin after the experts) for "
                                     "the pinned n-gram table (%.1f GB): pinned if it fits when it is read, else read from the "
                                     "page cache; the archive waits until memory is free", rn / GB, in.ngram / GB));
            if (!o.ngram_pinned && !in.ngram_can_map) {
                o.ngram_pinned = true;
                o.warn.push_back(fmt("the n-gram table (%.1f GB) does not fit the room after the experts (%.1f GB), and "
                                     "only the INT8 table can be read from the page cache: pinned anyway, the host may swap",
                                     in.ngram / GB, fit / GB));
            } else if (!o.ngram_pinned) {
                o.reduced = true;
                o.why.push_back(fmt("n-gram table not pinned: %.1f GB + a %.0f GB archive floor > %.1f GB of room after the "
                                    "experts; read from the page cache instead (staged rows, as FLASH_NGRAM_PIN=0)",
                                    in.ngram / GB, floor_ / GB, r / GB));
            }
        }
    }
    o.avail_short = o.experts_short || (o.ngram_pinned && in.ngram > rn);
    double left = r - (o.ngram_pinned ? in.ngram : 0.0);
    if (in.ngram > 0 && !o.ngram_pinned) {  // the mapped table's hot rows: half of what is left, at most 12 GB
        o.ngram_cache = floor01(std::min(12 * GB, std::max(0.0, left) / 2));
        left -= o.ngram_cache;
    }
    left = std::max(0.0, left);
    // 3. archive + pinned prefixes
    const bool au = in.arch_user >= 0, pu = in.pref_user >= 0;
    double a = au ? in.arch_user : in.arch_def, p = pu ? in.pref_user : in.pref_def;
    if (a <= 0 && in.prefix_needs_archive) p = 0;  // (Flash-Next: the prefixes live in the archive's pool)
    if (a + p > left) {
        const double user = (au ? a : 0.0) + (pu ? p : 0.0);
        bool cut = false;
        if (au && pu) {
        } else if (au) {
            const double np = floor01(std::min(p, std::max(0.0, left - a)));
            if (np < p) { o.why.push_back(fmt("pinned prefixes %.1f -> %.1f GB", p / GB, np / GB)); p = np; cut = true; }
        } else if (pu) {
            const double na = floor01(std::min(a, std::max(0.0, left - p)));
            if (na < a) { o.why.push_back(fmt("archive %.1f -> %.1f GB", a / GB, na / GB)); a = na; cut = true; }
        } else {
            const double s = std::max(0.0, left);
            const double np = floor01(p * s / (a + p)), na = floor01(s - np);
            o.why.push_back(fmt("archive %.1f -> %.1f GB", a / GB, na / GB));
            if (np < p) o.why.push_back(fmt("pinned prefixes %.1f -> %.1f GB", p / GB, np / GB));
            a = na;
            p = np;
            cut = true;
        }
        if (cut) {
            o.reduced = true;
            o.why.back() += fmt(" (%.1f GB left after the server", left / GB) + (in.experts > 0 ? ", the experts" : "");
            if (in.ngram > 0) o.why.back() += o.ngram_pinned ? " and the pinned n-gram table" : " and the n-gram page cache";
            o.why.back() += ")";
        }
        if (user > left) {
            std::string n;
            if (au) n += in.arch_env;
            if (pu) n += std::string(n.empty() ? "" : " + ") + in.pref_env;
            o.warn.push_back(fmt("%.1f GB set by ", user / GB) + n + fmt(" exceed the %.1f GB left: kept as set (the "
                                                                          "archive guard caps it at MemAvailable - margin as it fills)", left / GB));
        }
    }
    o.archive = a;
    // (a warning only: the defaults stay, a start is never refused for this; an explicit value is the user's choice)
    if (!au && in.arch_def > 0 && a < 2 * GB)
        o.warn.push_back(fmt("the KV archive is only %.1f GB: conversations that leave VRAM mostly cannot be restored and get "
                             "recomputed (slow re-prefills). QE_MEM_MARGIN_GB and QE_HOST_RESERVE_GB change what it gets", a / GB));
    o.prefix = a > 0 || !in.prefix_needs_archive ? p : 0;
    return o;
}

// the plan, after waiting up to wait_s for MemAvailable to come back when what is pinned at load does not fit it yet (a model
// switch: the previous server is still exiting); read_avail() reads MemAvailable again. *waited: the seconds waited
// busy(): a previous holder is still alive (gpu_others.h big_holder): flat seconds do not end the wait (review N-L2)
template <class F>
inline Out plan_wait(In& in, F read_avail, double wait_s, double* waited, bool (*busy)() = nullptr) {
    Out o = plan(in);
    // waits only while MemAvailable rises (a process exiting gives memory back): a host that is just short starts at once
    // (reviews: 27B L4, Flash-Next M4)
    double t = 0;
    int flat = 0;
    while (o.avail_short && t < wait_s) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
        t += 1;
        const double a = read_avail();
        const bool rising = a > in.avail + 0.5 * GB;
        in.avail = a;
        o = plan(in);
        flat = rising || (busy && busy()) ? 0 : flat + 1;
        if (flat >= 3) break;  // (three flat seconds with no previous holder alive: nothing is exiting, review N-L2)
    }
    if (waited) *waited = t;
    return o;
}

// the one start-up line
inline std::string line(const In& in, const Out& o) {
    std::string s = std::string(in.tag) + fmt("host RAM plan: MemTotal %.1f GB, MemAvailable %.1f GB", in.total / GB, in.avail / GB);
    if (in.cg_limit > 0) s += fmt(", cgroup limit %.1f GB (%.1f GB anonymous in it)", in.cg_limit / GB, in.cg_anon / GB);
    s += fmt(", reserve %.1f GB (OS + desktop)", o.reserve / GB);
    s += fmt(", server base %.1f GB", in.base / GB);
    if (in.fixed > 0) s += fmt(", engine buffers %.1f GB", in.fixed / GB);
    if (in.qfn) {
        s += fmt("; experts in host memory %.1f GB (estimate)", in.experts / GB);
        if (in.ngram > 0)
            s += o.ngram_pinned ? fmt(", n-gram table %.1f GB pinned", in.ngram / GB)
                                : fmt(", n-gram table %.1f GB from the page cache (%.1f GB kept for it)", in.ngram / GB, o.ngram_cache / GB);
    }
    s += fmt("; KV archive %.1f GB", o.archive / GB) + (in.arch_user >= 0 ? std::string(" (") + in.arch_env + ")" : "");
    s += fmt(" + pinned prefixes %.1f GB", o.prefix / GB) + (in.pref_user >= 0 ? std::string(" (") + in.pref_env + ")" : "");
    s += o.why.empty() ? std::string("; reduced: nothing") : "; reduced: ";
    for (size_t i = 0; i < o.why.size(); i++) s += (i ? "; " : "") + o.why[i];
    return s;
}

inline std::string json(const In& in_, const Out& o_, double experts_actual = -1) {
    // (non-finite values, e.g. an env value "inf", would make the JSON invalid: shown as -1)
    auto fin = [](double v) { return std::isfinite(v) ? v : -GB; };
    In in = in_;
    Out o = o_;
    for (double* v : {&in.total, &in.avail, &in.base, &in.fixed, &in.experts, &in.ngram, &in.cg_limit}) *v = fin(*v);
    for (double* v : {&o.reserve, &o.archive, &o.prefix, &o.ngram_cache}) *v = fin(*v);
    experts_actual = std::isfinite(experts_actual) ? experts_actual : -1;
    std::string s = fmt("{\"mem_total_gb\": %.2f, \"mem_available_gb\": %.2f, \"reserve_gb\": %.2f", in.total / GB, in.avail / GB, o.reserve / GB);
    if (in.cg_limit > 0) s += fmt(", \"cgroup_limit_gb\": %.2f", in.cg_limit / GB);
    s += fmt(", \"server_base_gb\": %.2f, \"engine_buffers_gb\": %.2f", in.base / GB, in.fixed / GB);
    if (in.qfn) {
        s += fmt(", \"experts_gb_estimate\": %.2f", in.experts / GB);
        if (experts_actual >= 0) s += fmt(", \"experts_gb\": %.2f", experts_actual / GB);
        s += fmt(", \"ngram_gb\": %.2f, \"ngram_page_cache_gb\": %.2f, \"ngram_pinned\": ", in.ngram / GB, o.ngram_cache / GB);
        s += o.ngram_pinned ? "true" : "false";
    }
    s += fmt(", \"archive_gb\": %.2f, \"prefix_gb\": %.2f, \"reduced\": ", o.archive / GB, o.prefix / GB);
    s += o.reduced ? "true" : "false";
    auto arr = [](const std::vector<std::string>& v) {
        std::string a = "[";
        for (size_t i = 0; i < v.size(); i++) {
            std::string x = v[i];
            for (char& c : x) if (c == '"' || c == '\\' || (unsigned char)c < 0x20) c = '\'';
            a += (i ? ", \"" : "\"") + x + "\"";
        }
        return a + "]";
    };
    s += ", \"why\": " + arr(o.why) + ", \"warnings\": " + arr(o.warn) + "}";
    return s;
}

}  // namespace qe_hostbudget
