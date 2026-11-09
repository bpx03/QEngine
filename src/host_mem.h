#pragma once
// Host memory guard of the KV archives (both engines): the archive never takes the host below
// MemAvailable - margin, so a large archive budget (QE_KV_ARCHIVE_GB / FLASH_KV_ARCHIVE_GB = 128 on a
// 251 GB host next to the engine's own ~64 GB, the desktop and docker) shrinks with a log line instead
// of the kernel's OOM killer ending the server.
//
// MemAvailable (/proc/meminfo) is free memory + reclaimable page cache / slab. Archive memory, pinned
// (cudaHostRegister'ed slabs) or pageable, is anonymous memory and already counted as used (a 4 GB
// archive slab lowers MemAvailable by 4 GB when it is faulted in, before it is registered); a pinned slab
// with free ranges in it is memory the archive owns but does not use, so the archive may grow by
//   (its free pinned bytes) + max(0, MemAvailable - margin).
// margin = max(8 GB, 5 % of MemTotal), QE_MEM_MARGIN_GB overrides it.
// QE_MEMAVAIL_GB (tests): pretend the host had this much MemAvailable at startup; the archive pools'
// own bytes (charged()) come off it as they are allocated. QE_MEMAVAIL_GB=@<file>: the same with the startup
// value read from the file (GB, re-read every 100 ms): another process growing, for the pressure path.
#include "env.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <unistd.h>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

namespace qe_hostmem {

// bytes the archive pools hold (pinned slabs + pageable blobs): the QE_MEMAVAIL_GB model's usage
inline std::atomic<long long>& charged() {
    static std::atomic<long long> c{0};
    return c;
}

inline size_t meminfo_kb(const char* key) {
    FILE* f = fopen("/proc/meminfo", "r");
    if (!f) return 0;
    char line[256];
    size_t v = 0;
    const size_t kl = strlen(key);
    while (fgets(line, sizeof line, f))
        if (!strncmp(line, key, kl) && line[kl] == ':') {
            v = strtoull(line + kl + 1, nullptr, 10);
            break;
        }
    fclose(f);
    return v;
}

inline size_t total() {
    static const size_t t = meminfo_kb("MemTotal") << 10;
    return t;
}

inline const char* override_env() {
    static const char* v = env_str("QE_MEMAVAIL_GB");
    return v;
}

// cgroup memory limit of this process (systemd-run -p MemoryMax, docker --memory): on every level from its cgroup up to the
// root that has a limit, room = limit - (usage - reclaimable page cache); the level with the least room binds (review M3: a
// parent slice shared with siblings can bind before a larger own limit). Usage counts anonymous, shmem (cudaHostAlloc'd
// memory, the FLASH_KV=ram pool: measured as shmem in memory.stat) and kernel memory; only file pages that are not shmem are
// reclaimable. limit = the binding level's limit, anon = limit - room (what is not available in it). {0, 0}: no limit.
// v2 or v1 is chosen by the unified hierarchy's presence (review M2: docker's private cgroup namespace shows "0::/", the
// container's own cgroup at /sys/fs/cgroup). The directories are found once; the files are read on each call. Read by the
// host-RAM plan at boot (host_budget.h) and by available() below (the runtime guard sees the limit as the host)
struct CgroupMem {
    double limit = 0, anon = 0;
};
namespace detail {
struct CgDirs {
    bool v2 = false;
    std::vector<std::string> dirs;  // own cgroup first, then the parents
};
inline const CgDirs& cg_dirs() {
    static const CgDirs d = [] {
        CgDirs r;
        std::string v2path, v1path;
        bool has0 = false;
        if (FILE* f = fopen("/proc/self/cgroup", "r")) {
            char line[1024];
            while (fgets(line, sizeof line, f)) {
                if (!strncmp(line, "0::", 3)) { has0 = true; v2path = line + 3; }
                const char* c = strchr(line, ':');
                if (c && !strncmp(c + 1, "memory:", 7)) v1path = c + 8;
            }
            fclose(f);
        }
        auto trim = [](std::string& p) { while (!p.empty() && (p.back() == '\n' || p.back() == '/')) p.pop_back(); };
        trim(v2path);
        trim(v1path);
        r.v2 = has0 && v1path.empty() && access("/sys/fs/cgroup/cgroup.controllers", R_OK) == 0;
        const std::string base = r.v2 ? "/sys/fs/cgroup" : "/sys/fs/cgroup/memory", path = r.v2 ? v2path : v1path;
        for (std::string p = path;; p = p.rfind('/') == std::string::npos ? std::string() : p.substr(0, p.rfind('/'))) {
            r.dirs.push_back(base + p);
            if (p.empty()) break;
        }
        return r;
    }();
    return d;
}
inline bool read_num(const std::string& file, double* v) {
    FILE* f = fopen(file.c_str(), "r");
    if (!f) return false;
    char b[64] = "";
    const bool ok = fscanf(f, "%63s", b) == 1 && strcmp(b, "max");
    fclose(f);
    if (ok) *v = atof(b);
    return ok;
}
}  // namespace detail
inline CgroupMem cgroup_mem() {
    CgroupMem r;
    const detail::CgDirs& d = detail::cg_dirs();
    double best = -1;
    for (const std::string& dir : d.dirs) {
        double lim = 0, use = 0;
        if (!detail::read_num(dir + (d.v2 ? "/memory.max" : "/memory.limit_in_bytes"), &lim) || lim <= 0 || lim >= 1e18) continue;
        if (!detail::read_num(dir + (d.v2 ? "/memory.current" : "/memory.usage_in_bytes"), &use)) continue;
        double file = 0, shmem = 0;
        if (FILE* s = fopen((dir + "/memory.stat").c_str(), "r")) {
            char k[64];
            double x;
            while (fscanf(s, "%63s %lf", k, &x) == 2) {
                if (!strcmp(k, d.v2 ? "file" : "cache")) file = x;
                else if (!strcmp(k, "shmem")) shmem = x;
            }
            fclose(s);
        }
        const double room = std::max(0.0, lim - std::max(0.0, use - std::max(0.0, file - shmem)));
        if (best < 0 || room < best) {
            best = room;
            r.limit = lim;
            r.anon = lim - room;
        }
    }
    return r;
}

// MemAvailable in bytes (cached for 100 ms unless `fresh`)
inline size_t available(bool fresh = false) {
    const char* ov = override_env();
    if (ov && ov[0] != '@') {
        const long long v = (long long)(atof(ov) * 1e9) - charged().load();
        return v > 0 ? (size_t)v : 0;
    }
    static std::mutex mu;
    static size_t last = 0;
    static std::chrono::steady_clock::time_point t;
    std::lock_guard<std::mutex> l(mu);
    const auto now = std::chrono::steady_clock::now();
    if (fresh || last == 0 || now - t > std::chrono::milliseconds(100)) {
        size_t v = 0;
        if (ov) {
            if (FILE* f = fopen(ov + 1, "r")) {
                double gb = 0;
                if (fscanf(f, "%lf", &gb) == 1) v = (size_t)(std::max(0.0, gb) * 1e9);
                fclose(f);
            }
        } else {
            v = meminfo_kb("MemAvailable") << 10;
            if (!v) v = (size_t)1 << 62;  // no /proc: no limit
            // a cgroup memory limit (docker --memory, systemd-run -p MemoryMax) is the host for this process: past it the
            // cgroup reclaims, swaps or OOM-kills inside it while the host's MemAvailable still looks fine. A full cgroup is
            // 1 byte, never 0 (0 means "not read yet" here; review M1: it used to read as no limit)
            const CgroupMem c = cgroup_mem();
            if (c.limit > 0) v = std::max<size_t>(1, std::min(v, (size_t)std::max(0.0, c.limit - c.anon)));
        }
        last = v ? v : 1;
        t = now;
    }
    if (ov) {  // the file is the host without the archive: its own bytes come off
        const long long v = (long long)last - charged().load();
        return v > 0 ? (size_t)v : 0;
    }
    return last;
}

// Load-time guard of the big pins (Flash-Next pin_host / pin_host_file). A pin in flight has already lowered MemAvailable by
// what it has touched: only its untouched part is still to come, so only that part counts against a pin starting next to it.
inline std::atomic<size_t>& pin_untouched() {
    static std::atomic<size_t> v{0};
    return v;
}
// one pin of cap bytes: charged in full at the start, its touched part comes off as it is filled
class PinCharge {
public:
    explicit PinCharge(size_t cap) : left_(cap) { before_ = pin_untouched().fetch_add(cap); }
    ~PinCharge() { pin_untouched().fetch_sub(left_.exchange(0)); }
    PinCharge(const PinCharge&) = delete;
    PinCharge& operator=(const PinCharge&) = delete;
    size_t others() const { return before_; }  // what the other pins in flight had not touched yet when this one started
    void touched(size_t n) {
        left_.fetch_sub(n);
        pin_untouched().fetch_sub(n);
    }
private:
    std::atomic<size_t> left_;
    size_t before_ = 0;
};
// a pin fits when MemAvailable holds it, the rest of the other pins in flight, and 2 GB
inline bool pin_fits(size_t avail, size_t cap, size_t others) { return avail >= cap + others + ((size_t)2 << 30); }

inline size_t margin() {
    static const size_t m = [] {
        const double gb = env_double("QE_MEM_MARGIN_GB", -1.0);  // empty / unparsable: the default, not 0 GB
        if (gb >= 0) return (size_t)(gb * 1e9);
        const CgroupMem c = override_env() ? CgroupMem{} : cgroup_mem();  // (a cgroup limit: 5 % of it, as the plan's margin)
        const double t = c.limit > 0 ? std::min((double)total(), c.limit) : (double)total();
        return std::max<size_t>((size_t)8e9, (size_t)(t / 20));
    }();
    return m;
}

// what a pin may take now: MemAvailable, capped by this process's cgroup limit (past it the cgroup's OOM killer ends the load)
inline size_t pin_available() { return available(true); }  // (available() includes the cgroup limit)

// bytes the archive may still take from the host (0 at or below the margin)
inline size_t headroom(bool fresh = false) {
    const size_t a = available(fresh), m = margin();
    return a > m ? a - m : 0;
}

// bytes the host is below the margin (the archive should give back)
inline size_t deficit(bool fresh = false) {
    const size_t a = available(fresh), m = margin();
    return a < m ? m - a : 0;
}

// "pid name RSS GB, ..." of the n processes with the largest resident memory (who holds the host's RAM: logged when
// pinning fails at startup)
inline std::string top_rss(int n = 5) {
    std::vector<std::pair<size_t, std::string>> v;
    if (DIR* d = opendir("/proc")) {
        while (dirent* e = readdir(d)) {
            if (e->d_name[0] < '0' || e->d_name[0] > '9') continue;
            char path[64];
            snprintf(path, sizeof path, "/proc/%s/status", e->d_name);
            FILE* f = fopen(path, "r");
            if (!f) continue;
            char line[256], name[64] = "?";
            size_t kb = 0;
            while (fgets(line, sizeof line, f)) {
                if (!strncmp(line, "Name:", 5)) sscanf(line + 5, "%63s", name);
                else if (!strncmp(line, "VmRSS:", 6)) kb = strtoull(line + 6, nullptr, 10);
            }
            fclose(f);
            if (kb) v.push_back({kb, std::string(e->d_name) + " " + name});
        }
        closedir(d);
    }
    std::sort(v.begin(), v.end(), [](auto& a, auto& b) { return a.first > b.first; });
    std::string s;
    for (int i = 0; i < n && i < (int)v.size(); i++) {
        char b[160];
        snprintf(b, sizeof b, "%s%s %.1f GB", i ? ", " : "", v[i].second.c_str(), v[i].first / 1048576.0);
        s += b;
    }
    return s;
}

// rate-limited log decision for the "budget capped" line: true when the cap changed by >= 2 GB or 60 s passed
struct CapLog {
    double last_gb = -1;
    std::chrono::steady_clock::time_point t;
    bool due(double cap_gb) {
        const auto now = std::chrono::steady_clock::now();
        if (last_gb >= 0 && std::abs(cap_gb - last_gb) < 2.0 && now - t < std::chrono::seconds(60)) return false;
        last_gb = cap_gb;
        t = now;
        return true;
    }
    void reset() { last_gb = -1; }
};

}  // namespace qe_hostmem
