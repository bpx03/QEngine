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
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
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
    static const char* v = getenv("QE_MEMAVAIL_GB") && *getenv("QE_MEMAVAIL_GB") ? getenv("QE_MEMAVAIL_GB") : nullptr;
    return v;
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
        }
        last = v ? v : (ov ? 1 : (size_t)1 << 62);  // no /proc: no limit
        t = now;
    }
    if (ov) {  // the file is the host without the archive: its own bytes come off
        const long long v = (long long)last - charged().load();
        return v > 0 ? (size_t)v : 0;
    }
    return last;
}

inline size_t margin() {
    static const size_t m = getenv("QE_MEM_MARGIN_GB") ? (size_t)(atof(getenv("QE_MEM_MARGIN_GB")) * 1e9)
                                                       : std::max<size_t>((size_t)8e9, total() / 20);
    return m;
}

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
