#pragma once
// VRAM other processes took on a card (VRAM-pressure checks, both engines). Only a shortage other processes caused
// makes an engine give memory back (resident experts to host, a smaller KV pool); its own graphs / workspaces have
// their own out-of-memory paths, and a giving-back caused by them would never be taken back.
//
// others(dev, own): (total - free) - this process's usage from NVML (libnvidia-ml via dlopen, no link dependency). Our
// entry is found by pid; in a container (a PID namespace: NVML lists host pids) by usage: the listed process whose
// usage is closest above `own` (the device bytes the engine tracks; untracked context / graphs come on top, <= 4 GB).
// -1: NVML gives nothing usable.
// Watch::growth: what the others took since the baseline (the smallest value seen: a process that exits lowers it);
// without NVML from free-memory changes not explained by the engine's own tracked bytes, minus `own_slack` (its
// untracked growth: graphs, workspaces). Logged once which way it judges.
#include <cuda_runtime.h>
#include <dlfcn.h>
#include <unistd.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>

#include "gpu_guard.h"

namespace qe_gpu_others {
struct ProcInfo {  // nvmlProcessInfo_v2_t
    unsigned int pid;
    unsigned long long usedGpuMemory;
    unsigned int gpuInstanceId, computeInstanceId;
};
struct Nvml {
    void* h = nullptr;
    int (*init)() = nullptr;
    int (*by_bus)(const char*, void**) = nullptr;
    int (*procs)(void*, unsigned int*, ProcInfo*) = nullptr;
    bool ok = false;
};
inline Nvml& nvml() {
    static Nvml n;
    static std::once_flag once;
    std::call_once(once, [] {
        n.h = dlopen("libnvidia-ml.so.1", RTLD_NOW | RTLD_LOCAL);
        if (!n.h) return;
        n.init = (int (*)())dlsym(n.h, "nvmlInit_v2");
        n.by_bus = (int (*)(const char*, void**))dlsym(n.h, "nvmlDeviceGetHandleByPciBusId_v2");
        n.procs = (int (*)(void*, unsigned int*, ProcInfo*))dlsym(n.h, "nvmlDeviceGetComputeRunningProcesses_v3");
        n.ok = n.init && n.by_bus && n.procs && n.init() == 0;
    });
    return n;
}
inline bool mem_info(int dev, size_t* f, size_t* t) {
    int cur = 0;
    cudaGetDevice(&cur);
    cudaSetDevice(dev);
    const bool ok = cudaMemGetInfo(f, t) == cudaSuccess;
    cudaSetDevice(cur);
    if (!ok) (void)cudaGetLastError();
    return ok;
}
// bytes of device `dev` (CUDA ordinal) held by other processes, or -1; *by_usage: our entry found by usage (container)
inline long long others(int dev, long long own, bool* by_usage = nullptr) {
    if (by_usage) *by_usage = false;
    Nvml& n = nvml();
    // (tests: QE_GPU_OTHERS_TEST=nopid finds our entry by usage, as in a container; =nonvml: as without NVML)
    static const char* test = getenv("QE_GPU_OTHERS_TEST");
    if (!n.ok || (test && !strcmp(test, "nonvml"))) return -1;
    char bus[32] = "";
    if (cudaDeviceGetPCIBusId(bus, sizeof bus, dev) != cudaSuccess) { (void)cudaGetLastError(); return -1; }
    void* h = nullptr;
    if (n.by_bus(bus, &h) != 0) return -1;
    ProcInfo pi[64];
    unsigned int cnt = 64;
    if (n.procs(h, &cnt, pi) != 0) return -1;
    long long mine = -1;
    for (unsigned i = 0; i < cnt; i++)
        if (pi[i].pid == (unsigned)getpid() && pi[i].usedGpuMemory < (1ULL << 50) && !(test && !strcmp(test, "nopid")))
            mine = (long long)pi[i].usedGpuMemory;
    if (mine < 0 && own > 0) {  // a PID namespace: the one entry whose usage lies above what we hold (within 4 GB)
        long long best = -1;
        int n_cand = 0;
        for (unsigned i = 0; i < cnt; i++) {
            const long long u = (long long)pi[i].usedGpuMemory;
            if (u >= (1LL << 50) || u < own || u > own + (4LL << 30)) continue;
            n_cand++;
            if (best < 0 || u < best) best = u;
        }
        if (n_cand > 1) return -1;  // (two candidates: ambiguous, the free-memory way instead of guessing, review N-L4)
        if (best >= 0 && by_usage) *by_usage = true;
        mine = best;
    }
    if (mine < 0) return -1;
    size_t f = 0, t = 0;
    if (!mem_info(dev, &f, &t)) return -1;
    const long long o = (long long)(t - f) - mine;
    return o > 0 ? o : 0;
}
// a process that runs a QEngine engine (its memory maps show libqflash / libqengine; another user's or a container's
// process is not readable: false)
inline bool qengine_pid(unsigned pid) {
    char path[64];
    snprintf(path, sizeof path, "/proc/%u/maps", pid);
    FILE* f = fopen(path, "r");
    if (!f) return false;
    char line[1024];
    bool hit = false;
    while (!hit && fgets(line, sizeof line, f)) hit = strstr(line, "libqflash") || strstr(line, "libqengine");
    fclose(f);
    return hit;
}
// at start: the previous QEngine server on these cards (the one that held their GPU locks before this process,
// gpu_guard.h) is still tearing down and holds >= min_bytes on one of our cards: the start-up waits keep waiting for
// it instead of ending after three flat seconds (review N-L2). A live server elsewhere, a sibling on other cards, the
// desktop or another application never counts (review R6-L5). A previous holder this process cannot see (another PID
// namespace, e.g. a container restart): any other process holding >= min_bytes on every one of our cards counts (NVML
// lists host pids there). Unknown (no NVML): false
inline bool big_holder(const int* devs, int n, long long min_bytes = 1LL << 30) {
    bool hidden = false;
    std::vector<unsigned> alive;
    for (auto& h : qe_gpu_guard::prev_holders()) {
        if (!h.visible) hidden = true;
        else if (qengine_pid(h.pid)) alive.push_back(h.pid);
    }
    if (alive.empty() && !hidden) return false;
    Nvml& nv = nvml();
    if (!nv.ok) return false;
    int cards_with_other = 0;
    for (int i = 0; i < n; i++) {
        char bus[32] = "";
        if (cudaDeviceGetPCIBusId(bus, sizeof bus, devs[i]) != cudaSuccess) { (void)cudaGetLastError(); continue; }
        void* h = nullptr;
        if (nv.by_bus(bus, &h) != 0) continue;
        ProcInfo pi[64];
        unsigned int cnt = 64;
        if (nv.procs(h, &cnt, pi) != 0) continue;
        bool other = false;
        for (unsigned k = 0; k < cnt; k++) {
            if (pi[k].pid == (unsigned)getpid() || pi[k].usedGpuMemory >= (1ULL << 50) || (long long)pi[k].usedGpuMemory < min_bytes)
                continue;
            if (std::find(alive.begin(), alive.end(), pi[k].pid) != alive.end()) return true;  // our previous holder
            other = true;
        }
        cards_with_other += other;
    }
    return hidden && n > 0 && cards_with_other == n;
}
struct Watch {
    // NVML: the reference is the others' usage at the first sample (the start plan was sized next to it); it is lowered
    // only after they stayed lower for 60 s (a process that left), to the most they held in that time: a neighbour that
    // dips and comes back is no pressure (review N-M3)
    long long base = -1, below_max = -1;
    double below_since = -1;
    long long f0 = -1, t0 = 0;  // fallback: free and own bytes at the reference point
    double fb_t0 = -1;          // fallback: the first sample (s), and whether the reference was retaken after warm-up
    bool fb_rebased = false;
    int mode = 0;             // 0 not yet, 1 NVML by pid, 2 NVML by usage, 3 free-memory deltas
    unsigned logged = 0;      // the modes logged so far (one line each)
    double t_last = -1e18;    // the last sample (seconds)
    long long last = 0;       // its result
    // bytes the others took since the reference (>= 0); t: now (s); free_now: this card's free memory; own: device bytes
    // the engine tracks; own_slack: its untracked growth allowance (fallback only); tag / dev for the log line
    long long growth(double t, int dev, size_t free_now, long long own, long long own_slack, const char* tag) {
        bool by_usage = false;
        const long long o = others(dev, own, &by_usage);
        const int m = o >= 0 ? (by_usage ? 2 : 1) : 3;
        if (m != mode && !(logged & (1u << m))) {
            fprintf(stderr, "%s GPU%d VRAM pressure is judged by %s\n", tag, dev,
                    m == 1 ? "other processes' usage (NVML)" : m == 2 ? "other processes' usage (NVML; this process found by its usage: a container)"
                           : "free-memory changes beyond the engine's own allocations (NVML per-process usage not available)");
            logged |= 1u << m;
        }
        mode = m;
        if (o >= 0) {
            if (base < 0) base = o;
            if (o < base) {
                if (below_since < 0) { below_since = t; below_max = o; }
                below_max = std::max(below_max, o);
                if (t - below_since >= 60) { base = below_max; below_since = -1; }
            } else {
                below_since = -1;
            }
            return std::max(0LL, o - base);
        }
        // (the tracked quantity is free + own: a give-back lowers own and raises free by the same, review N-M2). The
        // reference is taken again once after the first 60 s (warm-up: lazily captured graphs and workspaces are the
        // engine's own untracked growth, not pressure; review R4-L6); a neighbour that grew in that minute counts as
        // part of the start, as the NVML reference does
        if (fb_t0 < 0) fb_t0 = t;
        const bool rebase = !fb_rebased && t - fb_t0 >= 60;
        if (rebase) fb_rebased = true;
        if (f0 < 0 || rebase || (long long)free_now + own > f0 + t0) {
            f0 = (long long)free_now;
            t0 = own;
        }
        const long long d = (f0 + t0) - ((long long)free_now + own) - own_slack;
        return d > 0 ? d : 0;
    }
    // growth(), but NVML (milliseconds per call) only when `needed` (free below the low mark) or every `every_s`;
    // else the last result
    size_t f_last = 0;  // free memory at the last sample (growth_lazy)
    template <class Now>
    long long growth_lazy(bool needed, double every_s, Now now, int dev, size_t free_now, long long own, long long own_slack, const char* tag) {
        const double t = now();
        // (on demand only after a new drop of the free memory since the last sample: a card that stays below the low
        // mark at a steady size samples at the cadence, not every tick, review R5-L6)
        const bool drop = f_last == 0 || free_now + (64u << 20) < f_last;
        if (!(needed && drop) && t - t_last < every_s) return last;
        t_last = t;
        f_last = free_now ? free_now : 1;
        last = growth(t, dev, free_now, own, own_slack, tag);
        return last;
    }
};
}  // namespace qe_gpu_others
