#pragma once
// Start-up cache of the communication measurements (club-3090 #1926): the link probe's bandwidth / co-run tax per GPU
// pair (comm_select_path, ~0.7-1.2 s per boot) and the decode collectives' calibration (real decode steps on P2P stores
// vs host memory, both engines). Reused while everything that can change the answer is unchanged: the key holds the
// engine build and this file's format version, the engine / model / mode / card count, per card (in device order) its
// UUID, PCI address, maximum link speed and negotiated width, NUMA node, active NVLink links, the peer-access matrix,
// the kernel driver and CUDA versions, and the env knobs of the collectives. Any mismatch, a missing or corrupt file:
// measured again and rewritten. Files: $QE_DISK_CACHE or ~/.cache/qengine (Docker: the /root/.cache/qengine volume;
// without one the container just measures each start), comm_<what>_<key hash>.txt. Written to a temporary name and
// renamed (atomic): two servers starting at once both measure and the last rename wins, a reader never sees half a
// file. QE_COMM_CACHE=0: no reads or writes; QE_COMM_CACHE=refresh (or QE_DECODE_COMM=remeasure for the decode part):
// measure and rewrite. Entries expire after QE_COMM_CACHE_DAYS (30). Nothing is written when another compute process
// runs on one of the cards while measuring (NVML), or when the decode rounds disagree: such a boot measures again next
// time instead of fixing a noisy answer.
#include "../env.h"
#include "../gpu_guard.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <map>
#include <memory>
#include <atomic>
#include <thread>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>
#include <dlfcn.h>
#include <fcntl.h>
#include <limits.h>
#include <sys/stat.h>
#include <unistd.h>

extern "C" const char* qe_internal_build_id();  // src/build_id.cpp (generated at build time)

constexpr int kCommCacheVer = 1;  // bump when the cached quantities or their meaning change

inline std::string comm_sysfs_read(const std::string& p) {
    std::ifstream f(p);
    std::string s;
    std::getline(f, s);
    return s;
}
inline std::string comm_cache_dir() {
    if (const char* d = env_str("QE_DISK_CACHE"); d && *d) return d;
    const char* home = env_str("HOME");
    return home && *home ? std::string(home) + "/.cache/qengine" : std::string();
}
// 0 off, 1 read + write, 2 write only (refresh)
inline int comm_cache_mode() {
    static const int m = [] {
        const char* e = env_str("QE_COMM_CACHE");
        if (!e || !*e) return 1;
        std::string v(e);
        for (auto& ch : v) ch = (char)tolower(ch);
        if (v == "refresh") return 2;
        if (v == "1" || v == "on" || v == "true" || v == "yes") return 1;
        if (v == "0" || v == "off" || v == "false" || v == "no") return 0;
        fprintf(stderr, "[comm] QE_COMM_CACHE=%s: expected 0, 1 or refresh; using 1\n", e);
        return 1;
    }();
    return m;
}

// NVML through dlopen (no link dependency; absent: the fields read "?" and the busy check is skipped)
struct CommNvml {
    void* lib = nullptr;
    int (*init)() = nullptr;
    int (*by_bus)(const char*, void**) = nullptr;
    int (*procs)(void*, unsigned*, void*) = nullptr;
    int (*gprocs)(void*, unsigned*, void*) = nullptr;
    int (*plimit)(void*, unsigned*) = nullptr;
    int (*app_clock)(void*, int, unsigned*) = nullptr;
    int (*max_clock)(void*, int, unsigned*) = nullptr;
    int (*bar1)(void*, unsigned long long*) = nullptr;  // nvmlBAR1Memory_t: total, free, used
    int (*max_gen)(void*, unsigned*) = nullptr;
    int (*max_width)(void*, unsigned*) = nullptr;
    int (*gpc_off)(void*, int*) = nullptr;
    int (*putil)(void*, void*, unsigned*, unsigned long long) = nullptr;  // per-process utilization samples
    int (*nvl_state)(void*, unsigned, int*) = nullptr;  // NVLink link active (nvmlEnableState_t)
    int (*nvl_remote)(void*, unsigned, void*) = nullptr;  // the PCI address at the link's other end (nvmlPciInfo_t)
    bool ok = false;
    static CommNvml& get() {
        static CommNvml n = [] {
            CommNvml x;
            x.lib = dlopen("libnvidia-ml.so.1", RTLD_NOW | RTLD_LOCAL);
            if (!x.lib) return x;
            auto f = [&](const char* s) { return dlsym(x.lib, s); };
            x.init = (int (*)())f("nvmlInit_v2");
            x.by_bus = (int (*)(const char*, void**))f("nvmlDeviceGetHandleByPciBusId_v2");
            // the _v2 lists: nvmlProcessInfo_v2_t {pid, usedGpuMemory, gpuInstanceId, computeInstanceId}, 24 bytes
            x.procs = (int (*)(void*, unsigned*, void*))f("nvmlDeviceGetComputeRunningProcesses_v2");
            x.gprocs = (int (*)(void*, unsigned*, void*))f("nvmlDeviceGetGraphicsRunningProcesses_v2");
            x.plimit = (int (*)(void*, unsigned*))f("nvmlDeviceGetPowerManagementLimit");
            x.app_clock = (int (*)(void*, int, unsigned*))f("nvmlDeviceGetApplicationsClock");
            x.max_clock = (int (*)(void*, int, unsigned*))f("nvmlDeviceGetMaxClockInfo");
            x.bar1 = (int (*)(void*, unsigned long long*))f("nvmlDeviceGetBAR1MemoryInfo");
            x.max_gen = (int (*)(void*, unsigned*))f("nvmlDeviceGetMaxPcieLinkGeneration");
            x.max_width = (int (*)(void*, unsigned*))f("nvmlDeviceGetMaxPcieLinkWidth");
            x.gpc_off = (int (*)(void*, int*))f("nvmlDeviceGetGpcClkVfOffset");
            x.putil = (int (*)(void*, void*, unsigned*, unsigned long long))f("nvmlDeviceGetProcessUtilization");
            x.nvl_state = (int (*)(void*, unsigned, int*))f("nvmlDeviceGetNvLinkState");
            x.nvl_remote = (int (*)(void*, unsigned, void*))f("nvmlDeviceGetNvLinkRemotePciInfo_v2");
            x.ok = x.init && x.by_bus && x.init() == 0;
            return x;
        }();
        return n;
    }
    void* dev(const char* bus) const {
        void* h = nullptr;
        return ok && by_bus(bus, &h) == 0 ? h : nullptr;
    }
};
inline std::string comm_bus_id(int device) {
    char bus[32] = "";
    cudaDeviceGetPCIBusId(bus, sizeof bus, device);
    for (char* q = bus; *q; q++) *q = (char)tolower(*q);
    return bus;
}
// Busy-card check (review rounds 1-4): no measurement is cached when something else may have used the cards. Fail
// closed: no NVML, an NVML error or a list it cannot return means "busy". Per card snapshots of the compute processes
// (PID + used memory) and the graphics processes (NVML _v2 records {pid, usedGpuMemory, gpuInstanceId,
// computeInstanceId}, 24 bytes); boot snapshot at comm_select_path.
struct CommProcSnap {
    bool ok = false;
    std::vector<std::vector<unsigned>> pids;                                     // per card: compute + graphics PIDs
    std::vector<std::vector<std::pair<unsigned, unsigned long long>>> cmem;     // per card: compute PIDs, used memory
    std::vector<std::vector<unsigned>> gpids;                                   // per card: graphics PIDs
};
inline CommProcSnap comm_proc_snap(const int* devices, int ng) {
    CommProcSnap r;
    const CommNvml& n = CommNvml::get();
    if (!n.ok || !n.procs || !n.gprocs) return r;
    struct PI { unsigned pid; unsigned long long mem; unsigned gi, ci; };
    static_assert(sizeof(PI) == 24, "nvmlProcessInfo_v2_t");
    for (int i = 0; i < ng; i++) {
        void* h = n.dev(comm_bus_id(devices[i]).c_str());
        if (!h) return r;
        std::vector<unsigned> v, g;
        std::vector<std::pair<unsigned, unsigned long long>> cm;
        for (int kind = 0; kind < 2; kind++) {
            auto fn = kind ? n.gprocs : n.procs;
            unsigned cnt = 0;
            int e = fn(h, &cnt, nullptr);  // the count (NVML_ERROR_INSUFFICIENT_SIZE = 7 when > 0)
            if (e != 0 && e != 7) return r;
            std::vector<PI> b(cnt + 16);
            cnt = (unsigned)b.size();
            if (fn(h, &cnt, b.data()) != 0) return r;  // (grew past the room meanwhile: an error, i.e. busy)
            for (unsigned k = 0; k < cnt; k++) {
                v.push_back(b[k].pid);
                if (kind) g.push_back(b[k].pid);
                else cm.push_back({b[k].pid, b[k].mem});
            }
        }
        r.pids.push_back(std::move(v));
        r.cmem.push_back(std::move(cm));
        r.gpids.push_back(std::move(g));
    }
    r.ok = true;
    return r;
}
inline CommProcSnap& comm_boot_snap() {  // taken by comm_select_path before the probe
    static CommProcSnap s;
    return s;
}
inline bool comm_desktop_client(unsigned pid) {
    std::string c = comm_sysfs_read("/proc/" + std::to_string(pid) + "/comm");
    // /proc/<pid>/comm (15 characters at most): X servers, compositors, shells and their helpers, browsers
    static const char* const kDesk[] = {"Xorg", "Xwayland", "xwaylandvideo", "X", "kwin", "gnome-shell", "mutter",
                                        "plasmashell", "plasma", "kded", "ksmserver", "kscreen", "xdg-desktop",
                                        "gsd-", "gnome-", "sway", "Hyprland", "weston", "cosmic-", "labwc", "wayfire",
                                        "nvidia-settings", "chromium", "chrome", "firefox", "Web Content",
                                        "GPU Process", "electron"};
    for (const char* d : kDesk)
        if (c.rfind(d, 0) == 0) return true;
    // QE_COMM_CACHE_IGNORE=name,name: more processes of this display that do not count (prefixes of their comm)
    if (const char* e = env_str("QE_COMM_CACHE_IGNORE")) {
        std::stringstream ss(e);
        for (std::string w; std::getline(ss, w, ',');)
            if (!w.empty() && c.rfind(w, 0) == 0) return true;
    }
    return false;
}
inline unsigned long long comm_now_us() {
    return (unsigned long long)std::chrono::duration_cast<std::chrono::microseconds>(
               std::chrono::system_clock::now().time_since_epoch()).count();
}
inline bool comm_in_host_pidns(const CommProcSnap& now) {  // this process appears in NVML's lists of every card
    const unsigned me = (unsigned)getpid();
    for (auto& v : now.pids) {
        bool found = false;
        for (unsigned p : v) found |= p == me;
        if (!found) return false;
    }
    return !now.pids.empty();
}
// this process on card i as NVML lists it: its PID in the host's namespace; in a container (host PIDs) the compute
// process whose memory grew most since the boot snapshot (this server allocates between the two; review round 4 S7);
// 0: not identifiable (busy)
inline unsigned comm_self_pid(const CommProcSnap& now, size_t i, bool hostns) {
    if (hostns) return (unsigned)getpid();
    const CommProcSnap& b = comm_boot_snap();
    unsigned self = 0;
    long long best = 0;
    for (auto& [pid, mem] : now.cmem[i]) {
        if (mem == ~0ull) continue;  // (memory not reported)
        unsigned long long was = 0;
        if (b.ok && i < b.cmem.size())
            for (auto& [p0, m0] : b.cmem[i]) if (p0 == pid && m0 != ~0ull) was = m0;
        const long long grew = (long long)mem - (long long)was;
        if (grew > best) best = grew, self = pid;
    }
    return self;
}
// measured GPU use of other processes since `since_us` (NVML per-process utilization): 1 busy (why set), 0 not, -1
// unsupported (the list rule decides). Any other process above QE_COMM_BUSY_PCT (2) % SM, memory bandwidth (copies
// show there, review round 4 S2), encoder or decoder. A clock that stepped back past `since` is busy (S3).
inline int comm_util_busy(const int* devices, int ng, unsigned long long since_us, std::string& why) {
    const CommNvml& n = CommNvml::get();
    if (!n.putil) return -1;
    if (comm_now_us() < since_us) { why = "the clock stepped back while measuring"; return 1; }
    const CommProcSnap now = comm_proc_snap(devices, ng);
    if (!now.ok) { why = "NVML process lists unavailable"; return 1; }
    const bool hostns = comm_in_host_pidns(now);
    struct US { unsigned pid; unsigned long long ts; unsigned sm, mem, enc, dec; };
    static_assert(sizeof(US) == 32, "nvmlProcessUtilizationSample_t");
    static const double pct = env_double("QE_COMM_BUSY_PCT", 2.0);
    for (int i = 0; i < ng; i++) {
        void* h = n.dev(comm_bus_id(devices[i]).c_str());
        if (!h) { why = "NVML device lookup failed"; return 1; }
        const unsigned self = comm_self_pid(now, (size_t)i, hostns);
        if (!self) { why = "this process not identifiable in NVML (container)"; return 1; }
        unsigned cnt = 0;
        int e = n.putil(h, nullptr, &cnt, since_us);
        if (e == 6) continue;  // NVML_ERROR_NOT_FOUND: no samples since the start, nobody else ran
        if (e != 0 && e != 7) return -1;  // (unsupported: the list rule)
        std::vector<US> v(cnt + 32);
        cnt = (unsigned)v.size();
        e = n.putil(h, v.data(), &cnt, since_us);
        if (e == 6) continue;
        if (e != 0) return -1;
        for (unsigned k = 0; k < cnt; k++) {
            const US& u = v[k];
            if (u.pid == self) continue;
            if (std::max({u.sm, u.mem, u.enc, u.dec}) > pct) {
                why = "process " + std::to_string(u.pid) + " (" + comm_sysfs_read("/proc/" + std::to_string(u.pid) + "/comm") +
                      ") used card " + std::to_string(i) + " (" + std::to_string(u.sm) + " % SM, " + std::to_string(u.mem) +
                      " % memory) while measuring";
                return 1;
            }
        }
    }
    return 0;
}
// polled during a measurement and ORed (review round 4 S3). Measured on this box (driver 615): NVML returns the
// samples of running processes only; a process that exited is gone at once ("Not Found" right after a 1 s job and 5 s
// later). So the watch polls every QE_COMM_BUSY_POLL_MS (100) on a thread for the whole measurement, plus after every
// leg / probe phase; a job shorter than one poll interval can still pass unseen.
struct CommBusyWatch {
    // the state lives in a shared_ptr: a poll stuck in NVML (a card in a bad state) is detached at stop() after
    // QE_COMM_BUSY_JOIN_MS (2000) and the result counts as busy (not cached), start-up goes on (review round 5 T4)
    struct St {
        std::vector<int> devices;
        std::atomic<bool> busy{false}, unsupported{false}, stop{false}, done{true};
        std::string why;
        std::mutex mu;
        void poll(unsigned long long since_us) {
            if (busy) return;
            std::string w;
            const int r = comm_util_busy(devices.data(), (int)devices.size(), since_us, w);
            std::lock_guard<std::mutex> lk(mu);
            if (r == 1 && !busy) busy = true, why = w;
            if (r < 0) unsupported = true;
        }
    };
    std::shared_ptr<St> st;
    std::thread th;
    CommBusyWatch(const int* d, int n) : st(std::make_shared<St>()) { st->devices.assign(d, d + n); }
    void poll(unsigned long long since_us) { st->poll(since_us); }
    void start(unsigned long long since_us) {
        const int ms = std::max(10, env_int("QE_COMM_BUSY_POLL_MS", 100));
        st->done = false;
        std::shared_ptr<St> s = st;
        th = std::thread([s, since_us, ms] {
            while (!s->stop) {
                s->poll(since_us);
                for (int i = 0; i < ms / 10 && !s->stop; i++) usleep(10000);
            }
            s->done = true;
        });
    }
    void stop() {
        st->stop = true;
        if (!th.joinable()) return;
        const int lim = std::max(0, env_int("QE_COMM_BUSY_JOIN_MS", 2000)) / 10;
        for (int i = 0; i < lim && !st->done; i++) usleep(10000);
        if (st->done) {
            th.join();
        } else {  // (stuck in NVML: leave it, count the measurement as unverified)
            th.detach();
            std::lock_guard<std::mutex> lk(st->mu);
            if (!st->busy) st->busy = true, st->why = "the NVML busy poll did not return";
        }
    }
    bool busy() const { return st->busy; }
    bool unsupported() const { return st->unsupported; }
    std::string reason() { std::lock_guard<std::mutex> lk(st->mu); return st->why; }
    ~CommBusyWatch() { stop(); }
};
// true: do not cache; why: the reason. since_us: the start of the measurement; watch: the polls during it (or null).
// In the host's PID namespace any other compute (non-graphics) context on a card is busy too (S2: a copy-only job
// shows ~0 % SM); the display's clients hold graphics contexts and count only by their measured use.
inline bool comm_cache_busy(const int* devices, int ng, std::string& why, unsigned long long since_us = 0,
                            CommBusyWatch* watch = nullptr) {
    if (watch) watch->stop();
    if (watch && watch->busy()) { why = watch->reason(); return true; }
    const CommProcSnap& b = comm_boot_snap();
    const CommProcSnap now = comm_proc_snap(devices, ng);
    if (!b.ok || !now.ok) { why = "NVML process lists unavailable"; return true; }
    const unsigned me = (unsigned)getpid();
    const bool hostns = comm_in_host_pidns(now);
    if (hostns)
        for (size_t i = 0; i < now.cmem.size(); i++)
            for (auto& [pid, mem] : now.cmem[i]) {
                bool gfx = false;
                for (unsigned g : now.gpids[i]) gfx |= g == pid;
                if (pid != me && !gfx) {
                    why = "compute process " + std::to_string(pid) + " (" + comm_sysfs_read("/proc/" + std::to_string(pid) + "/comm") +
                          ") on card " + std::to_string(i);
                    return true;
                }
            }
    const int r = comm_util_busy(devices, ng, since_us, why);
    if (r == 1) return true;
    if (r == 0 && !(watch && watch->unsupported())) return false;
    for (size_t i = 0; i < now.pids.size(); i++) {  // fallback without per-process utilization: the process lists
        if (hostns) {
            for (unsigned p : now.pids[i])
                if (p != me && !comm_desktop_client(p)) {
                    why = "process " + std::to_string(p) + " (" + comm_sysfs_read("/proc/" + std::to_string(p) + "/comm") +
                          ") on card " + std::to_string(i);
                    return true;
                }
        } else {
            if (now.pids[i].empty()) { why = "container without NVML process mapping"; return true; }
            if (i < b.pids.size() && now.pids[i].size() > b.pids[i].size()) {
                why = "more processes on card " + std::to_string(i) + " than at boot";
                return true;
            }
        }
    }
    return false;
}
// the trained PCIe link of a card (review round 3 R3): sysfs current_link_speed / width polled every ms for up to
// QE_COMM_LINK_MS (200) while host -> card copies keep running, the highest speed seen (an idle card sits at 2.5 GT/s
// and ramps with its P-state; a riser or slot that trains lower shows only under load); stops early at max_link_speed
// or once the speed rose and held for 50 ms (a card pausing longer at an intermediate step can still key it: the
// [comm] link line shows the value; review round 5 T6)
inline std::string comm_trained_link(int device) {
    const std::string sys = "/sys/bus/pci/devices/" + comm_bus_id(device);
    const std::string mx = comm_sysfs_read(sys + "/max_link_speed");
    int prev = 0;
    cudaGetDevice(&prev);
    void *h = nullptr, *d = nullptr;
    cudaStream_t st = nullptr;
    const size_t n = 8u << 20;
    double best = 0.0;
    std::string bs = "?", wd = comm_sysfs_read(sys + "/current_link_width");
    bool ok = cudaSetDevice(device) == cudaSuccess && cudaMallocHost(&h, n) == cudaSuccess && cudaMalloc(&d, n) == cudaSuccess &&
              cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking) == cudaSuccess;
    int ms = 0, stable = 0;
    if (ok) {
        const int lim = std::max(1, env_int("QE_COMM_LINK_MS", 200));
        for (; ms < lim; ms++) {
            if (cudaStreamQuery(st) == cudaSuccess)
                for (int i = 0; i < 8; i++) cudaMemcpyAsync(d, h, n, cudaMemcpyHostToDevice, st);
            usleep(1000);
            const std::string cur = comm_sysfs_read(sys + "/current_link_speed");
            const double v = atof(cur.c_str());
            if (v > best) best = v, bs = cur, wd = comm_sysfs_read(sys + "/current_link_width"), stable = 0;
            else stable++;
            if (!mx.empty() && cur == mx && ms >= 5) break;  // at the card's maximum
            if (best > 2.5 && stable >= 50) break;            // risen and stable for 50 ms (a link that trains lower)
        }
        cudaStreamSynchronize(st);
    }
    if (st) cudaStreamDestroy(st);
    if (d) cudaFree(d);
    if (h) cudaFreeHost(h);
    if (!ok) cudaGetLastError();  // (only this function's own failed calls)
    cudaSetDevice(prev);
    fprintf(stderr, "[comm] link GPU%d: trained %s x%s (max %s) after %d ms of copies\n", device, bs.c_str(), wd.c_str(),
            mx.c_str(), ms);
    return bs + " x" + wd;
}
// the card's settings that change compute vs transfer time: power limit, application / max clocks, clock offset,
// BAR1 size (ReBAR), the PCIe generation / width the card and slot allow (the trained link: comm_trained_link), the
// active NVLink links and their peers
inline std::string comm_nvml_key(int device) {
    const CommNvml& n = CommNvml::get();
    void* h = n.dev(comm_bus_id(device).c_str());
    if (!h) return " nvml ?";
    std::ostringstream k;
    unsigned v = 0, w = 0;
    int o = 0;
    unsigned long long b[3] = {0, 0, 0};
    k << " pl " << (n.plimit && n.plimit(h, &v) == 0 ? (long)v : -1L);
    k << " app " << (n.app_clock && n.app_clock(h, 0, &v) == 0 ? (long)v : -1L) << "/"
      << (n.app_clock && n.app_clock(h, 2, &w) == 0 ? (long)w : -1L);
    k << " max " << (n.max_clock && n.max_clock(h, 0, &v) == 0 ? (long)v : -1L) << "/"
      << (n.max_clock && n.max_clock(h, 2, &w) == 0 ? (long)w : -1L);
    k << " off " << (n.gpc_off && n.gpc_off(h, &o) == 0 ? std::to_string(o) : std::string("?"));
    k << " bar1 " << (n.bar1 && n.bar1(h, b) == 0 ? (long long)(b[0] >> 20) : -1LL) << "M";
    k << " gen " << (n.max_gen && n.max_gen(h, &v) == 0 ? (long)v : -1L) << " x"
      << (n.max_width && n.max_width(h, &w) == 0 ? (long)w : -1L);
    // NVLink: the active links and the card at each one's other end (club-3090 #2000: a bridge installed after the
    // first boot left the PCIe-era probe in force, peer 26 instead of 49 GB/s). No bridge / no NVLink: "nvl -"
    k << " nvl";
    int nl = 0;
    for (unsigned l = 0; n.nvl_state && l < 18; l++) {  // NVML_NVLINK_MAX_LINKS
        int on = 0;
        if (n.nvl_state(h, l, &on) != 0 || !on) continue;
        // nvmlPciInfo_t: busIdLegacy[16], domain, bus, device, pciDeviceId, pciSubSystemId, busId[32]
        unsigned pci[32] = {};
        k << " " << l;
        if (n.nvl_remote && n.nvl_remote(h, l, pci) == 0) k << ":" << pci[4] << "." << pci[5] << "." << pci[6];
        nl++;
    }
    if (!nl) k << " -";
    // (locked clocks, nvidia-smi -lgc, are not readable: NVML and nvidia-smi report the VBIOS range; review round 3 R2.
    // After a clock change: QE_COMM_CACHE=refresh)
    return k.str();
}
// the hardware / software part of the key (device order matters: rank i is devices[i])
inline std::string comm_cache_hw_key_(const int* devices, int ng, bool p2p);
inline std::string comm_cache_hw_key(const int* devices, int ng, bool p2p) {  // memoized (the link read loads the cards)
    static std::map<std::string, std::string> memo;
    static std::mutex mu;  // (start-up threads only today; locked anyway, review round 3 R10)
    std::lock_guard<std::mutex> lk(mu);
    std::string id = std::to_string(p2p);
    for (int i = 0; i < ng; i++) id += "," + std::to_string(devices[i]);
    auto it = memo.find(id);
    if (it != memo.end()) return it->second;
    return memo[id] = comm_cache_hw_key_(devices, ng, p2p);
}
inline std::string comm_cache_hw_key_(const int* devices, int ng, bool p2p) {
    std::ostringstream k;
    k << "v" << kCommCacheVer << "|build " << qe_internal_build_id();
    int rt = 0, drv = 0;
    cudaRuntimeGetVersion(&rt);
    cudaDriverGetVersion(&drv);
    k << "|cuda rt " << rt << " drv " << drv << "|" << comm_sysfs_read("/proc/driver/nvidia/version").substr(0, 120);
    k << "|ngpu " << ng << " p2p " << (p2p ? 1 : 0);
    for (int i = 0; i < ng; i++) {
        const std::string bus = comm_bus_id(devices[i]), sys = "/sys/bus/pci/devices/" + bus;
        struct stat st;
        k << "|gpu " << i << " " << qe_gpu_guard::uuid_of(devices[i]) << " " << bus << " "
          << comm_sysfs_read(sys + "/max_link_speed") << " trained " << comm_trained_link(devices[i]) << " numa "
          << comm_sysfs_read(sys + "/numa_node") << " iommu " << (lstat((sys + "/iommu_group").c_str(), &st) == 0 ? 1 : 0)
          << comm_nvml_key(devices[i]);
    }
    {  // IOMMU on the kernel command line (passthrough vs translated peer routes)
        std::string cl = comm_sysfs_read("/proc/cmdline"), io;
        std::istringstream is(cl);
        for (std::string w; is >> w;)
            if (w.rfind("iommu", 0) == 0 || w.rfind("intel_iommu", 0) == 0 || w.rfind("amd_iommu", 0) == 0) io += " " + w;
        k << "|cmdline" << io;
    }
    k << "|peer";
    for (int i = 0; i < ng; i++)
        for (int j = 0; j < ng; j++) {
            int can = 0;
            if (i != j && cudaDeviceCanAccessPeer(&can, devices[i], devices[j]) != cudaSuccess) can = -1;
            k << (i == j ? "-" : std::to_string(can));
        }
    cudaGetLastError();
    // collectives' knobs (a changed one may change the answer)
    for (const char* v : {"QE_P2P", "QE_LL16", "QE_AR_SPLIT", "QE_AR_RSUM", "QE_AR_RS_MIN", "QE_AR_LOSSLESS", "QE_AG_LL",
                          "QE_EARLY_GEMV", "QE_EARLY_FUSED", "FLASH_XP2P", "FLASH_QX16_DEC", "QE_COMM_PROBE_MB",
                          "QE_COMM_CORUN_RUNS", "QE_COMM_CORUN_MS", "QE_COMM_TEST_TAX", "QE_DECODE_TEST_STORE_US"})
        if (const char* e = env_str(v)) k << "|" << v << "=" << e;
    return k.str();
}
inline std::string comm_cache_path(const std::string& what, const std::string& key) {
    const std::string dir = comm_cache_dir();
    if (dir.empty()) return "";
    unsigned long long h = 1469598103934665603ull;  // FNV-1a
    for (unsigned char ch : key) h = (h ^ ch) * 1099511628211ull;
    char name[64];
    snprintf(name, sizeof name, "/comm_%s_%016llx.txt", what.c_str(), h);
    return dir + name;
}
// the cached values of `what` under `key` (false: none, another key, or a corrupt file)
inline bool comm_cache_read(const std::string& what, const std::string& key, std::map<std::string, std::string>& kv,
                            std::string* when = nullptr) {
    static const double days = env_double("QE_COMM_CACHE_DAYS", 30.0);
    if (comm_cache_mode() != 1) return false;
    const std::string path = comm_cache_path(what, key);
    if (path.empty()) return false;
    std::ifstream f(path);
    if (!f) return false;
    std::string l;
    if (!std::getline(f, l) || l != "qe-comm-cache " + std::to_string(kCommCacheVer) + " " + what) return false;
    if (!std::getline(f, l) || l != "key " + key) return false;
    std::map<std::string, std::string> m;
    bool end = false;
    while (std::getline(f, l)) {
        if (l == "end") { end = true; break; }
        const size_t e = l.find('=');
        if (e == std::string::npos) return false;
        m[l.substr(0, e)] = l.substr(e + 1);
    }
    if (!end) return false;  // truncated
    // expiry: an old measurement is taken again (cards age, BIOS / driver settings drift without a key change)
    if (days > 0 && (!m.count("epoch") || difftime(time(nullptr), (time_t)atoll(m["epoch"].c_str())) > days * 86400.0))
        return false;
    if (when) *when = m.count("measured") ? m["measured"] : "?";
    kv = std::move(m);
    return true;
}
inline void comm_cache_write(const std::string& what, const std::string& key, std::map<std::string, std::string> kv) {
    if (comm_cache_mode() == 0) return;
    const std::string path = comm_cache_path(what, key);
    if (path.empty()) return;
    qe_gpu_guard::mkdirs(comm_cache_dir());
    char ts[32];
    const time_t t = time(nullptr);
    strftime(ts, sizeof ts, "%Y-%m-%d %H:%M:%S", localtime(&t));
    kv["measured"] = ts;
    kv["epoch"] = std::to_string((long long)t);
    // a fresh temporary name (mkstemp: O_CREAT | O_EXCL, never follows a planted link), then rename (atomic)
    std::string tmpl = path + ".tmp.XXXXXX";
    std::vector<char> tb(tmpl.begin(), tmpl.end());
    tb.push_back('\0');
    const int fd = mkstemp(tb.data());
    if (fd < 0) return;  // no cache directory: just measure each start
    fchmod(fd, 0644);
    std::string body = "qe-comm-cache " + std::to_string(kCommCacheVer) + " " + what + "\nkey " + key + "\n";
    for (auto& [a, b] : kv) body += a + "=" + b + "\n";
    body += "end\n";
    bool ok = write(fd, body.data(), body.size()) == (ssize_t)body.size();
    ok = fsync(fd) == 0 && ok;
    ok = close(fd) == 0 && ok;
    if (!ok || rename(tb.data(), path.c_str()) != 0) unlink(tb.data());
}

// the decode collectives' calibration (flash_capi.cpp / engine.cpp decode_calibrate): picks per mode (0 single sequence,
// 1 batched; kXfer* values) and the measured ms per step
struct DecodeCal {
    int x[2] = {0, 0};
    double tp[2] = {-1, -1}, th[2] = {-1, -1};
    std::string when;
};
// the model's identity: the real path of its directory and a hash of its config and weight index (another checkpoint
// under the same name, or the same name mounted elsewhere, is another model)
inline std::string comm_model_id(const std::string& model_dir) {
    char rp[PATH_MAX];
    std::string id = realpath(model_dir.c_str(), rp) ? rp : model_dir;
    unsigned long long h = 1469598103934665603ull;
    for (const char* f : {"config.json", "model.safetensors.index.json", "flash_next.json", "quant_config.json"}) {
        std::ifstream in(model_dir + "/" + f, std::ios::binary);
        for (char ch; in.get(ch);) h = (h ^ (unsigned char)ch) * 1099511628211ull;
    }
    char b[32];
    snprintf(b, sizeof b, " ckpt %016llx", h);
    return id + b;
}
inline std::string comm_decode_key(const int* devices, int ng, bool p2p, const char* engine, const std::string& model_dir,
                                   int k, int nb, int steps, int warm, double margin, const std::string& extra = "") {
    std::string base = comm_model_id(model_dir) + " " + extra;
    for (const char* v : {"FLASH_KV", "FLASH_GRAPH", "FLASH_EC", "FLASH_MAX_SEQS", "QE_MAX_SEQS", "FLASH_PAR", "QE_KV_SPLIT",
                          "QE_SPLIT", "FLASH_MTP_K", "QE_DECODE_CAL_ROUNDS"})
        if (const char* e = env_str(v)) base += std::string(" ") + v + "=" + e;
    char t[2048];
    snprintf(t, sizeof t, "|decode %s model %s k %d nb %d steps %d warm %d margin %.4f rounds %d", engine, base.c_str(), k,
             nb, steps, warm, margin, env_int("QE_DECODE_CAL_ROUNDS", 4));
    return comm_cache_hw_key(devices, ng, p2p) + t;
}
inline bool comm_decode_cache_get(const std::string& key, DecodeCal& d) {
    std::map<std::string, std::string> kv;
    if (!comm_cache_read("decode", key, kv, &d.when)) return false;
    return kv.count("pick") && kv.count("ms") &&
           sscanf(kv["pick"].c_str(), "%d %d", &d.x[0], &d.x[1]) == 2 &&
           sscanf(kv["ms"].c_str(), "%lf %lf %lf %lf", &d.tp[0], &d.th[0], &d.tp[1], &d.th[1]) == 4 &&
           (d.x[0] == 0 || d.x[0] == 2) && (d.x[1] == 0 || d.x[1] == 2);
}
inline void comm_decode_cache_put(const std::string& key, const DecodeCal& d) {
    char a[64], b[128];
    snprintf(a, sizeof a, "%d %d", d.x[0], d.x[1]);
    snprintf(b, sizeof b, "%.9g %.9g %.9g %.9g", d.tp[0], d.th[0], d.tp[1], d.th[1]);
    comm_cache_write("decode", key, {{"pick", a}, {"ms", b}});
}
