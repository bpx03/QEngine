// Host side of the collective watchdog (watchdog.cuh): the host-mapped record, per-device symbol
// setup, the fatal-error path of CUDA_CHECK / CHECK and fault injection for tests.
#include "../env.h"
#include "watchdog.cuh"

#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <string>
#include <cstdarg>
#include <cstdio>
#include <mutex>
#include <thread>
#include <vector>

namespace {
std::vector<void (*)(const QeWdDev&)>& wd_setters() {
    static std::vector<void (*)(const QeWdDev&)> v;
    return v;
}
std::mutex g_wd_mu;
QeWdHost* g_wd = nullptr;           // host pointer
int g_wd_dev[8] = {-1, -1, -1, -1, -1, -1, -1, -1};  // rank -> CUDA device
int g_wd_ngpu = 0;

int budget_ms_env() {
    static const int v = [] {
        return env_int("QE_COLL_TIMEOUT_MS", 20000);
    }();
    return v;
}

const char* site_name(unsigned s) {
    static const char* n[kWdSites] = {"?",
                                      "all-reduce flags (allreduce_kernel / exchange)",
                                      "prefill all-reduce flags (rows / rs_rows exchange)",
                                      "allgather flags (allgather_kernel / exchange)",
                                      "LL all-reduce (allreduce_ll_kernel)",
                                      "LL16 all-reduce, peer partials (allreduce_ll16_kernel)",
                                      "LL16 all-reduce, owner's sum (allreduce_ll16_kernel MODE 2)",
                                      "lossless LL all-reduce (allreduce_llc_kernel)",
                                      "quantized LL all-reduce (allreduce_llq_kernel)",
                                      "GEMV-fused LL16 all-reduce (ar_fused ll16_chunk)",
                                      "copy-engine all-reduce flags (ce_wait_kernel)",
                                      "early GEMV ready tag (gemv_w4h_early, same GPU)",
                                      "kv-split decode softmax stats flag",
                                      "kv-split prefill softmax stats flag",
                                      "attn_pkp work queue (same GPU)",
                                      "Flash-Next QSA exchange flags (q / partials / indexer, n-gram stage)",
                                      "Flash-Next row all-gather flags / acks"};
    return s < kWdSites ? n[s] : "?";
}

int dev_of(int rank) { return rank >= 0 && rank < 8 && g_wd_dev[rank] >= 0 ? g_wd_dev[rank] : rank; }

void crumbs_print(bool query_streams);
// Prints the records; true when a GPU timed out (the error is the watchdog's).
bool wd_report_host() {
    if (!g_wd) return false;
    // the other GPUs trap at their next check after the abort word (milliseconds): wait for them
    for (int i = 0; i < 50; i++) {
        int n = 0;
        for (int r = 0; r < g_wd_ngpu; r++) n += ((volatile QeWdRec*)&g_wd->rec[r])->kind != 0;
        if (n == g_wd_ngpu) break;
        usleep(10000);
    }
    if (env_str("QE_WD_DEBUG"))
        for (int r = 0; r < g_wd_ngpu; r++) {
            const volatile QeWdRec& x = g_wd->rec[r];
            fprintf(stderr, "[watchdog] abort %x rec %d: kind %u site %u peer %d want %x seen %x waited %u b %u t %u\n",
                    g_wd->abort, r, x.kind, x.site, x.peer, x.want, x.seen, x.waited, x.block, x.thread);
        }
    int first = -1;  // the GPU that raised the abort, else any that timed out
    const unsigned ab = g_wd->abort;
    if (ab >= 1 && (int)ab <= g_wd_ngpu && ((volatile QeWdRec*)&g_wd->rec[ab - 1])->kind == 1) first = (int)ab - 1;
    for (int r = 0; r < g_wd_ngpu && first < 0; r++)
        if (((volatile QeWdRec*)&g_wd->rec[r])->kind == 1) first = r;
    if (first < 0) {  // not a timeout: show which waits the abort stopped
        for (int r = 0; r < g_wd_ngpu; r++) {
            const volatile QeWdRec& x = g_wd->rec[r];
            if (x.kind == 2)
                fprintf(stderr, "qengine: GPU%d stopped by the abort in %s (block %u thread %u)\n", dev_of(r),
                        site_name(x.site), x.block, x.thread);
        }
        return false;
    }
    const volatile QeWdRec& f = g_wd->rec[first];
    const double ms = f.waited * 1.048576;
    if (f.peer >= 0)
        fprintf(stderr,
                "qengine: collective timeout: GPU%d (rank %d) waited %.1f s in %s for GPU%d (rank %d), expected 0x%x, "
                "last seen 0x%x (block %u thread %u)\n",
                dev_of(first), first, ms / 1000.0, site_name(f.site), dev_of(f.peer), f.peer, f.want, f.seen, f.block,
                f.thread);
    else
        fprintf(stderr,
                "qengine: collective timeout: GPU%d (rank %d) waited %.1f s in %s, expected 0x%x, last seen 0x%x "
                "(block %u thread %u)\n",
                dev_of(first), first, ms / 1000.0, site_name(f.site), f.want, f.seen, f.block, f.thread);
    // suspect: a GPU with no wait recorded (failed, or it skipped the collective), else the GPU the
    // others were waiting for (a GPU ahead of the rest waits for a peer too, but alone)
    std::vector<int> silent, votes(g_wd_ngpu, 0);
    for (int r = 0; r < g_wd_ngpu; r++) {
        const volatile QeWdRec& x = g_wd->rec[r];
        if (x.kind && x.peer >= 0 && x.peer < g_wd_ngpu) votes[x.peer]++;
        if (r == first) continue;
        if (x.kind == 0) silent.push_back(r);
        else if (x.peer >= 0)
            fprintf(stderr, "qengine:   GPU%d (rank %d) %s in %s (for GPU%d), expected 0x%x\n", dev_of(r), r,
                    x.kind == 1 ? "timed out" : "was waiting, stopped by the abort,", site_name(x.site), dev_of(x.peer), x.want);
        else
            fprintf(stderr, "qengine:   GPU%d (rank %d) %s in %s, expected 0x%x\n", dev_of(r), r,
                    x.kind == 1 ? "timed out" : "was waiting, stopped by the abort,", site_name(x.site), x.want);
    }
    for (int r : silent)
        fprintf(stderr, "qengine:   GPU%d (rank %d) recorded no wait: GPU%d not responding (failed GPU, or it skipped "
                        "the collective)\n", dev_of(r), r, dev_of(r));
    if (silent.empty()) {
        int s = 0;
        for (int r = 1; r < g_wd_ngpu; r++)
            if (votes[r] > votes[s]) s = r;
        if (votes[s] > 0)
            fprintf(stderr, "qengine:   GPU%d (rank %d) not responding or out of step (waited for by %d GPU%s)\n", dev_of(s), s,
                    votes[s], votes[s] > 1 ? "s" : "");
    }
    return true;
}
// The host-memory records first, then the workers' breadcrumbs. query_streams: also ask the driver for each stream's
// state (bounded by a 1 s deadline on a detached thread). Never on a host timeout: there the driver itself may be the
// thing that is stuck, and the process must still reach its exit (issue #3 D3).
bool wd_report(bool query_streams) {
    const bool r = wd_report_host();
    crumbs_print(query_streams);
    return r;
}

struct Crumb {
    std::atomic<const char*> what{nullptr};
    std::atomic<long long> a{0}, b{0}, n{0}, t_ns{0};
    std::atomic<int> dev{-1};
    std::atomic<void*> st[4] = {};
};
Crumb g_crumb[16];
long long now_ns_wd() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
std::atomic<int> g_fatal{0};
const auto g_load_time = std::chrono::steady_clock::now();
// the embedding program's last word before a fatal exit (qe_set_exit_hook: the server ends its open requests with an
// error event instead of a dropped connection). Run on its own thread, bounded by QE_EXIT_HOOK_S (5) seconds: the
// process exits even when the hook blocks. It must not call into the engine.
std::atomic<void (*)(const char*)> g_exit_hook{nullptr};
char g_exit_reason[640];
void run_exit_hook() {
    void (*f)(const char*) = g_exit_hook.exchange(nullptr);
    if (!f) return;
    static const double lim = env_double("QE_EXIT_HOOK_S", 5.0);
    static std::atomic<int> done{0};
    try {
        std::thread([f] { f(g_exit_reason); done.store(1); }).detach();
    } catch (...) {
        return;
    }
    for (int i = 0; i < (int)(lim * 100) && !done.load(); i++) usleep(10000);
}
void set_reason(const char* fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(g_exit_reason, sizeof g_exit_reason, fmt, ap);
    va_end(ap);
}
[[noreturn]] void fatal_exit(bool wd) {
    fflush(stderr);
    run_exit_hook();
    if (wd) {
        fprintf(stderr, "qengine: exiting (code 75) so the server can be restarted\n");
        fflush(stderr);
        fflush(stdout);
        _exit(75);
    }
    fflush(stderr);
    fflush(stdout);  // the server's persistent log (server/logtee.py) gets the last lines before the process ends
    if (env_int("QE_FATAL_EXIT", 0)) _exit(70);  // no core dump
    abort();
}
}  // namespace

void qe_wd_register(void (*set)(const QeWdDev&)) { wd_setters().push_back(set); }

int qe_wd_budget_ms() { return std::max(0, budget_ms_env()); }

void qe_wd_init(int rank, int device) {
    {
        std::lock_guard<std::mutex> lk(g_wd_mu);
        if (!g_wd) {
            void* p = nullptr;
            CUDA_CHECK(cudaHostAlloc(&p, sizeof(QeWdHost), cudaHostAllocMapped | cudaHostAllocPortable));
            memset(p, 0, sizeof(QeWdHost));
            g_wd = static_cast<QeWdHost*>(p);
        }
        if (rank >= 0 && rank < 8) {
            g_wd_dev[rank] = device;
            g_wd_ngpu = std::max(g_wd_ngpu, rank + 1);
        }
    }
    QeWdDev d{};
    CUDA_CHECK(cudaHostGetDevicePointer((void**)&d.wd, g_wd, 0));
    const int ms = budget_ms_env();
    d.budget = ms <= 0 ? 0xFFFFFFFFu  // 18-bit ticks in the wait state (watchdog.cuh): at most 2^17 (137 s)
                     : (unsigned)std::min(1ull << 17, std::max(1ull, (unsigned long long)ms * 1000000ull >> 20));
    d.rank = rank;
    d.claim = 0;
    for (auto f : wd_setters()) f(d);
    if (env_str("QE_WD_DEBUG"))
        fprintf(stderr, "[watchdog] rank %d device %d: %zu modules, budget %u ticks\n", rank, device, wd_setters().size(), d.budget);
}

void qe_fatal_begin() {
    if (g_wd && !g_wd->abort) g_wd->abort = 0xFFu;  // peers' waits trap at their next check
}

[[noreturn]] void qe_fatal_cuda(int err, const char* file, int line) {
    qe_fatal_begin();
    if (g_fatal.exchange(1)) {  // another thread reports and ends the process
        for (;;) pause();
    }
    const cudaError_t e = (cudaError_t)err;
    fprintf(stderr, "CUDA error %s at %s:%d: %s\n", cudaGetErrorName(e), file, line, cudaGetErrorString(e));
    set_reason("CUDA error %s (%s)", cudaGetErrorName(e), cudaGetErrorString(e));
    fatal_exit(wd_report(true));
}

[[noreturn]] void qe_fatal_check() {
    qe_fatal_begin();
    if (g_fatal.exchange(1)) {
        for (;;) pause();
    }
    set_reason("engine check failed (see the server log)");
    fatal_exit(false);
}

[[noreturn]] void qe_wd_host_timeout(const char* what) {
    qe_fatal_begin();
    if (g_fatal.exchange(1)) {
        for (;;) pause();
    }
    fprintf(stderr, "qengine: collective timeout (host): %s\n", what);
    set_reason("engine stall: %s", what);
    wd_report(false);  // the GPUs' view, if any trapped meanwhile; no driver call (a stuck driver must not block the exit)
    fatal_exit(true);
}

void qe_wd_crumb(int id, int device, const char* what, long long a, long long b) {
    if (id < 0 || id >= 16) return;
    Crumb& c = g_crumb[id];
    c.dev.store(device, std::memory_order_relaxed);
    c.a.store(a, std::memory_order_relaxed);
    c.b.store(b, std::memory_order_relaxed);
    c.t_ns.store(now_ns_wd(), std::memory_order_relaxed);
    c.n.fetch_add(1, std::memory_order_relaxed);
    c.what.store(what, std::memory_order_release);
}
void qe_wd_stream(int id, int device, void* stream) {
    if (id < 0 || id >= 16) return;
    g_crumb[id].dev.store(device);
    for (auto& s : g_crumb[id].st) {
        void* e = nullptr;
        if (s.compare_exchange_strong(e, stream)) return;
    }
}
void qe_wd_crumbs_print() { crumbs_print(false); }
namespace {
void crumbs_print(bool query_streams) {
    static std::atomic<int> once{0};
    if (once.exchange(1)) return;
    const long long now = now_ns_wd();
    // stream states: -1 = not queried / no answer before the deadline, 0 idle, 1 busy, 2 error. Shared with a detached
    // thread that may outlive the deadline (it then writes into a block nobody reads any more: leaked on purpose).
    struct Q {
        std::atomic<int> st[16][4];
        std::atomic<int> done{0};
    };
    Q* q = new Q;
    for (auto& row : q->st)
        for (auto& x : row) x.store(-1);
    bool answered = !query_streams;
    if (query_streams) {
        try {
            std::thread([q] {
                for (int i = 0; i < 16; i++)
                    for (int j = 0; j < 4; j++) {
                        void* p = g_crumb[i].st[j].load();
                        if (!p) continue;
                        const cudaError_t e = cudaStreamQuery((cudaStream_t)p);  // no device sync
                        q->st[i][j].store(e == cudaSuccess ? 0 : e == cudaErrorNotReady ? 1 : 2);
                    }
                (void)cudaGetLastError();
                q->done.store(1);
            }).detach();
            for (int i = 0; i < 100 && !q->done.load(); i++) usleep(10000);
            answered = q->done.load() != 0;
        } catch (...) {
        }
    }
    for (int i = 0; i < 16; i++) {
        Crumb& c = g_crumb[i];
        const char* w = c.what.load(std::memory_order_acquire);
        if (!w && c.dev.load() < 0) continue;
        std::string ss;
        for (int j = 0; j < 4; j++) {
            if (!c.st[j].load()) continue;
            const int v = q->st[i][j].load();
            ss += v == 0 ? " idle" : v == 1 ? " busy" : v == 2 ? " error" : " ?";
        }
        fprintf(stderr, "qengine:   worker %d GPU%d host: last launch '%s' (%lld, %lld), %lld launches, %.1f s ago; streams:%s\n", i,
                c.dev.load(), w ? w : "-", c.a.load(), c.b.load(), c.n.load(), w ? (now - c.t_ns.load()) / 1e9 : 0.0,
                ss.empty() ? " -" : ss.c_str());
    }
    if (!query_streams)
        fprintf(stderr, "qengine:   (stream states not queried on a host timeout: the driver may be the stuck part)\n");
    else if (!answered)
        fprintf(stderr, "qengine:   (the driver did not answer the stream queries within 1 s)\n");
    if (answered) delete q;
}
}  // namespace

bool qe_fault_skip(int rank, const char* kind) {
    static const int fr = env_int("QE_FAULT_SKIP_AR_RANK", -1);
    if (fr < 0 || rank != fr) return false;
    static const char* fk = env_str("QE_FAULT_KIND") ? env_str("QE_FAULT_KIND") : "ar";
    if (strcmp(fk, kind) != 0) return false;
    static const int after = env_int("QE_FAULT_AFTER", 1);
    // QE_FAULT_AFTER_S: the first collective at least this many seconds after the library was loaded (server tests:
    // the warm-up's collective count is not known in advance)
    static const double after_s = env_double("QE_FAULT_AFTER_S", -1.0);
    static std::atomic<int> n{0};
    static std::atomic<bool> fired{false};
    const int i = ++n;
    if (after_s >= 0) {
        if (fired || std::chrono::duration<double>(std::chrono::steady_clock::now() - g_load_time).count() < after_s) return false;
        fired = true;
    } else if (i != after) return false;
    fprintf(stderr, "[fault] rank %d skips %s collective #%d (QE_FAULT_SKIP_AR_RANK)\n", rank, kind, i);
    return true;
}

// the server's fatal-exit hook (see run_exit_hook); nullptr removes it
extern "C" __attribute__((visibility("default"))) void qe_set_exit_hook(void (*f)(const char*)) { g_exit_hook.store(f); }
