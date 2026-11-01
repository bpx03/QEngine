#pragma once
// Start-up diagnostics when CUDA cannot be used (issue #3 D5): after a crash the previous process can keep the GPUs
// while its core is written or while it is stuck in the driver, and a restart then failed with a bare
// "pin_host: cudaHostRegister: initialization error". These helpers name the processes on the GPUs and the likely
// causes. Also the fatal-signal handler (issue #3 5.2.4): a native backtrace of the faulting thread on SIGSEGV /
// SIGBUS / SIGFPE / SIGILL / SIGABRT, then the previous handler runs (Python's faulthandler: every Python stack).
#include "env.h"
#include <cuda_runtime.h>
#include <execinfo.h>
#include <signal.h>
#include <unistd.h>

#include <atomic>
#include <cstdio>
#include <cstring>
#include <string>

namespace qe_diag {

// "pid name MiB; ..." of the compute processes on the GPUs (nvidia-smi), "" when none or not available
inline std::string gpu_holders() {
    std::string out;
    FILE* f = popen("timeout 10 nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null", "r");
    if (!f) return out;
    char line[512];
    while (fgets(line, sizeof line, f)) {
        std::string s(line);
        while (!s.empty() && (s.back() == '\n' || s.back() == ' ')) s.pop_back();
        if (s.empty()) continue;
        if (!out.empty()) out += "; ";
        out += s;
    }
    pclose(f);
    return out;
}

// the message for a CUDA error at start-up that is not a plain out-of-memory
inline void explain_start_failure(const char* where, cudaError_t e) {
    const std::string h = gpu_holders();
    fprintf(stderr,
            "qflash: %s: %s (%s). CUDA cannot be used on these GPUs. Likely causes: a previous engine process still holds "
            "them (it may be writing a core dump or be stuck in the driver after a crash)%s%s; or the driver is in an "
            "error state (check `dmesg | grep -iE 'xid|nvrm'`). Wait for that process to end (or stop it); if the GPUs "
            "stay busy or nvidia-smi hangs, a reboot is needed.\n",
            where, cudaGetErrorName(e), cudaGetErrorString(e), h.empty() ? "" : "; processes on the GPUs now: ",
            h.c_str());
    fflush(stderr);
}

// ---- fatal signals
inline std::atomic<const char*>& job_tag() {  // the engine job in progress (FE::run), for the report
    static std::atomic<const char*> t{nullptr};
    return t;
}
inline struct sigaction* old_actions() {
    static struct sigaction a[65];
    return a;
}
inline void on_fatal(int sig, siginfo_t* si, void*) {
    char b[320];
    const char* tag = job_tag().load();
    const int n = snprintf(b, sizeof b,
                           "\nqengine: fatal signal %d (%s) at address %p in thread %ld; engine job: %s\nnative backtrace "
                           "(QE_CORE_DUMP=1 keeps a core dump):\n",
                           sig, sig == SIGSEGV ? "SIGSEGV" : sig == SIGBUS ? "SIGBUS" : sig == SIGABRT ? "SIGABRT" :
                           sig == SIGFPE ? "SIGFPE" : "SIGILL", si ? si->si_addr : nullptr, (long)gettid(), tag ? tag : "none");
    if (n > 0 && write(2, b, (size_t)n) < 0) {}
    void* bt[64];
    const int k = backtrace(bt, 64);
    backtrace_symbols_fd(bt, k, 2);
    // the previous handler (faulthandler: Python stacks, then the default action) gets the signal next: a fault
    // re-executes and faults again, a raise()d signal is pending until this handler returns
    sigaction(sig, &old_actions()[sig], nullptr);
    if (sig == SIGABRT) raise(sig);
}
// once per process; FLASH_FATAL_BT=0: off
inline void install_fatal_handler() {
    static bool done = false;
    if (done || !env_int("FLASH_FATAL_BT", 1)) return;
    done = true;
    void* bt[4];
    backtrace(bt, 4);  // loads libgcc now, not inside the handler
    for (int sig : {SIGSEGV, SIGBUS, SIGFPE, SIGILL, SIGABRT}) {
        struct sigaction sa{};
        sa.sa_sigaction = on_fatal;
        sa.sa_flags = SA_SIGINFO | SA_ONSTACK;
        sigemptyset(&sa.sa_mask);
        sigaction(sig, &sa, &old_actions()[sig]);
    }
}

}  // namespace qe_diag
