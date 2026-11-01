#pragma once
// Start-up diagnostics when CUDA cannot be used (issue #3 D5): after a crash the previous process can keep the GPUs
// while its core is written or while it is stuck in the driver, and a restart then failed with a bare
// "pin_host: cudaHostRegister: initialization error". These helpers name the processes on the GPUs and the likely
// causes. Also the fatal-signal handler (issue #3 5.2.4): a native backtrace of the faulting thread on SIGSEGV /
// SIGBUS / SIGFPE / SIGILL / SIGABRT, then the previous handler runs (Python's faulthandler: every Python stack).
#include "env.h"
#include <cuda_runtime.h>
#include <errno.h>
#include <execinfo.h>
#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <spawn.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#include <atomic>
#include <cstdio>
#include <cstring>
#include <string>

namespace qe_diag {

// stdout of argv (PATH lookup) within timeout_s, else what came so far. Never blocks past the deadline: on expiry the
// child is SIGKILLed and abandoned (no blocking waitpid: a child stuck in the driver (D state) ignores SIGKILL until its
// driver call returns; popen + `timeout` without -k waited for it forever)
inline std::string run_bounded(const char* const argv[], double timeout_s) {
    std::string out;
    int fd[2];
    if (pipe2(fd, O_CLOEXEC) != 0) return out;
    posix_spawn_file_actions_t fa;
    posix_spawn_file_actions_init(&fa);
    posix_spawn_file_actions_adddup2(&fa, fd[1], 1);
    posix_spawn_file_actions_addopen(&fa, 2, "/dev/null", O_WRONLY, 0);
    pid_t pid = -1;
    const int rc = posix_spawnp(&pid, argv[0], &fa, nullptr, const_cast<char* const*>(argv), environ);
    posix_spawn_file_actions_destroy(&fa);
    close(fd[1]);
    if (rc != 0) {
        close(fd[0]);
        return out;
    }
    struct timespec t0;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    bool eof = false;
    while (true) {
        struct timespec t;
        clock_gettime(CLOCK_MONOTONIC, &t);
        const double left = timeout_s - ((t.tv_sec - t0.tv_sec) + (t.tv_nsec - t0.tv_nsec) / 1e9);
        if (left <= 0) break;
        struct pollfd pf{fd[0], POLLIN, 0};
        const int r = poll(&pf, 1, (int)(left * 1000) + 1);
        if (r < 0 && errno == EINTR) continue;
        if (r <= 0) break;
        char b[512];
        const ssize_t n = read(fd[0], b, sizeof b);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) { eof = true; break; }
        out.append(b, (size_t)n);
    }
    close(fd[0]);
    int st = 0;
    if (!eof || waitpid(pid, &st, WNOHANG) == 0) {
        kill(pid, SIGKILL);
        for (int i = 0; i < 20 && waitpid(pid, &st, WNOHANG) == 0; i++) usleep(10000);  // reaped if it dies at once
    }
    return out;
}

// "pid name MiB; ..." of the compute processes on the GPUs (nvidia-smi, at most 10 s), "" when none or not available
inline std::string gpu_holders() {
    static const char* const argv[] = {"nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
                                       "--format=csv,noheader", nullptr};
    std::string out;
    const std::string raw = run_bounded(argv, 10.0);
    size_t a = 0;
    while (a < raw.size()) {
        size_t b = raw.find('\n', a);
        if (b == std::string::npos) b = raw.size();
        std::string s = raw.substr(a, b - a);
        a = b + 1;
        while (!s.empty() && (s.back() == '\r' || s.back() == ' ')) s.pop_back();
        if (s.empty()) continue;
        if (!out.empty()) out += "; ";
        out += s;
    }
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
