#pragma once
// Non-blocking stdout / stderr for the engine threads (issue #32). The server's fd 1 / fd 2 are a pipe to
// server/logtee.py (or llama-swap / journald); a slow reader (a tee swapped out under host-memory pressure, a
// rate-limited journal) fills the 64 KB pipe, and the next fprintf blocks the thread that prints. Inside a step that
// thread is a rank's launch thread (in TP rank 0's is the scheduler thread, and most log lines are rank 0's), so the
// other GPUs wait in a collective for it and the watchdog ends the process after QE_COLL_TIMEOUT_MS.
//
// install() points the C `stdout` / `stderr` streams at a memory buffer each (glibc fopencookie; `stdout` / `stderr`
// are assignable in glibc) and starts one thread that writes the buffers to the real fd 1 / fd 2 as fast as the reader
// takes them. A printing thread only appends under a mutex. While the reader keeps up, every line reaches the pipe
// within microseconds (a crash loses nothing: the bytes are in the pipe, as logtee wants); when it stalls, up to
// QE_LOG_BUF_MB (64) per stream wait in memory, more is dropped and counted ("[qengine] log: N bytes dropped").
// The fatal paths (watchdog.cu fatal_exit, the clear stops before _exit / execve, the server's os._exit through
// qe_log_drain) call drain() before they exit; drain gives up after ~5 s on a reader that takes nothing (review L1).
// A crash signal (SIGSEGV / SIGBUS / SIGFPE / SIGILL / SIGABRT: std::terminate, abort, assert) writes what is pending
// first, best effort, then the handler installed before (Python's faulthandler) runs (review L2). Code writing to
// fd 1 / fd 2 directly (Python's sys.stderr, faulthandler) is not affected. QE_LOG_ASYNC=0: the plain streams (off-switch).
#include <limits.h>
#include <poll.h>
#include <pthread.h>
#include <signal.h>
#include <unistd.h>
#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>

namespace qe_log {

struct Sink {
    int fd = -1;
    std::mutex mu;       // pending, dropped
    std::mutex wmu;      // one writer of fd at a time (the forwarder or drain)
    std::string pending;
    size_t cap = 64u << 20, dropped = 0;
};
struct State {
    Sink out, err;
    std::mutex mu;
    std::condition_variable cv;
    bool quit = false, wake = false;
    std::atomic<bool> forked{false};  // a fork's child: no forwarder thread there, write through
    std::atomic<bool> sync{false};    // after a drain (on the way out): later lines are written at once, bounded (re-review 3 L4)
    std::thread th;
};
inline State*& state() {
    static State* s = nullptr;
    return s;
}

inline void write_all(int fd, const char* p, size_t n) {
    while (n) {
        const ssize_t w = ::write(fd, p, n);
        if (w < 0) {
            if (errno == EINTR) continue;
            return;  // (a closed reader: nothing to do; the old stream would have failed the same way)
        }
        p += w;
        n -= (size_t)w;
    }
}

// as write_all, but gives up at `until` (a reader that takes nothing): small writes after POLLOUT, so a write
// of a blocking pipe cannot block for long
inline bool write_until(int fd, const char* p, size_t n, std::chrono::steady_clock::time_point until) {
    while (n) {
        const auto left = std::chrono::duration_cast<std::chrono::milliseconds>(until - std::chrono::steady_clock::now()).count();
        if (left <= 0) return false;
        pollfd pf{fd, POLLOUT, 0};
        const int r = ::poll(&pf, 1, (int)std::min<long long>(left, 1000));
        if (r < 0 && errno == EINTR) continue;
        if (r < 0 || (pf.revents & (POLLERR | POLLNVAL))) return false;
        if (r == 0) continue;
        const ssize_t w = ::write(fd, p, std::min<size_t>(n, PIPE_BUF));
        if (w < 0) {
            if (errno == EINTR || errno == EAGAIN) continue;
            return false;
        }
        p += w;
        n -= (size_t)w;
    }
    return true;
}

// the forwarder's turn for one sink: what is pending, written outside the buffer's mutex
inline bool flush_sink(Sink& k) {
    std::lock_guard<std::mutex> w(k.wmu);
    std::string chunk;
    size_t dropped = 0;
    {
        std::lock_guard<std::mutex> l(k.mu);
        chunk.swap(k.pending);
        dropped = k.dropped;
        k.dropped = 0;
    }
    if (chunk.empty() && !dropped) return false;
    write_all(k.fd, chunk.data(), chunk.size());
    if (dropped) {
        char line[160];
        const int n = snprintf(line, sizeof line, "[qengine] log: %zu bytes dropped (the log reader did not keep up; QE_LOG_BUF_MB)\n", dropped);
        write_all(k.fd, line, (size_t)n);
    }
    return true;
}

inline ssize_t cookie_write(void* c, const char* buf, size_t n) {
    Sink& k = *(Sink*)c;
    State* s = state();
    if (s->forked.load(std::memory_order_relaxed)) {  // a child after fork(): no forwarder, write through
        write_all(k.fd, buf, n);
        return (ssize_t)n;
    }
    if (s->sync.load(std::memory_order_relaxed)) {  // (atexit handlers after ours, static destructors, the final flush)
        write_until(k.fd, buf, n, std::chrono::steady_clock::now() + std::chrono::seconds(1));
        return (ssize_t)n;
    }
    {
        std::lock_guard<std::mutex> l(k.mu);
        if (k.pending.size() + n > k.cap) k.dropped += n;
        else k.pending.append(buf, n);
    }
    { std::lock_guard<std::mutex> l(s->mu); s->wake = true; }  // (under the mutex: no lost wake-up)
    s->cv.notify_one();
    return (ssize_t)n;
}

// writes everything pending now, from the calling thread (fatal paths before _exit / abort / execve). At most ~5 s in
// all (review L1): the wait for a forwarder write in progress and the writes themselves; a forwarder still writing
// then means a reader that takes nothing, and the rest is given up (not written next to it: no interleaved bytes)
inline void drain() {
    State* s = state();
    if (!s) return;
    fflush(stdout);
    fflush(stderr);
    if (s->forked.load()) return;  // (a fork's child writes through; its copy of the parent's buffer is not its own)
    const auto until = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    for (Sink* k : {&s->out, &s->err}) {
        bool got = false;
        while (!(got = k->wmu.try_lock()) && std::chrono::steady_clock::now() < until) std::this_thread::sleep_for(std::chrono::milliseconds(10));
        if (!got) continue;
        std::string chunk;
        {
            std::lock_guard<std::mutex> l(k->mu);
            chunk.swap(k->pending);
        }
        write_until(k->fd, chunk.data(), chunk.size(), until);
        k->wmu.unlock();
    }
    s->sync.store(true);
}

// a crash signal: what is pending, best effort (try-locks only, bounded writes), then the previous handler
inline struct sigaction* old_actions() {
    static struct sigaction a[32];
    return a;
}
inline void on_crash(int sig) {
    if (State* s = state(); s && !s->forked.load()) {
        const auto until = std::chrono::steady_clock::now() + std::chrono::seconds(2);
        for (Sink* k : {&s->out, &s->err}) {
            if (!k->mu.try_lock()) continue;
            write_until(k->fd, k->pending.data(), k->pending.size(), until);
            k->pending.clear();
            k->mu.unlock();
        }
    }
    sigaction(sig, &old_actions()[sig], nullptr);  // the handler before ours (faulthandler) or the default
    raise(sig);                                    // (blocked until this handler returns; a fault re-runs and lands there)
}
inline void install_crash_handlers() {
    for (int sig : {SIGSEGV, SIGBUS, SIGFPE, SIGILL, SIGABRT}) {
        struct sigaction a {};
        a.sa_handler = on_crash;
        sigemptyset(&a.sa_mask);
        a.sa_flags = SA_ONSTACK;
        sigaction(sig, &a, &old_actions()[sig]);
    }
}

// after a drain that did not end the process (a failed execve): buffered writes again
inline void resume() {
    if (State* s = state()) s->sync.store(false);
}

inline FILE* open_sink(Sink& k, int fd, size_t cap) {
    k.fd = fd;
    k.cap = cap;
    cookie_io_functions_t io{};
    io.write = cookie_write;
    FILE* f = fopencookie(&k, "w", io);
    if (f) setvbuf(f, nullptr, _IOLBF, 1 << 16);  // whole lines per append (as the line-buffered pipe stream did)
    return f;
}

// once per process (both engine libraries may call it: the first one wins, the streams then have no fd)
inline void install() {
    static std::once_flag once;
    std::call_once(once, [] {
        const char* v = getenv("QE_LOG_ASYNC");
        if (v && atoi(v) == 0) return;
        if (fileno(stderr) < 0 || fileno(stdout) < 0) return;  // already a buffered stream (another library's install)
        const char* mb = getenv("QE_LOG_BUF_MB");
        const size_t cap = (size_t)(mb && atoi(mb) > 0 ? atoi(mb) : 64) << 20;
        State* s = new State;  // (never freed: the streams live to the end of the process)
        state() = s;
        fflush(stdout);
        fflush(stderr);
        FILE* o = open_sink(s->out, fileno(stdout), cap);
        FILE* e = open_sink(s->err, fileno(stderr), cap);
        if (!o || !e) {
            if (o) fclose(o);
            if (e) fclose(e);
            state() = nullptr;
            delete s;
            return;
        }
        setvbuf(e, nullptr, _IONBF, 0);  // stderr: every call goes to the buffer at once (as before: unbuffered)
        pthread_atfork(nullptr, nullptr, [] { if (State* st = state()) st->forked.store(true); });
        s->th = std::thread([s] {
            std::unique_lock<std::mutex> l(s->mu);
            while (!s->quit) {
                s->wake = false;
                l.unlock();
                flush_sink(s->out);
                flush_sink(s->err);
                l.lock();
                s->cv.wait(l, [s] { return s->wake || s->quit; });
            }
        });
        s->th.detach();
        stdout = o;
        stderr = e;
        atexit([] { drain(); });
        install_crash_handlers();
    });
}

}  // namespace qe_log
