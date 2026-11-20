#pragma once
// Host-memory registration vs step launches (issue #32). cudaHostRegister / cudaHostUnregister of large ranges (and
// malloc_trim, which holds the allocator's locks while it walks the heap) can take long and hold the driver's locks;
// a rank whose thread then waits to launch its part of a step is late, and the GPUs already in that step's
// collectives wait for it (watchdog: QE_COLL_TIMEOUT_MS). The gate keeps them apart:
//   Step  (shared): one engine job (FE::run in TP, a stage task in mode B) from its first launch to its return;
//   Reg   (exclusive): one registration piece of a background pinner, an unregister, a malloc_trim.
// A Reg waits until no Step runs and keeps new Steps out while it runs; Steps and Regs take turns (a waiting Step
// goes before the next Reg, so a pinner that pins piece after piece cannot hold the steps back for long). Nested
// use on one thread is free (a Reg inside a Step only marks the call slow for the watchdog: QeWdSlow).
// While stage threads run steps on their own (mode B: async_steps() > 0) Steps are not gated (a stage task may wait on
// the host for another stage's task: holding one back could deadlock the pipeline) and a GPU can wait across stage
// tasks, so there a Reg only counts as a slow driver call for the device watchdog (QeWdSlow), for one piece.
// A background piece (the archive pinner) is not marked at once there: a registration that runs longer than
// min(2 s, QE_COLL_TIMEOUT_MS / 4) gets the mark then (a timer thread), so the normal sub-second pieces leave the device
// watchdog armed at boot (review L3) and a stalled one (issue #32) lets the waiting GPUs' clocks restart. Every gated
// Reg gets the same late mark: the FLASH_PAR=pp_prefill pipeline drives its stages outside the gated TP steps (review L4).
#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <mutex>
#include <thread>

void qe_wd_slow(int delta);  // kernels/watchdog.cu (QeWdHost::slow += delta)

namespace qe_drv {

struct State {
    std::mutex m;
    std::condition_variable cv;
    int steps = 0, steps_waiting = 0, regs_waiting = 0;
    std::chrono::steady_clock::time_point last_step{};  // the end of the last Step (idle gaps for a background Reg)
    bool reg = false, step_turn = false;
    std::atomic<int> async{0};
    // the late mark (see above): running timed Regs, the start of the oldest, marked yet; the timer thread's wake-up
    int tm_n = 0;
    bool tm_marked = false, tm_on = false;
    std::chrono::steady_clock::time_point tm_t0{};
    std::condition_variable tcv;
};
inline State& st() {
    static State* s = new State;  // (never freed: the timer thread may wait on it to the end of the process)
    return *s;
}
inline int late_ms() {
    static const int v = [] {
        const char* e = getenv("QE_COLL_TIMEOUT_MS");
        const int c = e && atoi(e) > 0 ? atoi(e) : 20000;
        return std::max(1, std::min(2000, c / 4));
    }();
    return v;
}
// under s.m: one more timed Reg running (the timer thread starts with the first one)
inline void timed_begin(State& s) {
    if (s.tm_n++ == 0) s.tm_t0 = std::chrono::steady_clock::now();
    if (!s.tm_on) {
        s.tm_on = true;
        std::thread([&s] {
            std::unique_lock<std::mutex> l(s.m);
            for (;;) {
                s.tcv.wait(l, [&] { return s.tm_n > 0 && !s.tm_marked; });
                const auto due = s.tm_t0 + std::chrono::milliseconds(late_ms());
                s.tcv.wait_until(l, due, [&] { return s.tm_n == 0 || s.tm_marked; });
                if (s.tm_n > 0 && !s.tm_marked && std::chrono::steady_clock::now() >= s.tm_t0 + std::chrono::milliseconds(late_ms())) {
                    s.tm_marked = true;
                    qe_wd_slow(1);
                }
            }
        }).detach();
    }
    s.tcv.notify_all();
}
inline void timed_end(State& s) {
    if (--s.tm_n == 0 && s.tm_marked) {
        s.tm_marked = false;
        qe_wd_slow(-1);
    }
    s.tcv.notify_all();
}
inline thread_local int tl_step = 0, tl_reg = 0;
inline std::atomic<int>& async_steps() { return st().async; }

struct Step {
    Step() {
        if (tl_step++ || tl_reg) return;
        State& s = st();
        if (s.async.load()) return;  // (mode B: not gated)
        gated_ = true;
        std::unique_lock<std::mutex> l(s.m);
        s.steps_waiting++;
        s.cv.wait(l, [&] { return !s.reg && (!s.regs_waiting || s.step_turn); });
        s.steps_waiting--;
        s.steps++;
        if (!s.steps_waiting) s.step_turn = false;
    }
    ~Step() {
        --tl_step;
        if (!gated_) return;
        State& s = st();
        {
            std::lock_guard<std::mutex> l(s.m);
            s.steps--;
            s.last_step = std::chrono::steady_clock::now();
        }
        s.cv.notify_all();
    }
    Step(const Step&) = delete;
    Step& operator=(const Step&) = delete;

private:
    bool gated_ = false;
};

// cancel (optional): gives up while waiting once *cancel is true (ok() false). idle_ms > 0 (a background pinner): waits
// for a gap of idle_ms without a step, at most max_wait_ms, then takes its turn anyway (decode steps follow each other
// with microsecond gaps: a registration then delays one step)
class Reg {
public:
    explicit Reg(const std::atomic<bool>* cancel = nullptr, int idle_ms = 0, int max_wait_ms = 0) {
        if (tl_reg++) return;
        if (tl_step) {  // inside a step on this thread: the call runs now, the peers' watchdogs wait for it
            slow_ = true;
            qe_wd_slow(1);
            return;
        }
        State& s = st();
        if (s.async.load()) {  // mode B: runs now, the waiting GPUs' clocks restart meanwhile
            if (idle_ms > 0) {  // (a background piece: the late mark only, review L3)
                std::lock_guard<std::mutex> l(s.m);
                timed_ = true;
                timed_begin(s);
                return;
            }
            slow_ = true;
            qe_wd_slow(1);
            return;
        }
        std::unique_lock<std::mutex> l(s.m);
        if (idle_ms > 0) {  // (not counted as waiting yet: the steps run on meanwhile)
            const auto t_end = std::chrono::steady_clock::now() + std::chrono::milliseconds(max_wait_ms);
            while (std::chrono::steady_clock::now() < t_end && !(cancel && cancel->load())) {
                const auto now = std::chrono::steady_clock::now();
                if (!s.steps && !s.reg && now - s.last_step >= std::chrono::milliseconds(idle_ms)) break;
                s.cv.wait_for(l, std::chrono::milliseconds(std::max(1, idle_ms / 4)));
            }
        }
        s.regs_waiting++;
        auto ready = [&] { return !s.reg && !s.steps && !(s.step_turn && s.steps_waiting); };
        if (cancel) {
            while (!ready() && !cancel->load()) s.cv.wait_for(l, std::chrono::milliseconds(2));
        } else {
            s.cv.wait(l, ready);
        }
        s.regs_waiting--;
        if (!ready()) {  // cancelled
            ok_ = false;
            l.unlock();
            s.cv.notify_all();
            return;
        }
        s.reg = held_ = timed_ = true;
        timed_begin(s);
    }
    ~Reg() {
        --tl_reg;
        if (slow_) qe_wd_slow(-1);
        if (!held_ && !timed_) return;
        State& s = st();
        if (!held_) {
            std::lock_guard<std::mutex> l(s.m);
            timed_end(s);
            return;
        }
        {
            std::lock_guard<std::mutex> l(s.m);
            timed_end(s);
            s.reg = false;
            if (s.steps_waiting) s.step_turn = true;
        }
        s.cv.notify_all();
    }
    bool ok() const { return ok_; }
    Reg(const Reg&) = delete;
    Reg& operator=(const Reg&) = delete;

private:
    bool ok_ = true, held_ = false, slow_ = false, timed_ = false;
};

// A Reg held across two calls on one thread (C API qe_drv_gate): the server's vision encoder on an engine card. With
// --vision torch starts CUDA first, so module loading is LAZY and torch / cuBLASLt load a kernel's module at its first
// launch (a new image shape): that load can synchronize the context under the driver's lock while this engine's
// collectives spin on the card, and the worker's next launch then waits behind it: a cross-card hang.
// hold(true) waits like a Reg (no step running, new steps kept out), hold(false) gives it back; a repeated call is ignored.
inline void hold(bool on) {
    static thread_local Reg* r = nullptr;
    if (on && !r) r = new Reg;
    else if (!on && r) {
        delete r;
        r = nullptr;
    }
}

}  // namespace qe_drv
