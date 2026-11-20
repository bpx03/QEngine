// CPU unit test of src/drv_gate.h (issue #32) and src/qe_log.h: registrations never overlap a step, steps and
// registrations take turns, a cancelled registration gives up, nesting is free; the log streams never block.
#include "drv_gate.h"
#include "qe_log.h"
#include <fcntl.h>
#include <cassert>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

static std::atomic<int> g_slow{0};
void qe_wd_slow(int d) { g_slow += d; }

#define EXPECT(c) do { if (!(c)) { fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); qe_log::drain(); _exit(1); } } while (0)

static void gate_tests() {
    using namespace std::chrono;
    std::atomic<int> in_step{0}, in_reg{0}, overlap{0};
    std::atomic<bool> stop{false};
    std::atomic<long> steps{0}, regs{0};
    // a pinner registering piece after piece and a step loop: never both at once, both make progress
    std::thread pin([&] {
        while (!stop) {
            qe_drv::Reg g;
            in_reg++;
            if (in_step) overlap++;
            std::this_thread::sleep_for(milliseconds(3));
            in_reg--;
            regs++;
        }
    });
    std::thread step([&] {
        while (!stop) {
            qe_drv::Step g;
            in_step++;
            if (in_reg) overlap++;
            std::this_thread::sleep_for(milliseconds(1));
            in_step--;
            steps++;
        }
    });
    std::this_thread::sleep_for(milliseconds(600));
    stop = true;
    pin.join();
    step.join();
    printf("gate: %ld steps, %ld registrations, %d overlaps\n", steps.load(), regs.load(), overlap.load());
    EXPECT(overlap == 0);
    EXPECT(steps > 50 && regs > 50);  // turns: neither side starves
    // a cancelled registration gives up while a step runs
    {
        std::atomic<bool> cancel{false};
        std::atomic<int> res{-1};
        qe_drv::Step* s = new qe_drv::Step;
        std::thread t([&] { qe_drv::Reg g(&cancel); res = g.ok() ? 1 : 0; });
        std::this_thread::sleep_for(milliseconds(50));
        EXPECT(res == -1);  // waits for the step
        cancel = true;
        t.join();
        EXPECT(res == 0);
        delete s;
    }
    // nesting: a Reg inside a Step on one thread runs at once and counts as slow for the watchdog
    {
        qe_drv::Step s;
        {
            qe_drv::Step s2;  // nested step: free
            qe_drv::Reg g;
            EXPECT(g.ok() && g_slow == 1);
        }
        EXPECT(g_slow == 0);
    }
    // a Reg after a step ended takes the gate at once
    {
        const auto t0 = steady_clock::now();
        qe_drv::Reg g;
        EXPECT(g.ok() && steady_clock::now() - t0 < milliseconds(50));
    }
    // hold (C API qe_drv_gate, the vision encoder): waits for the running step, keeps the next step out until released;
    // a repeated hold / release is ignored
    {
        qe_drv::Step* s = new qe_drv::Step;
        std::atomic<int> held{0}, stepped{0};
        std::thread v([&] {
            qe_drv::hold(true);
            qe_drv::hold(true);  // (ignored)
            held = 1;
            std::this_thread::sleep_for(milliseconds(30));
            EXPECT(stepped == 0);  // the next step waits for the release
            qe_drv::hold(false);
            qe_drv::hold(false);  // (ignored)
        });
        std::this_thread::sleep_for(milliseconds(30));
        EXPECT(held == 0);  // waits for the running step
        delete s;
        while (!held) std::this_thread::yield();
        std::thread n([&] { qe_drv::Step g; stepped = 1; });
        n.join();
        v.join();
        EXPECT(stepped == 1 && g_slow == 0);
        const auto t0 = steady_clock::now();
        qe_drv::Reg g;  // released: the gate is free again
        EXPECT(g.ok() && steady_clock::now() - t0 < milliseconds(50));
    }
    // mode B (async steps): steps are not gated, a Reg only marks the call slow
    qe_drv::async_steps().fetch_add(1);
    {
        qe_drv::Step s;
        std::atomic<int> res{-1};
        std::thread t([&] { qe_drv::Reg g; res = g_slow.load(); });
        t.join();
        EXPECT(res == 1 && g_slow == 0);
    }
    // mode B, a background piece (review L3): no mark at once; the late mark after min(2 s, QE_COLL_TIMEOUT_MS / 4) = 50 ms
    {
        qe_drv::Step s;  // (a stage task on this thread, the pinner on its own)
        std::atomic<int> at0{-1}, at150{-1};
        std::thread t([&] {
            qe_drv::Reg g(nullptr, 20, 2000);
            at0 = g.ok() ? g_slow.load() : -2;
            std::this_thread::sleep_for(milliseconds(150));
            at150 = g_slow.load();
        });
        t.join();
        EXPECT(at0 == 0 && at150 == 1);
    }
    EXPECT(g_slow == 0);
    qe_drv::async_steps().fetch_sub(1);
    // a gated registration that runs long gets the late mark too (review L4: pp_prefill stages run outside the steps)
    {
        qe_drv::Reg g;
        EXPECT(g.ok() && g_slow == 0);
        std::this_thread::sleep_for(milliseconds(150));
        EXPECT(g_slow == 1);
    }
    EXPECT(g_slow == 0);
    {  // a short one never does
        qe_drv::Reg g;
        std::this_thread::sleep_for(milliseconds(5));
    }
    std::this_thread::sleep_for(milliseconds(100));
    EXPECT(g_slow == 0);
}

static void log_tests() {
    // stderr / stdout on a pipe nobody reads: 10,000 lines must not block (the pipe holds 64 KB)
    fflush(stdout);
    int pe[2], po[2];
    EXPECT(pipe(pe) == 0 && pipe(po) == 0);
    const int save_err = dup(2), save_out = dup(1);
    dup2(pe[1], 2);
    dup2(po[1], 1);
    setenv("QE_LOG_BUF_MB", "1", 1);
    qe_log::install();
    EXPECT(fileno(stderr) < 0);
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < 10000; i++) fprintf(stderr, "line %05d of the engine log, padded to about sixty bytes......\n", i);
    for (int i = 0; i < 100; i++) printf("out %d\n", i);
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    // read it all back: in order, the excess past 1 MB counted as dropped
    fcntl(pe[0], F_SETFL, O_NONBLOCK);
    fcntl(po[0], F_SETFL, O_NONBLOCK);
    std::string got, out;
    char buf[1 << 16];
    for (int idle = 0; idle < 50;) {
        ssize_t n = read(pe[0], buf, sizeof buf);
        if (n > 0) { got.append(buf, n); idle = 0; }
        else { idle++; std::this_thread::sleep_for(std::chrono::milliseconds(10)); }
        ssize_t k = read(po[0], buf, sizeof buf);
        if (k > 0) out.append(buf, k);
    }
    dup2(save_err, 2);
    dup2(save_out, 1);
    char line[200];
    snprintf(line, sizeof line, "log: 10,000 lines in %.1f ms, %zu bytes back, stdout %zu bytes\n", ms, got.size(), out.size());
    qe_log::write_all(2, line, strlen(line));
    EXPECT(ms < 2000);
    EXPECT(got.find("line 00000") == 0);
    EXPECT(got.find("line 09999") != std::string::npos);  // ~620 KB < 1 MB: nothing dropped
    EXPECT(got.find("dropped") == std::string::npos);
    EXPECT(out.find("out 0\n") == 0 && out.find("out 99\n") != std::string::npos);
    size_t last = 0;
    for (int i = 0; i < 10000; i += 997) {  // in order
        char k[16];
        snprintf(k, sizeof k, "line %05d", i);
        const size_t p = got.find(k);
        EXPECT(p != std::string::npos && p >= last);
        last = p;
    }
}

// review L1: a reader that takes nothing bounds write_until and drain()
static void drain_tests() {
    using namespace std::chrono;
    int p[2];
    EXPECT(pipe(p) == 0);
    fcntl(p[1], F_SETFL, O_NONBLOCK);
    char fill[4096];
    memset(fill, 'x', sizeof fill);
    while (write(p[1], fill, sizeof fill) > 0) {}
    fcntl(p[1], F_SETFL, 0);  // (blocking again, as the server's pipe)
    auto t0 = steady_clock::now();
    EXPECT(!qe_log::write_until(p[1], fill, sizeof fill, t0 + milliseconds(200)));
    EXPECT(steady_clock::now() - t0 < milliseconds(1500));
    // drain with fd 2 on the full pipe: the forwarder blocks in its write, drain gives up after ~5 s
    const int save_err = dup(2);
    dup2(p[1], 2);
    fprintf(stderr, "a line the stuck reader never takes\n");
    std::this_thread::sleep_for(milliseconds(50));
    t0 = steady_clock::now();
    qe_log::drain();
    const double s = duration<double>(steady_clock::now() - t0).count();
    fcntl(p[0], F_SETFL, O_NONBLOCK);  // the reader comes back: the forwarder's write ends
    char buf[1 << 16];
    for (int i = 0; i < 50; i++) {
        while (read(p[0], buf, sizeof buf) > 0) {}
        std::this_thread::sleep_for(milliseconds(10));
    }
    dup2(save_err, 2);
    char line[120];
    snprintf(line, sizeof line, "drain: gave up after %.1f s on a stuck reader\n", s);
    qe_log::write_all(2, line, strlen(line));
    EXPECT(s > 4.0 && s < 7.0);
    // after a drain (on the way out) a line is written at once, not buffered (re-review 3 L4)
    int q[2];
    EXPECT(pipe(q) == 0);
    dup2(q[1], 2);
    fprintf(stderr, "after the drain\n");
    dup2(save_err, 2);
    fcntl(q[0], F_SETFL, O_NONBLOCK);
    const ssize_t n = read(q[0], buf, sizeof buf);
    EXPECT(n > 0 && std::string(buf, n).find("after the drain") == 0);
}

int main() {
    setenv("QE_COLL_TIMEOUT_MS", "200", 1);  // (late mark after 50 ms)
    gate_tests();
    log_tests();
    drain_tests();
    qe_log::write_all(2, "drv_gate_unit: PASS\n", 20);
    return 0;
}
