// [qfn-pf-bgap] host inputs of the next prefill batch built while the current batch runs on the GPUs.
//
// Between two prefill batches the host used to build the next batch's n-gram rows (PLE input: ~21 MB for 4096 rows,
// 8 freshly spawned threads), and every rank copied them into its own pinned staging buffer before its graph launch,
// all serial with idle GPUs. Here:
//  - NgPool: persistent workers for the row building (no thread spawns per batch);
//  - RowsAhead: two pinned (portable) row buffers; while batch i reads buffer b from the GPUs, a persistent helper
//    thread builds batch i + 1's rows into buffer b ^ 1. The consumer checks that the rows were built from exactly the
//    tokens it would use (same positions, same token window), else it builds them itself: the rows are the same bytes
//    either way. The ranks copy straight from the pinned buffer (Worker::prefill_batch's ple_pinned).
// FLASH_PF_AHEAD=0: no prefetch (rows built at the batch boundary and staged per rank as before; the persistent pool
// stays).
#pragma once
#include <cuda_runtime.h>
#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace flash_ng {

// persistent workers: run(nt, part) calls part(0..nt-1), part(0) on the caller, and returns when all are done.
// Callers are serialized (one batch of parts at a time).
class NgPool {
public:
    static NgPool& get() {  // (never destroyed: a helper still building at process exit finds it alive)
        static NgPool* p = new NgPool;
        return *p;
    }
    void run(int nt, const std::function<void(int)>& part) {
        if (nt <= 1) { part(0); return; }
        std::lock_guard<std::mutex> call(call_mu_);
        {
            std::unique_lock<std::mutex> l(mu_);
            while ((int)th_.size() < nt - 1) {
                const int id = (int)th_.size();
                th_.emplace_back([this, id] { loop(id + 1); });
            }
            fn_ = &part;
            nt_ = nt;
            left_ = nt - 1;
            gen_++;
        }
        cv_.notify_all();
        part(0);
        std::unique_lock<std::mutex> l(mu_);
        done_cv_.wait(l, [&] { return left_ == 0; });
        fn_ = nullptr;
    }
    ~NgPool() {
        {
            std::lock_guard<std::mutex> l(mu_);
            stop_ = true;
        }
        cv_.notify_all();
        for (auto& t : th_) t.join();
    }

private:
    void loop(int k) {
        unsigned long long seen = 0;
        for (;;) {
            const std::function<void(int)>* f = nullptr;
            {
                std::unique_lock<std::mutex> l(mu_);
                cv_.wait(l, [&] { return stop_ || gen_ != seen; });
                if (gen_ != seen) {  // (a pending part runs before stop_ is honoured: run() waits for it)
                    seen = gen_;
                    if (k < nt_) f = fn_;
                }
                if (!f) {
                    if (stop_) return;
                    continue;
                }
            }
            (*f)(k);
            std::lock_guard<std::mutex> l(mu_);
            if (--left_ == 0) done_cv_.notify_all();
        }
    }
    std::mutex call_mu_, mu_;
    std::condition_variable cv_, done_cv_;
    std::vector<std::thread> th_;
    const std::function<void(int)>* fn_ = nullptr;
    int nt_ = 0, left_ = 0;
    unsigned long long gen_ = 0;
    bool stop_ = false;
};

// rows t0 .. t0 + m - 1 of h, `row(h, t, out)` per row, over the persistent pool (8 parts from 256 rows, as before).
// cancel (optional): checked every 64 rows; a cancelled build leaves the rows incomplete (the caller drops them).
template <class RowFn>
inline void rows_par(const std::vector<int>& h, int t0, int m, unsigned short* out, size_t row_elems, RowFn row,
                     const std::atomic<bool>* cancel = nullptr) {
    const int nt = m >= 256 ? 8 : 1;
    auto part = [&](int k) {
        const int j1 = (int)((long)m * (k + 1) / nt);
        for (int j = (int)((long)m * k / nt); j < j1; j++) {
            if (cancel && (j & 63) == 0 && cancel->load(std::memory_order_relaxed)) return;
            row(h, t0 + j, out + (size_t)j * row_elems);
        }
    };
    if (nt == 1) { part(0); return; }
    NgPool::get().run(nt, part);
}

// The token window a batch's rows depend on: positions [s0, t0 + m) of the history (ngram_row reads h[t], h[t - 1],
// h[t - 2]; s0 = max(0, t0 - 2)).
struct RowsKey {
    int t0 = -1, m = 0, s0 = 0;
    std::vector<int> win;  // h[s0 .. t0 + m)
    bool same(int t0_, int m_, const std::vector<int>& h) const {
        if (t0_ != t0 || m_ != m || (int)h.size() < t0 + m) return false;
        const int s = std::max(0, t0 - 2);
        if (s != s0 || (int)win.size() != t0 + m - s0) return false;
        return std::equal(win.begin(), win.end(), h.begin() + s0);
    }
};

class RowsAhead {
public:
    // build(win, t0 - s0, m, out, cancel): the rows of window positions [t0 - s0, t0 - s0 + m) into out (H bf16 per row)
    using Build = std::function<void(const std::vector<int>& win, int t, int m, unsigned short* out, const std::atomic<bool>* cancel)>;
    RowsAhead(size_t row_bytes, Build b) : row_bytes_(row_bytes), build_(std::move(b)) {
        NgPool::get();  // (constructed first: the pool outlives this object at static destruction)
    }
    ~RowsAhead() {
        stop();
        for (int b = 0; b < 2; b++) release(b);  // (errors ignored: at process exit the runtime may be gone)
    }
    static bool enabled() {
        static const bool on = [] {
            const char* v = getenv("FLASH_PF_AHEAD");
            return !(v && *v == '0');
        }();
        return on;
    }
    void stop() {
        cancel_.store(true);
        {
            std::lock_guard<std::mutex> l(mu_);
            quit_ = true;
        }
        cv_.notify_all();
        if (th_.joinable()) th_.join();
    }
    // the rows of h[t0 .. t0 + m) for the batch about to run: the prepared buffer when it was built from the same
    // token window, else built now (h = the history the caller would build them from, t0 + m <= h.size()); a build in
    // flight for another window is cancelled, not waited for. The pointer stays valid until the second next take().
    // nullptr: no host memory for the rows (the caller builds them its old way).
    const unsigned short* take(int t0, int m, const std::vector<int>& h, bool* hit = nullptr) {
        {
            std::unique_lock<std::mutex> l(mu_);
            if (busy_ && !key_[job_].same(t0, m, h)) cancel_.store(true);
            done_cv_.wait(l, [&] { return !busy_; });
            cancel_.store(false);
        }
        int b = -1;
        if (ready_ >= 0 && key_[ready_].same(t0, m, h)) b = ready_;
        if (hit) *hit = b >= 0;
        if (b < 0) {
            b = cur_ < 0 ? 0 : cur_ ^ 1;  // (both buffers are free here: the batches that read them are done)
            if (!ensure(b, (size_t)m * row_bytes_)) { ready_ = -1; return nullptr; }
            fill_key(key_[b], t0, m, h);
            build_(key_[b].win, t0 - key_[b].s0, m, buf(b), nullptr);
        }
        cur_ = b;
        ready_ = -1;
        return buf(b);
    }
    // start building the rows of the batch after the one take() returned last (positions [t0, t0 + m), window
    // win = h[max(0, t0 - 2) .. t0 + m)) into the other buffer, on the helper thread; returns at once. Only into a
    // buffer that is large enough already (pinned memory is allocated on the consumer's thread, between batches).
    void ahead_win(int t0, int m, std::vector<int> win) {
        const int s0 = std::max(0, t0 - 2);
        if (m <= 0 || (int)win.size() != t0 + m - s0 || cur_ < 0) return;
        const int b = cur_ ^ 1;
        std::lock_guard<std::mutex> l(mu_);
        if (busy_ || quit_) return;  // (take() waited for the last build: not busy here)
        if (cap_[b] < (size_t)m * row_bytes_) {  // first larger batch: grow now (no GPU work of ours in flight)
            if (!ensure(b, (size_t)m * row_bytes_)) return;
        }
        key_[b].t0 = t0;
        key_[b].m = m;
        key_[b].s0 = s0;
        key_[b].win = std::move(win);
        if (!th_.joinable()) th_ = std::thread([this] { loop(); });
        job_ = b;
        busy_ = true;
        cv_.notify_all();
    }
    // both buffers sized for batches of `rows` rows now (at engine load: the first prompt does not pay the pinning);
    // pinned only (no pageable fallback here: a later take() tries again)
    void reserve(int rows) {
        std::lock_guard<std::mutex> l(mu_);
        if (busy_) return;
        for (int b = 0; b < 2; b++) ensure(b, (size_t)rows * row_bytes_, true);
    }
    // a build in flight is cancelled and waited for (another row builder is about to use the pool)
    void drop() {
        std::unique_lock<std::mutex> l(mu_);
        if (busy_) cancel_.store(true);
        done_cv_.wait(l, [&] { return !busy_; });
        cancel_.store(false);
        ready_ = -1;
    }
    bool host_pinned(const void* p) const {
        for (int b = 0; b < 2; b++)
            if (pinned_[b] && p == bufs_[b]) return true;
        return false;
    }

private:
    static void fill_key(RowsKey& k, int t0, int m, const std::vector<int>& h) {
        k.t0 = t0;
        k.m = m;
        k.s0 = std::max(0, t0 - 2);
        k.win.assign(h.begin() + k.s0, h.begin() + t0 + m);
    }
    unsigned short* buf(int b) { return (unsigned short*)bufs_[b]; }
    void release(int b) {
        if (bufs_[b]) {
            if (pinned_[b]) cudaFreeHost(bufs_[b]);
            else free(bufs_[b]);
        }
        bufs_[b] = nullptr;
        cap_[b] = 0;
        pinned_[b] = false;
    }
    // consumer thread only (between batches; the helper is idle)
    bool ensure(int b, size_t bytes, bool pinned_only = false) {
        if (cap_[b] >= bytes && (pinned_[b] || pinned_only == false)) return true;
        release(b);
        key_[b] = RowsKey();
        if (cudaHostAlloc(&bufs_[b], bytes, cudaHostAllocPortable) == cudaSuccess) pinned_[b] = true;
        else {
            cudaGetLastError();  // (host memory short: pageable rows, the ranks stage them as before)
            bufs_[b] = nullptr;
            if (pinned_only) return false;
            bufs_[b] = malloc(bytes);
            if (!bufs_[b]) return false;
        }
        cap_[b] = bytes;
        return true;
    }
    void loop() {
        for (;;) {
            int b;
            {
                std::unique_lock<std::mutex> l(mu_);
                cv_.wait(l, [&] { return quit_ || busy_; });
                if (quit_) {
                    busy_ = false;
                    done_cv_.notify_all();
                    return;
                }
                b = job_;
            }
            build_(key_[b].win, key_[b].t0 - key_[b].s0, key_[b].m, buf(b), &cancel_);
            {
                std::lock_guard<std::mutex> l(mu_);
                ready_ = cancel_.load() ? -1 : b;
                busy_ = false;
            }
            done_cv_.notify_all();
        }
    }
    size_t row_bytes_;
    Build build_;
    void* bufs_[2] = {nullptr, nullptr};
    size_t cap_[2] = {0, 0};
    bool pinned_[2] = {false, false};
    RowsKey key_[2];
    int cur_ = -1, ready_ = -1, job_ = 0;
    bool busy_ = false, quit_ = false;
    std::atomic<bool> cancel_{false};
    std::mutex mu_;
    std::condition_variable cv_, done_cv_;
    std::thread th_;
};

}  // namespace flash_ng
