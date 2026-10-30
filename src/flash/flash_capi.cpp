// C API of the Flash-Next engine for the server (server/qe_capi.h: tier 0, chunked prefill, cache, archive, monitor
// and the single functions). One sequence is resident on the GPUs; the other conversations wait in a host-RAM archive
// (KV segments + recurrent states, FLASH_KV_ARCHIVE_GB) and come back without a re-prefill; a new prompt starts from
// the longest cached prefix (the live sequence, an archived end state, or a checkpoint of the committed state at the
// system end / every FLASH_CKPT tokens / the prompt end - 1). Built as libqflash.so.
#include "../env.h"
#include <dirent.h>
#include <cuda_profiler_api.h>
#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <condition_variable>
#include <new>
#include <string>
#include <thread>
#include <future>
#include <vector>

#include "../json.h"
#include "flash.h"
#include "flash_steer.h"
#include "guard_rule.h"
#include "../host_mem.h"
#include "../gpu_guard.h"

using namespace flash;

namespace {
std::vector<std::string> split(const std::string& s, char c) {
    std::vector<std::string> v;
    std::string cur;
    for (char ch : s) {
        if (ch == c) { v.push_back(cur); cur.clear(); }
        else cur += ch;
    }
    if (!cur.empty()) v.push_back(cur);
    return v;
}

void ngram_row(const HostTables& ht, const std::vector<int>& h, int t, bf16* out) {
    // shifted token s is used iff no EOS sits in [t - s, t - 1] (the previous EOS ends the segment): only the two
    // previous positions decide (prefill-r4: the scan back to the last EOS was O(t) per token, O(n^2) per prompt:
    // 100K prompt 14 -> 60 ms of host time per 4K batch, serial with the GPU)
    // image rows (content ids >= kImgToken) hash as <|image_pad|>, like the HF / vLLM PLE input ids
    long long sh[3];
    sh[0] = dev_tok(h[t]);
    sh[1] = (t >= 1 && h[t - 1] != EOS) ? dev_tok(h[t - 1]) : EOS;
    sh[2] = (t >= 2 && h[t - 1] != EOS && h[t - 2] != EOS) ? dev_tok(h[t - 2]) : EOS;
    auto mul = [](long long a, long long b) { return (long long)((unsigned long long)a * (unsigned long long)b); };
    const long long big = mul(sh[0], ht.mult[0]) ^ mul(sh[1], ht.mult[1]);
    const long long tri = big ^ mul(sh[2], ht.mult[2]);
    for (int k = 0; k < PLE_HEADS; k++) {
        long long r = (k < 8 ? big : tri) % ht.vsz[k];
        if (r < 0) r += ht.vsz[k];
        const long long row = r + ht.voff[k];
        ht.row(row, out + k * PLE_DIM);
    }
}

// rows t0 .. t0 + m - 1 of the n-gram embedding (PLE input) in parallel: host reads of the n-gram table, serial with
// the GPU between prefill batches (prefill-r4)
static void ngram_rows(const HostTables& ht, const std::vector<int>& h, int t0, int m, bf16* out) {
    const int nt = m >= 256 ? 8 : 1;
    auto part = [&](int k) {
        for (int j = (int)((long)m * k / nt); j < (int)((long)m * (k + 1) / nt); j++) ngram_row(ht, h, t0 + j, out + (size_t)j * H);
    };
    if (nt == 1) { part(0); return; }
    std::vector<std::thread> th;
    for (int k = 1; k < nt; k++) th.emplace_back(part, k);
    part(0);
    for (auto& x : th) x.join();
}

// ---------------------------------------------------------------- host memory of the KV archive
// Pinned slabs (THP 2 MB pages: 4 KB pages slow every driver call), pinned in the background after
// startup up to the archive budget and carved first-fit with coalescing; an allocation that finds no pinned range
// falls back to pageable memory (slower copies, same result).
class HostPool {
public:
    ~HostPool() {
        { std::lock_guard<std::mutex> l(mu_); stop_ = true; }
        cv_.notify_all();
        if (th_.joinable()) th_.join();
    }
    // pins slabs in the background while less than `headroom` pinned bytes are free (on demand: a full budget of
    // 2 MB pages takes minutes to fault in next to a large page cache), up to `budget`, and only while the host
    // keeps its margin (host_mem.h: MemAvailable - margin >= two slabs; else it waits and retries)
    void start(size_t budget, size_t slab, size_t headroom) {
        if (!budget || !slab) return;
        th_ = std::thread([this, budget, slab, headroom] {
            double secs = 0;
            bool held = false;
            while (!stop_) {
                {
                    std::unique_lock<std::mutex> l(mu_);
                    cv_.wait(l, [&] { return stop_ || (pinned_ < budget && pinned_ < used_ + headroom); });
                    if (stop_) return;
                }
                const size_t sz = std::min(slab, budget - pinned_);
                if (qe_hostmem::headroom(true) < sz + slab) {  // the host is short of memory: not now
                    if (!held)
                        fprintf(stderr, "[qflash] archive pool: pinning held at %.1f GB (MemAvailable %.1f GB, margin %.1f GB)\n",
                                pinned_ / 1e9, qe_hostmem::available() / 1e9, qe_hostmem::margin() / 1e9);
                    held = true;
                    std::unique_lock<std::mutex> l(mu_);
                    cv_.wait_for(l, std::chrono::seconds(2), [&] { return stop_.load(); });
                    continue;
                }
                if (held) fprintf(stderr, "[qflash] archive pool: pinning resumed at %.1f GB\n", pinned_ / 1e9);
                held = false;
                const auto t0 = std::chrono::steady_clock::now();
                void* p = nullptr;
                if (posix_memalign(&p, 1 << 21, sz) != 0 || !p) break;
                static const bool thp = env_int("FLASH_ARCHIVE_THP", 1);
                if (thp) madvise(p, sz, MADV_HUGEPAGE);
                memset(p, 0, sz);  // fault the pages in without the driver
                if (stop_ || cudaHostRegister(p, sz, cudaHostRegisterPortable) != cudaSuccess) {
                    cudaGetLastError();
                    ::free(p);
                    if (stop_) break;
                    fprintf(stderr, "[qflash] archive pool: cudaHostRegister failed at %.1f GB pinned, retrying in 10 s\n", pinned_ / 1e9);
                    std::unique_lock<std::mutex> l(mu_);
                    cv_.wait_for(l, std::chrono::seconds(10), [&] { return stop_.load(); });
                    continue;
                }
                secs += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
                std::lock_guard<std::mutex> l(mu_);
                slabs_.push_back(Slab{(char*)p, sz, {{0, sz}}});
                pinned_ += sz;
                qe_hostmem::charged() += sz;
                if (env_int("FLASH_SLOT_LOG", 0))
                    fprintf(stderr, "[qflash] archive pool: %.1f GB pinned (%.1f s), %.2f GB used\n", pinned_ / 1e9, secs, used_ / 1e9);
            }
        });
    }
    char* alloc(size_t bytes, bool& pinned) {
        bytes = (bytes + 4095) / 4096 * 4096;
        {
            std::lock_guard<std::mutex> l(mu_);
            for (auto& sl : slabs_)
                for (auto it = sl.free.begin(); it != sl.free.end(); ++it)
                    if (it->second >= bytes) {
                        char* p = sl.base + it->first;
                        const size_t off = it->first, len = it->second;
                        sl.free.erase(it);
                        if (len > bytes) sl.free[off + bytes] = len - bytes;
                        used_ += bytes;
                        pused_ += bytes;
                        pinned = true;
                        cv_.notify_all();
                        return p;
                    }
            used_ += bytes;
        }
        cv_.notify_all();
        pinned = false;
        char* q = new (std::nothrow) char[bytes];
        if (q) qe_hostmem::charged() += bytes;
        return q;
    }
    void free(char* p, size_t bytes, bool pinned) {
        if (!p) return;
        bytes = (bytes + 4095) / 4096 * 4096;
        std::lock_guard<std::mutex> l(mu_);
        used_ -= bytes;
        if (!pinned) { delete[] p; qe_hostmem::charged() -= bytes; return; }
        pused_ -= bytes;
        for (auto& sl : slabs_) {
            if (p < sl.base || p >= sl.base + sl.size) continue;
            size_t off = p - sl.base, len = bytes;
            auto nx = sl.free.lower_bound(off);
            if (nx != sl.free.end() && nx->first == off + len) { len += nx->second; nx = sl.free.erase(nx); }
            if (nx != sl.free.begin()) {
                auto pv = std::prev(nx);
                if (pv->first + pv->second == off) { off = pv->first; len += pv->second; sl.free.erase(pv); }
            }
            sl.free[off] = len;
            return;
        }
    }
    // gives back to the host whole slabs nothing is carved from, up to `want` bytes (memory pressure); the
    // background pinning pins again once the host has room
    size_t release(size_t want) {
        std::lock_guard<std::mutex> l(mu_);
        size_t got = 0;
        for (size_t i = slabs_.size(); i-- > 0 && got < want;) {
            Slab& sl = slabs_[i];
            if (sl.free.size() != 1 || sl.free.begin()->first != 0 || sl.free.begin()->second != sl.size) continue;
            cudaHostUnregister(sl.base);
            cudaGetLastError();
            ::free(sl.base);
            got += sl.size;
            pinned_ -= sl.size;
            qe_hostmem::charged() -= sl.size;
            slabs_.erase(slabs_.begin() + i);
        }
        return got;
    }
    size_t used() const { std::lock_guard<std::mutex> l(mu_); return used_; }
    size_t pinned() const { std::lock_guard<std::mutex> l(mu_); return pinned_; }
    size_t pinned_free() const { std::lock_guard<std::mutex> l(mu_); return pinned_ - pused_; }
private:
    struct Slab { char* base; size_t size; std::map<size_t, size_t> free; };
    std::vector<Slab> slabs_;
    mutable std::mutex mu_;
    std::condition_variable cv_;
    std::thread th_;
    std::atomic<bool> stop_{false};
    size_t pinned_ = 0, used_ = 0, pused_ = 0;
};

// one host buffer per GPU: a KV segment or a recurrent state
struct HBlob {
    HostPool* pool = nullptr;
    std::vector<char*> p;
    std::vector<size_t> n;
    std::vector<char> pinned;
    size_t bytes = 0;
    HBlob() = default;
    HBlob(const HBlob&) = delete;
    ~HBlob() {
        for (size_t i = 0; i < p.size(); i++) pool->free(p[i], n[i], pinned[i]);
    }
    bool alloc(HostPool* pl, const std::vector<size_t>& sizes) {
        pool = pl;
        for (size_t s : sizes) {
            bool pin = false;
            char* q = pl->alloc(s, pin);
            p.push_back(q);
            n.push_back(s);
            pinned.push_back(pin);
            bytes += s;
            if (!q) return false;
        }
        return true;
    }
};
struct Seg : HBlob { int t0 = 0, t1 = 0; };          // KV of the positions [t0, t1)
struct State : HBlob {                               // recurrent state after the positions [0, pos)
    int pos = 0, prev_m = 0;
    bool pend = false;                               // with the deferred-commit rows of the last step (end state)
    int kind = 0;                                    // 0 end of a sequence, 1 hint (system end), 2 prompt end - 1, 3 periodic, 4 turn end
};
// A conversation known to the engine: resident in a sequence slot (its live tokens are the slot's hist) and/or
// archived in host memory.
struct Conv {
    int id = 0;
    int slot = -1;                            // resident in this slot (-1: archive only)
    std::vector<int> toks;                    // tokens of the archived state (resident: as of the last sync)
    std::vector<std::shared_ptr<Seg>> segs;   // archived KV, in order (later segments overwrite), covering [0, kv_n)
    int kv_n = 0;
    std::shared_ptr<State> end;               // state at toks.size() incl. the pending rows (null: none / dirty)
    std::vector<std::shared_ptr<State>> ck;   // committed states at positions <= kv_n (resident: <= live tokens)
    int prio = 1;                             // eviction value: 0 finished, 1 normal, 2 main thread
    unsigned long long used = 0, affinity = 0;
    long long steer = -1;                     // steering key it was computed with (FE::steer_key): KV / states never serve another
};

// A sequence slot: its recurrent state on the GPUs (Worker slot s) and a region of the KV pool in periods.
struct Slot {
    Conv* conv = nullptr;     // resident conversation (null: free)
    std::vector<int> hist;    // processed tokens
    int pending = -1;         // emitted, not yet processed
    int base = 0, cap = 0;    // KV region [base, base + cap) in periods (cap 0: none)
    bool active = false;      // a reply is being generated (qe_start* .. qe_release)
    bool dirty = false;       // a step launched ahead was dropped (see FE::drain)
    bool synced = false;      // the conversation's archive matches the device
    // preemption: a paused slot stays active (taken, generating) without a KV region: its rows [0, hist) are in its
    // conversation's archive segments (held: a resident conversation is never evicted), its recurrent state, pending
    // rows, speculative state and lookup history stay in the slot's device storage (outside the pool)
    bool paused = false;
    unsigned long long admit = 0, pause_t = 0;  // admission / pause order
    bool lk_stale = false;    // its lookup history on the devices is behind (batched steps ran)
    int bhist = -1;           // tokens of its batched-lookup history on the devices (-1: none / stale)
    int remain = -1;          // tokens the reply may still emit (qe_set_limit; -1: no limit)
    // stop sequences (qe_watch_stops / qe_trim): the last step's first row position and rows (-1: not undoable)
    int rb_pos = -1, rb_rows = 0;
    bool want_drafts = false;
    bool con = false;         // grammar-constrained (qe_constrain); its host buffers live in FE (mask_host / spec_io)
    bool pen = false;         // presence / frequency penalty on (qe_set_penalty): host-known tokens go to its counts
    bool nolk = false;        // lookup drafts off for this reply (qe_set_no_lookup: the server's loop guard)
    bool fault = false;       // the NaN guard failed this reply (FE::fault): its conversation is dropped, qe_release frees it
    unsigned long long used = 0;
    double idle_ms = 0;       // when its last reply ended (qe_release; 0: never released)
    // draft statistics since the slot was created (qe_draft_stats): steps, MTP drafted / accepted, lookup drafted / accepted
    long long ds[5] = {0, 0, 0, 0, 0};
    void count_step(bool lookup, int drafted, int emitted) {
        const int acc = std::max(0, emitted - 1);
        ds[0]++;
        ds[lookup ? 3 : 1] += std::max(drafted, acc);
        ds[lookup ? 4 : 2] += acc;
    }
};

struct FE {
    std::vector<int> devs;
    int ng = 3, ctx = 0, kmax = 3;
    SafeTensors st, mst;
    HostTables ht;
    NgramDev ngd{};
    Comm comm;
    std::vector<Plan> plans;
    std::vector<Worker*> W;
    // persistent workers
    std::function<void(int)> job;
    std::atomic<int> go{0}, done{0};
    std::atomic<bool> quit{false};
    std::vector<std::thread> pool;
    // sequence slots; cs = the slot the single-sequence path (prefill, speculative steps) is bound to
    std::vector<Slot> sl;
    int cs = 0, np = 0, period = 1536, reserve = 1024;
    // grammar tier host buffers per slot (pinned; the caller keeps views of them): outside Slot, which evictions and
    // qe_reset replace (a new buffer then would not be the one the caller fills)
    std::vector<unsigned*> mask_host;
    std::vector<int*> spec_io_h;
    unsigned* mh(int s) {
        if ((int)mask_host.size() < (int)sl.size()) mask_host.resize(sl.size(), nullptr);
        if (!mask_host.at(s)) {
            const size_t bytes = (size_t)Worker::MASK_ROWS * Worker::MASK_WORDS * 4;
            CUDA_CHECK(cudaSetDevice(devs[0]));
            CUDA_CHECK(cudaMallocHost((void**)&mask_host[s], bytes));
            memset(mask_host[s], 0xff, bytes);
        }
        return mask_host[s];
    }
    int* sio(int s) {
        if ((int)spec_io_h.size() < (int)sl.size()) spec_io_h.resize(sl.size(), nullptr);
        if (!spec_io_h.at(s)) {
            CUDA_CHECK(cudaSetDevice(devs[0]));
            CUDA_CHECK(cudaMallocHost((void**)&spec_io_h[s], 32 * 4));
            memset(spec_io_h[s], 0, 32 * 4);
        }
        return spec_io_h[s];
    }
    Slot& S() { return sl[cs]; }
    // the single-sequence decode of slot cs
    int k = 0;               // drafts of the running sequence (0: plain decoding)
    int lk_next = 0;         // device lookup drafts for the next step (FLASH_LOOKUP=1; 0: MTP step)
    bool prev_lk = false;    // the last step was a lookup step (only the fill's MTP draft exists)
    // lookahead (qe_set_lookahead): a step launched ahead, not yet returned
    bool lookahead = true, no_ahead = false, inflight = false, ahead_lk = false;
    int ahead_ks = 0;
    bool batch_api = true;   // FLASH_BATCH: the server sees every slot and steps several together (qe_step_batch)
    // drop a step launched ahead (any call that changes the sequence or reads the device). Its rows are never
    // committed; the slot's end state is not reused afterwards (dirty), its KV and checkpoints are.
    void drain() {
        if (!inflight) return;
        job = [this](int r) { W[r]->spec_discard(); };
        run();
        inflight = false;
        if (take_nf()) fault(cs, "dropped step", (int)S().hist.size());
        Slot& q = S();
        q.rb_pos = -1;  // the dropped step committed the rows of the step before it: no undo
        // the end state stays reusable: the next turn after a dropped step is bit-identical to one after a reply that
        // never launched ahead; FLASH_REUSE_DIRTY=0: not reused, as before
        static const bool reuse_dirty = env_int("FLASH_REUSE_DIRTY", 1);
        q.dirty = !reuse_dirty;
        // the dropped step advanced the speculative state on the devices: back to the host's (its own rows are not
        // committed, the ones before it are; drafts are guesses, the pending token repeated)
        if (k > 0 && q.pending >= 0 && q.hist.size() >= 1) {
            const int hn = q.hist.size();
            std::vector<int> d(k, q.pending);
            for (int r = 0; r < ng; r++) {
                W[r]->spec_init(q.pending, d.data(), k, hn, 0, q.hist[hn - 1], hn >= 2 ? q.hist[hn - 2] : EOS);
                W[r]->sv_set_live();
            }
            q.lk_stale = true;
        }
    }
    int last_reused = 0, last_copied = 0;
    bool last_from_archive = false;
    // activation steering (QE_STEER / QE_STEER_DIR): vectors loaded at startup; the next start's (vector, mode, layers,
    // scale) (qe_set_steering_ex, sticky; default: the server default vector with its defaults at scale 1) keys the
    // prefix match, the checkpoints and the archive: different keys never share KV
    std::vector<SteerSpec> steer;
    int steer_def = -1, steer_vec = -1, steer_mode = 0, steer_lo = 0, steer_hi = -1;
    float steer_scale = 1.f;
    std::string steer_json = "{\"default\": null, \"vectors\": []}";
    // key = vector | mode | lo | hi | scale bits (-1: off); steer_unkey decodes it
    long long steer_key() const {
        if (steer_vec < 0) return -1;
        uint32_t b;
        memcpy(&b, &steer_scale, 4);
        return ((((((long long)steer_vec * 2 + steer_mode) * 64 + steer_lo) * 64 + steer_hi)) << 32) | b;
    }
    static void steer_unkey(long long k, int& vec, int& mode, int& lo, int& hi, float& scale) {
        if (k < 0) { vec = -1; mode = 0; lo = 0; hi = -1; scale = 0.f; return; }
        const uint32_t b = (uint32_t)(k & 0xffffffffu);
        memcpy(&scale, &b, 4);
        long long h = k >> 32;
        hi = (int)(h % 64); h /= 64;
        lo = (int)(h % 64); h /= 64;
        mode = (int)(h % 2);
        vec = (int)(h / 2);
    }
    std::vector<bf16> ple;
    std::vector<int> outs;

    // ---- conversations: resident in slots and/or archived in host memory
    std::list<Conv> convs;
    HostPool hpool;
    size_t budget = 0;            // FLASH_KV_ARCHIVE_GB (host bytes of all archives)
    int next_id = 1, ckpt_every = 8192, ckpt_min = 512, ckpt_keep = 6, end_cut = 4096;
    unsigned long long tick = 0;
    const Conv* hold = nullptr;   // the conversation a start is restoring from (never evicted meanwhile)
    long long n_evict = 0, n_restore = 0, n_sync = 0, n_move = 0, n_grow = 0, n_slot_evict = 0, n_pause = 0, n_resume = 0;
    unsigned long long admit_tick = 0, pause_tick = 0;
    bool log = false;             // FLASH_SLOT_LOG=1: archive / restore / pool events on stderr
    // prompt being prefilled (qe_start_begin .. qe_start_finish)
    struct Pf {
        bool on = false;
        int slot = 0;
        std::vector<int> ids;
        int n = 0, i = 0, hint = 0, nxt = 0, reused = 0, copied = 0;
        bool from_archive = false;  // (qe_last_* after its finish: another start may have run in between)
        std::vector<int> cks;     // checkpoint positions still ahead (ascending)
        std::vector<int> drafts;
        // image prompt: device-side ids (image rows -> kImgToken + their row of the image table), the (t, h, w) RoPE
        // positions of every row, its rows [img0, img0 + img_n) of the image table
        std::vector<int> dev, r3;
        int img0 = 0, img_n = 0;
        double t_begin = 0, ms_sync = 0, ms_restore = 0, ms_pool = 0;
        bool multi_end = false;   // its last rows ran in a batch with other prompts (qe_prefill_multi): MTP chain row per slot
        int n_multi = 0;          // batches it shared with other prompts (log)
    };
    // prompts in flight (FLASH_MAX_PREFILLS, default 2): each on its own slot, prefilled chunk by chunk in any order
    // (a short prompt may run between the chunks of a long one); the slot's committed state lives in its storage
    // between chunks (bind switches the workspace)
    std::list<Pf> pfs;
    int max_pf = 2, last_pf_slot = 0;
    bf16* img = nullptr;          // image embedding table (pinned, mapped on every card): rows of the image prompts in flight
    size_t img_cap = 0;           // rows
    Pf* pf_of(int slot) {  // the prompt in flight on slot (slot < 0 or one exposed slot: the newest)
        if (pfs.empty()) return nullptr;
        if (slot < 0 || !batch_api) return &pfs.back();
        for (auto& x : pfs)
            if (x.slot == slot) return &x;
        return nullptr;
    }

    void run() {
        done.store(0);
        go.fetch_add(1, std::memory_order_acq_rel);
        job(0);
        while (done.load(std::memory_order_acquire) != ng - 1) std::this_thread::yield();
    }
    // the single-sequence path works on slot s from now on
    void bind(int s) {
        drain();
        const int b = sl[s].base;
        job = [&](int r) { W[r]->use_slot(s, b); };
        run();
        if (s != cs) {  // the lookup history on the devices belonged to the previous slot
            sl[s].lk_stale = true;
            lk_next = 0;
            prev_lk = false;
        }
        cs = s;
    }
    // trunk forward of toks at pos (n-gram rows from hist + toks; toks are appended to hist)
    int forward(const std::vector<int>& toks, int pos) {
        const int m = toks.size();
        auto& hist = S().hist;
        for (int i = 0; i < m; i++) {
            hist.push_back(toks[i]);
            ngram_row(ht, hist, (int)hist.size() - 1, ple.data() + (size_t)i * H);
        }
        job = [&, m, pos](int r) {
            std::vector<int> o(MAXM);
            W[r]->forward(toks.data(), pos, m, ple.data(), o.data(), nullptr, -1);
            if (r == 0) outs = o;
        };
        run();
        return outs[m - 1];
    }
    int mtp(const std::vector<int>& toks, int pos, bool chain) {
        const int m = toks.size();
        job = [&, m, pos, chain](int r) {
            std::vector<int> o(MAXM);
            W[r]->forward_mtp(toks.data(), pos, m, chain, o.data());
            if (r == 0) outs = o;
        };
        run();
        return outs[m - 1];
    }

    static double now_ms() {
        return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
    }

    // ---- KV pool regions (periods of 1536 positions = 512 rows on every card)
    int periods_for(int tokens) const { return std::max(1, (tokens + period - 1) / period); }
    // KV in RAM (lazy pinning): the pool may grow to what the host can still pin; never below the regions in use
    void refresh_np() {
        int top = 0;
        for (auto& q : sl) if (q.cap) top = std::max(top, q.base + q.cap);
        int v = W[0]->periods();
        for (int r = 0; r < ng; r++) v = std::min(v, W[r]->kv_ram_periods());
        np = std::max(top, v);
    }
    void update_top() {
        int top = 0;
        long long act = 0;
        for (auto& q : sl) {
            if (q.cap) top = std::max(top, q.base + q.cap);
            if (q.cap && q.active) act += (long long)q.cap * period;
        }
        const bool ram = W[0]->kv_stream();
        job = [&](int r) {
            if (ram) W[r]->set_kv_active(act);
            W[r]->set_kv_top(top);
            W[r]->kv_claim(0);  // expert-cache slots below the top -> KV (KV in RAM: block-cache entries of the active ones)
        };
        run();
    }
    // valid KV positions of a slot (its tokens + the rows of a step past them)
    int kv_len(const Slot& q) const { return std::min(q.cap * period, (int)q.hist.size() + MAXP + 1); }
    // first gap of `need` periods not used by any slot but `skip`
    int find_gap(int need, int skip) const {
        std::vector<std::pair<int, int>> used;
        for (int i = 0; i < (int)sl.size(); i++)
            if (i != skip && sl[i].cap) used.push_back({sl[i].base, sl[i].base + sl[i].cap});
        std::sort(used.begin(), used.end());
        int at = 0;
        for (auto& [a, b] : used) {
            if (a - at >= need) return at;
            at = std::max(at, b);
        }
        return np - at >= need ? at : -1;
    }
    void move_region(int s, int to) {
        Slot& q = sl[s];
        if (q.base == to) return;
        const int from = q.base, n = kv_len(q);
        job = [&](int r) { W[r]->kv_move(from, to, n); };
        run();
        q.base = to;
        n_move++;
        if (log) fprintf(stderr, "[qflash] pool: slot %d moved %d -> %d (%d tokens)\n", s, from, to, n);
    }
    // pack the regions of all slots but `skip` from the bottom (moves go down only)
    void compact(int skip) {
        std::vector<int> ord;
        for (int i = 0; i < (int)sl.size(); i++)
            if (i != skip && sl[i].cap) ord.push_back(i);
        std::sort(ord.begin(), ord.end(), [&](int a, int b) { return sl[a].base < sl[b].base; });
        int at = 0;
        for (int i : ord) {
            if (sl[i].base > at) move_region(i, at);
            at = sl[i].base + sl[i].cap;
        }
    }
    // region of slot s holds `tokens` positions (+ reserve): grows in place, moves to a gap, packs the others or
    // evicts idle slots (not `protect`); the slot bound to the single-sequence path is re-bound after any move
    bool ensure_region(int s, int tokens, const std::vector<int>& protect = {}) {
        refresh_np();
        Slot& q = sl[s];
        if (tokens > np * period) return false;
        const int need = std::min(np, periods_for(tokens + reserve));
        if (q.cap >= need) return true;
        const double t0 = now_ms();
        auto free_above = [&]() {
            if (q.base + need > np) return false;
            for (int i = 0; i < (int)sl.size(); i++)
                if (i != s && sl[i].cap && sl[i].base < q.base + need && sl[i].base + sl[i].cap > q.base + q.cap) return false;
            return true;
        };
        if (q.cap && free_above()) {
            n_grow++;
        } else if (q.cap) {
            // s holds rows: packing the others from the bottom (compact) would copy them over s's region before s
            // moves (KV of the growing sequence overwritten). Either a gap anywhere (s's own space counts) takes s, or
            // the others move away around s: those below it down, s down after them, those above it up to the top
            while (true) {
                const int at = find_gap(need, s);
                if (at >= 0) {
                    if (at != q.base) {
                        int top = at + need;  // the target rows leave the expert cache before the copy
                        for (auto& o : sl) if (o.cap) top = std::max(top, o.base + o.cap);
                        job = [&](int r) { W[r]->set_kv_top(top); W[r]->kv_claim(0); };
                        run();
                        move_region(s, at);
                    }
                    break;
                }
                int others = 0;
                for (int i = 0; i < (int)sl.size(); i++)
                    if (i != s && sl[i].cap) others += sl[i].cap;
                if (others + need <= np) {
                    make_room_around(s, need);
                    n_grow++;
                    break;
                }
                int v = -1;  // evict the least valuable idle slot
                for (int i = 0; i < (int)sl.size(); i++) {
                    if (i == s || !sl[i].cap || sl[i].active || std::find(protect.begin(), protect.end(), i) != protect.end()) continue;
                    if (v < 0 || prio_of(i) < prio_of(v) || (prio_of(i) == prio_of(v) && sl[i].used < sl[v].used)) v = i;
                }
                if (v < 0) return false;
                evict_slot(v);
            }
        } else {
            int at = find_gap(need, s);
            if (at < 0) {
                compact(s);
                at = find_gap(need, s);
            }
            while (at < 0) {  // evict the least valuable idle slot
                int v = -1;
                for (int i = 0; i < (int)sl.size(); i++) {
                    if (i == s || !sl[i].cap || sl[i].active || std::find(protect.begin(), protect.end(), i) != protect.end()) continue;
                    if (v < 0 || prio_of(i) < prio_of(v) || (prio_of(i) == prio_of(v) && sl[i].used < sl[v].used)) v = i;
                }
                if (v < 0) return false;
                evict_slot(v);
                compact(s);
                at = find_gap(need, s);
            }
            if (q.cap && at != q.base) {
                int top = at + need;  // the target rows leave the expert cache before the copy
                for (auto& o : sl) if (o.cap) top = std::max(top, o.base + o.cap);
                job = [&](int r) { W[r]->set_kv_top(top); W[r]->kv_claim(0); };
                run();
                move_region(s, at);
            }
            q.base = at;
        }
        q.cap = need;
        update_top();
        rebind();
        if (log) fprintf(stderr, "[qflash] pool: slot %d region [%d, %d) periods for %d tokens (%.1f ms)\n", s, q.base, q.base + q.cap, tokens,
                         now_ms() - t0);
        return true;
    }
    // regions of the other slots out of the way of s growing to `need` periods in place (every move is safe for
    // overlapping regions: down in ascending, up in descending order; kv_move orders its chunks by direction)
    void make_room_around(int s, int need) {
        Slot& q = sl[s];
        job = [&](int r) { W[r]->set_kv_top(np); W[r]->kv_claim(0); };  // the whole pool leaves the expert cache first
        run();
        std::vector<int> lo, hi;
        for (int i = 0; i < (int)sl.size(); i++)
            if (i != s && sl[i].cap) (sl[i].base < q.base ? lo : hi).push_back(i);
        std::sort(lo.begin(), lo.end(), [&](int a, int b) { return sl[a].base < sl[b].base; });
        std::sort(hi.begin(), hi.end(), [&](int a, int b) { return sl[a].base > sl[b].base; });
        int at = 0;
        for (int i : lo) {
            if (sl[i].base > at) move_region(i, at);
            at = sl[i].base + sl[i].cap;
        }
        if (q.base > at) move_region(s, at);
        int top = np;
        for (int j : hi) {
            const int t = top - sl[j].cap;
            CHECK(t >= sl[j].base, "flash pool: region %d [%d, +%d) cannot move up to %d", j, sl[j].base, sl[j].cap, t);
            if (t > sl[j].base) move_region(j, t);
            top = t;
        }
        CHECK(q.base + need <= top, "flash pool: no room around slot %d (%d + %d > %d)", s, q.base, need, top);
    }
    void rebind() {
        const int b = sl[cs].base;
        job = [&](int r) { W[r]->use_slot(cs, b); };
        run();
    }
    int prio_of(int s) const { return sl[s].conv ? sl[s].conv->prio : 0; }
    // ---- NaN guard (Worker::nf_check, FLASH_GUARD=1): a reply whose forward produced non-finite values fails alone
    // (the server returns an error for that request), its conversation and its archive are dropped (never reused,
    // the next turn recomputes it) and the other sequences go on. Repeated faults (more than FLASH_GUARD_MAX, default
    // 4, within 5 minutes) point at shared state (weights / expert cache): then the process ends as before. A repeat
    // of the same fault (same token history, position and step kind: one prompt re-sent, issue #2) counts once
    // (guard_rule.h).
    long long n_fault = 0;
    std::vector<std::string> fault_msg;  // per slot, until qe_slot_fault reads it
    std::vector<GuardEvent> fault_ev;
    unsigned take_nf() {
        unsigned m = 0;
        for (int r = 0; r < ng; r++) m |= W[r]->take_nf();
        return m;
    }
    void fault(int s, const char* what, int pos) {
        Slot& q = sl[s];
        char b[640];
        snprintf(b, sizeof b, "non-finite values in the forward (%s, slot %d, position %d, %zu tokens, conversation %d, "
                 "steering %s)", what, s, pos, q.hist.size(), q.conv ? q.conv->id : -1,
                 q.conv && q.conv->steer >= 0 ? "on" : "off");
        std::string env;
        for (char** v = environ; v && *v; v++)
            if (!strncmp(*v, "FLASH_", 6) || !strncmp(*v, "QE_", 3)) env += std::string(" ") + *v;
        fprintf(stderr, "[qflash] FAULT %s: the request fails, its conversation is dropped, the server goes on (env:%s)\n", b,
                env.c_str());
        fflush(stderr);
        n_fault++;
        static const int fmax = env_int("FLASH_GUARD_MAX", 4);
        const size_t nk = std::min(q.hist.size(), (size_t)std::max(0, pos));
        const uint64_t key = guard_key(q.hist.data(), nk, pos, what);
        const GuardVerdict v = guard_note(fault_ev, now_ms(), key, fmax);
        if (v.repeat)
            fprintf(stderr, "[qflash] GUARD: the same fault again (key %016llx: same %zu tokens, position %d, %s; %d times "
                            "within 5 minutes): counted once toward FLASH_GUARD_MAX (%d distinct of %d)\n",
                    (unsigned long long)key, nk, pos, what, v.hits, v.distinct, fmax);
        if (v.abort) {
            fprintf(stderr, "[flash] FATAL: %d distinct non-finite faults within 5 minutes (FLASH_GUARD_MAX %d): shared "
                            "state is likely corrupt; exiting\n", v.distinct, fmax);
            fflush(stderr);
            fflush(stdout);
            abort();
        }
        // its conversation (KV, states, archive) is forgotten on qe_release (here a caller may hold pointers into
        // convs); until then the slot emits nothing
        q.pending = -1;
        q.fault = true;
        q.synced = true;
        if ((int)fault_msg.size() < (int)sl.size()) fault_msg.resize(sl.size());
        fault_msg[s] = b;
    }
    // slot s becomes free: its conversation stays in the archive (synced first) or is forgotten
    void evict_slot(int s) {
        Slot& q = sl[s];
        if (q.conv) {
            if (budget && !q.synced) sync(s);
            Conv* c = q.conv;
            c->slot = -1;
            if (!(c->end || !c->ck.empty()) || !budget) forget(c);
            if (log) fprintf(stderr, "[qflash] pool: slot %d evicted (%zu tokens)\n", s, q.hist.size());
        }
        n_slot_evict++;
        q = Slot{};
    }
    // ---- KV in VRAM: the expert cache lives in the pool above the highest region (update_top), so an idle
    // conversation's region takes expert-cache slots from the sequences decoding next to it (2 GPUs, c=8 after 4 x 90K
    // bench prompts: 2,926 of 9,010 live slots, cold-group hit rate 27 vs 50 %, -15 % tok/s). While others decode
    // (batched steps), a conversation idle for FLASH_EC_RECLAIM_S seconds (default 30, 0 = off) leaves VRAM: only with
    // an archive (FLASH_KV_ARCHIVE_GB > 0), only once the archive holds all its rows and its end state (a later turn
    // restores it from host memory, never a new prefill), never the main thread (prio 2). The other regions are then
    // packed down and the expert cache takes the freed rows. At most one check per second.
    double reclaim_last = 0;
    long long n_reclaim = 0;
    void reclaim_idle() {
        static const double T = env_double("FLASH_EC_RECLAIM_S", 30.0);
        if (T <= 0 || !budget || !batch_api || !pfs.empty() || W[0]->kv_stream() || !W[0]->ec_enabled()) return;
        const double t = now_ms();
        if (t - reclaim_last < 1000.0) return;
        reclaim_last = t;
        int top0 = 0, n = 0, toks = 0;
        for (auto& q : sl) if (q.cap) top0 = std::max(top0, q.base + q.cap);
        for (int i = 0; i < (int)sl.size(); i++) {
            Slot& q = sl[i];
            if (!q.conv || !q.cap || q.active || q.paused || prio_of(i) >= 2 || q.idle_ms <= 0 || t - q.idle_ms < T * 1000.0) continue;
            if (!q.synced && !sync(i)) continue;  // the archive cannot take it: it stays resident
            if (!q.synced || !q.conv || !q.conv->end || q.conv->toks.size() != q.hist.size()) continue;  // no end state
            toks += (int)q.hist.size();
            evict_slot(i);
            n++;
        }
        if (!n) return;
        compact(-1);
        update_top();
        rebind();
        n_reclaim += n;
        int top1 = 0;
        for (auto& q : sl) if (q.cap) top1 = std::max(top1, q.base + q.cap);
        fprintf(stderr, "[qflash] pool: %d idle conversation%s (%d tokens, idle >= %.0f s) left VRAM for the expert cache "
                "(archived), pool top %d -> %d of %d periods (%.1f ms)\n", n, n > 1 ? "s" : "", toks, T, top0, top1, np,
                now_ms() - t);
    }
    void forget(Conv* c) {
        if (c->slot >= 0) { sl[c->slot].conv = nullptr; }
        for (auto it = convs.begin(); it != convs.end(); ++it)
            if (&*it == c) { convs.erase(it); return; }
    }

    // ---- archive plumbing
    static bool host_ptr(const void* p) {
        cudaPointerAttributes a{};
        if (cudaPointerGetAttributes(&a, p) != cudaSuccess) { cudaGetLastError(); return true; }
        return a.type != cudaMemoryTypeDevice && a.type != cudaMemoryTypeManaged;
    }
    static size_t bytes_of(const std::vector<Worker::Piece>& ps) {
        size_t s = 0;
        for (auto& p : ps) s += p.n;
        return s;
    }
    // copies between the pieces of every card and a blob (to_host: D2H), cards in parallel
    void xfer(HBlob& b, const std::function<void(int, std::vector<Worker::Piece>&)>& pieces, bool to_host) {
        job = [&](int r) {
            CUDA_CHECK(cudaSetDevice(W[r]->device()));
            std::vector<Worker::Piece> ps;
            pieces(r, ps);
            size_t off = 0;
            const cudaStream_t s = W[r]->stream();
            for (auto& q : ps) {
                // (cudaMemcpyDefault: with KV streaming the K / V pieces are host rows)
                // KV in RAM: the K / V pieces are host rows: host copies after the stream's work (a pageable archive
                // blob next to registered rows is not a valid async copy)
                if (host_ptr(q.p)) {
                    CUDA_CHECK(cudaStreamSynchronize(s));
                    if (to_host) memcpy(b.p[r] + off, q.p, q.n);
                    else memcpy(q.p, b.p[r] + off, q.n);
                } else if (to_host) CUDA_CHECK(cudaMemcpyAsync(b.p[r] + off, q.p, q.n, cudaMemcpyDefault, s));
                else CUDA_CHECK(cudaMemcpyAsync(q.p, b.p[r] + off, q.n, cudaMemcpyDefault, s));
                off += q.n;
            }
            CHECK(off == b.n[r], "flash archive: blob size mismatch (%zu vs %zu)", off, b.n[r]);
            CUDA_CHECK(cudaStreamSynchronize(s));
        };
        run();
    }
    // H2D of archived KV segment g (positions [a, z), region base b) limited to the positions < upto: the device pieces
    // of [a, min(z, upto)) are prefixes of those of [a, z) (same rows from r0, fewer rows / index blocks)
    void xfer_part(const HBlob& g, int b, int a, int z, int upto) {
        const int z2 = std::min(z, upto);
        if (z2 <= a) return;
        job = [&](int r) {
            CUDA_CHECK(cudaSetDevice(W[r]->device()));
            std::vector<Worker::Piece> full, part;
            W[r]->kv_pieces_at(b, a, z, full);
            W[r]->kv_pieces_at(b, a, z2, part);
            CHECK(part.size() == full.size() || part.empty(), "flash archive: segment pieces %zu vs %zu", part.size(), full.size());
            size_t off = 0;
            const cudaStream_t s = W[r]->stream();
            for (size_t i = 0; i < part.size(); i++) {
                CHECK(part[i].p == full[i].p && part[i].n <= full[i].n, "flash archive: segment piece %zu", i);
                if (host_ptr(part[i].p)) { CUDA_CHECK(cudaStreamSynchronize(s)); memcpy(part[i].p, g.p[r] + off, part[i].n); }
                else CUDA_CHECK(cudaMemcpyAsync(part[i].p, g.p[r] + off, part[i].n, cudaMemcpyDefault, s));
                off += full[i].n;
            }
            W[r]->kv_inval(b, a, z2);  // KV streaming: cached copies of the restored rows are stale
            CHECK(off == g.n[r] || part.empty(), "flash archive: blob size mismatch (%zu vs %zu)", off, g.n[r]);
            CUDA_CHECK(cudaStreamSynchronize(s));
        };
        run();
    }
    std::vector<size_t> sizes_of(const std::function<void(int, std::vector<Worker::Piece>&)>& pieces) {
        std::vector<size_t> v(ng);
        for (int r = 0; r < ng; r++) {
            std::vector<Worker::Piece> ps;
            pieces(r, ps);
            v[r] = bytes_of(ps);
        }
        return v;
    }
    // evict archived conversations (lowest value, least recently used first) until `need` more bytes fit the budget;
    // resident ones keep their slot and lose only their archive
    // the archive's byte limit now: the budget, capped by what the host can give (host_mem.h: the archive's free
    // pinned bytes + MemAvailable - margin on top of what it holds); logged while it caps the budget
    qe_hostmem::CapLog cap_log;
    size_t eff_budget() {
        const size_t used = hpool.used(), cap = std::min(budget, used + hpool.pinned_free() + qe_hostmem::headroom());
        if (cap < budget) {
            if (cap_log.due(cap / 1e9))
                fprintf(stderr, "[qflash] archive: host memory low, budget capped at %.1f of %.1f GB (%.1f GB used, "
                                "MemAvailable %.1f GB, margin %.1f GB)\n", cap / 1e9, budget / 1e9, used / 1e9,
                        qe_hostmem::available() / 1e9, qe_hostmem::margin() / 1e9);
        } else if (cap_log.last_gb >= 0) {
            fprintf(stderr, "[qflash] archive: host memory ok again, budget %.1f GB\n", budget / 1e9);
            cap_log.reset();
        }
        return cap;
    }
    // memory pressure from outside (another process grew): below the margin the archive gives memory back,
    // free pinned slabs first, then archived conversations by the eviction order of make_room
    void mem_guard() {
        if (!budget) return;
        size_t d = qe_hostmem::deficit();
        if (!d) return;
        d = qe_hostmem::deficit(true);
        if (!d) return;
        const size_t u0 = hpool.used(), p0 = hpool.pinned();
        size_t got = hpool.release(d);
        if (got < d) {
            const size_t target = u0 > d - got ? u0 - (d - got) : 0;
            make_room(0, nullptr, target);
            got += hpool.release(d - got);
        }
        static double last_log = -1e18;  // a persistent deficit (nothing left to give) is logged once a minute
        const double tnow = std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
        if (got == 0 && hpool.used() == u0 && tnow - last_log < 60) return;
        last_log = tnow;
        fprintf(stderr, "[qflash] archive: host memory below the margin by %.1f GB (MemAvailable %.1f GB, margin %.1f GB): "
                        "archive %.1f -> %.1f GB used, %.1f -> %.1f GB pinned\n", d / 1e9, qe_hostmem::available(true) / 1e9,
                qe_hostmem::margin() / 1e9, u0 / 1e9, hpool.used() / 1e9, p0 / 1e9, hpool.pinned() / 1e9);
    }
    bool make_room(size_t need, const Conv* keep, size_t limit = (size_t)-1) {
        if (limit == (size_t)-1) limit = eff_budget();
        while (hpool.used() + need > limit) {
            auto victim = convs.end();
            for (auto it = convs.begin(); it != convs.end(); ++it) {
                if (&*it == keep || &*it == hold || it->slot >= 0) continue;
                if (victim == convs.end() || it->prio < victim->prio || (it->prio == victim->prio && it->used < victim->used))
                    victim = it;
            }
            if (victim == convs.end()) return false;
            // the main thread (prio 2) gives back its newest part first and keeps a prefix it resumes from (its next
            // turn prefills only the tail instead of the whole context); the others go whole
            if (victim->prio >= 2 && trim_tail(*victim)) continue;
            // always logged when it costs a long re-prefill later (a lost main thread showed up only as a slow turn)
            if (log || victim->prio >= 2 || victim->toks.size() >= 16384)
                fprintf(stderr, "[qflash] archive: evict conversation %d (%zu tokens, prio %d) for %.1f MB (%.2f of %.2f GB used)\n",
                        victim->id, victim->toks.size(), victim->prio, need / 1e6, hpool.used() / 1e9, limit / 1e9);
            convs.erase(victim);
            n_evict++;
        }
        return true;
    }
    // drop an archived conversation's newest part: its end state, else the KV segments from its highest checkpoint
    // below the last segment on (and the states past them). False: nothing left to drop but the whole conversation.
    bool trim_tail(Conv& c) {
        if (c.end) {
            c.end = nullptr;
            return true;
        }
        if (c.segs.empty()) return false;
        int P = -1;
        for (auto& x : c.ck)
            if (x->pos > 0 && x->pos <= c.segs.back()->t0) P = std::max(P, x->pos);
        if (P < 0) return false;
        while (!c.segs.empty() && c.segs.back()->t0 >= P) c.segs.pop_back();
        c.kv_n = c.segs.empty() ? 0 : std::min(c.kv_n, c.segs.back()->t1);
        c.ck.erase(std::remove_if(c.ck.begin(), c.ck.end(), [&](auto& x) { return x->pos > c.kv_n; }), c.ck.end());
        fprintf(stderr, "[qflash] archive: conversation %d (%zu tokens, prio %d) trimmed to its checkpoint at %d (%.2f of %.2f GB used)\n",
                c.id, c.toks.size(), c.prio, P, hpool.used() / 1e9, budget / 1e9);
        return true;
    }
    template <class B>
    std::shared_ptr<B> new_blob(const std::vector<size_t>& sizes, const Conv* keep) {
        size_t tot = 0;
        for (size_t s : sizes) tot += (s + 4095) / 4096 * 4096;
        mem_guard();
        if (!make_room(tot, keep)) return nullptr;
        auto b = std::make_shared<B>();
        if (!b->alloc(&hpool, sizes)) return nullptr;
        return b;
    }
    // committed / end state of slot s -> host
    std::shared_ptr<State> save_state(int s, bool pend, int kind) {
        auto pc = [&](int r, std::vector<Worker::Piece>& ps) { W[r]->state_pieces_of(s, pend, ps); };
        auto st = new_blob<State>(sizes_of(pc), sl[s].conv);
        if (!st) return nullptr;
        xfer(*st, pc, true);
        st->pos = sl[s].hist.size();
        st->pend = pend;
        st->prev_m = 0;
        if (pend) {  // pending rows of the slot's last forward
            st->prev_m = W[0]->slot_prev_m(s);
        }
        st->kind = kind;
        return st;
    }
    // archive slot s's conversation: KV of [kv_n, min(hist, upto)); with `end`, then its end state
    bool sync(int s, int upto = 1 << 30, bool end = true) {
        Slot& q = sl[s];
        Conv* c = q.conv;
        if (!c || !budget) return false;
        const int hn = q.hist.size();
        const double t0 = now_ms();
        if (c->kv_n > hn) {  // the sequence was cut: drop what lies beyond
            while (!c->segs.empty() && c->segs.back()->t0 >= hn) c->segs.pop_back();
            c->kv_n = hn;
        }
        const int t1 = std::min(hn, upto);
        if (t1 > c->kv_n) {
            const int t0p = c->kv_n, b = q.base;
            auto pc = [&](int r, std::vector<Worker::Piece>& ps) { W[r]->kv_pieces_at(b, t0p, t1, ps); };
            auto sg = new_blob<Seg>(sizes_of(pc), c);
            if (!sg) return false;
            xfer(*sg, pc, true);
            sg->t0 = t0p;
            sg->t1 = t1;
            c->segs.push_back(sg);
            c->kv_n = t1;
        }
        if (t1 < hn || !end) return true;
        c->toks = q.hist;
        c->end = nullptr;
        if (!q.dirty) {
            c->end = save_state(s, true, 0);
            if (!c->end) return false;
        }
        q.synced = true;
        n_sync++;
        if (log) fprintf(stderr, "[qflash] archive: conversation %d (slot %d) synced at %d tokens (%.1f ms, %.2f GB host, %.1f GB pinned)\n",
                         c->id, s, hn, now_ms() - t0, hpool.used() / 1e9, hpool.pinned() / 1e9);
        return true;
    }
    // best cached start for ids[0, n): conversation, positions reused, state to load (null: the live end of a slot)
    // best cached start for ids[0, n): conversation, positions reused, state to load (null: the live end of a slot);
    // maxm: the longest common prefix with any known sequence (a divergence point worth a checkpoint)
    struct Src { Conv* c = nullptr; int L = 0; std::shared_ptr<State> st; bool live = false; int maxm = 0; };
    Src best(const int* ids, int n, bool any_slot) {
        const long long sk = steer_key();
        Src b;
        int brank = -1, maxm = 0;
        auto better = [&](int L, int rank) { return L > b.L || (L == b.L && L > 0 && rank > brank); };
        for (auto& c : convs) {
            if (c.steer != sk) continue;  // steered KV never serves an unsteered request, nor the reverse
            if (c.slot >= 0 && sl[c.slot].paused) continue;  // no rows on the device; its archive belongs to the slot
            if (c.slot >= 0 && sl[c.slot].fault) continue;  // failed by the NaN guard (dropped on its release)
            const bool res = c.slot >= 0;
            // a generating slot: its checkpoints (rows below them do not change), not its live end
            const bool busy = res && sl[c.slot].active && !any_slot;
            const std::vector<int>& t = res ? sl[c.slot].hist : c.toks;
            int M = 0;
            const int lim = std::min<int>(t.size(), n);
            while (M < lim && t[M] == ids[M]) M++;
            maxm = std::max(maxm, M);
            if (res && !busy && !sl[c.slot].dirty && M == (int)t.size() && M < n && better(M, 3)) {
                b = Src{&c, M, nullptr, true};
                brank = 3;
            }
            if (!res && c.end && M == (int)c.toks.size() && c.end->pos == M && M < n && c.kv_n >= M && better(M, 1)) {
                b = Src{&c, M, c.end, false};
                brank = 1;
            }
            for (auto& s : c.ck)
                if (s->pos <= M && s->pos < n && (res || c.kv_n >= s->pos) && better(s->pos, res ? 2 : 0)) {
                    b = Src{&c, s->pos, s, false};
                    brank = res ? 2 : 0;
                }
        }
        b.maxm = maxm;
        return b;
    }
    void add_ck(Conv* c, std::shared_ptr<State> s) {
        if (!s || !c) return;
        auto& v = c->ck;
        v.erase(std::remove_if(v.begin(), v.end(), [&](auto& x) { return x->pos == s->pos; }), v.end());
        v.push_back(s);
        while ((int)v.size() > ckpt_keep) {  // drop the oldest periodic, then prompt-end ones; hint states last
            int drop = -1;
            for (int want : {3, 5, 4, 2, 1}) {
                for (int i = 0; i < (int)v.size() - 1; i++)
                    if (v[i]->kind == want) { drop = i; break; }
                if (drop >= 0) break;
            }
            if (drop < 0) drop = 0;
            v.erase(v.begin() + drop);
        }
    }
    // ---- preemption: active sequences that do not all fit the pool
    bool prefilling(int s) const {
        for (auto& x : pfs)
            if (x.slot == s) return true;
        return false;
    }
    // region periods a running slot needs for its next step (+ the reserve: its floor)
    int floor_of(const Slot& q) const { return std::min(np, periods_for((int)q.hist.size() + MAXP + 2 + reserve)); }
    // slot v leaves the pool: its KV rows -> its conversation's archive (synced up to its tokens), region freed
    bool pause_slot(int v) {
        Slot& q = sl[v];
        if (q.paused || !q.active || !q.conv || !q.cap || q.pending < 0 || prefilling(v) || !budget) return false;
        if (v == cs) drain();
        const int hn = q.hist.size();
        if (!sync(v, hn, false) || q.conv->kv_n < hn) {
            if (log) fprintf(stderr, "[qflash] pool: slot %d (%d tokens) cannot be paused: no archive room\n", v, hn);
            return false;
        }
        if (log) fprintf(stderr, "[qflash] pool: slot %d paused at %d tokens (prio %d), region [%d, %d) freed\n", v, hn, prio_of(v), q.base,
                         q.base + q.cap);
        q.cap = 0;
        q.paused = true;
        q.pause_t = ++pause_tick;
        q.bhist = -1;
        n_pause++;
        update_top();
        rebind();
        return true;
    }
    // a paused slot comes back (same rows, same state): only when the room left keeps every running sequence's floor
    bool unpause_slot(int v, const std::vector<int>& protect) {
        Slot& q = sl[v];
        if (!q.paused) return true;
        const int hn = q.hist.size();
        int held = 0, deficit = 0;
        for (int j = 0; j < (int)sl.size(); j++) {
            const Slot& x = sl[j];
            if (!x.active || x.paused) continue;
            held += x.cap;
            if (!prefilling(j)) deficit += std::max(0, floor_of(x) - x.cap);
        }
        if (held + deficit + floor_of(q) > np) return false;
        std::vector<int> pr = protect;
        pr.push_back(v);
        if (!ensure_region(v, hn + MAXP + 2, pr)) return false;
        const int b = q.base;
        Conv* c = q.conv;
        job = [&](int r) { W[r]->kv_claim(hn); };
        run();
        for (auto& g : c->segs) xfer_part(*g, b, g->t0, g->t1, hn);
        q.paused = false;
        n_resume++;
        if (log) fprintf(stderr, "[qflash] pool: slot %d resumed at %d tokens in [%d, %d)\n", v, hn, q.base, q.base + q.cap);
        return true;
    }
    // room for slot s's next step (tokens positions); when the active sequences do not all fit, the one worth least
    // pauses (lowest prio, the most recently admitted among equals; never a prompt being prefilled). False: s has no
    // room (paused itself: q.paused, else it is alone and does not fit the pool)
    bool room_or_pause(int s, int tokens, const std::vector<int>& protect) {
        if (ensure_region(s, tokens, protect)) return true;
        while (true) {
            int v = -1, others = 0;
            for (int j = 0; j < (int)sl.size(); j++) {
                const Slot& x = sl[j];
                if (!x.active || x.paused || !x.cap || x.pending < 0 || prefilling(j)) continue;
                if (j != s) others++;
                if (v < 0 || prio_of(j) < prio_of(v) || (prio_of(j) == prio_of(v) && x.admit > sl[v].admit)) v = j;
            }
            if (others == 0 || v < 0 || !pause_slot(v)) return false;
            if (v == s) return false;
            if (ensure_region(s, tokens, protect)) return true;
        }
    }
    // paused slots of `slots` back in queue order (prio, then pause time), strictly head-of-line
    void resume_paused(const int* slots, int n) {
        std::vector<int> pz;
        for (int i = 0; i < n; i++)
            if (sl.at(slots[i]).paused) pz.push_back(slots[i]);
        std::stable_sort(pz.begin(), pz.end(), [&](int a, int b) {
            return prio_of(a) != prio_of(b) ? prio_of(a) > prio_of(b) : sl[a].pause_t < sl[b].pause_t;
        });
        for (int v : pz)
            if (!unpause_slot(v, std::vector<int>(slots, slots + n))) break;
    }
    int free_slot() const {
        for (int i = 0; i < (int)sl.size(); i++)
            if (!sl[i].conv && !sl[i].active) return i;
        return -1;
    }
};
}  // namespace

extern "C" {
int qe_set_steering_ex(void* p, int vec, float scale, int mode, int lo, int hi);

void* qe_create(const char* model_dir, const char* devices, int max_t, const char* split_s) {
    FE* e = new FE();
    const std::string model = model_dir;
    for (auto& x : split(devices && *devices ? devices : "0,1,2", ',')) e->devs.push_back(std::stoi(x));
    e->ng = e->devs.size();
    if (e->ng != 2 && e->ng != 3) { fprintf(stderr, "qflash: 2 or 3 GPUs required (%d given)\n", e->ng); delete e; return nullptr; }
    {  // one QEngine process per GPU (gpu_guard.h)
        const std::string busy = qe_gpu_guard::acquire(e->devs);
        if (!busy.empty()) { fprintf(stderr, "qflash: %s\n", busy.c_str()); delete e; return nullptr; }
    }
    e->ctx = max_t > 0 ? std::min(max_t, 262144) : 262144;
    e->kmax = env_int("FLASH_MTP_K", 3);
    // QE_STEER / QE_STEER_DIR (flash_steer.h): checked first, a refused vector stops the start
    try {
        e->steer = steer_load_env(H, LAYERS, model, &e->steer_def);
    } catch (std::exception& ex) {
        fprintf(stderr, "qflash: %s\n", ex.what());
        delete e;
        return nullptr;
    }
    if (e->steer.size() > 4096) { fprintf(stderr, "qflash: steering: more than 4096 vectors\n"); delete e; return nullptr; }
    static_assert(LAYERS <= 64, "steer_key: 6 bits per layer index");
    qe_set_steering_ex(e, -2, 1.f, -1, -1, -1);
    // expert groups per card: the split (one weight per GPU), else the default for the GPU count (3: 176,160,176)
    std::vector<double> wts = default_group_weights(e->ng);
    if (split_s && *split_s) {
        std::vector<double> v;
        for (auto& x : split(split_s, ',')) v.push_back(std::stod(x));
        if ((int)v.size() == e->ng) wts = v;
        else fprintf(stderr, "qflash: split %s has %zu weights for %d GPUs: default split used\n", split_s, v.size(), e->ng);
    }
    auto t0 = std::chrono::steady_clock::now();
    auto secs = [&]() { return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(); };
    {
        DIR* d = opendir(model.c_str());
        if (!d) { fprintf(stderr, "qflash: cannot open %s\n", model_dir); delete e; return nullptr; }
        for (dirent* de; (de = readdir(d));) {
            std::string n = de->d_name;
            if (n.size() > 12 && n.substr(n.size() - 12) == ".safetensors" && n.rfind("mtp", 0) != 0) e->st.open(model + "/" + n);
        }
        closedir(d);
    }
    e->mst.open(model + "/mtp.safetensors");
    e->mst.open(model + "/mtp-int4.safetensors");
    // the INT8 n-gram table (ngram_int8.safetensors, tools/convert_ngram_int8.py) is pinned instead of the BF16 one when
    // the model directory has it (51.8 instead of 102.4 GB of host memory; KL vs BF16 below the run-to-run noise,
    // see the README); FLASH_NGRAM=bf16 keeps the BF16 table, FLASH_NGRAM=int8 requires the INT8 one
    const char* ngv = env_str("FLASH_NGRAM");
    const bool ng8 = ngv ? !strcmp(ngv, "int8") : e->st.has("model.language_model.layers.1.ple.ple_embedding.ngram_q8.shard_0");
    {
        const TensorInfo& emb = e->st.get("model.language_model.embed_tokens.weight");
        e->ht.embed = (const bf16*)pin_host(emb.data, emb.nbytes);
        const std::string pe = "model.language_model.layers.1.ple.ple_embedding.";
        for (int i = 0; i < 128; i++) {
            if (ng8) e->ht.ngram8.push_back((const uint8_t*)e->st.get(pe + "ngram_q8.shard_" + std::to_string(i)).data);
            else e->ht.ngram.push_back((const bf16*)e->st.get(pe + "ngram_embedding.shard_" + std::to_string(i) + ".weight").data);
        }
        memcpy(e->ht.mult, e->st.get(pe + "layer_multipliers").data, 24);
        memcpy(e->ht.vsz, e->st.get(pe + "ngram_heads_vocab_sizes").data, 8 * PLE_HEADS);
        memcpy(e->ht.voff, e->st.get(pe + "ngram_heads_offsets").data, 8 * PLE_HEADS);
    }
    fprintf(stderr, "[qflash %.0fs] tensors mapped, embed pinned\n", secs());
    // n-gram table pinned + mapped (device-side gather in the speculative step; host gathers of prompt rows read the
    // pinned copy too: the file pages may be evicted): copied in the background while the workers load the weights
    std::vector<const char*> ng_host(128);
    // NUMA (several nodes): the n-gram table interleaved over the GPUs' nodes, cold experts on each GPU's node
    std::vector<int> ng_nodes;
    if (numa_nodes() > 1) {
        std::string nl;
        for (int d : e->devs) {
            const int nd = gpu_numa_node(d);
            nl += (nl.empty() ? "" : ", ") + std::string("GPU") + std::to_string(d) + " node " + std::to_string(nd);
            if (nd >= 0 && std::find(ng_nodes.begin(), ng_nodes.end(), nd) == ng_nodes.end()) ng_nodes.push_back(nd);
        }
        fprintf(stderr, "[qflash] NUMA: %s; cold experts on each GPU's node, n-gram table %s\n", nl.c_str(),
                ng_nodes.size() > 1 ? "interleaved over the GPUs' nodes" : "on the GPUs' node");
    }
    double ng_secs = 0;
    size_t ng_bytes = 0;
    std::promise<void> ng_prom;
    std::shared_future<void> ng_done = ng_prom.get_future().share();
    std::thread ng_th([&] {
        const double a = secs();
        uintptr_t lo = ~(uintptr_t)0, hi = 0;
        auto tens = [&](int i) -> const TensorInfo& {
            const std::string pe = "model.language_model.layers.1.ple.ple_embedding.";
            return ng8 ? e->st.get(pe + "ngram_q8.shard_" + std::to_string(i))
                       : e->st.get(pe + "ngram_embedding.shard_" + std::to_string(i) + ".weight");
        };
        for (int i = 0; i < 128; i++) {
            const TensorInfo& t = tens(i);
            lo = std::min(lo, (uintptr_t)t.data);
            hi = std::max(hi, (uintptr_t)t.data + t.nbytes);
        }
        size_t foff = 0;
        const std::string fpath = e->st.file_of((const void*)lo, &foff);
        // FLASH_NGRAM_PIN=0: the table stays in the file mapping (pageable, read from the page cache / disk on demand):
        // the GPU gathers read it through HMM (pageable memory access, open kernel modules), no host RAM is pinned
        const bool pinned = env_int("FLASH_NGRAM_PIN", 1);
        if (!pinned) {
            int pma = 0;
            CUDA_CHECK(cudaDeviceGetAttribute(&pma, cudaDevAttrPageableMemoryAccess, e->devs[0]));
            CHECK(pma, "FLASH_NGRAM_PIN=0 needs GPU access to pageable host memory (HMM, NVIDIA open kernel modules)");
            // row reads are random: no read-around on faults (with part of the file cached the kernel keeps
            // read-around on and each 162-byte row miss reads 128 KB: 48 GB read in a 4-minute run)
            if (env_int("FLASH_NGRAM_RANDOM", 1))
                madvise((void*)(lo & ~(uintptr_t)4095), hi - (lo & ~(uintptr_t)4095), MADV_RANDOM);
        }
        char* pin = !pinned ? (char*)lo
                  : fpath.empty() ? pin_host((const void*)lo, hi - lo, 32, true, ng_nodes)
                                  : pin_host_file(fpath, foff, hi - lo, (const void*)lo, 32, ng_nodes);
        for (int i = 0; i < 128; i++) {
            const char* h = pin + ((uintptr_t)tens(i).data - lo);
            void* dp = (void*)h;
            if (pinned) CUDA_CHECK(cudaHostGetDevicePointer(&dp, (void*)h, 0));
            if (ng8) e->ngd.q8[i] = (const uint8_t*)dp;
            else e->ngd.shard[i] = (const bf16*)dp;
            ng_host[i] = h;
        }
        e->ngd.is8 = ng8;
        ng_bytes = hi - lo;
        ng_secs = secs() - a;
        ng_prom.set_value();
    });
    e->comm = comm_create(e->ng, ARU_MAX * H, H);
    comm_enable_p2p(e->comm, e->devs.data());
    comm_select_path(e->comm, e->devs.data(), "FLASH_COMM", true);  // probe + prefill transfer path (before the workers)
    e->plans = make_plan(e->ng, wts);
    std::vector<int> draft_ids;
    if (FILE* f = fopen((model + "/token_rank.bin").c_str(), "rb")) {
        draft_ids.resize(65536);
        draft_ids.resize(fread(draft_ids.data(), 4, 65536, f));
        fclose(f);
    }
    std::vector<std::vector<int>> cold;
    const std::string cf = env_str("FLASH_COLD") ? env_str("FLASH_COLD") : model + "/cold_experts.json";
    if (FILE* f = fopen(cf.c_str(), "r")) {
        std::string js;
        char buf[65536];
        size_t n;
        while ((n = fread(buf, 1, sizeof(buf), f)) > 0) js.append(buf, n);
        fclose(f);
        Json j = Json::parse(js);
        for (auto& l : j["layers"].arr) {
            std::vector<int> v;
            for (auto& x : l.arr) v.push_back((int)x.i());
            cold.push_back(v);
        }
    }
    std::vector<float> prior;  // expert-cache seed: per-token routing probabilities [48][512] (tools/flash_expert_prior.py)
    {
        const std::string pf = env_str("FLASH_EC_PRIOR") ? env_str("FLASH_EC_PRIOR") : model + "/expert_prior.bin";
        if (FILE* f = fopen(pf.c_str(), "rb")) {
            prior.resize((size_t)LAYERS * NE);
            if (fread(prior.data(), 4, prior.size(), f) != prior.size()) prior.clear();
            fclose(f);
        }
    }
    e->W.resize(e->ng);
    {
        std::vector<std::thread> th;
        for (int r = 0; r < e->ng; r++)
            th.emplace_back([&, r] {
                e->W[r] = new Worker(e->plans[r], e->devs[r], e->comm, e->ctx);
                e->W[r]->set_cold(cold);
                e->W[r]->set_expert_prior(prior);
                e->W[r]->set_draft_vocab(draft_ids);
                const double a = secs();
                e->W[r]->load(e->st, e->ht, &e->mst);
                const double b = secs();
                // the expert budget (alloc_state) reads the free device memory: after the host registrations
                ng_done.wait();
                const double c = secs();
                e->W[r]->alloc_state();
                fprintf(stderr, "[qflash %.0fs] GPU%d: weights %.1f s, wait for the n-gram table %.1f s, state + expert groups %.1f s\n",
                        secs(), e->devs[r], b - a, c - b, secs() - c);
            });
        for (auto& t : th) t.join();
    }
    fprintf(stderr, "[qflash %.0fs] weights loaded\n", secs());
    if (!e->steer.empty()) {  // activation steering vectors (validated before the load)
        const int nv = e->steer.size();
        std::vector<int> lo(nv), hi(nv);
        std::vector<float> w;
        for (int v = 0; v < nv; v++) {
            lo[v] = e->steer[v].flo; hi[v] = e->steer[v].fhi;
            w.insert(w.end(), e->steer[v].w.begin(), e->steer[v].w.end());
            fprintf(stderr, "%s\n", e->steer[v].log_line().c_str());
        }
        for (int r = 0; r < e->ng; r++) e->W[r]->set_steer(nv, lo.data(), hi.data(), w.data());
        e->steer_json = steer_info_json(e->steer, e->steer_def);
        for (auto& x : e->steer) x.w.clear();
        fprintf(stderr, "steering: %d vector(s), default %s\n", nv, e->steer_def >= 0 ? e->steer[e->steer_def].name.c_str() : "off");
    }
    ng_th.join();
    for (int i = 0; i < 128; i++) {
        if (ng8) e->ht.ngram8[i] = (const uint8_t*)ng_host[i];
        else e->ht.ngram[i] = (const bf16*)ng_host[i];
    }
    memcpy(e->ngd.mult, e->ht.mult, sizeof(e->ngd.mult));
    memcpy(e->ngd.vsz, e->ht.vsz, sizeof(e->ngd.vsz));
    memcpy(e->ngd.voff, e->ht.voff, sizeof(e->ngd.voff));
    fprintf(stderr, "[qflash %.0fs] n-gram table %s %s (%.1f GB, %.1f s, overlapped with the weight load)\n", secs(),
            ng8 ? "INT8" : "BF16", !env_int("FLASH_NGRAM_PIN", 1) ? "mapped, not pinned" : "pinned",
            ng_bytes / 1e9, ng_secs);
    for (int r = 0; r < e->ng; r++) {
        for (int g = 0; g < e->ng; g++) {
            e->W[r]->set_peer(g, e->W[g]->xrecv(), e->W[g]->xflags());
            e->W[r]->set_qpeer(g, e->W[g]->xqrecv(), e->W[g]->xqflags());
            e->W[r]->set_ipeer(g, e->W[g]->xirecv(), e->W[g]->xiflags());
            e->W[r]->set_agpeer(g, e->W[g]);
        }
        e->W[r]->set_ngram(e->ngd);
        if (!env_int("FLASH_NGRAM_PIN", 1) && !(env_int("FLASH_NGRAM_HMM", 0)))
            e->W[r]->start_ngram_stage(&e->ht);  // rows staged by a host thread (FLASH_NGRAM_HMM=1: the GPU reads the mapping)
        fprintf(stderr, "[qflash] GPU%d: %.2f GB weights+state, %.2f GB cold experts in host memory, %.2f GiB free\n", e->devs[r],
                e->W[r]->weight_gb(), e->W[r]->host_gb(), e->W[r]->free_bytes() / 1073741824.0);
    }
    if (env_set("FLASH_TEST_VRAM_HOG_MB"))
        for (int r = 0; r < e->ng; r++) e->W[r]->test_hog(env_ll("FLASH_TEST_VRAM_HOG_MB", 0));
    for (int r = 1; r < e->ng; r++)
        e->pool.emplace_back([e, r] {
            int seen = 0;
            while (true) {
                int g;
                while ((g = e->go.load(std::memory_order_acquire)) == seen && !e->quit.load()) std::this_thread::yield();
                if (e->quit.load()) return;
                seen = g;
                e->job(r);
                e->done.fetch_add(1, std::memory_order_acq_rel);
            }
        });
    e->ple.resize((size_t)MAXM * H);
    e->outs.resize(MAXM);
    for (int r = 0; r < e->ng; r++) e->W[r]->reset();
    {  // host-RAM archive of the conversations (0: off, only the live sequence is reused)
        const double gb = env_double("FLASH_KV_ARCHIVE_GB", 48.0);
        e->budget = gb > 0 ? (size_t)(gb * 1e9) : 0;
        e->ckpt_every = env_int("FLASH_CKPT", e->ckpt_every);
        e->ckpt_min = env_int("FLASH_CKPT_MIN", e->ckpt_min);
        e->end_cut = env_int("FLASH_END_CUT", e->end_cut);
        e->reserve = env_int("FLASH_SLOT_RESERVE", e->reserve);
        e->sl.assign(e->W[0]->num_slots(), Slot{});
        e->np = e->W[0]->periods();
        e->period = e->W[0]->period();
        if (env_set("FLASH_CKPT_KEEP")) e->ckpt_keep = std::max(1, env_int("FLASH_CKPT_KEEP", e->ckpt_keep));
        e->log = env_int("FLASH_SLOT_LOG", 0);
        e->batch_api = env_int("FLASH_BATCH", 1);
        e->max_pf = e->batch_api ? std::max(1, std::min(Worker::PFSEG_MAX, env_int("FLASH_MAX_PREFILLS", 8))) : 1;
        const bool pin = env_int("FLASH_ARCHIVE_PINNED", 1);
        const double head = env_double("FLASH_ARCHIVE_HEADROOM_GB", 4.0);
        if (e->budget && pin) e->hpool.start(e->budget, (size_t)1 << 30, (size_t)(head * 1e9));
        fprintf(stderr, "[qflash] %zu sequence slots over a pool of %d periods; archive: %.1f GB host budget%s, checkpoints every %d tokens\n",
                e->sl.size(), e->np, e->budget / 1e9, pin ? " (pinning in the background)" : " (pageable)", e->ckpt_every);
    }
    fprintf(stderr, "[qflash %.0fs] ready, context %d\n", secs(), e->ctx);
    return e;
}

void qe_destroy(void* p) {
    FE* e = (FE*)p;
    if (!e) return;
    e->drain();
    e->quit.store(true);
    for (auto& t : e->pool) t.join();
    // GPU memory is released at process exit
}

int qe_max_context(void* p) { return ((FE*)p)->ctx; }
int qe_vocab(void*) { return VOCAB; }

// activation steering of the next start (QE_STEER / QE_STEER_DIR): vector, scale, mode and layers. Sticky until the
// next call; the server calls it before every start. A conversation keeps the key it started with (prefix reuse,
// checkpoints and the archive never mix KV of different (vector, mode, layers, scale): a changed request starts a
// fresh KV). vec: index in qe_steer_info's list, -1 off, -2 the server default (its defaults, scale 1); scale
// multiplies the mode's sidecar scale; mode -1 / 0 add / 1 project and lo, hi (-1, -1: the mode's default range;
// else inside the vector's file layers). Returns 0, or -1 for a bad value (nothing changed).
int qe_set_steering_ex(void* p, int vec, float scale, int mode, int lo, int hi) {
    FE* e = (FE*)p;
    if (vec == -2) { vec = e->steer_def; scale = 1.f; mode = -1; lo = hi = -1; }
    if (vec < -1 || vec >= (int)e->steer.size() || !std::isfinite(scale)) return -1;
    if (vec < 0) {
        e->steer_vec = -1; e->steer_mode = 0; e->steer_lo = 0; e->steer_hi = -1; e->steer_scale = 1.f;
        return 0;
    }
    if (!steer_resolve(e->steer[vec], mode, lo, hi).empty()) return -1;
    e->steer_vec = vec;
    e->steer_mode = mode;
    e->steer_lo = lo;
    e->steer_hi = hi;
    e->steer_scale = scale;
    return 0;
}
int qe_set_steering(void* p, int vec, float scale) { return qe_set_steering_ex(p, vec, scale, -1, -1, -1); }
const char* qe_steer_info(void* p) { return ((FE*)p)->steer_json.c_str(); }

void qe_set_sampling(void* p, float temperature, float top_p, int top_k, float min_p, unsigned long long seed) {
    FE* e = (FE*)p;
    e->drain();
    // the bound slot's parameters (the server calls this right after start_begin, which binds the request's slot);
    // every slot keeps its own (per-slot sampling in batched steps, reloaded when a slot is bound)
    for (int r = 0; r < e->ng; r++) e->W[r]->set_sampling(temperature, top_p, top_k, min_p, seed);
    if (e->cs >= 0) e->S().pen = false;
}

// presence / frequency penalty of the bound slot (call after qe_set_sampling, which turns it off): the counts
// restart with the distinct seed tokens (the server's loop guard: the repeated reply), then follow the reply
void qe_set_penalty(void* p, float presence, float frequency, const int* seed, int n) {
    FE* e = (FE*)p;
    e->drain();
    for (int r = 0; r < e->ng; r++) e->W[r]->set_penalty(presence, frequency, seed, n);
    if (e->cs >= 0) e->S().pen = presence != 0.f || frequency != 0.f;
}

// lookup drafts off / on for the slot's reply (reset by the next start of the slot)
void qe_set_no_lookup(void* p, int slot, int off) {
    FE* e = (FE*)p;
    e->drain();
    if (slot < 0 || slot >= (int)e->sl.size()) return;
    e->sl[slot].nolk = off != 0;
    for (int r = 0; r < e->ng; r++) e->W[r]->set_no_lookup(slot, off != 0);
}

// a slot for a new sequence of n tokens (free, or the least valuable idle one evicted to the archive), with its region
static int acquire_slot(FE* e, int n, const std::vector<int>& protect) {
    int y = e->free_slot();
    if (y < 0) {
        for (int i = 0; i < (int)e->sl.size(); i++) {
            if (e->sl[i].active || std::find(protect.begin(), protect.end(), i) != protect.end()) continue;
            if (y < 0 || e->prio_of(i) < e->prio_of(y) || (e->prio_of(i) == e->prio_of(y) && e->sl[i].used < e->sl[y].used)) y = i;
        }
        if (y < 0) return -1;
        e->evict_slot(y);
    }
    std::vector<int> pr = protect;
    pr.push_back(y);
    if (!e->ensure_region(y, n, pr)) return -1;
    return y;
}

// Prompt start, phase 1: pick the cached start (the live end of a resident sequence, a checkpoint of any
// conversation, an archived end state), get a slot + KV region for it, copy / restore what is not in place, plan
// the prefill of the rest. Returns the slot (0 with one exposed slot) or -1 (no room / vision).
int qe_start_begin(void* p, const int* ids, int n, int k, unsigned long long affinity, int hint, const void* img,
                   int img_rows, const int* rope3, int delta) {
    FE* e = (FE*)p;
    if ((int)e->pfs.size() >= e->max_pf) return -1;  // prompts in flight
    int img_cnt = 0;
    for (int i = 0; i < n; i++) img_cnt += ids[i] >= kImgToken;
    if (img_cnt != img_rows || (img_cnt && (!img || !rope3))) {
        fprintf(stderr, "[qflash] start: %d image rows in the prompt, %d embeddings%s\n", img_cnt, img_rows,
                img_cnt && !rope3 ? ", no positions" : "");
        return -1;
    }
    e->drain();
    e->no_ahead = false;
    {  // expert cache: per-request summary of the previous generation (server log), then fresh counters
        static const bool lg = env_int("FLASH_EC_LOG", 1);
        if (lg && e->W[0]->ec_stats().dec_steps > 0) fprintf(stderr, "[qflash] %s\n", ec_summary(e->W).c_str());
        for (int r = 0; r < e->ng; r++) e->W[r]->ec_reset_stats();
    }
    if (n <= 0 || n + 2 > qe_max_context(p)) return -1;
    const long long sk = e->steer_key();  // qe_set_steering before the start (the server sets it for every request)
    e->k = std::max(0, std::min(k, std::min(e->kmax, MAXM - 1)));
    FE::Pf pf;
    pf.t_begin = FE::now_ms();
    FE::Src s = e->best(ids, n, false);
    if (!e->budget && s.c && !s.live) s = FE::Src{};  // archive off: only a live sequence is reused
    e->hold = s.c;
    int L = 0, y = -1;
    e->last_copied = 0;
    e->last_from_archive = false;
    auto new_conv = [&](const FE::Src* from) -> Conv* {
        Conv d;
        d.id = e->next_id++;
        d.steer = sk;
        if (from) {
            for (auto& g : from->c->segs)
                if (g->t0 < from->L) d.segs.push_back(g);
            d.kv_n = std::min(from->c->kv_n, from->L);
            for (auto& c : from->c->ck)
                if (c->pos <= from->L) d.ck.push_back(c);
            d.prio = from->c->prio;
        }
        e->convs.push_back(std::move(d));
        return &e->convs.back();
    };
    Conv* target = nullptr;
    if (s.c && s.live) {  // the live end of a resident sequence: in place
        y = s.c->slot;
        L = s.L;
        target = s.c;
        if (!e->ensure_region(y, n, {y})) return e->hold = nullptr, -1;
        e->bind(y);
        // the live end before the turn extends it stays a checkpoint (a later fork of the same history, e.g. Claude
        // Code's progress summary, or the conversation's own next turn after such a fork, resumes from it): the end
        // state archived in idle time when there is one, else saved now (~70 MB per card)
        if (e->budget) {
            Slot& q0 = e->sl[y];
            std::shared_ptr<State> st;
            if (q0.synced && s.c->end && s.c->end->pos == (int)q0.hist.size()) st = s.c->end;
            else st = e->save_state(y, true, 4);
            if (st) {
                st->kind = 4;
                e->add_ck(s.c, st);
            }
        }
    } else if (s.c && s.c->slot >= 0) {  // a checkpoint of a resident sequence: its KV rows copied D2D into a new slot
        const int x = s.c->slot;
        const double t0 = FE::now_ms();
        y = acquire_slot(e, n, {x});
        L = s.L;
        if (y < 0) {  // no other slot (e.g. FLASH_MAX_SEQS=1): in place, the slot's conversation stays archived only
            if (e->sl[x].active) return e->hold = nullptr, -1;
            if (e->budget && !e->sl[x].synced) e->sync(x);
            y = x;
            if (!e->ensure_region(y, n, {y})) return e->hold = nullptr, -1;
            e->bind(y);
        } else {
            const int fb = e->sl[x].base, tb = e->sl[y].base;
            e->bind(y);
            e->job = [&](int r) { e->W[r]->kv_claim(L); e->W[r]->kv_move(fb, tb, L); };
            e->run();
        }
        const bool pend = s.st->pend;
        e->xfer(*s.st, [&](int r, std::vector<Worker::Piece>& ps) { e->W[r]->state_pieces_of(y, pend, ps); }, false);
        for (int r = 0; r < e->ng; r++) e->W[r]->set_slot_prev_m(y, pend ? s.st->prev_m : 0);
        if (y == x) e->sl[y].hist.resize(L);
        else e->sl[y].hist.assign(e->sl[x].hist.begin(), e->sl[x].hist.begin() + L);
        target = new_conv(&s);
        if (y == x) {  // the old conversation left the slot
            Conv* old = s.c;
            old->slot = -1;
            if (!(old->end || !old->ck.empty()) || !e->budget) e->forget(old);
            e->sl[y].conv = nullptr;
        }
        e->last_copied = L;
        e->n_restore++;
        pf.ms_restore = FE::now_ms() - t0;
    } else if (s.c) {  // archived: KV segments + state from host memory
        const double t0 = FE::now_ms();
        y = acquire_slot(e, n, {});
        if (y < 0) return e->hold = nullptr, -1;
        pf.ms_pool = FE::now_ms() - t0;
        L = s.L;
        e->bind(y);
        std::vector<Seg*> use;
        for (auto& g : s.c->segs)
            if (g->t0 < L) use.push_back(g.get());
        e->job = [&](int r) { e->W[r]->kv_claim(L); };
        e->run();
        const int b = e->sl[y].base;
        // only the positions < L: a segment can reach far past them (a conversation archived at 138K, a 25K prompt
        // resuming at its system-end checkpoint) and the new region is sized for this prompt only; the whole segment
        // wrote past the region's end into the next slot's KV / live expert-cache slots (or past the pool)
        for (Seg* g : use) e->xfer_part(*g, b, g->t0, g->t1, L);
        const bool pend = s.st->pend;
        e->xfer(*s.st, [&](int r, std::vector<Worker::Piece>& ps) { e->W[r]->state_pieces_of(y, pend, ps); }, false);
        for (int r = 0; r < e->ng; r++) e->W[r]->set_slot_prev_m(y, pend ? s.st->prev_m : 0);
        e->sl[y].hist.assign(s.c->toks.begin(), s.c->toks.begin() + L);
        if (s.st == s.c->end) {  // the whole conversation continues
            target = s.c;
            // as on the live path: the end before this turn extends it stays a checkpoint (a reply cut by the caller
            // before the engine's last step leaves an end state that is not a prefix of the next turn; the next turn
            // then resumes here instead of from scratch)
            if (s.st->kind != 4) {
                s.st->kind = 4;
                e->add_ck(s.c, s.st);
            }
        } else {
            target = new_conv(&s);
        }
        e->last_copied = L;
        e->last_from_archive = true;
        e->n_restore++;
        pf.ms_restore = FE::now_ms() - t0 - pf.ms_pool;
    } else {
        y = acquire_slot(e, n, {});
        if (y < 0) return e->hold = nullptr, -1;
        e->bind(y);
        e->job = [&](int r) { e->W[r]->reset_slot(y); };
        e->run();
        e->sl[y].hist.clear();
        target = new_conv(nullptr);
    }
    e->hold = nullptr;
    Slot& q = e->sl[y];
    if (q.conv != target) q.synced = false;
    q.conv = target;
    target->slot = y;
    q.dirty = false;
    q.bhist = -1;
    q.synced = false;
    q.used = target->used = ++e->tick;
    target->affinity = affinity;
    if (!e->budget) {  // archive off: forget the conversations without a slot
        for (auto it = e->convs.begin(); it != e->convs.end();)
            it = it->slot < 0 ? e->convs.erase(it) : std::next(it);
        target->ck.clear();
    }
    e->last_reused = L;
    pf.reused = L;
    pf.copied = e->last_copied;
    pf.from_archive = e->last_from_archive;
    pf.on = true;
    pf.slot = y;
    q.rb_pos = -1;
    q.want_drafts = false;
    // a new request starts unconstrained, also on the devices (evict_slot / qe_reset clear only the host flag)
    q.con = false;
    q.pen = false;  // set_sampling (next) turns the penalty off; qe_set_penalty after it
    if (q.nolk) {
        q.nolk = false;
        for (int r = 0; r < e->ng; r++) e->W[r]->set_no_lookup(y, false);
    }
    // the slot's steering flag follows its conversation's key (device flag at a fixed address: no graph changes)
    int sv, smd, slo, shi;  // the slot's steering follows its conversation's key
    float ssc;
    FE::steer_unkey(target->steer, sv, smd, slo, shi, ssc);
    e->job = [&](int r) { e->W[r]->set_constrained(y, false); e->W[r]->set_steer_slot(y, sv, smd, slo, shi, ssc); };
    e->run();
    q.active = true;  // taken (not evicted by another start) until released
    q.paused = false;
    if (e->W[0]->kv_stream()) e->update_top();  // KV in RAM: block-cache entries sized to the active regions
    q.admit = ++e->admit_tick;
    pf.ids.assign(ids, ids + n);
    pf.n = n;
    pf.i = L;
    pf.hint = hint;
    // multimodal prompt: image rows into the image table (after the rows of other image prompts in flight), device
    // ids, positions; the slot's RoPE shift (0: text) and no chunk table of an earlier prompt
    if (img_cnt) {
        int base = 0;
        for (auto& x : e->pfs)
            if (x.img_n) base = std::max(base, x.img0 + x.img_n);
        if ((size_t)(base + img_cnt) > e->img_cap) {
            const size_t cap = std::max<size_t>(base + img_cnt, 2 * e->img_cap);
            bf16* t = nullptr;
            CUDA_CHECK(cudaSetDevice(e->devs[0]));
            CUDA_CHECK(cudaHostAlloc((void**)&t, cap * H * sizeof(bf16), cudaHostAllocMapped | cudaHostAllocPortable));
            if (base) memcpy(t, e->img, (size_t)base * H * sizeof(bf16));
            e->job = [&](int r) { e->W[r]->set_image_table(t); };
            e->run();
            if (e->img) CUDA_CHECK(cudaFreeHost(e->img));
            e->img = t;
            e->img_cap = cap;
        }
        memcpy(e->img + (size_t)base * H, img, (size_t)img_cnt * H * sizeof(bf16));
        pf.dev.assign(ids, ids + n);
        for (int i = 0, j = 0; i < n; i++)
            if (ids[i] >= kImgToken) pf.dev[i] = kImgToken + base + j++;
        pf.r3.assign(rope3, rope3 + 3 * (size_t)n);
        pf.img0 = base;
        pf.img_n = img_cnt;
    } else {
        delta = 0;
    }
    e->job = [&](int r) { e->W[r]->set_rope_slot(y, delta); e->W[r]->set_rope_chunk(y, 0, 0, nullptr); };
    e->run();
    if (e->budget) {  // checkpoints of the committed state: the system end (hint), every ckpt_every, the prompt end - 1
        std::vector<int> c;
        if (hint > L && hint < n && hint >= e->ckpt_min) c.push_back(hint);
        // the divergence point from a known sequence (a third prompt sharing it resumes there)
        if (s.maxm > L + e->ckpt_min && s.maxm < n - 1) c.push_back(s.maxm);
        if (e->ckpt_every > 0)
            for (int q2 = (L / e->ckpt_every + 1) * e->ckpt_every; q2 < n - 1; q2 += e->ckpt_every) c.push_back(q2);
        if (n - 1 - L >= e->end_cut) c.push_back(n - 1);  // a shorter rest is cheaper to redo than to split off
        std::sort(c.begin(), c.end());
        c.erase(std::unique(c.begin(), c.end()), c.end());
        pf.cks = c;
    }
    e->pfs.push_back(std::move(pf));
    e->last_pf_slot = y;
    return e->batch_api ? y : 0;
}

static void take_ck(FE* e, const FE::Pf& pf, int pos) {
    const int kind = pos == pf.hint ? 1 : pos == pf.n - 1 ? 2 : pos % e->ckpt_every ? 5 : 3;  // 5: divergence point
    e->add_ck(e->sl[pf.slot].conv, e->save_state(pf.slot, false, kind));  // (the prompt's slot: qe_prefill_multi)
}

// Prompt start, phase 2: prefill up to `budget` more tokens (batches); 1 when the prompt is done
int qe_prefill_more_slot(void* p, int slot, int budget) {
    FE* e = (FE*)p;
    FE::Pf* pp = e->pf_of(slot);
    if (!pp) return 1;
    auto& pf = *pp;
    if (e->cs != pf.slot) e->bind(pf.slot);
    {
        const unsigned* mh = e->S().con ? e->mh(e->cs) : nullptr;  // constrained: mask 0 serves the prompt's last chunk
        e->job = [&](int r) { e->W[r]->ws_ensure(); if (mh) e->W[r]->upload_mask(mh, 1); };  // after batched steps the slot's state is in its storage
        e->run();
    }
    const int n = pf.n;
    const int* ids = pf.ids.data();
    const int* dids = pf.dev.empty() ? ids : pf.dev.data();  // what the devices embed (image rows: table rows)
    const int* r3 = pf.r3.empty() ? nullptr : pf.r3.data();
    const int pslot = pf.slot;
    auto& hist = e->S().hist;
    static const bool pfb = !(env_int("FLASH_PFB", 1) == 0);
    // big batches (past pf_batch_base(), buffers borrowed from the expert cache) only when the chunk budget asks for
    // them: the server's 2048-token chunks keep the base batch
    const int B = pfb ? (budget > e->W[0]->pf_batch_base() - PFM ? e->W[0]->pf_batch() : e->W[0]->pf_batch_base()) : 0;
    int done_tok = 0;
    while (pf.i < n && done_tok < budget) {
        const int i = pf.i;
        int m;
        if (pfb) m = e->W[0]->pf_batch_rows(n - i, B);  // nominal batch B - PFM; a tail of <= PFM rows joins the last one
        else m = std::min(PFM, n - i);
        while (!pf.cks.empty() && pf.cks.front() <= i) pf.cks.erase(pf.cks.begin());
        if (!pf.cks.empty() && pf.cks.front() < i + m) m = pf.cks.front() - i;  // a batch ends at the next checkpoint
        std::vector<bf16> rows((size_t)m * H);
        for (int j = 0; j < m; j++) hist.push_back(ids[i + j]);
        ngram_rows(e->ht, hist, (int)hist.size() - m, m, rows.data());
        const bool lastb = i + m >= n;
        int o = 0, d = 0;
        if (pfb) {  // batched prefill (MoE half over the whole batch), then the MTP fill
            std::vector<int> mt;
            if (e->k > 0)
                for (int j = 0; j < m; j++) mt.push_back(i + j + 1 < n ? dids[i + j + 1] : -1);  // last: the sampled token (device)
            e->job = [&](int r) {
                int oo, dd = -2;
                if (r3) e->W[r]->set_rope_chunk(pslot, i, m, r3);
                e->W[r]->prefill_batch(dids + i, i, m, rows.data(), &oo, e->k > 0 ? mt.data() : nullptr, lastb, &dd);
                if (r == 0) { o = oo; d = dd; }
            };
            e->run();
            if (e->k > 0) {
                if (d == -2) {  // MTP fill not in the batch (one chunk): separate pass
                    if (lastb) mt[m - 1] = o;
                    e->job = [&](int r) { int dd; e->W[r]->prefill_batch_mtp(mt.data(), i, m, &dd); if (r == 0) d = dd; };
                    e->run();
                }
                if (lastb) pf.drafts = {d};
            }
        } else {  // prefill chunks (rows committed at once), MTP KV in the same pass
            std::vector<int> ch(dids + i, dids + i + m);
            std::vector<int> mt;
            for (int j = 0; j < m; j++) mt.push_back(i + j + 1 < n ? dids[i + j + 1] : -1);
            if (e->k > 0 && !lastb) {
                e->job = [&](int r) { if (r3) e->W[r]->set_rope_chunk(pslot, i, m, r3); int oo, dd; e->W[r]->prefill(ch.data(), i, m, rows.data(), &oo, mt.data(), &dd); if (r == 0) { o = oo; d = dd; } };
                e->run();
            } else {
                e->job = [&](int r) { if (r3) e->W[r]->set_rope_chunk(pslot, i, m, r3); int oo; e->W[r]->prefill(ch.data(), i, m, rows.data(), &oo, nullptr, nullptr); if (r == 0) o = oo; };
                e->run();
                if (e->k > 0) {  // the last row's MTP token is the first generated one
                    mt[m - 1] = o;
                    e->job = [&](int r) { int dd; e->W[r]->prefill_mtp(mt.data(), i, m, &dd); if (r == 0) d = dd; };
                    e->run();
                }
            }
            if (lastb && e->k > 0) pf.drafts = {d};
        }
        pf.nxt = o;
        pf.i = i + m;
        done_tok += m;
        if (!pf.cks.empty() && pf.cks.front() == pf.i) {
            take_ck(e, pf, pf.i);
            pf.cks.erase(pf.cks.begin());
        }
    }
    return pf.i >= n ? 1 : 0;
}
int qe_prefill_more(void* p, int budget) { return qe_prefill_more_slot(p, -1, budget); }  // the newest prompt

// Several prompts in flight, one batch (batched-prefill): rows of the prompts on `slots` (in that order: the first get their
// rows first) up to `budget` rows (big batches as qe_prefill_more_slot when the budget asks for them). Each prompt's rows
// end at its next checkpoint, as on its own; a long prompt takes whole PFM chunks of what is left. The mixer runs per
// prompt, the MoE half once for all (each expert group read once per batch instead of once per prompt). Text prompts
// only (image rows / a RoPE shift, constrained requests: their own batches). done[i] = 1: prompt i is complete
// (qe_start_finish_slot next). Returns how many prompts got rows; 0 = no shared batch (fewer than 2 prompts fit, or
// FLASH_PF_MULTI=0): the caller prefills them one by one.
// 1: prefill passes are bound by copying the host-backed experts (the copies of one pass take longer than a base batch
// computes: > ~8 GB per card at ~25 GB/s, 2 cards; 3 cards hold <= ~5 GB host-backed): the server then runs the large
// batch whenever nothing decodes, also while requests wait (2 GPUs, 8 x 16K burst: +45 %; 3 GPUs: -2 % on average)
int qe_prefill_pass_bound(void* p) {
    FE* e = (FE*)p;
    long long mx = 0;
    for (int r = 0; r < e->ng; r++) mx = std::max(mx, e->W[r]->cold_bytes());
    return mx > (8LL << 30) ? 1 : 0;
}

int qe_prefill_multi(void* p, const int* slots, int nslots, int budget, int* done) {
    FE* e = (FE*)p;
    for (int i = 0; i < nslots; i++) done[i] = 0;
    static const bool on = env_int("FLASH_PF_MULTI", 1) != 0;
    static const bool pfb = !(env_int("FLASH_PFB", 1) == 0);
    if (!on || !pfb || !e->batch_api || nslots < 2) return 0;
    Worker& w0 = *e->W[0];
    const int B = budget > w0.pf_batch_base() - PFM ? w0.pf_batch() : w0.pf_batch_base();
    const int T = std::min(budget, B - PFM);  // nominal rows; a prompt that ends within B may take its tail too
    static const bool bound = qe_prefill_pass_bound(p) != 0;
    struct Pick { FE::Pf* pf; int idx, m; };
    std::vector<Pick> pk;
    int used = 0;
    for (int i = 0; i < nslots && (int)pk.size() < Worker::PFSEG_MAX; i++) {
        FE::Pf* pf = e->pf_of(slots[i]);
        if (!pf || pf->i >= pf->n || !pf->r3.empty() || pf->img_n || w0.rope_shift(pf->slot) || e->sl[pf->slot].con) continue;
        while (!pf->cks.empty() && pf->cks.front() <= pf->i) pf->cks.erase(pf->cks.begin());
        int want = pf->n - pf->i;
        if (!pf->cks.empty() && pf->cks.front() < pf->i + want) want = pf->cks.front() - pf->i;  // ends at the checkpoint
        const bool ends = pf->i + want >= pf->n;
        int m;
        if (used + want <= T || (ends && used + want <= B && want <= PFM + std::max(0, T - used))) m = want;
        else m = std::min(want, (T - used) / PFM * PFM);  // a part of a long prompt: whole chunks
        // a long prompt cut by the batch budget joins others only where passes are copy-bound (2 cards: packing saves
        // a whole pass); elsewhere it runs alone on its graph-cached batch (a shared batch with a cut part has a new
        // layout almost every time and runs eagerly: 3 GPUs 8 x 4K bursts -5..-9 %)
        if (m < want && !pk.empty() && !bound) continue;
        if (m <= 0) continue;
        pk.push_back({pf, i, m});
        used += m;
        if (used >= T) break;
    }
    // the layout must suit the collectives (units of >= 3 x cards rows ...): drop prompts from the end until it does
    while (pk.size() >= 2) {
        std::vector<int> sn;
        for (auto& x : pk) sn.push_back(x.m);
        if (w0.pf_multi_ok(sn.data(), (int)sn.size())) break;
        if (env_int("FLASH_PF_MULTI_LOG", 0)) {
            std::string d;
            for (int x : sn) d += " " + std::to_string(x);
            fprintf(stderr, "[qflash] prefill batch layout refused:%s\n", d.c_str());
        }
        pk.pop_back();
    }
    static const bool lg = env_int("FLASH_PF_MULTI_LOG", 0) != 0;
    if (pk.size() < 2) {
        if (lg) fprintf(stderr, "[qflash] prefill batch: %zu of %d prompts fit (budget %d, B %d)\n", pk.size(), nslots, budget, B);
        return 0;
    }
    e->drain();
    e->no_ahead = false;
    const int ns = (int)pk.size();
    int n = 0;
    for (auto& x : pk) n += x.m;
    std::vector<Worker::PfSeg> sg(ns);
    std::vector<int> toks(n), mt;
    std::vector<bf16> rows((size_t)n * H);
    if (e->k > 0) mt.resize(n);
    int r0 = 0;
    for (int j = 0; j < ns; j++) {
        FE::Pf& pf = *pk[j].pf;
        const int i = pf.i, m = pk[j].m;
        auto& hist = e->sl[pf.slot].hist;
        for (int t = 0; t < m; t++) hist.push_back(pf.ids[i + t]);
        ngram_rows(e->ht, hist, (int)hist.size() - m, m, rows.data() + (size_t)r0 * H);
        for (int t = 0; t < m; t++) {
            toks[r0 + t] = pf.ids[i + t];
            if (e->k > 0) mt[r0 + t] = i + t + 1 < pf.n ? pf.ids[i + t + 1] : -1;  // last: the sampled token (device)
        }
        sg[j] = {pf.slot, e->sl[pf.slot].base, i, m, e->W[0]->slot_prev_m(pf.slot), i + m >= pf.n};
        r0 += m;
    }
    std::vector<int> outs(ns, -1), drs(ns, -1);
    e->job = [&](int r) {
        std::vector<int> o(ns), d(ns);
        e->W[r]->prefill_multi(sg.data(), ns, toks.data(), rows.data(), e->k > 0 ? mt.data() : nullptr, o.data(), d.data());
        if (r == 0) { outs = o; drs = d; }
    };
    e->run();
    int k = 0;
    for (int j = 0; j < ns; j++) {
        FE::Pf& pf = *pk[j].pf;
        pf.i += pk[j].m;
        pf.n_multi++;
        if (sg[j].end) {
            pf.nxt = outs[k];
            if (e->k > 0) pf.drafts = {drs[k]};
            pf.multi_end = true;
            k++;
            done[pk[j].idx] = 1;
        }
        if (!pf.cks.empty() && pf.cks.front() == pf.i) {
            take_ck(e, pf, pf.i);
            pf.cks.erase(pf.cks.begin());
        }
    }
    if (lg) {
        std::string d;
        for (auto& x : pk) d += " " + std::to_string(x.pf->slot) + ":" + std::to_string(x.m);
        fprintf(stderr, "[qflash] prefill batch of %d prompts, %d rows (slot:rows%s)\n", ns, n, d.c_str());
    }
    return ns;
}

// device speculative state of the bound slot from its host state (after a prefill: nothing pending)
static void spec_start(FE* e, int nxt, const std::vector<int>& first_drafts, const int* ids, int n) {
    if (e->k > 0) {
        std::vector<int> drafts = first_drafts;
        for (int j = 1; j < e->k; j++) drafts.push_back(e->mtp({drafts.back()}, n + j - 1, true));
        for (int r = 0; r < e->ng; r++) {  // prefill committed its rows: nothing pending
            e->W[r]->spec_init(nxt, drafts.data(), e->k, n, 0, ids[n - 1], n >= 2 ? ids[n - 2] : EOS);
            e->W[r]->sv_set_live();
        }
        e->lk_next = 0;
        e->prev_lk = false;
        if (e->W[0]->lookup_on()) {  // history = prompt + the pending token
            std::vector<int> h(ids, ids + n);
            h.push_back(nxt);
            e->job = [&](int r) { e->W[r]->set_history(h.data(), (int)h.size(), e->k); };
            e->run();
            e->lk_next = e->W[0]->lookup_drafts();
        }
    }
}

// Prompt start, phase 3: the first token (and the drafts of the first step)
int qe_start_finish_slot(void* p, int slot) {
    FE* e = (FE*)p;
    FE::Pf* pp = e->pf_of(slot);
    if (!pp) return -1;
    auto& pf = *pp;
    if (pf.i < pf.n) qe_prefill_more_slot(p, slot, 1 << 30);
    if (e->cs != pf.slot) e->bind(pf.slot);
    const int n = pf.n;
    for (int r = 0; r < e->ng; r++) e->W[r]->nf_check("prefill", n);
    if (e->take_nf()) {  // the prompt's forward is garbage: this request fails (qe_slot_fault), the slot is freed on release
        e->fault(pf.slot, "prefill", n);
        e->last_pf_slot = pf.slot;
        for (auto it = e->pfs.begin(); it != e->pfs.end(); ++it)
            if (&*it == pp) { e->pfs.erase(it); break; }
        return -2;
    }
    if (pf.multi_end && e->k > 0) {  // the MTP chain continues from its prompt end (kept per slot by the shared batch)
        e->job = [&](int r) { e->W[r]->rmin_load(pf.slot); };
        e->run();
    }
    spec_start(e, pf.nxt, pf.drafts, pf.ids.data(), n);
    Slot& q = e->S();
    q.pending = pf.nxt;
    if (q.pen) {  // the first token (sampled at the prompt end) sits at position n
        const int t = pf.nxt;
        e->job = [&](int r) { e->W[r]->pen_host(&t, 1, n); };
        e->run();
    }
    q.active = true;
    q.remain = -1;
    if (q.con) {  // the drafts made at the prompt end, for the caller to check
        int* io = e->sio(e->cs);
        io[16] = 0;
        io[0] = std::min<int>(pf.drafts.size(), 15);
        for (int i = 0; i < io[0]; i++) io[1 + i] = pf.drafts[i];
    }
    q.lk_stale = false;
    q.synced = false;
    static const bool pft = env_str("FLASH_PFT") || (env_int("FLASH_SLOT_LOG", 0));
    if (pft)
        fprintf(stderr, "[qflash] start: %d tokens (reused %d, %s) in %.1f ms (pool %.1f, restore %.1f), slot %d [%d, %d), conversation %d, %zu known, %.2f GB host\n",
                n, pf.reused, pf.from_archive ? "archive" : pf.copied ? "checkpoint" : pf.reused ? "live" : "none",
                FE::now_ms() - pf.t_begin, pf.ms_pool, pf.ms_restore, e->cs, q.base, q.base + q.cap, q.conv->id, e->convs.size(),
                e->hpool.used() / 1e9);
    const int nxt = pf.nxt;
    e->last_pf_slot = pf.slot;  // qe_last_slot / qe_last_reused / source after a finish: that prompt's
    e->last_reused = pf.reused;
    e->last_copied = pf.copied;
    e->last_from_archive = pf.from_archive;
    for (auto it = e->pfs.begin(); it != e->pfs.end(); ++it)
        if (&*it == pp) { e->pfs.erase(it); break; }
    return nxt;
}
int qe_start_finish(void* p) { return qe_start_finish_slot(p, -1); }
int qe_num_prefills(void* p) { return (int)((FE*)p)->pfs.size(); }
int qe_prefill_pos(void* p, int slot) { FE* e = (FE*)p; const FE::Pf* x = e->pf_of(slot); return x ? x->i : -1; }

// a prompt with image rows (qe_start_begin + qe_start_finish): ids >= kImgToken are image rows (content-derived ids:
// the prefix cache / archive tell images apart), img their embeddings [img_rows][2560] bf16 in prompt order, rope3 the
// (t, h, w) M-RoPE position of every prompt row, delta the position shift of the text after the prompt
int qe_start_mm(void* p, const int* ids, int n, int k, unsigned long long affinity, int hint, const void* img, int img_rows,
                const int* rope3, int delta) {
    const int y = qe_start_begin(p, ids, n, k, affinity, hint, img, img_rows, rope3, delta);
    if (y < 0) return -1;
    return qe_start_finish_slot(p, y);
}
int qe_start3(void* p, const int* ids, int n, int k, unsigned long long affinity, int hint) {
    const int y = qe_start_begin(p, ids, n, k, affinity, hint, nullptr, 0, nullptr, 0);
    if (y < 0) return -1;
    return qe_start_finish_slot(p, y);  // the slot this start took (with several slots exposed: not always 0)
}

// the next step's shape: lookup drafts (found on the device by the last step) or an MTP step; false: no room
static bool next_step(FE* e, int& ks, bool& lk, bool ahead = false) {
    lk = e->k > 0 && e->lk_next > 0 && !e->S().nolk;
    ks = lk ? e->lk_next : e->k;  // after a lookup step the chain ran on the device when needed
    const int hn = e->S().hist.size();
    if (lk && hn + ks + 2 > e->ctx) { lk = false; ks = e->k; }
    if (hn + ks + 2 > e->ctx) return false;
    // the step's rows must lie in the slot's region (grows / moves / evicts idle slots; nothing is in flight here);
    // a step launched ahead never pauses another sequence
    return ahead ? e->ensure_region(e->cs, hn + ks + 2, {e->cs}) : e->room_or_pause(e->cs, hn + ks + 2, {e->cs});
}

static int spec_step_impl(void* p, int* out);
int qe_spec_step(void* p, int* out) {
    FE* e = (FE*)p;
    if (e->S().fault) return 0;
    const int n = spec_step_impl(p, out);
    if (e->take_nf()) {  // the step's rows (the bound slot's) had non-finite values: no tokens, the request fails
        e->fault(e->cs, "decode step", (int)e->S().hist.size());
        return 0;
    }
    return n;
}
static int spec_step_impl(void* p, int* out) {
    FE* e = (FE*)p;
    Slot& q = e->S();
    if (q.pending < 0) return 0;
    if (q.paused && !e->unpause_slot(e->cs, {e->cs})) return 0;  // stepped alone: resumes as soon as its region fits
    q.synced = false;
    if (e->k == 0) {
        if ((int)q.hist.size() + 2 > qe_max_context(p) || !e->room_or_pause(e->cs, (int)q.hist.size() + 2, {e->cs})) return 0;
        e->job = [&](int r) { e->W[r]->ws_ensure(); };
        e->run();
        const int pos = q.hist.size();
        const int t = e->forward({q.pending}, pos);
        if (q.pen) {
            e->job = [&](int r) { e->W[r]->pen_host(&t, 1, pos + 1); };
            e->run();
        }
        q.pending = t;
        out[0] = t;
        q.count_step(false, 0, 1);
        return 1;
    }
    if (q.con) {  // grammar-constrained: the caller's drafts (spec_io[16..]) verified with its masks, no lookahead / lookup
        e->drain();
        int* io = e->sio(e->cs);
        const unsigned* mhb = e->mh(e->cs);
        const int kv = std::max(0, std::min({io[16], e->k, 15}));
        const int hn = q.hist.size(), kk = std::max(kv, e->k);
        if (hn + kk + 2 > e->ctx || !e->room_or_pause(e->cs, hn + kk + 2, {e->cs})) return 0;
        e->job = [&](int r) {
            e->W[r]->sv_load();
            e->W[r]->upload_mask(mhb, kv + 1);
            e->W[r]->upload_drafts(io + 17, kv);
            e->W[r]->spec_launch_con(kv, e->k);
        };
        e->run();
        std::vector<int> em(MAXM + 1);
        int cnt = 0;
        e->job = [&](int r) {
            std::vector<int> v(MAXM + 1);
            const int c = e->W[r]->spec_finish(v.data());
            if (r == 0) { em = v; cnt = c; io[0] = e->W[0]->next_drafts(e->cs, e->k, io + 1); }
        };
        e->run();
        e->prev_lk = false;
        e->lk_next = 0;
        q.lk_stale = true;
        q.rb_pos = (int)q.hist.size();
        q.rb_rows = cnt;
        q.hist.push_back(q.pending);
        for (int i = 0; i + 1 < cnt; i++) q.hist.push_back(em[i]);
        q.pending = em[cnt - 1];
        for (int r = 0; r < e->ng; r++) e->W[r]->set_prev_m(cnt);
        for (int i = 0; i < cnt; i++) out[i] = em[i];
        if (q.remain >= 0) q.remain = std::max(0, q.remain - cnt);
        q.count_step(false, kv, cnt);
        return cnt;
    }
    if (!e->inflight) {  // the slot's speculative state may be in its stash (batched steps / a switch)
        e->job = [&](int r) { e->W[r]->sv_load(); };
        e->run();
        if (q.lk_stale) {
            q.lk_stale = false;
            e->lk_next = 0;
            e->prev_lk = false;
            if (e->W[0]->lookup_on()) {
                std::vector<int> h = q.hist;
                h.push_back(q.pending);
                e->job = [&](int r) { e->W[r]->set_history(h.data(), (int)h.size(), e->k); };
                e->run();
                e->lk_next = e->W[0]->lookup_drafts();
            }
        }
    }
    int ks;
    bool lk;
    if (e->inflight) {  // launched ahead by the previous call
        ks = e->ahead_ks;
        lk = e->ahead_lk;
    } else {
        if (!next_step(e, ks, lk)) return 0;
        e->job = [&](int r) { e->W[r]->spec_launch(ks, lk); };
        e->run();
    }
    e->inflight = false;
    std::vector<int> em(MAXM + 1);
    int cnt = 0;
    e->job = [&](int r) {
        std::vector<int> v(MAXM + 1);
        const int c = e->W[r]->spec_finish(v.data());
        if (r == 0) { em = v; cnt = c; }
    };
    e->run();
    e->prev_lk = lk;
    e->lk_next = e->W[0]->lookup_drafts();
    // processed now: the previous pending token + the accepted drafts (all emitted tokens but the last)
    q.rb_pos = (int)q.hist.size();
    q.rb_rows = cnt;
    q.hist.push_back(q.pending);
    for (int i = 0; i + 1 < cnt; i++) q.hist.push_back(em[i]);
    q.pending = em[cnt - 1];
    for (int r = 0; r < e->ng; r++) e->W[r]->set_prev_m(cnt);
    for (int i = 0; i < cnt; i++) out[i] = em[i];
    if (q.remain >= 0) q.remain = std::max(0, q.remain - cnt);  // 0: exhausted (-1: no limit)
    q.count_step(lk, ks, cnt);
    // lookahead: queue the next step now, so the GPU runs it while the caller handles these tokens. Not after an
    // end token, past the reply's limit, near a stop string (qe_set_no_ahead) or while a prefill waits.
    bool eos = false;
    for (int i = 0; i < cnt; i++) eos |= em[i] == EOS || em[i] == EOS + 2;
    static const bool la_env = env_int("FLASH_LOOKAHEAD", 1);
    if (la_env && e->lookahead && !e->no_ahead && !eos && q.remain != 0) {
        int ks2;
        bool lk2;
        if (next_step(e, ks2, lk2, true)) {
            e->job = [&](int r) { e->W[r]->spec_launch(ks2, lk2); };
            e->run();
            e->inflight = true;
            e->ahead_ks = ks2;
            e->ahead_lk = lk2;
        }
    }
    return cnt;
}

// ---- batch tier: several slots in one step
void qe_set_concurrent(void*, int) {}
void qe_resume(void* p, int slot) {
    FE* e = (FE*)p;
    if (e->cs != slot) e->bind(slot);
}
// one speculative step of n active slots together: out[i * 64 ..] gets cnt[i] tokens of slots[i]
void qe_step_batch(void* p, const int* slots, int n, int* out, int* cnt) {
    FE* e = (FE*)p;
    e->drain();
    CHECK(n >= 1 && n <= MAXM, "flash step_batch: %d slots", n);
    // drafts per sequence: as the single path, at most MAXM rows in all (FLASH_BATCH_K caps them)
    static const int kcap = env_int("FLASH_BATCH_K", 64);
    const int kb = e->k > 0 ? std::max(0, std::min({e->k, MAXM / n - 1, kcap})) : 0;
    std::vector<int> run, lims, idx;
    for (int i = 0; i < n; i++)  // grammar-constrained slots step alone (their masks / drafts are per step)
        if (e->sl.at(slots[i]).con && e->sl[slots[i]].pending >= 0) {
            if (e->cs != slots[i]) e->bind(slots[i]);
            cnt[i] = qe_spec_step(p, out + i * 64);
            if (cnt[i] == 0 && !e->sl[slots[i]].fault) { out[i * 64] = EOS; cnt[i] = 1; }
        }
    e->resume_paused(slots, n);
    e->reclaim_idle();  // idle conversations out of VRAM for the expert cache (KV in VRAM, archive on)
    for (int i = 0; i < n; i++) {
        Slot& q = e->sl.at(slots[i]);
        if (q.con) continue;
        cnt[i] = 0;
        if (q.pending < 0 || q.paused) continue;
        const int hn = q.hist.size();
        if (hn + kb + 2 > e->ctx || !e->room_or_pause(slots[i], hn + kb + 2, std::vector<int>(slots, slots + n))) {
            if (q.paused) continue;  // paused for a sequence worth more: no tokens until it resumes
            // the room is held by prompts being prefilled (never paused): wait for them (no tokens yet)
            if (!e->pfs.empty() && hn + kb + 2 <= e->ctx && hn + kb + 2 <= e->np * e->period) continue;  // fits the pool alone (ensure_region caps the reserve)
            out[i * 64] = EOS;  // no room even alone (longer than the pool / the context): the sequence ends
            cnt[i] = 1;
            continue;
        }
    }
    for (int i = 0; i < n; i++) {  // (a later slot may have paused an earlier one)
        Slot& q = e->sl.at(slots[i]);
        if (q.con || q.pending < 0 || q.paused || cnt[i]) continue;
        run.push_back(slots[i]);
        lims.push_back(q.remain > 0 ? std::min(kb, q.remain - 1) : kb);
        idx.push_back(i);
    }
    if (run.empty()) return;
    if (e->W[0]->batch_lookup())  // batched lookup: the histories of slots whose tokens changed outside batched steps
        for (int s : run) {
            Slot& q = e->sl[s];
            const int want = (int)q.hist.size() + 1;
            if (q.bhist == want) continue;
            std::vector<int> h = q.hist;
            h.push_back(q.pending);
            e->job = [&](int r) { e->W[r]->set_bhist(s, h.data(), (int)h.size()); };
            e->run();
            q.bhist = want;
        }
    std::vector<int> bases(e->sl.size());
    for (size_t s = 0; s < e->sl.size(); s++) bases[s] = e->sl[s].base;
    const int nr = run.size();
    {  // FLASH_PROF_STEPS=a,b: cudaProfilerStart before batched step a, Stop before step b (nsys capture range)
        static long long bstep = 0, pa = -1, pb = -1;
        static bool init = false;
        if (!init) { init = true; if (const char* v = env_str("FLASH_PROF_STEPS")) sscanf(v, "%lld,%lld", &pa, &pb); }
        if (bstep == pa || bstep == pb) {
            if (bstep == pa) cudaProfilerStart(); else cudaProfilerStop();
        }
        bstep++;
    }
    std::vector<int> o((size_t)nr * 64), c(nr);
    e->job = [&](int r) {
        std::vector<int> oo((size_t)nr * 64), cc(nr);
        e->W[r]->batch_launch(run.data(), bases.data(), lims.data(), nr, kb);
        e->W[r]->batch_finish(oo.data(), cc.data());
        if (r == 0) { o = oo; c = cc; }
    };
    e->run();
    if (const unsigned bad = e->take_nf()) {  // rows with non-finite values: their sequences fail, the others go on
        const int rps = e->W[0]->batch_rps();
        for (int j = 0; j < nr; j++) {
            const unsigned rows = (lims[j] + 1 >= 32 ? ~0u : ((1u << (lims[j] + 1)) - 1)) << (j * rps);
            if (j * rps < 32 && (bad & rows)) e->fault(run[j], "batch step", (int)e->sl[run[j]].hist.size());
        }
    }
    for (int j = 0; j < nr; j++) {
        Slot& q = e->sl[run[j]];
        const int i = idx[j], m = c[j];
        if (q.fault) { cnt[i] = 0; continue; }
        q.rb_pos = (int)q.hist.size();
        q.rb_rows = m;
        q.hist.push_back(q.pending);
        for (int t = 0; t + 1 < m; t++) q.hist.push_back(o[(size_t)j * 64 + t]);
        q.pending = o[(size_t)j * 64 + m - 1];
        q.synced = false;
        q.lk_stale = true;
        if (q.bhist >= 0) q.bhist += m;  // the devices appended the emitted tokens
        if (q.remain >= 0) q.remain = std::max(0, q.remain - m);
        for (int t = 0; t < m; t++) out[i * 64 + t] = o[(size_t)j * 64 + t];
        cnt[i] = m;
        q.count_step(false, lims[j], m);
    }
}

// draft statistics of a slot since it was created: {steps, MTP drafted, MTP accepted, lookup drafted, lookup accepted}
// (the server takes the difference over a request)
void qe_draft_stats(void* p, int slot, long long* out) {
    FE* e = (FE*)p;
    const Slot& q = e->batch_api ? e->sl.at(slot) : e->S();
    for (int i = 0; i < 5; i++) out[i] = q.ds[i];
}

// expert cache counters since the last start (all cards): {decode steps, cold reads, of them misses (host), copies,
// resident groups, cold groups, live slots, slots, prefill cold reads, prefill misses, p90 misses/step}
void qe_expert_cache_stats(void* p, long long* out) {
    FE* e = (FE*)p;
    e->drain();
    long long v[11] = {};
    std::vector<int> per;
    for (int r = 0; r < e->ng; r++) {
        const Worker::EcStats& s = e->W[r]->ec_stats();
        v[0] = std::max(v[0], s.dec_steps); v[1] += s.dec_touch; v[2] += s.dec_miss; v[3] += s.copies; v[4] += s.resident;
        v[5] += s.ncold; v[6] += s.live_slots; v[7] += s.slots; v[8] += s.pf_touch; v[9] += s.pf_miss;
        if (per.size() < s.step_miss.size()) per.resize(s.step_miss.size(), 0);
        for (size_t i = 0; i < s.step_miss.size(); i++) per[i] += s.step_miss[i];
    }
    std::sort(per.begin(), per.end());
    v[10] = per.empty() ? 0 : per[per.size() * 9 / 10];
    memcpy(out, v, sizeof v);
}

int qe_last_reused(void* p) { return ((FE*)p)->last_reused; }
int qe_last_slot(void* p) { FE* e = (FE*)p; return e->batch_api ? e->last_pf_slot : 0; }
void qe_release(void* p, int slot) {
    FE* e = (FE*)p;
    e->drain();
    Slot& q = e->sl.at(e->batch_api ? slot : e->cs);
    const int sid = e->batch_api ? slot : e->cs;
    e->pfs.remove_if([&](const FE::Pf& x) { return x.slot == sid; });  // a prompt abandoned in flight
    if (q.fault) {  // failed by the NaN guard: its conversation is gone, the slot and its region are freed
        q.active = false;
        if (q.con) {
            e->job = [&](int r) { e->W[r]->set_constrained(sid, false); };
            e->run();
        }
        if (q.conv) e->forget(q.conv);  // never a source again (the next turn recomputes it)
        e->evict_slot(sid);
        e->update_top();
        e->rebind();
        return;
    }
    if (q.paused) {  // released while paused (finished / cancelled): its rows are archived already, its end state now;
        // the conversation stays archive-only (a next turn resumes it from there)
        q.paused = false;
        q.active = false;
        e->evict_slot(sid);
        e->update_top();
        e->rebind();
        return;
    }
    q.active = false;
    q.idle_ms = FE::now_ms();
    if (e->W[0]->kv_stream()) e->update_top();  // KV in RAM: its block-cache entries back to the expert cache
    q.remain = -1;
    q.want_drafts = false;
    if (q.con) {
        q.con = false;
        const int s = e->batch_api ? slot : e->cs;
        e->job = [&](int r) { e->W[r]->set_constrained(s, false); };
        e->run();
    }
}
// the reply may emit at most `tokens` more tokens: no step is launched ahead past it
void qe_set_limit(void* p, int slot, int tokens) {
    FE* e = (FE*)p;
    e->sl.at(e->batch_api ? slot : e->cs).remain = tokens;
}
void qe_set_lookahead(void* p, int on) { ((FE*)p)->lookahead = on != 0; }
// a finished answer (token ids) into the global lookup corpus of every card (FLASH_CORPUS; drafts across requests)
void qe_lookup_corpus(void* p, const int* ids, int n) {
    FE* e = (FE*)p;
    e->drain();
    for (int r = 0; r < e->ng; r++) e->W[r]->corpus_add(ids, n);
}
void qe_set_no_ahead(void* p, int, int on) { ((FE*)p)->no_ahead = on != 0; }
// ---- grammar tier: structured output. qe_mask_buffer(slot): pinned [8][(vocab + 31) / 32] uint32 (bit t = token t allowed;
// row j: after the pending token and drafts 1..j, row 0 also for the prompt's last chunk); qe_spec_io(slot): pinned
// int[32]: [0] = the model's drafts for the next step, [1..] those drafts (written by start_finish / every step);
// [16] = drafts to verify next step, [17..] those tokens (written by the caller before every step). qe_constrain(slot,
// 1) after start_begin turns it on (a new start / qe_release ends it); constrained slots step alone, never ahead.
unsigned* qe_mask_buffer(void* p, int slot) { FE* e = (FE*)p; e->sl.at(slot); return e->mh(slot); }
int* qe_spec_io(void* p, int slot) { FE* e = (FE*)p; e->sl.at(slot); return e->sio(slot); }
void qe_constrain(void* p, int slot, int on) {
    FE* e = (FE*)p;
    if (slot == e->cs) e->drain();  // nothing of this slot may be queued ahead
    Slot& q = e->sl.at(slot);
    q.con = on != 0;
    if (q.con) {
        int* io = qe_spec_io(p, slot);
        io[0] = io[16] = 0;
        qe_mask_buffer(p, slot);
    }
    const unsigned* mh = q.con && slot == e->cs ? e->mh(slot) : nullptr;
    e->job = [&](int r) {
        e->W[r]->set_constrained(slot, on != 0);
        if (mh) e->W[r]->upload_mask(mh, 1);
    };
    e->run();
}
int qe_constrained(void* p, int slot) { return ((FE*)p)->sl.at(slot).con ? 1 : 0; }

// ---- stops tier: the drafts of a slot's next step (for the caller's stop check) and the undo of the last step's rows
// past a stop (the reply then ends where its stop starts, and the slot's cached state matches the trimmed reply)
void qe_watch_stops(void* p, int slot, int on) { ((FE*)p)->sl.at(slot).want_drafts = on != 0; }
int qe_drafts(void* p, int slot, int* out) {
    FE* e = (FE*)p;
    Slot& q = e->sl.at(slot);
    if (e->k <= 0 || q.pending < 0) return 0;
    int n = 0;
    e->job = [&](int r) { if (r == 0) n = e->W[0]->next_drafts(slot, e->k, out); };
    e->run();
    return n;
}
int qe_trim(void* p, int slot, int pos) {
    FE* e = (FE*)p;
    Slot& q = e->sl.at(slot);
    const int hn = q.hist.size();
    if (pos >= hn) return 1;  // nothing processed past pos (the last token is pending)
    if (q.rb_pos < 0 || pos < q.rb_pos || pos >= q.rb_pos + q.rb_rows) return 0;
    if (slot == e->cs && e->inflight) return 0;  // a step launched after it committed its rows
    if (e->cs != slot) e->bind(slot);
    const int r = pos - q.rb_pos;  // rows of the last step that stay (their commit is still deferred)
    q.pending = q.hist[pos];
    q.hist.resize(pos);
    const int h1 = pos >= 1 ? q.hist[pos - 1] : EOS, h0 = pos >= 2 ? q.hist[pos - 2] : EOS;
    std::vector<int> d(std::max(1, e->k), q.pending);
    e->job = [&](int w) {
        e->W[w]->sv_load();
        e->W[w]->spec_init(q.pending, d.data(), e->k, pos, r, h1, h0);
        e->W[w]->sv_set_live();
        e->W[w]->set_prev_m(r);
    };
    e->run();
    q.rb_pos = -1;
    q.synced = false;
    q.lk_stale = true;
    q.bhist = -1;
    e->lk_next = 0;
    e->prev_lk = false;
    return 1;
}
// debug: the full-vocabulary logits of the last head row (after a plain-decoding start,
// k = 0: the prompt's last position) in token order (rows outside the head: -inf); returns the vocabulary size
int qe_debug_logits(void* p, float* out) {
    FE* e = (FE*)p;
    e->drain();
    for (int t = 0; t < VOCAB; t++) out[t] = -INFINITY;
    for (int r = 0; r < e->ng; r++) {
        const int nv = e->W[r]->vslice();
        std::vector<float> l(nv);
        std::vector<int> m(nv);
        e->W[r]->copy_logits(l.data(), 1);
        e->W[r]->copy_lmap(m.data());
        for (int i = 0; i < nv; i++)
            if (m[i] >= 0 && m[i] < VOCAB) out[m[i]] = l[i];
    }
    return VOCAB;
}
// debug: slot s (idle) leaves the GPUs: its conversation goes to the host archive as on a pool eviction
int qe_debug_evict(void* p, int s) {
    FE* e = (FE*)p;
    e->drain();
    if (s < 0 || s >= (int)e->sl.size() || e->sl[s].active || !e->pfs.empty()) return -1;
    e->evict_slot(s);
    e->update_top();
    e->rebind();
    return 0;
}
// debug: rank r's pooled indexer keys (first nb blocks per QSA layer) -> path
void qe_debug_ikb(void* p, int r, const char* path, int nb) {
    FE* e = (FE*)p;
    e->drain();
    e->W[r]->dump_ikb(path, nb);
}
// Activation capture (steering/make_vector.py): prefill ids[0, n) unsteered in a scratch slot (no prefix reuse, the
// conversations stay as they are) and write out[LAYERS][H] = the mean over the positions [r0, n) of each layer
// output's residual (mean of the 4 HC streams), the point the steering vectors act on. Eager (no graphs). Returns 0,
// or -1 (prompts in flight / no slot).
int qe_capture(void* p, const int* ids, int n, int r0, float* out) {
    FE* e = (FE*)p;
    if (!e->pfs.empty() || n <= 0 || r0 < 0 || r0 >= n || n + 2 > e->ctx) return -1;
    e->drain();
    const int y = acquire_slot(e, n, {});
    if (y < 0) return -1;
    e->bind(y);
    e->job = [&](int r) { e->W[r]->reset_slot(y); e->W[r]->set_steer_slot(y, -1, 0, 0, -1, 1.f); e->W[r]->cap_begin(); };
    e->run();
    std::vector<int> hist;
    for (int i = 0; i < n; i += PFM) {
        const int m = std::min(PFM, n - i);
        std::vector<bf16> rows((size_t)m * H);
        for (int j = 0; j < m; j++) {
            hist.push_back(ids[i + j]);
            ngram_row(e->ht, hist, (int)hist.size() - 1, rows.data() + (size_t)j * H);
        }
        const int skip = std::max(0, std::min(m, r0 - i));
        e->job = [&](int r) { int o; e->W[r]->cap_skip(skip); e->W[r]->prefill(ids + i, i, m, rows.data(), &o, nullptr, nullptr); };
        e->run();
    }
    e->job = [&](int r) { e->W[r]->cap_end(r == 0 ? out : nullptr); };
    e->run();
    for (int k = 0; k < LAYERS * H; k++) out[k] /= (float)(n - r0);
    e->evict_slot(y);  // scratch: nothing to keep
    e->update_top();
    e->rebind();
    return 0;
}

void qe_reset(void* p) {
    FE* e = (FE*)p;
    e->drain();
    for (int s = 0; s < (int)e->sl.size(); s++) {
        e->job = [&](int r) { e->W[r]->reset_slot(s); };
        e->run();
        e->sl[s] = Slot{};
    }
    e->convs.clear();
    e->update_top();
    e->rebind();
    e->pfs.clear();
}

// ---- slots: FLASH_MAX_SEQS resident sequences share the KV pool; the server sees one (no batched steps yet)
int qe_num_slots(void* p) { FE* e = (FE*)p; return e->batch_api ? (int)e->sl.size() : 1; }
// KV pool of all slots together in positions (VRAM: max_context rounded up to a period; KV in RAM: the RAM pool,
// FLASH_KV_RAM_TOKENS as far as the host could pin it)
long long qe_pool_tokens(void* p) {
    FE* e = (FE*)p;
    return std::max<long long>(e->ctx, (long long)e->W[0]->periods() * e->period);
}
int qe_kv_in_ram(void* p) { return ((FE*)p)->W[0]->kv_stream() ? 1 : 0; }  // FLASH_KV=ram
int qe_can_admit(void* p, int n) {
    FE* e = (FE*)p;
    e->refresh_np();
    if ((int)e->pfs.size() >= e->max_pf || n > e->ctx) return 0;
    int act = 0, used = 0;
    for (auto& q : e->sl)
        if (q.active) { act++; used += q.paused ? e->floor_of(q) : q.cap; }
    if (!e->batch_api) return act == 0;
    // a slot that is not generating, and room next to the generating ones (idle ones can be evicted)
    // the region a start gets (ensure_region): n + reserve, at most the whole pool. Uncapped, a prompt within
    // max_prompt_tokens but within `reserve` of the pool size (2 GPUs, KV in VRAM: pool = context = 262144) never fit
    // and waited in the queue forever
    return act < (int)e->sl.size() && used + std::min(e->np, e->periods_for(n + e->reserve)) <= e->np;
}
// room for a new n-token sequence of value `prio` (the main thread): pause decoding sequences worth less (lowest prio,
// the most recently admitted first; never a prompt being prefilled, never one of the same or a higher prio), only as
// many as needed and only when that makes it fit (paused ones count as out: they resume when room frees). 1: fits now
int qe_pause_for(void* p, int n, int prio) {
    FE* e = (FE*)p;
    e->refresh_np();
    if (!e->batch_api || !e->budget || (int)e->pfs.size() >= e->max_pf || n > e->ctx) return 0;
    int act = 0, used = 0;
    std::vector<int> vs;
    for (int i = 0; i < (int)e->sl.size(); i++) {
        const Slot& q = e->sl[i];
        if (!q.active) continue;
        act++;
        if (q.paused) continue;
        used += q.cap;
        if (q.conv && q.cap && q.pending >= 0 && !e->prefilling(i) && e->prio_of(i) < prio) vs.push_back(i);
    }
    if (act >= (int)e->sl.size()) return 0;  // every slot is taken: pausing frees no slot
    const int need = std::min(e->np, e->periods_for(n + e->reserve));  // as qe_can_admit / ensure_region
    if (used + need <= e->np) return 1;
    std::stable_sort(vs.begin(), vs.end(), [&](int a, int b) {
        return e->prio_of(a) != e->prio_of(b) ? e->prio_of(a) < e->prio_of(b) : e->sl[a].admit > e->sl[b].admit;
    });
    size_t k = 0;
    int left = used;
    while (k < vs.size() && left + need > e->np) left -= e->sl[vs[k++]].cap;
    if (left + need > e->np) return 0;  // pausing all of them would not make it fit: pause none
    for (size_t i = 0; i < k; i++) {
        fprintf(stderr, "[qflash] pool: pausing slot %d (prio %d) for a %d-token start of prio %d\n", vs[i], e->prio_of(vs[i]), n, prio);
        if (!e->pause_slot(vs[i])) return 0;
    }
    return 1;
}

// ---- cache tier: prompt-cache bookkeeping
int qe_probe_cached(void* p, const int* ids, int n) {
    FE* e = (FE*)p;
    FE::Src s = e->best(ids, n, false);
    if (!e->budget && s.c && !s.live) return 0;
    return s.L;
}
void qe_set_prio(void* p, int slot, int prio) {
    FE* e = (FE*)p;
    Slot& q = e->sl.at(e->batch_api ? slot : e->cs);
    if (q.conv) q.conv->prio = prio;
}
int qe_num_pinned(void*) { return 0; }
int qe_last_copied(void* p) { return ((FE*)p)->last_copied; }
int qe_last_from_archive(void* p) { return ((FE*)p)->last_from_archive ? 1 : 0; }

// ---- archive tier: idle resident sequences -> host memory piece by piece (~budget bytes per GPU per call)
// NaN guard faults so far, and the slot's fault message (1, cleared on read; 0: the slot has none)
long long qe_fault_count(void* p) { return ((FE*)p)->n_fault; }
int qe_slot_fault(void* p, int slot, char* buf, int n) {
    FE* e = (FE*)p;
    if (slot < 0 || slot >= (int)e->fault_msg.size() || e->fault_msg[slot].empty()) return 0;
    if (buf && n > 0) snprintf(buf, n, "%s", e->fault_msg[slot].c_str());
    e->fault_msg[slot].clear();
    return 1;
}
// host memory pressure check (host_mem.h): the server calls it about once a second, also while idle
void qe_mem_guard(void* p) { ((FE*)p)->mem_guard(); }
int qe_archive_more(void* p, long long budget) {
    FE* e = (FE*)p;
    if (!e->budget || !e->pfs.empty()) return 1;
    e->mem_guard();
    for (int s = 0; s < (int)e->sl.size(); s++) {
        Slot& q = e->sl[s];
        if (!q.conv || q.synced || q.active) continue;
        if (s == e->cs) e->drain();
        const int hn = q.hist.size();
        if (q.conv->kv_n < hn) {
            // positions per call from the KV bytes per position of the busiest card (a full period)
            size_t per = 0;
            for (int r = 0; r < e->ng; r++) {
                std::vector<Worker::Piece> ps;
                e->W[r]->kv_pieces_at(0, 0, e->period, ps);
                per = std::max(per, FE::bytes_of(ps));
            }
            const long long np = std::max<long long>(1, budget / std::max<size_t>(per, 1));
            const int upto = (int)std::min<long long>(hn, ((long long)q.conv->kv_n / e->period + np) * e->period);
            if (!e->sync(s, upto, false)) return 1;
            if (upto < hn) return 0;
        }
        e->sync(s);
        return 0;
    }
    return 1;
}
void qe_persist(void* p) {  // no disk cache: archive the idle resident sequences
    while (!qe_archive_more(p, 1LL << 40)) {}
}
int qe_num_archives(void* p) {
    FE* e = (FE*)p;
    int c = 0;
    for (auto& v : e->convs) c += v.end || !v.ck.empty();
    return c;
}
int qe_num_disk_archives(void*) { return 0; }

// ---- monitor tier
void qe_slot_info(void* p, int i, long long* out) {
    FE* e = (FE*)p;
    Slot& q = e->batch_api ? e->sl.at(i) : e->S();
    out[0] = (long long)q.base * e->period;
    out[1] = (long long)q.cap * e->period;
    out[2] = q.hist.size();
    out[3] = q.active;
    out[4] = (long long)q.used;
    out[5] = q.conv ? q.conv->prio : 1;
    out[6] = q.paused;
}
void qe_archive_info(void* p, double* out) {
    FE* e = (FE*)p;
    out[0] = qe_num_archives(p);
    out[1] = (double)e->hpool.used();
    out[2] = 0;
    out[3] = 0;
    out[4] = (double)e->hpool.pinned();
    out[5] = (double)std::min(e->hpool.used(), e->hpool.pinned());
}
void qe_pool_counters(void* p, long long* out) {
    FE* e = (FE*)p;
    out[0] = e->n_grow;
    out[1] = e->n_move;
    out[2] = e->n_slot_evict;
    out[3] = e->n_pause;
    out[4] = e->n_resume;
    long long np2 = 0;
    for (auto& q : e->sl) np2 += q.paused;
    out[5] = np2;
}

}  // extern "C"
