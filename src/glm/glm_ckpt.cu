// GLM engine: prefix snapshots for longest-common-prefix reuse (round 18, registry memory_kv [glm-lcp-reuse]).
//
// What a resume at position q needs beyond the cache rows [0, q) that stay in place (MLA latents kc, pooled index keys
// pk, MTP layer likewise, all indexed by absolute position):
//  - KDA layers: recurrent state S and the conv history (cannot be truncated: taken at q);
//  - MLA layers (incl. MTP): the index-key / gate-score ring rows q-4..q-1 (a ring of kring rows: later tokens
//    overwrite them; they feed the pool that contains q-1 / q);
//  - the final normed hidden of row q-1 (mh row 0: the MTP cache row q-1 is rewritten with the new token at q, and an
//    identical prompt drafts from it).
// Snapshots live in pinned host memory (about 52 MB per card), at most GLM_CK_N, LRU, never past the host-RAM margin.
#include "glm.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <mutex>
#include <thread>

#include "../env.h"
#include "../host_mem.h"
#include "glm_kernels.cuh"

namespace glm {

template <class F> void Worker::ck_walk(int q, float* hid, F f) {
    const size_t nst = (size_t)p.nh() * KD * KD, nch = (size_t)9 * p.nh() * KD;
    for (auto& Y : L) {
        if (Y.S) { f(Y.S, nst * 4); f(Y.conv_hist, nch * 4); }
        if (Y.ki)
            for (int j = 0; j < POOL; j++) {
                const int ps = q - POOL + j;
                const size_t row = ps >= 0 ? (size_t)(ps % E->kring) : 0;
                f(ps >= 0 ? Y.ki + row * ID : nullptr, (size_t)ID * 4);
                f(ps >= 0 ? Y.gsc + row * ID : nullptr, (size_t)ID * 4);
            }
    }
    f(hid, (size_t)H * 4);
}

size_t Worker::ck_bytes() {
    size_t b = 0;
    ck_walk(POOL, nullptr, [&](float*, size_t n) { b += n; });
    return b;
}

void Worker::ck_io(char* host, int q, bool save, bool hid) {
    CUDA_CHECK(cudaSetDevice(dev));
    // save: the streams that write the saved state (compute, KDA recurrence, gate/up, all-reduce, side) are done;
    // the next window's expert staging (DMA streams) keeps running (review r18 L1)
    if (save)
        for (cudaStream_t s_ : {st, krst, gst, cms, pst, ags, ost})  // (review r20 L4: + the owner allgathers)
            if (s_) CUDA_CHECK(cudaStreamSynchronize(s_));
    size_t off = 0;
    ck_walk(q, !hid ? nullptr : save ? ck_h : mh, [&](float* d, size_t n) {
        if (d) {
            if (save) CUDA_CHECK(cudaMemcpyAsync(host + off, d, n, cudaMemcpyDeviceToHost, st));
            else CUDA_CHECK(cudaMemcpyAsync(d, host + off, n, cudaMemcpyHostToDevice, st));
        }
        off += n;
    });
    CUDA_CHECK(cudaStreamSynchronize(st));
}

void Worker::arc_io(char* blk, int k, int s, int e, bool save, bool all) {
    CUDA_CHECK(cudaSetDevice(dev));
    const int P = E->arc_page;
    const size_t kcb = (size_t)P * KVLR * 2, lb = kcb + (size_t)(P / POOL + 1) * ID * 4;
    int li = -1;
    for (int l = 0; l < (int)L.size(); l++) {
        Layer& Y = L[l];
        if (!Y.kc) continue;
        li++;
        if (!all && li % p.ngpu != p.rank) continue;
        const int sh = l == MTP_L ? 1 : 0;  // MTP row p = logical row p + 1 (it needs the token at p + 1)
        const int ps = std::max(0, s - sh), pe = e - sh, b0 = k * P - sh;
        if (pe <= ps) continue;
        char* h = blk + (size_t)li * lb;
        // rows of w bytes, device pitch dp (round 26: position-major rows of all MLA layers side by side), host packed
        auto cp = [&](void* d, char* hp, size_t w, size_t dp, size_t n) {
            if (save) CUDA_CHECK(cudaMemcpy2DAsync(hp, w, d, dp, w, n, cudaMemcpyDeviceToHost, st));
            else CUDA_CHECK(cudaMemcpy2DAsync(d, dp, hp, w, w, n, cudaMemcpyHostToDevice, st));
        };
        const size_t kst = E->mla_pos ? (size_t)E->mla_nl * KVLR : KVLR, pst = E->mla_pos ? (size_t)E->mla_nl * ID : ID;
        cp(Y.kc + (size_t)ps * kst, h + (size_t)(ps - b0) * KVLR * 2, KVLR * 2, kst * 2, (size_t)(pe - ps));
        // pool j is written with its last row 4j + 3 (pool_keys): pools [ps/4, pe/4) belong to these rows
        const int j0 = ps / POOL, j1 = pe / POOL, jb = b0 >= 0 ? b0 / POOL : -1;
        if (j1 > j0) cp(Y.pk + (size_t)j0 * pst, h + kcb + (size_t)(j0 - jb) * ID * 4, ID * 4, pst * 4, (size_t)(j1 - j0));
    }
}

// MTP cache row p from the hidden of row p (mh row 0) and the token at p + 1 (prefill set active)
void Worker::mtp_prev_row(int tok0, int p_) {
    Layer& Y = L[MTP_L];
    CUDA_CHECK(cudaMemcpyAsync(d_tok, &tok0, 4, cudaMemcpyHostToDevice, st));
    bf16* xc = (bf16*)R;
    float* xr = R + H;
    mtp_in(d_tok, E->emb, mh, mw.enorm, mw.hnorm, xc, 1, st);
    lin(mw.eh, nullptr, xc, xr, 1);
    add_rms(xr, nullptr, Y.in_ln, xf, xb, 1, st);
    mla_cache(MTP_L, 1, p_);
    CUDA_CHECK(cudaStreamSynchronize(st));
}

// [glm-rel3-cancel] the window-start save of a gated live continuation: one pinned buffer per card (ck_bytes, ~52 MB),
// taken once; no host room: no save (a stop then drops the live history, as a stop past the first window does)
bool Engine::pf_ws_ready() {
    if (!pf_ws.empty()) return pf_ws[0] != nullptr;
    pf_ws.assign(ngpu, nullptr);
    size_t tot = 0;
    for (int r = 0; r < ngpu; r++) tot += W[r]->ck_bytes();
    bool ok = qe_hostmem::headroom(true) >= tot;
    for (int r = 0; r < ngpu && ok; r++)
        if (cudaHostAlloc((void**)&pf_ws[r], W[r]->ck_bytes(), cudaHostAllocPortable) != cudaSuccess) {
            cudaGetLastError();
            pf_ws[r] = nullptr;
            ok = false;
        }
    if (!ok) {
        for (char*& x : pf_ws) if (x) { cudaFreeHost(x); x = nullptr; }
        fprintf(stderr, "[glm] no host room for the prefill cancel save: a cancelled continuation is prefilled again\n");
    }
    return ok;
}

int Engine::pf_window(int T) const {
    const int nwin = (T + chunk - 1) / chunk;
    return std::min(chunk, ((T + nwin - 1) / nwin + 63) / 64 * 64);
}

static unsigned long long tok_hash(const int* ids, int n) {
    unsigned long long h = 1469598103934665603ull;
    for (int i = 0; i < n; i++) { h ^= (unsigned)ids[i]; h *= 1099511628211ull; }
    return h ^ (unsigned long long)n;
}

void Engine::ck_init() {
    ck_on = env_int("GLM_CK", 1) != 0 && nseq == 1;
    if (env_int("GLM_CK", 1) != 0 && nseq != 1) fprintf(stderr, "[glm] prefix snapshots off with GLM_SEQS=2\n");
    ck_win = env_int("GLM_CK_WIN", 1) != 0;
    // [glm-rel-slice] room for every conversation served at once (GLM_CONVS): its prompt end, answer end and a cut
    {
        const int convs = std::max(1, std::min(64, env_int("GLM_CONVS", 8)));
        ck_max = std::max(ck_max, 3 * convs);
    }
    ck_max = std::max(1, env_int("GLM_CK_N", ck_max));
    ck_hint_min = env_int("GLM_CK_HINT_MIN", ck_hint_min);
    ck_div_min = env_int("GLM_CK_DIV_MIN", ck_div_min);
    ck_tail = std::max(0, env_int("GLM_CK_TAIL", ck_tail));
    ck_tail_min = env_int("GLM_CK_TAIL_MIN", ck_tail_min);
    ck_log = env_int("GLM_CK_LOG", 0) != 0;
    arc_on = ck_on && env_int("GLM_ARC", 1) != 0;
    arc_chk = env_int("GLM_ARC_CHK", 0) != 0;
    arc_page = std::max(64, env_int("GLM_ARC_PAGE", arc_page) / 64 * 64);
    // (release track: QE_KV_ARCHIVE_GB, the other models' archive budget, applies when GLM_ARC_GB is not set)
    // (review rel L14: QE_KV_ARCHIVE_GB in GiB, as the other engines read it)
    arc_cap = getenv("GLM_ARC_GB") || !getenv("QE_KV_ARCHIVE_GB")
                  ? (size_t)(std::max(0.0, env_double("GLM_ARC_GB", 32.0)) * 1e9)
                  : (size_t)(std::max(0.0, env_double("QE_KV_ARCHIVE_GB", 32.0)) * (double)(1ull << 30));
    {
        int nmla = 0;
        for (auto& Y : W[0]->L) nmla += Y.kc != nullptr;
        arc_blk = (size_t)nmla * ((size_t)arc_page * KVLR * 2 + (size_t)(arc_page / POOL + 1) * ID * 4);
    }
    if (arc_on && arc_cap < arc_blk) arc_on = false;
    if (arc_on)
        fprintf(stderr, "[glm] conversation archive: up to %.1f GB pinned host (%.1f KB per token, pages of %d), "
                "never past the host margin (GLM_ARC_GB, GLM_ARC=0 off)\n", arc_cap / 1e9, arc_blk / 1e3 / arc_page, arc_page);
    if (arc_on)  // review r19 M2: idle host pressure gives back the free blocks (pages in use go at the next request)
        std::thread([this] {
            while (true) {
                std::this_thread::sleep_for(std::chrono::milliseconds(500));
                if (qe_hostmem::deficit(true) > 0) {
                    const size_t f = arc_give_free();
                    if (f) fprintf(stderr, "[glm] host memory short: conversation archive gave back %.2f GB of free pages\n", f / 1e9);
                }
            }
        }).detach();
    if (!ck_on) return;
    std::atomic<bool> ok{true};
    run_all([&](Worker& w) {
        CUDA_CHECK(cudaSetDevice(w.dev));
        if (!w.ck_h && cudaMalloc(&w.ck_h, (size_t)H * 4) != cudaSuccess) {
            cudaGetLastError();
            w.ck_h = nullptr;
            ok = false;
        }
    });
    if (!ok) {  // review r18 L4: no VRAM for the hidden row: snapshots off, never a crash
        fprintf(stderr, "[glm] prefix snapshots off: no VRAM for their hidden-row buffer\n");
        ck_on = arc_on = false;
        return;
    }
    ck_lout.assign(VOCAB, 0.f);
}

int Engine::ck_take(int q, int kind) {
    (void)q;
    auto planned = [&](int i) {
        for (auto& pe : ck_plan) if (pe.second == i) return true;
        return false;
    };
    // review r18 L5: host memory taken by others since: give back the least recently used snapshot buffers
    while (!ck.empty() && qe_hostmem::deficit(true) > 0) {
        int v = -1;
        for (size_t i = 0; i < ck.size(); i++) {  // free entries first, then the least recently used
            if (planned((int)i) || ck[i].q < -1 || ck[i].pin) continue;
            const bool fi = ck[i].q == -1, fv = v >= 0 && ck[v].q == -1;
            if (v < 0 || (fi && !fv) || (fi == fv && ck[i].used < ck[v].used)) v = (int)i;
        }
        if (v < 0) break;
        for (char* x : ck[v].buf) if (x) cudaFreeHost(x);
        fprintf(stderr, "[glm] host memory short: prefix snapshot at %d freed\n", ck[v].q);
        if (!ck_plan.empty()) {  // (planned entries are indices: keep them stable)
            ck[v] = Ckpt();
            ck[v].buf.assign(ngpu, nullptr);
            ck[v].q = -2;  // no buffers: never reused
            continue;
        }
        ck.erase(ck.begin() + v);
    }
    for (size_t i = 0; i < ck.size(); i++)  // [glm-conv-arc]: an entry whose rows are gone is free
        if (ck[i].q > 0 && !ck[i].pin && !planned((int)i) && !ck_valid(ck[i])) ck[i].q = -1;
    for (size_t i = 0; i < ck.size(); i++)
        if (ck[i].q == -1 && !ck[i].pin && !planned((int)i)) return (int)i;
    // [glm-rel-slice] a parked reply's entry (kind 4) is taken beyond ck_max: the prefix snapshots keep their number
    int npin = 0;
    for (auto& c : ck) npin += c.pin;
    if ((int)ck.size() < ck_max + npin || kind == 4) {
        size_t tot = 0;
        std::vector<size_t> b(ngpu);
        for (int r = 0; r < ngpu; r++) tot += b[r] = W[r]->ck_bytes();
        if (qe_hostmem::headroom(true) >= tot) {
            Ckpt c;
            c.buf.assign(ngpu, nullptr);
            bool ok = true;
            for (int r = 0; r < ngpu && ok; r++) {
                if (cudaHostAlloc((void**)&c.buf[r], b[r], cudaHostAllocPortable) != cudaSuccess) {
                    cudaGetLastError();
                    c.buf[r] = nullptr;
                    ok = false;
                }
            }
            if (ok) {
                ck.push_back(std::move(c));
                return (int)ck.size() - 1;
            }
            for (char* x : c.buf) if (x) cudaFreeHost(x);
        }
    }
    if (kind == 4) return -1;  // (no host memory for its buffers: the reply stays bound)
    // [glm-rel-slice] an entry that a newer prompt of the same conversation extends (a prompt end or cut: tokens the
    // client sent; not an answer end, which a client that sends back another reply text never extends) is evicted
    // before another conversation's newest entries (8 agents: 17-19 of 30 later turns lost their whole prefix with
    // the plain LRU)
    auto superseded = [&](int i) {
        const Ckpt& c = ck[i];
        if (c.q <= 0) return false;
        for (size_t j = 0; j < ck.size(); j++) {
            const Ckpt& d = ck[j];
            if ((int)j == i || d.pin || (d.kind != 1 && d.kind != 2) || d.q <= c.q || (int)d.toks.size() < c.q ||
                (int)c.toks.size() != c.q || d.toks[c.q - 1] != c.toks[c.q - 1] || d.toks[0] != c.toks[0])
                continue;  // (review R3-L6: two cheap token checks before the full compare)
            if (!memcmp(d.toks.data(), c.toks.data(), (size_t)c.q * 4)) return true;
        }
        return false;
    };
    int v = -1;  // least recently used entry, window-end snapshots first (review r18 L6: a cut / prompt end is reused more)
    int vr = 0;
    for (size_t i = 0; i < ck.size(); i++) {
        if (planned((int)i) || ck[i].q < -1 || ck[i].pin) continue;
        // (re-review r25 R1: answer ends take part in the plain LRU; a class rank above it evicted the answer end just
        // saved by the same prefill's prompt-end take)
        const int r = ck[i].kind == 0 ? 2 : superseded((int)i) ? 1 : 0;
        if (v < 0 || r > vr || (r == vr && ck[i].used < ck[v].used)) { v = (int)i; vr = r; }
    }
    if (v >= 0) ck[v].q = -1;
    return v;
}

void Engine::ck_drop_after(int q) {
    for (auto& c : ck) if (c.q > q && !c.pin) c.q = -1;  // (q = -2: freed buffers, stays)
}

// ---- [glm-conv-arc] live row generations, conversation archive -------------------------------------------------
void Engine::lv_cut(int r) {
    while (!lv.empty() && lv.back()[0] >= r) lv.pop_back();
    if (!lv.empty() && lv.back()[1] > r) lv.back()[1] = r;
}
void Engine::lv_push(int a, int b) {
    lv_cut(a);
    if (lv_end() < a) lv.push_back({lv_end(), a, -1});  // (never happens: rows of unknown origin are never valid)
    if (b > a) lv.push_back({a, b, ++lv_gen});
}

static void lv_clip(std::vector<Engine::Seg>& v, int q) {
    while (!v.empty() && v.back()[0] >= q) v.pop_back();
    if (!v.empty() && v.back()[1] > q) v.back()[1] = q;
}

template <class F> bool Engine::ck_missing(const Ckpt& c, F f) const {
    for (const Seg& sg : c.segs) {
        const int x = sg[0], y = sg[1], g = sg[2];
        int ls = 0, le = 0;
        if (g >= 0)
            for (const Seg& v : lv)
                if (v[2] == g) { ls = v[0]; le = v[1]; break; }
        if (le <= ls || le <= x || ls >= y) {
            if (!f(g, x, y)) return false;
        } else {
            if (x < ls && !f(g, x, ls)) return false;
            if (le < y && !f(g, le, y)) return false;
        }
    }
    return true;
}

int Engine::arc_find(int g, int k) const {
    for (size_t i = 0; i < arc.size(); i++)
        if (arc[i].g == g && arc[i].k == k) return (int)i;
    return -1;
}

// per archive window of [a, b): f(k, x, y)
template <class F> static bool by_window(int P, int a, int b, F f) {
    for (int k = a / P; a < b; k++) {
        const int y = std::min(b, (k + 1) * P);
        if (!f(k, a, y)) return false;
        a = y;
    }
    return true;
}

bool Engine::ck_valid(const Ckpt& c) {
    if (c.q <= 0) return false;
    if (c.segs.empty() || c.segs.back()[1] != c.q) return false;
    return ck_missing(c, [&](int g, int a, int b) {
        if (g < 0 || !arc_on) return false;
        return by_window(arc_page, a, b, [&](int k, int x, int y) {
            const int i = arc_find(g, k);
            return i >= 0 && arc[i].s <= x && arc[i].e >= y;
        });
    });
}

void Engine::arc_touch(const Ckpt& c) {
    for (const Seg& sg : c.segs)
        by_window(arc_page, sg[0], sg[1], [&](int k, int, int) {
            const int i = arc_find(sg[2], k);
            if (i >= 0) arc[i].used = std::max(arc[i].used, c.used);
            return true;
        });
}

void Engine::arc_gc() {
    std::vector<char> keep(arc.size(), 0);
    for (auto& c : ck) {
        if (c.q <= 0) continue;
        if (!ck_valid(c)) { c.q = -1; continue; }
        for (const Seg& sg : c.segs)
            by_window(arc_page, sg[0], sg[1], [&](int k, int, int) {
                const int i = arc_find(sg[2], k);
                if (i >= 0) keep[i] = 1;
                return true;
            });
    }
    size_t j = 0;
    for (size_t i = 0; i < arc.size(); i++) {
        if (keep[i] || (arc_hold && arc[i].used >= arc_hold)) arc[j++] = arc[i];  // (pages an archive write holds stay)
        else { std::lock_guard<std::mutex> lk(arc_mu); arc_free.push_back(arc[i].buf); }
    }
    arc.resize(j);
}

char* Engine::arc_block(unsigned long long hold) {
    while (true) {
        {
            std::lock_guard<std::mutex> lk(arc_mu);
            if (!arc_free.empty()) {
                char* b = arc_free.back();
                arc_free.pop_back();
                return b;
            }
        }
        if ((arc_blocks + 1) * arc_blk <= arc_cap && qe_hostmem::headroom(true) >= arc_blk) {
            char* b = nullptr;
            if (cudaHostAlloc((void**)&b, arc_blk, cudaHostAllocPortable) == cudaSuccess) {
                std::lock_guard<std::mutex> lk(arc_mu);
                arc_blocks++;
                qe_hostmem::charged() += (long long)arc_blk;
                return b;
            }
            cudaGetLastError();
        }
        // full: the least recently used page goes (its conversation's snapshots lose their rows), then what nothing needs
        int v = -1;
        for (size_t i = 0; i < arc.size(); i++)
            if (arc[i].used < hold && (v < 0 || arc[i].used < arc[v].used)) v = (int)i;
        if (v < 0) return nullptr;
        { std::lock_guard<std::mutex> lk(arc_mu); arc_free.push_back(arc[v].buf); }
        arc.erase(arc.begin() + v);
        st_arc_evict++;
        arc_gc();
    }
}

// blocks not in use go back to the host while it is short (review r19 M2: also while idle, from the watcher thread)
size_t Engine::arc_give_free() {
    std::lock_guard<std::mutex> lk(arc_mu);
    size_t freed = 0;
    while (!arc_free.empty() && qe_hostmem::deficit(true) > 0) {
        cudaFreeHost(arc_free.back());
        arc_free.pop_back();
        arc_blocks--;
        qe_hostmem::charged() -= (long long)arc_blk;
        freed += arc_blk;
    }
    return freed;
}

void Engine::arc_trim() {
    if (!arc_on || qe_hostmem::deficit(true) == 0) return;
    size_t freed = arc_give_free();
    while (qe_hostmem::deficit(true) > 0 && !arc.empty()) {  // then pages in use, least recently used first
        int v = 0;
        for (size_t i = 1; i < arc.size(); i++) if (arc[i].used < arc[v].used) v = (int)i;
        { std::lock_guard<std::mutex> lk(arc_mu); arc_free.push_back(arc[v].buf); }
        arc.erase(arc.begin() + v);
        st_arc_evict++;
        arc_gc();
        freed += arc_give_free();
    }
    if (freed) fprintf(stderr, "[glm] host memory short: conversation archive gave back %.2f GB (%zu pages left)\n",
                       freed / 1e9, arc.size());
}

void Engine::arc_save(int x, unsigned long long floor) {
    if (!arc_on) return;
    const auto t0 = std::chrono::steady_clock::now();
    std::vector<char> ok(ck.size());
    for (size_t i = 0; i < ck.size(); i++) ok[i] = ck[i].q > 0 && ck_valid(ck[i]);
    const unsigned long long hold = arc_hold = ++ck_tick;
    struct Job { char* b; int k, s, e; };
    std::vector<Job> jobs;
    long rows = 0;
    bool full = false;
    for (const Seg& v : lv) {
        if (v[1] <= x || v[2] < 0 || full) continue;
        int hi = 0;  // the live rows of this writer that a valid snapshot needs end here
        for (size_t i = 0; i < ck.size(); i++) {
            if (!ok[i]) continue;
            for (const Seg& sg : ck[i].segs)
                if (sg[2] == v[2]) hi = std::max(hi, std::min(sg[1], v[1]));
        }
        const int a = std::max(v[0], x);
        if (a >= hi) continue;
        by_window(arc_page, a, hi, [&](int k, int xs, int ys) {
            int i = arc_find(v[2], k);
            if (i >= 0 && arc[i].s <= xs && arc[i].e >= ys) {  // already there
                arc[i].used = std::max(arc[i].used, hold);
                return true;
            }
            if (i >= 0) {  // more rows of the same writer in this window: they are live, copy the new range
                ArcPg& pg = arc[i];
                if (pg.e < xs || pg.s > ys) { pg.s = xs; pg.e = ys; }
                else { pg.s = std::min(pg.s, xs); pg.e = std::max(pg.e, ys); }
                pg.used = hold;
                jobs.push_back({pg.buf, k, xs, ys});
                rows += ys - xs;
                return true;
            }
            char* b = arc_block(floor ? std::min(floor, hold) : hold);
            if (!b) { full = true; return false; }
            ArcPg pg;
            pg.g = v[2]; pg.k = k; pg.s = xs; pg.e = ys; pg.used = hold; pg.buf = b;
            arc.push_back(pg);
            jobs.push_back({b, k, xs, ys});
            rows += ys - xs;
            return true;
        });
    }
    arc_hold = 0;
    if (jobs.empty()) return;
    run_all([&](Worker& w) {
        for (const Job& j : jobs) w.arc_io(j.b, j.k, j.s, j.e, true, false);
        CUDA_CHECK(cudaStreamSynchronize(w.st));
    });
    if (arc_chk) {  // the latents are replicated: every card's copy must equal the stored one
        std::vector<long> bad(ngpu, 0);
        run_all([&](Worker& w) {
            char* t = nullptr;
            CUDA_CHECK(cudaHostAlloc((void**)&t, arc_blk, cudaHostAllocPortable));
            for (const Job& j : jobs) {
                memcpy(t, j.b, arc_blk);
                w.arc_io(t, j.k, j.s, j.e, true, true);
                CUDA_CHECK(cudaStreamSynchronize(w.st));
                bad[w.p.rank] += memcmp(t, j.b, arc_blk) != 0;
            }
            cudaFreeHost(t);
        });
        fprintf(stderr, "[glm] archive check: %zu pages, differing per card", jobs.size());
        for (long b : bad) fprintf(stderr, " %ld", b);
        fprintf(stderr, "\n");
    }
    // the pages' LRU tick: their newest snapshot's; pages only snapshots that lost rows meanwhile needed go
    for (auto& pg : arc) pg.used = 0;
    for (auto& c : ck) if (c.q > 0) arc_touch(c);
    arc_gc();
    st_arc_save_rows += rows;
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    st_arc_save_ms += ms;
    if (ck_log)
        fprintf(stderr, "[glm] archive: %ld rows from %d saved in %zu page copies, %.1f ms (%zu pages, %zu blocks%s)\n",
                rows, x, jobs.size(), ms, arc.size(), arc_blocks, full ? ", full" : "");
}

bool Engine::arc_load(const Ckpt& c) {
    struct Job { char* b; int k, s, e; };
    std::vector<Job> jobs;
    long rows = 0;
    const bool ok = ck_missing(c, [&](int g, int a, int b) {
        return by_window(arc_page, a, b, [&](int k, int x, int y) {
            const int i = arc_find(g, k);
            if (i < 0 || arc[i].s > x || arc[i].e < y) return false;  // (review r19 L2: never abort, the caller prefills)
            jobs.push_back({arc[i].buf, k, x, y});
            rows += y - x;
            return true;
        });
    });
    if (!ok) return false;
    if (jobs.empty()) return true;
    const auto t0 = std::chrono::steady_clock::now();
    run_all([&](Worker& w) {
        for (const Job& j : jobs) w.arc_io(j.b, j.k, j.s, j.e, false, true);
        CUDA_CHECK(cudaStreamSynchronize(w.st));
    });
    st_arc_load_rows += rows;
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    st_arc_load_ms += ms;
    if (ck_log) fprintf(stderr, "[glm] archive: %ld rows restored in %zu page copies, %.1f ms\n", rows, jobs.size(), ms);
    return true;
}

void Engine::ck_forget() {
    for (auto& c : ck) {
        if (c.q > 0) c.q = -1;
        c.pin = false;  // [glm-rel-slice] a parked reply loses its state too (its resume re-prefills)
        c.tag = 0;
    }
    std::lock_guard<std::mutex> lk(arc_mu);
    for (auto& pg : arc) arc_free.push_back(pg.buf);
    arc.clear();
    lv.clear();
}

int Engine::ck_best(const int* ids, int n, int lcp) {
    (void)lcp;  // [glm-conv-arc]: any snapshot whose rows are live or archived, not only the live history's
    int bi = -1;
    for (size_t i = 0; i < ck.size(); i++) {
        const Ckpt& c = ck[i];
        if (c.q <= 0 || c.pin || c.q > n || (bi >= 0 && c.q <= ck[bi].q)) continue;
        if (c.q == n && (c.kind != 2 || c.logits.empty())) continue;  // nothing left to prefill: needs the logits
        if (c.hash != tok_hash(ids, c.q)) continue;
        if ((int)c.toks.size() != c.q || memcmp(c.toks.data(), ids, (size_t)c.q * 4)) continue;  // review r19 L1
        if (!ck_valid(c)) continue;
        bi = (int)i;
    }
    return bi;
}

bool Engine::ck_restore(int i) {
    const auto t0 = std::chrono::steady_clock::now();
    Ckpt& c = ck[i];
    // review r19 H1: the target is the newest use, so the save below never evicts its pages (or the shared ones)
    c.used = ++ck_tick;
    arc_touch(c);
    // the live rows from the first one this snapshot does not share on are overwritten (by the copy back or the prefill):
    // archive what other snapshots need first
    int x = c.q;
    for (size_t j = 0; j < lv.size(); j++) {
        if (j >= c.segs.size()) break;
        const Seg &a = lv[j], &b = c.segs[j];
        if (a == b) continue;
        x = a[0] == b[0] && a[2] == b[2] ? std::min(a[1], b[1]) : a[0];
        break;
    }
    arc_save(std::min(x, c.q), c.used);
    mla_need(c.q);  // round 26: the MLA pages under the restored rows are held before they are written
    if (!arc_load(c)) {  // (not reached: the save keeps the target's pages; the caller prefills from 0)
        fprintf(stderr, "[glm] prefix snapshot at %d lost its archived rows: prefilled instead\n", c.q);
        c.q = -1;
        return false;
    }
    lv = c.segs;
    kda_mat();  // a pending pointer-mode state is materialised first (it would overwrite S at the next forward)
    ver_pending = false;
    pos = c.q;
    run_all([&](Worker& w) {
        w.ck_io(c.buf[w.p.rank], c.q, false);
        CUDA_CHECK(cudaMemcpyAsync(w.d_pos, &pos, 4, cudaMemcpyHostToDevice, w.st));
        CUDA_CHECK(cudaStreamSynchronize(w.st));
    });
    c.used = ++ck_tick;
    arc_touch(c);
    st_ck_load++;
    st_ck_load_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    return true;
}

// spec_start's prefill: resume (live extension, snapshot, or nothing), forced cuts, snapshots at window ends / cuts /
// the prompt end. Returns the reused prefix length; last_tok (and lout) hold the last row's result.
int Engine::ck_prefill(const int* ids, int n, float* lout) {
    if (!ck_inited) { ck_init(); ck_inited = true; }
    int lcp = 0;
    const int hl = std::min((int)sp_hist.size(), pos);
    while (lcp < hl && lcp < n && sp_hist[lcp] == ids[lcp]) lcp++;
    int r = 0;
    bool restored = false;
    int ri = -1;  // the restored entry
    if (ck_on) {
        arc_trim();
        // [glm-conv-arc]: the decode rows of the last turn are live rows of their own writer
        if (lv_end() > pos) lv_cut(pos);
        if (lv_end() < pos) lv_push(lv_end(), pos);
    }
    // round 25 (review r25 M3): the previous reply's answer-end snapshot is taken only now, when this prompt does not
    // continue it live (another conversation is about to overwrite its rows); a continued conversation pays nothing
    const bool ans_was = ck_ans_due;  // [glm-rel3-cancel] (back if a cancel puts this conversation back live)
    if (ck_ans_due) {
        ck_ans_due = false;
        if (!(pos > 0 && (int)sp_hist.size() == pos && n > pos && lcp == pos)) ck_answer_end();
    }
    const int bi = ck_on ? ck_best(ids, n, lcp) : -1;
    if (pos > 0 && (int)sp_hist.size() == pos && n > pos && lcp == pos && !(bi >= 0 && ck[bi].q > pos)) {
        r = pos;
        det_restore();  // round 17: a continued conversation also starts from the seeded cache (no-op after seq_end)
    } else if (bi >= 0 && ck_restore(bi)) {  // (archives the live rows other snapshots need before they are overwritten)
        det_restore();
        r = ck[bi].q;
        restored = true;
        ri = bi;
        sp_hist.assign(ids, ids + r);
    } else {
        if (ck_on) arc_save(0);
        reset(true);
        sp_hist.clear();
    }
    CHECK(n - r + pos <= ctx, "prompt of %d tokens does not fit the context (%d)", n, ctx);
    if (r == n) {  // identical prompt: its prompt-end snapshot holds the last row's logits and hidden (mh)
        mla_set(pos);  // round 26: exactly the pages of the rows kept
        const Ckpt& c = ck[ri];
        last_tok = c.last_tok;
        if (lout) memcpy(lout, c.logits.data(), (size_t)VOCAB * 4);
        if (ck_log) fprintf(stderr, "[glm] prefix snapshot: %d of %d tokens, nothing to prefill\n", r, n);
        return r;
    }
    std::vector<int> cut;
    if (ck_on) {
        if (sp_hint >= ck_hint_min && sp_hint > r && sp_hint < n) cut.push_back(sp_hint);
        if (lcp - r >= ck_div_min && lcp > r && lcp < n) cut.push_back(lcp);
        if (ck_tail > 0 && n - r >= ck_tail_min && n - ck_tail > r) cut.push_back(n - ck_tail);
        for (int c : ck_dbg_cuts) if (c > r && c < n) cut.push_back(c);
        ck_dbg_cuts.clear();
        std::sort(cut.begin(), cut.end());
        cut.erase(std::unique(cut.begin(), cut.end()), cut.end());
    }
    cut.push_back(n);
    // round 26: exactly the pages of the rows kept + the first prefill call (GLM_DET: the seed minus those pages,
    // whatever came before; review r26 M1: a release does not refill slots the first prefill takes right back)
    mla_set(cut.front());
    int a = r;
    for (int b : cut) {
        std::vector<std::pair<int, int>> kinds;  // (position, kind)
        if (ck_on) {
            if (ck_win) {
                const int w = pf_window(b - a);
                for (int e = a + w; e < b; e += w) kinds.push_back({e, 0});
            }
            kinds.push_back({b, b == n ? 2 : 1});
        }
        ck_plan.clear();
        std::vector<std::pair<int, int>> plan_kind;
        for (auto& pk : kinds) {
            const int i = ck_take(pk.first, pk.second);
            if (i < 0) continue;
            ck_plan.push_back({pk.first, i});
            plan_kind.push_back(pk);
        }
        const auto plan = ck_plan;
        mtp_prev = mtp && a > 0 && (restored || a > r);
        if (ck_on) lv_push(a, b);
        const auto t0 = std::chrono::steady_clock::now();
        float* lo = b == n ? (lout ? lout : (ck_lout.empty() ? nullptr : ck_lout.data())) : nullptr;
        // [glm-rel3-cancel] the first call of a live continuation saves its start for a cancel (GLM_PF_KEEP_MIN rows)
        // (review R3-M1: every live continuation, GLM_PF_KEEP_MIN default 0: an unsaved one stops only at window starts)
        static const int keep_min = env_int("GLM_PF_KEEP_MIN", 0);
        pf_keep = a == r && r == pos && !restored && r > 0 && (int)sp_hist.size() == r && b - a >= keep_min;
        const bool live0 = pf_keep;
        forward(ids + a, b - a, lo, nullptr);
        pf_keep = false;
        if (pf_stop_rows >= 0) {  // [glm-rel-pfgate] cancelled: whole windows [a, a + rows) are live, the rest never ran
            b = a + pf_stop_rows;
            pf_aborted = true;
            if (ck_on && lv_end() > b) lv_cut(b);
        }
        sp_hist.insert(sp_hist.end(), ids + a, ids + b);
        for (size_t j = 0; j < plan.size(); j++) {
            Ckpt& c = ck[plan[j].second];
            if (plan[j].first > b) { c.q = -1; continue; }  // (planned past the cancel: never written)
            c.q = plan[j].first;
            c.kind = plan_kind[j].second;
            c.hash = tok_hash(ids, c.q);
            c.toks.assign(ids, ids + c.q);
            c.used = ++ck_tick;
            c.segs = lv;
            lv_clip(c.segs, c.q);
            c.last_tok = c.kind == 2 ? last_tok : -1;
            if (c.kind == 2 && lo) c.logits.assign(lo, lo + VOCAB);
            else c.logits.clear();
            st_ck_save++;
        }
        if (ck_log)
            fprintf(stderr, "[glm] prefill [%d, %d) of %d: %.0f ms, %zu snapshot(s)%s\n", a, b, n,
                    std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count(), plan.size(),
                    a == r && restored ? " (resumed from a snapshot)" : "");
        a = b;
        ck_plan.clear();  // review R3-L8: no stale indices protected by later takes
        if (pf_aborted) {
            // the live history is dropped (the MTP row b - 1 was never written): the next prompt resumes from the
            // window-end snapshot at b (ck_restore writes that row), or prefills.
            // [glm-rel3-cancel] a live continuation stopped in its first window is back at its start (forward restored
            // the KDA / conv / ring state, no row below it changed): it stays live, its answer-end snapshot still due
            // (review R3-M1: stopped before its first window ran: nothing changed, it stays live without a save too)
            if (pf_live_kept || (live0 && b == r)) ck_ans_due = ans_was;
            else sp_hist.clear();
            break;
        }
    }
    return r;
}

// round 25 ([glm-p3-turnend], plan3 P2): snapshot at the end of a decoded reply (qe_release, before the cache goes
// back to the seed). Without it a conversation whose rows another one overwrote resumed at its previous prompt end and
// prefilled its own last answer again. The cache holds sp_hist = prompt + reply minus its last token (spec_step stops
// at the server's limit / EOS); the hidden of row pos - 1 is the last verify's final-normed row sp_last_a - 1 (hfin).
void Engine::ck_answer_end() {
    if (!ck_inited || !ck_on || !mtp || sp_last_a <= 0 || pos <= 0 || (int)sp_hist.size() != pos || ver_pending) return;
    if (env_int("GLM_CK_ANS", 1) == 0) return;
    for (auto& c : ck)  // already saved (a repeated release)
        if (c.q == pos && (int)c.toks.size() == pos && !memcmp(c.toks.data(), sp_hist.data(), (size_t)pos * 4)) return;
    const auto t0 = std::chrono::steady_clock::now();
    if (lv_end() > pos) lv_cut(pos);
    if (lv_end() < pos) lv_push(lv_end(), pos);  // the reply's rows: live rows of their own writer
    const int i = ck_take(pos, 3);
    if (i < 0) return;
    kda_mat();  // pointer mode: S is the committed state from here on (idempotent)
    const int row = sp_last_a - 1;
    run_all([&](Worker& w) {
        CUDA_CHECK(cudaMemcpyAsync(w.ck_h, w.hfin + (size_t)row * H, H * 4, cudaMemcpyDeviceToDevice, w.st));
        w.ck_io(ck[i].buf[w.p.rank], pos, true);
    });
    Ckpt& c = ck[i];
    c.q = pos;
    c.kind = 3;
    c.hash = tok_hash(sp_hist.data(), pos);
    c.toks = sp_hist;
    c.used = ++ck_tick;
    c.segs = lv;
    lv_clip(c.segs, c.q);
    c.last_tok = -1;
    c.logits.clear();
    st_ck_save++;
    if (ck_log)
        fprintf(stderr, "[glm] answer-end snapshot at %d: %.1f ms\n", pos,
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
}


// ---- release track [glm-rel-slice]: park / resume a reply between two steps (time slicing between conversations) ----
int Engine::ck_find_tag(unsigned long long tag) const {
    if (!tag) return -1;
    for (size_t i = 0; i < ck.size(); i++)
        if (ck[i].pin && ck[i].tag == tag) return (int)i;
    return -1;
}

// every queued copy and kernel of the bound sequence has landed (boundary / admission copies included), the
// coordinator is idle: the slot table on the host is the slab's content
void Engine::conv_quiesce() {
    bg_flush();
    coord_idle();
    det_wait();
    run_all([&](Worker& w) {
        for (cudaStream_t s_ : {w.st, w.cst, w.bst, w.dmst, w.ast, w.pst, w.krst, w.gst, w.cms, w.ags, w.ost})
            if (s_) CUDA_CHECK(cudaStreamSynchronize(s_));
    });
}

// the MTP cache row r (= the token at r + 1) and the pool key of its pool, per card (each card holds every MLA layer)
static void mtp_row_io(Worker& w, std::vector<char>& b, int r, bool save) {
    if (!w.E->mtp || r < 0 || (int)w.L.size() <= MTP_L || !w.L[MTP_L].kc) { b.clear(); return; }
    CUDA_CHECK(cudaSetDevice(w.dev));
    Layer& Y = w.L[MTP_L];
    const Engine& E = *w.E;
    const size_t kst = E.mla_pos ? (size_t)E.mla_nl * KVLR : KVLR, pst = E.mla_pos ? (size_t)E.mla_nl * ID : ID;
    const size_t kb = (size_t)KVLR * 2, pb = (size_t)ID * 4;
    if (save) b.resize(kb + pb);
    else if (b.size() != kb + pb) return;
    char* d0 = (char*)(Y.kc + (size_t)r * kst);
    char* d1 = (char*)(Y.pk + (size_t)(r / POOL) * pst);
    if (save) {
        CUDA_CHECK(cudaMemcpyAsync(b.data(), d0, kb, cudaMemcpyDeviceToHost, w.st));
        CUDA_CHECK(cudaMemcpyAsync(b.data() + kb, d1, pb, cudaMemcpyDeviceToHost, w.st));
    } else {
        CUDA_CHECK(cudaMemcpyAsync(d0, b.data(), kb, cudaMemcpyHostToDevice, w.st));
        CUDA_CHECK(cudaMemcpyAsync(d1, b.data() + kb, pb, cudaMemcpyHostToDevice, w.st));
    }
    CUDA_CHECK(cudaStreamSynchronize(w.st));
}

bool Engine::conv_park(ConvPark& p) {
    if (!ck_inited) { ck_init(); ck_inited = true; }
    if (!ck_on || !arc_on || !mtp || lent || ver_pending || pos <= 0 || (int)sp_hist.size() != pos) return false;
    const auto t0 = std::chrono::steady_clock::now();
    conv_quiesce();
    if (!bg_pl.empty()) return false;  // (not reached: a step's background copies are joined at its verify's end)
    kda_mat();  // pointer mode: S is the committed state (the same values the next verify reads from the snapshot row)
    if (lv_end() > pos) lv_cut(pos);
    if (lv_end() < pos) lv_push(lv_end(), pos);  // the reply's rows: live rows of their own writer
    const int i = ck_take(pos, 4);
    if (i < 0) return false;
    Ckpt& c = ck[i];
    const int row = sp_last_a - 1;  // (-1 right after the start: the hidden is not read by the resumed step anyway)
    p.mtp_row.assign(ngpu, {});
    run_all([&](Worker& w) {
        if (row >= 0) CUDA_CHECK(cudaMemcpyAsync(w.ck_h, w.hfin + (size_t)row * H, H * 4, cudaMemcpyDeviceToDevice, w.st));
        w.ck_io(c.buf[w.p.rank], pos, true);
        mtp_row_io(w, p.mtp_row[w.p.rank], pos - 1, true);
    });
    c.q = pos;
    c.kind = 4;
    c.pin = true;
    c.tag = p.tag = ++park_tag;
    c.hash = 0;
    c.toks = sp_hist;
    c.used = (1ull << 62) + ++ck_tick;  // its archive pages go last (arc_block / arc_trim are LRU on this tick)
    c.segs = lv;
    while (!c.segs.empty() && c.segs.back()[0] >= c.q) c.segs.pop_back();
    if (!c.segs.empty() && c.segs.back()[1] > c.q) c.segs.back()[1] = c.q;
    c.last_tok = -1;
    c.logits.clear();
    arc_touch(c);
    // host state of the reply
    p.pos = pos; p.sp_t = sp_t; p.sp_k = sp_k; p.sp_kcur = sp_kcur; p.sp_reused = sp_reused; p.last_tok = last_tok;
    p.sp_last_a = sp_last_a;
    for (int j = 0; j < MR; j++) { p.sp_d[j] = sp_d[j]; p.sp_c[j] = sp_c[j]; }
    p.sp_hist.swap(sp_hist);
    sp_hist.clear();  // (nothing live continues this history while it is parked)
    p.sp_lout = sp_lout;
    // expert cache state (det_restore's set) and the MLA pages
    p.owner = slot_owner;
    p.free_ = free_slots;
    p.score = score;
    p.rng = rng;
    p.pend.clear();
    for (size_t s = 0; s < bg_pend.size(); s++) if (bg_pend[s]) p.pend.push_back((int)s);
    p.qprev = bg_qprev;
    p.mla = slot_mla;
    p.mla_rows = mla_rows;
    p.dirty = det_dirty;
    p.lend_canon = lend_canon;
    sp_undo = false;  // (a trim of the parked step is no longer possible)
    st_sw_park++;
    st_sw_park_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    return true;
}

// review R3-M2: a parked reply released while it waited (its client left): its pinned entry is the answer end of the
// turn so far (rows [0, pos), KDA / rings / hidden at pos): it becomes an ordinary answer-end snapshot
void Engine::conv_keep(ConvPark& p) {
    const int i = ck_find_tag(p.tag);
    if (i >= 0 && ck[i].q > 0) {
        Ckpt& c = ck[i];
        c.pin = false;
        c.tag = 0;
        c.kind = 3;
        c.hash = tok_hash(c.toks.data(), c.q);
        c.used = ++ck_tick;
        for (auto& pg : arc) pg.used = 0;  // (the pages' tick from their snapshots again, as arc_save does)
        for (auto& x : ck) if (x.q > 0) arc_touch(x);
        p.tag = 0;
    }
    conv_drop(p);
}
void Engine::conv_drop(ConvPark& p) {
    const int i = ck_find_tag(p.tag);
    if (i >= 0) {
        ck[i].q = -1;
        ck[i].pin = false;
        ck[i].tag = 0;
        ck[i].toks.clear();
        ck[i].segs.clear();
    }
    p.tag = 0;
    p.sp_hist.clear();
    p.mtp_row.clear();
}

int Engine::conv_resume(ConvPark& p) {
    CHECK(!lent, "resume while the prefill set is lent");
    const auto t0 = std::chrono::steady_clock::now();
    conv_quiesce();
    if (ck_ans_due) {  // the bound reply just ended: its answer-end snapshot first (as a next prompt's ck_prefill does)
        ck_ans_due = false;
        ck_answer_end();
    }
    const int i = ck_find_tag(p.tag);  // (after ck_answer_end: its take may move entries)
    if (i < 0 || ck[i].q != p.pos || !ck_valid(ck[i])) {  // its archived rows were given back (host memory short)
        st_sw_lost++;
        return 0;
    }
    Ckpt& c = ck[i];
    // the live rows from the first one this reply does not share on are overwritten: what other snapshots (parked
    // replies included) need is archived first (as ck_restore)
    int x = c.q;
    for (size_t j = 0; j < lv.size(); j++) {
        if (j >= c.segs.size()) break;
        const Seg &a = lv[j], &b = c.segs[j];
        if (a == b) continue;
        x = a[0] == b[0] && a[2] == b[2] ? std::min(a[1], b[1]) : a[0];
        break;
    }
    arc_save(std::min(x, c.q), c.used);
    if (!ck_valid(c)) {  // (not reached: the save keeps pinned pages)
        st_sw_lost++;
        return 0;
    }
    // the expert cache state of the reply: only the slots whose expert differs are copied (a slot under the current
    // MLA pages has no expert); the MLA pages of the reply are held again before its rows are copied back
    const int ns = (int)slot_owner.size();
    std::vector<std::pair<int, int>> sk, mv;  // (slot, key) from host memory; (slot, source slot) on the card
    {
        std::vector<char> dst(ns, 0);
        for (int s = 0; s < ns; s++) if (p.owner[s] >= 0 && slot_owner[s] != p.owner[s]) dst[s] = 1;
        // an expert the slab holds in another slot that this restore does not overwrite: copied on the card (the same
        // bytes, ~40x faster than PCIe); the rest from host memory
        for (int s = 0; s < ns; s++) {
            if (!dst[s]) continue;
            const int src = slot_of[p.owner[s]];
            if (src >= 0 && src != s && !dst[src]) mv.push_back({s, src});
            else sk.push_back({s, p.owner[s]});
        }
    }
    if (!mv.empty())
        run_all([&](Worker& w) {
            const size_t b = w.sl.bytes;
            for (auto& m : mv)
                CUDA_CHECK(cudaMemcpyAsync(w.slab + (size_t)m.first * b, w.slab + (size_t)m.second * b, b,
                                           cudaMemcpyDeviceToDevice, w.cst));
            CUDA_CHECK(cudaStreamSynchronize(w.cst));
        });
    slot_owner = p.owner;
    std::fill(slot_of.begin(), slot_of.end(), -1);
    for (int s = 0; s < ns; s++) if (slot_owner[s] >= 0) slot_of[slot_owner[s]] = s;
    free_slots = p.free_;
    score = p.score;
    rng = p.rng;
    std::fill(bg_pend.begin(), bg_pend.end(), 0);
    for (int s : p.pend) bg_pend[s] = 1;
    bg_pl.clear();
    bg_qprev = p.qprev;
    slot_mla = p.mla;
    mla_rows = p.mla_rows;
    det_dirty = p.dirty;
    lend_canon = p.lend_canon;
    fill_slots(sk);
    upload_tables();
    if (!arc_load(c)) {  // (not reached: ck_valid above)
        fprintf(stderr, "[glm] parked reply at %d lost its archived rows\n", c.q);
        st_sw_lost++;
        return -1;  // the cache table is the reply's, its rows are not: the caller re-prefills
    }
    lv = c.segs;
    ver_pending = false;
    pos = c.q;
    run_all([&](Worker& w) {
        w.ck_io(c.buf[w.p.rank], c.q, false);
        mtp_row_io(w, p.mtp_row[w.p.rank], pos - 1, false);
        // review R3-L2: the hidden of row pos - 1 (loaded into mh) back where a park / answer end right after reads it
        if (p.sp_last_a > 0)
            CUDA_CHECK(cudaMemcpyAsync(w.hfin + (size_t)(p.sp_last_a - 1) * H, w.mh, H * 4, cudaMemcpyDeviceToDevice, w.st));
        CUDA_CHECK(cudaMemcpyAsync(w.d_pos, &pos, 4, cudaMemcpyHostToDevice, w.st));
        CUDA_CHECK(cudaMemsetAsync(w.d_acc, 0, 8, w.st));  // (no pending pointer-mode KDA state: S was materialised)
        CUDA_CHECK(cudaStreamSynchronize(w.st));
    });
    sp_t = p.sp_t; sp_k = p.sp_k; sp_kcur = p.sp_kcur; sp_reused = p.sp_reused; last_tok = p.last_tok;
    sp_last_a = p.sp_last_a;
    for (int j = 0; j < MR; j++) { sp_d[j] = p.sp_d[j]; sp_c[j] = p.sp_c[j]; }
    sp_hist.swap(p.sp_hist);
    sp_lout.swap(p.sp_lout);
    sp_undo = false;
    commit_in_mtp = false;
    conv_drop(p);
    st_sw_res++;
    st_sw_slots += sk.size();
    st_sw_d2d += mv.size();
    st_sw_res_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    return 1;
}

}  // namespace glm
