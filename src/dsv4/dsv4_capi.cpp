#include <thread>
// Release prep [dsv4-rel-capi]: C API of the DSV4 engine for the server (ctypes): one sequence (one KV cache).
// The defaults are the measured best GGUF-only configuration (DSpark on, confidence gate 0.5, cache-aware routing off,
// 32K context); the arguments string takes the dsv4_run flags that make sense in a server.
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <algorithm>
#include <cstring>
#include <sstream>
#include <string>
#include <vector>
#include <list>
#include <unistd.h>

#include "dsv4.h"

using namespace dsv4;

namespace {
struct H {
    Engine* e = nullptr;
    Options o;
    std::vector<int> toks;   // the sequence as the engine holds it (prompt + emitted tokens except the next input)
    int next = -1;           // the next input token (the last emitted one), -1 = none
    std::vector<int> snap;   // [dsv4-srv-pesnap] tokens of the engine's prompt-end snapshot (the last prompt without its
                             // last token), empty = none
    int tier_reset = 0;      // [dsv4-rel-tierreset]: 1 = the dynamic tier back to its post-load state after each request
    std::thread rt;          // that reset, run after the request (the next call waits for it)
    double rt_ms = 0;
    void wait() {
        if (rt.joinable()) rt.join();
    }
    // [dsv4-rel1-arch] idle conversations in host RAM (LRU, front = oldest): the engine's blob + the C API's view
    struct Entry {
        std::vector<int> toks, snap;
        int next = -1;
        std::vector<uint8_t> blob;
        int snap_ok = 0;   // the blob holds the prompt-end snapshot at snap.size()
        int rid = 0;       // [dsv4-rel4-p2] the engine's parked id (its compressed K/V: VRAM pool or engine host copy)
        int live = 0;      // [dsv4-rel6-sched] a request still in service (its id): never a reuse source, never superseded
        int pfx = 0;       // [dsv4-rel6-xprefix] a shared prompt prefix: copied on use (stays), never superseded
    };
    std::list<Entry> arch;
    size_t arch_bytes = 0;
    double arch_cap_gib = 32;   // --archive-gib (0 = off)
    size_t park_failed = 0;     // [dsv4-rel9] review M1: a give-back park that did not fit, tried once per held length
    int last_src = 0;           // the last start's reuse: 0 none, 1 held sequence, 2 snapshot, 3 archive
    int live = 0;               // [dsv4-rel6-sched] the held conversation belongs to a request in service (its id), 0 = none
    int pfx_hint = 0;           // [dsv4-rel6-xprefix] the next start's shared prefix length (the system part), 0 = none
    int pf_follow = 0;          // [dsv4-xcall] the next dsv4_prefill_part: prompt tokens later prefill calls still feed
    long n_pfx_made = 0, n_pfx_used = 0;
    long n_saved = 0, n_loaded = 0, n_dropped = 0, n_skipped = 0;
    int nocache = 0;
    long steps = 0;            // the next start reuses nothing (one shot; cache_prompt false)
    // [dsv4-rel6-sched] DSV4_STATS=1 (benches): decode pass statistics summed over dsv4_step calls (as dsv4_run's spec
    // job reports them); off by default (the step then passes no timer struct, as before)
    bool stats = getenv("DSV4_STATS") && atoi(getenv("DSV4_STATS")) == 1;
    double st_ms = 0, st_lane = 0, st_route = 0, st_hostb = 0, st_laneb = 0;
    long st_pass = 0, st_tok = 0;
};
bool parse(const char* args, Options& o, std::string& err) {
    std::istringstream is(args ? args : "");
    std::string a;
    auto nx = [&](std::string& v) { return (bool)(is >> v); };
    while (is >> a) {
        std::string v;
        if (!nx(v)) { err = "missing value for " + a; return false; }
        if (a == "--gguf") o.gguf_model = v;
        else if (a == "--dspark-dir") o.dspark_dir = v;
        else if (a == "--cache-dir") o.cache_dir = v;
        else if (a == "--profile") o.profile = v;
        else if (a == "--ctx") o.maxctx = atoi(v.c_str());
        else if (a == "--ctx-static") o.ctx_static = std::max(0, atoi(v.c_str()));   // [dsv4-rel8-ctxdef]
        else if (a == "--kv-pool-mib") o.kv_pool_mib = std::max(0, atoi(v.c_str()));   // [dsv4-rel4-p2]
        else if (a == "--dspark") o.dspark = atoi(v.c_str());
        else if (a == "--dspark-tau") o.dspark_tau = (float)atof(v.c_str());
        else if (a == "--car-margin") o.car_margin = (float)atof(v.c_str());
        else if (a == "--drop-w") {   // [dsv4-rel2-dropw] 0 = exact routing (all 6); review dsv4_rel2 rel2b L1: validated
            char* end = nullptr;
            const float dv = strtof(v.c_str(), &end);
            if (end == v.c_str() || *end != 0 || !std::isfinite(dv) || dv < 0.f || dv > 1.5f) {
                err = "--drop-w: a value in [0, 1.5] (0 = exact routing)";
                return false;
            }
            o.drop_w = dv;
        }
        else if (a == "--cpu-threads") o.cpu_threads = atoi(v.c_str());
        else if (a == "--dyn-window") o.dyn_window = atoi(v.c_str());   // [dsv4-rel6-sched] tests: -1 = frozen tier
        else if (a == "--reserve-gb") o.reserve_gb = atof(v.c_str());
        else if (a == "--chunk") o.chunk = atoi(v.c_str());
        else if (a == "--host-budget-gib") o.host_budget_gib = atof(v.c_str());
        else if (a == "--tier-reset" || a == "--archive-gib") continue;   // (read by dsv4_create)
        else { err = "unknown argument " + a; return false; }
    }
    if (o.gguf_model.empty()) { err = "--gguf DIR is required"; return false; }
    return true;
}
// [dsv4-rel1-arch] tokens of `toks` (+ the pending `next`) that a prompt reuses as a whole-sequence prefix, or the
// snapshot length when the prompt extends the snapshot (snap_ok) and that is longer; mode 1 / 2, 0 = nothing
int reusable(const std::vector<int>& toks, int next, const std::vector<int>& snap, bool snap_ok, const int* ids, int n,
             int* mode) {
    int k = 0;
    const int m = (int)toks.size();
    while (k < m && k < n && toks[k] == ids[k]) k++;
    if (k != m) k = 0;
    if (k && next >= 0 && k < n && ids[k] != next) k = 0;
    if (k == n) k = 0;
    *mode = k ? 1 : 0;
    const int P = (int)snap.size();
    if (snap_ok && k < P && P < n && std::equal(snap.begin(), snap.end(), ids)) {
        k = P;
        *mode = 2;
    }
    return k;
}
double meminfo_gib(const char* key) {
    FILE* f = fopen("/proc/meminfo", "r");
    if (!f) return -1;
    char line[256];
    double v = -1;
    const size_t kl = strlen(key);
    while (fgets(line, sizeof line, f))
        if (!strncmp(line, key, kl) && line[kl] == ':') {
            v = atof(line + kl + 1) / (1024.0 * 1024.0);
            break;
        }
    fclose(f);
    return v;
}
// the archive's floor: it gives memory back before the engine's memory guard (DSV4_GUARD_GIB, 8 GiB, freeing up to
// + 4) drops expert host copies: guard + 4 + max(4 GiB, 3 % of RAM)
double arch_floor_gib() {
    static const double g = getenv("DSV4_GUARD_GIB") ? atof(getenv("DSV4_GUARD_GIB")) : 8.0;
    static const double tot = meminfo_gib("MemTotal");
    return g + 4.0 + std::max(4.0, 0.03 * std::max(0.0, tot));
}
// [dsv4-rel4-p2] host bytes of an entry: its blob + the engine's host copy of its compressed K/V (0 while in VRAM)
size_t ebytes(H* h, const H::Entry& e) { return e.blob.size() + (e.rid ? seq_res_host(h->e, e.rid) : 0); }
size_t arch_host(H* h) {   // the archive's host bytes (blobs + the engine's host copies of parked K/V)
    double hm = 0;
    seq_res_info(h->e, nullptr, nullptr, &hm, nullptr);
    return h->arch_bytes + (size_t)(hm * 1e6);
}
void arch_drop_front(H* h, const char* why) {
    H::Entry& e = h->arch.front();
    fprintf(stderr, "[dsv4] archive: dropped a conversation of %zu tokens (%.1f MB, %s); %zu left\n", e.toks.size(),
            ebytes(h, e) / 1e6, why, h->arch.size() - 1);
    h->arch_bytes -= e.blob.size();
    if (e.rid) seq_res_free(h->e, e.rid);
    h->arch.pop_front();
    h->n_dropped++;
}
// host memory pressure: LRU entries go while MemAvailable is below the floor
void arch_pressure(H* h) {
    double av = meminfo_gib("MemAvailable");
    const double fl = arch_floor_gib();
    if (av >= 0 && av < fl && h->e) {   // [dsv4-rel5-p3] the engine's cache of free pinned buffers goes before any entry
        seq_res_trim(h->e);
        av = meminfo_gib("MemAvailable");
    }
    while (!h->arch.empty() && av >= 0 && av < fl) {
        const double b = ebytes(h, h->arch.front()) / (double)(1 << 30);
        arch_drop_front(h, "host memory pressure");
        av += b;
    }
    // review rel4 M1: the engine's host copies of evicted K/V count against --archive-gib too
    while (!h->arch.empty() && arch_host(h) / (double)(1 << 30) > h->arch_cap_gib) arch_drop_front(h, "archive full");
    // review rel5 M2: the engine's cache of free pinned buffers fits in what the archive leaves under its cap
    double cmb = 0;
    if (h->e) seq_res_hostinfo(h->e, nullptr, &cmb, nullptr);
    if (cmb > 0 && (arch_host(h) + cmb * 1e6) / (double)(1 << 30) > h->arch_cap_gib) seq_res_trim(h->e);
}
// the active sequence into the archive (before the engine moves to another one). Entries it supersedes (a prefix of
// it) go; LRU entries make room; a save that cannot fit even with the archive empty evicts nothing and is skipped
void arch_save_active(H* h, int skip_rid = 0) {
    if (h->arch_cap_gib <= 0 || h->toks.empty()) return;
    std::vector<int> seq = h->toks;
    arch_pressure(h);
    const size_t need = seq_bytes(h->e, true);   // [dsv4-rel4-p2] parked: the compressed K/V stays in the VRAM pool
    const double av = meminfo_gib("MemAvailable");
    // review rel4 M4: skip_rid = the entry being restored (left the archive; its host copy goes with the unpark)
    auto held_now = [&] {
        const size_t a = arch_host(h), s = skip_rid ? seq_res_host(h->e, skip_rid) : 0;
        return a > s ? a - s : 0;
    };
    const size_t held = held_now();
    const double room = std::min(h->arch_cap_gib, (av < 0 ? h->arch_cap_gib : av - arch_floor_gib()) +
                                                      held / (double)(1 << 30));
    if (need == 0 || need / (double)(1 << 30) > room) {
        if (need) {
            h->n_skipped++;
            fprintf(stderr, "[dsv4] archive: a conversation of %zu tokens (%.1f MB) not kept: %.2f GiB room\n", seq.size(),
                    need / 1e6, room);
        }
        return;
    }
    for (auto it = h->arch.begin(); it != h->arch.end();) {   // superseded: its tokens are a prefix of this one
        if (!it->live && !it->pfx && it->toks.size() <= seq.size() && std::equal(it->toks.begin(), it->toks.end(), seq.begin()) &&
            (it->next < 0 || it->toks.size() == seq.size() || seq[it->toks.size()] == it->next)) {
            h->arch_bytes -= it->blob.size();
            if (it->rid) seq_res_free(h->e, it->rid);
            it = h->arch.erase(it);
        } else {
            ++it;
        }
    }
    while (!h->arch.empty() && (held_now() + need) / (double)(1 << 30) > room)
        arch_drop_front(h, "archive full");
    H::Entry e;
    bool snap_pre = false;
    try {   // review rel1 L: an allocation failure skips the save (degrade), never escapes the C API
        snap_pre = snap_pos(h->e) == (int)h->snap.size();   // (before the park resets the engine)
        if ((e.rid = seq_park(h->e, e.blob)) == 0) return;   // (the engine holds no sequence after a park)
    } catch (const std::bad_alloc&) {
        h->n_skipped++;
        fprintf(stderr, "[dsv4] archive: no host memory for a conversation of %zu tokens: not kept\n", seq.size());
        return;
    }
    e.toks.swap(seq);
    e.next = h->next;
    e.live = h->live;   // [dsv4-rel6-sched]
    h->live = 0;
    e.snap = h->snap;
    e.snap_ok = !h->snap.empty() && snap_pre;
    h->toks.clear();   // [dsv4-rel4-p2] parked: the engine holds no sequence now
    h->next = -1;
    h->snap.clear();
    h->arch_bytes += e.blob.size();
    h->arch.push_back(std::move(e));
    h->n_saved++;
}
// review rel4 M2: entries whose parked K/V the engine dropped under memory pressure leave the archive
void arch_prune(H* h) {
    for (auto it = h->arch.begin(); it != h->arch.end();)
        if (it->rid && !seq_res_alive(h->e, it->rid)) {
            h->arch_bytes -= it->blob.size();
            seq_res_free(h->e, it->rid);
            it = h->arch.erase(it);
            h->n_dropped++;
        } else {
            ++it;
        }
}
}  // namespace

extern "C" {
// args: "--gguf DIR [--ctx N] [--dspark 0|1] ..."; null on a bad argument (message on stderr)
void* dsv4_create(const char* args) {
    H* h = new H();
    h->o.dspark = 1;
    h->o.dspark_tau = 0.5f;    // measured best (R16-R23 duel config)
    h->o.car_margin = -1.f;    // cache-aware routing off with DSpark (measured)
    // [dsv4-dec7-mg] user 11-17: opt-in routing margin (approximates the routing: a cold expert choice within the
    // margin of a resident one is swapped for it, so a different expert runs); DSV4_ROUTE_MARGIN=<value>, --car-margin
    // in the args wins; default off
    if (const char* rm = getenv("DSV4_ROUTE_MARGIN"); rm && *rm) {   // review mg L1: a finite value in (0, 1], else off
        char* end = nullptr;
        const float v = strtof(rm, &end);
        if (end != rm && *end == 0 && std::isfinite(v) && v > 0.f && v <= 1.f) h->o.car_margin = v;
        else if (!(end != rm && *end == 0 && v <= 0.f))
            fprintf(stderr, "dsv4_create: DSV4_ROUTE_MARGIN=%s ignored (a value in (0, 1]; <= 0 = off)\n", rm);
    }
    // [dsv4-rel2-dropw] the R8 adaptive top-k (drop a 5th / 6th choice whose weight is below drop_w; lossy; default 0 = off):
    // DSV4_DROP_W=<value in [0, 1.5]> (0 = exact routing), --drop-w in the args wins
    if (const char* dw = getenv("DSV4_DROP_W"); dw && *dw) {
        char* end = nullptr;
        const float v = strtof(dw, &end);
        if (end != dw && *end == 0 && std::isfinite(v) && v >= 0.f && v <= 1.5f) h->o.drop_w = v;
        else fprintf(stderr, "dsv4_create: DSV4_DROP_W=%s ignored (a value in [0, 1.5]; 0 = exact routing)\n", dw);
    }
    h->o.maxctx = 32768;
    std::string err;
    if (!parse(args, h->o, err)) {
        fprintf(stderr, "dsv4_create: %s\n", err.c_str());
        delete h;
        return nullptr;
    }
    if (h->o.car_margin >= 0.f)   // review mg M2: approximate routing is never silent
        fprintf(stderr, "dsv4_create: routing margin %.3f ON: approximate routing (a cold expert choice within the margin of a "
                        "resident one is replaced by it)\n", h->o.car_margin);
    if (h->o.drop_w > 0.f)   // review dsv4_rel2 rel2b L2: lossy routing is never silent either
        fprintf(stderr, "dsv4_create: drop_w %.3f ON: adaptive top-k (a 5th / 6th expert choice below this weight is left "
                        "out: lossy)\n", h->o.drop_w);
    {   // --tier-reset 0|1 (default 0)
        std::istringstream is(args ? args : "");
        std::string a, v;
        while (is >> a)
            if (a == "--tier-reset" && (is >> v)) h->tier_reset = atoi(v.c_str());
            else if (a == "--archive-gib" && (is >> v)) h->arch_cap_gib = std::max(0.0, atof(v.c_str()));
    }
    h->e = create(h->o);
    if (h->e) h->o.maxctx = engine_max_ctx(h->e);   // [dsv4-rel8-ctxcap] a load that could not map it all capped it
    if (h->e) set_lz_seed(h->e, h->tier_reset != 0);   // [dsv4-rel8-giveback] review rel8 M1
    if (h->e) seq_res_floor(h->e, arch_floor_gib());   // review rel4 M1: evictions keep host copies above the archive's floor
    set_car(h->e, h->o.car_margin);
    set_dspark(h->e, -1, h->o.dspark_tau);
    return h;
}
void dsv4_destroy(void* p) {
    H* h = (H*)p;
    if (!h) return;
    h->wait();
    destroy(h->e);
    delete h;
}
int dsv4_vocab() { return V; }
int dsv4_eos() { return EOS; }
int dsv4_max_ctx(void* p) { return ((H*)p)->o.maxctx; }
// temperature <= 0: greedy; top_k 0: 64 (the sampler's candidates)
void dsv4_set_sampling(void* p, float temperature, float top_p, int top_k, float min_p, unsigned long long seed) {
    ((H*)p)->wait();
    set_sampling(((H*)p)->e, temperature, top_p, top_k, min_p, seed);
}
// tokens the engine already holds that prefix ids (the reusable part of a new prompt)
int dsv4_common_prefix(void* p, const int* ids, int n) {
    H* h = (H*)p;
    int k = 0;
    const int m = (int)h->toks.size();
    while (k < m && k < n && h->toks[k] == ids[k]) k++;
    return k == m ? k : 0;   // only a whole-sequence prefix continues (no truncation inside the KV)
    // ([dsv4-pf15-cone]: after a coned prefill the upper layers' window K/V is valid only on the last positions; a
    //  partial-prefix continuation would need a re-prefill: keep it whole-sequence or reset)
}
// start a turn: ids = the whole prompt; reuses the held sequence when it is a prefix (reuse_out: tokens reused).
// Returns the first generated token (sampled with the current parameters), -1 on a context overflow.
static int start_impl(H* h, const int* ids, int n, int* reuse_out, bool part) {
    h->wait();
    const int hint = h->pfx_hint;   // [dsv4-rel6-xprefix] one shot, cleared before any return (review rel6x L1)
    h->pfx_hint = 0;
    // [dsv4-xcall] one shot too; a dsv4_start ends the prompt (its head row is the reply's first token)
    static const bool noxcall = getenv("DSV4_NOXCALL") != nullptr;   // off-switch
    const int follow = part && !noxcall ? h->pf_follow : 0;
    h->pf_follow = 0;
    if (n < 1 || n + 16 > h->o.maxctx) return -1;   // (before anything moves: the held sequence stays)
    if (part && h->next < 0 && (int)h->toks.size() == n && std::equal(h->toks.begin(), h->toks.end(), ids)) {
        if (reuse_out) *reuse_out = n;   // [dsv4-rel6-sched] this part is held already
        h->nocache = 0;   // (review rel6 L2)
        return 0;
    }
    h->last_src = 0;
    // [dsv4-rel6-xprefix] a second prompt with the same shared part (>= 1,024 tokens) as a conversation held or archived
    // (not a request in service, not a prefix entry): that part is fed on its own first and kept as a prefix entry
    auto lcp = [&](const std::vector<int>& t) {
        int q = 0;
        while (q < (int)t.size() && q < n && t[q] == ids[q]) q++;
        return q;
    };
    bool shared_seen = false;
    if (hint >= 1024 && hint < n && h->arch_cap_gib > 0 && !h->nocache) {
        // (review rel6x M1: requests in service count too: only their tokens are compared)
        shared_seen = lcp(h->toks) >= hint;
        for (auto& e : h->arch)
            if (!shared_seen && !e.pfx && lcp(e.toks) >= hint) shared_seen = true;
        // review rel6x M2 / L6: split only when an entry of about this size can be kept (after LRU evictions of idle
        // conversations): ~3.3 KB per token of compressed K/V + index keys and ~24 MB of window rings
        const double av = meminfo_gib("MemAvailable");
        const double room = std::min(h->arch_cap_gib, av < 0 ? h->arch_cap_gib : av - arch_floor_gib() + arch_host(h) / (double)(1 << 30));
        if ((3400.0 * hint + 24e6) / (double)(1 << 30) > room) shared_seen = false;
    }
    if (h->nocache) {   // [dsv4-rel1-arch] cache_prompt false: a fresh prefill (the held conversation is kept first)
        h->nocache = 0;
        arch_save_active(h);
        reset(h->e);
        h->toks.clear();
        h->next = -1;
        h->snap.clear();
        h->live = 0;
    } else if (h->arch_cap_gib > 0) {   // [dsv4-rel1-arch] the conversation with the longest reusable part
        int mode = 0;
        // [dsv4-rel6-sched] review rel6 H1: a conversation of a request still in service is never reused by another
        // prompt (it would be rewound or extended under that request): it goes to the archive, the prompt starts fresh
        const int ka = h->live ? 0 : reusable(h->toks, h->next, h->snap, !h->snap.empty() && snap_pos(h->e) == (int)h->snap.size(), ids,
                                n, &mode);
        arch_prune(h);   // review rel4 M2: K/V dropped under memory pressure
        auto best = h->arch.end();
        int kb = ka;
        for (auto it = h->arch.begin(); it != h->arch.end(); ++it) {
            if (it->live) continue;
            int m2 = 0;
            const int k2 = reusable(it->toks, it->next, it->snap, it->snap_ok, ids, n, &m2);
            if (k2 > kb) kb = k2, best = it;
        }
        if (best != h->arch.end() && best->pfx) {   // [dsv4-rel6-xprefix] loaded from the prefix entry (it stays, LRU touch)
            std::list<H::Entry> keep;   // review rel6x L3: out of the list while the held one is saved (no copy; bytes stay counted)
            keep.splice(keep.end(), h->arch, best);
            arch_save_active(h);
            H::Entry& pe = keep.front();
            const int lr = seq_load(h->e, pe.blob);
            if (lr == (int)pe.toks.size()) {
                h->toks = pe.toks;
                h->next = -1;
                h->snap.clear();
                h->live = 0;
                h->n_pfx_used++;
                h->last_src = 3;
                h->arch.splice(h->arch.end(), keep);
            } else if (lr == -4) {   // [kb-dsv4-ctxgrow] no VRAM for its rows now: the entry stays, a fresh prefill
                h->arch.splice(h->arch.end(), keep);
                reset(h->e);
                h->toks.clear();
                h->next = -1;
                h->snap.clear();
                h->live = 0;
            } else {   // review rel6x L2: a broken entry leaves the archive
                fprintf(stderr, "[dsv4] archive: a shared prefix of %zu tokens did not load: dropped\n", pe.toks.size());
                h->arch_bytes -= pe.blob.size();
                h->n_dropped++;
                reset(h->e);
                h->toks.clear();
                h->next = -1;
                h->snap.clear();
                h->live = 0;
            }
        } else if (best != h->arch.end()) {
            H::Entry e = std::move(*best);
            h->arch_bytes -= e.blob.size();
            h->arch.erase(best);
            arch_save_active(h, e.rid);
            // [dsv4-rel4-p2] (seq_unpark drops the id when it fails)
            const int lr = e.rid ? seq_unpark(h->e, e.blob, e.rid) : seq_load(h->e, e.blob);
            if (lr == (int)e.toks.size()) {
                h->toks.swap(e.toks);
                h->next = e.next;
                h->snap.swap(e.snap);
                if (!e.snap_ok) h->snap.clear();
                h->live = e.live;
                h->n_loaded++;
                h->last_src = 3;
            } else if (lr == -4) {   // [kb-dsv4-ctxgrow] no VRAM for its rows now: the entry stays (review kb M3)
                h->arch_bytes += e.blob.size();
                h->arch.push_back(std::move(e));
                reset(h->e);
                h->toks.clear();
                h->next = -1;
                h->snap.clear();
            } else {   // not this engine's blob (never expected): a fresh sequence
                fprintf(stderr, "[dsv4] archive: restore failed (%zu tokens; its K/V was dropped under memory pressure): re-prefill\n",
                        e.toks.size());
                reset(h->e);
                h->toks.clear();
                h->next = -1;
                h->snap.clear();
            }
        } else if (ka == 0) {
            arch_save_active(h);   // another conversation: the held one is kept before the reset below
        }
    }
    if (h->live) {   // review rel6 H1: a request in service whose save was skipped: never reused by this prompt
        reset(h->e);
        h->toks.clear();
        h->next = -1;
        h->snap.clear();
        h->live = 0;
    }
    int k = dsv4_common_prefix(h, ids, n);
    if (h->next >= 0 && k == (int)h->toks.size() && k < n) {
        // the held sequence ends with an emitted token not fed yet: it must equal the prompt's next token
        if (ids[k] != h->next) k = 0;
    }
    if (k == n) k = 0;   // nothing new: re-run the last token's step (a regenerate)
    // [dsv4-srv-pesnap] the held sequence does not continue (or less than the snapshot): the last prompt's end state,
    // when this prompt extends it (a next turn without the earlier reasoning, a regenerate, an edited reply)
    const int P = (int)h->snap.size();
    if (k < P && P < n && snap_pos(h->e) == P && std::equal(h->snap.begin(), h->snap.end(), ids)) {
        if (snap_restore(h->e) == P) {
            h->toks = h->snap;
            k = P;
        } else {
            k = 0;
        }
    }
    if (k == 0) {
        reset(h->e);
        h->toks.clear();
    }
    h->next = -1;
    const int k_reused = k;   // (reported: the shared part fed here is not reused)
    if (shared_seen && k < hint) {   // [dsv4-rel6-xprefix] the shared part on its own (its prefill ends there: every layer valid)
        std::vector<int> pre(ids + k, ids + hint);
        (void)prefill(h->e, pre);
        h->toks.insert(h->toks.end(), pre.begin(), pre.end());
        h->next = -1;
        k = hint;
        H::Entry pe;
        try {
            snap_mark(h->e, 0);   // review rel6x L4: no stale prompt-end snapshot in the entry (snap_mark(n - 1) follows)
            const double need = seq_bytes(h->e, false) / (double)(1 << 30);
            auto room = [&] {
                const double av = meminfo_gib("MemAvailable");
                return std::min(h->arch_cap_gib - arch_host(h) / (double)(1 << 30), av < 0 ? 1e9 : av - arch_floor_gib());
            };
            // review rel6x M2: idle conversations (LRU, not in service, not shared prefixes) make room
            for (auto it = h->arch.begin(); it != h->arch.end() && need > room();) {
                if (it->live || it->pfx) { ++it; continue; }
                fprintf(stderr, "[dsv4] archive: dropped a conversation of %zu tokens for a shared prefix\n", it->toks.size());
                h->arch_bytes -= it->blob.size();
                if (it->rid) seq_res_free(h->e, it->rid);
                it = h->arch.erase(it);
                h->n_dropped++;
            }
            if (need <= room() && seq_save(h->e, pe.blob) > 0) {
                pe.toks = h->toks;
                pe.pfx = 1;
                h->arch_bytes += pe.blob.size();
                h->arch.push_back(std::move(pe));
                h->n_pfx_made++;
                arch_pressure(h);
            }
        } catch (const std::bad_alloc&) {   // no host memory: no prefix entry (degrade)
        }
    }
    std::vector<int> rest(ids + k, ids + n);
    static const bool nopesnap = getenv("DSV4_NOPESNAP") && atoi(getenv("DSV4_NOPESNAP")) == 1;   // off-switch
    if (!nopesnap && !part) snap_mark(h->e, n - 1);   // [dsv4-srv-pesnap] this prompt's end state (n - 1 >= k: rest >= 1 token)
    const int t = prefill(h->e, rest, nullptr, nullptr, follow);
    h->toks.insert(h->toks.end(), rest.begin(), rest.end());
    h->next = part ? -1 : t;   // [dsv4-rel6-sched] a prompt part: every token fed, nothing pending (the next part extends it)
    if (snap_pos(h->e) == n - 1) h->snap.assign(ids, ids + n - 1);
    else h->snap.clear();
    if (reuse_out) *reuse_out = k_reused;
    if (k_reused == 0) h->last_src = 0;
    else if (h->last_src != 3) h->last_src = k_reused == P && P > 0 ? 2 : 1;   // (review rel6x L5)
    return part ? 0 : t;
}
// [kb-dsv4-ctxgrow] a prefill whose context K/V cannot grow (CtxNoRoom, thrown before its rows moved): this request
// fails with -2; the engine keeps what it held at the throw (h->toks: a reusable prefix, no pending token)
// [dsv4-xcall] a CtxNoRoom thrown inside the re-prefill that completes a pending prompt part (xcone_flush) leaves the
// engine empty: the hold is gone then (the next start re-prefills)
static void hold_check(H* h) {
    if (engine_pos(h->e) == (int)h->toks.size()) return;
    h->toks.clear();
    h->next = -1;
    h->snap.clear();
}
static int start_safe(H* h, const int* ids, int n, int* reuse_out, bool part) {
    try {
        return start_impl(h, ids, n, reuse_out, part);
    } catch (const CtxNoRoom& ex) {
        // (review kb M3: the throw came before the prefill moved anything: the engine still holds h->toks, which stays
        // a reusable hold; only the prompt-end snapshot mark is dropped)
        fprintf(stderr, "[dsv4] request failed: %s\n", ex.what());
        hold_check(h);
        snap_mark(h->e, 0);
        h->next = -1;
        h->snap.clear();
        h->live = 0;
        h->last_src = 0;
        if (reuse_out) *reuse_out = 0;
        return -2;
    } catch (const PrefillAborted&) {
        // [srv-disconnect] the client left mid-prefill: prefill() stopped at a chunk start, so the engine holds a valid
        // prefix (the reused part + the chunks done; after a chunk that skipped layers 21-39 the state is pending,
        // xpend_until, which the next prefill continues or completes, as after a prompt part): it stays a reusable hold
        // (review disconnect H1: a reset lost the conversation the prompt extended, a resend re-prefilled all of it)
        const std::vector<int>& et = engine_tokens(h->e);
        if ((int)et.size() == engine_pos(h->e)) {
            h->toks = et;
        } else {   // (never expected: no hold)
            reset(h->e);
            h->toks.clear();
        }
        snap_mark(h->e, 0);
        h->next = -1;
        h->snap.clear();
        h->live = 0;
        h->last_src = 0;
        if (reuse_out) *reuse_out = 0;
        return -5;
    }
}
// [srv-disconnect] the cancel flag of the prompt the next dsv4_start / dsv4_prefill_part prefills (nullptr = none): read
// before every prefill chunk; non-zero stops the prefill there and the call returns -5 (the engine holds the prefix done)
void dsv4_set_abort(void* p, const int* flag) { set_prefill_abort(((H*)p)->e, flag); }
int dsv4_start(void* p, const int* ids, int n, int* reuse_out) { return start_safe((H*)p, ids, n, reuse_out, false); }
// [dsv4-rel6-sched] chunked prefill: feed the prompt prefix ids[0, n) (reusing what the engine or the archive holds, as
// dsv4_start) without a pending token and without the prompt-end snapshot; a later dsv4_start / dsv4_prefill_part of a
// longer prompt continues it. 0 = done, -1 = context overflow
int dsv4_prefill_part(void* p, const int* ids, int n, int* reuse_out) { return start_safe((H*)p, ids, n, reuse_out, true); }
// [dsv4-xcall] the next dsv4_prefill_part's prompt goes on for n more tokens, all fed by later dsv4_prefill_part /
// dsv4_start calls (not by a step): its tail may skip layers 21-39 (the cone across chunks, order-only as within one
// call). A wrong count costs a re-prefill of the held tokens, never a wrong state. DSV4_NOXCALL=1 off
void dsv4_prefill_follow(void* p, int n) { ((H*)p)->pf_follow = std::max(0, n); }
// [dsv4-rel6-sched] time-sliced service: make the conversation whose tokens + pending next token are exactly ids[0, n)
// the active one, without any prefill: the next dsv4_step feeds the pending token as if the conversation had never
// left (the held one goes to the archive). 1 = active (held already, or restored from the archive), 0 = not there
// (the caller continues with dsv4_start(ids), a re-prefill of what is not reusable)
int dsv4_resume(void* p, const int* ids, int n) {
    H* h = (H*)p;
    h->wait();
    auto same = [&](const std::vector<int>& t, int nx) {
        return nx >= 0 && (int)t.size() + 1 == n && ids[n - 1] == nx && std::equal(t.begin(), t.end(), ids);
    };
    if (n < 1) return 0;
    if (same(h->toks, h->next)) return 1;
    if (h->arch_cap_gib <= 0) return 0;
    arch_prune(h);
    for (auto it = h->arch.begin(); it != h->arch.end(); ++it) {
        if (!same(it->toks, it->next)) continue;
        H::Entry e = std::move(*it);
        h->arch_bytes -= e.blob.size();
        h->arch.erase(it);
        arch_save_active(h, e.rid);
        const int lr = e.rid ? seq_unpark(h->e, e.blob, e.rid) : seq_load(h->e, e.blob);
        if (lr == (int)e.toks.size()) {
            h->toks.swap(e.toks);
            h->next = e.next;
            h->snap.swap(e.snap);
            if (!e.snap_ok) h->snap.clear();
            h->live = e.live;
            h->n_loaded++;
            h->last_src = 3;
            return 1;
        }
        if (lr == -4) {   // [kb-dsv4-ctxgrow] no VRAM for its rows now: the entry stays, the caller re-prefills (review kb M3)
            h->arch_bytes += e.blob.size();
            h->arch.push_back(std::move(e));
            reset(h->e);
            h->toks.clear();
            h->next = -1;
            h->snap.clear();
            h->live = 0;
            return 0;
        }
        fprintf(stderr, "[dsv4] archive: resume failed (%zu tokens; its K/V was dropped under memory pressure)\n", e.toks.size());
        reset(h->e);
        h->toks.clear();
        h->next = -1;
        h->snap.clear();
        h->live = 0;
        h->n_dropped++;   // (review rel6 L3)
        return 0;
    }
    return 0;
}
// [dsv4-srv-stop] the reply ended before the last tokens the engine took (a stop string): keep the first `keep`
// tokens of the held sequence (prompt + reply, the last one as the next input). Returns 1 when the engine could move
// back exactly, 0 when it could not: then the hold is dropped (the next prompt continues from the snapshot or
// re-prefills)
int dsv4_trim(void* p, int keep) {
    H* h = (H*)p;
    h->wait();
    std::vector<int> seq = h->toks;
    if (h->next >= 0) seq.push_back(h->next);
    if (keep >= (int)seq.size()) return 1;
    if (keep >= 1 && trim(h->e, keep - 1)) {
        h->next = seq[keep - 1];
        seq.resize(keep - 1);
        h->toks.swap(seq);
        if (snap_pos(h->e) != (int)h->snap.size()) h->snap.clear();
        return 1;
    }
    h->toks.clear();   // dsv4_common_prefix of an empty hold is 0: the snapshot or a re-prefill
    h->next = -1;
    return 0;
}
int dsv4_snap_pos(void* p) { return ((H*)p)->snap.empty() ? -1 : (int)((H*)p)->snap.size(); }
// one step: feeds the pending token, writes up to cap new tokens (DSpark: 1 + accepted drafts) to out, returns the
// count (0 = the context is full). [dsv4-dec3-cap]: the pass itself accepts at most cap tokens, so the held sequence
// never runs past what the caller keeps at max_tokens (it broke the next turn's prefix reuse)
int dsv4_step(void* p, int* out, int cap) {
    H* h = (H*)p;
    h->wait();
    if (cap <= 0) return 0;   // (review dec3 L2: nothing fed, toks stays in step with the engine)
    if (++h->steps % 32 == 0) arch_pressure(h);   // review rel1 M2: also during a long reply (before the expert guard)
    if (h->next < 0 || (int)h->toks.size() + 16 > h->o.maxctx) return 0;
    std::vector<int> got;
    try {   // [kb-dsv4-ctxgrow] -2: the context K/V cannot grow (thrown before the pass moved anything: the hold stays)
        if (h->stats) {
            StepTimes tm;
            decode_spec(h->e, h->next, got, &tm, cap);
            h->st_ms += tm.total, h->st_lane += tm.cpu, h->st_route += tm.wait_route, h->st_hostb += tm.host_bytes;
            h->st_laneb += tm.lane_bytes, h->st_pass += tm.npass, h->st_tok += (long)got.size();
        } else {
            decode_spec(h->e, h->next, got, nullptr, cap);
        }
    } catch (const CtxNoRoom& ex) {
        fprintf(stderr, "[dsv4] request failed: %s\n", ex.what());
        hold_check(h);
        return -2;
    }
    h->toks.push_back(h->next);
    int n = 0;
    for (size_t i = 0; i < got.size() && n < cap; i++) out[n++] = got[i];
    for (int i = 0; i + 1 < n; i++) h->toks.push_back(out[i]);   // all but the last are inputs already fed
    h->next = n ? out[n - 1] : -1;
    return n;
}
// [shared-srv] grammar-constrained decode (server/dsv4_engine.py): one plain step that feeds the pending token and
// writes the step's logits [V] to out (returns V; 0 = nothing pending or the context is full); the caller picks the
// next token from them (its mask, its sampling) and sets it with dsv4_set_next
int dsv4_step_logits(void* p, float* out) {
    H* h = (H*)p;
    h->wait();
    if (++h->steps % 32 == 0) arch_pressure(h);
    if (h->next < 0 || (int)h->toks.size() + 16 > h->o.maxctx) return 0;
    try {
        decode_logits_plain(h->e, h->next, out);
    } catch (const CtxNoRoom& ex) {   // [kb-dsv4-ctxgrow] as dsv4_step: -2, nothing fed
        fprintf(stderr, "[dsv4] request failed: %s\n", ex.what());
        hold_check(h);
        return -2;
    }
    h->toks.push_back(h->next);
    h->next = -1;
    return V;
}
// [shared-srv] the pending token when none is: after dsv4_step_logits, or after a dsv4_prefill_part of a prompt
// without its last token (that token is then fed by the next step). 1 = set, 0 = refused (one is pending, bad id)
int dsv4_set_next(void* p, int t) {
    H* h = (H*)p;
    h->wait();
    if (h->next >= 0 || t < 0 || t >= V) return 0;
    h->next = t;
    return 1;
}
int dsv4_pos(void* p) { return (int)((H*)p)->toks.size(); }
// [dsv4-rel9] the held run lies above the base context: into the archive so the loan can go back (exact restore on the
// conversation's next turn). Archive off / a save that did not fit: tried once per held length, nothing logged again
// (review M1). A hold the C API already dropped (failed trim) is released in the engine.
static void park_for_giveback(H* h, const char* why) {
    if (h->toks.empty()) {
        reset(h->e);
        return;
    }
    if (h->arch_cap_gib <= 0 || h->toks.size() == h->park_failed) return;
    const size_t n = h->toks.size();
    arch_save_active(h);
    if (!h->toks.empty()) {
        h->park_failed = n;
        fprintf(stderr, "[dsv4] context: the held conversation (%zu tokens) not parked (%s): the loan stays until it leaves\n", n, why);
        return;
    }
    h->park_failed = 0;
    fprintf(stderr, "[dsv4] context: the held conversation (%zu tokens) to the archive %s (context give-back)\n", n, why);
}
// [dsv4-rel-tierreset]: the end of a request (after its last step): with --tier-reset 1 the dynamic tier goes back to
// its post-load state in the background (off the response; the next call waits for it). Returns the previous reset's ms
double dsv4_end_request(void* p) {
    H* h = (H*)p;
    h->wait();
    arch_pressure(h);   // [dsv4-rel1-arch]
    const double last = h->rt_ms;
    // dsv4_rel_review re-review M1 (long after longer): a held run above the base keeps the loan through the tier reset;
    // parked first, the reset gives it back and refills the seeds (the next turn restores the entry exactly)
    if (h->tier_reset && lz_held_above(h->e, false)) park_for_giveback(h, "before the tier reset");
    if (h->tier_reset) h->rt = std::thread([h] { h->rt_ms = tier_reset(h->e); });
    return last;
}
// [dsv4-rel1-arch] archive state: entries, bytes; the last start's source (0 none, 1 held, 2 snapshot, 3 archive);
// counters saved / loaded / dropped / skipped
int dsv4_archive_info(void* p, double* bytes, int* last_src, long* cnt4) {
    H* h = (H*)p;
    if (bytes) *bytes = (double)arch_host(h);
    if (last_src) *last_src = h->last_src;
    if (cnt4) cnt4[0] = h->n_saved, cnt4[1] = h->n_loaded, cnt4[2] = h->n_dropped, cnt4[3] = h->n_skipped;
    return (int)h->arch.size();
}
// [dsv4-rel4-p2] the compressed K/V pool: parked conversations whose K/V is in VRAM, their MB per card, the engine's host
// copies (MB); counters activated in place / moved on the device / restored from host / evicted to host / dropped
int dsv4_kvpool_info(void* p, double* vram_mb, double* host_mb, long* cnt5) {
    int n = 0;
    seq_res_info(((H*)p)->e, &n, vram_mb, host_mb, cnt5);
    return n;
}
// [dsv4-rel5-p3] host buffers of evicted K/V: pinned MB (in use + cache), cache MB; reused / new pinned / new pageable
void dsv4_kvhost_info(void* p, double* pinned_mb, double* cache_mb, long* cnt3) {
    seq_res_hostinfo(((H*)p)->e, pinned_mb, cache_mb, cnt3);
}
// between requests: give host memory back when MemAvailable is low (the server calls it about once a second)
void dsv4_archive_pressure(void* p) { arch_pressure((H*)p); }
// [dsv4-rel8-giveback] between requests (no request in service): a stale lazy-context loan goes back to the pool with
// the experts restored, off any request's TTFT. 1 = the context buffers are at their base
int dsv4_ctx_idle(void* p) {
    H* h = (H*)p;
    h->wait();
    // rel8 open item: the idle give-back could not run while the long conversation was still the held one; after the hold
    // time it goes to the archive first (exact restore on its next turn), then the loan goes back
    if (!h->live && lz_held_above(h->e, true)) park_for_giveback(h, "idle");
    return ctx_idle(h->e);
}
// the next dsv4_start reuses nothing (a fresh prefill; the held conversation goes to the archive first)
void dsv4_set_nocache(void* p) { ((H*)p)->nocache = 1; }
void dsv4_reset(void* p) {
    H* h = (H*)p;
    h->wait();
    reset(h->e);
    h->toks.clear();
    h->next = -1;
    h->snap.clear();
    h->live = 0;
}
// [dsv4-rel6-sched] the held conversation belongs to request `id` (> 0) while it is in service: no other prompt reuses
// it (review rel6 H1); dsv4_end_live(id) when the request ends, wherever its conversation is (held or archived)
void dsv4_set_live(void* p, int id) { ((H*)p)->live = id; }
void dsv4_end_live(void* p, int id) {
    H* h = (H*)p;
    if (id <= 0) return;
    if (h->live == id) h->live = 0;
    for (auto& e : h->arch)
        if (e.live == id) e.live = 0;
}
// [dsv4-rel6-sched] tokens of ids a dsv4_start would reuse (held conversation, its snapshot or the archive; requests in
// service excluded), without changing anything (review rel6 H2: a chunked prompt starts after them)
int dsv4_reusable(void* p, const int* ids, int n) {
    H* h = (H*)p;
    h->wait();
    if (n < 1) return 0;
    int m = 0, k = h->live ? 0 : reusable(h->toks, h->next, h->snap, !h->snap.empty() && snap_pos(h->e) == (int)h->snap.size(), ids, n, &m);
    if (h->arch_cap_gib > 0)
        for (auto& e : h->arch)
            if (!e.live && (!e.rid || seq_res_alive(h->e, e.rid))) k = std::max(k, reusable(e.toks, e.next, e.snap, e.snap_ok, ids, n, &m));
    return k;
}
// [dsv4-rel6-xprefix] the next dsv4_start / dsv4_prefill_part: its first `n` tokens are a part other prompts may share
// (the system message); counters: prefix entries made / used
void dsv4_set_prefix_hint(void* p, int n) { ((H*)p)->pfx_hint = std::max(0, n); }
void dsv4_prefix_info(void* p, long* made, long* used) {
    if (made) *made = ((H*)p)->n_pfx_made;
    if (used) *used = ((H*)p)->n_pfx_used;
}
// [dsv4-rel6-sched] DSV4_STATS=1: out7 = passes, tokens, pass ms, CPU lane ms, route wait ms, host GB read, lane GB
// (sums since the start); 0 = statistics off
int dsv4_stats(void* p, double* out7) {
    H* h = (H*)p;
    if (!h->stats) return 0;
    const double v[7] = {(double)h->st_pass, (double)h->st_tok, h->st_ms, h->st_lane, h->st_route, h->st_hostb / 1e9, h->st_laneb / 1e9};
    for (int i = 0; i < 7; i++) out7[i] = v[i];
    return 1;
}
}
