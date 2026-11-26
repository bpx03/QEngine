// [ink-srv-capi] Prompt cache of the Inkling C API across slots and in host memory (ink_capi.h: Entry, Source):
//   - a start reuses the longest saved point any source holds whose tokens prefix the prompt (at least one prompt token
//     left): a slot's live state at its held end (its own: in place; another's: ring_save / ring_load, rows copied in
//     VRAM), a slot's prompt-end snapshot, a system-part snapshot or an archived conversation in the host store;
//   - the system part (qe_start_begin's hint) is snapshotted once: the prefill stops there, rows + state go to the store;
//   - an idle slot's conversation goes to the store before its rows are reused (eviction, another prompt on its slot)
//     and in the background while the server idles (qe_archive_more).
// No truncate (conv states): only those exact points serve. LRU over every entry (no priority); budget
// QE_KV_ARCHIVE_GB (default 128) and MemAvailable minus QE_KV_ARCHIVE_MARGIN_GB (default 8). No disk.
#include "ink_capi.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fstream>
#include <string>
#include <unordered_set>

using namespace ink;

namespace ink {

static double env_gb(const char* k, double d) {
    const char* v = getenv(k);
    return v && *v ? atof(v) : d;
}
static size_t mem_available() {
    std::ifstream f("/proc/meminfo");
    std::string k, unit;
    size_t v = 0;
    while (f >> k >> v >> unit)
        if (k == "MemAvailable:") return v * 1024;
    return 0;
}
static int lcp(const std::vector<int>& a, const int* ids, int n) {
    const int m = std::min((int)a.size(), n);
    int i = 0;
    while (i < m && a[i] == ids[i]) i++;
    return i;
}
static bool prefix_of(const std::vector<int>& a, const int* ids, int n) {
    return (int)a.size() <= n && lcp(a, ids, n) == (int)a.size();
}

static void recount(Handle& h) {   // every blob once (row blobs are shared between entries)
    std::unordered_set<const void*> seen;
    size_t b = 0;
    for (const Entry& e : h.store) {
        b += e.ring.size() + e.snap.size() + 4 * (e.toks.size() + e.snap_toks.size());
        for (const RowPart& r : e.rows)
            if (r.blob && seen.insert(r.blob.get()).second) b += r.blob->size();
    }
    h.store_bytes = b;
}
static void release_prov(Handle& h) {   // slots pointing at blobs the store no longer holds let them go
    std::unordered_set<const void*> held;
    for (const Entry& e : h.store)
        for (const RowPart& r : e.rows) held.insert(r.blob.get());
    for (Slot& s : h.s)
        for (const RowPart& r : s.prov)
            if (!held.count(r.blob.get())) {
                s.prov_cut(0);
                break;
            }
}

size_t srv_park_bytes(const Park& pk) {
    return (pk.rows ? pk.rows->size() : 0) + pk.ring.size() + pk.snap.size() + 4 * (pk.toks.size() + pk.snap_toks.size());
}
// [ink-srv-review-fix] M2 / L9: host bytes outside the store's own count: paused slots' and parked replies' copies, and
// the row blobs of a partial idle archive that no store entry holds yet
static size_t host_extra(const Handle& h, bool with_arch);
size_t srv_host_extra(const Handle& h) { return host_extra(h, true); }
// [ink-archfix] host bytes of e that no store entry holds (each blob once): what storing e adds to store_bytes
static size_t unheld_bytes(const Handle& h, const Entry& e) {
    std::unordered_set<const void*> in_store, seen;
    for (const Entry& o : h.store)
        for (const RowPart& r : o.rows) in_store.insert(r.blob.get());
    size_t b = e.ring.size() + e.snap.size() + 4 * (e.toks.size() + e.snap_toks.size());
    for (const RowPart& r : e.rows)
        if (r.blob && !in_store.count(r.blob.get()) && seen.insert(r.blob.get()).second) b += r.blob->size();
    return b;
}
int srv_arch_kept(const Handle& h, int j) {
    if (h.arch_slot != j || h.arch.rows.empty()) return 0;
    const Slot& s = h.s[j];
    const int L = lcp(h.arch.toks, s.toks.data(), (int)s.toks.size());
    int k = 0;
    for (const RowPart& r : h.arch.rows) {
        if (r.r0 != k || r.r0 >= L) break;
        k = std::min(r.r0 + r.n, L);
    }
    return k;
}
static size_t host_extra(const Handle& h, bool with_arch) {
    size_t b = 0;
    for (const Slot& s : h.s)
        if (s.paused) b += srv_park_bytes(s.pk);
    for (const auto& kv : h.parked) b += srv_park_bytes(kv.second);
    if (with_arch && !h.arch.rows.empty()) {
        std::unordered_set<const void*> in_store;
        for (const Entry& e : h.store)
            for (const RowPart& r : e.rows) in_store.insert(r.blob.get());
        for (const RowPart& r : h.arch.rows)
            if (r.blob && !in_store.count(r.blob.get())) b += r.blob->size();
    }
    return b;
}

static size_t arch_margin() { return (size_t)(env_gb("QE_KV_ARCHIVE_MARGIN_GB", 8.0) * 1073741824.0); }
// `adding`: an entry about to be stored, its bytes allocated already (the finished idle archive, a system-part snapshot,
// a parked reply): the bytes the store does not hold count on both sides, as the other host bytes do ([ink-archfix] H1:
// they were counted in `extra` and again through the lower MemAvailable, so storing S bytes needed 2 S of headroom;
// M1: rows shared with the turn it replaces no longer count)
static void drop_partial(Handle& h) {
    h.arch = Entry{};
    h.arch_slot = -1;
}
// keep_partial: the idle archive's own fit (the partial is what it extends); every other caller (a pause / park copy, a
// system-part snapshot, a parked reply) gives up the partial before any stored entry (review2 M1: its rows are still in
// VRAM; a stored entry may be the only copy of a conversation)
// may_drop false: no stored entry is dropped for it (the idle background archive: its slot still holds the rows, while a
// stored entry may be the only copy of another conversation; seen on gpt-oss in its H1 repro)
static bool is_prot(const Handle& h, const Entry& e) {
    return h.protect_on && e.hint == h.protect_hint && e.toks == h.protect_toks;
}
// the store bytes of the protected entry (its blobs once): what dropping every other entry cannot free
static size_t prot_bytes(const Handle& h) {
    if (!h.protect_on) return 0;
    for (const Entry& e : h.store)
        if (is_prot(h, e)) {
            std::unordered_set<const void*> seen;
            size_t b = e.ring.size() + e.snap.size() + 4 * (e.toks.size() + e.snap_toks.size());
            for (const RowPart& r : e.rows)
                if (r.blob && seen.insert(r.blob.get()).second) b += r.blob->size();
            return b;
        }
    return 0;
}
static bool store_fit(Handle& h, size_t extra, const Entry* adding, bool keep_partial = false, bool may_drop = true) {
    const size_t cap = (size_t)(env_gb("QE_KV_ARCHIVE_GB", 128.0) * 1073741824.0);
    const size_t margin = arch_margin();
    const size_t avail = mem_available();
    auto held = [&] { return host_extra(h, adding != &h.arch) + (adding ? unheld_bytes(h, *adding) : 0); };
    size_t other = held();   // (not droppable here)
    const size_t lim = avail ? std::min(cap, h.store_bytes + other + (avail > margin ? avail - margin : 0)) : cap;
    if (!keep_partial && adding != &h.arch && h.arch_slot >= 0 && h.store_bytes + other + extra > lim) {
        drop_partial(h);
        other = held();
    }
    // never fits: drop nothing for it; the protected entry cannot be dropped, so its bytes count as fixed
    // ([ink-archprot] archprot_review H1: before, the other entries were all dropped and then the fit failed)
    if (other + extra + prot_bytes(h) > lim) return false;
    bool dropped = false;
    while (may_drop && !h.store.empty() && h.store_bytes + other + extra > lim) {
        auto lru = h.store.end();   // the oldest entry that is not the one a start in progress restores ([ink-archprot])
        for (auto it = h.store.begin(); it != h.store.end(); ++it)
            if (!is_prot(h, *it) && (lru == h.store.end() || it->use < lru->use))
                lru = it;
        if (lru == h.store.end()) break;
        h.store.erase(lru);
        h.dropped++;
        recount(h);
        other = held();   // (a dropped entry's blobs that `adding` / the partial archive share stay allocated)
        dropped = true;
    }
    if (dropped) release_prov(h);
    return h.store_bytes + other + extra <= lim;
}
bool srv_store_fit(Handle& h, size_t extra) { return store_fit(h, extra, nullptr); }

void srv_protect_begin(Handle& h, const int* ids, int n, int own, bool fallback) {
    srv_protect_end(h);
    const Source b = srv_source(h, -1, true, ids, n);
    int e = b.entry >= 0 ? b.entry : b.rows_entry;
    // review L1: the best source is a slot this start may evict: protect the best store fallback (archprot_review2 M1:
    // only then; never the start's own slot or a busy one, whose rows / state stay)
    const int k = b.state_slot >= 0 ? b.state_slot : b.rows_slot;
    const bool evictable = fallback && k >= 0 && k != own && b.rows_slot != own && h.s[k].idle();
    if (e < 0 && n >= 2 && evictable) {
        const int lim = n - 1;
        int best = 0;
        for (size_t i = 0; i < h.store.size(); i++) {   // (srv_source's store candidates)
            const Entry& o = h.store[i];
            const int L = (int)o.toks.size();
            int k = L >= 1 && L <= lim && !o.ring.empty() && prefix_of(o.toks, ids, n) ? L : 0;
            const int P = (int)o.snap_toks.size();
            if (P >= 1 && P <= lim && P <= L && !o.snap.empty() && prefix_of(o.snap_toks, ids, n) && lcp(o.toks, ids, n) >= P)
                k = std::max(k, P);
            if (k > best) {
                best = k;
                e = (int)i;
            }
        }
    }
    if (e < 0) return;
    h.protect_on = true;
    h.protect_hint = h.store[e].hint;
    h.protect_toks = h.store[e].toks;
}
void srv_protect_end(Handle& h) {
    h.protect_on = false;
    h.protect_toks.clear();
}

bool srv_store_put(Handle& h, Entry&& e);
static bool store_put(Handle& h, Entry&& e, bool may_drop = true);
bool srv_store_put(Handle& h, Entry&& e) { return store_put(h, std::move(e)); }
static bool store_put(Handle& h, Entry&& e, bool may_drop) {
    if (!store_fit(h, 0, &e, false, may_drop)) return false;
    // an archived conversation supersedes the archived ones it extends (an older turn of it); snapshots stay
    for (size_t i = 0; i < h.store.size();) {
        const Entry& o = h.store[i];
        const bool same = o.hint == e.hint && o.toks == e.toks;
        const bool older = !o.hint && !e.hint && prefix_of(o.toks, e.toks.data(), (int)e.toks.size()) &&
                           (o.snap_toks.empty() || o.snap_toks == e.snap_toks) &&
                           !is_prot(h, o);   // ([ink-archprot] kept for the start: a branch at an older turn)
        if (same || older)
            h.store.erase(h.store.begin() + i);
        else
            i++;
    }
    e.use = ++h.tick;
    h.store.push_back(std::move(e));
    recount(h);
    return true;
}

Source srv_source(const Handle& h, int me, bool me_live, const int* ids, int n) {
    Source b;
    const int lim = n - 1;   // at least one prompt token is prefilled (its row gives the first token)
    if (lim < 1) return b;
    std::vector<int> lc(h.s.size(), 0);
    for (size_t k = 0; k < h.s.size(); k++)
        if (h.s[k].cap > 0) lc[k] = std::min(lcp(h.s[k].toks, ids, n), lim);
    auto rank = [&](const Source& x) {   // more rows first; then no copy, a copy inside VRAM, an upload
        return (long long)x.c * 4 + (x.rows_slot == me && me >= 0 ? 3 : x.rows_slot >= 0 ? 2 : 1);
    };
    auto rows_for = [&](Source& x) {
        if (me >= 0 && lc[me] >= x.c) {
            x.rows_slot = me;
            return;
        }
        for (size_t k = 0; k < h.s.size(); k++)
            if (lc[k] >= x.c) {
                x.rows_slot = (int)k;
                return;
            }
        x.rows_slot = -1;
        x.rows_entry = x.entry;
    };
    auto offer = [&](Source x) {
        if (x.c < 1) return;
        rows_for(x);
        if (x.rows_slot < 0 && x.rows_entry < 0) return;
        if (b.c == 0 || rank(x) > rank(b)) b = x;
    };
    for (size_t k = 0; k < h.s.size(); k++) {
        const Slot& s = h.s[k];
        if (s.cap == 0) continue;
        const int held = s.held();
        // its live state at the held end (a slot mid-prefill too: its rows below pf_done are final)
        if (held >= 1 && held <= lim && lc[k] == held && ((int)k != me || me_live)) {
            Source x;
            x.c = held;
            x.state_slot = (int)k;
            offer(x);
        }
        // its prompt-end snapshot, while its rows below that point are intact
        const int P = (int)s.snap_toks.size();
        if (P >= 1 && P <= lim && !s.snap.empty() && held >= P && lc[k] >= P && prefix_of(s.snap_toks, ids, n)) {
            Source x;
            x.c = P;
            x.blob = &s.snap;
            offer(x);
        }
    }
    for (size_t i = 0; i < h.store.size(); i++) {
        const Entry& e = h.store[i];
        const int L = (int)e.toks.size();
        if (L >= 1 && L <= lim && !e.ring.empty() && prefix_of(e.toks, ids, n)) {
            Source x;
            x.c = L;
            x.blob = &e.ring;
            x.entry = (int)i;
            offer(x);
        }
        const int P = (int)e.snap_toks.size();
        if (P >= 1 && P <= lim && P <= L && !e.snap.empty() && prefix_of(e.snap_toks, ids, n) && lcp(e.toks, ids, n) >= P) {
            Source x;
            x.c = P;
            x.blob = &e.snap;
            x.entry = (int)i;
            offer(x);
        }
    }
    return b;
}

void srv_apply(Handle& h, int me, const Source& src, const int* ids) {
    Slot& s = h.s[me];
    h.last_copied = 0;
    h.last_from_store = false;
    if (src.c < 1) {
        s.toks.clear();
        s.prov_cut(0);
        srv_bind(h, me);
        h.e->reset();
        return;
    }
    // the state first (another slot's live state: through the host, the engine bound to that slot for the copy)
    std::vector<unsigned char> tmp;
    const std::vector<unsigned char>* blob = src.blob;
    if (src.state_slot >= 0 && src.state_slot != me) {
        srv_bind(h, src.state_slot);
        h.e->ring_save(tmp);
        blob = &tmp;
    }
    if (src.rows_slot >= 0 && src.rows_slot != me) {
        const Slot& d = h.s[src.rows_slot];
        h.e->copy_rows(d.base, s.base, src.c);
        s.prov = d.prov;
        s.prov_n = d.prov_n;
        h.last_copied = src.c;
        h.slot_copies++;
    } else if (src.rows_slot < 0) {
        const Entry& e = h.store[src.rows_entry];
        for (const RowPart& r : e.rows)
            if (r.r0 < src.c) h.e->rows_load(*r.blob, s.base + r.r0, std::min(r.n, src.c - r.r0));
        s.prov = e.rows;
        s.prov_n = (int)e.toks.size();
        h.last_copied = src.c;
        h.last_from_store = true;
        h.restored++;
        h.store[src.rows_entry].use = ++h.tick;
    }
    if (src.entry >= 0) h.store[src.entry].use = ++h.tick;
    s.toks.assign(ids, ids + src.c);
    s.prov_cut(src.c);
    srv_bind(h, me);
    if (src.state_slot != me) h.e->ring_load(*blob, src.c);   // (own live state: already at c)
}

int srv_archive_slot(Handle& h, int j, long max_rows, bool take_snap) {
    h.e->ring_finalize();   // [ink-srv-gap] (reads the slot's snapshot vector)
    Slot& s = h.s[j];
    if (s.archived || s.cap == 0 || s.toks.empty()) return 1;
    const int n = s.held();
    if (h.arch_slot != j || h.arch.toks != s.toks) {   // (a partial copy of another slot or an older state: start over)
        // [ink-archfix] M2: from the rows the store holds already (shared, not copied) or the rows of this slot's own
        // partial copy below the common prefix (a next turn arrived during it), whichever covers more
        const int kept = srv_arch_kept(h, j), pn = std::min(s.prov_n, n), have = std::max(kept, pn);
        std::vector<RowPart> rows;
        for (const RowPart& r : kept > pn ? h.arch.rows : s.prov)
            if (r.r0 < have) {
                RowPart q = r;
                q.n = std::min(r.n, have - r.r0);
                rows.push_back(q);
            }
        h.arch = Entry{};
        h.arch.toks = s.toks;
        h.arch_slot = j;
        h.arch.rows = std::move(rows);
        // M1: room for the rows still to copy (+ state, snapshot, its token list: review2 L2), not for the shared or kept
        // ones (the token list h.arch.toks, ~4 bytes per token, is left out: allocated already)
        const size_t need = h.row_bytes * (size_t)(n - have) + std::max(h.ring_est, s.snap.size()) + s.snap.size() +
                            4 * s.snap_toks.size();
        if (!store_fit(h, need, nullptr, true, max_rows <= 0)) {   // (idle chunks, max_rows > 0: within free room only)
            drop_partial(h);
            return -1;
        }
    }
    Entry& e = h.arch;
    int done = e.rows.empty() ? 0 : e.rows.back().r0 + e.rows.back().n;
    const int k = max_rows > 0 ? (int)std::min<long>(max_rows, n - done) : n - done;
    if (k > 0) {
        auto blob = std::make_shared<std::vector<unsigned char>>();
        h.e->rows_save(s.base + done, k, *blob);
        e.rows.push_back(RowPart{done, k, std::move(blob)});
        done += k;
    }
    if (done < n) return 0;
    srv_bind(h, j);
    h.e->ring_save(e.ring);
    h.ring_est = std::max(h.ring_est, e.ring.size());
    // its prompt-end snapshot comes along while its rows are part of these rows
    if (!s.snap.empty() && (int)s.snap_toks.size() <= n && std::equal(s.snap_toks.begin(), s.snap_toks.end(), s.toks.begin())) {
        e.snap_toks = s.snap_toks;
        if (take_snap) {
            e.snap = std::move(s.snap);
            s.snap.clear();
            s.snap_toks.clear();
        } else {
            e.snap = s.snap;
        }
    }
    std::vector<RowPart> parts = e.rows;
    const bool ok = store_put(h, std::move(e), max_rows <= 0);
    h.arch = Entry{};
    h.arch_slot = -1;
    if (!ok) return -1;
    s.prov = std::move(parts);
    s.prov_n = n;
    s.archived = true;
    h.archived++;
    return 1;
}

void srv_hint_save(Handle& h, int i) {
    Slot& s = h.s[i];
    const int H = s.pf_hint;
    s.pf_hint = 0;
    if (H < 1 || s.held() != H) return;
    for (const Entry& o : h.store)   // (the same system part is stored already)
        if (o.hint && o.toks == s.toks) return;
    Entry e;
    e.hint = true;
    e.toks = s.toks;
    h.e->ring_save(e.ring);   // (the engine is bound to slot i at H)
    const bool shared = s.prov_n >= H;
    if (shared) {
        for (const RowPart& r : s.prov)
            if (r.r0 < H) e.rows.push_back(RowPart{r.r0, std::min(r.n, H - r.r0), r.blob});
    } else {
        auto blob = std::make_shared<std::vector<unsigned char>>();
        h.e->rows_save(s.base, H, *blob);
        e.rows.push_back(RowPart{0, H, std::move(blob)});
    }
    std::vector<RowPart> parts = e.rows;
    if (!store_put(h, std::move(e))) return;
    if (!shared) {
        s.prov = std::move(parts);
        s.prov_n = H;
    }
    h.hint_saves++;
}

}  // namespace ink

extern "C" {

// cache group: what a start of ids would reuse now; no priority (LRU for all: set_prio is ignored, nothing pinned)
int qe_probe_cached(void* p, const int* ids, int n) try {
    Handle* h = (Handle*)p;
    if (!ids || n < 2) return 0;
    h->e->ring_finalize();
    return srv_source(*h, -1, true, ids, n).c;
} QE_CATCH(-1, 0)
void qe_set_prio(void* p, int slot, int prio) { (void)p; (void)slot; (void)prio; }
int qe_num_pinned(void* p) { (void)p; return 0; }
int qe_last_copied(void* p) { return ((Handle*)p)->last_copied; }
int qe_last_from_archive(void* p) { return ((Handle*)p)->last_from_store ? 1 : 0; }

// archive group: one idle slot's conversation into the store per call, `budget` bytes of rows at most; 1 nothing left,
// 2 not now (host memory short), 0 more to do. No disk archive (persist is a no-op).
int qe_archive_more(void* p, long long budget) try {
    Handle* h = (Handle*)p;
    const double now = std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
    bool held_back = false;   // [ink-archfix] H1: a slot whose archive the store refused waits (Slot::arch_fail_*)
    auto todo = [&](size_t j) {
        const Slot& s = h->s[j];
        if (!(s.idle() && !s.archived && s.cap > 0 && !s.toks.empty())) return false;
        if (s.arch_fail_t >= 0 && s.toks.size() == s.arch_fail_n && now - s.arch_fail_t < 60.0 &&
            mem_available() < s.arch_fail_avail + (size_t(1) << 30)) {
            held_back = true;
            return false;
        }
        return true;
    };
    // review2 L4: a partial copy whose slot lost its rows (fault, trim, eviction without a copy) or whose rows all
    // changed is of no use: given back now
    if (h->arch_slot >= 0) {
        const Slot& a = h->s[h->arch_slot];
        if (a.cap == 0 || a.toks.empty() || srv_arch_kept(*h, h->arch_slot) == 0) drop_partial(*h);
    }
    try {
        const long rows = std::max<long>(256, (long)((double)budget / (double)std::max<size_t>(1, h->row_bytes)));
        int j = h->arch_slot >= 0 && todo(h->arch_slot) ? h->arch_slot : -1;
        for (size_t k = 0; j < 0 && k < h->s.size(); k++)
            if (todo(k)) j = (int)k;
        if (j < 0) return held_back ? 2 : 1;
        const int r = srv_archive_slot(*h, j, rows);
        if (r < 0) {
            Slot& s = h->s[j];
            s.arch_fail_n = s.toks.size();
            s.arch_fail_avail = mem_available();
            s.arch_fail_t = now;
            return 2;
        }
        if (r == 0) return 0;
        h->s[j].arch_fail_t = -1;
        for (size_t k = 0; k < h->s.size(); k++)
            if (todo(k)) return 0;
        return held_back ? 2 : 1;
    } catch (const std::exception& ex) {
        fprintf(stderr, "qe_archive_more: %s\n", ex.what());
        srv_fault(*h, -1, ex.what());
        return 1;
    }
} QE_CATCH(-1, 1)
void qe_persist(void* p) { (void)p; }
int qe_num_archives(void* p) { return (int)((Handle*)p)->store.size(); }
int qe_num_disk_archives(void* p) { (void)p; return 0; }
void qe_mem_guard(void* p) try {
    Handle& h = *(Handle*)p;
    // [ink-archfix] L1 / review2 M1: below the margin the partial idle archive (outside the store; its rows are still in
    // VRAM) goes first, then stored entries as needed
    if (h.arch_slot >= 0) {
        const size_t avail = mem_available();
        if (avail && avail < arch_margin()) drop_partial(h);
    }
    srv_store_fit(h, 0);
} QE_CATCH(-1)

// monitor group: per slot {interval start, size, rows held, busy, last use, prio 1}; the store {entries, bytes, 0..};
// pool events {grow 0, move, evict, 0, 0, 0}
void qe_slot_info(void* p, int slot, long long* b) {
    Handle* h = (Handle*)p;
    for (int i = 0; i < 7; i++) b[i] = 0;
    if (slot < 0 || slot >= (int)h->s.size()) return;
    const Slot& s = h->s[slot];
    b[0] = s.base;
    b[1] = s.cap;
    b[2] = s.held();
    b[3] = !s.idle();
    b[4] = s.use;
    b[5] = 1;
    b[6] = s.paused ? 1 : 0;
}
void qe_archive_info(void* p, double* a) {
    Handle* h = (Handle*)p;
    a[0] = (double)h->store.size();
    a[1] = (double)(h->store_bytes + (h->arch_slot >= 0 ? unheld_bytes(*h, h->arch) : 0));   // ([ink-archfix] review2 L4: + a partial idle archive)
    a[2] = a[3] = a[4] = a[5] = 0;
}
void qe_pool_counters(void* p, long long* c) {
    Handle* h = (Handle*)p;
    c[0] = 0;
    c[1] = h->moves;
    c[2] = h->evictions;
    c[3] = h->pauses;
    c[4] = h->resumes;
    c[5] = 0;
    for (const Slot& s : h->s) c[5] += s.paused;
}

}  // extern "C"
