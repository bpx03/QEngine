// [srv-gptoss-cache] Prompt cache of the gpt-oss C API across slots and in host memory (capi_srv.h: Entry, Source):
//   - a start reuses the longest prefix any source holds: its own slot, another slot (rows copied inside VRAM, the rings
//     from that slot's live rings or its prompt-end snapshot), a system-part snapshot or an archived conversation in the
//     host store (rows + rings uploaded);
//   - the system part (qe_start_begin's hint: the end of the system / developer messages) is snapshotted once: the
//     prefill stops there, rows and rings go to the store, every later conversation with that system part starts there;
//   - an idle slot's conversation goes to the store before its rows are reused (eviction, another prompt on its slot),
//     and in the background while the server idles (qe_archive_more).
// The sliding layers need the rings at the reuse point: a slot's live rings serve points <= 128 rows behind its end,
// snapshots serve their exact point. LRU over every entry (no priority), budget QE_KV_ARCHIVE_GB / MemAvailable.
#include "capi_srv.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fstream>
#include <string>
#include <unordered_set>

using namespace gptoss;

namespace gptoss {

static double env_gb(const char* k, double d) {
    const char* v = getenv(k);
    return v && *v ? atof(v) : d;
}
static size_t mem_available() {   // bytes (MemAvailable of /proc/meminfo; 0 when unknown)
    std::ifstream f("/proc/meminfo");
    std::string k;
    size_t v = 0;
    std::string unit;
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

// store_bytes = every blob counted once (rings, snapshots, token lists, row blobs shared between entries)
// [review H2: per-entry shares through use_count drifted and wrapped]
static void recount(Handle& h) {
    std::unordered_set<const void*> seen;
    size_t b = 0;
    for (const Entry& e : h.store) {
        b += e.ring.size() + e.snap.size() + 4 * (e.toks.size() + e.snap_toks.size());
        for (const RowPart& r : e.rows)
            if (r.blob && seen.insert(r.blob.get()).second) b += r.blob->size();
    }
    h.store_bytes = b;
}
// slots whose provenance points at a blob the store no longer holds let it go (it would stay resident, uncounted;
// review H4); they only lose the sharing (their next archive copies all its rows)
static void release_prov(Handle& h) {
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

bool srv_store_fit(Handle& h, size_t extra) {
    const size_t cap = (size_t)(env_gb("QE_KV_ARCHIVE_GB", 128.0) * 1073741824.0);
    const size_t margin = (size_t)(env_gb("QE_KV_ARCHIVE_MARGIN_GB", 8.0) * 1073741824.0);
    const size_t avail = mem_available();   // (read once per call: review L7)
    // what the store holds plus what the host can still give
    const size_t lim = avail ? std::min(cap, h.store_bytes + (avail > margin ? avail - margin : 0)) : cap;
    if (extra > lim) return false;   // never fits: drop nothing for it [review M2]
    bool dropped = false;
    while (!h.store.empty() && h.store_bytes + extra > lim) {
        auto lru = std::min_element(h.store.begin(), h.store.end(), [](const Entry& a, const Entry& b) { return a.use < b.use; });
        h.store.erase(lru);
        h.dropped++;
        recount(h);
        dropped = true;
    }
    if (dropped) release_prov(h);
    return h.store_bytes + extra <= lim;
}

// fits first, then replaces what it supersedes: false = not stored (nothing dropped for it) [review M1]
static bool store_put(Handle& h, Entry&& e) {
    if (!srv_store_fit(h, e.bytes())) return false;
    // an archived conversation supersedes the archived ones it extends (an older turn of it); snapshots stay
    for (size_t i = 0; i < h.store.size();) {
        const Entry& o = h.store[i];
        const bool same = o.hint == e.hint && o.toks == e.toks;
        const bool older = !o.hint && !e.hint && prefix_of(o.toks, e.toks.data(), (int)e.toks.size());
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

bool srv_archive_cheap(const Handle& h, int j) {
    const Slot& s = h.s[j];
    if (s.archived || s.cap == 0 || s.toks.empty() || h.arch_slot == j) return true;
    static const long lim = [] {
        const char* v = getenv("GPTOSS_ARCHIVE_SYNC_ROWS");
        // Round 11: no limit by default. A conversation that is evicted without its copy is prefilled again when it
        // returns (100K rows: 25.7 s; 21.5 s with the 8-bit store); its copy costs the evicting start 1.4 s (0.74 s)
        // once and the return then takes 0.8 s (0.5 s) (pool_swap*.log). Before: 16,384 rows, so that a long copy
        // would not hold up the running streams (review M4); GPTOSS_ARCHIVE_SYNC_ROWS=16384 is that rule.
        return v && *v ? atol(v) : (1L << 30);
    }();
    return (long)s.toks.size() - s.prov_n <= lim;
}

Source srv_source(const Handle& h, int me, bool me_live, const int* ids, int n) {
    Source b;
    const int lim = n - 1;   // the last row is recomputed for its logits
    if (lim < 1) return b;
    std::vector<int> lc(h.s.size(), 0);
    for (size_t k = 0; k < h.s.size(); k++)
        if (h.s[k].cap > 0) lc[k] = std::min(lcp(h.s[k].toks, ids, n), lim);
    auto rank = [&](const Source& x) {   // more rows first; then no copy, a copy inside VRAM, an upload
        return (long long)x.c * 4 + (x.rows_slot == me && me >= 0 ? 3 : x.rows_slot >= 0 ? 2 : 1);
    };
    auto rows_for = [&](Source& x) {   // a row source for x.c: me, another slot, the entry that gave the rings
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
        const int held = (int)s.toks.size();
        // its live rings: points <= 128 rows behind its end (a slot mid-prefill too: its rows below pf_done are final)
        if (lc[k] > 0 && s.hi - lc[k] <= 128 && ((int)k != me || me_live)) {
            Source x;
            x.c = lc[k];
            x.hi = s.hi;
            x.ring_slot = (int)k;
            offer(x);
        }
        // its prompt-end snapshot, while its rows below that point are intact
        const int P = (int)s.snap_toks.size();
        if (P >= 1 && P <= lim && !s.snap.empty() && held >= P && lc[k] >= P && s.snap_hi - P <= 128 && prefix_of(s.snap_toks, ids, n)) {
            Source x;
            x.c = P;
            x.hi = s.snap_hi;
            x.ring_blob = &s.snap;
            offer(x);
        }
    }
    for (size_t i = 0; i < h.store.size(); i++) {
        const Entry& e = h.store[i];
        const int L = (int)e.toks.size(), c = std::min(lcp(e.toks, ids, n), lim);
        if (c > 0 && c <= L && e.ring_hi - c <= 128) {
            Source x;
            x.c = c;
            x.hi = e.ring_hi;
            x.ring_blob = &e.ring;
            x.entry = (int)i;
            offer(x);
        }
        const int P = (int)e.snap_toks.size();
        if (P >= 1 && P <= lim && !e.snap.empty() && c >= P && e.snap_hi - P <= 128 && prefix_of(e.snap_toks, ids, n)) {
            Source x;
            x.c = P;
            x.hi = e.snap_hi;
            x.ring_blob = &e.snap;
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
        h.e->reset();
        s.toks.clear();
        s.prov_cut(0);
        s.hi = 0;
        return;
    }
    // the rings first (another slot's live rings: through the host, the engine bound to it for the copy)
    std::vector<unsigned char> tmp;
    const std::vector<unsigned char>* ring = src.ring_blob;
    if (src.ring_slot >= 0 && src.ring_slot != me) {
        srv_bind(h, src.ring_slot);
        h.e->ring_save(tmp);
        ring = &tmp;
    }
    if (src.rows_slot >= 0 && src.rows_slot != me) {
        const Slot& d = h.s[src.rows_slot];
        h.e->copy_rows(d.base, s.base, src.c);
        s.prov = d.prov;   // (its provenance comes along)
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
    srv_bind(h, me);
    if (ring)
        h.e->ring_load(*ring, src.c);
    else
        h.e->truncate(src.c);   // its own live rings (hi - c <= 128)
    if (src.ring_slot != me) s.hi = src.hi;   // (own rings: their end stays)
    s.toks.assign(ids, ids + src.c);
    s.prov_cut(src.c);
}

int srv_archive_slot(Handle& h, int j, long max_rows) {
    Slot& s = h.s[j];
    if (s.archived || s.cap == 0 || s.toks.empty()) return 1;
    const int n = (int)s.toks.size();
    if (h.arch_slot != j || h.arch.toks != s.toks) {   // (a partial copy of another slot or an older state: start over)
        const size_t need = h.e->row_bytes() * (size_t)n + 2 * s.snap.size() + 4 * (size_t)n;
        if (!srv_store_fit(h, need)) return -1;
        h.arch = Entry{};
        h.arch.toks = s.toks;
        h.arch_slot = j;
        for (const RowPart& r : s.prov)   // the rows the store holds already: shared, not copied
            if (r.r0 < s.prov_n) {
                RowPart q = r;
                q.n = std::min(r.n, s.prov_n - r.r0);
                h.arch.rows.push_back(q);
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
    e.snap_toks = s.snap_toks;
    e.snap = s.snap;
    e.ring_hi = s.hi;
    e.snap_hi = s.snap_hi;
    std::vector<RowPart> parts = e.rows;
    const bool ok = store_put(h, std::move(e));
    h.arch = Entry{};
    h.arch_slot = -1;
    if (!ok) return -1;
    s.prov = std::move(parts);   // (the slot's rows are those parts now)
    s.prov_n = n;
    s.archived = true;
    h.archived++;
    return 1;
}

void srv_hint_save(Handle& h, int i) {
    Slot& s = h.s[i];
    const int H = s.pf_hint;
    s.pf_hint = 0;
    if (H < 1 || (int)s.toks.size() != H) return;
    Entry e;
    e.hint = true;
    e.toks = s.toks;
    h.e->ring_save(e.ring);   // (the engine is bound to slot i at H)
    e.ring_hi = s.hi;
    const bool shared = s.prov_n >= H;
    if (shared) {   // (its rows came from the store: shared)
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

}  // namespace gptoss

extern "C" {

// cache group (engine.py "cache"): what a start of ids would reuse now; no priority (LRU for all: set_prio is ignored,
// nothing is pinned)
int qe_probe_cached(void* p, const int* ids, int n) {
    Handle* h = (Handle*)p;
    if (!ids || n < 2) return 0;
    return srv_source(*h, -1, true, ids, n).c;
}
void qe_set_prio(void* p, int slot, int prio) { (void)p; (void)slot; (void)prio; }
int qe_num_pinned(void* p) { (void)p; return 0; }
int qe_last_copied(void* p) { return ((Handle*)p)->last_copied; }
int qe_last_from_archive(void* p) { return ((Handle*)p)->last_from_store ? 1 : 0; }

// archive group: one idle slot's conversation into the store per call (the server calls it while idle); 1 nothing
// left, 2 not now (the host is short of memory), 0 more to do. No disk archive (persist is a no-op).
int qe_archive_more(void* p, long long budget) {
    Handle* h = (Handle*)p;
    auto todo = [&](size_t j) {
        const Slot& s = h->s[j];
        return !s.gen && s.pf_done < 0 && !s.archived && s.cap > 0 && !s.toks.empty();
    };
    try {
        // `budget` bytes per card per call (engine.py): rows of that size over the cards
        const long rows = std::max<long>(256, (long)((double)budget * h->e->plan().ngpu / (double)h->e->row_bytes()));
        int j = h->arch_slot >= 0 && todo(h->arch_slot) ? h->arch_slot : -1;
        for (size_t k = 0; j < 0 && k < h->s.size(); k++)
            if (todo(k)) j = (int)k;
        if (j < 0) return 1;
        const int r = srv_archive_slot(*h, j, rows);
        if (r < 0) return 2;
        if (r == 0) return 0;
        for (size_t k = 0; k < h->s.size(); k++)
            if (todo(k)) return 0;
        return 1;
    } catch (const std::exception& ex) {
        fprintf(stderr, "qe_archive_more: %s\n", ex.what());
        if (((Handle*)p)->e && ((Handle*)p)->e->broken()) srv_fault(*(Handle*)p, -1, ex.what());   // (review rel1 L11: exit 75 now)
        return 1;
    }
}
void qe_persist(void* p) { (void)p; }
int qe_num_archives(void* p) { return (int)((Handle*)p)->store.size(); }
int qe_num_disk_archives(void* p) { (void)p; return 0; }
// host memory pressure: the store gives memory back below MemAvailable - margin
void qe_mem_guard(void* p) { srv_store_fit(*(Handle*)p, 0); }

// monitor group: per slot {interval start, size, rows held, generating, rows held, prio 1, paused 0}; the store
// {entries, bytes, 0 disk, 0, 0 pinned, 0}; pool events {grow 0, move, evict, pause 0, resume 0, paused 0}
void qe_slot_info(void* p, int slot, long long* b) {
    Handle* h = (Handle*)p;
    for (int i = 0; i < 7; i++) b[i] = 0;
    if (slot < 0 || slot >= (int)h->s.size()) return;
    const Slot& s = h->s[slot];
    b[0] = s.base;
    b[1] = s.cap;
    b[2] = (long long)s.toks.size();
    b[3] = s.gen || s.pf_done >= 0;
    b[4] = s.use;   // (last use, qe_capi.h; review L5)
    b[5] = 1;
}
void qe_archive_info(void* p, double* a) {
    Handle* h = (Handle*)p;
    a[0] = (double)h->store.size();
    a[1] = (double)h->store_bytes;
    a[2] = a[3] = a[4] = a[5] = 0;
}
void qe_pool_counters(void* p, long long* c) {
    Handle* h = (Handle*)p;
    c[0] = 0;
    c[1] = h->moves;
    c[2] = h->evictions;
    c[3] = c[4] = c[5] = 0;
}

}  // extern "C"
