#pragma once
// Flash-Next archive host memory: the bookkeeping of the pinned slabs (FE's HostPool: pinning, locking and counters; this:
// which ranges are free). First fit over the slabs in the order they were pinned, free ranges coalesced. Compaction under
// host-memory pressure: a plan places every piece of one slab in the free ranges of
// others before anything is copied, a claim carves exactly the planned range. No CUDA: tests/kv_evict_unit.cpp runs it on
// plain memory.
#include <algorithm>
#include <cstddef>
#include <cstring>
#include <map>
#include <utility>
#include <vector>

namespace kve {

struct SlabSet {
    struct Slab { char* base; size_t size; std::map<size_t, size_t> free; };  // offset -> length
    std::vector<Slab> slabs;

    static size_t round(size_t n) { return (n + 4095) / 4096 * 4096; }
    void add(char* base, size_t size) { slabs.push_back(Slab{base, size, {{0, size}}}); }
    // `bytes` (rounded) from the first slab with a free range that holds it; null: none
    char* carve(size_t bytes) {
        for (auto& sl : slabs)
            for (auto it = sl.free.begin(); it != sl.free.end(); ++it)
                if (it->second >= bytes) {
                    const size_t off = it->first, len = it->second;
                    sl.free.erase(it);
                    if (len > bytes) sl.free[off + bytes] = len - bytes;
                    return sl.base + off;
                }
        return nullptr;
    }
    // [p, p + bytes) back to its slab's free ranges, coalesced; false: p is in no slab
    bool put_back(char* p, size_t bytes) {
        for (auto& sl : slabs) {
            if (p < sl.base || p >= sl.base + sl.size) continue;
            size_t off = p - sl.base, len = bytes;
            auto nx = sl.free.lower_bound(off);
            if (nx != sl.free.end() && nx->first == off + len) { len += nx->second; nx = sl.free.erase(nx); }
            if (nx != sl.free.begin()) {
                auto pv = std::prev(nx);
                if (pv->first + pv->second == off) { off = pv->first; len += pv->second; sl.free.erase(pv); }
            }
            sl.free[off] = len;
            return true;
        }
        return false;
    }
    // whole free slabs, last pinned first, until `want` bytes: gone(base, size) for each (it unpins and frees), removed here
    template <class Gone>
    size_t drop_free(size_t want, Gone gone) {
        size_t got = 0;
        for (size_t i = slabs.size(); i-- > 0 && got < want;) {
            Slab& sl = slabs[i];
            if (sl.free.size() != 1 || sl.free.begin()->first != 0 || sl.free.begin()->second != sl.size) continue;
            gone(sl.base, sl.size);
            got += sl.size;
            slabs.erase(slabs.begin() + i);
        }
        return got;
    }
    static size_t used_of(const Slab& sl) {
        size_t fr = 0;
        for (auto& [o, n] : sl.free) fr += n;
        return sl.size - fr;
    }
    const Slab* find(const char* base) const {
        for (auto& sl : slabs)
            if (sl.base == base) return &sl;
        return nullptr;
    }
    // the partly used slabs, least used first (a free one: drop_free)
    std::vector<const char*> by_use() const {
        std::vector<std::pair<size_t, const char*>> v;
        for (auto& sl : slabs)
            if (const size_t u = used_of(sl)) v.push_back({u, sl.base});
        std::sort(v.begin(), v.end());
        std::vector<const char*> out;
        for (auto& x : v) out.push_back(x.second);
        return out;
    }
    // where every piece (rounded sizes) of slab `from` would go: free ranges of the other slabs, the most used slab first (it
    // is the last one compaction would try to empty), the largest piece first; never into `from` or `avoid` (slabs still to be
    // emptied). false: they do not all fit, and dst is empty. Nothing changes here: claim() carves the ranges.
    bool plan(const char* from, const std::vector<size_t>& sizes, const std::vector<const char*>& avoid, std::vector<char*>& dst) const {
        dst.assign(sizes.size(), nullptr);
        std::vector<std::pair<size_t, const Slab*>> to;
        for (auto& sl : slabs)
            if (sl.base != from && std::find(avoid.begin(), avoid.end(), sl.base) == avoid.end()) to.push_back({used_of(sl), &sl});
        std::sort(to.begin(), to.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
        std::vector<std::map<size_t, size_t>> fr;
        for (auto& t : to) fr.push_back(t.second->free);
        std::vector<size_t> ord(sizes.size());
        for (size_t i = 0; i < ord.size(); i++) ord[i] = i;
        std::sort(ord.begin(), ord.end(), [&](size_t a, size_t b) { return sizes[a] > sizes[b]; });
        for (size_t i : ord) {
            bool ok = false;
            for (size_t k = 0; k < to.size() && !ok; k++)
                for (auto it = fr[k].begin(); it != fr[k].end(); ++it)
                    if (it->second >= sizes[i]) {
                        const size_t off = it->first, len = it->second;
                        fr[k].erase(it);
                        if (len > sizes[i]) fr[k][off + sizes[i]] = len - sizes[i];
                        dst[i] = to[k].second->base + off;
                        ok = true;
                        break;
                    }
            if (!ok) { dst.clear(); return false; }
        }
        return true;
    }
    // carve exactly [p, p + bytes) (a planned range); false: it is no longer free (nothing changed)
    bool claim(char* p, size_t bytes) {
        for (auto& sl : slabs) {
            if (p < sl.base || p >= sl.base + sl.size) continue;
            const size_t off = p - sl.base;
            auto it = sl.free.upper_bound(off);
            if (it == sl.free.begin()) return false;
            --it;
            const size_t a = it->first, len = it->second;
            if (off + bytes > a + len) return false;
            sl.free.erase(it);
            if (off > a) sl.free[a] = off - a;
            if (a + len > off + bytes) sl.free[off + bytes] = a + len - off - bytes;
            return true;
        }
        return false;
    }
};

// One compaction pass (FE::compact_archive). pieces: the pinned pieces of the blobs that may
// move (a pointer to the blob's piece pointer and its size); everything else carved in a slab (disk writes, disk reads, prompt
// checkpoints) pins that slab. Per slab, least used first: all its bytes in use must be listed pieces and they must all fit
// (plan) before anything is copied; else the next slab. A slab that took a piece is not emptied in the same pass. At most `cap`
// bytes moved. Pool: slabs_by_use, slab_use -> {used, size}, plan, claim, free(p, n, true), release(want), slab_of. Returns the
// bytes released; `moved`: bytes copied; `held`: slabs skipped because something not listed holds bytes in them (a disk write,
// a disk read, a prompt's checkpoints: they let go later, so waiting can return them).
struct Piece { char** p; size_t n; };
template <class Pool>
size_t compact(Pool& pool, const std::vector<Piece>& pieces, size_t want, size_t cap, size_t& moved, int* held = nullptr) {
    size_t got = 0;
    moved = 0;
    if (held) *held = 0;
    std::vector<const char*> dests;
    const std::vector<const char*> order = pool.slabs_by_use();
    for (size_t oi = 0; oi < order.size() && got < want && moved < cap; oi++) {
        const char* base = order[oi];
        if (std::find(dests.begin(), dests.end(), base) != dests.end()) continue;
        const auto use = pool.slab_use(base);
        const size_t u = use.first, size = use.second;
        if (!u) continue;
        std::vector<const Piece*> in;
        std::vector<size_t> sizes;
        size_t sum = 0;
        for (const Piece& x : pieces)
            if (*x.p >= base && *x.p < base + size) {
                in.push_back(&x);
                sizes.push_back(SlabSet::round(x.n));
                sum += sizes.back();
            }
        if (sum != u) {                      // a holder that cannot move: this slab cannot be emptied now
            if (held) ++*held;
            continue;
        }
        if (moved && moved + sum > cap) break;
        std::vector<char*> dst;
        if (!pool.plan(base, sizes, {}, dst)) continue;
        bool ok = true;
        for (size_t k = 0; k < in.size(); k++) {
            if (!pool.claim(dst[k], in[k]->n)) { ok = false; break; }  // (another thread took the range meanwhile)
            memcpy(dst[k], *in[k]->p, in[k]->n);
            pool.free(*in[k]->p, in[k]->n, true);
            *in[k]->p = dst[k];
            moved += sizes[k];
        }
        for (char* q : dst)
            if (const char* x = pool.slab_of(q); x && std::find(dests.begin(), dests.end(), x) == dests.end()) dests.push_back(x);
        if (ok) got += pool.release(want - got);
    }
    return got;
}

}  // namespace kve
