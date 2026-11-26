// [ink-srv-capi] Server side of the Inkling 2 Mini C API (server/qe_capi.h): the handle shared by ink_capi.cpp (tiers 0,
// chunked, batch, sampling, limit, stops, faults, the slot pool) and ink_capi_cache.cpp (prompt cache across slots, the
// host-RAM archive, monitor). Ported from the gpt-oss C API (src/gptoss/capi_srv.*), with Inkling's rule:
//
//   NO TRUNCATE. Every layer application has recurrent conv states, so a sequence cannot be cut back. A start reuses
//   only a saved point: the slot's whole held sequence (the new prompt extends it), a prompt-end snapshot (ring_save
//   right after a prompt's prefill), a system-part snapshot (the hint) or an archived conversation (rows_save +
//   ring_save in host RAM). At least one prompt token is always prefilled after a restore.
//
// Slots (INK_SLOTS 1..16; default 8 on 2+ cards, 2 on one card (qe_ink_default_slots), halved while the pool is below
// Opts::pool_auto): each owns an interval [base, base + cap) of the
// full-attention pool rows (Engine::bind). A slot's interval keeps SLACK (= MROW) rows beyond its tokens: verify and
// drafter rows write K / V past the position (the engine's lim = cap - SLACK, lower for qe_set_limit). Intervals grow
// into free rows, move (Engine::copy_rows) to a free gap, or idle slots are evicted (least recently used first, their
// conversation archived to host RAM first).
#pragma once

#include <cstdint>
#include <memory>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#include "ink.h"
#include "../pool_order.h"

namespace ink {

constexpr int SLACK = MROW;      // rows an interval keeps past its tokens
constexpr int EOS_ID = 200006;   // <|content_model_end_sampling|>

struct Samp {   // qe_set_sampling (+ qe_set_penalty)
    float temp = 0.f, top_p = 1.f, min_p = 0.f;
    int top_k = 0;
    unsigned long long seed = 0;
    std::mt19937_64 rng{0};
    float pres = 0.f, freq = 0.f;
    std::unordered_map<int, int> cnt;   // reply token counts (+ the seed tokens once each)
    bool one() const { return top_k == 1; }
    bool filtered() const { return !one() && (top_k > 0 || top_p < 1.f || min_p > 0.f); }
    bool pen() const { return temp > 0.f && (pres != 0.f || freq != 0.f); }
    // [ink-samp-dev] the engine settles this filter on the device (drafts on): set by qe_set_sampling when the engine
    // has the device path (Engine::filter_ready, INK_SAMP_DEV != 0) and filter_on_device(top_k, top_p, min_p)
    bool devf = false;
    bool dev_filt() const { return devf && temp > 0.f && !one() && filtered() && !pen(); }
    // the token is picked on the host from row_logits (top_k > 64, penalties; no drafts)
    bool host() const { return pen() || (temp > 0.f && filtered() && !dev_filt()); }
    // the engine's temperature (exact Gumbel-max inside the engine; 0 = greedy). top_k 1 = the argmax.
    float eng_temp() const { return temp > 0.f && !one() && !host() ? temp : 0.f; }
    float pen_val(int id) const {
        const auto it = cnt.find(id);
        return it == cnt.end() || it->second <= 0 ? 0.f : freq * (float)it->second + pres;
    }
};

// [ink-samp-dev] the slot's sampling in the engine: temperature + seed, and the device filter (none when host-sampled)
inline void eng_samp(Engine& e, int slot, const Samp& sp) {
    e.set_sampling(slot, sp.eng_temp(), sp.seed);
    const bool d = sp.dev_filt();
    e.set_filter(slot, d ? sp.top_k : 0, d ? sp.top_p : 1.f, d ? sp.min_p : 0.f);
}

// rows [r0, r0 + n) of a store entry; the blob (Engine::rows_save) is shared by entries and slots holding those rows
struct RowPart {
    int r0 = 0, n = 0;
    std::shared_ptr<const std::vector<unsigned char>> blob;
};

// [ink-srv-park] a decoding sequence's whole state in host memory (pool rows + ring_save at its held end + the
// reply's bookkeeping): a paused slot keeps it (qe_pause_for), a parked reply has it under a handle (qe_park)
struct Park {
    std::vector<int> toks;
    int next = -1, pen_from = -1, orig = -1;
    Samp sp;
    long limit = -1;
    bool watch = false;
    std::shared_ptr<std::vector<unsigned char>> rows;
    std::vector<unsigned char> ring;
    std::vector<int> snap_toks;
    std::vector<unsigned char> snap;
    long long st[5] = {};
    unsigned long long admit = 0;   // [ink-resume] the reply's admission tick (the pause order), kept across a park
};

struct Slot {
    std::vector<int> toks;   // rows held (fed tokens: prompt + reply so far); the engine's position = toks.size()
    int next = -1;           // the token after the last row (the next input)
    int base = 0, cap = 0;   // pool rows [base, base + cap); cap 0 = no rows
    bool gen = false;        // generating (start_finish .. release)
    long use = 0;            // last use (LRU)
    Samp sp;
    int pen_from = -1;
    long limit = -1;         // qe_set_limit: tokens the reply may still emit (-1 none)
    bool watch = false;
    std::string fault;
    // [ink-archfix] H1 / review2 L1: an idle archive of this slot the store refused: not tried again while its rows stay
    // the same (arch_fail_n), MemAvailable has not grown by 1 GiB and less than 60 s passed; arch_fail_t < 0: none
    size_t arch_fail_n = 0, arch_fail_avail = 0;
    double arch_fail_t = -1;
    // prompt-end snapshot: the slot's state (Engine::ring_save) right after its last prompt's prefill, at snap_toks.size()
    std::vector<int> snap_toks;
    std::vector<unsigned char> snap;
    // a prompt in flight: pf_done = prompt rows held, -1 none
    std::vector<int> pf;
    int pf_done = -1, pf_reused = 0, pf_first = -1, pf_hint = 0;
    bool pf_started = false;
    bool archived = false;   // the store holds this slot's sequence as it is now
    std::vector<RowPart> prov;   // rows [0, prov_n) are exactly these store parts (shared by an archive of the slot)
    int prov_n = 0;
    long long st[5] = {};    // qe_draft_stats: steps, drafted (rows - 1), accepted, 0, 0
    long emit = 0;           // tick of the last emitted token (the pause order: least recently emitting first)
    bool paused = false;     // [ink-srv-park] generating, its state in `pk` (no rows in the pool: cap 0)
    long pause_ord = 0;
    unsigned long long admit = 0;   // [ink-resume] tick of the reply's start (qe_start_begin): the pause order
    Park pk;
    void prov_cut(int n) { prov_n = std::min(prov_n, std::max(0, n)); if (prov_n == 0) prov.clear(); }
    int held() const { return (int)toks.size(); }
    bool idle() const { return !gen && pf_done < 0; }
};

// host store: system-part snapshots (hint) and archived conversations; LRU over all, no priority
struct Entry {
    std::vector<int> toks;              // rows [0, len)
    std::vector<RowPart> rows;          // covering [0, len) in order
    std::vector<unsigned char> ring;    // the state at len (ring_save)
    std::vector<int> snap_toks;         // its prompt-end snapshot (archived conversations)
    std::vector<unsigned char> snap;
    bool hint = false;
    long use = 0;
};
// where a start's reused rows come from: c rows; the state at c from a slot's live state (state_slot: its held end),
// a blob (a snapshot or an entry's ring); the rows from a slot (rows_slot) or a store entry (rows_entry)
struct Source {
    int c = 0;
    int state_slot = -1, rows_slot = -1, rows_entry = -1, entry = -1;
    const std::vector<unsigned char>* blob = nullptr;
};

struct Handle {
    Engine* e = nullptr;
    std::vector<Slot> s;
    int seq_max = 0;             // tokens one conversation may hold
    int pool = 0;                // pool rows
    size_t row_bytes = 0;        // host bytes of one pool row (rows_save)
    int cur = -1, last_slot = 0, reused = 0, newest = -1;
    Samp pending;
    bool concurrent = false;
    long tick = 0, faults = 0, full_ends = 0, moves = 0, evictions = 0, compactions = 0, snap_used = 0;
    long batch_steps = 0, batch_rows = 0;
    long pauses = 0, resumes = 0, park_next = 0;
    std::unordered_map<long, Park> parked;   // qe_park handles
    std::vector<float> logits;   // [V] row_logits scratch
    // store
    std::vector<Entry> store;
    size_t store_bytes = 0;
    int last_copied = 0;
    bool last_from_store = false;
    long hint_saves = 0, archived = 0, restored = 0, slot_copies = 0, dropped = 0;
    int arch_slot = -1;          // idle archiving in progress (rows in parts)
    Entry arch;
    size_t ring_est = 0;         // [ink-archfix] review2 L2: the largest state blob of an archive so far (the size estimate)
    // [ink-archprot] while a start makes room (its own slot's archive, evictions), the store entry it will restore from is
    // never dropped for those archives (protect_on: that entry's tokens / hint flag)
    bool protect_on = false, protect_hint = false;
    std::vector<int> protect_toks;
    std::vector<int> protect_next;   // archprot_review M1: qe_protect_ids, the waiting request's prompt for the next qe_pause_for
};

// [ink-resume] pool rules for paused replies (pure; tests/inkling_cache_cpu.cpp checks them without a GPU)
// rows a sequence holding `held` tokens keeps free for its growth next to others (INK_RESUME_ROOM mode: 1 reply =
// min(room, ...) (default), 0 growth = all, 2 bounded = at most pool / 8); never more than its context left
// (seq_max - held) or its remaining max_tokens (limit >= 0) (resume_review H1: the uncapped reply room kept a paused
// reply near its context end out of the pool for good)
inline long pool_reserve(int mode, long room, long pool, int seq_max, int held, long limit) {
    long rem = std::max(0L, (long)seq_max - held);
    if (limit >= 0) rem = std::min(rem, limit);
    return mode == 1 ? std::min(room, rem) : mode == 2 ? std::min(rem, pool / 8) : rem;
}
// rows a paused reply of `held` tokens asks for to resume: next to running sequences (busy > 0) its next step's rows +
// its reserve; alone, one row past its tokens (as qe_unpark's fallback), which always fits: a paused reply holds at most
// seq_max - 1 tokens and seq_max <= pool - SLACK (H1: a paused reply resumes whenever it is alone)
inline long resume_want(long busy, int held, long reserve) {
    return busy == 0 ? (long)held + 1 + SLACK : (long)held + VR + SLACK + reserve;
}
// can a sequence ever hold `need` rows (tokens + SLACK)? false: past its context or the pool, it ends (length) and
// pausing others for it buys nothing (resume_review M1)
inline bool pool_fits_ever(long need, long pool, int seq_max) { return need <= pool && need - SLACK <= seq_max; }
// the reply to pause for a growing reply: among the decoding replies with rows, the growing one included, the most
// recently admitted (src/pool_order.h, as the 27B engine's room_or_pause; no priority); -1 none. `fixed` = rows of the
// sequences that cannot be paused (prompts being prefilled); rows(j) = a candidate's rows. Replies older than k are
// never paused before k in this order, so when fixed + their rows + need > pool no pause of a newer reply makes room
// and the growing reply is the one picked at once (resume_review M2, resume_review2 M-a: no needless copies first)
template <class Admit, class Rows>
int pause_pick(const std::vector<int>& cand, int k, Admit admit, Rows rows, long fixed, long need, long pool) {
    if (cand.empty()) return -1;
    long older = 0;
    for (int j : cand)
        if (j != k && admit(j) < admit(k)) older += rows(j);
    if (fixed + older + need > pool) return k;
    return qe_pool::pause_order(cand, admit).front();
}

// the slot selected on the engine at its held end; lim = cap - SLACK (capped at held + limit when `limited`)
void srv_bind(Handle& h, int i, bool limited = false);
// room for `need` rows (tokens + SLACK) on slot i: grow, move, compact, evict idle slots; false: no room
bool srv_room(Handle& h, int i, int need);
long srv_busy_rows(const Handle& h);
void srv_fault(Handle& h, int i, const char* what);
void srv_full(Handle& h, int i, int need);
int srv_pick_host(Handle& h, Slot& s, float* x);   // x: [V] capped logits (penalised in place)
// cache (ink_capi_cache.cpp)
Source srv_source(const Handle& h, int me, bool me_live, const int* ids, int n);
void srv_apply(Handle& h, int me, const Source& src, const int* ids);
// take_snap: the slot gives its prompt-end snapshot up (moved into the entry, not copied: 176 MB at >= 1K tokens)
int srv_archive_slot(Handle& h, int j, long max_rows = 0, bool take_snap = false);
void srv_hint_save(Handle& h, int i);
// the store's budget (QE_KV_ARCHIVE_GB, MemAvailable - QE_KV_ARCHIVE_MARGIN_GB) counts the paused / parked host copies
// and a partial idle archive too (srv_host_extra); drops store entries (LRU) until `extra` more bytes fit
bool srv_store_fit(Handle& h, size_t extra);
// [ink-archprot] a start of ids[0, n): protect the store entry it would restore from now (srv_source) until srv_protect_end;
// an archive that cannot fit without it is not stored (its conversation goes, as without room)
// own: the start's slot (-1 not chosen yet); fallback: also protect the best store fallback of an evictable slot source
void srv_protect_begin(Handle& h, const int* ids, int n, int own = -1, bool fallback = false);
void srv_protect_end(Handle& h);
size_t srv_host_extra(const Handle& h);
// [ink-archfix] M2 / L2: rows [0, k) of slot j's partial idle archive that still hold its sequence (below the common prefix)
int srv_arch_kept(const Handle& h, int j);
size_t srv_park_bytes(const Park& pk);
bool srv_store_put(Handle& h, Entry&& e);   // false: no room (nothing dropped for it)

// [ink-srv-review-fix] M3: no C++ exception may leave an extern "C" entry point (through ctypes that is std::terminate):
// every qe_* body that can throw is a function-try-block ending in QE_CATCH(slot, return value); the handler logs and
// routes to srv_fault (a sticky CUDA error ends the process with code 75 after the exit hook; else that slot faults)
void srv_caught(void* p, int slot, const char* where);   // (inside a catch handler only: it rethrows to read the message)
#define QE_CATCH(slot, ...)                     \
    catch (...) {                               \
        ::ink::srv_caught(p, (slot), __func__); \
        return __VA_ARGS__;                     \
    }

}  // namespace ink
