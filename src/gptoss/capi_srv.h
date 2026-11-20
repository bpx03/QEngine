// [srv-gptoss-capi] Server side of the gpt-oss C API (server/qe_capi.h): the handle shared by gptoss_capi.cpp (tiers 0,
// chunked, batch) and capi_srv.cpp (sampling, the max_tokens limit, the stops group, faults, the slot pool).
//
// Several sequences (slots, GPTOSS_SLOTS, default 8 for a pool >= 32,768 rows, else 4): each slot owns a ring set of the engine and an interval
// [base, base + cap) of the full layers' rows (Engine::bind). The intervals share the engine's rows (max_ctx = the pool):
// a slot grows into the free rows after it, else moves (Engine::copy_rows) to a free gap; idle slots (finished
// conversations, their rows kept for the next turn) are evicted least recently used first when room is needed.
// qe_step_batch steps the greedy and plain-temperature slots together in one engine step (Engine::step_rows, with the
// greedy slots' lookup drafts), the others one after another; prompts prefill in chunks between the steps
// (qe_prefill_more_slot).
#pragma once

#include <cstdint>
#include <memory>
#include <random>
#include <algorithm>
#include <unordered_map>
#include <string>
#include <vector>

#include "gptoss.h"

namespace gptoss {

struct Samp {                // qe_set_sampling: temperature <= 0 = greedy (the engine's argmax)
    float temp = 0.f, top_p = 1.f, min_p = 0.f;
    int top_k = 0;           // 0 = no top-k cut
    unsigned long long seed = 0;
    std::mt19937_64 rng{0};
    // top_k = 1 is the argmax whatever the other filters say (the server's "greedy with a penalty"): a row without
    // candidates that takes the argmax of its (penalised) logits on the cards
    bool one() const { return top_k == 1; }
    bool filtered() const { return !one() && (top_k > 0 || top_p < 1.f || min_p > 0.f); }
    // [gptoss-rel2-penalty] qe_set_penalty: logit -= frequency * c + presence * (c > 0), c = cnt[token] = its
    // occurrences in the reply so far (+ the seed tokens, once each); sampled replies only (qe_capi.h)
    float pres = 0.f, freq = 0.f;
    std::unordered_map<int, int> cnt;
    std::vector<int> seed_ids;   // [gptoss-rel3-pentrim] the seed tokens (distinct): cnt is rebuilt from them after a trim
    bool pen() const { return temp > 0.f && (pres != 0.f || freq != 0.f); }
    float pen_val(int id) const {
        const auto it = cnt.find(id);
        return it == cnt.end() || it->second <= 0 ? 0.f : freq * (float)it->second + pres;
    }
};

// [srv-gptoss-cache] rows [r0, r0 + n) of a store entry; the blob (Engine::rows_save, it may hold more rows than n) is
// shared by the entries and slots whose rows are those exact values
struct RowPart {
    int r0 = 0, n = 0;
    std::shared_ptr<const std::vector<unsigned char>> blob;
};
struct Slot {
    std::vector<int> toks;   // rows held (prompt + reply so far)
    int next = -1;           // the token after the last row (the next input)
    int base = 0, cap = 0;   // the full layers' rows [base, base + cap); cap 0 = no rows
    // [review H3 / H5] the rings' end: the highest position written + 1 (rejected drafts and an earlier, longer sequence
    // write past toks.size()). A ring row of position q is intact while nothing at q + 256 was written: the rings serve
    // a reuse point c when hi - c <= 128. snap_hi: hi when the prompt-end snapshot was saved
    int hi = 0, snap_hi = 0;
    bool gen = false;        // generating (between start_finish and release)
    long use = 0;            // last use (eviction order)
    Samp sp;
    bool pen_synced = false; // the engine's penalty row of this slot holds sp.cnt (Engine::pen_reset)
    int pen_from = -1;       // [gptoss-rel3-pentrim] position of the reply's first token in toks + next (-1: no reply yet)
    long limit = -1;         // qe_set_limit: tokens the reply may still emit (-1 = no limit)
    bool watch = false;      // qe_watch_stops
    std::string fault;       // the last fault's message (read once by qe_slot_fault)
    // prompt-end snapshot (the slot's rings after its last start's prefill; GPTOSS_SNAP=0: off)
    std::vector<int> snap_toks;
    std::vector<unsigned char> snap;
    // a prompt in flight (start_begin .. start_finish): pf_done = rows of pf held, -1 = none
    std::vector<int> pf;
    int pf_done = -1, pf_reused = 0, pf_first = -1;
    bool pf_started = false;  // a chunk ran (qe_set_sampling no longer applies to it)
    // grammar group (qe_constrain): the next token is picked from the allowed ones of mask[0] (the server's matcher
    // fills it); the logits come to the host for that (decode: the sampled head's copy; prefill: the dump)
    bool constrained = false;
    std::vector<uint32_t> mask;   // [8][MASK_WORDS]
    int io[32] = {};              // qe_spec_io: no model drafts (io[0] = 0); drafts to verify (io[16]) are not used
    // prompt-lookup drafts (greedy replies): the latest position after each 6-token context of toks + next
    std::unordered_map<unsigned long long, int> lk;
    int lk_known = 6;             // positions < lk_known are indexed
    int lk_miss = 0, lk_skip = 0; // the gate: drafted steps in a row with nothing accepted; steps left without drafting
    void lk_clear() { lk.clear(); lk_known = 6; lk_miss = lk_skip = 0; }
    // [srv-gptoss-cache] the system-part point of the prompt in flight (qe_start_begin's hint): the prefill stops there
    // once to keep a host snapshot (0 = none); archived = the store holds this slot's sequence as it is now
    int pf_hint = 0;
    bool archived = false;
    // the slot's rows [0, prov_n) are exactly these store parts (loaded from them, saved into them, or copied from a slot
    // whose rows were): an archive of the slot shares them and copies only the rows after them
    std::vector<RowPart> prov;
    int prov_n = 0;
    void prov_cut(int n) { prov_n = std::min(prov_n, std::max(0, n)); if (prov_n == 0) prov.clear(); }
};

// [srv-gptoss-cache] host store (capi_cache.cpp): system-part snapshots (hint: rows + rings at the end of the system
// part, shared by every conversation that starts with it) and archived conversations (an idle slot's sequence before
// its rows are reused: rows, rings at its end, its prompt-end snapshot). LRU over all, no priority; budget
// QE_KV_ARCHIVE_GB (default 128) and MemAvailable minus QE_KV_ARCHIVE_MARGIN_GB (default 8).
struct Entry {
    std::vector<int> toks;              // the rows held [0, len)
    std::vector<RowPart> rows;          // in parts (idle archiving copies a budget per call), in order, covering [0, len)
    std::vector<unsigned char> ring;    // the rings at len
    std::vector<int> snap_toks;         // its prompt-end snapshot (archived conversations)
    std::vector<unsigned char> snap;
    int ring_hi = 0, snap_hi = 0;       // the rings' end of ring / snap (Slot::hi when saved)
    bool hint = false;
    long use = 0;
    size_t bytes() const {   // an upper bound (shared blobs in full); the store counts each blob once (store_bytes)
        size_t b = ring.size() + snap.size() + 4 * (toks.size() + snap_toks.size());
        for (const RowPart& r : rows)
            if (r.blob) b += r.blob->size();
        return b;
    }
};
// where a start's reused rows come from (srv_source): c rows; the rings at c from a slot's live rings (ring_slot,
// held - c <= 128), a blob (ring_blob: a slot's snapshot or a store entry), the rows from a slot (rows_slot, in place
// when it is the start's own) or a store entry (rows_entry)
struct Source {
    int c = 0, hi = 0;   // hi: the rings' end the slot gets
    int ring_slot = -1, rows_slot = -1, rows_entry = -1, entry = -1;
    const std::vector<unsigned char>* ring_blob = nullptr;
};
constexpr int MASK_WORDS = (VOCAB + 31) / 32;

struct Handle {
    Engine* e = nullptr;
    std::vector<Slot> s;
    long long full_ends = 0;   // [gptoss-rel1-full] replies ended for lack of room
    int seq_max = 0;         // rows one conversation may hold (the context served; the engine's max_ctx() is the pool of all slots)
    int cur = -1;            // the slot qe_spec_step steps (qe_resume / the last start_finish)
    int bound = -1;          // the slot the engine is bound to
    int last_slot = 0, reused = 0;
    int newest = -1;         // the newest prompt in flight (qe_prefill_more / qe_start_finish)
    Samp pending;            // qe_set_sampling for the next start
    long lk_steps = 0, lk_rows = 0, lk_tokens = 0;   // lookup steps, rows fed, tokens emitted by them
    long batch_steps = 0, batch_rows = 0;            // joint engine steps of qe_step_batch, rows in them
    long tick = 0, faults = 0, snap_used = 0, cards = 0, fallbacks = 0, moves = 0, evictions = 0, compactions = 0;
    std::vector<float> logits;   // [VOCAB] of a sampled prefill's last row
    bool warned = false;
    bool concurrent = false;     // qe_set_concurrent
    // [srv-gptoss-cache]
    std::vector<Entry> store;
    size_t store_bytes = 0;
    int last_copied = 0;         // rows of the last start copied from another slot or the store
    bool last_from_store = false;
    long hint_saves = 0, archived = 0, restored = 0, slot_copies = 0, dropped = 0;
    int arch_slot = -1;          // idle archiving in progress: that slot's entry so far (rows in parts)
    Entry arch;
};

// the slot's sequence on the engine (its ring set, its rows, its position)
void srv_bind(Handle& h, int slot);
// the next token after a prefill (its result `greedy` = the argmax) or a decode step (its result = the argmax, or the
// cards' Gumbel draw when the reply samples): greedy, or sampled (prefill: from h.logits; decode: from the cards)
int srv_pick(Handle& h, Samp& sp, int greedy, bool prefill);
// before a prefill / decode: a sampled or constrained (masked) prefill copies its logits to the host; sets the
// engine's sampling (a masked decode step: the sampled head with its logits on the host)
void srv_arm(Handle& h, Samp& sp, bool prefill, bool masked);
// [gptoss-rel2-penalty] after srv_arm of a decode step of slot i: its penalty row on the cards (made once per reply) and
// the entry of the token the step feeds; a token the reply emitted is counted with srv_pen_count; srv_pen_logits
// subtracts the slot's penalties from full host logits (the first token, picked after the prefill)
void srv_arm_pen(Handle& h, int i);
void srv_pen_sync(Handle& h, int i);   // the row only (a joint step passes the newest token with its row)
void srv_pen_count(Slot& s, int tok);
void srv_pen_logits(const Slot& s, float* x);
// [gptoss-rel3-pentrim] after the slot's sequence was cut to seq[0, pos) (qe_trim): its counts = the seed tokens (once
// each) + the reply tokens still held, seq[pen_from, pos); the cards' row is made again before the next step
void srv_pen_trim(Slot& s, const std::vector<int>& seq, int pos);
// the token after a decode step whose engine result is engine_tok (masked slots: from the host logits)
int srv_pick_step(Handle& h, Slot& s, int engine_tok);
// after a start's prefill of ids[0, n) on the bound slot: keep the rings (the next turn diverges right after the prompt end)
void srv_snap_save(Handle& h, Slot& s, const int* ids, int n);
// rows of ids[0, n) a start on slot `s` can keep (truncate inside the ring window, or the snapshot); snap: via the snapshot
int srv_reuse(const Handle& h, const Slot& s, const int* ids, int n, bool* snap);
// room for `need` rows on slot i (grow, move, evict idle slots); false: no room
bool srv_room(Handle& h, int i, int need);
// rows held by the slots that are not idle (generating or prefilling)
long srv_busy_rows(const Handle& h);
// the next token of a constrained slot from x[0, NTOK) (the step's full logits; changed in place): greedy = the
// allowed argmax, else drawn from the allowed tokens with the slot's sampling
int srv_pick_masked(Handle& h, Slot& s, float* x);
// an engine error inside a call on slot i: counted, its message kept, the slot's sequence dropped
void srv_fault(Handle& h, int i, const char* what);
// [gptoss-rel1-full] the reply of slot i ends for lack of room for `need` rows (its rows stay; the server ends it with length)
void srv_full(Handle& h, int i, int need);
// [srv-gptoss-cache] (capi_cache.cpp)
// the best source of reused rows for a start of ids[0, n) on slot me (me_live: me's own rings are at its held end;
// false after a reuse through its snapshot); me = -1: any slot (qe_probe_cached)
Source srv_source(const Handle& h, int me, bool me_live, const int* ids, int n);
// makes the bound slot me hold ids[0, src.c) from src (rows and rings); its toks become that prefix
void srv_apply(Handle& h, int me, const Source& src, const int* ids);
// the slot's sequence into the store (once per change: Slot::archived), at most `max_rows` more rows per call (0: all);
// 1 done, 0 more to do (call again), -1 the store has no room
int srv_archive_slot(Handle& h, int j, long max_rows = 0);
// the archive of slot j now is a bounded copy (its partial idle archive, or at most GPTOSS_ARCHIVE_SYNC_ROWS rows past the
// rows the store holds; default 16384): eviction and a start on the slot archive only then, else the conversation goes
bool srv_archive_cheap(const Handle& h, int j);
// after a prefill chunk of slot i that ended at its hint: the system-part snapshot into the store
void srv_hint_save(Handle& h, int i);
// drop entries until the store fits its budget with `extra` more bytes; false: it never can
bool srv_store_fit(Handle& h, size_t extra);

}  // namespace gptoss
