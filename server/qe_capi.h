// C API that server/engine.py binds (ctypes). libqengine (Qwen3.8-27B) exports all of it; another
// engine (Qwen3.8-Flash-Next: build/libqflash.so or a model switch inside libqengine) needs TIER 0
// and may add the optional groups one by one: the server checks which symbols exist at load time
// and falls back when a group is incomplete (prints "engine: optional C API not provided ...").
// All functions are called from ONE host thread (the server's scheduler). Token ids are int32.
// "slot" = a sequence slot (KV region + recurrent state); a one-slot engine uses slot 0.
#pragma once
#ifdef __cplusplus
extern "C" {
#endif

// ======================================================= TIER 0: required (one sequence at a time)
// devices: CUDA ordinals "0,1,2"; max_t: requested max tokens per sequence (the engine may cap it,
// see qe_max_context); split: per-GPU weights "176,160,176" or "" (engine default). NULL on error.
void* qe_create(const char* model_dir, const char* devices, int max_t, const char* split);
void qe_destroy(void* e);
int qe_max_context(void* e);  // max prompt + reply tokens of one sequence
// Sampling of the current sequence (temperature <= 0: greedy). The server calls it right AFTER
// qe_start_begin (tier 2) or right BEFORE qe_start3 (tier 0): the parameters in force when a prompt
// starts (or set between start_begin and its first chunk) must apply to its first token and to
// every later step of that slot.
void qe_set_sampling(void* e, float temperature, float top_p, int top_k, float min_p, unsigned long long seed);
// Optional (both engines): presence / frequency penalty of the current sequence, called right after
// qe_set_sampling (which turns it off). Sampled rows only (temperature > 0): logit -= frequency * c +
// presence * (c > 0), c = occurrences of the token in the reply so far (+ the earlier verify rows of the
// step); the counts restart here with the n distinct seed tokens (each counts once; the server's loop
// guard passes the repeated reply). qe_set_no_lookup: no lookup drafts for the slot's reply (until its next start).
void qe_set_penalty(void* e, float presence, float frequency, const int* seed, int n);
void qe_set_no_lookup(void* e, int slot, int off);
// Prefill the prompt ids[0, n) and return the FIRST generated token (< 0: no room). The server
// re-sends the whole conversation every turn: reuse the longest cached prefix (at least the live
// sequence of the slot: previous prompt + the reply generated in it). k: max MTP drafts per step
// (0 = no MTP; the server passes --mtp-k). affinity: 64-bit conversation key (hash of the system
// prompt; a slot that served it before is preferred), hint: position of the end of the system part
// (a good snapshot point); both may be ignored.
int qe_start3(void* e, const int* ids, int n, int k, unsigned long long affinity, int hint);
// One decode step of the current sequence: writes 1..k+1 tokens (out has room for 64) and returns
// how many. The first follows the token returned by start / the previous step. The engine does not
// have to stop at an end token: the server stops calling and releases the slot.
int qe_spec_step(void* e, int* out);
int qe_last_reused(void* e);  // prompt tokens taken from cache by the last start
int qe_last_slot(void* e);    // slot of the last start / start_finish
void qe_release(void* e, int slot);  // generation over; its tokens stay cached for the next turn

// ======================================================= TIER 1: single functions (each optional)
int qe_vocab(void* e);                       // logits width (248320); else config.json vocab_size
int qe_num_slots(void* e);                   // default 1; > 1 requires tiers 2 + 3
long long qe_pool_tokens(void* e);           // KV pool size in tokens (monitoring); default max_context
int qe_can_admit(void* e, int n);            // a new n-token sequence fits now; default: a free slot and n <= max_context
int qe_pause_for(void* e, int n, int prio);  // pause decoding sequences of lower prio (lowest first, never a prefill) until a
                                             // new n-token sequence fits; 1: it fits now (nothing paused if it cannot)
void qe_set_limit(void* e, int slot, int tokens);  // reply ends after `tokens` more tokens: verify only drafts that fit
double qe_warmup(void* e, int modes);        // pre-build graphs (bit0 greedy, bit1 sampling); seconds
void qe_reset(void* e);                      // forget all cached prefixes
void qe_set_lookahead(void* e, int on);      // 0 while another prompt's chunks interleave (no step queued ahead)

// ======================================================= TIER 2 "chunked": prefill in chunks (all or none)
// start_begin returns the slot (< 0: no room); then prefill_more[_slot] until it returns 1, then
// start_finish[_slot] returns the first token (qe_last_reused / qe_last_slot updated). Up to 2
// prompts in flight; the _slot variants address one of them, the plain ones the newest.
// img/img_rows/rope3/delta: image rows (vision), NULL/0 for text.
int qe_start_begin(void* e, const int* ids, int n, int k, unsigned long long affinity, int hint,
                   const void* img, int img_rows, const int* rope3, int delta);
int qe_prefill_more(void* e, int budget);
int qe_start_finish(void* e);
int qe_prefill_more_slot(void* e, int slot, int budget);
int qe_start_finish_slot(void* e, int slot);
int qe_num_prefills(void* e);
int qe_prefill_pos(void* e, int slot);  // prompt rows prefilled so far, -1 if none in flight

// ======================================================= TIER 3 "batch": concurrent sequences (all or none)
void qe_set_concurrent(void* e, int on);  // 1: starts do not release the other slots
void qe_resume(void* e, int slot);        // make slot current for qe_spec_step
// one step of n active slots together: out[i * 64 ..] gets cnt[i] tokens of slots[i] (n <= 32)
void qe_step_batch(void* e, const int* slots, int n, int* out, int* cnt);

// ======================================================= TIER 4 "cache": prompt cache bookkeeping (all or none)
int qe_probe_cached(void* e, const int* ids, int n);  // tokens a start would reuse, no side effects
void qe_set_prio(void* e, int slot, int prio);        // eviction value 0 finished, 1 normal, 2 main agent
int qe_num_pinned(void* e);
int qe_last_copied(void* e);        // prompt tokens of the last start copied from another slot
int qe_last_from_archive(void* e);  // last start restored from the host archive

// ======================================================= TIER 5 "archive": host / disk archive (all or none)
int qe_archive_more(void* e, long long budget);  // archive idle slots, ~budget bytes/GPU; 1 = nothing left
void qe_persist(void* e);
int qe_num_archives(void* e);
int qe_num_disk_archives(void* e);

// ======================================================= TIER 6 "monitor" (all or none; dashboard)
void qe_slot_info(void* e, int i, long long* out);  // {offset, capacity, cached tokens, active, last use, prio}
void qe_archive_info(void* e, double* out);         // {ram n, ram bytes, disk n, disk bytes, pinned bytes, pinned used}
void qe_pool_counters(void* e, long long* out);     // {grow, move, evict}

// ======================================================= TIER 7 "stops": exact stop-sequence cut (all or none)
void qe_watch_stops(void* e, int slot, int on);  // keep next-step drafts (qe_drafts) and the undo of the last step
void qe_set_no_ahead(void* e, int slot, int on); // no step queued ahead of the one in flight
int qe_drafts(void* e, int slot, int* out);      // drafts the next step verifies (<= 64)
int qe_trim(void* e, int slot, int pos);         // end the slot's tokens at pos (within its last step): 1 = done

// ======================================================= TIER 8 "grammar": structured output (all or none + qe_vocab)
unsigned* qe_mask_buffer(void* e, int slot);  // pinned [8][(vocab + 31) / 32] allowed-token bitmasks
int* qe_spec_io(void* e, int slot);           // pinned int[32]: [0] model drafts n, [1..] drafts; [16] n to verify, [17..]
void qe_constrain(void* e, int slot, int on);
int qe_constrained(void* e, int slot);

// ======================================================= TIER 9 "vision" (+ qe_start_begin)
int qe_start_mm(void* e, const int* ids, int n, int k, unsigned long long affinity, int hint,
                const void* img, int img_rows, const int* rope3, int delta);

// ======================================================= TIER 10 "steering": activation steering (each optional)
// Vectors are loaded at qe_create (QE_STEER=<file.gguf>, QE_STEER_DIR=<dir>, QE_STEER_DEFAULT=<name>).
// qe_set_steering: the next start's vector (index in qe_steer_info, -1 off, -2 the server default) and scale
// (multiplies the sidecar scale); sticky; the engine keys its prefix reuse / checkpoints / archive by the selection.
int qe_set_steering(void* e, int vec, float scale);  // 0, or -1 for a bad index / scale
// the same with a per-request mode (-1 the vector's default, 0 add, 1 project) and layer range (lo, hi; -1, -1: the
// mode's default, else inside the vector's file_layers); the key is (vector, mode, layers, scale). 0, or -1 (unchanged)
int qe_set_steering_ex(void* e, int vec, float scale, int mode, int lo, int hi);
// JSON {"default": name|null, "vectors": [{name, mode, layers, scale (the defaults), file_layers, modes: {add|project:
// {layers, scale}}, description, intended, file}]}
const char* qe_steer_info(void* e);
// mean layer-output residual over positions [r0, n) of an unsteered scratch prefill -> out[layers][hidden]
int qe_capture(void* e, const int* ids, int n, int r0, float* out);

// ======================================================= TIER 11 "stats": per-request statistics (optional)
// draft statistics of a slot since its creation: out[5] = {verify steps, MTP drafted, MTP accepted, lookup drafted,
// lookup accepted}; the server prints their difference over a request (server/engine_stats.py, QE_STATS_LOG)
void qe_draft_stats(void* e, int slot, long long* out);

#ifdef __cplusplus
}
#endif
