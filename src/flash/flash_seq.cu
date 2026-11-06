// Sequence state of a Flash-Next worker as device ranges (slot switching, host-RAM KV archive, prefix reuse; the
// conversation bookkeeping lives in flash_capi.cpp).
//
// What makes a sequence on one card:
//  - QSA KV (trunk full-attention layers + the MTP layer): owned rows [0, kv_rows(n)) of the fp8 K / V arrays, their
//    per (row, head) scales and the pooled indexer keys of the blocks of those rows (row >> 2). Rows of positions
//    >= n hold leftovers (rejected verify rows, another sequence) and are never read by a sequence of length n.
//  - recurrent state (fixed size): DeltaNet S + conv history per linear layer, the PLE conv history, the raw indexer
//    key rings (IDX_TAIL positions by position mod IDX_TAIL, every QSA layer + MTP). With `pend`: the deferred-commit
//    inputs of the last forward (conv / recurrence / PLE rows) and the step counter whose parity picks the recurrence
//    buffer - a sequence saved after a speculative step resumes with its pending rows exactly as if it never left.
#include <cstring>
#include "../env.h"
#include "flash.h"
#include "../kernels/watchdog.cuh"
#include <map>
#include <cmath>
#include <algorithm>

namespace flash {

constexpr size_t DREC_BYTES = (2 * 128 + 2) * 4;  // a recurrence input row (k, v, beta, decay): DREC floats, flash_kernels.cuh

int Worker::kv_rows(int ntok) const {
    if (ntok <= 0) return 0;
    const int full = ntok / p_.period, rem = ntok % p_.period - p_.own0;
    return full * p_.own + std::max(0, std::min(p_.own, rem));
}

void Worker::kv_pieces(int t0, int t1, std::vector<Piece>& out) { kv_pieces_at(cur_base_, t0, t1, out); }

void Worker::kv_pieces_at(int base, int t0, int t1, std::vector<Piece>& out) {
    const int r0 = kv_rows(t0), r1 = kv_rows(t1);
    if (r1 <= r0) return;
    const size_t rb = (size_t)base * p_.own;  // region start row
    // pooled indexer keys: blocks of 4 rows; a block cut by r0 is copied again (its pooled key may have changed)
    const int b0 = r0 / IDX_R, b1 = (r1 + IDX_R - 1) / IDX_R;
    const size_t row = (size_t)NKV * HD;
    for (const KvBase& k : kvb_) {
        // streaming: the K / V rows are the host rows (copies use cudaMemcpyDefault; writers call kv_inval)
        uint8_t* kc = k.hm ? k.hm->hk : k.kc;
        uint8_t* vc = k.hm ? k.hm->hv : k.vc;
        float* ks = k.hm ? k.hm->hks : k.ks;
        float* vs = k.hm ? k.hm->hvs : k.vs;
        out.push_back({(char*)kc + (rb + r0) * row, (size_t)(r1 - r0) * row});
        out.push_back({(char*)vc + (rb + r0) * row, (size_t)(r1 - r0) * row});
        out.push_back({(char*)(ks + (rb + r0) * NKV), (size_t)(r1 - r0) * NKV * 4});
        out.push_back({(char*)(vs + (rb + r0) * NKV), (size_t)(r1 - r0) * NKV * 4});
        out.push_back({(char*)(k.ikb + (rb / IDX_R + b0) * IDX_HD), (size_t)(b1 - b0) * IDX_HD * 2});
    }
}

void Worker::state_pieces(bool pend, std::vector<Piece>& out) { state_pieces_of(cur_slot_, pend, out); }

// the workspace (alloc_state's per-layer state, which every single-sequence kernel / graph uses) as a slot
Worker::SlotState Worker::ws_state() const {
    SlotState q;
    for (auto* v : {&q.S, &q.conv_hist, &q.conv_pend, &q.rec_pend, &q.ple_hist, &q.ple_pend, &q.ik_tail}) v->assign(LAYERS + 1, nullptr);
    for (int li = 0; li <= LAYERS; li++) {
        const Layer& L = li < LAYERS ? L_[li] : mtp_;
        if (li == LAYERS && !has_mtp_) break;
        q.S[li] = L.S; q.conv_hist[li] = L.conv_hist; q.conv_pend[li] = L.conv_pend; q.rec_pend[li] = L.rec_pend;
        q.ple_hist[li] = L.ple_hist; q.ple_pend[li] = L.ple_pend; q.ik_tail[li] = L.ik_tail;
    }
    q.stepc = stepc_;
    return q;
}

void Worker::state_pieces_of(int s, bool pend, std::vector<Piece>& out) {
    // the bound slot's live state is in the workspace, the others' in their storage
    if (s == cur_slot_ && ws_live_) { pieces_of(ws_state(), pend, out); return; }
    pieces_of(slots_.at(s), pend, out);
}

void Worker::pieces_of(const SlotState& q, bool pend, std::vector<Piece>& out) const {
    const int vl = p_.v1 - p_.v0, kl = p_.k1 - p_.k0;
    const int ch = 2 * kl * DK + vl * DV;
    for (int li = 0; li <= LAYERS; li++) {
        if (q.S[li]) {
            out.push_back({(char*)q.S[li], (size_t)vl * DK * DV * 4});
            out.push_back({(char*)q.conv_hist[li], (size_t)3 * ch * 4});
            if (pend) {
                out.push_back({(char*)q.conv_pend[li], (size_t)MAXP * ch * 4});
                out.push_back({(char*)q.rec_pend[li], (size_t)2 * MAXP * vl * DREC_BYTES});
            }
        }
        if (q.ple_hist[li]) {
            out.push_back({(char*)q.ple_hist[li], (size_t)9 * RW * 4});
            if (pend) out.push_back({(char*)q.ple_pend[li], (size_t)MAXP * RW * 4});
        }
        if (q.ik_tail[li]) out.push_back({(char*)q.ik_tail[li], (size_t)IDX_TAIL * IDX_HD * 4});
    }
    if (pend) out.push_back({(char*)q.stepc, 4});
}

// FLASH_MAX_SEQS slots (default 16; a pipeline stage: Plan::max_seqs when set), each with its own state storage; the
// workspace (alloc_state) holds the bound one
void Worker::alloc_slots() {
    int ns = p_.max_seqs > 0 ? p_.max_seqs : env_int("FLASH_MAX_SEQS", 16);
    ns = std::max(1, std::min(16, ns));
    const int vl = p_.v1 - p_.v0, kl = p_.k1 - p_.k0;
    const int ch = 2 * kl * DK + vl * DV;
    slots_.assign(ns, SlotState{});
    spslot_ = (SampleParams*)dalloc((size_t)ns * sizeof(SampleParams));
    sson_.assign(ns, 0);
    sson_d_ = (int*)dalloc((size_t)ns * 4);
    con_slot_.assign(ns, 0);
    mask_d_ = (unsigned*)dalloc((size_t)MASK_ROWS * MASK_WORDS * 4);
    pen_ = (PenState*)dalloc((size_t)ns * pen_state_bytes(VOCAB));
    nolk_d_ = (int*)dalloc((size_t)ns * 4);
    pseed_d_ = (int*)dalloc((size_t)kPenSeedMax * 4);
    CUDA_CHECK(cudaMemset(nolk_d_, 0, (size_t)ns * 4));
    {
        std::vector<SampleParams> d(ns, SampleParams{1.f, 1.f, kTopK, 0.f, 0, 0});
        for (int s = 0; s < ns; s++) {
            d[s].pen = pen_of(s);
            pen_reset(pen_of(s), VOCAB, nullptr, 0, st_);
        }
        CUDA_CHECK(cudaStreamSynchronize(st_));
        CUDA_CHECK(cudaMemcpy(spslot_, d.data(), d.size() * sizeof(SampleParams), cudaMemcpyHostToDevice));
    }
    for (auto& q : slots_)
        for (auto* v : {&q.S, &q.conv_hist, &q.conv_pend, &q.rec_pend, &q.ple_hist, &q.ple_pend, &q.ik_tail}) v->assign(LAYERS + 1, nullptr);
    for (int s = 0; s < ns; s++) {
        SlotState& q = slots_[s];
        for (int li = 0; li <= LAYERS; li++) {
            Layer& L = li < LAYERS ? L_[li] : mtp_;
            if (li == LAYERS && !has_mtp_) break;
            auto get = [&](float* p0, size_t bytes) -> float* {
                if (!p0) return nullptr;
                float* p = (float*)dalloc(bytes);
                CUDA_CHECK(cudaMemset(p, 0, bytes));
                return p;
            };
            q.S[li] = get(L.S, (size_t)vl * DK * DV * 4);
            q.conv_hist[li] = get(L.conv_hist, (size_t)3 * ch * 4);
            q.conv_pend[li] = get(L.conv_pend, (size_t)MAXP * ch * 4);
            q.rec_pend[li] = get(L.rec_pend, (size_t)2 * MAXP * vl * DREC_BYTES);
            q.ple_hist[li] = get(L.ple_hist, (size_t)9 * RW * 4);
            q.ple_pend[li] = get(L.ple_pend, (size_t)MAXP * RW * 4);
            q.ik_tail[li] = get(L.ik_tail, (size_t)IDX_TAIL * IDX_HD * 4);
        }
    }
    kvb_.clear();
    for (int li = 0; li <= LAYERS; li++) {
        Layer& L = li < LAYERS ? L_[li] : mtp_;
        if (li == LAYERS && !has_mtp_) break;
        if (L.kc) kvb_.push_back({L.kc, L.vc, L.ks, L.vs, L.ikb, kvs_ ? &L.kvp.m : nullptr});
    }
    moe_dn_1w();
    gstash_.assign(ns, GraphSet{});
    gbase_.assign(ns, 0);
    guse_.assign(ns, 0);
    sprev_.assign(ns, 0);
    // per-slot step counters in one array (batched steps), pointer tables, per-slot speculative state; the KV region
    // start of the bound slot for every row (the single-sequence kernels read it: graphs serve every slot)
    stepcs_ = (unsigned*)dalloc((size_t)ns * 4);
    CUDA_CHECK(cudaMemset(stepcs_, 0, (size_t)ns * 4));
    for (int i = 0; i < ns; i++) slots_[i].stepc = stepcs_ + i;
    rbcur_ = (int*)dalloc((size_t)PFM * 4);
    CUDA_CHECK(cudaMemset(rbcur_, 0, (size_t)PFM * 4));
    mm_init(ns);
    ws_live_ = true;  // the workspace (alloc_state's state) holds slot 0 (zero, as its storage)
    {
        std::vector<float*> t((size_t)TB_N * (LAYERS + 1) * ns, nullptr);
        for (int k = 0; k < TB_N; k++)
            for (int li = 0; li <= LAYERS; li++)
                for (int i = 0; i < ns; i++) {
                    const SlotState& q = slots_[i];
                    const std::vector<float*>* v[TB_N] = {&q.S, &q.conv_hist, &q.conv_pend, &q.rec_pend, &q.ple_hist, &q.ple_pend, &q.ik_tail};
                    t[((size_t)k * (LAYERS + 1) + li) * ns + i] = (*v[k])[li];
                }
        tab_ = (float**)dalloc(t.size() * sizeof(float*));
        CUDA_CHECK(cudaMemcpy(tab_, t.data(), t.size() * sizeof(float*), cudaMemcpyHostToDevice));
    }
    kvi_.assign(LAYERS + 1, -1);
    {
        int ki = 0;
        for (int li = 0; li <= LAYERS; li++) {
            Layer& L = li < LAYERS ? L_[li] : mtp_;
            if (li == LAYERS && !has_mtp_) break;
            if (L.kc) kvi_[li] = ki++;
        }
    }
    svt_ = (int*)dalloc((size_t)ns * (MAXM + 1) * 4);
    svp_ = (int*)dalloc((size_t)ns * 2 * 4);
    svh_ = (int*)dalloc((size_t)ns * 2 * 4);
    CUDA_CHECK(cudaMemset(svt_, 0, (size_t)ns * (MAXM + 1) * 4));
    CUDA_CHECK(cudaMemset(svp_, 0, (size_t)ns * 2 * 4));
    CUDA_CHECK(cudaMemset(svh_, 0, (size_t)ns * 2 * 4));
    svpos_.assign(ns, 0);
    kvrb_d_ = (int*)dalloc((size_t)ns * 4);
    bseg_d_ = (int*)dalloc((size_t)2 * MAXM * 4);
    CUDA_CHECK(cudaHostAlloc((void**)&kvrb_h_, (size_t)ns * 4, cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc((void**)&bseg_h_, (size_t)2 * MAXM * 4, cudaHostAllocDefault));
    memset(kvrb_h_, 0, (size_t)ns * 4);
    memset(bseg_h_, 0, (size_t)2 * MAXM * 4);
    for (int** q : {&rpos_, &rbase_, &rslot_, &bvt_, &bpn_, &chpos_, &chbase_, &bacc_}) *q = (int*)dalloc((size_t)MAXM * 4);
    bdr_ = (int*)dalloc((size_t)MAXM * MAXM * 4);
    CUDA_CHECK(cudaMemset(bdr_, 0, (size_t)MAXM * MAXM * 4));
    if (has_mtp_) Rminb_ = (float*)dalloc((size_t)MAXM * RW * 4);
    if (has_mtp_) Rmins_ = (float*)dalloc((size_t)ns * RW * 4);  // prefill_multi: each slot's MTP chain row
    CUDA_CHECK(cudaHostAlloc((void**)&bem_h_, (size_t)MAXM * (MAXM + 2) * 4, cudaHostAllocMapped | cudaHostAllocPortable));
    memset(bem_h_, 0, (size_t)MAXM * (MAXM + 2) * 4);
    CUDA_CHECK(cudaHostGetDevicePointer((void**)&bem_, bem_h_, 0));
    blk_ = env_int("FLASH_BATCH_LOOKUP", 1) && !p_.pp();  // (pipeline stages: lookup drafts not wired yet)
    blk_min_ = env_int("FLASH_BATCH_LK_MIN", blk_min_);
    if (blk_ && ns > 1) {
        hdb_ = (int*)dalloc((size_t)ns * ((size_t)max_ctx_ + 64) * 4);
        hnb_ = (int*)dalloc((size_t)ns * 4);
        CUDA_CHECK(cudaMemset(hnb_, 0, (size_t)ns * 4));
        lkbb_ = (unsigned long long*)dalloc((size_t)MAXM * 8);
        CUDA_CHECK(cudaMemset(lkbb_, 0, (size_t)MAXM * 8));
        blkcnt_ = (int*)dalloc(4);
        CUDA_CHECK(cudaMemset(blkcnt_, 0, 4));
    } else {
        blk_ = false;
    }
    if (ns > 1 && p_.rank == 0)
        fprintf(stderr, "[flash] %d sequence slots: %.0f MB of recurrent state per extra slot and card\n", ns,
                (36.0 * vl * DK * DV * 4 + 36.0 * (3 + MAXP) * ch * 4 + 36.0 * 2 * MAXP * vl * DREC_BYTES + 10.0 * RW * 4 + 17.0 * RW * 4) / 1e6);
}

void Worker::bind_slot(int s, int base) {
    std::vector<int> v(PFM, base * p_.own);  // XROWS (= PFM) rows
    CUDA_CHECK(cudaMemcpyAsync(rbcur_, v.data(), v.size() * 4, cudaMemcpyHostToDevice, st_));
    const int mc[2] = {s, mmdel_h_.empty() ? 0 : mmdel_h_.at(s)};  // multimodal RoPE: the bound slot and its shift
    if (mmcur_) CUDA_CHECK(cudaMemcpyAsync(mmcur_, mc, 8, cudaMemcpyHostToDevice, st_));
    if (steer_csl_d_) {  // the single-sequence paths read the bound slot's steering {vector, mode, lo, hi} / scale
        CUDA_CHECK(cudaMemcpyAsync(steer_csl_d_, &steer_sl_h_.at((size_t)s * 4), 16, cudaMemcpyHostToDevice, st_));
        CUDA_CHECK(cudaMemcpyAsync(steer_csc_d_, &steer_sc_h_.at(s), 4, cudaMemcpyHostToDevice, st_));
    }
    CUDA_CHECK(cudaStreamSynchronize(st_));
    ctx_off_ = (long long)base * p_.period;
}

// workspace <-> slot storage (D2D, ~70 MB per card)
void Worker::ws_copy(int s, bool to_storage) {
    std::vector<Piece> w, q;
    pieces_of(ws_state(), true, w);
    pieces_of(slots_.at(s), true, q);
    CHECK(w.size() == q.size(), "flash: workspace / slot layout");
    for (size_t i = 0; i < w.size(); i++)
        CUDA_CHECK(cudaMemcpyAsync(to_storage ? q[i].p : w[i].p, to_storage ? w[i].p : q[i].p, w[i].n, cudaMemcpyDeviceToDevice, st_));
}
void Worker::ws_save() {
    if (!ws_live_) return;
    CUDA_CHECK(cudaSetDevice(dev_));
    ws_copy(cur_slot_, true);
    ws_live_ = false;
}
void Worker::ws_ensure() {
    if (ws_live_) return;
    CUDA_CHECK(cudaSetDevice(dev_));
    ws_copy(cur_slot_, false);
    ws_live_ = true;
}

// the single-sequence path works on slot s from now on: its state comes into the workspace (the previous slot's goes
// back to its storage), its KV region start into rbcur_. Graphs do not change.
void Worker::use_slot(int s, int base) {
    CHECK(s >= 0 && s < (int)slots_.size(), "flash: slot %d of %zu", s, slots_.size());
    if (s == cur_slot_ && base == cur_base_) return;
    CUDA_CHECK(cudaSetDevice(dev_));
    CUDA_CHECK(cudaStreamSynchronize(st_));
    if (s != cur_slot_) {
        sv_stash();  // the speculative state of the slot left behind -> its stash row (sv_load brings it back)
        ws_save();
        sprev_[cur_slot_] = prev_m_;
        if (spslot_) {  // sampling parameters (with their counter) follow the slot
            CUDA_CHECK(cudaMemcpyAsync(spslot_ + cur_slot_, sp_, sizeof(SampleParams), cudaMemcpyDeviceToDevice, st_));
            CUDA_CHECK(cudaMemcpyAsync(sp_, spslot_ + s, sizeof(SampleParams), cudaMemcpyDeviceToDevice, st_));
            sampling_ = sson_[s];
            CUDA_CHECK(cudaStreamSynchronize(st_));
        }
        cur_slot_ = s;
        con_ = con_slot_.empty() ? false : con_slot_[s] != 0;
        prev_m_ = sprev_[s];
        ws_ensure();
    }
    bind_slot(s, base);
    cur_base_ = base;
}

// every cached graph executable of this card except *keep (decode / MTP / speculative / constrained of the bound
// slot, stashed per-slot sets, batched, prefill) with its LRU tick. Destroyed ones keep their map entries with nullptr
// (= seen: captured again on the next use), so the references and iterators the callers hold stay valid. The
// destroyers sync first: an in-flight graph frees its memory only when it completes.
// Two tiers (review F4): the prefill-side graphs (single-sequence / batched / multi prefill, ~5x the size of a decode
// graph, one prompt length each, and their replay saves a small share of a long step) sort before every decode-side
// graph (decode / MTP / speculative / constrained / batched decode, launched every step): an LRU over the whole set
// would cycle through the decode graphs once their set outgrows the budget.
void Worker::graph_list(const cudaGraphExec_t* keep, std::vector<std::pair<unsigned long long, cudaGraphExec_t*>>& all) {
    unsigned long long tier = 1ull << 62;  // decode side (ticks never get near 2^62)
    auto d = [&](cudaGraphExec_t& x) {
        if (x && &x != keep) {
            auto it = xuse_.find(x);
            all.push_back({tier | (it == xuse_.end() ? 0ull : it->second), &x});
        }
    };
    for (auto& a : graph_) for (auto& b : a) for (auto& c : b) d(c);
    for (auto& a : mgraph_) for (auto& b : a) for (auto& c : b) d(c);
    for (auto& a : sgraph_) for (auto& b : a) for (auto& c : b) for (auto& x : c) d(x);
    for (auto& gs : gstash_) {
        for (auto& a : gs.g) for (auto& b : a) for (auto& c : b) d(c);
        for (auto& a : gs.mg) for (auto& b : a) for (auto& c : b) d(c);
        for (auto& a : gs.sg) for (auto& b : a) for (auto& c : b) for (auto& x : c) d(x);
    }
    for (auto* c : {&cgraph_, &bgraphs_}) for (auto& kv : *c) d(kv.second);
    tier = 0;  // prefill side
    for (auto& gs : gstash_) for (auto* c : {&gs.pg, &gs.pbg}) for (auto& kv : *c) d(kv.second);
    for (auto* c : {&pgraphs_, &pbgraphs_, &pmgraphs_}) for (auto& kv : *c) d(kv.second);
}

void Worker::graph_forget(cudaGraphExec_t x) {
    xuse_.erase(x);
    if (auto it = xbytes_.find(x); it != xbytes_.end()) {
        gbytes_ -= std::min(gbytes_.load(), it->second);
        xbytes_.erase(it);
    }
}

// the prefill-side graphs, then the decode-side ones, least recently launched first (never *keep), until the estimated total plus `need` fits the budget
int Worker::graph_fit(size_t need, const cudaGraphExec_t* keep, bool pf_only) {
    if (!gbudget_ || gbytes_ + need <= gbudget_) return 0;
    CUDA_CHECK(cudaStreamSynchronize(st_));  // an in-flight graph frees its memory only when it completes
    std::vector<std::pair<unsigned long long, cudaGraphExec_t*>> all;
    graph_list(keep, all);
    std::sort(all.begin(), all.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
    if (pf_only) {  // a prefill-side graph may only replace prefill-side ones (tier 0): no room there -> -1, run eagerly
        size_t pf = 0;
        for (auto& [tick, px] : all)
            if (!(tick >> 62))
                if (auto it = xbytes_.find(*px); it != xbytes_.end()) pf += it->second;
        if (gbytes_ - std::min(gbytes_.load(), pf) + need > gbudget_) return -1;
    }
    int n = 0;
    for (auto& [tick, px] : all) {
        if (gbytes_ + need <= gbudget_) break;
        if (pf_only && (tick >> 62)) break;
        cudaGraphExec_t& x = *px;
        graph_forget(x);
        CUDA_CHECK(cudaGraphExecDestroy(x));
        x = nullptr;
        n++;
    }
    graphs_evicted_ += n;
    gfit_ += n;
    return n;
}

int Worker::graph_drop_pf(const cudaGraphExec_t* keep) {
    CUDA_CHECK(cudaStreamSynchronize(st_));
    std::vector<std::pair<unsigned long long, cudaGraphExec_t*>> all;
    graph_list(keep, all);
    int n = 0;
    for (auto& [tick, px] : all)
        if (!(tick >> 62)) {
            graph_forget(*px);
            CUDA_CHECK(cudaGraphExecDestroy(*px));
            *px = nullptr;
            n++;
        }
    graphs_evicted_ += n;
    return n;
}

// every cached graph executable of this card except *keep (see graph_list), or the least recently launched fraction
int Worker::drop_graphs(const cudaGraphExec_t* keep, double frac) {
    CUDA_CHECK(cudaStreamSynchronize(st_));
    std::vector<std::pair<unsigned long long, cudaGraphExec_t*>> all;
    graph_list(keep, all);
    size_t n = all.size();
    xheld_ = n;
    if (frac < 1.0) {  // the least recently launched first (never launched = oldest)
        n = std::min(all.size(), (size_t)std::ceil(all.size() * frac));
        std::nth_element(all.begin(), all.begin() + (n ? n - 1 : 0), all.end(),
                         [](const auto& a, const auto& b) { return a.first < b.first; });
    }
    for (size_t i = 0; i < n; i++) {
        cudaGraphExec_t& x = *all[i].second;
        graph_forget(x);
        CUDA_CHECK(cudaGraphExecDestroy(x));
        x = nullptr;
    }
    graphs_evicted_ += n;
    return (int)n;
}

long long oom_reserve_mb() {
    static const long long mb = std::max(0LL, env_ll("FLASH_OOM_RESERVE_MB", 64));
    return mb;
}

// retake: after the reserve was used, only with real headroom (free >= 2 x reserve + 64 MB) and only after
// kRetakeAfter good instantiates / launches since: taken back right away, the next instantiate that needs more failed
// again and every cycle dropped graphs (review 2026-11-01 #4)
void Worker::oom_reserve_take(bool retake) {
    const long long mb = oom_reserve_mb();
    if (oom_res_ || mb <= 0) return;
    const size_t b = (size_t)mb << 20;
    if (retake && oom_ok_ < kRetakeAfter) return;
    if (free_bytes() < (retake ? 2 * b + ((size_t)64 << 20) : b + ((size_t)32 << 20))) {  // clearly free memory only
        if (retake) oom_ok_ = 0;  // next try after another kRetakeAfter good launches, not on every launch (review comm F2)
        return;
    }
    // stream-ordered (the device's default pool): freeing it later needs no device-wide sync (cudaFree syncs every
    // stream of the card, including side-stream kernels that wait for other ranks)
    oom_res_async_ = cudaMallocAsync(&oom_res_, b, st_) == cudaSuccess;
    if (oom_res_async_) {
        if (cudaStreamSynchronize(st_) != cudaSuccess) {
            (void)cudaGetLastError();
            oom_res_ = nullptr;
            return;
        }
    } else {
        (void)cudaGetLastError();
        oom_res_ = nullptr;
        if (cudaMalloc(&oom_res_, b) != cudaSuccess) {
            (void)cudaGetLastError();
            oom_res_ = nullptr;
            return;
        }
    }
    oom_res_bytes_ = b;
}

bool Worker::oom_reserve_free() {
    if (!oom_res_) return false;
    if (oom_res_async_) {
        CUDA_CHECK(cudaFreeAsync(oom_res_, st_));
        CUDA_CHECK(cudaStreamSynchronize(st_));  // the pool releases its free memory to the driver at a sync point ...
        cudaMemPool_t pool;
        if (cudaDeviceGetDefaultMemPool(&pool, dev_) == cudaSuccess) (void)cudaMemPoolTrimTo(pool, 0);  // ... and now
        (void)cudaGetLastError();
    } else {
        CUDA_CHECK(cudaStreamSynchronize(st_));
        CUDA_CHECK(cudaFree(oom_res_));
    }
    oom_res_ = nullptr;
    oom_ok_ = 0;
    return true;
}

// nodes of a graph including the bodies of child-graph nodes: {all, kernel nodes}
static void graph_nodes(cudaGraph_t g, size_t& all, size_t& kern) {
    size_t n = 0;
    if (cudaGraphGetNodes(g, nullptr, &n) != cudaSuccess || !n) { (void)cudaGetLastError(); return; }
    std::vector<cudaGraphNode_t> v(n);
    if (cudaGraphGetNodes(g, v.data(), &n) != cudaSuccess) { (void)cudaGetLastError(); return; }
    for (cudaGraphNode_t x : v) {
        all++;
        cudaGraphNodeType t;
        if (cudaGraphNodeGetType(x, &t) != cudaSuccess) { (void)cudaGetLastError(); continue; }
        if (t == cudaGraphNodeTypeKernel) kern++;
        else if (t == cudaGraphNodeTypeGraph) {
            cudaGraph_t c = nullptr;
            if (cudaGraphChildGraphNodeGetGraph(x, &c) == cudaSuccess && c) graph_nodes(c, all, kern);
            else (void)cudaGetLastError();
        }  // (a conditional node's body is not reachable through the runtime API: not counted)
    }
}

// graph instantiation with little free memory (graph_oom.h, club-3090#1846: 4 slots, the expert cache sized to the
// 256 MB margin, many conversation shapes): on OOM drop every other cached graph and retry once; if that fails too
// the caller runs the step eagerly and the next graph_oom::kCoolSteps missing graphs are not captured. Never aborts.
bool Worker::inst_graph(cudaGraphExec_t* e, cudaGraph_t g, const char* kind) {
    static const bool glog = env_int("FLASH_GRAPH_LOG", 0) != 0;
    size_t nodes = gextra_nodes_, knodes = 0;  // + conditional bodies captured into this graph (lk_chain, review F3)
    gextra_nodes_ = 0;
    graph_nodes(g, nodes, knodes);
    const size_t need = graph_oom::est_bytes(nodes, knodes);
    *e = nullptr;
    // larger than the whole budget (review F2): evicting would wipe every other graph and still not fit, and the next
    // instantiate would wipe this one again. Nothing is evicted; this step runs without a graph (the caller runs the
    // captured work eagerly; run_cached does not capture that key again)
    gbig_last_ = gbudget_ && need > gbudget_;
    if (gbig_last_) {
        if (gbig_++ == 0)
            fprintf(stderr, "[flash] GPU%d: a %s graph (%zu nodes, est %.0f MB) is larger than the graph budget (%.0f MB, "
                    "FLASH_GRAPH_MB%s): such steps run without a CUDA graph\n", dev_, kind, nodes, need / 1048576.0,
                    gbudget_ / 1048576.0, card_workers_ > 1 ? " split between the workers of the card" : "");
        return false;
    }
    const bool pf_side = !strcmp(kind, "prefill") || !strcmp(kind, "pf-batch") || !strcmp(kind, "pf-multi");
    const bool pf_only = pf_side && pf_window() > 0;
    pf_noroom_last_ = false;
    const int nfit = graph_fit(need, e, pf_only);  // the budget first: the oldest graphs make room
    if (nfit < 0) {  // a prompt-shaped graph would have to evict decode graphs: this step runs without a graph
        pf_noroom_++;
        pf_noroom_last_ = true;
        return false;
    }
    const double gf0 = glog ? free_bytes() / 1048576.0 : 0.0;
    static const graph_oom::Inject inj = [] {
        graph_oom::Inject x;
        x.at = graph_oom::parse_list(env_str("FLASH_TEST_GRAPH_OOM"));
        x.hard = env_int("FLASH_TEST_GRAPH_OOM_HARD", 0) != 0;
        x.rank = env_int("FLASH_TEST_GRAPH_OOM_RANK", -1);
        return x;
    }();
    const int call = ++ginst_calls_;
    *e = nullptr;
    // FLASH_GRAPH_PRIO ([fp-gprio]): cudaGraphInstantiateFlagUseNodePriority, i.e. replays keep the stream priorities
    // captured into the kernel nodes (the high-priority prefill collectives on cst_, FLASH_CST_PRIO); without it every
    // node runs at the launching stream's priority. 0 (default): no graph gets it (= master). 1: only pf-batch graphs
    // (prefill_batch, single-prompt batches) of >= FLASH_GRAPH_PRIO_MIN rows (default 4096); prefill() (pgraphs_),
    // prefill_mtp and prefill_multi never get it. 2: every graph, decode included. Off by default: flash_run 3 cards 90K
    // -7.5 % with mode 1, but in the server the decode stayed ~20 % slower for the rest of the process after long-prompt
    // batches ran with it (3 cards agent c=8 TPOT 8.6 -> 10.4 ms; mode 2 likewise), cause unknown (queued after 1.3.5)
    static const int gprio = env_int("FLASH_GRAPH_PRIO", 0);
    const bool pf_big = !strcmp(kind, "pf-batch") && gprio_rows_ >= env_int("FLASH_GRAPH_PRIO_MIN", 4096);
    const unsigned long long gfl = gprio == 2 || (gprio == 1 && pf_big) ? cudaGraphInstantiateFlagUseNodePriority : 0;
    cudaError_t r;
    {
        QeWdSlow slow;  // (an instantiate can take a while: the peers wait for this rank's launch)
        r = inj.first(call, p_.rank) ? cudaErrorMemoryAllocation : cudaGraphInstantiate(e, g, gfl);
    }
    if (r == cudaErrorMemoryAllocation) {
        (void)cudaGetLastError();
        *e = nullptr;
        graph_oom_++;
        const double f0 = free_bytes() / 1048576.0;
        oom_ok_ = 0;
        if (pf_only) {  // a prompt-shaped graph: only prefill-side graphs make room, once; the reserve and the decode
                        // graphs stay for decode-side instantiates
            const int n = graph_drop_pf(e);
            fprintf(stderr, "[flash] GPU%d: %s graph instantiate out of memory (%.0f MB free): dropped %d prefill-side graphs, "
                    "retrying\n", dev_, kind, f0, n);
            r = inj.retry(call, p_.rank) ? cudaErrorMemoryAllocation : cudaGraphInstantiate(e, g, gfl);
            if (r != cudaErrorMemoryAllocation) goto done;
            (void)cudaGetLastError();
            *e = nullptr;
            pf_noroom_++;
            pf_noroom_last_ = true;
            return false;
        }
        // the emergency reserve first (it exists for this; taken back only with real headroom), then the least
        // recently launched half of the graphs (issue #3 D4), then every other graph: their memory goes back to the
        // driver's pool (not shown as free), the next instantiate gets it
        if (oom_reserve_free()) {
            fprintf(stderr, "[flash] GPU%d: graph instantiate out of memory (%.0f MB free): freed the %.0f MB emergency "
                    "reserve, retrying\n", dev_, f0, oom_res_bytes_ / 1048576.0);
            r = inj.retry(call, p_.rank) ? cudaErrorMemoryAllocation : cudaGraphInstantiate(e, g, gfl);
            if (r != cudaErrorMemoryAllocation) goto done;
            (void)cudaGetLastError();
            *e = nullptr;
        }
        {
        static const double lru = env_double("FLASH_GRAPH_LRU_FRAC", 0.5);
        int n = lru > 0 && lru < 1 ? drop_graphs(e, lru) : drop_all_graphs(e);
        const size_t held = xheld_;
        fprintf(stderr, "[flash] GPU%d: graph instantiate out of memory (%.0f MB free): dropped %d of %zu cached graphs%s, "
                "retrying%s\n", dev_, f0, n, held, n < (int)held ? " (least recently used)" : "",
                inj.first(call, p_.rank) ? " (FLASH_TEST_GRAPH_OOM)" : "");
        r = inj.retry(call, p_.rank) ? cudaErrorMemoryAllocation : cudaGraphInstantiate(e, g, gfl);
        if (r == cudaErrorMemoryAllocation && n < (int)held) {
            (void)cudaGetLastError();
            *e = nullptr;
            const int n2 = drop_all_graphs(e);
            fprintf(stderr, "[flash] GPU%d: still out of memory: dropped the other %d cached graphs, retrying\n", dev_, n2);
            r = inj.retry(call, p_.rank) ? cudaErrorMemoryAllocation : cudaGraphInstantiate(e, g, gfl);
        }
        }
        // after the graphs (cheap, recoverable): resident expert groups to host memory (VRAM taken by another process),
        // before this step runs without a graph
        if (r == cudaErrorMemoryAllocation) {
            (void)cudaGetLastError();
            *e = nullptr;
            if (vram_reclaim((size_t)128 << 20, "graph instantiate out of memory")) {
                r = inj.retry(call, p_.rank) ? cudaErrorMemoryAllocation : cudaGraphInstantiate(e, g, 0);
                if (r != cudaErrorMemoryAllocation) goto done;
            }
        }
        if (r == cudaErrorMemoryAllocation) {
            (void)cudaGetLastError();
            *e = nullptr;
            graph_eager_++;
            gcool_.trip();
            fprintf(stderr, "[flash] GPU%d: graph instantiate still out of memory (%.0f MB free): this step and the next %d new "
                    "shapes run without graphs\n", dev_, free_bytes() / 1048576.0, graph_oom::kCoolSteps);
            fflush(stderr);
            return false;
        }
    }
done:
    CUDA_CHECK(r);
    xuse_[*e] = ++xtick_;  // the newest graph is not the first LRU victim before its first launch
    xbytes_[*e] = need;
    gbytes_ += need;
    gpeak_ = std::max(gpeak_.load(), gbytes_.load());
    if (glog) {
        const double gf1 = free_bytes() / 1048576.0;
        fprintf(stderr, "[graph] GPU%d %s: %zu nodes (%zu kernels), est %.1f MB, free %.1f -> %.1f MB (%+.1f), %zu live, "
                "est total %.1f / %.0f MB%s\n", dev_, kind, nodes, knodes, need / 1048576.0, gf0, gf1, gf1 - gf0, xuse_.size(),
                gbytes_.load() / 1048576.0, gbudget_ / 1048576.0, nfit ? (" (" + std::to_string(nfit) + " LRU graphs destroyed first)").c_str() : "");
    }
    if (!oom_res_ && oom_res_bytes_ && ++oom_ok_ >= kRetakeAfter) oom_reserve_take(true);
    return true;
}

void Worker::reset_slot(int s) {
    CUDA_CHECK(cudaSetDevice(dev_));
    std::vector<Piece> ps;
    pieces_of(slots_.at(s), true, ps);
    if (s == cur_slot_ && ws_live_) pieces_of(ws_state(), true, ps);
    for (auto& q : ps) CUDA_CHECK(cudaMemsetAsync(q.p, 0, q.n, st_));
    if (s == cur_slot_) prev_m_ = 0;
    else sprev_[s] = 0;
    CUDA_CHECK(cudaStreamSynchronize(st_));
}

void Worker::kv_move(int from, int to, int ntok) {
    if (from == to || ntok <= 0) return;
    CUDA_CHECK(cudaSetDevice(dev_));
    const int nr = kv_rows(ntok);
    const size_t row = (size_t)NKV * HD;
    const int shift = std::abs(from - to) * p_.own;
    const int chunk = std::max(IDX_R, std::min(shift, 8192) / IDX_R * IDX_R);
    const size_t fb = (size_t)from * p_.own, tb = (size_t)to * p_.own;
    std::vector<int> starts;
    for (int r = 0; r < nr; r += chunk) starts.push_back(r);
    if (to > from) std::reverse(starts.begin(), starts.end());  // moving up: highest chunk first
    if (kvs_) CUDA_CHECK(cudaStreamSynchronize(st_));  // host rows written by queued kernels before the host moves them
    for (int r : starts) {
        const int n = std::min(chunk, nr - r);
        const int b0 = r / IDX_R, b1 = (r + n + IDX_R - 1) / IDX_R;
        for (const KvBase& k : kvb_) {
            if (k.hm) {  // KV in RAM: host rows, moved on the host (a copy may span several pinned slabs)
                memmove(k.hm->hk + (tb + r) * row, k.hm->hk + (fb + r) * row, (size_t)n * row);
                memmove(k.hm->hv + (tb + r) * row, k.hm->hv + (fb + r) * row, (size_t)n * row);
                memmove(k.hm->hks + (tb + r) * NKV, k.hm->hks + (fb + r) * NKV, (size_t)n * NKV * 4);
                memmove(k.hm->hvs + (tb + r) * NKV, k.hm->hvs + (fb + r) * NKV, (size_t)n * NKV * 4);
            } else {
            CUDA_CHECK(cudaMemcpyAsync(k.kc + (tb + r) * row, k.kc + (fb + r) * row, (size_t)n * row, cudaMemcpyDeviceToDevice, st_));
            CUDA_CHECK(cudaMemcpyAsync(k.vc + (tb + r) * row, k.vc + (fb + r) * row, (size_t)n * row, cudaMemcpyDeviceToDevice, st_));
            CUDA_CHECK(cudaMemcpyAsync(k.ks + (tb + r) * NKV, k.ks + (fb + r) * NKV, (size_t)n * NKV * 4, cudaMemcpyDeviceToDevice, st_));
            CUDA_CHECK(cudaMemcpyAsync(k.vs + (tb + r) * NKV, k.vs + (fb + r) * NKV, (size_t)n * NKV * 4, cudaMemcpyDeviceToDevice, st_));
            }
            CUDA_CHECK(cudaMemcpyAsync(k.ikb + (tb / IDX_R + b0) * IDX_HD, k.ikb + (fb / IDX_R + b0) * IDX_HD, (size_t)(b1 - b0) * IDX_HD * 2,
                                       cudaMemcpyDeviceToDevice, st_));
        }
    }
    kv_inval(to, 0, ntok);  // streaming: the destination's cached blocks are stale
    CUDA_CHECK(cudaStreamSynchronize(st_));
}


// graph launch with little free memory: a launch that fails as out of memory (nothing was enqueued) drops every
// cached graph of the card (e included) and returns false; the caller then runs the same kernels eagerly. Never aborts.
// Test hook FLASH_TEST_GRAPH_LAUNCH_OOM=n: the n-th graph launch of each card (1-based) fails as out of memory.
bool Worker::launch_graph(cudaGraphExec_t& e) {
    static const int inj = env_int("FLASH_TEST_GRAPH_LAUNCH_OOM", 0);
    const bool fake = inj > 0 && ++glaunch_calls_ == inj;
    cudaError_t r = fake ? cudaErrorMemoryAllocation : cudaGraphLaunch(e, st_);
    if (r == cudaErrorMemoryAllocation && oom_reserve_free()) {  // the emergency reserve first (nothing was enqueued)
        (void)cudaGetLastError();
        graph_oom_++;
        fprintf(stderr, "[flash] GPU%d: graph launch out of memory: freed the %.0f MB emergency reserve, retrying\n", dev_,
                oom_res_bytes_ / 1048576.0);
        r = cudaGraphLaunch(e, st_);
    }
    if (r != cudaErrorMemoryAllocation) {
        CUDA_CHECK(r);
        xuse_[e] = ++xtick_;  // LRU (drop_graphs)
        if (!oom_res_ && oom_res_bytes_ && ++oom_ok_ >= kRetakeAfter) oom_reserve_take(true);
        return true;
    }
    (void)cudaGetLastError();
    graph_oom_++;
    oom_ok_ = 0;
    const int n = drop_all_graphs(nullptr);
    e = nullptr;
    gcool_.trip();
    fprintf(stderr, "[flash] GPU%d: graph launch out of memory (%.0f MB free): dropped %d cached graphs, this step runs "
            "without a graph%s\n", dev_, free_bytes() / 1048576.0, n, fake ? " (FLASH_TEST_GRAPH_LAUNCH_OOM)" : "");
    fflush(stderr);
    return false;
}

}  // namespace flash
