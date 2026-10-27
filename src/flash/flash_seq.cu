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
#include "flash.h"
#include <map>

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
        out.push_back({(char*)k.kc + (rb + r0) * row, (size_t)(r1 - r0) * row});
        out.push_back({(char*)k.vc + (rb + r0) * row, (size_t)(r1 - r0) * row});
        out.push_back({(char*)(k.ks + (rb + r0) * NKV), (size_t)(r1 - r0) * NKV * 4});
        out.push_back({(char*)(k.vs + (rb + r0) * NKV), (size_t)(r1 - r0) * NKV * 4});
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

// FLASH_MAX_SEQS slots (default 16), each with its own state storage; the workspace (alloc_state) holds the bound one
void Worker::alloc_slots() {
    int ns = getenv("FLASH_MAX_SEQS") ? atoi(getenv("FLASH_MAX_SEQS")) : 16;
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
        if (L.kc) kvb_.push_back({L.kc, L.vc, L.ks, L.vs, L.ikb});
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
    CUDA_CHECK(cudaHostAlloc((void**)&bem_h_, (size_t)MAXM * (MAXM + 2) * 4, cudaHostAllocMapped | cudaHostAllocPortable));
    memset(bem_h_, 0, (size_t)MAXM * (MAXM + 2) * 4);
    CUDA_CHECK(cudaHostGetDevicePointer((void**)&bem_, bem_h_, 0));
    blk_ = !getenv("FLASH_BATCH_LOOKUP") || atoi(getenv("FLASH_BATCH_LOOKUP"));
    if (getenv("FLASH_BATCH_LK_MIN")) blk_min_ = atoi(getenv("FLASH_BATCH_LK_MIN"));
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

void Worker::drop_graphs(GraphSet& g) {
    auto d = [](cudaGraphExec_t& x) { if (x) { CUDA_CHECK(cudaGraphExecDestroy(x)); x = nullptr; } };
    for (auto& a : g.g) for (auto& b : a) for (auto& c : b) d(c);
    for (auto& a : g.mg) for (auto& b : a) for (auto& c : b) d(c);
    for (auto& a : g.sg) for (auto& b : a) for (auto& c : b) for (auto& x : c) d(x);
    for (auto& [k, x] : g.pg) d(x);
    for (auto& [k, x] : g.pbg) d(x);
    g = GraphSet{};
}

// graph instantiation with little free memory (262K context leaves ~0.3 GB): on OOM drop the other slots' decode
// graphs and the prefill graphs, then retry once
void Worker::inst_graph(cudaGraphExec_t* e, cudaGraph_t g) {
    cudaError_t r = cudaGraphInstantiate(e, g, 0);
    if (r == cudaErrorMemoryAllocation) {
        (void)cudaGetLastError();
        for (int i = 0; i < (int)gstash_.size(); i++)
            if (i != cur_slot_) drop_graphs(gstash_[i]);
        for (auto* c : {&pgraphs_, &pbgraphs_}) {
            for (auto& kv : *c)
                if (kv.second) CUDA_CHECK(cudaGraphExecDestroy(kv.second));
            c->clear();
        }
        for (auto& kv : bgraphs_)
            if (kv.second) { CUDA_CHECK(cudaGraphExecDestroy(kv.second)); kv.second = nullptr; }
        fprintf(stderr, "[flash] GPU%d: graph instantiate out of memory, other graphs dropped\n", dev_);
        r = cudaGraphInstantiate(e, g, 0);
    }
    CUDA_CHECK(r);
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
    for (int r : starts) {
        const int n = std::min(chunk, nr - r);
        const int b0 = r / IDX_R, b1 = (r + n + IDX_R - 1) / IDX_R;
        for (const KvBase& k : kvb_) {
            CUDA_CHECK(cudaMemcpyAsync(k.kc + (tb + r) * row, k.kc + (fb + r) * row, (size_t)n * row, cudaMemcpyDeviceToDevice, st_));
            CUDA_CHECK(cudaMemcpyAsync(k.vc + (tb + r) * row, k.vc + (fb + r) * row, (size_t)n * row, cudaMemcpyDeviceToDevice, st_));
            CUDA_CHECK(cudaMemcpyAsync(k.ks + (tb + r) * NKV, k.ks + (fb + r) * NKV, (size_t)n * NKV * 4, cudaMemcpyDeviceToDevice, st_));
            CUDA_CHECK(cudaMemcpyAsync(k.vs + (tb + r) * NKV, k.vs + (fb + r) * NKV, (size_t)n * NKV * 4, cudaMemcpyDeviceToDevice, st_));
            CUDA_CHECK(cudaMemcpyAsync(k.ikb + (tb / IDX_R + b0) * IDX_HD, k.ikb + (fb / IDX_R + b0) * IDX_HD, (size_t)(b1 - b0) * IDX_HD * 2,
                                       cudaMemcpyDeviceToDevice, st_));
        }
    }
    CUDA_CHECK(cudaStreamSynchronize(st_));
}

}  // namespace flash
