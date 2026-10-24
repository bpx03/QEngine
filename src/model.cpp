#include <unistd.h>
#include <chrono>
#include "model.h"
#include "kernels/ar_fused.cuh"
#include "kernels/watchdog.cuh"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace {

std::string lp(int i) { return "model.language_model.layers." + std::to_string(i) + "."; }
int round_up(int x, int m) { return (x + m - 1) / m * m; }

// Scratch device buffer reused across tensor loads.
struct Scratch {
    void* p = nullptr;
    size_t n = 0;
    void* get(size_t bytes) {
        if (bytes > n) {
            if (p) CUDA_CHECK(cudaFree(p));
            CUDA_CHECK(cudaMalloc(&p, bytes));
            n = bytes;
        }
        return p;
    }
    ~Scratch() { if (p) cudaFree(p); }
};
thread_local Scratch g_scratch_q, g_scratch_s;

}  // namespace

Worker::Worker(const Config& cfg, const std::vector<CardPlan>& plan, int rank_, int device_,
               const Comm& comm)
    : rank(rank_), device(device_), c_(cfg), p_(plan[rank_]), ngpu_(plan.size()), comm_(comm) {}

size_t Worker::slot_token_bytes() const { return tok_bytes_; }

// Pool bytes per token: every local head contributes its share A/P of a head's bytes per key;
// rounded up to 256 so region bases stay aligned. A region of cap keys holds pm_cap(cap) rows
// per head, at most cap * A/P + A(P - A)/P: that excess (over all heads) is the slack every region
// takes beyond cap tokens.
void Worker::plan_bytes() {
    head_tok_bytes_ = (size_t)c_.n_full() * 2 * 256 * (kv_fp8_ ? 1 : 2) + (has_mtp ? (size_t)2 * 256 * (mtp_fp8() ? 1 : 2) : 0);
    double per_tok = 0, slack = 0;
    for (int j = 0; j < ash_.nkv; j++) {
        const PosMap& m = ash_.kv_pm[j];
        per_tok += (double)head_tok_bytes_ * m.A / m.P;
        if (m.A != m.P) slack += (double)head_tok_bytes_ * m.A * (m.P - m.A) / m.P;
    }
    tok_bytes_ = ((size_t)std::ceil(per_tok) + 255) / 256 * 256;
    slack_tokens_ = slack > 0 ? (size_t)std::ceil(slack / tok_bytes_) + 1 : 0;
}

void Worker::set_plan(const CardPlan& p) {
    p_ = p;
    for (int j = 0; j < ash_.nkv; j++) {
        ash_.kv_pm[j] = p_.kv_pm[j];
        ash_.kv_x[j] = p_.kv_x[j];
        // the pair's index in a card's buffers: 0 when the writer is the lower-ranked card
        xridx_[j] = p_.kv_peer[j] < rank ? 0 : 1;  // the peer writes into mine
        xsidx_[j] = rank < p_.kv_peer[j] ? 0 : 1;  // I write into the peer's
    }
    plan_bytes();
}

size_t Worker::slot_fixed_bytes() const {
    return (size_t)c_.n_lin() *
               (kStateSlots * dn_slot_floats_ * 4 + (size_t)kConvRing * dn_ch_ * 2 +
                (dn_defer_on() ? dn_pend_layer_floats() * 4 : 0)) + 256;
}

size_t Worker::free_bytes() const {
    CUDA_CHECK(cudaSetDevice(device));
    gemv_workspace_init();
    size_t fr, tot;
    CUDA_CHECK(cudaMemGetInfo(&fr, &tot));
    return fr;
}

void Worker::add_slot() {
    CUDA_CHECK(cudaSetDevice(device));
    SeqSlot sl;  // KV region: set_region (KV pool)
    for (auto& L : layers_) {
        if (L.full) {
            sl.kc.push_back(nullptr);
            sl.vc.push_back(nullptr);
        } else {
            sl.S.push_back((float*)dalloc(kStateSlots * dn_slot_floats_ * 4, false));
            sl.hist.push_back((bf16*)dalloc((size_t)kConvRing * dn_ch_ * 2, false));
        }
    }
    sl.rdelta = (int*)dalloc(4, false);
    CUDA_CHECK(cudaMemset(sl.rdelta, 0, 4));
    sl.dn_cur = (int*)dalloc(4, false);
    sl.dn_save = (int*)dalloc(4, false);
    if (dn_defer_on()) {
        sl.dn_pn = (int*)dalloc(8, false);
        CUDA_CHECK(cudaMemset(sl.dn_pn, 0, 8));
        const size_t pb = (size_t)c_.n_lin() * dn_pend_layer_floats() * 4;
        sl.dn_pend = (float*)dalloc(pb, false);
        CUDA_CHECK(cudaMemset(sl.dn_pend, 0, pb));
    }
    {  // speculative-decoding state of this sequence
        int* sp = (int*)dalloc(128 * 4, false);
        CUDA_CHECK(cudaMemset(sp, 0, 128 * 4));
        SpecState& ss = sl.spec;
        ss.tokens = (int*)dalloc(kChunk * 4, false);
        ss.out = (int*)dalloc(kChunk * 4, false);
        ss.mtp_tok = (int*)dalloc(kChunk * 4, false);
        ss.mtp_out = (int*)dalloc(kChunk * 4, false);
        ss.pos = sp;
        ss.emit = sp + 64;         // [32]: count + up to 1 + kMaxLookupK tokens
        ss.acc = sp + 32;
        ss.mtp_pos = sp + 33;
        ss.hrow = sp + 34;
        ss.drafts = sp + 40;       // [8]
        sl.pos8 = sp + 36;
        sl.mpos8 = sp + 37;
        ss.eos0 = c_.eos;
        ss.eos1 = c_.eos2;
        sl.sp = (SampleParams*)dalloc(sizeof(SampleParams), false);
        SampleParams d{1.f, 1.f, kTopK, 0.f, 0, 0};
        CUDA_CHECK(cudaMemcpy(sl.sp, &d, sizeof(d), cudaMemcpyHostToDevice));
        const size_t mw = mask_words() * (kMaxSpecK + 1);
        sl.mask = (unsigned*)dalloc(mw * 4, false);
        CUDA_CHECK(cudaMemset(sl.mask, 0xff, mw * 4));
    }
    if (lkb_cap_ > 0) lkb_alloc(sl);
    slots_.push_back(sl);
    const int prev = cur_slot_;
    select_slot(slots_.size() - 1);
    reset();
    if (prev >= 0) select_slot(prev);
}

// ---------------------------------------------------------------- KV pool
// A slot's KV region is cap tokens (+ region_slack) at token offset off of the pool: per
// full-attention layer k then v (the local heads one after another, pm_cap(cap) rows of 256 each:
// [cap] rows for a whole head, the rows of the owned positions for a split one), then the MTP
// layer's k and v. The layout depends on cap, so resizing a region moves its rows (move_region).
void Worker::alloc_pool(size_t tokens) {
    CUDA_CHECK(cudaSetDevice(device));
    pool_ = (char*)dalloc(tokens * slot_token_bytes(), false);
}

void Worker::region_arrays(size_t off, int cap, std::vector<std::pair<char*, size_t>>& out) const {
    out.clear();  // (base of array, bytes per row) in layout order
    char* p = pool_ + off * slot_token_bytes();
    const size_t e = kv_fp8_ ? 1 : 2, me = mtp_fp8() ? 1 : 2;
    const size_t rows = head_rows(ash_, ash_.nkv, cap);  // all local heads
    for (int f = 0; f < 2 * c_.n_full(); f++) {
        out.emplace_back(p, 256 * e);
        p += rows * 256 * e;
    }
    if (has_mtp)
        for (int f = 0; f < 2; f++) {
            out.emplace_back(p, 256 * me);
            p += rows * 256 * me;
        }
    CHECK((size_t)(p - (pool_ + off * slot_token_bytes())) <= ((size_t)cap + slack_tokens_) * slot_token_bytes(),
          "region of %d keys exceeds its pool units", cap);
}

void Worker::set_prefix(int slot, size_t off, int n) {
    SeqSlot& sl = slots_.at(slot);
    sl.split = n;
    sl.kc2.assign(c_.n_full(), nullptr);
    sl.vc2.assign(c_.n_full(), nullptr);
    sl.mkc2 = sl.mvc2 = nullptr;
    if (n > 0) {
        std::vector<std::pair<char*, size_t>> a;
        region_arrays(off, n, a);
        for (int f = 0; f < c_.n_full(); f++) {
            sl.kc2[f] = a[2 * f].first;
            sl.vc2[f] = a[2 * f + 1].first;
        }
        if (has_mtp) {
            sl.mkc2 = (const bf16*)a[2 * c_.n_full()].first;
            sl.mvc2 = (const bf16*)a[2 * c_.n_full() + 1].first;
        }
    }
    if (slot == cur_slot_) select_slot(slot);
}

void Worker::set_region(int slot, size_t off, int cap) {
    SeqSlot& sl = slots_.at(slot);
    std::vector<std::pair<char*, size_t>> a;
    region_arrays(off, cap, a);
    sl.max_t = cap;
    sl.off = off;
    for (int f = 0; f < c_.n_full(); f++) {
        sl.kc[f] = cap ? a[2 * f].first : nullptr;
        sl.vc[f] = cap ? a[2 * f + 1].first : nullptr;
    }
    if (has_mtp) {
        sl.mkc = cap ? (bf16*)a[2 * c_.n_full()].first : nullptr;
        sl.mvc = cap ? (bf16*)a[2 * c_.n_full() + 1].first : nullptr;
    }
    if (slot == cur_slot_) select_slot(slot);
}

// memmove on the device: chunks no longer than the distance, in the safe direction
void Worker::move_bytes(char* dst, const char* src, size_t len) {
    if (dst == src || !len) return;
    const size_t d = dst > src ? dst - src : src - dst;
    if (d >= len) {
        CUDA_CHECK(cudaMemcpyAsync(dst, src, len, cudaMemcpyDeviceToDevice, stream));
        return;
    }
    if (dst < src)
        for (size_t o = 0; o < len; o += d)
            CUDA_CHECK(cudaMemcpyAsync(dst + o, src + o, std::min(d, len - o), cudaMemcpyDeviceToDevice, stream));
    else
        for (size_t end = len; end > 0;) {
            const size_t c = std::min(d, end);
            end -= c;
            CUDA_CHECK(cudaMemcpyAsync(dst + end, src + end, c, cudaMemcpyDeviceToDevice, stream));
        }
}

// Moves rows [0, rows) (keys [split, split + rows)) of every (array, kv head) of a region from
// (off0, cap0) to (off1, cap1). Every array moves the same way (all up or all down: same cap, or
// same offset, so the array bases shift monotonically), so going through them from the far end in
// that direction never overwrites rows that have not moved yet.
void Worker::move_region(size_t off0, int cap0, size_t off1, int cap1, int split, int rows) {
    CUDA_CHECK(cudaSetDevice(device));
    const size_t tb = slot_token_bytes(), sl = slack_tokens_;  // any order when the bytes do not overlap
    const bool disjoint = off1 * tb >= (off0 + cap0 + sl) * tb || (off1 + cap1 + sl) * tb <= off0 * tb;
    CHECK(disjoint || off0 == off1 || cap0 == cap1, "move_region");
    std::vector<std::pair<char*, size_t>> a0, a1;
    region_arrays(off0, cap0, a0);
    region_arrays(off1, cap1, a1);
    struct Item { char* dst; const char* src; size_t len; };
    std::vector<Item> items;
    for (size_t i = 0; i < a0.size(); i++)
        for (int j = 0; j < ash_.nkv; j++) {
            const PosMap& m = ash_.kv_pm[j];
            const size_t n = pm_rank(m, split + rows) - pm_rank(m, split);
            items.push_back({a1[i].first + head_rows(ash_, j, cap1) * a1[i].second,
                             a0[i].first + head_rows(ash_, j, cap0) * a0[i].second, n * a0[i].second});
        }
    const bool up = off1 > off0 || cap1 > cap0;
    if (up) std::reverse(items.begin(), items.end());
    for (auto& it : items) move_bytes(it.dst, it.src, it.len);
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

void Worker::clear_graphs(int slot) {
    CUDA_CHECK(cudaSetDevice(device));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    graphs_.erase_if([&](const auto& k) { return std::get<3>(k) == slot; });
    for (auto it = eager_seen_.begin(); it != eager_seen_.end();)
        it = std::get<3>(*it) == slot ? eager_seen_.erase(it) : std::next(it);
    spec_graphs_.erase_if([&](const auto& k) { return std::get<3>(k) == slot; });
    const std::string tag = "|" + std::to_string(slot) + ":";  // batch keys: kd|slot:kv:..|..
    batch_graphs_.erase_if([&](const std::string& k) { return k.find(tag) != std::string::npos; });
}

void Worker::select_slot(int i) {
    SeqSlot& sl = slots_.at(i);
    cur_slot_ = i;
    ash_.max_t = sl.max_t;
    ash_.split = sl.split;
    ash_.rdelta = sl.rdelta;
    size_t f = 0, l = 0;
    for (auto& L : layers_) {
        if (L.full) {
            L.fa.kc = sl.kc[f];
            L.fa.vc = sl.vc[f];  // void* (bf16 or e4m3)
            L.fa.kc2 = sl.split ? sl.kc2[f] : nullptr;
            L.fa.vc2 = sl.split ? sl.vc2[f] : nullptr;
            f++;
        } else {
            L.dn.S = sl.S[l];
            L.dn.hist = sl.hist[l];
            l++;
        }
    }
    mtp_.kc = sl.mkc;
    mtp_.vc = sl.mvc;
    mtp_.kc2 = sl.mkc2;
    mtp_.vc2 = sl.mvc2;
    dn_cur_ = sl.dn_cur;
    dn_save_ = sl.dn_save;
    dn_pn_ = sl.dn_pn;
    dn_pend_ = sl.dn_pend;
    spec_ = sl.spec;
    tokens_ = spec_.tokens;
    pos_ = spec_.pos;
    out_tok_ = spec_.out;
    sp_ = sl.sp;
    sampling = sl.sampling;
    constrained = sl.constrained;
}

// Deferred commit: dn_stash_seg left the undo in (dn_save, dn_pn[1]): a step that stashed its rows
// committed the previous ones (S[cur] after it, nothing pending); a single-row step that wrote
// the state directly goes back to the old slot with the old pending rows.
void Worker::undo_spec() {
    CUDA_CHECK(cudaSetDevice(device));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaMemcpyAsync(dn_cur_, dn_save_, 4, cudaMemcpyDeviceToDevice, stream));
    if (dn_pn_) CUDA_CHECK(cudaMemcpyAsync(dn_pn_, dn_pn_ + 1, 4, cudaMemcpyDeviceToDevice, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

void Worker::dn_flush_cur(cudaStream_t st) {
    if (!dn_pn_) return;
    DnFlush r{};
    int li = 0;
    for (auto& L : layers_)
        if (!L.full) r.S[li++] = L.dn.S;
    r.cur = dn_cur_;
    r.pn = dn_pn_;
    r.pend = dn_pend_;
    dn_flush(r, li, dsh_, dn_slot_floats_, st);
}

// Host layout of the keys [lo, hi) of every (layer, local kv head, k|v) array: the rows this card
// owns (pm_rank(hi) - pm_rank(lo) per head), after the delta-net state
template <typename F>
static void archive_layout(size_t dn_bytes, int nfull, const AttnShape& s, bool mtp, size_t esz, size_t mesz,
                           int lo, int hi, size_t chunk, F&& piece) {
    int ci = 0;
    size_t off = 0;
    auto place = [&](size_t bytes) {
        if (off + bytes > chunk) { ci++; off = 0; }
        const size_t o = off;
        off += bytes;
        return std::make_pair(ci, o);
    };
    auto p = place(dn_bytes);
    piece(-1, 0, 0, p.first, p.second, (size_t)0);  // delta-net state
    for (int f = 0; f < nfull + (mtp ? 1 : 0); f++)
        for (int j = 0; j < s.nkv; j++) {
            const size_t rows = pm_rank(s.kv_pm[j], hi) - pm_rank(s.kv_pm[j], lo);
            for (int kvv = 0; kvv < 2; kvv++) {
                p = place(rows * 256 * (f < nfull ? esz : mesz));
                piece(f, j, kvv, p.first, p.second, rows);
            }
        }
}

int Worker::archive_chunks(int lo, int hi, size_t chunk) const {
    int last = 0;
    archive_layout(snapshot_bytes(), c_.n_full(), ash_, has_mtp, kv_fp8_ ? 1 : 2, mtp_fp8() ? 1 : 2, lo, hi, chunk,
                   [&](int, int, int, int ci, size_t, size_t) { last = ci; });
    return last + 1;
}

size_t Worker::archive_bytes(int lo, int hi) const {
    size_t end = 0;
    archive_layout(snapshot_bytes(), c_.n_full(), ash_, has_mtp, kv_fp8_ ? 1 : 2, mtp_fp8() ? 1 : 2, lo, hi,
                   ~(size_t)0 >> 1, [&](int f, int, int, int, size_t o, size_t rows) {
                       end = o + rows * 256 * (f < c_.n_full() ? (kv_fp8_ ? 1 : 2) : (mtp_fp8() ? 1 : 2));
                   });
    return end;  // the last piece's offset + its size
}

int Worker::archive_xfer(int slot, int lo, int hi, int copy_hi, void* const* chunks, size_t chunk, bool save,
                         bool with_dn, int piece, size_t budget) {
    CHECK(slot == cur_slot_ && copy_hi <= hi, "archive_xfer");
    const SeqSlot& sl = slots_.at(slot);
    CHECK(lo == sl.split, "archive_xfer: layout from %d, slot split %d", lo, sl.split);
    return region_xfer(sl.off, sl.max_t, sl.split, lo, hi, copy_hi, chunks, chunk, save, with_dn, piece, budget);
}

int Worker::region_xfer(size_t off, int cap, int split, int lo, int hi, int copy_hi, void* const* chunks,
                        size_t chunk, bool save, bool with_dn, int piece, size_t budget) {
    CUDA_CHECK(cudaSetDevice(device));
    CHECK(lo >= split && lo <= copy_hi && copy_hi <= hi && copy_hi - split <= cap, "region_xfer keys [%d, %d) of [%d, %d) into cap %d from %d",
          lo, copy_hi, lo, hi, cap, split);
    std::vector<std::pair<char*, size_t>> arr;
    region_arrays(off, cap, arr);
    const size_t esz = kv_fp8_ ? 1 : 2, mesz = mtp_fp8() ? 1 : 2;
    const int nfull = c_.n_full();
    int idx = 0, next = -1;
    size_t moved = 0;
    archive_layout(snapshot_bytes(), nfull, ash_, has_mtp, esz, mesz, lo, hi, chunk,
                   [&](int f, int j, int kvv, int ci, size_t o, size_t) {
        const int me = idx++;
        if (me < piece || next >= 0) return;
        if (moved >= budget) { next = me; return; }
        char* host = (char*)chunks[ci] + o;
        if (f < 0) {
            if (!with_dn) return;
            if (save) snapshot_save(host);
            else snapshot_restore(host);
            moved += snapshot_bytes();
            return;
        }
        const size_t e = f < nfull ? esz : mesz;
        const PosMap& m = ash_.kv_pm[j];
        const size_t n = pm_rank(m, copy_hi) - pm_rank(m, lo);  // owned keys to move
        const size_t row = pm_rank(m, lo) - pm_rank(m, split);   // their first row in the region
        char* dev = arr[2 * f + kvv].first + (head_rows(ash_, j, cap) + row) * 256 * e;
        const size_t bytes = n * 256 * e;
        if (!bytes) return;
        if (save) CUDA_CHECK(cudaMemcpyAsync(host, dev, bytes, cudaMemcpyDeviceToHost, stream));
        else CUDA_CHECK(cudaMemcpyAsync(dev, host, bytes, cudaMemcpyHostToDevice, stream));
        moved += bytes;
    });
    CUDA_CHECK(cudaStreamSynchronize(stream));
    return next;
}

void Worker::copy_keys(size_t src_off, int src_cap, int src_split, size_t dst_off, int dst_cap, int dst_split, int lo,
                       int hi) {
    if (hi <= lo) return;
    CUDA_CHECK(cudaSetDevice(device));
    CHECK(lo >= src_split && lo >= dst_split && hi - src_split <= src_cap && hi - dst_split <= dst_cap, "copy_keys [%d, %d)",
          lo, hi);
    std::vector<std::pair<char*, size_t>> a0, a1;
    region_arrays(src_off, src_cap, a0);
    region_arrays(dst_off, dst_cap, a1);
    for (size_t i = 0; i < a0.size(); i++)
        for (int j = 0; j < ash_.nkv; j++) {
            const PosMap& m = ash_.kv_pm[j];
            const size_t n = pm_rank(m, hi) - pm_rank(m, lo);
            if (!n) continue;
            CUDA_CHECK(cudaMemcpyAsync(
                a1[i].first + (head_rows(ash_, j, dst_cap) + pm_rank(m, lo) - pm_rank(m, dst_split)) * a1[i].second,
                a0[i].first + (head_rows(ash_, j, src_cap) + pm_rank(m, lo) - pm_rank(m, src_split)) * a0[i].second,
                n * a0[i].second, cudaMemcpyDeviceToDevice, stream));
        }
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

// ---------------------------------------------------------------- stats exchange (kv-pos-split)
void Worker::alloc_xchg() {
    CUDA_CHECK(cudaSetDevice(device));
    xdone_ = (unsigned*)dalloc(4, false);
    CUDA_CHECK(cudaMemset(xdone_, 0, 4));
    rtab_ = (double2*)dalloc((size_t)kChunk * 32 * sizeof(double2), false);
    bool any = false;
    for (int j = 0; j < ash_.nkv; j++) any |= ash_.kv_x[j] == 3;
    if (!any) return;
    const int gh = c_.n_q / c_.n_kv;
    xbuf_ = (float*)dalloc((size_t)2 * 2 * kChunk * gh * kXchgFloats * 4, false);
    xflags_ = (unsigned*)dalloc(4 * sizeof(unsigned), false);
    CUDA_CHECK(cudaMemset(xflags_, 0, 4 * sizeof(unsigned)));
}

void Worker::set_xchg_peer(int j, float* buf, unsigned* flags) {
    xpeer_buf_[j] = buf;
    xpeer_flags_[j] = flags;
}

// Exchange pointers of the next full-attention layer: receive slot = exchange index parity (two
// slots per peer; the all-reduce between layers already orders writer and reader)
void Worker::xchg_fill(AttnShape& a) {
    a.rtab = p_.pos_split ? rtab_ : nullptr;  // (the old layout keeps the in-kernel RoPE)
    a.xepoch = ctr_.epoch;
    a.xdone = xdone_;
    a.xidx = xidx_;
    const int gh = c_.n_q / c_.n_kv, slot = xidx_ & 1;
    bool any = false;
    for (int j = 0; j < ash_.nkv; j++) {
        if (ash_.kv_x[j] != 3) continue;
        any = true;
        CHECK(xbuf_ && xpeer_buf_[j] && xpeer_flags_[j], "exchange buffers of kv head %d not set", j);
        a.xo[j] = xbuf_ + ((size_t)(slot * 2 + xridx_[j]) * kChunk) * gh * kXchgFloats;
        a.xflag[j] = xflags_ + xridx_[j] * 2 + slot;
        a.xpo[j] = xpeer_buf_[j] + ((size_t)(slot * 2 + xsidx_[j]) * kChunk) * gh * kXchgFloats;
        a.xpflag[j] = xpeer_flags_[j] + xsidx_[j] * 2 + slot;
    }
    if (any) xidx_++;
}

void Worker::tune_gemm() {
    CUDA_CHECK(cudaSetDevice(device));
    gemm_tune_defer(true);  // one save of the table after all shapes
    const int H = c_.hidden, I = p_.inter.size();
    const Layer* fa = nullptr;
    const Layer* dn = nullptr;
    for (auto& L : layers_) {
        if (L.full && !fa) fa = &L;
        if (!L.full && !dn) dn = &L;
    }
    for (int t = 64; t <= kChunk; t *= 2) {  // same calls (shapes, operands) as forward_prefill
        if (fa) {
            matmul(fa->fa.qkv, xn_, H, proj_, proj_ld_, t, stream);
            matmul(fa->fa.o, act_, ow_, partial_, H, t, stream);
        }
        if (dn) {
            matmul(dn->dn.in, xn_, H, proj_, proj_ld_, t, stream);
            matmul(dn->dn.out, act_, dsh_.nv * 128, partial_, H, t, stream);
        }
        const Layer& L = layers_[0];
        if (L.gate_up.glu) matmul(L.gate_up, xn_, H, reinterpret_cast<float*>(act_), I, t, stream);
        else matmul(L.gate_up, xn_, H, proj_, proj_ld_, t, stream);
        matmul(L.down, act_, I, partial_, H, t, stream);
        if (has_mtp) {
            matmul(mtp_.fc, mtp_in_ + p_.fc_k.b, 2 * H, partial_, H, t, stream);
            matmul(mtp_.qkv, xn_, H, proj_, proj_ld_, t, stream);
            matmul(mtp_.o, act_, ow_, partial_, H, t, stream);
            matmul(mtp_.gate_up, xn_, H, proj_, proj_ld_, t, stream);
            matmul(mtp_.down, act_, I, partial_, H, t, stream);
        }
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    gemm_tune_defer(false);
}

void Worker::undo_slot(int i) {
    CUDA_CHECK(cudaSetDevice(device));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    SeqSlot& sl = slots_.at(i);
    CUDA_CHECK(cudaMemcpyAsync(sl.dn_cur, sl.dn_save, 4, cudaMemcpyDeviceToDevice, stream));
    if (sl.dn_pn) CUDA_CHECK(cudaMemcpyAsync(sl.dn_pn, sl.dn_pn + 1, 4, cudaMemcpyDeviceToDevice, stream));  // (see undo_spec)
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

void Worker::trim_slot(int i, int r, bool stash) {
    CUDA_CHECK(cudaSetDevice(device));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    SeqSlot& sl = slots_.at(i);
    if (stash) {
        dn_trim(sl.dn_cur, sl.dn_save, sl.dn_pn, r, stream);
    } else {
        copy_int(sl.dn_cur, sl.dn_save, stream);
        if (r > 0) {
            if (!trim_acc_) trim_acc_ = (int*)dalloc(4, false);
            const int acc = r - 1;
            CUDA_CHECK(cudaMemcpy(trim_acc_, &acc, 4, cudaMemcpyHostToDevice));
            DnReplay rp{};
            int li = 0;
            for (size_t l = 0; l < layers_.size(); l++)
                if (!layers_[l].full) {
                    rp.S[li] = sl.S[li];
                    rp.cv[li] = spec_cv_[li];
                    rp.qk[li] = spec_qk_[li];
                    rp.gb[li] = spec_gb_[li];
                    li++;
                }
            rp.n = li;
            dn_replay(rp, dn_ch_, dsh_, dn_slot_floats_, sl.dn_cur, trim_acc_, stream);
            dn_commit(sl.dn_cur, nullptr, 1, stream);
        }
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

void Worker::spec_resync(int i, int pos, int tok) {
    CUDA_CHECK(cudaSetDevice(device));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    const SpecState& ss = slots_.at(i).spec;
    CUDA_CHECK(cudaMemcpy(ss.tokens, &tok, 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ss.pos, &pos, 4, cudaMemcpyHostToDevice));
}

void* Worker::dalloc(size_t bytes, bool weight) {
    void* p;
    CUDA_CHECK(cudaMalloc(&p, bytes));
    (weight ? weight_bytes : cache_bytes) += bytes;
    allocs_.push_back(p);
    return p;
}

// Copy-engine AR synchronization without device-side spinning: the copy thread of the sender records
// an event after each copy and publishes it here; the receiver's worker thread takes it (host wait
// for the publication only - the copy threads issue asynchronous work and publish at once) and makes
// its comm stream wait on it (cross-device event wait).
namespace {
struct CeHub {
    std::mutex mu;
    std::condition_variable cv;
    std::map<std::tuple<unsigned, int, int, int>, cudaEvent_t> ev;  // (seq, kind, src, dst)
    void put(unsigned seq, int kind, int src, int dst, cudaEvent_t e) {
        {
            std::lock_guard<std::mutex> lk(mu);
            ev[std::make_tuple(seq, kind, src, dst)] = e;
        }
        cv.notify_all();
    }
    cudaEvent_t take(unsigned seq, int kind, int src, int dst) {
        std::unique_lock<std::mutex> lk(mu);
        const auto key = std::make_tuple(seq, kind, src, dst);
        // bounded (collective watchdog): a peer whose host thread never publishes would hang this worker
        const int ms = qe_wd_budget_ms();
        if (ms <= 0) cv.wait(lk, [&] { return ev.count(key) != 0; });
        else if (!cv.wait_for(lk, std::chrono::milliseconds(ms), [&] { return ev.count(key) != 0; })) {
            char b[200];
            snprintf(b, sizeof b, "copy-engine all-reduce: GPU rank %d waited %d ms for the copy event of rank %d "
                     "(seq %u kind %d): rank %d not responding", dst, ms, src, seq, kind, src);
            lk.unlock();
            qe_wd_host_timeout(b);
        }
        cudaEvent_t e = ev[key];
        ev.erase(key);
        return e;
    }
};
CeHub g_ce_hub;
}  // namespace

void Worker::ce_take_wait(unsigned seq, int kind, cudaStream_t st) {
    for (int g = 0; g < ngpu_; g++) {
        if (g == rank) continue;
        cudaEvent_t e = g_ce_hub.take(seq, kind, g, rank);
        CUDA_CHECK(cudaStreamWaitEvent(st, e, 0));
        ce_old_take_.push_back(e);  // destroyed later (the wait above may still be pending)
        if (ce_old_take_.size() > 512) {
            CUDA_CHECK(cudaEventDestroy(ce_old_take_.front()));
            ce_old_take_.pop_front();
        }
    }
}

void Worker::ce_push(const CeCmd& c) {
    if (!ce_thread_.joinable()) {
        for (auto& s : ce_st_) CUDA_CHECK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
        ce_thread_ = std::thread([this] { ce_loop(); });
    }
    {
        std::lock_guard<std::mutex> lk(ce_mu_);
        ce_q_.push_back(c);
    }
    ce_cv_.notify_one();
}

void Worker::ce_loop() {
    CUDA_CHECK(cudaSetDevice(device));
    for (;;) {
        CeCmd c;
        {
            std::unique_lock<std::mutex> lk(ce_mu_);
            ce_cv_.wait(lk, [&] { return !ce_q_.empty(); });
            c = ce_q_.front();
            ce_q_.pop_front();
        }
        if (c.stop) return;
        static const bool dbgce = getenv("QE_CE_DEBUG") != nullptr;
        if (dbgce && c.seq < 4) fprintf(stderr, "ce copy thread dev %d seq %u: %zu B -> %p, %zu B -> %p\n", device, c.seq, c.cp[0].bytes, c.cp[0].dst, c.cp[1].bytes, c.cp[1].dst);
        for (int j = 0; j < 2; j++) {
            if (c.peer[j] < 0) continue;
            CUDA_CHECK(cudaStreamWaitEvent(ce_st_[j], c.ev, 0));
            if (c.ncp[j] == 1 && c.cp[j].bytes)
                CUDA_CHECK(cudaMemcpyPeerAsync(c.cp[j].dst, c.cp[j].dev, c.cp[j].src, device, c.cp[j].bytes, ce_st_[j]));
            for (int u = 0; c.ncp[j] > 1 && u < c.ncp[j]; u++)
                CUDA_CHECK(cudaMemcpyPeerAsync(c.cpx[j][u].dst, c.cpx[j][u].dev, c.cpx[j][u].src, device,
                                               c.cpx[j][u].bytes, ce_st_[j]));
            cudaEvent_t done;
            CUDA_CHECK(cudaEventCreateWithFlags(&done, cudaEventDisableTiming));
            CUDA_CHECK(cudaEventRecord(done, ce_st_[j]));
            if (qe_fault_skip(rank, "ce")) continue;  // test only (QE_FAULT_KIND=ce): never published
            g_ce_hub.put(c.seq, c.kind, rank, c.peer[j], done);  // the receiver destroys it
        }
        if (dbgce && c.seq < 3) {
            for (int j = 0; j < 2; j++) {
                const cudaError_t q = cudaStreamQuery(ce_st_[j]);
                fprintf(stderr, "ce dev %d seq %u stream %d query %s ev %s\n", device, c.seq, j, cudaGetErrorName(q), cudaGetErrorName(cudaEventQuery(c.ev)));
            }
            usleep(200000);
            for (int j = 0; j < 2; j++)
                fprintf(stderr, "ce dev %d seq %u stream %d after 0.2 s: %s ev %s\n", device, c.seq, j, cudaGetErrorName(cudaStreamQuery(ce_st_[j])), cudaGetErrorName(cudaEventQuery(c.ev)));
        }
        ce_old_.push_back(c.ev);  // destroyed later (not while a wait on it may be pending)
        if (ce_old_.size() > 256) { CUDA_CHECK(cudaEventDestroy(ce_old_.front())); ce_old_.pop_front(); }
    }
}

Worker::~Worker() {
    if (ready_) {  // QE_EARLY_GEMV guard timeouts (each = a consumer that gave up waiting: wrong results)
        std::vector<unsigned> h(1024);
        if (cudaSetDevice(device) == cudaSuccess &&
            cudaMemcpy(h.data(), ready_ + 2048, 4096, cudaMemcpyDeviceToHost) == cudaSuccess) {
            unsigned long long t = 0;
            for (unsigned v : h) t += v;
            if (t) fprintf(stderr, "GPU%d: QE_EARLY_GEMV: %llu consumer waits timed out\n", device, t);
        }
    }
    cudaSetDevice(device);
    if (ce_thread_.joinable()) {
        CeCmd c;
        c.stop = true;
        ce_push(c);
        ce_thread_.join();
    }
    cudaDeviceSynchronize();
    for (cudaStream_t s : ce_st_)
        if (s) cudaStreamDestroy(s);
    graphs_.clear();
    spec_graphs_.clear();
    batch_graphs_.clear();
    for (void* p : allocs_) cudaFree(p);
    for (auto e : evs_) cudaEventDestroy(e);
    for (cudaStream_t s : {stream, comm_stream_, pf_stream_})
        if (s) cudaStreamDestroy(s);
}

W4 Worker::load_w4(const SafeTensors& st, const std::vector<RowSpec>& rows, int c0, int c1,
                   bool glu) {
    const TensorInfo& t0 = st.get(rows[0].name + ".weight_packed");
    const int K = t0.shape[1] * 8;
    if (c0 < 0) { c0 = 0; c1 = K; }
    const int kl = c1 - c0;
    int n = 0;
    for (auto& r : rows) n += r.r1 - r.r0;
    CHECK(n % 16 == 0 && kl % 128 == 0 && c0 % 128 == 0, "w4 %s n=%d k=%d", rows[0].name.c_str(),
          n, kl);
    int32_t* tq = (int32_t*)g_scratch_q.get((size_t)n * kl / 2);
    bf16* ts = (bf16*)g_scratch_s.get((size_t)n * (kl / 128) * 2);
    int off = 0;
    for (auto& r : rows) {
        const TensorInfo& q = st.get(r.name + ".weight_packed");
        const TensorInfo& s = st.get(r.name + ".weight_scale");
        CHECK(q.shape[1] * 8 == K, "K mismatch in %s", r.name.c_str());
        const int h = r.r1 - r.r0;
        CUDA_CHECK(cudaMemcpy2D(tq + (size_t)off * (kl / 8), kl / 2,
                                q.data + ((size_t)r.r0 * (K / 8) + c0 / 8) * 4, K / 2, kl / 2, h,
                                cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy2D(ts + (size_t)off * (kl / 128), kl / 128 * 2,
                                s.data + ((size_t)r.r0 * (K / 128) + c0 / 128) * 2, K / 128 * 2,
                                kl / 128 * 2, h, cudaMemcpyHostToDevice));
        off += h;
    }
    W4 w;
    w.n = n;
    w.k = kl;
    w.glu = glu;
    uint4* dq = (uint4*)dalloc(w4_qbytes(n, kl), true);
    bf16* ds = (bf16*)dalloc(w4_sbytes(n, kl), true);
    if (glu) {  // interleave gate/up rows per 16-row tile, via a temporary copy
        int32_t* gq;
        bf16* gs;
        CUDA_CHECK(cudaMalloc(&gq, (size_t)n * kl / 2));
        CUDA_CHECK(cudaMalloc(&gs, (size_t)n * (kl / 128) * 2));
        glu_interleave_rows(tq, ts, n, kl, gq, gs, stream);
        repack_w4(gq, gs, n, kl, dq, ds, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaFree(gq));
        CUDA_CHECK(cudaFree(gs));
    } else {
        repack_w4(tq, ts, n, kl, dq, ds, stream);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    w.q = dq;
    w.s = ds;
    return w;
}

bf16* Worker::gather_bf16(const SafeTensors& st, const std::vector<RowSpec>& rows, int pad_to,
                          int c0, int c1, int* n_out, int* k_out) {
    const int K = st.get(rows[0].name).shape[1];
    if (c0 < 0) { c0 = 0; c1 = K; }
    const int kl = c1 - c0;
    int n = 0;
    for (auto& r : rows) n += round_up(r.r1 - r.r0, pad_to);
    CHECK(n % 16 == 0 && kl % 16 == 0, "w16 rows %d k %d", n, kl);
    bf16* tmp = (bf16*)g_scratch_q.get((size_t)n * kl * 2);
    CUDA_CHECK(cudaMemsetAsync(tmp, 0, (size_t)n * kl * 2, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    int off = 0;
    for (auto& r : rows) {
        const TensorInfo& t = st.get(r.name);
        CHECK(t.dtype == "BF16" && t.shape[1] == K, "w16 %s", r.name.c_str());
        CUDA_CHECK(cudaMemcpy2D(tmp + (size_t)off * kl, (size_t)kl * 2,
                                t.data + ((size_t)r.r0 * K + c0) * 2, (size_t)K * 2,
                                (size_t)kl * 2, r.r1 - r.r0, cudaMemcpyHostToDevice));
        off += round_up(r.r1 - r.r0, pad_to);
    }
    *n_out = n;
    *k_out = kl;
    return tmp;
}

W16 Worker::load_w16(const SafeTensors& st, const std::vector<RowSpec>& rows, int pad_to, int c0,
                     int c1) {
    W16 w;
    bf16* tmp = gather_bf16(st, rows, pad_to, c0, c1, &w.n, &w.k);
    uint4* d = (uint4*)dalloc(w16_bytes(w.n, w.k), true);
    repack_w16(tmp, w.n, w.k, d, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    w.w = d;
    return w;
}

Lin Worker::load_lin(const SafeTensors& st, const std::vector<RowSpec>& rows, int pad_to,
                     bool as4, int c0, int c1) {
    Lin l;
    l.is4 = as4;
    if (!as4) {
        l.f = load_w16(st, rows, pad_to, c0, c1);
        return l;
    }
    int n, k;
    bf16* tmp = gather_bf16(st, rows, pad_to, c0, c1, &n, &k);
    int32_t* q = (int32_t*)g_scratch_s.get((size_t)n * k / 2 + (size_t)n * (k / 128) * 2);
    bf16* sc = (bf16*)((char*)q + (size_t)n * k / 2);
    quantize_w4_rtn(tmp, n, k, q, sc, stream);
    l.q.n = n;
    l.q.k = k;
    uint4* dq = (uint4*)dalloc(w4_qbytes(n, k), true);
    bf16* ds = (bf16*)dalloc(w4_sbytes(n, k), true);
    repack_w4(q, sc, n, k, dq, ds, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    l.q.q = dq;
    l.q.s = ds;
    return l;
}

void Worker::load_mtp(const SafeTensors& st) {
    const int H = c_.hidden;
    const std::string a = "mtp.layers.0.self_attn.", l = "mtp.layers.0.";
    const char* env = getenv("QE_MTP_W4");
    const bool w4 = env ? atoi(env) != 0 : true;
    mtp_.fc = load_lin(st, {{"mtp.fc.weight", 0, H}}, 16, w4, p_.fc_k.b, p_.fc_k.e);
    mtp_.qkv = load_lin(st,
                        {{a + "q_proj.weight", p_.q_heads.b * 512, p_.q_heads.e * 512},
                         {a + "k_proj.weight", p_.kv_heads.b * 256, p_.kv_heads.e * 256},
                         {a + "v_proj.weight", p_.kv_heads.b * 256, p_.kv_heads.e * 256}},
                        16, w4);
    mtp_.o = load_lin(st, {{a + "o_proj.weight", 0, H}}, 16, w4, p_.o_heads.b * 256,
                      p_.o_heads.e * 256);
    mtp_.gate_up = load_lin(st,
                            {{l + "mlp.gate_proj.weight", p_.inter.b, p_.inter.e},
                             {l + "mlp.up_proj.weight", p_.inter.b, p_.inter.e}},
                            16, w4);
    mtp_.down = load_lin(st, {{l + "mlp.down_proj.weight", 0, H}}, 16, w4, p_.inter.b, p_.inter.e);
    mtp_.pre_e = load_bf16(st, "mtp.pre_fc_norm_embedding.weight", -1, -1);
    mtp_.pre_h = load_bf16(st, "mtp.pre_fc_norm_hidden.weight", -1, -1);
    mtp_.in_norm = load_bf16(st, l + "input_layernorm.weight", -1, -1);
    mtp_.post_norm = load_bf16(st, l + "post_attention_layernorm.weight", -1, -1);
    mtp_.q_norm = load_bf16(st, a + "q_norm.weight", -1, -1);
    mtp_.k_norm = load_bf16(st, a + "k_norm.weight", -1, -1);
    mtp_.norm = load_bf16(st, "mtp.norm.weight", -1, -1);
    has_mtp = true;
}

bf16* Worker::load_bf16(const SafeTensors& st, const std::string& name, int b, int e) {
    const TensorInfo& t = st.get(name);
    CHECK(t.dtype == "BF16", "%s dtype %s", name.c_str(), t.dtype.c_str());
    size_t row = t.nbytes / 2 / t.shape[0];  // elements per leading index
    if (e < 0) { b = 0; e = t.shape[0]; }
    bf16* d = (bf16*)dalloc((e - b) * row * 2, true);
    CUDA_CHECK(cudaMemcpy(d, t.data + b * row * 2, (e - b) * row * 2, cudaMemcpyHostToDevice));
    return d;
}

float* Worker::load_f32(const SafeTensors& st, const std::string& name, int b, int e,
                        bool neg_exp) {
    const TensorInfo& t = st.get(name);
    std::vector<float> h(e - b);
    for (int i = b; i < e; i++) {
        float v;
        if (t.dtype == "BF16") {
            uint32_t u = (uint32_t)((const uint16_t*)t.data)[i] << 16;
            memcpy(&v, &u, 4);
        } else {
            v = ((const float*)t.data)[i];
        }
        h[i - b] = neg_exp ? -std::exp(v) : v;
    }
    float* d = (float*)dalloc(h.size() * 4, true);
    CUDA_CHECK(cudaMemcpy(d, h.data(), h.size() * 4, cudaMemcpyHostToDevice));
    return d;
}

void Worker::load(const SafeTensors& st, const SafeTensors* mtp_st, const bf16* embed_mapped) {
    CUDA_CHECK(cudaSetDevice(device));
    gemv_workspace_init();  // wide-GEMV scratch (also counted before slot planning)
    // ~7 MB of VRAM per decode-step graph; GPU headroom after the slots is ~1 GB on a card
    // that also drives a desktop
    spec_graphs_.cap = getenv("QE_GRAPHS_SPEC") ? atoi(getenv("QE_GRAPHS_SPEC")) : 64;
    batch_graphs_.cap = getenv("QE_GRAPHS_BATCH") ? atoi(getenv("QE_GRAPHS_BATCH")) : 24;
    graphs_.cap = 16;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    {
        int lo, hi;  // all-reduces get the higher priority so their blocks start promptly
        CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&lo, &hi));
        CUDA_CHECK(cudaStreamCreateWithPriority(&comm_stream_, cudaStreamNonBlocking, hi));
        CUDA_CHECK(cudaStreamCreateWithFlags(&pf_stream_, cudaStreamNonBlocking));
        // L2 prefetch of the next weights beside each decode all-reduce: default off since the RS+AG all-reduce
        // (8x19K 18,19 -> 17,86 ms/step with 0; n=1 4K 11,44 vs 11,45); QE_PF_MB=5 restores the old 5 MB
        pf_bytes_ = (size_t)(getenv("QE_PF_MB") ? atof(getenv("QE_PF_MB")) : 0.0) * 1e6;
        evs_.resize(16);
        for (auto& e : evs_) CUDA_CHECK(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
    }
    embed_ = embed_mapped;
    if (const char* e = getenv("QE_KV_FP8")) kv_fp8_ = atoi(e) != 0;
    if (const char* e = getenv("QE_GLU")) glu_ = atoi(e) != 0;
    attention_init();
    attention_prefill_init();
    dn_chunk_init();
    qe_wd_init(rank, device);  // collective watchdog: budget, rank and the host record on this device
    const int H = c_.hidden;

    // --- shapes of this card's slice ---
    const int qpk = c_.n_q / c_.n_kv, vpk = c_.lin_nv / c_.lin_nk;
    ash_ = {};
    ash_.nq = p_.q_heads.size();
    ash_.nkv = p_.kv_heads.size();
    ash_.max_t = 0;  // set by select_slot
    CHECK(ash_.nkv <= kMaxLocalKv && ash_.nq <= 32, "attn shape");
    for (int k = 0; k < ash_.nkv; k++) {
        const int gk = p_.kv_heads.b + k;
        ash_.kv_qb[k] = std::max(gk * qpk, p_.q_heads.b) - p_.q_heads.b;
        ash_.kv_qe[k] = std::min((gk + 1) * qpk, p_.q_heads.e) - p_.q_heads.b;
    }
    for (int h = 0; h < ash_.nq; h++) ash_.q2kv[h] = (p_.q_heads.b + h) / qpk - p_.kv_heads.b;
    // o_proj heads (the attention output rows): a subset of the q heads computed here
    CHECK(p_.o_heads.b >= p_.q_heads.b && p_.o_heads.e <= p_.q_heads.e, "o_proj heads outside the q heads");
    ash_.ob = p_.o_heads.b - p_.q_heads.b;
    ash_.no = p_.o_heads.size();
    ow_ = ash_.no * 256;
    ash_.dchunk_min = getenv("QE_DEC_CHUNK_MIN") ? atoi(getenv("QE_DEC_CHUNK_MIN")) : -1;
    if (const char* e = getenv("QE_KV_XDIV")) ash_.xdiv = atoi(e);
    if (const char* e = getenv("QE_KV_XSKIP")) ash_.xskip = atoi(e);
    set_plan(p_);
    dsh_ = {};
    dsh_.nk = p_.k_heads.size();
    dsh_.nv = p_.v_heads.size();
    dsh_.ab_pad = round_up(dsh_.nv, 8);
    dsh_.ab_splits = kAbSplits;
    dsh_.ab_rows = kChunk;
    CHECK(dsh_.nv <= 64, "delta shape");
    for (int v = 0; v < dsh_.nv; v++) dsh_.v2k[v] = (p_.v_heads.b + v) / vpk - p_.k_heads.b;
    dn_ch_ = (2 * dsh_.nk + dsh_.nv) * 128;
    const int I = p_.inter.size();
    const int kd = c_.lin_nk * 128;  // global key dim of in_proj_qkv sections

    layers_.resize(c_.layers);
    for (int i = 0; i < c_.layers; i++) {
        Layer& L = layers_[i];
        const std::string pre = lp(i);
        L.full = c_.is_full[i];
        L.in_norm = load_bf16(st, pre + "input_layernorm.weight", -1, -1);
        L.post_norm = load_bf16(st, pre + "post_attention_layernorm.weight", -1, -1);
        if (L.full) {
            const std::string a = pre + "self_attn.";
            L.fa.qkv = load_w4(st,
                               {{a + "q_proj", p_.q_heads.b * 512, p_.q_heads.e * 512},
                                {a + "k_proj", p_.kv_heads.b * 256, p_.kv_heads.e * 256},
                                {a + "v_proj", p_.kv_heads.b * 256, p_.kv_heads.e * 256}},
                               -1, -1);
            L.fa.o = load_w4(st, {{a + "o_proj", 0, H}}, p_.o_heads.b * 256, p_.o_heads.e * 256);
            L.fa.q_norm = load_bf16(st, a + "q_norm.weight", -1, -1);
            L.fa.k_norm = load_bf16(st, a + "k_norm.weight", -1, -1);
            if (st.has(a + "k_scale") && st.has(a + "v_scale")) {
                auto scalar = [&](const std::string& n) {
                    const TensorInfo& t = st.get(n);
                    uint32_t u = (uint32_t)((const uint16_t*)t.data)[0] << 16;
                    float v;
                    memcpy(&v, &u, 4);
                    return t.dtype == "BF16" ? v : ((const float*)t.data)[0];
                };
                L.fa.ks = scalar(a + "k_scale");
                L.fa.vs = scalar(a + "v_scale");
                mtp_ks_ = std::max(mtp_ks_, L.fa.ks);
                mtp_vs_ = std::max(mtp_vs_, L.fa.vs);
            } else {
                kv_fp8_ = false;  // no calibrated scales: keep a bf16 cache
            }
        } else {
            const std::string a = pre + "linear_attn.";
            const int kb = p_.k_heads.b * 128, ke = p_.k_heads.e * 128;
            const int vb = p_.v_heads.b * 128, ve = p_.v_heads.e * 128;
            L.dn.in = load_w4(st,
                              {{a + "in_proj_qkv", kb, ke},
                               {a + "in_proj_qkv", kd + kb, kd + ke},
                               {a + "in_proj_qkv", 2 * kd + vb, 2 * kd + ve},
                               {a + "in_proj_z", vb, ve}},
                              -1, -1);
            L.dn.ab = load_w16(st,
                               {{a + "in_proj_b.weight", p_.v_heads.b, p_.v_heads.e},
                                {a + "in_proj_a.weight", p_.v_heads.b, p_.v_heads.e}},
                               8);
            if (gemv_h16_on()) {  // fp16 copy read from the XH activations by the decode step
                L.dn.abh = L.dn.ab;
                L.dn.abh.w = (uint4*)dalloc(w16_bytes(L.dn.ab.n, L.dn.ab.k), true);
                w16_to_h(L.dn.ab, (uint4*)L.dn.abh.w, stream);
                CUDA_CHECK(cudaStreamSynchronize(stream));
            }
            L.dn.out = load_w4(st, {{a + "out_proj", 0, H}}, vb, ve);
            // conv weight [10240][1][4] -> local channels in the same q|k|v order
            {
                const TensorInfo& t = st.get(a + "conv1d.weight");
                bf16* d = (bf16*)dalloc((size_t)dn_ch_ * 4 * 2, true);
                const int segs[3][2] = {{kb, ke}, {kd + kb, kd + ke}, {2 * kd + vb, 2 * kd + ve}};
                int off = 0;
                for (auto& sg : segs) {
                    const int nrow = sg[1] - sg[0];
                    CUDA_CHECK(cudaMemcpy(d + (size_t)off * 4, t.data + (size_t)sg[0] * 8,
                                          (size_t)nrow * 8, cudaMemcpyHostToDevice));
                    off += nrow;
                }
                L.dn.conv_w = d;
            }
            L.dn.A = load_f32(st, a + "A_log", p_.v_heads.b, p_.v_heads.e, true);
            L.dn.dt_bias = load_f32(st, a + "dt_bias", p_.v_heads.b, p_.v_heads.e, false);
            L.dn.norm_w = load_bf16(st, a + "norm.weight", -1, -1);
            dn_slot_floats_ = (size_t)dsh_.nv * 128 * 128;
            // lazy state writes of single-row steps (dn_lazy_of): state written every T-th step
            dsh_.lazy_t = getenv("QE_DN_LAZY") ? atoi(getenv("QE_DN_LAZY")) : 3;
        }
        L.gate_up = load_w4(st,
                            {{pre + "mlp.gate_proj", p_.inter.b, p_.inter.e},
                             {pre + "mlp.up_proj", p_.inter.b, p_.inter.e}},
                            -1, -1, glu_);
        L.down = load_w4(st, {{pre + "mlp.down_proj", 0, H}}, p_.inter.b, p_.inter.e);
    }
    final_norm_ = load_bf16(st, "model.language_model.norm.weight", -1, -1);
    if (mtp_st) {
        load_mtp(*mtp_st);
        const char* env = getenv("QE_DRAFT_HEAD_W4");
        const bool w4 = env ? atoi(env) != 0 : true;
        if (draft_ids.empty()) {
            draft_head_ = load_lin(st, {{"lm_head.weight", p_.vocab.b, p_.vocab.e}}, 16, w4);
            draft_v_ = p_.vocab.size();
        } else {
            // gather the selected rows on the host, then quantize like any draft weight
            const TensorInfo& t = st.get("lm_head.weight");
            const int K = t.shape[1], n = draft_ids.size();
            CHECK(n % 16 == 0, "draft vocab share %d", n);
            std::vector<bf16> rows((size_t)n * K);
            for (int i = 0; i < n; i++)
                memcpy(&rows[(size_t)i * K], t.data + (size_t)draft_ids[i] * K * 2, (size_t)K * 2);
            bf16* tmp = (bf16*)g_scratch_q.get((size_t)n * K * 2);
            CUDA_CHECK(cudaMemcpy(tmp, rows.data(), rows.size() * 2, cudaMemcpyHostToDevice));
            draft_head_.is4 = w4;
            if (w4) {
                int32_t* q = (int32_t*)g_scratch_s.get((size_t)n * K / 2 + (size_t)n * (K / 128) * 2);
                bf16* sc = (bf16*)((char*)q + (size_t)n * K / 2);
                quantize_w4_rtn(tmp, n, K, q, sc, stream);
                draft_head_.q.n = n;
                draft_head_.q.k = K;
                uint4* dq = (uint4*)dalloc(w4_qbytes(n, K), true);
                bf16* ds = (bf16*)dalloc(w4_sbytes(n, K), true);
                repack_w4(q, sc, n, K, dq, ds, stream);
                draft_head_.q.q = dq;
                draft_head_.q.s = ds;
            } else {
                draft_head_.f.n = n;
                draft_head_.f.k = K;
                uint4* d = (uint4*)dalloc(w16_bytes(n, K), true);
                repack_w16(tmp, n, K, d, stream);
                draft_head_.f.w = d;
            }
            CUDA_CHECK(cudaStreamSynchronize(stream));
            draft_map_ = (int*)dalloc(n * 4, true);
            CUDA_CHECK(cudaMemcpy(draft_map_, draft_ids.data(), n * 4, cudaMemcpyHostToDevice));
            draft_v_ = n;
        }
    }
    {
        // INT8 (group 128, like GGUF Q8_0) halves the largest per-token read; QE_HEAD_W8=0 keeps bf16
        const char* e = getenv("QE_HEAD_W8");
        const std::vector<RowSpec> rows = {{"lm_head.weight", p_.vocab.b, p_.vocab.e}};
        if (e && atoi(e) == 0) {
            lm_head_.f = load_w16(st, rows, 16);
        } else {
            int n, k;
            bf16* tmp = gather_bf16(st, rows, 16, -1, -1, &n, &k);
            lm_head_.is8 = true;
            lm_head_.q8.n = n;
            lm_head_.q8.k = k;
            uint4* q = (uint4*)dalloc(w8_qbytes(n, k), true);
            bf16* sc = (bf16*)dalloc(w4_sbytes(n, k), true);
            quantize_w8(tmp, n, k, q, sc, stream);
            CUDA_CHECK(cudaStreamSynchronize(stream));
            lm_head_.q8.q = q;
            lm_head_.q8.s = sc;
        }
    }

    {
        std::vector<float> f(32);
        for (int i = 0; i < 32; i++)
            f[i] = (float)(1.0 / std::pow(c_.rope_theta, (double)(2 * i) / c_.rot_dim));
        inv_freq_ = (float*)dalloc(32 * 4, true);
        CUDA_CHECK(cudaMemcpy(inv_freq_, f.data(), 32 * 4, cudaMemcpyHostToDevice));
        rope3_ = (int*)dalloc((2 + 3 * (size_t)kChunk) * 4, true);
        CUDA_CHECK(cudaMemset(rope3_, 0, 8));
        ash_.rpos3 = rope3_;
    }

    // --- activations (sized for a prefill chunk; decode uses the first rows) ---
    const int M = kChunk;
    proj_ld_ = std::max({ash_.nq * 512 + 2 * ash_.nkv * 256, dn_ch_ + dsh_.nv * 128, 2 * I});
    const int act_w = std::max({ow_, dsh_.nv * 128, I});
    resid_ = (float*)dalloc((size_t)M * H * 4, false);
    partial_ = (float*)dalloc((size_t)M * H * 4, false);
    proj_ = (float*)dalloc((size_t)M * proj_ld_ * 4, false);
    ab_ = (float*)dalloc((size_t)kAbSplits * M * 2 * dsh_.ab_pad * 4, false);
    conv_out_ = (float*)dalloc((size_t)M * dn_ch_ * 4, false);
    dno_ = (float*)dalloc((size_t)M * dsh_.nv * 128 * 4, false);
    dn_qk_ = (float*)dalloc((size_t)M * 2 * dsh_.nk * 128 * 4, false);
    dn_gb_ = (float*)dalloc((size_t)M * 2 * dsh_.nv * 4, false);
    attn_work_ = (float*)dalloc(attn_work_floats(ash_) * kMaxSegs * 4, false);  // per batched sequence
    for (auto& L : layers_)
        if (!L.full) {
            spec_cv_.push_back((float*)dalloc((size_t)kMaxBatchRows * dn_ch_ * 4, false));
            spec_qk_.push_back((float*)dalloc((size_t)kMaxBatchRows * 2 * dsh_.nk * 128 * 4, false));
            spec_gb_.push_back((float*)dalloc((size_t)kMaxBatchRows * 2 * dsh_.nv * 4, false));
        }
    CHECK((int)spec_cv_.size() <= kMaxLinLayers, "too many delta-net layers");
    pattn_work_floats_ = attention_prefill_work_floats(5 * 1024, ash_.nq);  // splits * rows <= 5120
    pattn_work_ = (float*)dalloc(pattn_work_floats_ * 4, false);
    pattn_ksc_ = (uint2*)dalloc((size_t)ash_.nkv * kPattnKscRows * sizeof(uint2), false);
    ash_.ksc = pattn_ksc_;
    ash_.ksc_ld = kPattnKscRows;
    pattn_kmu_ = (float*)dalloc((size_t)ash_.nkv * 256 * sizeof(float), false);
    ash_.kmu = pattn_kmu_;
    {  // windowed precomputed K / V of the prefill attention live in the delta-net buffers, which are idle
       // during a full-attention layer: K int8 [nkv][w][256] in dno_, V fp16 [nkv][w][256] in conv_out_
        const size_t kb = (size_t)M * dsh_.nv * 128 * 4, vb = (size_t)M * dn_ch_ * 4;
        int w = (int)std::min(kb / ((size_t)ash_.nkv * 256), vb / ((size_t)ash_.nkv * 512)) / 64 * 64;
        static const int wmax = getenv("QE_PATTN_KVW") ? atoi(getenv("QE_PATTN_KVW")) : 1 << 30;
        w = std::min(w, wmax);
        if (w >= 4096) {
            ash_.kvq = (uint8_t*)dno_;
            ash_.kvh = (uint4*)conv_out_;
            ash_.kvw = w;
        }
    }
    {  // persistent windowed prefill attention (attn_pkp_kernel): two K / V window buffers per kv head and the
       // Q fragments, packed first-fit-decreasing into the delta-net buffers that are idle during a
       // full-attention layer (conv_out_, dno_, dn_qk_); the largest window (multiple of 1024 rows) that fits
        struct Reg { char* p; size_t cap, used; };
        const int nkv = ash_.nkv, qt_max = kChunk / 128;
        const size_t qf_b = (size_t)qt_max * ash_.nq * 8 * 8 * 32 * 16, qs_b = (size_t)qt_max * ash_.nq * 128 * 4;
        static const int wenv = getenv("QE_PATTN_KVW2") ? atoi(getenv("QE_PATTN_KVW2")) : 32768;
        for (int w = wenv / 1024 * 1024; w >= 4096; w -= 1024) {
            std::vector<Reg> regs = {{(char*)conv_out_, (size_t)M * dn_ch_ * 4, 0},
                                     {(char*)dno_, (size_t)M * dsh_.nv * 128 * 4, 0},
                                     {(char*)dn_qk_, (size_t)M * 2 * dsh_.nk * 128 * 4, 0}};
            // pieces: (bytes, kind, buffer, kv): kind 0 = V, 1 = K, 2 = Q fragments, 3 = Q scales
            struct Piece { size_t b; int kind, buf, kv; char* at; };
            std::vector<Piece> pcs;
            for (int b = 0; b < 2; b++)
                for (int j = 0; j < nkv; j++) {
                    pcs.push_back({(size_t)w * 512, 0, b, j, nullptr});
                    pcs.push_back({(size_t)w * 256, 1, b, j, nullptr});
                }
            pcs.push_back({qf_b, 2, 0, 0, nullptr});
            pcs.push_back({qs_b, 3, 0, 0, nullptr});
            std::stable_sort(pcs.begin(), pcs.end(), [](const Piece& x, const Piece& y) { return x.b > y.b; });
            bool ok = true;
            for (Piece& pc : pcs) {
                for (Reg& r : regs) {
                    const size_t off = (r.used + 255) / 256 * 256;
                    if (off + pc.b <= r.cap) { pc.at = r.p + off; r.used = off + pc.b; break; }
                }
                ok &= pc.at != nullptr;
            }
            if (!ok) continue;
            pscr_ = PattnScratch{};
            pscr_.w = w;
            for (const Piece& pc : pcs) {
                if (pc.kind == 0) pscr_.vh[pc.buf][pc.kv] = (uint4*)pc.at;
                else if (pc.kind == 1) pscr_.kq[pc.buf][pc.kv] = (uint8_t*)pc.at;
                else if (pc.kind == 2) pscr_.qf = (uint4*)pc.at;
                else pscr_.qsq = (float*)pc.at;
            }
            pscr_.qt_max = qt_max;
            pscr_.ctr = (int*)dalloc(kPkpCtrInts * sizeof(int), false);
            ash_.pscr = &pscr_;
            if (rank == 0) fprintf(stderr, "[pattn] persistent windows: 2 x %d keys per kv head\n", w);
            break;
        }
    }
    xn_ = (bf16*)dalloc((size_t)M * H * 2, false);
    act_ = (bf16*)dalloc((size_t)M * act_w * 2, false);
    h16_ = gemv_h16_on();
    if (h16_) {
        xh_act_ld_ = act_w;
        xh_xn_ = (__half*)dalloc((size_t)kMaxBatchRows * H * 2, false);
        xh_act_ = (__half*)dalloc((size_t)kMaxBatchRows * act_w * 2, false);
        xh_tmp_ = (__half*)dalloc((size_t)kMaxBatchRows * std::max({H, act_w, 2 * H}) * 2, false);
    }
    qbuf_ = (bf16*)dalloc((size_t)M * ash_.nq * 256 * 2, false);
    plan_bytes();  // kv_fp8_ / has_mtp are known now
    alloc_xchg();
    logits_ = (float*)dalloc((size_t)kMaxBatchRows * p_.vocab.size() * 4, false);
    pairs_ = (float*)dalloc(kMaxBatchRows * 2 * 4, false);
    pos8_ = (int*)dalloc(4, false);
    pairs_all_ = (float*)dalloc(ngpu_ * kMaxBatchRows * 2 * 4, false);
    topk_ = (float*)dalloc(kMaxBatchRows * kTopK * 2 * 4, false);
    topk_all_ = (float*)dalloc(ngpu_ * kMaxBatchRows * kTopK * 2 * 4, false);
    pos_mb_ = (int*)dalloc(4, false);
    bout_ = (int*)dalloc(kMaxBatchRows * 4, false);
    bmout_ = (int*)dalloc(kMaxBatchRows * 4, false);
    mtp_in_ = (bf16*)dalloc((size_t)M * 2 * H * 2, false);
    mtp_h_ = (bf16*)dalloc((size_t)M * H * 2, false);
    mtp_resid_ = (float*)dalloc((size_t)M * H * 4, false);
    ctr_.seq = (unsigned*)dalloc(4, false);
    ctr_.done = (unsigned*)dalloc(8, false);
    ctr_.go = (unsigned*)dalloc(4, false);
    ctr_.bar = (unsigned*)dalloc(8, false);
    ctr_.ss = (float*)dalloc(kCommBlocks * kMaxBatchRows * 4, false);
    ctr_.epoch = (unsigned*)dalloc(4, false);
    CUDA_CHECK(cudaMemset(ctr_.epoch, 0, 4));
    {  // copy-engine AR: device ring of flag values v[i] = i (16-bit sequence numbers, copied by DMA)
        std::vector<unsigned> v(kCeVals);
        for (unsigned i = 0; i < kCeVals; i++) v[i] = i;
        ce_vals_ = (unsigned*)dalloc(kCeVals * sizeof(unsigned), false);
        CUDA_CHECK(cudaMemcpy(ce_vals_, v.data(), kCeVals * sizeof(unsigned), cudaMemcpyHostToDevice));
    }
    ready_ = (unsigned*)dalloc(4096 * 4, false);
    CUDA_CHECK(cudaMemset(ready_, 0, 4096 * 4));
    if (const char* e = getenv("QE_LL")) use_ll_ = atoi(e) != 0;
    CUDA_CHECK(cudaMemset(ctr_.bar, 0, 8));
    CUDA_CHECK(cudaMemset(ctr_.seq, 0, 4));
    CUDA_CHECK(cudaMemset(ctr_.done, 0, 8));
    CUDA_CHECK(cudaMemset(ctr_.go, 0, 4));
    CUDA_CHECK(cudaDeviceSynchronize());
}

void Worker::reset() {
    CUDA_CHECK(cudaSetDevice(device));
    CUDA_CHECK(cudaMemsetAsync(dn_cur_, 0, 4, stream));
    CUDA_CHECK(cudaMemsetAsync(dn_save_, 0, 4, stream));
    if (dn_pn_) CUDA_CHECK(cudaMemsetAsync(dn_pn_, 0, 8, stream));
    for (auto& L : layers_) {
        if (L.full) continue;
        CUDA_CHECK(cudaMemsetAsync(L.dn.S, 0, dn_slot_floats_ * 4, stream));  // slot 0
        CUDA_CHECK(cudaMemsetAsync(L.dn.hist, 0, (size_t)kConvRing * dn_ch_ * 2, stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

size_t Worker::snapshot_bytes() const {
    return (size_t)c_.n_lin() * (dn_slot_floats_ * 4 + (size_t)kConvRing * dn_ch_ * 2);
}

void Worker::snapshot_save(void* dst) {
    CUDA_CHECK(cudaSetDevice(device));
    if (dn_pn_) {  // the state after the last step's accepted rows
        dn_flush_cur(stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    int cur;
    CUDA_CHECK(cudaMemcpy(&cur, dn_cur_, 4, cudaMemcpyDeviceToHost));
    char* d = (char*)dst;
    for (auto& L : layers_) {
        if (L.full) continue;
        CUDA_CHECK(cudaMemcpyAsync(d, L.dn.S + (size_t)cur * dn_slot_floats_, dn_slot_floats_ * 4,
                                   cudaMemcpyDefault, stream));
        d += dn_slot_floats_ * 4;
        CUDA_CHECK(cudaMemcpyAsync(d, L.dn.hist, (size_t)kConvRing * dn_ch_ * 2,
                                   cudaMemcpyDefault, stream));
        d += (size_t)kConvRing * dn_ch_ * 2;
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

void Worker::snapshot_restore(const void* src) {
    CUDA_CHECK(cudaSetDevice(device));
    CUDA_CHECK(cudaMemsetAsync(dn_cur_, 0, 4, stream));
    if (dn_pn_) CUDA_CHECK(cudaMemsetAsync(dn_pn_, 0, 8, stream));
    const char* d = (const char*)src;
    for (auto& L : layers_) {
        if (L.full) continue;
        CUDA_CHECK(cudaMemcpyAsync(L.dn.S, d, dn_slot_floats_ * 4, cudaMemcpyDefault, stream));
        d += dn_slot_floats_ * 4;
        CUDA_CHECK(cudaMemcpyAsync(L.dn.hist, d, (size_t)kConvRing * dn_ch_ * 2,
                                   cudaMemcpyDefault, stream));
        d += (size_t)kConvRing * dn_ch_ * 2;
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

void Worker::set_sampling(const SampleParams& p, bool on) {
    CUDA_CHECK(cudaSetDevice(device));
    SeqSlot& sl = slots_.at(cur_slot_);
    CUDA_CHECK(cudaMemcpy(sl.sp, &p, sizeof(p), cudaMemcpyHostToDevice));
    sl.sampling = sampling = on;
}

void Worker::set_constrained(int i, bool on) {
    slots_.at(i).constrained = on;
    if (i == cur_slot_) constrained = on;
}

void Worker::upload_mask(int i, const unsigned* host, int rows) {
    CUDA_CHECK(cudaSetDevice(device));
    CUDA_CHECK(cudaMemcpyAsync(slots_.at(i).mask, host, mask_words() * 4 * rows,
                               cudaMemcpyHostToDevice, stream));
}

void Worker::upload_drafts(int i, const int* host, int n) {
    if (n <= 0) return;
    CUDA_CHECK(cudaSetDevice(device));
    CUDA_CHECK(cudaMemcpyAsync(slots_.at(i).spec.tokens + 1, host, n * 4, cudaMemcpyHostToDevice, stream));
}

void Worker::apply_head_masks() {
    for (const HeadMask& h : head_masks_)
        mask_logits(logits_, p_.vocab.size(), p_.vocab.b, h.bits, h.r0, h.rows, h.stride, stream);
    head_masks_.clear();
}

// Instantiate a captured graph; when the GPU is out of memory, drop every cached graph (they
// are rebuilt on demand) and retry instead of failing the request.
void Worker::instantiate(cudaGraphExec_t* ge, cudaGraph_t g) {
    cudaError_t err = cudaGraphInstantiate(ge, g, 0);
    if (err == cudaErrorMemoryAllocation) {
        cudaGetLastError();
        graphs_.clear();
        spec_graphs_.clear();
        batch_graphs_.clear();
        fprintf(stderr, "GPU%d: out of memory instantiating a CUDA graph; graph caches cleared\n", device);
        err = cudaGraphInstantiate(ge, g, 0);
    }
    CUDA_CHECK(err);
}

// debug (QE_DN_DUMP=1): delta-net state slots of the current slot before /
// after a small plain pass, as checksums
void Worker::dn_dump(const char* tag, int m) {
    CUDA_CHECK(cudaStreamSynchronize(stream));
    int cur = 0, pn[2] = {0, 0};
    CUDA_CHECK(cudaMemcpy(&cur, dn_cur_, 4, cudaMemcpyDeviceToHost));
    if (dn_pn_) CUDA_CHECK(cudaMemcpy(pn, dn_pn_, 8, cudaMemcpyDeviceToHost));
    std::vector<uint32_t> h(dn_slot_floats_);
    unsigned long long cs[2] = {0, 0};
    for (auto& L : layers_) {
        if (L.full) continue;
        for (int k = 0; k < 2; k++) {
            CUDA_CHECK(cudaMemcpy(h.data(), L.dn.S + (size_t)k * dn_slot_floats_, dn_slot_floats_ * 4, cudaMemcpyDeviceToHost));
            for (uint32_t x : h) cs[k] = cs[k] * 1000003ull + x;
        }
    }
    int pos = 0;
    CUDA_CHECK(cudaMemcpy(&pos, pos_, 4, cudaMemcpyDeviceToHost));
    fprintf(stderr, "dn_dump r%d %s slot %d m %d pos %d cur %d pn %d/%d S0 %016llx S1 %016llx\n", rank, tag, cur_slot_,
            m, pos, cur, pn[0], pn[1], cs[0], cs[1]);
}

void Worker::run(int m, bool logits) {
    static const bool dump = getenv("QE_DN_DUMP") && atoi(getenv("QE_DN_DUMP"));
    if (dump && rank == 0) dn_dump("before", m);
    run_(m, logits);
    if (dump && rank == 0) dn_dump("after", m);
}

void Worker::run_(int m, bool logits) {
    if (debug_hidden) return forward(m, logits);
    auto key = std::make_tuple(m, logits, mode(), cur_slot_);
    cudaGraphExec_t ge = graphs_.get(key);
    if (!ge) {
        cudaGraph_t g;
        CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
        forward(m, logits);
        CUDA_CHECK(cudaStreamEndCapture(stream, &g));
        instantiate(&ge, g);
        CUDA_CHECK(cudaGraphDestroy(g));
        graphs_.put(key, ge);
    }
    CUDA_CHECK(cudaGraphLaunch(ge, stream));
}

void Worker::forward(int m, bool logits, bool spec) {
    const int H = c_.hidden, I = p_.inter.size();
    const float eps = c_.eps;
    cudaStream_t st = stream;
    comm_begin_unit(ctr_, st);
    ll_idx_ = 0;
    xidx_ = 0;
    if (!spec) dn_flush_cur(st);  // deferred commit of the last verification step
    embed_gather(embed_, tokens_, m, H, resid_, st);
    const int qw = ow_, vw = dsh_.nv * 128;
    __half* xhx = m <= rows_cap_ ? xh_xn_ : nullptr;   // XH copies (null: fallback packing / off)
    __half* xha = m <= rows_cap_ ? xh_act_ : nullptr;
    rmsnorm(resid_, layers_[0].in_norm, m, H, eps, xn_, st, xhx);
    if (p_.pos_split) rope_table(ash_, pos_, m, inv_freq_, rtab_, st);  // (only rdelta / rpos3 of the shape matter)
    int li = 0;  // delta-net layer index
    bool early_in = false;  // this layer's input GEMV already launched by allreduce_early
    for (int i = 0; i < c_.layers; i++) {
        Layer& L = layers_[i];
        // xn_ holds input_layernorm(resid) here (fused into the previous all-reduce)
        if (L.full) {
            if (!early_in) mm4(L.fa.qkv, xn_, H, proj_, proj_ld_, m, st, xhx, H);
            AttnShape a = layer_shape(&L);
            a.pf_p0 = pf_p0_;
            xchg_fill(a);
            attn_prep(proj_, proj_ld_, a, L.fa.q_norm, L.fa.k_norm, eps, inv_freq_, pos_, m,
                      qbuf_, L.fa.kc, L.fa.vc, st);
            verify_ = spec;
            attend(a, qbuf_, L.fa.kc, L.fa.vc, pos_, m, nullptr, nullptr, xha);
            verify_ = false;
            // prefill attention (m > 8) does not write XH: fallback packing
            mm4(L.fa.o, act_, qw, partial_, H, m, st, (m <= kMaxGemvTokens || (spec && m <= kPendRows)) ? xha : nullptr, qw);
        } else {
            const size_t abs = (size_t)kChunk * 2 * dsh_.ab_pad;
            static const bool ab_par = !getenv("QE_AB_PAR") || atoi(getenv("QE_AB_PAR"));
            if (early_in) {  // dn_in runs on pf_stream_ (allreduce_early): ab here, then join
                gemv_w16h(L.dn.abh, xhx, H, ab_, 2 * dsh_.ab_pad, m, st, kAbSplits, abs);
                early_join();
            } else if (m <= kMaxGemvTokens && ab_par) {  // the small gate projection beside in_proj
                cudaEvent_t fork = ev(), join = ev();
                CUDA_CHECK(cudaEventRecord(fork, st));
                CUDA_CHECK(cudaStreamWaitEvent(pf_stream_, fork, 0));
                if (xhx) gemv_w16h(L.dn.abh, xhx, H, ab_, 2 * dsh_.ab_pad, m, pf_stream_, kAbSplits, abs);
                else gemv_w16(L.dn.ab, xn_, H, ab_, 2 * dsh_.ab_pad, m, pf_stream_, kAbSplits, abs);
                CUDA_CHECK(cudaEventRecord(join, pf_stream_));
                mm4(L.dn.in, xn_, H, proj_, proj_ld_, m, st, xhx, H);
                CUDA_CHECK(cudaStreamWaitEvent(st, join, 0));
            } else {
                mm4(L.dn.in, xn_, H, proj_, proj_ld_, m, st, xhx, H);
                if (xhx)  // XH path (m <= rows_cap_): bf16 xn_ is not written for this norm
                    gemv_w16h(L.dn.abh, xhx, H, ab_, 2 * dsh_.ab_pad, m, st, kAbSplits, abs);
                else if (m <= kMaxGemvTokens)
                    gemv_w16(L.dn.ab, xn_, H, ab_, 2 * dsh_.ab_pad, m, st, kAbSplits, abs);
                else
                    gemm_w16(L.dn.ab, xn_, H, ab_, 2 * dsh_.ab_pad, m, st, kAbSplits, abs);
            }
            // speculative verification keeps each layer's inputs for dn_replay
            float* cv = spec ? spec_cv_[li] : conv_out_;
            float* qk = spec ? spec_qk_[li] : dn_qk_;
            float* gb = spec ? spec_gb_[li] : dn_gb_;
            const int* pn = spec && dn_defer1_on() ? dn_pn_ : nullptr;  // deferred commit
            const float* pend = pn ? dn_pend_ + li * dn_pend_layer_floats() : nullptr;
            li++;
            if (dn_fuse_on() && m <= 13) {  // dn_convqk: ring slots of the step must not overlap
                dn_convqk(proj_, proj_ld_, dn_ch_, L.dn.conv_w, L.dn.hist, pos_, m, cv, dsh_, ab_,
                          L.dn.A, L.dn.dt_bias, qk, gb, st);
                dn_core_only(cv, dn_ch_, dsh_, L.dn.S, dn_slot_floats_, dn_cur_, m, spec, qk, gb,
                             dno_, st, pn, pend);
            } else {
                dn_conv(proj_, proj_ld_, dn_ch_, L.dn.conv_w, L.dn.hist, pos_, m, cv, st);
                dn_core(cv, dn_ch_, dsh_, ab_, L.dn.A, L.dn.dt_bias, L.dn.S, dn_slot_floats_,
                        dn_cur_, m, spec, qk, gb, dno_, st, 0, pn, pend);
            }
            dn_out(dno_, proj_ + dn_ch_, proj_ld_, L.dn.norm_w, dsh_.nv, eps, m, act_, st, xha);
            mm4(L.dn.out, act_, vw, partial_, H, m, st, xha, vw);
        }
        if (L.gate_up.glu && allreduce_early(m, L.post_norm, L.gate_up, reinterpret_cast<float*>(act_), I, xha, I, true, 1)) {
        } else if (allreduce_pf(resid_, m, L.post_norm, xn_, &L.gate_up), L.gate_up.glu) {
            // fused silu(gate) * up epilogue -> bf16 act_ (+ XH copy)
            mm4(L.gate_up, xn_, H, reinterpret_cast<float*>(act_), I, m, st, xhx, H, xha, I);
        } else {
            mm4(L.gate_up, xn_, H, proj_, proj_ld_, m, st, xhx, H);
            silu_mul(proj_, proj_ld_, I, m, act_, st, xha);
        }
        mm4(L.down, act_, I, partial_, H, m, st, xha, I);
        const bf16* next_norm = i + 1 < c_.layers ? layers_[i + 1].in_norm : final_norm_;
        const W4* nxt = nullptr;
        if (i + 1 < c_.layers) nxt = layers_[i + 1].full ? &layers_[i + 1].fa.qkv : &layers_[i + 1].dn.in;
        early_in = nxt && !debug_hidden && allreduce_early(m, next_norm, *nxt, proj_, proj_ld_, nullptr, 0,
                                                           layers_[i + 1].full, layers_[i + 1].full ? 2 : 4);
        if (!early_in) allreduce_pf(resid_, m, next_norm, xn_, nxt);
        if (debug_hidden) {
            CUDA_CHECK(cudaMemcpyAsync(debug_hidden + (size_t)i * m * H, resid_,
                                       (size_t)m * H * 4, cudaMemcpyDeviceToHost, st));
        }
    }
    if (!spec) dn_commit(dn_cur_, nullptr, m, st);
    if (logits) {
        // prompt chunks only need the last row; spec verification samples in run_spec
        // (a lookup verification of up to 16 rows: every row)
        const bool all = m <= kMaxGemvTokens || (spec && m <= kPendRows);
        const bf16* x = all ? xn_ : xn_ + (size_t)(m - 1) * H;
        const int rows = all ? m : 1;
        int* out = all ? out_tok_ : out_tok_ + m - 1;
        mask_current(rows, spec);
        if (!sampling) {
            head(lm_head_, x, rows, out);
        } else {
            head_topk(lm_head_, x, rows);
            if (!spec) sample_rows(topk_all_, ngpu_, rows, sp_, out, st);
        }
    }
}

// the bf16 copy of a fused norm output is needed unless the H16 kernels consume the XH copy instead:
// `next` is the W4 matrix fed by this norm (the DeltaNet `ab` projection reads XH too); the final norm
// (next == null) feeds lm_head in bf16
bf16* Worker::xn_bf16(bf16* xn, int m, const W4* next) {
    if (!h16_ || m > rows_cap_ || xn != xn_ || !next || dec_gemm_rows(m)) return xn;
    return nullptr;
}

// QE_EARLY_GEMV: bit mask of the consumer sites (1 gate_up, 2 next full-attention qkv, 4 next DeltaNet in);
// 1 alone is read as "all" (7)
static int early_gemv_mask() {
    static const int v = [] {
        const int e = getenv("QE_EARLY_GEMV") ? atoi(getenv("QE_EARLY_GEMV")) : 0;
        return e == 1 ? 7 : e;
    }();
    return v;
}

bool Worker::allreduce_early(int m, const bf16* norm, const W4& w, float* y, int ldy, __half* yh, int ldyh,
                             bool join, int site) {
    static const int ll16 = getenv("QE_LL16") ? atoi(getenv("QE_LL16")) : 1;
    static const int ranks = getenv("QE_EARLY_RANKS") ? atoi(getenv("QE_EARLY_RANKS")) : 7;  // GPUs using it
    if (!(early_gemv_mask() & site) || !((ranks >> rank) & 1) || !h16_ || ngpu_ < 2 || m > 8 || !use_ll_ || ll16 <= 0 || m < ll16 || !xh_xn_)
        return false;
    static const int depth = getenv("QE_EARLY_DEPTH") ? atoi(getenv("QE_EARLY_DEPTH")) : -1;
    static const bool fused = getenv("QE_EARLY_FUSED") ? atoi(getenv("QE_EARLY_FUSED")) != 0 : true;
    if (fused && !comm_.p2p) return false;
    const int ll = next_ll(m);
    if (ll < 0) return false;
    if (fused) {  // the all-reduce runs inside the consumer GEMV (blocks 0 .. m-1): one kernel, one stream
        FusedAR fa{};
        fa.resid = resid_;
        fa.partial = partial_;
        fa.h = c_.hidden;
        fa.rank = rank;
        fa.idx = ll;
        fa.m = m;
        fa.c = comm_;
        fa.epoch = ctr_.epoch;
        fa.norm_w = norm;
        fa.eps = c_.eps;
        fa.xh = xh_xn_;
        fa.ready = ready_;
        gemv_w4h_early(w, xh_xn_, c_.hidden, y, ldy, m, depth >= 0 ? depth : 2, ready_, 0, stream, ctr_.epoch, ll,
                       yh, ldyh, &fa);
        early_join_ev_ = nullptr;
        return true;
    }
    cudaEvent_t fork = ev();
    CUDA_CHECK(cudaEventRecord(fork, stream));
    CUDA_CHECK(cudaStreamWaitEvent(pf_stream_, fork, 0));
    CUDA_CHECK(cudaStreamWaitEvent(comm_stream_, fork, 0));
    CommCounters c2 = ctr_;
    c2.ready = ready_;
    // the all-reduce on the high-priority stream (its blocks must be resident before the waiting GEMV
    // fills the SMs: block dispatch prefers higher-priority streams), then the waiting GEMV
    allreduce_add(resid_, partial_, m, c_.hidden, comm_, rank, c2, comm_stream_, norm, c_.eps, nullptr, ll, false,
                  xh_xn_);
    cudaEvent_t ar_done = ev();
    CUDA_CHECK(cudaEventRecord(ar_done, comm_stream_));
    gemv_w4h_early(w, xh_xn_, c_.hidden, y, ldy, m, depth >= 0 ? depth : 2, ready_, 0,
                   pf_stream_, ctr_.epoch, ll, yh, ldyh);
    early_join_ev_ = ev();
    CUDA_CHECK(cudaEventRecord(early_join_ev_, pf_stream_));
    CUDA_CHECK(cudaStreamWaitEvent(stream, ar_done, 0));  // resid_ / xh_xn_ for the main stream
    if (join) early_join();
    return true;
}

void Worker::early_join() {
    if (early_join_ev_) CUDA_CHECK(cudaStreamWaitEvent(stream, early_join_ev_, 0));
}

void Worker::allreduce_pf(float* resid, int m, const bf16* norm, bf16* xn, const W4* next) {
    const int ll = next_ll(m);
    if (!next || pf_bytes_ == 0 || m > rows_cap_) {
        allreduce_add(resid, partial_, m, c_.hidden, comm_, rank, ctr_, stream, norm, c_.eps,
                      xn_bf16(xn, m, next), ll, false, xn == xn_ && m <= rows_cap_ ? xh_xn_ : nullptr);
        return;
    }
    cudaEvent_t fork = ev(), join = ev();
    CUDA_CHECK(cudaEventRecord(fork, stream));
    CUDA_CHECK(cudaStreamWaitEvent(pf_stream_, fork, 0));
    prefetch_l2(*next, pf_bytes_, pf_stream_);
    CUDA_CHECK(cudaEventRecord(join, pf_stream_));
    allreduce_add(resid, partial_, m, c_.hidden, comm_, rank, ctr_, stream, norm, c_.eps,
                  xn_bf16(xn, m, next), ll, false, xn == xn_ && m <= rows_cap_ ? xh_xn_ : nullptr);
    CUDA_CHECK(cudaStreamWaitEvent(stream, join, 0));
}

// ---------------------------------------------------------------- batched decoding

void Worker::forward_batch(const std::vector<int>& slots, const std::vector<int>& r0,
                           const std::vector<int>& ms) {
    const int H = c_.hidden, I = p_.inter.size(), ld = proj_ld_;
    const int qw = ow_, qbw = ash_.nq * 256, vw = dsh_.nv * 128, ldab = 2 * dsh_.ab_pad;
    const float eps = c_.eps;
    cudaStream_t st = stream;
    RowsCap cap_guard(rows_cap_, kMaxBatchRows);  // batched step: up to 64 rows on the decode kernels
    int M = 0;
    for (int m : ms) M += m;
    comm_begin_unit(ctr_, st);
    ll_idx_ = 0;
    xidx_ = 0;
    for (size_t i = 0; i < slots.size(); i++)
        embed_gather(embed_, slots_[slots[i]].spec.tokens, ms[i], H, resid_ + (size_t)r0[i] * H, st);
    rmsnorm(resid_, layers_[0].in_norm, M, H, eps, xn_, st, xh_xn_);
    {  // RoPE table of all rows (positions / shifts per sequence)
        AttnSegs g{};
        g.n = slots.size();
        for (size_t i = 0; i < slots.size(); i++) {
            SeqSlot& sl = slots_[slots[i]];
            g.c[i].pos = sl.spec.pos;
            g.c[i].rdelta = sl.rdelta;
            g.c[i].m = ms[i];
            g.max_m = std::max(g.max_m, ms[i]);
        }
        if (p_.pos_split) rope_table_seg(g, ash_, inv_freq_, rtab_, st);
    }
    int f = 0, l = 0;
    for (int li = 0; li < c_.layers; li++) {
        Layer& L = layers_[li];
        if (L.full) {
            mm4(L.fa.qkv, xn_, H, proj_, ld, M, st, xh_xn_, H);
            AttnSegs g{};  // all sequences in one launch per kernel
            // (a lookup verification of > 8 rows: two segments, the second at position + 8 - every row's result
            // as in any decode step)
            for (size_t i = 0; i < slots.size(); i++) {
                SeqSlot& sl = slots_[slots[i]];
                for (int h0 = 0; h0 < ms[i]; h0 += kMaxGemvTokens) {
                    const int z = g.n++, rr = r0[i] + h0, mm = std::min(kMaxGemvTokens, ms[i] - h0);
                    CHECK(z < kMaxSegs, "attention segments");
                    g.c[z] = AttnSeg{proj_ + (size_t)rr * ld, qbuf_ + (size_t)rr * qbw, sl.kc[f],
                                     sl.vc[f], h0 ? sl.pos8 : sl.spec.pos, sl.rdelta, mm, sl.max_t,
                                     attn_work_ + attn_work_floats(ash_) * z, act_ + (size_t)rr * qw,
                                     xh_act_ ? xh_act_ + (size_t)rr * qw : nullptr,
                                     sl.split ? sl.kc2[f] : nullptr, sl.split ? sl.vc2[f] : nullptr, sl.split};
                    g.max_m = std::max(g.max_m, mm);
                }
            }
            AttnShape a = layer_shape(&L);
            xchg_fill(a);
            attention_seg(g, a, ld, L.fa.q_norm, L.fa.k_norm, eps, inv_freq_, st);
            f++;
            mm4(L.fa.o, act_, qw, partial_, H, M, st, xh_act_, qw);
        } else {
            {  // the small gate projection beside in_proj (side stream)
                cudaEvent_t fork = ev(), join = ev();
                CUDA_CHECK(cudaEventRecord(fork, st));
                CUDA_CHECK(cudaStreamWaitEvent(pf_stream_, fork, 0));
                if (h16_) gemv_w16h(L.dn.abh, xh_xn_, H, ab_, ldab, M, pf_stream_, kAbSplits, (size_t)kChunk * ldab);
                else gemv_w16(L.dn.ab, xn_, H, ab_, ldab, M, pf_stream_, kAbSplits, (size_t)kChunk * ldab);
                CUDA_CHECK(cudaEventRecord(join, pf_stream_));
                mm4(L.dn.in, xn_, H, proj_, ld, M, st, xh_xn_, H);
                CUDA_CHECK(cudaStreamWaitEvent(st, join, 0));
            }
            DnSegs g{};  // all sequences in one launch per kernel
            g.n = slots.size();
            for (size_t i = 0; i < slots.size(); i++) {
                SeqSlot& sl = slots_[slots[i]];
                g.r0[i] = r0[i];
                g.m[i] = ms[i];
                g.pos[i] = sl.spec.pos;
                g.hist[i] = sl.hist[l];
                g.S[i] = sl.S[l];
                g.cur[i] = sl.dn_cur;
                g.pn[i] = sl.dn_pn;
                g.pend[i] = sl.dn_pend ? sl.dn_pend + l * dn_pend_layer_floats() : nullptr;
                g.fin[i] = sl.dn_pn && ms[i] == 1 && dn_direct_on() && dn_lazy_t(dsh_) == 1;
            }
            if (dn_fuse_on()) {
                dn_convqk_seg(g, proj_, ld, dn_ch_, L.dn.conv_w, spec_cv_[l], dsh_, ab_, L.dn.A,
                              L.dn.dt_bias, spec_qk_[l], spec_gb_[l], st);
            } else {
                dn_conv_seg(g, proj_, ld, dn_ch_, L.dn.conv_w, spec_cv_[l], st);
                dn_qk_rows(spec_cv_[l], dn_ch_, dsh_, ab_, L.dn.A, L.dn.dt_bias, M, spec_qk_[l],
                           spec_gb_[l], st);
            }
            dn_core_seg(g, spec_cv_[l], dn_ch_, dsh_, spec_qk_[l], spec_gb_[l], dn_slot_floats_, dno_,
                        st);
            l++;
            dn_out(dno_, proj_ + dn_ch_, ld, L.dn.norm_w, dsh_.nv, eps, M, act_, st, xh_act_);
            mm4(L.dn.out, act_, vw, partial_, H, M, st, xh_act_, vw);
        }
        allreduce_pf(resid_, M, L.post_norm, xn_, &L.gate_up);
        if (L.gate_up.glu) {
            mm4(L.gate_up, xn_, H, reinterpret_cast<float*>(act_), I, M, st, xh_xn_, H, xh_act_, I);
        } else {
            mm4(L.gate_up, xn_, H, proj_, ld, M, st, xh_xn_, H);
            silu_mul(proj_, ld, I, M, act_, st, xh_act_);
        }
        mm4(L.down, act_, I, partial_, H, M, st, xh_act_, I);
        const bf16* next_norm = li + 1 < c_.layers ? layers_[li + 1].in_norm : final_norm_;
        const W4* nxt = nullptr;
        if (li + 1 < c_.layers)
            nxt = layers_[li + 1].full ? &layers_[li + 1].fa.qkv : &layers_[li + 1].dn.in;
        allreduce_pf(resid_, M, next_norm, xn_, nxt);
    }
    for (size_t i = 0; i < slots.size(); i++)  // grammar-constrained sequences: a mask per row
        if (slots_[slots[i]].constrained)
            head_masks_.push_back({r0[i], ms[i], slots_[slots[i]].mask, mask_words()});
    head(lm_head_, xn_, M, bout_);
    bool any_sampling = false;
    for (int sidx : slots) any_sampling |= slots_[sidx].sampling;
    if (any_sampling) {  // candidates for the sampled sequences (same logits)
        topk_local(logits_, p_.vocab.size(), M, p_.vocab.b, topk_, st);
        allgather_small(topk_, M * kTopK * 2, topk_all_, comm_, rank, ctr_, st);
    }
}

void Worker::mtp_batch(const std::vector<int>& slots, const std::vector<int>& r0,
                       const std::vector<int>& ms, const std::vector<const bf16*>& hsrc) {
    const int H = c_.hidden, I = p_.inter.size();
    const float eps = c_.eps;
    cudaStream_t st = stream;
    RowsCap cap_guard(rows_cap_, kMaxBatchRows);  // batched step: up to 64 rows on the decode kernels
    int M = 0;
    for (int m : ms) M += m;
    comm_begin_unit(ctr_, st);
    ll_idx_ = 0;
    xidx_ = 0;
    for (size_t i = 0; i < slots.size(); i++) {
        const SpecState& ss = slots_[slots[i]].spec;
        mtp_prep(embed_, ss.mtp_tok, hsrc[i], ss.hrow, mtp_.pre_e, mtp_.pre_h, eps, ms[i], H,
                 mtp_in_ + (size_t)r0[i] * 2 * H, st);
    }
    mm4(mtp_.fc, mtp_in_ + p_.fc_k.b, 2 * H, partial_, H, M, st);
    CUDA_CHECK(cudaMemsetAsync(mtp_resid_, 0, (size_t)M * H * 4, st));
    allreduce_add(mtp_resid_, partial_, M, H, comm_, rank, ctr_, st, mtp_.in_norm, eps,
                  h16_ && !dec_gemm_rows(M) ? nullptr : xn_, next_ll(M), false, xh_xn_);
    mm4(mtp_.qkv, xn_, H, proj_, proj_ld_, M, st, xh_xn_, H);
    const int qw = ow_, qbw = ash_.nq * 256;
    AttnSegs g{};
    for (size_t i = 0; i < slots.size(); i++) {
        SeqSlot& sl = slots_[slots[i]];
        for (int h0 = 0; h0 < ms[i]; h0 += kMaxGemvTokens) {  // > 8 rows: two segments (see forward_batch)
            const int z = g.n++, rr = r0[i] + h0, mm = std::min(kMaxGemvTokens, ms[i] - h0);
            CHECK(z < kMaxSegs, "attention segments");
            g.c[z] = AttnSeg{proj_ + (size_t)rr * proj_ld_, qbuf_ + (size_t)rr * qbw, sl.mkc, sl.mvc,
                             h0 ? sl.mpos8 : sl.spec.mtp_pos, sl.rdelta, mm, sl.max_t,
                             attn_work_ + attn_work_floats(ash_) * z, act_ + (size_t)rr * qw,
                             xh_act_ ? xh_act_ + (size_t)rr * qw : nullptr, sl.mkc2, sl.mvc2, sl.split};
            g.max_m = std::max(g.max_m, mm);
        }
    }
    AttnShape a = layer_shape(nullptr);
    if (p_.pos_split) rope_table_seg(g, a, inv_freq_, rtab_, st);
    xchg_fill(a);
    attention_seg(g, a, proj_ld_, mtp_.q_norm, mtp_.k_norm, eps, inv_freq_, st);
    mm4(mtp_.o, act_, qw, partial_, H, M, st, xh_act_, qw);
    allreduce_add(mtp_resid_, partial_, M, H, comm_, rank, ctr_, st, mtp_.post_norm, eps,
                  h16_ && !dec_gemm_rows(M) ? nullptr : xn_, next_ll(M), false, xh_xn_);
    mm4(mtp_.gate_up, xn_, H, proj_, proj_ld_, M, st, xh_xn_, H);
    silu_mul(proj_, proj_ld_, I, M, act_, st, xh_act_);
    mm4(mtp_.down, act_, I, partial_, H, M, st, xh_act_, I);
    allreduce_add(mtp_resid_, partial_, M, H, comm_, rank, ctr_, st, mtp_.norm, eps, xn_,
                  next_ll(M));
    CUDA_CHECK(cudaMemcpyAsync(mtp_h_, xn_, (size_t)M * H * 2, cudaMemcpyDeviceToDevice, st));
    head(draft_head_, xn_, M, bmout_, true);
}

// QE_DEC_GEMM: the prefill GEMM sizes its thread-local buffers and tunes a new (shape, token bucket) on first
// use, neither of which may happen inside a graph capture: one eager call of every decode weight at the
// largest batch (garbage in scratch buffers, before the first capture of a batched step with such rows)
void Worker::dec_gemm_warm(int) {
    const int H = c_.hidden, I = p_.inter.size();
    int t = kMaxBatchRows;
    cudaStream_t st = stream;
    auto run = [&](const W4& w, const bf16* x) {
        if (w.glu) gemm_w4(w, x, w.k, reinterpret_cast<float*>(act_), w.n / 2, t, st);
        else gemm_w4(w, x, w.k, w.k == H ? proj_ : partial_, w.k == H ? proj_ld_ : H, t, st);
    };
    // buffer sizes of a whole prefill chunk (no growth later), then the decode row counts (tuner buckets,
    // stream-K plans of 33..64 rows)
    for (int tt : {kChunk, kMaxBatchRows, kMaxGemvW4Tokens + 1}) {
    t = tt;
    for (auto& L : layers_) {
        if (L.full) { run(L.fa.qkv, xn_); run(L.fa.o, act_); }
        else { run(L.dn.in, xn_); run(L.dn.out, act_); }
        run(L.gate_up, xn_);
        run(L.down, act_);
        break;  // shapes repeat: one layer of each kind is enough
    }
    for (auto& L : layers_)
        if (L.full) { run(L.fa.qkv, xn_); run(L.fa.o, act_); break; }
    if (mtp_.fc.is4) {
        run(mtp_.fc.q, mtp_in_);
        run(mtp_.qkv.q, xn_);
        run(mtp_.o.q, act_);
        run(mtp_.gate_up.q, xn_);
        run(mtp_.down.q, act_);
    }
    }
    CUDA_CHECK(cudaStreamSynchronize(st));
    dec_gemm_warm_ = true;
    (void)I;
}

void Worker::run_spec_batch(const std::vector<int>& slots, const std::vector<int>& kvs, int kd,
                            const std::vector<int>* lkr, const std::vector<int>* kav) {
    attn_rs_ = false;  // row split only for the single-sequence step (launch_step)
    std::string key = std::to_string(kd);
    for (size_t i = 0; i < slots.size(); i++) {
        const int lr = lkr ? (*lkr)[i] : 0;
        key += "|" + std::to_string(slots[i]) + ":" + std::to_string(kvs[i]) + ":" +
               std::to_string((int)slots_[slots[i]].sampling + 2 * (int)slots_[slots[i]].constrained) +
               (lr > 0 ? "L" + std::to_string(kav ? std::min((*kav)[i], lr) : 0) : "");
    }
    cudaGraphExec_t bge = batch_graphs_.get(key);
    if (!bge) {
        const auto tcap0 = std::chrono::steady_clock::now();
        const int n = slots.size(), H = c_.hidden;
        std::vector<int> r0(n), ms(n), r1(n), ones(n, 1);
        int M = 0;
        for (int i = 0; i < n; i++) {
            r0[i] = M;
            ms[i] = kvs[i] + 1;
            M += ms[i];
            r1[i] = i;
        }
        CHECK(M <= kMaxBatchRows && kd <= kMaxSpecK, "batch rows %d", M);
        for (int i = 0; i < n; i++) CHECK(ms[i] <= kPendRows, "rows %d of sequence %d", ms[i], i);
        cudaStream_t st = stream;
        cudaGraph_t g;
        {
            RowsCap cg(rows_cap_, kMaxBatchRows);
            if (dec_gemm_rows(M) && !dec_gemm_warm_) dec_gemm_warm(M);
        }
        CUDA_CHECK(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal));
        const bool defer = dn_defer_on();
        if (!defer)
            for (int sidx : slots) copy_int(slots_[sidx].dn_save, slots_[sidx].dn_cur, st);
        LkBatch lb;  // lookup drafts (lkb_on): fill at the start, history + scan at the end
        bool any_lk = false, any8 = false;
        if (lkb_on()) {
            lb.n = n;
            for (int i = 0; i < n; i++) {
                SeqSlot& sl = slots_[slots[i]];
                lb.d[i] = sl.lk;
                lb.tokens[i] = sl.spec.tokens;
                lb.emit[i] = sl.spec.emit;
                lb.rows[i] = lkr ? (*lkr)[i] : 0;
                lb.kav[i] = kav ? (*kav)[i] : 0;
                lb.pos[i] = sl.spec.pos;
                lb.pos8[i] = ms[i] > kMaxGemvTokens ? sl.pos8 : nullptr;
                any_lk |= lb.rows[i] > 0;
                any8 |= ms[i] > kMaxGemvTokens;
            }
            lookup_fill_batch(lb, st);  // (always: the same topology with or without lookup rows - graph updates)
        }
        CHECK(!any8 || lkb_on(), "batch rows > 8 per sequence need QE_LKB");
        forward_batch(slots, r0, ms);
        std::vector<SpecState> v(n);
        // acceptance of all sequences in one launch (QE_SPEC_BATCH=0: one launch per sequence; same
        // arithmetic): 32 serial single-thread launches cost 0.3 ms per step at n = 32
        static const bool sb = !getenv("QE_SPEC_BATCH") || atoi(getenv("QE_SPEC_BATCH"));
        AcceptBatch ab;
        ab.n = sb ? n : 0;
        ab.m = M;
        for (int i = 0; i < n; i++) {
            SeqSlot& sl = slots_[slots[i]];
            v[i] = sl.spec;
            v[i].out = bout_ + r0[i];
            if (sb) {
                ab.s[i] = v[i];
                ab.p[i] = sl.sampling ? sl.sp : nullptr;
                ab.k[i] = kvs[i];
                ab.r0[i] = r0[i];
            } else if (sl.sampling) {
                spec_accept_sampled(topk_all_, ngpu_, kvs[i], sl.sp, v[i], st, M, r0[i]);
            } else {
                spec_accept(v[i], kvs[i], st);
            }
        }
        spec_accept_batch(topk_all_, ngpu_, ab, st);
        if (defer) {  // accepted rows -> stash, committed by the next step (one launch for all)
            DnStashSegs r{};
            r.n = n;
            int nl = 0;
            for (auto& L : layers_)
                if (!L.full) {
                    r.cv[nl] = spec_cv_[nl];
                    r.qk[nl] = spec_qk_[nl];
                    r.gb[nl] = spec_gb_[nl];
                    nl++;
                }
            for (int i = 0; i < n; i++) {
                SeqSlot& sl = slots_[slots[i]];
                r.r0[i] = r0[i];
                r.cur[i] = sl.dn_cur;
                r.pn[i] = sl.dn_pn;
                r.pend[i] = sl.dn_pend;
                r.acc[i] = sl.spec.acc;
                r.save[i] = sl.dn_save;
                r.fin[i] = ms[i] == 1 && dn_direct_on() && dn_lazy_t(dsh_) == 1;
                r.m[i] = ms[i];
            }
            dn_stash_seg(r, nl, dn_ch_, dsh_, st);
        } else {  // delta-net states after each sequence's accepted rows, one launch for all
            DnReplaySegs rp{};
            rp.n = n;
            int nl = 0;
            for (auto& L : layers_)
                if (!L.full) {
                    rp.cv[nl] = spec_cv_[nl];
                    rp.qk[nl] = spec_qk_[nl];
                    rp.gb[nl] = spec_gb_[nl];
                    for (int i = 0; i < n; i++) rp.S[i][nl] = slots_[slots[i]].S[nl];
                    nl++;
                }
            for (int i = 0; i < n; i++) {
                rp.r0[i] = r0[i];
                rp.cur[i] = slots_[slots[i]].dn_cur;
                rp.acc[i] = slots_[slots[i]].spec.acc;
            }
            dn_replay_seg(rp, nl, dn_ch_, dsh_, dn_slot_floats_, st);
        }
        std::vector<const bf16*> hs(n);
        for (int i = 0; i < n; i++) hs[i] = xn_ + (size_t)r0[i] * H;
        if (lkb_on()) {  // the MTP pass's second segments: mtp_pos (set by accept) + 8 (always: same topology)
            LkBatch pb = lb;
            for (int i = 0; i < n; i++) {
                pb.pos[i] = slots_[slots[i]].spec.mtp_pos;
                pb.pos8[i] = ms[i] > kMaxGemvTokens ? slots_[slots[i]].mpos8 : nullptr;
            }
            lookup_pos8_batch(pb, st);
        }
        mtp_batch(slots, r0, ms, hs);
        SpecBatch sbt;
        for (int i = 0; i < n; i++) {
            v[i].mtp_out = bmout_ + r0[i];
            if (sb) sbt.s[sbt.n++] = v[i];
            else spec_select(v[i], st);
        }
        spec_select_batch(sbt, st);
        for (int r = 1; r < kd; r++) {
            for (int i = 0; i < n; i++)
                hs[i] = mtp_h_ + (size_t)(r == 1 ? r0[i] : i) * H;
            mtp_batch(slots, r1, ones, hs);
            for (int i = 0; i < n; i++) {
                v[i].mtp_out = bmout_ + i;
                spec_append(v[i], r, st);
            }
        }
        if (sb) {
            SpecBatch fb;
            for (int i = 0; i < n; i++) fb.s[fb.n++] = v[i];
            spec_finish_batch(fb, kd, st);
        } else {
            for (int i = 0; i < n; i++) spec_finish(v[i], kd, st);
        }
        if (lkb_on()) lookup_step_batch(lb, st);  // drafts for the next step (count at emit[31])
        CUDA_CHECK(cudaStreamEndCapture(st, &g));
        const auto tcap1 = std::chrono::steady_clock::now();
        if (!batch_update(key, g, &bge)) instantiate(&bge, g);
        static double tcap_tot = 0;
        tcap_tot += std::chrono::duration<double, std::milli>(tcap1 - tcap0).count();
        CUDA_CHECK(cudaGraphDestroy(g));
        batch_graphs_.put(key, bge);
        static const bool stats = getenv("QE_GRAPH_STATS") && atoi(getenv("QE_GRAPH_STATS"));
        if (stats) {  // capture + instantiate cost of batch graphs (rank 0)
            static int cnt = 0;
            static double tot = 0;
            tot += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tcap0).count();
            if (rank == 0 && ++cnt % 20 == 0)
                fprintf(stderr, "[graphs] %d batch captures, %.1f ms each, capture %.1f (n=%d), updates %lld ok %lld failed\n", cnt,
                        tot / cnt, tcap_tot / cnt, n, bupd_ok_, bupd_fail_);
        }
    }
    CUDA_CHECK(cudaGraphLaunch(bge, stream));
    blaunch_[1] = blaunch_[0];
    blaunch_[0] = bge;
}

// structure of a batched-step key: the sequences and their modes without the row counts
static std::string batch_skey(const std::string& key) {
    std::string r;
    size_t i = key.find('|');
    if (i == std::string::npos) return key;
    r = key.substr(0, i);  // kd
    while (i != std::string::npos) {
        const size_t j = key.find('|', i + 1);
        const std::string t = key.substr(i + 1, j == std::string::npos ? std::string::npos : j - i - 1);
        const size_t a = t.find(':'), b = t.find(':', a + 1);
        std::string mode = t.substr(b + 1);
        const size_t l = mode.find('L');
        if (l != std::string::npos) mode = mode.substr(0, l);
        r += "|" + t.substr(0, a) + ":" + mode;
        i = j;
    }
    return r;
}

bool Worker::batch_update(const std::string& key, cudaGraph_t g, cudaGraphExec_t* out) {
    static const bool on = !getenv("QE_GRAPH_UPDATE") || atoi(getenv("QE_GRAPH_UPDATE"));
    if (!on) return false;
    const std::string sk = batch_skey(key);
    // the least recently used cached graph of the same structure that is not (maybe) in flight
    auto best = batch_graphs_.m.end();
    for (auto it = batch_graphs_.m.begin(); it != batch_graphs_.m.end(); ++it) {
        if (it->second.g == blaunch_[0] || it->second.g == blaunch_[1]) continue;
        if (batch_skey(it->first) != sk) continue;
        if (best == batch_graphs_.m.end() || it->second.used < best->second.used) best = it;
    }
    if (best == batch_graphs_.m.end()) return false;
    cudaGraphExecUpdateResultInfo info;
    if (cudaGraphExecUpdate(best->second.g, g, &info) != cudaSuccess) {
        cudaGetLastError();
        bupd_fail_++;
        return false;
    }
    *out = best->second.g;
    batch_graphs_.m.erase(best);  // now the new key's graph (put by the caller)
    bupd_ok_++;
    return true;
}

void Worker::head_topk(const Lin& w, const bf16* x, int m) {
    const int V = p_.vocab.size();
    gemv(w, x, c_.hidden, logits_, V, m, stream);
    apply_head_masks();
    topk_local(logits_, V, m, p_.vocab.b, topk_, stream);
    allgather_small(topk_, m * kTopK * 2, topk_all_, comm_, rank, ctr_, stream);
}

int dec38_max() {
    static const int v = getenv("QE_DEC38_MAX") ? atoi(getenv("QE_DEC38_MAX")) : kD8RsMax;
    return v;
}

AttnShape Worker::layer_shape(const Layer* L, int win) const {
    AttnShape a = ash_;
    a.kc2 = L ? L->fa.kc2 : mtp_.kc2;
    a.vc2 = L ? L->fa.vc2 : mtp_.vc2;
    a.fp8 = L ? kv_fp8_ : mtp_fp8();
    a.ks = L ? L->fa.ks : mtp_ks_ * mtp_sc_mul_;
    a.vs = L ? L->fa.vs : mtp_vs_ * mtp_sc_mul_;
    // MTP drafts only need recent context: a sliding window bounds their KV reads at long
    // context. win < 0 means the dec8_first default (32K beyond 144K); the recursive draft
    // passes pass a tighter window (see run_spec), the verification pass keeps full attention.
    if (!L && a.fp8) a.win = win;
    // QE_DEC_CHUNK_MIN: fp8 decode split rule (default -1 = ~38 splits up to 52K keys; 0 = old L/82; N > 0 = old
    // rule with >= N keys per split). Length-only, so it never breaks MTP == greedy / batched == alone.
    static const int dcm = getenv("QE_DEC_CHUNK_MIN") ? atoi(getenv("QE_DEC_CHUNK_MIN")) : -1;
    a.dchunk_min = dcm;
    a.d38max = dec38_max();
    a.d8rs = L && attn_rs_ ? 1 : 0;
    if (!L) a.rpos3 = nullptr;  // MTP drafts: text positions (image rows only matter for drafts)
    return a;
}

void Worker::attend(const AttnShape& a, const bf16* q, const void* kc, const void* vc,
                    const int* pos, int m, const float* proj, bf16* out, __half* xh) {
    if (!proj) proj = proj_;
    if (!out) out = act_;
    if (m <= kMaxGemvTokens) {
        attention(q, kc, vc, a, proj, proj_ld_, pos, m, attn_work_, out, stream, xh);
    } else if (verify_ && m <= kPendRows) {
        // a lookup verification of up to 16 rows: the decode attention in two halves of <= 8 rows (each row's
        // result as in any decode step; the second half's exchange in its own slot)
        const int qbw = ash_.nq * 256;
        attention(q, kc, vc, a, proj, proj_ld_, pos, kMaxGemvTokens, attn_work_, out, stream, xh);
        AttnShape a2 = a;
        xchg_fill(a2);
        spec_add_pos(pos8_, pos, kMaxGemvTokens, stream);
        const size_t r0 = kMaxGemvTokens;
        attention(q + r0 * qbw, kc, vc, a2, proj + r0 * proj_ld_, proj_ld_, pos8_, m - kMaxGemvTokens,
                  attn_work_ + attn_work_floats(ash_), out + r0 * ow_, stream, xh ? xh + r0 * ow_ : nullptr);
    } else
        attention_prefill(q, kc, vc, a, proj, proj_ld_, pos, m, out, stream, pattn_work_,
                          pattn_work_floats_);
}

// bf16 output mode of the W4A8 GEMM (gemm.cu of the prefill-gemm-floor branch): the partial rows
// that only feed the deferred prefill all-reduce are written as bf16 (the AR rounds them to bf16
// anyway: bit-identical sums, half the bytes). Weak: without that GEMM the mode is off.
extern void gemm_set_out_bf16(bool on) __attribute__((weak));
static bool part_bf16_on(bool defer) {
    return defer && &gemm_set_out_bf16 != nullptr && env_int("QE_AR_B16", 1) != 0;
}
// matmul into the partial rows of a deferred all-reduce, in bf16 when pb
void Worker::matmul_part(const W4& w, const bf16* x, int ldx, int m, size_t row0, bool pb) {
    if (pb) gemm_set_out_bf16(true);
    matmul(w, x, ldx, pb ? reinterpret_cast<float*>(reinterpret_cast<bf16*>(partial_) + row0 * c_.hidden)
                         : partial_ + row0 * c_.hidden, c_.hidden, m, stream);
    if (pb) gemm_set_out_bf16(false);
}

// QE_DBG_SKIP (timing experiments only, the results are wrong): bit 1 = deferred AR tail (one row
// kept, it zeroes the GEMM counter), 2 = DeltaNet between in_proj and out_proj, 4 = MTP pass of
// prefill chunks, 8 = prefill attention core, 16 = the ab projection, 32 = the all-reduce wait,
// 64 = the prefill all-reduce kernels themselves
static int dbg_skip() { return env_int("QE_DBG_SKIP", 0); }

void Worker::mixer_rows(int i, int r0, int mb, const int* pos, int slot_off, bool pb) {
    Layer& L = layers_[i];
    const int H = c_.hidden;
    const float eps = c_.eps;
    cudaStream_t st = stream;
    const bf16* x = xn_ + (size_t)r0 * H;
    float* proj = proj_ + (size_t)r0 * proj_ld_;
    if (L.full) {
        const int qw = ow_, qbw = ash_.nq * 256;
        matmul(L.fa.qkv, x, H, proj, proj_ld_, mb, st);
        AttnShape a = layer_shape(&L);
        a.xrow0 = r0;
        a.pf_p0 = pf_p0_ >= 0 ? pf_p0_ + r0 : -1;
        xchg_fill(a);
        attn_prep(proj, proj_ld_, a, L.fa.q_norm, L.fa.k_norm, eps, inv_freq_, pos, mb,
                  qbuf_ + (size_t)r0 * qbw, L.fa.kc, L.fa.vc, st);
        if (!(dbg_skip() & 8)) attend(a, qbuf_ + (size_t)r0 * qbw, L.fa.kc, L.fa.vc, pos, mb, proj, act_ + (size_t)r0 * qw);
        matmul_part(L.fa.o, act_ + (size_t)r0 * qw, qw, mb, r0, pb);
    } else {
        const int ldab = 2 * dsh_.ab_pad, vw = dsh_.nv * 128;
        matmul(L.dn.in, x, H, proj, proj_ld_, mb, st);
        const size_t abs = (size_t)kChunk * ldab;
        float* ab = ab_ + (size_t)r0 * ldab;
        DeltaShape ds = dsh_;  // ab_splits = 1 with the single-slice prefill projection
        const bool ab1 = dn_prefill_fused(mb) && env_int("QE_DN_AB", 1) && dn_ab_prefill_ok(L.dn.ab);
        if (ab1) ds.ab_splits = 1;
        if (mb <= kMaxGemvTokens) gemv_w16(L.dn.ab, x, H, ab, ldab, mb, st, kAbSplits, abs);
        else if (dbg_skip() & 16) {
        } else if (ab1) dn_ab_prefill(L.dn.ab, xbf_cur_ ? xbf_cur_ : x, H, ab, ldab, mb, st);
        else gemm_w16(L.dn.ab, xbf_cur_ ? xbf_cur_ : x, H, ab, ldab, mb, st, kAbSplits, abs);
        float* cv = conv_out_ + (size_t)r0 * dn_ch_;
        if (dbg_skip() & 2) {
        } else if (dn_prefill_fused(mb)) {
            dn_prefill(proj, proj_ld_, dn_ch_, L.dn.conv_w, L.dn.hist, pos, ds, ab, L.dn.A, L.dn.dt_bias,
                       L.dn.S, dn_slot_floats_, dn_cur_, mb, cv, dn_qk_ + (size_t)r0 * 2 * dsh_.nk * 128,
                       dn_gb_ + (size_t)r0 * 2 * dsh_.nv, dno_ + (size_t)r0 * vw, st, slot_off);
        } else {
            dn_conv(proj, proj_ld_, dn_ch_, L.dn.conv_w, L.dn.hist, pos, mb, cv, st);
            dn_core(cv, dn_ch_, dsh_, ab, L.dn.A, L.dn.dt_bias, L.dn.S, dn_slot_floats_, dn_cur_, mb,
                    false, dn_qk_ + (size_t)r0 * 2 * dsh_.nk * 128, dn_gb_ + (size_t)r0 * 2 * dsh_.nv,
                    dno_ + (size_t)r0 * vw, st, slot_off);
        }
        {
            int8_t* xq = nullptr;
            float* xsc = nullptr;
            int G = 0;
            gemm_i8_prequant_out(act_ + (size_t)r0 * vw, mb, vw, r0, &xq, &xsc, &G);
            dn_out(dno_ + (size_t)r0 * vw, proj + dn_ch_, proj_ld_, L.dn.norm_w, dsh_.nv, eps, mb,
                   act_ + (size_t)r0 * vw, st, nullptr, xq, xsc, G);
        }
        matmul_part(L.dn.out, act_ + (size_t)r0 * vw, vw, mb, r0, pb);
    }
}

void Worker::forward_prefill(int m, bool logits) {
    const int H = c_.hidden, I = p_.inter.size();
    const float eps = c_.eps;
    cudaStream_t st = stream, cs = comm_stream_;
    const int m0 = m < 256 ? m / 2 : std::min(m - 64, std::max(128, (m / 2 + 127) / 128 * 128));
    const int r0s[2] = {0, m0}, mbs[2] = {m0, m - m0};
    dn_flush_cur(st);
    add_int(pos_mb_, pos_, m0, st);
    const int* poss[2] = {pos_, pos_mb_};
    if (p_.pos_split) {  // the stats flags need a fresh epoch per prefill chunk (old layout: as before)
        comm_begin_unit(ctr_, st);
        ll_idx_ = 0;
        xidx_ = 0;
    }
    embed_gather(embed_, tokens_, m, H, resid_, st);
    rmsnorm(resid_, layers_[0].in_norm, m, H, eps, xn_, st);
    for (int b = 0; b < 2; b++) {  // RoPE table of both micro-batches (rows r0s[b] + i)
        AttnShape a = ash_;
        a.xrow0 = r0s[b];
        if (p_.pos_split) rope_table(a, poss[b], mbs[b], inv_freq_, rtab_, st);
    }
    cudaEvent_t ready[2] = {nullptr, nullptr};  // xn rows of micro-batch b are valid after this
    // deferred tail of the all-reduce (P2P): the few comm blocks only move the data, the residual
    // update + norm run on this stream with all SMs once the rows are needed
    const bf16* pending[2] = {nullptr, nullptr};
    bool pending_glu[2] = {false, false};  // the rows feed the MLP gate_up next (else the mixer)
    // the rows only feed a W4A8 GEMM (MLP gate_up, full-attention qkv): the deferred tail writes
    // just their int8 copy (QE_ARF_XN=1: also the bf16 rows, as before)
    bool pending_i8only[2] = {false, false};
    const bool arf_xn = env_int("QE_ARF_XN", 0) != 0;
    bool defer_ok[2];
    for (int b = 0; b < 2; b++) defer_ok[b] = allreduce_deferrable(comm_, mbs[b], H, r0s[b]);
    bool pb[2];  // partial rows of b in bf16 (GEMM output mode, see part_bf16_on)
    for (int b = 0; b < 2; b++) pb[b] = part_bf16_on(defer_ok[b]);
    // owner-finished all-reduce (QE_AR_OWNFIN, comm.cuh OwnFin): the owners of the rows add, norm and
    // quantize them and send the int8 (and, for the DeltaNet ab projection / final norm, bf16) rows
    // to every GPU: no finishing kernel over all rows on this stream; the residual stays valid on
    // the owned rows only (same owner ranges for every collective of a micro-batch)
    const int g_gu = gemm_i8_prequant_group(H, true), g_in = gemm_i8_prequant_group(H, false);
    const bool of_v2 = gemm_i8_variant() >= 2;
    // copy-engine all-reduce (QE_AR_CE): both micro-batches deferred, 3 GPUs (two peers per GPU)
    const bool ce_on = env_int("QE_AR_CE", 1) && ngpu_ == 3 && ce_ar_ok(comm_, mbs[0], H, r0s[0]) &&
                       ce_ar_ok(comm_, mbs[1], H, r0s[1]) && defer_ok[0] && defer_ok[1];
    // owner finish on the SM all-reduce (QE_AR_OWNFIN, off: slower) or on the copy-engine one (QE_AR_CEOF)
    // (merge prefill-night-combo: the CE finish kernel writes only the gemm_i8t record layout (variant 1); the SM
    // owner finish of prefill-rest-x-gemm writes both, as before)
    const bool of_on = (env_int("QE_AR_OWNFIN", 0) || (ce_on && env_int("QE_AR_CEOF", 0))) && defer_ok[0] &&
                       defer_ok[1] && allreduce_ownfin_ok(comm_, H) &&
                       (ce_on ? gemm_i8_variant() == 1 : gemm_i8_variant() >= 1) && g_gu > 0 && g_in > 0 &&
                       !(dbg_skip() & 1);
    int pending_G[2] = {0, 0};
    bool pending_wxn[2] = {false, false};
    xbf_[0] = xbf_[1] = nullptr;
    auto reduce = [&](int b, const bf16* norm, bool glu_next, bool i8only) {  // all-reduce rows of b on the comm stream
        // copy-engine AR with fp32 partials: pack them to bf16 on this stream with all SMs (~20 us; on a
        // few comm-stream blocks it took ~230 us and slowed the concurrent kernels)
        const bool ce_pack_main = ce_on && !pb[b] && !(dbg_skip() & 64) && (of_on || env_int("QE_CE_PACK_MAIN", 1));
        if (ce_pack_main) {
            const size_t o = (size_t)r0s[b] * H;
            ce_pack(partial_ + o, mbs[b], H, comm_, rank, r0s[b], st, 0);
        }
        cudaEvent_t e = ev();
        CUDA_CHECK(cudaEventRecord(e, st));
        CUDA_CHECK(cudaStreamWaitEvent(cs, e, 0));
        const size_t o = (size_t)r0s[b] * H;
        const float* part = pb[b] ? reinterpret_cast<const float*>(reinterpret_cast<bf16*>(partial_) + o) : partial_ + o;
        OwnFin of;
        if (of_on) {
            of.G = norm == final_norm_ ? 0 : glu_next ? g_gu : g_in;
            of.wxn = of.G == 0 || !i8only || arf_xn;
            of.v2 = of_v2;
        }
        if (dbg_skip() & 64) {  // (timing only: no all-reduce at all)
        } else if (ce_on && of_on) {  // copy-engine, owner-finished: int8 / bf16 normed rows by DMA, no ar_finish
            const unsigned seq = ++ce_seq_;
            const void* rows = part;
            if (!pb[b]) rows = comm_.pstage[rank] + (size_t)r0s[b] * (H / 4);  // packed on the compute stream
            auto cmd = [&](bool rs) {
                CeCmd c;
                CUDA_CHECK(cudaEventCreateWithFlags(&c.ev, cudaEventDisableTiming));
                CUDA_CHECK(cudaEventRecord(c.ev, cs));
                c.seq = seq;
                c.kind = rs ? 0 : 1;
                for (int j = 0, k = 0; j < ngpu_; j++) {
                    if (j == rank) continue;
                    c.peer[k] = j;
                    if (rs) {
                        c.cp[k] = ce_rs_copy(comm_, rank, j, mbs[b], H, r0s[b], rows, seq);
                        c.ncp[k] = 1;
                    } else {
                        CeCopy x[3];
                        c.ncp[k] = ce_of_copies(comm_, rank, j, mbs[b], H, r0s[b], of.G, of.wxn, x);
                        for (int u = 0; u < c.ncp[k]; u++) c.cpx[k][u] = x[u];
                    }
                    k++;
                }
                ce_push(c);
            };
            cmd(true);
            ce_take_wait(seq, 0, cs);
            ce_finish(resid_ + o, part, pb[b], mbs[b], H, comm_, rank, norm, eps, r0s[b], of.G, of.wxn, cs);
            cmd(false);
            ce_take_wait(seq, 1, cs);
        } else if (ce_on && !of_on) {  // copy-engine all-reduce: only local kernels on this GPU's SMs
            const unsigned seq = ++ce_seq_;
            const void* rows = part;
            if (!pb[b]) {
                if (!ce_pack_main) ce_pack(part, mbs[b], H, comm_, rank, r0s[b], cs, env_int("QE_CE_PACK_BLOCKS", 8));
                rows = comm_.pstage[rank] + (size_t)r0s[b] * (H / 4);
            }
            auto cmd = [&](bool rs) {
                CeCmd c;
                CUDA_CHECK(cudaEventCreateWithFlags(&c.ev, cudaEventDisableTiming));
                CUDA_CHECK(cudaEventRecord(c.ev, cs));
                c.seq = seq;
                c.kind = rs ? 0 : 1;
                for (int j = 0, k = 0; j < ngpu_; j++) {
                    if (j == rank) continue;
                    c.peer[k] = j;
                    c.cp[k++] = rs ? ce_rs_copy(comm_, rank, j, mbs[b], H, r0s[b], rows, seq)
                                   : ce_ag_copy(comm_, rank, j, mbs[b], H, r0s[b], seq);
                }
                ce_push(c);
            };
            cmd(true);
            ce_take_wait(seq, 0, cs);  // the peers' rows of this GPU's owner range landed
            ce_sum(part, pb[b], mbs[b], H, comm_, rank, r0s[b], seq, cs);
            cmd(false);
            ce_take_wait(seq, 1, cs);  // all sums landed
        } else {
            allreduce_add(resid_ + o, part, mbs[b], H, comm_, rank, ctr_, cs, norm, eps,
                          xn_ + o, -1, true, nullptr, defer_ok[b], r0s[b], pb[b], of_on ? &of : nullptr);
        }
        if (of_on) { pending[b] = norm; pending_G[b] = of.G; pending_wxn[b] = of.wxn; }
        else if (defer_ok[b]) { pending[b] = norm; pending_glu[b] = glu_next; pending_i8only[b] = i8only && !arf_xn; }
        ready[b] = ev();
        CUDA_CHECK(cudaEventRecord(ready[b], cs));
    };
    static const bool nowait = getenv("QE_DBG_NOWAIT") != nullptr;
    auto wait = [&](int b) {
        if (ready[b] && !nowait && !(dbg_skip() & 32)) CUDA_CHECK(cudaStreamWaitEvent(st, ready[b], 0));
        if (pending[b] && of_on) {  // the owners wrote the rows into this GPU's receive buffers
            const size_t o = (size_t)r0s[b] * H;
            const int G = pending_G[b];
            if (G > 0) {
                const size_t rec_off = of_v2 ? (size_t)(r0s[b] + 15) / 16 * 16 * ((size_t)(H / 128) * (128 / G) * 2 * 2)
                                             : (size_t)r0s[b] * ((size_t)(H / 128) * ((128 / G + 1) / 2) * 4);
                gemm_i8_prequant_ext(xn_ + o, mbs[b], H, G, comm_.pxq[rank] + o, comm_.pxsc[rank] + rec_off);
                zero_int(gemm_i8_sk_ctr(), st);
            }
            xbf_[b] = pending_wxn[b] ? reinterpret_cast<const bf16*>(comm_.pxn[rank]) + o : nullptr;
            pending[b] = nullptr;
        } else if (pending[b]) {
            const size_t o = (size_t)r0s[b] * H;
            int8_t* xq = nullptr;
            float* xsc = nullptr;
            int G = 0;
            int* ctr = nullptr;
            // the W4A8 GEMM consuming these rows takes its int8 input straight from this kernel
            gemm_i8_prequant(xn_ + o, mbs[b], H, r0s[b], pending_glu[b], &xq, &xsc, &G, &ctr);
            allreduce_finish(resid_ + o, (dbg_skip() & 1) ? 1 : mbs[b], H, comm_, rank, pending[b], eps, xn_ + o, r0s[b], st, xq, xsc, G,
                             xq ? ctr : nullptr, !(xq && pending_i8only[b]));
            pending[b] = nullptr;
        }
    };
    for (int i = 0; i < c_.layers; i++) {
        Layer& L = layers_[i];
        for (int b = 0; b < 2; b++) {
            wait(b);
            xbf_cur_ = xbf_[b];  // bf16 input rows of the ab projection (owner-finished AR), else xn_
            mixer_rows(i, r0s[b], mbs[b], poss[b], b == 0 ? 0 : m0, pb[b]);
            xbf_cur_ = nullptr;
            reduce(b, L.post_norm, L.gate_up.glu, true);  // MLP input: W4A8 gate_up only
        }
        const bf16* next_norm = i + 1 < c_.layers ? layers_[i + 1].in_norm : final_norm_;
        for (int b = 0; b < 2; b++) {
            wait(b);
            const size_t ox = (size_t)r0s[b] * H;
            float* proj = proj_ + (size_t)r0s[b] * proj_ld_;
            bf16* act = act_ + (size_t)r0s[b] * I;
            if (L.gate_up.glu) {
                matmul(L.gate_up, xn_ + ox, H, reinterpret_cast<float*>(act), I, mbs[b], st);
            } else {
                matmul(L.gate_up, xn_ + ox, H, proj, proj_ld_, mbs[b], st);
                silu_mul(proj, proj_ld_, I, mbs[b], act, st);
            }
            matmul_part(L.down, act, I, mbs[b], r0s[b], pb[b]);
            // next mixer input: full attention reads only the int8 copy (qkv GEMM), DeltaNet also
            // the bf16 rows (ab projection), the final norm feeds lm_head / MTP in bf16
            reduce(b, next_norm, false, i + 1 < c_.layers && layers_[i + 1].full);
        }
    }
    wait(0);
    wait(1);
    if (of_on)  // final norm rows (lm_head, MTP input) from the receive buffer
        CUDA_CHECK(cudaMemcpyAsync(xn_, comm_.pxn[rank], (size_t)m * H * 2, cudaMemcpyDeviceToDevice, st));
    dn_commit(dn_cur_, nullptr, m, st);
    if (logits) {
        mask_current(1, false);
        if (!sampling) {
            head(lm_head_, xn_ + (size_t)(m - 1) * H, 1, out_tok_ + m - 1);
        } else {
            head_topk(lm_head_, xn_ + (size_t)(m - 1) * H, 1);
            sample_rows(topk_all_, ngpu_, 1, sp_, out_tok_ + m - 1, st);
        }
    }
}

void Worker::head(const Lin& w, const bf16* x, int m, int* out, bool draft) {
    const int V = draft ? draft_v_ : p_.vocab.size();
    gemv(w, x, c_.hidden, logits_, V, m, stream);
    if (!draft) apply_head_masks();
    argmax_local(logits_, V, m, p_.vocab.b, pairs_, stream, draft ? draft_map_ : nullptr);
    allgather_small(pairs_, 2 * m, pairs_all_, comm_, rank, ctr_, stream);
    argmax_final(pairs_all_, ngpu_, m, out, stream);
}

// recursive MTP draft passes (2..k) may use a tighter sliding window than the verification pass:
// they only affect which tokens are proposed, never the output. QE_MTP_WINDOW sets it (default:
// the dec8_first rule, i.e. 32K beyond 144K); QE_MTP_WINDOW2 overrides it for these passes only.
static int mtp_draft_win() {
    static const int w = getenv("QE_MTP_WINDOW2") ? atoi(getenv("QE_MTP_WINDOW2"))
                       : getenv("QE_MTP_WINDOW")  ? atoi(getenv("QE_MTP_WINDOW"))  : -1;
    return w;
}

void Worker::mtp_forward(int m, const int* pos, const int* tok, const bf16* hsrc, const int* hrow,
                         bool logits, int win) {
    const int H = c_.hidden, I = p_.inter.size();
    const float eps = c_.eps;
    cudaStream_t st = stream;
    comm_begin_unit(ctr_, st);
    ll_idx_ = 0;
    xidx_ = 0;
    mtp_prep(embed_, tok, hsrc, hrow, mtp_.pre_e, mtp_.pre_h, eps, m, H, mtp_in_, st);
    mm4(mtp_.fc, mtp_in_ + p_.fc_k.b, 2 * H, partial_, H, m, st);
    CUDA_CHECK(cudaMemsetAsync(mtp_resid_, 0, (size_t)m * H * 4, st));
    __half* xhx = m <= rows_cap_ ? xh_xn_ : nullptr;
    __half* xha = m <= rows_cap_ ? xh_act_ : nullptr;
    allreduce_add(mtp_resid_, partial_, m, H, comm_, rank, ctr_, st, mtp_.in_norm, eps,
                  xhx ? nullptr : xn_, next_ll(m), false, xhx);
    mm4(mtp_.qkv, xn_, H, proj_, proj_ld_, m, st, xhx, H);
    AttnShape a = layer_shape(nullptr, win);  // MTP cache stays bf16 (no calibrated scales)
    if (p_.pos_split) rope_table(a, pos, m, inv_freq_, rtab_, st);
    xchg_fill(a);
    attn_prep(proj_, proj_ld_, a, mtp_.q_norm, mtp_.k_norm, eps, inv_freq_, pos, m, qbuf_,
              mtp_.kc, mtp_.vc, st);
    verify_ = m <= kPendRows;
    attend(a, qbuf_, mtp_.kc, mtp_.vc, pos, m, nullptr, nullptr, xha);
    verify_ = false;
    mm4(mtp_.o, act_, ow_, partial_, H, m, st, m <= kPendRows ? xha : nullptr, ow_);
    allreduce_add(mtp_resid_, partial_, m, H, comm_, rank, ctr_, st, mtp_.post_norm, eps,
                  xhx ? nullptr : xn_, next_ll(m), false, xhx);
    mm4(mtp_.gate_up, xn_, H, proj_, proj_ld_, m, st, xhx, H);
    silu_mul(proj_, proj_ld_, I, m, act_, st, xha);
    mm4(mtp_.down, act_, I, partial_, H, m, st, xha, I);
    allreduce_add(mtp_resid_, partial_, m, H, comm_, rank, ctr_, st, mtp_.norm, eps, xn_,
                  next_ll(m));
    CUDA_CHECK(cudaMemcpyAsync(mtp_h_, xn_, (size_t)m * H * 2, cudaMemcpyDeviceToDevice, st));
    if (logits) {
        if (m <= kPendRows) head(draft_head_, xn_, m, spec_.mtp_out, true);  // (16: lookup verification rows)
        else head(draft_head_, xn_ + (size_t)(m - 1) * H, 1, spec_.mtp_out + m - 1, true);
    }
}

// MTP layer over a prefill chunk (hidden rows xn_, tokens spec_.mtp_tok, positions pos_) in the two
// micro-batches of forward_prefill: each all-reduce runs on the comm stream next to the other
// micro-batch's GEMMs, with the deferred tail (resid + norm + int8 of the next GEMM) on this stream.
// Same per-row arithmetic as mtp_forward (QE_MTP_OVERLAP=0: mtp_forward over all rows).
void Worker::mtp_forward_prefill(int m, bool logits) {
    const int H = c_.hidden, I = p_.inter.size();
    const float eps = c_.eps;
    cudaStream_t st = stream, cs = comm_stream_;
    const int m0 = m < 256 ? m / 2 : std::min(m - 64, std::max(128, (m / 2 + 127) / 128 * 128));
    const int r0s[2] = {0, m0}, mbs[2] = {m0, m - m0};
    add_int(pos_mb_, pos_, m0, st);
    const int* poss[2] = {pos_, pos_mb_};
    comm_begin_unit(ctr_, st);
    ll_idx_ = 0;
    xidx_ = 0;
    mtp_prep(embed_, spec_.mtp_tok, xn_, nullptr, mtp_.pre_e, mtp_.pre_h, eps, m, H, mtp_in_, st);
    CUDA_CHECK(cudaMemsetAsync(mtp_resid_, 0, (size_t)m * H * 4, st));
    AttnShape a0 = layer_shape(nullptr);
    for (int b = 0; b < 2; b++) {
        AttnShape a = a0;
        a.xrow0 = r0s[b];
        if (p_.pos_split) rope_table(a, poss[b], mbs[b], inv_freq_, rtab_, st);
    }
    cudaEvent_t ready[2] = {nullptr, nullptr};
    const bf16* pending[2] = {nullptr, nullptr};
    bool pending_glu[2] = {false, false}, pending_i8only[2] = {false, false};
    bool defer_ok[2];
    for (int b = 0; b < 2; b++) defer_ok[b] = allreduce_deferrable(comm_, mbs[b], H, r0s[b]);
    bool pb[2];
    for (int b = 0; b < 2; b++) pb[b] = part_bf16_on(defer_ok[b]) && mtp_.fc.is4 && mtp_.o.is4 && mtp_.down.is4;
    auto reduce = [&](int b, const bf16* norm, bool glu_next, bool i8only) {
        cudaEvent_t e = ev();
        CUDA_CHECK(cudaEventRecord(e, st));
        CUDA_CHECK(cudaStreamWaitEvent(cs, e, 0));
        const size_t o = (size_t)r0s[b] * H;
        const float* part = pb[b] ? reinterpret_cast<const float*>(reinterpret_cast<bf16*>(partial_) + o) : partial_ + o;
        allreduce_add(mtp_resid_ + o, part, mbs[b], H, comm_, rank, ctr_, cs, norm, eps,
                      xn_ + o, -1, true, nullptr, defer_ok[b], r0s[b], pb[b]);
        if (defer_ok[b]) { pending[b] = norm; pending_glu[b] = glu_next; pending_i8only[b] = i8only; }
        ready[b] = ev();
        CUDA_CHECK(cudaEventRecord(ready[b], cs));
    };
    auto wait = [&](int b) {
        if (ready[b]) CUDA_CHECK(cudaStreamWaitEvent(st, ready[b], 0));
        if (pending[b]) {
            const size_t o = (size_t)r0s[b] * H;
            int8_t* xq = nullptr;
            float* xsc = nullptr;
            int G = 0;
            int* ctr = nullptr;
            gemm_i8_prequant(xn_ + o, mbs[b], H, r0s[b], pending_glu[b], &xq, &xsc, &G, &ctr);
            allreduce_finish(mtp_resid_ + o, mbs[b], H, comm_, rank, pending[b], eps, xn_ + o, r0s[b], st, xq,
                             xsc, G, xq ? ctr : nullptr, !(xq && pending_i8only[b]));
            pending[b] = nullptr;
        }
    };
    // Only the MTP KV cache (keys / values of every prompt row) and the drafts after the LAST row
    // are ever used: the attention + MLP output of the other rows feeds nothing (prefill_select
    // takes row m - 1, the recursion hrow = m - 1). QE_MTP_LAST (default 1): rows -> fc, norm, qkv,
    // K/V into the cache; then, for the prompt's last chunk only, the last row once more through
    // the whole layer as a decode-sized pass (mtp_forward(1)), its outputs moved to row m - 1.
    const bool last_only = env_int("QE_MTP_LAST", 1) != 0;
    if (last_only && logits)  // keep the main model's last hidden row (the passes below overwrite xn_)
        CUDA_CHECK(cudaMemcpyAsync(mtp_h_ + (size_t)(m - 1) * H, xn_ + (size_t)(m - 1) * H, (size_t)H * 2,
                                   cudaMemcpyDeviceToDevice, st));
    for (int b = 0; b < 2; b++) {  // fc: [embedding ; hidden] -> H
        if (pb[b]) matmul_part(mtp_.fc.q, mtp_in_ + (size_t)r0s[b] * 2 * H + p_.fc_k.b, 2 * H, mbs[b], r0s[b], true);
        else mm4(mtp_.fc, mtp_in_ + (size_t)r0s[b] * 2 * H + p_.fc_k.b, 2 * H, partial_ + (size_t)r0s[b] * H, H,
                 mbs[b], st);
        reduce(b, mtp_.in_norm, false, true);  // -> qkv only
    }
    const int qw = ow_, qbw = ash_.nq * 256;
    for (int b = 0; b < 2; b++) {
        wait(b);
        const int r0 = r0s[b], mb = mbs[b];
        float* proj = proj_ + (size_t)r0 * proj_ld_;
        mm4(mtp_.qkv, xn_ + (size_t)r0 * H, H, proj, proj_ld_, mb, st);
        AttnShape a = a0;
        a.xrow0 = r0;
        xchg_fill(a);
        attn_prep(proj, proj_ld_, a, mtp_.q_norm, mtp_.k_norm, eps, inv_freq_, poss[b], mb,
                  qbuf_ + (size_t)r0 * qbw, mtp_.kc, mtp_.vc, st);
        if (last_only) continue;
        attend(a, qbuf_ + (size_t)r0 * qbw, mtp_.kc, mtp_.vc, poss[b], mb, proj, act_ + (size_t)r0 * qw);
        if (pb[b]) matmul_part(mtp_.o.q, act_ + (size_t)r0 * qw, qw, mb, r0, true);
        else mm4(mtp_.o, act_ + (size_t)r0 * qw, qw, partial_ + (size_t)r0 * H, H, mb, st);
        reduce(b, mtp_.post_norm, mtp_.gate_up.is4 && mtp_.gate_up.q.glu, true);  // -> gate_up only
    }
    if (last_only) {
        if (!logits) return;
        // the prompt's last row as a one-row MTP pass (its K/V are rewritten with the same row)
        add_int(pos_mb_, pos_, m - 1, st);
        mtp_forward(1, pos_mb_, spec_.mtp_tok + m - 1, mtp_h_ + (size_t)(m - 1) * H, nullptr, true);
        CUDA_CHECK(cudaMemcpyAsync(mtp_h_ + (size_t)(m - 1) * H, mtp_h_, (size_t)H * 2, cudaMemcpyDeviceToDevice, st));
        CUDA_CHECK(cudaMemcpyAsync(spec_.mtp_out + m - 1, spec_.mtp_out, 4, cudaMemcpyDeviceToDevice, st));
        return;
    }
    for (int b = 0; b < 2; b++) {
        wait(b);
        const int r0 = r0s[b], mb = mbs[b];
        float* proj = proj_ + (size_t)r0 * proj_ld_;
        bf16* act = act_ + (size_t)r0 * I;
        mm4(mtp_.gate_up, xn_ + (size_t)r0 * H, H, proj, proj_ld_, mb, st);
        silu_mul(proj, proj_ld_, I, mb, act, st);
        if (pb[b]) matmul_part(mtp_.down.q, act, I, mb, r0, true);
        else mm4(mtp_.down, act, I, partial_ + (size_t)r0 * H, H, mb, st);
        reduce(b, mtp_.norm, false, false);  // final MTP norm: bf16 rows (draft head, recursion)
    }
    wait(0);
    wait(1);
    CUDA_CHECK(cudaMemcpyAsync(mtp_h_, xn_, (size_t)m * H * 2, cudaMemcpyDeviceToDevice, st));
    if (logits) head(draft_head_, xn_ + (size_t)(m - 1) * H, 1, spec_.mtp_out + m - 1, true);
}

static int overlap_min() {
    static const int v = getenv("QE_OVERLAP_MIN") ? atoi(getenv("QE_OVERLAP_MIN")) : kOverlapMin;
    return v;
}

void Worker::set_rope_delta(int delta) {
    CUDA_CHECK(cudaSetDevice(device));
    CUDA_CHECK(cudaMemcpy(slots_.at(cur_slot_).rdelta, &delta, 4, cudaMemcpyHostToDevice));
}

void Worker::prefill_chunk(const int* ids, int i0, int m, int n, int k, const int* rope3) {
    const bool last = i0 + m == n;
    pf_p0_ = i0;
    CUDA_CHECK(cudaMemcpyAsync(tokens_, ids + i0, m * 4, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(pos_, &i0, 4, cudaMemcpyHostToDevice, stream));
    // multimodal prompt: (t, h, w) RoPE positions of this chunk's rows (header: rows, first index)
    std::vector<int> r3;
    if (rope3) {
        CHECK(m <= kChunk, "chunk %d", m);
        r3.resize(2 + 3 * (size_t)m);
        r3[0] = m;
        r3[1] = i0;
        std::copy(rope3 + 3 * (size_t)i0, rope3 + 3 * (size_t)(i0 + m), r3.begin() + 2);
        CUDA_CHECK(cudaMemcpyAsync(rope3_, r3.data(), r3.size() * 4, cudaMemcpyHostToDevice, stream));
    }
    static const bool overlap = !getenv("QE_PREFILL_OVERLAP") || atoi(getenv("QE_PREFILL_OVERLAP"));
    // a short pass without a cached graph, first use since the slot's graphs were dropped (the
    // prompt-end token of a long prompt: the graphs go whenever the slot's region moves, so nearly
    // every new prompt captured and instantiated a whole-model graph for a single launch): launch
    // it directly; a repeated key is captured as before (QE_PF_EAGER=0: always a graph)
    const auto gkey = std::make_tuple(m, last, mode(), cur_slot_);
    if (m <= kMaxGemvTokens && env_int("QE_PF_EAGER", 1) && !debug_hidden && !graphs_.m.count(gkey) &&
        eager_seen_.insert(gkey).second)
        forward(m, last);
    else if (m <= kMaxGemvTokens) run(m, last);
    else if (overlap && m >= overlap_min() && !debug_hidden) forward_prefill(m, last);
    else forward(m, last);  // big chunks: launch overhead is negligible, no graph
    if (dump_hidden && rank == 0)  // training data for the MTP head (needs k > 0: all rows normed)
        CUDA_CHECK(cudaMemcpyAsync(dump_hidden + (size_t)i0 * c_.hidden, xn_, (size_t)m * c_.hidden * 2,
                                   cudaMemcpyDeviceToHost, stream));
    if (has_mtp && k > 0) {
        // pairs (h_i, x_{i+1}); the token after the prompt comes from the main model's argmax
        const int known = last ? m - 1 : m;
        if (known > 0)
            CUDA_CHECK(cudaMemcpyAsync(spec_.mtp_tok, ids + i0 + 1, known * 4,
                                       cudaMemcpyHostToDevice, stream));
        if (last) spec_copy_last(spec_, m, stream);
        const bool mtp_ov = env_int("QE_MTP_OVERLAP", 1) != 0;
        if (dbg_skip() & 4) {
        } else if (mtp_ov && overlap && m >= overlap_min() && !debug_hidden && !rope3)
            mtp_forward_prefill(m, last);
        else
            mtp_forward(m, pos_, spec_.mtp_tok, xn_, nullptr, last);
        if (last) {
            spec_prefill_select(spec_, m, n, stream);
            for (int r = 1; r < k; r++) {
                mtp_forward(1, spec_.mtp_pos, spec_.mtp_tok, mtp_h_, spec_.hrow, true, mtp_draft_win());
                spec_append(spec_, r, stream);
            }
            spec_finish_prefill(spec_, k, stream);
            CUDA_CHECK(cudaMemcpyAsync(pos_, &n, 4, cudaMemcpyHostToDevice, stream));
        }
    }
    if (rope3) CUDA_CHECK(cudaMemsetAsync(rope3_, 0, 4, stream));  // later passes: text positions
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

void Worker::lk_init(int cap, int mmin, int lmax) {
    CUDA_CHECK(cudaSetDevice(device));
    lk_dev_.cap = cap;
    lk_dev_.mmin = mmin;
    lk_dev_.lmax = lmax;
    lk_dev_.hd = (int*)dalloc((size_t)cap * 4, false);
    lk_dev_.hn = (int*)dalloc(4, false);
    lk_dev_.best = (unsigned long long*)dalloc(8, false);
    lk_dev_.lk = (int*)dalloc(32 * 4, false);
    CUDA_CHECK(cudaMemset(lk_dev_.hn, 0, 4));
    CUDA_CHECK(cudaMemset(lk_dev_.best, 0, 8));
    CUDA_CHECK(cudaMemset(lk_dev_.lk, 0, 32 * 4));
    CUDA_CHECK(cudaStreamCreateWithFlags(&lk_cst_, cudaStreamNonBlocking));
    // batched steps: a history per slot (QE_LKB=0: off)
    if (!getenv("QE_LKB") || atoi(getenv("QE_LKB"))) {
        lkb_cap_ = cap;
        for (auto& sl : slots_) lkb_alloc(sl);
    }
}

void Worker::lkb_alloc(SeqSlot& sl) {
    LookupDev& d = sl.lk;
    d.cap = lkb_cap_;
    d.mmin = lk_dev_.mmin;
    d.lmax = lk_dev_.lmax;
    d.hd = (int*)dalloc((size_t)d.cap * 4, false);
    d.hn = (int*)dalloc(4, false);
    d.best = (unsigned long long*)dalloc(8, false);
    d.lk = (int*)dalloc(32 * 4, false);
    CUDA_CHECK(cudaMemset(d.hn, 0, 4));
    CUDA_CHECK(cudaMemset(d.best, 0, 8));
    CUDA_CHECK(cudaMemset(d.lk, 0, 32 * 4));
}

void Worker::lkb_sync(int i, const int* tail, int from, int n) {
    CUDA_CHECK(cudaSetDevice(device));
    const LookupDev& d = slots_[i].lk;
    n = std::min(n, d.cap);
    if (n > from)  // pinned source (Engine::lkb_pin_): an asynchronous copy, the host never waits
        CUDA_CHECK(cudaMemcpyAsync(d.hd + from, tail, (size_t)(n - from) * 4, cudaMemcpyHostToDevice, stream));
    lookup_set_len_scan(d, n, stream);
}

void Worker::lk_set_history(const int* ids, int n) {
    if (!lk_on()) return;
    CUDA_CHECK(cudaSetDevice(device));
    n = std::min(n, lk_dev_.cap);
    CUDA_CHECK(cudaMemcpyAsync(lk_dev_.hd, ids, (size_t)n * 4, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(lk_dev_.hn, &n, 4, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemsetAsync(lk_dev_.best, 0, 8, stream));
    lookup_scan_emit(lk_dev_, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

void Worker::run_spec(int kv, int kd, bool launch, bool lookup) {
    auto key = std::make_tuple(kv, kd | (lookup ? 256 : 0), mode(), cur_slot_);
    cudaGraphExec_t sge = spec_graphs_.get(key);
    if (!sge) {
        cudaGraph_t g;
        CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
        const bool defer = dn_pn_ != nullptr && dn_defer1_on();
        if (dn_pn_ && !defer) {  // pending rows of a batched step first, then as on master
            dn_flush_cur(stream);
            dn_single_begin(dn_cur_, dn_save_, dn_pn_, stream);
        } else if (!defer) {
            copy_int(dn_save_, dn_cur_, stream);
        }
        if (lookup) lookup_to_tokens(lk_dev_, spec_.tokens, kv, stream);  // the previous step's lookup drafts
        forward(kv + 1, true, true);
        if (sampling) spec_accept_sampled(topk_all_, ngpu_, kv, sp_, spec_, stream);
        else spec_accept(spec_, kv, stream);
        if (defer) {  // accepted rows -> stash (committed by the next step)
            DnStashSegs r{};
            r.n = 1;
            int li = 0;
            for (auto& L : layers_)
                if (!L.full) {
                    r.cv[li] = spec_cv_[li];
                    r.qk[li] = spec_qk_[li];
                    r.gb[li] = spec_gb_[li];
                    li++;
                }
            r.r0[0] = 0;
            r.cur[0] = dn_cur_;
            r.pn[0] = dn_pn_;
            r.pend[0] = dn_pend_;
            r.acc[0] = spec_.acc;
            r.save[0] = dn_save_;
            r.m[0] = kv + 1;
            dn_stash_seg(r, li, dn_ch_, dsh_, stream);
        } else {
            DnReplay rp{};
            int li = 0;
            for (auto& L : layers_)
                if (!L.full) {
                    rp.S[li] = L.dn.S;
                    rp.cv[li] = spec_cv_[li];
                    rp.qk[li] = spec_qk_[li];
                    rp.gb[li] = spec_gb_[li];
                    li++;
                }
            rp.n = li;
            dn_replay(rp, dn_ch_, dsh_, dn_slot_floats_, dn_cur_, spec_.acc, stream);
            dn_commit(dn_cur_, nullptr, 1, stream);
        }
        // kd == 0: the MTP pass only fills its KV cache, no draft head
        mtp_forward(kv + 1, spec_.mtp_pos, spec_.mtp_tok, xn_, spec_.hrow, kd > 0);
        spec_select(spec_, stream);
        auto chain = [&]() {
            for (int r = 1; r < kd; r++) {
                mtp_forward(1, spec_.mtp_pos, spec_.mtp_tok, mtp_h_, spec_.hrow, true, mtp_draft_win());
                spec_append(spec_, r, stream);
            }
        };
        if (lookup && lk_on() && kd > 1) {
            // lookup of this step first; the chain (drafts 2..kd) in an IF node: only when no lookup drafts follow
            // (the emitted tokens are appended to the history before the scan, so the scan runs after accept)
            lookup_append(lk_dev_, spec_.emit, stream);
            lookup_scan_emit(lk_dev_, stream, spec_.emit + 31);
            cudaStreamCaptureStatus cs;
            cudaGraph_t cg;
            const cudaGraphNode_t* deps;
            size_t nd;
            CUDA_CHECK(cudaStreamGetCaptureInfo(stream, &cs, nullptr, &cg, nullptr, nullptr, nullptr));
            cudaGraphConditionalHandle h;
            CUDA_CHECK(cudaGraphConditionalHandleCreate(&h, cg, 0, 0));
            lookup_cond(lk_dev_, h, stream);
            CUDA_CHECK(cudaStreamGetCaptureInfo(stream, &cs, nullptr, &cg, &deps, nullptr, &nd));
            cudaGraphNodeParams cp = {};
            cp.type = cudaGraphNodeTypeConditional;
            cp.conditional.handle = h;
            cp.conditional.type = cudaGraphCondTypeIf;
            cp.conditional.size = 1;
            cudaGraphNode_t node;
            CUDA_CHECK(cudaGraphAddNode(&node, cg, deps, nullptr, nd, &cp));
            CUDA_CHECK(cudaStreamUpdateCaptureDependencies(stream, &node, nullptr, 1, cudaStreamSetCaptureDependencies));
            cudaGraph_t body = cp.conditional.phGraph_out[0];
            cudaStream_t s0 = stream;
            stream = lk_cst_;
            CUDA_CHECK(cudaStreamBeginCaptureToGraph(stream, body, nullptr, nullptr, 0, cudaStreamCaptureModeThreadLocal));
            chain();
            CUDA_CHECK(cudaStreamEndCapture(stream, &body));
            stream = s0;
            spec_finish(spec_, kd, stream);
        } else {
            chain();
            spec_finish(spec_, kd, stream);
            if (lk_on()) {  // drafts for the next step (their count at emit[31])
                lookup_append(lk_dev_, spec_.emit, stream);
                lookup_scan_emit(lk_dev_, stream, spec_.emit + 31);
            }
        }
        CUDA_CHECK(cudaStreamEndCapture(stream, &g));
        instantiate(&sge, g);
        CUDA_CHECK(cudaGraphDestroy(g));
        spec_graphs_.put(key, sge);
    }
    if (launch) CUDA_CHECK(cudaGraphLaunch(sge, stream));
}
