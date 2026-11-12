// DFlash2 block drafter of the 27B: the Worker side. Weights replicated on every card
// (no collectives except the top-16 gather of a sharded draft head), our W4 / W16 GEMVs and the kernels of
// kernels/dflash.cu. The drafter's context ring holds K / V of the last ~2K positions per drafter layer, written from
// the residual taps of the rows the target computes (verification steps, prompt chunks in the window).
#include <chrono>
#include <cmath>
#include <cstring>
#include <functional>
#include <vector>

#include "model.h"

namespace {
constexpr int kTapLayers[kDflTaps] = {5, 19, 33, 47, 61};
constexpr float kEps = 1e-6f;
constexpr int kQkv = (kDflNq + 2 * kDflNkv) * kDflHd;  // 6144
constexpr int kInter = 17408;
constexpr int kCtxRows = 16;  // context rows per GEMV pass (XH scratch of mm4 holds 64 x 10240 halves)

uint16_t f16_to_bf16(uint16_t h) {
    const uint32_t s = (h & 0x8000u) << 16, e = (h >> 10) & 31, m = h & 1023;
    float f;
    if (e == 0) f = std::ldexp((float)m, -24);
    else if (e == 31) f = m ? NAN : INFINITY;
    else f = std::ldexp((float)(m | 1024), (int)e - 25);
    uint32_t u;
    memcpy(&u, &f, 4);
    u |= s;
    u += 0x7FFF + ((u >> 16) & 1);  // round to nearest even
    return (uint16_t)(u >> 16);
}
// an allocation of the drafter's load failed (VRAM taken meanwhile): the load gives up, the engine drafts with MTP
struct DflNoMem {
    size_t bytes;
};
void* dfl_tmp_alloc(size_t b) {
    void* p = nullptr;
    if (cudaMalloc(&p, b) != cudaSuccess) {
        (void)cudaGetLastError();
        throw DflNoMem{b};
    }
    return p;
}
struct TmpBuf {  // a load temporary, freed on every path
    void* p = nullptr;
    explicit TmpBuf(size_t b) : p(dfl_tmp_alloc(b)) {}
    ~TmpBuf() { if (p) cudaFree(p); }
    TmpBuf(const TmpBuf&) = delete;
    TmpBuf& operator=(const TmpBuf&) = delete;
};
}  // namespace

struct Worker::Dfl {
    std::vector<void*> mem;      // every device allocation of the drafter (freed by dfl_unload)
    size_t wb = 0, cb = 0;       // bytes counted into the worker's weight / cache totals
    struct L {
        bf16 *in_ln, *post_ln, *qn, *kn, *acb, *mcb;
        W16 akp, mkp;
        W8 akp8, mkp8;  // (kp8)
        W4 qkv, kv, o, gu, down;
    } L[kDflLayers];
    W4 fc;
    W16 selh;
    bf16 *hn, *norm;
    float* inv;
    const bf16 *pred_cb, *succ_cb;
    bool full_head = false;
    bool tp = false;
    bool kp8 = false;  // kernel projections INT8
    int cb_rows = 0;   // candidate ids below it
    bool bcast = false;  // TP: rank 0's drafts on every card (review L1)
    float* tokg = nullptr;  // their gather [ngpu][8]
    int ib = 0, ie = kInter;  // this card's MLP rows
    float* red = nullptr;     // all-reduce target [8][H]
    int head_v = 0, head_off = 0;
    const int* head_map = nullptr;
    // buffers
    bf16* tap;            // [kChunk][5][H]
    float *y, *kvp;       // context: [kCtxRows][H], [kCtxRows][2048]
    bf16* th;             // [kCtxRows][H]
    float *x, *dyn, *proj, *ov, *work, *logits, *tk, *pairs, *pairs_all, *unary, *hp;
    bf16 *h, *hc, *q, *kb, *vb, *att, *act, *hid, *pc, *sc;
    int *tok, *cand;
    std::vector<DflRing> rings;  // host copies of the descriptors
    DflRing* ring_all = nullptr; // device [n]
    DflRing* ring_cur = nullptr; // device: the selected one (read by the kernels)
    size_t ring_bytes = 0, wbytes = 0;
};

size_t Worker::dfl_weight_bytes(const SafeTensors& st, int ngpu) {
    // the MLP is split over the cards (QE_DF_TP=1, dfl_load): the largest card's share (review L9)
    const bool tp = ngpu > 1 && env_int("QE_DF_TP", 1) != 0;
    const double mlp = tp ? (double)((kInter / 128 + ngpu - 1) / ngpu) * 128 / kInter : 1.0;
    size_t b = 0;
    for (int l = 0; l < kDflLayers; l++) {
        const std::string p = "layers." + std::to_string(l) + ".";
        for (const char* n : {"self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj"})
            b += st.get(p + n + ".weight_packed").nbytes + st.get(p + n + ".weight_scale").nbytes;
        for (const char* n : {"mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"})
            b += (size_t)(mlp * (st.get(p + n + ".weight_packed").nbytes + st.get(p + n + ".weight_scale").nbytes));
        const size_t kp = st.get(p + "attention_conv.kernel_projection.weight").nbytes + st.get(p + "mlp_conv.kernel_projection.weight").nbytes;
        b += env_int("QE_DF_KP8", 1) ? kp / 2 + kp / 128 : kp;  // INT8 + group scales
    }
    b += st.get("fc.weight_packed").nbytes + st.get("fc.weight_scale").nbytes;
    b += st.get("candidate_selector.hidden_projection.weight").nbytes;
    return b + ((size_t)kChunk * kDflTaps * kDflHid * 2) + ((size_t)96 << 20);  // tap buffer + small buffers
}

std::string Worker::dfl_schema(const SafeTensors& st, int vocab) {
    auto chk = [&](const std::string& n, std::vector<long long> shape, std::initializer_list<const char*> dt) -> std::string {
        if (!st.has(n)) return "missing tensor " + n;
        const TensorInfo& t = st.get(n);
        bool ok = false;
        for (const char* x : dt) ok |= t.dtype == x;
        if (!ok) return n + ": dtype " + t.dtype;
        if (shape.size() != t.shape.size()) return n + ": rank " + std::to_string(t.shape.size());
        for (size_t i = 0; i < shape.size(); i++)
            if (shape[i] >= 0 ? t.shape[i] != shape[i] : t.shape[i] < -shape[i]) return n + ": shape dim " + std::to_string(i) + " = " + std::to_string(t.shape[i]);
        return "";
    };
    auto w4 = [&](const std::string& n, long long rows, long long k) {
        std::string e = chk(n + ".weight_packed", {rows, k / 8}, {"I32"});
        return e.empty() ? chk(n + ".weight_scale", {rows, k / 128}, {"F16", "BF16"}) : e;
    };
    const long long H = kDflHid;
    std::vector<std::string> errs = {
        w4("fc", H, kDflTaps * H), chk("hidden_norm.weight", {H}, {"BF16"}), chk("norm.weight", {H}, {"BF16"}),
        chk("candidate_selector.hidden_projection.weight", {kDflRank, H}, {"BF16"}),
        chk("candidate_selector.predecessor_codebook", {-(long long)vocab, kDflRank}, {"BF16"}),
        chk("candidate_selector.successor_codebook", {-(long long)vocab, kDflRank}, {"BF16"})};
    for (int l = 0; l < kDflLayers; l++) {
        const std::string p = "layers." + std::to_string(l) + ".";
        for (const char* n : {"input_layernorm.weight", "post_attention_layernorm.weight"}) errs.push_back(chk(p + n, {H}, {"BF16"}));
        for (const char* n : {"self_attn.q_norm.weight", "self_attn.k_norm.weight"}) errs.push_back(chk(p + n, {kDflHd}, {"BF16"}));
        for (const char* n : {"attention_conv.base_kernel", "mlp_conv.base_kernel"}) errs.push_back(chk(p + n, {2, 2, H}, {"BF16"}));
        for (const char* n : {"attention_conv.kernel_projection.weight", "mlp_conv.kernel_projection.weight"})
            errs.push_back(chk(p + n, {kDflKp, H}, {"BF16"}));
        errs.push_back(w4(p + "self_attn.q_proj", kDflNq * kDflHd, H));
        errs.push_back(w4(p + "self_attn.k_proj", kDflNkv * kDflHd, H));
        errs.push_back(w4(p + "self_attn.v_proj", kDflNkv * kDflHd, H));
        errs.push_back(w4(p + "self_attn.o_proj", H, kDflNq * kDflHd));
        errs.push_back(w4(p + "mlp.gate_proj", kInter, H));
        errs.push_back(w4(p + "mlp.up_proj", kInter, H));
        errs.push_back(w4(p + "mlp.down_proj", H, kInter));
    }
    if (st.has("layers." + std::to_string(kDflLayers) + ".input_layernorm.weight")) errs.push_back("more than 5 layers");
    for (auto& e : errs)
        if (!e.empty()) return e;
    return "";
}

size_t Worker::dfl_ring_bytes() const { return (size_t)kDflLayers * 2 * kDflRing * kDflNkv * kDflHd * 2 + kDflRing * 4; }

// a compressed-tensors W4A16 matrix (rows [r0, r1) of several tensors concatenated, columns [c0, c1)) with fp16 group
// scales -> our W4 layout
static W4 dfl_w4(const SafeTensors& st, const std::vector<std::string>& names, bool glu, cudaStream_t s,
                 const std::function<void*(size_t)>& alloc, int r0 = -1, int r1 = -1, int c0 = -1, int c1 = -1) {
    const TensorInfo& t0 = st.get(names[0] + ".weight_packed");
    const int K = (int)t0.shape[1] * 8;
    if (c0 < 0) c0 = 0, c1 = K;
    const int kl = c1 - c0;
    CHECK(c0 % 128 == 0 && kl % 128 == 0, "dflash w4 %s columns %d..%d", names[0].c_str(), c0, c1);
    int n = 0;
    for (auto& nm : names) {
        const TensorInfo& q = st.get(nm + ".weight_packed");
        CHECK(q.dtype == "I32" && q.shape[1] * 8 == K, "%s: packed shape", nm.c_str());
        n += r0 < 0 ? (int)q.shape[0] : r1 - r0;
    }
    CHECK(n % 16 == 0, "dflash w4 %s n %d", names[0].c_str(), n);
    std::vector<int32_t> hq((size_t)n * kl / 8);
    std::vector<uint16_t> hs((size_t)n * (kl / 128));
    size_t row = 0;
    for (auto& nm : names) {
        const TensorInfo& q = st.get(nm + ".weight_packed");
        const TensorInfo& sc = st.get(nm + ".weight_scale");
        CHECK(sc.shape[1] * 128 == K && (sc.dtype == "F16" || sc.dtype == "BF16"), "%s: scales", nm.c_str());
        const int a = r0 < 0 ? 0 : r0, b = r0 < 0 ? (int)q.shape[0] : r1;
        const int32_t* qs = (const int32_t*)q.data;
        const uint16_t* ss = (const uint16_t*)sc.data;
        for (int r = a; r < b; r++, row++) {
            memcpy(hq.data() + row * (kl / 8), qs + (size_t)r * (K / 8) + c0 / 8, (size_t)kl / 2);
            for (int g = 0; g < kl / 128; g++) {
                const uint16_t v = ss[(size_t)r * (K / 128) + c0 / 128 + g];
                hs[row * (kl / 128) + g] = sc.dtype == "BF16" ? v : f16_to_bf16(v);
            }
        }
    }
    TmpBuf tqb(hq.size() * 4), tsb(hs.size() * 2);
    int32_t* tq = (int32_t*)tqb.p;
    bf16* ts = (bf16*)tsb.p;
    CUDA_CHECK(cudaMemcpyAsync(tq, hq.data(), hq.size() * 4, cudaMemcpyHostToDevice, s));
    CUDA_CHECK(cudaMemcpyAsync(ts, hs.data(), hs.size() * 2, cudaMemcpyHostToDevice, s));
    W4 w;
    w.n = n;
    w.k = kl;
    w.glu = glu;
    w.bits = 4;
    uint4* dq = (uint4*)alloc(w4_qbytes(n, kl));
    bf16* ds = (bf16*)alloc(w4_sbytes(n, kl));
    if (glu) {
        TmpBuf gqb(hq.size() * 4), gsb(hs.size() * 2);
        glu_interleave_rows(tq, ts, n, kl, (int32_t*)gqb.p, (bf16*)gsb.p, s, 4);
        repack_w4((int32_t*)gqb.p, (bf16*)gsb.p, n, kl, dq, ds, s);
        CUDA_CHECK(cudaStreamSynchronize(s));
    } else {
        repack_w4(tq, ts, n, kl, dq, ds, s);
    }
    CUDA_CHECK(cudaStreamSynchronize(s));
    w.q = dq;
    w.s = ds;
    return w;
}

static W16 dfl_w16(const SafeTensors& st, const std::string& name, cudaStream_t s, const std::function<void*(size_t)>& alloc) {
    const TensorInfo& t = st.get(name);
    CHECK(t.dtype == "BF16" && t.shape.size() == 2, "%s", name.c_str());
    const int n = (int)t.shape[0], k = (int)t.shape[1];
    TmpBuf tb(t.nbytes);
    bf16* tmp = (bf16*)tb.p;
    CUDA_CHECK(cudaMemcpyAsync(tmp, t.data, t.nbytes, cudaMemcpyHostToDevice, s));
    W16 w;
    w.n = n;
    w.k = k;
    uint4* d = (uint4*)alloc(w16_bytes(n, k));
    repack_w16(tmp, n, k, d, s);
    CUDA_CHECK(cudaStreamSynchronize(s));
    w.w = d;
    return w;
}

// a bf16 matrix as RTN INT8 (group 128) in the W8 layout (the conv kernel projections, QE_DF_KP8=1)
static W8 dfl_w8(const SafeTensors& st, const std::string& name, cudaStream_t s, const std::function<void*(size_t)>& alloc) {
    const TensorInfo& t = st.get(name);
    CHECK(t.dtype == "BF16" && t.shape.size() == 2, "%s", name.c_str());
    const int n = (int)t.shape[0], k = (int)t.shape[1];
    TmpBuf tb(t.nbytes);
    bf16* tmp = (bf16*)tb.p;
    CUDA_CHECK(cudaMemcpyAsync(tmp, t.data, t.nbytes, cudaMemcpyHostToDevice, s));
    W8 w;
    w.n = n;
    w.k = k;
    uint4* q = (uint4*)alloc(w8_qbytes(n, k));
    bf16* sc = (bf16*)alloc((size_t)n * (k / 128) * 2);
    quantize_w8(tmp, n, k, q, sc, s);
    CUDA_CHECK(cudaStreamSynchronize(s));
    w.q = q;
    w.s = sc;
    return w;
}

bool Worker::dfl_load(const SafeTensors& st, const bf16* pred_cb, const bf16* succ_cb, int rings, bool full_head, int cb_rows) {
    CUDA_CHECK(cudaSetDevice(device));
    Dfl* d = new Dfl();
    // non-fatal allocations (review M4): VRAM taken between the engine's check and here ends the load, never the server
    auto walloc = [&](size_t b) { void* p = dfl_tmp_alloc(b); d->mem.push_back(p); d->wb += b; return p; };
    auto calloc_ = [&](size_t b) { void* p = dfl_tmp_alloc(b); d->mem.push_back(p); d->cb += b; return p; };
    auto bfv = [&](const std::string& name) {
        const TensorInfo& t = st.get(name);
        CHECK(t.dtype == "BF16", "%s dtype %s", name.c_str(), t.dtype.c_str());
        bf16* p = (bf16*)walloc(t.nbytes);
        CUDA_CHECK(cudaMemcpy(p, t.data, t.nbytes, cudaMemcpyHostToDevice));
        return p;
    };
    try {
    // MLP split over the cards (QE_DF_TP=1, default with more than one card): rows of gate / up and columns of down in
    // 128-column groups, one all-reduce per layer; the rest of the drafter is replicated
    d->tp = ngpu_ > 1 && env_int("QE_DF_TP", 1) != 0;
    d->ib = d->tp ? (int)((long)(kInter / 128) * rank / ngpu_) * 128 : 0;
    d->ie = d->tp ? (int)((long)(kInter / 128) * (rank + 1) / ngpu_) * 128 : kInter;
    d->kp8 = env_int("QE_DF_KP8", 1) != 0;  // (INT8: the same drafts on the short set, -66 MB, -0.02 ms)
    d->fc = dfl_w4(st, {"fc"}, false, stream, walloc);
    CHECK(d->fc.k == kDflTaps * kDflHid && d->fc.n == kDflHid, "dflash fc %d x %d", d->fc.n, d->fc.k);
    d->hn = bfv("hidden_norm.weight");
    d->norm = bfv("norm.weight");
    d->selh = dfl_w16(st, "candidate_selector.hidden_projection.weight", stream, walloc);
    for (int l = 0; l < kDflLayers; l++) {
        const std::string p = "layers." + std::to_string(l) + ".";
        auto& L = d->L[l];
        L.in_ln = bfv(p + "input_layernorm.weight");
        L.post_ln = bfv(p + "post_attention_layernorm.weight");
        L.qn = bfv(p + "self_attn.q_norm.weight");
        L.kn = bfv(p + "self_attn.k_norm.weight");
        L.acb = bfv(p + "attention_conv.base_kernel");
        L.mcb = bfv(p + "mlp_conv.base_kernel");
        if (d->kp8) {
            L.akp8 = dfl_w8(st, p + "attention_conv.kernel_projection.weight", stream, walloc);
            L.mkp8 = dfl_w8(st, p + "mlp_conv.kernel_projection.weight", stream, walloc);
            CHECK(L.akp8.n == kDflKp && L.akp8.k == kDflHid, "dflash kernel projection %d x %d", L.akp8.n, L.akp8.k);
        } else {
            L.akp = dfl_w16(st, p + "attention_conv.kernel_projection.weight", stream, walloc);
            L.mkp = dfl_w16(st, p + "mlp_conv.kernel_projection.weight", stream, walloc);
            CHECK(L.akp.n == kDflKp && L.akp.k == kDflHid, "dflash kernel projection %d x %d", L.akp.n, L.akp.k);
        }
        L.qkv = dfl_w4(st, {p + "self_attn.q_proj", p + "self_attn.k_proj", p + "self_attn.v_proj"}, false, stream, walloc);
        CHECK(L.qkv.n == kQkv && L.qkv.k == kDflHid, "dflash qkv %d x %d", L.qkv.n, L.qkv.k);
        // view of the k | v rows (context projections): whole 16-row tiles from row 4096
        L.kv = L.qkv;
        L.kv.n = 2 * kDflNkv * kDflHd;
        L.kv.q = reinterpret_cast<const uint4*>(reinterpret_cast<const char*>(L.qkv.q) + (size_t)kDflNq * kDflHd * kDflHid / 2);
        L.kv.s = L.qkv.s + (size_t)kDflNq * kDflHd * (kDflHid / 128);
        L.o = dfl_w4(st, {p + "self_attn.o_proj"}, false, stream, walloc);
        L.gu = dfl_w4(st, {p + "mlp.gate_proj", p + "mlp.up_proj"}, true, stream, walloc, d->ib, d->ie);
        L.down = dfl_w4(st, {p + "mlp.down_proj"}, false, stream, walloc, -1, -1, d->ib, d->ie);
        CHECK(L.gu.n == 2 * (d->ie - d->ib) && L.down.k == d->ie - d->ib, "dflash mlp %d / %d", L.gu.n, L.down.k);
    }
    {
        std::vector<float> inv(kDflHd / 2);
        for (int i = 0; i < kDflHd / 2; i++) inv[i] = (float)(1.0 / std::pow(1e7, (double)(2 * i) / kDflHd));
        d->inv = (float*)walloc(inv.size() * 4);
        CUDA_CHECK(cudaMemcpy(d->inv, inv.data(), inv.size() * 4, cudaMemcpyHostToDevice));
    }
    d->pred_cb = pred_cb;
    d->cb_rows = std::min(cb_rows, c_.vocab);
    d->bcast = ngpu_ > 1 && env_int("QE_DF_BCAST", 1) != 0;
    d->succ_cb = succ_cb;
    d->full_head = full_head || draft_v_ == 0;
    if (d->full_head) {
        d->head_v = p_.vocab.size();
        d->head_off = p_.vocab.b;
        d->head_map = nullptr;
    } else {
        d->head_v = draft_v_;
        d->head_off = p_.vocab.b;
        d->head_map = draft_map_;
    }
    // buffers
    const int H = kDflHid;
    auto fb = [&](size_t n) { return (float*)calloc_(n * 4); };
    auto bb = [&](size_t n) { return (bf16*)calloc_(n * 2); };
    d->tap = bb((size_t)kChunk * kDflTaps * H);
    d->y = fb((size_t)kCtxRows * H);
    d->th = bb((size_t)kCtxRows * H);
    d->kvp = fb((size_t)kCtxRows * 2 * kDflNkv * kDflHd);
    d->x = fb((size_t)kDflBlock * H);
    d->dyn = fb((size_t)kDflBlock * kDflKp);
    d->proj = fb((size_t)kDflBlock * kQkv);
    d->ov = fb((size_t)kDflBlock * H);
    d->red = fb((size_t)kDflBlock * H);
    d->work = fb(dfl_attn_work_floats());
    d->logits = fb((size_t)kDflBlock * d->head_v);
    d->tk = fb(dfl_topk_chunk_floats(d->head_v, kDflBlock));
    d->pairs = fb((size_t)kDflBlock * kDflTopK * 2);
    d->pairs_all = fb((size_t)ngpu_ * kDflBlock * kDflTopK * 2);
    d->tokg = fb((size_t)ngpu_ * kDflBlock);
    d->unary = fb((size_t)kDflBlock * kDflTopK);
    d->hp = fb((size_t)kDflBlock * kDflRank);
    d->h = bb((size_t)kDflBlock * H);
    d->hc = bb((size_t)kDflBlock * H);
    d->q = bb((size_t)kDflBlock * kDflNq * kDflHd);
    d->kb = bb((size_t)kDflBlock * kDflNkv * kDflHd);
    d->vb = bb((size_t)kDflBlock * kDflNkv * kDflHd);
    d->att = bb((size_t)kDflBlock * kDflNq * kDflHd);
    d->act = bb((size_t)kDflBlock * kInter);
    d->hid = bb((size_t)kDflBlock * H);
    d->pc = bb((size_t)(1 + kDflBlock * kDflTopK) * kDflRank);
    d->sc = bb((size_t)kDflBlock * kDflTopK * kDflRank);
    d->tok = (int*)calloc_(kDflBlock * 4);
    d->cand = (int*)calloc_(kDflBlock * kDflTopK * 4);
    // rings
    const size_t kvb = (size_t)kDflRing * kDflNkv * kDflHd * 2;
    for (int r = 0; r < rings; r++) {
        DflRing g;
        for (int l = 0; l < kDflLayers; l++) g.k[l] = (bf16*)calloc_(kvb), g.v[l] = (bf16*)calloc_(kvb);
        g.tag = (int*)calloc_(kDflRing * 4);
        CUDA_CHECK(cudaMemset(g.tag, 0xff, kDflRing * 4));
        d->rings.push_back(g);
    }
    d->ring_all = (DflRing*)calloc_(sizeof(DflRing) * std::max(1, rings));
    d->ring_cur = (DflRing*)calloc_(sizeof(DflRing));
    CUDA_CHECK(cudaMemcpy(d->ring_all, d->rings.data(), sizeof(DflRing) * rings, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d->ring_cur, d->rings.data(), sizeof(DflRing), cudaMemcpyHostToDevice));
    d->ring_bytes = dfl_ring_bytes();
    d->wbytes = d->wb;
    CUDA_CHECK(cudaDeviceSynchronize());
    if (env_int("QE_DF_TEST_OOM", -1) == rank) throw DflNoMem{1};  // test only: this card's load fails at its end
    } catch (const DflNoMem& e) {
        CUDA_CHECK(cudaDeviceSynchronize());
        for (void* p : d->mem) cudaFree(p);
        fprintf(stderr, "GPU%d: DFlash2 drafter: out of memory loading it (%.1f MB more requested after %.2f GB): MTP drafts\n", device,
                e.bytes / 1048576.0, (d->wb + d->cb) / 1e9);
        delete d;
        return false;
    }
    weight_bytes += d->wb;
    cache_bytes += d->cb;
    dfl_ = d;
    if (rank == 0)
        printf("DFlash2 drafter: %.2f GB weights + %.2f GB buffers and %d context rings per card, MLP %s, head %s (%d rows per card)\n",
               d->wb / 1e9, d->cb / 1e9, rings, d->tp ? "split over the cards" : "on every card", d->full_head ? "INT8 verify head" : "INT4 draft vocabulary",
               d->head_v);
    return true;
}

void Worker::dfl_unload() {
    if (!dfl_) return;
    CUDA_CHECK(cudaSetDevice(device));
    CUDA_CHECK(cudaDeviceSynchronize());
    drop_graphs();  // (graphs may hold the drafter's buffers)
    for (void* p : dfl_->mem) cudaFree(p);
    weight_bytes -= dfl_->wb;
    cache_bytes -= dfl_->cb;
    delete dfl_;
    dfl_ = nullptr;
}

void Worker::dfl_ring_select(int ring) {
    CUDA_CHECK(cudaMemcpyAsync(dfl_->ring_cur, dfl_->ring_all + ring, sizeof(DflRing), cudaMemcpyDeviceToDevice, stream));
}

void Worker::dfl_ring_inval(int from) { ::dfl_ring_inval(dfl_->ring_cur, nullptr, std::max(0, from), stream); }

void Worker::dfl_tap_layer(int i, int m, int r0) {
    for (int j = 0; j < kDflTaps; j++)
        if (kTapLayers[j] == i) dfl_tap(resid_ + (size_t)r0 * c_.hidden, m, j, dfl_->tap, r0, stream);
}

void Worker::dfl_ctx_rows(int row0, int m, const int* pos) {
    Dfl& d = *dfl_;
    const int H = kDflHid;
    cudaStream_t st = stream;
    for (int r = 0; r < m; r += kCtxRows) {
        const int mb = std::min(kCtxRows, m - r);
        mm4(d.fc, d.tap + (size_t)(row0 + r) * kDflTaps * H, kDflTaps * H, d.y, H, mb, st);
        dfl_rms(d.y, H, d.hn, mb, kEps, d.th, st);
        for (int l = 0; l < kDflLayers; l++) {
            mm4(d.L[l].kv, d.th, H, d.kvp, 2 * kDflNkv * kDflHd, mb, st);
            dfl_kv_ctx(d.kvp, d.L[l].kn, d.inv, pos, row0 + r, mb, kEps, d.ring_cur, l, l == kDflLayers - 1, st);
        }
    }
}

void Worker::dfl_block() {
    Dfl& d = *dfl_;
    const int H = kDflHid, B = kDflBlock, M = kDflDrafts;
    cudaStream_t st = stream;
    if (ngpu_ > 1) begin_unit(st);  // (LL epoch of the drafter's collectives)
    dfl_set_block_tokens(d.tok, spec_.tokens, st);
    embed_gather(embed_, d.tok, B, H, d.x, st);
    for (int l = 0; l < kDflLayers; l++) {
        auto& L = d.L[l];
        dfl_rms(d.x, H, L.in_ln, B, kEps, d.h, st);
        if (d.kp8) gemv_w8(L.akp8, d.h, H, d.dyn, kDflKp, B, st);
        else gemv_w16(L.akp, d.h, H, d.dyn, kDflKp, B, st);
        dfl_conv(nullptr, d.h, d.dyn, 0, L.acb, B, d.hc, nullptr, st);
        mm4(L.qkv, d.hc, H, d.proj, kQkv, B, st);
        dfl_qkv_block(d.proj, L.qn, L.kn, d.inv, spec_.pos, B, kEps, d.q, d.kb, d.vb, st);
        dfl_attn(d.q, d.kb, d.vb, d.ring_cur, l, spec_.pos, d.work, d.att, st);
        mm4(L.o, d.att, kDflNq * kDflHd, d.ov, H, B, st);
        dfl_conv(d.ov, nullptr, d.dyn, 1, L.acb + 2 * H, B, nullptr, d.x, st);
        dfl_rms(d.x, H, L.post_ln, B, kEps, d.h, st);
        if (d.kp8) gemv_w8(L.mkp8, d.h, H, d.dyn, kDflKp, B, st);
        else gemv_w16(L.mkp, d.h, H, d.dyn, kDflKp, B, st);
        dfl_conv(nullptr, d.h, d.dyn, 0, L.mcb, B, d.hc, nullptr, st);
        const int I = d.ie - d.ib;
        mm4(L.gu, d.hc, H, reinterpret_cast<float*>(d.act), I, B, st);
        mm4(L.down, d.act, I, d.ov, H, B, st);
        const float* mo = d.ov;
        if (d.tp) {  // the MLP's partial sums of every card
            CUDA_CHECK(cudaMemsetAsync(d.red, 0, (size_t)B * H * 4, st));
            allreduce_add(d.red, d.ov, B, H, comm_, rank, ctr_, st, nullptr, 0.f, nullptr, next_ll(B));
            mo = d.red;
        }
        dfl_conv(mo, nullptr, d.dyn, 1, L.mcb + 2 * H, B, nullptr, d.x, st);
    }
    dfl_rms(d.x + H, H, d.norm, M, kEps, d.hid, st);  // rows 1..7
    if (d.full_head) gemv(lm_head_, d.hid, H, d.logits, d.head_v, M, st);
    else mm4(draft_head_, d.hid, H, d.logits, d.head_v, M, st);
    dfl_topk_local(d.logits, d.head_v, M, d.head_map, d.head_off, d.tk, d.pairs, st);
    const float* g = d.pairs;
    if (ngpu_ > 1) {
        allgather_small(d.pairs, M * kDflTopK * 2, d.pairs_all, comm_, rank, ctr_, st);
        g = d.pairs_all;
    }
    dfl_topk_merge(g, ngpu_, M, d.unary, d.cand, st);
    gemv_w16(d.selh, d.hid, H, d.hp, kDflRank, M, st);
    dfl_cb_fetch(d.pred_cb, d.succ_cb, spec_.tokens, d.cand, M, d.pc, d.sc, d.cb_rows, st);
    dfl_select(d.unary, d.cand, d.hp, d.pc, d.sc, M, spec_.tokens, d.cb_rows, st);
    if (d.bcast) {  // every card verifies the same drafts even if a replicated kernel ever differed per card
        allgather_small(reinterpret_cast<const float*>(spec_.tokens), kDflBlock, d.tokg, comm_, rank, ctr_, st);
        dfl_take_row0(d.tokg, spec_.tokens, st);
    }
}

double Worker::dfl_bench(int iters) {
    CUDA_CHECK(cudaSetDevice(device));
    cudaGraph_t g;
    cudaGraphExec_t ge;
    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    dfl_block();
    CUDA_CHECK(cudaStreamEndCapture(stream, &g));
    CUDA_CHECK(cudaGraphInstantiate(&ge, g, 0));
    cudaEvent_t a, b;
    CUDA_CHECK(cudaEventCreate(&a));
    CUDA_CHECK(cudaEventCreate(&b));
    for (int i = 0; i < 3; i++) CUDA_CHECK(cudaGraphLaunch(ge, stream));
    CUDA_CHECK(cudaEventRecord(a, stream));
    for (int i = 0; i < iters; i++) CUDA_CHECK(cudaGraphLaunch(ge, stream));
    CUDA_CHECK(cudaEventRecord(b, stream));
    CUDA_CHECK(cudaEventSynchronize(b));
    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, a, b));
    cudaGraphExecDestroy(ge);
    cudaGraphDestroy(g);
    return ms / iters;
}
