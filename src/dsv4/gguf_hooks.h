#pragma once
// R16 [dsv4-gguf-iq]: engine hooks of the GGUF expert mode (--gguf-experts DIR, opt-in; defaults untouched).
// Included by dsv4.cu after the Engine definition. Every routed expert's host copy = the GGUF bytes of Maya-S
// (gguf_iq.h engine layout: lossless re-pack, d split out), the decode VRAM tier = the non-FP4 dynamic pool as in
// the M23 mode (all experts have a host copy, so no static FP4 slots and no swap tier), with one region per layer
// (equal bytes per layer; the edge layers' experts are larger), the CPU lane and the prefill dequant per matrix
// format. Needs an M23-style pack for the metadata (ranks, Engram, codebooks): every expert a host format.
// Included INSIDE namespace dsv4 of dsv4.cu (which includes gguf_gpu.cuh, gguf_iq.h, gguf_reader.h at its top).

// Pool layout: with the pool split (dsv4.cu psplit, the default with P2P) every card holds a region per layer
// (Card::ploff / ps0 / pnl, sized by pool_fmt per layer: dsv4.cu); without it, the owner card holds one region per
// own layer, laid out here (poff / s0, card-local slot ids).
struct GgState {
    GgufModel m;
    XFmt fmt[NL] = {};       // expert format per layer
    size_t poff[NL] = {};    // no split: byte offset of the layer's region in its owner card's pool
    int s0[NL] = {};         // no split: first card-local pool slot id of the layer
};

static size_t gg_xb(const Engine* E, int l) { return expert_bytes(E->gg->fmt[l]); }
// no split: slot (card-local id) of layer l in its owner card's pool
static uint8_t* gg_pool_ptr(Engine* E, int l, int slot) {
    return E->cards[E->L[l].card].dpool + E->gg->poff[l] + (size_t)(slot - E->gg->s0[l]) * gg_xb(E, l);
}

// open the directory, check every layer's three expert tensors, set every expert's host format
static void gg_setup(Engine* E, const std::string& dir) {
    // review M5: a placement file would put its resident set into static FP4 slots from the checkpoint (not Maya-S's
    // bytes): the GGUF mode has no static set
    DCHECK(E->o.placement_in.empty(), "--placement-in cannot be combined with --gguf-experts (its resident experts would run "
           "from the checkpoint's FP4, not from the GGUF)");
    E->gg = new GgState;
    std::string err;
    DCHECK(E->gg->m.open_dir(dir, &err), "--gguf-experts: %s", err.c_str());
    double tot = 0;
    std::map<int, int> nfmt;
    for (int l = 0; l < NL; l++) {
        XFmt mf[3];
        for (int m = 0; m < 3; m++) {
            const GgTensor* t = E->gg->m.expert_tensor(l, m);
            DCHECK(t, "--gguf-experts: layer %d matrix %d missing in %s", l, m, dir.c_str());
            mf[m] = gg_mat_fmt(t->type);
            DCHECK(mf[m] != (XFmt)255, "--gguf-experts: %s is %s (supported: IQ2_XXS, IQ3_XXS, IQ2_S, IQ3_S)", t->name.c_str(),
                   gg_type_name(t->type));
            const uint64_t K = m < 2 ? D : INTER, N = m < 2 ? INTER : D;
            DCHECK(t->sized && t->ne.size() == 3 && t->ne[0] == K && t->ne[1] == N && t->ne[2] == NE, "--gguf-experts: %s dims", t->name.c_str());
        }
        DCHECK(mf[0] == mf[1], "--gguf-experts: layer %d gate %d / up %d formats differ (the decode GEMV takes one)", l,
               (int)mf[0], (int)mf[1]);
        E->gg->fmt[l] = gg_expert_fmt(mf[0], mf[1], mf[2]);
        nfmt[E->gg->fmt[l]]++;
        for (int e = 0; e < NE; e++) {
            DCHECK(E->L[l].hfmt[e] >= 0, "--gguf-experts needs a pack with a host format for every expert (M23 pack); layer %d expert %d has none", l, e);
            E->L[l].hfmt[e] = (int8_t)E->gg->fmt[l];
        }
        tot += (double)NE * gg_xb(E, l);
    }
    E->pfmt = E->gg->fmt[NL / 2];
    E->m23conv = false;
    logf_("GGUF experts: %s, %zu files, %.2f GB of routed experts (host copies = the GGUF bytes)\n", dir.c_str(),
          E->gg->m.files().size(), tot / 1e9);
    for (auto& kv : nfmt) {
        const XFmt f = (XFmt)kv.first;
        logf_("  %2d layers: gate %s, up %s, down %s, %.4f MB per expert\n", kv.second, gg_type_name(gg_type_of(mat_fmt(f, 0))),
              gg_type_name(gg_type_of(mat_fmt(f, 1))), gg_type_name(gg_type_of(mat_fmt(f, 2))), expert_bytes(f) / 1e6);
    }
}

static void gg_card_init(Engine* E, Card& c) {
    std::vector<uint8_t> hg(GG_GRID_BYTES);
    memcpy(hg.data() + GG_OFF_IQ2XXS, gg_iq2xxs_grid, 2048);
    memcpy(hg.data() + GG_OFF_IQ3XXS, gg_iq3xxs_grid, 1024);
    memcpy(hg.data() + GG_OFF_IQ2S, gg_iq2s_grid, 8192);
    memcpy(hg.data() + GG_OFF_IQ3S, gg_iq3s_grid, 2048);
    c.d_gg = (uint8_t*)dmalloc(GG_GRID_BYTES);
    DCK(cudaMemcpy(c.d_gg, hg.data(), GG_GRID_BYTES, cudaMemcpyHostToDevice));
}

// no split: the card's dynamic pool, `bytes` split evenly over its layers, each layer's region in its own expert size
static void gg_pool_alloc(Engine* E, Card& c, size_t bytes) {
    const int nl = c.l1 - c.l0;
    // [kb-dsv4-load] VRAM taken by another process since the sizing: a smaller pool, never an abort (morning list 39);
    // DSV4_TEST_POOLFAIL=n: the first n tries fail (test)
    static int test_fail = getenv("DSV4_TEST_POOLFAIL") ? atoi(getenv("DSV4_TEST_POOLFAIL")) : 0;
    void* pool = nullptr;
    for (int tries = 0;; tries++) {
        const size_t pb = nl ? bytes / nl : 0;
        size_t want = 0;
        for (int l = c.l0; l < c.l1; l++) want += (size_t)(pb / gg_xb(E, l)) * gg_xb(E, l);
        if (test_fail-- <= 0 && cudaMalloc(&pool, std::max<size_t>(want, 1)) == cudaSuccess) break;
        cudaGetLastError();
        pool = nullptr;
        DCHECK(bytes > 0 && tries < 32, "card %d: no VRAM for the dynamic expert tier (taken by another process while loading)", c.dev);
        const size_t nb = bytes > ((size_t)1 << 30) ? bytes / 8 * 7 - ((size_t)256 << 20) : 0;
        logf_("WARNING card %d: the dynamic tier (%.2f GB) could not be allocated: VRAM was taken meanwhile; %.2f GB instead\n",
              c.dev, want / 1e9, nb / 1e9);
        bytes = nb;
    }
    const size_t per_b = nl ? bytes / nl : 0;
    size_t off = 0;
    int s = 0;
    for (int l = c.l0; l < c.l1; l++) {
        const int per = (int)(per_b / gg_xb(E, l));
        E->gg->poff[l] = off;
        E->gg->s0[l] = s;
        Layer& W = E->L[l];
        for (int i = 0; i < per; i++) W.dpool.push_back(s + i);
        W.dyn_e.assign(per, -1);
        W.pend_e.assign(per, -1);
        W.pend_job.assign(per, -1);
        W.pend_t.assign(per, 0);
        W.pend_ev.resize(per);
        for (auto& ev : W.pend_ev) DCK(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming));
        off += (size_t)per * gg_xb(E, l);
        s += per;
    }
    c.ndyn = s;
    c.dpool = (uint8_t*)pool;
    {  // (accounted as dmalloc does)
        int d = 0;
        cudaGetDevice(&d);
        if (d >= 0 && d < 3) g_mb[d][g_mtag] += (double)std::max<size_t>(off, 1);
    }
    logf_("card %d: dynamic tier %d GGUF slots (%.2f GB; per layer %d..%d)\n", c.dev, c.ndyn, off / 1e9,
          nl ? (int)(per_b / gg_xb(E, c.l0)) : 0, nl ? (int)(per_b / gg_xb(E, c.l1 - 1)) : 0);
}

static int gg_load_threads() { return std::max(4, std::min(32, (int)std::thread::hardware_concurrency() - 2)); }
// bytes of the loader's per-thread read buffers at their largest (one matrix slice each; review L4: in the RAM plan)
static size_t gg_load_scratch(const Engine* E) {
    size_t mx = 0;
    for (int l = 0; l < NL; l++) {
        const int Nm[3] = {INTER, INTER, D}, Km[3] = {D, D, INTER};
        for (int m = 0; m < 3; m++) mx = std::max(mx, mat_bytes(mat_fmt(E->gg->fmt[l], m), Nm[m], Km[m]));
    }
    return mx * (size_t)gg_load_threads();
}

// host copies of layer l's experts (the load's predicate, as m23_convert_layer) from the GGUF into ar, re-packed;
// pread per matrix + POSIX_FADV_DONTNEED (no page cache left behind); the files stay read-only
static double g_ld_alloc_ms = 0, g_ld_copy_ms = 0, g_ld_copy_bg_ms = 0;   // (R24: background read time)
static long long g_ld_seed_bytes = 0;
static std::atomic<long long> g_ld_read_ns{0}, g_ld_pack_ns{0}, g_ld_bytes{0};   // [dsv4-rel-load128] load breakdown
static void gg_copy_layer(Engine* E, int l, uint8_t* ar, bool (*want)(Engine*, int, int)) {
    std::vector<std::pair<int, size_t>> todo;
    size_t off = 0;
    const size_t XB = gg_xb(E, l);
    for (int e = 0; e < NE; e++)
        if (want(E, l, e)) {
            todo.push_back({e, off});
            off += XB;
        }
    const XFmt xf = E->gg->fmt[l];
    const GgTensor* t[3];
    for (int m = 0; m < 3; m++) t[m] = E->gg->m.expert_tensor(l, m);
    std::atomic<size_t> next{0};
    std::atomic<int> nbuf{0};   // slices read through the page cache (buffered fallback)
    // R24 [dsv4-r24-odirect]: the slices are read with O_DIRECT (4 KB aligned superset into an aligned buffer, no page
    // cache copy, no cache pressure on a 128 GB box); a file that refuses O_DIRECT, or an EINVAL read, falls back to the
    // buffered fd. Same bytes. DSV4_NODIRECT=1: buffered as before
    static const bool nodirect = getenv("DSV4_NODIRECT") && atoi(getenv("DSV4_NODIRECT")) != 0;
    std::vector<int> dfd(E->gg->m.files().size(), -1);
    if (!nodirect)
        for (size_t f = 0; f < dfd.size(); f++) dfd[f] = open(E->gg->m.files()[f].path.c_str(), O_RDONLY | O_DIRECT);
    auto wk = [&] {
        std::vector<uint8_t> buf;
        uint8_t* abuf = nullptr;
        size_t abn = 0;
        while (true) {
            const size_t u = next.fetch_add(1);   // R24: (expert, matrix) units: 3x the parallelism of a 32-expert chunk
            if (u >= 3 * todo.size()) break;
            const size_t i = u / 3;
            const int e = todo[i].first;
            MatView mv[3];
            expert_views(xf, ar + todo[i].second, mv);
            const int Nm[3] = {INTER, INTER, D}, Km[3] = {D, D, INTER};
            for (int m = (int)(u % 3); m == (int)(u % 3); m++) {   // (one matrix: the unit's)
                size_t nb;
                const uint8_t* src = E->gg->m.expert_slice(*t[m], e, &nb);
                DCHECK(src, "--gguf-experts: no slice for layer %d expert %d matrix %d", l, e, m);
                const GgFile& F = E->gg->m.files()[t[m]->file];
                const off_t fo = (off_t)(src - F.base);
                const auto tr0 = std::chrono::steady_clock::now();
                const uint8_t* srcb = nullptr;
                const int dd = dfd[t[m]->file];
                if (dd >= 0) {   // R24: aligned superset [a0, a1) of [fo, fo + nb)
                    const off_t a0 = fo & ~(off_t)4095, a1 = (fo + (off_t)nb + 4095) & ~(off_t)4095;
                    const size_t an = (size_t)(a1 - a0);
                    if (an > abn) {
                        free(abuf);
                        abuf = (uint8_t*)aligned_alloc(4096, an);
                        abn = abuf ? an : 0;
                    }
                    size_t got = 0;
                    bool ok = abuf != nullptr;
                    while (ok && got < (size_t)(fo + (off_t)nb - a0)) {
                        const ssize_t r = pread(dd, abuf + got, an - got, a0 + (off_t)got);
                        if (r < 0 && errno == EINTR) continue;
                        if (r <= 0) ok = false;   // EINVAL / short at the end: the buffered path below
                        else got += (size_t)r;
                    }
                    if (ok) srcb = abuf + (fo - a0);
                }
                if (!srcb) {
                    nbuf++;
                    buf.resize(nb);   // one buffer per thread, reused (<= 5.1 MB, review L4)
                    size_t got = 0;
                    while (got < nb) {
                        const ssize_t r = pread(F.fd, buf.data() + got, nb - got, fo + (off_t)got);
                        if (r < 0 && errno == EINTR) continue;   // review L5
                        DCHECK(r > 0, "%s: %s at offset %lld", F.path.c_str(), r == 0 ? "unexpected end of file" : strerror(errno),
                               (long long)(fo + (off_t)got));
                        got += (size_t)r;
                    }
                    srcb = buf.data();
                }
                const auto tr1 = std::chrono::steady_clock::now();
                gg_repack(mat_fmt(xf, m), srcb, Nm[m], Km[m], (uint8_t*)mv[m].q);
                const auto tr2 = std::chrono::steady_clock::now();
                g_ld_read_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(tr1 - tr0).count();
                g_ld_pack_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(tr2 - tr1).count();
                g_ld_bytes += (long long)nb;
            }
        }
        free(abuf);
    };
    std::vector<std::thread> th;
    const int nt = gg_load_threads();
    for (int i = 0; i < nt; i++) th.emplace_back(wk);
    for (auto& x : th) x.join();
    for (int d : dfd)
        if (d >= 0) close(d);
    // review L3: drop the whole files' pages (slices are not page aligned; per-slice DONTNEED left partial pages)
    // R24: only when a slice took the buffered path (O_DIRECT leaves no page cache; a whole-file DONTNEED per call
    // costs a walk over the files' cache for each of the 128 GB plan's ~150 seed chunks)
    if (nbuf.load())
        for (auto& F : E->gg->m.files()) posix_fadvise(F.fd, 0, 0, POSIX_FADV_DONTNEED);
}

// R21 [dsv4-r21-dp4a]: 1 = the R16 fp32 kernel (DSV4_GG_FP32=1 at start, or the job "set gg_fp32 1" between jobs);
// review r21 L10: the value counts (DSV4_GG_FP32=0 = dp4a)
static int g_gg_fp32 = getenv("DSV4_GG_FP32") && atoi(getenv("DSV4_GG_FP32")) != 0;
// decode GEMV of layer l's pool experts on card k (gate / up: which 0, down: which 1); tbl = card k's table: split =
// in-layer indices (W.d_dls[k], pool_lidx) from the region at ploff[l]; no split = card-local ids (W.d_dslot) from s0
static void gg_moe(Engine* E, int k, int l, int which, const float* x, const int* ids, const int* tbl, float* out, int R,
                   cudaStream_t st) {
    const XFmt xf = E->gg->fmt[l];
    Card& c = E->cards[k];
    const uint8_t* base = E->psplit ? c.dpool + c.ploff[l] : c.dpool + E->gg->poff[l];
    const int s0 = E->psplit ? 0 : E->gg->s0[l];
    // R21 [dsv4-r21-dp4a]: default = int8 x per 32 + dp4a, a lane per 32-weight sub-block (not bit-identical: the CPU
    // lane's activation level); DSV4_GG_FP32=1 = the R16 fp32 kernel (exact reference)
    if (!g_gg_fp32 && c.d_q8) {
        const int n = which ? R * TOPK * INTER : R * D;
        int8_t* q = c.d_q8 + (which ? (size_t)MAXR * D : 0);
        float* sc = c.d_q8s + (which ? (size_t)MAXR * D / 32 : 0);
        gg_q8x_launch(x, n, q, sc, st);
        gg_moe_launch_q2<4, 1>(which, mat_fmt(xf, which ? 2 : 0), q, sc, ids, tbl, base, s0, gg_xb(E, l), xf, c.d_gg, out, R, st);
        return;
    }
    gg_moe_launch<GG_RPW, GG_U>(which, mat_fmt(xf, which ? 2 : 0), x, ids, tbl, base, s0, gg_xb(E, l), xf, c.d_gg, out, R, st);
}

