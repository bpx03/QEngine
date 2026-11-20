// [gptoss-r15-dflash] Host side of the DFlash block drafter: loader, the drafter pass, Engine::draft_*.
// Included inside namespace gptoss by gptoss.cu (it needs Impl / Rank). Kernels and structures: gptoss_dflash.cuh.
//
// Split over the cards (tensor parallel, as the target): KV heads as the target's plan (q rows, k / v rows, o_proj
// columns of the card's heads), the MLP's 7,680 intermediate rows in slices that even out the bytes per card, fc's
// 14,400 input columns in equal slices. Two collectives per layer (o_proj, down) + one for fc: k_ar_t with the residual
// add and the next norm, as in the target's step. One card alone would read 1.57 GB per pass (~2 ms); three cards read
// ~0.5 GB each.
// Context store per ring set: K (k_norm + RoPE applied) and V as fp16, [layer][slot row][KV head][64]. The first
// quarter of the rows is kept (the prompt's start), the rest is a ring of the latest rows; a row's RoPE position is its
// sequence position, so the set of rows may have gaps.

static inline uint16_t df_f2bf(float f) {   // round to nearest even
    uint32_t u;
    memcpy(&u, &f, 4);
    return (uint16_t)((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16);
}

// [phase 2] every device buffer of the drafter in one allocation (a pass over the loader that only counts, then the real
// one): ~100 separate allocations cost ~70 MiB of allocation granularity per card
template <class T>
static T* df_put(DfRank& d, const void* src, size_t n) {
    const size_t b = (n * sizeof(T) + 255) & ~(size_t)255;
    if (!d.abase) {
        d.aoff += b;
        return nullptr;
    }
    GCHECK(d.aoff + b <= d.acap, "drafter arena: %zu + %zu of %zu bytes", d.aoff, b, d.acap);
    T* p = (T*)(d.abase + d.aoff);
    d.aoff += b;
    if (src) CU(cudaMemcpy(p, src, n * sizeof(T), cudaMemcpyHostToDevice));
    else CU(cudaMemset(p, 0, b));
    return p;
}

// [phase 2] host quantiser of the drafter's matrices: INT4 codes (nibble - 8, -7..7) and an fp16 scale (absmax / 7)
// per 32 inputs, the P2 gate's i4 form (dq_replay.py). src = bf16 rows [N][K] with row stride lds_src elements.
static DfQ4 df_q4(DfRank& d, const uint16_t* src, size_t ld_src, int N, int K) {
    GCHECK(N % 16 == 0 && K % 32 == 0, "drafter matrix %d x %d", N, K);
    DfQ4 q;
    q.N = N;
    q.K = K;
    q.ldc = (size_t)K / 2;
    q.lds = (size_t)K / 32;
    if (!d.abase) {   // counting pass
        df_put<uint8_t>(d, nullptr, (size_t)N * q.ldc);
        df_put<__half>(d, nullptr, (size_t)N * q.lds);
        return q;
    }
    std::vector<uint8_t> c((size_t)N * q.ldc);
    std::vector<__half> sc((size_t)N * q.lds);
    auto f = [](uint16_t b) {
        const uint32_t u = (uint32_t)b << 16;
        float v;
        memcpy(&v, &u, 4);
        return v;
    };
    for (int i = 0; i < N; i++)
        for (int g = 0; g < K / 32; g++) {
            const uint16_t* w = src + (size_t)i * ld_src + 32 * g;
            float am = 0.f;
            for (int j = 0; j < 32; j++) am = std::max(am, std::fabs(f(w[j])));
            const __half hs = __float2half_rn(am / 7.f);
            const float s = __half2float(hs);
            sc[(size_t)i * q.lds + g] = hs;
            uint8_t* o = c.data() + (size_t)i * q.ldc + 16 * g;
            for (int j = 0; j < 32; j += 2) {
                int c2[2];
                for (int h = 0; h < 2; h++) {
                    const float v = s > 0.f ? f(w[j + h]) / s : 0.f;
                    c2[h] = std::max(-7, std::min(7, (int)std::lrint(v))) + 8;
                }
                o[j / 2] = (uint8_t)(c2[0] | (c2[1] << 4));
            }
        }
    q.c = df_put<uint8_t>(d, c.data(), c.size());
    q.s = df_put<__half>(d, sc.data(), sc.size());
    return q;
}

// out[col][row] (+)= W x[col] for R <= 64 columns (k_q4_m with the fewest 8-column tiles that hold R)
static void df_q4m(cudaStream_t st, const DfQ4& q, const bf16* x, int ldx, int R, const bf16* bias, float* out, int ldo, int acc = 0) {
    const int w = q4_warps(q.K);
    const unsigned nb = (unsigned)(q.N / 16);
    if (R <= 8)
        k_q4_m<1><<<nb, 32 * w, dm_smem(1, w), st>>>(q.c, q.s, q.N, q.K, q.ldc, q.lds, x, ldx, R, bias, out, ldo, acc);
    else if (R <= 16)
        k_q4_m<2><<<nb, 32 * w, dm_smem(2, w), st>>>(q.c, q.s, q.N, q.K, q.ldc, q.lds, x, ldx, R, bias, out, ldo, acc);
    else if (R <= 32)
        k_q4_m<4><<<nb, 32 * w, dm_smem(4, w), st>>>(q.c, q.s, q.N, q.K, q.ldc, q.lds, x, ldx, R, bias, out, ldo, acc);
    else   // (at most 8 warps: 32 KB of reduction space)
        k_q4_m<8><<<nb, 32 * std::min(w, 8), dm_smem(8, std::min(w, 8)), st>>>(q.c, q.s, q.N, q.K, q.ldc, q.lds, x, ldx, R, bias, out, ldo, acc);
}

void Impl::df_load(Rank& r) {
    r.df = new DfRank;
    DfRank& d = *r.df;
    size_t f0 = 0, f1 = 0, tot = 0;
    CU(cudaMemGetInfo(&f0, &tot));
    const Files& F = dfiles;
    auto bf = [&](const std::string& n, size_t elems) { return (const uint16_t*)F.get(n, "BF16", elems * 2).p; };
    const int NQ = NH * HD, NK = NKV * HD;   // 4096 / 512
    {   // MLP slices: even out the weight bytes per card (all matrices have the same bits per weight)
        const double A = 8.0 * HD * H + 2.0 * HD * H + (double)H * 8 * HD, M = 3.0 * H;   // weights per KV head / per intermediate row
        const double T = (NKV * A + DF_INTER * M) / ngpu;
        int j = 0;
        for (int g = 0; g < ngpu; g++) {
            int jn = g + 1 < ngpu ? (int)std::lround((T - plan.kvn[g] * A) / M / 32.0) * 32 : DF_INTER - j;
            jn = std::max(32, std::min(jn, DF_INTER - j - 32 * (ngpu - 1 - g)));
            if (g == r.rank) d.j0 = j, d.jn = jn;
            j += jn;
        }
    }
    const int kvn = r.kvn, kn_ = kvn * HD;
    d.nkv = DF_L * 2 * kn_;
    const int cap = dfh->cap;
    d.lrows = (size_t)cap + DF_PAD;
    d.srows = (size_t)DF_L * d.lrows;
    d.NT = cap / DF_TK + 2;
    for (int pass = 0; pass < 2; pass++) {   // pass 0 counts the bytes, pass 1 loads
    if (pass) {
        d.acap = d.aoff;
        d.aoff = 0;
        d.abase = r.alloc<char>(d.acap, "drafter");
    }
    std::vector<uint16_t> kv((size_t)d.nkv * H), bkv(d.nkv), kn(DF_L * HD), tmp;
    auto vec = [&](const std::string& n, size_t elems) { return df_put<bf16>(d, bf(n, elems), elems); };
    for (int l = 0; l < DF_L; l++) {
        const std::string p = "layers." + std::to_string(l) + ".", a = p + "self_attn.", m = p + "mlp.";
        DfLayerW& w = d.L[l];
        const uint16_t* wq = bf(a + "q_proj.weight", (size_t)NQ * H);
        const uint16_t* wk = bf(a + "k_proj.weight", (size_t)NK * H);
        const uint16_t* wv = bf(a + "v_proj.weight", (size_t)NK * H);
        w.wq = df_q4(d, wq + (size_t)r.kv0 * 8 * HD * H, H, r.ncol, H);
        memcpy(kv.data() + (size_t)l * 2 * kn_ * H, wk + (size_t)r.kv0 * HD * H, (size_t)kn_ * H * 2);
        memcpy(kv.data() + ((size_t)l * 2 + 1) * kn_ * H, wv + (size_t)r.kv0 * HD * H, (size_t)kn_ * H * 2);
        const uint16_t* bq = bf(a + "q_proj.bias", NQ);
        const uint16_t* bk = bf(a + "k_proj.bias", NK);
        const uint16_t* bv = bf(a + "v_proj.bias", NK);
        w.bq = df_put<bf16>(d, bq + r.kv0 * 8 * HD, r.ncol);
        memcpy(bkv.data() + (size_t)l * 2 * kn_, bk + r.kv0 * HD, kn_ * 2);
        memcpy(bkv.data() + ((size_t)l * 2 + 1) * kn_, bv + r.kv0 * HD, kn_ * 2);
        const uint16_t* wo = bf(a + "o_proj.weight", (size_t)H * NQ);
        w.wo = df_q4(d, wo + r.kv0 * 8 * HD, NQ, H, r.ncol);
        w.bo = vec(a + "o_proj.bias", H);
        const uint16_t* wg = bf(m + "gate_proj.weight", (size_t)DF_INTER * H);
        const uint16_t* wu = bf(m + "up_proj.weight", (size_t)DF_INTER * H);
        tmp.resize((size_t)2 * d.jn * H);
        memcpy(tmp.data(), wg + (size_t)d.j0 * H, (size_t)d.jn * H * 2);
        memcpy(tmp.data() + (size_t)d.jn * H, wu + (size_t)d.j0 * H, (size_t)d.jn * H * 2);
        w.wgu = df_q4(d, tmp.data(), H, 2 * d.jn, H);
        const uint16_t* wd = bf(m + "down_proj.weight", (size_t)H * DF_INTER);
        w.wdn = df_q4(d, wd + d.j0, DF_INTER, H, d.jn);
        w.n1 = vec(p + "input_layernorm.weight", H);
        w.n2 = vec(p + "post_attention_layernorm.weight", H);
        w.qn = vec(a + "q_norm.weight", HD);
        w.kn = vec(a + "k_norm.weight", HD);
        memcpy(kn.data() + l * HD, bf(a + "k_norm.weight", HD), HD * 2);
    }
    d.wkv = df_q4(d, kv.data(), H, d.nkv, H);
    d.bkvall = df_put<bf16>(d, bkv.data(), bkv.size());
    for (int l = 0; l < DF_L; l++) d.L[l].bkv = d.bkvall + (size_t)l * 2 * kn_;
    d.knall = df_put<bf16>(d, kn.data(), kn.size());
    d.wfc = df_q4(d, bf("fc.weight", (size_t)H * DF_FC), DF_FC, H, DF_FC);
    if (lanes_on() && cfg.pf_owner) {   // the prefill's tap path (off: GPTOSS_DF_PF=0, the caller feeds the prompt's rows)
        const char* e = getenv("GPTOSS_DF_PF");
        if (!e || atoi(e) != 0) {
            // taps of the own rows that stay in the store: at most cap rows of a chunk survive, spread over the cards
            // by the lanes' row-owner split: ~1,140 per card for the ring part, but card 0 also owns the kept first
            // quarter (positions < head), so on a prompt of more than ~5K rows in one chunk card 0 drops up to ~540
            // of its oldest ring rows (review L4: holes in the context, lower acceptance only; a larger cap on card 0
            // would cost ~29 MB of its K/V pool). GPTOSS_DF_PTAP = rows.
            const char* pe = getenv("GPTOSS_DF_PTAP");
            d.pcap = std::min({std::max(cfg.chunk, MAXR), cap, pe ? std::max(64, atoi(pe)) : 1280});
            d.ptap = df_put<bf16>(d, nullptr, (size_t)d.pcap * DF_FC);
            d.ppos = df_put<int>(d, nullptr, (size_t)std::max(cfg.chunk, MAXR));
            if (d.abase) d.ppos_host = r.halloc<int>((size_t)std::max(cfg.chunk, MAXR) * 4);
        }
    }
    d.hn = vec("hidden_norm.weight", H);
    d.nf = vec("norm.weight", H);
    const char* c = "drafter";
    d.ring = df_put<bf16>(d, nullptr, (size_t)DF_RING * DF_FC);
    d.Tg = df_put<bf16>(d, nullptr, (size_t)16 * DF_FC);
    d.C = df_put<bf16>(d, nullptr, (size_t)16 * H);
    d.XN = df_put<bf16>(d, nullptr, (size_t)DF_MAXB * H);
    d.AO = df_put<bf16>(d, nullptr, (size_t)DF_MAXB * r.ncol);
    d.act = df_put<bf16>(d, nullptr, (size_t)DF_MAXB * d.jn);
    d.Hc = df_put<float>(d, nullptr, (size_t)16 * H);
    d.Hs = df_put<float>(d, nullptr, (size_t)DF_MAXB * H);
    d.part = df_put<float>(d, nullptr, (size_t)16 * H);
    d.Y = df_put<float>(d, nullptr, (size_t)DF_MAXB * (r.nrows));
    d.Ykv = df_put<float>(d, nullptr, (size_t)16 * d.nkv);
    d.GU = df_put<float>(d, nullptr, (size_t)DF_MAXB * 2 * d.jn);
    d.apart = df_put<float>(d, nullptr, (size_t)kvn * d.NT * 64 * DF_PW);
    d.Qh = df_put<__half>(d, nullptr, (size_t)DF_MAXB * r.nq * HD);
    const size_t st = (size_t)cfg.slots * d.srows * kvn * HD;
    d.Ks = df_put<__half>(d, nullptr, st);
    d.Vs = df_put<__half>(d, nullptr, st);
    d.par = df_put<DfPar>(d, nullptr, 1);
    if (d.abase) d.par_host = r.halloc<DfPar>(sizeof(DfPar));
    if (d.abase) d.h_host = r.halloc<float>((size_t)DF_MAXB * H * 4);
    if (d.abase) d.res_host = r.halloc<float>(2 * DF_MAXB * 4);
    if (d.abase) d.tg_host = r.halloc<bf16>((size_t)16 * DF_FC * 2);
    }
    CU(cudaMemGetInfo(&f1, &tot));
    d.bytes = f0 - f1;
    if (cfg.verbose)
        fprintf(stderr, "gptoss: drafter on card %d: %.0f MiB (4-bit weights + %d ring sets x %d context rows + %d prefill tap rows), MLP rows %d..%d\n", r.dev,
                d.bytes / 1048576.0, cfg.slots, cap, d.pcap, d.j0, d.j0 + d.jn - 1);
}
void Impl::df_free(Rank& r) {
    if (r.df)
        for (auto& kv : r.df->gx)
            if (kv.second) cudaGraphExecDestroy(kv.second);
    delete r.df;
    r.df = nullptr;
    if (r.rank == 0) {
        delete dfh;
        dfh = nullptr;
    }
}

// taps of a decode step's rows (inside the step graph)
void Impl::df_tap(Rank& r, int L, int R) {
    for (int i = 0; i < DF_NTAP; i++)
        if (DF_TAP_LAYER[i] == L) k_df_tap<<<R, 960, 0, r.st>>>(r.Hs, r.df->ring, r.pos, i);
}

// one drafter pass of a card (par_host says what): new context rows, then the block
void Impl::df_launch(Rank& r, int slot, bool host_taps, int ntg) {
    DfRank& d = *r.df;
    cudaStream_t st = r.st;
    const DfPar& hp = *d.par_host;
    const int na = hp.na, B = hp.B, kvn = r.kvn;
    CU(cudaMemcpyAsync(d.par, d.par_host, sizeof(DfPar), cudaMemcpyHostToDevice, st));
    k_begin<<<1, 1, 0, st>>>(r.step);
    __half* Ks = d.Ks + (size_t)slot * d.srows * kvn * HD;
    __half* Vs = d.Vs + (size_t)slot * d.srows * kvn * HD;
    if (na > 0) {
        if (host_taps)
            CU(cudaMemcpyAsync(d.Tg, d.tg_host, (size_t)na * DF_FC * 2, cudaMemcpyHostToDevice, st));
        else
            k_df_gather<<<na, 1024, 0, st>>>(d.ring, d.par, d.Tg);
        df_q4m(st, d.wfc, d.Tg, DF_FC, na, nullptr, d.Hc, H);   // the whole fc on every card: no collective
        k_resnorm<<<na, 960, 0, st>>>(d.Hc, nullptr, nullptr, nullptr, nullptr, d.hn, d.C, H);
        df_q4m(st, d.wkv, d.C, H, na, d.bkvall, d.Ykv, d.nkv);
        k_df_kvstore<<<dim3(na, DF_L * 2 * kvn), 32, 0, st>>>(d.Ykv, kvn, d.knall, d.par, r.invf, att, Ks, Vs, d.lrows);
    }
    if (B < 2) return;
    CU(cudaMemcpyAsync(d.Hs, d.h_host, (size_t)B * H * 4, cudaMemcpyHostToDevice, st));
    k_resnorm<<<B, 960, 0, st>>>(d.Hs, nullptr, nullptr, nullptr, nullptr, d.L[0].n1, d.XN, H);
    const int nqkv = r.nrows, kn2 = 2 * kvn * HD;
    for (int l = 0; l < DF_L; l++) {
        const DfLayerW& w = d.L[l];
        __half* Kl = Ks + (size_t)l * d.lrows * kvn * HD;
        __half* Vl = Vs + (size_t)l * d.lrows * kvn * HD;
        df_q4m(st, w.wq, d.XN, H, B, w.bq, d.Y, nqkv);
        df_q4m(st, d.wkv.rows(l * kn2, kn2), d.XN, H, B, w.bkv, d.Y + r.ncol, nqkv);
        k_df_qkv<<<dim3(B, r.nq + 2 * kvn), 32, 0, st>>>(d.Y, r.nq, kvn, w.qn, w.kn, d.par, r.invf, att, d.Qh, Kl, Vl);
        k_df_attn<<<dim3(ntg, kvn), 256, 0, st>>>(d.Qh, Kl, Vl, d.par, r.nq, kvn, d.apart, d.NT);
        k_df_comb<<<dim3(r.nq, B), 64, 0, st>>>(d.apart, d.par, r.nq, d.NT, d.AO);
        df_q4m(st, w.wo, d.AO, r.ncol, B, r.rank == 0 ? w.bo : nullptr, d.part, H);
        k_ar_t<<<B, 960, 0, st>>>(d.part, d.Hs, w.n2, d.XN, r.ar, r.rank, ngpu, r.step, 1 + 2 * l, nullptr, 0, 0);
        df_q4m(st, w.wgu, d.XN, H, B, nullptr, d.GU, 2 * d.jn);
        k_df_act<<<(B * d.jn + 255) / 256, 256, 0, st>>>(d.GU, d.jn, B, d.act);
        df_q4m(st, w.wdn, d.act, d.jn, B, nullptr, d.part, H);
        k_ar_t<<<B, 960, 0, st>>>(d.part, d.Hs, l + 1 < DF_L ? d.L[l + 1].n1 : d.nf, d.XN, r.ar, r.rank, ngpu, r.step, 2 + 2 * l, nullptr, 0, 0);
    }
    // the draft rows (1 .. B - 1) through the target's head: certified screen + exact rescoring, or the full head
    const int R1 = B - 1;
    const bf16* x = d.XN + H;
    if (r.s8) {
        const int nb = (r.vn + 31) / 32;
        const bool scm = R1 > 1 && cfg.lm_screen_mma;
        const int nbs = scm ? (r.vn + 127) / 128 : nb;
        k_xmax<<<R1, 960, 0, st>>>(x, r.xmax);
        if (R1 == 1)
            k_lm_screen1<<<nb, 256, 0, st>>>(r.s8, r.s8base, x, r.xmax, r.vn, r.lmlo, r.lmhi, r.lomax);
        else if (scm)
            k_lm_screen_m<1><<<nbs, 256, 0, st>>>(r.s8, r.s8base, x, r.xmax, r.vn, r.lmlo, r.lmhi, r.lomax, R1);
        else
            k_lm_screen<<<nb, 256, 0, st>>>(r.s8, r.s8base, x, r.xmax, r.vn, r.lmlo, r.lmhi, r.lomax, R1);
        k_lm_thr<<<R1, 1024, 0, st>>>(r.lomax, nbs, r.tb, r.rank, ngpu, r.step, r.thr);
        k_lm_rescore<<<dim3((nb + 7) / 8, R1), 256, 0, st>>>(r.lm, x, r.lmhi, r.thr, r.vn, r.lbv, r.lbi);
        k_argbest<<<R1, 1024, 0, st>>>(r.lbv, r.lbi, ((nb + 7) / 8) * 8, r.v0, r.res);
    } else {
        k_gemv<4><<<(r.vn + 31) / 32, 256, 0, st>>>(r.lm, r.vn, H, x, nullptr, r.logits, R1, H, r.vn);
        k_argmax<<<R1, 1024, 0, st>>>(r.logits, r.vn, r.v0, r.res);
    }
    CU(cudaMemcpyAsync(d.res_host, r.res, (size_t)R1 * 8, cudaMemcpyDeviceToHost, st));
}

// runs the pass staged in card 0's par_host on every card; out[0 .. B - 2] = the drafts
void Impl::df_pass(int slot, int na, bool host_taps, int B, int bpos, int token, int* out) {
    (void)na;
    (void)bpos;
    ar_epoch();   // [gptoss-rel2-tagwrap] the pass advances the step counter
    DfPar& p = *R[0].df->par_host;
    if (B >= 2) {
        float* h = R[0].df->h_host;
        embed_row(token, h);
        embed_row(DF_MASK, h + H);
        for (int b = 2; b < B; b++) memcpy(h + (size_t)b * H, h + H, H * 4);
    }
    for (int g = 1; g < ngpu; g++) {
        *R[g].df->par_host = p;
        if (B >= 2) memcpy(R[g].df->h_host, R[0].df->h_host, (size_t)B * H * 4);
        if (host_taps && p.na > 0) memcpy(R[g].df->tg_host, R[0].df->tg_host, (size_t)p.na * DF_FC * 2);
    }
    struct Guard {
        Impl& m;
        bool ok = false;
        ~Guard() { if (!ok) m.broken.store(true); }
    } guard{*this};
    // the tile grid of the attention launches: the smallest class that holds the context's tiles + the block's (tiles
    // past the context leave at once); one graph per (slot, new rows, block rows, class, tap source)
    const int nt = (p.nv + DF_TK - 1) / DF_TK + 1, NT = R[0].df->NT;
    int ntg = 5;
    while (ntg < nt) ntg = std::min(NT, 4 * (ntg - 1) + 1);
    const int Bk = p.B >= 2 ? p.B : 0;
    // (review M2 / L6) 64-bit key: na reaches 16 (5 bits), ntg is not truncated
    const unsigned long long key = (unsigned long long)slot << 48 | (unsigned long long)p.na << 40 | (unsigned long long)Bk << 32 |
                                   (unsigned long long)host_taps << 31 | (unsigned long long)ntg;
    const bool use_graph = cfg.graph;
    pool.run([this, slot, host_taps, ntg, key, use_graph](int g) {
        Rank& r = R[g];
        if (use_graph) {
            cudaGraphExec_t& ge = r.df->gx[key];
            if (!ge) {
                cudaGraph_t gr = nullptr;
                CU(cudaStreamBeginCapture(r.st, cudaStreamCaptureModeRelaxed));
                df_launch(r, slot, host_taps, ntg);
                CU(cudaStreamEndCapture(r.st, &gr));
                CU(cudaGraphInstantiate(&ge, gr, 0));
                CU(cudaGraphDestroy(gr));
            }
            CU(cudaGraphLaunch(ge, r.st));
        } else
            df_launch(r, slot, host_taps, ntg);
        CU(cudaStreamSynchronize(r.st));
    });
    guard.ok = true;
    for (int b = 0; b + 1 < B; b++) {
        float bv = 0;
        int bi = -1;
        for (int g = 0; g < ngpu; g++) {
            const float v = R[g].df->res_host[2 * b];
            int i;
            memcpy(&i, R[g].df->res_host + 2 * b + 1, 4);
            if (bi < 0 || v > bv || (v == bv && i < bi)) {
                bv = v;
                bi = i;
            }
        }
        out[b] = bi >= 0 && bi < VOCAB ? bi : 0;   // a draft is only a proposal: a bad drafter row must not stop the engine
    }
}

// queued context rows of the slot (16 per pass), the last pass with the block
void Impl::df_flush(int slot, int B, int token, int* out) {
    DfSlot& s = dfh->slot[slot];
    std::vector<std::pair<long long, int>> live;
    for (const auto& e : s.pend) {
        if (e.first >= df_tw - DF_RING)
            live.push_back(e);
        else
            dfh->dropped++;
    }
    s.pend.clear();
    size_t i = 0;
    do {
        const int na = (int)std::min<size_t>(16, live.size() - i);
        const bool last = i + na >= live.size();
        DfPar& p = *R[0].df->par_host;
        p = DfPar{};
        p.na = na;
        for (int a = 0; a < na; a++) {
            p.ring[a] = (int)(live[i + a].first % DF_RING);
            p.cpos[a] = live[i + a].second;
            p.cslot[a] = dfh->slot_of(live[i + a].second);
        }
        p.B = last ? B : 0;
        p.bpos = s.cnt;
        p.nv = std::min(s.cnt, dfh->cap);
        p.cap = dfh->cap;
        if (na > 0 || p.B >= 2) df_pass(slot, na, false, p.B, p.bpos, token, out);
        i += na;
    } while (i < live.size());
}

// (review M3) the store rows of positions [a, b) that sit in the ring part (positions >= cap) are zeroed on every card:
// after a cut they would stay among the attended rows (positions < cap sit in slots >= the new count, outside the pass);
// a zero row is a hole (lower acceptance only, as any missing row)
void Impl::df_clear(int slot, int a, int b) {
    const int cap = dfh->cap, head = dfh->head, rl = cap - head;
    a = std::max(a, cap);
    if (b <= a) return;
    std::vector<std::pair<int, int>> runs;   // slot rows [x, y)
    if (b - a >= rl) runs.push_back({head, cap});
    else {
        const int x = dfh->slot_of(a), y = x + (b - a);
        if (y <= cap) runs.push_back({x, y});
        else runs.push_back({x, cap}), runs.push_back({head, head + (y - cap)});
    }
    pool.run([&](int g) {
        Rank& r = R[g];
        DfRank& d = *r.df;
        const size_t rw = (size_t)r.kvn * 64;   // halfs per store row
        for (int t = 0; t < 2; t++) {
            __half* base = (t ? d.Vs : d.Ks) + (size_t)slot * d.srows * rw;
            for (int l = 0; l < DF_L; l++)
                for (auto& q : runs) CU(cudaMemsetAsync(base + ((size_t)l * d.lrows + q.first) * rw, 0, (size_t)(q.second - q.first) * rw * 2, r.st));
        }
    });
}

bool Engine::has_drafter() const { return p_->dfh != nullptr; }
int Engine::draft_count(int slot) const {
    GCHECK(p_->dfh && slot >= 0 && slot < p_->cfg.slots, "draft_count: slot %d", slot);
    return p_->dfh->slot[slot].cnt;
}
void Engine::draft_commit(int slot, int row0, int n) {
    Impl& m = *p_;
    GCHECK(m.dfh && slot >= 0 && slot < m.cfg.slots && row0 >= 0 && n >= 0 && row0 + n <= m.df_last_n, "draft_commit: slot %d rows %d + %d of %d", slot,
           row0, n, m.df_last_n);
    DfSlot& s = m.dfh->slot[slot];
    for (int i = 0; i < n; i++)   // (review L5) the rows are this slot's, in order, from its context's end
        GCHECK(m.df_rs[row0 + i] == slot && m.df_rp[row0 + i] == s.cnt + i, "draft_commit: row %d of the last step is slot %d position %d, the context of slot %d is at %d",
               row0 + i, m.df_rs[row0 + i], m.df_rp[row0 + i], slot, s.cnt + i);
    for (int i = 0; i < n; i++) s.pend.push_back({m.df_last + row0 + i, s.cnt++});
    if (s.pend.size() > 4 * DF_RING) {   // never flushed (drafting is off): the oldest rows' taps left the ring long ago
        const size_t k = s.pend.size() - DF_RING;
        s.pend.erase(s.pend.begin(), s.pend.begin() + k);
        m.dfh->dropped += (long)k;
    }
}
void Engine::draft_truncate(int slot, int n) {
    Impl& m = *p_;
    GCHECK(m.dfh && slot >= 0 && slot < m.cfg.slots && n >= 0, "draft_truncate: slot %d at %d", slot, n);
    DfSlot& s = m.dfh->slot[slot];
    if (n < s.cnt) m.df_clear(slot, n, s.cnt);   // (review M3) rows of later positions in the ring would stay visible
    s.cnt = std::min(s.cnt, n);
    size_t k = 0;
    for (const auto& e : s.pend)
        if (e.second < n) s.pend[k++] = e;
    s.pend.resize(k);
}
void Engine::draft_warm(int slot, int maxb) {
    Impl& m = *p_;
    GCHECK(m.dfh && slot >= 0 && slot < m.cfg.slots && maxb >= 2 && maxb <= DF_MAXB, "draft_warm: slot %d, blocks up to %d", slot, maxb);
    const DfSlot& s = m.dfh->slot[slot];
    int out[DF_MAXB];
    // the new rows go to the pad slots behind the context (the block's own scratch rows), so the store is unchanged
    auto pass = [&](int na, int B) {
        DfPar& p = *m.R[0].df->par_host;
        p = DfPar{};
        p.na = na;
        for (int a = 0; a < na; a++) p.ring[a] = 0, p.cpos[a] = s.cnt, p.cslot[a] = m.dfh->cap + DF_PAD - 1 - a;
        p.B = B;
        p.bpos = s.cnt;
        p.nv = std::min(s.cnt, m.dfh->cap);
        p.cap = m.dfh->cap;
        m.df_pass(slot, na, false, B, s.cnt, 0, out);
    };
    pass(16, 0);
    for (int na = 1; na <= 5; na++)
        for (int B = 2; B <= maxb; B++) pass(na, B);
}
void Engine::draft_drop(int slot, int keep) {
    Impl& m = *p_;
    GCHECK(m.dfh && slot >= 0 && slot < m.cfg.slots && keep >= 0, "draft_drop: slot %d keep %d", slot, keep);
    DfSlot& s = m.dfh->slot[slot];
    if (s.pend.size() > (size_t)keep) {
        const size_t k = s.pend.size() - keep;
        s.pend.erase(s.pend.begin(), s.pend.begin() + k);
        m.dfh->dropped += (long)k;
    }
}
void Engine::draft_block(int slot, int pos, int token, int B, int* out) {
    Impl& m = *p_;
    GCHECK(!m.broken.load(), "the engine stopped after an earlier error");
    GCHECK(m.dfh && slot >= 0 && slot < m.cfg.slots, "draft_block: slot %d", slot);
    GCHECK(B <= DF_MAXB && (B < 2 || (token >= 0 && token < VOCAB && out)), "draft_block: block of %d rows (<= %d), token %d", B, DF_MAXB, token);
    GCHECK(pos == m.dfh->slot[slot].cnt, "draft_block: position %d, the drafter context holds %d rows", pos, m.dfh->slot[slot].cnt);
    m.df_flush(slot, B, token, out);
}
void Engine::draft_feed(int slot, const float* taps, int n) {
    Impl& m = *p_;
    GCHECK(!m.broken.load(), "the engine stopped after an earlier error");
    GCHECK(m.dfh && slot >= 0 && slot < m.cfg.slots && n >= 0, "draft_feed: slot %d, %d rows", slot, n);
    m.df_flush(slot, 0, 0, nullptr);   // rows queued before these
    DfSlot& s = m.dfh->slot[slot];
    for (int o = 0; o < n; o += 8) {
        const int na = std::min(8, n - o);
        DfPar& p = *m.R[0].df->par_host;
        p = DfPar{};
        p.na = na;
        uint16_t* t = (uint16_t*)m.R[0].df->tg_host;
        for (int a = 0; a < na; a++) {
            const float* src = taps + (size_t)(o + a) * DF_FC;
            for (int i = 0; i < DF_FC; i++) t[(size_t)a * DF_FC + i] = df_f2bf(src[i]);
            p.cpos[a] = s.cnt;
            p.cslot[a] = m.dfh->slot_of(s.cnt);
            s.cnt++;
        }
        p.nv = std::min(s.cnt, m.dfh->cap);
        p.cap = m.dfh->cap;
        m.df_pass(slot, na, true, 0, 0, 0, nullptr);
    }
}
double Engine::draft_vram_mb() const {
    size_t b = 0;
    for (const Rank& r : p_->R)
        if (r.df) b = std::max(b, r.df->bytes);
    return b / 1048576.0;
}

// [prefill] which own rows of the chunk keep their taps (k_df_ptap): the rows that are still in the store when the chunk
// ends, i.e. positions < head and the last cap - head of the chunk; per card at most pcap of them (the rest is dropped).
// Every card works out the plan of every card (the same arithmetic), so the compact survivor list is known everywhere:
// card g's survivors are rows poff[g] .. poff[g] + pcnt[g] of it, ppos_host their positions.
void Impl::df_prefill_plan(Rank& r, const Lane* lanes, int nl, int pos0c, int Tc) {
    DfRank& d = *r.df;
    const int cap = dfh->cap, head = dfh->head, ts = pos0c + Tc - (cap - head);
    int M = 0;
    for (int g = 0; g < ngpu; g++) {
        int b = 0;
        d.poff[g] = M;
        for (int l = 0; l < nl; l++) {
            const Lane& ln = lanes[l];
            const int lo = (int)((long long)ln.T * g / ngpu), n = (int)((long long)ln.T * (g + 1) / ngpu) - lo, p0 = ln.pos0 + lo;
            const int nh = std::max(0, std::min(n, head - p0)), s0 = std::max(nh, std::min(n, ts - p0));
            if (g == r.rank) d.pbase[l] = b, d.pnh[l] = nh, d.ps0[l] = s0;
            for (int i = 0; i < n; i++)
                if ((i < nh || i >= s0) && b < d.pcap) {
                    d.ppos_host[M++] = p0 + i;
                    b++;
                }
        }
        d.pcnt[g] = b;
    }
    d.pm = M;
}

// [prefill] the chunk's surviving rows into the bound slot's drafter context, on the compute stream after the chunk's
// last collective: every card makes c = hidden_norm(fc(taps)) of its own survivors (the whole fc, 4-bit, 64 rows per
// product, into r.Hs / r.XN rows of the compact list: free after the chunk), the peers pull the bf16 rows, then every
// card makes its heads' K / V of all survivors.
void Impl::df_prefill_ctx(Rank& r, const Lane* lanes, int nl, int pos0c, int Tc) {
    (void)lanes;
    (void)nl;
    (void)pos0c;
    (void)Tc;
    DfRank& d = *r.df;
    cudaStream_t st = r.st;
    // (review H1) the peers pull this card's final normed rows (XN) for their logits: no write to XN / Hs before every
    // card has drained those pulls
    meet(r);
    const int m = d.pcnt[r.rank], o0 = d.poff[r.rank];
    for (int o = 0; o < m; o += 64)
        df_q4m(st, d.wfc, d.ptap + (size_t)o * DF_FC, DF_FC, std::min(64, m - o), nullptr, r.Hs + (size_t)(o0 + o) * H, H);
    if (m) k_resnorm<<<(unsigned)m, 960, 0, st>>>(r.Hs + (size_t)o0 * H, nullptr, nullptr, nullptr, nullptr, d.hn, r.XN + (size_t)o0 * H, H);
    if (d.pm) CU(cudaMemcpyAsync(d.ppos, d.ppos_host, (size_t)d.pm * 4, cudaMemcpyHostToDevice, st));
    drain(st);
    meet(r);
    for (int g = 0; g < ngpu; g++)
        if (g != r.rank && d.pcnt[g])
            CU(cudaMemcpyPeerAsync(r.XN + (size_t)d.poff[g] * H, r.dev, R[g].XN + (size_t)d.poff[g] * H, R[g].dev, (size_t)d.pcnt[g] * H * 2, st));
    const int nb = std::min(64, std::max(1, (int)((size_t)std::max(lane_cap, MAXR) * r.nrows / d.nkv)));   // rows per k / v product (r.Y)
    __half* Ks = d.Ks + (size_t)b_slot * d.srows * r.kvn * HD;
    __half* Vs = d.Vs + (size_t)b_slot * d.srows * r.kvn * HD;
    for (int o = 0; o < d.pm; o += nb) {
        const int n = std::min(nb, d.pm - o);
        df_q4m(st, d.wkv, r.XN + (size_t)o * H, H, n, d.bkvall, r.Y, d.nkv);
        k_df_kvstore_p<<<dim3(n, DF_L * 2 * r.kvn), 32, 0, st>>>(r.Y, r.kvn, d.knall, d.ppos + o, dfh->cap, dfh->head, r.invf, att, Ks, Vs, d.lrows);
    }
    drain(st);
    meet(r);   // no card starts the next chunk (new rows in XN) while a peer still pulls
}
