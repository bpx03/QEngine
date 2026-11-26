// Prefill path of Engine::Impl ([ink-pf-*], branch inkling-prefill). Included inside struct Engine::Impl (ink.cu),
// so the shared file only carries the entry points. Kernels: ink_prefill.cuh.
bool pfx = !(getenv("INK_PFX") && atoi(getenv("INK_PFX")) == 0);   // grouped expert GEMM (INK_PFX=0: per-token GEMVs)
__half *pX16 = nullptr, *pH16 = nullptr;
float *pY = nullptr, *pPW = nullptr;
int *pPENT = nullptr, *pOFF = nullptr, *pUNITS = nullptr, *pNU = nullptr;
bool pfat = !(getenv("INK_PFATT") && atoi(getenv("INK_PFATT")) == 0);   // tensor-core prefill attention
bool pfmg = !(getenv("INK_PFMG") && atoi(getenv("INK_PFMG")) == 0);   // tensor-core message gates (hi/lo)
bool pcmp = getenv("INK_PATTN_CMP") != nullptr;   // compare with the fp32 CUDA-core kernels (slow)
__half* pProjT[NL] = {};
float *pAOf = nullptr, *pAOr = nullptr;
double* pErr = nullptr;
int pmaxu = 0, pcap = 0;
int ch = CH;    // prefill chunk capacity in rows (ladder 8K / 4K / 2K / 1K, [ink-pf-chunk])
int chs = CH;   // [ink-pf-wave] split size of longer prompts (<= ch; a remainder <= ch runs as one chunk)
__nv_bfloat16 *pKs = nullptr, *pVs = nullptr;   // sliding K / V staging of the chunk [NKV][ch][HD]
int* pZero = nullptr;
int* pDrPos = nullptr;   // drafter ring rows of a long chunk: first written position
float *pZp = nullptr, *pZs = nullptr;
float* pBig = nullptr;   // P (qkv out) aliased with the MoE slot outputs + h (both live in different phases)
// bytes per chunk row of everything sized by the chunk (alloc_state + pf_alloc + drafter), see alloc_state
size_t pf_row_bytes() const { return pf_row_bytes(dr_on); }
size_t pf_row_bytes(bool dr) const {
    int npmax = 0;
    for (int l = 0; l < NL; l++) npmax = std::max(npmax, L[l].NP);
    size_t b = (pfx ? 11 : 26) * (size_t)H * 4 + 3 * H * 2 + (size_t)QD * 4 + NQ * DREL * 4 + NQ * 4 + TOPK * 4 + QD * 2 + 2 * H * 2 + TOPK * 4 +
               3 * GR * 4 + NE * 4 + 2 * H * 2 + 4;
    if (pfx) b += std::max((size_t)npmax * 4, (size_t)TOPK * H * 4 + (size_t)TOPK * I * 2) + H * 2 + TOPK * 8 + 2 * KVD * 2;
    else b += (size_t)npmax * 4 + (size_t)TOPK * I * 4;
    if (dr) b += 3 * (size_t)H * 4 + (pfx ? 0 : 4 * H * 2 + 2 * KVD * 4);   // [ink-pf-dralias] DF / DKVb inside pBig
    if (tps && pfx) b += (size_t)(1 + 2 * (sh.N - 1)) * H * 4;   // [ink-tp-pf] partial rows + 2 receive slots per peer
    if (pf2) b += ((size_t)10 * H + npmax) * 4;   // [ink-tp-pf2] P and the message / attention-input rows outside pBig
    return b;
}
int pf_choose_chunk() {
    if (!pfx || !pfat || pcmp) return chs = CH;   // the per-token / CUDA-core paths keep 1K (ring slack)
    if (const char* e = getenv("INK_CHUNK")) return chs = std::max(64, std::min(8192, atoi(e)));
    size_t fr = 0, tot = 0;
    CU(cudaMemGetInfo(&fr, &tot));
    // review M1: the chunk comes from the card's total memory minus this engine's own allocations (the same on every
    // run of a configuration: chunking changes cuBLAS kernels / sum orders); less free memory (another process) only
    // steps it down further
    const size_t own = tot > dev_bytes + ((size_t)450 << 20) ? tot - dev_bytes - ((size_t)450 << 20) : 0;   // CUDA context ~0.48 GB measured
    fprintf(stderr, "[ink-pf] memory: total %.2f GB, engine %.2f GB, free %.2f GB (context + others %.2f GB)\n", tot / 1e9, dev_bytes / 1e9,
            fr / 1e9, (tot - fr - dev_bytes) / 1e9);
    fr = std::min(fr, own);
    size_t fixed = (size_t)160 << 20;   // measured ~70 MB allocated after this point + graph / context headroom   // fixed buffers after this point (attention partials, head rows, workspace) + headroom
    fixed += batch_row_bytes() + srv_graph_room() + pf2_bytes();   // review tp2 M3   // [ink-srv-merge] the MROW-row buffers of the batch build, row-step graphs
    if (dr_on) {   // drafter K / V: E4M3 + fp32 scale per (head, slot) ([ink-dr-fp8]), slots as dr_alloc (review2 L4)
        size_t slots = 0;
        // [ink-srv-merge] every sequence slot has its drafter caches; [ink-srv-capi] the global layer's is one pool for
        // all slots (dr_alloc), counted once (S x the pool reserved ~0.8 GB too much at 8 slots: 256-row chunks)
        for (int i = 0; i < 3; i++) slots += (size_t)dr_cap(i) * (i == 1 && !getenv("INK_DR_TEST") ? 1 : S);   // (dsko's rule; SKO_G may not exist yet)
        // block buffers for min(S, 8) sequences
        fixed += slots * NKV * (2 * HD + 8) + ((size_t)96 << 20) + dr_blk_bytes(std::min(S, 8)) - dr_blk_bytes(1);
    }
    const size_t rb = pf_row_bytes();
    int c = 8192;
    while (c > 256 && fixed + (size_t)c * rb > fr) c /= 2;   // ladder 8K .. 1K; below 1K only under VRAM pressure (never refuse)
    // [ink-pf-wave] grow to the largest multiple of 64 rows that fits (<= 8K), then pick, within the top quarter,
    // the size whose attention grid (rows / 64 x the KV heads, per TP sub-chunk; one CTA per SM) fills its last wave best
    if (!getenv("INK_NOWAVE")) {
        int cm = c;
        while (cm + 64 <= 8192 && fixed + (size_t)(cm + 64) * rb <= fr) cm += 64;
        int dev = 0, nsm = 82;
        CU(cudaGetDevice(&dev));
        CU(cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev));
        double best = -1;
        for (int t = cm; t >= std::max(64, cm * 3 / 4); t -= 64) {
            // last-wave fill x a mild preference for larger chunks (per-chunk fixed costs); exponent from 120K runs:
            // drafter on 1920 / 2496 / 2560 rows 25.42 / 25.18 / 24.74 s, off 7872 / 8192 rows 21.75 / 22.04 s
            // [ink-pf3-wave] a TP chunk runs as pfk sub-chunks (rows split as in the prefill wavefront), each attention
            // grid = its row blocks x the rank's KV heads: the mean fill of those grids
            double fill = 0;
            const int nsub = tps && pf2 ? pfk : 1;
            for (int j = 0; j < nsub; j++) {
                const int a0 = (int)((long long)t * j / nsub) & ~63, a1 = j + 1 < nsub ? (int)((long long)t * (j + 1) / nsub) & ~63 : t;
                const int ctas = (a1 - a0 + 63) / 64 * nkvA(), waves = (ctas + nsm - 1) / nsm;
                fill += ctas ? (double)ctas / ((double)waves * nsm) / nsub : 0;
            }
            const double e = fill * pow((double)t, 0.3);
            if (e > best * (1 + 1e-9)) best = e, c = t;   // ties keep the larger size
        }
    }
    int cap = c;
    if (!getenv("INK_NOWAVE")) {
        while (cap + 64 <= 8192 && fixed + (size_t)(cap + 64) * rb <= fr) cap += 64;
    }
    chs = c;
    fprintf(stderr, "[ink-pf] chunk %d rows, split %d (%.0f KB/row, %.2f GB free, %.2f GB for the chunk)\n", cap, c, rb / 1024.0, fr / 1e9,
            (double)cap * rb / 1e9);
    return cap;
}
float* pf_big(size_t C, int npmax) {
    const size_t n = std::max((size_t)npmax * 4, (size_t)TOPK * H * 4 + (size_t)TOPK * I * 2) * C;
    pBig = (float*)dmalloc(n, &dev_bytes);
    return pBig;
}
// [ink-pf-qfuse] k_qpost inside k_pattn's prologue (k_pattn<., true>); the fused form's main loop runs a few % slower,
// so it is used only where the per-row saving wins: sliding layers, and global layers up to QF_GMAX keys
static constexpr int QF_GMAX = 16384;
bool pqfuse(const Layer& y, int T) const {
    static const bool off = getenv("INK_NOQFUSE") != nullptr;
    return pfx && pfat && !pcmp && !off && (!y.full || pos + T <= QF_GMAX);
}
bool pf_psplit() const { return !(pfx && pfat && !pcmp); }   // the CUDA-core prefill attention (PSPLIT partials) can run
bool pf_stage(const AppState& s) const { return pfx && pfat && s.ring && !vR; }   // unit grid bound, rows the prefill buffers hold
void pf_alloc(int C) {
    pcap = C;
    if (tps) {   // [ink-tp-pf] collective buffers of the chunk: partial rows, receive rows [slot][peer][C][H], events
        tpP = (float*)dmalloc((size_t)C * H * 4, &dev_bytes);
        if (sh.N > 1) tpRX = (float*)dmalloc((size_t)2 * (sh.N - 1) * C * H * 4, &dev_bytes);
        for (int j = 0; j < PS_MAX; j++)
            for (int i = 0; i < 2; i++) {
                CU(cudaEventCreateWithFlags(&tpES[j][i], cudaEventDisableTiming));
                CU(cudaEventCreateWithFlags(&tpEF[j][i], cudaEventDisableTiming));
            }
    }
    const size_t n8 = (size_t)C * TOPK;
    pmaxu = (int)(n8 / pf::XBN) + NE;
    pX16 = (__half*)dmalloc((size_t)C * H * 2, &dev_bytes);
    pY = pBig;   // [C * 8][H] fp32, then h [C * 8][I] fp16 behind it (P aliases the front: dead after attention)
    pH16 = (__half*)(pBig + n8 * H);
    pKs = (__nv_bfloat16*)dmalloc((size_t)NKV * C * HD * 2, &dev_bytes);
    pVs = (__nv_bfloat16*)dmalloc((size_t)NKV * C * HD * 2, &dev_bytes);
    pZero = (int*)dmalloc(16, &dev_bytes);
    pDrPos = (int*)dmalloc(16, &dev_bytes);
    {   // K-split partials of the message-gate z (chunks <= 4K): 4 x 3 gates x 2K rows or 2 x 3 x 4K
        const size_t rows = 3 * std::max((size_t)4 * std::min(C, 2048), C > 2048 ? (size_t)2 * std::min(C, 4096) : 0);   // review M2
        pZp = (float*)dmalloc(rows * 2 * GR * 4, &dev_bytes);
        pZs = (float*)dmalloc(rows * 2 * 4, &dev_bytes);
        for (int j = 1; pf2 && j < pfk; j++) {
            pZp2[j] = (float*)dmalloc(rows * 2 * GR * 4, &dev_bytes);
            pZs2[j] = (float*)dmalloc(rows * 2 * 4, &dev_bytes);
        }
    }
    CU(cudaMemset(pZero, 0, 16));
    pPW = (float*)dmalloc(n8 * 4, &dev_bytes);
    pPENT = (int*)dmalloc(n8 * 4, &dev_bytes);
    pOFF = (int*)dmalloc((NE + 1) * 4, &dev_bytes);
    pUNITS = (int*)dmalloc((size_t)pmaxu * 12, &dev_bytes);
    pNU = (int*)dmalloc(16, &dev_bytes);
    for (int j = 1; pf2 && j < pfk; j++) {   // [ink-tp-pf2] sub-chunk routing plans
        pOFF2[j] = (int*)dmalloc((NE + 1) * 4, &dev_bytes);
        pUNITS2[j] = (int*)dmalloc((size_t)pmaxu * 12, &dev_bytes);
        pNU2[j] = (int*)dmalloc(16, &dev_bytes);
    }
    CU(cudaFuncSetAttribute(pf::k_xg<0, 2, 0, 8>, cudaFuncAttributeMaxDynamicSharedMemorySize, pf::xg_smem<0, 2>(H)));
    CU(cudaFuncSetAttribute(pf::k_xg<0, 2, 1, 8>, cudaFuncAttributeMaxDynamicSharedMemorySize, pf::xg_smem<0, 2>(I)));
    CU(cudaFuncSetAttribute(pf::k_xg<1, 2, 0, 8>, cudaFuncAttributeMaxDynamicSharedMemorySize, pf::xg_smem<1, 2>(H)));
    CU(cudaFuncSetAttribute(pf::k_xg<1, 2, 1, 8>, cudaFuncAttributeMaxDynamicSharedMemorySize, pf::xg_smem<1, 2>(I)));
    CU(cudaFuncSetAttribute(pf::k_pattn<false>, cudaFuncAttributeMaxDynamicSharedMemorySize, pf::PA_SMEM));
    CU(cudaFuncSetAttribute(pf::k_pattn<true>, cudaFuncAttributeMaxDynamicSharedMemorySize, pf::PA_SMEM));
    CU(cudaFuncSetAttribute(pf::k_pattn<false, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, pf::PA_SMEM));
    CU(cudaFuncSetAttribute(pf::k_pattn<true, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, pf::PA_SMEM));
    {   // projT (fp16 [REL][DREL]) per layer; bf16 -> fp16 is exact for normal fp16 magnitudes (count the others)
        long long inex = 0;
        for (int l = 0; l < NL; l++) {
            const Tens& pr = f.get("model.llm.layers." + std::to_string(l) + ".attn.rel_logits_proj.proj");
            std::vector<__half> h((size_t)REL * DREL);
            for (int i = 0; i < DREL; i++)
                for (int d = 0; d < REL; d++) {
                    const float v = bf2f(((const uint16_t*)pr.p)[(size_t)i * REL + d]);
                    h[(size_t)d * DREL + i] = __float2half_rn(v);
                    inex += (float)__half2float(h[(size_t)d * DREL + i]) != v;
                }
            pProjT[l] = (__half*)dmalloc(h.size() * 2, &dev_bytes);
            CU(cudaMemcpy(pProjT[l], h.data(), h.size() * 2, cudaMemcpyHostToDevice));
        }
        fprintf(stderr, "[ink-pf] rel-bias proj in fp16: %lld of %d values inexact\n", inex, NL * REL * DREL);
    }
    if (pcmp) {
        pAOf = (float*)dmalloc((size_t)C * QD * 4, &dev_bytes);
        pAOr = (float*)dmalloc((size_t)C * QD * 4, &dev_bytes);
        pErr = (double*)dmalloc(64, &dev_bytes);
        CU(cudaMemset(pErr, 0, 64));
    }
    // MXFP8 exactness condition of the fp16 decode: every E8M0 scale in [113, 128]
    const std::string pfx_ = "model.llm.layers.";
    for (int l = 0; l < NL; l++)
        if (L[l].fmt == 1)
            for (const char* w : {"w13", "w2"}) {
                const Tens& s = f.get(pfx_ + std::to_string(l) + ".mlp.experts." + w + "_weight_scale");
                const auto mm = std::minmax_element(s.p, s.p + s.n);
                ICHECK(*mm.first >= 113 && *mm.second <= 128, "layer %d %s: E8M0 scales %d..%d outside the exact fp16 range", l, w,
                       *mm.first, *mm.second);
            }
}
// routed experts of T rows: XMF (normed fp32) + SC (router scores) -> MO [T][H] (sum over the 8 slots, k order)
void pf_moe(const Layer& y, int T, int part = 3) {   // [ink-tp-sp] part 1: fp16 input + top-8 of the rows; 2: plan + GEMMs
    const int n8 = T * TOPK;
    if (part & 1) {
        if (!pfuse || getenv("INK_NOF2H")) pf::k_f2h<<<(T * H + 255) / 256, 256, 0, st>>>(XMF, pX16, (size_t)T * H);   // else written by k_prouter
        pf::k_ptopk<<<(T + 7) / 8, 256, 0, st>>>(SC, y.rb, T, IDS, WTS);
    }
    if (!(part & 2)) return;
    pf::k_pplan<<<1, 1024, 0, st>>>(IDS, n8, pOFF, pUNITS, pNU);
    pf::k_pfill<<<NE, pf::PF_NT, 0, st>>>(IDS, WTS, n8, pOFF, pPENT, pPW);
    const int mu = n8 / pf::XBN + NE;
    const int Ie = tps ? sh.nu() * TP_UNIT : I;   // [ink-tp-pf] a TP rank: its intermediate columns (compact weights)
    const pf::XgArgs ag{y.c13, y.s13, y.fmt ? nullptr : y.f13, 2 * Ie, H, pX16, pPENT, pPW, pUNITS, pNU, 1, pH16, nullptr};
    const pf::XgArgs ad{y.c2, y.s2, y.fmt ? nullptr : y.f2, H, Ie, pH16, pPENT, pPW, pUNITS, pNU, 0, nullptr, pY};
    if (!Ie) return;
    if (y.fmt == 0) {
        pf::k_xg<0, 2, 0, 8><<<dim3(2 * Ie / pf::XBM, mu), 256, pf::xg_smem<0, 2>(H), st>>>(ag);
        pf::k_xg<0, 2, 1, 8><<<dim3(H / pf::XBM, mu), 256, pf::xg_smem<0, 2>(Ie), st>>>(ad);
    } else {
        pf::k_xg<1, 2, 0, 8><<<dim3(2 * Ie / pf::XBM, mu), 256, pf::xg_smem<1, 2>(H), st>>>(ag);
        pf::k_xg<1, 2, 1, 8><<<dim3(H / pf::XBM, mu), 256, pf::xg_smem<1, 2>(Ie), st>>>(ad);
    }
    if (!pfuse && !tps) pf::k_pcomb<<<(T * (H / 4) + 255) / 256, 256, 0, st>>>(pY, T, MO);   // (TP: combined into the partial; review tp2 L9)
}
bool pfuse = !(getenv("INK_PFFUSE") && atoi(getenv("INK_PFFUSE")) == 0);   // fused row kernels ([ink-pf-rowfuse])
// post-MoE: combine + conv + residual -> hout, conv state from the last 3 MO rows
void pf_conv_out(const Layer& y, const AppState& s, int T, float* hout) {
    if (getenv("INK_PCONV_CHK")) {   // debug: reference k_pcomb + k_conv into scratch, compare
        static float *ref = nullptr, *mo2 = nullptr;
        static double* acc = nullptr;
        if (!ref) {
            CU(cudaMalloc(&ref, (size_t)ch * H * 4));
            CU(cudaMalloc(&mo2, (size_t)ch * H * 4));
            CU(cudaMalloc(&acc, 64));
        }
        CU(cudaMemsetAsync(acc, 0, 64, st));
        pf::k_pcomb<<<(T * (H / 4) + 255) / 256, 256, 0, st>>>(pY, T, mo2);
        k_conv<<<(T * H + 255) / 256, 256, 0, st>>>(mo2, T, H, s.mst, y.mw, Rr, ref);
        pf::k_pconv<<<dim3(H / 128, (T + pf::PC_R - 1) / pf::PC_R), 128, 0, st>>>(pY, T, s.mst, y.mw, Rr, hout, MO);
        pf::k_errsum<<<256, 256, 0, st>>>(hout, ref, (size_t)T * H, acc);
        double e[3];
        CU(cudaMemcpyAsync(e, acc, 24, cudaMemcpyDeviceToHost, st));
        CU(cudaStreamSynchronize(st));
        float mx;
        memcpy(&mx, &e[2], 4);
        fprintf(stderr, "[pconv chk] rel %.3e max %.3e\n", sqrt(e[0] / e[1]), mx);
    } else
    pf::k_pconv<<<dim3(H / 128, (T + pf::PC_R - 1) / pf::PC_R), 128, 0, st>>>(pY, T, s.mst, y.mw, Rr, hout, MO);
    k_cstate<<<(H + 255) / 256, 256, 0, st>>>(MO, H, T, H, s.mst);
}
// prefill attention of T rows at positions pos.. (K / V of the rows already in the cache)
void pf_attn(const Layer& y, const AppState& s, int l, int T) {
    const bool sg = pf_stage(s);
    if (s.ring && (s.cap & (s.cap - 1))) throw std::runtime_error("k_pattn: ring capacity must be a power of two");   // [ink-pf-attn-ld] slot mask
    const pf::A_pattn a{Qf, Rf, Sf, P, pcP(T, y) ? y.NPr : y.NP, pProjT[l], s.K, s.V, s.cap, s.ring, !y.full, pos, T, AO, pcmp ? pAOf : nullptr,
                        sg ? pKs : nullptr, sg ? pVs : nullptr, ch, pqfuse(y, T) ? y.qs : nullptr, y.rs, y.full, kv0A(),
                        pcP(T, y) ? y.dq : 0, pcP(T, y) ? y.dr : 0, pcP(T, y) ? y.dg : 0, pcP(T, y) ? y.ds : 0};
    if (!nkvA()) return;   // [ink-tp-pf] a TP rank without attention heads
    if (a.qs) (y.full ? pf::k_pattn<false, true> : pf::k_pattn<true, true>)<<<dim3((T + 63) / 64, nkvA()), 384, pf::PA_SMEM, st>>>(a);
    else (y.full ? pf::k_pattn<false> : pf::k_pattn<true>)<<<dim3((T + 63) / 64, nkvA()), 384, pf::PA_SMEM, st>>>(a);
    static const bool qchk = getenv("INK_QFUSE_CHK") != nullptr;   // review3 L2: fused vs k_qpost + plain kernel, bitwise
    if (qchk && a.qs) {
        static bf16* ao2 = nullptr;
        static unsigned long long* bad = nullptr;
        if (!ao2) {
            CU(cudaMalloc(&ao2, (size_t)ch * QD * 2));
            CU(cudaMalloc(&bad, 8));
        }
        k_qpost<<<dim3(T, NQ), 128, 0, st>>>(P, y.NP, y.qs, y.rs, y.full, dpos, Qf, Rf, Sf);
        pf::A_pattn b = a;
        b.qs = nullptr;
        b.AO = ao2;
        b.AOf = nullptr;
        (y.full ? pf::k_pattn<false> : pf::k_pattn<true>)<<<dim3((T + 63) / 64, NKV), 384, pf::PA_SMEM, st>>>(b);
        CU(cudaMemsetAsync(bad, 0, 8, st));
        k_cmp_u32<<<256, 256, 0, st>>>((const uint32_t*)AO, (const uint32_t*)ao2, (size_t)T * QD / 2, bad);
        unsigned long long hb = 0;
        CU(cudaMemcpyAsync(&hb, bad, 8, cudaMemcpyDeviceToHost, st));
        CU(cudaStreamSynchronize(st));
        static long long nchk = 0, nbad = 0;
        nchk++;
        nbad += hb != 0;
        if (hb || nchk % 48 == 0) fprintf(stderr, "[qfuse-chk] layer %d T %d: %llu words differ (%lld / %lld launches differ)\n", l, T, hb, nbad, nchk);
    }
    if (sg) {   // the chunk's last rows (up to the ring size) into the ring
        const int j0 = std::max(0, T - s.cap);
        pf::k_ringcp<<<dim3(T - j0, nkvA()), 32, 0, st>>>(pKs, pVs, ch, j0, s.K, s.V, s.cap, pos, kv0A());
    }
    if (pcmp) {   // reference: CUDA-core split kernel + fp32 merge
        const int ns = attn_splits(T, !y.full);
        go(k_attn_part, 0, dim3(dim3(T, NQ, ns)), 128, A_attn_part{Qf, Rf, y.proj, s.K, s.V, s.cap, s.ring, !y.full, dpos, PM, PL, PA});
        pf::k_pcomb_ref<<<dim3(T, NQ), 128, 0, st>>>(PM, PL, PA, ns, Sf, P, y.NP, pAOr);
        pf::k_errsum<<<256, 256, 0, st>>>(pAOf, pAOr, (size_t)T * QD, pErr + (y.full ? 3 : 0));
    }
}
void pf_report() {
    if (!pcmp) return;
    double e[6];
    CU(cudaMemcpy(e, pErr, 48, cudaMemcpyDeviceToHost));
    float mx0, mx1;
    memcpy(&mx0, &e[2], 4);
    memcpy(&mx1, &e[5], 4);
    fprintf(stderr, "[ink-pf] attention vs fp32 CUDA-core: sliding rel L2 %.3e max %.3e | global rel L2 %.3e max %.3e\n", sqrt(e[0] / e[1]), mx0,
            sqrt(e[3] / e[4]), mx1);
}
// message gates of T rows (ng gates): MgArgs as the decode kernels, z in ZB
void pf_mg(const MgArgs& m, int ng, int T) {
    pf::PMg a{};
    for (int g = 0; g < ng; g++) {
        a.h[g] = m.h[g];
        a.m[g] = m.m[g];
        a.down[g] = m.down[g];
        a.f[g] = m.f[g];
        a.g[g] = m.g[g];
        a.out[g] = m.out[g];
    }
    a.z = ZB;
    a.T = T;
    const int ks = T <= 2048 ? 4 : T <= 4096 ? 2 : 1;   // [ink-pf-mgz-ks] fill the SMs at small chunks
    if (ks > 1) {
        a.zp = pZp;
        a.ssp = pZs;
    }
    pf::k_pmgz<<<dim3((T + 63) / 64, ng, ks), 128, 0, st>>>(a);
    if (ks > 1) pf::k_pmgzr<<<(ng * T * GR + 255) / 256, 256, 0, st>>>(pZp, pZs, ks, ng, T, ZB);
    pf::k_pmgo<<<dim3((T + 63) / 64, H / 64, ng), 256, 0, st>>>(a);
}
// item 7 [ink-pf-ckpt]: prefix checkpoints in pinned host RAM (LRU archive). State of a prefix p: K / V rows [0, p) of
// the 8 global applications, the last min(p, 1024) positions of the 40 sliding rings, the 4 conv states of every
// application, the drafter's K / V (when loaded), the token history, the last row's final hidden (argmax after a
// restore without a suffix). Block states / residuals / messages are per token: nothing else carries.
struct PfSnap {
    int pos = 0;
    std::vector<int> hist;
    uint8_t* host = nullptr;
    size_t bytes = 0;
    uint64_t used = 0;
};
std::map<int, PfSnap> parch;
size_t parch_bytes = 0, parch_cap = (size_t)(getenv("INK_ARCHIVE_GB") ? atof(getenv("INK_ARCHIVE_GB")) : 64.0) * (1ull << 30);
int parch_next = 1;
uint64_t parch_clock = 0;
// copy list of (device pointer, bytes) for the state at prefix p; dir 0: device -> host, 1: host -> device
template <class F>
void pf_state_ranges(int p, F&& f) {
    for (int a = 0; a < NAPP; a++) {
        const AppState& s = A[a];
        for (int kv = 0; kv < 2; kv++) {
            bf16* base = kv ? s.V : s.K;
            for (int h = 0; h < NKV; h++) {
                bf16* hb = base + (size_t)h * s.cap * HD;
                if (!s.ring) f((uint8_t*)hb, (size_t)p * HD * 2);
                else {   // positions [max(0, p - WIN), p) in slot space (wraps once at most)
                    const int lo = std::max(0, p - WIN), n = p - lo, s0 = lo % s.cap, n1 = std::min(n, s.cap - s0);
                    f((uint8_t*)(hb + (size_t)s0 * HD), (size_t)n1 * HD * 2);
                    if (n > n1) f((uint8_t*)hb, (size_t)(n - n1) * HD * 2);
                }
            }
        }
        f((uint8_t*)s.kst, 3 * KVD * 4);
        f((uint8_t*)s.vst, 3 * KVD * 4);
        f((uint8_t*)s.ast, 3 * H * 4);
        f((uint8_t*)s.mst, 3 * H * 4);
    }
    if (dr_on)
        for (int i = 0; i < 3; i++)
            for (int kv = 0; kv < 2; kv++) {   // [ink-dr-fp8] E4M3 rows + one fp32 scale per (head, slot)
                uint8_t* base = kv ? dV(i) : dK(i);   // the selected slot's drafter caches (batch)
                float* sb = kv ? dVS(i) : dKS(i);
                const bool ring = i != 1;
                const size_t n = ring ? dcap[i] : std::min(p + 64, dcap[i]);
                for (int h = 0; h < NKV; h++) {
                    f(base + (size_t)h * dcap[i] * HD, n * HD);
                    f((uint8_t*)(sb + (size_t)h * dcap[i]), n * 4);
                }
            }
    if (dr_on) f((uint8_t*)FM, H * 2);   // drafter: final feature of the last row (review M3)
    f((uint8_t*)FIN, H * 4);    // row 0 after pf_last_row()
    f((uint8_t*)FINB, H * 2);
}
void pf_last_row() {   // the last forward's last row -> row 0 of FIN / FINB (argmax_last(1) reads row 0)
    if (last_T > 1) {
        CU(cudaMemcpyAsync(FIN, FIN + (size_t)(last_T - 1) * H, H * 4, cudaMemcpyDeviceToDevice, st));
        CU(cudaMemcpyAsync(FINB, FINB + (size_t)(last_T - 1) * H, H * 2, cudaMemcpyDeviceToDevice, st));
        last_T = 1;
    }
}
int pf_save() {
    pf_last_row();
    size_t n = 0;
    pf_state_ranges(pos, [&](uint8_t*, size_t b) { n += b; });
    while (!parch.empty() && parch_bytes + n > parch_cap) {   // LRU eviction
        auto v = parch.begin();
        for (auto it = parch.begin(); it != parch.end(); ++it)
            if (it->second.used < v->second.used) v = it;
        CU(cudaFreeHost(v->second.host));
        parch_bytes -= v->second.bytes;
        parch.erase(v);
    }
    if (n > parch_cap) {   // review L4: larger than the whole archive
        fprintf(stderr, "[ink-pf] checkpoint skipped: %.2f GB > archive cap\n", n / 1e9);
        return 0;
    }
    PfSnap sn;
    sn.pos = pos;
    sn.hist = hist;
    sn.bytes = n;
    sn.used = ++parch_clock;
    if (cudaHostAlloc((void**)&sn.host, n, 0) != cudaSuccess) {   // host RAM pressure: no checkpoint, never a failure
        cudaGetLastError();
        fprintf(stderr, "[ink-pf] checkpoint skipped: no %.2f GB of pinned host memory\n", n / 1e9);
        return 0;
    }
    size_t o = 0;
    pf_state_ranges(pos, [&](uint8_t* d, size_t b) {
        CU(cudaMemcpyAsync(sn.host + o, d, b, cudaMemcpyDeviceToHost, st));
        o += b;
    });
    CU(cudaStreamSynchronize(st));
    parch_bytes += n;
    const int id = parch_next++;
    parch[id] = std::move(sn);
    return id;
}
bool pf_restore(int id) {
    auto it = parch.find(id);
    if (it == parch.end()) return false;
    PfSnap& sn = it->second;
    sn.used = ++parch_clock;
    reset();
    size_t o = 0;
    pf_state_ranges(sn.pos, [&](uint8_t* d, size_t b) {
        CU(cudaMemcpyAsync(d, sn.host + o, b, cudaMemcpyHostToDevice, st));
        o += b;
    });
    pos = sn.pos;
    hist = sn.hist;
    lk_map.clear();
    lk_done = 0;
    last_T = 1;
    CU(cudaStreamSynchronize(st));
    return true;
}
// start-up warm-up ([ink-pf-warm]): one prefill chunk of 2,048 dummy tokens loads the prefill kernels (lazy module
// loading) and cuBLAS's kernels for that shape, so the first request runs warm; state reset afterwards
void pf_warm() {
    if (!pfx || getenv("INK_NOWARM")) return;
    const int n = std::min(2048, ch);
    std::vector<int> ids(n);
    for (int i = 0; i < n; i++) ids[i] = 1000 + (i * 7919) % 150000;
    cut = (getenv("INK_NOCUT") || (dr_on && getenv("INK_NOCUTDR"))) ? 0 : 2;   // [ink-pf-cut] the tail shapes of the last application too
    cut_s = dr_on ? (getenv("INK_CUTW") ? std::max(16, atoi(getenv("INK_CUTW"))) : 1024) : CUT_S;
    forward(ids.data(), n);
    cut = 0;
    CU(cudaStreamSynchronize(st));
    for (int a = 0; a < NAPP; a++) {   // review L6: defined (zero) cache contents again, as after alloc_state
        CU(cudaMemsetAsync(A[a].K, 0, (size_t)NKV * A[a].cap * HD * 2, st));
        CU(cudaMemsetAsync(A[a].V, 0, (size_t)NKV * A[a].cap * HD * 2, st));
    }
    if (dr_on)
        for (int i = 0; i < 3; i++) {
            CU(cudaMemsetAsync(DK[i], 0, (size_t)NKV * dcap[i] * HD, st));   // [ink-dr-fp8] E4M3 bytes + scales
            CU(cudaMemsetAsync(DV[i], 0, (size_t)NKV * dcap[i] * HD, st));
            CU(cudaMemsetAsync(DKS[i], 0, (size_t)NKV * dcap[i] * 4, st));
            CU(cudaMemsetAsync(DVS[i], 0, (size_t)NKV * dcap[i] * 4, st));
        }
    reset();
    CU(cudaStreamSynchronize(st));
}
void pf_free() {   // review L4: pinned snapshots back to the host
    for (auto& kv : parch) cudaFreeHost(kv.second.host);
    parch.clear();
    parch_bytes = 0;
    if (rthr.joinable()) rthr.join();
    if (rpend) {   // (a pending snapshot: its copy ends, its vector is not filled any more)
        cudaEventSynchronize(ev_r1);
        rpend = nullptr;
    }
    if (stg) cudaFreeHost(stg);   // [ink-srv-gap] the staging buffer
    stg = nullptr;
    stg_cap = 0;
}
