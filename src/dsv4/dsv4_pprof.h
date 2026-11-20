// DSV4_PPROF [dsv4-pf-pprof]: prefill stage profile (opt-in, never on by default; included by dsv4.cu after PP()).
// CUDA events at fixed points of every layer on the compute and copy streams of all cards, read ONCE at the end of the
// prefill (no per-layer sync: the overlap stays real, unlike DSV4_PTIME). Each event is stamped with the host time of its
// record, so an interval whose start the GPU reached before the host had enqueued the work shows host latency.
//   DSV4_PPROF=1 (or "set pprof 1" in a jobs file): stages per layer, remote cards, links, host timers, counters
//   DSV4_PPROF=2: + one in 8 routed experts per card timed by sub-stage (wait, dequant, gather + GEMM w1/w3, GLU + q_fp8,
//                  GEMM w2, scatter / D2H), by class and row bin
// An interval between two consecutive events of one stream is charged to the later event's stage.
#include <array>
#include <sched.h>
static int g_pprof = getenv("DSV4_PPROF") ? atoi(getenv("DSV4_PPROF")) : 0;
enum PfK : int {
    PK_HAND, PK_L0, PK_ENGRAM, PK_HCPRE, PK_STG, PK_QKV, PK_CMPIDX, PK_WQB, PK_ATT, PK_WOA, PK_WOB, PK_HCFFN, PK_ROUTE,
    PK_SHARED, PK_MRES, PK_MHOST, PK_POST,   // owner compute stream
    PK_RWAIT, PK_RXQ, PK_REXP,               // remote compute stream: wait for the owner's xq, xq / rows copies, experts
    PK_CPWAIT, PK_COPY,                      // copy streams: wait for a free stage buffer, the copy
    PK_XPRE, PK_XWAIT, PK_XDQ, PK_XG1, PK_XGQ, PK_XG2, PK_XSC,   // sampled experts (PPROF=2)
    PK_EMBED, PK_TAIL, PK_N
};
static const char* PKN[PK_N] = {"pipe wait+handoff", "layer start", "engram", "hc mixes+pre", "attn share stage", "wq_a+wkv+kv",
                                "compressor+indexer", "wq_b", "attn core", "wo_a", "wo_b", "hc post+ffn pre", "router+route D2H",
                                "shared expert", "MoE resident", "MoE host", "hc post", "remote wait", "remote xq/rows",
                                "remote experts", "copy wait", "copy", "x pre", "x wait", "x dequant", "x gather+gemm13",
                                "x glu+q", "x gemm2", "x scatter", "embed", "tail (draft window, head)"};
struct PfEv {
    int card, strm, l, k, fam, sid, aux;   // strm 0 compute, 1 copy; fam: stage the interval counts for in the tables
    double h, bytes;                       // host ms since h0 at the record; bytes (copies)
    cudaEvent_t ev;
};
struct PfLayerHost { double rsync = 0, build = 0, enq = 0, egw = 0, ssrc = 0; int nssrc = 0, cpu = -1; };
struct PfCnt {   // per layer and card: experts / rows by class
    int ex[10] = {}, rows[10] = {};
};
enum { PC_LOCAL, PC_REMRES, PC_POOLREM, PC_HOSTOWN, PC_HOSTREM, PC_P2POWN, PC_BORROW, PC_SEED, PC_BACK, PC_COPYLESS, PC_N };
static const char* PCN[PC_N] = {"owner-local", "remote-resident", "pool-remote", "host-staged on owner", "host-staged on remote",
                                "peer-staged on owner", "borrowed", "seeded", "sent back to owner", "copy-less"};
struct PfState {
    bool on = false;
    int lvl = 0, nsamp[3] = {};
    std::vector<cudaEvent_t> pool[3];
    size_t used[3] = {};
    std::vector<PfEv> evs;
    cudaEvent_t t0[3] = {};
    clk::time_point h0;
    PfLayerHost lh[NL];
    PfCnt cnt[NL][3];
    long long hist_ex[7] = {}, hist_rows[7] = {};
    double hist_bytes[7] = {};
    // host timers (ms)
    double embed = 0, egsub = 0, egdone = -1, borrow_on = 0, borrow_off = 0, draft = 0, head = 0, ring = 0, fwd = 0;
    long egrows = 0;
    int cpu0 = -1, cpu1 = -1, ntok = 0, nchunk = 0;
    std::vector<int> cpus;
};
static PfState g_pf;
static inline bool pf_on() { return g_pf.on; }
static double pf_hms() { return std::chrono::duration<double, std::milli>(clk::now() - g_pf.h0).count(); }
static cudaEvent_t pf_ev(Engine* E, int card) {
    auto& P = g_pf.pool[card];
    if (g_pf.used[card] == P.size()) {   // grow (a long prompt): created on the card
        DevGuard dg(E->cards[card].dev);
        const size_t add = std::max<size_t>(4096, P.size());
        for (size_t i = 0; i < add; i++) {
            cudaEvent_t ev;
            DCK(cudaEventCreate(&ev));
            P.push_back(ev);
        }
    }
    return P[g_pf.used[card]++];
}
// record a profile event on stream s of card `card` (s must belong to that card)
static std::mutex g_pf_mu;   // [dsv4-pf-mtq]: helper threads record too
static std::atomic<int> g_pf_sid{0};
static int pf_rec(Engine* E, int card, int strm, cudaStream_t s, int l, int k, int fam = -1, double bytes = 0, int sid = -1,
                  int aux = 0) {
    if (!g_pf.on || tl_cap) return -1;
    std::lock_guard<std::mutex> lk(g_pf_mu);
    PfEv v;
    v.card = card; v.strm = strm; v.l = l; v.k = k; v.fam = fam < 0 ? k : fam; v.sid = sid; v.aux = aux; v.bytes = bytes;
    v.ev = pf_ev(E, card);
    DCK(cudaEventRecord(v.ev, s));
    v.h = pf_hms();
    g_pf.evs.push_back(v);
    return (int)g_pf.evs.size() - 1;
}
static int pf_card(Engine* E, const Card& c) { return (int)(&c - E->cards.data()); }
static void pf_begin(Engine* E, int ntok) {
    g_pf.on = false;
    if (g_pprof <= 0 || E->cards.size() > 3) return;
    const int nc = (int)E->cards.size();
    for (int k = 0; k < nc; k++) {   // pool and anchors created before the clock starts
        DevGuard dg(E->cards[k].dev);
        if (!g_pf.t0[k]) DCK(cudaEventCreate(&g_pf.t0[k]));
        const size_t want = std::min<size_t>(16384 + (size_t)ntok * 12, 65536);   // review pf L4 (grows on demand)
        while (g_pf.pool[k].size() < want) {
            cudaEvent_t ev;
            DCK(cudaEventCreate(&ev));
            g_pf.pool[k].push_back(ev);
        }
        g_pf.used[k] = 0;
        DCK(cudaDeviceSynchronize());
    }
    g_pf.evs.clear();
    g_pf.evs.reserve(65536);
    for (auto& x : g_pf.lh) x = PfLayerHost();
    for (auto& a : g_pf.cnt) for (auto& b : a) b = PfCnt();
    for (int i = 0; i < 7; i++) { g_pf.hist_ex[i] = g_pf.hist_rows[i] = 0; g_pf.hist_bytes[i] = 0; }
    g_pf.embed = g_pf.egsub = g_pf.borrow_on = g_pf.borrow_off = g_pf.draft = g_pf.head = g_pf.ring = g_pf.fwd = 0;
    g_pf.egdone = -1;
    g_pf.egrows = 0;
    g_pf.nsamp[0] = g_pf.nsamp[1] = g_pf.nsamp[2] = 0;
    g_pf.cpus.clear();
    g_pf.ntok = ntok;
    g_pf.nchunk = 0;
    g_pf.lvl = g_pprof;
    g_pf.h0 = clk::now();
    for (int k = 0; k < nc; k++) {
        DevGuard dg(E->cards[k].dev);
        DCK(cudaEventRecord(g_pf.t0[k], E->cards[k].st));
    }
    g_pf.cpu0 = sched_getcpu();
    g_pf.on = true;
}
static void pf_cpu() {
    if (!g_pf.on) return;
    int c = sched_getcpu();
    if (std::find(g_pf.cpus.begin(), g_pf.cpus.end(), c) == g_pf.cpus.end()) g_pf.cpus.push_back(c);
}
static int pf_bin(int m) { return m <= 1 ? 0 : m <= 4 ? 1 : m <= 8 ? 2 : m <= 16 ? 3 : m <= 32 ? 4 : m <= 64 ? 5 : 6; }
static bool pf_sample(int card) { return g_pf.on && g_pf.lvl >= 2 && (g_pf.nsamp[card]++ % 8) == 0; }
static void pf_end(Engine* E, double wall_ms) {
    if (!g_pf.on) return;
    g_pf.on = false;
    g_pf.cpu1 = sched_getcpu();
    const int nc = (int)E->cards.size();
    for (int k = 0; k < nc; k++) {
        DevGuard dg(E->cards[k].dev);
        DCK(cudaDeviceSynchronize());
    }
    auto& V = g_pf.evs;
    std::vector<double> g(V.size());
    for (size_t i = 0; i < V.size(); i++) {
        float ms = 0.f;
        DCK(cudaEventElapsedTime(&ms, g_pf.t0[V[i].card], V[i].ev));
        g[i] = ms;
    }
    // intervals per (card, stream): charged to the later event
    std::vector<double> iv(V.size(), 0.0), lag(V.size(), 0.0);
    {
        int last[3][2] = {{-1, -1}, {-1, -1}, {-1, -1}};
        for (size_t i = 0; i < V.size(); i++) {
            int& p = last[V[i].card][V[i].strm];
            const double g0 = p < 0 ? 0.0 : g[p];
            iv[i] = std::max(0.0, g[i] - g0);
            // host latency (upper bound): the GPU had reached the previous event before the host recorded this one
            lag[i] = std::max(0.0, std::min(std::min(V[i].h, g[i]) - g0, iv[i]));
            p = (int)i;
        }
    }
    if (getenv("DSV4_PFTL")) {   // [dsv4-pf5-bal] debug: the events of one layer in GPU time order (card 0's clock base each)
        const int tl = atoi(getenv("DSV4_PFTL"));
        for (size_t i = 0; i < V.size(); i++)
            if (V[i].l == tl)
                logf_("pftl L %d card %d strm %d %-22s fam %-22s at %9.3f ms (iv %7.3f) host %9.3f sid %d\n", tl, V[i].card, V[i].strm,
                      PKN[V[i].k], PKN[V[i].fam], g[i], iv[i], V[i].h, V[i].sid);
    }
    // per layer stage table (owner compute stream), remote experts and links per card
    static double st[NL][PK_N], lg[NL][PK_N];
    static double rex[NL][3], lnk[NL][3], lgb[NL][3], lwait[NL][3], rwt[NL][3];
    memset(st, 0, sizeof st); memset(lg, 0, sizeof lg); memset(rex, 0, sizeof rex); memset(lnk, 0, sizeof lnk);
    memset(lgb, 0, sizeof lgb); memset(lwait, 0, sizeof lwait); memset(rwt, 0, sizeof rwt);
    double tot[PK_N] = {}, totlag[PK_N] = {}, pre = 0, tail = 0;
    double ras[3][3] = {};
    double gfirst = 1e30, glast[3] = {}, busy[3] = {}, linkb[3] = {}, linkgb[3] = {};
    for (size_t i = 0; i < V.size(); i++) {
        const PfEv& v = V[i];
        const int l = v.l;
        glast[v.card] = std::max(glast[v.card], g[i]);
        if (v.k == PK_EMBED) { gfirst = std::min(gfirst, g[i] - iv[i]); pre += iv[i]; continue; }
        if (v.k == PK_TAIL) { tail += iv[i]; continue; }
        if (l < 0 || l >= NL) continue;
        const int own = E->L[l].card;
        if (v.strm == 1) {
            if (v.k == PK_COPY) { lnk[l][v.card] += iv[i]; lgb[l][v.card] += v.bytes / 1e9; linkb[v.card] += iv[i]; linkgb[v.card] += v.bytes / 1e9; }
            else lwait[l][v.card] += iv[i];
            continue;
        }
        if (v.card == own) {
            st[l][v.fam] += iv[i];
            lg[l][v.fam] += lag[i];
            tot[v.fam] += iv[i];
            totlag[v.fam] += lag[i];
            if (v.fam != PK_HAND) busy[v.card] += iv[i];
        } else {
            // [dsv4-pf5-bal] the attention split on the other cards: X1 wait (from the stream's previous event), share, wo_b
            if (v.fam == PK_STG || v.fam == PK_ATT || v.fam == PK_WOB) {
                ras[v.card][v.fam == PK_STG ? 0 : v.fam == PK_ATT ? 1 : 2] += iv[i];
                continue;
            }
            if (v.fam == PK_RWAIT) rwt[l][v.card] += iv[i];
            else { rex[l][v.card] += iv[i]; busy[v.card] += iv[i]; }
        }
    }
    logf_("pprof: %d tokens, %d chunk(s), level %d, %zu events; wall %.1f ms; forward %.1f ms (host); host thread on CPU %d -> %d "
          "(%zu CPUs seen)\n", g_pf.ntok, g_pf.nchunk, g_pf.lvl, V.size(), wall_ms, g_pf.fwd, g_pf.cpu0, g_pf.cpu1, g_pf.cpus.size());
    logf_("pprof host: embed+H2D+sync %.1f, engram submit %.1f (%ld unique rows, batch done at %.1f ms after the prefill start), "
          "borrow on %.1f / off %.1f, draft window %.1f, head %.1f, ring sync %.1f ms\n", g_pf.embed, g_pf.egsub, g_pf.egrows,
          g_pf.egdone, g_pf.borrow_on, g_pf.borrow_off, g_pf.draft, g_pf.head, g_pf.ring);
    logf_("pprof layer table (ms; owner stream stages | wall = sum of owner stages | remote experts per card | link busy ms/GB per "
          "card | host: route sync, list build, enqueue, engram wait, stage_src blocked)\n");
    logf_("pprof  L  c   hcpre   stg   qkv  cmpi   wqb  attn   woa   wob  hcffn route shrd  mres mhost  post   eg |  wall |"
          "  rx0   rx1   rx2 | lk0  gb0  lk1  gb1  lk2  gb2 | rsync build   enq  egw  ssrc | lag\n");
    double sw = 0;
    for (int l = 0; l < NL; l++) {
        double w = 0, lgs = 0;
        for (int k = PK_L0; k <= PK_POST; k++) { w += st[l][k]; lgs += lg[l][k]; }
        sw += w;
        const PfLayerHost& h = g_pf.lh[l];
        logf_("pprof %2d %2d %6.1f %5.1f %5.1f %5.1f %5.1f %5.1f %5.1f %5.1f %6.1f %5.1f %4.1f %5.1f %5.1f %5.1f %4.1f | %5.1f |"
              " %5.1f %5.1f %5.1f | %4.1f %4.2f %4.1f %4.2f %4.1f %4.2f | %5.1f %5.1f %5.1f %4.1f %5.1f | %4.1f\n",
              l, E->L[l].card, st[l][PK_HCPRE] + st[l][PK_L0], st[l][PK_STG], st[l][PK_QKV], st[l][PK_CMPIDX], st[l][PK_WQB],
              st[l][PK_ATT], st[l][PK_WOA], st[l][PK_WOB], st[l][PK_HCFFN], st[l][PK_ROUTE], st[l][PK_SHARED], st[l][PK_MRES],
              st[l][PK_MHOST], st[l][PK_POST], st[l][PK_ENGRAM], w, rex[l][0], rex[l][1], rex[l][2], lnk[l][0], lgb[l][0],
              lnk[l][1], lgb[l][1], lnk[l][2], lgb[l][2], h.rsync, h.build, h.enq, h.egw, h.ssrc, lgs);
    }
    logf_("pprof totals (owner stream, ms; lag = host-latency upper bound inside the stage):\n");
    for (int k = PK_HAND; k <= PK_POST; k++)
        if (tot[k] > 0) logf_("pprof   %-22s %8.1f ms  lag %7.1f\n", PKN[k], tot[k], totlag[k]);
    double gend = std::max(glast[0], std::max(glast[1], glast[2]));
    logf_("pprof critical path: embed %.1f + layers %.1f + pipe waits %.1f + tail %.1f = %.1f ms; GPU span %.1f ms; wall %.1f ms\n",
          pre, sw, tot[PK_HAND], tail, pre + sw + tot[PK_HAND] + tail, gend - (gfirst < 1e29 ? gfirst : 0), wall_ms);
    for (int k = 0; k < nc; k++) {
        double own = 0, rem = 0, rwait = 0;
        for (int l = 0; l < NL; l++) {
            if (E->L[l].card == k) for (int j = PK_L0; j <= PK_POST; j++) own += st[l][j];
            rem += rex[l][k];
            rwait += rwt[l][k];
        }
        logf_("pprof card %d: own layers %.1f ms, remote experts %.1f ms (+ %.1f wait), link busy %.1f ms for %.2f GB (%.1f GB/s while "
              "busy)\n", k, own, rem, rwait, linkb[k], linkgb[k], linkb[k] > 0 ? linkgb[k] / linkb[k] * 1e3 : 0.0);
    }
    for (int k = 0; k < nc; k++)
        if (ras[k][1] > 0)
            logf_("pprof card %d attention split as a helper: X1 wait %.1f (incl. idle before), share %.1f, wo_b + o2 waits %.1f ms\n", k,
                  ras[k][0], ras[k][1], ras[k][2]);
    {   // [dsv4-pf6-link] copy streams (invariant, review pf6: during a layer one host thread enqueues on each card's cp, so
        // the event order of (card, stream 1) is its stream order): DMA time per copy (PK_COPY, from the event just before the memcpy) by size, and
        // the gaps before copies (PK_CPWAIT: stream waits + host enqueue latency, lag = host-latency upper bound)
        const double lim[5] = {2e6, 6e6, 10e6, 16e6, 1e30};
        for (int k = 0; k < nc; k++) {
            double n[5] = {}, ms[5] = {}, gb[5] = {}, gap = 0, glag = 0, dmin = 1e30, dmax = 0, oms = 0, ogb = 0;
            int ng = 0, no = 0;
            for (size_t i = 0; i < V.size(); i++) {
                if (V[i].card != k || V[i].strm != 1) continue;
                if (V[i].k == PK_CPWAIT) { gap += iv[i]; glag += lag[i]; ng++; continue; }
                if (V[i].k != PK_COPY) continue;
                if (V[i].aux != 1) { no++; oms += iv[i]; ogb += V[i].bytes / 1e9; continue; }   // peer / borrowed copies (review pf6 M2)
                int b = 0;
                while (V[i].bytes > lim[b]) b++;
                n[b]++; ms[b] += iv[i]; gb[b] += V[i].bytes / 1e9;
                if (V[i].bytes > 6e6 && iv[i] > 0) { const double r = V[i].bytes / 1e6 / iv[i]; dmin = std::min(dmin, r); dmax = std::max(dmax, r); }
            }
            std::string sb;
            const char* bn[5] = {"<2MB", "2-6MB", "6-10MB", "10-16MB", ">16MB"};
            double tms = 0, tgb = 0;
            for (int b = 0; b < 5; b++) {
                tms += ms[b]; tgb += gb[b];
                if (n[b] == 0) continue;
                char t[128];
                snprintf(t, sizeof t, " %s: %.0f x %.2f GB %.1f ms = %.1f GB/s;", bn[b], n[b], gb[b], ms[b], gb[b] / ms[b] * 1e3);
                sb += t;
            }
            logf_("pprof link %d host copies: DMA %.1f ms for %.2f GB = %.1f GB/s (per copy > 6 MB %.1f .. %.1f GB/s);%s peer / borrowed "
                  "copies %d: %.1f ms for %.2f GB; copy stream idle before %d copies %.1f ms (host lag <= %.1f)\n", k, tms, tgb,
                  tms > 0 ? tgb / tms * 1e3 : 0.0, dmin < 1e29 ? dmin : 0.0, dmax, sb.c_str(), no, oms, ogb, ng, gap, glag);
        }
    }
    {   // link busy during each owner's dense phase (layer start .. route) and during its MoE phase
        for (int k = 0; k < nc; k++) {
            double moe = 0, dense = 0, b = 0;
            for (int l = 0; l < NL; l++) {
                for (int j = PK_L0; j <= PK_ROUTE; j++) dense += st[l][j];
                for (int j = PK_SHARED; j <= PK_POST; j++) moe += st[l][j];
                b += lnk[l][k];
            }
            logf_("pprof link %d: busy %.1f ms of the owners' MoE phases %.1f ms (%.0f %%); dense phases %.1f ms\n", k, b, moe,
                  moe > 0 ? 100.0 * b / moe : 0.0, dense);
        }
    }
    {   // [dsv4-pf15-pfx] link busy by owner phase in GPU time (the cards' t0 anchors are recorded together after device
        // syncs): each copy interval split by its overlap with the owners' dense windows (layer start .. route) and MoE
        // windows (shared expert .. hc post); prefetch copies (sid -7, pfx_schedule) counted apart
        std::vector<std::pair<double, double>> dw, mw;
        for (size_t i = 0; i < V.size(); i++) {
            const PfEv& v = V[i];
            if (v.strm != 0 || v.l < 0 || v.l >= NL || v.card != E->L[v.l].card || iv[i] <= 0) continue;
            if (v.fam >= PK_L0 && v.fam <= PK_ROUTE) dw.push_back({g[i] - iv[i], g[i]});
            else if (v.fam >= PK_SHARED && v.fam <= PK_POST) mw.push_back({g[i] - iv[i], g[i]});
        }
        auto ov = [](const std::vector<std::pair<double, double>>& w, double a, double b) {
            double s = 0;
            for (const auto& x : w) s += std::max(0.0, std::min(b, x.second) - std::max(a, x.first));
            return s;
        };
        double dtot = 0;
        for (const auto& x : dw) dtot += x.second - x.first;
        for (int k = 0; k < nc; k++) {
            double bd = 0, bm = 0, bo = 0, pd = 0, pm = 0, pgb = 0;
            for (size_t i = 0; i < V.size(); i++) {
                const PfEv& v = V[i];
                if (v.card != k || v.strm != 1 || v.k != PK_COPY || iv[i] <= 0) continue;
                const double a = g[i] - iv[i], d = ov(dw, a, g[i]), m = ov(mw, a, g[i]);
                bd += d; bm += m; bo += std::max(0.0, iv[i] - d - m);   // (review pf15 L2: owners' windows of different cards may overlap)
                if (v.sid == -7) { pd += d; pm += m; pgb += v.bytes / 1e9; }
            }
            logf_("pprof link %d by phase: dense %.1f of %.1f ms busy, MoE %.1f, other %.1f; prefetch copies %.2f GB (%.1f ms in dense, "
                  "%.1f in MoE)\n", k, bd, dtot, bm, bo, pgb, pd, pm);
        }
        // per layer: first / last prefetch copy of each card relative to the owner's dense window (start .. route end)
        double s0 = 0, s1 = 0, nl = 0;
        for (int l = 0; l < NL; l++) {
            double d0 = 1e30, d1 = -1e30, p0 = 1e30, p1 = -1e30;
            for (size_t i = 0; i < V.size(); i++) {
                const PfEv& v = V[i];
                if (v.l != l || iv[i] <= 0) continue;
                if (v.strm == 0 && v.card == E->L[l].card && v.fam >= PK_L0 && v.fam <= PK_ROUTE) { d0 = std::min(d0, g[i] - iv[i]); d1 = std::max(d1, g[i]); }
                if (v.strm == 1 && v.k == PK_COPY && v.sid == -7) { p0 = std::min(p0, g[i] - iv[i]); p1 = std::max(p1, g[i]); }
            }
            if (d0 > 1e29 || p0 > 1e29) continue;
            s0 += p0 - d0; s1 += p1 - d1; nl++;
        }
        if (nl > 0)
            logf_("pprof prefetch timing: first copy starts %.2f ms after the dense window start, last ends %.2f ms after its end (mean "
                  "of %.0f layers; dense window %.2f ms mean)\n", s0 / nl, s1 / nl, nl, dtot / nl);
    }
    {   // counters
        PfCnt t;
        for (int l = 0; l < NL; l++)
            for (int k = 0; k < 3; k++)
                for (int j = 0; j < PC_N; j++) { t.ex[j] += g_pf.cnt[l][k].ex[j]; t.rows[j] += g_pf.cnt[l][k].rows[j]; }
        std::string s;
        for (int j = 0; j < PC_N; j++) {
            char b[96];
            snprintf(b, sizeof b, "%s%s %d (%d rows)", j ? ", " : "", PCN[j], t.ex[j], t.rows[j]);
            s += b;
        }
        logf_("pprof experts: %s\n", s.c_str());
        const char* bn[7] = {"1", "2-4", "5-8", "9-16", "17-32", "33-64", "65+"};
        std::string hs;
        for (int i = 0; i < 7; i++) {
            char b[96];
            snprintf(b, sizeof b, "%s%s: %lld ex / %lld rows / %.2f GB", i ? ", " : "", bn[i], g_pf.hist_ex[i], g_pf.hist_rows[i],
                     g_pf.hist_bytes[i] / 1e9);
            hs += b;
        }
        logf_("pprof host experts by rows: %s\n", hs.c_str());
        double ss = 0; int ns = 0;
        for (auto& h : g_pf.lh) { ss += h.ssrc; ns += h.nssrc; }
        double rs = 0, bl = 0, en = 0;
        for (auto& h : g_pf.lh) { rs += h.rsync; bl += h.build; en += h.enq; }
        logf_("pprof host sums: route sync %.1f, list build %.1f, MoE enqueue %.1f, stage_src blocked %.1f ms over %d copy-less "
              "experts\n", rs, bl, en, ss, ns);
    }
    if (g_pf.lvl >= 2) {   // sampled experts: sub-stage means by class and row bin
        // class: 0 owner-local, 1 owner-staged, 2 remote-resident, 3 remote-staged; aux = class * 8 + bin
        struct Acc { double t[7] = {}; long n = 0; };
        std::map<int, Acc> A;
        std::map<int, std::array<int, 8>> S;   // sid -> event index per sub-stage (XPRE .. XSC)
        for (size_t i = 0; i < V.size(); i++)
            if (V[i].sid >= 0 && V[i].k >= PK_XPRE && V[i].k <= PK_XSC) {
                auto& a = S[V[i].sid];
                if (V[i].k == PK_XPRE) a.fill(-1);
                a[V[i].k - PK_XPRE] = (int)i;
            }
        for (auto& [sid, a] : S) {
            bool ok = true;
            for (int j = 0; j < 7; j++) ok &= a[j] >= 0;
            if (!ok) continue;
            const int aux = V[a[0]].aux;
            Acc& x = A[aux];
            for (int j = 1; j < 7; j++) x.t[j] += g[a[j]] - g[a[j - 1]];
            x.n++;
        }
        const char* cn[4] = {"owner-local", "owner-staged", "remote-resident", "remote-staged"};
        const char* bn[7] = {"1", "2-4", "5-8", "9-16", "17-32", "33-64", "65+"};
        logf_("pprof sampled experts (1 in 8 per card; mean us): class / rows: n, wait, dequant, gather+gemm13, glu+q, gemm2, "
              "scatter|D2H, sum excl. wait\n");
        for (auto& [key, x] : A) {
            const int cl = key / 8, b = key % 8;
            double sum = 0;
            for (int j = 2; j < 7; j++) sum += x.t[j];
            logf_("pprof   %-15s %-5s n %4ld: wait %7.1f dq %6.1f g13 %6.1f glu %5.1f g2 %6.1f sc %5.1f | %7.1f\n", cn[cl & 3],
                  bn[b & 7], x.n, 1e3 * x.t[1] / x.n, 1e3 * x.t[2] / x.n, 1e3 * x.t[3] / x.n, 1e3 * x.t[4] / x.n,
                  1e3 * x.t[5] / x.n, 1e3 * x.t[6] / x.n, 1e3 * sum / x.n);
        }
    }
}
