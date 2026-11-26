// ink_run: Inkling 2 Mini engine CLI.
//   ink_run tf  SET.json OUTDIR [--gen N] [--dump0]   per item: final hidden (fp32 .bin [T][2304]); chat items: greedy
//                                                     continuation of N tokens (gen.json); --dump0: item 0 layer outputs
//   ink_run bench [--ctx a,b,c] [--n N] [--pf a,b]    decode tok/s after a synthetic prompt of each depth; prefill tok/s
// Options: --max-ctx N, --model DIR
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "../json.h"
#include "ink.h"
#include "ink_tp.h"

using clk = std::chrono::steady_clock;
static double secs(clk::time_point a) { return std::chrono::duration<double>(clk::now() - a).count(); }

static std::vector<int> csv(const char* s) {
    std::vector<int> v;
    std::stringstream ss(s);
    std::string x;
    while (std::getline(ss, x, ',')) v.push_back(atoi(x.c_str()));
    return v;
}

// [ink-srv-samp] tests: INK_TEMP > 0 samples every slot (seed INK_SEED + slot); the first token after a prefill
// [ink-samp-dev] INK_TOPK / INK_TOPP / INK_MINP: the slot's filter with the temperature (device path when supported)
static void samp_filt(ink::Engine& e, int slot) {
    e.set_filter(slot, getenv("INK_TOPK") ? atoi(getenv("INK_TOPK")) : 0, getenv("INK_TOPP") ? (float)atof(getenv("INK_TOPP")) : 1.f,
                 getenv("INK_MINP") ? (float)atof(getenv("INK_MINP")) : 0.f);
}
// INK_MIXED=1 (review M3): slot % 3 == 0 greedy, 1 plain temperature, 2 filtered (one batch holds all three forms)
static int samp_first(ink::Engine& e, int slot) {
    const int mix = getenv("INK_MIXED") ? slot % 3 : 2;
    if (!getenv("INK_TEMP") || mix == 0) {
        e.set_sampling(slot, 0.f, 0);
        e.set_filter(slot, 0, 1.f, 0.f);
        return e.argmax();
    }
    e.set_sampling(slot, atof(getenv("INK_TEMP")), (getenv("INK_SEED") ? atoll(getenv("INK_SEED")) : 1) + slot);
    if (mix == 2) samp_filt(e, slot);
    else e.set_filter(slot, 0, 1.f, 0.f);
    return e.sample_last();
}
int main(int argc, char** argv) {
    try {
        if (argc < 2) {
            fprintf(stderr, "usage: ink_run tf SET OUTDIR [--gen N] [--dump0] | bench [--ctx ..] [--n N] [--pf ..]\n");
            return 2;
        }
        const std::string mode = argv[1];
        if (mode == "tp-plan") {   // [ink-tp-plan] the planner's splits: tp-plan N [w0,w1,..] [depth]
            const int n = argc > 2 ? atoi(argv[2]) : 2;
            std::vector<double> w;
            if (argc > 3)
                for (int v : csv(argv[3])) w.push_back(v / 100.0);
            printf("%s", ink::tp_plan_str(ink::tp_plan(n, w, argc > 4 ? atoi(argv[4]) : 4096)).c_str());
            return 0;
        }
        ink::Opts o;
        int gen = 32, nstep = 128, abn = 0, abv = 1, spk = 4, sfx = 512;
        bool dump0 = false, fake = false, dtf = false;
        int dtfr = 0;   // [ink-tp-tcx] --dtfr R: tf items teacher-forced as row steps of R rows (forced tokens as drafts)
        std::vector<int> ctxs = {0, 4096, 16384}, pfs = {1024, 8192};
        std::vector<std::string> pos;
        for (int i = 2; i < argc; i++) {
            const std::string a = argv[i];
            if (a == "--gen") gen = atoi(argv[++i]);
            else if (a == "--dump0") dump0 = true;
            else if (a == "--max-ctx") o.max_ctx = atoi(argv[++i]);
            else if (a == "--model") o.model_dir = argv[++i];
            else if (a == "--ctx") ctxs = csv(argv[++i]);
            else if (a == "--pf") pfs = csv(argv[++i]);
            else if (a == "--n") nstep = atoi(argv[++i]);
            else if (a == "--sfx") sfx = atoi(argv[++i]);
            else if (a == "--ab") abn = atoi(argv[++i]);
            else if (a == "--abv") abv = atoi(argv[++i]);   // the variant compared with 0
            else if (a == "--fake") fake = true;
            else if (a == "--dtf") dtf = true;
            else if (a == "--dtfr") dtfr = atoi(argv[++i]);
            else if (a == "--k") spk = atoi(argv[++i]);
            else if (a == "--slots") o.slots = atoi(argv[++i]);
            else if (a == "--drafter") o.drafter = true;   // tf items: prefill the prompt, then decode step by step through the text   // decode depth without prefill (speed only)
            else pos.push_back(a);
        }
        auto t0 = clk::now();
        if (mode == "dtest" || mode == "spec" || mode == "vbench" || mode == "bspec") o.drafter = true;
        ink::Engine e(o);
        fprintf(stderr, "[ink] loaded in %.1f s\n", secs(t0));
        if (o.drafter && !e.drafter_on())   // review r4 M2: the memory fit turned the drafter off
            fprintf(stderr, "[ink] NOTE: drafter off after the memory fit (max_ctx %d): spec numbers below are plain decoding\n", e.max_ctx());
        if (e.max_ctx() < o.max_ctx) o.max_ctx = e.max_ctx();
        if (mode == "tf") {
            if (pos.size() < 2) throw std::runtime_error("tf SET OUTDIR");
            std::ifstream in(pos[0]);
            std::string js((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
            Json set = Json::parse(js);
            std::string gj = "{";
            for (size_t k = 0; k < set.arr.size(); k++) {
                const Json& it = set.arr[k];
                std::vector<int> ids;
                for (auto& x : it["ids"].arr) ids.push_back((int)x.i());
                const std::string name = it["name"].str;
                e.reset();
                e.set_dump(dump0 && k == 0);
                std::vector<float> hid(ids.size() * ink::H);
                auto t1 = clk::now();
                const int np = it.has("n_prompt") ? (int)it["n_prompt"].i() : (int)ids.size();
                if (dtfr > 0 && it["kind"].str == "tf" && np > 0 && np < (int)ids.size()) {
                    // rows s .. s + R - 1 carry the forced tokens: every row's context is the forced text, so each row's
                    // hidden is the teacher-forced one; the cache keeps anchor + accepted rows, the next call starts there
                    e.prefill(ids.data(), np, hid.data());
                    std::vector<char> got(ids.size(), 0);
                    const int sl = 0;
                    size_t s = np;
                    long long calls = 0, rows = 0;
                    while (s < ids.size()) {
                        const int r = (int)std::min<size_t>(dtfr, ids.size() - s);
                        std::vector<int> arg(r);
                        int acc = 0;
                        e.step_rows_k(&sl, &r, ids.data() + s, 1, arg.data(), &acc);
                        for (int j = 0; j < r; j++)
                            if (!got[s + j]) {
                                e.row_hidden(j, hid.data() + (s + j) * ink::H);
                                got[s + j] = 1;
                            }
                        calls++;
                        rows += r;
                        s += acc + 1;
                    }
                    fprintf(stderr, "  dtfr %d: %lld calls, %.2f rows / call\n", dtfr, calls, (double)rows / calls);
                } else if (dtf && it["kind"].str == "tf" && np > 0 && np < (int)ids.size()) {
                    e.prefill(ids.data(), np, hid.data());
                    for (size_t s = np; s < ids.size(); s++) {
                        e.step(ids[s]);
                        e.last_hidden(hid.data() + s * ink::H);
                    }
                } else
                    e.prefill(ids.data(), (int)ids.size(), hid.data());
                const double tp = secs(t1);
                FILE* f = fopen((pos[1] + "/" + name + ".hid.bin").c_str(), "wb");
                if (!f) throw std::runtime_error("cannot write " + pos[1]);
                fwrite(hid.data(), 4, hid.size(), f);
                fclose(f);
                if (dump0 && k == 0) {
                    const auto& d = e.dumped();
                    FILE* g = fopen((pos[1] + "/layers0.bin").c_str(), "wb");
                    fwrite(d.data(), 4, d.size(), g);
                    fclose(g);
                    const auto& r = e.routed();
                    g = fopen((pos[1] + "/route0.bin").c_str(), "wb");
                    fwrite(r.data(), 4, r.size(), g);
                    fclose(g);
                    e.set_dump(false);
                }
                fprintf(stderr, "%-13s T=%5zu prefill %.3f s\n", name.c_str(), ids.size(), tp);
                if (getenv("INK_VSTAT")) {
                    long long tot = 0, sub = 0, zero = 0, big = 0;
                    e.v_stats(&tot, &sub, &zero, &big);
                    fprintf(stderr, "  V values %lld: fp16-subnormal range %lld, zero %lld, > 65504 %lld\n", tot, sub, zero, big);
                }
                if (it["kind"].str == "chat" && gen > 0) {
                    std::vector<int> out;
                    int nx = e.argmax();
                    auto t2 = clk::now();
                    for (int s = 0; s < gen; s++) {
                        out.push_back(nx);
                        if (s + 1 < gen) nx = e.step(nx);
                    }
                    fprintf(stderr, "  greedy %d tokens %.3f s\n", gen, secs(t2));
                    if (gj.size() > 1) gj += ",";
                    gj += "\"" + name + "\":[";
                    for (size_t s = 0; s < out.size(); s++) gj += (s ? "," : "") + std::to_string(out[s]);
                    gj += "]";
                }
            }
            gj += "}";
            std::ofstream(pos[1] + "/gen.json") << gj;
        } else if (mode == "dtest") {   // drafter vs the CPU reference: IN.json [{name, ids, anchors}] -> OUT.json drafts
            if (pos.size() < 2) throw std::runtime_error("dtest IN.json OUT.json");
            std::ifstream in(pos[0]);
            std::string js((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
            Json set = Json::parse(js);
            std::string oj = "{";
            for (size_t k = 0; k < set.arr.size(); k++) {
                const Json& it = set.arr[k];
                std::vector<int> ids, an;
                for (auto& x : it["ids"].arr) ids.push_back((int)x.i());
                for (auto& x : it["anchors"].arr) an.push_back((int)x.i());
                std::sort(an.rbegin(), an.rend());   // descending: a block's K / V at [t, t+8) never hides a later anchor's context
                e.reset();
                e.prefill(ids.data(), (int)ids.size());
                auto t1 = clk::now();
                std::string r = "{";
                for (size_t i = 0; i < an.size(); i++) {
                    int d[7];
                    float p[7];
                    e.draft_at(an[i], ids[an[i]], d, p);
                    r += (i ? ",\"" : "\"") + std::to_string(an[i]) + "\":[";
                    for (int j = 0; j < 7; j++) r += (j ? "," : "") + std::to_string(d[j]);
                    for (int j = 0; j < 7; j++) r += "," + std::to_string(p[j]);
                    r += "]";
                }
                r += "}";
                fprintf(stderr, "%-12s T=%5zu anchors %zu  %.3f s (%.2f ms/block)\n", it["name"].str.c_str(), ids.size(), an.size(), secs(t1),
                        1e3 * secs(t1) / std::max<size_t>(an.size(), 1));
                oj += (k ? ",\"" : "\"") + it["name"].str + "\":" + r;
            }
            oj += "}";
            std::ofstream(pos[1]) << oj;
        } else if (mode == "bspec") {   // [ink-b-draft] slots with the batched drafter vs each alone (plain greedy): identical? tok/s
            if (pos.size() < 1) throw std::runtime_error("bspec SET.json [--slots N] [--gen G]");
            std::ifstream in(pos[0]);
            std::string js((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
            Json set = Json::parse(js);
            const int nb = getenv("INK_BN") ? atoi(getenv("INK_BN")) : e.slots();
            std::vector<std::vector<int>> pr(nb), a(nb);
            std::vector<std::string> nm(nb);
            const int rot = getenv("INK_BROT") ? atoi(getenv("INK_BROT")) : 0;   // slot i gets item (i + rot) % items
            for (int i = 0; i < nb; i++) {
                const Json& it = set.arr[(i + rot) % set.arr.size()];
                nm[i] = it["name"].str;
                for (auto& x : it["ids"].arr) pr[i].push_back((int)x.i());
                if (it.has("n_prompt")) pr[i].resize((int)it["n_prompt"].i());
            }
            if (!getenv("INK_B_NOALONE"))
                for (int i = 0; i < nb; i++) {
                    e.select(i);
                    e.reset();
                    e.prefill(pr[i].data(), (int)pr[i].size());
                    int nx = samp_first(e, i);
                    for (int s = 0; s < gen; s++) {
                        a[i].push_back(nx);
                        if (s + 1 < gen) nx = e.step(nx);
                    }
                }
            if (getenv("INK_BREV")) {   // [ink-srv-pool] the batched runs on reversed pool intervals
                const int S = e.slots();
                std::vector<int> bs(S), lm(S);
                for (int i = 0; i < S; i++) bs[i] = e.slot_base(i), lm[i] = e.slot_lim(i);
                for (int i = 0; i < S; i++) e.bind(i, bs[S - 1 - i], 0, lm[S - 1 - i]);
                fprintf(stderr, "[bspec] slot intervals reversed\n");
            }
            std::vector<int> pols = {0, getenv("INK_BPOL") ? atoi(getenv("INK_BPOL")) : 2};
            if (getenv("INK_BPOLS")) pols = csv(getenv("INK_BPOLS"));
            for (int pol : pols) {
                e.set_bpol(pol);
                std::vector<std::vector<int>> b(nb);
                std::vector<int> sl(nb), tk(nb), out((size_t)nb * ink::VR), no(nb);
                for (int i = 0; i < nb; i++) {
                    e.select(i);
                    e.reset();
                    e.prefill(pr[i].data(), (int)pr[i].size());
                    tk[i] = samp_first(e, i);
                    b[i].push_back(tk[i]);
                }
                e.select(0);
                e.spec_rows_stats(true);
                long long emitted = 0;
                double t = 0;
                int steps = 0;
                while (true) {   // all slots step until every one has gen tokens (aggregate over the steps where all run)
                    std::vector<int> act, at;
                    for (int i = 0; i < nb; i++)
                        if ((int)b[i].size() < gen) {
                            act.push_back(i);
                            at.push_back(tk[i]);
                        }
                    if ((int)act.size() < nb) break;
                    auto t1 = clk::now();
                    e.spec_rows(act.data(), at.data(), (int)act.size(), out.data(), no.data());
                    const double dt = secs(t1);
                    if (getenv("INK_BSDBG")) fprintf(stderr, "[bsdbg] call %.0f us\n", dt * 1e6);
                    if (++steps > 4) {
                        t += dt;
                        for (int z = 0; z < (int)act.size(); z++) emitted += no[z];
                    }
                    for (int z = 0; z < (int)act.size(); z++) {
                        const int i = act[z];
                        for (int q = 0; q < no[z]; q++) b[i].push_back(out[(size_t)z * ink::VR + q]);
                        tk[i] = b[i].back();
                    }
                }
                int bad = 0, cmp = 0;
                for (int i = 0; i < nb; i++) {
                    if (a[i].empty()) continue;
                    const size_t L = std::min(a[i].size(), b[i].size());
                    cmp++;
                    bad += !std::equal(a[i].begin(), a[i].begin() + L, b[i].begin());
                    if (getenv("INK_BDIV")) {   // [ink-tp-tcx] tokens until the first batch != alone token
                        size_t k = 0;
                        while (k < L && a[i][k] == b[i][k]) k++;
                        printf("div slot %d item %s: first diff %s%zu of %zu\n", i, nm[i].c_str(), k == L ? ">= " : "", k, L);
                    }
                }
                printf("bspec c %d policy %d: %.1f tok/s aggregate (%d steps; %s); prefix == alone %d / %d\n", nb, pol,
                       t > 0 ? emitted / t : 0.0, steps, e.spec_rows_stats(true).c_str(), cmp - bad, cmp);
                if (bad) printf("BSPEC FAIL\n");
            }
        } else if (mode == "batch") {   // [ink-b-rows] slots: each item alone (select + step) vs all together (step_rows): identical?
            if (pos.size() < 1) throw std::runtime_error("batch SET.json [--slots N] [--gen G]");
            std::ifstream in(pos[0]);
            std::string js((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
            Json set = Json::parse(js);
            const int S = e.slots();
            const int nb = getenv("INK_BN") ? atoi(getenv("INK_BN")) : S;   // rows per step_rows (slots used)
            std::vector<std::vector<int>> pr(nb);
            for (int i = 0; i < nb; i++) {
                const Json& it = set.arr[i % set.arr.size()];
                for (auto& x : it["ids"].arr) pr[i].push_back((int)x.i());
                const int np = it.has("n_prompt") && !getenv("INK_BFULL") ? (int)it["n_prompt"].i() : (int)pr[i].size();
                pr[i].resize(np);
                if (getenv("INK_BCUT")) pr[i].resize(std::min<size_t>(pr[i].size(), atoi(getenv("INK_BCUT")) + 97 * i));
            }
            std::vector<std::vector<int>> a(nb), b(nb);
            double ta = 0;
            if (!getenv("INK_B_NOALONE"))
                for (int i = 0; i < nb; i++) {
                    e.select(i);
                    e.reset();
                    e.prefill(pr[i].data(), (int)pr[i].size());
                    int nx = samp_first(e, i);
                    auto t1 = clk::now();
                    for (int s = 0; s < gen; s++) {
                        a[i].push_back(nx);
                        if (s + 1 < gen) nx = e.step(nx);
                    }
                    ta += secs(t1);
                }
            std::vector<int> sl(nb), tk(nb), out(nb);
            if (getenv("INK_BREV")) {   // [ink-srv-pool] the batch run on reversed pool intervals (alone: the defaults)
                std::vector<int> bs(S), lm(S);
                for (int i = 0; i < S; i++) bs[i] = e.slot_base(i), lm[i] = e.slot_lim(i);
                for (int i = 0; i < S; i++) e.bind(i, bs[S - 1 - i], 0, lm[S - 1 - i]);
                fprintf(stderr, "[batch] slot intervals reversed\n");
            }
            for (int i = 0; i < nb; i++) {
                e.select(i);
                e.reset();
                e.prefill(pr[i].data(), (int)pr[i].size());
                tk[i] = samp_first(e, i);
                sl[i] = i;
                b[i].push_back(tk[i]);
            }
            e.select(0);
            std::vector<double> ms;
            const int bk = getenv("INK_BK") ? atoi(getenv("INK_BK")) : 1;   // rows per slot (anchor + bk - 1 drafts)
            const int junk = getenv("INK_BJUNK") ? atoi(getenv("INK_BJUNK")) : 0;   // 1 / junk of the drafts wrong
            long long rows_tot = 0, emit_tot = 0;
            if (bk == 1)
                for (int s = 1; s < gen; s++) {
                    auto t1 = clk::now();
                    e.step_rows(sl.data(), tk.data(), nb, out.data());
                    ms.push_back(secs(t1) * 1e3);
                    for (int i = 0; i < nb; i++) b[i].push_back(tk[i] = out[i]);
                }
            else {   // drafts from the alone run's text (all right, or wrong with probability 1 / junk): emitted must equal it
                uint32_t rg = 777u;
                std::vector<int> nr(nb), toks, arg((size_t)nb * bk), acc(nb);
                for (int it = 0; it < 4 * gen; it++) {
                    bool all = true;
                    for (int i = 0; i < nb; i++) all &= (int)b[i].size() >= gen;
                    if (all) break;
                    toks.clear();
                    for (int i = 0; i < nb; i++) {
                        nr[i] = bk;
                        const size_t L = b[i].size();
                        toks.push_back(b[i].back());
                        for (int d = 1; d < bk; d++) {
                            int t = L - 1 + d < a[i].size() ? a[i][L - 1 + d] : 17;
                            rg = rg * 1664525u + 1013904223u;
                            if (junk && (rg >> 8) % junk == 0) t = 1000 + (rg >> 20) % 5000;
                            toks.push_back(t);
                        }
                    }
                    auto t1 = clk::now();
                    e.step_rows_k(sl.data(), nr.data(), toks.data(), nb, arg.data(), acc.data());
                    ms.push_back(secs(t1) * 1e3);
                    for (int i = 0; i < nb; i++) {
                        for (int q = 0; q <= acc[i]; q++) b[i].push_back(arg[(size_t)i * bk + q]);
                        emit_tot += acc[i] + 1;
                    }
                    rows_tot += (long long)nb * bk;
                }
                for (int i = 0; i < nb; i++) b[i].resize(std::min<size_t>(b[i].size(), gen));
                printf("rows / step %d, tokens / step %.2f\n", nb * bk, (double)emit_tot / std::max<size_t>(1, ms.size()));
            }
            int bad = 0;
            for (int i = 0; i < nb; i++) {
                const bool same = a[i].empty() || a[i] == b[i];
                bad += !same;
                int fd = -1;
                for (size_t j = 0; j < a[i].size() && fd < 0; j++)
                    if (a[i][j] != b[i][j]) fd = (int)j;
                printf("slot %d prompt %zu: %s%s (first diff %d)\n", i, pr[i].size(), a[i].empty() ? "batch only" : same ? "== alone" : "DIFFERS",
                       "", fd);
            }
            std::vector<double> m2(ms.begin() + std::min<size_t>(ms.size(), 8), ms.end());
            std::sort(m2.begin(), m2.end());
            double mean = 0;
            for (double v : m2) mean += v;
            mean /= std::max<size_t>(1, m2.size());
            printf("batch rows %d: step_rows mean %.3f ms, median %.3f ms (%zu steps after 8 warm-up) = %.1f tok/s aggregate; alone %.1f tok/s per slot (mean)\n",
                   nb, mean, m2.empty() ? 0.0 : m2[m2.size() / 2], m2.size(), nb * 1e3 / mean, ta > 0 ? nb * (gen - 1) / ta : 0.0);
            printf("BATCH %s (%d / %d slots identical)\n", bad ? "FAIL" : "PASS", nb - bad, nb);
        } else if (mode == "spec") {   // greedy plain vs speculative (k rows) on the chat items: identical tokens?
            if (pos.size() < 2) throw std::runtime_error("spec SET.json OUTDIR");
            std::ifstream in(pos[0]);
            std::string js((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
            Json set = Json::parse(js);
            std::string gj = "{";
            int bad = 0, nit = 0;
            const int fpre = getenv("INK_FAKEPRE") ? atoi(getenv("INK_FAKEPRE")) : 0;
            for (size_t k = 0; k < set.arr.size(); k++) {
                const Json& it = set.arr[k];
                std::vector<int> ids;
                for (auto& x : it["ids"].arr) ids.push_back((int)x.i());
                const int np = it.has("n_prompt") ? (int)it["n_prompt"].i() : (int)ids.size();
                if (it["kind"].str != "chat" && !getenv("INK_SPEC_ALL")) continue;
                std::vector<int> pr(ids.begin(), ids.begin() + np);
                std::vector<int> a, b;
                double ta = 0, tb = 0;
                int steps = 0;
                if (!getenv("INK_SPEC_NOPLAIN")) {
                    if (fpre) e.fake_depth(fpre);   // INK_FAKEPRE: a synthetic (zero K / V) prefix before the prompt
                    else e.reset();
                    if (getenv("INK_TEMP")) e.set_sampling(0, atof(getenv("INK_TEMP")), (getenv("INK_SEED") ? atoll(getenv("INK_SEED")) : 1) + k);
                    if (getenv("INK_TEMP")) samp_filt(e, 0);
                    e.prefill(pr.data(), np);
                    int nx = getenv("INK_TEMP") ? e.sample_last() : e.argmax();
                    auto t1 = clk::now();
                    for (int s = 0; s < gen; s++) {
                        a.push_back(nx);
                        if (nx == 200006 && getenv("INK_EOS")) break;
                        if (s + 1 < gen) nx = e.step(nx);
                    }
                    ta = secs(t1);
                }
                if (fpre) e.fake_depth(fpre);
                else e.reset();
                if (getenv("INK_TEMP")) e.set_sampling(0, atof(getenv("INK_TEMP")), (getenv("INK_SEED") ? atoll(getenv("INK_SEED")) : 1) + k);
                if (getenv("INK_TEMP")) samp_filt(e, 0);
                e.prefill(pr.data(), np);
                int nx = getenv("INK_TEMP") ? e.sample_last() : e.argmax();
                auto t2 = clk::now();
                b.push_back(nx);
                const bool eos = getenv("INK_EOS") != nullptr;
                const int lim = (eos && !a.empty()) ? (int)a.size() : gen;
                bool hit = eos && b.back() == 200006;
                while ((int)b.size() < lim && !hit) {
                    int out[8];
                    const int n = e.spec_step(nx, spk, out);
                    steps++;
                    for (int j = 0; j < n && !hit; j++) {
                        b.push_back(out[j]);
                        hit = eos && out[j] == 200006;   // stop at the first end token inside a step's output
                    }
                    nx = out[n - 1];
                }
                tb = secs(t2);
                if ((int)b.size() > lim) b.resize(lim);
                if (!a.empty() && a.size() < b.size()) b.resize(a.size());
                const int ng = (int)b.size();
                const bool same = a.empty() || a == b;
                nit++;
                if (!same) {
                    bad++;
                    int d = 0;
                    while (a[d] == b[d]) d++;
                    fprintf(stderr, "%-12s MISMATCH at token %d\n", it["name"].str.c_str(), d);
                }
                fprintf(stderr, "%-12s   %s\n", it["name"].str.c_str(), e.spec_stats(true).c_str());
                fprintf(stderr, "%-12s %s  plain %.1f tok/s  spec k=%d %.1f tok/s  %.2f tok/step  (%d tokens)\n", it["name"].str.c_str(),
                        same ? "same" : "DIFF", ta > 0 ? a.size() / ta : 0., spk, (ng - 1) / tb, (double)(ng - 1) / std::max(steps, 1), ng);
                if (gj.size() > 1) gj += ",";
                gj += "\"" + it["name"].str + "\":[";
                for (size_t s = 0; s < b.size(); s++) gj += (s ? "," : "") + std::to_string(b[s]);
                gj += "]";
            }
            gj += "}";
            std::ofstream(pos[1] + "/spec.json") << gj;
            printf("spec: %d / %d items identical to plain greedy\n", nit - bad, nit);
            if (getenv("INK_TEMP")) printf("filter stats: %s, host fallbacks %lld\n", e.filter_stats().c_str(), e.filter_fallbacks());
        } else if (mode == "vbench") {   // verify step ms per k and drafter block ms at each depth (--ctx), plain step for reference
            for (int c : ctxs) {
                int nx = 1000;
                if (fake) e.fake_depth(std::max(c, 16));   // --fake: depth without a prefill (synthetic zero K / V)
                else {
                    std::vector<int> ids(std::max(c, 16));
                    for (int i = 0; i < (int)ids.size(); i++) ids[i] = 1000 + (i * 7919) % 150000;
                    e.reset();
                    e.prefill(ids.data(), (int)ids.size());
                    nx = e.argmax();
                }
                for (int w = 0; w < 4; w++) nx = e.step(nx);
                e.sync();
                auto t1 = clk::now();
                for (int s = 0; s < nstep; s++) nx = e.step(nx);
                const double pl = 1e3 * secs(t1) / nstep;
                printf("depth %6d plain %.3f ms", c, pl);
                double dm = 0;
                std::vector<int> kl = {2, 3, 4, 6, 8};
                if (getenv("INK_VB_K")) kl = csv(getenv("INK_VB_K"));
                for (int k : kl) {
                    double vm;
                    e.bench_spec(k, nstep / 2, &vm, &dm);
                    printf("  k%d %.3f", k, vm);
                }
                printf("  drafter %.3f ms\n", dm);
                fflush(stdout);
            }
        } else if (mode == "samptest") {   // [ink-srv-samp] engine sampled tokens vs the host Gumbel-max over the full head's
            // logits (same noise function): ink_run samptest SET.json --gen G, env INK_TEMP (default 1.0)
            std::ifstream in(pos[0]);
            std::string js((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
            Json set = Json::parse(js);
            const float T = getenv("INK_TEMP") ? atof(getenv("INK_TEMP")) : 1.f;
            auto gum = [](uint32_t seed, int step, int row) {
                uint32_t h = seed ^ (uint32_t)step * 0x9E3779B9u ^ (uint32_t)row * 0x85EBCA6Bu;
                h ^= h >> 16; h *= 0x7feb352du; h ^= h >> 15; h *= 0x846ca68bu; h ^= h >> 16;
                const double u = ((h >> 9) + 0.5) * (1.0 / 8388608.0);
                return -log(-log(u));
            };
            std::vector<float> lg(ink::V);
            long same = 0, tot = 0, greedy_eq = 0;
            for (size_t k = 0; k < set.arr.size(); k++) {
                const Json& it = set.arr[k];
                std::vector<int> ids;
                for (auto& x : it["ids"].arr) ids.push_back((int)x.i());
                if (it.has("n_prompt")) ids.resize((int)it["n_prompt"].i());
                const unsigned long long seed = 7 + k;
                const uint32_t s32 = (uint32_t)(seed ^ (seed >> 32));
                e.reset();
                e.set_sampling(0, T, seed);
                e.prefill(ids.data(), (int)ids.size());
                int nx = e.sample_last();
                for (int s = 0; s < gen; s++) {
                    e.row_logits(0, lg.data());
                    const int p = e.pos() - 1;
                    int bi = 0, ga = 0;
                    double bv = -1e300;
                    for (int v = 0; v < ink::V; v++) {
                        const double x = (double)lg[v] / T + gum(s32, p, v);
                        if (x > bv) bv = x, bi = v;
                        if (lg[v] > lg[ga]) ga = v;
                    }
                    same += bi == nx;
                    greedy_eq += ga == nx;
                    tot++;
                    nx = e.step(nx);
                }
            }
            printf("samptest T %.2f: engine == host Gumbel-max %ld / %ld, == greedy %ld / %ld\n", T, same, tot, greedy_eq, tot);
        } else if (mode == "filttest") {   // [ink-samp-dev] engine filtered tokens vs a host reference over the full head
            // (the C API's rule: top_k, min_p, top_p; then Gumbel-max with the engine's noise over the kept set):
            // ink_run filttest SET.json --gen G; env INK_TEMP (0.6), INK_TOPK (20), INK_TOPP (0.95), INK_MINP (0)
            // INK_FDIST=N: also N seeds of the first token of item 0 vs the filtered softmax (chi-square, KL)
            std::ifstream in(pos[0]);
            std::string js((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
            Json set = Json::parse(js);
            const float T = getenv("INK_TEMP") ? atof(getenv("INK_TEMP")) : 0.6f;
            const int K = getenv("INK_TOPK") ? atoi(getenv("INK_TOPK")) : 20;
            const float P = getenv("INK_TOPP") ? atof(getenv("INK_TOPP")) : 0.95f, M = getenv("INK_MINP") ? atof(getenv("INK_MINP")) : 0.f;
            auto gum = [](uint32_t seed, int step, int row) {
                uint32_t h = seed ^ (uint32_t)step * 0x9E3779B9u ^ (uint32_t)row * 0x85EBCA6Bu;
                h ^= h >> 16; h *= 0x7feb352du; h ^= h >> 15; h *= 0x846ca68bu; h ^= h >> 16;
                const float u = ((h >> 9) + 0.5f) * (1.f / 8388608.f);
                return -logf(-logf(u));
            };
            std::vector<float> lg(ink::V);
            // kept set of the host rule: (ids, probabilities) in order
            auto kept = [&](std::vector<int>& id, std::vector<double>& pr) {
                std::vector<int> ix(ink::V);
                for (int v = 0; v < ink::V; v++) ix[v] = v;
                auto hi = [&](int a, int b) { return lg[a] > lg[b] || (lg[a] == lg[b] && a < b); };
                const int n = K > 0 ? K : ink::V;
                std::partial_sort(ix.begin(), ix.begin() + std::min(n, 4096), ix.end(), hi);
                ix.resize(std::min(n, 4096));
                std::vector<double> q;
                for (int v : ix) q.push_back(exp((double)(lg[v] - lg[ix[0]]) / T));
                size_t m = ix.size();
                if (M > 0) { m = 1; while (m < ix.size() && q[m] >= M) m++; }
                if (P < 1) {
                    double s = 0, a = 0;
                    if (K <= 0 && M <= 0)   // top_p alone: the mass of the whole vocabulary (the list holds the top 4096)
                        for (int v = 0; v < ink::V; v++) s += exp((double)(lg[v] - lg[ix[0]]) / T);
                    else
                        for (size_t i = 0; i < m; i++) s += q[i];
                    size_t u = 0;
                    while (u < m && a < P * s) a += q[u++];
                    m = std::max<size_t>(u, 1);
                }
                id.assign(ix.begin(), ix.begin() + m);
                double s = 0;
                for (size_t i = 0; i < m; i++) s += q[i];
                pr.clear();
                for (size_t i = 0; i < m; i++) pr.push_back(q[i] / s);
            };
            long same = 0, tot = 0, ksz = 0, gfirst = 0;
            for (size_t k = 0; k < set.arr.size(); k++) {
                const Json& it = set.arr[k];
                std::vector<int> ids;
                for (auto& x : it["ids"].arr) ids.push_back((int)x.i());
                if (it.has("n_prompt")) ids.resize((int)it["n_prompt"].i());
                const unsigned long long seed = 11 + k;
                const uint32_t s32 = (uint32_t)(seed ^ (seed >> 32));
                e.reset();
                e.set_sampling(0, T, seed);
                e.set_filter(0, K, P, M);
                e.prefill(ids.data(), (int)ids.size());
                int nx = e.sample_last();
                for (int s = 0; s < gen; s++) {
                    e.row_logits(0, lg.data());
                    const int p = e.pos() - 1;
                    std::vector<int> id;
                    std::vector<double> pr;
                    kept(id, pr);
                    int bi = id[0];
                    float bv = -INFINITY;
                    for (int v : id) {
                        const float x = lg[v] * (1.f / T) + gum(s32, p, v);
                        if (x > bv || (x == bv && v < bi)) bv = x, bi = v;
                    }
                    same += bi == nx;
                    gfirst += nx == id[0];
                    ksz += (long)id.size();
                    tot++;
                    if (bi != nx && same + 5 > tot) printf("  item %zu step %d: engine %d host %d (kept %zu)\n", k, s, nx, bi, id.size());
                    nx = e.step(nx);
                }
            }
            printf("filttest T %.2f top_k %d top_p %.2f min_p %.2f: engine == host %ld / %ld, kept set mean %.2f, top-1 drawn %ld / %ld, host fallbacks %lld\n", T, K, P, M,
                   same, tot, tot ? (double)ksz / tot : 0.0, gfirst, tot, e.filter_fallbacks());
            printf("filter stats: %s\n", e.filter_stats().c_str());
            if (const int ns = getenv("INK_FDIST") ? atoi(getenv("INK_FDIST")) : 0) {
                // the first (item, prefix) whose kept set has >= INK_FMIN (6) tokens and a top probability < 0.5
                std::vector<int> ids;
                const int fmin = getenv("INK_FMIN") ? atoi(getenv("INK_FMIN")) : 6;
                {
                    std::vector<int> idk;
                    std::vector<double> prk;
                    bool found = false;
                    for (size_t k = 0; k < set.arr.size() && !found; k++) {
                        std::vector<int> all;
                        for (auto& x : set.arr[k]["ids"].arr) all.push_back((int)x.i());
                        const int np0 = set.arr[k].has("n_prompt") ? (int)set.arr[k]["n_prompt"].i() : (int)all.size();
                        for (int cut = np0; cut <= (int)all.size() && cut < 3000 && !found; cut += 5) {
                            e.reset();
                            e.prefill(all.data(), cut);
                            e.row_logits(0, lg.data());
                            kept(idk, prk);
                            if ((int)idk.size() >= fmin && prk[0] < 0.5) {
                                ids.assign(all.begin(), all.begin() + cut);
                                found = true;
                                printf("filtdist position: item %zu prefix %d\n", k, cut);
                            }
                        }
                    }
                    if (!found) throw std::runtime_error("filtdist: no position with a wide kept set");
                }
                const int np = (int)ids.size();
                e.reset();
                e.prefill(ids.data(), np - 1);
                const int sid = e.save_prefix();
                std::vector<int> id;
                std::vector<double> pr;
                std::vector<long> cnt;
                long outside = 0;
                for (int r = 0; r < ns; r++) {
                    if (!e.restore_prefix(sid)) throw std::runtime_error("prefix evicted");
                    e.set_sampling(0, T, 1000003ull * r + 17 + (getenv("INK_FSEED") ? atoll(getenv("INK_FSEED")) : 0));   // INK_FSEED: another seed block
                    e.set_filter(0, K, P, M);
                    e.prefill(ids.data() + np - 1, 1);
                    if (r == 0) {
                        e.row_logits(0, lg.data());
                        kept(id, pr);
                        cnt.assign(id.size(), 0);
                    }
                    const int t = e.sample_last();
                    const auto f = std::find(id.begin(), id.end(), t);
                    if (f == id.end()) outside++;
                    else cnt[f - id.begin()]++;
                }
                double chi = 0, kl = 0;
                for (size_t i = 0; i < id.size(); i++) {
                    const double ex = pr[i] * ns;
                    chi += (cnt[i] - ex) * (cnt[i] - ex) / ex;
                    if (cnt[i]) kl += (double)cnt[i] / ns * log((double)cnt[i] / ns / pr[i]);
                }
                printf("filtdist %d seeds, kept %zu tokens: outside the kept set %ld, chi-square %.2f (df %zu), KL(emp || target) %.5f\n", ns, id.size(), outside, chi,
                       id.size() - 1, kl);
                for (size_t i = 0; i < std::min<size_t>(id.size(), 8); i++) printf("  token %d p %.4f observed %.4f\n", id[i], pr[i], (double)cnt[i] / ns);
            }
        } else if (mode == "srvtest") {   // [ink-srv-pool] ring_save / ring_load, rows_save / rows_load, copy_rows vs the
            // original slot: ink_run srvtest SET.json --slots 3 [--drafter]; per item: slot 0 prefills n - 1 rows, saves,
            // prefills the last token and decodes gen tokens (spec_step with the drafter, else step); slot 1 restores from
            // the host blobs at another base, slot 2 from slot 0's rows by copy_rows; all three must emit the same tokens
            if (pos.size() < 1 || e.slots() < 3) throw std::runtime_error("srvtest SET.json --slots 3");
            std::ifstream in(pos[0]);
            std::string js((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
            Json set = Json::parse(js);
            const int lim = e.slot_lim(0);
            int bad = 0, items = 0;
            for (auto& it : set.arr) {
                std::vector<int> ids;
                for (auto& x : it["ids"].arr) ids.push_back((int)x.i());
                if (it.has("n_prompt") && !getenv("INK_BFULL")) ids.resize((int)it["n_prompt"].i());
                if ((int)ids.size() + gen + 16 > lim || ids.size() < 2) continue;
                const int n = (int)ids.size();
                std::vector<unsigned char> ring, rows;
                std::vector<std::vector<int>> outs(3);
                std::vector<std::string> stats(3);
                auto run = [&](int slot) {
                    if (getenv("INK_SRVDBG")) fprintf(stderr, "[srvtest] slot %d pos %d lim %d base %d\n", slot, e.pos(), e.slot_lim(slot), e.slot_base(slot));
                    e.prefill(&ids[n - 1], 1);
                    int nx = e.argmax();
                    std::vector<int>& o = outs[slot];
                    e.spec_stats(true);
                    while ((int)o.size() < gen && nx != 200006) {   // up to EOS (spec output ends there)
                        o.push_back(nx);
                        if (e.drafter_on()) {
                            int buf[64];
                            const int k = e.spec_step(nx, -8, buf);
                            if (k < 1) break;
                            for (int j = 0; j + 1 < k; j++) o.push_back(buf[j]);
                            nx = buf[k - 1];
                        } else
                            nx = e.step(nx);
                    }
                    if ((int)o.size() > gen) o.resize(gen);
                    stats[slot] = e.spec_stats(true);
                };
                e.select(0);
                e.bind(0, 0, 0, lim);
                e.reset();
                e.prefill(ids.data(), n - 1);
                e.ring_save(ring);
                e.rows_save(0, n - 1, rows);
                float pa[8] = {}, pb[8] = {}, pc[8] = {};
                if (e.drafter_on()) e.draft_probe(ids[n - 1], pa);
                run(0);
                e.select(1);
                const int b1 = getenv("INK_SRVSELF") ? 0 : e.pool() - lim;   // another place in the pool (SELF: slot 0's)
                e.bind(1, b1, 0, lim);
                e.reset();
                e.rows_load(rows, b1, n - 1);
                e.ring_load(ring, n - 1);
                if (e.drafter_on()) e.draft_probe(ids[n - 1], pb);
                run(1);
                e.select(2);
                e.bind(2, lim, 0, lim);
                e.reset();
                e.copy_rows(0, lim, n - 1);   // slot 0's rows (its reply rows after them are not copied)
                e.ring_load(ring, n - 1);
                if (e.drafter_on()) {
                    e.draft_probe(ids[n - 1], pc);
                    printf("probe: %s / %s (%.4f %.4f %.4f | %.4f %.4f %.4f)\n", memcmp(pa, pb, 32) ? "DIFF" : "same", memcmp(pa, pc, 32) ? "DIFF" : "same",
                           pa[1], pa[2], pa[3], pb[1], pb[2], pb[3]);
                }
                run(2);
                const bool ok = outs[1] == outs[0] && outs[2] == outs[0];
                bad += !ok;
                items++;
                printf("%-14s n %d ring %.1f MB rows %.1f MB: %s | %s | %s | %s\n", it["name"].str.c_str(), n, ring.size() / 1e6, rows.size() / 1e6,
                       ok ? "same" : "DIFF", stats[0].c_str(), stats[1].c_str(), stats[2].c_str());
            }
            printf("srvtest: %d / %d items identical\n", items - bad, items);
        } else if (mode == "prefix") {   // [ink-pf-ckpt] restored prefix + suffix vs cold: ink_run prefix SET.json --pf A --sfx B
            if (pos.empty()) throw std::runtime_error("prefix SET.json");
            std::ifstream in(pos[0]);
            std::string js((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
            Json set = Json::parse(js);
            std::vector<int> all;
            const int A = pfs[0], B = sfx;
            while ((int)all.size() < std::max(A + B, 1007)) {   // review L3
                const size_t n0 = all.size();
                for (auto& it : set.arr)
                    for (auto& x : it["ids"].arr) all.push_back((int)x.i());
                if (all.size() == n0) throw std::runtime_error("prefix: the set has no token ids");
            }
            std::vector<float> h1((size_t)(A + B) * ink::H), h2((size_t)B * ink::H);
            auto greedy = [&](int n) {
                std::vector<int> g;
                int nx = e.argmax();
                for (int s = 0; s < n; s++) {
                    g.push_back(nx);
                    if (s + 1 < n) nx = e.step(nx);
                }
                return g;
            };
            e.reset();
            auto t1 = clk::now();
            e.prefill(all.data(), A + B, h1.data());
            const double tc = secs(t1);
            const auto g1 = greedy(32);
            e.reset();
            e.prefill(all.data(), A);
            t1 = clk::now();
            const int id = e.save_prefix();
            const double ts = secs(t1);
            e.reset();
            e.prefill(all.data() + 7, 1000);   // something else in between
            t1 = clk::now();
            if (!e.restore_prefix(id)) throw std::runtime_error("restore failed");
            const double tr = secs(t1);
            t1 = clk::now();
            e.prefill(all.data() + A, B, h2.data());
            const double tsx = secs(t1);
            const auto g2 = greedy(32);
            double md = 0, ss = 0, sr = 0;
            size_t same = 0;
            for (size_t i = 0; i < h2.size(); i++) {
                const double d = (double)h2[i] - h1[(size_t)A * ink::H + i];
                md = std::max(md, std::fabs(d));
                ss += d * d;
                sr += (double)h1[(size_t)A * ink::H + i] * h1[(size_t)A * ink::H + i];
                same += h2[i] == h1[(size_t)A * ink::H + i];
            }
            printf("prefix %d + suffix %d: cold %.3f s | save %.3f s (%.2f GB archive) | restore %.3f s + suffix %.3f s = TTFT %.3f s\n", A, B, tc, ts,
                   e.archive_bytes() / 1e9, tr, tsx, tr + tsx);
            printf("suffix hidden vs cold: %zu of %zu identical, max |d| %.3e, rel L2 %.3e; greedy 32 after: %s\n", same, h2.size(), md,
                   std::sqrt(ss / sr), g1 == g2 ? "identical" : "DIFF");
            return 0;
        } else if (mode == "bench") {
            for (int pf : pfs) {
                std::vector<int> ids(pf);
                for (int i = 0; i < pf; i++) ids[i] = 1000 + (i * 7919) % 150000;
                e.reset();
                e.sync();
                auto t1 = clk::now();
                e.prefill(ids.data(), pf);
                const double s = secs(t1);
                printf("prefill %6d tokens: %.3f s = %.1f tok/s\n", pf, s, pf / s);
            }
            if (abn) {   // same-process A/B: variants 0 / 1 alternate abn times at the first depth
                const int c = ctxs[0];
                int nx = 1000;
                if (fake) e.fake_depth(c);
                else {
                    e.reset();
                    std::vector<int> ids(std::max(c, 1));
                    for (int i = 0; i < (int)ids.size(); i++) ids[i] = 1000 + (i * 7919) % 150000;
                    e.prefill(ids.data(), (int)ids.size());
                    nx = e.argmax();
                }
                double sum[2] = {0, 0};
                for (int r = 0; r < abn; r++)
                    for (int vi = 0; vi < 2; vi++) {
                        const int v = (r & 1) ? 1 - vi : vi;   // ABBA: cancels the depth drift
                        e.set_ab(v ? abv : 0);
                        for (int w = 0; w < 4; w++) nx = e.step(nx);
                        e.sync();
                        auto t1 = clk::now();
                        for (int s = 0; s < nstep; s++) nx = e.step(nx);
                        const double s = secs(t1);
                        sum[v] += s;
                        printf("ab %d run %d at depth ~%d: %.3f ms/step\n", v, r, e.pos(), 1e3 * s / nstep);
                    }
                printf("A/B mean: v0 %.3f ms/step, v1 %.3f ms/step (%+.2f %%)\n", 1e3 * sum[0] / abn / nstep,
                       1e3 * sum[1] / abn / nstep, 100.0 * (sum[1] - sum[0]) / sum[0]);
                return 0;
            }
            for (int c : ctxs) {
                int nx = 1000;
                if (fake) e.fake_depth(c);
                else {
                    e.reset();
                    std::vector<int> ids(std::max(c, 1));
                    for (int i = 0; i < (int)ids.size(); i++) ids[i] = 1000 + (i * 7919) % 150000;
                    e.prefill(ids.data(), (int)ids.size());
                    nx = e.argmax();
                }
                for (int w = 0; w < 4; w++) nx = e.step(nx);
                e.sync();
                auto t1 = clk::now();
                for (int s = 0; s < nstep; s++) nx = e.step(nx);
                const double s = secs(t1);
                printf("decode at %sdepth %6d: %d steps %.3f s = %.1f tok/s (%.3f ms/step)\n", fake ? "fake " : "", c, nstep, s, nstep / s,
                       1e3 * s / nstep);
            }
        } else
            throw std::runtime_error("unknown mode " + mode);
        return 0;
    } catch (const std::exception& ex) {
        fprintf(stderr, "%s\n", ex.what());
        return 1;
    }
}
