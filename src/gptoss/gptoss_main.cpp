// gptoss_run: bring-up and bench harness of the gpt-oss-120b engine.
//   gptoss_run --model DIR --ids FILE [--prefill N] [--gen M] [--tf] [--dump DIR --dump-rows R] [--logits]
//              [--ctx N] [--chunk N] [--devices 0,1,2] [--no-graph] [--reps K]
//   --ids FILE      token ids, whitespace separated; may be repeated: the jobs run one after another on one load,
//                   job i dumps into DIR/i
//   --prefill N     prefill the first N ids in one call (default: all, or 0 with --tf)
//   --gen M         then generate M tokens greedily (timed: decode tok/s)
//   --rows R        with --tf: R rows per decode step (the multi-row step; predictions must equal --rows 1)
//   --lookup R      --gen with prompt-lookup drafts verified in multi-row steps of at most R rows: the last 6 tokens
//                   are looked up in the history; a match proposes the tokens that followed it (8, or 16 after a
//                   match of >= 24). The ids must equal the plain --gen run.
//   --tf            teacher-forced decode of the ids after the prefilled part (one decode step per id); prints the
//                   greedy prediction of every row
//   --dump DIR      write hl.bin ([36][rows][2880] residual after every layer for the first R rows), hfinal.bin
//                   (prefill: [N][2880]) and, with --logits, logits.bin ([rows][201088], decode rows of --tf)
#include "gptoss.h"
#include "gptoss_dfpolicy.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fstream>
#include <sstream>
#include <string>
#include <unistd.h>
#include <unordered_map>
#include <vector>

using namespace gptoss;
using clk = std::chrono::steady_clock;
static double secs(clk::time_point a) { return std::chrono::duration<double>(clk::now() - a).count(); }
static std::string loadavg() {
    std::ifstream f("/proc/loadavg");
    std::string s;
    std::getline(f, s);
    return s;
}
static void write_bin(const std::string& path, const void* p, size_t n) {
    FILE* f = fopen(path.c_str(), "wb");
    if (!f || fwrite(p, 1, n, f) != n) {
        fprintf(stderr, "cannot write %s\n", path.c_str());
        exit(1);
    }
    fclose(f);
}

static int run(int argc, char** argv);
int main(int argc, char** argv) {
    try {
        return run(argc, argv);
    } catch (const std::exception& e) {
        fprintf(stderr, "%s\n", e.what());
        return 1;
    }
}
static std::string default_dir(const char* env, const char* name) {
    if (const char* v = getenv(env); v && *v) return v;
    char exe[4096];
    const ssize_t n = readlink("/proc/self/exe", exe, sizeof exe - 1);
    std::string d = n > 0 ? std::string(exe, n) : std::string(".");
    d = d.substr(0, d.find_last_of('/') == std::string::npos ? 0 : d.find_last_of('/'));   // build directory
    return (d.empty() ? std::string(".") : d) + "/../models/" + name;
}
static int run(int argc, char** argv) {
    Config cfg;
    // [gptoss-rel1] default model: GPTOSS_MODEL, else <tree>/models/gpt-oss-120b next to the build directory (a link to
    // the downloaded checkpoint, as the tests read their tokenizers); --model wins
    cfg.model_dir = default_dir("GPTOSS_MODEL", "gpt-oss-120b");
    std::vector<std::string> ids_files;
    std::string dump_root, pf_logits;
    int npre = -1, ngen = 0, dump_rows = 0, reps = 1, gen_jobs = 1 << 30, rows_step = 1, lookup = 0;
    bool tf = false, want_logits = false, batch = false, swap_test = false;
    int dflash = 0, df_from = -1;   // [gptoss-r15-dflash]
    std::string df_check;
    bool df_hostfeed = false, df_auto = false;
    std::vector<int> gens;
    int batch_pre = 0;
    float temp = 0.f;
    unsigned long long seed = 1;
    float ppres = 0.f, pfreq = 0.f;
    bool pen_on = false;
    for (int i = 1; i < argc; i++) {
        const std::string a = argv[i];
        auto next = [&]() -> const char* {
            if (i + 1 >= argc) {
                fprintf(stderr, "%s needs a value\n", a.c_str());
                exit(2);
            }
            return argv[++i];
        };
        if (a == "--model") cfg.model_dir = next();
        else if (a == "--ids") ids_files.push_back(next());
        else if (a == "--prefill") npre = atoi(next());
        else if (a == "--gen") ngen = atoi(next());
        else if (a == "--gen-list") {   // tokens to generate per job (comma separated; jobs beyond the list take --gen)
            std::stringstream ss(next());
            std::string t;
            while (std::getline(ss, t, ',')) gens.push_back(atoi(t.c_str()));
        }
        else if (a == "--gen-jobs") gen_jobs = atoi(next());   // only the first N jobs generate
        else if (a == "--rows") rows_step = atoi(next());   // --tf: this many teacher-forced rows per decode step (1..16)
        else if (a == "--lookup") lookup = std::min(16, atoi(next()));   // --gen with prompt-lookup drafts, at most this many rows per step (2..16)
        else if (a == "--dflash") dflash = std::min(8, atoi(next()));   // drafter loaded; --gen with blocks of this many rows (1: plain steps, the drafter only follows)
        else if (a == "--dflash-auto") dflash = std::max(2, std::min(8, atoi(next()))), df_auto = true;   // block size per step by the policy (gptoss_dfpolicy.h), at most this
        else if (a == "--dflash-dir") cfg.dflash_dir = next();
        else if (a == "--dflash-hostfeed") df_hostfeed = true;   // the prompt's taps through the dump hook and draft_feed (the prefill then runs its dump path)
        else if (a == "--dflash-check") df_check = next();   // after a prefill of all ids: the block-4 drafts of every start >= --dflash-from, to this file
        else if (a == "--dflash-from") df_from = atoi(next());
        else if (a == "--batch") batch = true;   // teacher-forced: one row of every --ids job per step (step_rows), prints pred per job
        else if (a == "--batch-pre") batch_pre = atoi(next());   // --batch: the first N ids of every job go in by prefill, the timed steps start at row N
        else if (a == "--tf") tf = true;
        else if (a == "--temp") temp = (float)atof(next());   // [gptoss-r15-samp2] --gen / --batch: sampled steps (Gumbel-max, plain temperature); the head is the engine's default (GPTOSS_SAMP=full: the full head)
        else if (a == "--seed") seed = strtoull(next(), nullptr, 10);
        else if (a == "--pen") {   // [gptoss-rel2-penalty] --pen PRESENCE,FREQUENCY with --gen: the penalised reply (one row per step; --temp 0: the penalised argmax), then the CPU reference over the full head's logits
            std::string v = next();
            const size_t c = v.find(',');
            ppres = (float)atof(v.substr(0, c).c_str());
            pfreq = c == std::string::npos ? 0.f : (float)atof(v.substr(c + 1).c_str());
            pen_on = true;
        }
        else if (a == "--dump") dump_root = next();
        else if (a == "--dump-rows") dump_rows = atoi(next());
        else if (a == "--logits") want_logits = true;
        else if (a == "--pf-logits") pf_logits = next();   // the prefill's last-row logits to this file (no layer dump: the default collective runs)
        else if (a == "--ctx") cfg.max_ctx = atoi(next());
        else if (a == "--chunk") cfg.chunk = atoi(next());
        else if (a == "--no-graph") cfg.graph = false;
        else if (a == "--swap") swap_test = true;   // after the prefill: full-layer rows + rings to a host blob, rows zeroed, and back (timed); the run must continue as without it
        else if (a == "--reps") reps = atoi(next());
        else if (a == "--devices") {
            cfg.devices.clear();
            std::stringstream ss(next());
            std::string t;
            while (std::getline(ss, t, ',')) cfg.devices.push_back(atoi(t.c_str()));
        } else {
            fprintf(stderr, "unknown argument %s\n", a.c_str());
            return 2;
        }
    }
    if (!pf_logits.empty() && !dump_root.empty()) {
        fprintf(stderr, "--pf-logits and --dump exclude each other (a dump changes the prefill path)\n");
        return 2;
    }
    if (batch_pre > 0 && !batch) {   // review r9 L8: it was ignored silently
        fprintf(stderr, "--batch-pre needs --batch\n");
        return 2;
    }
    if (temp > 0.f && (lookup > 1 || !dump_root.empty() || want_logits || (tf && !batch))) {   // review r15 #3
        fprintf(stderr, "--temp samples in --gen (one row per step) and in --tf --batch only: not with --lookup, --dump, --logits or --tf alone\n");
        return 2;
    }
    std::vector<std::vector<int>> jobs;
    int nmax = 0;
    for (auto& fn : ids_files) {
        std::ifstream f(fn);
        std::vector<int> ids;
        int v;
        while (f >> v) ids.push_back(v);
        if (ids.empty()) {
            fprintf(stderr, "no ids in %s\n", fn.c_str());
            return 2;
        }
        nmax = std::max(nmax, (int)ids.size());
        jobs.push_back(ids);
    }
    if (jobs.empty()) {
        fprintf(stderr, "no ids (--ids FILE)\n");
        return 2;
    }
    for (int v : gens) ngen = std::max(ngen, v);   // the context is sized by the longest
    const int ngen_all = ngen;
    if (cfg.max_ctx < nmax + ngen + 1) cfg.max_ctx = nmax + ngen + 1;
    if (dflash > 0 && cfg.dflash_dir.empty()) cfg.dflash_dir = default_dir("GPTOSS_DFLASH_MODEL", "gpt-oss-120b-DFlash");
    const int npre_arg = npre;
    if (batch) {
        cfg.slots = (int)jobs.size();
        cfg.max_ctx = std::max(cfg.max_ctx, (int)jobs.size() * (nmax + 1));
    }
    Engine eng(cfg);
    if (batch) {
        // every job is a sequence of its own (slot j, full-layer rows from j * (nmax + 1)); the predictions of a job must
        // equal its --tf run alone
        const int nj = (int)jobs.size();
        if (nj > 16) {
            fprintf(stderr, "--batch takes at most 16 jobs\n");
            return 2;
        }
        std::vector<std::vector<int>> pred(nj);
        long rowsum = 0;
        if (batch_pre > 0) {
            for (int j = 0; j < nj; j++) {
                if ((int)jobs[j].size() <= batch_pre) {
                    fprintf(stderr, "--batch-pre %d: job %d has only %d ids\n", batch_pre, j, (int)jobs[j].size());
                    return 2;
                }
                eng.bind(j, j * (nmax + 1), 0, nmax + 1);
                eng.prefill(jobs[j].data(), batch_pre);
            }
            eng.bind(0, 0, 0, nmax + 1);
            printf("batch-pre: %d rows per sequence by prefill, context %d\n", batch_pre, eng.max_ctx());
        }
        const auto t0 = clk::now();
        for (int i = std::max(0, batch_pre); i < nmax; i++) {
            StepRow rows[16];
            int who[16], res[16], n = 0;
            for (int j = 0; j < nj; j++)
                if (i < (int)jobs[j].size()) {
                    rows[n] = StepRow{j, j * (nmax + 1), i, jobs[j][i]};
                    who[n++] = j;
                }
            if (temp > 0.f) {
                RowSamp rs[16];
                for (int k = 0; k < n; k++) rs[k] = RowSamp{temp, seed + (unsigned long long)who[k]};
                eng.step_rows_sampled(rows, rs, n, res);
            } else
                eng.step_rows(rows, n, res);
            for (int k = 0; k < n; k++) pred[who[k]].push_back(res[k]);
            rowsum += n;
        }
        const double s = secs(t0);
        printf("batch of %d sequences: %ld rows %.3f s = %.1f rows/s, loadavg %s\n", nj, rowsum, s, rowsum / s, loadavg().c_str());
        for (int j = 0; j < nj; j++) {
            printf("pred");
            for (int v : pred[j]) printf(" %d", v);
            printf("\n");
        }
        return 0;
    }
    printf("load %.1f s, context %d, chunk %d, loadavg %s\n", eng.load_seconds(), eng.max_ctx(), cfg.chunk, loadavg().c_str());
    if (pen_on) {
        // [gptoss-rel2-penalty] (A) the engine's penalised steps (default head: two-stage, the penalty in its bounds; a
        // GPTOSS_SAMP=full run takes the full head); (B) reference: the same ids teacher-forced through the full head
        // WITHOUT a penalty, its raw logits on the host, then on the CPU: x = l - (frequency * c + presence) for the
        // tokens of the reply so far, the Gumbel-max draw of x / T with the reply's seed (or the argmax), lowest id of
        // equal scores. Every id of (A) must be (B)'s.
        const std::vector<int>& ids = jobs[0];
        const int n = (int)ids.size(), np = npre_arg < 0 ? n : std::min(npre_arg, n), NTOK = 200019;
        const float it = temp > 0.f ? 1.f / temp : 1.f;
        std::vector<float> penv(VOCAB, 0.f);
        std::vector<int> cnt(VOCAB, 0);
        auto count = [&](int id) {
            cnt[id]++;
            penv[id] = pfreq * (float)cnt[id] + ppres;
        };
        const int first = eng.prefill(ids.data(), np);
        std::vector<int> A{first};
        count(first);
        eng.pen_reset(0, nullptr, nullptr, 0);
        int t = first;
        auto t0 = clk::now();
        for (int i = 0; i < ngen; i++) {
            eng.set_sampling(temp > 0.f ? temp : 1.f, seed, false, temp > 0.f);
            eng.set_sampling_penalty(0, t, penv[t]);
            t = eng.decode(t);
            count(t);
            A.push_back(t);
            if (i == 0) t0 = clk::now();   // (the first step makes the graph)
        }
        const double sa = secs(t0);
        printf("decode with penalties (presence %g, frequency %g, temperature %g): %d tokens %.3f s = %.2f tok/s, loadavg %s\n", ppres, pfreq, temp,
               ngen - 1, sa, (ngen - 1) / sa, loadavg().c_str());
        printf("gen");
        for (int v : A) printf(" %d", v);
        printf("\n");
        int distinct = 0, repeats = 0;
        for (int v = 0; v < VOCAB; v++) distinct += cnt[v] > 0, repeats += cnt[v] > 1;
        eng.bind(0, 0, 0, eng.max_ctx());
        eng.prefill(ids.data(), np);
        std::fill(penv.begin(), penv.end(), 0.f);
        std::fill(cnt.begin(), cnt.end(), 0);
        count(first);
        auto mix = [](unsigned long long x) {
            x += 0x9E3779B97F4A7C15ull;
            x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
            x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
            return x ^ (x >> 31);
        };
        std::vector<float> x(VOCAB);
        int bad = 0, moved = 0;
        for (int i = 0; i < ngen; i++) {
            const int pos = eng.pos();
            eng.set_sampling(1.f, seed, true);
            eng.decode(A[i]);
            for (int g = 0; g < eng.plan().ngpu; g++) {
                const SampleCard sc = eng.sample_card(g);
                memcpy(x.data() + sc.v0, sc.logits, (size_t)sc.vn * 4);
            }
            int best = -1, plain = -1;
            float bs = 0.f, ps = 0.f, second = -INFINITY;
            for (int j = 0; j < NTOK; j++) {
                if (!std::isfinite(x[j])) continue;
                float sc = x[j] - penv[j], pl = x[j];
                if (temp > 0.f) {
                    const unsigned long long r = mix(seed ^ mix(((unsigned long long)(unsigned)pos << 32) | (unsigned)j));
                    const float u = ((float)(r >> 41) + 0.5f) * (1.0f / 8388608.0f);
                    const float gn = -logf(-logf(u));
                    sc = fmaf(sc, it, gn);
                    pl = fmaf(pl, it, gn);
                }
                if (best < 0 || sc > bs) second = best < 0 ? second : bs, bs = sc, best = j;
                else if (sc > second) second = sc;
                if (plain < 0 || pl > ps) ps = pl, plain = j;
            }
            moved += best != plain;
            if (best != A[i + 1]) {
                if (bad++ < 8) printf("step %d: engine %d, reference %d (score %.9g, runner-up %.9g)\n", i, A[i + 1], best, bs, second);
            }
            count(A[i + 1]);
        }
        printf("PEN CHECK %s: %d of %d ids equal to the CPU reference; the penalty changed the pick at %d steps; %d distinct tokens, %d repeated\n",
               bad ? "FAIL" : "PASS", ngen - bad, ngen, moved, distinct, repeats);
        return bad ? 1 : 0;
    }
    if (dflash > 0 && !eng.has_drafter()) {
        fprintf(stderr, "--dflash: the drafter was not loaded\n");
        return 2;
    }
    if (eng.has_drafter()) printf("drafter: %.0f MiB on the fullest card\n", eng.draft_vram_mb());

    const bool dumping = !dump_root.empty();
    if (dumping && dump_rows <= 0) dump_rows = 1;
    for (size_t job = 0; job < jobs.size(); job++)
    for (int rep = 0; rep < reps; rep++) {
        const std::vector<int>& ids = jobs[job];
        const int n = (int)ids.size();
        npre = npre_arg < 0 ? (tf ? 0 : n) : std::min(npre_arg, n);
        const std::string dump_dir = dump_root + "/" + std::to_string(job);
        std::vector<float> hl, hfinal, logits, lrow;
        Dump d;
        if (dumping) {
            if (system(("mkdir -p '" + dump_dir + "'").c_str()) != 0) return 1;
            hl.assign((size_t)NL * dump_rows * H, 0.f);
            d.rows = dump_rows;
        }
        printf("job %zu: %d ids\n", job, n);
        eng.reset();
        eng.set_sampling(0.f, 0, false);
        int next_tok = -1;
        if (npre > 0) {
            if (dumping) {
                d.layers = hl.data();
                hfinal.assign((size_t)npre * H, 0.f);
                d.hfinal = hfinal.data();
                eng.set_dump(d);
            }
            std::vector<float> dtap;   // [gptoss-r15-dflash] prototype: the prefill rows' taps through the dump hook
            if (!dumping && eng.has_drafter() && df_hostfeed) {
                dtap.assign((size_t)5 * npre * H, 0.f);
                Dump dd;
                dd.layers = dtap.data();
                dd.rows = npre;
                dd.taps = true;
                eng.set_dump(dd);
            }
            std::vector<float> pfl;
            if (!dumping && !pf_logits.empty() && !(eng.has_drafter() && df_hostfeed)) {
                pfl.assign(VOCAB, 0.f);
                Dump dl;
                dl.logits = pfl.data();
                eng.set_dump(dl);
            }
            const auto t0 = clk::now();
            next_tok = eng.prefill(ids.data(), npre);
            const double s = secs(t0);
            printf("prefill %d tokens %.3f s = %.0f tok/s, next %d, loadavg %s\n", npre, s, npre / s, next_tok, loadavg().c_str());
            if (!pfl.empty()) {
                write_bin(pf_logits, pfl.data(), pfl.size() * 4);
                eng.set_dump(Dump{});
            }
            if (!dtap.empty()) {
                eng.set_dump(Dump{});
                const auto t1 = clk::now();
                std::vector<float> rows((size_t)npre * 5 * H);
                for (int t = 0; t < npre; t++)
                    for (int i = 0; i < 5; i++) memcpy(&rows[((size_t)t * 5 + i) * H], &dtap[((size_t)i * npre + t) * H], H * 4);
                eng.draft_truncate(0, 0);
                eng.draft_feed(0, rows.data(), npre);
                printf("drafter context: %d rows in %.3f s\n", npre, secs(t1));
            }
            if (eng.has_drafter() && !dumping) {
                if (!df_check.empty()) {
                    FILE* f = fopen((df_check + "." + std::to_string(job)).c_str(), "w");
                    if (!f) return 1;
                    const int B = dflash >= 2 ? dflash : 4;
                    int dr[8];
                    const auto t2 = clk::now();
                    int cnt = 0;
                    for (int s0 = npre - 2; s0 >= std::max(1, df_from); s0--, cnt++) {
                        eng.draft_truncate(0, s0);
                        eng.draft_block(0, s0, ids[s0], B, dr);
                        fprintf(f, "%d", s0);
                        for (int k = 0; k < B - 1; k++) fprintf(f, " %d", dr[k]);
                        fprintf(f, "\n");
                    }
                    fclose(f);
                    printf("drafter check: %d blocks of %d, %.3f ms per pass\n", cnt, B, 1e3 * secs(t2) / std::max(1, cnt));
                }
            }
            if (dumping) {
                write_bin(dump_dir + "/hl.bin", hl.data(), hl.size() * 4);
                write_bin(dump_dir + "/hfinal.bin", hfinal.data(), hfinal.size() * 4);
                eng.set_dump(Dump{});
            }
        }
        if (swap_test && npre > 0) {
            std::vector<unsigned char> blob, ring, zero;
            auto t0 = clk::now();
            eng.rows_save(0, npre, blob);
            eng.ring_save(ring);
            const double ss = secs(t0);
            zero.assign(blob.size(), 0);
            eng.rows_load(zero, 0, npre);
            t0 = clk::now();
            eng.rows_load(blob, 0, npre);
            eng.ring_load(ring, npre);
            const double sl = secs(t0);
            printf("swap %d rows: %.1f MB (%zu B per row); to host %.3f s = %.2f GB/s, back %.3f s = %.2f GB/s\n", npre, blob.size() / 1e6, eng.row_bytes(), ss,
                   blob.size() / 1e9 / ss, sl, blob.size() / 1e9 / sl);
        }
        if (tf && npre < n) {
            // teacher-forced decode rows: per-row layer dumps (stored transposed below) and logits
            const int rows = n - npre;
            std::vector<int> pred(rows);
            std::vector<float> one((size_t)NL * H);
            if (dumping && npre == 0) hl.assign((size_t)NL * dump_rows * H, 0.f);
            if (dumping && want_logits) {
                logits.resize((size_t)rows * VOCAB);
                lrow.resize(VOCAB);
            }
            const auto t0 = clk::now();
            for (int i = 0; i < rows; i++) {
                if (dumping) {
                    Dump dd;
                    if (npre == 0 && i < dump_rows) {
                        dd.layers = one.data();
                        dd.rows = 1;
                    }
                    if (want_logits) dd.logits = lrow.data();
                    eng.set_dump(dd);
                }
                if (rows_step > 1 && !dumping) {
                    const int R = std::min(rows_step, rows - i);
                    eng.decode_rows(&ids[npre + i], R, &pred[i]);
                    i += R - 1;
                    continue;
                }
                pred[i] = eng.decode(ids[npre + i]);
                if (dumping && npre == 0 && i < dump_rows)
                    for (int L = 0; L < NL; L++) memcpy(&hl[((size_t)L * dump_rows + i) * H], &one[(size_t)L * H], H * 4);
                if (dumping && want_logits) memcpy(&logits[(size_t)i * VOCAB], lrow.data(), (size_t)VOCAB * 4);
            }
            const double s = secs(t0);
            printf("tf decode %d rows %.3f s = %.1f tok/s%s\n", rows, s, rows / s, dumping ? " (dump mode, eager)" : "");
            printf("pred");
            for (int v : pred) printf(" %d", v);
            printf("\n");
            if (dumping) {
                if (npre == 0) write_bin(dump_dir + "/hl.bin", hl.data(), hl.size() * 4);
                if (want_logits) write_bin(dump_dir + "/logits.bin", logits.data(), logits.size() * 4);
                eng.set_dump(Dump{});
            }
            next_tok = pred.back();
        }
        if ((job < gens.size() ? gens[job] : ngen_all) > 0 && (int)job < gen_jobs) {
            const int ngen = job < gens.size() ? gens[job] : ngen_all;
            std::vector<int> out;
            out.push_back(next_tok);
            if (temp > 0.f) eng.set_sampling(temp, seed, false);   // [gptoss-r15-samp2]
            int t = eng.decode(next_tok);   // warm-up step (graph capture), not timed
            if (eng.has_drafter()) eng.draft_commit(0, 0, 1);   // [gptoss-r15-dflash] the warm-up step's row
            out.push_back(t);
            if (lookup > 1 || dflash > 1) {   // capture the step graph of every row count before the timed part (review L3)
                const int before = eng.pos();
                int rowsv[16], res[16];
                for (int k = 0; k < 16; k++) rowsv[k] = t;
                for (int R = 2; R <= std::max(lookup, dflash) && before + R < eng.max_ctx(); R++) {
                    eng.decode_rows(rowsv, R, res);
                    eng.truncate(before);
                }
            }
            if (dflash > 1) {   // the drafter's pass graphs of every block size (no new rows), before the timed part
                int dr[8];
                for (int B = 2; B <= dflash; B++) eng.draft_block(0, eng.pos(), t, B, dr);
                eng.draft_warm(0, dflash);   // [phase 2] the graphs of passes with new rows too (a graph made in the timed part costs ms)
            }
            if (df_auto) {   // [phase 2] the process-wide cost table as a warm process has it: one timed 4-row step and drafter pass (untimed part)
                DfCosts& C = DfCosts::get();
                const int b = DfCosts::bucket(eng.pos()), before = eng.pos();
                int rowsv[4] = {t, t, t, t}, res[4], dr[8];
                auto ta = clk::now();
                eng.decode(t);
                DfCosts::upd(C.tv[b][1], C.hv[b][1], 1e3 * secs(ta));
                eng.truncate(before);
                ta = clk::now();
                eng.decode_rows(rowsv, 4, res);
                DfCosts::upd(C.tv[b][4], C.hv[b][4], 1e3 * secs(ta));
                eng.truncate(before);
                ta = clk::now();
                eng.draft_block(0, eng.pos(), t, 4, dr);
                DfCosts::upd(C.td[b][4], C.hd[b][4], 1e3 * secs(ta));
            }
            const auto t0 = clk::now();
            const std::string la = loadavg();
            const int timed = ngen - 2, half = timed / 2;
            double s_half = 0;
            long steps = 0, rows_sum = 0;
            double df_ms = 0, ver_ms = 0, flush_ms = 0;
            long idle = 0, since_block = 0;
            long df_n = 0, lk_steps = 0;
            // the rows the engine holds before out[0]: the prefilled ids, plus the teacher-forced ones after a --tf part
            // (review m2 L9: the lookup history left the teacher-forced ids out; the engine holds them)
            const int held0 = tf && npre < n ? n : npre;
            if (dflash > 0) {   // [gptoss-r15-dflash] block drafts (+ the lookup route for copies of >= 24 tokens when --lookup is set)
                std::vector<int> hist(ids.begin(), ids.begin() + held0);
                hist.insert(hist.end(), out.begin(), out.end());
                std::unordered_map<unsigned long long, int> idx;
                auto key = [&](int p) {
                    unsigned long long h = 1469598103934665603ull;
                    for (int k = p - 6; k < p; k++) h = (h ^ (unsigned long long)hist[k]) * 1099511628211ull;
                    return h;
                };
                int known = 6;
                DfPolicy pol;
                pol.maxb = dflash;
                pol.reset();
                long bhist[9] = {};
                int sh[8], sh_k = 0;   // [phase 2] shadow block in flight
                bool sh_on = false;
                double sh_ms = 0, shadow_ms = 0;
                while ((int)out.size() < ngen) {
                    const int n0 = (int)hist.size();
                    int rowsv[16], res[16], nd = 0;
                    rowsv[0] = t;
                    const int bwant = df_auto ? pol.choose(eng.pos()) : dflash;
                    const int pos_step = eng.pos();
                    double dms = 0;
                    bool drafted = false;
                    const int room = std::min(ngen - (int)out.size() - 1, eng.max_ctx() - eng.pos() - 1);
                    bool lk = false;
                    if (lookup > 1 && n0 >= 6) {
                        for (; known < n0; known++) idx[key(known)] = known;
                        auto it = idx.find(key(n0));
                        if (it != idx.end()) {
                            const int j = it->second;
                            bool same = true;
                            for (int k = 1; k <= 6 && same; k++) same = hist[j - k] == hist[n0 - k];
                            int m = 6;
                            while (same && m < 64 && j - m - 1 >= 0 && hist[j - m - 1] == hist[n0 - m - 1]) m++;
                            if (same && m >= 24) {
                                const int want = std::min(std::min(16, lookup) - 1, room);
                                for (; nd < want && j + nd < n0; nd++) rowsv[1 + nd] = hist[j + nd];
                                lk = nd > 0;
                            }
                        }
                    }
                    if (!lk && df_auto && pol.want_shadow && room >= 3) {   // [phase 2] a shadow block: the drafter alone, its drafts judged by the next plain steps
                        eng.draft_drop(0, 16);   // after plain steps only the newest rows join the context (one pass of 16)
                        const auto tf = clk::now();
                        eng.draft_block(0, eng.pos(), 0, 0, nullptr);
                        flush_ms += 1e3 * secs(tf);
                        const auto ta = clk::now();
                        eng.draft_block(0, eng.pos(), t, 4, sh);
                        sh_ms = 1e3 * secs(ta);
                        shadow_ms += sh_ms;
                        sh_k = 0;
                        sh_on = true;
                        since_block = 0;
                        pol.shadow_started();
                    } else if (!lk && bwant >= 2 && room >= 1) {
                        if (df_auto && since_block > 16) eng.draft_drop(0, 16);   // a block after plain steps: only the newest rows
                        if (++since_block > 4) {   // rows queued over several steps: their passes are not the block's time
                            const auto tf = clk::now();
                            eng.draft_block(0, eng.pos(), 0, 0, nullptr);
                            flush_ms += 1e3 * secs(tf);
                        }
                        since_block = 0;
                        const auto ta = clk::now();
                        const int B = std::min(bwant, room + 1);
                        eng.draft_block(0, eng.pos(), t, B, rowsv + 1);
                        nd = B - 1;
                        dms = 1e3 * secs(ta);
                        df_ms += dms;
                        df_n++;
                        drafted = true;
                    } else if (!df_auto && eng.draft_count(0) == eng.pos() && (++idle & 15) == 0) {
                        const auto ta = clk::now();
                        eng.draft_block(0, eng.pos(), 0, 0, nullptr);   // --dflash 1: keep the context current (16 rows per pass); the policy feeds only while a block runs
                        flush_ms += 1e3 * secs(ta);
                    } else if (eng.draft_count(0) != eng.pos())
                        fprintf(stderr, "drafter context %d != position %d\n", eng.draft_count(0), eng.pos());
                    lk_steps += lk;
                    if (!drafted) since_block++;
                    const int before = eng.pos();
                    const auto tb = clk::now();
                    if (temp > 0.f && nd > 0) {   // [phase 2] sampled verify: every row draws its own token (seeded by (seed, position));
                        // a draft is kept only if it equals the token the row before it drew, so the ids are those of plain sampled steps
                        StepRow sr[16];
                        RowSamp rsv[16];
                        for (int k = 0; k <= nd; k++) sr[k] = StepRow{0, 0, before + k, rowsv[k]}, rsv[k] = RowSamp{temp, seed};
                        eng.step_rows_sampled(sr, rsv, nd + 1, res);
                        eng.bind(0, 0, before + nd + 1, eng.max_ctx());
                    } else
                        eng.decode_rows(rowsv, nd + 1, res);   // (one row: sampled when --temp is set, set_sampling)
                    const double vms = 1e3 * secs(tb);
                    ver_ms += vms;
                    int acc = 0;
                    while (acc < nd && rowsv[1 + acc] == res[acc]) acc++;
                    if (acc < nd) eng.truncate(before + acc + 1);
                    eng.draft_commit(0, 0, acc + 1);
                    if (!lk) pol.seen(nd + 1, dms, vms, acc, drafted, pos_step);
                    bhist[std::min(nd + 1, 8)]++;
                    for (int k = 0; k <= acc; k++) {
                        out.push_back(res[k]);
                        hist.push_back(res[k]);
                        if (sh_on) {   // the shadow's drafts against the tokens as they come
                            if (res[k] == sh[sh_k] && ++sh_k < 3) continue;
                            pol.shadow_seen(4, sh_k, sh_ms);
                            sh_on = false;
                        }
                    }
                    t = res[acc];
                    steps++;
                    rows_sum += nd + 1;
                    if (s_half == 0 && (int)out.size() - 2 >= half) s_half = secs(t0);
                }
                out.resize(ngen);
                if (df_auto) {
                    printf("policy: rows per step 1..8+:");
                    for (int b = 1; b <= 8; b++) printf(" %ld", bhist[b]);
                    printf("; acceptance %.2f; verify ms R1 %.2f R4 %.2f; drafter ms B4 %.2f; context flush %.1f ms; shadows %ld (%.1f ms, %ld left the block off); final %d, shadow gap %ld\n",
                           pol.acc[0], pol.tv1(), pol.verify_ms(4), pol.draft_ms(4), flush_ms, pol.tries, shadow_ms, pol.fails, pol.cur, pol.try_gap);
                }
            } else
            if (lookup > 1) {
                // history = prompt + everything emitted; idx[hash of 6 tokens] = the latest position that follows them
                std::vector<int> hist(ids.begin(), ids.begin() + held0);
                hist.insert(hist.end(), out.begin(), out.end());
                std::unordered_map<unsigned long long, int> idx;
                auto key = [&](int p) {   // the 6 tokens before position p
                    unsigned long long h = 1469598103934665603ull;
                    for (int k = p - 6; k < p; k++) h = (h ^ (unsigned long long)hist[k]) * 1099511628211ull;
                    return h;
                };
                int known = 6;   // positions < known are indexed
                while ((int)out.size() < ngen) {
                    const int n0 = (int)hist.size();   // hist[n0 - 1] == t, the token to feed
                    int rowsv[16], res[16], nd = 0;
                    rowsv[0] = t;
                    if (n0 >= 6) {
                        for (; known < n0; known++) idx[key(known)] = known;
                        auto it = idx.find(key(n0));
                        if (it != idx.end()) {
                            const int j = it->second;
                            bool same = true;
                            for (int k = 1; k <= 6 && same; k++) same = hist[j - k] == hist[n0 - k];
                            if (same) {
                                int m = 6;
                                while (m < 64 && j - m - 1 >= 0 && hist[j - m - 1] == hist[n0 - m - 1]) m++;
                                const int want = std::min(std::min(m >= 24 ? 16 : 8, lookup - 1), ngen - (int)out.size() - 1);
                                for (; nd < want && j + nd < n0; nd++) rowsv[1 + nd] = hist[j + nd];
                            }
                        }
                    }
                    const int before = eng.pos();
                    eng.decode_rows(rowsv, nd + 1, res);
                    int acc = 0;
                    while (acc < nd && rowsv[1 + acc] == res[acc]) acc++;
                    if (acc < nd) eng.truncate(before + acc + 1);
                    for (int k = 0; k <= acc; k++) {
                        out.push_back(res[k]);
                        hist.push_back(res[k]);
                    }
                    t = res[acc];
                    steps++;
                    rows_sum += nd + 1;
                    if (s_half == 0 && (int)out.size() - 2 >= half) s_half = secs(t0);
                }
                out.resize(ngen);
            } else
            for (int i = 2; i < ngen; i++) {
                t = eng.decode(t);
                out.push_back(t);
                if (i - 1 == half) s_half = secs(t0);
            }
            const double s = secs(t0);
            if (dflash > 0)
                printf("dflash %d: %ld steps, %.3f tokens per step, %.2f rows per step, drafter %.3f ms x %ld, verify %.3f ms per step, lookup steps %ld\n", dflash, steps,
                       (double)timed / steps, (double)rows_sum / steps, df_n ? df_ms / df_n : 0., df_n, ver_ms / std::max(1L, steps), lk_steps);
            else if (lookup > 1) printf("lookup drafts: %ld steps, %.2f tokens per step, %.2f rows per step\n", steps, (double)timed / steps, (double)rows_sum / steps);
            if (timed > 0) printf("decode %d tokens %.3f s = %.2f tok/s (%.3f ms/token), loadavg %s\n", timed, s, timed / s, 1e3 * s / timed, la.c_str());
            if (half > 0)
                printf("  first half %d tokens %.2f tok/s, second half %d tokens %.2f tok/s, loadavg after %s\n", half, half / s_half,
                       timed - half, (timed - half) / (s - s_half), loadavg().c_str());
            printf("gen");
            for (int v : out) printf(" %d", v);
            printf("\n");
        }
    }
    return 0;
}
