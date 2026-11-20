#include <sys/resource.h>
// dsv4_run: M1 CLI for DeepSeek-V4.1-Flash-1017.
//   dsv4_run --packed DIR --ids FILE [--gen N] [--ppl] [--check K --dump DIR] [--placement FILE] [--chunk C]
//            [--ctx N] [--cpu-threads T] [--out FILE]
// ids FILE: whitespace-separated token ids (prompt). Output: generated ids (one line) to --out or stdout;
// timings on stderr.
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cerrno>
#include <cstring>
#include <sys/stat.h>
#include <unistd.h>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

#include "dsv4.h"

using clk = std::chrono::steady_clock;
static double since(clk::time_point t) { return std::chrono::duration<double>(clk::now() - t).count(); }

static int g_argc;
static char** g_argv;
static int g_ids_arg = -1;   // review r13 L4: argv index of the --ids value (a pipe is replaced by a temp copy)
static bool g_ids_tmp = false;   // that copy was made (the re-exec'd child gets --ids-is-tmp and unlinks it)
static void reload_without_dspark() {   // R13: exec this binary again with --dspark 0 appended (all memory released)
    static std::vector<char*> av;
    for (int i = 0; i < g_argc; i++) av.push_back(g_argv[i]);
    static char a1[] = "--dspark", a2[] = "0", a3[] = "--no-ram-reserve", a4[] = "--ids-is-tmp";
    av.push_back(a1);
    av.push_back(a2);
    av.push_back(a3);   // the test hook's shortfall applies to the refused load only (was an unsetenv: issue #11 rule)
    if (g_ids_tmp) av.push_back(a4);
    av.push_back(nullptr);
    fflush(stdout);
    fflush(stderr);
    execv("/proc/self/exe", av.data());
    perror("execv");
    exit(3);
}
// review r15 M5 / S2: rows of two logit dumps [K][V] compared bit by bit; any NaN in either counts apart (a fail)
struct RowCmp {
    int nbad = 0, first = -1, amd = 0, nnan = 0;
    double worst = 0;
    std::string per;   // max |d| of the first 12 rows
};
static RowCmp cmp_rows(const float* A, const float* B, int K) {
    RowCmp r;
    for (int i = 0; i < K; i++) {
        const float* a = A + (size_t)i * dsv4::V;
        const float* b = B + (size_t)i * dsv4::V;
        const bool diff = memcmp(a, b, (size_t)dsv4::V * 4) != 0;
        double md = 0;
        int a1 = 0, a2 = 0;
        bool nan = false;
        for (int v = 0; v < dsv4::V; v++) {
            nan |= std::isnan(a[v]) || std::isnan(b[v]);
            const double d = fabs((double)a[v] - (double)b[v]);
            if (d > md) md = d;
            if (a[v] > a[a1]) a1 = v;
            if (b[v] > b[a2]) a2 = v;
        }
        r.nnan += nan;
        r.amd += a1 != a2;
        if (diff || nan) {
            r.nbad++;
            if (r.first < 0) r.first = i;
        }
        r.worst = std::max(r.worst, md);
        if (i < 12) {
            char t[32];
            snprintf(t, sizeof t, " %.2g%s", md, nan ? "(nan)" : "");
            r.per += t;
        }
    }
    return r;
}

int main(int argc, char** argv) {
    g_argc = argc;
    g_argv = argv;
    dsv4::Options o;
    o.reload_without_dspark = reload_without_dspark;
    std::string ids_file, out_file, jobs_file;
    bool ids_is_tmp = false;   // --ids-is-tmp: the re-exec of a DSpark refusal, --ids is the parent's temp copy
    int gen = 0, check = 0;
    bool ppl = false;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto nx = [&] { return std::string(argv[++i]); };
        if (a == "--packed") o.packed = nx();
        else if (a == "--gguf-experts") o.gguf_dir = nx();   // R16 [dsv4-gguf-iq]: Maya-S GGUF dir (with an M23 --packed)
        else if (a == "--ckpt") o.ckpt = nx();
        else if (a == "--gguf") o.gguf_model = nx();      // release prep [dsv4-rel-gguf]: the whole model from a GGUF dir
        else if (a == "--dspark-dir") o.dspark_dir = nx();
        else if (a == "--cache-dir") o.cache_dir = nx();
        else if (a == "--profile") o.profile = nx();
        else if (a == "--ids") { ids_file = nx(); g_ids_arg = i; }
        else if (a == "--no-ram-reserve") o.ram_reserve_off = true;
        else if (a == "--ids-is-tmp") ids_is_tmp = true;
        else if (a == "--gen") gen = atoi(nx().c_str());
        else if (a == "--ppl") ppl = true;
        else if (a == "--check") check = atoi(nx().c_str());
        else if (a == "--dump") o.dump = nx();
        else if (a == "--placement") o.placement_out = nx();
        else if (a == "--placement-in") o.placement_in = nx();
        else if (a == "--chunk") o.chunk = atoi(nx().c_str());
        else if (a == "--ctx") o.maxctx = atoi(nx().c_str());
        else if (a == "--cpu-threads") o.cpu_threads = atoi(nx().c_str());
        else if (a == "--reserve-gb") o.reserve_gb = atof(nx().c_str());
        else if (a == "--dyn") o.dyn = atoi(nx().c_str());
        else if (a == "--asplit") o.asplit = atoi(nx().c_str());
        else if (a == "--pdist") o.pdist = atoi(nx().c_str());
        else if (a == "--swap") o.swap = atoi(nx().c_str());
        else if (a == "--swap-cap") o.swap_cap = atof(nx().c_str());
        else if (a == "--dyn-margin") { o.dyn_margin = atof(nx().c_str()); o.dyn_margin_set = true; }
        else if (a == "--xsplit") o.xsplit = atoi(nx().c_str());
        else if (a == "--borrow") o.borrow = atoi(nx().c_str());
        else if (a == "--dyn-window") o.dyn_window = atoi(nx().c_str());   // < 0: no admissions (checks)
        else if (a == "--hsplit") o.hsplit = atoi(nx().c_str());
        else if (a == "--attn-tc") o.attn_tc = atoi(nx().c_str());
        else if (a == "--prefetch") o.prefetch = atoi(nx().c_str());
        else if (a == "--pf-early") o.pf_early = atoi(nx().c_str());
        else if (a == "--prefetch-mr") o.prefetch_mr = atoi(nx().c_str());
        else if (a == "--small-prefill") o.small_prefill = atoi(nx().c_str());
        else if (a == "--out") out_file = nx();
        else if (a == "--dspark") o.dspark = atoi(nx().c_str());
        else if (a == "--dspark-n") o.dspark_n = atoi(nx().c_str());
        else if (a == "--dspark-cb3") o.dspark_cb3 = atoi(nx().c_str());
        else if (a == "--dspark-adapt") o.dspark_adapt = atoi(nx().c_str());
        else if (a == "--band-cb3") o.band_cb3 = atoi(nx().c_str());
        else if (a == "--dspark-tau") o.dspark_tau = atof(nx().c_str());
        else if (a == "--nv-cap") o.nv_cap = atof(nx().c_str());
        else if (a == "--car-margin") o.car_margin = atof(nx().c_str());
        else if (a == "--drop-w") o.drop_w = atof(nx().c_str());
        else if (a == "--pool-min") o.pool_min = atoi(nx().c_str());
        else if (a == "--host-budget-gib") o.host_budget_gib = atof(nx().c_str());
        else if (a == "--jobs") jobs_file = nx();
        else {
            fprintf(stderr, "unknown arg %s\n", a.c_str());
            return 2;
        }
    }
    auto read_ids = [](const std::string& fn) {
        std::vector<int> v;
        std::ifstream f(fn);
        int t;
        while (f >> t) v.push_back(t);
        return v;
    };
    std::vector<int> ids = read_ids(ids_file);
    if (ids_is_tmp) {   // re-review N2: this is the re-exec'd child: drop the parent's copy
        if (g_ids_arg >= 0) unlink(ids_file.c_str());
    } else if (g_ids_arg >= 0) {   // review r13 L4: a pipe / device cannot be read again after a re-exec: copy it
        struct stat sb;
        if (stat(ids_file.c_str(), &sb) == 0 && (S_ISFIFO(sb.st_mode) || S_ISCHR(sb.st_mode) || S_ISSOCK(sb.st_mode))) {
            static std::string tpath;
            const char* td = getenv("TMPDIR");
            tpath = std::string(td && *td ? td : "/tmp") + "/dsv4_ids_XXXXXX";
            int fd = mkstemp(tpath.data());
            if (fd < 0) {
                fprintf(stderr, "error: --ids %s is a pipe and no temp copy could be made in %s (%s): a DSpark refusal "
                        "could not reload it\n", ids_file.c_str(), td && *td ? td : "/tmp", strerror(errno));
            } else {
                std::string t;
                for (int x : ids) t += std::to_string(x) + " ";
                if (write(fd, t.data(), t.size()) != (ssize_t)t.size()) perror("ids copy");
                close(fd);
                g_argv[g_ids_arg] = tpath.data();
                g_ids_tmp = true;   // the child (--ids-is-tmp) unlinks it after reading
                atexit([] { unlink(tpath.c_str()); });      // no re-exec: removed at exit
            }
        }
    }
    if (ids.empty() && jobs_file.empty()) {
        fprintf(stderr, "no prompt ids\n");
        return 2;
    }
    dsv4::Engine* E = dsv4::create(o);
    dsv4::print_memory(E);
    if (!jobs_file.empty()) {
        // one load, several jobs: lines "ppl IDS" | "gen IDS N OUT" (greedy, timings) | "prefill IDS REPEAT"
        std::ifstream jf(jobs_file);
        std::string kind, fn;
        while (jf >> kind >> fn) {
            if (kind == "set") {   // "set prefetch N": runtime option between jobs
                int v;
                jf >> v;
                if (fn == "prefetch") dsv4::set_prefetch(E, v);
                if (fn == "prefetch_mr") dsv4::set_prefetch_mr(E, v);
                if (fn == "pf_early") dsv4::set_pf_early(E, v);
                if (fn == "nv_cap") dsv4::set_nv_cap(E, (float)v);
                if (fn == "car_margin1000") dsv4::set_car(E, v / 1000.f);   // -1 = off
                if (fn == "drop_w1000") dsv4::set_drop(E, v / 1000.f);   // 0 = off
                if (fn == "dyn_window") dsv4::set_dyn_window(E, v);   // R16: -1 = freeze the tier (md5 checks)
                if (fn == "small_prefill") dsv4::set_small_prefill(E, v);
                if (fn == "pprof") dsv4::set_pprof(E, v);   // [dsv4-pf-pprof] prefill profile level per job
                if (fn == "f32k") dsv4::set_f32k(E, v);
                if (fn == "q1") dsv4::set_q1(E, v);
                if (fn == "runahead") dsv4::set_runahead(E, v);
                if (fn == "fuse") dsv4::set_fuse(E, v);
                if (fn == "qrb") dsv4::set_qrb(E, v);
                if (fn == "dspark_n") dsv4::set_dspark(E, v, -1.f);
                if (fn == "dspark_tau100") dsv4::set_dspark(E, -1, v / 100.f);
                if (fn == "lookup") dsv4::set_lookup(E, v);
                if (fn == "gg_fp32") dsv4::set_gg_fp32(E, v);   // R21: GGUF pool kernel per job (1 = R16 fp32, 0 = dp4a)
                if (fn == "sleep_s") {   // R24: idle between jobs (an observer reads /proc, e.g. pagemap)
                    fprintf(stderr, "JOB sleep %d s\n", v);
                    std::this_thread::sleep_for(std::chrono::seconds(v));
                }
                {   // [dsv4-rel-sample]: sampling (temperature 0 = greedy)
                    static float st = 0.f, sp = 1.f;
                    static int sk = 0;
                    static long ss = 0;
                    bool hit = true;
                    if (fn == "sample_t100") st = v / 100.f;
                    else if (fn == "sample_p100") sp = v / 100.f;
                    else if (fn == "sample_k") sk = v;
                    else if (fn == "sample_seed") ss = v;
                    else hit = false;
                    if (hit) dsv4::set_sampling(E, st, sp, sk, 0.f, (unsigned long long)ss);
                }
                fprintf(stderr, "JOB set %s %d\n", fn.c_str(), v);
                continue;
            }
            std::vector<int> p = read_ids(fn);
            dsv4::reset(E);
            if (kind == "ppl") {
                std::vector<float> nll;
                std::vector<int> am;
                auto t0 = clk::now();
                dsv4::prefill(E, p, &nll, &am);
                double s = 0;
                int ok = 0, n = (int)p.size() - 1;
                for (int i = 0; i < n; i++) {
                    s += nll[i];
                    ok += am[i] == p[i + 1];
                }
                fprintf(stderr, "JOB ppl %s: ppl %.4f top1 %.4f over %d positions (%.1f s)\n", fn.c_str(), exp(s / n),
                        (double)ok / n, n, since(t0));
            } else if (kind == "prefill") {
                int rep;
                jf >> rep;
                for (int r = 0; r < rep; r++) {
                    dsv4::reset(E);
                    auto t0 = clk::now();
                    dsv4::prefill(E, p);
                    double tp = since(t0);
                    fprintf(stderr, "JOB prefill %s: %zu tokens %.3f s = %.1f tok/s (TTFT)\n", fn.c_str(), p.size(), tp,
                            p.size() / tp);
                }
            } else if (kind == "tflp" || kind == "tflpe") {   // R14: "tflp IDS K OUT": last K tokens teacher-forced
                // one by one, the log-softmax of every position as float32 [K][V] to OUT (engine-side KL gate).
                // review r14 M3: tflp runs decode() = the default step-graph path; tflpe the eager forward_rows path.
                // review r14 L9: OUT starts with a 32-byte header (TFLP1, K, V, FNV-1a hash of the ids file's tokens).
                // review r14 L5: reset() keeps the dynamic tier's state (residency, scores, homes): for A/B dumps use
                // the same job order in both runs, or a fresh process
                int K;
                std::string of;
                jf >> K >> of;
                if (!(K > 0 && K < (int)p.size())) {   // review r14 L4
                    fprintf(stderr, "JOB %s %s: K %d must be in 1 .. %zu (tokens - 1): skipped\n", kind.c_str(), fn.c_str(), K,
                            p.size() - 1);
                    continue;
                }
                FILE* fo = fopen(of.c_str(), "wb");
                if (!fo) {
                    fprintf(stderr, "JOB %s %s: cannot open %s: %s: skipped\n", kind.c_str(), fn.c_str(), of.c_str(), strerror(errno));
                    continue;
                }
                uint64_t h = 1469598103934665603ull;
                for (int v : p)
                    for (int b = 0; b < 4; b++) { h ^= (uint8_t)((uint32_t)v >> (8 * b)); h *= 1099511628211ull; }
                char hdr[32] = {'T', 'F', 'L', 'P', '1', 0, 0, 0};
                const int32_t kv[2] = {K, dsv4::V};
                memcpy(hdr + 8, kv, 8);
                memcpy(hdr + 16, &h, 8);
                bool ok = fwrite(hdr, 1, 32, fo) == 32;
                std::vector<int> pre(p.begin(), p.end() - K), tail(p.end() - K, p.end());
                std::vector<float> L((size_t)dsv4::V);
                dsv4::prefill(E, pre);
                const bool eager = kind == "tflpe";
                for (int i = 0; i < K && ok; i++) {
                    if (eager) dsv4::forward_rows(E, &tail[i], 1, L.data());
                    else dsv4::decode_logits(E, tail[i], L.data());
                    double mx = -1e30, se = 0;
                    for (float v : L) mx = std::max(mx, (double)v);
                    for (float v : L) se += exp((double)v - mx);
                    const double lz = mx + log(se);
                    for (float& v : L) v = (float)((double)v - lz);
                    ok = fwrite(L.data(), 4, L.size(), fo) == L.size();
                }
                ok = fclose(fo) == 0 && ok;
                if (!ok) {
                    fprintf(stderr, "JOB %s %s: write to %s failed (%s): output removed\n", kind.c_str(), fn.c_str(), of.c_str(),
                            strerror(errno));
                    unlink(of.c_str());
                } else
                    fprintf(stderr, "JOB %s %s: %d positions (%s path) -> %s\n", kind.c_str(), fn.c_str(), K,
                            eager ? "eager" : "decode", of.c_str());
                dsv4::reset(E);
            } else if (kind == "tfgr") {   // review r14 M3 graph-path check: "tfgr IDS K": last K tokens teacher-forced
                // through decode() (step graph) and again through the eager forward_rows path from a fresh prefill;
                // with the frozen tier (--dyn-window -1) both must be bit-identical (R10: same kernels, same order)
                int K;
                jf >> K;
                if (!(K > 0 && K < (int)p.size())) {
                    fprintf(stderr, "JOB tfgr %s: K %d must be in 1 .. %zu: skipped\n", fn.c_str(), K, p.size() - 1);
                    continue;
                }
                std::vector<int> pre(p.begin(), p.end() - K), tail(p.end() - K, p.end());
                std::vector<float> A((size_t)K * dsv4::V), B((size_t)K * dsv4::V);
                dsv4::prefill(E, pre);
                for (int i = 0; i < K; i++) dsv4::decode_logits(E, tail[i], &A[(size_t)i * dsv4::V]);
                dsv4::reset(E);
                dsv4::prefill(E, pre);
                for (int i = 0; i < K; i++) dsv4::forward_rows(E, &tail[i], 1, &B[(size_t)i * dsv4::V]);
                const RowCmp r = cmp_rows(A.data(), B.data(), K);
                fprintf(stderr, "JOB tfgr %s K %d: graph vs eager %d of %d positions differ (first %d), NaN rows %d, argmax "
                        "differs %d, max |dlogit| %.3g%s\n", fn.c_str(), K, r.nbad, K, r.first, r.nnan, r.amd, r.worst,
                        r.nbad ? ": FAIL" : ": bit-identical");
                dsv4::reset(E);
            } else if (kind == "snapchk") {   // [dsv4-srv-pesnap] "snapchk IDS K": the snapshot at the end of IDS minus K
                // minus 1 (the server's prompt-end point), restored right after the prefill and again after its K + 1 rows
                // (rings, tails, compressed rows, Engram tokens past it rewritten); the K + 1 rows teacher-forced after each
                // restore must be bit-identical (frozen tier: --dyn-window -1)
                int K;
                jf >> K;
                if (!(K > 0 && K + 2 < (int)p.size())) { fprintf(stderr, "JOB snapchk %s: bad K\n", fn.c_str()); continue; }
                std::vector<int> pre(p.begin(), p.end() - K), tail(p.end() - K, p.end());
                const int P = (int)pre.size() - 1;
                dsv4::snap_mark(E, P);
                dsv4::prefill(E, pre);
                if (dsv4::snap_pos(E) != P) { fprintf(stderr, "JOB snapchk %s: no snapshot at %d: FAIL\n", fn.c_str(), P); dsv4::reset(E); continue; }
                std::vector<float> A((size_t)(K + 1) * dsv4::V), B((size_t)(K + 1) * dsv4::V);
                auto pass = [&](float* X) {
                    const int r = dsv4::snap_restore(E);
                    dsv4::forward_rows(E, &pre[P], 1, X);
                    for (int i = 0; i < K; i++) dsv4::forward_rows(E, &tail[i], 1, X + (size_t)(i + 1) * dsv4::V);
                    return r;
                };
                const int r1 = pass(A.data()), r2 = pass(B.data());
                const RowCmp r = cmp_rows(A.data(), B.data(), K + 1);
                fprintf(stderr, "JOB snapchk %s P %d (%s) K %d: restores %d / %d, after-generation vs immediate %d of %d rows "
                        "differ (first %d), NaN rows %d, argmax differs %d, max |dlogit| %.3g%s\n", fn.c_str(), P, P & 1 ? "odd" : "even",
                        K, r1, r2, r.nbad, K + 1, r.first, r.nnan, r.amd, r.worst, r.nbad || r1 != P || r2 != P ? ": FAIL" : ": bit-identical");
                dsv4::reset(E);
            } else if (kind == "trimchk") {   // [dsv4-srv-stop] "trimchk IDS G D K": greedy DSpark decode of >= G tokens,
                // trim(pos - D), then K rows of the generated sequence from there teacher-forced (1-row forward_rows); the
                // reference reaches the same position with capped passes (no trim) and runs the same K rows: bit-identical
                // when trim says it could (frozen tier: --dyn-window -1)
                int G, D, K;
                jf >> G >> D >> K;
                std::vector<int> fed(p);
                int t = dsv4::prefill(E, p);
                std::vector<int> got;
                while ((int)fed.size() - (int)p.size() < G && t != dsv4::EOS) {
                    got.clear();
                    dsv4::decode_spec(E, t, got, nullptr);
                    fed.push_back(t);
                    for (size_t i = 0; i + 1 < got.size(); i++) fed.push_back(got[i]);
                    t = got.back();
                }
                std::vector<int> full(fed);
                full.push_back(t);
                const int L = (int)fed.size() - D;
                if (!(D >= 0 && L > (int)p.size() && K >= 1 && L + K <= (int)full.size())) {
                    fprintf(stderr, "JOB trimchk %s: bad G / D / K (fed %zu)\n", fn.c_str(), fed.size());
                    dsv4::reset(E);
                    continue;
                }
                const int ok = dsv4::trim(E, L);
                std::vector<float> A((size_t)K * dsv4::V), B((size_t)K * dsv4::V);
                if (ok)
                    for (int i = 0; i < K; i++) dsv4::forward_rows(E, &full[L + i], 1, &A[(size_t)i * dsv4::V]);
                dsv4::reset(E);
                t = dsv4::prefill(E, p);
                int pos = (int)p.size(), bad = t != full[pos];
                while (pos < L && !bad) {
                    got.clear();
                    dsv4::decode_spec(E, t, got, nullptr, L - pos);
                    for (size_t i = 0; i < got.size(); i++) bad |= got[i] != full[pos + 1 + i];
                    pos += (int)got.size();
                    t = got.back();
                }
                if (ok && !bad)
                    for (int i = 0; i < K; i++) dsv4::forward_rows(E, &full[L + i], 1, &B[(size_t)i * dsv4::V]);
                const RowCmp r = cmp_rows(A.data(), B.data(), K);
                fprintf(stderr, "JOB trimchk %s G %d D %d (L %d, %s) K %d: trim %d, reference %s; %d of %d rows differ (first %d), "
                        "argmax differs %d, max |dlogit| %.3g%s\n", fn.c_str(), G, D, L, L & 1 ? "odd" : "even", K, ok,
                        bad ? "diverged" : "same tokens", r.nbad, K, r.first, r.amd, r.worst,
                        !ok ? ": trim declined" : bad || r.nbad ? ": FAIL" : ": bit-identical");
                dsv4::reset(E);
            } else if (kind == "tfcmp") {   // "tfcmp IDS K R": last K tokens one by one vs in passes of R rows
                int K, R;
                jf >> K >> R;
                std::vector<int> pre(p.begin(), p.end() - K), tail(p.end() - K, p.end());
                std::vector<float> A((size_t)K * dsv4::V), B((size_t)K * dsv4::V);
                dsv4::prefill(E, pre);
                for (int i = 0; i < K; i++) dsv4::forward_rows(E, &tail[i], 1, &A[(size_t)i * dsv4::V]);
                dsv4::reset(E);
                dsv4::prefill(E, pre);
                for (int i = 0; i < K; i += R) {
                    int m = std::min(R, K - i);
                    dsv4::forward_rows(E, &tail[i], m, &B[(size_t)i * dsv4::V]);
                }
                std::vector<float> C((size_t)K * dsv4::V);
                dsv4::reset(E);
                dsv4::prefill(E, pre);
                for (int i = 0; i < K; i++) dsv4::forward_rows(E, &tail[i], 1, &C[(size_t)i * dsv4::V]);
                auto cmp = [&](const std::vector<float>& X, const char* nm) {
                    const RowCmp r = cmp_rows(A.data(), X.data(), K);
                    fprintf(stderr, "JOB tfcmp %s %s K %d R %d: %d of %d positions differ (first %d), NaN rows %d, argmax differs "
                            "%d, max |dlogit| %.3g; first 12:%s\n", fn.c_str(), nm, K, R, r.nbad, K, r.first, r.nnan, r.amd,
                            r.worst, r.per.c_str());
                };
                cmp(C, "1-row again");
                cmp(B, "R-row passes");
            } else if (kind == "tfspd") {   // [dsv4-rel-tierreset] fixed-text speed: "tfspd IDS K R": prefill all but the last K
                // tokens, then the last K in passes of R rows (forward_rows, the dynamic tier fed as in decode); ms / token
                int K, R;
                jf >> K >> R;
                if (!(K > 0 && K < (int)p.size() && R >= 1 && R <= 8)) { fprintf(stderr, "JOB tfspd %s: bad K / R\n", fn.c_str()); continue; }
                std::vector<int> pre(p.begin(), p.end() - K), tail(p.end() - K, p.end());
                dsv4::prefill(E, pre);
                std::vector<float> lg((size_t)8 * dsv4::V);
                auto ta = clk::now();
                for (int i = 0; i < K; i += R) dsv4::forward_rows(E, &tail[i], std::min(R, K - i), lg.data());
                const double ms = since(ta) * 1e3;
                fprintf(stderr, "JOB tfspd %s K %d R %d: %.2f ms / token (%.2f tok/s)\n", fn.c_str(), K, R, ms / K, K * 1e3 / ms);
                static const bool trs2 = getenv("DSV4_TIERRESET") != nullptr;
                if (trs2) fprintf(stderr, "JOB tfspd %s tier reset %.1f ms\n", fn.c_str(), dsv4::tier_reset(E));
            } else if (kind == "srows") {   // [dsv4-rel-sample] "srows IDS N T1 .. TN OUT": prefill(IDS), then ONE pass of
                // the N rows T1 .. TN (as a verify pass), the N x V logits to OUT (the rows the speculative acceptance reads)
                int n;
                jf >> n;
                std::vector<int> rt(n);
                for (int& x : rt) jf >> x;
                std::string of;
                jf >> of;
                dsv4::prefill(E, p);
                std::vector<float> lg((size_t)n * dsv4::V);
                dsv4::forward_rows(E, rt.data(), n, lg.data());
                FILE* fo = fopen(of.c_str(), "wb");
                if (!fo) fprintf(stderr, "JOB srows %s: cannot open %s: %s\n", fn.c_str(), of.c_str(), strerror(errno));   // review rel2 L5
                else {
                    fwrite(lg.data(), 4, lg.size(), fo);
                    fclose(fo);
                    fprintf(stderr, "JOB srows %s %d rows -> %s\n", fn.c_str(), n, of.c_str());
                }
                dsv4::reset(E);
            } else if (kind == "sdist" || kind == "slogits") {
                // [dsv4-rel-sample] distribution gate: "sdist IDS FIXT N OUT": N times reset + prefill(IDS) + one
                // decode_spec(FIXT): the first emitted token (its law must be p(. | IDS, FIXT)) and the accepted count,
                // one line each; "slogits IDS FIXT OUT": the logits of that next token (decode_logits, V floats)
                int fixt, N = 1;
                std::string of;
                jf >> fixt;
                if (kind == "sdist") jf >> N;
                jf >> of;
                FILE* fo = fopen(of.c_str(), kind == "sdist" ? "w" : "wb");
                if (kind == "slogits") {
                    dsv4::prefill(E, p);
                    std::vector<float> lg(dsv4::V);
                    dsv4::decode_logits(E, fixt, lg.data());
                    fwrite(lg.data(), 4, lg.size(), fo);
                } else
                    for (int i = 0; i < N; i++) {
                        if (i) dsv4::reset(E);
                        dsv4::prefill(E, p);
                        std::vector<int> got;
                        dsv4::StepTimes tm;
                        dsv4::decode_spec(E, fixt, got, &tm);
                        fprintf(fo, "%d %d", got.empty() ? -1 : got[0], (int)got.size() - 1);
                        for (int x : got) fprintf(fo, " %d", x);   // the emitted run (accept-branch gate: later positions)
                        fprintf(fo, "\n");
                    }
                fclose(fo);
                fprintf(stderr, "JOB %s %s: -> %s\n", kind.c_str(), fn.c_str(), of.c_str());
            } else if (kind == "spec") {   // greedy with DSpark: "spec IDS N OUT"
                int N;
                std::string of;
                jf >> N >> of;
                std::vector<int> ov;
                auto t0 = clk::now();
                int t = dsv4::prefill(E, p);
                double tp = since(t0);
                ov.push_back(t);
                dsv4::StepTimes acc, byr[7];
                int hist[8] = {};
                int ncall = 0;
                static const bool speclog = getenv("DSV4_SPECLOG") != nullptr;
                std::vector<int> got;
                struct rusage ru0;
                getrusage(RUSAGE_SELF, &ru0);
                dsv4::lane_prof(E, true);   // R24: decode passes only
                auto t1 = clk::now();
                int ntok = 0;
                while ((int)ov.size() < N && t != dsv4::EOS && (int)p.size() + (int)ov.size() + 12 < o.maxctx) {
                    dsv4::StepTimes tm;
                    got.clear();
                    dsv4::decode_spec(E, t, got, &tm);
                    for (int x : got) {
                        if ((int)ov.size() >= N) break;
                        ov.push_back(x);
                        if (x == dsv4::EOS) break;
                    }
                    t = ov.back() == dsv4::EOS ? dsv4::EOS : got.back();
                    ntok += (int)got.size();
                    hist[std::min(7, tm.nacc)]++;
                    ncall++;
                    if (speclog)   // R17: one line per call
                        fprintf(stderr, "SPECCALL pos %zu rows %d acc %d plain %d ms %.2f route %.2f lane %.2f head %.2f draft %.2f "
                                "GB %.3f ucold %d enq %.2f bnd %.2f\n", p.size() + ov.size(), tm.rows, tm.nacc, tm.plain, tm.total, tm.wait_route,
                                tm.cpu, tm.head, tm.draft, tm.host_bytes / 1e9, tm.nucold, tm.enq, tm.bnd);
                    acc.total += tm.total; acc.wait_route += tm.wait_route; acc.cpu += tm.cpu; acc.head += tm.head;
                    acc.draft += tm.draft; acc.ncold += tm.ncold; acc.nhot += tm.nhot; acc.host_bytes += tm.host_bytes;
                    acc.lane_bytes += tm.lane_bytes; acc.nnat += tm.nnat; acc.commit += tm.commit; acc.bnd += tm.bnd; acc.engram += tm.engram;
                    acc.npass += tm.npass; acc.nacc += tm.nacc; acc.gpu_attn += tm.gpu_attn;
                    acc.npred += tm.npred; acc.npred_hit += tm.npred_hit; acc.npred_e += tm.npred_e; acc.npred_e_hit += tm.npred_e_hit;
                    acc.pf_mb += tm.pf_mb; acc.pf_wait += tm.pf_wait; acc.nucold += tm.nucold;
                    if (tm.npass == 1 && tm.rows >= 1 && tm.rows <= 6) {   // [dsv4-dec1-pfmr] per pass row count
                        dsv4::StepTimes& b = byr[tm.rows];
                        b.npass++; b.cpu += tm.cpu; b.lane_bytes += tm.lane_bytes; b.wait_route += tm.wait_route; b.total += tm.total;
                        b.npred += tm.npred; b.npred_hit += tm.npred_hit; b.npred_e += tm.npred_e; b.npred_e_hit += tm.npred_e_hit;
                        b.pf_mb += tm.pf_mb; b.pf_wait += tm.pf_wait; b.nucold += tm.nucold;
                        b.cv_cold += tm.cv_cold; b.cv_e += tm.cv_e; b.cv_p += tm.cv_p; b.cv_ep += tm.cv_ep; b.win_e += tm.win_e; b.win_p += tm.win_p;
                    }
                }
                if (getenv("DSV4_PFCOV"))   // [dsv4-dec1-pfmr]
                    for (int r = 1; r <= 6; r++) {
                        const dsv4::StepTimes& b = byr[r];
                        if (!b.npass) continue;
                        const double n = b.npass, nl = 40.0 * b.npass;
                        fprintf(stderr, "JOB spec %s pfcov R %d: passes %d ms %.2f route %.2f lane %.2f ms %.3f GB = %.1f GB/s; distinct cold/layer %.2f; "
                                "prefetched/layer %.2f used %.2f; early %.2f used %.2f; pf MB/layer %.1f; pred wait ms/pass %.2f; "
                                "cold union covered: early %.3f same %.3f either %.3f; window ms/layer early %.3f same %.3f\n",
                                fn.c_str(), r, b.npass, b.total / n, b.wait_route / n, b.cpu / n, b.lane_bytes / 1e9 / n,
                                b.cpu > 0 ? b.lane_bytes / 1e6 / b.cpu : 0.0, b.nucold / nl, b.npred / nl, b.npred_hit / nl, b.npred_e / nl,
                                b.npred_e_hit / nl, b.pf_mb / nl, b.pf_wait / n, b.cv_cold ? (double)b.cv_e / b.cv_cold : 0.0,
                                b.cv_cold ? (double)b.cv_p / b.cv_cold : 0.0, b.cv_cold ? (double)b.cv_ep / b.cv_cold : 0.0, b.win_e / nl, b.win_p / nl);
                    }
                double td = since(t1);
                FILE* f = fopen(of.c_str(), "w");
                for (size_t i = 0; i < ov.size(); i++) fprintf(f, "%s%d", i ? " " : "", ov[i]);
                fprintf(f, "\n");
                fclose(f);
                int np = std::max(1, acc.npass);
                fprintf(stderr, "JOB spec %s: prefill %zu tok %.3f s (net %.3f s); %zu tokens in %.2f s = %.2f tok/s; %d passes, %.3f tokens/pass "
                        "(accepted drafts hist 0..7: %d %d %d %d %d %d %d %d); per pass %.2f ms: route wait %.2f, CPU lane %.2f, head %.2f, "
                        "draft %.2f, rest %.2f; host GB/pass %.3f, GB/token %.3f; effective hit %.3f; calls %d\n",
                        fn.c_str(), p.size(), tp, tp - dsv4::instr_ms(E) / 1e3, ov.size(), td, ntok / td, acc.npass, (double)ntok / np, hist[0], hist[1],
                        hist[2], hist[3], hist[4], hist[5], hist[6], hist[7], acc.total / np, acc.wait_route / np, acc.cpu / np, acc.head / np,
                        acc.draft / np, (acc.total - acc.wait_route - acc.cpu - acc.head - acc.draft) / np,
                        acc.host_bytes / 1e9 / np, acc.host_bytes / 1e9 / std::max(1, ntok),
                        (double)acc.nhot / std::max(1, acc.nhot + acc.ncold), ncall);
                static const bool trs = getenv("DSV4_TIERRESET") != nullptr;   // [dsv4-rel-tierreset]: after each request
                if (trs) fprintf(stderr, "JOB spec %s tier reset %.1f ms\n", fn.c_str(), dsv4::tier_reset(E));
                {   // advisor H17 / idea 3: lane-only bytes and rate, admission bytes, file-read experts, major faults
                    struct rusage ru1;
                    getrusage(RUSAGE_SELF, &ru1);
                    fprintf(stderr, "JOB spec %s lane: lane GB/pass %.3f (%.1f GB/s over the lane ms), admission GB/pass %.3f, "
                            "file-read experts/pass %.2f, major faults/pass %.1f; tier host ms/pass: commit %.3f, boundary apply %.3f; "
                            "engram enqueue ms/pass %.3f\n", fn.c_str(), acc.lane_bytes / 1e9 / np,
                            acc.cpu > 0 ? acc.lane_bytes / 1e6 / acc.cpu : 0.0, (acc.host_bytes - acc.lane_bytes) / 1e9 / np,
                            (double)acc.nnat / np, (double)(ru1.ru_majflt - ru0.ru_majflt) / np, acc.commit / np, acc.bnd / np, acc.engram / np);
                    const std::string lp = dsv4::lane_prof(E, true);   // R24 [dsv4-r24-laneprof]
                    if (!lp.empty()) fprintf(stderr, "JOB spec %s laneprof: %s\n", fn.c_str(), lp.c_str());
                }
            } else if (kind == "gen") {
                int N;
                std::string of;
                jf >> N >> of;
                std::vector<int> ov;
                auto t0 = clk::now();
                int t = dsv4::prefill(E, p);
                double tp = since(t0);
                ov.push_back(t);
                dsv4::StepTimes acc;
                int steps = 0;
                for (int j = 1; j < N && t != dsv4::EOS && (int)p.size() + j < o.maxctx; j++) {
                    dsv4::StepTimes tm;
                    t = dsv4::decode(E, t, &tm);
                    ov.push_back(t);
                    if (j > 2) {
                        acc.total += tm.total; acc.wait_route += tm.wait_route; acc.cpu += tm.cpu; acc.head += tm.head;
                        acc.engram += tm.engram; acc.ncold += tm.ncold; acc.nhot += tm.nhot; acc.ndyn += tm.ndyn; acc.nadmit += tm.nadmit; acc.gpu_attn += tm.gpu_attn; acc.npred += tm.npred; acc.npred_hit += tm.npred_hit; acc.nprev_hit += tm.nprev_hit; acc.pf_wait += tm.pf_wait; acc.host_bytes += tm.host_bytes;
                        acc.npred_e += tm.npred_e; acc.npred_e_hit += tm.npred_e_hit; acc.pf_mb += tm.pf_mb;
                        steps++;
                    }
                }
                FILE* f = fopen(of.c_str(), "w");
                for (size_t i = 0; i < ov.size(); i++) fprintf(f, "%s%d", i ? " " : "", ov[i]);
                fprintf(f, "\n");
                fclose(f);
                if (steps)
                    fprintf(stderr, "JOB gen %s: prefill %zu tok %.3f s; %zu tokens; steady %d steps %.2f ms/step = %.2f tok/s; "
                            "split/step: route wait %.2f, CPU lane %.2f, head %.2f, engram host %.2f, rest %.2f ms; "
                            "cold %.2f hot %.2f (of which dynamic %.2f) admissions %.2f per token-layer; GPU attention half %.3f ms/layer; prefetched %.2f per token-layer, used %.2f; cold experts also used by the previous token %.2f; prediction wait %.2f ms/step; host GB/token %.3f, effective hit %.3f; early prefetched %.2f used %.2f; prefetch MB per token-layer %.1f\n",
                            fn.c_str(), p.size(), tp, ov.size(), steps, acc.total / steps, 1e3 * steps / acc.total,
                            acc.wait_route / steps, acc.cpu / steps, acc.head / steps, acc.engram / steps,
                            (acc.total - acc.wait_route - acc.cpu - acc.head) / steps, acc.ncold / (40.0 * steps),
                            acc.nhot / (40.0 * steps), acc.ndyn / (40.0 * steps), acc.nadmit / (40.0 * steps), acc.gpu_attn / (40.0 * steps), acc.npred / (40.0 * steps), acc.npred_hit / (40.0 * steps), acc.nprev_hit / (40.0 * steps), acc.pf_wait / steps, acc.host_bytes / 1e9 / steps, (double)acc.nhot / (acc.nhot + acc.ncold),
                            acc.npred_e / (40.0 * steps), acc.npred_e_hit / (40.0 * steps), acc.pf_mb / (40.0 * steps));
            }
        }
        dsv4::destroy(E);
        return 0;
    }
    std::vector<int> outv;
    if (ppl) {
        std::vector<float> nll;
        std::vector<int> am;
        auto t0 = clk::now();
        dsv4::prefill(E, ids, &nll, &am);
        double dt = since(t0);
        double s = 0;
        int ok = 0, n = (int)ids.size() - 1;
        for (int i = 0; i < n; i++) {
            s += nll[i];
            ok += am[i] == ids[i + 1];
        }
        fprintf(stderr, "ppl %.4f top1 %.4f over %d positions (%.1f s)\n", exp(s / n), (double)ok / n, n, dt);
        printf("ppl %.6f top1 %.6f n %d\n", exp(s / n), (double)ok / n, n);
    } else if (check > 0) {
        std::vector<int> pre(ids.begin(), ids.end() - check);
        int t = dsv4::prefill(E, pre);
        outv.push_back(t);
        dsv4::set_dump_step(E, 0);
        for (int j = 0; j < check; j++) outv.push_back(dsv4::decode(E, ids[ids.size() - check + j]));
    } else {
        auto t0 = clk::now();
        int t = dsv4::prefill(E, ids);
        double tp = since(t0);
        fprintf(stderr, "prefill %zu tokens: %.3f s = %.1f tok/s (TTFT)\n", ids.size(), tp, ids.size() / tp);
        outv.push_back(t);
        dsv4::StepTimes acc;
        auto t1 = clk::now();
        int steps = 0;
        for (int j = 1; j < gen && t != dsv4::EOS && (int)ids.size() + j < o.maxctx; j++) {
            dsv4::StepTimes tm;
            t = dsv4::decode(E, t, &tm);
            outv.push_back(t);
            if (j > 2) {   // skip the first steps (warm-up)
                acc.total += tm.total;
                acc.wait_route += tm.wait_route;
                acc.cpu += tm.cpu;
                acc.head += tm.head;
                acc.engram += tm.engram;
                acc.ncold += tm.ncold;
                acc.nhot += tm.nhot;
                steps++;
            }
        }
        double td = since(t1);
        if (steps) {
            fprintf(stderr, "decode %zu tokens in %.2f s; steady %d steps: %.2f ms/step = %.2f tok/s\n", outv.size() - 1,
                    td, steps, acc.total / steps, 1e3 * steps / acc.total);
            fprintf(stderr, "  split per step: wait for GPU route %.2f ms, CPU lane %.2f ms, head %.2f ms, engram host "
                    "%.2f ms, rest %.2f ms; cold experts %.2f / hot %.2f per token-layer\n",
                    acc.wait_route / steps, acc.cpu / steps, acc.head / steps, acc.engram / steps,
                    (acc.total - acc.wait_route - acc.cpu - acc.head) / steps, acc.ncold / (40.0 * steps),
                    acc.nhot / (40.0 * steps));
        }
    }
    FILE* f = out_file.empty() ? stdout : fopen(out_file.c_str(), "w");
    for (size_t i = 0; i < outv.size(); i++) fprintf(f, "%s%d", i ? " " : "", outv[i]);
    if (!outv.empty()) fprintf(f, "\n");
    if (f != stdout) fclose(f);
    dsv4::destroy(E);
    return 0;
}
