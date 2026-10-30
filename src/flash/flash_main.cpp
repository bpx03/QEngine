#include "../env.h"
#include <sys/mman.h>
#include <cuda_profiler_api.h>
// flash_run: load Qwen3.8-Flash-Next (qengine INT4 checkpoint) on 2 or 3 GPUs, run a prompt and greedy decode.
//   flash_run [--model DIR] [--dev 0,1,2] [--ctx 262144] [--w 176,160,176] --ids 1,2,3 [--gen 16] [--logits out.bin]
//   (--w: expert-group weights per GPU, default 176,160,176 on 3 GPUs, equal on 2)
#include <dirent.h>
#include <cstdint>

#include <algorithm>
#include <cmath>
#include <unistd.h>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#include <atomic>
#include <functional>
#include <vector>

#include "../gpu_guard.h"
#include "flash.h"
#include "flash_steer.h"
#include "../json.h"

using namespace flash;

static std::vector<std::string> split(const std::string& s, char c) {
    std::vector<std::string> v;
    std::string cur;
    for (char ch : s) {
        if (ch == c) { v.push_back(cur); cur.clear(); }
        else cur += ch;
    }
    if (!cur.empty()) v.push_back(cur);
    return v;
}

// 16 n-gram embedding rows (bigram heads 0..7, trigram heads 8..15) of token t of the history h -> out[2560]
static void ngram_row(const HostTables& ht, const std::vector<int>& h, int t, bf16* out) {
    // shifted tokens within the EOS-delimited segment (previous EOS excluded from the current position)
    // shifted token s is used iff no EOS sits in [t - s, t - 1] (the previous EOS ends the segment): only the two
    // previous positions decide (prefill-r4: the scan back to the last EOS was O(t) per token, O(n^2) per prompt:
    // 100K prompt 14 -> 60 ms of host time per 4K batch, serial with the GPU)
    long long sh[3];
    sh[0] = h[t];
    sh[1] = (t >= 1 && h[t - 1] != EOS) ? h[t - 1] : EOS;
    sh[2] = (t >= 2 && h[t - 1] != EOS && h[t - 2] != EOS) ? h[t - 2] : EOS;
    auto mul = [](long long a, long long b) { return (long long)((unsigned long long)a * (unsigned long long)b); };
    long long big = mul(sh[0], ht.mult[0]) ^ mul(sh[1], ht.mult[1]);
    long long tri = big ^ mul(sh[2], ht.mult[2]);
    for (int k = 0; k < PLE_HEADS; k++) {
        const long long mixed = k < 8 ? big : tri;
        long long r = mixed % ht.vsz[k];
        if (r < 0) r += ht.vsz[k];
        const long long row = r + ht.voff[k];
        ht.row(row, out + k * PLE_DIM);
    }
}

// rows t0 .. t0 + m - 1 of the n-gram embedding (PLE input) in parallel: host reads of the n-gram table, serial with
// the GPU between prefill batches (prefill-r4)
static void ngram_rows(const HostTables& ht, const std::vector<int>& h, int t0, int m, bf16* out) {
    const int nt = m >= 256 ? 8 : 1;
    auto part = [&](int k) {
        for (int j = (int)((long)m * k / nt); j < (int)((long)m * (k + 1) / nt); j++) ngram_row(ht, h, t0 + j, out + (size_t)j * H);
    };
    if (nt == 1) { part(0); return; }
    std::vector<std::thread> th;
    for (int k = 1; k < nt; k++) th.emplace_back(part, k);
    part(0);
    for (auto& x : th) x.join();
}

int main(int argc, char** argv) {
    std::string ids_file, cold_file = "default";
    std::string model = "models/Qwen3.8-Flash-Next-QE-INT4", ids_s, logits_path, devs_s, w_s;
    int gen = 16, ctx = 262144, kdraft = 0;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto nxt = [&]() { return std::string(argv[++i]); };
        if (a == "--model") model = nxt();
        else if (a == "--ids") {
            ids_s = nxt();
            if (!ids_s.empty() && ids_s[0] == '@') {  // --ids @file: comma-separated ids from a file (long prompts)
                FILE* f = fopen(ids_s.c_str() + 1, "r");
                if (!f) { fprintf(stderr, "cannot open %s\n", ids_s.c_str() + 1); return 1; }
                std::string all;
                char buf[65536];
                size_t n;
                while ((n = fread(buf, 1, sizeof(buf), f)) > 0) all.append(buf, n);
                fclose(f);
                while (!all.empty() && (all.back() == '\n' || all.back() == ' ')) all.pop_back();
                ids_s = all;
            }
        }
        else if (a == "--gen") gen = std::stoi(nxt());
        else if (a == "--ctx") ctx = std::stoi(nxt());
        else if (a == "--dev") devs_s = nxt();
        else if (a == "--w") w_s = nxt();
        else if (a == "--logits") logits_path = nxt();
        else if (a == "--k") kdraft = std::stoi(nxt());
        else if (a == "--ids-file") ids_file = nxt();
        else if (a == "--cold") cold_file = nxt();
    }
    std::vector<int> devs, prompt;
    if (devs_s.empty()) {  // default: every visible GPU (CUDA_VISIBLE_DEVICES), at most 3
        int n = 0;
        CUDA_CHECK(cudaGetDeviceCount(&n));
        for (int i = 0; i < std::min(n, 3); i++) devs_s += (i ? "," : "") + std::to_string(i);
    }
    for (auto& x : split(devs_s, ',')) devs.push_back(std::stoi(x));
    {  // one QEngine process per GPU (gpu_guard.h)
        const std::string busy = qe_gpu_guard::acquire(devs);
        if (!busy.empty()) { fprintf(stderr, "flash_run: %s\n", busy.c_str()); return 3; }
    }
    for (auto& x : split(ids_s, ',')) prompt.push_back(std::stoi(x));
    std::vector<double> wts;
    for (auto& x : split(w_s, ',')) wts.push_back(std::stod(x));
    if (wts.empty()) wts = default_group_weights((int)devs.size());
    std::vector<std::vector<int>> prompts_all;  // --ids-file: one prompt per line (plain greedy, for statistics)
    if (!ids_file.empty()) {
        FILE* f = fopen(ids_file.c_str(), "r");
        char* line = nullptr;
        size_t cap = 0;
        while (getline(&line, &cap, f) > 0) {
            std::vector<int> pv;
            for (auto& x : split(line, ',')) if (!x.empty() && x != "\n") pv.push_back(std::stoi(x));
            if (!pv.empty()) prompts_all.push_back(pv);
        }
        fclose(f);
        if (!prompts_all.empty()) prompt = prompts_all[0];
    }
    if (prompt.empty()) { fprintf(stderr, "--ids required\n"); return 1; }
    const int ng = devs.size();
    auto t0 = std::chrono::steady_clock::now();
    auto secs = [&]() { return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(); };

    SafeTensors st;
    {
        DIR* d = opendir(model.c_str());
        for (dirent* e; (e = readdir(d));) {
            std::string n = e->d_name;
            if (n.size() > 12 && n.substr(n.size() - 12) == ".safetensors" && n.rfind("mtp", 0) != 0) st.open(model + "/" + n);
        }
        closedir(d);
    }
    SafeTensors mst;  // MTP: BF16 dense parts + the INT4 experts
    bool use_mtp = kdraft > 0;
    if (use_mtp) {
        mst.open(model + "/mtp.safetensors");
        mst.open(model + "/mtp-int4.safetensors");
    }
    HostTables ht;
    // INT8 n-gram table when the model directory has it (FLASH_NGRAM=bf16 / int8 forces one)
    const char* ngv = env_str("FLASH_NGRAM");
    const bool ng8 = ngv ? !strcmp(ngv, "int8") : st.has("model.language_model.layers.1.ple.ple_embedding.ngram_q8.shard_0");
    {
        const TensorInfo& emb = st.get("model.language_model.embed_tokens.weight");
        ht.embed = (const bf16*)pin_host(emb.data, emb.nbytes);
        const std::string pe = "model.language_model.layers.1.ple.ple_embedding.";
        for (int i = 0; i < 128; i++) {
            if (ng8) ht.ngram8.push_back((const uint8_t*)st.get(pe + "ngram_q8.shard_" + std::to_string(i)).data);
            else ht.ngram.push_back((const bf16*)st.get(pe + "ngram_embedding.shard_" + std::to_string(i) + ".weight").data);
        }
        memcpy(ht.mult, st.get(pe + "layer_multipliers").data, 24);
        memcpy(ht.vsz, st.get(pe + "ngram_heads_vocab_sizes").data, 8 * PLE_HEADS);
        memcpy(ht.voff, st.get(pe + "ngram_heads_offsets").data, 8 * PLE_HEADS);
    }
    printf("[%.1fs] tensors mapped, embed pinned\n", secs());
    NgramDev ngd{};
    if (kdraft > 0) {  // n-gram table pinned + mapped for the device-side gather of the speculative step
        uintptr_t lo = ~(uintptr_t)0, hi = 0;
        auto tens = [&](int i) -> const TensorInfo& {
            const std::string pe = "model.language_model.layers.1.ple.ple_embedding.";
            return ng8 ? st.get(pe + "ngram_q8.shard_" + std::to_string(i)) : st.get(pe + "ngram_embedding.shard_" + std::to_string(i) + ".weight");
        };
        for (int i = 0; i < 128; i++) {
            const TensorInfo& t = tens(i);
            lo = std::min(lo, (uintptr_t)t.data);
            hi = std::max(hi, (uintptr_t)t.data + t.nbytes);
        }
        // file-backed mappings cannot be registered: copied into page-locked THP memory (pin_host)
        const size_t b = lo, e = hi;
        size_t foff = 0;
        const std::string fpath = st.file_of((const void*)b, &foff);
        const bool pinned = env_int("FLASH_NGRAM_PIN", 1);  // 0: file mapping (HMM)
        if (!pinned && (env_int("FLASH_NGRAM_RANDOM", 1)))  // random rows: no read-around
            madvise((void*)(b & ~(uintptr_t)4095), e - (b & ~(uintptr_t)4095), MADV_RANDOM);
        char* pin = !pinned ? (char*)b
                  : fpath.empty() ? pin_host((const void*)b, e - b, 32, true) : pin_host_file(fpath, foff, e - b, (const void*)b);
        for (int i = 0; i < 128; i++) {
            const char* h = pin + ((uintptr_t)tens(i).data - b);  // host gathers from the pinned copy
            void* dp = (void*)h;
            if (pinned) CUDA_CHECK(cudaHostGetDevicePointer(&dp, (void*)h, 0));
            if (ng8) { ngd.q8[i] = (const uint8_t*)dp; ht.ngram8[i] = (const uint8_t*)h; }
            else { ngd.shard[i] = (const bf16*)dp; ht.ngram[i] = (const bf16*)h; }
        }
        ngd.is8 = ng8;
        memcpy(ngd.mult, ht.mult, sizeof(ngd.mult));
        memcpy(ngd.vsz, ht.vsz, sizeof(ngd.vsz));
        memcpy(ngd.voff, ht.voff, sizeof(ngd.voff));
        printf("[%.1fs] n-gram table pinned (%.1f GB)\n", secs(), (e - b) / 1e9);
    }
    Comm comm = comm_create(ng, ARU_MAX * H, H);
    const bool p2p = comm_enable_p2p(comm, devs.data());
    printf("all-reduce: %s\n", p2p ? "P2P" : "host memory");
    comm_select_path(comm, devs.data(), "FLASH_COMM", true);
    auto plans = make_plan(ng, wts);
    std::vector<int> draft_ids;  // MTP draft vocabulary: the most frequent tokens (token_rank.bin)
    {
        const int nd = env_int("FLASH_DRAFT_V", 65536);
        FILE* f = fopen((model + "/token_rank.bin").c_str(), "rb");
        if (f && nd > 0) {
            draft_ids.resize(nd);
            draft_ids.resize(fread(draft_ids.data(), 4, nd, f));
            fclose(f);
        }
    }
    std::vector<std::vector<int>> cold;
    if (cold_file == "default") cold_file = model + "/cold_experts.json";
    if (cold_file != "none") {
        FILE* f = fopen(cold_file.c_str(), "r");
        if (f) {
            std::string js;
            char buf[65536];
            size_t n;
            while ((n = fread(buf, 1, sizeof(buf), f)) > 0) js.append(buf, n);
            fclose(f);
            Json j = Json::parse(js);
            for (auto& l : j["layers"].arr) {
                std::vector<int> v;
                for (auto& e : l.arr) v.push_back((int)e.i());
                cold.push_back(v);
            }
            printf("cold experts: %zu per layer in host memory (%s)\n", cold.empty() ? 0 : cold[0].size(), cold_file.c_str());
        }
    }
    std::vector<float> prior;  // expert-cache seed: per-token routing probabilities [48][512] (tools/flash_expert_prior.py)
    {
        const std::string pf = env_str("FLASH_EC_PRIOR") ? env_str("FLASH_EC_PRIOR") : model + "/expert_prior.bin";
        if (FILE* f = fopen(pf.c_str(), "rb")) {
            prior.resize((size_t)LAYERS * NE);
            if (fread(prior.data(), 4, prior.size(), f) != prior.size()) prior.clear();
            fclose(f);
        }
    }
    // a context above the native 262,144: YaRN with factor context / 262,144 (FLASH_YARN overrides)
    if (ctx > NATIVE_CTX) set_yarn_factor((double)ctx / NATIVE_CTX);
    std::vector<Worker*> W(ng);
    std::vector<int> nw(ng);
    {
        std::vector<std::thread> th;
        for (int r = 0; r < ng; r++)
            th.emplace_back([&, r] {
                W[r] = new Worker(plans[r], devs[r], comm, ctx);
                W[r]->set_cold(cold);
                W[r]->set_expert_prior(prior);
                W[r]->set_draft_vocab(draft_ids);
                W[r]->load(st, ht, use_mtp ? &mst : nullptr);
                nw[r] = W[r]->nallocs();
                W[r]->alloc_state();
            });
        for (auto& t : th) t.join();
    }
    for (int r = 0; r < ng; r++) {
        for (int g = 0; g < ng; g++) {
            W[r]->set_peer(g, W[g]->xrecv(), W[g]->xflags());
            W[r]->set_qpeer(g, W[g]->xqrecv(), W[g]->xqflags());
            W[r]->set_ipeer(g, W[g]->xirecv(), W[g]->xiflags());
            W[r]->set_agpeer(g, W[g]);
        }
        W[r]->set_ngram(ngd);
        if (kdraft > 0 && !env_int("FLASH_NGRAM_PIN", 1) && !(env_int("FLASH_NGRAM_HMM", 0)))
            W[r]->start_ngram_stage(&ht);
    }
    if ((env_str("QE_STEER") && *env_str("QE_STEER")) || (env_str("QE_STEER_DIR") && *env_str("QE_STEER_DIR"))) {  // flash_steer.h
        std::vector<SteerSpec> sv;
        int def = -1;
        try {
            sv = steer_load_env(H, LAYERS, model, &def);
        } catch (std::exception& ex) {
            fprintf(stderr, "flash_run: %s\n", ex.what());
            return 1;
        }
        const int nv = sv.size();
        std::vector<int> lo(nv), hi(nv);
        std::vector<float> w;
        for (int v = 0; v < nv; v++) {
            lo[v] = sv[v].flo; hi[v] = sv[v].fhi;
            w.insert(w.end(), sv[v].w.begin(), sv[v].w.end());
            printf("%s\n", sv[v].log_line().c_str());
        }
        for (int r = 0; r < ng; r++) W[r]->set_steer(nv, lo.data(), hi.data(), w.data());
        int vec = def;  // QE_STEER_VEC=<name> picks another one, QE_STEER_OFF=1: none
        if (const char* n = env_str("QE_STEER_VEC"))
            for (int v = 0; v < nv; v++) if (sv[v].name == n) vec = v;
        if (vec < 0 && nv == 1) vec = 0;
        if (env_int("QE_STEER_OFF", 0)) vec = -1;
        const float sc = env_double("QE_STEER_SCALE", 1.f);
        // QE_STEER_MODE=add|project, QE_STEER_LAYERS=lo:hi override the vector's defaults (as a request would)
        int smd = -1, slo = -1, shi = -1;
        if (const char* m = env_str("QE_STEER_MODE"); m && *m) smd = !strcmp(m, "project") ? 1 : !strcmp(m, "add") ? 0 : -9;
        if (const char* l = env_str("QE_STEER_LAYERS"); l && *l && sscanf(l, "%d:%d", &slo, &shi) != 2) slo = shi = -9;
        if (vec >= 0) {
            const std::string err = steer_resolve(sv[vec], smd, slo, shi);
            if (!err.empty()) { fprintf(stderr, "flash_run: %s\n", err.c_str()); return 1; }
        }
        for (int r = 0; r < ng; r++) W[r]->set_steer_slot(0, vec, smd, slo, shi, sc);
        printf("steering: run with %s, mode %s, layers %d..%d, scale %g\n", vec >= 0 ? sv[vec].name.c_str() : "off",
               smd == 1 ? "project" : "add", slo, shi, sc);
    } else if (const char* sy = env_str("FLASH_STEER_SYNTH")) {  // gate / debug: random unit directions, mode:lo:hi:alpha:seed
        auto f = split(sy, ':');
        const int mode = f.at(0) == "project" ? 1 : 0, lo = std::stoi(f.at(1)), hi = std::stoi(f.at(2));
        const float al = f.size() > 3 ? std::stof(f[3]) : 1.f;
        unsigned long long x = f.size() > 4 ? std::stoull(f[4]) : 1;
        std::vector<float> w((size_t)2 * LAYERS * H, 0.f);  // [2 modes][LAYERS][H]: the synthetic one in its mode
        for (int l = lo; l <= hi; l++) {
            double n2 = 0;
            for (int d = 0; d < H; d++) {
                x = x * 6364136223846793005ull + 1442695040888963407ull;
                const float u = ((x >> 40) & 0xffffff) / 16777216.f - 0.5f;
                w[((size_t)mode * LAYERS + l) * H + d] = u;
                n2 += (double)u * u;
            }
            const float sc = (mode ? sqrtf(al) : al) / (float)sqrt(n2);
            for (int d = 0; d < H; d++) w[((size_t)mode * LAYERS + l) * H + d] *= sc;
        }
        for (int r = 0; r < ng; r++) W[r]->set_steer(1, &lo, &hi, w.data());
        const bool off = env_int("QE_STEER_OFF", 0);
        for (int r = 0; r < ng; r++) W[r]->set_steer_slot(0, off ? -1 : 0, mode, lo, hi, 1.f);
        printf("steering (synthetic): mode=%s, layers %d..%d, alpha %g%s\n", mode ? "project" : "add", lo, hi, al, off ? ", off" : "");
    }
    printf("[%.1fs] loaded; context capacity %d tokens (QSA KV split by position)\n", secs(), ctx);
    for (int r = 0; r < ng; r++)
        printf("  GPU%d: allocated %.2f GB (+%.2f GB cold experts in host memory), free %.2f GiB\n", devs[r], W[r]->weight_gb(),
               W[r]->host_gb(), W[r]->free_bytes() / 1073741824.0);

    {
        const Worker::EcStats& es = W[0]->ec_stats();
        printf("expert cache: GPU0 %d cold groups, %d slots in the KV pool tail + big-batch buffers (%d of them borrowed tails; %d live, %d resident), seed H2D %.1f GB/s\n",
               es.ncold, es.slots, es.bt_slots, es.live_slots, es.resident, es.seed_gbs);
    }
    if (env_int("FLASH_GSUM", 0))
        for (int r = 0; r < ng; r++) {
            unsigned long long e = 0, o = 0;
            W[r]->gsum(nw[r], &e, &o);
            printf("GSUM rank %d experts %016llx other %016llx\n", r, e, o);
        }
    if (env_str("FLASH_CSUM"))
        for (int r = 0; r < ng; r++) printf("  GPU%d weights checksum %llx (%d allocations)\n", devs[r], W[r]->checksum(0, nw[r]), nw[r]);
    // persistent worker threads (spin on a step counter: no thread start per step)
    std::function<void(int)> job;
    std::atomic<int> go{0}, done{0};
    std::atomic<bool> quit{false};
    std::vector<std::thread> pool;
    for (int r = 1; r < ng; r++)
        pool.emplace_back([&, r] {
            int seen = 0;
            while (true) {
                int g;
                while ((g = go.load(std::memory_order_acquire)) == seen && !quit.load()) {}
                if (quit.load()) return;
                seen = g;
                job(r);
                done.fetch_add(1, std::memory_order_acq_rel);
            }
        });
    auto pool_run = [&]() {
        done.store(0);
        go.fetch_add(1, std::memory_order_acq_rel);
        job(0);
        while (done.load(std::memory_order_acquire) != ng - 1) {}
    };
    std::vector<int> hist;
    std::vector<bf16> ple((size_t)MAXM * H);
    std::vector<int> out(MAXM);
    const int nv0 = plans[0].voc1 - plans[0].voc0;
    std::vector<float> logits(nv0);
    std::vector<int> outs(MAXM);
    // trunk forward of m rows; n-gram rows from hist + toks (push: append toks to hist), commit: see Worker::forward
    auto step2 = [&](const std::vector<int>& toks, int pos, bool want_logits, bool push, int commit) {
        const int m = toks.size();
        std::vector<int> hh = hist;
        for (int i = 0; i < m; i++) {
            hh.push_back(toks[i]);
            ngram_row(ht, hh, (int)hh.size() - 1, ple.data() + (size_t)i * H);
        }
        if (push) hist = hh;
        job = [&, m, pos, want_logits, commit](int r) {
            std::vector<int> o(MAXM);
            W[r]->forward(toks.data(), pos, m, ple.data(), o.data(), (want_logits && r == 0) ? logits.data() : nullptr, commit);
            if (r == 0) outs = o;
        };
        pool_run();
        return outs[m - 1];
    };
    auto mtp = [&](const std::vector<int>& toks, int pos, bool chain) {
        const int m = toks.size();
        job = [&, m, pos, chain](int r) {
            std::vector<int> o(MAXM);
            W[r]->forward_mtp(toks.data(), pos, m, chain, o.data());
            if (r == 0) outs = o;
        };
        pool_run();
        return outs[m - 1];
    };
    auto step = [&](const std::vector<int>& toks, int pos, bool want_logits) {
        const int m = toks.size();
        for (int i = 0; i < m; i++) {
            hist.push_back(toks[i]);
            ngram_row(ht, hist, (int)hist.size() - 1, ple.data() + (size_t)i * H);
        }
        job = [&, m, pos, want_logits](int r) {
            std::vector<int> o(MAXM);
            W[r]->forward(toks.data(), pos, m, ple.data(), o.data(), (want_logits && r == 0) ? logits.data() : nullptr);
            if (r == 0) out = o;
        };
        pool_run();
        return out[m - 1];
    };
    if (env_str("FLASH_TF")) {
        // teacher-forced next-token log-probabilities: FLASH_TF = file with one sequence per line (comma-separated ids,
        // optionally "N:" in front = rows to evaluate, default FLASH_TF_N or 64): the first n - N tokens are prefilled
        // (rows committed at once), the last N run as decode forwards of <= 16 rows (deferred commit) and every row's
        // full-vocabulary log-softmax (fp32, vocabulary order) goes to FLASH_TF_OUT, sequence after sequence
        FILE* in = fopen(env_str("FLASH_TF"), "r");
        FILE* fo = fopen(env_str("FLASH_TF_OUT"), "wb");
        CHECK(in && fo, "FLASH_TF / FLASH_TF_OUT");
        const int defN = env_int("FLASH_TF_N", 64);
        std::vector<std::vector<int>> maps(ng);
        std::vector<std::vector<float>> lg(ng);
        for (int r = 0; r < ng; r++) {
            maps[r].resize(W[r]->vslice());
            W[r]->copy_lmap(maps[r].data());
            lg[r].resize((size_t)16 * W[r]->vslice());
        }
        std::vector<float> full(VOCAB);
        std::vector<double> lp(VOCAB);
        char* line = nullptr;
        size_t cap = 0;
        int nseq = 0;
        long long nrows = 0;
        const double t0 = secs();
        while (getline(&line, &cap, in) > 0) {
            std::string sl(line);
            int N = defN;
            const size_t colon = sl.find(':');
            if (colon != std::string::npos) { N = atoi(sl.substr(0, colon).c_str()); sl = sl.substr(colon + 1); }
            std::vector<int> ids;
            for (auto& x : split(sl, ',')) if (!x.empty() && x != "\n") ids.push_back(atoi(x.c_str()));
            const int n = ids.size();
            N = std::min(N, n - 1);
            for (int r = 0; r < ng; r++) W[r]->reset();
            hist.clear();
            const int np0 = n - N;
            static const int tfb = env_int("FLASH_TF_BATCH", 0);  // prefill_batch of up to tfb rows
            for (int i = 0; tfb > 0 && i < np0; i += tfb) {
                const int m = std::min(tfb, np0 - i);
                std::vector<int> ch(ids.begin() + i, ids.begin() + i + m);
                std::vector<bf16> rows((size_t)m * H);
                for (int j = 0; j < m; j++) {
                    hist.push_back(ch[j]);
                    ngram_row(ht, hist, (int)hist.size() - 1, rows.data() + (size_t)j * H);
                }
                job = [&](int r) { int oo; W[r]->prefill_batch(ch.data(), i, m, rows.data(), &oo, nullptr, false, nullptr); };
                pool_run();
            }
            for (int i = 0; tfb == 0 && i < np0; i += PFM) {  // prefill (rows committed)
                const int m = std::min(PFM, np0 - i);
                std::vector<int> ch(ids.begin() + i, ids.begin() + i + m);
                std::vector<bf16> rows((size_t)m * H);
                for (int j = 0; j < m; j++) {
                    hist.push_back(ch[j]);
                    ngram_row(ht, hist, (int)hist.size() - 1, rows.data() + (size_t)j * H);
                }
                job = [&](int r) { int oo; W[r]->prefill(ch.data(), i, m, rows.data(), &oo, nullptr, nullptr); };
                pool_run();
            }
            for (int i = np0; i < n; i += 16) {
                const int m = std::min(16, n - i);
                std::vector<int> ch(ids.begin() + i, ids.begin() + i + m);
                step2(ch, i, false, true, -1);
                for (int r = 0; r < ng; r++) W[r]->copy_logits(lg[r].data(), m);
                for (int j = 0; j < m; j++) {
                    for (int r = 0; r < ng; r++) {
                        const int nv = W[r]->vslice();
                        for (int q = 0; q < nv; q++) full[maps[r][q]] = lg[r][(size_t)j * nv + q];
                    }
                    double mx = -1e300, se = 0;
                    for (int v = 0; v < VOCAB; v++) mx = std::max(mx, (double)full[v]);
                    for (int v = 0; v < VOCAB; v++) se += exp((double)full[v] - mx);
                    const double lse = mx + log(se);
                    for (int v = 0; v < VOCAB; v++) full[v] = (float)((double)full[v] - lse);
                    fwrite(full.data(), 4, VOCAB, fo);
                    nrows++;
                }
            }
            nseq++;
        }
        fclose(fo);
        fclose(in);
        printf("teacher-forced: %d sequences, %lld rows in %.1f s\n", nseq, nrows, secs() - t0);
        fflush(stdout);
        _exit(0);  // the worker pool threads are not joined
    }
    if (env_str("FLASH_DET")) {  // determinism: the prompt twice, residuals of every layer compared (rank 0)
        const int m0 = std::min((int)prompt.size(), MAXP);
        std::vector<float> d1((size_t)LAYERS * m0 * RW), d2(d1.size());
        std::vector<int> ch(prompt.begin(), prompt.begin() + m0);
        for (int rep = 0; rep < 4; rep++) {
            for (int r = 0; r < ng; r++) W[r]->reset();
            hist.clear();
            W[0]->dump_ = rep == 0 ? d1.data() : d2.data();
            step(ch, 0, false);
            W[0]->dump_ = nullptr;
            if (rep == 0) continue;
            int first = -1;
            double md = 0;
            for (int l = 0; l < LAYERS && first < 0; l++)
                for (size_t i = 0; i < (size_t)m0 * RW; i++) {
                    const double d = fabs(d1[(size_t)l * m0 * RW + i] - d2[(size_t)l * m0 * RW + i]);
                    if (d > 0) { first = l; md = d; break; }
                }
            printf("rep %d: first differing layer %d (diff %.3g)\n", rep, first, md);
        }
        quit.store(true);
        for (auto& t : pool) t.join();
        return 0;
    }
    if (prompts_all.size() > 1) {
        double tt = 0;
        long ntok = 0;
        for (size_t pi = 0; pi < prompts_all.size(); pi++) {
            const auto& pr = prompts_all[pi];
            for (int r = 0; r < ng; r++) W[r]->reset();
            hist.clear();
            int t = 0;
            for (int i = 0; i < (int)pr.size(); i += MAXP) {
                std::vector<int> ch(pr.begin() + i, pr.begin() + std::min((int)pr.size(), i + MAXP));
                t = step(ch, i, false);
            }
            const double a0 = secs();
            for (int g = 1; g < gen && t != EOS && t != EOS + 2; g++) { t = step({t}, (int)hist.size(), false); ntok++; }
            tt += secs() - a0;
            printf("prompt %zu/%zu: %zu tokens\n", pi + 1, prompts_all.size(), pr.size());
        }
        printf("multi-prompt: %ld decode tokens, %.1f tok/s\n", ntok, ntok / tt);
        if (env_str("FLASH_STATS")) {
            std::vector<int> stv;
            W[0]->get_stats(stv);
            FILE* f = fopen(env_str("FLASH_STATS"), "wb");
            fwrite(stv.data(), 4, stv.size(), f);
            fclose(f);
        }
        quit.store(true);
        for (auto& t : pool) t.join();
        return 0;
    }
    const int nrep = env_set("FLASH_PFREP") ? std::max(1, env_int("FLASH_PFREP", 1)) : 1;  // timing: run the prompt N times
    const size_t hist0 = hist.size();
    double tp = 0;
    int nxt = 0;
    std::vector<int> drafts;
    const int np = prompt.size();
    std::vector<float> dumpbuf;
    int last_commit = -1;
    for (int rep = 0; rep < nrep; rep++) {
    hist.resize(hist0);
    drafts.clear();
    for (int r = 0; r < ng; r++) W[r]->reset();
    if (env_str("FLASH_DUMP")) {  // residual after every layer of the first prompt chunk (rank 0)
        dumpbuf.resize((size_t)LAYERS * MAXM * RW);
        W[0]->dump_ = dumpbuf.data();
    }
    if (env_str("FLASH_RDUMP")) W[0]->rdump_.assign((size_t)LAYERS * np * TOPK, -1);  // debug: prefill routing (FLASH_GRAPH=0)
    tp = secs();
    const bool pfp = env_int("FLASH_PF", 1);
    const bool pfb = pfp && !(env_int("FLASH_PFB", 1) == 0);
    if (pfb) {  // batched prefill: n <= pf_batch() rows per call (MoE half over the whole batch), then the MTP fill
        const int B = W[0]->pf_batch((long long)np + gen + 8);  // (rows past pfbn_mid_ only while the KV stays below the alias)
        std::vector<int> cuts;  // FLASH_RUN_CUTS=p,q,...: a batch also ends at these positions (the server's checkpoints)
        if (const char* c = env_str("FLASH_RUN_CUTS")) for (auto& x : split(c, ',')) if (!x.empty()) cuts.push_back(std::stoi(x));
        for (int i = 0, m = 0; i < np; i += m) {
            m = W[0]->pf_batch_rows(np - i, B);  // nominal batch B - PFM; a tail of <= PFM rows joins the last one
            for (int c : cuts) if (c > i && c < i + m) { m = c - i; break; }
            if (env_str("FLASH_RUN_CUTS") && env_str("FLASH_PFT")) printf("  batch [%d, %d)\n", i, i + m);
            std::vector<int> ch(prompt.begin() + i, prompt.begin() + i + m);
            std::vector<bf16> rows((size_t)m * H);
            const double tg0 = secs();
            for (int j = 0; j < m; j++) hist.push_back(ch[j]);
            ngram_rows(ht, hist, (int)hist.size() - m, m, rows.data());
            if (env_str("FLASH_PFT_HOST")) fprintf(stderr, "[pft] batch at %d: n-gram rows %.2f ms\n", i, (secs() - tg0) * 1e3);
            int o = 0, d = 0;
            const bool last = i + m >= np;
            std::vector<int> mt;
            if (use_mtp)
                for (int j = 0; j < m; j++) mt.push_back(i + j + 1 < np ? prompt[i + j + 1] : -1);
            job = [&](int r) {
                int oo, dd = -2;
                W[r]->prefill_batch(ch.data(), i, m, rows.data(), &oo, use_mtp ? mt.data() : nullptr, last, &dd);
                if (r == 0) { o = oo; d = dd; }
            };
            pool_run();
            if (use_mtp) {
                if (d == -2) {
                    if (last) mt[m - 1] = o;
                    job = [&](int r) { int dd; W[r]->prefill_batch_mtp(mt.data(), i, m, &dd); if (r == 0) d = dd; };
                    pool_run();
                }
                if (last) drafts = {d};
            }
            nxt = o;
            if (env_str("FLASH_PFT")) printf("  batch at %d (%d rows): %.1f ms\n", i, m, (secs() - tp) * 1e3), tp = secs();
            {  // FLASH_PROF_BATCHES=a,b,c,...: cudaProfilerStart before / Stop after each listed batch (nsys capture range)
                static std::vector<int> pb;
                static bool init = false;
                if (!init) { init = true; if (const char* e = env_str("FLASH_PROF_BATCHES")) for (auto& x : split(e, ',')) pb.push_back(std::stoi(x)); }
                static int bcount = 0;
                const int bi = bcount++;
                if (std::find(pb.begin(), pb.end(), bi) != pb.end()) { for (int r = 0; r < ng; r++) { cudaSetDevice(devs[r]); cudaDeviceSynchronize(); } cudaProfilerStop(); }
                if (std::find(pb.begin(), pb.end(), bi + 1) != pb.end()) { for (int r = 0; r < ng; r++) { cudaSetDevice(devs[r]); cudaDeviceSynchronize(); } cudaProfilerStart(); }
            }
        }
        last_commit = 0;
    } else if (pfp) {  // prefill path: chunks of PFM rows, MTP KV filled in the same call
        for (int i = 0; i < np; i += PFM) {
            const int m = std::min(PFM, np - i);
            std::vector<int> ch(prompt.begin() + i, prompt.begin() + i + m);
            std::vector<bf16> rows((size_t)m * H);
            for (int j = 0; j < m; j++) {
                hist.push_back(ch[j]);
                ngram_row(ht, hist, (int)hist.size() - 1, rows.data() + (size_t)j * H);
            }
            std::vector<int> mt;
            for (int j = 0; j < m; j++) mt.push_back(i + j + 1 < np ? prompt[i + j + 1] : -1);
            const bool last = i + m >= np;
            // the last row's MTP token is the first generated one: run the trunk first, then the MTP with it
            int o = 0, d = 0;
            if (!last || !use_mtp) {
                job = [&](int r) { int oo, dd; W[r]->prefill(ch.data(), i, m, rows.data(), &oo, use_mtp ? mt.data() : nullptr, &dd); if (r == 0) { o = oo; d = dd; } };
                pool_run();
            } else {
                job = [&](int r) { int oo; W[r]->prefill(ch.data(), i, m, rows.data(), &oo, nullptr, nullptr); if (r == 0) o = oo; };
                pool_run();
                mt[m - 1] = o;
                // MTP over the chunk rows: the host-path MTP fill (hidden rows = the trunk residual of this chunk)
                job = [&](int r) { int dd; W[r]->prefill_mtp(mt.data(), i, m, &dd); if (r == 0) d = dd; };
                pool_run();
            }
            nxt = o;
            if (last && use_mtp) drafts = {d};
            if (env_str("FLASH_PFT")) printf("  chunk %d (%d rows): %.1f ms\n", i / PFM, m, (secs() - tp) * 1e3), tp = secs();
        }
        last_commit = 0;
    }
    for (int i = pfp ? np : 0; i < np; i += MAXP) {
        std::vector<int> ch(prompt.begin() + i, prompt.begin() + std::min(np, i + MAXP));
        nxt = step2(ch, i, i + MAXP >= np, true, -1);
        if (use_mtp) {  // MTP KV over the prompt: row j = (h_{i+j}, t_{i+j+1})
            std::vector<int> mt;
            for (int j = 0; j < (int)ch.size(); j++) mt.push_back(i + j + 1 < np ? prompt[i + j + 1] : nxt);
            const int d = mtp(mt, i, false);
            if (i + MAXP >= np) drafts = {d};
        }
    }
    if (use_mtp)
        for (int j = 1; j < kdraft; j++) drafts.push_back(mtp({drafts.back()}, np + j - 1, true));
    if (!W[0]->rdump_.empty()) {  // [48][np][10] int32
        FILE* f = fopen(env_str("FLASH_RDUMP"), "wb");
        fwrite(W[0]->rdump_.data(), 4, W[0]->rdump_.size(), f);
        fclose(f);
        W[0]->rdump_.clear();
    }
    tp = secs() - tp;
    if (nrep > 1) printf("prefill run %d: %d tokens in %.1f ms; next %d\n", rep, np, tp * 1e3, nxt);
    if (W[0]->dbgh_) {  // determinism check: per layer hashes of this repetition vs the first one (every card)
        static std::vector<std::vector<unsigned long long>> h0(ng);
        static const char* kind[12] = {"R after mixer", "routing", "MoE partial", "x (MoE input)", "x (QSA input)", "qkv proj",
                                       "indexer proj", "q/k prep", "block selection", "attention partial", "attention combined", "o proj partial"};
        for (int r = 0; r < ng; r++) {
            std::vector<unsigned long long> h;
            W[r]->dbg_hashes(h);
            if (rep == 0) {
                h0[r] = h;
                if (const char* o = env_str("FLASH_DBGHASH_OUT")) {  // cross-process comparison
                    FILE* f = fopen((std::string(o) + "." + std::to_string(r)).c_str(), "wb");
                    fwrite(h.data(), 8, h.size(), f);
                    fclose(f);
                }
                continue;
            }
            int fl = -1, fk = -1, nd = 0;
            const int K = Worker::DBGK;
            for (int l = 0; l <= LAYERS; l++)
                for (int k : {4, 5, 6, 7, 8, 9, 10, 11, 0, 3, 1, 2})  // in execution order within a layer
                    if (h[l * K + k] != h0[r][l * K + k]) { nd++; if (fl < 0) { fl = l; fk = k; } }
            printf("rep %d rank %d: %s (first differing: layer %d, %s; %d of %d hashes differ), next token %d\n", rep, r,
                   fl < 0 ? "IDENTICAL" : "differs", fl, fk >= 0 ? kind[fk] : "-", nd, (LAYERS + 1) * Worker::DBGK, nxt);
        }
    }
    for (int r = 0; r < ng; r++) W[r]->dbg_xqrep(rep);
    if (W[0]->rowh_) {  // FLASH_DBGHASH=2: per (layer, row) hashes vs the first repetition: first differing per kind
        static std::vector<std::vector<unsigned long long>> r0(ng);
        static std::vector<std::vector<double>> l0(ng);
        static const char* kn[4] = {"xh before MoE", "xq after MoE (in place)", "MoE partial", "routing"};
        for (int r = 0; r < ng; r++) {
            std::vector<unsigned long long> h;
            W[r]->dbg_rowh(h);
            std::vector<double> l1;
            W[r]->dbg_rowl1(l1);
            if (rep == 0) { r0[r] = h; l0[r] = l1; continue; }
            const int RN = W[r]->rowh_n_;
            for (int l = 0, done = 0; l < LAYERS && !done; l++)  // MoE partial L1 of the first differing rows
                for (int row = 0; row < RN; row++)
                    if (h[((size_t)l * RN + row) * 4 + 2] != r0[r][((size_t)l * RN + row) * 4 + 2]) {
                        for (int q = std::max(0, row - 2); q < std::min(RN, row + 10); q++)
                            printf("rowl1 rep %d rank %d layer %d row %d: %.6f vs %.6f (rel %.2e)%s\n", rep, r, l, q, l1[(size_t)l * RN + q],
                                   l0[r][(size_t)l * RN + q], fabs(l1[(size_t)l * RN + q] - l0[r][(size_t)l * RN + q]) / (l0[r][(size_t)l * RN + q] + 1e-30),
                                   h[((size_t)l * RN + q) * 4 + 2] != r0[r][((size_t)l * RN + q) * 4 + 2] ? " *" : "");
                        done = 1;
                        break;
                    }
            for (int k = 0; k < 4; k++) {
                int fl = -1, fr = -1, nd = 0, nl = 0, lastl = -1;
                for (int l = 0; l < LAYERS; l++)
                    for (int row = 0; row < RN; row++)
                        if (h[((size_t)l * RN + row) * 4 + k] != r0[r][((size_t)l * RN + row) * 4 + k]) {
                            if (fl < 0) { fl = l; fr = row; }
                            if (l == fl) { nd++; if (lastl < 0) lastl = row; nl = row; }
                        }
                if (fl >= 0) printf("rowh rep %d rank %d %s: first layer %d, rows %d..%d (%d rows differ there)\n", rep, r, kn[k], fl, fr, nl, nd);
            }
        }
    }
    if (nrep > 1 && env_str("FLASH_LOGITS_NEXT")) {  // check every repetition (graph replays)
        // FLASH_LOGITS_NEXT_K=k: the same forward k times after this prefill (decode determinism on a fixed state)
        const int kn = env_set("FLASH_LOGITS_NEXT_K") ? std::max(1, env_int("FLASH_LOGITS_NEXT_K", 1)) : 1;
        for (int q = 0; q < kn; q++) {
            step2({nxt}, np, true, false, 0);
            FILE* f = fopen((std::string(env_str("FLASH_LOGITS_NEXT")) + "." + std::to_string(rep) + (kn > 1 ? "." + std::to_string(q) : "")).c_str(), "wb");
            fwrite(logits.data(), 4, nv0, f);
            fclose(f);
        }
    }
    }
    if (W[0]->dump_) {
        W[0]->dump_ = nullptr;
        const int m0 = std::min((int)prompt.size(), MAXP);
        FILE* f = fopen(env_str("FLASH_DUMP"), "wb");
        fwrite(dumpbuf.data(), 4, (size_t)LAYERS * m0 * RW, f);
        fclose(f);
    }
    printf("prompt %zu tokens in %.3f s; next %d\n", prompt.size(), tp, nxt);
    if (env_int("FLASH_XBYTES", 0)) for (int r = 0; r < ng; r++) W[r]->xb_report(prompt.size() * std::max(1, nrep));
    if (env_str("FLASH_LOGITS_NEXT")) {  // check: logits (GPU0 vocab slice) of the forward of the first generated token
        step2({nxt}, np, true, true, 0);
        FILE* f = fopen(env_str("FLASH_LOGITS_NEXT"), "wb");
        fwrite(logits.data(), 4, nv0, f);
        fclose(f);
        quit.store(true);
        for (auto& t : pool) t.join();
        return 0;
    }
    {
        std::vector<std::pair<float, int>> top;
        for (int i = 0; i < nv0; i++) top.push_back({logits[i], i + plans[0].voc0});
        std::partial_sort(top.begin(), top.begin() + 5, top.end(), [](auto& a, auto& b) { return a.first > b.first; });
        printf("GPU0 vocab slice top5:");
        for (int i = 0; i < 5; i++) printf(" %d (%.3f)", top[i].second, top[i].first);
        printf("\n");
        if (!logits_path.empty()) {
            FILE* f = fopen(logits_path.c_str(), "wb");
            fwrite(logits.data(), 4, nv0, f);
            fclose(f);
        }
    }
    for (int r = 0; r < ng; r++)
        if (W[r]->nf_check("prefill", np)) {
            fprintf(stderr, "[flash] FATAL: non-finite values in the prefill\n");
            exit(1);
        }
    printf("prefill %s\n", ec_summary(W).c_str());
    for (int r = 0; r < ng; r++) W[r]->ec_reset_stats();
    std::vector<int> gen_toks = {nxt};
    std::vector<double> dt;
    if (env_set("FLASH_TEMP"))  // sampling (temperature, top-p 0.8, top-k 20 as the serving benchmark)
        for (int r = 0; r < ng; r++) W[r]->set_sampling(env_double("FLASH_TEMP", 0.0), 0.8f, 20, 0.f, 1234);
    if (use_mtp && !env_str("FLASH_HOSTSPEC")) {  // device-side speculative steps (one graph per step)
        const int last_chunk = np - (np - 1) / MAXP * MAXP;
        for (int r = 0; r < ng; r++)
            W[r]->spec_init(nxt, drafts.data(), kdraft, np, last_commit >= 0 ? last_commit : last_chunk, prompt[np - 1], np >= 2 ? prompt[np - 2] : EOS);
        int steps = 0, lk_steps = 0, lk_rows = 0, lk_tok = 0, Lnext = 0;
        bool prev_lk = false;
        if (W[0]->lookup_on()) {  // device lookup drafts: history = prompt + the pending token
            std::vector<int> h(prompt.begin(), prompt.begin() + np);
            h.push_back(nxt);
            if (const char* cf = env_str("FLASH_CORPUS_FILE")) {  // lookup corpus: one answer (comma-separated ids) per line
                std::vector<std::vector<int>> ans;
                if (FILE* f = fopen(cf, "r")) {
                    char* line = nullptr;
                    size_t cap = 0;
                    while (getline(&line, &cap, f) > 0) {
                        std::vector<int> a;
                        for (auto& x : split(line, ',')) if (!x.empty() && x != "\n") a.push_back(std::stoi(x));
                        if (!a.empty()) ans.push_back(a);
                    }
                    fclose(f);
                }
                size_t nt = 0;
                for (auto& a : ans) nt += a.size();
                job = [&](int r) { for (auto& a : ans) W[r]->corpus_add(a.data(), (int)a.size()); };
                pool_run();
                printf("lookup corpus: %zu answers, %zu tokens\n", ans.size(), nt);
            }
            job = [&](int r) { W[r]->set_history(h.data(), (int)h.size(), kdraft); };
            pool_run();
            Lnext = W[0]->lookup_drafts();
        }
        std::vector<int> em(MAXM + 1);
        const double t0s = secs();
        while ((int)gen_toks.size() < gen) {
            const double a0 = secs();
            const bool lk = Lnext > 0;
            const int ks = lk ? Lnext : kdraft;  // after a lookup step the chain ran on the device when needed
            static const bool xrank = env_int("FLASH_XRANK", 0);  // debug: per-rank emitted tokens
            static const int ecv = env_int("FLASH_EC_VERIFY", 0);
            std::vector<std::vector<int>> per(ng);
            job = [&](int r) {
                std::vector<int> e(MAXM + 1);
                const int n = W[r]->spec_step(ks, e.data(), lk);
                e[MAXM] = n;
                per[r].assign(e.begin(), e.begin() + n);
                per[r].push_back(n);
                if (ecv >= 2) { char w[40]; snprintf(w, sizeof(w), "after spec step %d", steps); W[r]->ec_verify(w); }
                if (r == 0) em = e;
            };
            pool_run();
            if (xrank)
                for (int r = 1; r < ng; r++)
                    if (per[r] != per[0]) {
                        printf("XRANK MISMATCH at step %d: rank 0 emitted", steps);
                        for (int t : per[0]) printf(" %d", t);
                        printf(" | rank %d emitted", r);
                        for (int t : per[r]) printf(" %d", t);
                        printf("\n");
                        fflush(stdout);
                    }
            for (int i = 0; i < em[MAXM]; i++) gen_toks.push_back(em[i]);
            dt.push_back(secs() - a0);
            steps++;
            if (lk) { lk_steps++; lk_rows += ks; lk_tok += em[MAXM]; }
            prev_lk = lk;
            static const bool lklog = env_int("FLASH_LK_LOG", 0);
            if (lklog)  // per step: lookup?, rows - 1, emitted, GPU ms, then the next step's match length and drafts
                printf("lkstep %d %d %d %.3f %d %d\n", (int)lk, ks, em[MAXM], W[0]->gpu_ms_.empty() ? 0.f : W[0]->gpu_ms_.back(),
                       W[0]->lookup_match(), W[0]->lookup_drafts());
            Lnext = W[0]->lookup_drafts();
        }
        if (W[0]->lookup_on())
            printf("lookup: %d of %d steps (%d draft rows, %.2f tokens/lookup step)\n", lk_steps, steps, lk_rows,
                   lk_steps ? (double)lk_tok / lk_steps : 0.0);
        const double el = secs() - t0s;
        unsigned ch = 0;
        for (int r = 0; r < ng; r++) ch += W[r]->cold_hits();
        {
            std::vector<float> g = W[0]->gpu_ms_;
            std::sort(g.begin(), g.end());
            double sm = 0;
            for (float x : g) sm += x;
            if (!g.empty())
                printf("GPU step time (rank 0 graph): mean %.2f ms, median %.2f, p90 %.2f -> %.1f tok/s at GPU time\n", sm / g.size(),
                       g[g.size() / 2], g[g.size() * 9 / 10], (gen_toks.size() - 1) / (sm / 1e3));
        }
        printf("MTP k=%d (device loop): %d steps, %.2f tokens/step, %.1f tok/s; cold group hits %.2f/step (all cards, incl. prompt)\n",
               kdraft, steps, (double)(gen_toks.size() - 1) / steps, (gen_toks.size() - 1) / el, (double)ch / steps);
        gen = 0;
    } else if (use_mtp) {
        int pos = np, steps = 0, acc_tot = 0, commit = -1;
        const double t0s = secs();
        while ((int)gen_toks.size() < gen) {
            std::vector<int> v = {nxt};
            v.insert(v.end(), drafts.begin(), drafts.end());
            const double a0 = secs();
            step2(v, pos, false, false, commit);
            std::vector<int> o(outs.begin(), outs.begin() + v.size());
            int a = 0;
            while (a < kdraft && v[a + 1] == o[a]) a++;
            for (int j = 0; j <= a; j++) hist.push_back(v[j]);
            for (int j = 0; j <= a; j++) gen_toks.push_back(o[j]);
            // drafts for the next step: MTP over the accepted rows, then the chain
            std::vector<int> mt(o.begin(), o.begin() + a + 1);
            drafts = {mtp(mt, pos, false)};
            for (int j = 1; j < kdraft; j++) drafts.push_back(mtp({drafts.back()}, pos + a + j, true));
            dt.push_back(secs() - a0);
            nxt = o[a];
            pos += a + 1;
            commit = a + 1;
            steps++;
            acc_tot += a;
        }
        const double el = secs() - t0s;
        printf("MTP k=%d: %d steps, %.2f tokens/step (accepted drafts %.2f), %.1f tok/s\n", kdraft, steps,
               (double)(gen_toks.size() - 1) / steps, (double)acc_tot / steps, (gen_toks.size() - 1) / el);
        gen = 0;
    }
    for (int g = 1; g < gen; g++) {
        const double a = secs();
        nxt = step({nxt}, (int)hist.size(), false);
        dt.push_back(secs() - a);
        gen_toks.push_back(nxt);
    }
    printf("generated:");
    for (int t : gen_toks) printf(" %d", t);
    printf("\n");
    if (!dt.empty()) {
        std::sort(dt.begin(), dt.end());
        double mean = 0;
        for (double x : dt) mean += x;
        mean /= dt.size();
        printf("decode step (host-timed, CUDA graph per step): median %.2f ms, mean %.2f, p10 %.2f, p90 %.2f, min %.2f ms\n",
               dt[dt.size() / 2] * 1e3, mean * 1e3, dt[dt.size() / 10] * 1e3, dt[dt.size() * 9 / 10] * 1e3, dt[0] * 1e3);
    }
    printf("decode %s\n", ec_summary(W).c_str());
    for (auto* w : W) { w->selwin_report(); w->kvs_report(); }
    if (env_int("FLASH_EC_VERIFY", 0))
        for (int r = 0; r < ng; r++) W[r]->ec_verify("after decode");
    if (env_int("FLASH_JOBCHK", 0)) {
        for (int r = 0; r < ng; r++) { CUDA_CHECK(cudaSetDevice(devs[r])); CUDA_CHECK(cudaDeviceSynchronize()); W[r]->jobchk_report(); }
    }
    if (env_str("FLASH_EC_STEPS")) {  // per decode step: misses (all cards) and host-timed step ms
        FILE* f = fopen(env_str("FLASH_EC_STEPS"), "w");
        std::vector<int> per;
        for (int r = 0; r < ng; r++) {
            const auto& sm = W[r]->ec_stats().step_miss;
            if (per.size() < sm.size()) per.resize(sm.size(), 0);
            for (size_t i = 0; i < sm.size(); i++) per[i] += sm[i];
        }
        const std::vector<float>& g = W[0]->gpu_ms_;
        for (size_t i = 0; i < per.size(); i++) fprintf(f, "%d %.3f\n", per[i], i < g.size() ? g[i] : 0.f);
        fclose(f);
    }
    if (env_str("FLASH_STATS")) {  // append routing counts ([49][512] int32) to the file
        std::vector<int> stv;
        W[0]->get_stats(stv);
        FILE* f = fopen(env_str("FLASH_STATS"), "ab");
        fwrite(stv.data(), 4, stv.size(), f);
        fclose(f);
    }
    quit.store(true);
    for (auto& t : pool) t.join();
    printf("ids:");
    for (size_t i = 0; i < prompt.size(); i++) printf("%s%d", i ? "," : "", prompt[i]);
    for (int t : gen_toks) printf(",%d", t);
    printf("\n");
    return 0;
}
