#include "env.h"
#include "engine.h"

#include <sys/mman.h>
#include "snap_policy.h"

#include <chrono>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <tuple>

#include <dirent.h>
#include <fcntl.h>
#include <limits.h>
#include <sys/stat.h>
#include <unistd.h>

GpuPool::GpuPool(const std::vector<int>& devices) : devices_(devices) {
    for (size_t i = 0; i < devices.size(); i++) threads_.emplace_back([this, i] { loop(i); });
}

GpuPool::~GpuPool() {
    {
        std::lock_guard<std::mutex> l(mu_);
        stop_ = true;
        gen_++;
    }
    cv_.notify_all();
    for (auto& t : threads_) t.join();
}

void GpuPool::loop(int rank) {
    CUDA_CHECK(cudaSetDevice(devices_[rank]));
    unsigned long seen = 0;
    while (true) {
        std::function<void(int)> job;
        {
            std::unique_lock<std::mutex> l(mu_);
            cv_.wait(l, [&] { return gen_ != seen; });
            seen = gen_;
            if (stop_) return;
            job = job_;
        }
        job(rank);
        {
            std::lock_guard<std::mutex> l(mu_);
            busy_ &= ~(1u << rank);
            if (--pending_ == 0) done_cv_.notify_all();
        }
    }
}

void GpuPool::run(const std::function<void(int)>& job) {
    std::unique_lock<std::mutex> l(mu_);
    job_ = job;
    pending_ = threads_.size();
    busy_ = (1u << threads_.size()) - 1;
    gen_++;
    cv_.notify_all();
    // host-side diagnostics only (no kill: a job may legitimately be long, e.g. pinning the archive);
    // stuck device waits are ended by the collective watchdog (kernels/watchdog.cuh)
    for (int k = 1; !done_cv_.wait_for(l, std::chrono::seconds(60), [&] { return pending_ == 0; }); k++) {
        fprintf(stderr, "qengine: GPU job running for %d s, still busy:", 60 * k);
        for (size_t r = 0; r < threads_.size(); r++)
            if (busy_ & (1u << r)) fprintf(stderr, " GPU%d (rank %zu)", devices_[r], r);
        fprintf(stderr, "\n");
    }
}

Engine::Engine(const std::string& model_dir, const std::vector<int>& devices, int max_t,
               const std::vector<double>& split_weights) {
    auto t0 = std::chrono::steady_clock::now();
    model_dir_ = model_dir;
    cfg_ = Config::load(model_dir);
    // QE_STEER / QE_STEER_DIR (src/steer.h): validated first; a refused vector throws (qe_create returns null)
    steer_ = qsteer::steer_load_env(cfg_.hidden, cfg_.layers, model_dir, &steer_def_);
    if (steer_.size() > 4096) throw std::runtime_error("steering: more than 4096 vectors");
    if (!steer_.empty() && cfg_.layers > 64) throw std::runtime_error("steering: the cache key holds 6 bits per layer index");
    std::vector<double> weights = split_weights;
    if (weights.empty())
        if (const char* e = env_str("QE_SPLIT")) {  // relative per-GPU work, e.g. "1,1.04,0.98"
            std::stringstream ss(e);
            for (std::string x; std::getline(ss, x, ',');) weights.push_back(std::stod(x));
        }
    // QE_KV_SPLIT: position split of the kv heads (config.h KvSplit).
    // Unset / "auto" (default): GPU0's share of kv1 and GPU2's share of kv2 are chosen in plan_slots
    // from the free memory of the cards after the weights (every card's pool fills); "x,y": those
    // shares (0 < x, y < 1); "0": the old layout (kv1 / kv2 duplicated on two cards).
    // QE_KV_PERIOD: positions per ownership period (default 1024). Needs P2P between the cards
    // and 3 GPUs with 4 kv heads (else the old layout).
    {
        const char* e = env_str("QE_KV_SPLIT");
        if (env_set("QE_KV_PERIOD")) kvs_.period = std::max(64, env_int("QE_KV_PERIOD", kvs_.period) / 32 * 32);
        const bool can = devices.size() == 3 && cfg_.n_kv == 4;
        if (!e || !strcmp(e, "auto")) {
            kvs_auto_ = can;
            if (can) kvs_.a01 = kvs_.a12 = kvs_.to_a(0.4);  // placeholder until plan_slots
            else if (e) fprintf(stderr, "QE_KV_SPLIT: needs 3 GPUs and 4 kv heads, old layout\n");
        } else if (atof(e) > 0) {
            double x = 0, y = 0;
            if (sscanf(e, "%lf,%lf", &x, &y) < 2) y = x;
            if (can) {
                kvs_.a01 = kvs_.to_a(x);
                kvs_.a12 = kvs_.to_a(y);
            } else {
                fprintf(stderr, "QE_KV_SPLIT: needs 3 GPUs and 4 kv heads, old layout\n");
            }
        }
    }
    {   // the stats exchange lives in the batched fp8 attention path (one flag per launch): bf16
        // KV or a bf16 MTP cache (QE_KV_FP8=0 / QE_MTP_FP8=0) go through attention() per sequence
        const bool kv8 = env_int("QE_KV_FP8", 1) != 0;
        const bool mtp8 = kv8 && (env_int("QE_MTP_FP8", 1) != 0);
        if ((kvs_.on() || kvs_auto_) && !(kv8 && mtp8)) {
            fprintf(stderr, "QE_KV_SPLIT: needs fp8 KV and MTP caches (QE_KV_FP8=1, QE_MTP_FP8=1), old layout\n");
            kvs_ = KvSplit{kvs_.period, 0, 0};
            kvs_auto_ = false;
        }
    }
    plan_ = make_plan(cfg_, devices.size(), weights, kvs_);
    print_plan(cfg_, plan_);
    const std::vector<std::string> weight_files = safetensors_files(model_dir);
    for (const std::string& f : weight_files) st_.open(f);
    // mixed precision: QE_W4_FROM=<INT4 checkpoint dir> supplies the layer matrices of the QE_W4_LAYERS classes
    // (comma list of qkv, o, attn = qkv + o, in, out, dn = in + out, mlp = gate_up + down; default mlp); the
    // rest come from model_dir (e.g. INT8). Same tensor names in both checkpoints; the norms an AWQ checkpoint
    // folded its smoothing into come along (Worker::load).
    if (const char* w4dir = env_str("QE_W4_FROM")) {
        const std::string spec = env_str("QE_W4_LAYERS") ? env_str("QE_W4_LAYERS") : "mlp";
        std::stringstream ss(spec);
        for (std::string t; std::getline(ss, t, ',');) {
            if (t == "qkv") w4_mask_ |= Worker::W4C_QKV;
            else if (t == "o") w4_mask_ |= Worker::W4C_O;
            else if (t == "attn") w4_mask_ |= Worker::W4C_QKV | Worker::W4C_O;
            else if (t == "in") w4_mask_ |= Worker::W4C_IN;
            else if (t == "out") w4_mask_ |= Worker::W4C_OUT;
            else if (t == "dn") w4_mask_ |= Worker::W4C_IN | Worker::W4C_OUT;
            else if (t == "gate_up") w4_mask_ |= Worker::W4C_GU;
            else if (t == "down") w4_mask_ |= Worker::W4C_DOWN;
            else if (t == "mlp") w4_mask_ |= Worker::W4C_GU | Worker::W4C_DOWN;
            else CHECK(false, "QE_W4_LAYERS: unknown class %s", t.c_str());
        }
        // AWQ couples up_proj and down_proj (up carries down's smoothing): gate_up and down only together
        CHECK(!(w4_mask_ & Worker::W4C_GU) == !(w4_mask_ & Worker::W4C_DOWN),
              "QE_W4_LAYERS: gate_up and down must come from the same checkpoint (AWQ smoothing up -> down): use mlp");
        w4_st_.reset(new SafeTensors);
        std::string key = std::string("|w4:") + w4dir + ":" + spec;
        for (const std::string& f : safetensors_files(w4dir)) {
            w4_st_->open(f);
            struct stat stt;
            if (stat(f.c_str(), &stt) == 0) key += ":" + std::to_string(stt.st_size) + ":" + std::to_string(stt.st_mtime);
        }
        w4_key_ = key;
        fprintf(stderr, "mixed precision: %s layers from %s\n", spec.c_str(), w4dir);
    }
    // QE_MTP_FILE: another MTP head (e.g. a fine-tuned one), same tensor names
    const std::string mtp_file = env_str("QE_MTP_FILE") ? env_str("QE_MTP_FILE") : model_dir + "/model_mtp.safetensors";
    if (FILE* f = fopen(mtp_file.c_str(), "rb")) {
        fclose(f);
        mtp_st_.reset(new SafeTensors);
        mtp_st_->open(mtp_file);
    }
    CUDA_CHECK(cudaMallocHost(&emit_host_, 2 * 64 * 4));
    {  // GEMM tuning table: one per model (weights: path, sizes and times of the files)
        std::ostringstream fp;
        char rp[PATH_MAX];
        fp << (realpath(model_dir_.c_str(), rp) ? rp : model_dir_.c_str());
        std::vector<std::string> files = weight_files;  // single-file checkpoints keep their old key
        files.push_back(mtp_file);
        for (const std::string& f : files) {
            struct stat stt;
            if (stat(f.c_str(), &stt) == 0) fp << "|" << f.substr(f.rfind('/') + 1) << ":" << stt.st_size << ":" << stt.st_mtime;
        }
        fp << w4_key_;
        gemm_tune_set_model(fp.str());
    }
    if (env_set("QE_MTP_ADAPT")) adapt_ = env_int("QE_MTP_ADAPT", adapt_);
    if (const char* e = env_str("QE_SPEC_COST")) sscanf(e, "%lf,%lf", &cost0_, &cost1_);
    if (env_set("QE_SNAPSHOTS")) max_snaps_ = std::max(1, env_int("QE_SNAPSHOTS", max_snaps_));
    // snapshot store: checkpoints keep max_snaps_/2 entries of their own (snap_policy.h)
    snap_reserve0_ = max_snaps_ / 2;
    if (env_set("QE_SNAP_CKPT_RESERVE")) snap_reserve0_ = std::max(0, std::min(max_snaps_, env_int("QE_SNAP_CKPT_RESERVE", snap_reserve0_)));
    // checkpoint spacing of a prompt's prefill (state saved every QE_CKPT tokens, multiple of 256)
    if (env_set("QE_CKPT")) ckpt_every_ = std::max(256, env_int("QE_CKPT", ckpt_every_) / 256 * 256);
    // prompt-end state one token before the end when at least QE_END_CUT tokens are prefilled
    // (an identical prompt then reuses n-1 instead of falling back to a checkpoint); 0 = off
    if (env_set("QE_END_CUT")) end_cut_ = std::max(0, env_int("QE_END_CUT", end_cut_));
    // tail checkpoints: in the last QE_TAIL_CKPT tokens of a prompt, a checkpoint at chunk
    // boundaries at least kChunk apart (no extra passes): requests sharing everything but the
    // tail (RAG: one document, other chunks / questions) resume close to where they diverge
    if (env_set("QE_TAIL_CKPT")) tail_ckpt_ = std::max(0, env_int("QE_TAIL_CKPT", tail_ckpt_));
    // a prefill keeps at most QE_CKPT_KEEP of its periodic / tail checkpoints (older ones are
    // overwritten): a 200K prompt no longer flushes every other conversation's checkpoints
    if (env_set("QE_CKPT_KEEP")) ckpt_keep_ = std::max(1, env_int("QE_CKPT_KEEP", ckpt_keep_));

    // Embedding table in pinned host memory, read by every GPU over PCIe (one row per token).
    const TensorInfo& emb = st_.get("model.language_model.embed_tokens.weight");
    {  // out of host memory right after a crash (the previous process's memory is freed while its core is written):
        // retried for up to QE_PIN_WAIT_S (60) seconds, with the largest processes in the log
        static const int wait_s = env_int("QE_PIN_WAIT_S", 60);
        cudaError_t e = cudaHostAlloc(&embed_, emb.nbytes, cudaHostAllocMapped | cudaHostAllocPortable);
        for (int i = 0; e == cudaErrorMemoryAllocation && i < wait_s; i++) {
            cudaGetLastError();
            if (i == 0)
                fprintf(stderr, "qengine: pinning the embeddings (%.2f GB): out of host memory (MemAvailable %.1f GB); "
                                "retrying for up to %d s; largest processes: %s\n", emb.nbytes / 1e9,
                        qe_hostmem::available(true) / 1e9, wait_s, qe_hostmem::top_rss().c_str());
            sleep(1);
            e = cudaHostAlloc(&embed_, emb.nbytes, cudaHostAllocMapped | cudaHostAllocPortable);
        }
        CUDA_CHECK(e);
    }
    memcpy(embed_, emb.data, emb.nbytes);

    comm_ = comm_create(devices.size(), kChunk * cfg_.hidden, cfg_.hidden);
    devs_ = devices;
    if (comm_enable_p2p(comm_, devices.data()))
        printf("P2P: decode all-reduce GPU to GPU (peer access between all %zu GPUs)\n", devices.size());
    // startup probe + prefill transfer path (QE_COMM=auto|p2p|ce|host): the copy-engine prefill all-reduce stays the
    // default; p2p = SM-store all-reduce; host (or auto when host staging is markedly faster than both peer paths,
    // e.g. GPUs on different NUMA nodes behind slow links): no peer access, every collective through host memory
    if (comm_select_path(comm_, devices.data(), "QE_COMM", false) == kXferHost && comm_.p2p) {
        comm_free_p2p(comm_, devices.data());
        printf("QE_COMM: collectives through pinned host memory (no peer access)\n");
    }
    if (kvs_.on() && !comm_.p2p) {  // the stats exchange needs peer access: fall back to the old layout
        fprintf(stderr, "QE_KV_SPLIT: no P2P between the GPUs, old KV layout\n");
        kvs_ = KvSplit{};
        kvs_auto_ = false;
        plan_ = make_plan(cfg_, devices.size(), weights, kvs_);
    }
    for (size_t r = 0; r < devices.size(); r++)
        workers_.emplace_back(new Worker(cfg_, plan_, r, devices[r], comm_));
    // MTP drafts only over the most frequent tokens (tools/draft_vocab.py ranking), split
    // across GPUs; QE_DRAFT_VOCAB=0 uses the full vocabulary
    {
        int nd = env_int("QE_DRAFT_VOCAB", 65536);
        FILE* f = fopen((model_dir + "/token_rank.bin").c_str(), "rb");
        if (f && nd > 0 && mtp_st_) {
            nd = std::min(nd, cfg_.vocab) / (16 * (int)devices.size()) * 16 * devices.size();
            std::vector<int> rank(nd);
            CHECK(fread(rank.data(), 4, nd, f) == (size_t)nd, "token_rank.bin");
            std::sort(rank.begin(), rank.end());
            const int per = nd / devices.size();
            for (size_t r = 0; r < devices.size(); r++)
                workers_[r]->draft_ids.assign(rank.begin() + r * per, rank.begin() + (r + 1) * per);
            printf("MTP draft vocabulary: %d most frequent tokens\n", nd);
        } else if (!f && nd > 0 && mtp_st_) {
            // the full-vocabulary draft head is ~4x the bytes per draft (INT8 setups: ~0.3 ms of each ~0.7 ms draft)
            fprintf(stderr, "MTP draft vocabulary: no %s/token_rank.bin, drafting over the full vocabulary "
                            "(slower; copy it from the INT4 checkpoint, it depends only on the tokenizer)\n",
                    model_dir.c_str());
        }
        if (f) fclose(f);
    }
    pool_.reset(new GpuPool(devices));
    for (auto& w : workers_) {
        w->w4_st = w4_st_.get();
        w->w4_mask = w4_mask_;
    }
    pool_->run([&](int r) { workers_[r]->load(st_, mtp_st_.get(), embed_); });
    // the stats exchange of a position-split head: each card writes into its peer's rows / flags
    for (size_t r = 0; r < workers_.size(); r++)
        for (int j = 0; j < plan_[r].kv_heads.size(); j++)
            if (plan_[r].kv_x[j] == 3) {
                const Worker& peer = *workers_[plan_[r].kv_peer[j]];
                workers_[r]->set_xchg_peer(j, peer.xchg_buf(), peer.xchg_flags());
            }
    plan_slots(max_t);
    if (!steer_.empty()) {  // activation steering vectors (validated above), on every GPU
        const int nv = steer_.size();
        std::vector<int> lo(nv), hi(nv);
        std::vector<float> w;
        for (int v = 0; v < nv; v++) {
            lo[v] = steer_[v].flo;
            hi[v] = steer_[v].fhi;
            w.insert(w.end(), steer_[v].w.begin(), steer_[v].w.end());
            printf("%s\n", steer_[v].log_line().c_str());
        }
        pool_->run([&](int r) { workers_[r]->set_steer(nv, lo.data(), hi.data(), w.data()); });
        steer_json_ = qsteer::steer_info_json(steer_, steer_def_);
        for (auto& x : steer_) std::vector<float>().swap(x.w);
        printf("steering: %d vector(s), default %s\n", nv, steer_def_ >= 0 ? steer_[steer_def_].name.c_str() : "off");
    }
    set_steering(-2, 1.f, -1, -1, -1);
    CUDA_CHECK(cudaSetDevice(workers_[0]->device));
    for (auto& ev : emit_ev_) CUDA_CHECK(cudaEventCreate(&ev));
    for (auto& ev : tstart_) CUDA_CHECK(cudaEventCreate(&ev));
    double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    for (auto& w : workers_)
        printf("GPU%d: weights %.2f GB, cache+act %.2f GB\n", w->device, w->weight_bytes / 1e9,
               w->cache_bytes / 1e9);
    printf("loaded in %.1fs\n", secs);
    // Pinning snapshot buffers costs ~20 ms per GPU each: done in the background at startup
    // instead of on the request path.
    std::vector<size_t> sizes;
    for (auto& w : workers_) sizes.push_back(w->snapshot_bytes());
    archive_budget_ = (size_t)((env_double("QE_KV_ARCHIVE_GB", 32.0)) *
                               (1ull << 30));
    if (env_set("QE_ARCHIVE_MIN")) archive_min_ = env_int("QE_ARCHIVE_MIN", archive_min_);
    if (env_set("QE_SHARE_PREFIX")) share_prefix_ = env_int("QE_SHARE_PREFIX", share_prefix_) != 0;
    pool_log_ = env_int("QE_POOL_LOG", 0);
    prefix_budget_ = (size_t)((env_double("QE_PREFIX_GB", 8.0)) * (1ull << 30));
    disk_init();
    if (env_int("QE_ARCHIVE_PINNED", 1))
        hpool_.start(archive_budget_ + prefix_budget_,
                     (size_t)((env_double("QE_ARCHIVE_SLAB_GB", 4.0)) * (1ull << 30)));
    alloc_thread_ = std::thread([this, sizes] {
        // errors (e.g. the process exiting while pinning) just end the background allocation
        for (int i = 0; i < max_snaps_ && !stop_alloc_; i++) {
            std::vector<void*> b;
            for (size_t sz : sizes) {
                void* p;
                if (cudaHostAlloc(&p, sz, cudaHostAllocPortable) != cudaSuccess) return;
                b.push_back(p);
            }
            std::lock_guard<std::mutex> lk(snap_mu_);
            free_bufs_.push_back(std::move(b));
        }
    });
}

Engine::~Engine() {
    if (disk_thread_.joinable()) {  // pending writes are finished first
        {
            std::lock_guard<std::mutex> l(disk_mu_);
            disk_stop_ = true;
        }
        disk_cv_.notify_all();
        disk_thread_.join();
    }
    stop_alloc_ = true;
    if (alloc_thread_.joinable()) alloc_thread_.join();
    for (auto& b : free_bufs_)
        for (void* p : b) cudaFreeHost(p);
    pool_.reset();
    workers_.clear();
    for (auto& s : snaps_)
        for (void* p : s.bufs) cudaFreeHost(p);
    if (embed_) cudaFreeHost(embed_);
    if (comm_.p2p) comm_free_p2p(comm_, devs_.data());
    for (void* p : {(void*)comm_.data, (void*)comm_.flags, (void*)comm_.ll})
        if (p) cudaFreeHost(p);
    if (batch_host_) cudaFreeHost(batch_host_);
    if (img_table_) cudaFreeHost(img_table_);
    if (emit_host_) cudaFreeHost(emit_host_);
}

// Main slot: as much context as fits up to main_cap (the model maximum by default); the rest
// of the free memory becomes subagent slots. QE_SLOTS="262144,65536,..." overrides.
// KV pool: every GPU allocates the same number of tokens once; sequence slots (logical, up to
// kMaxBatch decoding together) get regions of it on demand (ensure_cap), so one long
// conversation can use most of the memory and many short ones share it.
void Engine::plan_slots(int main_cap) {
    int nslots = env_int("QE_MAX_SEQS", 16);
    size_t cap_tokens = env_ll("QE_POOL_TOKENS", SIZE_MAX);
    if (const char* e = env_str("QE_SLOTS")) {  // legacy: "a,b,c" = 3 slots sharing a+b+c tokens
        std::stringstream ss(e);
        size_t sum = 0;
        nslots = 0;
        for (std::string x; std::getline(ss, x, ',');) sum += std::stoll(x), nslots++;
        cap_tokens = sum;
    }
    CHECK(nslots >= 1 && nslots <= kMaxSegs, "slots %d (max %d)", nslots, kMaxSegs);
    for (int i = 0; i < nslots; i++) {
        pool_->run([&](int r) { workers_[r]->add_slot(); });
        seqs_.push_back(Seq{});
    }
    // VRAM left outside the KV pool, per card in worker order: "1.5" for every card or a list
    // "1.5,3,1.5" (more room on the card driving the desktop); a short list repeats its last value
    std::vector<size_t> reserve(workers_.size(), (size_t)1.5e9);
    if (const char* e = env_str("QE_RESERVE_GB")) {
        std::stringstream ss(e);
        size_t r = 0;
        double last = 1.5;
        for (std::string x; std::getline(ss, x, ',') && r < reserve.size();) reserve[r++] = (size_t)((last = atof(x.c_str())) * 1e9);
        for (; r < reserve.size(); r++) reserve[r] = (size_t)(last * 1e9);
    }
    // pool tokens every card fits with its free memory past the reserve: the cards hold the same
    // token offsets, so the smallest (free_r - reserve_r) / bytes per token_r
    auto pool_fit = [&]() {
        size_t fit = SIZE_MAX;
        for (size_t r = 0; r < workers_.size(); r++) {
            const size_t fr = workers_[r]->free_bytes(), res = reserve[r];
            fit = std::min(fit, (fr > res ? fr - res : 0) / workers_[r]->slot_token_bytes());
        }
        return fit;
    };
    size_t fit_fresh = 0, fit_used = 0;
    if (kvs_auto_) {
        // shares from the free memory after the weights: with x = GPU0's share of kv1 and y =
        // GPU2's of kv2 the cards hold 1 + x, 2 - x - y, 1 + y heads per token; every card's pool
        // fills when f_r / (heads_r) are equal: per = (f0 + f1 + f2) / 4 head-tokens; x = f0 / per - 1
        std::vector<double> f(workers_.size());
        for (size_t r = 0; r < workers_.size(); r++) {
            const size_t fr = workers_[r]->free_bytes(), res = reserve[r];
            f[r] = fr > res ? (double)(fr - res) : 0.0;
        }
        const double per = (f[0] + f[1] + f[2]) / 4;
        const double x = std::min(0.55, std::max(0.25, f[0] / per - 1)), y = std::min(0.55, std::max(0.25, f[2] / per - 1));
        // the shares move in steps of 32 / period: of the neighbours of the exact shares (within the
        // same [0.25, 0.55] bounds) take the pair whose pool is the largest (the cards' bytes per token)
        const int alo = kvs_.to_a(0.55), ahi = kvs_.to_a(0.25);
        auto try_split = [&](int a, int b) {
            kvs_.a01 = a;
            kvs_.a12 = b;
            kv_split_maps(plan_, kvs_);
            pool_->run([&](int r) { workers_[r]->set_plan(plan_[r]); });
            return pool_fit();
        };
        int best_a = kvs_.to_a(x), best_b = kvs_.to_a(y);
        const int a0 = best_a, b0 = best_b;
        for (int da = -64; da <= 64; da += 32)
            for (int db = -64; db <= 64; db += 32) {
                const int a = a0 + da, b = b0 + db;
                if (a < alo || a > ahi || b < alo || b > ahi) continue;
                const size_t t = try_split(a, b);
                if (t > fit_fresh) fit_fresh = t, best_a = a, best_b = b;
            }
        kvs_.a01 = best_a;
        kvs_.a12 = best_b;
        fit_used = fit_fresh;
        // reproducible across restarts: the free memory moves with the desktop on GPU1, so the
        // shares chosen once are kept (QE_KV_SPLIT_FILE, default $QE_DISK_CACHE or
        // ~/.cache/qengine: kv_split_<model>_<gpus>.txt) while their pool stays within 1 % of the
        // fresh choice's; a larger loss (other reserves, other weights, other cards) picks and stores anew
        const char* home = env_str("HOME");
        std::string dir = env_str("QE_DISK_CACHE") && *env_str("QE_DISK_CACHE") ? env_str("QE_DISK_CACHE")
                          : home ? std::string(home) + "/.cache/qengine" : std::string();
        std::string file = env_str("QE_KV_SPLIT_FILE") ? env_str("QE_KV_SPLIT_FILE") : "";
        if (file.empty() && !dir.empty()) {
            std::string base = model_dir_;
            while (!base.empty() && base.back() == '/') base.pop_back();
            base = base.substr(base.find_last_of('/') + 1);
            file = dir + "/kv_split_" + base + "_g" + std::to_string(workers_.size()) + ".txt";
        }
        bool stored = false;
        if (!file.empty()) {
            int p = 0, a = 0, b = 0;
            if (FILE* fh = fopen(file.c_str(), "r")) {
                if (fscanf(fh, "%d %d %d", &p, &a, &b) == 3 && p == kvs_.period && a % 32 == 0 && b % 32 == 0 &&
                    a >= alo && b >= alo && a <= ahi && b <= ahi) {
                    const size_t t = try_split(a, b);
                    if ((double)t >= 0.99 * (double)fit_fresh) stored = true, best_a = a, best_b = b, fit_used = t;
                }
                fclose(fh);
            }
            kvs_.a01 = best_a;
            kvs_.a12 = best_b;
            if (!stored) {
                if (!dir.empty()) mkdir(dir.c_str(), 0755);
                if (FILE* fh = fopen(file.c_str(), "w")) {
                    fprintf(fh, "%d %d %d\n", kvs_.period, kvs_.a01, kvs_.a12);
                    fclose(fh);
                }
            }
        }
        kv_split_maps(plan_, kvs_);
        pool_->run([&](int r) { workers_[r]->set_plan(plan_[r]); });
        printf("KV split: GPU0 holds %.3f of kv1, GPU2 %.3f of kv2 (free past the reserve %.2f / %.2f / %.2f GB, "
               "pool %zu tokens, fresh choice %zu)%s\n",
               1.0 - (double)kvs_.a01 / kvs_.period, 1.0 - (double)kvs_.a12 / kvs_.period, f[0] / 1e9, f[1] / 1e9, f[2] / 1e9,
               fit_used, fit_fresh, stored ? " [shares kept from the previous run]" : "");
    }
    // every card holds the same token offsets; its bytes per token follow its share of the kv
    // heads (kv-pos-split), so the pool is the smallest (free / bytes per token)
    // lookup drafts on the GPU (QE_LOOKUP=1): the current sequence's history on every card (1 MB), before the pool
    lk_on_ = has_mtp() && (env_int("QE_LOOKUP", 1));  // QE_LOOKUP=0: off
    if (lk_on_) {
        const int mmin = env_int("QE_LK_MIN", 24);  // shorter matches: MTP drafts as well (k up to 7)
        const int lmax = env_set("QE_LK_LMAX") ? std::max(1, std::min(kMaxLookupK, env_int("QE_LK_LMAX", kMaxLookupK))) : kMaxLookupK;
        pool_->run([&](int r) { workers_[r]->lk_init(kMaxPos + 64, mmin, lmax); });
        lkb_on_ = workers_[0]->lkb_on();
        lkb_rows_ = env_int("QE_LKB_ROWS", lkb_rows_);  // rows a batched step may grow to
        lkb_hold_ = env_int("QE_LKB_HOLD", lkb_hold_);  // steps without a find before MTP again
        lkb_min_ = env_int("QE_LKB_MIN", lkb_min_);     // fewest lookup drafts worth a step
        lkb_len_.assign(nslots, -1);
        lkb_miss_.assign(nslots, 0);
        lkb_dirty_.assign(nslots, 0);
        lkb_mode_.assign(nslots, 0);
        lkb_ema_.assign(nslots, 6.f);
        lkb_cool_.assign(nslots, 0);
    }
    const size_t fit = pool_fit();
    for (auto& w : workers_) slack_ = std::max(slack_, w->region_slack());
    pool_tokens_ = std::min(fit, cap_tokens) / 1024 * 1024;
    CHECK(pool_tokens_ >= 8192 + slack_, "not enough GPU memory for a KV cache");
    max_ctx_ = (int)std::min<size_t>(std::min<size_t>(main_cap, kMaxPos), pool_tokens_ - slack_);
    pool_->run([&](int r) { workers_[r]->alloc_pool(pool_tokens_); });
    use_slot(0);
    // the first conversation lands in slot 0: a region now keeps the warm-up graphs valid
    CHECK(ensure_cap(0, std::min<int>(65536, max_ctx_)), "KV pool");
    printf("KV pool: %zu tokens shared by %d sequence slots (max context %d)%s\n", pool_tokens_, nslots, max_ctx_,
           kvs_.on() ? " (kv heads split by position)" : "");
    if (kvs_.on())
        for (auto& w : workers_)
            printf("GPU%d: %zu B per pool token, regions take +%zu tokens\n", w->device, w->slot_token_bytes(),
                   w->region_slack());
}

int Engine::round_cap(int need) const {  // headroom so that a growing conversation moves rarely
    static const size_t head = env_ll("QE_POOL_HEADROOM", 4096);  // 16384 left 32 regions larger than a 2-GPU pool
    static const double frac = env_double("QE_POOL_GROW_FRAC", 0.25);
    const size_t want = ((size_t)need + head + (size_t)(need * frac) + 255) / 256 * 256;
    return (int)std::max<size_t>(need, std::min<size_t>(want, max_ctx_));
}

// The least a region grows to when the pool is tight: need plus a floor of headroom (QE_POOL_MIN_HEAD,
// default 2048, or need / 16 for long contexts). Growing by exactly need re-places the region on
// the next step again: every place() rewrites all its rows (the layout depends on the capacity, in
// chunks no longer than the growth) and drops its CUDA graphs (an agent session fell from 150 to 10 tok/s at 75K).
int Engine::min_cap(int need) const {
    static const size_t head = env_ll("QE_POOL_MIN_HEAD", 2048);
    const size_t want = ((size_t)need + std::max<size_t>(head, (size_t)need / 16) + 255) / 256 * 256;
    return (int)std::max<size_t>(need, std::min<size_t>(want, max_ctx_));
}

size_t Engine::free_after(int s) const {
    const Seq& q = seqs_[s];
    size_t nxt = pool_tokens_;
    for (size_t j = 0; j < seqs_.size(); j++)
        if ((int)j != s && seqs_[j].max_t > 0 && seqs_[j].off >= q.off) nxt = std::min(nxt, seqs_[j].off);
    for (auto& p : pres_)
        if (p.off >= q.off) nxt = std::min(nxt, p.off);
    return nxt - (q.off + units_of(q));
}

// occupied [begin, end) intervals of the pool: conversation regions and prefix regions
void Engine::used_intervals(std::vector<std::pair<size_t, size_t>>& out) const {
    out.clear();
    for (auto& x : seqs_)
        if (x.max_t > 0) out.push_back({x.off, x.off + units_of(x)});
    for (auto& p : pres_) out.push_back({p.off, p.off + units(p.n)});
    std::sort(out.begin(), out.end());
}

// KV rows of slot s that hold data: [0, pos), or for the prompt being prefilled also the rows
// written so far ([0, i) of its prefill state; pos and seq are only set when it finishes).
int Engine::held_rows(int s) const {
    const PrefillState* p = pf_of(s);
    return p ? std::max(seqs_[s].pos, p->i) : seqs_[s].pos;
}

// Moves / resizes slot s's region to (off, cap) keeping its rows, on every GPU.
void Engine::place(int s, size_t off, int cap) {
    Seq& q = seqs_[s];
    static const bool log = env_int("QE_POOL_LOG", 0);
    if (log)
        printf("pool: slot %d [%zu +%d) -> [%zu +%d), %d rows%s\n", s, q.off, q.max_t, off, cap, held_rows(s),
               q.active ? " (active)" : ""), fflush(stdout);
    // the region holds the own rows [split, cap): cap is the logical capacity
    const int rows_old = rows_of(q), rows_new = cap > 0 ? cap - q.split : 0;
    CHECK(rows_new >= 0, "place: capacity %d below the shared prefix %d", cap, q.split);
    if (q.max_t > 0 && cap > q.max_t) n_grow_++;
    else if (q.max_t > 0 && q.off != off) n_move_++;
    if (q.max_t > 0 && (q.off != off || q.max_t != cap)) {
        const int rows = std::min(std::min(rows_old, rows_new), std::max(0, held_rows(s) - q.split) + 2 * kMaxSpecK + 4);
        pool_->run([&](int r) { workers_[r]->move_region(q.off, rows_old, off, rows_new, q.split, rows); });
    }
    q.off = off;
    q.max_t = cap;
    pool_->run([&](int r) {
        workers_[r]->set_region(s, off, rows_new);
        workers_[r]->clear_graphs(s);
    });
}

// Regions packed from offset 0, address order kept: every move goes down behind the regions
// already packed, so no region's rows are overwritten before they move.
void Engine::compact() {
    std::vector<int> order;
    for (size_t j = 0; j < seqs_.size(); j++)
        if (seqs_[j].max_t > 0) order.push_back(j);
    std::sort(order.begin(), order.end(), [&](int a, int b) { return seqs_[a].off < seqs_[b].off; });
    std::vector<std::pair<size_t, size_t>> fixed;  // prefix regions stay where they are
    for (auto& p : pres_) fixed.push_back({p.off, p.off + units(p.n)});
    std::sort(fixed.begin(), fixed.end());
    size_t cur = 0;
    for (int j : order) {
        const size_t rows = units_of(seqs_[j]);
        for (auto& f : fixed)
            if (cur < f.second && cur + rows > f.first) cur = f.second;
        if (seqs_[j].off != cur) place(j, cur, seqs_[j].max_t);
        cur += rows;
    }
}

// First gap that holds `want` tokens without overlapping any region, slot s's own included (its
// rows move there): s is placed in it.
bool Engine::place_in_gap(int s, int want) {
    std::vector<std::pair<size_t, size_t>> used;
    used_intervals(used);
    used.push_back({pool_tokens_, pool_tokens_});
    const size_t rows = units(want - seqs_[s].split);
    size_t cur = 0;
    for (auto& u : used) {
        if (u.first >= cur + rows) {
            place(s, cur, want);
            return true;
        }
        cur = std::max(cur, u.second);
    }
    return false;
}

bool Engine::try_grow(int s, int want) {
    Seq& q = seqs_[s];
    // pool units the region grows by: its own rows cost rows + slack (kv-pos-split), so a region
    // that holds no own rows yet (max_t == split: everything read from the prefix) needs the slack too
    const size_t grow = units(want - q.split) - units_of(q);
    if (q.max_t > 0 && free_after(s) >= grow) {  // room right behind it
        place(s, q.off, want);
        return true;
    }
    if (place_in_gap(s, want)) return true;
    compact();
    if (q.max_t > 0 && free_after(s) >= grow) {
        place(s, q.off, want);
        return true;
    }
    if (place_in_gap(s, want)) return true;
    // s holds rows and the free room is split around regions above it (e.g. a prompt resuming
    // from an idle slot's snapshot low in the pool with an active sequence above it, or that
    // sequence growing below the prompt's region): after compact() the room below s is gone;
    // move the regions above s up to the pool end, the room gathers right behind s
    if (q.max_t <= 0) return false;
    std::vector<std::pair<size_t, size_t>> used;
    used_intervals(used);
    size_t total = grow;
    for (auto& u : used) total += u.second - u.first;
    if (total > pool_tokens_) return false;
    pack_up_above(s);
    if (free_after(s) >= grow) {
        place(s, q.off, want);
        return true;
    }
    return false;
}

// Regions above slot s move up as far as they go (from the top, address order kept: each move
// goes up into room already vacated, so no rows are overwritten before they move); prefix
// regions stay where they are.
void Engine::pack_up_above(int s) {
    const size_t off_s = seqs_[s].off;
    std::vector<int> order;
    for (size_t j = 0; j < seqs_.size(); j++)
        if ((int)j != s && seqs_[j].max_t > 0 && seqs_[j].off > off_s) order.push_back(j);
    std::sort(order.begin(), order.end(), [&](int a, int b) { return seqs_[a].off > seqs_[b].off; });
    std::vector<std::pair<size_t, size_t>> fixed;
    for (auto& p : pres_) fixed.push_back({p.off, p.off + units(p.n)});
    size_t lim = pool_tokens_;
    for (int j : order) {
        const size_t rows = units_of(seqs_[j]);
        size_t end = lim;
        for (bool moved = true; moved;) {
            moved = false;
            for (auto& f : fixed)
                if (end > f.first && end < f.second + rows) {
                    end = f.first;
                    moved = true;
                }
        }
        if (end >= rows && end - rows > seqs_[j].off) place(j, end - rows, seqs_[j].max_t);
        lim = seqs_[j].off;
    }
}

// Slot s gets room for `need` positions: grows in place, else moves to a free gap (compacting the
// pool first if needed); else inactive conversations give back their unused room, active ones
// their headroom, and the least recently used inactive ones are archived (host memory / disk) and
// dropped from the GPU. Never evicts `protect` slots. false: the pool cannot hold it.
bool Engine::ensure_cap(int s, int need, const std::vector<int>& protect, Room mode) {
    if (need > max_ctx_) return false;
    if (seqs_[s].max_t >= need) return true;
    finish();
    const int want = round_cap(need), floor = min_cap(need);
    if (try_grow(s, want)) return true;
    // inactive slots: trim to their contents
    for (size_t j = 0; j < seqs_.size(); j++) {
        Seq& x = seqs_[j];
        const int keep = x.pos ? std::min(x.max_t, (x.pos + 2 * kMaxSpecK + 4 + 1023) / 1024 * 1024) : 0;
        if ((int)j != s && !x.active && x.max_t > keep) place(j, x.off, keep);
    }
    if (try_grow(s, want) || try_grow(s, floor)) return true;
    // active slots: trim their headroom down to the floor (min_cap: below it they would grow again
    // within a few steps); the prompt being prefilled keeps room for all of it
    auto trim_active = [&](bool tight) {
        for (size_t j = 0; j < seqs_.size(); j++) {
            Seq& x = seqs_[j];
            if ((int)j == s || !x.active || pf_of(j)) continue;
            const int end = x.pos + 2 * kMaxSpecK + 4;
            const int keep = std::min(x.max_t, tight ? (end + 255) / 256 * 256 : min_cap(end));
            if (keep > x.pos && x.max_t > keep) place(j, x.off, keep);
        }
    };
    if (mode != FREE) trim_active(false);
    // shared prefix regions nobody reads go first (cheap: they come back from their host copy;
    // evicting a conversation detaches it, so they are retried after every eviction); then
    // inactive conversations: finished ones first, the main thread last (prio), least recently
    // used within a class (the evicted context goes to the archive). They go before s grows by
    // less than the floor: a region squeezed to exactly need is re-placed on every step.
    while (true) {
        if (try_grow(s, want) || try_grow(s, floor)) return true;
        if (drop_lru_prefix()) continue;
        if (!evict_one(s, protect)) break;
    }
    // nothing left to evict: the active slots give back their floor too, and s takes the most
    // headroom that fits (halving from the floor down to exactly need). Decode steps pause a
    // conversation instead (room_or_pause): this is for a sequence alone in the pool.
    if (mode != SQUEEZE) return false;
    trim_active(true);
    for (int h = floor - need; h >= 256; h /= 2)
        if (try_grow(s, std::min(max_ctx_, (need + h + 255) / 256 * 256))) return true;
    if (try_grow(s, need)) return true;
    // (rate-limited: a caller that keeps stepping a sequence without room asks again every call)
    static long n_full = 0;
    const long k = ++n_full;
    if (k <= 10 || k % 1000 == 0)
        fprintf(stderr, "KV pool full: %d tokens needed for slot %d (pool %zu)%s\n", need, s, pool_tokens_,
                k == 10 ? " (further messages every 1000th)" : "");
    return false;
}

// Archive and drop the idle conversation worth least (see ensure_cap); false: none left.
bool Engine::evict_one(int keep_slot, const std::vector<int>& protect) {
    int v = -1;
    for (size_t j = 0; j < seqs_.size(); j++) {
        const Seq& x = seqs_[j];
        if ((int)j == keep_slot || x.active || x.max_t == 0) continue;
        if (std::find(protect.begin(), protect.end(), (int)j) != protect.end()) continue;
        if (v < 0 || x.prio < seqs_[v].prio || (x.prio == seqs_[v].prio && x.used < seqs_[v].used)) v = j;
    }
    if (v < 0) return false;
    const int keep_cur = cur_;
    use_slot(v);
    archive_slot(v);  // host memory / disk: a later request resumes it from there
    n_evict_++;
    seqs_[v].pos = 0;
    seqs_[v].seq.clear();
    use_slot(keep_cur);
    detach(v);
    place(v, seqs_[v].off, 0);
    return true;
}

// ---------------------------------------------------------------- preemption (pause / resume)
// Two or more conversations decoding in a pool too small for all of them: instead of squeezing
// regions step by step (a re-place per step) or ending one with its end token, one is paused:
// its KV and delta-net state go to a held archive in host memory, its region is freed and its
// steps return no tokens (the caller's stream stalls). It resumes where it stopped (same rows,
// same state, no prefill) once its region fits again without trimming a running sequence.
bool Engine::room_or_pause(int s, int need) {
    if (ensure_cap(s, need, {}, FLOOR)) return true;
    while (true) {
        // contenders: decoding sequences (prompts being prefilled were admitted with their room)
        int v = -1, others = 0;
        for (size_t j = 0; j < seqs_.size(); j++) {
            const Seq& x = seqs_[j];
            if (!x.active || x.paused || x.max_t <= 0 || pf_of(j)) continue;
            if ((int)j != s) others++;
            if (v < 0 || x.prio < seqs_[v].prio || (x.prio == seqs_[v].prio && x.admit > seqs_[v].admit)) v = j;
        }
        if (others == 0 || v < 0 || !pause_slot(v)) break;  // alone (or not archivable): squeeze
        if (v == s) return false;
        if (ensure_cap(s, need, {}, FLOOR)) return true;
    }
    return ensure_cap(s, need);
}

bool Engine::pause_slot(int v) {
    finish();
    Seq& q = seqs_[v];
    if (q.paused || !q.active || q.pos <= q.split) return false;
    const int keep_cur = cur_;
    use_slot(v);
    archive_slot(v, true);
    KvArchive* arch = nullptr;
    for (auto& a : archives_)
        if (a.ram && a.sk == q.sk && a.n == q.pos && a.split == q.split &&
            std::equal(q.seq.begin(), q.seq.begin() + q.pos, a.toks.begin()))
            arch = &a;
    if (keep_cur >= 0 && keep_cur != v) use_slot(keep_cur);
    if (!arch) {
        if (pool_log_) printf("pool: slot %d (%d tokens) cannot be paused: no archive room\n", v, q.pos), fflush(stdout);
        return false;
    }
    arch->hold++;
    if (pool_log_)
        printf("pool: slot %d paused at %d tokens (prio %d), region [%zu +%d) freed\n", v, q.pos, q.prio, q.off, q.max_t),
            fflush(stdout);
    // the rows are gone: nothing may copy them (q.seq / q.pos describe the region's contents).
    // The slot's device tokens, drafts and lookup history live outside the pool and stay: it
    // resumes with the same step it would have taken (greedy replies equal an undisturbed run)
    paused_[v] = Paused{arch, q.pos, std::move(q.seq), n_release_, ++tick_};
    q.seq.clear();
    q.pos = 0;
    place(v, q.off, 0);
    q.paused = true;
    q.rb_epoch = -1;
    n_pause_++;
    return true;
}

bool Engine::unpause_slot(int v) {
    Seq& q = seqs_[v];
    auto it = paused_.find(v);
    if (!q.paused || it == paused_.end()) return true;
    Paused& p = it->second;
    p.seen = n_release_;
    const int need = std::min(p.pos + 2 * (kMaxSpecK + 1) + 1, max_ctx_);
    // hysteresis: only when the room left keeps every running sequence's floor (min_cap) too, so
    // that none of them has to pause a conversation again right away. Idle conversations and
    // unread prefixes count as room (ensure_cap archives / drops them).
    {
        size_t held = 0, deficit = 0;
        for (size_t j = 0; j < seqs_.size(); j++) {
            const Seq& x = seqs_[j];
            if (!x.active || x.max_t <= 0) continue;
            held += units_of(x);
            if (!x.paused && !pf_of(j)) {
                const size_t fl = units(min_cap(std::min(x.pos + 2 * kMaxSpecK + 4, max_ctx_)) - x.split);
                if (fl > units_of(x)) deficit += fl - units_of(x);
            }
        }
        for (auto& pr : pres_)
            if (pr.refs > 0) held += units(pr.n);
        if (held + deficit + units(min_cap(need) - q.split) > pool_tokens_) return false;
    }
    if (!ensure_cap(v, need, {}, FREE)) return false;
    finish();
    const int keep_cur = cur_;
    use_slot(v);
    KvArchive* a = p.arch;
    pool_->run([&](int r) {
        void* b = a->ram->at(r).get();
        workers_[r]->archive_xfer(v, q.split, a->n, a->n, &b, kArchiveChunk, false, true);
    });
    if (keep_cur >= 0 && keep_cur != v) use_slot(keep_cur);
    a->hold--;
    a->used = ++tick_;
    q.pos = p.pos;
    q.seq = std::move(p.seq);
    paused_.erase(it);
    q.paused = false;
    q.rb_epoch = -1;
    n_unpause_++;
    if (pool_log_) printf("pool: slot %d resumed at %d tokens in [%zu +%d)\n", v, q.pos, q.off, q.max_t), fflush(stdout);
    return true;
}

// ---------------------------------------------------------------- shared prefix regions
int Engine::find_prefix(const int* ids, int n, long long sk) const {
    for (size_t i = 0; i < pres_.size(); i++)
        if (pres_[i].sk == sk && pres_[i].n <= n && std::equal(pres_[i].toks.begin(), pres_[i].toks.end(), ids)) return i;
    return -1;
}

bool Engine::drop_lru_prefix() {
    int v = -1;
    for (size_t i = 0; i < pres_.size(); i++)
        if (pres_[i].refs == 0 && (v < 0 || pres_[i].used < pres_[v].used)) v = i;
    if (v < 0) return false;
    if (pool_log_) printf("pool: prefix %d tokens at %zu dropped\n", pres_[v].n, pres_[v].off), fflush(stdout);
    pres_.erase(pres_.begin() + v);
    for (auto& q : seqs_)
        if (q.pre > v) q.pre--;
    return true;
}

// Offset of a free gap of n rows for a prefix region, taken from the top of the pool (so the
// conversations, packed from the bottom, never have to move it): compaction, unreferenced
// prefixes and idle conversations give way in that order. SIZE_MAX: no room.
size_t Engine::prefix_room(int n, const std::vector<int>& protect) {
    finish();  // regions may move (compaction): no step queued ahead may be in flight
    auto top_gap = [&]() -> size_t {
        std::vector<std::pair<size_t, size_t>> used;
        used_intervals(used);
        const size_t need = units(n);
        size_t cur = pool_tokens_;
        for (auto it = used.rbegin(); it != used.rend(); ++it) {
            if (cur >= it->second && cur - it->second >= need) return cur - need;
            cur = std::min(cur, it->first);
        }
        return cur >= need ? cur - need : ~(size_t)0;
    };
    size_t off = top_gap();
    if (off != ~(size_t)0) return off;
    compact();
    if ((off = top_gap()) != ~(size_t)0) return off;
    while (drop_lru_prefix())
        if ((off = top_gap()) != ~(size_t)0) return off;
    while (evict_one(-1, protect)) {
        compact();
        if ((off = top_gap()) != ~(size_t)0) return off;
    }
    return ~(size_t)0;
}

// A pool region with the KV rows of a pinned prefix (from its host copy): index in pres_, -1
// if the pool has no room.
int Engine::materialize_prefix(const KvArchive& a, const std::vector<int>& protect) {
    const int have = find_prefix(a.toks.data(), a.n, a.sk);
    if (have >= 0 && pres_[have].n == a.n) return have;
    const size_t off = prefix_room(a.n, protect);
    if (off == ~(size_t)0) return -1;
    GpuPrefix p;
    p.toks = a.toks;
    p.n = a.n;
    p.sk = a.sk;
    p.off = off;
    p.used = ++tick_;
    pool_->run([&](int r) {
        void* b = a.ram->at(r).get();
        workers_[r]->region_xfer(off, a.n, 0, 0, a.n, a.n, &b, kArchiveChunk, false, false);
    });
    if (pool_log_) printf("pool: prefix %d tokens materialized at %zu\n", a.n, off), fflush(stdout);
    pres_.push_back(std::move(p));
    return pres_.size() - 1;
}

void Engine::attach(int slot, int pre) {
    Seq& q = seqs_[slot];
    if (q.pre == pre) return;
    detach(slot);
    GpuPrefix& p = pres_.at(pre);
    q.pre = pre;
    q.split = p.n;
    if (q.max_t > 0) q.max_t += p.n;  // the own region keeps its rows: they are keys from split on
    p.refs++;
    p.used = ++tick_;
    pool_->run([&](int r) {
        workers_[r]->set_prefix(slot, p.off, p.n);
        workers_[r]->clear_graphs(slot);
    });
}

void Engine::detach(int slot) {
    Seq& q = seqs_[slot];
    if (q.pre < 0) return;
    pres_.at(q.pre).refs--;
    if (q.max_t > 0) q.max_t -= q.split;
    q.pre = -1;
    q.split = 0;
    pool_->run([&](int r) {
        workers_[r]->set_prefix(slot, 0, 0);
        workers_[r]->clear_graphs(slot);
    });
}

// Keys [lo, hi) of slot src into slot dst (each side's own region or shared prefix; keys the
// destination reads from its prefix are not written).
void Engine::copy_rows(int dst, int src, int lo, int hi) {
    const Seq &d = seqs_[dst], &s = seqs_[src];
    lo = std::max(lo, d.split);
    if (hi <= lo) return;
    std::vector<int> cuts = {lo, hi};
    if (s.split > lo && s.split < hi) cuts.push_back(s.split);
    std::sort(cuts.begin(), cuts.end());
    for (size_t i = 0; i + 1 < cuts.size(); i++) {
        const int a = cuts[i], b = cuts[i + 1];
        const bool from_pre = b <= s.split;
        const size_t src_off = from_pre ? pres_.at(s.pre).off : s.off;
        const int src_cap = from_pre ? pres_.at(s.pre).n : rows_of(s), src_split = from_pre ? 0 : s.split;
        pool_->run([&](int r) {
            workers_[r]->copy_keys(src_off, src_cap, src_split, d.off, rows_of(d), d.split, a, b);
        });
    }
}

void Engine::slot_info(int i, long long* out) const {
    const Seq& q = seqs_.at(i);
    const auto p = paused_.find(i);
    out[0] = q.off; out[1] = q.max_t; out[2] = p != paused_.end() ? p->second.pos : q.pos; out[3] = q.active;
    out[4] = q.used; out[5] = q.prio; out[6] = q.paused;
}

int Engine::num_pinned() const {
    int c = 0;
    for (auto& a : archives_) c += a.pinned;
    return c;
}

void Engine::archive_info(double* out) const {
    out[0] = out[1] = out[2] = out[3] = 0;
    out[4] = hpool_.pinned_bytes();
    out[5] = hpool_.used_bytes();
    for (auto& a : archives_) {
        if (a.ram) out[0] += 1, out[1] += a.total;
        if (!a.path.empty()) out[2] += 1, out[3] += a.total;
    }
}

// headroom counted per sequence at admission (its growth before it has to move); QE_ADMIT_HEADROOM overrides
static size_t admit_headroom() {
    static const size_t ah = env_ll("QE_ADMIT_HEADROOM", 8192);  // 16384 admitted only 22 of 32 sequences on 2 GPUs
    return ah;
}

bool Engine::pause_for(int n, int prio) {
    const size_t ah = admit_headroom();
    if (n > max_ctx_) return false;
    bool slot = false;
    size_t used = 0;
    struct V { int j; size_t u; };
    std::vector<V> vs;
    for (size_t i = 0; i < seqs_.size(); i++) {
        const Seq& q = seqs_[i];
        if (!q.active) slot = true;
        else if (pf_of(i)) used += units_of(q);
        else if (!q.paused) {
            const size_t u = units(std::min<size_t>((size_t)q.pos - q.split + ah, max_ctx_));
            used += u;
            if (q.prio < prio && q.max_t > 0 && q.pos > q.split) vs.push_back({(int)i, u});
        }
    }
    for (auto& p : pres_) used += units(p.n);
    const size_t need = units(std::min<size_t>((size_t)n + ah, max_ctx_));
    if (!slot) return false;  // every slot is taken: pausing frees no slot
    if (used + need <= pool_tokens_) return true;
    std::stable_sort(vs.begin(), vs.end(), [&](const V& a, const V& b) {
        const Seq &x = seqs_[a.j], &y = seqs_[b.j];
        return x.prio != y.prio ? x.prio < y.prio : x.admit > y.admit;
    });
    size_t k = 0, left = used;
    while (k < vs.size() && left + need > pool_tokens_) left -= vs[k++].u;
    if (left + need > pool_tokens_) return false;  // pausing all of them would not make it fit: pause none
    for (size_t i = 0; i < k; i++) {
        if (pool_log_)
            printf("pool: pausing slot %d (prio %d) for a %d-token start of prio %d\n", vs[i].j, seqs_[vs[i].j].prio, n, prio),
                fflush(stdout);
        if (!pause_slot(vs[i].j)) return false;
    }
    return true;
}

bool Engine::can_admit(int n) const {
    const size_t ah = admit_headroom();
    bool slot = false;
    size_t used = 0;
    for (size_t i = 0; i < seqs_.size(); i++) {
        const Seq& q = seqs_[i];
        if (!q.active) slot = true;
        else if (pf_of(i)) used += units_of(q);  // its region already holds the whole prompt
        else if (q.paused) used += units(std::min<size_t>((size_t)paused_.at(i).pos - q.split + ah, max_ctx_));  // resumes first
        else used += units(std::min<size_t>((size_t)q.pos - q.split + ah, max_ctx_));
    }
    for (auto& p : pres_) used += units(p.n);
    return slot && n <= max_ctx_ && used + units(std::min<size_t>((size_t)n + ah, max_ctx_)) <= pool_tokens_;
}

unsigned* Engine::mask_buffer(int slot) {
    CHECK(slot >= 0 && slot < (int)seqs_.size(), "mask_buffer slot %d", slot);
    if (mask_host_.size() < seqs_.size()) mask_host_.resize(seqs_.size(), nullptr);
    if (!mask_host_[slot]) {
        const size_t bytes = ((size_t)cfg_.vocab + 31) / 32 * 4 * (kMaxSpecK + 1);
        CUDA_CHECK(cudaSetDevice(workers_[0]->device));
        CUDA_CHECK(cudaMallocHost(&mask_host_[slot], bytes));
        memset(mask_host_[slot], 0xff, bytes);
    }
    return mask_host_[slot];
}

int* Engine::spec_io(int slot) {
    CHECK(slot >= 0 && slot < (int)seqs_.size(), "spec_io slot %d", slot);
    if (spec_io_.size() < seqs_.size()) spec_io_.resize(seqs_.size(), nullptr);
    if (!spec_io_[slot]) {
        CUDA_CHECK(cudaSetDevice(workers_[0]->device));
        CUDA_CHECK(cudaMallocHost(&spec_io_[slot], 32 * 4));
        memset(spec_io_[slot], 0, 32 * 4);
    }
    return spec_io_[slot];
}

int Engine::con_kv(int slot, int cap) {
    const int* io = spec_io(slot);
    return std::max(0, std::min({io[16], cap, kMaxSpecK}));
}

void Engine::push_mask(int slot) {
    const unsigned* h = mask_buffer(slot);
    pool_->run([&](int r) { workers_[r]->upload_mask(slot, h); });
}

void Engine::constrain(int slot, bool on) {
    Seq& q = seqs_.at(slot);
    if (on && !q.constrained) finish_for(slot);  // nothing of this slot may be queued ahead
    q.constrained = on;
    if (on) {
        int* io = spec_io(slot);
        io[0] = io[16] = 0;
        push_mask(slot);
    }
    pool_->run([&](int r) { workers_[r]->set_constrained(slot, on); });
}

void Engine::use_slot(int i) {
    lk_slot_ = -1;  // the device lookup history is the current sequence's
    cur_ = i;
    pool_->run([&](int r) { workers_[r]->select_slot(i); });
}

double Engine::warmup(int modes) {
    const auto t0 = std::chrono::steady_clock::now();
    finish();
    pool_->run([&](int r) { workers_[r]->tune_gemm(); });  // prefill GEMM configurations
    if (!has_mtp()) return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    const int keep = cur_;
    // the decode attention row split is part of the graph key, and launch_step turns it on while a sequence is
    // below the ~38-split length (every short conversation): warm those graphs (they were warmed with it off,
    // and the first steps of the first reply captured them again, ~25 ms each)
    for (auto& w : workers_) w->attn_rs_ = true;
    for (int sl = 0; sl < 1; sl++) {  // the first slot (where the first conversation lands)
        use_slot(sl);
        for (int smp = 0; smp < 2; smp++) {
            if (!(modes & (1 << smp))) continue;
            for (auto& w : workers_) w->set_sampling(samp_, smp != 0);
            pool_->run([&](int r) {
                for (int kv = 0; kv <= kMaxSpecK; kv++)
                    for (int kd = 1; kd <= kMaxSpecK; kd++) workers_[r]->run_spec(kv, kd, false);
            });
        }
        for (auto& w : workers_) w->set_sampling(samp_, false);
    }
    use_slot(keep);
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

void Engine::reset_slot() {
    finish_for(cur_);
    if (bg_.slot == cur_) {  // its content goes: the half-made archive with it
        archive_used_ -= bg_.a.total;
        bg_ = BgArchive{};
    }
    detach(cur_);
    Seq& q = seqs_[cur_];
    q.pos = 0;
    q.seq.clear();
    pool_->run([&](int r) { workers_[r]->reset(); });
}

void Engine::reset() {  // forget every cached prefix (slots and snapshots)
    if (bg_.slot >= 0) archive_used_ -= bg_.a.total;
    bg_ = BgArchive{};
    reset_slot();
    for (size_t i = 0; i < seqs_.size(); i++) {
        detach(i);
        seqs_[i].pos = 0;
        seqs_[i].seq.clear();
        seqs_[i].paused = false;  // (its archive goes below)
    }
    paused_.clear();
    pres_.clear();
    std::lock_guard<std::mutex> lk(snap_mu_);
    for (auto& sn : snaps_) free_bufs_.push_back(std::move(sn.bufs));
    snaps_.clear();
    archives_.clear();  // disk files stay (forgotten until the next start)
    archive_used_ = disk_used_ = prefix_used_ = 0;
}

static int lcp_of(const std::vector<int>& a, const int* ids, int n) {
    int i = 0, e = std::min((int)a.size(), n);
    while (i < e && a[i] == ids[i]) i++;
    return i;
}
// the same for tokens computed with steering key ak by a request of key k: another key matches nothing
static int lcpk(const std::vector<int>& a, long long ak, long long k, const int* ids, int n) {
    return ak == k ? lcp_of(a, ids, n) : 0;
}

// ---------------------------------------------------------------- context archive
namespace {
uint64_t fnv64(const int* p, int n) {
    uint64_t h = 1469598103934665603ull;
    for (int i = 0; i < n; i++) {
        uint32_t v = (uint32_t)p[i];
        for (int b = 0; b < 4; b++) { h ^= (v >> (8 * b)) & 0xff; h *= 1099511628211ull; }
    }
    return h;
}
constexpr char kArchMagic[8] = {'Q', 'K', 'V', 'A', '0', '0', '0', '2'};
template <typename T>
void put(std::vector<char>& b, const T& v) {
    const char* p = (const char*)&v;
    b.insert(b.end(), p, p + sizeof(T));
}
bool read_all(int fd, void* dst, size_t n, size_t off) {
    char* d = (char*)dst;
    while (n) {
        const ssize_t r = pread(fd, d, std::min(n, (size_t)1 << 30), off);
        if (r <= 0) return false;
        d += r; off += r; n -= r;
    }
    return true;
}
}  // namespace

Engine::HostPool::~HostPool() {
    stop_ = true;
    if (th_.joinable()) th_.join();
    for (auto& sl : slabs_) {
        cudaHostUnregister(sl.base);
        ::free(sl.base);
    }
}

// Slabs are malloc'ed, pre-faulted on the CPU and then page-locked with cudaHostRegister
// (portable): ~15 GB/s and no stall of the GPUs, whereas cudaHostAlloc (2.3 GB/s) blocks every
// CUDA call of the other threads for the duration (a 4 GB slab: 0.6-2.4 s).
void Engine::HostPool::start(size_t budget, size_t slab) {
    if (!budget || !slab) return;
    th_ = std::thread([this, budget, slab] {
        const auto t0 = std::chrono::steady_clock::now();
        size_t total = 0;
        bool held = false;
        while (!stop_ && total < budget) {
            const size_t sz = std::min(slab, budget - total);
            // never below the host's margin (host_mem.h): wait while it lacks two slabs (the archive's own
            // pressure release gives slabs back; re-pinning needs the extra slab, no ping-pong)
            if (qe_hostmem::headroom(true) < sz + slab) {
                if (!held)
                    printf("archive pool: pinning held at %.1f GB (MemAvailable %.1f GB, margin %.1f GB)\n", total / 1e9,
                           qe_hostmem::available() / 1e9, qe_hostmem::margin() / 1e9), fflush(stdout);
                held = true;
                for (int i = 0; i < 20 && !stop_; i++) usleep(100000);
                std::lock_guard<std::mutex> l(mu_);
                total = pinned_;  // slabs released meanwhile
                continue;
            }
            if (held) printf("archive pool: pinning resumed at %.1f GB\n", total / 1e9), fflush(stdout);
            held = false;
            void* p = nullptr;
            if (posix_memalign(&p, 1 << 21, sz) != 0 || !p) {
                fprintf(stderr, "archive pool: out of host memory at %.1f GB\n", total / 1e9);
                return;
            }
            // 2 MB pages (THP is madvise-only on this host): 4 KB pages made the registration ~11M pages for
            // 43 GB and slowed every decode step by ~1.5 ms at n = 1; QE_ARCHIVE_THP=0: 4 KB
            static const bool thp = env_int("QE_ARCHIVE_THP", 1);
            if (thp) madvise(p, sz, MADV_HUGEPAGE);
            memset(p, 0, sz);  // fault the pages in without the driver
            if (stop_) { ::free(p); return; }
            if (cudaHostRegister(p, sz, cudaHostRegisterPortable) != cudaSuccess) {
                cudaGetLastError();
                ::free(p);
                fprintf(stderr, "archive pool: pinning stopped at %.1f GB\n", total / 1e9);
                return;
            }
            std::lock_guard<std::mutex> l(mu_);
            slabs_.push_back(Slab{(char*)p, sz, {{0, sz}}});
            pinned_ += sz;
            qe_hostmem::charged() += sz;
            total = pinned_;
        }
        const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        printf("archive pool: %.1f GB pinned in %.1f s\n", total / 1e9, secs);
        {  // 2 MB pages actually backing the slabs (4 KB pages cost ~1.5 ms per decode step, see above)
            size_t huge = 0;
            if (FILE* f = fopen("/proc/self/smaps", "r")) {
                char line[512];
                bool in = false;
                while (fgets(line, sizeof line, f)) {
                    unsigned long lo, hi;
                    if (sscanf(line, "%lx-%lx ", &lo, &hi) == 2 && strchr(line, '-') < strchr(line, ' ')) {
                        in = false;
                        std::lock_guard<std::mutex> l(mu_);
                        for (auto& sl : slabs_)
                            if ((char*)lo < sl.base + sl.size && (char*)hi > sl.base) in = true;
                        continue;
                    }
                    size_t kb;
                    if (in && sscanf(line, "AnonHugePages: %zu kB", &kb) == 1) huge += kb << 10;
                }
                fclose(f);
            }
            printf("archive THP: %.1f GB of %.1f GB on 2 MB pages%s\n", huge / 1e9, total / 1e9,
                   huge < total * 9 / 10 ? " (WARNING: < 90 %, decode steps may be slower)" : "");
        }
        fflush(stdout);
    });
}

char* Engine::HostPool::alloc(size_t bytes, bool& pinned) {
    bytes = (bytes + 4095) / 4096 * 4096;
    {
        std::lock_guard<std::mutex> l(mu_);
        for (auto& sl : slabs_)
            for (auto it = sl.free.begin(); it != sl.free.end(); ++it)
                if (it->second >= bytes) {  // first fit
                    char* p = sl.base + it->first;
                    const size_t off = it->first, len = it->second;
                    sl.free.erase(it);
                    if (len > bytes) sl.free[off + bytes] = len - bytes;
                    used_ += bytes;
                    pinned = true;
                    return p;
                }
    }
    pinned = false;
    qe_hostmem::charged() += bytes;
    return new char[bytes];
}

size_t Engine::HostPool::release(size_t want) {
    std::lock_guard<std::mutex> l(mu_);
    size_t got = 0;
    for (size_t i = slabs_.size(); i-- > 0 && got < want;) {
        Slab& sl = slabs_[i];
        if (sl.free.size() != 1 || sl.free.begin()->first != 0 || sl.free.begin()->second != sl.size) continue;
        cudaHostUnregister(sl.base);
        cudaGetLastError();
        ::free(sl.base);
        got += sl.size;
        pinned_ -= sl.size;
        qe_hostmem::charged() -= sl.size;
        slabs_.erase(slabs_.begin() + i);
    }
    return got;
}

void Engine::HostPool::free(char* p, size_t bytes, bool pinned) {
    if (!p) return;
    if (!pinned) {
        delete[] p;
        qe_hostmem::charged() -= (bytes + 4095) / 4096 * 4096;
        return;
    }
    bytes = (bytes + 4095) / 4096 * 4096;
    std::lock_guard<std::mutex> l(mu_);
    for (auto& sl : slabs_) {
        if (p < sl.base || p >= sl.base + sl.size) continue;
        size_t off = p - sl.base, len = bytes;
        auto nx = sl.free.lower_bound(off);
        if (nx != sl.free.end() && nx->first == off + len) {  // merge with the next free range
            len += nx->second;
            sl.free.erase(nx);
        }
        if (!sl.free.empty()) {
            auto pv = sl.free.lower_bound(off);
            if (pv != sl.free.begin()) {
                --pv;
                if (pv->first + pv->second == off) {  // merge with the previous one
                    off = pv->first;
                    len += pv->second;
                    sl.free.erase(pv);
                }
            }
        }
        sl.free[off] = len;
        used_ -= bytes;
        return;
    }
    CHECK(false, "archive pool: foreign pointer");
}

int Engine::num_disk_archives() const {
    int c = 0;
    for (auto& a : archives_) c += !a.path.empty();
    return c;
}

// Remove an archive entirely (memory and file).
void Engine::drop_archive(std::list<KvArchive>::iterator it) {
    if (it->pinned) prefix_used_ -= it->total;
    else if (it->ram) archive_used_ -= it->total;
    disk_forget(*it);
    archives_.erase(it);
}

// Delete the archive's file (or cancel its pending write); the memory copy stays.
void Engine::disk_forget(KvArchive& a) {
    if (a.path.empty()) return;
    {
        std::lock_guard<std::mutex> l(a.disk->mu);
        if (a.disk->done) unlink(a.path.c_str());
        else a.disk->cancel = true;
    }
    disk_used_ -= a.total;
    a.path.clear();
    a.disk.reset();
}

// Free host memory for `need` more bytes of an archive of priority `prio`: finished
// conversations first, then least recently used; the main thread's archives (prio 2) only make
// room for another main-thread archive. Pinned prefixes are not part of the budget. false: no
// room (the caller skips archiving). An archive that is also on disk only loses its memory copy.
size_t Engine::ram_budget() {
    const size_t cap = std::min(archive_budget_, archive_used_ + hpool_.pinned_free() + qe_hostmem::headroom());
    if (cap < archive_budget_) {
        if (cap_log_.due(cap / 1e9))
            printf("archive: host memory low, budget capped at %.1f of %.1f GB (%.1f GB used, MemAvailable %.1f GB, "
                   "margin %.1f GB)\n", cap / 1e9, archive_budget_ / 1e9, archive_used_ / 1e9,
                   qe_hostmem::available() / 1e9, qe_hostmem::margin() / 1e9), fflush(stdout);
    } else if (cap_log_.last_gb >= 0) {
        printf("archive: host memory ok again, budget %.1f GB\n", archive_budget_ / 1e9), fflush(stdout);
        cap_log_.reset();
    }
    return cap;
}

// memory pressure from outside (another process grew): below the margin, free pinned slabs go back to the host
// first, then archives in ram_make_room's order (any prio, the main thread's last)
void Engine::mem_guard() {
    if (!archive_budget_) return;
    if (!qe_hostmem::deficit()) return;
    const size_t d = qe_hostmem::deficit(true);
    if (!d) return;
    const size_t u0 = archive_used_, p0 = hpool_.pinned_bytes();
    size_t got = hpool_.release(d);
    if (got < d) {
        ram_make_room(0, 2, u0 > d - got ? u0 - (d - got) : 0);
        got += hpool_.release(d - got);
    }
    static double last_log = -1e18;  // a persistent deficit (nothing left to give) is logged once a minute
    const double tnow = std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
    if (got == 0 && archive_used_ == u0 && tnow - last_log < 60) return;
    last_log = tnow;
    printf("archive: host memory below the margin by %.1f GB (MemAvailable %.1f GB, margin %.1f GB): archive %.1f -> "
           "%.1f GB used, %.1f -> %.1f GB pinned\n", d / 1e9, qe_hostmem::available(true) / 1e9, qe_hostmem::margin() / 1e9,
           u0 / 1e9, archive_used_ / 1e9, p0 / 1e9, hpool_.pinned_bytes() / 1e9), fflush(stdout);
}

bool Engine::ram_make_room(size_t need, int prio, size_t limit) {
    if (limit == (size_t)-1) limit = ram_budget();
    while (archive_used_ + need > limit) {
        auto lru = archives_.end();
        for (auto it = archives_.begin(); it != archives_.end(); ++it) {
            if (!it->ram || it->pinned || it->hold) continue;
            if (it->prio >= 2 && prio < 2) continue;
            if (lru == archives_.end() || it->prio < lru->prio || (it->prio == lru->prio && it->used < lru->used))
                lru = it;
        }
        if (lru == archives_.end()) return false;
        if (pool_log_)
            printf("archive: drop %d tokens prio %d (%.2f GB) for %.2f GB of prio %d (used %.2f of %.2f GB)\n", lru->n,
                   lru->prio, lru->total / 1e9, need / 1e9, prio, archive_used_ / 1e9, limit / 1e9), fflush(stdout);
        bool on_disk = !lru->path.empty();
        if (on_disk) {
            std::lock_guard<std::mutex> l(lru->disk->mu);
            on_disk = !(lru->disk->done && !lru->disk->ok);  // a failed write is not a copy
        }
        if (on_disk) {
            lru->ram.reset();
            archive_used_ -= lru->total;
        } else {
            drop_archive(lru);
        }
    }
    return true;
}

void Engine::disk_make_room(size_t need) {
    while (disk_used_ + need > disk_budget_) {
        auto lru = archives_.end();
        for (auto it = archives_.begin(); it != archives_.end(); ++it)
            if (!it->path.empty() && !it->hold && (lru == archives_.end() || it->used < lru->used)) lru = it;
        if (lru == archives_.end()) return;
        if (lru->ram) disk_forget(*lru);
        else drop_archive(lru);
    }
}

// Save the selected slot's context (KV of [0, pos) and the delta-net state at pos, plus the
// states of up to kMaxSubs checkpoints inside it) before the slot is overwritten.
void Engine::archive_slot(int slot, bool force) {
    Seq& q = seqs_[slot];
    const int n = q.pos;
    if (bg_.slot == slot && bg_.n == n) {  // partly archived in the background: finish it
        bg_finish(true);
        return;
    }
    if (bg_.slot == slot) {  // content changed since: start over
        archive_used_ -= bg_.a.total;
        bg_ = BgArchive{};
    }
    // force (pause_slot): any length, and only an archive ending exactly at pos holds its state
    if ((n < archive_min_ && !force) || archive_budget_ == 0 || (force && n <= q.split)) return;
    for (auto& a : archives_)
        if (a.sk == q.sk && (force ? a.n == n && a.split == q.split && a.ram : a.n >= n) &&
            std::equal(q.seq.begin(), q.seq.begin() + n, a.toks.begin())) {
            a.used = ++tick_;  // already archived
            return;
        }
    const int ng = workers_.size();
    // checkpoint states inside [0, n): live snapshots, and those of the archives this one
    // supersedes (e.g. the system-prompt state of a conversation restored after a restart)
    struct Sub { int pos; std::vector<const char*> src; std::shared_ptr<Bufs> hold; };
    std::vector<Sub> subs;
    auto have = [&](int p) {
        for (auto& x : subs) if (x.pos == p) return true;
        return false;
    };
    for (auto& sn : snaps_)
        if (sn.sk == q.sk && sn.pos >= 512 && sn.pos < n && !have(sn.pos) &&
            std::equal(sn.toks.begin(), sn.toks.begin() + sn.pos, q.seq.begin())) {
            Sub x{sn.pos, {}, nullptr};
            for (int r = 0; r < ng; r++) x.src.push_back((const char*)sn.bufs[r]);
            subs.push_back(std::move(x));
        }
    std::vector<std::list<KvArchive>::iterator> superseded;
    for (auto it = archives_.begin(); it != archives_.end(); ++it)
        if (it->sk == q.sk && it->n <= n && !it->pinned && !it->hold &&
            std::equal(it->toks.begin(), it->toks.begin() + it->n, q.seq.begin()))
            superseded.push_back(it);
    for (auto it : superseded)
        for (size_t si = 0; si < it->subs.size(); si++) {
            if (have(it->subs[si])) continue;
            Sub x{it->subs[si], {}, it->ram};
            if (!x.hold) {  // on disk only: read just this state
                bool ok = false;
                {
                    std::unique_lock<std::mutex> l(it->disk->mu);
                    it->disk->cv.wait(l, [&] { return it->disk->done; });
                    ok = it->disk->ok;
                }
                const int fd = ok ? open(it->path.c_str(), O_RDONLY) : -1;
                if (fd < 0) break;
                x.hold = std::make_shared<Bufs>(&hpool_);
                size_t o = it->data_off;
                for (int r = 0; r < ng && ok; r++) {
                    const size_t sb = workers_[r]->snapshot_bytes();
                    x.hold->add(sb);
                    ok = read_all(fd, x.hold->back().get(), sb,
                                  o + workers_[r]->archive_bytes(0, it->n) + si * sb);
                    o += it->sizes[r];
                }
                close(fd);
                if (!ok) break;
                for (int r = 0; r < ng; r++) x.src.push_back(x.hold->at(r).get());
            } else {
                for (int r = 0; r < ng; r++)
                    x.src.push_back(it->ram->at(r).get() + workers_[r]->archive_bytes(it->split, it->n) +
                                    si * workers_[r]->snapshot_bytes());
            }
            subs.push_back(std::move(x));
        }
    std::sort(subs.begin(), subs.end(), [](const Sub& a, const Sub& b) { return a.pos > b.pos; });
    if ((int)subs.size() > kMaxSubs) subs.resize(kMaxSubs);

    KvArchive a;
    a.sk = q.sk;
    a.split = q.split;  // rows [0, split) stay with the shared prefix (named by the tokens)
    for (int r = 0; r < ng; r++) {
        a.sizes.push_back(workers_[r]->archive_bytes(a.split, n) + subs.size() * workers_[r]->snapshot_bytes());
        a.total += a.sizes.back();
    }
    if (a.total > archive_budget_ || !ram_make_room(a.total, q.prio)) {
        if (pool_log_) printf("archive: no room for %d tokens prio %d (%.2f GB)\n", n, q.prio, a.total / 1e9), fflush(stdout);
        return;
    }
    if (pool_log_)
        printf("archive: slot %d %d tokens prio %d, %zu checkpoints, %.2f GB (used %.2f of %.2f GB)\n", slot, n, q.prio,
               subs.size(), a.total / 1e9, (archive_used_ + a.total) / 1e9, archive_budget_ / 1e9), fflush(stdout);
    a.prio = q.prio;
    a.ram = std::make_shared<Bufs>(&hpool_);
    for (size_t sz : a.sizes) a.ram->add(sz);
    for (auto& x : subs) a.subs.push_back(x.pos);
    for (int r = 0; r < ng; r++)
        for (size_t si = 0; si < subs.size(); si++)
            memcpy(a.ram->at(r).get() + workers_[r]->archive_bytes(a.split, n) + si * workers_[r]->snapshot_bytes(),
                   subs[si].src[r], workers_[r]->snapshot_bytes());
    subs.clear();
    for (auto it : superseded) drop_archive(it);  // this archive covers them
    a.toks.assign(q.seq.begin(), q.seq.begin() + n);
    a.n = n;
    a.used = ++tick_;
    pool_->run([&](int r) {
        void* b = a.ram->at(r).get();
        workers_[r]->archive_xfer(slot, a.split, n, n, &b, kArchiveChunk, true, true);
    });
    archive_used_ += a.total;
    archives_.push_back(std::move(a));
    // (steered conversations stay in host memory: the disk cache has no steering key)
    if (!disk_dir_.empty() && !archives_.back().split && archives_.back().sk < 0) disk_enqueue(archives_.back());
}

// Prepare the archive of an idle slot for incremental copying (pieces copied by archive_more).
bool Engine::bg_prepare(int slot) {
    Seq& q = seqs_[slot];
    const int n = q.pos;
    const int ng = workers_.size();
    BgArchive b;
    b.slot = slot;
    b.n = n;
    // checkpoint states inside [0, n) from the live snapshots (the archive's subs)
    std::vector<std::pair<int, const Snapshot*>> subs;
    for (auto& sn : snaps_)
        if (sn.sk == q.sk && sn.pos >= 512 && sn.pos < n && std::equal(sn.toks.begin(), sn.toks.begin() + sn.pos, q.seq.begin()) &&
            std::none_of(subs.begin(), subs.end(), [&](auto& x) { return x.first == sn.pos; }))
            subs.push_back({sn.pos, &sn});
    std::sort(subs.begin(), subs.end(), [](auto& x, auto& y) { return x.first > y.first; });
    if ((int)subs.size() > kMaxSubs) subs.resize(kMaxSubs);
    KvArchive& a = b.a;
    a.sk = q.sk;
    a.split = q.split;
    for (int r = 0; r < ng; r++) {
        a.sizes.push_back(workers_[r]->archive_bytes(a.split, n) + subs.size() * workers_[r]->snapshot_bytes());
        a.total += a.sizes.back();
    }
    if (a.total > archive_budget_ || !ram_make_room(a.total, q.prio)) return false;
    a.ram = std::make_shared<Bufs>(&hpool_);
    for (size_t sz : a.sizes) a.ram->add(sz);
    for (auto& x : subs) a.subs.push_back(x.first);
    for (int r = 0; r < ng; r++)
        for (size_t si = 0; si < subs.size(); si++)
            memcpy(a.ram->at(r).get() + workers_[r]->archive_bytes(a.split, n) + si * workers_[r]->snapshot_bytes(),
                   subs[si].second->bufs[r], workers_[r]->snapshot_bytes());
    a.toks.assign(q.seq.begin(), q.seq.begin() + n);
    a.n = n;
    a.prio = q.prio;
    b.piece.assign(ng, 0);
    archive_used_ += a.total;  // reserved now (ram_make_room made room for it)
    bg_ = std::move(b);
    return true;
}

// Copy the remaining pieces (all of them, or about `budget` bytes per GPU) of the background
// archive; registers it when complete.
void Engine::bg_finish(bool all) {
    if (bg_.slot < 0) return;
    const size_t budget = ~(size_t)0;
    (void)all;
    const int keep = cur_;
    if (cur_ != bg_.slot) use_slot(bg_.slot);
    finish_for(bg_.slot);
    KvArchive& a = bg_.a;
    pool_->run([&](int r) {
        if (bg_.piece[r] < 0) return;
        void* b = a.ram->at(r).get();
        bg_.piece[r] = workers_[r]->archive_xfer(bg_.slot, a.split, a.n, a.n, &b, kArchiveChunk, true, true,
                                                 bg_.piece[r], budget);
    });
    if (keep != cur_) use_slot(keep);
    if (std::all_of(bg_.piece.begin(), bg_.piece.end(), [](int p) { return p < 0; })) {
        // archives this one covers go (as in archive_slot)
        for (auto it = archives_.begin(); it != archives_.end();) {
            auto nx = std::next(it);
            if (it->sk == a.sk && it->n <= a.n && !it->pinned && !it->hold &&
                std::equal(it->toks.begin(), it->toks.begin() + it->n, a.toks.begin()))
                drop_archive(it);
            it = nx;
        }
        a.used = ++tick_;
        if (pool_log_)
            printf("archive: slot %d %d tokens prio %d in the background, %zu checkpoints, %.2f GB (used %.2f of %.2f GB)\n",
                   bg_.slot, a.n, a.prio, a.subs.size(), a.total / 1e9, archive_used_ / 1e9, archive_budget_ / 1e9), fflush(stdout);
        archives_.push_back(std::move(a));
        if (!disk_dir_.empty() && !archives_.back().split && archives_.back().sk < 0) disk_enqueue(archives_.back());
        bg_ = BgArchive{};
    }
}

bool Engine::archive_more(size_t budget) {
    static const size_t min_bytes = 1 << 20;
    if (budget < min_bytes) budget = min_bytes;
    if (bg_.slot < 0) {
        // the idle conversation worth most: highest priority, then the longest (its eviction
        // would block a request the longest), skipping what is archived already
        int best = -1;
        for (size_t j = 0; j < seqs_.size(); j++) {
            const Seq& q = seqs_[j];
            if (q.active || q.pos < archive_min_ || pf_of(j)) continue;
            bool have = false;
            for (auto& a : archives_)
                if (a.sk == q.sk && a.n >= q.pos && (a.ram || !a.path.empty()) &&
                    std::equal(q.seq.begin(), q.seq.begin() + q.pos, a.toks.begin())) { have = true; break; }
            if (have) continue;
            if (best < 0 || q.prio > seqs_[best].prio || (q.prio == seqs_[best].prio && q.pos > seqs_[best].pos)) best = j;
        }
        if (best < 0 || !bg_prepare(best)) return true;
    }
    const int slot = bg_.slot;
    const int keep = cur_;
    if (cur_ != slot) use_slot(slot);
    finish_for(slot);
    KvArchive& a = bg_.a;
    pool_->run([&](int r) {
        if (bg_.piece[r] < 0) return;
        void* b = a.ram->at(r).get();
        bg_.piece[r] = workers_[r]->archive_xfer(slot, a.split, a.n, a.n, &b, kArchiveChunk, true, true,
                                                 bg_.piece[r], budget);
    });
    if (keep != cur_) use_slot(keep);
    if (std::all_of(bg_.piece.begin(), bg_.piece.end(), [](int p) { return p < 0; })) bg_finish(true);
    return false;
}

// Pin the state of the selected slot at pos (a prefix shared by conversations of one kind: the
// end of the system prompt and tools) as a host copy outside the archive LRU. Within
// QE_PREFIX_GB: the least recently used pinned prefix of the lowest priority gives way.
void Engine::pin_prefix(int slot, int pos, unsigned long long affinity) {
    Seq& q = seqs_[slot];
    if (pos < 256 || prefix_budget_ == 0 || q.split > 0) return;  // attached: its prefix exists
    const PrefillState* p = pf_of(slot);
    const int* ids = p ? p->ids.data() : q.seq.data();
    for (auto& a : archives_)
        if (a.pinned && a.sk == q.sk && a.n == pos && std::equal(ids, ids + pos, a.toks.begin())) {
            a.used = ++tick_;
            a.affinity = affinity;
            a.prio = std::max(a.prio, q.prio);
            return;
        }
    const int ng = workers_.size();
    KvArchive a;
    for (int r = 0; r < ng; r++) {
        a.sizes.push_back(workers_[r]->archive_bytes(0, pos));
        a.total += a.sizes.back();
    }
    if (a.total > prefix_budget_) return;
    while (prefix_used_ + a.total > prefix_budget_) {
        auto v = archives_.end();
        for (auto it = archives_.begin(); it != archives_.end(); ++it)
            if (it->pinned && !it->hold && (v == archives_.end() || it->prio < v->prio || (it->prio == v->prio && it->used < v->used)))
                v = it;
        if (v == archives_.end()) return;
        prefix_used_ -= v->total;
        archives_.erase(v);
    }
    a.ram = std::make_shared<Bufs>(&hpool_);
    for (size_t sz : a.sizes) a.ram->add(sz);
    a.toks.assign(ids, ids + pos);
    a.n = pos;
    a.used = ++tick_;
    a.prio = q.prio;
    a.pinned = true;
    a.affinity = affinity;
    a.sk = q.sk;
    pool_->run([&](int r) {
        void* b = a.ram->at(r).get();
        workers_[r]->archive_xfer(slot, 0, pos, pos, &b, kArchiveChunk, true, true);
    });
    prefix_used_ += a.total;
    if (pool_log_)
        printf("archive: pinned prefix %d tokens of slot %d (%.2f GB, pinned %.2f of %.2f GB)\n", pos, slot, a.total / 1e9,
               prefix_used_ / 1e9, prefix_budget_ / 1e9), fflush(stdout);
    archives_.push_back(std::move(a));
    // the shared region: a copy of the rows on the GPU (other slots attach to it instead of
    // copying the prefix into their own regions)
    if (share_prefix_ && find_prefix(ids, pos, q.sk) < 0) {
        std::vector<int> protect = {slot};
        const size_t off = prefix_room(pos, protect);
        if (off == ~(size_t)0) return;
        GpuPrefix p;
        p.toks.assign(ids, ids + pos);
        p.n = pos;
        p.sk = q.sk;
        p.off = off;
        p.used = ++tick_;
        pool_->run([&](int r) { workers_[r]->copy_keys(q.off, rows_of(q), 0, off, pos, 0, 0, pos); });
        if (pool_log_) printf("pool: prefix %d tokens shared at %zu\n", pos, off), fflush(stdout);
        pres_.push_back(std::move(p));
    }
}

// the pinned archive holding the rows [0, split) an archive refers to (nullptr: none, so the
// archive cannot be restored)
const Engine::KvArchive* Engine::prefix_host(const KvArchive& a) const {
    if (!a.split) return &a;
    for (auto& x : archives_)
        if (x.pinned && x.ram && x.sk == a.sk && x.n == a.split && std::equal(x.toks.begin(), x.toks.end(), a.toks.begin())) return &x;
    return nullptr;
}

int Engine::archive_coverage(const int* ids, int n) const {
    int best = 0;
    const long long sk = steer_key();
    for (auto& a : archives_) {
        if (a.sk != sk || (!a.ram && a.path.empty())) continue;
        if (a.split && !prefix_host(a)) continue;
        const int l = lcp_of(a.toks, ids, n);
        if (a.n < n && l >= a.n) best = std::max(best, a.n);
        else
            for (int s : a.subs)
                if (s <= l && s < n) best = std::max(best, s);
    }
    return best;
}

// The position start() would resume this prompt from (see start_begin): the live state of a
// slot it extends, else the longest snapshot whose KV some slot holds, else an archive.
int Engine::probe_cached(const int* ids, int n) const {
    int best = 0;
    const long long sk = steer_key();  // (the selection set before this request's start, when the caller set it)
    for (auto& q : seqs_)
        if (q.sk == sk && q.pos > 0 && q.pos < n && !q.active && lcp_of(q.seq, ids, n) == q.pos) best = std::max(best, q.pos);
    if (best) return best;
    for (auto& s : snaps_) {
        if (s.sk != sk || s.pos >= n || s.pos <= best || lcp_of(s.toks, ids, n) < s.pos) continue;
        bool held = false;
        for (auto& q : seqs_)
            if (q.pos >= s.pos && lcpk(q.seq, q.sk, sk, ids, n) >= s.pos) { held = true; break; }
        if (!held)
            for (auto& a : archives_)
                if (a.n >= s.pos && lcpk(a.toks, a.sk, sk, ids, n) >= s.pos) { held = true; break; }
        if (held) best = s.pos;
    }
    return std::max(best, archive_coverage(ids, n));
}

// Write an archive to the disk cache in the background.
void Engine::disk_enqueue(KvArchive& a) {
    if (a.total > disk_budget_) return;
    disk_make_room(a.total);
    char name[64];
    snprintf(name, sizeof name, "/%016llx_%d.qkv", (unsigned long long)fnv64(a.toks.data(), a.n), a.n);
    a.path = disk_dir_ + name;
    a.disk = std::make_shared<DiskState>();
    disk_used_ += a.total;
    DiskJob j;
    j.ram = a.ram;
    j.sizes = a.sizes;
    j.path = a.path;
    j.st = a.disk;
    std::vector<char>& h = j.header;
    h.insert(h.end(), kArchMagic, kArchMagic + 8);
    put(h, (uint32_t)fingerprint_.size());
    h.insert(h.end(), fingerprint_.begin(), fingerprint_.end());
    put(h, (int32_t)a.n);
    put(h, (uint32_t)a.subs.size());
    for (int p : a.subs) put(h, (int32_t)p);
    put(h, (uint32_t)a.sizes.size());
    for (size_t sz : a.sizes) put(h, (uint64_t)sz);
    const char* t = (const char*)a.toks.data();
    h.insert(h.end(), t, t + (size_t)a.n * 4);
    a.data_off = h.size();
    {
        std::lock_guard<std::mutex> l(disk_mu_);
        disk_q_.push_back(std::move(j));
    }
    disk_cv_.notify_all();
}

void Engine::disk_writer() {
    while (true) {
        DiskJob j;
        {
            std::unique_lock<std::mutex> l(disk_mu_);
            disk_cv_.wait(l, [&] { return disk_stop_ || !disk_q_.empty(); });
            if (disk_q_.empty()) return;  // stopping with nothing left to write
            j = std::move(disk_q_.front());
            disk_q_.pop_front();
            disk_busy_++;
        }
        bool ok = false;
        {
            std::lock_guard<std::mutex> l(j.st->mu);
            if (j.st->cancel) j.st->done = true;
        }
        if (!j.st->done) {
            const std::string tmp = j.path + ".tmp";
            FILE* f = fopen(tmp.c_str(), "wb");
            if (f) {
                ok = fwrite(j.header.data(), 1, j.header.size(), f) == j.header.size();
                for (size_t r = 0; ok && r < j.sizes.size(); r++)
                    ok = fwrite(j.ram->at(r).get(), 1, j.sizes[r], f) == j.sizes[r];
                ok = (fclose(f) == 0) && ok;
            }
            j.ram.reset();
            std::lock_guard<std::mutex> l(j.st->mu);
            if (ok && !j.st->cancel) ok = rename(tmp.c_str(), j.path.c_str()) == 0;
            else ok = false;
            if (!ok) unlink(tmp.c_str());
            j.st->ok = ok;
            j.st->done = true;
            if (!ok && !j.st->cancel) fprintf(stderr, "disk cache: write failed: %s\n", j.path.c_str());
        }
        j.st->cv.notify_all();
        {
            std::lock_guard<std::mutex> l(disk_mu_);
            disk_busy_--;
        }
        disk_cv_.notify_all();
    }
}

void Engine::disk_wait_idle() {
    std::unique_lock<std::mutex> l(disk_mu_);
    disk_cv_.wait(l, [&] { return disk_q_.empty() && disk_busy_ == 0; });
}

// Bring a disk-only archive back into host memory. false: unreadable (the entry is dropped by
// the caller).
bool Engine::archive_load(KvArchive& a) {
    if (a.ram) return true;
    if (a.path.empty()) return false;
    {
        std::unique_lock<std::mutex> l(a.disk->mu);
        a.disk->cv.wait(l, [&] { return a.disk->done; });
        if (!a.disk->ok) return false;
    }
    const int fd = open(a.path.c_str(), O_RDONLY);
    if (fd < 0) return false;
    auto bufs = std::make_shared<Bufs>(&hpool_);
    std::vector<size_t> off(a.sizes.size());
    for (size_t r = 0, o = a.data_off; r < a.sizes.size(); o += a.sizes[r], r++) {
        bufs->add(a.sizes[r]);
        off[r] = o;
    }
    std::atomic<bool> ok{true};
    pool_->run([&](int r) {  // one reader per GPU section (parallel requests to the drive)
        if (!read_all(fd, bufs->at(r).get(), a.sizes[r], off[r])) ok = false;
    });
    close(fd);
    if (!ok) return false;
    utimensat(AT_FDCWD, a.path.c_str(), nullptr, 0);  // recently used (LRU across restarts)
    ram_make_room(a.total, a.prio);
    a.ram = bufs;
    archive_used_ += a.total;
    return true;
}

// Disk cache setup: identity of the model / split (archives of another configuration are
// ignored), index of the existing files (LRU order by modification time), writer thread.
void Engine::disk_init() {
    const char* d = env_str("QE_DISK_CACHE");
    if (!d || !*d) return;
    disk_dir_ = d;
    disk_budget_ = (size_t)((env_double("QE_DISK_CACHE_GB", 200.0)) *
                            (1ull << 30));
    mkdir(disk_dir_.c_str(), 0755);
    std::ostringstream fp;
    char rp[PATH_MAX];
    // v3: archives written before the KV-pool compaction fix (23781c9) may hold overwritten rows
    fp << "v3|" << (realpath(model_dir_.c_str(), rp) ? rp : model_dir_.c_str());
    if (env_str("QE_MTP_FILE")) fp << "|mtp:" << env_str("QE_MTP_FILE");
    std::vector<std::string> files = safetensors_files(model_dir_);  // single-file: same key as before
    files.push_back(model_dir_ + "/model_mtp.safetensors");
    for (const std::string& f : files) {
        struct stat stt;
        if (stat(f.c_str(), &stt) == 0) fp << "|" << stt.st_size << ":" << stt.st_mtime;
    }
    fp << w4_key_;  // mixed precision: other weights
    fp << "|g" << workers_.size() << "|" << (env_str("QE_SPLIT") ? env_str("QE_SPLIT") : "");
    if (kvs_.on()) fp << "|kvsplit" << kvs_.period << "," << kvs_.a01 << "," << kvs_.a12;
    for (auto& w : workers_) fp << "|" << w->archive_bytes(0, 1) << "," << w->archive_bytes(0, 2) << "," << w->snapshot_bytes();
    fingerprint_ = fp.str();

    struct Found { KvArchive a; time_t mt; };
    std::vector<Found> found;
    if (DIR* dir = opendir(disk_dir_.c_str())) {
        while (dirent* e = readdir(dir)) {
            const std::string nm = e->d_name, path = disk_dir_ + "/" + nm;
            if (nm.size() > 4 && nm.compare(nm.size() - 4, 4, ".tmp") == 0) { unlink(path.c_str()); continue; }
            if (nm.size() < 5 || nm.compare(nm.size() - 4, 4, ".qkv") != 0) continue;
            FILE* f = fopen(path.c_str(), "rb");
            if (!f) continue;
            KvArchive a;
            bool ok = false;
            do {
                char mg[8];
                uint32_t fl, ns, ngs;
                int32_t n;
                if (fread(mg, 1, 8, f) != 8 || memcmp(mg, kArchMagic, 8) != 0) break;
                if (fread(&fl, 4, 1, f) != 1 || fl > 65536) break;
                std::string fps(fl, '\0');
                if (fread(&fps[0], 1, fl, f) != fl || fps != fingerprint_) break;
                if (fread(&n, 4, 1, f) != 1 || n <= 0) break;
                if (fread(&ns, 4, 1, f) != 1 || ns > kMaxSubs) break;
                a.subs.resize(ns);
                if (ns && fread(a.subs.data(), 4, ns, f) != ns) break;
                if (fread(&ngs, 4, 1, f) != 1 || ngs != workers_.size()) break;
                bool sz_ok = true;
                for (uint32_t r = 0; r < ngs; r++) {
                    uint64_t sz;
                    if (fread(&sz, 8, 1, f) != 1) { sz_ok = false; break; }
                    sz_ok &= sz == workers_[r]->archive_bytes(0, n) + ns * workers_[r]->snapshot_bytes();
                    a.sizes.push_back(sz);
                    a.total += sz;
                }
                if (!sz_ok) break;
                a.toks.resize(n);
                if (fread(a.toks.data(), 4, n, f) != (size_t)n) break;
                a.n = n;
                a.data_off = ftell(f);
                struct stat stt;
                if (stat(path.c_str(), &stt) != 0 || (size_t)stt.st_size != a.data_off + a.total) break;
                a.path = path;
                a.disk = std::make_shared<DiskState>();
                a.disk->done = a.disk->ok = true;
                found.push_back({std::move(a), stt.st_mtime});
                ok = true;
            } while (false);
            fclose(f);
            (void)ok;
        }
        closedir(dir);
    }
    std::sort(found.begin(), found.end(), [](const Found& x, const Found& y) { return x.mt < y.mt; });
    for (auto& fa : found) {
        fa.a.used = ++tick_;
        disk_used_ += fa.a.total;
        archives_.push_back(std::move(fa.a));
    }
    disk_make_room(0);
    printf("disk cache %s: %zu archives, %.1f GB (budget %.0f GB)\n", disk_dir_.c_str(),
           archives_.size(), disk_used_ / 1e9, disk_budget_ / 1e9);
    disk_thread_ = std::thread([this] { disk_writer(); });
}

// Archive every live slot (e.g. before the server exits) and wait until the files are written.
void Engine::persist() {
    finish();
    const int keep = cur_;
    for (int sl = 0; sl < (int)seqs_.size(); sl++)
        if (seqs_[sl].pos >= archive_min_) {
            use_slot(sl);
            archive_slot(sl);
        }
    use_slot(keep);
    if (!disk_dir_.empty()) disk_wait_idle();
}

void Engine::step(const int* ids, int m, bool want_logits) {
    Seq& q = seqs_[cur_];
    if (q.pos + m > q.max_t) CHECK(ensure_cap(cur_, std::min(q.pos + m + 1, max_ctx_)), "KV pool full");
    CHECK(q.pos + m <= q.max_t, "context full");
    const int pos = q.pos;
    const unsigned* mh = q.constrained ? mask_buffer(cur_) : nullptr;
    pool_->run([&](int r) {
        Worker& w = *workers_[r];
        if (mh) w.upload_mask(cur_, mh);
        CUDA_CHECK(cudaMemcpyAsync(w.tokens_, ids, m * 4, cudaMemcpyHostToDevice, w.stream));
        CUDA_CHECK(cudaMemcpyAsync(w.pos_, &pos, 4, cudaMemcpyHostToDevice, w.stream));
        w.set_pf_p0(pos);
        if (m <= kMaxGemvTokens) w.run(m, want_logits);
        else w.forward(m, want_logits);
        CUDA_CHECK(cudaStreamSynchronize(w.stream));
    });
    q.pos += m;
    last_rows_ = m <= kMaxGemvTokens ? m : 1;
}

int Engine::prefill(const int* ids, int n) {
    for (int i = 0; i < n; i += kChunk) {
        const int m = std::min(kChunk, n - i);
        step(ids + i, m, i + m == n);
    }
    const int m = (n - 1) % kChunk + 1;
    int tok;
    CUDA_CHECK(cudaSetDevice(workers_[0]->device));
    CUDA_CHECK(cudaMemcpy(&tok, workers_[0]->out_tok_ + m - 1, 4, cudaMemcpyDeviceToHost));
    return tok;
}

int Engine::decode(int tok) {
    step(&tok, 1, true);
    int out;
    CUDA_CHECK(cudaSetDevice(workers_[0]->device));
    CUDA_CHECK(cudaMemcpy(&out, workers_[0]->out_tok_, 4, cudaMemcpyDeviceToHost));
    return out;
}

// ---------------------------------------------------------------- activation steering
long long Engine::steer_key() const {
    if (steer_vec_ < 0) return -1;
    uint32_t b;
    memcpy(&b, &steer_scale_, 4);
    return ((((((long long)steer_vec_ * 2 + steer_mode_) * 64 + steer_lo_) * 64 + steer_hi_)) << 32) | b;
}

void Engine::steer_unkey(long long k, int& vec, int& mode, int& lo, int& hi, float& scale) {
    if (k < 0) { vec = -1; mode = 0; lo = 0; hi = -1; scale = 0.f; return; }
    const uint32_t b = (uint32_t)(k & 0xffffffffu);
    memcpy(&scale, &b, 4);
    long long h = k >> 32;
    hi = (int)(h % 64); h /= 64;
    lo = (int)(h % 64); h /= 64;
    mode = (int)(h % 2);
    vec = (int)(h / 2);
}

int Engine::set_steering(int vec, float scale, int mode, int lo, int hi) {
    if (vec == -2) { vec = steer_def_; scale = 1.f; mode = -1; lo = hi = -1; }
    if (vec < -1 || vec >= (int)steer_.size() || !std::isfinite(scale)) return -1;
    if (vec < 0) {
        steer_vec_ = -1; steer_mode_ = 0; steer_lo_ = 0; steer_hi_ = -1; steer_scale_ = 1.f;
        return 0;
    }
    if (!qsteer::steer_resolve(steer_[vec], mode, lo, hi).empty()) return -1;
    steer_vec_ = vec;
    steer_mode_ = mode;
    steer_lo_ = lo;
    steer_hi_ = hi;
    steer_scale_ = scale;
    return 0;
}

void Engine::steer_slot(int slot) {
    if (steer_.empty() || !workers_[0]->steer_loaded()) return;
    int v, md, lo, hi;
    float sc;
    steer_unkey(seqs_.at(slot).sk, v, md, lo, hi, sc);
    pool_->run([&](int r) { workers_[r]->set_steer_slot(slot, v, md, lo, hi, sc); });  // no-op when unchanged
}

int Engine::capture(const int* ids, int n, int r0, float* out) {
    if (!pfs_.empty() || n <= 0 || r0 < 0 || r0 >= n || n + 2 > max_ctx_) return -1;
    finish();
    const int sv = steer_vec_, sm = steer_mode_, sl = steer_lo_, sh = steer_hi_;
    const float ss = steer_scale_;
    set_steering(-1, 1.f, -1, -1, -1);  // unsteered: the point the vectors will act on
    no_reuse_ = true;
    const int slot = start_begin(ids, n, has_mtp() ? 1 : 0, 0, 0, nullptr, 0, nullptr, 0);
    no_reuse_ = false;
    steer_vec_ = sv, steer_mode_ = sm, steer_lo_ = sl, steer_hi_ = sh, steer_scale_ = ss;
    if (slot < 0) return -1;
    pool_->run([&](int r) { workers_[r]->cap_begin(r0); });
    while (!prefill_more_slot(slot, INT_MAX)) {
    }
    start_finish_slot(slot);
    pool_->run([&](int r) { workers_[r]->cap_end(r == 0 ? out : nullptr); });
    release(slot);
    const size_t L = (size_t)cfg_.layers * cfg_.hidden;
    for (size_t k = 0; k < L; k++) out[k] /= (float)(n - r0);
    return 0;
}

void Engine::set_sampling(float temperature, float top_p, int top_k, float min_p,
                          unsigned long long seed) {
    const bool on = temperature > 0.f;
    SampleParams p{on ? temperature : 1.f, top_p > 0.f && top_p <= 1.f ? top_p : 1.f,
                   top_k > 0 && top_k <= kTopK ? top_k : kTopK, min_p, seed, 0};
    // applied to whichever slot the next start() picks (and to the current one now)
    samp_ = p;
    samp_on_ = on;
    for (auto& w : workers_) w->set_sampling(p, on);
    seqs_[cur_].pen = false;
}

void Engine::set_penalty(float presence, float frequency, const int* seed, int n) {
    pool_->run([&](int r) { workers_[r]->set_penalty(presence, frequency, seed, n); });
    seqs_[cur_].pen = presence != 0.f || frequency != 0.f;
}

void Engine::set_no_lookup(int slot, bool off) {
    if (slot >= 0 && slot < (int)seqs_.size()) seqs_[slot].no_lookup = off;
}

void Engine::take_snapshot(const int* ids, int pos, int kind, int replace) {
    const long long sk = seqs_[cur_].sk;  // the state's steering key: the slot's conversation's
    for (auto& s : snaps_)
        if (s.slot == cur_ && s.sk == sk && s.pos == pos && std::equal(s.toks.begin(), s.toks.end(), ids)) {
            s.used = ++tick_;
            s.kind = std::max(s.kind, kind);
            return;
        }
    Snapshot* s = nullptr;
    // a long prompt's rolling checkpoints: overwrite its own oldest one (position `replace`)
    if (replace > 0)
        for (auto& x : snaps_)
            if (x.slot == cur_ && x.sk == sk && x.pos == replace && x.kind == 0 && std::equal(x.toks.begin(), x.toks.end(), ids)) {
                s = &x;
                break;
            }
    if (s) {
    } else if ((int)snaps_.size() < max_snaps_) {
        snaps_.emplace_back();
        s = &snaps_.back();
        {
            std::lock_guard<std::mutex> lk(snap_mu_);
            if (!free_bufs_.empty()) {
                s->bufs = std::move(free_bufs_.back());
                free_bufs_.pop_back();
            }
        }
        if (s->bufs.empty())
            for (auto& w : workers_) {
                void* p;
                CUDA_CHECK(cudaHostAlloc(&p, w->snapshot_bytes(), cudaHostAllocPortable));
                s->bufs.push_back(p);
            }
    } else {  // evict: LRU of the kind at or over its quota (snap_policy.h)
        s = &snaps_[snap_victim(snaps_, kind, snap_reserve0_, [&](const Snapshot& x) {
            const bool ok = x.slot >= 0 && x.slot < (int)seqs_.size();
            const int end = ok ? std::max(seqs_[x.slot].pos, x.slot == cur_ && pf_of(cur_) ? pf_of(cur_)->n : 0) : 0;
            return (ok ? seqs_[x.slot].prio : 1) + (x.pos >= 32768 && x.pos + 8192 >= end);
        })];
    }
    s->slot = cur_;
    s->pos = pos;
    s->kind = kind;
    s->sk = sk;
    s->toks.assign(ids, ids + pos);
    s->used = ++tick_;
    pool_->run([&](int r) { workers_[r]->snapshot_save(s->bufs[r]); });
}

// Slot choice: (1) a slot whose live state the prompt extends, (2) the slot with the longest
// reusable snapshot, unless that would overwrite a different conversation's much longer cache,
// (3) the slot that last served the same conversation key, (4) an empty slot (largest first),
// else the slot with the least cached context (prefixes held by other slots are copied over). Slots must have room for the prompt plus a reasonable answer.
int Engine::pick_slot(const int* ids, int n, int k, unsigned long long affinity) {
    // any idle slot fits: regions grow on demand (the pool decides what is evicted)
    auto fits = [&](const Seq& q, int) { return !q.active && n + k + 1 <= max_ctx_; };
    const long long sk = steer_key();  // cached states of another steering key are not reusable
    int best = -1;
    for (int i = 0; i < (int)seqs_.size(); i++) {
        const Seq& q = seqs_[i];
        if (q.sk == sk && q.pos > 0 && q.pos < n && fits(q, 0) && lcp_of(q.seq, ids, n) == q.pos &&
            (best < 0 || q.pos > seqs_[best].pos))
            best = i;
    }
    if (best >= 0) return best;
    int best_r = 0;
    for (int i = 0; i < (int)seqs_.size(); i++) {
        const Seq& q = seqs_[i];
        if (!fits(q, 0)) continue;
        const bool same = (affinity && q.affinity == affinity) || q.affinity == 0;
        const int kv_ok = lcpk(q.seq, q.sk, sk, ids, n);
        for (auto& sn : snaps_)
            if (sn.slot == i && sn.sk == sk && sn.pos <= kv_ok && sn.pos < n && sn.pos > best_r &&
                sn.pos >= 256 && (same || 2 * sn.pos >= q.pos) && lcp_of(sn.toks, ids, n) >= sn.pos) {
                best_r = sn.pos;
                best = i;
            }
    }
    if (best >= 0) return best;
    for (int extra : {4096, 0}) {
        if (affinity)
            for (int i = 0; i < (int)seqs_.size(); i++)
                if (seqs_[i].affinity == affinity && fits(seqs_[i], extra) &&
                    (best < 0 || seqs_[i].used > seqs_[best].used))
                    best = i;
        if (best >= 0) return best;
        for (int i = 0; i < (int)seqs_.size(); i++)
            if (seqs_[i].used == 0 && fits(seqs_[i], extra) &&
                (best < 0 || seqs_[i].max_t > seqs_[best].max_t))
                best = i;
        if (best >= 0) return best;
        // overwrite the slot whose cached context is cheapest to lose (finished conversations
        // first, the main thread last; then fewest tokens; ties: least recently used), so e.g. a
        // waiting main agent's long context survives parallel subagents
        for (int i = 0; i < (int)seqs_.size(); i++) {
            const Seq &q = seqs_[i];
            if (!fits(q, extra)) continue;
            const Seq* b = best < 0 ? nullptr : &seqs_[best];
            if (!b || q.prio < b->prio || (q.prio == b->prio && (q.pos < b->pos || (q.pos == b->pos && q.used < b->used))))
                best = i;
        }
        if (best >= 0) return best;
    }
    CHECK(false, "prompt of %d tokens does not fit any slot", n);
    return 0;
}

void Engine::observe(int kv, int a) {
    if (adapt_ == 2 && kv > 0) {  // all drafts accepted -> longer, early rejection -> shorter
        if (a == kv) k_heur_ = std::min(k_heur_ + 1, k_);
        else if (a + 2 <= kv) k_heur_ = std::max(k_heur_ - 1, 2);
        return;
    }
    // depth i is tried only if drafts 1..i-1 were accepted; only tried depths decay, so the
    // estimates of depths not explored lately stay at their last value instead of the prior
    for (int i = 1; i <= kv; i++) {
        acc_num_[i] *= 0.95;
        acc_den_[i] *= 0.95;
        if (i <= a) {
            acc_num_[i] += 1;
            acc_den_[i] += 1;
        } else {
            acc_den_[i] += 1;
            break;
        }
    }
}

void Engine::observe_time(int buf) {
    float ms = 0.f;
    if (cudaEventElapsedTime(&ms, tstart_[buf], emit_ev_[buf]) != cudaSuccess || ms <= 0.f) {
        cudaGetLastError();
        return;
    }
    const double x = buf_kv_[buf], y = ms, d = 0.97;
    // a step whose graph was being captured includes host time: skip it (and other outliers)
    if (buf_captured_[buf]) return;
    double c0 = cost0_, c1 = cost1_;
    if (rs_[0] > 8) {
        const double mx = rs_[1] / rs_[0], my = rs_[3] / rs_[0], var = rs_[2] / rs_[0] - mx * mx;
        if (var > 0.25) c1 = std::min(3.0, std::max(0.2, (rs_[4] / rs_[0] - mx * my) / var));
        c0 = std::max(1.0, my - c1 * mx);
    }
    if (y > 2.0 * (c0 + c1 * x) + 2.0) return;
    for (auto& v : rs_) v *= d;
    rs_[0] += 1; rs_[1] += x; rs_[2] += x * x; rs_[3] += y; rs_[4] += x * y;
}

int Engine::next_k() {
    if (!adapt_ || k_ <= 1) return k_;
    if (adapt_ == 2) return std::min(std::max(k_heur_, 1), k_);
    static const bool measured = env_int("QE_COST_MEASURED", 1);
    double c0 = cost0_, c1 = cost1_;
    if (measured && rs_[0] > 8) {
        const double mx = rs_[1] / rs_[0], my = rs_[3] / rs_[0];
        const double var = rs_[2] / rs_[0] - mx * mx;
        if (var > 0.25) c1 = std::min(3.0, std::max(0.2, (rs_[4] / rs_[0] - mx * my) / var));
        c0 = std::max(1.0, my - c1 * mx);
    }
    double best = 0, prod = 1, e = 1;
    int bk = 1;
    for (int k = 1; k <= k_; k++) {
        // prior: optimistic, slowly decaying with depth, so unexplored depths get tried
        const double prior = 0.9 * pow(0.97, k - 1), w0 = 3.0;
        prod *= (acc_num_[k] + w0 * prior) / (acc_den_[k] + w0);
        e += prod;
        const double score = e / (c0 + c1 * k);
        if (score > best) {
            best = score;
            bk = k;
        }
    }
    return bk;
}

int Engine::start(const int* ids, int n, int k, unsigned long long affinity, int hint) {
    return start_mm(ids, n, k, affinity, hint, nullptr, 0, nullptr, 0);
}

int Engine::start_mm(const int* ids, int n, int k, unsigned long long affinity, int hint,
                     const bf16* img, int img_rows, const int* rope3, int delta) {
    if (start_begin(ids, n, k, affinity, hint, img, img_rows, rope3, delta) < 0) return -1;
    while (!prefill_more(INT_MAX)) {
    }
    return start_finish();
}

// Prompt prefill in three parts so that a server can run decode steps of other sequences
// between its chunks: start_begin (slot, prompt cache, chunk plan), prefill_more (next chunks),
// start_finish (first token). Returns the slot, -1 if the pool cannot hold the prompt.
int Engine::start_begin(const int* ids, int n, int k, unsigned long long affinity, int hint,
                        const bf16* img, int img_rows, const int* rope3, int delta) {
    // a batched step queued ahead for the running sequences stays: it finishes before this
    // prompt's chunks on the same streams and the next step_batch returns its tokens
    finish_single();
    CHECK((int)pfs_.size() < kMaxPrefills, "start_begin: %zu prefills already in progress", pfs_.size());
    {
        int cnt = 0;
        for (int i = 0; i < n; i++) cnt += ids[i] >= kImgToken;
        CHECK(cnt == img_rows && (!cnt || (img && rope3)), "image rows %d, embeddings %d", cnt,
              img_rows);
    }
    CHECK(k >= 0 && k <= kMaxSpecK, "k=%d (max %d)", k, kMaxSpecK);
    CHECK(n >= 1, "empty prompt");
    if (!has_mtp()) k = 0;
    k_ = k;
    last_copied_ = 0;
    last_from_archive_ = false;
    if (!concurrent_)
        for (auto& q : seqs_) q.active = false;
    const int kstart = next_k();  // drafts produced at the end of the prompt
    // QE_TIMING=1: one line per start with the time of each phase (slot choice, archiving the
    // slot's previous context, pool room, cache search, restore copy)
    static const bool timing = env_int("QE_TIMING", 0);
    auto now = [] { return std::chrono::steady_clock::now(); };
    auto ms = [](auto a, auto b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
    const auto t0 = now();
    const long long sk = steer_key();  // this request's steering (set_steering): its cached states only
    const int slot = pick_slot(ids, n, k, affinity);
    finish_for(slot);
    const int prev_cur = cur_;
    if (slot != cur_) use_slot(slot);
    const auto t1 = now();
    const int victim = seqs_[slot].pos;
    if (lcpk(seqs_[slot].seq, seqs_[slot].sk, sk, ids, n) < seqs_[slot].pos) archive_slot(slot);
    const auto t2 = now();
    for (auto& w : workers_) w->set_sampling(samp_, samp_on_);
    Seq& q = seqs_[slot];
    q.pen = q.no_lookup = false;  // per reply (set_penalty / set_no_lookup after this start)
    if (q.sk != sk) {  // another key: the old content is archived (above) and not reused
        q.pos = 0;
        q.seq.clear();
        q.sk = sk;
    }
    steer_slot(slot);  // the slot's device steering = its conversation's key (stream-ordered, no graph change)
    const unsigned long long prev_aff = q.affinity;
    q.used = ++tick_;
    q.affinity = affinity;
    q.active = true;
    q.admit = ++tick_;
    q.stale = false;
    q.end = 0;
    q.want_drafts = q.no_ahead = false;
    q.drafts.clear();
    q.rb_epoch = -1;
    epoch_++;

    // Prompt cache: the slot's KV caches hold q.seq at positions [0, q.pos). Resume from the
    // live state if the prompt extends it, else from the longest snapshot of this slot still
    // consistent with both the prompt and the KV contents; at least one prompt token is always
    // recomputed for logits.
    const int kv_ok = no_reuse_ ? 0 : lcp_of(q.seq, ids, n);
    int from = 0;
    // slots holding a prefix of the prompt keep their rows while room is made (unless a pinned
    // prefix / archive in host memory covers about as much of it)
    std::vector<int> protect;
    {
        const int covered = archive_coverage(ids, n);
        for (size_t y = 0; y < seqs_.size(); y++)
            if ((int)y != slot && lcpk(seqs_[y].seq, seqs_[y].sk, sk, ids, n) >= std::max(256, covered + 4096)) protect.push_back(y);
    }
    const int need = std::min(n + 2 * kMaxSpecK + 4, max_ctx_);
    pool_->run([&](int r) { workers_[r]->set_rope_delta(delta); });
    // ---- the source of the prompt's cached part (decided before the room is made: the slot's
    // layout - attached to a shared prefix or not - depends on it)
    enum { LIVE, SNAP, OTHER, ATTACH, ARCH, NONE } mode = NONE;
    Snapshot* best = nullptr;
    // cost: 0 this slot / a shared prefix (no copy), 1 other slot, 2 archive in memory, 3 disk
    int best_src = -1, best_pos = 0, best_cost = 0, best_sub = -1, pre = -1;
    KvArchive* best_arch = nullptr;
    const KvArchive* host_pre = nullptr;  // pinned host copy of the archive's prefix rows
    // the archives restored from stay until the copy is done: materialize_prefix / ensure_cap
    // may evict other slots, whose archiving makes room by dropping (or superseding) archives
    struct Hold {
        const KvArchive* a[2] = {nullptr, nullptr};
        void add(const KvArchive* x) { x->hold++; (a[0] ? a[1] : a[0]) = x; }
        ~Hold() { for (auto x : a) if (x) x->hold--; }
    } held;
    const char* src = "live";
    const auto t3 = now();
    if (kv_ok == q.pos && q.pos > 0 && q.pos < n) {
        from = q.pos;
        mode = LIVE;
        // The state at the live end (prompt + reply) before this turn extends it: a fork of the
        // conversation (Claude Code's subagent-progress summary: the same history plus a short
        // question, arriving right after the reply) or the conversation's own next turn when
        // the fork took this slot first resumes from here instead of recomputing the reply.
        take_snapshot(ids, from, 1);  // kept like a prompt end (the conversation's next turn)
    } else if (!no_reuse_) {
        // Longest snapshot (of any slot) of a prefix of the prompt whose KV entries some slot
        // still holds: this slot itself (no copy), or another one (KV prefix copied over, e.g.
        // parallel subagents sharing the system prompt and tools); else an archive.
        auto consider = [&](int pos, int cost, Snapshot* sn, int src, KvArchive* ar, int sub = -1) {
            if (pos <= 0 || pos >= n) return;
            if (pos < best_pos || (pos == best_pos && cost >= best_cost && (best || best_arch))) return;
            best_pos = pos; best_cost = cost; best = sn; best_src = src; best_arch = ar; best_sub = sub;
        };
        auto arch_cost = [&](const KvArchive& a) { return a.pinned && share_prefix_ ? 0 : a.ram ? 2 : 3; };
        for (auto& sn : snaps_) {
            if (sn.sk != sk || sn.pos >= n || lcp_of(sn.toks, ids, n) < sn.pos) continue;
            // this slot's rows (unless they end inside its shared prefix: see the ARCH case)
            if (sn.pos <= kv_ok && sn.pos >= q.split) { consider(sn.pos, 0, &sn, slot, nullptr); continue; }
            int y0 = -1;
            for (int y = 0; y < (int)seqs_.size() && y0 < 0; y++)
                if (y != slot && seqs_[y].sk == sk && seqs_[y].pos >= sn.pos && lcp_of(seqs_[y].seq, ids, n) >= sn.pos) y0 = y;
            if (y0 >= 0) { consider(sn.pos, 1, &sn, y0, nullptr); continue; }
            for (auto& a : archives_)
                if (a.sk == sk && a.n >= sn.pos && (!a.split || prefix_host(a)) && lcp_of(a.toks, ids, n) >= sn.pos)
                    consider(sn.pos, a.ram ? 2 : 3, &sn, -1, &a);
        }
        for (auto& a : archives_) {  // the archive's own state at its end, or a checkpoint in it
            if (a.sk != sk || (a.split && !prefix_host(a))) continue;
            const int l = lcp_of(a.toks, ids, n), c = arch_cost(a);
            if (a.n < n && l >= a.n) consider(a.n, c, nullptr, -1, &a);
            else
                for (size_t si = 0; si < a.subs.size(); si++)
                    if (a.subs[si] <= l) consider(a.subs[si], c, nullptr, -1, &a, si);
        }
        if (best_arch && !archive_load(*best_arch)) {  // unreadable file: forget it, recompute
            for (auto it = archives_.begin(); it != archives_.end(); ++it)
                if (&*it == best_arch) { drop_archive(it); break; }
            best = nullptr;
            best_arch = nullptr;
        }
        if (best_arch) {
            from = best_pos;
            best_arch->used = ++tick_;
            held.add(best_arch);
            if (best_arch->pinned && share_prefix_ && !best && best_sub < 0) {
                // the whole prefix: attach to its shared region (materialized if needed)
                pre = materialize_prefix(*best_arch, protect);
                mode = pre >= 0 ? ATTACH : ARCH;
            } else {
                mode = ARCH;
                if (best_arch->split) {
                    host_pre = prefix_host(*best_arch);
                    held.add(host_pre);
                    // attach only if the prompt resumes at or past the prefix end: a slot
                    // attached to a prefix longer than its restored part would read the prefix's
                    // rows past `from` and write keys below split at negative own rows; else
                    // the prefix rows [0, from) are copied into the own region (below)
                    pre = share_prefix_ && from >= host_pre->n ? materialize_prefix(*host_pre, protect) : -1;
                }
            }
        } else if (best) {
            from = best_pos;
            mode = best_src == slot ? SNAP : OTHER;
        }
    }
    // ---- the slot's layout for the new content. Its old tokens are forgotten now (they were
    // archived above): until the prefill ends, q.seq / q.pos describe only the rows restored
    // below, so no other request copies rows that are being overwritten.
    // a background archive of this slot still copying its old rows: complete it before they
    // are rewritten (archive_slot above only runs when the prompt does not extend the slot,
    // e.g. not for a retry of the identical prompt, whose restore may change the layout)
    if (mode != LIVE && bg_.slot == slot) bg_finish(true);
    if (mode == SNAP) {
        q.pos = from;
        q.seq.resize(from);
    } else if (mode != LIVE) {
        q.pos = 0;
        q.seq.clear();
    }
    if (mode == OTHER) {
        detach(slot);
        const Seq& y = seqs_[best_src];
        if (y.pre >= 0 && y.split <= from) attach(slot, y.pre);
    } else if (mode == ATTACH || mode == ARCH) {
        detach(slot);
        if (pre >= 0) attach(slot, pre);
    } else if (mode == NONE) {
        detach(slot);
    }
    const auto t4 = now();
    // ---- room for the prompt and the first steps. Protected slots (copy sources of a shared
    // prefix) may be all there is to evict: then evict them too (the prefix comes back from the
    // archive) rather than refusing a prompt can_admit accepted
    // the slot whose rows are copied (OTHER) is protected too; if it still has to go (the second
    // attempt), the prompt is recomputed instead of copying rows of a region that was dropped
    if (mode == OTHER && std::find(protect.begin(), protect.end(), best_src) == protect.end()) protect.push_back(best_src);
    bool room = ensure_cap(slot, need, protect) || ensure_cap(slot, need, {});
    if (room && mode == OTHER && seqs_[best_src].pos < from) {
        if (pool_log_) printf("pool: copy source slot %d evicted: slot %d recomputes from 0\n", best_src, slot), fflush(stdout);
        detach(slot);
        mode = NONE;
        best = nullptr;
        from = 0;
        room = ensure_cap(slot, need, {});
    }
    if (!room) {
        // the pool cannot hold it next to the running sequences: leave the engine as it was
        // (a caller stepping its single running sequence relies on the current slot). The old
        // content of this slot is in the archive (if it was worth it); its rows are dropped.
        if (mode != LIVE && mode != SNAP) {
            detach(slot);
            q.pos = 0;
            q.seq.clear();
        }
        q.active = false;
        q.affinity = prev_aff;
        if (prev_cur >= 0 && prev_cur != slot) {
            use_slot(prev_cur);
            resync(seqs_[prev_cur], prev_cur);
        }
        return -1;
    }
    // ---- bring the cached part in
    if (mode == SNAP || mode == OTHER) {
        best->used = ++tick_;
        if (mode == OTHER) copy_rows(slot, best_src, 0, from);
        pool_->run([&](int r) { workers_[r]->snapshot_restore(best->bufs[r]); });
        src = mode == OTHER ? "other slot" : "snapshot";
    } else if (mode == ATTACH) {
        best_arch->used = ++tick_;
        KvArchive* a = best_arch;
        pool_->run([&](int r) { workers_[r]->snapshot_restore(a->ram->at(r).get()); });  // its delta-net state
        src = "attached";
    } else if (mode == ARCH) {
        best_arch->used = ++tick_;
        if (best) best->used = ++tick_;
        KvArchive* a = best_arch;
        const bool dn_end = best == nullptr && best_sub < 0;
        const int own0 = a->split;  // first key in the archive's own rows
        pool_->run([&](int r) {
            char* base = a->ram->at(r).get();
            void* b = base;
            if (host_pre && q.split == 0) {  // no shared region: the prefix rows into the own region
                void* hb = host_pre->ram->at(r).get();
                workers_[r]->region_xfer(q.off, rows_of(q), 0, 0, host_pre->n, std::min(from, host_pre->n), &hb,
                                         kArchiveChunk, false, false);
            }
            workers_[r]->region_xfer(q.off, rows_of(q), q.split, own0, a->n, std::max(own0, from), &b,
                                     kArchiveChunk, false, dn_end);
            if (best_sub >= 0)
                workers_[r]->snapshot_restore(base + workers_[r]->archive_bytes(own0, a->n) +
                                              best_sub * workers_[r]->snapshot_bytes());
        });
        if (best) pool_->run([&](int r) { workers_[r]->snapshot_restore(best->bufs[r]); });
        src = best_arch->pinned ? "pinned" : "archive";
    } else if (mode == NONE) {
        reset_slot();
        src = "none";
    }
    if (mode == OTHER || mode == ATTACH || mode == ARCH) {  // rows [0, from) now hold the prompt's prefix
        q.pos = from;
        q.seq.assign(ids, ids + from);
    }
    last_copied_ = (mode == OTHER || mode == ATTACH || mode == ARCH) ? from : 0;
    last_from_archive_ = mode == ATTACH || mode == ARCH;
    const auto t5 = now();
    last_reused_ = from;
    if (timing)
        printf("timing: start slot %d n %d from %d (%s): pick %.1f ms, archive previous %d tok %.1f ms, "
               "search + layout %.1f ms, pool room + restore %.1f ms\n", slot, n, from, src, ms(t0, t1), victim,
               ms(t1, t2), ms(t3, t4), ms(t4, t5)), fflush(stdout);
    // Checkpoints besides every 8K tokens and the prompt end: the caller's hint (end of the
    // system prompt) and the point where this prompt leaves the longest known prefix (another
    // conversation diverging at the same place, e.g. a per-project path, can resume there).
    std::vector<int> marks;
    // the hint state (end of the system part) is pinned (pin_prefix) and, with sharing on, its
    // rows become the shared region of that kind of agent
    const int pin = hint;
    PrefillState pf;
    if (pin > from && pin < n) {
        marks.push_back(pin);
        pf.pin = pin;
    }
    // the prompt-end state is saved at n - 1 (the last token prefilled in a pass of its own): a
    // state at n cannot serve the identical prompt again (at least one token is recomputed for
    // the logits), so a repeat used to fall back to the last checkpoint. Only for long prefills:
    // the extra one-token pass costs about one decode step.
    pf.tail_from = tail_ckpt_ > 0 ? n - tail_ckpt_ : n;
    pf.last_ck = from;
    if (end_cut_ > 0 && n - from >= end_cut_ && n - 1 > from && !img_rows) {
        marks.push_back(n - 1);
        pf.end_snap = n - 1;
    }
    {
        int div = 0;
        for (auto& q2 : seqs_) div = std::max(div, lcpk(q2.seq, q2.sk, sk, ids, n));
        for (auto& sn : snaps_) div = std::max(div, lcpk(sn.toks, sn.sk, sk, ids, n));
        if (div > from + 512 && div < n) marks.push_back(div);
    }
    std::sort(marks.begin(), marks.end());
    // image rows: the device-side ids index the image table (rows [from, n) filled from img)
    std::vector<int> dev_ids;
    if (img_rows > 0) {
        if ((size_t)img_rows > img_cap_) {
            if (img_table_) CUDA_CHECK(cudaFreeHost(img_table_));
            img_cap_ = std::max<size_t>(img_rows, 2 * img_cap_);
            CUDA_CHECK(cudaHostAlloc(&img_table_, img_cap_ * cfg_.hidden * 2,
                                     cudaHostAllocMapped | cudaHostAllocPortable));
            pool_->run([&](int r) {
                CUDA_CHECK(cudaSetDevice(workers_[r]->device));
                set_image_table(img_table_);
                set_image_table_spec(img_table_);
            });
        }
        dev_ids.assign(ids, ids + n);
        for (int i = 0, j = 0; i < n; i++)
            if (ids[i] >= kImgToken) {
                if (i >= from)
                    memcpy(img_table_ + (size_t)j * cfg_.hidden, img + (size_t)j * cfg_.hidden, (size_t)cfg_.hidden * 2);
                dev_ids[i] = kImgToken + j++;
            }
    }
    pf.slot = slot;
    pf.n = n;
    pf.i = from;
    pf.kstart = kstart;
    pf.affinity = affinity;
    pf.reused = last_reused_;
    pf.copied = last_copied_;
    pf.from_archive = last_from_archive_;
    pf.ids.assign(ids, ids + n);
    pf.dev_ids = std::move(dev_ids);
    if (img_rows) pf.rope3.assign(rope3, rope3 + 3 * (size_t)n);
    pf.marks = std::move(marks);
    pfs_.push_back(std::move(pf));
    if (seqs_[slot].constrained) constrain(slot, false);  // a new request: unconstrained unless set again
    return slot;
}

// Prefills up to `budget` more prompt tokens (whole chunks: <= kChunk, cut at the marks and at
// every 8192 tokens, where snapshots are taken). true once the prompt is done.
bool Engine::prefill_more(int budget) {
    return prefill_more_slot(pfs_.empty() ? -1 : pfs_.back().slot, budget);
}

bool Engine::prefill_more_slot(int slot, int budget) {
    PrefillState* pp = pf_of(slot);
    if (!pp || pp->i >= pp->n) return true;
    epoch_++;  // the chunks overwrite the step buffers (trim)
    PrefillState& pf = *pp;
    finish_for(pf.slot);  // a single step queued ahead (a batched one of other sequences stays)
    if (cur_ != pf.slot) use_slot(pf.slot);
    const int* pids = pf.dev_ids.empty() ? pf.ids.data() : pf.dev_ids.data();
    const int* r3 = pf.rope3.empty() ? nullptr : pf.rope3.data();
    int done = 0;
    while (pf.i < pf.n && done < budget) {
        const int i = pf.i, n = pf.n;
        int m = std::min(std::min(kChunk, n - i), std::max(budget - done, 1));
        m = std::min(m, (i / ckpt_every_ + 1) * ckpt_every_ - i);
        for (int mk : pf.marks)
            if (mk > i && mk < i + m) m = mk - i;
        // a chunk just past a multiple of 256 rows costs a whole extra block of the prefill
        // GEMMs (1036 tokens: 309 ms; 1024 + 12 in a small pass of their own: 275 ms)
        static const int tail = env_int("QE_PREFILL_TAIL", 32);
        if (m > 256 && m % 256 > 0 && m % 256 <= tail) m -= m % 256;
        pf.last_m = m;
        pool_->run([&](int r) { workers_[r]->prefill_chunk(pids, i, m, n, pf.kstart, r3); });
        pf.i += m;
        done += m;
        const bool mark = std::find(pf.marks.begin(), pf.marks.end(), pf.i) != pf.marks.end();
        if (pf.i < n && (mark || pf.i % ckpt_every_ == 0 || (pf.i >= pf.tail_from && pf.i - pf.last_ck >= kChunk))) {
            if (mark) {
                take_snapshot(pf.ids.data(), pf.i, pf.i == pf.end_snap ? 1 : 0);
            } else {  // periodic / tail: rolling, at most ckpt_keep_ per prefill
                int rep = 0;
                if ((int)pf.rolling.size() >= ckpt_keep_) {
                    rep = pf.rolling.front();
                    pf.rolling.erase(pf.rolling.begin());
                }
                take_snapshot(pf.ids.data(), pf.i, 0, rep);
                pf.rolling.push_back(pf.i);
            }
            pf.last_ck = pf.i;
        }
        if (pf.i == pf.pin) pin_prefix(pf.slot, pf.i, pf.affinity);
    }
    return pf.i >= pf.n;
}

int Engine::start_finish() {
    CHECK(!pfs_.empty(), "start_finish: no prefill in progress");
    return start_finish_slot(pfs_.back().slot);
}

int Engine::start_finish_slot(int slot) {
    PrefillState* pp = pf_of(slot);
    CHECK(pp && pp->i >= pp->n, "start_finish before the prompt is done");
    PrefillState& pf = *pp;
    if (cur_ != pf.slot) use_slot(pf.slot);
    Seq& q = seqs_[pf.slot];
    const int n = pf.n, last_m = pf.last_m;
    last_rows_ = last_m <= kMaxGemvTokens ? last_m : 1;
    q.pos = n;
    q.seq = pf.ids;
    if (!pf.end_snap) take_snapshot(pf.ids.data(), n, 1);  // else saved at n - 1 (start_begin)
    int tok;
    CUDA_CHECK(cudaSetDevice(workers_[0]->device));
    CUDA_CHECK(cudaMemcpy(&tok, workers_[0]->out_tok_ + last_m - 1, 4, cudaMemcpyDeviceToHost));
    q.last_tok = q.pending = tok;
    if (q.pen) pool_->run([&](int r) { workers_[r]->pen_host(&tok, 1, n); });  // the first token sits at n
    q.k_avail = pf.kstart;
    lk_slot_ = -1;
    lkb_inval(pf.slot);
    if (q.constrained) {  // the drafts made at the prompt end, for the caller to check
        int* io = spec_io(pf.slot);
        io[16] = 0;
        io[0] = has_mtp() ? pf.kstart : 0;
        if (io[0] > 0)
            CUDA_CHECK(cudaMemcpy(io + 1, workers_[0]->spec_.tokens + 1, io[0] * 4, cudaMemcpyDeviceToHost));
    }
    last_reused_ = pf.reused;
    last_copied_ = pf.copied;
    last_from_archive_ = pf.from_archive;
    pfs_.erase(pfs_.begin() + (pp - pfs_.data()));
    return tok;
}

void Engine::launch_step(int buf, int base, int lk) {
    // a grammar-constrained sequence verifies the caller's drafts (spec_io) with per-row masks;
    // the drafts the step makes go back to the caller. lk > 0: a lookup step verifying lk device lookup drafts.
    const bool con = seqs_[cur_].constrained;
    const int kv0 = con ? con_kv(cur_, k_) : lk > 0 ? lk : seqs_[cur_].k_avail, kd = next_k();
    int kv = clamp_kv(seqs_[cur_], kv0, base);
    const bool look = lk > 0 && kv > 0;
    if (lk > 0 && !look) kv = 0;
    buf_lk_[buf] = look;
    buf_kv_[buf] = kv;
    buf_clamped_[buf] = kv < kv0;
    buf_kd_[buf] = kd;
    epoch_++;
    seqs_[cur_].launches++;
    const bool wd = seqs_[cur_].want_drafts && kd > 0;
    seqs_[cur_].k_avail = kd;
    buf_captured_[buf] = !workers_[0]->has_spec_graph(kv, kd, look);
    const unsigned* mh = con ? mask_buffer(cur_) : nullptr;
    int* io = con ? spec_io(cur_) : nullptr;
    if (io) io[0] = kd;
    // decode attention row split (bit-identical): one sequence while its keys (with the launch-ahead slack)
    // stay in the ~38-split region
    const bool rs = seqs_[cur_].pos + 2 * (kMaxSpecK + 1) + 8 <= dec38_max();
    for (auto& wp : workers_) wp->attn_rs_ = rs;
    buf_captured_[buf] = !workers_[0]->has_spec_graph(kv, kd, look);
    pool_->run([&](int r) {
        Worker& w = *workers_[r];
        if (io) {
            w.upload_mask(cur_, mh, kv + 1);
            w.upload_drafts(cur_, io + 17, kv);
        }
        if (r == 0) CUDA_CHECK(cudaEventRecord(tstart_[buf], w.stream));
        w.run_spec(kv, kd, true, look);
        if (r == 0) {
            CUDA_CHECK(cudaMemcpyAsync(emit_host_ + buf * 64, w.spec_.emit, (w.lk_on() ? 32 : kv + 2) * 4,
                                       cudaMemcpyDeviceToHost, w.stream));
            if (wd)  // the next step's drafts (stop sequences: Seq::drafts)
                CUDA_CHECK(cudaMemcpyAsync(emit_host_ + buf * 64 + 32, w.spec_.tokens + 1, kd * 4,
                                           cudaMemcpyDeviceToHost, w.stream));
            if (io && kd > 0)
                CUDA_CHECK(cudaMemcpyAsync(io + 1, w.spec_.tokens + 1, kd * 4, cudaMemcpyDeviceToHost, w.stream));
            CUDA_CHECK(cudaEventRecord(emit_ev_[buf], w.stream));
        }
    });
}

void Engine::finish_single() {
    if (!inflight_) return;
    pool_->run([&](int r) { workers_[r]->undo_spec(); });
    inflight_ = false;
    seqs_[cur_].stale = true;
    lk_slot_ = -1;  // the undone step appended its tokens to the device history
}

void Engine::finish_batch() {  // undo the batched step queued ahead
    if (!binflight_) return;
    std::vector<int> live;
    for (int sl : bslots_)
        if (batch_live(sl)) live.push_back(sl);
    pool_->run([&](int r) {
        for (int sl : live) workers_[r]->undo_slot(sl);
    });
    for (int sl : live) {
        seqs_[sl].stale = true;
        if (lkb_on_) lkb_dirty_[sl] = 1;  // the undone step appended its tokens to the slot's history
    }
    binflight_ = false;
    bdrop_.clear();
}

void Engine::finish() {
    finish_single();
    finish_batch();
}

bool Engine::batch_live(int slot) const {
    return binflight_ && std::find(bslots_.begin(), bslots_.end(), slot) != bslots_.end() &&
           std::find(bdrop_.begin(), bdrop_.end(), slot) == bdrop_.end();
}

void Engine::finish_for(int slot) {
    finish_single();
    if (batch_live(slot)) finish_batch();
}

void Engine::resync(Seq& q, int slot) {
    if (!q.stale || q.paused) return;
    pool_->run([&](int r) { workers_[r]->spec_resync(slot, q.pos, q.pending); });
    q.k_avail = 0;
    q.stale = false;
    if (slot == cur_) lk_slot_ = -1;
}

void Engine::release(int slot) {
    if (slot == cur_) finish_single();
    if (batch_live(slot)) {  // undo the queued batched step for this sequence only
        pool_->run([&](int r) { workers_[r]->undo_slot(slot); });
        seqs_[slot].stale = true;
        bdrop_.push_back(slot);
    }
    lkb_inval(slot);
    if (seqs_.at(slot).active) n_release_++;  // room may have freed: paused slots try to resume
    if (seqs_[slot].paused) {  // released while paused: its archive stays (the next turn resumes from it)
        auto it = paused_.find(slot);
        it->second.arch->hold--;
        paused_.erase(it);
        seqs_[slot].paused = false;
        detach(slot);
    }
    seqs_.at(slot).active = false;
    seqs_[slot].full = false;
    seqs_[slot].end = 0;
    seqs_[slot].want_drafts = seqs_[slot].no_ahead = false;
    seqs_[slot].drafts.clear();
    if (seqs_[slot].constrained) constrain(slot, false);
}

bool Engine::trim(int slot, int pos) {
    Seq& q = seqs_.at(slot);
    if (pos >= q.pos) return true;  // the slot holds no token past pos (the last one is pending)
    // the rows of a single-sequence step are in buffers every launch overwrites; a batched step's
    // stash is the slot's own (only its own next step changes it)
    const bool valid = q.rb_batched ? q.rb_epoch >= 0 && q.rb_launch == q.launches : q.rb_epoch == epoch_;
    if (!valid || pos < q.rb_pos || pos >= q.rb_pos + q.rb_rows || (q.active && batch_live(slot)))
        return false;
    // batched steps and deferred single steps stash their rows; with QE_DN_DEFER=0 a batched
    // step's rows are gone (its replay buffers are shared)
    const bool stash = q.rb_batched || dn_defer1_on();
    if (q.rb_batched && !dn_defer_on()) return false;
    if (slot == cur_ && inflight_) return false;  // (a step queued ahead: rb_epoch says so too)
    const int r = pos - q.rb_pos;
    pool_->run([&](int w) { workers_[w]->trim_slot(slot, r, stash); });
    q.pending = q.seq[pos];
    q.seq.resize(pos);
    q.pos = pos;
    q.stale = true;
    if (slot == cur_) lk_slot_ = -1;
    lkb_inval(slot);
    q.k_avail = 0;
    q.rb_epoch = -1;
    return true;
}

void Engine::resume(int slot) {
    if (slot != cur_) {
        finish();
        use_slot(slot);
    }
    resync(seqs_[slot], slot);
}

static constexpr int kMaxBatch = kMaxSegs;  // sequences per batched step

void Engine::launch_batch(int buf, bool ahead) {
    lk_slot_ = -1;  // batched steps do not keep the lookup history
    const int n = bslots_.size();
    int kvs[kMaxBatch];
    // grammar-constrained sequences: (index, masks, spec_io) - the caller's drafts and masks
    std::vector<std::tuple<int, const unsigned*, int*>> con;
    for (int i = 0; i < n; i++) {
        Seq& q = seqs_[bslots_[i]];
        const int kv0 = q.constrained ? con_kv(bslots_[i], bcap_) : std::min(q.k_avail, bcap_);
        // a step queued ahead runs after the one in flight: the slot's position then is at most
        // q.pos + that step's drafts + 1 (step_batch keeps a slot out unless a token fits)
        int base = q.pos;
        if (ahead) {
            const auto& f = bstep_[buf ^ 1];
            const auto it = std::find(f.begin(), f.end(), bslots_[i]);
            if (it != f.end()) base += bkvs_[buf ^ 1][it - f.begin()] + 1;
        }
        // (the room left after a lookup step in flight: its rows may exceed the drafts step_batch made room for)
        kvs[i] = bkvs_[buf][i] = std::min(clamp_kv(q, kv0, base), std::max(0, q.max_t - base - 1));
        bclamp_[buf][i] = kvs[i] < kv0;
        if (q.constrained) con.emplace_back(i, mask_buffer(bslots_[i]), spec_io(bslots_[i]));
    }
    // lookup drafts (QE_LKB): slots in lookup mode verify the drafts their history found at the end of their last
    // step (on the cards: no host round trip, the step queued ahead keeps its lookahead) within a row budget
    int lkr[kMaxBatch] = {}, kav[kMaxBatch] = {};
    bool any_lkr = false;
    if (lkb_on_) {
        // histories of the slots not in the step in flight: the host's tokens are exact, upload what the cards lack
        // the tail goes through a pinned per-slot buffer: a pageable copy may block the host thread behind the
        // step in flight (no blocking host calls in a unit whose cards run collectives)
        std::vector<std::pair<int, int>> sync;  // (slot, tail tokens), from = lkb_len_ before
        std::vector<int> from;
        for (int i = 0; i < n; i++) {
            const int s = bslots_[i];
            const auto& f = bstep_[buf ^ 1];
            if (ahead && std::find(f.begin(), f.end(), s) != f.end()) continue;  // in flight: its history follows
            Seq& q = seqs_[s];
            const int hn = (int)q.seq.size() + 1;
            int fr = lkb_len_[s];
            if (fr < 0 || fr > hn) fr = 0;
            if (fr == hn && !lkb_dirty_[s] && lkb_len_[s] >= 0) {
                static const bool chk = env_int("QE_LKB_CHECK", 0);
                if (chk) {  // debug: the cards' history of a slot that needs no upload equals the host's tokens
                    Worker& w0 = *workers_[0];
                    CUDA_CHECK(cudaSetDevice(w0.device));
                    CUDA_CHECK(cudaStreamSynchronize(w0.stream));
                    const LookupDev& d = w0.slot_lk(s);
                    int dn = 0;
                    CUDA_CHECK(cudaMemcpy(&dn, d.hn, 4, cudaMemcpyDeviceToHost));
                    std::vector<int> h(std::max(dn, 1));
                    if (dn > 0) CUDA_CHECK(cudaMemcpy(h.data(), d.hd, (size_t)dn * 4, cudaMemcpyDeviceToHost));
                    bool ok = dn == hn && h[hn - 1] == q.pending;
                    for (int j = 0; ok && j + 1 < hn; j++) ok = h[j] == q.seq[j];
                    CHECK(ok, "QE_LKB_CHECK: slot %d device history %d tokens != host %d", s, dn, hn);
                }
                continue;
            }
            if ((int)lkb_pin_.size() <= s) lkb_pin_.resize(seqs_.size(), nullptr);
            if (!lkb_pin_[s]) CUDA_CHECK(cudaHostAlloc((void**)&lkb_pin_[s], (size_t)(kMaxPos + 64) * 4, cudaHostAllocPortable));
            int nt = 0;
            for (int j = fr; j < hn && j < kMaxPos + 64; j++) lkb_pin_[s][nt++] = j + 1 < hn ? q.seq[j] : q.pending;
            sync.emplace_back(s, nt);
            from.push_back(fr);
            lkb_len_[s] = hn;
            lkb_dirty_[s] = 0;
        }
        if (!sync.empty())
            pool_->run([&](int r) {
                for (size_t j = 0; j < sync.size(); j++)
                    workers_[r]->lkb_sync(sync[j].first, lkb_pin_[sync[j].first], from[j], from[j] + sync[j].second);
            });
        // row budget: the step may grow to lkb_rows_ rows (never past kMaxBatchRows / kMaxSegs segments)
        int rows = 0, nlk = 0, base_lk = 0;
        for (int i = 0; i < n; i++) rows += kvs[i] + 1;
        bool cand[kMaxBatch] = {};
        for (int i = 0; i < n; i++) {
            const Seq& q = seqs_[bslots_[i]];
            if (!lkb_mode_[bslots_[i]] || q.constrained || q.want_drafts || q.no_lookup) continue;
            cand[i] = true;
            nlk++;
            base_lk += kvs[i] + 1;
        }
        // grammar-constrained slots with more planned drafts than the step's cap (forced tokens, the server's lookup
        // drafts: the model's own drafts never exceed the cap) take them from the same budget first
        {
            const int target = std::min(kMaxBatchRows, std::max(rows, lkb_rows_));
            int spare = target - rows;
            for (int i = 0; i < n && spare > 0; i++) {
                Seq& q = seqs_[bslots_[i]];
                if (!q.constrained) continue;
                int base = q.pos;
                {
                    const auto& f = bstep_[buf ^ 1];
                    const auto it = std::find(f.begin(), f.end(), bslots_[i]);
                    if (ahead && it != f.end()) base += bkvs_[buf ^ 1][it - f.begin()] + 1;
                }
                const int want = std::min(clamp_kv(q, con_kv(bslots_[i], kMaxSpecK), base),
                                          std::max(0, q.max_t - base - 1));
                if (want <= kvs[i]) continue;
                const int add = std::min(want - kvs[i], spare);
                kvs[i] = bkvs_[buf][i] = kvs[i] + add;
                spare -= add;
                rows += add;
            }
        }
        if (nlk > 0) {
            // greedy by the slots' recent lookup acceptance: each lookup slot replaces its kv drafts by Li, the extra
            // rows come from the budget (target - rows); slots that do not fit keep their MTP drafts
            const int target = std::min(kMaxBatchRows, std::max(rows, lkb_rows_));
            int spare = target - rows;
            static const float lkb_mul = env_double("QE_LKB_MUL", 1.5f);
            static const bool lkq = env_int("QE_LKB_Q", 1);
            std::vector<int> order;
            for (int i = 0; i < n; i++)
                if (cand[i]) order.push_back(i);
            std::stable_sort(order.begin(), order.end(), [&](int x, int y) {
                return lkb_ema_[bslots_[x]] > lkb_ema_[bslots_[y]];
            });
            int segs = n;
            for (int i : order) {
                Seq& q = seqs_[bslots_[i]];
                int base = q.pos;
                {
                    const auto& f = bstep_[buf ^ 1];
                    const auto it = std::find(f.begin(), f.end(), bslots_[i]);
                    if (ahead && it != f.end()) base += bkvs_[buf ^ 1][it - f.begin()] + 1;
                }
                int Li = clamp_kv(q, std::min(kMaxLookupK, q.max_t - base - (kMaxSpecK + 1) - 2), base);
                // rows as the slot's recent lookup steps accepted (EMA x 1.5 + 2: room for a longer run)
                if (lkb_mul > 0) Li = std::min(Li, (int)(lkb_ema_[bslots_[i]] * lkb_mul + 2.f));
                Li = std::min(Li, kvs[i] + spare);
                if (Li + 1 > kMaxGemvTokens && segs + 1 > kMaxSegs) Li = kMaxGemvTokens - 1;
                // few distinct shapes (every new combination of rows is a graph capture + instantiation, ~20 ms):
                // 4, 7 (8 rows: one attention segment) or 12 drafts (QE_LKB_Q=0: any count)
                if (lkq) Li = Li >= 12 ? 12 : Li >= 7 ? 7 : Li >= 4 ? 4 : 0;
                if (Li < lkb_min_ || Li <= kvs[i]) continue;
                if (Li + 1 > kMaxGemvTokens) segs++;
                spare -= Li - kvs[i];
                lkr[i] = Li;
                kav[i] = q.k_avail;
                kvs[i] = bkvs_[buf][i] = Li;
                bclamp_[buf][i] = false;
                any_lkr = true;
            }
        }
    }
    for (int i = 0; i < n; i++) blkr_[buf][i] = lkr[i];
    bstep_[buf] = bslots_;
    epoch_++;
    for (int s : bslots_) seqs_[s].launches++;
    const int kd = std::min(next_k(), bcap_);
    bkd_[buf] = kd;
    for (int i = 0; i < n; i++) seqs_[bslots_[i]].k_avail = kd;
    for (auto& c : con) std::get<2>(c)[0] = kd;
    const std::vector<int> kv(kvs, kvs + n), lkv(lkr, lkr + n), kavv(kav, kav + n);
    pool_->run([&](int r) {
        Worker& w = *workers_[r];
        for (auto& [i, mh, io] : con) {
            w.upload_mask(bslots_[i], mh, kvs[i] + 1);
            w.upload_drafts(bslots_[i], io + 17, kvs[i]);
        }
        w.run_spec_batch(bslots_, kv, kd, any_lkr ? &lkv : nullptr, any_lkr ? &kavv : nullptr);
        if (r == 0) {
            for (int i = 0; i < n; i++)
                CUDA_CHECK(cudaMemcpyAsync(batch_host_ + (buf * kMaxBatch + i) * 64,
                                           w.slot_spec(bslots_[i]).emit, (lkb_on_ ? 32 : kvs[i] + 2) * 4,
                                           cudaMemcpyDeviceToHost, w.stream));
            for (int i = 0; i < n; i++)
                if (kd > 0 && seqs_[bslots_[i]].want_drafts && !seqs_[bslots_[i]].constrained)
                    CUDA_CHECK(cudaMemcpyAsync(batch_host_ + (buf * kMaxBatch + i) * 64 + 32,
                                               w.slot_spec(bslots_[i]).tokens + 1, kd * 4,
                                               cudaMemcpyDeviceToHost, w.stream));
            for (auto& [i, mh, io] : con)
                if (kd > 0)
                    CUDA_CHECK(cudaMemcpyAsync(io + 1, w.slot_spec(bslots_[i]).tokens + 1, kd * 4,
                                               cudaMemcpyDeviceToHost, w.stream));
            CUDA_CHECK(cudaEventRecord(bev_[buf], w.stream));
        }
    });
}

void Engine::step_batch(const int* slots, int n, int* out, int* cnt) {
    CHECK(n >= 1 && n <= kMaxBatch && k_ > 0, "step_batch n=%d k=%d", n, k_);
    finish_single();  // the single-sequence step queued ahead, if any
    if (!batch_host_) {
        CUDA_CHECK(cudaSetDevice(workers_[0]->device));
        CUDA_CHECK(cudaMallocHost(&batch_host_, 2 * kMaxBatch * 64 * 4));
        for (auto& e : bev_) CUDA_CHECK(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
    }
    // rows per batched step (up to 8 sequences): 24 = 2 MTP drafts each at 8 sequences, +14% tok/s
    // over 16 rows (1 draft) at ctx 0 and +15% at 19K, +28-32% at 6 sequences; 32 rows (3 drafts at
    // 8) costs 0.7 ms per extra row and loses at ctx 0
    static const int rows = env_int("QE_BATCH_ROWS", 24);
    const std::vector<int> sl(slots, slots + n);
    // drafts per sequence. More than 8 sequences: as many rows per sequence as at 8 (rows / 8, at
    // least 2 = one draft each) while they fit in kMaxBatchRows: 2 drafts up to 10 sequences with 24
    // rows (9 x 19K: 27 rows 990 tok/s vs 878 with 18), 1 draft up to 16, none beyond
    // QE_BATCH_MAXROWS caps the rows of a step (32 = the policy before the wide GEMV: no drafts beyond 16
    // sequences); 33..64 rows run the wide kernel (gemv_w4hx)
    static const int maxrows = std::max(8, std::min(kMaxBatchRows, env_int("QE_BATCH_MAXROWS", 64)));
    // beyond 10 sequences one draft each (2 rows): 2 drafts at 16 sequences (48 rows) measured -13 %, one draft
    // at 24 / 32 sequences (48 / 64 rows) +13 / +24 % over none
    const int brows = std::min(maxrows, n <= 8 ? rows : n * (n <= 10 ? std::max(2, rows / 8) : 2));
    const int cap = std::max(0, std::min(kMaxSpecK, brows / n - 1));
    // index of each sequence in the step queued ahead (-1: not in it). The set changed: the queued
    // step stays if every sequence still in it is in the new set (released ones were undone alone
    // in release(), new ones get their first tokens from the next step); else it is undone.
    std::vector<int> from(n, -1);
    if (binflight_) {
        bool keep = true;
        for (size_t j = 0; j < bslots_.size() && keep; j++) {
            if (!batch_live(bslots_[j])) continue;
            const auto it = std::find(sl.begin(), sl.end(), bslots_[j]);
            if (it == sl.end()) keep = false;
            else from[it - sl.begin()] = j;
        }
        for (int i = 0; i < n && keep; i++) {
            const Seq& q = seqs_.at(sl[i]);
            if (from[i] < 0 && !q.full && !q.paused && q.pos + 2 * (cap + 1) + 1 > q.max_t) keep = false;  // needs to grow
            if (q.paused && paused_.at(sl[i]).seen != n_release_) keep = false;  // room may have freed: try to resume
        }
        if (!keep) {
            finish_batch();
            std::fill(from.begin(), from.end(), -1);
        } else {
            std::vector<int> live;  // the new set without the sequences that have no room (or are paused)
            for (int i = 0; i < n; i++)
                if (!seqs_[sl[i]].full && !seqs_[sl[i]].paused) live.push_back(sl[i]);
            if (live != bslots_) {
                bslots_ = live;
                bdrop_.clear();
                bcap_ = cap;
                for (int i = 0; i < n; i++)
                    if (from[i] < 0 && !seqs_[sl[i]].full && !seqs_[sl[i]].paused) {
                        Seq& q = seqs_[sl[i]];
                        CHECK(q.active, "slot %d not active", sl[i]);
                        resync(q, sl[i]);
                    }
            }
        }
    }
    // sequences without room (Seq::full, set here once and cleared by release): their end token on
    // this and every later call that still passes them, never in a launched step
    std::vector<char> full(n, 0);
    if (!binflight_) {
        std::vector<int> run;
        // paused sequences resume first (in admission order) when room may have freed: a slot was
        // released since they last tried, or every other one of the set is paused / ended
        bool lone = true;
        for (int i = 0; i < n; i++) lone &= seqs_[sl[i]].paused || seqs_[sl[i]].full;
        std::vector<int> order(n);
        for (int i = 0; i < n; i++) order[i] = i;
        // resume queue: highest prio first, then in pause order (FIFO); strictly in that order (a
        // later one never overtakes the head, so none starves), as many as fit
        std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
            const Seq &x = seqs_[sl[a]], &y = seqs_[sl[b]];
            if (x.paused != y.paused) return x.paused;
            if (!x.paused) return false;
            return x.prio != y.prio ? x.prio > y.prio : paused_.at(sl[a]).at < paused_.at(sl[b]).at;
        });
        bool blocked = false;
        for (int i : order) {
            Seq& q = seqs_.at(sl[i]);
            CHECK(q.active, "slot %d not active", sl[i]);
            if (q.paused) {
                Paused& p = paused_.at(sl[i]);
                if (!blocked && (lone || p.seen != n_release_)) blocked = !unpause_slot(sl[i]);
                if (q.paused) p.seen = n_release_;  // (tried again after the next release)
            }
            if (!q.full && !q.paused) {
                resync(q, sl[i]);
                if (q.pos + 2 * (cap + 1) + 1 > q.max_t &&
                    !room_or_pause(sl[i], std::min(q.pos + 2 * (cap + 1) + 1, max_ctx_)) && !q.paused)
                    q.full = true;  // no room left even alone: this sequence ends, the others step without it
            }
        }
        // (a sequence may pause another one of the set that was checked before it)
        for (int i = 0; i < n; i++) {
            Seq& q = seqs_[sl[i]];
            if (q.full || q.paused) {
                full[i] = 1;
                cnt[i] = q.paused ? 0 : 1;  // paused: no tokens (the stream stalls), ended: its end token
                if (!q.paused) out[i * 64] = cfg_.eos;
                continue;
            }
            CHECK(q.pos + cap + 1 <= q.max_t, "context full");
            from[i] = run.size();
            run.push_back(sl[i]);
        }
        if (run.empty()) return;
        bslots_ = run;
        bdrop_.clear();
        bcap_ = cap;
        launch_batch(bbuf_, false);
        binflight_ = true;
    } else {
        for (int i = 0; i < n; i++)
            if (seqs_[sl[i]].full || seqs_[sl[i]].paused) {
                full[i] = 1;
                cnt[i] = seqs_[sl[i]].paused ? 0 : 1;
                if (!seqs_[sl[i]].paused) out[i * 64] = cfg_.eos;
            }
    }
    // queue the next step before waiting for this one (if two steps still fit every slot)
    bool ahead = lookahead_;
    for (int i = 0; i < n; i++)
        if (!full[i]) ahead &= seqs_[sl[i]].pos + 2 * (bcap_ + 1) <= seqs_[sl[i]].max_t;
    if (ahead) {
        // a grammar-constrained sequence in the step waited for now has no mask for its next token
        // yet: it stays out of the step queued ahead (and joins the one after), so the other
        // sequences keep their lookahead and the constrained one advances every other step
        std::vector<int> nxt;
        // a slot whose reply limit (set_limit) might be reached by the step in flight stays out
        // too: it ends now or steps again from its exact position
        for (int i = 0; i < n; i++) {
            const Seq& q = seqs_[sl[i]];
            const bool room = q.end == 0 || q.pos + (from[i] >= 0 ? bkvs_[bbuf_][from[i]] + 1 : 0) < q.end;
            if (!full[i] && room && !((q.constrained || q.no_ahead) && from[i] >= 0)) nxt.push_back(sl[i]);
        }
        if (nxt.empty()) {
            ahead = false;
        } else if (nxt != bslots_) {
            bslots_ = nxt;
            bdrop_.clear();
        }
    }
    if (ahead) launch_batch(bbuf_ ^ 1, true);
    CUDA_CHECK(cudaSetDevice(workers_[0]->device));
    CUDA_CHECK(cudaEventSynchronize(bev_[bbuf_]));
    for (int i = 0; i < n; i++) {
        Seq& q = seqs_[sl[i]];
        if (from[i] < 0) {  // joined after the queued step was launched (or ended: no room)
            if (!full[i]) cnt[i] = 0;
            continue;
        }
        const int* em = batch_host_ + (bbuf_ * kMaxBatch + from[i]) * 64;
        const int c = em[0];
        cnt[i] = c;
        for (int j = 0; j < c; j++) out[i * 64 + j] = em[1 + j];
        const bool lkstep = blkr_[bbuf_][from[i]] > 0;
        q.count_step(lkstep, lkstep ? blkr_[bbuf_][from[i]] : bkvs_[bbuf_][from[i]], c);
        if (!q.constrained && !bclamp_[bbuf_][from[i]] && !lkstep) observe(bkvs_[bbuf_][from[i]], c - 1);
        if (lkb_on_) {
            const int s = sl[i];
            if (lkb_len_[s] >= 0) lkb_len_[s] += c;
            static const float ema_min = env_double("QE_LKB_EMA_MIN", 2.f);
            if (lkstep) {
                lkb_ema_[s] = 0.7f * lkb_ema_[s] + 0.3f * (c - 1);
                if (lkb_ema_[s] < ema_min) lkb_mode_[s] = 0, lkb_cool_[s] = 16, lkb_ema_[s] = 3.5f;  // no better than MTP
            }
            if (lkb_cool_[s] > 0) lkb_cool_[s]--;
            else if (em[31] > 0) lkb_mode_[s] = 1, lkb_miss_[s] = 0;
            else if (++lkb_miss_[s] >= lkb_hold_) lkb_mode_[s] = 0;
            lkb_slotsteps_++;
            if (lkstep) lkb_steps_++, lkb_rows_sum_ += blkr_[bbuf_][from[i]], lkb_tok_ += c;
        }
        if (q.want_drafts) q.drafts.assign(em + 32, em + 32 + bkd_[bbuf_]);
        q.rb_pos = q.pos;
        q.rb_rows = c;
        q.rb_batched = true;
        q.rb_epoch = ahead && std::find(bslots_.begin(), bslots_.end(), sl[i]) != bslots_.end() ? -1 : epoch_;
        q.rb_launch = q.rb_epoch < 0 ? -1 : q.launches;
        q.pos += c;
        q.seq.push_back(q.pending);
        for (int j = 0; j + 1 < c; j++) q.seq.push_back(out[i * 64 + j]);
        q.pending = out[i * 64 + c - 1];
        q.used = ++tick_;
    }
    bbuf_ ^= 1;
    binflight_ = ahead;
    static const int lks = env_int("QE_LKB_STATS", 0);
    if (lkb_on_ && lks > 0 && ++lkb_bsteps_ % lks == 0)
        fprintf(stderr, "[lkb] %lld batched steps, %lld slot-steps, lookup %lld (%.1f%%), %.2f rows %.2f tok per lookup step\n",
                lkb_bsteps_, lkb_slotsteps_, lkb_steps_, 100.0 * lkb_steps_ / std::max(1LL, lkb_slotsteps_),
                (double)lkb_rows_sum_ / std::max(1LL, lkb_steps_), (double)lkb_tok_ / std::max(1LL, lkb_steps_));
}

int Engine::spec_step(int* out) {
    Seq& q = seqs_[cur_];
    if (q.paused) {  // stepped alone: resume as soon as its region fits (else no tokens yet)
        if (!unpause_slot(cur_)) return 0;
        resync(q, cur_);
    }
    if (k_ == 0) {  // plain decoding
        q.seq.push_back(q.pending);
        out[0] = q.last_tok = q.pending = decode(q.last_tok);
        if (q.pen) {
            const int t = out[0], at = q.pos;  // decode advanced pos to the new token's
            pool_->run([&](int r) { workers_[r]->pen_host(&t, 1, at); });
        }
        q.count_step(false, 0, 1);
        return 1;
    }
    if (binflight_) {  // switching from batched steps: undo the queued one
        finish_batch();
        resync(q, cur_);
    }
    if (!inflight_ && q.pos + 2 * (k_ + 1) + 1 > q.max_t &&  // room for this step and the next
        !room_or_pause(cur_, std::min(q.pos + 2 * (k_ + 1) + 1, max_ctx_))) {
        if (q.paused) return 0;  // paused for a sequence worth more: no tokens until it resumes
        out[0] = cfg_.eos;  // no room left even alone: the sequence ends here (like reaching max tokens)
        return 1;
    }
    CHECK(q.pos + k_ + 1 <= q.max_t, "context full");
    // lookup drafts (QE_LOOKUP): the device history must be this sequence's; a lookup step is launched once the
    // previous step's result is known (no step queued ahead while the text is being copied)
    const bool lk_ok = lk_on_ && !q.constrained && !q.want_drafts && !q.no_lookup;
    if (lk_ok && !inflight_ && lk_slot_ != cur_) {
        std::vector<int> h(q.seq);
        h.push_back(q.pending);
        pool_->run([&](int r) { workers_[r]->lk_set_history(h.data(), (int)h.size()); });
        CUDA_CHECK(cudaSetDevice(workers_[0]->device));
        CUDA_CHECK(cudaMemcpy(&lk_next_, workers_[0]->lk_dev_ptr(), 4, cudaMemcpyDeviceToHost));
        lk_slot_ = cur_;
    }
    const bool lk_live = lk_ok && lk_slot_ == cur_;
    if (!inflight_) {
        int L = lk_live ? lk_next_ : 0;
        if (L > 0) {  // room for the lookup rows and a following MTP step
            const int need = q.pos + L + 2 * (k_ + 1) + 1;
            if (need > q.max_t) ensure_cap(cur_, std::min(need, max_ctx_));
            L = std::min(L, std::max(0, q.max_t - q.pos - 2 * (k_ + 1) - 1));
        }
        launch_step(buf_, q.pos, L);
        inflight_ = true;
    }
    // queue the next step before waiting for this one (if two steps still fit the slot, and the
    // reply's limit leaves room for a token even if every draft of this one is accepted)
    const int worst = q.pos + buf_kv_[buf_] + 1;  // the slot's position after this step, at most
    ahead_ = lookahead_ && !q.constrained && !q.no_ahead && q.pos + 2 * (k_ + 1) <= q.max_t &&
             (q.end == 0 || worst < q.end) && !(lk_live && (lk_next_ > 0 || buf_lk_[buf_]));
    if (ahead_) launch_step(buf_ ^ 1, worst, 0);
    CUDA_CHECK(cudaSetDevice(workers_[0]->device));
    CUDA_CHECK(cudaEventSynchronize(emit_ev_[buf_]));
    const int* em = emit_host_ + buf_ * 64;
    const int cnt = em[0];
    for (int i = 0; i < cnt; i++) out[i] = em[1 + i];
    q.count_step(buf_lk_[buf_] != 0, buf_kv_[buf_], cnt);
    if (!q.constrained && !buf_lk_[buf_]) {  // kd = 0 / lookup steps would skew the draft statistics / step-time model
        if (!buf_clamped_[buf_]) observe(buf_kv_[buf_], cnt - 1);  // capped drafts: biased
        observe_time(buf_);
    }
    if (lk_live) {
        lk_next_ = em[31];  // the lookup drafts this step found for the next one
        lk_steps_ += buf_lk_[buf_];
        lk_rows_ += buf_lk_[buf_] ? buf_kv_[buf_] : 0;
        lk_tok_ += buf_lk_[buf_] ? cnt : 0;
        steps_all_++;
    }
    if (q.want_drafts) q.drafts.assign(em + 32, em + 32 + buf_kd_[buf_]);
    q.rb_pos = q.pos;
    q.rb_rows = cnt;
    q.rb_epoch = ahead_ ? -1 : epoch_;
    q.rb_batched = false;
    buf_ ^= 1;
    inflight_ = ahead_;
    q.pos += cnt;
    // processed inputs this step: the pending token and the accepted drafts out[0..cnt-1)
    q.seq.push_back(q.pending);
    for (int i = 0; i + 1 < cnt; i++) q.seq.push_back(out[i]);
    q.pending = out[cnt - 1];
    return cnt;
}

void Engine::debug_hidden(const int* ids, int n, float* out) {
    CHECK(n <= kMaxGemvTokens, "debug_hidden n");
    reset();
    std::vector<float*> bufs(workers_.size());
    for (size_t r = 0; r < workers_.size(); r++) {
        CUDA_CHECK(cudaMallocHost(&bufs[r], (size_t)cfg_.layers * n * cfg_.hidden * 4));
        workers_[r]->debug_hidden = bufs[r];
    }
    step(ids, n, true);
    memcpy(out, bufs[0], (size_t)cfg_.layers * n * cfg_.hidden * 4);
    for (size_t r = 0; r < workers_.size(); r++) {
        workers_[r]->debug_hidden = nullptr;
        cudaFreeHost(bufs[r]);
    }
}

void Engine::logits(float* out, int m) {
    // m == 1: the last row (the prompt end / last token), whatever the last forward's row count
    const int first = m == 1 ? last_rows_ - 1 : 0;
    for (size_t r = 0; r < workers_.size(); r++) {
        const CardPlan& p = plan_[r];
        CUDA_CHECK(cudaSetDevice(workers_[r]->device));
        CUDA_CHECK(cudaMemcpy2D(out + p.vocab.b, (size_t)cfg_.vocab * 4,
                                workers_[r]->logits_ + (size_t)first * p.vocab.size(),
                                (size_t)p.vocab.size() * 4, (size_t)p.vocab.size() * 4, m,
                                cudaMemcpyDeviceToHost));
    }
}
