#pragma once
// Release prep [dsv4-rel-gguf]: the whole model from a GGUF split set (Maya-S, --gguf DIR) plus the DSpark drafter
// container next to it (dspark/dspark.bin + dspark.json: a JSON index of name, type, offset, bytes, shape), with no
// checkpoint and no pack. Included INSIDE namespace dsv4 of dsv4.cu, after gguf_hooks.h.
//
// What it does at start (gm_prepare, before create() reads anything):
//  - opens the split set, checks the model's metadata against the engine's constants;
//  - registers every non-expert tensor in E->stf under the checkpoint's names (the loader is unchanged): F32 / BF16 as
//    they are, Q8_0 as dtype "Q8_0" (the engine splits it into int8 + fp16 scales), the router weight as BF16 (the GGUF
//    stores the checkpoint's BF16 values as F32: checked to round-trip exactly);
//  - the derived data the pack used to carry goes into a per-model cache dir (user cache, never a download):
//    pack.json (expert usage ranks, formats, Engram rows), codebooks.json (constants), engram_hash.bin (from the GGUF
//    metadata), draft_cb3.bin (the drafter's MXFP4 experts in cb3, converted once); E->o.packed points there.
// Engram rows stay in the GGUF shards (Q6_K, read with O_DIRECT per token, gm_engram()).

#include <sys/stat.h>
#include <unistd.h>

struct GmState {
    GgufModel m;
    std::string cache;                          // per-model cache dir
    std::vector<std::vector<uint8_t>> owned;    // host buffers behind converted tensors (BF16 router weights)
    const uint8_t* dsp = nullptr;               // dspark.bin mapped (read-only)
    size_t dsp_size = 0;
    int eg_fd[2] = {-1, -1};                    // Engram tables: file and data offset per engram layer (Q6_K rows)
    size_t eg_off[2] = {};
    size_t eg_rows[2] = {};
};

static bool gm_mkdirs(const std::string& d) {
    for (size_t i = 1; i <= d.size(); i++)
        if (i == d.size() || d[i] == '/') {
            const std::string p = d.substr(0, i);
            if (mkdir(p.c_str(), 0755) != 0 && errno != EEXIST) return false;
        }
    return true;
}
static bool gm_exists(const std::string& p, size_t size = (size_t)-1) {
    struct stat sb;
    return stat(p.c_str(), &sb) == 0 && (size == (size_t)-1 || (size_t)sb.st_size == size);
}
// atomic: tmp + rename; the tmp name carries the pid (review rel M3: two processes on one cache dir). must = false:
// best effort (a failed write is logged, the file keeps its old content), for writes in the middle of decode
static bool gm_write(const std::string& path, const void* p, size_t n, bool must = true) {
    const std::string t = path + ".tmp." + std::to_string((long)getpid());
    FILE* f = fopen(t.c_str(), "wb");
    bool ok = f != nullptr;
    if (f) ok = fwrite(p, 1, n, f) == n;
    if (f) ok = fclose(f) == 0 && ok;
    ok = ok && rename(t.c_str(), path.c_str()) == 0;
    if (!ok) {
        const int er = errno;
        unlink(t.c_str());
        DCHECK(!must, "cache: cannot write %s: %s", path.c_str(), strerror(er));
        static int nw = 0;
        if (nw++ < 3) fprintf(stderr, "warning: cache: cannot write %s: %s (kept the old file)\n", path.c_str(), strerror(er));
    }
    return ok;
}
static uint64_t gm_fnv(uint64_t h, const void* p, size_t n) {
    for (size_t i = 0; i < n; i++) h = (h ^ ((const uint8_t*)p)[i]) * 1099511628211ull;
    return h;
}
static std::string gm_kv(const GgufModel& m, const std::string& k) {
    std::string v;
    DCHECK(m.kv(k, &v), "--gguf: metadata key %s missing", k.c_str());
    return v;
}
static TensorInfo gm_ti(const GgTensor& t) {   // a GGUF tensor as a checkpoint-style TensorInfo (shape outermost first)
    TensorInfo ti;
    ti.dtype = t.type == GG_F32 ? "F32" : t.type == GG_BF16 ? "BF16" : t.type == GG_Q8_0 ? "Q8_0" : gg_type_name(t.type);
    for (size_t i = t.ne.size(); i-- > 0;) ti.shape.push_back((long long)t.ne[i]);
    ti.data = (const char*)t.data;
    ti.nbytes = t.nbytes;
    return ti;
}

// the drafter container (dspark.json + dspark.bin): names as the checkpoint's; returns false when absent
static bool gm_dspark(Engine* E, GmState& G, std::vector<std::pair<std::string, TensorInfo>>* mx) {
    const std::string dir = E->o.dspark_dir;
    const std::string jp = dir + "/dspark.json", bp = dir + "/dspark.bin";
    if (!gm_exists(jp) || !gm_exists(bp)) return false;
    Json j = Json::parse(rd_file(jp));
    const int fd = open(bp.c_str(), O_RDONLY);
    DCHECK(fd >= 0, "open %s: %s", bp.c_str(), strerror(errno));
    struct stat sb;
    DCHECK(fstat(fd, &sb) == 0, "stat %s", bp.c_str());
    G.dsp_size = (size_t)sb.st_size;
    void* m = mmap(nullptr, G.dsp_size, PROT_READ, MAP_SHARED, fd, 0);
    DCHECK(m != MAP_FAILED, "mmap %s", bp.c_str());
    close(fd);
    G.dsp = (const uint8_t*)m;
    const Json& ts = j["tensors"];
    for (size_t i = 0; i < ts.arr.size(); i++) {
        const Json& t = ts[i];
        TensorInfo ti;
        ti.dtype = t[1].str;
        const size_t off = (size_t)t[2].i(), nb = (size_t)t[3].i();
        DCHECK(off + nb <= G.dsp_size && off + nb >= off, "dspark.bin: %s past the file (partial download?)", t[0].str.c_str());
        size_t ne = 1;
        for (size_t k = 0; k < t[4].arr.size(); k++) { ti.shape.push_back(t[4][k].i()); ne *= (size_t)t[4][k].i(); }
        const size_t want = ti.dtype == "F32" ? ne * 4 : ti.dtype == "BF16" ? ne * 2 : ti.dtype == "Q8_0" ? ne / 32 * 34
                            : ti.dtype == "MXFP4" ? ne / 32 * 17 : 0;
        DCHECK(want == nb, "dspark.json: %s %s: %zu bytes for %zu values", t[0].str.c_str(), ti.dtype.c_str(), nb, ne);
        ti.data = (const char*)G.dsp + off;
        ti.nbytes = nb;
        if (ti.dtype == "MXFP4") mx->push_back({t[0].str, ti});
        else E->stf.put(t[0].str, ti);
    }
    return true;
}

// ggml MXFP4 rows [N][K] (17 B per 32: e8m0, then element j in the low nibble of byte j, j + 16 in its high nibble) ->
// the checkpoint's FP4 layout (byte j = elements 2j | 2j+1 << 4, e8m0 per 32 separately): same codes, same scales
static void gm_mxfp4_to_fp4(const uint8_t* src, int N, int K, uint8_t* q4, uint8_t* s8) {
    for (int r = 0; r < N; r++)
        for (int b = 0; b < K / 32; b++) {
            const uint8_t* p = src + ((size_t)r * (K / 32) + b) * 17;
            s8[(size_t)r * (K / 32) + b] = p[0];
            uint8_t v[32];
            for (int j = 0; j < 16; j++) { v[j] = p[1 + j] & 15; v[j + 16] = p[1 + j] >> 4; }
            uint8_t* o = q4 + (size_t)r * (K / 2) + b * 16;
            for (int j = 0; j < 16; j++) o[j] = (uint8_t)(v[2 * j] | (v[2 * j + 1] << 4));
        }
}

// the drafter's 3 x 128 experts in cb3 (the pack tool's RTN, dsv4_m23.h m23_quant_mat with 3 bits), once per model
static void gm_draft_cb3(Engine* E, GmState& G, const std::vector<std::pair<std::string, TensorInfo>>& mx,
                         const float* cb3) {
    const std::string out = G.cache + "/draft_cb3.bin";
    const size_t XB3 = expert_bytes(XF_CB3), tot = (size_t)3 * DS_NE * XB3;
    if (gm_exists(out, tot)) return;
    std::map<std::string, const TensorInfo*> by;
    for (auto& p : mx) by[p.first] = &p.second;
    auto t0 = clk::now();
    logf_("first start: converting the drafter's experts to cb3 (cached in %s) ...\n", G.cache.c_str());
    std::vector<uint8_t> buf(tot);
    std::atomic<int> next{0};
    auto wk = [&] {
        std::vector<uint8_t> q4((size_t)INTER * D / 2), s8((size_t)INTER * D / 32);
        for (int t; (t = next.fetch_add(1)) < 3 * DS_NE;) {
            const int i = t / DS_NE, e = t % DS_NE;
            MatView mv[3];
            expert_views(XF_CB3, buf.data() + (size_t)t * XB3, mv);
            const char* wn[3] = {"w1", "w3", "w2"};
            const int Nm[3] = {INTER, INTER, D}, Km[3] = {D, D, INTER};
            for (int k = 0; k < 3; k++) {
                const std::string nm = "mtp." + std::to_string(i) + ".ffn.experts." + std::to_string(e) + "." + wn[k] + ".weight";
                auto it = by.find(nm);
                DCHECK(it != by.end(), "dspark.json: %s missing", nm.c_str());
                DCHECK(it->second->shape.size() == 2 && it->second->shape[0] == Nm[k] && it->second->shape[1] == Km[k],
                       "dspark: %s shape", nm.c_str());
                gm_mxfp4_to_fp4((const uint8_t*)it->second->data, Nm[k], Km[k], q4.data(), s8.data());
                m23_quant_mat(q4.data(), s8.data(), Nm[k], Km[k], 3, cb3, (uint8_t*)mv[k].q);
            }
        }
    };
    std::vector<std::thread> th;
    const int nt = std::max(4, std::min(32, (int)std::thread::hardware_concurrency() - 2));
    for (int k = 0; k < nt; k++) th.emplace_back(wk);
    for (auto& t : th) t.join();
    gm_write(out, buf.data(), buf.size());
    logf_("drafter experts converted in %.1f s\n", ms_since(t0) / 1e3);
}

// usage ranks [NL][NE] (0 = most routed): the cache's learned profile, else --profile / the shipped default, else id
// order. A profile file: "dsv4-profile 1" then NL lines of NE counts (the engine adds what it routes, gm_profile_save)
static bool gm_read_profile(const std::string& path, std::vector<std::vector<double>>& cnt) {
    FILE* f = fopen(path.c_str(), "r");
    if (!f) return false;
    char hdr[32] = {};
    int ver = 0;
    bool ok = fscanf(f, "%31s %d", hdr, &ver) == 2 && !strcmp(hdr, "dsv4-profile") && ver == 1;
    cnt.assign(NL, std::vector<double>(NE, 0.0));
    for (int l = 0; l < NL && ok; l++)
        for (int e = 0; e < NE && ok; e++) ok = fscanf(f, "%lf", &cnt[l][e]) == 1;
    fclose(f);
    return ok;
}
static void gm_ranks(Engine* E, GmState& G, std::vector<std::vector<int>>& rank) {
    std::vector<std::vector<double>> cnt;
    std::string src;
    const std::string mine = G.cache + "/expert_profile.txt";
    // coordinator 11-14 (greedy = f(prompt, settings, files)): the ranks come from a fixed source at every start:
    // --profile FILE (a calibration, e.g. data/dsv4/expert_profile_maya_s.txt, or a pack.json), or none (expert id
    // order); the routing counts are learned in-process only. --profile learned = opt-in persistence: the cache's
    // learned profile is read at start and written back (then greedy text depends on earlier runs)
    const bool learned = E->o.profile == "learned";
    if (!learned && !E->o.profile.empty() && gm_read_profile(E->o.profile, cnt)) src = E->o.profile;
    else if (learned && gm_read_profile(mine, cnt)) src = mine + " (learned, persisted)";
    else if (!learned && !E->o.profile.empty() && gm_exists(E->o.profile)) {   // a pack.json (rank lists): the old packs' ranking
        Json pk = Json::parse(rd_file(E->o.profile));
        cnt.assign(NL, std::vector<double>(NE, 0.0));
        for (int l = 0; l < NL; l++)   // (the ranks are global, 0 .. NL * NE - 1: counts stay positive)
            for (int e = 0; e < NE; e++) cnt[l][e] = (double)(NL * NE - pk["rank"][l][e].i());
        src = E->o.profile + " (pack ranks)";
    }
    if (src.empty() && !E->o.profile.empty())   // review rel2 L4: a profile was asked for and not found
        fprintf(stderr, "warning: --profile %s: not readable (or no learned profile yet): expert id order\n", E->o.profile.c_str());
    rank.assign(NL, std::vector<int>(NE));
    for (int l = 0; l < NL; l++) {
        std::vector<int> ord(NE);
        std::iota(ord.begin(), ord.end(), 0);
        if (!cnt.empty()) std::stable_sort(ord.begin(), ord.end(), [&](int a, int b) { return cnt[l][a] > cnt[l][b]; });
        for (int r = 0; r < NE; r++) rank[l][ord[r]] = r;
    }
    if (cnt.empty()) cnt.assign(NL, std::vector<double>(NE, 0.0));
    for (auto& row : cnt) {   // decay: a layer's counts are kept below 1e6 uses, so new routing moves the profile
        double t = 0;
        for (double v : row) t += v;
        if (t > 1e6)
            for (double& v : row) v *= 1e6 / t;
    }
    E->use_cnt = cnt;
    // review rel M3: the ranks pick the seeds and the experts without a host copy, so the residency and with it the
    // exact greedy text; their hash makes a changed (learned) profile visible between runs (--profile: fixed)
    uint64_t h = 1469598103934665603ull;
    for (auto& r : rank) h = gm_fnv(h, r.data(), r.size() * sizeof(int));
    logf_("expert profile: %s (ranks %016llx)\n", src.empty() ? "none (expert id order; routing counts learned in-process)" : src.c_str(),
          (unsigned long long)h);
}
// the learned profile: decayed counts of what this machine routes (written at exit and every few thousand passes)
static void gm_profile_save(Engine* E) {
    if (!E->gm || E->use_cnt.empty() || E->o.profile != "learned") return;   // persistence is opt-in (--profile learned)
    std::string s = "dsv4-profile 1\n";
    char b[32];
    for (int l = 0; l < NL; l++) {
        for (int e = 0; e < NE; e++) {
            snprintf(b, sizeof b, e ? " %.6g" : "%.6g", E->use_cnt[l][e]);
            s += b;
        }
        s += "\n";
    }
    gm_write(E->gm->cache + "/expert_profile.txt", s.data(), s.size(), false);   // review rel M3: never aborts a reply
}

static const char* GM_CODEBOOKS =
    "{\"cb3g128\":[-0.7926860076574863,-0.427133593481455,-0.18559296319269603,-0.014089978040486726,0.12658389307139,"
    "0.2650761187326036,0.4410803952004012,0.7927438091524425],\"cb2g128\":[-0.6217325137677024,-0.16203358956381922,"
    "0.22063079026371848,0.6215696373642079],\"eg2\":[-0.6060453717414115,-0.17991394995196633,0.1793634578673588,"
    "0.6063921946990395],\"eg3\":[-0.8160799801066795,-0.4987095414813292,-0.2780186143062115,-0.09240842541483893,"
    "0.08549995289062157,0.2743659805455622,0.4986994779782875,0.8156691842397921],\"eg4\":[-0.862003542566486,"
    "-0.5788327285789711,-0.42039000587767583,-0.32044864115594246,-0.23503208105771176,-0.16367154235190506,"
    "-0.0949480984508481,-0.028400094050682348,0.033973406200194865,0.0978761865853695,0.16363930316181297,"
    "0.23517595154477525,0.32037343501464954,0.42040434591767833,0.579187631568072,0.8619995051176506]}";

// R24 [dsv4-r24-load]: the non-expert tensors (dense Q8_0 / F32 / BF16, head, embedding) are read later through the
// MADV_RANDOM mapping by one thread (4 KB faults: ~40 s for ~9 GB): read them into the page cache first with large
// parallel preads (the mapped bytes are the same file pages: no change in any value). Routed experts (read
// explicitly later) and the Engram tables (O_DIRECT rows) are skipped. DSV4_NOWARM=1: off
static void gm_warm(const GgufModel& M) {
    if (getenv("DSV4_NOWARM") && atoi(getenv("DSV4_NOWARM")) != 0) return;
    const auto t0 = std::chrono::steady_clock::now();
    struct Pc { int fd; off_t o; size_t n; };
    std::vector<Pc> pcs;
    const size_t P = 32u << 20;
    size_t tot = 0;
    for (const GgTensor& t : M.tensors()) {
        if (!t.sized || t.file < 0 || t.name.find("_exps.") != std::string::npos ||
            t.name.find("engram_embd") != std::string::npos)
            continue;
        const GgFile& F = M.files()[t.file];
        const off_t a = (off_t)((F.data_off + t.offset) & ~(uint64_t)4095);
        const off_t b = (off_t)(F.data_off + t.offset + t.nbytes);
        for (off_t o = a; o < b; o += (off_t)P) pcs.push_back({F.fd, o, (size_t)std::min<off_t>((off_t)P, b - o)});
        tot += t.nbytes;
    }
    std::atomic<size_t> next{0};
    std::vector<std::thread> th;
    for (int i = 0; i < 16; i++)
        th.emplace_back([&] {
            std::vector<uint8_t> buf(P);
            for (size_t i2; (i2 = next.fetch_add(1)) < pcs.size();) {
                size_t got = 0;
                while (got < pcs[i2].n) {
                    const ssize_t r = pread(pcs[i2].fd, buf.data(), pcs[i2].n - got, pcs[i2].o + (off_t)got);
                    if (r <= 0) break;   // best effort: the mapping reads it later anyway
                    got += (size_t)r;
                }
            }
        });
    for (auto& x : th) x.join();
    logf_("non-expert tensors read ahead: %.2f GB in %.1f s (16 threads)\n", tot / 1e9,
          std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
}

static void gm_prepare(Engine* E) {
    Options& o = E->o;
    E->gm = new GmState;
    GmState& G = *E->gm;
    std::string err;
    DCHECK(G.m.open_dir(o.gguf_model, &err), "--gguf: %s", err.c_str());
    const GgufModel& M = G.m;
    // ---- the model must be the one the engine is built for
    DCHECK(gm_kv(M, "general.architecture") == "deepseek41", "--gguf: architecture %s (deepseek41 expected)",
           gm_kv(M, "general.architecture").c_str());
    auto num = [&](const char* k) { return atoll(gm_kv(M, std::string("deepseek41.") + k).c_str()); };
    const long long want[][2] = {{num("block_count"), NL}, {num("embedding_length"), D}, {num("expert_count"), NE},
                                 {num("expert_used_count"), TOPK}, {num("expert_feed_forward_length"), INTER},
                                 {num("attention.q_lora_rank"), QLR}, {num("attention.key_length"), HD},
                                 {num("attention.head_count"), NH}, {num("attention.output_lora_rank"), OLR},
                                 {num("attention.output_group_count"), OG}};
    for (auto& w : want) DCHECK(w[0] == w[1], "--gguf: model dimension %lld, the engine is built for %lld", w[0], w[1]);
    {
        std::vector<int64_t> cr;
        DCHECK(M.arr_i64("deepseek41.attention.compress_ratios", &cr) && cr.size() >= (size_t)NL, "--gguf: compress_ratios");
        for (int l = 0; l < NL; l++) DCHECK(cr[l] == ratio_of(l), "--gguf: layer %d compress ratio %lld vs %d", l, (long long)cr[l], ratio_of(l));
    }
    gm_warm(M);   // R24 [dsv4-r24-load] (review r24 L8: after the model checks)
    // ---- cache dir: per model (file names and sizes of the split set + the drafter's size)
    uint64_t h = 1469598103934665603ull;
    for (const GgFile& f : M.files()) {
        const std::string n = f.path.substr(f.path.rfind('/') + 1);
        h = gm_fnv(h, n.data(), n.size());
        h = gm_fnv(h, &f.size, sizeof f.size);
    }
    if (o.dspark_dir.empty()) {
        std::string d = o.gguf_model;
        while (d.size() > 1 && d.back() == '/') d.pop_back();
        o.dspark_dir = d.substr(0, d.rfind('/')) + "/dspark";
    }
    struct stat dsb;
    if (stat((o.dspark_dir + "/dspark.bin").c_str(), &dsb) == 0) h = gm_fnv(h, &dsb.st_size, sizeof dsb.st_size);
    if (o.cache_dir.empty()) {
        const char* x = getenv("XDG_CACHE_HOME");
        const char* hm = getenv("HOME");
        o.cache_dir = std::string(x && *x ? x : (std::string(hm ? hm : "/tmp") + "/.cache")) + "/qengine/dsv4";
    }
    char hx[24];
    snprintf(hx, sizeof hx, "%016llx", (unsigned long long)h);
    G.cache = o.cache_dir + "/" + hx;
    DCHECK(gm_mkdirs(G.cache), "cache dir %s: %s", G.cache.c_str(), strerror(errno));
    // ---- tensors under the checkpoint's names
    auto T = [&](const std::string& n) -> const GgTensor& {
        const GgTensor* t = M.find(n);
        DCHECK(t && t->sized && t->data, "--gguf: tensor %s missing (incomplete download?)", n.c_str());
        return *t;
    };
    auto put = [&](const std::string& ck, const std::string& gn) { E->stf.put(ck, gm_ti(T(gn))); };
    auto put_bf16 = [&](const std::string& ck, const std::string& gn) {   // F32 holding BF16 values -> BF16 (exact)
        const GgTensor& t = T(gn);
        TensorInfo ti = gm_ti(t);
        if (t.type == GG_F32) {
            const size_t n = t.nbytes / 4;
            G.owned.emplace_back(n * 2);
            uint16_t* d = (uint16_t*)G.owned.back().data();
            for (size_t i = 0; i < n; i++) {
                uint32_t u;
                memcpy(&u, t.data + i * 4, 4);
                DCHECK((u & 0xffff) == 0, "--gguf: %s is not BF16-exact (value %zu)", gn.c_str(), i);
                d[i] = (uint16_t)(u >> 16);
            }
            ti.dtype = "BF16";
            ti.data = (const char*)d;
            ti.nbytes = n * 2;
        }
        E->stf.put(ck, ti);
    };
    for (int l = 0; l < NL; l++) {
        const std::string p = "layers." + std::to_string(l) + ".", a = p + "attn.", f = p + "ffn.", b = "blk." + std::to_string(l) + ".";
        put(a + "wq_a.weight", b + "attn_q_a.weight");
        put(a + "wkv.weight", b + "attn_kv.weight");
        put(a + "wq_b.weight", b + "attn_q_b.weight");
        put(a + "wo_a.weight", b + "attn_output_a.weight");
        put(a + "wo_b.weight", b + "attn_output_b.weight");
        put(a + "q_norm.weight", b + "attn_q_a_norm.weight");
        put(a + "kv_norm.weight", b + "attn_kv_a_norm.weight");
        put(a + "attn_sink", b + "attn_sinks.weight");
        put(p + "attn_norm.weight", b + "attn_norm.weight");
        put(p + "ffn_norm.weight", b + "ffn_norm.weight");
        for (const char* hk : {"attn", "ffn"})
            for (const char* q : {"fn", "scale", "base"})
                put(p + "hc_" + hk + "_" + q, b + "hc_" + hk + "_" + q + ".weight");
        if (is_kv_src(l)) {
            put(a + "compressor.wkv.weight", b + "attn_compressor_kv.weight");
            put(a + "compressor.norm.weight", b + "attn_compressor_norm.weight");
            if (ratio_of(l) > 1) put(a + "compressor.wgate.weight", b + "attn_compressor_gate.weight");
            put(a + "indexer.wk.weight", b + "indexer.attn_k.weight");
            put(a + "indexer.k_norm.weight", b + "indexer.k_norm.weight");
        }
        if (is_idx_src(l)) {
            put(a + "indexer.wq_b.weight", b + "indexer.attn_q_b.weight");
            put(a + "indexer.weights_proj.weight", b + "indexer.proj.weight");
        }
        put_bf16(f + "gate.weight", b + "ffn_gate_inp.weight");
        put(f + "gate.bias", b + "exp_probs_b.bias");
        put(f + "shared_experts.w1.weight", b + "ffn_gate_shexp.weight");
        put(f + "shared_experts.w3.weight", b + "ffn_up_shexp.weight");
        put(f + "shared_experts.w2.weight", b + "ffn_down_shexp.weight");
        if (is_engram(l)) {
            put(p + "engram.wkv.weight", b + "engram_kv.weight");
            put(p + "engram.qk", b + "engram_gate.weight");   // = q_weight x k_weight (the engine's product)
        }
    }
    put("head.weight", "output.weight");
    put("norm.weight", "output_norm.weight");
    put("embed.weight", "token_embd.weight");
    // ---- Engram tables: Q6_K rows in the Engram shards (read per token with O_DIRECT, never loaded)
    for (int li = 0; li < 2; li++) {
        const GgTensor& t = T("blk." + std::to_string(li ? 14 : 1) + ".engram_embd.weight");
        DCHECK(t.type == GG_Q6_K && t.ne.size() == 2 && t.ne[0] == EG_HD, "--gguf: Engram table %d: %s, %zu dims (Q6_K x 256 expected)",
               li, gg_type_name(t.type), t.ne.size());
        const GgFile& F = M.files()[t.file];
        G.eg_off[li] = F.data_off + t.offset;
        G.eg_rows[li] = t.ne[1];
        G.eg_fd[li] = open(F.path.c_str(), O_RDONLY | O_DIRECT);
        if (G.eg_fd[li] < 0) {
            logf_("warning: O_DIRECT open %s: %s; buffered Engram reads\n", F.path.c_str(), strerror(errno));
            G.eg_fd[li] = open(F.path.c_str(), O_RDONLY);
        }
        DCHECK(G.eg_fd[li] >= 0, "open %s: %s", F.path.c_str(), strerror(errno));
    }
    // ---- derived data in the cache
    gm_write(G.cache + "/codebooks.json", GM_CODEBOOKS, strlen(GM_CODEBOOKS));
    {   // engram_hash.bin: the dsv4_pack.py layout (int32 V tmap | int64 pad | primes [2][3][8] | mult [2][4] | offs [2][24])
        std::vector<int64_t> tm, pr, mu, of;
        DCHECK(M.arr_i64("deepseek41.engram.token_map", &tm) && tm.size() == (size_t)V, "--gguf: engram.token_map");
        DCHECK(M.arr_i64("deepseek41.engram.primes", &pr) && pr.size() == 48, "--gguf: engram.primes");
        DCHECK(M.arr_i64("deepseek41.engram.multipliers", &mu) && mu.size() == 8, "--gguf: engram.multipliers");
        DCHECK(M.arr_i64("deepseek41.engram.offsets", &of) && of.size() == 48, "--gguf: engram.offsets");
        const int64_t pad = atoll(gm_kv(M, "deepseek41.engram.pad_token_id").c_str());
        std::string b((size_t)V * 4 + 8 + (48 + 8 + 48) * 8, '\0');
        char* q = &b[0];
        for (int i = 0; i < V; i++) { const int32_t v = (int32_t)tm[i]; memcpy(q, &v, 4); q += 4; }
        memcpy(q, &pad, 8); q += 8;
        memcpy(q, pr.data(), 48 * 8); q += 48 * 8;
        memcpy(q, mu.data(), 8 * 8); q += 8 * 8;
        memcpy(q, of.data(), 48 * 8);
        gm_write(G.cache + "/engram_hash.bin", b.data(), b.size());
    }
    std::vector<std::vector<int>> rank;
    gm_ranks(E, G, rank);
    {   // pack.json: an M23-style pack (every expert a host format; the GGUF mode re-sets the formats), Engram on NVMe
        std::string s = "{\"hot\":0,\"cold\":\"gguf\",\"engram\":{\"bits\":8,\"hot_bits\":0,\"rows\":[" +
                        std::to_string(G.eg_rows[0]) + "," + std::to_string(G.eg_rows[1]) + "],\"source\":\"checkpoint\"},\"rank\":[";
        for (int l = 0; l < NL; l++) {
            s += l ? ",[" : "[";
            for (int e = 0; e < NE; e++) s += (e ? "," : "") + std::to_string(rank[l][e]);
            s += "]";
        }
        s += "],\"fmt\":[";
        for (int l = 0; l < NL; l++) {
            s += l ? ",[" : "[";
            for (int e = 0; e < NE; e++) s += e ? ",3" : "3";
            s += "]";
        }
        s += "]}";
        gm_write(G.cache + "/pack.json", s.data(), s.size());
    }
    std::vector<std::pair<std::string, TensorInfo>> mx;
    const bool have_ds = gm_dspark(E, G, &mx);
    if (o.dspark && !have_ds) {
        logf_("warning: no DSpark drafter in %s (dspark.bin + dspark.json): DSpark off\n", o.dspark_dir.c_str());
        o.dspark = 0;
    }
    if (have_ds && o.dspark) {
        o.dspark_cb3 = 1;   // the drafter's experts always run as cb3 here (converted once from its MXFP4)
        float cb3[8];
        Json cbj = Json::parse(GM_CODEBOOKS);
        for (int i = 0; i < 8; i++) cb3[i] = (float)cbj["cb3g128"][i].f();
        gm_draft_cb3(E, G, mx, cb3);
    }
    o.packed = G.cache;
    o.gguf_dir = o.gguf_model;
    logf_("GGUF model: %s (%zu files), drafter %s, cache %s\n", o.gguf_model.c_str(), M.files().size(),
          have_ds ? o.dspark_dir.c_str() : "none", G.cache.c_str());
}
