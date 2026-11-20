// GLM-5.3-Flash engine: loader, per-card workers, forward (see glm.h)
#include "glm.h"
#include "glm_cpu.h"

#include <immintrin.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <dlfcn.h>
#include <algorithm>

#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <map>
#include <set>
#include <mutex>

#include "../env.h"
#include "../json.h"
#include "../safetensors.h"
#include "glm_kernels.cuh"
#include "../kernels/watchdog.cuh"

namespace glm {

static inline double glm_us_since(std::chrono::steady_clock::time_point t) {
    return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t).count();
}
static double now_s() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
#define CUBLAS_CHECK(x)                                                              \
    do {                                                                             \
        cublasStatus_t s__ = (x);                                                    \
        CHECK(s__ == CUBLAS_STATUS_SUCCESS, "cublas %d at %s:%d", (int)s__, __FILE__, __LINE__); \
    } while (0)

// ---------------------------------------------------------------- checkpoint access

static SafeTensors* g_st = nullptr;
static const std::string P = "model.language_model.";

static const TensorInfo& T_(const std::string& n) { return g_st->get(n); }
static std::vector<float> host_f32(const std::string& n) {
    const TensorInfo& t = T_(n);
    size_t cnt = 1;
    for (auto d : t.shape) cnt *= d;
    std::vector<float> v(cnt);
    if (t.dtype == "F32") memcpy(v.data(), t.data, cnt * 4);
    else {
        CHECK(t.dtype == "BF16", "%s dtype %s", n.c_str(), t.dtype.c_str());
        const uint16_t* s = (const uint16_t*)t.data;
        for (size_t i = 0; i < cnt; i++) {
            uint32_t u = (uint32_t)s[i] << 16;
            memcpy(&v[i], &u, 4);
        }
    }
    return v;
}
// VRAM accounting by kind (per worker thread)
static thread_local std::map<std::string, double>* g_mem = nullptr;
static thread_local const char* g_tag = "other";
// round 21 (review r21b M1): data files next to the library at run time (<root>/build/libqglm.so -> <root>/tools/glm), so a
// build dir that was moved after configuring (dev deploy: dev.new -> dev) still finds them; GLM_SRC_DIR as fallback
std::string glm_data_file(const char* name) {
    Dl_info di;
    if (dladdr((void*)&glm_data_file, &di) && di.dli_fname) {
        std::string p = di.dli_fname;
        for (int up = 0; up < 2 && p.rfind('/') != std::string::npos; up++) p = p.substr(0, p.rfind('/'));
        const std::string f = p + "/tools/glm/" + name;
        if (access(f.c_str(), R_OK) == 0) return f;
    }
    return std::string(GLM_SRC_DIR) + "/tools/glm/" + name;
}
// round 21 [glm-r21-dvbal]: the MTP draft vocabulary (GLM_DRAFT_VOCAB = list of int32 token ids, "0" = full head); default
// the 49K list next to the library when it exists (a missing list is logged once at load)
static std::string draft_vocab_path() {
    if (const char* dv = env_str("GLM_DRAFT_VOCAB")) return dv;
    const std::string d = glm_data_file("draft_vocab_49k.i32");
    return access(d.c_str(), R_OK) == 0 ? d : std::string("0");
}
static void mem_add(double b) { if (g_mem) (*g_mem)[g_tag] += b; }
// bump allocator (prefill sets inside the lent slab tail): base == nullptr -> dry run (size only)
struct Bump {
    bool on = false;
    char* base = nullptr;
    size_t off = 0;
};
static thread_local Bump g_bump;
template <class T>
static T* dalloc(size_t n) {
    void* p = nullptr;
    if (g_bump.on) {
        const size_t o = (g_bump.off + 255) & ~(size_t)255;
        g_bump.off = o + n * sizeof(T) + 16;
        if (!g_bump.base) return (T*)(uintptr_t)(o + 256);  // dry run: never dereferenced
        CUDA_CHECK(cudaMemset(g_bump.base + o, 0, n * sizeof(T)));
        return (T*)(g_bump.base + o);
    }
    mem_add((double)(n * sizeof(T) + 16));
    CUDA_CHECK(cudaMalloc(&p, n * sizeof(T) + 16));
    CUDA_CHECK(cudaMemset(p, 0, n * sizeof(T)));
    return (T*)p;
}
static float* up_f32(const std::vector<float>& v) {
    float* d = dalloc<float>(v.size());
    CUDA_CHECK(cudaMemcpy(d, v.data(), v.size() * 4, cudaMemcpyHostToDevice));
    return d;
}
static float* up_f32(const std::string& n) { return up_f32(host_f32(n)); }
// rows [r0, r1) of a BF16 [rows][cols] tensor into dst (row stride cols)
static void cp_rows(bf16* dst, const std::string& n, int r0, int r1) {
    const TensorInfo& t = T_(n);
    CHECK(t.dtype == "BF16" && t.shape.size() >= 2, "%s", n.c_str());
    const size_t cols = t.nbytes / 2 / t.shape[0];
    CUDA_CHECK(cudaMemcpy(dst, t.data + (size_t)r0 * cols * 2, (size_t)(r1 - r0) * cols * 2, cudaMemcpyHostToDevice));
}
// cols [c0, c1) of a BF16 [rows][cols] tensor -> dst [rows][c1 - c0]
static bf16* up_cols(const std::string& n, int c0, int c1) {
    const TensorInfo& t = T_(n);
    CHECK(t.dtype == "BF16" && t.shape.size() == 2, "%s", n.c_str());
    const size_t rows = t.shape[0], cols = t.shape[1];
    bf16* d = dalloc<bf16>(rows * (c1 - c0));
    CUDA_CHECK(cudaMemcpy2D(d, (size_t)(c1 - c0) * 2, t.data + (size_t)c0 * 2, cols * 2, (size_t)(c1 - c0) * 2, rows,
                            cudaMemcpyHostToDevice));
    return d;
}
static W8 to8(bf16* t, int N, int K, bool q4 = false) {
    W8 w;
    w.N = N; w.K = K;
    w.q = dalloc<int8_t>((size_t)N * K);
    w.s = dalloc<float>((size_t)N * (K / 128));
    if (q4) quant4f(t, N, K, w.q, w.s, 0);
    else quant8(t, N, K, w.q, w.s, 0);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaFree(t));
    mem_add(-(double)((size_t)N * K * 2 + 16));
    return w;
}
static float h2f_(uint16_t h) {  // IEEE fp16 -> fp32 (no F16C in this translation unit)
    const uint32_t sg = (uint32_t)(h & 0x8000) << 16, e = (h >> 10) & 31, m = h & 1023;
    uint32_t u;
    if (e == 0) {
        if (!m) u = sg;
        else { float f = std::ldexp((float)m, -24); memcpy(&u, &f, 4); u |= sg; }
    } else if (e == 31) u = sg | 0x7F800000u | (m << 13);
    else u = sg | ((e + 112) << 23) | (m << 13);
    float f;
    memcpy(&f, &u, 4);
    return f;
}
// round 9: INT4 weight from GPTQ codes (<name>.qcodes int8 [N][K], <name>.qscales fp16 [N][K/128]): parts = (tensor
// base name, row range); columns [c0, c1) of every part (128-aligned); rows padded with zeros to N
static W8 load4(const std::vector<std::tuple<std::string, int, int>>& parts, int N, int c0, int c1) {
    const int K = c1 - c0;
    CHECK(c0 % 128 == 0 && K % 128 == 0, "load4: columns %d..%d", c0, c1);
    std::vector<uint8_t> q((size_t)N * K / 2, 0x00);
    std::vector<float> sc((size_t)N * (K / 128), 0.f);
    int row = 0;
    for (auto& [nm, r0, r1] : parts) {
        const TensorInfo& tq = T_(nm + "qcodes");
        const TensorInfo& ts = T_(nm + "qscales");
        CHECK(tq.dtype == "I8" && ts.dtype == "F16", "load4 %s: %s / %s", nm.c_str(), tq.dtype.c_str(), ts.dtype.c_str());
        const size_t KF = tq.shape[1];
        const int8_t* qs = (const int8_t*)tq.data;
        const uint16_t* ss = (const uint16_t*)ts.data;
        for (int r = r0; r < r1; r++, row++) {
            for (int k = 0; k < K; k += 2) {
                const int a = qs[(size_t)r * KF + c0 + k], b = qs[(size_t)r * KF + c0 + k + 1];
                q[(size_t)row * (K / 2) + k / 2] = (uint8_t)((a & 15) | ((b & 15) << 4));
            }
            for (int g = 0; g < K / 128; g++) sc[(size_t)row * (K / 128) + g] = h2f_(ss[(size_t)r * (KF / 128) + c0 / 128 + g]);
        }
    }
    CHECK(row <= N, "load4: %d rows > %d", row, N);
    W8 w;
    w.N = N; w.K = K;
    w.q4 = dalloc<uint8_t>(q.size());
    CUDA_CHECK(cudaMemcpy(w.q4, q.data(), q.size(), cudaMemcpyHostToDevice));
    w.s = up_f32(sc);
    return w;
}
static float* up_rows_f32(const std::string& n, int r0, int r1) {
    std::vector<float> v = host_f32(n);
    const size_t cols = v.size() / T_(n).shape[0];
    return up_f32(std::vector<float>(v.begin() + (size_t)r0 * cols, v.begin() + (size_t)r1 * cols));
}

// ---------------------------------------------------------------- worker

__global__ void k_gather_rows8(const int8_t* q, const float* sc, const int* rows, int K, int8_t* oq, float* os) {
    const int r = blockIdx.x, src = rows[r];
    for (int i = threadIdx.x; i < K / 16; i += blockDim.x)
        reinterpret_cast<int4*>(oq + (size_t)r * K)[i] = reinterpret_cast<const int4*>(q + (size_t)src * K)[i];
    for (int i = threadIdx.x; i < K / 128; i += blockDim.x) os[(size_t)r * (K / 128) + i] = sc[(size_t)src * (K / 128) + i];
}
static void gather_rows8(const int8_t* q, const float* sc, const int* rows, int n, int K, int8_t* oq, float* os) {
    k_gather_rows8<<<n, 128>>>(q, sc, rows, K, oq, os);
    CUDA_CHECK(cudaGetLastError());
}
void Worker::init(Engine* e, const Plan& pl, int device) {
    E = e;
    p = pl;
    dev = device;
    CUDA_CHECK(cudaSetDevice(dev));
    CUDA_CHECK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&cst, cudaStreamNonBlocking));
    {
        int lo = 0, hi = 0;
        CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&lo, &hi));
        CUDA_CHECK(cudaStreamCreateWithPriority(&ast, cudaStreamNonBlocking, hi));
        CUDA_CHECK(cudaStreamCreateWithPriority(&cms, cudaStreamNonBlocking, hi));
        CUDA_CHECK(cudaStreamCreateWithPriority(&ags, cudaStreamNonBlocking, hi));
        // review r20-L1: [4] 64-bit flags (call | tag << 32), then the block counter
        CUDA_CHECK(cudaMalloc(&agf, 12 * sizeof(unsigned)));
        CUDA_CHECK(cudaMemset(agf, 0, 12 * sizeof(unsigned)));
        agdone = agf + 8;
        CUDA_CHECK(cudaStreamCreateWithPriority(&bst, cudaStreamNonBlocking, lo));
        CUDA_CHECK(cudaEventCreateWithFlags(&ev_bg, cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&ev_sel, cudaEventDisableTiming));
        for (int j = 0; j < 4; j++) CUDA_CHECK(cudaEventCreateWithFlags(&ev_c[j], cudaEventDisableTiming));
        for (int j = 0; j < PNMAX; j++) {
            CUDA_CHECK(cudaEventCreateWithFlags(&ev_pa[j], cudaEventDisableTiming));
            CUDA_CHECK(cudaEventCreateWithFlags(&ev_pm[j], cudaEventDisableTiming));
            CUDA_CHECK(cudaEventCreateWithFlags(&ev_ps[j], cudaEventDisableTiming));
            CUDA_CHECK(cudaEventCreateWithFlags(&ev_oa[j], cudaEventDisableTiming));
            CUDA_CHECK(cudaEventCreateWithFlags(&ev_om[j], cudaEventDisableTiming));
        }
        for (int g = 0; g < 2; g++) {
            CUDA_CHECK(cudaEventCreateWithFlags(&ev_af[g], cudaEventDisableTiming));
            CUDA_CHECK(cudaEventCreateWithFlags(&ev_aj[g], cudaEventDisableTiming));
            for (int j = 0; j < 4; j++) CUDA_CHECK(cudaEventCreateWithFlags(&ev_side[g][j], cudaEventDisableTiming));
        }
        CUDA_CHECK(cudaEventCreateWithFlags(&ev_ah, cudaEventDisableTiming));
        CUDA_CHECK(cudaStreamCreateWithPriority(&pst, cudaStreamNonBlocking, hi));
        for (int j = 0; j < 2; j++) CUDA_CHECK(cudaEventCreateWithFlags(&ev_l2[j], cudaEventDisableTiming));
    }
    CUBLAS_CHECK(cublasCreate(&cb));
    CUBLAS_CHECK(cublasSetStream(cb, st));
    CUBLAS_CHECK(cublasSetMathMode(cb, CUBLAS_DEFAULT_MATH));  // fp32 SGEMM stays fp32 (no TF32)
    kernels_init();
    ctr.seq = dalloc<unsigned>(1);
    ctr.done = dalloc<unsigned>(2);
    ctr.go = dalloc<unsigned>(1);
    ctr.bar = dalloc<unsigned>(2);
    ctr.ss = dalloc<float>(kCommBlocks * kMaxBatchRows);
    ctr.epoch = dalloc<unsigned>(1);
    ctr.ready = dalloc<unsigned>(4096);

    const int nh = p.nh(), ni = p.ne(), nsh = p.ns(), nd = p.d1 - p.d0;
    g_mem = &mem;
    g_tag = "tables / counters";
    sl.set(ni);
    slots_all = dalloc<int>((size_t)(NL + 1) * NE);
    CUDA_CHECK(cudaMemset(slots_all, 0xff, (size_t)(NL + 1) * NE * 4));
    d_seq = dalloc<unsigned>(1);
    d_wflag = dalloc<unsigned>(2 * (NL + 1));
    CUDA_CHECK(cudaMemset(d_wflag, 0xff, 2 * (NL + 1) * 4));
    d_bgdone = dalloc<int>(1);
    CUDA_CHECK(cudaMemset(d_bgdone, 0, 4));
    d_bgneed = dalloc<int>(NL + 1);
    CUDA_CHECK(cudaMemset(d_bgneed, 0, (NL + 1) * 4));
    d_pos = dalloc<int>(1);
    CUDA_CHECK(cudaMalloc(&cbws, 32 << 20));
    mem["cuBLAS workspace"] += 32 << 20;
    CUBLAS_CHECK(cublasSetWorkspace(cb, cbws, 32 << 20));
    // round 20: owner work of the row-owner prefill on its own stream / cuBLAS handle (next to st's attention); review r21
    // L1: only with the opt-in owner mode (its 32 MB workspace and handle cost expert slots otherwise)
    if (E->pf_own && p.ngpu > 1) {
        int lo_ = 0, hi_ = 0;
        CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&lo_, &hi_));
        CUDA_CHECK(cudaStreamCreateWithPriority(&ost, cudaStreamNonBlocking, hi_));
        CUBLAS_CHECK(cublasCreate(&cbo));
        CUBLAS_CHECK(cublasSetStream(cbo, ost));
        CUBLAS_CHECK(cublasSetMathMode(cbo, CUBLAS_DEFAULT_MATH));
        CUDA_CHECK(cudaMalloc(&cbows, 32 << 20));
        mem["cuBLAS workspace"] += 32 << 20;  // review r20b-L4: the owner handle's workspace in the VRAM report too
        CUBLAS_CHECK(cublasSetWorkspace(cbo, cbows, 32 << 20));
        for (int j = 0; j < PNMAX; j++) CUDA_CHECK(cudaEventCreateWithFlags(&ev_of[j], cudaEventDisableTiming));
    }
    d_nj = dalloc<int>(4);
    d_posb = dalloc<int>(1);
    d_am2 = dalloc<int>(4 * MR);
    d_mse = dalloc<float>(MR);
    bnd_scr = dalloc<float>((size_t)8 * 128 * 36 + 8 * 36 + 64);
    bnd_bar = dalloc<unsigned>(2);
    CUDA_CHECK(cudaMemset(bnd_bar, 0, 8));
    d_am = dalloc<int>(2 * 256);
    d_pids = dalloc<int>(MR * TOPK);
    d_pw = dalloc<float>(MR * TOPK);
    pred_part = dalloc<float>((size_t)8 * MR * NE + 64);  // RKS (glm_kernels.cu) K-slices  // next-layer prediction router on the side stream
    pred_lg = dalloc<float>((size_t)MR * NE + 64);
    lmix = dalloc<float>((size_t)MR * MIXS);
    lpost = dalloc<float>((size_t)MR * 4);
    lcomb = dalloc<float>((size_t)MR * 16);
    lxf = dalloc<float>((size_t)MR * H);
    lxb = dalloc<bf16>((size_t)MR * H);
    lw = dalloc<float>((size_t)MR * TOPK);
    d_lids = dalloc<int>((size_t)MR * TOPK);
    for (int j = 0; j < 2; j++) CUDA_CHECK(cudaEventCreateWithFlags(&ev_late[j], cudaEventDisableTiming));
    mla_scr = dalloc<float>(mla_split_scratch(p.nh(), MR) + 64);
    if (env_int("GLM_MLA_UBF", 1)) wu32 = dalloc<float>((size_t)p.nh() * (DN + DV) * KVLR + 64);  // prefill: fp32 wuk / wuv of one layer
    L.resize(E->mtp ? MTP_L + 1 : E->nl);
    std::vector<int> lays;
    for (int l = 0; l < E->nl; l++) lays.push_back(l);
    if (E->mtp) lays.push_back(MTP_L);
    for (int l : lays) {
        Layer& Y = L[l];
        const std::string b = P + "layers." + std::to_string(l) + ".";
        Y.mla = is_mla(l);
        Y.dense = is_dense(l);
        const char* blk[2] = {"attn", "ffn"};
        g_tag = l == MTP_L ? "MTP layer (non-expert)" : "mHC + norms";
        for (int k = 0; k < 2 && l != MTP_L; k++) {
            Y.hc_fn[k] = up_f32(b + "hc_" + blk[k] + "_fn");
            Y.hc_base[k] = up_f32(b + "hc_" + blk[k] + "_base");
            Y.hc_scale[k] = up_f32(b + "hc_" + blk[k] + "_scale");
        }
        Y.in_ln = up_f32(b + "input_layernorm.weight");
        Y.post_ln = up_f32(b + "post_attention_layernorm.weight");
        const std::string a = b + "self_attn.";
        if (l != MTP_L) g_tag = Y.mla ? "MLA weights" : "KDA weights";
        if (!Y.mla) {
            const int HK = nh * KD;
            const int rows = (3 * HK + 2 * KD + nh + 63) / 64 * 64;  // padded: aligned GEMM output rows
            // round 9: GPTQ INT4 KDA projections when the checkpoint carries the codes (GLM_KDA_INT4=0: INT8 of the BF16)
            const bool k4 = g_st->has(a + "q_proj.qcodes") && env_int("GLM_KDA_INT4", 1) != 0;
            if (k4) {
                Y.kw_in = load4({{a + "q_proj.", p.h0 * KD, p.h1 * KD}, {a + "k_proj.", p.h0 * KD, p.h1 * KD},
                                 {a + "v_proj.", p.h0 * KD, p.h1 * KD}, {a + "f_a_proj.", 0, KD}, {a + "g_a_proj.", 0, KD},
                                 {a + "b_proj.", p.h0, p.h1}}, rows, 0, H);
            } else {
            bf16* kwt = dalloc<bf16>((size_t)rows * H);
            cp_rows(kwt, a + "q_proj.weight", p.h0 * KD, p.h1 * KD);
            cp_rows(kwt + (size_t)HK * H, a + "k_proj.weight", p.h0 * KD, p.h1 * KD);
            cp_rows(kwt + (size_t)2 * HK * H, a + "v_proj.weight", p.h0 * KD, p.h1 * KD);
            cp_rows(kwt + (size_t)3 * HK * H, a + "f_a_proj.weight", 0, KD);
            cp_rows(kwt + (size_t)(3 * HK + KD) * H, a + "g_a_proj.weight", 0, KD);
            cp_rows(kwt + (size_t)(3 * HK + 2 * KD) * H, a + "b_proj.weight", p.h0, p.h1);
            Y.kw_in = to8(kwt, rows, H, env_int("GLM_KDA_W4F", 0) != 0);
            }
            {   // f_b / g_b stored transposed [128][nh 128]: coalesced inline GEMVs at decode, OP_N sgemm in prefill
                float* fb = up_rows_f32(a + "f_b_proj.weight", p.h0 * KD, p.h1 * KD);
                float* gb = up_rows_f32(a + "g_b_proj.weight", p.h0 * KD, p.h1 * KD);
                Y.kf_b = dalloc<float>((size_t)nh * KD * KD);
                Y.kg_b = dalloc<float>((size_t)nh * KD * KD);
                transpose(fb, Y.kf_b, nh * KD, KD, 0);
                transpose(gb, Y.kg_b, nh * KD, KD, 0);
                CUDA_CHECK(cudaDeviceSynchronize());
                CUDA_CHECK(cudaFree(fb));
                CUDA_CHECK(cudaFree(gb));
                mem_add(-2.0 * nh * KD * KD * 4);
            }
            {
                std::vector<float> cw;
                for (const char* x : {"q", "k", "v"}) {
                    std::vector<float> c = host_f32(a + x + "_conv1d.weight");  // [8192][1][4]
                    cw.insert(cw.end(), c.begin() + (size_t)p.h0 * KD * CONV, c.begin() + (size_t)p.h1 * KD * CONV);
                }
                Y.kconv = up_f32(cw);
            }
            Y.kA = up_rows_f32(a + "A_log", p.h0, p.h1);
            Y.kdt = up_rows_f32(a + "dt_bias", p.h0 * KD, p.h1 * KD);
            Y.ko_norm = up_f32(a + "o_norm.weight");
            Y.ko = k4 ? load4({{a + "o_proj.", 0, H}}, H, p.h0 * KD, p.h1 * KD)
                      : to8(up_cols(a + "o_proj.weight", p.h0 * KD, p.h1 * KD), H, HK, env_int("GLM_KDA_W4F", 0) != 0);
            g_tag = "KDA state + verify snapshots";
            Y.S = dalloc<float>((size_t)nh * KD * KD);
            Y.snap = dalloc<float>((size_t)MR * nh * KD * KD);
            Y.conv_rows = dalloc<float>((size_t)MR * 3 * HK);
            Y.conv_hist = dalloc<float>((size_t)3 * 3 * HK);
        } else {
            bf16* mwt = dalloc<bf16>((size_t)MLA_IN * H);
            size_t o = 0;
            cp_rows(mwt, a + "q_a_proj.weight", 0, QLR); o += QLR;
            cp_rows(mwt + o * H, a + "kv_a_proj_with_mqa.weight", 0, KVLR); o += KVLR;
            cp_rows(mwt + o * H, a + "indexer.wk.weight", 0, ID); o += ID;
            cp_rows(mwt + o * H, a + "indexer.weights_proj.weight", 0, IH); o += IH;
            cp_rows(mwt + o * H, a + "indexer.index_kpool_compress_gate", 0, ID);
            Y.mw_in = to8(mwt, MLA_IN, H);
            Y.q_a_ln = up_f32(a + "q_a_layernorm.weight");
            Y.kv_a_ln = up_f32(a + "kv_a_layernorm.weight");
            Y.ik_w = up_f32(a + "indexer.k_norm.weight");
            Y.ik_b = up_f32(a + "indexer.k_norm.bias");
            Y.ape = up_f32(a + "indexer.index_kpool_compress_ape");
            bf16* mqt = dalloc<bf16>((size_t)(nh * DN + IH * ID) * QLR);
            cp_rows(mqt, a + "q_b_proj.weight", p.h0 * DN, p.h1 * DN);
            cp_rows(mqt + (size_t)nh * DN * QLR, a + "indexer.wq_b.weight", 0, IH * ID);
            Y.mw_q = to8(mqt, nh * DN + IH * ID, QLR);
            {
                std::vector<float> kvb = host_f32(a + "kv_b_proj.weight");  // [64 * 512][512]
                std::vector<float> uk((size_t)nh * DN * KVLR), uv((size_t)nh * DV * KVLR);
                for (int hh = 0; hh < nh; hh++) {
                    const float* src = kvb.data() + (size_t)(p.h0 + hh) * (DN + DV) * KVLR;
                    memcpy(uk.data() + (size_t)hh * DN * KVLR, src, (size_t)DN * KVLR * 4);
                    memcpy(uv.data() + (size_t)hh * DV * KVLR, src + (size_t)DN * KVLR, (size_t)DV * KVLR * 4);
                }
                if (!env_int("GLM_MLA_UBF", 1)) {
                    Y.wuk = up_f32(uk);
                    Y.wuv = up_f32(uv);
                } else {  // round 8: bf16 only (exact: kv_b_proj is BF16 in the checkpoint); prefill widens per layer
                    auto upb = [](const std::vector<float>& v) {
                        std::vector<bf16> b(v.size());
                        for (size_t i = 0; i < v.size(); i++) b[i] = __float2bfloat16(v[i]);
                        bf16* d = dalloc<bf16>(b.size());
                        CUDA_CHECK(cudaMemcpy(d, b.data(), b.size() * 2, cudaMemcpyHostToDevice));
                        return d;
                    };
                    Y.wuk_b = upb(uk);
                    Y.wuv_b = upb(uv);
                }
            }
            Y.mo = to8(up_cols(a + "o_proj.weight", p.h0 * DV, p.h1 * DV), H, nh * DV);
            const size_t C = E->ctx;
            g_tag = "MLA cache (latent, ring, pools)";
            if (!E->mla_pos) {  // round 26: else position-major in the slab (Engine::load, after the slot count)
                Y.kc = dalloc<bf16>(C * KVLR);
                Y.pk = dalloc<float>(C / POOL * ID + ID);
            }
            Y.ki = dalloc<float>((size_t)E->kring * ID);
            Y.gsc = dalloc<float>((size_t)E->kring * ID);
        }
        const std::string m = b + "mlp.";
        if (l != MTP_L) g_tag = Y.dense ? "dense MLP (layers 0-2)" : "router + shared expert";
        if (Y.dense) {
            bf16* dt_ = dalloc<bf16>((size_t)2 * nd * H);
            cp_rows(dt_, m + "gate_proj.weight", p.d0, p.d1);
            cp_rows(dt_ + (size_t)nd * H, m + "up_proj.weight", p.d0, p.d1);
            Y.dw_gu = to8(dt_, 2 * nd, H);
            Y.dw_d = to8(up_cols(m + "down_proj.weight", p.d0, p.d1), H, nd);
        } else {
            Y.router = up_f32(m + "gate.weight");
            Y.rbias = up_f32(m + "gate.e_score_correction_bias");
            bf16* st_ = dalloc<bf16>((size_t)2 * nsh * H);
            cp_rows(st_, m + "shared_experts.gate_proj.weight", p.s0, p.s1);
            cp_rows(st_ + (size_t)nsh * H, m + "shared_experts.up_proj.weight", p.s0, p.s1);
            Y.sw_gu = to8(st_, 2 * nsh, H);
            Y.sw_d = to8(up_cols(m + "shared_experts.down_proj.weight", p.s0, p.s1), H, nsh);
            Y.slot = slots_all + (size_t)l * NE;
        }
    }
    g_tag = "lm_head + final norm";
    final_norm = up_f32(P + "norm.weight");
    {   // commit tables of the KDA layers
        std::vector<float*> a, b, c_, d;
        for (auto& Y : L)
            if (Y.S) { a.push_back(Y.S); b.push_back(Y.snap); c_.push_back(Y.conv_hist); d.push_back(Y.conv_rows); }
        nkda = (int)a.size();
        auto up = [&](std::vector<float*>& v) {
            float** p_ = dalloc<float*>(v.size() + 1);
            CUDA_CHECK(cudaMemcpy(p_, v.data(), v.size() * sizeof(float*), cudaMemcpyHostToDevice));
            return p_;
        };
        d_kS = up(a); d_ksnap = up(b); d_khist = up(c_); d_krows = up(d);
    }
    g_tag = "tables / counters";
    d_acc = dalloc<int>(2);  // [0] accepted rows, [1] pending KDA snapshot (round 11)
    CUDA_CHECK(cudaMemset(d_acc, 0, 8));
    hfin = dalloc<float>((size_t)MR * H);
    g_tag = "MTP layer (non-expert)";
    if (E->mtp) {
        const std::string b = P + "layers." + std::to_string(MTP_L) + ".";
        bf16* t = dalloc<bf16>((size_t)H * 2 * H);
        cp_rows(t, b + "eh_proj.weight", 0, H);
        mw.eh = to8(t, H, 2 * H);
        mw.enorm = up_f32(b + "enorm.weight");
        mw.hnorm = up_f32(b + "hnorm.weight");
        mw.head_norm = up_f32(b + "shared_head.norm.weight");
        CUDA_CHECK(cudaMalloc(&mtp_slab, (size_t)NE * sl.bytes));
        mem["MTP experts (resident)"] += (double)NE * sl.bytes;
        std::vector<int> id(NE);
        for (int e = 0; e < NE; e++) id[e] = e;
        CUDA_CHECK(cudaMemcpy(slots_all + (size_t)MTP_L * NE, id.data(), NE * 4, cudaMemcpyHostToDevice));
        mx_ = dalloc<float>((size_t)MR * H);
        mh = dalloc<float>((size_t)MR * H);
        mhn = dalloc<float>((size_t)MR * H);
        xcat = dalloc<bf16>((size_t)MR * 2 * H);
        d_mtok = dalloc<int>(MR);
        d_mpos = dalloc<int>(1);
        d_mseq = dalloc<unsigned>(1);
        d_chres = dalloc<int>(64);  // round 19 device chain
        CUDA_CHECK(cudaMemset(d_chres, 0, 64 * 4));
        d_chpos = dalloc<int>(MR);
        for (auto& c : chs) CUDA_CHECK(cudaStreamCreateWithFlags(&c, cudaStreamNonBlocking));
        mtp_cnt = dalloc<unsigned long long>(NE);
    }
    {
        // lm_head in row blocks (bf16 temp of one block at a time)
        const int nv = p.v1 - p.v0;
        lm.N = nv; lm.K = H;
        lm.q = dalloc<int8_t>((size_t)nv * H);
        lm.s = dalloc<float>((size_t)nv * (H / 128));
        bf16* t = dalloc<bf16>((size_t)8192 * H);
        for (int r0 = 0; r0 < nv; r0 += 8192) {
            const int r1 = std::min(nv, r0 + 8192);
            cp_rows(t, "lm_head.weight", p.v0 + r0, p.v0 + r1);
            quant8(t, r1 - r0, H, lm.q + (size_t)r0 * H, lm.s + (size_t)r0 * (H / 128), 0);
            CUDA_CHECK(cudaDeviceSynchronize());
        }
        CUDA_CHECK(cudaFree(t));
        mem_add(-(double)((size_t)8192 * H * sizeof(bf16) + 16));  // review r21b-L5: the temp leaves the VRAM report too
    }
    if (env_int("GLM_MTP_Q4", 0) && E->mtp) {
        CHECK(env_int("GLM_W8A8", 1) != 0, "GLM_MTP_Q4=1 needs the W8A8 path (GLM_W8A8=1)");  // r12 (review C7)  // round 11: draft-only INT4 copies (lm_head slice + MTP eh); verify untouched;
        // opt-in (manager decision): MTP -0.2..0.3 ms, acceptance up on 6 of 7 prompts, flashh 2.25 -> 2.07 tok/step
        auto conv = [&](W8& a) {
            W8 b; b.N = a.N; b.K = a.K;
            b.q4 = dalloc<uint8_t>((size_t)a.N * a.K / 2);
            b.s = dalloc<float>((size_t)a.N * (a.K / 128));
            int8_to4(a.q, a.s, a.N, a.K, b.q4, b.s, 0);
            CUDA_CHECK(cudaDeviceSynchronize());
            return b;
        };
        if (draft_vocab_path() == "0") lm4 = conv(lm);  // r12 (review C7): unused when the draft vocabulary head (GLM_DRAFT_VOCAB) is on
        if (mw.eh.q) {  // replaced in place: the INT4 eh also feeds the MTP prefill (MTP KV rows of the prompt; draft-only)
            W8 e4 = conv(mw.eh);
            CUDA_CHECK(cudaFree(mw.eh.q)); CUDA_CHECK(cudaFree(mw.eh.s));
            mem_add(-(double)((size_t)mw.eh.N * mw.eh.K + (size_t)mw.eh.N * (mw.eh.K / 128) * 4));
            mw.eh = e4;
        }
    }
    {   // draft-only vocabulary for the MTP head (GLM_DRAFT_VOCAB = file of sorted int32 token ids, "0" = full vocab):
        // the drafter's argmax over a frequent-token subset reads ~1/3 of the lm_head; verify still uses the full head
        // opt-in: flashh acceptance 2.10 -> 1.90 tok/step with the 49K list (narrative / code unchanged), MTP -0.1..0.2 ms
        const std::string path = E->mtp ? draft_vocab_path() : std::string("0");  // review r21b L3: no MTP, no draft head
        if (E->mtp && path == "0" && !env_str("GLM_DRAFT_VOCAB") && p.rank == 0)
            fprintf(stderr, "[glm] warning: MTP draft vocabulary list not found (%s); the drafts read the full head\n",
                    glm_data_file("draft_vocab_49k.i32").c_str());
        FILE* f = path == "0" ? nullptr : fopen(path.c_str(), "rb");
        if (f) {
            std::vector<int> all, ids;
            int v;
            while (fread(&v, 4, 1, f) == 1)
                if (v >= 0 && v < VOCAB) all.push_back(v);  // review r21b L2: ids inside the vocabulary, each once
            fclose(f);
            std::sort(all.begin(), all.end());
            all.erase(std::unique(all.begin(), all.end()), all.end());
            // round 21 [glm-r21-dvbal] (GLM_DRAFT_BAL, default 1): the list round-robin over the cards (each reads ~1/3 of
            // it; the old split by vocabulary range put 42.9K of 51.6K frequent rows on card 0); rows of other cards'
            // vocabulary ranges come from the checkpoint, quantized as the lm_head (quant8 per row: identical rows)
            const bool bal = env_int("GLM_DRAFT_BAL", 1) != 0;
            for (size_t i = 0; i < all.size(); i++)
                if (bal ? (int)(i % p.ngpu) == p.rank : (all[i] >= p.v0 && all[i] < p.v1)) ids.push_back(all[i]);
            const int nd = (int)ids.size();
            h_dmap = ids;
            d_dmap = dalloc<int>(nd);
            CUDA_CHECK(cudaMemcpy(d_dmap, ids.data(), (size_t)nd * 4, cudaMemcpyHostToDevice));
            lm_d.N = nd; lm_d.K = H;
            lm_d.q = dalloc<int8_t>((size_t)nd * H);
            lm_d.s = dalloc<float>((size_t)nd * (H / 128));
            if (bal) {
                const int B = 2048;
                bf16* t = dalloc<bf16>((size_t)B * H);
                for (int r0 = 0; r0 < nd; r0 += B) {
                    const int r1 = std::min(nd, r0 + B);
                    for (int r = r0; r < r1; r++) cp_rows(t + (size_t)(r - r0) * H, "lm_head.weight", ids[r], ids[r] + 1);
                    quant8(t, r1 - r0, H, lm_d.q + (size_t)r0 * H, lm_d.s + (size_t)r0 * (H / 128), 0);
                    CUDA_CHECK(cudaDeviceSynchronize());
                }
                CUDA_CHECK(cudaFree(t));
                mem_add(-(double)((size_t)B * H * sizeof(bf16) + 16));  // review r21b-L5
            } else {
                std::vector<int> loc(nd);
                for (int i = 0; i < nd; i++) loc[i] = ids[i] - p.v0;
                int* d_loc = dalloc<int>(nd);
                CUDA_CHECK(cudaMemcpy(d_loc, loc.data(), (size_t)nd * 4, cudaMemcpyHostToDevice));
                gather_rows8(lm.q, lm.s, d_loc, nd, H, lm_d.q, lm_d.s);
                CUDA_CHECK(cudaDeviceSynchronize());
                CUDA_CHECK(cudaFree(d_loc));
                mem_add(-(double)((size_t)nd * 4 + 16));
            }
            fprintf(stderr, "[glm] MTP draft vocabulary %s: %d rows on card %d%s\n", path.c_str(), nd, dev, bal ? " (balanced)" : "");
        }
    }
    sl.set(ni);
    // pinned host mirrors (largest chunk), prefill staging and the DMA landing ring (persistent: seeding uses stage[0])
    const int rows = E->chunk;
    CUDA_CHECK(cudaMallocHost(&h_rall, (size_t)rows * TOPK * 4));
    CUDA_CHECK(cudaMallocHost(&h_wall, (size_t)rows * TOPK * 4));
    CUDA_CHECK(cudaMallocHost(&h_ids, (size_t)rows * TOPK * 4));
    CUDA_CHECK(cudaMallocHost(&h_w, (size_t)rows * TOPK * 4));
    CUDA_CHECK(cudaMallocHost(&h_jobs, 2 * JR * TOPK * sizeof(XJob)));
    CUDA_CHECK(cudaMallocHost(&h_gujobs, NE * sizeof(GuJob)));
    g_tag = "prefill staging + dequant";
    for (int b = 0; b < 3; b++) stage[b] = (char*)dalloc<char>(sl.bytes);
    for (int j = 0; j < NLAND; j++) {
        land[j] = (char*)dalloc<char>(sl.bytes);
        CUDA_CHECK(cudaEventCreateWithFlags(&ev_land[j], cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&ev_lfree[j], cudaEventDisableTiming));
        if (j == 0) {
            CUDA_CHECK(cudaEventCreateWithFlags(&ev_staged, cudaEventDisableTiming));
            CUDA_CHECK(cudaEventCreateWithFlags(&ev_moe_done, cudaEventDisableTiming));
            for (int g = 0; g < NGRP; g++) CUDA_CHECK(cudaEventCreateWithFlags(&ev_grp[g], cudaEventDisableTiming));
        }
    }
    CUDA_CHECK(cudaStreamCreateWithFlags(&dmst, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&gst, cudaStreamNonBlocking));
    {   // r12 (prefill review F11): the KDA recurrence of part i at high priority next to part i + 1's GEMMs (its 88
        // CTAs queued behind GEMM CTAs at default priority); GLM_PF_KRHI=0: on gst as before
        int lo_ = 0, hi_ = 0;
        CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&lo_, &hi_));
        if (env_int("GLM_PF_KRHI", 1)) CUDA_CHECK(cudaStreamCreateWithPriority(&krst, cudaStreamNonBlocking, hi_));
        else krst = gst;
    }
    CUDA_CHECK(cudaEventCreateWithFlags(&ev_q8, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&ev_ids, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&ev_srs, cudaEventDisableTiming));
    for (int j = 0; j < 2; j++) {
        CUDA_CHECK(cudaEventCreateWithFlags(&ev_ka[j], cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&ev_kb[j], cudaEventDisableTiming));
    }
    for (int j = 0; j < 256; j++) {
        CUDA_CHECK(cudaEventCreateWithFlags(&ev_sr[j], cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&ev_sf[j], cudaEventDisableTiming));
    }
    for (int b = 0; b < 2; b++) {
        CUDA_CHECK(cudaEventCreateWithFlags(&ev_deq[b], cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&ev_use[b], cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&ev_gu[b], cudaEventDisableTiming));
    }
    CUDA_CHECK(cudaDeviceSynchronize());
}

int Worker::pf_arows(int rows) const {
    if (!E->pf_pn || !E->pf_w8 || p.ngpu < 2 || rows < 4096) return rows;
    const int n = std::max(2, (rows + E->pf_part - 1) / E->pf_part);
    return ((rows + n - 1) / n + 63) / 64 * 64;
}
void Worker::alloc_acts(int rows, int arows) {
    CUDA_CHECK(cudaSetDevice(dev));
    g_mem = &mem;
    g_tag = rows > 2 * MR ? "activations (prefill chunk)" : "activations (decode rows)";
    const bool pf = rows > 2 * MR;  // decode set: up to 2 sequences x MR rows
    mx = rows;
    if (arows <= 0) arows = rows;
    mxa = arows;
    const int wrows = rows;  // per-row buffers of the whole window
    rows = arows;            // attention / dense-MLP scratch: one part
    const int nh = p.nh();
    R = dalloc<float>((size_t)wrows * RW);
    mix = dalloc<float>((size_t)wrows * MIXS);
    post = dalloc<float>((size_t)wrows * 4);
    // round 16 (review r15 H1 / L1): pdef is a Worker field, not part of Acts, so only the decode set allocates it (a
    // prefill set would leave it in the lent slab tail after unlend() and change the prefill bump layout)
    if (!pf && !g_bump.on) pdef = dalloc<float>((size_t)MR * 4);
    comb = dalloc<float>((size_t)wrows * 16);
    xf = dalloc<float>((size_t)wrows * H);
    xb = dalloc<bf16>((size_t)wrows * H);
    part = dalloc<float>((size_t)wrows * H);
    sum = dalloc<float>((size_t)wrows * H);
    // k_wait_add reads / writes part and sum as float4 (row views at r0 * H keep the 16-byte alignment); review r13 L6
    CHECK(((uintptr_t)part | (uintptr_t)sum) % 16 == 0, "part / sum: float4 alignment");
    ssp = dalloc<float>((size_t)wrows * (H / 256));
    mixp = dalloc<float>((size_t)wrows * std::max(H / 256, merge_mix_pf_parts()) * (NMIX + 1));  // decode boundary / prefill merge_mix_pf partials
    // scratch: the largest per-op need (KDA: proj + gf + gate + q, k, v, g, o; MLA: proj + qq + qlat + ctx + ov; experts)
    const size_t kda = (size_t)(3 * nh * KD + 2 * KD + nh + 64) + 7 * nh * KD + nh +
                       (pf && E->pf_kc ? (size_t)nh * (kda_chunk_floats(KD, 1) / KD) + 64 : 0);  // chunked rec scratch
    const size_t mlaf = (size_t)MLA_IN + IH + (nh * DN + IH * ID) + 2 * nh * KVLR + nh * DV;
    const size_t mlp = (size_t)2 * std::max(p.d1 - p.d0, std::max(p.ne(), p.ns())) + H;
    // round 10: two KDA scratch sets in the parts path (the recurrence of part i next to part i + 1's projections)
    const size_t kdan = pf && E->pf_kov && !E->pf_kc && arows < wrows && wrows >= 24576 ? 2 * kda : kda;
    big_n = (size_t)rows * std::max(std::max(kdan, mlaf), mlp) + (size_t)rows * (E->ctx / POOL + 64);
    if (pf && E->pf_w8 && arows < wrows) big_n = std::max(big_n, (size_t)wrows * 2 * TOPK + 64);  // MoE W4A8: row lists
    else if (pf) big_n = std::max(big_n, (size_t)wrows * (2 * p.ne() + H + 2 * TOPK) + 64);  // MoE GEMM path
    big = dalloc<float>(big_n);
    bigb_n = (size_t)rows * std::max(QLR + nh * DV, (p.d1 - p.d0) + H);
    if (pf) bigb_n = std::max(bigb_n, (size_t)3 * std::max(p.ne(), p.ns()) * H);
    bigb = dalloc<bf16>(bigb_n + (pf ? (size_t)6 * p.ne() * H : 0));  // + 2 dequantized experts (GEMM path)
    logits = dalloc<float>((size_t)std::min(wrows, 256) * (p.v1 - p.v0));
    d_ids = dalloc<int>((size_t)wrows * TOPK);
    d_w = dalloc<float>((size_t)wrows * TOPK);
    d_tok = dalloc<int>(wrows);
    d_sel = dalloc<int>((size_t)rows * MAXSEL);
    d_nsel = dalloc<int>(rows);
    if (pf) {   // MLA prefill on tensor cores: union key lists, masks, scores of a query block; INT8 GEMM dequant
        ucap = env_int("GLM_MLA_UCAP", 8192);
        uw = E->ctx / 32 + 64;  // r12: per query block bitmap / prefix stride (words); every block of a part at once
        ublk_n = E->pf_ubatch ? (rows + UQB - 1) / UQB : 1;
        ubits = dalloc<unsigned>((size_t)ublk_n * uw);
        upre = dalloc<int>((size_t)ublk_n * uw);
        ukeys = dalloc<int>((size_t)E->ctx + 64);
        d_u = dalloc<int>(ublk_n);
        if (!h_ublk_pin) CUDA_CHECK(cudaMallocHost(&h_ublk_pin, 8192 * 4));  // review r16 M1: once per worker, never lost
        h_ublk = h_ublk_pin;
        CHECK(ublk_n <= 8192, "MLA union batch: %d blocks", ublk_n);
        ukb = dalloc<bf16>((size_t)(E->ctx + 64) * KVLR);
        umask = dalloc<unsigned char>((size_t)UQB * ucap + 64);
        uS = dalloc<float>((size_t)UQB * nh * ucap);
        uP = dalloc<bf16>((size_t)UQB * nh * ucap);
        uqb = dalloc<bf16>((size_t)UQB * nh * KVLR);
        psel = dalloc<int>((size_t)2 * rows * TOPB);
        h16x = dalloc<__half>((size_t)2048 * 6144);
        m8x = dalloc<int8_t>((size_t)wrows * H);
        m8s = dalloc<float>((size_t)wrows * (H / 128));
        m8h = dalloc<int8_t>((size_t)wrows * p.ne());
        m8hs = dalloc<float>((size_t)wrows * (p.ne() / 64));
        if (E->pf_gu2) {
            m8h2 = dalloc<int8_t>((size_t)wrows * p.ne());  // round 10: second h buffer (gate/up of the next expert)
            m8hs2 = dalloc<float>((size_t)wrows * (p.ne() / 64));
        }
        sr_n = E->pf_w8 && arows < wrows ? std::min(256, E->pf_ring) : 0;
        if (sr_n && E->pf_gub > 1) {  // round 10: h of every (row, expert) pair of a group (grouped gate/up)
            m8hg = dalloc<int8_t>((size_t)wrows * TOPK * p.ne());
            m8hgs = dalloc<float>((size_t)wrows * TOPK * (p.ne() / 64));
        }
        sring = sr_n ? dalloc<char>((size_t)sr_n * sl.bytes) : nullptr;
        if (!(E->pf_w8 && arows < wrows)) m16h = dalloc<__half>((size_t)wrows * TOPK * p.ne());  // h of every (row, expert) pair (batched gate/up)
        d_gujobs = dalloc<GuJob>(NE);
        if (E->pf_ahead) cstage = dalloc<char>((size_t)NE * sl.bytes);  // one staged block per expert id
        if (!(E->pf_w8 && arows < wrows)) m16x = dalloc<__half>((size_t)wrows * H);
        h16y = dalloc<__half>((size_t)2048 * 8192);
        pnsel = dalloc<int>((size_t)2 * rows);
        deq_n = (size_t)8192 * H;
        deq = dalloc<bf16>(deq_n);
    }
    if (!pf) g_tag = "decode scratch (expert GEMV, jobs)";
    qx = dalloc<int8_t>((size_t)2 * MAXR * 2 * H);
    qxs = dalloc<float>((size_t)2 * MAXR * (2 * H / 128));
    max_jobs = 2 * JR * TOPK;  // GEMV path: the step's hot + admitted lists (<= 32 each), generic rows <= JR
    d_jobs = dalloc<XJob>(pf ? max_jobs : std::max(max_jobs, 4 * MXE));  // decode: [group][hot | admitted]
    gu = dalloc<float>((size_t)max_jobs * KSPL * JR * 2 * p.ne());
    xhb = dalloc<float>((size_t)max_jobs * JR * p.ne());
    xyp = dalloc<float>((size_t)max_jobs * 2 * JR * H);
    d_njt = dalloc<int>(4);
}

size_t Worker::pf_bytes(int rows) {
    const Acts keep = *this;
    g_bump = Bump{true, nullptr, 0};
    alloc_acts(rows, pf_arows(rows));
    const size_t b = g_bump.off;
    g_bump = Bump{};
    static_cast<Acts&>(*this) = keep;
    clear_pf_fields();
    return b + 4096;
}
void Worker::use_pf(char* base, int rows) {
    CUDA_CHECK(cudaSetDevice(dev));
    g_bump = Bump{true, base, 0};
    alloc_acts(rows, pf_arows(rows));
    g_bump = Bump{};
    CUDA_CHECK(cudaDeviceSynchronize());
}
void Worker::save_bank(SeqBank& b) {
    b.ls.resize(L.size());
    for (size_t l = 0; l < L.size(); l++) {
        Layer& Y = L[l];
        b.ls[l] = LState{Y.S, Y.snap, Y.conv_rows, Y.conv_hist, Y.kc, Y.ki, Y.gsc, Y.pk};
    }
    b.d_pos = d_pos; b.d_acc = d_acc; b.d_mpos = d_mpos; b.hfin = hfin; b.mx_ = mx_; b.mh = mh; b.mhn = mhn;
    b.d_kS = d_kS; b.d_ksnap = d_ksnap; b.d_khist = d_khist; b.d_krows = d_krows;
    for (int i = 0; i < MR; i++) { b.gstep[i] = gstep[i]; b.gstep_s[i] = gstep_s[i]; b.gmtp[i] = gmtp[i]; }
    b.gcommit = gcommit;
}
void Worker::load_bank(const SeqBank& b) {
    for (size_t l = 0; l < L.size(); l++) {
        Layer& Y = L[l];
        const LState& x = b.ls[l];
        Y.S = x.S; Y.snap = x.snap; Y.conv_rows = x.conv_rows; Y.conv_hist = x.conv_hist;
        Y.kc = x.kc; Y.ki = x.ki; Y.gsc = x.gsc; Y.pk = x.pk;
    }
    d_pos = b.d_pos; d_acc = b.d_acc; d_mpos = b.d_mpos; hfin = b.hfin; mx_ = b.mx_; mh = b.mh; mhn = b.mhn;
    d_kS = b.d_kS; d_ksnap = b.d_ksnap; d_khist = b.d_khist; d_krows = b.d_krows;
    for (int i = 0; i < MR; i++) { gstep[i] = b.gstep[i]; gstep_s[i] = b.gstep_s[i]; gmtp[i] = b.gmtp[i]; }
    gcommit = b.gcommit;
}
void Worker::bind(int s) {
    if (s == cur_seq) return;
    save_bank(banks[cur_seq]);
    load_bank(banks[s]);
    cur_seq = s;
}
void Worker::alloc_seq() {
    CUDA_CHECK(cudaSetDevice(dev));
    if (banks.empty()) { banks.resize(1); save_bank(banks[0]); }
    SeqBank b;
    const int nh = p.nh(), HK = nh * KD;
    g_mem = &mem;
    g_tag = "second sequence (KDA, MLA caches)";
    b.ls.resize(L.size());
    std::vector<float*> a, bb, c_, d;
    for (size_t l = 0; l < L.size(); l++) {
        Layer& Y = L[l];
        LState& x = b.ls[l];
        if (Y.S) {
            x.S = dalloc<float>((size_t)nh * KD * KD);
            x.snap = dalloc<float>((size_t)MR * nh * KD * KD);
            x.conv_rows = dalloc<float>((size_t)MR * 3 * HK);
            x.conv_hist = dalloc<float>((size_t)3 * 3 * HK);
            CUDA_CHECK(cudaMemset(x.S, 0, (size_t)nh * KD * KD * 4));
            CUDA_CHECK(cudaMemset(x.conv_hist, 0, (size_t)9 * HK * 4));
            a.push_back(x.S); bb.push_back(x.snap); c_.push_back(x.conv_hist); d.push_back(x.conv_rows);
        }
        if (Y.kc) {  // (GLM_SEQS=2 keeps GLM_MLA_POS off: one ctx-sized array per layer and sequence)
            const size_t C = E->ctx;
            x.kc = dalloc<bf16>(C * KVLR);
            x.pk = dalloc<float>(C / POOL * ID + ID);
            x.ki = dalloc<float>((size_t)E->kring * ID);
            x.gsc = dalloc<float>((size_t)E->kring * ID);
        }
    }
    auto up = [&](std::vector<float*>& v) {
        float** p_ = dalloc<float*>(v.size() + 1);
        CUDA_CHECK(cudaMemcpy(p_, v.data(), v.size() * sizeof(float*), cudaMemcpyHostToDevice));
        return p_;
    };
    b.d_kS = up(a); b.d_ksnap = up(bb); b.d_khist = up(c_); b.d_krows = up(d);
    b.d_pos = dalloc<int>(1);
    CUDA_CHECK(cudaMemset(b.d_pos, 0, 4));
    b.d_acc = dalloc<int>(2);
    CUDA_CHECK(cudaMemset(b.d_acc, 0, 8));
    b.hfin = dalloc<float>((size_t)MR * H);
    if (E->mtp) {
        b.mx_ = dalloc<float>((size_t)MR * H);
        b.mh = dalloc<float>((size_t)MR * H);
        b.mhn = dalloc<float>((size_t)MR * H);
        b.d_mpos = dalloc<int>(1);
    }
    banks.push_back(b);
}
// review r16 L2: Worker-only prefill fields set by a prefill alloc_acts (slab tail or dry-run addresses) back to "none"
void Worker::clear_pf_fields() {
    sring = nullptr;
    cstage = nullptr;
    sr_n = 0;
}
void Worker::use_dec() {
    static_cast<Acts&>(*this) = dec_acts;
    clear_pf_fields();
}
void Worker::view(int r0) { view_from(*this, dec_acts, r0); }
void Worker::view_from(Acts& a, const Acts& b, int r0) {
    a.R = b.R + (size_t)r0 * RW; a.mix = b.mix + (size_t)r0 * MIXS;
    a.mixp = b.mixp ? b.mixp + (size_t)r0 * std::max(H / 256, merge_mix_pf_parts()) * (NMIX + 1) : nullptr; a.post = b.post + (size_t)r0 * 4;
    a.comb = b.comb + (size_t)r0 * 16; a.xf = b.xf + (size_t)r0 * H; a.xb = b.xb + (size_t)r0 * H;
    a.part = b.part + (size_t)r0 * H; a.sum = b.sum + (size_t)r0 * H; a.ssp = b.ssp + (size_t)r0 * (H / 256);
    a.d_ids = b.d_ids + (size_t)r0 * TOPK; a.d_w = b.d_w + (size_t)r0 * TOPK;
}
void Worker::gemm_f16(const __half* Wt, const __half* X, __half* Y, int N, int T, int K) {
    const __half one = __float2half(1.f), zero = __float2half(0.f);
    CUBLAS_CHECK(cublasGemmEx(cb, CUBLAS_OP_T, CUBLAS_OP_N, N, T, K, &one, Wt, CUDA_R_16F, K, X, CUDA_R_16F, K, &zero,
                              Y, CUDA_R_16F, N, CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT));
}
void Worker::gemm_bf16(const bf16* Wt, const bf16* X, float* Y, int N, int T, int K, int ldy, float beta) {
    const float one = 1.f;
    CUBLAS_CHECK(cublasGemmEx(cb, CUBLAS_OP_T, CUBLAS_OP_N, N, T, K, &one, Wt, CUDA_R_16BF, K, X, CUDA_R_16BF, K, &beta,
                              Y, CUDA_R_32F, ldy < 0 ? N : ldy, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
}
void Worker::lin(const W8& w, const float* xf32, const bf16* xb16, float* Y, int T, int ldy, bool acc, int ldx,
                 const int* dp, int nskip) {
    if (ldy < 0) ldy = w.N;
    if (T <= MAXR) {
        if (E->w8a8) {  // two-pass int8 activations + dp4a (the fp32 path is instruction-bound for several rows)
            const int lx = ldx < 0 ? w.K : ldx;
            if (xf32) quant_x<float>(xf32, lx, w.K, T, qx, qxs, st);
            else quant_x<bf16>(xb16, lx, w.K, T, qx, qxs, st);
            if (w.q4) gemv4q(w.q4, w.s, w.N, w.K, qx, qxs, Y, ldy, T, acc, st);
            else gemv8q(w.q, w.s, w.N, w.K, qx, qxs, Y, ldy, T, acc, st, dp, nskip);
        } else {
            CHECK(!w.q4, "INT4 weights need GLM_W8A8=1");
            gemv8(w.q, w.s, w.N, w.K, xf32, xb16, ldx < 0 ? w.K : ldx, Y, ldy, T, acc, st);
        }
        return;
    }
    CHECK(xb16 && ldx < 0, "lin: GEMM path needs bf16 rows");
    const int rows = (int)(deq_n / w.K);
    if (E->pf_d16 && h16x && w.K <= 6144 && !w.q4) {  // fp16 inputs and accumulation, rows in blocks of 2048, column blocks <= 8192
        __half* wh = (__half*)deq;
        const int nb = std::min(rows, 8192);
        for (int n0 = 0; n0 < w.N; n0 += nb) {
            const int nc = std::min(nb, w.N - n0);
            dequant8h(w.q + (size_t)n0 * w.K, w.s + (size_t)n0 * (w.K / 128), nc, w.K, wh, st);
            for (int t0 = 0; t0 < T; t0 += 2048) {
                const int tc = std::min(2048, T - t0);
                b2h_rows(xb16 + (size_t)t0 * w.K, w.K, w.K, tc, h16x, st);
                gemm_f16(wh, h16x, h16y, nc, tc, w.K);
                acc16(Y + (size_t)t0 * ldy + n0, ldy, h16y, nc, tc, acc, st);
            }
        }
        return;
    }
    for (int n0 = 0; n0 < w.N; n0 += rows) {
        const int nc = std::min(rows, w.N - n0);
        if (w.q4) dequant4(w.q4 + (size_t)n0 * (w.K / 2), w.s + (size_t)n0 * (w.K / 128), nc, w.K, deq, st);
        else dequant8(w.q + (size_t)n0 * w.K, w.s + (size_t)n0 * (w.K / 128), nc, w.K, deq, st);
        gemm_bf16(deq, xb16, Y + n0, nc, T, w.K, ldy, acc ? 1.f : 0.f);
    }
}

void Worker::sgemm(const float* Wt, const float* X, float* Y, int N, int T, int K, int ldw, int ldx, int ldy) {
    const float one = 1.f, zero = 0.f;
    CUBLAS_CHECK(cublasSgemm(cb, CUBLAS_OP_T, CUBLAS_OP_N, N, T, K, &one, Wt, ldw < 0 ? K : ldw, X, ldx < 0 ? K : ldx,
                             &zero, Y, ldy < 0 ? N : ldy));
}

void Worker::allreduce(int T) {
    if (p.ngpu == 1) {
        CUDA_CHECK(cudaMemcpyAsync(sum, part, (size_t)T * H * 4, cudaMemcpyDeviceToDevice, st));
        return;
    }
    if (ar_idx >= 0 && T <= MAXR) {
        if (E->ar_ll && E->ar_ll12) ar_ll12(part, sum, T * H, par, p.rank, p.ngpu, ar_seq ? ar_seq : d_seq, ar_idx++, st);
        else if (E->ar_ll) ar_ll(part, sum, T * H, par, p.rank, p.ngpu, ar_seq ? ar_seq : d_seq, ar_idx++, st);
        else ar_p2p(part, sum, T * H, par, p.rank, p.ngpu, ar_seq ? ar_seq : d_seq, ar_idx++, st);
        return;
    }
    CUDA_CHECK(cudaMemsetAsync(sum, 0, (size_t)T * H * 4, st));
    // prefill rows in blocks of E->ar_rows (the comm buffers stay small: they are not lent from the slab)
    for (int r0 = 0; r0 < T; r0 += E->ar_rows) {
        const int nr = std::min(E->ar_rows, T - r0);
        allreduce_add(sum + (size_t)r0 * H, part + (size_t)r0 * H, nr, H, E->comm, p.rank, ctr, st, nullptr, 0.f, nullptr,
                      ll_idx >= 0 && nr <= kMaxBatchRows ? ll_idx++ : -1);
    }
}

void Worker::mhc_pre(int l, int blk, int T, bool have_mix) {
    Layer& Y = L[l];
    if (!have_mix && E->pf_mm && T > MR && E->dump_dir.empty()) {  // mixes + sum of squares in one pass over R
        merge_mix_pf(R, post, comb, nullptr, Y.hc_fn[blk], mixp, T, st);
        have_mix = true;
    }
    if (have_mix) {  // partials mixp[t][16][25] of the merge (fixed summation order in k_mhc_pre)
        if (E->pf_mhc2) mhc_pre_pf(R, Y.hc_base[blk], Y.hc_scale[blk], blk ? Y.post_ln : Y.in_ln, post, comb, xf, xb, T, mixp,
                                   merge_mix_pf_parts(), st);
        else glm::mhc_pre(R, mix, Y.hc_base[blk], Y.hc_scale[blk], blk ? Y.post_ln : Y.in_ln, post, comb, xf, xb, T, st, true,
                          nullptr, mixp, merge_mix_pf_parts());
        return;
    }
    sgemm(Y.hc_fn[blk], R, mix, NMIX, T, RW, -1, -1, MIXS);
    // after a merge (every block but the first after the embedding) its partial sums of squares replace a pass over R
    glm::mhc_pre(R, mix, Y.hc_base[blk], Y.hc_scale[blk], blk ? Y.post_ln : Y.in_ln, post, comb, xf, xb, T, st, false,
                 E->pf_ss && (l > 0 || blk == 1) ? ssp : nullptr);
}
bool Worker::pmerge(int l, int blk, int T) {
    if (E->pf_mm && l < E->nl && T > MR && E->dump_dir.empty()) {
        merge_mix_pf(R, post, comb, sum, L[l].hc_fn[blk], mixp, T, st);
        return true;
    }
    glm::merge(R, post, comb, sum, T, st, ssp);
    return false;
}

// L2 prefetch of weight bytes (evict_last) on the side stream, next to a dependent chain that leaves DRAM idle
__global__ void k_l2pf(const char* p, size_t n) {
    for (size_t i = ((size_t)blockIdx.x * blockDim.x + threadIdx.x) * 128; i < n; i += (size_t)gridDim.x * blockDim.x * 128)
        asm volatile("prefetch.global.L2::evict_last [%0];" ::"l"(p + i));
}
void Worker::l2pf_fork(const void* ptr, size_t n) {
    if (!n) return;
    CUDA_CHECK(cudaEventRecord(ev_l2[0], st));
    CUDA_CHECK(cudaStreamWaitEvent(pst, ev_l2[0], 0));
    k_l2pf<<<E->l2pf_blocks, 256, 0, pst>>>((const char*)ptr, n);
    CUDA_CHECK(cudaEventRecord(ev_l2[1], pst));
    l2_rec = true;
}
void Worker::l2pf_join() {
    if (l2_rec) CUDA_CHECK(cudaStreamWaitEvent(st, ev_l2[1], 0));
    l2_rec = false;
}
// prefill KDA in two halves (round 10): kda_a = projections, gates, conv / prep of a part into scratch set z;
// the recurrence runs on gst; kda_b = output norm / gate and o_proj of the part
struct KdaBufs { float *proj, *gf, *gate, *q, *k, *v, *g, *o, *beta; };
static KdaBufs kda_bufs(float* base, int T, int ldp, int HK) {
    KdaBufs b;
    b.proj = base; b.gf = b.proj + (size_t)T * ldp; b.gate = b.gf + (size_t)T * HK; b.q = b.gate + (size_t)T * HK;
    b.k = b.q + (size_t)T * HK; b.v = b.k + (size_t)T * HK; b.g = b.v + (size_t)T * HK; b.o = b.g + (size_t)T * HK;
    b.beta = b.o + (size_t)T * HK;
    return b;
}
float* Worker::kda_set(int z) {
    const int nh = p.nh();
    const size_t kda = (size_t)(3 * nh * KD + 2 * KD + nh + 64) + 7 * nh * KD + nh;
    return big + (size_t)z * mxa * kda;
}
void Worker::kda_a(int l, int T, int z) {
    Layer& Y = L[l];
    const int nh = p.nh(), HK = nh * KD, ldp = Y.kw_in.N;
    KdaBufs b = kda_bufs(kda_set(z), T, ldp, HK);
    lin(Y.kw_in, xf, xb, b.proj, T);
    const float one = 1.f, zero = 0.f;
    CUBLAS_CHECK(cublasSgemm(cb, CUBLAS_OP_N, CUBLAS_OP_N, HK, T, KD, &one, Y.kf_b, HK, b.proj + 3 * HK, ldp, &zero, b.gf, HK));
    CUBLAS_CHECK(cublasSgemm(cb, CUBLAS_OP_N, CUBLAS_OP_N, HK, T, KD, &one, Y.kg_b, HK, b.proj + 3 * HK + KD, ldp, &zero,
                             b.gate, HK));
    kda_prep(b.proj, ldp, nh, Y.kconv, Y.conv_hist, b.gf, Y.kA, Y.kdt, b.q, b.k, b.v, b.g, b.beta, T, st, nullptr, 0, nullptr);
}
void Worker::kda_r(int l, int T, int z, cudaStream_t s_) {
    Layer& Y = L[l];
    const int nh = p.nh(), HK = nh * KD;
    KdaBufs b = kda_bufs(kda_set(z), T, Y.kw_in.N, HK);
    kda_rec(b.q, b.k, b.v, b.g, b.beta, Y.S, b.o, T, nh, s_, nullptr);
}
void Worker::kda_b(int l, int T, int z) {
    Layer& Y = L[l];
    const int nh = p.nh(), HK = nh * KD;
    KdaBufs b = kda_bufs(kda_set(z), T, Y.kw_in.N, HK);
    bf16* ob = bigb;
    kda_out(b.o, b.gate, Y.ko_norm, ob, T, nh, st);
    lin(Y.ko, nullptr, ob, part, T);
}
void Worker::kda(int l, int T, int n0, bool verify, int r0) {
    (void)n0;
    Layer& Y = L[l];
    const int nh = p.nh(), HK = nh * KD;
    const int ldp = Y.kw_in.N;
    float* proj = big;
    float* gf = proj + (size_t)T * ldp;
    float* gate = gf + (size_t)T * HK;
    float* q = gate + (size_t)T * HK;
    float* k = q + (size_t)T * HK;
    float* v = k + (size_t)T * HK;
    float* g = v + (size_t)T * HK;
    float* o = g + (size_t)T * HK;
    float* beta = o + (size_t)T * HK;
    lin(Y.kw_in, xf, xb, proj, T);
    stamp(3);
    if (verify && T <= MR && E->l2pf_ko) l2pf_fork(Y.ko.q, std::min(E->l2pf_ko, (size_t)Y.ko.N * Y.ko.K));
    const bool fz = verify && T <= MR && E->fuse_kda;  // decode: gate GEMVs inside prep / out
    const float one = 1.f, zero = 0.f;
    if (fz) kda_gates(Y.kf_b, Y.kg_b, proj, ldp, HK, T, gf, gate, st);
    else {  // transposed [128][HK] weights: Y[T][HK] = X[T][128] W
        CUBLAS_CHECK(cublasSgemm(cb, CUBLAS_OP_N, CUBLAS_OP_N, HK, T, KD, &one, Y.kf_b, HK, proj + 3 * HK, ldp, &zero, gf, HK));
        CUBLAS_CHECK(cublasSgemm(cb, CUBLAS_OP_N, CUBLAS_OP_N, HK, T, KD, &one, Y.kg_b, HK, proj + 3 * HK + KD, ldp, &zero,
                                 gate, HK));
    }
    stamp(4);
    const size_t nst = (size_t)nh * KD * KD, C = (size_t)3 * nh * KD;
    // verify rows r0..: conv inputs of rows < r0 from conv_rows, state from the snapshot of row r0 - 1
    if (fz && E->fuse_kda > 1)
        kda_prep_f(proj, ldp, nh, Y.kconv, Y.conv_hist, Y.kf_b, Y.kA, Y.kdt, q, k, v, g, beta, T, st, Y.conv_rows + r0 * C, r0,
                   Y.conv_rows);
    else
        kda_prep(proj, ldp, nh, Y.kconv, Y.conv_hist, gf, Y.kA, Y.kdt, q, k, v, g, beta, T, st,
                 verify ? Y.conv_rows + r0 * C : nullptr, r0, verify ? Y.conv_rows : nullptr);
    stamp(5);
    if (!verify && E->pf_kc && T >= 64)  // round 10 (P-5): chunked delta rule (GLM_PF_KC=0: token-serial kernel)
        kda_chunk_rec(q, k, v, g, beta, Y.S, o, T, nh, beta + (size_t)T * nh + 64 - ((size_t)T * nh) % 4, st);
    else
        kda_rec(q, k, v, g, beta, r0 ? Y.snap + (r0 - 1) * nst : Y.S, o, T, nh, st, verify ? Y.snap + r0 * nst : nullptr,
                verify && !r0 && E->kda_ptr ? d_acc + 1 : nullptr);
    stamp(6);
    bf16* ob = bigb;
    if (fz && E->fuse_kda > 1) kda_out_f(o, proj, ldp, Y.kg_b, Y.ko_norm, ob, T, nh, st);
    else kda_out(o, gate, Y.ko_norm, ob, T, nh, st);
    stamp(7);
    lin(Y.ko, nullptr, ob, part, T);
    l2pf_join();
    stamp(8);
}

void Worker::mla(int l, int T, int n0) {
    Layer& Y = L[l];
    const int nh = p.nh();
    const int ldq = nh * DN + IH * ID;
    float* proj = big;
    float* wi = proj + (size_t)T * MLA_IN;
    float* qq = wi + (size_t)T * IH;
    float* qlat = qq + (size_t)T * ldq;
    float* ctx = qlat + (size_t)T * nh * KVLR;
    float* ov = ctx + (size_t)T * nh * KVLR;
    float* sc = ov + (size_t)T * nh * DV;
    bf16* qr = bigb;
    lin(Y.mw_in, xf, xb, proj, T);
    mla_prep(proj, Y.q_a_ln, Y.kv_a_ln, Y.ik_w, Y.ik_b, qr, Y.kc, Y.ki, Y.gsc, wi, n0, T, st, nullptr, E->kring);
    lin(Y.mw_q, nullptr, qr, qq, T);
    const int p0 = n0 / POOL, p1 = (n0 + T) / POOL;
    pool_keys(Y.ki, Y.gsc, Y.ape, Y.pk, p0, p1, st, E->kring);
    // indexer + top-k of the query rows split over the cards (pool lists exchanged peer to peer, then expanded on every
    // card): the scores are O(T x pools) and were replicated on every card
    const bool split = E->idx_split && p.ngpu > 1 && p1 > TOPB && T >= 64 * p.ngpu && psel;
    const int q0 = split ? T * p.rank / p.ngpu : 0, q1 = split ? T * (p.rank + 1) / p.ngpu : T;
    int* ps = split ? psel + (size_t)sel_par * mxa * TOPB : nullptr;
    int* pn = split ? pnsel + (size_t)sel_par * mxa : nullptr;
    if (p1 > TOPB) {
        const size_t room = big_n - (size_t)(sc - big);
        const int tb = (int)std::max<size_t>(1, std::min<size_t>(T, room / (size_t)p1));
        for (int t0 = q0; t0 < q1; t0 += tb) {
            const int nt = std::min(tb, q1 - t0);
            if (env_int("GLM_IDX_TC", 1))
                idx_score_tc(qq + (size_t)t0 * ldq + nh * DN, ldq, wi + (size_t)t0 * IH, Y.pk, sc, p1, n0 + t0, nt, st);
            else
                idx_score(qq + (size_t)t0 * ldq + nh * DN, ldq, wi + (size_t)t0 * IH, Y.pk, sc, p1, n0 + t0, nt, st);
            if (split) select_tokens(sc, p1, n0 + t0, d_sel + (size_t)t0 * MAXSEL, pn + t0, nt, st, nullptr, ps + (size_t)t0 * TOPB);
            else select_tokens(sc, p1, n0 + t0, d_sel + (size_t)t0 * MAXSEL, d_nsel + t0, nt, st);
        }
        if (split) {
            for (auto& wp : E->W) {
                Worker& o = *wp;
                if (o.p.rank == p.rank) continue;
                const size_t off = (size_t)sel_par;
                CUDA_CHECK(cudaMemcpyPeerAsync(o.psel + off * o.mxa * TOPB + (size_t)q0 * TOPB, o.dev, ps + (size_t)q0 * TOPB, dev,
                                               (size_t)(q1 - q0) * TOPB * 4, st));
                CUDA_CHECK(cudaMemcpyPeerAsync(o.pnsel + off * o.mxa + q0, o.dev, pn + q0, dev, (size_t)(q1 - q0) * 4, st));
            }
            CUDA_CHECK(cudaEventRecord(ev_sel, st));
            E->host_barrier();  // every card has recorded its event
            for (auto& wp : E->W)
                if (wp->p.rank != p.rank) CUDA_CHECK(cudaStreamWaitEvent(st, wp->ev_sel, 0));
            E->host_barrier();  // nobody re-records its event before the others waited on it
            sel_expand(ps, pn, n0, d_sel, d_nsel, T, st);
            sel_par ^= 1;
        }
    } else {
        CUDA_CHECK(cudaMemsetAsync(d_nsel, 0xff, (size_t)T * 4, st));
    }
    // q_lat[t][h] = q[t][h] . WUK[h]
    const float one = 1.f, zero = 0.f;
    const bool ubf = E->pf_ubf && Y.wuk_b && Y.wuv_b && T > MR;  // round 10: bf16 tensor cores (GLM_PF_UBF=0: fp32 SGEMM)
    if (ubf) {
        bf16* qqb = bigb;  // qr is consumed
        f2b_rows(qq, ldq, qqb, nh * DN, T, st);
        CUBLAS_CHECK(cublasGemmStridedBatchedEx(cb, CUBLAS_OP_N, CUBLAS_OP_N, KVLR, T, DN, &one, Y.wuk_b, CUDA_R_16BF, KVLR,
                                                (long long)DN * KVLR, qqb, CUDA_R_16BF, nh * DN, DN, &zero, qlat, CUDA_R_32F,
                                                nh * KVLR, KVLR, nh, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
    } else
        CUBLAS_CHECK(cublasSgemmStridedBatched(cb, CUBLAS_OP_N, CUBLAS_OP_N, KVLR, T, DN, &one, wuk32(Y, st), KVLR,
                                               (long long)DN * KVLR, qq, ldq, DN, &zero, qlat, nh * KVLR, KVLR, nh));
    if (T >= 64 && env_int("GLM_MLA_TC", 1)) {  // tensor cores: per block of queries, GEMMs over the union of their keys
        const float one_ = 1.f, zero_ = 0.f;
        int fb0 = 0, fb1 = 0;  // pending run of fallback query blocks [fb0, fb1)
        auto flush_fb = [&]() {
            if (fb1 > fb0) mla_attn_tc(qlat, Y.kc, d_sel, d_nsel, ctx, nh, n0, fb0, fb1 - fb0, 1.f / 16.f, st);
            fb0 = fb1 = 0;
        };
        // r12 (prefill review F1 fix A): the unions of every query block of this call in one pass and one D2H + sync
        // (was one per 64-query block: 512 syncs per layer at 32K); GLM_PF_UBATCH=0: per block as before
        const bool ubatch = E->pf_ubatch;
        const int nw_all = (n0 + T + 31) / 32 + 1;
        if (ubatch) {
            CHECK((T + UQB - 1) / UQB <= ublk_n && nw_all <= uw, "MLA union batch: %d blocks / %d words", (T + UQB - 1) / UQB, nw_all);
            mla_union_all(d_sel, d_nsel, T, UQB, n0, nw_all, uw, ubits, upre, d_u, st);
            CUDA_CHECK(cudaMemcpyAsync(h_ublk, d_u, (size_t)((T + UQB - 1) / UQB) * 4, cudaMemcpyDeviceToHost, st));
            const auto t0 = std::chrono::steady_clock::now();
            CUDA_CHECK(cudaStreamSynchronize(st));
            hw_mla_us += glm_us_since(t0);
        }
        for (int t0 = 0; t0 < T; t0 += UQB) {
            const int nb = std::min(UQB, T - t0);
            int U = 0;
            const unsigned* bb = ubits;
            const int* pb = upre;
            if (ubatch) {
                U = h_ublk[t0 / UQB];
                bb = ubits + (size_t)(t0 / UQB) * uw;
                pb = upre + (size_t)(t0 / UQB) * uw;
            } else {
                mla_union(d_sel, d_nsel, t0, nb, n0, E->ctx, ubits, upre, ukeys, d_u, st);
                CUDA_CHECK(cudaMemcpyAsync(&U, d_u, 4, cudaMemcpyDeviceToHost, st));
                CUDA_CHECK(cudaStreamSynchronize(st));
            }
            const int Up = (U + 7) & ~7;
            if (Up > ucap) {  // a huge union (sparse queries at depth): per-query tensor-core kernel (GLM_MLA_SPTC=0: CUDA cores)
                // round 10: consecutive fallback blocks go out as one launch after the loop (a 64-CTA grid per block
                // left most SMs idle; GLM_PF_SPB=0: per block)
                if (E->pf_spb && env_int("GLM_MLA_SPTC", 1)) {
                    if (fb1 == t0) fb1 = t0 + nb;
                    else { flush_fb(); fb0 = t0; fb1 = t0 + nb; }
                    continue;
                }
                if (env_int("GLM_MLA_SPTC", 1))
                    mla_attn_tc(qlat, Y.kc, d_sel, d_nsel, ctx, nh, n0, t0, nb, 1.f / 16.f, st);
                else
                    mla_attn(qlat + (size_t)t0 * nh * KVLR, Y.kc, d_sel + (size_t)t0 * MAXSEL, d_nsel + t0,
                             ctx + (size_t)t0 * nh * KVLR, nh, n0 + t0, nb, 1.f / 16.f, st);
                continue;
            }
            if (ubatch) mla_ukeys(bb, pb, (n0 + t0 + nb + 31) / 32 + 1, ukeys, st);
            CUDA_CHECK(cudaMemsetAsync(umask, 0, (size_t)nb * Up, st));
            CUDA_CHECK(cudaMemsetAsync(ukb + (size_t)U * KVLR, 0, (size_t)(Up - U) * KVLR * 2 + 16, st));
            mla_union_rest(Y.kc, d_sel, d_nsel, bb, pb, ukeys, U, t0, nb, n0, ukb, umask, st);
            qscale(qlat + (size_t)t0 * nh * KVLR, uqb, (size_t)nb * nh * KVLR, 1.f / 16.f, st);
            gemm_bf16(ukb, uqb, uS, Up, nb * nh, KVLR);
            usoftmax(uS, umask, Up, nh, nb * nh, uP, st);
            CUBLAS_CHECK(cublasGemmEx(cb, CUBLAS_OP_N, CUBLAS_OP_N, KVLR, nb * nh, Up, &one_, ukb, CUDA_R_16BF, KVLR, uP,
                                      CUDA_R_16BF, Up, &zero_, ctx + (size_t)t0 * nh * KVLR, CUDA_R_32F, KVLR,
                                      CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
        }
        flush_fb();
    } else {
        mla_attn(qlat, Y.kc, d_sel, d_nsel, ctx, nh, n0, T, 1.f / 16.f, st);
    }
    // ov[t][h] = WUV[h] . ctx[t][h]
    bf16* ovb = bigb;
    if (ubf) {  // ctx -> bf16 into the (free) qlat rows, ov straight to bf16
        bf16* ctxb = reinterpret_cast<bf16*>(qlat);
        f2b(ctx, ctxb, (size_t)T * nh * KVLR, st);
        CUBLAS_CHECK(cublasGemmStridedBatchedEx(cb, CUBLAS_OP_T, CUBLAS_OP_N, DV, T, KVLR, &one, Y.wuv_b, CUDA_R_16BF, KVLR,
                                                (long long)DV * KVLR, ctxb, CUDA_R_16BF, nh * KVLR, KVLR, &zero, ovb,
                                                CUDA_R_16BF, nh * DV, DV, nh, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
    } else {
        CUBLAS_CHECK(cublasSgemmStridedBatched(cb, CUBLAS_OP_T, CUBLAS_OP_N, DV, T, KVLR, &one, wuv32(Y, st), KVLR,
                                               (long long)DV * KVLR, ctx, nh * KVLR, KVLR, &zero, ov, nh * DV, DV, nh));
        f2b(ov, ovb, (size_t)T * nh * DV, st);
    }
    lin(Y.mo, nullptr, ovb, part, T);
}

void Worker::mlp(int l, int T) {
    Layer& Y = L[l];
    if (!Y.dense) return moe(l, T);
    const int nd = p.d1 - p.d0;
    float* gub = big;
    lin(Y.dw_gu, xf, xb, gub, T);
    bf16* hb = bigb;
    glu(gub, hb, nd, T, st);
    lin(Y.dw_d, nullptr, hb, part, T);
}

// DMA (landing ring) + untile of every non-resident expert of layer l into cstage[e]; the untiles wait for the previous
// MoE phase (ev_moe_done: its kernels read cstage), ev_staged marks the end
void Worker::stage_cold(int l) {
    CUDA_CHECK(cudaStreamWaitEvent(cst, ev_moe_done, 0));
    int nc = 0;
    for (int e = 0; e < NE; e++) {
        if (!E->slot_of.empty() && E->slot_of[(size_t)l * NE + e] >= 0) continue;
        const char* src = E->hx_layer[l] + (size_t)e * E->hxl.stride + E->hxl.off[p.rank];
        const SliceL& hs = E->hxl.lay(E->hx_layer[l] + (size_t)e * E->hxl.stride, p.rank);
        const int j = nc++ % NLAND;
        CUDA_CHECK(cudaStreamWaitEvent(dmst, ev_lfree[j], 0));
        CUDA_CHECK(cudaMemcpyAsync(land[j], src, hs.bytes, cudaMemcpyHostToDevice, dmst));
        CUDA_CHECK(cudaEventRecord(ev_land[j], dmst));
        CUDA_CHECK(cudaStreamWaitEvent(cst, ev_land[j], 0));
        untile(land[j], cstage + (size_t)e * sl.bytes, sl, 128, cst, &hs);
        CUDA_CHECK(cudaEventRecord(ev_lfree[j], cst));
    }
    CUDA_CHECK(cudaEventRecord(ev_staged, cst));
    staged_l = l;
}

void Worker::sr_start(int l) {
    sr_l = -1;
    if (!sring || l == MTP_L || L[l].dense || E->slot_of.empty()) return;
    sr_l = l;
    sr_list.clear();
    for (int e = 0; e < NE; e++) {
        sr_pos[e] = -1;
        if (E->slot_of[(size_t)l * NE + e] < 0) { sr_pos[e] = (int)sr_list.size(); sr_list.push_back(e); }
    }
    sr_issued = 0;
    // the ring's slots may still be read by the previous layer's expert kernels on st
    CUDA_CHECK(cudaEventRecord(ev_srs, st));
    CUDA_CHECK(cudaStreamWaitEvent(cst, ev_srs, 0));
    sr_issue(std::min((int)sr_list.size(), sr_n));
}
void Worker::sr_issue(int upto) {
    for (; sr_issued < upto; sr_issued++) {
        const int e = sr_list[sr_issued], k = sr_issued % sr_n, j = sr_issued % NLAND;
        const char* src = E->hx_layer[sr_l] + (size_t)e * E->hxl.stride + E->hxl.off[p.rank];
        const SliceL& hs = E->hxl.lay(E->hx_layer[sr_l] + (size_t)e * E->hxl.stride, p.rank);
        CUDA_CHECK(cudaStreamWaitEvent(dmst, ev_lfree[j], 0));
        CUDA_CHECK(cudaMemcpyAsync(land[j], src, hs.bytes, cudaMemcpyHostToDevice, dmst));
        CUDA_CHECK(cudaEventRecord(ev_land[j], dmst));
        CUDA_CHECK(cudaStreamWaitEvent(cst, ev_land[j], 0));
        if (sr_issued >= sr_n) CUDA_CHECK(cudaStreamWaitEvent(cst, ev_sf[k], 0));  // the slot's previous expert is done
        untile(land[j], sring + (size_t)k * sl.bytes, sl, 128, cst, &hs);
        CUDA_CHECK(cudaEventRecord(ev_lfree[j], cst));
        CUDA_CHECK(cudaEventRecord(ev_sr[k], cst));
    }
}

void Worker::moe_body(int l, int T) {
    Layer& Y = L[l];
    const int ni = p.ne();
    // router + shared expert -> part, in parts of <= mxa rows (their scratch is sized for one part)
    {
        const Acts base = *this;
        for (int r0 = 0; r0 < T && !own_routed; r0 += mxa) {  // round 20: owner mode routed every row already
            const int nr = std::min(mxa, T - r0);
            view_from(*this, base, r0);
            float* lg = big;
            if (E->router_rb) router_rows(l, xf, nr, d_ids, d_w);
            else {
                sgemm(Y.router, xf, lg, NE, nr, H);
                route(lg, Y.rbias, d_ids, d_w, nr, st);
            }
        }
        // routing to the host first: the host builds the row lists while the card runs the shared expert
        static_cast<Acts&>(*this) = base;
        CUDA_CHECK(cudaMemcpyAsync(h_ids, d_ids, (size_t)T * TOPK * 4, cudaMemcpyDeviceToHost, st));
        CUDA_CHECK(cudaMemcpyAsync(h_w, d_w, (size_t)T * TOPK * 4, cudaMemcpyDeviceToHost, st));
        CUDA_CHECK(cudaEventRecord(ev_ids, st));
        for (int r0 = 0; r0 < T; r0 += mxa) {
            const int nr = std::min(mxa, T - r0);
            view_from(*this, base, r0);
            float* gub = big;
            lin(Y.sw_gu, xf, xb, gub, nr);
            bf16* hb = bigb;
            glu(gub, hb, p.ns(), nr, st);
            lin(Y.sw_d, nullptr, hb, part, nr);
        }
        static_cast<Acts&>(*this) = base;
    }
    {
        const auto t0 = std::chrono::steady_clock::now();
        CUDA_CHECK(cudaEventSynchronize(ev_ids));
        hw_moe_us += glm_us_since(t0);
    }
    // rows per expert
    std::vector<std::vector<int>> rows(NE);
    std::vector<std::vector<float>> ws(NE);
    // review r13 L2: GLM_SKIP_TAU is the one source of the lossy default; round 17: the prefill has its own threshold
    // (GLM_PF_TAU, default 0 = lossless) since a long prompt's state carries every skipped pick into the whole answer
    const float tau = E->pf_tau;
    for (int t = 0; t < T; t++)
        for (int k = 0; k < TOPK; k++) {
            const int e = h_ids[t * TOPK + k];
            // round 10: the skip rule follows the residency before the prefill set was lent (wider windows lend more
            // slots; their experts are DMA'd, not skipped: same quality at any window; GLM_PF_SKIPLENT=1 old rule)
            const bool cold = l != MTP_L && (E->slot_of.empty() || E->slot_of[(size_t)l * NE + e] < 0) &&
                              (E->pf_skiplent || E->slot_pre.empty() || E->slot_pre[(size_t)l * NE + e] < 0);
            if (tau > 0 && cold && h_w[t * TOPK + k] < tau * RSF) continue;
            rows[e].push_back(t);
            ws[e].push_back(h_w[t * TOPK + k]);
        }
    // round 20 ([glm-arch2-pflane], GLM_PF_LANE_ROWS): cold experts with few rows go to the CPU lane (rank 0 adds their
    // outputs to its partials); every card drops them from its GPU lists (same routing and residency on every card)
    if (E->pf_lane_rows > 0 && E->lane && l != MTP_L && !sring && !own && T >= 8 && T <= E->pf_lane_maxt && !E->slot_of.empty()) {
        lane_e.clear(); lane_rows.clear(); lane_w.clear();
        for (int e = 0; e < NE; e++)
            if (!rows[e].empty() && (int)rows[e].size() <= E->pf_lane_rows && E->slot_of[(size_t)l * NE + e] < 0) {
                lane_e.push_back(e);
                lane_rows.push_back(rows[e]);
                lane_w.push_back(ws[e]);
                rows[e].clear();
                ws[e].clear();
            }
        if (!lane_e.empty() && p.rank == 0) lane_start(l, T);
        else { lane_e.clear(); lane_rows.clear(); lane_w.clear(); }
    }
    auto xptr = [&](int e) {
        XPtr x;
        const int s = l == MTP_L ? e : E->slot_of.empty() ? -1 : E->slot_of[(size_t)l * NE + e];
        if (s >= 0) {
            const char* b = (l == MTP_L ? mtp_slab : slab) + (size_t)s * sl.bytes;
            x.gq = (const int*)(b + sl.gq); x.uq = (const int*)(b + sl.uq); x.dq = (const int*)(b + sl.dq);
            x.gs = (const __half*)(b + sl.gs); x.us = (const __half*)(b + sl.us); x.ds = (const __half*)(b + sl.ds);
            x.ld = ni; x.c0 = 0; x.r0 = 0;
        } else {
            const char* b = E->hx_layer[l] + (size_t)e * E->hxl.stride + E->hxl.off[p.rank];
            x.gq = (const int*)(b + sl.gq); x.uq = (const int*)(b + sl.uq); x.dq = (const int*)(b + sl.dq);
            x.gs = (const __half*)(b + sl.gs); x.us = (const __half*)(b + sl.us); x.ds = (const __half*)(b + sl.ds);
            x.ld = ni; x.c0 = 0; x.r0 = 0;
            x.host = true;
        }
        return x;
    };
    bool anycold = false;
    for (int e = 0; e < NE && !anycold; e++) anycold = !rows[e].empty() && xptr(e).host;
    if (T <= JR && !anycold) {
        int nj = 0;
        for (int e = 0; e < NE; e++) {
            if (rows[e].empty()) continue;
            XJob& J = h_jobs[nj++];
            J.x = xptr(e);
            J.nr = (int)rows[e].size();
            for (int i = 0; i < J.nr; i++) { J.r[i] = rows[e][i]; J.w[i] = ws[e][i]; }
        }
        CUDA_CHECK(cudaMemcpyAsync(d_jobs, h_jobs, nj * sizeof(XJob), cudaMemcpyHostToDevice, st));
        expert_gemv(d_jobs, nj, xf, gu, part, ni, T, st, nullptr, xhb, xyp, d_njt);
        CUDA_CHECK(cudaStreamSynchronize(st));  // h_jobs reuse
        return;
    }
    // GEMM path: per expert, deterministic order; dequantization (PCIe reads for cold experts) double-buffered on the
    // copy stream, overlapped with the previous expert's GEMMs; all row lists uploaded once
    bf16* Wb[2] = {bigb + bigb_n, bigb + bigb_n + (size_t)3 * ni * H};
    bf16* xg = bigb;                         // gathered rows [n][4096]
    bf16* hb = bigb + (size_t)T * H;         // glu [n][ni]
    float* gub = big;                        // [n][2 ni]
    float* y = gub + (size_t)T * 2 * ni;     // [n][4096]
    // W4A8 path with a part-sized scratch (round 10 windows): the row lists only, at the start of big
    int* drows = E->pf_w8 && m8x && mxa < mx ? (int*)big : (int*)(y + (size_t)T * H);  // [T * 8]
    float* dw = (float*)(drows + (size_t)T * TOPK);
    std::vector<int> off(NE + 1, 0);
    for (int e = 0; e < NE; e++) {
        off[e + 1] = off[e] + (int)rows[e].size();
        memcpy(h_rall + off[e], rows[e].data(), rows[e].size() * 4);
        memcpy(h_wall + off[e], ws[e].data(), ws[e].size() * 4);
    }
    if (E->pf_rlk) h2d_words2(drows, h_rall, (int*)dw, (const int*)h_wall, off[NE], st);  // r27: off the H2D engine
    else {
        CUDA_CHECK(cudaMemcpyAsync(drows, h_rall, (size_t)off[NE] * 4, cudaMemcpyHostToDevice, st));
        CUDA_CHECK(cudaMemcpyAsync(dw, h_wall, (size_t)off[NE] * 4, cudaMemcpyHostToDevice, st));
    }
    CUDA_CHECK(cudaEventRecord(ev_use[0], st));
    CUDA_CHECK(cudaEventRecord(ev_use[1], st));
    for (int j = 0; j < NLAND; j++) CUDA_CHECK(cudaEventRecord(ev_lfree[j], cst));
    int ncold = 0;
    int i = 0;
    const bool f16 = E->pf_f16;
    const bool w8 = E->pf_w8 && m8x;
    const bool w16 = !w8 && !E->pf_hyb && E->pf_w16 && m16h;
    const bool batched = w16 && E->pf_w16b;
    const bool tuned = w16 && E->pf_w16t && m16x;
    if (tuned) b2h_rows(xb, H, H, T, m16x, st);
    if (batched && !tuned && E->pf_ahead && cstage && l != MTP_L && T >= 2048) {
        // cold experts staged into cstage[e] (DMA ring + untile on the copy stream), gate/up batched: the resident
        // experts in one launch, the cold ones in groups of pf_grp as their staging lands (mode 2; mode 1 = the whole
        // layer staged a layer ahead, one launch). Down / scatter per expert in expert order: the same kernels and
        // order as the default (bit-identical)
        const int mode = E->pf_ahead, G = mode == 1 ? NE : E->pf_grp;
        if (mode == 1 && staged_l != l) stage_cold(l);
        std::vector<XPtr> xs(NE);
        std::vector<int> cold;
        int nj = 0, maxn = 0;
        for (int e = 0; e < NE; e++) {
            const int n = off[e + 1] - off[e];
            if (!n) continue;
            XPtr xp = xptr(e);
            if (xp.host) { cold.push_back(e); continue; }
            xs[e] = xp;
            h_gujobs[nj].w = xp; h_gujobs[nj].off = off[e]; h_gujobs[nj].n = n;
            nj++;
            maxn = std::max(maxn, n);
        }
        const int nres = nj, resmax = maxn;
        for (int e : cold) {
            XPtr xp = xptr(e);
            const char* d = cstage + (size_t)e * sl.bytes;
            xp.gq = (const int*)(d + sl.gq); xp.uq = (const int*)(d + sl.uq); xp.dq = (const int*)(d + sl.dq);
            xp.gs = (const __half*)(d + sl.gs); xp.us = (const __half*)(d + sl.us); xp.ds = (const __half*)(d + sl.ds);
            xp.host = false;
            xs[e] = xp;
            h_gujobs[nj].w = xp; h_gujobs[nj].off = off[e]; h_gujobs[nj].n = off[e + 1] - off[e];
            nj++;
        }
        CUDA_CHECK(cudaMemcpyAsync(d_gujobs, h_gujobs, nj * sizeof(GuJob), cudaMemcpyHostToDevice, st));
        if (mode != 1) {  // stage this layer's cold experts now, one event per group
            CUDA_CHECK(cudaStreamWaitEvent(cst, ev_moe_done, 0));
            for (size_t c = 0; c < cold.size(); c++) {
                const int e = cold[c];
                const char* src = E->hx_layer[l] + (size_t)e * E->hxl.stride + E->hxl.off[p.rank];
                const SliceL& hs = E->hxl.lay(E->hx_layer[l] + (size_t)e * E->hxl.stride, p.rank);
                const int j = (int)(c % NLAND);
                CUDA_CHECK(cudaStreamWaitEvent(dmst, ev_lfree[j], 0));
                CUDA_CHECK(cudaMemcpyAsync(land[j], src, hs.bytes, cudaMemcpyHostToDevice, dmst));
                CUDA_CHECK(cudaEventRecord(ev_land[j], dmst));
                CUDA_CHECK(cudaStreamWaitEvent(cst, ev_land[j], 0));
                untile(land[j], cstage + (size_t)e * sl.bytes, sl, 128, cst, &hs);
                CUDA_CHECK(cudaEventRecord(ev_lfree[j], cst));
                if ((int)(c % G) == G - 1 || c + 1 == cold.size()) CUDA_CHECK(cudaEventRecord(ev_grp[(c / G) % NGRP], cst));
            }
        }
        if (nres) moe16_gu_batch(xb, drows, d_gujobs, nres, resmax, ni, m16h, st);
        int dn_next = 0;  // next expert id whose down kernel is not issued yet
        auto issue_dn = [&](int upto) {  // down of experts [dn_next, upto) in expert order
            for (; dn_next < upto; dn_next++) {
                const int n = off[dn_next + 1] - off[dn_next];
                if (n) moe16_dn(m16h + (size_t)off[dn_next] * ni, drows + off[dn_next], dw + off[dn_next], n, xs[dn_next], ni, part, st);
            }
        };
        for (size_t g0 = 0; g0 < cold.size(); g0 += G) {
            const int ng = (int)std::min(cold.size() - g0, (size_t)G);
            CUDA_CHECK(cudaStreamWaitEvent(st, mode == 1 ? ev_staged : ev_grp[(g0 / G) % NGRP], 0));
            int gm = 0;
            for (int q = 0; q < ng; q++) gm = std::max(gm, off[cold[g0 + q] + 1] - off[cold[g0 + q]]);
            moe16_gu_batch(xb, drows, d_gujobs + nres + g0, ng, gm, ni, m16h, st);
            issue_dn(cold[g0 + ng - 1] + 1);
        }
        issue_dn(NE);
        CUDA_CHECK(cudaEventRecord(ev_moe_done, st));
        staged_l = -1;
        if (mode == 1) {  // next MoE layer of this window (the MTP layer has no cold experts), or layer 3 of the next
            const int nx = l + 1 < E->nl ? l + 1 : (pf_more ? 3 : -1);
            if (nx >= 3) stage_cold(nx);
        }
        return;  // h_gujobs is rewritten only after the next moe()'s stream synchronisation
    }
    if (batched) {  // the resident experts' gate/up in one launch (their h rows at the row-list offsets)
        int nj = 0, maxn = 0;
        for (int e = 0; e < NE; e++) {
            const int n = off[e + 1] - off[e];
            if (!n) continue;
            XPtr xp = xptr(e);
            if (xp.host) continue;
            h_gujobs[nj].w = xp; h_gujobs[nj].off = off[e]; h_gujobs[nj].n = n;
            nj++;
            maxn = std::max(maxn, n);
        }
        if (nj) {
            CUDA_CHECK(cudaMemcpyAsync(d_gujobs, h_gujobs, nj * sizeof(GuJob), cudaMemcpyHostToDevice, st));
            if (tuned) moe16t_gu_batch(m16x, drows, d_gujobs, nj, maxn, ni, m16h, st);
            else moe16_gu_batch(xb, drows, d_gujobs, nj, maxn, ni, m16h, st);
        }
    }
    if (w8) moe8_quant(xb, T, m8x, m8s, st);
    const bool gu2 = w8 && E->pf_gu2 && m8h2;
    if (gu2) {  // the side stream starts after the quantised rows and the row lists are on the card
        CUDA_CHECK(cudaEventRecord(ev_q8, st));
        CUDA_CHECK(cudaStreamWaitEvent(gst, ev_q8, 0));
    }
    if (w8 && !gu2 && sr_l != l && !sr_early) sr_start(l);
    const bool ring = w8 && !gu2 && sr_l == l;
    if (ring && m8hg && E->pf_gub > 1) {
        // groups of G experts (gate/up in one launch, downs in order); resident experts first (GLM_PF_HOTFIRST, their
        // compute runs while the ring's DMA of the cold ones proceeds), then the cold ones in ring order
        std::vector<int> ord;
        ord.reserve(NE);
        if (E->pf_hotfirst) {
            for (int e = 0; e < NE; e++) if (sr_pos[e] < 0) ord.push_back(e);
            for (int e = 0; e < NE; e++) if (sr_pos[e] >= 0) ord.push_back(e);
        } else for (int e = 0; e < NE; e++) ord.push_back(e);
        const int G = E->pf_gub;
        for (int g0 = 0; g0 < NE; g0 += G) {
            const int g1 = std::min(NE, g0 + G);
            int maxn = 0, lastq = -1;
            for (int a = g0; a < g1; a++) {
                const int x = ord[a];
                const int nx = off[x + 1] - off[x];
                const int q = sr_pos[x];
                XPtr xp = xptr(x);
                if (q >= 0) {
                    sr_issue(std::min((int)sr_list.size(), q + 1));
                    if (nx) CUDA_CHECK(cudaStreamWaitEvent(st, ev_sr[q % sr_n], 0));
                    const char* d = sring + (size_t)(q % sr_n) * sl.bytes;
                    xp.gq = (const int*)(d + sl.gq); xp.uq = (const int*)(d + sl.uq); xp.dq = (const int*)(d + sl.dq);
                    xp.gs = (const __half*)(d + sl.gs); xp.us = (const __half*)(d + sl.us); xp.ds = (const __half*)(d + sl.ds);
                    xp.host = false;
                    lastq = q;
                }
                h_gujobs[a].w = xp; h_gujobs[a].off = off[x]; h_gujobs[a].n = nx;
                maxn = std::max(maxn, nx);
            }
            CUDA_CHECK(cudaMemcpyAsync(d_gujobs + g0, h_gujobs + g0, (size_t)(g1 - g0) * sizeof(GuJob), cudaMemcpyHostToDevice, st));
            moe8n_gu_batch(m8x, m8s, drows, d_gujobs + g0, g1 - g0, maxn, ni, m8hg, m8hgs, st);  // round 23 (GLM_PF_MOEN=0: old)
            for (int a = g0; a < g1; a++) {
                const int x = ord[a];
                const int nx = off[x + 1] - off[x];
                if (nx) moe8n_dn(m8hg + (size_t)off[x] * ni, m8hgs + (size_t)off[x] * (ni / 64), drows + off[x], dw + off[x], nx,
                                h_gujobs[a].w, ni, part, st);
                const int q = sr_pos[x];
                if (q >= 0) CUDA_CHECK(cudaEventRecord(ev_sf[q % sr_n], st));
            }
            if (lastq >= 0) sr_issue(std::min((int)sr_list.size(), lastq + 1 + sr_n));
        }
        sr_l = -1;
        return;
    }
    for (int e = 0; e < NE; e++) {
        const int n = off[e + 1] - off[e];
        if (ring) {  // cold experts from the staging ring (DMA started with the layer's attention), hot from their slots
            const int q = sr_pos[e];
            if (q >= 0) {
                if (n) {
                    sr_issue(std::min((int)sr_list.size(), q + 1));
                    const int k = q % sr_n;
                    CUDA_CHECK(cudaStreamWaitEvent(st, ev_sr[k], 0));
                    XPtr xp = xptr(e);
                    const char* d = sring + (size_t)k * sl.bytes;
                    xp.gq = (const int*)(d + sl.gq); xp.uq = (const int*)(d + sl.uq); xp.dq = (const int*)(d + sl.dq);
                    xp.gs = (const __half*)(d + sl.gs); xp.us = (const __half*)(d + sl.us); xp.ds = (const __half*)(d + sl.ds);
                    xp.host = false;
                    moe8_expert(m8x, m8s, drows + off[e], dw + off[e], n, xp, ni, m8h, m8hs, part, st);
                }
                CUDA_CHECK(cudaEventRecord(ev_sf[q % sr_n], st));
                sr_issue(std::min((int)sr_list.size(), q + 1 + sr_n));
            } else if (n) {
                moe8_expert(m8x, m8s, drows + off[e], dw + off[e], n, xptr(e), ni, m8h, m8hs, part, st);
            }
            continue;
        }
        if (!n) continue;
        const int b = i++ & 1;
        bf16* Wgu = Wb[b];
        bf16* Wd = Wgu + (size_t)2 * ni * H;
        CUDA_CHECK(cudaStreamWaitEvent(cst, ev_use[b], 0));
        XPtr xp = xptr(e);
        if (xp.host) {  // cold: DMA on its own stream into a ring of landing blocks (runs ahead), untile, dequantize
            char* d = stage[b];
            const int j = ncold++ % NLAND;
            CUDA_CHECK(cudaStreamWaitEvent(dmst, ev_lfree[j], 0));
            const SliceL& hs = E->hxl.lay(E->hx_layer[l] + (size_t)e * E->hxl.stride, p.rank);
            CUDA_CHECK(cudaMemcpyAsync(land[j], xp.gq, hs.bytes, cudaMemcpyHostToDevice, dmst));
            CUDA_CHECK(cudaEventRecord(ev_land[j], dmst));
            CUDA_CHECK(cudaStreamWaitEvent(cst, ev_land[j], 0));
            untile(land[j], d, sl, 128, cst, &hs);
            CUDA_CHECK(cudaEventRecord(ev_lfree[j], cst));
            xp.gq = (const int*)(d + sl.gq); xp.uq = (const int*)(d + sl.uq); xp.dq = (const int*)(d + sl.dq);
            xp.gs = (const __half*)(d + sl.gs); xp.us = (const __half*)(d + sl.us); xp.ds = (const __half*)(d + sl.ds);
        }
        if (w16) {  // fused fp16 tensor-core experts (gather / glu / scatter, fp16 accumulation, no dequantisation pass)
            CUDA_CHECK(cudaEventRecord(ev_deq[b], cst));
            CUDA_CHECK(cudaStreamWaitEvent(st, ev_deq[b], 0));
            if (tuned) moe16t_expert(m16x, drows + off[e], dw + off[e], n, xp, ni, m16h + (size_t)off[e] * ni, part,
                                     !(batched && !xp.host), st);
            else if (batched && !xp.host) moe16_dn(m16h + (size_t)off[e] * ni, drows + off[e], dw + off[e], n, xp, ni, part, st);
            else if (batched) {
                moe16_expert(xb, drows + off[e], dw + off[e], n, xp, ni, m16h + (size_t)off[e] * ni, part, st);
            } else moe16_expert(xb, drows + off[e], dw + off[e], n, xp, ni, m16h, part, st);
            CUDA_CHECK(cudaEventRecord(ev_use[b], st));
            continue;
        }
        if (gu2) {  // gate/up on gst (h buffer b), down on st in expert order; ev_use[b]: down done (h, stage free)
            int8_t* hq = b ? m8h2 : m8h;
            float* hs = b ? m8hs2 : m8hs;
            CUDA_CHECK(cudaEventRecord(ev_deq[b], cst));
            CUDA_CHECK(cudaStreamWaitEvent(gst, ev_deq[b], 0));
            CUDA_CHECK(cudaStreamWaitEvent(gst, ev_use[b], 0));
            moe8_gu(m8x, m8s, drows + off[e], n, xp, ni, hq, hs, gst);
            CUDA_CHECK(cudaEventRecord(ev_gu[b], gst));
            CUDA_CHECK(cudaStreamWaitEvent(st, ev_gu[b], 0));
            moe8_dn(hq, hs, drows + off[e], dw + off[e], n, xp, ni, part, st);
            CUDA_CHECK(cudaEventRecord(ev_use[b], st));
            continue;
        }
        if (w8) {  // W4A8 tensor-core experts: gather / glu / scatter fused, no dequantisation
            CUDA_CHECK(cudaEventRecord(ev_deq[b], cst));
            CUDA_CHECK(cudaStreamWaitEvent(st, ev_deq[b], 0));
            moe8_expert(m8x, m8s, drows + off[e], dw + off[e], n, xp, ni, m8h, m8hs, part, st);
            CUDA_CHECK(cudaEventRecord(ev_use[b], st));
            continue;
        }
        if (f16) {  // opt-in: fp16 inputs and fp16 accumulation (2x the tensor-core rate of bf16 with fp32 accumulation)
            __half *Wg = (__half*)Wgu, *Wdd = (__half*)Wd, *xg16 = (__half*)xg, *hb16 = (__half*)hb;
            __half *gu16 = (__half*)gub, *y16 = (__half*)y;
            deq_expert16(xp, ni, Wg, Wdd, cst);
            CUDA_CHECK(cudaEventRecord(ev_deq[b], cst));
            CUDA_CHECK(cudaStreamWaitEvent(st, ev_deq[b], 0));
            gather_rows16(xb, drows + off[e], xg16, n, st);
            gemm_f16(Wg, xg16, gu16, 2 * ni, n, H);
            glu16(gu16, hb16, ni, n, st);
            if (E->pf_hyb) moe16t_dn(hb16, drows + off[e], dw + off[e], n, xp, ni, part, st);  // fused down + scatter
            else {
                gemm_f16(Wdd, hb16, y16, H, n, ni);
                scatter_add16(part, y16, drows + off[e], dw + off[e], n, st);
            }
            CUDA_CHECK(cudaEventRecord(ev_use[b], st));
            continue;
        }
        deq_expert(xp, ni, Wgu, Wd, cst);
        CUDA_CHECK(cudaEventRecord(ev_deq[b], cst));
        CUDA_CHECK(cudaStreamWaitEvent(st, ev_deq[b], 0));
        gather_rows(xb, drows + off[e], xg, n, st);
        gemm_bf16(Wgu, xg, gub, 2 * ni, n, H);
        glu(gub, hb, ni, n, st);
        gemm_bf16(Wd, hb, y, H, n, ni);
        scatter_add(part, y, drows + off[e], dw + off[e], n, st);
        CUDA_CHECK(cudaEventRecord(ev_use[b], st));
    }
    if (ring) sr_l = -1;
}

// round 20 (review r20 M1, GLM_ROUTER_RB): the prefill router in fixed blocks of RB rows (a short tail is padded into
// rxp), so the SGEMM shape and with it a row's logits never depend on how many rows share the call
void Worker::router_rows(int l, const float* x, int n, int* ids, float* w) {
    const int RB = E->router_rb;
    if (!rlg) {
        CUDA_CHECK(cudaMalloc(&rlg, (size_t)RB * NE * 4));
        CUDA_CHECK(cudaMalloc(&rxp, (size_t)RB * H * 4));
    }
    for (int r0 = 0; r0 < n; r0 += RB) {
        const int nr = std::min(RB, n - r0);
        const float* xs = x + (size_t)r0 * H;
        if (nr < RB) {
            CUDA_CHECK(cudaMemcpyAsync(rxp, xs, (size_t)nr * H * 4, cudaMemcpyDeviceToDevice, st));
            xs = rxp;
        }
        sgemm(L[l].router, xs, rlg, NE, RB, H);
        route(rlg, L[l].rbias, ids + (size_t)r0 * TOPK, w + (size_t)r0 * TOPK, nr, st);
    }
}
void Worker::moe(int l, int T) {
    moe_body(l, T);
    if (!lane_e.empty()) lane_finish(T);
}
// rank 0: the lane experts' rows (xf, fp32) to the host next to the GPU experts; the lane runs after the GPU work of
// the layer is queued (lane_finish), its per-row sums go back into sum and are added to part before the all-reduce
// review r20b M2: only the lane's rows cross PCIe (gathered on the card, scattered back)
__global__ void k_lane_gather(const float* x, const int* idx, int n, float* out) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < (size_t)n * (H / 4); i += (size_t)gridDim.x * blockDim.x)
        reinterpret_cast<float4*>(out)[i] = reinterpret_cast<const float4*>(x + (size_t)idx[i / (H / 4)] * H)[i % (H / 4)];
}
__global__ void k_lane_scatter_add(float* part, const float* y, const int* idx, int n) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < (size_t)n * H; i += (size_t)gridDim.x * blockDim.x)
        part[(size_t)idx[i / H] * H + i % H] += y[i];
}
// review r21 M1: every buffer of the prefill lane at load (rank 0), sized for the largest job (pf_lane_maxt rows,
// NE * pf_lane_rows pairs); a failure returns false and the caller turns the lane off (GPU path), never an abort
bool Worker::lane_alloc() {
    CUDA_CHECK(cudaSetDevice(dev));
    const size_t mt = (size_t)E->pf_lane_maxt, np = (size_t)NE * E->pf_lane_rows;
    bool ok = cudaMallocHost(&lane_xh, mt * H * 4) == cudaSuccess && cudaMallocHost(&lane_yh, mt * H * 4) == cudaSuccess &&
              cudaMallocHost(&lane_ih, mt * 4) == cudaSuccess && cudaMalloc(&lane_id, mt * 4) == cudaSuccess &&
              cudaStreamCreateWithFlags(&lst, cudaStreamNonBlocking) == cudaSuccess &&
              cudaEventCreateWithFlags(&ev_lx, cudaEventDisableTiming) == cudaSuccess &&
              cudaEventCreateWithFlags(&ev_ly, cudaEventDisableTiming) == cudaSuccess;
    if (ok) {
        try { lane_yp.resize(np * H); } catch (const std::bad_alloc&) { ok = false; }
    }
    ok = ok && E->lane->pf_reserve((int)mt, (int)np);
    if (ok) ok = cudaEventRecord(ev_ly, st) == cudaSuccess;
    if (!ok) {
        (void)cudaGetLastError();
        if (lane_xh) cudaFreeHost(lane_xh);
        if (lane_yh) cudaFreeHost(lane_yh);
        if (lane_ih) cudaFreeHost(lane_ih);
        if (lane_id) cudaFree(lane_id);
        lane_xh = lane_yh = nullptr; lane_ih = nullptr; lane_id = nullptr;
        std::vector<float>().swap(lane_yp);
    }
    return ok;
}
void Worker::lane_start(int l, int T) {
    CHECK(T <= E->pf_lane_maxt && lane_xh, "prefill lane rows");
    lane_l = l;
    // the rows the lane experts use, ascending; lane_map[row] = compact index
    lane_map.assign(T, -1);
    for (auto& v : lane_rows) for (int r : v) lane_map[r] = 0;
    lane_n = 0;
    { const auto t0 = std::chrono::steady_clock::now();
      CUDA_CHECK(cudaEventSynchronize(ev_ly));  // the previous layer's copy of lane_ih / lane_yh has finished
      hw_lane_us += glm_us_since(t0); }
    for (int r = 0; r < T; r++) if (lane_map[r] == 0) { lane_map[r] = lane_n; lane_ih[lane_n++] = r; }
    CUDA_CHECK(cudaMemcpyAsync(lane_id, lane_ih, (size_t)lane_n * 4, cudaMemcpyHostToDevice, st));
    k_lane_gather<<<256, 256, 0, st>>>(xf, lane_id, lane_n, sum);  // sum is free during the MoE
    CUDA_CHECK(cudaEventRecord(ev_lx, st));
    CUDA_CHECK(cudaStreamWaitEvent(lst, ev_lx, 0));
    CUDA_CHECK(cudaMemcpyAsync(lane_xh, sum, (size_t)lane_n * H * 4, cudaMemcpyDeviceToHost, lst));
    CUDA_CHECK(cudaEventRecord(ev_lx, lst));
}
void Worker::lane_finish(int T) {
    (void)T;
    const int ne = (int)lane_e.size();
    std::vector<const char*> ex(ne);
    std::vector<int> off(ne + 1, 0), rr;
    std::vector<float> ww;
    for (int i = 0; i < ne; i++) {
        ex[i] = E->hx_layer[lane_l] + (size_t)lane_e[i] * E->hxl.stride;
        off[i + 1] = off[i] + (int)lane_rows[i].size();
        for (int r : lane_rows[i]) rr.push_back(lane_map[r]);
        ww.insert(ww.end(), lane_w[i].begin(), lane_w[i].end());
    }
    CHECK(lane_yp.size() >= (size_t)off[ne] * H, "prefill lane pairs");  // sized at load (lane_alloc)
    { const auto t0 = std::chrono::steady_clock::now();
      CUDA_CHECK(cudaEventSynchronize(ev_lx));
      hw_lane_us += glm_us_since(t0); }
    const auto t0 = std::chrono::steady_clock::now();
    // (per row, the lane sums its pairs in expert order: fixed summation order)
    E->lane->run_pf(lane_xh, lane_n, ne, ex.data(), off.data(), rr.data(), ww.data(), lane_yp.data(), lane_yh);
    const auto t1 = std::chrono::steady_clock::now();
    E->st_pflane_us += std::chrono::duration<double, std::micro>(t1 - t0).count();
    E->st_pflane_e += ne;
    E->st_pflane_rows += off[ne];
    // the gathered x rows in sum were read by the D2H (ev_lx): sum takes the lane's row sums now
    CUDA_CHECK(cudaMemcpyAsync(sum, lane_yh, (size_t)lane_n * H * 4, cudaMemcpyHostToDevice, st));
    k_lane_scatter_add<<<512, 256, 0, st>>>(part, sum, lane_id, lane_n);
    CUDA_CHECK(cudaEventRecord(ev_ly, st));  // review r20b L5: no full sync; the next lane_start waits for this copy
    lane_e.clear(); lane_rows.clear(); lane_w.clear();
}

// ---------------------------------------------------------------- round 24: copy-engine prefill all-reduce
// ([glm-r24-arce]; idea of the 27B QE_AR_CE path). The SM form (allreduce_rs_rows_kernel, 4 or 64 blocks of 512 threads
// storing bf16 rows into the peers over PCIe) holds whole SMs for ~3-5 ms per call: no GEMM CTA (222 registers x 256)
// fits next to one of its blocks, so the overlapped GEMMs / merges ran 15-30 % slower (r23 trace). Here the rows move
// by cudaMemcpyPeerAsync on a copy thread per card; only ce_pack / ce_sum / the fp32 write-out run on the SMs. Same
// arithmetic as the SM form (bf16-rounded partials, owner sums in the fixed GPU order, bf16 sums, 0 + sum): bit-identical.
void GCeHub::put(unsigned seq, int kind, int src, int dst, cudaEvent_t e) {
    {
        std::lock_guard<std::mutex> lk(mu);
        ev[std::make_tuple(seq, kind, src, dst)] = e;
    }
    cv.notify_all();
}
// review r24-L3: nullptr after 120 s (a real hang: the caller reports it as an engine stall, exit 75, restartable)
cudaEvent_t GCeHub::take(unsigned seq, int kind, int src, int dst) {
    std::unique_lock<std::mutex> lk(mu);
    const auto key = std::make_tuple(seq, kind, src, dst);
    if (!cv.wait_for(lk, std::chrono::seconds(120), [&] { return ev.count(key) != 0; })) return nullptr;
    cudaEvent_t e = ev[key];
    ev.erase(key);
    return e;
}
// review r24-L3: what the stall report needs: the waiting card's call, the sender's last published call to it and the
// sender's copy-queue depth (a sender far behind vs one whose copy thread is stuck in the driver)
[[noreturn]] void Worker::ce_stall(unsigned seq, int kind, int src) {
    long long last = -1;
    size_t hub = 0;
    {
        std::lock_guard<std::mutex> lk(E->ce_hub.mu);
        hub = E->ce_hub.ev.size();
        for (auto& kv : E->ce_hub.ev)
            if (std::get<2>(kv.first) == src && std::get<3>(kv.first) == p.rank)
                last = std::max(last, (long long)std::get<0>(kv.first));
    }
    Worker* swp = this;
    for (auto& w : E->W)
        if (w->p.rank == src) swp = w.get();
    Worker& sw = *swp;
    size_t q = 0;
    {
        std::lock_guard<std::mutex> lk(sw.ce_mu);
        q = sw.ce_q.size();
    }
    char what[256];
    snprintf(what, sizeof what,
             "GLM copy-engine all-reduce: card %d waited 120 s for card %d (call %u kind %d); card %d's calls so far %u, "
             "its copy queue %zu, pending events %zu, its last unclaimed event for card %d: call %lld",
             dev, sw.dev, seq, kind, sw.dev, sw.ce_seq /* (unsynchronised read: report only) */, q, hub, dev, last);
    qe_wd_host_timeout(what);
}
void Worker::ce_push(const GCeCmd& c) {
    if (!ce_st[0]) {  // first use: copy streams + the copy thread
        for (auto& s : ce_st) CUDA_CHECK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
        ce_th = std::thread([this] { ce_loop(); });
        ce_th.detach();  // the engine is never destroyed (process exit ends the thread)
    }
    {
        std::lock_guard<std::mutex> lk(ce_mu);
        ce_q.push_back(c);
    }
    ce_cv.notify_one();
}
// the only host thread that issues peer copies: a worker never blocks in cudaMemcpyPeerAsync (full copy queue) while
// its peers wait for its kernels (the 27B 10-18 hang)
void Worker::ce_loop() {
    CUDA_CHECK(cudaSetDevice(dev));
    {   // review r24 M3: created inside run_all, the thread inherited the worker's pin (card 0: the caller CPU): all CPUs
        cpu_set_t cs;
        CPU_ZERO(&cs);
        const long n = sysconf(_SC_NPROCESSORS_ONLN);
        for (long i = 0; i < n && i < CPU_SETSIZE; i++) CPU_SET(i, &cs);
        pthread_setaffinity_np(pthread_self(), sizeof cs, &cs);
    }
    for (;;) {
        GCeCmd c;
        {
            std::unique_lock<std::mutex> lk(ce_mu);
            ce_cv.wait(lk, [&] { return !ce_q.empty(); });
            c = ce_q.front();
            ce_q.pop_front();
        }
        if (c.stop) return;
        for (int j = 0; j < 3; j++) {
            if (c.peer[j] < 0) continue;
            CUDA_CHECK(cudaStreamWaitEvent(ce_st[j], c.ev, 0));
            if (c.cp[j].bytes)
                CUDA_CHECK(cudaMemcpyPeerAsync(c.cp[j].dst, c.cp[j].dev, c.cp[j].src, dev, c.cp[j].bytes, ce_st[j]));
            cudaEvent_t done;
            CUDA_CHECK(cudaEventCreateWithFlags(&done, cudaEventDisableTiming));
            CUDA_CHECK(cudaEventRecord(done, ce_st[j]));
            E->ce_hub.put(c.seq, c.kind, p.rank, c.peer[j], done);  // the receiver destroys it (later)
            if (c.kind == 1) {  // review r24 M1: this card's own copies done (AG after RS on the same streams)
                if (j > 0) CUDA_CHECK(cudaStreamWaitEvent(ce_st[0], done, 0));
            }
        }
        if (c.kind == 1) {
            cudaEvent_t self;
            CUDA_CHECK(cudaEventCreateWithFlags(&self, cudaEventDisableTiming));
            CUDA_CHECK(cudaEventRecord(self, ce_st[0]));
            E->ce_hub.put(c.seq, 2, p.rank, p.rank, self);
        }
        ce_old.push_back(c.ev);  // destroyed later (not while a wait on it may be pending)
        if (ce_old.size() > 256) { CUDA_CHECK(cudaEventDestroy(ce_old.front())); ce_old.pop_front(); }
    }
}
void Worker::ce_take(unsigned seq, int kind, cudaStream_t s_) {
    for (int g = 0; g < p.ngpu; g++) {
        if (g == p.rank) continue;
        const auto t0 = std::chrono::steady_clock::now();
        cudaEvent_t e = E->ce_hub.take(seq, kind, g, p.rank);
        if (!e) ce_stall(seq, kind, g);
        hw_ce_us += glm_us_since(t0);
        CUDA_CHECK(cudaStreamWaitEvent(s_, e, 0));
        ce_old_take.push_back(e);
        if (ce_old_take.size() > 512) { CUDA_CHECK(cudaEventDestroy(ce_old_take.front())); ce_old_take.pop_front(); }
    }
}
// sum rows = 0 + bf16 sums (the SM form adds the sums to the zeroed rows: the same bits, -0 included)
__global__ void __launch_bounds__(256) k_ce_out(const uint2* __restrict__ s2, float* __restrict__ out, int n4) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += gridDim.x * blockDim.x) {
        const uint2 w = __ldcv(s2 + i);
        const float2 a = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&w.x));
        const float2 b = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&w.y));
        float4 r = make_float4(0.f, 0.f, 0.f, 0.f);
        // review r24 L6: 0 + x written as __fadd_rn (fast-math is on for this target and may fold a plain 0 + x)
        r.x = __fadd_rn(r.x, a.x); r.y = __fadd_rn(r.y, a.y); r.z = __fadd_rn(r.z, b.x); r.w = __fadd_rn(r.w, b.y);
        reinterpret_cast<float4*>(out)[i] = r;
    }
}
void Worker::ar_ce(float* sum_rows, const float* part_rows, int nr, cudaStream_t s_) {
    const Comm& c = E->comm;
    const unsigned seq = ++ce_seq;
    ce_pack(part_rows, nr, H, c, p.rank, 0, s_, 0);
    auto cmd = [&](int kind) {
        GCeCmd x;
        CUDA_CHECK(cudaEventCreateWithFlags(&x.ev, cudaEventDisableTiming));
        CUDA_CHECK(cudaEventRecord(x.ev, s_));
        x.seq = seq;
        x.kind = kind;
        for (int j = 0, k = 0; j < p.ngpu; j++) {
            if (j == p.rank) continue;
            x.peer[k] = j;
            x.cp[k++] = kind == 0 ? ce_rs_copy(c, p.rank, j, nr, H, 0, c.pstage[p.rank], seq) : ce_ag_copy(c, p.rank, j, nr, H, 0, seq);
        }
        ce_push(x);
    };
    // the waits are host-taken events, never a spinning kernel: a device-side flag wait (one warp in ce_wait, flags copied
    // after the rows) hung every run here: the pending peer copies did not run while the card spun (Xid 43 by the
    // watchdog; with cudaMemcpyPeerAsync and cudaMemcpyAsync alike, r24 step 2)
    cmd(0);
    ce_take(seq, 0, s_);  // the peers' rows of this card's owner range landed in prs1
    ce_sum(part_rows, false, nr, H, c, p.rank, 0, seq, s_);
    cmd(1);
    ce_take(seq, 1, s_);  // every owner's sums landed in prs2
    {   // review r24 M1: and this card's own outgoing copies are done, so no later spinning kernel (an SM all-reduce on
        // cms, the decode) runs while they are pending
        const auto t0 = std::chrono::steady_clock::now();
        cudaEvent_t e = E->ce_hub.take(seq, 2, p.rank, p.rank);
        if (!e) ce_stall(seq, 2, p.rank);
        hw_ce_us += glm_us_since(t0);
        CUDA_CHECK(cudaStreamWaitEvent(s_, e, 0));
        ce_old_take.push_back(e);
        if (ce_old_take.size() > 512) { CUDA_CHECK(cudaEventDestroy(ce_old_take.front())); ce_old_take.pop_front(); }
    }
    const int n4 = nr * H / 4;
    k_ce_out<<<std::min(1024, (n4 + 255) / 256), 256, 0, s_>>>(c.prs2[p.rank], sum_rows, n4);
    CUDA_CHECK(cudaGetLastError());
}
void Worker::ar_on(cudaStream_t s_, int T, bool ov) {
    // GLM_AR_CE=2 (default): only the overlapped all-reduces (the exposed ones keep the 64-block SM form: nothing
    // runs next to them, and the event hub's host coupling costs more there)
    bool all_ce = E->ar_ce && (ov || E->ar_ce_mode != 2) && p.ngpu > 1 && p.ngpu <= 4;
    for (int r0 = 0; all_ce && r0 < T; r0 += E->ar_rows) all_ce = ce_ar_ok(E->comm, std::min(E->ar_rows, T - r0), H, 0);
    if (all_ce) {
        for (int r0 = 0; r0 < T; r0 += E->ar_rows)
            ar_ce(sum + (size_t)r0 * H, part + (size_t)r0 * H, std::min(E->ar_rows, T - r0), s_);
        return;
    }
    CUDA_CHECK(cudaMemsetAsync(sum, 0, (size_t)T * H * 4, s_));
    for (int r0 = 0; r0 < T; r0 += E->ar_rows) {
        const int nr = std::min(E->ar_rows, T - r0);
        allreduce_add(sum + (size_t)r0 * H, part + (size_t)r0 * H, nr, H, E->comm, p.rank, ctr, s_, nullptr, 0.f, nullptr,
                      -1, ov);
    }
}
// ---------------------------------------------------------------- round 20: row-owner prefill ([glm-arch1-owner])
// Each prefill all-reduce block of nr rows has row owners (rows [nr g / ngpu, nr (g + 1) / ngpu) on card g, comm.cu).
// GLM_PF_OWN: the owner alone merges the block output into its rows of R, runs the next block's mHC (and the router),
// and sends xb (+ the routing) to the peers; the peers never touch R, post, comb, xf of rows they don't own.
std::vector<std::pair<int, int>> Worker::own_ranges(int T) const {
    std::vector<std::pair<int, int>> v;
    for (int r0 = 0; r0 < T; r0 += E->ar_rows) {
        const int nr = std::min(E->ar_rows, T - r0);
        const int ob = (int)((long)nr * p.rank / p.ngpu), oe = (int)((long)nr * (p.rank + 1) / p.ngpu);
        if (oe > ob) v.push_back({r0 + ob, oe - ob});
    }
    return v;
}
bool Worker::own_ok(int T) const {  // the same answer on every card (all ranks' ranges are checked)
    for (int i = 0; i < pn_n; i++) {
        const int n = pn_b[i + 1] - pn_b[i];
        for (int r0 = 0; r0 < n; r0 += E->ar_rows) {
            const int nr = std::min(E->ar_rows, n - r0);
            if (!rs_own_ok(E->comm, nr, H)) return false;
            for (int g = 0; g < p.ngpu; g++)
                if ((long)nr * (g + 1) / p.ngpu - (long)nr * g / p.ngpu <= MR) return false;
        }
    }
    for (int r0 = 0; r0 < T; r0 += mxa)  // MoE router / shared-expert chunks: the GEMM path (bf16 rows) only
        if (std::min(mxa, T - r0) <= MAXR) return false;
    for (int i = 0; i < pn_n; i++) if (pn_b[i + 1] - pn_b[i] <= MAXR) return false;
    return T > 0;
}
void Worker::ar_own(cudaStream_t s_, int T, bool ov) {
    for (int r0 = 0; r0 < T; r0 += E->ar_rows) {
        const int nr = std::min(E->ar_rows, T - r0);
        allreduce_rs_own(sum + (size_t)r0 * H, part + (size_t)r0 * H, nr, H, E->comm, p.rank, ctr, s_, ov);
    }
}
struct GAg {
    int n = 0;
    const char* src[4] = {};
    char* dst[4][4] = {};
    int rb[4] = {};
    unsigned* flag[4] = {};
};
__device__ __forceinline__ void g_st_release_sys(unsigned* p, unsigned v) {
    asm volatile("st.release.sys.global.u32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}
__device__ __forceinline__ unsigned g_ld_acquire_sys(const unsigned* p) {
    unsigned v;
    asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    return v;
}
__device__ __forceinline__ void g_st_release_sys64(unsigned long long* p, unsigned long long v) {
    asm volatile("st.release.sys.global.u64 [%0], %1;" ::"l"(p), "l"(v) : "memory");
}
__device__ __forceinline__ unsigned long long g_ld_acquire_sys64(const unsigned long long* p) {
    unsigned long long v;
    asm volatile("ld.acquire.sys.global.u64 %0, [%1];" : "=l"(v) : "l"(p) : "memory");
    return v;
}
// owner rows [m rank / ngpu, m (rank + 1) / ngpu) of the arrays -> the same rows of every peer (P2P stores); the last
// block to finish fences, raises this card's flag on every peer and waits for every peer's flag (>= s: monotonic).
// Review r20-L1: a flag is one 64-bit word, the call number s and the call's tag (rows, count, arrays) written at once;
// a peer flag at exactly s must carry this card's tag, else the cards issue different allgather sequences (a host
// mismatch the count alone would hide): reported and trapped. A peer already past s is not checked (its word moved on).
// Review r20-L2: the wait is bounded by the collective watchdog (QE_COLL_TIMEOUT_MS) like the comm.cu waits.
__global__ void __launch_bounds__(512) k_ag_own(GAg a, int m, int rank, int ngpu, unsigned s, unsigned tag,
                                                unsigned* done) {
    const int ob = (int)((long)m * rank / ngpu), oe = (int)((long)m * (rank + 1) / ngpu);
    for (int j = 0; j < a.n; j++) {
        const int w = a.rb[j] / 16;
        for (int r = ob + blockIdx.x; r < oe; r += gridDim.x) {
            const uint4* src = reinterpret_cast<const uint4*>(a.src[j] + (size_t)r * a.rb[j]);
            for (int i = threadIdx.x; i < w; i += blockDim.x) {
                const uint4 v = src[i];
                for (int g = 0; g < ngpu; g++)
                    if (g != rank) reinterpret_cast<uint4*>(a.dst[j][g] + (size_t)r * a.rb[j])[i] = v;
            }
        }
    }
    __syncthreads();
    if (threadIdx.x) return;
    __threadfence_system();
    if (atomicAdd(done, 1u) != gridDim.x - 1) return;
    *done = 0;
    __threadfence_system();
    const unsigned long long word = (unsigned long long)tag << 32 | s;
    for (int g = 0; g < ngpu; g++)
        if (g != rank) g_st_release_sys64(reinterpret_cast<unsigned long long*>(a.flag[g]) + rank, word);
    for (int g = 0; g < ngpu; g++) {
        if (g == rank) continue;
        const unsigned long long* f = reinterpret_cast<const unsigned long long*>(a.flag[rank]) + g;
        unsigned long long v = g_ld_acquire_sys64(f);
        for (unsigned ws = 0; (int)((unsigned)v - s) < 0; v = g_ld_acquire_sys64(f)) {
            __nanosleep(64);
            QE_WD_POLL(ws, kWdAllgather, g, s, (unsigned)v);
        }
        if ((unsigned)v == s && (unsigned)(v >> 32) != tag) {
            printf("[glm] owner allgather: card %d call %u tag %08x, card %d sent tag %08x (host call sequences differ)\n",
                   rank, s, tag, g, (unsigned)(v >> 32));
            __trap();
        }
    }
}
void Worker::ag_own(cudaStream_t s_, int r0, int T, int what, bool ov) {
    static const int ovb = env_int("GLM_OWN_AGB_OV", 4), fb = env_int("GLM_OWN_AGB", 32);
    for (int b0 = 0; b0 < T; b0 += E->ar_rows) {
        const int nr = std::min(E->ar_rows, T - b0);
        const size_t row = (size_t)r0 + b0;
        GAg a;
        auto add = [&](int k, int rb) {
            char* base[4] = {};
            for (int g = 0; g < p.ngpu; g++) {
                const Engine::OwnPtr& q = E->own_ptr[g];
                base[g] = k == 0 ? q.xb : k == 1 ? q.ids : k == 2 ? q.w : q.sum;
            }
            a.src[a.n] = base[p.rank] + row * rb;
            for (int g = 0; g < p.ngpu; g++) a.dst[a.n][g] = base[g] + row * rb;
            a.rb[a.n++] = rb;
        };
        if (what & 1) add(0, H * 2);
        if (what & 2) { add(1, TOPK * 4); add(2, TOPK * 4); }
        if (what & 4) add(3, H * 4);
        for (int g = 0; g < p.ngpu; g++) a.flag[g] = E->own_flag[g];
        const int nb = std::max(1, std::min(nr / p.ngpu, ov ? ovb : fb));
        // review r20-L1: the call's tag, the same on every card for the same logical call
        const unsigned tag = (unsigned)row * 2654435761u ^ (unsigned)nr * 40503u ^ (unsigned)what * 0x9E3779B9u;
        k_ag_own<<<nb, 512, 0, s_>>>(a, nr, p.rank, p.ngpu, ++ag_seq, tag, agdone);
        CUDA_CHECK(cudaGetLastError());
    }
}
void Worker::own_fin(int l, int blk, int T, bool mrg) {
    const Acts cur = *this;
    struct Swap {  // GLM_OWN_SIDE: every launch below goes to ost / cbo
        Worker& w; bool on; cudaStream_t s0; cublasHandle_t c0;
        Swap(Worker& w_, bool on_) : w(w_), on(on_), s0(w_.st), c0(w_.cb) { if (on) { w.st = w.ost; w.cb = w.cbo; } }
        ~Swap() { if (on) { w.st = s0; w.cb = c0; } }
    } sw(*this, E->own_side);
    const bool rt = blk == 1 && !L[l].dense;
    for (auto& o : own_ranges(T)) {
        view_from(*this, cur, o.first);
        const bool hm = mrg ? pmerge(l, blk, o.second) : false;
        mhc_pre(l, blk, o.second, hm);
        if (rt) {
            if (E->router_rb) router_rows(l, xf, o.second, d_ids, d_w);
            else {  // router logits in this view's part rows (free: the all-reduce read them; the MoE overwrites them)
                sgemm(L[l].router, xf, part, NE, o.second, H);
                route(part, L[l].rbias, d_ids, d_w, o.second, st);
            }
        }
    }
    static_cast<Acts&>(*this) = cur;
}

// prefill layer with the rows in halves A / B: A's attention all-reduce runs next to B's attention, and B's MoE
// all-reduce next to the next layer's A attention (all all-reduces in one order on the comm stream: same on every card)
void Worker::layer_ov(int l, int T, int n0) {
    const Acts base = *this;
    const int ha = (T / 2 + 63) / 64 * 64, hb = T - ha;
    auto V = [&](int r0) { view_from(*this, base, r0); };
    auto rec_wait = [&](cudaStream_t from, cudaStream_t to, cudaEvent_t e) {
        CUDA_CHECK(cudaEventRecord(e, from));
        CUDA_CHECK(cudaStreamWaitEvent(to, e, 0));
    };
    bool mixA = ov_mixA, mixB = false;
    auto attn = [&](int r0, int nt, bool hm) {
        V(r0);
        mhc_pre(l, 0, nt, hm);
        if (L[l].mla) mla(l, nt, n0 + r0);
        else kda(l, nt, n0 + r0);
    };
    attn(0, ha, mixA);
    rec_wait(st, cms, ev_c[0]);
    V(0); ar_on(cms, ha, true);
    CUDA_CHECK(cudaEventRecord(ev_c[1], cms));
    if (ov_pend) {  // the previous layer's half B: its MoE all-reduce ran next to A's attention above
        CUDA_CHECK(cudaStreamWaitEvent(st, ev_c[3], 0));
        V(ha); mixB = pmerge(l, 0, hb);
        ov_pend = false;
    }
    attn(ha, hb, mixB);
    rec_wait(st, cms, ev_c[2]);
    V(ha); ar_on(cms, hb, true);
    CUDA_CHECK(cudaEventRecord(ev_c[3], cms));
    CUDA_CHECK(cudaStreamWaitEvent(st, ev_c[1], 0));
    V(0); const bool m1 = pmerge(l, 1, ha);
    CUDA_CHECK(cudaStreamWaitEvent(st, ev_c[3], 0));
    V(ha); pmerge(l, 1, hb);
    static_cast<Acts&>(*this) = base;
    mhc_pre(l, 1, T, m1);
    mlp(l, T);
    rec_wait(st, cms, ev_c[0]);
    V(0); ar_on(cms, ha, false);
    CUDA_CHECK(cudaEventRecord(ev_c[1], cms));
    V(ha); ar_on(cms, hb, true);
    CUDA_CHECK(cudaEventRecord(ev_c[3], cms));
    CUDA_CHECK(cudaStreamWaitEvent(st, ev_c[1], 0));
    V(0); ov_mixA = pmerge(l + 1, 0, ha);
    static_cast<Acts&>(*this) = base;
    ov_pend = true;
    ov_ha = ha;
    ov_T = T;
}
void Worker::ov_finish() {
    if (!ov_pend) return;
    const Acts base = *this;
    CUDA_CHECK(cudaStreamWaitEvent(st, ev_c[3], 0));
    view_from(*this, base, ov_ha);
    glm::merge(R, post, comb, sum, ov_T - ov_ha, st, ssp);
    static_cast<Acts&>(*this) = base;
    ov_pend = false;
    ov_mixA = false;
}

void Worker::pn_split(int T) {
    // small edge parts: the exposed all-reduces per layer are the first part's (MoE) and the last part's (attention);
    // the middle in even parts of <= mxa rows (GLM_PF_EDGE=0: even parts only)
    std::vector<int> sz;
    const int e = E->pf_edge;
    std::vector<int> head, tail;
    // measured: edges help at 8K (2,361 -> 2,404 tok/s) and cost at 32K (3,285 -> 3,201: more parts, more indexer
    // exchanges): only for windows below 3 parts
    if (e > 0 && T >= 4 * e && T < 3 * E->pf_part) { head = {e}; tail = {e}; }
    else if (E->pf_head > 0 && T >= 3 * E->pf_part) head = {E->pf_head};  // a small first part: its MoE AR is exposed
    int mid = T;
    for (int x : head) mid -= x;
    for (int x : tail) mid -= x;
    if (!tail.empty()) { tail.back() += mid % 64; mid -= mid % 64; }  // only the last part may end off the 64 grid
    const int nm = std::max(head.empty() ? 2 : 1, (mid + mxa - 1) / mxa);
    sz = head;
    int done = 0;
    for (int i = 0; i < nm; i++) {
        const int nx = i + 1 == nm ? mid : std::min(mid, (int)(((long)mid * (i + 1) / nm + 63) / 64 * 64));
        sz.push_back(nx - done);
        done = nx;
    }
    for (int x : tail) sz.push_back(x);
    CHECK((int)sz.size() <= PNMAX, "pn_split: %d parts", (int)sz.size());
    pn_n = (int)sz.size();
    pn_b[0] = 0;
    for (int i = 0; i < pn_n; i++) {
        pn_b[i + 1] = pn_b[i] + sz[i];
        CHECK(sz[i] > 0 && sz[i] <= mxa && (i + 1 == pn_n || pn_b[i + 1] % 64 == 0), "pn_split %d %d", T, mxa);
    }
    CHECK(pn_b[pn_n] == T, "pn_split sum");
}
// prefill window in parts (round 10, P-1 / P-3): part i's attention all-reduce runs next to part i + 1's attention;
// the MoE runs once over the whole window (one pass over the cold experts) and its all-reduce goes out per part, each
// next to the next layer's attention of the earlier parts. Exposed per layer: the last part's attention AR and the
// first part's MoE AR (those two run with all comm blocks). All all-reduces in one order on cms on every card.
void Worker::layer_pn(int l, int T, int n0) {
    // the layer's cold experts start crossing PCIe with the attention (GLM_PF_RING_EARLY=1) or with the MoE (0)
    if (sr_early) sr_start(l);
    const Acts base = *this;
    const int n = pn_n;
    const int* b = pn_b;
    auto V = [&](int i) { view_from(*this, base, b[i]); };
    auto rows = [&](int i) { return b[i + 1] - b[i]; };
    auto ar_part = [&](int i, cudaEvent_t done, bool ov) {
        CUDA_CHECK(cudaEventRecord(ev_ps[i], st));
        CUDA_CHECK(cudaStreamWaitEvent(cms, ev_ps[i], 0));
        V(i);
        if (own) ar_own(cms, rows(i), ov);
        else ar_on(cms, rows(i), ov);
        CUDA_CHECK(cudaEventRecord(done, cms));
    };
    // round 20 (owner mode): own rows finished on st, then sent to the peers on ags; ev = the rows complete here
    auto own_send = [&](int i, int what, bool ov, cudaEvent_t ev) {
        CUDA_CHECK(cudaEventRecord(ev, st));
        CUDA_CHECK(cudaStreamWaitEvent(ags, ev, 0));
        ag_own(ags, b[i], rows(i), what, ov);
        CUDA_CHECK(cudaEventRecord(ev, ags));
    };
    auto post_attn = [&](int i) {  // merge of part i's attention output + its MLP-block mHC
        V(i);
        if (own && E->own_side) {  // ost: after this part's attention all-reduce, next to st's next attention
            CUDA_CHECK(cudaStreamWaitEvent(ost, ev_pa[i], 0));
            own_fin(l, 1, rows(i), true);
            CUDA_CHECK(cudaEventRecord(ev_om[i], ost));
            CUDA_CHECK(cudaStreamWaitEvent(ags, ev_om[i], 0));
            ag_own(ags, b[i], rows(i), L[l].dense ? 1 : 3, true);
            CUDA_CHECK(cudaEventRecord(ev_om[i], ags));
            return;
        }
        CUDA_CHECK(cudaStreamWaitEvent(st, ev_pa[i], 0));
        if (own) {
            own_fin(l, 1, rows(i), true);
            own_send(i, L[l].dense ? 1 : 3, true, ev_om[i]);
            return;
        }
        const bool hm = pmerge(l, 1, rows(i));
        mhc_pre(l, 1, rows(i), hm);
    };
    // measured: 32K 3,387 -> 3,443; 8K / 16K -4 / -2 % (the recurrence slows 0.85 -> 1.23 s next to the GEMMs, and
    // the second scratch set lends more slots): windows >= 24K only
    const bool kov = !L[l].mla && E->pf_kov && !E->pf_kc && mxa < mx && T >= 24576 && mx >= 24576;
    unsigned own_sent = 0;
    auto pre_attn = [&](int i) {
        V(i);
        bool hm = false;
        if (pn_pend) {  // the previous layer's MoE output of these rows
            if (own) {
                // GLM_OWN_LA=1: part i + 1's owner work and send are queued before part i's attention (its send runs
                // next to that attention); the order on st is the same on every card
                auto fin_send = [&](int k) {
                    if (k >= n || own_sent >> k & 1) return;
                    V(k);
                    if (E->own_side) {  // ost: after this part's MoE all-reduce; st only waits for the rows
                        CUDA_CHECK(cudaStreamWaitEvent(ost, ev_pm[k], 0));
                        own_fin(l, 0, rows(k), true);
                        CUDA_CHECK(cudaEventRecord(ev_of[k], ost));
                        CUDA_CHECK(cudaStreamWaitEvent(ags, ev_of[k], 0));
                        ag_own(ags, b[k], rows(k), 1, E->own_agov);
                        CUDA_CHECK(cudaEventRecord(ev_oa[k], ags));
                    } else {
                        CUDA_CHECK(cudaStreamWaitEvent(st, ev_pm[k], 0));
                        own_fin(l, 0, rows(k), true);
                        own_send(k, 1, E->own_agov, ev_oa[k]);
                    }
                    own_sent |= 1u << k;
                };
                fin_send(i);
                if (E->own_la) fin_send(i + 1);
                V(i);
                CUDA_CHECK(cudaStreamWaitEvent(st, ev_oa[i], 0));
                return;
            }
            CUDA_CHECK(cudaStreamWaitEvent(st, ev_pm[i], 0));
            hm = pmerge(l, 0, rows(i));
        }
        mhc_pre(l, 0, rows(i), hm);  // (layer 0: every card computes every row from the embedding)
    };
    if (kov) {  // KDA: part i's recurrence (gst) next to part i + 1's projections / prep and part i - 1's output (st)
        auto fin = [&](int i) {
            V(i);
            CUDA_CHECK(cudaStreamWaitEvent(st, ev_kb[i & 1], 0));
            kda_b(l, rows(i), i & 1);
            ar_part(i, ev_pa[i], i + 1 < n);
            if (i > 0) post_attn(i - 1);
        };
        for (int i = 0; i < n; i++) {
            pre_attn(i);
            kda_a(l, rows(i), i & 1);
            CUDA_CHECK(cudaEventRecord(ev_ka[i & 1], st));
            CUDA_CHECK(cudaStreamWaitEvent(krst, ev_ka[i & 1], 0));
            kda_r(l, rows(i), i & 1, krst);
            CUDA_CHECK(cudaEventRecord(ev_kb[i & 1], krst));
            if (i > 0) fin(i - 1);
        }
        fin(n - 1);
    } else {
        for (int i = 0; i < n; i++) {
            pre_attn(i);
            if (L[l].mla) mla(l, rows(i), n0 + b[i]);
            else kda(l, rows(i), n0 + b[i]);
            ar_part(i, ev_pa[i], i + 1 < n);
            if (i > 0) post_attn(i - 1);
        }
    }
    post_attn(n - 1);
    pn_pend = false;
    if (L[l].dense) {
        for (int i = 0; i < n; i++) {
            V(i);
            if (own) CUDA_CHECK(cudaStreamWaitEvent(st, ev_om[i], 0));
            mlp(l, rows(i));
            ar_part(i, ev_pm[i], i > 0);
        }
    } else {
        static_cast<Acts&>(*this) = base;
        if (own) for (int i = 0; i < n; i++) CUDA_CHECK(cudaStreamWaitEvent(st, ev_om[i], 0));
        own_routed = own;
        mlp(l, T);
        own_routed = false;
        for (int i = 0; i < n; i++) ar_part(i, ev_pm[i], i > 0);
    }
    static_cast<Acts&>(*this) = base;
    pn_pend = true;
}
void Worker::pn_finish() {
    if (!pn_pend) return;
    const Acts base = *this;
    if (own) {  // round 20: owners merge + final norm their rows, then the hidden rows (sum) go to the peers; xb = bf16(sum)
        for (int i = 0; i < pn_n; i++) {
            const int n = pn_b[i + 1] - pn_b[i];
            view_from(*this, base, pn_b[i]);
            const Acts cur = *this;
            CUDA_CHECK(cudaStreamWaitEvent(st, ev_pm[i], 0));
            for (auto& o : own_ranges(n)) {
                view_from(*this, cur, o.first);
                glm::merge(R, post, comb, sum, o.second, st, ssp);
                glm::final_norm(R, this->final_norm, xb, sum, o.second, st);
            }
            CUDA_CHECK(cudaEventRecord(ev_om[i], st));
            CUDA_CHECK(cudaStreamWaitEvent(ags, ev_om[i], 0));
            ag_own(ags, pn_b[i], n, 4, false);
            CUDA_CHECK(cudaEventRecord(ev_om[i], ags));
        }
        static_cast<Acts&>(*this) = base;
        for (int i = 0; i < pn_n; i++) CUDA_CHECK(cudaStreamWaitEvent(st, ev_om[i], 0));
        f2b(sum, xb, (size_t)pn_b[pn_n] * H, st);
        pn_pend = false;
        return;
    }
    for (int i = 0; i < pn_n; i++) {
        view_from(*this, base, pn_b[i]);
        CUDA_CHECK(cudaStreamWaitEvent(st, ev_pm[i], 0));
        glm::merge(R, post, comb, sum, pn_b[i + 1] - pn_b[i], st, ssp);
    }
    static_cast<Acts&>(*this) = base;
    pn_pend = false;
}

void Worker::layer(int l, int T, int n0) {
    mhc_pre(l, 0, T, l > 0 && pf_lmix);  // pf_lmix: the previous layer's merge left the mixes
    if (L[l].mla) mla(l, T, n0);
    else kda(l, T, n0);
    allreduce(T);
    const bool m1 = pmerge(l, 1, T);
    mhc_pre(l, 1, T, m1);
    mlp(l, T);
    allreduce(T);
    pf_lmix = pmerge(l + 1, 0, T);
}

// ---------------------------------------------------------------- engine

void Engine::host_barrier() {
    const unsigned g = hb_gen.load(std::memory_order_acquire);
    if (hb_cnt.fetch_add(1, std::memory_order_acq_rel) + 1 == ngpu) {
        hb_cnt.store(0, std::memory_order_relaxed);
        hb_gen.fetch_add(1, std::memory_order_acq_rel);
    } else {
        while (hb_gen.load(std::memory_order_acquire) == g) _mm_pause();
    }
}

// one call per card in parallel. GLM_POOL (default 1): persistent spinning threads pinned to cores outside the CPU lane
// (GLM_POOL_CPUS, default "23,47"; the caller runs card 0) instead of new threads per call
void Engine::run_all(const std::function<void(Worker&)>& f) {
    if (W.size() > 1 && pool_on < 0) pool_on = env_int("GLM_POOL", 1);
    if (pool_on <= 0 || W.size() < 2 || std::this_thread::get_id() != pool_owner && pool_th.size()) {
        // review rel L4: a caller other than the pool owner (the prefill job thread) spawns its card threads; with the
        // pool up they take the pool's CPUs (card 0: the caller CPU), never a lane core or its SMT sibling
        const bool pinned = pool_th.size() > 0;
        std::vector<std::thread> th;
        for (size_t i = 0; i < W.size(); i++) {
            const int cpu = !pinned ? -1 : i == 0 ? caller_cpu : i - 1 < pool_cpus.size() ? pool_cpus[i - 1] : -1;
            auto& w = W[i];
            th.emplace_back([&f, &w, cpu] {
                if (cpu >= 0) {
                    cpu_set_t cs;
                    CPU_ZERO(&cs);
                    CPU_SET(cpu, &cs);
                    pthread_setaffinity_np(pthread_self(), sizeof cs, &cs);
                }
                CUDA_CHECK(cudaSetDevice(w->dev));
                f(*w);
            });
        }
        for (auto& t : th) t.join();
        return;
    }
    if (pool_th.empty()) {
        pool_owner = std::this_thread::get_id();
        // round 11 (review 2 F8): the owner runs card 0 and spins in cudaStreamSynchronize for the whole step; unpinned it
        // landed on an SMT sibling of a lane core. GLM_CALLER_CPU (default 24 = sibling of the coordinator's core 0), -1 off
        // r12 (review C3): pinned only inside run_all (the owner's mask is restored on return), so threads that the
        // application or the engine create later do not inherit {cc}; skipped when cc is not an online CPU
        if (const int cc = env_int("GLM_CALLER_CPU", hcpu.caller); cc >= 0 && cc < CPU_SETSIZE && cc < (int)sysconf(_SC_NPROCESSORS_ONLN)
                                                          && pthread_getaffinity_np(pthread_self(), sizeof caller_mask, &caller_mask) == 0)
            caller_cpu = cc;
        caller_step = env_int("GLM_CALLER_STEP", 0) != 0;
        std::vector<int> cpus = hcpu.pool;  // [glm-rel-host] (23, 47 on the 24-core box)
        if (const char* pc = getenv("GLM_POOL_CPUS")) {
            cpus.clear();
            for (const char* q = pc; *q;) { cpus.push_back(atoi(q)); while (*q && *q != ',') q++; if (*q) q++; }
        }
        pool_cpus = cpus;
        for (size_t i = 1; i < W.size(); i++)
            pool_th.emplace_back([this, i, cpus] {
                if (i - 1 < cpus.size()) {
                    cpu_set_t cs;
                    CPU_ZERO(&cs);
                    CPU_SET(cpus[i - 1], &cs);
                    pthread_setaffinity_np(pthread_self(), sizeof cs, &cs);
                }
                CUDA_CHECK(cudaSetDevice(W[i]->dev));
                unsigned seen = 0;
                const long spin_us = idle_spin_us("GLM_POOL_SPIN_US");
                for (;;) {
                    // [glm-idle] spin while calls come back to back, then block on pool_gen (idle server)
                    seen = pool_gate.wait(pool_gen, seen, spin_us);
                    if (pool_quit) return;
                    (*pool_fn)(*W[i]);
                    pool_done.fetch_add(1, std::memory_order_acq_rel);
                }
            });
    }
    const bool pin = caller_cpu >= 0 && !caller_held;  // round 18: a step-wide pin (GLM_CALLER_STEP) holds it already
    if (pin) {
        cpu_set_t cs;
        CPU_ZERO(&cs);
        CPU_SET(caller_cpu, &cs);
        pthread_setaffinity_np(pthread_self(), sizeof cs, &cs);
    }
    pool_fn = &f;
    pool_done.store(0, std::memory_order_relaxed);
    pool_gate.bump(pool_gen);
    CUDA_CHECK(cudaSetDevice(W[0]->dev));
    f(*W[0]);
    while (pool_done.load(std::memory_order_acquire) < (unsigned)(W.size() - 1)) _mm_pause();
    if (pin) pthread_setaffinity_np(pthread_self(), sizeof caller_mask, &caller_mask);
}
// round 18 (GLM_CALLER_STEP=1): the caller pinned once for a whole decode step (its verify, commit and MTP passes)
// instead of per run_all; the mask is restored on return, so threads created between steps do not inherit it
Engine::StepPin::StepPin(Engine* e_) : e(e_) {
    on = e->caller_step && e->caller_cpu >= 0 && !e->caller_held && !e->pool_th.empty()
         && std::this_thread::get_id() == e->pool_owner;
    if (!on) return;
    cpu_set_t cs;
    CPU_ZERO(&cs);
    CPU_SET(e->caller_cpu, &cs);
    pthread_setaffinity_np(pthread_self(), sizeof cs, &cs);
    e->caller_held = true;
}
Engine::StepPin::~StepPin() {
    if (!on) return;
    e->caller_held = false;
    pthread_setaffinity_np(pthread_self(), sizeof e->caller_mask, &e->caller_mask);
}

void Engine::load(const std::string& d, int ctx_tokens) {
    dir = d;
    ctx = (ctx_tokens + 255) / 256 * 256;
    nl = env_int("GLM_LAYERS", NL);
    mla_pos = env_int("GLM_MLA_POS", 1) != 0 && env_int("GLM_SEQS", 1) <= 1;  // round 26 ([glm-p3-mlapos]); the
    // pages follow one sequence's position: GLM_SEQS=2 keeps the per-layer arrays
    mla_pg = std::max(256, env_int("GLM_MLA_PAGE", mla_pg) / 256 * 256);
    const char* cvd = getenv("GLM_NGPU");
    int nd = 0;
    CUDA_CHECK(cudaGetDeviceCount(&nd));
    if (!cvd && !want_devs.empty()) {  // [glm-rel-host] the server's --devices (the first 3 of more)
        devs = want_devs;
        if (devs.size() > 3) {
            fprintf(stderr, "[glm] %zu cards given: GLM uses the first 3\n", devs.size());
            devs.resize(3);
        }
        ngpu = (int)devs.size();
    } else {
        ngpu = cvd ? atoi(cvd) : std::min(nd, 3);
        for (int i = 0; i < ngpu; i++) devs.push_back(i);
    }
    if (cvd && !want_devs.empty()) fprintf(stderr, "[glm] GLM_NGPU=%s: the server's --devices are not used\n", cvd);
    CHECK(ngpu >= 1 && ngpu <= 3, "GLM runs on 1-3 cards, %d asked (--devices / GLM_NGPU)", ngpu);
    for (int i = 0; i < ngpu; i++)  // review rel L16
        for (int j = 0; j < i; j++) CHECK(devs[i] != devs[j], "card %d listed twice (--devices)", devs[i]);
    for (int d : devs) CHECK(d >= 0 && d < nd, "card %d asked, %d visible", d, nd);
    if (ngpu < 3) fprintf(stderr, "[glm] %d card(s): %s\n", ngpu, ngpu == 1 ? "one card holds every dense weight (VRAM permitting)" : "fewer expert slots per card than on 3 cards");
    hcpu = host_cpus(ngpu - 1);
    fprintf(stderr, "[glm] host threads: %s\n", hcpu.why.c_str());
    {   // [glm-rel-host] routed expert columns per card: the server's split, else 704 / 640 / 704 with the 640 on the
        // card with the least free VRAM (a desktop / other process; ties: card 1, the old fixed choice)
        std::vector<int> ec = want_ecols;
        if (ec.empty() && ngpu == 3 && !getenv("GLM_ECOLS") && !getenv("GLM_EGROUPS")) {
            size_t fr[3] = {};
            for (int r = 0; r < 3; r++) {
                size_t t = 0;
                CUDA_CHECK(cudaSetDevice(devs[r]));
                CUDA_CHECK(cudaMemGetInfo(&fr[r], &t));
            }
            int lo = 1;
            for (int r = 0; r < 3; r++) if (fr[r] + (256u << 20) < fr[lo]) lo = r;
            ec = {704, 704, 704};
            ec[lo] = 640;
            // review rel L10: the split follows the free VRAM (a fixed 640 card costs slots when another card is the least
            // free: R30 3,675 -> 3,547 slots, -5.7 % tok/s); a split other than the tie default changes the partial-sum
            // order, so the line says when output bits will differ from loads with the default
            fprintf(stderr, "[glm] expert columns %d / %d / %d (free VRAM %.1f / %.1f / %.1f GB; GLM_ECOLS or --split to set)%s\n",
                    ec[0], ec[1], ec[2], fr[0] / 1e9, fr[1] / 1e9, fr[2] / 1e9,
                    lo != 1 ? "; NOTE: not the default 704 / 640 / 704, so output bits differ from loads with that split" : "");
            CUDA_CHECK(cudaSetDevice(devs[0]));
        }
        if (!ec.empty() && (int)ec.size() != ngpu) {
            fprintf(stderr, "[glm] split ignored: %zu values for %d cards\n", ec.size(), ngpu);
            ec.clear();
        }
        bool ok = true;
        int sum = 0;
        for (int x : ec) { ok &= x > 0 && x % 64 == 0; sum += x; }
        if (!ec.empty() && (!ok || sum != INTER)) {
            fprintf(stderr, "[glm] split ignored: columns must be multiples of 64 summing to %d\n", INTER);
            ec.clear();
        }
        plans = make_plan(ngpu, ec.empty() ? nullptr : ec.data());
    }
    const double t0 = now_s();
    SafeTensors* stp = new SafeTensors();
    SafeTensors& st = *stp;
    std::vector<std::string> shard_files;
    {
        std::ifstream f(dir + "/model.safetensors.index.json");
        CHECK(f.good(), "no index in %s", dir.c_str());
        std::string s((std::istreambuf_iterator<char>(f)), {});
        // files from the index (json.h parser)
        Json v = Json::parse(s);
        std::vector<std::string> files;
        std::map<std::string, std::set<std::string>> names;  // the index decides which file serves a name
        for (auto& kv : v["weight_map"].obj) {
            const std::string& fn = kv.second.str;
            if (std::find(files.begin(), files.end(), fn) == files.end()) files.push_back(fn);
            names[fn].insert(kv.first);
        }
        // re-quantised expert files (glm_quant.py mkckpt: "q_*") shadow the same names in the base shards: open them last
        std::stable_partition(files.begin(), files.end(), [](const std::string& f) { return f.rfind("q_", 0) != 0; });
        for (auto& fn : files) { st.open(dir + "/" + fn, &names[fn]); shard_files.push_back(dir + "/" + fn); }
        if (const char* mf = getenv("GLM_MTP_FILE")) {  // tuned MTP non-expert weights (same names) shadow the checkpoint
            st.open(mf, nullptr, true);  // review r17 M1: same names by design (master's duplicate check would throw)
            fprintf(stderr, "[glm] MTP weights from %s\n", mf);
        }
    }
    g_st = &st;
    // host experts: pinned, mapped, original layout
    hx_layer.assign(NL + 1, nullptr);
    {
        std::vector<std::thread> th;
        std::mutex mu;
        const int nth = 24;
        gu3 = false;
        {
            std::ifstream mf(dir + "/qglm.json");  // calibrated re-quantised experts (tools/glm/glm_quant.py mkckpt)
            if (mf.good()) {
                std::string ms((std::istreambuf_iterator<char>(mf)), {});
                Json mv = Json::parse(ms);
                gu3 = mv["gate_up_bits"].num == 3 && env_int("GLM_GU3", 1) != 0;
                gu3_mixed = gu3 && mv.has("mixed") && mv["mixed"].b;
            }
        }
        hxl.set(plans, gu3, gu3_mixed);
        if (gu3) fprintf(stderr, "[glm] host gate/up INT3 (%s), host stride %.2f MB per expert\n",
                         gu3_mixed ? "mixed tiers: per expert by code range" : "all experts", hxl.stride / 1e6);
        const size_t lb = (size_t)NE * hxl.stride;
        for (int l = 3; l < nl; l++) {
            void* m = mmap(nullptr, lb, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
            CHECK(m != MAP_FAILED, "mmap %zu", lb);
            madvise(m, lb, env_int("GLM_THP", 1) ? MADV_HUGEPAGE : MADV_NOHUGEPAGE);  // round 8: 2 MB pages, pre-faulted below (GLM_THP=0: 4 KB)
            hx_layer[l] = (char*)m;
        }
        if (env_int("GLM_THP", 1) >= 1) {  // round 8: fault the 2 MB pages in before the checkpoint reads fill the page cache
            const double tp = now_s();
            std::vector<std::thread> tp_th;
            std::atomic<int> nx{0};
            for (int i = 0; i < nth; i++)
                tp_th.emplace_back([&] {
                    for (int l; (l = 3 + nx++) < nl;)
                        for (size_t o = 0; o < lb; o += 2 << 20) ((volatile char*)hx_layer[l])[o] = 0;
                });
            for (auto& t : tp_th) t.join();
            fprintf(stderr, "[glm] host copy pre-faulted on huge pages: %.1f s\n", now_s() - tp);
        }
        std::atomic<int> next{0}, n_g3{0};
        const int njob = (nl - 3) * NE;
        for (int i = 0; i < nth; i++)
            th.emplace_back([&] {
                for (;;) {
                    const int j = next++;
                    if (j >= njob) break;
                    const int l = 3 + j / NE, e = j % NE;
                    const std::string b = P + "layers." + std::to_string(l) + ".mlp.experts." + std::to_string(e) + ".";
                    const char* src[6];
                    const char* nm[6] = {"gate_proj.qweight", "gate_proj.scales", "up_proj.qweight", "up_proj.scales",
                                         "down_proj.qweight", "down_proj.scales"};
                    const size_t sz[6] = {HX_GQ, HX_GS, HX_GQ, HX_GS, HX_DQ, HX_DS};
                    for (int q = 0; q < 6; q++) {
                        const TensorInfo& t = st.get(b + nm[q]);
                        CHECK(t.nbytes == sz[q], "%s%s %zu", b.c_str(), nm[q], t.nbytes);
                        src[q] = t.data;
                    }
                    const bool g3 = gu3_mixed && hx_fits_int3(src[0], src[2]);
                    if (g3) n_g3++;
                    hx_pack_expert(hxl, hx_layer[l] + (size_t)e * hxl.stride, src[0], src[1], src[2], src[3], src[4], src[5], g3);
                    if (e == 0 || e == 287) {  // spot check: symmetric zero point
                        const TensorInfo& z = st.get(b + "gate_proj.qzeros");
                        CHECK(((const uint32_t*)z.data)[0] == 0x77777777u, "asymmetric zero %s", b.c_str());
                    }
                }
            });
        for (auto& t : th) t.join();
        const double t1 = now_s();
        for (int l = 3; l < nl; l++)
            CUDA_CHECK(cudaHostRegister(hx_layer[l], lb, cudaHostRegisterPortable | cudaHostRegisterMapped));
        fprintf(stderr, "[glm] host experts %.1f GB: copy %.1f s, register %.1f s%s\n",
                (double)lb * (nl - 3) / 1e9, t1 - t0, now_s() - t1,
                gu3_mixed ? (" (gu3 experts: " + std::to_string(n_g3.load()) + ")").c_str() : "");
        if (FILE* f = fopen("/proc/self/smaps_rollup", "r")) {  // round 8: huge-page coverage of the host copy
            char ln[256];
            while (fgets(ln, sizeof ln, f))
                if (!strncmp(ln, "AnonHugePages", 13)) fprintf(stderr, "[glm] GLM_THP=%d, process %s", env_int("GLM_THP", 1), ln);
            fclose(f);
        }
    }
    // embedding: pinned host
    {
        const TensorInfo& t = st.get(P + "embed_tokens.weight");
        CUDA_CHECK(cudaHostAlloc(&emb, t.nbytes, cudaHostAllocPortable | cudaHostAllocMapped));
        memcpy(emb, t.data, t.nbytes);
    }
    W.resize(ngpu);
    for (int r = 0; r < ngpu; r++) {
        W[r] = std::make_unique<Worker>();
        W[r]->dev = devs[r];
        W[r]->p = plans[r];
    }
    mtp = nl == NL && env_int("GLM_MTP", 1);
    mtp_count = env_int("GLM_MTP_COUNT", 0) != 0;
    if (mtp) mtp_rbias = host_f32(P + "layers." + std::to_string(MTP_L) + ".mlp.gate.e_score_correction_bias");
    pf_pn = env_int("GLM_PF_PN", 1) != 0;
    pf_part = std::max(4096, (env_int("GLM_PF_PART", 8192) + 63) / 64 * 64);
    pf_edge = (env_int("GLM_PF_EDGE", 2048) + 63) / 64 * 64;
    pf_skiplent = env_int("GLM_PF_SKIPLENT", 0) != 0;
    pf_mhc2 = env_int("GLM_PF_MHC2", 1) != 0;
    pf_gu2 = env_int("GLM_PF_GU2", 0) != 0;
    pf_spb = env_int("GLM_PF_SPB", 1) != 0;
    pf_kov = env_int("GLM_PF_KOV", 1) != 0;
    pf_kc = env_int("GLM_PF_KC", 0) != 0;  // built, exact to 3e-7, but 5.2 vs 5.6 ms at 8K rows (latency-bound step): opt-in
    pf_gub = env_int("GLM_PF_GUB", 8);
    pf_hotfirst = env_int("GLM_PF_HOTFIRST", 0) != 0;  // 32K 3,551 -> 3,552, 8K 2,812 -> 2,678: off
    pf_head = (env_int("GLM_PF_HEAD", 0) + 63) / 64 * 64;
    pf_ring = env_int("GLM_PF_RING", 128);
    pf_ring_early = env_int("GLM_PF_RING_EARLY", -1);  // -1: by window size  // 32K 3,278 -> 3,293, 8K 2,403 -> 2,316 (event overhead per expert): opt-in
    // round 10: windows up to 64K rows with the parts (per-row window state ~130 KB, attention scratch per part)
    chunk = (env_int("GLM_CHUNK", pf_pn && env_int("GLM_PF_W8", 1) ? 65536 : 16384) + 63) / 64 * 64;
    pf_ov = env_int("GLM_PF_OV", 1) != 0;
    pf_cancel_mid = env_int("GLM_PF_CANCEL_MID", 1) != 0;  // [glm-rel3-cancel] 0: a cancel stops at window starts only
    pf_ss = env_int("GLM_PF_SS", 1) != 0;
    pf_mm = env_int("GLM_PF_MM", 1) != 0;
    pf_own = env_int("GLM_PF_OWN", 0) != 0;  // round 20: opt-in (exact with GLM_ROUTER_RB; 32K -3 % at load 12-20, 8K level)
    own_agov = env_int("GLM_OWN_PRE_OV", 0) != 0;
    own_la = env_int("GLM_OWN_LA", 0) != 0;
    own_side = env_int("GLM_OWN_SIDE", 1) != 0;
    router_rb = env_int("GLM_ROUTER_RB", 0);
    pf_lane_rows = env_int("GLM_PF_LANE_ROWS", 16);  // round 20: default 16 (gate 3: 1K -24 %, closer to the bf16 path); 0 = off
    pf_lane_maxt = env_int("GLM_PF_LANE_MAXT", 4096);
    pf_ubf = env_int("GLM_PF_UBF", 1) != 0;
    pf_f16 = env_int("GLM_PF_F16", 1) != 0;
    pf_d16 = env_int("GLM_PF_D16", 0) != 0;
    // round 24: copy-engine all-reduce for the overlapped prefill all-reduces (2, default: 32K -2.9 % with the packed
    // KDA recurrence, 8K +0.7 %); 1 = every prefill all-reduce (32K -2.4 %, 8K +1.8 %); 0 = SM P2P stores (off-switch)
    ar_ce_mode = env_int("GLM_AR_CE", 2);
    ar_ce = ar_ce_mode != 0;
    pf_rlk = env_int("GLM_PF_RLK", 1) != 0;  // round 27 (P4 b): prefill MoE row lists by a zero-copy kernel
    pf_w8 = env_int("GLM_PF_W8", 1) != 0;  // default on since round 6 (lossy knee, manager 2026-10-31); GLM_PF_W8=0 = lossless off-switch
    pf_w16 = env_int("GLM_PF_W16", 1) != 0;  // gate: TF and 47K on the floor
    pf_w16b = env_int("GLM_PF_W16B", 1) != 0;
    pf_ahead = env_int("GLM_PF_AHEAD", 0);  // 32K / 110K: off 2,534 / 2,394, mode 1 2,474 / 2,369, mode 2 2,458 / 2,331
    pf_grp = std::max(1, env_int("GLM_PF_GRP", 16));
    pf_w16t = env_int("GLM_PF_W16T", 0) != 0;
    pf_hyb = env_int("GLM_PF_HYB", 0) != 0;  // 32K 2,564 vs 2,578 fused: off
    pf_mtpfull = env_int("GLM_PF_MTPFULL", 0) != 0;  // r12: full MTP prefill pass (off-switch of F5)
    pf_ubatch = env_int("GLM_PF_UBATCH", 1) != 0;    // r12: batched MLA unions (one sync per call)
    pfstat = env_int("GLM_PFSTAT", 0) != 0;
    idx_split = env_int("GLM_IDX_SPLIT", 1) != 0;
    if (!pf_w8) pf_pn = false;  // the parts need the W4A8 expert path (its scratch is per row lists only)
    if (pf_w8 && !moe8n_mode())  // review r23-L4: read here once (single thread) and logged
        fprintf(stderr, "[glm] GLM_PF_MOEN=0: W4A8 prefill expert kernels in the round-6 form (round-23 kernels off)\n");
    kring = (pf_pn ? std::min(chunk, pf_part) : chunk) + 64;  // index-key ring: rows of one MLA call (+ a pool's tail)
    mla_nl = NL / 4 + 1;  // round 26: MLA layers 3, 7, .., 43 and the MTP layer
    mla_kcb = ((size_t)ctx * mla_nl * KVLR * 2 + 255) & ~(size_t)255;
    mla_pkb = ((size_t)(ctx / POOL + 1) * mla_nl * ID * 4 + 255) & ~(size_t)255;
    run_all([&](Worker& w) {
        w.init(this, w.p, w.dev);
        // review r26 L4: always set (a second engine in the process may use the other layout)
        mla_set_strides(mla_pos ? mla_nl * KVLR : KVLR, mla_pos ? mla_nl * ID : ID);
    });
    if (mtp) {  // MTP experts: packed once, resident on every card (its slices), host copy dropped
        HxLayout hxm;  // the MTP experts stay INT4 (AutoRound) in every checkpoint
        hxm.set(plans, false);
        std::vector<char> tmp((size_t)NE * hxm.stride);
        std::vector<std::thread> th;
        std::atomic<int> next{0};
        for (int i = 0; i < 16; i++)
            th.emplace_back([&] {
                for (int e; (e = next++) < NE;) {
                    const std::string b = P + "layers." + std::to_string(MTP_L) + ".mlp.experts." + std::to_string(e) + ".";
                    const char* nm[6] = {"gate_proj.qweight", "gate_proj.scales", "up_proj.qweight", "up_proj.scales",
                                         "down_proj.qweight", "down_proj.scales"};
                    const char* src[6];
                    for (int q = 0; q < 6; q++) src[q] = st.get(b + nm[q]).data;
                    hx_pack_expert(hxm, tmp.data() + (size_t)e * hxm.stride, src[0], src[1], src[2], src[3], src[4], src[5]);
                }
            });
        for (auto& t : th) t.join();
        run_all([&](Worker& w) {  // host slices are tile-major: land in a temporary block, untile into the slot
            char* t = nullptr;
            CUDA_CHECK(cudaMalloc(&t, w.sl.bytes));
            for (int e = 0; e < NE; e++) {
                CUDA_CHECK(cudaMemcpy(t, tmp.data() + (size_t)e * hxm.stride + hxm.off[w.p.rank], w.sl.bytes, cudaMemcpyHostToDevice));
                untile(t, w.mtp_slab + (size_t)e * w.sl.bytes, w.sl, 128, 0);
            }
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaFree(t));
        });
    }
    // release the checkpoint mapping and its page cache (the pinned copy holds the experts; RAM is the bottleneck)
    delete stp;
    g_st = nullptr;
    for (auto& fn : shard_files) {
        const int fd = open(fn.c_str(), O_RDONLY);
        if (fd >= 0) { posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED); close(fd); }
    }
    fprintf(stderr, "[glm] weights on %d cards: %.1f s\n", ngpu, now_s() - t0);
    ar_rows = std::min(chunk, std::max(64, (env_int("GLM_AR_ROWS", 4096) + 63) / 64 * 64));  // round 20: 8192 = one all-reduce block (one owned range) per part
    comm = comm_create(ngpu, ar_rows * H, H);
    if (ngpu > 1 && env_int("GLM_P2P", 1)) {
        const bool pp = comm_enable_p2p(comm, devs.data());
        fprintf(stderr, "[glm] P2P %s\n", pp ? "on" : "off");
        p2p = pp;
        if (pp) {
            std::vector<P2pAr> loc(ngpu);
            for (int r = 0; r < ngpu; r++) loc[r] = ar_p2p_alloc(devs[r], ngpu, MAXR * H);
            for (int r = 0; r < ngpu; r++) {
                W[r]->par.maxn = MAXR * H;
                for (int g = 0; g < ngpu; g++) { W[r]->par.rbuf[g] = loc[g].rbuf[0]; W[r]->par.flag[g] = loc[g].flag[0]; }
            }
        }
    }
    if (ar_ce && !(p2p && comm.pstage[0] && comm.cfl[0])) ar_ce = false;  // review r24 M2: no CE buffers (QE_AR_CE=0)
    if (ar_ce)
        run_all([&](Worker& w) {  // round 24: load the kernels of the copy-engine all-reduce now (scratch rows; prs2 / pstage
                                  // are rewritten before any read)
            const int m = 3 * ngpu;
            float* tmp = nullptr;
            CUDA_CHECK(cudaMalloc(&tmp, (size_t)m * H * 4));
            CUDA_CHECK(cudaMemsetAsync(tmp, 0, (size_t)m * H * 4, w.st));
            ce_pack(tmp, m, H, comm, w.p.rank, 0, w.st, 0);
            ce_sum(tmp, false, m, H, comm, w.p.rank, 0, 0, w.st);
            k_ce_out<<<1, 256, 0, w.st>>>(comm.prs2[w.p.rank], tmp, m * H / 4);
            CUDA_CHECK(cudaStreamSynchronize(w.st));
            CUDA_CHECK(cudaFree(tmp));
        });
    kernel_env_init();  // review r27 L4
    run_all([&](Worker&) { kda_rec_pk_init(); idx_dec_init(); });  // review r24 L1: the packed recurrence's shared-memory attribute at load
    run_all([&](Worker& w) {   // decode set; the prefill set lives in the lent slab tail
        w.alloc_acts(2 * MR);
        w.dec_acts = static_cast<Acts&>(w);
    });
    nseq = std::max(1, std::min(2, env_int("GLM_SEQS", 1)));
    eseq.assign(nseq, EngSeq{});
    if (nseq > 1) run_all([&](Worker& w) { w.alloc_seq(); });
    // expert cache slab: what is left minus a margin, the same slot count on every card
    size_t minfree = (size_t)-1;
    std::vector<size_t> avail(ngpu, 0);  // round 26: bytes the slab may take per card
    for (auto& w : W) {
        CUDA_CHECK(cudaSetDevice(w->dev));
        size_t fr = 0, tot = 0;
        CUDA_CHECK(cudaMemGetInfo(&fr, &tot));
        const size_t margin = (size_t)env_int("GLM_MARGIN_MB", 300) << 20;
        avail[w->p.rank] = fr > margin ? (fr - margin) & ~(size_t)255 : 0;
        const int ns = fr > margin ? (int)((fr - margin) / w->sl.bytes) : 0;
        fprintf(stderr, "[glm] card %d: free %.2f GB, slice %.2f MB -> %d slots\n", w->dev, fr / 1e9, w->sl.bytes / 1e6, ns);
        minfree = std::min(minfree, (size_t)ns);
    }
    const int ns = std::min<int>((int)minfree, env_int("GLM_SLOTS", 1 << 30));
    // round 26 (GLM_MLA_POS): each card's slab takes its free bytes (at most the slots + the region) and the MLA region
    // sits at its end; slots below the region's start on every card (mla_s0) are always experts' (and lendable), the
    // ones above hold experts while the rows in use do not reach them. A card with spare bytes past the common slot
    // count (smaller slices) keeps its region there.
    // The cards whose region overlaps the slots start it at the same slot index (the lowest that fits on each), so
    // equal slices put the rows in use under the same slots (else the union of their slot sets is taken).
    mla_s0 = ns;
    if (mla_pos) {  // [glm-rel-host] a context whose MLA region does not fit next to the prefill set is lowered (never
                    // refused: the user may run what loads), with a warning
        const int want = ctx;
        auto fits = [&] {
            const size_t R_ = mla_kcb + mla_pkb;
            int s0 = ns;
            for (auto& w : W) {
                if (avail[w->p.rank] < R_) return false;
                const size_t o = avail[w->p.rank] - R_;
                s0 = std::min<int>(s0, (int)(std::min(o, (size_t)ns * w->sl.bytes) / w->sl.bytes));
            }
            return s0 > 0 && s0 >= lend_slots(std::min(chunk, 16384)) + 64;
        };
        while (ctx > 8192 && !fits()) {
            ctx = std::max(8192, ctx - 16384);
            mla_kcb = ((size_t)ctx * mla_nl * KVLR * 2 + 255) & ~(size_t)255;
            mla_pkb = ((size_t)(ctx / POOL + 1) * mla_nl * ID * 4 + 255) & ~(size_t)255;
        }
        if (ctx != want)
            fprintf(stderr, "[glm] WARNING: context lowered %d -> %d (the MLA cache of %d positions does not fit next to the "
                    "expert slots on these cards)\n", want, ctx, want);
        const size_t R = mla_kcb + mla_pkb;
        int sr = ns;
        for (auto& w : W) {
            CHECK(avail[w->p.rank] >= R, "card %d: no room for the MLA cache of %d positions (%.2f GB)", w->dev, ctx, R / 1e9);
            const size_t o = avail[w->p.rank] - R;
            if (o < (size_t)ns * w->sl.bytes) sr = std::min<int>(sr, (int)(o / w->sl.bytes));
        }
        for (auto& w : W) {
            const size_t o = avail[w->p.rank] - R;
            w->mla_off = o >= (size_t)ns * w->sl.bytes ? (size_t)ns * w->sl.bytes : (size_t)sr * w->sl.bytes;
            w->slab_b = std::max((size_t)ns * w->sl.bytes, w->mla_off + R);
            mla_s0 = std::min<int>(mla_s0, (int)(w->mla_off / w->sl.bytes));
        }
    }
    CHECK(mla_s0 > 0, "slab of %d slots cannot hold the MLA cache of %d positions (%.2f GB per card)", ns, ctx,
          (mla_kcb + mla_pkb) / 1e9);
    // the widest window that leaves >= 1/4 of the slots resident (round 10: windows up to 64K rows)
    while (chunk > 16384 && lend_slots(chunk) > mla_s0 * 3 / 4) chunk -= 8192;
    CHECK(mla_s0 >= lend_slots(chunk), "slab of %d slots cannot hold the prefill set (%d slots)", mla_s0, lend_slots(chunk));
    fprintf(stderr, "[glm] prefill set of %d rows lends %d slots\n", chunk, lend_slots(chunk));
    run_all([&](Worker& w) {
        w.nslots = ns;
        if (!mla_pos) w.slab_b = (size_t)ns * w.sl.bytes;
        if (w.slab_b) CUDA_CHECK(cudaMalloc(&w.slab, w.slab_b));
        if (!mla_pos) return;
        // review r26-6: the region is not cleared: rows past the positions written so far hold whatever expert bytes the
        // slots under them held (NaN patterns included). Every reader bounds its rows by the position (attention, pooled
        // keys, the archive / snapshot copies), so an unwritten row is never read.
        char* base = w.slab + w.mla_off;  // [pos][layer][512] bf16, then [pool][layer][128] fp32
        for (int l = 0; l < (int)w.L.size(); l++) {
            Layer& Y = w.L[l];
            if (!Y.mla) continue;
            Y.kc = (bf16*)base + (size_t)mla_li(l) * KVLR;
            Y.pk = (float*)(base + mla_kcb) + (size_t)mla_li(l) * ID;
            if (!w.banks.empty()) { w.banks[w.cur_seq].ls[l].kc = Y.kc; w.banks[w.cur_seq].ls[l].pk = Y.pk; }
        }
    });
    run_all([&](Worker& w) {
        w.mem["expert cache slots"] += (double)w.nslots * w.sl.bytes;
        if (mla_pos) w.mem["MLA cache at the slab end (shares slots)"] += (double)(w.slab_b - (size_t)w.nslots * w.sl.bytes);
        size_t fr = 0, tot = 0;
        CUDA_CHECK(cudaMemGetInfo(&fr, &tot));
        double used = 0;
        for (auto& kv : w.mem) used += kv.second;
        w.mem["(CUDA context, comm, graphs, other)"] = (double)(tot - fr) - used;
        w.mem["(free)"] = (double)fr;
    });
    {
        std::string s = "[glm] VRAM by kind (GB), cards 0 / 1 / 2:\n";
        for (auto& kv : W[0]->mem) {
            char line[160];
            snprintf(line, sizeof line, "  %-40s", kv.first.c_str());
            s += line;
            for (auto& w : W) { snprintf(line, sizeof line, " %7.3f", w->mem[kv.first] / 1e9); s += line; }
            s += "\n";
        }
        fprintf(stderr, "%s", s.c_str());
    }
    slot_of.assign((size_t)(NL + 1) * NE, -1);
    slot_owner.assign(ns, -1);
    slot_mla.assign(ns, 0);
    if (mla_pos) {  // the first page is always held (the seed never places experts under it)
        slot_mla = mla_mask(mla_page_rows(0));
        mla_rows = mla_page_rows(0);
        int nm = 0;
        for (char c : slot_mla) nm += c;
        fprintf(stderr, "[glm] MLA cache position-major in the slab from slot %d (%.2f GB per card at %d positions; "
                "pages of %d positions, %d slots under the first)\n", mla_s0, (mla_kcb + mla_pkb) / 1e9, ctx, mla_pg, nm);
    }
    coord_start();  // cache seeded at load: the prefill already finds the prior's experts resident
    policy_init();
    fprintf(stderr, "[glm] ready: %d expert slots per card (%.1f GB on rank 0), %.1f s\n", ns,
            (double)ns * W[0]->sl.bytes / 1e9, now_s() - t0);
}

// [glm-rel-pfgate] worker `rank` before a layer of a gated prefill (win_start: layer 0 of a window). true: stop here.
// [glm-rel3-cancel] stop_win is set only while every card waits at the same gate (job_cancel), so every card stops at
// that gate: at a window start, or (pf_cancel_mid) before any layer inside the window
bool Engine::pf_gate(int rank, bool win_start, int prog) {
    std::unique_lock<std::mutex> lk(pfg.mu);
    const bool here = win_start || (pf_cancel_mid && pf_mid_ok);
    if (here && pfg.stop_win) return true;
    if (rank == 0) pfg.prog = prog;
    while (!pfg.free && pfg.used[rank] >= pfg.credit) {
        pfg.waiting++;
        pfg.cv.notify_all();
        pfg.cv.wait(lk);
        pfg.waiting--;
        if (here && pfg.stop_win) return true;
    }
    pfg.used[rank]++;
    return false;
}

// [glm-rel3-cancel] a window stopped before layer l (every card at the same gate, so every card has queued exactly the
// layers < l, collectives included): its queued work runs to its end on every stream of this card (the copy-engine
// thread has issued this card's copies of every all-reduce queued so far: ar_ce takes their events before it returns),
// then the per-window state that the next layer or window would consume goes. The rows the window wrote stay dead
// (the caller rolls back to the window start).
void Worker::pf_abort() {
    CUDA_CHECK(cudaSetDevice(dev));
    for (cudaStream_t s_ : {st, krst, gst, cms, pst, ags, ost, cst, dmst, lst, bst, ce_st[0], ce_st[1], ce_st[2]})
        if (s_) CUDA_CHECK(cudaStreamSynchronize(s_));
    pn_pend = ov_pend = ov_mixA = pf_lmix = false;
    own = own_routed = false;
    staged_l = sr_l = -1;
    lane_e.clear(); lane_rows.clear(); lane_w.clear();
}

// one forward of T tokens on every card; argmax per row (am) and full logits of the rows in lrows (lout [n][VOCAB])
void Engine::forward(const int* tok, int T, float* lout, std::vector<int>* am, const std::vector<int>* lrows) {
    CHECK(pos + T <= ctx, "context full (%d + %d > %d)", pos, T, ctx);
    CHECK(!ver_pending, "prefill after an uncommitted verify (KDA pointer mode)");  // r12 (review C4)
    bg_flush();
    det_wait();  // round 17: an asynchronous cache restore has landed
    kda_mat();  // round 11: the prefill reads S
    if (bg_bound) run_all([&](Worker& w) { CUDA_CHECK(cudaStreamSynchronize(w.bst)); });  // boundary copies landed
    const int n0 = pos;
    mla_need(pos + T);  // round 26: the MLA pages under the new rows are taken before the prefill set is lent
    // windows of equal size (round 10: one window as wide as the prompt where VRAM allows)
    lend(pf_window(T));
    std::vector<std::vector<float>> mxv(ngpu), parts(ngpu);
    std::vector<std::vector<int>> mxi(ngpu);
    std::vector<float> lastv(ngpu, -1e30f);
    std::vector<int> lasti(ngpu, 0);
    const bool gated = pfg.on;  // [glm-rel-pfgate]
    std::vector<int> stop_at(ngpu, -1);
    // [glm-rel3-cancel] a gated live continuation (ck_prefill: from the live end, no MTP row rewritten) saves its start
    const bool keep = gated && pf_keep && n0 > 0 && !mtp_prev && pf_ws_ready();
    pf_live_kept = false;
    {
        std::lock_guard<std::mutex> lk(pfg.mu);
        pf_mid_ok = !(gated && pf_keep && !keep);  // review R3-M1: an unsaved live continuation stops at window starts only
    }
    run_all([&](Worker& w) {
        const int r = w.p.rank;
        const int nv = w.p.v1 - w.p.v0;
        // round 18: a forward resuming at a cut / restored snapshot also writes the MTP cache row n0 - 1 (its hidden in
        // mh, the next token tok[0]): a split prefill never writes that row otherwise
        if (mtp && mtp_prev && n0 > 0) w.mtp_prev_row(tok[0], n0 - 1);
        if (keep) {  // [glm-rel3-cancel] the state a stop in window 0 goes back to
            const auto ts = std::chrono::steady_clock::now();
            w.ck_io(pf_ws[r], n0, true, false);
            if (ck_log && r == 0)
                fprintf(stderr, "[glm] cancel save at %d: %.1f ms\n", n0,
                        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - ts).count());
        }
        for (int c0 = 0; c0 < T; c0 += w.mx) {
            const int nt = std::min(w.mx, T - c0);
            if (gated && pf_gate(r, true, n0 + c0)) {  // (a cancel: the window is not started)
                w.pf_abort();  // [glm-rel3-cancel] (also the next window's early expert staging done)
                stop_at[r] = c0;
                break;
            }
            CUDA_CHECK(cudaMemcpyAsync(w.d_tok, tok + c0, nt * 4, cudaMemcpyHostToDevice, w.st));
            embed(w.d_tok, emb, w.R, nt, w.st);
            if (!dump_dir.empty() && r == 0) w.dump("R_in", -1, w.R, (size_t)nt * RW);
            const bool pn = w.mxa < w.mx || (pf_pn && pf_w8 && ngpu > 1 && nt >= 4096 && dump_dir.empty());
            CHECK(!pn || dump_dir.empty(), "dumps need GLM_PF_PN=0");
            const bool ov = !pn && pf_ov && ngpu > 1 && nt >= 4096 && dump_dir.empty();
            w.pf_more = c0 + nt < T && T - c0 - nt >= 2048;
            if (pn) w.pn_split(nt);
            // round 20: row-owner merge / mHC / router (opt-in GLM_PF_OWN=1); P2P parts windows only
            w.own = pn && pf_own && ngpu > 1 && dump_dir.empty() && pf_mm && pf_mhc2 && w.own_ok(nt);
            if (pfstat && r == 0) fprintf(stderr, "[glm] window %d rows: %d parts, owner %s\n", nt, pn ? w.pn_n : 1, w.own ? "on" : "off");
            // ring DMA next to the attention below 12K rows (DMA-bound MoE: 4K +15 %, 8K +8 %), with the MoE above
            // (the DMA next to the attention all-reduces costs 1-3 % at 16-32K)
            w.sr_early = pf_ring_early >= 0 ? pf_ring_early != 0 : nt < 20480;  // 16K: 3,029 -> 3,159 with the grouped gate/up
            bool stop = false;
            for (int l = 0; l < nl; l++) {
                if (gated && l > 0 && pf_gate(r, false, n0 + c0 + (int)((long long)nt * l / nl))) { stop = true; break; }
                if (pn) w.layer_pn(l, nt, n0 + c0);
                else if (ov) w.layer_ov(l, nt, n0 + c0);
                else w.layer(l, nt, n0 + c0);
                if (!dump_dir.empty() && r == 0) w.dump("R", l, w.R, (size_t)nt * RW);
            }
            if (stop) {  // [glm-rel3-cancel] a cancel inside the window: back to its start
                w.pf_abort();
                stop_at[r] = c0;
                break;
            }
            const bool owned = pn && w.own;
            if (pn) w.pn_finish();
            w.own = false;
            if (ov) w.ov_finish();
            if (!owned) final_norm(w.R, w.final_norm, w.xb, w.sum, nt, w.st);  // (owner mode: done by pn_finish)
            int cke = -1;  // round 18: snapshot due at this window end
            for (auto& pe : ck_plan) if (pe.first == n0 + c0 + nt) cke = pe.second;
            if (cke >= 0)
                CUDA_CHECK(cudaMemcpyAsync(w.ck_h, w.sum + (size_t)(nt - 1) * H, H * 4, cudaMemcpyDeviceToDevice, w.st));
            if (hid_dump && r == 0)
                CUDA_CHECK(cudaMemcpyAsync(hid_dump + (size_t)c0 * H, w.sum, (size_t)nt * H * 4, cudaMemcpyDeviceToHost, w.st));
            // round 17 (advisor P1, GLM_PF_LAST1=0 off): only the prompt's last row is read (no argmax list, no selected
            // rows) -> one row through the lm_head instead of its 256-row block, ~200 KB to the host instead of ~53 MB
            const bool last1 = pf_last1 && !am && !lrows && c0 + nt == T;
            if (last1) {
                w.lin(w.lm, nullptr, w.xb + (size_t)(nt - 1) * H, w.logits, 1);
                std::vector<float> hl((size_t)nv);
                CUDA_CHECK(cudaMemcpyAsync(hl.data(), w.logits, hl.size() * 4, cudaMemcpyDeviceToHost, w.st));
                CUDA_CHECK(cudaStreamSynchronize(w.st));
                int bi = 0;
                for (int j = 1; j < nv; j++) if (hl[j] > hl[bi]) bi = j;
                lastv[r] = hl[bi];
                lasti[r] = w.p.v0 + bi;
                if (lout) memcpy(lout + w.p.v0, hl.data(), (size_t)nv * 4);
            }
            // logits in blocks of 256 rows: local argmax (+ selected full rows)
            for (int b0 = 0; b0 < nt && !last1; b0 += 256) {
                const int nb = std::min(256, nt - b0);
                const bool need = am || (lrows && std::any_of(lrows->begin(), lrows->end(), [&](int x) {
                    return x >= c0 + b0 && x < c0 + b0 + nb; })) || c0 + b0 + nb == T;
                if (!need) continue;
                w.lin(w.lm, nullptr, w.xb + (size_t)b0 * H, w.logits, nb);
                std::vector<float> hl((size_t)nb * nv);
                CUDA_CHECK(cudaMemcpyAsync(hl.data(), w.logits, hl.size() * 4, cudaMemcpyDeviceToHost, w.st));
                CUDA_CHECK(cudaStreamSynchronize(w.st));
                for (int i = 0; i < nb; i++) {
                    const int row = c0 + b0 + i;
                    const float* x = hl.data() + (size_t)i * nv;
                    if (am) {
                        int bi = 0;
                        for (int j = 1; j < nv; j++) if (x[j] > x[bi]) bi = j;
                        mxv[r].push_back(x[bi]);
                        mxi[r].push_back(w.p.v0 + bi);
                    }
                    if (row == T - 1) {
                        int bi = 0;
                        for (int j = 1; j < nv; j++) if (x[j] > x[bi]) bi = j;
                        lastv[r] = x[bi];
                        lasti[r] = w.p.v0 + bi;
                    }
                    int li = -1;
                    if (lrows) { auto it = std::find(lrows->begin(), lrows->end(), row); if (it != lrows->end()) li = (int)(it - lrows->begin()); }
                    else if (lout && row == T - 1) li = 0;
                    if (li >= 0 && lout) memcpy(lout + (size_t)li * VOCAB + w.p.v0, x, (size_t)nv * 4);
                }
            }
            if (mtp) w.mtp_prefill(tok, T, c0, nt, n0);  // after the logits: it reuses xb / R
            if (cke >= 0) w.ck_io(ck[cke].buf[r], n0 + c0 + nt, true);
        }
        CUDA_CHECK(cudaStreamSynchronize(w.st));
    });
    pf_stop_rows = stop_at[0];
    for (int r = 1; r < ngpu; r++) CHECK(stop_at[r] == stop_at[0], "gated prefill: cards stopped at different windows");
    if (pf_stop_rows >= 0) {  // [glm-rel-pfgate] a cancel: the rows of the whole windows stay (a consistent state)
        if (pf_stop_rows == 0 && keep) {  // [glm-rel3-cancel] stopped in window 0: the KDA / conv / ring state of n0 back
            run_all([&](Worker& w) { w.ck_io(pf_ws[w.p.rank], n0, false, false); });
            pf_live_kept = true;
        }
        pos += pf_stop_rows;
        mtp_prev = false;
        ck_plan.clear();
        run_all([&](Worker& w) {
            CUDA_CHECK(cudaMemcpyAsync(w.d_pos, &pos, 4, cudaMemcpyHostToDevice, w.st));
            CUDA_CHECK(cudaStreamSynchronize(w.st));
        });
        unlend();
        return;
    }
    if (am) {
        am->assign(T, 0);
        for (int t = 0; t < T; t++) {
            int best = 0;
            for (int r = 1; r < ngpu; r++) if (mxv[r][t] > mxv[best][t]) best = r;
            (*am)[t] = mxi[best][t];
        }
    }
    {
        int best = 0;
        for (int r = 1; r < ngpu; r++) if (lastv[r] > lastv[best]) best = r;
        last_tok = lasti[best];
    }
    pos += T;
    mtp_prev = false;
    ck_plan.clear();
    run_all([&](Worker& w) {
        CUDA_CHECK(cudaMemcpyAsync(w.d_pos, &pos, 4, cudaMemcpyHostToDevice, w.st));
        CUDA_CHECK(cudaStreamSynchronize(w.st));
    });
    {
        const auto tu = std::chrono::steady_clock::now();
        unlend();
        if (pfstat && st_pflane_e)
            fprintf(stderr, "[glm] prefill %d rows: CPU lane %lld experts, %lld rows, %.1f ms\n", T, st_pflane_e, st_pflane_rows,
                    st_pflane_us / 1e3);
        st_pflane_us = 0; st_pflane_e = st_pflane_rows = 0;
        if (pfstat)
            for (auto& wp : W) {
                Worker& w = *wp;
                fprintf(stderr, "[glm] prefill %d rows: card %d host waits: peer events %.1f ms, MLA union %.1f, MoE routing %.1f, lane %.1f\n",
                        T, w.p.rank, w.hw_ce_us / 1e3, w.hw_mla_us / 1e3, w.hw_moe_us / 1e3, w.hw_lane_us / 1e3);
                w.hw_ce_us = w.hw_mla_us = w.hw_moe_us = w.hw_lane_us = 0;
            }
        if (pfstat) fprintf(stderr, "[glm] prefill %d rows: unlend (re-seed) %.1f ms\n", T,
                            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tu).count());
    }
}

// MTP rows of a prefill chunk: positions n0 + c0 + i with the hidden of row i (in sum) and token c0 + i + 1; the last
// prompt row (no next token yet) only keeps its hidden for the first decode catch-up (mh row 0)
void Worker::mtp_prefill(const int* tok, int T, int c0, int nt, int n0) {
    Layer& Y = L[MTP_L];
    if (c0 + nt == T) CUDA_CHECK(cudaMemcpyAsync(mh, sum + (size_t)(nt - 1) * H, H * 4, cudaMemcpyDeviceToDevice, st));
    const int nr = c0 + nt == T ? nt - 1 : nt;
    if (nr <= 0) return;
    CUDA_CHECK(cudaMemcpyAsync(d_tok, tok + c0 + 1, (size_t)nr * 4, cudaMemcpyHostToDevice, st));
    bf16* xc = (bf16*)R;
    float* xr = R + (size_t)nr * H;
    mtp_in(d_tok, E->emb, sum, mw.enorm, mw.hnorm, xc, nr, st);
    // keep the eh input rows: R is free after the final norm
    lin(mw.eh, nullptr, xc, xr, nr);
    add_rms(xr, nullptr, Y.in_ln, xf, xb, nr, st);
    // r12 (prefill review F5): decode recomputes the MTP pass from mh, so the prefill only has to fill the MTP layer's
    // cache (latents, index keys, pool keys); its attention, MoE and both all-reduces over every prompt row were dead
    // work. GLM_PF_MTPFULL=1: the old full pass (same cache rows)
    const bool full = E->pf_mtpfull;
    {   // in parts of <= mxa rows (scratch and index-key ring of one part)
        const Acts base = *this;
        for (int r0 = 0; r0 < nr; r0 += mxa) {
            const int m = std::min(mxa, nr - r0);
            view_from(*this, base, r0);
            if (full) {
                mla(MTP_L, m, n0 + c0 + r0);
                allreduce(m);
            } else mla_cache(MTP_L, m, n0 + c0 + r0);
        }
        static_cast<Acts&>(*this) = base;
    }
    if (full) {
        add_rms(xr, sum, Y.post_ln, xf, xb, nr, st);
        moe(MTP_L, nr);
        allreduce(nr);
    }
    CUDA_CHECK(cudaStreamSynchronize(st));
}
// cache rows of an MLA layer only (what decode reads): projections, latents / index keys, pool keys
void Worker::mla_cache(int l, int T, int n0) {
    Layer& Y = L[l];
    float* proj = big;
    float* wi = proj + (size_t)T * MLA_IN;
    bf16* qr = bigb;
    lin(Y.mw_in, xf, xb, proj, T);
    mla_prep(proj, Y.q_a_ln, Y.kv_a_ln, Y.ik_w, Y.ik_b, qr, Y.kc, Y.ki, Y.gsc, wi, n0, T, st, nullptr, E->kring);
    pool_keys(Y.ki, Y.gsc, Y.ape, Y.pk, n0 / POOL, (n0 + T) / POOL, st, E->kring);
}

void Worker::dump(const char* name, int l, const float* d, size_t n) {
    std::vector<float> h(n);
    CUDA_CHECK(cudaMemcpyAsync(h.data(), d, n * 4, cudaMemcpyDeviceToHost, st));
    CUDA_CHECK(cudaStreamSynchronize(st));
    char fn[512];
    snprintf(fn, sizeof fn, "%s/%s_%d_p%d.f32", E->dump_dir.c_str(), name, l, E->pos);
    FILE* f = fopen(fn, "wb");
    CHECK(f, "dump %s", fn);
    fwrite(h.data(), 4, n, f);
    fclose(f);
}

void Engine::reset(bool keep_ck) {
    pos = 0;
    if (keep_ck) lv.clear();  // [glm-conv-arc]: the archive keeps what snapshots need (ck_prefill saved it first)
    else ck_forget();         // round 18: no cache rows behind any prefix snapshot any more
    ver_pending = false;  // the state is zeroed below
    bg_flush();
    run_all([&](Worker& w) { CUDA_CHECK(cudaMemsetAsync(w.d_pos, 0, 4, w.st)); });
    run_all([&](Worker& w) {
        for (auto& Y : w.L) {
            if (Y.S) CUDA_CHECK(cudaMemsetAsync(Y.S, 0, (size_t)w.p.nh() * KD * KD * 4, w.st));
            if (w.d_acc) CUDA_CHECK(cudaMemsetAsync(w.d_acc, 0, 8, w.st));
            if (Y.conv_hist) CUDA_CHECK(cudaMemsetAsync(Y.conv_hist, 0, (size_t)9 * w.p.nh() * KD * 4, w.st));
        }
        CUDA_CHECK(cudaStreamSynchronize(w.st));
    });
    det_restore();  // round 17: a new sequence starts from the seeded cache state (GLM_DET=0 off)
    // round 26: only the first MLA page stays held (no row is live after a reset); ck_prefill's fresh start (keep_ck)
    // sets the pages of its first prefill instead (review r26 M1: no seed refill of slots the prefill takes again)
    if (!keep_ck) mla_set(0);
}

}  // namespace glm
