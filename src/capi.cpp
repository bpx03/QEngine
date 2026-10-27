#include <cstring>
#include <vector>
#include <random>
// C API used by the Python tests (ctypes) and the server.
#include "kernels/gemv.cuh"
#include "kernels/ops.cuh"

extern "C" {

// ---- kernel test hooks: all pointers are device pointers, default stream ----
void qe_repack_w4(const int32_t* q, const bf16* s, int n, int k, void* dq, void* ds) {
    repack_w4(q, s, n, k, (uint4*)dq, (bf16*)ds, 0);
}
void qe_repack_w16(const bf16* w, int n, int k, void* dw) { repack_w16(w, n, k, (uint4*)dw, 0); }

void qe_gemv_w4(const void* q, const void* s, int n, int k, const bf16* x, float* y, int m) {
    W4 w{(const uint4*)q, (const bf16*)s, n, k};
    gemv_workspace_init();
    gemv_w4(w, x, k, y, n, m, 0);
}
void qe_gemv_w4g(const void* q, const void* s, int n, int k, const bf16* x, float* y, int m, int glu) {
    W4 w{(const uint4*)q, (const bf16*)s, n, k, glu != 0};
    gemv_workspace_init();
    gemv_w4(w, x, k, y, glu ? n / 2 : n, m, 0);
}
// fp16-accumulate GEMV test hook: packs x into XH itself; glu: y is the bf16 act buffer, yh (optional)
// the XH act buffer of the same width
void qe_gemv_w4h(const void* q, const void* s, int n, int k, const bf16* x, float* y, int m, int glu,
                 void* xh_ws, void* yh) {
    W4 w{(const uint4*)q, (const bf16*)s, n, k, glu != 0};
    gemv_workspace_init();
    xh_pack(x, k, k, m, (__half*)xh_ws, k, 0);
    gemv_w4h(w, (const __half*)xh_ws, k, y, glu ? n / 2 : n, m, 0, (__half*)yh, glu ? n / 2 : n);
}
void qe_xh_pack(const bf16* x, int k, int m, void* xh, int ldxh) { xh_pack(x, k, k, m, (__half*)xh, ldxh, 0); }
// kernel only (x already in XH form, stride k)
void qe_gemv_w4hx(const void* q, const void* s, int n, int k, const __half* xh, float* y, int m, int glu, int ldx) {
    W4 w{(const uint4*)q, (const bf16*)s, n, k, glu != 0};
    gemv_workspace_init();
    gemv_w4h(w, xh, ldx, y, glu ? n / 2 : n, m, 0);
}
// wide kernel only (gemv_w4hx), returns 0 if unsupported
int qe_gemv_w4hxx(const void* q, const void* s, int n, int k, const __half* xh, float* y, int m, int glu, int ldx) {
    W4 w{(const uint4*)q, (const bf16*)s, n, k, glu != 0};
    gemv_workspace_init();
    return gemv_w4hx(w, xh, ldx, y, glu ? n / 2 : n, m, 0) ? 1 : 0;
}
int qe_skinny_w4(const void* q, const void* s, int n, int k, const bf16* x, float* y, int m, int glu) {
    W4 w{(const uint4*)q, (const bf16*)s, n, k, glu != 0};
    skinny_init();
    return skinny_w4(w, x, k, y, glu ? n / 2 : n, m, 0) ? 1 : 0;
}
void qe_gemm_w4(const void* q, const void* s, int n, int k, const bf16* x, float* y, int t) {
    W4 w{(const uint4*)q, (const bf16*)s, n, k};
    gemm_w4(w, x, k, y, n, t, 0);
}
// co-scheduling experiments: kernels on explicit (prioritized, non-blocking) streams
void* qe_stream_create(int priority) {
    cudaStream_t s;
    CUDA_CHECK(cudaStreamCreateWithPriority(&s, cudaStreamNonBlocking, priority));
    return s;
}
void qe_stream_sync(void* s) { CUDA_CHECK(cudaStreamSynchronize((cudaStream_t)s)); }
void qe_gemm_w4_s(const void* q, const void* s, int n, int k, const bf16* x, float* y, int t, void* st) {
    W4 w{(const uint4*)q, (const bf16*)s, n, k};
    gemm_w4(w, x, k, y, n, t, (cudaStream_t)st);
}
void qe_gemv_w4_s(const void* q, const void* s, int n, int k, const bf16* x, float* y, int m, void* st) {
    gemv_workspace_init();
    W4 w{(const uint4*)q, (const bf16*)s, n, k};
    gemv_w4(w, x, k, y, n, m, (cudaStream_t)st);
}
void qe_gemm_w4_i8(const void* q, const void* s, int n, int k, const bf16* x, float* y, int t, int G,
                   int ci, int ks) {
    W4 w{(const uint4*)q, (const bf16*)s, n, k};
    gemm_w4_i8(w, x, k, y, n, t, 0, G, ci, ks);
}
void qe_gemm_w4_i8_glu(const void* q, const void* s, int n, int k, const bf16* x, void* act, int t,
                       int G, int ci, int ks) {
    W4 w{(const uint4*)q, (const bf16*)s, n, k, true};
    gemm_w4_i8(w, x, k, (float*)act, n / 2, t, 0, G, ci, ks);
}
void qe_gemm_set_out_bf16(int on) { gemm_set_out_bf16(on != 0); }
void qe_gemm_set_sk_maxparts(int n) { gemm_set_sk_maxparts(n); }
void qe_gemm_w4_t(const void* q, const void* s, int n, int k, const bf16* x, float* y, int t) {  // gemm_w4 (tuned)
    W4 w{(const uint4*)q, (const bf16*)s, n, k};
    gemm_w4(w, x, k, y, n, t, 0);
}
void qe_gemm_w16(const void* wp, int n, int k, const bf16* x, float* y, int t) {
    W16 w{(const uint4*)wp, n, k};
    gemm_w16(w, x, k, y, n, t, 0);
}
void qe_quantize_w8(const bf16* w, int n, int k, void* dq, void* ds) {
    quantize_w8(w, n, k, (uint4*)dq, (bf16*)ds, 0);
}
void qe_gemv_w8(const void* q, const void* s, int n, int k, const bf16* x, float* y, int m) {
    W8 w{(const uint4*)q, (const bf16*)s, n, k};
    gemv_w8(w, x, k, y, n, m, 0);
}
void qe_gemv_workspace_init() { gemv_workspace_init(); }
void qe_occ(int threads) { gemv_occupancy_report(threads); }
// dn_in + ab-projection co-scheduling microbench: mode 0 dn_in only, 1 ab only,
// 2 engine pattern (fork: ab on st2 beside dn_in on st1, join), 3 ab then dn_in on st1, 4 dn_in then ab on st1
extern int g_w4hw3_mode;
void qe_set_w4hw3_mode(int v) { g_w4hw3_mode = v; }

void qe_set_carveout(int pct, int which) { gemv_set_carveout(pct, which); }
void qe_dnin_ab(int mode, const void* q, const void* s, int n, int k, const __half* xh, float* y, int m,
                const void* abw, int abn, float* aby, void* st1, void* st2) {
    static cudaEvent_t fork = nullptr, join = nullptr;
    if (!fork) {
        CUDA_CHECK(cudaEventCreateWithFlags(&fork, cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&join, cudaEventDisableTiming));
    }
    gemv_workspace_init();
    W4 w{(const uint4*)q, (const bf16*)s, n, k, false};
    W16 a{(const uint4*)abw, abn, k};
    cudaStream_t s1 = (cudaStream_t)st1, s2 = (cudaStream_t)st2;
    const size_t abs = (size_t)kChunk * abn;
    if (mode == 0) gemv_w4h(w, xh, k, y, n, m, s1);
    else if (mode == 1) gemv_w16h(a, xh, k, aby, abn, m, s1, kAbSplits, abs);
    else if (mode == 2) {
        CUDA_CHECK(cudaEventRecord(fork, s1));
        CUDA_CHECK(cudaStreamWaitEvent(s2, fork, 0));
        gemv_w16h(a, xh, k, aby, abn, m, s2, kAbSplits, abs);
        CUDA_CHECK(cudaEventRecord(join, s2));
        gemv_w4h(w, xh, k, y, n, m, s1);
        CUDA_CHECK(cudaStreamWaitEvent(s1, join, 0));
    } else if (mode == 3) {
        gemv_w16h(a, xh, k, aby, abn, m, s1, kAbSplits, abs);
        gemv_w4h(w, xh, k, y, n, m, s1);
    } else {
        gemv_w4h(w, xh, k, y, n, m, s1);
        gemv_w16h(a, xh, k, aby, abn, m, s1, kAbSplits, abs);
    }
}
void qe_gemv_w16(const void* wp, int n, int k, const bf16* x, float* y, int m) {
    W16 w{(const uint4*)wp, n, k};
    gemv_w16(w, x, k, y, n, m, 0);
}
}

// ---- engine ----
#include <sstream>

#include "engine.h"

extern "C" {

// devices: comma separated CUDA ordinals, e.g. "0,1,2"; split: optional weights "1,1,1"
void* qe_create(const char* model_dir, const char* devices, int max_t, const char* split) {
    std::vector<int> devs;
    std::vector<double> w;
    std::stringstream ss(devices);
    for (std::string x; std::getline(ss, x, ',');) devs.push_back(std::stoi(x));
    if (split && *split) {
        std::stringstream s2(split);
        for (std::string x; std::getline(s2, x, ',');) w.push_back(std::stod(x));
    }
    try {
        return new Engine(model_dir, devs, max_t, w);
    } catch (std::runtime_error& ex) {  // a refused steering vector (QE_STEER / QE_STEER_DIR): no engine
        fprintf(stderr, "qengine: %s\n", ex.what());
        return nullptr;
    }
}
void qe_destroy(void* e) { delete (Engine*)e; }
void qe_reset(void* e) { ((Engine*)e)->reset(); }
int qe_prefill(void* e, const int* ids, int n) { return ((Engine*)e)->prefill(ids, n); }
int qe_decode(void* e, int tok) { return ((Engine*)e)->decode(tok); }
void qe_debug_hidden(void* e, const int* ids, int n, float* out) {
    ((Engine*)e)->debug_hidden(ids, n, out);
}
void qe_logits(void* e, float* out, int m) { ((Engine*)e)->logits(out, m); }
int qe_start(void* e, const int* ids, int n, int k) { return ((Engine*)e)->start(ids, n, k); }
int qe_start2(void* e, const int* ids, int n, int k, unsigned long long affinity) {
    return ((Engine*)e)->start(ids, n, k, affinity);
}
int qe_start3(void* e, const int* ids, int n, int k, unsigned long long affinity, int hint) {
    return ((Engine*)e)->start(ids, n, k, affinity, hint);
}
int qe_start_mm(void* e, const int* ids, int n, int k, unsigned long long affinity, int hint,
                const void* img, int img_rows, const int* rope3, int delta) {
    return ((Engine*)e)->start_mm(ids, n, k, affinity, hint, (const bf16*)img, img_rows, rope3, delta);
}
int qe_start_begin(void* e, const int* ids, int n, int k, unsigned long long affinity, int hint,
                   const void* img, int img_rows, const int* rope3, int delta) {
    return ((Engine*)e)->start_begin(ids, n, k, affinity, hint, (const bf16*)img, img_rows, rope3, delta);
}
int qe_prefill_more(void* e, int budget) { return ((Engine*)e)->prefill_more(budget); }
int qe_start_finish(void* e) { return ((Engine*)e)->start_finish(); }
int qe_prefill_more_slot(void* e, int slot, int budget) { return ((Engine*)e)->prefill_more_slot(slot, budget); }
int qe_start_finish_slot(void* e, int slot) { return ((Engine*)e)->start_finish_slot(slot); }
int qe_num_prefills(void* e) { return ((Engine*)e)->num_prefills(); }
int qe_prefill_pos(void* e, int slot) { return ((Engine*)e)->prefill_pos(slot); }
int qe_archive_more(void* e, long long budget) { return ((Engine*)e)->archive_more((size_t)budget); }
void qe_set_lookahead(void* e, int on) { ((Engine*)e)->set_lookahead(on != 0); }
int qe_last_slot(void* e) { return ((Engine*)e)->last_slot(); }
int qe_num_slots(void* e) { return ((Engine*)e)->num_slots(); }
int qe_slot_capacity(void* e, int i) { return ((Engine*)e)->slot_capacity(i); }
int qe_max_context(void* e) { return ((Engine*)e)->max_context(); }
long long qe_pool_tokens(void* e) { return (long long)((Engine*)e)->pool_tokens(); }
int qe_can_admit(void* e, int n) { return ((Engine*)e)->can_admit(n); }
int qe_pause_for(void* e, int n, int prio) { return ((Engine*)e)->pause_for(n, prio) ? 1 : 0; }
void qe_set_prio(void* e, int slot, int prio) { ((Engine*)e)->set_prio(slot, prio); }
void qe_set_limit(void* e, int slot, int tokens) { ((Engine*)e)->set_limit(slot, tokens); }
void qe_watch_stops(void* e, int slot, int on) { ((Engine*)e)->watch_stops(slot, on != 0); }
void qe_set_no_ahead(void* e, int slot, int on) { ((Engine*)e)->set_no_ahead(slot, on != 0); }
int qe_drafts(void* e, int slot, int* out) { return ((Engine*)e)->drafts(slot, out); }
int qe_trim(void* e, int slot, int pos) { return ((Engine*)e)->trim(slot, pos) ? 1 : 0; }
int qe_probe_cached(void* e, const int* ids, int n) { return ((Engine*)e)->probe_cached(ids, n); }
int qe_num_pinned(void* e) { return ((Engine*)e)->num_pinned(); }
void qe_slot_info(void* e, int i, long long* out) { ((Engine*)e)->slot_info(i, out); }
void qe_archive_info(void* e, double* out) { ((Engine*)e)->archive_info(out); }
void qe_pool_counters(void* e, long long* out) { ((Engine*)e)->pool_counters(out); }
int qe_spec_step(void* e, int* out) { return ((Engine*)e)->spec_step(out); }
void qe_draft_stats(void* e, int slot, long long* out) { ((Engine*)e)->draft_stats(slot, out); }
void qe_finish(void* e) { ((Engine*)e)->finish(); }
double qe_warmup(void* e, int modes) { return ((Engine*)e)->warmup(modes); }
int qe_last_copied(void* e) { return ((Engine*)e)->last_copied(); }
int qe_last_from_archive(void* e) { return ((Engine*)e)->last_from_archive(); }
int qe_num_archives(void* e) { return ((Engine*)e)->num_archives(); }
int qe_num_disk_archives(void* e) { return ((Engine*)e)->num_disk_archives(); }
void qe_persist(void* e) { ((Engine*)e)->persist(); }
void qe_set_dump_hidden(void* e, uint16_t* out) { ((Engine*)e)->set_dump_hidden(out); }
void qe_release(void* e, int slot) { ((Engine*)e)->release(slot); }
void qe_resume(void* e, int slot) { ((Engine*)e)->resume(slot); }
void qe_set_concurrent(void* e, int on) { ((Engine*)e)->set_concurrent(on); }
void qe_step_batch(void* e, const int* slots, int n, int* out, int* cnt) {
    ((Engine*)e)->step_batch(slots, n, out, cnt);
}
void qe_set_sampling(void* e, float temperature, float top_p, int top_k, float min_p,
                     unsigned long long seed) {
    ((Engine*)e)->set_sampling(temperature, top_p, top_k, min_p, seed);
}
unsigned* qe_mask_buffer(void* e, int slot) { return ((Engine*)e)->mask_buffer(slot); }
int* qe_spec_io(void* e, int slot) { return ((Engine*)e)->spec_io(slot); }
void qe_constrain(void* e, int slot, int on) { ((Engine*)e)->constrain(slot, on != 0); }
int qe_constrained(void* e, int slot) { return ((Engine*)e)->constrained(slot) ? 1 : 0; }
int qe_last_reused(void* e) { return ((Engine*)e)->last_reused(); }
int qe_vocab(void* e) { return ((Engine*)e)->config().vocab; }
int qe_layers(void* e) { return ((Engine*)e)->config().layers; }
// activation steering (QE_STEER / QE_STEER_DIR / QE_STEER_DEFAULT, src/steer.h; server/qe_capi.h tier 10): the next
// start's vector (index in qe_steer_info, -1 off, -2 the server default), scale, mode (-1 / 0 add / 1 project) and
// layers (-1, -1: the mode's default); sticky. Returns 0, or -1 for a bad value (unchanged).
int qe_set_steering_ex(void* e, int vec, float scale, int mode, int lo, int hi) {
    return ((Engine*)e)->set_steering(vec, scale, mode, lo, hi);
}
int qe_set_steering(void* e, int vec, float scale) { return ((Engine*)e)->set_steering(vec, scale, -1, -1, -1); }
const char* qe_steer_info(void* e) { return ((Engine*)e)->steer_info(); }
// mean layer-output residual over positions [r0, n) of an unsteered prefill from position 0 -> out[layers][hidden]
int qe_capture(void* e, const int* ids, int n, int r0, float* out) { return ((Engine*)e)->capture(ids, n, r0, out); }
}

// ---- comm microbenchmark ----
#include <chrono>
#include <cmath>
#include <thread>

extern "C" double qe_bench_allreduce2(const char* devices, int n, int iters, int ll) {
    std::vector<int> devs;
    std::stringstream ss(devices);
    for (std::string x; std::getline(ss, x, ',');) devs.push_back(std::stoi(x));
    const int ng = devs.size();
    Comm c = comm_create(ng, n, 5120);
    comm_enable_p2p(c, devs.data());
    std::vector<double> us(ng);
    std::vector<std::thread> th;
    for (int r = 0; r < ng; r++)
        th.emplace_back([&, r] {
            CUDA_CHECK(cudaSetDevice(devs[r]));
            cudaStream_t st;
            CUDA_CHECK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
            float *resid, *part;
            CommCounters ctr;
            CUDA_CHECK(cudaMalloc(&resid, n * 4));
            CUDA_CHECK(cudaMalloc(&part, n * 4));
            CUDA_CHECK(cudaMalloc(&ctr.seq, 4));
            CUDA_CHECK(cudaMalloc(&ctr.done, 8));
            CUDA_CHECK(cudaMalloc(&ctr.go, 4));
            CUDA_CHECK(cudaMalloc(&ctr.bar, 8));
            CUDA_CHECK(cudaMemset(ctr.bar, 0, 8));
            CUDA_CHECK(cudaMemset(ctr.seq, 0, 4));
            CUDA_CHECK(cudaMemset(ctr.done, 0, 8));
            CUDA_CHECK(cudaMemset(ctr.go, 0, 4));
            CUDA_CHECK(cudaMalloc(&ctr.ss, kCommBlocks * kMaxBatchRows * 4));
            CUDA_CHECK(cudaMalloc(&ctr.epoch, 4));
            CUDA_CHECK(cudaMemset(ctr.epoch, 0, 4));
            bf16* xn;
            bf16* nw;
            CUDA_CHECK(cudaMalloc(&xn, n * 2));
            CUDA_CHECK(cudaMalloc(&nw, 5120 * 2));
            CUDA_CHECK(cudaMemset(nw, 0, 5120 * 2));
            CUDA_CHECK(cudaMemset(part, 0, n * 4));
            __half* xh = nullptr;   // QE_BENCH_XH=1: also write the XH copy (gemv_h16.cuh) like the engine
            if (getenv("QE_BENCH_XH") && atoi(getenv("QE_BENCH_XH"))) CUDA_CHECK(cudaMalloc(&xh, n * 2));
            const int xhmode = getenv("QE_BENCH_XH") ? atoi(getenv("QE_BENCH_XH")) : 0;  // 2: XH only (no bf16 xn)
            // capture 100 back-to-back collectives in a graph to remove launch overhead
            cudaGraph_t g;
            cudaGraphExec_t ge;
            CUDA_CHECK(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal));
            comm_begin_unit(ctr, st);
            for (int i = 0; i < 100; i++)
                allreduce_add(resid, part, n / 5120, 5120, c, r, ctr, st, nw, 1e-6f, xhmode == 2 ? nullptr : xn, ll ? i : -1, false, xh);
            CUDA_CHECK(cudaStreamEndCapture(st, &g));
            CUDA_CHECK(cudaGraphInstantiate(&ge, g, 0));
            CUDA_CHECK(cudaGraphLaunch(ge, st));
            CUDA_CHECK(cudaStreamSynchronize(st));
            auto t0 = std::chrono::steady_clock::now();
            for (int i = 0; i < iters; i++) CUDA_CHECK(cudaGraphLaunch(ge, st));
            CUDA_CHECK(cudaStreamSynchronize(st));
            us[r] = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0)
                        .count() / (iters * 100.0);
        });
    for (auto& t : th) t.join();
    double mx = 0;
    for (double u : us) mx = std::max(mx, u);
    return mx;
}
extern "C" double qe_bench_allreduce(const char* devices, int n, int iters) {
    return qe_bench_allreduce2(devices, n, iters, 0);
}

// Microbenchmark: decode attention (one GPU, 2 kv heads x 4 q heads, fp8 cache) over ctx keys
// with m query rows; returns us per call.
extern "C" double qe_bench_attn(int ctx, int m, int iters, int fp8) {
    AttnShape s{};
    s.nq = 8; s.nkv = 2;
    // the real per-GPU grouping: 6 + 2 q heads (24 q heads / 4 kv heads split over 3 GPUs)
    // QE_BENCH_G0=4: GPU1's 4 + 4
    const int g0 = getenv("QE_BENCH_G0") ? atoi(getenv("QE_BENCH_G0")) : 6;
    s.kv_qb[0] = 0; s.kv_qe[0] = g0; s.kv_qb[1] = g0; s.kv_qe[1] = 8;
    for (int h = 0; h < 8; h++) s.q2kv[h] = h < g0 ? 0 : 1;
    s.max_t = ctx + 64; s.fp8 = fp8; s.ks = 0.02f; s.vs = 0.02f;
    const size_t esz = fp8 ? 1 : 2;
    void *kc, *vc;
    CUDA_CHECK(cudaMalloc(&kc, (size_t)2 * s.max_t * 256 * esz));
    CUDA_CHECK(cudaMalloc(&vc, (size_t)2 * s.max_t * 256 * esz));
    CUDA_CHECK(cudaMemset(kc, 0x31, (size_t)2 * s.max_t * 256 * esz));
    CUDA_CHECK(cudaMemset(vc, 0x29, (size_t)2 * s.max_t * 256 * esz));
    bf16* q; float* proj; float* work; bf16* out; int* pos;
    CUDA_CHECK(cudaMalloc(&q, (size_t)m * 8 * 256 * 2));
    CUDA_CHECK(cudaMemset(q, 0x3c, (size_t)m * 8 * 256 * 2));
    CUDA_CHECK(cudaMalloc(&proj, (size_t)m * 8 * 512 * 4));
    CUDA_CHECK(cudaMemset(proj, 0, (size_t)m * 8 * 512 * 4));
    const size_t wf = attn_work_floats(s);
    CUDA_CHECK(cudaMalloc(&work, wf * 4));
    CUDA_CHECK(cudaMalloc(&out, (size_t)m * 8 * 256 * 2));
    CUDA_CHECK(cudaMalloc(&pos, 4));
    const int p0 = ctx - m;
    CUDA_CHECK(cudaMemcpy(pos, &p0, 4, cudaMemcpyHostToDevice));
    attention_prefill_init();
    attention_decode_tc_init();
    for (int i = 0; i < 3; i++) attention(q, kc, vc, s, proj, 8 * 512, pos, m, work, out, 0);
    cudaEvent_t a, b;
    cudaEventCreate(&a); cudaEventCreate(&b);
    cudaEventRecord(a);
    for (int i = 0; i < iters; i++) attention(q, kc, vc, s, proj, 8 * 512, pos, m, work, out, 0);
    cudaEventRecord(b);
    cudaEventSynchronize(b);
    float ms;
    cudaEventElapsedTime(&ms, a, b);
    cudaFree(kc); cudaFree(vc); cudaFree(q); cudaFree(proj); cudaFree(work); cudaFree(out); cudaFree(pos);
    return ms * 1e3 / iters;
}


// GATE microbenchmark: batched fp8 decode attention (attention_seg: prep + attn_dec8d_seg + combine)
// for nseg sequences of ctx keys with m rows each; per-GPU shapes as qe_bench_attn. cmin = min keys
// per split (QE_DEC_CHUNK_MIN value: -1 = default 38-split rule, 0 = old L/82). Returns us per call of the full attention_seg; *us_attn = dec8_seg only.
extern "C" double qe_bench_attn_seg(int nseg, int ctx, int m, int iters, int cmin, double* us_attn) {
    AttnShape s{};
    s.nq = 8; s.nkv = 2;
    const int g0 = getenv("QE_BENCH_G0") ? atoi(getenv("QE_BENCH_G0")) : 6;
    s.kv_qb[0] = 0; s.kv_qe[0] = g0; s.kv_qb[1] = g0; s.kv_qe[1] = 8;
    for (int h = 0; h < 8; h++) s.q2kv[h] = h < g0 ? 0 : 1;
    s.max_t = ctx + 64; s.fp8 = 1; s.ks = 0.02f; s.vs = 0.02f; s.dchunk_min = cmin;
    const int ld = 8 * 512 + 2 * 512;
    const size_t kvb = (size_t)2 * s.max_t * 256;
    const size_t wf = attn_work_floats(s);
    AttnSegs g{};
    g.n = nseg; g.max_m = m;
    std::vector<void*> bufs;
    auto al = [&](size_t b, int v) { void* p; CUDA_CHECK(cudaMalloc(&p, b)); CUDA_CHECK(cudaMemset(p, v, b)); bufs.push_back(p); return p; };
    bf16* qn_w = (bf16*)al(512, 0); bf16* kn_w = (bf16*)al(512, 0); float* invf = (float*)al(1024, 0);
    for (int i = 0; i < nseg; i++) {
        AttnSeg& c = g.c[i];
        c.proj = (float*)al((size_t)m * ld * 4, 0);
        c.q = (bf16*)al((size_t)m * 8 * 256 * 2, 0x3c);
        c.kc = al(kvb, 0x31); c.vc = al(kvb, 0x29);
        const int p0 = ctx - m;
        int* pos = (int*)al(4, 0);
        CUDA_CHECK(cudaMemcpy(pos, &p0, 4, cudaMemcpyHostToDevice));
        c.pos = pos; c.rdelta = nullptr; c.m = m; c.max_t = s.max_t;
        c.work = (float*)al(wf * 4, 0);
        c.out = (bf16*)al((size_t)m * 8 * 256 * 2, 0);
        c.xh = nullptr;
    }
    attention_prefill_init();
    attention_decode_tc_init();
    const int G = g0 * m;
    for (int i = 0; i < 3; i++) attention_seg(g, s, ld, qn_w, kn_w, 1e-6f, invf, 0);
    cudaEvent_t a, b;
    cudaEventCreate(&a); cudaEventCreate(&b);
    float ms1, ms2;
    cudaEventRecord(a);
    for (int i = 0; i < iters; i++) attention_seg(g, s, ld, qn_w, kn_w, 1e-6f, invf, 0);
    cudaEventRecord(b); cudaEventSynchronize(b); cudaEventElapsedTime(&ms1, a, b);
    cudaEventRecord(a);
    for (int i = 0; i < iters; i++) attention_dec8_seg(g, s, G, 0);
    cudaEventRecord(b); cudaEventSynchronize(b); cudaEventElapsedTime(&ms2, a, b);
    CUDA_CHECK(cudaGetLastError());
    for (void* p : bufs) cudaFree(p);
    if (us_attn) *us_attn = ms2 * 1e3 / iters;
    return ms1 * 1e3 / iters;
}

// Microbenchmark: DeltaNet recurrence (dn_core incl. dn_qk) over m tokens on one GPU with the
// per-GPU shapes (16 v heads, 6 k heads); returns us per call.
// Chunked vs sequential delta rule on random inputs: max |o| error relative to max |o|, and the
// same for the final state (returns o error; *serr receives the state error).
extern "C" double qe_test_dn_chunk(int m, int seed, double* serr) {
    DeltaShape s{};
    s.nk = 6; s.nv = 16;
    for (int h = 0; h < 16; h++) s.v2k[h] = h * 6 / 16;
    s.ab_pad = 16; s.ab_splits = 1; s.ab_rows = m;
    const int ch = 2 * s.nk * 128 + s.nv * 128;
    const size_t slot = (size_t)s.nv * 128 * 128;
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::uniform_real_distribution<float> ud(0.f, 1.f);
    std::vector<float> hcv((size_t)m * ch), hab((size_t)m * 2 * s.ab_pad), hA(64), hdt(64), hS(2 * slot);
    for (auto& x : hcv) x = nd(rng);
    for (auto& x : hab) x = nd(rng);
    for (auto& x : hA) x = -(ud(rng) * 2.f + 0.05f);  // A = -exp(A_log) < 0
    for (int h = 0; h < 16; h += 5) hA[h] = -80.f;     // decays that underflow to 0 (real data)
    for (auto& x : hdt) x = ud(rng) * 2.f - 1.f;
    for (auto& x : hS) x = nd(rng) * 0.05f;
    float *cv, *ab, *A, *dt, *S1, *S2, *qk, *gb, *o1, *o2;
    int* cur;
    auto up = [](float** d, const std::vector<float>& h) {
        CUDA_CHECK(cudaMalloc(d, h.size() * 4));
        CUDA_CHECK(cudaMemcpy(*d, h.data(), h.size() * 4, cudaMemcpyHostToDevice));
    };
    up(&cv, hcv); up(&ab, hab); up(&A, hA); up(&dt, hdt); up(&S1, hS); up(&S2, hS);
    CUDA_CHECK(cudaMalloc(&qk, (size_t)m * 2 * s.nk * 128 * 4));
    CUDA_CHECK(cudaMalloc(&gb, (size_t)m * 2 * s.nv * 4));
    CUDA_CHECK(cudaMalloc(&o1, (size_t)m * s.nv * 128 * 4));
    CUDA_CHECK(cudaMalloc(&o2, (size_t)m * s.nv * 128 * 4));
    CUDA_CHECK(cudaMalloc(&cur, 4));
    CUDA_CHECK(cudaMemset(cur, 0, 4));
    dn_chunk_init();
    dn_core_seq(cv, ch, s, ab, A, dt, S1, slot, cur, m, qk, gb, o1, 0);  // reference recurrence
    dn_core_chunked(cv, ch, s, qk, gb, S2, slot, cur, m, o2, 0, 0);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<float> a((size_t)m * s.nv * 128), b(a.size()), sa(slot), sb(slot);
    CUDA_CHECK(cudaMemcpy(a.data(), o1, a.size() * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(b.data(), o2, b.size() * 4, cudaMemcpyDeviceToHost));
    const int dst = m % 2;
    CUDA_CHECK(cudaMemcpy(sa.data(), S1 + dst * slot, slot * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(sb.data(), S2 + dst * slot, slot * 4, cudaMemcpyDeviceToHost));
    auto rel = [](const std::vector<float>& x, const std::vector<float>& y) {
        double mx = 0, e = 0;
        for (size_t i = 0; i < x.size(); i++) {
            if (!std::isfinite(x[i]) || !std::isfinite(y[i])) return (double)INFINITY;  // NaN must fail
            mx = std::max(mx, (double)std::fabs(x[i]));
            e = std::max(e, (double)std::fabs(x[i] - y[i]));
        }
        return e / mx;
    };
    auto amax = [](const std::vector<float>& x) {
        double mx = 0; size_t nan = 0;
        for (float v : x) { if (std::isnan(v)) nan++; else mx = std::max(mx, (double)std::fabs(v)); }
        return std::make_pair(mx, nan);
    };
    if (getenv("QE_DN_DEBUG")) {
        auto p1 = amax(a), p2 = amax(b), p3 = amax(sa), p4 = amax(sb);
        fprintf(stderr, "o seq max %g nan %zu | o chunk max %g nan %zu | S seq %g/%zu chunk %g/%zu\n",
                p1.first, p1.second, p2.first, p2.second, p3.first, p3.second, p4.first, p4.second);
    }
    *serr = rel(sa, sb);
    const double oe = rel(a, b);
    for (float* p : {cv, ab, A, dt, S1, S2, qk, gb, o1, o2}) cudaFree(p);
    cudaFree(cur);
    return oe;
}

extern "C" double qe_bench_dn(int m, int iters, int chunked) {
    DeltaShape s{};
    s.nk = 6; s.nv = 16;
    for (int h = 0; h < 16; h++) s.v2k[h] = h * 6 / 16;
    s.ab_pad = 16; s.ab_splits = 1; s.ab_rows = m;
    const int ch = 2 * s.nk * 128 + s.nv * 128;
    float *cv, *ab, *A, *dt, *S, *qk, *gb, *o;
    int* cur;
    CUDA_CHECK(cudaMalloc(&cv, (size_t)m * ch * 4));
    CUDA_CHECK(cudaMalloc(&ab, (size_t)m * 2 * s.ab_pad * 4));
    CUDA_CHECK(cudaMalloc(&A, 64 * 4));
    CUDA_CHECK(cudaMalloc(&dt, 64 * 4));
    const size_t slot = (size_t)s.nv * 128 * 128;
    CUDA_CHECK(cudaMalloc(&S, 2 * slot * 4));
    CUDA_CHECK(cudaMalloc(&qk, (size_t)m * 2 * s.nk * 128 * 4));
    CUDA_CHECK(cudaMalloc(&gb, (size_t)m * 2 * s.nv * 4));
    CUDA_CHECK(cudaMalloc(&o, (size_t)m * s.nv * 128 * 4));
    CUDA_CHECK(cudaMalloc(&cur, 4));
    CUDA_CHECK(cudaMemset(cv, 0x3c, (size_t)m * ch * 4));  // ~0.01
    CUDA_CHECK(cudaMemset(ab, 0, (size_t)m * 2 * s.ab_pad * 4));
    CUDA_CHECK(cudaMemset(A, 0xbf, 64 * 4));
    CUDA_CHECK(cudaMemset(dt, 0, 64 * 4));
    CUDA_CHECK(cudaMemset(S, 0, 2 * slot * 4));
    CUDA_CHECK(cudaMemset(cur, 0, 4));
    if (chunked) { dn_chunk_init(); setenv("QE_DN_CHUNK", "1", 0); }
    for (int i = 0; i < 2; i++)
        dn_core(cv, ch, s, ab, A, dt, S, slot, cur, m, false, qk, gb, o, 0, 0);
    cudaEvent_t a, b;
    cudaEventCreate(&a); cudaEventCreate(&b);
    cudaEventRecord(a);
    for (int i = 0; i < iters; i++) dn_core(cv, ch, s, ab, A, dt, S, slot, cur, m, false, qk, gb, o, 0, 0);
    cudaEventRecord(b);
    cudaEventSynchronize(b);
    float ms;
    cudaEventElapsedTime(&ms, a, b);
    cudaFree(cv); cudaFree(ab); cudaFree(A); cudaFree(dt); cudaFree(S); cudaFree(qk); cudaFree(gb);
    cudaFree(o); cudaFree(cur);
    return ms * 1e3 / iters;
}

// Correctness check of the host all-reduce (rows / RS rows): partial[r][i] = (rank + 1) * f(r, i),
// resid = 0; returns the max abs error of resid vs the exact sum on GPU 0.
extern "C" double qe_test_allreduce(const char* devices, int m, int overlapped) {
    std::vector<int> devs;
    std::stringstream ss(devices);
    for (std::string x; std::getline(ss, x, ',');) devs.push_back(std::stoi(x));
    const int ng = devs.size(), h = 5120;
    Comm c = comm_create(ng, std::max(m, 2048) * h, h);
    if (comm_enable_p2p(c, devs.data())) printf("qe_test_allreduce: P2P path (m=%d)\n", m);
    std::vector<double> err(ng, 0.0);
    std::vector<std::thread> th;
    for (int r = 0; r < ng; r++)
        th.emplace_back([&, r] {
            CUDA_CHECK(cudaSetDevice(devs[r]));
            cudaStream_t st;
            CUDA_CHECK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
            float *resid, *part;
            CommCounters ctr;
            CUDA_CHECK(cudaMalloc(&resid, (size_t)m * h * 4));
            CUDA_CHECK(cudaMalloc(&part, (size_t)m * h * 4));
            CUDA_CHECK(cudaMalloc(&ctr.seq, 4)); CUDA_CHECK(cudaMemset(ctr.seq, 0, 4));
            CUDA_CHECK(cudaMalloc(&ctr.done, 8)); CUDA_CHECK(cudaMemset(ctr.done, 0, 8));
            CUDA_CHECK(cudaMalloc(&ctr.go, 4)); CUDA_CHECK(cudaMemset(ctr.go, 0, 4));
            CUDA_CHECK(cudaMalloc(&ctr.bar, 8)); CUDA_CHECK(cudaMemset(ctr.bar, 0, 8));
            CUDA_CHECK(cudaMalloc(&ctr.ss, kCommBlocks * kMaxBatchRows * 4));
            CUDA_CHECK(cudaMalloc(&ctr.epoch, 4)); CUDA_CHECK(cudaMemset(ctr.epoch, 0, 4));
            std::vector<float> hp((size_t)m * h);
            for (int i = 0; i < m; i++)
                for (int j = 0; j < h; j++) hp[(size_t)i * h + j] = (r + 1) * (float)((i * 7 + j % 13) % 17 - 8);
            CUDA_CHECK(cudaMemcpy(part, hp.data(), hp.size() * 4, cudaMemcpyHostToDevice));
            // QE_AR_REPS collectives accumulated into resid: an error in any of them stays visible
            for (int rep = 0; rep < 3; rep++) {
                CUDA_CHECK(cudaMemset(resid, 0, (size_t)m * h * 4));
                allreduce_add(resid, part, m, h, c, r, ctr, st, nullptr, 0.f, nullptr, -1, overlapped != 0);
                CUDA_CHECK(cudaStreamSynchronize(st));
            }
            std::vector<float> out((size_t)m * h);
            CUDA_CHECK(cudaMemcpy(out.data(), resid, out.size() * 4, cudaMemcpyDeviceToHost));
            double e = 0;
            const float tot = ng * (ng + 1) / 2;
            for (int i = 0; i < m; i++)
                for (int j = 0; j < h; j++)
                    e = std::max(e, (double)std::fabs(out[(size_t)i * h + j] - tot * ((i * 7 + j % 13) % 17 - 8)));
            err[r] = e;
        });
    for (auto& t : th) t.join();
    double e = 0;
    for (double x : err) e = std::max(e, x);
    return e;
}

// Stress test of the LL all-reduce paths (decode: m <= 16 rows): reps collectives, the partials
// cycle through 3 different buffers (the protocol's slots cycle through 2, so stale or torn data
// shows up in the result); returns the max abs error of the accumulated residual.
extern "C" double qe_test_ll(const char* devices, int m, int reps) {
    std::vector<int> devs;
    std::stringstream ss(devices);
    for (std::string x; std::getline(ss, x, ',');) devs.push_back(std::stoi(x));
    const int ng = devs.size(), h = 5120;
    Comm c = comm_create(ng, kMaxBatchRows * h, h);
    if (comm_enable_p2p(c, devs.data())) printf("qe_test_ll: P2P path\n");
    std::vector<double> err(ng, 0.0);
    std::vector<std::thread> th;
    auto val = [](int r, int b, int i, int j) { return (r + 1) * (float)((i * 7 + j % 13 + 5 * b) % 17 - 8); };
    for (int r = 0; r < ng; r++)
        th.emplace_back([&, r] {
            CUDA_CHECK(cudaSetDevice(devs[r]));
            cudaStream_t st;
            CUDA_CHECK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
            float *resid, *part[3];
            CommCounters ctr;
            CUDA_CHECK(cudaMalloc(&resid, (size_t)m * h * 4));
            CUDA_CHECK(cudaMemset(resid, 0, (size_t)m * h * 4));
            for (int b = 0; b < 3; b++) {
                CUDA_CHECK(cudaMalloc(&part[b], (size_t)m * h * 4));
                std::vector<float> hp((size_t)m * h);
                for (int i = 0; i < m; i++)
                    for (int j = 0; j < h; j++) hp[(size_t)i * h + j] = val(r, b, i, j);
                CUDA_CHECK(cudaMemcpy(part[b], hp.data(), hp.size() * 4, cudaMemcpyHostToDevice));
            }
            CUDA_CHECK(cudaMalloc(&ctr.seq, 4)); CUDA_CHECK(cudaMemset(ctr.seq, 0, 4));
            CUDA_CHECK(cudaMalloc(&ctr.done, 8)); CUDA_CHECK(cudaMemset(ctr.done, 0, 8));
            CUDA_CHECK(cudaMalloc(&ctr.go, 4)); CUDA_CHECK(cudaMemset(ctr.go, 0, 4));
            CUDA_CHECK(cudaMalloc(&ctr.bar, 8)); CUDA_CHECK(cudaMemset(ctr.bar, 0, 8));
            CUDA_CHECK(cudaMalloc(&ctr.ss, kCommBlocks * kMaxBatchRows * 4));
            CUDA_CHECK(cudaMalloc(&ctr.epoch, 4)); CUDA_CHECK(cudaMemset(ctr.epoch, 0, 4));
            for (int rep = 0; rep < reps; rep++) {
                if (rep % 64 == 0) comm_begin_unit(ctr, st);  // as in the model: epoch per unit
                allreduce_add(resid, part[rep % 3], m, h, c, r, ctr, st, nullptr, 0.f, nullptr,
                              rep % 64, false);
            }
            CUDA_CHECK(cudaStreamSynchronize(st));
            std::vector<float> out((size_t)m * h);
            CUDA_CHECK(cudaMemcpy(out.data(), resid, out.size() * 4, cudaMemcpyDeviceToHost));
            double e = 0;
            const float tot = ng * (ng + 1) / 2;
            for (int i = 0; i < m; i++)
                for (int j = 0; j < h; j++) {
                    double ref = 0;
                    for (int b = 0; b < 3; b++) ref += (double)((reps - b + 2) / 3) * tot * val(0, b, i, j);
                    e = std::max(e, std::fabs(out[(size_t)i * h + j] - ref));
                }
            err[r] = e;
        });
    for (auto& t : th) t.join();
    double e = 0;
    for (double x : err) e = std::max(e, x);
    return e;
}

// Bit-identity check of the decode LL all-reduce protocols: reps collectives
// with random partials (seeded, different per GPU and rep) and the fused norm; copies every GPU's
// final residual and normed rows (as float) to out[ng][2][m][h]. Run once per protocol setting
// (QE_AR_RSUM / QE_AR_RS_MIN / QE_LL16 are read once per process) and compare the dumps.
extern "C" void qe_test_ll_dump(const char* devices, int m, int reps, unsigned seed, float* out) {
    std::vector<int> devs;
    std::stringstream ss(devices);
    for (std::string x; std::getline(ss, x, ',');) devs.push_back(std::stoi(x));
    const int ng = devs.size(), h = 5120;
    Comm c = comm_create(ng, kMaxBatchRows * h, h);
    comm_enable_p2p(c, devs.data());
    std::vector<std::thread> th;
    for (int r = 0; r < ng; r++)
        th.emplace_back([&, r] {
            CUDA_CHECK(cudaSetDevice(devs[r]));
            cudaStream_t st;
            CUDA_CHECK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
            float *resid, *part[3];
            bf16 *xn, *nw;
            CommCounters ctr;
            CUDA_CHECK(cudaMalloc(&resid, (size_t)m * h * 4));
            CUDA_CHECK(cudaMalloc(&xn, (size_t)m * h * 2));
            CUDA_CHECK(cudaMalloc(&nw, h * 2));
            std::mt19937 rng(seed * 7919u + r * 104729u);
            std::normal_distribution<float> nd(0.f, 1.f);
            {
                std::vector<float> hr((size_t)m * h);
                std::mt19937 rr(seed);  // identical starting residual on every GPU
                for (auto& v : hr) v = 4.f * nd(rr);
                CUDA_CHECK(cudaMemcpy(resid, hr.data(), hr.size() * 4, cudaMemcpyHostToDevice));
                std::vector<bf16> hw(h);
                for (auto& v : hw) v = __float2bfloat16(0.1f * nd(rr));
                CUDA_CHECK(cudaMemcpy(nw, hw.data(), h * 2, cudaMemcpyHostToDevice));
            }
            for (int b = 0; b < 3; b++) {
                CUDA_CHECK(cudaMalloc(&part[b], (size_t)m * h * 4));
                std::vector<float> hp((size_t)m * h);
                for (auto& v : hp) v = nd(rng) * (1.f + 3.f * (rng() % 64 == 0));
                CUDA_CHECK(cudaMemcpy(part[b], hp.data(), hp.size() * 4, cudaMemcpyHostToDevice));
            }
            CUDA_CHECK(cudaMalloc(&ctr.seq, 4)); CUDA_CHECK(cudaMemset(ctr.seq, 0, 4));
            CUDA_CHECK(cudaMalloc(&ctr.done, 8)); CUDA_CHECK(cudaMemset(ctr.done, 0, 8));
            CUDA_CHECK(cudaMalloc(&ctr.go, 4)); CUDA_CHECK(cudaMemset(ctr.go, 0, 4));
            CUDA_CHECK(cudaMalloc(&ctr.bar, 8)); CUDA_CHECK(cudaMemset(ctr.bar, 0, 8));
            CUDA_CHECK(cudaMalloc(&ctr.ss, kCommBlocks * kMaxBatchRows * 4));
            CUDA_CHECK(cudaMalloc(&ctr.epoch, 4)); CUDA_CHECK(cudaMemset(ctr.epoch, 0, 4));
            for (int rep = 0; rep < reps; rep++) {
                if (rep % 64 == 0) comm_begin_unit(ctr, st);
                allreduce_add(resid, part[rep % 3], m, h, c, r, ctr, st, nw, 1e-6f, xn, rep % 64, false);
            }
            CUDA_CHECK(cudaStreamSynchronize(st));
            float* o = out + (size_t)r * 2 * m * h;
            CUDA_CHECK(cudaMemcpy(o, resid, (size_t)m * h * 4, cudaMemcpyDeviceToHost));
            std::vector<bf16> hx((size_t)m * h);
            CUDA_CHECK(cudaMemcpy(hx.data(), xn, hx.size() * 2, cudaMemcpyDeviceToHost));
            for (size_t i = 0; i < hx.size(); i++) o[(size_t)m * h + i] = __bfloat162float(hx[i]);
        });
    for (auto& t : th) t.join();
}

// Reproduction of LL16 tag aliasing: rows hi rows of a collective at (epoch 1,
// idx 2) are left unwritten while `lo`-row collectives run for 127 units; the next hi-row collective
// at (epoch 129, idx 0) uses the same slot and its tag equals the stale words' tag
// (129*1024 - (1*1024 + 2) = 2 * 65535). GPU2 starts that collective `delay_ms` late, so the others
// poll its stale words first. Partials: rank r sends (r + 1) * 1 before and (r + 1) * 2 in the final
// collective (sum 12, exact); returns the max abs error of the final sums on GPUs 0..2 (rows lo..hi).
extern "C" double qe_test_ll_alias(const char* devices, int hi, int lo, int delay_ms) {
    std::vector<int> devs;
    std::stringstream ss(devices);
    for (std::string x; std::getline(ss, x, ',');) devs.push_back(std::stoi(x));
    const int ng = devs.size(), h = 5120;
    Comm c = comm_create(ng, kMaxBatchRows * h, h);
    comm_enable_p2p(c, devs.data());
    std::vector<double> err(ng, 0.0);
    std::vector<std::thread> th;
    for (int r = 0; r < ng; r++)
        th.emplace_back([&, r] {
            CUDA_CHECK(cudaSetDevice(devs[r]));
            cudaStream_t st;
            CUDA_CHECK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
            float *resid, *pa, *pb;
            CommCounters ctr;
            CUDA_CHECK(cudaMalloc(&resid, (size_t)hi * h * 4));
            CUDA_CHECK(cudaMalloc(&pa, (size_t)hi * h * 4));
            CUDA_CHECK(cudaMalloc(&pb, (size_t)hi * h * 4));
            std::vector<float> v((size_t)hi * h, (float)(r + 1));
            CUDA_CHECK(cudaMemcpy(pa, v.data(), v.size() * 4, cudaMemcpyHostToDevice));
            for (auto& x : v) x *= 2.f;
            CUDA_CHECK(cudaMemcpy(pb, v.data(), v.size() * 4, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMalloc(&ctr.seq, 4)); CUDA_CHECK(cudaMemset(ctr.seq, 0, 4));
            CUDA_CHECK(cudaMalloc(&ctr.done, 8)); CUDA_CHECK(cudaMemset(ctr.done, 0, 8));
            CUDA_CHECK(cudaMalloc(&ctr.go, 4)); CUDA_CHECK(cudaMemset(ctr.go, 0, 4));
            CUDA_CHECK(cudaMalloc(&ctr.bar, 8)); CUDA_CHECK(cudaMemset(ctr.bar, 0, 8));
            CUDA_CHECK(cudaMalloc(&ctr.ss, kCommBlocks * kMaxBatchRows * 4));
            CUDA_CHECK(cudaMalloc(&ctr.epoch, 4)); CUDA_CHECK(cudaMemset(ctr.epoch, 0, 4));
            comm_begin_unit(ctr, st);  // epoch 1
            for (int i = 0; i < 3; i++) allreduce_add(resid, pa, hi, h, c, r, ctr, st, nullptr, 0.f, nullptr, i, false);
            for (int u = 0; u < 127; u++) {  // epochs 2..128: only rows < lo
                comm_begin_unit(ctr, st);
                allreduce_add(resid, pa, lo, h, c, r, ctr, st, nullptr, 0.f, nullptr, 0, false);
            }
            CUDA_CHECK(cudaStreamSynchronize(st));
            CUDA_CHECK(cudaMemset(resid, 0, (size_t)hi * h * 4));
            if (r == ng - 1) std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms));
            comm_begin_unit(ctr, st);  // epoch 129
            allreduce_add(resid, pb, hi, h, c, r, ctr, st, nullptr, 0.f, nullptr, 0, false);
            CUDA_CHECK(cudaStreamSynchronize(st));
            std::vector<float> out((size_t)hi * h);
            CUDA_CHECK(cudaMemcpy(out.data(), resid, out.size() * 4, cudaMemcpyDeviceToHost));
            double e = 0;
            for (size_t i = (size_t)lo * h; i < out.size(); i++) e = std::max(e, (double)std::fabs(out[i] - 12.f));
            err[r] = e;
        });
    for (auto& t : th) t.join();
    double e = 0;
    for (int r = 0; r < ng; r++) {
        printf("qe_test_ll_alias: GPU%d max err %.3g\n", r, err[r]);
        e = std::max(e, err[r]);
    }
    return e;
}

// Microbenchmark: prefill attention (one GPU, fp8 cache, real 6+2 grouping): m new queries after
// ctx cached keys; returns us per call.
extern "C" double qe_bench_pattn(int ctx, int m, int iters) {
    AttnShape s{};
    s.nq = 8; s.nkv = 2;
    s.kv_qb[0] = 0; s.kv_qe[0] = 6; s.kv_qb[1] = 6; s.kv_qe[1] = 8;
    for (int h = 0; h < 8; h++) s.q2kv[h] = h < 6 ? 0 : 1;
    s.max_t = ctx + m + 64; s.fp8 = 1; s.ks = 0.02f; s.vs = 0.02f;
    void *kc, *vc;
    CUDA_CHECK(cudaMalloc(&kc, (size_t)2 * s.max_t * 256));
    CUDA_CHECK(cudaMalloc(&vc, (size_t)2 * s.max_t * 256));
    CUDA_CHECK(cudaMemset(kc, 0x31, (size_t)2 * s.max_t * 256));
    CUDA_CHECK(cudaMemset(vc, 0x29, (size_t)2 * s.max_t * 256));
    bf16* q; float* proj; bf16* out; int* pos; float* work;
    CUDA_CHECK(cudaMalloc(&q, (size_t)m * 8 * 256 * 2));
    CUDA_CHECK(cudaMemset(q, 0x3c, (size_t)m * 8 * 256 * 2));
    CUDA_CHECK(cudaMalloc(&proj, (size_t)m * 8 * 512 * 4));
    CUDA_CHECK(cudaMemset(proj, 0, (size_t)m * 8 * 512 * 4));
    CUDA_CHECK(cudaMalloc(&out, (size_t)m * 8 * 256 * 2));
    const size_t wf = attention_prefill_work_floats(1024 * 12, 8);
    CUDA_CHECK(cudaMalloc(&work, wf * 4));
    CUDA_CHECK(cudaMalloc(&pos, 4));
    CUDA_CHECK(cudaMemcpy(pos, &ctx, 4, cudaMemcpyHostToDevice));
    attention_prefill_init();
    for (int i = 0; i < 2; i++) attention_prefill(q, kc, vc, s, proj, 8 * 512, pos, m, out, 0, work, wf);
    cudaEvent_t a, b;
    cudaEventCreate(&a); cudaEventCreate(&b);
    cudaEventRecord(a);
    for (int i = 0; i < iters; i++) attention_prefill(q, kc, vc, s, proj, 8 * 512, pos, m, out, 0, work, wf);
    cudaEventRecord(b);
    cudaEventSynchronize(b);
    float ms;
    cudaEventElapsedTime(&ms, a, b);
    cudaFree(kc); cudaFree(vc); cudaFree(q); cudaFree(proj); cudaFree(out); cudaFree(work); cudaFree(pos);
    return ms * 1e3 / iters;
}

// Prefill attention microbenchmark in the real per-GPU shape of the default position split
//: 12 local q heads in two groups of 6; layout 0 = GPU0 (kv0 whole, kv1 owned
// 416 of every 1024 positions), 1 = GPU1 (608 of 1024 of both), 2 = GPU2 (kv2 416/1024, kv3 whole),
// 3 = legacy 6+2 whole heads (8 q heads). One micro-batch of m queries at positions [ctx, ctx + m),
// split-K as in the engine (work buffer of 5 * 1024 rows). Random K/V (fp8, sd kv_sd) and Q (sd q_sd).
// res: [0] us per call (kernel + combine), [1] us kernel only, [2] useful FLOP per call, [3] splits,
// [4] max |out - ref|, [5] rms(out - ref) / rms(ref), [6] min cos over checked (row, head)
// out_host (optional): bf16 outputs [m][nq][256]; check_rows > 0: fp32 reference on that many rows
// spread over the micro-batch.
void qe_pkp_trace(unsigned long long* buf);
extern int g_pattn_kvpre, g_pkp_splits;
// bench: prefill attention path override (kvpre -1 = env/default, 0 off, 1 per-window launches, 2 persistent;
// splits -1 = env/default)
extern "C" void qe_pattn_mode(int kvpre, int splits) { g_pattn_kvpre = kvpre; g_pkp_splits = splits; }
// bench: SM clock (MHz) measured over ~1 ms of spinning on device 0's current context
double qe_sm_mhz_dev();
extern "C" double qe_sm_mhz() { return qe_sm_mhz_dev(); }
extern "C" int qe_pattn_real(int ctx, int m, int layout, int iters, float q_sd, float kv_sd, int check_rows, double* res,
                  void* out_host) {
    AttnShape s{};
    const bool legacy = layout == 3;
    s.nq = legacy ? 8 : 12;
    s.nkv = 2;
    s.kv_qb[0] = 0; s.kv_qe[0] = 6; s.kv_qb[1] = 6; s.kv_qe[1] = s.nq;
    for (int h = 0; h < s.nq; h++) s.q2kv[h] = h < 6 ? 0 : 1;
    const int P = 1024;
    if (layout == 0) { s.kv_pm[0] = PosMap{P, P, 0}; s.kv_pm[1] = PosMap{P, 416, 416}; }
    else if (layout == 1) { s.kv_pm[0] = PosMap{P, 608, 0}; s.kv_pm[1] = PosMap{P, 608, 0}; }
    else if (layout == 2) { s.kv_pm[0] = PosMap{P, 416, 416}; s.kv_pm[1] = PosMap{P, P, 0}; }
    else { s.kv_pm[0] = PosMap{P, P, 0}; s.kv_pm[1] = PosMap{P, P, 0}; }
    s.max_t = (ctx + m + 1023) / 1024 * 1024 + 1024;
    s.fp8 = 1; s.ks = 1.f; s.vs = 1.f;
    s.pf_p0 = ctx;  // as the engine: the first query position is known (int8 Q K^T from QE_PATTN_QK8_MIN)
    const size_t rows = head_rows(s, 2, s.max_t);
    void *kc, *vc;
    CUDA_CHECK(cudaMalloc(&kc, rows * 256));
    CUDA_CHECK(cudaMalloc(&vc, rows * 256));
    pattn_fill(kc, rows * 256, 0, kv_sd, 1, 0);
    pattn_fill(vc, rows * 256, 0, kv_sd, 2, 0);
    bf16* q; float* proj; bf16* out; int* pos; float* work;
    CUDA_CHECK(cudaMalloc(&q, (size_t)m * s.nq * 256 * 2));
    pattn_fill(q, (size_t)m * s.nq * 256, 1, q_sd, 3, 0);
    CUDA_CHECK(cudaMalloc(&proj, (size_t)m * s.nq * 512 * 4));
    CUDA_CHECK(cudaMemset(proj, 0, (size_t)m * s.nq * 512 * 4));  // gate 0 -> sigmoid 0.5
    CUDA_CHECK(cudaMalloc(&out, (size_t)m * s.nq * 256 * 2));
    const size_t wf = attention_prefill_work_floats(5 * 1024, s.nq);
    CUDA_CHECK(cudaMalloc(&work, wf * 4));
    CUDA_CHECK(cudaMalloc(&pos, 4));
    CUDA_CHECK(cudaMemcpy(pos, &ctx, 4, cudaMemcpyHostToDevice));
    uint2* ksc = nullptr;  // per-key int8 K scale scratch as in the engine (QE_PATTN_KSC=0: computed in-kernel)
    if (!getenv("QE_PATTN_KSC") || atoi(getenv("QE_PATTN_KSC"))) {
        CUDA_CHECK(cudaMalloc(&ksc, (size_t)s.nkv * kPattnKscRows * sizeof(uint2)));
        s.ksc = ksc;
        s.ksc_ld = kPattnKscRows;
        CUDA_CHECK(cudaMalloc(&s.kmu, (size_t)s.nkv * 256 * sizeof(float)));
        s.kvw = getenv("QE_PATTN_KVW") ? atoi(getenv("QE_PATTN_KVW")) : 32768;
        CUDA_CHECK(cudaMalloc(&s.kvq, (size_t)s.nkv * s.kvw * 256));
        CUDA_CHECK(cudaMalloc(&s.kvh, (size_t)s.nkv * s.kvw * 512));
    }
    static PattnScratch ps;  // persistent windowed attention: own buffers (the engine carves them out of idle ones)
    std::vector<void*> pfree;
    if (ksc) {
        ps = PattnScratch{};
        ps.w = getenv("QE_PATTN_KVW2") ? atoi(getenv("QE_PATTN_KVW2")) : 16384;
        for (int b = 0; b < 2; b++)
            for (int j = 0; j < s.nkv; j++) {
                CUDA_CHECK(cudaMalloc(&ps.kq[b][j], (size_t)ps.w * 256));
                CUDA_CHECK(cudaMalloc(&ps.vh[b][j], (size_t)ps.w * 512));
                pfree.push_back(ps.kq[b][j]);
                pfree.push_back(ps.vh[b][j]);
            }
        ps.qt_max = (m + 127) / 128;
        if (!getenv("QE_PATTN_QF") || atoi(getenv("QE_PATTN_QF"))) {
            CUDA_CHECK(cudaMalloc(&ps.qf, (size_t)ps.qt_max * s.nq * 8 * 8 * 32 * 16));
            CUDA_CHECK(cudaMalloc(&ps.qsq, (size_t)ps.qt_max * s.nq * 128 * 4));
            pfree.push_back(ps.qf);
            pfree.push_back(ps.qsq);
        }
        CUDA_CHECK(cudaMalloc(&ps.ctr, kPkpCtrInts * 4));
        pfree.push_back(ps.ctr);
        s.pscr = &ps;
    }
    unsigned long long* trace = nullptr;
    if (getenv("QE_PKP_TRACE")) {
        CUDA_CHECK(cudaMalloc(&trace, 65536 * 128));
        CUDA_CHECK(cudaMemset(trace, 0, 65536 * 128));
        qe_pkp_trace(trace);
        pfree.push_back(trace);
    }
    attention_prefill_init();
    const int splits = attention_prefill_splits(s, m, wf, true);
    double flop = 0;
    for (int h = 0; h < s.nq; h++) {
        const PosMap& pm = s.kv_pm[s.q2kv[h]];
        for (int i = 0; i < m; i++) flop += 4.0 * 256 * pm_rank(pm, ctx + i + 1);
    }
    for (int i = 0; i < 2; i++) attention_prefill(q, kc, vc, s, proj, s.nq * 512, pos, m, out, 0, work, wf);
    cudaEvent_t a, b;
    cudaEventCreate(&a); cudaEventCreate(&b);
    float ms = 0, ms_k = 0;
    cudaEventRecord(a);
    for (int i = 0; i < iters; i++) attention_prefill(q, kc, vc, s, proj, s.nq * 512, pos, m, out, 0, work, wf);
    cudaEventRecord(b);
    cudaEventSynchronize(b);
    cudaEventElapsedTime(&ms, a, b);
    cudaEventRecord(a);
    for (int i = 0; i < iters; i++) attention_prefill_kernel(q, kc, vc, s, proj, s.nq * 512, pos, m, out, 0, work, splits);
    cudaEventRecord(b);
    cudaEventSynchronize(b);
    cudaEventElapsedTime(&ms_k, a, b);
    attention_prefill(q, kc, vc, s, proj, s.nq * 512, pos, m, out, 0, work, wf);
    CUDA_CHECK(cudaDeviceSynchronize());
    res[0] = ms * 1e3 / iters; res[1] = ms_k * 1e3 / iters; res[2] = flop; res[3] = splits;
    res[4] = res[5] = 0; res[6] = 1;
    std::vector<uint16_t> oh((size_t)m * s.nq * 256);
    CUDA_CHECK(cudaMemcpy(oh.data(), out, oh.size() * 2, cudaMemcpyDeviceToHost));
    if (out_host) memcpy(out_host, oh.data(), oh.size() * 2);
    if (check_rows > 0) {
        float* ref;
        CUDA_CHECK(cudaMalloc(&ref, (size_t)s.nq * 256 * 4));
        std::vector<float> rh((size_t)s.nq * 256);
        double se = 0, sr = 0, mx = 0, mincos = 1;
        for (int c = 0; c < check_rows; c++) {
            const int r = check_rows == 1 ? m - 1 : (int)((long long)c * (m - 1) / (check_rows - 1));
            pattn_reference(q, kc, vc, s, pos, m, r, 1, ref, 0);
            CUDA_CHECK(cudaMemcpy(rh.data(), ref, rh.size() * 4, cudaMemcpyDeviceToHost));
            for (int h = 0; h < s.nq; h++) {
                double dot = 0, n1 = 0, n2 = 0;
                for (int d = 0; d < 256; d++) {
                    const uint32_t bits = (uint32_t)oh[((size_t)r * s.nq + h) * 256 + d] << 16;
                    float o; memcpy(&o, &bits, 4);
                    const double rv = 0.5 * rh[h * 256 + d], e = o - rv;
                    se += e * e; sr += rv * rv; mx = std::max(mx, std::fabs(e));
                    dot += o * rv; n1 += (double)o * o; n2 += rv * rv;
                }
                mincos = std::min(mincos, dot / std::sqrt(n1 * n2 + 1e-300));
                if (getenv("QE_PCHK_V") && h < 2) {
                    printf("row %d head %d cos %.6f |o| %.4f |ref| %.4f  o[0..3] ", r, h, dot / std::sqrt(n1 * n2 + 1e-300), std::sqrt(n1), std::sqrt(n2));
                    for (int d = 0; d < 4; d++) {
                        const uint32_t bits = (uint32_t)oh[((size_t)r * s.nq + h) * 256 + d] << 16;
                        float o; memcpy(&o, &bits, 4);
                        printf("%.4f/%.4f ", o, 0.5 * rh[h * 256 + d]);
                    }
                    printf("\n");
                }
            }
        }
        res[4] = mx; res[5] = std::sqrt(se / (sr + 1e-300)); res[6] = mincos;
        cudaFree(ref);
    }
    if (ksc) cudaFree(ksc), cudaFree(s.kmu), cudaFree(s.kvq), cudaFree(s.kvh);
    if (trace) {  // the last call's items -> QE_PKP_TRACE file
        std::vector<unsigned long long> th(65536 * 16);
        CUDA_CHECK(cudaMemcpy(th.data(), trace, th.size() * 8, cudaMemcpyDeviceToHost));
        FILE* f = fopen(getenv("QE_PKP_TRACE"), "wb");
        fwrite(th.data(), 8, th.size(), f);
        fclose(f);
        qe_pkp_trace(nullptr);
    }
    for (void* p : pfree) cudaFree(p);
    cudaFree(kc); cudaFree(vc); cudaFree(q); cudaFree(proj); cudaFree(out); cudaFree(work); cudaFree(pos);
    return 0;
}

// ---- kv-pos-split microbenchmark ----
// Per full-attention layer of a batched decode step on `devices` (3 GPUs): fp8 decode attention of
// nseq sequences x rows over ctx keys + the softmax-stats exchange + combine + the LL all-reduce
// that follows (M rows x 5120), 17 layers per graph launch. mode 0: the old layout (2 full kv
// heads, 8 q heads per GPU, no exchange); mode 1: the position split: GPU0 kv0 + kv1 share, GPU1
// the rest of kv1 and kv2, GPU2 kv3 + kv2 share (12 q / o_proj heads each; GPU1 owns the first
// a01 / a12 keys of every period). mode bit 2 (value 2): no all-reduce. Returns the slowest GPU's ms
// per 17 layers; per-GPU times in out_ms[3].
extern "C" double qe_bench_kvsplit(const char* devices, int ctx, int nseq, int rows, int period, int a01,
                                   int a12, int iters, int mode, double* out_ms) {
    std::vector<int> devs;
    std::stringstream ss(devices);
    for (std::string x; std::getline(ss, x, ',');) devs.push_back(std::stoi(x));
    const int ng = devs.size();
    CHECK(ng == 3, "kvsplit bench needs 3 GPUs");
    const bool split = mode & 1, ar = !(mode & 2);
    const int M = nseq * rows, H = 5120, layers = 17;
    CHECK(M <= kMaxBatchRows, "rows");
    Comm c = comm_create(ng, M * H, H);
    CHECK(comm_enable_p2p(c, devs.data()), "P2P needed");
    // each card's receive rows / flags (written by its peers over P2P): [2 slots][2 peers][M][6][2]
    float* xbuf[3] = {};
    unsigned* xflags[3] = {};
    for (int r = 0; r < ng; r++) {
        CUDA_CHECK(cudaSetDevice(devs[r]));
        CUDA_CHECK(cudaMalloc(&xbuf[r], (size_t)2 * 2 * M * 6 * kXchgFloats * 4));
        CUDA_CHECK(cudaMalloc(&xflags[r], 2 * 2 * sizeof(unsigned)));
        CUDA_CHECK(cudaMemset(xflags[r], 0, 2 * 2 * sizeof(unsigned)));
    }
    auto xrows = [&](int card, int idx, int slot) { return xbuf[card] + ((size_t)(slot * 2 + idx) * M) * 6 * kXchgFloats; };
    // head j of card r is split with card peer: receive rows / flag on r (index 0 when the peer is
    // the lower-ranked card), the peer's rows / flag this card writes (index 0 when r is lower)
    auto xwire = [&](AttnShape& sl, int r, int j, int peer, int slot) {
        const int ri = peer < r ? 0 : 1, si = r < peer ? 0 : 1;
        sl.xo[j] = xrows(r, ri, slot);
        sl.xflag[j] = xflags[r] + ri * 2 + slot;
        sl.xpo[j] = xrows(peer, si, slot);
        sl.xpflag[j] = xflags[peer] + si * 2 + slot;
    };
    std::vector<double> us(ng);
    std::vector<std::thread> th;
    for (int r = 0; r < ng; r++)
        th.emplace_back([&, r] {
            CUDA_CHECK(cudaSetDevice(devs[r]));
            cudaStream_t st;
            CUDA_CHECK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
            CommCounters ctr;
            CUDA_CHECK(cudaMalloc(&ctr.seq, 4));
            CUDA_CHECK(cudaMalloc(&ctr.done, 8));
            CUDA_CHECK(cudaMalloc(&ctr.go, 4));
            CUDA_CHECK(cudaMalloc(&ctr.bar, 8));
            CUDA_CHECK(cudaMalloc(&ctr.ss, kCommBlocks * kMaxBatchRows * 4));
            CUDA_CHECK(cudaMalloc(&ctr.epoch, 4));
            CUDA_CHECK(cudaMemset(ctr.seq, 0, 4));
            CUDA_CHECK(cudaMemset(ctr.done, 0, 8));
            CUDA_CHECK(cudaMemset(ctr.go, 0, 4));
            CUDA_CHECK(cudaMemset(ctr.bar, 0, 8));
            CUDA_CHECK(cudaMemset(ctr.epoch, 0, 4));
            unsigned* xdone;
            CUDA_CHECK(cudaMalloc(&xdone, 4));
            CUDA_CHECK(cudaMemset(xdone, 0, 4));
            // ---- shape ----
            AttnShape s{};
            s.nkv = 2;
            s.fp8 = 1; s.ks = 0.02f; s.vs = 0.02f;
            if (!split) {
                s.nq = 8;
                const int g0 = r == 0 ? 6 : (r == 1 ? 4 : 2);  // today's grouping on each GPU
                s.kv_qb[0] = 0; s.kv_qe[0] = g0; s.kv_qb[1] = g0; s.kv_qe[1] = 8;
                for (int h = 0; h < 8; h++) s.q2kv[h] = h < g0 ? 0 : 1;
                for (int j = 0; j < 2; j++) s.kv_pm[j] = PosMap{period, period, 0};
                s.ob = 0; s.no = 8;
            } else {
                s.nq = 12;
                s.kv_qb[0] = 0; s.kv_qe[0] = 6; s.kv_qb[1] = 6; s.kv_qe[1] = 12;
                for (int h = 0; h < 12; h++) s.q2kv[h] = h / 6;
                const PosMap full{period, period, 0};
                s.ob = 0; s.no = 12;
                if (r == 0) {  // kv0 full, kv1: keys GPU1 does not own
                    s.kv_pm[0] = full;
                    s.kv_pm[1] = PosMap{period, period - a01, period - a01};
                    s.kv_x[1] = 3;
                } else if (r == 1) {
                    s.kv_pm[0] = PosMap{period, a01, 0};
                    s.kv_pm[1] = PosMap{period, a12, 0};
                    s.kv_x[0] = s.kv_x[1] = 3;
                } else {
                    s.kv_pm[0] = PosMap{period, period - a12, period - a12};
                    s.kv_pm[1] = full;
                    s.kv_x[0] = 3;
                }
            }
            s.xepoch = ctr.epoch;
            s.xdone = xdone;
            s.xdiv = getenv("KVB_XDIV") ? atoi(getenv("KVB_XDIV")) : 0;
            double2* rtab = nullptr;
            CUDA_CHECK(cudaMalloc(&rtab, (size_t)M * 32 * sizeof(double2)));
            if (!getenv("KVB_NOTAB")) s.rtab = rtab;
            // ---- per-sequence buffers ----
            const int cap = ctx + 64;
            s.max_t = cap;
            const size_t kvrows = head_rows(s, 2, cap);  // both heads
            const int ld = s.nq * 512 + 2 * s.nkv * 256;
            AttnSegs g{};
            g.n = nseq;
            g.max_m = rows;
            bf16* q; float* proj; float* work; bf16* out; int* pos;
            CUDA_CHECK(cudaMalloc(&q, (size_t)M * s.nq * 256 * 2));
            CUDA_CHECK(cudaMemset(q, 0x3c, (size_t)M * s.nq * 256 * 2));
            CUDA_CHECK(cudaMalloc(&proj, (size_t)M * ld * 4));
            CUDA_CHECK(cudaMemset(proj, 0, (size_t)M * ld * 4));
            const size_t wf = attn_work_floats(s);
            CUDA_CHECK(cudaMalloc(&work, wf * 4 * nseq));
            CUDA_CHECK(cudaMalloc(&out, (size_t)M * s.no * 256 * 2));
            CUDA_CHECK(cudaMalloc(&pos, 4));
            const int p0 = ctx - rows;
            CUDA_CHECK(cudaMemcpy(pos, &p0, 4, cudaMemcpyHostToDevice));
            std::vector<void*> kcs(nseq), vcs(nseq);
            for (int i = 0; i < nseq; i++) {
                CUDA_CHECK(cudaMalloc(&kcs[i], kvrows * 256));
                CUDA_CHECK(cudaMalloc(&vcs[i], kvrows * 256));
                CUDA_CHECK(cudaMemset(kcs[i], 0x31, kvrows * 256));
                CUDA_CHECK(cudaMemset(vcs[i], 0x29, kvrows * 256));
                AttnSeg& sg = g.c[i];
                sg.proj = proj + (size_t)i * rows * ld;
                sg.q = q + (size_t)i * rows * s.nq * 256;
                sg.kc = kcs[i];
                sg.vc = vcs[i];
                sg.pos = pos;
                sg.rdelta = nullptr;
                sg.m = rows;
                sg.max_t = cap;
                sg.work = work + wf * i;
                sg.out = out + (size_t)i * rows * s.no * 256;
                sg.xh = nullptr;
            }
            bf16 *qn, *kn; float* inv_freq;
            CUDA_CHECK(cudaMalloc(&qn, 256 * 2)); CUDA_CHECK(cudaMemset(qn, 0, 512));
            CUDA_CHECK(cudaMalloc(&kn, 256 * 2)); CUDA_CHECK(cudaMemset(kn, 0, 512));
            CUDA_CHECK(cudaMalloc(&inv_freq, 32 * 4)); CUDA_CHECK(cudaMemset(inv_freq, 0, 128));
            float *resid, *part; bf16 *xn, *nw;
            CUDA_CHECK(cudaMalloc(&resid, (size_t)M * H * 4));
            CUDA_CHECK(cudaMalloc(&part, (size_t)M * H * 4));
            CUDA_CHECK(cudaMemset(part, 0, (size_t)M * H * 4));
            CUDA_CHECK(cudaMalloc(&xn, (size_t)M * H * 2));
            CUDA_CHECK(cudaMalloc(&nw, H * 2));
            CUDA_CHECK(cudaMemset(nw, 0, H * 2));
            attention_init();
            attention_prefill_init();
            // ---- the graph: 17 layers of attention (+ exchange) + all-reduce ----
            cudaGraph_t gr;
            cudaGraphExec_t ge;
            CUDA_CHECK(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal));
            comm_begin_unit(ctr, st);
            rope_table_seg(g, s, inv_freq, rtab, st);
            for (int l = 0; l < layers; l++) {
                AttnShape sl = s;
                sl.xidx = l;
                if (split) {  // alternate the receive slot
                    const int slot = l & 1;
                    if (r == 0) xwire(sl, 0, 1, 1, slot);
                    if (r == 1) { xwire(sl, 1, 0, 0, slot); xwire(sl, 1, 1, 2, slot); }
                    if (r == 2) xwire(sl, 2, 0, 1, slot);
                }
                attention_seg(g, sl, ld, qn, kn, 1e-6f, inv_freq, st);
                if (ar) allreduce_add(resid, part, M, H, c, r, ctr, st, nw, 1e-6f, xn, l, false, nullptr);
            }
            CUDA_CHECK(cudaStreamEndCapture(st, &gr));
            CUDA_CHECK(cudaGraphInstantiate(&ge, gr, 0));
            for (int i = 0; i < 3; i++) CUDA_CHECK(cudaGraphLaunch(ge, st));
            CUDA_CHECK(cudaStreamSynchronize(st));
            auto t0 = std::chrono::steady_clock::now();
            for (int i = 0; i < iters; i++) CUDA_CHECK(cudaGraphLaunch(ge, st));
            CUDA_CHECK(cudaStreamSynchronize(st));
            us[r] = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / iters;
            CUDA_CHECK(cudaGraphExecDestroy(ge));
            CUDA_CHECK(cudaGraphDestroy(gr));
            for (int i = 0; i < nseq; i++) { cudaFree(kcs[i]); cudaFree(vcs[i]); }
            for (void* p : {(void*)q, (void*)proj, (void*)work, (void*)out, (void*)pos, (void*)qn, (void*)kn, (void*)rtab,
                            (void*)inv_freq, (void*)resid, (void*)part, (void*)xn, (void*)nw, (void*)xdone,
                            (void*)ctr.seq, (void*)ctr.done, (void*)ctr.go, (void*)ctr.bar, (void*)ctr.ss,
                            (void*)ctr.epoch})
                cudaFree(p);
            cudaStreamDestroy(st);
        });
    for (auto& t : th) t.join();
    double mx = 0;
    for (int r = 0; r < ng; r++) {
        if (out_ms) out_ms[r] = us[r] / 1e3;
        mx = std::max(mx, us[r] / 1e3);
    }
    for (int r = 0; r < ng; r++) {
        CUDA_CHECK(cudaSetDevice(devs[r]));
        cudaFree(xbuf[r]);
        cudaFree(xflags[r]);
    }
    comm_free_p2p(c, devs.data());
    for (void* p : {(void*)c.data, (void*)c.flags, (void*)c.ll})
        if (p) cudaFreeHost(p);
    return mx;
}

// ---- DeltaNet decode state traffic: one verification step of 48 delta-net
// layers for nseg sequences of m rows each, per-GPU shapes (16 v heads, 6 k heads), every
// sequence's 48 states in its own memory (> L2) as in the engine.
// mode 0: master (dn_core_seg read-only per layer + dn_replay_seg of npend rows + commit)
// mode 1: deferred commit (dn_core_seg replays npend stashed rows, writes the state; dn_stash_seg)
// mode 2: dn_core_seg only (read-only), mode 3: dn_replay_seg only
namespace {
struct DnBench {
    int nseg = 0, m = 0, L = 48, ch = 0, prf = 0;
    DeltaShape s{};
    size_t slot = 0;
    std::vector<float*> S, cv, qk, gb;
    float *o = nullptr, *pend = nullptr;
    int *cur = nullptr, *acc = nullptr, *pn = nullptr, *save = nullptr;
    DnBench(int nseg_, int m_, int lazy = 1) : nseg(nseg_), m(m_) {
        s.nk = 6; s.nv = 16;
        s.lazy_t = lazy;
        for (int h = 0; h < 16; h++) s.v2k[h] = h * 6 / 16;
        s.ab_pad = 16; s.ab_splits = 1; s.ab_rows = nseg * m;
        ch = 2 * s.nk * 128 + s.nv * 128;
        prf = dn_pend_row_floats(s);
        slot = (size_t)s.nv * 128 * 128;
        const int rows = nseg * m;
        for (int i = 0; i < L * nseg; i++) {
            float* p;
            CUDA_CHECK(cudaMalloc(&p, 2 * slot * 4));
            CUDA_CHECK(cudaMemset(p, 0, 2 * slot * 4));
            S.push_back(p);
        }
        for (int l = 0; l < L; l++) {
            float *a, *b, *c;
            CUDA_CHECK(cudaMalloc(&a, (size_t)rows * ch * 4));
            CUDA_CHECK(cudaMalloc(&b, (size_t)rows * 2 * s.nk * 128 * 4));
            CUDA_CHECK(cudaMalloc(&c, (size_t)rows * 2 * s.nv * 4));
            CUDA_CHECK(cudaMemset(a, 0, (size_t)rows * ch * 4));
            CUDA_CHECK(cudaMemset(b, 0, (size_t)rows * 2 * s.nk * 128 * 4));
            CUDA_CHECK(cudaMemset(c, 0, (size_t)rows * 2 * s.nv * 4));
            cv.push_back(a); qk.push_back(b); gb.push_back(c);
        }
        CUDA_CHECK(cudaMalloc(&o, (size_t)rows * s.nv * 128 * 4));
        CUDA_CHECK(cudaMalloc(&pend, (size_t)nseg * L * dn_pcap(s) * prf * 4));
        CUDA_CHECK(cudaMemset(pend, 0, (size_t)nseg * L * dn_pcap(s) * prf * 4));
        for (int** p : {&cur, &acc, &pn, &save}) {
            CUDA_CHECK(cudaMalloc(p, 2 * kMaxSegs * 4));
            CUDA_CHECK(cudaMemset(*p, 0, 2 * kMaxSegs * 4));
        }
    }
    ~DnBench() {
        for (float* p : S) cudaFree(p);
        for (auto* v : {&cv, &qk, &gb})
            for (float* p : *v) cudaFree(p);
        for (void* p : {(void*)o, (void*)pend, (void*)cur, (void*)acc, (void*)pn, (void*)save}) cudaFree(p);
    }
    float* pend_of(int z, int l) const { return pend + ((size_t)z * L + l) * dn_pcap(s) * prf; }
    void core(bool defer, cudaStream_t st) {
        for (int l = 0; l < L; l++) {
            DnSegs g{};
            g.n = nseg;
            for (int z = 0; z < nseg; z++) {
                g.r0[z] = z * m;
                g.m[z] = m;
                g.S[z] = S[(size_t)l * nseg + z];
                g.cur[z] = cur + z;
                if (defer) {
                    g.pn[z] = pn + 2 * z;
                    g.pend[z] = pend_of(z, l);
                    g.fin[z] = m == 1 && dn_lazy_t(s) == 1;
                }
            }
            dn_core_seg(g, cv[l], ch, s, qk[l], gb[l], slot, o, st);
        }
    }
    void replay(cudaStream_t st) {
        DnReplaySegs rp{};
        rp.n = nseg;
        for (int l = 0; l < L; l++) {
            rp.cv[l] = cv[l]; rp.qk[l] = qk[l]; rp.gb[l] = gb[l];
            for (int z = 0; z < nseg; z++) rp.S[z][l] = S[(size_t)l * nseg + z];
        }
        for (int z = 0; z < nseg; z++) { rp.r0[z] = z * m; rp.cur[z] = cur + z; rp.acc[z] = acc + z; }
        dn_replay_seg(rp, L, ch, s, slot, st);
    }
    void stash(cudaStream_t st) {
        DnStashSegs r{};
        r.n = nseg;
        for (int l = 0; l < L; l++) { r.cv[l] = cv[l]; r.qk[l] = qk[l]; r.gb[l] = gb[l]; }
        for (int z = 0; z < nseg; z++) {
            r.r0[z] = z * m; r.cur[z] = cur + z; r.pn[z] = pn + 2 * z; r.save[z] = save + z; r.fin[z] = m == 1 && dn_lazy_t(s) == 1; r.m[z] = m; r.pend[z] = pend_of(z, 0); r.acc[z] = acc + z;
        }
        dn_stash_seg(r, L, ch, s, st);
    }
    void flush(int z, cudaStream_t st) {
        DnFlush r{};
        for (int l = 0; l < L; l++) r.S[l] = S[(size_t)l * nseg + z];
        r.cur = cur + z; r.pn = pn + 2 * z; r.pend = pend_of(z, 0);
        dn_flush(r, L, s, slot, st);
    }
    void step(int mode, cudaStream_t st) {
        if (mode == 0) { core(false, st); replay(st); }
        else if (mode == 1) { core(true, st); stash(st); }
        else if (mode == 2) core(false, st);
        else replay(st);
    }
};
}  // namespace

// us per step (CUDA graph of one step, iters launches); npend = accepted rows per sequence (acc + 1)
extern "C" double qe_bench_dn_state(int nseg, int m, int npend, int mode, int iters, int lanes, int lazy) {
    dn_core2_set(lanes);
    CHECK(nseg >= 1 && nseg <= kMaxSegs && m >= 1 && m <= kPendRows && npend >= 1 && npend <= m, "args");
    DnBench b(nseg, m, lazy);
    std::vector<int> a(nseg, npend - 1), p(2 * nseg, 0);
    for (int z = 0; z < nseg; z++) p[2 * z] = npend;
    CUDA_CHECK(cudaMemcpy(b.acc, a.data(), nseg * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(b.pn, p.data(), 2 * nseg * 4, cudaMemcpyHostToDevice));
    cudaStream_t st;
    CUDA_CHECK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
    cudaGraph_t g;
    cudaGraphExec_t ge;
    CUDA_CHECK(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal));
    b.step(mode, st);
    CUDA_CHECK(cudaStreamEndCapture(st, &g));
    CUDA_CHECK(cudaGraphInstantiate(&ge, g, 0));
    for (int i = 0; i < 3; i++) CUDA_CHECK(cudaGraphLaunch(ge, st));
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0); cudaEventCreate(&e1);
    cudaEventRecord(e0, st);
    for (int i = 0; i < iters; i++) CUDA_CHECK(cudaGraphLaunch(ge, st));
    cudaEventRecord(e1, st);
    CUDA_CHECK(cudaEventSynchronize(e1));
    float ms;
    cudaEventElapsedTime(&ms, e0, e1);
    CUDA_CHECK(cudaGetLastError());
    cudaGraphExecDestroy(ge); cudaGraphDestroy(g); cudaStreamDestroy(st);
    cudaEventDestroy(e0); cudaEventDestroy(e1);
    return ms * 1e3 / iters;
}

// Master (replay) vs deferred commit over `steps` random verification steps of nseg sequences
// with m rows (random acceptance, random flushes and undos): returns the number of output / final
// state floats that differ bitwise (0 = bit identical).
extern "C" long long qe_test_dn_defer(int nseg, int m, int steps, int seed, int lanes, int lazy) {
    DnBench A(nseg, m), B(nseg, m, lazy);
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::uniform_real_distribution<float> ud(0.f, 1.f);
    const int rows = nseg * m;
    for (int i = 0; i < A.L * nseg; i++) {  // same random initial states (slot 0)
        std::vector<float> h(A.slot);
        for (auto& x : h) x = nd(rng) * 0.05f;
        CUDA_CHECK(cudaMemcpy(A.S[i], h.data(), A.slot * 4, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(B.S[i], h.data(), A.slot * 4, cudaMemcpyHostToDevice));
    }
    long long bad = 0;
    std::vector<float> oa((size_t)rows * A.s.nv * 128), ob(oa.size());
    for (int t = 0; t < steps; t++) {
        for (int l = 0; l < A.L; l++) {
            std::vector<float> hcv((size_t)rows * A.ch), hqk((size_t)rows * 2 * A.s.nk * 128), hgb((size_t)rows * 2 * A.s.nv);
            for (auto& x : hcv) x = nd(rng);
            for (auto& x : hqk) x = nd(rng) * 0.0884f;
            for (size_t r = 0; r < (size_t)rows; r++)
                for (int h = 0; h < A.s.nv; h++) {
                    hgb[r * 2 * A.s.nv + h] = ud(rng) < 0.1f ? 0.f : 0.5f + 0.5f * ud(rng);
                    hgb[r * 2 * A.s.nv + A.s.nv + h] = ud(rng);
                }
            for (DnBench* b : {&A, &B}) {
                CUDA_CHECK(cudaMemcpy(b->cv[l], hcv.data(), hcv.size() * 4, cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemcpy(b->qk[l], hqk.data(), hqk.size() * 4, cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemcpy(b->gb[l], hgb.data(), hgb.size() * 4, cudaMemcpyHostToDevice));
            }
        }
        std::vector<int> acc(nseg);
        for (auto& x : acc) x = rng() % m;
        CUDA_CHECK(cudaMemcpy(A.acc, acc.data(), nseg * 4, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(B.acc, acc.data(), nseg * 4, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(A.save, A.cur, nseg * 4, cudaMemcpyDeviceToDevice));
        dn_core2_set(0);  // reference: master kernels
        A.step(0, 0);
        dn_core2_set(lanes);
        B.step(1, 0);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(oa.data(), A.o, oa.size() * 4, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(ob.data(), B.o, ob.size() * 4, cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < oa.size(); i++) bad += memcmp(&oa[i], &ob[i], 4) != 0;
        for (int z = 0; z < nseg; z++) {
            const unsigned r = rng() % 8;
            if (r == 0) {  // undo this step for z (engine: launch-ahead step dropped)
                CUDA_CHECK(cudaMemcpy(A.cur + z, A.save + z, 4, cudaMemcpyDeviceToDevice));
                CUDA_CHECK(cudaMemcpy(B.cur + z, B.save + z, 4, cudaMemcpyDeviceToDevice));
                CUDA_CHECK(cudaMemcpy(B.pn + 2 * z, B.pn + 2 * z + 1, 4, cudaMemcpyDeviceToDevice));
            } else if (r == 1) {  // somebody reads the state (snapshot / prefill)
                B.flush(z, 0);
            }
        }
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    for (int z = 0; z < nseg; z++) B.flush(z, 0);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<int> ca(nseg), cb(nseg), pb(2 * nseg);
    CUDA_CHECK(cudaMemcpy(ca.data(), A.cur, nseg * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(cb.data(), B.cur, nseg * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(pb.data(), B.pn, 2 * nseg * 4, cudaMemcpyDeviceToHost));
    std::vector<float> sa(A.slot), sb(A.slot);
    for (int z = 0; z < nseg; z++) {
        if (pb[2 * z] != 0) bad += 1000000;
        for (int l = 0; l < A.L; l++) {
            CUDA_CHECK(cudaMemcpy(sa.data(), A.S[(size_t)l * nseg + z] + ca[z] * A.slot, A.slot * 4, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(sb.data(), B.S[(size_t)l * nseg + z] + cb[z] * A.slot, A.slot * 4, cudaMemcpyDeviceToHost));
            for (size_t i = 0; i < A.slot; i++) bad += memcmp(&sa[i], &sb[i], 4) != 0;
        }
    }
    CUDA_CHECK(cudaGetLastError());
    return bad;
}

// dn_core2 (lanes) vs dn_core_dev on a plain (non-speculative) pass of m rows from slot cur:
// returns the number of differing output / state floats
extern "C" long long qe_test_dn_core2_plain(int m, int cur0, int lanes, int seed, int np) {
    DnBench A(1, m), B(1, m);
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::uniform_real_distribution<float> ud(0.f, 1.f);
    std::vector<float> h(2 * A.slot), hcv((size_t)m * A.ch), hqk((size_t)m * 2 * A.s.nk * 128), hgb((size_t)m * 2 * A.s.nv);
    for (auto& x : h) x = nd(rng) * 0.05f;
    for (auto& x : hcv) x = nd(rng);
    for (auto& x : hqk) x = nd(rng) * 0.0884f;
    for (auto& x : hgb) x = 0.5f + 0.5f * ud(rng);
    for (DnBench* b : {&A, &B}) {
        CUDA_CHECK(cudaMemcpy(b->S[0], h.data(), h.size() * 4, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(b->cv[0], hcv.data(), hcv.size() * 4, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(b->qk[0], hqk.data(), hqk.size() * 4, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(b->gb[0], hgb.data(), hgb.size() * 4, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(b->cur, &cur0, 4, cudaMemcpyHostToDevice));
    }
    if (np > 0) {  // spec pass with np pending rows
        std::vector<float> hp((size_t)A.L * kPendRows * A.prf);
        for (auto& x : hp) x = nd(rng) * 0.1f;
        int pn2[2] = {np, 0};
        for (DnBench* b : {&A, &B}) {
            CUDA_CHECK(cudaMemcpy(b->pend, hp.data(), hp.size() * 4, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(b->pn, pn2, 8, cudaMemcpyHostToDevice));
        }
    }
    const bool spec = np >= 0;
    dn_core2_set(0);
    dn_core_only(A.cv[0], A.ch, A.s, A.S[0], A.slot, A.cur, m, spec, A.qk[0], A.gb[0], A.o, 0,
                 spec ? A.pn : nullptr, spec ? A.pend : nullptr);
    dn_core2_set(lanes);
    dn_core_only(B.cv[0], B.ch, B.s, B.S[0], B.slot, B.cur, m, spec, B.qk[0], B.gb[0], B.o, 0,
                 spec ? B.pn : nullptr, spec ? B.pend : nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());
    long long bad = 0;
    std::vector<float> a((size_t)m * A.s.nv * 128), b(a.size()), sa(2 * A.slot), sb(sa.size());
    CUDA_CHECK(cudaMemcpy(a.data(), A.o, a.size() * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(b.data(), B.o, b.size() * 4, cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < a.size(); i++) bad += memcmp(&a[i], &b[i], 4) != 0;
    CUDA_CHECK(cudaMemcpy(sa.data(), A.S[0], sa.size() * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(sb.data(), B.S[0], sb.size() * 4, cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < sa.size(); i++) bad += memcmp(&sa[i], &sb[i], 4) != 0;
    return bad;
}
