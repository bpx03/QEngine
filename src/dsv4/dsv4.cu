// DSV4 M1 engine: load, placement, prefill (layer-major over the chunk) and decode (c = 1). See dsv4.h.
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/ioctl.h>
#include <linux/fiemap.h>
#include <linux/fs.h>
#include <linux/io_uring.h>
#include <unistd.h>
#include <immintrin.h>
#include <cerrno>

#include <algorithm>
#include <cstdarg>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <condition_variable>
#include <functional>
#include <map>
#include <set>
#include <array>
#include <cublasLt.h>
#include <cuda.h>   // [dsv4-ctx-lazy] VMM types only (the entry points are resolved at run time, no -lcuda)
#include <tuple>
#include <type_traits>
#include <initializer_list>
#include <mutex>
#include <numeric>
#include <sstream>
#include <stdexcept>

#include "../json.h"
#include "../safetensors.h"
#include "dsv4.h"
#include "dsv4_cpu.h"
#include "dsv4_kernels.cuh"
#include "dsv4_m23.h"
#include "gguf_gpu.cuh"   // R16 [dsv4-gguf-iq]
#include "gguf_fx.cuh"   // [dsv4-pf2-fx]
#include "gguf_fxl.cuh"  // [dsv4-pf10-fxl]
#include "draft_gpu.cuh"  // R22 [dsv4-r22-dmoe8]
#include "dsv4_sample.h"  // release prep [dsv4-rel-sample]
#include "gguf_iq.h"
#include "gguf_reader.h"
#include <cuda_profiler_api.h>
#include <nvtx3/nvToolsExt.h>

namespace dsv4 {

using clk = std::chrono::steady_clock;
static double ms_since(clk::time_point t) { return std::chrono::duration<double, std::milli>(clk::now() - t).count(); }
static inline unsigned cdiv(size_t a, size_t b) { return (unsigned)((a + b - 1) / b); }
constexpr int GS = 1 << 16;   // R9: global slot id = card * GS + slot index on that card
static inline int gs_card(int g) { return g / GS; }
static inline int gs_idx(int g) { return g % GS; }
struct DevGuard {             // current device = d for the scope (launches / event records on another card's streams)
    int prev = -1;
    explicit DevGuard(int d) {
        DCK(cudaGetDevice(&prev));
        if (prev != d) DCK(cudaSetDevice(d));
        else prev = -1;
    }
    ~DevGuard() {
        if (prev >= 0) DCK(cudaSetDevice(prev));
    }
};
static void logf_(const char* fmt, ...) __attribute__((format(printf, 1, 2)));
static void logf_(const char* fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vfprintf(stderr, fmt, ap);
    va_end(ap);
    fflush(stderr);
}

// ---------------------------------------------------------------- structures

// FP4 bytes of an expert from the checkpoint (O_DIRECT) for admissions of experts whose host copy is not FP4
// (pack D): read into a pinned bounce buffer, then copied to the VRAM slot (dyn_apply), visible at a fixed token delay
struct XJob {
    std::atomic<int> state{0};    // 0 free, 1 queued, 2 read, 3 copy issued
    uint8_t* buf = nullptr;       // pinned, 6 aligned segments
    int fd[6] = {}, len[6] = {};
    size_t off[6] = {}, boff[6] = {}, delta[6] = {};
    // [dsv4-rel-ram]: set = no file runs: prep(J) fills buf (a GGUF expert re-packed from the file's mapping)
    void (*prep)(XJob*) = nullptr;
    const void* ctx = nullptr;
    int pl = 0, pe = 0;
};
constexpr int XJ_PER_CARD = 6;
constexpr size_t XJ_BYTES = 18800640 + 6 * 8192;
constexpr int PRED_ROWS = 6, PRED_STRIDE = 64;
static_assert(PRED_ROWS == 6, "the k_gemv_f32k row switches (predictor, DSV4_EARLYMR) end at 6 rows (review dec2 L6)");   // expert prediction for passes of up to 6 rows (DSpark verify)
struct XReader {
    std::mutex mu;
    std::condition_variable cv;
    std::vector<XJob*> q;
    bool quit = false;
    std::vector<std::thread> th;
    std::atomic<long> retries{0}, failed{0};
    void worker() {
        while (true) {
            XJob* J;
            {
                std::unique_lock<std::mutex> lk(mu);
                cv.wait(lk, [&] { return quit || !q.empty(); });
                if (quit) return;
                J = q.front();   // FIFO (review r7 M2): the first layers' reads first
                q.erase(q.begin());
            }
            bool bad = false;
            if (J->prep) {   // [dsv4-rel-ram]
                J->prep(J);
                J->state.store(2, std::memory_order_release);
                continue;
            }
            for (int i = 0; i < 6 && !bad; i++) {
                size_t a0 = J->off[i] & ~(size_t)4095, a1 = (J->off[i] + J->len[i] + 4095) & ~(size_t)4095;
                bool ok = false;
                for (int attempt = 0; attempt < 20; attempt++) {   // review r6 M3: give up and degrade, never abort
                    ssize_t r = pread(J->fd[i], J->buf + J->boff[i], a1 - a0, a0);
                    if (r >= (ssize_t)(J->off[i] - a0 + J->len[i])) { ok = true; break; }
                    retries++;
                    if (attempt < 3) fprintf(stderr, "expert read retry %d: %s\n", attempt, r < 0 ? strerror(errno) : "short");
                    std::this_thread::sleep_for(std::chrono::milliseconds(attempt < 10 ? 1 : 20));
                }
                if (!ok) { failed++; bad = true; break; }
                J->delta[i] = J->off[i] - a0;
            }
            J->state.store(bad ? 4 : 2, std::memory_order_release);   // 4: failed, the slot stays empty
        }
    }
    void push(XJob* J) {
        {
            std::lock_guard<std::mutex> lk(mu);
            q.push_back(J);
        }
        cv.notify_one();
    }
    void stop() {
        {
            std::lock_guard<std::mutex> lk(mu);
            quit = true;
        }
        cv.notify_all();
        for (auto& t : th) t.join();
    }
};

struct Layer {
    int card = 0, ratio = 0, lid = 0;
    bool dspark = false;                      // DSpark draft stage (window + whole block, 128 experts top-3)
    bool xcb3 = false;                        // R8: its experts are cb3g128 (xslots holds 128 x expert_bytes(XF_CB3))
    uint8_t* xslots = nullptr;                // DSpark: this stage's 128 FP4 experts in VRAM
    F8 wq_a, wq_b, wkv, wo_a, wo_b, i_wq_b, sh_w1, sh_w3, sh_w2, e_wkv;
    F8 wqkv;                                  // R10: [wq_a; wkv] (main layers; wq_a / wkv are views into it)
    float *q_norm = nullptr, *kv_norm = nullptr, *sink = nullptr, *attn_norm = nullptr, *ffn_norm = nullptr;
    float *hc_fn[2] = {}, *hc_scale[2] = {}, *hc_base[2] = {};
    // R10: the BF16 projections stay BF16 in VRAM (widened exactly in the GEMV)
    const bf16 *c_wkv = nullptr, *c_wgate = nullptr;
    F8 cq_wkv, cq_wgate;   // [dsv4-rel-q8]: the compressor linears as Q8_0 (GGUF mode), else unused
    float* c_norm = nullptr;
    const bf16 *i_wproj = nullptr, *i_wk = nullptr;
    float* i_knorm = nullptr;
    const bf16* gate_w = nullptr;
    float* gate_b = nullptr;
    float* e_qk = nullptr;
    bf16* kvw = nullptr;                      // window KV: [ARING][512] ring (row p & kmask) or [maxctx][512] (DSV4_NOKVRING=1)
    int kmask = -1;                           // [dsv4-rel2-p1] ARING - 1 with the ring, -1 = absolute rows
    bf16* kvs = nullptr;                      // [dsv4-rel2-p1] prefill scratch [chunk + WIN][512] (ring mode, main layers)
    int kvs_p0 = 0;                           //   position of scratch row 0 during a prefill call (s - WIN)
    bool kvs_on = false;                      //   a prefill call holds the scratch (kv_in .. kv_out)
    bf16* kvsnap = nullptr;                   // [dsv4-rel2-p1] the ring at the prompt-end snapshot (ring mode)
    int* d_slot = nullptr;                    // [NE] global slot (card * GS + index) or -1, on the owner (routing: CAR, prediction)
    int* d_ls[3] = {};                        // R9: [NE] slot index on card k or -1, on card k (the GEMV kernels read it)
    int* d_dls[3] = {};                       // R16 pool split: [NE] pool slot index on card k or -1, on card k
    std::vector<int> h_slot;                  // global slot or -1
    std::vector<int8_t> hfmt;                 // host tier format per expert (-1: no host copy)
    std::vector<const uint8_t*> hx;           // host copy per expert
    std::vector<int> rank;                    // global usage rank
    // dynamic tier (decode): cb3 copies of host experts in this layer's share of the card's pool
    int* d_dslot = nullptr;                   // [NE] pool slot or -1 (device; what the kernels read)
    std::vector<int> h_dslot;                 // host mirror, kept equal to the device table in stream order
    std::vector<int> dpool;                   // pool slots owned by this layer
    std::vector<int> dyn_e;                   // expert in each owned slot (-1 free)
    std::vector<int> pend_e;                  // expert being copied into each owned slot (-1 none)
    std::vector<cudaEvent_t> pend_ev;
    std::vector<float> score;                 // LFU score per expert (lazy decay)
    std::vector<int> score_t, miss_t;         // token of the last score update / last miss
    // [dsv4-rel-tierreset]: the tier state at the end of the load (seeded pool), restored per request on opt-in
    std::vector<int> seed_e, seed_score_t, seed_miss_t;
    std::vector<float> seed_score;
    std::vector<int8_t> inpool;               // FP4 dynamic tier: expert sits in one of this layer's pool slots
    std::vector<int8_t> pin;                  // [dsv4-rel-ram]: a GGUF seed without a host copy: static in its pool slot
    std::vector<int8_t> bor;                  // R9: evicted for the current prefill (workspace borrowing)
    std::vector<int8_t> ck;                   // host copy is FP4 from the checkpoint (not in the pack file)
    std::vector<int8_t> own;                  // host copy is its own allocation (freeable by the memory guard)
    std::vector<int> pend_job, pend_t;        // pending admission from the checkpoint: card job index, decision token
    // dsv4-fut [dsv4-fut-early]: one spare pool slot of the owner card (outside the borrowable tail, pend_e = -3), filled
    // in the pass by the predictor-driven early admission; early_e: expert being copied into it this pass (-1 none)
    int spare_g = -1, early_e = -1, early_seq = 0, early_ap = 0;
    cudaEvent_t early_ev = nullptr;
    std::vector<int> pend_m;                  // R13 swap tier: victim (no host copy) whose slot bytes are being copied
                                              // D2H into pend_h; it stays resident until that copy is done (-1 none)
    std::vector<uint8_t*> pend_h;
    // R8 attention split: card k's share of this layer's heads (wo_a groups g0 .. g0 + ng - 1) and wo_b rows
    // [n0, n0 + nn); on the owner views into the full weights, on the other cards own copies of the slices
    struct AShare {
        int g0 = 0, ng = 0, n0 = 0, nn = 0;
        F8 wq_b, wo_a, wo_b;
        float* sink = nullptr;
        bf16* ring = nullptr;   // other cards: window KV ring [ARING][512] (position p at row p % ARING)
    } ash[3];
};
constexpr int ARING = 256;                    // >= WIN + MAXR
constexpr int X1W = (QLR + HD / 2 + ITOPK + HD / 2 + IDIM / 2 + 2) / 3, X2W = 3 * OLR / 3, X3W = 1728 / 3;   // LL12 words per row

struct SrcCache {
    bf16* ckv = nullptr;   // [cap][512]   the active sequence's rows (a view into the pool at its base page)
    bf16* ik = nullptr;    // [cap][128]
    bf16* pck = nullptr;   // [dsv4-rel4-p2] the pool: [pages * KVPG][512] / [pages * KVPG][128], same page ids on every card
    bf16* pik = nullptr;
};
constexpr int KVPG = 256;   // [dsv4-rel4-p2] rows per page of the compressed K/V pool

// [dsv4-pf-nst] prefill expert stage ring depth: 8 measured no gain over 3 (round 1, i3) and costs ~92 MB of VRAM per
// card (review pf M3): 3
constexpr int NST = 3;
struct Card {
    int dev = 0, l0 = 0, l1 = 0;
    cudaStream_t st = nullptr, cp = nullptr, st2 = nullptr;   // st2: decode expert predictor (beside the attention)
    cudaStream_t cpd = nullptr;                              // R13 swap tier: slot -> host copies (beside cp's H2D)
    cudaStream_t cpe = nullptr;                              // dsv4-fut: early admissions (high priority, beside cp)
    cudaStream_t cxs = nullptr;                              // [dsv4-pf2-cpu]: xq D2H for the CPU experts (beside st)
    int* d_early = nullptr;                                  // dsv4-fut: [NL][4] early request (e, local slot, seq), device
    // R10 step graph: est = the eager compute stream (always the original st: the decode thread's tier updates, the graph
    // launch); cst / cst2 = capture streams, swapped into st / st2 by the capturing thread for the capture
    cudaStream_t est = nullptr, cst = nullptr, cst2 = nullptr;
    cudaEvent_t ev_p0 = nullptr, ev_p1 = nullptr;
    cudaEvent_t ev_e0 = nullptr, ev_e1 = nullptr;   // R11 early predictor fork / join (side stream st2)
    cudaEvent_t ev_s0 = nullptr, ev_s1 = nullptr;   // R17 shared expert fork / join (side stream st2)
    float* d_elg = nullptr;                          // R11 early predictor logits [NE], ids / weights [64]
    int* d_epid = nullptr;
    float* d_epw = nullptr;
    cublasHandle_t blas = nullptr;
    uint8_t* slots = nullptr;
    int nslots = 0, used_slots = 0;
    // R9 workspace borrowing: the chunk-sized prefill buffers carved from slots [nslots - nborrow, nslots) during a long
    // prefill; decode uses MAXR-row buffers (dws)
    struct WS {
        float *h = nullptr, *x = nullptr, *ao = nullptr, *qr = nullptr;
        bf16 *xb = nullptr, *deq = nullptr, *w13 = nullptr, *w2 = nullptr;
        int* sel = nullptr;
        uint8_t* cand = nullptr;
        uint8_t* stage[NST] = {};
        size_t xb_n = 0;   // [dsv4-pf8-xqh]
    } dws;
    int nborrow = 0;
    bool big = true;
    float *cosv[2] = {}, *sinv[2] = {};
    SrcCache src[4];
    float *tail_kv[4] = {}, *tail_sc[4] = {};    // ratio-2 compressor tail row (source owner card)
    float *sv_kv[4] = {}, *sv_sc[4] = {};        // DSpark: compressor input rows of the last verify pass (rollback)
    float *snap_kv[4] = {}, *snap_sc[4] = {};    // [dsv4-srv-pesnap] ratio-2 tail row at the snapshot position
    // DSpark (last card): main hidden of the target layers [WIN][3 * 5120], main_x, draft embeddings, results
    float *d_mh = nullptr, *d_mx = nullptr, *d_zero = nullptr, *d_conf = nullptr, *d_embf = nullptr;
    bf16* d_demb = nullptr;
    int* d_dtok = nullptr;
    unsigned long long* d_mkey = nullptr;   // R22 [dsv4-r22-mk8]: per draft row the packed (logit, id) argmax
    char* h_dout = nullptr;                      // pinned: draft ids [8] | confidences [8]
    // persistent activations (prefill chunk sized; decode uses row 0..)
    float *h = nullptr, *pre = nullptr, *apre = nullptr, *apost = nullptr, *acomb = nullptr, *fpre = nullptr,
          *fpost = nullptr, *fcomb = nullptr, *mix = nullptr, *ss = nullptr;
    float *x = nullptr, *ao = nullptr, *qr = nullptr, *kvt = nullptr;
    bf16* qrb = nullptr;                         // R15: decode qr FP8-rounded as bf16 [MAXR][QLR] (wq_b input, X1)
    bf16* xb = nullptr;                          // bf16 activations [chunk][8192]
    size_t xb_n = 0;                             // [dsv4-pf8-xqh] its size (bf16 elements)
    bf16* xbs = nullptr;                         // R18 (review r17 L6): linq scratch of the side stream [MAXR][8192]
    float* f32s = nullptr;                       // R18: same, fp32 [MAXR][8192]
    int *sel = nullptr;                          // [chunk][512]
    uint8_t* cand = nullptr;                     // [chunk][nbk_max]
    int *ids = nullptr, *rows = nullptr;
    float *rw = nullptr, *rwt = nullptr;
    float* f32tmp = nullptr;                     // decode / small temporaries [MAXR][32768]
    int cx_l = -1, cx_g0 = 0, cx_ng = 0;         // R11: the last compressor call's new rows (sent in attn_split's X1)
    float* f32tmp2 = nullptr;
    float* d_pbuf = nullptr;                     // [dsv4-dec1-plate] the predictor's own xp / logits / ids (late join)
    cudaEvent_t ev_p2 = nullptr;                 // [dsv4-dec1-plate] predictor input read (c.h / c.apre free again)
    float* ones = nullptr;                       // [chunk] of 1 (shared expert GLU weight)
    // phase region (attention / MoE / engram temporaries)
    char* phase = nullptr;
    size_t phase_bytes = 0;
    bf16* deq = nullptr;                         // dequantized weight slice
    size_t deq_elems = 0;
    bf16 *w13 = nullptr, *w2 = nullptr;          // expert dequant scratch
    uint8_t* dpool = nullptr;                    // dynamic tier pool: ndyn x expert_bytes(XF_CB3)
    int ndyn = 0;
    // R16 pool split: this card's pool region of layer l (byte offset, first slot index, slots); per-layer expert sizes
    // (pool_fmt) so a format with larger edge-layer experts (GGUF) fits the same layout
    size_t ploff[NL] = {};
    size_t dpool_bytes = 0;                  // R19: bytes of the split pool (its tail is borrowed by long prefills)
    int ps0[NL] = {}, pnl[NL] = {};
    // [dsv4-ctx-lazy] lazy context reservation (VMM): buffers whose size follows --max-ctx (compressed K/V + index keys
    // of the 4 sources, the rope tables) are address space for the full window, mapped only for the base context at
    // load; growth maps 2 MB chunks lent by the expert pool (loan zones: the last slots of each layer's pool region,
    // outside the prefill borrow tail). lzb: 0-3 ckv, 4-7 ik, 8-11 rope (cos 0, sin 0, cos 1, sin 1)
    struct LzBuf {
        unsigned long long va = 0;       // CUdeviceptr
        size_t cap = 0, base = 0, mapped = 0;   // reserved / mapped at load / mapped now (bytes, 2 MB multiples)
        unsigned long long hbase = 0;    // the base's own handle
        std::vector<int> ch;             // lent chunks mapped after the base, in order
        std::vector<unsigned long long> fresh;   // own handles mapped after the base (no zone could lend)
    } lzb[12];
    struct LzChunk { size_t off; unsigned long long h; int user = -1; };   // a zone chunk: dpool offset, handle, buffer or -1
    std::vector<LzChunk> lzc;
    int lz_nmax[NL] = {};                // loan zone per layer: its last lz_nmax slots (0: no zone)
    int lz_n[NL] = {};                   // of those, evicted now (the chunks inside them can be lent)
    int lz_c0[NL] = {}, lz_c1[NL] = {};  // the zone's chunks in lzc [c0, c1), from the region end downwards
    bool lz_vmm = false;                 // dpool is a VMM mapping (zone chunks are their own handles)
    std::vector<CUmemAccessDesc> lz_acc; // this card + the peers with peer access
    float nv_credit = 0.f;                       // NVMe admission credit of this card (review r7 M3: per card)
    int nv_t = 0;
    XJob* xj = nullptr;                          // [XJ_PER_CARD] FP4 admissions read from the checkpoint
    uint4 *rx1 = nullptr, *rx2 = nullptr, *rx3 = nullptr;   // R8 attention split: LL12 receive words
    uint4 *rx4 = nullptr, *rx5 = nullptr, *rx5h = nullptr;  // R9 expert split: x / ids / w from the owner; choice rows + headers per sender
    uint4 *rxh = nullptr, *rxc = nullptr, *rxx = nullptr;   // R10 per-card graphs: hand-off (h, pre, sel | cand), head input x
    uint8_t *stg_w[3] = {}, *stg_s[3] = {};      // R8: prefill staging of the full wq_b / wo_a / wo_b of one layer
                                                 // ([dsv4-pf14-owb]: [2] also receives the helpers' wo_b rows; stg_l = -1)
    uint8_t* natbuf[NST] = {};
    float* hrows = nullptr;
    uint8_t* hoff = nullptr;                     // [dsv4-rel-hostx]: decode hand-off rows to the next card (pinned)
    cudaEvent_t ev_off = nullptr;                      // [dsv4-rel-hostx]: this card's distributed-prefill result rows for the owner (pinned, mapped)                     // [dsv4-rel-ram]: pinned re-pack buffers of no-copy GGUF experts (prefill staging)
    int stg_l = -1;
    // [dsv4-pf4-asplit] prefill attention split: cublasLt (shares run with the algorithm of the full slice), workspace,
    // events (0: owner X1 ready, 1: o2 columns ready, 2: o2 pulls done, 3: wo_b rows ready), the selection held (tag)
    cublasLtHandle_t lt = nullptr;
    void* lt_ws = nullptr;
    bool as_init = false;   // [dsv4-pf5-idx] the split's events / stream created (lt may exist without them: the indexer)
    std::map<std::array<int, 7>, cublasLtMatmulAlgo_t> lt_algo;
    cudaEvent_t ev_as[4] = {};   // ([dsv4-pf14-owb]: on the owner [2] = the helpers' wo_b rows in its staging)
    std::vector<cudaEvent_t> ev_x1;   // [dsv4-pf5-x1b] other card: X1 rows of sub-block b landed (exchange stream); [64] + 1
    std::vector<cudaEvent_t> ev_ab;   // (owb: owner [3 b + 1] = its o2 pulls of block b) v2 per sub-block b: [3 b] o2 rows ready, [3 b + 1] pulls done, [3 b + 2] wo_b rows ready
    cudaStream_t ass = nullptr;       // v2: the exchange stream (o2 pulls; the owner's ao pulls)
    long sel_tag = -1;
    float* d_ysl = nullptr;                      // R8: this card's wo_b rows [MAXR][1728] before the push to the owner
    int* h_upd = nullptr;                        // pinned ring of table values for device updates
    unsigned upd_i = 0;
    uint8_t* stage[NST] = {};
    cudaEvent_t st_ready[NST] = {}, st_free[NST] = {};
    std::vector<cudaEvent_t> ev_px;   // R11 distributed prefill: per assigned expert, its rows' result ready (this card)
    // [dsv4-pf3-pfx] prefetch slots at the top of the phase region (slot j at phase + phase_bytes - (j + 1) * slot bytes):
    // copy done (cp) / last read done (the stream that computed the expert, or st for an unused slot)
    std::vector<cudaEvent_t> pf_rdy, pf_fre;
    int pf_ns = 0;                    // slots of this card the current layer's prefetch may use
    // [dsv4-pf12-pfxb] slots [pf_np, pf_ns) lie in the prefetch arena of the pool-tail borrow (pfa, pfa_bytes; long chunks)
    int pf_np = 0;                    // of them at the top of the phase region
    uint8_t* pfa = nullptr;
    size_t pfa_bytes = 0;
    cudaEvent_t ev_pfp = nullptr;     // predictor counts landed in E->h_pfcnt (owner stream)
    cudaEvent_t ev_stg = nullptr;     // the attention shares staged (owner stream): the owner's own prefetch copies wait
    cudaEvent_t ev_pdone = nullptr;   // R12 (review r11 L1): owner's scatter-adds of remote rows done (this card's stream)
    cudaStream_t sxc = nullptr;                    // [dsv4-pf8-ovl] owner compute stream of the early experts
    cublasHandle_t blas2 = nullptr;                // (its cuBLAS handle)
    cudaEvent_t ev_ox = nullptr, ev_oxe = nullptr; // start / all early experts done
    std::vector<cudaEvent_t> ev_oe;                // per early expert
    cudaEvent_t ev_xqd = nullptr;   // [dsv4-pf8-xqh] packed xq pushed to the helpers (owner, on cxs)
    cudaEvent_t ev_hs = nullptr;    // [dsv4-pf9-hsum] this helper's dense partial sum complete (its stream)
    std::vector<cudaEvent_t> ev_hr;   // [dsv4-pf15-hsred] chunks of the helpers' combined sum done (its exchange stream)
    cudaEvent_t ev_cxq = nullptr, ev_cy = nullptr, ev_q = nullptr;   // [dsv4-pf2-cpu]: xq D2H done / the CPU rows' adds done (owner)
    cudaEvent_t ev = nullptr, ev_route = nullptr;
    cudaEvent_t ev_brw = nullptr;                // R19: the borrowed slots' copy-back done (table updates wait on the GPU)
    cudaEvent_t ev_dyn = nullptr;                // R9: admissions' "after the victim's reads" (own event: the enqueue thread
                                                 // records ev at the hand-offs concurrently)
    cudaEvent_t tp0 = nullptr, tp1 = nullptr;
    cudaEvent_t ev_eg1 = nullptr;                // [dsv4-dec5-egz] decode: last read of layer 14's mapped staging
    cudaEvent_t ev_eg = nullptr;                 // last copy out of the pinned Engram staging    // timing (GPU attention half per decode layer)
    // decode host-mapped exchange
    char* d_rpack = nullptr;  // decode routing pack: ids [MAXR*6] | w [MAXR*6] | xq [MAXR][5120] (one D2H)
    char* h_rpack = nullptr;
    float* h_x = nullptr;     // pinned [MAXR][5120] xq of the MoE layer
    int* h_ids = nullptr;     // pinned [MAXR][6]
    float* h_w = nullptr;     // pinned [MAXR][6]
    float* h_y = nullptr;     // pinned [MAXR][5120] cold result (mapped: the combine reads it, no H2D copy)
    float* d_hy_map = nullptr;
    float* d_ycold = nullptr;
    float* d_gu = nullptr;    // [MAXR][6][4608]
    float* d_hh = nullptr;    // [MAXR][6][2304]
    float* d_yx = nullptr;    // [MAXR][6][5120]
    float* d_ydef = nullptr;  // R21 [dsv4-r21-defer] KL emulation: [MAXR][HC][D] deferred lane stream terms
    int8_t* d_q8 = nullptr;   // R21 [dsv4-r21-dp4a]: GGUF pool x / h as int8 per 32 ([MAXR][5120] then [MAXR][6][2304])
    float* d_q8s = nullptr;   // their scales (one per 32)
    int8_t* d_di8 = nullptr;  // R22 [dsv4-r22-di8]: drafter int8 linears' x (int8 per 32, [MAXR][32768]) + scales
    float* d_di8s = nullptr;
    int8_t* d_dq8 = nullptr;  // R22 [dsv4-r22-dmoe8]: drafter MoE x / h as int8 per 32 (last card, with the int8 drafter)
    float* d_dq8s = nullptr;
    float *d_shg = nullptr, *d_shy = nullptr;
    float* part = nullptr;    // attention split partials
    bf16* d_embrow = nullptr;
    uint8_t* d_egrows = nullptr;
    int* d_egoff = nullptr;
    uint8_t* d_egbits = nullptr;
    float* d_cb = nullptr;    // codebooks: cb3 [8], cb2 [4], eg2 [4], eg3 [8], eg4 [16]
    uint8_t* d_gg = nullptr;  // R16 [dsv4-gguf-iq]: the GGUF i-quant grids (gguf_hooks.h), GGUF mode only
    float* d_logits = nullptr;
    float* d_hx = nullptr;    // R9 head split: head input rows [MAXR][5120] (other cards)
    int* d_tok = nullptr;
    float* d_nll = nullptr;
    int* d_cnt = nullptr;     // R10: block counters of k_route_pub (0: routing, 1: predictor)
    bf16* d_xqs = nullptr;    // R10: q_fp8 of the FFN input (one row, bf16), from the fused mHC kernel
    bf16* d_obf = nullptr;    // R10: q_fp8 of the attention output [NH * HD] (one row), the wo_a input
    // R19 [dsv4-r19-pipe]: the second micro-batch's row buffers of a pipelined verify pass (MAXR rows); swapped into
    // the fields above by the enqueue thread while it enqueues that batch's work
    struct MBC {
        float *h = nullptr, *pre = nullptr, *fpre = nullptr, *apre = nullptr, *apost = nullptr, *acomb = nullptr,
              *fpost = nullptr, *fcomb = nullptr, *mix = nullptr, *ss = nullptr, *x = nullptr, *ao = nullptr, *qr = nullptr;
        bf16 *qrb = nullptr, *d_xqs = nullptr, *d_obf = nullptr;
        int* sel = nullptr;
        uint8_t* cand = nullptr;
        char* d_rpack = nullptr;
        float *d_gu = nullptr, *d_hh = nullptr, *d_yx = nullptr, *d_shg = nullptr, *d_shy = nullptr, *d_ysl = nullptr,
              *d_elg = nullptr, *d_epw = nullptr, *d_q8s = nullptr;
        int8_t* d_q8 = nullptr;
        int* d_epid = nullptr;
    } mbc;
};
static void mbc_swap(Card& c) {
    auto& m = c.mbc;
    std::swap(c.h, m.h); std::swap(c.pre, m.pre); std::swap(c.fpre, m.fpre); std::swap(c.apre, m.apre);
    std::swap(c.apost, m.apost); std::swap(c.acomb, m.acomb); std::swap(c.fpost, m.fpost); std::swap(c.fcomb, m.fcomb);
    std::swap(c.mix, m.mix); std::swap(c.ss, m.ss); std::swap(c.x, m.x); std::swap(c.ao, m.ao); std::swap(c.qr, m.qr);
    std::swap(c.qrb, m.qrb); std::swap(c.d_xqs, m.d_xqs); std::swap(c.d_obf, m.d_obf); std::swap(c.sel, m.sel);
    std::swap(c.cand, m.cand); std::swap(c.d_rpack, m.d_rpack); std::swap(c.d_gu, m.d_gu); std::swap(c.d_hh, m.d_hh);
    std::swap(c.d_yx, m.d_yx); std::swap(c.d_shg, m.d_shg); std::swap(c.d_shy, m.d_shy); std::swap(c.d_ysl, m.d_ysl);
    std::swap(c.d_elg, m.d_elg); std::swap(c.d_epw, m.d_epw); std::swap(c.d_epid, m.d_epid);
    std::swap(c.d_q8, m.d_q8); std::swap(c.d_q8s, m.d_q8s);
}

struct EgTable {
    int bits = 0;
    size_t rowbytes = 0;
    uint8_t* base = nullptr;            // [rows][rowbytes]
    size_t nrows = 0;
    std::vector<int64_t> hot_ids;       // sorted rows at hot_bits
    int hot_bits = 0;
    size_t hot_rowbytes = 0;
    uint8_t* hot = nullptr;
};


// Engram rows lossless from the checkpoint on NVMe (user 2026-11-07): O_DIRECT 4 KiB reads (no page cache), a
// persistent I/O pool, rows deduplicated per batch, prefetched from token ids before the layer needs them. A row =
// 256 FP8 bytes (embed.weight) + 8 e8m0 bytes (embed.scale) -> 264 B staged per (token, slot). A failed or short read
// is retried (logged) until it succeeds: never a garbage row.
// [dsv4-pf-egwait]: Engram I/O threads (sync 4 KB O_DIRECT preads; 990 PRO: 16 -> 202 K, 32 -> 359 K, 64 -> 389 K IOPS)
static int eg_threads() { static const int n = getenv("DSV4_EGTHREADS") ? std::max(1, atoi(getenv("DSV4_EGTHREADS"))) : 32; return n; }
static const bool g_egwait_all = getenv("DSV4_EGWAITALL") != nullptr;   // off-switch: layer 1 waits for the whole batch
// [dsv4-pf5-egu] Engram reads by io_uring: a few threads with deep queues (990 PRO random 4 KB O_DIRECT: 32 sync threads
// 350 K IOPS, 4 threads x 128 in flight ~830 K); a failed or short read falls back to the sync retry path. DSV4_EGURING=0:
// off (sync preads); DSV4_EGUTHREADS (8, [dsv4-pf10-egq]) / DSV4_EGUQD (128)
static int eg_uring() { static const int v = getenv("DSV4_EGURING") ? atoi(getenv("DSV4_EGURING")) : 1; return v; }
// [dsv4-pf10-egq] 8 rings (was 4): a new prompt's rows (cold in the drive) arrive 503 -> 322-329 ms after the start
// (2.5K), layer 1 waits 51 -> 12 ms; a repeated prompt's rows ~200 ms either way (more rings do not help: 12 / 16 slower)
// (review pf10 L2) capped by the CPU count when unset: min(8, max(1, hardware threads / 4))
static int eg_uthreads() {
    static const int n = getenv("DSV4_EGUTHREADS") ? std::max(1, atoi(getenv("DSV4_EGUTHREADS")))
                                                   : std::min(8, std::max(1, (int)std::thread::hardware_concurrency() / 4));
    return n;
}
// [dsv4-dec5-egprof] DSV4_EGPROF=1: decode Engram batch timers (review dec5 L2-L4: every clock call behind the flag)
static const bool g_egprof = getenv("DSV4_EGPROF") != nullptr;
// [dsv4-dec5-egcap] reads one ring claims per submit round in a decode-sized batch (< 4096 jobs); 0 = no cap
static int eg_ucap() { static const int n = getenv("DSV4_EGUCAP") ? atoi(getenv("DSV4_EGUCAP")) : 16; return n > 0 ? n : 1 << 30; }
static int eg_uqd() { static const int n = getenv("DSV4_EGUQD") ? std::clamp(atoi(getenv("DSV4_EGUQD")), 1, 1024) : 128; return n; }
struct URing {   // a minimal io_uring (raw syscalls: no liburing on the build host)
    int fd = -1;
    unsigned *shead = nullptr, *stail = nullptr, *smask = nullptr, *sarr = nullptr, *chead = nullptr, *ctail = nullptr,
             *cmask = nullptr;
    io_uring_sqe* sqes = nullptr;
    io_uring_cqe* cqes = nullptr;
    void* m[3] = {};
    size_t ml[3] = {};
    void* dbufs = nullptr;   // buffers of a ring that failed mid-batch (kept until the ring is closed in stop)
    int err = 0;   // (review pf10 L6) errno of a failed init, saved before the cleanup
    bool init(unsigned qd) {
        io_uring_params p{};
        fd = (int)syscall(__NR_io_uring_setup, qd, &p);
        if (fd < 0) { err = errno; return false; }
        {   // (review pf10 M2) IORING_OP_READ (5.6+) must be supported, else every row would fail and retry
            const size_t pb = sizeof(io_uring_probe) + 256 * sizeof(io_uring_probe_op);
            std::vector<uint8_t> pbuf(pb, 0);
            io_uring_probe* pr = (io_uring_probe*)pbuf.data();
            const int rr = (int)syscall(__NR_io_uring_register, fd, IORING_REGISTER_PROBE, pr, 256);
            if (rr < 0 || pr->last_op < IORING_OP_READ || !(pr->ops[IORING_OP_READ].flags & IO_URING_OP_SUPPORTED)) {
                err = rr < 0 ? errno : EOPNOTSUPP;
                close(fd);
                fd = -1;
                return false;
            }
        }
        {   // (review pf10 L1) bound the io-wq workers of a buffered fd (5.15+; errors ignored)
            unsigned mw[2] = {4, 0};
            (void)syscall(__NR_io_uring_register, fd, IORING_REGISTER_IOWQ_MAX_WORKERS, mw, 2);
        }
        ml[0] = p.sq_off.array + p.sq_entries * 4;
        ml[1] = p.cq_off.cqes + p.cq_entries * sizeof(io_uring_cqe);
        ml[2] = p.sq_entries * sizeof(io_uring_sqe);
        const off_t mo[3] = {IORING_OFF_SQ_RING, IORING_OFF_CQ_RING, IORING_OFF_SQES};
        for (int i = 0; i < 3; i++) {
            m[i] = mmap(nullptr, ml[i], PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE, fd, mo[i]);
            if (m[i] == MAP_FAILED) {
                err = errno;
                m[i] = nullptr;
                close_();
                return false;
            }
        }
        char *sq = (char*)m[0], *cq = (char*)m[1];
        shead = (unsigned*)(sq + p.sq_off.head);
        stail = (unsigned*)(sq + p.sq_off.tail);
        smask = (unsigned*)(sq + p.sq_off.ring_mask);
        sarr = (unsigned*)(sq + p.sq_off.array);
        chead = (unsigned*)(cq + p.cq_off.head);
        ctail = (unsigned*)(cq + p.cq_off.tail);
        cmask = (unsigned*)(cq + p.cq_off.ring_mask);
        cqes = (io_uring_cqe*)(cq + p.cq_off.cqes);
        sqes = (io_uring_sqe*)m[2];
        return true;
    }
    void close_() {
        for (int i = 0; i < 3; i++)
            if (m[i]) munmap(m[i], ml[i]), m[i] = nullptr;
        if (fd >= 0) close(fd);
        fd = -1;
        free(dbufs);   // (review pf10 L4) a failed ring's buffers, freed once its fd is closed (in-flight reads cancelled)
        dbufs = nullptr;
    }
};
struct EgReader {
    int fdw[2] = {-1, -1}, fds[2] = {-1, -1};
    int fdm[2] = {-1, -1};   // [dsv4-pf17-egmirror] O_DIRECT fds of verified reflink mirrors of the Q6_K tables (-1 none)
    // [dsv4-pf17-egmirror] review pf17b M2: a mirror that fails a read, or whose size / mtime / ctime changed since the
    // load, is dropped for the rest of the process (the original serves every row; never zero rows for a mirror fault)
    std::atomic<bool> mbad[2] = {false, false};
    struct stat mst[2] = {};
    void mirror_drop(int li, const char* why) {
        if (!mbad[li].exchange(true)) fprintf(stderr, "WARNING engram: mirror of layer %d dropped (%s): original file only from now on\n", li ? 14 : 1, why);
    }
    bool mirror_ok(int li) {   // once per prefill batch: unchanged since the load
        if (fdm[li] < 0 || mbad[li].load()) return false;
        struct stat a{};
        if (fstat(fdm[li], &a) != 0 || a.st_size != mst[li].st_size || a.st_mtim.tv_sec != mst[li].st_mtim.tv_sec ||
            a.st_mtim.tv_nsec != mst[li].st_mtim.tv_nsec || a.st_ctim.tv_sec != mst[li].st_ctim.tv_sec ||
            a.st_ctim.tv_nsec != mst[li].st_ctim.tv_nsec) {
            mirror_drop(li, "changed on disk since the load");
            return false;
        }
        return true;
    }
    size_t offw[2] = {}, offs[2] = {};        // file offset of row 0 of weight / scale, per engram layer
    struct Job { int fd; size_t off; int len; uint8_t* dst; int li; int fda = -1; };   // fda: the original, for a mirrored job
    std::atomic<size_t> done_l[2] = {0, 0};   // [dsv4-pf-egwait]: finished jobs per Engram layer (layer 1 waits for its own)
    size_t n_l[2] = {0, 0};
    std::vector<Job> jobs;
    // (review pf10 H1) claim word = (batch tag << 32) | index; a worker claims only with its batch's tag (CAS), so a
    // stale worker can never take indices of a newer batch against an older batch's count
    std::atomic<uint64_t> next{0};
    std::atomic<size_t> done{0}, njobs{0};
    bool claim(uint64_t seen, size_t nj, size_t& i) {
        const uint64_t tag = (uint32_t)seen;
        uint64_t v = next.load(std::memory_order_acquire);
        while (true) {
            if ((v >> 32) != tag || (v & 0xffffffffu) >= nj) return false;
            if (next.compare_exchange_weak(v, v + 1, std::memory_order_acq_rel, std::memory_order_acquire)) {
                i = (size_t)(v & 0xffffffffu);
                DCHECK(i < jobs.size(), "engram claim %zu of %zu", i, jobs.size());
                return true;
            }
        }
    }
    std::atomic<uint64_t> gen{0};
    std::atomic<bool> quit{false};
    std::vector<std::thread> th;
    std::atomic<long> retries{0}, failed{0}, consec{0};
    std::atomic<long long> tdone_ns{0};   // [dsv4-pf-pprof]: steady-clock time the last job of the batch finished
    // one job's bytes [a0, a0 + n) were read into buf (r = bytes read, < 0 failed): copy the row out or retry synchronously
    void finish(const Job& J, uint8_t* buf, ssize_t r) {
        const size_t a0 = J.off & ~(size_t)4095, a1 = (J.off + J.len + 4095) & ~(size_t)4095;
        bool ok = r >= (ssize_t)(J.off - a0 + J.len);
        int fd = J.fd;
        if (!ok && J.fda >= 0 && consec.load() < 16) {   // review pf17b M2: a mirrored row failed: the original, once
            mirror_drop(J.li, r < 0 ? strerror(errno) : "short read");
            fd = J.fda;
            r = pread(fd, buf, a1 - a0, a0);
            ok = r >= (ssize_t)(J.off - a0 + J.len);
        }
        for (int attempt = 0; !ok && attempt < (consec.load() >= 16 ? 0 : 200); attempt++) {   // = worker()'s retry loop
            retries++;
            if (attempt < 3 || attempt % 100 == 0)
                fprintf(stderr, "engram read retry %d (fd %d off %zu: %s)\n", attempt, fd, a0, r < 0 ? strerror(errno) : "short read");
            if (attempt) std::this_thread::sleep_for(std::chrono::milliseconds(attempt < 10 ? 1 : 50));   // (review pf10 M2)
            r = pread(fd, buf, a1 - a0, a0);
            ok = r >= (ssize_t)(J.off - a0 + J.len);
        }
        if (ok) {
            memcpy(J.dst, buf + (J.off - a0), J.len);
            consec.store(0, std::memory_order_relaxed);
        } else {
            memset(J.dst, 0, J.len);
            long nf = failed++;
            long nc = consec++;
            if (nf < 8) fprintf(stderr, "WARNING engram read failed (fd %d off %zu): zero row used\n", J.fd, a0);
            if (nc == 15) fprintf(stderr, "WARNING engram: 16 rows failed, Engram reads off (zero rows from now on)\n");
        }
        if (done.fetch_add(1, std::memory_order_release) + 1 == njobs.load())
            tdone_ns.store(std::chrono::duration_cast<std::chrono::nanoseconds>(clk::now().time_since_epoch()).count());
        if (g_egprof) {   // [dsv4-dec5-egprof] (review dec5 L2 / L3): latest finish time of the layer, before the count
            const long long t = std::chrono::duration_cast<std::chrono::nanoseconds>(clk::now().time_since_epoch()).count();
            long long o = tl_ns[J.li].load();
            while (o < t && !tl_ns[J.li].compare_exchange_weak(o, t)) {}
        }
        done_l[J.li].fetch_add(1, std::memory_order_release);
    }
    std::atomic<long long> tl_ns[2] = {0, 0};   // [dsv4-dec5-egprof]: time each layer's last row of the batch finished
    // [dsv4-pf5-egu] io_uring worker: up to qd reads in flight (8 KB buffer each: a row may cross a 4 KB page)
    void worker_uring(URing* U, int qd) {
        uint8_t *bufs = nullptr, *sb = nullptr;   // sb: the worker's sync scratch (review pf5 L1: never a ring slot)
        DCHECK(posix_memalign((void**)&bufs, 4096, (size_t)qd * 8192) == 0, "memalign");
        DCHECK(posix_memalign((void**)&sb, 4096, 8192) == 0, "memalign");
        std::vector<int> fb;
        std::vector<size_t> jid(qd);
        uint64_t seen = 0;
        while (true) {
            uint64_t g;
            int spins = 0;
            while ((g = gen.load(std::memory_order_acquire)) == seen) {
                if (quit) {
                    free(bufs);
                    free(sb);
                    U->close_();
                    return;
                }
                if (++spins < 2000) _mm_pause();
                else gen.wait(seen);
            }
            seen = g;
            const size_t nj = njobs.load(std::memory_order_acquire);   // review pf5 L6: this batch's count
            fb.clear();
            for (int i = 0; i < qd; i++) fb.push_back(i);
            int inflight = 0;
            bool more = true;
            while (more || inflight) {
                unsigned tail = *U->stail, add = 0;
                const unsigned capn = nj < 4096 ? (unsigned)eg_ucap() : ~0u;   // [dsv4-dec5-egcap] decode batches: spread over the rings
                while (more && !fb.empty() && add < capn) {
                    size_t i = 0;
                    if (!claim(seen, nj, i)) {
                        more = false;
                        break;
                    }
                    const Job& J = jobs[i];
                    if (consec.load() >= 16) {   // the device counts as gone: zero rows at once (finish's rule)
                        finish(J, sb, -1);
                        continue;
                    }
                    const int b = fb.back();
                    fb.pop_back();
                    jid[b] = i;
                    const size_t a0 = J.off & ~(size_t)4095, a1 = (J.off + J.len + 4095) & ~(size_t)4095;
                    const unsigned idx = (tail + add) & *U->smask;
                    io_uring_sqe* e = &U->sqes[idx];
                    memset(e, 0, sizeof *e);
                    e->opcode = IORING_OP_READ;
                    e->fd = J.fd;
                    e->off = a0;
                    e->addr = (unsigned long long)(uintptr_t)(bufs + (size_t)b * 8192);
                    e->len = (unsigned)(a1 - a0);
                    e->user_data = (unsigned long long)b;
                    U->sarr[idx] = idx;
                    add++;
                }
                __atomic_store_n(U->stail, tail + add, __ATOMIC_RELEASE);
                inflight += (int)add;
                if (!inflight) continue;
                // review pf5 M1: submit every SQE the kernel has not consumed yet (a partial or -EAGAIN submit leaves
                // some between head and tail); wait for a completion only when something was submitted
                const unsigned pend = tail + add - __atomic_load_n(U->shead, __ATOMIC_ACQUIRE);
                const int r = (int)syscall(__NR_io_uring_enter, U->fd, pend, (unsigned)inflight > pend ? 1 : 0,
                                           IORING_ENTER_GETEVENTS, nullptr, 0);
                if (r < 0 && (errno == EAGAIN || errno == EBUSY)) sched_yield();
                if (r < 0 && errno != EINTR && errno != EAGAIN && errno != EBUSY) {
                    // the ring failed: every job in flight is read again synchronously (own buffer: the kernel may still
                    // write the ring buffers, which are left allocated), and this worker reads with pread from now on
                    fprintf(stderr, "WARNING engram io_uring enter: %s: sync reads from now on\n", strerror(errno));
                    std::vector<bool> busy(qd, true);
                    for (int b : fb) busy[b] = false;
                    for (int b = 0; b < qd; b++)
                        if (busy[b]) finish(jobs[jid[b]], sb, -1);
                    free(sb);
                    U->dbufs = bufs;   // (not closed or freed here: requests may be in flight; stop() closes the ring)
                    worker_sync_from(seen);
                    return;
                }
                unsigned h = *U->chead;
                const unsigned ct = __atomic_load_n(U->ctail, __ATOMIC_ACQUIRE);
                for (; h != ct; h++) {
                    const io_uring_cqe* cq = &U->cqes[h & *U->cmask];
                    const int b = (int)cq->user_data;
                    const int res = cq->res;
                    errno = res < 0 ? -res : 0;
                    finish(jobs[jid[b]], bufs + (size_t)b * 8192, res);
                    fb.push_back(b);
                    inflight--;
                }
                __atomic_store_n(U->chead, h, __ATOMIC_RELEASE);
            }
        }
    }
    void worker() { worker_sync_from(0); }
    // the sync worker; seen = the batch generation already taken (a failed io_uring worker continues the current batch)
    void worker_sync_from(uint64_t seen0) {
        uint8_t* buf = nullptr;
        DCHECK(posix_memalign((void**)&buf, 4096, 8192) == 0, "memalign");
        uint64_t seen = seen0;
        bool cont = seen0 != 0;   // finish the current batch's remaining jobs first
        while (true) {
            if (!cont) {
                uint64_t g;
                int spins = 0;
                while ((g = gen.load(std::memory_order_acquire)) == seen) {
                    if (quit) { free(buf); return; }
                    if (++spins < 2000) _mm_pause();
                    else gen.wait(seen);   // futex sleep until the next submit (review r2 M5)
                }
                seen = g;
            }
            cont = false;
            const size_t nj = njobs.load(std::memory_order_acquire);   // review pf5 L6: this batch's count
            while (true) {
                size_t i = 0;
                if (!claim(seen, nj, i)) break;
                const Job& J = jobs[i];
                size_t a0 = J.off & ~(size_t)4095, a1 = (J.off + J.len + 4095) & ~(size_t)4095;
                size_t n = a1 - a0;
                bool ok = false;
                // review r8 M2: after 16 failed rows the device counts as gone: zero rows at once, no more retries
                // review r9 L6: 16 failures in a row (reset on a good row), not over the whole uptime
                int fd = J.fd;
                for (int attempt = 0; attempt < (consec.load() >= 16 ? 0 : 200); attempt++) {   // ~10 s, then degrade
                    ssize_t r = pread(fd, buf, n, a0);
                    if (r >= (ssize_t)(J.off - a0 + J.len)) { ok = true; break; }   // enough bytes (EOF short reads ok, review r2 M4)
                    if (J.fda >= 0 && fd != J.fda) {   // review pf17b M2: a mirrored row failed: the original from now on
                        mirror_drop(J.li, r < 0 ? strerror(errno) : "short read");
                        fd = J.fda;
                        attempt--;
                        continue;
                    }
                    retries++;
                    if (attempt < 3 || attempt % 100 == 0)
                        fprintf(stderr, "engram read retry %d (fd %d off %zu: %s)\n", attempt, fd, a0,
                                r < 0 ? strerror(errno) : "short read");
                    std::this_thread::sleep_for(std::chrono::milliseconds(attempt < 10 ? 1 : 50));
                }
                if (ok) {
                    memcpy(J.dst, buf + (J.off - a0), J.len);
                    consec.store(0, std::memory_order_relaxed);
                }
                else {   // a zero row (the Engram branch adds nothing for this n-gram slot), loudly counted
                    memset(J.dst, 0, J.len);
                    long nf = failed++;
                    long nc = consec++;
                    if (nf < 8) fprintf(stderr, "WARNING engram read failed (fd %d off %zu): zero row used\n", J.fd, a0);
                    if (nc == 15) fprintf(stderr, "WARNING engram: 16 rows failed, Engram reads off (zero rows from now on)\n");
                }
                // review pf M1: `done` first, so a per-layer wait that returned implies these jobs' `done` increments landed
                if (done.fetch_add(1, std::memory_order_release) + 1 == njobs.load())   // [dsv4-pf-pprof]: batch end
                    tdone_ns.store(std::chrono::duration_cast<std::chrono::nanoseconds>(clk::now().time_since_epoch()).count());
                done_l[J.li].fetch_add(1, std::memory_order_release);
            }
        }
    }
    std::vector<URing*> rings;
    bool uring = false;
    void start(int nthreads) {
        if (eg_uring()) {   // [dsv4-pf5-egu] (review pf10 M1): the rings that set up; sync preads only if ring 0 fails
            const int nt = eg_uthreads(), qd = eg_uqd();
            int er = 0;
            for (int i = 0; i < nt; i++) {
                URing* U = new URing;
                if (!U->init((unsigned)qd)) {
                    er = U->err;
                    delete U;
                    break;
                }
                rings.push_back(U);
            }
            if (!rings.empty() && (int)rings.size() < nt)
                fprintf(stderr, "engram: %zu of %d io_uring rings (%s)\n", rings.size(), nt, strerror(er));
            try {   // (review pf10 L3) run with the threads that started
                for (size_t i = 0; i < rings.size(); i++) {
                    th.emplace_back([this, i, qd] { worker_uring(rings[i], qd); });
                    nur++;
                }
            } catch (const std::system_error& ex) {
                fprintf(stderr, "engram: %d of %zu ring threads started (%s)\n", nur, rings.size(), ex.what());
            }
            if (nur > 0) {
                uring = true;
                return;
            }
            for (URing* U : rings) {
                U->close_();
                delete U;
            }
            rings.clear();
            fprintf(stderr, "engram: io_uring unavailable (%s): sync preads\n", strerror(er));
        }
        try {
            for (int i = 0; i < nthreads; i++) th.emplace_back([this] { worker(); });
        } catch (const std::system_error& ex) {
            fprintf(stderr, "engram: %zu of %d reader threads started (%s)\n", th.size(), nthreads, ex.what());
        }
        DCHECK(!th.empty(), "engram: no reader thread could start");
    }
    int nur = 0;   // io_uring worker threads running (rings beyond it are never used)
    void stop() {
        quit = true;
        gen.fetch_add(1);
        gen.notify_all();
        for (auto& t : th) t.join();
        th.clear();
        for (URing* U : rings) {   // (review pf10 L4) rings closed by their worker on quit are no-ops here; a ring that
            U->close_();           // failed mid-batch (or never got a thread) is closed now
            delete U;
        }
        rings.clear();
    }
    // submit rows (both layers: li per row) into dst (264 B per row); returns at once
    bool q6k = false;   // [dsv4-rel-q6k]: one 210 B Q6_K row per job (GGUF tables), else FP8 weight + e8m0 scale rows
    void submit(const std::vector<std::pair<int, int64_t>>& rows, uint8_t* dst) {
        // review pf M1: the previous batch fully done (a per-layer wait may have returned before its last done_l
        // increments; none may land after the counters are reset below)
        while (done.load(std::memory_order_acquire) < njobs || done_l[0].load(std::memory_order_acquire) + done_l[1].load() < njobs) _mm_pause();
        // stale workers of the last batch take nothing (review r2 M1, pf10 H1): the next batch's tag, index 0; no
        // worker holds that tag before gen is bumped below
        next.store((uint64_t)(uint32_t)(gen.load() + 1) << 32);
        jobs.resize(rows.size() * (q6k ? 1 : 2));
        n_l[0] = n_l[1] = 0;
        const bool big = rows.size() >= 4096;
        const bool um[2] = {big && mirror_ok(0), big && mirror_ok(1)};   // review pf17b M2: checked once per prefill batch
        for (size_t i = 0; i < rows.size(); i++) {
            int li = rows[i].first;
            int64_t r = rows[i].second;
            if (q6k) {
                // [dsv4-pf17-egmirror] prefill batches: every second row of a layer through its mirror inode (btrfs
                // serializes O_DIRECT reads per inode: ~490-520 K reads/s per file with 8 rings)
                const bool mi = um[li] && (i & 1);
                jobs[i] = {mi ? fdm[li] : fdw[li], offw[li] + (size_t)r * 210, 210, dst + i * 264, li, mi ? fdw[li] : -1};
                n_l[li]++;
                continue;
            }
            jobs[2 * i] = {fdw[li], offw[li] + (size_t)r * 256, 256, dst + i * 264, li};
            jobs[2 * i + 1] = {fds[li], offs[li] + (size_t)r * 8, 8, dst + i * 264 + 256, li};
            n_l[li] += 2;
        }
        done_l[0].store(0);
        done_l[1].store(0);
        tl_ns[0].store(0);
        tl_ns[1].store(0);
        done.store(0);
        tdone_ns.store(0);
        DCHECK(jobs.size() < 0xffffffffu, "engram batch %zu", jobs.size());
        njobs.store(jobs.size());
        gen.fetch_add(1, std::memory_order_release);
        gen.notify_all();
    }
    void wait() {
        while (done.load(std::memory_order_acquire) < njobs) _mm_pause();
    }
    // [dsv4-pf-egwait]: only Engram layer li's rows (eg_prefetch sorts the rows by layer: layer 1's are taken first)
    void wait(int li) {
        while (done_l[li].load(std::memory_order_acquire) < n_l[li]) _mm_pause();
    }
};

struct HashTabs {
    std::vector<int32_t> tmap;
    int64_t pad = 0;
    int64_t primes[2][3][8], mult[2][4], offs[2][24];
};

struct GgState;   // R16 [dsv4-gguf-iq] (gguf_hooks.h)
struct GmState;   // release prep [dsv4-rel-gguf] (gguf_model.h)
class Engine {
public:
    Options o;
    SafeTensors stf;
    std::vector<Card> cards;
    Layer L[NL];
    CpuLane lane;
    PfCpu pfcpu;                       // [dsv4-pf2-cpu] prefill experts of <= CPUPF_MAX rows on the CPU
    uint16_t* h_cxq = nullptr;         // [dsv4-pf2-cpu] xq of a chunk (pinned) [chunk][D] bf16
    float* h_cy = nullptr;             // [dsv4-pf2-cpu] CPU expert rows (pinned, mapped: the owner adds them in place)
    cudaEvent_t cy_last = nullptr;     // [dsv4-pf2-cpu] the previous layer's ev_cy (h_cy free again)
    // [dsv4-pf3-pfx] prefill prefetch of predicted host experts during the dense phase: the layer whose experts sit in
    // the slots, per expert its card (-1 none) and slot, the slot size, the predictor's counts (pinned)
    int pfx_l = -1;
    // [dsv4-pf15-cone] causal cone: prefill() allows it (no teacher-forced rows wanted); rows the last card holds after
    // forward (= n without the cone); first position a continuation may start from (rollback guard; 0 = anywhere)
    bool cone_ok = false;
    bool cone_lo = false;   // [dsv4-ctx2-xcone] this chunk runs layers 0-19 + layer 20's K/V / compressor only
    // [dsv4-ctx2-xcone] the last chunk fed did (review ctx1 L1); [dsv4-xcall] it persists across prefill calls (the held
    // state's layers 21-39 are pending: only a prefill of >= xcone_need() more rows completes it), saved in the seq blob
    bool xcone_prev = false;
    // [dsv4-xcall] review M1: the state is exact only from this position on (last layer-20-only chunk's end +
    // xcone_need(); 0 = complete): a held state with pos < xpend_until is pending (saved in the seq blob)
    int xpend_until = 0;
    int fwd_rows = 0, cone_min = 0;
    bool mh39_done = false;            // [dsv4-pf17-cone39] layer 39's DSpark capture taken before its K/V-only pass
    std::vector<int> pfx_card, pfx_j;
    size_t pfx_xb = 0;
    int* h_pfcnt = nullptr;
    const bf16* embed = nullptr;  // [V][5120] read in place from the mmapped checkpoint (page cache, no anon copy)
    const uint8_t* embed_q = nullptr;   // [dsv4-rel-q8]: the GGUF's Q8_0 embedding (rows decoded on use)
    bf16* h_emb = nullptr;       // pinned staging [chunk][5120]
    bf16* head = nullptr;        // on the last card (without the head split)
    bool hsplit = false;         // R9: head rows [hr[k], hr[k+1]) on card k
    bf16* head_k[3] = {};
    int hr[4] = {};
    cudaEvent_t ev_hx = nullptr, ev_hd[3] = {};
    cudaEvent_t ev_probe[3] = {};   // review r9 M4: never recorded (context error probe of the host spins); r10 L3: per card
    // R10 whole-step decode graph (one multi-device graph per step parity, captured one step ahead on the enqueue
    // thread; per-step parameters come from the capture through cudaGraphExecUpdate, instantiate on a topology change)
    struct GSlot {
        cudaGraphExec_t ex[3] = {};      // R10: one single-device graph per card (no cross-card edges)
        cudaGraph_t pend[3] = {};      // captured graph whose update failed in a capture beside a running step: it is
                                         // instantiated at the boundary (instantiate / exec destroy while a step graph
                                         // waits for the lane deadlocked: R10-4)
        bool ready = false;              // captured for position s, not launched yet
        int s = -1, seq = 0;
        long nfwd = 0;                   // eager forwards when captured (any other forward invalidates it)
        int tc0[4] = {}, tn0[4] = {}, tc1[4] = {}, tn1[4] = {};   // compressor state before / after the step
    } gs[2];
    bool graph = false;                  // R10 default (DSV4_NOGRAPH=1 off)
    bool graph_cfg = false;              // review rel M2: the configured graph flag (sampling turns it off per request only)
    bool cap_pending = false;            // a capture job was given to the enqueue thread
    std::atomic<bool> cap_cancel{false};
    const int* pf_abort = nullptr;       // [srv-disconnect] the server's cancel flag of the prompt being prefilled
    bool cap_spec = false;                // R19: the pending capture is draft_block's speculative one  // R19 [dsv4-r19-gahead]: an eager forward drops a speculative capture early
    bool dyn_early = false;              // the tier apply of the coming step was done in the last step's head window
    std::atomic<bool> dyn_bnd{false};    // R11 (review r10 M1): eager one-row step applies the tier at its boundary, as the graph
    int cseq = 0;                        // seq of the step being captured
    long nfwd = 0;
    long g_upd = 0, g_inst = 0;          // graph updates / instantiations (log at exit)
    long g_ahead = 0, g_cancel = 0;      // R19: speculative captures in draft_block / captures cancelled
    int* h_tok = nullptr;                // pinned: the step's argmax (graph D2H)
    volatile int* egflag = nullptr;      // mapped [2 * 16]: Engram rows of layer li staged (seq)
    int* d_egflag = nullptr;
    float* fnorm = nullptr;
    EgTable eg[2];
    HashTabs hash;
    bool eg_nvme = false;
    EgReader egr;
    uint8_t* h_egraw = nullptr;              // fetched rows [n * 48][264] (pinned)
    std::vector<int> eg_map;                 // (token, layer, slot) -> unique row index of the fetched batch
    int eg_batch_s = -1, eg_batch_n = 0;     // tokens covered by the submitted batch
    float cb[5][16] = {};        // cb3, cb2, eg2, eg3, eg4
    std::vector<uint8_t*> host_arenas;
    int n_pageable = 0;
    // DSpark draft head (opt-in): 3 stages on the last card, main_proj / norms / Markov head / confidence head
    double hbudget = 0;                      // pinned host budget for experts (bytes)
    int guard_t = 0;                         // memory guard: steps since the last MemAvailable check
    bool guard_empty = false;                // nothing left to drop
    bool dspark = false;
    Layer M[3];
    F8 mp;
    float *mnorm = nullptr, *dnorm = nullptr, *conf_w = nullptr;
    bf16 *mk_emb = nullptr, *mk_head = nullptr;
    int8_t* mk_q = nullptr;   // R22 [dsv4-r22-mk8]: the Markov head as int8 (mk_head freed) + a scale per row
    float* mk_s = nullptr;
    std::vector<bf16> h_noise;               // embedding row of the noise token
    bf16* h_demb = nullptr;                  // pinned staging of the block's embedding rows
    int mh_pos0 = 0, mh_n = 0;               // positions of the captured main hidden rows (d_mh rows 0 .. mh_n - 1)
    int sv_base[4] = {}, sv_m[4] = {};       // position of saved compressor row 0, rows saved
    int sv_pass = -1;                        // [dsv4-srv-stop] pos right after the last verify pass's rollback (sv rows are
                                             // that pass's; -1 after any other forward)
    int sv_pb[4] = {}, sv_pm[4] = {};        // review dec4 M1: that pass's sv_base / sv_m (a speculative capture in
                                             // draft_block rewrites the live fields on the enqueue thread)
    // [dsv4-srv-pesnap] prompt-end snapshot: the sequence state at position snap_pos (the prompt without its last token,
    // so a next turn that drops the reasoning / regenerates still extends it). Window K/V, compressed K/V, index keys and
    // the drafter's K/V are stored by absolute position and never rewritten below pos, Engram hashes come from toks, the
    // peers' rings are rebuilt from the owner's K/V: only the ratio-2 compressor tail rows and a few counters are saved
    int snap_req = -1, snap_pos = -1, snap_cone = 0;
    int snap_got[4] = {};
    bool have_draft = false;
    // R8 adaptive DSpark: running ms per token of verify passes and of plain steps (EMA), current mode and its counter
    double ds_ms_spec = 0, ds_ms_plain = 0;
    double ds_wms = 0, ds_wtok = 0;     // R19 [dsv4-r19-judge]: the passes' ms and tokens since the last judgement
    bool ds_plain = true;
    int ds_cnt = 0;
    double ds_prior = 0;                // R19: plain ms/token measured in earlier sequences of this process
    bool ds_probe = false;              // R19: plain steps measuring the rate after a spec start without a prior
    int lk_hold = 0;                         // R17: calls left without prompt lookup (after a miss)
    int drafts[8] = {};
    float dconf[8] = {};
    float dstat[16] = {};                 // R20 [dsv4-r20-dssel]: per draft row top-1 logit, top-2 logit, log-sum-exp
    double enq_ms = 0;                    // R20: the enqueue thread's last job (ms)
    bool eg_pre = false;                  // R20 [dsv4-r20-egahead]: Engram rows of the next pass's tokens fetched ahead
    int eg_pre_s = 0, eg_pre_n = 0, eg_pre_t[8] = {};
    double ds_lam = 0;                    // R20: tokens per ms (EWMA over passes; 0 = not yet set)
    double ds_T[6] = {22.3, 29.3, 36.8, 42.3, 48.4, 54.1};   // R20: ms per call verifying n drafts (EWMA, n = 0..5)
    // sequence state
    int pos = 0;
    int dyn_clock = 0;
    int seed_dc = 0;                    // [dsv4-rel-tierreset]
    float seed_swap_credit = 0.f, seed_nv_credit = 0.f;   // review rel3 H1: the admission rate limiters too
    int seed_swap_t = 0, seed_nv_t = 0;
    std::vector<std::pair<float, int>> seed_card_nv;
    bool asplit = false;                // R8 attention split on (o.asplit and P2P between every pair)
    bool hostx = false;                 // release prep [dsv4-rel-hostx]: no P2P: the splits' LL12 receive words live in
                                        // mapped pinned host memory (peers store over PCIe to the host, the owner polls
                                        // there); DSV4_NOHOSTX=1 = the splits off without P2P (as before)
    bool psplit = false;                // R16: non-FP4 pool (M23 / GGUF) split over the 3 cards (DSV4_NOPSPLIT=1 off)
    bool xsplit = false;                // R9 expert split on (o.xsplit, asplit, FP4 dynamic tier or none, no cb3 band)
    bool borrow = false;                // R9 workspace borrowing on (o.borrow, xsplit, FP4 dynamic tier)
    bool gborrow = false;               // R19 [dsv4-r19-pool]: borrowing from the split M23 / GGUF pool (tail of each card's pool)
    size_t borrow_bytes = 0;            // carved bytes per card
    int borrow_rows = 0;                // rows of the current carve (0: not borrowed)
    std::vector<std::tuple<int, int, int>> borrowed;   // (layer, pool index, expert) evicted for the current prefill
    unsigned xseq = 0;                  // forward passes (exchange tags)
    int pass_clock = 0;                 // forward passes of decode (= tokens at one row); NVMe admission ages (r7 M2)
    bool dyn_fp4 = false;
    bool m23conv = false;   // R16: host copies converted from the checkpoint at load (pack.json "convert": "m23")
    XFmt pfmt = XF_CB3;   // R16: format of the non-FP4 dynamic pool (cb3, or M23 for a Maya-budget pack)
    GgState* gg = nullptr;   // R16 [dsv4-gguf-iq]: --gguf-experts (expert bytes from Maya-S's GGUF), else null
    GmState* gm = nullptr;   // release prep [dsv4-rel-gguf]: --gguf (the whole model from the GGUF), else null
    std::vector<std::vector<double>> use_cnt;   // [dsv4-rel-gguf]: routed-expert counts (learned profile, decayed)
    bool eg_q6k = false;     // [dsv4-rel-gguf]: Engram rows are the GGUF's Q6_K (210 B), else the checkpoint's FP8
    F8 head_q, head_qk[3];   // [dsv4-rel-q8]: Q8_0 head (whole / per-card rows with the head split)
    void* samp = nullptr;    // [dsv4-rel-sample]: sampler block on the last card (null / !samp_on: greedy)
    bool samp_on = false;
    int* h_emit = nullptr;   // pinned: the sampled acceptance [count, tokens]
    float* h_hx = nullptr;             // [dsv4-rel-hostx]: the head input rows in mapped host memory (no-P2P head split)
    float* h_hl[3] = {};               // the other cards' logit slices on their way to the last card (pinned)
    // R13 swap tier: free host FP4 expert buffers (own = separately allocated), homes for evicted experts without a
    // host copy; an admitted FP4-host expert gives its buffer back once resident
    bool swap = false;
    std::vector<std::pair<uint8_t*, bool>> hfree;
    // [dsv4-rel-xswap]: exclusive swap for the GGUF pool on the 128 GB plan: free pinned homes per expert size (a
    // victim without a host copy goes D2H into one; an admitted expert's copy becomes one at visibility)
    bool gswap = false;
    std::map<size_t, std::vector<uint8_t*>> ghome;
    long ngswap = 0, ngrecyc = 0;
    std::vector<uint8_t> nocopy;   // load: resident pool experts whose host copy is not loaded (over the budget)
    long nswap = 0, nrecyc = 0;
    float swap_credit = 0.f;
    int swap_t = 0;
    XReader xr;
    float nv_credit = 0.f;              // NVMe admission budget (o.nv_cap per token, burst 4 tokens)
    int nv_t = 0;
    std::atomic<bool> nv_off{false};    // checkpoint reads failed / not openable: no NVMe admissions (degrade; r9 L1: both threads)
    std::atomic<long> nv_fail{0};
    bool spec_defer = false;            // DSpark verify pass: tier updates wait for the accepted length (review r6 M1/M2)
    std::vector<std::vector<int>> spec_ids;   // per layer: the pass's routed ids, R x TOPK
    std::map<std::string, int> xfd;     // O_DIRECT fds of the checkpoint shards (FP4 admissions)        // dynamic tier over the FP4 slots (packs with FP4 host copies): pool = the slots
                                 // that hold experts with a host copy, reassigned through d_slot
    // run-ahead decode: per layer host buffers / mapped flags
    bool runahead = true;
    bool ra_multi = true;            // run-ahead also for passes of 2..8 rows (DSpark verify); DSV4_NORAM=1 off
    bool ra_on = false;              // the current forward runs ahead
    // R9 enqueue thread (default, DSV4_NOENQT=1 off; turns run-ahead on): all CUDA launches of a forward from a second host thread, the
    // decode thread only waits for routings and runs the CPU lane (the launches leave the lane's critical path)
    bool enq_thread = false;
    std::thread enq_th;
    std::atomic<int> enq_go{0}, enq_done{0};
    std::function<void()> enq_job;
    bool enq_quit = false;
    int ra_R = 1;                    // its rows
    int seq = 0;
    char* h_rp[NL] = {};
    char* d_rp[NL] = {};             // R10: device view of h_rp (mapped)
    volatile int* rflag = nullptr;   // mapped [NL]
    int* d_rflag = nullptr;
    float* hy[NL] = {};              // mapped [MAXR][5120] per layer
    float* d_hy[NL] = {};
    volatile int* yflag = nullptr;   // mapped [NL]
    bool early = false;              // dsv4-fut [dsv4-fut-early] on (DSV4_EARLY=1, GGUF split pool, run-ahead, no pipe)
    volatile int* h_edec = nullptr;  // mapped [NL][16]: the routing kernel's decision (0 seq, 1 taken)
    int* d_edec = nullptr;
    volatile int* pflag = nullptr;   // mapped [NL]: predicted ids of the layer published (L3 prefetch)
    int* d_pflag = nullptr;
    volatile int* hpred = nullptr;   // mapped [NL][PRED_STRIDE]: predicted ids, rows x TOPK
    int* d_hpred = nullptr;
    volatile int* pflag2 = nullptr;  // R11 early prediction (layer l from layer l-1's FFN input): flag, ids
    int* d_pflag2 = nullptr;
    volatile int* hpred2 = nullptr;
    int* d_hpred2 = nullptr;
    std::vector<int> prev_ids;       // [NL][TOPK] last decode token's experts (statistics)
    int* d_yflag = nullptr;           // dynamic-tier token clock: never reset (review r2 M2)
    // R19 [dsv4-r19-pipe] two-half verify pipeline: the second micro-batch's per-layer handshakes (device views swapped
    // in by the enqueue thread, host views by the decode thread: each thread touches only its own half)
    struct MBD {
        char* d_rp[NL] = {};
        int *d_rflag = nullptr, *d_yflag = nullptr, *d_pflag = nullptr, *d_pflag2 = nullptr, *d_hpred = nullptr,
            *d_hpred2 = nullptr;
        float* d_hy[NL] = {};
        unsigned xseq = 0;
    } mbd;
    struct MBH {
        char* h_rp[NL] = {};
        volatile int *rflag = nullptr, *yflag = nullptr, *pflag = nullptr, *pflag2 = nullptr, *hpred = nullptr,
                     *hpred2 = nullptr;
        float* hy[NL] = {};
    } mbh;
    bool pipe = false;                // R19: pipelined verify passes (DSpark, enqueue thread; opt-in DSV4_PIPE=1)
    int pipe_min = 3;                 // rows from which a pass is split (DSV4_PIPE_MIN)
    int mb_host = 0;                  // decode thread: the batch whose host views are in (0 / 1)
    bool pipe_pass = false;           // review r19 L8: the forward running now is a pipelined pass (decode thread)
    std::vector<int> toks;
    int tail_n[4] = {};          // ratio-2 compressor tail rows (0 / 1) per source
    int tc[4] = {};              // compressed rows per source
    // [dsv4-rel4-p2] compressed K/V pool (per source: pages of KVPG rows, the same page ids on every card). The active
    // sequence is one contiguous run [ab, ab + ares) (its view c.src[si].ckv / ik, so every kernel reads it as before);
    // parked sequences keep their runs (a switch back activates them in place, or moves them on the device) until the
    // active one grows into them, then they go to host RAM (or are dropped under memory pressure)
    int kvP[4] = {}, kvcap[4] = {}, ab[4] = {}, ares[4] = {};
    // [dsv4-ctx-lazy] (Card::lzb): pages / rope positions mapped on every card now (kvM, ropeM) and at load (kvB, ropeB)
    bool lz = false;
    int kvM[4] = {}, kvB[4] = {}, ropeM = 0, ropeB = 0;
    std::vector<float> lz_rope[4];       // host rope tables for the full window (cos 0, sin 0, cos 1, sin 1)
    std::vector<std::tuple<int, int, int>> lz_ev;   // (layer, global slot, expert) evicted for the loan
    std::vector<std::tuple<int, int, int>> lz_ev_lent;   // lz_ev entries of slots a give-back kept lent (review L9)
    clk::time_point lz_t_long{};         // the last time the active sequence held pages above the base
    int ctx_cap = 0;                     // [dsv4-rel8-ctxcap] > 0: the load could map only this many positions (maxctx at ready)
    bool lz_seed = false;                // [dsv4-rel8-giveback] --tier-reset: a give-back refills freed slots with their seeds
    long lz_cnt[4] = {};                 // grows, shrinks, slots evicted, chunks from fresh memory
    std::vector<int> kvown[4];   // page owner: 0 free, -1 the active sequence, rid > 0 a parked one
    // [dsv4-rel5-p3] a host copy of evicted rows: pinned when possible (async copies on every card's link), pageable as
    // the fallback; n = bytes in use, cap = the buffer's size (a reused buffer may be larger)
    struct HostKv {
        uint8_t* p = nullptr;
        size_t n = 0, cap = 0;
        bool pin = false;
        bool empty() const { return n == 0; }
        size_t size() const { return p ? cap : 0; }   // review rel5 M2: the archive counts the buffer, not the rows
    };
    struct KvRes {
        int base[4] = {}, np[4] = {}, tc[4] = {};
        HostKv host;                 // after eviction: the rows per source (ckv then ik) from the owner card
        bool dead = false;           // evicted without a host copy (memory pressure)
    };
    std::map<int, KvRes> kvres;
    int kv_rid = 0;
    size_t kv_host = 0;
    long kv_n[5] = {};           // in place, moved, from host, evicted to host, dropped (per conversation; moved: per run)
    std::multimap<size_t, HostKv> kvfree;   // [dsv4-rel5-p3] freed host buffers kept for the next eviction (by cap)
    size_t kvfree_b = 0, kv_pinned = 0;     // cached bytes; pinned bytes held (in use + cached)
    long kvh_n[3] = {};                     // host buffers reused, allocated pinned, allocated pageable
    double kv_floor_gib = 12.0;  // an eviction keeps a host copy only while MemAvailable stays above this (C API: its floor)
    // decode scratch on host
    std::vector<CpuJob> jobs;
    uint8_t* h_egrows = nullptr;
    uint8_t* h_egdz = nullptr;   // [dsv4-dec5-egz] decode staging, mapped: [2 Engram layers][MAXR x 24 rows][264 B]
    int* h_egoff = nullptr;
    uint8_t* h_egbits = nullptr;
    int* h_rows = nullptr;
    float* h_rw = nullptr;
    int* h_pids = nullptr;
    StepTimes* tm = nullptr;
    int dump_step = -1;          // -1: prefill dumps (when o.dump set), >= 0: decode step index
};

// review r9 H1: global slot g lies in its card's borrowable tail (overwritten by long prefills): its expert must stay in
// the pool (evicted / restored by borrow_ws), so the memory guard and the host RAM plan never make it static
static bool in_tail(const Engine* E, int g) {
    if (!E->borrow || g < 0) return false;
    const Card& k = E->cards[gs_card(g)];
    return gs_idx(g) >= k.used_slots - k.nborrow;
}

// review r9 M4: bounded host spin on a device-written (mapped) flag: every ~1 ms of spinning after the first second it
// probes the CUDA context (a GPU trap is sticky: any call returns it) and aborts with the error; 120 s without the
// flag and without an error aborts too (never a silent hang)
static void hspin(Engine* E, const volatile int* f, int v, const char* what);
// R10: true on the thread that is capturing the decode step graph (its launches go to the capture streams; host-side work
// of the step is done at the step boundary instead)
static thread_local bool tl_cap = false;
static int g_wl = -1;   // layer the decode thread is processing (diagnostics)
// R11 predictor records (DSV4_PREDREC=PREFIX): per decode position (mod REC_T) and layer the FFN router input x, the
// predictor input xp (fp16, written by the GPU into mapped host memory), the routing ids, the predicted ids, the
// prefetched ids and the residency at routing time; dumped to PREFIX_<k>.bin at each reset / destroy
constexpr int REC_T = 512;
struct RecRow { short ids[TOPK], pred[TOPK], pref[TOPK]; short npref, pad; uint8_t res[NE / 8]; };
static const char* g_rec = getenv("DSV4_PREDREC");
static __half* g_rec_x = nullptr;     // mapped [REC_T][NL][2][D]
static __half* g_rec_dx = nullptr;
static std::vector<RecRow> g_rec_rows;   // [REC_T][NL]
static thread_local int tl_rec_s = 0;   // position of the layer being enqueued (capture runs one step ahead)
static const bool g_admlog = getenv("DSV4_ADMLOG") != nullptr;   // R11: admission issue / apply log (M1 diagnosis)
// dsv4-fut [dsv4-fut-rec]: routing records of every pass (DSV4_FUTREC=PREFIX, off = no work): int32 words, tagged
//   1 layer : l, R, pass_clock, dyn_clock, npred, ids[R*TOPK] (-1: left out), pred[npred], res[12], pend[12] (bitmaps)
//   2 commit: R, accepted rows, pass_clock, dyn_clock, pos, lookup, row tokens[R], accepted tokens[accepted rows]
//   3 prefill chunk: l, n, dyn_clock, counts[NE]
// kept in host memory, written to PREFIX_<k>.bin at each reset / destroy
static const char* g_fut = getenv("DSV4_FUTREC");
static std::vector<int32_t> g_fut_buf;
static std::mutex g_fut_mu;
static int g_fut_k = 0;
static int g_rec_s = 0, g_rec_lo = 1 << 30, g_rec_hi = -1, g_rec_k = 0;
__global__ void k_rec(const float* x, __half* dst) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < D) dst[i] = __float2half_rn(x[i]);
}
static int g_htrace = getenv("DSV4_HTRACE") ? atoi(getenv("DSV4_HTRACE")) : -1;   // R10: per-layer hashes at this decode pos
static unsigned long long* g_hh = nullptr;   // mapped [NL]: hash of h after each layer (graph)
static unsigned long long g_lx[NL], g_ly[NL];
static int g_lid[NL][TOPK];
static unsigned long long hhash(const void* p, size_t n) {
    const uint32_t* w = (const uint32_t*)p;
    unsigned long long h = 0;
    for (size_t i = 0; i < n; i++) {
        unsigned long long v = (unsigned long long)w[i] * 0x9E3779B97F4A7C15ull + i;
        h += v ^ (v >> 29);
    }
    return h;
}
static inline int kseq(const Engine* E) { return tl_cap ? E->cseq : E->seq; }   // seq baked into the step's kernels
// R19 [dsv4-r19-pipe]: the micro-batch being enqueued (enqueue thread) and its first row in the pass
static thread_local int tl_mb = 0, tl_row0 = 0;
static thread_local bool tl_part1 = false;   // layer_fwd stops after the expert launches (the combine comes later)
static void mbd_swap(Engine* E) {   // enqueue thread: device views + exchange tags
    auto& m = E->mbd;
    for (int l = 0; l < NL; l++) { std::swap(E->d_rp[l], m.d_rp[l]); std::swap(E->d_hy[l], m.d_hy[l]); }
    std::swap(E->d_rflag, m.d_rflag); std::swap(E->d_yflag, m.d_yflag); std::swap(E->d_pflag, m.d_pflag);
    std::swap(E->d_pflag2, m.d_pflag2); std::swap(E->d_hpred, m.d_hpred); std::swap(E->d_hpred2, m.d_hpred2);
    std::swap(E->xseq, m.xseq);
    for (auto& c : E->cards) mbc_swap(c);
}
static void mbh_swap(Engine* E) {   // decode thread: host views
    auto& m = E->mbh;
    for (int l = 0; l < NL; l++) { std::swap(E->h_rp[l], m.h_rp[l]); std::swap(E->hy[l], m.hy[l]); }
    std::swap(E->rflag, m.rflag); std::swap(E->yflag, m.yflag); std::swap(E->pflag, m.pflag);
    std::swap(E->pflag2, m.pflag2); std::swap(E->hpred, m.hpred); std::swap(E->hpred2, m.hpred2);
    E->mb_host ^= 1;
}
static void cap_wait(Engine* E);

// ---------------------------------------------------------------- small launch helpers

static void q_fp8(cudaStream_t s, const float* x, int ldx, int n, int K, float* o, int ldo, bf16* ob, int ldb,
                  int brow = 0, int bmask = -1) {
    size_t warps = (size_t)n * (K / 32);
    k_q_fp8<<<cdiv(warps * 32, 256), 256, 0, s>>>(x, ldx, n, K, o, ldo, ob, ldb, brow, bmask);
}
static void q_fp4(cudaStream_t s, float* x, int ldx, int n, int K, int blk, int e4, bf16* ob, int ldb) {
    size_t warps = (size_t)n * (K / 32);
    k_q_fp4<<<cdiv(warps * 32, 256), 256, 0, s>>>(x, ldx, n, K, blk, e4, ob, ldb);
}
static void rms(cudaStream_t s, const float* x, int ldx, const float* w, float* y, int ldy, int n, int K) {
    if (n > 0) k_rms<<<n, 256, 0, s>>>(x, ldx, w, y, ldy, K);
}
static void rope(cudaStream_t s, float* x, int ldx, int n, int nh, int hd, const float* c, const float* sn, int pos0,
                 int pstride, int inv) {
    if (n > 0) k_rope<<<n, 256, 0, s>>>(x, ldx, n, nh, hd, c, sn, pos0, pstride, inv);
}
// [dsv4-pf3-lazy] debug DSV4_BLASLOG=1: cuBLAS calls whose host enqueue took > 5 ms (lazy kernel loads), logged
static const bool g_blaslog = getenv("DSV4_BLASLOG") != nullptr;
static void blas_slow(const char* what, clk::time_point tb, int n, int N, int K) {
    const double ms = std::chrono::duration<double, std::milli>(clk::now() - tb).count();
    int d = -1;
    cudaGetDevice(&d);
    if (ms > 5.0) logf_("BLASLOG %s dev %d n %d N %d K %d: enqueue %.1f ms\n", what, d, n, N, K, ms);
}
// C[n][N] (ldc) = X[n][K] (ldx) . W[N][K]^T (ldw), bf16 in, fp32 out
static void gemm_bf16(cublasHandle_t h, const bf16* X, int ldx, const bf16* W, int ldw, float* C, int ldc, int n, int N,
                      int K, float beta = 0.f) {
    float alpha = 1.f;
    const auto tb = g_blaslog ? clk::now() : clk::time_point{};
    auto r = cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, N, n, K, &alpha, W, CUDA_R_16BF, ldw, X, CUDA_R_16BF, ldx,
                          &beta, C, CUDA_R_32F, ldc, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    if (g_blaslog) blas_slow("gemm_bf16", tb, n, N, K);
    DCHECK(r == CUBLAS_STATUS_SUCCESS, "cublasGemmEx %d (n %d N %d K %d)", (int)r, n, N, K);
}
static void sgemm(cublasHandle_t h, const float* X, int ldx, const float* W, int ldw, float* C, int ldc, int n, int N,
                  int K) {
    float alpha = 1.f, beta = 0.f;
    const auto tb = g_blaslog ? clk::now() : clk::time_point{};
    auto r = cublasSgemm(h, CUBLAS_OP_T, CUBLAS_OP_N, N, n, K, &alpha, W, ldw, X, ldx, &beta, C, ldc);
    if (g_blaslog) blas_slow("sgemm", tb, n, N, K);
    DCHECK(r == CUBLAS_STATUS_SUCCESS, "cublasSgemm %d", (int)r);
}


#define GEMV_DISPATCH(KERN, R, GRID, STREAM, ...)                                      \
    switch (R) {                                                                       \
        case 1: KERN<1><<<GRID, 256, 0, STREAM>>>(__VA_ARGS__); break;                 \
        case 2: KERN<2><<<GRID, 256, 0, STREAM>>>(__VA_ARGS__); break;                 \
        case 3: KERN<3><<<GRID, 256, 0, STREAM>>>(__VA_ARGS__); break;                 \
        case 4: KERN<4><<<GRID, 256, 0, STREAM>>>(__VA_ARGS__); break;                 \
        case 5: KERN<5><<<GRID, 256, 0, STREAM>>>(__VA_ARGS__); break;                 \
        case 6: KERN<6><<<GRID, 256, 0, STREAM>>>(__VA_ARGS__); break;                 \
        case 7: KERN<7><<<GRID, 256, 0, STREAM>>>(__VA_ARGS__); break;                 \
        case 8: KERN<8><<<GRID, 256, 0, STREAM>>>(__VA_ARGS__); break;                 \
        default: DCHECK(false, "gemv rows %d", R);                                     \
    }

#define GEMV_DISPATCH_LPR(LPR, R, GRID, SMEM, STREAM, ...)                             \
    switch (R) {                                                                       \
        case 1: k_gemv_f8<1, LPR><<<GRID, 256, SMEM, STREAM>>>(__VA_ARGS__); break;    \
        case 2: k_gemv_f8<2, LPR><<<GRID, 256, SMEM, STREAM>>>(__VA_ARGS__); break;    \
        case 3: k_gemv_f8<3, LPR><<<GRID, 256, SMEM, STREAM>>>(__VA_ARGS__); break;    \
        case 4: k_gemv_f8<4, LPR><<<GRID, 256, SMEM, STREAM>>>(__VA_ARGS__); break;    \
        case 5: k_gemv_f8<5, LPR><<<GRID, 256, SMEM, STREAM>>>(__VA_ARGS__); break;    \
        case 6: k_gemv_f8<6, LPR><<<GRID, 256, SMEM, STREAM>>>(__VA_ARGS__); break;    \
        case 7: k_gemv_f8<7, LPR><<<GRID, 256, SMEM, STREAM>>>(__VA_ARGS__); break;    \
        case 8: k_gemv_f8<8, LPR><<<GRID, 256, SMEM, STREAM>>>(__VA_ARGS__); break;    \
        default: DCHECK(false, "gemv rows %d", R);                                     \
    }
#define GEMV_DISPATCH_T(LPR, R, GRID, SMEM, STREAM, ...)                               \
    switch (R) {                                                                       \
        case 2: k_gemv_f8t<2, LPR><<<GRID, 256, SMEM, STREAM>>>(__VA_ARGS__); break;   \
        case 3: k_gemv_f8t<3, LPR><<<GRID, 256, SMEM, STREAM>>>(__VA_ARGS__); break;   \
        case 4: k_gemv_f8t<4, LPR><<<GRID, 256, SMEM, STREAM>>>(__VA_ARGS__); break;   \
        case 5: k_gemv_f8t<5, LPR><<<GRID, 256, SMEM, STREAM>>>(__VA_ARGS__); break;   \
        case 6: k_gemv_f8t<6, LPR><<<GRID, 256, SMEM, STREAM>>>(__VA_ARGS__); break;   \
        case 7: k_gemv_f8t<7, LPR><<<GRID, 256, SMEM, STREAM>>>(__VA_ARGS__); break;   \
        case 8: k_gemv_f8t<8, LPR><<<GRID, 256, SMEM, STREAM>>>(__VA_ARGS__); break;   \
        default: DCHECK(false, "gemv rows %d", R);                                     \
    }
#define GEMV_DISPATCH_SM(KERN, R, GRID, SMEM, STREAM, ...)                             \
    switch (R) {                                                                       \
        case 1: KERN<1><<<GRID, 256, SMEM, STREAM>>>(__VA_ARGS__); break;              \
        case 2: KERN<2><<<GRID, 256, SMEM, STREAM>>>(__VA_ARGS__); break;              \
        case 3: KERN<3><<<GRID, 256, SMEM, STREAM>>>(__VA_ARGS__); break;              \
        case 4: KERN<4><<<GRID, 256, SMEM, STREAM>>>(__VA_ARGS__); break;              \
        case 5: KERN<5><<<GRID, 256, SMEM, STREAM>>>(__VA_ARGS__); break;              \
        case 6: KERN<6><<<GRID, 256, SMEM, STREAM>>>(__VA_ARGS__); break;              \
        case 7: KERN<7><<<GRID, 256, SMEM, STREAM>>>(__VA_ARGS__); break;              \
        case 8: KERN<8><<<GRID, 256, SMEM, STREAM>>>(__VA_ARGS__); break;              \
        default: DCHECK(false, "gemv rows %d", R);                                     \
    }

static bool E_fuse_off = getenv("DSV4_NOFUSE") != nullptr;
// R20 [dsv4-r20-idxskip] ❌ as a default (route wait -0.2 ms per call; an attention order change): opt-in DSV4_IDXSKIP=1
static const bool g_topk_old_h = getenv("DSV4_IDXSKIP") == nullptr;   // true: the R19 top-k order, no indexer skip
static const bool g_dssel = getenv("DSV4_DSSEL") != nullptr;   // R20 [dsv4-r20-dssel] ❌ as a default: verify length by the selector (opt-in)
// R20 [dsv4-r20-xh16]: bf16 xq in the run-ahead pack; review r20 M2: only the fused routing tail writes bf16, so one
// predicate for the kernel, the lane and the trace hash (DSV4_NOROUTEPUB=1 = the fp32 pack)
static const bool g_xh16 = getenv("DSV4_NOXH16") == nullptr && getenv("DSV4_NOROUTEPUB") == nullptr;
static const int g_spin_us = getenv("DSV4_SPIN_US") ? atoi(getenv("DSV4_SPIN_US")) : 0;   // R20 diagnostic: delay per layer (verify passes)
// R20 [dsv4-r20-ggsc] (cache agent's replay, wt/dsv4-cache): GGUF pool scores seeded 1 (was 3), decay 0.99
// per token (was 0.999), the prompt's routing fed into the scores as in the FP4 tier; A/B: DSV4_GGSC=1
static const bool g_ggsc = getenv("DSV4_GGSC") != nullptr;
// dsv4-fut [dsv4-fut-early] (replay wt/dsv4-fut): the multi-row predictor's ids drive an in-pass admission of
// one expert per layer into a spare pool slot of the owner card; the routing kernel takes it only if the copy has landed
// (else it becomes an ordinary admission, visible 2 passes later). DSV4_EARLY=1 (A/B)
static const bool g_early_env = getenv("DSV4_EARLY") != nullptr;
static long g_early_n[5];   // requests, landed in time, late, routing already published at the request, copy done when the host read the decision
static bool E_embm_off = getenv("DSV4_NOEMBM") != nullptr;   // R17: decode embedding rows read in place off
static bool E_f8t_off = getenv("DSV4_NOF8T") != nullptr;    // R18: K-tiled multi-row FP8 GEMV off
static bool E_dsplainstart = getenv("DSV4_DSSPECSTART") == nullptr;   // R19 [dsv4-r19-specstart] ❌: spec start opt-in (DSV4_DSSPECSTART=1)
static bool E_shs_off = getenv("DSV4_NOSHS") != nullptr;     // R17: shared expert on the side stream off
// R15: decode qr quantized once (FP8-rounded bf16, the GEMV's own rounding: exact) for wq_b on every card, X1 carries it
// (half the qr bytes); DSV4_NOQRB=1: fp32 qr, each wq_b block quantizes it. linq takes a bf16 input only with the fusion on
static bool E_qrb_off = getenv("DSV4_NOQRB") != nullptr;
static bool qrb_on() { return !E_qrb_off && !E_fuse_off; }
static bool E_f32k_off = getenv("DSV4_NOF32K") != nullptr;   // split-K fp32 GEMV off (old kernel)
static bool E_q1_off = false;
static bool E_qkv1_off = getenv("DSV4_NOQKV1") != nullptr;   // R8: one input quantization for wq_a + wkv (decode)
static bool E_moev2 = getenv("DSV4_MOEV2") != nullptr;   // round-2 hot-expert GEMV (uncoalesced x reads, ~240 GB/s)
static bool E_ycopy = getenv("DSV4_YCOPY") != nullptr;   // old H2D copy of the cold result   // set q1 0: input quantization in every GEMV block again (round 3)
static bool E_ptime = getenv("DSV4_PTIME") != nullptr;      // prefill: per-layer timing log (syncs each layer)
static const bool E_attn2_off = getenv("DSV4_NOATTN2") != nullptr;   // [dsv4-pf-attn2] off: the old prefill attention kernel
static bool E_rp_off = getenv("DSV4_NOROUTEPUB") != nullptr;  // R10: fused routing tail off (k_route + q_fp8 + D2H + k_flag)
static bool E_hcf_off = getenv("DSV4_NOHCF") != nullptr;      // R10: fused mHC boundaries off
static const bool g_defer = getenv("DSV4_DEFER") != nullptr;   // R21 [dsv4-r21-defer] KL emulation (opt-in)
static const bool E_hcfr_off = getenv("DSV4_NOHCFR") != nullptr;   // R20 [dsv4-r20-hcfr]: fused mHC for 2 .. MAXR rows off
static bool hcfr_on(int n) { return n > 1 && n <= MAXR && !E_hcfr_off && !E_hcf_off; }
static const bool E_hcfr4_off = getenv("DSV4_NOHCFR4") != nullptr;   // [dsv4-dec2-hcfr2] off: k_hc_fused_r (rows in a loop)
static bool E_rf_off = getenv("DSV4_NOROPEF") != nullptr;     // R10: ropes inside the split attention kernels off
static bool E_kvt_off = getenv("DSV4_NOKVT") != nullptr;      // R10: fused window-KV tail off
static bool E_wqkv_off = getenv("DSV4_NOWQKV") != nullptr;    // R10: wq_a + wkv as one GEMV off
static bool E_gup_off = getenv("DSV4_NOGUP") != nullptr;      // R10: graph upload ahead of the launch off
static bool E_eapp_off = getenv("DSV4_NOEAPP") != nullptr;    // R10: next step's tier apply in the head-wait window off
static bool E_oq_off = getenv("DSV4_NOOQ") != nullptr;        // R10: wo_a input quantized by the attention store off
static bool E_gluq_off = getenv("DSV4_NOGLUQ") != nullptr;    // R10: fused expert GLU + q_fp8 off
// [dsv4-pf2-fx]: prefill routed experts of <= FXMAX rows as the fused IQ kernels (k_fx13 + k_fx2) instead of
// dequant + cuBLAS; DSV4_NOFX=1 off, DSV4_FXMAX=rows (default 448 since [dsv4-pf10-fxl] (128 with DSV4_NOFXL=1),
// clamped to 0 .. 65536; review pf11 L9)
static bool E_fx_off = getenv("DSV4_NOFX") != nullptr;
// [dsv4-pf10-fxl] DSV4_MHIST=1: histogram of the GPU routed experts' rows (count, rows) per m bin, printed at destroy
static const bool E_mhist = getenv("DSV4_MHIST") != nullptr;
static std::atomic<long long> g_mh_n[8], g_mh_r[8], g_mh_f[4][2];   // review pf11 M2: + 113..448 rows per w1 format
static inline void mhist_add(int m, XFmt f) {
    if (!E_mhist) return;
    const int b = m <= 8 ? 0 : m <= 32 ? 1 : m <= 112 ? 2 : m <= 256 ? 3 : m <= 448 ? 4 : m <= 1024 ? 5 : m <= 2048 ? 6 : 7;
    g_mh_n[b]++;
    g_mh_r[b] += m;
    if (m > 112 && m <= 448) {
        const int g = mat_fmt(f, 0) == XF_IQ2XXS ? 0 : mat_fmt(f, 0) == XF_IQ3XXS ? 1 : mat_fmt(f, 0) == XF_IQ2S ? 2 : 3;
        g_mh_f[g][0]++;
        g_mh_f[g][1] += m;
    }
}
// [dsv4-pf10-fxl]: default 448 (k_fxm above DSV4_FXLMIN rows; cuBLAS wins beyond ~450 rows), clamped to 0 .. 65536
static int E_fx_max = std::clamp(getenv("DSV4_FXMAX") ? atoi(getenv("DSV4_FXMAX")) : (getenv("DSV4_NOFXL") ? 128 : 448), 0, 65536);
// [dsv4-pf2-cpu]: prefill host experts of <= DSV4_CPUPF_MAX rows (default 8, 0 = off) may run on the CPU (exact
// bf16 weights, fp32 dot: order-only change), placed by the [dsv4-pf-lb] time model with the CPU as a 4th "card" of
// DSV4_CPUPF_BW GB/s (default 70) for all its threads; DSV4_NOCPUPF=1 off
static const int E_cpu_max = getenv("DSV4_NOCPUPF") ? 0 : std::clamp(getenv("DSV4_CPUPF_MAX") ? atoi(getenv("DSV4_CPUPF_MAX")) : 8, 0, 8);
static const double E_cpu_bw = std::clamp(getenv("DSV4_CPUPF_BW") ? atof(getenv("DSV4_CPUPF_BW")) : 70.0, 1.0, 1000.0);   // (review cpu L5)
static bool pfcpu_on(const Engine* E) { return E_cpu_max > 0 && E->gg && E->o.cpu_threads >= 4; }
static double g_eg_wait = 0, g_route_sync = 0;
// decode GPU profile (DSV4_DPROF=1): events at fixed points of each layer on every 8th decode step
static bool E_dprof = getenv("DSV4_DPROF") != nullptr;
constexpr int NPP = 11;
static const char* PPN[NPP] = {"start", "engram", "hc mixes + pre", "q / kv proj", "compressor + indexer", "wq_b + attention",
                               "wo_a + wo_b", "hc post + ffn mixes + pre", "router + route D2H", "GPU hot + shared experts", "lane wait + combine + post"};
static cudaEvent_t g_pe[NL][NPP];
static bool g_pon = false;
static double g_pacc[NPP] = {};
static double g_paccR[MAXR + 1][NPP] = {};   // R20: verify passes by rows (DSV4_DPROF)
static int g_pnR[MAXR + 1] = {};
static int g_psteps = 0;
// R9 nsys window (DSV4_NSYS=a,b: cudaProfilerStart at decode step a, stop at b; NVTX host ranges while on)
static int g_ns_a = -1, g_ns_b = -1, g_dstep = 0;
static bool g_nvtx = false;
struct NvR {
    bool on;
    NvR(const char* n) : on(g_nvtx) { if (on) nvtxRangePushA(n); }
    ~NvR() { if (on) nvtxRangePop(); }
};
// R22 drafter profile (DSV4_DRPROF=1): cudaEvents on the last card's stream at fixed points of draft_window /
// draft_block (every call), read after the block's own sync; per point mean ms (GPU time between consecutive points,
// launch gaps of the eager stream included) + the host enqueue time of the block
static const bool E_drprof = getenv("DSV4_DRPROF") != nullptr;
constexpr int NDP = 96;
static cudaEvent_t g_dev[NDP];
static const char* g_dlab[NDP];
static int g_dn = -1;                 // -1: closed
static std::vector<std::pair<std::string, double>> g_dpacc;   // label -> sum ms (first-seen order)
static long g_dcalls = 0;
static double g_denq = 0, g_dwall = 0;
static std::chrono::steady_clock::time_point g_dt0;
static void dp_open(cudaStream_t st) {
    if (!E_drprof || g_dn >= 0) return;
    if (!g_dev[0]) for (int i = 0; i < NDP; i++) DCK(cudaEventCreate(&g_dev[i]));
    g_dn = 0;
    g_dt0 = std::chrono::steady_clock::now();
    DCK(cudaEventRecord(g_dev[g_dn], st));
    g_dlab[g_dn++] = "start";
}
static inline void DPT(const char* lab, cudaStream_t st) {
    if (g_dn < 0 || g_dn >= NDP) return;
    DCK(cudaEventRecord(g_dev[g_dn], st));
    g_dlab[g_dn++] = lab;
}
static cudaEvent_t g_dhe[3][2];   // R22: the head GEMV share of cards 0 / 1 on their own streams (draft calls)
static bool g_dhon = false;
// R23 [dsv4-r23-hgap] DSV4_HPROF=1: every eager head GEMV (target verify / one-row, drafter) timed per card on its own
// stream by cudaEvents (only the GEMV: after the x hand-off wait), read one call later (no host sync added)
static const bool E_hprof = getenv("DSV4_HPROF") != nullptr;
struct HProf { cudaEvent_t e[3][2] = {}; int kind = -1, m = 0, nc = 0; };   // kind 0 target, 1 drafter
static HProf g_hp[2];
static int g_hpi = 0;
static double g_hpacc[2][MAXR + 1][3] = {};
static long g_hpn[2][MAXR + 1] = {};
// R23 [dsv4-r23-hstore]: the split head's m >= 2 rows with k_gemv_bf16s (block-staged 32-128 B stores into the last
// card's logits instead of 4 B remote stores per (row, x row); RPW rows per warp by the microbench, hgap/b3-b4);
// every row bitwise = k_gemv_bf16 = the one-row head. DSV4_NOHSTORE=1 = k_gemv_bf16
static const bool g_hstore = getenv("DSV4_NOHSTORE") == nullptr;
template <int R, int RPW>   // R23 (review r23 L2): the grid from RPW (8 RPW rows per block), never written twice
static void launch_bf16s(const bf16* W, int N, const float* x, float* y, cudaStream_t st) {
    k_gemv_bf16s<R, RPW><<<cdiv(N, 8 * RPW), 256, 0, st>>>(W, N, D, x, D, y, V);
}
static void head_rows_gemv(int m, const bf16* W, int N, const float* x, float* y, cudaStream_t st) {
    if (!g_hstore || m == 1) {
        GEMV_DISPATCH(k_gemv_bf16, m, cdiv(N, 8), st, W, N, D, x, D, y, V);
        return;
    }
    switch (m) {
        case 2: launch_bf16s<2, 1>(W, N, x, y, st); break;
        case 3: launch_bf16s<3, 4>(W, N, x, y, st); break;
        case 4: launch_bf16s<4, 4>(W, N, x, y, st); break;
        case 5: launch_bf16s<5, 4>(W, N, x, y, st); break;
        case 6: launch_bf16s<6, 2>(W, N, x, y, st); break;
        case 7: launch_bf16s<7, 4>(W, N, x, y, st); break;
        case 8: launch_bf16s<8, 4>(W, N, x, y, st); break;
        default: DCHECK(false, "head rows %d", m);
    }
}
static void hp_read(HProf& h) {
    if (h.kind < 0) return;
    for (int k = 0; k < h.nc; k++) {
        float ms = 0.f;
        // R23 (review r23 L4): never blocks (a sample not done yet is skipped), never clears an unrelated error
        if (cudaEventQuery(h.e[k][1]) != cudaSuccess || cudaEventElapsedTime(&ms, h.e[k][0], h.e[k][1]) != cudaSuccess) {
            h.kind = -1;
            return;
        }
        g_hpacc[h.kind][h.m][k] += ms;
    }
    g_hpn[h.kind][h.m]++;
    h.kind = -1;
}
static double g_dhacc[3] = {};
static int* g_drids = nullptr;   // R22 profile: the drafter's routing per stage (pinned host, 3 x 48 ints)
static long g_dpairs = 0, g_ddist = 0, g_dstages = 0;
static void dp_close() {   // after the stream sync
    if (g_dn < 0) return;
    if (g_drids) {
        for (int k = 0; k < 3; k++) {
            const int* r = g_drids + 48 * k;
            if (r[47] != 1) continue;   // stage not recorded this call
            int seen[16], ns = 0;
            for (int i = 0; i < 5 * TOPK; i++) {
                if (i % TOPK >= 3) continue;   // the drafter's top-3 (padding ids after)
                g_dpairs++;
                bool dup = false;
                for (int j = 0; j < ns; j++) dup |= seen[j] == r[i];
                if (!dup && ns < 16) seen[ns++] = r[i];
            }
            g_ddist += ns;
            g_dstages++;
            g_drids[48 * k + 47] = 0;
        }
    }
    if (g_dhon) {
        for (int k = 0; k < 2; k++) {
            if (!g_dhe[k][0]) continue;   // R23 (review r22 L1): 2 cards: one head share
            float ms = 0.f;
            DCK(cudaEventElapsedTime(&ms, g_dhe[k][0], g_dhe[k][1]));
            g_dhacc[k] += ms;
        }
        g_dhon = false;
    }
    for (int i = 1; i < g_dn; i++) {
        float ms = 0.f;
        DCK(cudaEventElapsedTime(&ms, g_dev[i - 1], g_dev[i]));
        size_t j = 0;
        while (j < g_dpacc.size() && g_dpacc[j].first != g_dlab[i]) j++;
        if (j == g_dpacc.size()) g_dpacc.push_back({g_dlab[i], 0.0});
        g_dpacc[j].second += ms;
    }
    float tot = 0.f;
    DCK(cudaEventElapsedTime(&tot, g_dev[0], g_dev[g_dn - 1]));
    g_dwall += tot;
    g_dcalls++;
    g_dn = -1;
}
static void ns_tick() {
    if (g_ns_a == -1) {
        const char* v = getenv("DSV4_NSYS");
        g_ns_a = -2;
        if (v) sscanf(v, "%d,%d", &g_ns_a, &g_ns_b);
    }
    g_dstep++;
    if (g_dstep == g_ns_a) { DCK(cudaProfilerStart()); g_nvtx = true; }
    if (g_dstep == g_ns_b) { DCK(cudaProfilerStop()); g_nvtx = false; }
}
static inline void PP(int l, int k, cudaStream_t st) {
    if (g_pon && l < NL && !tl_cap) DCK(cudaEventRecord(g_pe[l][k], st));   // (R20: drafter layers / captures never)
}
#include "dsv4_pprof.h"   // [dsv4-pf-pprof] prefill profile (DSV4_PPROF, opt-in)

// FP8-weight linear with the reference's input quantization: y[n][N] = q_fp8(x) . W^T. grp: grouped wo_a
// (W rows per group; x columns per group = W.K, x row stride ldx)
// xq (optional, R8): x already quantized to FP8-rounded bf16 [n][Kx] by the caller (shared by several GEMVs on the
// same input, e.g. wq_a and wkv): the GEMV stages it with 16 B copies instead of quantizing in every block (same values)
// returns whether the bf16 input xq was used (review r15 N5: callers that need it check this, not a copy of the rule)
static bool linq(Card& c, const F8& W, const float* x, int ldx, int n, float* y, int ldy, int grp = 0,
                 const bf16* xq = nullptr, cudaStream_t sx = nullptr) {
    cudaStream_t s = sx ? sx : c.st;   // R17: sx = a side stream (the shared expert beside the routed experts)
    if (W.sq) {   // release prep [dsv4-rel-q8]: a GGUF Q8_0 weight (int8 + fp16 per 32)
        const int Kx = grp ? W.K * (W.N / grp) : W.K;
        if (n <= MAXR) {   // decode rows: x to int8 per 32, dp4a (the drafter's int8 kernel with per-32 weight scales)
            DCHECK(c.d_di8, "Q8_0 linear: no int8 scratch on card %d", c.dev);
            const size_t so = sx ? (size_t)MAXR * NH * HD : 0;
            i8_linear(s, (const int8_t*)W.w, W.sq, W.N, W.K, grp, x, ldx, n, y, ldy, c.d_di8 + so, c.d_di8s + so / 32,
                      (size_t)MAXR * NH * HD, true);
            return false;
        }
        DCHECK(!sx, "Q8_0 linear: prefill rows on a side stream");
        // prefill: x as bf16 (no FP8 rounding: the weights are not FP8), weights dequantized in row slices, bf16 GEMM
        k_to_bf16s<<<cdiv((size_t)n * Kx, 256), 256, 0, s>>>(x, ldx, n, Kx, c.xb, Kx);
        const int ng = grp ? W.N / grp : 1;
        const int rows_g = grp ? grp : W.N;
        int slice = std::max<int>(32, (int)std::min<size_t>(rows_g, c.deq_elems / W.K) / 32 * 32);
        for (int g = 0; g < ng; g++)
            for (int r0 = 0; r0 < rows_g; r0 += slice) {
                int nr = std::min(slice, rows_g - r0);
                int rr = g * rows_g + r0;
                k_deq_q8<<<cdiv((size_t)nr * W.K / 2, 256), 256, 0, s>>>((const int8_t*)W.w, W.sq, W.K, rr, nr, c.deq);
                gemm_bf16(c.blas, c.xb + (size_t)g * W.K, Kx, c.deq, W.K, y + rr, ldy, n, nr, W.K);
            }
        return false;
    }
    if (W.si) {   // R22 [dsv4-r22-di8]: a drafter int8 tensor (decode rows only); a side-stream call has its own scratch
        DCHECK(n <= MAXR, "int8 linear: n %d", n);
        const size_t so = sx ? (size_t)MAXR * NH * HD : 0;
        i8_linear(s, (const int8_t*)W.w, W.si, W.N, W.K, grp, x, ldx, n, y, ldy, c.d_di8 + so, c.d_di8s + so / 32,
                  (size_t)MAXR * NH * HD);   // (the scratch half: R23 review r22 L3 / r23 L7, checked in i8_linear)
        return false;
    }
    const int Kx = grp ? W.K * (W.N / grp) : W.K;   // columns of x per row
    // R18 (review r17 L6): a side-stream call uses its own scratch (never the main stream's c.f32tmp / c.xb)
    DCHECK(!sx || (n <= MAXR && Kx <= 8192), "linq on a side stream: n %d Kx %d", n, Kx);
    float* const ft = sx ? c.f32s : c.f32tmp;
    bf16* const xbt = sx ? c.xbs : c.xb;
    if (n <= MAXR) {
        const int lpr = gemv_lpr(W.K), rpb = 8 * (32 / lpr);
        // R18: 2-8 rows at K >= 2048 stage x per K tile (k_gemv_f8t: same per-row arithmetic, 1.2-1.6x faster at 4-8 rows)
        const bool tiled = n >= 2 && W.K >= 2048 && !E_f8t_off;
        size_t smem = tiled ? (size_t)n * 64 * lpr * 2
                            : (size_t)n * W.K * 2 <= F8_SMEM_MAX ? (size_t)n * W.K * 2 : 0;   // R17: up to 96 KB
        // input quantization fused into the GEMV's staging when x goes through shared memory (round 3)
        const float* xin = x;
        int lx = ldx, qx = 1;
        bool used = false;
        if (xq && smem && !E_fuse_off && Kx % 8 == 0) {
            used = true;
            xin = (const float*)xq;
            lx = Kx;
            qx = 2;
        } else if (!smem || E_fuse_off) {
            q_fp8(s, x, ldx, n, Kx, ft, Kx, nullptr, 0);
            xin = ft;
            lx = Kx;
            qx = 0;
        } else if ((size_t)cdiv(W.N, rpb) * W.K * n > ((size_t)1 << 20) && !E_q1_off && Kx % 8 == 0) {
            // many blocks: quantize once to bf16 instead of in every block (wo_a / wo_b / wq_b / shared w1, w3)
            q_fp8(s, x, ldx, n, Kx, nullptr, 0, xbt, Kx);
            xin = (const float*)xbt;
            lx = Kx;
            qx = 2;
        }
        if (tiled) {
            if (lpr == 32) GEMV_DISPATCH_T(32, n, cdiv(W.N, rpb), smem, s, W.w, W.s, W.N, W.K, xin, lx, y, ldy, grp, qx)
            else if (lpr == 16) GEMV_DISPATCH_T(16, n, cdiv(W.N, rpb), smem, s, W.w, W.s, W.N, W.K, xin, lx, y, ldy, grp, qx)
            else GEMV_DISPATCH_T(8, n, cdiv(W.N, rpb), smem, s, W.w, W.s, W.N, W.K, xin, lx, y, ldy, grp, qx)
        } else if (lpr == 32) GEMV_DISPATCH_LPR(32, n, cdiv(W.N, rpb), smem, s, W.w, W.s, W.N, W.K, xin, lx, y, ldy, grp, qx)
        else if (lpr == 16) GEMV_DISPATCH_LPR(16, n, cdiv(W.N, rpb), smem, s, W.w, W.s, W.N, W.K, xin, lx, y, ldy, grp, qx)
        else GEMV_DISPATCH_LPR(8, n, cdiv(W.N, rpb), smem, s, W.w, W.s, W.N, W.K, xin, lx, y, ldy, grp, qx)
        return used;
    }
    q_fp8(s, x, ldx, n, Kx, nullptr, 0, c.xb, Kx);
    const int ng = grp ? W.N / grp : 1;
    const int rows_g = grp ? grp : W.N;
    int slice = std::max<int>(32, (int)std::min<size_t>(rows_g, c.deq_elems / W.K) / 32 * 32);
    for (int g = 0; g < ng; g++)
        for (int r0 = 0; r0 < rows_g; r0 += slice) {
            int nr = std::min(slice, rows_g - r0);
            int rr = g * rows_g + r0;
            k_deq_f8<<<cdiv((size_t)nr * W.K / 2, 256), 256, 0, s>>>(W.w, W.s, W.K, rr, nr, c.deq);
            gemm_bf16(c.blas, c.xb + (size_t)g * W.K, Kx, c.deq, W.K, y + rr, ldy, n, nr, W.K);
        }
    return false;
}
// fp32-weight (or BF16-weight, R10) linear, no input quantization
template <typename WT>
static void lin32(Card& c, const WT* W, int N, int K, const float* x, int ldx, int n, float* y, int ldy) {
    if (n <= MAXR) {
        if (N <= 4096 && K >= 1024 && !E_f32k_off) {   // split-K: small N would leave most SMs idle
            switch (n) {
                case 1: k_gemv_f32k<1, 4><<<cdiv(N, 2), 256, 0, c.st>>>(W, N, K, x, ldx, y, ldy); break;
                case 2: k_gemv_f32k<2, 4><<<cdiv(N, 2), 256, 0, c.st>>>(W, N, K, x, ldx, y, ldy); break;
                case 3: k_gemv_f32k<3, 4><<<cdiv(N, 2), 256, 0, c.st>>>(W, N, K, x, ldx, y, ldy); break;
                case 4: k_gemv_f32k<4, 4><<<cdiv(N, 2), 256, 0, c.st>>>(W, N, K, x, ldx, y, ldy); break;
                case 5: k_gemv_f32k<5, 4><<<cdiv(N, 2), 256, 0, c.st>>>(W, N, K, x, ldx, y, ldy); break;
                case 6: k_gemv_f32k<6, 4><<<cdiv(N, 2), 256, 0, c.st>>>(W, N, K, x, ldx, y, ldy); break;
                case 7: k_gemv_f32k<7, 4><<<cdiv(N, 2), 256, 0, c.st>>>(W, N, K, x, ldx, y, ldy); break;
                default: k_gemv_f32k<8, 4><<<cdiv(N, 2), 256, 0, c.st>>>(W, N, K, x, ldx, y, ldy); break;
            }
            return;
        }
        GEMV_DISPATCH(k_gemv_f32, n, cdiv(N, 8), c.st, W, N, K, x, ldx, y, ldy);
        return;
    }
    const float* Wf = (const float*)W;
    if constexpr (!std::is_same_v<WT, float>) {   // prefill: widen into the (chunk-mode) dequant scratch
        DCHECK(c.deq && (size_t)N * K * 4 <= c.deq_elems * 2, "lin32 bf16 scratch");
        k_bf16_f32<<<cdiv((size_t)N * K, 256), 256, 0, c.st>>>(W, (size_t)N * K, (float*)c.deq);
        Wf = (const float*)c.deq;
    }
    sgemm(c.blas, x, ldx, Wf, K, y, ldy, n, N, K);
}

// ---------------------------------------------------------------- loading

static std::string rd_file(const std::string& p) {
    std::ifstream f(p, std::ios::binary);
    if (!f) throw std::runtime_error("cannot read " + p);
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

// R9 VRAM audit: bytes allocated per card and category (g_mtag set by the loader)
static const char* MTAG[] = {"context/other", "workspaces (prefill + decode)", "compressed KV / index caches", "layer weights",
                             "attention shares", "window KV", "head", "DSpark drafter", "expert slots", "exchange rings",
                             "slot tables / small"};
constexpr int NMTAG = 11;
static int g_mtag = 0;
static double g_mb[3][NMTAG] = {};
static void* dmalloc(size_t b) {
    void* p = nullptr;
    DCK(cudaMalloc(&p, std::max<size_t>(b, 16)));
    int d = 0;
    cudaGetDevice(&d);
    if (d >= 0 && d < 3) g_mb[d][g_mtag] += (double)std::max<size_t>(b, 16);
    return p;
}
static void dfree(void* p, size_t b) {   // R23 (review r22 L8): a dmalloc'd buffer back, with its accounting
    if (!p) return;
    DCK(cudaFree(p));
    int d = 0;
    cudaGetDevice(&d);
    if (d >= 0 && d < 3) g_mb[d][g_mtag] -= (double)std::max<size_t>(b, 16);
}
// [dsv4-rel2-p1] window K/V layout: a ring of ARING rows per layer on the owner (row p & (ARING - 1), the peers' ring
// layout), a fixed prefill scratch of chunk + WIN rows per main layer (never borrowed from the pool) and a snapshot
// ring; DSV4_NOKVRING=1 = the old [maxctx] rows by absolute position
static const bool g_kring = getenv("DSV4_NOKVRING") == nullptr;
static double kv_bytes(const Options& o, bool main) {
    if (!g_kring) return (double)o.maxctx * HD * 2;
    return (double)(2 * 256 + (main ? o.chunk + 128 : 0)) * HD * 2;   // ring + snapshot ring (+ scratch)
}
static void kv_alloc(Engine* E, Layer& W, bool main);
// [dsv4-rel2-p1b] the indexer's score workspace tiled over the keys: S holds 64 query rows x 32 heads x IDX_TK keys
// (tiles of IDX_TK; a context whose sources hold <= IDX_TK rows runs one tile = the old calls); DSV4_NOIDXTILE=1 = S
// over all maxctx keys
static const bool g_idxtile = getenv("DSV4_NOIDXTILE") == nullptr;
constexpr int IDX_TK = 32768;
static int idx_tk(int maxctx) { return g_idxtile ? std::min(maxctx, IDX_TK) : maxctx; }
static size_t idx_ws(int maxctx) {
    return (size_t)64 * 32 * idx_tk(maxctx) * 4 + (size_t)64 * maxctx * 4 * 2 + (size_t)64 * 4096 * 6;
}
static float* up_f32(const TensorInfo& t) {   // BF16 / F32 tensor -> fp32 device
    size_t n = t.nbytes / (t.dtype == "BF16" ? 2 : 4);
    std::vector<float> h(n);
    if (t.dtype == "BF16") {
        const uint16_t* s = (const uint16_t*)t.data;
        for (size_t i = 0; i < n; i++) {
            uint32_t u = (uint32_t)s[i] << 16;
            memcpy(&h[i], &u, 4);
        }
    } else {
        DCHECK(t.dtype == "F32", "dtype %s", t.dtype.c_str());
        memcpy(h.data(), t.data, n * 4);
    }
    float* d = (float*)dmalloc(n * 4);
    DCK(cudaMemcpy(d, h.data(), n * 4, cudaMemcpyHostToDevice));
    return d;
}
static const bf16* up_bf16(const TensorInfo& t) {   // R10: BF16 tensor as is
    DCHECK(t.dtype == "BF16", "dtype %s (BF16 expected)", t.dtype.c_str());
    bf16* d = (bf16*)dmalloc(t.nbytes);
    DCK(cudaMemcpy(d, t.data, t.nbytes, cudaMemcpyHostToDevice));
    return d;
}
// release prep [dsv4-rel-q8]: a GGUF Q8_0 tensor (dtype "Q8_0": blocks of fp16 d + 32 int8, row-major [N][K]) rows
// [r0, r0 + nr) -> device int8 [nr][K] + fp16 scales [nr][K / 32] at w / sq + row offset (lossless split)
static void q8_split_rows(const TensorInfo& t, int r0, int nr, int8_t* dw, __half* ds) {
    const int K = (int)t.shape[1], nb = K / 32;
    DCHECK(t.dtype == "Q8_0" && K % 32 == 0 && t.nbytes == (size_t)t.shape[0] * nb * 34, "Q8_0 tensor layout");
    const size_t rc = 1024;   // rows per host chunk
    std::vector<int8_t> hw(rc * K);
    std::vector<uint16_t> hs(rc * nb);
    for (int c0 = 0; c0 < nr; c0 += (int)rc) {
        const int n = std::min<int>((int)rc, nr - c0);
        for (int r = 0; r < n; r++) {
            const uint8_t* src = (const uint8_t*)t.data + (size_t)(r0 + c0 + r) * nb * 34;
            for (int b = 0; b < nb; b++) {
                memcpy(&hs[(size_t)r * nb + b], src + (size_t)b * 34, 2);
                memcpy(&hw[(size_t)r * K + (size_t)b * 32], src + (size_t)b * 34 + 2, 32);
            }
        }
        DCK(cudaMemcpy(dw + (size_t)c0 * K, hw.data(), (size_t)n * K, cudaMemcpyHostToDevice));
        DCK(cudaMemcpy(ds + (size_t)c0 * nb, hs.data(), (size_t)n * nb * 2, cudaMemcpyHostToDevice));
    }
}
static F8 up_q8(const TensorInfo& t, int r0 = 0, int nr = -1) {
    F8 f;
    f.N = nr < 0 ? (int)t.shape[0] - r0 : nr;
    f.K = (int)t.shape[1];
    int8_t* dw = (int8_t*)dmalloc((size_t)f.N * f.K);
    __half* ds = (__half*)dmalloc((size_t)f.N * (f.K / 32) * 2);
    q8_split_rows(t, r0, f.N, dw, ds);
    f.w = (const uint8_t*)dw;
    f.sq = ds;
    return f;
}
static F8 up_f8(SafeTensors& st, const std::string& name) {
    if (st.get(name + ".weight").dtype == "Q8_0") return up_q8(st.get(name + ".weight"));
    const TensorInfo& w = st.get(name + ".weight");
    const TensorInfo& s = st.get(name + ".scale");
    F8 f;
    f.N = (int)w.shape[0];
    f.K = (int)w.shape[1];
    DCHECK(s.shape[0] * 32 == f.N && s.shape[1] * 32 == f.K, "%s scale shape", name.c_str());
    uint8_t* dw = (uint8_t*)dmalloc(w.nbytes);
    uint8_t* ds = (uint8_t*)dmalloc(s.nbytes);
    DCK(cudaMemcpy(dw, w.data, w.nbytes, cudaMemcpyHostToDevice));
    DCK(cudaMemcpy(ds, s.data, s.nbytes, cudaMemcpyHostToDevice));
    f.w = dw;
    f.s = ds;
    return f;
}

// R8: two FP8 weights with the same K stacked by rows ([a; b]) in one allocation: one GEMV launch for both (same values)
static F8 up_f8_cat(SafeTensors& st, const std::string& na, const std::string& nb) {
    if (st.get(na + ".weight").dtype == "Q8_0") {   // [dsv4-rel-q8]: the same stacking for Q8_0 rows
        const TensorInfo& qa = st.get(na + ".weight");
        const TensorInfo& qb = st.get(nb + ".weight");
        DCHECK(qb.dtype == "Q8_0" && qa.shape[1] == qb.shape[1], "%s / %s: Q8_0 shapes", na.c_str(), nb.c_str());
        F8 f;
        const int Na = (int)qa.shape[0], Nb = (int)qb.shape[0];
        f.N = Na + Nb;
        f.K = (int)qa.shape[1];
        int8_t* dw = (int8_t*)dmalloc((size_t)f.N * f.K);
        __half* ds = (__half*)dmalloc((size_t)f.N * (f.K / 32) * 2);
        q8_split_rows(qa, 0, Na, dw, ds);
        q8_split_rows(qb, 0, Nb, dw + (size_t)Na * f.K, ds + (size_t)Na * (f.K / 32));
        f.w = (const uint8_t*)dw;
        f.sq = ds;
        return f;
    }
    const TensorInfo& wa = st.get(na + ".weight");
    const TensorInfo& sa = st.get(na + ".scale");
    const TensorInfo& wb = st.get(nb + ".weight");
    const TensorInfo& sb = st.get(nb + ".scale");
    DCHECK(wa.shape[1] == wb.shape[1] && wa.shape[0] % 32 == 0, "%s / %s shapes", na.c_str(), nb.c_str());
    F8 f;
    f.N = (int)(wa.shape[0] + wb.shape[0]);
    f.K = (int)wa.shape[1];
    uint8_t* dw = (uint8_t*)dmalloc(wa.nbytes + wb.nbytes);
    uint8_t* ds = (uint8_t*)dmalloc(sa.nbytes + sb.nbytes);
    DCK(cudaMemcpy(dw, wa.data, wa.nbytes, cudaMemcpyHostToDevice));
    DCK(cudaMemcpy(dw + wa.nbytes, wb.data, wb.nbytes, cudaMemcpyHostToDevice));
    DCK(cudaMemcpy(ds, sa.data, sa.nbytes, cudaMemcpyHostToDevice));
    DCK(cudaMemcpy(ds + sa.nbytes, sb.data, sb.nbytes, cudaMemcpyHostToDevice));
    f.w = dw;
    f.s = ds;
    return f;
}

// rope tables as dsv4_model.rope_table: kind 0 = theta 10000 plain, kind 1 = 160000 + YaRN (x16 from 64K)
static void rope_tables(int maxpos, int kind, std::vector<float>& cs, std::vector<float>& sn) {
    const int dim = RD;
    double base = kind ? 160000.0 : 10000.0;
    float fr[32];
    for (int j = 0; j < 32; j++) fr[j] = 1.f / powf((float)base, (float)(2 * j) / (float)dim);
    if (kind) {
        const double orig = 65536, rfac = 16, bfast = 32, bslow = 1;
        auto corr = [&](double rot) { return dim * log(orig / (rot * 2 * M_PI)) / (2 * log(base)); };
        int low = std::max((int)floor(corr(bfast)), 0), high = std::min((int)ceil(corr(bslow)), dim - 1);
        for (int j = 0; j < 32; j++) {
            float ramp = std::min(std::max(((float)j - (float)low) / std::max((float)(high - low), 1e-3f), 0.f), 1.f);
            float smooth = 1.f - ramp;
            fr[j] = fr[j] / (float)rfac * (1.f - smooth) + fr[j] * smooth;
        }
    }
    cs.resize((size_t)maxpos * 32);
    sn.resize((size_t)maxpos * 32);
    for (int p = 0; p < maxpos; p++)
        for (int j = 0; j < 32; j++) {
            float a = (float)p * fr[j];
            cs[(size_t)p * 32 + j] = (float)cos((double)a);
            sn[(size_t)p * 32 + j] = (float)sin((double)a);
        }
}

static void read_into(const std::string& path, void* dst, size_t bytes) {
    int fd = open(path.c_str(), O_RDONLY);
    DCHECK(fd >= 0, "open %s", path.c_str());
    // parallel preads in 64 MB pieces
    const size_t piece = (size_t)64 << 20;
    std::atomic<size_t> next{0};
    auto rdr = [&] {
        while (true) {
            size_t off = next.fetch_add(piece);
            if (off >= bytes) return;
            size_t n = std::min(piece, bytes - off);
            size_t got = 0;
            while (got < n) {
                ssize_t r = pread(fd, (char*)dst + off + got, n - got, off + got);
                DCHECK(r > 0, "pread %s", path.c_str());
                got += r;
            }
        }
    };
    std::vector<std::thread> th;
    for (int i = 0; i < 8; i++) th.emplace_back(rdr);
    for (auto& t : th) t.join();
    close(fd);
}

// pinned host memory, else pageable (degrade under RAM pressure, review M4); aborts only if both fail
// [dsv4-rel-thp] (advisor 2c, coordinator 11-14): DSV4_THP=1: large host arenas (>= 64 MB) as 2 MB-aligned anonymous
// mappings with MADV_HUGEPAGE (transparent huge pages, madvise mode), touched, then registered with CUDA: the lane
// reads them through 2 MB TLB entries instead of 4 KB. Off by default until measured
static double mem_avail_gib();
static const bool g_thp = getenv("DSV4_THP") && atoi(getenv("DSV4_THP")) != 0;
static std::mutex g_thp_mu;
static std::map<void*, std::pair<void*, size_t>> g_thp_maps;   // aligned pointer -> (mapping, length)
// R24 [dsv4-r24-pin]: default for large arenas (DSV4_HOSTALLOC=1: cudaHostAlloc as before): an anonymous mapping on
// 4 KB pages (MADV_NOHUGEPAGE), faulted in by 16 threads, then cudaHostRegister: measured 34.6 GB/s touch + 25.8 GB/s
// register vs 3.2 GB/s for cudaHostAlloc (bench/results/dsv4/r24/lb.txt)
static const bool g_hreg = !getenv("DSV4_HOSTALLOC") || atoi(getenv("DSV4_HOSTALLOC")) == 0;
static uint8_t* host_alloc_thp(size_t b, bool huge = true) {
    const size_t H = 2u << 20, len = (b + H - 1) / H * H + H;
    void* m = mmap(nullptr, len, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (m == MAP_FAILED) return nullptr;
    uint8_t* a = (uint8_t*)(((uintptr_t)m + H - 1) & ~(uintptr_t)(H - 1));
    madvise(a, (b + H - 1) / H * H, huge ? MADV_HUGEPAGE : MADV_NOHUGEPAGE);
    // review r24 M1: like the driver's cudaHostAlloc mappings, never in a core dump, never copied into a fork child
    madvise(m, len, MADV_DONTDUMP);
    madvise(m, len, MADV_DONTFORK);
    if (huge) for (size_t o = 0; o < b; o += 4096) a[o] = 0;   // fault in (huge pages where the kernel has them)
    else {   // R24: 16 threads, 64 MB pieces
        std::atomic<size_t> nx{0};
        std::vector<std::thread> th;
        for (int i = 0; i < 16; i++)
            th.emplace_back([&] {
                for (size_t o0; (o0 = nx.fetch_add(64u << 20)) < b;)
                    for (size_t o = o0; o < std::min(b, o0 + (64u << 20)); o += 4096) ((volatile uint8_t*)a)[o] = 0;
            });
        for (auto& x : th) x.join();
    }
    if (cudaHostRegister(a, b, cudaHostRegisterPortable) != cudaSuccess) {
        cudaGetLastError();
        munmap(m, len);
        return nullptr;
    }
    std::lock_guard<std::mutex> lk(g_thp_mu);
    g_thp_maps[a] = {m, len};
    return a;
}
static uint8_t* host_alloc(size_t b, int* n_pageable) {
    uint8_t* p = nullptr;
    // review rel3 M2: the touch faults anonymous pages (no failure path): only with room for them beyond the margin
    if ((g_thp || g_hreg) && b >= (64u << 20) && mem_avail_gib() * 1073741824.0 > (double)b + 12.0 * 1073741824.0 &&
        (p = host_alloc_thp(b, g_thp)))
        return p;
    if (cudaHostAlloc(&p, b, cudaHostAllocPortable) == cudaSuccess) return p;
    cudaGetLastError();
    p = (uint8_t*)malloc(b);
    DCHECK(p, "host RAM: cannot allocate %.2f GB for experts (pinned and pageable failed)", b / 1e9);
    (*n_pageable)++;
    return p;
}

#include "gguf_hooks.h"   // R16 [dsv4-gguf-iq]

// review r13 L7: a swap home must be pinned (a D2H into pageable memory blocks the decode thread)
static bool is_pinned(const void* p) {
    cudaPointerAttributes a{};
    if (cudaPointerGetAttributes(&a, p) != cudaSuccess) {
        cudaGetLastError();
        return false;
    }
    return a.type == cudaMemoryTypeHost;
}
static void host_free(void* p) {
    if (g_thp || g_hreg) {
        std::lock_guard<std::mutex> lk(g_thp_mu);
        auto it = g_thp_maps.find(p);
        if (it != g_thp_maps.end()) {
            cudaHostUnregister(p);
            munmap(it->second.first, it->second.second);
            g_thp_maps.erase(it);
            return;
        }
    }
    if (cudaFreeHost(p) != cudaSuccess) {   // pageable fallback of host_alloc
        cudaGetLastError();
        free(p);
    }
}
// /proc/meminfo MemAvailable and /proc/self/status (VmRSS, RssShmem, VmHWM), GiB
static double mem_avail_gib() {
    double v = -1;
    FILE* f = fopen("/proc/meminfo", "r");
    if (!f) return v;
    char k[64];
    long x;
    while (fscanf(f, "%63s %ld kB\n", k, &x) == 2)
        if (!strcmp(k, "MemAvailable:")) v = x / 1048576.0;
    fclose(f);
    return v;
}
static void log_mem(const char* when) {
    double rss = 0, shm = 0, hwm = 0, tot = 0;
    char line[256];
    if (FILE* f = fopen("/proc/self/status", "r")) {
        while (fgets(line, sizeof line, f)) {
            long x;
            if (sscanf(line, "VmRSS: %ld", &x) == 1) rss = x / 1048576.0;
            if (sscanf(line, "RssShmem: %ld", &x) == 1) shm = x / 1048576.0;
            if (sscanf(line, "VmHWM: %ld", &x) == 1) hwm = x / 1048576.0;
        }
        fclose(f);
    }
    if (FILE* f = fopen("/proc/meminfo", "r")) {
        while (fgets(line, sizeof line, f)) {
            long x;
            if (sscanf(line, "MemTotal: %ld", &x) == 1) tot = x / 1048576.0;
        }
        fclose(f);
    }
    logf_("memory %s: RSS %.2f GiB (shmem %.2f), peak RSS %.2f GiB, MemTotal %.2f GiB, MemAvailable %.2f GiB\n", when, rss,
          shm, hwm, tot, mem_avail_gib());
}
static XFmt pool_fmt(const Engine* E, int l);   // R16 pool helpers (defined beside set_slot)
static Card& pool_card(Engine* E, Card& c, int slot);
static uint8_t* pool_slot_ptr(Engine* E, Card& c, int l, int slot);
static uint8_t* pool_base(Engine* E, Card& k, int l);
static int pool_lidx(Engine* E, int l, int slot);
static void set_pslot(Engine* E, Layer& W, int e, int g);
static void tier_snapshot(Engine* E);   // [dsv4-rel-tierreset]
static void tail_readahead(Engine* E);  // [dsv4-rel-load128]
static double mem_avail_gib();
static int m23_threads();
static double m23_conv_bytes();
static void m23_convert_list(Engine* E, int l, const std::vector<std::pair<int, uint8_t*>>& todo);
// ---- [dsv4-ctx-lazy] lazy context reservation: CUDA VMM entry points, resolved from the driver at run time (no -lcuda)
struct LzDrv {
    decltype(&cuMemCreate) create = nullptr;
    decltype(&cuMemRelease) release = nullptr;
    decltype(&cuMemMap) map = nullptr;
    decltype(&cuMemUnmap) unmap = nullptr;
    decltype(&cuMemSetAccess) access = nullptr;
    decltype(&cuMemAddressReserve) reserve = nullptr;
    decltype(&cuMemAddressFree) vfree = nullptr;
    bool ok = false;
};
static LzDrv& lzd() {
    static LzDrv d = [] {
        LzDrv r;
        auto get = [](const char* n, void** f) {
            cudaDriverEntryPointQueryResult q;
            if (cudaGetDriverEntryPointByVersion(n, f, 12000, cudaEnableDefault, &q) != cudaSuccess || q != cudaDriverEntryPointSuccess) {
                (void)cudaGetLastError();
                *f = nullptr;
                return false;
            }
            return true;
        };
        r.ok = get("cuMemCreate", (void**)&r.create) && get("cuMemRelease", (void**)&r.release) && get("cuMemMap", (void**)&r.map) &&
               get("cuMemUnmap", (void**)&r.unmap) && get("cuMemSetAccess", (void**)&r.access) &&
               get("cuMemAddressReserve", (void**)&r.reserve) && get("cuMemAddressFree", (void**)&r.vfree);
        return r;
    }();
    return d;
}
constexpr size_t LZG = 2u << 20;   // the VMM granularity of these cards (one loan chunk)
static size_t lz_up(size_t b) { return (b + LZG - 1) / LZG * LZG; }
static size_t lz_dn(size_t b) { return b / LZG * LZG; }
// zero-filled new memory (as cudaMalloc's fresh pages): 1 the context buffers, 2 the pool (DSV4_LZ_ZERO, test knob)
static int lz_zero() {
    static const int v = getenv("DSV4_LZ_ZERO") ? atoi(getenv("DSV4_LZ_ZERO")) : 3;
    return v;
}
// the base context mapped at load (DSV4_CTX_BASE, default 32,768 positions)
static int lz_base_ctx() {
    static const int v = getenv("DSV4_CTX_BASE") ? std::max(4096, atoi(getenv("DSV4_CTX_BASE"))) : 32768;
    return v;
}
static bool lz_mk(const Card& c, size_t b, unsigned long long* h) {
    CUmemAllocationProp pr = {};
    pr.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    pr.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    pr.location.id = c.dev;
    CUmemGenericAllocationHandle hh = 0;
    if (lzd().create(&hh, b, &pr, 0) != CUDA_SUCCESS) return false;
    *h = (unsigned long long)hh;
    return true;
}
static bool lz_map(const Card& c, unsigned long long va, size_t b, unsigned long long h) {
    if (lzd().map((CUdeviceptr)va, b, 0, (CUmemGenericAllocationHandle)h, 0) != CUDA_SUCCESS) return false;
    if (lzd().access((CUdeviceptr)va, b, c.lz_acc.data(), c.lz_acc.size()) == CUDA_SUCCESS) return true;
    lzd().unmap((CUdeviceptr)va, b);   // (review kb L: no mapping left behind on a failed access)
    return false;
}
// a lazily mapped buffer: cap bytes of address space, base bytes mapped (own memory, VRAM audit tag g_mtag)
static void lz_buf_init(Card& c, Card::LzBuf& b, size_t cap, size_t base) {
    b.cap = lz_up(std::max<size_t>(cap, 1));
    b.base = b.mapped = std::min(b.cap, lz_up(std::max<size_t>(base, 1)));
    CUdeviceptr va = 0;
    DCHECK(lzd().reserve(&va, b.cap, LZG, 0, 0) == CUDA_SUCCESS, "card %d: VMM address reservation of %zu bytes failed (DSV4_NOCTXLAZY=1: static context reservation)", c.dev, b.cap);
    b.va = (unsigned long long)va;
    DCHECK(lz_mk(c, b.base, &b.hbase) && lz_map(c, b.va, b.base, b.hbase), "card %d: VMM mapping of %zu bytes failed (DSV4_NOCTXLAZY=1: static context reservation)", c.dev, b.base);
    if (lz_zero() & 1) DCK(cudaMemset((void*)(uintptr_t)b.va, 0, b.base));
    if (c.dev >= 0 && c.dev < 3) g_mb[c.dev][g_mtag] += (double)b.base;
}
// the whole buffer mapped from own memory (a card without loan zones; at load)
static bool lz_buf_full(Card& c, Card::LzBuf& b) {
    while (b.mapped < b.cap) {
        unsigned long long h = 0;
        if (!lz_mk(c, LZG, &h)) return false;
        if (!lz_map(c, b.va + b.mapped, LZG, h)) {
            lzd().release((CUmemGenericAllocationHandle)h);
            return false;
        }
        b.fresh.push_back(h);
        b.ch.push_back(-(int)b.fresh.size());   // < 0: own handle fresh[-1 - x]
        b.mapped += LZG;
        if (c.dev >= 0 && c.dev < 3) g_mb[c.dev][2] += (double)LZG;
    }
    b.base = b.mapped;   // review rel7 M2: part of the base (never given back); ch / fresh keep the handles
    return true;
}
// [dsv4-rel8-ctxcap] (rel7 review L6) card c has no loan zones and cannot map its context buffers for the full maxctx
// (not enough VRAM): instead of aborting the load (rule: if the model loads, it runs), the partial mapping is undone,
// the context is capped to the positions whose buffers fit in 3/4 of the card's free VRAM above `keep` (the rest stays
// for the expert pool; at least the base context, always mapped), the K/V page counts and every card's buffer caps
// follow, and this card maps that in full. maxctx itself drops at the end of create() (ctx_cap); a warning says so.
// base: the buffers' bases before the failed full mapping
static void lz_cap_ctx(Engine* E, Card& c, const size_t base[12], size_t keep) {
    for (int j = 0; j < 12; j++) {   // undo: everything above the old base (all of it own fresh handles at load)
        Card::LzBuf& b = c.lzb[j];
        while (b.mapped > base[j] && !b.ch.empty()) {
            const int x = b.ch.back();
            b.ch.pop_back();
            b.mapped -= LZG;
            DCHECK(lzd().unmap(b.va + b.mapped, LZG) == CUDA_SUCCESS, "card %d: context cap: unmap failed", c.dev);
            if (x < 0) lzd().release((CUmemGenericAllocationHandle)b.fresh[-1 - x]);
            if (c.dev >= 0 && c.dev < 3) g_mb[c.dev][2] -= (double)LZG;
        }
        b.ch.clear();
        b.fresh.clear();
        b.base = b.mapped;
    }
    size_t fr = 0, to = 0;
    DCK(cudaMemGetInfo(&fr, &to));
    const size_t room = fr > keep ? (fr - keep) / 4 * 3 : 0;
    const size_t pk = (size_t)KVPG * HD * 2, pi = (size_t)KVPG * IDIM * 2;
    auto pages = [&](int si, int m) { return ((si == 3 ? m : m / 2 + 1) + KVPG - 1) / KVPG + (E->kvP[si] - E->kvcap[si]); };
    auto caps = [&](int m, size_t cap[12]) {
        for (int si = 0; si < 4; si++) cap[si] = lz_up((size_t)pages(si, m) * pk), cap[4 + si] = lz_up((size_t)pages(si, m) * pi);
        for (int t = 8; t < 12; t++) cap[t] = lz_up((size_t)m * 32 * 4);
    };
    auto cost = [&](int m) {
        size_t cap[12], t = 0;
        caps(m, cap);
        for (int j = 0; j < 12; j++) t += cap[j] > c.lzb[j].base ? std::min(cap[j], c.lzb[j].cap) - std::min(c.lzb[j].base, cap[j]) : 0;
        return t;
    };
    const int m0 = E->ctx_cap ? E->ctx_cap : E->o.maxctx;
    int m = m0 / 4096 * 4096;
    while (m > E->ropeB && cost(m) > room) m -= 4096;
    if (m <= E->ropeB) m = E->ropeB;
    for (int tr = 0; tr < 2; tr++) {
        size_t cap[12];
        caps(m, cap);
        for (int si = 0; si < 4; si++) {
            const int extra = E->kvP[si] - E->kvcap[si];
            E->kvcap[si] = ((si == 3 ? m : m / 2 + 1) + KVPG - 1) / KVPG;
            E->kvP[si] = std::max(E->kvB[si], E->kvcap[si] + extra);
            E->kvown[si].assign(E->kvP[si], 0);
            E->kvM[si] = std::min(E->kvM[si], E->kvP[si]);
        }
        E->ropeM = std::min(E->ropeM, m);
        for (auto& cc : E->cards) {   // every card's growth stops at the capped window
            size_t over = 0;
            for (int j = 0; j < 12; j++) {
                over += cc.lzb[j].mapped > cap[j] ? cc.lzb[j].mapped - cap[j] : 0;
                cc.lzb[j].cap = std::max(cc.lzb[j].base, std::min(cc.lzb[j].cap, cap[j]));
            }
            if (tr == 0 && &cc != &c && over)   // review rel8 L4: a card mapped in full before the cap keeps it (its pool is sized)
                logf_("card %d: %.0f MB of context buffers stay mapped above the capped context (mapped in full before the cap)\n",
                      cc.dev, over / 1e6);
        }
        E->ctx_cap = m;
        bool ok = true;
        for (auto& b : c.lzb) ok = ok && lz_buf_full(c, b);
        if (ok) break;
        DCHECK(tr == 0 && m > E->ropeB, "card %d: no VRAM even for the base context buffers (%d positions)", c.dev, E->ropeB);
        size_t b2[12];
        for (int j = 0; j < 12; j++) b2[j] = base[j];
        for (int j = 0; j < 12; j++) {   // undo again; the base context only
            Card::LzBuf& b = c.lzb[j];
            while (b.mapped > b2[j] && !b.ch.empty()) {
                const int x = b.ch.back();
                b.ch.pop_back();
                b.mapped -= LZG;
                DCHECK(lzd().unmap(b.va + b.mapped, LZG) == CUDA_SUCCESS, "card %d: context cap: unmap failed", c.dev);
                if (x < 0) lzd().release((CUmemGenericAllocationHandle)b.fresh[-1 - x]);
                if (c.dev >= 0 && c.dev < 3) g_mb[c.dev][2] -= (double)LZG;
            }
            b.ch.clear();
            b.fresh.clear();
            b.base = b.mapped;
        }
        m = E->ropeB;
    }
    logf_("WARNING card %d: no VRAM to map the context buffers for %d positions (no loan zones): the context is capped to %d "
          "positions (%.1f GB free; a smaller --max-ctx or more free VRAM gives the requested one)\n", c.dev, m0, m, fr / 1e9);
}
// review rel7 M2: the pages / positions every card maps for good (bases) and now
static void lz_sync_base(Engine* E) {
    for (int si = 0; si < 4; si++) {
        size_t b = SIZE_MAX, m = SIZE_MAX;
        for (auto& c : E->cards) {
            b = std::min({b, c.lzb[si].base / ((size_t)KVPG * HD * 2), c.lzb[4 + si].base / ((size_t)KVPG * IDIM * 2)});
            m = std::min({m, c.lzb[si].mapped / ((size_t)KVPG * HD * 2), c.lzb[4 + si].mapped / ((size_t)KVPG * IDIM * 2)});
        }
        E->kvB[si] = (int)std::min<size_t>(E->kvP[si], b);
        E->kvM[si] = (int)std::min<size_t>(E->kvP[si], m);
    }
    size_t b = SIZE_MAX, m = SIZE_MAX;
    for (auto& c : E->cards)
        for (int t = 8; t < 12; t++) b = std::min(b, c.lzb[t].base / (32 * 4)), m = std::min(m, c.lzb[t].mapped / (32 * 4));
    E->ropeB = (int)std::min<size_t>(E->o.maxctx, b);
    E->ropeM = (int)std::min<size_t>(E->o.maxctx, m);
}
// the loan zones of card c for a pool of off bytes with its layer layout (ploff / pnl): layers whose region ends below
// the prefill borrow tail, the last nmax slots each, nmax so that the zones hold every buffer's growth (lz_nmax)
static bool lz_zones(Engine* E, Card& c, size_t off) {
    size_t lmax = 0;
    for (auto& b : c.lzb) lmax += b.cap - b.mapped;
    for (int l = 0; l < NL; l++) c.lz_nmax[l] = c.lz_n[l] = 0;
    if (!lmax) return true;
    const size_t tail = E->gborrow ? E->borrow_bytes + 256 + LZG : 0;
    std::vector<int> el;
    for (int l = 0; l < NL; l++) {
        const size_t xb = expert_bytes(pool_fmt(E, l));
        if (c.pnl[l] >= 8 && c.ploff[l] + (size_t)c.pnl[l] * xb + tail <= off) el.push_back(l);
    }
    if (el.empty()) return false;
    for (int extra = 2; extra <= 64; extra++) {
        size_t capz = 0;
        bool more = false;
        for (int l : el) {
            const size_t xb = expert_bytes(pool_fmt(E, l));
            const int n = std::min(c.pnl[l] / 2, (int)((lmax / el.size() + xb - 1) / xb) + extra);
            more |= n < c.pnl[l] / 2;
            const size_t end = c.ploff[l] + (size_t)c.pnl[l] * xb, lo = lz_up(c.ploff[l] + (size_t)(c.pnl[l] - n) * xb), hi = lz_dn(end);
            capz += hi > lo ? hi - lo : 0;
            c.lz_nmax[l] = hi > lo ? n : 0;
        }
        if (capz >= lmax + lmax / 8) return true;   // (1/8 spare: the boundary chunks of partly lent slots)
        if (!more) break;
    }
    for (int l = 0; l < NL; l++) c.lz_nmax[l] = 0;
    return false;
}
// the split pool of card c (off bytes) as one VMM range: one handle per stretch between the loan zones, a 2 MB handle
// per zone chunk (lent to the context buffers on demand)
// [kb-dsv4-load] false (everything undone) when the VRAM is not there any more: another process took it between the
// sizing (cudaMemGetInfo) and these mappings; the caller sizes the pool again (morning list 39: degrade, never abort).
// DSV4_TEST_POOLFAIL=n: the first n pool mappings fail (test)
static bool lz_pool_alloc(Engine* E, Card& c, size_t off) {
    (void)E;
    const size_t tot = lz_up(std::max<size_t>(off, 1));
    CUdeviceptr va = 0;
    if (lzd().reserve(&va, tot, LZG, 0, 0) != CUDA_SUCCESS) return false;
    c.lzc.clear();
    std::vector<std::pair<size_t, size_t>> done;   // mapped (offset, bytes), for the undo
    static std::atomic<int> test_fail{getenv("DSV4_TEST_POOLFAIL") ? atoi(getenv("DSV4_TEST_POOLFAIL")) : 0};
    bool ok = test_fail.fetch_sub(1) <= 0;
    auto big = [&](size_t a, size_t b) {
        if (!ok || b <= a) return;
        unsigned long long h = 0;
        if (!lz_mk(c, b - a, &h)) { ok = false; return; }
        if (lzd().map(va + a, b - a, 0, (CUmemGenericAllocationHandle)h, 0) != CUDA_SUCCESS) ok = false;
        else done.push_back({a, b - a});
        lzd().release((CUmemGenericAllocationHandle)h);   // the mapping keeps it
    };
    size_t cur = 0;
    for (int l = 0; l < NL && ok; l++) {
        c.lz_c0[l] = c.lz_c1[l] = (int)c.lzc.size();
        if (!c.lz_nmax[l]) continue;
        const size_t xb = expert_bytes(pool_fmt(E, l));
        const size_t lo = lz_up(c.ploff[l] + (size_t)(c.pnl[l] - c.lz_nmax[l]) * xb), hi = lz_dn(c.ploff[l] + (size_t)c.pnl[l] * xb);
        big(cur, lo);
        for (size_t o = hi; o > lo && ok; o -= LZG) {
            unsigned long long h = 0;
            if (!lz_mk(c, LZG, &h)) { ok = false; break; }
            if (lzd().map(va + o - LZG, LZG, 0, (CUmemGenericAllocationHandle)h, 0) != CUDA_SUCCESS) {
                lzd().release((CUmemGenericAllocationHandle)h);
                ok = false;
                break;
            }
            done.push_back({o - LZG, LZG});
            c.lzc.push_back({o - LZG, h, -1});
        }
        c.lz_c1[l] = (int)c.lzc.size();
        cur = hi;
    }
    big(cur, tot);
    if (ok && lzd().access(va, tot, c.lz_acc.data(), c.lz_acc.size()) != CUDA_SUCCESS) ok = false;
    if (!ok) {
        for (auto& [o, b] : done) lzd().unmap(va + o, b);
        for (auto& ch : c.lzc) lzd().release((CUmemGenericAllocationHandle)ch.h);
        c.lzc.clear();
        lzd().vfree(va, tot);
        cudaGetLastError();
        return false;
    }
    c.dpool = (uint8_t*)(uintptr_t)va;
    c.lz_vmm = true;
    if (lz_zero() & 2) DCK(cudaMemset(c.dpool, 0, tot));
    if (c.dev >= 0 && c.dev < 3) g_mb[c.dev][g_mtag] += (double)off;
    return true;
}
// the pool memory back (the pool is sized again, review r19 M1; nothing lent at load)
static void pool_free(Card& c) {
    if (!c.dpool) return;
    if (!c.lz_vmm) {
        DCK(cudaFree(c.dpool));
    } else {
        DCK(cudaDeviceSynchronize());
        const CUdeviceptr va = (CUdeviceptr)(uintptr_t)c.dpool;
        // every stretch and chunk is its own mapping: unmap them in address order
        std::vector<std::pair<size_t, size_t>> m;   // (offset, bytes) of the chunk mappings
        for (auto& ch : c.lzc) m.push_back({ch.off, LZG});
        std::sort(m.begin(), m.end());
        const size_t tot = lz_up(std::max<size_t>(c.dpool_bytes, 1));
        size_t cur = 0;
        for (auto& [o, b] : m) {
            if (o > cur) lzd().unmap(va + cur, o - cur);
            lzd().unmap(va + o, b);
            cur = o + b;
        }
        if (tot > cur) lzd().unmap(va + cur, tot - cur);
        for (auto& ch : c.lzc) lzd().release((CUmemGenericAllocationHandle)ch.h);
        c.lzc.clear();
        lzd().vfree(va, tot);
        c.lz_vmm = false;
    }
    c.dpool = nullptr;
}
static void lz_shrink(Engine* E, bool force, bool restore = true);   // [dsv4-ctx-lazy] (with the K/V pool helpers)
static size_t lz_restore(Engine* E);   // [dsv4-rel8-giveback]
static size_t lz_seed_fill(Engine* E, const std::vector<std::pair<int, int>>& freed);
// [dsv4-ctx-lazy] pool slot g lies in a loan zone (the last lz_nmax slots of its layer's region on its card)
static bool in_lzone(const Engine* E, int g) {
    if (!E->lz || g < 0) return false;
    const Card& k = E->cards[gs_card(g)];
    const int idx = gs_idx(g);
    for (int l = 0; l < NL; l++)
        if (idx >= k.ps0[l] && idx < k.ps0[l] + k.pnl[l]) return k.lz_nmax[l] > 0 && idx - k.ps0[l] >= k.pnl[l] - k.lz_nmax[l];
    return false;
}
// review r19 H1: global pool slot g of a split M23 / GGUF pool lies in its card's largest borrowable region (the tail
// [dpool_bytes - borrow_bytes - 256, dpool_bytes) a chunk-sized prefill carves): its expert must keep an exact host
// copy (evicted / restored by gborrow_ws), so the seeding and the memory guard never make it static (as in_tail).
// [dsv4-ctx-lazy]: the loan zones count as such a region too (their slots may be lent to the context)
static bool in_gtail(const Engine* E, int g) {
    if (in_lzone(E, g)) return true;
    if (!E->gborrow || g < 0) return false;
    const Card& k = E->cards[gs_card(g)];
    const size_t need = E->borrow_bytes + 256;
    if (k.dpool_bytes <= need) return true;
    const int idx = gs_idx(g);
    for (int l = NL - 1; l >= 0; l--)
        if (idx >= k.ps0[l] && idx < k.ps0[l] + k.pnl[l])
            return k.ploff[l] + (size_t)(idx - k.ps0[l] + 1) * expert_bytes(pool_fmt(E, l)) > k.dpool_bytes - need;
    return false;
}
static void read_runs(const std::string& path, void* dst, const std::vector<std::pair<size_t, size_t>>& runs) {
    int fd = open(path.c_str(), O_RDONLY);
    DCHECK(fd >= 0, "open %s", path.c_str());
    std::vector<size_t> doff(runs.size());
    size_t acc = 0;
    for (size_t i = 0; i < runs.size(); i++) {
        doff[i] = acc;
        acc += runs[i].second;
    }
    std::atomic<size_t> next{0};
    auto rdr = [&] {
        while (true) {
            size_t i = next.fetch_add(1);
            if (i >= runs.size()) return;
            size_t got = 0, n = runs[i].second;
            while (got < n) {
                ssize_t r = pread(fd, (char*)dst + doff[i] + got, n - got, runs[i].first + got);
                DCHECK(r > 0, "pread %s", path.c_str());
                got += r;
            }
        }
    };
    std::vector<std::thread> th;
    for (int i = 0; i < 8; i++) th.emplace_back(rdr);
    for (auto& t : th) t.join();
    close(fd);
}

// R16: host copies of layer l's experts (ascending id, the ones the load reads: a host format, not resident, not a
// checkpoint copy) converted from the checkpoint's FP4 into M23 (dsv4_m23.h = dsv4_pack.py's rule), into ar back to
// back. pread per tensor (no mmap faults) and POSIX_FADV_DONTNEED after it, so the 289 GB pass leaves no page cache.
static int m23_threads() { return std::max(4, std::min(32, (int)std::thread::hardware_concurrency() - 2)); }
// review r16 L6: the converter threads' read buffers (one FP4 expert each), counted in the host RAM plan
static double m23_conv_bytes() { return (double)m23_threads() * expert_bytes(XF_FP4); }
static void m23_convert_list(Engine* E, int l, const std::vector<std::pair<int, uint8_t*>>& todo) {
    for (auto& t : todo)
        DCHECK(E->L[l].hfmt[t.first] == XF_M23, "layer %d expert %d: format %d in an M23 pack", l, t.first, (int)E->L[l].hfmt[t.first]);
    std::atomic<size_t> next{0};
    auto wk = [&] {
        std::map<std::string, int> fds;
        std::vector<uint8_t> buf[6];
        while (true) {
            const size_t i = next.fetch_add(1);
            if (i >= todo.size()) break;
            const int e = todo[i].first;
            const std::string px = "layers." + std::to_string(l) + ".ffn.experts." + std::to_string(e) + ".";
            const uint8_t* w[3];
            const uint8_t* sc[3];
            int k = 0;
            for (const char* wn : {"w1", "w3", "w2"})
                for (const char* kn : {".weight", ".scale"}) {
                    const TensorInfo& t = E->stf.get(px + wn + kn);
                    {   // review r16 L7: the shapes the conversion assumes (w1 / w3 [2304][5120], w2 [5120][2304])
                        const size_t N = wn[1] == '2' ? D : INTER, K = wn[1] == '2' ? INTER : D;
                        const size_t want = kn[1] == 'w' ? N * K / 2 : N * K / 32;
                        DCHECK(t.nbytes == want, "m23: %s%s%s has %zu bytes, %zu expected", px.c_str(), wn, kn, t.nbytes, want);
                    }
                    size_t foff = 0;
                    const std::string path = E->stf.file_of(t.data, &foff);
                    DCHECK(!path.empty(), "m23: no file for %s%s%s", px.c_str(), wn, kn);
                    auto it = fds.find(path);
                    if (it == fds.end()) {
                        const int fd = open(path.c_str(), O_RDONLY);
                        DCHECK(fd >= 0, "open %s", path.c_str());
                        it = fds.emplace(path, fd).first;
                    }
                    buf[k].resize(t.nbytes);
                    size_t got = 0;
                    while (got < t.nbytes) {
                        const ssize_t r = pread(it->second, buf[k].data() + got, t.nbytes - got, (off_t)(foff + got));
                        if (r < 0 && errno == EINTR) continue;   // review r16 L6
                        DCHECK(r > 0, "pread %s", path.c_str());
                        got += (size_t)r;
                    }
                    posix_fadvise(it->second, (off_t)foff, (off_t)t.nbytes, POSIX_FADV_DONTNEED);
                    (kn[1] == 'w' ? w : sc)[k / 2] = buf[k].data();
                    k++;
                }
            m23_expert(w, sc, E->cb[0], E->cb[1], todo[i].second);
        }
        for (auto& f : fds) close(f.second);
    };
    std::vector<std::thread> th;
    const int nt = std::min<int>(m23_threads(), (int)todo.size());
    for (int i = 0; i < nt; i++) th.emplace_back(wk);
    for (auto& t : th) t.join();
}

static void load_hash(Engine* E) {
    // tools/dsv4/dsv4_pack.py hash: int32 V tmap | int64 pad | primes [2][3][8] | mult [2][4] | offs [2][24]
    std::string b = rd_file(E->o.packed + "/engram_hash.bin");
    size_t need = (size_t)V * 4 + 8 + (48 + 8 + 48) * 8;
    DCHECK(b.size() == need, "engram_hash.bin size %zu != %zu", b.size(), need);
    const char* p = b.data();
    E->hash.tmap.resize(V);
    memcpy(E->hash.tmap.data(), p, (size_t)V * 4);
    p += (size_t)V * 4;
    memcpy(&E->hash.pad, p, 8);
    p += 8;
    memcpy(E->hash.primes, p, 48 * 8);
    p += 48 * 8;
    memcpy(E->hash.mult, p, 8 * 8);
    p += 8 * 8;
    memcpy(E->hash.offs, p, 48 * 8);
}
// rows [24] of engram layer li for the token at index i of toks (the last n-1 tokens before it as history)
static void eg_hash(const Engine* E, const std::vector<int>& toks, int i, int li, int64_t* out) {
    const HashTabs& H = E->hash;
    int64_t tk[4];
    for (int sh = 0; sh < 4; sh++) tk[sh] = i - sh >= 0 ? (int64_t)H.tmap[toks[i - sh]] : H.pad;
    int64_t roll = tk[0] * H.mult[li][0];
    for (int k = 1; k < 4; k++) {
        roll ^= tk[k] * H.mult[li][k];
        for (int h = 0; h < 8; h++) out[(k - 1) * 8 + h] = roll % H.primes[li][k - 1][h] + H.offs[li][(k - 1) * 8 + h];
    }
}

// DSpark draft head on the last card: 3 window-only stages (mtp.0..2) with all 128 experts each in VRAM (FP4
// checkpoint bytes), main_proj + main_norm (stage 0), final norm, Markov head and confidence head (stage 2)
static constexpr int DS_NE = 128, DS_NOISE = 128799, DS_T0 = 37;
static double host_budget(const Options& o) {   // pinned host bytes for experts (review r4 M5)
    if (o.host_budget_gib > 0) return o.host_budget_gib * 1073741824.0;
    long pages = sysconf(_SC_PHYS_PAGES), psz = sysconf(_SC_PAGE_SIZE);
    double b = (double)pages * psz - 26.0 * 1073741824.0;   // 24 GiB OS headroom + ~2 GiB buffers
    FILE* f = fopen("/proc/meminfo", "r");   // and what is free now - 16 GiB: the rest of the process (CUDA contexts, staging, embed copy, hash tables) took ~6 GiB beyond the expert plan (R6g: 4.2 GiB left at a 10 GiB margin); OOM kill at load, R6d
    if (f) {
        char k[64];
        long v;
        while (fscanf(f, "%63s %ld kB\n", k, &v) == 2)
            if (!strcmp(k, "MemAvailable:")) b = std::min(b, v * 1024.0 - 16.0 * 1073741824.0);
        fclose(f);
    }
    return b;
}
static double dspark_bytes(const Options& o) {
    struct stat sb;
    const bool cb3 = o.dspark_cb3 && stat((o.packed + "/draft_cb3.bin").c_str(), &sb) == 0 &&
                     (size_t)sb.st_size == 3 * DS_NE * expert_bytes(XF_CB3);   // R8: as load_dspark decides
    return 3 * (190e6 + kv_bytes(o, false) + DS_NE * (double)expert_bytes(cb3 ? XF_CB3 : XF_FP4)) + 80e6 + 2 * 66.2e6 +
           (double)WIN * 3 * D * 4 + 30e6;
}
// R22 [dsv4-r22-dmoe8]: the drafter's cb3 MoE in int8 (repacked experts, prmt codebook, dp4a); DSV4_NODMOE8=1 = the
// R19 fp32 kernel (decided at load: the expert layout differs)
static const bool g_dmoe8 = getenv("DSV4_NODMOE8") == nullptr;
// R22 [dsv4-r22-di8]: the drafter's attention / shared-expert FP8 linears as int8 + dp4a; DSV4_NODI8=1 = FP8 (at load)
static const bool g_di8 = getenv("DSV4_NODI8") == nullptr;
// R22 [dsv4-r22-mk8]: Markov bias + argmax fused, int8 head; DSV4_NOMK8=1 = the R8 kernels (at load)
static const bool g_mk8 = getenv("DSV4_NOMK8") == nullptr;
static void load_dspark(Engine* E) {
    SafeTensors& st = E->stf;
    Card& c = E->cards.back();
    DCK(cudaSetDevice(c.dev));
    const size_t XB = expert_bytes(XF_FP4);
    // R23 (review r22 L8): one accounted load-time scratch (the cb3 repack, the Markov head conversion in chunks)
    uint8_t* scr = nullptr;
    size_t scrb = 0;
    auto scratch = [&](size_t b) {
        if (b > scrb) { dfree(scr, scrb); scr = (uint8_t*)dmalloc(b); scrb = b; }
        return scr;
    };
    for (int i = 0; i < 3; i++) {
        Layer& W = E->M[i];
        W.card = (int)E->cards.size() - 1;
        W.ratio = 0;
        W.lid = NL + i;
        W.dspark = true;
        std::string p = "mtp." + std::to_string(i) + ".";
        std::string a = p + "attn.";
        W.wq_a = up_f8(st, a + "wq_a");
        W.wq_b = up_f8(st, a + "wq_b");
        W.wkv = up_f8(st, a + "wkv");
        W.wo_a = up_f8(st, a + "wo_a");
        W.wo_b = up_f8(st, a + "wo_b");
        W.q_norm = up_f32(st.get(a + "q_norm.weight"));
        W.kv_norm = up_f32(st.get(a + "kv_norm.weight"));
        W.sink = up_f32(st.get(a + "attn_sink"));
        W.attn_norm = up_f32(st.get(p + "attn_norm.weight"));
        W.ffn_norm = up_f32(st.get(p + "ffn_norm.weight"));
        const char* hk[2] = {"attn", "ffn"};
        for (int j = 0; j < 2; j++) {
            W.hc_fn[j] = up_f32(st.get(p + "hc_" + hk[j] + "_fn"));
            W.hc_scale[j] = up_f32(st.get(p + "hc_" + hk[j] + "_scale"));
            W.hc_base[j] = up_f32(st.get(p + "hc_" + hk[j] + "_base"));
        }
        std::string f = p + "ffn.";
        W.gate_w = up_bf16(st.get(f + "gate.weight"));
        W.gate_b = up_f32(st.get(f + "gate.bias"));
        W.sh_w1 = up_f8_cat(st, f + "shared_experts.w1", f + "shared_experts.w3");   // R8: [w1; w3] one GEMV
        W.sh_w2 = up_f8(st, f + "shared_experts.w2");
        if (g_di8 && !W.wq_a.sq) {   // R22 [dsv4-r22-di8]: the drafter's own FP8 linears as int8 (in place; scales per row and 128); a GGUF drafter's are Q8_0 already
            for (F8* t : {&W.wq_a, &W.wq_b, &W.wkv, &W.wo_a, &W.wo_b, &W.sh_w1, &W.sh_w2}) {
                DCHECK(t->K % 128 == 0 && !t->si, "drafter int8: K %d", t->K);
                float* si = (float*)dmalloc((size_t)t->N * (t->K / 128) * 4);
                k_f8_to_i8<<<cdiv((size_t)t->N * (t->K / 128) * 32, 256), 256>>>((uint8_t*)t->w, t->s, si, t->N, t->K);
                t->si = si;
            }
            DCK(cudaDeviceSynchronize());
            for (F8* t : {&W.wq_a, &W.wq_b, &W.wkv, &W.wo_a, &W.wo_b, &W.sh_w1, &W.sh_w2}) {   // R23 (review r22 L11)
                dfree((void*)t->s, (size_t)(t->N / 32) * (t->K / 32));   // the e8m0 scales: unused by the int8 path
                t->s = nullptr;
            }
            if (!c.d_di8) {
                c.d_di8 = (int8_t*)dmalloc((size_t)2 * MAXR * NH * HD);   // main stream, then the side stream's
                c.d_di8s = (float*)dmalloc((size_t)2 * MAXR * NH * HD / 32 * 4);
                i8_smem_init();
            }
        }
        kv_alloc(E, W, false);   // [dsv4-rel2-p1]
        std::vector<int> tab(DS_NE + 1);
        for (int e = 0; e < DS_NE; e++) tab[e] = e;
        tab[DS_NE] = -1;   // the router's padding id (TOPK-wide layout, 3 used)
        W.d_slot = (int*)dmalloc(tab.size() * 4);
        DCK(cudaMemcpy(W.d_slot, tab.data(), tab.size() * 4, cudaMemcpyHostToDevice));
        const std::string dcb = E->o.packed + "/draft_cb3.bin";
        struct stat sb;
        if (E->o.dspark_cb3 && stat(dcb.c_str(), &sb) == 0 && (size_t)sb.st_size == 3 * DS_NE * expert_bytes(XF_CB3)) {
            const size_t XB3 = expert_bytes(XF_CB3), nb = (size_t)DS_NE * XB3;
            W.xcb3 = true;
            W.xslots = (uint8_t*)dmalloc(nb);
            std::vector<uint8_t> buf(nb);
            int fd = open(dcb.c_str(), O_RDONLY);
            DCHECK(fd >= 0 && pread(fd, buf.data(), nb, (off_t)(i * nb)) == (ssize_t)nb, "read %s", dcb.c_str());
            close(fd);
            DCK(cudaMemcpy(W.xslots, buf.data(), nb, cudaMemcpyHostToDevice));
            if (g_dmoe8) {   // R22 [dsv4-r22-dmoe8]: selector layout for the int8 kernel (in place)
                cb3_repack(W.xslots, DS_NE, XB3, scratch((size_t)INTER * (D / 128) * 48), 0);
                DCK(cudaDeviceSynchronize());
                if (!c.d_dq8) {
                    c.d_dq8 = (int8_t*)dmalloc((size_t)MAXR * (D + TOPK * INTER));
                    c.d_dq8s = (float*)dmalloc((size_t)MAXR * (D + TOPK * INTER) / 32 * 4);
                }
            }
            continue;
        }
        W.xslots = (uint8_t*)dmalloc((size_t)DS_NE * XB);
        for (int e = 0; e < DS_NE; e++) {
            std::string px = f + "experts." + std::to_string(e) + ".";
            uint8_t* dst = W.xslots + (size_t)e * XB;
            size_t off = 0;
            for (const char* w : {"w1", "w3", "w2"}) {
                const TensorInfo& q = st.get(px + w + ".weight");
                const TensorInfo& sc = st.get(px + w + ".scale");
                DCK(cudaMemcpy(dst + off, q.data, q.nbytes, cudaMemcpyHostToDevice));
                DCK(cudaMemcpy(dst + off + q.nbytes, sc.data, sc.nbytes, cudaMemcpyHostToDevice));
                off += q.nbytes + sc.nbytes;
            }
            DCHECK(off == XB, "dspark expert bytes %zu", off);
        }
    }
    E->mp = up_f8(st, "mtp.0.main_proj");
    E->mnorm = up_f32(st.get("mtp.0.main_norm.weight"));
    E->dnorm = up_f32(st.get("mtp.2.norm.weight"));
    E->conf_w = up_f32(st.get("mtp.2.confidence_head.proj.weight"));
    for (int k = 0; k < 2; k++) {
        const TensorInfo& t = st.get(k ? "mtp.2.markov_head.head.weight" : "mtp.2.markov_head.embed.weight");
        DCHECK(t.nbytes == (size_t)V * 256 * 2, "markov bytes %zu", t.nbytes);
        if (k && g_mk8) {   // R22 [dsv4-r22-mk8]; R23 (review r22 L8): converted in chunks, the bf16 head never on the card
            E->mk_q = (int8_t*)dmalloc((size_t)V * 256);
            E->mk_s = (float*)dmalloc((size_t)V * 4);
            const int rc = 8192;   // rows per chunk (4 MB)
            bf16* tb = (bf16*)scratch((size_t)rc * 256 * 2);
            for (int r0 = 0; r0 < V; r0 += rc) {   // (legacy stream: each copy waits for the previous chunk's kernel)
                const int nr = std::min(rc, V - r0);
                DCK(cudaMemcpy(tb, (const uint8_t*)t.data + (size_t)r0 * 512, (size_t)nr * 512, cudaMemcpyHostToDevice));
                k_mk_to_i8<<<cdiv((size_t)nr * 32, 256), 256>>>(tb, E->mk_q + (size_t)r0 * 256, E->mk_s + r0, nr);
            }
            DCK(cudaDeviceSynchronize());
            c.d_mkey = (unsigned long long*)dmalloc(64);
            continue;
        }
        bf16* d = (bf16*)dmalloc(t.nbytes);
        DCK(cudaMemcpy(d, t.data, t.nbytes, cudaMemcpyHostToDevice));
        (k ? E->mk_head : E->mk_emb) = d;
    }
    dfree(scr, scrb);
    c.d_mh = (float*)dmalloc((size_t)WIN * 3 * D * 4);
    c.d_mx = (float*)dmalloc((size_t)MAXR * D * 4);
    c.d_zero = (float*)dmalloc((size_t)MAXR * D * 4);
    DCK(cudaMemset(c.d_zero, 0, (size_t)MAXR * D * 4));
    c.d_conf = (float*)dmalloc(256);   // (R20: + the selector stats at 16)
    c.d_embf = (float*)dmalloc((size_t)MAXR * 256 * 4);
    c.d_demb = (bf16*)dmalloc((size_t)MAXR * D * 2);
    c.d_dtok = (int*)dmalloc(64);
    DCK(cudaHostAlloc(&c.h_dout, 256, cudaHostAllocPortable));   // (R20: + 15 draft-row stats at 128)
    DCK(cudaHostAlloc(&E->h_demb, (size_t)MAXR * D * 2, cudaHostAllocPortable | cudaHostAllocMapped));
    E->dspark = true;
    DCHECK(E->L[DS_T0].card == (int)E->cards.size() - 1, "DSpark: target layer %d is not on the last card", DS_T0);
    logf_("DSpark draft head loaded on card %d (3 stages, %d experts each%s, block %d)\n", c.dev, DS_NE,
          E->M[0].xcb3 ? " as cb3g128" : "", E->o.dspark_n);
}

static void mem_guard(Engine* E, bool force);
// [dsv4-pf12-pfxb]: host-expert prefetch arena carved with the pool-tail borrow (GGUF) for chunks of at least
// DSV4_PFXB_MIN rows (default 6145: where the phase regions leave no prefetch slots): DSV4_PFXB slots (default 32,
// 0 = off) of the largest GGUF expert; set at load (g_pfxa_bytes), part of borrow_need so the load plan's borrowable
// tail (in_gtail) and the carve agree
static size_t g_pfxa_bytes = 0;
static const int E_pfxb_n = std::clamp(getenv("DSV4_PFXB") ? atoi(getenv("DSV4_PFXB")) : 32, 0, 448);   // 64: 10K 4.949 s, 32: 4.84
static const int E_pfxb_min = getenv("DSV4_PFXB_MIN") ? std::max(MAXR + 1, atoi(getenv("DSV4_PFXB_MIN"))) : 6145;
static size_t pfxa_need(size_t C) { return C >= (size_t)E_pfxb_min ? (g_pfxa_bytes + 255) & ~(size_t)255 : 0; }
// R9 borrowing: bytes of the chunk-sized prefill buffers for C rows (the carve in borrow_ws, same order / alignment);
// arena: + the prefetch arena of a borrowed carve ([dsv4-pf12-pfxb]; not in the separate no-borrow allocation)
static size_t borrow_need(size_t C, int maxctx, bool arena = true) {
    auto al = [](size_t b) { return (b + 255) & ~(size_t)255; };
    const size_t XB = expert_bytes(XF_FP4);
    return al(C * HC * D * 4) + 2 * al(C * D * 4) + al(C * QLR * 4) +
           al(std::max({C * 8192, std::min<size_t>(C, 512) * 32768, (size_t)MAXR * (32768 + D)}) * 2) + al(C * ITOPK * 4) +
           al(C * (size_t)(maxctx / CAND_BS + 1)) + al((size_t)4096 * 8192 * 2) + al((size_t)2 * INTER * D * 2) +
           al((size_t)D * INTER * 2) + NST * al(XB) + (arena ? pfxa_need(C) : 0);
}
// FP4 bytes of expert (l, e) from the mapped checkpoint into dst (device), slot layout w1 w3 w2 (weight, scale each)
static void ck_copy(Engine* E, int l, int e, uint8_t* dst, cudaStream_t s);
static void ws_carve(Card& c, uint8_t* p, size_t C, int maxctx, size_t arena = 0);
static bool ws_alloc_big_all(Engine* E);
static void ws_big_carve(Engine* E, Card& c, void* p);

// review r9 M1 / R10-10 (one helper since review r14 M1): the borrowable tail [n - nborrow, n) of a card's resident list
// (rank order) holds only experts with an FP4 host copy: every other tail expert is swapped with the coldest FP4-host
// expert before the tail. Same resident set, so residency does not depend on the host RAM budget at load
static void tail_swap(Engine* E, std::vector<std::pair<int, int>>& a, int nborrow) {
    const int n = (int)a.size(), t0 = n - nborrow;
    int j = t0 - 1;
    for (int i = std::max(t0, 0); i < n; i++) {
        const int hf = E->L[a[i].second / NE].hfmt[a[i].second % NE];
        if (hf == XF_FP4) continue;
        while (j >= 0) {
            const int hj = E->L[a[j].second / NE].hfmt[a[j].second % NE];
            if (hj == XF_FP4) break;
            j--;
        }
        if (j < 0) break;
        std::swap(a[i], a[j--]);
    }
}
// R14 test hook DSV4_VRAM_STEAL_MB=MB[@card][,late]: an app that takes VRAM after the plan (review r14 L3: one card only
// with @card; ',late' takes it after the re-measure, right before the slot cudaMalloc, to run the retry loop)
static void vram_steal_hook(int k, bool late) {
    const char* st = getenv("DSV4_VRAM_STEAL_MB");
    if (!st) return;
    const char* at = strchr(st, '@');
    if (at && atoi(at + 1) != k) return;
    if ((strstr(st, "late") != nullptr) != late) return;
    void* hog = nullptr;
    if (cudaMalloc(&hog, (size_t)(atof(st) * 1048576)) != cudaSuccess) cudaGetLastError();
    else logf_("test hook: %.0f MB taken on card %d after the plan%s\n", atof(st), k, late ? " (after the re-measure)" : "");
}

#include "gguf_model.h"   // release prep [dsv4-rel-gguf]
// [dsv4-rel-hostx]: an LL12 receive buffer: device memory with P2P, else mapped pinned host memory (portable: every
// card stores / polls through the same address over PCIe); zeroed
static void* xalloc(Engine* E, size_t b) {
    if (!E->hostx) return dmalloc(b);
    void* h = nullptr;
    DCK(cudaHostAlloc(&h, b, cudaHostAllocPortable | cudaHostAllocMapped));
    memset(h, 0, b);
    return h;
}
// [dsv4-rel-ram]: an expert with no host copy (GGUF mode): its three matrices in the GGUF mapping (page cache)
static void gg_nat_ptrs(const Engine* E, int l, int e, const uint8_t* p[3]) {
    for (int m = 0; m < 3; m++) {
        size_t nb;
        p[m] = E->gg->m.expert_slice(*E->gg->m.expert_tensor(l, m), e, &nb);
        DCHECK(p[m], "GGUF: no slice for layer %d expert %d matrix %d", l, e, m);
    }
}
// a lane job: the host copy, else (GGUF) the file's bytes in place
static CpuJob lane_job(const Engine* E, const Layer& W, int e, int row, float w) {
    CpuJob J{W.hx[e], (uint8_t)W.hfmt[e], row, w};
    if (!J.base) {
        DCHECK(E->gg, "layer %d expert %d: no host copy", W.lid, e);
        gg_nat_ptrs(E, W.lid, e, J.nat);
        J.base = J.nat[0];
    }
    return J;
}
// the engine-layout bytes of expert e of layer l re-packed from the GGUF into dst (expert_bytes of its format)
// [dsv4-rel-load128]: the slices are read with pread (large sequential reads, page cache) instead of 4 KB faults on the
// MADV_RANDOM mapping (ld2 / ld3: the prefill's re-packs of the cold tail 26.8 -> 31-40 s); the same bytes go into
// gg_repack, so the result is identical
static void gg_repack_expert(const Engine* E, int l, int e, uint8_t* dst) {
    const XFmt xf = E->gg->fmt[l];
    MatView mv[3];
    expert_views(xf, dst, mv);
    const int Nm[3] = {INTER, INTER, D}, Km[3] = {D, D, INTER};
    static thread_local std::vector<uint8_t> buf;
    for (int m = 0; m < 3; m++) {
        const GgTensor& t = *E->gg->m.expert_tensor(l, m);
        size_t nb;
        const uint8_t* src = E->gg->m.expert_slice(t, e, &nb);
        DCHECK(src, "GGUF: no slice for layer %d expert %d matrix %d", l, e, m);
        const GgFile& F = E->gg->m.files()[t.file];
        const off_t fo = (off_t)(src - F.base);
        buf.resize(nb);
        size_t got = 0;
        while (got < nb) {
            const ssize_t r = pread(F.fd, buf.data() + got, nb - got, fo + (off_t)got);
            if (r < 0 && errno == EINTR) continue;
            if (r <= 0) break;
            got += (size_t)r;
        }
        gg_repack(mat_fmt(xf, m), got == nb ? buf.data() : src, Nm[m], Km[m], (uint8_t*)mv[m].q);   // (read error: the mapping)
    }
}
// [dsv4-rel-ram]: the host bytes a prefill stages for expert e into stage buffer b of card c: the host copy, else the
// GGUF expert re-packed into the card's pinned buffer b (after that buffer's previous copy has finished)
static const uint8_t* stage_src(const Engine* E, Card& c, const Layer& W, int e, int b) {
    if (W.hx[e]) return W.hx[e];
    if (!c.natbuf[b]) {
        size_t mx = 0;
        for (int l = 0; l < NL; l++) mx = std::max(mx, expert_bytes(E->gg->fmt[l]));
        DCK(cudaHostAlloc((void**)&c.natbuf[b], mx, cudaHostAllocPortable));
    }
    const auto tq = clk::now();
    DCK(cudaEventSynchronize(c.st_ready[b]));   // the last copy out of this buffer is done
    gg_repack_expert(E, W.lid, e, c.natbuf[b]);
    if (pf_on()) { std::lock_guard<std::mutex> lk(g_pf_mu); g_pf.lh[W.lid].ssrc += ms_since(tq); g_pf.lh[W.lid].nssrc++; }   // [dsv4-pf-pprof]
    return c.natbuf[b];
}
// the layer hand-off p -> c (h, pre, sel, cand of n rows) on c.st; [dsv4-rel-hostx] without P2P and for decode rows:
// D2H on p's stream into its pinned buffer, then H2D on c's stream (a staged peer copy would block the host)
static void hand_off(Engine* E, Card& c, Card& p, int n) {
    const size_t ldc = E->o.maxctx / CAND_BS + 1;
    const size_t b0 = (size_t)n * HC * D * 4, b1 = (size_t)n * HC * 4, b2 = (size_t)n * ITOPK * 4, b3 = (size_t)n * ldc;
    static const bool hostoff = getenv("DSV4_HOSTOFF") != nullptr;   // hx7: explicit D2H + H2D measured slower: opt-in
    if (!E->hostx || n > MAXR || !hostoff) {
        DCK(cudaMemcpyAsync(c.h, p.h, b0, cudaMemcpyDefault, c.st));
        DCK(cudaMemcpyAsync(c.pre, p.pre, b1, cudaMemcpyDefault, c.st));
        DCK(cudaMemcpyAsync(c.sel, p.sel, b2, cudaMemcpyDefault, c.st));
        DCK(cudaMemcpyAsync(c.cand, p.cand, b3, cudaMemcpyDefault, c.st));
        return;
    }
    if (!p.hoff) {
        DevGuard dg(p.dev);
        DCK(cudaHostAlloc((void**)&p.hoff, (size_t)MAXR * (HC * D * 4 + HC * 4 + ITOPK * 4 + ldc) + 4096, cudaHostAllocPortable));
        DCK(cudaEventCreateWithFlags(&p.ev_off, cudaEventDisableTiming));
    }
    uint8_t* q = p.hoff;
    const size_t o1 = b0, o2 = o1 + b1, o3 = o2 + b2;
    {
        DevGuard dg(p.dev);
        DCK(cudaMemcpyAsync(q, p.h, b0, cudaMemcpyDeviceToHost, p.st));
        DCK(cudaMemcpyAsync(q + o1, p.pre, b1, cudaMemcpyDeviceToHost, p.st));
        DCK(cudaMemcpyAsync(q + o2, p.sel, b2, cudaMemcpyDeviceToHost, p.st));
        DCK(cudaMemcpyAsync(q + o3, p.cand, b3, cudaMemcpyDeviceToHost, p.st));
        DCK(cudaEventRecord(p.ev_off, p.st));
    }
    DCK(cudaStreamWaitEvent(c.st, p.ev_off, 0));
    DCK(cudaMemcpyAsync(c.h, q, b0, cudaMemcpyHostToDevice, c.st));
    DCK(cudaMemcpyAsync(c.pre, q + o1, b1, cudaMemcpyHostToDevice, c.st));
    DCK(cudaMemcpyAsync(c.sel, q + o2, b2, cudaMemcpyHostToDevice, c.st));
    DCK(cudaMemcpyAsync(c.cand, q + o3, b3, cudaMemcpyHostToDevice, c.st));
}
static void xzero(Engine* E, void* p, size_t b) {
    if (E->hostx) memset(p, 0, b);
    else DCK(cudaMemset(p, 0, b));
}
// the embedding row of token t as bf16 (the checkpoint's BF16 as is; [dsv4-rel-q8] a GGUF Q8_0 row decoded: int8 x fp16
// rounded once to bf16, as the prefill GEMM's weights)
static void emb_row(const Engine* E, int t, bf16* dst) {
    if (!E->embed_q) { memcpy(dst, E->embed + (size_t)t * D, (size_t)D * 2); return; }
    const uint8_t* r = E->embed_q + (size_t)t * (D / 32) * 34;
    for (int b = 0; b < D / 32; b++) {
        uint16_t h;
        memcpy(&h, r + b * 34, 2);
        __half_raw hr;
        hr.x = h;
        const float d = __half2float(__half(hr));
        const int8_t* q = (const int8_t*)(r + b * 34 + 2);
        for (int j = 0; j < 32; j++) dst[b * 32 + j] = __float2bfloat16(d * (float)q[j]);
    }
}
// [dsv4-pf3-lazy] cuBLAS loads its kernels lazily (CUDA lazy loading): the first prefill of a process loaded them
// (~54 ms; ~1.2 s when another thread held the mmap lock, e.g. a reader of /proc/PID/smaps_rollup). The prefill's cuBLAS
// shapes run once per card here, on the empty split pool as scratch (garbage values, results unused), before the pool is
// seeded. Review pf3 M2: a cuBLAS error here (no VRAM for kernel code) is logged and ends the warm-up of that card, never
// the load. DSV4_BLASWARM=0 off, 1 = one GEMM of each kind, 2 (default) = the shapes below
// [dsv4-pf8-ovl] the owner compute stream + its cuBLAS handle, at load after the slots (their count stays; cublasCreate
// allocates its workspace), never inside a prefill; 3 cards (distributed prefill) only. Review pf8 M1: a failure (VRAM)
// leaves them unset and the prefill keeps one stream
static void ovl_init(Engine* E) {
    if (E->cards.size() != 3 || E->o.chunk <= MAXR || !E->o.pdist || !E->asplit) return;
    for (auto& c : E->cards) {
        DevGuard dg(c.dev);
        if (c.sxc) continue;
        cudaStream_t s = nullptr;
        cublasHandle_t h = nullptr;
        cudaEvent_t e0 = nullptr, e1 = nullptr;
        bool ok = cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking) == cudaSuccess &&
                  cudaEventCreateWithFlags(&e0, cudaEventDisableTiming) == cudaSuccess &&
                  cudaEventCreateWithFlags(&e1, cudaEventDisableTiming) == cudaSuccess && cublasCreate(&h) == CUBLAS_STATUS_SUCCESS;
        if (ok) ok = cublasSetStream(h, s) == CUBLAS_STATUS_SUCCESS && cublasSetMathMode(h, CUBLAS_DEFAULT_MATH) == CUBLAS_STATUS_SUCCESS;
        if (!ok) {
            if (h) cublasDestroy(h);
            if (e0) cudaEventDestroy(e0);
            if (e1) cudaEventDestroy(e1);
            if (s) cudaStreamDestroy(s);
            (void)cudaGetLastError();
            logf_("card %d: owner compute stream not created (prefill keeps one stream)\n", c.dev);
            continue;
        }
        c.sxc = s; c.blas2 = h; c.ev_ox = e0; c.ev_oxe = e1;
    }
}
static void blas_warm(Engine* E) {
    static const int lvl = getenv("DSV4_BLASWARM") ? atoi(getenv("DSV4_BLASWARM")) : 2;
    if (lvl <= 0 || lvl > 2) return;
    const auto t0 = clk::now();
    int ncall = 0;
    for (auto& c : E->cards) {
        DevGuard dg(c.dev);
        const size_t pb = c.dpool_bytes ? c.dpool_bytes : 0;
        const int nmax = std::max(MAXR + 1, std::min(E->o.chunk, lvl >= 2 ? 10240 : 2048));
        const size_t need = (size_t)nmax * HC * D * 4 + (size_t)nmax * D * 4 + (size_t)2 * INTER * D * 2 + (64 << 20);
        if (!c.dpool || pb < need) {   // (review pf3 L3: only the split pool has a byte size here)
            if (c.dpool && pb) logf_("card %d: cuBLAS warm-up skipped (pool %.2f GB < %.2f GB scratch)\n", c.dev, pb / 1e9, need / 1e9);
            else logf_("card %d: cuBLAS warm-up skipped (no split pool to use as scratch)\n", c.dev);
            continue;
        }
        uint8_t* p = c.dpool;
        float* X = (float*)p;                                       // [nmax][HC * D] fp32 (also read as bf16)
        float* Y = (float*)(p + (size_t)nmax * HC * D * 4);         // [nmax][D] fp32 (>= [nmax][2 * INTER])
        bf16* Wb = (bf16*)((uint8_t*)Y + (size_t)nmax * D * 4);     // [2 * INTER][D] bf16 (any weight)
        bool ok = true;
        auto gb = [&](const bf16* x, int ldx, int ldc, int n, int N, int K) {   // = gemm_bf16 without the abort
            if (!ok) return;
            float al = 1.f, be = 0.f;
            ok = cublasGemmEx(c.blas, CUBLAS_OP_T, CUBLAS_OP_N, N, n, K, &al, Wb, CUDA_R_16BF, ldx, x, CUDA_R_16BF, ldx, &be, Y,
                              CUDA_R_32F, ldc, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT) == CUBLAS_STATUS_SUCCESS;
            ncall += ok;
        };
        auto sg = [&](int ldx, int ldc, int n, int N, int K) {   // = sgemm without the abort
            if (!ok) return;
            float al = 1.f, be = 0.f;
            ok = cublasSgemm(c.blas, CUBLAS_OP_T, CUBLAS_OP_N, N, n, K, &al, (const float*)Wb, ldx, X, ldx, &be, Y, ldc) ==
                 CUBLAS_STATUS_SUCCESS;
            ncall += ok;
        };
        DCK(cudaMemsetAsync(p, 0, need, c.st));                     // (zeros: no NaN / denormal paths)
        std::vector<int> ns;
        if (lvl == 1) ns = {std::min(nmax, 512)};
        else for (int n = 16; n <= nmax; n *= 2) ns.push_back(n);
        if (lvl >= 2 && ns.back() != nmax) ns.push_back(nmax);
        for (int n : ns) {
            sg(HC * D, NMIX, n, NMIX, HC * D);           // hc mixes
            sg(D, NE, n, NE, D);                         // router
            gb((const bf16*)X, D, NE, n, NE, D);         // [dsv4-pf3-pfx] predictor
        }
        // routed experts on the cuBLAS path (> DSV4_FXMAX rows): w1 / w3 and w2
        std::vector<int> ms;
        if (lvl == 1) ms = {256};
        else for (int m = 129; m <= std::min(nmax, 4096); m = m < 512 ? m + 64 : m * 3 / 2) ms.push_back(m);
        for (int m : ms) {
            gb((const bf16*)X, D, 2 * INTER, m, 2 * INTER, D);
            gb((const bf16*)X, INTER, D, m, D, INTER);
        }
        const cudaError_t se = cudaStreamSynchronize(c.st);
        if (!ok || se != cudaSuccess) {
            logf_("WARNING card %d: cuBLAS warm-up stopped (%s): its kernels load at the first prefill\n", c.dev,
                  se != cudaSuccess ? cudaGetErrorString(se) : "cuBLAS error");
            (void)cudaGetLastError();
        }
    }
    logf_("cuBLAS warm-up (level %d): %d GEMMs on %zu cards in %.0f ms\n", lvl, ncall, E->cards.size(), ms_since(t0));
}
// [dsv4-pf17-egmirror] review pf17b M3: the extents of [off, off + len) of fd, merged where both the logical and the
// physical ranges continue; false (err set) on an ioctl failure, a hole or an extent whose bytes are not plainly on disk
static bool eg_fiemap(int fd, size_t off, size_t len, std::vector<std::array<uint64_t, 3>>& out, std::string& err) {
    constexpr int NX = 256;
    std::vector<uint64_t> mem((sizeof(fiemap) + NX * sizeof(fiemap_extent)) / 8 + 1);
    const uint64_t end = off + len;
    uint64_t pos = off;
    bool last = false;
    out.clear();
    while (pos < end && !last) {
        fiemap* fm = (fiemap*)mem.data();
        memset(fm, 0, mem.size() * 8);
        fm->fm_start = pos;
        fm->fm_length = end - pos;
        fm->fm_flags = FIEMAP_FLAG_SYNC;
        fm->fm_extent_count = NX;
        if (ioctl(fd, FS_IOC_FIEMAP, fm) != 0) { err = std::string("FIEMAP: ") + strerror(errno); return false; }
        if (fm->fm_mapped_extents == 0) { err = "FIEMAP: no extents"; return false; }
        for (unsigned i = 0; i < fm->fm_mapped_extents; i++) {
            const fiemap_extent& x = fm->fm_extents[i];
            if (x.fe_flags & (FIEMAP_EXTENT_UNKNOWN | FIEMAP_EXTENT_DELALLOC | FIEMAP_EXTENT_ENCODED | FIEMAP_EXTENT_DATA_INLINE |
                              FIEMAP_EXTENT_DATA_TAIL | FIEMAP_EXTENT_UNWRITTEN | FIEMAP_EXTENT_NOT_ALIGNED)) {
                err = "FIEMAP: extent not plainly on disk (flags " + std::to_string(x.fe_flags) + ")";
                return false;
            }
            if (!out.empty() && out.back()[0] + out.back()[2] == x.fe_logical && out.back()[1] + out.back()[2] == x.fe_physical)
                out.back()[2] += x.fe_length;
            else
                out.push_back({x.fe_logical, x.fe_physical, x.fe_length});
            pos = x.fe_logical + x.fe_length;
            last |= (x.fe_flags & FIEMAP_EXTENT_LAST) != 0;
        }
    }
    // no holes inside the range (a hole reads as zeros in both files, but then the "same extents" proof is void)
    if (out.empty() || out[0][0] > off || out.back()[0] + out.back()[2] < end) { err = "FIEMAP: range not fully mapped"; return false; }
    for (size_t i = 1; i < out.size(); i++)
        if (out[i][0] != out[i - 1][0] + out[i - 1][2]) { err = "FIEMAP: hole in the range"; return false; }
    return true;
}
// [dsv4-pf17-egmirror] opt-in DSV4_EGMIRROR=<mirror of Engram layer 1's table file>,<mirror of layer 14's> (',' between
// the two; ':' is accepted as the separator only when the value has no ','; an empty first entry ",<path>" mirrors only
// layer 14; one path when both tables sit in the same file mirrors both): reflink copies (cp --reflink=always on btrfs /
// XFS: no extra data blocks) of the GGUF shards that hold the Q6_K Engram tables. A second inode lifts the per-file
// O_DIRECT read cap; prefill batches read every second row through it. Review pf17b M3: a mirror is used only if it is
// another file of the same size on the same filesystem, not older than the original, and FIEMAP shows the table's byte
// range on exactly the same physical extents in both files (shared extents = the same bytes), plus 64 sampled table pages
// equal. The mirror must not be modified while the engine runs: a read error, or a changed size / mtime / ctime (checked per
// prefill batch), drops it for the rest of the process (EgReader::mirror_drop). Off by default (unset)
static bool g_egm_seen = false;
static bool g_embra_chk = true;   // [dsv4-pf10-embra] the first prefill's page-cache check (pprof on)   // eg_mirror_open ran (review pf17b L5)
static bool env_on(const char* k) { const char* v = getenv(k); return v && atoi(v) != 0; }   // review pf17b L6
static void eg_mirror_open(Engine* E) {
    g_egm_seen = true;
    const char* ev = getenv("DSV4_EGMIRROR");
    if (!ev || !*ev) return;
    std::string list(ev);
    const char sep = list.find(',') != std::string::npos ? ',' : ':';   // review pf17b L2
    std::vector<std::string> ps;
    for (size_t a = 0;;) {
        const size_t c = list.find(sep, a);
        std::string t = list.substr(a, c == std::string::npos ? std::string::npos : c - a);
        while (!t.empty() && t.front() == ' ') t.erase(t.begin());
        while (!t.empty() && t.back() == ' ') t.pop_back();
        ps.push_back(t);
        if (c == std::string::npos) break;
        a = c + 1;
    }
    if (ps.size() > 2) logf_("warning: DSV4_EGMIRROR: %zu entries, only the first two are used\n", ps.size());
    struct stat o0{}, o1{};   // review pf17b L3: both tables in one file + one path -> the path mirrors both
    if (ps.size() == 1 && E->egr.fdw[0] >= 0 && E->egr.fdw[1] >= 0 && fstat(E->egr.fdw[0], &o0) == 0 &&
        fstat(E->egr.fdw[1], &o1) == 0 && o0.st_dev == o1.st_dev && o0.st_ino == o1.st_ino)
        ps.push_back(ps[0]);
    for (int li = 0; li < 2; li++) {
        const int lay = li ? 14 : 1;
        const std::string path = li < (int)ps.size() ? ps[li] : "";
        if (E->egr.fdw[li] < 0) continue;
        if (path.empty()) {
            logf_("engram: layer %d table: no mirror given\n", lay);
            continue;
        }
        std::string why;
        const int fd = open(path.c_str(), O_RDONLY | O_DIRECT);
        struct stat a{}, b{};
        const size_t t0 = E->egr.offw[li], tl = (size_t)E->eg[li].nrows * 210;
        if (fd < 0) why = std::string("open: ") + strerror(errno);
        else if (fstat(fd, &a) != 0 || fstat(E->egr.fdw[li], &b) != 0) why = std::string("fstat: ") + strerror(errno);
        else if (a.st_size != b.st_size) why = "size " + std::to_string(a.st_size) + " != " + std::to_string(b.st_size);
        else if (a.st_dev == b.st_dev && a.st_ino == b.st_ino) why = "the same file as the original";   // (review pf17b L4)
        else if (a.st_dev != b.st_dev) why = "on another filesystem (not a reflink)";
        else if (a.st_mtim.tv_sec < b.st_mtim.tv_sec || (a.st_mtim.tv_sec == b.st_mtim.tv_sec && a.st_mtim.tv_nsec < b.st_mtim.tv_nsec))
            why = "older than the original (the original was replaced after the mirror was made)";
        else if (tl == 0 || t0 + tl > (size_t)b.st_size) why = "table range outside the file";
        if (why.empty()) {
            std::vector<std::array<uint64_t, 3>> xa, xb;
            std::string err;
            if (!eg_fiemap(fd, t0, tl, xa, err) || !eg_fiemap(E->egr.fdw[li], t0, tl, xb, err)) why = err;
            else if (xa != xb) why = "the table's extents differ (not a reflink of this file, or the original changed)";
        }
        if (why.empty()) {   // a cheap byte check on top: the header page and 63 pages inside the table
            uint8_t *p = nullptr, *q = nullptr;
            if (posix_memalign((void**)&p, 4096, 4096) != 0 || posix_memalign((void**)&q, 4096, 4096) != 0) why = "no buffer";
            const size_t pa = t0 / 4096, pn = std::max<size_t>(1, (t0 + tl) / 4096 - pa);
            uint64_t x = 0x243f6a8885a308d3ull;
            for (int k = 0; why.empty() && k < 64; k++) {
                x = x * 6364136223846793005ull + 1442695040888963407ull;
                const off_t o = (off_t)((k == 0 ? 0 : pa + (x >> 11) % pn) * 4096);
                const ssize_t r1 = pread(fd, p, 4096, o), r2 = pread(E->egr.fdw[li], q, 4096, o);
                if (r1 <= 0 || r1 != r2) why = std::string("read at ") + std::to_string(o) + " failed" + (r1 < 0 ? std::string(": ") + strerror(errno) : "");
                else if (memcmp(p, q, (size_t)r1) != 0) why = "page at offset " + std::to_string(o) + " differs";
            }
            free(p);
            free(q);
        }
        if (why.empty()) {
            E->egr.fdm[li] = fd;
            E->egr.mst[li] = a;
            logf_("engram: layer %d table also read through the mirror %s (reflink proven by FIEMAP)\n", lay, path.c_str());
        } else {
            if (fd >= 0) close(fd);
            logf_("warning: engram mirror %s for layer %d not used (%s): original file only\n", path.c_str(), lay, why.c_str());
        }
    }
}
Engine* create(const Options& opt) {
    auto t0 = clk::now();
    std::string ldtl_s;   // R24 [dsv4-r24-load]: elapsed seconds at the load steps (one log line)
    auto ldtl = [&](const char* what) {
        char b[96];
        snprintf(b, sizeof b, "%s%s %.1f", ldtl_s.empty() ? "" : ", ", what, ms_since(t0) / 1e3);
        ldtl_s += b;
    };
    Engine* E = new Engine();
    E->o = opt;
    E->ds_plain = E_dsplainstart;   // R19 [dsv4-r19-specstart]: the first sequence too (reset() sets it for the next ones)
    const Options& o = E->o;
    // checkpoint, or (release prep [dsv4-rel-gguf]) the GGUF model: its tensors under the checkpoint's names, the
    // derived data in the cache dir (o.packed points there)
    if (!o.gguf_model.empty()) {
        gm_prepare(E);
        // [dsv4-rel-q8]: qr travels as fp32 (bf16 qr is exact only for FP8-rounded values, which Q8_0 weights do not see)
        E_qrb_off = true;
    }
    else {
        if (E->o.ckpt.empty() && getenv("DSV4_CKPT")) E->o.ckpt = getenv("DSV4_CKPT");   // (no private default path)
        DCHECK(!o.ckpt.empty(), "no model: --gguf <Maya-S dir> (or a checkpoint: --ckpt <dir> / DSV4_CKPT)");
    }
    if (o.gguf_model.empty())
        for (int i = 1; i <= 48; i++) {
            char nm[64];
            snprintf(nm, sizeof nm, "/model-%05d-of-00048.safetensors", i);
            E->stf.open(o.ckpt + nm);
        }
    SafeTensors& st = E->stf;
    ldtl("gguf/ckpt open");   // R24 [dsv4-r24-load] load timeline
    // pack meta
    Json pk = Json::parse(rd_file(o.packed + "/pack.json"));
    const int hot_conv = (int)pk["hot"].i();
    Json cbj = Json::parse(rd_file(o.packed + "/codebooks.json"));
    const char* cbn[5] = {"cb3g128", "cb2g128", "eg2", "eg3", "eg4"};
    const int cbk[5] = {8, 4, 4, 8, 16};
    for (int i = 0; i < 5; i++)
        for (int j = 0; j < cbk[i]; j++) E->cb[i][j] = (float)cbj[cbn[i]][j].f();
    // usage ranks (pack.json carries the converter's ranking: rank[l][e])
    for (int l = 0; l < NL; l++) {
        E->L[l].rank.resize(NE);
        E->L[l].hfmt.resize(NE);
        for (int e = 0; e < NE; e++) {
            E->L[l].rank[e] = (int)pk["rank"][l][e].i();
            E->L[l].hfmt[e] = (int8_t)pk["fmt"][l][e].i();
        }
        E->L[l].ratio = ratio_of(l);
        E->L[l].lid = l;
    }
    {   // packs with FP4 host copies (B / C): the dynamic tier reassigns FP4 slots (no cb3 pool)
        bool fp4h = false;
        for (int l = 0; l < NL; l++)
            for (int e = 0; e < NE; e++) fp4h |= E->L[l].hfmt[e] == XF_FP4;
        if (fp4h && E->o.dyn) {
            E->dyn_fp4 = true;
            logf_("pack has FP4 host experts: dynamic tier over the FP4 slots\n");
        }
        // R16: a Maya-budget pack (every host expert M23): the pool holds M23 copies; no other codebook format beside it
        int nm23 = 0, ncb = 0;
        for (int l = 0; l < NL; l++)
            for (int e = 0; e < NE; e++) {
                nm23 += E->L[l].hfmt[e] == XF_M23;
                ncb += E->L[l].hfmt[e] == XF_CB3 || E->L[l].hfmt[e] == XF_CB2;
            }
        if (nm23) {
            DCHECK(!ncb && !fp4h, "an M23 pack takes no cb3 / cb2 / FP4 host experts beside it (%d / %d)", ncb, (int)fp4h);
            E->pfmt = XF_M23;
            E->m23conv = pk.has("convert") && pk["convert"].str == "m23";
            for (int i = 0; i + 1 < 8; i++) DCHECK(E->cb[0][i] < E->cb[0][i + 1], "cb3 codebook not ascending");   // review r16 L7
            for (int i = 0; i + 1 < 4; i++) DCHECK(E->cb[1][i] < E->cb[1][i + 1], "cb2 codebook not ascending");
            logf_("pack: %d experts in M23 (Maya byte budget: w1 / w3 cb2g128 + w2 cb3g128, %.3f MB each)%s\n", nm23,
                  expert_bytes(XF_M23) / 1e6, E->m23conv ? ", converted from the checkpoint at load" : "");
            if (E->o.band_cb3) { logf_("warning: --band-cb3 with an M23 pack: band off\n"); E->o.band_cb3 = 0; }
            if (E->o.dspark) logf_("warning: DSpark with an M23 pack is not measured (drafter stays FP4 / cb3)\n");
        }
        if (!E->o.gguf_dir.empty()) {   // R16 [dsv4-gguf-iq]: M23 pack metadata, expert bytes from the GGUF
            DCHECK(nm23 && E->o.dyn, "--gguf-experts needs an M23 pack (every expert a host format) and the dynamic tier");
            gg_setup(E, E->o.gguf_dir);
            if (g_ggsc) E->o.dyn_decay = 0.99f;   // R20 [dsv4-r20-ggsc]
        }
    }
    ldtl("pack meta");   // R24 [dsv4-r24-load] load timeline
    // ---- layer split by memory (estimate): per-layer non-expert bytes + hot experts of the converter's set
    int ndev = 0;
    DCK(cudaGetDeviceCount(&ndev));
    DCHECK(ndev >= 3, "need 3 GPUs, have %d", ndev);
    const int NC = 3;
    E->cards.resize(NC);
    const size_t XB = expert_bytes(XF_FP4);
    std::vector<double> lb(NL), lb0(NL);   // lb0: non-expert part only
    for (int l = 0; l < NL; l++) {
        double b = 175e6 + kv_bytes(o, true) + (is_kv_src(l) ? 22e6 : 0) + (is_idx_src(l) ? 6e6 : 0);
        if (is_engram(l)) b += 157e6 + 4 * D * 4;
        lb0[l] = b;
        int nh = 0;
        for (int e = 0; e < NE; e++) nh += E->L[l].rank[e] < hot_conv;
        b += (double)nh * XB;
        lb[l] = b;
    }
    for (int k = 0; k < NC; k++) {
        DCK(cudaSetDevice(k));
        DCK(cudaSetDeviceFlags(cudaDeviceScheduleSpin));   // lowest latency for the per-layer routing wait
        DCK(cudaFree(0));
        E->cards[k].dev = k;
    }
    bool p2p_fail = false;
    // release prep: DSV4_NOP2P=1 = a stock driver (no peer access enabled at all; the driver stages cross-card copies)
    const bool nop2p = getenv("DSV4_NOP2P") != nullptr;
    if (nop2p) logf_("DSV4_NOP2P: peer access not enabled (stock-driver behaviour)\n");
    auto can_peer = [&](int k, int j) {
        int ok = 0;
        return !nop2p && cudaDeviceCanAccessPeer(&ok, k, j) == cudaSuccess && ok;
    };
    for (int k = 0; k < NC; k++) {   // review L15: direct peer copies (hand-offs, compressed-row broadcasts)
        DCK(cudaSetDevice(k));
        for (int j = 0; j < NC; j++) {
            if (j != k && can_peer(k, j)) {
                cudaError_t e = cudaDeviceEnablePeerAccess(j, 0);
                if (e != cudaSuccess && e != cudaErrorPeerAccessAlreadyEnabled) {
                    logf_("peer %d -> %d: %s\n", k, j, cudaGetErrorString(e));
                    p2p_fail = true;   // review r8 L6: the splits need it (a store would fault)
                }
                cudaGetLastError();
            }
        }
    }
    if (o.asplit) {   // R8: the attention split needs direct stores between every pair of cards
        bool all = true;
        for (int k = 0; k < NC; k++)
            for (int j = 0; j < NC; j++) {
                if (j != k && !can_peer(k, j)) all = false;
            }
        E->asplit = all && !p2p_fail;
        // (GGUF pool mode only: its splits exchange through LL12 receive words and staged copies alone)
        E->hostx = !E->asplit && NC == 3 && !o.gguf_dir.empty() && !getenv("DSV4_NOHOSTX");
        if (E->hostx) E->asplit = true;
        if (E->hostx && !getenv("DSV4_NOLLGATE")) {   // [dsv4-rel-pollbo]: one poller per block on host words
            const int one = 1;
            for (int k = 0; k < NC; k++) {
                DevGuard dg(E->cards[k].dev);
                DCK(cudaMemcpyToSymbol(c_llgate, &one, sizeof one));
            }
        }
        logf_(all && !p2p_fail ? "attention split over the 3 cards (P2P)\n"
              : E->hostx ? "no P2P between every pair of cards: attention split over the 3 cards through host memory\n"
                         : "warning: no P2P between every pair of cards: attention split off\n");
    }
    // R9 expert split: needs the P2P stores of the attention split; the cb3 tiers (dynamic cb3 pool, static band) stay
    // on the owner card, so it is off with them
    E->xsplit = o.xsplit && E->asplit && !o.band_cb3 && (E->dyn_fp4 || !o.dyn);
    if (o.xsplit && !E->xsplit) logf_("expert split off (needs P2P, FP4 dynamic tier or --dyn 0, no --band-cb3)\n");
    if (E->xsplit) logf_("hot experts of every layer split over the 3 cards (P2P)\n");
    // R16: the non-FP4 pool (M23 / GGUF packs: every expert has a host copy) spread over the 3 cards, each card running its
    // own pool experts of every layer on a copy of x / ids / w (the R9 X4 / X5 exchanges); results added in routing order
    E->psplit = !E->xsplit && E->asplit && o.dyn && !E->dyn_fp4 && E->pfmt != XF_CB3 && !o.band_cb3 && !getenv("DSV4_NOPSPLIT");
    if (E->psplit) logf_("pool experts of every layer split over the 3 cards (P2P)\n");
    E->borrow = o.borrow && E->xsplit && E->dyn_fp4;
    // R19 [dsv4-r19-pool]: the split M23 / GGUF pool lends its tail to long prefills (every expert has a host copy)
    E->gborrow = o.borrow && !E->borrow && E->psplit && !getenv("DSV4_NOGBORROW");
    if (E->gborrow) logf_("workspace borrowing from the split pool's tail (prefill only)\n");
    g_pfxa_bytes = 0;
    // review pf12 M1: only where the prefetch can run (pfx_on's static conditions); E_pfx_max = DSV4_PFX_MAX > 0
    const bool pfx_can = !getenv("DSV4_NOPFX") && !(getenv("DSV4_PFX_MAX") && atoi(getenv("DSV4_PFX_MAX")) <= 0) &&
                         o.pdist && E->asplit && E->cards.size() == 3 && !E->hostx && !getenv("DSV4_NOLB") &&
                         !getenv("DSV4_PDCHK");
    if (E->gborrow && E->gg && E_pfxb_n > 0 && pfx_can) {   // [dsv4-pf12-pfxb]
        size_t mx = 0;
        for (int l = 0; l < NL; l++) mx = std::max(mx, expert_bytes(E->gg->fmt[l]));
        g_pfxa_bytes = (size_t)E_pfxb_n * mx;
        logf_("prefill prefetch arena planned: %d slots (%.2f GB per card) in the borrowed tail for chunks of >= %d rows\n",
              E_pfxb_n, g_pfxa_bytes / 1e9, E_pfxb_min);
    }
    // [dsv4-rel-hostx]: through host memory the head split needs the Q8_0 head (whole-row copies, no remote stores)
    E->hsplit = o.hsplit && E->asplit && (!E->hostx || !o.gguf_model.empty());
    if (o.borrow && !E->borrow && !E->gborrow) logf_("workspace borrowing off (needs the expert split and the FP4 dynamic tier)\n");   // r9 L5
    if (o.hsplit && !E->hsplit) logf_("output head split off (needs P2P between every pair)\n");
    ldtl("peer setup");   // R24 [dsv4-r24-load] load timeline
    g_mtag = 1;
    // ---- per card: streams, cuBLAS, rope tables, caches, workspaces, weights
    std::vector<float> cs, sn;
    // [dsv4-rel4-p2] the compressed K/V pool: one sequence's capacity in pages per source + the --kv-pool-mib share
    // (sources 1 : 1 : 1 : 2, in bytes: review rel4 L5), the same on every card (one page map); review rel4 M3: the extra
    // never exceeds the smallest free VRAM over the cards - 8 GiB (the experts need it), with a warning
    {
        size_t xb = (size_t)std::max(0, o.kv_pool_mib) << 20;
        if (xb) {
            size_t mf = SIZE_MAX;
            for (int k = 0; k < NC; k++) {
                size_t fr = 0, to = 0;
                DCK(cudaSetDevice(E->cards[k].dev));
                DCK(cudaMemGetInfo(&fr, &to));
                mf = std::min(mf, fr);
            }
            const size_t lim = mf > (8ull << 30) ? mf - (8ull << 30) : 0;
            if (xb > lim) {
                logf_("WARNING --kv-pool-mib %d: %.1f GB free on the fullest card; the extra K/V pool is %.0f MB per card\n",
                      o.kv_pool_mib, mf / 1e9, lim / 1e6);
                xb = lim;
            }
        }
        const size_t pgb = (size_t)KVPG * (HD + IDIM) * 2;
        auto kv_size = [&] {
            for (int s = 0; s < 4; s++) {
                const int cap_rows = s == 3 ? o.maxctx : o.maxctx / 2 + 1;
                E->kvcap[s] = (cap_rows + KVPG - 1) / KVPG;
                E->kvP[s] = E->kvcap[s] + (int)(xb * (s == 3 ? 2 : 1) / 5 / pgb);
                E->kvown[s].assign(E->kvP[s], 0);
                E->kvB[s] = E->kvM[s] = E->kvP[s];
            }
            E->ropeB = E->ropeM = o.maxctx;
        };
        kv_size();
        // [dsv4-ctx-lazy] a window above the base context: address space for all of it, memory for the base (GGUF split
        // pool only: the loan zones live there); DSV4_NOCTXLAZY=1 = the static reservation
        // review rel7 M4 (measured: DSV4_NOP2P=1 failed the first compressed-row broadcast, cudaMemcpyAsync Default
        // between two cards' VMM memory: invalid argument): lazy only with peer access between every pair
        bool allp = !p2p_fail;
        for (int k = 0; k < NC; k++)
            for (int j = 0; j < NC; j++) allp = allp && (j == k || can_peer(k, j));
        E->lz = E->gg && E->psplit && o.dyn && o.maxctx > lz_base_ctx() && !getenv("DSV4_NOCTXLAZY") && lzd().ok && allp;
        // [dsv4-rel8-ctxdef] the server's default context (no --max-ctx): the lazy reservation serves it for free; the
        // static one maps every position at load, so the default is ctx_static there (an explicit --max-ctx always wins)
        if (o.ctx_static > 0) {
            if (!E->lz && o.maxctx > o.ctx_static) {
                const int m0 = o.maxctx;
                E->o.maxctx = o.ctx_static;
                kv_size();
                logf_("context: %d positions, the default with the static context reservation (%s); the lazy one serves %d "
                      "by default; --max-ctx sets the context\n", o.maxctx,
                      getenv("DSV4_NOCTXLAZY") ? "DSV4_NOCTXLAZY=1" : !allp ? "it needs peer access between every pair of cards"
                                                                          : "the lazy one needs the GGUF split pool",
                      m0);
            } else {
                logf_("context: %d positions, the default%s; --max-ctx sets the context\n", o.maxctx,
                      E->lz ? " with the lazy context reservation (memory for the base context, the rest lent by the expert pool on demand)"
                            : getenv("DSV4_NOCTXLAZY") ? " with the static context reservation (DSV4_NOCTXLAZY=1)"
                            : !allp ? " with the static context reservation (the lazy one needs peer access between every pair of cards)"
                                    : " with the static context reservation (the lazy one needs the GGUF split pool and the dynamic tier)");
            }
            // review rel150 M4: the static reservation maps every position at load (cudaMalloc aborts when it does not fit):
            // a server default context halves (not below 32768) until its K/V + rope buffers leave 8 GiB free on the fullest
            // card for the experts, as the --kv-pool-mib extra above; an explicit --max-ctx is not touched
            if (!E->lz) {
                size_t mf = SIZE_MAX;
                for (int k = 0; k < NC; k++) {
                    size_t fr = 0, to = 0;
                    DCK(cudaSetDevice(E->cards[k].dev));
                    DCK(cudaMemGetInfo(&fr, &to));
                    mf = std::min(mf, fr);
                }
                auto need = [&] {
                    size_t b = (size_t)4 * o.maxctx * 32 * 4;   // rope cos / sin of both kinds
                    for (int s = 0; s < 4; s++) b += (size_t)E->kvP[s] * pgb;
                    return b;
                };
                const int m0 = o.maxctx;
                while (o.maxctx > 32768 && need() + (8ull << 30) > mf) {
                    E->o.maxctx = std::max(32768, o.maxctx / 2);
                    kv_size();
                }
                if (o.maxctx < m0)
                    logf_("WARNING context: %d positions instead of the default %d: the static context reservation maps every "
                          "position at load and the fullest card has %.1f GB free (8 GB stay for the experts); --max-ctx sets it\n",
                          o.maxctx, m0, mf / 1e9);
            }
        }
        if (!E->lz && !allp && E->gg && o.maxctx > lz_base_ctx() && !getenv("DSV4_NOCTXLAZY"))
            logf_("context reservation: static for all %d positions (the lazy one needs peer access between every pair of cards)\n", o.maxctx);
        if (E->lz) {
            const int bc = lz_base_ctx();
            for (int s = 0; s < 4; s++)
                E->kvB[s] = std::min(E->kvP[s], ((s == 3 ? bc : bc / 2 + 1) + KVPG - 1) / KVPG + (E->kvP[s] - E->kvcap[s]));
            E->ropeB = bc;
            for (int k = 0; k < NC; k++) {   // VMM access: the card itself and every peer that reaches it directly
                Card& c = E->cards[k];
                c.lz_acc.clear();
                for (int j = 0; j < NC; j++) {
                    if (j != k && (p2p_fail || !can_peer(j, k))) continue;
                    CUmemAccessDesc a = {};
                    a.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
                    a.location.id = E->cards[j].dev;
                    a.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
                    c.lz_acc.push_back(a);
                }
            }
        }
    }
    for (int k = 0; k < NC; k++) {
        Card& c = E->cards[k];
        DCK(cudaSetDevice(c.dev));
        DCK(cudaStreamCreateWithFlags(&c.st, cudaStreamNonBlocking));
        DCK(cudaStreamCreateWithFlags(&c.cp, cudaStreamNonBlocking));
        DCK(cudaStreamCreateWithFlags(&c.cpd, cudaStreamNonBlocking));
        {
            int lo = 0, hi = 0;
            DCK(cudaDeviceGetStreamPriorityRange(&lo, &hi));
            DCK(cudaStreamCreateWithPriority(&c.cpe, cudaStreamNonBlocking, hi));
        }
        DCK(cudaStreamCreateWithFlags(&c.st2, cudaStreamNonBlocking));
        c.est = c.st;
        DCK(cudaStreamCreateWithFlags(&c.cst, cudaStreamNonBlocking));
        DCK(cudaStreamCreateWithFlags(&c.cst2, cudaStreamNonBlocking));
        DCK(cudaEventCreateWithFlags(&c.ev_p0, cudaEventDisableTiming));
        DCK(cudaEventCreateWithFlags(&c.ev_p1, cudaEventDisableTiming));
        DCK(cudaEventCreateWithFlags(&c.ev_e0, cudaEventDisableTiming));
        DCK(cudaEventCreateWithFlags(&c.ev_e1, cudaEventDisableTiming));
        DCK(cudaEventCreateWithFlags(&c.ev_s0, cudaEventDisableTiming));
        DCK(cudaEventCreateWithFlags(&c.ev_s1, cudaEventDisableTiming));
        c.d_elg = (float*)dmalloc(NE * PRED_ROWS * 4);   // ([dsv4-dec1-pfmr]: R rows)
        c.d_epid = (int*)dmalloc(64 * 4);
        c.d_epw = (float*)dmalloc(64 * 4);
        DCHECK(cublasCreate(&c.blas) == CUBLAS_STATUS_SUCCESS, "cublas");
        cublasSetStream(c.blas, c.st);
        cublasSetMathMode(c.blas, CUBLAS_DEFAULT_MATH);
        if (getenv("DSV4_BLASWARM") && atoi(getenv("DSV4_BLASWARM")) == 9 && o.chunk > MAXR) {
            // [dsv4-pf3-lazy] debug: the rejected pf2 warm-up ([dsv4-pf2-blaswarm]: here, in temporary buffers), kept to
            // find why it ended in 'illegal instruction' at the first prefill
            const int mx = std::min(o.chunk, 2048);
            bf16 *tx = nullptr, *tw = nullptr;
            float* ty = nullptr;
            if (cudaMalloc(&tx, (size_t)mx * D * 2) == cudaSuccess && cudaMalloc(&tw, (size_t)2 * INTER * D * 2) == cudaSuccess &&
                cudaMalloc(&ty, (size_t)mx * D * 4) == cudaSuccess) {
                for (int m = 129; m <= mx; m = m < 256 ? m + 32 : m * 3 / 2) {
                    gemm_bf16(c.blas, tx, D, tw, D, ty, 2 * INTER, m, 2 * INTER, D);
                    gemm_bf16(c.blas, tx, INTER, tw, INTER, ty, D, m, D, INTER);
                }
                DCK(cudaStreamSynchronize(c.st));
            }
            cudaGetLastError();
            if (tx) cudaFree(tx);
            if (tw) cudaFree(tw);
            if (ty) cudaFree(ty);
        }
        for (int kind = 0; kind < 2; kind++) {
            if (E->lz) {   // [dsv4-ctx-lazy] the host tables stay for the growth; the cards map the base positions
                if (E->lz_rope[2 * kind].empty()) rope_tables(o.maxctx, kind, E->lz_rope[2 * kind], E->lz_rope[2 * kind + 1]);
                for (int t = 0; t < 2; t++) {
                    Card::LzBuf& b = c.lzb[8 + 2 * kind + t];
                    lz_buf_init(c, b, (size_t)o.maxctx * 32 * 4, (size_t)E->ropeB * 32 * 4);
                    DCK(cudaMemcpy((void*)(uintptr_t)b.va, E->lz_rope[2 * kind + t].data(), std::min(b.mapped, (size_t)o.maxctx * 32 * 4),
                                   cudaMemcpyHostToDevice));
                }
                c.cosv[kind] = (float*)(uintptr_t)c.lzb[8 + 2 * kind].va;
                c.sinv[kind] = (float*)(uintptr_t)c.lzb[9 + 2 * kind].va;
                continue;
            }
            rope_tables(o.maxctx, kind, cs, sn);
            c.cosv[kind] = (float*)dmalloc(cs.size() * 4);
            c.sinv[kind] = (float*)dmalloc(sn.size() * 4);
            DCK(cudaMemcpy(c.cosv[kind], cs.data(), cs.size() * 4, cudaMemcpyHostToDevice));
            DCK(cudaMemcpy(c.sinv[kind], sn.data(), sn.size() * 4, cudaMemcpyHostToDevice));
        }
        for (int s = 0; s < 4; s++) {
            g_mtag = 2;
            if (E->lz) {   // [dsv4-ctx-lazy]
                lz_buf_init(c, c.lzb[s], (size_t)E->kvP[s] * KVPG * HD * 2, (size_t)E->kvB[s] * KVPG * HD * 2);
                lz_buf_init(c, c.lzb[4 + s], (size_t)E->kvP[s] * KVPG * IDIM * 2, (size_t)E->kvB[s] * KVPG * IDIM * 2);
                c.src[s].pck = c.src[s].ckv = (bf16*)(uintptr_t)c.lzb[s].va;
                c.src[s].pik = c.src[s].ik = (bf16*)(uintptr_t)c.lzb[4 + s].va;
            } else {
            c.src[s].pck = c.src[s].ckv = (bf16*)dmalloc((size_t)E->kvP[s] * KVPG * HD * 2);
            c.src[s].pik = c.src[s].ik = (bf16*)dmalloc((size_t)E->kvP[s] * KVPG * IDIM * 2);
            }
            g_mtag = 1;
            c.tail_kv[s] = (float*)dmalloc(HD * 4);
            c.tail_sc[s] = (float*)dmalloc(HD * 4);
            c.snap_kv[s] = (float*)dmalloc(HD * 4);   // [dsv4-srv-pesnap] (4 KB each)
            c.snap_sc[s] = (float*)dmalloc(HD * 4);
            if (o.dspark) {
                c.sv_kv[s] = (float*)dmalloc((size_t)(MAXR + 1) * HD * 4);
                c.sv_sc[s] = (float*)dmalloc((size_t)(MAXR + 1) * HD * 4);
            }
        }
        if (E->lz) {   // [dsv4-ctx-lazy] what the base mapping holds (2 MB multiples: a little above the base context)
            for (int s = 0; s < 4; s++)
                E->kvB[s] = E->kvM[s] = std::min<int>(E->kvP[s], (int)std::min(c.lzb[s].mapped / ((size_t)KVPG * HD * 2),
                                                                                c.lzb[4 + s].mapped / ((size_t)KVPG * IDIM * 2)));
            size_t rm = SIZE_MAX;
            for (int t = 8; t < 12; t++) rm = std::min(rm, c.lzb[t].cap ? c.lzb[t].mapped / (32 * 4) : SIZE_MAX);
            E->ropeB = E->ropeM = (int)std::min<size_t>(o.maxctx, rm);
            if (k == 0) {
                size_t lm = 0;
                for (auto& b : c.lzb) lm += b.cap - b.mapped;
                logf_("context reservation: lazy, K/V and rope tables mapped for %d of %d positions; up to %.0f MB per card "
                      "come from the expert pool when a longer context arrives (DSV4_NOCTXLAZY=1: static)\n",
                      std::min(E->ropeB, 2 * (E->kvB[0] - (E->kvP[0] - E->kvcap[0])) * KVPG), o.maxctx, lm / 1e6);
            }
        }
        const size_t C = o.chunk;
        // R9 borrowing: these buffers in MAXR rows now, the chunk-sized ones carved from the tail slots per prefill
        const bool bw = E->borrow || E->gborrow;   // R19: either borrowing
        const size_t CB = bw ? (size_t)MAXR : C;
        const size_t XBR = std::max({CB * 8192, std::min<size_t>(CB, 512) * 32768, (size_t)MAXR * (32768 + D)});
        if (bw) E->borrow_bytes = borrow_need(C, o.maxctx);
        c.h = (float*)dmalloc(CB * HC * D * 4);
        for (float** p : {&c.pre, &c.apre, &c.apost, &c.fpre, &c.fpost}) *p = (float*)dmalloc(C * HC * 4);
        c.acomb = (float*)dmalloc(C * 16 * 4);
        c.fcomb = (float*)dmalloc(C * 16 * 4);
        c.mix = (float*)dmalloc(C * NMIX * 4);
        c.ss = (float*)dmalloc(C * 4);
        c.x = (float*)dmalloc(CB * D * 4);
        c.ao = (float*)dmalloc(CB * D * 4);
        c.qr = (float*)dmalloc(CB * QLR * 4);
        c.qrb = (bf16*)dmalloc((size_t)MAXR * QLR * 2);
        c.kvt = (float*)dmalloc((C + 1) * HD * 4 * 2);
        c.xb = (bf16*)dmalloc(XBR * 2);   // review M2: wo_a input; R8: + decode xq1
        c.xb_n = XBR;
        c.sel = (int*)dmalloc(CB * ITOPK * 4);
        c.cand = (uint8_t*)dmalloc(CB * (size_t)(o.maxctx / CAND_BS + 1));
        c.ids = (int*)dmalloc(C * TOPK * 4);
        c.rows = (int*)dmalloc(C * TOPK * 4);
        c.rw = (float*)dmalloc(C * TOPK * 4);
        c.rwt = (float*)dmalloc(C * TOPK * 4);
        c.f32tmp = (float*)dmalloc((size_t)MAXR * 32768 * 4);
        c.xbs = (bf16*)dmalloc((size_t)MAXR * 8192 * 2);   // R18: the shared expert's linq scratch on st2
        c.f32s = (float*)dmalloc((size_t)MAXR * 8192 * 4);
        c.f32tmp2 = (float*)dmalloc((size_t)MAXR * 32768 * 4);
        c.d_pbuf = (float*)dmalloc(((size_t)PRED_ROWS * (D + NE) + 2 * PRED_STRIDE) * 4);
        DCK(cudaEventCreateWithFlags(&c.ev_p2, cudaEventDisableTiming));
        {
            std::vector<float> one(C, 1.f);
            c.ones = (float*)dmalloc(C * 4);
            DCK(cudaMemcpy(c.ones, one.data(), C * 4, cudaMemcpyHostToDevice));
        }
        c.deq_elems = (size_t)4096 * 8192;
        if (!bw) {   // prefill only
            c.deq = (bf16*)dmalloc(c.deq_elems * 2);
            c.w13 = (bf16*)dmalloc((size_t)2 * INTER * D * 2);
            c.w2 = (bf16*)dmalloc((size_t)D * INTER * 2);
        }
        for (int i = 0; i < NST; i++) {
            if (!bw) c.stage[i] = (uint8_t*)dmalloc(XB);
            DCK(cudaEventCreateWithFlags(&c.st_ready[i], cudaEventDisableTiming));
            DCK(cudaEventCreateWithFlags(&c.st_free[i], cudaEventDisableTiming));
        }
        DCK(cudaEventCreateWithFlags(&c.ev, cudaEventDisableTiming));
        DCK(cudaEventCreateWithFlags(&c.ev_dyn, cudaEventDisableTiming));
        DCK(cudaEventCreateWithFlags(&c.ev_brw, cudaEventDisableTiming));
        DCK(cudaEventCreateWithFlags(&c.ev_route, cudaEventDisableTiming));
        DCK(cudaEventCreateWithFlags(&c.ev_eg, cudaEventDisableTiming));
        DCK(cudaEventCreateWithFlags(&c.ev_eg1, cudaEventDisableTiming));
        DCK(cudaEventCreate(&c.tp0));
        DCK(cudaEventCreate(&c.tp1));
        // (review pf3 L5: pfx_bound mirrors these layouts: change both together)
        // phase region: max of attention (sub 512 queries: q, o 2 x 512 x 32768 x 4, o2 512 x 8192 x 4),
        // indexer (64 x 32 x maxctx x 4 + 64 x maxctx x 4 x 2), MoE (C x (5120 x 2 + 4608 x 4 + 2304 x 4 + 2304 x 2 +
        // 5120 x 4)), engram (1024 x 24 x 256 x 4 + 1024 x 25600 x 4)
        size_t att = (size_t)2 * 512 * 32768 * 4 + (size_t)512 * 8192 * 4;
        size_t idx = idx_ws(o.maxctx);   // [dsv4-rel2-p1b]
        size_t moe = C * ((size_t)D * 2 + 2 * INTER * 4 + INTER * 4 + INTER * 2 + (size_t)D * 4);
        size_t egb = (size_t)1024 * EG_ROWS * EG_HD * 4 + (size_t)1024 * (HC + 1) * D * 4;
        c.phase_bytes = std::max({att + idx, moe, egb, (size_t)C * D * 4 * 2});
        c.phase = (char*)dmalloc(c.phase_bytes);
        c.dws.h = c.h; c.dws.x = c.x; c.dws.ao = c.ao; c.dws.qr = c.qr; c.dws.xb = c.xb; c.dws.xb_n = c.xb_n; c.dws.sel = c.sel; c.dws.cand = c.cand;
        c.dws.deq = c.deq; c.dws.w13 = c.w13; c.dws.w2 = c.w2;
        for (int i = 0; i < NST; i++) c.dws.stage[i] = c.stage[i];
        c.big = !bw;
        {
            size_t rb = (size_t)MAXR * TOPK * 8 + (size_t)MAXR * D * 4;
            c.d_rpack = (char*)dmalloc(rb);
            c.d_cnt = (int*)dmalloc(64);
            c.d_xqs = (bf16*)dmalloc((size_t)D * 2);
            c.d_obf = (bf16*)dmalloc((size_t)NH * HD * 2);
            DCK(cudaMemset(c.d_cnt, 0, 64));
            DCK(cudaHostAlloc(&c.h_rpack, rb, cudaHostAllocPortable));
        }
        DCK(cudaHostAlloc(&c.h_x, (size_t)MAXR * D * 4, cudaHostAllocPortable));
        DCK(cudaHostAlloc(&c.h_ids, (size_t)MAXR * TOPK * 4, cudaHostAllocPortable));
        DCK(cudaHostAlloc(&c.h_w, (size_t)MAXR * TOPK * 4, cudaHostAllocPortable));
        DCK(cudaHostAlloc(&c.h_y, (size_t)MAXR * D * 4, cudaHostAllocPortable | cudaHostAllocMapped));
        DCK(cudaHostGetDevicePointer((void**)&c.d_hy_map, c.h_y, 0));
        c.d_ycold = (float*)dmalloc((size_t)MAXR * D * 4);
        if (E->asplit) {
            g_mtag = 9;
            c.rx1 = (uint4*)xalloc(E, (size_t)MAXR * X1W * 16);
            c.rx2 = (uint4*)xalloc(E, (size_t)NC * MAXR * X2W * 16);
            c.rx3 = (uint4*)xalloc(E, (size_t)NC * MAXR * X3W * 16);
            xzero(E, c.rx1, (size_t)MAXR * X1W * 16);
            xzero(E, c.rx2, (size_t)NC * MAXR * X2W * 16);
            xzero(E, c.rx3, (size_t)NC * MAXR * X3W * 16);
            c.d_ysl = (float*)dmalloc((size_t)MAXR * 1728 * 4);
            c.rxh = (uint4*)xalloc(E, (size_t)7008 * 16);   // (20480 + 4 + 512 + 2) / 3 words
            c.rxc = (uint4*)xalloc(E, (size_t)((o.maxctx / CAND_BS + 1 + 3) / 4 + 3) / 3 * 16 + 64);
            c.rxx = (uint4*)xalloc(E, (size_t)((D + 2) / 3) * 16);
            xzero(E, c.rxh, (size_t)7008 * 16);
            xzero(E, c.rxc, (size_t)((o.maxctx / CAND_BS + 1 + 3) / 4 + 3) / 3 * 16 + 64);
            xzero(E, c.rxx, (size_t)((D + 2) / 3) * 16);
            if (E->xsplit || E->psplit) {
                const size_t w4 = (size_t)MAXR * ((D + 2 * TOPK + 2) / 3), w5 = (size_t)NC * MAXR * TOPK;
                c.rx4 = (uint4*)xalloc(E, w4 * 16);
                c.rx5 = (uint4*)xalloc(E, w5 * YW * 16);
                c.rx5h = (uint4*)xalloc(E, w5 * 16);
                xzero(E, c.rx4, w4 * 16);
                xzero(E, c.rx5, w5 * YW * 16);
                xzero(E, c.rx5h, w5 * 16);
            }
            g_mtag = 1;
            const size_t SN[3] = {(size_t)NH * HD, (size_t)OG * OLR, (size_t)D}, SK[3] = {QLR, OLR * 4, (size_t)OG * OLR};
            for (int i = 0; i < 3; i++) {
                c.stg_w[i] = (uint8_t*)dmalloc(SN[i] * SK[i]);
                // [dsv4-rel-q8]: Q8_0 weights carry an fp16 scale per row and 32 columns
                c.stg_s[i] = (uint8_t*)dmalloc(o.gguf_model.empty() ? (SN[i] / 32) * (SK[i] / 32) : SN[i] * (SK[i] / 32) * 2);
            }
        }
        c.d_gu = (float*)dmalloc((size_t)MAXR * TOPK * 2 * INTER * 4);
        c.d_hh = (float*)dmalloc((size_t)MAXR * TOPK * INTER * 4);
        c.d_yx = (float*)dmalloc((size_t)MAXR * TOPK * D * 4);
        c.d_shg = (float*)dmalloc((size_t)MAXR * 3 * INTER * 4);
        c.d_shy = (float*)dmalloc((size_t)MAXR * D * 4);
        c.part = (float*)dmalloc((size_t)MAXR * NH * 32 * (HD + 2) * 4);
        if (k == 0) c.d_embrow = (bf16*)dmalloc(C * D * 2);
        c.d_egrows = (uint8_t*)dmalloc((size_t)1024 * EG_ROWS * 264 + 16);   // +16: k_eg_decode reads 1 byte past a row
        c.d_egoff = (int*)dmalloc((size_t)1024 * EG_ROWS * 4);
        c.d_egbits = (uint8_t*)dmalloc((size_t)1024 * EG_ROWS);
        c.d_cb = (float*)dmalloc(5 * 16 * 4);
        DCK(cudaMemcpy(c.d_cb, E->cb, sizeof E->cb, cudaMemcpyHostToDevice));
        if (E->gg) {
            gg_card_init(E, c);   // R16 [dsv4-gguf-iq]
            if (!g_gg_fp32) {   // R21 [dsv4-r21-dp4a] (review r21 L4: none with the fp32 kernel chosen at start, so the
                // off-switch sizes the pools as before R21; a later "set gg_fp32 0" then stays on fp32: no buffers)
                c.d_q8 = (int8_t*)dmalloc((size_t)MAXR * (D + TOPK * INTER));
                c.d_q8s = (float*)dmalloc((size_t)MAXR * (D + TOPK * INTER) / 32 * 4);
            }
            if (g_defer) c.d_ydef = (float*)dmalloc((size_t)MAXR * HC * D * 4);   // R21 [dsv4-r21-defer] (only with DEFER)
        }
        if (o.dspark && getenv("DSV4_PIPE") && getenv("DSV4_NOENQT") == nullptr) {   // (review r19 L10: = E->pipe) R19 [dsv4-r19-pipe] (opt-in, ❌ as a default): the second micro-batch's rows
            auto& m = c.mbc;
            const size_t R8 = MAXR;
            m.h = (float*)dmalloc(R8 * HC * D * 4);
            for (float** q : {&m.pre, &m.fpre, &m.apre, &m.apost, &m.fpost}) *q = (float*)dmalloc(R8 * HC * 4);
            m.acomb = (float*)dmalloc(R8 * 16 * 4);
            m.fcomb = (float*)dmalloc(R8 * 16 * 4);
            m.mix = (float*)dmalloc(R8 * NMIX * 4);
            m.ss = (float*)dmalloc(R8 * 4);
            m.x = (float*)dmalloc(R8 * D * 4);
            m.ao = (float*)dmalloc(R8 * D * 4);
            m.qr = (float*)dmalloc(R8 * QLR * 4);
            m.qrb = (bf16*)dmalloc(R8 * QLR * 2);
            m.d_xqs = (bf16*)dmalloc((size_t)D * 2);
            m.d_obf = (bf16*)dmalloc((size_t)NH * HD * 2);
            m.sel = (int*)dmalloc(R8 * ITOPK * 4);
            m.cand = (uint8_t*)dmalloc(R8 * (size_t)(o.maxctx / CAND_BS + 1));
            m.d_rpack = (char*)dmalloc((size_t)MAXR * TOPK * 8 + (size_t)MAXR * D * 4);
            m.d_gu = (float*)dmalloc(R8 * TOPK * 2 * INTER * 4);
            m.d_hh = (float*)dmalloc(R8 * TOPK * INTER * 4);
            m.d_yx = (float*)dmalloc(R8 * TOPK * D * 4);
            if (E->gg && !g_gg_fp32) {   // R21 [dsv4-r21-dp4a]
                m.d_q8 = (int8_t*)dmalloc(R8 * (D + TOPK * INTER));
                m.d_q8s = (float*)dmalloc(R8 * (D + TOPK * INTER) / 32 * 4);
            }
            m.d_shg = (float*)dmalloc(R8 * 3 * INTER * 4);
            m.d_shy = (float*)dmalloc(R8 * D * 4);
            if (c.d_ysl) m.d_ysl = (float*)dmalloc(R8 * 1728 * 4);
            m.d_elg = (float*)dmalloc(NE * PRED_ROWS * 4);
            m.d_epid = (int*)dmalloc(64 * 4);
            m.d_epw = (float*)dmalloc(64 * 4);
        }
        c.d_tok = (int*)dmalloc(C * 4);
        c.d_nll = (float*)dmalloc(C * 4);
        DCK(cudaFuncSetAttribute(k_attn, cudaFuncAttributeMaxDynamicSharedMemorySize, AT_SMEM));
        DCK(cudaFuncSetAttribute(k_attn_s, cudaFuncAttributeMaxDynamicSharedMemorySize, AT_SMEM));   // [dsv4-dec3-atts]
        DCK(cudaFuncSetAttribute(k_attn_s, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared));   // (review atts L1: 2 blocks per SM)
        DCK(cudaFuncSetAttribute(k_attn_o, cudaFuncAttributeMaxDynamicSharedMemorySize, AT_SMEM));   // [dsv4-dec2-attl] off path
        DCK(cudaFuncSetAttribute(k_attn_tc, cudaFuncAttributeMaxDynamicSharedMemorySize, TC_SMEM));
        DCK(cudaFuncSetAttribute(k_attn_tc2, cudaFuncAttributeMaxDynamicSharedMemorySize, TC2_SMEM));   // [dsv4-pf-attn2]
    }
    ldtl("cards (streams, cuBLAS, workspaces)");   // R24 [dsv4-r24-load] load timeline
    // ---- layer split (after the split-independent workspaces): maximize the smallest spare over the 3 cards of
    // free - reserve - head (last card) - sum over its layers of (non-expert + must-be-resident experts)
    {
        std::vector<double> cap(NC);
        for (int k = 0; k < NC; k++) {
            DCK(cudaSetDevice(k));
            size_t fr, tot;
            DCK(cudaMemGetInfo(&fr, &tot));
            cap[k] = (double)fr - o.reserve_gb * 1e9 - (k == NC - 1 ? (double)MAXR * V * 4 : 0) -
                     (E->hsplit ? (double)V * D * 2 / NC : k == NC - 1 ? (double)V * D * 2 : 0);
            if (k == NC - 1 && o.dspark) cap[k] -= dspark_bytes(o);
            // R8 shares: the per-layer estimate counts a layer's full wq_b / wo_a / wo_b (117.4 MB) on its owner; with
            // the split the owner keeps 32.7 MB of it, every other card 42.3 MB, + 117 MB staging + rings (review r8 M1:
            // per layer count, in search() below)
        }
        std::vector<double> pre(NL + 1, 0);
        for (int l = 0; l < NL; l++) pre[l + 1] = pre[l] + lb[l];
        double best = -1e30;
        int ba = 1, bb = 2;
        const bool per_layer = pre[NL] - std::accumulate(lb0.begin(), lb0.end(), 0.0) < 1e9;   // must-resident < 1 GB
        // review r8 M1: the attention-split correction of a card holding L layers (owner share 32.7 MB of the 117.4 MB the
        // estimate counts per own layer, 42.3 MB of every other layer, + 124 MB staging and rings)
        auto acorr = [&](int L) { return E->asplit ? 124e6 + 42.3e6 * (NL - L) - (117.4e6 - 32.7e6) * L : 0.0; };
        auto search = [&] {   // DSpark: its target layers 37..39 must sit on the last card (y <= DS_T0)
            best = -1e30;
            for (int x = 1; x < NL - 1; x++)
                for (int y = x + 1; y < (o.dspark ? DS_T0 + 1 : NL); y++) {
                    const double c0 = cap[0] - acorr(x), c1 = cap[1] - acorr(y - x), c2 = cap[2] - acorr(NL - y);
                    double m = std::min({c0 - pre[x], c1 - (pre[y] - pre[x]), c2 - (pre[NL] - pre[y])});
                    if (per_layer)   // no must-resident set (every expert has a host copy): the pools are per layer,
                        m = std::min({(c0 - pre[x]) / x, (c1 - (pre[y] - pre[x])) / (y - x),   // so balance
                                      (c2 - (pre[NL] - pre[y])) / (NL - y)});                    // slots per layer (R7-6)
                    if (m > best) {
                        best = m;
                        ba = x;
                        bb = y;
                    }
                }
        };
        search();
        auto take_placement = [&] {   // the earlier run's split too (review M3: same layout across restarts)
            if (o.placement_in.empty()) return;
            Json pj = Json::parse(rd_file(o.placement_in));
            if (pj.has("split")) {
                ba = (int)pj["split"][0].i();
                bb = (int)pj["split"][1].i();
                best = std::min({cap[0] - acorr(ba) - pre[ba], cap[1] - acorr(bb - ba) - (pre[bb] - pre[ba]),
                                 cap[2] - acorr(NL - bb) - (pre[NL] - pre[bb])});
            }
        };
        take_placement();
        double ne_last = 0;
        for (int l = bb; l < NL; l++) ne_last += lb0[l];
        // review r6 M4 / r7 M5: the drafter leaves the last card less than 1 GB for expert slots beyond its non-expert
        // layers (must-resident experts may degrade to the lane, the drafter may not), or a loaded split puts a DSpark
        // target layer (37..39) off the last card: the model runs, DSpark off
        if (o.dspark && (cap[NC - 1] - acorr(NL - bb) - ne_last < 1e9 || bb > DS_T0)) {
            logf_("warning: DSpark needs %.2f GB on card %d that is not there (left for experts %.2f GB, split at %d): DSpark off\n",
                  dspark_bytes(o) / 1e9, NC - 1, (cap[NC - 1] - ne_last) / 1e9, bb);
            E->o.dspark = 0;
            cap[NC - 1] += dspark_bytes(o);
            search();
            take_placement();
        }
        E->cards[0].l0 = 0;
        E->cards[0].l1 = E->cards[1].l0 = ba;
        E->cards[1].l1 = E->cards[2].l0 = bb;
        E->cards[2].l1 = NL;
        for (int k = 0; k < NC; k++) {
            Card& c = E->cards[k];
            for (int l = c.l0; l < c.l1; l++) E->L[l].card = k;
            double ld = (k == 0 ? pre[ba] : k == 1 ? pre[bb] - pre[ba] : pre[NL] - pre[bb]);
            logf_("card %d: layers %d..%d, free %.2f GB, planned non-expert + must-resident %.2f GB, spare %.2f GB\n",
                  k, c.l0, c.l1 - 1, cap[k] / 1e9, ld / 1e9, (cap[k] - ld) / 1e9);
        }
        if (best <= 0)   // degrade: the must-resident experts that find no slot run from host RAM (FP4, checkpoint)
            logf_("WARNING: the pack's resident set does not fit (smallest spare %.2f GB): the lowest-ranked must-resident "
                  "experts get FP4 host copies from the checkpoint\n", best / 1e9);
    }
    ldtl("layer split");   // R24 [dsv4-r24-load] load timeline
    // ---- layer weights
    g_mtag = 3;
    for (int l = 0; l < NL; l++) {
        Layer& W = E->L[l];
        DCK(cudaSetDevice(E->cards[W.card].dev));
        std::string p = "layers." + std::to_string(l) + ".";
        std::string a = p + "attn.";
        // R10: [wq_a; wkv] in one allocation (one decode GEMV over both: same input), wq_a / wkv are views
        W.wqkv = up_f8_cat(st, a + "wq_a", a + "wkv");
        W.wq_a = W.wqkv;
        W.wq_a.N = QLR;
        W.wkv = W.wqkv;
        W.wkv.N = HD;
        W.wkv.w = W.wqkv.w + (size_t)QLR * W.wqkv.K;
        if (W.wqkv.sq) W.wkv.sq = W.wqkv.sq + (size_t)QLR * (W.wqkv.K / 32);
        else W.wkv.s = W.wqkv.s + (size_t)(QLR / 32) * (W.wqkv.K / 32);
        if (E->asplit) {   // R8: only the per-card shares are stored (below); prefill stages the full matrices
            auto shape = [&](const std::string& nm) {
                const TensorInfo& w = st.get(nm + ".weight");
                F8 f;
                f.N = (int)w.shape[0];
                f.K = (int)w.shape[1];
                return f;
            };
            W.wq_b = shape(a + "wq_b");
            W.wo_a = shape(a + "wo_a");
            W.wo_b = shape(a + "wo_b");
        } else {
            W.wq_b = up_f8(st, a + "wq_b");
            W.wo_a = up_f8(st, a + "wo_a");
            W.wo_b = up_f8(st, a + "wo_b");
        }
        W.q_norm = up_f32(st.get(a + "q_norm.weight"));
        W.kv_norm = up_f32(st.get(a + "kv_norm.weight"));
        W.sink = up_f32(st.get(a + "attn_sink"));
        W.attn_norm = up_f32(st.get(p + "attn_norm.weight"));
        W.ffn_norm = up_f32(st.get(p + "ffn_norm.weight"));
        const char* hk[2] = {"attn", "ffn"};
        for (int i = 0; i < 2; i++) {
            W.hc_fn[i] = up_f32(st.get(p + "hc_" + hk[i] + "_fn"));
            W.hc_scale[i] = up_f32(st.get(p + "hc_" + hk[i] + "_scale"));
            W.hc_base[i] = up_f32(st.get(p + "hc_" + hk[i] + "_base"));
        }
        if (is_kv_src(l)) {
            if (st.get(a + "compressor.wkv.weight").dtype == "Q8_0") {   // [dsv4-rel-q8]: GGUF compressor linears
                W.cq_wkv = up_q8(st.get(a + "compressor.wkv.weight"));
                if (W.ratio > 1) W.cq_wgate = up_q8(st.get(a + "compressor.wgate.weight"));
            } else {
                W.c_wkv = up_bf16(st.get(a + "compressor.wkv.weight"));
                if (W.ratio > 1) W.c_wgate = up_bf16(st.get(a + "compressor.wgate.weight"));
            }
            W.c_norm = up_f32(st.get(a + "compressor.norm.weight"));
            W.i_wk = up_bf16(st.get(a + "indexer.wk.weight"));
            W.i_knorm = up_f32(st.get(a + "indexer.k_norm.weight"));
        }
        if (is_idx_src(l)) {
            W.i_wq_b = up_f8(st, a + "indexer.wq_b");
            W.i_wproj = up_bf16(st.get(a + "indexer.weights_proj.weight"));
        }
        std::string f = p + "ffn.";
        W.gate_w = up_bf16(st.get(f + "gate.weight"));
        W.gate_b = up_f32(st.get(f + "gate.bias"));
        W.sh_w1 = up_f8_cat(st, f + "shared_experts.w1", f + "shared_experts.w3");   // R8: [w1; w3] one GEMV
        W.sh_w2 = up_f8(st, f + "shared_experts.w2");
        if (is_engram(l)) {
            W.e_wkv = up_f8(st, p + "engram.wkv");
            if (st.has(p + "engram.qk")) {   // [dsv4-rel-gguf]: the GGUF stores the product q x k (F32, = ours bitwise)
                const TensorInfo& t = st.get(p + "engram.qk");
                DCHECK(t.dtype == "F32" && t.nbytes == (size_t)HC * D * 4, "engram gate %s %zu", t.dtype.c_str(), t.nbytes);
                W.e_qk = up_f32(t);
            } else {
            const TensorInfo& q = st.get(p + "engram.q_weight");
            const TensorInfo& kk = st.get(p + "engram.k_weight");
            std::vector<float> qk((size_t)HC * D);
            for (size_t i = 0; i < qk.size(); i++) {
                uint32_t a1 = (uint32_t)((const uint16_t*)q.data)[i] << 16, b1 = (uint32_t)((const uint16_t*)kk.data)[i] << 16;
                float fa, fb;
                memcpy(&fa, &a1, 4);
                memcpy(&fb, &b1, 4);
                qk[i] = fa * fb;
            }
            W.e_qk = (float*)dmalloc(qk.size() * 4);
            DCK(cudaMemcpy(W.e_qk, qk.data(), qk.size() * 4, cudaMemcpyHostToDevice));
            }
        }
        g_mtag = 5;
        kv_alloc(E, W, true);   // [dsv4-rel2-p1]
        g_mtag = 10;
        W.d_slot = (int*)dmalloc(NE * 4);
        W.h_slot.assign(NE, -1);
        DCK(cudaMemset(W.d_slot, 0xff, NE * 4));
        for (int k = 0; k < NC; k++) {   // R9: local slot tables on every card (the owner's only without the split)
            if (k != W.card && !E->xsplit) continue;
            DevGuard dg(E->cards[k].dev);
            W.d_ls[k] = (int*)dmalloc(NE * 4);
            DCK(cudaMemset(W.d_ls[k], 0xff, NE * 4));
        }
        if (E->psplit)
            for (int k = 0; k < NC; k++) {   // R16: local pool tables on every card
                DevGuard dg(E->cards[k].dev);
                W.d_dls[k] = (int*)dmalloc(NE * 4);
                DCK(cudaMemset(W.d_dls[k], 0xff, NE * 4));
            }
        W.d_dslot = (int*)dmalloc(NE * 4);
        W.h_dslot.assign(NE, -1);
        DCK(cudaMemcpy(W.d_dslot, W.h_dslot.data(), NE * 4, cudaMemcpyHostToDevice));
        W.score.assign(NE, 0.f);
        W.score_t.assign(NE, 0);
        W.miss_t.assign(NE, -1000000);
        W.hx.assign(NE, nullptr);
        g_mtag = 3;
    }
    ldtl("layer weights");   // R24 [dsv4-r24-load] load timeline
    g_mtag = 4;
    if (E->asplit) {   // R8: attention shares (owner: 2 wo_a groups, the others 3 each; wo_b rows 1696 / 1728 / 1696)
        auto view = [](const F8& f, int r0, int nr) {
            F8 v = f;
            v.w = f.w + (size_t)r0 * f.K;
            if (f.sq) v.sq = f.sq + (size_t)r0 * (f.K / 32);
            else v.s = f.s + (size_t)(r0 / 32) * (f.K / 32);
            v.N = nr;
            return v;
        };
        auto rows = [&](const std::string& name, int r0, int nr) {   // own copy of rows [r0, r0 + nr) of an FP8 weight
            if (st.get(name + ".weight").dtype == "Q8_0") return up_q8(st.get(name + ".weight"), r0, nr);
            const TensorInfo& w = st.get(name + ".weight");
            const TensorInfo& s = st.get(name + ".scale");
            F8 f;
            f.N = nr;
            f.K = (int)w.shape[1];
            size_t wb = (size_t)nr * f.K, sb = (size_t)(nr / 32) * (f.K / 32);
            uint8_t* dw = (uint8_t*)dmalloc(wb);
            uint8_t* ds = (uint8_t*)dmalloc(sb);
            DCK(cudaMemcpy(dw, (const uint8_t*)w.data + (size_t)r0 * f.K, wb, cudaMemcpyHostToDevice));
            DCK(cudaMemcpy(ds, (const uint8_t*)s.data + (size_t)(r0 / 32) * (f.K / 32), sb, cudaMemcpyHostToDevice));
            f.w = dw;
            f.s = ds;
            return f;
        };
        const int NNr[3] = {1696, 1728, 1696};
        for (int l = 0; l < NL; l++) {
            Layer& W = E->L[l];
            int order[3] = {W.card, -1, -1}, q = 1;
            for (int k = 0; k < NC; k++)
                if (k != W.card) order[q++] = k;
            int g0 = 0, n0 = 0;
            for (int i = 0; i < NC; i++) {
                const int k = order[i];
                Layer::AShare& A = W.ash[k];
                A.g0 = g0;
                A.ng = i == 0 ? 2 : 3;
                A.n0 = n0;
                A.nn = NNr[i];
                g0 += A.ng;
                n0 += A.nn;
                std::string a = "layers." + std::to_string(l) + ".attn.";
                DCK(cudaSetDevice(E->cards[k].dev));
                (void)view;
                A.wq_b = rows(a + "wq_b", A.g0 * AT_HG * HD, A.ng * AT_HG * HD);   // each matrix stored once over the cards
                A.wo_a = rows(a + "wo_a", A.g0 * OLR, A.ng * OLR);
                A.wo_b = rows(a + "wo_b", A.n0, A.nn);
                if (i == 0) A.sink = W.sink;
                else {
                    A.sink = up_f32(st.get(a + "attn_sink"));
                    A.ring = (bf16*)dmalloc((size_t)ARING * HD * 2);
                }
            }
            DCHECK(g0 == OG && n0 == D, "attention shares");
        }
    }
    ldtl("attention shares");   // R24 [dsv4-r24-load] load timeline
    g_mtag = 6;
    for (int k = 0; k < NC; k++) {   // R17: FP8 GEMV x staging up to 96 KB of shared memory on every card
        DevGuard dg(E->cards[k].dev);
        int optin = 0;   // R18 (review r17 L5): the device must allow it (sm_86: 99 KB; sm_75 / sm_89 differ)
        DCK(cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, E->cards[k].dev));
        DCHECK(optin >= F8_SMEM_MAX, "card %d allows %d B of shared memory per block, the FP8 GEMV staging needs %d: "
               "rebuild with -DF8_SMEM_MAX=%d", k, optin, F8_SMEM_MAX, 48 * 1024);
        f8_smem_init();
        if (E->gm && !E->cards[k].d_di8) {   // [dsv4-rel-q8]: every card runs Q8_0 linears on int8 x (main + side stream halves)
            Card& ck = E->cards[k];
            ck.d_di8 = (int8_t*)dmalloc((size_t)2 * MAXR * NH * HD);
            ck.d_di8s = (float*)dmalloc((size_t)2 * MAXR * NH * HD / 32 * 4);
            i8_smem_init();
        }
        if (!g_topk_old_h) {   // R20 [dsv4-r20-idxskip] opt-in: the new top-k order (review r20 L1: device default = R19 order)
            const bool t = false;
            DCK(cudaMemcpyToSymbol(g_topk_old, &t, sizeof t));
        }
        DCK(cudaGetLastError());
    }
    {   // head, final norm on the last card
        Card& c = E->cards[NC - 1];
        DCK(cudaSetDevice(c.dev));
        const TensorInfo& h = st.get("head.weight");
        const bool hq8 = h.dtype == "Q8_0";   // [dsv4-rel-q8]: the GGUF head (half the bytes of BF16)
        if (E->hsplit) {   // R9: rows in thirds (multiples of 8 = the GEMV's rows per block)
            for (int k = 0; k <= NC; k++) E->hr[k] = k == NC ? V : (int)((size_t)V * k / NC / 8 * 8);
            for (int k = 0; k < NC; k++) {
                DevGuard dg(E->cards[k].dev);
                if (hq8) E->head_qk[k] = up_q8(h, E->hr[k], E->hr[k + 1] - E->hr[k]);
                else {
                    size_t nb = (size_t)(E->hr[k + 1] - E->hr[k]) * D * 2;
                    E->head_k[k] = (bf16*)dmalloc(nb);
                    DCK(cudaMemcpy(E->head_k[k], (const char*)h.data + (size_t)E->hr[k] * D * 2, nb, cudaMemcpyHostToDevice));
                }
                E->cards[k].d_hx = (float*)dmalloc((size_t)MAXR * D * 4);
                // R10: own vocab slice (graph head); [dsv4-rel-q8] Q8_0: MAXR rows (the slices are computed locally,
                // then copied as whole rows into the last card's logits)
                if (k < NC - 1) E->cards[k].d_logits = (float*)dmalloc((size_t)(hq8 ? MAXR : 1) * V * 4);
                DCK(cudaEventCreateWithFlags(&E->ev_hd[k], cudaEventDisableTiming));
            }
            DCK(cudaEventCreateWithFlags(&E->ev_hx, cudaEventDisableTiming));
            logf_("output head split over the cards: rows");   // (R23, review r23 L8: the cards there are)
            for (int k = 0; k < NC; k++) logf_(" %s%d", k ? "/ " : "", E->hr[k + 1] - E->hr[k]);
            logf_("\n");
        } else if (hq8) E->head_q = up_q8(h);
        else {
            E->head = (bf16*)dmalloc(h.nbytes);
            DCK(cudaMemcpy(E->head, h.data, h.nbytes, cudaMemcpyHostToDevice));
        }
        E->fnorm = up_f32(st.get("norm.weight"));
        for (size_t k = 0; k < E->cards.size() && k < 3; k++) {
            DevGuard dg(E->cards[k].dev);
            DCK(cudaEventCreateWithFlags(&E->ev_probe[k], cudaEventDisableTiming));
        }
        c.d_logits = (float*)dmalloc((size_t)MAXR * V * 4);
        const TensorInfo& em = st.get("embed.weight");
        if (em.dtype == "Q8_0") {   // [dsv4-rel-q8]
            DCHECK(em.nbytes == (size_t)V * (D / 32) * 34, "embed.weight Q8_0 size");
            E->embed_q = (const uint8_t*)em.data;
            // ([dsv4-pf10-embra]: the table's readahead is issued at the end of create(), after the seeding's DONTNEED)
        } else {
            DCHECK(em.nbytes == (size_t)V * D * 2, "embed.weight size");
            E->embed = (const bf16*)em.data;
        }
        DCK(cudaHostAlloc(&E->h_emb, (size_t)o.chunk * D * 2, cudaHostAllocPortable | cudaHostAllocMapped));   // R17: decode rows read in place
        if (o.dspark) {
            E->h_noise.resize(D);
            emb_row(E, DS_NOISE, E->h_noise.data());
            g_mtag = 7;
            load_dspark(E);
            g_mtag = 6;
        }
    }
    ldtl("head");
    logf_("load timeline (s): %s\n", ldtl_s.c_str());
    logf_("non-expert weights loaded in %.1f s\n", ms_since(t0) / 1e3);
    g_mtag = 8;
    // ---- expert slots: what is left on each card; hot set = best ranks of the card's layers. Every expert without
    // a host copy must be resident.
    {
        auto t1 = clk::now();
        std::vector<int> order;
        E->hbudget = host_budget(o);   // once, before anything is pinned
        double dem_left = E->hbudget - 1073741824.0;   // RAM left for demotions: budget - pack copies - 1 GiB
        {   // review r7 L8: minus the Engram tables in RAM and the bounce buffers (taken off the budget later too)
            const Json& ej = pk["engram"];
            if (!(ej.has("source") && ej["source"].str == "checkpoint"))
                for (int li = 0; li < 2; li++) dem_left -= (double)ej["rows"][li].i() * (8 + 32 * (double)ej["bits"].i());
            if (E->dyn_fp4) dem_left -= (double)NC * XJ_PER_CARD * XJ_BYTES;
        }
        for (int l = 0; l < NL; l++)
            for (int e = 0; e < NE; e++)
                if (E->L[l].hfmt[e] >= 0) dem_left -= expert_bytes((XFmt)E->L[l].hfmt[e]);
        if (o.band_cb3 && !E->dyn_fp4 && o.dyn) {   // review r8 H1: both would use c.dpool / c.ndyn
            logf_("warning: --band-cb3 with the cb3 dynamic tier (pack without FP4 host copies) is not supported: band off\n");
            E->o.band_cb3 = 0;
        }
        std::vector<std::pair<int, int>> band;   // R8 mixed VRAM tier: (layer, expert) in band_cb3.bin order
        if (o.band_cb3) {
            Json bj = Json::parse(rd_file(o.packed + "/band_cb3.json"));
            for (auto& v : bj["experts"].arr) band.push_back({(int)v[0].i(), (int)v[1].i()});
            struct stat sb;
            DCHECK(stat((o.packed + "/band_cb3.bin").c_str(), &sb) == 0 &&
                   (size_t)sb.st_size == band.size() * expert_bytes(XF_CB3), "band_cb3.bin size");
        }
        // R9 expert split: one ranked candidate list over all layers, every expert of it to the card with the least
        // traffic share of its layer so far (G2' usage shares, PACK/usage.json; the owner starts with its shared
        // expert's ~1.9 expert loads); slot counts from each card's free VRAM
        std::vector<std::vector<std::pair<int, int>>> assigned(NC);
        if (E->xsplit) {
            std::vector<std::vector<float>> share(NL, std::vector<float>(NE, 0.f));
            bool have_share = false;
            {
                struct stat sb;
                if (stat((o.packed + "/usage.json").c_str(), &sb) == 0) {
                    Json uj = Json::parse(rd_file(o.packed + "/usage.json"));
                    for (int l = 0; l < NL; l++)
                        for (int e = 0; e < NE; e++) share[l][e] = (float)uj["share"][l][e].num;
                    have_share = true;
                }
            }
            if (!have_share) {
                logf_("warning: no %s/usage.json: expert placement balanced by rank (tools/dsv4/dsv4_usage.py writes it)\n", o.packed.c_str());
                for (int l = 0; l < NL; l++)
                    for (int e = 0; e < NE; e++) share[l][e] = 1.f / (50.f + (float)E->L[l].rank[e]);
            }
            std::vector<int> cap(NC);
            size_t keep = (size_t)(o.reserve_gb * 1e9);
            for (int k = 0; k < NC; k++) {
                DCK(cudaSetDevice(E->cards[k].dev));
                size_t fr, tot;
                DCK(cudaMemGetInfo(&fr, &tot));
                cap[k] = fr > keep ? (int)((fr - keep) / XB) : 0;
                E->cards[k].nslots = cap[k];
            }
            std::vector<std::pair<int, int>> cand;
            int must = 0;
            for (int l = 0; l < NL; l++)
                for (int e = 0; e < NE; e++) {
                    cand.push_back({E->L[l].rank[e], l * NE + e});
                    must += E->L[l].hfmt[e] < 0;
                }
            std::sort(cand.begin(), cand.end());
            int nsum = cap[0] + cap[1] + cap[2];
            int ns = std::min<int>(nsum, (int)cand.size());
            std::vector<int> pin_card(NL * NE, -1);   // R11: card of each resident expert from --placement-in
            if (!o.placement_in.empty()) {   // review M3: the exact resident set of an earlier run
                Json pj = Json::parse(rd_file(o.placement_in));
                std::vector<std::pair<int, int>> fixed;
                const bool hc = pj.has("card");
                for (int l = 0; l < NL; l++) {
                    const auto& rl = pj["resident"][l].arr;
                    // review r11 L4: a "card" list of another length (edited / truncated file) is ignored for the layer
                    const bool hcl = hc && (int)pj["card"].arr.size() > l && pj["card"][l].arr.size() == rl.size();
                    if (hc && !hcl)
                        logf_("warning: %s layer %d: card list does not match the resident list: cards from this run\n",
                              o.placement_in.c_str(), l);
                    std::vector<char> seen(NE, 0);   // review r12 L1: ids out of range / twice in a layer are skipped
                    for (size_t j = 0; j < rl.size(); j++) {
                        const int e = (int)rl[j].i();
                        if (e < 0 || e >= NE || seen[e]) {
                            logf_("warning: %s layer %d: expert id %d out of range or repeated: skipped\n",
                                  o.placement_in.c_str(), l, e);
                            continue;
                        }
                        seen[e] = 1;
                        fixed.push_back({E->L[l].rank[e], l * NE + e});
                        if (hcl) pin_card[l * NE + e] = (int)pj["card"][l][j].i();
                    }
                }
                std::sort(fixed.begin(), fixed.end());
                if ((int)fixed.size() > nsum) {   // R11: degrade (the coldest ranks of the file run on the lane)
                    logf_("WARNING placement needs %zu slots, %d fit: the %zu coldest of the file stay in host RAM\n",
                          fixed.size(), nsum, fixed.size() - nsum);
                    fixed.resize(nsum);
                }
                cand = fixed;
                ns = (int)fixed.size();
                if (!hc) logf_("warning: %s has no per-expert cards: expert split from this run's free VRAM\n", o.placement_in.c_str());
            }
            if (nsum < must)
                logf_("WARNING: only %d slots for %d experts without a host copy: %d of them will run from host RAM\n", nsum,
                      must, must - nsum);
            std::vector<std::vector<float>> load(NL, std::vector<float>(NC, 0.f));
            for (int l = 0; l < NL; l++) load[l][E->L[l].card] = 1.88f / TOPK;   // shared expert (35 MB FP8) on the owner
            std::vector<int> used(NC, 0);
            int nmoved = 0, nbad = 0, mv_l = -1, mv_e = -1, mv_k = -1;
            for (int i = 0; i < ns; i++) {
                int l = cand[i].second / NE, e = cand[i].second % NE;
                int b = -1;
                const int pk2 = pin_card[l * NE + e];
                if (pk2 >= 0 && pk2 < NC && used[pk2] < cap[pk2]) b = pk2;   // R11: as the file says
                else {   // review r11 M1: pinned card full (or no pin): balance over the cards with free slots
                    if (pk2 >= NC) nbad++;   // review r12 L2: a card id of another card count, counted apart
                    else if (pk2 >= 0 && nmoved++ == 0) { mv_l = l; mv_e = e; mv_k = pk2; }
                    for (int k = 0; k < NC; k++) {
                        if (used[k] >= cap[k]) continue;
                        if (b < 0 || load[l][k] < load[l][b] - 1e-9f ||
                            (fabsf(load[l][k] - load[l][b]) <= 1e-9f && cap[k] - used[k] > cap[b] - used[b]))
                            b = k;
                    }
                }
                if (b < 0) break;
                assigned[b].push_back(cand[i]);
                used[b]++;
                load[l][b] += share[l][e];
            }
            if (nmoved)   // review r11 M1: one summary line instead of one per expert
                logf_("warning: %d experts of %s found their card full (first: layer %d expert %d card %d): balanced instead\n",
                      nmoved, o.placement_in.c_str(), mv_l, mv_e, mv_k);
            if (nbad)
                logf_("warning: %d experts of %s name a card >= %d (file from another card count): balanced instead\n",
                      nbad, o.placement_in.c_str(), NC);
            double imb = 0, tl = 0;   // traffic-weighted: the busiest card's share over the mean, per layer
            for (int l = 0; l < NL; l++) {
                float mx = 0, sm = 0;
                for (int k = 0; k < NC; k++) { mx = std::max(mx, load[l][k]); sm += load[l][k]; }
                imb += mx;
                tl += sm / NC;
            }
            const int C0 = E->o.chunk;
            if (E->borrow) {   // R9: the tail slots (borrowed by long prefills) hold only experts with a host copy
                // (rank order: the tail is the coldest; must-resident ones there get FP4 host copies below)
                // review r9 M3: a card with too few slots degrades (smaller prefill chunk, else no borrowing and
                // prefill in rows of MAXR) instead of aborting the load
                auto fits = [&] {
                    for (int k = 0; k < NC; k++)
                        if ((int)((borrow_need(E->o.chunk, o.maxctx) + XB - 1) / XB) >= (int)assigned[k].size()) return false;
                    return true;
                };
                while (!fits() && E->o.chunk > 256) E->o.chunk /= 2;
                if (!fits()) {
                    E->borrow = false;
                    E->o.chunk = MAXR;
                    logf_("WARNING borrowing off: too few expert slots for the prefill workspace; prefill runs in rows of %d\n", MAXR);
                } else {
                    if (E->o.chunk != C0) logf_("WARNING prefill chunk %d -> %d (expert slots too few to borrow the workspace)\n", C0, E->o.chunk);
                    E->borrow_bytes = borrow_need(E->o.chunk, o.maxctx);
                }
                for (int k = 0; k < NC && E->borrow; k++) {
                    E->cards[k].nborrow = (int)((E->borrow_bytes + XB - 1) / XB);
                    // review r9 M1: no cb3 / cb2-host expert in the tail (a borrowed one would run lossy in prefill), and
                    // (R10-10) no expert without a host copy either: swap them with the coldest FP4-host experts before
                    // the tail. Same resident set per card, so residency no longer depends on the host RAM budget at load
                    // (it decided how many tail experts got checkpoint copies: run-to-run md5 changes)
                    tail_swap(E, assigned[k], E->cards[k].nborrow);
                }
            }
            logf_("expert split: slots %d / %d / %d of %d / %d / %d, busiest card per layer %.2f x the mean (traffic shares)\n",
                  used[0], used[1], used[2], cap[0], cap[1], cap[2], imb / std::max(1e-9, tl));
        }
        for (int k = 0; k < NC; k++) {
            Card& c = E->cards[k];
            DCK(cudaSetDevice(c.dev));
            if (!band.empty()) {   // the card's band experts first (cb3 pool = c.dpool, table d_dslot), then FP4 slots
                const size_t XB3 = expert_bytes(XF_CB3);
                std::vector<int> mine;
                for (size_t i = 0; i < band.size(); i++)
                    if (band[i].first >= c.l0 && band[i].first < c.l1) mine.push_back((int)i);
                c.ndyn = (int)mine.size();
                c.dpool = (uint8_t*)dmalloc((size_t)std::max(1, c.ndyn) * XB3);
                std::vector<uint8_t> buf(XB3);
                int fd = open((o.packed + "/band_cb3.bin").c_str(), O_RDONLY);
                DCHECK(fd >= 0, "open band_cb3.bin");
                for (int j = 0; j < c.ndyn; j++) {
                    auto [l, e] = band[mine[j]];
                    DCHECK(pread(fd, buf.data(), XB3, (off_t)mine[j] * XB3) == (ssize_t)XB3, "read band_cb3.bin");
                    DCK(cudaMemcpy(c.dpool + (size_t)j * XB3, buf.data(), XB3, cudaMemcpyHostToDevice));
                    E->L[l].h_dslot[e] = j;
                }
                close(fd);
                for (int l = c.l0; l < c.l1; l++)
                    DCK(cudaMemcpy(E->L[l].d_dslot, E->L[l].h_dslot.data(), NE * 4, cudaMemcpyHostToDevice));
                logf_("card %d: %d band experts resident as cb3 (%.2f GB)\n", k, c.ndyn, c.ndyn * XB3 / 1e9);
            }
            size_t fr, tot;
            DCK(cudaMemGetInfo(&fr, &tot));
            size_t keep = (size_t)(o.reserve_gb * 1e9);
            if (!E->xsplit) c.nslots = fr > keep ? (int)((fr - keep) / XB) : 0;
            std::vector<std::pair<int, int>> cand;   // (rank, l * NE + e)
            int must = 0;
            if (!E->xsplit)
            for (int l = c.l0; l < c.l1; l++)
                for (int e = 0; e < NE; e++) {
                    if (E->L[l].h_dslot[e] >= 0) continue;   // R8: in the cb3 band
                    cand.push_back({E->L[l].rank[e], l * NE + e});
                    must += E->L[l].hfmt[e] < 0;
                }
            if (!E->xsplit && c.nslots < must)   // degrade (review M4): the rest of the must set runs from host RAM (FP4 from the checkpoint)
                logf_("WARNING card %d: only %d slots for %d experts without a host copy (free %.2f GB): %d of them will be "
                      "loaded into host RAM from the checkpoint and run on the CPU lane (slower decode)\n", k, c.nslots,
                      must, fr / 1e9, must - c.nslots);
            std::sort(cand.begin(), cand.end());
            int ns = std::min<int>(c.nslots, (int)cand.size());
            if (o.dyn && !E->dyn_fp4) ns = std::min(must, c.nslots);   // cb3 tier: static FP4 = the experts without a host copy
            if (E->xsplit) {
                cand = assigned[k];
                ns = (int)cand.size();
            } else if (!o.placement_in.empty()) {   // review M3: the exact resident set of an earlier run
                Json pj = Json::parse(rd_file(o.placement_in));
                std::vector<std::pair<int, int>> fixed;
                for (int l = c.l0; l < c.l1; l++) {
                    std::vector<char> seen(NE, 0);   // review r13 L6: the r12 L1 range / repeat check here too
                    for (auto& v : pj["resident"][l].arr) {
                        const int e = (int)v.i();
                        if (e < 0 || e >= NE || seen[e]) {
                            logf_("warning: %s layer %d: expert id %d out of range or repeated: skipped\n", o.placement_in.c_str(), l, e);
                            continue;
                        }
                        seen[e] = 1;
                        fixed.push_back({E->L[l].rank[e], l * NE + e});
                    }
                }
                std::sort(fixed.begin(), fixed.end());
                DCHECK((int)fixed.size() <= c.nslots, "card %d: placement needs %zu slots, %d fit", k, fixed.size(), c.nslots);
                cand = fixed;
                ns = (int)fixed.size();
            }
            {   // R14: VRAM can shrink between the plan and this allocation (a browser / desktop app on the card): measure
                // again, keep the reserve, drop the coldest planned experts (they get host copies below), retry smaller
                vram_steal_hook(k, false);
                size_t fr2, tot2;
                DCK(cudaMemGetInfo(&fr2, &tot2));
                const int ns0 = ns;
                const size_t slack = keep / 2;
                // review r15 M1: the borrowable tail + 1 must fit beside half the reserve; if not, degrade as fits() does
                // at the plan: a smaller prefill chunk (smaller tail on every card), else borrowing off (prefill in rows
                // of MAXR; the tails of cards already allocated become plain slots). Never an abort here
                auto set_tails = [&] {
                    E->borrow_bytes = borrow_need(E->o.chunk, o.maxctx);
                    for (auto& cc : E->cards) cc.nborrow = (int)((E->borrow_bytes + XB - 1) / XB);
                };
                auto tail_step = [&] {   // one degrade step: halve the chunk, or borrowing off below 256
                    if (E->o.chunk > 256) {
                        E->o.chunk /= 2;
                        set_tails();
                        return;
                    }
                    E->borrow = false;
                    E->o.chunk = MAXR;
                    for (auto& cc : E->cards) cc.nborrow = 0;
                };
                auto tail_fits = [&] { return (size_t)std::min(ns, c.nborrow + 1) * XB + slack <= fr2; };
                auto tail_log = [&](int C1) {
                    if (!E->borrow)
                        logf_("WARNING card %d: %.2f GB free now: borrowing off (the borrowable tail does not fit); prefill "
                              "runs in rows of %d\n", k, fr2 / 1e9, MAXR);
                    else
                        logf_("WARNING card %d: %.2f GB free now: prefill chunk %d -> %d (borrowable tail %d slots)\n", k,
                              fr2 / 1e9, C1, E->o.chunk, c.nborrow);
                    for (int j = 0; j < k; j++) {   // review r15 N2: the earlier cards' tails as they are now
                        const Card& cj = E->cards[j];
                        int occ = 0;
                        for (int l = 0; l < NL; l++)
                            for (int e = 0; e < NE; e++) occ += E->L[l].h_slot[e] >= 0 && gs_card(E->L[l].h_slot[e]) == j;
                        if (E->borrow)
                            logf_("  card %d: borrowable tail now slots %d..%d; %d of its %d slots empty\n", j,
                                  cj.used_slots - cj.nborrow, cj.used_slots - 1, cj.used_slots - occ, cj.used_slots);
                        else
                            logf_("  card %d: no borrowable tail now; %d of its %d slots empty\n", j, cj.used_slots - occ,
                                  cj.used_slots);
                    }
                };
                if (E->borrow && c.nborrow > 0 && !tail_fits()) {
                    const int C1 = E->o.chunk;
                    while (E->borrow && !tail_fits()) tail_step();
                    tail_log(C1);
                }
                int nmin = E->borrow ? std::min(ns, c.nborrow + 1) : 0;
                // review r14 M2: shrink only when the slots would take more than half of the reserve; ordinary desktop
                // jitter (tens to a few hundred MB) stays inside the reserve, so the resident set (and a --placement-in
                // check) stays as planned
                if (fr2 + slack < (size_t)ns * XB + keep) ns = std::max(nmin, (int)((fr2 > keep ? fr2 - keep : 0) / XB));
                vram_steal_hook(k, true);
                void* sp = nullptr;
                while (cudaMalloc(&sp, std::max<size_t>((size_t)ns * XB, 16)) != cudaSuccess) {
                    cudaGetLastError();
                    DCK(cudaMemGetInfo(&fr2, &tot2));   // review r14 L3: the messages show the value after the retry
                    if (ns <= nmin && E->borrow) {   // review r15 N1: VRAM taken after the measure: degrade the tail too
                        const int C1 = E->o.chunk;
                        tail_step();
                        tail_log(C1);
                        nmin = E->borrow ? std::min(ns, c.nborrow + 1) : 0;
                        continue;   // the same ns once more, then smaller down to the new nmin
                    }
                    DCHECK(ns > nmin, "card %d: cannot allocate even %d expert slots (%.2f GB free)", k, ns, fr2 / 1e9);
                    // first try: what fits beside the full reserve now; then 16 smaller per failure
                    const int fit = (int)((fr2 > keep ? fr2 - keep : 0) / XB);
                    ns = std::max(nmin, fit < ns ? fit : ns - 16);
                }
                c.slots = (uint8_t*)sp;
                g_mb[k][g_mtag] += (double)ns * XB;
                if (ns < ns0) {
                    // review r14 M1 / r15 L1: cut the coldest of the plan (rank order), but keep the nborrow coldest
                    // FP4-host experts as the new borrowable tail (they cost no host RAM there); review r15 M2: only the
                    // planned experts [0, ns0) are candidates for the cut
                    std::sort(cand.begin(), cand.end());
                    cand.resize(std::min(cand.size(), (size_t)ns0));
                    const int need = E->borrow ? c.nborrow : 0;
                    std::vector<std::pair<int, int>> tailv, rest;
                    for (int i = (int)cand.size() - 1; i >= 0; i--) {
                        const int l = cand[i].second / NE, e = cand[i].second % NE;
                        if ((int)tailv.size() < need && E->L[l].hfmt[e] == XF_FP4) tailv.push_back(cand[i]);
                        else rest.push_back(cand[i]);
                    }
                    std::reverse(tailv.begin(), tailv.end());
                    std::reverse(rest.begin(), rest.end());
                    const int keepn = std::max(0, ns - (int)tailv.size());
                    int ndrop_must = 0;
                    for (int i = keepn; i < (int)rest.size(); i++) {
                        const int l = rest[i].second / NE, e = rest[i].second % NE;
                        if (E->L[l].hfmt[e] < 0) {
                            ndrop_must++;
                            dem_left -= XB;   // review r14 L2: its checkpoint copy (degrade loop below) comes out of the budget
                        }
                    }
                    rest.resize(keepn);
                    cand = rest;
                    cand.insert(cand.end(), tailv.begin(), tailv.end());
                    int tail_must = 0;
                    if ((int)tailv.size() < need) {   // too few FP4-host experts left: the swap fills what it can
                        tail_swap(E, cand, need);
                        for (int i = ns - need; i < ns; i++)
                            tail_must += i >= 0 && E->L[cand[i].second / NE].hfmt[cand[i].second % NE] != XF_FP4;
                    }
                    logf_("WARNING card %d: %.2f GB free now (less than planned): %d instead of %d expert slots, the %d coldest "
                          "planned experts of the card (outside the borrowable tail) run from host RAM (%d of them get "
                          "checkpoint copies)%s\n", k, fr2 / 1e9, ns, ns0, ns0 - ns, ndrop_must,
                          o.placement_in.empty() ? "" : ": residency differs from the --placement-in file");
                    if (tail_must)
                        logf_("WARNING card %d: %d experts without an FP4 host copy in the borrowable tail (too few FP4-host "
                              "experts left after the cut)\n", k, tail_must);
                }
                cand.resize(std::min(cand.size(), (size_t)ns));
            }
            c.used_slots = ns;
            for (int i = 0; i < ns; i++) {
                int l = cand[i].second / NE, e = cand[i].second % NE;
                Layer& W = E->L[l];
                W.h_slot[e] = k * GS + i;   // R9: global slot id
                std::string px = "layers." + std::to_string(l) + ".ffn.experts." + std::to_string(e) + ".";
                uint8_t* dst = c.slots + (size_t)i * XB;
                size_t off = 0;
                for (const char* w : {"w1", "w3", "w2"}) {
                    const TensorInfo& q = st.get(px + w + ".weight");
                    const TensorInfo& s = st.get(px + w + ".scale");
                    DCK(cudaMemcpy(dst + off, q.data, q.nbytes, cudaMemcpyHostToDevice));
                    DCK(cudaMemcpy(dst + off + q.nbytes, s.data, s.nbytes, cudaMemcpyHostToDevice));
                    off += q.nbytes + s.nbytes;
                }
                DCHECK(off == XB, "expert bytes %zu", off);
            }
            logf_("card %d: %d expert slots (%.2f GB), free before %.2f GB\n", k, ns, ns * XB / 1e9, fr / 1e9);
            if (E->dyn_fp4 && (o.pool_min > 0 || E->borrow)) {   // demote must-resident experts of the lowest ranks into the pool
                int elig = 0, ndem = 0;
                for (int i = 0; i < ns; i++) elig += E->L[cand[i].second / NE].hfmt[cand[i].second % NE] >= 0;
                // R9: every borrowable tail slot must hold an expert with a host copy (evicted / restored per prefill)
                // R11: with --placement-in the borrowable tail always gets its copies (a check's residency must not
                // depend on the host RAM budget at load; ~92 experts = 1.6 GiB per card beyond the budget at most)
                const bool tail_all = E->borrow && !o.placement_in.empty();
                int nover = 0;
                for (int i = ns - 1; i >= 0 && (elig + ndem < o.pool_min || (E->borrow && i >= ns - c.nborrow)) &&
                                     (dem_left >= (double)XB || (tail_all && i >= ns - c.nborrow)); i--) {
                    int l = cand[i].second / NE, e = cand[i].second % NE;
                    Layer& W = E->L[l];
                    if (W.hfmt[e] >= 0) continue;
                    uint8_t* dst = host_alloc(XB, &E->n_pageable);
                    std::string px = "layers." + std::to_string(l) + ".ffn.experts." + std::to_string(e) + ".";
                    size_t off = 0;
                    for (const char* wn : {"w1", "w3", "w2"}) {
                        const TensorInfo& q = st.get(px + wn + ".weight");
                        const TensorInfo& sc = st.get(px + wn + ".scale");
                        memcpy(dst + off, q.data, q.nbytes);
                        memcpy(dst + off + q.nbytes, sc.data, sc.nbytes);
                        off += q.nbytes + sc.nbytes;
                    }
                    W.hx[e] = dst;
                    W.hfmt[e] = XF_FP4;
                    if (W.ck.empty()) W.ck.assign(NE, 0);
                    W.ck[e] = 1;
                    if (W.own.empty()) W.own.assign(NE, 0);
                    W.own[e] = 1;
                    ndem++;
                    nover += dem_left < (double)XB;
                    dem_left -= XB;
                }
                if (nover) logf_("card %d: %d tail experts demoted beyond the host RAM budget (--placement-in: fixed residency)\n", k, nover);
                logf_("card %d: %d must-resident experts demoted into the dynamic pool (FP4 host copies from the "
                      "checkpoint, %.2f GiB)\n", k, ndem, ndem * XB / 1073741824.0);
                if (E->borrow) {
                    // review r9 M3: a tail expert still without a host copy (host RAM budget short) leaves its slot (the
                    // slot stays empty, the expert runs from host RAM via the must-resident degrade path below)
                    int nfree = 0;
                    for (int i = ns - c.nborrow; i < ns; i++) {
                        int l = cand[i].second / NE, e = cand[i].second % NE;
                        if (E->L[l].hfmt[e] >= 0) continue;
                        E->L[l].h_slot[e] = -1;
                        nfree++;
                    }
                    if (nfree) logf_("WARNING card %d: %d tail experts without a host copy (host RAM budget) leave VRAM for "
                                     "the borrowable region (they run on the CPU lane)\n", k, nfree);
                    logf_("card %d: slots %d..%d borrowed by long prefills (%.2f GB)\n", k, ns - c.nborrow, ns - 1,
                          E->borrow_bytes / 1e9);
                }
            }
            if (E->dyn_fp4) {   // pools: after all cards (R9: a layer's pool slots sit on every card)
                DCK(cudaHostAlloc(&c.h_upd, 4096 * 4, cudaHostAllocPortable));
                c.xj = new XJob[XJ_PER_CARD];
                for (int j = 0; j < XJ_PER_CARD; j++) DCK(cudaHostAlloc((void**)&c.xj[j].buf, XJ_BYTES, cudaHostAllocPortable));
            } else if (o.dyn && E->gg && !E->psplit) {   // R16 [dsv4-gguf-iq]: one region per layer (its own expert size)
                size_t fr2, tot2;
                DCK(cudaMemGetInfo(&fr2, &tot2));
                gg_pool_alloc(E, c, fr2 > keep ? fr2 - keep : 0);
                DCK(cudaHostAlloc(&c.h_upd, 4096 * 4, cudaHostAllocPortable));
            } else if (o.dyn) {
                size_t fr2, tot2;
                DCK(cudaMemGetInfo(&fr2, &tot2));
                const size_t XB3 = expert_bytes(E->pfmt);
                int nl = E->psplit ? NL : c.l1 - c.l0;   // R16 pool split: this card holds a share of every layer
                int per = fr2 > keep ? (int)((fr2 - keep) / XB3) / nl : 0;
                c.ndyn = per * nl;
                if (E->psplit) {   // equal bytes per layer, slots by the layer's expert size
                    // review r19 M1: one sizing for any card (a card sized before may be sized again, below)
                    auto psize = [&](Card& c, int k) {
                        DevGuard dgp(c.dev);
                        size_t kp = keep;   // (review L7: the re-sizing margin stays with this card)
                        int tries = 0;
                    for (int lzt = 0; lzt < 2; lzt++) {   // [dsv4-ctx-lazy]: a second sizing after mapping the context in full
                        size_t fr2, tot2;
                        DCK(cudaMemGetInfo(&fr2, &tot2));
                    size_t bpl = fr2 > kp ? (fr2 - kp) / NL : 0;
                    // R19 test knob DSV4_POOL_BPL_MB=a,b,c: cap each card's pool bytes per layer (an exactness check across
                    // days needs the same pool when the desktop's VRAM use moved; residency changes the numerics)
                    if (const char* v = getenv("DSV4_POOL_BPL_MB")) {
                        int ki = 0;
                        for (const char* q = v; *q; ki++) {
                            if (ki == k) bpl = std::min(bpl, (size_t)(atof(q) * 1e6));
                            const char* cm = strchr(q, ',');
                            if (!cm) break;
                            q = cm + 1;
                        }
                    }
                    size_t off = 0;
                    int sidx = 0;
                    for (int l = 0; l < NL; l++) {
                        const size_t xb = expert_bytes(pool_fmt(E, l));
                        c.ploff[l] = off;
                        c.ps0[l] = sidx;
                        c.pnl[l] = (int)(bpl / xb);
                        off += (size_t)c.pnl[l] * xb;
                        sidx += c.pnl[l];
                    }
                    c.ndyn = sidx;
                    if (E->lz && lzt == 0 && !lz_zones(E, c, off)) {   // [dsv4-ctx-lazy] no room to lend: this card maps it all now
                        bool ok = true;
                        size_t base0[12];
                        for (int j = 0; j < 12; j++) base0[j] = c.lzb[j].base;
                        for (auto& b : c.lzb) ok = ok && lz_buf_full(c, b);
                        if (!ok) lz_cap_ctx(E, c, base0, kp);   // [dsv4-rel8-ctxcap] cap the context, never abort (rel7 L6)
                        for (int t = 8; t < 12; t++)
                            DCK(cudaMemcpy((void*)(uintptr_t)c.lzb[t].va, E->lz_rope[t - 8].data(), std::min(c.lzb[t].cap, (size_t)o.maxctx * 32 * 4),
                                           cudaMemcpyHostToDevice));
                        logf_("card %d: pool too small for loan zones: the context buffers are mapped in full (%d positions)\n", c.dev,
                              E->ctx_cap ? E->ctx_cap : o.maxctx);   // review rel8 L5
                        lz_sync_base(E);   // review rel7 M2
                        continue;
                    }
                    bool zn = false;
                    for (int l = 0; l < NL; l++) zn |= c.lz_nmax[l] > 0;
                    bool got = false;
                    if (E->lz && zn) {
                        got = lz_pool_alloc(E, c, off);
                    } else {
                        static std::atomic<int> test_fail{getenv("DSV4_TEST_POOLFAIL") ? atoi(getenv("DSV4_TEST_POOLFAIL")) : 0};
                        void* pp = nullptr;
                        got = test_fail.fetch_sub(1) <= 0 && cudaMalloc(&pp, std::max<size_t>(off, 1)) == cudaSuccess;
                        cudaGetLastError();
                        if (got) {
                            c.dpool = (uint8_t*)pp;
                            if (c.dev >= 0 && c.dev < 3) g_mb[c.dev][g_mtag] += (double)std::max<size_t>(off, 1);
                        }
                    }
                    if (!got) {   // [kb-dsv4-load] the VRAM went to another process since the sizing: size again, smaller
                        DCHECK(++tries <= 8, "card %d: the expert pool could not be mapped after 8 sizings (VRAM taken by "
                               "another process while loading)", c.dev);
                        kp += (size_t)512 << 20;   // (a margin more each time)
                        logf_("WARNING card %d: the expert pool (%.2f GB) could not be mapped: VRAM was taken meanwhile; "
                              "sizing it again from the free VRAM now\n", c.dev, off / 1e9);
                        lzt--;
                        continue;
                    }
                    c.dpool_bytes = off;
                    break;
                    }
                    };
                    psize(c, k);
                    per = c.pnl[NL / 2];
                    // R19: the borrowed tail must hold the prefill buffers of a full chunk. Review r19 M1: if this card's
                    // pool cannot, borrowing goes off for the engine (degrade, never abort): every card gets its chunk
                    // buffers, the cards sized so far give the room back from their pools and are sized again
                    // [dsv4-pf12-pfxb]: the prefetch arena goes first (the borrow itself stays)
                    if (E->gborrow && g_pfxa_bytes && c.dpool_bytes < E->borrow_bytes + 256 &&
                        c.dpool_bytes >= borrow_need(E->o.chunk, o.maxctx, false) + 256) {
                        logf_("WARNING card %d: pool %.2f GB too small for the prefill prefetch arena: arena off\n", k,
                              c.dpool_bytes / 1e9);
                        g_pfxa_bytes = 0;
                        E->borrow_bytes = borrow_need(E->o.chunk, o.maxctx);
                    }
                    if (E->gborrow && c.dpool_bytes < E->borrow_bytes + 256) {
                        logf_("WARNING card %d: pool %.2f GB < prefill buffers %.2f GB: workspace borrowing off\n", k,
                              c.dpool_bytes / 1e9, E->borrow_bytes / 1e9);
                        E->gborrow = false;
                        for (int j = 0; j <= k; j++) {
                            Card& cj = E->cards[j];
                            DevGuard dgj(cj.dev);
                            pool_free(cj);   // [dsv4-ctx-lazy] (a VMM pool too)
                        }
                        ws_alloc_big_all(E);   // review r20 L2: every card or none (no stranded buffers)
                        for (int j = 0; j <= k; j++) {
                            Card& cj = E->cards[j];
                            DevGuard dgj(cj.dev);
                            {
                                psize(cj, j);
                                if (j < k) logf_("card %d: pool sized again without borrowing: %d slots, %.2f GB\n", j,
                                                 cj.ndyn, cj.dpool_bytes / 1e9);
                            }
                        }
                        per = c.pnl[NL / 2];
                    }
                } else
                c.dpool = (uint8_t*)dmalloc((size_t)std::max(c.ndyn, 1) * XB3);
                if (!E->psplit)
                for (int l = c.l0; l < c.l1; l++) {
                    Layer& W = E->L[l];
                    for (int i = 0; i < per; i++) W.dpool.push_back((l - c.l0) * per + i);
                    W.dyn_e.assign(per, -1);
                    W.pend_e.assign(per, -1);
                    W.pend_job.assign(per, -1);   // review r7 H1: dyn_apply reads these in both tiers
                    W.pend_t.assign(per, 0);
                    W.pend_ev.resize(per);
                    for (auto& ev : W.pend_ev) DCK(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming));
                }
                DCK(cudaHostAlloc(&c.h_upd, 4096 * 4, cudaHostAllocPortable));
                double pgb = c.ndyn * (double)XB3;   // review re-r N1: per-layer expert sizes in the split layout (GGUF)
                if (E->psplit) {
                    pgb = 0;
                    for (int l = 0; l < NL; l++) pgb += (double)c.pnl[l] * expert_bytes(pool_fmt(E, l));
                }
                logf_("card %d: dynamic tier %d %s slots (%d per layer%s, %.2f GB)\n", k, c.ndyn,
                      E->gg ? "GGUF" : E->pfmt == XF_M23 ? "M23" : "cb3", per, E->psplit ? (E->gg ? " of all 40 at the mid layers" : " of all 40") : "",
                      pgb / 1e9);
            }
        }
        blas_warm(E);   // [dsv4-pf3-lazy]: on the empty pools (scratch), before they are seeded
        ovl_init(E);    // [dsv4-pf8-ovl]
        g_mtag = 10;
        // device slot tables: the owner's routing table (global ids) and every card's local table (R9)
        for (int l = 0; l < NL; l++) {
            Layer& W = E->L[l];
            {
                DevGuard dg(E->cards[W.card].dev);
                DCK(cudaMemcpy(W.d_slot, W.h_slot.data(), NE * 4, cudaMemcpyHostToDevice));
            }
            for (int k = 0; k < NC; k++) {
                if (!W.d_ls[k]) continue;
                std::vector<int> t(NE, -1);
                for (int e = 0; e < NE; e++)
                    if (W.h_slot[e] >= 0 && gs_card(W.h_slot[e]) == k) t[e] = gs_idx(W.h_slot[e]);
                DevGuard dg(E->cards[k].dev);
                DCK(cudaMemcpy(W.d_ls[k], t.data(), NE * 4, cudaMemcpyHostToDevice));
            }
        }
        if (E->psplit) {   // review r16 L5: no pool slot on any card: no split (no empty exchanges every layer)
            int tot = 0;
            for (int k = 0; k < NC; k++) tot += E->cards[k].ndyn;
            if (!tot) {
                if (E->gborrow) {   // review r19 M1: degrade (chunk buffers allocated, borrowing off), never abort
                    E->gborrow = false;
                    logf_("WARNING no pool slots on any card: workspace borrowing off\n");
                    ws_alloc_big_all(E);
                }
                E->psplit = false;
                logf_("pool split off: no pool slots on any card\n");
            }
        }
        if (E->psplit) {   // R16: layer l's pool = slots on every card, interleaved card by card (seeding spreads the load)
            for (int l = 0; l < NL; l++) {
                Layer& W = E->L[l];
                int mx = 0;
                for (int k = 0; k < NC; k++) mx = std::max(mx, E->cards[k].pnl[l]);
                for (int i = 0; i < mx; i++)
                    for (int k = 0; k < NC; k++)
                        if (i < E->cards[k].pnl[l]) W.dpool.push_back(k * GS + E->cards[k].ps0[l] + i);
                const size_t n = W.dpool.size();
                W.dyn_e.assign(n, -1);
                W.pend_e.assign(n, -1);
                W.pend_job.assign(n, -1);
                W.pend_t.assign(n, 0);
                W.pend_m.assign(n, -1);       // [dsv4-rel-xswap]
                W.pend_h.assign(n, nullptr);
                W.pend_ev.resize(n);
                for (size_t i = 0; i < n; i++) {   // on the slot's card (its copy stream records it)
                    DevGuard dg(E->cards[gs_card(W.dpool[i])].dev);
                    DCK(cudaEventCreateWithFlags(&W.pend_ev[i], cudaEventDisableTiming));
                }
            }
        }
        E->swap = E->o.swap && E->dyn_fp4;
        if (!E->swap && !E->gg && E->o.dyn && E->o.dyn_margin_set)   // re-review N5 (R20: the GGUF pool takes it)
            logf_("warning: --dyn-margin applies to the swap tier only (swap off here): ignored\n");
        if (E->dyn_fp4) {   // pool of each layer = its slots holding an expert with an FP4 host copy
            std::vector<int> np(NC, 0);
            for (int l = 0; l < NL; l++) {
                Layer& W = E->L[l];
                W.inpool.assign(NE, 0);
                for (int e = 0; e < NE; e++)   // R13 swap tier: the experts without a host copy too (never in the tail)
                    if (W.h_slot[e] >= 0 && (W.hfmt[e] >= 0 || (E->swap && !in_tail(E, W.h_slot[e])))) {   // pack D: cb3-host experts too (FP4 from NVMe)
                        W.dpool.push_back(W.h_slot[e]);
                        W.dyn_e.push_back(e);
                        W.inpool[e] = 1;
                        W.score[e] = 3.f;   // review r4 M3: the converter's hot set is not churned at the start
                        np[gs_card(W.h_slot[e])]++;
                    }
                W.pend_e.assign(W.dpool.size(), -1);
                W.pend_job.assign(W.dpool.size(), -1);
                W.pend_t.assign(W.dpool.size(), 0);
                W.pend_m.assign(W.dpool.size(), -1);
                W.pend_h.assign(W.dpool.size(), nullptr);
                W.pend_ev.resize(W.dpool.size());
                for (size_t i = 0; i < W.dpool.size(); i++) {   // on the slot's card (its copy stream records it)
                    DevGuard dg(E->cards[gs_card(W.dpool[i])].dev);
                    DCK(cudaEventCreateWithFlags(&W.pend_ev[i], cudaEventDisableTiming));
                }
            }
            for (int k = 0; k < NC; k++)
                logf_("card %d: dynamic FP4 tier: %d of the %d slots reassignable\n", k, np[k], E->cards[k].used_slots);
        }
        logf_("hot experts loaded in %.1f s\n", ms_since(t1) / 1e3);
    }
    // ---- host tiers: Engram tables first (pageable), then the pinned experts
    {
        auto t1 = clk::now();
        load_hash(E);
        const Json& ej = pk["engram"];
        E->eg_nvme = ej.has("source") && ej["source"].str == "checkpoint";
        if (E->eg_nvme && E->gm) {   // [dsv4-rel-q6k]: the GGUF's Q6_K tables (gguf_model.h opened them)
            for (int li = 0; li < 2; li++) {
                E->egr.fdw[li] = E->gm->eg_fd[li];
                E->egr.offw[li] = E->gm->eg_off[li];
                E->eg[li].nrows = E->gm->eg_rows[li];
            }
            E->egr.q6k = E->eg_q6k = true;
            eg_mirror_open(E);
            E->egr.start(eg_threads());
            DCK(cudaHostAlloc(&E->h_egraw, (size_t)o.chunk * 48 * 264, cudaHostAllocPortable));
            if (E->egr.uring) logf_("engram: Q6_K rows read from the GGUF with O_DIRECT (io_uring: %d threads x %d in flight)\n", E->egr.nur, eg_uqd());
            else logf_("engram: Q6_K rows read from the GGUF with O_DIRECT (%d I/O threads)\n", eg_threads());
        } else if (E->eg_nvme) {
            for (int li = 0; li < 2; li++) {
                int l = li == 0 ? 1 : 14;
                for (int k2 = 0; k2 < 2; k2++) {
                    std::string nm = "layers." + std::to_string(l) + ".engram.embed." + (k2 ? "scale" : "weight");
                    size_t off = 0;
                    std::string path = st.file_of(st.get(nm).data, &off);
                    int fd = open(path.c_str(), O_RDONLY | O_DIRECT);
                    if (fd < 0) {   // review r7 M6: no O_DIRECT here (tmpfs, FUSE): buffered reads (page cache)
                        logf_("warning: O_DIRECT open %s: %s; buffered Engram reads\n", path.c_str(), strerror(errno));
                        fd = open(path.c_str(), O_RDONLY);
                    }
                    DCHECK(fd >= 0, "open %s: %s (the checkpoint is mapped, so this should not happen)", path.c_str(), strerror(errno));
                    (k2 ? E->egr.fds[li] : E->egr.fdw[li]) = fd;
                    (k2 ? E->egr.offs[li] : E->egr.offw[li]) = off;
                }
                E->eg[li].nrows = (size_t)ej["rows"][li].i();
            }
            E->egr.start(eg_threads());
            DCK(cudaHostAlloc(&E->h_egraw, (size_t)o.chunk * 48 * 264, cudaHostAllocPortable));
            if (E->egr.uring)   // (review pf10 L6)
                logf_("engram: lossless FP8 rows read from the checkpoint with O_DIRECT (io_uring: %d threads x %d in flight)\n", E->egr.nur, eg_uqd());
            else logf_("engram: lossless FP8 rows read from the checkpoint with O_DIRECT (%d I/O threads)\n", eg_threads());
        }
        size_t etot = 0;
        for (int li = 0; li < 2 && !E->eg_nvme; li++) {
            EgTable& T = E->eg[li];
            T.bits = (int)ej["bits"].i();
            T.rowbytes = 8 + 32 * T.bits;
            T.nrows = (size_t)ej["rows"][li].i();
            size_t b = T.nrows * T.rowbytes;
            T.base = (uint8_t*)mmap(nullptr, b, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
            DCHECK(T.base != MAP_FAILED, "engram alloc %zu", b);
            // no MADV_HUGEPAGE: with most of the RAM pinned, THP compaction made this read 6x slower (965 s)
            read_into(o.packed + "/engram_" + std::to_string(li) + ".bin", T.base, b);
            etot += b;
            T.hot_bits = (int)ej["hot_bits"].i();
            if (T.hot_bits) {
                T.hot_rowbytes = 8 + 32 * T.hot_bits;
                std::string ids = rd_file(o.packed + "/engram_hot_" + std::to_string(li) + ".ids");
                T.hot_ids.resize(ids.size() / 8);
                memcpy(T.hot_ids.data(), ids.data(), ids.size());
                size_t hb = T.hot_ids.size() * T.hot_rowbytes;
                T.hot = (uint8_t*)malloc(hb);
                read_into(o.packed + "/engram_hot_" + std::to_string(li) + ".bin", T.hot, hb);
                etot += hb + ids.size();
            }
        }
        logf_("engram tables: %.2f GiB in %.1f s\n", etot / 1073741824.0, ms_since(t1) / 1e3);
        t1 = clk::now();
        size_t tot = 0;
        // RAM plan (review r4 M5): needed = host experts not resident + checkpoint copies (demoted / degraded);
        // optional = the pack copies of resident pool experts. Over the budget: the best-ranked pool residents lose
        // their copy and stay static.
        auto isck = [](const Layer& W, int e) { return !W.ck.empty() && W.ck[e]; };
        // review r16 M1: in an M23 pack the experts the pool will start with (top ranks per layer) get their own host
        // copies, which the RAM plan may leave out up front (the expert then stays static in its slot) and the memory guard
        // can drop later; the other host experts live in per-layer arenas
        const bool m23pool = E->pfmt == XF_M23 && !E->dyn_fp4 && o.dyn;
        if (m23pool)
            for (int l = 0; l < NL; l++) {
                Layer& W = E->L[l];
                W.inpool.assign(NE, 0);
                W.own.assign(NE, 0);
                std::vector<int> ord;
                for (int e = 0; e < NE; e++)
                    if (W.hfmt[e] >= 0 && W.h_slot[e] < 0 && W.h_dslot[e] < 0) ord.push_back(e);
                std::sort(ord.begin(), ord.end(), [&](int a, int b) { return W.rank[a] < W.rank[b]; });
                for (size_t i = 0; i < ord.size() && i < W.dpool.size(); i++) W.inpool[ord[i]] = 1;
            }
        auto pooled = [&](const Layer& W, int e) { return (E->dyn_fp4 || m23pool) && !W.inpool.empty() && W.inpool[e]; };
        {
            // review r6 M5: the budget was taken before the Engram tables were read into RAM, and the admission
            // bounce buffers are pinned beside the experts: both come off it here
            const double fixed = (double)etot + (E->dyn_fp4 ? (double)NC * XJ_PER_CARD * XJ_BYTES : 0.0) +   // r7 L11
                                 (E->m23conv ? m23_conv_bytes() : 0.0);   // review r16 L6
            double budget = E->hbudget - fixed;
            double need = 0, opt = 0;
            std::vector<std::pair<int, int>> optl;   // (rank, l * NE + e)
            for (int l = 0; l < NL; l++) {
                Layer& W = E->L[l];
                for (int e = 0; e < NE; e++) {
                    if (W.h_dslot[e] >= 0) continue;   // R8: cb3 band resident, no host copy needed
                    if (m23pool && pooled(W, e)) {   // review r16 M1
                        opt += expert_bytes((XFmt)W.hfmt[e]);
                        optl.push_back({W.rank[e], l * NE + e});
                        continue;
                    }
                    if (isck(W, e) || (W.hfmt[e] < 0 && W.h_slot[e] < 0)) need += XB;
                    else if (W.hfmt[e] >= 0 && W.h_slot[e] < 0) need += expert_bytes((XFmt)W.hfmt[e]);
                    else if (W.hfmt[e] >= 0 && pooled(W, e)) {
                        opt += expert_bytes((XFmt)W.hfmt[e]);
                        optl.push_back({W.rank[e], l * NE + e});
                    }
                }
            }
            int ndrop = 0;
            E->nocopy.assign((size_t)NL * NE, 0);
            if (E->gm && need + opt > budget) {   // [dsv4-rel-ram]: GGUF: experts without a host copy run from the file
                // first the hottest part of the pool's seed set (top ranks per layer: they start in VRAM and stay
                // there, static), then the rarest by rank; an expert without a copy is read from the GGUF mapping by
                // the lane and re-packed for admissions. [dsv4-rel-rampin] ❌ (DSV4_RAMPIN < 1: only that top fraction
                // of each layer's seeds goes first): at 0.5 the hit came back (0.74 / 0.77 / 0.72) but ~3,600 experts
                // ran from the mapping in 4 KB faults (33-50K major faults per pass, lane 7-13 GB/s: 18 / 19 / 8 tok/s
                // vs 84 / 66 / 43 at 1). Default 1 (every seed pinned) until the cold tail leaves the fault path
                static const double pinf = std::clamp(getenv("DSV4_RAMPIN") ? atof(getenv("DSV4_RAMPIN")) : 1.0, 0.0, 1.0);   // ram3 ❌ at 0.5: default 1 (ram2)
                std::vector<std::tuple<int, int, int>> cand;   // (class 0 pinned seed / 1 other, -rank, l * NE + e)
                for (int l = 0; l < NL; l++) {
                    Layer& W = E->L[l];
                    std::vector<int> ord;
                    for (int e = 0; e < NE; e++)
                        if (W.hfmt[e] >= 0 && W.h_slot[e] < 0 && W.h_dslot[e] < 0) ord.push_back(e);
                    std::sort(ord.begin(), ord.end(), [&](int x, int y) { return W.rank[x] < W.rank[y]; });
                    const size_t npin = (size_t)(pinf * (double)W.dpool.size() + 0.5);
                    for (size_t i = 0; i < ord.size(); i++)
                        cand.push_back({i < npin ? 0 : 1, i < npin ? 0 : -W.rank[ord[i]], l * NE + ord[i]});
                }
                std::sort(cand.begin(), cand.end());
                // review rel2 M1: the exclusive swap's home ring is reserved before the copies are dropped (it was added
                // after, so the plan ended over budget)
                const bool gsw = E->o.swap && E->psplit && o.dyn && getenv("DSV4_GSWAP") && atoi(getenv("DSV4_GSWAP")) != 0;
                const int NH = std::max(8, atoi(getenv("DSV4_GSWAP_HOMES") ? getenv("DSV4_GSWAP_HOMES") : "48"));
                std::map<size_t, int> nsz;
                for (int l = 0; l < NL; l++) nsz[expert_bytes(pool_fmt(E, l))]++;
                if (gsw)
                    for (auto& [sz, nl] : nsz) need += (double)NH * (double)sz;
                double nat = 0;
                int nseed = 0;
                for (auto& t : cand) {
                    if (need + opt <= budget) break;
                    const int le = std::get<2>(t);
                    const double xb = (double)expert_bytes((XFmt)E->L[le / NE].hfmt[le % NE]);
                    E->nocopy[le] = 1;
                    need -= xb;
                    nat += xb;
                    nseed += std::get<0>(t) == 0;
                    ndrop++;
                }
                // [dsv4-rel-xswap]: exclusive swap: the seeds without a copy stay dynamic (an evicted one goes D2H into a
                // free home), so the RAM saved is the resident set's. 🟡 xs1 (122 GiB hog): 79.61 / 67.55 / 42.45 vs ram2
                // 83.69 / 65.92 / 43.28 tok/s: narrative hit 0.45 -> 0.61 but the lane slower per byte (69 vs 93 GB/s:
                // the rare copy-less tail on the fault path, swap D2H); the home ring ran dry (1,846 swaps, 1,750 back).
                // Opt-in DSV4_GSWAP=1 until the cold tail leaves the fault path
                E->gswap = gsw;
                logf_("host RAM: %d experts (%.2f GiB) without a host copy (%d of them start in VRAM%s): the "
                      "others are read from the GGUF files (page cache) when routed to the CPU lane, re-packed for admissions\n",
                      ndrop, nat / 1073741824.0, nseed, E->gswap ? ", dynamic: exclusive swap" : " and stay there");
                if (E->gswap) {   // a ring of free homes per expert size; each swap takes one, each admitted copy gives one
                    double hb = 0;
                    for (auto& [sz, nl] : nsz)
                        for (int i = 0; i < NH; i++) {
                            int npg = 0;
                            uint8_t* b = host_alloc(sz, &npg);
                            if (npg) { host_free(b); break; }
                            E->ghome[sz].push_back(b);
                            hb += (double)sz;
                        }
                    logf_("exclusive swap: %d free homes per expert size (%zu sizes, %.2f GiB)\n", NH, nsz.size(), hb / 1073741824.0);
                }
            }
            if (need + opt > budget) {
                std::sort(optl.begin(), optl.end());
                int nnt = 0;   // review r13 L1: only non-tail copies can become homes
                for (auto& pr : optl) nnt += !in_tail(E, E->L[pr.second / NE].h_slot[pr.second % NE]);
                // review r20 M1: a static M23 / GGUF pool expert must get a slot outside the borrowable tail (the
                // seeding puts static experts there first): per layer at most as many drops as such slots
                std::vector<int> nsa(NL, 0), ndl(NL, 0);
                if (m23pool)
                    for (int l = 0; l < NL; l++)
                        for (int g : E->L[l].dpool) nsa[l] += !in_gtail(E, g);
                for (auto& pr : optl) {
                    if (need + opt <= budget) break;
                    Layer& W = E->L[pr.second / NE];
                    int e = pr.second % NE;
                    if (in_tail(E, W.h_slot[e])) continue;   // review r9 H1
                    if (E->swap && nnt - ndrop <= 64) break;   // R13: keep >= 64 non-tail copies as the first homes
                    if (m23pool) {   // review r16 M1: seeded from the checkpoint, static in its slot, no host copy
                        if (ndl[pr.second / NE] >= nsa[pr.second / NE]) continue;   // review r20 M1
                        ndl[pr.second / NE]++;
                        E->nocopy[pr.second] = 1;
                        opt -= expert_bytes((XFmt)W.hfmt[e]);
                        ndrop++;
                        continue;
                    }
                    if (E->swap) {   // R13: stays in the pool without a host copy (the swap tier evicts it via D2H)
                        E->nocopy[pr.second] = 1;
                        opt -= expert_bytes((XFmt)W.hfmt[e]);
                        ndrop++;
                        continue;
                    }
                    for (size_t j = 0; j < W.dyn_e.size(); j++)
                        if (W.dyn_e[j] == e) {   // nothing pending at load time: drop the pool entry
                            W.dpool.erase(W.dpool.begin() + j);
                            W.dyn_e.erase(W.dyn_e.begin() + j);
                            W.pend_e.erase(W.pend_e.begin() + j);
                            W.pend_job.erase(W.pend_job.begin() + j);
                            W.pend_t.erase(W.pend_t.begin() + j);
                            if (!W.pend_m.empty()) { W.pend_m.erase(W.pend_m.begin() + j); W.pend_h.erase(W.pend_h.begin() + j); }
                            DCK(cudaEventDestroy(W.pend_ev[j]));
                            W.pend_ev.erase(W.pend_ev.begin() + j);
                            break;
                        }
                    W.inpool[e] = 0;
                    opt -= expert_bytes((XFmt)W.hfmt[e]);
                    ndrop++;
                }
            }
            logf_("host RAM plan: needed %.2f GiB + pool copies %.2f GiB, budget %.2f GiB (after %.2f GiB Engram + bounce)"
                  " (%d pool slots made static)\n", need / 1073741824.0, opt / 1073741824.0, budget / 1073741824.0,
                  fixed / 1073741824.0, ndrop);
            if (need + opt > budget)   // review r6 M6: say how much is missing; the memory guard frees pool copies later
                logf_("warning: host RAM plan is %.2f GiB OVER the budget (pageable fallback; the memory guard drops "
                      "resident pool copies if MemAvailable falls below 8 GiB). Use a smaller pack or free RAM.\n",
                      (need + opt - budget) / 1073741824.0);
            // R13 (OOM incident R13-7: an over-budget DSpark load ran into the kernel OOM killer, which also took the
            // desktop): what is still to be allocated must fit in MemAvailable - DSV4_HOSTGUARD_GIB (default 4) now,
            // else stop the load with the reason instead of allocating pageable memory past it
            double done = 0;   // checkpoint copies already allocated at the slot phase (counted in need)
            for (int l = 0; l < NL; l++)
                for (int e = 0; e < NE; e++)
                    if (isck(E->L[l], e) && E->L[l].hx[e]) done += XB;
            static const double hg = getenv("DSV4_HOSTGUARD_GIB") ? atof(getenv("DSV4_HOSTGUARD_GIB")) : 4.0;
            // DSV4_RAM_RESERVE_GIB: test hook, pretends that much less is available (checks the refusal paths)
            const double rsv = !o.ram_reserve_off && getenv("DSV4_RAM_RESERVE_GIB") ? atof(getenv("DSV4_RAM_RESERVE_GIB")) : 0.0;
            const double G = 1073741824.0, ma = mem_avail_gib();
            // review r13 L2: the up to 32 own swap homes allocated after this check count too
            // R16 [dsv4-gguf-iq] review L4: the GGUF loader's per-thread read buffers (freed after the load) count too
            const double av = ma * G - rsv * G,
                         left = need + opt - done + (E->swap ? 32.0 * XB : 0.0) + (E->gg ? (double)gg_load_scratch(E) : 0.0);
            const bool known = ma >= 0;   // review r13 L3: a reserve above MemAvailable is "negative", not "unknown"
            // an opt-in feature that leaves < 8 GiB free is refused (one line, the load restarts without it), before
            // the bulk of the host experts is allocated
            if (known && o.dspark && left > av - 8.0 * G && o.reload_without_dspark) {
                logf_("DSpark refused: with its drafter the load still needs %.2f GiB of host RAM, MemAvailable is %.2f GiB "
                      "(< 8 GiB would be left): loading again without DSpark\n", left / G, av / G);
                o.reload_without_dspark();
            }
            if (known && left > av - 8.0 * G && left <= av - hg * G)
                logf_("WARNING host RAM: %.2f GiB to load, MemAvailable %.2f GiB: only %.2f GiB will be left (< 8)\n",
                      left / G, av / G, (av - left) / G);
            if (known && left > av - hg * G) {
                fprintf(stderr, "error: host RAM: the load still needs %.2f GiB but MemAvailable is %.2f GiB (guard %.1f GiB)%s. "
                        "Free host RAM or use a smaller pack; stopping before the OOM killer would.\n",
                        left / 1073741824.0, av / 1073741824.0, hg,
                        o.dspark ? " - DSpark's drafter moved experts to host RAM: run without --dspark" : "");
                exit(3);
            }
        }
        // R24 [dsv4-r24-load]: GGUF host copies: the read of a layer overlaps the next layer's pinning (one read in
        // flight; the same bytes to the same places). DSV4_LDSYNC=1: one after the other (as before)
        const bool ld_async = E->gg && !E->m23conv && !(getenv("DSV4_LDSYNC") && atoi(getenv("DSV4_LDSYNC")) != 0);
        std::thread ld_th;
        for (int l = 0; l < NL; l++) {
            Layer& W = E->L[l];
            size_t b = 0;
            // only the experts that are not resident in VRAM (file offsets: all host experts in id order)
            std::vector<std::pair<size_t, size_t>> runs;   // (file offset, bytes) of the needed experts
            size_t foff = 0;
            std::vector<std::pair<int, uint8_t*>> own_todo;   // R16 M23: own copies converted with the layer
            for (int e = 0; e < NE; e++)
                if (W.hfmt[e] >= 0 && !isck(W, e)) {
                    size_t xb = expert_bytes((XFmt)W.hfmt[e]);
                    if (m23pool && pooled(W, e)) {   // review r16 M1: own allocation, or none (static, seeded from the checkpoint)
                        if (!E->nocopy[(size_t)l * NE + e]) {
                            uint8_t* d1 = host_alloc(xb, &E->n_pageable);
                            if (E->m23conv) own_todo.push_back({e, d1});
                            else read_runs(o.packed + "/experts_l" + std::to_string(l) + ".bin", d1, {{foff, xb}});
                            W.hx[e] = d1;
                            W.own[e] = 1;
                            tot += xb;
                        }
                    } else if (W.h_slot[e] < 0 && W.h_dslot[e] < 0 && !(E->gm && E->nocopy[(size_t)l * NE + e])) {
                        b += xb;
                        runs.push_back({foff, xb});
                    } else if (pooled(W, e) && E->nocopy[(size_t)l * NE + e]) {   // R13: no copy (swap tier)
                        W.hfmt[e] = -1;   // (foff below still advances by this expert's bytes: xb taken above)
                    } else if (pooled(W, e)) {   // resident pool expert: own allocation (the memory guard can drop it)
                        uint8_t* d1 = host_alloc(xb, &E->n_pageable);
                        read_runs(o.packed + "/experts_l" + std::to_string(l) + ".bin", d1, {{foff, xb}});
                        W.hx[e] = d1;
                        if (W.own.empty()) W.own.assign(NE, 0);
                        W.own[e] = 1;
                        tot += xb;
                    }
                    foff += xb;
                }
            if (!b) {
                if (!own_todo.empty()) m23_convert_list(E, l, own_todo);
                continue;
            }
            const auto ta0 = clk::now();
            uint8_t* ar = host_alloc(b, &E->n_pageable);
            g_ld_alloc_ms += ms_since(ta0);   // [dsv4-rel-load128]: pinning (cudaHostAlloc) of the layer's arena
            const auto tc0 = clk::now();
            auto ggwant = [](Engine* E2, int l2, int e) {
                const Layer& W2 = E2->L[l2];
                return W2.hfmt[e] >= 0 && !(!W2.ck.empty() && W2.ck[e]) && W2.h_slot[e] < 0 && W2.h_dslot[e] < 0 &&
                       !(E2->gm && E2->nocopy[(size_t)l2 * NE + e]);
            };
            if (E->gg && ld_async) {   // R24 [dsv4-r24-load]: layer l's read runs while layer l + 1's arena is pinned
                if (ld_th.joinable()) ld_th.join();   // (the wait counts as read time below)
                ld_th = std::thread([E, l, ar, ggwant] {
                    const auto tq = clk::now();
                    gg_copy_layer(E, l, ar, ggwant);
                    g_ld_copy_bg_ms += ms_since(tq);
                });
            } else if (E->gg)   // R16 [dsv4-gguf-iq]: from the GGUF (same experts as the read below)
                gg_copy_layer(E, l, ar, ggwant);
            else if (!E->m23conv) read_runs(o.packed + "/experts_l" + std::to_string(l) + ".bin", ar, runs);
            g_ld_copy_ms += ms_since(tc0);
            E->host_arenas.push_back(ar);
            size_t off = 0;
            for (int e = 0; e < NE; e++)
                if (W.hfmt[e] >= 0 && !isck(W, e) && W.h_slot[e] < 0 && W.h_dslot[e] < 0 && !(m23pool && pooled(W, e)) &&
                    !(E->gm && E->nocopy[(size_t)l * NE + e])) {
                    W.hx[e] = ar + off;
                    if (E->m23conv) own_todo.push_back({e, ar + off});   // R16: from the checkpoint, no pack file
                    off += expert_bytes((XFmt)W.hfmt[e]);
                }
            if (E->m23conv) m23_convert_list(E, l, own_todo);
            tot += b;
        }
        if (ld_th.joinable()) {   // R24: the last layer's read (the seeding below copies from the host copies)
            const auto tj = clk::now();
            ld_th.join();
            g_ld_copy_ms += ms_since(tj);
        }
        if (E->gg && !E->dyn_fp4 && o.dyn) {   // R16 [dsv4-gguf-iq]: fill the GGUF pool by usage rank (every expert has a host copy)
            int nseed = 0;
            uint8_t* stg = nullptr;   // [dsv4-rel-load128]: pinned staging of the copy-less seeds (freed after seeding)
            // R24 [dsv4-r24-seed2]: two stagings in turn: chunk k is read while chunk k - 1's copies run; a staging is
            // reused only after the events recorded behind its copies (one per card)
            uint8_t* stg2 = nullptr;
            std::vector<cudaEvent_t> sev(2 * E->cards.size(), nullptr);
            bool sev_used[2] = {false, false};
            int sbuf = 0;
            double seed_rd_ms = 0;   // R24: time in the seed reads (the rest: slot copies, waits, tables)
            size_t stgb = 0;
            const long long sb0 = g_ld_bytes.load();   // review rel2 L3: the seeding's reads, logged apart
            const auto tsd = clk::now();
            for (int l = 0; l < NL; l++) {
                Layer& W = E->L[l];
                if (W.dpool.empty()) continue;
                Card& c = E->cards[W.card];
                std::vector<int> ord;
                for (int e = 0; e < NE; e++)   // [dsv4-rel-ram]: GGUF experts without a copy seed from the file
                    if ((W.hx[e] || (E->gm && W.hfmt[e] >= 0)) && W.h_slot[e] < 0 && W.h_dslot[e] < 0) ord.push_back(e);
                std::sort(ord.begin(), ord.end(), [&](int a, int b) { return W.rank[a] < W.rank[b]; });
                // review rel L5 (as the M23 seeding, review r19 H1): experts without a host copy (pinned) take slots
                // outside the borrowable tail, the others the tail slots first; no pinned expert: slot order as before
                std::vector<size_t> sa, sb;
                for (size_t q = 0; q < W.dpool.size(); q++) (in_gtail(E, W.dpool[q]) ? sb : sa).push_back(q);
                bool anyst = false;
                for (size_t i = 0; i < ord.size() && i < W.dpool.size(); i++) anyst |= !W.hx[ord[i]];
                size_t ia = 0, ib = 0;
                std::vector<std::pair<int, size_t>> ncs;   // [dsv4-rel-load128]: copy-less seeds (e, dpool index)
                for (size_t i = 0; i < ord.size() && i < W.dpool.size(); i++) {
                    const int e = ord[i];
                    size_t j;
                    if (sb.empty() || !anyst) j = i;
                    else if (!W.hx[e]) j = ia < sa.size() ? sa[ia++] : sb[ib++];
                    else j = ib < sb.size() ? sb[ib++] : sa[ia++];
                    Card& sc = pool_card(E, c, W.dpool[j]);
                    DevGuard dg(sc.dev);
                    if (W.hx[e]) DCK(cudaMemcpy(pool_slot_ptr(E, c, l, W.dpool[j]), W.hx[e], expert_bytes(pool_fmt(E, l)), cudaMemcpyHostToDevice));
                    else {   // [dsv4-rel-ram]
                        if (W.pin.empty()) W.pin.assign(NE, 0);
                        W.pin[e] = !E->gswap;   // [dsv4-rel-xswap]: with the exclusive swap no seed is static
                        ncs.push_back({e, j});   // re-packed below, all of the layer at once (threads, pread)
                    }
                    W.dyn_e[j] = e;
                    W.h_dslot[e] = W.dpool[j];
                    W.score[e] = g_ggsc ? 1.f : 3.f;   // R20 [dsv4-r20-ggsc]: seed 1 (cache replay) when on
                    nseed++;
                }
                if (!ncs.empty()) {   // [dsv4-rel-load128]: ld1 spent ~350 s re-packing these one by one through the
                    // mapping (4 KB faults under the hog): now the layer's copy-less seeds are read with pread by the
                    // loader's threads into one pinned staging (gg_copy_layer, expert id order), then copied to their slots
                    // review rel2 M2: chunks of at most 32 experts (<= ~0.4 GB, inside the plan's margin), pinned when the
                    // host allows it, else pageable (host_alloc never aborts first; the copies are then synchronous)
                    const size_t XB1 = expert_bytes(pool_fmt(E, l));
                    const size_t CH = 32;
                    if (stgb < CH * XB1) {
                        for (int b2 = 0; b2 < 2; b2++)   // (the old stagings' copies must be done before the free)
                            if (sev_used[b2])
                                for (size_t k = 0; k < E->cards.size(); k++) DCK(cudaEventSynchronize(sev[b2 * E->cards.size() + k]));
                        sev_used[0] = sev_used[1] = false;
                        sbuf = 0;   // re-review H1: a new pair starts at staging 0 (stg2 may be null: pageable fallback)
                        if (stg) host_free(stg);
                        if (stg2) host_free(stg2);
                        stgb = CH * XB1;
                        int npg = 0;
                        stg = host_alloc(stgb, &npg);
                        stg2 = npg ? nullptr : host_alloc(stgb, &npg);   // pageable: one staging, synchronous as before
                        if (stg2 && npg) { host_free(stg2); stg2 = nullptr; }
                    }
                    static std::vector<uint8_t> want;
                    for (size_t c0 = 0; c0 < ncs.size(); c0 += CH) {
                        const size_t c1 = std::min(ncs.size(), c0 + CH);
                        want.assign(NE, 0);
                        for (size_t q = c0; q < c1; q++) want[ncs[q].first] = 1;
                        if (stg2) sbuf ^= 1;
                        uint8_t* sbp = sbuf && stg2 ? stg2 : stg;
                        if (sev_used[sbuf])
                            for (size_t k = 0; k < E->cards.size(); k++) DCK(cudaEventSynchronize(sev[sbuf * E->cards.size() + k]));
                        const auto tq0 = clk::now();
                        gg_copy_layer(E, l, sbp, [](Engine*, int, int e) { return want[e] != 0; });
                        seed_rd_ms += ms_since(tq0);
                        std::vector<size_t> off(NE, 0);
                        size_t o2 = 0;
                        for (int e = 0; e < NE; e++)
                            if (want[e]) { off[e] = o2; o2 += XB1; }
                        for (size_t q = c0; q < c1; q++) {
                            const auto [e, j] = ncs[q];
                            Card& sc = pool_card(E, c, W.dpool[j]);
                            DevGuard dg(sc.dev);
                            DCK(cudaMemcpyAsync(pool_slot_ptr(E, c, l, W.dpool[j]), sbp + off[e], XB1, cudaMemcpyHostToDevice, sc.cp));
                        }
                        if (!stg2)
                            for (auto& k : E->cards) {
                                DevGuard dg(k.dev);
                                DCK(cudaStreamSynchronize(k.cp));   // the staging is reused by the next chunk
                            }
                        else {
                            for (size_t k = 0; k < E->cards.size(); k++) {
                                DevGuard dg(E->cards[k].dev);
                                cudaEvent_t& ev = sev[sbuf * E->cards.size() + k];
                                if (!ev) DCK(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming));
                                DCK(cudaEventRecord(ev, E->cards[k].cp));
                            }
                            sev_used[sbuf] = true;
                        }
                    }
                }
                {
                    DevGuard dg(c.dev);
                    DCK(cudaMemcpy(W.d_dslot, W.h_dslot.data(), NE * 4, cudaMemcpyHostToDevice));
                }
                if (E->psplit)
                    for (int k = 0; k < NC; k++) {
                        std::vector<int> t(NE, -1);
                        for (int e = 0; e < NE; e++)
                            if (W.h_dslot[e] >= 0 && gs_card(W.h_dslot[e]) == k) t[e] = pool_lidx(E, l, W.h_dslot[e]);
                        DevGuard dg(E->cards[k].dev);
                        DCK(cudaMemcpy(W.d_dls[k], t.data(), NE * 4, cudaMemcpyHostToDevice));
                    }
            }
            for (auto& k : E->cards) {   // R24: the last chunks' copies before the stagings go
                DevGuard dg(k.dev);
                DCK(cudaStreamSynchronize(k.cp));
            }
            for (size_t i = 0; i < sev.size(); i++)
                if (sev[i]) {
                    DevGuard dg(E->cards[i % E->cards.size()].dev);
                    cudaEventDestroy(sev[i]);
                }
            if (stg) host_free(stg);
            if (stg2) host_free(stg2);
            g_ld_seed_bytes = g_ld_bytes.load() - sb0;
            // re-review L2: one whole-file DONTNEED at the end of the seeding (btrfs compressed extents go through the
            // page cache even with O_DIRECT; also drops the non-expert read-ahead pages as the per-call DONTNEED did)
            for (auto& F : E->gg->m.files()) posix_fadvise(F.fd, 0, 0, POSIX_FADV_DONTNEED);
            if (g_ld_seed_bytes)
                logf_("copy-less seeds re-packed from the GGUF: %.2f GB in %.1f s (reads + re-pack %.1f s)\n", g_ld_seed_bytes / 1e9,
                      ms_since(tsd) / 1e3, seed_rd_ms / 1e3);
            logf_("GGUF pool seeded with %d experts by usage rank%s\n", nseed, E->psplit ? " (split over the cards)" : "");
        }
        if (m23pool) {   // R16: fill the M23 pool with its seed set (top ranks per layer, R16 M1: chosen before the RAM plan)
            const size_t PB = expert_bytes(XF_M23);
            int nseed = 0, nstat = 0, nstat_tail = 0;
            uint8_t* tmp = nullptr;   // an expert without a host copy: converted here, then copied to its slot
            for (int l = 0; l < NL; l++) {
                Layer& W = E->L[l];
                if (W.dpool.empty()) continue;
                Card& c = E->cards[W.card];
                std::vector<int> ord;
                for (int e = 0; e < NE; e++)
                    if (W.inpool[e]) ord.push_back(e);
                std::sort(ord.begin(), ord.end(), [&](int a, int b) { return W.rank[a] < W.rank[b]; });
                std::vector<size_t> stat;   // dpool indices of experts without a host copy: static after seeding
                // review r19 H1: a static expert takes a slot outside the borrowable tail (in_gtail), experts with a
                // host copy take the tail slots first; with no borrowing: slot order as before
                std::vector<size_t> sa, sb;   // slots outside / inside the tail
                for (size_t q = 0; q < W.dpool.size(); q++) (in_gtail(E, W.dpool[q]) ? sb : sa).push_back(q);
                size_t ia = 0, ib = 0;
                bool anyst = false;   // (only a layer with static experts changes its slot order)
                for (size_t i = 0; i < ord.size() && i < W.dpool.size(); i++) anyst |= !W.hx[ord[i]];
                for (size_t i = 0; i < ord.size() && i < W.dpool.size(); i++) {
                    const int e = ord[i];
                    const uint8_t* src = W.hx[e];
                    const bool st0 = !src;
                    size_t j;
                    if (sb.empty() || !anyst) j = i;
                    else if (st0) j = ia < sa.size() ? sa[ia++] : sb[ib++];
                    else j = ib < sb.size() ? sb[ib++] : sa[ia++];
                    if (st0 && in_gtail(E, W.dpool[j])) nstat_tail++;
                    if (!src) {
                        if (!tmp) tmp = host_alloc(PB, &E->n_pageable);
                        m23_convert_list(E, l, {{e, tmp}});
                        src = tmp;
                        stat.push_back(j);
                    }
                    Card& sc = pool_card(E, c, W.dpool[j]);
                    DevGuard dg(sc.dev);
                    DCK(cudaMemcpy(pool_slot_ptr(E, c, l, W.dpool[j]), src, PB, cudaMemcpyHostToDevice));
                    W.dyn_e[j] = e;
                    W.h_dslot[e] = W.dpool[j];
                    W.score[e] = 3.f;   // as the FP4 plan's seed (review r4 M3)
                    nseed++;
                }
                std::sort(stat.begin(), stat.end());   // (review r19 H1: slots no longer in seeding order)
                for (size_t k = stat.size(); k-- > 0;) {   // no host copy: the expert keeps its slot for good
                    const size_t q = stat[k];
                    W.dpool.erase(W.dpool.begin() + q);
                    W.dyn_e.erase(W.dyn_e.begin() + q);
                    W.pend_e.erase(W.pend_e.begin() + q);
                    W.pend_job.erase(W.pend_job.begin() + q);
                    W.pend_t.erase(W.pend_t.begin() + q);
                    DCK(cudaEventDestroy(W.pend_ev[q]));
                    W.pend_ev.erase(W.pend_ev.begin() + q);
                    nstat++;
                }
                {
                    DevGuard dg(c.dev);
                    DCK(cudaMemcpy(W.d_dslot, W.h_dslot.data(), NE * 4, cudaMemcpyHostToDevice));
                }
                if (E->psplit)
                    for (int k = 0; k < NC; k++) {   // local tables
                        std::vector<int> t(NE, -1);
                        for (int e = 0; e < NE; e++)
                            if (W.h_dslot[e] >= 0 && gs_card(W.h_dslot[e]) == k) t[e] = pool_lidx(E, l, W.h_dslot[e]);
                        DevGuard dg(E->cards[k].dev);
                        DCK(cudaMemcpy(W.d_dls[k], t.data(), NE * 4, cudaMemcpyHostToDevice));
                    }
            }
            if (tmp) host_free(tmp);
            logf_("M23 pool seeded with %d experts by usage rank%s; %d of them without a host copy (host RAM plan): static\n",
                  nseed, E->psplit ? " (split over the cards)" : "", nstat);
            if (nstat_tail)   // review r19 H1: more static experts than slots outside the borrowable tail in some layer
                logf_("WARNING %d static pool experts in the borrowable tail: long prefills run in rows of %d while they stay\n",
                      nstat_tail, MAXR);
        }
        // must-resident experts that found no VRAM slot: FP4 bytes from the checkpoint into host RAM (degrade)
        int nck = 0;
        for (int l = 0; l < NL; l++)
            for (int e = 0; e < NE; e++) {
                Layer& W = E->L[l];
                if (W.h_slot[e] >= 0 || W.h_dslot[e] >= 0 || W.hfmt[e] >= 0) continue;
                const size_t XB = expert_bytes(XF_FP4);
                uint8_t* dst = host_alloc(XB, &E->n_pageable);
                std::string px = "layers." + std::to_string(l) + ".ffn.experts." + std::to_string(e) + ".";
                size_t off = 0;
                for (const char* wn : {"w1", "w3", "w2"}) {
                    const TensorInfo& q = st.get(px + wn + ".weight");
                    const TensorInfo& sc = st.get(px + wn + ".scale");
                    memcpy(dst + off, q.data, q.nbytes);
                    memcpy(dst + off + q.nbytes, sc.data, sc.nbytes);
                    off += q.nbytes + sc.nbytes;
                }
                W.hx[e] = dst;
                W.hfmt[e] = XF_FP4;
                tot += XB;
                nck++;
            }
        if (nck) logf_("WARNING: %d must-resident experts run from host RAM (FP4, CPU lane)\n", nck);
        if (E->swap) {   // R13: the FP4 copies of resident pool experts become the free homes (no RAM change)
            int nh = 0, npool = 0;
            for (int l = 0; l < NL; l++) {
                Layer& W = E->L[l];
                for (size_t j = 0; j < W.dpool.size(); j++) {
                    const int e = W.dyn_e[j];
                    if (e < 0) continue;
                    npool++;
                    if (!W.hx[e] || W.hfmt[e] != XF_FP4 || in_tail(E, W.dpool[j]) || !is_pinned(W.hx[e])) continue;
                    E->hfree.push_back({(uint8_t*)W.hx[e], !W.own.empty() && W.own[e]});
                    W.hx[e] = nullptr;
                    W.hfmt[e] = -1;
                    if (!W.own.empty()) W.own[e] = 0;
                    nh++;
                }
            }
            // a few homes of their own when none came free (the RAM plan kept no non-tail copy): each swap takes one,
            // each admitted FP4-host expert gives one back, so a small ring keeps the tier moving
            for (int i = nh; i < 32; i++) {
                int npg = 0;
                uint8_t* hb = host_alloc(XB, &npg);
                if (npg) { host_free(hb); break; }
                E->hfree.push_back({hb, true});
                nh++;
            }
            logf_("swap tier: %d pool slots (every slot outside the borrowable tail), %d free host homes (%.2f GiB)\n",
                  npool, nh, nh * (double)XB / 1073741824.0);
        }
        if (E->n_pageable) logf_("WARNING: %d host expert arenas are pageable (pinning failed: low RAM)\n", E->n_pageable);
        logf_("host experts: %.2f GiB in %.1f s\n", tot / 1073741824.0, ms_since(t1) / 1e3);
        if (E->gg)   // [dsv4-rel-load128]
            logf_("host experts load: arena pinning %.1f s, read + re-pack %.1f s wall (%d threads: pread %.1f s, re-pack %.1f s "
                  "summed over threads; %.2f GB read = %.2f GB/s wall)\n", g_ld_alloc_ms / 1e3, g_ld_copy_ms / 1e3,
                  gg_load_threads(), g_ld_read_ns / 1e9, g_ld_pack_ns / 1e9, (g_ld_bytes - g_ld_seed_bytes) / 1e9,
                  g_ld_copy_ms > 0 ? (g_ld_bytes - g_ld_seed_bytes) / 1e6 / g_ld_copy_ms : 0.0);
        if (E->gg && g_ld_copy_bg_ms > 0)   // R24 [dsv4-r24-load]
            logf_("host experts load overlapped: reads %.1f s in the background = %.2f GB/s (wall read waits %.1f s above; the "
                  "GB/s above divides by the waits only)\n", g_ld_copy_bg_ms / 1e3,
                  (g_ld_bytes - g_ld_seed_bytes) / 1e6 / g_ld_copy_bg_ms, g_ld_copy_ms / 1e3);
        DCK(cudaHostAlloc(&E->h_egrows, (size_t)1024 * EG_ROWS * 264, cudaHostAllocPortable));
        DCK(cudaHostAlloc(&E->h_egdz, (size_t)2 * MAXR * 24 * 264, cudaHostAllocPortable | cudaHostAllocMapped));
        DCK(cudaHostAlloc(&E->h_egoff, (size_t)1024 * EG_ROWS * 4, cudaHostAllocPortable));
        DCK(cudaHostAlloc(&E->h_egbits, (size_t)1024 * EG_ROWS, cudaHostAllocPortable));
        DCK(cudaHostAlloc(&E->h_rows, (size_t)o.chunk * TOPK * 4, cudaHostAllocPortable));
        DCK(cudaHostAlloc(&E->h_rw, (size_t)o.chunk * TOPK * 4, cudaHostAllocPortable));
        DCK(cudaHostAlloc(&E->h_pids, (size_t)o.chunk * TOPK * 4, cudaHostAllocPortable));
    }
    {
        for (int l = 0; l < NL; l++)
            for (int e = 0; e < NE; e++)
                DCHECK(E->L[l].h_slot[e] >= 0 || E->L[l].h_dslot[e] >= 0 || E->L[l].hx[e] || E->gm, "layer %d expert %d has neither a slot nor a host copy",
                       l, e);
        if (!o.placement_out.empty()) {
            FILE* f = fopen(o.placement_out.c_str(), "w");
            fprintf(f, "{\"split\": [%d, %d], \"resident\": [", E->cards[0].l1, E->cards[1].l1);
            for (int l = 0; l < NL; l++) {
                fprintf(f, "%s[", l ? ", " : "");
                bool first = true;
                for (int e = 0; e < NE; e++)
                    if (E->L[l].h_slot[e] >= 0) {
                        fprintf(f, "%s%d", first ? "" : ", ", e);
                        first = false;
                    }
                fprintf(f, "]");
            }
            fprintf(f, "], \"card\": [");   // R11: card of each resident expert (same order)
            for (int l = 0; l < NL; l++) {
                fprintf(f, "%s[", l ? ", " : "");
                bool first = true;
                for (int e = 0; e < NE; e++)
                    if (E->L[l].h_slot[e] >= 0) {
                        fprintf(f, "%s%d", first ? "" : ", ", gs_card(E->L[l].h_slot[e]));
                        first = false;
                    }
                fprintf(f, "]");
            }
            fprintf(f, "]}\n");
            fclose(f);
        }
    }
    {   // run-ahead decode buffers
        size_t rb = (size_t)MAXR * TOPK * 8 + (size_t)MAXR * D * 4;
        void* p = nullptr;
        DCK(cudaHostAlloc(&p, NL * 64, cudaHostAllocPortable | cudaHostAllocMapped));
        E->rflag = (volatile int*)p;
        DCK(cudaHostGetDevicePointer((void**)&E->d_rflag, p, 0));
        DCK(cudaHostAlloc(&p, NL * 64, cudaHostAllocPortable | cudaHostAllocMapped));
        E->yflag = (volatile int*)p;
        DCK(cudaHostGetDevicePointer((void**)&E->d_yflag, p, 0));
        for (int l = 0; l < NL; l++) {
            E->rflag[l * 16] = -1;
            E->yflag[l * 16] = -1;
            DCK(cudaHostAlloc((void**)&E->h_rp[l], rb, cudaHostAllocPortable | cudaHostAllocMapped));   // R10: written by the GPU
            DCK(cudaHostGetDevicePointer((void**)&E->d_rp[l], E->h_rp[l], 0));
            DCK(cudaHostAlloc((void**)&E->hy[l], (size_t)MAXR * D * 4, cudaHostAllocPortable | cudaHostAllocMapped));
            DCK(cudaHostGetDevicePointer((void**)&E->d_hy[l], E->hy[l], 0));
        }
        DCK(cudaHostAlloc(&p, NL * 64, cudaHostAllocPortable | cudaHostAllocMapped));
        E->pflag = (volatile int*)p;
        DCK(cudaHostGetDevicePointer((void**)&E->d_pflag, p, 0));
        DCK(cudaHostAlloc(&p, NL * PRED_STRIDE * 4, cudaHostAllocPortable | cudaHostAllocMapped));
        E->hpred = (volatile int*)p;
        if (g_rec) {
            DCK(cudaHostAlloc((void**)&g_rec_x, (size_t)REC_T * NL * 2 * D * 2, cudaHostAllocPortable | cudaHostAllocMapped));
            DCK(cudaHostGetDevicePointer((void**)&g_rec_dx, g_rec_x, 0));
            g_rec_rows.assign((size_t)REC_T * NL, RecRow{});
            logf_("predictor records on (%s_<k>.bin, %.2f GiB pinned)\n", g_rec, (double)REC_T * NL * 2 * D * 2 / 1073741824.0);
        }
        DCK(cudaHostGetDevicePointer((void**)&E->d_hpred, p, 0));
        for (int l = 0; l < NL; l++) E->pflag[l * 16] = -1;
        DCK(cudaHostAlloc(&p, NL * 64, cudaHostAllocPortable | cudaHostAllocMapped));
        E->pflag2 = (volatile int*)p;
        DCK(cudaHostGetDevicePointer((void**)&E->d_pflag2, p, 0));
        for (int l = 0; l < NL; l++) E->pflag2[l * 16] = -1;
        DCK(cudaHostAlloc(&p, NL * PRED_STRIDE * 4, cudaHostAllocPortable | cudaHostAllocMapped));
        E->hpred2 = (volatile int*)p;
        DCK(cudaHostGetDevicePointer((void**)&E->d_hpred2, p, 0));
        // R19 [dsv4-r19-pipe]: opt-in (DSV4_PIPE=1): exact, but no faster here (a half-batch's GPU chain costs ~85 % of
        // the full pass's, the GGUF pool GEMV is latency bound, so two halves need ~1.7x the GPU time the lane hides)
        E->pipe = o.dspark && getenv("DSV4_PIPE") && getenv("DSV4_NOENQT") == nullptr;
        if (getenv("DSV4_PIPE_MIN")) E->pipe_min = std::max(2, atoi(getenv("DSV4_PIPE_MIN")));
        E->early = g_early_env && E->gg && E->psplit && o.dyn && !E->pipe && !E_rp_off && o.prefetch > 0;
        if (g_early_env) {
            DCK(cudaHostAlloc(&p, NL * 64, cudaHostAllocPortable | cudaHostAllocMapped));
            E->h_edec = (volatile int*)p;
            DCK(cudaHostGetDevicePointer((void**)&E->d_edec, p, 0));
            for (int l = 0; l < NL * 16; l++) E->h_edec[l] = -1;
            for (auto& c : E->cards) {
                DevGuard dg(c.dev);
                c.d_early = (int*)dmalloc(NL * 4 * 4);
                DCK(cudaMemset(c.d_early, 0xff, NL * 4 * 4));
            }
            for (auto& W : E->L) {
                DevGuard dg(E->cards[W.card].dev);
                DCK(cudaEventCreateWithFlags(&W.early_ev, cudaEventDisableTiming));
            }
            logf_("early admission (DSV4_EARLY): %s\n", E->early ? "on" : "off (needs the GGUF split pool, run-ahead, prefetch, no pipe)");
        }
        if (E->pipe) {   // R19 [dsv4-r19-pipe]: the second micro-batch's handshakes
            auto& mh = E->mbh;
            auto& md = E->mbd;
            auto flags = [&](volatile int** h, int** d, size_t bytes, bool init) {
                void* q = nullptr;
                DCK(cudaHostAlloc(&q, bytes, cudaHostAllocPortable | cudaHostAllocMapped));
                *h = (volatile int*)q;
                DCK(cudaHostGetDevicePointer((void**)d, q, 0));
                if (init)
                    for (int l = 0; l < NL; l++) (*h)[l * 16] = -1;
            };
            flags(&mh.rflag, &md.d_rflag, NL * 64, true);
            flags(&mh.yflag, &md.d_yflag, NL * 64, true);
            flags(&mh.pflag, &md.d_pflag, NL * 64, true);
            flags(&mh.pflag2, &md.d_pflag2, NL * 64, true);
            flags(&mh.hpred, &md.d_hpred, NL * PRED_STRIDE * 4, false);
            flags(&mh.hpred2, &md.d_hpred2, NL * PRED_STRIDE * 4, false);
            for (int l = 0; l < NL; l++) {
                DCK(cudaHostAlloc((void**)&mh.h_rp[l], rb, cudaHostAllocPortable | cudaHostAllocMapped));
                DCK(cudaHostGetDevicePointer((void**)&md.d_rp[l], mh.h_rp[l], 0));
                DCK(cudaHostAlloc((void**)&mh.hy[l], (size_t)MAXR * D * 4, cudaHostAllocPortable | cudaHostAllocMapped));
                DCK(cudaHostGetDevicePointer((void**)&md.d_hy[l], mh.hy[l], 0));
            }
            logf_("verify passes of >= %d rows run as two pipelined micro-batches (DSV4_PIPE=1)\n", E->pipe_min);
        }
        E->prev_ids.assign(NL * TOPK, -1);
        E->runahead = getenv("DSV4_RUNAHEAD") != nullptr;   // measured ~1 % slower than the synchronous path (R3-3): opt-in
        E->enq_thread = getenv("DSV4_NOENQT") == nullptr;   // R9 default (R9-5); DSV4_NOENQT=1: one host thread
        if (E->enq_thread) {
            E->runahead = true;
            E->enq_th = std::thread([E] {
                cpu_set_t cs;   // the SMT sibling of the last lane core; R13: a core of its own when the lane leaves one
                CPU_ZERO(&cs);  // (the capture of the next step ran beside lane worker 23, the barrier waits for it)
                // review r14 L6: cores from the CPU count and the SMT state instead of a fixed 24 (lane = CPUs 0 .. t-1);
                // a free core of its own if t < cores, else the last CPU (the SMT sibling of the last core) if outside
                // the lane; review r13 L9 / re-review N3: no free CPU outside the lane -> unpinned
                const int t = E->o.cpu_threads, nproc = (int)sysconf(_SC_NPROCESSORS_ONLN);
                int smt = 0;
                if (FILE* fs = fopen("/sys/devices/system/cpu/smt/active", "r")) {
                    if (fscanf(fs, "%d", &smt) != 1) smt = 0;
                    fclose(fs);
                }
                const int ncore = smt == 1 ? std::max(1, nproc / 2) : nproc;
                const int ecpu = t < ncore ? t : (nproc - 1 >= t ? nproc - 1 : -1);
                if (ecpu < 0) logf_("warning: no CPU outside the lane's %d for the enqueue thread: it runs unpinned\n", t);
                else {
                    CPU_SET(ecpu, &cs);   // (review r14 S3: the mask only when pinning)
                    if (pthread_setaffinity_np(pthread_self(), sizeof(cs), &cs) != 0)   // review r9 L2
                        logf_("warning: enqueue thread pinning to CPU %d failed (runs unpinned)\n", ecpu);
                }
                int seen = 0;
                while (true) {
                    E->enq_go.wait(seen, std::memory_order_acquire);
                    seen = E->enq_go.load(std::memory_order_acquire);
                    if (E->enq_quit) return;
                    const auto tq = clk::now();
                    E->enq_job();
                    E->enq_ms = ms_since(tq);   // R20: enqueue time of the job (read after enq_done)
                    E->enq_done.store(1, std::memory_order_release);
                }
            });
        }
        E->ra_multi = getenv("DSV4_NORAM") == nullptr;
        // R10 whole-step decode graph: needs the enqueue thread (capture one step ahead), NVMe Engram rows, no DSpark
        // review r10 H2: the graph's hand-off and head are LL12 words, allocated only with the attention split (P2P, 3 cards)
        // R17: DSV4_DSGRAPH=1: DSpark keeps the graph for its one-row steps (verify passes stay eager); opt-in: R17-3
        // measured the mix slower (verify passes' lane +12-30 %, unexplained)
        E->graph = getenv("DSV4_NOGRAPH") == nullptr && E->enq_thread && E->eg_nvme &&
                   (!E->dspark || getenv("DSV4_DSGRAPH") != nullptr) && o.dump.empty() &&
                   E->asplit && E->cards.size() == 3;
        if (getenv("DSV4_NOGRAPH") == nullptr && !E->graph)
            logf_("step graph off (needs the enqueue thread, NVMe Engram rows, the attention split over 3 cards)\n");
        if (E->graph) logf_("decode step as one CUDA graph over the cards (captured one step ahead)\n");
        E->graph_cfg = E->graph;
        DCK(cudaHostAlloc((void**)&E->h_tok, 64, cudaHostAllocPortable));
        void* pe = nullptr;
        DCK(cudaHostAlloc(&pe, 2 * 64, cudaHostAllocPortable | cudaHostAllocMapped));
        E->egflag = (volatile int*)pe;
        DCK(cudaHostGetDevicePointer((void**)&E->d_egflag, pe, 0));
        E->egflag[0] = E->egflag[16] = -1;
    }
    if (E->dyn_fp4) {
        E->xr.th.emplace_back([E] { E->xr.worker(); });
        E->xr.th.emplace_back([E] { E->xr.worker(); });
    }
    if (E->gm) {   // [dsv4-rel-ram]: admissions of experts without a host copy: re-packed from the GGUF by 4 readers
        for (auto& c : E->cards)
            if (!c.xj) {
                c.xj = new XJob[XJ_PER_CARD];
                for (int j = 0; j < XJ_PER_CARD; j++) DCK(cudaHostAlloc((void**)&c.xj[j].buf, XJ_BYTES, cudaHostAllocPortable));
            }
        for (int i = 0; i < 4; i++) E->xr.th.emplace_back([E] { E->xr.worker(); });
    }
    if (E->gg && !E_fx_off) {   // [dsv4-pf2-fx]: the fused kernels loaded now, not inside the first prefill
        // review pf11 L7: a card that refuses one (opt-in shared memory) turns that path off instead of failing the load
        bool fx_ok_all = true, fxm_ok_all = true;
        for (auto& c : E->cards) {
            DevGuard dg(c.dev);
            fx_ok_all = fx_ok_all && fx_warm();
            fxm_ok_all = fxm_ok_all && fxm_warm();
        }
        if (!fx_ok_all) {
            E_fx_off = true;
            logf_("WARNING: fused IQ expert kernels (k_fx) not usable on this device: prefill experts take dequant + cuBLAS\n");
        } else if (!fxm_ok_all) {
            g_fxm_off = true;
            E_fx_max = std::min(E_fx_max, 128);
            logf_("WARNING: tiled fused IQ expert kernel (k_fxm) not usable on this device: k_fx up to 128 rows, cuBLAS above\n");
        }
        if (!E_fx_off && fxl_min() < E_fx_max)
            logf_("prefill fused experts: k_fx up to %d rows, k_fxm %d .. %d rows\n", fxl_min(), fxl_min() + 1, E_fx_max);
        else if (!E_fx_off) logf_("prefill fused experts: k_fx up to %d rows (k_fxm off)\n", E_fx_max);
    }
    // [dsv4-rel3-isa] the lane's kernels: AVX-512 where the CPU has the Zen 4 set, else AVX2 (DSV4_CPU_ISA=avx2 forces it)
    DCHECK(o.cpu_threads <= 0 || cpu_isa() >= 0, "the DSV4 CPU lane needs an x86-64-v3 CPU (AVX2, FMA, F16C, BMI2)");
    E->lane.start(o.cpu_threads, 0);
    logf_("CPU lane: %d threads, %s kernels%s\n", o.cpu_threads, cpu_isa_name(E->lane.isa()),
          getenv("DSV4_CPU_ISA") ? " (DSV4_CPU_ISA)" : "");
    // [dsv4-pf2-cpu]: workers on the lane's CPUs 1 .. t - 1 (idle during a prefill; they block between runs: review cpu
    // H1), DSV4_CPUPF_SMT=1: + their siblings; only where moe_prefill can use them (distributed prefill on 3 cards)
    if (pfcpu_on(E) && o.pdist && E->asplit && E->cards.size() == 3 && o.chunk > MAXR) {
        const int t = o.cpu_threads, np = (int)sysconf(_SC_NPROCESSORS_ONLN);
        std::vector<int> cpus;
        for (int k = 1; k < t && k < np; k++) cpus.push_back(k);
        if (getenv("DSV4_CPUPF_SMT"))
            for (int k = 1; k < t && k + np / 2 < np - 4; k++) cpus.push_back(k + np / 2);   // (top 4: helpers / enqueue)
        // review cpu L6: no pinned buffers -> no CPU path (a warning, not an abort)
        if (cudaHostAlloc((void**)&E->h_cxq, (size_t)o.chunk * D * 2, cudaHostAllocPortable) == cudaSuccess &&
            cudaHostAlloc((void**)&E->h_cy, (size_t)NE * E_cpu_max * D * 4, cudaHostAllocPortable | cudaHostAllocMapped) == cudaSuccess) {
            E->pfcpu.start(cpus);
            logf_("prefill CPU experts: <= %d rows on %d CPU threads (%.0f GB/s modelled)\n", E_cpu_max, (int)cpus.size(), E_cpu_bw);
        } else {
            cudaGetLastError();
            if (E->h_cxq) { cudaFreeHost(E->h_cxq); E->h_cxq = nullptr; }
            logf_("WARNING: prefill CPU experts off (no pinned buffers)\n");
        }
    }
    E->lane.set_codebooks(E->cb[0], E->cb[1]);
    E->jobs.reserve(MAXR * TOPK);
    tier_snapshot(E);   // [dsv4-rel-tierreset]
    tail_readahead(E);  // [dsv4-rel-load128]
    // [dsv4-pf10-embra] the Q8_0 embedding's rows are read on use from the file mapping: a prompt's first touch of its
    // tokens' rows faulted them from the disk on the prefill's critical path (cold page cache: 2.5K embed 171 vs 26 ms
    // warm, 10K 363 ms). Kernel readahead of the whole table (703 MB), asynchronous (no wait, no copy, page cache only:
    // same bytes, outputs unchanged). Review pf17b M1: issued here, after every whole-file POSIX_FADV_DONTNEED of the
    // load (GGUF seeding, copy-less seeds), which dropped it when issued at the head step. DSV4_NOEMBRA=1 off
    // gate rerun: one MADV_WILLNEED over the table read only ~4 MB (mincore 0.6 % at the first prefill): the kernel caps
    // a WILLNEED readahead at max(ra_pages, io_pages) per call, so the table is advised in 128 KB steps (each submits
    // its reads and returns; ~5,500 calls)
    if (E->embed_q && !env_on("DSV4_NOEMBRA")) {
        const uintptr_t pg = 4096, a0 = (uintptr_t)E->embed_q & ~(pg - 1), a1 = (uintptr_t)E->embed_q + (size_t)V * (D / 32) * 34;
        for (uintptr_t a = a0; a < a1; a += 128 << 10) madvise((void*)a, std::min<uintptr_t>(128 << 10, a1 - a), MADV_WILLNEED);
    }
    if (getenv("DSV4_EGMIRROR") && *getenv("DSV4_EGMIRROR") && !g_egm_seen)   // review pf17b L5
        logf_("warning: DSV4_EGMIRROR ignored: no GGUF Engram tables read from the disk in this configuration\n");
    if (E->ctx_cap > 0 && E->ctx_cap < E->o.maxctx) {   // [dsv4-rel8-ctxcap] (lz_cap_ctx)
        logf_("WARNING: max context %d -> %d positions: the context buffers could not be mapped in full (see above)\n",
              E->o.maxctx, E->ctx_cap);
        E->o.maxctx = E->ctx_cap;
    }
    logf_("engine ready in %.1f s\n", ms_since(t0) / 1e3);
    log_mem("at ready");
    mem_guard(E, true);
    return E;
}

void print_memory(Engine* E) {
    for (auto& c : E->cards) {
        DCK(cudaSetDevice(c.dev));
        size_t fr, tot;
        DCK(cudaMemGetInfo(&fr, &tot));
        int nh = 0;
        for (int l = c.l0; l < c.l1; l++)
            for (int e = 0; e < NE; e++) nh += E->L[l].h_slot[e] >= 0;
        logf_("card %d: used %.2f GB of %.2f, free %.2f; layers %d..%d; resident experts %d\n", c.dev,
              (tot - fr) / 1e9, tot / 1e9, fr / 1e9, c.l0, c.l1 - 1, nh);
        double sum = 0;   // R9 VRAM audit
        for (int t = 1; t < NMTAG; t++) sum += g_mb[c.dev][t];
        std::string line;
        char b[96];
        snprintf(b, sizeof b, "  %s %.3f", MTAG[0], ((double)(tot - fr) - sum) / 1e9);
        line += b;
        for (int t = 1; t < NMTAG; t++) {
            snprintf(b, sizeof b, " | %s %.3f", MTAG[t], g_mb[c.dev][t] / 1e9);
            line += b;
        }
        logf_("card %d VRAM GB:%s\n", c.dev, line.c_str());
    }
}

static void fut_dump();
static void rec_dump() {
    if (!g_rec_x || g_rec_hi < g_rec_lo) return;
    const int n = std::min(REC_T, g_rec_hi - g_rec_lo + 1), s0 = g_rec_hi - n + 1;
    char nm[512];
    snprintf(nm, sizeof nm, "%s_%d.bin", g_rec, g_rec_k++);
    FILE* f = fopen(nm, "wb");
    if (f) {
        int hd[4] = {n, s0, (int)sizeof(RecRow), REC_T};
        fwrite(hd, 4, 4, f);
        for (int i = 0; i < n; i++) fwrite(&g_rec_rows[(size_t)((s0 + i) % REC_T) * NL], sizeof(RecRow), NL, f);
        for (int i = 0; i < n; i++) fwrite(g_rec_x + (size_t)((s0 + i) % REC_T) * NL * 2 * D, 2, (size_t)NL * 2 * D, f);
        fclose(f);
        logf_("predictor records: %d positions (from %d) -> %s\n", n, s0, nm);
    }
    g_rec_lo = 1 << 30;
    g_rec_hi = -1;
}
static void dacc_score(Engine* E);
// [dsv4-rel-tierreset] (coordinator 11-14): greedy text must not depend on earlier requests of the same process; the
// dynamic tier's residency is history, and the GPU pool and the CPU lane are not bit-identical per expert. Snapshot
// the GGUF pool's state after the load; tier_reset() puts it back (changed slots re-copied from their host copies,
// the LFU scores / miss times / clock restored). Synchronous; GGUF split pool only (else a no-op)
// [dsv4-rel-load128] (coordinator 11-14: ld2's prefill 26.8 -> 39.8 s): the experts with neither a host copy nor a
// VRAM slot are read through the GGUF mapping (the lane, the prefill's re-pack); the loader dropped the files' pages.
// Ask the kernel to read them ahead (MADV_WILLNEED: the syscall runs inside create, the reads mostly after) right after the load, hottest first, within
// MemAvailable - 8 GiB - 2 GiB (page cache stays reclaimable). DSV4_NOTAILRA=1: off
static void tail_readahead(Engine* E) {
    if (!E->gm || getenv("DSV4_NOTAILRA")) return;
    std::vector<std::tuple<int, int, int>> tl;   // (rank, l, e)
    for (int l = 0; l < NL; l++) {
        const Layer& W = E->L[l];
        for (int e = 0; e < NE; e++)
            if (W.hfmt[e] >= 0 && !W.hx[e] && W.h_dslot[e] < 0 && W.h_slot[e] < 0) tl.push_back({W.rank[e], l, e});
    }
    if (tl.empty()) return;
    std::sort(tl.begin(), tl.end());
    const double av = mem_avail_gib();
    double budget = (av - 10.0) * 1073741824.0, done = 0;
    const long pg = sysconf(_SC_PAGESIZE);
    int n = 0;
    for (auto& [rk, l, e] : tl) {
        size_t tot = 0;
        for (int m = 0; m < 3; m++) {
            size_t nb;
            const uint8_t* p = E->gg->m.expert_slice(*E->gg->m.expert_tensor(l, m), e, &nb);
            if (!p) continue;
            tot += nb;
        }
        if (done + (double)tot > budget) break;
        for (int m = 0; m < 3; m++) {
            size_t nb;
            const uint8_t* p = E->gg->m.expert_slice(*E->gg->m.expert_tensor(l, m), e, &nb);
            if (!p) continue;
            uint8_t* a = (uint8_t*)((uintptr_t)p & ~(uintptr_t)(pg - 1));
            madvise(a, (size_t)(p + nb - a), MADV_WILLNEED);
        }
        done += (double)tot;
        n++;
    }
    logf_("copy-less cold tail: %d of %zu experts (%.2f GiB) read ahead into the page cache (MemAvailable %.1f GiB)\n", n,
          tl.size(), done / 1073741824.0, av);
}
static void tier_snapshot(Engine* E) {
    if (!E->gg || !E->psplit || !E->o.dyn) return;
    for (auto& W : E->L) {
        W.seed_e = W.dyn_e;
        W.seed_score.assign(W.score.begin(), W.score.end());
        W.seed_score_t = W.score_t;
        W.seed_miss_t = W.miss_t;
    }
    E->seed_dc = E->dyn_clock;
    E->seed_swap_credit = E->swap_credit;
    E->seed_swap_t = E->swap_t;
    E->seed_nv_credit = E->nv_credit;
    E->seed_nv_t = E->nv_t;
    E->seed_card_nv.clear();
    for (auto& c : E->cards) E->seed_card_nv.push_back({c.nv_credit, c.nv_t});
}
std::string lane_prof(Engine* E, bool reset) { return E->lane.prof_report(reset); }

double tier_reset(Engine* E) {
    // review rel3 M1 / L1: not with the early admissions (their spare slot) or the swap tier; said once
    if (!E->gg || !E->psplit || !E->o.dyn || E->gswap || E->early || E->seed_card_nv.empty()) {
        static bool said = false;
        if (!said) logf_("tier reset: not available in this configuration (GGUF split pool, no swap / early admissions)\n");
        said = true;
        return -1;
    }
    auto t0 = clk::now();
    cap_wait(E);
    lz_shrink(E, true);   // [dsv4-ctx-lazy] the seeds' slots back first (lent ones are skipped below otherwise)
    for (auto& c : E->cards) {
        DevGuard dg(c.dev);
        DCK(cudaDeviceSynchronize());
    }
    int nch = 0;
    for (auto& W : E->L) {   // pending admissions dropped (their copies are done after the sync)
        Card& c = E->cards[W.card];
        for (size_t i = 0; i < W.dpool.size(); i++) {
            if (W.pend_job[i] >= 0) {
                XJob& J = c.xj[W.pend_job[i]];
                while (J.state.load(std::memory_order_acquire) == 1) std::this_thread::sleep_for(std::chrono::microseconds(50));
                J.state.store(0, std::memory_order_release);
                W.pend_job[i] = -1;
            }
            if (W.pend_e[i] >= 0) W.pend_e[i] = -1;
            if (W.dyn_e[i] != W.seed_e[i] && W.dyn_e[i] >= 0) {   // pass 1: the changed slots' experts out
                set_pslot(E, W, W.dyn_e[i], -1);
                W.dyn_e[i] = -1;
            }
        }
    }
    for (auto& c : E->cards) {
        DevGuard dg(c.dev);
        DCK(cudaDeviceSynchronize());   // table updates before the slots are overwritten
    }
    std::vector<uint8_t> tmp;
    for (auto& W : E->L) {   // pass 2: the seeds' bytes back into their slots
        Card& c = E->cards[W.card];
        const size_t XB1 = expert_bytes(pool_fmt(E, W.lid));
        for (size_t i = 0; i < W.dpool.size(); i++) {
            const int e = W.seed_e[i];
            if (W.dyn_e[i] == e || e < 0 || W.pend_e[i] == -4) continue;   // [dsv4-ctx-lazy]: a lent slot stays lent
            Card& sc = pool_card(E, c, W.dpool[i]);
            DevGuard dg(sc.dev);
            if (W.hx[e]) DCK(cudaMemcpyAsync(pool_slot_ptr(E, c, W.lid, W.dpool[i]), W.hx[e], XB1, cudaMemcpyHostToDevice, sc.cp));
            else {   // a copy-less seed (static, so normally never moved)
                tmp.resize(XB1);
                gg_repack_expert(E, W.lid, e, tmp.data());
                DCK(cudaMemcpy(pool_slot_ptr(E, c, W.lid, W.dpool[i]), tmp.data(), XB1, cudaMemcpyHostToDevice));
            }
            nch++;
        }
    }
    for (auto& c : E->cards) {
        DevGuard dg(c.dev);
        DCK(cudaDeviceSynchronize());
    }
    for (auto& W : E->L) {   // pass 3: tables, scores, miss times
        for (size_t i = 0; i < W.dpool.size(); i++) {
            const int e = W.seed_e[i];
            if (W.dyn_e[i] == e || e < 0 || W.pend_e[i] == -4) continue;   // [dsv4-ctx-lazy]: a lent slot stays lent
            set_pslot(E, W, e, W.dpool[i]);
            W.dyn_e[i] = e;
        }
        std::copy(W.seed_score.begin(), W.seed_score.end(), W.score.begin());
        W.score_t = W.seed_score_t;
        W.miss_t = W.seed_miss_t;
    }
    E->dyn_clock = E->seed_dc;
    E->swap_credit = E->seed_swap_credit;
    E->swap_t = E->seed_swap_t;
    E->nv_credit = E->seed_nv_credit;
    E->nv_t = E->seed_nv_t;
    for (size_t k = 0; k < E->cards.size(); k++) {
        E->cards[k].nv_credit = E->seed_card_nv[k].first;
        E->cards[k].nv_t = E->seed_card_nv[k].second;
    }
    for (auto& c : E->cards) {
        DevGuard dg(c.dev);
        DCK(cudaDeviceSynchronize());
    }
    const double ms = ms_since(t0);
    logf_("tier reset: %d slots restored in %.1f ms\n", nch, ms);
    return ms;
}
// per-sequence diagnostic reports (opt-in recorders), also when a sequence moves back to its snapshot (review dec4 L6)
static void seq_stats_flush(Engine* E) {
    rec_dump();
    fut_dump();
    dacc_score(E);
    if (E->early && g_early_n[0]) {   // dsv4-fut [dsv4-fut-early]: per prompt (reset comes before the next one)
        logf_("early admissions: %ld requested, %ld in time (%.1f %%), %ld late; routing already out at the request %ld, "
              "copy done when the host had the routing %ld\n", g_early_n[0], g_early_n[1], 100.0 * g_early_n[1] / g_early_n[0],
              g_early_n[2], g_early_n[3], g_early_n[4]);
        for (long& x : g_early_n) x = 0;
    }
}
// ---------------------------------------------------------------- [dsv4-rel4-p2] compressed K/V pool
static const int KV_SRCS[4] = {2, 8, 14, 20};
static void seq_quiesce(Engine* E);
// the active sequence's views at its base pages (every card: the same page ids)
static void kv_bind(Engine* E) {
    for (auto& c : E->cards)
        for (int si = 0; si < 4; si++) {
            c.src[si].ckv = c.src[si].pck + (size_t)E->ab[si] * KVPG * HD;
            c.src[si].ik = c.src[si].pik + (size_t)E->ab[si] * KVPG * IDIM;
        }
}
// a base page for a sequence of source si that may grow to kvcap pages: the window [b, b + kvcap) with the fewest pages
// of other parked sequences (lowest b on a tie; self = a parked id whose own pages do not count)
static int kv_pick(Engine* E, int si, int self) {
    const std::vector<int>& own = E->kvown[si];
    const int P = E->kvP[si], W = E->kvcap[si];
    int best = 0, bc = INT32_MAX, cnt = 0;
    for (int p = 0; p < W; p++) cnt += own[p] > 0 && own[p] != self;
    for (int b = 0;; b++) {
        if (cnt < bc) bc = cnt, best = b;
        if (b + W >= P || bc == 0) break;
        cnt += (own[b + W] > 0 && own[b + W] != self) - (own[b] > 0 && own[b] != self);
    }
    return best;
}
static void kv_dev_sync(Engine* E) {
    for (auto& c : E->cards) {
        DevGuard dg(c.dev);
        DCK(cudaDeviceSynchronize());
    }
}
// [dsv4-rel5-p3] host buffers for evicted K/V. A buffer is pinned (anonymous mapping + cudaHostRegister from 32 MiB, the
// R24 path; cudaHostAlloc below; malloc when neither works), so evictions and restores run as async copies on every
// card's link at once. A freed buffer stays in a cache for the next eviction (no allocation per eviction) while the
// cache holds at most min(DSV4_KVH_CACHE_MIB, default 4096, half of MemAvailable above the archive floor); the cache
// goes first under memory pressure (seq_res_trim, and before an eviction that would cross the floor)
static const bool g_kvh_log = getenv("DSV4_KVH_LOG") && atoi(getenv("DSV4_KVH_LOG")) != 0;   // timing lines per copy
static size_t kvh_keep_cap(Engine* E) {
    static const double cap_mib = getenv("DSV4_KVH_CACHE_MIB") ? atof(getenv("DSV4_KVH_CACHE_MIB")) : 4096.0;
    const double av = mem_avail_gib();
    if (av < 0) return 0;
    const double room = (av - E->kv_floor_gib) / 2;
    return room <= 0 ? 0 : (size_t)(std::min(cap_mib / 1024.0, room) * 1073741824.0);
}
static void kvh_release(Engine* E, Engine::HostKv& h) {
    if (!h.p) return;
    if (h.pin) {
        host_free(h.p);
        E->kv_pinned -= h.cap;
    } else {
        free(h.p);
    }
    h = Engine::HostKv();
}
static void kvh_trim(Engine* E, size_t keep) {   // the largest cached buffers go first
    while (E->kvfree_b > keep && !E->kvfree.empty()) {
        auto it = std::prev(E->kvfree.end());
        E->kvfree_b -= it->second.cap;
        kvh_release(E, it->second);
        E->kvfree.erase(it);
    }
}
static void kvh_put(Engine* E, Engine::HostKv& h) {
    if (!h.p) return;
    const size_t keep = kvh_keep_cap(E);
    if (h.pin && E->kvfree_b + h.cap <= keep) {
        h.n = 0;
        E->kvfree_b += h.cap;
        E->kvfree.emplace(h.cap, h);
        h = Engine::HostKv();
    } else {
        kvh_release(E, h);
    }
    kvh_trim(E, keep);
}
// a buffer of b bytes: the smallest cached one that fits (at most 1.5 b + 64 MiB, so a small copy does not hold a big
// buffer), else a new one. false: no host memory
static bool kvh_reuse(Engine* E, Engine::HostKv& h, size_t b) {
    auto it = E->kvfree.lower_bound(b);
    if (it == E->kvfree.end() || it->first > b + b / 2 + (64u << 20)) return false;
    h = it->second;
    E->kvfree_b -= h.cap;
    E->kvfree.erase(it);
    h.n = b;
    E->kvh_n[0]++;
    return true;
}
static bool kvh_get(Engine* E, Engine::HostKv& h, size_t b) {
    if (kvh_reuse(E, h, b)) return true;
    const size_t cap = (b + (2u << 20) - 1) / (2u << 20) * (2u << 20);
    static const bool off = getenv("DSV4_KVH_PINNED") && atoi(getenv("DSV4_KVH_PINNED")) == 0;   // off-switch: pageable
    uint8_t* p = nullptr;
    bool pin = !off;
    if (pin && (g_thp || g_hreg) && cap >= (32u << 20)) p = host_alloc_thp(cap, false);   // review rel5 H1 (host_free)
    if (pin && !p && cudaHostAlloc((void**)&p, cap, cudaHostAllocPortable) != cudaSuccess) {
        cudaGetLastError();
        p = nullptr;
    }
    if (!p) {
        pin = false;
        p = (uint8_t*)malloc(cap);
        if (!p) return false;
    }
    h.p = p;
    h.n = b;
    h.cap = cap;
    h.pin = pin;
    if (pin) E->kv_pinned += cap;
    E->kvh_n[pin ? 1 : 2]++;
    return true;
}
// a parked sequence leaves VRAM: its rows (owner card) into host RAM when MemAvailable keeps 12 GiB after the copy, else
// dropped (its unpark then fails: the caller re-prefills)
static void kv_evict(Engine* E, int rid) {
    auto it = E->kvres.find(rid);
    if (it == E->kvres.end()) return;
    Engine::KvRes& r = it->second;
    size_t b = 0;
    for (int si = 0; si < 4; si++) b += (size_t)r.tc[si] * (HD + IDIM) * 2;
    const auto t0 = std::chrono::steady_clock::now();
    // review rel5 L4: a fitting cached buffer costs no new memory (no floor check); else the cache goes first when the
    // new buffer would cross the floor
    bool keep = !r.dead && b > 0 && kvh_reuse(E, r.host, b);   // review rel5 L1: nothing to keep for b == 0
    if (!keep && !r.dead && b > 0) {
        double av = mem_avail_gib();
        if (av >= 0 && av - b / 1073741824.0 < E->kv_floor_gib && E->kvfree_b) {   // [dsv4-rel5-p3] the cache goes first
            kvh_trim(E, 0);
            av = mem_avail_gib();
        }
        keep = av >= 0 && av - b / 1073741824.0 >= E->kv_floor_gib && kvh_get(E, r.host, b);   // review rel4 M1
    }
    if (keep) {
        const double t_alloc = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        kv_dev_sync(E);
        const auto t1 = std::chrono::steady_clock::now();
        uint8_t* o = r.host.p;
        for (int si = 0; si < 4; si++) {   // [dsv4-rel5-p3] async on the owner cards' streams (pinned: in parallel)
            if (r.tc[si] <= 0) continue;
            Card& c = E->cards[E->L[KV_SRCS[si]].card];
            DevGuard dg(c.dev);
            DCK(cudaMemcpyAsync(o, c.src[si].pck + (size_t)r.base[si] * KVPG * HD, (size_t)r.tc[si] * HD * 2,
                                cudaMemcpyDeviceToHost, c.st));
            o += (size_t)r.tc[si] * HD * 2;
            DCK(cudaMemcpyAsync(o, c.src[si].pik + (size_t)r.base[si] * KVPG * IDIM, (size_t)r.tc[si] * IDIM * 2,
                                cudaMemcpyDeviceToHost, c.st));
            o += (size_t)r.tc[si] * IDIM * 2;
        }
        kv_dev_sync(E);   // the pages are free for others after this
        if (g_kvh_log)
            logf_("K/V pool: evicted %.1f MB to host (%s) in %.2f ms (+ %.2f ms buffer)\n", b / 1e6,
                  r.host.pin ? "pinned" : "pageable",
                  std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t1).count(), t_alloc);
        E->kv_host += r.host.size();
        E->kv_n[3]++;
    } else {
        kvh_release(E, r.host);
        r.dead = true;
        E->kv_n[4]++;
    }
    for (int si = 0; si < 4; si++) {
        for (int p = r.base[si]; p < r.base[si] + r.np[si]; p++)
            if (E->kvown[si][p] == rid) E->kvown[si][p] = 0;
        r.np[si] = 0;
    }
}
// a parked run of source si out of the pages [lo, hi): moved on the device (every card) to the highest free run of its
// length outside them, else the whole parked sequence leaves VRAM (kv_evict)
static void kv_move_out(Engine* E, int rid, int si, int lo, int hi) {
    auto it = E->kvres.find(rid);   // review rel4 L1
    if (it == E->kvres.end()) return;
    Engine::KvRes& r = it->second;
    const int n = r.np[si];
    std::vector<int>& own = E->kvown[si];
    int to = -1;
    for (int e = E->lz ? E->kvM[si] : E->kvP[si], run = 0; e-- > 0;) {   // scan down: run = free pages from e up ([dsv4-ctx-lazy]: mapped ones)
        run = own[e] == 0 && (e >= hi || e < lo) ? run + 1 : 0;
        if (run >= n) { to = e; break; }
    }
    if (to < 0 || n <= 0) {
        kv_evict(E, rid);
        return;
    }
    for (auto& c : E->cards) {
        DevGuard dg(c.dev);
        DCK(cudaMemcpyAsync(c.src[si].pck + (size_t)to * KVPG * HD, c.src[si].pck + (size_t)r.base[si] * KVPG * HD,
                            (size_t)r.tc[si] * HD * 2, cudaMemcpyDeviceToDevice, c.st));
        DCK(cudaMemcpyAsync(c.src[si].pik + (size_t)to * KVPG * IDIM, c.src[si].pik + (size_t)r.base[si] * KVPG * IDIM,
                            (size_t)r.tc[si] * IDIM * 2, cudaMemcpyDeviceToDevice, c.st));
    }
    kv_dev_sync(E);   // review rel4 H1 / H2: later writes to the vacated pages (other streams, X1, sync copies) after it
    for (int p = r.base[si]; p < r.base[si] + n; p++)
        if (own[p] == rid) own[p] = 0;
    for (int p = to; p < to + n; p++) own[p] = rid;
    r.base[si] = to;
    E->kv_n[1]++;
}
// ---- [dsv4-ctx-lazy] growth and give-back of the lazily mapped context buffers
static const double E_lz_hold = getenv("DSV4_CTX_HOLD_S") ? atof(getenv("DSV4_CTX_HOLD_S")) : 30.0;
static size_t lz_pgb(int j) { return j < 4 ? (size_t)KVPG * HD * 2 : (size_t)KVPG * IDIM * 2; }
// chunks of zone l on card c that can be lent now (inside its evicted tail slots), counted from the region end
static int lz_avail(Engine* E, const Card& c, int l) {
    if (!c.lz_nmax[l]) return 0;
    const size_t xb = expert_bytes(pool_fmt(E, l));
    const size_t t = lz_up(c.ploff[l] + (size_t)(c.pnl[l] - c.lz_n[l]) * xb), hi = lz_dn(c.ploff[l] + (size_t)c.pnl[l] * xb);
    return hi > t ? std::min(c.lz_c1[l] - c.lz_c0[l], (int)((hi - t) / LZG)) : 0;
}
// pool slot q of layer l's region on card k leaves the tier for the loan: its expert out of the tables (the lane serves
// it), a pending admission into it dropped; marked -4 (lent: neither free nor a victim). false: a static expert sits
// there (not lendable). The caller synced the devices (pending copies done)
static bool lz_evict_slot(Engine* E, int k, int l, int q) {
    Layer& W = E->L[l];
    const int g = k * GS + E->cards[k].ps0[l] + q;
    for (size_t i = 0; i < W.dpool.size(); i++) {
        if (W.dpool[i] != g) continue;
        if (W.pend_job[i] >= 0) {   // a GGUF re-pack admission in flight: let the reader finish, drop it (as gborrow_ws)
            XJob& J = E->cards[W.card].xj[W.pend_job[i]];
            int js;
            for (auto tw = clk::now(); (js = J.state.load(std::memory_order_acquire)) == 1;) {
                std::this_thread::sleep_for(std::chrono::microseconds(50));
                DCHECK(ms_since(tw) < 120000, "GGUF expert re-pack (context loan): no answer for 120 s");
            }
            J.state.store(0, std::memory_order_release);
            W.pend_job[i] = -1;
        }
        if (!W.pend_m.empty() && W.pend_m[i] >= 0) {   // a swap's victim D2H is done (synced): its bytes are its host copy
            W.hx[W.pend_m[i]] = W.pend_h[i];
            W.pend_m[i] = -1;
            W.pend_h[i] = nullptr;
        }
        const int e = W.dyn_e[i];
        if (e >= 0) {
            set_pslot(E, W, e, -1);
            if (!W.inpool.empty()) W.inpool[e] = 0;
            E->lz_ev.push_back({l, g, e});
        }
        W.dyn_e[i] = -1;
        W.pend_e[i] = -4;
        E->lz_cnt[2]++;
        return true;
    }
    for (int e = 0; e < NE; e++)
        if (W.h_dslot[e] == g) return false;
    return true;   // a slot outside the pool lists that holds nothing
}
// the mapped pages / positions every card holds now
static void lz_sync_counts(Engine* E) {
    for (int si = 0; si < 4; si++) {
        size_t m = SIZE_MAX;
        for (auto& c : E->cards)
            if (c.lzb[si].cap) m = std::min({m, c.lzb[si].mapped / lz_pgb(si), c.lzb[4 + si].mapped / lz_pgb(4 + si)});
        E->kvM[si] = (int)std::min<size_t>(E->kvP[si], m);
    }
    size_t m = SIZE_MAX;
    for (auto& c : E->cards)
        for (int t = 8; t < 12; t++)
            if (c.lzb[t].cap) m = std::min(m, c.lzb[t].mapped / (32 * 4));
    E->ropeM = (int)std::min<size_t>(E->o.maxctx, m);
}
// [kb-dsv4-loanmap] (morning list 40) a driver-level map / unmap failure inside a loan or give-back fails that growth (the
// request: -2 / CtxNoRoom) instead of aborting; a chunk that can go neither to the buffer nor back to the pool is lost:
// kLzLost marks it (never lent again, its pool slots stay out of the tier: nothing is written to an unmapped range).
// DSV4_TEST_LZMAPFAIL=n: the n-th map of a loan / give-back fails (test; DSV4_TEST_LZMAPFAIL_ALL=1: that one and all after)
constexpr int kLzLost = 1 << 20;
static bool lz_map_lg(const Card& c, unsigned long long va, size_t b, unsigned long long h) {
    static const int nth = getenv("DSV4_TEST_LZMAPFAIL") ? atoi(getenv("DSV4_TEST_LZMAPFAIL")) : 0;
    static const bool all = getenv("DSV4_TEST_LZMAPFAIL_ALL") && atoi(getenv("DSV4_TEST_LZMAPFAIL_ALL"));
    static int n = 0;
    if (nth > 0 && (++n == nth || (all && n > nth))) {
        logf_("WARNING test: DSV4_TEST_LZMAPFAIL: map %d fails\n", n);
        return false;
    }
    return lz_map(c, va, b, h);
}
// every buffer mapped up to top[si] pages of each source (-1: as is) and rpos rope positions (-1: as is), in steps of at
// least 64 pages / 16,384 positions. Memory: chunks of the loan zones (their slots' experts evicted, round by round one
// more tail slot of every zone, so the loss spreads over the layers), else own memory
static bool lz_ensure(Engine* E, const int top[4], int rpos) {
    if (!E->lz) return true;
    int tp[4];
    bool any = false;
    for (int si = 0; si < 4; si++) {
        tp[si] = E->kvM[si];
        if (top[si] > E->kvM[si]) tp[si] = std::min(E->kvP[si], std::max(top[si], E->kvM[si] + 64)), any = true;
    }
    int rp = E->ropeM;
    if (rpos > E->ropeM) rp = std::min(E->o.maxctx, std::max(rpos, E->ropeM + 16384)), any = true;
    if (!any) return true;
    // [kb-dsv4-ctxgrow] test hook: DSV4_TEST_LZFAIL=N, every growth past N positions fails as if the VRAM were taken
    static const int test_fail = getenv("DSV4_TEST_LZFAIL") ? atoi(getenv("DSV4_TEST_LZFAIL")) : 0;
    if (test_fail > 0 && rp > test_fail) {
        logf_("WARNING context: no VRAM for the K/V of %d positions (DSV4_TEST_LZFAIL): the request fails\n", rp);
        return false;
    }
    bool ok = true;
    const auto t0 = clk::now();
    seq_quiesce(E);
    kv_dev_sync(E);   // pending admission copies into the zones are done
    auto tgt = [&](int j) {
        return j < 8 ? lz_up((size_t)tp[j & 3] * lz_pgb(j)) : lz_up((size_t)rp * 32 * 4);
    };
    int nev = 0, nfresh = 0, nlent = 0;
    const size_t ev0 = E->lz_ev.size();
    for (size_t k = 0; k < E->cards.size(); k++) {   // evict rounds until the zones hold this card's chunks
        Card& c = E->cards[k];
        size_t need = 0;
        for (int j = 0; j < 12; j++) {
            const size_t t = std::min(c.lzb[j].cap, tgt(j));
            if (t > c.lzb[j].mapped) need += (t - c.lzb[j].mapped) / LZG;
        }
        auto avail = [&] {
            size_t n = 0;
            for (int l = 0; l < NL; l++)
                if (c.lz_nmax[l]) {
                    int used = 0;
                    for (int x = c.lz_c0[l]; x < c.lz_c1[l]; x++) used += c.lzc[x].user >= 0;
                    n += (size_t)std::max(0, lz_avail(E, c, l) - used);
                }
            return n;
        };
        while (avail() < need) {
            bool step = false;
            for (int l = 0; l < NL; l++) {
                if (c.lz_n[l] >= c.lz_nmax[l]) continue;
                if (lz_evict_slot(E, (int)k, l, c.pnl[l] - 1 - c.lz_n[l])) {
                    c.lz_n[l]++;
                    nev++;
                    step = true;
                } else {
                    c.lz_nmax[l] = c.lz_n[l];   // a static expert: the zone ends here
                }
            }
            if (!step) break;
        }
    }
    kv_dev_sync(E);   // the table updates ran before the slots' bytes leave the pool
    for (auto& c : E->cards) {
        DevGuard dg(c.dev);
        for (int j = 0; j < 12; j++) {
            Card::LzBuf& b = c.lzb[j];
            const size_t t = std::min(b.cap, tgt(j)), m0 = b.mapped;
            while (b.mapped < t) {
                int x = -1;
                for (int l = 0; l < NL && x < 0; l++) {
                    const int av = lz_avail(E, c, l);
                    for (int y = c.lz_c0[l]; y < c.lz_c0[l] + av && y < c.lz_c1[l]; y++)
                        if (c.lzc[y].user < 0) { x = y; break; }
                }
                if (x >= 0) {
                    Card::LzChunk& ch = c.lzc[x];
                    const CUdeviceptr pv = (CUdeviceptr)(uintptr_t)c.dpool + ch.off;
                    if (lzd().unmap(pv, LZG) != CUDA_SUCCESS) {   // (the chunk stays in the pool)
                        cudaGetLastError();
                        logf_("WARNING card %d: context loan: unmap from the pool failed: the growth fails\n", c.dev);
                        ok = false;
                        break;
                    }
                    if (!lz_map_lg(c, b.va + b.mapped, LZG, ch.h)) {
                        cudaGetLastError();
                        const bool back = lz_map(c, pv, LZG, ch.h);   // back where it was (its slots stay lent until the give-back)
                        if (!back) ch.user = kLzLost;
                        logf_("WARNING card %d: context loan: map failed: the growth fails%s\n", c.dev,
                              back ? "" : " (the chunk is lost: its pool slots stay out of the tier)");
                        ok = false;
                        break;
                    }
                    ch.user = j;
                    b.ch.push_back(x);
                    nlent++;
                } else {   // no zone chunk left: own memory (a card without zones, or a short pool)
                    unsigned long long h = 0;
                    if (!lz_mk(c, LZG, &h)) { ok = false; break; }
                    if (!lz_map(c, b.va + b.mapped, LZG, h)) {
                        lzd().release((CUmemGenericAllocationHandle)h);
                        ok = false;
                        break;
                    }
                    b.fresh.push_back(h);
                    b.ch.push_back(-(int)b.fresh.size());
                    nfresh++;
                    E->lz_cnt[3]++;
                }
                b.mapped += LZG;
            }
            if (j < 8 && b.mapped > m0 && (lz_zero() & 1))   // review rel7 M1: lent chunks hold expert bytes
                DCK(cudaMemset((void*)(uintptr_t)(b.va + m0), 0, b.mapped - m0));
            if (j >= 8 && b.mapped > m0) {   // the rope values of the new positions
                const size_t hb = std::min(b.mapped, E->lz_rope[j - 8].size() * 4);
                if (hb > m0)
                    DCK(cudaMemcpy((void*)(uintptr_t)(b.va + m0), (const uint8_t*)E->lz_rope[j - 8].data() + m0, hb - m0, cudaMemcpyHostToDevice));
            }
        }
    }
    lz_sync_counts(E);
    E->lz_cnt[0]++;
    if (!ok)
        logf_("WARNING context: no VRAM for the K/V of %d positions on a card without loan room: the request fails "
              "(a smaller --max-ctx avoids it)\n", rp);
    logf_("context: buffers grown to %d positions (K/V pages %d / %d / %d / %d): %d pool slots evicted, %d chunks lent, %d "
          "own (%.1f ms; %zu experts out of VRAM)\n", std::min(E->ropeM, 2 * (E->kvM[0] - (E->kvP[0] - E->kvcap[0])) * KVPG),
          E->kvM[0], E->kvM[1], E->kvM[2], E->kvM[3], nev, nlent, nfresh, ms_since(t0), E->lz_ev.size() - ev0);
    return ok;
}
// the loan back to the pool: the parked runs above the base move below it (or to host RAM), the chunks go back to
// their zones and the evicted experts with a host copy back into their slots (else the slot is free). Not while the
// active run lies above the base; unless force, only DSV4_CTX_HOLD_S (30 s) after the last long sequence was active.
// [dsv4-rel8-giveback] (rel7 review L5) restore false: the evicted experts are not copied back now (their slots are
// free; admissions may refill them): the give-back inside a fresh request's kv_reserve skips that synchronous H2D work
// (hundreds of experts at 262K). lz_ev stays as the to-restore list (review rel8 L1): the server's idle tick (ctx_idle)
// restores what is still missing. --tier-reset (lz_seed, review rel8 M1): the freed slots get their seeds at once, so
// the mode's state does not depend on an earlier long request
// something is mapped above a buffer's base (review kb M2: per buffer, not kvM / ropeM, which are minimums over the
// buffers: a growth that failed half-way leaves some buffers grown while the minimums stay at the base)
// [kb-dsv4-loanmap] the zone slots of layer l a give-back must keep out of the tier: from the lowest one a chunk not in
// the pool (lost, or still lent) lies over, to the top (the zone stays one top range)
static int lz_keep_n(const Engine* E, const Card& c, int l) {
    const size_t xb = expert_bytes(pool_fmt(E, l));
    for (int q = c.pnl[l] - c.lz_n[l]; q < c.pnl[l]; q++) {
        const size_t a0 = c.ploff[l] + (size_t)q * xb, a1 = a0 + xb;
        for (int x = c.lz_c0[l]; x < c.lz_c1[l]; x++)
            if (c.lzc[x].user >= 0 && c.lzc[x].off < a1 && c.lzc[x].off + LZG > a0) return c.pnl[l] - q;
    }
    return 0;
}
static bool lz_lent(const Engine* E) {
    bool lent = E->ropeM > E->ropeB;
    for (int si = 0; si < 4; si++) lent |= E->kvM[si] > E->kvB[si];
    for (auto& c : E->cards) {
        for (auto& b : c.lzb) lent |= b.cap && b.mapped > b.base && !b.ch.empty();
        // re-review 3 L3: zone slots evicted for a growth whose maps all failed (no buffer grew): they go back too
        for (int l = 0; l < NL && !lent; l++) lent |= c.lz_n[l] > 0 && c.lz_n[l] > lz_keep_n(E, c, l);
    }
    return lent;
}
static void lz_shrink(Engine* E, bool force, bool restore) {
    if (!E->lz) return;
    if (!lz_lent(E)) return;
    for (int si = 0; si < 4; si++)
        if (E->ares[si] && E->ab[si] + E->ares[si] > E->kvB[si]) return;
    if (!force && std::chrono::duration<double>(clk::now() - E->lz_t_long).count() < E_lz_hold) return;
    const auto t0 = clk::now();
    seq_quiesce(E);
    for (int si = 0; si < 4; si++)
        for (int p = E->kvB[si]; p < E->kvM[si]; p++) {
            const int o = E->kvown[si][p];
            if (o > 0 && !E->kvres.count(o)) E->kvown[si][p] = 0;   // review rel7 M5: a stale owner
            else if (o > 0) kv_move_out(E, o, si, E->kvB[si], E->kvP[si]);   // below the base, else host RAM
        }
    for (int si = 0; si < 4; si++)
        for (int p = E->kvB[si]; p < E->kvP[si]; p++)
            if (E->kvown[si][p] != 0) {   // review rel7 M5: not given back this time (nothing changed yet)
                logf_("context give-back skipped: page %d of source %d still in use\n", p, si);
                return;
            }
    kv_dev_sync(E);
    int nret = 0;
    for (auto& c : E->cards) {
        DevGuard dg(c.dev);
        for (auto& b : c.lzb) {
            while (b.mapped > b.base && !b.ch.empty()) {   // (review rel7 M2: a load-time full mapping is base)
                const int x = b.ch.back();
                b.ch.pop_back();
                b.mapped -= LZG;
                if (lzd().unmap(b.va + b.mapped, LZG) != CUDA_SUCCESS) {   // it stays in the buffer (lent: a later give-back)
                    cudaGetLastError();
                    b.ch.push_back(x);
                    b.mapped += LZG;
                    logf_("WARNING card %d: context give-back: unmap failed: %zu bytes stay above the base\n", c.dev, b.mapped - b.base);
                    break;
                }
                if (x >= 0) {
                    Card::LzChunk& ch = c.lzc[x];
                    if (lz_map_lg(c, (unsigned long long)(uintptr_t)c.dpool + ch.off, LZG, ch.h)) {
                        ch.user = -1;
                        nret++;
                    } else if (cudaGetLastError(), lz_map(c, b.va + b.mapped, LZG, ch.h)) {   // back into the buffer
                        b.ch.push_back(x);
                        b.mapped += LZG;
                        logf_("WARNING card %d: context give-back: map into the pool failed: the chunk stays lent\n", c.dev);
                        break;
                    } else {
                        cudaGetLastError();
                        ch.user = kLzLost;
                        logf_("WARNING card %d: context give-back: a chunk could be mapped neither into the pool nor back: it "
                              "is lost (its pool slots stay out of the tier)\n", c.dev);
                    }
                } else {
                    lzd().release((CUmemGenericAllocationHandle)b.fresh[-1 - x]);
                }
            }
            if (b.ch.empty()) b.fresh.clear();
        }
    }
    lz_sync_counts(E);
    // the slots back to the tier: lent marks cleared, the evicted experts with an exact host copy back where they were
    std::vector<std::pair<int, int>> freed;   // (layer, pool index) of the slots given back
    for (size_t k = 0; k < E->cards.size(); k++) {
        Card& c = E->cards[k];
        for (int l = 0; l < NL; l++) {
            Layer& W = E->L[l];
            // [kb-dsv4-loanmap] a chunk not in the pool (lost, or still lent): the zone keeps every slot from the lowest one
            // over it to the top lent (the zone stays one top range: lz_avail / the next loan count from the top)
            const int keep_n = lz_keep_n(E, c, l);
            for (int q = c.pnl[l] - c.lz_n[l]; q < c.pnl[l] - keep_n; q++) {
                const int g = (int)k * GS + c.ps0[l] + q;
                for (size_t i = 0; i < W.dpool.size(); i++)
                    if (W.dpool[i] == g && W.pend_e[i] == -4) W.pend_e[i] = -1, freed.push_back({l, (int)i});
            }
            c.lz_n[l] = keep_n;
        }
    }
    if (!E->lz_ev_lent.empty() && !freed.empty()) {   // review L9: the slots kept lent before and freed now get theirs back
        std::vector<std::tuple<int, int, int>> still;
        for (const auto& t : E->lz_ev_lent) {
            const auto [l, g, e] = t;
            bool fr = false;
            for (auto [fl, fi] : freed) fr = fr || (fl == l && E->L[l].dpool[fi] == g);
            (fr ? E->lz_ev : still).push_back(t);
        }
        E->lz_ev_lent.swap(still);
    }
    size_t nback = 0;
    if (E->lz_seed) {   // [dsv4-rel8-giveback] review rel8 M1: the seeds, not the eviction-time experts
        nback = lz_seed_fill(E, freed);
        E->lz_ev.clear();
    } else if (restore) {
        nback = lz_restore(E);
    }
    E->lz_cnt[1]++;
    logf_("context: buffers back to the base (%d positions): %d chunks returned to the pool, %zu experts back%s (%.1f ms)\n",
          E->ropeB, nret, nback, E->lz_seed ? " (seeds)" : restore ? "" : " (in a request: the restore waits for an idle tick)",
          ms_since(t0));
}
// [dsv4-rel8-giveback] the experts evicted for a loan (lz_ev) back into their slots, from their exact host copies: only
// slots still free, experts neither resident nor pending (admissions may have refilled them since). Returns how many
static size_t lz_restore(Engine* E) {
    std::vector<std::tuple<int, int, int>> back;   // (layer, pool index, expert)
    std::set<long> done;   // review rel7 L1: an expert evicted twice comes back once
    std::set<long> taken;  // a slot evicted twice (refilled in between) takes one expert back (its dyn_e is set below)
    for (auto [l, g, e] : E->lz_ev) {
        Layer& W = E->L[l];
        if (W.h_dslot[e] >= 0 || !W.hx[e] || W.hfmt[e] != pool_fmt(E, l) || done.count((long)l * NE + e)) continue;
        bool pend = false;
        for (int pe : W.pend_e) pend |= pe == e;
        if (pend) continue;
        for (size_t i = 0; i < W.dpool.size(); i++)
            if (W.dpool[i] == g && W.dyn_e[i] < 0 && W.pend_e[i] == -1 && taken.insert((long)l * (1L << 20) + (long)i).second) {
                Card& sc = E->cards[gs_card(g)];
                DevGuard dg(sc.dev);
                DCK(cudaMemcpyAsync(pool_slot_ptr(E, sc, l, g), W.hx[e], expert_bytes(pool_fmt(E, l)), cudaMemcpyHostToDevice, sc.cp));
                back.push_back({l, (int)i, e});
                done.insert((long)l * NE + e);   // review rel re-review L1: only once a slot took it (another loan's entry may still)
                break;
            }
    }
    kv_dev_sync(E);   // the bytes are in before the tables point at them
    for (auto [l, i, e] : back) {
        Layer& W = E->L[l];
        set_pslot(E, W, e, W.dpool[i]);
        if (!W.inpool.empty()) W.inpool[e] = 1;
        W.dyn_e[i] = e;
    }
    kv_dev_sync(E);
    // review L9: the entries of slots kept lent (a lost or still-lent chunk, keep_n) wait for the give-back that frees them
    std::vector<std::tuple<int, int, int>> rest;
    for (auto [l, g, e] : E->lz_ev) {
        const Layer& W = E->L[l];
        if (done.count((long)l * NE + e) || W.h_dslot[e] >= 0) continue;
        bool lent = false;
        for (size_t i = 0; i < W.dpool.size() && !lent; i++) lent = W.dpool[i] == g && W.pend_e[i] == -4;
        if (lent) rest.push_back({l, g, e});
    }
    E->lz_ev_lent.insert(E->lz_ev_lent.end(), rest.begin(), rest.end());
    E->lz_ev.clear();
    return back.size();
}
// [dsv4-rel8-giveback] review rel8 M1 (--tier-reset): the given-back slots get their seed experts (tier_reset's pass 2
// for those slots only): free slots whose seed is neither resident elsewhere nor pending. Returns how many
static size_t lz_seed_fill(Engine* E, const std::vector<std::pair<int, int>>& freed) {
    std::vector<std::pair<int, int>> back;   // (layer, pool index)
    std::vector<uint8_t> tmp;
    for (auto [l, i] : freed) {
        Layer& W = E->L[l];
        if ((size_t)i >= W.seed_e.size()) continue;   // (no tier snapshot in this configuration)
        const int e = W.seed_e[i];
        if (e < 0 || W.dyn_e[i] >= 0 || W.pend_e[i] != -1 || W.h_dslot[e] >= 0) continue;
        bool pend = false;
        for (int pe : W.pend_e) pend |= pe == e;
        if (pend) continue;
        const int g = W.dpool[i];
        Card& sc = E->cards[gs_card(g)];
        DevGuard dg(sc.dev);
        const size_t XB1 = expert_bytes(pool_fmt(E, l));
        if (W.hx[e]) DCK(cudaMemcpyAsync(pool_slot_ptr(E, sc, l, g), W.hx[e], XB1, cudaMemcpyHostToDevice, sc.cp));
        else {   // a copy-less seed (as tier_reset)
            tmp.resize(XB1);
            gg_repack_expert(E, l, e, tmp.data());
            DCK(cudaMemcpy(pool_slot_ptr(E, sc, l, g), tmp.data(), XB1, cudaMemcpyHostToDevice));
        }
        back.push_back({l, i});
    }
    kv_dev_sync(E);
    for (auto [l, i] : back) {
        Layer& W = E->L[l];
        set_pslot(E, W, W.seed_e[i], W.dpool[i]);
        if (!W.inpool.empty()) W.inpool[W.seed_e[i]] = 1;
        W.dyn_e[i] = W.seed_e[i];
    }
    kv_dev_sync(E);
    return back.size();
}
int engine_max_ctx(Engine* E) { return E->o.maxctx; }
void set_lz_seed(Engine* E, bool on) { E->lz_seed = on; }
// the held (active) run lies above the base context, so a give-back must wait for it to leave (park); hold = also only
// once DSV4_CTX_HOLD_S passed since the last long step (rel8 open item, dsv4_rel_review re-review M1 long-after-longer)
bool lz_held_above(Engine* E, bool hold) {
    if (!E->lz || !lz_lent(E)) return false;
    // review rel9 L1: the tier reset gives nothing back in these configurations (tier_reset's early return)
    if (!hold && (E->gswap || E->early || E->seed_card_nv.empty())) return false;
    if (hold && std::chrono::duration<double>(clk::now() - E->lz_t_long).count() < E_lz_hold) return false;
    for (int si = 0; si < 4; si++)
        if (E->ares[si] && E->ab[si] + E->ares[si] > E->kvB[si]) return true;
    return false;
}
// [dsv4-rel8-giveback] between requests (the server calls it about once a second while nothing is in service or
// queued): a stale loan goes back to the pool here, with the experts restored, instead of inside the next short
// request (rel7 L5). Same rules as the in-request give-back: not while the active run lies above the base,
// DSV4_CTX_HOLD_S after the last long sequence. Once nothing is lent, the experts an in-request give-back left out
// come back (review rel8 L1). Returns 1 if the buffers are at their base afterwards
int ctx_idle(Engine* E) {
    if (!E->lz) return 1;
    lz_shrink(E, false, true);
    const bool lent = lz_lent(E);
    if (!lent && !E->lz_ev.empty() && !E->lz_seed) {
        const auto t0 = clk::now();
        seq_quiesce(E);
        const size_t n = lz_restore(E);
        logf_("context: %zu experts back after an in-request give-back (idle, %.1f ms)\n", n, ms_since(t0));
    }
    return lent ? 0 : 1;
}
// pages [b, b + n) of source si for the active sequence (parked runs there move away or leave VRAM first; the
// active one may grow to [ab, ab + kvcap))
static void kv_claim(Engine* E, int si, int b, int n) {
    for (int p = b; p < b + n; p++) {
        const int o = E->kvown[si][p];
        if (o > 0) {
            seq_quiesce(E);   // no capture or fetch ahead running while the copy syncs the devices (rare: an eviction)
            kv_move_out(E, o, si, E->ab[si], E->ab[si] + E->kvcap[si]);
        }
        E->kvown[si][p] = -1;
    }
}
// the active sequence holds the pages its rows up to rows[si] need (a fresh one first gets its base)
static void kv_reserve_rows(Engine* E, const int rows[4]) {
    bool rebind = false;
    if (E->lz) {   // [dsv4-ctx-lazy] a fresh short sequence gives a stale loan back; the pages it will hold are mapped
        bool fresh = true, fits = rows[3] <= E->ropeB;
        int top[4];
        for (int si = 0; si < 4; si++) {
            const int need = std::min(E->kvcap[si], (std::max(rows[si], 1) + KVPG - 1) / KVPG);
            fresh &= E->ares[si] == 0;
            fits &= need <= E->kvB[si] - (E->kvP[si] - E->kvcap[si]);
        }
        if (fresh && fits) lz_shrink(E, E->lz_seed, false);   // [dsv4-rel8-giveback] no synchronous expert restore here; --tier-reset: no hold (re-review M1)
        for (int si = 0; si < 4; si++) {
            const int need = std::min(E->kvcap[si], (std::max(rows[si], 1) + KVPG - 1) / KVPG);
            top[si] = (E->ares[si] ? E->ab[si] : kv_pick(E, si, 0)) + need;
        }
        if (!lz_ensure(E, top, rows[3]))   // [kb-dsv4-ctxgrow] the request fails, the engine goes on (was a DCHECK abort)
            throw CtxNoRoom("context of " + std::to_string(rows[3]) + " positions: no VRAM for its K/V (a smaller --max-ctx avoids it)");
    }
    for (int si = 0; si < 4; si++) {
        const int need = std::min(E->kvcap[si], (std::max(rows[si], 1) + KVPG - 1) / KVPG);
        if (need <= E->ares[si]) continue;
        if (E->ares[si] == 0) {
            const int b = kv_pick(E, si, 0);
            rebind |= b != E->ab[si];
            E->ab[si] = b;
        }
        kv_claim(E, si, E->ab[si] + E->ares[si], need - E->ares[si]);
        E->ares[si] = need;
    }
    if (rebind) {   // a new base (a fresh sequence only): captured step graphs hold the old views
        kv_bind(E);
        E->nfwd++;
    }
    for (int si = 0; si < 4 && E->lz; si++)
        if (E->ab[si] + E->ares[si] > E->kvB[si]) E->lz_t_long = clk::now();
}
// rows of every source up to position pos_end (+ a margin of 1,024 positions, so captures ahead stay covered)
static void kv_reserve(Engine* E, int pos_end) {
    pos_end = std::min(E->o.maxctx, pos_end + 1024);
    const int rows[4] = {pos_end / 2 + 1, pos_end / 2 + 1, pos_end / 2 + 1, pos_end};
    kv_reserve_rows(E, rows);
}
// the active sequence gives its pages back (reset)
static void kv_release(Engine* E) {
    for (int si = 0; si < 4; si++) {
        for (int p = E->ab[si]; p < E->ab[si] + E->ares[si]; p++)
            if (E->kvown[si][p] == -1) E->kvown[si][p] = 0;
        E->ares[si] = 0;
    }
}

void reset(Engine* E) {
    if (E->eg_pre) {   // R20 [dsv4-r20-egahead]: a batch fetched for a step that will not come
        E->egr.wait();
        E->eg_pre = false;
    }
    E->cap_cancel.store(true, std::memory_order_release);   // review r19 L2: cancel a pending capture, do not finish it
    cap_wait(E);
    seq_stats_flush(E);
    E->nfwd++;
    E->pos = 0;
    E->cone_min = 0;   // [dsv4-pf15-cone]
    E->xcone_prev = false;   // [dsv4-xcall]
    E->xpend_until = 0;
    E->have_draft = false;
    E->mh_n = 0;
    // R8 adaptive DSpark: each sequence measures again. R19 [dsv4-r19-specstart]: start with passes; the plain rate
    // is the prior from earlier sequences, else measured by 8 plain steps after the first 8 passes
    E->ds_plain = E_dsplainstart;
    E->ds_probe = false;
    E->lk_hold = 0;
    E->ds_cnt = 0;
    E->ds_ms_spec = 0;
    E->ds_wms = E->ds_wtok = 0;
    E->ds_lam = 0;   // R20: tokens / ms per sequence (T[n] stays: the engine's own cost table)
    E->ds_ms_plain = E_dsplainstart ? 0 : E->ds_prior;
    E->toks.clear();
    for (int s = 0; s < 4; s++) E->tail_n[s] = E->tc[s] = 0;
    kv_release(E);   // [dsv4-rel4-p2]
    E->snap_req = E->snap_pos = -1;   // [dsv4-srv-pesnap]
    E->sv_pass = -1;
}

// ---------------------------------------------------------------- forward pieces

static void dump(Engine* E, Card& c, int l, const float* h, int n) {
    if (E->o.dump.empty()) return;
    std::vector<float> b((size_t)n * HC * D);
    DCK(cudaStreamSynchronize(c.st));
    DCK(cudaMemcpy(b.data(), h, b.size() * 4, cudaMemcpyDeviceToHost));
    char nm[512];
    if (E->dump_step < 0) snprintf(nm, sizeof nm, "%s/l%02d_p.f32", E->o.dump.c_str(), l);
    else snprintf(nm, sizeof nm, "%s/l%02d_d%03d.f32", E->o.dump.c_str(), l, E->dump_step);
    FILE* f = fopen(nm, "wb");
    fwrite(b.data(), 4, b.size(), f);
    fclose(f);
}

// mHC mixes of h [n] with (fn, scale, base) -> pre, post, comb
static void hc_mixes(Card& c, const Layer& W, int which, const float* h, int n, float* pre, float* post, float* comb) {
    if (n <= MAXR) {
        k_hc_dot<<<dim3(NMIX + 1, n), 256, 0, c.st>>>(h, W.hc_fn[which], c.mix, c.ss);
    } else {
        sgemm(c.blas, h, HC * D, W.hc_fn[which], HC * D, c.mix, NMIX, n, NMIX, HC * D);
        k_sumsq<<<n, 256, 0, c.st>>>(h, HC * D, c.ss);
    }
    k_hc_finish<<<cdiv((size_t)n * 32, 256), 256, 0, c.st>>>(c.mix, c.ss, W.hc_scale[which], W.hc_base[which], n, pre,
                                                               post, comb);
}

// [dsv4-dec5-egz] decode Engram rows read in place by the decode kernel from a mapped staging buffer per Engram layer
// (no H2D copy behind the admissions, no layer-14 wait for layer 1's copy); DSV4_NOEGZ=1 = the copy path
static const bool g_egz = getenv("DSV4_NOEGZ") == nullptr;
// [dsv4-dec5-egprof] decode Engram batches, times after the submit (g_egprof above)
static long long g_egp_sub = 0;
static int g_egp_rows = 0;
static long g_egp_n = 0;
static double g_egp[7][5] = {};
static double g_egq[2][4] = {};
// engram for the tokens at positions s .. s + n - 1 (toks holds them): h += gate * value
static void engram(Engine* E, Card& c, const Layer& W, int l, int s, int n) {
    const int li = l == 1 ? 0 : 1;
    if (tl_cap) {   // R10 step graph (one row, NVMe rows): the decode thread stages the rows after the launch and raises
                    // egflag[li]; the graph waits for it, copies the 24 rows from the pinned staging and decodes them
        DCHECK(E->eg_nvme && n == 1, "graph engram needs NVMe rows and one row");
        float* emb = (float*)c.phase;
        float* kv = emb + (size_t)1024 * EG_ROWS * EG_HD;
        k_hwait<<<1, 1, 0, c.st>>>(E->d_egflag + li * 16, kseq(E));
        DCK(cudaMemcpyAsync(c.d_egrows, E->h_egrows + (size_t)li * 24 * 264, (size_t)24 * 264, cudaMemcpyHostToDevice, c.st));
        if (E->eg_q6k) k_eg_decode_q6k<<<24, 256, 0, c.st>>>(c.d_egrows, 24, emb);
        else k_eg_decode_fp8<<<24, 256, 0, c.st>>>(c.d_egrows, 24, emb);
        linq(c, W.e_wkv, emb, EG_DIM, 1, kv, (HC + 1) * D);
        k_eg_gate<<<dim3(1, HC), 256, 0, c.st>>>(kv, W.e_qk, c.h);
        return;
    }
    if (E->eg_nvme) {   // lossless rows fetched from NVMe (prefetched at the start of the forward)
        DCHECK(E->eg_batch_s + tl_row0 == s && tl_row0 + n <= E->eg_batch_n,
               "engram batch mismatch");   // (R19: a pipelined pass's micro-batches are row ranges of the batch)
        auto tw = clk::now();
        if (g_egwait_all) E->egr.wait();
        else E->egr.wait(li);   // [dsv4-pf-egwait]: this layer's rows only (the batch's rows of the other layer may still be in flight)
        g_eg_wait += ms_since(tw);
        if (g_egprof && n <= MAXR && li == 0 && g_egp_sub) {   // [dsv4-dec5-egprof]: submit -> layer-1 rows done, arrival at layer 1
            const long long now = std::chrono::duration_cast<std::chrono::nanoseconds>(clk::now().time_since_epoch()).count();
            const int R = std::min(n, 6);
            g_egp[R][0] += 1; g_egp[R][1] += (E->egr.tl_ns[0].load() - g_egp_sub) / 1e6;
            g_egp[R][2] += (std::chrono::duration_cast<std::chrono::nanoseconds>(tw.time_since_epoch()).count() - g_egp_sub) / 1e6;
            g_egp[R][3] += (now - g_egp_sub) / 1e6; g_egp[R][4] += g_egp_rows;
            if (++g_egp_n % 64 == 0) {
                std::string o;
                char b[160];
                for (int r = 1; r <= 6; r++) if (g_egp[r][0] > 0) {
                    const double k = g_egp[r][0];
                    snprintf(b, sizeof b, " R%d n %.0f rows %.0f: L1 rows done %.3f, host at layer 1 %.3f, wait end %.3f;", r, k, g_egp[r][4] / k, g_egp[r][1] / k, g_egp[r][2] / k, g_egp[r][3] / k);
                    o += b;
                }
                logf_("EGPROF (ms after submit, mean):%s\n", o.c_str());
                for (int q = 0; q < 2; q++)
                    logf_("EGPROF layer %d host ms per call: wait + evsync %.3f, of it evsync %.3f, stage + H2D enqueue %.3f (%.0f calls)\n",
                          q ? 14 : 1, g_egq[q][0] / std::max(1.0, g_egq[q][3]), g_egq[q][1] / std::max(1.0, g_egq[q][3]),
                          g_egq[q][2] / std::max(1.0, g_egq[q][3]), g_egq[q][3]);
            }
        }
        const int SUB = 1024;
        float* emb = (float*)c.phase;
        float* kv = emb + (size_t)SUB * EG_ROWS * EG_HD;
        if (g_egz && E->eg_q6k && n <= MAXR && E->h_egdz) {   // [dsv4-dec5-egz] decode: own mapped staging per Engram layer
            cudaEvent_t ev = li ? c.ev_eg1 : c.ev_eg;
            uint8_t* stg = E->h_egdz + (size_t)li * MAXR * 24 * 264;
            DCK(cudaEventSynchronize(ev));   // the previous pass's read of this layer's staging
            for (int i = 0; i < n; i++)
                for (int j = 0; j < 24; j++)
                    memcpy(stg + ((size_t)i * 24 + j) * 264,
                           E->h_egraw + (size_t)E->eg_map[((size_t)(tl_row0 + i) * 2 + li) * 24 + j] * 264, 264);
            k_eg_decode_q6k_h<<<n * 24, 256, 0, c.st>>>(stg, n * 24, emb);
            DCK(cudaEventRecord(ev, c.st));
            linq(c, W.e_wkv, emb, EG_DIM, n, kv, (HC + 1) * D);
            k_eg_gate<<<dim3(n, HC), 256, 0, c.st>>>(kv, W.e_qk, c.h);
            return;
        }
        for (int i0 = 0; i0 < n; i0 += SUB) {
            int m = std::min(SUB, n - i0);
            const auto tev = g_egprof ? clk::now() : clk::time_point{};
            // pinned staging reuse: the previous copy out of it (never the stream: in run-ahead decode the stream waits on
            // the host); review dec5 M1: the copy may sit on another card's stream (layers 1 and 14 on different cards)
            for (auto& k : E->cards) DCK(cudaEventSynchronize(k.ev_eg));
            if (g_egprof && n <= MAXR) g_egq[li][0] += ms_since(tw), g_egq[li][1] += ms_since(tev), g_egq[li][3] += 1;
            const auto tcp = g_egprof ? clk::now() : clk::time_point{};
            for (int i = 0; i < m; i++)
                for (int j = 0; j < 24; j++)
                    memcpy(E->h_egrows + ((size_t)i * 24 + j) * 264,
                           E->h_egraw + (size_t)E->eg_map[((size_t)(tl_row0 + i0 + i) * 2 + li) * 24 + j] * 264, 264);
            DCK(cudaMemcpyAsync(c.d_egrows, E->h_egrows, (size_t)m * 24 * 264, cudaMemcpyHostToDevice, c.st));
            DCK(cudaEventRecord(c.ev_eg, c.st));
            if (g_egprof && n <= MAXR) g_egq[li][2] += ms_since(tcp);
            if (E->eg_q6k) k_eg_decode_q6k<<<m * 24, 256, 0, c.st>>>(c.d_egrows, m * 24, emb);
            else k_eg_decode_fp8<<<m * 24, 256, 0, c.st>>>(c.d_egrows, m * 24, emb);
            linq(c, W.e_wkv, emb, EG_DIM, m, kv, (HC + 1) * D);
            k_eg_gate<<<dim3(m, HC), 256, 0, c.st>>>(kv, W.e_qk, c.h + (size_t)i0 * HC * D);
        }
        return;
    }
    const EgTable& T = E->eg[li];
    const int SUB = 1024;
    float* emb = (float*)c.phase;
    float* kv = emb + (size_t)SUB * EG_ROWS * EG_HD;
    for (int i0 = 0; i0 < n; i0 += SUB) {
        int m = std::min(SUB, n - i0);
        // gather rows on the host (previous copy out of the pinned buffer finished; review dec5 M1: on any card)
        for (auto& k : E->cards) DCK(cudaEventSynchronize(k.ev_eg));
        size_t off = 0;
        int64_t r24[24];
        for (int i = 0; i < m; i++) {
            eg_hash(E, E->toks, s + i0 + i, li, r24);
            for (int j = 0; j < 24; j++) {
                int64_t r = r24[j];
                const uint8_t* src;
                int bits;
                size_t rb;
                auto it = T.hot_bits ? std::lower_bound(T.hot_ids.begin(), T.hot_ids.end(), r) : T.hot_ids.end();
                if (T.hot_bits && it != T.hot_ids.end() && *it == r) {
                    src = T.hot + (size_t)(it - T.hot_ids.begin()) * T.hot_rowbytes;
                    bits = T.hot_bits;
                    rb = T.hot_rowbytes;
                } else {
                    DCHECK(r >= 0 && (size_t)r < T.nrows, "engram row %lld", (long long)r);
                    src = T.base + (size_t)r * T.rowbytes;
                    bits = T.bits;
                    rb = T.rowbytes;
                }
                memcpy(E->h_egrows + off, src, rb);
                E->h_egoff[i * 24 + j] = (int)off;
                E->h_egbits[i * 24 + j] = (uint8_t)bits;
                off += (rb + 7) & ~(size_t)7;
            }
        }
        DCK(cudaMemcpyAsync(c.d_egrows, E->h_egrows, off, cudaMemcpyHostToDevice, c.st));
        DCK(cudaMemcpyAsync(c.d_egoff, E->h_egoff, (size_t)m * 24 * 4, cudaMemcpyHostToDevice, c.st));
        DCK(cudaMemcpyAsync(c.d_egbits, E->h_egbits, (size_t)m * 24, cudaMemcpyHostToDevice, c.st));
        DCK(cudaEventRecord(c.ev_eg, c.st));
        k_eg_decode<<<m * 24, 256, 0, c.st>>>(c.d_egrows, c.d_egoff, c.d_egbits, m * 24, c.d_cb + 2 * 16,
                                               c.d_cb + 3 * 16, c.d_cb + 4 * 16, emb);
        linq(c, W.e_wkv, emb, EG_DIM, m, kv, (HC + 1) * D);
        k_eg_gate<<<dim3(m, HC), 256, 0, c.st>>>(kv, W.e_qk, c.h + (size_t)i0 * HC * D);
    }
}

// compressor (kv-source layer l) on x [n] at positions s..: appends compressed rows + index keys to the source cache
// on this card and broadcasts them to the other cards
static void compressor(Engine* E, Card& c, const Layer& W, int l, int s, int n) {
    const int si = src_index(l), r = W.ratio;
    float* kvb = c.kvt;                          // [n + 1][512]
    float* scb = c.kvt + (size_t)(E->o.chunk + 1) * HD;
    int off = r == 2 ? E->tail_n[si] : 0;
    if (W.cq_wkv.sq) linq(c, W.cq_wkv, c.x, D, n, kvb + (size_t)off * HD, HD);   // [dsv4-rel-q8]
    else lin32(c, W.c_wkv, HD, D, c.x, D, n, kvb + (size_t)off * HD, HD);
    if (r == 2) {
        if (W.cq_wgate.sq) linq(c, W.cq_wgate, c.x, D, n, scb + (size_t)off * HD, HD);
        else lin32(c, W.c_wgate, HD, D, c.x, D, n, scb + (size_t)off * HD, HD);
        if (off) {
            DCK(cudaMemcpyAsync(kvb, c.tail_kv[si], HD * 4, cudaMemcpyDeviceToDevice, c.st));
            DCK(cudaMemcpyAsync(scb, c.tail_sc[si], HD * 4, cudaMemcpyDeviceToDevice, c.st));
        }
    }
    int m = n + off;
    // [dsv4-srv-pesnap] the snapshot position P is odd: its tail row is the input row of position P - 1 (rows here are
    // positions s - off ..); eager prefill only (snap_req is set inside prefill, captures are cancelled there)
    if (E->snap_req >= 0 && r == 2 && !tl_cap && (E->snap_req & 1)) {
        const int q = E->snap_req - 1 - (s - off);
        if (q >= 0 && q < m) {
            DCK(cudaMemcpyAsync(c.snap_kv[si], kvb + (size_t)q * HD, HD * 4, cudaMemcpyDeviceToDevice, c.st));
            DCK(cudaMemcpyAsync(c.snap_sc[si], scb + (size_t)q * HD, HD * 4, cudaMemcpyDeviceToDevice, c.st));
            E->snap_got[si] = 1;
        }
    }
    if (E->dspark && r == 2 && n <= MAXR && tl_mb == 1) {   // R19: the second micro-batch appends its rows
        const int o2 = s - E->sv_base[si];
        DCHECK(o2 >= 0 && o2 + n <= MAXR + 1, "pipelined pass: compressor rows %d + %d", o2, n);
        DCK(cudaMemcpyAsync(c.sv_kv[si] + (size_t)o2 * HD, kvb + (size_t)off * HD, (size_t)n * HD * 4, cudaMemcpyDeviceToDevice, c.st));
        DCK(cudaMemcpyAsync(c.sv_sc[si] + (size_t)o2 * HD, scb + (size_t)off * HD, (size_t)n * HD * 4, cudaMemcpyDeviceToDevice, c.st));
        E->sv_m[si] = o2 + n;
    } else if (E->dspark && r == 2 && n <= MAXR) {   // DSpark rollback: keep this pass's input rows (positions s - off ..)
        DCK(cudaMemcpyAsync(c.sv_kv[si], kvb, (size_t)m * HD * 4, cudaMemcpyDeviceToDevice, c.st));
        DCK(cudaMemcpyAsync(c.sv_sc[si], scb, (size_t)m * HD * 4, cudaMemcpyDeviceToDevice, c.st));
        E->sv_base[si] = s - off;
        E->sv_m[si] = m;
    }
    int ng = r == 2 ? m / 2 : m;
    int g0 = r == 2 ? (s - off) / 2 : s;
    if (r == 2) {
        if (m % 2) {
            DCK(cudaMemcpyAsync(c.tail_kv[si], kvb + (size_t)(m - 1) * HD, HD * 4, cudaMemcpyDeviceToDevice, c.st));
            DCK(cudaMemcpyAsync(c.tail_sc[si], scb + (size_t)(m - 1) * HD, HD * 4, cudaMemcpyDeviceToDevice, c.st));
        }
        E->tail_n[si] = m % 2;
    }
    const bool via_x1 = E->asplit && n <= MAXR && !W.dspark;   // = attention()'s split condition
    c.cx_l = l;
    c.cx_g0 = g0;
    c.cx_ng = ng;
    if (ng == 0) return;
    float* lat = (float*)c.phase;                // [ng][512]
    float* ikf = lat + (size_t)ng * HD;          // [ng][128]
    float* lat2 = ikf + (size_t)ng * IDIM;       // [ng][512]
    if (r == 2) k_comp_pool2<<<cdiv((size_t)ng * HD, 256), 256, 0, c.st>>>(kvb, scb, ng, lat2);
    else DCK(cudaMemcpyAsync(lat2, kvb, (size_t)ng * HD * 4, cudaMemcpyDeviceToDevice, c.st));
    rms(c.st, lat2, HD, W.c_norm, lat, HD, ng, HD);
    // index keys
    lin32(c, W.i_wk, IDIM, HD, lat, HD, ng, ikf, IDIM);
    rms(c.st, ikf, IDIM, W.i_knorm, ikf, IDIM, ng, IDIM);
    rope(c.st, ikf, IDIM, ng, 1, IDIM, c.cosv[1], c.sinv[1], g0 * r, r, 0);
    q_fp4(c.st, ikf, IDIM, ng, IDIM, 32, 0, c.src[si].ik + (size_t)g0 * IDIM, IDIM);
    // compressed KV
    rope(c.st, lat, HD, ng, 1, HD, c.cosv[1], c.sinv[1], g0 * r, r, 0);
    q_fp4(c.st, lat, HD, ng, HD, 16, 1, c.src[si].ckv + (size_t)g0 * HD, HD);
    E->tc[si] = g0 + ng;
    // broadcast the new rows to the cards that hold later consumers
    for (auto& o2 : E->cards) {
        if (via_x1) break;   // review r10 M2: in attn_split's X1
        if (o2.dev == c.dev || (o2.l1 <= l && !E->asplit)) continue;   // R8 split: every card attends
        // (UVA copies over P2P: capturable in the decode step graph, R10)
        DCK(cudaMemcpyAsync(o2.src[si].ckv + (size_t)g0 * HD, c.src[si].ckv + (size_t)g0 * HD, (size_t)ng * HD * 2,
                            cudaMemcpyDefault, c.st));
        DCK(cudaMemcpyAsync(o2.src[si].ik + (size_t)g0 * IDIM, c.src[si].ik + (size_t)g0 * IDIM, (size_t)ng * IDIM * 2,
                            cudaMemcpyDefault, c.st));
    }
}

// [dsv4-pf5-idx] the indexer's per-64-query work hoisted (prefill): i_wq_b / i_wproj prepared once, the projections over
// many 64-query blocks as one cublasLt call with the 64-block algorithm, scores / reduce / top-k over QBe-query blocks.
// Same per-block shapes' algorithms and per-row kernels: bit-identical to the 64-query loop. Off DSV4_NOIDXH=1
static const bool E_idxh_off = getenv("DSV4_NOIDXH") != nullptr;
static bool lt_init(Card& k);
static bool lt_gemm(Card& k, const bf16* X, int ldx, const bf16* W, int ldw, float* C, int ldc, int n, int N, int K, int Nref,
                    bool check_only = false, int nref = 0, int ldc_ref = 0);
// the hoisted path's Lt shapes validated once per (rows, N, nref) (a failure: the loop)
static const bool E_idxt_off = getenv("DSV4_NOIDXT") != nullptr;   // causal trim of the score GEMM off
// [dsv4-ctx1-sr] decode (<= MAXR rows): fused index score-reduce from IDXSR_MINT keys (microbench: break-even ~3,072, slower than the old
// pair at 600 keys, 1.4x at 4,096, 4.5x from 20K); DSV4_NOIDXSR=1 off. [dsv4-ctx1-cand] candidates only past layer 20
// when the source holds more than CAND_K blocks; DSV4_NOIDXC=1 off
static const bool E_idxsr_off = getenv("DSV4_NOIDXSR") != nullptr;
static const bool E_idxc_off = getenv("DSV4_NOIDXC") != nullptr;
// [dsv4-ctx2-idxf] prefill (hoisted indexer): index scores on tensor cores fused with the head reduce (k_idx_pf, no S
// tensor; order-only); DSV4_NOIDXPF=1 = cuBLAS score GEMM + k_idx_reduce
static const bool E_idxpf_off = getenv("DSV4_NOIDXPF") != nullptr;
static const int IDXSR_MINT = getenv("DSV4_IDXSR_MINT") ? std::max(0, atoi(getenv("DSV4_IDXSR_MINT"))) : 3072;
// [dsv4-rel2-p1b] ldc_run > 0: the run writes a key tile of ldc_run columns (N's algorithm, Nrun <= ldc_run columns)
static bool idxh_check(Card& c, int rows, int N, int K, int nref, int Nrun = 0, int ldc_run = 0) {
    if (Nrun <= 0) Nrun = N;
    if (ldc_run <= 0) ldc_run = N;
    static std::map<std::pair<const Card*, std::array<int, 6>>, bool> ok;   // review pf5 L3: per Card (a reload re-checks)
    if (ok.size() > 8192) ok.clear();   // review pf5 L2: bounded (a re-check only costs host time)
    const auto key = std::make_pair((const Card*)&c, std::array<int, 6>{rows, N, K, nref, Nrun, ldc_run});
    auto f = ok.find(key);
    if (f != ok.end() && c.lt_algo.count({N, nref, K, K, K, N, 0})) return f->second;
    // (X / W / C pointers unused by the check; the ld's are the call's: x K, w K, c ldc_run; the algorithm N's)
    const bool r = lt_gemm(c, nullptr, K, nullptr, K, nullptr, ldc_run, rows, Nrun, K, N, true, nref, N);
    if (!r) logf_("indexer hoist: cublasLt shape check failed (rows %d N %d K %d ref %d): 64-query loop\n", rows, N, K, nref);
    return ok[key] = r;
}
static size_t al256(size_t b) { return (b + 255) & ~(size_t)255; }
// returns false when the hoisted path can't run for this call (nothing enqueued)
static bool indexer_hoist(Engine* E, Card& c, const Layer& W, int l, int s, int n, int si, int T, int k, int r, int nbk,
                          int ldc) {
    constexpr int QB = 64, QP = 512, NQ = IH * IDIM;
    const F8& Wq = W.i_wq_b;
    if (E_idxh_off || n <= QB || Wq.si || Wq.N != NQ || Wq.K != QLR || (size_t)NQ * QLR > c.deq_elems ||
        (size_t)IH * D * 2 > c.deq_elems || !lt_init(c))
        return false;
    // the region the 64-query loop may use (the alloc / pfx_bound formula: attention + indexer)
    const size_t budget = (size_t)2 * 512 * 32768 * 4 + (size_t)512 * 8192 * 4 + idx_ws(E->o.maxctx);   // [dsv4-rel2-p1b]
    const size_t b_qib = al256((size_t)n * NQ * 2), b_wi = al256((size_t)n * IH * 4), b_qi = al256((size_t)QP * NQ * 4);
    const size_t used = b_qib + b_wi + b_qi;
    const int TKh = std::min(T, idx_tk(E->o.maxctx));   // [dsv4-rel2-p1b] score key tile (= T up to IDX_TK keys)
    const bool fz = !E_idxpf_off;   // [dsv4-ctx2-idxf] fused tensor-core scores: no S
    const size_t per = (fz ? 0 : (size_t)IH * TKh * 4) + (size_t)T * 4 + (size_t)nbk * 4;   // S, sc, bs per query row
    if (used + 3 * 256 + per * QB > budget) return false;
    int QBe = (int)std::min<size_t>((budget - used - 3 * 256) / per / QB * QB, (size_t)(n + QB - 1) / QB * QB);
    const int nfull = n / QB * QB, ntail = n - nfull;
    for (int b0 = 0; b0 < nfull; b0 += QP)
        if (!idxh_check(c, std::min(QP, nfull - b0), NQ, QLR, QB)) return false;
    // causal trim: a block's score GEMM stops at its last row's visible count (the columns past it are -inf in the
    // reduce, which no longer reads them); T's algorithm on the first Nb rows of W (= the asplit share rule)
    // (rounded up to 8: with T a multiple of 4 the heuristic's kernel needs an aligned row count, tests/idx_lt_test.cu)
    auto ncols = [&](int b0, int nb) { return std::min(T, (std::max(1, (s + b0 + nb) / r) + 7) / 8 * 8); };
    for (int b0 = 0; b0 < nfull && !fz; b0 += QBe) {
        const int nbf = std::min(QBe, nfull - b0), nc = E_idxt_off ? T : ncols(b0, nbf);
        for (int t0 = 0; t0 < nc; t0 += TKh)   // [dsv4-rel2-p1b] each tile's run shape (ldc = its width) with T's algorithm
            if (!idxh_check(c, nbf * IH, T, IDIM, QB * IH, std::min(TKh, nc - t0), std::min(TKh, T - t0))) return false;
    }
    char* p = c.phase;
    bf16* qib = (bf16*)p;                    // [n][4096]
    float* wi = (float*)(p + b_qib);         // [n][32]
    float* qi = (float*)(p + b_qib + b_wi);  // [QP][4096]
    float* S = (float*)(p + used);           // [QBe * 32][TKh] (none when fused)
    float* sc = (float*)((char*)S + (fz ? 0 : al256((size_t)QBe * IH * TKh * 4)));   // [QBe][T]
    float* bs = (float*)((char*)sc + al256((size_t)QBe * T * 4));       // [QBe][nbk]
    DCHECK((char*)bs + (size_t)QBe * nbk * 4 <= c.phase + budget, "indexer hoist region");
    cudaStream_t st = c.st;
    // i_wq_b over the full 64-blocks: linq's prefill input conversion (all rows at once: per-row ops), weights once
    if (nfull > 0) {
        if (Wq.sq) {
            k_to_bf16s<<<cdiv((size_t)nfull * QLR, 256), 256, 0, st>>>(c.qr, QLR, nfull, QLR, c.xb, QLR);
            k_deq_q8<<<cdiv((size_t)NQ * QLR / 2, 256), 256, 0, st>>>((const int8_t*)Wq.w, Wq.sq, QLR, 0, NQ, c.deq);
        } else {
            q_fp8(st, c.qr, QLR, nfull, QLR, nullptr, 0, c.xb, QLR);
            k_deq_f8<<<cdiv((size_t)NQ * QLR / 2, 256), 256, 0, st>>>(Wq.w, Wq.s, QLR, 0, NQ, c.deq);
        }
        for (int b0 = 0; b0 < nfull; b0 += QP) {
            const int m = std::min(QP, nfull - b0);
            DCHECK(lt_gemm(c, c.xb + (size_t)b0 * QLR, QLR, c.deq, QLR, qi, NQ, m, NQ, QLR, NQ, false, QB), "indexer hoist: Lt");
            rope(st, qi, NQ, m, IH, IDIM, c.cosv[1], c.sinv[1], s + b0, 1, 0);
            q_fp4(st, qi, NQ, m, NQ, 32, 0, qib + (size_t)b0 * NQ, NQ);
        }
    }
    if (ntail > 0) {   // the last block exactly as the loop runs it (GEMV path at <= MAXR rows)
        linq(c, Wq, c.qr + (size_t)nfull * QLR, QLR, ntail, qi, NQ);
        rope(st, qi, NQ, ntail, IH, IDIM, c.cosv[1], c.sinv[1], s + nfull, 1, 0);
        q_fp4(st, qi, NQ, ntail, NQ, 32, 0, qib + (size_t)nfull * NQ, NQ);
    }
    // i_wproj: widened once (lin32's prefill path), one sgemm per 64-block as the loop (the tail via lin32: GEMV at <= MAXR)
    k_bf16_f32<<<cdiv((size_t)IH * D, 256), 256, 0, st>>>(W.i_wproj, (size_t)IH * D, (float*)c.deq);
    for (int b0 = 0; b0 < nfull; b0 += QB)
        sgemm(c.blas, c.x + (size_t)b0 * D, D, (const float*)c.deq, D, wi + (size_t)b0 * IH, IH, QB, IH, D);
    if (ntail > 0) lin32(c, W.i_wproj, IH, D, c.x + (size_t)nfull * D, D, ntail, wi + (size_t)nfull * IH, IH);
    // scores, reduce, candidates, top-k over QBe-query blocks (the score GEMM: 64-block algorithm over the full blocks,
    // the tail block's own GemmEx as the loop)
    for (int b0 = 0; b0 < n; b0 += QBe) {
        const int nb = std::min(QBe, n - b0);
        const int nbf = std::min(nb, std::max(0, nfull - b0));
        const int nc = nbf > 0 ? (E_idxt_off ? T : ncols(b0, nbf)) : 0;
        if (fz) k_idx_pf<<<cdiv(nb, IPF_R), 256, 0, st>>>(qib + (size_t)b0 * NQ, wi + (size_t)b0 * IH, nb, c.src[si].ik, T, s + b0, r, sc, T);
        for (int t0 = 0; t0 < T && !fz; t0 += TKh) {   // [dsv4-rel2-p1b] key tiles (one = the old calls)
            const int tn = std::min(TKh, T - t0);
            const bf16* ikt = c.src[si].ik + (size_t)t0 * IDIM;
            if (nc > t0)
                DCHECK(lt_gemm(c, qib + (size_t)b0 * NQ, IDIM, ikt, IDIM, S, tn, nbf * IH, std::min(tn, nc - t0), IDIM, T,
                               false, QB * IH, T), "indexer hoist: Lt");
            if (nb > nbf)
                gemm_bf16(c.blas, qib + (size_t)(b0 + nbf) * NQ, IDIM, ikt, IDIM, S + (size_t)nbf * IH * tn, tn,
                          (nb - nbf) * IH, tn, IDIM);
            k_idx_reduce<<<cdiv((size_t)nb * tn, 256), 256, 0, st>>>(S, wi + (size_t)b0 * IH, nb, tn, s + b0, r, sc + t0, T, t0);
        }
        if (l == CAND_SRC) {
            k_cand_blocks<<<cdiv((size_t)nb * nbk, 256), 256, 0, st>>>(sc, T, nb, T, s + b0, r, bs, nbk);
            k_topk<<<nb, 1024, 0, st>>>(bs, nbk, nbk, CAND_K, nullptr, 0, c.cand + (size_t)b0 * ldc, ldc);
        } else if (l > CAND_SRC) {
            k_cand_mask<<<cdiv((size_t)nb * T, 256), 256, 0, st>>>(sc, T, nb, T, c.cand + (size_t)b0 * ldc, ldc);
        }
        k_topk<<<nb, 1024, 0, st>>>(sc, T, T, k, c.sel + (size_t)b0 * ITOPK, ITOPK, nullptr, 0);
    }
    return true;
}

// indexer (index-source layer l): selection of compressed positions for queries s..s+n-1 into c.sel ([n][512]);
// layer 20 also writes the candidate blocks c.cand ([n][nbk])
static void indexer(Engine* E, Card& c, const Layer& W, int l, int s, int n, int* kout) {
    const int src = kv_src_of(l), si = src_index(src), r = W.ratio;
    const int T = E->tc[si];
    const int k = std::min(ITOPK, (s + n) / r);
    *kout = k;
    if (k <= 0 || T == 0) {
        *kout = 0;
        return;
    }
    const int nbk = (T + CAND_BS - 1) / CAND_BS;
    const int ldc = E->o.maxctx / CAND_BS + 1;
    // R20 [dsv4-r20-idxskip]: every visible compressed row is selected when T <= ITOPK (context <= ~1K at ratio 2), so
    // the selection needs no scores: index order (= k_topk's order when all are taken), the candidate mask likewise;
    // the indexer's projections, scores and top-k are skipped (opt-in DSV4_IDXSKIP=1)
    if (T <= ITOPK && !g_topk_old_h) {   // (review r20 L1: DSV4_NOIDXSKIP dropped, = without DSV4_IDXSKIP)
        k_idx_all<<<n, 256, 0, c.st>>>(n, s, r, k, c.sel, ITOPK, l == CAND_SRC ? c.cand : nullptr, ldc, nbk);
        return;
    }
    if (n > MAXR && indexer_hoist(E, c, W, l, s, n, si, T, k, r, nbk, ldc)) return;
    const int QB = 64;
    const int TKm = idx_tk(E->o.maxctx);              // [dsv4-rel2-p1b] key tile capacity at QB rows
    float* S = (float*)c.phase;                       // [QB * 32][TKm]
    float* sc = S + (size_t)QB * IH * TKm;            // [QB][T]
    float* bs = sc + (size_t)QB * E->o.maxctx;        // [QB][nbk]
    float* qi = bs + (size_t)QB * E->o.maxctx;        // [QB][4096]
    bf16* qib = (bf16*)(qi + (size_t)QB * 4096);      // [QB][4096]
    float* wi = (float*)(qib + (size_t)QB * 4096);    // [QB][32]
    // [dsv4-ctx1-sr] / [dsv4-ctx1-cand] decode rows: fused score-reduce (no S) from IDXSR_MINT keys; layers past the
    // candidate source with more blocks than CAND_K score only their candidates (compact, index order) and take the
    // top-k over that list. Bit-identical to the loop below (tests/idx_sr_test.cu).
    if (n <= MAXR && !E_idxsr_off && T >= IDXSR_MINT) {
        linq(c, W.i_wq_b, c.qr, QLR, n, qi, IH * IDIM);
        rope(c.st, qi, IH * IDIM, n, IH, IDIM, c.cosv[1], c.sinv[1], s, 1, 0);
        q_fp4(c.st, qi, IH * IDIM, n, IH * IDIM, 32, 0, qib, IH * IDIM);
        lin32(c, W.i_wproj, IH, D, c.x, D, n, wi, IH);
        const bf16* ik = c.src[si].ik;
        if (l > CAND_SRC && nbk > CAND_K && !E_idxc_off) {
            constexpr int NCP = CAND_K * CAND_BS;
            static_assert(NCP % IDXSR_NT == 0, "k_idx_sr grid NCP / IDXSR_NT covers every candidate key");   // review ctx1 L9 / rel9 L6
            int* blist = (int*)bs;   // [n][CAND_K]
            float* scc = S;          // [n][NCP]
            DCHECK((size_t)n * NCP <= (size_t)QB * IH * TKm && (size_t)n * CAND_K <= (size_t)QB * E->o.maxctx, "idx cand ws");
            k_cand_list<<<n, 1024, 0, c.st>>>(c.cand, ldc, nbk, blist, CAND_K);
            k_idx_sr<true><<<dim3(NCP / IDXSR_NT, n), IDXSR_NT, 0, c.st>>>(qib, wi, n, ik, T, s, r, scc, NCP, blist, CAND_K);
            k_topk<<<n, 1024, 0, c.st>>>(scc, NCP, NCP, k, c.sel, ITOPK, nullptr, 0, blist);
            return;
        }
        k_idx_sr<false><<<cdiv(T, IDXSR_NT), IDXSR_NT, 0, c.st>>>(qib, wi, n, ik, T, s, r, sc, T, nullptr, 0);
        if (l == CAND_SRC) {
            k_cand_blocks<<<cdiv((size_t)n * nbk, 256), 256, 0, c.st>>>(sc, T, n, T, s, r, bs, nbk);
            k_topk<<<n, 1024, 0, c.st>>>(bs, nbk, nbk, CAND_K, nullptr, 0, c.cand, ldc);
        } else if (l > CAND_SRC) {
            k_cand_mask<<<cdiv((size_t)n * T, 256), 256, 0, c.st>>>(sc, T, n, T, c.cand, ldc);
        }
        k_topk<<<n, 1024, 0, c.st>>>(sc, T, T, k, c.sel, ITOPK, nullptr, 0);
        return;
    }
    for (int b0 = 0; b0 < n; b0 += QB) {
        int nb = std::min(QB, n - b0);
        linq(c, W.i_wq_b, c.qr + (size_t)b0 * QLR, QLR, nb, qi, IH * IDIM);
        rope(c.st, qi, IH * IDIM, nb, IH, IDIM, c.cosv[1], c.sinv[1], s + b0, 1, 0);
        q_fp4(c.st, qi, IH * IDIM, nb, IH * IDIM, 32, 0, qib, IH * IDIM);
        lin32(c, W.i_wproj, IH, D, c.x + (size_t)b0 * D, D, nb, wi, IH);
        // scale 128^-0.5 * 32^-0.5 = 1/64 folded into the reduce input below
        // [dsv4-rel2-p1b] over key tiles of TKm x QB / nb (one tile = the whole T up to TKm keys at QB rows)
        const int tk = (int)std::min<size_t>((size_t)T, (size_t)TKm * QB / nb);
        // review ctx1 M1: prefill rows here (a fallback of the hoist, chunks of 9..64 rows) score as the hoist does
        if (n > MAXR && !E_idxpf_off)
            k_idx_pf<<<cdiv(nb, IPF_R), 256, 0, c.st>>>(qib, wi, nb, c.src[si].ik, T, s + b0, r, sc, T);
        for (int t0 = 0; t0 < T && !(n > MAXR && !E_idxpf_off); t0 += tk) {
            const int tn = std::min(tk, T - t0);
            const bf16* ikt = c.src[si].ik + (size_t)t0 * IDIM;
            if (n <= MAXR)   // fixed summation order for any row count: a verify pass matches one-row steps exactly
                k_idx_score<<<cdiv(tn, 8), 256, 0, c.st>>>(qib, nb * IH, ikt, tn, S);
            else
                gemm_bf16(c.blas, qib, IDIM, ikt, IDIM, S, tn, nb * IH, tn, IDIM);
            k_idx_reduce<<<cdiv((size_t)nb * tn, 256), 256, 0, c.st>>>(S, wi, nb, tn, s + b0, r, sc + t0, T, t0);
        }
        if (l == CAND_SRC) {
            k_cand_blocks<<<cdiv((size_t)nb * nbk, 256), 256, 0, c.st>>>(sc, T, nb, T, s + b0, r, bs, nbk);
            k_topk<<<nb, 1024, 0, c.st>>>(bs, nbk, nbk, CAND_K, nullptr, 0, c.cand + (size_t)b0 * ldc, ldc);
        } else if (l > CAND_SRC) {
            k_cand_mask<<<cdiv((size_t)nb * T, 256), 256, 0, c.st>>>(sc, T, nb, T, c.cand + (size_t)b0 * ldc, ldc);
        }
        k_topk<<<nb, 1024, 0, c.st>>>(sc, T, T, k, c.sel + (size_t)b0 * ITOPK, ITOPK, nullptr, 0);
    }
}

// ---- R8 attention split (decode, n <= MAXR): the owner computes q low rank, window KV, compressor and indexer as
// before, then sends qr, the new window rows and the selection to the other two cards (X1); every card runs wq_b,
// attention and wo_a for its wo_a groups, the o2 slices are all-gathered (X2), every card runs its wo_b rows, and the
// owner gathers them (X3). Each output element is computed by the same kernel arithmetic as unsplit: bit-identical.
static XSeg xseg(std::initializer_list<std::tuple<void*, int, int, int, int>> segs) {   // (base, ld, width, r0, rmask) in u32
    XSeg g{};
    int tot = 0;
    for (auto& t : segs) {
        g.p[g.nseg] = (uint32_t*)std::get<0>(t);
        g.ld[g.nseg] = std::get<1>(t);
        g.w[g.nseg] = std::get<2>(t);
        g.r0[g.nseg] = std::get<3>(t);
        g.rmask[g.nseg] = std::get<4>(t);
        g.nr[g.nseg] = 1 << 30;
        tot += g.w[g.nseg++];
    }
    g.wpr = (tot + 2) / 3;
    g.tot = tot;
    return g;
}
// R11: one more segment of nr rows (rows >= nr of the message carry zeros there and store nothing)
static void xseg_add(XSeg& g, void* p, int ld, int w, int nr) {
    DCHECK(g.nseg < 5, "XSeg segments");
    g.p[g.nseg] = (uint32_t*)p;
    g.ld[g.nseg] = ld;
    g.w[g.nseg] = w;
    g.r0[g.nseg] = 0;
    g.rmask[g.nseg] = -1;
    g.nr[g.nseg++] = nr;
    g.tot += w;
    g.wpr = (g.tot + 2) / 3;
}
static void xpush(Card& c, const XSeg& g, int n, uint4* d0, uint4* d1, unsigned tag) {
    k_xpush<<<cdiv((size_t)n * g.wpr, 128), 128, 0, c.st>>>(g, n, d0, d1, tag);
}
static void xpull(Card& c, const XSeg& g, int n, const uint4* src, unsigned tag) {
    k_xpull<<<cdiv((size_t)n * g.wpr, 128), 128, 0, c.st>>>(g, n, src, tag);
}
static bool E_xp2_off = getenv("DSV4_NOXPULL2") != nullptr;   // R10: one launch for the two peers' pulls
static void xpull2(Card& c, const XSeg& g1, const uint4* s1, const XSeg& g2, const uint4* s2, int n, unsigned tag) {
    if (E_xp2_off) {
        xpull(c, g1, n, s1, tag);
        xpull(c, g2, n, s2, tag);
        return;
    }
    const unsigned nb1 = cdiv((size_t)n * g1.wpr, 128), nb2 = cdiv((size_t)n * g2.wpr, 128);
    k_xpull2<<<nb1 + nb2, 128, 0, c.st>>>(g1, s1, g2, s2, n, (int)nb1, tag);
}
// [dsv4-dec2-attl] (review dec2 M1) DSV4_NOATTL=1: the R10 attention kernel k_attn_o (same values)
static const bool E_attl_off = getenv("DSV4_NOATTL") != nullptr;
// [dsv4-dec3-atts] k_attn_s (2 blocks per SM, same values) when every block has at most one 32-key tile (nk = an upper
// bound on every row's keys: overestimating only picks k_attn; decode / verify rows always: nk <= WIN + ITOPK);
// DSV4_NOATTS=1 off
static const bool E_atts_off = getenv("DSV4_NOATTS") != nullptr;
static auto attn_kern(int nk = 1 << 30, int nsplit = 1) {
    if (E_attl_off) return k_attn_o;
    return !E_atts_off && (long long)nsplit * AT_KT >= nk ? k_attn_s : k_attn;
}
// [dsv4-dec2-x2q] X2 as wo_b's int8 input (Q8_0 wo_b, decode rows; DSV4_NOX2Q=1 off: fp32 o2 columns)
static const bool E_x2q_off = getenv("DSV4_NOX2Q") != nullptr;
// (review dec2 L2 / L3: one decision per layer for sender and receivers: every share's wo_b Q8_0 and >= 1 head group,
// every card's int8 scratch allocated)
static bool x2q_on(const Engine* E, const Layer& W, int n) {
    if (E_x2q_off || n > MAXR) return false;
    for (int j = 0; j < (int)E->cards.size() && j < 3; j++)
        if (!W.ash[j].wo_b.sq || W.ash[j].ng <= 0 || !E->cards[j].d_di8) return false;
    return true;
}
// R9: one card's share in 3 stages (0: wq_b; 1: rope, attention, wo_a, X2 push; 2: X2 pulls, wo_b, X3 push); attn_split
// issues each stage on all cards before the next, so no card waits for the host to enqueue another card's whole share
// (nsys R9: the owner's share started ~65 us after its X1 push)
static void attn_share(Engine* E, Card& k, int ki, const Layer& W, int l, int s, int n, int nsel, int kind,
                       const bf16* kvw, int kmask, unsigned tag, int stage) {
    const Layer::AShare& A = W.ash[ki];
    cudaStream_t st = k.st;
    const int SQ = 512;
    float* q = (float*)k.phase;
    float* o = q + (size_t)SQ * NH * HD;
    float* o2 = o + (size_t)SQ * NH * HD;
    const int h0 = A.g0 * AT_HG, nh = A.ng * AT_HG;
    if (stage == 0) {
        // review r15 L3 / N5: with qrb a peer's fp32 qr is stale: linq must have taken the bf16 input
        const bool qu = linq(k, A.wq_b, k.qr, QLR, n, q + (size_t)h0 * HD, NH * HD, 0, qrb_on() ? k.qrb : nullptr);
        DCHECK(!qrb_on() || qu, "wq_b: linq did not take the bf16 qr (n %d)", n);
        return;
    }
    if (stage == 1) {
        const bool rf = !E_rf_off;   // R10: both ropes inside the attention kernels
        if (!rf) rope(st, q + (size_t)h0 * HD, NH * HD, n, nh, HD, k.cosv[kind], k.sinv[kind], s, 1, 0);
        const int nk = std::min(WIN, s + n) + nsel;
        const int nsplit = std::max(1, std::min(32, (nk + AT_KT - 1) / AT_KT));
        const SrcCache* sc = W.ratio ? &k.src[src_index(kv_src_of(l))] : nullptr;
        bf16* oq = rf && n == 1 && !E_oq_off ? k.d_obf : nullptr;   // R10: wo_a input quantized by the attention store
        attn_kern(nk, nsplit)<<<dim3(n, A.ng, nsplit), 256, AT_SMEM, st>>>(q, s, kvw, sc ? sc->ckv : nullptr, nsel ? k.sel : nullptr, nsel,
                                                           ITOPK, A.sink, o, nsplit, k.part, W.ratio, -1, WIN, A.g0, kmask,
                                                           rf ? k.cosv[kind] : nullptr, rf ? k.sinv[kind] : nullptr, oq);
        if (nsplit > 1)
            k_attn_comb<<<dim3(n, nh), 256, 0, st>>>(k.part, nsplit, A.sink, o, h0, s, rf ? k.cosv[kind] : nullptr,
                                                      rf ? k.sinv[kind] : nullptr, oq);
        if (!rf) rope(st, o + (size_t)h0 * HD, NH * HD, n, nh, HD, k.cosv[kind], k.sinv[kind], s, 1, 1);
        linq(k, A.wo_a, o + (size_t)A.g0 * AT_HG * HD, NH * HD, n, o2 + (size_t)A.g0 * OLR, OG * OLR, OLR,
             oq ? oq + (size_t)h0 * HD : nullptr);
        // X2: my o2 columns to the other two cards
        uint4* dst[2] = {};
        int nd = 0;
        for (int j = 0; j < 3; j++)
            if (j != ki) dst[nd++] = E->cards[j].rx2 + (size_t)ki * MAXR * X2W;
        if (x2q_on(E, W, n)) {   // [dsv4-dec2-x2q]: my columns as wo_b's int8 input (k_q8r's values), sent int8 + scales
            int8_t* q = k.d_di8 + (size_t)A.g0 * OLR;
            float* qs = k.d_di8s + (size_t)A.g0 * OLR / 32;
            const int kc = A.ng * OLR;
            k_q8rs<<<cdiv((size_t)n * kc, 256), 256, 0, st>>>(o2 + (size_t)A.g0 * OLR, OG * OLR, n, kc, q, OG * OLR, qs,
                                                              OG * OLR / 32);
            xpush(k, xseg({{q, OG * OLR / 4, kc / 4, 0, -1}, {qs, OG * OLR / 32, kc / 32, 0, -1}}), n, dst[0], dst[1], tag + 1);
            return;
        }
        xpush(k, xseg({{o2 + (size_t)A.g0 * OLR, OG * OLR, A.ng * OLR, 0, -1}}), n, dst[0], dst[1], tag + 1);
        return;
    }
    const bool x2q = x2q_on(E, W, n);
    {   // theirs into mine (R10: both peers in one launch)
        int js[2], nj = 0;
        for (int j = 0; j < 3; j++)
            if (j != ki) js[nj++] = j;
        const Layer::AShare &B0 = W.ash[js[0]], &B1 = W.ash[js[1]];
        if (x2q) {   // [dsv4-dec2-x2q]: their int8 columns + scales straight into wo_b's input
            auto seg = [&](const Layer::AShare& B) {
                return xseg({{k.d_di8 + (size_t)B.g0 * OLR, OG * OLR / 4, B.ng * OLR / 4, 0, -1},
                             {k.d_di8s + (size_t)B.g0 * OLR / 32, OG * OLR / 32, B.ng * OLR / 32, 0, -1}});
            };
            xpull2(k, seg(B0), k.rx2 + (size_t)js[0] * MAXR * X2W, seg(B1), k.rx2 + (size_t)js[1] * MAXR * X2W, n, tag + 1);
        } else
        xpull2(k, xseg({{o2 + (size_t)B0.g0 * OLR, OG * OLR, B0.ng * OLR, 0, -1}}), k.rx2 + (size_t)js[0] * MAXR * X2W,
               xseg({{o2 + (size_t)B1.g0 * OLR, OG * OLR, B1.ng * OLR, 0, -1}}), k.rx2 + (size_t)js[1] * MAXR * X2W, n,
               tag + 1);
    }
    const bool own = ki == W.card;
    float* y = own ? k.ao + A.n0 : k.d_ysl;
    if (x2q)   // the whole int8 input is in place: the GEMV without its k_q8r (same values)
        i8_linear(st, (const int8_t*)A.wo_b.w, A.wo_b.sq, A.wo_b.N, A.wo_b.K, 0, nullptr, 0, n, y, own ? D : 1728, k.d_di8,
                  k.d_di8s, (size_t)MAXR * NH * HD, true);
    else
    linq(k, A.wo_b, o2, OG * OLR, n, y, own ? D : 1728);
    if (!own)   // X3: my wo_b rows to the owner
        xpush(k, xseg({{k.d_ysl, 1728, A.nn, 0, -1}}), n, E->cards[W.card].rx3 + (size_t)ki * MAXR * X3W, nullptr, tag + 2);
}
static bool E_astage_off = getenv("DSV4_NOASTAGE") != nullptr;   // R9: old order (each card's whole share in turn)
static void attn_split(Engine* E, Card& c, const Layer& W, int l, int s, int n, int nsel, int kind) {
    const unsigned tag = (E->xseq * 64u + (unsigned)l) * 8u + 1u;   // X1..X3 = + 0..2; R9 X4 / X5 = + 3 / + 4
    const int oi = W.card;
    // X1 from the owner: qr [n][1280] fp32, window rows [n][512] bf16 (= 256 u32), selection [n][512] when used
    void* q1 = qrb_on() ? (void*)c.qrb : (void*)c.qr;
    const int qw1 = qrb_on() ? QLR / 2 : QLR;
    XSeg g1 = nsel ? xseg({{q1, qw1, qw1, 0, -1}, {W.kvw, HD / 2, HD / 2, s, W.kmask}, {c.sel, ITOPK, ITOPK, 0, -1}})
                   : xseg({{q1, qw1, qw1, 0, -1}, {W.kvw, HD / 2, HD / 2, s, W.kmask}});
    // review r10 M2: a kv source's new compressed rows (ckv, index keys) ride in X1 (LL12: data and tag in one store),
    // not as a CE copy published only by later LL12 words
    const bool cx = is_kv_src(l) && W.ratio;
    const int csi = cx ? src_index(l) : 0, cng = cx && c.cx_l == l ? c.cx_ng : 0, cg0 = cx && c.cx_l == l ? c.cx_g0 : 0;
    if (cx) {
        xseg_add(g1, c.src[csi].ckv + (size_t)cg0 * HD, HD / 2, HD / 2, cng);
        xseg_add(g1, c.src[csi].ik + (size_t)cg0 * IDIM, IDIM / 2, IDIM / 2, cng);
    }
    uint4* dst[2] = {};
    int nd = 0;
    for (int j = 0; j < 3; j++)
        if (j != oi) dst[nd++] = E->cards[j].rx1;
    xpush(c, g1, n, dst[0], dst[1], tag);
    auto share = [&](int j, int stage) {
        Card& k = E->cards[j];
        DCK(cudaSetDevice(k.dev));
        const Layer::AShare& A = W.ash[j];
        if (stage == 0 && j != oi) {
            void* qp = qrb_on() ? (void*)k.qrb : (void*)k.qr;
            XSeg gp = nsel ? xseg({{qp, qw1, qw1, 0, -1}, {A.ring, HD / 2, HD / 2, s, ARING - 1}, {k.sel, ITOPK, ITOPK, 0, -1}})
                           : xseg({{qp, qw1, qw1, 0, -1}, {A.ring, HD / 2, HD / 2, s, ARING - 1}});
            if (cx) {
                xseg_add(gp, k.src[csi].ckv + (size_t)cg0 * HD, HD / 2, HD / 2, cng);
                xseg_add(gp, k.src[csi].ik + (size_t)cg0 * IDIM, IDIM / 2, IDIM / 2, cng);
            }
            xpull(k, gp, n, k.rx1, tag);
        }
        attn_share(E, k, j, W, l, s, n, nsel, kind, j == oi ? W.kvw : A.ring, j == oi ? W.kmask : ARING - 1, tag, stage);
    };
    int ord[3], no = 0;   // stages 0, 1: owner first; stage 2: the others first (the owner then waits for their X3)
    ord[no++] = oi;
    for (int j = 0; j < 3; j++)
        if (j != oi) ord[no++] = j;
    if (E_astage_off) {
        for (int i = 1; i < 3; i++)
            for (int stg = 0; stg < 3; stg++) share(ord[i], stg);
        for (int stg = 0; stg < 3; stg++) share(oi, stg);
    } else {
        for (int stg = 0; stg < 2; stg++)
            for (int i = 0; i < 3; i++) share(ord[i], stg);
        share(ord[1], 2);
        share(ord[2], 2);
        share(oi, 2);
    }
    DCK(cudaSetDevice(c.dev));
    {   // X3: the other cards' wo_b rows into c.ao (R10: one launch)
        int js[2], nj = 0;
        for (int j = 0; j < 3; j++)
            if (j != oi) js[nj++] = j;
        const Layer::AShare &B0 = W.ash[js[0]], &B1 = W.ash[js[1]];
        xpull2(c, xseg({{c.ao + B0.n0, D, B0.nn, 0, -1}}), c.rx3 + (size_t)js[0] * MAXR * X3W,
               xseg({{c.ao + B1.n0, D, B1.nn, 0, -1}}), c.rx3 + (size_t)js[1] * MAXR * X3W, n, tag + 2);
    }
}
// [dsv4-rel2-p1] (see kv_bytes)
static void kv_alloc(Engine* E, Layer& W, bool main) {
    if (!g_kring) {
        W.kvw = (bf16*)dmalloc((size_t)E->o.maxctx * HD * 2);
        W.kmask = -1;
        return;
    }
    static_assert((ARING & (ARING - 1)) == 0 && ARING >= WIN + MAXR, "ARING");
    W.kvw = (bf16*)dmalloc((size_t)ARING * HD * 2);
    W.kvsnap = (bf16*)dmalloc((size_t)ARING * HD * 2);
    DCK(cudaMemset(W.kvw, 0, (size_t)ARING * HD * 2));
    DCK(cudaMemset(W.kvsnap, 0, (size_t)ARING * HD * 2));
    W.kmask = ARING - 1;
    if (main) W.kvs = (bf16*)dmalloc((size_t)(E->o.chunk + WIN) * HD * 2);
}
// the base a prefill call (n > MAXR) addresses window rows by absolute position: the scratch shifted by its first
// position (ring mode, between kv_in and kv_out), else the [maxctx] rows
static inline bf16* kv_pf(const Layer& W) {
    if (!W.kvs) return W.kvw;
    DCHECK(W.kvs_on, "window K/V scratch used outside a prefill call (layer %d)", W.lid);
    return W.kvs - (ptrdiff_t)W.kvs_p0 * HD;
}
// copy positions [a, b) (b - a <= ARING) between the ring (row p & mask) and an absolute base, in <= 2 pieces
static void kv_ring_copy(cudaStream_t st, bf16* ring, bf16* abs_base, int a, int b, bool to_ring) {
    for (int p = a; p < b;) {
        const int r = p & (ARING - 1), m = std::min(b - p, ARING - r);
        bf16* rp = ring + (size_t)r * HD;
        bf16* ap = abs_base + (size_t)p * HD;
        DCK(cudaMemcpyAsync(to_ring ? rp : ap, to_ring ? ap : rp, (size_t)m * HD * 2, cudaMemcpyDeviceToDevice, st));
        p += m;
    }
}
// before layer l's part of a prefill call of rows [s, s + n): the WIN positions before s from the ring into the
// scratch head (the call's rows follow them); after it: the last ARING positions back into the ring
static void kv_in(Engine* E, int l, int s, int n) {
    Layer& W = E->L[l];
    if (!W.kvs) return;
    DCHECK(n <= E->o.chunk && !W.kvs_on, "kv_in: layer %d rows %d (chunk %d)", l, n, E->o.chunk);
    W.kvs_p0 = s - WIN;
    W.kvs_on = true;
    if (s > 0) kv_ring_copy(E->cards[W.card].st, W.kvw, kv_pf(W), std::max(0, s - WIN), s, false);
}
static void kv_out(Engine* E, int l, int s, int n) {
    Layer& W = E->L[l];
    if (!W.kvs) return;
    kv_ring_copy(E->cards[W.card].st, W.kvw, kv_pf(W), std::max(std::max(0, s - WIN), s + n - ARING), s + n, true);
    W.kvs_on = false;
}
// R8: after a prefill chunk that ran unsplit, the other cards' window rings get the last ARING window rows
static void ring_sync(Engine* E) {
    if (!E->asplit || E->pos == 0) return;
    const int p1 = E->pos, p0 = std::max(0, p1 - ARING);
    for (int l = 0; l < NL; l++) {
        Layer& W = E->L[l];
        Card& c = E->cards[W.card];
        DCK(cudaSetDevice(c.dev));
        for (int j = 0; j < (int)E->cards.size(); j++) {
            if (j == W.card) continue;
            for (int a = p0; a < p1;) {
                int r = a % ARING, m = std::min(p1 - a, ARING - r);
                DCK(cudaMemcpyPeerAsync(W.ash[j].ring + (size_t)r * HD, E->cards[j].dev, W.kvw + (size_t)(a & W.kmask) * HD, c.dev,
                                        (size_t)m * HD * 2, c.st));
                a += m;
            }
        }
    }
    for (auto& c : E->cards) {
        DCK(cudaSetDevice(c.dev));
        DCK(cudaStreamSynchronize(c.st));
    }
}

// R21 [dsv4-r21-defer] (KL emulation only, opt-in DSV4_DEFER=1; not with DSV4_PIPE): layer l's CPU-lane rows join
// after layer l + 1's attention (run-ahead decode passes; never the last layer, never across a card boundary)
static inline bool defer_on(Engine* E, int l, int n) {
    // review r21 L2 / L3: the deferring combine needs the fused combine (no DSV4_NOHCF) and no pipelined pass; one
    // predicate for both ends (combine at l, k_def_add at l + 1)
    return g_defer && !E_hcf_off && !E->pipe && E->ra_on && n <= MAXR && l >= 0 && l < NL - 1 &&
           E->L[l + 1].card == E->L[l].card && E->cards[E->L[l].card].d_ydef;
}
// ---- [dsv4-pf4-asplit] prefill attention split over the cards (n > MAXR, Q8_0 shares). The owner computes q low rank,
// window KV, compressor and indexer as before; the other cards pull qr (bf16, the conversion the Q8_0 linq does), the
// window rows and the selection; every card runs wq_b, rope, attention (k_attn_tc3 on its heads), rope, wo_a on its head
// groups, then the o2 columns (bf16, the wo_b input conversion) are all-gathered, every card runs its wo_b rows, the owner
// pulls them into c.ao. wo_a per group = the unsplit call; wq_b / wo_b on a share's rows run cublasLt with the algorithm
// the heuristic picks for the unsplit slice (= cublasGemmEx's choice: bit-identical, tests/asplit_gemm_test.cu). Off
// DSV4_NOPFAS=1; DSV4_PFAS_MIN (512) rows at least.
static bool E_pfas_off = getenv("DSV4_NOPFAS") != nullptr;
static const bool E_pfxd_off = getenv("DSV4_NOPFXD") != nullptr;   // [dsv4-pf4-pfxd] off: the predictor before the attention
static bool g_pfx_defer = false;
static bool pfas_ok(Engine* E, const Layer& W, int n);
static bool pfas_may(const Engine* E, int l, int n) { return pfas_ok((Engine*)E, E->L[l], n); }
static void pfx_predict(Engine* E, Card& c, const Layer& W, int n);
static const int E_pfas_min = getenv("DSV4_PFAS_MIN") ? std::max(MAXR + 1, atoi(getenv("DSV4_PFAS_MIN"))) : 512;
constexpr size_t LT_WS = (size_t)4 << 20;
static const bool E_pfas1 = getenv("DSV4_PFAS1") != nullptr;   // v1 order (whole-chunk exchange phases)
static const bool E_tc5_off = getenv("DSV4_NOTC5") != nullptr;  // [dsv4-pf13-attnsrc] off: k_attn_tc3 (same values)
static const bool E_x1b_off = getenv("DSV4_NOX1B") != nullptr;   // [dsv4-pf5-x1b] off: X1 in one piece before the shares
// [dsv4-pf14-owb] owner-side wo_b from DSV4_PFWOB_MIN rows (DSV4_NOPFWOB=1 off: every card its wo_b rows + fp32 ao pulls)
static const bool E_pfwob_off = getenv("DSV4_NOPFWOB") != nullptr;
static const int E_pfwob_min = getenv("DSV4_PFWOB_MIN") ? atoi(getenv("DSV4_PFWOB_MIN")) : 2048;
// C[n][N] (ldc) = X . W^T with the algorithm for the reference shape (Nref rows; same n, K, ld's)
// review pf4 M2: lt_algo() / lt_check() run in pfas_ok for every shape of the chunk (a failure turns the split off for it);
// lt_gemm itself then only runs validated (algorithm, shape) pairs
// cublasLt handle + workspace of a card (once; the attention split and the indexer hoist); false: none (callers degrade)
static bool lt_init(Card& k) {
    if (k.lt && k.lt_ws) return true;
    DevGuard dg(k.dev);
    if (!k.lt && cublasLtCreate(&k.lt) != CUBLAS_STATUS_SUCCESS) {
        k.lt = nullptr;
        return false;
    }
    if (!k.lt_ws && cudaMalloc(&k.lt_ws, LT_WS) != cudaSuccess) {
        k.lt_ws = nullptr;
        cudaGetLastError();
        return false;
    }
    return true;
}
// [dsv4-pf5-idx] nref > 0: the algorithm of the reference shape with nref columns (tokens) runs on n columns (the
// indexer's 64-query block shape over many blocks: bit-identical, tests/idx_lt_test.cu)
// [dsv4-rel2-p1b] ldc_ref > 0: the algorithm is the one of the C layout ldc_ref (a key tile written with its own ldc)
static bool lt_gemm(Card& k, const bf16* X, int ldx, const bf16* W, int ldw, float* C, int ldc, int n, int N, int K, int Nref,
                    bool check_only, int nref, int ldc_ref) {
    if (nref <= 0) nref = n;
    const int ldc_run = ldc;
    if (ldc_ref <= 0) ldc_ref = ldc;
    auto mk = [&](int m, int n, cublasLtMatmulDesc_t& op, cublasLtMatrixLayout_t& a, cublasLtMatrixLayout_t& b,
                  cublasLtMatrixLayout_t& c, int ldc) {
        DCHECK(cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F, CUDA_R_32F) == CUBLAS_STATUS_SUCCESS, "lt desc");
        cublasOperation_t t = CUBLAS_OP_T, nn = CUBLAS_OP_N;
        cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &t, sizeof t);
        cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB, &nn, sizeof nn);
        cublasLtMatrixLayoutCreate(&a, CUDA_R_16BF, K, m, ldw);
        cublasLtMatrixLayoutCreate(&b, CUDA_R_16BF, K, n, ldx);
        cublasLtMatrixLayoutCreate(&c, CUDA_R_32F, m, n, ldc);
    };
    auto fr = [](cublasLtMatmulDesc_t op, cublasLtMatrixLayout_t a, cublasLtMatrixLayout_t b, cublasLtMatrixLayout_t c) {
        cublasLtMatmulDescDestroy(op);
        cublasLtMatrixLayoutDestroy(a);
        cublasLtMatrixLayoutDestroy(b);
        cublasLtMatrixLayoutDestroy(c);
    };
    const std::array<int, 7> key{Nref, nref, K, ldx, ldw, ldc_ref, 0};
    if (k.lt_algo.size() > 8192) k.lt_algo.clear();   // review pf5 L2: bounded (the heuristic is asked again)
    auto it = k.lt_algo.find(key);
    if (it == k.lt_algo.end()) {
        cublasLtMatmulDesc_t op;
        cublasLtMatrixLayout_t a, b, c;
        mk(Nref, nref, op, a, b, c, ldc_ref);
        cublasLtMatmulPreference_t pr;
        cublasLtMatmulPreferenceCreate(&pr);
        size_t wl = LT_WS;
        cublasLtMatmulPreferenceSetAttribute(pr, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &wl, sizeof wl);
        cublasLtMatmulHeuristicResult_t r[1];
        int nr = 0;
        auto st = cublasLtMatmulAlgoGetHeuristic(k.lt, op, a, b, c, c, pr, 1, r, &nr);
        cublasLtMatmulPreferenceDestroy(pr);
        fr(op, a, b, c);
        if (st != CUBLAS_STATUS_SUCCESS || nr <= 0) {
            logf_("cublasLt heuristic %d (N %d n %d K %d)\n", (int)st, Nref, n, K);
            return false;
        }
        it = k.lt_algo.emplace(key, r[0].algo).first;
    }
    cublasLtMatmulDesc_t op;
    cublasLtMatrixLayout_t a, b, c;
    mk(N, n, op, a, b, c, ldc_run);
    if (check_only) {
        cublasLtMatmulHeuristicResult_t hr;
        const auto st = cublasLtMatmulAlgoCheck(k.lt, op, a, b, c, c, &it->second, &hr);
        fr(op, a, b, c);
        if (st != CUBLAS_STATUS_SUCCESS || hr.workspaceSize > LT_WS) {
            logf_("cublasLt algorithm check %d (N %d of %d, n %d K %d)\n", (int)st, N, Nref, n, K);
            return false;
        }
        return true;
    }
    const float al = 1.f, be = 0.f;
    auto r = cublasLtMatmul(k.lt, op, &al, W, a, X, b, &be, C, c, C, c, &it->second, k.lt_ws, LT_WS, k.st);
    DCHECK(r == CUBLAS_STATUS_SUCCESS, "cublasLtMatmul %d (N %d of %d, n %d K %d)", (int)r, N, Nref, n, K);
    fr(op, a, b, c);
    return true;
}
// rows [grow0, grow0 + Wsh.N) of a Q8_0 matrix of Nfull rows (share Wsh) on card k: the unsplit linq's row slices cut to
// the share, each with its slice's algorithm; y = the full output (row r at y + r). review pf4 L7: the share is
// dequantized once per layer (pfas_deq, the pieces back to back at wd) and its GEMMs run per sub-block (pfas_gemm)
static void pfas_deq(Card& k, const F8& Wsh, bf16* wd) {
    k_deq_q8<<<cdiv((size_t)Wsh.N * Wsh.K / 2, 256), 256, 0, k.st>>>((const int8_t*)Wsh.w, Wsh.sq, Wsh.K, 0, Wsh.N, wd);
}
static bool pfas_gemm(Card& k, const F8& Wsh, const bf16* wd, int grow0, int Nfull, int slice, const bf16* X, int ldx, int m,
                      float* y, int ldy, bool check_only = false) {
    for (int s0 = 0; s0 < Nfull; s0 += slice) {
        const int s1 = std::min(Nfull, s0 + slice), a = std::max(s0, grow0), b = std::min(s1, grow0 + Wsh.N);
        if (a >= b) continue;
        if (!lt_gemm(k, X, ldx, wd + (size_t)(a - grow0) * Wsh.K, Wsh.K, y + a, ldy, m, b - a, Wsh.K, s1 - s0, check_only))
            return false;
    }
    return true;
}
static bool pfas_ok(Engine* E, const Layer& W, int n) {
    if (E_pfas_off || !E->asplit || E->hostx || W.dspark || n < E_pfas_min || n > 64 * 512 || !E->o.attn_tc || E_attn2_off ||
        E->cards.size() != 3)   // (review pf4 M1: not in the no-P2P host-exchange mode)
        return false;
    for (int j = 0; j < 3; j++) {
        const Layer::AShare& A = W.ash[j];
        if (!A.wq_b.sq || !A.wo_a.sq || !A.wo_b.sq || !E->cards[j].big) return false;
        if (A.ng * AT_HG != (j == W.card ? 16 : 24)) return false;   // the k_attn_tc3 instances
    }
    for (auto& k : E->cards) {
        if (k.as_init) continue;
        DevGuard dg(k.dev);
        bool ok = lt_init(k);
        for (auto& e : k.ev_as) ok = ok && cudaEventCreateWithFlags(&e, cudaEventDisableTiming) == cudaSuccess;
        k.ev_x1.assign(65, nullptr);
        for (auto& e : k.ev_x1) ok = ok && cudaEventCreateWithFlags(&e, cudaEventDisableTiming) == cudaSuccess;
        k.ev_ab.assign(3 * 64, nullptr);
        for (auto& e : k.ev_ab) ok = ok && cudaEventCreateWithFlags(&e, cudaEventDisableTiming) == cudaSuccess;
        ok = ok && cudaStreamCreateWithFlags(&k.ass, cudaStreamNonBlocking) == cudaSuccess;
        // review pf4 L3: the function attribute is per device
        ok = ok && cudaFuncSetAttribute(k_attn_tc3<1, 16>, cudaFuncAttributeMaxDynamicSharedMemorySize, tc3_smem<1, 16>()) == cudaSuccess;
        ok = ok && cudaFuncSetAttribute(k_attn_tc3<2, 24>, cudaFuncAttributeMaxDynamicSharedMemorySize, tc3_smem<2, 24>()) == cudaSuccess;
        ok = ok && cudaFuncSetAttribute(k_attn_tc5<16>, cudaFuncAttributeMaxDynamicSharedMemorySize, tc5_smem<16>()) == cudaSuccess;
        ok = ok && cudaFuncSetAttribute(k_attn_tc5<24>, cudaFuncAttributeMaxDynamicSharedMemorySize, tc5_smem<24>()) == cudaSuccess;
        if (!ok) {   // review pf4 L5: degrade (the split stays off; what was created is kept, a few MB)
            cudaGetLastError();
            logf_("prefill attention split: no cublasLt / workspace / events on card %d: off\n", k.dev);
            E_pfas_off = true;
            return false;
        }
        k.as_init = true;
    }
    // review pf4 M2: every (algorithm, share shape) of this chunk validated once (cached per n); a failure: unsplit
    static std::map<std::pair<int, int>, bool> okn;   // (n, owner card): the shares' row ranges depend on the owner
    auto f = okn.find({n, W.card});
    if (f != okn.end()) return f->second;
    bool ok = true;
    const Card& co = E->cards[W.card];
    const int slq = std::max<int>(32, (int)std::min<size_t>(NH * HD, co.deq_elems / QLR) / 32 * 32);
    const int slb = std::max<int>(32, (int)std::min<size_t>(D, co.deq_elems / (OG * OLR)) / 32 * 32);
    for (int j = 0; j < 3 && ok; j++) {
        Card& k = E->cards[j];
        DevGuard dg(k.dev);
        const Layer::AShare& A = W.ash[j];
        for (int m : {std::min(n, 512), n % 512}) {
            if (m <= 0) continue;
            ok = ok && pfas_gemm(k, A.wq_b, nullptr, A.g0 * AT_HG * HD, NH * HD, slq, nullptr, QLR, m, nullptr, NH * HD, true);
            ok = ok && pfas_gemm(k, A.wo_b, nullptr, A.n0, D, slb, nullptr, OG * OLR, m, nullptr, D, true);
        }
    }
    if (!ok) logf_("prefill attention split: off for %d rows (cublasLt check)\n", n);
    okn[{n, W.card}] = ok;
    return ok;
}
// [dsv4-pf14-owb] the owner runs the whole wo_b on the unsplit linq's row slices (one GEMM per slice and sub-block; a share's
// sub-range on the slice's algorithm gives the same values, tests/asplit_gemm_test.cu): validated once per (n, owner);
// a failure keeps the per-card wo_b (the split stays on)
static bool pfwob_ok(Engine* E, const Layer& W, int n) {
    if (E_pfwob_off || E_pfas1 || E_x1b_off || n < E_pfwob_min) return false;   // (n <= 64 x 512: pfas_ok)
    Card& c = E->cards[W.card];
    // review pf14 L1: Q8_0 shares, and the staging scales sized for them (GGUF mode)
    if (!c.stg_w[2] || !c.stg_s[2] || E->o.gguf_model.empty()) return false;
    for (int j = 0; j < 3; j++)
        if (!W.ash[j].wo_b.sq) return false;
    static std::map<std::pair<int, int>, bool> okn;
    auto f = okn.find({n, W.card});
    if (f != okn.end()) return f->second;
    // the full matrix bf16 in the owner's attention region (q, o: dead after its share), below the prefetch slots
    static_assert((size_t)D * OG * OLR * 2 <= (size_t)2 * 512 * 32768 * 4, "owner-side wo_b: bf16 matrix over q / o");
    bool ok = true;
    const int slb = std::max<int>(32, (int)std::min<size_t>(D, c.deq_elems / (OG * OLR)) / 32 * 32);
    F8 full = W.ash[W.card].wo_b;
    full.N = D;
    DevGuard dg(c.dev);
    for (int m : {std::min(n, 512), n % 512})
        if (m > 0) ok = ok && pfas_gemm(c, full, nullptr, 0, D, slb, nullptr, OG * OLR, m, nullptr, D, true);
    if (!ok) logf_("prefill owner-side wo_b: off for %d rows (cublasLt check)\n", n);
    okn[{n, W.card}] = ok;
    return ok;
}
static void attn_pfsplit(Engine* E, Card& c, const Layer& W, int l, int s, int n, int nsel, int kind, bool pf) {
    const int oi = W.card, SQ = 512;
    const int p0 = std::max(0, s - (WIN - 1));
    const SrcCache* scs[3] = {};
    // the unsplit linq's row slices (owner's dequant scratch)
    const int slq = std::max<int>(32, (int)std::min<size_t>(NH * HD, c.deq_elems / QLR) / 32 * 32);
    const int slb = std::max<int>(32, (int)std::min<size_t>(D, c.deq_elems / (OG * OLR)) / 32 * 32);
    bf16* xqo = (bf16*)c.ao;   // owner: bf16 qr in c.ao (free until wo_b)
    k_to_bf16s<<<cdiv((size_t)n * QLR, 256), 256, 0, c.st>>>(c.qr, QLR, n, QLR, xqo, QLR);
    DCK(cudaEventRecord(c.ev_as[0], c.st));
    if (g_pfx_defer) pfx_predict(E, c, W, n);   // [dsv4-pf4-pfxd]
    g_pfx_defer = false;
    if (c.ev_stg) DCK(cudaEventRecord(c.ev_stg, c.st));   // [dsv4-pf3-pfx]: its copies start now (no staging)
    if (pf) pf_rec(E, W.card, 0, c.st, l, PK_STG);
    int lidx = l;
    while (lidx > 0 && !is_idx_src(lidx)) lidx--;
    const long tag = (long)E->nfwd * 64 + lidx;
    c.sel_tag = tag;
    const size_t kwo = ((size_t)n * QLR + 127) / 128 * 128;   // remote: window rows after the bf16 qr in k.qr
    const bool x1b = !E_pfas1 && !E_x1b_off && (n + SQ - 1) / SQ <= 64;
    const bool owb = x1b && pfwob_ok(E, W, n);   // [dsv4-pf14-owb]
    if (owb) {   // the helpers' Q8_0 wo_b rows into the owner's wo_b staging (its inbound link is idle during X1)
        DCK(cudaSetDevice(c.dev));
        DCK(cudaStreamWaitEvent(c.ass, c.ev_as[0], 0));   // after this card's earlier reads of the staging (its stream)
        const int K = OG * OLR;
        for (int j = 0; j < 3; j++) {
            if (j == oi) continue;
            const F8& B = W.ash[j].wo_b;
            DCK(cudaMemcpyPeerAsync(c.stg_w[2] + (size_t)W.ash[j].n0 * K, c.dev, B.w, E->cards[j].dev, (size_t)B.N * K, c.ass));
            DCK(cudaMemcpyPeerAsync(c.stg_s[2] + (size_t)W.ash[j].n0 * (K / 32) * 2, c.dev, B.sq, E->cards[j].dev,
                                    (size_t)B.N * (K / 32) * 2, c.ass));
        }
        DCK(cudaEventRecord(c.ev_as[2], c.ass));
        if (c.stg_l != l) c.stg_l = -1;   // the staging no longer holds another layer's full matrices
    }
    for (int j = 0; j < 3; j++) {
        Card& k = E->cards[j];
        scs[j] = W.ratio ? &k.src[src_index(kv_src_of(l))] : nullptr;
        if (j == oi) continue;
        DevGuard dg(k.dev);
        if (x1b) {   // [dsv4-pf5-x1b] X1 by sub-block on the exchange stream: qr rows, the window rows up to the block's end,
            // the selection rows; after this card's earlier reads of k.qr / k.sel (its compute stream)
            DCK(cudaEventRecord(k.ev_x1[64], k.st));
            DCK(cudaStreamWaitEvent(k.ass, k.ev_x1[64], 0));
            DCK(cudaStreamWaitEvent(k.ass, c.ev_as[0], 0));
            const bool sl = nsel && k.sel_tag != tag;
            for (int b0 = 0, b = 0; b0 < n; b0 += SQ, b++) {
                const int m = std::min(SQ, n - b0);
                DCK(cudaMemcpyPeerAsync((bf16*)k.qr + (size_t)b0 * QLR, k.dev, xqo + (size_t)b0 * QLR, c.dev, (size_t)m * QLR * 2, k.ass));
                const int lo = b ? s + b0 : p0, hi = s + b0 + m;   // window positions [lo, hi)
                DCK(cudaMemcpyPeerAsync((bf16*)k.qr + kwo + (size_t)(lo - p0) * HD, k.dev, kv_pf(W) + (size_t)lo * HD, c.dev,
                                        (size_t)(hi - lo) * HD * 2, k.ass));
                if (sl)
                    DCK(cudaMemcpyPeerAsync(k.sel + (size_t)b0 * ITOPK, k.dev, c.sel + (size_t)b0 * ITOPK, c.dev,
                                            (size_t)m * ITOPK * 4, k.ass));
                DCK(cudaEventRecord(k.ev_x1[b], k.ass));
            }
            if (sl) k.sel_tag = tag;
            continue;
        }
        DCK(cudaStreamWaitEvent(k.st, c.ev_as[0], 0));
        DCK(cudaMemcpyPeerAsync(k.qr, k.dev, xqo, c.dev, (size_t)n * QLR * 2, k.st));
        DCK(cudaMemcpyPeerAsync((bf16*)k.qr + kwo, k.dev, kv_pf(W) + (size_t)p0 * HD, c.dev, (size_t)(s + n - p0) * HD * 2, k.st));
        if (nsel && k.sel_tag != tag) {
            DCK(cudaMemcpyPeerAsync(k.sel, k.dev, c.sel, c.dev, (size_t)n * ITOPK * 4, k.st));
            k.sel_tag = tag;
        }
    }
    int ord[3] = {-1, -1, oi}, no = 0;   // the other cards first (theirs is the larger share)
    for (int j = 0; j < 3; j++)
        if (j != oi) ord[no++] = j;
    for (int jj = 0; jj < 3; jj++) {
        const int j = ord[jj];
        Card& k = E->cards[j];
        DevGuard dg(k.dev);
        const Layer::AShare& A = W.ash[j];
        const int h0 = A.g0 * AT_HG, nh = A.ng * AT_HG;
        const bool own = j == oi;
        const bf16* xq = own ? xqo : (const bf16*)k.qr;
        const bf16* kvp = own ? kv_pf(W) : (const bf16*)k.qr + kwo - (size_t)p0 * HD;
        const int* selp = own ? c.sel : k.sel;
        float* q = (float*)k.phase;
        float* o = q + (size_t)SQ * NH * HD;
        float* o2 = o + (size_t)SQ * NH * HD;
        bf16* ob = (bf16*)q;   // wo_a input (bf16 of o) over the dead q, the unsplit layout [m][NH * HD]
        // review pf4 L7: the share's wq_b and wo_a dequantized once per layer (15.7 M + 12.6 M of the 33.5 M deq elements)
        bf16* wqd = k.deq;
        bf16* wad = k.deq + ((size_t)A.wq_b.N * A.wq_b.K + 127) / 128 * 128;
        DCHECK((size_t)(wad - k.deq) + (size_t)A.wo_a.N * A.wo_a.K <= k.deq_elems, "attention split: dequant scratch");
        if (pf && !own && !x1b) pf_rec(E, j, 0, k.st, l, PK_RWAIT, PK_STG);   // [dsv4-pf5-bal] X1 landed
        pfas_deq(k, A.wq_b, wqd);
        pfas_deq(k, A.wo_a, wad);
        for (int b0 = 0; b0 < n; b0 += SQ) {
            const int m = std::min(SQ, n - b0);
            if (x1b && !own) {
                DCK(cudaStreamWaitEvent(k.st, k.ev_x1[b0 / SQ], 0));
                if (pf && b0 == 0) pf_rec(E, j, 0, k.st, l, PK_RWAIT, PK_STG);   // X1 block 0 landed (review pf5 L5)
            }
            pfas_gemm(k, A.wq_b, wqd, h0 * HD, NH * HD, slq, xq + (size_t)b0 * QLR, QLR, m, q, NH * HD);
            rope(k.st, q + (size_t)h0 * HD, NH * HD, m, nh, HD, k.cosv[kind], k.sinv[kind], s + b0, 1, 0);
            const int* sp = nsel ? selp + (size_t)b0 * ITOPK : nullptr;
            const bf16* ck = scs[j] ? scs[j]->ckv : nullptr;
            if (E_tc5_off && nh == 16)
                k_attn_tc3<1, 16><<<m, 256, tc3_smem<1, 16>(), k.st>>>(q, s + b0, kvp, ck, sp, nsel, ITOPK, A.sink, o, h0);
            else if (E_tc5_off)
                k_attn_tc3<2, 24><<<m, 256, tc3_smem<2, 24>(), k.st>>>(q, s + b0, kvp, ck, sp, nsel, ITOPK, A.sink, o, h0);
            else if (nh == 16)   // [dsv4-pf13-attnsrc] transposed core, bit-identical to k_attn_tc3
                k_attn_tc5<16><<<m, 256, tc5_smem<16>(), k.st>>>(q, s + b0, kvp, ck, sp, nsel, ITOPK, A.sink, o, h0);
            else
                k_attn_tc5<24><<<m, 256, tc5_smem<24>(), k.st>>>(q, s + b0, kvp, ck, sp, nsel, ITOPK, A.sink, o, h0);
            rope(k.st, o + (size_t)h0 * HD, NH * HD, m, nh, HD, k.cosv[kind], k.sinv[kind], s + b0, 1, 1);
            k_to_bf16s<<<cdiv((size_t)m * nh * HD, 256), 256, 0, k.st>>>(o + (size_t)h0 * HD, NH * HD, m, nh * HD,
                                                                       ob + (size_t)h0 * HD, NH * HD);
            for (int g = A.g0; g < A.g0 + A.ng; g++) {   // = linq's grouped call, group by group (slice = the group)
                const int K = A.wo_a.K;
                gemm_bf16(k.blas, ob + (size_t)g * K, NH * HD, wad + (size_t)(g - A.g0) * OLR * K, K, o2 + (size_t)g * OLR,
                          OG * OLR, m, OLR, K);
            }
            k_to_bf16s<<<cdiv((size_t)m * A.ng * OLR, 256), 256, 0, k.st>>>(o2 + (size_t)A.g0 * OLR, OG * OLR, m, A.ng * OLR,
                                                                          k.xb + (size_t)b0 * OG * OLR + (size_t)A.g0 * OLR, OG * OLR);
            if (!E_pfas1) DCK(cudaEventRecord(k.ev_ab[3 * (b0 / SQ)], k.st));
        }
        DCK(cudaEventRecord(k.ev_as[1], k.st));
        if (pf && !own) pf_rec(E, j, 0, k.st, l, PK_REXP, PK_ATT);
    }
    if (owb) {   // [dsv4-pf14-owb] only the owner gathers o2 (by sub-block) and runs the whole wo_b into c.ao
        const int nb = (n + SQ - 1) / SQ, K = OG * OLR;
        DCK(cudaSetDevice(c.dev));
        for (int b = 0; b < nb; b++) {
            const int b0 = b * SQ, m = std::min(SQ, n - b0);
            for (int i = 0; i < 3; i++) {
                if (i == oi) continue;
                const Layer::AShare& B = W.ash[i];
                DCK(cudaStreamWaitEvent(c.ass, E->cards[i].ev_ab[3 * b], 0));
                const size_t off = (size_t)b0 * K + (size_t)B.g0 * OLR;
                DCK(cudaMemcpy2DAsync(c.xb + off, (size_t)K * 2, E->cards[i].xb + off, (size_t)K * 2, (size_t)B.ng * OLR * 2, m,
                                      cudaMemcpyDefault, c.ass));
            }
            DCK(cudaEventRecord(c.ev_ab[3 * b + 1], c.ass));
        }
        if (pf) pf_rec(E, W.card, 0, c.st, l, PK_WOA);
        for (int j = 0; j < 3; j++) {   // a helper's o2 columns stay until the owner pulled them (k.xb is reused after)
            if (j == oi) continue;
            Card& k = E->cards[j];
            DevGuard dg(k.dev);
            DCK(cudaStreamWaitEvent(k.st, c.ev_ab[3 * (nb - 1) + 1], 0));
            if (pf) pf_rec(E, j, 0, k.st, l, PK_REXP, PK_WOB);   // (review pf14 L4: owb = the wait for the owner's pulls only)
        }
        DCK(cudaSetDevice(c.dev));
        // the full wo_b bf16 over the owner's dead q / o (its share is done on this stream): own rows from its share first,
        // then the helpers' rows from the staging (review pf14 L5: the dequant overlaps the staging / X1 tails)
        bf16* wd = (bf16*)c.phase;
        for (int jj = 0; jj < 3; jj++) {
            const int j = jj == 0 ? oi : ord[jj - 1];
            const Layer::AShare& A = W.ash[j];
            if (jj == 1) DCK(cudaStreamWaitEvent(c.st, c.ev_as[2], 0));   // the helpers' rows landed
            const int8_t* w = j == oi ? (const int8_t*)A.wo_b.w : (const int8_t*)c.stg_w[2] + (size_t)A.n0 * K;
            const __half* sq = j == oi ? A.wo_b.sq : (const __half*)c.stg_s[2] + (size_t)A.n0 * (K / 32);
            k_deq_q8<<<cdiv((size_t)A.nn * K / 2, 256), 256, 0, c.st>>>(w, sq, K, 0, A.nn, wd + (size_t)A.n0 * K);
        }
        for (int i = 0; i < 3; i++)   // the others' X1 pulls read xqo (c.ao), which wo_b overwrites
            if (i != oi) DCK(cudaStreamWaitEvent(c.st, E->cards[i].ev_x1[(n - 1) / SQ], 0));
        F8 full = W.ash[oi].wo_b;
        full.N = D;
        for (int b = 0; b < nb; b++) {
            const int b0 = b * SQ, m = std::min(SQ, n - b0);
            DCK(cudaStreamWaitEvent(c.st, c.ev_ab[3 * b + 1], 0));
            // review pf14 L2: a skipped GEMM (heuristic failure after an algorithm-cache reset) must not pass silently
            DCHECK(pfas_gemm(c, full, wd, 0, D, slb, c.xb + (size_t)b0 * K, K, m, c.ao + (size_t)b0 * D, D),
                   "owner-side wo_b GEMM (%d rows)", m);
        }
        if (pf) pf_rec(E, W.card, 0, c.st, l, PK_WOB);
        return;
    }
    if (!E_pfas1) {   // v2: the exchanges per sub-block on each card's exchange stream, beside the shares / wo_b
        const int nb = (n + SQ - 1) / SQ;
        for (int jj = 0; jj < 3; jj++) {   // o2 all-gather by sub-block
            const int j = ord[jj];
            Card& k = E->cards[j];
            DevGuard dg(k.dev);
            for (int b = 0; b < nb; b++) {
                const int b0 = b * SQ, m = std::min(SQ, n - b0);
                for (int i = 0; i < 3; i++) {
                    if (i == j) continue;
                    const Layer::AShare& B = W.ash[i];
                    DCK(cudaStreamWaitEvent(k.ass, E->cards[i].ev_ab[3 * b], 0));
                    const size_t off = (size_t)b0 * OG * OLR + (size_t)B.g0 * OLR;
                    DCK(cudaMemcpy2DAsync(k.xb + off, (size_t)OG * OLR * 2, E->cards[i].xb + off, (size_t)OG * OLR * 2,
                                          (size_t)B.ng * OLR * 2, m, cudaMemcpyDefault, k.ass));
                }
                DCK(cudaEventRecord(k.ev_ab[3 * b + 1], k.ass));
            }
        }
        if (pf) pf_rec(E, W.card, 0, c.st, l, PK_WOA);
        for (int jj = 0; jj < 3; jj++) {   // wo_b by sub-block as its o2 rows arrive
            const int j = ord[jj];
            Card& k = E->cards[j];
            DevGuard dg(k.dev);
            const Layer::AShare& A = W.ash[j];
            if (j == oi && x1b)   // [dsv4-pf5-x1b] the others' X1 pulls read xqo (c.ao), which this wo_b overwrites
                for (int i = 0; i < 3; i++)
                    if (i != oi) DCK(cudaStreamWaitEvent(k.st, E->cards[i].ev_x1[(n - 1) / SQ], 0));
            pfas_deq(k, A.wo_b, k.deq);   // (after this card's wo_a on its stream)
            for (int b = 0; b < nb; b++) {
                const int b0 = b * SQ, m = std::min(SQ, n - b0);
                DCK(cudaStreamWaitEvent(k.st, k.ev_ab[3 * b + 1], 0));
                pfas_gemm(k, A.wo_b, k.deq, A.n0, D, slb, k.xb + (size_t)b0 * OG * OLR, OG * OLR, m, k.ao + (size_t)b0 * D, D);
                DCK(cudaEventRecord(k.ev_ab[3 * b + 2], k.st));
            }
            if (pf && j != oi) pf_rec(E, j, 0, k.st, l, PK_REXP, PK_WOB);
            for (int i = 0; i < 3; i++)   // the others' pulls of my o2 columns are done before my stream goes on
                if (i != j) DCK(cudaStreamWaitEvent(k.st, E->cards[i].ev_ab[3 * (nb - 1) + 1], 0));
        }
        DCK(cudaSetDevice(c.dev));
        // review pf4 H1: the pulls overwrite c.ao, which holds the owner's bf16 qr (xqo) until its last sub-block's wq_b
        DCK(cudaStreamWaitEvent(c.ass, c.ev_as[1], 0));
        if (x1b)   // review pf5 H1: the others' X1 pulls read xqo (c.ao) until their last block; the ao pulls overwrite it
            for (int i = 0; i < 3; i++)
                if (i != oi) DCK(cudaStreamWaitEvent(c.ass, E->cards[i].ev_x1[(n - 1) / SQ], 0));
        for (int b = 0; b < nb; b++) {   // the other cards' wo_b rows into c.ao by sub-block
            const int b0 = b * SQ, m = std::min(SQ, n - b0);
            for (int j = 0; j < 3; j++) {
                if (j == oi) continue;
                const Layer::AShare& B = W.ash[j];
                DCK(cudaStreamWaitEvent(c.ass, E->cards[j].ev_ab[3 * b + 2], 0));
                DCK(cudaMemcpy2DAsync(c.ao + (size_t)b0 * D + B.n0, (size_t)D * 4, E->cards[j].ao + (size_t)b0 * D + B.n0,
                                      (size_t)D * 4, (size_t)B.nn * 4, m, cudaMemcpyDefault, c.ass));
            }
        }
        DCK(cudaEventRecord(c.ev_as[3], c.ass));
        DCK(cudaStreamWaitEvent(c.st, c.ev_as[3], 0));
        if (pf) pf_rec(E, W.card, 0, c.st, l, PK_WOB);
        return;
    }
    if (pf) pf_rec(E, W.card, 0, c.st, l, PK_ATT);
    for (int jj = 0; jj < 3; jj++) {   // o2 all-gather (each card pulls the others' columns)
        const int j = ord[jj];
        Card& k = E->cards[j];
        DevGuard dg(k.dev);
        for (int i = 0; i < 3; i++) {
            if (i == j) continue;
            const Layer::AShare& B = W.ash[i];
            DCK(cudaStreamWaitEvent(k.st, E->cards[i].ev_as[1], 0));
            DCK(cudaMemcpy2DAsync(k.xb + (size_t)B.g0 * OLR, (size_t)OG * OLR * 2, E->cards[i].xb + (size_t)B.g0 * OLR,
                                  (size_t)OG * OLR * 2, (size_t)B.ng * OLR * 2, n, cudaMemcpyDefault, k.st));
        }
        DCK(cudaEventRecord(k.ev_as[2], k.st));
    }
    if (pf) pf_rec(E, W.card, 0, c.st, l, PK_WOA);
    for (int jj = 0; jj < 3; jj++) {
        const int j = ord[jj];
        Card& k = E->cards[j];
        DevGuard dg(k.dev);
        const Layer::AShare& A = W.ash[j];
        pfas_deq(k, A.wo_b, k.deq);
        for (int b0 = 0; b0 < n; b0 += SQ) {
            const int m = std::min(SQ, n - b0);
            pfas_gemm(k, A.wo_b, k.deq, A.n0, D, slb, k.xb + (size_t)b0 * OG * OLR, OG * OLR, m, k.ao + (size_t)b0 * D, D);
        }
        for (int i = 0; i < 3; i++)   // the others' pulls of my o2 columns are done before my stream goes on
            if (i != j) DCK(cudaStreamWaitEvent(k.st, E->cards[i].ev_as[2], 0));
        DCK(cudaEventRecord(k.ev_as[3], k.st));
    }
    DCK(cudaSetDevice(c.dev));
    for (int j = 0; j < 3; j++) {
        if (j == oi) continue;
        const Layer::AShare& B = W.ash[j];
        DCK(cudaStreamWaitEvent(c.st, E->cards[j].ev_as[3], 0));
        DCK(cudaMemcpy2DAsync(c.ao + B.n0, (size_t)D * 4, E->cards[j].ao + B.n0, (size_t)D * 4, (size_t)B.nn * 4, n,
                              cudaMemcpyDefault, c.st));
    }
    if (pf) pf_rec(E, W.card, 0, c.st, l, PK_WOB);
}

// the attention sub-block of layer l for tokens s..s+n-1 (normed input in c.x) -> c.ao [n][5120]
static void attention(Engine* E, Card& c, const Layer& W, int l, int s, int n, int* selk) {
    cudaStream_t st = c.st;
    // review r15 L2: the bf16 qr only where attn_split reads it (same condition as its call below)
    const bool qb = qrb_on() && E->asplit && n <= MAXR && !W.dspark;
    const int kind = W.ratio ? 1 : 0;
    // [dsv4-rel2-p1] window rows: a prefill call addresses the scratch by absolute position, a pass (<= MAXR rows) the
    // ring (row p & kmask)
    DCHECK(!W.dspark || n <= MAXR, "drafter stage with %d rows", n);
    bf16* const kvb = n > MAXR ? kv_pf(W) : W.kvw;
    const int kvm = n > MAXR ? -1 : W.kmask;
    // q low rank; decode (R8): x quantized once for wq_a and wkv (both quantized it in every GEMV block: 25 + 21 us
    // per layer for 9 MB of weights)
    const bf16* xq1 = nullptr;
    if (n <= MAXR && !E_qkv1_off && !W.wq_a.si) {   // R23 (review r22 L2): the int8 drafter linears quantize x themselves
        const bool din = !W.dspark && l > 0 && defer_on(E, l - 1, n);   // R21: layer_fwd ran the unfused boundary
        if (din || (!(n == 1 && !E_hcf_off) && !(hcfr_on(n) && !W.dspark)))   // (R10 / R20: the fused mHC kernel wrote it; the
            // drafter's stages run the unfused boundaries)
            q_fp8(st, c.x, D, n, D, nullptr, 0, c.xb + (size_t)MAXR * 32768, D);   // past the region linq's own q_fp8 uses
        xq1 = c.xb + (size_t)MAXR * 32768;
    }
    if (xq1 && W.wqkv.w && !E_kvt_off && !E_wqkv_off) {   // R10: wq_a + wkv as one GEMV (same rows, bit-identical)
        // review r10 H1: past linq's own x scratch (c.f32tmp[0 .. n*5120) for n >= 5 rows or E_fuse_off)
        float* y = c.f32tmp + (size_t)MAXR * D;   // [n][1792]
        linq(c, W.wqkv, c.x, D, n, y, QLR + HD, 0, xq1);
        // q norm (blocks n ..) and the window-KV tail (blocks 0 .. n - 1) in one launch
        k_kv_tail<<<2 * n, 256, 0, st>>>(y + QLR, QLR + HD, W.kv_norm, c.cosv[kind], c.sinv[kind], s, kvb,
                                         n, y, W.q_norm, c.qr, qb ? c.qrb : nullptr, s, kvm);
    } else {
    linq(c, W.wq_a, c.x, D, n, c.qr, QLR, 0, xq1);
    rms(st, c.qr, QLR, W.q_norm, c.qr, QLR, n, QLR);
    if (qb) q_fp8(st, c.qr, QLR, n, QLR, nullptr, 0, c.qrb, QLR);   // R15: the bf16 qr for wq_b / X1
    // window kv (FP8-rounded, bf16 exact) for all n
    {
        float* kv = c.ao;   // temporary [n][512] (ao is written at the end)
        linq(c, W.wkv, c.x, D, n, kv, HD, 0, xq1);
        if (n <= MAXR && !E_kvt_off)   // R10: rms + rope + q_fp8 in one launch (bit-identical)
            k_kv_tail<<<n, 256, 0, st>>>(kv, HD, W.kv_norm, c.cosv[kind], c.sinv[kind], s, kvb, 0, nullptr, nullptr,
                                         nullptr, nullptr, s, kvm);
        else {
        rms(st, kv, HD, W.kv_norm, kv, HD, n, HD);
        rope(st, kv, HD, n, 1, HD, c.cosv[kind], c.sinv[kind], s, 1, 0);
        q_fp8(st, kv, HD, n, HD, nullptr, 0, kvb, HD, s, kvm);
        }
    }
    }
    if (n <= MAXR) PP(l, 3, st);
    const bool pf = n > MAXR && !W.dspark && pf_on();   // [dsv4-pf-pprof]
    if (pf) pf_rec(E, W.card, 0, st, l, PK_QKV);
    if (W.dspark) DPT("att q_a/kv + norms", st);
    const SrcCache* sc = nullptr;
    if (W.ratio) {
        int src = kv_src_of(l);
        if (is_kv_src(l)) compressor(E, c, W, l, s, n);
        if (is_idx_src(l)) indexer(E, c, W, l, s, n, selk);
        sc = &c.src[src_index(src)];
    }
    if (n <= MAXR) PP(l, 4, st);
    if (pf) pf_rec(E, W.card, 0, st, l, PK_CMPIDX);
    const int nsel = W.ratio ? *selk : 0;
    if (E->asplit && n <= MAXR && !W.dspark) {
        attn_split(E, c, W, l, s, n, nsel, kind);
        if (n <= MAXR) PP(l, 5, st);
        if (n <= MAXR) PP(l, 6, st);
        return;
    }
    if (n > MAXR && pfas_ok(E, W, n)) {   // [dsv4-pf4-asplit]
        attn_pfsplit(E, c, W, l, s, n, nsel, kind, pf);
        return;
    }
    // R8 split: prefill gathers the full matrices of this layer from the three cards' shares into the staging buffer
    F8 wq_b = W.wq_b, wo_a = W.wo_a, wo_b = W.wo_b;
    if (E->asplit && !W.dspark) {
        F8* f[3] = {&wq_b, &wo_a, &wo_b};
        if (c.stg_l != l) {
            for (int j = 0; j < (int)E->cards.size(); j++) {
                const Layer::AShare& A = W.ash[j];
                const F8* sl[3] = {&A.wq_b, &A.wo_a, &A.wo_b};
                const int r0[3] = {A.g0 * AT_HG * HD, A.g0 * OLR, A.n0};
                for (int i = 0; i < 3; i++) {
                    const int K = f[i]->K;
                    DCK(cudaMemcpyPeerAsync(c.stg_w[i] + (size_t)r0[i] * K, c.dev, sl[i]->w, E->cards[j].dev,
                                            (size_t)sl[i]->N * K, st));
                    if (sl[i]->sq)   // [dsv4-rel-q8]
                        DCK(cudaMemcpyPeerAsync(c.stg_s[i] + (size_t)r0[i] * (K / 32) * 2, c.dev, sl[i]->sq, E->cards[j].dev,
                                                (size_t)sl[i]->N * (K / 32) * 2, st));
                    else
                        DCK(cudaMemcpyPeerAsync(c.stg_s[i] + (size_t)(r0[i] / 32) * (K / 32), c.dev, sl[i]->s, E->cards[j].dev,
                                                (size_t)(sl[i]->N / 32) * (K / 32), st));
                }
            }
            c.stg_l = l;
        }
        if (c.ev_stg) DCK(cudaEventRecord(c.ev_stg, st));   // [dsv4-pf3-pfx]
        if (pf) pf_rec(E, W.card, 0, st, l, PK_STG);
        for (int i = 0; i < 3; i++) {
            f[i]->w = c.stg_w[i];
            const F8* sh[3] = {&W.ash[0].wq_b, &W.ash[0].wo_a, &W.ash[0].wo_b};   // (Q8_0 or FP8: as the shares)
            if (sh[i]->sq) f[i]->sq = (const __half*)c.stg_s[i];
            else f[i]->s = c.stg_s[i];
        }
    }
    // attention in sub-blocks of queries
    const int SQ = 512;
    float* q = (float*)c.phase;
    float* o = q + (size_t)SQ * NH * HD;
    float* o2 = o + (size_t)SQ * NH * HD;
    for (int b0 = 0; b0 < n; b0 += SQ) {
        int m = std::min(SQ, n - b0);
        linq(c, wq_b, c.qr + (size_t)b0 * QLR, QLR, m, q, NH * HD);
        rope(st, q, NH * HD, m, NH, HD, c.cosv[kind], c.sinv[kind], s + b0, 1, 0);
        if (pf) pf_rec(E, W.card, 0, st, l, PK_WQB);
        if (W.dspark) DPT("att wq_b + rope", st);
        int nsplit = 1, nk_a = 1 << 30;   // nk_a [dsv4-dec3-atts]: an upper bound on every row's keys (unknown: k_attn)
        const int kend = W.dspark ? s + n - 1 : -1, wlen = W.dspark ? WIN + n : WIN;
        if (m <= MAXR) {   // grid split of the last row (the largest); each row uses its own inside the kernel
            int nk = (W.dspark ? std::min(wlen, kend + 1) : std::min(WIN, s + b0 + m)) + nsel;
            nsplit = std::max(1, std::min(32, (nk + AT_KT - 1) / AT_KT));
            nk_a = nk;
        }
        if (m > MAXR && E->o.attn_tc) {
            if (E_attn2_off)
                k_attn_tc<<<m, 256, TC_SMEM, st>>>(q, s + b0, kvb, sc ? sc->ckv : nullptr,
                                               nsel ? c.sel + (size_t)b0 * ITOPK : nullptr, nsel, ITOPK, W.sink, o);
            else   // [dsv4-pf-attn2]: the same output, 2.3x faster (bank-conflict-free fragments, ldmatrix.trans for V)
                k_attn_tc2<<<m, 256, TC2_SMEM, st>>>(q, s + b0, kvb, sc ? sc->ckv : nullptr,
                                               nsel ? c.sel + (size_t)b0 * ITOPK : nullptr, nsel, ITOPK, W.sink, o);
        } else
        attn_kern(nk_a, nsplit)<<<dim3(m, NH / AT_HG, nsplit), 256, AT_SMEM, st>>>(
            q, s + b0, kvb, sc ? sc->ckv : nullptr, nsel ? c.sel + (size_t)b0 * ITOPK : nullptr, nsel, ITOPK, W.sink, o,
            nsplit, c.part, W.ratio, kend, wlen, 0, kvm, nullptr, nullptr, nullptr);   // (the kernel's defaults: a pointer launch)
        if (nsplit > 1) k_attn_comb<<<dim3(m, NH), 256, 0, st>>>(c.part, nsplit, W.sink, o);
        rope(st, o, NH * HD, m, NH, HD, c.cosv[kind], c.sinv[kind], s + b0, 1, 1);
        if (pf) pf_rec(E, W.card, 0, st, l, PK_ATT);
        if (W.dspark) DPT("att core + rope", st);
        if (n <= MAXR) PP(l, 5, st);
        // o: q_fp8 inside linq over the grouped input; wo_a grouped (1024 rows per group, 4096 columns each)
        linq(c, wo_a, o, NH * HD, m, o2, OG * OLR, OLR);
        if (pf) pf_rec(E, W.card, 0, st, l, PK_WOA);
        if (W.dspark) DPT("att wo_a", st);
        linq(c, wo_b, o2, OG * OLR, m, c.ao + (size_t)b0 * D, D);
        if (pf) pf_rec(E, W.card, 0, st, l, PK_WOB);
        if (W.dspark) DPT("att wo_b", st);
    }
    if (n <= MAXR) PP(l, 6, st);
    // sel buffer note: rows [0, n) * ITOPK hold the selection used by the following non-index layers
    (void)nsel;
}

static void dset(Card& c, int* dtab, int idx, int val);
static void draft_window(Engine* E);
static uint8_t* slot_ptr(Engine* E, int g);

// MoE (prefill: all rows on the GPU; decode: hot on the GPU, cold on the CPU lane) for xm = c.x [n] -> c.ao
static void dyn_update(Engine* E, Card& c, Layer& W, const int* ids, int n);
// expert prediction (L3 prefetch) is on for this pass shape; must match the forward's predictor condition
static bool pred_on(const Engine* E, int R) {
    return E->o.prefetch > 0 && R <= PRED_ROWS && (R == 1 || E->o.prefetch_mr);
}
// R11 early prediction of layer l (computed on layer l-1's owner at its routing): one row, same card as l-1
// [dsv4-dec1-pfmr] DSV4_EARLYMR=1 (gate): the early prediction also for verify passes of 2..PRED_ROWS rows
static const bool g_early_mr = getenv("DSV4_EARLYMR") != nullptr;
static const bool E_pfcov = getenv("DSV4_PFCOV") != nullptr;   // [dsv4-dec1-pfmr] coverage / window counters (host only)
// [dsv4-dec1-plate] DSV4_PREDLATE=1: the same-layer predictor joins the main stream only after its input read (its
// GEMV + routing on the side stream run beside the FFN boundary and the router; own buffers); its outputs feed only the
// host prefetch, so the model output is unchanged. Not with the R19 micro-batch pipe (shared buffers)
static const bool g_plate = getenv("DSV4_PREDLATE") && atoi(getenv("DSV4_PREDLATE")) != 0;
static bool early_on(const Engine* E, int l, int R) {
    return E->o.pf_early > 0 && (R == 1 || (g_early_mr && R <= PRED_ROWS)) && pred_on(E, 1) && l > 0 && l < NL && E->L[l].card == E->L[l - 1].card;
}
// predicted cold experts of a pass, most rows first (an expert several rows share is read once and used by all)
static int pred_pick(Engine* E, Layer& W, int l, int R, CpuJob* xs, std::vector<int>& pref,
                     const volatile int* hp = nullptr, int nmax = -1);
// tier update after a layer's routing; in a DSpark verify pass it waits for the accepted length so that rejected
// draft rows never feed the scores or admissions (review r6 M1 / M2): spec_commit()
static void dyn_route(Engine* E, Card& c, Layer& W, int l, const int* ids, int R, const float* w) {
    int f[MAXR * TOPK];
    if (E->o.drop_w > 0) {   // R8 adaptive top-k: left-out choices (weight 0) are not uses (-1: skipped by dyn_update)
        for (int i = 0; i < R * TOPK; i++) f[i] = w[i] == 0.f ? -1 : ids[i];
        ids = f;
    }
    if (g_fut) {   // dsv4-fut: every row of the pass, the predictor's ids, the residency at this point
        std::lock_guard<std::mutex> lk(g_fut_mu);
        const int np = pred_on(E, R) && E->hpred ? R * TOPK : 0;
        auto& b = g_fut_buf;
        b.insert(b.end(), {1, l, R, E->pass_clock, E->dyn_clock, np});
        b.insert(b.end(), ids, ids + (size_t)R * TOPK);
        for (int i = 0; i < np; i++) b.push_back((int)E->hpred[l * PRED_STRIDE + i]);
        int32_t rs[NE / 32] = {}, pd[NE / 32] = {};
        for (int e = 0; e < NE; e++)
            if (W.h_slot[e] >= 0 || W.h_dslot[e] >= 0) rs[e >> 5] |= 1u << (e & 31);
        for (int pe : W.pend_e)
            if (pe >= 0) pd[pe >> 5] |= 1u << (pe & 31);
        b.insert(b.end(), rs, rs + NE / 32);
        b.insert(b.end(), pd, pd + NE / 32);
    }
    if (!E->use_cnt.empty()) {   // [dsv4-rel-gguf]: the learned expert profile (decode rows; saved now and then)
        auto& u = E->use_cnt[l];
        for (int i = 0; i < R * TOPK; i++)
            if (ids[i] >= 0 && ids[i] < NE) u[ids[i]] += 1.0;
        static long n0 = 0;
        if (l == 0 && ++n0 % 8192 == 0) gm_profile_save(E);
    }
    if (E->spec_defer && E->mb_host) E->spec_ids[l].insert(E->spec_ids[l].end(), ids, ids + (size_t)R * TOPK);   // R19
    else if (E->spec_defer && (R > 1 || E->pipe_pass)) E->spec_ids[l].assign(ids, ids + (size_t)R * TOPK);   // review r19 L8
    else dyn_update(E, c, W, ids, R * TOPK);
}
static void spec_commit(Engine* E, int rows) {
    for (int l = 0; l < NL; l++) {
        auto& v = E->spec_ids[l];
        if (v.empty()) continue;
        Layer& W = E->L[l];
        Card& c = E->cards[W.card];
        DCK(cudaSetDevice(c.dev));
        dyn_update(E, c, W, v.data(), std::min((int)v.size(), rows * TOPK));
        v.clear();
    }
}
// review r21 L8: the buffer goes to the current file once it holds 64 M words (256 MB), at a commit (whole records,
// a pass never splits); the file of one reset period is PREFIX_<k>.bin, appended until the next reset / destroy
static constexpr size_t FUT_FLUSH_WORDS = (size_t)64 << 20;
static bool g_fut_started = false;   // the current file has been written once (later writes append)
static void fut_write_locked() {   // g_fut_mu held
    if (g_fut_buf.empty()) return;
    char nm[512];
    snprintf(nm, sizeof nm, "%s_%d.bin", g_fut, g_fut_k);
    FILE* f = fopen(nm, g_fut_started ? "ab" : "wb");
    if (!f) {
        logf_("warning: future records: cannot open %s (%s): %zu words dropped\n", nm, strerror(errno), g_fut_buf.size());
    } else {
        const size_t w = fwrite(g_fut_buf.data(), 4, g_fut_buf.size(), f);
        if (fclose(f) != 0 || w != g_fut_buf.size())
            logf_("warning: future records: short write to %s (%zu of %zu words)\n", nm, w, g_fut_buf.size());
        else
            logf_("future records: %zu words -> %s\n", g_fut_buf.size(), nm);
        g_fut_started = true;
    }
    g_fut_buf.clear();
}
static void fut_commit(Engine* E, int R, int na, const int* rt, const int* at, int lk) {
    if (!g_fut) return;
    std::lock_guard<std::mutex> g(g_fut_mu);
    auto& b = g_fut_buf;
    b.insert(b.end(), {2, R, na, E->pass_clock, E->dyn_clock, E->pos, lk});
    b.insert(b.end(), rt, rt + R);
    b.insert(b.end(), at, at + na);
    if (b.size() >= FUT_FLUSH_WORDS) fut_write_locked();
}
static void fut_dump() {
    if (!g_fut) return;
    std::lock_guard<std::mutex> g(g_fut_mu);
    fut_write_locked();
    if (g_fut_started) g_fut_k++;
    g_fut_started = false;
}
// [dsv4-pf-ilv] staged / resident interleave: a card's processing order in which the staged experts keep their order
// (the 3-buffer copy ring hands them out in order) and resident ones run while the next staged copy is still in
// flight (before: all residents first, the links idle behind a full ring). A timing model only: the order never
// changes a sum (remote cards write per-expert rows; the owner adds in `order`), so the result is the same.
static const bool E_ilv_off = getenv("DSV4_NOILV") != nullptr;
// [dsv4-pf-mtq] the remote cards' prefill expert work enqueued by helper threads (the one host thread fed ~12 API
// calls per expert to all 3 cards: ~25-30 ms per layer, as long as a card's expert math); DSV4_NOMTQ=1: one thread
static const bool E_mtq_off = getenv("DSV4_NOMTQ") != nullptr;
static const bool E_xqh_off = getenv("DSV4_NOXQH") != nullptr;   // [dsv4-pf8-xqh] (in moe_prefill)
static const bool E_xqh_ew = getenv("DSV4_XQH_EW") != nullptr;
static const bool E_xqh_pull = getenv("DSV4_XQH_PULL") != nullptr;   // the helpers pull the packed rows after c.ev (any n)
static const bool E_hsum_off = getenv("DSV4_NOHSUM") != nullptr;   // [dsv4-pf9-hsum] (in moe_prefill)
// the share of a helper's rows (its last experts) sent as rows after the dense sum (hides the sum's read at the tail), for
// chunks of < DSV4_HSUM_TMAXN rows (default 4096); larger chunks: dense sum only (gitlog 2nd net 1.642 / 1.664 s at tail 0,
// 1.612 / 1.629 at 0.12, 1.644 at 0.25; 10K 4.934 s at 0, 5.356 at 0.12, 5.542 at 0.25)
// [dsv4-pf15-hsred] the two helpers' dense sums combined on the helpers before the owner reads them: rows [0, n/2) on
// the second helper (S_b += S_a), rows [n/2, n) on the first (S_a += S_b), by chunks on their exchange streams, so the
// owner's inbound link carries one n x D fp32 sum instead of two (10K: 2 x 205 MB = ~16 ms per layer after the helpers
// end). Order-only: the owner adds (S_a + S_b) once instead of S_a, then S_b. DSV4_NOHSRED=1 off; DSV4_HSRED_NCH chunks
// per half (8)
static const bool E_hsred_off = getenv("DSV4_NOHSRED") != nullptr;
static const int E_hsred_nch = std::clamp(getenv("DSV4_HSRED_NCH") ? atoi(getenv("DSV4_HSRED_NCH")) : 8, 1, 64);
static const double E_hsum_tail = std::clamp(getenv("DSV4_HSUM_TAIL") ? atof(getenv("DSV4_HSUM_TAIL")) : 0.12, 0.0, 0.9);
static const int E_hsum_tmaxn = getenv("DSV4_HSUM_TMAXN") ? atoi(getenv("DSV4_HSUM_TMAXN")) : 4096;
static const bool E_lb_off = getenv("DSV4_NOLB") != nullptr;   // [dsv4-pf-lb] off: host experts balanced by bytes (R11)
static const cpu_set_t g_init_aff = [] {   // review pf M2: the allowed CPUs at load time (before the lane pins threads)
    cpu_set_t a;
    CPU_ZERO(&a);
    if (sched_getaffinity(0, sizeof a, &a) != 0) for (int k = 0; k < (int)sysconf(_SC_NPROCESSORS_ONLN); k++) CPU_SET(k, &a);
    return a;
}();
static void pf_helper_pin(const Engine* E, int i) {
    // review pf M2: a CPU of the allowed set outside the lane (CPUs 0 .. cpu_threads - 1) and not the enqueue thread's
    // (the last allowed one), counted from the top; none -> the allowed set minus the lane (or minus CPU 0)
    const cpu_set_t al = g_init_aff;   // (the caller is pinned to CPU 0: the process's mask from before any pinning)
    const int t = E->o.cpu_threads;
    std::vector<int> fr;
    for (int k = CPU_SETSIZE - 1; k >= 0; k--)
        if (CPU_ISSET(k, &al) && k >= t) fr.push_back(k);
    cpu_set_t cs;
    CPU_ZERO(&cs);
    if ((int)fr.size() > 1 + i) CPU_SET(fr[1 + i], &cs);   // fr[0]: the enqueue thread's CPU
    else if (!fr.empty()) for (int k : fr) CPU_SET(k, &cs);
    else for (int k = 1; k < CPU_SETSIZE; k++) if (CPU_ISSET(k, &al)) CPU_SET(k, &cs);
    if (CPU_COUNT(&cs) == 0 || pthread_setaffinity_np(pthread_self(), sizeof(cs), &cs) != 0) {
        static std::atomic<bool> said{false};
        if (!said.exchange(true)) logf_("warning: prefill helper thread pinning failed (it runs with the caller's mask)\n");
    }
}
struct IlvModel {
    static constexpr double LBW = 25.0e6;   // link bytes per ms (25 GB/s measured while busy)
    static constexpr double DQ = 0.12;      // ms from an expert's start to its stage buffer free (dequant)
    static double comp(int m) { return 0.25 + 0.0008 * m; }   // ms per expert (run B samples, deq4)
    // review pf2 M1: a fused expert (fx_test: 52 us at m <= 8, 66 at 18, 99-115 at 33-64, 199-221 at 95-128) holds
    // its stage buffer until k_fx2 ends (stage free = its whole compute)
    static double comp_fx(int m) { return 0.05 + 0.0013 * m; }
    static bool fx(int m, XFmt f) { return !E_fx_off && m <= E_fx_max && fx_ok(f); }
    static double cost(int m, XFmt f) { return fx(m, f) ? comp_fx(m) : comp(m); }
    static double dq(int m, XFmt f) { return fx(m, f) ? comp_fx(m) : DQ; }
};
// rows[i] = the item's rows, fm[i] = its expert format (cost model only)
static std::vector<int> ilv_plan(const std::vector<char>& sg, const std::vector<size_t>& bytes, const std::vector<int>& rows,
                                 const std::vector<XFmt>& fm, double g0) {
    std::vector<int> S, R, out;
    for (int i = 0; i < (int)sg.size(); i++) (sg[i] ? S : R).push_back(i);
    std::vector<double> ready(S.size(), 0.0);
    double lf = 0;
    for (size_t j = 0; j < std::min<size_t>(NST, S.size()); j++) { lf += bytes[S[j]] / IlvModel::LBW; ready[j] = lf; }
    double g = g0;
    size_t js = 0, jr = 0;
    while (js < S.size() || jr < R.size()) {
        if (js < S.size() && (jr >= R.size() || ready[js] <= g)) {
            const int i = S[js];
            const double cs = std::max(g, ready[js]);
            g = cs + IlvModel::cost(rows[i], fm[i]);
            if (js + NST < S.size()) {
                lf = std::max(lf, cs + IlvModel::dq(rows[i], fm[i])) + bytes[S[js + NST]] / IlvModel::LBW;
                ready[js + NST] = lf;
            }
            js++;
            out.push_back(i);
        } else {
            const int i = R[jr++];
            g += IlvModel::cost(rows[i], fm[i]);
            out.push_back(i);
        }
    }
    return out;
}
__global__ void k_pert_scale(float* x, size_t n, float f) {   // [dsv4-pf2] DSV4_PERT
    size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (t < n) x[t] *= f;
}
// [dsv4-pf3-pfx] prefill: host experts copied while the owner runs the attention (the links idle ~1.1 s of 2.4 s at 2.5K).
// A predictor (the decode predictor's form: the FFN norm on h with this layer's attention pre mix, bf16 GEMM with the
// gate, top-6) counts the rows per expert; the host waits for the counts after enqueuing the attention and copies the
// predicted host experts of >= DSV4_PFX_MIN rows (default 32 since [dsv4-pf12-pfxmin]; 9 before) into slots at the top
// of each card's phase region, above what this chunk uses there, and (chunks >= DSV4_PFXB_MIN rows) in the borrowed
// prefetch arena ([dsv4-pf12-pfxb]). moe_prefill keeps the lb model's CPU set (exact) and places
// a prefetched expert on its slot's card (the card never changes a sum). DSV4_NOPFX=1 off; DSV4_PFX_MAX slots per card
// (default 64); DSV4_PFX_NMAX: chunks of more rows run without it unless the borrowed arena exists (default 6144);
// DSV4_PFXLOG=1: hits per layer
static const bool E_pfx_off = getenv("DSV4_NOPFX") != nullptr;
static const int E_pfx_max = std::clamp(getenv("DSV4_PFX_MAX") ? atoi(getenv("DSV4_PFX_MAX")) : 64, 0, 512);
// [dsv4-pf12-pfxmin]: default 32 (was 9; 2.5K sweep 9 / 16 / 24 / 32 / 48: 2nd 1.615 / 1.603 / 1.572 / 1.554 / 1.594 s)
static const int E_pfx_min = getenv("DSV4_PFX_MIN") ? atoi(getenv("DSV4_PFX_MIN")) : 32;
static const int E_pfx_nmax = getenv("DSV4_PFX_NMAX") ? atoi(getenv("DSV4_PFX_NMAX")) : 6144;
static const bool E_pfx_log = getenv("DSV4_PFXLOG") != nullptr;
__global__ void k_pfx_count(const int* ids, int m, int* cnt) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t < m) atomicAdd(cnt + ids[t], 1);
}
static bool pfx_on(const Engine* E, int n) {   // [dsv4-pf12-pfxb]: above PFX_NMAX only with the borrowed arena
    return !E_pfx_off && E_pfx_max > 0 && n > MAXR && (n <= E_pfx_nmax || E->cards[0].pfa) && E->o.pdist && E->asplit && E->cards.size() == 3 &&
           !E->hostx && !E_lb_off && E->cards[0].big && !getenv("DSV4_PDCHK");
}
static uint8_t* pfx_slot(Card& k, size_t xb, int j) {   // [dsv4-pf12-pfxb]: slots past pf_np in the borrowed arena
    return j < k.pf_np ? (uint8_t*)k.phase + k.phase_bytes - (size_t)(j + 1) * xb : k.pfa + (size_t)(j - k.pf_np) * xb;
}
// phase bytes this chunk may use on card k during layer l (owner: the attention / indexer / Engram / MoE regions as laid
// out in create; another card: its distributed-prefill region for an estimated half of the routed rows; capacity is
// enforced in moe_prefill, which sends experts past it back to the owner)
static size_t pfx_bound(const Engine* E, int k, int l, int n) {
    const Layer& W = E->L[l];
    if (k == W.card) {
        const size_t att = (size_t)2 * 512 * 32768 * 4 + (size_t)512 * 8192 * 4;
        const size_t idx = idx_ws(E->o.maxctx);   // [dsv4-rel2-p1b]
        const size_t moe = (size_t)n * ((size_t)D * 2 + 2 * INTER * 4 + INTER * 4 + INTER * 2 + (size_t)D * 4);
        const size_t egb = (size_t)1024 * EG_ROWS * EG_HD * 4 + (size_t)1024 * (HC + 1) * D * 4;
        const size_t prd = (size_t)n * ((size_t)D * 6 + NE * 4 + TOPK * 8) + NE * 4 + 4096;
        return std::max({att + idx, egb, moe, prd, (size_t)n * D * 4 * 2}) + (16 << 20);
    }
    // [dsv4-pf4-asplit]: the attention split uses this card's attention region (q, o, o2) as the owner does (review pf4 L1:
    // only where it can run)
    const size_t att = pfas_may(E, l, n) ? (size_t)2 * 512 * 32768 * 4 + (size_t)512 * 8192 * 4 : 0;
    return std::max(att, (size_t)n * D * 2 + (size_t)n * TOPK / 2 * D * 4 + (size_t)n * (2 * INTER * 4 + INTER * 4 + INTER * 2)) +
           (16 << 20);
}
// on the owner's stream before the attention: counts of the predicted routing into E->h_pfcnt, then ev_pfp
static void pfx_predict(Engine* E, Card& c, const Layer& W, int n) {
    cudaStream_t st = c.st;
    float* xp = (float*)c.phase;                        // [n][D]
    bf16* xb = (bf16*)(xp + (size_t)n * D);             // [n][D]
    float* lg = (float*)(xb + (size_t)n * D);           // [n][NE]
    int* pid = (int*)(lg + (size_t)n * NE);             // [n][TOPK]
    float* pw = (float*)(pid + (size_t)n * TOPK);       // [n][TOPK]
    int* cnt = (int*)(pw + (size_t)n * TOPK);           // [NE]
    if (!E->h_pfcnt) DCK(cudaHostAlloc((void**)&E->h_pfcnt, NE * 4, cudaHostAllocPortable));
    if (!c.ev_pfp) DCK(cudaEventCreateWithFlags(&c.ev_pfp, cudaEventDisableTiming));
    if (!c.ev_stg) DCK(cudaEventCreateWithFlags(&c.ev_stg, cudaEventDisableTiming));
    k_hc_pre_rms<<<n, 1024, 0, st>>>(c.h, c.apre, W.ffn_norm, n, xp);
    k_to_bf16s<<<cdiv((size_t)n * D, 256), 256, 0, st>>>(xp, D, n, D, xb, D);
    gemm_bf16(c.blas, xb, D, W.gate_w, D, lg, NE, n, NE, D);
    k_route<<<cdiv((size_t)n * 32, 256), 256, 0, st>>>(lg, W.gate_b, n, pid, pw);
    DCK(cudaMemsetAsync(cnt, 0, NE * 4, st));
    k_pfx_count<<<cdiv((size_t)n * TOPK, 256), 256, 0, st>>>(pid, n * TOPK, cnt);
    DCK(cudaMemcpyAsync(E->h_pfcnt, cnt, NE * 4, cudaMemcpyDeviceToHost, st));
    DCK(cudaEventRecord(c.ev_pfp, st));
}
// slots each card may use for layer l (kc.pf_ns), events created; returns their sum (review pf3 L2: 0 = no predictor)
static int pfx_slots(Engine* E, int l, int n) {
    if (!E->pfx_xb) {   // slot size: the largest host expert of any layer (review pf3 M1: -1 = no host copy, skipped)
        for (int l2 = 0; l2 < NL; l2++)
            for (int e = 0; e < NE; e++)
                if (E->L[l2].hfmt[e] >= 0) E->pfx_xb = std::max(E->pfx_xb, expert_bytes((XFmt)E->L[l2].hfmt[e]));
        E->pfx_card.assign(NE, -1);
        E->pfx_j.assign(NE, -1);
    }
    if (!E->pfx_xb) return 0;
    int tot = 0;
    for (int k = 0; k < (int)E->cards.size(); k++) {
        Card& kc = E->cards[k];
        const size_t b = pfx_bound(E, k, l, n);
        kc.pf_np = kc.phase_bytes > b ? (int)std::min<size_t>(E_pfx_max, (kc.phase_bytes - b) / E->pfx_xb) : 0;
        kc.pf_ns = kc.pf_np + (kc.pfa ? (int)(kc.pfa_bytes / E->pfx_xb) : 0);
        if ((int)kc.pf_rdy.size() < kc.pf_ns) {
            DevGuard dg(kc.dev);
            while ((int)kc.pf_rdy.size() < std::max(E_pfx_max, kc.pf_ns)) {
                cudaEvent_t a, f;
                DCK(cudaEventCreateWithFlags(&a, cudaEventDisableTiming));
                DCK(cudaEventCreateWithFlags(&f, cudaEventDisableTiming));
                kc.pf_rdy.push_back(a);
                kc.pf_fre.push_back(f);
            }
        }
        tot += kc.pf_ns;
    }
    return tot;
}
// after the attention is enqueued: wait for the counts, then copy the predicted host experts into the cards' slots
// (kc.pf_ns from pfx_slots = capacity; afterwards the slots that hold an expert)
static void pfx_schedule(Engine* E, int l, int n) {
    (void)n;
    Layer& W = E->L[l];
    const int NC = (int)E->cards.size();
    std::fill(E->pfx_card.begin(), E->pfx_card.end(), -1);
    E->pfx_l = l;
    const size_t xb = E->pfx_xb;
    Card& c = E->cards[W.card];
    {
        DevGuard dg(c.dev);
        DCK(cudaEventSynchronize(c.ev_pfp));
    }
    const int* pc = E->h_pfcnt;
    std::vector<int> cand;
    for (int e = 0; e < NE; e++)   // (review pf3 M1: a host copy larger than the slot, e.g. a later FP4 swap copy, is skipped)
        if (pc[e] >= E_pfx_min && W.h_slot[e] < 0 && W.h_dslot[e] < 0 && (W.bor.empty() || !W.bor[e]) && W.hx[e] &&
            W.hfmt[e] >= 0 && expert_bytes((XFmt)W.hfmt[e]) <= xb)
            cand.push_back(e);
    std::sort(cand.begin(), cand.end(), [&](int a, int b) { return pc[a] != pc[b] ? pc[a] > pc[b] : a < b; });
    // review pf3 L1: free pool slots take the prompt's most used streamed experts of the pool format (moe_prefill's
    // seeding, from the owner's stage buffer): the same number of the most predicted ones is left out
    if (E->o.dyn && !E->dyn_fp4 && !W.dpool.empty()) {
        int nfree = 0;
        for (size_t j = 0; j < W.dpool.size(); j++) nfree += W.dyn_e[j] < 0 && W.pend_e[j] == -1;
        if (nfree > 0) {
            std::vector<int> keep;
            for (int e : cand)
                if (nfree > 0 && W.hfmt[e] == pool_fmt(E, W.lid)) nfree--;
                else keep.push_back(e);
            cand.swap(keep);
        }
    }
    // the cards other than the owner first (the owner adds every expert's rows and computes its residents), round robin
    std::vector<int> used(NC, 0), seq;
    for (int k = 1; k < NC; k++) seq.push_back((W.card + k) % NC);
    seq.push_back(W.card);
    size_t ci = 0;
    for (int pass = 0; pass < 2 && ci < cand.size(); pass++)
        for (bool any = true; any && ci < cand.size();) {
            any = false;
            for (int k : seq) {
                if (ci >= cand.size()) break;
                if (pass == 0 && k == W.card) continue;   // the owner's slots only after the others are full
                Card& kc = E->cards[k];
                if (used[k] >= kc.pf_ns) continue;
                const int e = cand[ci++], j = used[k]++;
                any = true;
                E->pfx_card[e] = k;
                E->pfx_j[e] = j;
                DevGuard dg(kc.dev);
                DCK(cudaStreamWaitEvent(kc.cp, kc.pf_fre[j], 0));
                // the attention shares (P2P staging on the owner's critical path) go first: every card's copies wait for
                // them (owner only: 2nd gitlog 2.175 s; all cards: 2.135 s)
                // ([dsv4-pf12-pfxb]: starting them after the split's exchanges instead was slower, 10K 5.40 s; review pf12 L2)
                if (j == 0) DCK(cudaStreamWaitEvent(kc.cp, c.ev_stg, 0));
                if (pf_on()) pf_rec(E, k, 1, kc.cp, l, PK_CPWAIT);   // [dsv4-pf6-link]: the copy interval = the DMA only
                DCK(cudaMemcpyAsync(pfx_slot(kc, xb, j), W.hx[e], expert_bytes((XFmt)W.hfmt[e]), cudaMemcpyHostToDevice, kc.cp));
                if (pf_on()) pf_rec(E, k, 1, kc.cp, l, PK_COPY, -1, (double)expert_bytes((XFmt)W.hfmt[e]), -7, 1);   // sid -7: prefetch copy [dsv4-pf15-pfx]
                DCK(cudaEventRecord(kc.pf_rdy[j], kc.cp));
            }
        }
    for (int k = 0; k < NC; k++) E->cards[k].pf_ns = used[k];   // the slots holding experts (moe_prefill keeps them)
    DCK(cudaSetDevice(c.dev));
}
static void moe_prefill(Engine* E, Card& c, Layer& W, int l, int n) {
    cudaStream_t st = c.st;
    const bool pf = pf_on();   // [dsv4-pf-pprof]
    const auto pf_tm = clk::now();
    const int pfo = W.card;
    // routing
    float* logits = (float*)c.phase;
    lin32(c, W.gate_w, NE, D, c.x, D, n, logits, NE);
    k_route<<<cdiv((size_t)n * 32, 256), 256, 0, st>>>(logits, W.gate_b, n, c.ids, c.rw);
    // xq (bf16, FP8-rounded) of all rows
    bf16* xq = (bf16*)c.phase;   // overwrites logits after the route (stream order)
    // copy ids / weights to the host (route reads logits before q_fp8 writes xq: same stream)
    DCK(cudaMemcpyAsync(E->h_pids, c.ids, (size_t)n * TOPK * 4, cudaMemcpyDeviceToHost, st));
    DCK(cudaMemcpyAsync(E->h_rw, c.rw, (size_t)n * TOPK * 4, cudaMemcpyDeviceToHost, st));
    if (pf) pf_rec(E, pfo, 0, st, l, PK_ROUTE);
    auto trs = clk::now();
    DCK(cudaStreamSynchronize(st));
    g_route_sync += ms_since(trs);
    const double pf_rs = ms_since(trs);
    if (pf) g_pf.lh[l].rsync += pf_rs;
    // [dsv4-pf8-xqh] xq to the helpers (and the CPU workers) at about half the bytes: E4M3 codes + one power-of-2
    // exponent per 32 values (k_q_fp8_pk, written beside the bf16 rows), pushed by the owner's side stream into the
    // tail of each helper's xb (free in the MoE phase: the front holds at most n x 5120 bf16 rows: linq's q_fp8 of
    // the shared expert, gathers) and rebuilt there bit-exactly (k_fp8_unpk); the CPU path's D2H of the packed rows
    // follows the pushes on the same stream. DSV4_NOXQH=1: bf16 rows pulled by the helpers as before
    const size_t pkb = (size_t)n * D + (size_t)n * (D / 32);
    auto xb_tail = [&](Card& k) -> uint8_t* {
        if (k.xb_n * 2 < pkb + (size_t)n * D * 2 + 512) return nullptr;
        return (uint8_t*)(((uintptr_t)k.xb + k.xb_n * 2 - pkb) & ~(uintptr_t)255);
    };
    const bool xqh = E->o.pdist && E->asplit && E->cards.size() == 3 && !E->hostx && !E_xqh_off && !getenv("DSV4_PDCHK") &&
                     std::all_of(E->cards.begin(), E->cards.end(), [&](Card& k) { return xb_tail(k) != nullptr; });
    uint8_t* opk = xqh ? xb_tail(c) : nullptr;
    // pushed early from the owner (beside the shared expert) for chunks of >= DSV4_XQH_PUSH_MIN rows (default 4096; 10K
    // 5.895 pulled -> 5.785 s pushed), pulled by the helpers after c.ev below (2.5K: 1.711 pulled vs 1.730 pushed)
    static const int xqh_pmin = getenv("DSV4_XQH_PUSH_MIN") ? atoi(getenv("DSV4_XQH_PUSH_MIN")) : 4096;
    const bool xq_pull = E_xqh_pull || n < xqh_pmin;
    if (xqh)
        k_q_fp8_pk<<<cdiv((size_t)n * D, 256), 256, 0, st>>>(c.x, D, n, D, xq, D, opk, (int8_t*)(opk + (size_t)n * D));
    else
        q_fp8(st, c.x, D, n, D, nullptr, 0, xq, D);
    // (a chunk of < DSV4_CPUPF_MINN rows, default 512, keeps every expert on the GPUs: an 18-token prompt ran +23 ms with
    // the CPU path, each expert being one worker's ~2-3 ms)
    static const int cpu_minn = getenv("DSV4_CPUPF_MINN") ? atoi(getenv("DSV4_CPUPF_MINN")) : 512;
    const bool cpu_l = E->pfcpu.started() && E->o.pdist && E->asplit && E->cards.size() == 3 && n <= E->o.chunk && n >= cpu_minn &&
                       !getenv("DSV4_PDCHK");   // [dsv4-pf2-cpu]
    if (cpu_l || xqh) {   // xq to the host beside the shared expert (its own stream: st does not wait for the D2H)
        if (!c.cxs) {
            DCK(cudaStreamCreateWithFlags(&c.cxs, cudaStreamNonBlocking));
            DCK(cudaEventCreateWithFlags(&c.ev_cxq, cudaEventDisableTiming));
            DCK(cudaEventCreateWithFlags(&c.ev_cy, cudaEventDisableTiming));
            DCK(cudaEventCreateWithFlags(&c.ev_q, cudaEventDisableTiming));
        }
        if (!c.ev_xqd) DCK(cudaEventCreateWithFlags(&c.ev_xqd, cudaEventDisableTiming));
        DCK(cudaEventRecord(c.ev_q, st));   // xq written (the D2H is enqueued once the layer has CPU jobs: review cpu L2)
    }
    if (xqh && !xq_pull) {   // [dsv4-pf8-xqh] the pushes (the helper threads wait ev_xqd, then rebuild the bf16 rows)
        DCK(cudaStreamWaitEvent(c.cxs, c.ev_q, 0));
        for (int k = 0; k < (int)E->cards.size(); k++)
            if (k != W.card) DCK(cudaMemcpyAsync(xb_tail(E->cards[k]), opk, pkb, cudaMemcpyDefault, c.cxs));
        DCK(cudaEventRecord(c.ev_xqd, c.cxs));
    }
    // shared expert first (all rows) into c.ao
    float* Y = (float*)(xq + (size_t)n * D);                    // [n][5120]
    float* H = Y + (size_t)n * D;                               // [n][4608]
    float* hf = H + (size_t)n * 2 * INTER;                      // [n][2304]
    bf16* hq = (bf16*)(hf + (size_t)n * INTER);                 // [n][2304]
    bf16* Xe = c.xb;                                            // [n][5120] gathered rows
    {
        // g, u via the shared FP8 weights (input quantization identical to xq)
        linq(c, W.sh_w1, c.x, D, n, H, 2 * INTER);   // [w1; w3]
        k_glu_rows<<<cdiv((size_t)n * INTER, 256), 256, 0, st>>>(H, c.ones, n, hf);
        linq(c, W.sh_w2, hf, INTER, n, c.ao, D);
    }
    if (pf) pf_rec(E, pfo, 0, st, l, PK_SHARED);
    const auto pf_tb = clk::now();
    // per-expert row lists: hot experts first (ascending), then the host ones (ascending)
    std::vector<int> cnt(NE, 0);
    for (int i = 0; i < n * TOPK; i++) cnt[E->h_pids[i]]++;
    if (g_fut) {
        std::lock_guard<std::mutex> g(g_fut_mu);
        g_fut_buf.insert(g_fut_buf.end(), {3, l, n, E->dyn_clock});
        g_fut_buf.insert(g_fut_buf.end(), cnt.begin(), cnt.end());
    }
    if (E->dyn_fp4 || (E->gg && g_ggsc)) {   // review r4 M3 (R20: + GGUF with ggsc): the prompt's routing feeds the LFU scores (as a rate over 64 tokens), no admission
        const int t = E->dyn_clock;
        for (int e = 0; e < NE; e++)
            if (cnt[e]) {
                W.score[e] = W.score[e] * powf(E->o.dyn_decay, (float)(t - W.score_t[e])) + 64.f * cnt[e] / n;
                W.score_t[e] = t;
            }
    }
    std::vector<int> start(NE + 1, 0), order;
    // R9: the accumulation order (resident experts first, then host ones, ascending) does not depend on where a resident
    // expert sits (another card's slot: staged over P2P) or on borrowing (evicted for this prefill: staged from its host
    // copy), so the prefill sums are bit-identical to the unsplit engine (review pf9 M1: not with [dsv4-pf9-hsum], where
    // the helpers' partial sums follow the expert -> card assignment; DSV4_NOHSUM=1 restores it)
    auto resident = [&](int e) { return W.h_slot[e] >= 0 || W.h_dslot[e] >= 0 || (!W.bor.empty() && W.bor[e]); };
    // R16 pool split: a pool expert is local only on the owner's card; one on another card is computed there (premote)
    auto plocal = [&](int e) { return W.h_dslot[e] >= 0 && (!E->psplit || gs_card(W.h_dslot[e]) == W.card); };
    auto premote = [&](int e) { return E->psplit && W.h_dslot[e] >= 0 && gs_card(W.h_dslot[e]) != W.card; };
    auto local = [&](int e) { return (W.h_slot[e] >= 0 && gs_card(W.h_slot[e]) == W.card) || plocal(e); };
    auto remote = [&](int e) { return W.h_slot[e] >= 0 && gs_card(W.h_slot[e]) != W.card; };
    // borrowed with a lossy host copy (FP4 tier): FP4 bytes from the checkpoint; R19: the split pool's host copies are its
    // exact bytes, so a borrowed M23 / GGUF expert is staged from its host copy
    auto bck = [&](int e) { return !E->gborrow && !W.bor.empty() && W.bor[e] && W.hfmt[e] != XF_FP4; };
    auto mfmt = [&](int e) {   // review pf2 M1: the format an expert is computed in (cost model only)
        if (W.h_slot[e] >= 0 || bck(e)) return XF_FP4;
        if (W.h_dslot[e] >= 0 && (W.bor.empty() || !W.bor[e])) return pool_fmt(E, W.lid);
        return (XFmt)W.hfmt[e];
    };
    for (int e = 0; e < NE; e++)
        if (cnt[e] && resident(e)) order.push_back(e);
    (void)order.size();
    for (int e = 0; e < NE; e++)
        if (cnt[e] && !resident(e)) order.push_back(e);
    static const bool hostlog = getenv("DSV4_PFHOSTLOG") != nullptr;   // [dsv4-pf9-lm] debug: host experts per layer and chunk
    if (hostlog) {
        std::string ids;
        double by = 0;
        int nh = 0;
        for (int e = 0; e < NE; e++)
            if (cnt[e] && !resident(e) && W.hfmt[e] >= 0) {
                ids += std::to_string(e) + ":" + std::to_string(cnt[e]) + ",";
                by += (double)expert_bytes((XFmt)W.hfmt[e]);
                nh++;
            }
        logf_("pfhost layer %d n %d host %d bytes %.0f ids %s\n", l, n, nh, by, ids.c_str());
    }
    // [dsv4-pf15-cone] gate (DSV4_CONELOG=1, diagnostic only): this layer's host experts with all rows vs only the last
    // rows(l) = min(n, 127 x (40 - l) + 1) rows (layers > 20): GPU-staged (> DSV4_CPUPF_MAX rows) and CPU-class counts / bytes
    static const bool conelog = getenv("DSV4_CONELOG") != nullptr;
    if (conelog && l > CAND_SRC) {
        const int rl = std::min(n, 127 * (NL - l) + 1);
        std::vector<int> cc(NE, 0);
        for (int i = n - rl; i < n; i++)
            for (int k = 0; k < TOPK; k++) cc[E->h_pids[i * TOPK + k]]++;
        double bf = 0, bc = 0;
        int gf = 0, gc = 0, pf_ = 0, pc = 0;
        for (int e = 0; e < NE; e++) {
            if (resident(e) || W.hfmt[e] < 0) continue;
            const double b = (double)expert_bytes((XFmt)W.hfmt[e]);
            if (cnt[e] > E_cpu_max) { bf += b; gf++; } else if (cnt[e] > 0) pf_++;   // (review pf15 L5)
            if (cc[e] > E_cpu_max) { bc += b; gc++; } else if (cc[e] > 0) pc++;
        }
        logf_("cone layer %d n %d rows %d: GPU host experts %d -> %d (%.2f -> %.2f GB), CPU-class %d -> %d\n", l, n, rl, gf, gc,
              bf / 1e9, bc / 1e9, pf_, pc);
    }
    // dynamic tier seeding: free pool slots take the prompt's most used streamed experts (copied from the staging
    // buffer while it holds them)
    std::vector<int> seed_slot(NE, -1);
    if (E->o.dyn && !E->dyn_fp4 && !W.dpool.empty()) {
        std::vector<int> freev;
        for (size_t j = 0; j < W.dpool.size(); j++)
            if (W.dyn_e[j] < 0 && W.pend_e[j] == -1) freev.push_back((int)j);   // (R19: -2 = borrowed, not free)
        std::vector<int> cs;
        std::vector<char> pend(NE, 0);   // review r16 L2: never seed an expert whose admission is still in flight
        for (int pe : W.pend_e)
            if (pe >= 0) pend[pe] = 1;
        for (int e = 0; e < NE; e++)
            if (cnt[e] && !resident(e) && !pend[e] && W.hfmt[e] == pool_fmt(E, W.lid)) cs.push_back(e);
        std::sort(cs.begin(), cs.end(), [&](int a, int b) { return cnt[a] != cnt[b] ? cnt[a] > cnt[b] : a < b; });
        for (size_t i = 0; i < std::min(freev.size(), cs.size()); i++) seed_slot[cs[i]] = freev[i];
    }
    std::vector<int> base(NE, 0);
    int acc = 0;
    for (int e : order) {
        base[e] = acc;
        acc += cnt[e];
    }
    std::vector<int> fill(NE, 0);
    std::vector<float> wts((size_t)n * TOPK);
    for (int i = 0; i < n; i++)
        for (int k = 0; k < TOPK; k++) {
            int e = E->h_pids[i * TOPK + k];
            int p = base[e] + fill[e]++;
            E->h_rows[p] = i;
            wts[p] = E->h_rw[i * TOPK + k];
        }
    DCK(cudaMemcpyAsync(c.rows, E->h_rows, (size_t)acc * 4, cudaMemcpyHostToDevice, st));
    DCK(cudaMemcpyAsync(c.rwt, wts.data(), (size_t)acc * 4, cudaMemcpyHostToDevice, st));
    // R11 distributed prefill: every expert of `order` is computed on one card: the owner's own residents (and the cb3
    // band / borrowed / seeded ones) on the owner, another card's residents on that card (no 17.7 MB P2P staging), host
    // experts balanced by staged bytes over the 3 cards (each card stages over its own PCIe link); a remote card writes
    // the expert's rows [m][5120] into its phase region and the owner adds them (k_scatter_add, peer reads) in `order`:
    // same products, same addition order as the one-card path (bit-identical; [dsv4-pf9-hsum] regroups: order-only)
    const int NC = (int)E->cards.size();
    // [dsv4-rel-hostx]: without P2P a remote card's result rows go through a pinned mapped host buffer (D2H by that card,
    // read in place by the owner's scatter-add)
    const bool pd = E->o.pdist && E->asplit && NC == 3;
    static const bool pdchk = getenv("DSV4_PDCHK") != nullptr;   // check: the owner computes all, remote results compared
    std::vector<int> asg(order.size(), W.card);
    if (pd && E_lb_off) {
        std::vector<double> ld(NC, 0.0);
        for (size_t oi = 0; oi < order.size(); oi++) {
            const int e = order[oi];
            if (local(e) || bck(e) || seed_slot[e] >= 0 || (!W.bor.empty() && W.bor[e])) continue;
            if (remote(e)) { asg[oi] = gs_card(W.h_slot[e]); continue; }
            if (premote(e)) { asg[oi] = gs_card(W.h_dslot[e]); continue; }   // R16: on the pool slot's card
            int b = 0;
            for (int k = 1; k < NC; k++)
                if (ld[k] < ld[b] - 1.0) b = k;
            asg[oi] = b;
            ld[b] += (double)expert_bytes((XFmt)W.hfmt[e]);
        }
    } else if (pd) {
        // [dsv4-pf-lb]: host experts (and borrowed ones with an exact host copy: R19) go to the card whose modelled
        // finish time max(link, math) grows least; the residents' math (fixed cards) and the owner's adds are counted
        // first. The card that computes an expert never changes a sum (the owner adds every expert's rows in `order`),
        // except with [dsv4-pf9-hsum] (helpers' partial sums: the assignment sets the grouping, review pf9 M1)
        std::vector<double> lk(NC, 0.0), cp(NC, 0.0);
        std::vector<char> mov(order.size(), 0);
        for (size_t oi = 0; oi < order.size(); oi++) {
            const int e = order[oi], m = cnt[e];
            const bool bor = !W.bor.empty() && W.bor[e];
            cp[W.card] += 0.01;   // its add on the owner
            const double ce = IlvModel::cost(m, mfmt(e));
            if (local(e) || bck(e) || seed_slot[e] >= 0 || (bor && !W.hx[e])) { cp[W.card] += ce; continue; }
            if (remote(e)) { asg[oi] = gs_card(W.h_slot[e]); cp[asg[oi]] += ce; continue; }
            if (premote(e)) { asg[oi] = gs_card(W.h_dslot[e]); cp[asg[oi]] += ce; continue; }
            mov[oi] = 1;
        }
        // [dsv4-pf2-cpu] the CPU workers' finish times (ms; start after the xq D2H); one expert per worker (review cpu M1)
        std::vector<double> cpl(cpu_l ? std::max(1, E->pfcpu.threads()) : 0, 1.0);
        for (size_t oi = 0; oi < order.size(); oi++) {
            if (!mov[oi]) continue;
            const int e = order[oi], m = cnt[e];
            const double tl = (double)expert_bytes((XFmt)W.hfmt[e]) / IlvModel::LBW, tc = IlvModel::cost(m, (XFmt)W.hfmt[e]);
            int b = 0;
            double fb = 1e30;
            for (int k = 0; k < NC; k++) {
                const double f = std::max(lk[k] + tl, cp[k] + tc);
                if (f < fb - 1e-9) { fb = f; b = k; }
            }
            if (cpu_l && m <= E_cpu_max && W.hx[e] && PfCpu::fmt_ok(W.hfmt[e])) {
                // per worker: the aggregate rate / threads (microbench: m <= 2 ~95, 4 ~80, 8 ~55-60 GB/s at 23 threads;
                // DSV4_CPUPF_BW absorbs the DRAM shared with the links: review cpu M2)
                const double tcpu = (double)expert_bytes((XFmt)W.hfmt[e]) * (double)cpl.size() /
                                    (E_cpu_bw * 1e6 * (m <= 2 ? 1.0 : m <= 4 ? 0.88 : 0.66));
                auto it = std::min_element(cpl.begin(), cpl.end());
                if (*it + tcpu < fb - 1e-9) {
                    asg[oi] = NC;
                    *it += tcpu;
                    continue;
                }
            }
            asg[oi] = b;
            lk[b] += tl;
            cp[b] += tc;
        }
    }
    // [dsv4-pf3-pfx] the CPU set above is the lb model's as before; the GPU experts are placed again with the prefetched
    // ones on their slot's card (no link time), the others by the same rule (seeded ones stay on the owner)
    const bool pfx = pd && E->pfx_l == l && pfx_on(E, n) &&
                     std::any_of(E->pfx_card.begin(), E->pfx_card.end(), [](int k) { return k >= 0; });
    std::vector<int> pfj(order.size(), -1);   // slot of a prefetched expert computed from it (on card asg)
    if (pfx) {
        std::vector<double> lk(NC, 0.0), cp(NC, 0.0);
        std::vector<int> mv;
        for (size_t oi = 0; oi < order.size(); oi++) {
            const int e = order[oi], m = cnt[e];
            const bool bor = !W.bor.empty() && W.bor[e];
            cp[W.card] += 0.01;
            const double ce = IlvModel::cost(m, mfmt(e));
            if (local(e) || bck(e) || seed_slot[e] >= 0 || (bor && !W.hx[e])) { cp[W.card] += ce; continue; }
            if (remote(e) || premote(e)) { cp[asg[oi]] += ce; continue; }
            if (asg[oi] == NC) continue;
            if (E->pfx_card[e] >= 0) {
                asg[oi] = E->pfx_card[e];
                pfj[oi] = E->pfx_j[e];
                cp[asg[oi]] += IlvModel::cost(m, (XFmt)W.hfmt[e]);
                continue;
            }
            mv.push_back((int)oi);
        }
        for (int oi : mv) {
            const int e = order[oi], m = cnt[e];
            const double tl = (double)expert_bytes((XFmt)W.hfmt[e]) / IlvModel::LBW, tc = IlvModel::cost(m, (XFmt)W.hfmt[e]);
            int b = 0;
            double fb = 1e30;
            for (int k = 0; k < NC; k++) {
                const double f = std::max(lk[k] + tl, cp[k] + tc);
                if (f < fb - 1e-9) { fb = f; b = k; }
            }
            asg[oi] = b;
            lk[b] += tl;
            cp[b] += tc;
        }
        if (E_pfx_log) {
            int np = 0, hit = 0, mgpu = 0;
            double bh = 0, bg = 0;
            for (int e = 0; e < NE; e++) np += E->pfx_card[e] >= 0;
            for (size_t oi = 0; oi < order.size(); oi++) {
                const int e = order[oi];
                if (resident(e) || asg[oi] == NC || seed_slot[e] >= 0) continue;
                mgpu++;
                bg += (double)expert_bytes((XFmt)W.hfmt[e]);
                if (pfj[oi] >= 0) { hit++; bh += (double)expert_bytes((XFmt)W.hfmt[e]); }
            }
            logf_("pfx layer %d n %d: %d prefetched (slots %d / %d / %d), %d of %d GPU host experts hit, %.2f of %.2f GB\n", l, n,
                  np, E->cards[0].pf_ns, E->cards[1].pf_ns, E->cards[2].pf_ns, hit, mgpu, bh / 1e9, bg / 1e9);
        }
    }
    // [dsv4-pf2-cpu] CPU experts (asg = NC): rows into h_cy in order, started now (the workers wait for the xq D2H
    // and for the previous layer's adds of h_cy); the owner adds each one's rows at its place in `order`
    std::vector<PfCpu::Job> cjobs;
    std::vector<int> cjob(order.size(), -1);
    {
        size_t off = 0;
        for (size_t oi = 0; oi < order.size(); oi++)
            if (asg[oi] == NC) {
                const int e = order[oi];
                cjob[oi] = (int)cjobs.size();
                cjobs.push_back({W.hx[e], (uint8_t)W.hfmt[e], cnt[e], E->h_rows + base[e], wts.data() + base[e], E->h_cy + off * D});
                off += (size_t)cnt[e];
            }
    }
    const auto cpu_t0 = clk::now();
    // review cpu L3: the workers read cjobs / wts / h_rows: every exit from here on waits for them first
    struct CpuJoin { PfCpu* p = nullptr; ~CpuJoin() { if (p) p->wait_all(); } } cpu_join;
    if (!cjobs.empty()) {
        DCK(cudaStreamWaitEvent(c.cxs, c.ev_q, 0));
        if (xqh)   // [dsv4-pf8-xqh] the packed rows (after the pushes: same stream)
            DCK(cudaMemcpyAsync(E->h_cxq, opk, pkb, cudaMemcpyDeviceToHost, c.cxs));
        else
            DCK(cudaMemcpyAsync(E->h_cxq, xq, (size_t)n * D * 2, cudaMemcpyDeviceToHost, c.cxs));
        DCK(cudaEventRecord(c.ev_cxq, c.cxs));
        cpu_join.p = &E->pfcpu;
        cudaEvent_t exq = c.ev_cxq, ely = E->cy_last;
        E->pfcpu.run(E->h_cxq, (int)cjobs.size(), cjobs.data(), [exq, ely] {
            DCK(cudaEventSynchronize(exq));
            if (ely) DCK(cudaEventSynchronize(ely));
        }, xqh ? (const int8_t*)((const uint8_t*)E->h_cxq + (size_t)n * D) : nullptr);
    }
    // remote work first (it runs beside the owner's own loop)
    std::vector<int> yoff(order.size(), -1), pxi(order.size(), -1);
    std::vector<char> rwork(NC, 0);   // R12 (review r11 L1): cards that wrote rows into their phase region
    // [dsv4-pf9-hsum] helpers that add their experts' rows into one dense partial sum S_k (their own order); the owner
    // adds its own / CPU rows in `order`, then S_1, S_2 (order-only: the sums are regrouped). DSV4_NOHSUM=1 off
    const bool hsum = pd && !E_hsum_off && !E->hostx && !pdchk;
    std::vector<char> hsk(NC, 0);
    std::vector<int> mmax_k(NC, 0);
    std::unique_ptr<std::atomic<uint8_t>[]> hsr(new std::atomic<uint8_t>[NC]);   // S_k's event recorded
    for (int k = 0; k < NC; k++) hsr[k].store(0, std::memory_order_relaxed);
    std::unique_ptr<std::atomic<uint8_t>[]> pxr(new std::atomic<uint8_t>[order.size() + 1]);   // [dsv4-pf-mtq]
    for (size_t i = 0; i < order.size(); i++) pxr[i].store(0, std::memory_order_relaxed);
    std::vector<std::function<void()>> rfn;
    std::vector<std::thread> rth;
    struct RthJoin { std::vector<std::thread>& t; ~RthJoin() { for (auto& x : t) if (x.joinable()) x.join(); } } rth_join{rth};
    if (pd) {
        // [dsv4-pf8-xqh] the pushes done before c.ev: the helpers keep one cross-card wait (a second one per helper cost
        // ~2 ms of host enqueue lag per layer at 2.5K); DSV4_XQH_EW=1: the helpers wait ev_xqd themselves (earlier unpack)
        if (xqh && !E_xqh_ew && !xq_pull) DCK(cudaStreamWaitEvent(st, c.ev_xqd, 0));
        DCK(cudaEventRecord(c.ev, st));   // xq, rows, weights ready on the owner
        for (int k = 0; k < NC; k++) {
            if (k == W.card) continue;
            Card& r = E->cards[k];
            std::vector<int> mine;
            size_t rows = 0;
            int mmax = 0;
            for (size_t oi = 0; oi < order.size(); oi++)
                if (asg[oi] == k) mmax = std::max(mmax, cnt[order[oi]]);
            // the card's phase region holds xq, the per-expert result rows and one expert's temporaries; experts past
            // its capacity go back to the owner (long chunks)
            const size_t fixed = (size_t)n * D * 2 + (size_t)mmax * (2 * INTER * 4 + INTER * 4 + INTER * 2) + 4096;
            const size_t avail = r.phase_bytes - (pfx ? (size_t)std::min(r.pf_ns, r.pf_np) * E->pfx_xb : 0);   // [dsv4-pf3-pfx] below the slots
            size_t cap_rows = avail > fixed ? (avail - fixed) / ((size_t)D * 4) : 0;
            // [dsv4-pf9-hsum] dense partial sum S [n][D] fp32 of its first experts (one expert's rows [mmax][D] as the
            // temporary), its last experts (up to DSV4_HSUM_TAIL of its rows) as rows; only where S has fewer bytes
            // than the rows it replaces
            size_t trows = 0;
            for (size_t oi = 0; oi < order.size(); oi++)
                if (asg[oi] == k) trows += (size_t)cnt[order[oi]];
            const size_t rbmax = n < E_hsum_tmaxn ? (size_t)(E_hsum_tail * (double)trows) : 0;
            const size_t fixed_hs = fixed + ((size_t)n + mmax + rbmax) * D * 4;
            if (hsum && trows - rbmax >= (size_t)n && avail >= fixed_hs) {
                hsk[k] = 1;
                cap_rows = (size_t)-1 / 2;
            }
            if (E->hostx) {   // [dsv4-rel-hostx]: the host rows buffer (one chunk of rows)
                const size_t hc = (size_t)std::max(E->o.chunk, MAXR);
                if (!r.hrows) DCK(cudaHostAlloc((void**)&r.hrows, hc * D * 4, cudaHostAllocPortable | cudaHostAllocMapped));
                cap_rows = std::min(cap_rows, hc);
            }
            int nback = 0;
            for (size_t oi = 0; oi < order.size(); oi++)
                if (asg[oi] == k) {
                    if (rows + cnt[order[oi]] > cap_rows) {
                        asg[oi] = W.card;
                        pfj[oi] = -1;   // [dsv4-pf3-pfx] (the owner stages it: the slot is on this card)
                        nback++;
                        continue;
                    }
                    mine.push_back((int)oi);
                    yoff[oi] = hsk[k] ? 0 : (int)rows;
                    rows += cnt[order[oi]];
                }
            if (pf) g_pf.cnt[l][W.card].ex[PC_BACK] += nback;
            mmax_k[k] = mmax;
            if (nback) {
                static int said = 0;
                if (said++ < 3) logf_("prefill layer %d n %d: %d experts of card %d back to the owner (phase region)\n", l, n, nback, k);
            }
            if (mine.empty()) continue;
            rwork[k] = 1;
            if (hsk[k] && !r.ev_hs) {
                DevGuard dg(r.dev);
                DCK(cudaEventCreateWithFlags(&r.ev_hs, cudaEventDisableTiming));
            }
            if ((int)r.ev_px.size() < NE) {   // (created here, before any helper thread reads the vector)
                DevGuard dg(r.dev);
                for (int i = (int)r.ev_px.size(); i < NE; i++) {
                    cudaEvent_t ev;
                    DCK(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming));
                    r.ev_px.push_back(ev);
                }
            }
            // [dsv4-pf-mtq]: this card's expert work is enqueued by a helper thread (beside the owner's enqueue); the
            // owner waits for an expert's event only after the helper recorded it (pxr). Invariants (review pf L3): c.ev
            // is not re-recorded before the helpers are joined (below, before the R12 events); helpers never take
            // seeded experts (seed_slot experts stay on the owner) and touch only their own card's state
            // (r.stage / st_ready / st_free / natbuf / blas / ev_px[j]) plus pxi / pxr entries of their own experts
            rfn.push_back([&, k, rows, mmax, mine, rbmax]() mutable {
            Card& r = E->cards[k];
            DevGuard dg(r.dev);
            cudaStream_t rs = r.st;
            const bool hs = hsk[k];
            bf16* rxq = (bf16*)r.phase;
            float* rY = (float*)(rxq + (size_t)n * D);   // (hsum: one expert's rows [mmax][D], then S [n][D])
            float* rS = hs ? rY + (size_t)mmax * D : nullptr;   // then the tail experts' rows [rbmax][D]
            float* rH = hs ? rS + ((size_t)n + rbmax) * D : rY + rows * D;
            float* rhf = rH + (size_t)mmax * 2 * INTER;
            bf16* rhq = (bf16*)(rhf + (size_t)mmax * INTER);
            if (xqh) {   // [dsv4-pf8-xqh] the pushed packed rows -> bf16 rows (bit-exact), then the lists
                uint8_t* rpk = xb_tail(r);
                DCK(cudaStreamWaitEvent(rs, E_xqh_ew && !xq_pull ? c.ev_xqd : c.ev, 0));
                if (xq_pull) DCK(cudaMemcpyAsync(rpk, opk, pkb, cudaMemcpyDefault, rs));   // (the helper pulls)
                k_fp8_unpk<<<cdiv((size_t)n * D / 8, 256), 256, 0, rs>>>(rpk, (const int8_t*)(rpk + (size_t)n * D), n, D, rxq);
                if (E_xqh_ew && !xq_pull) DCK(cudaStreamWaitEvent(rs, c.ev, 0));
                if (pf) pf_rec(E, k, 0, rs, l, PK_RWAIT);
            } else {
                DCK(cudaStreamWaitEvent(rs, c.ev, 0));
                if (pf) pf_rec(E, k, 0, rs, l, PK_RWAIT);
                DCK(cudaMemcpyAsync(rxq, xq, (size_t)n * D * 2, cudaMemcpyDefault, rs));
            }
            DCK(cudaMemcpyAsync(r.rows, c.rows, (size_t)acc * 4, cudaMemcpyDefault, rs));
            DCK(cudaMemcpyAsync(r.rwt, c.rwt, (size_t)acc * 4, cudaMemcpyDefault, rs));
            if (hs) DCK(cudaMemsetAsync(rS, 0, (size_t)n * D * 4, rs));   // [dsv4-pf9-hsum]
            if (pf) pf_rec(E, k, 0, rs, l, PK_RXQ);
            if (!E_ilv_off) {   // [dsv4-pf-ilv]: residents while the staged copies are in flight (rows go to yoff: any order)
                std::vector<char> sgf(mine.size());
                std::vector<size_t> by(mine.size());
                std::vector<int> rr(mine.size());
                std::vector<XFmt> fm(mine.size());
                for (size_t i = 0; i < mine.size(); i++) {
                    const int e = order[mine[i]];
                    sgf[i] = !remote(e) && !premote(e) && pfj[mine[i]] < 0;   // [dsv4-pf3-pfx]: prefetched = resident
                    by[i] = sgf[i] ? expert_bytes((XFmt)W.hfmt[e]) : 0;   // (review pf L6: only staged ones are read)
                    rr[i] = cnt[e];
                    fm[i] = remote(e) ? XF_FP4 : premote(e) ? pool_fmt(E, W.lid) : (XFmt)W.hfmt[e];
                }
                const std::vector<int> pl = ilv_plan(sgf, by, rr, fm, 2.0);
                std::vector<int> m2(mine.size());
                for (size_t i = 0; i < pl.size(); i++) m2[i] = mine[pl[i]];
                mine.swap(m2);
            }
            std::vector<int> stg;   // staged ones (host experts) in this card's order
            for (int oi : mine)
                if (!remote(order[oi]) && !premote(order[oi]) && pfj[oi] < 0) stg.push_back(oi);
            int nA = (int)mine.size();   // [dsv4-pf9-hsum] experts mine[0 .. nA) into S, the rest as rows (yoff past S)
            if (hs) {
                size_t rb = 0;
                while (nA > 0 && rb + (size_t)cnt[order[mine[nA - 1]]] <= rbmax) rb += (size_t)cnt[order[mine[--nA]]];
                size_t o2 = (size_t)mmax + n;
                for (int i = nA; i < (int)mine.size(); i++) {
                    yoff[mine[i]] = (int)o2;
                    o2 += (size_t)cnt[order[mine[i]]];
                }
                if (nA == 0) {
                    DCK(cudaEventRecord(r.ev_hs, rs));
                    hsr[k].store(1, std::memory_order_release);
                }
            }
            int pos = 0;
            auto rcopy = [&](int si) {
                const int b = si % NST, e = order[stg[si]];
                DCK(cudaStreamWaitEvent(r.cp, r.st_free[b], 0));
                const uint8_t* src = stage_src(E, r, W, e, b);   // [dsv4-pf6-link]: before the event (copy interval = DMA)
                if (pf) pf_rec(E, k, 1, r.cp, l, PK_CPWAIT);
                DCK(cudaMemcpyAsync(r.stage[b], src, expert_bytes((XFmt)W.hfmt[e]), cudaMemcpyHostToDevice, r.cp));
                if (pf) pf_rec(E, k, 1, r.cp, l, PK_COPY, -1, (double)expert_bytes((XFmt)W.hfmt[e]), -1, 1);
                DCK(cudaEventRecord(r.st_ready[b], r.cp));
            };
            for (int si = 0; si < std::min<int>(NST, (int)stg.size()); si++) rcopy(si);
            int si = 0, j = 0;
            for (int oi : mine) {
                const int e = order[oi], m = cnt[e];
                mhist_add(m, (XFmt)W.hfmt[e]);
                const bool inA = hs && pos < nA;   // [dsv4-pf9-hsum]
                const bool lastA = hs && pos == nA - 1;
                pos++;
                const uint8_t* wb;
                XFmt f;
                const bool sg = !remote(e) && !premote(e) && pfj[oi] < 0;
                const bool xs = pf_sample(k);   // [dsv4-pf-pprof] PPROF=2: this expert's sub-stages
                const int xsid = xs ? g_pf_sid.fetch_add(1) : -1, xaux = (sg ? 3 : 2) * 8 + pf_bin(m);
                if (xs) pf_rec(E, k, 0, rs, l, PK_XPRE, PK_REXP, 0, xsid, xaux);
                if (pfj[oi] >= 0) {   // [dsv4-pf3-pfx]: copied into this card's slot during the attention
                    DCK(cudaStreamWaitEvent(rs, r.pf_rdy[pfj[oi]], 0));
                    wb = pfx_slot(r, E->pfx_xb, pfj[oi]);
                    f = (XFmt)W.hfmt[e];
                } else if (remote(e)) {
                    wb = slot_ptr(E, W.h_slot[e]);   // this card's own slot
                    f = XF_FP4;
                } else if (premote(e)) {   // R16: this card's own pool slot
                    wb = pool_slot_ptr(E, c, W.lid, W.h_dslot[e]);
                    f = pool_fmt(E, W.lid);
                } else {
                    DCK(cudaStreamWaitEvent(rs, r.st_ready[si % NST], 0));
                    wb = r.stage[si % NST];
                    f = (XFmt)W.hfmt[e];
                }
                if (xs) pf_rec(E, k, 0, rs, l, PK_XWAIT, PK_REXP, 0, xsid, xaux);
                MatView mv[3];
                expert_views(f, wb, mv);
                if (!E_fx_off && m <= E_fx_max && fx_ok(f)) {   // [dsv4-pf2-fx]
                    float* Yo = rY + (size_t)yoff[oi] * D;
                    if (xs) pf_rec(E, k, 0, rs, l, PK_XDQ, PK_REXP, 0, xsid, xaux);
                    fxd_part(f, 0, mv, rxq, r.rows + base[e], r.rwt + base[e], m, r.d_gg, rhq, Yo, rs);
                    if (xs) pf_rec(E, k, 0, rs, l, PK_XG1, PK_REXP, 0, xsid, xaux);
                    if (xs) pf_rec(E, k, 0, rs, l, PK_XGQ, PK_REXP, 0, xsid, xaux);
                    fxd_part(f, 1, mv, rxq, r.rows + base[e], r.rwt + base[e], m, r.d_gg, rhq, Yo, rs);
                    if (sg) {   // the stage buffer was read by the fused kernels
                        DCK(cudaEventRecord(r.st_free[si % NST], rs));
                        if (si + NST < (int)stg.size()) rcopy(si + NST);
                        si++;
                    }
                    if (pfj[oi] >= 0) DCK(cudaEventRecord(r.pf_fre[pfj[oi]], rs));   // [dsv4-pf3-pfx] slot read
                    if (xs) pf_rec(E, k, 0, rs, l, PK_XG2, PK_REXP, 0, xsid, xaux);
                    if (E->hostx)   // [dsv4-rel-hostx]
                        DCK(cudaMemcpyAsync(r.hrows + (size_t)yoff[oi] * D, Yo, (size_t)m * D * 4, cudaMemcpyDeviceToHost, rs));
                    if (inA) k_scatter_add<<<cdiv((size_t)m * D, 256), 256, 0, rs>>>(Yo, r.rows + base[e], m, rS);   // [dsv4-pf9-hsum]
                    if (xs) pf_rec(E, k, 0, rs, l, PK_XSC, PK_REXP, 0, xsid, xaux);
                    pxi[oi] = j;
                    DCK(cudaEventRecord(r.ev_px[j++], rs));
                    if (lastA) {
                        DCK(cudaEventRecord(r.ev_hs, rs));
                        hsr[k].store(1, std::memory_order_release);
                    }
                    pxr[oi].store(inA ? 2 : 1, std::memory_order_release);
                    continue;
                }
                const int Nm[3] = {INTER, INTER, D}, Km[3] = {D, D, INTER};
                bf16* dsts[3] = {r.w13, r.w13 + (size_t)INTER * D, r.w2};
                for (int q = 0; q < 3; q++) {
                    if (f == XF_FP4)
                        k_deq_fp4v<<<cdiv((size_t)Nm[q] * Km[q] / 16, 256), 256, 0, rs>>>(mv[q].q, (const uint8_t*)mv[q].s,
                                                                                         Nm[q], Km[q], dsts[q]);
                    else if (is_iq(mat_fmt(f, q)))   // R16 [dsv4-gguf-iq]
                        gg_deq_launch(mat_fmt(f, q), mv[q].q, mv[q].s, Nm[q], Km[q], r.d_gg, dsts[q], rs);
                    else
                        k_deq_cb8<<<cdiv((size_t)Nm[q] * Km[q] / 8, 256), 256, 0, rs>>>(
                            mv[q].q, (const __half*)mv[q].s, Nm[q], Km[q], cb_bits(mat_fmt(f, q)),
                            r.d_cb + (mat_fmt(f, q) == XF_CB3 ? 0 : 16), dsts[q]);
                }
                if (xs) pf_rec(E, k, 0, rs, l, PK_XDQ, PK_REXP, 0, xsid, xaux);
                if (sg) {
                    DCK(cudaEventRecord(r.st_free[si % NST], rs));
                    if (si + NST < (int)stg.size()) rcopy(si + NST);
                    si++;
                }
                if (pfj[oi] >= 0) DCK(cudaEventRecord(r.pf_fre[pfj[oi]], rs));   // [dsv4-pf3-pfx] dequantized
                const int* rws = r.rows + base[e];
                float* Yo = rY + (size_t)yoff[oi] * D;
                k_gather_bf16<<<cdiv((size_t)m * D / 8, 256), 256, 0, rs>>>(rxq, D, rws, m, r.xb);
                gemm_bf16(r.blas, r.xb, D, r.w13, D, rH, 2 * INTER, m, 2 * INTER, D);
                if (xs) pf_rec(E, k, 0, rs, l, PK_XG1, PK_REXP, 0, xsid, xaux);
                k_glu_rows<<<cdiv((size_t)m * INTER, 256), 256, 0, rs>>>(rH, r.rwt + base[e], m, rhf);
                q_fp8(rs, rhf, INTER, m, INTER, nullptr, 0, rhq, INTER);
                if (xs) pf_rec(E, k, 0, rs, l, PK_XGQ, PK_REXP, 0, xsid, xaux);
                gemm_bf16(r.blas, rhq, INTER, r.w2, INTER, Yo, D, m, D, INTER);
                if (xs) pf_rec(E, k, 0, rs, l, PK_XG2, PK_REXP, 0, xsid, xaux);
                if (E->hostx)   // [dsv4-rel-hostx]
                    DCK(cudaMemcpyAsync(r.hrows + (size_t)yoff[oi] * D, Yo, (size_t)m * D * 4, cudaMemcpyDeviceToHost, rs));
                if (inA) k_scatter_add<<<cdiv((size_t)m * D, 256), 256, 0, rs>>>(Yo, rws, m, rS);   // [dsv4-pf9-hsum]
                if (xs) pf_rec(E, k, 0, rs, l, PK_XSC, PK_REXP, 0, xsid, xaux);
                pxi[oi] = j;
                DCK(cudaEventRecord(r.ev_px[j++], rs));
                if (lastA) {
                    DCK(cudaEventRecord(r.ev_hs, rs));
                    hsr[k].store(1, std::memory_order_release);
                }
                pxr[oi].store(inA ? 2 : 1, std::memory_order_release);
            }
            if (pf) pf_rec(E, k, 0, rs, l, PK_REXP);
            });
        }
        if (E_mtq_off || rfn.size() < 1)
            for (auto& f : rfn) f();
        else
            for (size_t i = 0; i < rfn.size(); i++)
                try {
                    rth.emplace_back([&, i] {
                        pf_helper_pin(E, (int)i);
                        rfn[i]();
                    });
                } catch (const std::system_error&) {   // review pf L1: no thread: this card's work inline
                    rfn[i]();
                }
        DCK(cudaSetDevice(c.dev));
    }
    std::vector<int> shadow;   // (pdchk) remote assignment kept for the comparison; the owner computes everything
    static unsigned* d_badk[3] = {};
    unsigned* d_bad = nullptr;
    if (pd && pdchk) {
        shadow = asg;
        for (auto& a : asg) a = W.card;
        if (!d_badk[W.card]) d_badk[W.card] = (unsigned*)dmalloc(64);   // (current device = the owner)
        d_bad = d_badk[W.card];
        DCK(cudaMemsetAsync(d_bad, 0, 8, st));
    }
    // staged experts of the owner (host, other card, borrowed): through the copy stream, 3 buffers, in processing order
    std::vector<int> cold, sidx(order.size(), -1);
    for (size_t oi = 0; oi < order.size(); oi++)
        if (!local(order[oi]) && asg[oi] == W.card && pfj[oi] < 0) {   // [dsv4-pf3-pfx] (prefetched: not staged)
            sidx[oi] = (int)cold.size();
            cold.push_back(order[oi]);
        }
    const int nc = (int)cold.size();
    auto enqueue_copy = [&](int ci) {
        int b = ci % NST;
        int e = cold[ci];
        DCK(cudaStreamWaitEvent(c.cp, c.st_free[b], 0));
        // [dsv4-pf6-link] (review pf6 M1): a host copy's source (re-packed when there is no host copy) before the event
        const bool hcp = !remote(e) && !bck(e) && !(!W.hx[e] && premote(e));
        const uint8_t* hsrc = hcp ? stage_src(E, c, W, e, b) : nullptr;
        if (pf) pf_rec(E, pfo, 1, c.cp, l, PK_CPWAIT);
        if (remote(e))
            DCK(cudaMemcpyPeerAsync(c.stage[b], c.dev, slot_ptr(E, W.h_slot[e]), E->cards[gs_card(W.h_slot[e])].dev,
                                    expert_bytes(XF_FP4), c.cp));
        else if (bck(e)) ck_copy(E, l, e, c.stage[b], c.cp);   // review r9 M1: borrowed, lossy host copy: FP4 bytes
        else if (!W.hx[e] && premote(e))   // review r16 M1: a static pool expert (no host copy) back on the owner: its slot
            DCK(cudaMemcpyPeerAsync(c.stage[b], c.dev, pool_slot_ptr(E, c, W.lid, W.h_dslot[e]),
                                    E->cards[gs_card(W.h_dslot[e])].dev, expert_bytes(pool_fmt(E, W.lid)), c.cp));
        else DCK(cudaMemcpyAsync(c.stage[b], hsrc, expert_bytes((XFmt)W.hfmt[e]), cudaMemcpyHostToDevice, c.cp));
        if (pf)   // (bytes: peer / host copy sizes as enqueued above; aux 1 = host -> card copy, review pf6 M2)
            pf_rec(E, pfo, 1, c.cp, l, PK_COPY, -1,
                   (double)(remote(e) ? expert_bytes(XF_FP4) : bck(e) ? expert_bytes(XF_FP4)
                            : (!W.hx[e] && premote(e)) ? expert_bytes(pool_fmt(E, W.lid)) : expert_bytes((XFmt)W.hfmt[e])),
                   -1, hcp ? 1 : 0);
        DCK(cudaEventRecord(c.st_ready[b], c.cp));
    };
    if (pf) {   // [dsv4-pf-pprof] counters per class (card that computes the expert) and the host-expert histogram
        g_pf.lh[l].build += ms_since(pf_tb);
        for (size_t oi = 0; oi < order.size(); oi++) {
            const int e = order[oi], m = cnt[e], k = asg[oi];
            if (k >= NC) continue;   // [dsv4-pf2-cpu] (CPU experts: the pfcpu log line)
            PfCnt& C = g_pf.cnt[l][k];
            auto add = [&](int cl) { C.ex[cl]++; C.rows[cl] += m; };
            bool host = false;
            if (!W.bor.empty() && W.bor[e]) add(PC_BORROW);
            else if (k != W.card) {
                if (remote(e)) add(PC_REMRES);
                else if (premote(e)) add(PC_POOLREM);
                else { add(PC_HOSTREM); host = true; }
            } else if (local(e)) add(PC_LOCAL);
            else if (bck(e)) add(PC_BORROW);
            else if (remote(e) || (!W.hx[e] && premote(e))) add(PC_P2POWN);
            else { add(PC_HOSTOWN); host = true; }
            if (seed_slot[e] >= 0) add(PC_SEED);
            if (host && !W.hx[e]) add(PC_COPYLESS);
            if (host) {
                const int b = pf_bin(m);
                g_pf.hist_ex[b]++;
                g_pf.hist_rows[b] += m;
                g_pf.hist_bytes[b] += (double)expert_bytes((XFmt)W.hfmt[e]);
            }
        }
    }
    for (int ci = 0; ci < std::min(NST, nc); ci++) enqueue_copy(ci);
    bool pf_res = false;
    // [dsv4-pf-ilv] owner: its experts (local and staged) are computed first, in an interleaved order (staged ones in
    // their copy order, locals while a copy is in flight), each into its own rows of the phase region's spare part,
    // then all rows are added in `order` (the same sums in the same order). The owner stream no longer stops at a
    // remote card's rows while it still has work of its own. Experts past the spare rows run in order as before.
    std::vector<float*> eptr(order.size(), nullptr);
    std::vector<std::pair<char, int>> ops;   // (1 = early compute, 0 = in order, oi)
    ops.reserve(order.size() * 2);
    if (!E_ilv_off && shadow.empty()) {
        uint8_t* sp = (uint8_t*)(hq + (size_t)n * INTER);
        sp = (uint8_t*)(((uintptr_t)sp + 255) & ~(uintptr_t)255);
        const size_t used = (size_t)(sp - (uint8_t*)c.phase);
        const size_t avail = c.phase_bytes - (pfx ? (size_t)std::min(c.pf_ns, c.pf_np) * E->pfx_xb : 0);   // [dsv4-pf3-pfx] below the slots
        size_t cap = avail > used ? (avail - used) / ((size_t)D * 4) : 0;
        std::vector<int> own;
        std::vector<char> sgf;
        std::vector<size_t> by;
        std::vector<int> rr;
        std::vector<XFmt> fm;
        for (size_t oi = 0; oi < order.size(); oi++)
            if (asg[oi] == W.card) {
                const int e = order[oi];
                own.push_back((int)oi);
                fm.push_back(sidx[oi] >= 0 ? (remote(e) || bck(e) ? XF_FP4 : (XFmt)W.hfmt[e]) : mfmt(e));
                sgf.push_back(sidx[oi] >= 0);
                by.push_back(remote(e) || bck(e) ? expert_bytes(XF_FP4)
                              : (!W.hx[e] && premote(e)) ? expert_bytes(pool_fmt(E, W.lid)) : expert_bytes((XFmt)W.hfmt[e]));
                rr.push_back(cnt[e]);
            }
        const std::vector<int> pl = ilv_plan(sgf, by, rr, fm, 2.0);
        // arenas: the spare part of the phase region, then the rows of Y above the largest in-order expert
        int mloc = 0;
        for (int r0 : rr) mloc = std::max(mloc, r0);
        size_t top = 0, ytop = (size_t)mloc;
        for (int i : pl) {   // a prefix of the plan (staged ones stay a prefix of the copy order)
            const int oi = own[i], m = cnt[order[oi]];
            if (top + (size_t)m <= cap) {
                eptr[oi] = (float*)sp + top * D;
                top += (size_t)m;
            } else if (ytop + (size_t)m <= (size_t)n) {
                eptr[oi] = Y + ytop * D;
                ytop += (size_t)m;
            } else
                break;
            ops.push_back({1, oi});
        }
    }
    // [dsv4-pf8-ovl] the owner's early experts (own arena rows) run on a second stream (sxc, own cuBLAS handle) while
    // st adds the rows in `order` (remote rows read over the link, CPU rows): compute beside the link-bound adds. Same
    // kernels on the same inputs, same add order (bit-identical). DSV4_NOOVL=1: one stream as before
    static const bool ovl_off = getenv("DSV4_NOOVL") != nullptr || getenv("DSV4_FXCHK") != nullptr;
    std::vector<int> eidx(order.size(), -1);
    int neo = 0;
    for (const auto& op : ops)
        if (op.first == 1) eidx[op.second] = neo++;
    const bool ovl = !ovl_off && neo > 0 && shadow.empty() && pd && c.sxc && c.blas2;   // review pf8 M1: 3 cards, made at load
    bool ojoin = false;
    if (ovl) {
        while ((int)c.ev_oe.size() < neo) {
            cudaEvent_t ev;
            DCK(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming));
            c.ev_oe.push_back(ev);
        }
        DCK(cudaEventRecord(c.ev_ox, st));   // xq, rows, weights, shared expert enqueued before
        DCK(cudaStreamWaitEvent(c.sxc, c.ev_ox, 0));
    }
    for (size_t oi = 0; oi < order.size(); oi++) ops.push_back({0, (int)oi});
    bool oend = false;
    // [dsv4-pf9-hsum] S_1, S_2 added in card order (on st) once each helper recorded its event
    bool sdone = false;
    auto add_s = [&]() {
        sdone = true;
        int hk[3], nh = 0;
        for (int k = 0; k < NC; k++)
            if (hsk[k] && rwork[k]) hk[nh++] = k;
        if (!E_hsred_off && !E->hostx && nh == 2) {   // [dsv4-pf15-hsred]
            for (int i = 0; i < 2; i++)
                for (int sp = 0; !hsr[hk[i]].load(std::memory_order_acquire); sp++) if (sp < 4096) _mm_pause(); else std::this_thread::yield();
            Card* H2[2] = {&E->cards[hk[0]], &E->cards[hk[1]]};
            float* S2[2];
            for (int i = 0; i < 2; i++) S2[i] = (float*)((bf16*)H2[i]->phase + (size_t)n * D) + (size_t)mmax_k[hk[i]] * D;
            const int hr = n / 2, nch = std::min(E_hsred_nch, std::max(1, hr));
            auto rr = [&](int half, int ci, int& r0, int& r1) {   // rows of chunk ci of half 0 [0, hr) / 1 [hr, n)
                const int a = half ? hr : 0, b = half ? n : hr;
                r0 = a + (int)((long)(b - a) * ci / nch);
                r1 = a + (int)((long)(b - a) * (ci + 1) / nch);
            };
            for (int half = 0; half < 2; half++) {   // half 0 combined on the second helper, half 1 on the first
                const int x = half ? 0 : 1, y = 1 - x;
                Card& X = *H2[x];
                DevGuard dg(X.dev);
                while ((int)X.ev_hr.size() < 64) {
                    cudaEvent_t ev;
                    DCK(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming));
                    X.ev_hr.push_back(ev);
                }
                DCK(cudaStreamWaitEvent(X.ass, X.ev_hs, 0));
                DCK(cudaStreamWaitEvent(X.ass, H2[y]->ev_hs, 0));
                for (int ci = 0; ci < nch; ci++) {
                    int r0, r1;
                    rr(half, ci, r0, r1);
                    if (r1 > r0)
                        k_add4<<<cdiv((size_t)(r1 - r0) * D / 4, 256), 256, 0, X.ass>>>((float4*)(S2[x] + (size_t)r0 * D),
                                                                                       (const float4*)(S2[y] + (size_t)r0 * D),
                                                                                       (size_t)(r1 - r0) * D / 4);
                    DCK(cudaEventRecord(X.ev_hr[ci], X.ass));
                }
            }
            DCK(cudaSetDevice(c.dev));
            if (pf) pf_rec(E, pfo, 0, st, l, PK_XWAIT, pf_res ? PK_MHOST : PK_MRES, 0, -9);
            for (int ci = 0; ci < nch; ci++)
                for (int half = 0; half < 2; half++) {
                    const int x = half ? 0 : 1;
                    int r0, r1;
                    rr(half, ci, r0, r1);
                    DCK(cudaStreamWaitEvent(st, H2[x]->ev_hr[ci], 0));
                    if (r1 > r0)
                        k_add4<<<cdiv((size_t)(r1 - r0) * D / 4, 256), 256, 0, st>>>((float4*)(c.ao + (size_t)r0 * D),
                                                                                    (const float4*)(S2[x] + (size_t)r0 * D),
                                                                                    (size_t)(r1 - r0) * D / 4);
                }
            if (pf) pf_rec(E, pfo, 0, st, l, PK_XSC, pf_res ? PK_MHOST : PK_MRES, 0, -9);
            return;
        }
        for (int k = 0; k < NC; k++) {
            if (!hsk[k] || !rwork[k]) continue;
            Card& r = E->cards[k];
            for (int sp = 0; !hsr[k].load(std::memory_order_acquire); sp++) if (sp < 4096) _mm_pause(); else std::this_thread::yield();
            if (pf) pf_rec(E, pfo, 0, st, l, PK_XWAIT, pf_res ? PK_MHOST : PK_MRES, 0, -9);   // [dsv4-pf15] PFTL: before the S wait
            DCK(cudaStreamWaitEvent(st, r.ev_hs, 0));
            if (pf) pf_rec(E, pfo, 0, st, l, PK_XG2, pf_res ? PK_MHOST : PK_MRES, 0, -9);   // S landed (helper done)
            const float* rS = (const float*)((bf16*)r.phase + (size_t)n * D) + (size_t)mmax_k[k] * D;
            k_add4<<<cdiv((size_t)n * D / 4, 256), 256, 0, st>>>((float4*)c.ao, (const float4*)rS, (size_t)n * D / 4);
            if (pf) pf_rec(E, pfo, 0, st, l, PK_XSC, pf_res ? PK_MHOST : PK_MRES, 0, -9);   // S added (peer read)
        }
    };
    for (const auto& op : ops) {
        const size_t oi = (size_t)op.second;
        const bool early = op.first == 1;
        if (ovl && !early && !oend) {   // all early experts enqueued on sxc
            DCK(cudaEventRecord(c.ev_oxe, c.sxc));
            oend = true;
        }
        int e = order[oi];
        int m = cnt[e];
        if (pf && !early && !pf_res && !resident(e)) { pf_rec(E, pfo, 0, st, l, PK_MRES); pf_res = true; }
        if (asg[oi] == NC) {   // [dsv4-pf2-cpu]: computed on the CPU: add its rows (mapped host memory) here, in order
            const int j = cjob[oi];
            for (int sp = 0; !E->pfcpu.done(j); sp++) if (sp < 4096) _mm_pause(); else std::this_thread::yield();
            k_scatter_add<<<cdiv((size_t)m * D, 256), 256, 0, st>>>(cjobs[j].y, c.rows + base[e], m, c.ao);
            continue;
        }
        if (asg[oi] != W.card) {   // R11: computed on card asg: add its rows here, in order
            Card& r = E->cards[asg[oi]];
            uint8_t pv;
            for (int sp = 0; !(pv = pxr[oi].load(std::memory_order_acquire)); sp++) if (sp < 4096) _mm_pause(); else std::this_thread::yield();   // [dsv4-pf-mtq]: its event is recorded
            if (pv == 2) continue;   // [dsv4-pf9-hsum] in S_k
            if (hsk[asg[oi]] && !sdone) add_s();   // (the S sums before the first tail rows of a helper with S)
            const float* Yr = E->hostx ? r.hrows + (size_t)yoff[oi] * D   // [dsv4-rel-hostx]
                                       : (const float*)((bf16*)r.phase + (size_t)n * D) + (size_t)yoff[oi] * D;
            DCK(cudaStreamWaitEvent(st, r.ev_px[pxi[oi]], 0));
            k_scatter_add<<<cdiv((size_t)m * D, 256), 256, 0, st>>>(Yr, c.rows + base[e], m, c.ao);
            continue;
        }
        if (!early && eptr[oi]) {   // computed early: its rows now, at its place
            if (ovl) DCK(cudaStreamWaitEvent(st, c.ev_oe[eidx[oi]], 0));   // [dsv4-pf8-ovl]
            k_scatter_add<<<cdiv((size_t)m * D, 256), 256, 0, st>>>(eptr[oi], c.rows + base[e], m, c.ao);
            continue;
        }
        float* Yd = early ? eptr[oi] : Y;
        // [dsv4-pf8-ovl] early experts on the owner's compute stream (es); the first in-order compute waits for all of them
        const cudaStream_t es = early && ovl ? c.sxc : st;
        const cublasHandle_t eb = early && ovl ? c.blas2 : c.blas;
        if (ovl && !early && !ojoin) {
            DCK(cudaStreamWaitEvent(st, c.ev_oxe, 0));
            ojoin = true;
        }
        const uint8_t* wb;
        XFmt f;
        int ci = sidx[oi];
        const bool xs = pf_sample(pfo);   // [dsv4-pf-pprof] PPROF=2
        const int xsid = xs ? g_pf_sid.fetch_add(1) : -1, xaux = (ci >= 0 ? 1 : 0) * 8 + pf_bin(m);
        const int xfam = pf_res ? PK_MHOST : PK_MRES;
        if (xs) pf_rec(E, pfo, 0, es, l, PK_XPRE, xfam, 0, xsid, xaux);
        if (pfj[oi] >= 0) {   // [dsv4-pf3-pfx]: copied into the owner's slot during the attention
            DCK(cudaStreamWaitEvent(es, c.pf_rdy[pfj[oi]], 0));
            wb = pfx_slot(c, E->pfx_xb, pfj[oi]);
            f = (XFmt)W.hfmt[e];
        } else if (ci < 0 && W.h_slot[e] >= 0) {
            wb = slot_ptr(E, W.h_slot[e]);
            f = XF_FP4;
        } else if (ci < 0) {
            wb = pool_slot_ptr(E, c, W.lid, W.h_dslot[e]);   // owner-local pool slot (review r16: premote experts never get here)
            f = pool_fmt(E, W.lid);
        } else {
            DCK(cudaStreamWaitEvent(es, c.st_ready[ci % NST], 0));
            wb = c.stage[ci % NST];
            f = remote(e) || bck(e) ? XF_FP4 : (XFmt)W.hfmt[e];
        }
        if (xs) pf_rec(E, pfo, 0, es, l, PK_XWAIT, xfam, 0, xsid, xaux);
        MatView mv[3];
        expert_views(f, wb, mv);
        const int Nm[3] = {INTER, INTER, D}, Km[3] = {D, D, INTER};
        bf16* dsts[3] = {c.w13, c.w13 + (size_t)INTER * D, c.w2};
        const bool fx = !E_fx_off && m <= E_fx_max && fx_ok(f);   // [dsv4-pf2-fx]
        mhist_add(m, f);
        for (int j = 0; j < 3 && !fx; j++) {
            if (f == XF_FP4)
                k_deq_fp4v<<<cdiv((size_t)Nm[j] * Km[j] / 16, 256), 256, 0, es>>>(mv[j].q, (const uint8_t*)mv[j].s,
                                                                                  Nm[j], Km[j], dsts[j]);
            else if (is_iq(mat_fmt(f, j)))   // R16 [dsv4-gguf-iq]
                gg_deq_launch(mat_fmt(f, j), mv[j].q, mv[j].s, Nm[j], Km[j], c.d_gg, dsts[j], es);
            else
                k_deq_cb8<<<cdiv((size_t)Nm[j] * Km[j] / 8, 256), 256, 0, es>>>(
                    mv[j].q, (const __half*)mv[j].s, Nm[j], Km[j], cb_bits(mat_fmt(f, j)),
                    c.d_cb + (mat_fmt(f, j) == XF_CB3 ? 0 : 16), dsts[j]);
        }
        if (xs) pf_rec(E, pfo, 0, es, l, PK_XDQ, xfam, 0, xsid, xaux);
        if (ci >= 0 && seed_slot[e] >= 0) {   // seed the dynamic tier from the staged bytes
            int j = seed_slot[e];
            const size_t XB3 = expert_bytes(pool_fmt(E, W.lid));
            DCK(cudaMemcpyAsync(pool_slot_ptr(E, c, W.lid, W.dpool[j]), wb, XB3, cudaMemcpyDeviceToDevice, es));   // P2P when split
            W.dyn_e[j] = e;
            set_pslot(E, W, e, W.dpool[j]);
            W.score[e] = 64.f * cnt[e] / n;   // as a rate over 64 tokens
            W.score_t[e] = E->dyn_clock;
        }
        const int* rws = c.rows + base[e];
        if (fx) {   // [dsv4-pf2-fx] (the stage buffer is free after the fused kernels read it)
            fxd_part(f, 0, mv, xq, rws, c.rwt + base[e], m, c.d_gg, hq, Yd, es);
            if (xs) pf_rec(E, pfo, 0, es, l, PK_XG1, xfam, 0, xsid, xaux);
            if (xs) pf_rec(E, pfo, 0, es, l, PK_XGQ, xfam, 0, xsid, xaux);
            fxd_part(f, 1, mv, xq, rws, c.rwt + base[e], m, c.d_gg, hq, Yd, es);
            // [dsv4-pf2-fx] debug DSV4_FXCHK=1: the old path beside the fused one on the owner card, compared (experts
            // computed on the other cards are not checked: DSV4_PDCHK=1 moves all of them to the owner); logs rel L2 > 1e-3
            static const bool fxchk = getenv("DSV4_FXCHK") != nullptr;
            if (fxchk && m <= 512) {
                static float* Yc[8] = {};   // (debug only: kept for the process, 512 x D floats per card)
                if (!Yc[c.dev]) DCK(cudaMalloc(&Yc[c.dev], (size_t)512 * D * 4));
                for (int j = 0; j < 3; j++) gg_deq_launch(mat_fmt(f, j), mv[j].q, mv[j].s, Nm[j], Km[j], c.d_gg, dsts[j], es);
                k_gather_bf16<<<cdiv((size_t)m * D / 8, 256), 256, 0, es>>>(xq, D, rws, m, Xe);
                gemm_bf16(eb, Xe, D, c.w13, D, H, 2 * INTER, m, 2 * INTER, D);
                k_glu_rows<<<cdiv((size_t)m * INTER, 256), 256, 0, es>>>(H, c.rwt + base[e], m, hf);
                q_fp8(es, hf, INTER, m, INTER, nullptr, 0, hq, INTER);
                gemm_bf16(eb, hq, INTER, c.w2, INTER, Yc[c.dev], D, m, D, INTER);
                std::vector<float> a((size_t)m * D), b((size_t)m * D);
                DCK(cudaMemcpyAsync(a.data(), Yc[c.dev], a.size() * 4, cudaMemcpyDeviceToHost, es));
                DCK(cudaMemcpyAsync(b.data(), Yd, b.size() * 4, cudaMemcpyDeviceToHost, es));
                DCK(cudaStreamSynchronize(es));
                double se = 0, sr = 0;
                for (size_t i = 0; i < a.size(); i++) { se += (double)(a[i] - b[i]) * (a[i] - b[i]); sr += (double)a[i] * a[i]; }
                if (se > 1e-6 * sr) logf_("FXCHK layer %d expert %d m %d fmt %d: rel L2 %.3e (staged %d)\n", l, e, m, (int)f, sqrt(se / (sr + 1e-30)), ci >= 0 ? 1 : 0);
            }
        }
        if (ci >= 0) {
            DCK(cudaEventRecord(c.st_free[ci % NST], es));
            if (ci + NST < nc) enqueue_copy(ci + NST);
        }
        if (pfj[oi] >= 0) DCK(cudaEventRecord(c.pf_fre[pfj[oi]], es));   // [dsv4-pf3-pfx] slot read
        if (fx) {
            if (xs) pf_rec(E, pfo, 0, es, l, PK_XG2, xfam, 0, xsid, xaux);
        } else {
        k_gather_bf16<<<cdiv((size_t)m * D / 8, 256), 256, 0, es>>>(xq, D, rws, m, Xe);
        gemm_bf16(eb, Xe, D, c.w13, D, H, 2 * INTER, m, 2 * INTER, D);
        if (xs) pf_rec(E, pfo, 0, es, l, PK_XG1, xfam, 0, xsid, xaux);
        k_glu_rows<<<cdiv((size_t)m * INTER, 256), 256, 0, es>>>(H, c.rwt + base[e], m, hf);
        q_fp8(es, hf, INTER, m, INTER, nullptr, 0, hq, INTER);
        if (xs) pf_rec(E, pfo, 0, es, l, PK_XGQ, xfam, 0, xsid, xaux);
        gemm_bf16(eb, hq, INTER, c.w2, INTER, Yd, D, m, D, INTER);
        if (xs) pf_rec(E, pfo, 0, es, l, PK_XG2, xfam, 0, xsid, xaux);
        }
        if (early && ovl) {   // [dsv4-pf8-ovl] its rows written: the in-order add on st waits for this event
            DCK(cudaEventRecord(c.ev_oe[eidx[oi]], c.sxc));
        }
        if (early) continue;   // [dsv4-pf-ilv] (no sampled scatter event: its add comes later)
        if (!shadow.empty() && shadow[oi] != W.card) {
            Card& r = E->cards[shadow[oi]];
            const float* Yr = E->hostx ? r.hrows + (size_t)yoff[oi] * D   // [dsv4-rel-hostx]
                                       : (const float*)((bf16*)r.phase + (size_t)n * D) + (size_t)yoff[oi] * D;
            for (int sp = 0; !pxr[oi].load(std::memory_order_acquire); sp++) if (sp < 4096) _mm_pause(); else std::this_thread::yield();   // [dsv4-pf-mtq]
            DCK(cudaStreamWaitEvent(st, r.ev_px[pxi[oi]], 0));
            k_cmp_bits<<<cdiv((size_t)m * D, 256), 256, 0, st>>>(Y, Yr, (size_t)m * D, d_bad);
        }
        k_scatter_add<<<cdiv((size_t)m * D, 256), 256, 0, st>>>(Y, rws, m, c.ao);
        if (xs) pf_rec(E, pfo, 0, st, l, PK_XSC, xfam, 0, xsid, xaux);
    }
    if (ovl) {   // [dsv4-pf8-ovl] (every early expert was waited by its add; the join keeps sxc ordered before reuse)
        if (!oend) DCK(cudaEventRecord(c.ev_oxe, c.sxc));
        DCK(cudaStreamWaitEvent(st, c.ev_oxe, 0));
    }
    if (!sdone) add_s();   // [dsv4-pf9-hsum] (no tail rows: S last)
    if (pf) {
        if (!pf_res) pf_rec(E, pfo, 0, st, l, PK_MRES);
        pf_rec(E, pfo, 0, st, l, PK_MHOST);
    }
    for (auto& t : rth) t.join();   // [dsv4-pf-mtq]: every remote card's work is enqueued
    rth.clear();
    DCK(cudaSetDevice(c.dev));
    if (xqh && !xq_pull) DCK(cudaStreamWaitEvent(st, c.ev_xqd, 0));   // [dsv4-pf8-xqh] the pushes read opk before anything reuses it
    if (!cjobs.empty()) {   // [dsv4-pf2-cpu]
        E->pfcpu.wait_all();
        cpu_join.p = nullptr;
        DCK(cudaEventRecord(c.ev_cy, st));
        E->cy_last = c.ev_cy;
        if (pf) {
            size_t rows = 0, by = 0;
            for (const auto& J : cjobs) { rows += J.m; by += expert_bytes((XFmt)J.fmt); }
            logf_("pfcpu layer %d: %zu experts, %zu rows, %.2f GB, CPU busy %.1f ms, done %.1f ms after the start\n", l, cjobs.size(),
                  rows, by / 1e9, E->pfcpu.busy_ms, ms_since(cpu_t0));
        }
    }
    if (E->pfx_l == l) {   // [dsv4-pf3-pfx] unused slots (mispredicted, CPU, seeded, sent back): their copy done before
        // the card's stream reuses the region (and before the next copy into the slot)
        std::vector<char> usd(NE, 0);
        if (pfx)
            for (size_t oi = 0; oi < order.size(); oi++)
                if (pfj[oi] >= 0) usd[order[oi]] = 1;
        for (int e = 0; e < NE; e++)
            if (E->pfx_card[e] >= 0 && !usd[e]) {
                Card& k = E->cards[E->pfx_card[e]];
                DevGuard dg(k.dev);
                DCK(cudaStreamWaitEvent(k.st, k.pf_rdy[E->pfx_j[e]], 0));
                DCK(cudaEventRecord(k.pf_fre[E->pfx_j[e]], k.st));
            }
        E->pfx_l = -1;
        DCK(cudaSetDevice(c.dev));
    }
    // R12 (review r11 L1): a remote card's later work (next layer / next chunk) may overwrite its phase region only after
    // the owner has read its rows; host syncs ordered this before, now an explicit event does
    for (int k = 0; k < NC; k++) {
        if (!rwork[k]) continue;
        {
            DevGuard dg(c.dev);
            if (!c.ev_pdone) DCK(cudaEventCreateWithFlags(&c.ev_pdone, cudaEventDisableTiming));
            DCK(cudaEventRecord(c.ev_pdone, st));
        }
        for (int k2 = k; k2 < NC; k2++)
            if (rwork[k2]) {
                DevGuard dg(E->cards[k2].dev);
                DCK(cudaStreamWaitEvent(E->cards[k2].st, c.ev_pdone, 0));
            }
        break;
    }
    if (!shadow.empty()) {
        unsigned hb[2] = {};
        DCK(cudaMemcpyAsync(hb, d_bad, 8, cudaMemcpyDefault, st));
        DCK(cudaStreamSynchronize(st));
        int nr = 0;
        for (int a : shadow) nr += a != W.card;
        logf_("pdchk layer %d n %d: %d experts computed remotely, %u output values differ from the owner's\n", l, n, nr, hb[0]);
    }
    if (pf) g_pf.lh[l].enq += ms_since(pf_tm) - pf_rs;
    // [dsv4-pf2-klfloor] quality-gate noise band (debug): DSV4_PERT=eps:layer scales the layer's FFN output (shared +
    // routed experts, prefill chunks only) by (1 + eps); layer -1 = every layer; a value without ':' is ignored
    static const char* pe = getenv("DSV4_PERT");
    static const bool pe_ok = pe && strchr(pe, ':');
    if (pe_ok) {
        const float eps = (float)atof(pe);
        const char* cl = strchr(pe, ':');
        if (atoi(cl + 1) == l || atoi(cl + 1) < 0) k_pert_scale<<<cdiv((size_t)n * D, 256), 256, 0, c.st>>>(c.ao, (size_t)n * D, 1.f + eps);
    }
    // [dsv4-pf9-hsum] debug DSV4_AODUMP=PREFIX: the FFN output (shared + routed) of layers 0 / 1 of every chunk to
    // PREFIX_<chunk number>_<n>_<layer>.bin (fp32 [n][D]): per-stage error of an order-only change from two runs
    static const char* aod = getenv("DSV4_AODUMP");
    static int aod_k = 0;   // review pf9 L3: a running chunk number (equal-size chunks do not overwrite each other)
    if (aod && l == 0) aod_k++;
    if (aod && l <= 1) {
        std::vector<float> h((size_t)n * D);
        DCK(cudaMemcpyAsync(h.data(), c.ao, h.size() * 4, cudaMemcpyDeviceToHost, c.st));
        DCK(cudaStreamSynchronize(c.st));
        char fn[512];
        snprintf(fn, sizeof fn, "%s_%d_%d_%d.bin", aod, aod_k, n, l);
        if (FILE* f = fopen(fn, "wb")) { fwrite(h.data(), 4, h.size(), f); fclose(f); }
    }
}

// device table write of one entry (stream order on the compute stream; pinned ring source)
static void dset(Card& c, int* dtab, int idx, int val) {
    DevGuard g(c.dev);   // R9: run-ahead calls this with another card current
    k_set_int<<<1, 1, 0, c.est>>>(dtab + idx, val);   // a kernel, not an H2D copy: never queues behind an admission
}
// R9: residency of expert e in FP4 global slot g (-1: none): host mirror, the owner's routing table (any card) and the
// slot card's local table (GEMV kernels), each in stream order of its card
static void set_slot(Engine* E, Layer& W, int e, int g) {
    const int old = W.h_slot[e];
    W.h_slot[e] = g;
    dset(E->cards[W.card], W.d_slot, e, g);
    const int k = g >= 0 ? gs_card(g) : old >= 0 ? gs_card(old) : -1;
    if (old >= 0 && g >= 0 && gs_card(old) != k) dset(E->cards[gs_card(old)], W.d_ls[gs_card(old)], e, -1);
    if (k >= 0) dset(E->cards[k], W.d_ls[k], e, g >= 0 ? gs_idx(g) : -1);
}
static uint8_t* slot_ptr(Engine* E, int g) { return E->cards[gs_card(g)].slots + (size_t)gs_idx(g) * expert_bytes(XF_FP4); }
// R16: non-FP4 pool helpers (the same names as the GGUF branch's): format of layer l's pool, a slot's card and address.
// Pool split: slot ids are global (card * GS + index on that card); else indices into the owner card's pool
static XFmt pool_fmt(const Engine* E, int l) { return E->gg ? E->gg->fmt[l] : E->pfmt; }   // (GGUF mode: per layer)
static Card& pool_card(Engine* E, Card& c, int slot) { return E->psplit ? E->cards[gs_card(slot)] : c; }
// pool split: card k's layer-l region starts at ploff[l]; the GEMV kernels get that base and in-layer indices (d_dls)
static uint8_t* pool_base(Engine* E, Card& k, int l) { return k.dpool + (E->psplit ? k.ploff[l] : 0); }
static int pool_lidx(Engine* E, int l, int slot) { return E->psplit ? gs_idx(slot) - E->cards[gs_card(slot)].ps0[l] : slot; }
static uint8_t* pool_slot_ptr(Engine* E, Card& c, int l, int slot) {
    if (E->gg && !E->psplit) return gg_pool_ptr(E, l, slot);   // R16 [dsv4-gguf-iq]: per-layer regions of the owner
    const size_t xb = expert_bytes(pool_fmt(E, l));
    return E->psplit ? pool_base(E, E->cards[gs_card(slot)], l) + (size_t)pool_lidx(E, l, slot) * xb : c.dpool + (size_t)slot * xb;
}
// residency of expert e in pool slot g (-1: none): host mirror, the owner's routing table and (pool split) the slot
// card's local table, each in stream order of its card (as set_slot)
static void set_pslot(Engine* E, Layer& W, int e, int g) {
    const int old = W.h_dslot[e];
    W.h_dslot[e] = g;
    dset(E->cards[W.card], W.d_dslot, e, g);
    if (!E->psplit) return;
    const int k = g >= 0 ? gs_card(g) : old >= 0 ? gs_card(old) : -1;
    if (old >= 0 && g >= 0 && gs_card(old) != k) dset(E->cards[gs_card(old)], W.d_dls[gs_card(old)], e, -1);
    if (k >= 0) dset(E->cards[k], W.d_dls[k], e, g >= 0 ? pool_lidx(E, W.lid, g) : -1);
}
// R14 swap verifier (DSV4_SWAPCHK=1): FP4 bytes of (l, e) as stored in the checkpoint vs a host buffer
static const bool g_swapchk = getenv("DSV4_SWAPCHK") != nullptr;
static long g_chk_n = 0, g_chk_bad = 0;
static void swap_check(Engine* E, int l, int e, const uint8_t* got, const char* what, int g = -1) {
    std::string px = "layers." + std::to_string(E->L[l].lid) + ".ffn.experts." + std::to_string(e) + ".";
    size_t off = 0;
    bool ok = true;
    for (const char* wn : {"w1", "w3", "w2"})
        for (const char* kn : {".weight", ".scale"}) {
            const TensorInfo& t = E->stf.get(px + wn + kn);
            ok &= memcmp(got + off, t.data, t.nbytes) == 0;
            off += t.nbytes;
        }
    g_chk_n++;
    if (!ok) {   // stop at the first one (coordinator): expert, slot, step
        g_chk_bad++;
        logf_("SWAPCHK MISMATCH %s: layer %d expert %d slot %d (card %d index %d) pos %d pass %d after %ld checks\n", what, l, e,
              g, g >= 0 ? gs_card(g) : -1, g >= 0 ? gs_idx(g) : -1, E->pos, E->pass_clock, g_chk_n);
        DCHECK(false, "swap verifier: bytes differ from the checkpoint");
    }
}
static void slot_check(Engine* E, int l, int e, int g, const char* what) {   // read the slot back (sync) and compare
    static std::vector<uint8_t> buf(expert_bytes(XF_FP4));
    Card& k = E->cards[gs_card(g)];
    DevGuard dg(k.dev);
    DCK(cudaStreamSynchronize(k.cp));
    DCK(cudaMemcpy(buf.data(), slot_ptr(E, g), buf.size(), cudaMemcpyDeviceToHost));
    swap_check(E, l, e, buf.data(), what, g);
}
// finished copies become visible (before this layer's kernels of the step)
// R21 [dsv4-r21-adm1]: minage = passes from the decision (spec_commit at the end of pass p, pend_t = p) to visibility:
// 2 (default) = at the boundary of pass p + 2; 1 (DSV4_ADM1=1, multi-row pass boundaries only) = at the boundary of
// pass p + 1, after the draft, waiting there for a copy the draft did not cover (never inside a pass)
static void dyn_apply(Engine* E, Card& c, Layer& W, int ahead = 0, int minage = 2) {   // ahead: called one step early (R10)
    const size_t XB = expert_bytes(XF_FP4);
    for (size_t i = 0; i < W.dpool.size(); i++) {
        int e = W.pend_e[i];
        if (e < 0) continue;
        if (W.pend_job[i] >= 0) {   // FP4 bytes from the checkpoint: copy at decision + 2 tokens, visible at + 3
            XJob& J = c.xj[W.pend_job[i]];
            const int age = E->pass_clock + ahead - W.pend_t[i];   // in passes (review r7 M2): a DSpark pass is one step
            if (J.state.load(std::memory_order_acquire) != 3) {
                if (age < 2) continue;
                int js;
                for (auto tw = clk::now(); (js = J.state.load(std::memory_order_acquire)) != 2 && js != 4;) {
                    std::this_thread::sleep_for(std::chrono::microseconds(50));
                    DCHECK(ms_since(tw) < 120000, "NVMe expert read (admission): no answer for 120 s");   // review r10 L4
                }
                if (js == 4) {   // read failed (review r6 M3): drop the admission, the slot stays free, lane serves e
                    J.state.store(0, std::memory_order_release);
                    W.pend_job[i] = -1;
                    W.pend_e[i] = -1;
                    if (++E->nv_fail >= 8 && !E->nv_off) {   // 8 failures in a row (reset on success, review r7 L10)
                        E->nv_off = true;
                        fprintf(stderr, "warning: %ld checkpoint expert reads failed: NVMe admissions off\n", E->nv_fail.load());
                    }
                    continue;
                }
                if (J.prep) {   // [dsv4-rel-ram]: the re-packed GGUF expert into its pool slot, one copy
                    Card& sc = pool_card(E, c, W.dpool[i]);
                    DevGuard dg(sc.dev);
                    DCK(cudaEventRecord(sc.ev_dyn, sc.est));
                    DCK(cudaStreamWaitEvent(sc.cp, sc.ev_dyn, 0));
                    DCK(cudaMemcpyAsync(pool_slot_ptr(E, c, W.lid, W.dpool[i]), J.buf, expert_bytes(pool_fmt(E, W.lid)),
                                        cudaMemcpyHostToDevice, sc.cp));
                    DCK(cudaEventRecord(W.pend_ev[i], sc.cp));
                    J.state.store(3, std::memory_order_release);
                    continue;
                }
                Card& sc = E->cards[gs_card(W.dpool[i])];   // R9: the slot's card
                DevGuard dg(sc.dev);
                DCK(cudaEventRecord(sc.ev_dyn, sc.est));   // after the victim's last reads (it was invalidated earlier)
                DCK(cudaStreamWaitEvent(sc.cp, sc.ev_dyn, 0));
                uint8_t* dst = slot_ptr(E, W.dpool[i]);
                size_t o2 = 0;
                for (int j = 0; j < 6; j++) {
                    DCK(cudaMemcpyAsync(dst + o2, J.buf + J.boff[j] + J.delta[j], J.len[j], cudaMemcpyHostToDevice, sc.cp));
                    o2 += J.len[j];
                }
                DCK(cudaEventRecord(W.pend_ev[i], sc.cp));
                J.state.store(3, std::memory_order_release);
                continue;
            }
            if (age < 3) continue;
            DCK(cudaEventSynchronize(W.pend_ev[i]));
            E->nv_fail = 0;
            J.state.store(0, std::memory_order_release);
            W.pend_job[i] = -1;
        } else if (!W.pend_m.empty() && W.pend_m[i] >= 0) {
            // R13 swap, phase 1: the victim's bytes are in its new home after 2 passes: the victim moves to the lane,
            // then the candidate's copy into the slot (as an ordinary admission: visible 2 passes later)
            if (E->pass_clock + ahead - W.pend_t[i] < 2) continue;
            DCK(cudaEventSynchronize(W.pend_ev[i]));
            const int ve = W.pend_m[i];
            if (E->gswap) {   // [dsv4-rel-xswap]: the victim's bytes are its host copy now; then the candidate's copy
                W.hx[ve] = W.pend_h[i];
                set_pslot(E, W, ve, -1);
                W.dyn_e[i] = -1;
                W.pend_m[i] = -1;
                W.pend_h[i] = nullptr;
                Card& sc = pool_card(E, c, W.dpool[i]);
                DevGuard dg(sc.dev);
                DCK(cudaEventRecord(sc.ev_dyn, sc.est));   // after the victim's last reads
                DCK(cudaStreamWaitEvent(sc.cp, sc.ev_dyn, 0));
                const size_t PB = expert_bytes(pool_fmt(E, W.lid));
                const uint8_t* src = W.hx[e];
                if (!src) {   // a candidate without a copy: re-packed from the GGUF (synchronous; rare: the RAM path)
                    static uint8_t* tmp = nullptr;
                    static size_t tmpb = 0;
                    if (tmpb < PB) { if (tmp) cudaFreeHost(tmp); DCK(cudaHostAlloc((void**)&tmp, PB, cudaHostAllocPortable)); tmpb = PB; }
                    DCK(cudaStreamSynchronize(sc.cp));
                    gg_repack_expert(E, W.lid, e, tmp);
                    src = tmp;
                }
                DCK(cudaMemcpyAsync(pool_slot_ptr(E, c, W.lid, W.dpool[i]), src, PB, cudaMemcpyHostToDevice, sc.cp));
                if (src != W.hx[e]) DCK(cudaStreamSynchronize(sc.cp));   // the re-pack buffer is reused
                DCK(cudaEventRecord(W.pend_ev[i], sc.cp));
                W.pend_t[i] = E->pass_clock + ahead;
                if (E->tm) E->tm->host_bytes += PB;
                continue;
            }
            if (g_swapchk) swap_check(E, W.lid, ve, W.pend_h[i], "home after D2H");
            W.hx[ve] = W.pend_h[i];
            W.hfmt[ve] = XF_FP4;
            set_slot(E, W, ve, -1);
            W.inpool[ve] = 0;
            W.dyn_e[i] = -1;
            W.pend_m[i] = -1;
            W.pend_h[i] = nullptr;
            Card& sc = E->cards[gs_card(W.dpool[i])];
            DevGuard dg(sc.dev);
            DCK(cudaEventRecord(sc.ev_dyn, sc.est));   // after the victim's last reads
            DCK(cudaStreamWaitEvent(sc.cp, sc.ev_dyn, 0));
            DCK(cudaMemcpyAsync(slot_ptr(E, W.dpool[i]), W.hx[e], XB, cudaMemcpyHostToDevice, sc.cp));
            DCK(cudaEventRecord(W.pend_ev[i], sc.cp));
            W.pend_t[i] = E->pass_clock + ahead;
            if (g_admlog) logf_("adm swap move pc %d ahead %d l %d e %d victim %d slot %zu\n", E->pass_clock, ahead, W.lid, e, ve, i);
            if (E->tm) E->tm->host_bytes += XB;
            continue;
        } else {
            // R10 graph: dyn_apply runs at the step boundary, right after the step whose tier update issued the copy
            // (it starts when that step's graph ends): visible one step later, so the boundary never waits on a copy
            // review r10 M1: the same 2-pass rule without the graph (DSV4_NOGRAPH=1 stays an exact reference)
            if (E->pass_clock + ahead - W.pend_t[i] < minage) continue;
            DCK(cudaEventSynchronize(W.pend_ev[i]));   // done long ago (one token later); waiting keeps runs deterministic
        }
        W.pend_e[i] = -1;
        W.dyn_e[i] = e;
        if (g_admlog) logf_("adm apply pc %d ahead %d l %d e %d slot %zu\n", E->pass_clock, ahead, W.lid, e, i);
        if (E->dyn_fp4) {
            if (g_swapchk) slot_check(E, W.lid, e, W.dpool[i], "slot at visibility");
            set_slot(E, W, e, W.dpool[i]);
            W.inpool[e] = 1;
            // R13 swap: a resident FP4-host expert gives its host buffer to the free homes (the lane no longer reads it:
            // the tables say resident from here; prefetch workers only ever read). Not in the borrowable tail.
            bool dup = false;   // review r13 H1 (guard): never recycle the copy of an expert listed in another entry
            for (size_t k = 0; k < W.dyn_e.size(); k++) dup |= k != i && W.dyn_e[k] == e;
            if (E->swap && W.hx[e] && W.hfmt[e] == XF_FP4 && !in_tail(E, W.dpool[i]) && !dup && is_pinned(W.hx[e])) {
                E->hfree.push_back({(uint8_t*)W.hx[e], !W.own.empty() && W.own[e]});
                W.hx[e] = nullptr;
                W.hfmt[e] = -1;
                if (!W.own.empty()) W.own[e] = 0;
                E->nrecyc++;
            }
        } else {
            set_pslot(E, W, e, W.dpool[i]);
            // [dsv4-rel-xswap]: a resident expert's host copy becomes a free home (not in the borrowable tail: a
            // borrow restores from the copy or the file; not if another entry lists it; pinned only, for async D2H)
            bool dup = false;
            for (size_t k = 0; k < W.dyn_e.size(); k++) dup |= k != i && W.dyn_e[k] == e;
            if (E->gswap && W.hx[e] && !in_gtail(E, W.dpool[i]) && !dup && is_pinned(W.hx[e])) {
                E->ghome[expert_bytes(pool_fmt(E, W.lid))].push_back((uint8_t*)W.hx[e]);
                W.hx[e] = nullptr;
                E->ngrecyc++;
            }
        }
    }
}
// policy after the step's routing of this layer: LFU with decay, admit on the second miss within the window
static void early_finalize(Engine* E, Layer& W, const int* ids, int n);
static void dyn_update(Engine* E, Card& c, Layer& W, const int* ids, int n) {
    early_finalize(E, W, ids, n);   // dsv4-fut [dsv4-fut-early] (off: returns at once)
    const int t = E->dyn_clock;
    auto sc = [&](int e) { return W.score[e] * powf(E->o.dyn_decay, (float)(t - W.score_t[e])); };
    for (int i = 0; i < n; i++) {
        int e = ids[i];
        if (e < 0) continue;   // R8: left out by adaptive top-k
        W.score[e] = sc(e) + 1.f;
        W.score_t[e] = t;
    }
    if (W.dpool.empty()) return;
    for (int i = 0; i < n; i++) {
        int e = ids[i];
        if (e < 0) continue;
        const XFmt pf = E->dyn_fp4 ? XF_FP4 : pool_fmt(E, W.lid);   // format of the pool slots (review r16 L1)
        // [dsv4-rel-ram]: a GGUF expert without a host copy is admitted from the file (re-packed by a reader thread)
        const bool ngg = E->gm && !W.hx[e] && W.hfmt[e] >= 0;
        if (W.h_slot[e] >= 0 || W.h_dslot[e] >= 0 || (!W.hx[e] && !ngg)) continue;
        if (!W.bor.empty() && W.bor[e]) continue;   // review r13 H1: evicted for the prefill only, restored after it
        const bool nv = (E->dyn_fp4 && W.hfmt[e] != XF_FP4) || ngg;   // FP4 bytes from the checkpoint / GGUF re-pack
        if (!E->dyn_fp4 && W.hfmt[e] != pf) continue;
        int jf = -1;
        if (nv) {
            if (E->nv_off) continue;
            for (int j = 0; j < XJ_PER_CARD; j++)
                if (c.xj[j].state.load(std::memory_order_acquire) == 0) { jf = j; break; }
            if (jf < 0) continue;   // all checkpoint reads of this card busy: stays on the lane
        }
        bool dup = false;   // rows of one pass that share an expert are one miss (review r6 M1)
        for (int k = 0; k < i; k++) dup |= ids[k] == e;
        if (dup) continue;
        bool pending = false;
        for (int pe : W.pend_e) pending |= pe == e;
        if (pending) continue;
        bool second = t - W.miss_t[e] <= E->o.dyn_window;
        W.miss_t[e] = t;
        if (!second) continue;
        if (nv && E->o.nv_cap > 0) {   // rate cap (NVMe bandwidth, R7 plan), burst of 4 tokens; split evenly over the
            // cards so that the first layers do not spend the whole engine's credit (review r7 M3)
            const float cap = E->o.nv_cap / (float)E->cards.size();
            c.nv_credit = std::min(4.f * cap + 1.f, c.nv_credit + cap * (float)(t - c.nv_t));
            c.nv_t = t;
            if (c.nv_credit < 1.f) continue;
        }
        // victim: a free slot, else the lowest score not used by this step and not being copied
        // R13 swap tier: a victim without a host copy needs a free home (and an FP4-host candidate: not the NVMe path)
        bool homes = E->swap && !nv && !E->hfree.empty();
        std::vector<uint8_t*>* gh = nullptr;   // [dsv4-rel-xswap]: this layer's size class of free homes
        if (E->gswap) {
            auto it = E->ghome.find(expert_bytes(pool_fmt(E, W.lid)));
            if (it != E->ghome.end() && !it->second.empty()) { gh = &it->second; homes = true; }
        }
        if (homes && E->o.swap_cap > 0) {   // R13: rate cap of the swaps (PCIe beside the LL12 exchanges)
            E->swap_credit = std::min(4.f * E->o.swap_cap + 1.f, E->swap_credit + E->o.swap_cap * (float)(t - E->swap_t));
            E->swap_t = t;
            homes = E->swap_credit >= 1.f;
        }
        int v = -1;
        float best = 1e30f;
        for (size_t j = 0; j < W.dpool.size(); j++) {
            if (W.pend_e[j] != -1) continue;   // copying, or borrowed (-2)
            int ve = W.dyn_e[j];
            if (ve < 0) { v = (int)j; break; }
            // [dsv4-rel-ram]: a GGUF victim without a copy falls back to the file, except the seeds that never had one
            // (static: evicting them would move the whole VRAM set onto NVMe reads)
            // [dsv4-rel-xswap]: with the exclusive swap a copy-less victim waits for a free home (never the file path)
            if (!W.hx[ve] && !homes && (!E->gm || E->gswap || (!W.pin.empty() && W.pin[ve]))) continue;
            bool used = false;
            for (int k = 0; k < n; k++) used |= ids[k] == ve;
            if (used) continue;
            float s2 = sc(ve);
            if (s2 < best) { best = s2; v = (int)j; }
        }
        // replace only a colder resident: always replacing (the G2' simulation's rule) thrashed: 1.7 admissions per
        // token-layer, 13.8 MB each, and the copies cost more than the hits saved (R2-6)
        // review r13 L10: the cb3 / non-swap tiers keep the old rule. R20 [dsv4-r20-ggmargin]: the GGUF pool (every expert
        // has a host copy, no swap needed) takes the margin when --dyn-margin is given (A/B; default unchanged until measured)
        const float mg = E->swap || (E->gg && E->o.dyn_margin_set) ? E->o.dyn_margin : 0.f;
        if (v < 0 || (W.dyn_e[v] >= 0 && best + mg >= sc(e))) continue;
        if (W.dyn_e[v] >= 0 && !W.hx[W.dyn_e[v]] && E->gm && gh && homes) {   // [dsv4-rel-xswap]: GGUF pool swap
            const int ve = W.dyn_e[v];
            uint8_t* hb = gh->back();
            gh->pop_back();
            Card& vc = pool_card(E, c, W.dpool[v]);
            DevGuard dg(vc.dev);
            DCK(cudaEventRecord(vc.ev_dyn, vc.est));   // after the victim's last reads issued so far
            DCK(cudaStreamWaitEvent(vc.cpd, vc.ev_dyn, 0));
            DCK(cudaMemcpyAsync(hb, pool_slot_ptr(E, c, W.lid, W.dpool[v]), expert_bytes(pool_fmt(E, W.lid)),
                                cudaMemcpyDeviceToHost, vc.cpd));
            DCK(cudaEventRecord(W.pend_ev[v], vc.cpd));
            W.pend_m[v] = ve;
            W.pend_h[v] = hb;
            W.pend_e[v] = e;
            W.pend_t[v] = E->pass_clock;
            E->ngswap++;
            E->swap_credit -= 1.f;
            if (g_admlog) logf_("adm issue gswap pc %d l %d e %d victim %d slot %d\n", E->pass_clock, W.lid, e, ve, v);
            if (E->tm) {
                E->tm->nadmit++;
                E->tm->host_bytes += expert_bytes(pool_fmt(E, W.lid));   // the D2H write
            }
            continue;
        }
        if (W.dyn_e[v] >= 0 && !W.hx[W.dyn_e[v]] && !E->gm) {   // R13 swap: the victim's slot bytes to a free home first
            const int ve = W.dyn_e[v];
            auto hh = E->hfree.back();
            E->hfree.pop_back();
            if (g_swapchk)   // R14 verifier: a free home must not still be some expert's host copy
                for (int l2 = 0; l2 < NL; l2++)
                    for (int x = 0; x < NE; x++)
                        DCHECK(E->L[l2].hx[x] != hh.first, "swap verifier: free home is the host copy of layer %d expert %d", l2, x);
            Card& vc = E->cards[gs_card(W.dpool[v])];
            DevGuard dg(vc.dev);
            DCK(cudaMemcpyAsync(hh.first, slot_ptr(E, W.dpool[v]), expert_bytes(XF_FP4), cudaMemcpyDeviceToHost, vc.cpd));
            DCK(cudaEventRecord(W.pend_ev[v], vc.cpd));
            W.pend_m[v] = ve;
            W.pend_h[v] = hh.first;
            if (W.own.empty()) W.own.assign(NE, 0);
            W.own[ve] = hh.second;   // carried with the buffer (the memory guard frees only own allocations)
            W.pend_e[v] = e;
            W.pend_t[v] = E->pass_clock;
            E->nswap++;
            E->swap_credit -= 1.f;
            if (g_admlog) logf_("adm issue swap pc %d l %d e %d victim %d slot %d\n", E->pass_clock, W.lid, e, ve, v);
            if (E->tm) {
                E->tm->nadmit++;
                E->tm->host_bytes += expert_bytes(XF_FP4);   // the D2H write (the H2D read is counted at its issue)
            }
            continue;
        }
        if (W.dyn_e[v] >= 0) {   // invalidate the victim in stream order, before the copy may overwrite it
            const int ve = W.dyn_e[v];
            if (E->dyn_fp4) {
                set_slot(E, W, ve, -1);
                W.inpool[ve] = 0;
            } else {
                set_pslot(E, W, ve, -1);
            }
            W.dyn_e[v] = -1;
        }
        if (nv && ngg) {   // [dsv4-rel-ram]
            XJob& J = c.xj[jf];
            J.prep = [](XJob* j) { gg_repack_expert((const Engine*)j->ctx, j->pl, j->pe, j->buf); };
            J.ctx = E;
            J.pl = W.lid;
            J.pe = e;
            J.state.store(1, std::memory_order_release);
            E->xr.push(&J);
            c.nv_credit -= 1.f;   // review rel L1: a re-pack reads the file too (page cache or NVMe): charged like nv
            W.pend_job[v] = jf;
            W.pend_t[v] = E->pass_clock;
            W.pend_e[v] = e;
            if (E->tm) {
                E->tm->nadmit++;
                E->tm->host_bytes += 2.0 * expert_bytes(pf);
            }
            continue;
        }
        if (nv) {
            XJob& J = c.xj[jf];
            J.prep = nullptr;
            std::string px = "layers." + std::to_string(W.lid) + ".ffn.experts." + std::to_string(e) + ".";
            size_t bo = 0;
            int j = 0;
            bool nofile = false;
            for (const char* wn : {"w1", "w3", "w2"})
                for (const char* kn : {".weight", ".scale"}) {
                    if (nofile) continue;
                    const TensorInfo& t = E->stf.get(px + wn + kn);
                    size_t off = 0;
                    std::string path = E->stf.file_of(t.data, &off);
                    auto it = E->xfd.find(path);
                    if (it == E->xfd.end()) {
                        int fd = open(path.c_str(), O_RDONLY | O_DIRECT);
                        if (fd < 0) {   // review r6 M3: no O_DIRECT here (tmpfs, FUSE): buffered reads, else no admissions
                            fprintf(stderr, "warning: O_DIRECT open %s: %s; buffered reads\n", path.c_str(), strerror(errno));
                            fd = open(path.c_str(), O_RDONLY);
                        }
                        if (fd < 0) {
                            fprintf(stderr, "warning: cannot open %s: NVMe admissions off\n", path.c_str());
                            E->nv_off = true;
                            nofile = true;   // review r7 L10: the slot stays free; the layer's RAM admissions go on
                            continue;
                        }
                        it = E->xfd.emplace(path, fd).first;
                    }
                    J.fd[j] = it->second;
                    J.off[j] = off;
                    J.len[j] = (int)t.nbytes;
                    J.boff[j] = bo;
                    bo += ((off + t.nbytes + 4095) & ~(size_t)4095) - (off & ~(size_t)4095);
                    j++;
                }
            if (nofile) continue;
            DCHECK(bo <= XJ_BYTES, "expert bounce %zu", bo);
            J.state.store(1, std::memory_order_release);
            E->xr.push(&J);
            c.nv_credit -= 1.f;
            W.pend_job[v] = jf;
            W.pend_t[v] = E->pass_clock;
            if (g_admlog) logf_("adm issue nv pc %d l %d e %d slot %d\n", E->pass_clock, W.lid, e, v);
            W.pend_e[v] = e;
            if (E->tm) {
                E->tm->nadmit++;
                E->tm->host_bytes += 2.0 * expert_bytes(XF_FP4);   // DMA into the bounce buffer, then out of it
            }
            continue;
        }
        Card& sc = E->dyn_fp4 ? E->cards[gs_card(W.dpool[v])] : pool_card(E, c, W.dpool[v]);   // R9 / R16: the slot's card
        DevGuard dg(sc.dev);
        DCK(cudaEventRecord(sc.ev_dyn, sc.est));
        DCK(cudaStreamWaitEvent(sc.cp, sc.ev_dyn, 0));
        const size_t PB = expert_bytes(pf);
        uint8_t* dst = E->dyn_fp4 ? slot_ptr(E, W.dpool[v]) : pool_slot_ptr(E, c, W.lid, W.dpool[v]);
        DCK(cudaMemcpyAsync(dst, W.hx[e], PB, cudaMemcpyHostToDevice, sc.cp));
        DCK(cudaEventRecord(W.pend_ev[v], sc.cp));
        W.pend_e[v] = e;
        W.pend_job[v] = -1;   // review rel H1 (guard): a host-copy admission never inherits a stale re-pack job
        W.pend_t[v] = E->pass_clock;
        if (g_admlog) logf_("adm issue ram pc %d l %d e %d slot %d\n", E->pass_clock, W.lid, e, v);
        if (E->tm) {
            E->tm->nadmit++;
            E->tm->host_bytes += PB;
        }
    }
}

static int pred_pick(Engine* E, Layer& W, int l, int R, CpuJob* xs, std::vector<int>& pref,
                     const volatile int* hp, int nmax) {
    int ce[PRED_STRIDE], cn[PRED_STRIDE], m = 0;
    if (!hp) hp = E->hpred;
    if (nmax < 0) nmax = E->o.prefetch;
    for (int i = 0; i < R * TOPK; i++) {
        int e = hp[l * PRED_STRIDE + i];
        if (e < 0 || e >= NE || W.h_slot[e] >= 0 || W.h_dslot[e] >= 0 || !W.hx[e]) continue;
        int j = 0;
        while (j < m && ce[j] != e) j++;
        if (j == m) { ce[m] = e; cn[m++] = 0; }
        cn[j]++;
    }
    int nx = 0;
    while (nx < nmax && nx < TOPK) {   // stable: ties keep the order of first use (row 0 first)
        int b = -1;
        for (int j = 0; j < m; j++)
            if (cn[j] > 0 && (b < 0 || cn[j] > cn[b])) b = j;
        if (b < 0) break;
        xs[nx++] = {W.hx[ce[b]], (uint8_t)W.hfmt[ce[b]], 0, 0.f};
        pref.push_back(ce[b]);
        cn[b] = 0;
    }
    return nx;
}
// dsv4-fut [dsv4-fut-early]: the routing kernel of this pass shape carries the early decision (run-ahead pass with the
// predictor); must match moe_process's request condition
static bool early_req(const Engine* E, int R) { return E->early && E->ra_on && pred_on(E, R); }
static int dpool_idx(const Layer& W, int g) {
    for (size_t j = 0; j < W.dpool.size(); j++)
        if (W.dpool[j] == g) return (int)j;
    return -1;
}
// the coldest resident of the owner card's pool slots of layer W outside the borrowable tail, not in `busy` and not
// being copied (-1: none); score at clock t
static int early_victim(Engine* E, Layer& W, const int* busy, int nb, int t, float* best_s) {
    int v = -1;
    float best = 1e30f;
    for (size_t j = 0; j < W.dpool.size(); j++) {
        const int g = W.dpool[j];
        if (W.pend_e[j] != -1 || gs_card(g) != W.card || in_gtail(E, g)) continue;
        const int ve = W.dyn_e[j];
        if (ve >= 0) {
            bool used = false;
            for (int k = 0; k < nb; k++) used |= busy[k] == ve;
            if (used) continue;
        }
        const float s2 = ve < 0 ? -1.f : W.score[ve] * powf(E->o.dyn_decay, (float)(t - W.score_t[ve]));
        if (s2 < best) { best = s2; v = (int)j; }
    }
    if (best_s) *best_s = best;
    return v;
}
// at the layer's tier update (end of its pass): the early expert becomes resident (taken) or an ordinary admission
// (late: visible 2 passes later); a used spare is replaced by evicting the coldest owner-card resident
static void early_finalize(Engine* E, Layer& W, const int* ids, int n) {
    if (!E->early || (E->o.dyn_window < 0 && W.early_e < 0 && W.spare_g < 0)) return;   // (window -1: frozen tier, no spare)
    if (W.early_e >= 0) {
        const int j = dpool_idx(W, W.spare_g), e = W.early_e;
        if (W.early_ap) {
            W.pend_e[j] = -1;
            W.dyn_e[j] = e;
            set_pslot(E, W, e, W.spare_g);
            g_early_n[1]++;
        } else {
            W.h_dslot[e] = -1;   // (never set: the lane served it)
            W.pend_e[j] = e;     // an ordinary admission issued in this pass (dyn_apply: event, then visible)
            W.pend_t[j] = E->pass_clock;
            std::swap(W.pend_ev[j], W.early_ev);   // the slot's event = the early copy's (the old one serves the next request)
            g_early_n[2]++;
        }
        W.early_e = -1;
        W.spare_g = -1;
    }
    if (W.spare_g < 0) {
        const int v = early_victim(E, W, ids, n, E->dyn_clock, nullptr);
        if (v < 0) return;
        if (W.dyn_e[v] >= 0) {
            set_pslot(E, W, W.dyn_e[v], -1);
            W.dyn_e[v] = -1;
        }
        W.pend_e[v] = -3;   // the spare: neither free nor a victim
        W.spare_g = W.dpool[v];
    }
}
// moe_process, after the layer's prediction landed: today's admission test on the predicted ids (second miss within the
// window, colder resident on the owner card), the first passing expert is copied into the spare now
static void early_issue(Engine* E, Layer& W, int l, int R) {
    if (!early_req(E, R) || W.spare_g < 0 || W.early_e >= 0 || E->o.dyn_window < 0) return;
    const int t = E->dyn_clock;
    int pr[PRED_STRIDE], np = 0;
    for (int i = 0; i < R * TOPK; i++) {
        const int e = E->hpred[l * PRED_STRIDE + i];
        if (e >= 0 && e < NE) pr[np++] = e;
    }
    float vs = 0.f;
    if (early_victim(E, W, pr, np, t, &vs) < 0) return;
    const XFmt pf = pool_fmt(E, W.lid);
    for (int i = 0; i < np; i++) {
        const int e = pr[i];
        if (W.h_slot[e] >= 0 || W.h_dslot[e] >= 0 || !W.hx[e] || W.hfmt[e] != pf) continue;
        if (!W.bor.empty() && W.bor[e]) continue;
        bool pending = false;
        for (int pe : W.pend_e) pending |= pe == e;
        if (pending || t - W.miss_t[e] > E->o.dyn_window) continue;
        if (vs >= W.score[e] * powf(E->o.dyn_decay, (float)(t - W.score_t[e])) + 1.f) continue;
        Card& oc = E->cards[W.card];
        DevGuard dg(oc.dev);
        DCK(cudaEventRecord(oc.ev_dyn, oc.est));   // review r21 L9: after the spare's last reads, as every admission copy
        DCK(cudaStreamWaitEvent(oc.cpe, oc.ev_dyn, 0));
        DCK(cudaMemcpyAsync(pool_slot_ptr(E, oc, l, W.spare_g), W.hx[e], expert_bytes(pf), cudaMemcpyHostToDevice, oc.cpe));
        DCK(cudaEventRecord(W.early_ev, oc.cpe));
        k_early_ready<<<1, 1, 0, oc.cpe>>>(oc.d_early + l * 4, e, pool_lidx(E, l, W.spare_g), E->seq);
        if (E->rflag[l * 16] == E->seq) g_early_n[3]++;
        W.early_e = e;
        W.early_seq = E->seq;
        W.early_ap = 0;
        W.miss_t[e] = t;
        g_early_n[0]++;
        if (E->tm) {
            E->tm->nadmit++;
            E->tm->host_bytes += expert_bytes(pf);
        }
        return;
    }
}
// [dsv4-dec7-combh] the combine + FFN post after the lane: 4 columns per thread (16-byte loads of the host-mapped lane
// rows, 4x fewer blocks polling the flag; bit-identical); DSV4_NOCPV=1 = the old one-column kernel
static const bool E_cpv_off = getenv("DSV4_NOCPV") != nullptr;
static void comb_post(cudaStream_t st, const float* y, const float* ycold, const int* yflag, int seq, const float* shared,
                      int R, float* out, float* h, const float* post, const float* comb) {
    if (E_cpv_off) k_moe_comb_post<<<cdiv((size_t)R * D, 256), 256, 0, st>>>(y, ycold, yflag, seq, shared, R, out, h, post, comb);
    else k_moe_comb_post_v<4><<<cdiv((size_t)R * D / 4, 256), 256, 0, st>>>(y, ycold, yflag, seq, shared, R, out, h, post, comb);
}
static void moe_decode(Engine* E, Card& c, Layer& W, int l, int R) {
    cudaStream_t st = c.st;
    auto t0 = clk::now();
    if (E->o.dyn && !tl_cap && !E->dyn_bnd.load(std::memory_order_relaxed)) dyn_apply(E, c, W);   // R10 graph / R11 eager one row: at the step boundary
    float* logits = c.f32tmp2;
    lin32(c, W.gate_w, NE, D, c.x, D, R, logits, NE);
    if (g_rec_dx && R == 1) k_rec<<<D / 256, 256, 0, st>>>(c.x, g_rec_dx + (((size_t)(tl_rec_s % REC_T) * NL + l) * 2) * D);
    int* rids = (int*)c.d_rpack;
    float* rw = (float*)(rids + MAXR * TOPK);
    float* xq = rw + MAXR * TOPK;
    const bool ra = E->ra_on;
    char* hrp = ra ? E->h_rp[l] : c.h_rpack;
    if (ra && !E_rp_off) {   // R10: routing + q_fp8 + the host pack (mapped writes) + flag in one launch (bit-identical)
        char* drp = E->d_rp[l];
        k_route_pub<<<route_pub_grid(R, true), 384, 0, st>>>(
            logits, W.gate_b, R, c.x, rids, rw, xq, (int*)drp, (float*)(drp + (size_t)MAXR * TOPK * 4),
            (float*)(drp + (size_t)MAXR * TOPK * 8), c.d_cnt, E->d_rflag + l * 16, kseq(E), W.d_slot, E->o.car_margin,
            (c.ndyn || E->psplit) ? W.d_dslot : nullptr, E->o.drop_w, g_xh16, early_req(E, R) ? c.d_early + l * 4 : nullptr,
            early_req(E, R) ? W.d_dls[W.card] : nullptr, early_req(E, R) ? E->d_edec + l * 16 : nullptr);
    } else {
    k_route<<<cdiv((size_t)R * 32, 256), 256, 0, st>>>(logits, W.gate_b, R, rids, rw, W.d_slot, E->o.car_margin,
                                                        (c.ndyn || E->psplit) ? W.d_dslot : nullptr, E->o.drop_w);
    q_fp8(st, c.x, D, R, D, xq, D, nullptr, 0);
    DCK(cudaMemcpyAsync(hrp, c.d_rpack, (size_t)MAXR * TOPK * 8 + (size_t)R * D * 4, cudaMemcpyDeviceToHost, st));
    if (ra || pred_on(E, R))   // routing pack landed (run-ahead: host wakes; else the prefetch stops)
        k_flag<<<1, 1, 0, st>>>(E->d_rflag + l * 16, kseq(E));
    }
    const bool early = ra && early_on(E, l + 1, R);
    if (early) {   // R11: layer l+1's router on this layer's FFN input, beside the expert phase (the GPU waits for the lane)
        const Layer& W2 = E->L[l + 1];
        cudaStream_t s2 = c.st2;
        DCK(cudaEventRecord(c.ev_e0, st));
        DCK(cudaStreamWaitEvent(s2, c.ev_e0, 0));
        switch (R) {   // ([dsv4-dec1-pfmr]: R rows with DSV4_EARLYMR)
            case 1: k_gemv_f32k<1, 4><<<cdiv(NE, 2), 256, 0, s2>>>(W2.gate_w, NE, D, c.x, D, c.d_elg, NE); break;
            case 2: k_gemv_f32k<2, 4><<<cdiv(NE, 2), 256, 0, s2>>>(W2.gate_w, NE, D, c.x, D, c.d_elg, NE); break;
            case 3: k_gemv_f32k<3, 4><<<cdiv(NE, 2), 256, 0, s2>>>(W2.gate_w, NE, D, c.x, D, c.d_elg, NE); break;
            case 4: k_gemv_f32k<4, 4><<<cdiv(NE, 2), 256, 0, s2>>>(W2.gate_w, NE, D, c.x, D, c.d_elg, NE); break;
            case 5: k_gemv_f32k<5, 4><<<cdiv(NE, 2), 256, 0, s2>>>(W2.gate_w, NE, D, c.x, D, c.d_elg, NE); break;
            default: k_gemv_f32k<6, 4><<<cdiv(NE, 2), 256, 0, s2>>>(W2.gate_w, NE, D, c.x, D, c.d_elg, NE); break;
        }
        k_route_pub<<<route_pub_grid(R, false), 384, 0, s2>>>(c.d_elg, W2.gate_b, R, nullptr, c.d_epid, c.d_epw, nullptr,
                                                              E->d_hpred2 + (l + 1) * PRED_STRIDE, nullptr, nullptr,
                                                              c.d_cnt + 1, E->d_pflag2 + (l + 1) * 16, kseq(E), W2.d_slot,
                                                              E->o.car_margin, (c.ndyn || E->psplit) ? W2.d_dslot : nullptr, E->o.drop_w);
        DCK(cudaEventRecord(c.ev_e1, s2));
    }
    const int* h_ids = (const int*)hrp;
    const float* h_w = (const float*)(h_ids + MAXR * TOPK);
    const float* h_xq = h_w + MAXR * TOPK;
    if (E->tm && !ra) DCK(cudaEventRecord(c.tp1, st));   // before ev_route: complete when the host wakes up
    if (R <= MAXR) PP(l, 8, st);
    if (!tl_cap) DCK(cudaEventRecord(c.ev_route, st));
    // R17: the shared expert on the side stream beside the routed experts (it needs only c.x); joined before the combine.
    // R18 (review r17 L8 / L11): one body for both streams; the fork comes after the X4 push to the other cards
    const bool shs = !E_shs_off;
    auto shared_expert = [&](cudaStream_t ss) {
        linq(c, W.sh_w1, c.x, D, R, c.d_shg, 2 * INTER, 0, R == 1 && !E_hcf_off && !(l > 0 && defer_on(E, l - 1, R)) ? c.d_xqs : nullptr,
             ss == st ? nullptr : ss);   // [w1; w3]: g in columns 0 .. 2303, u in 2304 .. 4607 (R10: x quantized by mHC)
        k_glu_rows<<<cdiv((size_t)R * INTER, 256), 256, 0, ss>>>(c.d_shg, c.ones, R, c.d_shg + (size_t)R * 2 * INTER);
        linq(c, W.sh_w2, c.d_shg + (size_t)R * 2 * INTER, INTER, R, c.d_shy, D, 0, nullptr, ss == st ? nullptr : ss);
    };
    // GPU: hot experts + shared expert (queued now, run while the CPU works)
    const size_t XB = expert_bytes(XF_FP4);
    const int* ls = W.d_ls[W.card];
    const unsigned xtag = (E->xseq * 64u + (unsigned)l) * 8u + 4u;   // X4 (x / ids / w), X5 = + 1 (choice rows)
    const int* dls = E->psplit ? W.d_dls[W.card] : W.d_dslot;   // R16: the owner's pool table (local slot indices)
    if (E->xsplit || E->psplit) {   // R9: x / ids / w to the other cards first (their experts start ~5 us later)
        XSeg g4 = xseg({{xq, D, D, 0, -1}, {rids, TOPK, TOPK, 0, -1}, {rw, TOPK, TOPK, 0, -1}});
        uint4* dst[2] = {};
        int nd = 0;
        for (int j = 0; j < (int)E->cards.size(); j++)
            if (j != W.card) dst[nd++] = E->cards[j].rx4;
        xpush(c, g4, R, dst[0], dst[1], xtag);
    }
    if (shs) {
        DCK(cudaEventRecord(c.ev_s0, st));
        DCK(cudaStreamWaitEvent(c.st2, c.ev_s0, 0));
        shared_expert(c.st2);
        DCK(cudaEventRecord(c.ev_s1, c.st2));
    }
    // gate/up: 160 chunks per row, 32 lanes x 5 loads; down: 72 chunks, 8 lanes x 9 loads (round 2)
    // advisor H2 / H3: in GGUF mode with no static FP4 slot on this card the FP4 gate / up writes nothing (every choice
    // has no slot) and the FP4 down only zero-fills d_yx: skipped, the zero-fill as one memset (same zero bits)
    static const bool fp4noop = getenv("DSV4_FP4NOOP") != nullptr;   // 1 = the old launches (A/B)
    const bool nofp4 = E->gg && c.used_slots == 0 && !E->xsplit && !fp4noop;
    if (nofp4) {
    } else if (E_moev2 && !E->xsplit) k_moe_gemv2<32, 5><<<dim3(2 * INTER / 8, TOPK * R), 256, 0, st>>>(xq, 0, rids, ls, c.slots, XB, c.d_gu);
    else k_moe_gemv5<8><<<dim3(2 * INTER / 64, TOPK * R), 256, 0, st>>>(xq, 0, rids, ls, c.slots, XB, c.d_gu);   // round 5: coalesced
    if (E->xsplit) {   // R9: the other cards run their slots' experts of this layer on a copy of xq / ids / w
        for (int j = 0; j < (int)E->cards.size(); j++) {
            if (j == W.card) continue;
            Card& k = E->cards[j];
            DevGuard dg(k.dev);
            int* kids = (int*)k.d_rpack;
            float* kw = (float*)(kids + MAXR * TOPK);
            float* kxq = kw + MAXR * TOPK;
            xpull(k, xseg({{kxq, D, D, 0, -1}, {kids, TOPK, TOPK, 0, -1}, {kw, TOPK, TOPK, 0, -1}}), R, k.rx4, xtag);
            const int* kls = W.d_ls[j];
            k_moe_gemv5<8><<<dim3(2 * INTER / 64, TOPK * R), 256, 0, k.st>>>(kxq, 0, kids, kls, k.slots, XB, k.d_gu);
            if (!E_gluq_off)   // R10: GLU + q_fp8 in one launch
                k_moe_glu_q<<<cdiv((size_t)R * TOPK * INTER, 256), 256, 0, k.st>>>(k.d_gu, kw, kids, kls, R, k.d_hh, nullptr);
            else {
            k_moe_glu<<<cdiv((size_t)R * TOPK * INTER, 256), 256, 0, k.st>>>(k.d_gu, kw, kids, kls, R, k.d_hh, nullptr);
            q_fp8(k.st, k.d_hh, INTER, R * TOPK, INTER, k.d_hh, INTER, nullptr, 0);
            }
            k_moe_gemv5<8><<<dim3(D / 64, TOPK * R), 256, 0, k.st>>>(k.d_hh, 1, kids, kls, k.slots, XB, k.d_yx);
            k_ypush<<<R * TOPK, 256, 0, k.st>>>(kids, kls, k.d_yx, c.rx5 + (size_t)j * MAXR * TOPK * YW,
                                                 c.rx5h + (size_t)j * MAXR * TOPK, xtag + 1);
        }
    }
    const size_t XB3 = expert_bytes(pool_fmt(E, l));   // review r16 L1: the layer's slot stride
    if (E->psplit) {   // R16: the other cards run their pool experts of this layer on a copy of xq / ids / w
        for (int j = 0; j < (int)E->cards.size(); j++) {
            if (j == W.card) continue;
            Card& k = E->cards[j];   // (a card without pool slots still answers X5: empty table)
            DevGuard dg(k.dev);
            int* kids = (int*)k.d_rpack;
            float* kw = (float*)(kids + MAXR * TOPK);
            float* kxq = kw + MAXR * TOPK;
            xpull(k, xseg({{kxq, D, D, 0, -1}, {kids, TOPK, TOPK, 0, -1}, {kw, TOPK, TOPK, 0, -1}}), R, k.rx4, xtag);
            const int* kdl = W.d_dls[j];
            if (E->gg) gg_moe(E, j, l, 0, kxq, kids, kdl, k.d_gu, R, k.st);   // R16 [dsv4-gguf-iq]
            else
            k_moe_m23w<2><<<dim3(2 * INTER / M23_RPB, TOPK * R), 256, 0, k.st>>>(kxq, 0, kids, kdl, pool_base(E, k, l), XB3, k.d_cb, k.d_gu);
            if (!E_gluq_off)   // review r16 L3: the same off-switch as the owner and the R9 remote path
                k_moe_glu_q<<<cdiv((size_t)R * TOPK * INTER, 256), 256, 0, k.st>>>(k.d_gu, kw, kids, kdl, R, k.d_hh, nullptr);
            else {
                k_moe_glu<<<cdiv((size_t)R * TOPK * INTER, 256), 256, 0, k.st>>>(k.d_gu, kw, kids, kdl, R, k.d_hh, nullptr);
                q_fp8(k.st, k.d_hh, INTER, R * TOPK, INTER, k.d_hh, INTER, nullptr, 0);
            }
            if (E->gg) gg_moe(E, j, l, 1, k.d_hh, kids, kdl, k.d_yx, R, k.st);
            else
            k_moe_m23w<3><<<dim3(D / M23_RPB, TOPK * R), 256, 0, k.st>>>(k.d_hh, 1, kids, kdl, pool_base(E, k, l), XB3, k.d_cb, k.d_yx);
            k_ypush<<<R * TOPK, 256, 0, k.st>>>(kids, kdl, k.d_yx, c.rx5 + (size_t)j * MAXR * TOPK * YW,
                                                 c.rx5h + (size_t)j * MAXR * TOPK, xtag + 1);
        }
    }
    if (c.ndyn && E->gg)   // R16 [dsv4-gguf-iq]: Maya-S's own bytes (dls: local slot indices of this card)
        gg_moe(E, W.card, l, 0, xq, rids, dls, c.d_gu, R, st);
    else if (c.ndyn && pool_fmt(E, l) == XF_M23)   // R16: Maya-budget pool
        k_moe_m23w<2><<<dim3(2 * INTER / M23_RPB, TOPK * R), 256, 0, st>>>(xq, 0, rids, dls, pool_base(E, c, l), XB3, c.d_cb,
                                                                            c.d_gu);   // R16 v2 (k_moe_m23: v1)
    else if (c.ndyn)
        k_moe_cb3<8><<<dim3(2 * INTER / 32, TOPK * R), 256, 0, st>>>(xq, 0, rids, W.d_dslot, c.dpool, XB3, c.d_cb,
                                                                     c.d_gu);
    if (!E_gluq_off)
        k_moe_glu_q<<<cdiv((size_t)R * TOPK * INTER, 256), 256, 0, st>>>(c.d_gu, rw, rids, ls, R, c.d_hh,
                                                                          c.ndyn ? dls : nullptr);
    else {
    k_moe_glu<<<cdiv((size_t)R * TOPK * INTER, 256), 256, 0, st>>>(c.d_gu, rw, rids, ls, R, c.d_hh,
                                                                    c.ndyn ? dls : nullptr);
    q_fp8(st, c.d_hh, INTER, R * TOPK, INTER, c.d_hh, INTER, nullptr, 0);
    }
    if (nofp4) DCK(cudaMemsetAsync(c.d_yx, 0, (size_t)R * TOPK * D * sizeof(float), st));
    else if (E_moev2 && !E->xsplit) k_moe_gemv2<8, 9><<<dim3(D / 32, TOPK * R), 256, 0, st>>>(c.d_hh, 1, rids, ls, c.slots, XB, c.d_yx);
    else k_moe_gemv5<8><<<dim3(D / 64, TOPK * R), 256, 0, st>>>(c.d_hh, 1, rids, ls, c.slots, XB, c.d_yx);
    if (c.ndyn && E->gg)
        gg_moe(E, W.card, l, 1, c.d_hh, rids, dls, c.d_yx, R, st);
    else if (c.ndyn && pool_fmt(E, l) == XF_M23)
        k_moe_m23w<3><<<dim3(D / M23_RPB, TOPK * R), 256, 0, st>>>(c.d_hh, 1, rids, dls, pool_base(E, c, l), XB3, c.d_cb, c.d_yx);
    else if (c.ndyn)   // after the FP4 down (which zeroes the non-static rows)
        k_moe_cb3<2><<<dim3(D / 128, TOPK * R), 256, 0, st>>>(c.d_hh, 1, rids, W.d_dslot, c.dpool, XB3, c.d_cb, c.d_yx);
    if (shs) DCK(cudaStreamWaitEvent(st, c.ev_s1, 0));
    else shared_expert(st);
    if (E->xsplit || E->psplit) {   // R9: the other cards' choice rows into d_yx (their rows here are zero: not in this card's table)
        int sd[2] = {-1, -1}, ns = 0;
        for (int j = 0; j < (int)E->cards.size(); j++)
            if (j != W.card) sd[ns++] = j;
        k_ypull<<<R * TOPK, 256, 0, st>>>(c.rx5, c.rx5h, MAXR * TOPK, sd[0], sd[1], xtag + 1, c.d_yx);
    }
    if (R <= MAXR) PP(l, 9, st);
    if (early) DCK(cudaStreamWaitEvent(st, c.ev_e1, 0));   // c.x and the side stream's buffers are reused next layer
    if (ra && tl_part1) return;   // R19 [dsv4-r19-pipe]: the combine is enqueued after the other micro-batch's layer
    if (ra) {   // the combine waits on the GPU for the CPU lane's result: the host continues enqueueing
        if (defer_on(E, l, R))   // R21 [dsv4-r21-defer] KL emulation (defer_on implies the fused combine)
            k_moe_comb_post_def<<<cdiv((size_t)R * D, 256), 256, 0, st>>>(c.d_yx, E->d_hy[l], E->d_yflag + l * 16, kseq(E),
                                                                           c.d_shy, R, c.ao, c.h, c.fpost, c.fcomb, c.d_ydef);
        else if (!E_hcf_off)   // R10: + the FFN post (h update) in the same launch
            comb_post(st, c.d_yx, E->d_hy[l], E->d_yflag + l * 16, kseq(E), c.d_shy, R, c.ao, c.h, c.fpost, c.fcomb);
        else
        k_moe_comb_wait<<<cdiv((size_t)R * D, 256), 256, 0, st>>>(c.d_yx, E->d_hy[l], E->d_yflag + l * 16, kseq(E),
                                                                   c.d_shy, R, c.ao);
        return;
    }
    // host: wait for the routing, run the cold experts
    std::vector<int> pref;   // prefetched cold experts (statistics)
    NvR* nvw = new NvR("route wait");
    if (pred_on(E, R)) {
        auto tw = clk::now();
        hspin(E, E->pflag + l * 16, E->seq, "predicted ids");
        std::atomic_thread_fence(std::memory_order_acquire);
        if (E->tm) E->tm->pf_wait += ms_since(tw);
        CpuJob xs[TOPK];
        int nx = pred_pick(E, W, l, R, xs, pref);
        if (nx) E->lane.prefetch(nx, xs, E->rflag + l * 16, E->seq);
    }
    DCK(cudaEventSynchronize(c.ev_route));
    delete nvw;
    NvR* nvl = new NvR("lane");
    auto t1 = clk::now();
    if (E->tm) {
        float ms = 0.f;
        DCK(cudaEventElapsedTime(&ms, c.tp0, c.tp1));
        E->tm->gpu_attn += ms;
    }
    E->jobs.clear();
    for (int r = 0; r < R; r++)
        for (int k = 0; k < TOPK; k++) {
            int e = h_ids[r * TOPK + k];
            if (E->o.drop_w > 0 && h_w[r * TOPK + k] == 0.f) continue;   // R8 adaptive top-k: left out
            if (W.h_slot[e] >= 0) {
                if (E->tm && E->dyn_fp4 && W.inpool[e]) E->tm->ndyn++;
                continue;
            }
            if (W.h_dslot[e] >= 0) {
                if (E->tm) E->tm->ndyn++;
                continue;
            }
            bool again = false;   // host bytes: an expert shared by rows of a pass is read once (lane groups by expert)
            const CpuJob J0 = lane_job(E, W, e, r, h_w[r * TOPK + k]);
            for (const CpuJob& J : E->jobs) again |= J.base == J0.base;
            E->jobs.push_back(J0);
            if (E->tm && !again) { const double xb_ = expert_bytes((XFmt)W.hfmt[e]); E->tm->host_bytes += xb_; E->tm->lane_bytes += xb_; E->tm->nucold++; E->tm->nnat += J0.nat[0] != nullptr; }
        }
    if (!E->jobs.empty()) E->lane.run(R, h_xq, (int)E->jobs.size(), E->jobs.data(), c.h_y);
    else {
        E->lane.clear_pf();   // review r11 L5: no run, so drop this layer's prefetch bases here
        memset(c.h_y, 0, (size_t)R * D * 4);
    }
    auto t2 = clk::now();
    if (E->tm && R == 1) {
        for (int e : pref) {
            E->tm->npred++;
            for (int k = 0; k < TOPK; k++) E->tm->npred_hit += h_ids[k] == e;
        }
        for (int k = 0; k < TOPK; k++) {   // cold now (on the lane) and used by the previous token
            int e = h_ids[k];
            if (W.h_slot[e] >= 0 || W.h_dslot[e] >= 0) continue;
            for (int q = 0; q < TOPK; q++) E->tm->nprev_hit += E->prev_ids[l * TOPK + q] == e;
        }
        for (int k = 0; k < TOPK; k++) E->prev_ids[l * TOPK + k] = h_ids[k];
    }
    const float* ycold = c.d_hy_map;   // read over PCIe by the combine (review r4 M4: no H2D behind admissions)
    if (E_ycopy) {
        DCK(cudaMemcpyAsync(c.d_ycold, c.h_y, (size_t)R * D * 4, cudaMemcpyHostToDevice, st));
        ycold = c.d_ycold;
    }
    delete nvl;
    NvR* nvd = new NvR("dyn_route + comb");
    if (E->o.dyn) dyn_route(E, c, W, l, h_ids, R, h_w);
    k_moe_comb<<<cdiv((size_t)R * D, 256), 256, 0, st>>>(c.d_yx, ycold, c.d_shy, R, c.ao);
    delete nvd;
    if (E->tm) {
        E->tm->wait_route += std::chrono::duration<double, std::milli>(t1 - t0).count();
        E->tm->cpu += std::chrono::duration<double, std::milli>(t2 - t1).count();
        E->tm->ncold += (int)E->jobs.size();
        E->tm->nhot += R * TOPK - (int)E->jobs.size();
    }
}

// run-ahead host part of one decode layer: wait for its routing pack, run the cold experts, publish the result
static void moe_process(Engine* E, int l) {
    Layer& W = E->L[l];
    Card& c = E->cards[W.card];
    auto t0 = clk::now();
    g_wl = l;
    std::vector<int> pref, pref_e;
    const int R = E->ra_R;
    const bool erl = early_on(E, l, R);
    clk::time_point te0 = t0, tp0 = t0;   // [dsv4-dec1-pfmr] counters: early ids seen / own ids seen
    if (erl) {   // R11: the early prediction from layer l-1, until this layer's own prediction lands
        hspin(E, E->pflag2 + l * 16, E->seq, "early predicted ids");
        te0 = clk::now();
        std::atomic_thread_fence(std::memory_order_acquire);
        CpuJob xs[TOPK];
        int nx = pred_pick(E, W, l, R, xs, pref_e, E->hpred2, E->o.pf_early);
        if (nx) {
            E->lane.prefetch(nx, xs, E->pflag + l * 16, E->seq);
            if (E->tm) E->tm->pf_mb += E->lane.pf_lines * 64.0 * E->lane.threads() / 1e6;
        }
    }
    if (pred_on(E, R)) {   // predicted experts first (run-ahead + L3 prefetch)
        auto tw = clk::now();   // (R11: after the early prefetch, which ends when this flag lands)
        hspin(E, E->pflag + l * 16, E->seq, "predicted ids");
        std::atomic_thread_fence(std::memory_order_acquire);
        if (E->tm) E->tm->pf_wait += ms_since(tw);
        tp0 = clk::now();
        if (E->tm && erl) E->tm->win_e += std::chrono::duration<double, std::milli>(tp0 - te0).count();
        CpuJob xs[TOPK];
        int nx = pred_pick(E, W, l, R, xs, pref);
        if (nx) {
            E->lane.prefetch(nx, xs, E->rflag + l * 16, E->seq);
            if (E->tm) E->tm->pf_mb += E->lane.pf_lines * 64.0 * E->lane.threads() / 1e6;
        }
        early_issue(E, W, l, R);   // dsv4-fut [dsv4-fut-early]
    }
    {
        NvR nw("ra route wait");
        hspin(E, E->rflag + l * 16, E->seq, "routing pack");
    }
    if (W.early_e >= 0 && W.early_seq == E->seq) {   // the routing kernel's decision (written before the flag)
        std::atomic_thread_fence(std::memory_order_acquire);
        DCHECK(E->h_edec[l * 16] == E->seq, "early admission: no decision for layer %d", l);
        W.early_ap = E->h_edec[l * 16 + 1];
        if (cudaEventQuery(W.early_ev) == cudaSuccess) g_early_n[4]++;
        else (void)cudaGetLastError();
        if (W.early_ap) W.h_dslot[W.early_e] = W.spare_g;   // resident for this pass's lane split
    }
    NvR nvl("ra lane");
    std::atomic_thread_fence(std::memory_order_acquire);
    auto t1 = clk::now();
    const int* h_ids = (const int*)E->h_rp[l];
    const float* h_w = (const float*)(h_ids + MAXR * TOPK);
    const float* h_xq = h_w + MAXR * TOPK;
    E->jobs.clear();
    for (int rk = 0; rk < R * TOPK; rk++) {
        int e = h_ids[rk];
        if (E->o.drop_w > 0 && h_w[rk] == 0.f) continue;   // R8 adaptive top-k: left out
        if (W.h_slot[e] >= 0) {
            if (E->tm && E->dyn_fp4 && W.inpool[e]) E->tm->ndyn++;
            continue;
        }
        if (W.h_dslot[e] >= 0) {
            if (E->tm) E->tm->ndyn++;
            continue;
        }
        bool again = false;
        const CpuJob J0 = lane_job(E, W, e, rk / TOPK, h_w[rk]);
        for (const CpuJob& J : E->jobs) again |= J.base == J0.base;
        E->jobs.push_back(J0);
        if (E->tm && !again) { const double xb_ = expert_bytes((XFmt)W.hfmt[e]); E->tm->host_bytes += xb_; E->tm->lane_bytes += xb_; E->tm->nucold++; E->tm->nnat += J0.nat[0] != nullptr; }
    }
    if (!E->jobs.empty()) {
        E->lane.set_x16(g_xh16);   // R20 [dsv4-r20-xh16]: the run-ahead pack's xq is bf16
        E->lane.run(R, h_xq, (int)E->jobs.size(), E->jobs.data(), E->hy[l]);
        E->lane.set_x16(false);
    } else {
        E->lane.clear_pf();   // review r11 L5: no run, so drop this layer's prefetch bases here
        memset(E->hy[l], 0, (size_t)R * D * 4);
    }
    if (g_rec_x && R == 1) {
        RecRow& rr = g_rec_rows[(size_t)(g_rec_s % REC_T) * NL + l];
        memset(&rr, 0, sizeof rr);
        for (int k = 0; k < TOPK; k++) {
            rr.ids[k] = (short)h_ids[k];
            rr.pred[k] = pred_on(E, 1) ? (short)E->hpred[l * PRED_STRIDE + k] : (short)-1;
            rr.pref[k] = k < (int)pref.size() ? (short)pref[k] : (short)-1;
        }
        rr.npref = (short)pref.size();
        for (int e = 0; e < NE; e++)
            if (W.h_slot[e] >= 0 || W.h_dslot[e] >= 0) rr.res[e >> 3] |= (uint8_t)(1u << (e & 7));
        if (l == NL - 1) { g_rec_lo = std::min(g_rec_lo, g_rec_s); g_rec_hi = std::max(g_rec_hi, g_rec_s); }
    }
    if (g_htrace != -1 && (g_htrace == -2 || E->pos == g_htrace)) {
        g_lx[l] = hhash(h_xq, (size_t)R * D * (g_xh16 ? 2 : 4) / 4);
        g_ly[l] = hhash(E->hy[l], (size_t)R * D);
        for (int k = 0; k < TOPK; k++) g_lid[l][k] = h_ids[k];
    }
    std::atomic_thread_fence(std::memory_order_release);
    E->yflag[l * 16] = E->seq;
    auto t2 = clk::now();
    if (E->tm)
        for (int e : pref_e) {
            E->tm->npred_e++;
            for (int k = 0; k < R * TOPK; k++) E->tm->npred_e_hit += h_ids[k] == e;
        }
    if (E->tm)
        for (int e : pref) {   // used by any row of the pass (review r7 L12)
            E->tm->npred++;
            bool used = false;
            for (int k = 0; k < R * TOPK; k++) used |= h_ids[k] == e;
            E->tm->npred_hit += used;
        }
    if (E->tm && E_pfcov) {   // [dsv4-dec1-pfmr] coverage of the pass's distinct cold experts by the prediction unions
        const bool po = pred_on(E, R);
        if (po) E->tm->win_p += std::chrono::duration<double, std::milli>(t1 - tp0).count();
        for (int rk = 0; rk < R * TOPK; rk++) {
            const int e = h_ids[rk];
            if ((E->o.drop_w > 0 && h_w[rk] == 0.f) || W.h_slot[e] >= 0 || W.h_dslot[e] >= 0) continue;
            bool again = false, ie = false, ip = false;
            for (int q = 0; q < rk; q++) again |= h_ids[q] == e && !(E->o.drop_w > 0 && h_w[q] == 0.f);
            if (again) continue;
            for (int k = 0; k < R * TOPK; k++) {
                ie |= erl && E->hpred2[l * PRED_STRIDE + k] == e;
                ip |= po && E->hpred[l * PRED_STRIDE + k] == e;
            }
            E->tm->cv_cold++;
            E->tm->cv_e += ie;
            E->tm->cv_p += ip;
            E->tm->cv_ep += ie || ip;
        }
    }
    if (E->o.dyn) dyn_route(E, c, W, l, h_ids, R, h_w);
    if (E->tm) {
        E->tm->wait_route += std::chrono::duration<double, std::milli>(t1 - t0).count();
        E->tm->cpu += std::chrono::duration<double, std::milli>(t2 - t1).count();
        E->tm->ncold += (int)E->jobs.size();
        E->tm->nhot += R * TOPK - (int)E->jobs.size();
    }
}

// one layer for tokens s..s+n-1 on its card (streams c.h, pre c.pre)
// DSpark main hidden of layer l (37..39): the mean over the HC streams of the attention input h of the last min(n, WIN)
// rows (all n rows on the GEMV paths); positions recorded for draft_window
static void ds_capture(Engine* E, Card& c, int l, int s, int n) {
    int m = n <= MAXR ? n : std::min(n, WIN), row0 = n - m;
    k_hc_mean<<<cdiv((size_t)m * D, 256), 256, 0, c.st>>>(c.h, row0, m, c.d_mh + (size_t)(l - DS_T0) * D + (size_t)tl_row0 * 3 * D,
                                                          3 * D);   // (R19: a second micro-batch at its rows)
    if (!tl_cap) {   // R17: a step graph captured ahead (enqueue thread) leaves the host fields to the decode thread
        E->mh_pos0 = s + row0 - tl_row0;
        E->mh_n = m + tl_row0;
    }
}
static void layer_fwd(Engine* E, int l, int s, int n, int* selk) {
    tl_rec_s = s;
    Layer& W = E->L[l];   // (mutable: the dynamic tier's state)
    Card& c = E->cards[W.card];
    cudaStream_t st = c.st;
    char nvn[16];
    snprintf(nvn, sizeof nvn, "L%02d", l);
    NvR nv_l(nvn);
    const bool prof = n <= MAXR && E->tm;
    if (prof && !E->ra_on) DCK(cudaEventRecord(c.tp0, st));
    if (n <= MAXR) PP(l, 0, st);
    const bool pf = n > MAXR && pf_on();   // [dsv4-pf-pprof]
    if (pf) {
        pf_cpu();
        g_pf.lh[l].cpu = sched_getcpu();
        pf_rec(E, W.card, 0, st, l, PK_L0, l == c.l0 && W.card > 0 ? PK_HAND : PK_L0);
    }
    if (is_engram(l)) {
        auto t0 = clk::now();
        const double egw0 = g_eg_wait;
        engram(E, c, W, l, s, n);
        if (E->tm && !tl_cap) E->tm->engram += ms_since(t0);
        if (pf) { g_pf.lh[l].egw += g_eg_wait - egw0; pf_rec(E, W.card, 0, st, l, PK_ENGRAM); }
    }
    if (n <= MAXR) PP(l, 1, st);
    if (E->dspark && l >= DS_T0 && l < DS_T0 + 3 && !(l == NL - 1 && E->mh39_done))   // DSpark main hidden: the attention input of layers 37..39
        ds_capture(E, c, l, s, n);   // ([dsv4-pf17-cone39]: layer 39's was taken before its K/V-only pass)
    // attention sub-block
    const bool din = l > 0 && defer_on(E, l - 1, n);   // R21: the previous layer's lane rows are added here
    const bool hcf = n == 1 && !E_hcf_off && !din;   // R10: mixes + Sinkhorn + pre_rms in one launch (bit-identical)
    if (hcf)
        k_hc_fused<false><<<NMIX + 2, 1024, 0, st>>>(c.h, nullptr, nullptr, nullptr, W.hc_fn[0], W.hc_scale[0], W.hc_base[0],
                                                     c.mix, c.ss, c.apre, c.apost, c.acomb, c.pre, W.attn_norm, c.x,
                                                     (unsigned*)c.d_cnt + 4,
                                                     E_qkv1_off ? nullptr : c.xb + (size_t)MAXR * 32768);   // = xq1
    else if (hcfr_on(n) && !din)   // R20 [dsv4-r20-hcfr]: the same fused boundary for 2 .. MAXR rows (bit-identical per row)
    {
        if (!E_hcfr4_off)   // [dsv4-dec2-hcfr2]: rows side by side, fn loads batched (bit-identical)
            k_hc_fused_r4<false><<<hcfr4_grid(n), 256, 0, st>>>(c.h, nullptr, nullptr, nullptr, W.hc_fn[0], W.hc_scale[0],
                                                               W.hc_base[0], c.mix, c.ss, c.apre, c.apost, c.acomb, c.pre,
                                                               W.attn_norm, c.x, (unsigned*)c.d_cnt + 4, n,
                                                               E_qkv1_off ? nullptr : c.xb + (size_t)MAXR * 32768);
        else
        k_hc_fused_r<false><<<NMIX + 2, 1024, 0, st>>>(c.h, nullptr, nullptr, nullptr, W.hc_fn[0], W.hc_scale[0], W.hc_base[0],
                                                       c.mix, c.ss, c.apre, c.apost, c.acomb, c.pre, W.attn_norm, c.x,
                                                       (unsigned*)c.d_cnt + 4, n,
                                                       E_qkv1_off ? nullptr : c.xb + (size_t)MAXR * 32768);
    }
    else {
    hc_mixes(c, W, 0, c.h, n, c.apre, c.apost, c.acomb);
    k_hc_pre_rms<<<n, 1024, 0, st>>>(c.h, c.pre, W.attn_norm, n, c.x);
    }
    if (n <= MAXR) PP(l, 2, st);
    if (pf) pf_rec(E, W.card, 0, st, l, PK_HCPRE);
    const bool pfx = pfx_on(E, n) && pfx_slots(E, l, n) > 0;   // [dsv4-pf3-pfx]: predicted routing before the attention, copies during it
    // [dsv4-pf4-pfxd]: with the attention split the predictor runs after the X1 hand-off (the owner waits for the larger
    // shares of the other cards anyway); the predictor only places the prefetch: outputs unchanged
    g_pfx_defer = pfx && !E_pfxd_off && pfas_ok(E, W, n);
    if (pfx && !g_pfx_defer) pfx_predict(E, c, W, n);
    const bool pred = pred_on(E, n);
    const bool plate = pred && g_plate && !E->pipe && n <= MAXR;
    if (pred) {
        // predicted routing of this layer's FFN: its router on the pre-attention stream with the FFN's pre weights,
        // on a side stream beside the attention (joined before k_hc_post changes c.h)
        cudaStream_t s2 = c.st2;
        DCK(cudaEventRecord(c.ev_p0, st));
        DCK(cudaStreamWaitEvent(s2, c.ev_p0, 0));
        float* const pb = plate ? c.d_pbuf : c.f32tmp2;
        float* xp = pb;
        float* lg = pb + PRED_ROWS * D;
        int* pid = (int*)(pb + PRED_ROWS * (D + NE));
        float* pw = pb + PRED_ROWS * (D + NE) + PRED_STRIDE;
        k_hc_pre_rms<<<n, 1024, 0, s2>>>(c.h, c.apre, W.ffn_norm, n, xp);
        if (plate) DCK(cudaEventRecord(c.ev_p2, s2));
        if (g_rec_dx && n == 1) k_rec<<<D / 256, 256, 0, s2>>>(xp, g_rec_dx + (((size_t)(s % REC_T) * NL + l) * 2 + 1) * D);
        switch (n) {
            case 1: k_gemv_f32k<1, 4><<<cdiv(NE, 2), 256, 0, s2>>>(W.gate_w, NE, D, xp, D, lg, NE); break;
            case 2: k_gemv_f32k<2, 4><<<cdiv(NE, 2), 256, 0, s2>>>(W.gate_w, NE, D, xp, D, lg, NE); break;
            case 3: k_gemv_f32k<3, 4><<<cdiv(NE, 2), 256, 0, s2>>>(W.gate_w, NE, D, xp, D, lg, NE); break;
            case 4: k_gemv_f32k<4, 4><<<cdiv(NE, 2), 256, 0, s2>>>(W.gate_w, NE, D, xp, D, lg, NE); break;
            case 5: k_gemv_f32k<5, 4><<<cdiv(NE, 2), 256, 0, s2>>>(W.gate_w, NE, D, xp, D, lg, NE); break;
            default: k_gemv_f32k<6, 4><<<cdiv(NE, 2), 256, 0, s2>>>(W.gate_w, NE, D, xp, D, lg, NE); break;
        }
        if (!E_rp_off)   // R10: routing + mapped ids + flag in one launch
            k_route_pub<<<route_pub_grid(n, false), 384, 0, s2>>>(lg, W.gate_b, n, nullptr, pid, pw, nullptr,
                                                                   E->d_hpred + l * PRED_STRIDE, nullptr, nullptr, c.d_cnt + 1,
                                                                   E->d_pflag + l * 16, kseq(E), W.d_slot, E->o.car_margin,
                                                                   (c.ndyn || E->psplit) ? W.d_dslot : nullptr, E->o.drop_w);
        else {
        k_route<<<cdiv((size_t)n * 32, 256), 256, 0, s2>>>(lg, W.gate_b, n, pid, pw, W.d_slot, E->o.car_margin,
                                                            (c.ndyn || E->psplit) ? W.d_dslot : nullptr, E->o.drop_w);
        k_pub_ids<<<1, 1, 0, s2>>>(pid, n, E->d_hpred + l * PRED_STRIDE, E->d_pflag + l * 16, kseq(E));
        }
        DCK(cudaEventRecord(c.ev_p1, s2));
    }
    attention(E, c, W, l, s, n, selk);
    if (pfx) pfx_schedule(E, l, n);
    if (pred) DCK(cudaStreamWaitEvent(st, plate ? c.ev_p2 : c.ev_p1, 0));
    if (hcf)   // R10: post + grid barrier + mixes + Sinkhorn + pre_rms in one launch
        k_hc_fused<true><<<NMIX + 2, 1024, 0, st>>>(c.h, c.ao, c.apost, c.acomb, W.hc_fn[1], W.hc_scale[1], W.hc_base[1],
                                                    c.mix, c.ss, c.fpre, c.fpost, c.fcomb, c.apre, W.ffn_norm, c.x,
                                                    (unsigned*)c.d_cnt + 4, c.d_xqs);   // shared expert input
    else if (hcfr_on(n) && !din)
    {
        if (!E_hcfr4_off)
            k_hc_fused_r4<true><<<hcfr4_grid(n), 256, 0, st>>>(c.h, c.ao, c.apost, c.acomb, W.hc_fn[1], W.hc_scale[1],
                                                              W.hc_base[1], c.mix, c.ss, c.fpre, c.fpost, c.fcomb, c.apre,
                                                              W.ffn_norm, c.x, (unsigned*)c.d_cnt + 4, n);
        else
        k_hc_fused_r<true><<<NMIX + 2, 1024, 0, st>>>(c.h, c.ao, c.apost, c.acomb, W.hc_fn[1], W.hc_scale[1], W.hc_base[1],
                                                      c.mix, c.ss, c.fpre, c.fpost, c.fcomb, c.apre, W.ffn_norm, c.x,
                                                      (unsigned*)c.d_cnt + 4, n);
    }
    else {
    k_hc_post<<<cdiv((size_t)n * D, 256), 256, 0, st>>>(c.h, c.ao, c.apost, c.acomb, n);
    if (din) k_def_add<<<cdiv((size_t)n * D, 256), 256, 0, st>>>(c.h, c.d_ydef, c.acomb, n);   // R21 [dsv4-r21-defer]
    // FFN sub-block
    hc_mixes(c, W, 1, c.h, n, c.fpre, c.fpost, c.fcomb);
    k_hc_pre_rms<<<n, 1024, 0, st>>>(c.h, c.apre, W.ffn_norm, n, c.x);
    }
    if (n <= MAXR) PP(l, 7, st);
    if (pf) pf_rec(E, W.card, 0, st, l, PK_HCFFN);
    if (g_spin_us > 0 && n > 1 && n <= MAXR) k_spin<<<1, 1, 0, st>>>((long long)g_spin_us * 1000);   // R20 diagnostic
    auto tpl = clk::now();
    double eg0 = g_eg_wait, rs0 = g_route_sync;
    if (n <= MAXR) moe_decode(E, c, W, l, n);
    else moe_prefill(E, c, W, l, n);
    if (tl_part1) return;   // R19 [dsv4-r19-pipe]: layer_fwd2 later
    if (E_ptime && n > MAXR) {
        double enq = ms_since(tpl);
        DCK(cudaStreamSynchronize(st));
        logf_("ptime layer %d card %d: route sync %.1f ms (attention etc. before it), MoE enqueue %.1f ms, MoE GPU tail %.1f ms, "
              "engram wait so far %.1f ms\n", l, W.card, g_route_sync - rs0, enq - (g_route_sync - rs0), ms_since(tpl) - enq,
              g_eg_wait);
        (void)eg0;
    }
    if (plate) DCK(cudaStreamWaitEvent(st, c.ev_p1, 0));   // [dsv4-dec1-plate] side stream joined (captures, buffer reuse)
    if (!(E->ra_on && n <= MAXR && !E_hcf_off))   // (R10: fused into the run-ahead combine)
        k_hc_post<<<cdiv((size_t)n * D, 256), 256, 0, st>>>(c.h, c.ao, c.fpost, c.fcomb, n);
    if (n <= MAXR) PP(l, 10, st);
    if (pf) pf_rec(E, W.card, 0, st, l, PK_POST);
    if (tl_cap && g_htrace != -1) {
        if (!g_hh) {
            DCK(cudaHostAlloc((void**)&g_hh, NL * 8, cudaHostAllocPortable | cudaHostAllocMapped));
        }
        unsigned long long* dh = nullptr;
        DCK(cudaHostGetDevicePointer((void**)&dh, g_hh, 0));
        k_bhash<<<1, 1024, 0, st>>>((const uint32_t*)c.h, HC * D, dh + l);
    }
    std::swap(c.pre, c.fpre);   // next layer's attention uses this FFN's pre
    dump(E, c, l, c.h, n);
}

// R19 [dsv4-r19-pipe]: the second half of layer_fwd for a micro-batch of a pipelined pass (ra, n <= MAXR): the combine
// (waits on the GPU for the lane) + FFN post, then the stream swap, exactly as layer_fwd ends
static void layer_fwd2(Engine* E, int l, int n) {
    Layer& W = E->L[l];
    Card& c = E->cards[W.card];
    DCK(cudaSetDevice(c.dev));
    cudaStream_t st = c.st;
    if (!E_hcf_off)
        comb_post(st, c.d_yx, E->d_hy[l], E->d_yflag + l * 16, kseq(E), c.d_shy, n, c.ao, c.h, c.fpost, c.fcomb);
    else {
        k_moe_comb_wait<<<cdiv((size_t)n * D, 256), 256, 0, st>>>(c.d_yx, E->d_hy[l], E->d_yflag + l * 16, kseq(E), c.d_shy, n,
                                                                   c.ao);
        k_hc_post<<<cdiv((size_t)n * D, 256), 256, 0, st>>>(c.h, c.ao, c.fpost, c.fcomb, n);
    }
    std::swap(c.pre, c.fpre);
    dump(E, c, l, c.h, n);
}

// whole stack for tokens s..s+n-1 (toks already appended); embedding rows -> card 0 ... last card; returns on the
// last card with c.h / c.pre holding the final streams
// submit the Engram rows of tokens s..s+n-1 (both layers, deduplicated) to the NVMe reader
static const bool eg_rsort_off = getenv("DSV4_EGNORSORT") != nullptr;   // [dsv4-pf17-egrep] off-switch
static void eg_prefetch(Engine* E, int s, int n) {
    if (E->eg_pre) {   // R20 [dsv4-r20-egahead]: rows fetched after the last draft block for positions s .. s + 5
        E->eg_pre = false;
        bool ok = s == E->eg_pre_s && n <= E->eg_pre_n;
        for (int i = 0; i < n && ok; i++) ok = E->toks[s + i] == E->eg_pre_t[i];
        if (ok) {   // the batch's rows (and eg_map) cover this forward's rows 0 .. n - 1
            E->eg_batch_s = s;
            E->eg_batch_n = E->eg_pre_n;
            return;
        }
        E->egr.wait();   // another sequence of tokens: the stale batch finishes before its staging is reused
    }
    std::vector<std::pair<int, int64_t>> rows;
    std::vector<int64_t> keys((size_t)n * 48);
    int64_t r24[24];
    for (int i = 0; i < n; i++)
        for (int li = 0; li < 2; li++) {
            eg_hash(E, E->toks, s + i, li, r24);
            for (int j = 0; j < 24; j++) {
                DCHECK(r24[j] >= 0 && (size_t)r24[j] < E->eg[li].nrows, "engram row %lld", (long long)r24[j]);
                keys[((size_t)i * 2 + li) * 24 + j] = ((int64_t)li << 40) | r24[j];
            }
        }
    E->eg_map.resize(keys.size());
    if (!eg_rsort_off && keys.size() > 4096 && keys.size() < (1u << 20) &&
        std::all_of(keys.begin(), keys.end(), [](int64_t k) { return k >= 0 && k < (1LL << 42); })) {
        // [dsv4-pf17-egrep] prefill-sized batches: one LSD radix sort of (key << 20 | position) on the 42 key bits (3
        // passes of 14 bits), then one scan: the same unique rows in the same ascending order and the same eg_map as
        // sort + unique + lower_bound (10K: submit 66-69 -> 27 ms before the Engram reads start; unit test on 480 K keys
        // 59 -> 11 ms, rows and map identical)
        const size_t nk = keys.size();
        std::vector<uint64_t> a(nk), b(nk);
        for (size_t k = 0; k < nk; k++) a[k] = ((uint64_t)keys[k] << 20) | k;
        std::vector<uint32_t> h(1u << 14);
        for (int pass = 0; pass < 3; pass++) {
            const int sh = 20 + 14 * pass;
            std::fill(h.begin(), h.end(), 0u);
            for (uint64_t v : a) h[(v >> sh) & 16383]++;
            uint32_t acc = 0;
            for (auto& x : h) {
                const uint32_t c = x;
                x = acc;
                acc += c;
            }
            for (uint64_t v : a) b[h[(v >> sh) & 16383]++] = v;
            a.swap(b);
        }
        rows.reserve(nk);
        uint64_t prev = ~0ull;
        for (uint64_t v : a) {
            const uint64_t key = v >> 20;
            if (key != prev) {
                rows.push_back({(int)(key >> 40), (int64_t)(key & ((1ULL << 40) - 1))});
                prev = key;
            }
            E->eg_map[v & 0xfffff] = (int)rows.size() - 1;
        }
    } else {   // decode-sized batches (and DSV4_EGNORSORT=1): sort + unique + lower_bound
        std::vector<int64_t> u(keys);
        std::sort(u.begin(), u.end());
        u.erase(std::unique(u.begin(), u.end()), u.end());
        for (size_t k = 0; k < keys.size(); k++) E->eg_map[k] = (int)(std::lower_bound(u.begin(), u.end(), keys[k]) - u.begin());
        rows.reserve(u.size());
        for (int64_t k : u) rows.push_back({(int)(k >> 40), k & ((1LL << 40) - 1)});
    }
    E->egr.submit(rows, E->h_egraw);
    if (g_egprof && n <= MAXR) g_egp_sub = std::chrono::duration_cast<std::chrono::nanoseconds>(clk::now().time_since_epoch()).count(), g_egp_rows = (int)rows.size();
    E->eg_batch_s = s;
    E->eg_batch_n = n;
}

// R20 [dsv4-r20-egahead]: after a draft block, the next pass's tokens (t0 at the next position, then the drafts) are
// known; their Engram rows are fetched from NVMe now (the pass that follows waited ~1-1.5 ms for them at R 4-6). The
// next forward uses the batch when its tokens are a prefix of these (any gate length, a one-row step too); else it
// waits for the batch and fetches its own (exact either way: the same rows)
static const bool g_egahead_off = getenv("DSV4_EGAHEAD") == nullptr;   // ❌ as a default (route wait +1.4-2.1 ms at R 1-3): opt-in
static void eg_prefetch_ahead(Engine* E, int t0, const int* dr, int nd) {
    if (!E->eg_nvme || g_egahead_off || E->eg_pre) return;
    const int s = (int)E->toks.size();
    nd = std::min(nd, 7);
    if (s + 1 + nd > E->o.maxctx) return;
    E->toks.push_back(t0);
    for (int j = 0; j < nd; j++) E->toks.push_back(dr[j]);
    eg_prefetch(E, s, 1 + nd);
    E->toks.resize(s);
    E->eg_pre = true;
    E->eg_pre_s = s;
    E->eg_pre_n = 1 + nd;
    E->eg_pre_t[0] = t0;
    for (int j = 0; j < nd; j++) E->eg_pre_t[1 + j] = dr[j];
}
static void cap_wait(Engine* E);
// [dsv4-pf15-cone] causal cone at prefill: layers > CAND_SRC (20) read far positions only through layer 20's compressed
// KV / index keys and their own K/V inside WIN = 128 rows, so layer l needs its output only on the last
// 128 + 127 x (38 - l) rows and runs on the last min(n, 127 x (40 - l) + 1 + DSV4_CONE_M) rows (its first 127 rows
// are don't-care: their window reaches positions this layer did not write, zeroed first so they stay finite and do
// not depend on history). Layers <= 20 run every row (layer 20: compressor, indexer keys, candidates of all rows).
// The rows are moved to the front of h / pre / sel / cand before the layer (positions s + a ..). After the prefill the
// window K/V of every layer is valid on the last 128 + DSV4_CONE_M positions only: rollback deeper than that, or any
// continuation from inside the sequence, must re-prefill (cone_min guard). Default on; DSV4_NOCONE=1 (or DSV4_CONE=0) off
// (NLL gate: tflp 800 means 1.603 / 1.622 / 1.632 vs band 1.605-1.621; 10K 4.145 -> 2.883 s)
static const bool E_cone = getenv("DSV4_NOCONE") == nullptr && !(getenv("DSV4_CONE") && atoi(getenv("DSV4_CONE")) == 0);
static const int E_cone_m = std::clamp(getenv("DSV4_CONE_M") ? atoi(getenv("DSV4_CONE_M")) : 16, 0, 4096);
static int cone_rows(int l, int n) { return l <= CAND_SRC ? n : std::min(n, 127 * (NL - l) + 1 + E_cone_m); }
// [dsv4-ctx2-xcone] cone across chunks: rows a chunk must be followed by to skip layers 21-39 (layer 21's cone + a full
// zeroed window + the 64-row shift step); DSV4_NOXCONE=1 off
static const bool E_xcone_off = getenv("DSV4_NOXCONE") != nullptr;
static int xcone_need() { return cone_rows(CAND_SRC + 1, 1 << 30) + WIN + 64; }
// rows [d, m) of card c's per-row prefill state to [0, m - d) (stream order, blocks of d rows: no overlap)
static void cone_shift(Engine* E, Card& c, int d, int m) {
    const size_t ldc = E->o.maxctx / CAND_BS + 1;
    for (int r0 = d; r0 < m; r0 += d) {
        const int k = std::min(d, m - r0);
        DCK(cudaMemcpyAsync(c.h + (size_t)(r0 - d) * HC * D, c.h + (size_t)r0 * HC * D, (size_t)k * HC * D * 4, cudaMemcpyDeviceToDevice, c.st));
        DCK(cudaMemcpyAsync(c.pre + (size_t)(r0 - d) * HC, c.pre + (size_t)r0 * HC, (size_t)k * HC * 4, cudaMemcpyDeviceToDevice, c.st));
        DCK(cudaMemcpyAsync(c.sel + (size_t)(r0 - d) * ITOPK, c.sel + (size_t)r0 * ITOPK, (size_t)k * ITOPK * 4, cudaMemcpyDeviceToDevice, c.st));
        DCK(cudaMemcpyAsync(c.cand + (size_t)(r0 - d) * ldc, c.cand + (size_t)r0 * ldc, (size_t)k * ldc, cudaMemcpyDeviceToDevice, c.st));
    }
    for (auto& k : E->cards) k.sel_tag = -1;   // the helpers' copies of sel hold the old row layout
}
// [dsv4-pf16-conekv] the cone's rows whose output no later layer reads (the first rows of layers 20-38 held beyond the next
// layer's cone) need only this layer's window K/V (and layer 20's compressed rows / index keys, written per call): the
// attention input mixes, wkv + its tail and the compressor run on them alone; the layer then runs on the rest (its
// attention core, wo, FFN / MoE). Off DSV4_NOCONEKV=1 (= the round-15 cone)
static const bool E_conekv_off = getenv("DSV4_NOCONEKV") != nullptr;
static const bool E_cone39_off = getenv("DSV4_NOCONE39") != nullptr;   // [dsv4-pf17-cone39]
static void layer_kv(Engine* E, int l, int s, int n) {
    const Layer& W = E->L[l];
    Card& c = E->cards[W.card];
    cudaStream_t st = c.st;
    // (review pf16 L1: no hc_mixes here: the attention input reads c.pre, the previous FFN's mix; layer_fwd's mixes are
    // only read by its own attention / FFN boundary)
    k_hc_pre_rms<<<n, 1024, 0, st>>>(c.h, c.pre, W.attn_norm, n, c.x);
    const int kind = W.ratio ? 1 : 0;
    float* kv = c.ao;   // temporary [n][512], as attention()'s prefill path
    linq(c, W.wkv, c.x, D, n, kv, HD, 0, nullptr);
    rms(st, kv, HD, W.kv_norm, kv, HD, n, HD);
    rope(st, kv, HD, n, 1, HD, c.cosv[kind], c.sinv[kind], s, 1, 0);
    q_fp8(st, kv, HD, n, HD, nullptr, 0, kv_pf(W) + (size_t)s * HD, HD);
    if (W.ratio && is_kv_src(l)) compressor(E, c, W, l, s, n);
}
static void forward(Engine* E, int s, int n) {
    E->cap_cancel.store(true, std::memory_order_release);   // R19: a capture of a step that will not come stops early
    cap_wait(E);   // R10: no capture may run beside an eager forward (it swaps the streams)
    E->nfwd++;
    E->fwd_rows = n;   // [dsv4-pf15-cone] (the prefill path below may hold fewer rows)
    E->mh39_done = false;   // [dsv4-pf17-cone39] (review pf17 L1: also cleared here, not only after the cone loop)
    const bool dyn_was_early = E->dyn_early;
    E->dyn_early = false;   // eager forwards apply the tier themselves   // R10: invalidates a step graph captured ahead
    E->xseq++;
    const auto pf_te = clk::now();
    if (E->eg_nvme) eg_prefetch(E, s, n);
    if (pf_on() && n > MAXR) { g_pf.egsub += ms_since(pf_te); g_pf.egrows += (long)E->egr.njobs.load(); }
    // [dsv4-pf10-embra] a prefill chunk's embedding rows (host Q8_0 decode, ~58 ms per 10K rows on one thread) decoded by
    // up to 8 threads after the Engram submit (pf17: started before it, they slowed its hashing 17 -> 66 ms at 10K and so
    // the start of the reads); same rows, same bytes
    // review pf17b M4 / L7: slices 1..nt-1 on threads (a thread that fails to start: its slice on this thread, output
    // unchanged), slice 0 on this thread; nt = 1 (fewer than 512 rows) = the serial loop
    static const bool emb_mt_off = env_on("DSV4_NOEMBMT");
    const int nt = n > MAXR && !emb_mt_off ? std::min(8, std::max(1, n / 256)) : 1;
    auto emb_slice = [E, s, n, nt](int t) {
        for (int i = (int)((long)n * t / nt); i < (int)((long)n * (t + 1) / nt); i++)
            emb_row(E, E->toks[s + i], E->h_emb + (size_t)i * D);
    };
    std::vector<std::thread> embt;
    int ns = 1;   // slices handed to threads: 1..ns-1
    try {
        for (; ns < nt; ns++) embt.emplace_back(emb_slice, ns);
    } catch (const std::system_error&) {
    }
    if (pf_on() && n > MAXR && g_embra_chk) {   // [dsv4-pf10-embra] once: is the table in the page cache (mincore)
        g_embra_chk = false;
        const uintptr_t pg = 4096, a0 = (uintptr_t)E->embed_q & ~(pg - 1);
        if (E->embed_q) {
            const size_t len = (uintptr_t)E->embed_q + (size_t)V * (D / 32) * 34 - a0;
            std::vector<unsigned char> mv((len + pg - 1) / pg);
            if (mincore((void*)a0, len, mv.data()) == 0) {
                size_t r = 0;
                for (unsigned char v : mv) r += v & 1;
                logf_("embed table in the page cache at the first prefill: %.1f %% of %zu pages\n", 100.0 * r / mv.size(), mv.size());
            }
        }
    }
    Card& c0 = E->cards[0];
    DCK(cudaSetDevice(c0.dev));
    emb_slice(0);
    for (int t = ns; t < nt; t++) emb_slice(t);   // (threads that did not start)
    for (auto& t : embt) t.join();
    // R17: decode rows read in place from the mapped staging (an H2D copy queues behind the tier's admission copies on
    // the copy engine: up to ~3 ms per pass); prefill chunks keep the copy
    const bool embm = n <= MAXR && !E_embm_off;
    if (!embm) DCK(cudaMemcpyAsync(c0.d_embrow, E->h_emb, (size_t)n * D * 2, cudaMemcpyHostToDevice, c0.st));
    k_embed<<<cdiv((size_t)n * D, 256), 256, 0, c0.st>>>(embm ? E->h_emb : c0.d_embrow, n, c0.h, c0.pre);
    if (n > MAXR) pf_rec(E, 0, 0, c0.st, -1, PK_EMBED);
    DCK(cudaStreamSynchronize(c0.st));   // h_emb reuse
    if (pf_on() && n > MAXR) g_pf.embed += ms_since(pf_te);
    int selk = 0;
    E->ra_on = E->o.dump.empty() && ((n == 1 && E->runahead) || (n > 1 && n <= MAXR && E->ra_multi));
    E->ra_R = n;
    g_rec_s = s;
    if (E->ra_on) {   // enqueue two layers ahead of the host's processing
        E->seq++;
        auto enq = [&](int l) {
            Layer& W = E->L[l];
            Card& c = E->cards[W.card];
            DCK(cudaSetDevice(c.dev));
            if (l == c.l0 && W.card > 0) {
                Card& p = E->cards[W.card - 1];
                DCK(cudaEventRecord(p.ev, p.st));
                DCK(cudaStreamWaitEvent(c.st, p.ev, 0));
                hand_off(E, c, p, n);
                DCK(cudaEventRecord(c.ev, c.st));
                DCK(cudaStreamWaitEvent(p.st, c.ev, 0));
            }
            layer_fwd(E, l, s, n, &selk);
        };
        if (E->enq_thread) {   // R9: the whole forward enqueued by the enqueue thread, the lane here
            // review r10 M1: one-row steps apply the tier at the step boundary on this thread (all layers, before the
            // enqueue), and the next step's apply after the lane loop, exactly as the step graph does: the enqueue
            // thread no longer applies it per layer (it ran beside this thread's admissions: timing-dependent)
            const bool bnd = E->o.dyn && n == 1;   // re-review N1: always (DSV4_NOEAPP only drops the apply ahead)
            // review r13 M1 / M3: multi-row passes apply the tier at their boundary too (on this thread): applied per
            // layer by the enqueue thread it raced this thread's dyn_update on E->hfree and the cards' ev_dyn. Same
            // result: an admission issued in this pass matures two passes later, never inside it
            const bool bndm = E->o.dyn && n > 1;
            if (bnd || bndm) {
                static const int adm_age = getenv("DSV4_ADM1") ? 1 : 2;   // R21 [dsv4-r21-adm1]
                const auto tbd = clk::now();
                if (!dyn_was_early)
                    for (int l = 0; l < NL; l++) {
                        Layer& W = E->L[l];
                        DevGuard dg(E->cards[W.card].dev);
                        dyn_apply(E, E->cards[W.card], W, 0, bndm ? adm_age : 2);
                    }
                if (E->tm) E->tm->bnd += ms_since(tbd);
                E->dyn_bnd.store(true, std::memory_order_relaxed);
            }
            // R19 [dsv4-r19-pipe]: a pass of >= pipe_min rows as two micro-batches A (rows 0..a-1) and B (a..n-1), interleaved
            // per layer so that the CPU lane of one runs beside the GPU chain of the other: GPU order per card A1(0) B1(0)
            // A2(0) A1(1) B2(0) B1(1) ... (1 = up to the expert launches, 2 = the combine that waits for the lane), lane
            // order A(0) B(0) A(1) B(1) ... B's attention at layer l follows A's (its keys are written first); each row is
            // computed as in a one-row step (the pass's rows already are), so the results are those of the R-row pass
            const bool pipe = E->pipe && n >= E->pipe_min && n <= MAXR && !tl_cap;
            E->pipe_pass = pipe;   // review r19 L8: half A of a 2-row pass has R == 1 but is a verify row
            const int pa = (n + 1) / 2, pb = n - pa;
            int selk2[2] = {0, 0};
            if (pipe) {
                DCK(cudaSetDevice(c0.dev));
                DCK(cudaMemcpyAsync(c0.mbc.h, c0.h + (size_t)pa * HC * D, (size_t)pb * HC * D * 4, cudaMemcpyDeviceToDevice, c0.st));
                DCK(cudaMemcpyAsync(c0.mbc.pre, c0.pre + (size_t)pa * HC, (size_t)pb * HC * 4, cudaMemcpyDeviceToDevice, c0.st));
                E->mbd.xseq = E->xseq + 1;   // B's exchange tags (A keeps E->xseq)
            }
            auto enq_mb = [&](int mb, int l, bool p1) {
                if (mb) { mbd_swap(E); tl_mb = 1; tl_row0 = pa; }
                const int sb = mb ? s + pa : s, nb = mb ? pb : pa;
                Layer& W = E->L[l];
                Card& c = E->cards[W.card];
                DCK(cudaSetDevice(c.dev));
                if (p1) {
                    if (l == c.l0 && W.card > 0) {   // hand-off of this micro-batch's rows (as enq)
                        Card& p = E->cards[W.card - 1];
                        DCK(cudaEventRecord(p.ev, p.st));
                        DCK(cudaStreamWaitEvent(c.st, p.ev, 0));
                        hand_off(E, c, p, nb);
                        DCK(cudaEventRecord(c.ev, c.st));
                        DCK(cudaStreamWaitEvent(p.st, c.ev, 0));
                    }
                    tl_part1 = true;
                    layer_fwd(E, l, sb, nb, &selk2[mb]);
                    tl_part1 = false;
                } else {
                    layer_fwd2(E, l, nb);
                }
                if (mb) { mbd_swap(E); tl_mb = 0; tl_row0 = 0; }
            };
            if (pipe)
                E->enq_job = [&] {
                    enq_mb(0, 0, true);
                    enq_mb(1, 0, true);
                    for (int l = 0; l < NL; l++) {
                        enq_mb(0, l, false);
                        if (l + 1 < NL) enq_mb(0, l + 1, true);
                        enq_mb(1, l, false);
                        if (l + 1 < NL) enq_mb(1, l + 1, true);
                    }
                };
            else
            E->enq_job = [&] {
                for (int l = 0; l < NL; l++) enq(l);
            };
            E->enq_done.store(0, std::memory_order_relaxed);
            E->enq_go.fetch_add(1, std::memory_order_release);
            E->enq_go.notify_one();
            if (pipe) {
                for (int l = 0; l < NL; l++) {   // lane order A(l), B(l)
                    E->ra_R = pa;
                    moe_process(E, l);
                    mbh_swap(E);
                    E->ra_R = pb;
                    moe_process(E, l);
                    mbh_swap(E);
                }
                E->ra_R = n;
            } else
            for (int l = 0; l < NL; l++) moe_process(E, l);
            hspin(E, (const volatile int*)&E->enq_done, 1, "enqueue thread");
            std::atomic_thread_fence(std::memory_order_acquire);
            if (E->tm) E->tm->enq += E->enq_ms;   // R20: launch-bound check
            if (pipe) {   // B's final rows behind A's on the last card (the head and the drafter read rows 0..n-1)
                Card& cl = E->cards.back();
                DCK(cudaSetDevice(cl.dev));
                DCK(cudaMemcpyAsync(cl.h + (size_t)pa * HC * D, cl.mbc.h, (size_t)pb * HC * D * 4, cudaMemcpyDeviceToDevice, cl.st));
                DCK(cudaMemcpyAsync(cl.pre + (size_t)pa * HC, cl.mbc.pre, (size_t)pb * HC * 4, cudaMemcpyDeviceToDevice, cl.st));
                E->xseq = std::max(E->xseq, E->mbd.xseq);
            }
            if (g_htrace != -1 && n == 1 && (g_htrace == -2 || s == g_htrace))   // R11: same lines as the graph path (h hash 0)
                for (int l = 0; l < NL; l++)
                    logf_("htrace pos %d layer %2d h %016llx lane-in %016llx lane-out %016llx ids %d %d %d %d %d %d\n", s, l, 0ull,
                          g_lx[l], g_ly[l], g_lid[l][0], g_lid[l][1], g_lid[l][2], g_lid[l][3], g_lid[l][4], g_lid[l][5]);
            E->pipe_pass = false;
            if (bndm) E->dyn_bnd.store(false, std::memory_order_relaxed);
            if (bnd) E->dyn_bnd.store(false, std::memory_order_relaxed);
            if (bnd && !E_eapp_off) {
                for (int l = 0; l < NL; l++) {
                    Layer& W = E->L[l];
                    DevGuard dg(E->cards[W.card].dev);
                    dyn_apply(E, E->cards[W.card], W, 1);
                }
                E->dyn_early = true;
            }
            DCK(cudaSetDevice(E->cards.back().dev));
            return;
        }
        const int AHEAD = 2;
        for (int l = 0; l < std::min(AHEAD, NL); l++) enq(l);
        for (int l = 0; l < NL; l++) {
            moe_process(E, l);
            if (l + AHEAD < NL) enq(l + AHEAD);
        }
        DCK(cudaSetDevice(E->cards.back().dev));
        return;
    }
    const bool cone = E_cone && E->cone_ok && n > MAXR;
    int a = 0;   // [dsv4-pf15-cone] rows dropped so far (the card holds positions s + a .. s + n - 1)
    const bool lo = cone && E->cone_lo;   // [dsv4-ctx2-xcone]
    for (int k = 0; k < (int)E->cards.size(); k++) {
        Card& c = E->cards[k];
        if (lo && c.l0 > CAND_SRC) break;
        DCK(cudaSetDevice(c.dev));
        const int m = n - a;
        if (k > 0) {   // hand-off from the previous card
            Card& p = E->cards[k - 1];
            DCK(cudaEventRecord(p.ev, p.st));
            DCK(cudaStreamWaitEvent(c.st, p.ev, 0));
            DCK(cudaMemcpyPeerAsync(c.h, c.dev, p.h, p.dev, (size_t)m * HC * D * 4, c.st));
            DCK(cudaMemcpyPeerAsync(c.pre, c.dev, p.pre, p.dev, (size_t)m * HC * 4, c.st));
            DCK(cudaMemcpyPeerAsync(c.sel, c.dev, p.sel, p.dev, (size_t)m * ITOPK * 4, c.st));
            size_t ldc = E->o.maxctx / CAND_BS + 1;
            DCK(cudaMemcpyPeerAsync(c.cand, c.dev, p.cand, p.dev, (size_t)m * ldc, c.st));
            // the previous card's stream must not reuse its buffers before the copies finished
            DCK(cudaEventRecord(c.ev, c.st));
            DCK(cudaStreamWaitEvent(p.st, c.ev, 0));
        }
        for (int l = c.l0; l < c.l1; l++) {
            if (lo && l > CAND_SRC) break;
            if (n > MAXR) kv_in(E, l, s, n);   // [dsv4-rel2-p1]
            if (lo && l == CAND_SRC) {   // [dsv4-ctx2-xcone] later chunks read layer 20's window K/V + compressed rows only
                layer_kv(E, l, s, n);
                kv_out(E, l, s, n);
                continue;
            }
            if (cone) {
                // review pf15c M1: shift only by >= 64 rows (a small first drop joins the next layer's 127; extra front
                // rows are more don't-care rows), so a shift is at most n / 64 blocks
                const int na = n - cone_rows(l, n);
                if (na - a >= 64) {
                    cone_shift(E, c, na - a, n - a);
                    a = na;
                }
                // the window positions this layer reads but does not write (before its first row): zeros (review L1)
                // review ctx1 L1: after a chunk that skipped layers 21-39 ([dsv4-ctx2-xcone]) the window rows before s
                // hold stale ring rows too: zeroed as well
                const bool xprev = E->xcone_prev && l > CAND_SRC;
                const int wlo = std::max(xprev ? std::max(0, s - (WIN - 1)) : s, s + a - (WIN - 1));
                if ((a > 0 || xprev) && wlo < s + a) DCK(cudaMemsetAsync(kv_pf(E->L[l]) + (size_t)wlo * HD, 0, (size_t)(s + a - wlo) * HD * 2, c.st));
                if (!E_conekv_off && l >= CAND_SRC) {   // [dsv4-pf16-conekv] output rows needed: the next layer's cone (>= WIN;
                    // at layer 39 only the head row: see [dsv4-pf17-cone39] below)
                    const int m = n - a;
                    int keep = std::min(m, std::max(WIN, l + 1 < NL ? cone_rows(l + 1, n) : 0));
                    // [dsv4-pf17-cone39] the last layer: only the head row's output is read; the DSpark capture of its
                    // input (last WIN held rows) is taken first, then MAXR + 1 rows stay (the prefill kernels, not the
                    // GEMV paths); the others get their K/V only. DSV4_NOCONE39=1 off (keeps WIN rows)
                    if (l == NL - 1 && !E_cone39_off && tl_row0 == 0) {
                        keep = std::min(m, MAXR + 1);
                        if (E->dspark && m - keep >= 64) {
                            ds_capture(E, c, l, s + a, m);
                            E->mh39_done = true;
                        }
                    }
                    const int o = m - keep;
                    if (o >= 64) {
                        layer_kv(E, l, s + a, o);
                        cone_shift(E, c, o, m);
                        a += o;
                    }
                }
            }
            layer_fwd(E, l, s + a, n - a, &selk);
            if (n > MAXR) kv_out(E, l, s, n);   // [dsv4-rel2-p1]
        }
    }
    E->mh39_done = false;
    E->fwd_rows = n - a;
    if (a > 0) E->cone_min = std::max(E->cone_min, s + n - 1 - E_cone_m);
}

// ---------------------------------------------------------------- R10 whole-step decode graph
// One CUDA graph per decode step over the 3 cards (multi-device capture from card 0's stream): embedding H2D, the 40
// layers (attention split, expert split, LL12 exchanges, predictor side streams, lane handshakes on mapped flags),
// output head split and argmax D2H. Host-side work of a step runs at the boundary on the decode thread: the embedding
// row into pinned memory, Engram rows (NVMe, staged after the launch; the graph waits for a mapped flag), tier updates
// (dyn_apply), and the lane as before. The graph of step s + 1 is captured while step s runs (enqueue thread); its
// per-step parameters (position, tags, seq, compressor offsets, grids) come from the capture and go into the exec of
// its parity through cudaGraphExecUpdate (instantiate when the topology changed).
static void enq_layer(Engine* E, int l, int s, int n, int* selk) {
    Layer& W = E->L[l];
    Card& c = E->cards[W.card];
    DCK(cudaSetDevice(c.dev));
    if (l == c.l0 && W.card > 0 && tl_cap) {   // per-card graphs: the hand-off as LL12 words (no cross-card event)
        Card& p = E->cards[W.card - 1];
        const unsigned tag = (E->xseq * 64u + (unsigned)l) * 8u + 6u;
        const int ldc = E->o.maxctx / CAND_BS + 1;
        XSeg g1p = xseg({{p.h, HC * D, HC * D, 0, -1}, {p.pre, HC, HC, 0, -1}, {p.sel, ITOPK, ITOPK, 0, -1}});
        XSeg g2p = xseg({{p.cand, (ldc + 3) / 4, (ldc + 3) / 4, 0, -1}});
        {
            DevGuard dg(p.dev);
            xpush(p, g1p, 1, c.rxh, nullptr, tag);
            xpush(p, g2p, 1, c.rxc, nullptr, tag);
        }
        xpull2(c, xseg({{c.h, HC * D, HC * D, 0, -1}, {c.pre, HC, HC, 0, -1}, {c.sel, ITOPK, ITOPK, 0, -1}}), c.rxh,
               xseg({{c.cand, (ldc + 3) / 4, (ldc + 3) / 4, 0, -1}}), c.rxc, 1, tag);
        layer_fwd(E, l, s, n, selk);
        return;
    }
    if (l == c.l0 && W.card > 0) {
        Card& p = E->cards[W.card - 1];
        DCK(cudaEventRecord(p.ev, p.st));
        DCK(cudaStreamWaitEvent(c.st, p.ev, 0));
        hand_off(E, c, p, n);
        DCK(cudaEventRecord(c.ev, c.st));
        DCK(cudaStreamWaitEvent(p.st, c.ev, 0));
    }
    layer_fwd(E, l, s, n, selk);
}
static void head_gemv(Engine* E, const float* x, int m, bool draft = false);
// capture the step at position s (token row from E->h_emb) into gs[s & 1]; host state of the step (compressor offsets)
// is computed and then restored: decode commits it at the launch
static void cap_step(Engine* E, int s, bool async) {
    Engine::GSlot& G = E->gs[s & 1];
    const int NC = (int)E->cards.size();
    G.ready = false;
    for (int k = 0; k < NC; k++)
        if (G.pend[k]) {
            DCK(cudaGraphDestroy(G.pend[k]));
            G.pend[k] = nullptr;
        }
    for (int i = 0; i < 4; i++) { G.tc0[i] = E->tc[i]; G.tn0[i] = E->tail_n[i]; }
    G.s = s;
    G.nfwd = E->nfwd;
    E->cseq = E->seq + 1;
    G.seq = E->cseq;
    tl_cap = true;
    for (auto& c : E->cards) { std::swap(c.st, c.cst); std::swap(c.st2, c.cst2); }
    // one capture per card (no cross-card events in the decode path: hand-off, head and exchanges are device words)
    for (auto& c : E->cards) {
        DevGuard dg(c.dev);
        DCK(cudaStreamBeginCapture(c.st, cudaStreamCaptureModeRelaxed));
    }
    Card& c0 = E->cards[0];
    DCK(cudaSetDevice(c0.dev));
    E->xseq++;
    if (E_embm_off) DCK(cudaMemcpyAsync(c0.d_embrow, E->h_emb, (size_t)D * 2, cudaMemcpyHostToDevice, c0.st));
    k_embed<<<cdiv((size_t)D, 256), 256, 0, c0.st>>>(E_embm_off ? c0.d_embrow : E->h_emb, 1, c0.h, c0.pre);   // R17
    E->ra_on = true;
    E->ra_R = 1;
    int selk = 0;
    bool cancelled = false;
    for (int l = 0; l < NL; l++) {
        if (async && E->cap_cancel.load(std::memory_order_acquire)) { cancelled = true; break; }
        enq_layer(E, l, s, 1, &selk);
    }
    if (!cancelled) {   // head split: x to the other cards as LL12 words, their rows written into the last card's logits, then a flag
        Card& c = E->cards.back();
        const int last = NC - 1;
        const unsigned tag = (E->xseq * 64u + 63u) * 8u + 1u;
        DCK(cudaSetDevice(c.dev));
        float* xx = c.f32tmp2;
        k_hc_pre<<<cdiv((size_t)D, 256), 256, 0, c.st>>>(c.h, c.pre, 1, xx);
        rms(c.st, xx, D, E->fnorm, xx, D, 1, D);
        if (E->hsplit) {
            // (R10-9: plain remote logits stores + a flag were not ordered over PCIe: a stale slice now and then; each peer
            // reduces its slice to (max, index) and sends one LL12 word instead)
            DCHECK(NC == 3, "graph head split needs 3 cards");
            xpush(c, xseg({{xx, D, D, 0, -1}}), 1, E->cards[0].rxx, E->cards[1].rxx, tag);
            for (int k = 0; k < last; k++) {
                Card& pk = E->cards[k];
                DevGuard dg(pk.dev);
                xpull(pk, xseg({{pk.d_hx, D, D, 0, -1}}), 1, pk.rxx, tag);
                const int N = E->hr[k + 1] - E->hr[k];
                if (E->head_qk[k].sq) linq(pk, E->head_qk[k], pk.d_hx, D, 1, pk.d_logits + E->hr[k], V);   // [dsv4-rel-q8]
                else k_gemv_bf16<1><<<cdiv(N, 8), 256, 0, pk.st>>>(E->head_k[k], N, D, pk.d_hx, D, pk.d_logits + E->hr[k], V);
                k_argmax_put<<<1, 1024, 0, pk.st>>>(pk.d_logits, E->hr[k], E->hr[k + 1], c.rxx + 1 + k, tag + 1);
            }
            const int N = V - E->hr[last];
            if (E->head_qk[last].sq) linq(c, E->head_qk[last], xx, D, 1, c.d_logits + E->hr[last], V);
            else k_gemv_bf16<1><<<cdiv(N, 8), 256, 0, c.st>>>(E->head_k[last], N, D, xx, D, c.d_logits + E->hr[last], V);
            k_argmax_comb<<<1, 1024, 0, c.st>>>(c.d_logits, E->hr[last], V, c.rxx + 1, c.rxx + 2, tag + 1, c.d_tok);
        } else {
            head_gemv(E, xx, 1);
            k_argmax<<<1, 1024, 0, c.st>>>(c.d_logits, V, c.d_tok);
        }
        DCK(cudaMemcpyAsync(E->h_tok, c.d_tok, 4, cudaMemcpyDeviceToHost, c.st));
    }
    cudaGraph_t g[3] = {};
    for (int k = 0; k < NC; k++) {
        Card& c = E->cards[k];
        DevGuard dg(c.dev);
        DCK(cudaStreamEndCapture(c.st, &g[k]));
    }
    for (auto& c : E->cards) { std::swap(c.st, c.cst); std::swap(c.st2, c.cst2); }
    tl_cap = false;
    for (int i = 0; i < 4; i++) {
        G.tc1[i] = E->tc[i];
        G.tn1[i] = E->tail_n[i];
        E->tc[i] = G.tc0[i];
        E->tail_n[i] = G.tn0[i];
    }
    if (cancelled) {   // R19: a partial capture: nothing launched, the slot stays not ready
        for (int k = 0; k < NC; k++)
            if (g[k]) DCK(cudaGraphDestroy(g[k]));
        E->g_cancel++;
        return;
    }
    for (int k = 0; k < NC; k++) {
        DevGuard dg(E->cards[k].dev);
        bool ok = false;
        if (G.ex[k]) {
            cudaGraphExecUpdateResultInfo info;
            const cudaError_t ue = cudaGraphExecUpdate(G.ex[k], g[k], &info);
            ok = ue == cudaSuccess;
            // review r10 L7: only a refused update (topology / node change) falls back to instantiation
            DCHECK(ok || ue == cudaErrorGraphExecUpdateFailure, "graph update card %d: %s", k, cudaGetErrorString(ue));
            if (!ok) cudaGetLastError();
        }
        if (ok) {
            E->g_upd++;
            DCK(cudaGraphDestroy(g[k]));
            if (async && !E_gup_off) DCK(cudaGraphUpload(G.ex[k], E->cards[k].est));   // behind the running step: the launch
                                                                                     // then only submits
        } else if (async) {
            G.pend[k] = g[k];   // topology changed (e.g. the attention split count): instantiate at the boundary
        } else {
            if (G.ex[k]) DCK(cudaGraphExecDestroy(G.ex[k]));
            DCK(cudaGraphInstantiate(&G.ex[k], g[k], 0));
            E->g_inst++;
            DCK(cudaGraphDestroy(g[k]));
        }
    }
    G.ready = true;
}
static void cap_inst(Engine* E, Engine::GSlot& G) {   // decode thread, no step graph running
    for (size_t k = 0; k < E->cards.size(); k++) {
        if (!G.pend[k]) continue;
        DevGuard dg(E->cards[k].dev);
        if (G.ex[k]) DCK(cudaGraphExecDestroy(G.ex[k]));
        DCK(cudaGraphInstantiate(&G.ex[k], G.pend[k], 0));
        DCK(cudaGraphDestroy(G.pend[k]));
        G.pend[k] = nullptr;
        E->g_inst++;
    }
}
static void cap_wait(Engine* E) {   // the capture job given to the enqueue thread is done
    if (!E->cap_pending) return;
    // (forward() sets cap_cancel first: a speculative capture then stops at its next layer)
    hspin(E, (const volatile int*)&E->enq_done, 1, "step graph capture");
    std::atomic_thread_fence(std::memory_order_acquire);
    E->cap_pending = false;
    E->cap_spec = false;   // review r19 L1: a waited-for capture is no longer the speculative one
}
// Engram rows of the step's token (both layers) into the pinned staging, then the flags the graph waits on
static void eg_stage(Engine* E, int seq) {
    E->egr.wait();
    for (int li = 0; li < 2; li++)
        for (int j = 0; j < 24; j++)
            memcpy(E->h_egrows + ((size_t)li * 24 + j) * 264, E->h_egraw + (size_t)E->eg_map[(size_t)li * 24 + j] * 264, 264);
    std::atomic_thread_fence(std::memory_order_release);
    for (int li = 0; li < 2; li++) E->egflag[li * 16] = seq;
}
// one decode step through the step graph; returns the greedy token
static double g_gb[5] = {};   // R10 boundary timers (ms summed): entry -> launch, launch, Engram staging, lane loop, head wait
static long g_gbn = 0;
static int decode_graph(Engine* E, int t) {
    const int s = E->pos;
    auto tb0 = clk::now();
    static const bool dbg = getenv("DSV4_GDBG") != nullptr;
    if (dbg) logf_("graph step pos %d seq %d t %.3f\n", s, E->seq, std::chrono::duration<double>(clk::now().time_since_epoch()).count());
    emb_row(E, t, E->h_emb);   // read in place by the graph's k_embed (R17); the previous
    // step's head wait ordered the previous step's k_embed before this write
    eg_prefetch(E, s, 1);
    cap_wait(E);
    Engine::GSlot& G = E->gs[s & 1];
    bool same = G.ready && G.s == s && G.nfwd == E->nfwd && G.seq == E->seq + 1;
    for (int i = 0; i < 4 && same; i++) same = G.tc0[i] == E->tc[i] && G.tn0[i] == E->tail_n[i];
    if (!same) cap_step(E, s, false);   // first step after a prefill, or the state changed: capture here
    else cap_inst(E, G);
    if (E->o.dyn && !E->dyn_early)
        for (int l = 0; l < NL; l++) {
            Layer& W = E->L[l];
            Card& c = E->cards[W.card];
            DevGuard dg(c.dev);
            dyn_apply(E, c, W);
        }
    E->dyn_early = false;
    for (int i = 0; i < 4; i++) { E->tc[i] = G.tc1[i]; E->tail_n[i] = G.tn1[i]; }
    E->seq = G.seq;
    G.ready = false;
    g_rec_s = s;
    auto tb1 = clk::now();
    for (int k = 0; k < (int)E->cards.size(); k++) {   // each card's graph on its own eager stream (tier updates
        DevGuard dg(E->cards[k].dev);                        // of that card stay in stream order around it)
        // R19 (review r18 N1): cap_wait above restored st from the capture swap, so st == est here and the eager work
        // queued since the last graph (drafter window, rollback copies, tier work) precedes the launch in stream order
        DCHECK(E->cards[k].st == E->cards[k].est, "card %d: st != est at the graph launch", k);
        DCK(cudaGraphLaunch(G.ex[k], E->cards[k].est));
    }
    auto tb2 = clk::now();
    // the next step's graph is captured while this one runs
    if (s + 1 < E->o.maxctx) {
        E->enq_job = [E, s] { cap_step(E, s + 1, true); };
        E->cap_cancel.store(false, std::memory_order_release);   // (R19: only an eager forward cancels)
        E->enq_done.store(0, std::memory_order_relaxed);
        E->cap_pending = true;
        E->enq_go.fetch_add(1, std::memory_order_release);
        E->enq_go.notify_one();
    }
    auto tb3 = clk::now();
    for (int l = 0; l < NL; l++) {
        moe_process(E, l);
        if (l == 0) eg_stage(E, E->seq);   // (first Engram layer is 1: the NVMe wait hides behind layer 0)
    }
    if (E->o.dyn && !E_eapp_off) {   // the next step's tier apply while the head runs (its table writes queue behind
        for (int l = 0; l < NL; l++) {   // this step's graph on every card, before the next launch)
            Layer& W = E->L[l];
            Card& c = E->cards[W.card];
            DevGuard dg(c.dev);
            dyn_apply(E, c, W, 1);
        }
        E->dyn_early = true;
    }
    auto tb4 = clk::now();
    {
        NvR nh("head wait");
        DCK(cudaStreamSynchronize(E->cards.back().est));
    }
    auto tb5 = clk::now();
    auto dd = [](clk::time_point a, clk::time_point b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
    g_gb[0] += dd(tb0, tb1); g_gb[1] += dd(tb1, tb2); g_gb[2] += dd(tb2, tb3); g_gb[3] += dd(tb3, tb4); g_gb[4] += dd(tb4, tb5);
    g_gbn++;
    DCK(cudaSetDevice(E->cards.back().dev));
    if (g_htrace != -1 && (g_htrace == -2 || s == g_htrace) && g_hh)
        for (int l = 0; l < NL; l++)
            logf_("htrace pos %d layer %2d h %016llx lane-in %016llx lane-out %016llx ids %d %d %d %d %d %d\n", s, l, g_hh[l],
                  g_lx[l], g_ly[l], g_lid[l][0], g_lid[l][1], g_lid[l][2], g_lid[l][3], g_lid[l][4], g_lid[l][5]);
    return *(volatile int*)E->h_tok;
}

// logits of rows [0, n) of the last card's final streams (rows idx) -> argmax / nll
// logits [m][V] of the normed rows x [m][5120] on the last card into its d_logits; R9 head split: the other cards get x
// (peer copy), run their rows and write them straight into the last card's buffer
// R22 [dsv4-r22-hmma]: the drafter's head rows (draft = true, m = 5) on the tensor cores; DSV4_NOHMMA=1 off
static const bool g_hmma = getenv("DSV4_NOHMMA") == nullptr;
static void head_gemv(Engine* E, const float* x, int m, bool draft) {
    Card& c = E->cards.back();
    const bool tc = draft && g_hmma && m == 5;
    HProf* hq = nullptr;   // R23 DSV4_HPROF
    if (E_hprof && !tl_cap && E->hsplit) {
        hq = &g_hp[g_hpi];
        g_hpi ^= 1;
        hp_read(*hq);   // the call before the previous one: long done
        hq->kind = draft ? 1 : 0;
        hq->m = m;
        hq->nc = (int)E->cards.size();
        for (int k = 0; k < hq->nc; k++)
            if (!hq->e[k][0]) {
                DevGuard dg(E->cards[k].dev);
                DCK(cudaEventCreate(&hq->e[k][0]));
                DCK(cudaEventCreate(&hq->e[k][1]));
            }
    }
    if (E->head_q.sq || E->head_qk[0].sq) {   // [dsv4-rel-q8]: Q8_0 head, dp4a (x int8 per 32)
        if (!E->hsplit) {
            linq(c, E->head_q, x, D, m, c.d_logits, V);
            return;
        }
        const int NCq = (int)E->cards.size(), lastq = NCq - 1;
        static const bool nohosthead = getenv("DSV4_NOHOSTHEAD") != nullptr;   // repro switch (race hunt)
        if (E->hostx && !nohosthead) {   // [dsv4-rel-hostx]: no staged peer copies (each one blocks): x to mapped host memory, read in
            // place by the other cards; their slices D2H, then H2D into the last card's logits (all async)
            if (!E->h_hx) {
                DCK(cudaHostAlloc((void**)&E->h_hx, (size_t)MAXR * D * 4, cudaHostAllocPortable | cudaHostAllocMapped));
                for (int k = 0; k < lastq; k++)
                    DCK(cudaHostAlloc((void**)&E->h_hl[k], (size_t)MAXR * (E->hr[k + 1] - E->hr[k]) * 4, cudaHostAllocPortable));
            }
            DCK(cudaMemcpyAsync(E->h_hx, x, (size_t)m * D * 4, cudaMemcpyDeviceToHost, c.st));
            DCK(cudaEventRecord(E->ev_hx, c.st));
            for (int k = 0; k < lastq; k++) {
                Card& p = E->cards[k];
                DevGuard dg(p.dev);
                DCK(cudaStreamWaitEvent(p.st, E->ev_hx, 0));
                const int N = E->hr[k + 1] - E->hr[k];
                linq(p, E->head_qk[k], E->h_hx, D, m, p.d_logits, N);
                DCK(cudaMemcpyAsync(E->h_hl[k], p.d_logits, (size_t)m * N * 4, cudaMemcpyDeviceToHost, p.st));
                DCK(cudaEventRecord(E->ev_hd[k], p.st));
            }
            linq(c, E->head_qk[lastq], x, D, m, c.d_logits + E->hr[lastq], V);
            for (int k = 0; k < lastq; k++) {
                const int N = E->hr[k + 1] - E->hr[k];
                DCK(cudaStreamWaitEvent(c.st, E->ev_hd[k], 0));
                DCK(cudaMemcpy2DAsync(c.d_logits + E->hr[k], (size_t)V * 4, E->h_hl[k], (size_t)N * 4, (size_t)N * 4, m,
                                      cudaMemcpyHostToDevice, c.st));
            }
            return;
        }
        for (int k = 0; k < lastq; k++)
            DCK(cudaMemcpyAsync(E->cards[k].d_hx, x, (size_t)m * D * 4, cudaMemcpyDefault, c.st));
        DCK(cudaEventRecord(E->ev_hx, c.st));
        for (int k = 0; k < lastq; k++) {   // slices computed on their card, then whole rows copied (no 4 B remote stores)
            Card& p = E->cards[k];
            DevGuard dg(p.dev);
            DCK(cudaStreamWaitEvent(p.st, E->ev_hx, 0));
            const int N = E->hr[k + 1] - E->hr[k];
            if (hq) DCK(cudaEventRecord(hq->e[k][0], p.st));
            linq(p, E->head_qk[k], p.d_hx, D, m, p.d_logits, N);
            DCK(cudaMemcpy2DAsync(c.d_logits + E->hr[k], (size_t)V * 4, p.d_logits, (size_t)N * 4, (size_t)N * 4, m,
                                  cudaMemcpyDefault, p.st));
            if (hq) DCK(cudaEventRecord(hq->e[k][1], p.st));
            DCK(cudaEventRecord(E->ev_hd[k], p.st));
        }
        if (hq) DCK(cudaEventRecord(hq->e[lastq][0], c.st));
        linq(c, E->head_qk[lastq], x, D, m, c.d_logits + E->hr[lastq], V);
        if (hq) DCK(cudaEventRecord(hq->e[lastq][1], c.st));
        for (int k = 0; k < lastq; k++) DCK(cudaStreamWaitEvent(c.st, E->ev_hd[k], 0));
        return;
    }
    if (!E->hsplit) {
        if (tc) head_mma_launch<5>(E->head, V, x, c.d_logits, V, c.st);
        else
        GEMV_DISPATCH(k_gemv_bf16, m, cdiv(V, 8), c.st, E->head, V, D, x, D, c.d_logits, V);
        return;
    }
    const int NC = (int)E->cards.size(), last = NC - 1;
    for (int k = 0; k < last; k++)
        DCK(cudaMemcpyAsync(E->cards[k].d_hx, x, (size_t)m * D * 4, cudaMemcpyDefault, c.st));
    DCK(cudaEventRecord(E->ev_hx, c.st));
    for (int k = 0; k < last; k++) {
        Card& p = E->cards[k];
        DevGuard dg(p.dev);
        DCK(cudaStreamWaitEvent(p.st, E->ev_hx, 0));
        const int N = E->hr[k + 1] - E->hr[k];
        // R23 (review r22 M1): drafter heads only, never inside a capture (target heads / the look-ahead graph)
        const bool hp = draft && !tl_cap && g_dn >= 0 && k < 2;
        if (hp) {
            if (!g_dhe[k][0]) { DCK(cudaEventCreate(&g_dhe[k][0])); DCK(cudaEventCreate(&g_dhe[k][1])); }
            DCK(cudaEventRecord(g_dhe[k][0], p.st));
            g_dhon = true;
        }
        if (hq) DCK(cudaEventRecord(hq->e[k][0], p.st));
        if (tc) head_mma_launch<5>(E->head_k[k], N, p.d_hx, c.d_logits + E->hr[k], V, p.st);
        else head_rows_gemv(m, E->head_k[k], N, p.d_hx, c.d_logits + E->hr[k], p.st);
        if (hq) DCK(cudaEventRecord(hq->e[k][1], p.st));
        if (hp) DCK(cudaEventRecord(g_dhe[k][1], p.st));
        DCK(cudaEventRecord(E->ev_hd[k], p.st));
    }
    const int N = V - E->hr[last];
    if (hq) DCK(cudaEventRecord(hq->e[last][0], c.st));
    if (tc) head_mma_launch<5>(E->head_k[last], N, x, c.d_logits + E->hr[last], V, c.st);
    else head_rows_gemv(m, E->head_k[last], N, x, c.d_logits + E->hr[last], c.st);
    if (hq) DCK(cudaEventRecord(hq->e[last][1], c.st));
    for (int k = 0; k < last; k++) DCK(cudaStreamWaitEvent(c.st, E->ev_hd[k], 0));
}
static void head_rows(Engine* E, int row0, int n, int* argmax_out, const int* tgt, float* nll) {
    Card& c = E->cards.back();
    DCK(cudaSetDevice(c.dev));
    for (int i0 = 0; i0 < n; i0 += MAXR) {
        int m = std::min(MAXR, n - i0);
        float* xx = c.f32tmp2;
        k_hc_pre<<<cdiv((size_t)m * D, 256), 256, 0, c.st>>>(c.h + (size_t)(row0 + i0) * HC * D,
                                                              c.pre + (size_t)(row0 + i0) * HC, m, xx);
        rms(c.st, xx, D, E->fnorm, xx, D, m, D);
        head_gemv(E, xx, m);
        if (argmax_out && E->samp_on && !nll) {   // [dsv4-rel-sample]: one sampled token per row
            samp_rows(E->samp, c.d_logits, V, m, c.d_tok, c.st);
            DCK(cudaMemcpyAsync(argmax_out + i0, c.d_tok, m * 4, cudaMemcpyDeviceToHost, c.st));
        } else if (argmax_out) {
            k_argmax<<<m, 1024, 0, c.st>>>(c.d_logits, V, c.d_tok);
            DCK(cudaMemcpyAsync(argmax_out + i0, c.d_tok, m * 4, cudaMemcpyDeviceToHost, c.st));
        }
        if (nll) {
            DCK(cudaMemcpyAsync(c.d_tok + MAXR, tgt + i0, m * 4, cudaMemcpyHostToDevice, c.st));
            k_nll<<<m, 1024, 0, c.st>>>(c.d_logits, V, c.d_tok + MAXR, c.d_nll);
            DCK(cudaMemcpyAsync(nll + i0, c.d_nll, m * 4, cudaMemcpyDeviceToHost, c.st));
        }
        DCK(cudaStreamSynchronize(c.st));
    }
}

// R9 workspace borrowing: on = evict the experts of every card's tail slots and carve the chunk-sized prefill buffers
// from them; off = MAXR-row buffers back, the evicted experts copied back from their host copies (FP4)
static void ck_copy(Engine* E, int l, int e, uint8_t* dst, cudaStream_t s) {
    std::string px = "layers." + std::to_string(E->L[l].lid) + ".ffn.experts." + std::to_string(e) + ".";
    size_t off = 0;
    for (const char* wn : {"w1", "w3", "w2"})
        for (const char* kn : {".weight", ".scale"}) {
            const TensorInfo& t = E->stf.get(px + wn + kn);
            DCK(cudaMemcpyAsync(dst + off, t.data, t.nbytes, cudaMemcpyHostToDevice, s));
            off += t.nbytes;
        }
    DCHECK(off == expert_bytes(XF_FP4), "expert bytes %zu", off);
}
// R9 workspace borrowing: on = evict the experts of every card's last slots and carve the prefill buffers for C rows
// from them; off = MAXR-row buffers back, the evicted experts copied back (FP4 host copy, else the checkpoint's FP4
// bytes: review r9 M1). Review r9 M2: the carve is sized by the prompt (C = min(chunk, prompt)), so a short prompt
// evicts / restores a few slots instead of the whole tail. Returns false (nothing changed) if a slot of the region
// holds an expert that cannot be evicted (review r9 H1: walks h_slot, not the pool lists)
static bool gborrow_ws(Engine* E, bool on, int C);
static bool borrow_ws(Engine* E, bool on, int C = 0) {
    if (E->gborrow) return gborrow_ws(E, on, C);
    if (!E->borrow) return false;
    const size_t XB = expert_bytes(XF_FP4);
    if (on) {
        C = std::max(MAXR + 1, std::min(C > 0 ? C : E->o.chunk, E->o.chunk));
        const int nb = (int)((borrow_need(C, E->o.maxctx) + XB - 1) / XB);
        for (auto& c : E->cards) DCHECK(nb <= c.nborrow, "borrow %d > tail %d", nb, c.nborrow);
        auto in_reg = [&](int g) {
            const Card& k = E->cards[gs_card(g)];
            return g >= 0 && gs_idx(g) >= k.used_slots - nb;
        };
        // pre-scan: every expert in the region must sit in a pool slot of its layer and have a host or checkpoint copy
        for (int l = 0; l < NL; l++) {
            Layer& W = E->L[l];
            for (int e = 0; e < NE; e++) {
                if (!in_reg(W.h_slot[e])) continue;
                bool pooled = false;
                for (size_t i = 0; i < W.dpool.size(); i++) pooled |= W.dpool[i] == W.h_slot[e] && W.dyn_e[i] == e;
                if (!pooled || !W.hx[e]) {
                    static bool said = false;
                    if (!said) logf_("WARNING borrowing: layer %d expert %d in the borrowable region is not evictable: prefill "
                                     "runs in rows of %d (slow) this time\n", l, e, MAXR);
                    said = true;
                    return false;
                }
            }
        }
        for (auto& c : E->cards) {
            DevGuard dg(c.dev);
            DCK(cudaDeviceSynchronize());
        }
        E->borrowed.clear();
        for (int l = 0; l < NL; l++) {
            Layer& W = E->L[l];
            for (size_t i = 0; i < W.dpool.size(); i++) {
                const int g = W.dpool[i];
                if (!in_reg(g)) continue;
                if (W.pend_job[i] >= 0) {   // an NVMe read in flight: let the reader finish, drop it
                    XJob& J = E->cards[W.card].xj[W.pend_job[i]];
                    int js;
                    for (auto tw = clk::now(); (js = J.state.load(std::memory_order_acquire)) == 1;) {
                        std::this_thread::sleep_for(std::chrono::microseconds(50));
                        DCHECK(ms_since(tw) < 120000, "NVMe expert read (borrow): no answer for 120 s");   // review r10 L4
                    }
                    J.state.store(0, std::memory_order_release);
                    W.pend_job[i] = -1;
                }
                const int e = W.dyn_e[i];
                if (e >= 0) {
                    set_slot(E, W, e, -1);
                    W.inpool[e] = 0;
                    if (W.bor.empty()) W.bor.assign(NE, 0);
                    W.bor[e] = 1;   // keeps its place in the prefill's accumulation order
                    E->borrowed.push_back({l, (int)i, e});
                }
                W.dyn_e[i] = -1;
                W.pend_e[i] = -2;   // borrowed: neither free nor a victim
            }
        }
        for (int l = 0; l < NL; l++)   // review r9 H1: nothing may point into the region now
            for (int e = 0; e < NE; e++) DCHECK(!in_reg(E->L[l].h_slot[e]), "borrow: layer %d expert %d still in the region", l, e);
        for (auto& c : E->cards) {
            DevGuard dg(c.dev);
            DCK(cudaDeviceSynchronize());   // the table updates ran before the region is overwritten
            uint8_t* p = c.slots + (size_t)(c.used_slots - nb) * XB;
            auto take = [&](size_t b) {
                uint8_t* q = p;
                p += (b + 255) & ~(size_t)255;
                return q;
            };
            const size_t Cs = (size_t)C;
            c.h = (float*)take(Cs * HC * D * 4);
            c.x = (float*)take(Cs * D * 4);
            c.ao = (float*)take(Cs * D * 4);
            c.qr = (float*)take(Cs * QLR * 4);
            c.xb = (bf16*)take(std::max({Cs * 8192, std::min<size_t>(Cs, 512) * 32768, (size_t)MAXR * (32768 + D)}) * 2);
            c.xb_n = std::max({Cs * 8192, std::min<size_t>(Cs, 512) * 32768, (size_t)MAXR * (32768 + D)});
            c.sel = (int*)take(Cs * ITOPK * 4);
            c.cand = take(Cs * (size_t)(E->o.maxctx / CAND_BS + 1));
            c.deq = (bf16*)take((size_t)4096 * 8192 * 2);
            c.w13 = (bf16*)take((size_t)2 * INTER * D * 2);
            c.w2 = (bf16*)take((size_t)D * INTER * 2);
            for (int i = 0; i < NST; i++) c.stage[i] = take(XB);
            DCHECK(p <= c.slots + (size_t)c.used_slots * XB, "borrowed region overflow");
            c.big = true;
        }
        E->borrow_rows = C;
        return true;
    }
    for (auto& c : E->cards) {
        DevGuard dg(c.dev);
        DCK(cudaDeviceSynchronize());
    }
    for (auto& c : E->cards) {
        c.h = c.dws.h; c.x = c.dws.x; c.ao = c.dws.ao; c.qr = c.dws.qr; c.xb = c.dws.xb; c.xb_n = c.dws.xb_n; c.sel = c.dws.sel;
        c.cand = c.dws.cand; c.deq = c.dws.deq; c.w13 = c.dws.w13; c.w2 = c.dws.w2;
        for (int i = 0; i < NST; i++) c.stage[i] = c.dws.stage[i];
        c.big = false;
        c.pfa = nullptr;   // [dsv4-pf12-pfxb]
        c.pfa_bytes = 0;
    }
    E->borrow_rows = 0;
    for (auto [l, i, e] : E->borrowed) {   // copy back: FP4 host copy, else the checkpoint's FP4 bytes (review r9 M1)
        Layer& W = E->L[l];
        Card& k = E->cards[gs_card(W.dpool[i])];
        DevGuard dg(k.dev);
        if (W.hx[e] && W.hfmt[e] == XF_FP4) DCK(cudaMemcpyAsync(slot_ptr(E, W.dpool[i]), W.hx[e], XB, cudaMemcpyHostToDevice, k.cp));
        else ck_copy(E, l, e, slot_ptr(E, W.dpool[i]), k.cp);
    }
    for (auto& c : E->cards) {
        DevGuard dg(c.dev);
        DCK(cudaStreamSynchronize(c.cp));
    }
    for (int l = 0; l < NL; l++) {
        for (auto& pe : E->L[l].pend_e)
            if (pe == -2) pe = -1;
        if (!E->L[l].bor.empty()) std::fill(E->L[l].bor.begin(), E->L[l].bor.end(), 0);
    }
    for (auto [l, i, e] : E->borrowed) {
        Layer& W = E->L[l];
        if (W.h_slot[e] >= 0) continue;   // admitted elsewhere meanwhile
        bool pend = false;   // review r13 H1: an admission of e still pending elsewhere: that one wins
        for (int pe : W.pend_e) pend |= pe == e;
        if (pend) continue;
        set_slot(E, W, e, W.dpool[i]);
        W.inpool[e] = 1;
        W.dyn_e[i] = e;
    }
    E->borrowed.clear();
    return true;
}
// R19 [dsv4-r19-pool] borrowing from the split M23 / GGUF pool: on = evict the experts whose slots overlap the last
// borrow_need(C) bytes of every card's pool (the last layers' regions) and carve the prefill buffers there; off = buffers
// back, the evicted experts copied back from their host copies (= the slot bytes) into the same slots. During the prefill
// an evicted expert keeps its place in the accumulation order (bor) and is staged from its host copy: same bytes, same
// products and order, so the prefill is bit-identical to one without borrowing
static void ws_carve(Card& c, uint8_t* p, size_t C, int maxctx, size_t arena) {
    auto take = [&](size_t b) {
        uint8_t* q = p;
        p += (b + 255) & ~(size_t)255;
        return q;
    };
    c.h = (float*)take(C * HC * D * 4);
    c.x = (float*)take(C * D * 4);
    c.ao = (float*)take(C * D * 4);
    c.qr = (float*)take(C * QLR * 4);
    c.xb = (bf16*)take(std::max({C * 8192, std::min<size_t>(C, 512) * 32768, (size_t)MAXR * (32768 + D)}) * 2);
    c.xb_n = std::max({C * 8192, std::min<size_t>(C, 512) * 32768, (size_t)MAXR * (32768 + D)});
    c.sel = (int*)take(C * ITOPK * 4);
    c.cand = take(C * (size_t)(maxctx / CAND_BS + 1));
    c.deq = (bf16*)take((size_t)4096 * 8192 * 2);
    c.w13 = (bf16*)take((size_t)2 * INTER * D * 2);
    c.w2 = (bf16*)take((size_t)D * INTER * 2);
    for (int i = 0; i < NST; i++) c.stage[i] = take(expert_bytes(XF_FP4));
    c.pfa = arena ? take(arena) : nullptr;   // [dsv4-pf12-pfxb] (borrowed carves only)
    c.pfa_bytes = arena;
}
// review r19 M1: borrowing turned off at load (pool too small): the chunk-sized prefill buffers get their own
// allocation (the borrow layout, one cudaMalloc); without VRAM for them the card keeps its MAXR-row buffers and
// prefill runs in rows of MAXR (degrade, never abort)
// review r20 L2: all cards or none. Every card runs every chunk, so one card without the buffers means rows of MAXR for
// all: the cards that got them give them back (their pools are sized after this), and the failures are counted
static bool ws_alloc_big_all(Engine* E) {
    const size_t nb = borrow_need(E->o.chunk, E->o.maxctx, false);
    std::vector<void*> bp(E->cards.size(), nullptr);
    int nfail = 0;
    for (size_t j = 0; j < E->cards.size(); j++) {
        Card& c = E->cards[j];
        if (c.big) continue;
        DevGuard dg(c.dev);
        if (cudaMalloc(&bp[j], nb) != cudaSuccess) {
            (void)cudaGetLastError();
            bp[j] = nullptr;
            nfail++;
            logf_("WARNING device %d: no VRAM for the prefill chunk buffers (%.2f GB)\n", c.dev, nb / 1e9);
        }
    }
    if (nfail) {
        for (size_t j = 0; j < E->cards.size(); j++)
            if (bp[j]) {
                DevGuard dg(E->cards[j].dev);
                DCK(cudaFree(bp[j]));
            }
        logf_("WARNING prefill chunk buffers missing on %d card(s): given back on the others, prefill runs in rows of %d\n",
              nfail, MAXR);
        return false;
    }
    for (size_t j = 0; j < E->cards.size(); j++) {
        Card& c = E->cards[j];
        if (c.big) continue;
        DevGuard dg(c.dev);
        if (c.dev >= 0 && c.dev < 3) g_mb[c.dev][1] += (double)nb;   // VRAM audit: workspaces
        ws_big_carve(E, c, bp[j]);
    }
    return true;
}
static void ws_big_carve(Engine* E, Card& c, void* p) {
    for (void* q : {(void*)c.h, (void*)c.x, (void*)c.ao, (void*)c.qr, (void*)c.xb, (void*)c.sel, (void*)c.cand}) DCK(cudaFree(q));
    ws_carve(c, (uint8_t*)p, (size_t)E->o.chunk, E->o.maxctx);
    c.dws.h = c.h; c.dws.x = c.x; c.dws.ao = c.ao; c.dws.qr = c.qr; c.dws.xb = c.xb; c.dws.xb_n = c.xb_n; c.dws.sel = c.sel; c.dws.cand = c.cand;
    c.dws.deq = c.deq; c.dws.w13 = c.w13; c.dws.w2 = c.w2;
    for (int i = 0; i < NST; i++) c.dws.stage[i] = c.stage[i];
    c.big = true;
}
static bool gborrow_ws(Engine* E, bool on, int C) {
    if (on) {
        C = std::max(MAXR + 1, std::min(C > 0 ? C : E->o.chunk, E->o.chunk));
        // review pf12 M2: a region that only fails because of the prefetch arena is borrowed without it this time
        bool arena = pfxa_need((size_t)C) > 0;
        size_t need = borrow_need(C, E->o.maxctx, arena) + 256;
        for (auto& c : E->cards) DCHECK(need <= c.dpool_bytes, "borrow %zu > pool %zu", need, c.dpool_bytes);
        auto in_reg = [&](int g) {   // the slot's bytes overlap [dpool_bytes - need, dpool_bytes) of its card
            if (g < 0) return false;
            const Card& k = E->cards[gs_card(g)];
            const int idx = gs_idx(g);
            for (int l = NL - 1; l >= 0; l--)
                if (idx >= k.ps0[l] && idx < k.ps0[l] + k.pnl[l])
                    return k.ploff[l] + (size_t)(idx - k.ps0[l] + 1) * expert_bytes(pool_fmt(E, l)) > k.dpool_bytes - need;
            return false;
        };
        // every expert in the region needs its exact host copy. Review r19 H1: walk h_dslot, not the pool lists (a
        // static expert, one without a host copy, is in no list); the seeding and the memory guard keep them out
        // (in_gtail), so this is the last guard only
        for (bool retry = true; retry;) {
            retry = false;
            for (int l = 0; l < NL && !retry; l++) {
                Layer& W = E->L[l];
                for (int e = 0; e < NE; e++)
                    if (in_reg(W.h_dslot[e]) && ((!W.hx[e] && !E->gm) || W.hfmt[e] != pool_fmt(E, l))) {   // [dsv4-rel-ram]: GGUF: the file
                        if (arena) {   // review pf12 M2
                            arena = false;
                            need = borrow_need(C, E->o.maxctx, false) + 256;
                            retry = true;
                            break;
                        }
                        static long nfb = 0;   // review r19 L5: counted, rate-limited
                        if (nfb++ < 3 || nfb % 100 == 0)
                            logf_("WARNING borrowing (%ld times): layer %d expert %d in the borrowed region has no exact host "
                                  "copy: prefill runs in rows of %d this time\n", nfb, l, e, MAXR);
                        return false;
                    }
            }
        }
        for (auto& c : E->cards) {
            DevGuard dg(c.dev);
            DCK(cudaDeviceSynchronize());   // pending admission copies into the region are done
        }
        E->borrowed.clear();
        for (int l = 0; l < NL; l++) {
            Layer& W = E->L[l];
            for (size_t i = 0; i < W.dpool.size(); i++) {
                if (!in_reg(W.dpool[i])) continue;
                if (W.pend_job[i] >= 0) {   // review rel H1: a GGUF re-pack admission in flight: let the reader finish, drop it
                    XJob& J = E->cards[W.card].xj[W.pend_job[i]];
                    int js;
                    for (auto tw = clk::now(); (js = J.state.load(std::memory_order_acquire)) == 1;) {
                        std::this_thread::sleep_for(std::chrono::microseconds(50));
                        DCHECK(ms_since(tw) < 120000, "GGUF expert re-pack (borrow): no answer for 120 s");
                    }
                    J.state.store(0, std::memory_order_release);   // (a state-3 copy is done: cudaDeviceSynchronize above)
                    W.pend_job[i] = -1;
                }
                if (!W.pend_m.empty() && W.pend_m[i] >= 0) {   // [dsv4-rel-xswap]: a swap's victim D2H is done (synced
                    // above): the victim keeps those bytes as its host copy; the candidate is dropped below
                    W.hx[W.pend_m[i]] = W.pend_h[i];
                    W.pend_m[i] = -1;
                    W.pend_h[i] = nullptr;
                }
                const int e = W.dyn_e[i];
                if (e >= 0) {
                    set_pslot(E, W, e, -1);
                    if (!W.inpool.empty()) W.inpool[e] = 0;
                    if (W.bor.empty()) W.bor.assign(NE, 0);
                    W.bor[e] = 1;
                    E->borrowed.push_back({l, (int)i, e});
                }
                W.dyn_e[i] = -1;
                W.pend_e[i] = -2;   // borrowed (a pending admission into it is dropped): neither free nor a victim
            }
        }
        for (int l = 0; l < NL; l++)
            for (int e = 0; e < NE; e++) DCHECK(!in_reg(E->L[l].h_dslot[e]), "borrow: layer %d expert %d still in the region", l, e);
        for (auto& c : E->cards) {
            DevGuard dg(c.dev);
            DCK(cudaDeviceSynchronize());   // the table updates ran before the region is overwritten
            uint8_t* p = c.dpool + ((c.dpool_bytes - need + 255) & ~(size_t)255);
            ws_carve(c, p, (size_t)C, E->o.maxctx, arena ? pfxa_need((size_t)C) : 0);
            DCHECK(c.stage[NST - 1] + expert_bytes(XF_FP4) <= c.dpool + c.dpool_bytes, "borrowed region overflow");
            DCHECK(!c.pfa || c.pfa + c.pfa_bytes <= c.dpool + c.dpool_bytes, "borrowed prefetch arena overflow");
            c.big = true;
        }
        E->borrow_rows = C;
        return true;
    }
    for (auto& c : E->cards) {
        DevGuard dg(c.dev);
        DCK(cudaDeviceSynchronize());
        c.h = c.dws.h; c.x = c.dws.x; c.ao = c.dws.ao; c.qr = c.dws.qr; c.xb = c.dws.xb; c.xb_n = c.dws.xb_n; c.sel = c.dws.sel;
        c.cand = c.dws.cand; c.deq = c.dws.deq; c.w13 = c.dws.w13; c.w2 = c.dws.w2;
        for (int i = 0; i < NST; i++) c.stage[i] = c.dws.stage[i];
        c.big = false;
        c.pfa = nullptr;   // [dsv4-pf12-pfxb]
        c.pfa_bytes = 0;
    }
    E->borrow_rows = 0;
    for (auto [l, i, e] : E->borrowed) {   // the host copy is the slot's bytes
        Layer& W = E->L[l];
        Card& k = E->cards[gs_card(W.dpool[i])];
        DevGuard dg(k.dev);
        if (!W.hx[e]) {   // [dsv4-rel-ram]: no host copy: re-packed from the GGUF (synchronous, after the prefill)
            static uint8_t* tmp = nullptr;
            if (!tmp) {
                size_t mx = 0;
                for (int l2 = 0; l2 < NL; l2++) mx = std::max(mx, expert_bytes(E->gg->fmt[l2]));
                DCK(cudaHostAlloc((void**)&tmp, mx, cudaHostAllocPortable));
            }
            DCK(cudaStreamSynchronize(k.cp));
            gg_repack_expert(E, l, e, tmp);
            DCK(cudaMemcpyAsync(pool_slot_ptr(E, k, l, W.dpool[i]), tmp, expert_bytes(pool_fmt(E, l)), cudaMemcpyHostToDevice, k.cp));
            DCK(cudaStreamSynchronize(k.cp));
        } else
        DCK(cudaMemcpyAsync(pool_slot_ptr(E, k, l, W.dpool[i]), W.hx[e], expert_bytes(pool_fmt(E, l)), cudaMemcpyHostToDevice, k.cp));
    }
    // no host wait: every card's eager stream waits for all copy-backs before the table updates below (set_pslot runs
    // its k_set_int on est), so a kernel reads a restored slot only after its bytes landed; the lane's host mirror
    // may say "resident" earlier, which only moves work to the GPU in stream order
    for (auto& k : E->cards) {
        DevGuard dg(k.dev);
        DCK(cudaEventRecord(k.ev_brw, k.cp));
    }
    for (auto& c : E->cards) {
        DevGuard dg(c.dev);
        for (auto& k : E->cards) DCK(cudaStreamWaitEvent(c.est, k.ev_brw, 0));
    }
    for (int l = 0; l < NL; l++) {
        for (auto& pe : E->L[l].pend_e)
            if (pe == -2) pe = -1;
        if (!E->L[l].bor.empty()) std::fill(E->L[l].bor.begin(), E->L[l].bor.end(), 0);
    }
    for (auto [l, i, e] : E->borrowed) {
        Layer& W = E->L[l];
        if (W.h_dslot[e] >= 0) continue;   // seeded elsewhere meanwhile
        bool pend = false;
        for (int pe : W.pend_e) pend |= pe == e;
        if (pend) continue;
        set_pslot(E, W, e, W.dpool[i]);
        if (!W.inpool.empty()) W.inpool[e] = 1;
        W.dyn_e[i] = e;
    }
    E->borrowed.clear();
    return true;
}
static double g_instr_ms = 0;   // [dsv4-pf6-net]
static void snap_finish(Engine* E);
// [dsv4-xcall] the held state is pending (pos < xpend_until: a chunk that skipped layers 21-39 ended less than xcone_need()
// rows ago) and the next use is not a long enough prefill: the held tokens are prefilled again from the start (complete state; rare: a
// cancelled chunked prompt continued otherwise, a decode / teacher-forced run right after a part)
static void xcone_flush(Engine* E, const char* why) {
    if (E->pos >= E->xpend_until) return;
    std::vector<int> t = E->toks;
    logf_("[dsv4] xcall: held prompt part of %zu tokens completed by a re-prefill (%s)\n", t.size(), why);
    const int sr = E->snap_req;   // review L1: a prompt-end snapshot requested for the call that flushes stays requested
    // review L3: the sequence's DSpark bookkeeping survives the flush (as across seq_save / seq_load), and the re-prefill's
    // head row (never used: the caller holds the pending token) takes no draw of the seeded sampler
    const bool d_plain = E->ds_plain, d_probe = E->ds_probe, so = E->samp_on;
    const int d_cnt = E->ds_cnt, d_lk = E->lk_hold;
    const double d_spec = E->ds_ms_spec, d_pl = E->ds_ms_plain, d_wms = E->ds_wms, d_wtok = E->ds_wtok, d_lam = E->ds_lam;
    reset(E);
    E->samp_on = false;
    if (!t.empty()) (void)prefill(E, t);
    E->samp_on = so;
    E->ds_plain = d_plain;
    E->ds_probe = d_probe;
    E->ds_cnt = d_cnt;
    E->lk_hold = d_lk;
    E->ds_ms_spec = d_spec;
    E->ds_ms_plain = d_pl;
    E->ds_wms = d_wms;
    E->ds_wtok = d_wtok;
    E->ds_lam = d_lam;
    if (sr >= 0) snap_mark(E, sr);
}
int prefill(Engine* E, const std::vector<int>& ids, std::vector<float>* nll, std::vector<int>* am_all, int follow) {
    follow = std::max(0, follow);
    // [dsv4-xcall] a pending held state continues only through a coned prefill of >= xcone_need() rows (this call + the
    // rows later calls still feed); else it is completed first
    if (E->pos < E->xpend_until && (nll || !E->o.dump.empty() || E->pos + ids.size() + (size_t)follow < (size_t)E->xpend_until))
        xcone_flush(E, "short continuation");
    g_instr_ms = 0;
    E->sv_pass = -1;   // [dsv4-srv-stop]
    E->cap_cancel.store(true, std::memory_order_release);   // review r19 L2: cancel a pending capture, do not finish it
    cap_wait(E);
    kv_reserve(E, E->pos + (int)ids.size());   // [dsv4-rel4-p2]
    int out = -1;
    E->dump_step = -1;
    // [dsv4-pf-pprof]: prefill profile (opt-in), chunked prompts only (a short prompt runs on the decode path)
    if (g_pprof > 0 && (int)ids.size() > E->o.small_prefill && E->o.dump.empty()) pf_begin(E, (int)ids.size());
    auto pf_t0 = clk::now();
    // short prompts: rows of MAXR through the decode path (hot experts on the GPU, cold on the CPU lane) instead of
    // streaming every touched cold expert over PCIe (a 15-token prompt streamed ~68 GB: 1.25 s)
    auto pick_step = [&]() {
    int step = (int)ids.size() <= E->o.small_prefill && E->o.dump.empty() ? MAXR : E->o.chunk;   // review r4 M6
    // R9 borrowing (review r9 M2): the carve sized by this prompt; not evictable (H1 check): rows of MAXR this time
    // review r20 L3: a prompt of <= MAXR rows runs on the decode path anyway: no borrow (it was restored at once)
    {
    const auto tb = clk::now();
    if (step > MAXR && (int)ids.size() > MAXR && (E->borrow || E->gborrow) && !E->cards[0].big &&
        !borrow_ws(E, true, (int)std::min<size_t>(ids.size(), (size_t)E->o.chunk)))
        step = MAXR;
    if (pf_on()) g_pf.borrow_on += ms_since(tb);
    }
    bool allbig = true;
    for (auto& c : E->cards) allbig &= c.big;
    if (step > MAXR && !allbig) step = MAXR;   // review r19 M1: no chunk buffers (degraded load): rows of MAXR
    if (E->cards[0].big && E->borrow_rows) step = std::min(step, E->borrow_rows);
    return step;
    };
    int step = pick_step();
    bool aborted = false;   // [srv-disconnect]
    size_t ab_at = 0;
    // [dsv4-xcall] rows of MAXR (no cone; review M1: also a call of <= MAXR rows): the pending state completed first
    if (E->pos < E->xpend_until && (step <= MAXR || (int)ids.size() <= MAXR)) {
        xcone_flush(E, step <= MAXR && (int)ids.size() > MAXR ? "no chunk buffers" : "rows of MAXR");
        kv_reserve(E, E->pos + (int)ids.size());
        // xcall review L2: the flush's prefill ran a profile of its own (this call's starts again here) and released
        // the held K/V in its reset: this call's borrow is tried once more
        if (g_pprof > 0 && (int)ids.size() > E->o.small_prefill && E->o.dump.empty()) {
            pf_begin(E, (int)ids.size());
            pf_t0 = clk::now();   // review rel9 L5: the total without the flush
        }
        if (step <= MAXR && (int)ids.size() > MAXR) step = pick_step();
    }
    for (size_t s0 = 0; s0 < ids.size(); s0 += step) {
        // [srv-disconnect] the client left: stop before this chunk (the latency is one chunk, not the whole prompt)
        if (E->pf_abort && __atomic_load_n(E->pf_abort, __ATOMIC_ACQUIRE)) {
            aborted = true;
            ab_at = s0;
            break;
        }
        int n = (int)std::min<size_t>(step, ids.size() - s0);
        // review r19 M2: a final chunk of <= MAXR rows runs on the decode path: give the borrowed slots back first (their
        // experts would go to the CPU lane: inexact vs no borrowing, and slower)
        if (n <= MAXR && E->borrow_rows && (E->borrow || E->gborrow)) borrow_ws(E, false);
        DCHECK(E->pos + n <= E->o.maxctx, "context %d > maxctx %d", E->pos + n, E->o.maxctx);
        for (int i = 0; i < n; i++) DCHECK(ids[s0 + i] >= 0 && ids[s0 + i] < V, "token id %d", ids[s0 + i]);
        E->toks.insert(E->toks.end(), ids.begin() + s0, ids.begin() + s0 + n);
        auto tf = clk::now();
        if (pf_on()) g_pf.nchunk++;
        E->cone_ok = !nll && E->o.dump.empty();   // [dsv4-pf15-cone] teacher-forced rows / layer dumps need every row (review L3)
        // [dsv4-ctx2-xcone] a chunk followed (in this call) by at least XCONE_NEED more rows: layers 21-39 read none of
        // its rows (the last chunk's cone at layer 21 starts past them with a zeroed window), so it stops after layer
        // 20's K/V; no head row, no drafter window (the last chunk writes both)
        const bool lo = E->cone_ok && E_cone && !E_xcone_off && n > MAXR && ids.size() + (size_t)follow - (s0 + n) >= (size_t)xcone_need();
        E->cone_lo = lo;
        forward(E, E->pos, n);
        E->cone_ok = false;
        E->cone_lo = false;
        E->xcone_prev = lo;
        if (lo) {
            if (pf_on()) g_pf.fwd += ms_since(tf);
            E->cone_min = std::max(E->cone_min, E->pos + n);   // review ctx1 L2: layers 21-39 hold none of these rows
            E->pos += n;
            E->xpend_until = E->pos + xcone_need();   // [dsv4-xcall] review M1
            continue;
        }
        const int nr = E->fwd_rows;   // rows the last card holds (the last one = the chunk's last position)
        if (pf_on()) { g_pf.fwd += ms_since(tf); tf = clk::now(); }
        if (E->dspark) {
            draft_window(E);   // the captured tail of this chunk (last <= 128 positions)
            g_dn = -1;         // R22 drafter profile: not a draft call
            E->have_draft = false;
        }
        if (pf_on()) { g_pf.draft += ms_since(tf); tf = clk::now(); }
        if (nll) {
            std::vector<int> tgt(n), am(n);
            std::vector<float> nl(n);
            for (int i = 0; i < n; i++) tgt[i] = s0 + i + 1 < ids.size() ? ids[s0 + i + 1] : 0;
            head_rows(E, 0, n, am.data(), tgt.data(), nl.data());
            nll->insert(nll->end(), nl.begin(), nl.end());
            if (am_all) am_all->insert(am_all->end(), am.begin(), am.end());
            out = am[n - 1];
        } else {
            head_rows(E, nr - 1, 1, &out, nullptr, nullptr);
        }
        if (pf_on()) {
            g_pf.head += ms_since(tf);
            tf = clk::now();
            Card& cl = E->cards.back();
            pf_rec(E, (int)E->cards.size() - 1, 0, cl.st, -1, PK_TAIL);
        }
        E->pos += n;
        if (n > MAXR) ring_sync(E);
        if (pf_on()) g_pf.ring += ms_since(tf);
        static const bool pfh = getenv("DSV4_PFHASH") != nullptr;   // [dsv4-pf] exactness check: prompt-end state hash
        if (pfh) {
            Card& cl = E->cards.back();
            DevGuard dg(cl.dev);
            DCK(cudaDeviceSynchronize());
            const auto th = clk::now();   // [dsv4-pf6-net]: the hash's D2H + host loop, reported apart (after the sync: review pf6 L1)
            std::vector<uint32_t> hb((size_t)nr * HC * D);   // [dsv4-pf15-cone]: the rows held (all rows without the cone)
            DCK(cudaMemcpy(hb.data(), cl.h, hb.size() * 4, cudaMemcpyDeviceToHost));
            uint64_t h = 1469598103934665603ull, hl = h;
            for (size_t i = 0; i < hb.size(); i++) { h ^= hb[i]; h *= 1099511628211ull; }
            for (size_t i = (size_t)(nr - 1) * HC * D; i < hb.size(); i++) { hl ^= hb[i]; hl *= 1099511628211ull; }
            logf_("pfhash chunk at %d, %d rows: h all rows %016llx, last row %016llx, first token %d%s\n", E->pos - n, n,
                  (unsigned long long)h, (unsigned long long)hl, out, nr < n ? " (cone: rows held only)" : "");   // review pf15c L2
            g_instr_ms += ms_since(th);
        }
    }
    {
    const auto tb = clk::now();
    if (E->cards[0].big && (E->borrow || E->gborrow)) borrow_ws(E, false);
    if (pf_on()) {
        g_pf.borrow_off += ms_since(tb);
        const long long td = E->egr.tdone_ns.load();
        if (td) g_pf.egdone = (td - std::chrono::duration_cast<std::chrono::nanoseconds>(g_pf.h0.time_since_epoch()).count()) / 1e6;
        const auto te = clk::now();
        pf_end(E, ms_since(pf_t0));
        g_instr_ms += ms_since(te);
    }
    }
    if (aborted) {   // [srv-disconnect] (the chunk workspace is back above; the caller resets the partial sequence)
        logf_("prefill cancelled after %zu of %zu rows (the client left)\n", ab_at, ids.size());
        throw PrefillAborted("prefill cancelled");
    }
    if (E->snap_req >= 0) snap_finish(E);   // [dsv4-srv-pesnap]
    return out;
}

// ---------------------------------------------------------------- DSpark: draft head, verify, rollback

// the draft stages' window KV for the captured main positions mh_pos0 .. + mh_n - 1:
// main_x = main_norm(main_proj(cat of the 3 target layers' stream means)); per stage kv_norm(wkv(main_x)), rope, FP8
static void draft_window(Engine* E) {
    NvR nvw("draft window");
    Card& c = E->cards.back();
    DCK(cudaSetDevice(c.dev));
    cudaStream_t st = c.st;
    g_dn = -1;   // R22 profile: a window not followed by a block (plain steps) is dropped at the next window
    dp_open(st);
    for (int i0 = 0; i0 < E->mh_n; i0 += MAXR) {
        int m = std::min(MAXR, E->mh_n - i0), p = E->mh_pos0 + i0;
        linq(c, E->mp, c.d_mh + (size_t)i0 * 3 * D, 3 * D, m, c.d_mx, D);
        rms(st, c.d_mx, D, E->mnorm, c.d_mx, D, m, D);
        DPT("window main_proj + norm", st);
        for (int k = 0; k < 3; k++) {
            Layer& W = E->M[k];
            float* kv = c.ao;
            linq(c, W.wkv, c.d_mx, D, m, kv, HD);
            rms(st, kv, HD, W.kv_norm, kv, HD, m, HD);
            rope(st, kv, HD, m, 1, HD, c.cosv[0], c.sinv[0], p, 1, 0);
            q_fp8(st, kv, HD, m, HD, nullptr, 0, W.kvw, HD, p, W.kmask);
        }
    }
    E->mh_n = 0;
    DPT("window kv x3", st);
}

// MoE of a draft stage: 128 experts top-3, all in VRAM (no host part)
static const bool dmoe_g = getenv("DSV4_NODMOEG") == nullptr;
static const bool E_gahead_off = getenv("DSV4_NOGAHEAD") != nullptr;   // R19: no speculative capture in draft_block
// R22 [dsv4-r22-dshs]: the drafter's shared expert on the side stream beside its routed experts; DSV4_NODSHS=1 off
static const bool g_dshs = getenv("DSV4_NODSHS") == nullptr;
static void moe_draft(Engine* E, Card& c, Layer& W, int R) {
    cudaStream_t st = c.st;
    const bool shs = g_dshs && W.sh_w1.si;   // (int8 linears: own side-stream scratch)
    if (shs) {   // needs only c.x, joined before the combine
        DCK(cudaEventRecord(c.ev_s0, st));
        DCK(cudaStreamWaitEvent(c.st2, c.ev_s0, 0));
        linq(c, W.sh_w1, c.x, D, R, c.d_shg, 2 * INTER, 0, nullptr, c.st2);
        k_glu_rows<<<cdiv((size_t)R * INTER, 256), 256, 0, c.st2>>>(c.d_shg, c.ones, R, c.d_shg + (size_t)R * 2 * INTER);
        linq(c, W.sh_w2, c.d_shg + (size_t)R * 2 * INTER, INTER, R, c.d_shy, D, 0, nullptr, c.st2);
        DCK(cudaEventRecord(c.ev_s1, c.st2));
    }
    float* logits = c.f32tmp2;
    lin32(c, W.gate_w, DS_NE, D, c.x, D, R, logits, DS_NE);
    DPT("moe gate", st);
    int* rids = (int*)c.d_rpack;
    float* rw = (float*)(rids + MAXR * TOPK);
    float* xq = rw + MAXR * TOPK;
    k_route_d<<<cdiv((size_t)R * 32, 256), 256, 0, st>>>(logits, W.gate_b, R, rids, rw);
    if (g_dn >= 0 && R == 5) {   // R22 profile: distinct experts per stage
        if (!g_drids) { DCK(cudaHostAlloc(&g_drids, 3 * 48 * 4, cudaHostAllocPortable)); memset(g_drids, 0, 3 * 48 * 4); }
        const int k = W.lid - NL;
        DCK(cudaMemcpyAsync(g_drids + 48 * k, rids, 5 * TOPK * 4, cudaMemcpyDeviceToHost, st));
        g_drids[48 * k + 47] = 1;
    }
    const size_t XB = expert_bytes(XF_FP4);
    if (W.xcb3 && c.d_dq8) {   // R22 [dsv4-r22-dmoe8]: x int8 per 32 straight from c.x (no FP8 rounding first)
        const size_t XB3 = expert_bytes(XF_CB3);
        int8_t* hq = c.d_dq8 + (size_t)MAXR * D;
        float* hs = c.d_dq8s + (size_t)MAXR * D / 32;
        gg_q8x_launch(c.x, R * D, c.d_dq8, c.d_dq8s, st);
        DPT("moe route + q", st);
        lchk_pre();   // (R23, review r22 L5: DSV4_LCHECK=1)
        // pair-major grid: choices of one row band in the same wave (shared experts from L2)
        k_dmoe8<4, 10, true, false, true><<<dim3(TOPK * R, 2 * INTER / 64), 256, 0, st>>>(c.d_dq8, c.d_dq8s, 0, rids, W.d_slot,
                                                                                          W.xslots, XB3, c.d_cb, c.d_gu);
        lchk("k_dmoe8 gate/up");
        DPT("moe cb3 gate/up", st);
        k_moe_glu<<<cdiv((size_t)R * TOPK * INTER, 256), 256, 0, st>>>(c.d_gu, rw, rids, W.d_slot, R, c.d_hh, nullptr);
        gg_q8x_launch(c.d_hh, R * TOPK * INTER, hq, hs, st);
        DCK(cudaMemsetAsync(c.d_yx, 0, (size_t)R * TOPK * D * 4, st));   // padding rows
        DPT("moe glu + q + memset", st);
        k_dmoe8<2, 9, true, false, true><<<dim3(TOPK * R, D / 128), 256, 0, st>>>(hq, hs, 1, rids, W.d_slot, W.xslots, XB3,
                                                                                  c.d_cb, c.d_yx);
        lchk("k_dmoe8 down");
        DPT("moe cb3 down", st);
    } else if (W.xcb3) {   // R8: cb3 drafter experts (the slot table is the identity, padding id -> -1)
        q_fp8(st, c.x, D, R, D, xq, D, nullptr, 0);
        DPT("moe route + q", st);
        const size_t XB3 = expert_bytes(XF_CB3);
        if (dmoe_g)   // R19 [dsv4-r19-dmoe]: x staged in shared memory as bf16 (bit-identical), DSV4_NODMOEG=1 off
            k_moe_cb3s<8><<<dim3(2 * INTER / 32, TOPK * R), 256, 0, st>>>(xq, 0, rids, W.d_slot, W.xslots, XB3, c.d_cb, c.d_gu);
        else
        k_moe_cb3<8><<<dim3(2 * INTER / 32, TOPK * R), 256, 0, st>>>(xq, 0, rids, W.d_slot, W.xslots, XB3, c.d_cb, c.d_gu);
        DPT("moe cb3 gate/up", st);
        k_moe_glu<<<cdiv((size_t)R * TOPK * INTER, 256), 256, 0, st>>>(c.d_gu, rw, rids, W.d_slot, R, c.d_hh, nullptr);
        q_fp8(st, c.d_hh, INTER, R * TOPK, INTER, c.d_hh, INTER, nullptr, 0);
        DCK(cudaMemsetAsync(c.d_yx, 0, (size_t)R * TOPK * D * 4, st));   // padding rows (the FP4 down zeroes them itself)
        DPT("moe glu + q + memset", st);
        if (dmoe_g)
            k_moe_cb3s<2><<<dim3(D / 128, TOPK * R), 256, 0, st>>>(c.d_hh, 1, rids, W.d_slot, W.xslots, XB3, c.d_cb, c.d_yx);
        else
        k_moe_cb3<2><<<dim3(D / 128, TOPK * R), 256, 0, st>>>(c.d_hh, 1, rids, W.d_slot, W.xslots, XB3, c.d_cb, c.d_yx);
        DPT("moe cb3 down", st);
    } else {
    q_fp8(st, c.x, D, R, D, xq, D, nullptr, 0);
    k_moe_gemv5<8><<<dim3(2 * INTER / 64, TOPK * R), 256, 0, st>>>(xq, 0, rids, W.d_slot, W.xslots, XB, c.d_gu);
    k_moe_glu<<<cdiv((size_t)R * TOPK * INTER, 256), 256, 0, st>>>(c.d_gu, rw, rids, W.d_slot, R, c.d_hh, nullptr);
    q_fp8(st, c.d_hh, INTER, R * TOPK, INTER, c.d_hh, INTER, nullptr, 0);
    k_moe_gemv5<8><<<dim3(D / 64, TOPK * R), 256, 0, st>>>(c.d_hh, 1, rids, W.d_slot, W.xslots, XB, c.d_yx);
    }
    if (shs) {
        DCK(cudaStreamWaitEvent(st, c.ev_s1, 0));
        DPT("shared expert join (side stream)", st);
    } else {
    linq(c, W.sh_w1, c.x, D, R, c.d_shg, 2 * INTER);   // [w1; w3]: g in columns 0 .. 2303, u in 2304 .. 4607
    DPT("shared w1w3 fp8", st);
    k_glu_rows<<<cdiv((size_t)R * INTER, 256), 256, 0, st>>>(c.d_shg, c.ones, R, c.d_shg + (size_t)R * 2 * INTER);
    linq(c, W.sh_w2, c.d_shg + (size_t)R * 2 * INTER, INTER, R, c.d_shy, D);
    DPT("shared glu + w2 fp8", st);
    }
    k_moe_comb<<<cdiv((size_t)R * D, 256), 256, 0, st>>>(c.d_yx, c.d_zero, c.d_shy, R, c.ao);
    DPT("moe comb", st);
}

// R22 drafter acceptance gate (DSV4_DACC=1, opt-in): every call is a one-row step followed by a full draft block, so
// the target's text does not depend on the drafter; each block (position s, drafts for s + 1 .. s + 5, confidences) is
// scored against the realised tokens at the next reset / exit: mean leading matches of the full block, of the tau-gated
// block (= accepted drafts per pass of the gated policy), gate length, conditional acceptance per row
static const bool g_dacc = getenv("DSV4_DACC") != nullptr;
struct DaccBlk { int s; int d[5]; float c[5]; };
static std::vector<DaccBlk> g_dacc_b;
static void dacc_score(Engine* E) {
    if (!g_dacc || g_dacc_b.empty()) return;
    const std::vector<int>& h = E->toks;
    long n = 0, sl = 0, sg = 0, sn = 0, tried[5] = {}, ok[5] = {};
    for (const DaccBlk& b : g_dacc_b) {
        if (b.s + 5 >= (int)h.size()) continue;
        int L = 0;
        while (L < 5 && b.d[L] == h[b.s + 1 + L]) L++;
        for (int i = 0; i <= std::min(L, 4); i++) { tried[i]++; ok[i] += i < L; }
        float cp = 1.f;
        int k = 0;
        while (k < 5 && (cp *= 1.f / (1.f + expf(-b.c[k]))) >= E->o.dspark_tau) k++;
        n++; sl += L; sg += std::min(L, k); sn += k;
    }
    if (n) logf_("DACC %ld blocks: full-block accepted %.4f, gated (tau %.2f) accepted %.4f, gate length %.4f; per row "
                 "conditional %.3f %.3f %.3f %.3f %.3f\n", n, (double)sl / n, E->o.dspark_tau, (double)sg / n, (double)sn / n,
                 tried[0] ? (double)ok[0] / tried[0] : 0, tried[1] ? (double)ok[1] / tried[1] : 0,
                 tried[2] ? (double)ok[2] / tried[2] : 0, tried[3] ? (double)ok[3] / tried[3] : 0,
                 tried[4] ? (double)ok[4] / tried[4] : 0);
    g_dacc_b.clear();
}

// draft block: row 0 = tok at position s, rows 1..4 = the noise token; 3 stages, then head + Markov bias
// (sequential over the block: each row's bias uses the previous row's draft) -> E->drafts [5], E->dconf [5]
static void draft_block(Engine* E, int s, int tok) {
    NvR nvb("draft block");
    constexpr int B = 5;
    Card& c = E->cards.back();
    DCK(cudaSetDevice(c.dev));
    cudaStream_t st = c.st;
    dp_open(st);
    emb_row(E, tok, E->h_demb);
    for (int i = 1; i < B; i++) memcpy(E->h_demb + (size_t)i * D, E->h_noise.data(), (size_t)D * 2);
    if (E_embm_off) DCK(cudaMemcpyAsync(c.d_demb, E->h_demb, (size_t)B * D * 2, cudaMemcpyHostToDevice, st));
    k_embed<<<cdiv((size_t)B * D, 256), 256, 0, st>>>(E_embm_off ? c.d_demb : E->h_demb, B, c.h, c.pre);   // R17: in place
    int selk = 0;
    for (int k = 0; k < 3; k++) {
        Layer& W = E->M[k];
        DPT(k == 0 ? "embed" : "hc post (ffn)", st);
        hc_mixes(c, W, 0, c.h, B, c.apre, c.apost, c.acomb);
        k_hc_pre_rms<<<B, 1024, 0, st>>>(c.h, c.pre, W.attn_norm, B, c.x);
        DPT("hc mixes + pre (attn)", st);
        attention(E, c, W, W.lid, s, B, &selk);
        k_hc_post<<<cdiv((size_t)B * D, 256), 256, 0, st>>>(c.h, c.ao, c.apost, c.acomb, B);
        hc_mixes(c, W, 1, c.h, B, c.fpre, c.fpost, c.fcomb);
        k_hc_pre_rms<<<B, 1024, 0, st>>>(c.h, c.apre, W.ffn_norm, B, c.x);
        DPT("hc post + mixes + pre (ffn)", st);
        moe_draft(E, c, W, B);
        k_hc_post<<<cdiv((size_t)B * D, 256), 256, 0, st>>>(c.h, c.ao, c.fpost, c.fcomb, B);
        std::swap(c.pre, c.fpre);
    }
    float* xx = c.f32tmp2;                 // [5][5120] collapsed streams (confidence input)
    float* xn = c.f32tmp2 + (size_t)B * D;  // normed (head input)
    DPT("hc post (ffn)", st);
    k_hc_pre<<<cdiv((size_t)B * D, 256), 256, 0, st>>>(c.h, c.pre, B, xx);
    rms(st, xx, D, E->dnorm, xn, D, B, D);
    DPT("head pre + norm", st);
    head_gemv(E, xn, B, true);
    DPT("output head (3 cards)", st);
    static const bool head2 = getenv("DSV4_HEAD2") != nullptr;   // R22 diagnostic: the head again, timed apart
    if (head2) {
        head_gemv(E, xn, B, true);
        DPT("output head again (diagnostic)", st);
    }
    if (E->mk_q) {   // R22 [dsv4-r22-mk8]
        lchk_pre();
        k_mk_init<<<1, 32, 0, st>>>(c.d_dtok, tok, c.d_mkey);
        for (int i = 0; i < B; i++)
            k_markov8<<<82 * 6, 256, 0, st>>>(E->mk_q, E->mk_s, E->mk_emb, i, c.d_dtok, c.d_mkey, c.d_logits + (size_t)i * V,
                                              c.d_embf, V);
        k_mk_last<<<1, 32, 0, st>>>(c.d_dtok, c.d_mkey, B);
        lchk("k_markov8");
    } else
    k_set_int<<<1, 1, 0, st>>>(c.d_dtok, tok);
    for (int i = 0; i < B && !E->mk_q; i++) {
        k_markov<<<cdiv(V, 8), 256, 0, st>>>(E->mk_head, E->mk_emb, c.d_dtok + i, c.d_logits + (size_t)i * V,
                                            c.d_embf + i * 256);
        k_argmax<<<1, 1024, 0, st>>>(c.d_logits + (size_t)i * V, V, c.d_dtok + i + 1);
    }
    DPT("markov + argmax x5", st);
    k_conf<<<B, 256, 0, st>>>(xx, c.d_embf, E->conf_w, c.d_conf);
    if (g_dssel) k_dstats<<<B, 1024, 0, st>>>(c.d_logits, V, c.d_conf + 16);   // R20: the selector's features
    DCK(cudaMemcpyAsync(c.h_dout, c.d_dtok + 1, B * 4, cudaMemcpyDeviceToHost, st));
    DCK(cudaMemcpyAsync(c.h_dout + 64, c.d_conf, (g_dssel ? 16 + 3 * B : B) * 4, cudaMemcpyDeviceToHost, st));
    DPT("conf + D2H", st);
    if (g_dn >= 0) g_denq += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - g_dt0).count();
    // R19 [dsv4-r19-gahead]: with the step graph, the graph of the next one-row step (pos) is captured on the enqueue
    // thread while this block runs (its kernels are all queued on the real stream; the decode thread only waits below):
    // a gated-out call or a plain step then launches it without the ~12 ms synchronous capture; an eager pass cancels it
    if (E->graph && !E_gahead_off && E->pos + 1 < E->o.maxctx && !E->cap_pending) {
        const Engine::GSlot& G = E->gs[E->pos & 1];
        const bool have = G.ready && G.s == E->pos && G.nfwd == E->nfwd && G.seq == E->seq + 1;
        if (!have) {
            const int sp = E->pos;
            E->cap_cancel.store(false, std::memory_order_release);
            E->enq_job = [E, sp] { cap_step(E, sp, true); };
            E->enq_done.store(0, std::memory_order_relaxed);
            E->cap_pending = true;
            E->cap_spec = true;
            E->enq_go.fetch_add(1, std::memory_order_release);
            E->enq_go.notify_one();
            E->g_ahead++;
        }
    }
    DCK(cudaStreamSynchronize(st));
    dp_close();
    memcpy(E->drafts, c.h_dout, B * 4);
    memcpy(E->dconf, c.h_dout + 64, B * 4);
    if (g_dacc) {
        DaccBlk b{s, {}, {}};
        for (int i = 0; i < B; i++) b.d[i] = E->drafts[i], b.c[i] = E->dconf[i];
        g_dacc_b.push_back(b);
    }
    if (g_dssel) memcpy(E->dstat, c.h_dout + 64 + 64, 3 * B * 4);
    E->have_draft = true;
}

// keep the first L positions of the sequence after a verify pass: compressor tails / compressed row counts back to
// L (ratio-2 tail row from the saved pass input), tokens (Engram hashes) truncated; window KV and compressed rows
// past L are rewritten by the next pass before anyone reads them
static void rollback(Engine* E, int L) {
    // [dsv4-pf15-cone]: after a coned prefill the upper layers' window K/V is valid only on its last positions
    DCHECK(L >= E->cone_min, "rollback to %d inside a coned prefill (valid from %d): re-prefill needed", L, E->cone_min);
    const int srcs[4] = {2, 8, 14, 20};
    for (int si = 0; si < 4; si++) {
        if (ratio_of(srcs[si]) == 1) {
            E->tc[si] = std::min(E->tc[si], L);
            continue;
        }
        E->tc[si] = std::min(E->tc[si], L / 2);
        int odd = L % 2;
        if (odd && E->tail_n[si] == 1 && E->sv_base[si] + E->sv_m[si] - 1 == L - 1) {
            E->tail_n[si] = 1;   // the tail already holds position L - 1
            continue;
        }
        E->tail_n[si] = odd;
        if (odd) {
            int row = L - 1 - E->sv_base[si];
            DCHECK(row >= 0 && row < E->sv_m[si], "rollback: source %d row %d of %d", srcs[si], row, E->sv_m[si]);
            Card& c = E->cards[E->L[srcs[si]].card];
            DCK(cudaSetDevice(c.dev));
            DCK(cudaMemcpyAsync(c.tail_kv[si], c.sv_kv[si] + (size_t)row * HD, HD * 4, cudaMemcpyDeviceToDevice, c.st));
            DCK(cudaMemcpyAsync(c.tail_sc[si], c.sv_sc[si] + (size_t)row * HD, HD * 4, cudaMemcpyDeviceToDevice, c.st));
        }
    }
    E->toks.resize(L);
    E->pos = L;
    DCK(cudaSetDevice(E->cards.back().dev));
}

// [dsv4-srv-pesnap] / [dsv4-srv-stop]: sequence state moves outside a verify pass
// before moving the sequence: no Engram rows fetched ahead, no pending / speculative step graph (as reset())
static void seq_quiesce(Engine* E) {
    if (E->eg_pre) {
        E->egr.wait();
        E->eg_pre = false;
    }
    E->cap_cancel.store(true, std::memory_order_release);
    cap_wait(E);
    E->cap_spec = false;
}
// after moving the sequence to E->pos: no draft block (it was for another position), captured graphs invalid, the
// peers' window rings rebuilt from the owner's window K/V (positions pos - ARING .. pos - 1; generation past pos
// overwrote them), every card idle again
static void seq_moved(Engine* E) {
    E->have_draft = false;
    E->nfwd++;
    E->sv_pass = -1;
    if (E->asplit) ring_sync(E);
    for (auto& c : E->cards) {
        DevGuard dg(c.dev);
        DCK(cudaStreamSynchronize(c.st));
    }
}
// [dsv4-rel2-p1] the window ring holds the last ARING positions written (at most pos + MAXR - 1, a verify pass's
// rows): a continuation at L needs its WIN positions before L there
static int kv_floor(Engine* E) { return g_kring ? E->pos + MAXR - (ARING - WIN) : 0; }
void snap_mark(Engine* E, int P) {
    E->snap_pos = -1;   // the old snapshot is replaced (or dropped when this one cannot be taken)
    E->snap_req = P >= 1 && P >= E->pos ? P : -1;
    for (int& g : E->snap_got) g = 0;
}
// [dsv4-rel2-p1] every window ring (main layers on their owners, drafter stages) to its snapshot ring or back
static void kv_snap(Engine* E, bool save) {
    auto cp = [&](Layer& W) {
        Card& c = E->cards[W.card];
        DevGuard dg(c.dev);
        DCK(cudaMemcpyAsync(save ? W.kvsnap : W.kvw, save ? W.kvw : W.kvsnap, (size_t)ARING * HD * 2,
                            cudaMemcpyDeviceToDevice, c.st));
    };
    for (int l = 0; l < NL; l++) cp(E->L[l]);
    if (E->dspark)
        for (int k = 0; k < 3; k++) cp(E->M[k]);
}
// end of the prefill that ran with a request: the state at snap_req is complete (tail rows copied in compressor())
static void snap_finish(Engine* E) {
    const int P = E->snap_req;
    E->snap_req = -1;
    bool ok = P <= E->pos && P >= E->cone_min && P >= kv_floor(E);
    const int srcs[4] = {2, 8, 14, 20};
    for (int si = 0; si < 4 && ok; si++)
        if (ratio_of(srcs[si]) == 2 && (P & 1)) ok = E->snap_got[si];
    E->snap_pos = ok ? P : -1;
    E->snap_cone = E->cone_min;
    if (ok && g_kring) kv_snap(E, true);   // [dsv4-rel2-p1] the rings at P (rows past P are rewritten before use)
}
int snap_pos(Engine* E) { return E->snap_pos; }
int engine_pos(Engine* E) { return E->pos; }
const std::vector<int>& engine_tokens(Engine* E) { return E->toks; }
int snap_restore(Engine* E) {
    const int P = E->snap_pos;
    if (P < 0 || P > E->pos || P > (int)E->toks.size()) return -1;
    seq_quiesce(E);
    seq_stats_flush(E);   // review dec4 L6
    const int srcs[4] = {2, 8, 14, 20};
    for (int si = 0; si < 4; si++) {
        if (ratio_of(srcs[si]) == 1) {
            E->tc[si] = P;
            continue;
        }
        E->tc[si] = P / 2;
        E->tail_n[si] = P & 1;
        if (P & 1) {
            Card& c = E->cards[E->L[srcs[si]].card];
            DevGuard dg(c.dev);
            DCK(cudaMemcpyAsync(c.tail_kv[si], c.snap_kv[si], HD * 4, cudaMemcpyDeviceToDevice, c.st));
            DCK(cudaMemcpyAsync(c.tail_sc[si], c.snap_sc[si], HD * 4, cudaMemcpyDeviceToDevice, c.st));
        }
    }
    E->toks.resize(P);
    E->pos = P;
    E->cone_min = E->snap_cone;
    E->xpend_until = 0;
    E->xcone_prev = false;   // [dsv4-xcall] (P is before any pending rows: the snapshot is taken at a complete state)
    if (g_kring) kv_snap(E, false);   // [dsv4-rel2-p1]
    E->mh_n = 0;   // the drafter's K/V below P was written by the prefill that took the snapshot
    // a fresh sequence's DSpark bookkeeping (as reset(): the same state a full re-prefill starts from)
    E->ds_plain = E_dsplainstart;
    E->ds_probe = false;
    E->lk_hold = 0;
    E->ds_cnt = 0;
    E->ds_ms_spec = 0;
    E->ds_wms = E->ds_wtok = 0;
    E->ds_lam = 0;
    E->ds_ms_plain = E_dsplainstart ? 0 : E->ds_prior;
    seq_moved(E);
    return P;
}
// [dsv4-srv-stop] keep the first L positions (L <= pos) after the reply ended inside a pass (stop string): exact when
// the state at L can be rebuilt: L >= cone_min (the window K/V there is valid), and for an odd L the ratio-2 tail row
// of position L - 1 is at hand (the last verify pass's saved rows, or the snapshot); else 0 and nothing changes
int trim(Engine* E, int L) {
    if (L == E->pos) return 1;
    if (L < 1 || L > E->pos || L < E->cone_min) return 0;
    if (L != E->snap_pos && L < kv_floor(E)) return 0;   // [dsv4-rel2-p1] ring: its window rows were overwritten
    // review dsv4_rel2 M1: the ring keeps rows up to the old pos + MAXR - 1 after the move: a later move back is bounded by
    // this floor, not by the new pos
    const int kfl = kv_floor(E);
    if (L == E->snap_pos) return snap_restore(E) == L;
    seq_quiesce(E);   // review dec4 M1: no capture running while the pass's rows are checked / used
    if (L & 1) {
        if (!E->dspark || E->sv_pass != E->pos) return 0;
        const int srcs[4] = {2, 8, 14, 20};
        for (int si = 0; si < 4; si++) {
            if (ratio_of(srcs[si]) != 2) continue;
            const int row = L - 1 - E->sv_pb[si];
            if (row < 0 || row >= E->sv_pm[si]) return 0;
        }
        for (int si = 0; si < 4; si++) E->sv_base[si] = E->sv_pb[si], E->sv_m[si] = E->sv_pm[si];
    } else {   // no tail row needed (rollback() only reads the saved rows for an odd L)
        for (int& m : E->sv_m) m = 0;
    }
    E->sv_pass = -1;
    rollback(E, L);
    if (E->snap_pos > L) E->snap_pos = -1;
    // drafter main-hidden rows still pending (a one-row step's) below L: into its window first
    if (E->mh_n > 0) {
        E->mh_n = std::max(0, std::min(E->mh_n, L - E->mh_pos0));
        if (E->mh_n) draft_window(E);
    }
    E->mh_n = 0;
    E->cone_min = std::max(E->cone_min, kfl);   // review dsv4_rel2 M1
    seq_moved(E);
    return 1;
}

// [dsv4-rel1-arch] whole-sequence export / import (several conversations through a host archive): everything a
// continuation reads, by absolute position: the src rows [0, tc), the window K/V rows [lo, pos) of every layer and
// DSpark stage in two ranges (review rel1 M1): [pos - ARING, pos) (the window and the peers' rings) and the snapshot's
// [snap_pos - ARING, snap_pos) below it, the ratio-2
// tail and snapshot rows, the drafter's pending main-hidden rows and the scalars. Rows below lo stay whatever the
// cards hold (never read by a continuation at >= lo + ARING: cone_min says so). Load = the same bytes back + seq_moved
// (the path trim / snap_restore take): exact by construction.
namespace {
struct SeqHdr {
    uint32_t magic = 0x31515344u, ver = 4;   // 2: [dsv4-rel2-p1] kring; 3: [dsv4-rel4-p2] park; 4: [dsv4-xcall] xpend
    int pos = 0, lo = 0, slo = 0, shi = 0, cone_min = 0, snap_pos = -1, snap_cone = 0, mh_pos0 = 0, mh_n = 0, ntok = 0, maxctx = 0, dspark = 0;
    int tc[4] = {}, tail_n[4] = {};
    int ds_plain = 0, ds_probe = 0, ds_cnt = 0, lk_hold = 0;
    double ds_ms_spec = 0, ds_ms_plain = 0, ds_wms = 0, ds_wtok = 0, ds_lam = 0;
    int kring = 0;   // [dsv4-rel2-p1] window K/V as rings: the whole ring + the snapshot ring instead of row ranges
    int parked = 0;  // [dsv4-rel4-p2] 1 = seq_park's blob: the src rows stay with the engine (not in the blob)
    int xpend = 0;   // [dsv4-xcall] the last chunk skipped layers 21-39 (Engine::xcone_prev)
    int xpend_until = 0;   // [dsv4-xcall] review M1
};
struct SeqPiece {   // one device range of the blob
    int card;
    void* dev;
    size_t bytes;
};
// the device ranges of the state at h (same order for save and load); src pieces once (their owner card)
void seq_pieces(Engine* E, const SeqHdr& h, std::vector<SeqPiece>& v) {
    const int srcs[4] = {2, 8, 14, 20};
    if (h.kring) {   // [dsv4-rel2-p1] the ring (positions pos - ARING .. pos - 1), the snapshot ring when one is held
        for (int r = 0; r < (h.snap_pos >= 0 ? 2 : 1); r++) {
            for (int l = 0; l < NL; l++)
                v.push_back({E->L[l].card, r ? E->L[l].kvsnap : E->L[l].kvw, (size_t)ARING * HD * 2});
            if (h.dspark)
                for (int k = 0; k < 3; k++)
                    v.push_back({(int)E->cards.size() - 1, r ? E->M[k].kvsnap : E->M[k].kvw, (size_t)ARING * HD * 2});
        }
    }
    for (int r = 0; r < 2 && !h.kring; r++) {   // window rows: [lo, pos), then the snapshot's [slo, shi) (empty: shi <= slo)
        const int a = r ? h.slo : h.lo, n = r ? h.shi - h.slo : h.pos - h.lo;
        if (n <= 0) continue;
        for (int l = 0; l < NL; l++)
            v.push_back({E->L[l].card, E->L[l].kvw + (size_t)a * HD, (size_t)n * HD * 2});
        if (h.dspark)
            for (int k = 0; k < 3; k++)
                v.push_back({(int)E->cards.size() - 1, E->M[k].kvw + (size_t)a * HD, (size_t)n * HD * 2});
    }
    for (int si = 0; si < 4; si++) {
        const int k = E->L[srcs[si]].card;
        Card& c = E->cards[k];
        if (h.tc[si] > 0 && !h.parked) {
            v.push_back({k, c.src[si].ckv, (size_t)h.tc[si] * HD * 2});
            v.push_back({k, c.src[si].ik, (size_t)h.tc[si] * IDIM * 2});
        }
        v.push_back({k, c.tail_kv[si], (size_t)HD * 4});
        v.push_back({k, c.tail_sc[si], (size_t)HD * 4});
        v.push_back({k, c.snap_kv[si], (size_t)HD * 4});
        v.push_back({k, c.snap_sc[si], (size_t)HD * 4});
    }
    if (h.dspark && h.mh_n > 0) v.push_back({(int)E->cards.size() - 1, E->cards.back().d_mh, (size_t)h.mh_n * 3 * D * 4});
}
void seq_ranges(Engine* E, SeqHdr& h) {
    h.lo = std::max(0, E->pos - ARING);
    h.slo = h.shi = 0;
    if (E->snap_pos >= 0 && E->snap_pos <= E->pos) {
        h.slo = std::max(0, E->snap_pos - ARING);
        h.shi = std::min(E->snap_pos, h.lo);
    }
}
void seq_sync_all(Engine* E) {
    for (auto& c : E->cards) {
        DevGuard dg(c.dev);
        DCK(cudaDeviceSynchronize());
    }
}
}  // namespace

size_t seq_bytes(Engine* E, bool park) {
    if (E->pos <= 0) return 0;
    SeqHdr h;
    h.parked = park;
    h.pos = E->pos;
    h.snap_pos = E->snap_pos;
    seq_ranges(E, h);
    h.dspark = E->dspark;
    h.kring = g_kring;
    h.mh_n = E->mh_n;
    for (int i = 0; i < 4; i++) h.tc[i] = E->tc[i];
    std::vector<SeqPiece> v;
    seq_pieces(E, h, v);
    size_t b = sizeof(SeqHdr) + E->toks.size() * 4;
    for (auto& p : v) b += p.bytes;
    return b;
}
static size_t seq_save_i(Engine* E, std::vector<uint8_t>& out, bool park);
size_t seq_save(Engine* E, std::vector<uint8_t>& out) { return seq_save_i(E, out, false); }
static size_t seq_save_i(Engine* E, std::vector<uint8_t>& out, bool park) {
    out.clear();
    if (E->pos <= 0 || E->snap_req >= 0 || (int)E->toks.size() != E->pos) return 0;
    seq_quiesce(E);
    seq_stats_flush(E);
    seq_sync_all(E);
    SeqHdr h;
    h.parked = park;
    h.pos = E->pos;
    seq_ranges(E, h);
    h.cone_min = E->cone_min;
    h.snap_pos = E->snap_pos;
    h.snap_cone = E->snap_cone;
    h.mh_pos0 = E->mh_pos0;
    h.mh_n = E->mh_n;
    h.ntok = (int)E->toks.size();
    h.maxctx = E->o.maxctx;
    h.dspark = E->dspark;
    h.kring = g_kring;
    for (int i = 0; i < 4; i++) h.tc[i] = E->tc[i], h.tail_n[i] = E->tail_n[i];
    h.ds_plain = E->ds_plain;
    h.ds_probe = E->ds_probe;
    h.ds_cnt = E->ds_cnt;
    h.lk_hold = E->lk_hold;
    h.ds_ms_spec = E->ds_ms_spec;
    h.ds_ms_plain = E->ds_ms_plain;
    h.ds_wms = E->ds_wms;
    h.ds_wtok = E->ds_wtok;
    h.ds_lam = E->ds_lam;
    h.xpend = E->xcone_prev;   // [dsv4-xcall]
    h.xpend_until = E->xpend_until;
    std::vector<SeqPiece> v;
    seq_pieces(E, h, v);
    size_t b = sizeof(SeqHdr) + (size_t)h.ntok * 4;
    for (auto& p : v) b += p.bytes;
    out.resize(b);
    uint8_t* o = out.data();
    memcpy(o, &h, sizeof h);
    o += sizeof h;
    memcpy(o, E->toks.data(), (size_t)h.ntok * 4);
    o += (size_t)h.ntok * 4;
    for (auto& p : v) {
        DevGuard dg(E->cards[p.card].dev);
        DCK(cudaMemcpy(o, p.dev, p.bytes, cudaMemcpyDeviceToHost));
        o += p.bytes;
    }
    return b;
}
// [dsv4-rel4-p2] the parked id's rows become the active sequence's run: in place when its base leaves room to grow to
// the sequence capacity, else moved on the device (every card) to a picked base, or from the host copy (owner card, then
// the peers as seq_load); parked runs in the way leave VRAM first. false = dropped (no rows anywhere)
// 1 = taken, 0 = dropped (no rows anywhere), -4 = no VRAM to map its rows now (nothing moved: the parked rows stay)
static int kv_take(Engine* E, int rid) {
    auto it = E->kvres.find(rid);
    if (it == E->kvres.end() || it->second.dead) return 0;
    Engine::KvRes& r = it->second;
    bool inpl = false, moved = false;
    if (E->lz) {   // every destination's pages + the rope of its positions mapped first, in one growth (review rel7 L3,
        int top[4] = {-1, -1, -1, -1};   // kb re-review M1': a later source's failure after an earlier one moved)
        for (int si = 0; si < 4; si++)
            if (!(r.np[si] > 0 && r.base[si] + E->kvcap[si] <= E->kvP[si]))
                top[si] = kv_pick(E, si, rid) + std::max((r.tc[si] + KVPG - 1) / KVPG, 1);
        if (!lz_ensure(E, top, r.tc[3])) return -4;
    }
    for (int si = 0; si < 4; si++) {
        const int np = (r.tc[si] + KVPG - 1) / KVPG;
        if (r.np[si] > 0 && r.base[si] + E->kvcap[si] <= E->kvP[si]) {   // in place
            E->ab[si] = r.base[si];
            for (int p = r.base[si]; p < r.base[si] + r.np[si]; p++) E->kvown[si][p] = -1;
            E->ares[si] = r.np[si];
            inpl = true;
            continue;
        }
        const int b = kv_pick(E, si, rid);
        if (E->lz) {   // [dsv4-ctx-lazy] the destination pages mapped first
            int top[4] = {-1, -1, -1, -1};
            top[si] = b + std::max(np, 1);
            // (mapped above already; only when an eviction moved this pick: sources 0..si-1 are taken by then, so a
            // failure drops the entry as before instead of keeping half-taken rows: kb re-review 2 M1'')
            if (!lz_ensure(E, top, -1)) return si > 0 ? 0 : -4;
        }
        for (int p = b; p < b + std::max(np, 1); p++)   // others in the way move away or leave VRAM (own pages stay)
            if (E->kvown[si][p] > 0 && E->kvown[si][p] != rid) kv_move_out(E, E->kvown[si][p], si, b, b + std::max(np, 1));
        if (r.np[si] > 0) {   // device move, every card; dst below src: forward chunks of at most the distance
            const size_t d = (size_t)(r.base[si] - b) * KVPG;
            for (auto& c : E->cards) {
                DevGuard dg(c.dev);
                for (size_t i = 0; i < (size_t)r.tc[si]; i += d) {
                    const size_t m = std::min(d, (size_t)r.tc[si] - i);
                    DCK(cudaMemcpyAsync(c.src[si].pck + ((size_t)b * KVPG + i) * HD, c.src[si].pck + ((size_t)r.base[si] * KVPG + i) * HD,
                                        m * HD * 2, cudaMemcpyDeviceToDevice, c.st));
                    DCK(cudaMemcpyAsync(c.src[si].pik + ((size_t)b * KVPG + i) * IDIM,
                                        c.src[si].pik + ((size_t)r.base[si] * KVPG + i) * IDIM, m * IDIM * 2,
                                        cudaMemcpyDeviceToDevice, c.st));
                }
            }
            for (int p = r.base[si]; p < r.base[si] + r.np[si]; p++)
                if (E->kvown[si][p] == rid) E->kvown[si][p] = 0;
            moved = true;
        }
        for (int p = b; p < b + std::max(np, 1); p++) E->kvown[si][p] = -1;
        E->ab[si] = b;
        E->ares[si] = std::max(np, 1);
        r.np[si] = 0;
    }
    kv_bind(E);
    E->nfwd++;
    if (moved) E->kv_n[1]++;
    else if (inpl) E->kv_n[0]++;
    for (int si = 0; si < 4 && E->lz; si++)   // [dsv4-ctx-lazy]
        if (E->ab[si] + E->ares[si] > E->kvB[si]) E->lz_t_long = clk::now();
    kv_dev_sync(E);   // review rel4 H1: the device moves done before the host rows are written (sync copies, other streams)
    if (!r.host.empty()) {   // [dsv4-rel5-p3] from host: every card's copy straight from the (pinned) buffer, all links at once
        const auto t0 = std::chrono::steady_clock::now();
        const uint8_t* s = r.host.p;
        for (int si = 0; si < 4; si++) {
            if (r.tc[si] <= 0) continue;
            const size_t bk = (size_t)r.tc[si] * HD * 2, bi = (size_t)r.tc[si] * IDIM * 2;
            if (r.host.pin) {
                for (auto& c : E->cards) {
                    DevGuard dg(c.dev);
                    DCK(cudaMemcpyAsync(c.src[si].ckv, s, bk, cudaMemcpyHostToDevice, c.st));
                    DCK(cudaMemcpyAsync(c.src[si].ik, s + bk, bi, cudaMemcpyHostToDevice, c.st));
                }
            } else {   // review rel5 L3: pageable (pinning failed / off-switch): the owner once, then the peers (as before)
                Card& c = E->cards[E->L[KV_SRCS[si]].card];
                {
                    DevGuard dg(c.dev);
                    DCK(cudaMemcpy(c.src[si].ckv, s, bk, cudaMemcpyHostToDevice));
                    DCK(cudaMemcpy(c.src[si].ik, s + bk, bi, cudaMemcpyHostToDevice));
                }
                for (auto& o2 : E->cards) {
                    if (o2.dev == c.dev) continue;
                    DCK(cudaMemcpyPeer(o2.src[si].ckv, o2.dev, c.src[si].ckv, c.dev, bk));
                    DCK(cudaMemcpyPeer(o2.src[si].ik, o2.dev, c.src[si].ik, c.dev, bi));
                }
            }
            s += bk + bi;
        }
        kv_dev_sync(E);   // the rows are in place before any other stream reads them; the buffer may go back to the cache
        if (g_kvh_log)
            logf_("K/V pool: restored %.1f MB from host (%s) to %d cards in %.2f ms\n", r.host.n / 1e6,
                  r.host.pin ? "pinned" : "pageable", (int)E->cards.size(),
                  std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
        E->kv_host -= r.host.size();
        E->kv_n[2]++;
    }
    kvh_put(E, r.host);   // review rel5 L1 (a no-op without a buffer)
    E->kvres.erase(it);
    return 1;
}
static int seq_load_i(Engine* E, const std::vector<uint8_t>& in, int rid);
int seq_load(Engine* E, const std::vector<uint8_t>& in) { return seq_load_i(E, in, 0); }
int seq_park(Engine* E, std::vector<uint8_t>& out) {
    if (seq_save_i(E, out, true) == 0) return 0;
    const int rid = ++E->kv_rid;
    Engine::KvRes& r = E->kvres[rid];
    for (int si = 0; si < 4; si++) {   // the active run (its used pages) changes owner; the rest is free again
        r.base[si] = E->ab[si];
        r.tc[si] = E->tc[si];
        r.np[si] = std::min(E->ares[si], (E->tc[si] + KVPG - 1) / KVPG);
        for (int p = E->ab[si]; p < E->ab[si] + E->ares[si]; p++)
            if (E->kvown[si][p] == -1) E->kvown[si][p] = p < E->ab[si] + r.np[si] ? rid : 0;
        E->ares[si] = 0;
    }
    reset(E);
    return rid;
}
int seq_unpark(Engine* E, const std::vector<uint8_t>& in, int id) {
    if (id <= 0) return -1;
    const int r = seq_load_i(E, in, id);
    if (r < 0 && r != -4) seq_res_free(E, id);   // (-4: no VRAM now, the parked rows stay: kb re-review M1')
    return r;
}
void seq_res_free(Engine* E, int id) {
    auto it = E->kvres.find(id);
    if (it == E->kvres.end()) return;
    for (int si = 0; si < 4; si++)
        for (int p = it->second.base[si]; p < it->second.base[si] + it->second.np[si]; p++)
            if (E->kvown[si][p] == id) E->kvown[si][p] = 0;
    E->kv_host -= it->second.host.size();
    kvh_put(E, it->second.host);   // [dsv4-rel5-p3]
    E->kvres.erase(it);
}
bool seq_res_alive(Engine* E, int id) {
    auto it = E->kvres.find(id);
    return it != E->kvres.end() && !it->second.dead;
}
void seq_res_floor(Engine* E, double gib) { E->kv_floor_gib = gib; }
size_t seq_res_host(Engine* E, int id) {
    auto it = E->kvres.find(id);
    return it == E->kvres.end() ? 0 : it->second.host.size();
}
// [dsv4-rel5-p3] the host buffers: pinned MB held (in use + cache), cache MB, counters reused / new pinned / new pageable
void seq_res_hostinfo(Engine* E, double* pinned_mb, double* cache_mb, long cnt[3]) {
    if (pinned_mb) *pinned_mb = E->kv_pinned / 1e6;
    if (cache_mb) *cache_mb = E->kvfree_b / 1e6;
    if (cnt)
        for (int i = 0; i < 3; i++) cnt[i] = E->kvh_n[i];
}
void seq_res_trim(Engine* E) { kvh_trim(E, 0); }
void seq_res_info(Engine* E, int* nvram, double* vram_mb, double* host_mb, long cnt[5]) {
    int n = 0;
    double vb = 0;
    for (auto& [id, r] : E->kvres) {
        bool in = false;
        for (int si = 0; si < 4; si++) {
            in |= r.np[si] > 0;
            vb += (double)r.np[si] * KVPG * (HD + IDIM) * 2;
        }
        n += in;
    }
    if (nvram) *nvram = n;
    if (vram_mb) *vram_mb = vb / 1e6;
    if (host_mb) *host_mb = E->kv_host / 1e6;
    if (cnt)
        for (int i = 0; i < 5; i++) cnt[i] = E->kv_n[i];
}
static int seq_load_i(Engine* E, const std::vector<uint8_t>& in, int rid) {
    SeqHdr h;
    if (in.size() < sizeof h) return -1;
    memcpy(&h, in.data(), sizeof h);
    if (h.magic != SeqHdr().magic || h.ver != SeqHdr().ver || h.maxctx != E->o.maxctx || h.dspark != (int)E->dspark || h.kring != (int)g_kring ||
        h.pos <= 0 || h.pos > E->o.maxctx || h.ntok != h.pos || h.lo < 0 || h.lo > h.pos || h.slo < 0 || h.shi > h.lo ||
        h.mh_n < 0 || h.mh_n > WIN || h.snap_pos > h.pos)
        return -1;
    for (int si = 0; si < 4; si++)   // review rel1 L: the header's row counts within this engine's buffers
        if (h.tc[si] < 0 || h.tc[si] > (si == 3 ? E->o.maxctx : E->o.maxctx / 2 + 1)) return -1;
    std::vector<SeqPiece> v;
    seq_pieces(E, h, v);
    size_t b = sizeof h + (size_t)h.ntok * 4;
    for (auto& p : v) b += p.bytes;
    if (b != in.size() || h.parked != (rid > 0)) return -1;
    if (rid > 0) {   // [dsv4-rel4-p2] the parked rows: the same counts as the blob's
        auto it = E->kvres.find(rid);
        if (it == E->kvres.end() || it->second.dead) return -1;
        for (int si = 0; si < 4; si++)
            if (it->second.tc[si] != h.tc[si]) return -1;
    }
    reset(E);   // quiesce, counters (the bookkeeping is overwritten below)
    if (rid > 0) {   // review rel7 M3 (a dead entry was refused above)
        const int kt = kv_take(E, rid);
        if (kt <= 0) return kt < 0 ? -4 : -1;   // [kb-dsv4-ctxgrow] -4: the parked rows stay for a later try
    }
    else {
        try {
            kv_reserve_rows(E, h.tc);   // [dsv4-rel4-p2] pages for the rows (the views move to the new base)
        } catch (const CtxNoRoom& ex) {   // [kb-dsv4-ctxgrow] no VRAM for its rows now: a failed restore (the caller re-prefills or fails)
            logf_("WARNING archive: a conversation of %d positions not restored now: %s\n", h.pos, ex.what());
            return -4;   // (review kb M3: the blob is fine; the caller keeps its archive entry)
        }
    }
    v.clear();
    seq_pieces(E, h, v);
    seq_sync_all(E);
    const uint8_t* s = in.data() + sizeof h;
    E->toks.assign((const int*)s, (const int*)s + h.ntok);
    s += (size_t)h.ntok * 4;
    const int srcs[4] = {2, 8, 14, 20};
    for (auto& p : v) {
        DevGuard dg(E->cards[p.card].dev);
        DCK(cudaMemcpy(p.dev, s, p.bytes, cudaMemcpyHostToDevice));
        s += p.bytes;
    }
    // the src rows on every other card (each card attends with the split; without it the later consumers): the same
    // rows the compressor broadcast, copied from the owner
    for (int si = 0; si < 4; si++) {
        if (h.tc[si] <= 0 || h.parked) continue;   // (parked rows: placed by kv_take)
        Card& c = E->cards[E->L[srcs[si]].card];
        for (auto& o2 : E->cards) {
            if (o2.dev == c.dev) continue;
            DCK(cudaMemcpyPeer(o2.src[si].ckv, o2.dev, c.src[si].ckv, c.dev, (size_t)h.tc[si] * HD * 2));
            DCK(cudaMemcpyPeer(o2.src[si].ik, o2.dev, c.src[si].ik, c.dev, (size_t)h.tc[si] * IDIM * 2));
        }
    }
    seq_sync_all(E);   // review rel1 H1: cudaMemcpyPeer runs on the legacy stream, the engine's streams do not wait on it
    E->pos = h.pos;
    for (int i = 0; i < 4; i++) E->tc[i] = h.tc[i], E->tail_n[i] = h.tail_n[i];
    // a continuation below pos would read window rows not restored (only the snapshot's rows below lo are): moves back
    // go through the snapshot only
    E->cone_min = h.lo > 0 ? h.pos : h.cone_min;
    E->snap_pos = h.snap_pos;
    E->snap_cone = h.snap_pos >= 0 && h.slo > 0 ? std::max(h.snap_cone, h.snap_pos) : h.snap_cone;
    E->mh_pos0 = h.mh_pos0;
    E->mh_n = h.mh_n;
    E->ds_plain = h.ds_plain;
    E->ds_probe = h.ds_probe;
    E->ds_cnt = h.ds_cnt;
    E->lk_hold = h.lk_hold;
    E->ds_ms_spec = h.ds_ms_spec;
    E->ds_ms_plain = h.ds_ms_plain;
    E->ds_wms = h.ds_wms;
    E->ds_wtok = h.ds_wtok;
    E->ds_lam = h.ds_lam;
    E->xcone_prev = h.xpend != 0;   // [dsv4-xcall]
    E->xpend_until = h.xpend_until;
    seq_moved(E);
    return E->pos;
}

// memory guard (decode thread, between steps): when MemAvailable falls below 8 GiB (other processes or our growth),
// the host copies of resident pool experts are freed (they stay in VRAM as static experts, admissions over them
// stop) until 12 GiB are free again, before the kernel's OOM killer has to act. Safe here: the lane is idle, the
// prefetch only reads cold experts, pending admissions read the admitted expert's copy, not these.
static void mem_guard(Engine* E, bool force) {
    if (!force && ++E->guard_t < 32) return;
    E->guard_t = 0;
    static const double lo = getenv("DSV4_GUARD_GIB") ? atof(getenv("DSV4_GUARD_GIB")) : 8.0, hi = lo + 4.0;
    double av = mem_avail_gib();
    // review r13 M2: with the swap tier the free homes are the first thing to give back (own allocations, down to a
    // floor that keeps the tier moving); the latch only holds while no such home is left
    const size_t HFLOOR = 8;
    auto own_homes = [&] {
        size_t n = 0;
        for (auto& h : E->hfree) n += h.second;
        return n;
    };
    if (E->guard_empty && own_homes() > HFLOOR) E->guard_empty = false;
    if (av >= 0 && av < lo && E->kvfree_b) {   // review rel5 M1: the K/V host cache goes before any expert copy
        logf_("memory guard: MemAvailable %.2f GiB: K/V host cache released (%.2f GiB)\n", av, E->kvfree_b / 1073741824.0);
        kvh_trim(E, 0);
        av = mem_avail_gib();
    }
    if (av < 0 || av >= lo || E->guard_empty) return;
    logf_("WARNING memory guard: MemAvailable %.2f GiB < %.1f GiB: dropping host copies of resident pool experts\n", av, lo);
    int nd = 0, nh = 0;
    double freed = 0;
    size_t nown = own_homes();
    double hfreed = 0;
    for (size_t j = E->hfree.size(); j-- > 0 && av < hi && nown > HFLOOR;) {
        if (!E->hfree[j].second) continue;   // arena slices cannot be freed one by one
        host_free(E->hfree[j].first);
        E->hfree.erase(E->hfree.begin() + j);
        nown--;
        hfreed += expert_bytes(XF_FP4);
        if (++nh % 16 == 0) av = mem_avail_gib();
    }
    if (nh || !E->hfree.empty())   // re-review N4 / R1: homes apart from copies; slice homes cannot be given back
        logf_("memory guard: %d free swap homes released (%.2f GiB); free homes left %zu own + %zu arena slices (not "
              "releasable)\n", nh, hfreed / 1073741824.0, nown, E->hfree.size() - nown);
    av = mem_avail_gib();
    int ntail = 0;   // review r20 L4: copies kept because their slot is in the borrowable tail (logged)
    for (int l = 0; l < NL && av < hi; l++) {
        Layer& W = E->L[l];
        if (W.own.empty() || W.inpool.empty()) continue;
        for (size_t j = 0; j < W.dyn_e.size() && av < hi;) {
            int e = W.dyn_e[j];
            // review r16 M1: the M23 pool (non-FP4) too: a resident expert with its own host copy gives the copy back
            // and stays static in its slot (the lane never reads a resident expert; no admission can evict it)
            const bool res = e >= 0 && (E->dyn_fp4 ? W.h_slot[e] >= 0 : W.h_dslot[e] >= 0);
            const bool cand = e >= 0 && W.pend_e[j] < 0 && W.own[e] && res && W.hx[e];
            const bool tail = (E->dyn_fp4 && in_tail(E, W.dpool[j])) || in_gtail(E, W.dpool[j]);
            if (cand && tail) ntail++;
            if (!cand || tail) { j++; continue; }   // review r19 H1: nor the split pool's borrowable tail
            W.dpool.erase(W.dpool.begin() + j);
            W.dyn_e.erase(W.dyn_e.begin() + j);
            W.pend_e.erase(W.pend_e.begin() + j);
            W.pend_job.erase(W.pend_job.begin() + j);
            W.pend_t.erase(W.pend_t.begin() + j);
            if (!W.pend_m.empty()) { W.pend_m.erase(W.pend_m.begin() + j); W.pend_h.erase(W.pend_h.begin() + j); }
            DCK(cudaEventDestroy(W.pend_ev[j]));
            W.pend_ev.erase(W.pend_ev.begin() + j);
            W.inpool[e] = 0;
            host_free((void*)W.hx[e]);
            W.hx[e] = nullptr;
            W.own[e] = 0;
            freed += expert_bytes((XFmt)W.hfmt[e]);
            nd++;
            if (nd % 16 == 0) av = mem_avail_gib();
        }
    }
    av = mem_avail_gib();
    if (!nd && !nh) E->guard_empty = true;
    logf_("memory guard: %d pool copies dropped (%.2f GiB, those slots are static now), %d kept (slot in the borrowable "
          "tail), MemAvailable %.2f GiB%s\n", nd, freed / 1073741824.0, ntail, av, nd || nh ? "" : " - nothing left to drop");
}

// R17: after a one-row decode() step: with the step graph, the drafter's main-hidden row (d_mh row 0, written by the
// graph) is position pos - 1; the next step's capture (enqueue thread) must be done before the drafter launches on the
// eager streams (the capture swaps them), and every card's graph done before the drafter's head split uses them
static void ds_after_plain(Engine* E) {
    if (!E->graph) return;
    cap_wait(E);
    for (auto& c : E->cards) {
        DevGuard dg(c.dev);
        DCK(cudaStreamSynchronize(c.est));
    }
    E->mh_pos0 = E->pos - 1;
    E->mh_n = 1;
}
// R17 prompt lookup beside DSpark (ideas: the 27B / QFN n-gram drafts, Maya's suffix drafts): the longest recent
// earlier occurrence (>= DSV4_LOOKUP tokens, at most 32) of the history's suffix (toks + the pending token t); its
// following tokens are the drafts. 0 = no match.
static int g_lk_min = getenv("DSV4_LOOKUP") ? atoi(getenv("DSV4_LOOKUP")) : 0;
static int lk_min() { return g_lk_min; }
void set_lookup(Engine* E, int n) { (void)E; g_lk_min = n; }
static int lookup_find(Engine* E, int t, int* out, int maxd) {
    const int Lmin = lk_min();
    if (Lmin <= 0 || maxd <= 0) return 0;
    const std::vector<int>& h = E->toks;
    const int N = (int)h.size() + 1;   // h + t
    auto at = [&](int i) { return i == N - 1 ? t : h[i]; };
    if (N < Lmin + 2) return 0;
    int bj = -1, bl = 0;
    const int jlo = std::max(Lmin - 1, N - 1 - 65536);
    for (int j = N - 3; j >= jlo; j--) {   // R18 (review r17 L3): j <= N - 3: at least one following token
        if (h[j] != t) continue;
        int L = 1;
        while (L < 32 && j - L >= 0 && h[j - L] == at(N - 1 - L)) L++;
        if (L > bl) { bl = L; bj = j; if (L >= 32) break; }
    }
    if (bl < Lmin) return 0;
    const int nd = std::min(maxd, std::max(2, bl));
    int k = 0;
    for (int i = bj + 1; i < N - 1 && k < nd; i++) out[k++] = h[i];
    return k;
}
// R8 adaptive DSpark after 8 passes: plain steps when the passes' running ms/token loses; R19: no plain rate yet (spec
// start, first sequence): 8 plain measuring steps, then the same comparison (ds_probe)
// R19 [dsv4-r19-judge]: the passes' rate is their total ms / total tokens over the window (the per-pass EMA let a short
// run of 1-token passes win; code lost 32-token plain runs at ~16.9 ms against passes at ~12 ms / token) and must lose
// by 10 % (DSV4_DSJUDGE_OLD=1: the R8 rule)
static const bool E_judge_old = getenv("DSV4_DSJUDGE_OLD") != nullptr;
static void ds_judge(Engine* E) {
    E->ds_cnt = 0;
    const double wr = E->ds_wtok > 0 ? E->ds_wms / E->ds_wtok : 0;
    E->ds_wms = E->ds_wtok = 0;
    if (!E_judge_old && wr > 0) E->ds_ms_spec = wr;
    if (E->ds_ms_plain <= 0) {
        E->ds_plain = true;
        E->ds_probe = true;   // ds_ms_spec keeps the passes' rate for the judgement after the probe
        return;
    }
    if (E->ds_ms_spec > E->ds_ms_plain * (E_judge_old ? 1.0 : 1.10)) {
        E->ds_plain = true;
        E->ds_ms_spec = 1;   // marks "lost": the next probe after 32 plain tokens
    }
}
static long g_lk[3];   // R17 lookup passes, drafts verified, drafts accepted
static int decode_i(Engine* E, int t, StepTimes* tm, bool from_spec);
static long g_conf[10][2];   // DSpark confidence calibration: drafts tested / accepted per sigmoid bucket
// R20 [dsv4-r20-dssel] (design: wt/dsv4-draft): drafts to verify, n* = argmax over n of
// 1 + S_1 + .. + S_n - lambda * T[n]; S_n = p_1 .. p_n, p_j = sigmoid(b + w_c conf + w_p (l1 - lse) + w_g (l1 - l2)) per
// position (weights fit offline on the recorded Maya-S passes); lambda = tokens / ms (EWMA), T[n] = ms of a call
// verifying n drafts incl. its next draft block (EWMA). Exact: only the number of verified drafts changes
static int ds_len(Engine* E, int nmax) {
    static const float Wt[5][4] = {{0.7137f, 0.3009f, 2.1908f, 0.1584f}, {0.4570f, 0.3883f, 1.8158f, 0.1972f},
                                   {0.6434f, 0.4103f, 2.3066f, 0.1982f}, {0.1147f, 0.3327f, 1.2333f, 0.3065f},
                                   {0.3487f, 0.3967f, 1.0735f, 0.2570f}};
    const double lam = E->ds_lam > 0 ? E->ds_lam : 3.0 / E->ds_T[5];
    double S = 1, gain = 1, best = 1 - lam * E->ds_T[0];
    int nb = 0;
    for (int n = 1; n <= std::min(nmax, 5); n++) {
        const float* f = E->dstat + 3 * (n - 1);
        const float* w = Wt[n - 1];
        const double z = w[0] + w[1] * E->dconf[n - 1] + w[2] * (f[0] - f[2]) + w[3] * (f[0] - f[1]);
        S *= 1.0 / (1.0 + exp(-z));
        gain += S;
        const double v = gain - lam * E->ds_T[n];
        if (v > best) { best = v; nb = n; }
    }
    return nb;
}
static void ds_len_upd(Engine* E, int n, int tok, double ms) {   // after a call that verified n drafts
    if (!g_dssel || n < 0 || n > 5 || ms <= 0) return;
    E->ds_T[n] = 0.75 * E->ds_T[n] + 0.25 * ms;
    const double r = tok / ms;
    E->ds_lam = E->ds_lam > 0 ? 0.75 * E->ds_lam + 0.25 * r : r;
}
int decode_spec(Engine* E, int t, std::vector<int>& out, StepTimes* tm, int cap) {
    xcone_flush(E, "decode");   // [dsv4-xcall]
    kv_reserve(E, E->pos + MAXR);   // [dsv4-rel4-p2]
    if (!E->dspark) {   // DSpark off (not requested, or switched off at load for lack of VRAM): plain steps
        cap_wait(E);
        out.push_back(decode(E, t, tm));
        return 1;
    }
    if (g_dacc) E->o.dspark_adapt = 0;   // R22 acceptance gate: no plain runs, no lookup
    int lk[MAXR];
    int nlk = (E->lk_hold > 0 ? (E->lk_hold--, 0) : lookup_find(E, t, lk, MAXR - 1));   // (host only)
    if (g_dacc) nlk = 0;
    if (E->pos + nlk + 1 + 5 > E->o.maxctx) nlk = 0;
    if (E->cap_spec) {   // R19 [dsv4-r19-gahead]: the capture kicked in draft_block is kept only for a one-row graph step
        bool one = E->o.dspark_adapt && E->ds_plain && !nlk;
        if (!one && E->have_draft && !nlk) {   // the same gate as below: no confident draft = a one-row step
            const int n0 = std::max(1, std::min(5, E->o.dspark_n));
            float cp = 1.f;
            int k = 0;
            if (g_dssel) k = ds_len(E, n0);   // R20 (the same rule as below)
            else if (E->o.dspark_tau > 0)
                while (k < n0 && (cp *= 1.f / (1.f + expf(-E->dconf[k]))) >= E->o.dspark_tau) k++;
            else k = n0;
            one = k == 0;
        }
        if (!one) E->cap_cancel.store(true, std::memory_order_release);
        E->cap_spec = false;
    }
    cap_wait(E);
    if (E->o.dspark_adapt && E->ds_plain && !nlk) {   // R8: plain steps (measured), the drafter's window kept current
        auto ta = clk::now();
        out.push_back(decode(E, t, tm));
        ds_after_plain(E);
        draft_window(E);
        E->have_draft = false;
        double ms = ms_since(ta);
        E->ds_ms_plain = E->ds_ms_plain > 0 ? 0.75 * E->ds_ms_plain + 0.25 * ms : ms;
        E->ds_prior = E->ds_ms_plain;
        // first 8 steps measure the plain rate; after a loss, 32 plain tokens before the next probe
        if (E->ds_probe && E->ds_cnt + 1 >= 8) {   // R19: the measuring steps after a spec start: judge the passes now
            E->ds_probe = false;
            if (E->ds_ms_spec <= E->ds_ms_plain) {   // passes win: back to them
                E->ds_plain = false;
                E->ds_cnt = 0;
                E->ds_ms_spec = 0;
                return 1;
            }
            E->ds_ms_spec = 1;   // lost: the plain run continues to 32 tokens
        }
        if (++E->ds_cnt >= (E->ds_ms_spec > 0 ? 32 : 8)) {
            E->ds_plain = false;
            E->ds_cnt = 0;
            E->ds_ms_spec = 0;
        }
        return 1;
    }
    mem_guard(E, false);
    DCHECK(t >= 0 && t < V, "token id %d", t);
    ns_tick();
    NvR nv_pass("spec pass");
    int nd = std::max(1, std::min(5, E->o.dspark_n));
    if (E->pos + nd + 1 + 5 > E->o.maxctx) {   // no room for a block: plain steps (the draft window goes stale)
        out.push_back(decode_i(E, t, tm, true));   // R19 (review r18 N2): mem_guard / ns_tick ran above
        E->have_draft = false;
        return 1;
    }
    auto t0 = clk::now();
    const bool plain_mode = E->o.dspark_adapt && E->ds_plain;   // here only with a lookup match
    float pc[8] = {};
    if (!plain_mode) {
        if (!E->have_draft) {
            draft_window(E);
            draft_block(E, E->pos, t);
        }
        for (int j = 0; j < nd; j++) pc[j] = 1.f / (1.f + expf(-E->dconf[j]));
        if (g_dssel) nd = ds_len(E, nd);   // R20 [dsv4-r20-dssel]
        else if (E->o.dspark_tau > 0) {   // confidence-gated block length
            float cp = 1.f;
            int k = 0;
            while (k < nd && (cp *= pc[k]) >= E->o.dspark_tau) k++;
            nd = k;
        }
        if (g_dacc) nd = 0;   // R22: one-row steps, a full block at every position
    } else {
        nd = 0;
        if (E->mh_n) draft_window(E);
    }
    // R17: the lookup's drafts instead of DSpark's when they are longer and agree on the first token
    const bool use_lk = nlk > nd && (nd == 0 || lk[0] == E->drafts[0]);
    const int* dd = use_lk ? lk : E->drafts;
    if (use_lk) nd = nlk;
    if (cap > 0) nd = std::min(nd, cap - 1);   // [dsv4-dec3-cap] (review dec3 L1): a shorter block, honest statistics
    if (nd == 0 && E->graph) {   // R17: nothing to verify: a one-row step through the step graph
        const int r = decode_i(E, t, tm, true);   // (R18 L2: mem_guard / ns_tick already done above)
        out.push_back(r);
        ds_after_plain(E);
        auto t3 = clk::now();
        draft_window(E);
        draft_block(E, E->pos, r);
        if (tm) tm->draft += ms_since(t3), tm->total += ms_since(t3);
        ds_len_upd(E, 0, 1, ms_since(t0));
        if (E->o.dspark_adapt) {   // R18 (review r17 L1): measured as a pass of one token (as the eager R = 1 pass was)
            const double mt = ms_since(t0);
            E->ds_ms_spec = E->ds_ms_spec > 0 ? 0.75 * E->ds_ms_spec + 0.25 * mt : mt;
            E->ds_wms += mt;
            E->ds_wtok += 1;
            if (++E->ds_cnt >= 8) ds_judge(E);
        }
        return 1;
    }
    E->tm = tm;
    const int R = nd + 1, s = E->pos;
    E->toks.push_back(t);
    for (int j = 0; j < nd; j++) E->toks.push_back(dd[j]);
    E->seq++;
    if (E->o.dyn) {
        E->spec_ids.resize(NL);
        E->spec_defer = true;
    }
    if (E_dprof && !g_pe[0][0])
        for (int l = 0; l < NL; l++) {
            DevGuard dg(E->cards[E->L[l].card].dev);
            for (int k = 0; k < NPP; k++) DCK(cudaEventCreate(&g_pe[l][k]));
        }
    g_pon = E_dprof && (E->pass_clock % 2 == 0);   // R20: per-phase GPU profile of verify passes (owner streams)
    forward(E, s, R);
    if (g_pon) {
        g_pon = false;
        for (int l = 0; l < NL; l++) {
            DevGuard dg(E->cards[E->L[l].card].dev);
            if (cudaEventSynchronize(g_pe[l][NPP - 1]) != cudaSuccess) { cudaGetLastError(); continue; }
            for (int k = 1; k < NPP; k++) {
                float ms = 0.f;
                if (cudaEventElapsedTime(&ms, g_pe[l][k - 1], g_pe[l][k]) != cudaSuccess) { cudaGetLastError(); continue; }
                g_paccR[R][k] += ms;
            }
        }
        g_pnR[R]++;
    }
    E->spec_defer = false;
    auto t1 = clk::now();
    int g[MAXR];
    int a = 0;
    if (E->samp_on) {   // [dsv4-rel-sample]: exact speculative sampling (deterministic drafts), EOS ends the run
        head_rows(E, 0, R, nullptr, nullptr, nullptr);   // logits of rows 0..nd on the last card
        Card& lc = E->cards.back();
        int in[MAXR + 1];
        in[0] = t;
        for (int j = 0; j < nd; j++) in[j + 1] = dd[j];
        samp_accept(E->samp, lc.d_logits, V, in, nd, s, EOS, -1, E->h_emit, lc.st);
        DCK(cudaStreamSynchronize(lc.st));
        a = E->h_emit[0] - 1;
        DCHECK(a >= 0 && a <= nd, "sampled acceptance %d of %d", a, nd);
        for (int j = 0; j <= a; j++) g[j] = E->h_emit[1 + j];
    } else {
    head_rows(E, 0, R, g, nullptr, nullptr);
    while (a < nd && dd[a] == g[a]) a++;
    }
    auto t2 = clk::now();
    if (use_lk) {
        g_lk[0]++; g_lk[1] += nd; g_lk[2] += a;
        if (a == 0) E->lk_hold = 8;   // a miss: no lookup for the next 8 calls
    }
    for (int j = 0; j < nd && j <= a && !use_lk; j++) {   // calibration: draft j tested (all before it accepted)
        int b = std::min(9, (int)(pc[j] * 10));
        g_conf[b][0]++;
        g_conf[b][1] += a > j;
    }
    for (int j = 0; j < a; j++)   // review r6 H1: stop at the first EOS (nothing after it is output or kept)
        if (g[j] == EOS) { a = j; break; }
    for (int j = 0; j <= a; j++) out.push_back(g[j]);
    if (E->o.dyn) {   // rows 0..a are real positions; the rejected rows never feed the tier
        const auto tcm = clk::now();
        spec_commit(E, a + 1);
        if (tm) tm->commit += ms_since(tcm);   // advisor H1: the tier's host work after the head, timed directly
    }
    if (g_fut) {
        int rt[MAXR];
        rt[0] = t;
        for (int j = 0; j < nd; j++) rt[j + 1] = dd[j];
        fut_commit(E, R, a + 1, rt, g, use_lk);
    }
    rollback(E, s + a + 1);
    E->sv_pass = E->pos;   // [dsv4-srv-stop] sv holds this pass's compressor input rows
    for (int si = 0; si < 4; si++) E->sv_pb[si] = E->sv_base[si], E->sv_pm[si] = E->sv_m[si];
    E->dyn_clock += a + 1;
    E->pass_clock += 1;
    // next block: the drafter's window gets the accepted positions s .. s + a, then drafts from g[a] at s + a + 1
    E->mh_pos0 = s;
    E->mh_n = a + 1;
    auto t3 = clk::now();
    draft_window(E);
    if (plain_mode) E->have_draft = false;   // R17: a lookup pass in the plain mode: the window only
    else {
        draft_block(E, s + a + 1, g[a]);
        eg_prefetch_ahead(E, g[a], E->drafts, 5);   // R20 [dsv4-r20-egahead]
    }
    if (tm) {
        tm->head += ms_since(t1) - ms_since(t2);
        tm->draft += ms_since(t3);
        tm->total += ms_since(t0);
        tm->npass += 1;
        tm->nacc += a;
        tm->rows = R;
        tm->plain = use_lk ? 2 : 0;
    }
    E->tm = nullptr;
    if (!use_lk && !plain_mode) ds_len_upd(E, nd, a + 1, ms_since(t0));
    if (E->o.dspark_adapt && !plain_mode) {   // R8: per-token cost of this pass (incl. its draft) vs the plain steps' running cost
        double mt = ms_since(t0) / (a + 1);
        E->ds_ms_spec = E->ds_ms_spec > 0 ? 0.75 * E->ds_ms_spec + 0.25 * mt : mt;
        E->ds_wms += ms_since(t0);
        E->ds_wtok += a + 1;
        if (++E->ds_cnt >= 8) ds_judge(E);
    }
    return a + 1;
}

void forward_rows(Engine* E, const int* t, int n, float* out) {
    xcone_flush(E, "rows");   // [dsv4-xcall]
    DCHECK(n >= 1 && n <= MAXR && E->pos + n <= E->o.maxctx, "forward_rows n %d", n);
    kv_reserve(E, E->pos + n);   // [dsv4-rel4-p2]
    E->sv_pass = -1;   // [dsv4-srv-stop]
    ns_tick();   // R20: DSV4_NSYS windows count forward_rows calls too (tfcmp passes at a fixed R)
    NvR nv_rows("rows pass");
    for (int i = 0; i < n; i++) E->toks.push_back(t[i]);
    if (n > 1 || !E->runahead) E->seq++;
    forward(E, E->pos, n);
    std::vector<int> am(n);
    head_rows(E, 0, n, am.data(), nullptr, nullptr);
    Card& c = E->cards.back();
    DCK(cudaMemcpy(out, c.d_logits, (size_t)n * V * 4, cudaMemcpyDeviceToHost));
    E->pos += n;
    E->dyn_clock += n;
    E->pass_clock += 1;
}
static void hspin(Engine* E, const volatile int* f, int v, const char* what) {
    if (*f == v) return;
    auto t0 = clk::now();
    for (unsigned it = 1;; it++) {
        if (*f == v) return;
        _mm_pause();
        if ((it & 8191) == 0) {
            double ms = ms_since(t0);
            if (ms < 1000) continue;
            static thread_local bool said = false;
            if (ms > 5000 && !said) {
                said = true;
                logf_("WARNING still waiting for %s after 5 s (flag %d, want %d, layer %d)\n", what, *f, v, g_wl);
            }
            for (int k = 0; k < 3; k++) {   // never recorded: success unless that card's context is broken
                if (!E->ev_probe[k]) continue;
                cudaError_t er = cudaEventQuery(E->ev_probe[k]);
                DCHECK(er == cudaSuccess || er == cudaErrorNotReady, "waiting for %s: CUDA error %s (card %d)", what,
                       cudaGetErrorString(er), k);
            }
            DCHECK(ms < 120000, "waiting for %s: no answer for 120 s (flag %d, want %d)", what, *f, v);
        }
    }
}
static void opt_changed(Engine* E);
void set_fuse(Engine* E, int on) { opt_changed(E); E_fuse_off = !on; }
void set_sampling(Engine* E, float temperature, float top_p, int top_k, float min_p, unsigned long long seed) {
    cap_wait(E);
    Card& c = E->cards.back();
    DevGuard dg(c.dev);
    if (!E->samp) {
        E->samp = dmalloc(samp_bytes());
        DCK(cudaHostAlloc((void**)&E->h_emit, 64, cudaHostAllocPortable));
    }
    E->samp_on = temperature > 0.f;
    // the step graph's head is argmax only: off while sampling, back on for greedy (review rel M2: it stayed off for the
    // rest of the process; a held graph is invalidated through G.seq / nfwd by the eager steps in between)
    const bool g = E->graph_cfg && !E->samp_on;
    if (g != E->graph) logf_("sampling: step graph %s\n", g ? "on" : "off");
    E->graph = g;
    samp_set(E->samp, temperature > 0.f ? temperature : 1.f, top_p, top_k, min_p, seed, c.st);
    DCK(cudaStreamSynchronize(c.st));
}
void set_qrb(Engine* E, int on) {   // R15: in-process A/B
    if (E->gm && on) {   // review rel L4: GGUF mode keeps qrb off (the Q8_0 linq path does not take the bf16 qr)
        logf_("set_qrb: qrb stays off in GGUF mode\n");
        return;
    }
    opt_changed(E);
    E_qrb_off = !on;
}
void set_dspark(Engine* E, int n, float tau) {
    if (n >= 0) E->o.dspark_n = n;
    if (tau >= 0) E->o.dspark_tau = tau;
}

// R18 (review r17 L2): from_spec = called by decode_spec after its own mem_guard / ns_tick (counted once)
static int decode_i(Engine* E, int t, StepTimes* tm, bool from_spec) {
    kv_reserve(E, E->pos + 1);   // [dsv4-rel4-p2]
    DCHECK(E->pos + 1 <= E->o.maxctx, "context %d reached maxctx %d", E->pos + 1, E->o.maxctx);   // review H1
    E->sv_pass = -1;   // [dsv4-srv-stop] a one-row step (graph captures may have set sv ahead)
    DCHECK(t >= 0 && t < V, "token id %d", t);
    if (!from_spec) mem_guard(E, false);
    auto t0 = clk::now();
    E->tm = tm;
    E->toks.push_back(t);
    if (!from_spec) ns_tick();
    NvR nv_step("decode step");
    if (!E->runahead && !E->graph) E->seq++;   // (run-ahead increments it in forward, the graph path in the capture)
    if (E_dprof && !g_pe[0][0]) {
        for (int l = 0; l < NL; l++) {
            DCK(cudaSetDevice(E->cards[E->L[l].card].dev));
            for (int k = 0; k < NPP; k++) DCK(cudaEventCreate(&g_pe[l][k]));
        }
    }
    int out = -1;
    auto t1 = clk::now();
    if (E->graph) {   // R10: the whole step as one graph (head inside)
        out = decode_graph(E, t);
        t1 = clk::now();
    } else {
    g_pon = E_dprof && (E->seq % 8 == 0);
    forward(E, E->pos, 1);
    t1 = clk::now();
    head_rows(E, 0, 1, &out, nullptr, nullptr);
    }
    if (g_pon) {
        for (int l = 0; l < NL; l++) {
            DCK(cudaSetDevice(E->cards[E->L[l].card].dev));
            cudaError_t er = cudaEventSynchronize(g_pe[l][NPP - 1]);
            if (er != cudaSuccess) { logf_("dprof: layer %d sync: %s\n", l, cudaGetErrorString(er)); cudaGetLastError(); continue; }
            for (int k = 1; k < NPP; k++) {
                float ms = 0.f;
                er = cudaEventElapsedTime(&ms, g_pe[l][k - 1], g_pe[l][k]);
                if (er != cudaSuccess) { logf_("dprof: layer %d point %d: %s\n", l, k, cudaGetErrorString(er)); cudaGetLastError(); continue; }
                g_pacc[k] += ms;
            }
        }
        DCK(cudaSetDevice(E->cards.back().dev));
        g_psteps++;
        g_pon = false;
    }
    if (g_fut) fut_commit(E, 1, 1, &t, &out, 0);
    E->pos += 1;
    E->dyn_clock += 1;
    E->pass_clock += 1;
    if (tm) {
        tm->head += ms_since(t1);
        tm->total += ms_since(t0);
        tm->rows = 1;
        tm->plain = 1;
    }
    E->tm = nullptr;
    if (E->dump_step >= 0) E->dump_step++;
    return out;
}
int decode(Engine* E, int t, StepTimes* tm) {
    xcone_flush(E, "decode");   // [dsv4-xcall]
    return decode_i(E, t, tm, false);
}

// review r14 M3: one decode() step (graph path by default: capture ahead, boundary / head-window applies, Engram
// staging, mem_guard) and the logits it produced; waits for the next step's capture first (it is waited for at the next
// step's start anyway), then copies on the stream the head ran on
int decode_logits(Engine* E, int t, float* out) {
    xcone_flush(E, "decode");   // [dsv4-xcall]
    const int r = decode(E, t, nullptr);
    cap_wait(E);
    const int NC = (int)E->cards.size(), last = NC - 1;
    if (E->graph && E->hsplit) {   // the graph's head split: each card keeps its logits slice (only the argmax travels)
        for (int k = 0; k < NC; k++) {
            Card& c = E->cards[k];
            DevGuard dg(c.dev);
            const int r0 = E->hr[k], r1 = k == last ? V : E->hr[k + 1];
            DCK(cudaMemcpyAsync(out + r0, c.d_logits + r0, (size_t)(r1 - r0) * 4, cudaMemcpyDeviceToHost, c.est));
            DCK(cudaStreamSynchronize(c.est));   // (stream order: after this card's step graph)
        }
        return r;
    }
    Card& c = E->cards.back();
    DevGuard dg(c.dev);
    cudaStream_t s = E->graph ? c.est : c.st;
    DCK(cudaMemcpyAsync(out, c.d_logits, (size_t)V * 4, cudaMemcpyDeviceToHost, s));
    DCK(cudaStreamSynchronize(s));
    return r;
}
// [shared-srv] a plain step (no drafts) whose logits [V] come back, for a grammar-constrained reply: the token is
// chosen by the caller. The DSpark drafter's window stays current, as after decode_spec's plain steps (R8)
int decode_logits_plain(Engine* E, int t, float* out) {
    xcone_flush(E, "decode");   // [dsv4-xcall]
    kv_reserve(E, E->pos + MAXR);   // [dsv4-rel4-p2]
    if (E->cap_spec) {   // R19: the speculative capture of a draft block: no such step comes
        E->cap_cancel.store(true, std::memory_order_release);
        E->cap_spec = false;
    }
    cap_wait(E);
    const int r = decode_logits(E, t, out);
    if (E->dspark) {
        ds_after_plain(E);
        draft_window(E);
        E->have_draft = false;
    }
    return r;
}
void set_dump_step(Engine* E, int s) { E->dump_step = s; }
// review r10 M3: options baked into a step graph captured ahead: wait for the capture, then invalidate it
static void opt_changed(Engine* E) {
    cap_wait(E);
    E->nfwd++;
}
void set_prefetch(Engine* E, int n) { opt_changed(E); E->o.prefetch = n; }
void set_prefetch_mr(Engine* E, int on) { opt_changed(E); E->o.prefetch_mr = on; }
void set_pf_early(Engine* E, int n) { opt_changed(E); E->o.pf_early = n; }   // R12: in-process A/B (frozen tier)
void set_nv_cap(Engine* E, float cap) { E->o.nv_cap = cap; }
void set_car(Engine* E, float m) { opt_changed(E); E->o.car_margin = m; }
void set_gg_fp32(Engine* E, int v) { opt_changed(E); g_gg_fp32 = v != 0; }   // R21 (captured graphs bake the kernel)
void set_dyn_window(Engine* E, int w) { opt_changed(E); E->o.dyn_window = w; }   // R16 [dsv4-gguf-iq]
void set_drop(Engine* E, float w) { opt_changed(E); E->o.drop_w = w; }
void set_small_prefill(Engine* E, int n) { E->o.small_prefill = n; }
void set_prefill_abort(Engine* E, const int* flag) { E->pf_abort = flag; }
void set_pprof(Engine* E, int lvl) { (void)E; g_pprof = lvl; }   // [dsv4-pf-pprof]
double instr_ms(Engine* E) { (void)E; return g_instr_ms; }   // [dsv4-pf6-net]
void set_f32k(Engine* E, int on) { opt_changed(E); E_f32k_off = !on; }
void set_q1(Engine* E, int on) { opt_changed(E); E_q1_off = !on; }
void set_runahead(Engine* E, int on) { opt_changed(E); E->runahead = on != 0; }

void destroy(Engine* E) {
    cap_wait(E);
    gm_profile_save(E);   // [dsv4-rel-gguf]
    dacc_score(E);
    rec_dump();
    fut_dump();
    if (E->early && g_early_n[0])
        logf_("early admissions: %ld requested, %ld in time (%.1f %%), %ld late; routing already out at the request %ld, "
              "copy done when the host had the routing %ld\n", g_early_n[0], g_early_n[1], 100.0 * g_early_n[1] / g_early_n[0],
              g_early_n[2], g_early_n[3], g_early_n[4]);
    if (E->g_upd + E->g_inst) logf_("step graph: %ld updates, %ld instantiations, %ld captures ahead of a draft, %ld cancelled\n",
                                     E->g_upd, E->g_inst, E->g_ahead, E->g_cancel);
    if (g_swapchk) logf_("SWAPCHK: %ld checks, %ld mismatches\n", g_chk_n, g_chk_bad);
    if (E->swap) logf_("swap tier: %ld swaps (victims without a host copy), %ld host buffers recycled, %zu free homes\n",
                       E->nswap, E->nrecyc, E->hfree.size());
    if (E->gswap) {
        size_t nh = 0;
        for (auto& [sz, v] : E->ghome) nh += v.size();
        logf_("exclusive swap: %ld swaps, %ld copies recycled into homes, %zu free homes\n", E->ngswap, E->ngrecyc, nh);
    }
    if (g_gbn) logf_("step graph host ms/step: boundary %.3f, launch %.3f, engram staging %.3f, lane loop %.2f, head wait %.3f\n",
                     g_gb[0] / g_gbn, g_gb[1] / g_gbn, g_gb[2] / g_gbn, g_gb[3] / g_gbn, g_gb[4] / g_gbn);
    if (E->enq_th.joinable()) {
        E->enq_quit = true;
        E->enq_go.fetch_add(1);
        E->enq_go.notify_one();
        E->enq_th.join();
    }
    if (g_psteps) {
        double tot = 0;
        for (int k = 1; k < NPP; k++) tot += g_pacc[k];
        logf_("decode GPU profile over %d steps (ms per step, summed over 40 layers; total %.2f):\n", g_psteps, tot / g_psteps);
        for (int k = 1; k < NPP; k++) logf_("  %-28s %6.2f\n", PPN[k], g_pacc[k] / g_psteps);
    }
    for (int r = 1; r <= MAXR; r++) {
        if (!g_pnR[r]) continue;
        double tot = 0;
        for (int k = 1; k < NPP; k++) tot += g_paccR[r][k];
        logf_("verify-pass GPU profile R=%d over %d passes (ms per pass, 40 layers, owner streams; total %.2f):", r, g_pnR[r],
              tot / g_pnR[r]);
        for (int k = 1; k < NPP; k++) logf_(" %s %.2f;", PPN[k], g_paccR[r][k] / g_pnR[r]);
        logf_("\n");
    }
    if (g_dcalls) {
        logf_("drafter profile over %ld calls (window + block, card %d stream, ms per call): GPU %.3f, host enqueue %.3f\n",
              g_dcalls, E->cards.back().dev, g_dwall / g_dcalls, g_denq / g_dcalls);
        for (auto& [k, v] : g_dpacc) logf_("  %-30s %7.3f\n", k.c_str(), v / g_dcalls);
        logf_("  head GEMV share card 0 / 1 (own streams) %.3f / %.3f\n", g_dhacc[0] / g_dcalls, g_dhacc[1] / g_dcalls);
        if (g_dstages) logf_("  drafter MoE: %.2f distinct experts of %.2f row choices per stage\n", (double)g_ddist / g_dstages,
                             (double)g_dpairs / g_dstages);
    }
    if (E_hprof) {   // R23 [dsv4-r23-hgap]
        hp_read(g_hp[0]);
        hp_read(g_hp[1]);
        for (int kd = 0; kd < 2; kd++)
            for (int r = 1; r <= MAXR; r++)
                if (g_hpn[kd][r]) {
                    logf_("head profile %s R=%d over %ld calls (ms per card, GEMV only):", kd ? "drafter" : "target", r,
                          g_hpn[kd][r]);
                    for (int k = 0; k < (int)E->cards.size(); k++)   // (R23, review r23 L5: the cards there are)
                        logf_(" %s%.3f", k ? "/ " : "", g_hpacc[kd][r][k] / g_hpn[kd][r]);
                    logf_("\n");
                }
    }
    if (g_lk[0]) logf_("prompt lookup: %ld passes, %ld drafts verified, %ld accepted\n", g_lk[0], g_lk[1], g_lk[2]);
    if (g_conf[0][0] + g_conf[5][0] + g_conf[9][0]) {
        logf_("DSpark confidence calibration (sigmoid bucket: tested, accepted rate):");
        for (int b = 0; b < 10; b++) logf_(" %.1f: %ld %.2f", b / 10.0, g_conf[b][0], g_conf[b][0] ? (double)g_conf[b][1] / g_conf[b][0] : 0.0);
        logf_("\n");
    }
    log_mem("at exit");
    E->lane.stop();
    if (E->pfcpu.started()) E->pfcpu.stop();
    if (E->eg_nvme) E->egr.stop();
    for (int k = 0; k < 2; k++)   // review pf17b L1
        if (E->egr.fdm[k] >= 0) { close(E->egr.fdm[k]); E->egr.fdm[k] = -1; }
    if (E_mhist) {
        const char* nm[8] = {"<=8", "9-32", "33-112", "113-256", "257-448", "449-1024", "1025-2048", ">2048"};
        for (int b = 0; b < 8; b++)
            logf_("mhist %s: %lld experts, %lld rows\n", nm[b], g_mh_n[b].load(), g_mh_r[b].load());
        const char* fn[4] = {"IQ2_XXS", "IQ3_XXS", "IQ2_S", "IQ3_S"};
        for (int g = 0; g < 4; g++)
            logf_("mhist 113-448 w1 %s: %lld experts, %lld rows\n", fn[g], g_mh_f[g][0].load(), g_mh_f[g][1].load());
    }
    if (E->dyn_fp4 || E->gm) E->xr.stop();
    if (E->gm) {   // review rel L6: the Engram fds and the drafter mapping (after the readers stopped)
        for (int k = 0; k < 2; k++)
            if (E->gm->eg_fd[k] >= 0) close(E->gm->eg_fd[k]);
        if (E->gm->dsp) munmap((void*)E->gm->dsp, E->gm->dsp_size);
        delete E->gm;
        E->gm = nullptr;
    }
    for (auto& [id, r] : E->kvres) kvh_release(E, r.host);   // [dsv4-rel5-p3]
    kvh_trim(E, 0);
    delete E->gg;   // R16 [dsv4-gguf-iq] (null outside the GGUF mode)
    delete E;
}

}  // namespace dsv4
