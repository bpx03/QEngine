#pragma once
// GLM-5.3-Flash engine (libqglm) for 3 x RTX 3090: TP over the cards (KDA / MLA heads 22/21/21, expert and dense MLP
// intermediate columns in 128-wide groups 6/5/5), mHC residual 4 x 4096 replicated in fp32, routed experts INT4 g128
// (Intel AutoRound, auto_gptq packing) in a dynamic per-card VRAM cache (each card caches its column slice of the
// same experts) backed by the full set in pinned host RAM; cold experts of a decode step are computed by the CPU
// lane (AVX-512 VNNI) in parallel with the GPUs. Reference math: tools/glm/glm_model.py. Log:.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <tuple>
#include <functional>
#include <map>
#include <mutex>
#include <random>
#include <memory>
#include <string>
#include <thread>
#include <array>
#include <vector>

#include "../common.h"
#include "../kernels/comm.cuh"

namespace glm {

// [glm-idle] spin-then-block wait on a generation counter (idea from QFN [idle-cpu]): a waiter pause-spins up to
// spin_us after its last job (decode jobs come back to back), then sleeps on the counter (futex) counted in `sleepers`.
// Every change of the counter goes through bump() / set(), which wakes sleepers. Both sides are seq_cst (store-load):
// either the waiter sees the new value before it sleeps, or the bumper sees its count, so no wake-up is lost.
// spin_us < 0: never sleeps (the old busy wait). Bodies in glm_cpu.cpp (host C++).
struct IdleGate {
    alignas(64) std::atomic<int> sleepers{0};
    // returns the counter's value once it differs from `seen`
    unsigned wait(const std::atomic<unsigned>& v, unsigned seen, long spin_us);
    void bump(std::atomic<unsigned>& v);              // v += 1, wakes sleepers
    void set(std::atomic<unsigned>& v, unsigned x);   // v = x, wakes sleepers
};
// GLM_LANE_SPIN_US / GLM_COORD_SPIN_US / GLM_POOL_SPIN_US (default 20,000 us: about one decode step; the step boundary,
// the largest gap between jobs inside a decode, is 2.5-4.8 ms), GLM_LANE_PF_SPIN_US (1 s, after a prefill-lane job);
// -1 = spin forever; a value that is set is logged
long idle_spin_us(const char* env, long def = 20000);

constexpr int H = 4096, HC = 4, RW = HC * H, NL = 45, MTP_L = 45;
constexpr int NE = 288, TOPK = 8, INTER = 2048, DINTER = 12288, VOCAB = 154880;
constexpr int KH = 64, KD = 128, CONV = 4;                  // KDA
constexpr int NH = 64, QLR = 1536, KVLR = 512, DN = 256, DV = 256;  // MLA (NoPE)
constexpr int IH = 32, ID = 128, POOL = 4, ITOPK = 2048, TOPB = ITOPK / POOL;  // DSA indexer
constexpr int MAXSEL = TOPB * POOL + POOL - 1;              // selected tokens of one query at most (2051)
constexpr float EPS = 1e-5f, HC_EPS = 1e-6f, RSF = 2.5f, CLAMP = 10.f, GLB = -5.f, LN_EPS = 1e-6f;
constexpr int SINK = 20, NMIX = 24, MIXS = 32;  // mix row stride (index 24: sum of squares of the streams)
constexpr int EOS0 = 154820, EOS1 = 154827, EOS2 = 154829;
constexpr int MLA_IN = QLR + KVLR + ID + IH + ID;           // fused x-GEMM rows of an MLA layer (2336)
constexpr int MAXR = 16;                                    // rows of the INT8 GEMV path (dense weights)
constexpr int KSPL = 8;                                     // K splits of the expert gate/up GEMV

inline bool is_mla(int l) { return l == MTP_L || l % 4 == 3; }
inline bool is_dense(int l) { return l < 3; }

// one card's share
struct Plan {
    int rank = 0, ngpu = 3;
    int h0 = 0, h1 = 0;      // KDA and MLA heads [h0, h1)
    int e0 = 0, e1 = 0;      // routed expert intermediate columns (multiples of 64: uneven split by free VRAM)
    int s0 = 0, s1 = 0;      // shared expert intermediate columns (128-groups: its INT8 down needs K in 128-groups)
    int d0 = 0, d1 = 0;      // dense MLP intermediate columns
    int v0 = 0, v1 = 0;      // lm_head rows
    int nh() const { return h1 - h0; }
    int ne() const { return e1 - e0; }
    int ns() const { return s1 - s0; }
};
// ecols: routed expert columns per card (multiples of 64, sum INTER) or nullptr (GLM_ECOLS / the default)
std::vector<Plan> make_plan(int ngpu, const int* ecols = nullptr);
// release track R6 ([glm-rel-host]): host threads placed from the machine's topology (the process's allowed CPUs, one
// thread per physical core first): coordinator on the first core, the caller on its SMT sibling, the card-pool threads on
// the last core (+ sibling), the CPU lane on the cores between. On the 24-core box: 0 / 24 / 23,47 / 1-22 (the old fixed
// values). why = one log line
struct HostCpus {
    int coord = 0, caller = -1;
    std::vector<int> pool, lane;
    std::string why;
};
HostCpus host_cpus(int npool);

// byte layout of one expert slice on a card (own intermediate columns [e0, e1), ni = e1 - e0):
// gq [512][ni] i32 | uq [512][ni] i32 | gs [32][ni] f16 | us [32][ni] f16 | dq [ni/8][4096] i32 | ds [ni/64][4096] f16
// (down scales per 64 rows: each 128-group's scale twice, so a slice may start at any multiple of 64)
struct SliceL {
    int ni = 0;
    size_t gq = 0, uq = 0, gs = 0, us = 0, dq = 0, ds = 0, bytes = 0;
    // gu3 (host slices only, round 6): gate / up codes INT3 (calibrated checkpoints, levels -4..3, zero 4), tile-major
    // [ni/32][128][3][32] i32 per 32 K-rows: w0 / w1 = 2-bit codes of k8 rows 4kb+{0,1} / 4kb+{2,3}, w2 = high bits
    // (byte b, bit 2(t & 1) + p of w0/w1 and bit 2t + p of w2 = row 8(4kb + t) + 2b + p); GPU slots stay INT4
    bool gu3 = false;
    void set(int n, bool g3 = false);
};
// host copy of one expert (original AutoRound layout, contiguous):
// gq [512][2048] | gs [32][2048] f16 | uq | us | dq [256][4096] | ds [16][4096] f16
constexpr size_t HX_GQ = (size_t)512 * 2048 * 4, HX_GS = (size_t)32 * 2048 * 2;
constexpr size_t HX_DQ = (size_t)256 * 4096 * 4, HX_DS = (size_t)16 * 4096 * 2;
constexpr size_t HX_BYTES = 2 * (HX_GQ + HX_GS) + HX_DQ + HX_DS;
constexpr size_t HX_OFF_GQ = 0, HX_OFF_GS = HX_GQ, HX_OFF_UQ = HX_GQ + HX_GS, HX_OFF_US = 2 * HX_GQ + HX_GS,
                 HX_OFF_DQ = 2 * (HX_GQ + HX_GS), HX_OFF_DS = HX_OFF_DQ + HX_DQ;

// INT8 weight, symmetric per 128-group of a row: W[n][k] = q[n][k] * s[n][k / 128]
struct W8 {
    uint8_t* q4 = nullptr;  // round 9: INT4 storage (two codes per byte, low nibble = even k; GPTQ codes, scales in s)
    int8_t* q = nullptr;
    float* s = nullptr;
    int N = 0, K = 0;
};

// pinned host copy of one expert, repacked per card: [slice of rank 0][slice of rank 1][slice of rank 2], each slice in
// that card's SliceL layout, so a card's slice is one contiguous block (admission, staging and seeding are single copies)
struct HxLayout {
    int ns = 0;
    size_t stride = 0;
    size_t off[4] = {};
    SliceL sl[4];
    int e0[4] = {};
    // mixed mode (round 6, calibrated tiers: hot experts INT4, cold experts INT3 gate/up): sl = INT4 slices, sl3 = gu3
    // slices at the same offsets, one flag byte per expert at flag_off (1 = gu3)
    bool mixed = false;
    SliceL sl3[4];
    size_t flag_off = 0;
    void set(const std::vector<Plan>& p, bool gu3 = false, bool mix = false);
    const char* slice(const char* expert, int r) const { return expert + off[r]; }
    const SliceL& lay(const char* expert, int r) const { return mixed && expert[flag_off] ? sl3[r] : sl[r]; }
};
// pack one expert from the checkpoint tensors (AutoRound layout) into dst (host code, glm_cpu.cpp)
void hx_pack_expert(const HxLayout& L, char* dst, const void* gq, const void* gs, const void* uq, const void* us,
                    const void* dq, const void* ds, bool g3 = false);
// gate / up codes of an expert within the INT3 range (nibbles 4..11)?
bool hx_fits_int3(const void* gq, const void* uq);

constexpr int JR = 4;   // rows of an expert GEMV job (decode / verify step rows; generic small-T path)
// one GEMV job of the expert kernels: an expert slice (device or host-mapped) and its rows
// pointers of an expert slice: device slot (ld = ni, c0 = r0 = 0) or host copy (ld = 2048, c0 = r0 = e0)
struct XPtr {
    const int *gq = nullptr, *uq = nullptr, *dq = nullptr;
    const __half *gs = nullptr, *us = nullptr, *ds = nullptr;
    int ld = 0, c0 = 0, r0 = 0;
    bool host = false;
    bool g3 = false;      // gate / up as INT3 planes [128][3][ld] i32 (SliceL::gu3 row-major analog; decode GEMV only)
};
struct GuJob {            // prefill: gate/up job of one resident expert (rows at off of the layer's row lists)
    XPtr w;
    int off = 0, n = 0;
};
struct XJob {
    XPtr x;
    int nr = 0;               // rows of this job
    int r[JR];                // row indices into x
    float w[JR];              // gate weights
};

// sampling candidates of one verify row on one card (host-mapped): top-TKC of its vocab slice + max / sum exp
constexpr int TKC = 64;
struct TopK {
    float v[TKC];
    int i[TKC];
    float mx, se;
};

// decode-step handshake (host-mapped, one per MoE layer); a step has m <= MR rows (verify rows: last token + drafts)
constexpr int MR = 4, MXE = TOPK * MR;  // rows per decode step, distinct experts of a step at most
struct alignas(64) MoeReq {   // GPU rank 0 -> host
    unsigned seq;
    int m, ncold;
    int ids[MR][TOPK];        // routing of every row (policy counts)
    int hot[MR][TOPK];
    int ce[MXE], cnr[MXE], crow[MXE][MR];  // cold experts and their rows / weights
    float cw[MXE][MR];
    int pids[MR][TOPK];       // predicted top-8 of the next MoE layer (router(l + 1) on this layer's MoE input)
    float pw[MR][TOPK];       // their predicted gate weights (round 9: prefetch order / tau filter)
    int lids[MR][TOPK];       // round 9: late prediction (router(l + 1) on the streams after layer l, block-1 mHC of l + 1)
    float lw[MR][TOPK];
    unsigned lseq;
    unsigned pseq;
    unsigned eseq;            // round 13 test (GLM_PF_EARLY): raised when rank 0 enters the MoE block (prefetch stops)
    alignas(64) float x[MR][H];
};
struct alignas(64) MoeDec {   // host -> all GPUs
    unsigned seq;
    int l, nadm;
    int nbgd = 0;             // round 11: deferred background admissions after them (victims only here; GLM_BG_BOUND)
    int nbg;                  // round 8: background admissions at [nadm, nadm + nbg): copied off the critical path, the
                              // lane computes their rows this step (GLM_ADMIT_RULE=2)
    int e[MXE], slot[MXE], vl[MXE], ve[MXE], nr[MXE], row[MXE][MR];
    float w[MXE][MR];
    int nzc;                  // PCIe share: cold experts computed by the GPUs zero-copy from the host slices
    int ze[MXE], znr[MXE], zrow[MXE][MR];
    float zw[MXE][MR];
};
struct BgList;  // glm_step.cu
struct BgQ {   // round 11: background admissions deferred to the step boundary (host -> GPUs, mapped)
    static constexpr int CAP = 256;
    int n;
    int l[CAP], e[CAP], slot[CAP];
};
struct alignas(64) MoeOut {   // host CPU lane -> GPU rank 0
    unsigned seq;
    alignas(64) float y[MR][H];
};
std::string glm_data_file(const char* name);  // round 21: <lib root>/tools/glm/<name> (GLM_SRC_DIR fallback)
struct CpuLane;
// fp32 P2P all-reduce of decode rows (glm_ar.cu): receive slots and flags of every rank (peer pointers)
struct P2pAr {
    float* rbuf[4] = {};
    unsigned* flag[4] = {};
    int maxn = 0;
};
void ar_p2p(const float* part, float* sum, int n, const P2pAr& a, int rank, int ngpu, const unsigned* step, int idx,
            cudaStream_t st);
P2pAr ar_p2p_alloc(int dev, int ngpu, int maxn);
void ar_ll12(const float* part, float* sum, int n, const P2pAr& a, int rank, int ngpu, const unsigned* step, int idx,
             cudaStream_t st);
void ar_ll(const float* part, float* sum, int n, const P2pAr& a, int rank, int ngpu, const unsigned* step, int idx,
           cudaStream_t st);

struct Layer {
    bool mla = false, dense = false;
    float *hc_fn[2] = {}, *hc_base[2] = {}, *hc_scale[2] = {};  // [24][16384], [24], [3] (attn, ffn)
    float *in_ln = nullptr, *post_ln = nullptr;
    // KDA
    W8 kw_in;                // [3 nh 128 + 128 (f_a) + 128 (g_a) + nh (b)][4096]
    float *kf_b = nullptr, *kg_b = nullptr;  // [nh 128][128]
    float* kconv = nullptr;  // [3 nh 128][4]
    float *kA = nullptr, *kdt = nullptr, *ko_norm = nullptr;  // [nh], [nh 128], [128]
    W8 ko;                   // [4096][nh 128]
    float* S = nullptr;      // state [nh][128 v][128 k]
    float* conv_hist = nullptr;  // [3][3 nh 128]
    float* snap = nullptr;       // decode verify: state after each row [MR][nh][128][128]
    float* conv_rows = nullptr;  // decode verify: conv inputs of the rows [MR][3 nh 128]
    // MLA
    W8 mw_in;                // [MLA_IN][4096]
    float *q_a_ln = nullptr, *kv_a_ln = nullptr, *ik_w = nullptr, *ik_b = nullptr, *ape = nullptr;
    W8 mw_q;                 // [nh 256 + 4096][1536] (q_b own heads, indexer wq_b)
    float *wuk = nullptr, *wuv = nullptr;  // [nh][256][512]
    bf16 *wuk_b = nullptr, *wuv_b = nullptr;  // round 8: bf16 copies for the decode up projections (GLM_MLA_UBF)
    W8 mo;                   // [4096][nh 256]
    bf16* kc = nullptr;      // latent cache [ctx][512]
    float *ki = nullptr, *gsc = nullptr, *pk = nullptr;  // index keys / gate scores: ring [kring][128]; pooled keys [ctx/4][128]
    // MLP
    W8 dw_gu, dw_d;          // dense: [2 nd][4096], [4096][nd]
    float* router = nullptr; // [288][4096] fp32
    float* rbias = nullptr;  // [288]
    W8 sw_gu, sw_d;          // shared expert slice: [2 ni][4096], [4096][ni]
    int* slot = nullptr;     // device [288]: cache slot or -1 (inside Worker::slots_all)
};

// MTP layer extras
struct MtpW {
    W8 eh;                   // [4096][8192]
    float *enorm = nullptr, *hnorm = nullptr, *head_norm = nullptr;
};

// activation buffers of one row capacity. Two sets per card: the decode set (MR rows, own allocations: the decode
// graphs capture these pointers) and the prefill set (chunk rows), placed in the tail of the expert-cache slab only
// while a prefill runs (those slots are lent and re-seeded afterwards)
struct Acts {
    int mx = 0;
    int mxa = 0;               // rows of the attention scratch (big, bigb, d_sel, psel): prefill part rows
    float *R = nullptr;        // streams [mx][4][4096]
    float *mix = nullptr, *post = nullptr, *comb = nullptr;  // [mx][24], [mx][4], [mx][16]
    float* xf = nullptr;       // normed block input fp32 [mx][4096]
    bf16* xb = nullptr;        // same bf16
    float *part = nullptr, *sum = nullptr;  // [mx][4096]
    float* mixp = nullptr;     // decode: merge's partial mHC mixes of the next block [mx][16][25]
    float* ssp = nullptr;      // prefill: merge's partial sums of squares of the new streams [mx][16]
    float* big = nullptr;      // scratch fp32
    size_t big_n = 0;
    bf16* bigb = nullptr;      // scratch bf16
    size_t bigb_n = 0;
    float* logits = nullptr;   // [min(mx, 256)][v1 - v0]
    int* d_ids = nullptr;      // [mx][8]
    float* d_w = nullptr;      // [mx][8]
    int* d_tok = nullptr;      // [mx]
    int* d_sel = nullptr;      // MLA selection [mx][MAXSEL] (+ count)
    int* d_nsel = nullptr;
    XJob* d_jobs = nullptr;    // [jobs]
    int max_jobs = 0;
    float* gu = nullptr;       // expert gate/up partials [jobs][KSPL][MAXR][2 ni]
    float *xhb = nullptr, *xyp = nullptr;  // expert GEMV scratch (glu rows, down partials)
    int* d_njt = nullptr;
    int8_t* qx = nullptr;      // W8A8 GEMV: quantized activation rows (2 passes) and their scales
    float* qxs = nullptr;
    // prefill only (mx > MR)
    bf16* deq = nullptr;       // dequantization scratch
    size_t deq_n = 0;
    int ucap = 8192;           // MLA prefill on tensor cores: union key lists, masks, scores of a query block
    unsigned* ubits = nullptr;
    int *upre = nullptr, *ukeys = nullptr, *d_u = nullptr;
    int uw = 0, ublk_n = 1;    // r12: union bitmap stride (words) and query blocks per call (batched unions)
    int* h_ublk = nullptr;     // pinned: union sizes of the blocks of one call (points at Worker::h_ublk_pin)
    bf16 *ukb = nullptr, *uP = nullptr, *uqb = nullptr;
    unsigned char* umask = nullptr;
    float* uS = nullptr;
    int *psel = nullptr, *pnsel = nullptr;
    __half *h16x = nullptr, *h16y = nullptr;
    int8_t *m8x = nullptr, *m8h = nullptr, *m8h2 = nullptr, *m8hg = nullptr;  // prefill W4A8 experts: int8 activations / h and their scales
    float *m8s = nullptr, *m8hs = nullptr, *m8hs2 = nullptr, *m8hgs = nullptr;
    __half* m16h = nullptr;    // prefill fused fp16 experts: glu rows
    GuJob* d_gujobs = nullptr;
    __half* m16x = nullptr;    // fp16 copy of the MoE input rows (tuned kernels' cp.async source)
    GuJob* h_gujobs = nullptr;  // prefill fp16 dense GEMMs: activation rows, output tile  // prefill: selected pools of the query rows (2 buffers by call parity)
};

// round 24 ([glm-r24-arce]): copy-engine prefill all-reduce. The sender's copy thread records an event after each
// peer copy and publishes it here; the receiver's worker takes it and makes its comm stream wait on it.
struct GCeHub {
    std::mutex mu;
    std::condition_variable cv;
    std::map<std::tuple<unsigned, int, int, int>, cudaEvent_t> ev;  // (seq, kind, src, dst)
    void put(unsigned seq, int kind, int src, int dst, cudaEvent_t e);
    cudaEvent_t take(unsigned seq, int kind, int src, int dst);
};
struct GCeCmd {
    cudaEvent_t ev = nullptr;  // recorded on the worker's comm stream; the copies wait on it
    CeCopy cp[3];
    int peer[3] = {-1, -1, -1};
    int kind = 0;              // 0 = RS rows, 1 = AG sums
    unsigned seq = 0;
    bool stop = false;
};
struct Engine;

// per-sequence decode state of a card (c > 1: one bank per sequence; bank 0 = the original fields). bind(s) swaps the
// bank's pointers into the Layer / Worker fields, so every kernel and graph captured under a binding uses that sequence
struct LState {
    float *S = nullptr, *snap = nullptr, *conv_rows = nullptr, *conv_hist = nullptr;
    bf16* kc = nullptr;
    float *ki = nullptr, *gsc = nullptr, *pk = nullptr;
};
struct SeqBank {
    std::vector<LState> ls;
    int *d_pos = nullptr, *d_acc = nullptr, *d_mpos = nullptr;
    float *hfin = nullptr, *mx_ = nullptr, *mh = nullptr, *mhn = nullptr;
    float **d_kS = nullptr, **d_ksnap = nullptr, **d_khist = nullptr, **d_krows = nullptr;
    cudaGraphExec_t gstep[MR] = {}, gstep_s[MR] = {}, gcommit = nullptr, gmtp[MR] = {};
};

// per-card worker (one host thread each)
struct Worker : Acts {
    Engine* E = nullptr;
    Plan p;
    int dev = 0;
    cudaStream_t st = nullptr, cst = nullptr;  // compute, copies
    cudaStream_t bst = nullptr;  // round 8: background admission copies (GLM_ADMIT_RULE=2)
    cudaEvent_t ev_bg = nullptr;
    bool bg_rec = false;
    void join_bg();
    cudaStream_t ast = nullptr;  // decode: admission copies (high priority) concurrent with the hot-expert GEMVs
    cudaEvent_t ev_af[2] = {}, ev_aj[2] = {}, ev_ah = nullptr;
    cudaStream_t pst = nullptr;     // decode side stream: x copy to the host request, next-layer prediction router
    cudaEvent_t ev_side[2][4] = {};  // fork / join of pst per row group (x copy, prediction)
    float* pred_part = nullptr;     // prediction router partials / logits (apart from big: the shared expert uses it)
    float* pred_lg = nullptr;
    // round 9 late prediction (GLM_PF_LATE): block-1 mHC of layer l + 1 on the streams after layer l, its router
    float *lmix = nullptr, *lpost = nullptr, *lcomb = nullptr, *lxf = nullptr, *lw = nullptr;
    bf16* lxb = nullptr;
    int* d_lids = nullptr;
    cudaEvent_t ev_late[2] = {};
    bool late_rec = false;
    float* mla_scr = nullptr;
    float* wu32 = nullptr;          // round 8: fp32 wuk / wuv of the current layer (prefill) when only bf16 copies are kept
    const float* wuk32(const Layer& Y, cudaStream_t s);
    const float* wuv32(const Layer& Y, cudaStream_t s);       // split-K MLA decode partials (GLM_MLA_SPLIT)
    cudaEvent_t ev_l2[2] = {};
    bool l2_rec = false;
    void l2pf_fork(const void* ptr, size_t n);  // L2 prefetch of weights on pst (GLM_L2PF_KO)
    void l2pf_join();  // fork / join of ast inside the step graph (per row group)
    // intra-step wavefront (4a): rows split into group 0 [0, wa) and group 1 [wa, m); grp = the group being issued
    int grp = 0;
    bool aj_rec[2] = {};        // ev_aj[g] recorded in the step being captured
    bool ah_rec = false;        // round 15: ahead admissions for the next layer are in flight on ast (ev_ah)
    bool ah_on = false;         // round 15: this step body issues / waits for ahead admissions (plain step_body only)
    bool side_rec[2] = {};      // prediction recorded on pst for this group (joined in moe_post)
    int* d_posb = nullptr;     // decode position of group 1's first row (d_pos + wa)
    void view(int r0);         // per-row activation pointers of the decode set offset by r0 rows
    static void view_from(Acts& a, const Acts& b, int r0);
    // prefill all-reduce overlap: the rows in two halves, the comm of one half next to the other half's compute
    cudaEvent_t ev_sel = nullptr;  // prefill indexer split: this card's selected rows sent to the peers
    int sel_par = 0;
    float* bnd_scr = nullptr;  // fused boundary: partial mixes / sums of squares
    unsigned* bnd_bar = nullptr;
    unsigned long long* stp = nullptr;  // GLM_STAMP=2: this layer's stamp row (fine split of the attention part)
    void stamp(int i);
    void stampl(int l, int i);
    cudaStream_t cms = nullptr;
    cudaEvent_t ev_c[4] = {};
    bool ov_mixA = false, pf_lmix = false;  // prefill: the last merge already left the next block's mixes
    bool ov_pend = false;      // the previous layer's MoE all-reduce of half B is still running on cms
    int ov_ha = 0, ov_T = 0;
    void ar_on(cudaStream_t s_, int T, bool overlapped);  // all-reduce part -> sum of the current view (T rows)
    // round 24 ([glm-r24-arce]): the prefill all-reduce on the copy engines (only local kernels on the SMs)
    unsigned ce_seq = 0;
    // round 25 (GLM_PFSTAT): host time blocked on peer events / in device syncs during a prefill (us)
    double hw_ce_us = 0, hw_mla_us = 0, hw_moe_us = 0, hw_lane_us = 0;
    std::thread ce_th;
    std::mutex ce_mu;
    std::condition_variable ce_cv;
    std::deque<GCeCmd> ce_q;
    cudaStream_t ce_st[3] = {};
    std::deque<cudaEvent_t> ce_old, ce_old_take;
    void ce_push(const GCeCmd& c);
    void ce_loop();
    void ce_take(unsigned seq, int kind, cudaStream_t s_);
    [[noreturn]] void ce_stall(unsigned seq, int kind, int src);  // review r24-L3: a 120 s hub wait (engine stall report)
    void ar_ce(float* sum_rows, const float* part_rows, int nr, cudaStream_t s_);
    void layer_ov(int l, int T, int n0);
    void ov_finish();          // after the last layer: merge of half B
    // round 10 (P-1 / P-3): prefill window of T rows in pn_n parts; attention + its all-reduce per part (pipelined on
    // cms), MoE over the whole window (one pass over the cold experts), its all-reduce per part
    static constexpr int PNMAX = 16;
    int pn_n = 0, pn_b[PNMAX + 1] = {};
    bool pn_pend = false;
    cudaEvent_t ev_pa[PNMAX] = {}, ev_pm[PNMAX] = {}, ev_ps[PNMAX] = {};
    void pn_split(int T);      // parts of T rows (each <= mxa, multiples of 64)
    void layer_pn(int l, int T, int n0);
    void pn_finish();
    // round 20 ([glm-arch1-owner], GLM_PF_OWN): the window's merge, mHC, router and final norm run once per row on its
    // reduce-scatter owner, which publishes xb (+ routing, + the final hidden) instead of the sum
    bool own = false;          // this window runs owner mode (set in forward)
    bool own_routed = false;   // moe(): d_ids / d_w already hold the owners' routing of every row
    cudaEvent_t ev_oa[PNMAX] = {}, ev_om[PNMAX] = {};
    cudaStream_t ags = nullptr;  // owner allgathers (own flags, not cms: they must not queue behind the MoE all-reduces)
    cudaStream_t ost = nullptr;  // owner work (GLM_OWN_SIDE)
    cublasHandle_t cbo = nullptr;
    void* cbows = nullptr;
    cudaEvent_t ev_of[PNMAX] = {};  // GLM_OWN_SIDE: owner work of part k finished on ost
    unsigned* agf = nullptr;     // [4] allgather flags: agf[g] = the last call of card g that landed here
    unsigned* agdone = nullptr;  // blocks finished in the running allgather
    unsigned ag_seq = 0;         // allgather calls so far (same sequence on every card)
    std::vector<std::pair<int, int>> own_ranges(int T) const;  // (start, rows) owned in a view of T rows
    void ar_own(cudaStream_t s_, int T, bool ov);               // reduce-scatter: own rows of sum
    void ag_own(cudaStream_t s_, int r0, int T, int what, bool ov);  // own rows of window rows [r0, r0 + T) to the peers (1 xb, 2 routing, 4 sum)
    void own_fin(int l, int blk, int T, bool merge);            // own rows of the view: merge, mHC (, router)
    bool own_ok(int T) const;                                   // every own range of every part > MR rows
    cudaEvent_t ev_ka[2] = {}, ev_kb[2] = {};
    float* kda_set(int z);
    void kda_a(int l, int T, int z);
    void kda_r(int l, int T, int z, cudaStream_t s_);
    void kda_b(int l, int T, int z);
    void wave_body(int m, int wa, bool pair = false);
    std::vector<SeqBank> banks;  // c > 1: per-sequence state (banks[cur_seq] is stale while bound: live in the fields)
    int cur_seq = 0;
    void save_bank(SeqBank& b);
    void load_bank(const SeqBank& b);
    void bind(int s);
    void alloc_seq();          // a second sequence's state (KDA, MLA caches, MTP rows, positions)
    cudaGraphExec_t gpair[MR + 1][MR + 1] = {};
    int* d_am2 = nullptr;      // argmax of the rows of a pair step
    void pair_body(int ma, int mb);
    void pair_step(int ma, int mb);
    void moe_pre(int l, int m);
    void moe_post(int l, int m, bool early = false, bool nolane = false);
    int* h_ublk_pin = nullptr;  // review r16 M1: the pinned h_ublk buffer, allocated once (Acts copies only point at it)
    float* pdef = nullptr;     // round 15 (GLM_LANE_DEFER): block-1 post weights of the MoE layer whose lane rows are pending
    int def_qi = -1;           // its request index (out[def_qi]), -1 = nothing pending
    cublasHandle_t cb = nullptr;
    CommCounters ctr;
    std::vector<Layer> L;
    float* final_norm = nullptr;
    W8 lm;                   // [v1 - v0][4096]
    W8 lm4;                  // round 11: RTN INT4 copy of the lm_head slice for the MTP drafts only (GLM_MTP_Q4)
    W8 lm_d;                 // MTP draft vocabulary rows of this card's slice (GLM_DRAFT_VOCAB), N = 0: full lm
    std::vector<int> h_dmap; // draft row -> token id
    // expert cache slab
    char* slab = nullptr;
    int nslots = 0;
    size_t slab_b = 0;    // round 26: slab bytes (slots + the MLA region's part past them)
    size_t mla_off = 0;   // round 26: byte offset of the MLA region in the slab (GLM_MLA_POS)
    SliceL sl;
    // pinned host mirrors (sized for the largest prefill chunk)
    int* h_ids = nullptr;
    float* h_w = nullptr;
    XJob* h_jobs = nullptr;
    int* slots_all = nullptr;  // device [NL + 1][288]
    unsigned* d_seq = nullptr; // decode step counter
    unsigned* d_wflag = nullptr;  // round 22: k_wait_add [2][NL + 1] x {seen seq, poller ticket}
    int* d_bgdone = nullptr;   // round 17 (GLM_BG_PERL): boundary copies landed on this card (monotonic count)
    int* d_bgneed = nullptr;   // round 17: [NL + 1] count plan(l) waits for (copies of layers <= l), from E->bg_need
    int* d_nj = nullptr;       // [2] job counts (hot, admitted)
    int* d_am = nullptr;       // argmax (value bits, index)
    int* d_pids = nullptr;     // predicted experts of the next layer
    float* d_pw = nullptr;
    float* d_mse = nullptr;    // MTP row softmax normalisers
    int* d_pos = nullptr;      // decode position (graph-replayable steps)
    int* d_acc = nullptr;      // accepted rows of the last verify step
    float* hfin = nullptr;     // final normed hidden of the step rows [MR][H] (MTP input)
    int ll_idx = -1;           // LL all-reduce index within a step (-1: fp32 path)
    int ar_idx = -1;           // fp32 P2P all-reduce index within a step (-1: off)
    P2pAr par;
    cudaGraphExec_t gstep[MR] = {}, gcommit = nullptr, gstep_s[MR] = {};  // gstep_s: sampling variant
    void* cbws = nullptr;      // cuBLAS workspace (graph capture)
    int nkda = 0;              // KDA layers and their commit tables (device pointer arrays)
    float **d_kS = nullptr, **d_ksnap = nullptr, **d_khist = nullptr, **d_krows = nullptr;
    // release track [glm-rel-stops]: the KDA conv history before the last watched commit (nkda x 9 HK floats, lazy)
    float* khbak = nullptr;
    bool hist_io(bool save, int back = 0);  // save: copy it; restore: copy it back and move d_pos back by `back` rows
    // MTP drafter (layer 45): resident experts, own inputs
    MtpW mw;
    char* mtp_slab = nullptr;
    float *mx_ = nullptr, *mh = nullptr, *mhn = nullptr;  // MTP residual, input h rows, head-normed output
    bf16* xcat = nullptr;
    int *d_mtok = nullptr, *d_mpos = nullptr;
    unsigned* d_mseq = nullptr;
    unsigned long long* mtp_cnt = nullptr;  // MTP routing counts of the draft passes (GLM_MTP_COUNT=1)
    const unsigned* ar_seq = nullptr;  // tag counter of the P2P all-reduce (null: d_seq)
    cudaGraphExec_t gmtp[MR] = {};
    cudaGraphExec_t gcmtp[MR] = {};  // round 11: commit + MTP pass
    // round 19 ([glm-mtp-dev]): the whole draft chain in one graph (catch-up pass, then chain passes under graph
    // conditionals): the cards exchange their vocab slice's argmax / normaliser by an all-reduce, pick the draft and the
    // confidence gate on the device; one host round trip per step instead of one per pass. [commit][k - 2][a - 1]
    cudaGraphExec_t gchain[2][MR][MR] = {};
    cudaStream_t chs[MR] = {};     // capture streams of the conditional bodies (one per nesting level)
    int* d_chres = nullptr;        // drafts [0, MR), conf bits [MR, 2 MR), kc [2 MR]; + q (double) after
    int* d_chpos = nullptr;        // positions of the passes [MR]
    int* d_dmap = nullptr;         // draft vocabulary row -> token (lm_d), null: ix + v0
    int mtp_core(int m, const int* dpos);  // returns the rows whose results are in d_am / d_mse (GLM_MTP_LAST1: 1)
    void chain_pick(int row, int j, int k, void* next);
    void chain_body(int a, int k);
    void mtp_body(int m);
    void mtp_prefill(const int* tok, int T, int c0, int nt, int n0);
    void mtp(int m, bool bg = false);
    void moe_mtp(int m);
    void step(int m);
    void step_body(int m, bool samp = false);
    void commit();
    void commit_body();
    void moe_dec(int l, int m);
    void mla_dec(int l, int m, const int* dp = nullptr);

    std::map<std::string, double> mem;  // VRAM by kind (bytes)
    void init(Engine* e, const Plan& pl, int device);
    void alloc_acts(int rows, int arows = 0);   // allocate (arows: attention scratch rows, 0 = rows) the current Acts for rows (bump allocator when active)
    Acts dec_acts;               // decode set (own allocations)
    size_t pf_bytes(int rows);   // bytes of a prefill set of rows (dry run)
    void use_pf(char* base, int rows);
    int pf_arows(int rows) const;  // attention scratch rows of a prefill set of rows  // prefill set placed at base (inside the lent slab tail)
    void use_dec();
    void clear_pf_fields();
    // forward of T rows at positions n0.. (prefill chunk or decode / verify rows); writes local logits if want
    void forward(const int* tok, int T, int n0, bool want_logits);
    void layer(int l, int T, int n0);
    void mhc_pre(int l, int blk, int T, bool have_mix = false);
    // prefill merge of the rows in view; with E->pf_mm also the mixes of the next block (l, blk) (returns true then)
    bool pmerge(int l, int blk, int T);
    void merge(int T);
    void kda(int l, int T, int n0, bool verify = false, int r0 = 0);
    void mla(int l, int T, int n0);
    void mla_cache(int l, int T, int n0);
    void mlp(int l, int T);
    void moe(int l, int T);
    void moe_body(int l, int T);
    float *rlg = nullptr, *rxp = nullptr;  // round 20: fixed-block router scratch
    void router_rows(int l, const float* x, int n, int* ids, float* w);
    // round 20 prefill lane ([glm-arch2-pflane]): this layer's lane experts (all cards), rank 0 runs them
    std::vector<int> lane_e;
    std::vector<std::vector<int>> lane_rows;
    std::vector<std::vector<float>> lane_w;
    int lane_l = -1;
    float *lane_xh = nullptr, *lane_yh = nullptr;
    std::vector<float> lane_yp;
    cudaStream_t lst = nullptr;
    cudaEvent_t ev_lx = nullptr, ev_ly = nullptr;
    int* lane_ih = nullptr; int* lane_id = nullptr; int lane_n = 0;
    std::vector<int> lane_map;
    bool lane_alloc();
    void lane_start(int l, int T);
    void lane_finish(int T);
    void allreduce(int T);
    void dump(const char* name, int l, const float* d, size_t n);
    void gemm_f16(const __half* W, const __half* X, __half* Y, int N, int T, int K);
    void gemm_bf16(const bf16* W, const bf16* X, float* Y, int N, int T, int K, int ldy = -1, float beta = 0.f);
    // Y[T][N] (+)= X . W^T for an INT8 weight: T <= MAXR -> INT8 GEMV (x fp32 or bf16), else dequantized row chunks + cuBLAS
    // (needs xb, the bf16 copy). ldx / ldy: row strides.
    void lin(const W8& w, const float* xf32, const bf16* xb16, float* Y, int T, int ldy = -1, bool acc = false,
             int ldx = -1, const int* dp = nullptr, int nskip = 0);
    // MLA prefill on tensor cores (query blocks of UQB)
    static constexpr int UQB = 64;
    int* h_rall = nullptr;     // prefill MoE row lists (pinned)
    float* h_wall = nullptr;
    cudaEvent_t ev_deq[2] = {}, ev_use[2] = {};
    // round 10: prefill cold-expert staging ring (sring, ns slots in the prefill set): the layer's cold experts are
    // DMA'd + untiled into it from the start of the layer's attention, consumed by moe() in expert order
    char* sring = nullptr;
    int sr_n = 0, sr_l = -1, sr_issued = 0;
    bool sr_early = false;
    std::vector<int> sr_list;  // cold experts of layer sr_l (expert order)
    int sr_pos[288] = {};      // expert -> index in sr_list (-1: not cold)
    cudaEvent_t ev_sr[256] = {}, ev_sf[256] = {}, ev_srs = nullptr;
    void sr_start(int l);
    void sr_issue(int upto);
    cudaStream_t krst = nullptr; // r12: prefill KDA recurrence (high priority; GLM_PF_KRHI=0: = gst)
    cudaStream_t gst = nullptr;  // round 10: prefill expert gate/up (next expert) next to the down on st
    cudaEvent_t ev_gu[2] = {}, ev_q8 = nullptr, ev_ids = nullptr;
    static constexpr int NLAND = 4;  // prefill: ring of DMA landing blocks (tile-major), own copy stream
    char* land[NLAND] = {};
    cudaEvent_t ev_land[NLAND] = {}, ev_lfree[NLAND] = {};
    cudaStream_t dmst = nullptr;
    char* stage[3] = {};       // prefill: cold expert slices staged by DMA (2 = tile-major landing)
    // prefill run-ahead (GLM_PF_AHEAD): every cold expert of a MoE layer staged (DMA + untile) into its own block
    // cstage[e] before the layer's MoE phase (during the previous layer's attention), then one gate/up launch over all
    // experts of the layer
    char* cstage = nullptr;
    int staged_l = -1;         // layer whose cold experts are (being) staged into cstage
    bool pf_more = false;      // set by the prefill driver: another window of >= 2048 rows follows this one
    cudaEvent_t ev_staged = nullptr, ev_moe_done = nullptr;
    static constexpr int NGRP = 32;
    cudaEvent_t ev_grp[NGRP] = {};
    void stage_cold(int l);
    // round 18 ([glm-lcp-reuse]): prefix snapshot of this card's sequence state at position q (KDA S + conv history,
    // the index-key / gate-score ring rows q-4..q-1 of every MLA layer incl. MTP, hidden row q-1). hid: device row of
    // the hidden (save: ck_h, set at the window end; load: mh row 0). The MLA latents / pool keys stay in place.
    template <class F> void ck_walk(int q, float* hid, F f);
    size_t ck_bytes();
    void ck_io(char* host, int q, bool save, bool hid = true);  // synchronous on st (hid: + the hidden row ck_h / mh)
    void pf_abort();           // [glm-rel3-cancel] a gated prefill stopped between layers: queued work done, window state off
    // [glm-conv-arc]: logical rows [s, e) of archive window k (rows k*page .. (k+1)*page) between the MLA caches of
    // this card (latents + complete pool keys; the MTP layer's physical rows are shifted by one: its row p holds the token
    // at p + 1) and a host page. save: only this card's share of the layers (stored once, replicated on every card);
    // all: every layer (load, or the GLM_ARC_CHK comparison)
    void arc_io(char* blk, int k, int s, int e, bool save, bool all);
    float* ck_h = nullptr;     // hidden of the last row of the window just finished (device, H)
    void mtp_prev_row(int tok0, int p);  // MTP cache row p from mh (hidden p) and the token at p + 1
    void sgemm(const float* W, const float* X, float* Y, int N, int T, int K, int ldw = -1, int ldx = -1, int ldy = -1);
};

struct EngSeq {  // host-side speculative state of one sequence (c > 1)
    int pos = 0, sp_t = -1, sp_k = 0, sp_kcur = 0, sp_reused = 0, last_tok = -1;
    int sp_d[MR] = {};
    float sp_c[MR] = {};
    std::vector<int> sp_hist;
    std::vector<float> sp_lout;
};
struct Engine {
    std::string dir;
    int ngpu = 3;
    std::vector<int> devs;
    std::vector<Plan> plans;
    std::vector<std::unique_ptr<Worker>> W;
    Comm comm;
    int nl = NL;               // layers loaded (test mode GLM_LAYERS)
    int ctx = 0;               // context capacity (tokens)
    int kring = 0;             // ring rows of the index keys / gate scores (>= prefill chunk + 3)
    int pos = 0;               // tokens in the cache
    bf16* emb = nullptr;       // host pinned [VOCAB][4096]
    char* hx = nullptr;        // host experts [layer][288][HX_BYTES] (layers 3..44), pinned mapped
    std::vector<char*> hx_layer;  // per layer base (host), nullptr for dense
    HxLayout hxl;              // host expert layout (stride hxl.stride per expert)
    bool gu3 = false;          // host gate/up INT3 (checkpoint marker qglm.json, GLM_GU3=0 off; round 6)
    bool gu3_mixed = false;    // per expert (marker "mixed": true): INT3 where the gate/up codes fit, INT4 otherwise
    // dynamic cache (host side, identical slots on every card)
    std::vector<int> slot_of;  // [layer * 288 + e] -> slot or -1
    std::vector<int> slot_owner;  // [slot] -> key or -1
    // debug dump
    std::string dump_dir;
    // decode step: handshake, coordinator, CPU lane, cache policy
    MoeReq* req = nullptr;
    MoeDec* dec = nullptr;
    BgQ* bgq = nullptr;          // round 11 (GLM_BG_BOUND = max deferred per step, 0 off)
    int* bg_flag = nullptr;      // per rank: 1 while the boundary copies run
    // round 17 (GLM_BG_PERL=1): plan(l) waits only for the boundary copies of layers <= l (queue sorted by layer, one
    // slot-table entry + done count per copy) instead of all of them before layer 3's plan
    bool bg_perl = false;
    int* bg_need = nullptr;      // pinned host [NL + 1], copied into each card's d_bgneed at the verify graph head
    int bg_cum = 0, bg_cum0 = 0; // copies enqueued so far (all cards enqueue the same list)
    std::vector<int> bg_qprev;   // slots of the queue whose copies land during the next verify (kept in bg_pend)
    int bg_bound = 0, bg_bound_blk = 24, first_moe = 3;
    bool bg_dma = true, bg_sync = false;
    bool dbg_slotchk = false;
    int dbg_delay[4] = {-1, -1, -1, 0}, dbg_hog[3] = {0, 0, 0};  // round 11 debug
    unsigned long long st_bgq = 0;
    std::vector<char> bg_pend = std::vector<char>(1 << 16, 0);  // per slot: background copy pending this step
    std::vector<int> bg_pl;
    const char** d_hxl = nullptr;  // device-visible table of hx_layer
    BgList* bg_ql = nullptr;  // r12: snapshot of the step's queue (by value into the kernels)
    int bg_n = 0;
    bool bg_due = false;      // boundary copies prepared, per-card enqueue pending (after the first MTP launch)
    int pfstat = 0;           // GLM_PFSTAT (captured into the graphs)
    int caller_cpu = -1;      // r12: run_all owner pin (GLM_CALLER_CPU), applied only inside run_all
    cpu_set_t caller_mask;
    bool caller_step = false, caller_held = false;  // round 18: GLM_CALLER_STEP, a StepPin is active
    struct StepPin { Engine* e; bool on = false; explicit StepPin(Engine* e_); ~StepPin(); };
    bool ver_pending = false; // r12 (review C4): a verify not yet committed (KDA pointer mode)
    void bg_prep();
    void bg_enqueue(Worker& w);
    void bg_flush();
    MoeOut* out = nullptr;
    int* h_tok = nullptr;
    int* h_am = nullptr;
    int* h_acc = nullptr;
    CpuLane* lane = nullptr;
    std::thread coord;
    volatile bool coord_stop = false;
    std::atomic<unsigned> coord_done{0};
    // [glm-idle] the coordinator sleeps between steps: coord_go = steps issued (set in verify / verify2 before the GPUs
    // start the step), the coordinator waits for it at layer 3 (spin GLM_COORD_SPIN_US, then block)
    std::atomic<unsigned> coord_go{0};
    IdleGate coord_gate;
    std::vector<float> score;
    std::vector<int> free_slots;
    // round 17 (GLM_DET, default 1): greedy output independent of earlier requests. The cache state after the seed is
    // kept and restored at every new sequence (reset), so residency, and with it the tau skip and the GPU / CPU-lane
    // split of each expert's numerics, depend only on this sequence's tokens. GLM_DET=0: the cache carries over.
    // Limit (round 20, review r20b M3): the promise is per prefill split. The prefill CPU lane (GLM_PF_LANE_ROWS) picks
    // experts by their row count in a window, so a prompt prefilled in one window and the same prompt split at a hint /
    // snapshot / window end give different numerics; so does the prefill router (one SGEMM per chunk: a row's logits
    // depend on the rows sharing the call, unless GLM_ROUTER_RB is set) and the window size (lent slots decide which
    // experts are cold). Exactness gates compare like with like (lcp / conv use the same cut in the reference); a
    // split == one-shot check needs GLM_PF_LANE_ROWS=0 and GLM_ROUTER_RB set, and can still differ through the lend.
    int det = 1;
    bool pf_last1 = true;
    float pf_tau = 0.f;      // round 17: prefill skip threshold (GLM_PF_TAU, default 0 = lossless prefill)
    bool mtp_last1 = false;  // round 17 (advisor D1): MTP catch-up computes MoE / head for the last row only (GLM_MTP_LAST1)  // round 17 (advisor P1): prefill logits of the last row only when nothing else is read (GLM_PF_LAST1=0 off)
    std::vector<int> canon_owner, canon_free;
    std::vector<float> canon_score;
    std::mt19937 canon_rng;
    unsigned long long st_det_n = 0, st_det_slots = 0;
    double st_det_us = 0;
    bool det_dirty = false;      // the cache state moved since the seed / the last restore (a step or a re-seed ran)
    bool lend_canon = false;     // round 18: this lend started from the seeded state (unlend restores it exactly)
    bool det_pending = false;    // an asynchronous restore's copies may still run (det_wait joins them)
    int* det_tab = nullptr;      // pinned host copy of the slot table for the asynchronous upload
    void det_restore(bool async = false);
    void det_wait();
    void seq_end();              // a sequence ended: restore in the background (server release, generate end)
    void fill_slots(const std::vector<std::pair<int, int>>& sk, bool sync = true);
    void reseed(const std::vector<std::pair<int, int>>& sk);  // round 27: after a prefill, in the background (det_wait joins)  // (slot, key): DMA + untile on every card
    std::mt19937 rng{12345};
    float decay = 0.999f;
    int admit_cap = 16;
    int admit_lcap = MXE;
    int admit_icap = MXE, admit_eff = 1;  // round 13: in-step admission cap (GLM_ADMIT_ICAP), lane-cost model without skipped experts (GLM_ADMIT_EFF)
    int admit_effi = 1;  // round 14 (review r13b M1): in-step places only for cold experts with a row >= tau (GLM_ADMIT_EFFI)
    float adm_hotn = 0.f, adm_tgr = 0.f;  // round 14 (review r13b L1): row-dependent cost-model terms
    bool ah_stat = false;
    double st_ah[5][3] = {};  // round 15 gate: per top-N of the prediction listed / effective cold / admitted in-step; [4] = totals
    unsigned long long st_ins = 0, st_ins_so = 0, st_bg_so = 0, a_ins0 = 0, a_ins_so0 = 0, a_bg_so0 = 0;  // in-step admissions, of them skip-only; skip-only background
    // round 8: admission only inside the lane's shadow (GLM_ADMIT_RULE=1): admit a cold experts of a layer only while
    // max(hot, a * tp) + a * tg <= max(hot, (ncold - a) * tl), i.e. the GPU MoE path does not end after the lane
    int admit_rule = 2;
    int bg_blocks = 2;         // blocks per background admission (few: PCIe-bound, keep SMs for the step)
    int admit_ahead = 0;       // round 15 (GLM_ADMIT_AHEAD=N): at layer l, admit the top-N predicted effective cold experts of l + 1
    int ah_blocks = 12;        // blocks per ahead admission copy (GLM_AH_BLOCKS)
    bool ah_force = false;     // GLM_AH_FORCE=1: ahead admissions skip the LFU score test of admit()
    MoeDec* dec_ah = nullptr;  // host -> GPUs: ahead admissions decided at layer l (index l), copied before plan(l + 1)
    unsigned long long st_ah_n = 0, st_ah_hit = 0, a_ah_n0 = 0, a_ah_hit0 = 0;  // ahead admissions, of them routed at l + 1
    float adm_tp = 190.f, adm_tg = 18.f, adm_tl = 170.f, adm_hot = 150.f;
    float pcie_share = 0.f;
    bool ar_ll12 = true;       // decode all-reduce: 16-byte words of 3 fp32 + tag (GLM_AR_LL12=0: 8-byte LL)
    bool ar_ll = true;         // decode all-reduce: LL (value, tag) protocol (GLM_AR_LL=0: flags)
    bool chain_raw = false;    // MTP chain input: raw MTP output instead of the head-normed one
    float skip_tau = 0.10f;   // LOSSY default 0.10 since round 13 (GLM_SKIP_TAU=0.06: round-8 default, 0: off): skip cold (CPU-lane) expert rows with gate weight < tau
    unsigned long long st_skip = 0;
    bool w8a8 = true;          // decode GEMV with two-pass int8 activations (GLM_W8A8=0: fp32 activations)
    bool lookup = false;       // lookup drafts from the session history (trigram match), MTP chain otherwise
    unsigned long long st_lookup = 0;
    bool prefetch = true;      // CPU lane prefetches the predicted cold experts of the next layer during the GPU phase
    unsigned long long st_pf = 0;
    double st_pf_cov = 0, st_pf_use = 0, st_pf_rank[8] = {}, st_pf_rank_use[8] = {};  // GLM_PFSTAT
    double st_pf_why[4] = {};  // unused prefetch: mispredicted / admitted read-through / skipped by tau / hot (round 9 audit)
    std::vector<std::vector<const char*>> prev_ex = std::vector<std::vector<const char*>>(64);
    double st_reuse[4][2] = {}, st_bound_us = 0;
    std::chrono::steady_clock::time_point t_last_lane{};
    std::chrono::steady_clock::time_point t_lane_end{}, clk0{};
    double st_tl[10] = {}, tsc_mhz = 1.0;
    uint64_t tsc0 = 0;
    uint64_t tsc_at(std::chrono::steady_clock::time_point t) const {
        return tsc0 + (uint64_t)(std::chrono::duration<double, std::micro>(t - clk0).count() * tsc_mhz);
    }
    unsigned long long st_steps0 = 0, a_cpu0 = 0, a_bg0 = 0, a_adm0 = 0, a_pb0 = 0, a_skip0 = 0, a_pfbg0 = 0;
    int pf_late = 0;           // round 9: late next-layer prediction (1: statistics only, 2: prefetch switches to it)
    double st_pa[2][3] = {}, st_pan = 0;  // precision of the first 1 / 2 / 4 prefetch candidates, early / late
    int pf_order = 0;          // round 9: prefetch list by predicted weight summed over rows, picks below the skip tau dropped (opt-in GLM_PF_ORDER=1; neutral)
    bool pf_bg = false;        // round 9: a prefetched cold expert is admitted only in the background (lane computes it from cache), never read-through (opt-in GLM_PF_BG=1; neutral)
    unsigned long long st_pf_bg = 0;
    int pf_max = 8;    // fraction of the non-admitted cold experts computed zero-copy by the GPUs
    int pf_early = 1;     // round 13 (default; GLM_PF_EARLY=0 off; neutral at tau 0.06, +1.5 % at 0.10): prefetch aborts when rank 0 enters the MoE block
    bool use_graph = true, use_ll = false, p2p = false;
    int fuse_kda = 1;          // decode KDA gates: 1 one GEMV kernel for both, 2 inline in prep / out, 0 cuBLAS (GLM_FUSE_KDA)
    bool kda_ptr = true;       // round 11: KDA commit by pointer selection (GLM_KDA_PTR=0: copy snap[a-1] -> S)
    bool fuse_commit = true;   // round 11: commit inside the first MTP pass graph (GLM_FUSE_COMMIT=0: own graph)
    bool commit_in_mtp = false;  // set by spec_step for the next MTP pass
    void kda_mat();
    bool fuse_bnd = true;      // decode: merge + next block's mHC in one kernel with grid barriers (GLM_FUSE_BND=0: off)
    bool fuse_mix = false;     // decode: mHC mixes of the next block from the merged values (GLM_FUSE_MIX=1)
    size_t l2pf_ko = 0;        // decode KDA: bytes of the ko weights prefetched into L2 during the small-op chain
    int l2pf_blocks = 16;
    int mla_tc = 24;           // round 8: decode MLA attention on tensor cores, value = splits (GLM_MLA_DTC, 0 = off)
    bool mla_ubf = true;      // round 8: decode wuk / wuv from bf16 copies (GLM_MLA_UBF)
    bool pf_rlk = true;       // round 27 (P4 b): prefill MoE row lists uploaded by a kernel, not the H2D engine (GLM_PF_RLK)
    bool idx_skip = true;     // round 27 (D4 b): decode skips the indexer's wq_b rows below TOPB pools (GLM_IDX_SKIP)
    bool mla_split = true;     // decode MLA attention split over the token list (GLM_MLA_SPLIT=0: one pass per head group)
    int lane_defer = 0;        // round 15: 1 = the CPU lane's rows of layer l are added after layer l + 1's attention (lossy, Maya-style);
                               // 2 = check mode, added right after layer l's merge and the next block's mHC recomputed (exact up to order)
    // round 16: split deferral (with GLM_LANE_DEFER 1 / 2). Lane experts whose largest row weight is below GLM_DEFER_W
    // (normalised, <= 0: all), in layers with at least GLM_DEFER_MIN lane experts, go to a second lane pass published to
    // out2 and added at l + 1; the rest stay exact at layer l (rank 0 waits for them as without deferral)
    bool def_split = false;
    float defer_w = 0.f;
    int defer_min = 0;
    MoeOut* out2 = nullptr;
    unsigned long long st_defa = 0, st_defb = 0, st_defb_rows = 0, a_defa0 = 0, a_defb0 = 0, a_defbr0 = 0;
    bool ar_early = false;     // decode: MoE all-reduce of the GPU partials during the lane, lane rows added after (GLM_AR_EARLY=1; no gain)
    bool side = true;          // decode: x copy + next-layer prediction on a side stream (GLM_SIDE=0: off)
    bool wave = false;         // intra-step wavefront for verify steps of >= 2 rows (GLM_WAVE=1; GPU-bound today)
    int wave_a = 1;            // rows of group 0 (GLM_WAVE_A)
    std::atomic<int> wave_cur{0};  // group-0 rows of the step being run (0: one group), read by the coordinator
    bool wait_old = false;     // round 22: GLM_WAIT_OLD=1: every k_wait_add block polls the host flag (round-13 form)
    bool admit_par = true;     // admission copies on their own stream, concurrent with the hot experts (GLM_ADMIT_PAR=0: serial)
    float admit_margin = 0.5f;  // TinyLFU hysteresis: admit if count >= victim + margin
    unsigned long long st_bg = 0, st_hit = 0, st_miss = 0, st_admit = 0, st_cpu = 0, st_zc = 0, st_cpu_rows = 0, st_adm_rows = 0;
    double st_cpu_us = 0, st_wait_us = 0;  // CPU-lane time, coordinator waiting for the GPU (layers 4..)
    void policy_init();
    int admit(int key, const int* now, int nnow, int* vkey, bool force = false);
    void coord_loop();
    void coord_start();
    bool mtp = false;          // MTP drafter loaded
    bool mtp_count = false;    // count the MTP routing of draft passes (prior for the resident MTP subset)
    std::vector<float> mtp_rbias;  // original MTP router bias (host); the device copy is masked to the resident subset
    std::vector<int> mtp_slot;     // MTP expert -> index in mtp_slab or -1 (draft routing is steered to resident ones)
    void mtp_mask(const std::vector<int>& resident);  // steer the MTP router to these experts (bias of the rest -1e30)
    int last_tok = -1;         // greedy token after the last forward
    double st_dec_us = 0, st_ver_us = 0, st_mtp_us = 0;
    unsigned long long st_steps = 0, st_rows = 0, st_acc = 0, st_dacc[MR] = {};
    int generate(const int* prompt, int T, int n, int k, int* out);  // greedy with MTP chain drafts (k per step)
    // speculative state (server C API): current input token, drafts, cached tokens (prefix reuse)
    int sp_t = -1, sp_k = 0, sp_kcur = 0, sp_reused = 0;
    int sp_d[MR] = {};
    std::vector<int> sp_hist;
    // round 25 ([glm-p3-turnend]): the server's reply limit (tokens the next step may still accept, 0 = none) and the
    // stop at the first EOS cap the accepted rows before the commit, so the cache never runs past the delivered reply;
    // sp_last_a = rows accepted by the last step (its last row's hidden is hfin row sp_last_a - 1)
    int sp_cap = 0, sp_last_a = 0;
    bool sp_eos_stop = false;
    // release track: [glm-rel-stops] sp_watch (the server watches stop strings): each step keeps its undo (the conv
    // history before its commit, no commit fused into the MTP graph), sp_undo = the last step can be trimmed;
    // [glm-rel-mask] sp_first: a constrained start picks its first token from the prompt end's full logits
    bool sp_watch = false, sp_undo = false;
    int sp_test_a = 0;                 // check tool (glm_test_accept): the next constrained step accepts at most this
    std::function<int(const float*)> sp_first;
    float* h_rowlg = nullptr;          // pinned [VOCAB] (row_logits)
    void row_logits(int j, float* out);  // full logits of row j of the last verify (every card's slice)
    // constrained step: rows [sp_t, drafts[0, nd)], pick(j, logits of row j) -> token (the server's row masks);
    // accepted while the pick equals the next row; same cap / EOS / commit / draft rules as spec_step
    int spec_step_pick(int* out, const int* drafts, int nd, const std::function<int(int, float*)>& pick);
    bool spec_trim(int c);             // the cache ends at c rows (inside the last watched step): true = done
    bool undo_prep();                  // sp_watch: save the conv history before the commit (false: no undo)
    // [glm-rel-pfgate] a prefill run on a job thread of the C API returns to the server between layers: every worker
    // waits at a gate before each layer until the server grants it (host waits only: the queued GPU work, the kernels
    // and their order are those of one call, so the result is bit-identical). A cancel stops at the next window start
    // (pf_stop_rows: rows of the forward done; the state there is complete: KDA / MLA / MTP rows of whole windows)
    struct PfGate {
        std::mutex mu;
        std::condition_variable cv;
        bool on = false, free = false, stop_win = false;
        long credit = 0;
        std::vector<long> used;
        int waiting = 0;
        int prog = 0;                  // rows done (approximate inside a window)
    };
    PfGate pfg;
    // [glm-rel3-cancel] stop_win also stops at the gate before any layer (GLM_PF_CANCEL_MID=0: window starts only); the
    // window's rows are then dead (pf_stop_rows = its start). A gated live continuation (pf_keep, set by ck_prefill)
    // saves the KDA / conv / ring state at its start (pf_ws, pinned host per card), so a stop inside its first window
    // puts the conversation back exactly (pf_live_kept: sp_hist stays live)
    bool pf_cancel_mid = true;
    bool pf_keep = false, pf_live_kept = false;
    bool pf_mid_ok = true;             // review R3-M1: mid-window stops allowed (not for an unsaved live continuation)
    std::vector<char*> pf_ws;
    bool pf_ws_ready();                // the save buffers (allocated once, false without host room)
    int pf_stop_rows = -1;             // forward(): rows done when a cancel stopped it, -1 = whole forward
    bool pf_aborted = false;           // ck_prefill / spec_start: the prompt was cut short by a cancel
    bool pf_gate(int rank, bool win_start, int prog);  // true: stop here (window start after a cancel)
    void ck_answer_end();              // snapshot at the reply's end (GLM_CK_ANS=0 off)
    bool ck_ans_due = false;           // qe_release: take it at the next ck_prefill unless that one continues live
    int spec_draft(int a, const int* toks, int pos0, int src);
    int spec_start(const int* ids, int n, int k);  // returns the first generated token
    int spec_step(int* out);                         // 1..k+1 tokens
    int* h_mtok = nullptr;
    int* h_mpos = nullptr;
    int* h_mam = nullptr;
    int* h_chres = nullptr;    // round 19: device chain results per card [ngpu][2 MR + 1]
    bool dev_chain = true;     // GLM_MTP_DEV (P2P all-reduce only)
    unsigned long long st_chain_chk = 0, st_chain_bad = 0;
    bool chain_chk = false;    // GLM_MTP_DEV_CHK: the per-pass host path's pick recomputed from the exchanged values
    unsigned long long* stamps = nullptr;
    unsigned long long* stamps2 = nullptr;  // GLM_STAMP=2: [ngpu][NL + 1][16] globaltimer points
    std::vector<double> st2_acc;
    unsigned long long st2_n = 0;  // GLM_STAMP: GPU globaltimer stamps per MoE layer (rank 0, mapped)
    double st_g_attn = 0, st_g_plan = 0, st_g_moe = 0;
    unsigned long long st_g_n = 0;
    int stamp_lvl = 0;          // GLM_STAMP: 1 per-layer split, 2 KDA kernels, 3 detailed timeline incl. host handoffs
    long long* hts = nullptr;   // GLM_STAMP=1/3: host CLOCK_REALTIME ns per layer: request seen, decision, lane out published
    std::vector<double> st3;
    FILE* stamp_dump = nullptr;  // GLM_STAMP_DUMP: raw stamps per verify step
    float* h_mse = nullptr;    // MTP rows: softmax normaliser of each card's vocab slice (draft confidence)
    float conf_tau = 0.f;      // confidence-gated draft length: row j drafted while prod of MTP probs >= tau (GLM_CONF_TAU)
    FILE* skip_dump = nullptr;  // round 17 gate (GLM_SKIPDUMP=file): per step and MoE layer, the tau-skipped lane picks
    FILE* conf_log = nullptr;  // GLM_CONF_LOG: per step the drafts' MTP probs and the accepted count
    float sp_c[MR] = {};
    // sampling (temperature > 0): verify rows return top-TKC candidates per card; MTP drafts stay greedy, so a draft
    // d is accepted with probability p(d) and a rejection samples from p without d (exact speculative rule for a
    // point-mass drafter). p = softmax(logits / T) truncated by top_k (<= TKC exact), min_p, top_p, renormalised.
    float samp_T = 0.f, samp_top_p = 1.f, samp_min_p = 0.f;
    int samp_top_k = 0;
    std::mt19937_64 srng{1};
    TopK* h_tk = nullptr;      // mapped [ngpu][MR]
    float* h_invT = nullptr;   // mapped
    unsigned long long st_samp_rej = 0;
    void set_sampling(float T, float top_p, int top_k, float min_p, unsigned long long seed);
    // candidate distribution of verify row j (merged over the cards); returns kept tokens and probabilities
    void row_dist(int j, std::vector<int>& tok, std::vector<double>& p);
    int sample_from(const std::vector<int>& tok, const std::vector<double>& p, int exclude);
    void full_dist(const float* logits, std::vector<int>& tok, std::vector<double>& p);
    std::vector<float> sp_lout;  // full logits of the last prompt row (sampling: first token)
    int decode(int tok);  // one token at the current position, returns the greedy next token
    void verify(const int* toks, int m, int* out);  // m rows from pos, argmax per row, nothing committed
    void commit(int a);   // accept the first a rows of the last verify

    void load(const std::string& dir, int ctx_tokens);
    float* hid_dump = nullptr; // forward(): final normed hidden of every row -> host [T][4096] fp32 (MTP training data)
    // prefill buffers lent from the slab tail: evict the last slots (coordinator idle), place the prefill sets there;
    // afterwards re-seed those slots with the highest-count non-resident experts
    int chunk = 2048;          // prefill chunk rows (GLM_CHUNK)
    bool idx_split = true;     // prefill DSA indexer + top-k split by query rows over the cards (GLM_IDX_SPLIT=0: replicated)
    std::atomic<int> hb_cnt{0};
    std::atomic<unsigned> hb_gen{0};
    void host_barrier();
    bool pf_hyb = false;       // prefill experts: cuBLAS fp16 gate/up (dequant pass) + fused down / scatter kernel (GLM_PF_HYB)
    bool pf_w16t = false;       // fused fp16: 128-row tiles, cp.async pipeline, ldmatrix (GLM_PF_W16T=0: 64-row kernels)
    bool pf_w16b = true;       // fused fp16: resident experts' gate/up batched in one launch (GLM_PF_W16B)
    int pf_ahead = 0;          // fused fp16 cold experts staged into per-expert blocks, gate/up batched (GLM_PF_AHEAD: 0 off,
                               // 1 whole layer staged a layer ahead, 2 staged in the layer's own MoE phase in groups)
    int pf_grp = 16;           // GLM_PF_GRP: cold experts per batched gate/up launch (mode 2)
    bool pf_w16 = true;        // prefill experts: fused fp16 core (GLM_PF_W16=0: dequant + cuBLAS)
    bool pf_w8 = true;         // prefill experts W4A8 on INT8 tensor cores with fused gather / glu / scatter (default; GLM_PF_W8=0 off)
    bool pf_ubatch = true;     // r12: GLM_PF_UBATCH=0 builds the MLA unions per query block (one sync each)
    bool pf_mtpfull = false;   // r12: GLM_PF_MTPFULL=1 runs the whole MTP layer in the prefill (dead work)
    bool pf_d16 = false;       // prefill dense GEMMs fp16 / fp16 accumulation (GLM_PF_D16=1; slower today)
    bool ar_ce = true;         // round 24: prefill all-reduce on the copy engines (GLM_AR_CE=0: SM P2P stores)
    int ar_ce_mode = 2;        // 2: the overlapped all-reduces only (default), 1: all
    GCeHub ce_hub;
    bool pf_f16 = true;        // prefill expert GEMMs fp16 with fp16 accumulation (GLM_PF_F16=0: bf16 / fp32 acc)
    bool pf_ubf = true;        // round 10: prefill MLA wuk / wuv on bf16 tensor cores (GLM_PF_UBF=0: fp32 SGEMM)
    bool pf_pn = true;         // round 10: prefill windows in parts (P-1 / P-3, GLM_PF_PN=0: old halves / chunks)
    int pf_part = 8192;        // max rows of a part (GLM_PF_PART)
    int pf_ring_early = -1;    // GLM_PF_RING_EARLY=1 / 0: ring DMA with the attention / with the MoE; -1: below 12K rows
    int pf_ring = 128;         // round 10: cold-expert staging ring slots (GLM_PF_RING, 0 = old 2-buffer path)
    int pf_head = 0;           // rows of a small first part for windows >= 3 parts (GLM_PF_HEAD, 0 = even)
    bool pf_hotfirst = false;  // round 10: resident experts first in the grouped ring path (opt-in GLM_PF_HOTFIRST=1)
    int pf_gub = 8;            // round 10: experts per grouped gate/up launch in the ring path (GLM_PF_GUB, <= 1 off)
    bool pf_kc = false;        // round 10 (P-5): prefill KDA as the chunked delta rule (opt-in GLM_PF_KC=1)
    bool pf_kov = true;        // round 10: KDA recurrence of a part next to the next part's projections (GLM_PF_KOV=0 off)
    bool pf_spb = true;        // round 10: MLA per-query fallback blocks in one launch per run (GLM_PF_SPB=0: per block)
    bool pf_gu2 = false;       // round 10: expert gate/up on a side stream next to the previous expert's down (opt-in GLM_PF_GU2=1)
    bool pf_mhc2 = true;       // round 10: prefill mHC pre with 256 threads per row, streams in registers (GLM_PF_MHC2=0 off)
    bool pf_skiplent = false;  // GLM_PF_SKIPLENT=1: experts cold only by the lend count as cold for the skip rule
    std::vector<int> slot_pre; // slot_of before the prefill set was lent
    int pf_edge = 2048;        // rows of the first / last part of windows < 3 parts (GLM_PF_EDGE, 0 = even parts)
    bool pf_mm = true;         // round 10: prefill merge + next block's mixes in one pass, no fp32 SGEMM (GLM_PF_MM=0 off)
    bool pf_ss = true;         // prefill mhc_pre takes the merge's sums of squares (GLM_PF_SS=0: own pass)
    int pf_lane_rows = 16;     // round 20: cold experts with <= this many rows of a prefill window go to the CPU lane (GLM_PF_LANE_ROWS, 0 off)
    // ... in windows of <= this many rows (GLM_PF_LANE_MAXT). Review r20b-L3: the lane also needs a window without the
    // cold-expert ring, and a window of >= 4096 rows has one at the defaults (2+ cards, GLM_PF_PN / W4A8 on: pf_arows),
    // so values above 4096 change nothing there (they matter only with GLM_PF_PN=0, GLM_PF_W8=0 or on one card)
    int pf_lane_maxt = 4096;
    double st_pflane_us = 0; long long st_pflane_e = 0, st_pflane_rows = 0;
    int router_rb = 0;         // round 20 (review M1): prefill router in fixed blocks of this many rows (GLM_ROUTER_RB, 0 = one call per chunk)
    bool own_side = true;      // round 20: owner work on its own stream next to st (GLM_OWN_SIDE=0: on st)
    bool own_la = false;       // round 20: owner work + send of part i + 1 queued before part i's attention (GLM_OWN_LA)
    bool own_agov = false;     // round 20: the attention-side allgather with the few-block grid (GLM_OWN_PRE_OV)
    bool pf_own = false;       // round 20: row-owner merge / mHC / router in parts windows (opt-in GLM_PF_OWN=1)
    struct OwnPtr { char *xb, *sum, *ids, *w; };
    OwnPtr own_ptr[4] = {};    // every card's window arrays (set at the lend)
    unsigned* own_flag[4] = {};  // every card's agf
    bool pf_ov = true;         // prefill all-reduce of one row half next to the other half's compute (GLM_PF_OV=0: off)
    int lent = 0;              // slots lent to the prefill set right now
    int ar_rows = 4096;        // rows per prefill all-reduce call (comm buffer size)
    unsigned long long steps_issued = 0;
    int lend_slots(int rows);  // slots a prefill set of rows needs (max over cards)
    void lend(int rows);
    void unlend();
    // round 26 ([glm-p3-mlapos], plan3 D1): the MLA latent cache and the pooled index keys of every MLA layer live
    // position-major at the end of each card's expert slab ([pos][layer][512] bf16, then [pool][layer][128] fp32);
    // only the slots under the rows in use (pages of mla_pg positions, + a margin) are taken from the experts, so the
    // configured context no longer costs slots. GLM_MLA_POS=0: one ctx-sized array per layer outside the slab (as before).
    bool mla_pos = true;
    int mla_nl = 0;            // MLA layers (incl. MTP)
    int mla_s0 = 0;            // slots below the region on every card (the prefill set is lent from the top of them)
    int mla_pg = 4096;         // page (positions, GLM_MLA_PAGE)
    int mla_rows = 0;          // rows the held pages cover
    size_t mla_kcb = 0, mla_pkb = 0;  // region bytes per card: latents, pooled keys (256-aligned)
    std::vector<char> slot_mla;       // per slot: under the rows in use (owner -1, never free / sampled)
    unsigned long long st_mla_take = 0, st_mla_rel = 0;  // slots taken / released (statistics)
    int mla_li(int l) const { return l == MTP_L ? NL / 4 : l / 4; }  // MLA index of layer l (3, 7, .., 43, MTP)
    int mla_page_rows(int rows) const;            // rows of the pages that hold rows [0, rows) + the margin
    std::vector<char> mla_mask(int rows) const;   // slots under rows [0, rows) on any card
    void mla_set(int rows);    // hold exactly the pages of mla_page_rows(rows) (take: evict; release: seed / free)
    void mla_need(int rows) { if (mla_pos && mla_page_rows(rows) > mla_rows) mla_set(rows); }
    int lend_top() const { return mla_pos ? mla_s0 : (int)slot_owner.size(); }  // end of the lendable slots
    bool canon_now() const;    // GLM_DET: the cache is the seed minus the slots under the held pages
    void upload_tables();      // host slot_of (+ MTP map) -> every card's slots_all
    void coord_idle();
    void run_all(const std::function<void(Worker&)>& f);
    int pool_on = -1;                       // GLM_POOL: persistent per-card host threads for run_all
    std::vector<std::thread> pool_th;
    std::thread::id pool_owner;
    std::vector<int> pool_cpus;             // review rel L4: the pool threads' CPUs (the spawn path pins to them too)
    std::atomic<unsigned> pool_gen{0}, pool_done{0};
    IdleGate pool_gate;                     // [glm-idle] pool threads: spin GLM_POOL_SPIN_US, then block on pool_gen
    volatile bool pool_quit = false;
    const std::function<void(Worker&)>* pool_fn = nullptr;
    // T tokens at the current position; returns logits of the last row (full vocab) if out != nullptr
    void forward(const int* tok, int T, float* lout, std::vector<int>* am, const std::vector<int>* lrows = nullptr);
    int pf_window(int T) const;  // rows of one prefill window of a T-row forward (window ends at multiples of it)
    // round 18 ([glm-lcp-reuse]): prefix snapshots in pinned host memory (nseq 1). A snapshot at q is valid while the
    // cache rows [0, q) are the ones it was taken with: entries past a resume point are dropped there. Taken at prefill
    // window ends, forced cuts (server hint = end of the system part, divergence point, tail) and the prompt end
    // (+ its last-row logits: an identical prompt prefills nothing). GLM_CK=0 off.
    struct Ckpt {
        int q = -1;                    // tokens [0, q) cached; -1 = free
        int kind = 0;                  // 0 window end, 1 cut, 2 prompt end, 3 answer end (round 25)
        int last_tok = -1;
        unsigned long long hash = 0, used = 0;
        std::vector<float> logits;     // kind 2: logits of row q - 1 (VOCAB)
        std::vector<char*> buf;        // per card, pinned
        std::vector<std::array<int, 3>> segs;  // [glm-conv-arc] rows [0, q) by writer: (s, e, generation)
        std::vector<int> toks;         // tokens [0, q) (review r19 L1: compared, not only hashed)
        bool pin = false;              // [glm-rel-slice] kind 4: a parked reply's state (never evicted, never a resume point)
        unsigned long long tag = 0;    // its park id (entries may move in the vector)
    };
    // [glm-conv-arc] (round 19): which forward / decode turn wrote the live cache rows. Every write range gets a new
    // generation; a snapshot records the ranges of its rows [0, q). Rows another conversation is about to overwrite are
    // archived first (pinned host pages, once for all cards) when a valid snapshot needs them; a restore copies back the
    // ranges whose live generation differs, so the restored rows are the snapshot's own bytes. A snapshot is valid while
    // every one of its ranges is live or archived. GLM_ARC=0: no archive (one live lineage, as round 18).
    using Seg = std::array<int, 3>;
    std::vector<Seg> lv;               // live rows [0, lv_end) in order
    int lv_gen = 0;
    int lv_end() const { return lv.empty() ? 0 : lv.back()[1]; }
    void lv_cut(int r);
    void lv_push(int a, int b);        // rows [a, b) written now (cut at a first)
    struct ArcPg {
        int g = -1, k = -1, s = 0, e = 0;  // generation, window, logical rows [s, e) held
        unsigned long long used = 0;
        char* buf = nullptr;
    };
    std::vector<ArcPg> arc;
    std::vector<char*> arc_free;       // allocated blocks not in use
    size_t arc_blocks = 0, arc_blk = 0, arc_cap = 0;  // blocks allocated, bytes per block, budget (bytes)
    unsigned long long arc_hold = 0;   // during arc_save: its pages (used >= arc_hold) are never freed
    int arc_page = 2048;
    bool arc_on = true, arc_chk = false;
    unsigned long long st_arc_save_rows = 0, st_arc_load_rows = 0, st_arc_evict = 0;
    double st_arc_save_ms = 0, st_arc_load_ms = 0;
    template <class F> bool ck_missing(const Ckpt& c, F f) const;  // f(g, x, y) per range not live; false: one stopped
    bool ck_valid(const Ckpt& c);      // every range live or archived
    int arc_find(int g, int k) const;
    char* arc_block(unsigned long long hold);  // a free block (alloc within budget / host headroom, else LRU eviction)
    void arc_save(int x, unsigned long long floor = 0);  // archive live rows [x, lv_end) valid snapshots need; floor: pages used >= floor stay
    bool arc_load(const Ckpt& c);      // copy back the snapshot's ranges that are not live (false: a page is gone, nothing copied)
    void arc_touch(const Ckpt& c);
    void arc_gc();                     // pages no valid snapshot needs back to the free list
    void arc_trim();                   // host memory short: free blocks, then LRU pages
    size_t arc_give_free();            // free blocks back to the host while it is short (any thread)
    std::mutex arc_mu;                 // arc_free / arc_blocks (the watcher thread gives free blocks back)
    void ck_forget();                  // drop every snapshot and page (external reset)
    std::vector<Ckpt> ck;
    bool ck_on = true, ck_win = true;
    int ck_max = 16, ck_hint_min = 4096, ck_div_min = 4096, ck_tail = 0, ck_tail_min = 32768;
    unsigned long long ck_tick = 0, st_ck_save = 0, st_ck_load = 0;
    double st_ck_save_ms = 0, st_ck_load_ms = 0;
    std::vector<int> ck_dbg_cuts;      // check tool (glm_set_cuts): extra cuts of the next spec_start
    int sp_hint = 0;                   // next spec_start: server hint (end of the system part), 0 none
    bool mtp_prev = false;             // next forward: also the MTP cache row pos - 1 (hidden in mh, token tok[0])
    std::vector<std::pair<int, int>> ck_plan;  // next forward: (absolute position, entry) to save at window ends
    std::vector<float> ck_lout;
    void ck_init();
    int ck_take(int q, int kind);      // an entry to save into (free, or LRU victim; -1: none fits host RAM)
    void ck_drop_after(int q);         // entries past q are no longer backed by the cache rows
    int ck_best(const int* ids, int n, int lcp);  // valid entry with the largest q <= lcp (q == n only with logits)
    bool ck_restore(int i);            // false: its archived rows are gone (nothing changed but the entry)
    int ck_prefill(const int* ids, int n, float* lout);  // spec_start's resume + prefill; returns the reused length
    bool ck_inited = false, ck_log = false;
    void reset(bool keep_ck = false);  // keep_ck: snapshots / archive stay (ck_prefill's fresh start)
    // release track [glm-rel-slice]: time slicing between conversations (nseq 1). conv_park saves the bound reply's
    // whole state between two steps: a pinned snapshot at q = pos (KDA S + conv history, index rings, hidden; the MLA
    // rows stay in place and are archived only when another conversation overwrites them), the MTP cache row pos - 1
    // and its pool key (not part of the rows [0, pos) a snapshot covers), the host speculative state and the expert
    // cache state det_restore resets (slot table, free list, counts, victim rng, pending flags, MLA pages). conv_resume
    // puts all of it back (only the slots whose expert differs are copied), so the next step is the one the reply
    // would have made without the switch: the output is bit-identical to the reply run alone.
    struct ConvPark {
        unsigned long long tag = 0;    // its pinned snapshot entry (0: none)
        int pos = 0, sp_t = -1, sp_k = 0, sp_kcur = 0, sp_reused = 0, last_tok = -1, sp_last_a = 0;
        int sp_d[MR] = {};
        float sp_c[MR] = {};
        std::vector<int> sp_hist;
        std::vector<float> sp_lout;
        std::vector<int> owner, free_, pend, qprev;
        std::vector<float> score;
        std::mt19937 rng;
        std::vector<char> mla;
        int mla_rows = 0;
        bool dirty = false, lend_canon = false;
        std::vector<std::vector<char>> mtp_row;  // per card: MTP cache row pos - 1 (latent) + the pool key of its pool
    };
    bool conv_park(ConvPark& p);       // false: not possible now (nothing changed, the reply stays bound)
    int conv_resume(ConvPark& p);      // 1: resumed exactly; 0: its rows were lost (nothing changed; the caller re-prefills)
    void conv_drop(ConvPark& p);       // a parked reply ends without a resume
    void conv_keep(ConvPark& p);       // the same, its state kept as an answer-end snapshot
    void conv_quiesce();               // every copy / step of the bound sequence has landed
    int ck_find_tag(unsigned long long tag) const;
    unsigned long long park_tag = 0;
    unsigned long long st_sw_park = 0, st_sw_res = 0, st_sw_slots = 0, st_sw_lost = 0, st_sw_d2d = 0;
    double st_sw_park_ms = 0, st_sw_res_ms = 0;
    // c > 1 (two sequences): rows of both in one step, the wavefront schedule with one row group per sequence
    int nseq = 1, cur_seq = 0;
    std::vector<EngSeq> eseq;
    // release track R6 ([glm-rel-host]): host threads from the topology (load), the server's card list / split
    HostCpus hcpu;
    std::vector<int> want_devs;        // qe_create's devices ("0,1,2"; empty: 0..GLM_NGPU-1)
    std::vector<int> want_ecols;       // qe_create's split ("704,640,704"; empty: from the free VRAM at load)
    int* h_am2 = nullptr;
    void bind_seq(int s);
    void verify2(const int* ra, int ma, const int* rb, int mb, int* oa, int* ob);
    void spec_step2(int* oa, int* na, int* ob, int* nb);
    int generate2(const int* pa, int ta, const int* pb, int tb, int n, int k, int* outa, int* outb, int* got);
};

}  // namespace glm
