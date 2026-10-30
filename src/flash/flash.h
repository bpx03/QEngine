#pragma once
// Qwen3.8-Flash-Next (qwen4_exp) decode engine for 2 or 3 GPUs: EP experts, TP GDN / QSA heads, QSA KV split by
// token position across the cards (each card owns `own` of every `period` positions), hyper-connection
// residual 4 x 2560 replicated in fp32.
#include <string>
#include <vector>
#include <map>
#include <thread>
#include <atomic>
#include <algorithm>
#include <cstdio>
#include <cstdint>
#include <cstddef>
#include <cstring>
#include <cuda_fp16.h>

#include "../common.h"
#include "../kernels/comm.cuh"
#include "../kernels/gemv.cuh"
#include "../kernels/ops.cuh"
#include "../kernels/sample.cuh"
#include "../safetensors.h"
#include "graph_oom.h"

namespace flash {
struct Xchg;  // flash_kernels.cuh: a QSA exchange (receive buffers, flags, counters)
// block unions of the QSA prefill attention's row groups (flash_pfq.cuh k_qpf_union): U[g], ublk[g][QP_UMAX], um[g][QP_UMAX]
struct QpUnion {
    int* U = nullptr;
    int* ublk = nullptr;
    unsigned char* um = nullptr;
    unsigned* ctr = nullptr;  // persistent k_qsa_pfsel: [item fetch counter, exited blocks] (null: one item per block)
    int S = 0;                // its key splits
};

constexpr int H = 2560, HC = 4, RW = HC * H, LAYERS = 48, NE = 512, TOPK = 10, INTER = 640;
constexpr int VOCAB = 248320, NQ = 24, NKV = 2, HD = 256, ROT = 64;
constexpr int LNK = 16, LNV = 48, DK = 128, DV = 128;
constexpr int LOWRANK = 320, HCD_ROWS = 336, HCU_K = 384;  // Wd + 4 inject rows (pad 16), Wu K padded to g128
constexpr int IDX_NH = 4, IDX_HD = 128, IDX_BUDGET = 2048, IDX_R = 4, IDX_TOPB = IDX_BUDGET / IDX_R;
// raw indexer keys by position mod IDX_TAIL: a verify forward writes up to MAXP rows past the accepted end, so with a
// ring of 4 the keys of a rejected row (pos + 4) overwrote the accepted ones (pos) its block still needed (the block's
// pooled key went wrong for good); 32 >= 4 + MAXP keeps every key a pooling reads
constexpr int IDX_TAIL = 32;
constexpr int PLE_LAYER = 1, PLE_HEADS = 16, PLE_DIM = 160, PLE_CONV = 4, PLE_DIL = 3, NGRAM_ROWS = 2500012;
constexpr int PFM = 512;               // rows per prefill chunk
// rows of one prefill collective (all-reduce + HC on own rows + row all-gather) at most: a collective covers whole
// mixer chunks (FLASH_ARU, default 2 chunks), the MoE half runs per collective unit (prefill-r2)
constexpr int ARU_MAX = 2 * PFM;
constexpr int MAXM = 32;
constexpr int MAXP = 17;                // rows of a deferred-commit forward (verify: 1 + drafts); prompts use prefill()                // rows per forward (decode / verify / prefill chunks): 8-token mma tiles x 4
constexpr float EPS = 1e-6f;
constexpr double THETA = 1e7;
constexpr int EOS = 248044;
constexpr int IMAGE_PAD = 248056;  // <|image_pad|>: what the n-gram embedding (PLE) and the lookup see of an image row
inline int dev_tok(int t) { return t >= kImgToken ? IMAGE_PAD : t; }

// per-slot state pointer tables of the batched steps: S, conv history / pending, recurrence pending, PLE history /
// pending, indexer key ring
enum { TB_S = 0, TB_CH, TB_CP, TB_RP, TB_PH, TB_PP, TB_IT, TB_N };

// activation steering arguments of a kernel (see k_steer, flash_kernels.cuh)
struct SteerArg {
    const float* w = nullptr;    // [vectors][2 modes][LAYERS][H]
    const float* dw = nullptr;   // [vectors][2 modes][LAYERS][HC][HCD_ROWS]
    const int* sl = nullptr;     // per slot {vector (-1 off), mode, lo, hi}
    const float* sc = nullptr;   // per slot scale
    const int* rslot = nullptr;
    int li = -1;  // layer whose output is steered (-1: none)
};

// KV streaming (FLASH_KV_STREAM, flash_kernels.cuh k_kv_need): host rows of a layer's K / V + its page table
struct KvMap {
    const int* pt = nullptr;
    uint8_t *hk = nullptr, *hv = nullptr;
    float *hks = nullptr, *hvs = nullptr;
};
// a layer's block cache (VRAM) and its residency state
struct KvPool {
    int* pt = nullptr;                 // [logical blocks]
    unsigned long long* st = nullptr;  // [P]
    unsigned* ctl = nullptr;           // [0] call id, [1] clock hand, [2] finished blocks of the call, [3] misses, [4] fallbacks, [5] hits, [6] usable blocks
    int P = 0;
    uint8_t *kc = nullptr, *vc = nullptr;  // device block rows
    float *ks = nullptr, *vs = nullptr;
    KvMap m;
};
constexpr bool kKvRamDefault = false;  // FLASH_KV default (ram | vram)
bool kv_ram_mode();          // FLASH_KV / FLASH_KV_STREAM
long long kv_ram_tokens();   // FLASH_KV_RAM_TOKENS
// YaRN RoPE scaling (opt-in; model card: rope_type yarn, factor 4.0, original_max_position_embeddings 262144):
// FLASH_YARN=<factor> (0 / 1: off), else the factor the caller set for a context above the native one (context /
// 262,144; set_yarn_factor before the Workers are made). Contexts above 262,144 need FLASH_KV=ram.
constexpr int NATIVE_CTX = 262144;
void set_yarn_factor(double f);
double yarn_factor();        // 0: off
// ranks: 2 or 3 (every split below divides: 24 QSA q heads, 48 DeltaNet value / 16 key heads, 512-position KV
// ownership per period of 512 x ngpu positions)
struct Plan {
    int rank = 0, ngpu = 3;
    std::vector<int> gcard;  // expert group (e * 5 + g, e = 512 shared) -> card
    int v0 = 0, v1 = 0;     // DeltaNet value heads
    int k0 = 0, k1 = 0;     // DeltaNet key heads used by them
    int q0 = 0, q1 = 0;     // QSA q heads whose gate / o_proj live here (the q part of all 24 heads is everywhere)
    int voc0 = 0, voc1 = 0; // lm_head rows
    int period = 1536, own0 = 0, own = 512;  // KV positions p with (p % period) in [own0, own0 + own)
};
// weights: relative share of expert groups per card (e.g. 176,160,176)
std::vector<Plan> make_plan(int ngpu, const std::vector<double>& group_weights);
// default expert-group weights for ngpu cards: 3 -> 176,160,176 (the desktop card GPU1 gets less), else equal
std::vector<double> default_group_weights(int ngpu);
// local expert groups of a card under a plan (the shared expert's included): bound of its MoE job lists
int plan_groups(const Plan& p);

struct HcW {     // one hyper-connection read: Wd (+ inject rows) and Wu, INT8 g128
    W8 down, up;
    W16 down16, up16;       // debug precision switch FLASH_HC16=1 (bf16 weights)
    // fused path (default): down with the norm weight folded in (W8, K = 10240), up permuted + INT8 g64
    W8 fd;
    uint4* uq = nullptr;
    __half* us = nullptr;
    bf16* norm = nullptr;   // hc_norm weight [10240] (applied as 1 + w)
    bool inject = true;
};
// Experts are split TP-wise in 128-wide groups of the intermediate dimension (5 per expert, the shared expert is
// expert 512): a group is gate_up rows [gate 128g.. | up 128g..] (256 rows, GLU-interleaved per 16-row tile, K =
// 2560) + down columns [2560 x 128]. Groups are spread over the cards by a placement table (balanced load; the
// desktop card gets fewer), partial sums go into the layer's all-reduce.
constexpr int GRP = 128, NGRP = INTER / GRP;
constexpr size_t GG_Q = (size_t)2 * GRP * H / 2, GG_S = (size_t)2 * GRP * (H / 128) * 2;
constexpr size_t GD_Q = (size_t)H * GRP / 2, GD_S = (size_t)H * (GRP / 128) * 2;
constexpr size_t GRP_BYTES = GG_Q + GG_S + GD_Q + GD_S;
constexpr int MAXJOB = 896;              // job lists: at least this (3 cards: >= the groups of any card, 2565 x 0.34 + margin);
                                         // a card with more local groups (2 cards: ~1283) sizes them by plan_groups
constexpr int PF_SHJ = 64;               // prefill: rows per shared-expert job (the shared expert sees every row)
constexpr int MAXJOB_PF = 3072;  // prefill jobs: (expert, 64-row chunk, local group) <= 896 + 2 x rows x 11 / 64 (4096 rows)
struct MoeJob {            // device descriptor of one local expert group
    const uint4* gq; const __half* gs;
    const uint4* dq; const __half* ds;
};
// Dynamic expert cache (cold groups, see Worker::ec_*): the kernels resolve a cold local group ci (gmap index >= ndev)
// through a device pointer table, entry = device slot address, or host-mapped address | 1 (served zero-copy over
// PCIe); every read sets touch[ci] (host-mapped bytes: the host sees which cold groups each launch used).
struct ColdTab {
    const uintptr_t* ptr = nullptr;
    unsigned char* touch = nullptr;
    __host__ __device__ ColdTab(std::nullptr_t = nullptr) {}
    __host__ __device__ ColdTab(const uintptr_t* p, unsigned char* t) : ptr(p), touch(t) {}
};
#ifdef __CUDACC__
__device__ __forceinline__ const char* grp_ptr(const char* groups, int li, int ndev, const ColdTab& ct, unsigned* coldcnt) {
    if (li < ndev) return groups + (size_t)li * GRP_BYTES;
    const int ci = li - ndev;
    const uintptr_t p = ct.ptr[ci];
    ct.touch[ci] = 1;
    if ((p & 1) && coldcnt) atomicAdd(coldcnt, 1u);
    return reinterpret_cast<const char*>(p & ~(uintptr_t)1);
}
#endif
struct PfJob {             // prefill: one local expert group and its rows [off, off + n) of the token / weight lists
    MoeJob w;
    int off, n;
    int aoff;              // its rows of the activation buffer (each group of an expert has its own)
    int ng;                // first job of an expert: its local groups (consecutive jobs), else 0
};
struct Layer {
    bool full = false;
    HcW hca, hcm;
    // DeltaNet
    W4 dn_in, dn_out;      // in rows [q kl*128][k kl*128][v vl*128][z vl*128]
    W16 dn_ab;             // rows [b vl][a vl] (32)
    bf16* conv_w = nullptr;  // [ch][4]
    float *A = nullptr, *dtb = nullptr;
    bf16* dn_norm = nullptr;
    float* S = nullptr;    // state [vl][128][128] fp32 (k-major: S[k][v])
    float* conv_hist = nullptr;  // [3][ch] last committed inputs of the conv
    float* conv_pend = nullptr;  // [MAXM][ch] inputs of the last forward (deferred commit)
    float* rec_pend = nullptr;   // [2][MAXM][vl][DREC] recurrence inputs of the last forward
    // QSA
    W8 qkv, o;             // INT8; qkv rows [q 24 heads x 256][gate local heads x 256][k 2 x 256][v 2 x 256]
    W8 qx;                 // FLASH_QX_LOCAL: q rows of the peers' heads (owners ascending), prefill q without the exchange
    W16 idx;               // index_qk_proj [640, 2560]
    bf16 *q_norm = nullptr, *k_norm = nullptr, *iq_norm = nullptr, *ik_norm = nullptr;
    uint8_t *kc = nullptr, *vc = nullptr;  // fp8 e4m3 [rows][2][256], rows = owned positions
    float *ks = nullptr, *vs = nullptr;    // per (row, head) scales [rows][2]
    bf16* ikb = nullptr;   // indexer block keys of owned complete blocks [rows / 4][128]
    float* ik_tail = nullptr;  // raw indexer keys of the last positions [IDX_TAIL][128] by position mod IDX_TAIL (all cards)
    KvPool kvp;                // KV streaming: block cache + host rows (kc / vc / ks / vs are then the cache's rows)
    // MoE
    W16 router;            // [528, 2560]: 512 router rows, row 512 = shared_expert_gate, zero pad (BF16; FLASH_ROUTER8=0)
    W8 router8;            // the same INT8 g128 (default: KL at the BF16 floor)
    char* groups = nullptr;   // local expert groups, stride GRP_BYTES (device: indices < ndev)
    char* hgroups = nullptr;  // cold groups in mapped pinned host memory (indices >= ndev): the full host copy
    char* hgroups_h = nullptr;  // the same, host address
    int ndev = 1 << 30;
    int ncold = 0, coff = 0;  // cold groups of this layer, their first entry of the expert-cache table
    std::vector<int> cexp;    // expert of every cold group
    int* gmap = nullptr;      // device [513][5]: local group index or -1
    int* stats = nullptr;     // routing counts per expert (FLASH_STATS)
    // PLE (layer 1)
    W4 ple_key, ple_val;
    bf16 *ple_nk = nullptr, *ple_nq = nullptr, *ple_nc = nullptr, *ple_conv = nullptr;
    float* ple_hist = nullptr;  // [9][10240] normed gated values of the last 9 committed tokens
    float* ple_pend = nullptr;  // [MAXM][10240] of the last forward
};


struct NgramDev {          // n-gram table for the device-side gather (pinned + mapped host memory)
    const bf16* shard[128];
    const uint8_t* q8[128];  // INT8 table (FLASH_NGRAM=int8): rows of 160 int8 + fp16 scale (NGRAM_Q8_ROW bytes)
    int is8 = 0;
    // FLASH_NGRAM_STAGE=1 (table not pinned): the step's row ids go to the host (req, mapped), a host thread copies the
    // rows into the mapped staging buffer (stage, [MAXM * 16][NGRAM_Q8_ROW]) and the kernels read them from there
    long long* req = nullptr;
    const uint8_t* stage = nullptr;
    unsigned* flags = nullptr;  // [0] request seq (GPU), [1] done seq (host), [2] rows requested
    int phase = 0;              // set per launch: 0 gather from the table, 1 write row ids, 2 gather from stage
    long long mult[3];
    long long vsz[PLE_HEADS], voff[PLE_HEADS];
};
// Pinned + mapped host memory without cudaHostAlloc (2-3 GB/s, holds the driver lock): 2 MB-aligned THP memory,
// filled (src) or zeroed (src null) by `threads` threads, each page-locking its own chunk with
// cudaHostRegister(Mapped | Portable) as soon as it is filled (chunked: ~8-12 GB/s for the 102 GB n-gram file vs
// 2.7); chunked memory is for kernel access only (a cudaMemcpy across two registrations fails with invalid value),
// otherwise one registration of the whole buffer after the fill. Device address == host address (UVA). Never freed.
// numa: NUMA nodes for the pages (one: preferred, several: interleaved; empty or a single-node machine: default)
char* pin_host(const void* src, size_t n, int threads = 32, bool chunked = false, const std::vector<int>& numa = {});
// The same for n bytes of a file at off whose mapping is src (chunked registration): 8 MB blocks resident in the
// page cache are copied, the others read with O_DIRECT (no page-cache churn: the file is larger than what stays
// cached next to its pinned copy, and its reads evicted the expert weights read next); the file's cached pages are
// dropped afterwards (POSIX_FADV_DONTNEED). Returns the address of byte off (the buffer may start up to 4 KB earlier).
char* pin_host_file(const std::string& path, size_t off, size_t n, const void* src, int threads = 32,
                    const std::vector<int>& numa = {});

constexpr int NGRAM_Q8_ROW = PLE_DIM + 2;  // INT8 n-gram row: 160 int8, then the fp16 scale (x = q * scale)
struct HostTables {        // shared by the workers (host memory)
    const bf16* embed = nullptr;     // mapped pinned [vocab][2560]
    std::vector<const bf16*> ngram;  // 128 shards [2500012][160] (mmap)
    std::vector<const uint8_t*> ngram8;  // FLASH_NGRAM=int8: 128 shards [2500012][162] (tools/convert_ngram_int8.py)
    long long mult[3] = {0, 0, 0};
    long long vsz[PLE_HEADS] = {}, voff[PLE_HEADS] = {};
    // one table row -> out[160] (bf16), from the INT8 table when it is loaded
    void row(long long r, bf16* out) const {
        if (!ngram8.empty()) {
            const uint8_t* q = ngram8[r / NGRAM_ROWS] + (size_t)(r % NGRAM_ROWS) * NGRAM_Q8_ROW;
            __half_raw hr;
            hr.x = (unsigned short)(q[PLE_DIM] | (q[PLE_DIM + 1] << 8));
            const float sc = __half2float(__half(hr));
            for (int i = 0; i < PLE_DIM; i++) out[i] = __float2bfloat16((float)(int8_t)q[i] * sc);
        } else {
            memcpy(out, ngram[r / NGRAM_ROWS] + (size_t)(r % NGRAM_ROWS) * PLE_DIM, PLE_DIM * 2);
        }
    }
};

class Worker {
public:
    Worker(const Plan& p, int device, const Comm& comm, int max_ctx);
    void load(const SafeTensors& st, const HostTables& ht, const SafeTensors* mtp = nullptr);
    void alloc_state();             // KV pool for max_ctx, DeltaNet states, buffers
    size_t free_bytes() const;
    void test_hog(long long keep_mb);  // FLASH_TEST_VRAM_HOG_MB (flash.cu)
    std::vector<void*> hog_;
    void reset();                   // new sequence
    // m rows: tokens[i] at position pos + i; ple rows (16 n-gram embeddings per row, host-gathered) in ple_host.
    // Writes the argmax token of every row to out (host), logits of the last row optionally to logits_host.
    // commit: rows of the PREVIOUS forward that become state (DeltaNet, PLE conv); -1 = all of them
    void forward(const int* tokens, int pos, int m, const bf16* ple_rows, int* out, float* logits_host = nullptr,
                 int commit = -1);
    // prefill chunk of m <= PFM rows at pos (rows committed at once); argmax/sample of the last row -> *out;
    // mtp: also the MTP block over the rows (tokens shifted: mtp_toks[i] = t_{pos+i+1}) -> its draft in *draft
    void prefill(const int* tokens, int pos, int m, const bf16* ple_rows, int* out, const int* mtp_toks, int* draft);
    void prefill_mtp(const int* mtp_toks, int pos, int m, int* draft);  // MTP block over the last prefill chunk's rows
    // batched prefill of n <= pf_batch() rows: layer by layer, the mixer (attention / DeltaNet) in chunks of PFM rows,
    // the MoE half over all n rows at once (each expert group read once per batch); argmax/sample of the last row
    // ctx_end: the absolute context end of the coming batches (0: the sequence regions, kv_top_); rows past
    // pfbn_mid_ live in the tails of the KV arrays (al_*), usable while the KV rows in use stay below them
    int pf_batch(long long ctx_end = 0) const { return pfbn_ > pfbn_mid_ && !al_room(ctx_end) ? pfbn_mid_ : pfbn_; }
    // rows of a prefill batch whose buffers never take memory from the expert cache (the rest of pf_batch() is
    // borrowed from it for the batch, prefill-r3)
    int pf_batch_base() const { return pfbn_base_; }
    // rows of the next prefill batch with r rows left and batches of up to B rows: nominal B - PFM (a tail of <= PFM
    // rows joins the last one); with big batches (B > pf_batch_base()) the last two are balanced when the plain split
    // would leave a tail of <= pf_batch_base() rows and both halves can be big: a small
    // batch stages only the base part of the cold groups (no ring extension) and runs the MoE over the whole batch
    // (2 GPUs 10K: 4096 + 4096 + 2105 -> 4096 + 3584 + 2617); only with a ring extension (pfs_ext_, L1)
    int pf_batch_rows(int r, int B) const {
        if (r <= B) return r;
        const int nb = B - PFM;
        if (pfs_ext_ && B > pfbn_base_ && r - nb <= pfbn_base_) {  // (3 GPUs: no extension, 10K +1.5 % when balanced)
            const int m = ((r + 1) / 2 + PFM - 1) / PFM * PFM;
            if (m <= nb && r - m > pfbn_base_) return m;
        }
        return nb;
    }
    // mtp_toks (optional): the MTP fill as a 49th layer of the batch (last: the prompt's last batch, the last row's MTP
    // token is the sampled one); *draft = the MTP draft of the last row, or -2 when the caller must run prefill_batch_mtp
    void prefill_batch(const int* tokens, int pos, int n, const bf16* ple_rows, int* out, const int* mtp_toks = nullptr,
                       bool last = false, int* draft = nullptr);
    void prefill_batch_mtp(const int* mtp_toks, int pos, int n, int* draft);  // MTP block over the batch's rows
    // several prompts in one prefill batch (batched-prefill): segment j = rows [pos, pos + n) of slot `slot` (KV region at
    // period `base`; `commit` pending rows of its last forward committed first; `end`: its prompt ends here, so its last
    // row gets the head (sampled with the slot's parameters) and, with mtp_toks, the MTP draft). Rows of the segments are
    // concatenated in tokens / ple_rows / mtp_toks; the mixer runs per segment (its own positions, KV region and recurrent
    // state), HC and the MoE half over all rows at once (each expert group read once for every prompt). out[j] / draft[j]
    // for the segments with end set, in segment order. The workspace keeps the bound slot's state.
    struct PfSeg { int slot, base, pos, n, commit; bool end; };
    static constexpr int PFSEG_MAX = 16;
    bool pf_multi_ok(const int* segn, int nseg) const;  // the layout fits (units, collectives, buffers)
    // host-backed (cold) expert bytes of this card: every prefill pass reads / stages them over PCIe
    long long cold_bytes() const {
        long long n = 0;
        for (const Layer& L : L_) n += L.ncold;
        return n * (long long)GRP_BYTES;
    }
    void prefill_multi(const PfSeg* sg, int nseg, const int* tokens, const bf16* ple_rows, const int* mtp_toks, int* out,
                       int* draft);
    void rmin_load(int slot);  // the MTP chain residual of slot's prompt end (prefill_multi) into the chain input
    // MTP head over m rows at positions pos..: row i = (hidden, tokens[i]); chain = false: hidden rows of the last
    // trunk forward (rows 0..m-1), true: the MTP residual of the last row of the previous MTP call (m = 1).
    // out[i] = draft token predicted by row i.
    void forward_mtp(const int* tokens, int pos, int m, bool chain, int* out);
    bool has_mtp() const { return has_mtp_; }
    // device-side speculative decoding (one graph per step): spec_init with the state after the prompt (last
    // emitted token, k drafts, its position, rows of the last forward to commit, the two committed tokens before it)
    void spec_init(int last_tok, const int* drafts, int k, int pos, int commit, int h1, int h0);
    int spec_step(int k, int* emitted, bool lookup = false);  // returns the number of emitted tokens (1..k + 1)
    // the same in two halves (lookahead: the next step is launched before the host handles this one's tokens)
    void spec_launch(int k, bool lookup = false);
    // grammar-constrained decoding: the bound slot verifies kv caller drafts (upload_drafts) with a mask per row
    // (upload_mask: [rows][MASK_WORDS] bits, row j = allowed after the pending token and drafts 1..j) and drafts kd
    void set_constrained(int slot, bool on);
    bool constrained() const { return con_; }
    void upload_mask(const unsigned* host, int rows);
    void upload_drafts(const int* d, int n);
    void spec_launch_con(int kv, int kd);
    static constexpr int MASK_WORDS = (VOCAB + 31) / 32, MASK_ROWS = 8;
    int spec_finish(int* emitted);
    void spec_discard();
    // lookup drafts (FLASH_LOOKUP=1): the token history (prompt + emitted, incl. the pending token) before the first
    // step; after every step lookup_drafts() = the drafts found on the device for the next one (0: none)
    void set_history(const int* ids, int n, int kfull);  // kfull: drafts of an MTP step
    void corpus_add(const int* ids, int n);  // a finished answer into the lookup corpus (between requests)
    int lookup_drafts() const { return lk_on_ ? em_h_[MAXM + 2] : 0; }
    int lookup_match() const { return lk_on_ ? em_h_[MAXM + 2 + 31] : 0; }
    bool lookup_on() const { return lk_on_; }
    void set_ngram(const NgramDev& n) { ngp_ = n; }
    // table not pinned (FLASH_NGRAM_PIN=0): the step's rows go through a mapped staging buffer filled by a host thread
    void start_ngram_stage(const HostTables* ht);
    // temperature <= 0: greedy; else sampled verification rows / first token (same seed on every card)
    void set_sampling(float temperature, float top_p, int top_k, float min_p, unsigned long long seed);
    // presence / frequency penalty of the bound slot (after set_sampling, which turns it off); its counts restart
    // with the distinct seed tokens. pen_host: host-known tokens of the bound slot at positions pos.. (first token,
    // steps without drafts)
    void set_penalty(float presence, float frequency, const int* seed, int n);
    void pen_host(const int* toks, int n, int pos);
    // per slot: lookup drafts off (device flag read by the batched lookup pick; the single path asks the host)
    void set_no_lookup(int slot, bool off);
    // multimodal prompts (Qwen M-RoPE, see rope_freq in flash_kernels.cuh): the RoPE position shift of slot s (text
    // after images: cache index + delta), and the (t, h, w) positions of a prefill chunk's rows [i0 - 3, i0 + m) of
    // slot s (r3: 3 ints per prompt row; null: slot s has no table, e.g. a text prompt after an image one). The image
    // embedding table (host-mapped, all cards): rows of ids >= kImgToken (set_image_table).
    void set_rope_slot(int s, int delta);
    void set_rope_chunk(int s, int i0, int m, const int* r3);
    void set_image_table(const bf16* t);
    void set_prev_m(int n) { prev_m_ = n; }
    void set_cold(const std::vector<std::vector<int>>& c) { cold_ = c; }
    // per-token routing probability [48][512] (router trace): cold groups seed the expert cache hottest first
    void set_expert_prior(const std::vector<float>& f) { ec_prior_ = f; }
    // dynamic expert cache: cold groups cached in the unused tail of the KV pool (see ec_tick)
    struct EcStats {
        long long dec_steps = 0, dec_touch = 0, dec_miss = 0;  // decode launches, cold-group reads, of them from host
        long long pf_calls = 0, pf_touch = 0, pf_miss = 0;     // prefill / other launches
        long long copies = 0, ev_kv = 0, ev_freq = 0, revived = 0, ev_kv_pend = 0;  // ev_kv_pend: reclaimed with a copy in flight
        int ncold = 0, resident = 0, live_slots = 0, slots = 0, bt_slots = 0;  // bt_slots: of them in the borrowed big-batch tails
        double seed_gbs = 0;   // H2D bandwidth of the startup seed copy
        std::vector<unsigned short> step_miss;  // per decode launch (this card)
    };
    const EcStats& ec_stats();   // collects the touches of the last launch first
    void ec_reset_stats();
    // debug (FLASH_EC_VERIFY=1): every resident expert-cache slot and the staging ring's last two layers checked against
    // their host groups (checksums); prints the mismatches. Returns their count.
    int ec_verify(const char* where);
    // debug (FLASH_GSUM=1): placement-independent weight hashes of this card: every routed expert group by identity
    // (layer, expert, group) wherever it lives (device or host-backed), and the non-expert weight allocations [0, nw)
    void gsum(int nw, unsigned long long* experts, unsigned long long* other);
    void set_draft_vocab(const std::vector<int>& ids) { draft_ids_ = ids; }
    double host_gb() const { return host_bytes_ / 1e9; }
    void get_stats(std::vector<int>& out);
    unsigned cold_hits();
    std::vector<float> gpu_ms_;  // GPU time of every speculative step graph     // cold expert groups routed on this card so far  // [48 + MTP][512] routing counts
    void dump_resid(float* host, int m);  // R after the last forward ([m][10240])
    int device() const { return dev_; }
    unsigned long long checksum(int first, int last);  // debug: of allocations [first, last)
    int nallocs() const { return (int)allocs_.size(); }
    float* dump_ = nullptr;
    // FLASH_RDUMP (debug, FLASH_GRAPH=0): top-10 experts of every prefill row per layer -> rdump_[layer][pos][10]
    std::vector<int> rdump_;
    int rd_pos_ = 0;
    // attention exchange (P2P): this card's receive buffer / flags, and the peers' (set after all workers exist)
    float* xrecv() const { return xrecv_; }
    unsigned* xflags() const { return xflags_; }
    void set_peer(int g, float* recv, unsigned* flags) { xpr_[g] = recv; xpf_[g] = flags; }
    void set_agpeer(int g, const Worker* w);  // sequence-parallel HC prefill: the peer's x / xh / R batch rows and flags
    float* xqrecv() const { return xqrecv_; }
    unsigned* xqflags() const { return xqflags_; }
    void set_qpeer(int g, float* recv, unsigned* flags) { xqpr_[g] = recv; xqpf_[g] = flags; }
    float* xirecv() const { return xirecv_; }
    unsigned* xiflags() const { return xiflags_; }
    void set_ipeer(int g, float* recv, unsigned* flags) { xipr_[g] = recv; xipf_[g] = flags; }   // debug: host [48][m][10240] residual after every layer of the next forward
    double weight_gb() const { return wbytes_ / 1e9; }

    // ---- sequence state (flash_seq.cu): slot switching, host-RAM archive, prefix reuse. Device ranges in a fixed
    // order; the caller copies them on stream() (cudaMemcpyAsync) and synchronizes.
    struct Piece { char* p; size_t n; };
    int kv_rows(int ntok) const;  // owned KV rows of the positions [0, ntok) on this card
    // KV of the positions [t0, t1) (trunk QSA layers + MTP: fp8 K/V rows, their scales, pooled indexer keys)
    void kv_pieces(int t0, int t1, std::vector<Piece>& out);
    // recurrent state of the sequence: DeltaNet S + conv history, PLE history, indexer raw-key rings; pend: also the
    // deferred-commit buffers of the last forward (conv / recurrence / PLE inputs) and the step counter (parity)
    void state_pieces(bool pend, std::vector<Piece>& out);
    void kv_claim(int ntok) { CUDA_CHECK(cudaSetDevice(dev_)); ec_tick(ntok, 1); }  // expert-cache slots over rows < ntok -> KV
    cudaStream_t stream() const { return st_; }
    int prev_m() const { return prev_m_; }
    // sequence slots (FLASH_MAX_SEQS): recurrent state per slot, KV of slot s in a region of the one pool starting at
    // period `base` (rows base * own on every card). use_slot points the layers (and the graph caches) at slot s;
    // kv_top: periods in use by all regions (the expert cache stays above them).
    int num_slots() const { return (int)slots_.size(); }
    int periods() const { return rows_ / p_.own; }
    int period() const { return p_.period; }
    int cur_slot() const { return cur_slot_; }
    void use_slot(int s, int base);
    void set_kv_top(int periods) { kv_top_ = (long long)periods * p_.period; kv_host_ensure(periods, false); }
    // KV in RAM: positions of the regions of the active sequences (generating / prefilling; -1: unknown = all regions).
    // The block cache is sized to them, not to the pool top: an idle conversation's host rows need no cache entries
    // (a 262K conversation idle under t8 kept 1.3 GB of entries away from the expert cache)
    void set_kv_active(long long pos) { kv_act_ = pos; }
    // KV in RAM: periods the pool may use now (pinned + what the host can still pin, at most periods())
    int kv_ram_periods() const;
    void reset_slot(int s);  // zero the recurrent state of slot s
    // instantiate (graph_oom.h): on OOM drop every cached graph and retry; false = no graph, the caller runs the
    // captured work eagerly (it never ran)
    bool inst_graph(cudaGraphExec_t* e, cudaGraph_t g);
    bool launch_graph(cudaGraphExec_t& e);  // false: out of memory, every graph dropped, run eagerly (flash_seq.cu)
    int glaunch_calls_ = 0;
    int drop_all_graphs(const cudaGraphExec_t* keep);  // every cached graph but *keep; map entries stay (= seen)
    bool graph_skip() { return gcool_.skip(); }        // a missing graph is not captured now (after a failed retry)
    int slot_prev_m(int s) const { return s == cur_slot_ ? prev_m_ : sprev_.at(s); }  // pending commit rows of slot s
    int rope_shift(int s) const { return mmdel_h_.empty() ? 0 : mmdel_h_.at(s); }  // multimodal RoPE shift of slot s
    void set_slot_prev_m(int s, int v) { if (s == cur_slot_) prev_m_ = v; else sprev_.at(s) = v; }
    // KV rows of the positions [t0, t1) of the region at period `base` (kv_pieces: the current slot's region)
    void kv_pieces_at(int base, int t0, int t1, std::vector<Piece>& out);
    void state_pieces_of(int s, bool pend, std::vector<Piece>& out);
    // KV of positions [0, ntok) from the region at period `from` to the one at `to` (D2D, chunks in ascending order:
    // safe when moving down)
    void kv_move(int from, int to, int ntok);
    // batched speculative steps over several slots (flash_batch.cuh): segment i = slot slots[i] (KV region at period
    // bases[slot] for every slot), k drafts each, at most lims[i] of them accepted; out[i * 64 ..] / cnt[i] per segment
    void batch_launch(const int* slots, const int* bases, const int* lims, int nseg, int k);
    void batch_finish(int* out, int* cnt);
    void sv_load();          // the bound slot's speculative state from its stash (after batched steps / a switch)
    void ws_ensure();        // the bound slot's recurrent state into the workspace (after batched steps)
    void sv_set_live() { sv_live_ = true; }  // spec_init wrote the bound slot's state
    // teacher-forced evaluation (flash_run FLASH_TF): this card's logits slice of the last forward's m rows (permuted
    // order) and its row -> token map
    int vslice() const { return p_.voc1 - p_.voc0; }
    void copy_logits(float* host, int m) {
        CUDA_CHECK(cudaSetDevice(dev_));
        CUDA_CHECK(cudaMemcpy(host, logits_, (size_t)m * vslice() * 4, cudaMemcpyDeviceToHost));
    }
    void copy_lmap(int* host) {
        CUDA_CHECK(cudaSetDevice(dev_));
        CUDA_CHECK(cudaMemcpy(host, lmap_, (size_t)vslice() * 4, cudaMemcpyDeviceToHost));
    }
    int next_drafts(int slot, int k, int* out);  // the drafts slot's next step verifies (MTP drafts; device read)
    void set_bhist(int slot, const int* ids, int n);  // batched lookup: the slot's token history (prompt + emitted)
    bool batch_lookup() const { return blk_; }
    std::vector<float> bgpu_ms_;  // GPU time of every batched step
    // activation steering (QE_STEER / QE_STEER_DIR, flash_steer.h): nv vectors, w = [nv][2 modes][LAYERS][H] with the
    // sidecar strength folded in (add: alpha v; project: sqrt(alpha) unit v), vector v present on layers flo[v]..fhi[v];
    // applied to every HC stream of the residual at the output of the slot's layers. Set once after alloc_state,
    // before any forward (graphs capture the buffer addresses).
    void set_steer(int nv, const int* flo, const int* fhi, const float* w);
    bool steer_loaded() const { return steer_w_ != nullptr; }
    // per slot: the vector (-1: off, the default), its mode (0 add, 1 project), layers lo..hi (inside the vector's) and
    // scale (device values at fixed addresses: no graph changes)
    void set_steer_slot(int s, int vec, int mode, int lo, int hi, float scale);
    int steer_vec(int s) const { return steer_w_ ? steer_sl_h_.at((size_t)s * 4) : -1; }
    // activation capture (qe_capture, eager prefill): per layer the sum over rows >= skip of the layer output's stream
    // mean; cap_begin zeroes, cap_skip sets the rows to skip in the next forward, cap_end copies out [LAYERS][H]
    void cap_begin();
    void cap_skip(int skip) { cap_skip_ = skip; }
    void cap_end(float* host);
    void selwin_report();  // FLASH_SELWIN: selected blocks vs recent-position windows / a recency cache (stderr)
    void kvs_report();  // KV streaming: block cache hits / misses / host fallbacks (stderr)
    bool kv_stream() const { return kvs_; }
    bool ec_enabled() const { return ec_on_; }  // dynamic expert cache on (cold groups exist, FLASH_EC != 0)
    // KV streaming: the host rows of the region at period `base`, positions [t0, t1) changed (archive restore, moves):
    // drop their cached blocks in every layer (stream order)
    void kv_inval(int base, int t0, int t1);

private:
    struct SlotState {
        std::vector<float*> S, conv_hist, conv_pend, rec_pend, ple_hist, ple_pend, ik_tail;  // per layer (+ MTP ik_tail)
        unsigned* stepc = nullptr;
    };
    struct GraphSet {
        cudaGraphExec_t g[2][2][MAXM + 1] = {}, mg[2][2][MAXM + 1] = {}, sg[2][2][2][MAXM] = {};
        std::map<long long, cudaGraphExec_t> pg, pbg;
    };
    struct KvBase { uint8_t *kc, *vc; float *ks, *vs; bf16* ikb; const KvMap* hm; };  // hm: streamed rows (host)
    std::vector<SlotState> slots_;
    std::vector<GraphSet> gstash_;
    std::vector<int> gbase_, sprev_;
    unsigned* mask_d_ = nullptr;       // constrained: masks of the bound slot's step (device)
    bool con_ = false;                 // the bound slot is grammar-constrained
    std::vector<char> con_slot_;
    std::map<long long, cudaGraphExec_t> cgraph_;  // constrained step graphs (sampling, selection, kv, kd)
    SampleParams* spslot_ = nullptr;  // per slot (device): the bound slot's live copy is sp_
    std::vector<char> sson_;          // per slot: sampling on (temperature > 0)
    int* sson_d_ = nullptr;           // the same flags on the device (batched sampling)
    bool bsamp_ = false;              // the batched step being built samples (some slot has temperature > 0)
    PenState* pen_ = nullptr;         // per slot penalty counts (pen_state_bytes(VOCAB) each)
    PenState* pen_of(int s) const { return (PenState*)((char*)pen_ + (size_t)s * pen_state_bytes(VOCAB)); }
    int* nolk_d_ = nullptr;           // per slot: 1 = no lookup drafts (batched steps)
    static constexpr int kPenSeedMax = 8192;
    int* pseed_d_ = nullptr;          // set_penalty's seed tokens (device, kPenSeedMax)
    bool vchain_ = false;             // trunk rows are verify rows of one sequence (penalty: rows see earlier rows)
    void head_sample_b(int m, int* dst, const int* rs = nullptr);  // rs: slot of every row (default rslot_)
    std::vector<unsigned long long> guse_;  // per slot: last use of its graph set (LRU bound FLASH_SLOT_GRAPHS)
    unsigned long long gtick_ = 0;
    std::vector<KvBase> kvb_;   // slot-0 (pool start) KV pointers per layer, MTP last
    int cur_slot_ = 0, cur_base_ = 0;
    long long ctx_off_ = 0, kv_top_ = 0, kv_act_ = -1;  // expert cache: positions of the current region's start / of all regions
    void alloc_slots();
    void bind_slot(int s, int base);
    int* rbcur_ = nullptr;   // [XROWS] KV region start row of the bound slot (read by the single-sequence kernels)
    // multimodal RoPE (MmPos, flash_kernels.cuh): {bound slot, its shift}, shift per slot, chunk table (+ host copies)
    int *mmcur_ = nullptr, *mmdel_ = nullptr, *mmtab_ = nullptr;
    int mmtab_rows_ = 0, mmtab_slot_ = -1;
    std::vector<int> mmdel_h_;
    void mm_init(int ns);
    bool ws_live_ = false;   // the workspace holds the bound slot's state (else its storage does)
    SlotState ws_state() const;
    void pieces_of(const SlotState& q, bool pend, std::vector<Piece>& out) const;
    void ws_copy(int s, bool to_storage);
    void ws_save();
    // batched steps
    float** tab_ = nullptr;          // device pointer tables [TB_N][LAYERS + 1][slots] of the per-slot state
    std::vector<int> kvi_;           // layer (MTP = LAYERS) -> kvb_ index
    unsigned* stepcs_ = nullptr;     // [slots] step counters (rec_pend parity)
    int *svt_ = nullptr, *svp_ = nullptr, *svh_ = nullptr;  // per slot: next verify tokens [MAXM + 1], {pos, commit}, committed pair
    std::vector<int> svpos_;         // per slot: host position of the next verify row
    bool sv_live_ = false;           // vtok_ / posd_ / hist2_ hold the bound slot's state (else its stash row does)
    int *kvrb_d_ = nullptr, *kvrb_h_ = nullptr;  // [slots] KV region start row
    int *bseg_d_ = nullptr, *bseg_h_ = nullptr;  // [2][MAXM] segment slots, accept limits
    int *rpos_ = nullptr, *rbase_ = nullptr, *rslot_ = nullptr, *bvt_ = nullptr, *bpn_ = nullptr;  // per row / segment
    int *chpos_ = nullptr, *chbase_ = nullptr, *bacc_ = nullptr, *bdr_ = nullptr;
    float* Rminb_ = nullptr;         // [MAXM][RW] MTP chain inputs per segment
    int *bem_h_ = nullptr, *bem_ = nullptr;  // host-mapped [MAXM][MAXM + 2] emitted tokens per segment
    std::map<long long, cudaGraphExec_t> bgraphs_;
    bool blk_ = false;               // FLASH_BATCH_LOOKUP: lookup drafts in batched steps
    int blk_min_ = 8;                // shortest match used (FLASH_BATCH_LK_MIN)
    int *hdb_ = nullptr, *hnb_ = nullptr, *blkcnt_ = nullptr;  // per-slot histories [slots][max_ctx + 64], lengths
    unsigned long long* lkbb_ = nullptr;  // [MAXM] best match per segment
    int bnseg_ = 0;
    std::vector<int> bslots_;
    float* const* tab(int kind, int li) const;
    void sv_stash();
    void ple_b(Layer& L, int li, int m, int rps);
    void gdn_b(Layer& L, int li, int m, int rps);
    void qsa_b(Layer& L, int li, int m, const int* rp, const int* rb, const int* rs, int rps);
    void trunk_b(int m, const int* toks, int nseg, int rps);
    void bdump(int li, int part, int m, int rps);
  public:
    void moe_dn_1w();
  private:
    void mtp_core_b(int m, bool chain, const int* toks, int* out, const int* rp, const int* rb, const int* rs, int rps);
    void batch_body(int nseg, int k);
    void* dalloc(size_t bytes);
    void body(int m);
    void body_mtp(int m, bool chain);
    void head(int m, int* dst, bool draft = false);
    void head_sample(int m, int* dst);
    void trunk(int m, const int* toks);
    void mtp_core(int m, bool chain, const int* toks, int* out, const int* last_row);
    void spec_body(int k, bool lookup = false, int kd = -1);  // kd >= 0: draft chain length (constrained steps)
    void lk_chain();
  public:
    // guard: rows of the last launch whose lm_head input had non-finite values (bit r = row r), logged; also
    // accumulated for take_nf (FLASH_GUARD: 1 the caller fails those sequences, 2 log only, 3 abort, 0 off)
    unsigned nf_check(const char* what, int pos);
    unsigned take_nf() { const unsigned m = nf_pending_; nf_pending_ = 0; return m; }
    int batch_rps() const { return brps_; }
    void dump_ikb(const char* path, int nb);
  private:
    void alloc_full(Layer& L);
    // expert cache
    void load_groups(const SafeTensors& st, Layer& L, const std::string& pre, const std::vector<char>& isc);
    void load_deferred_groups();
    struct DeferL { const SafeTensors* st; std::string pre; int li; };
    std::vector<DeferL> ec_defer_l_;
    bool ec_defer_ = false;
    void ec_init();
    void ec_tick(int ctx_end, int kind);  // before every launch: kind 0 decode step, 1 prefill / other
    void ec_collect();
    ColdTab ctab(const Layer& L) const { return ColdTab(ec_tab_d_ ? ec_tab_d_ + L.coff : nullptr, ec_touch_d_ ? ec_touch_d_ + L.coff : nullptr); }
    // prefill staging of cold groups (own ring pfs_ring_): the copy engine copies a layer's host-backed groups into one half
    // of a device ring two layers ahead; the prefill MoE of that layer reads them through pfs_tab_d_ (ring addresses)
    // sequence-parallel HC in prefill (hcsp_ok): each card runs the hyper-connection reads only on its own rows of
    // every chunk (the rows its all-reduce reduces) and all-gathers x / xh, instead of HC on all rows on every card
    char* agdst_[3][4] = {};      // [x, xh, R][rank] batch-row arrays of every card
    unsigned *agflags_ = nullptr, *agseq_ = nullptr, *agscnt_ = nullptr;  // [2 channels][4 senders], [2], [2][4]
    unsigned* agpf_[4] = {};      // every card's agflags_
    float *sp_hu_ = nullptr, *sp_hss_ = nullptr, *sp_rinv_ = nullptr;
    __half* sp_hh_ = nullptr;
    unsigned long long* sp_det_ = nullptr;
    void hc_rows(const HcW& w, float* R, float* inj, bf16* x, __half* xh, int m, cudaStream_t s);
    // ch: protocol channel (one per stream); rr: [2 * rank] row range of every card (needed by the host-staged path)
    void row_ag(unsigned amask, int r0, int r1, cudaStream_t s, int ch, const int* rr = nullptr, int rowb0 = 0);
    // staged prefill transfers (comm_.xfer != kXferP2P, comm_select_path): pinned host staging of the row all-gather
    // ([x, xh, R] regions of AG_CAP own rows) and of the QSA exchanges ([4 peers][XROWS][XPER] for q and partials), the
    // host-mapped flag words, the device outbox of the staged exchanges, and the exchange counters of the batch
    static constexpr int AG_CAP = ARU_MAX / 2 + 2;
    char* hag_h_ = nullptr;
    unsigned* hagf_ = nullptr;          // [2 channels][data, ack][4 senders]
    char* hag_p_[4] = {};
    unsigned* hagf_p_[4] = {};
    float* xouti_ = nullptr;  // CE: device outbox of the indexer-key exchange
    int xi_k_ = 0;            // indexer exchanges of the batch so far (staged slot parity)
    float *xout_ = nullptr, *xoutc_ = nullptr, *hxq_ = nullptr, *hxc_ = nullptr;  // device outboxes of q / partials
    unsigned *xsw_ = nullptr, *agw_ = nullptr;  // CE: sequence-number words copied into the peers' flags after the rows
    unsigned* hxf_ = nullptr;           // [q, partials][4 senders]
    float *hxq_p_[4] = {}, *hxc_p_[4] = {};
    unsigned* hxf_p_[4] = {};
    unsigned* xdummy_ = nullptr;        // flag words the send kernels of a staged exchange raise (the copies follow)
    bool xstage_ = false;               // inside a prefill batch (counters aligned by k_xpar_align)
    int xq_k_ = 0, xc_k_ = 0;           // QSA exchanges of the batch so far
    void alloc_staging();
    void xchg_send(int e, Xchg& x, int par, int mr, size_t rowb, size_t per, cudaStream_t sx = nullptr,
                   const unsigned* flagw = nullptr);  // flagw: the flag value precomputed (pipelined partials, CE)
    cudaStream_t xcs_ = nullptr;                  // CE: copy stream of the pipelined partial exchange (FLASH_QSA_PIPE_CE)
    cudaEvent_t xcf_ = nullptr, xce_[2] = {};     // its fork event, copy-done event per outbox parity
    bool xce_rec_[2] = {};                        // recorded in this batch (waits / joins only on those: graph capture)  // after the send kernel (into the outbox)
    Xchg xchg_outbox(int e, const Xchg& x, int par, size_t per) const;        // the send kernel's view: outbox + dummy flags
    void trunk_batch_sp(int n, unsigned selmask);
    int pf_units(int n, int* ub, int* cb) const;
    bool hcsp_ok(int n) const;
    char* pfs_ring_ = nullptr;          // [2][pfs_cap_] groups (null: no staging)
    struct PfsRun { int p0, n; char* addr; };
    std::vector<PfsRun> pfs_runs_[2];   // contiguous slot runs of each ring half (one copy per run)
    bool pfs_now_ = false;
    int pfs_cap_ = 0;                   // groups per ring half (layers with more stage the first pfs_cap_, rest zero-copy)
    // ring extension (FLASH_PF_STAGE_XMB, prefill-compute): ring positions [pfs_cap0_, pfs_cap_) of each half live in
    // pfs_ext_, expert-cache slots between big batches (a bt region, as the big-batch buffer tails); a batch without
    // them stages the first pfs_cap0_ groups (pfs_tab0_d_) and reads the rest zero-copy as before
    int pfs_cap0_ = 0;
    char* pfs_ext_ = nullptr;
    bool pfs_ext_lent_ = false;         // the extension is cache slots outside big batches
    bool pfs_full_ = false;             // this batch stages pfs_cap_ groups per layer (else pfs_cap0_)
    // prefill batches from this position on run the QSA attention per row (k_qsa_dec) instead of k_qsa_pfsel
    static constexpr long long kPfselMax = 8192;
    bool pfdec_ = false;                // this prefill batch's QSA attention per row (k_qsa_dec), kPfselMax
    uintptr_t* pfs_tab0_d_ = nullptr;   // the table with pfs_cap0_ staged groups per layer
    int pfs_capn() const { return pfs_full_ ? pfs_cap_ : pfs_cap0_; }
    char* pfs_addr(int half, int p) const {
        return p < pfs_cap0_ ? pfs_ring_ + ((size_t)half * pfs_cap0_ + p) * GRP_BYTES
                             : pfs_ext_ + ((size_t)half * (pfs_cap_ - pfs_cap0_) + (p - pfs_cap0_)) * GRP_BYTES;
    }
    long long pfs_resv_ = 0;            // bytes kept free for the ring by the expert budget
    uintptr_t* pfs_tab_d_ = nullptr;    // per cold group: its ring address (or host address | 1 past pfs_cap_)
    unsigned char* pfs_touch_d_ = nullptr;
    cudaStream_t dst_ = nullptr;
    cudaEvent_t pfs_cp_[2] = {}, pfs_used_[2] = {};
    bool pfs_use_ = false;              // set while a staged prefill batch is being built (moe() picks pfs_tab_d_)
    ColdTab stab(const Layer& L) const { return ColdTab((pfs_full_ ? pfs_tab_d_ : pfs_tab0_d_) + L.coff, pfs_touch_d_ + L.coff); }
    void pfs_init();
    void pfs_copy(int li);
    struct EcSlot { char* addr; size_t off; int ci; cudaEvent_t ev; bool live; bool bt = false; int prev = -1; bool ik = false; bool al = false; };  // al: a KV-array tail that is also mapped as big-batch rows (al_*)  // ik: in the tail of an indexer-key array (KV streaming)  // bt: a tail of the big-batch buffers (prev: its group when taken)  // pf: lent to the prefill staging ring
    std::vector<float> ec_prior_;
    std::vector<EcSlot> ec_slots_;
    std::vector<int> ec_slot_of_;      // cold group -> slot or -1
    std::vector<int> ec_cand_;         // copy candidates (missed groups of the last launch)
    std::vector<char> ec_pend_;        // copy in flight
    std::vector<double> ec_ls_;        // decayed access count, log domain: log(score) - t * log(decay)
    std::vector<uintptr_t> ec_host_;   // device-visible host address of every cold group
    std::vector<const char*> ec_src_;  // host address (copy source)
    std::vector<unsigned long long> ec_hsum_;  // FLASH_EC_VERIFY: host checksum per cold group (0: not computed yet)
    unsigned long long* ec_vsum_d_ = nullptr;   // FLASH_EC_VERIFY: device checksums [slots + ring]
    uintptr_t* ec_tab_d_ = nullptr;    // device table
    uintptr_t* ec_tab_h_ = nullptr;    // pinned mirror
    unsigned char *ec_touch_h_ = nullptr, *ec_touch_d_ = nullptr;
    cudaStream_t ec_cs_ = nullptr;
    cudaEvent_t ec_evt_ = nullptr;
    bool ec_on_ = false, ec_dirty_ = false;
    int ec_n_ = 0, ec_kind_ = -1, ec_maxcopy_ = 16;
    long long ec_now_ = 0;
    size_t ec_bound_ = (size_t)-1;
    double ec_logd_ = 0, ec_hyst_ = 0.5;
    EcStats ec_st_;
    struct Bufs {
        int rows = 0;
        float *R, *R2 = nullptr, *inj, *hu, *hss, *gates, *proj, *proj2, *part, *y, *qbuf, *rlog, *yd, *gw, *gsh;
        bf16 *x, *ab, *pler, *mtp_en = nullptr, *mtp_hn = nullptr;
        __half *xh, *ah, *pleh;
        int* gsel;
        float *Rm = nullptr, *Rm2 = nullptr, *Rsave = nullptr, *mtp_fe = nullptr, *mtp_fh = nullptr;
    };
    void alloc_rows(int n, Bufs& b, int nb = 0);
    static std::vector<size_t> bt_row_bytes();
    void* bt_buf(int n, size_t rb);  // nb > n: the buffers of the batched MoE half get nb rows
    void trunk_batch(int n, unsigned selmask);
    unsigned batch_sel(int pos, int n);  // chunks with block selection -> bits; also stages the chunk positions
    int pfbn_ = PFM;
    __half* moe_slots_ = nullptr;  // MoE down slot buffer (aliases the mixer chunk scratch, alloc_rows)
    int moe_slot_rows_ = 0;
    __half* moe_slots2_ = nullptr;  // its extension for the MoE parts of big batches (a borrowed region)
    int moe_slot_rows2_ = 0;
    // big prefill batches (prefill-r3): the batch buffers hold pfbn_ rows, their rows past pfbn_base_ (and the slot
    // extension) are expert cache slots outside such batches (bt_regs_: [address, bytes]); bt_want_: the coming
    // launch needs them, bt_taken_: given back by the cache
    int pfbn_base_ = PFM;
    std::vector<std::pair<char*, size_t>> bt_regs_;
    bool bt_want_ = false, bt_taken_ = false;
    // Big-batch rows past pfbn_mid_ (FLASH_PFB_ALIAS, default on with KV in VRAM): their buffer tails are mapped
    // (CUDA VMM) onto the same physical memory as the last al_arr_ bytes of every KV K / V array. Those bytes are
    // expert-cache slots (al) until a batch of more than pfbn_mid_ rows takes them (evicted, refilled after it, as the
    // bt slots) and KV only once the pool is nearly full; such a batch runs only while the KV rows in use end below
    // al_off_ (al_room; else pf_batch() gives pfbn_mid_). So the bigger batch costs no static expert residency.
    int pfbn_mid_ = PFM;
    size_t al_gran_ = 0, al_arr_ = 0, al_off_ = 0;   // VMM granularity, alias bytes per KV array, its offset there
    std::vector<unsigned long long> al_pool_;         // physical chunks (al_gran_ each) of the KV tails, in order
    size_t al_next_ = 0;                              // next unused chunk of al_pool_
    int al_ndev_ = 0, al_devs_[8] = {};               // devices granted access to the mappings (own + P2P peers)
    bool al_want_ = false, al_taken_ = false;
    bool al_room(long long ctx_end) const;            // the KV rows of the pool up to ctx_end (or kv_top_) end below al_off_
    size_t kv_bound_rows(long long ctx_abs) const;    // KV rows in use for an absolute context end (as ec_tick)
    char* vmm_map(size_t own_bytes, size_t al_chunks, bool to_pool);  // own head + al chunks (from / into al_pool_)
    void bt_clear_y(int n);  // big batch: its borrowed rows of the y accumulator zeroed (rows [pfbn_base_, n))
    int* pfinv_ = nullptr;         // slot of every (row, k) of the batch's routing
    bool mtpb_ = false, mtpb_last_ = false;  // prefill_batch with the MTP layer
    float* dnqk_ = nullptr;  // chunked DeltaNet inputs: normalized q / k [PFM][2][nk][128], decay / beta [PFM][2][nv]
    float* dngb_ = nullptr;
    int* zero_d_ = nullptr;
    int attn_split_ = 1;
    bool dec_k_ = true;
    int dec_split_ = 8;          // decode attention: key splits of the tensor-core kernel (scratch per split)
    float* attn_spl_ = nullptr;  // device 0 (state slot of the chunked DeltaNet)
    int* cposd_ = nullptr;  // device [chunk][pos, commit]
    // prefill batch geometry: chunk c = rows [pf_co_[c], pf_co_[c + 1]); prefill_batch: PFM rows each, prefill_multi:
    // every segment cut into PFM-row chunks from its first row (pf_layout)
    int pf_nch_ = 0;
    std::vector<int> pf_co_;
    int chunk_off(int c, int n) const { return pfm_ ? pf_co_[c] : std::min(n, c * PFM); }
    // multi-sequence batch being built (prefill_multi): segment of every chunk, first / last chunk of every segment,
    // batch rows (exclusive end) of the segments that end their prompt
    bool pfm_ = false;
    int pfm_nseg_ = 0;
    std::vector<int> pf_cseg_, pf_sfirst_, pf_slast_, pf_end_;
    unsigned pf_decmask_ = 0;
    int pf_layout(const int* segn, int nseg, std::vector<int>& co, std::vector<int>& cseg) const;  // chunk count
    int pf_units_multi(const std::vector<int>& co, int* ub, int* cb) const;
    void pf_swap(int li, int seg, bool load);  // segment's recurrent state of layer li <-> the workspace
    int *rbch_d_ = nullptr, *rbch_h_ = nullptr;      // [chunk][PFM] KV region start row of the chunk's slot
    int *segslot_d_ = nullptr, *segslot_h_ = nullptr; // [PFSEG_MAX] slot of every segment
    int *rsl_d_ = nullptr, *rsl_h_ = nullptr;        // [pfbn_] slot of every batch row (steering)
    int *hsl_d_ = nullptr, *hsl_h_ = nullptr;        // [PFSEG_MAX] slot of every head row (sampling)
    int* pfmo_h_ = nullptr;                          // pinned [2][PFSEG_MAX] heads: sampled tokens, MTP drafts
    float* mhR_ = nullptr;                           // [PFSEG_MAX][RW] residual rows of the segments' prompt ends
    float* Rmins_ = nullptr;                         // [slots][RW] MTP chain residual of each slot's prompt end
    std::map<long long, cudaGraphExec_t> pmgraphs_;
    std::map<std::vector<int>, long long> pm_ids_;   // layout -> graph key
    int* cposh_ = nullptr;  // pinned staging
    std::map<long long, cudaGraphExec_t> pbgraphs_;
    void use(const Bufs& b);
    Bufs dec_, pfb_;
    int brows_ = MAXM;
    PfJob* pfjobs_ = nullptr;
    int* mtok_ = nullptr;
    __half* hh_pf_ = nullptr;
    float* rinv_pf_ = nullptr;
    int* pftl_ = nullptr;
    float* pfwl_ = nullptr;
    __half* pfact_ = nullptr;
    size_t pfact_bytes_ = 0;
    bool rq_attr_ = false;
    size_t qpf_attr_ = 0;
    QpUnion qpu_;
    int nsm_ = 0;
    int maxlg_ = 0;  // most local groups of one expert on this card (prefill job bound)
    void eager_stage_sync();  // staged transfers in an eager batch: bound the enqueued work (see flash.cu)
    unsigned* qnan_rec_ = nullptr;  // FLASH_QPF_NAN: [checks][own q, received q, attention out, exchange] (host-mapped)
    int qnan_n_ = 0;
    std::vector<std::pair<int, int>> qnan_meta_;  // (layer, chunk position) per check
    unsigned* jobchk_ = nullptr;  // FLASH_JOBCHK: host-mapped [overflows, max njobs per tag 1..2, bound of that call per tag]
public:
    void jobchk_report() const {
        if (jobchk_) fprintf(stderr, "[jobchk] rank %d: prefill max %u jobs (bound %u), decode max %u (bound %u), %u overflows\n", p_.rank,
                             jobchk_[2], jobchk_[5], jobchk_[3], jobchk_[6], jobchk_[0]);
    }
private:  // block unions of the prefill attention row groups (flash_pfq.cuh k_qpf_union)
    int maxloc_ = 0;  // most local groups of one expert on this card  // k_qsa_pfsel's dynamic shared memory set on this device  // k_rows_q's dynamic shared memory set on this device
    int* possub_ = nullptr;
    bool pf_ = false;          // prefill forward (m > MAXM): GEMM / gathered-MoE paths, rows committed at once
    void prefill_body(int m, bool mtp_fill, int hrow_last);
    template <class F> void run_cached(std::map<long long, cudaGraphExec_t>& cache, long long key, F fn);
    std::map<long long, cudaGraphExec_t> pgraphs_;
    std::map<const void*, std::map<long long, unsigned long long>> graph_use_;  // LRU ticks per graph cache
  public:
    // FLASH_XBYTES=1 (test): bytes this card sends to its peers in prefill forwards, by exchange kind (+8: MTP block);
    // counted when the work is enqueued, a graph's bytes counted at capture and added on every replay
    enum { XB_RS_MIX = 0, XB_AG_MOEIN, XB_RS_MOE, XB_AG_MIXIN, XB_Q, XB_PART, XB_IDX, XB_MISC, XB_N = 16 };
    unsigned long long xb_[XB_N] = {};
    void xb_report(size_t ntok) const;
  private:
    int xb_base_ = 0;
    std::map<std::pair<const void*, long long>, std::vector<unsigned long long>> xb_key_;
    void xb_add(int kind, size_t bytes) { if (pf_) xb_[xb_base_ + kind] += bytes; }
    unsigned long long graph_tick_ = 0;
  public:
    long long graphs_evicted_ = 0, graph_oom_ = 0, graph_eager_ = 0;  // LRU evictions, instantiate OOMs, failed retries
    graph_oom::Cool gcool_;
    int ginst_calls_ = 0;  // instantiate calls of this card (FLASH_TEST_GRAPH_OOM)
    int graphs_live() const {
        int n = 0;
        for (auto* c : {&pgraphs_, &pbgraphs_}) for (auto& kv : *c) n += kv.second != nullptr;
        return n;
    }
  private:
    int *pfh_tok_ = nullptr, *pfh_mt_ = nullptr, *pfh_pos_ = nullptr, *pfh_out_ = nullptr;
    bf16* pfh_ple_ = nullptr;
    void hc_read(const HcW& w, int m, bool inject);
    void hc_write(int m);
    void flush_y(int m);
    bool ypend_ = false;
    float *R2_ = nullptr, *Rm2_ = nullptr;  // ping-pong partners of R_ / Rm_ (fused residual update)
    void gdn(Layer& L, int li, int pos, int m);
    void qsa(Layer& L, int li, int pos, int m);
    void moe(Layer& L, int m);
    // pipelined QSA prefill chunks (prefill-r3, FLASH_QSA_PIPE): qsa() with qsa_split_ = 1 stops after sending the
    // chunk's partials (buffers of slot qsa_buf_), qsa_back(slot) combines them and runs o_proj later, after the next
    // chunk's front half (the partial exchange flies under it)
    void qsa_back(int slot);
    struct QsaBack { Layer* L = nullptr; const float* prow = nullptr; float* ax = nullptr; int mr = 0, ld = 0, ql = 0, nqp = 0; };
    QsaBack qpend_[2];
    int qsa_split_ = 0, qsa_buf_ = 0, qsa_ahead_ = 0;
    float *projB_ = nullptr, *attnB_ = nullptr;  // slot 1 copies of proj_ / attn_x_ (in the MoE activation buffer)
    // prefill row GEMM of a W8 matrix: W8A8 (k_rows_q) for m >= 64 prefill rows, else gemv_w8_rows
    void rows8(const W8& w, const bf16* x, int ldx, float* y, int ldy, int m, cudaStream_t st);
    void ple(Layer& L, int pos, int m, const bf16* rows);

    Plan p_;
    int dev_, max_ctx_, rows_ = 0;  // owned KV rows per card (the pool: logical rows when streaming)
    // KV streaming (FLASH_KV_STREAM): rows_ = logical pool rows (FLASH_KV_STREAM_CTX positions, host memory), srows_ =
    // rows of one sequence of max_ctx (selection buffers, device K / V arrays: block cache + expert-cache tail),
    // kvs_P_ = cache blocks per layer, kvs_pe_ = blocks the cache may use now (the rest lent to the expert cache)
    bool kvs_ = false;
    int srows_ = 0, kvs_P_ = 0, kvs_pe_ = -1;
    size_t kvs_host_ = 0;  // pinned host bytes of the streamed rows
    // lazy pinning: each host array is reserved address space (mmap), pinned in slabs of kvh_slab_ rows as the
    // regions' top grows (cudaHostRegister; device pointer = host pointer), released 2 slabs above the top
    bool kvh_lazy_ = false;
    int kvh_slab_ = 0, kvh_pinned_ = 0, kvh_min_ = 0;  // rows (kvh_min_: pinned at start, kept)
    struct HArr { char* p; size_t row; };
    std::vector<HArr> kvh_arr_;
    void kv_host_ensure(int periods, bool may_release);  // release: launch time only, after kvh_low_ calls in a row
    int kvh_low_ = 0;
    void kvh_pin(int r0, int r1);  // cudaHostRegister rows [r0, r1) of every host array
    // one slab pinned ahead in a background thread (kvh_bg_ rows when it is done), joined before the rows are used
    std::thread kvh_th_;
    std::atomic<bool> kvh_done_{false};
    int kvh_bg_ = 0;
    int* psel_ = nullptr;  // KV streaming: resolved blocks of the decode rows keys [MAXM][srows_ / 4] (residency pass -> k_qsa_dec)
    size_t ec_ibound_ = (size_t)-1;  // KV streaming: indexer-key bytes per array in use (expert-cache slots above)
    void kv_need(Layer& L, int m, const int* pos0, const int* rpos, const int* rbase, int* psel = nullptr);  // no-selection rows
    void kvs_set_pe(int pe);
    KvMap kvmap(const Layer& L) const { return kvs_ ? L.kvp.m : KvMap(); }
    Comm comm_;
    CommCounters ctr_;
    cudaStream_t st_ = nullptr;
    size_t wbytes_ = 0, host_bytes_ = 0;
    std::vector<std::vector<int>> cold_;
    std::vector<int> draft_ids_;
    int ndraft_ = 0;
    unsigned* coldcnt_ = nullptr;
    int* gsel_ = nullptr;
    float *gw_ = nullptr, *gsh_ = nullptr;
    int* lmap_ = nullptr;
    std::vector<std::pair<void*, size_t>> allocs_;
    // debug FLASH_LEND_CHECK=1 (lend_audit, flash.cu): memory lent to the expert cache vs buffers written meanwhile.
    // lend_own_: allocations that contain lent memory -> their bytes that are never lent (0: lent / reclaimed by the
    // KV bound at run time); lend_sub_: own regions outside allocs_ (VMM heads) and sub-buffers carved out of them
    std::vector<std::pair<uintptr_t, size_t>> lend_own_;
    std::vector<std::tuple<uintptr_t, size_t, const char*>> lend_sub_;
    std::vector<std::pair<uintptr_t, int>> lend_slots_;  // sorted slot addresses -> ec_slots_ index (lend_use)
    bool lend_done_ = false;
    void lend_audit();
    void lend_use(const void* p, size_t bytes, const char* what);
    std::vector<Layer> L_;
    HcW mixer_;
    // MTP head
    bool has_mtp_ = false;
    Layer mtp_;
    HcW mtp_mixer_;
    W8 mtp_fce_, mtp_fch_;
    bf16 *mtp_pre_e_ = nullptr, *mtp_pre_h_ = nullptr;
    float* Rm_ = nullptr;     // MTP residual [MAXM][10240]
    bf16* mtp_en_ = nullptr;  // normed embedding [MAXM][2560]
    bf16* mtp_hn_ = nullptr;  // normed hidden streams [MAXM][4][2560]
    float* mtp_fe_ = nullptr; // fc_embedding out [MAXM][2560]
    float* mtp_fh_ = nullptr; // fc_hidden out [MAXM * 4][2560]
    float* Rsave_ = nullptr;  // trunk residual of the last verify forward (MTP input rows)
    W8 lm_;
    W8 dlm_;               // balanced draft head (this card's share of the draft vocabulary by rank)
    int* dmap_ = nullptr;
    const bf16* embed_ = nullptr;
    int ll_idx_ = 0;
    // buffers
    float* R_ = nullptr;      // [MAXM][10240]
    bf16* hn_ = nullptr;      // normed HC input [MAXM][10240]
    float* z_ = nullptr;      // [MAXM][336]
    bf16* zb_ = nullptr;      // [MAXM][384]
    float* inj_ = nullptr;    // [MAXM][4]
    float* hu_ = nullptr;     // fused HC: per-stream down sums [MAXM][4][336]
    float* hss_ = nullptr;    // and sums of squares [MAXM][4]
    float* gates_ = nullptr;  // [MAXM][10240]
    bf16* x_ = nullptr;       // block input [MAXM][2560]
    __half* xh_ = nullptr;    // the same in XH layout
    int projw_ = 9216, outw_ = 2048;  // row widths of proj_ (in-projections) and ah_ / ab_ (out-projection inputs)
    float* proj_ = nullptr;   // [MAXM][projw_]
    float* proj2_ = nullptr;  // [MAXM][640]
    __half* ah_ = nullptr;    // XH input of an out projection [MAXM][2048]
    bf16* ab_ = nullptr;      // bf16 input of the INT8 o_proj [MAXM][2048]
    float* part_ = nullptr;   // [MAXM][2560]
    float* y_ = nullptr;      // [MAXM][2560]
    float* attn_x_ = nullptr; // exchange rows [MAXM][24 * 258]
    float* attn_all_ = nullptr;  // [ngpu][MAXM][24 * 258]
    float* qbuf_ = nullptr;   // roped/normed q [MAXM][24][256], k/v rows [MAXM][2][256] after it
    // MoE
    float* rlog_ = nullptr;   // router logits [MAXM][528]
    float* rfpart_ = nullptr;  // router v5: K-split partials
    unsigned* rfcnt_ = nullptr;  // router v5: arrival counter
    int* njobs_ = nullptr;    // device
    int maxjob_ = MAXJOB, maxjob_pf_ = MAXJOB_PF;  // job list capacities of this card (decode, prefill)
    MoeJob* jobs_ = nullptr;  // [maxjob_]
    float* jw_ = nullptr;     // job weights [MAXJOB][MAXM]
    __half* acth_ = nullptr;  // weighted GLU activations XH [MAXJOB][MAXM][128]
    int* jrow_ = nullptr;     // grouped MoE (m > 8): per job {row count, rows} [MAXJOB][MAXM + 1]
    float* yd_ = nullptr;     // scratch [MAXM][10240] (PLE)
    int* toks_ = nullptr;     // device tokens [MAXM]
    int* posd_ = nullptr;     // device position
    bf16* pler_ = nullptr;    // device n-gram rows [MAXM][2560]
    __half* pleh_ = nullptr;  // XH of them
    float* logits_ = nullptr; // [MAXM][vocab slice]
    float *pairs_ = nullptr, *pairs_all_ = nullptr;
    int* otok_ = nullptr;
    float* xrecv_ = nullptr;
    unsigned *xflags_ = nullptr, *xseq_ = nullptr, *xdone_ = nullptr, *xscnt_ = nullptr, *xqscnt_ = nullptr, *xiscnt_ = nullptr;
    float* xpr_[4] = {};
    bool qsa_a_ = false;
    bool agq_on_ = false;     // MoE input quantized + routed on the owner, compact rows gathered
    bool moe_pre_ = false;    // this moe() call's rows are quantized and routed already (gsel_ / gw_ / gsh_ at the rows)
    bf16* agq_x_ = nullptr;   // owner scratch [AG_CAP][H]: router input x
    int8_t* agq_r8_ = nullptr;
    float2* agq_r8s_ = nullptr;
    float* agq_rlog_ = nullptr;
    bool qx_on_ = false;      // FLASH_QX_LOCAL: prefill q of the peers' heads computed here (no q exchange)
    float* qxb_ = nullptr;    // its fp32 projection rows [PFM][(NQ - local) x HD]
    float* xqrecv_ = nullptr;
    unsigned *xqflags_ = nullptr, *xqseq_ = nullptr, *xqdone_ = nullptr;
    float* xqpr_[4] = {};
    unsigned* xqpf_[4] = {};
    unsigned* xpf_[4] = {};
    int *tok_h_ = nullptr, *pos_h_ = nullptr, *out_h_ = nullptr;  // pinned staging (graph inputs / outputs)
    bf16* ple_h_ = nullptr;
    cudaGraphExec_t graph_[2][2][MAXM + 1] = {};  // [sampling][selection][m]
    cudaGraphExec_t mgraph_[2][2][MAXM + 1] = {};
    cudaGraphExec_t sgraph_[2][2][2][MAXM] = {};
    bool lk_on_ = false;
    int lk_mmin_ = 6, lk_llo_ = 8, lk_hi_ = 24, lk_lmax_ = 16, lk_kfull_ = -1;
    cudaStream_t lkst_ = nullptr;
    bool hcs_ = true;
    float* hcp_ = nullptr;  // decode HC down group partials [MAXM][HC][20][HCD_ROWS]
    unsigned long long* detpart_ = nullptr;  // FLASH_DETERMINISTIC: fixed-point MoE partial accumulator [pfbn_][H]
  public:
    static constexpr int DBGK = 12;          // FLASH_DBGHASH kinds per layer (flash_main prints the names)
    unsigned long long* dbgh_ = nullptr;     // FLASH_DBGHASH: per layer hashes of the prefill intermediates
    unsigned long long* rowh_ = nullptr;     // FLASH_DBGHASH=2: per (layer, row) hashes [LAYERS][pfbn_][4] (debug)
    double* rowl1_ = nullptr;
    void* xqsnap_ = nullptr;
    void* actsnap_ = nullptr;     // FLASH_DBG_GU2X: the first gate_up's output (debug)      // FLASH_DBG_XQCHK: copy of the MoE gate_up inputs (debug)
    unsigned* xqrec_ = nullptr;   // FLASH_DBG_XQCHK: host-mapped records of changed rows
    void dbg_xqrep(int rep) {
        if (!xqrec_) return;
        CUDA_CHECK(cudaSetDevice(dev_));
        CUDA_CHECK(cudaStreamSynchronize(st_));
        const unsigned n = xqrec_[0];
        if (n) printf("xqchk rep %d rank %d: %u changed rows during gate_up\n", rep, p_.rank, n);
        for (unsigned k = 0; k < n && k < 12; k++) {
            const unsigned* e = xqrec_ + 8 + k * 6;
            printf("  layer %u %s row %u word %u: now %08x was %08x\n", e[0], e[1] == 2 ? "act (gate_up run 2 vs 1)" : e[1] ? "tl" : "xh/xq", e[2], e[3], e[4], e[5]);
        }
        xqrec_[0] = 0;
    }                // FLASH_DBGHASH=2: per (layer, row) L1 norm of the MoE partial (debug)
    void dbg_rowl1(std::vector<double>& out) {
        out.assign((size_t)LAYERS * rowh_n_, 0.0);
        if (!rowl1_) return;
        CUDA_CHECK(cudaMemcpy(out.data(), rowl1_, out.size() * 8, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemset(rowl1_, 0, out.size() * 8));
    }
    int rowh_n_ = 0;
    void dbg_rowh(std::vector<unsigned long long>& out) {
        out.assign((size_t)LAYERS * rowh_n_ * 4, 0);
        if (!rowh_) return;
        CUDA_CHECK(cudaSetDevice(dev_));
        CUDA_CHECK(cudaStreamSynchronize(st_));
        CUDA_CHECK(cudaMemcpy(out.data(), rowh_, out.size() * 8, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemset(rowh_, 0, out.size() * 8));
    }
    void dbg_hashes(std::vector<unsigned long long>& out) {
        out.assign((size_t)(LAYERS + 1) * DBGK, 0);
        if (!dbgh_) return;
        CUDA_CHECK(cudaSetDevice(dev_));
        CUDA_CHECK(cudaStreamSynchronize(st_));
        CUDA_CHECK(cudaMemcpy(out.data(), dbgh_, out.size() * 8, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemset(dbgh_, 0, out.size() * 8));
    }
  private:
    unsigned *nf_h_ = nullptr, *nf_d_ = nullptr, nf_seen_ = 0;  // non-finite guard counter (host-mapped)
    unsigned *nfr_h_ = nullptr, *nfr_d_ = nullptr;   // per row: non-finite seen (host-mapped, MAXM rows)
    unsigned nf_pending_ = 0;                        // rows flagged since the last take_nf
    int nf_calls_ = 0;                               // checks so far (FLASH_GUARD_TEST_AT)
    int brps_ = 1;                                   // rows per segment of the last batched step
    int *hd_ = nullptr, *hn_d_ = nullptr, *lk_ = nullptr;
    // uniform lookup result (decode hang seen in testing): every rank's lk_ gathered, rank 0's installed everywhere
    static constexpr int kLkWords = 1 + 31;  // count + up to 31 drafts (lk_lmax_ <= MAXP - 1 = 16)
    float* lkall_ = nullptr;
    unsigned* lkdiff_ = nullptr;  // host-mapped: lookups where a rank's own result differed from rank 0's
public:
    unsigned lk_disagreements() const { return lkdiff_ ? *(volatile unsigned*)lkdiff_ : 0u; }
private:
    unsigned long long* lkbest_ = nullptr;
    // global answer corpus for the lookup (FLASH_CORPUS tokens, default 0 = off): device copy on every card
    int *cd_ = nullptr, *cn_d_ = nullptr;
    unsigned long long* lkbestc_ = nullptr;
    int ccap_ = 0, cpos_ = 0, cfill_ = 0;
    void lk_scan_emit();
    int *lkrg_ = nullptr, *lkrp_ = nullptr;  // lookup region state (device) / region policy (nullptr: global)
    bool sel_on_ = false;     // indexer block selection in the graphs being built (context > 2051)
    int spos_ = 0;            // position of the device-side speculative loop
    float* xirecv_ = nullptr;
    unsigned *xiflags_ = nullptr, *xiseq_ = nullptr, *xidone_ = nullptr;
    float* xipr_[4] = {};
    unsigned* xipf_[4] = {};
    float* iscore_ = nullptr;
    bool idx_gk_ = false;     // block-selection keys in global memory (k_idx_local<true>): a card's blocks exceed smem
    int *isel_ = nullptr, *insel_ = nullptr;
    unsigned long long* selst_ = nullptr;  // FLASH_SELSTAT counters
    unsigned *sw_stamp_ = nullptr, *sw_cid_ = nullptr;  // FLASH_SELWIN (KV streaming gate)
    unsigned long long* sw_c_ = nullptr;
    unsigned* iloc_ = nullptr;
    bool sampling_ = false;
    SampleParams* sp_ = nullptr;
    float *cand_ = nullptr, *cand_all_ = nullptr;
    int* curpos_ = nullptr;   // positions of the layers being run (trunk posd_, MTP mposd_)
    int *motok_ = nullptr, *mposd_ = nullptr, *vtok_ = nullptr, *acc_ = nullptr, *dr_ = nullptr, *hist2_ = nullptr;
    int *em_ = nullptr, *em_h_ = nullptr;  // emitted tokens (host-mapped)
    NgramDev ngp_{};
    cudaEvent_t ev0_ = nullptr, ev1_ = nullptr;
    // batched prefill: all-reduces + residual updates on a second stream, overlapped with the next chunks' compute
    cudaStream_t cst_ = nullptr;
    // CE: second comm stream for the part of a prefill collective after the reduce-scatter (HC read,
    // router / quantization, row all-gather), so the next unit's reduce-scatter goes on the link meanwhile
    cudaStream_t cst2_ = nullptr;
    cudaEvent_t evs_ = nullptr;
    cudaStream_t pfs_ = nullptr;            // decode: L2 prefetch of the next HC weights (graph side branch)
    cudaStream_t pfs2_ = nullptr;  // side stream of the lookup chain's conditional body capture
    cudaEvent_t pfe0_ = nullptr, pfe1_ = nullptr;
    const HcW* next_hc_ = nullptr;          // the HC read after the next all-reduce (prefetch target)
    int hcl2_ = 2;                          // FLASH_HCL2 (default 2): bit 0 down weights during the AR (slower), bit 1 up weights during mid
    std::vector<cudaEvent_t> evp_, evd_;  // per chunk: partial ready (compute), reduced + applied (comm)
    cudaEvent_t evm_ = nullptr;
    cudaEvent_t pfq0_ = nullptr, pfq1_ = nullptr;  // prefill q exchange on the side stream (fork / join)           // MoE partials of the batch ready
    int prev_m_ = 0;
    unsigned* stepc_ = nullptr;
    float* Rmin_ = nullptr;   // MTP chain input row
    // steering (set_steer): vectors, per-slot {vector, mode, lo, hi} and scale [slots] and the bound slot's copy [1]
    float *steer_w_ = nullptr, *steer_dw_ = nullptr, *steer_sc_d_ = nullptr, *steer_csc_d_ = nullptr;
    int *steer_sl_d_ = nullptr, *steer_csl_d_ = nullptr;
    int steer_nv_ = 0, steer_lo_ = LAYERS, steer_hi_ = -1;  // union of the vectors' file layers
    std::vector<int> steer_sl_h_, steer_lo_v_, steer_hi_v_;
    std::vector<float> steer_sc_h_;
    float* cap_ = nullptr;           // capture accumulators [LAYERS][H] (active between cap_begin / cap_end)
    float* cap_buf_ = nullptr;
    int cap_skip_ = 0;
    int steer_pend_ = -1;                // a layer output to steer, applied by the next HC read (fused) or flush
    const int* steer_prs_ = nullptr;     // its row -> slot map (nullptr: the bound slot)
    SteerArg steer_arg(int li, const int* rslot) const;
    // rows [0, m) of R at layer li's output now (y: pending update or null), or mark it pending (fused into the next
    // decode HC read; flush_y / an unfused read launch the separate kernel)
    void steer(int li, int m, const int* rslot, cudaStream_t s, float* R, const float* inj, float* y);
    void steer_mark(int li, const int* rslot) { if (steer_w_ && li >= steer_lo_ && li <= steer_hi_) { steer_pend_ = li; steer_prs_ = rslot; } }
    void steer_flush(int m);  // a pending layer-output steering as the separate kernel (before the pending update)
};

// expert cache statistics of all cards: per decode launch misses summed over the cards (steps aligned)
inline std::string ec_summary(std::vector<Worker*>& W) {
    long long st = 0, t = 0, ms = 0, pt = 0, pm = 0, cp = 0, ek = 0, ef = 0, ekp = 0;
    int nc = 0, res = 0, live = 0, slots = 0;
    double bw = 0;
    std::vector<int> per;
    for (Worker* w : W) {
        const Worker::EcStats& s = w->ec_stats();
        st = std::max(st, s.dec_steps); t += s.dec_touch; ms += s.dec_miss; pt += s.pf_touch; pm += s.pf_miss;
        cp += s.copies; ek += s.ev_kv; ekp += s.ev_kv_pend; ef += s.ev_freq; nc += s.ncold; res += s.resident; live += s.live_slots; slots += s.slots;
        bw = std::max(bw, s.seed_gbs);
        if (per.size() < s.step_miss.size()) per.resize(s.step_miss.size(), 0);
        for (size_t i = 0; i < s.step_miss.size(); i++) per[i] += s.step_miss[i];
    }
    std::vector<int> so = per;
    std::sort(so.begin(), so.end());
    char b[640];
    snprintf(b, sizeof b,
             "expert cache: %lld decode steps, misses/step mean %.2f p50 %d p90 %d max %d (all cards), cold-group hit rate "
             "%.1f%% decode (%lld/%lld) %.1f%% prefill (%lld/%lld); copies %lld, evicted kv %lld (copy in flight %lld) freq %lld; resident %d/%d "
             "cold groups, live slots %d/%d; seed H2D %.1f GB/s",
             st, st ? (double)ms / st : 0.0, so.empty() ? 0 : so[so.size() / 2], so.empty() ? 0 : so[so.size() * 9 / 10],
             so.empty() ? 0 : so.back(), t ? 100.0 * (t - ms) / t : 100.0, t - ms, t, pt ? 100.0 * (pt - pm) / pt : 100.0,
             pt - pm, pt, cp, ek, ekp, ef, res, nc, live, slots, bw);
    return b;
}

}  // namespace flash
