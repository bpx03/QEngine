// Decode / verify step of the GLM engine (m <= MR rows: the last token + drafts), c = 1. Routing and the expert plan on
// the device; cold experts handed to the host through mapped memory. The host coordinator (one thread) decides per cold
// expert: admit (read-through: a GPU kernel on every card copies its contiguous slice from the pinned host copy into a
// cache slot, then computes it from the slot; the bytes cross DRAM once) or CPU lane (host cores, result added on rank
// 0). TinyLFU admission with hysteresis, capped admissions per step. Kernels wait on sequence words in mapped memory, so a
// step is one CUDA graph per card; KDA states of the verify rows are snapshots, committed after acceptance.
#include <immintrin.h>
#include <pthread.h>
#include <sched.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <random>
#include <unordered_map>
#include <cuda_profiler_api.h>

#include "../env.h"
#include "glm.h"
#include "glm_cpu.h"
#include "glm_kernels.cuh"

namespace glm {

#define CUBLAS_CHECK_(x)                                                                          \
    do {                                                                                          \
        cublasStatus_t s__ = (x);                                                                 \
        CHECK(s__ == CUBLAS_STATUS_SUCCESS, "cublas %d at %s:%d", (int)s__, __FILE__, __LINE__); \
    } while (0)

__device__ __forceinline__ unsigned ld_vol(const volatile unsigned* p) { return *p; }

__global__ void k_seq_inc(unsigned* seq) { *seq += 1; }
__global__ void k_stamp(unsigned long long* d) {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    *(volatile unsigned long long*)d = t;
}
__global__ void k_pos_add(int* p, const int* a) { *p += *a; }
__global__ void k_count(const int* ids, int n, unsigned long long* cnt) {
    if ((int)threadIdx.x < n) atomicAdd(cnt + ids[threadIdx.x], 1ull);
}

__device__ void set_job(XJob& J, const char* slab, const SliceL& sl, int s) {
    const char* b = slab + (size_t)s * sl.bytes;
    J.x.gq = (const int*)(b + sl.gq); J.x.uq = (const int*)(b + sl.uq); J.x.dq = (const int*)(b + sl.dq);
    J.x.gs = (const __half*)(b + sl.gs); J.x.us = (const __half*)(b + sl.us); J.x.ds = (const __half*)(b + sl.ds);
    J.x.ld = sl.ni; J.x.c0 = 0; J.x.r0 = 0; J.x.host = false;
}

// plan: distinct experts of the m rows (one warp: lane i = selection i, row i / 8); hot ones -> jobs (slot pointers),
// cold ones -> host request (rank 0)
// rank 0: the MoE input rows -> the host request (many blocks, float4; k_plan then publishes without copying)
__global__ void k_xreq(const float* x, int n4, MoeReq* req) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += gridDim.x * blockDim.x)
        reinterpret_cast<float4*>(&req->x[0][0])[i] = reinterpret_cast<const float4*>(x)[i];
    __threadfence_system();
}
__global__ void k_plan(const int* ids, const float* wts, int m, const int* slot, const char* slab, const __grid_constant__ SliceL sl, XJob* jobs,
                       int* njobs, MoeReq* req, const float* x, const unsigned* seq, bool copy_x = true,
                       const volatile int* bw_done = nullptr, const int* bw_need = nullptr) {
    __shared__ int se[MXE];
    __shared__ float sw[MXE];
    if (bw_done) {  // round 17 (GLM_BG_PERL): this layer's boundary copies (and every earlier layer's) have landed
        if (threadIdx.x == 0) {
            const int need = *bw_need;
            long long n = 0;
            while ((int)((unsigned)*bw_done - (unsigned)need) < 0)  // review r17 L3: wrap-safe (monotonic counters)
                if (++n > (1ll << 34)) __trap();
            __threadfence();
        }
        __syncthreads();
    }
    if (threadIdx.x < 32) {
        const int ln = threadIdx.x, n = m * TOPK;
        const bool v = ln < n;
        const int e = v ? ids[ln] : -1 - ln;
        se[ln] = e;
        sw[ln] = v ? wts[ln] : 0.f;
        __syncwarp();
        int first = ln;  // first lane with the same expert
        for (int j = 0; j < ln; j++)
            if (se[j] == e) { first = j; break; }
        const bool lead = v && first == ln;
        const int s_ = v ? __ldcg(slot + e) : -1;  // round 11: L2 (the boundary copies update the table from another stream)
        const unsigned hotm = __ballot_sync(0xffffffffu, lead && s_ >= 0);
        const unsigned coldm = __ballot_sync(0xffffffffu, lead && s_ < 0);
        const unsigned below = (1u << ln) - 1u;
        if (req && v) { req->ids[ln / TOPK][ln % TOPK] = e; req->hot[ln / TOPK][ln % TOPK] = s_ >= 0; }
        if (lead) {
            // rows / weights go straight to their destination (no runtime-indexed local arrays: no stack frame)
            int* rd = nullptr;
            float* wd = nullptr;
            int* nd = nullptr;
            if (s_ >= 0) {
                XJob& J = jobs[__popc(hotm & below)];
                set_job(J, slab, sl, s_);
                rd = J.r; wd = J.w; nd = &J.nr;
            } else if (req) {
                const int c = __popc(coldm & below);
                req->ce[c] = e;
                rd = req->crow[c]; wd = req->cw[c]; nd = &req->cnr[c];
            }
            if (nd) {
                int nr = 0;
                for (int j = ln; j < n; j++)
                    if (se[j] == e) { rd[nr] = j / TOPK; wd[nr] = sw[j]; nr++; }
                *nd = nr;
            }
        }
        if (ln == 0) {
            *njobs = __popc(hotm);
            if (req) { req->m = m; req->ncold = __popc(coldm); }
        }
    }
    if (!req) return;
    if (copy_x)
        for (int i = threadIdx.x; i < m * H; i += blockDim.x) req->x[i / H][i % H] = x[i];
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) {
        __threadfence_system();
        *(volatile unsigned*)&req->seq = *seq;
    }
}

__global__ void k_eflag(MoeReq* req, const unsigned* seq) { *(volatile unsigned*)&req->eseq = *seq; }
__global__ void k_pub_pred(const int* pids, const float* pw, int m, MoeReq* req, const unsigned* seq) {
    const int i = threadIdx.x;
    if (i < m * TOPK) { req->pids[i / TOPK][i % TOPK] = pids[i]; req->pw[i / TOPK][i % TOPK] = pw[i]; }
    __threadfence_system();
    __syncthreads();
    if (i == 0) *(volatile unsigned*)&req->pseq = *seq;
}

__global__ void k_pub_late(const int* ids, const float* w, int m, MoeReq* req, const unsigned* seq) {
    const int i = threadIdx.x;
    if (i < m * TOPK) { req->lids[i / TOPK][i % TOPK] = ids[i]; req->lw[i / TOPK][i % TOPK] = w[i]; }
    __threadfence_system();
    __syncthreads();
    if (i == 0) *(volatile unsigned*)&req->lseq = *seq;
}

__global__ void k_wait_dec(const MoeDec* dec, const unsigned* seq) {
    const unsigned s = *seq;
    long long n = 0;
    while (ld_vol(&dec->seq) != s)
        if (++n > (1ll << 34)) __trap();
    __threadfence_system();
}

// read-through admission: this card's slice (one contiguous tile-major block of the host copy) -> row-major slot.
// grid (MXE, 48)
// one card slice: tile-major host block -> row-major slot; threads (blockIdx.y, threadIdx.x) of gridDim.y blocks
__device__ __forceinline__ void admit_copy(const char* src, char* dst, SliceL sl, SliceL ss) {
    const int ni = sl.ni;
    const size_t ng = (size_t)512 * (ni / 4), ns = (size_t)32 * (ni / 8), nd = (size_t)(ni / 8) * (H / 4),
                 nds = (size_t)(ni / 64) * (H / 8);
    const size_t tot = 2 * ng + 2 * ns + nd + nds;
    for (size_t i = (size_t)blockIdx.y * blockDim.x + threadIdx.x; i < tot; i += (size_t)gridDim.y * blockDim.x) {
        size_t jj = i, so, dof;
        if (jj < 2 * ng) {
            const bool up = jj >= ng;
            jj %= ng;
            const size_t r = jj / (ni / 4), c = (jj % (ni / 4)) * 4;
            dof = (up ? sl.uq : sl.gq) + (r * ni + c) * 4;
            if (ss.gu3) {
                *reinterpret_cast<uint4*>(dst + dof) = gu3_expand(src + (up ? ss.uq : ss.gq), r, c);
                continue;
            }
            so = (up ? ss.uq : ss.gq) + (((c / 32) * 512 + r) * 32 + c % 32) * 4;
        } else if ((jj -= 2 * ng) < 2 * ns) {
            const bool up = jj >= ns;
            jj %= ns;
            const size_t g = jj / (ni / 8), c = (jj % (ni / 8)) * 8;
            dof = (up ? sl.us : sl.gs) + (g * ni + c) * 2; so = (up ? ss.us : ss.gs) + (((c / 32) * 32 + g) * 32 + c % 32) * 2;
        } else if ((jj -= 2 * ns) < nd) {
            const size_t r = jj / (H / 4), c = (jj % (H / 4)) * 4;
            dof = sl.dq + (r * H + c) * 4; so = ss.dq + (((c / 32) * (ni / 8) + r) * 32 + c % 32) * 4;
        } else {
            jj -= nd;
            const size_t g = jj / (H / 8), c = (jj % (H / 8)) * 8;
            dof = sl.ds + (g * H + c) * 2; so = ss.ds + (((c / 32) * (ni / 64) + g) * 32 + c % 32) * 2;
        }
        *reinterpret_cast<uint4*>(dst + dof) = *reinterpret_cast<const uint4*>(src + so);
    }
}
// read-through admission: this card's slice (one contiguous tile-major block of the host copy) -> row-major slot.
// grid (MXE, 48)
__global__ void k_admit(const MoeDec* dec, const char* hxl, size_t stride, size_t soff, char* slab, SliceL sl, SliceL ss,
                        SliceL ss3, size_t flag_off, int bgm = 0) {
    int j = blockIdx.x;
    if (bgm) { if (j >= dec->nbg) return; j += dec->nadm; }
    else if (j >= dec->nadm) return;
    if (flag_off && hxl[(size_t)dec->e[j] * stride + flag_off]) ss = ss3;  // mixed tiers: this expert is gu3
    admit_copy(hxl + (size_t)dec->e[j] * stride + soff, slab + (size_t)dec->slot[j] * sl.bytes, sl, ss);
}
// round 11: background admissions deferred to the step boundary (GLM_BG_BOUND): the queue of the finished step, copied
// while the GPU runs the lm_head / MTP and the host DRAM is idle; grid (n, blocks)
// the queue is passed by value (kernel parameters are fixed at launch): the coordinator refills the host queue for the
// next step as soon as rank 0 passes layer 3, while a slower rank's copies / table update may still be pending
struct BgList {
    static constexpr int CAP = 64;
    short l[CAP], e[CAP];
    int slot[CAP];
};
__global__ void k_bgq(const BgList q_, int n, const char* const* hx_layer, size_t stride, size_t soff, char* slab, SliceL sl,
                      SliceL ss, SliceL ss3, size_t flag_off) {
    const int j = blockIdx.x;
    if (j >= n) return;
    const BgList* q = &q_;
    const char* hx = hx_layer[q->l[j]];
    if (flag_off && hx[(size_t)q->e[j] * stride + flag_off]) ss = ss3;
    admit_copy(hx + (size_t)q->e[j] * stride + soff, slab + (size_t)q->slot[j] * sl.bytes, sl, ss);
}
__global__ void k_bgq_fin(const BgList q_, int n, int* slots_all, volatile int* flag) {
    if (threadIdx.x != 0) return;
    const BgList* q = &q_;
    for (int j = 0; j < n; j++) slots_all[q->l[j] * NE + q->e[j]] = q->slot[j];
    __threadfence_system();
    *flag = 0;
}
// round 11 debug (anomaly bisect): spin for ns on the GPU clock (GLM_DELAY = rank,layer,blk,us; GLM_HOG = blocks,us)
__global__ void k_spin(unsigned long long ns) {
    unsigned long long t0, t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
    do { asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); } while (t - t0 < ns);
}
// round 17 (GLM_BG_PERL): one boundary copy landed: its slot-table entry, then the card's done count
__global__ void k_bgq_one(int key, int slot, int* slots_all, volatile int* done, int v) {
    slots_all[key] = slot;
    __threadfence();
    *done = v;
}
__global__ void k_bgq_wait(const volatile int* flag, int count) {
    long long n = 0;
    while (*flag != 0)
        if (++n > (1ll << 34)) __trap();
    __threadfence();
    if (!count) return;  // r12 (review P2): the host-memory counters only under GLM_PFSTAT
    if (n) atomicAdd((int*)(flag + 8), 1);  // debug: waits that spun
    atomicAdd((int*)(flag + 12), 1);
}
// after the copies: slot tables (admitted -> slot, victims -> -1) and the admitted experts' jobs
__global__ void k_admit_fin(const MoeDec* dec, int* slots_all, const char* slab, const __grid_constant__ SliceL sl, XJob* jobs, int* njobs,
                            const char* hxl, size_t stride, size_t soff) {
    if (threadIdx.x != 0) return;
    const int n = dec->nadm;
    for (int j = 0; j < n + dec->nbg + dec->nbgd; j++)  // background victims too (incl. the deferred ones): their slots are overwritten after this
        if (dec->ve[j] >= 0) slots_all[dec->vl[j] * NE + dec->ve[j]] = -1;
    for (int j = 0; j < n; j++) {
        slots_all[dec->l * NE + dec->e[j]] = dec->slot[j];
        XJob& J = jobs[j];
        set_job(J, slab, sl, dec->slot[j]);
        J.nr = dec->nr[j];
        for (int i = 0; i < J.nr; i++) { J.r[i] = dec->row[j][i]; J.w[i] = dec->w[j][i]; }
    }
    for (int j = 0; j < dec->nzc; j++) {  // zero-copy jobs: this card's slice in the host copy
        XJob& J = jobs[n + j];
        const char* b = hxl + (size_t)dec->ze[j] * stride + soff;
        J.x.gq = (const int*)(b + sl.gq); J.x.uq = (const int*)(b + sl.uq); J.x.dq = (const int*)(b + sl.dq);
        J.x.gs = (const __half*)(b + sl.gs); J.x.us = (const __half*)(b + sl.us); J.x.ds = (const __half*)(b + sl.ds);
        J.x.ld = sl.ni; J.x.c0 = 0; J.x.r0 = 0; J.x.host = true;
        J.nr = dec->znr[j];
        for (int i = 0; i < J.nr; i++) { J.r[i] = dec->zrow[j][i]; J.w[i] = dec->zw[j][i]; }
    }
    *njobs = n + dec->nzc;
}
// round 15: ahead admissions of layer dec->l (decided at the previous layer): victims released, entries set
__global__ void k_ah_fin(const MoeDec* dec, int* slots_all) {
    if (threadIdx.x != 0) return;
    for (int j = 0; j < dec->nbg; j++) {
        if (dec->ve[j] >= 0) slots_all[dec->vl[j] * NE + dec->ve[j]] = -1;
        slots_all[dec->l * NE + dec->e[j]] = dec->slot[j];
    }
}
// round 8: background admissions are usable once copied (from the next step on)
__global__ void k_admit_bgfin(const MoeDec* dec, int* slots_all) {
    if (threadIdx.x != 0) return;
    for (int j = dec->nadm; j < dec->nadm + dec->nbg; j++) slots_all[dec->l * NE + dec->e[j]] = dec->slot[j];
}
// rank 0: wait for the CPU lane's rows and add them
__device__ __forceinline__ unsigned long long gtimer() {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}
// round 15 (GLM_LANE_DEFER): every rank adds the CPU lane's rows of a MoE layer into the residual streams,
// R[t][k][i] += pd[t][k] * y[t][i] (pd = that layer's block-1 post weights), after waiting for the lane
// review r16 L4: one polling thread per card (k_wait_out), then the add without polling
__global__ void k_wait_out(const MoeOut* out, const unsigned* seq, unsigned long long* ts) {
    if (ts) ts[0] = gtimer();  // stamps 14 / 15: entry, lane rows seen
    const unsigned s = *seq;
    long long n = 0;
    while (ld_vol(&out->seq) != s)
        if (++n > (1ll << 34)) __trap();
    __threadfence_system();
    if (ts) ts[1] = gtimer();
}
__global__ void k_def_add(const MoeOut* out, float* R, const float* pd, int m) {
    const int n4 = m * H / 4;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += gridDim.x * blockDim.x) {
        float4 v;
        const float4* src = reinterpret_cast<const float4*>(&out->y[0][0]);
        asm volatile("ld.volatile.global.v4.f32 {%0, %1, %2, %3}, [%4];" : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w) : "l"(src + i) : "memory");
        const int t = i / (H / 4), c = i % (H / 4);
#pragma unroll
        for (int k = 0; k < HC; k++) {
            const float w = pd[t * 4 + k];
            float4* d = reinterpret_cast<float4*>(R + (size_t)t * RW + (size_t)k * H) + c;
            float4 r = *d;
            r.x += w * v.x; r.y += w * v.y; r.z += w * v.z; r.w += w * v.w;
            *d = r;
        }
    }
}
__global__ void k_cp4(const float* a, float* b, int n) { if ((int)threadIdx.x < n) b[threadIdx.x] = a[threadIdx.x]; }
// round 22 ([glm-r22-poll]): one poller (the first block to arrive, ld.acquire.sys on the host flag) publishes the seen seq into a device
// flag (st.release.gpu) that the other blocks spin on (L2): 16 blocks reading the same host line serialised at the host
// bridge (~7 us to see an already set flag) and each paid a __threadfence_system (~2-3 us); GLM_WAIT_OLD=1 = the
// round-13 form (every block polls the host)
__device__ __forceinline__ unsigned ld_acq_sys(const unsigned* p) {
    unsigned v;
    asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    return v;
}
__device__ __forceinline__ unsigned ld_acq_gpu(const unsigned* p) {
    unsigned v;
    asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    return v;
}
__global__ void k_wait_add(const MoeOut* out, const unsigned* seq, float* part, int m, unsigned long long* ts,
                           unsigned* wflag) {
    if (threadIdx.x == 0) {
        const unsigned s = *seq;
        long long n = 0;
        if (!wflag) {
            while (ld_vol(&out->seq) != s)
                if (++n > (1ll << 34)) __trap();
            __threadfence_system();
        } else if (atomicExch(wflag + 1, s) != s) {  // the first block to arrive is the poller (review r22 M1: no
                                                     // dependence on block 0 being resident); s only grows
            while (ld_acq_sys(&out->seq) != s)
                if (++n > (1ll << 34)) __trap();
            asm volatile("st.release.gpu.global.u32 [%0], %1;" ::"l"(wflag), "r"(s) : "memory");
        } else {
            while (ld_acq_gpu(wflag) != s)
                if (++n > (1ll << 34)) __trap();
        }
        if (ts && blockIdx.x == 0) ts[0] = gtimer();
    }
    __syncthreads();
    // round 13: one 16-byte volatile load per thread, all of a thread's loads issued before any add (4-byte volatile
    // loads interleaved with the stores were serial PCIe round trips: ~12 us per layer for 48 KB); same adds
    const int n4 = m * H / 4;
    for (int i0 = blockIdx.x * blockDim.x + threadIdx.x; i0 < n4; i0 += 2 * gridDim.x * blockDim.x) {
        const int i1 = i0 + gridDim.x * blockDim.x;
        float4 v0, v1 = make_float4(0.f, 0.f, 0.f, 0.f);
        const float4* src = reinterpret_cast<const float4*>(&out->y[0][0]);
        asm volatile("ld.volatile.global.v4.f32 {%0, %1, %2, %3}, [%4];" : "=f"(v0.x), "=f"(v0.y), "=f"(v0.z), "=f"(v0.w) : "l"(src + i0) : "memory");
        if (i1 < n4)
            asm volatile("ld.volatile.global.v4.f32 {%0, %1, %2, %3}, [%4];" : "=f"(v1.x), "=f"(v1.y), "=f"(v1.z), "=f"(v1.w) : "l"(src + i1) : "memory");
        float4* d = reinterpret_cast<float4*>(part);
        float4 p0 = d[i0];
        p0.x += v0.x; p0.y += v0.y; p0.z += v0.z; p0.w += v0.w;
        d[i0] = p0;
        if (i1 < n4) {
            float4 p1 = d[i1];
            p1.x += v1.x; p1.y += v1.y; p1.z += v1.z; p1.w += v1.w;
            d[i1] = p1;
        }
    }
    if (ts && blockIdx.x == 0) {
        __syncthreads();
        if (threadIdx.x == 0) ts[1] = gtimer();
    }
}

// ---------------------------------------------------------------- worker side

static long long rt_ns() {
    timespec t;
    clock_gettime(CLOCK_REALTIME, &t);
    return (long long)t.tv_sec * 1000000000ll + t.tv_nsec;
}
// morning list 26 (a): a graph whose instantiation finds no VRAM left (the slab takes the free memory at load; GLM_SEQS=2
// adds 25 pair graphs per card) runs eagerly from then on (the same kernels in the same order), with one line; bodies
// that only exist in a graph (the device MTP chain's conditional nodes: eager_ok false) still fail as before
static cudaGraphExec_t const kGraphEager = reinterpret_cast<cudaGraphExec_t>(uintptr_t(1));
static void run_graph(cudaGraphExec_t& g, cudaStream_t st, const std::function<void()>& body, bool use,
                      bool eager_ok = true) {
    if (!use || g == kGraphEager) { body(); return; }
    if (!g) {
        CUDA_CHECK(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal));
        body();
        cudaGraph_t gr;
        CUDA_CHECK(cudaStreamEndCapture(st, &gr));
        cudaGraphExec_t ge = nullptr;
        const cudaError_t e = cudaGraphInstantiate(&ge, gr, 0);
        CUDA_CHECK(cudaGraphDestroy(gr));
        if (e == cudaErrorMemoryAllocation && eager_ok) {
            cudaGetLastError();
            static std::atomic<int> said{0};
            if (!said.exchange(1))
                fprintf(stderr, "[glm] WARNING: a step graph does not fit in the free VRAM: that step shape runs without "
                        "a graph (slower host launch, same results)\n");
            g = kGraphEager;
            body();
            return;
        }
        CUDA_CHECK(e);
        g = ge;
    }
    CUDA_CHECK(cudaGraphLaunch(g, st));
}


void Worker::stamp(int i) {
    if (stp) k_stamp<<<1, 1, 0, st>>>(stp + i);
}
// GLM_STAMP=1/3: per-layer globaltimer points of rank 0, group 0 (E->stamps[l * 16 + i])
void Worker::stampl(int l, int i) {
    if (E->stamps && p.rank == 0 && grp == 0) k_stamp<<<1, 1, 0, st>>>(E->stamps + l * 16 + i);
    // GLM_STAMP=4: every rank, its own clock (card skew: which card enters the all-reduces last)
    if (E->stamp_lvl == 4 && grp == 0) k_stamp<<<1, 1, 0, st>>>(E->stamps2 + ((size_t)p.rank * (NL + 1) + l) * 16 + i);
}
void Worker::moe_dec(int l, int m) {
    moe_pre(l, m);
    moe_post(l, m);
}
// MoE of group grp at layer l, first half: routing, plan / host request, admission fork, shared + hot experts
void Worker::moe_pre(int l, int m) {
    Layer& Y = L[l];
    const int ni = p.ne(), qi = grp * (NL + 1) + l;
    XJob* J0 = d_jobs + grp * 2 * MXE;
    int* NJ = d_nj + grp * 2;
    stampl(l, 0);
    float* lg = big;
    float* rpart = big + (size_t)4 * MR * NE;  // router partials (past both logit blocks)
    MoeReq* rq = p.rank == 0 ? E->req + qi : nullptr;
    // side stream (GLM_SIDE=1): the x copy to the host request runs next to the router (joined before the plan
    // publishes), the next-layer prediction after the plan (joined before the lane result is added)
    const bool side = E->side && rq;
    if (rq && E->pf_early) k_eflag<<<1, 1, 0, st>>>(rq, d_seq);  // round 13: stop the lane prefetch now (DRAM quiet for xreq / plan)
    if (side) {
        CUDA_CHECK(cudaEventRecord(ev_side[grp][0], st));
        CUDA_CHECK(cudaStreamWaitEvent(pst, ev_side[grp][0], 0));
        k_xreq<<<12, 256, 0, pst>>>(xf, m * H / 4, rq);
        CUDA_CHECK(cudaEventRecord(ev_side[grp][1], pst));
    }
    router_dec(Y.router, xf, m, rpart, lg, st);
    route(lg, Y.rbias, d_ids, d_w, m, st);
    // the other group's admissions (victim slots of any layer) must have landed before this plan reads the slot table
    if (aj_rec[1 - grp]) CUDA_CHECK(cudaStreamWaitEvent(st, ev_aj[1 - grp], 0));
    if (side) CUDA_CHECK(cudaStreamWaitEvent(st, ev_side[grp][1], 0));
    if (E->bg_bound && !E->bg_perl && grp == 0 && (l == E->first_moe || (E->first_moe == -2 && l == 3))) k_bgq_wait<<<1, 1, 0, st>>>(E->bg_flag + p.rank, E->pfstat);  // round 11
    if (!side && rq) k_xreq<<<12, 256, 0, st>>>(xf, m * H / 4, rq);
    if (ah_rec) {  // round 15: this layer's ahead admissions (copied during the previous layer) before the plan
        CUDA_CHECK(cudaStreamWaitEvent(st, ev_ah, 0));
        ah_rec = false;
    }
    const bool bw = E->bg_bound && E->bg_perl && grp == 0;
    k_plan<<<1, 256, 0, st>>>(d_ids, d_w, m, Y.slot, slab, sl, J0, NJ, rq, xf, d_seq, false, bw ? d_bgdone : nullptr,
                              bw ? d_bgneed + l : nullptr);
    stampl(l, 1);
    side_rec[grp] = false;
    if (rq && l + 1 < E->nl && !L[l + 1].dense) {  // next layer's predicted experts (prefetch hint for the CPU lane)
        if (side) {
            CUDA_CHECK(cudaEventRecord(ev_side[grp][2], st));
            CUDA_CHECK(cudaStreamWaitEvent(pst, ev_side[grp][2], 0));
            router_dec(L[l + 1].router, xf, m, pred_part, pred_lg, pst);
            route(pred_lg, L[l + 1].rbias, d_pids, d_pw, m, pst);
            k_pub_pred<<<1, 32, 0, pst>>>(d_pids, d_pw, m, rq, d_seq);
            CUDA_CHECK(cudaEventRecord(ev_side[grp][3], pst));
            side_rec[grp] = true;
        } else {
            float* lg2 = big + (size_t)m * NE;
            router_dec(L[l + 1].router, xf, m, rpart, lg2, st);
            route(lg2, L[l + 1].rbias, d_pids, d_pw, m, st);
            k_pub_pred<<<1, 32, 0, st>>>(d_pids, d_pw, m, rq, d_seq);
        }
    }
    // admission copies (PCIe-bound) on a high-priority stream, concurrent with the shared and hot experts: victims are
    // never experts of this layer's rows (nor the previous request's admissions), and the admitted jobs / count live
    // apart from the hot ones
    cudaStream_t as = E->admit_par ? ast : st;
    if (as != st) {
        CUDA_CHECK(cudaEventRecord(ev_af[grp], st));
        CUDA_CHECK(cudaStreamWaitEvent(as, ev_af[grp], 0));
    }
    k_wait_dec<<<1, 1, 0, as>>>(E->dec + qi, d_seq);
    k_admit<<<dim3(MXE, 48), 256, 0, as>>>(E->dec + qi, E->hx_layer[l], E->hxl.stride, E->hxl.off[p.rank], slab, sl,
                                         E->hxl.sl[p.rank], E->hxl.sl3[p.rank], E->hxl.mixed ? E->hxl.flag_off : 0);
    k_admit_fin<<<1, 32, 0, as>>>(E->dec + qi, slots_all, slab, sl, J0 + MXE, NJ + 1, E->hx_layer[l],
                                  E->hxl.stride, E->hxl.off[p.rank]);
    if (as != st) {
        CUDA_CHECK(cudaEventRecord(ev_aj[grp], as));
        aj_rec[grp] = true;
        if (E->admit_rule == 2) {  // background copies on their own low-priority stream, joined at the end of the step
            CUDA_CHECK(cudaStreamWaitEvent(bst, ev_aj[grp], 0));
            k_admit<<<dim3(MXE, E->bg_blocks), 256, 0, bst>>>(E->dec + qi, E->hx_layer[l], E->hxl.stride, E->hxl.off[p.rank], slab, sl,
                                                  E->hxl.sl[p.rank], E->hxl.sl3[p.rank], E->hxl.mixed ? E->hxl.flag_off : 0, 1);
            k_admit_bgfin<<<1, 32, 0, bst>>>(E->dec + qi, slots_all);
            CUDA_CHECK(cudaEventRecord(ev_bg, bst));
            bg_rec = true;
        }
    }
    if (ah_on && l + 1 < E->nl && !L[l + 1].dense) {  // round 15: next layer's ahead admissions, after ours (not in ev_aj);
                                                       // ah_on implies admit_par (as != st), review r15 L8
        k_wait_dec<<<1, 1, 0, as>>>(E->dec_ah + l, d_seq);
        k_admit<<<dim3(MXE, E->ah_blocks), 256, 0, as>>>(E->dec_ah + l, E->hx_layer[l + 1], E->hxl.stride, E->hxl.off[p.rank], slab, sl,
                                                       E->hxl.sl[p.rank], E->hxl.sl3[p.rank], E->hxl.mixed ? E->hxl.flag_off : 0, 1);
        k_ah_fin<<<1, 32, 0, as>>>(E->dec_ah + l, slots_all);
        CUDA_CHECK(cudaEventRecord(ev_ah, as));
        ah_rec = true;
    }
    {   // shared expert
        float* gub = big;
        lin(Y.sw_gu, xf, xb, gub, m);
        bf16* hb = bigb;
        glu(gub, hb, p.ns(), m, st);
        lin(Y.sw_d, nullptr, hb, part, m);
    }
    expert_gemv(J0, MXE, xf, gu, part, ni, m, st, NJ, xhb, xyp, nullptr);
    stampl(l, 3);
}
// second half: admitted experts, the CPU lane's rows (rank 0)
void Worker::moe_post(int l, int m, bool early, bool nolane) {
    const int ni = p.ne(), qi = grp * (NL + 1) + l;
    XJob* J0 = d_jobs + grp * 2 * MXE;
    int* NJ = d_nj + grp * 2;
    if (E->admit_par) CUDA_CHECK(cudaStreamWaitEvent(st, ev_aj[grp], 0));
    stampl(l, 4);
    expert_gemv(J0 + MXE, MXE, xf, gu, part, ni, m, st, NJ + 1, xhb, xyp, nullptr);
    stampl(l, 5);
    if (side_rec[grp]) CUDA_CHECK(cudaStreamWaitEvent(st, ev_side[grp][3], 0));  // xf is rewritten after the MoE
    side_rec[grp] = false;
    if (early) {  // GLM_AR_EARLY: all-reduce the GPU partials while the lane runs, then every card adds the lane rows
        if (E->stamp_lvl == 4) stampl(l, 13);
        allreduce(m);
        k_wait_add<<<16, 256, 0, st>>>(E->out + qi, d_seq, sum, m, E->stamps && grp == 0 ? E->stamps + l * 16 + 10 : nullptr,
                                       E->wait_old ? nullptr : d_wflag + 2 * qi);
    } else if (p.rank == 0 && !nolane)
        k_wait_add<<<16, 256, 0, st>>>(E->out + qi, d_seq, part, m, E->stamps && grp == 0 ? E->stamps + l * 16 + 10 : nullptr,
                                       E->wait_old ? nullptr : d_wflag + 2 * qi);
    stampl(l, 2);
}

const float* Worker::wuk32(const Layer& Y, cudaStream_t s) {
    if (Y.wuk) return Y.wuk;
    b2f(Y.wuk_b, wu32, (size_t)p.nh() * DN * KVLR, s);
    return wu32;
}
const float* Worker::wuv32(const Layer& Y, cudaStream_t s) {
    if (Y.wuv) return Y.wuv;
    float* d = wu32 + (size_t)p.nh() * DN * KVLR;
    b2f(Y.wuv_b, d, (size_t)p.nh() * DV * KVLR, s);
    return d;
}
void Worker::join_bg() {
    if (bg_rec) CUDA_CHECK(cudaStreamWaitEvent(st, ev_bg, 0));
    bg_rec = false;
}
static bool is_eos(int t) { return t == EOS0 || t == EOS1 || t == EOS2; }
__global__ void k_set_posb(int* pb, const int* p, int a) { *pb = *p + a; }

// c > 1: one step over the rows of two sequences (ma rows of sequence 0 at rows [0, ma), mb of sequence 1 after them)
void Worker::pair_body(int ma, int mb) {
    const int m = ma + mb, s0 = cur_seq;
    k_seq_inc<<<1, 1, 0, st>>>(d_seq);
    ar_idx = E->p2p && env_int("GLM_AR_P2P", 1) ? 0 : -1;
    CUDA_CHECK(cudaMemcpyAsync(d_tok, E->h_tok, (size_t)m * 4, cudaMemcpyHostToDevice, st));
    embed(d_tok, E->emb, R, m, st);
    aj_rec[0] = aj_rec[1] = false;
    grp = 0;
    ah_on = ah_rec = false;  // review r15 L2: ahead admissions and the lane deferral are plain step_body only
    def_qi = -1;
    wave_body(m, ma, true);
    join_bg();
    ar_idx = -1;
    for (int g = 0; g < 2; g++) {  // final norm into each sequence's own hfin (the MTP catch-up input)
        bind(g);
        view(g ? ma : 0);
        glm::final_norm(R, final_norm, xb, hfin, g ? mb : ma, st);
    }
    view(0);
    bind(s0);
    const int nv = p.v1 - p.v0;
    lin(lm, nullptr, xb, logits, m);
    argmax_rows(logits, nv, m, d_am2, st);
    CUDA_CHECK(cudaMemcpyAsync(E->h_am2 + (size_t)p.rank * 4 * MR, d_am2, (size_t)m * 8, cudaMemcpyDeviceToHost, st));
}
void Worker::pair_step(int ma, int mb) {
    run_graph(gpair[ma][mb], st, [&] { pair_body(ma, mb); }, E->use_graph);
    CUDA_CHECK(cudaStreamSynchronize(st));
}

// intra-step wavefront (lab 4a): group 0 = rows [0, wa), group 1 = rows [wa, m). Per MoE layer the GPU issues
// g0.post(l), g0.attn(l+1), g0.pre(l+1), g1.post(l), g1.attn(l+1), g1.pre(l+1), so the CPU lane always has the other
// group's cold experts queued while the GPU runs one group's attention. Group 1's attention at l needs group 0's KV /
// KDA snapshot at l, written earlier in stream order.
void Worker::wave_body(int m, int wa, bool pair) {
    // pair: group g = sequence g (own KDA / MLA state, own positions; the groups do not attend to each other)
    const int mb = m - wa;
    if (!pair) k_set_posb<<<1, 1, 0, st>>>(d_posb, d_pos, wa);
    auto blk0 = [&](int l, int g) {
        Layer& Y = L[l];
        const int r0 = g ? wa : 0, T = g ? mb : wa;
        grp = g;
        if (pair) bind(g);
        view(r0);
        mhc_mix(R, Y.hc_fn[0], mix, T, st);
        glm::mhc_pre(R, mix, Y.hc_base[0], Y.hc_scale[0], Y.in_ln, post, comb, xf, xb, T, st, true);
        if (Y.mla) mla_dec(l, T, pair || !g ? d_pos : d_posb);
        else kda(l, T, 0, true, pair ? 0 : r0);
        allreduce(T);
        glm::merge(R, post, comb, sum, T, st);
    };
    auto pre = [&](int l, int g) {
        Layer& Y = L[l];
        const int r0 = g ? wa : 0, T = g ? mb : wa;
        grp = g;
        view(r0);
        mhc_mix(R, Y.hc_fn[1], mix, T, st);
        glm::mhc_pre(R, mix, Y.hc_base[1], Y.hc_scale[1], Y.post_ln, post, comb, xf, xb, T, st, true);
        moe_pre(l, T);
    };
    auto post_ = [&](int l, int g) {
        const int r0 = g ? wa : 0, T = g ? mb : wa;
        grp = g;
        view(r0);
        moe_post(l, T);
        allreduce(T);
        glm::merge(R, post, comb, sum, T, st);
    };
    int l0 = 0;
    if (pair)  // dense layers per sequence (own attention state), no CPU work
        for (; l0 < E->nl && L[l0].dense; l0++) {
            Layer& Y = L[l0];
            for (int g = 0; g < 2; g++) {
                const int r0 = g ? wa : 0, T = g ? mb : wa;
                grp = 0;
                bind(g);
                view(r0);
                for (int blk = 0; blk < 2; blk++) {
                    mhc_mix(R, Y.hc_fn[blk], mix, T, st);
                    glm::mhc_pre(R, mix, Y.hc_base[blk], Y.hc_scale[blk], blk ? Y.post_ln : Y.in_ln, post, comb, xf, xb, T, st, true);
                    if (blk == 0) {
                        if (Y.mla) mla_dec(l0, T);
                        else kda(l0, T, 0, true, 0);
                    } else mlp(l0, T);
                    allreduce(T);
                    glm::merge(R, post, comb, sum, T, st);
                }
            }
        }
    for (; l0 < E->nl && L[l0].dense; l0++) {  // dense layers: all rows together
        Layer& Y = L[l0];
        grp = 0;
        view(0);
        for (int blk = 0; blk < 2; blk++) {
            mhc_mix(R, Y.hc_fn[blk], mix, m, st);
            glm::mhc_pre(R, mix, Y.hc_base[blk], Y.hc_scale[blk], blk ? Y.post_ln : Y.in_ln, post, comb, xf, xb, m, st, true);
            if (blk == 0) {
                if (Y.mla) mla_dec(l0, m);
                else kda(l0, m, 0, true);
            } else mlp(l0, m);
            allreduce(m);
            glm::merge(R, post, comb, sum, m, st);
        }
    }
    if (l0 < E->nl) {
        blk0(l0, 0); pre(l0, 0);
        blk0(l0, 1); pre(l0, 1);
    }
    for (int l = l0; l < E->nl; l++) {
        post_(l, 0);
        if (l + 1 < E->nl) { blk0(l + 1, 0); pre(l + 1, 0); }
        post_(l, 1);
        if (l + 1 < E->nl) { blk0(l + 1, 1); pre(l + 1, 1); }
    }
    grp = 0;
    view(0);
}

// MLA at decode (m rows from the device position: graph-replayable)
void Worker::mla_dec(int l, int m, const int* dp) {
    if (!dp) dp = d_pos;
    Layer& Y = L[l];
    const int nh = p.nh();
    const int ldq = nh * DN + IH * ID;
    float* proj = big;
    float* wi = proj + (size_t)m * MLA_IN;
    float* qq = wi + (size_t)m * IH;
    float* qlat = qq + (size_t)m * ldq;
    float* ctxv = qlat + (size_t)m * nh * KVLR;
    float* ov = ctxv + (size_t)m * nh * KVLR;
    float* sc = ov + (size_t)m * nh * DV;
    bf16* qr = bigb;
    const bool sx = E->stamp_lvl == 5 && l != MTP_L;  // round 26 (GLM_STAMP=5): phase stamps of the MLA layers
    lin(Y.mw_in, xf, xb, proj, m);
    if (sx) stamp(3);
    mla_prep(proj, Y.q_a_ln, Y.kv_a_ln, Y.ik_w, Y.ik_b, qr, Y.kc, Y.ki, Y.gsc, wi, 0, m, st, dp, E->kring);
    if (sx) stamp(4);
    lin(Y.mw_q, nullptr, qr, qq, m, -1, false, -1, E->idx_skip ? dp : nullptr, nh * DN);  // r27: indexer rows below 2K
    if (sx) stamp(5);
    pool_keys_dec(Y.ki, Y.gsc, Y.ape, Y.pk, dp, m, st, E->kring);
    if (sx) stamp(6);
    const int ld = E->ctx / POOL;
    idx_score(qq + nh * DN, ldq, wi, Y.pk, sc, ld, 0, m, st, dp);
    if (sx) stamp(7);
    select_tokens(sc, ld, 0, d_sel, d_nsel, m, st, dp);
    if (sx) stamp(8);
    const float one = 1.f, zero = 0.f;
    const bool ubf = E->mla_ubf && Y.wuk_b && m <= MR;
    if (ubf) mla_uk_bf16(Y.wuk_b, qq, ldq, qlat, nh, m, st);
    else CUBLAS_CHECK_(cublasSgemmStridedBatched(cb, CUBLAS_OP_N, CUBLAS_OP_N, KVLR, m, DN, &one, wuk32(Y, st), KVLR,
                                            (long long)DN * KVLR, qq, ldq, DN, &zero, qlat, nh * KVLR, KVLR, nh));
    if (E->mla_tc > 0 && m <= MR && nh <= 24) mla_attn_tc_dec(qlat, Y.kc, d_sel, d_nsel, ctxv, nh, 0, m, 1.f / 16.f, st, dp, mla_scr, E->mla_tc);
    else if (E->mla_split && m <= MR) mla_attn_split(qlat, Y.kc, d_sel, d_nsel, ctxv, nh, 0, m, 1.f / 16.f, st, dp, mla_scr);
    else mla_attn(qlat, Y.kc, d_sel, d_nsel, ctxv, nh, 0, m, 1.f / 16.f, st, dp);
    if (ubf) mla_uv_bf16(Y.wuv_b, ctxv, ov, nh, m, st);
    else CUBLAS_CHECK_(cublasSgemmStridedBatched(cb, CUBLAS_OP_T, CUBLAS_OP_N, DV, m, KVLR, &one, wuv32(Y, st), KVLR,
                                            (long long)DV * KVLR, ctxv, nh * KVLR, KVLR, &zero, ov, nh * DV, DV, nh));
    if (sx) stamp(12);
    bf16* ovb = bigb;
    f2b(ov, ovb, (size_t)m * nh * DV, st);
    lin(Y.mo, nullptr, ovb, part, m);
    if (sx) stamp(13);
}

void Worker::step_body(int m, bool samp) {
    k_seq_inc<<<1, 1, 0, st>>>(d_seq);
    if (E->bg_bound && (E->first_moe == -1 || E->first_moe == -3)) k_bgq_wait<<<1, 1, 0, st>>>(E->bg_flag + p.rank, E->pfstat);  // at the graph head
    if (E->bg_bound && E->first_moe == -2) k_bgq_wait<<<1, 1, 0, st>>>(E->bg_flag + 4 + p.rank, E->pfstat);  // debug: copies only
    stampl(NL, 0);
    ar_idx = E->p2p && env_int("GLM_AR_P2P", 1) ? 0 : -1;
    CUDA_CHECK(cudaMemcpyAsync(d_tok, E->h_tok, (size_t)m * 4, cudaMemcpyHostToDevice, st));
    if (E->bg_bound && E->bg_perl) CUDA_CHECK(cudaMemcpyAsync(d_bgneed, E->bg_need, (NL + 1) * 4, cudaMemcpyHostToDevice, st));
    embed(d_tok, E->emb, R, m, st);
    aj_rec[0] = aj_rec[1] = false;
    grp = 0;
    def_qi = -1;
    ah_rec = false;
    ah_on = E->admit_ahead > 0 && E->admit_par && !E->wave;  // the coordinator publishes dec_ah under the same test
    const int wa = E->wave && m >= 2 ? std::min(E->wave_a, m - 1) : 0;
    if (wa) wave_body(m, wa);
    else for (int l = 0; l < E->nl; l++) {
        Layer& Y = L[l];
        stp = E->stamps2 && ((E->stamp_lvl == 2 && !Y.dense && !Y.mla) || (E->stamp_lvl == 5 && Y.mla))
                  ? E->stamps2 + ((size_t)p.rank * (NL + 1) + l) * 16 : nullptr;  // round 26: 5 = MLA layers (D4 gate)
        // round 15: defer this layer's lane rows (not the last layer: the final norm needs them)
        const bool dfr = E->lane_defer && !Y.dense && l + 1 < E->nl;
        for (int blk = 0; blk < 2; blk++) {
            if (E->dbg_delay[0] == p.rank && E->dbg_delay[1] == l && E->dbg_delay[2] == blk)
                k_spin<<<1, 32, 0, st>>>((unsigned long long)E->dbg_delay[3] * 1000);
            if (!blk) stamp(0);
            const bool first = l == 0 && blk == 0;
            // round 15 check mode (GLM_LANE_DEFER=2): the pending lane rows go in before this block's mHC, recomputed
            const bool dchk = !blk && def_qi >= 0 && E->lane_defer == 2;
            if (dchk) {
                if (late_rec) {  // review r15 L3: the late prediction still reads R (streams after layer l - 1)
                    CUDA_CHECK(cudaStreamWaitEvent(st, ev_late[1], 0));
                    late_rec = false;
                }
                const MoeOut* o2 = (E->def_split ? E->out2 : E->out) + def_qi;
                k_wait_out<<<1, 1, 0, st>>>(o2, d_seq, nullptr);
                k_def_add<<<16, 256, 0, st>>>(o2, R, pdef, m);
                def_qi = -1;
            }
            const bool fused = E->fuse_mix && !first && !dchk;  // the previous merge produced this block's mixes
            if (!(E->fuse_bnd && !first && !dchk)) {  // fused boundary: the previous block's kernel already produced xf / xb
                if (!fused) mhc_mix(R, Y.hc_fn[blk], mix, m, st);
                if (!blk) stamp(1);
                glm::mhc_pre(R, mix, Y.hc_base[blk], Y.hc_scale[blk], blk ? Y.post_ln : Y.in_ln, post, comb, xf, xb, m, st,
                             true, nullptr, fused ? mixp : nullptr);
            } else if (!blk) stamp(1);
            if (!blk) stamp(2);
            if (!blk) stampl(l, 6);
            if (blk == 0) {
                if (Y.mla) mla_dec(l, m);
                else kda(l, m, 0, true);
            } else {
                stamp(11);
                if (Y.dense) mlp(l, m);
                else if (E->ar_early) {
                    moe_pre(l, m);
                    moe_post(l, m, true);
                } else if (dfr) {  // round 15: no lane wait here; the rows are added at the next layer's attention merge
                    moe_pre(l, m);
                    moe_post(l, m, false, !E->def_split);  // split: rank 0 still adds the exact part (out) here
                    k_cp4<<<1, 32, 0, st>>>(post, pdef, m * 4);
                    def_qi = grp * (NL + 1) + l;
                } else moe_dec(l, m);
            }
            const bool done_ar = blk && !Y.dense && E->ar_early;  // the MoE all-reduce ran inside moe_post
            if (E->stamp_lvl == 4 && !done_ar) stampl(l, blk ? 13 : 12);  // all-reduce entry
            if (!blk && late_rec) {  // the late prediction read R (streams after layer l - 1) before this merge
                CUDA_CHECK(cudaStreamWaitEvent(st, ev_late[1], 0));
                late_rec = false;
            }
            if (!done_ar) allreduce(m);
            if (!blk && def_qi >= 0) {  // GLM_LANE_DEFER=1: layer l - 1's lane rows, before this attention's merge
                const MoeOut* o2 = (E->def_split ? E->out2 : E->out) + def_qi;
                k_wait_out<<<1, 1, 0, st>>>(o2, d_seq, E->stamps && p.rank == 0 ? E->stamps + l * 16 + 14 : nullptr);
                k_def_add<<<16, 256, 0, st>>>(o2, R, pdef, m);
                def_qi = -1;
            }
            stampl(l, blk ? 8 : 7);
            if (!blk) stamp(9);
            const float* fnn = blk == 0 ? Y.hc_fn[1] : l + 1 < E->nl ? L[l + 1].hc_fn[0] : nullptr;
            if (E->fuse_bnd && fnn) {  // merge + the next block's mHC in one kernel
                Layer& Z = blk == 0 ? Y : L[l + 1];
                const int nb = blk == 0 ? 1 : 0;
                glm::boundary(R, post, comb, sum, Z.hc_fn[nb], Z.hc_base[nb], Z.hc_scale[nb], nb ? Z.post_ln : Z.in_ln, xf, xb,
                              m, bnd_scr, bnd_bar, st);
            } else if (E->fuse_mix && fnn) glm::merge_mix(R, post, comb, sum, fnn, mixp, m, st);
            else glm::merge(R, post, comb, sum, m, st);
            if (blk) stampl(l, 9);
            if (!blk) stamp(10);
            if (blk && E->pf_late && p.rank == 0 && l + 1 < E->nl && !L[l + 1].dense && !L[l].dense) {
                // round 9: late prediction of layer l + 1's routing from the streams after layer l (published to req[l])
                Layer& Z = L[l + 1];
                CUDA_CHECK(cudaEventRecord(ev_late[0], st));
                CUDA_CHECK(cudaStreamWaitEvent(pst, ev_late[0], 0));
                mhc_mix(R, Z.hc_fn[1], lmix, m, pst);
                glm::mhc_pre(R, lmix, Z.hc_base[1], Z.hc_scale[1], Z.post_ln, lpost, lcomb, lxf, lxb, m, pst, true, nullptr, nullptr);
                router_dec(Z.router, lxf, m, pred_part, pred_lg, pst);
                route(pred_lg, Z.rbias, d_lids, lw, m, pst);
                k_pub_late<<<1, 32, 0, pst>>>(d_lids, lw, m, E->req + l, d_seq);
                CUDA_CHECK(cudaEventRecord(ev_late[1], pst));
                late_rec = true;
            }
        }
        stp = nullptr;
    }
    if (late_rec) { CUDA_CHECK(cudaStreamWaitEvent(st, ev_late[1], 0)); late_rec = false; }
    join_bg();
    ar_idx = -1;
    stampl(NL, 2);
    glm::final_norm(R, final_norm, xb, hfin, m, st);
    const int nv = p.v1 - p.v0;
    lin(lm, nullptr, xb, logits, m);
    if (samp) {
        topk_rows(logits, nv, m, E->h_invT, E->h_tk + (size_t)p.rank * MR, st);
        return;
    }
    argmax_rows(logits, nv, m, d_am, st);
    CUDA_CHECK(cudaMemcpyAsync(E->h_am + (size_t)p.rank * 2 * MR, d_am, (size_t)m * 8, cudaMemcpyDeviceToHost, st));
    stampl(NL, 3);
}

// MTP layer 45 for m rows: inputs d_mtok (token t + 1), mh (final normed hidden of row t), position d_mpos;
// writes the MTP cache rows, mhn (head-normed output, the next chain input) and the argmax per row
void Worker::moe_mtp(int m) {
    Layer& Y = L[MTP_L];
    const int ni = p.ne();
    float* lg = big;
    sgemm(Y.router, xf, lg, NE, m, H);
    route(lg, Y.rbias, d_ids, d_w, m, st);
    if (E->mtp_count) k_count<<<1, 32, 0, st>>>(d_ids, m * TOPK, mtp_cnt);
    k_plan<<<1, 256, 0, st>>>(d_ids, d_w, m, Y.slot, mtp_slab, sl, d_jobs, d_nj, nullptr, xf, d_seq);
    float* gub = big;
    lin(Y.sw_gu, xf, xb, gub, m);
    bf16* hb = bigb;
    glu(gub, hb, p.ns(), m, st);
    lin(Y.sw_d, nullptr, hb, part, m);
    expert_gemv(d_jobs, MXE, xf, gu, part, ni, m, st, d_nj, xhb, xyp, nullptr);
}
__global__ void k_seq_inc2(unsigned* s) { *s += 1; }


// MTP layer pass over m rows (inputs mh rows, d_mtok, position *dpos): cache rows, MoE, head, argmax + normalisers in
// d_am / d_mse. Returns the rows those results have (GLM_MTP_LAST1: 1, the last row moved to row 0)
int Worker::mtp_core(int m, const int* dpos) {
    Layer& Y = L[MTP_L];
    mtp_in(d_mtok, E->emb, mh, mw.enorm, mw.hnorm, xcat, m, st);
    lin(mw.eh, nullptr, xcat, mx_, m);
    add_rms(mx_, nullptr, Y.in_ln, xf, xb, m, st);
    mla_dec(MTP_L, m, dpos);
    allreduce(m);
    add_rms(mx_, sum, Y.post_ln, xf, xb, m, st);
    // round 17 (advisor D1, GLM_MTP_LAST1=0 off): only the last row's draft is read; the earlier rows of a catch-up pass
    // needed their cache rows (written by mla_dec above), not their MoE / head. The last row moves to row 0.
    if (E->mtp_last1 && m > 1) {
        CUDA_CHECK(cudaMemcpyAsync(xf, xf + (size_t)(m - 1) * H, (size_t)H * 4, cudaMemcpyDeviceToDevice, st));
        CUDA_CHECK(cudaMemcpyAsync(xb, xb + (size_t)(m - 1) * H, (size_t)H * 2, cudaMemcpyDeviceToDevice, st));
        CUDA_CHECK(cudaMemcpyAsync(mx_, mx_ + (size_t)(m - 1) * H, (size_t)H * 4, cudaMemcpyDeviceToDevice, st));
        m = 1;
    }
    moe_mtp(m);
    allreduce(m);
    add_rms(mx_, sum, mw.head_norm, mhn, xb, m, st);
    const W8& lh = lm_d.N ? lm_d : lm4.q4 ? lm4 : lm;
    lin(lh, nullptr, xb, logits, m);
    argmax_rows(logits, lh.N, m, d_am, st);
    row_se(logits, lh.N, m, d_am, d_mse, st);
    return m;
}

void Worker::mtp_body(int m) {
    k_seq_inc2<<<1, 1, 0, st>>>(d_mseq);
    ar_seq = d_mseq;
    ar_idx = E->p2p && env_int("GLM_AR_P2P", 1) ? 200 : -1;  // tags disjoint from the step's (idx < 200)
    CUDA_CHECK(cudaMemcpyAsync(d_mtok, E->h_mtok, (size_t)m * 4, cudaMemcpyHostToDevice, st));
    CUDA_CHECK(cudaMemcpyAsync(d_mpos, E->h_mpos, 4, cudaMemcpyHostToDevice, st));
    m = mtp_core(m, d_mpos);
    CUDA_CHECK(cudaMemcpyAsync(E->h_mam + (size_t)p.rank * 2 * MR, d_am, (size_t)m * 8, cudaMemcpyDeviceToHost, st));
    CUDA_CHECK(cudaMemcpyAsync(E->h_mse + (size_t)p.rank * MR, d_mse, (size_t)m * 4, cudaMemcpyDeviceToHost, st));
    ar_idx = -1;
    ar_seq = nullptr;
}

// ---- round 19 [glm-mtp-dev]: the draft chain on the device ------------------------------------------------------------
// this card's (max logit, token, normaliser) of row `row` into its slot of a zeroed 3 x ngpu vector (the all-reduce sum
// is then an exact gather: x + 0 = x)
__global__ void k_chain_put(const int* am, const float* mse, int row, const int* dmap, int v0, int rank, int ngpu, float* part) {
    const int i = threadIdx.x;
    if (i >= 3 * ngpu) return;
    float v = 0.f;
    if (i / 3 == rank) {
        const int f = i % 3, ix = am[row * 2 + 1];
        v = f == 0 ? __int_as_float(am[row * 2]) : f == 1 ? (float)(dmap ? dmap[ix] : ix + v0) : mse[row];
    }
    part[i] = v;
}
// the host's pick of mtp_pass / spec_draft on the gathered values: best card (first of the largest), draft token, MTP
// confidence, the gate (product of confidences >= tau) and the next pass's condition
__global__ void k_chain_pick(const float* g, int ngpu, int j, int k, float tau, int* res, double* q,
                             cudaGraphConditionalHandle next, int has_next) {
    int best = 0;
    for (int r = 1; r < ngpu; r++)
        if (g[r * 3] > g[best * 3]) best = r;
    const double M = g[best * 3];
    double Z = 0;
    // (review r19 L4: no contraction, the host's separate rounding of each product and sum)
    for (int r = 0; r < ngpu; r++) Z = __dadd_rn(Z, __dmul_rn((double)g[r * 3 + 2], exp(__dadd_rn((double)g[r * 3], -M))));
    const float conf = (float)__ddiv_rn(1.0, fmax(Z, 1.0));
    res[j] = (int)g[best * 3 + 1];
    res[MR + j] = __float_as_int(conf);
    int kc;
    if (j == 0) {
        *q = conf;
        kc = tau > 0 && *q < tau ? 0 : 1;
    } else {
        *q = __dmul_rn(*q, (double)conf);
        kc = res[2 * MR];
        if (!(tau > 0 && *q < tau)) kc = j + 1;
    }
    res[2 * MR] = kc;
    if (has_next) cudaGraphSetConditional(next, kc == j + 1 && j + 1 < k ? 1u : 0u);
}

void Worker::chain_pick(int row, int j, int k, void* next) {
    const int* dm = lm_d.N ? d_dmap : nullptr;
    k_chain_put<<<1, 32, 0, st>>>(d_am, d_mse, row, dm, lm_d.N ? 0 : p.v0, p.rank, p.ngpu, part);
    if (E->ar_ll12) ar_ll12(part, sum, 3 * p.ngpu, par, p.rank, p.ngpu, ar_seq, ar_idx++, st);
    else ar_ll(part, sum, 3 * p.ngpu, par, p.rank, p.ngpu, ar_seq, ar_idx++, st);
    cudaGraphConditionalHandle h = next ? *(cudaGraphConditionalHandle*)next : 0;
    k_chain_pick<<<1, 1, 0, st>>>(sum, p.ngpu, j, k, E->conf_tau, d_chres, (double*)(d_chres + 2 * MR + 2), h, next != nullptr);
    if (E->chain_chk)  // the exchanged values of this pass, for the host comparison
        CUDA_CHECK(cudaMemcpyAsync(E->h_chres + (size_t)(E->ngpu + p.rank) * 64 + j * 3 * E->ngpu, sum,
                                   (size_t)3 * p.ngpu * 4, cudaMemcpyDeviceToHost, st));
}

// an IF node at the current point of the capture on `st`, its body captured (on chs) by f; the handle is created in
// the graph being captured and set by the kernel just before
static cudaGraphConditionalHandle cond_handle(cudaStream_t s) {
    cudaStreamCaptureStatus cs;
    cudaGraph_t g;
    CUDA_CHECK(cudaStreamGetCaptureInfo(s, &cs, nullptr, &g, nullptr, nullptr, nullptr));
    cudaGraphConditionalHandle h;
    CUDA_CHECK(cudaGraphConditionalHandleCreate(&h, g, 0, cudaGraphCondAssignDefault));
    return h;
}
static void cond_if(cudaStream_t s, cudaStream_t body_st, cudaGraphConditionalHandle h, const std::function<void()>& f) {
    cudaStreamCaptureStatus cs;
    cudaGraph_t g;
    const cudaGraphNode_t* deps;
    const cudaGraphEdgeData* ed;
    size_t nd;
    CUDA_CHECK(cudaStreamGetCaptureInfo(s, &cs, nullptr, &g, &deps, &ed, &nd));
    cudaGraphNodeParams np = {};
    np.type = cudaGraphNodeTypeConditional;
    np.conditional.handle = h;
    np.conditional.type = cudaGraphCondTypeIf;
    np.conditional.size = 1;
    cudaGraphNode_t node;
    CUDA_CHECK(cudaGraphAddNode(&node, g, deps, ed, nd, &np));
    CUDA_CHECK(cudaStreamUpdateCaptureDependencies(s, &node, nullptr, 1, cudaStreamSetCaptureDependencies));
    cudaGraph_t body = np.conditional.phGraph_out[0];
    CUDA_CHECK(cudaStreamBeginCaptureToGraph(body_st, body, nullptr, nullptr, 0, cudaStreamCaptureModeThreadLocal));
    f();
    CUDA_CHECK(cudaStreamEndCapture(body_st, &body));
}

// catch-up pass over a rows (+ the fused commit) and chain passes 2..k, all on the device
void Worker::chain_body(int a, int k) {
    if (E->commit_in_mtp) commit_body();
    k_seq_inc2<<<1, 1, 0, st>>>(d_mseq);
    ar_seq = d_mseq;
    ar_idx = 200;
    CUDA_CHECK(cudaMemcpyAsync(d_mtok, E->h_mtok, (size_t)a * 4, cudaMemcpyHostToDevice, st));
    CUDA_CHECK(cudaMemcpyAsync(d_chpos, E->h_mpos, (size_t)k * 4, cudaMemcpyHostToDevice, st));
    const int mm = mtp_core(a, d_chpos);
    std::function<void(int, int)> pass = [&](int j, int row) {  // pick of pass j (its results in row), then pass j + 1
        if (j + 1 >= k) { chain_pick(row, j, k, nullptr); return; }
        cudaGraphConditionalHandle h = cond_handle(st);
        chain_pick(row, j, k, &h);
        cudaStream_t cur = st;
        cudaStream_t body = chs[j];  // (each nesting level captures on a stream of its own)
        cond_if(cur, body, h, [&] {
            st = body;
            CUBLAS_CHECK_(cublasSetStream(cb, st));
            CUBLAS_CHECK_(cublasSetWorkspace(cb, cbws, 32 << 20));  // review r19 M1: SetStream drops the workspace
            const float* src = E->chain_raw ? mx_ : mhn;
            const int srow = j == 0 && !E->mtp_last1 ? a - 1 : 0;  // (as spec_draft: chain from the last row's output)
            CUDA_CHECK(cudaMemcpyAsync(mh, src + (size_t)srow * H, (size_t)H * 4, cudaMemcpyDeviceToDevice, st));
            CUDA_CHECK(cudaMemcpyAsync(d_mtok, d_chres + j, 4, cudaMemcpyDeviceToDevice, st));
            mtp_core(1, d_chpos + j + 1);
            pass(j + 1, 0);
            st = cur;
            CUBLAS_CHECK_(cublasSetStream(cb, st));
            CUBLAS_CHECK_(cublasSetWorkspace(cb, cbws, 32 << 20));  // review r19 M1: SetStream drops the workspace
        });
    };
    pass(0, mm - 1);
    CUDA_CHECK(cudaMemcpyAsync(E->h_chres + (size_t)p.rank * 64, d_chres, (size_t)(2 * MR + 1) * 4, cudaMemcpyDeviceToHost, st));
    ar_idx = -1;
    ar_seq = nullptr;
}

void Worker::mtp(int m, bool bg) {
    if (E->commit_in_mtp) run_graph(gcmtp[m - 1], st, [&] { commit_body(); mtp_body(m); }, E->use_graph);
    else run_graph(gmtp[m - 1], st, [&] { mtp_body(m); }, E->use_graph);
    if (bg) E->bg_enqueue(*this);
    CUDA_CHECK(cudaStreamSynchronize(st));
}

void Worker::commit_body() {
    CUDA_CHECK(cudaMemcpyAsync(d_acc, E->h_acc, 4, cudaMemcpyHostToDevice, st));
    if (nkda) kda_commit(d_kS, d_ksnap, d_khist, d_krows, nkda, p.nh() * KD * KD, 3 * p.nh() * KD, d_acc, st, E->kda_ptr);
    k_pos_add<<<1, 1, 0, st>>>(d_pos, d_acc);
}

void Worker::step(int m) {
    if (E->samp_T > 0) run_graph(gstep_s[m - 1], st, [&] { step_body(m, true); }, E->use_graph);
    else run_graph(gstep[m - 1], st, [&] { step_body(m); }, E->use_graph);
    CUDA_CHECK(cudaStreamSynchronize(st));
}
void Worker::commit() {
    run_graph(gcommit, st, [&] { commit_body(); }, E->use_graph);
    CUDA_CHECK(cudaStreamSynchronize(st));
}

void Engine::bind_seq(int s) {
    if (s == cur_seq) return;
    EngSeq& o = eseq[cur_seq];
    o.pos = pos; o.sp_t = sp_t; o.sp_k = sp_k; o.sp_kcur = sp_kcur; o.sp_reused = sp_reused; o.last_tok = last_tok;
    for (int i = 0; i < MR; i++) { o.sp_d[i] = sp_d[i]; o.sp_c[i] = sp_c[i]; }
    o.sp_hist.swap(sp_hist);
    o.sp_lout.swap(sp_lout);
    EngSeq& n = eseq[s];
    pos = n.pos; sp_t = n.sp_t; sp_k = n.sp_k; sp_kcur = n.sp_kcur; sp_reused = n.sp_reused; last_tok = n.last_tok;
    for (int i = 0; i < MR; i++) { sp_d[i] = n.sp_d[i]; sp_c[i] = n.sp_c[i]; }
    sp_hist.swap(n.sp_hist);
    sp_lout.swap(n.sp_lout);
    run_all([&](Worker& w) { w.bind(s); });
    cur_seq = s;
}
// rows of both sequences in one step (greedy): argmax per row -> oa[ma], ob[mb]; nothing committed
void Engine::verify2(const int* ra, int ma, const int* rb, int mb, int* oa, int* ob) {
    CHECK(nseq == 2 && ma >= 1 && mb >= 1 && ma <= MR && mb <= MR, "verify2 %d + %d rows", ma, mb);
    CHECK(eseq[cur_seq == 0 ? 1 : 0].pos + (cur_seq == 0 ? mb : ma) <= ctx && pos + (cur_seq == 0 ? ma : mb) <= ctx,
          "context full");
    if (!coord.joinable()) { coord_start(); policy_init(); }
    if (!h_am2) CUDA_CHECK(cudaHostAlloc(&h_am2, sizeof(int) * 4 * MR * 4, cudaHostAllocPortable));
    for (int i = 0; i < ma; i++) h_tok[i] = ra[i];
    for (int i = 0; i < mb; i++) h_tok[ma + i] = rb[i];
    CHECK(!lent, "verify2 with the prefill set active");
    wave_cur.store(ma, std::memory_order_release);
    det_wait();  // review r27 H1: a background re-seed after a prefill has landed
    steps_issued++;
    coord_gate.set(coord_go, (unsigned)steps_issued);  // [glm-idle] wakes a sleeping coordinator
    run_all([&](Worker& w) { w.pair_step(ma, mb); });
    for (int i = 0; i < ma + mb; i++) {
        int best = 0;
        for (int r = 1; r < ngpu; r++)
            if (((float*)h_am2)[(r * 2 * MR + i) * 2] > ((float*)h_am2)[(best * 2 * MR + i) * 2]) best = r;
        const int t = h_am2[(best * 2 * MR + i) * 2 + 1] + W[best]->p.v0;
        if (i < ma) oa[i] = t; else ob[i - ma] = t;
    }
}
void Engine::spec_step2(int* outa, int* na, int* outb, int* nb) {
    int rows[2][MR], m[2], o[2][MR], pos0[2];
    for (int s = 0; s < 2; s++) {
        bind_seq(s);
        m[s] = 1 + sp_kcur;
        rows[s][0] = sp_t;
        for (int j = 0; j < sp_kcur; j++) rows[s][1 + j] = sp_d[j];
        pos0[s] = pos;
    }
    bind_seq(0);
    const auto tv = std::chrono::steady_clock::now();
    verify2(rows[0], m[0], rows[1], m[1], o[0], o[1]);
    st_ver_us += std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - tv).count();
    for (int s = 0; s < 2; s++) {
        bind_seq(s);
        int a = 1;
        while (a < m[s] && o[s][a - 1] == rows[s][a]) a++;
        commit(a);
        st_steps++;
        st_rows += m[s];
        st_acc += a;
        for (int j = 0; j + 1 < a; j++) st_dacc[j]++;
        sp_hist.insert(sp_hist.end(), rows[s], rows[s] + a);
        int* out = s ? outb : outa;
        for (int j = 0; j < a; j++) out[j] = o[s][j];
        *(s ? nb : na) = a;
        sp_t = o[s][a - 1];
        if (pos + 1 + sp_k < ctx) spec_draft(a, o[s], pos0[s], 1);
        else sp_kcur = 0;
    }
    bind_seq(0);
}
// two greedy generations at once (c = 2): n tokens each at most; got[2] = tokens produced
int Engine::generate2(const int* pa, int ta, const int* pb, int tb, int n, int k, int* outa, int* outb, int* got) {
    CHECK(nseq == 2, "generate2 needs GLM_SEQS=2 at load");
    int* outs[2] = {outa, outb};
    const int* pr[2] = {pa, pb};
    const int tl[2] = {ta, tb};
    bool done[2] = {false, false};
    for (int s = 0; s < 2; s++) {
        bind_seq(s);
        sp_hist.clear();
        pos = 0;
        const int t = spec_start(pr[s], tl[s], k);
        got[s] = 0;
        outs[s][got[s]++] = t;
        done[s] = is_eos(t) || got[s] >= n;
    }
    bind_seq(0);
    const auto td = std::chrono::steady_clock::now();
    int ba[MR], bb[MR], na = 0, nb = 0;
    while (!done[0] || !done[1]) {
        if (!done[0] && !done[1]) spec_step2(ba, &na, bb, &nb);
        else {
            const int s = done[0] ? 1 : 0;
            bind_seq(s);
            const int a = spec_step(s ? bb : ba);
            (s ? nb : na) = a;
            (s ? na : nb) = 0;
            bind_seq(0);
        }
        for (int s = 0; s < 2; s++) {
            const int c = s ? nb : na;
            const int* b = s ? bb : ba;
            for (int j = 0; j < c && !done[s]; j++) {
                outs[s][got[s]++] = b[j];
                if (is_eos(b[j]) || got[s] >= n) done[s] = true;
            }
        }
    }
    st_dec_us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - td).count();
    return got[0] + got[1];
}

// m rows (tokens at pos .. pos + m - 1) -> greedy argmax of every row (out[m]). Not committed, and with the KDA pointer
// commit (GLM_KDA_PTR, default) not repeatable: the rows overwrite the snapshot bank the next verify starts from, so
// every verify must be followed by commit(a >= 1) (or the fused commit of spec_step) before the next verify / prefill
void Engine::verify(const int* toks, int m, int* outm) {
    CHECK(m >= 1 && m <= MR && pos + m <= ctx, "verify %d rows at %d", m, pos);
    sp_undo = false;  // [glm-rel-stops] the verify overwrites the per-row states a trim needs
    CHECK(!ver_pending, "verify: the previous verify was not committed (KDA pointer mode)");  // r12 (review C4)
    bg_flush();
    if (!coord.joinable()) {
        coord_start();
        policy_init();
    }
    for (int i = 0; i < m; i++) h_tok[i] = toks[i];
    CHECK(W[0]->mx == 2 * MR && !lent, "verify with the prefill set active");
    mla_need(pos + m);  // round 26: a step that reaches the next MLA page takes it first (every 4,096 positions)
    wave_cur.store(wave && m >= 2 ? std::min(wave_a, m - 1) : 0, std::memory_order_release);
    det_wait();
    det_dirty = true;  // round 17: admissions and counts move during the step
    steps_issued++;
    coord_gate.set(coord_go, (unsigned)steps_issued);  // [glm-idle] wakes a sleeping coordinator
    if (bg_bound && bg_sync) run_all([&](Worker& w) { CUDA_CHECK(cudaStreamSynchronize(w.bst)); });  // debug
    if (hts) hts[NL * 8 + 0] = rt_ns();
    run_all([&](Worker& w) { w.step(m); });
    ver_pending = kda_ptr;
    if (dbg_slotchk) {  // debug: GPU slot tables vs the host's slot_of (deferred entries of this step still pending)
        std::vector<int> pend(slot_of.size(), 0);
        for (int j = 0; j < bgq->n; j++) pend[bgq->l[j] * NE + bgq->e[j]] = 1;
        for (auto& wp : W) {
            Worker* w = &*wp;
            std::vector<int> t((size_t)NL * NE);
            CUDA_CHECK(cudaSetDevice(w->dev));
            CUDA_CHECK(cudaMemcpy(t.data(), w->slots_all, t.size() * 4, cudaMemcpyDeviceToHost));
            int bad = 0;
            for (int k = 3 * NE; k < NL * NE; k++) {
                const int want = pend[k] ? -1 : slot_of[k];
                if (t[k] != want) {
                    if (bad < 4) fprintf(stderr, "[glm] slotchk step %llu rank %d: l %d e %d gpu %d host %d (pend %d)\n",
                                         (unsigned long long)st_steps, w->p.rank, k / NE, k % NE, t[k], slot_of[k], pend[k]);
                    bad++;
                }
            }
            if (bad) fprintf(stderr, "[glm] slotchk step %llu rank %d: %d mismatches\n", (unsigned long long)st_steps, w->p.rank, bad);
        }
        CUDA_CHECK(cudaSetDevice(W[0]->dev));
    }
    if (bg_bound) bg_prep();  // r12 (review P1): enqueued after the first MTP graph launch (mtp_pass) or by bg_flush()
    if (dbg_hog[0] > 0 || dbg_hog[2] > 0) bg_due = true;
    for (int s : bg_pl) bg_pend[s] = 0;  // the step's background copies are joined (or queued before the next plans)
    bg_pl.clear();
    if (bg_perl) {  // round 17: queued boundary copies land during the NEXT verify (per-layer waits), so their slots
                    // stay out of the victim sampling until that verify ends; the previous queue's copies have landed
        for (int s : bg_qprev) bg_pend[s] = 0;
        bg_qprev.clear();
        if (bg_due)
            for (int j = 0; j < bg_n; j++) { bg_pend[bg_ql->slot[j]] = 1; bg_qprev.push_back(bg_ql->slot[j]); }
    }
    if (hts) hts[NL * 8 + 1] = rt_ns();
    if (stamps2 && stamp_lvl == 4) {
        if (!stamp_dump) if (const char* sd = getenv("GLM_STAMP_DUMP")) stamp_dump = fopen(sd, "wb");
        if (stamp_dump) {
            fwrite(stamps2, 8, (size_t)4 * (NL + 1) * 16, stamp_dump);
            fflush(stamp_dump);
        }
    } else if (stamps2 && stamp_lvl == 5) {  // round 26: MLA layer phases (all ranks, mean), us summed per step
        if (st2_acc.empty()) st2_acc.assign(16, 0.0);
        static const int seq[] = {2, 3, 4, 5, 6, 7, 8, 12, 13, 9};
        for (int l = 0; l < nl; l++) {
            if (!is_mla(l)) continue;
            for (int i = 1; i < 10; i++) {
                double a = 0;
                for (int r = 0; r < ngpu; r++) {
                    const unsigned long long* q = stamps2 + ((size_t)r * (NL + 1) + l) * 16;
                    a += ((long long)q[seq[i]] - (long long)q[seq[i - 1]]) * 1e-3;
                }
                st2_acc[i] += a / ngpu;
            }
        }
        st2_n++;
    } else if (stamps2) {  // fine split of the KDA attention part (all ranks) and the all-reduce entry skew
        if (st2_acc.empty()) st2_acc.assign(16, 0.0);
        for (int l = 4; l < nl; l++) {
            if (is_mla(l)) continue;
            for (int i = 1; i <= 11; i++) {
                if (i == 9) {  // AR: entry skew = max - min of the ranks' stamp 8; duration on the slowest-entry rank
                    unsigned long long mn = ~0ull, mxv = 0;
                    for (int r = 0; r < ngpu; r++) {
                        const unsigned long long e = stamps2[((size_t)r * (NL + 1) + l) * 16 + 8];
                        mn = std::min(mn, e); mxv = std::max(mxv, e);
                    }
                    st2_acc[12] += (mxv - mn) * 1e-3;
                }
                double a = 0;
                for (int r = 0; r < ngpu; r++) {
                    const unsigned long long* q = stamps2 + ((size_t)r * (NL + 1) + l) * 16;
                    a += (q[i] - q[i - 1]) * 1e-3;
                }
                st2_acc[i] += a / ngpu;
            }
        }
        st2_n++;
    }
    if (stamps) {  // GPU-side split of the step (rank 0): attention part (wait_add(l) -> MoE input of l + 1),
                   // router + plan, MoE phase (plan -> CPU lane result added)
        for (int l = 3; l + 1 < nl; l++) {
            const unsigned long long* a = stamps + l * 16, *b = stamps + (l + 1) * 16;
            st_g_attn += (b[0] - a[2]) * 1e-3;
            st_g_plan += (b[1] - b[0]) * 1e-3;
            st_g_moe += (b[2] - b[1]) * 1e-3;
        }
        if (stamp_dump) {  // raw dump: per step [NL + 1][16] GPU ns, [NL + 1][8] host ns, rows
            fwrite(stamps, 8, (NL + 1) * 16, stamp_dump);
            fwrite(hts, 8, (NL + 1) * 8, stamp_dump);
            long long mm = m;
            fwrite(&mm, 8, 1, stamp_dump);
            fflush(stamp_dump);
        }
        if (stamp_lvl == 3) {  // detailed per-layer timeline, layers 4..nl-2 (MoE), us summed over the layers
            if (st3.empty()) st3.assign(34, 0.0);
            auto d = [](unsigned long long x, unsigned long long y) { return ((double)x - (double)y) * 1e-3; };
            for (int l = 4; l + 1 < nl; l++) {
                const unsigned long long* a = stamps + l * 16, *b = stamps + (l + 1) * 16;
                const long long* h = hts + l * 8, *hn = hts + (l + 1) * 8;
                const int o = is_mla(l) ? 16 : 0;  // KDA / MLA layers apart
                st3[o + 0] += d(a[7], a[6]);   // attention incl. its all-reduce
                st3[o + 1] += d(a[0], a[7]);   // boundary to the MoE block (merge + mHC)
                st3[o + 2] += d(a[1], a[0]);   // router + route + plan (request published)
                st3[o + 3] += d(a[3], a[1]);   // shared + hot experts
                st3[o + 4] += d(a[4], a[3]);   // admission copies wait
                st3[o + 5] += d(a[5], a[4]);   // admitted experts
                st3[o + 6] += d(a[2], a[5]);   // wait for the CPU lane (+ add)
                st3[o + 7] += d(a[8], a[2]);   // MoE all-reduce
                st3[o + 8] += d(b[6], a[8]);   // boundary to the next layer's attention
                st3[o + 9] += d(b[1], a[2]);   // GPU chain: lane added (l) -> request (l + 1)
                st3[o + 10] += d(h[0], a[1]);  // GPU request store -> host sees it (clock offset included)
                st3[o + 11] += (double)(h[1] - h[0]) * 1e-3;  // host: admission decision
                st3[o + 12] += (double)(h[2] - h[1]) * 1e-3;  // host: lane run (incl. prefetch stop)
                st3[o + 13] += d(a[2], h[2]);  // host publish -> GPU added (clock offset included)
                st3[o + 14] += (double)(hn[0] - h[2]) * 1e-3;  // host-measured: lane done (l) -> request (l + 1)
                st3[o + 15] += 1;
                if (lane_defer == 1) st3[32 + (o ? 1 : 0)] += d(b[15], b[14]);  // round 15: GPU wait for layer l's lane rows at l + 1
            }
        }
        st_g_n++;
    }
    if (samp_T > 0) {   // rows' distributions via row_dist (spec_step)
        for (int i = 0; i < m; i++) outm[i] = -1;
        return;
    }
    for (int i = 0; i < m; i++) {
        int best = 0;
        for (int r = 1; r < ngpu; r++)
            if (((float*)h_am)[(r * MR + i) * 2] > ((float*)h_am)[(best * MR + i) * 2]) best = r;
        outm[i] = h_am[(best * MR + i) * 2 + 1] + W[best]->p.v0;
    }
}
// r12 (review P1): the host half of the boundary copies (snapshot of the queue by value, flags) runs right after the
// verify; the per-card enqueue (DMA + untile per entry, ~6 API calls each) runs after the first MTP graph is launched
// (mtp_pass), so the GPU no longer idles while the host issues it. bg_flush() issues it at every other entry point.
void Engine::bg_prep() {
    const int n = bgq->n;
    if (!n) return;
    if (!bg_ql) bg_ql = new BgList;
    int ord[BgQ::CAP];
    for (int j = 0; j < n; j++) ord[j] = j;
    if (bg_perl) std::stable_sort(ord, ord + n, [&](int a, int b) { return bgq->l[a] < bgq->l[b]; });  // round 17
    for (int i = 0; i < n; i++) {
        const int j = ord[i];
        bg_ql->l[i] = (short)bgq->l[j]; bg_ql->e[i] = (short)bgq->e[j]; bg_ql->slot[i] = bgq->slot[j];
    }
    if (bg_perl) {  // plan(l) of the next verify waits for the copies of layers <= l
        bg_cum0 = bg_cum;
        int c = 0;
        for (int l = 0; l <= NL; l++) {
            while (c < n && bg_ql->l[c] <= l) c++;
            bg_need[l] = bg_cum + c;
        }
        bg_cum += n;
    }
    for (int r = 0; r < ngpu; r++) bg_flag[r] = bg_flag[4 + r] = 1;
    std::atomic_thread_fence(std::memory_order_seq_cst);
    bg_n = n;
    bg_due = true;
    st_bgq += n;
    bgq->n = 0;
}
void Engine::bg_enqueue(Worker& w) {
    const int n = bg_n;
    if (n) {
        const BgList& ql = *bg_ql;
        if (bg_dma) {  // copy engine into the landing ring + untile (no long-running SM blocks next to the MTP pass)
            for (int j = 0; j < n; j++) {
                const char* base = hx_layer[ql.l[j]] + (size_t)ql.e[j] * hxl.stride;  // review S2: from the snapshot
                const SliceL& hs = hxl.lay(base, w.p.rank);
                const int k = j % Worker::NLAND;
                CUDA_CHECK(cudaStreamWaitEvent(w.dmst, w.ev_lfree[k], 0));
                CUDA_CHECK(cudaMemcpyAsync(w.land[k], base + hxl.off[w.p.rank], hs.bytes, cudaMemcpyHostToDevice, w.dmst));
                CUDA_CHECK(cudaEventRecord(w.ev_land[k], w.dmst));
                CUDA_CHECK(cudaStreamWaitEvent(w.bst, w.ev_land[k], 0));
                untile(w.land[k], w.slab + (size_t)ql.slot[j] * w.sl.bytes, w.sl, 64, w.bst, &hs);
                CUDA_CHECK(cudaEventRecord(w.ev_lfree[k], w.bst));
                if (bg_perl) k_bgq_one<<<1, 1, 0, w.bst>>>(ql.l[j] * NE + ql.e[j], ql.slot[j], w.slots_all, w.d_bgdone, bg_cum0 + j + 1);
            }
        } else
            k_bgq<<<dim3(n, bg_bound_blk), 256, 0, w.bst>>>(ql, n, d_hxl, hxl.stride, hxl.off[w.p.rank], w.slab, w.sl,
                                                           hxl.sl[w.p.rank], hxl.sl3[w.p.rank], hxl.mixed ? hxl.flag_off : 0);
        if (first_moe <= -2) {  // debug: copies done -> flag[4 + rank], then a 1 ms gap before the slot table update
            k_bgq_fin<<<1, 32, 0, w.bst>>>(ql, 0, w.slots_all, bg_flag + 4 + w.p.rank);
            k_spin<<<1, 32, 0, w.bst>>>(1000000ull);
        }
        // review r17 L1: under GLM_BG_PERL every entry was written by its k_bgq_one; a second write here could restore an
        // entry the next verify already evicted (nothing waits for this kernel any more), so it only clears the flag
        k_bgq_fin<<<1, 32, 0, w.bst>>>(ql, bg_perl ? 0 : n, w.slots_all, bg_flag + w.p.rank);
    }
    if (dbg_hog[0] > 0) k_spin<<<dbg_hog[0], 256, 0, w.bst>>>((unsigned long long)dbg_hog[1] * 1000);
    for (int j = 0; j < dbg_hog[2]; j++) {  // PCIe hog: DMA into the landing ring, nothing consumes it. r12 (review C5):
        const int k = j % Worker::NLAND;    // waits for the ring block's last untile, so no queued slice is overwritten
        CUDA_CHECK(cudaStreamWaitEvent(w.dmst, w.ev_lfree[k], 0));
        CUDA_CHECK(cudaMemcpyAsync(w.land[k], hx_layer[3] + (size_t)j * hxl.stride + hxl.off[w.p.rank],
                                   hxl.sl[w.p.rank].bytes, cudaMemcpyHostToDevice, w.dmst));
    }
}
void Engine::bg_flush() {
    if (!bg_due) return;
    run_all([&](Worker& w) { bg_enqueue(w); });
    bg_due = false;
    bg_n = 0;
}
void Engine::kda_mat() {
    run_all([&](Worker& w) {
        if (w.nkda && w.d_acc) {
            kda_materialize(w.d_kS, w.d_ksnap, w.nkda, w.p.nh() * KD * KD, w.d_acc, w.st);
            CUDA_CHECK(cudaStreamSynchronize(w.st));
        }
    });
}
void Engine::commit(int a) {
    CHECK(a >= 1, "commit(%d): at least the verified row", a);  // r12 (review C4): commit(0) would read a stale state
    bg_flush();
    ver_pending = false;
    h_acc[0] = a;
    run_all([&](Worker& w) { w.commit(); });
    pos += a;
}
int Engine::decode(int tok) {
    int o;
    verify(&tok, 1, &o);
    commit(1);
    return o;
}

// MTP pass over m rows: h rows from src (device, per worker: hfin rows, mhn last row, or mh already set), tokens,
// position of row 0 -> argmax of the last row
static int mtp_pass(Engine* E, int m, const int* toks, int pos0, int src, int srow = 0, float* conf = nullptr) {
    const bool bg = E->bg_due;  // r12: the step's boundary copies are enqueued right after this graph's launch
    for (int i = 0; i < m; i++) E->h_mtok[i] = toks[i];
    E->h_mpos[0] = pos0;
    E->run_all([&](Worker& w) {
        if (src == 1) CUDA_CHECK(cudaMemcpyAsync(w.mh, w.hfin, (size_t)m * H * 4, cudaMemcpyDeviceToDevice, w.st));
        if (src == 2) CUDA_CHECK(cudaMemcpyAsync(w.mh, (E->chain_raw ? w.mx_ : w.mhn) + (size_t)srow * H, (size_t)H * 4,
                                                 cudaMemcpyDeviceToDevice, w.st));
        w.mtp(m, bg);
    });
    E->bg_due = false;
    if (bg) E->bg_n = 0;
    int best = 0;
    const int i = E->mtp_last1 ? 0 : m - 1;  // round 17: the last row's results sit in row 0 under GLM_MTP_LAST1
    for (int r = 1; r < E->ngpu; r++)
        if (((float*)E->h_mam)[(r * MR + i) * 2] > ((float*)E->h_mam)[(best * MR + i) * 2]) best = r;
    if (conf) {  // MTP softmax probability of the drafted token
        const double M = ((float*)E->h_mam)[(best * MR + i) * 2];
        double Z = 0;
        for (int r = 0; r < E->ngpu; r++) Z += E->h_mse[r * MR + i] * std::exp(((float*)E->h_mam)[(r * MR + i) * 2] - M);
        *conf = (float)(1.0 / std::max(Z, 1.0));
    }
    const Worker& wb = *E->W[best];
    const int ix = E->h_mam[(best * MR + i) * 2 + 1];
    return wb.lm_d.N ? wb.h_dmap[ix] : ix + wb.p.v0;
}


// Speculative decoding state machine (greedy, MTP chain drafts): spec_start = prefix reuse + prefill + first drafts,
// spec_step = verify [t, d1..dk] -> accept a, commit, MTP catch-up over the accepted rows (cache rows + draft 1), chain
// drafts 2..k from the MTP's own output. generate() and the server's C API (qe_start3 / qe_spec_step) use it.
static inline float h_i2f(int v) { float f; memcpy(&f, &v, 4); return f; }
static inline int h_f2i(float f) { int v; memcpy(&v, &f, 4); return v; }
// round 19 [glm-mtp-dev]: the chain in one graph per card; same drafts, confidences and gate as the passes below
static void chain_run(Engine* E, int a, const int* toks, int pos0, int src) {
    const int k = E->sp_k;
    const bool bg = E->bg_due;
    for (int i = 0; i < a; i++) E->h_mtok[i] = toks[i];
    E->h_mpos[0] = pos0;
    for (int j = 1; j < k; j++) E->h_mpos[j] = pos0 + a - 1 + j;
    const int cm = E->commit_in_mtp ? 1 : 0;
    E->run_all([&](Worker& w) {
        if (src == 1) CUDA_CHECK(cudaMemcpyAsync(w.mh, w.hfin, (size_t)a * H * 4, cudaMemcpyDeviceToDevice, w.st));
        run_graph(w.gchain[cm][k - 2][a - 1], w.st, [&] { w.chain_body(a, k); }, true, false);
        if (bg) E->bg_enqueue(w);
        CUDA_CHECK(cudaStreamSynchronize(w.st));
    });
    E->bg_due = false;
    if (bg) E->bg_n = 0;
    E->commit_in_mtp = false;
    const int* r = E->h_chres;
    for (int j = 0; j < k; j++) { E->sp_d[j] = r[j]; E->sp_c[j] = h_i2f(r[MR + j]); }
    E->sp_kcur = r[2 * MR];
    if (E->chain_chk) {  // the host pick (mtp_pass / spec_draft) from the exchanged values of each pass that ran
        const int ng = E->ngpu;
        double q = 0;
        int kc = 0;
        for (int j = 0; j < k && (j == 0 || kc == j); j++) {
            const float* g = (const float*)(r + (size_t)ng * 64 + j * 3 * ng);
            int best = 0;
            for (int x = 1; x < ng; x++) if (g[x * 3] > g[best * 3]) best = x;
            const double M = g[best * 3];
            double Z = 0;
            for (int x = 0; x < ng; x++) Z += g[x * 3 + 2] * std::exp(g[x * 3] - M);
            const float c = (float)(1.0 / std::max(Z, 1.0));
            if (j == 0) { q = c; kc = E->conf_tau > 0 && q < E->conf_tau ? 0 : 1; }
            else { q *= c; if (!(E->conf_tau > 0 && q < E->conf_tau)) kc = j + 1; }
            bool ok = (int)g[best * 3 + 1] == E->sp_d[j] && h_f2i(c) == h_f2i(E->sp_c[j]);
            for (int x = 1; x < ng; x++)  // every card picked the same
                ok &= !memcmp(r + (size_t)x * 64, r, (2 * MR + 1) * 4);
            if (!ok && E->st_chain_bad++ < 20)
                fprintf(stderr, "[glm] device chain check: pass %d differs (host %d %.9g, device %d %.9g)\n", j,
                        (int)g[best * 3 + 1], c, E->sp_d[j], E->sp_c[j]);
        }
        if (kc != E->sp_kcur && E->st_chain_bad++ < 20)
            fprintf(stderr, "[glm] device chain check: kc host %d device %d\n", kc, E->sp_kcur);
        E->st_chain_chk++;
    }
}

int Engine::spec_draft(int a, const int* toks, int pos0, int src) {
    sp_kcur = sp_k;
    if (!sp_k) return 0;
    const auto t0 = std::chrono::steady_clock::now();
    if (dev_chain && sp_k >= 2 && src != 2) {
        if (hts) for (int i = 2; i <= 5; i++) hts[NL * 8 + i] = 0;
        chain_run(this, a, toks, pos0, src);
        if (hts) hts[NL * 8 + 2] = hts[NL * 8 + 5] = rt_ns();
        st_mtp_us += std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
        return sp_k;
    }
    if (hts) for (int i = 2; i <= 5; i++) hts[NL * 8 + i] = 0;  // round 17 step edge: MTP passes of this step (host ns)
    sp_d[0] = mtp_pass(this, a, toks, pos0, src, 0, sp_c);
    if (hts) hts[NL * 8 + 2] = rt_ns();
    commit_in_mtp = false;  // catch-up: MTP cache rows of the accepted positions + draft 1
    // confidence gate (GLM_CONF_TAU): keep drafting while the product of the drafts' MTP probabilities stays >= tau
    // (each verify row costs ~8 ms of CPU-lane bytes); with tau = 0 the chain is the fixed k
    double q = sp_c[0];
    int kc = conf_tau > 0 && q < conf_tau ? 0 : 1;
    for (int j = 1; j < sp_k && kc == j; j++) {
        const int tk = sp_d[j - 1];
        sp_d[j] = mtp_pass(this, 1, &tk, pos0 + a - 1 + j, 2, j == 1 && !mtp_last1 ? a - 1 : 0, sp_c + j);  // chain from the last row's output
        if (hts) hts[NL * 8 + 2 + j] = rt_ns();
        q *= sp_c[j];
        if (!(conf_tau > 0 && q < conf_tau)) kc = j + 1;
    }
    sp_kcur = kc;
    if (hts) hts[NL * 8 + 5] = rt_ns();
    st_mtp_us += std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
    return sp_k;
}

int Engine::spec_start(const int* ids, int n, int k) {
    CHECK(mtp || k == 0, "spec_start: no MTP layer");
    if (!coord.joinable()) {
        coord_start();
        policy_init();
    }
    w8a8 = env_int("GLM_W8A8", 1) != 0;
    pf_max = env_int("GLM_PF_MAX", pf_max);
    prefetch = env_int("GLM_PREFETCH", 1) != 0;
    sp_k = std::min(k, MR - 1);
    // prefix reuse: the cache holds sp_hist (tokens at positions 0..pos-1); a prompt that extends it continues there,
    // else (round 18) from the prefix snapshot with the longest prefix it shares (glm_ckpt.cu)
    sp_undo = false;
    const bool want_lout = samp_T > 0 || (bool)sp_first;  // [glm-rel-mask] a constrained start reads them too
    if (want_lout) sp_lout.assign(VOCAB, 0.f);
    pf_aborted = false;
    sp_reused = ck_prefill(ids, n, want_lout ? sp_lout.data() : nullptr);
    sp_hint = 0;
    if (pf_aborted) return -1;  // [glm-rel-pfgate] cancelled mid-prompt: cache = the whole windows done (sp_hist)
    sp_t = last_tok;
    if (sp_first) {     // [glm-rel-mask] the server's mask (and the slot's sampling) picks the first token
        const int t = sp_first(sp_lout.data());
        if (t >= 0) sp_t = t;
    } else if (samp_T > 0) {   // first token: exact over the full vocabulary of the last prompt row
        std::vector<int> tk;
        std::vector<double> pp;
        full_dist(sp_lout.data(), tk, pp);
        sp_t = sample_from(tk, pp, -1);
    }
    sp_last_a = 0;  // (the prompt end has its own snapshot)
    spec_draft(1, &sp_t, n - 1, 0);  // mh row 0 = hidden of the last prompt row (set by the prefill)
    return sp_t;
}

int Engine::spec_step(int* outv) {
    StepPin pin_(this);
    // review r25 M1: rows past the reply's limit are not verified at all (fewer CPU-lane rows at the end and while a
    // thinking budget holds the limit at 1)
    const int m = sp_cap > 0 ? std::min(1 + sp_kcur, sp_cap) : 1 + sp_kcur;
    CHECK(pos + m <= ctx, "context full");
    int rows[MR], o[MR];
    rows[0] = sp_t;
    for (int j = 0; j + 1 < m; j++) rows[1 + j] = sp_d[j];
    const int pos0 = pos;
    const auto tv = std::chrono::steady_clock::now();
    verify(rows, m, o);
    const double ver_us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - tv).count();
    st_ver_us += ver_us;
    int a = 1;
    if (samp_T > 0) {   // speculative sampling with a point-mass (greedy) drafter
        std::uniform_real_distribution<double> U(0.0, 1.0);
        std::vector<int> tk;
        std::vector<double> pp;
        for (int j = 0; j < m; j++) {
            row_dist(j, tk, pp);
            if (j + 1 < m) {
                const int d = rows[j + 1];
                double pd = 0;
                for (size_t c = 0; c < tk.size(); c++) if (tk[c] == d) pd = pp[c];
                if (U(srng) < pd) { o[j] = d; continue; }
                o[j] = sample_from(tk, pp, d);
                st_samp_rej++;
                a = j + 1;
                break;
            }
            o[j] = sample_from(tk, pp, -1);
            a = m;
        }
    } else {
        while (a < m && o[a - 1] == rows[a]) a++;
    }
    const int a_nat = a;  // review r25 L2: the statistics count the model's acceptance, not the reply-end cut
    // round 25 ([glm-p3-turnend]): the reply ends inside this step (EOS, or the server's limit): rows past it are not
    // accepted, so the cache holds exactly the delivered reply (minus its last token) and the next turn extends it
    if (sp_eos_stop)
        for (int j = 0; j + 1 < a; j++)
            if (is_eos(o[j])) { a = j + 1; break; }
    if (sp_cap > 0 && a > sp_cap) a = sp_cap;
    sp_last_a = a;
    const bool undo = sp_watch && undo_prep();  // [glm-rel-stops] before the commit changes the conv history
    // round 11: the commit runs inside the first MTP pass graph when a draft pass follows (one host round trip less).
    // (review rel L6 tried the fused commit for watched steps too: the stop trim then failed its exactness gate,
    // glm_rel3 merge gate 13:42: reverted; a watched step keeps its own commit)
    if (fuse_commit && !undo && sp_k && pos + a + 1 + sp_k < ctx) {
        h_acc[0] = a;
        pos += a;
        commit_in_mtp = true;
        ver_pending = false;  // committed by the first MTP graph, which spec_draft launches next
    } else commit(a);
    if (conf_log) {
        fprintf(conf_log, "%d %d", m - 1, a_nat - 1);
        for (int j = 0; j < m - 1; j++) fprintf(conf_log, " %.4f", sp_c[j]);
        fprintf(conf_log, " %d %.0f\n", pos0, ver_us);  // round 15: position and verify us of the step
    }
    st_steps++;
    st_rows += m;
    st_acc += a_nat;
    for (int j = 0; j + 1 < a_nat; j++) st_dacc[j]++;
    sp_hist.insert(sp_hist.end(), rows, rows + a);  // the cache now holds these rows
    for (int j = 0; j < a; j++) outv[j] = o[j];
    sp_t = o[a - 1];
    if (pos + 1 + sp_k < ctx) spec_draft(a, o, pos0, 1);
    else sp_kcur = 0;
    sp_undo = undo;
    return a;
}

// ---------------------------------------------------------------- release track: constrained steps, stop trim

// review rel L13: the cards' slices are copied at once on each card's stream (the verify left them idle), then one
// sync per card; was a synchronous legacy-stream copy per card in turn (out is pinned: h_rowlg)
void Engine::row_logits(int j, float* out) {
    for (auto& w : W) {
        CUDA_CHECK(cudaSetDevice(w->dev));
        const size_t nv = (size_t)(w->p.v1 - w->p.v0);
        CUDA_CHECK(cudaMemcpyAsync(out + w->p.v0, w->logits + (size_t)j * nv, nv * 4, cudaMemcpyDeviceToHost, w->st));
    }
    for (auto& w : W) {
        CUDA_CHECK(cudaSetDevice(w->dev));
        CUDA_CHECK(cudaStreamSynchronize(w->st));
    }
    CUDA_CHECK(cudaSetDevice(W[0]->dev));
}

// [glm-rel-mask] a constrained step (grammar, tool_choice, forced think close, logprobs): the server plans the drafts
// (forced tokens, grammar-valid MTP drafts) and a mask per row; each row's full logits come to the host (only while the
// rows before it were accepted) and pick() chooses from them. Exact: a verify row is the model's distribution after its
// prefix, the mask only removes tokens. Cost: VOCAB x 4 bytes per read row and one host pass over them
int Engine::spec_step_pick(int* outv, const int* drafts, int nd, const std::function<int(int, float*)>& pick) {
    StepPin pin_(this);
    int m = 1 + std::max(0, std::min(nd, MR - 1));
    if (sp_cap > 0) m = std::min(m, sp_cap);
    CHECK(pos + m <= ctx, "context full");
    if (!h_rowlg) CUDA_CHECK(cudaHostAlloc(&h_rowlg, (size_t)VOCAB * 4, cudaHostAllocPortable));
    int rows[MR], o[MR];
    rows[0] = sp_t;
    for (int j = 0; j + 1 < m; j++) rows[1 + j] = drafts[j];
    const int pos0 = pos;
    const auto tv = std::chrono::steady_clock::now();
    verify(rows, m, o);
    int a = 0;
    for (int j = 0; j < m; j++) {
        row_logits(j, h_rowlg);
        int t = pick(j, h_rowlg);
        if (t < 0 || t >= VOCAB) {  // nothing allowed (the server's grammar is in its error state): the raw argmax
            t = 0;
            for (int i = 1; i < VOCAB; i++) if (h_rowlg[i] > h_rowlg[t]) t = i;
        }
        o[j] = t;
        a = j + 1;
        if (sp_eos_stop && is_eos(t)) break;
        if (j + 1 < m && t != rows[j + 1]) break;
    }
    st_ver_us += std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - tv).count();
    const int a_nat = a;
    if (sp_cap > 0 && a > sp_cap) a = sp_cap;
    if (sp_test_a > 0 && a > sp_test_a) a = sp_test_a;
    sp_test_a = 0;
    sp_last_a = a;
    const bool undo = sp_watch && undo_prep();
    commit(a);
    st_steps++;
    st_rows += m;
    st_acc += a_nat;  // (review rel L8: the natural acceptance, as spec_step)
    for (int j = 0; j + 1 < a_nat; j++) st_dacc[j]++;
    sp_hist.insert(sp_hist.end(), rows, rows + a);
    for (int j = 0; j < a; j++) outv[j] = o[j];
    sp_t = o[a - 1];
    if (pos + 1 + sp_k < ctx) spec_draft(a, o, pos0, 1);  // the MTP cache follows the picked tokens
    else sp_kcur = 0;
    sp_undo = undo;
    return a;
}

__global__ void k_hist_io(float* const* hist, float* bak, int n, bool save, int* pos, int back) {
    float* h = hist[blockIdx.x];
    float* b = bak + (size_t)blockIdx.x * n;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        if (save) b[i] = h[i];
        else h[i] = b[i];
    }
    if (!save && blockIdx.x == 0 && threadIdx.x == 0) *pos -= back;
}
__global__ void k_pos_back(int* pos, int back) { *pos -= back; }
bool Worker::hist_io(bool save, int back) {
    if (!nkda) {  // review rel L7: the device position moves back on a card without KDA layers too
        if (!save && back) {
            k_pos_back<<<1, 1, 0, st>>>(d_pos, back);
            CUDA_CHECK(cudaStreamSynchronize(st));
        }
        return true;
    }
    const int n = 3 * 3 * p.nh() * KD;  // conv_hist: 3 x 3 x HK floats per KDA layer
    if (!khbak) {
        if (!save) return false;
        if (cudaMalloc(&khbak, (size_t)nkda * n * 4) != cudaSuccess) {
            cudaGetLastError();
            khbak = nullptr;
            return false;
        }
    }
    k_hist_io<<<nkda, 256, 0, st>>>(d_khist, khbak, n, save, d_pos, back);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(st));
    return true;
}
bool Engine::undo_prep() {
    if (nseq != 1) return false;
    std::atomic<int> ok{1};
    run_all([&](Worker& w) { if (!w.hist_io(true)) ok = 0; });
    return ok.load() != 0;
}
// [glm-rel-stops] the client's text ends inside the last step: keep c cache rows (pos - a < c <= pos). The verify's
// per-row KDA states are intact until the next verify, the conv history comes back from its pre-commit copy and the
// commit runs again with c - pos0 rows: the state equals a step that accepted those rows (the round-25 cap path).
// MLA / MTP cache rows are position-indexed (rows past c are rewritten before they are read); the drafts of the cut
// step are dropped (the next step verifies its pending token alone)
bool Engine::spec_trim(int c) {
    if (c == pos) return true;
    const int a = sp_last_a, pos0 = pos - a;
    if (!sp_undo || ver_pending || a <= 0 || c <= pos0 || c > pos || (int)sp_hist.size() != pos) return false;
    const int r = c - pos0;
    std::atomic<int> ok{1};
    run_all([&](Worker& w) { if (!w.hist_io(false, a)) ok = 0; });
    CHECK(ok.load(), "trim: conv history copy missing");
    pos = pos0;
    commit(r);
    sp_t = sp_hist[c];
    sp_hist.resize(c);
    sp_last_a = r;
    sp_kcur = 0;
    sp_undo = false;
    if (lv_end() > pos) lv_cut(pos);
    return true;
}

int Engine::generate(const int* prompt, int T, int n, int k, int* outv) {
    sp_hist.clear();
    pos = 0;  // fresh conversation (no prefix reuse in the benchmark path)
    int t = spec_start(prompt, T, k);
    int produced = 0;
    outv[produced++] = t;
    const bool prof = env_int("GLM_PROF_DEC", 0) != 0;
    if (prof) run_all([](Worker&) { cudaProfilerStart(); });
    st_steps0 = st_steps; a_cpu0 = st_cpu; a_bg0 = st_bg; a_adm0 = st_admit; a_skip0 = st_skip; a_pfbg0 = st_pf_bg;
    a_ins0 = st_ins; a_ins_so0 = st_ins_so; a_bg_so0 = st_bg_so;
    for (auto& a : st_ah) a[0] = a[1] = a[2] = 0;
    a_ah_n0 = st_ah_n; a_ah_hit0 = st_ah_hit;
    a_defa0 = st_defa; a_defb0 = st_defb; a_defbr0 = st_defb_rows;
    a_pb0 = lane ? lane->pbytes.load() : 0;
    st_pf_cov = st_pf_use = 0;
    for (auto& w : st_pf_why) w = 0;
    st_bound_us = 0;
    st_pan = 0;
    for (auto& x : st_tl) x = 0;
    if (lane) for (auto& x : lane->tl_acc) x = 0;
    for (auto& a : st_pa) a[0] = a[1] = a[2] = 0;
    for (auto& u : st_reuse) u[0] = u[1] = 0;
    const auto td = std::chrono::steady_clock::now();
    int buf[MR];
    // round 14, bench only: fixed output length (same-length speed pairs). generate() only, not generate2() (review r14 L5)
    const bool ign = env_int("GLM_IGNORE_EOS", 0) != 0;
    // round 25 (gate of [glm-p3-ahead]): host time between steps as the server spends it (busy, us; 0 = none)
    const int gap_us = env_int("GLM_STEP_GAP_US", 0);
    while (produced < n && (ign || !is_eos(t))) {
        if (gap_us > 0 && produced > 1) {
            const auto tg = std::chrono::steady_clock::now();
            while (std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - tg).count() < gap_us) {}
        }
        const int a = spec_step(buf);
        for (int j = 0; j < a && produced < n; j++) {
            outv[produced++] = buf[j];
            t = buf[j];
            if (!ign && is_eos(t)) break;
        }
    }
    st_dec_us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - td).count();
    if (bg_bound && env_int("GLM_PFSTAT", 0)) fprintf(stderr, "[glm] bg boundary: %llu deferred copies; layer-%d waits that spun / ran, rank 0: %d / %d\n", st_bgq, first_moe, bg_flag[8], bg_flag[12]);
    if (prof) run_all([](Worker&) { cudaProfilerStop(); });
    // round 14 (review r13b M1): in-step / background admissions per step, skip-only ones. Review r14 L6: under EFFI the
    // in-step skip-only count is 0 by construction (it means something only with GLM_ADMIT_EFFI=0); the background
    // skip-only count includes the prefetch-background admissions
    if (env_int("GLM_ADMSTAT", 0)) {
        const double ns = std::max<double>(1, st_steps - st_steps0);
        if (det) fprintf(stderr, "[glm] det restores so far: %llu, %.1f slots copied and %.2f ms per restore (host side; asynchronous ones only enqueue)\n", st_det_n,
                         (double)st_det_slots / std::max(1ull, st_det_n), st_det_us / 1e3 / std::max(1ull, st_det_n));
        fprintf(stderr, "[glm] admissions per step: in-step %.2f (skip-only %.2f), background %.2f (skip-only %.2f)\n",
                (st_ins - a_ins0) / ns, (st_ins_so - a_ins_so0) / ns, (st_bg - a_bg0) / ns, (st_bg_so - a_bg_so0) / ns);
    }
    if (def_split) {
        const double ns = std::max<double>(1, st_steps - st_steps0);
        fprintf(stderr, "[glm] split deferral per step: lane experts exact at l %.1f, deferred to l + 1 %.1f (rows %.1f)\n",
                (st_defa - a_defa0) / ns, (st_defb - a_defb0) / ns, (st_defb_rows - a_defbr0) / ns);
    }
    if (admit_ahead) {
        const double ns = std::max<double>(1, st_steps - st_steps0);
        fprintf(stderr, "[glm] ahead admissions per step: %.2f, routed at the next layer %.2f\n", (st_ah_n - a_ah_n0) / ns,
                (st_ah_hit - a_ah_hit0) / ns);
    }
    if (ah_stat) {  // round 15 gate: admissions one layer ahead
        const double ns = std::max<double>(1, st_steps - st_steps0);
        fprintf(stderr, "[glm] ahead gate per step: in-step admissions %.2f in %.2f layers, effective cold %.2f; predicted "
                "top-1/2/4/all: listed %.2f/%.2f/%.2f/%.2f, effective cold %.2f/%.2f/%.2f/%.2f, admitted in-step %.2f/%.2f/%.2f/%.2f\n",
                st_ah[4][0] / ns, st_ah[4][1] / ns, st_ah[4][2] / ns, st_ah[0][0] / ns, st_ah[1][0] / ns, st_ah[2][0] / ns,
                st_ah[3][0] / ns, st_ah[0][1] / ns, st_ah[1][1] / ns, st_ah[2][1] / ns, st_ah[3][1] / ns, st_ah[0][2] / ns,
                st_ah[1][2] / ns, st_ah[2][2] / ns, st_ah[3][2] / ns);
    }
    if (stamps2 && st2_n && stamp_lvl == 5) {
        const char* nm[] = {"", "mw_in GEMV", "prep", "mw_q GEMV", "pool", "idx_score", "select", "uk+attn+uv", "f2b+mo GEMV",
                            "all-reduce"};
        // review r26-7: the real count of main-model MLA layers (mla_nl includes the MTP layer)
        fprintf(stderr, "[glm] MLA layers of a verify (%d layers, mean of ranks), us per step:", mla_nl - 1);
        double t = 0;
        for (int i = 1; i < 10; i++) { fprintf(stderr, " %s %.1f;", nm[i], st2_acc[i] / st2_n); t += st2_acc[i] / st2_n; }
        fprintf(stderr, " total %.1f (%llu steps)\n", t, st2_n);
        st2_acc.assign(16, 0.0);
        st2_n = 0;
    }
    if (stamps2 && st2_n) {
        const char* nm[] = {"", "mhc_mix", "mhc_pre", "KDA in GEMV", "f/g sgemm x2", "prep+conv", "recurrence", "out norm", "ko GEMV",
                            "all-reduce", "merge", "blk1 mix+pre", "AR entry skew"};
        fprintf(stderr, "[glm] KDA layer attention part, us per layer (mean of ranks):");
        for (int i = 1; i <= 12; i++) fprintf(stderr, " %s %.1f;", nm[i], st2_acc[i] / st2_n / 31);
        fprintf(stderr, "\n");
        st2_acc.assign(16, 0.0);
        st2_n = 0;
    }
    if (stamps && st_g_n) {
        fprintf(stderr, "[glm] GPU split per step (rank 0, layers 4..%d): attention part %.2f ms, router+plan %.2f ms, MoE phase "
                "(plan -> lane result added) %.2f ms\n", nl - 1, st_g_attn / st_g_n / 1e3, st_g_plan / st_g_n / 1e3,
                st_g_moe / st_g_n / 1e3);
        if (!st3.empty()) {
            const char* nm[] = {"attn+AR", "bnd0", "router+plan", "shared+hot", "admit wait", "admitted", "lane wait", "MoE AR",
                                "bnd1", "GPU chain", "req->host", "host decide", "host lane", "pub->GPU", "host chain"};
            for (int o = 0; o < 32; o += 16) {
                const double n = st3[o + 15];
                if (n <= 0) continue;
                fprintf(stderr, "[glm] %s layers (%.0f per step), us per layer:", o ? "MLA" : "KDA", n / st_g_n);
                for (int i = 0; i < 15; i++) fprintf(stderr, " %s %.1f;", nm[i], st3[o + i] / n);
                if (lane_defer == 1) fprintf(stderr, " deferred-lane wait at l + 1 %.1f;", st3[32 + (o ? 1 : 0)] / n);
                fprintf(stderr, "\n");
            }
            st3.assign(34, 0.0);
        }
        st_g_attn = st_g_plan = st_g_moe = 0;
        st_g_n = 0;
    }
    if (env_int("GLM_PFSTAT", 0)) {
        fprintf(stderr, "[glm] prefetch: covered %.1f experts, used %.1f (%.3f); by list position:", st_pf_cov, st_pf_use,
                st_pf_use / std::max(1e-9, st_pf_cov));
        for (int i = 0; i < 8; i++) fprintf(stderr, " %.0f/%.0f", st_pf_rank_use[i], st_pf_rank[i]);
        fprintf(stderr, "\n");
        const double ns = std::max<double>(1, st_steps - st_steps0), xg = (double)hxl.stride / 1e9;
        fprintf(stderr, "[glm] DRAM audit per step (GB): lane %.3f (bg-admitted also on the lane %.3f), read-through "
                "admitted %.3f, prefetch touched %.3f = used %.3f + unused %.3f [mispredicted %.3f, admitted %.3f, "
                "skipped by tau %.3f, hot %.3f]; skipped rows %.1f; prefetched -> background admissions %.2f; boundary (last lane -> layer-3 request) %.2f ms; lane experts also on the previous step's lane, layer 3 / 4 / 5 / 6+: %.2f / %.2f / %.2f / %.2f; prediction precision @1/2/4 early %.3f/%.3f/%.3f late %.3f/%.3f/%.3f\n",
                (st_cpu - a_cpu0) * xg / ns, (st_bg - a_bg0) * xg / ns, (st_admit - a_adm0 - (st_bg - a_bg0)) * xg / ns,
                (lane->pbytes.load() - a_pb0) / 1e9 / ns, st_pf_use * xg / ns, (st_pf_cov - st_pf_use) * xg / ns,
                st_pf_why[0] * xg / ns, st_pf_why[1] * xg / ns, st_pf_why[2] * xg / ns, st_pf_why[3] * xg / ns,
                (double)(st_skip - a_skip0) / ns, (double)(st_pf_bg - a_pfbg0) / ns, st_bound_us / 1e3 / ns,
                st_reuse[0][0] / std::max(1.0, st_reuse[0][1]), st_reuse[1][0] / std::max(1.0, st_reuse[1][1]),
                st_reuse[2][0] / std::max(1.0, st_reuse[2][1]), st_reuse[3][0] / std::max(1.0, st_reuse[3][1]),
                st_pa[0][0] / std::max(1.0, st_pan), st_pa[0][1] / std::max(1.0, st_pan), st_pa[0][2] / std::max(1.0, st_pan),
                st_pa[1][0] / std::max(1.0, st_pan), st_pa[1][1] / std::max(1.0, st_pan), st_pa[1][2] / std::max(1.0, st_pan));
        if (lane && lane->tl_on) {
            const double* A = lane->tl_acc;
            const double k = 1.0 / tsc_mhz / 1e3 / ns;  // ticks -> ms per step
            fprintf(stderr, "[glm] lane timeline per step (ms): %.1f calls; wake %.2f, quantize x + barrier %.2f, gate/up span %.2f (mean "
                    "thread idle %.2f), barriers + quantize h %.2f, down span %.2f (idle %.2f), tail %.2f, run_rows wall %.2f\n",
                    A[0] / ns, A[1] * k, A[2] * k, A[3] * k, A[4] * k, A[5] * k, A[6] * k, A[7] * k, A[8] * k, A[9] * k);
            fprintf(stderr, "[glm] coordinator per step (ms, layers 4..): request -> decisions %.2f, prefetch-stop wait %.2f, GPU windows %.2f "
                    "(lane end -> prefetch start %.2f, no prefetch %.2f), prefetch ran dry before the request %.0f of %.0f windows "
                    "(%.2f ms all threads, mean thread dry %.2f), last prefetch thread stops %.2f after the request\n",
                    st_tl[0] / 1e3 / ns, st_tl[1] / 1e3 / ns, st_tl[6] / 1e3 / ns, st_tl[8] / 1e3 / ns, st_tl[5] / 1e3 / ns,
                    st_tl[3], st_tl[7], st_tl[2] / 1e3 / ns, st_tl[4] / 1e3 / ns, st_tl[9] / 1e3 / ns);
        }
        st_pf_cov = st_pf_use = 0;
        for (int i = 0; i < 8; i++) st_pf_rank[i] = st_pf_rank_use[i] = 0;
        for (auto& w : st_pf_why) w = 0;
    }
    if (env_int("GLM_MEMLOG", 0))
        run_all([&](Worker& w) {
            size_t fr = 0, tot = 0;
            CUDA_CHECK(cudaMemGetInfo(&fr, &tot));
            fprintf(stderr, "[glm] card %d free after generate: %.0f MB\n", w.dev, fr / 1e6);
        });
    seq_end();  // round 17: the cache goes back to the seed in the background (off the next request's TTFT)
    return produced;
}

void Engine::mtp_mask(const std::vector<int>& resident) {
    std::vector<float> b(NE, -1e30f);
    for (int e : resident) b[e] = mtp_rbias[e];
    run_all([&](Worker& w) {
        CUDA_CHECK(cudaMemcpy(w.L[MTP_L].rbias, b.data(), NE * 4, cudaMemcpyHostToDevice));
    });
}

// ---------------------------------------------------------------- sampling

void Engine::set_sampling(float T, float top_p, int top_k, float min_p, unsigned long long seed) {
    samp_T = T > 0 ? T : 0.f;
    samp_top_p = top_p > 0 && top_p < 1 ? top_p : 1.f;
    samp_top_k = top_k > 0 ? top_k : 0;
    samp_min_p = min_p > 0 ? min_p : 0.f;
    if (h_invT) *h_invT = samp_T > 0 ? 1.f / samp_T : 1.f;
    srng.seed(seed ? seed : std::random_device{}());
}
// candidates sorted by logit desc (cl, ci), global max M and partition Z at 1/T -> truncated, renormalised p
static void truncate_dist(const std::vector<std::pair<float, int>>& c, double M, double Z, float invT, int top_k,
                          float top_p, float min_p, std::vector<int>& tok, std::vector<double>& p) {
    tok.clear(); p.clear();
    size_t n = c.size();
    if (top_k > 0) n = std::min(n, (size_t)top_k);
    double s = 0;
    for (size_t i = 0; i < n; i++) {
        const double q = std::exp(((double)c[i].first - M) * invT) / Z;
        tok.push_back(c[i].second); p.push_back(q); s += q;
    }
    for (auto& q : p) q /= s;
    if (top_p < 1) {
        double cum = 0;
        size_t k = 0;
        while (k < p.size()) { cum += p[k++]; if (cum >= top_p) break; }
        tok.resize(k); p.resize(k);
    }
    if (min_p > 0 && !p.empty()) {
        const double th = min_p * p[0];
        size_t k = 1;
        while (k < p.size() && p[k] >= th) k++;
        tok.resize(k); p.resize(k);
    }
    s = 0;
    for (auto q : p) s += q;
    for (auto& q : p) q /= s;
}
void Engine::row_dist(int j, std::vector<int>& tok, std::vector<double>& p) {
    const float invT = 1.f / samp_T;
    double M = -1e300;
    for (int r = 0; r < ngpu; r++) M = std::max(M, (double)h_tk[r * MR + j].mx);
    double Z = 0;
    std::vector<std::pair<float, int>> c;
    for (int r = 0; r < ngpu; r++) {
        const TopK& t = h_tk[r * MR + j];
        Z += (double)t.se * std::exp(((double)t.mx - M) * invT);
        for (int k = 0; k < TKC; k++) c.push_back({t.v[k], t.i[k] + W[r]->p.v0});
    }
    std::sort(c.begin(), c.end(), [](auto& a, auto& b) { return a.first > b.first || (a.first == b.first && a.second < b.second); });
    truncate_dist(c, M, Z, invT, samp_top_k, samp_top_p, samp_min_p, tok, p);
}
void Engine::full_dist(const float* lg, std::vector<int>& tok, std::vector<double>& p) {
    const float invT = 1.f / samp_T;
    double M = -1e300;
    for (int i = 0; i < VOCAB; i++) M = std::max(M, (double)lg[i]);
    double Z = 0;
    for (int i = 0; i < VOCAB; i++) Z += std::exp(((double)lg[i] - M) * invT);
    std::vector<std::pair<float, int>> c(VOCAB);
    for (int i = 0; i < VOCAB; i++) c[i] = {lg[i], i};
    const int k = std::min(VOCAB, std::max(TKC, samp_top_k));
    std::partial_sort(c.begin(), c.begin() + k, c.end(), [](auto& a, auto& b) { return a.first > b.first || (a.first == b.first && a.second < b.second); });
    c.resize(k);
    truncate_dist(c, M, Z, invT, samp_top_k, samp_top_p, samp_min_p, tok, p);
}
int Engine::sample_from(const std::vector<int>& tok, const std::vector<double>& p, int exclude) {
    double s = 0;
    for (size_t i = 0; i < tok.size(); i++) if (tok[i] != exclude) s += p[i];
    if (s <= 0) return exclude >= 0 ? exclude : (tok.empty() ? 0 : tok[0]);
    std::uniform_real_distribution<double> U(0.0, s);
    double u = U(srng);
    for (size_t i = 0; i < tok.size(); i++) {
        if (tok[i] == exclude) continue;
        if ((u -= p[i]) <= 0) return tok[i];
    }
    for (size_t i = tok.size(); i-- > 0;) if (tok[i] != exclude) return tok[i];
    return exclude;
}

// ---------------------------------------------------------------- prefill buffers lent from the slab tail

void Engine::coord_idle() {
    if (!coord.joinable()) return;
    while (coord_done.load(std::memory_order_acquire) != (unsigned)steps_issued) _mm_pause();
}
int Engine::lend_slots(int rows) {
    int n = 0;
    for (auto& w : W) n = std::max(n, (int)((w->pf_bytes(rows) + w->sl.bytes - 1) / w->sl.bytes));
    return n;
}
void Engine::upload_tables() {
    std::vector<int> tab(slot_of);
    if (mtp) for (int e = 0; e < NE; e++) tab[(size_t)MTP_L * NE + e] = e;
    run_all([&](Worker& w) {
        CUDA_CHECK(cudaMemcpyAsync(w.slots_all, tab.data(), tab.size() * 4, cudaMemcpyHostToDevice, w.cst));
        CUDA_CHECK(cudaStreamSynchronize(w.cst));
    });
}
void Engine::lend(int rows) {
    CHECK(!lent, "lend: already lent");
    coord_idle();
    const int ns = (int)slot_owner.size();
    const int n = lend_slots(rows);
    CHECK(n <= ns, "lend %d of %d slots", n, ns);
    slot_pre = slot_of;  // residency before the lend (prefill skip rule)
    // round 18: a lend from the seeded state (GLM_DET) re-seeds the seed's own owners, so the cache after any number of
    // prefill windows / forwards is the seed again (a split prefill and a resumed one decode from the same state)
    lend_canon = det && !det_dirty && !det_pending && canon_now();
    const int hi = lend_top();  // round 26: the slots below the MLA region (GLM_MLA_POS=0: the slab's end)
    CHECK(n <= hi, "lend %d of %d slots below the MLA cache", n, hi);
    for (int s = hi - n; s < hi; s++) {
        const int k = slot_owner[s];
        if (k >= 0) slot_of[k] = -1;
        slot_owner[s] = -1;
    }
    free_slots.erase(std::remove_if(free_slots.begin(), free_slots.end(), [&](int s) { return s >= hi - n && s < hi; }),
                     free_slots.end());
    upload_tables();
    lent = n;
    const int s0 = hi - n;
    run_all([&](Worker& w) { w.use_pf(w.slab + (size_t)s0 * w.sl.bytes, rows); });
    for (int g = 0; g < ngpu; g++) {  // round 20: owner mode writes into the peers' window rows (P2P)
        Worker& w = *W[g];
        own_ptr[g] = OwnPtr{(char*)w.xb, (char*)w.sum, (char*)w.d_ids, (char*)w.d_w};
        own_flag[g] = w.agf;
    }
}
void Engine::unlend() {
    if (!lent) return;
    run_all([&](Worker& w) {
        CUDA_CHECK(cudaStreamSynchronize(w.st));
        w.use_dec();
    });
    const int ns = (int)slot_owner.size(), hi = lend_top(), s0 = hi - lent;
    if (lend_canon) {  // round 18: back to the seed exactly (the prefill admits nothing outside the lent tail)
        bool same = true;
        for (int q = 0; q < ns && same; q++)
            if (q < s0 || q >= hi) same = slot_owner[q] == (slot_mla[q] ? -1 : canon_owner[q]);
        if (same) {
            std::vector<std::pair<int, int>> sk;
            for (int q = s0; q < hi; q++) {
                slot_owner[q] = canon_owner[q];
                if (canon_owner[q] >= 0) {
                    slot_of[canon_owner[q]] = q;
                    sk.push_back({q, canon_owner[q]});
                }
            }
            free_slots = canon_free;
            if (mla_pos)
                free_slots.erase(std::remove_if(free_slots.begin(), free_slots.end(), [&](int s) { return slot_mla[s] != 0; }),
                                 free_slots.end());
            reseed(sk);
            lent = 0;
            slot_pre.clear();
            lend_canon = false;
            return;  // det_dirty stays false: nothing moved since the seed
        }
    }
    lend_canon = false;
    // re-seed: the highest decayed counts among the non-resident experts
    std::vector<std::pair<float, int>> v;
    for (int l = 3; l < nl; l++)
        for (int e = 0; e < NE; e++) {
            const int key = l * NE + e;
            if (slot_of[key] < 0) v.push_back({score[key], key});
        }
    const int nseed = std::min<int>(lent, (int)v.size());
    std::partial_sort(v.begin(), v.begin() + nseed, v.end(), [](auto& a, auto& b) { return a.first > b.first; });
    std::vector<std::pair<int, int>> sk(nseed);
    for (int i = 0; i < nseed; i++) {
        slot_owner[s0 + i] = v[i].second;
        slot_of[v[i].second] = s0 + i;
        sk[i] = {s0 + i, v[i].second};
    }
    for (int s = s0 + nseed; s < hi; s++) free_slots.push_back(s);
    reseed(sk);
    lent = 0;
    slot_pre.clear();
    det_dirty = true;  // round 17: the re-seed may place other experts than the lend evicted
}
// round 27 ([glm-r27-reseed]): the re-seed after a prefill runs in the background, as det_restore's copies: the copies
// and the slot-table upload go on the copy streams, and the next verify / forward / page change joins them in det_wait
// (same final state; the prefill's TTFT no longer holds 0.03-0.13 s of copies). GLM_RESEED_ASYNC=0: synchronous.
void Engine::reseed(const std::vector<std::pair<int, int>>& sk) {
    static const bool async = env_int("GLM_RESEED_ASYNC", 1) != 0;
    if (!async) { fill_slots(sk); upload_tables(); return; }
    det_wait();  // review r27 L1: det_tab is not rewritten under an earlier upload from it
    fill_slots(sk, false);
    const size_t nt = slot_of.size();
    if (!det_tab) CUDA_CHECK(cudaHostAlloc(&det_tab, nt * 4, cudaHostAllocPortable));
    memcpy(det_tab, slot_of.data(), nt * 4);
    if (mtp) for (int e = 0; e < NE; e++) det_tab[(size_t)MTP_L * NE + e] = e;
    run_all([&](Worker& w) { CUDA_CHECK(cudaMemcpyAsync(w.slots_all, det_tab, nt * 4, cudaMemcpyHostToDevice, w.cst)); });
    det_pending = true;
}
void Engine::fill_slots(const std::vector<std::pair<int, int>>& sk, bool sync) {
    if (sk.empty()) return;
    run_all([&](Worker& w) {
        for (size_t i = 0; i < sk.size(); i++) {
            const int l = sk[i].second / NE, e = sk[i].second % NE;
            const char* src = hx_layer[l] + (size_t)e * hxl.stride + hxl.off[w.p.rank];
            const SliceL& hs = hxl.lay(hx_layer[l] + (size_t)e * hxl.stride, w.p.rank);
            const int j = (int)(i % Worker::NLAND);
            CUDA_CHECK(cudaStreamWaitEvent(w.dmst, w.ev_lfree[j], 0));
            CUDA_CHECK(cudaMemcpyAsync(w.land[j], src, hs.bytes, cudaMemcpyHostToDevice, w.dmst));
            CUDA_CHECK(cudaEventRecord(w.ev_land[j], w.dmst));
            CUDA_CHECK(cudaStreamWaitEvent(w.cst, w.ev_land[j], 0));
            untile(w.land[j], w.slab + (size_t)sk[i].first * w.sl.bytes, w.sl, 64, w.cst, &hs);
            CUDA_CHECK(cudaEventRecord(w.ev_lfree[j], w.cst));
        }
        if (sync) CUDA_CHECK(cudaStreamSynchronize(w.cst));
    });
}
// round 26 ([glm-p3-mlapos]) ------------------------------------------------ MLA cache pages in the slab
int Engine::mla_page_rows(int rows) const {
    const long r = ((long)std::max(0, rows) + 64 + mla_pg - 1) / mla_pg * mla_pg;  // + 64: verify rows and MTP passes
    return (int)std::min<long>(r, ctx);
}
std::vector<char> Engine::mla_mask(int rows) const {
    std::vector<char> m(slot_owner.size(), 0);
    if (!mla_pos || rows <= 0) return m;
    const size_t kb = (size_t)rows * mla_nl * KVLR * 2, pb = (size_t)(rows / POOL + 1) * mla_nl * ID * 4;
    for (auto& w : W) {
        const size_t sb = w->sl.bytes, base = w->mla_off;
        auto mark = [&](size_t a, size_t b) {  // bytes [a, b) of the slab
            for (size_t s = a / sb; s * sb < b && s < m.size(); s++) m[s] = 1;
        };
        mark(base, base + kb);
        mark(base + mla_kcb, base + mla_kcb + pb);
    }
    return m;
}
bool Engine::canon_now() const {
    if (canon_owner.size() != slot_owner.size()) return false;
    for (size_t s = 0; s < slot_owner.size(); s++)
        if (slot_owner[s] != (slot_mla[s] ? -1 : canon_owner[s])) return false;
    return true;
}
// Take or release pages so that exactly mla_page_rows(rows) rows are held. Taking evicts the experts under the new
// rows (they become cold); releasing gives the slots back: from the seeded state (GLM_DET, nothing moved since) their
// seed owners are copied in, so the cache is again "seed minus the held pages", else they join the free list. The
// caller guarantees that the released rows are not live (ck_prefill / reset: rows past pos are dead by then).
void Engine::mla_set(int rows) {
    if (!mla_pos || slot_owner.empty()) return;
    const int want = mla_page_rows(rows);
    if (want == mla_rows) return;
    CHECK(!lent, "MLA pages change while the prefill set is lent");
    const std::vector<char> nm = mla_mask(want);
    if (nm == slot_mla) { mla_rows = want; return; }  // review r26 L2: the new rows lie under slots already held
    bg_flush();  // as det_restore: queued boundary copies issued and landed before slots change hands
    coord_idle();
    det_wait();
    run_all([&](Worker& w) {
        CUDA_CHECK(cudaStreamSynchronize(w.st));
        if (w.bst) CUDA_CHECK(cudaStreamSynchronize(w.bst));
    });
    std::fill(bg_pend.begin(), bg_pend.end(), 0);
    bg_pl.clear();
    bg_qprev.clear();
    const bool canon = det && !det_dirty && !canon_owner.empty() && canon_now();
    std::vector<std::pair<int, int>> sk;
    const int ns = (int)slot_owner.size();
    for (int s = 0; s < ns; s++) {
        if (nm[s] && !slot_mla[s]) {  // take
            const int k = slot_owner[s];
            if (k >= 0) slot_of[k] = -1;
            slot_owner[s] = -1;
            st_mla_take++;
        } else if (!nm[s] && slot_mla[s]) {  // release
            st_mla_rel++;
            if (canon && canon_owner[s] >= 0) {
                slot_owner[s] = canon_owner[s];
                slot_of[canon_owner[s]] = s;
                sk.push_back({s, canon_owner[s]});
            } else if (!canon) free_slots.push_back(s);
        }
    }
    slot_mla = nm;
    mla_rows = want;
    if (canon) {
        free_slots = canon_free;
        free_slots.erase(std::remove_if(free_slots.begin(), free_slots.end(), [&](int s) { return slot_mla[s] != 0; }),
                         free_slots.end());
    } else free_slots.erase(std::remove_if(free_slots.begin(), free_slots.end(), [&](int s) { return slot_mla[s] != 0; }),
                            free_slots.end());
    fill_slots(sk);
    upload_tables();
}

// round 17 (GLM_DET): back to the cache state right after the seed. Only the slots whose expert changed are copied;
// the decayed counts, the free list and the victim sampler's rng return too, so every sequence starts from exactly
// the state a fresh engine has. Run in the background when a sequence ends (seq_end: server release, generate end;
// the copies run on the copy streams and the next forward / verify joins them in det_wait), else at the next start
// (reset, or spec_start on prefix reuse) when the state is still dirty. The coordinator is idle and nothing is lent.
void Engine::det_restore(bool async) {
    if (!det || canon_owner.empty() || lent || !det_dirty) return;
    bg_flush();  // review r17 L2: a queued boundary batch is issued (and synchronised below) before the slots are rewritten
    coord_idle();
    det_wait();
    run_all([&](Worker& w) {  // background / boundary admission copies into the slab have landed
        CUDA_CHECK(cudaStreamSynchronize(w.st));
        if (w.bst) CUDA_CHECK(cudaStreamSynchronize(w.bst));
    });
    const auto t0 = std::chrono::steady_clock::now();
    const int ns = (int)slot_owner.size();
    std::vector<std::pair<int, int>> sk;
    // round 26: the slots under the held MLA pages stay out (their rows are live); mla_set seeds them when released
    for (int s = 0; s < ns; s++)
        if (canon_owner[s] >= 0 && !slot_mla[s] && slot_owner[s] != canon_owner[s]) sk.push_back({s, canon_owner[s]});
    std::fill(slot_of.begin(), slot_of.end(), -1);
    slot_owner = canon_owner;
    for (int s = 0; s < ns; s++)
        if (slot_mla[s]) slot_owner[s] = -1;
        else if (slot_owner[s] >= 0) slot_of[slot_owner[s]] = s;
    free_slots = canon_free;
    free_slots.erase(std::remove_if(free_slots.begin(), free_slots.end(), [&](int s) { return slot_mla[s] != 0; }),
                     free_slots.end());
    score = canon_score;
    rng = canon_rng;
    std::fill(bg_pend.begin(), bg_pend.end(), 0);
    bg_pl.clear();
    bg_qprev.clear();
    fill_slots(sk, false);
    // slot tables after the copies, in the copy stream's order (pinned source, so the upload is asynchronous too)
    const size_t nt = slot_of.size();
    if (!det_tab) CUDA_CHECK(cudaHostAlloc(&det_tab, nt * 4, cudaHostAllocPortable));
    memcpy(det_tab, slot_of.data(), nt * 4);
    if (mtp) for (int e = 0; e < NE; e++) det_tab[(size_t)MTP_L * NE + e] = e;
    run_all([&](Worker& w) { CUDA_CHECK(cudaMemcpyAsync(w.slots_all, det_tab, nt * 4, cudaMemcpyHostToDevice, w.cst)); });
    det_pending = true;
    det_dirty = false;
    if (!async) det_wait();
    st_det_n++;
    st_det_slots += sk.size();
    st_det_us += std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
}
void Engine::det_wait() {
    if (!det_pending) return;
    run_all([&](Worker& w) { CUDA_CHECK(cudaStreamSynchronize(w.cst)); });
    det_pending = false;
}
void Engine::seq_end() { det_restore(true); }

// ---------------------------------------------------------------- host coordinator

void Engine::coord_start() {
    if (coord.joinable()) return;
    if (!req) {
        CUDA_CHECK(cudaHostAlloc(&req, sizeof(MoeReq) * 2 * (NL + 1), cudaHostAllocPortable | cudaHostAllocMapped));
        CUDA_CHECK(cudaHostAlloc(&dec, sizeof(MoeDec) * 2 * (NL + 1), cudaHostAllocPortable | cudaHostAllocMapped));
        CUDA_CHECK(cudaHostAlloc(&dec_ah, sizeof(MoeDec) * (NL + 1), cudaHostAllocPortable | cudaHostAllocMapped));
        memset((void*)dec_ah, 0, sizeof(MoeDec) * (NL + 1));
        CUDA_CHECK(cudaHostAlloc(&out, sizeof(MoeOut) * 2 * (NL + 1), cudaHostAllocPortable | cudaHostAllocMapped));
        memset((void*)req, 0, sizeof(MoeReq) * 2 * (NL + 1));
        memset((void*)dec, 0, sizeof(MoeDec) * 2 * (NL + 1));
        memset((void*)out, 0, sizeof(MoeOut) * 2 * (NL + 1));
        CUDA_CHECK(cudaHostAlloc(&bgq, sizeof(BgQ), cudaHostAllocPortable | cudaHostAllocMapped));
        CUDA_CHECK(cudaHostAlloc(&bg_flag, 64, cudaHostAllocPortable | cudaHostAllocMapped));
        CUDA_CHECK(cudaHostAlloc(&d_hxl, sizeof(char*) * (NL + 1), cudaHostAllocPortable | cudaHostAllocMapped));
        memset((void*)bgq, 0, sizeof(BgQ));
        memset((void*)bg_flag, 0, 64);
        for (int l = 0; l <= NL; l++) d_hxl[l] = l < (int)hx_layer.size() ? hx_layer[l] : nullptr;
        CUDA_CHECK(cudaHostAlloc(&h_tok, 256, cudaHostAllocPortable));
        CUDA_CHECK(cudaHostAlloc(&h_am, 256, cudaHostAllocPortable));
        CUDA_CHECK(cudaHostAlloc(&h_acc, 64, cudaHostAllocPortable));
        CUDA_CHECK(cudaHostAlloc(&h_mtok, 64, cudaHostAllocPortable));
        CUDA_CHECK(cudaHostAlloc(&h_mpos, 64, cudaHostAllocPortable));
        CUDA_CHECK(cudaHostAlloc(&h_mam, 256, cudaHostAllocPortable));
        CUDA_CHECK(cudaHostAlloc(&h_mse, 256, cudaHostAllocPortable));
        CUDA_CHECK(cudaHostAlloc(&h_chres, (size_t)2 * ngpu * 64 * 4, cudaHostAllocPortable));
        memset(h_chres, 0, (size_t)2 * ngpu * 64 * 4);
        if (env_int("GLM_STAMP", 0) == 2 || env_int("GLM_STAMP", 0) == 4 || env_int("GLM_STAMP", 0) == 5) {
            CUDA_CHECK(cudaHostAlloc(&stamps2, (size_t)4 * (NL + 1) * 16 * 8, cudaHostAllocPortable | cudaHostAllocMapped));
            memset(stamps2, 0, (size_t)4 * (NL + 1) * 16 * 8);
        }
        stamp_lvl = env_int("GLM_STAMP", 0);
        if (stamp_lvl == 1 || stamp_lvl == 3) {
            CUDA_CHECK(cudaHostAlloc(&stamps, (NL + 1) * 16 * 8, cudaHostAllocPortable | cudaHostAllocMapped));
            memset(stamps, 0, (NL + 1) * 16 * 8);
            hts = new long long[(NL + 1) * 8]();
            if (const char* sd = getenv("GLM_STAMP_DUMP")) stamp_dump = fopen(sd, "wb");
        }
        CUDA_CHECK(cudaHostAlloc(&h_tk, sizeof(TopK) * 4 * MR, cudaHostAllocPortable | cudaHostAllocMapped));
        CUDA_CHECK(cudaHostAlloc(&h_invT, 64, cudaHostAllocPortable | cudaHostAllocMapped));
        *h_invT = samp_T > 0 ? 1.f / samp_T : 1.f;
    }
    if (!lane) {
        lane = new CpuLane();
        if (getenv("GLM_CPU_THREADS") || getenv("GLM_CPU_FIRST") || getenv("GLM_LANE_CPUS"))
            lane->start(env_int("GLM_CPU_THREADS", 22), env_int("GLM_CPU_FIRST", 1), &hxl);
        else {  // [glm-rel-host] the cores between the coordinator's and the pool's (1-22 on the 24-core box)
            lane->cpus = hcpu.lane;
            lane->start((int)hcpu.lane.size(), hcpu.lane.front(), &hxl);
        }
        if (pf_lane_rows > 0 && pf_lane_maxt > 0 && !W[0]->lane_alloc()) {  // review r21 M1: degrade, never abort
            fprintf(stderr, "[glm] warning: prefill CPU lane buffers could not be allocated; prefill lane off (GPU path)\n");
            pf_lane_rows = 0;
        }
    }
    decay = (float)env_double("GLM_LFU_DECAY", 0.99);
    use_graph = env_int("GLM_GRAPH", 1) != 0;
    use_ll = env_int("GLM_LL", 0) != 0;
    admit_margin = (float)env_double("GLM_ADMIT_MARGIN", 0.5);
    pcie_share = 0.f;  // zero-copy GPU jobs would need the row-major layout (host copy is tile-major)
    prefetch = env_int("GLM_PREFETCH", 1) != 0;
    lookup = env_int("GLM_LOOKUP", 0) != 0;  // 3 trigram drafts: slower on the standard set (4-row unions)
    w8a8 = env_int("GLM_W8A8", 1) != 0;
    skip_tau = (float)env_double("GLM_SKIP_TAU", 0.10);  // round 13 knee under the Maya-M bar (lossy; was 0.06 since round 8); 0 = off
    if (skip_tau > 0) fprintf(stderr, "[glm] lossy skip (default, GLM_SKIP_TAU=0 off): cold experts with gate weight < %.3f are skipped\n", skip_tau);
    // round 17 (coordinator, lossy budget): the prefill is lossless by default. The tau 0.10 skip in a long prompt's
    // prefill changed answers (32K prompt: first token '<tool_call>' instead of 'OK' at ctx 163840; last-row KL vs
    // tau 0 0.92 / 1.89 at ctx 40960 / 163840); it bought only 1.3 / 3.6 / 5.7 % prefill speed at 8K / 32K / 110K.
    // GLM_PF_TAU=x sets a prefill threshold (lossy); decode keeps GLM_SKIP_TAU.
    pf_tau = (float)env_double("GLM_PF_TAU", 0.0);
    if (pf_tau > 0) fprintf(stderr, "[glm] lossy prefill skip (GLM_PF_TAU): cold experts with gate weight < %.3f are skipped\n", pf_tau);
    chain_raw = env_int("GLM_CHAIN_RAW", 0) != 0;
    // round 19 [glm-mtp-dev]: the draft chain on the device (needs graphs, the P2P all-reduce and one sequence)
    // (review r19 L5: the chain's exchanges use the LL protocol, so GLM_AR_LL=0 runs the per-pass host path)
    dev_chain = env_int("GLM_MTP_DEV", 1) != 0 && use_graph && p2p && ngpu > 1 && env_int("GLM_AR_P2P", 1) != 0 && nseq == 1 &&
                env_int("GLM_AR_LL", 1) != 0;
    chain_chk = dev_chain && env_int("GLM_MTP_DEV_CHK", 0) != 0;
    ar_ll = env_int("GLM_AR_LL", 1) != 0;
    ar_ll12 = env_int("GLM_AR_LL12", 1) != 0;
    pf_max = env_int("GLM_PF_MAX", 8);
    pf_order = env_int("GLM_PF_ORDER", 0);
    pf_early = env_int("GLM_PF_EARLY", 1);  // round 13: default at tau 0.10 (+1.5 %); GLM_PF_EARLY=0 off. Review r13b L5: in the opt-in wave / pair
    // modes (GLM_WAVE=1, GLM_SEQS=2) group 0 raises the next layer's flag before group 1's lane ends, so the prefetch is practically off there
    pf_late = env_int("GLM_PF_LATE", 0);
    if (env_int("GLM_PFSTAT", 0) && lane) {  // round 9 timeline: TSC rate
        const auto a = std::chrono::steady_clock::now();
        const uint64_t t = __rdtsc();
        while (std::chrono::steady_clock::now() - a < std::chrono::milliseconds(20)) {}
        tsc_mhz = (double)(__rdtsc() - t) / std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - a).count();
        tsc0 = __rdtsc(); clk0 = std::chrono::steady_clock::now();
        lane->tl_on = true;
    }
    pf_bg = env_int("GLM_PF_BG", 0) != 0;
    admit_cap = env_int("GLM_ADMIT_CAP", 64);
    admit_lcap = env_int("GLM_ADMIT_LCAP", MXE);  // admissions per request (PCIe time per layer)
    // round 13: in-step admissions per request. Only under GLM_ADMIT_RULE=2 (default) does a background branch take the
    // rest (total still bounded by admit_lcap, so ICAP > LCAP has no effect); under rule 0 / 1 it is a second total cap
    admit_icap = std::max(0, env_int("GLM_ADMIT_ICAP", MXE));  // review r13b L2: negative = 0
    ah_stat = env_int("GLM_AHSTAT", 0) != 0;
    admit_ahead = std::max(0, std::min(MXE, env_int("GLM_ADMIT_AHEAD", 0)));  // round 15 gate
    // review r15 L11: ahead copies are bounded only by N and the step budget, not by GLM_ADMIT_LCAP / ICAP nor the
    // admission cost model; plan(l + 1) waits for them, so copies that miss layer l's window land on the critical path
    ah_blocks = std::max(1, env_int("GLM_AH_BLOCKS", 12));
    ah_force = env_int("GLM_AH_FORCE", 0) != 0;  // round 15 gate counters (admissions one layer ahead)
    admit_effi = env_int("GLM_ADMIT_EFFI", 1);  // round 14 default (review r13b M1): skip-only cold experts get no in-step place (GLM_ADMIT_EFFI=0 off)
    admit_eff = env_int("GLM_ADMIT_EFF", 1);  // round 13 default: the lane-cost model counts only cold experts with a row above tau (GLM_ADMIT_EFF=0 off)
    admit_par = env_int("GLM_ADMIT_PAR", 1) != 0;
    wait_old = env_int("GLM_WAIT_OLD", 0) != 0;
    admit_rule = env_int("GLM_ADMIT_RULE", 2);
    bg_blocks = env_int("GLM_ADMIT_BG_BLOCKS", 2);
    adm_tp = (float)env_double("GLM_ADMIT_TP_US", 190.0);
    adm_tg = (float)env_double("GLM_ADMIT_TG_US", 18.0);
    adm_tl = (float)env_double("GLM_ADMIT_TL_US", 170.0);  // round 16: 95 -> 170 (measured lane ~115 us/expert + tail; alternated fixed-text pair -0.5..-2.0 %)
    adm_hot = (float)env_double("GLM_ADMIT_HOT_US", 150.0);
    adm_hotn = (float)env_double("GLM_ADMIT_HOTN_US", 0.0);  // round 14 (L1): hot-path us per unique hot expert
    adm_tgr = (float)env_double("GLM_ADMIT_TGR_US", 0.0);    // round 14 (L1): admitted GEMV us per extra row
    wave = env_int("GLM_WAVE", 0) != 0;
    side = env_int("GLM_SIDE", 1) != 0;
    ar_early = env_int("GLM_AR_EARLY", 0) != 0;  // measured: host wait 9.7 vs 9.7 ms, no gain; changes the sum order
    lane_defer = env_int("GLM_LANE_DEFER", 0);  // round 15 gate (lossy); 2 = exact check of the plumbing
    if (lane_defer < 0 || lane_defer > 2) {  // review r15 L9 / r16 L3: out of range = off
        fprintf(stderr, "[glm] GLM_LANE_DEFER=%d: only 0 / 1 / 2, deferral off\n", lane_defer);
        lane_defer = 0;
    }
    if (lane_defer && ar_early) {  // review r16 L1: the GLM_AR_EARLY body adds only the lane's first output
        fprintf(stderr, "[glm] GLM_LANE_DEFER=%d is not supported with GLM_AR_EARLY=1: deferral off\n", lane_defer);
        lane_defer = 0;
    }
    defer_w = (float)env_double("GLM_DEFER_W", 0.0);
    defer_min = env_int("GLM_DEFER_MIN", 0);
    if (!lane_defer && (defer_w > 0 || defer_min > 0))
        fprintf(stderr, "[glm] GLM_DEFER_W / GLM_DEFER_MIN are ignored without GLM_LANE_DEFER\n");
    def_split = lane_defer && (defer_w > 0 || defer_min > 0);
    if (def_split && !out2) {  // second lane pass output (host -> all ranks), only with the split deferral
        CUDA_CHECK(cudaHostAlloc(&out2, sizeof(MoeOut) * 2 * (NL + 1), cudaHostAllocPortable | cudaHostAllocMapped));
        memset((void*)out2, 0, sizeof(MoeOut) * 2 * (NL + 1));
    }
    if (lane_defer && (wave || nseq > 1))  // review r16 L3: only the plain single-sequence body defers
        fprintf(stderr, "[glm] GLM_LANE_DEFER=%d: wave steps (GLM_WAVE, m >= 2) and pair steps (GLM_SEQS=2) run without "
                "deferral; m = 1 / single-sequence steps still defer\n", lane_defer);
    mla_split = env_int("GLM_MLA_SPLIT", 1) != 0;
    mla_tc = env_int("GLM_MLA_DTC", 24);
    mla_ubf = env_int("GLM_MLA_UBF", 1) != 0;
    idx_skip = env_int("GLM_IDX_SKIP", 1) != 0;  // round 27 (D4 b)
    l2pf_ko = (size_t)(env_double("GLM_L2PF_KO", 0) * 1048576);
    l2pf_blocks = env_int("GLM_L2PF_BLK", 16);
    fuse_mix = env_int("GLM_FUSE_MIX", 0) != 0;
    fuse_bnd = env_int("GLM_FUSE_BND", 1) != 0;
    fuse_kda = env_int("GLM_FUSE_KDA", 1);  // opt-in: ~1.5 us per layer, within noise
    conf_tau = (float)env_double("GLM_CONF_TAU", 0.7);  // 0: fixed k
    // round 11: up to 16 background admission copies per step deferred to the step boundary (DMA + untile next to the
    // lm_head / MTP), joined at the head of the next verify graph (GLM_BG_BOUND=0: in-step copies as in round 8)
    // r12 (review C1): the pair path (GLM_SEQS=2) has no boundary join, so the deferral is single-sequence only
    bg_bound = admit_rule == 2 && nseq == 1 ? std::max(0, std::min(BgList::CAP, env_int("GLM_BG_BOUND", 16))) : 0;
    bg_bound_blk = env_int("GLM_BG_BOUND_BLK", 24);
    bg_dma = env_int("GLM_BG_BOUND_DMA", 1) != 0;
    // round 17: per-layer waits for the boundary copies (DMA path, plain single-sequence body, wait point layer 3)
    bg_perl = bg_bound && bg_dma && !wave && env_int("GLM_BG_WAITL", 3) == 3 && env_int("GLM_BG_PERL", 1) != 0;  // round 17 default (Maya-like +3.7 %); 0 = the layer-3 wait
    if (!bg_need) {
        CUDA_CHECK(cudaHostAlloc(&bg_need, (NL + 1) * 4, cudaHostAllocPortable));
        memset(bg_need, 0, (NL + 1) * 4);
    }
    bg_sync = env_int("GLM_BG_SYNC", 0) != 0;
    if (const char* dl = getenv("GLM_DELAY")) sscanf(dl, "%d,%d,%d,%d", &dbg_delay[0], &dbg_delay[1], &dbg_delay[2], &dbg_delay[3]);
    dbg_slotchk = env_int("GLM_SLOTCHK", 0) != 0;
    if (const char* hg = getenv("GLM_HOG")) sscanf(hg, "%d,%d,%d", &dbg_hog[0], &dbg_hog[1], &dbg_hog[2]);
    // where the next verify waits for the boundary copies: 3 = before layer 3's plan (default; layers 0-2 overlap the
    // copies), -1 = graph head; -2 / -3 debug splits. Exact since the queue goes to the kernels by value (r11 race fix)
    // r12 (review C2): only 3 (= the first MoE layer, from the layer table) and -1 are exact; -2 / -3 are the bisect
    // splits and need GLM_BG_DEBUG=1; anything else would drop the wait or place it after plans that depend on it
    {
        int fm = 0;
        while (fm < nl && is_dense(fm)) fm++;
        const int wl = env_int("GLM_BG_WAITL", 3);
        const bool dbg = env_int("GLM_BG_DEBUG", 0) != 0;
        CHECK(wl == 3 || wl == -1 || (dbg && (wl == -2 || wl == -3)), "GLM_BG_WAITL=%d: use 3 (first MoE layer) or -1 (graph head)", wl);
        first_moe = wl == 3 ? fm : wl;
    }
    pfstat = env_int("GLM_PFSTAT", 0) != 0;
    kda_ptr = env_int("GLM_KDA_PTR", 1) != 0 && nseq == 1;
    fuse_commit = env_int("GLM_FUSE_COMMIT", 1) != 0 && nseq == 1;
    if (const char* sd = getenv("GLM_SKIPDUMP")) skip_dump = fopen(sd, "wb");  // round 17 gate
    if (const char* cl = getenv("GLM_CONF_LOG")) conf_log = fopen(cl, "a");  // opt-in: GPU-bound today (two passes ~25 ms GPU-only vs lane ~21 ms)
    wave_a = std::max(1, env_int("GLM_WAVE_A", 1));
    coord_stop = false;
    coord = std::thread([this] {
        cpu_set_t cs;
        CPU_ZERO(&cs);
        CPU_SET(env_int("GLM_COORD_CPU", hcpu.coord), &cs);
        pthread_setaffinity_np(pthread_self(), sizeof cs, &cs);
        coord_loop();
    });
}

void Engine::policy_init() {
    score.assign((size_t)(NL + 1) * NE, 0.f);
    const char* pf = getenv("GLM_PRIOR");
    std::string path = pf ? pf : glm_data_file("expert_prior_gen.f32");
    FILE* f = fopen(path.c_str(), "rb");
    std::vector<float> prior((size_t)NL * NE, 0.f);
    if (f) {
        const size_t n = fread(prior.data(), 4, prior.size(), f);
        fclose(f);
        fprintf(stderr, "[glm] expert prior %s (%zu)\n", path.c_str(), n);
    } else fprintf(stderr, "[glm] no expert prior (%s): cache starts empty\n", path.c_str());
    // seed: the most frequent experts of the prior fill the slots
    std::vector<std::pair<float, int>> v;
    for (int l = 3; l < nl; l++)
        for (int e = 0; e < NE; e++) v.push_back({prior[(size_t)l * NE + e], l * NE + e});
    std::sort(v.begin(), v.end(), [](auto& a, auto& b) { return a.first > b.first; });
    const int ns = (int)slot_owner.size();
    std::vector<int> use;  // round 26: the slots not under the held MLA pages, in order
    for (int s = 0; s < ns; s++) if (!slot_mla[s]) use.push_back(s);
    const int nseed = std::min<int>((int)use.size(), f ? (int)v.size() : 0);
    double tot = 0;
    for (auto& x : v) tot += x.first;
    for (int i = 0; i < nseed; i++) {
        const int key = v[i].second, s = use[i];
        slot_owner[s] = key;
        slot_of[key] = s;
        // steady-state decayed count of an expert picked with probability q per token: q / (1 - decay)
        score[key] = (float)(v[i].first / std::max(1e-9, tot) * TOPK * (nl - 3) / (1.0 - decay));
    }
    for (int i = nseed; i < (int)use.size(); i++) free_slots.push_back(use[i]);
    run_all([&](Worker& w) {
        std::vector<int> tab((size_t)(NL + 1) * NE, -1);
        if (mtp) for (int e = 0; e < NE; e++) tab[(size_t)MTP_L * NE + e] = e;  // MTP experts: resident slab
        for (int i = 0; i < nseed; i++) {
            const int s = use[i];
            tab[slot_owner[s]] = s;
            const int l = slot_owner[s] / NE, e = slot_owner[s] % NE;
            const char* src = hx_layer[l] + (size_t)e * hxl.stride + hxl.off[w.p.rank];
            const SliceL& hs = hxl.lay(hx_layer[l] + (size_t)e * hxl.stride, w.p.rank);
            char* dst = w.slab + (size_t)s * w.sl.bytes;
            CUDA_CHECK(cudaMemcpyAsync(w.stage[0], src, hs.bytes, cudaMemcpyHostToDevice, w.cst));  // DMA, then untile
            untile(w.stage[0], dst, w.sl, 64, w.cst, &hs);
        }
        CUDA_CHECK(cudaMemcpyAsync(w.slots_all, tab.data(), tab.size() * 4, cudaMemcpyHostToDevice, w.cst));
        CUDA_CHECK(cudaStreamSynchronize(w.cst));
    });
    fprintf(stderr, "[glm] cache seeded: %d of %d slots\n", nseed, ns);
    if (env_int("GLM_TAU_ALL", 0) && skip_tau > 0) {  // round 17 gate: the skip rule without residency (KL gate only)
        const float v = skip_tau * RSF;
        run_all([&](Worker&) { set_tau_all(v); });
        fprintf(stderr, "[glm] GLM_TAU_ALL=1: every pick with weight < %.3f is skipped (resident ones too)\n", skip_tau);
    }
    pf_last1 = env_int("GLM_PF_LAST1", 1) != 0;
    mtp_last1 = env_int("GLM_MTP_LAST1", 0) != 0;
    det = env_int("GLM_DET", 1);
    if (det && nseq > 1) {  // the sequences share one cache: each one's residency depends on the other's tokens
        fprintf(stderr, "[glm] GLM_SEQS=%d: the cache is shared, so greedy output is not independent of the other sequence (GLM_DET off)\n", nseq);
        det = 0;
    }
    canon_owner = slot_owner;
    canon_free = free_slots;
    canon_score = score;
    canon_rng = rng;
    if (det && pf_bg) fprintf(stderr, "[glm] GLM_PF_BG=1 makes admissions depend on timing: greedy output is not independent of earlier requests\n");  // review r17 L7
    if (!det) fprintf(stderr, "[glm] GLM_DET=0: the expert cache carries over between sequences (greedy output can depend on earlier requests)\n");
}

int Engine::admit(int key, const int* now, int nnow, int* vkey, bool force) {
    *vkey = -1;
    if (!free_slots.empty()) {
        const int s = free_slots.back();
        free_slots.pop_back();
        return s;
    }
    const int ns = (int)slot_owner.size();
    if (!ns) return -1;
    int best = -1;
    float bs = 1e30f;
    // review r26-5: a draw on a slot without an expert (held by the MLA region at long context, ~20 % of the slots at
    // 262K) is drawn again (at most 128 draws), so the victim is still the best of 32 candidates; without such slots
    // the draws, and so the admissions, are the same as before
    for (int i = 0, t = 0; i < 32 && t < 128; t++) {
        const int s = (int)(rng() % ns);
        const int k = slot_owner[s];
        if (k < 0) continue;
        i++;
        if (bg_pend[s]) continue;  // round 11: a background copy into this slot is still pending (this step)
        bool used = false;
        for (int j = 0; j < nnow; j++) used |= now[j] == k;
        if (used) continue;
        if (score[k] < bs) { bs = score[k]; best = s; }
    }
    if (best < 0 || (!force && score[key] < bs + admit_margin)) return -1;
    *vkey = slot_owner[best];
    return best;
}

void Engine::coord_loop() {
    const char* ex[MXE];
    int enr[MXE], erows[MXE * CpuLane::MR];
    float ew[MXE * CpuLane::MR];
    int padm[MXE], npadm = 0;  // keys admitted by the previous request (their copies / GEMVs may still be running)
    int ahk[MXE], nahk = 0;    // round 15: keys admitted ahead for the next layer
    const char* pfl[MXE];      // last prefetch list (usefulness statistics) and the lane's touched bytes before it
    int npf = 0;
    unsigned long long pb0 = 0;
    unsigned s_done = 0;
    const long coord_spin = idle_spin_us("GLM_COORD_SPIN_US");
    while (!coord_stop) {
        const unsigned s = s_done + 1;
        int budget = admit_cap;
        int ng = 1;
        npadm = 0;
        for (int l = 3; l < nl; l++)
            for (int g = 0; g < ng; g++) {
                const int qi = g * (NL + 1) + l;
                MoeReq* r = req + qi;
                const auto tw = std::chrono::steady_clock::now();
                if (hts && g == 0) hts[l * 8 + 3] = rt_ns();
                if (l == 3 && g == 0) {  // [glm-idle] the step is not issued yet: spin GLM_COORD_SPIN_US, then block
                    unsigned go = coord_go.load(std::memory_order_acquire);
                    while ((int)(go - s) < 0) go = coord_gate.wait(coord_go, go, coord_spin);
                }
                while (*(volatile unsigned*)&r->seq != s) {
                    if (coord_stop) return;
                    _mm_pause();
                }
                const auto tc = std::chrono::steady_clock::now();
                const uint64_t tc_t = __rdtsc();
                if (hts && g == 0) hts[l * 8 + 0] = rt_ns();
                if (l > 3) st_wait_us += std::chrono::duration<double, std::micro>(tc - tw).count();
                std::atomic_thread_fence(std::memory_order_acquire);
                if (l == 3 && g == 0) {
                    ng = wave_cur.load(std::memory_order_acquire) ? 2 : 1;

                }
                MoeDec* d = dec + qi;
                const int m = r->m;
                int keys[MR * TOPK + MXE];
                for (int i = 0; i < m; i++)
                    for (int k = 0; k < TOPK; k++) {
                        keys[i * TOPK + k] = l * NE + r->ids[i][k];
                        score[keys[i * TOPK + k]] += 1.f;
                        if (r->hot[i][k]) st_hit++;
                        else st_miss++;
                    }
                int nkeys = m * TOPK;
                if (g == 0 && nahk) {  // round 15: ahead admissions of the previous layer that this layer routes
                    for (int j = 0; j < nahk; j++) {
                        bool hit = false;
                        for (int i = 0; i < nkeys; i++) hit |= keys[i] == ahk[j];
                        st_ah_hit += hit;
                    }
                    nahk = 0;
                }
                for (int i = 0; i < npadm; i++) keys[nkeys++] = padm[i];
                if (pf_late && g == 0 && l > 3) {  // round 9: precision of the early / late prediction lists vs the cold set
                    MoeReq* rp = req + l - 1;
                    while (*(volatile unsigned*)&rp->lseq != s && !coord_stop) _mm_pause();
                    std::atomic_thread_fence(std::memory_order_acquire);
                    for (int v = 0; v < 2; v++) {
                        int lst[MR * TOPK], nlst = 0;
                        for (int k = 0; k < TOPK; k++)
                            for (int i = 0; i < rp->m; i++) {
                                const int e = v ? rp->lids[i][k] : rp->pids[i][k];
                                if (slot_of[l * NE + e] >= 0) continue;
                                bool dup = false;
                                for (int j = 0; j < nlst; j++) dup |= lst[j] == e;
                                if (!dup) lst[nlst++] = e;
                            }
                        const int at[3] = {1, 2, 4};
                        for (int a = 0; a < 3; a++) {
                            int hits = 0, n = std::min(at[a], nlst);
                            for (int j = 0; j < n; j++)
                                for (int c = 0; c < r->ncold; c++) hits += r->ce[c] == lst[j];
                            st_pa[v][a] += n ? (double)hits / n : 0.0;
                        }
                    }
                    st_pan += 1;
                }
                int nadm = 0, nex = 0, nzc = 0, nbg = 0;
                int bge[MXE], bgs[MXE], bgv[MXE];
                bool effc[MXE];  // round 14: cold expert c has a row >= tau (the lane would compute it)
                for (int c = 0; c < r->ncold; c++) {
                    bool any = skip_tau <= 0;
                    // review r14 L4: the lane's own predicate (`cw < thr` skips), so a NaN weight counts as kept on both sides
                    for (int i = 0; i < r->cnr[c] && !any; i++) any = !(r->cw[c][i] < skip_tau * RSF);
                    effc[c] = any;
                }
                // round 15 gate (GLM_AHSTAT=1): admissions one layer ahead. P = the previous layer's router(l) prediction,
                // unique experts by predicted weight, not resident now, predicted weight >= tau; compared with the
                // effective cold set and with the in-step admissions this layer makes below
                int ahp[MXE], nahp = 0;
                if (ah_stat && g == 0 && l > 3) {
                    MoeReq* rp = req + l - 1;
                    while (*(volatile unsigned*)&rp->pseq != s && !coord_stop) _mm_pause();
                    std::atomic_thread_fence(std::memory_order_acquire);
                    float pwv[MXE];
                    for (int i = 0; i < rp->m; i++)
                        for (int k = 0; k < TOPK; k++) {
                            const int e = rp->pids[i][k];
                            const float w = rp->pw[i][k];
                            if (slot_of[l * NE + e] >= 0 || (skip_tau > 0 && w < skip_tau * RSF)) continue;  // the lane's predicate (review r15 L10)
                            int j = 0;
                            while (j < nahp && ahp[j] != e) j++;
                            if (j == nahp) { ahp[nahp] = e; pwv[nahp++] = w; }
                            else pwv[j] = std::max(pwv[j], w);
                        }
                    for (int i = 1; i < nahp; i++)  // by predicted weight, largest first
                        for (int j = i; j > 0 && pwv[j] > pwv[j - 1]; j--) { std::swap(pwv[j], pwv[j - 1]); std::swap(ahp[j], ahp[j - 1]); }
                }
                int lcap = std::min(admit_lcap, admit_icap);  // round 13: in-step cap (GLM_ADMIT_ICAP); background keeps admit_lcap
                if (admit_rule) {  // largest a whose PCIe copies + GEMVs end inside the lane of the other nc - a effective experts
                    int nc = r->ncold;
                    // round 13 (GLM_ADMIT_EFF): skipped experts cost the lane nothing. Review r13b L3: only under rule 2,
                    // where skip-only experts still reach the cache through the background branch (rule 1 has none)
                    if (admit_eff && skip_tau > 0 && admit_rule == 2) {
                        nc = 0;
                        for (int c = 0; c < r->ncold; c++) nc += effc[c];
                    }
                    // round 14 (review r13b L1): the hot path grows with the unique hot experts of the m rows
                    // (GLM_ADMIT_HOTN_US per expert) and an admitted expert's GEMV with its rows (GLM_ADMIT_TGR_US per
                    // row past the first); both 0 = the round-8 constants
                    float hot = adm_hot, tg = adm_tg + adm_tgr * (m - 1);
                    if (adm_hotn > 0) {
                        int nu = 0;
                        for (int j = 0; j < m * TOPK; j++) {
                            bool dup = false;
                            for (int q = 0; q < j && !dup; q++) dup = keys[q] == keys[j];
                            nu += !dup;
                        }
                        hot += adm_hotn * std::max(0, nu - r->ncold);
                    }
                    int a = 0;
                    while (a < nc &&
                           std::max(hot, (a + 1) * adm_tp) + (a + 1) * tg <=
                               std::max(hot, (nc - a - 1) * adm_tl))
                        a++;
                    lcap = std::min(lcap, a);
                }
                // PCIe share: the first experts that are not admitted go to the GPUs (zero-copy), the rest to the CPU lane
                int zmax = (int)std::lround(pcie_share * r->ncold);
                // round 9: experts the lane has prefetched for this layer (list order, threads advance together)
                int npfc = 0;
                if (pf_bg && admit_rule == 2 && npf && g == 0)
                    npfc = std::min(npf, (int)((double)(lane->pbytes.load() - pb0) / (double)hxl.stride + 0.5));
                npadm = 0;
                // round 17 gate record (GLM_SKIPDUMP): step, layer, lane experts, partly skipped rows (their expert is
                // read anyway) and fully skipped experts (sum of w^2, rows)
                struct { int s, l, nlane, npart; float epart; int nfull; float fe[16]; int fn[16]; } sk{};
                sk.s = (int)s; sk.l = l;
                for (int c = 0; c < r->ncold; c++) {
                    const int key = l * NE + r->ce[c];
                    int vkey = -1, slot = -1;
                    bool bg = false, inpf = false;
                    if (npfc) {
                        const char* p_ = hx_layer[l] + (size_t)r->ce[c] * hxl.stride;
                        for (int j = 0; j < npfc; j++) inpf |= pfl[j] == p_;
                    }
                    if (inpf) {  // read once from DRAM by the prefetch: the lane computes it from cache, the copy is background
                        if (admit_rule == 2 && budget > 0 && nadm + nbg < admit_lcap) {
                            slot = admit(key, keys, nkeys, &vkey);
                            bg = slot >= 0;
                            if (bg) st_pf_bg++;
                        }
                    // review r14 L1: EFFI only under rule 2 (rule 0 / 1 have no background branch, so skip-only experts
                    // would never reach the cache); L2: the in-step test also keeps the total under admit_lcap
                    } else if (budget > 0 && nadm < lcap && nadm + nbg < admit_lcap &&
                               (!admit_effi || admit_rule != 2 || effc[c]))
                        slot = admit(key, keys, nkeys, &vkey);
                    else if (admit_rule == 2 && budget > 0 && nadm + nbg < admit_lcap) {
                        slot = admit(key, keys, nkeys, &vkey);
                        bg = slot >= 0;
                    }
                    if (bg) {  // copied in the background; the lane computes its rows below
                        budget--;
                        st_admit++;
                        st_bg++;
                        st_bg_so += !effc[c];
                        padm[npadm++] = key;
                        bge[nbg] = c; bgs[nbg] = slot; bgv[nbg] = vkey; nbg++;
                        bg_pend[slot] = 1;
                        bg_pl.push_back(slot);
                        if (vkey >= 0) slot_of[vkey] = -1;
                        slot_of[key] = slot;
                        slot_owner[slot] = key;
                    }
                    if (slot >= 0 && !bg) {
                        budget--;
                        st_admit++;
                        st_ins++;
                        st_ins_so += !effc[c];
                        padm[npadm++] = key;
                        d->e[nadm] = r->ce[c];
                        d->slot[nadm] = slot;
                        d->vl[nadm] = vkey >= 0 ? vkey / NE : -1;
                        d->ve[nadm] = vkey >= 0 ? vkey % NE : -1;
                        d->nr[nadm] = r->cnr[c];
                        for (int i = 0; i < r->cnr[c]; i++) { d->row[nadm][i] = r->crow[c][i]; d->w[nadm][i] = r->cw[c][i]; }
                        nadm++;
                        if (vkey >= 0) slot_of[vkey] = -1;
                        slot_of[key] = slot;
                        slot_owner[slot] = key;
                    } else if (nzc < zmax) {
                        d->ze[nzc] = r->ce[c];
                        d->znr[nzc] = r->cnr[c];
                        for (int i = 0; i < r->cnr[c]; i++) { d->zrow[nzc][i] = r->crow[c][i]; d->zw[nzc][i] = r->cw[c][i]; }
                        nzc++;
                    } else {
                        // lossy skip, default tau 0.10 since round 13 (GLM_SKIP_TAU=0.06: round-8 default, 0: off): drop the rows whose gate weight is below tau
                        int nr = 0;
                        float e2 = 0.f;
                        int ns_ = 0;
                        for (int i = 0; i < r->cnr[c]; i++) {
                            if (r->cw[c][i] < skip_tau * RSF) {  // tau on the normalised weights
                                st_skip++;
                                const float wn = r->cw[c][i] / RSF;
                                e2 += wn * wn; ns_++;
                                continue;
                            }
                            erows[nex * CpuLane::MR + nr] = r->crow[c][i];
                            ew[nex * CpuLane::MR + nr] = r->cw[c][i];
                            nr++;
                        }
                        if (skip_dump && g == 0) {
                            if (nr) { sk.nlane++; sk.npart += ns_; sk.epart += e2; }
                            else if (sk.nfull < 16) { sk.fe[sk.nfull] = e2; sk.fn[sk.nfull] = ns_; sk.nfull++; }
                        }
                        if (nr) {
                            ex[nex] = hx_layer[l] + (size_t)r->ce[c] * hxl.stride;
                            enr[nex] = nr;
                            nex++;
                        }
                    }
                }
                if (skip_dump && g == 0) { fwrite(&sk, sizeof sk, 1, skip_dump); if (l + 1 == nl) fflush(skip_dump); }  // review r17 L11
                // round 11 (GLM_BG_BOUND): background copies deferred to the step boundary while the queue has room;
                // their victims are still released here (k_admit_fin), the copy + slot entry land at the boundary
                int nde = bg_bound ? std::max(0, std::min(nbg, bg_bound - bgq->n)) : 0;
                for (int i = 0; i < nbg; i++) {
                    const int j = nadm + i;
                    d->e[j] = r->ce[bge[i]];
                    d->slot[j] = bgs[i];
                    d->vl[j] = bgv[i] >= 0 ? bgv[i] / NE : -1;
                    d->ve[j] = bgv[i] >= 0 ? bgv[i] % NE : -1;
                    d->nr[j] = 0;
                    if (i >= nbg - nde) {
                        const int q = bgq->n++;
                        bgq->l[q] = l; bgq->e[q] = r->ce[bge[i]]; bgq->slot[q] = bgs[i];
                    }
                }
                if (ah_stat && g == 0 && l > 3) {
                    static const int NS[4] = {1, 2, 4, MXE};
                    int nce = 0;
                    for (int c = 0; c < r->ncold; c++) nce += effc[c];
                    st_ah[4][0] += nadm; st_ah[4][1] += nadm > 0; st_ah[4][2] += nce;
                    for (int v = 0; v < 4; v++) {
                        const int n = std::min(NS[v], nahp);
                        for (int j = 0; j < n; j++) {
                            bool inc = false, ina = false;
                            for (int c = 0; c < r->ncold; c++) inc |= r->ce[c] == ahp[j] && effc[c];
                            for (int i = 0; i < nadm; i++) ina |= d->e[i] == ahp[j];
                            st_ah[v][0] += 1; st_ah[v][1] += inc; st_ah[v][2] += ina;
                        }
                    }
                }
                d->l = l;
                d->nbgd = nde;
                nbg -= nde;
                d->nbg = nbg;
                d->nadm = nadm;
                d->nzc = nzc;
                st_zc += nzc;
                std::atomic_thread_fence(std::memory_order_release);
                *(volatile unsigned*)&d->seq = s;
                if (hts && g == 0) hts[l * 8 + 1] = rt_ns();
                st_cpu += nex;
                for (int i = 0; i < nex; i++) st_cpu_rows += enr[i];
                for (int i = 0; i < nadm; i++) st_adm_rows += d->nr[i];
                const auto t0 = std::chrono::steady_clock::now();
                const uint64_t t0_t = __rdtsc();
                const bool had_pf = npf != 0;
                lane->wait();  // a prefetch of this layer (if any) has stopped
                if (lane->tl_on && g == 0 && l > 3) {
                    const uint64_t nowt = __rdtsc();
                    const double us_dec = std::chrono::duration<double, std::micro>(t0 - tc).count();
                    st_tl[0] += us_dec;  // request -> decisions published
                    st_tl[1] += (double)(nowt - t0_t) / tsc_mhz;  // prefetch stop wait
                    if (had_pf) {
                        bool all_ex = true;
                        uint64_t mx = 0;
                        for (int i = 0; i < lane->nth; i++) { all_ex &= !lane->pf_ab[i]; mx = std::max(mx, lane->pf_end[i]); }
                        if (all_ex) { st_tl[2] += std::max(0.0, ((double)tc_t - (double)mx) / tsc_mhz); st_tl[3] += 1; }  // prefetch ran dry
                        // per-thread spread of the prefetch ends (threads that ran dry before the abort)
                        double idle = 0;
                        for (int i = 0; i < lane->nth; i++)
                            if (!lane->pf_ab[i]) idle += std::max(0.0, ((double)tc_t - (double)lane->pf_end[i]) / tsc_mhz);
                        st_tl[4] += idle / lane->nth;
                        if (!all_ex) st_tl[9] += std::max(0.0, ((double)mx - (double)tc_t) / tsc_mhz);  // last thread's abort latency
                    } else st_tl[5] += std::chrono::duration<double, std::micro>(tc - t_lane_end).count();  // no prefetch at all
                    st_tl[6] += std::chrono::duration<double, std::micro>(tc - t_lane_end).count();  // GPU window
                    st_tl[7] += 1;
                }
                if (hts && g == 0) hts[l * 8 + 4] = rt_ns();
                if (npf && g == 0) {  // prefetch usefulness: experts covered in list order (threads advance together)
                    const double xb = (double)hxl.stride;
                    double cov = (double)(lane->pbytes.load() - pb0) / xb;
                    for (int i = 0; i < npf && cov > 0; i++) {
                        const double f = std::min(1.0, cov);
                        cov -= 1.0;
                        bool used = false;
                        for (int j = 0; j < nex; j++) used |= ex[j] == pfl[i];
                        st_pf_cov += f;
                        if (used) st_pf_use += f;
                        else {  // round 9 audit: why the prefetched bytes were not used by the lane
                            const int e = (int)((pfl[i] - hx_layer[l]) / hxl.stride), key = l * NE + e;
                            bool routed = false, rt = false, cold = false;
                            for (int j = 0; j < m * TOPK; j++) routed |= keys[j] == key;
                            for (int j = 0; j < nadm; j++) rt |= d->e[j] == e;
                            for (int c = 0; c < r->ncold; c++) cold |= r->ce[c] == e;
                            const int why = !routed ? 0 : rt ? 1 : cold ? 2 : 3;  // mispredicted / admitted / skipped / hot
                            st_pf_why[why] += f;
                        }
                        st_pf_rank[std::min(i, 7)] += f;
                        if (used) st_pf_rank_use[std::min(i, 7)] += f;
                    }
                    npf = 0;
                }
                if (g == 0) {  // round 9 audit: lane experts that the previous step's lane also computed at this layer
                    auto& pv = prev_ex[l];
                    int hitp = 0;
                    for (int i = 0; i < nex; i++)
                        for (const char* q : pv) hitp += q == ex[i];
                    st_reuse[std::min(l - 3, 3)][0] += hitp;
                    st_reuse[std::min(l - 3, 3)][1] += nex;
                    pv.assign(ex, ex + nex);
                    if (l == 3 && t_last_lane.time_since_epoch().count()) {
                        const double b = std::chrono::duration<double, std::micro>(tc - t_last_lane).count();
                        if (b < 1e5) st_bound_us += b;  // not across generate calls
                    }
                }
                // round 15 (GLM_ADMIT_AHEAD): while the lane runs, admit the top-N predicted effective cold experts of
                // layer l + 1 (router(l + 1) on this layer's MoE input); the GPUs copy them after this layer's in-step
                // admissions and plan(l + 1) waits for them, so they are hot there and the lane skips them
                auto ahead = [&]() {
                    MoeDec* da = dec_ah + l;
                    int n = 0;
                    if (ng == 1) {
                        MoeReq* rp = req + l;
                        while (*(volatile unsigned*)&rp->pseq != s && !coord_stop) _mm_pause();
                        std::atomic_thread_fence(std::memory_order_acquire);
                        int ce_[MR * TOPK], nc = 0;
                        float cw_[MR * TOPK];
                        for (int i = 0; i < rp->m; i++)
                            for (int k = 0; k < TOPK; k++) {
                                const int e = rp->pids[i][k];
                                const float w = rp->pw[i][k];
                                if (slot_of[(l + 1) * NE + e] >= 0 || (skip_tau > 0 && w < skip_tau * RSF)) continue;
                                int j = 0;
                                while (j < nc && ce_[j] != e) j++;
                                if (j == nc) { ce_[nc] = e; cw_[nc++] = w; }
                                else cw_[j] = std::max(cw_[j], w);
                            }
                        // victims never come from layer l's keys nor from any expert predicted at l + 1 (review r15 M1:
                        // a resident predicted expert evicted here would turn cold at plan(l + 1)); nkeys <= 2 MXE
                        int prot[2 * MR * TOPK + 2 * MXE], np_ = 0;
                        for (int i = 0; i < nkeys; i++) prot[np_++] = keys[i];
                        for (int i = 0; i < rp->m; i++)
                            for (int k = 0; k < TOPK; k++) {
                                const int key = (l + 1) * NE + rp->pids[i][k];
                                int j = nkeys;
                                while (j < np_ && prot[j] != key) j++;
                                if (j == np_) prot[np_++] = key;
                            }
                        for (int a = 0; a < admit_ahead && budget > 0; a++) {
                            int b = -1;
                            for (int j = 0; j < nc; j++)
                                if (cw_[j] >= 0.f && (b < 0 || cw_[j] > cw_[b])) b = j;
                            if (b < 0) break;
                            cw_[b] = -1.f;
                            const int key = (l + 1) * NE + ce_[b];
                            int vkey = -1;
                            const int slot = admit(key, prot, np_, &vkey, ah_force);
                            if (slot < 0) continue;
                            budget--;
                            st_admit++;
                            st_ah_n++;  // key is already in prot (every predicted key is)
                            ahk[nahk++] = key;
                            da->e[n] = ce_[b];
                            da->slot[n] = slot;
                            da->vl[n] = vkey >= 0 ? vkey / NE : -1;
                            da->ve[n] = vkey >= 0 ? vkey % NE : -1;
                            da->nr[n] = 0;
                            n++;
                            if (vkey >= 0) slot_of[vkey] = -1;
                            slot_of[key] = slot;
                            slot_owner[slot] = key;
                        }
                    }
                    da->l = l + 1;
                    da->nadm = 0;
                    da->nbg = n;
                    da->nbgd = 0;
                    da->nzc = 0;
                    std::atomic_thread_fence(std::memory_order_release);
                    *(volatile unsigned*)&da->seq = s;
                };
                const bool ah_now = admit_ahead > 0 && admit_par && g == 0 && l + 1 < nl && !wave;
                void (*tramp)(void*) = [](void* f) { (*static_cast<decltype(ahead)*>(f))(); };
                // round 16 split deferral: exact part (A) first, the deferred part (B) after it into out2 (single sequence)
                const bool spl = def_split && !ar_early && ng == 1 && l + 1 < nl;  // r16 L1 (ar_early also clears lane_defer)
                int na = nex;
                if (spl) {
                    const float wth = defer_w > 0 ? defer_w * RSF : 1e30f;
                    const bool big = nex >= defer_min;
                    const char* tx[MXE]; int tn[MXE], tr[MXE * CpuLane::MR]; float tw[MXE * CpuLane::MR];
                    int nb = 0;
                    na = 0;
                    for (int pass = 0; pass < 2; pass++)
                        for (int i = 0; i < nex; i++) {
                            float mw = 0.f;
                            for (int j = 0; j < enr[i]; j++) mw = std::max(mw, ew[i * CpuLane::MR + j]);
                            const bool defd = big && mw < wth;
                            if (defd != (pass == 1)) continue;
                            const int o = pass ? na + nb : na;
                            tx[o] = ex[i]; tn[o] = enr[i];
                            for (int j = 0; j < enr[i]; j++) { tr[o * CpuLane::MR + j] = erows[i * CpuLane::MR + j]; tw[o * CpuLane::MR + j] = ew[i * CpuLane::MR + j]; }
                            if (pass) { nb++; st_defb_rows += enr[i]; } else na++;
                        }
                    for (int i = 0; i < nex; i++) {
                        ex[i] = tx[i]; enr[i] = tn[i];
                        for (int j = 0; j < tn[i]; j++) { erows[i * CpuLane::MR + j] = tr[i * CpuLane::MR + j]; ew[i * CpuLane::MR + j] = tw[i * CpuLane::MR + j]; }
                    }
                    st_defa += na;
                    st_defb += nb;
                }
                lane->run_rows(&r->x[0][0], m, na, ex, enr, erows, ew, &out[qi].y[0][0], ah_now ? tramp : nullptr, &ahead);
                if (spl) {
                    std::atomic_thread_fence(std::memory_order_release);
                    *(volatile unsigned*)&out[qi].seq = s;
                    lane->run_rows(&r->x[0][0], m, nex - na, ex + na, enr + na, erows + na * CpuLane::MR, ew + na * CpuLane::MR,
                                   &out2[qi].y[0][0]);
                    std::atomic_thread_fence(std::memory_order_release);
                    *(volatile unsigned*)&out2[qi].seq = s;
                }
                if (g == 0 && l == nl - 1) t_last_lane = std::chrono::steady_clock::now();
                if (g == 0) t_lane_end = std::chrono::steady_clock::now();
                st_cpu_us += std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
                std::atomic_thread_fence(std::memory_order_release);
                *(volatile unsigned*)&out[qi].seq = s;
                if (hts && g == 0) hts[l * 8 + 2] = rt_ns();
                // prefetch after the last group of the layer: group 0's prediction of the next layer, until group 0's
                // next request lands
                if (prefetch && g == ng - 1 && l + 1 < nl) {
                    MoeReq* r0 = req + l;  // group 0's request of this layer (pids = its router(l + 1) prediction)
                    while (*(volatile unsigned*)&r0->pseq != s) _mm_pause();
                    std::atomic_thread_fence(std::memory_order_acquire);
                    const char* pl[MXE];
                    int np = 0;
                    const int m0 = r0->m;
                    if (pf_order) {  // round 9: candidates by predicted weight summed over the rows; below-tau picks dropped
                        int ce_[MR * TOPK], nc = 0;
                        float cs[MR * TOPK], cm[MR * TOPK];
                        for (int i = 0; i < m0; i++)
                            for (int k = 0; k < TOPK; k++) {
                                const int e = r0->pids[i][k];
                                if (slot_of[(l + 1) * NE + e] >= 0) continue;
                                const float w = r0->pw[i][k];
                                int j = 0;
                                while (j < nc && ce_[j] != e) j++;
                                if (j == nc) { ce_[nc] = e; cs[nc] = 0.f; cm[nc] = 0.f; nc++; }
                                cs[j] += w;
                                cm[j] = std::max(cm[j], w);
                            }
                        for (int a = 0; a < nc && np < pf_max; a++) {
                            int b = -1;
                            for (int j = 0; j < nc; j++)
                                if (cs[j] >= 0.f && (b < 0 || cs[j] > cs[b])) b = j;
                            if (b < 0) break;
                            const bool keep = cm[b] >= skip_tau * RSF;
                            cs[b] = -1.f;
                            if (keep) pl[np++] = hx_layer[l + 1] + (size_t)ce_[b] * hxl.stride;
                        }
                    } else
                    for (int k = 0; k < TOPK && np < pf_max; k++)
                        for (int i = 0; i < m0 && np < pf_max; i++) {
                            const int e = r0->pids[i][k], key = (l + 1) * NE + e;
                            if (slot_of[key] >= 0) continue;
                            const char* p_ = hx_layer[l + 1] + (size_t)e * hxl.stride;
                            bool dup = false;
                            for (int j = 0; j < np; j++) dup |= pl[j] == p_;
                            if (!dup) pl[np++] = p_;
                        }
                    MoeReq* rn = req + l + 1;
                    const unsigned cur = *(volatile unsigned*)&rn->seq;
                    const unsigned ecur = *(volatile unsigned*)&rn->eseq;
                    if (cur != s && np && (!pf_early || ecur != s)) {  // the next request is not there yet
                        st_pf += np;
                        npf = np;
                        for (int j = 0; j < np; j++) pfl[j] = pl[j];
                        pb0 = lane->pbytes.load();
                        if (lane->tl_on) st_tl[8] += std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t_lane_end).count();
                        if (pf_early) lane->prefetch(np, pl, &rn->eseq, ecur);
                        else lane->prefetch(np, pl, &rn->seq, cur);
                    }
                }
            }
        // round 17 (review r17 L4 / L5): the per-step decay at the end of the step, before it is reported done, so a
        // restore or seed (which waits for done) never races it, nothing runs while the GPU waits at layer 3, and the
        // first step after a seed / restore sees the counts undecayed (= the round-16 order)
        for (auto& x : score) x *= decay;
        s_done = s;
        coord_done.store(s, std::memory_order_release);
    }
}

}  // namespace glm
